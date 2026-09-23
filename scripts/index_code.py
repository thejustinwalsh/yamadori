#!/usr/bin/env python
"""Build the code index the MCP server searches.

    python scripts/index_code.py C:/path/to/repo [more/paths ...]

Chunking is structure-aware, not character-count based. Splitting mid-function
degrades retrieval quality more than any other single choice here: an embedding
of half a function matches poorly against a question about what that function
does. We split on top-level definition boundaries and only fall back to line
windows for files we cannot parse.

Embeddings come from the same llama-swap endpoint the agents use, so there is
no second copy of the model and nothing extra to keep running.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.request

import numpy as np

import symbols as sym
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'mcp'))
import guard

# The MODEL SERVER, not the proxy. This pointed at the proxy's port, which
# does not serve /v1/embeddings, so every embedding request 404'd and every
# chunk fell through to the zero-vector fallback. That is how both package
# indexes came to be written entirely dead.
STACK = os.environ.get("LLAMA_STACK_URL", "http://127.0.0.1:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "embeddings")
HERE = os.path.dirname(os.path.abspath(__file__))
INDEX_DB = os.environ.get("CODE_INDEX_DB", os.path.join(HERE, "..", "index", "code.sqlite3"))

# Extensions worth indexing. Weighted toward systems + graphics work:
# Rust/C++/Zig/WASM, TypeScript, and shader languages.
EXTS = {
    # systems
    ".rs", ".c", ".h", ".cpp", ".hpp", ".cc", ".hh", ".cxx", ".zig",
    # wasm
    ".wat", ".wast",
    # web / ts
    ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs",
    # shaders / gpu -- WebGPU, GL, Metal
    ".wgsl", ".glsl", ".vert", ".frag", ".comp", ".hlsl", ".metal", ".msl",
    # mobile / native bridges
    ".swift", ".kt", ".m", ".mm", ".java",
    # scripting + config
    ".py", ".go", ".cs", ".sh", ".ps1", ".sql",
    ".yaml", ".yml", ".toml", ".json", ".md", ".txt",
}

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist",
             "build", "target", ".next", ".cache", "installer_files", "site-packages"}

MAX_CHUNK_LINES = 120     # keep chunks inside the embedder's 8192-token window
MIN_CHUNK_CHARS = 40      # skip trivial fragments
BATCH = 32

# Top-level definition starts, per language family. Used only as a fallback
# when tree-sitter cannot parse a file (see ast_chunker.py).
DEF_RE = re.compile(
    r"^(?:\s{0,3})(?:"
    r"def\s+\w+|class\s+\w+|async\s+def\s+\w+|"                       # python
    r"(?:export\s+)?(?:async\s+)?function\s+\w+|"                      # js/ts
    r"(?:export\s+)?(?:const|let|var)\s+\w+\s*=\s*(?:async\s*)?[(<]|"  # js arrow fns
    r"(?:export\s+)?(?:default\s+)?(?:interface|type|enum|class|namespace)\s+\w+|"  # ts
    r"func\s+(?:\([^)]*\)\s*)?\w+|"                                    # go
    r"(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:unsafe\s+)?(?:extern\s+\"[^\"]*\"\s+)?fn\s+\w+|"
    r"(?:pub(?:\([^)]*\))?\s+)?(?:struct|enum|impl|trait|mod|union)\s+|"  # rust
    r"(?:pub\s+)?(?:export\s+)?(?:inline\s+)?fn\s+\w+|"                # zig fn
    r"(?:pub\s+)?const\s+\w+\s*=\s*(?:packed\s+|extern\s+)?(?:struct|enum|union|opaque)\b|"  # zig types
    r"(?:template\s*<[^>]*>\s*)?(?:class|struct|namespace|union)\s+\w+|"  # c++
    r"[A-Za-z_][\w:<>,\s\*&]*::\w+\s*\(|"                              # c++ out-of-line defs
    r"@(?:compute|vertex|fragment)\b|"                                  # wgsl entry points
    r"(?:fn|struct|var<[^>]*>|@group)\s*\w*|"                           # wgsl decls
    r"(?:__kernel|__global__)\s+|"                                      # gpu kernels
    r"(?:public|private|protected|static|final|open|override|\s)*\s*"
    r"(?:class|interface|enum|object|protocol|extension)\s+\w+|"        # java/kt/swift
    r"#{1,6}\s+"                                                        # markdown headings
    r")"
)


def _listed_files(root: str, listing: str):
    """The files named in INDEX_FILES that sit under `root`, in list order.

    A PACKAGE is not walked the way a repository is. Its real code is often
    only in dist/ or build/, which SKIP_DIRS prunes -- correctly, for a
    user's repository, where those are their own build output. The package
    selection (deps.code_files) decides what to read and hands the result
    over as a list; the root still names the package directory, so stored
    paths keep their `dist/...` prefix and read_file_range can resolve them.
    Same credential and size checks as the walk: a list is not a bypass.
    """
    base = os.path.abspath(root)
    with open(listing, encoding="utf-8") as f:
        names = [ln.strip() for ln in f if ln.strip()]
    for p in names:
        ap = os.path.abspath(p)
        if not ap.lower().startswith(base.lower() + os.sep):
            continue
        if os.path.splitext(ap)[1].lower() not in EXTS or not os.path.isfile(ap):
            continue
        ok, why = guard.should_index(ap)
        if not ok:
            print(f"  skipped {os.path.basename(ap)}: {why}", file=sys.stderr)
            continue
        if os.path.getsize(ap) > 1_500_000:
            continue
        yield ap


def iter_files(root: str):
    # An explicit file list replaces the walk. Unset for every repository
    # index, so their behaviour is unchanged; set only by deps.index_package.
    listing = os.environ.get("INDEX_FILES")
    if listing:
        yield from _listed_files(root, listing)
        return
    # A single file is a legitimate "root": incremental refresh names the files
    # that changed rather than re-walking a tree of thousands.
    if os.path.isfile(root):
        if os.path.splitext(root)[1].lower() in EXTS:
            yield root
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in EXTS:
                p = os.path.join(dirpath, fn)
                # Credentials live in the same formats as config. Indexing
                # them copies them into a database and then feeds them to a
                # model; measured before this check, secrets.json and
                # credentials.toml were both ingested outright.
                ok, why = guard.should_index(p)
                if not ok:
                    print(f"  skipped {fn}: {why}", file=sys.stderr)
                    continue
                try:
                    if os.path.getsize(p) > 1_500_000:
                        continue
                except OSError:
                    continue
                yield p


def chunk_file(path: str) -> list[tuple[int, int, str]]:
    # Prefer real syntax trees; fall back to the regex heuristic only for
    # files tree-sitter has no grammar for (or cannot parse).
    try:
        from ast_chunker import chunk_with_ast
        ast_chunks = chunk_with_ast(path)
        if ast_chunks:
            return ast_chunks
    except ImportError:
        pass

    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
    except OSError:
        return []
    if not lines:
        return []

    # Boundaries at top-level definitions; always include line 0.
    bounds = [0] + [i for i, ln in enumerate(lines) if i and DEF_RE.match(ln)]
    bounds = sorted(set(bounds))

    spans: list[tuple[int, int]] = []
    for i, start in enumerate(bounds):
        end = bounds[i + 1] if i + 1 < len(bounds) else len(lines)
        # split oversized spans into windows rather than truncating
        while end - start > MAX_CHUNK_LINES:
            spans.append((start, start + MAX_CHUNK_LINES))
            start += MAX_CHUNK_LINES
        if end > start:
            spans.append((start, end))

    out = []
    for s, e in spans:
        text = "\n".join(lines[s:e]).strip()
        if len(text) >= MIN_CHUNK_CHARS:
            out.append((s + 1, e, text))   # 1-indexed line numbers for humans
    return out


# The embedding server runs at -c 8192; anything longer is rejected with a
# bare HTTP 400 that aborts the whole indexing run. Generated headers and
# large type-declaration files exceed it routinely, so cap here rather than
# discovering it 2000 files in. The full text is still what gets stored and
# returned -- only the vector is computed from the head of the chunk.
EMBED_MAX_CHARS = 12000
EMBED_DIM = 1024          # Qwen3-Embedding-0.6B


# SUSPECT. This says BM25 beat semantic search on the gauntlet index, and the
# cut rule fired. That comparison was made against indexes now known to hold
# nothing but zero vectors, where "semantic search" ranked by an all-zero
# similarity array -- so it was BM25 against arbitrary order, and BM25 winning
# tells us nothing. Do not cite it. Re-run against a rebuilt index before
# either restoring or confirming this default.
NO_EMBED = os.environ.get("INDEX_NO_EMBED") == "1"

# A long index run makes thousands of requests and the connection does drop.
# It dropped once on three.js at chunk ~1760 of 15021, the run stopped, and
# the truncated index was left on disk looking perfectly healthy: every vector
# it did contain was fine. Losing 88% of a corpus is not a condition an index
# can report about itself, so the transport has to survive it instead.
EMBED_RETRIES = int(os.environ.get("INDEX_EMBED_RETRIES", "5"))


def _retry(texts: list[str], attempt: int, err: Exception) -> np.ndarray:
    delay = 2 ** attempt
    print(f"  embedding request failed ({type(err).__name__}: {err}), retry "
          f"{attempt + 1}/{EMBED_RETRIES} in {delay}s", file=sys.stderr)
    time.sleep(delay)
    return embed_batch(texts, attempt + 1)


def embed_batch(texts: list[str], attempt: int = 0) -> np.ndarray:
    if NO_EMBED:
        return np.zeros((len(texts), EMBED_DIM), dtype=np.float32)
    texts = [t[:EMBED_MAX_CHARS] for t in texts]
    req = urllib.request.Request(
        f"{STACK}/v1/embeddings",
        data=json.dumps({"model": EMBED_MODEL, "input": texts}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            d = json.load(r)
    # ORDER MATTERS. HTTPError subclasses URLError, so listing the transport
    # clause first swallowed every HTTP error into the retry path and made the
    # per-chunk isolation below unreachable. That regression was introduced
    # while adding retries and cost a whole three.js run.
    except urllib.error.HTTPError as e:
        # 5xx is the server having a bad moment -- the same request usually
        # succeeds. 4xx is this request being unacceptable, and retrying an
        # oversized chunk five times just takes five times as long to fail.
        if e.code >= 500 and attempt < EMBED_RETRIES:
            return _retry(texts, attempt, e)
        if len(texts) == 1:
            raise
        # One oversized or malformed chunk must not cost the whole run. Fall
        # back to one-at-a-time so the bad chunk is isolated and skipped.
        rows = []
        for t in texts:
            try:
                rows.append(embed_batch([t])[0])
            except Exception:                              # noqa: BLE001
                print(f"  skipped a chunk ({len(t)} chars): {e}", file=sys.stderr)
                rows.append(np.zeros(EMBED_DIM, dtype=np.float32))
                _DEAD["n"] += 1
        return np.vstack(rows)
    except (ConnectionResetError, urllib.error.URLError, TimeoutError,
            OSError) as e:
        # A dropped connection is transient and the same request will usually
        # succeed. Backing off matters because the usual cause is the server
        # being busy.
        if attempt >= EMBED_RETRIES:
            raise
        return _retry(texts, attempt, e)
    v = np.array([row["embedding"] for row in d["data"]], dtype=np.float32)
    return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-9, None)


# A zero vector is not a bad embedding, it is the ABSENCE of one: it has
# cosine similarity 0 with every query, so those chunks sort arbitrarily and
# semantic search silently returns noise. The per-chunk fallback above exists
# so one oversized chunk cannot kill a long run, but it is only safe while it
# stays rare.
#
# It did not stay rare. Both package indexes were written with the embedding
# server unreachable, every single chunk fell through to the fallback, and
# 18,750 zero vectors were committed under a progress bar reading "indexed".
# Search over three.js and typegpu returned whatever `argpartition` happened
# to surface from an all-zero similarity array, for days, with no error
# anywhere -- and retrieval quality, including a verdict on the reranker, was
# judged on that.
#
# So the run now checks itself and refuses to leave a dead index on disk.
_DEAD = {"n": 0}
MAX_DEAD_FRACTION = float(os.environ.get("INDEX_MAX_DEAD", "0.02"))


def mark_complete(con: sqlite3.Connection, total: int, n_files: int) -> None:
    """Record that the run reached the end.

    A crash leaves a truncated index that looks perfectly healthy, because
    every row it did write is correct -- three.js stopped at chunk 1,760 of
    15,021 and nothing downstream could tell. Completeness is not a property
    a partial index can report about itself, so it is written once, last, and
    anything that reads the index checks for it.
    """
    con.execute("CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT)")
    con.executemany(
        "INSERT INTO meta(k, v) VALUES(?, ?) "
        "ON CONFLICT(k) DO UPDATE SET v = excluded.v",
        [("complete", "1"), ("chunks", str(total)), ("files", str(n_files)),
         ("dead", str(_DEAD["n"])), ("built", str(int(time.time())))])
    con.commit()


def is_complete(db: str) -> bool:
    try:
        con = sqlite3.connect(db)
        try:
            row = con.execute(
                "SELECT v FROM meta WHERE k='complete'").fetchone()
        finally:
            con.close()
        return bool(row) and row[0] == "1"
    except sqlite3.Error:
        return False      # no meta table: built before this existed, or partial


def check_alive(con: sqlite3.Connection, total: int) -> None:
    """Refuse to finish a run that produced an index nothing can search."""
    if not total:
        return
    dead = _DEAD["n"]
    if dead / total <= MAX_DEAD_FRACTION:
        if dead:
            print(f"  {dead} of {total} chunks have no embedding "
                  f"({dead / total:.2%}), within tolerance")
        return
    con.rollback()
    raise SystemExit(
        "\nREFUSING TO WRITE A DEAD INDEX\n"
        f"  {dead} of {total} chunks ({dead / total:.1%}) got no embedding.\n"
        "  An index of zero vectors answers every query with noise and\n"
        "  reports no error, so this run is being discarded.\n"
        f"  Check the embedding server at {STACK} is serving {EMBED_MODEL!r}.")


def main() -> None:
    roots = sys.argv[1:]
    if not roots:
        print(__doc__)
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(INDEX_DB)), exist_ok=True)
    con = sqlite3.connect(INDEX_DB)
    con.execute("""CREATE TABLE IF NOT EXISTS chunks(
        id INTEGER PRIMARY KEY, path TEXT, start INT, end INT, text TEXT, vec BLOB)""")
    sym.ensure_schema(con)
    # Record which directories were indexed so read_file can resolve the
    # relative paths that find_definition/search_code hand back.
    con.execute("CREATE TABLE IF NOT EXISTS roots(path TEXT PRIMARY KEY)")
    # INDEX_APPEND is how repos.refresh() re-indexes just the files that
    # changed. It has already deleted the rows for those paths, so wiping here
    # would throw away the entire index to add three files back.
    append = os.environ.get("INDEX_APPEND") == "1"
    if not append:
        con.execute("DELETE FROM chunks")
        con.execute("DELETE FROM defs")
        con.execute("DELETE FROM refs")
        con.execute("DELETE FROM roots")
        for r in roots:
            con.execute("INSERT OR IGNORE INTO roots VALUES(?)", (os.path.abspath(r),))
    con.commit()

    pending: list[tuple[str, int, int, str]] = []
    total = 0

    def flush():
        nonlocal total
        if not pending:
            return
        vecs = embed_batch([t for _, _, _, t in pending])
        con.executemany(
            "INSERT INTO chunks(path,start,end,text,vec) VALUES(?,?,?,?,?)",
            [(p, s, e, t, vecs[i].tobytes()) for i, (p, s, e, t) in enumerate(pending)],
        )
        con.commit()
        total += len(pending)
        pending.clear()
        print(f"  indexed {total} chunks", end="\r", flush=True)

    # With INDEX_APPEND the argv entries are files, so relative paths have to
    # be computed against the indexed root recorded in the database.
    stored_roots = [r[0] for r in con.execute("SELECT path FROM roots").fetchall()]

    def rel_to_root(path: str, fallback: str) -> str:
        ap = os.path.abspath(path)
        for sr in stored_roots:
            if ap.lower().startswith(os.path.abspath(sr).lower() + os.sep):
                return os.path.relpath(ap, sr).replace("\\", "/")
        return os.path.relpath(ap, fallback).replace("\\", "/")

    for root in roots:
        print(f"scanning {root}")
        base = root if os.path.isdir(root) else os.path.dirname(root)
        for path in iter_files(root):
            rel = rel_to_root(path, base)

            # Symbol/call-site index: exact-match navigation, no GPU needed.
            # Built alongside the vector index so one pass over the tree does both.
            d_rows, r_rows = sym.extract(path, rel)
            if d_rows:
                con.executemany("INSERT INTO defs VALUES(?,?,?,?,?,?)", d_rows)
            if r_rows:
                con.executemany("INSERT INTO refs VALUES(?,?,?,?,?)", r_rows)

            for s, e, text in chunk_file(path):
                pending.append((rel.replace("\\", "/"), s, e, text))
                if len(pending) >= BATCH:
                    flush()
    flush()

    check_alive(con, total)
    n_files = con.execute("SELECT COUNT(DISTINCT path) FROM chunks").fetchone()[0]
    mark_complete(con, total, n_files)
    con.close()
    print(f"\ndone: {total} chunks from {n_files} files -> {os.path.abspath(INDEX_DB)}")


if __name__ == "__main__":
    main()
