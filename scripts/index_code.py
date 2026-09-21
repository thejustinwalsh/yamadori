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
import urllib.error
import urllib.request

import numpy as np

import symbols as sym
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'mcp'))
import guard

STACK = os.environ.get("LLAMA_STACK_URL", "http://127.0.0.1:1234")
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


def iter_files(root: str):
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


def embed_batch(texts: list[str]) -> np.ndarray:
    texts = [t[:EMBED_MAX_CHARS] for t in texts]
    req = urllib.request.Request(
        f"{STACK}/v1/embeddings",
        data=json.dumps({"model": EMBED_MODEL, "input": texts}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            d = json.load(r)
    except urllib.error.HTTPError as e:
        # One oversized or malformed chunk must not cost the whole run. Fall
        # back to one-at-a-time so the bad chunk is isolated and skipped.
        if len(texts) == 1:
            raise
        rows = []
        for t in texts:
            try:
                rows.append(embed_batch([t])[0])
            except Exception:                              # noqa: BLE001
                print(f"  skipped a chunk ({len(t)} chars): {e}", file=sys.stderr)
                rows.append(np.zeros(EMBED_DIM, dtype=np.float32))
        return np.vstack(rows)
    v = np.array([row["embedding"] for row in d["data"]], dtype=np.float32)
    return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-9, None)


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

    n_files = con.execute("SELECT COUNT(DISTINCT path) FROM chunks").fetchone()[0]
    con.close()
    print(f"\ndone: {total} chunks from {n_files} files -> {os.path.abspath(INDEX_DB)}")


if __name__ == "__main__":
    main()
