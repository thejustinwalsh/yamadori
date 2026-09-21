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
import urllib.request

import numpy as np

import symbols as sym

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
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in EXTS:
                p = os.path.join(dirpath, fn)
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


def embed_batch(texts: list[str]) -> np.ndarray:
    req = urllib.request.Request(
        f"{STACK}/v1/embeddings",
        data=json.dumps({"model": EMBED_MODEL, "input": texts}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.load(r)
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
    con.execute("DELETE FROM chunks")          # full rebuild; incremental is future work
    con.execute("DELETE FROM defs")
    con.execute("DELETE FROM refs")
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

    for root in roots:
        print(f"scanning {root}")
        for path in iter_files(root):
            rel = os.path.relpath(path, root).replace("\\", "/")

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
