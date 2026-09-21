#!/usr/bin/env python
"""MCP server exposing codebase search backed by the local llama-stack.

The model sees ONE tool, `search_code`. It has no idea an embedding model or a
reranker exist. Inside, each call runs:

    query -> embeddings model -> cosine search over the index
          -> reranker (cross-encoder) -> top N snippets

Both models are served by llama-swap on the same port as the chat model, so
this server needs no GPU of its own and no separate model management.

Why a tool and not a prompt-injecting proxy: tool results append at the END of
the conversation, which leaves the cached prompt prefix intact. Injecting
retrieved text into the prefix invalidates the KV cache every turn, and on this
hardware a cold 65k prefill costs ~51s versus ~0 when cached.

Transport is stdio, the standard for local MCP clients.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from dataclasses import dataclass

import numpy as np
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import symbols as sym

STACK = os.environ.get("LLAMA_STACK_URL", "http://127.0.0.1:1234")
INDEX_DB = os.environ.get("CODE_INDEX_DB",
                          os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "..", "index", "code.sqlite3"))
EMBED_MODEL = os.environ.get("EMBED_MODEL", "embeddings")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "reranker")

# Retrieve wide, rerank narrow. The embedding model is cheap and recall-oriented;
# the cross-encoder is expensive and precision-oriented.
CANDIDATES = 40
# Per-document cap when reranking. Keeps query+docs inside the cross-encoder's
# context; the untruncated text is still returned to the caller.
RERANK_DOC_CHARS = 1200
DEFAULT_TOP_K = 5


def _post(path: str, payload: dict, timeout: int = 120) -> dict:
    req = urllib.request.Request(
        f"{STACK}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


# Qwen3-Embedding is asymmetric: QUERIES must carry an instruction prefix,
# documents must not. Embedding a bare query puts it in a different region of
# the space than the corpus, and retrieval silently returns near-random results
# -- which is exactly what we saw: a question about tool-call parsing matched a
# pydantic response model at score 0.002.
QUERY_INSTRUCT = ("Instruct: Given a question about a codebase, retrieve the "
                  "source code that answers it\nQuery: ")


def embed(texts: list[str], is_query: bool = False) -> np.ndarray:
    if is_query:
        texts = [QUERY_INSTRUCT + t for t in texts]
    d = _post("/v1/embeddings", {"model": EMBED_MODEL, "input": texts})
    vecs = np.array([row["embedding"] for row in d["data"]], dtype=np.float32)
    # normalise so a dot product is cosine similarity
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.clip(norms, 1e-9, None)


def rerank(query: str, docs: list[str], top_n: int) -> list[tuple[int, float]]:
    d = _post("/v1/rerank", {"model": RERANK_MODEL, "query": query,
                             "documents": docs, "top_n": top_n})
    return [(r["index"], r["relevance_score"]) for r in d.get("results", [])]


@dataclass
class Chunk:
    path: str
    start: int
    end: int
    text: str


def _db() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(os.path.abspath(INDEX_DB)), exist_ok=True)
    con = sqlite3.connect(INDEX_DB)
    con.execute("""CREATE TABLE IF NOT EXISTS chunks(
        id INTEGER PRIMARY KEY, path TEXT, start INT, end INT,
        text TEXT, vec BLOB)""")
    sym.ensure_schema(con)
    return con


def load_index() -> tuple[list[Chunk], np.ndarray | None]:
    con = _db()
    rows = con.execute("SELECT path,start,end,text,vec FROM chunks").fetchall()
    con.close()
    if not rows:
        return [], None
    chunks = [Chunk(r[0], r[1], r[2], r[3]) for r in rows]
    mat = np.vstack([np.frombuffer(r[4], dtype=np.float32) for r in rows])
    return chunks, mat


def search(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    chunks, mat = load_index()
    if mat is None:
        return []

    qv = embed([query], is_query=True)[0]
    sims = mat @ qv
    n = min(CANDIDATES, len(chunks))
    top = np.argpartition(-sims, n - 1)[:n]
    top = top[np.argsort(-sims[top])]

    cand = [chunks[i] for i in top]
    # Truncate for reranking only. A cross-encoder reads query+document
    # together, so N large chunks can overflow its context and it then scores
    # everything 0.0000 -- silently replacing a good embedding ranking with
    # noise. The full text is still what we return to the caller.
    docs = [f"{c.path}:{c.start}-{c.end}\n{c.text[:RERANK_DOC_CHARS]}" for c in cand]

    embed_order = [(i, float(sims[top[i]])) for i in range(min(top_k, len(cand)))]
    try:
        ranked = rerank(query, docs, min(top_k, len(docs)))
        # Degenerate output (all zeros / empty) means the reranker could not
        # judge these documents. Embedding order is a far better answer than
        # an arbitrary permutation, so prefer it.
        if not ranked or all(s <= 1e-6 for _, s in ranked):
            ranked = embed_order
    except Exception:
        ranked = embed_order

    out = []
    for idx, score in ranked:
        c = cand[idx]
        out.append(dict(path=c.path, start=c.start, end=c.end,
                        score=round(float(score), 4), text=c.text))
    return out


# --------------------------------------------------------------------------
# Minimal MCP stdio server. Implemented directly against the JSON-RPC protocol
# so the stack has no extra pip dependency beyond numpy.
# --------------------------------------------------------------------------

TOOLS = [
    {
        "name": "search_code",
        "description": (
            "Search the indexed codebase for snippets relevant to a natural-language "
            "query. Returns file path, line range and source text, best match first. "
            "Use it to locate definitions, call sites, and usage examples before "
            "editing code."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": "What you are looking for, in plain language."},
                "top_k": {"type": "integer",
                          "description": f"How many snippets to return (default {DEFAULT_TOP_K}).",
                          "default": DEFAULT_TOP_K},
            },
            "required": ["query"],
        },
    },
    {
        "name": "find_definition",
        "description": (
            "Find where a symbol is DEFINED. Exact name match, instant, no embeddings. "
            "Use this whenever you know the identifier -- a function, struct, class, "
            "trait, type or method name -- instead of search_code, which is for "
            "fuzzy questions where you do not know the name."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"symbol": {"type": "string",
                                      "description": "Exact identifier, e.g. parse_hunks or Buffer"}},
            "required": ["symbol"],
        },
    },
    {
        "name": "find_references",
        "description": (
            "Find where a symbol is USED -- call sites and mentions. Use before "
            "changing a signature or deleting code, to see what depends on it. "
            "Set calls_only to list only actual invocations."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "calls_only": {"type": "boolean", "default": False},
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "index_status",
        "description": "Report how many code chunks are indexed and which files are covered.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def handle(req: dict) -> dict | None:
    mid, method = req.get("id"), req.get("method")

    def ok(result):
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    if method == "initialize":
        return ok({
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "llama-stack-code-search", "version": "1.0.0"},
        })

    if method in ("notifications/initialized", "initialized"):
        return None  # notification: no response

    if method == "tools/list":
        return ok({"tools": TOOLS})

    if method == "tools/call":
        p = req.get("params", {})
        name, args = p.get("name"), p.get("arguments", {}) or {}
        try:
            if name == "search_code":
                hits = search(args["query"], int(args.get("top_k", DEFAULT_TOP_K)))
                if not hits:
                    text = ("No results. The index may be empty -- run "
                            "scripts/index_code.py against your repository first.")
                else:
                    text = "\n\n".join(
                        f"### {h['path']}:{h['start']}-{h['end']}  (score {h['score']})\n"
                        f"```\n{h['text']}\n```" for h in hits)
            elif name == "find_definition":
                con = _db()
                rows = sym.find_definition(con, args["symbol"])
                con.close()
                if not rows:
                    text = f"No definition found for '{args['symbol']}'."
                else:
                    text = "\n".join(
                        f"{k:<10} {n}  ->  {p}:{st}-{en}\n    {ln}"
                        for n, k, p, st, en, ln in rows)

            elif name == "find_references":
                con = _db()
                rows = sym.find_references(con, args["symbol"],
                                           bool(args.get("calls_only", False)))
                con.close()
                if not rows:
                    text = f"No references found for '{args['symbol']}'."
                else:
                    calls = [r for r in rows if r[1] == "call"]
                    other = [r for r in rows if r[1] != "call"]
                    parts = []
                    if calls:
                        parts.append("CALL SITES:\n" + "\n".join(
                            f"  {p}:{ln}  {tx}" for _, _, p, ln, tx in calls))
                    if other:
                        parts.append("OTHER REFERENCES:\n" + "\n".join(
                            f"  {p}:{ln}  {tx}" for _, _, p, ln, tx in other[:30]))
                    text = "\n\n".join(parts)

            elif name == "index_status":
                chunks, mat = load_index()
                files = sorted({c.path for c in chunks})
                text = (f"{len(chunks)} chunks across {len(files)} files.\n"
                        + "\n".join(f"  {f}" for f in files[:40]))
                if len(files) > 40:
                    text += f"\n  ... and {len(files) - 40} more"
            else:
                return {"jsonrpc": "2.0", "id": mid,
                        "error": {"code": -32601, "message": f"unknown tool {name}"}}
            return ok({"content": [{"type": "text", "text": text}]})
        except Exception as e:
            return ok({"content": [{"type": "text",
                                    "text": f"search failed: {type(e).__name__}: {e}"}],
                       "isError": True})

    return {"jsonrpc": "2.0", "id": mid,
            "error": {"code": -32601, "message": f"unknown method {method}"}}


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        resp = handle(req)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
