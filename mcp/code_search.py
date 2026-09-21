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


# Laya decision engine. Separate process AND separate venv: laya needs a newer
# transformers than the rest of the stack, so it cannot share an interpreter.
LAYA_URL = os.environ.get("LAYA_URL", "http://127.0.0.1:1237")


def _post_json(url: str, payload: dict, timeout: int = 120) -> dict:
    """POST to an absolute URL, for services that are not behind llama-swap."""
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


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
        # Only ORDER matters from a reranker; the absolute scale does not.
        #
        # An earlier version rejected any result where every score was below
        # 1e-6, on the assumption scores are probabilities. They are not always:
        # this setup returns correctly-ordered scores around 1e-13, and that
        # guard was silently discarding a good ranking in favour of raw
        # embedding order -- which is exactly the degradation it existed to
        # prevent.
        #
        # The real degenerate case is a reranker that cannot separate the
        # documents at all, i.e. every score identical. Detect that instead.
        scores = [s for _, s in ranked]
        if not ranked or (len(set(scores)) == 1):
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
        "name": "judge",
        "description": (
            "Get a typed decision with a calibrated probability in ~25ms, from a small "
            "classifier (Laya) rather than by reasoning. It does NOT explain and cannot "
            "write -- it only decides. Use it to gate work you would otherwise think "
            "through: is this diff an improvement, is this failure infrastructure or a "
            "real regression, is this snippet relevant, is this worth benchmarking. "
            "Omit options for a true/false judgement."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "state": {"type": "string",
                          "description": "The text being judged: diff, log, test output, snippet."},
                "question": {"type": "string",
                             "description": "What to decide, phrased as an instruction."},
                "options": {"type": "array", "items": {"type": "string"},
                            "description": "Choices for a multiple-choice decision. "
                                           "Omit for true/false."},
            },
            "required": ["state", "question"],
        },
    },
    {
        "name": "grep",
        "description": (
            "Literal / regex search across the indexed files. FAST and EXACT. "
            "Try this BEFORE search_code whenever the thing you want probably "
            "appears verbatim somewhere -- a function name, an error string, a "
            "config key, an import, a TODO. Most code questions have a literal "
            "anchor and this finds it precisely; search_code is for when the "
            "codebase uses different words than you would."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string",
                            "description": "Regex. Case-insensitive by default."},
                "glob": {"type": "string",
                         "description": "Restrict to paths containing this substring, e.g. '.rs' or 'src/'."},
                "max_results": {"type": "integer", "description": "Default 40."},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "read_file",
        "description": (
            "Read exact lines from an indexed file. Use this immediately after "
            "find_definition or search_code, which return a path and line range -- "
            "do NOT search again for something you already have the location of. "
            "Reads the real file on disk, not the index, so it reflects edits."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string",
                         "description": "Path exactly as returned by the other tools."},
                "start": {"type": "integer", "description": "First line (1-indexed). Default 1."},
                "end": {"type": "integer", "description": "Last line. Default start+120."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "compact",
        "description": (
            "Compress long text -- a conversation, a log, a large file -- down to the "
            "parts that matter, keeping identifiers, numbers, file paths and errors "
            "verbatim. Use when context is filling up or before feeding a long "
            "artefact into further reasoning. Returns prose, not the original."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "focus": {"type": "string",
                          "description": "What to preserve detail about. Optional but "
                                         "makes the result far more useful."},
                "max_words": {"type": "integer", "description": "Target length. Default 300."},
            },
            "required": ["text"],
        },
    },
    {
        "name": "index_status",
        "description": "Report how many code chunks are indexed and which files are covered.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "verify",
        "description": (
            "Run a project check and return its output. THIS IS THE POINT OF THE LOOP: "
            "you are not done when the edit looks right, you are done when a check "
            "passes. Run `lint` after any edit (seconds), and `test` when behaviour "
            "changed (minutes). Call with no arguments to list what this project "
            "offers. If a check fails, read the error, fix it with apply_edit, and "
            "run it again."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "check": {"type": "string",
                          "description": "Name of the check, e.g. 'lint' or 'test'. "
                                         "Omit to list available checks."},
            },
        },
    },
]

# Only these may be run, and only as named here. The agent chooses a LABEL, never
# a command line, so text arriving from a source file, a search result or a model
# cannot reach a shell. Adding a project means adding a row, not widening this.
VERIFY_CHECKS = {
    "lint": ["npm", "run", "lint-core"],
    "test": ["npm", "run", "test-unit"],
    "build": ["npm", "run", "build"],
}
VERIFY_TIMEOUT = int(os.environ.get("VERIFY_TIMEOUT", "900"))


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

            elif name == "judge":
                opts = args.get("options") or []
                if opts:
                    # empty criterion strings are legal: the option label is the
                    # description when no elaboration is given
                    q = {"v": {"type": "choice", "instructions": args["question"],
                               "criteria": {o: "" for o in opts}}}
                else:
                    q = {"v": {"type": "noul", "instructions": args["question"]}}
                try:
                    d = _post_json(LAYA_URL + "/decide",
                                   {"state": args["state"], "questions": q})
                    a = d["answers"]["v"]
                    if "choice" in a:
                        probs = sorted(a.get("probabilities", {}).items(), key=lambda x: -x[1])
                        text = (f"{a['choice']}   (confidence {a.get('confidence')})\n"
                                + "\n".join(f"  {k}: {v:.3f}" for k, v in probs))
                    else:
                        text = (f"probability true: {a.get('noul'):.3f}   "
                                f"(confidence {a.get('confidence')})")
                    text += f"\n[{d.get('elapsed_ms')} ms]"
                except Exception as e:
                    text = (f"judge unavailable ({type(e).__name__}: {e}). "
                            "Is the laya service running on 1237?")

            elif name == "grep":
                import re as _re
                pat = args["pattern"]
                globsub = (args.get("glob") or "").replace("\\", "/")
                limit = int(args.get("max_results", 40))
                try:
                    rx = _re.compile(pat, _re.IGNORECASE)
                except _re.error as e:
                    text = f"bad regex: {e}"
                else:
                    con = _db()
                    roots = [r[0] for r in con.execute("SELECT path FROM roots").fetchall()]
                    rows = con.execute("SELECT DISTINCT path FROM chunks").fetchall()
                    con.close()
                    hits, scanned = [], 0
                    # Models pass real glob patterns ("*.js", "src/**/*.rs"),
                    # not substrings. Substring matching silently returned
                    # zero files for "*.js" and the agent could not tell the
                    # difference between "no matches" and "bad filter".
                    import fnmatch as _fn
                    is_glob = any(c in globsub for c in "*?[")

                    def _match(rel: str) -> bool:
                        if not globsub:
                            return True
                        if is_glob:
                            return (_fn.fnmatch(rel, globsub)
                                    or _fn.fnmatch(os.path.basename(rel), globsub)
                                    or _fn.fnmatch(rel, f"*{globsub.lstrip('*/')}"))
                        return globsub in rel

                    for (rel,) in sorted(rows):
                        if not _match(rel):
                            continue
                        for root in roots:
                            p = os.path.abspath(os.path.join(root, rel))
                            if not p.startswith(os.path.abspath(root)) or not os.path.isfile(p):
                                continue
                            scanned += 1
                            try:
                                with open(p, encoding="utf-8", errors="replace") as fh:
                                    for n, line in enumerate(fh, 1):
                                        if rx.search(line):
                                            hits.append(f"{rel}:{n}: {line.rstrip()[:180]}")
                                            if len(hits) >= limit:
                                                break
                            except OSError:
                                pass
                            break
                        if len(hits) >= limit:
                            break
                    # "in 0 files" is a dead end: the agent cannot tell a bad
                    # glob from an unindexed directory, so it permutes the glob
                    # forever (observed: 14 consecutive greps for docs/, which
                    # is not indexed at all). Say what IS indexed instead.
                    if hits:
                        text = "\n".join(hits)
                    elif scanned == 0 and globsub:
                        tops = sorted({(r.split("/", 1)[0] if "/" in r else ".")
                                       for (r,) in rows})
                        text = (f"glob {globsub!r} matched 0 of {len(rows)} indexed files.\n"
                                f"Indexed roots: {', '.join(roots) or '(none)'}\n"
                                f"Top-level dirs: {', '.join(tops[:20])}\n"
                                f"Nothing outside these roots is indexed. "
                                f"Omit `glob` to search all {len(rows)} files.")
                    else:
                        text = f"no matches for /{pat}/ in {scanned} files"
                    if len(hits) >= limit:
                        text += f"\n[truncated at {limit}]"

            elif name == "read_file":
                rel = args["path"].replace("\\", "/")
                start = max(1, int(args.get("start", 1)))
                end = int(args.get("end", start + 120))
                con = _db()
                roots = [r[0] for r in con.execute("SELECT path FROM roots").fetchall()]
                con.close()
                # Resolve against the roots that were actually indexed, and
                # refuse anything that escapes them -- the path arrives from a
                # model and must not be able to read arbitrary files.
                target = None
                for root in roots:
                    cand = os.path.abspath(os.path.join(root, rel))
                    if cand.startswith(os.path.abspath(root)) and os.path.isfile(cand):
                        target = cand
                        break
                if target is None:
                    # A bare "not found" leaves the agent guessing again. Offer
                    # indexed files with the same basename -- the usual cause is
                    # a plausible but wrong directory, e.g. guessing
                    # renderers/webgl/WebGLRenderer.js for renderers/WebGLRenderer.js
                    base = os.path.basename(rel).lower()
                    con = _db()
                    near = [r[0] for r in con.execute("SELECT DISTINCT path FROM chunks").fetchall()
                            if os.path.basename(r[0]).lower() == base][:8]
                    con.close()
                    text = f"'{rel}' not found under any indexed root."
                    if near:
                        text += "\nDid you mean:\n" + "\n".join(f"  {p}" for p in near)
                    elif not roots:
                        text += "\nNothing is indexed — run scripts/index_code.py first."
                else:
                    with open(target, encoding="utf-8", errors="replace") as fh:
                        lines = fh.read().splitlines()
                    end = min(end, len(lines))
                    body = "\n".join(f"{i:>5}  {lines[i-1]}"
                                     for i in range(start, end + 1))
                    text = f"{rel}:{start}-{end}  ({len(lines)} lines total)\n```\n{body}\n```"

            elif name == "compact":
                focus = args.get("focus") or "anything an engineer would need to continue the work"
                maxw = int(args.get("max_words", 300))
                prompt = (
                    f"Compress the following into at most {maxw} words.\n"
                    f"Preserve verbatim: identifiers, file paths, line numbers, error "
                    f"strings and measurements. Drop pleasantries and repetition.\n"
                    f"Keep detail about: {focus}\n"
                    f"Return only the compressed text.\n\n---\n{args['text']}"
                )
                try:
                    d = _post("/v1/chat/completions", {
                        "model": "bonsai-agent",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max(256, maxw * 3),
                    }, timeout=900)
                    text = (d["choices"][0]["message"].get("content") or "").strip()
                except Exception as e:
                    text = f"compaction failed: {type(e).__name__}: {e}"

            elif name == "index_status":
                chunks, mat = load_index()
                files = sorted({c.path for c in chunks})
                text = (f"{len(chunks)} chunks across {len(files)} files.\n"
                        + "\n".join(f"  {f}" for f in files[:40]))
                if len(files) > 40:
                    text += f"\n  ... and {len(files) - 40} more"

            elif name == "verify":
                import subprocess
                check = (args.get("check") or "").strip().lower()
                con = _db()
                roots = [r[0] for r in con.execute("SELECT path FROM roots").fetchall()]
                con.close()
                # Checks run from the project root, which is the parent of the
                # indexed src/ tree -- package.json lives there, not in src/.
                cwd = os.path.dirname(roots[0].rstrip("\\/")) if roots else os.getcwd()
                if not check:
                    text = ("checks available (pass one as `check`):\n"
                            + "\n".join(f"  {k:<8} {' '.join(v)}"
                                        for k, v in VERIFY_CHECKS.items())
                            + f"\nproject root: {cwd}")
                elif check not in VERIFY_CHECKS:
                    text = (f"unknown check {check!r}. Available: "
                            f"{', '.join(VERIFY_CHECKS)}.")
                else:
                    cmd = VERIFY_CHECKS[check]
                    try:
                        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                                           timeout=VERIFY_TIMEOUT, shell=(os.name == "nt"),
                                           encoding="utf-8", errors="replace")
                        out = ((p.stdout or "") + (p.stderr or "")).strip()
                        # Failures put the useful lines at the END; truncating the
                        # head of a passing run costs nothing, truncating the tail
                        # of a failing one throws away the error.
                        if len(out) > 6000:
                            out = out[:1500] + "\n...[trimmed]...\n" + out[-4000:]
                        text = (f"{check}: {'PASSED' if p.returncode == 0 else 'FAILED'} "
                                f"(exit {p.returncode})\n\n{out or '(no output)'}")
                    except subprocess.TimeoutExpired:
                        text = f"{check}: TIMED OUT after {VERIFY_TIMEOUT}s."
                    except FileNotFoundError:
                        text = f"{check}: cannot run {cmd[0]!r} -- not on PATH."
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
