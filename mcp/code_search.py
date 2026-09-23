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

import difflib
import json
import os
import re
import sqlite3
import sys
from dataclasses import dataclass

import numpy as np
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import symbols as sym
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fusion
import rings

# The MODEL SERVER, not the proxy. These are different ports and confusing
# them fails in the worst possible way: /v1/embeddings 404s, the caller
# catches it, and search returns an empty list that is indistinguishable from
# "nothing matched". A benchmark run scored 0.0% across every condition
# before anyone noticed the index was never consulted.
STACK = os.environ.get("LLAMA_STACK_URL", "http://127.0.0.1:11434")
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

# Reranking is skipped above this k. This is a LATENCY decision, not an
# accuracy one -- be careful not to restate it as the latter.
#
# scripts/eval_rerank.py, 11 queries with a known correct file:
#
#   correct at #1     embed 1/11    rerank 3/11    Fisher exact p ~= 0.6: NOT
#                                                  a result. Do not cite it.
#   correct in top-5  embed 7/11    rerank 7/11    identical, for ~1s/query
#   mean rank         5.36          5.36
#
# What the numbers do support: the reranker permutes within the top 5 and does
# not change which files reach the caller at the shipped cutoff. Since all five
# snippets land in context and get read, paying a second to reorder them buys
# nothing measurable. Skipping it takes search_code from ~1100ms to 96ms.
#
# THAT LATENCY FIGURE, QUALIFIED WHERE IT IS QUOTED. One search_code call,
# warm (index loaded, both models resident), embeddings ON, over the three.js
# index -- the same corpus scripts/eval_rerank.py uses, where the cross-encoder
# scores the full CANDIDATES=40 pool. The ONLY thing varying between 1100 and
# 96 is whether that cross-encoder round trip happens.
#
# It is a DIFFERENT measurement from the 1284ms -> 729ms quoted in
# docs/BUILD.md and docs/PLAN.md. That one is a search_fused call on the
# gauntlet index where the varying thing is CODE_SEARCH_SEMANTIC=1 -> 0, i.e.
# the EMBEDDING round trip. Different component, different corpus, different
# entry point. The two savings do not compose and neither number may be quoted
# in place of the other.
#
# Two caveats that keep this provisional: the corpus was three.js, which is in
# every training set, and 11 queries is far too few to detect anything but a
# huge effect. Re-run against an uncontaminated index before drawing a
# conclusion about accuracy in either direction.
RERANK_MAX_K = 2


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


def _inside(root: str, path: str) -> bool:
    """True only when `path` is `root` or below it, after resolving links.

    A plain prefix test is not containment: `C:/pkg` is a prefix of
    `C:/pkg-secrets/x`, and a junction or symlink inside a root can point
    anywhere. Resolve both, fold case the way Windows does, and compare
    whole path components.
    """
    try:
        r = os.path.normcase(os.path.realpath(root))
        q = os.path.normcase(os.path.realpath(path))
        return os.path.commonpath([r, q]) == r
    except ValueError:                   # different drives, or an empty path
        return False


def _db() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(os.path.abspath(INDEX_DB)), exist_ok=True)
    con = sqlite3.connect(INDEX_DB)
    con.execute("""CREATE TABLE IF NOT EXISTS chunks(
        id INTEGER PRIMARY KEY, path TEXT, start INT, end INT,
        text TEXT, vec BLOB)""")
    # `roots` IS PART OF THE SCHEMA, and until now only scripts/index_code.py
    # created it. Three tools read it -- find_by_pattern, read_file_range and
    # run_check all do `SELECT path FROM roots` -- so against any database this
    # function made rather than the indexer (a half-built index, a dependency
    # index, or the proxy's own `_no_repository_bound.sqlite3`, which IS
    # created here) every one of them raised
    #
    #     OperationalError: no such table: roots
    #
    # which is the exact failure the tool audit was opened for. It had been
    # fixed for `db is None` by gating run_check behind a bound repository, and
    # the other door was left open: bind a repository whose index lacks the
    # table and the same exception comes back, now wearing a TOOL_RAISED
    # envelope that tells the agent an operator must fix it.
    #
    # Creating it empty is the correct repair rather than catching the error at
    # each call site: all three tools already handle zero roots as "nothing is
    # indexed", which is the truth about such a database.
    con.execute("CREATE TABLE IF NOT EXISTS roots(path TEXT PRIMARY KEY)")
    sym.ensure_schema(con)
    return con


# The resident index, keyed by which database it came from and that file's
# state. The BM25 side below has been cached since it was written; the vectors
# never were, so every query re-read 62 MB of BLOBs out of sqlite and rebuilt
# the matrix. Measured on three.js: 116 ms per query to load, against 1.4 ms to
# actually multiply it. The search was eighty times cheaper than fetching the
# thing being searched.
#
# Keyed on (path, mtime, size) so re-indexing invalidates it without anyone
# having to remember to. Only one index is held: the alternative is unbounded
# memory across every dependency a caller mentions, and a swap costs 119 ms.
_INDEX_CACHE: tuple = ()
_INDEX_DATA: tuple = ([], None)


def _index_signature() -> tuple:
    try:
        st = os.stat(INDEX_DB)
        return (os.path.abspath(INDEX_DB), st.st_mtime_ns, st.st_size)
    except OSError:
        return (os.path.abspath(INDEX_DB), 0, 0)


def has_index() -> bool:
    """Is there an index at all, as opposed to one that did not match?

    These are different facts and the tools must report them differently. An
    empty result is retryable; an absent index is terminal, and an agent that
    cannot tell them apart will keep rephrasing a query against a database
    that does not exist.
    """
    try:
        chunks, mat = load_index()
        return mat is not None and len(chunks) > 0
    except Exception:                                            # noqa: BLE001
        return False


def index_state() -> dict:
    """What is in the index, as facts. No advice, no adjectives."""
    try:
        chunks, mat = load_index()
        return {"exists": mat is not None and len(chunks) > 0,
                "chunks": len(chunks),
                "files": len({c.path for c in chunks}),
                "database": os.path.abspath(INDEX_DB),
                "readable": True}
    except Exception as e:                                       # noqa: BLE001
        return {"exists": False, "chunks": 0, "files": 0,
                "database": os.path.abspath(INDEX_DB),
                "readable": False, "read_error": f"{type(e).__name__}: {e}"}


def index_summary() -> str:
    st = index_state()
    return f"{st['chunks']:,} chunks across {st['files']:,} files"


# ---------------------------------------------------------------------------
# THE TOOL RESULT ENVELOPE
#
# Every result a tool returns is a fact about what happened, structured, with
# no advice in it. That is a deliberate reversal.
#
# What it replaced, verbatim: "No results. Nothing is indexed for this
# repository yet, or the query matched nothing. Try find_by_pattern with a
# literal you expect, or describe_index to see what is covered."
#
# Three things wrong with that sentence. It merges two different facts behind
# an "or", so the agent cannot tell which happened. It never says the call
# failed. And it ends by recommending another tool that fails identically.
#
# MEASURED: a model issued the same search twelve times over 842 seconds and
# never answered the question. It was not malfunctioning -- it was told its
# query might have missed and that another tool might work, so it kept trying.
# The loop was written by us.
#
# `retryable` is the field that matters, and it is a FACT rather than a
# suggestion: false means no arguments to this tool can ever succeed in this
# conversation. What to do about that is the agent's decision, not ours, and
# telling it ("answer from your own knowledge") is how a tool result starts
# steering the answer instead of informing it.
#
# JSON because the tool protocol gives us one string and nothing else --
# OpenAI's tool message is {role, tool_call_id, content}; there is no is_error
# flag as there is in Anthropic's API. Structure has to live inside content.
# It is also the shape a decision model can consume without parsing prose.
# ---------------------------------------------------------------------------
def ok_result(tool: str, **facts) -> str:
    return json.dumps({"tool": tool, "ok": True, **facts}, default=str)


def error_result(tool: str, code: str, reason: str, retryable: bool,
                 **facts) -> str:
    return json.dumps({"tool": tool, "ok": False, "error": code,
                       "reason": reason, "retryable": retryable, **facts},
                      default=str)


def no_index_error(tool: str) -> str:
    """No corpus is bound to this conversation.

    Carries THREE things, and the third is the one that was missing:

      what happened   the call did not run, and why
      the data        what the index actually contains right now
      the remedy      what would change it, and WHO can do it

    The remedy is a fact about this server, not a guess: the catalogue is read
    off disk, and binding a package the server holds genuinely makes search
    work through packages.search_discovered. That is different in kind from
    the sentence this replaced -- "Try find_by_pattern with a literal you
    expect" -- which was a guess about a tool that fails identically, and
    which a model followed twelve times over 842 seconds.

    `fixable_by` matters because most remedies are not the agent's to apply.
    An agent that knows the operator must act can say so instead of retrying.
    """
    # The remedies NAME the tool and stop. They do not list what this server
    # holds: we do not know which source the caller wants, the inventory is
    # ours rather than theirs, and a list invites the agent to pick from it
    # instead of asking about the code in front of it. bind_project_context
    # carries its own description; that is where "what it needs" belongs.
    remedies = [
        {"fixable_by": "agent",
         "applies_when": "the question involves a library the project imports",
         "action": "call bind_project_context with the versions from the manifest",
         "effect": ("if this server holds that library's source, search is "
                    "then served from it")},
        {"fixable_by": "user",
         "applies_when": "the question is about the user's own code",
         "action": "the user pastes the relevant code into the conversation",
         "effect": "the code is then in context and needs no index",
         "why_not_the_agent": ("this server cannot see the user's filesystem; "
                               "no tool call can reach it")},
    ]
    return error_result(
        tool, "NO_INDEX",
        ("No code index is bound to this conversation, so the search did not "
         "run. This is not an empty result set."),
        retryable=False,
        index=index_state(),
        affects=["find_by_meaning", "find_by_pattern", "find_definition_opt",
                 "find_references", "read_file_range"],
        note=("Every tool in `affects` returns this error regardless of "
              "arguments, until a remedy below is applied."),
        remedies=remedies)


def retrievers_failed_error(tool: str, diag: dict) -> str:
    """The index exists, but nothing was able to read it.

    Previously invisible: both retriever handlers were `except: pass`, so a
    total outage produced an empty result and the tool said the query missed.
    An agent cannot fix a broken retriever and should not try -- it can only
    report it, which it can only do if it is told.
    """
    return error_result(
        tool, "RETRIEVERS_UNAVAILABLE",
        ("The index exists but no retriever completed, so nothing was "
         "searched. A result of zero matches would be misleading here."),
        retryable=False,
        index=index_state(),
        retrievers=diag.get("retrievers", {}),
        remedies=[{
            "fixable_by": "operator",
            "action": ("check the errors in `retrievers`, then restart the "
                       "stack with scripts/start-stack.bat"),
            "effect": "the retrievers load and search runs again",
            "why_not_the_agent": ("a failing service is outside this "
                                  "conversation; report it rather than retry"),
        }])


def bad_arguments_error(tool: str, problem: str, fix: str) -> str:
    """The CALLER made a mistake, and only the caller can undo it.

    WHAT THIS REPLACED, AND WHY IT MATTERED

    Every argument mistake fell through to the generic handler at the bottom of
    `handle()` and came back as, verbatim:

        {"tool": "find_by_pattern", "ok": false, "error": "TOOL_RAISED",
         "reason": "KeyError: 'pattern'", "retryable": false,
         "remedies": [{"fixable_by": "operator", ...}]}

    Three of those four fields are wrong, and they are wrong in the direction
    that stops work:

      reason       a raw Python exception, which contract 2 exists to forbid.
                   Other shapes seen: "ValueError: invalid literal for int()
                   with base 10: 'lots'", "AttributeError: 'int' object has no
                   attribute 'replace'", and "ProgrammingError: Error binding
                   parameter 1: type 'list' is not supported", which leaks the
                   storage layer into a conversation.
      retryable    FALSE. Contract 4 says false means no arguments can ever
                   succeed. A model that forgot `pattern` was told that adding
                   it could not help.
      fixable_by   OPERATOR. The one party who cannot see the call.
      error code   TOOL_RAISED, indistinguishable from the server actually
                   breaking.

    Worse, a non-string `query` never reached this path at all: it raised
    inside BOTH retrievers' `except` blocks in search_fused, so the tool
    reported RETRIEVERS_UNAVAILABLE -- an outage that had not happened -- for
    a mistake the agent had made and could have corrected in one call.

    So: a distinct code, retryable TRUE, and the agent named as the fixer,
    because it is the only party holding the thing that is wrong.
    """
    return error_result(
        tool, "BAD_ARGUMENTS", problem, retryable=True,
        remedies=[{"fixable_by": "agent",
                   "applies_when": "always -- the arguments came from you",
                   "action": fix,
                   "effect": "the call then runs normally"}])


# tool -> (required strings, optional strings, integer arguments)
#
# Declared as data rather than checked inline so that a tool added to TOOLS
# without a row here is visible, and so the coverage test can enumerate it.
_ARGSPEC: dict[str, tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = {
    "find_by_meaning":     (("query",),   (),                      ("top_k",)),
    "find_definition_opt": (("symbol",),  (),                      ()),
    "find_references":     (("symbol",),  (),                      ()),
    "find_by_pattern":     (("pattern",), ("glob",),               ("max_results",)),
    "read_file_range":     (("path",),    (),                      ("start", "end")),
    "summarize_text":      (("text",),    ("focus",),              ("max_words",)),
    "describe_index":      ((),           (),                      ()),
    "run_check":           ((),           ("check",),              ()),
    "judge":               (("state", "question"), (),             ()),
    "record_step":         ((), ("kind", "summary", "detail", "outcome"), ()),
    "read_rings":          ((),           ("kind",),               ("limit",)),
}

# Booleans, checked separately because `bool(x)` is not a parser.
#
# `find_references` did `bool(args.get("calls_only", False))`, and models send
# JSON-ish strings: bool("yes"), bool("no"), bool("false") and bool("0") are
# ALL True. A caller asking for calls_only=false got calls_only=true and a
# narrowed result set, with nothing anywhere saying the flag had been
# inverted. There is no safe coercion here -- "false" cannot be told from a
# typo -- so the only honest answer is to say the argument is not a boolean.
_BOOL_ARGS: dict[str, tuple[str, ...]] = {
    "find_references": ("calls_only",),
}


def validate_args(name: str, args: dict) -> str | None:
    """None if the arguments are usable, otherwise the error to return.

    Only three things are checked, and each one corresponds to a measured
    crash rather than a style preference:

      a required argument is absent      -> KeyError
      an argument is the wrong type      -> AttributeError / TypeError /
                                            sqlite3.ProgrammingError
      a count is zero or negative        -> a WRONG ANSWER, not a crash:
          find_by_pattern with max_results=-1 hit `len(hits) >= limit` on the
          first file and returned {"matches": 0, "files_scanned": 1} for a
          pattern that matches. A false negative is the worst of the three,
          because nothing about it looks like a failure.

    Enormous values are deliberately NOT rejected: top_k=1e9, max_results=1e9
    and limit=1e9 were all measured to answer correctly and quickly, because
    they are used as slice and LIMIT bounds. Refusing them would be inventing
    a failure.
    """
    spec = _ARGSPEC.get(name)
    if spec is None:
        return None
    required, optional, integers = spec

    for key in required:
        if key not in args or args[key] is None:
            return bad_arguments_error(
                name, f"Required argument {key!r} was not supplied, so the "
                      f"tool did not run.",
                f"call {name} again with {key!r} set")
    for key in required + optional:
        if key in args and args[key] is not None and not isinstance(args[key], str):
            return bad_arguments_error(
                name, f"Argument {key!r} must be a string; got "
                      f"{type(args[key]).__name__}.",
                f"call {name} again with {key!r} as a string")
    for key in required:
        if not str(args[key]).strip():
            return bad_arguments_error(
                name, f"Argument {key!r} is empty. An empty {key} does not "
                      f"describe anything to look for.",
                f"call {name} again with a non-empty {key!r}")
    for key in integers:
        if key not in args or args[key] is None:
            continue
        v = args[key]
        if isinstance(v, bool) or not isinstance(v, (int, str)):
            return bad_arguments_error(
                name, f"Argument {key!r} must be a whole number; got "
                      f"{type(v).__name__}.",
                f"call {name} again with {key!r} as an integer")
        try:
            n = int(v)
        except (TypeError, ValueError):
            return bad_arguments_error(
                name, f"Argument {key!r} must be a whole number; got {v!r}.",
                f"call {name} again with {key!r} as an integer")
        # `start` may be clamped up to 1 without losing information -- line 0
        # and line -5 both unambiguously mean "the beginning". Every other
        # count changes the ANSWER when it goes below one.
        if n < 1 and key != "start":
            return bad_arguments_error(
                name, f"Argument {key!r} is {n}. A value below 1 asks for no "
                      f"results at all, which would be reported as though "
                      f"nothing matched.",
                f"call {name} again with {key!r} of 1 or more")
    for key in _BOOL_ARGS.get(name, ()):
        if key in args and args[key] is not None and not isinstance(args[key], bool):
            return bad_arguments_error(
                name, f"Argument {key!r} must be true or false; got "
                      f"{args[key]!r}, which would have been read as "
                      f"{bool(args[key])} whatever it says.",
                f"call {name} again with {key!r} as a JSON boolean")
    return None


def load_index() -> tuple[list[Chunk], np.ndarray | None]:
    global _INDEX_CACHE, _INDEX_DATA
    sig = _index_signature()
    if sig == _INDEX_CACHE:
        return _INDEX_DATA

    con = _db()
    rows = con.execute("SELECT path,start,end,text,vec FROM chunks").fetchall()
    con.close()
    if not rows:
        return [], None
    chunks = [Chunk(r[0], r[1], r[2], r[3]) for r in rows]
    # np.vstack on 15,021 separate buffers allocates 15,021 times. One
    # frombuffer over the concatenated bytes allocates once.
    dim = len(rows[0][4]) // 4
    mat = np.frombuffer(b"".join(r[4] for r in rows),
                        dtype=np.float32).reshape(len(rows), dim)
    _INDEX_CACHE, _INDEX_DATA = sig, (chunks, mat)
    return chunks, mat


_LEX: "fusion.Lexical | None" = None
_LEX_SIG: tuple = ()


def _lexical(chunks: list[Chunk]) -> "fusion.Lexical":
    """Build the BM25 side once per index, not once per query."""
    global _LEX, _LEX_SIG
    sig = (len(chunks), chunks[0].path if chunks else "", INDEX_DB)
    if _LEX is None or _LEX_SIG != sig:
        _LEX = fusion.Lexical([(c.path, c.text) for c in chunks])
        _LEX_SIG = sig
    return _LEX


def search_fused(query: str, top_k: int = DEFAULT_TOP_K,
                 diag: dict | None = None) -> list[dict]:
    """Three independent retrievers, fused, with agreement as the confidence.

    Their failure modes differ -- embeddings miss on vocabulary mismatch,
    lexical misses when the query words are everywhere, symbol lookup misses
    anything that is not a declaration -- so a file all three surface is
    supported by three different kinds of evidence. That is a far better
    confidence signal than a cosine score which, measured on this corpus,
    separates genuine from nonsense queries by only 0.03.
    """
    chunks, mat = load_index()
    if mat is None:
        return []

    wide = max(top_k * 3, 10)
    rankings: dict[str, list[str]] = {}

    # 1. semantic -- OFF BY DEFAULT. Measured on the gauntlet index (60k chunks
    # across four private repos, 120 gold queries), embeddings lost to BM25
    # over the same chunks: recall@5 77/120 vs 92/120, McNemar p=0.0041. Fusing
    # them did not rescue it: 87/120, and fused vs keyword was 3 wins to 8,
    # p=0.2266 -- indistinguishable. The pre-registered rule was "cut unless
    # fusion beats BM25 significantly", so it is cut.
    #
    # Caveat kept deliberately: the gold queries are derived from symbol names,
    # which flatters lexical matching, and the hand-written conceptual set that
    # would favour embeddings has not been run yet. Set CODE_SEARCH_SEMANTIC=1
    # to re-enable and re-measure.
    # WHICH RETRIEVERS ACTUALLY RAN.
    #
    # Both handlers below are `except: pass`, so a retriever that throws just
    # disappears. With all of them gone the fusion returns nothing, and the
    # tool reported "the index is populated, your query missed" -- a total
    # outage wearing the clothes of a bad query. The module docstring already
    # warns about the embeddings version of this ("a benchmark run scored 0.0%
    # across every condition before anyone noticed the index was never
    # consulted"), and the same shape was still here one layer down.
    #
    # So the failures are recorded rather than swallowed, and the caller
    # states them. Being off BY CONFIGURATION is recorded too: an agent told
    # "no match" deserves to know the semantic retriever was never consulted.
    if diag is None:
        diag = {}
    diag.setdefault("retrievers", {})
    sim_by_path: dict[str, float] = {}
    text_by_path: dict[str, tuple] = {}
    semantic_on = os.environ.get("CODE_SEARCH_SEMANTIC", "0") == "1"
    diag["retrievers"]["semantic"] = (
        {"ran": False, "reason": "disabled by configuration",
         "config": "CODE_SEARCH_SEMANTIC=0",
         "why": "measured to lose to lexical on this corpus; set to 1 to re-enable"}
        if not semantic_on else {"ran": True})
    if semantic_on:
        sem = search(query, wide)
        rankings["semantic"] = [h["path"] for h in sem]
        sim_by_path = {h["path"]: h["sim"] for h in sem}
        text_by_path = {h["path"]: (h["start"], h["end"], h["text"]) for h in sem}

    # 2. lexical
    try:
        lex = _lexical(chunks)
        idxs = lex.search(query, wide)
        seen, lex_paths = set(), []
        for i in idxs:
            p = chunks[i].path
            if p not in seen:
                seen.add(p)
                lex_paths.append(p)
                text_by_path.setdefault(p, (chunks[i].start, chunks[i].end, chunks[i].text))
        rankings["lexical"] = lex_paths
        diag["retrievers"]["lexical"] = {"ran": True, "results": len(lex_paths)}
    except Exception as e:                                       # noqa: BLE001
        diag["retrievers"]["lexical"] = {
            "ran": False, "reason": "failed",
            "error": f"{type(e).__name__}: {e}"}

    # 3. symbol table -- NOT a ranker. It answers a yes/no question ("is there
    # a declaration named this here"), so it is collected as a tag and never
    # enters the fusion. Its rows arrive in table order, and letting that order
    # act as a rank made both the RRF contribution and the confidence tier
    # depend on SQLite's insertion order.
    symbol_hits: set[str] = set()
    symbol_name: dict[str, str] = {}
    try:
        con = _db()
        words = set(fusion.content_words(query))
        if words:
            for name, path, start, end in con.execute(
                    "SELECT name, path, start, end FROM defs").fetchall():
                parts = set(fusion.content_words(name))
                if parts and parts <= words:
                    symbol_hits.add(path)
                    symbol_name.setdefault(path, name)
                    text_by_path.setdefault(path, (start, end, ""))
        con.close()
        diag["retrievers"]["symbols"] = {"ran": True, "hits": len(symbol_hits)}
    except Exception as e:                                       # noqa: BLE001
        diag["retrievers"]["symbols"] = {
            "ran": False, "reason": "failed",
            "error": f"{type(e).__name__}: {e}"}

    ran = [k for k, v in diag["retrievers"].items() if v.get("ran")]
    failed = [k for k, v in diag["retrievers"].items()
              if not v.get("ran") and v.get("reason") == "failed"]
    diag["retrievers_ran"] = ran
    diag["retrievers_failed"] = failed
    # No retriever ran at all: the search did not happen. That is a different
    # situation from a search that happened and matched nothing, and the two
    # must never be reported with the same words.
    diag["searched"] = bool(ran)

    rows = fusion.fuse(rankings)
    out = []
    for r in rows[:max(top_k * 2, 8)]:
        start, end, text = text_by_path.get(r["path"], (0, 0, ""))
        is_sym = r["path"] in symbol_hits
        out.append({**r, "tier": fusion.tier(r, symbol_hit=is_sym),
                    "declares": symbol_name.get(r["path"]),
                    "sim": sim_by_path.get(r["path"]),
                    "start": start, "end": end, "text": text})
    # A declaration match is the strongest evidence available and may sit
    # outside both rankers' top-N, so surface it even when they missed it.
    known = {r["path"] for r in out}
    for p in sorted(symbol_hits - known)[:2]:
        start, end, text = text_by_path.get(p, (0, 0, ""))
        out.insert(0, {"path": p, "score": 0.0, "found_by": [], "best_rank": 0,
                       "tier": "exact", "declares": symbol_name.get(p),
                       "sim": None, "start": start, "end": end, "text": text})
    return out


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

    if top_k > RERANK_MAX_K:
        # Measured: no gain at this cutoff. Skip the GPU round trip.
        ranked = embed_order
    else:
        try:
            ranked = rerank(query, docs, min(top_k, len(docs)))
            # Only ORDER matters from a reranker; the absolute scale does not.
            #
            # An earlier version rejected any result where every score was below
            # 1e-6, on the assumption scores are probabilities. They are not
            # always: this setup returns correctly-ordered scores around 1e-13,
            # and that guard was silently discarding a good ranking in favour of
            # raw embedding order -- exactly the degradation it existed to
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
        # Two different numbers, and only one of them means anything absolute.
        # `score` is the cross-encoder's, useful for ORDER only -- it comes back
        # around 1e-13 on correctly-ranked results. `sim` is cosine similarity
        # against the query embedding, which IS calibrated, so it is the only
        # one a threshold may be applied to.
        out.append(dict(path=c.path, start=c.start, end=c.end,
                        score=round(float(score), 4),
                        sim=round(float(sims[top[idx]]), 4), text=c.text))
    return out


# --------------------------------------------------------------------------
# Minimal MCP stdio server. Implemented directly against the JSON-RPC protocol
# so the stack has no extra pip dependency beyond numpy.
# --------------------------------------------------------------------------

# `judge` is implemented below and reachable over the Laya HTTP/WS service, but
# it is deliberately NOT in TOOLS: it is not calibrated enough to put in front
# of a model. scripts/eval_judge.py, on unambiguous yes/no engineering
# questions, scored 6/10 against a 5/10 coin flip, and the errors were
# systematic rather than noisy -- three of four misses were false positives on
# the NEGATIVE cases, all landing at 0.67-0.69:
#
#   printf(ptr) then free(ptr)   "uses memory after free"   want F, got 0.69
#   f(): number { return 42 }    "return type is wrong"     want F, got 0.68
#   borrow-legal Rust            "violates borrow rules"    want F, got 0.67
#
# It is scoring what the passage is ABOUT, not whether the proposition holds --
# the same failure that made it rate the nonsense query "quantum teapot
# recursion" above every genuine one. A judge a model trusts and that is wrong
# turns an open question into a confident wrong answer, which is worse than
# having no judge. Re-expose it only when eval_judge.py passes.
TOOLS_DISABLED_JUDGE = """
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
"""

TOOLS = [
    {
        "name": "find_by_meaning",
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
        "name": "find_definition_opt",
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
            "Answer 'what depends on this?'. Returns every place a symbol is "
            "USED -- call sites, imports, mentions -- which is the opposite of "
            "find_definition_opt, which returns the ONE place it is declared.\n"
            "Reach for this when the request says: what uses X, what calls X, "
            "what breaks if I rename or delete X, is X dead code, which files "
            "import X, what depends on X.\n"
            "A request about renaming, removing, or the blast radius of a "
            "change is this tool, not a definition lookup and not a text "
            "search: it also reports a symbol that IS declared but has no "
            "users, which is the answer to 'is this dead code'.\n"
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
        "name": "find_by_pattern",
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
        "name": "read_file_range",
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
        "name": "summarize_text",
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
        "name": "describe_index",
        "description": "Report how many code chunks are indexed and which files are covered.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "run_check",
        "description": (
            "Run a project check and return its output. THIS IS THE POINT OF THE LOOP: "
            "you are not done when the edit looks right, you are done when a check "
            "passes. Run `lint` after any edit (seconds), and `test` when behaviour "
            "changed (minutes). Call with no arguments to list what this project "
            "offers. If a check fails, read the error, fix the code with your "
            "own editing tools, and run it again."
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

# ---------------------------------------------------------------------------
# TWO SURFACES, AND THE LINE BETWEEN THEM.
#
# TOOLS is the MCP surface: what an outside client may call.
# INTERNAL_TOOLS is what only this stack's own reasoning invokes, via the proxy.
#
# The work log used to be appended here -- `TOOLS = TOOLS + rings.TOOLS` -- so
# it sat on both. That breaks the moment a tool needs something only one caller
# can supply: `record_step` is scoped to a conversation, and the conversation
# key is the PROXY's knowledge. Reached over MCP there is no key, so it either
# errors for a legitimate caller or writes into a shared log -- which is the
# cross-session disclosure this already produced, where a caller with no
# repository was handed another session's notes.
#
# THE RULE: a tool offered on both surfaces must take everything it needs from
# its arguments and its configuration, and assume nothing about who called it.
# Search qualifies -- it searches whichever index it is pointed at, and the
# caller decides which. The work log does not, so it is internal only.
#
# Nothing is removed from what the MODEL can call: the proxy injects both
# lists. What changes is that an outside MCP client is no longer offered a
# tool that cannot work correctly for it.
# ---------------------------------------------------------------------------
INTERNAL_TOOLS = list(rings.TOOLS)
ALL_TOOLS = TOOLS + INTERNAL_TOOLS


# Cosine below this is clear junk. Deliberately conservative: measured, the
# worst genuine query scored 0.476 and the best nonsense 0.445, so a cutoff
# placed between them would be fitted to a handful of samples and would
# sometimes discard good results. This catches only the obvious case.
#
# Laya was tried for the ambiguous band -- "does this snippet answer this
# query" -- across four phrasings, and separated nothing: the nonsense query
# "quantum teapot recursion" scored 0.83-0.85, HIGHER than every genuine
# query, on all four. It is a good classifier for decisions with named
# options; snippet relevance is not one of them.
WEAK_SIM = 0.40


def _no_symbol(symbol: str, kind: str) -> str:
    """Explain a symbol miss well enough that the next call is obvious.

    A bare "not found" is indistinguishable from a typo, a casing difference,
    a symbol that lives outside the indexed roots, and an empty index. The
    caller cannot pick a next step without knowing which, so say which.
    """
    con = _db()
    try:
        names = [r[0] for r in con.execute("SELECT DISTINCT name FROM defs").fetchall()]
    except sqlite3.Error:
        names = []
    con.close()
    return _near_miss(symbol, names, kind)


# Paths that are ABOUT the library rather than part of it. A reference here is
# real, and it is still the wrong thing to show first: a demo shows one way to
# use a symbol, while the library source shows what it is.
_PERIPHERAL = re.compile(
    r"(?:^|/)(?:examples?|demos?|benches?|benchmarks?|docs?|website|"
    r"__tests__|tests?|spec|e2e|fixtures?|dist|build|vendor|node_modules)/"
    r"|\.(?:test|spec|bench)\.[jt]sx?$|\.min\.[jt]s$", re.I)


def _rank_refs(rows: list) -> list:
    """Library source first, then the files that use a symbol most.

    Rows arrive in sqlite insertion order, which is directory-walk order, and
    the caller truncates to 30. Ordering therefore decides what the model
    sees, and walk order is arbitrary: `positionLocal` in three.js led with
    examples/jsm/generators and buried NodeMaterial.js, which holds six
    references and is the implementation.

    Density is the second key because a file referencing a symbol six times is
    more likely to be where it lives than one mentioning it once in an import.
    """
    if not rows:
        return rows
    per_file: dict[str, int] = {}
    for r in rows:
        per_file[r[2]] = per_file.get(r[2], 0) + 1
    return sorted(
        rows,
        key=lambda r: (bool(_PERIPHERAL.search(r[2])),   # library source first
                       -per_file[r[2]],                  # then densest use
                       r[2],                             # stable within a file
                       r[3] if isinstance(r[3], int) else 0))


def _near_miss(symbol: str, names: list[str], kind: str) -> str:
    if not names:
        # Terminal, not a miss -- see no_index_error(). "Run
        # scripts/index_code.py" is advice the AGENT cannot act on: it
        # runs on the server, and the agent is on someone else's machine.
        return no_index_error(f"the {kind} lookup")
    near = difflib.get_close_matches(symbol, names, n=6, cutoff=0.6)
    if not near:
        low = symbol.lower()
        near = [n for n in names if low in n.lower()][:6]
    out = f"No {kind} for {symbol!r} among {len(names)} indexed symbols."
    if near:
        out += "\nClosest indexed names:\n" + "\n".join(f"  {n}" for n in near)
    else:
        out += ("\nNo similar name is indexed either, so this symbol is probably "
                "defined outside the indexed roots, or comes from a dependency. "
                "grep for it as a literal to find usages.")
    return out


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
        # THE SURFACE SPLIT, ENFORCED RATHER THAN DESCRIBED.
        #
        # TOOLS is the MCP surface, INTERNAL_TOOLS is ours, and the split only
        # ever existed in the lists handed out. `judge` is advertised in
        # NEITHER, and `tools/call name:"judge"` dispatched anyway: the
        # boundary was a comment, and a caller who knew the name walked past
        # it. A name nobody is offered is not callable.
        #
        # Reported as an unknown tool, the same as any other, because
        # confirming that a hidden name exists is itself information.
        if name not in {t["name"] for t in ALL_TOOLS}:
            return {"jsonrpc": "2.0", "id": mid,
                    "error": {"code": -32601, "message": f"unknown tool {name}"}}
        # BEFORE the dispatch, not inside it. Every tool below indexes its
        # arguments directly -- args["query"], args["pattern"].replace(...),
        # int(args["top_k"]) -- and each of those is a different exception
        # arriving in the caller's conversation with the operator named as the
        # only person who can fix it. One gate, one answer, and the answer says
        # who actually holds the mistake.
        bad = validate_args(name, args)
        if bad is not None:
            return ok({"content": [{"type": "text", "text": bad}],
                       "isError": True})
        try:
            if name == "find_by_meaning":
                # search_fused, not search. Plain search() is the pure-semantic
                # arm the cut rule eliminated -- BM25 beat it 92/120 to 77/120
                # on the gauntlet index -- and this tool was still calling it.
                diag: dict = {}
                hits = search_fused(args["query"],
                                    int(args.get("top_k", DEFAULT_TOP_K)),
                                    diag)
                if not hits:
                    # TWO DIFFERENT SITUATIONS, AND THEY MUST NOT SOUND ALIKE.
                    #
                    # This used to answer both with "Nothing is indexed for
                    # this repository yet, OR the query matched nothing. Try
                    # find_by_pattern..." -- which tells the agent its query
                    # may have missed, and invites a retry.
                    #
                    # MEASURED: against an empty index a model issued the
                    # identical search twelve times over 842 seconds and never
                    # answered. It was not malfunctioning. We told it to try
                    # again, so it did. The loop was our wording.
                    # Three situations, three answers. They were one.
                    if not has_index():
                        text = no_index_error("find_by_meaning")
                    elif not diag.get("searched"):
                        text = retrievers_failed_error("find_by_meaning", diag)
                    else:
                        # A PARTIAL OUTAGE IS NOT A CLEAN MISS.
                        #
                        # This branch is reached whenever at least one
                        # retriever ran, and it said "Every available
                        # retriever ran" regardless -- while shipping, in the
                        # same object, `lexical: {"ran": false, "reason":
                        # "failed"}`. The prose contradicted the data beside
                        # it, and the prose is what a model reads. An agent
                        # was told to trust a negative that came from half the
                        # index.
                        failed = diag.get("retrievers_failed") or []
                        reason = (
                            "Every available retriever ran and no chunk "
                            "scored above threshold for this query."
                            if not failed else
                            ("The retrievers that ran found nothing, but "
                             + ", ".join(failed) + " failed, so part of the "
                             "index was never searched. This is a weaker "
                             "negative than a clean miss: do not conclude the "
                             "code is absent, and report the failure."))
                        text = ok_result(
                            "find_by_meaning", matches=0,
                            query=args["query"], index=index_state(),
                            retryable=True,
                            retrievers=diag.get("retrievers", {}),
                            retrievers_failed=failed,
                            complete=not failed,
                            reason=reason)
                else:
                    # Results are GROUPED BY EVIDENCE, not returned as a flat
                    # ranked list. A tier states a condition the caller can act
                    # on; a score invites treating small differences as
                    # meaningful when they are not. Naming the evidence is also
                    # what lets the caller disagree with the ranking, which a
                    # bare list does not.
                    tiers = {"exact": [], "close": [], "alternative": []}
                    for h in hits:
                        tiers.setdefault(h.get("tier", "alternative"), []).append(h)
                    label = {
                        "exact": ("TAPROOT -- a declaration with this name is "
                                  "here. Strongest evidence available."),
                        "close": ("BRANCH -- both independent retrievers "
                                  "surfaced this. Two different failure modes "
                                  "had to agree."),
                        "alternative": ("SHOOT -- one retriever only. "
                                        "Unproven; read before relying on it."),
                    }
                    parts = []
                    for tier in ("exact", "close", "alternative"):
                        rows = tiers.get(tier) or []
                        if not rows:
                            continue
                        parts.append(f"== {label[tier]}")
                        for h in rows[:5]:
                            why = ", ".join(h.get("found_by") or []) or "symbol table"
                            decl = (f"  declares {h['declares']}"
                                    if h.get("declares") else "")
                            parts.append(
                                f"### {h['path']}:{h['start']}-{h['end']}"
                                f"   [{why}]{decl}")
                            if h.get("text"):
                                parts.append("```\n" + h["text"][:1400] + "\n```")
                    text = "\n".join(parts)

            elif name == "find_definition_opt":
                con = _db()
                rows = sym.find_definition(con, args["symbol"])
                con.close()
                if not rows:
                    text = _no_symbol(args["symbol"], "definition")
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
                    # "No references" means two completely different things and
                    # the caller has to know which: a symbol that IS defined and
                    # has no callers is dead code -- a real answer -- while a
                    # symbol that is not indexed is a failed lookup.
                    con = _db()
                    defined = sym.find_definition(con, args["symbol"])
                    con.close()
                    if defined:
                        where = ", ".join(f"{p}:{st}" for _, _, p, st, _, _ in defined[:3])
                        text = (f"'{args['symbol']}' is defined ({where}) but nothing "
                                f"references it in the index. Either it is dead code, "
                                f"or its callers live outside the indexed roots.")
                    else:
                        text = _no_symbol(args["symbol"], "references")
                else:
                    # Truncation without ordering is the bug. `other[:30]` kept
                    # whatever sqlite returned first, which is insertion order,
                    # which is directory-walk order. Asking three.js for
                    # `positionLocal` led with examples/jsm/generators and
                    # buried src/materials/nodes/NodeMaterial.js, which holds
                    # six references and is the library. A model reads the
                    # first few lines and answers from a demo.
                    calls = _rank_refs([r for r in rows if r[1] == "call"])
                    other = _rank_refs([r for r in rows if r[1] != "call"])
                    parts = []
                    if calls:
                        parts.append("CALL SITES:\n" + "\n".join(
                            f"  {p}:{ln}  {tx}" for _, _, p, ln, tx in calls[:30]))
                    if other:
                        parts.append("OTHER REFERENCES:\n" + "\n".join(
                            f"  {p}:{ln}  {tx}" for _, _, p, ln, tx in other[:30]))
                    shown = len(calls[:30]) + len(other[:30])
                    if len(rows) > shown:
                        files = len({r[2] for r in rows})
                        parts.append(f"({len(rows)} references across {files} "
                                     f"files, {shown} shown, library source "
                                     f"first. Use calls_only to narrow, or "
                                     f"read_file_range on a path above.)")
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

            elif name == "find_by_pattern":
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
                            if not _inside(root, p) or not os.path.isfile(p):
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
                        text = (no_index_error("find_by_pattern")
                                if not has_index() else
                                ok_result("find_by_pattern", matches=0,
                                          pattern=pat, files_scanned=scanned,
                                          index=index_state(), retryable=True,
                                          reason=("The index is populated and "
                                                  "the regex matched no line "
                                                  "in any scanned file.")))
                    if len(hits) >= limit:
                        text += f"\n[truncated at {limit}]"

            elif name == "read_file_range":
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
                    if _inside(root, cand) and os.path.isfile(cand):
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
                    want_end = end
                    end = min(end, len(lines))
                    if start > len(lines) or end < start:
                        # AN EMPTY CODE FENCE IS AN EMPTY RESULT WEARING
                        # PUNCTUATION. Measured, `start=3, end=1` returned
                        #
                        #     src/pool.ts:3-1  (4 lines total)
                        #     ```
                        #     ```
                        #
                        # and `start=1000000000` returned the same body under
                        # the header "1000000000-4". Both passed contract 1 on
                        # a character count and failed it on meaning: the
                        # caller cannot tell a file with no such lines from a
                        # reader that broke. Say which, and say it is the
                        # agent's to fix -- it is, and one call fixes it.
                        text = error_result(
                            "read_file_range", "EMPTY_RANGE",
                            (f"{rel} has {len(lines)} lines. The requested "
                             f"range {start}-{want_end} selects none of them, "
                             f"so nothing was read. This is not an empty "
                             f"file."),
                            retryable=True, path=rel, lines_in_file=len(lines),
                            requested={"start": start, "end": want_end},
                            remedies=[{"fixable_by": "agent",
                                       "action": (f"ask again for a range "
                                                  f"inside 1-{len(lines)}, "
                                                  f"with end >= start"),
                                       "effect": "the lines are returned"}])
                    else:
                        body = "\n".join(f"{i:>5}  {lines[i-1]}"
                                         for i in range(start, end + 1))
                        text = (f"{rel}:{start}-{end}  ({len(lines)} lines "
                                f"total)\n```\n{body}\n```")

            elif name == "summarize_text":
                focus = args.get("focus") or "anything an engineer would need to continue the work"
                maxw = int(args.get("max_words", 300))
                prompt = (
                    f"Compress the following into at most {maxw} words.\n"
                    f"Preserve verbatim: identifiers, file paths, line numbers, error "
                    f"strings and measurements. Drop pleasantries and repetition.\n"
                    f"Keep detail about: {focus}\n"
                    f"Return only the compressed text.\n\n---\n{args['text']}"
                )
                # Through the one door (mcp/model.py): the proxy's own effort
                # mapping and budget rule. This used to send max_tokens=900
                # straight to llama-swap; the model spent all 900 thinking and
                # returned 0 characters (measured live, 2026-09-22). The answer
                # allowance is now the summary's own size -- ~1.6 tokens per
                # word covers identifiers and paths -- with thinking on top.
                import model
                try:
                    text = model.ask(
                        [{"role": "user", "content": prompt}],
                        effort="low", max_tokens=int(maxw * 1.6) + 64,
                        timeout=3600).strip()
                    if not text:
                        # finish=stop with nothing written. Not a budget
                        # problem (that raises BudgetEvent), so it is not
                        # something retrying or shortening is known to fix.
                        text = error_result(
                            "summarize_text", "MODEL_RETURNED_NOTHING",
                            ("The model finished without writing a summary. "
                             "The text was not compressed; the original is "
                             "unchanged and still in your context."),
                            retryable=False,
                            remedies=[{"fixable_by": "agent",
                                       "action": "continue without compacting",
                                       "effect": "the original text is still in context"},
                                      {"fixable_by": "operator",
                                       "action": "check logs/proxy.log for this call",
                                       "why_not_the_agent": "the model's output is outside this conversation"}])
                except model.BudgetEvent as e:
                    text = error_result(
                        "summarize_text", "TOKEN_LIMIT",
                        ("The summary was not finished: " + str(e) + ". "
                         "Nothing was compressed; the original is unchanged."),
                        retryable=False,
                        detail=json.dumps(e.usage),
                        remedies=[{"fixable_by": "agent",
                                   "action": ("continue without compacting, or "
                                              "ask for fewer words (max_words)"),
                                   "effect": "a smaller answer fits the allowance"},
                                  {"fixable_by": "operator",
                                   "action": ("raise YAMADORI_REASONING_CAP or "
                                              "YAMADORI_ANSWER_MIN (mcp/tiers.py)"),
                                   "why_not_the_agent": "the budget is server configuration"}])
                except Exception as e:
                    # WAS: `text = f"compaction failed: {type(e).__name__}: {e}"`
                    # -- "compaction failed: URLError: <urlopen error [WinError
                    # 10061] ...>", which is contract 2 exactly: a raw
                    # exception handed to an agent as though it were an answer,
                    # with nothing in it to act on and no statement of whether
                    # trying again could help. It could: this is a reachability
                    # failure, not a bad request.
                    text = error_result(
                        "summarize_text", "MODEL_UNAVAILABLE",
                        ("The summarising model could not be reached, so "
                         "nothing was compressed. Your text is unchanged."),
                        retryable=True,
                        detail=f"{type(e).__name__}: {e}",
                        remedies=[
                            {"fixable_by": "agent",
                             "action": ("continue without compacting, or try "
                                        "once more"),
                             "effect": "the original text is still in context"},
                            {"fixable_by": "operator",
                             "action": ("check the model server named in "
                                        "LLAMA_STACK_URL is running"),
                             "effect": "summarisation works again",
                             "why_not_the_agent": ("the service is outside "
                                                   "this conversation")}])

            elif name == "describe_index":
                chunks, mat = load_index()
                files = sorted({c.path for c in chunks})
                text = (f"{len(chunks)} chunks across {len(files)} files.\n"
                        + "\n".join(f"  {f}" for f in files[:40]))
                if len(files) > 40:
                    text += f"\n  ... and {len(files) - 40} more"

            elif name == "run_check":
                import subprocess
                check = (args.get("check") or "").strip().lower()
                con = _db()
                roots = [r[0] for r in con.execute("SELECT path FROM roots").fetchall()]
                con.close()
                # Checks run from the project root, which is the parent of the
                # indexed src/ tree -- package.json lives there, not in src/.
                #
                # THE `or os.getcwd()` THAT USED TO BE HERE ran the check in
                # the SERVER's own working directory whenever the index had no
                # roots -- which, until the schema fix in _db() above, was any
                # database this module created. `npm run build` in the
                # llama-stack checkout is not what any caller asked for, and
                # the listing branch printed the server's absolute path into
                # the conversation as "project root".
                cwd = os.path.dirname(roots[0].rstrip("\\/")) if roots else ""
                if not check:
                    text = ("checks available (pass one as `check`):\n"
                            + "\n".join(f"  {k:<8} {' '.join(v)}"
                                        for k, v in VERIFY_CHECKS.items())
                            + (f"\nproject root: {cwd}" if cwd else
                               "\nNo indexed project root, so none of these "
                               "can be run yet."))
                elif not cwd:
                    text = error_result(
                        "run_check", "NO_PROJECT_ROOT",
                        ("The index names no project root, so there is no "
                         "directory to run this check in. Nothing was run."),
                        retryable=False,
                        remedies=[{"fixable_by": "user",
                                   "action": ("run the check locally and "
                                              "paste the output in"),
                                   "effect": "the result is then in context",
                                   "why_not_the_agent": ("this server has no "
                                                         "copy of the project "
                                                         "to run it in")}])
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
                text = rings.handle(name, args)
                if text is None:
                    return {"jsonrpc": "2.0", "id": mid,
                            "error": {"code": -32601, "message": f"unknown tool {name}"}}
            return ok({"content": [{"type": "text", "text": text}]})
        except Exception as e:                                   # noqa: BLE001
            # Structured, not "search failed: OperationalError: no such table:
            # roots". A raw exception string is our bug arriving in someone's
            # conversation dressed as an answer, and it tells the agent nothing
            # it can act on -- including whether trying again could ever help.
            return ok({"content": [{"type": "text", "text": error_result(
                name, "TOOL_RAISED", f"{type(e).__name__}: {e}",
                retryable=False,
                remedies=[{"fixable_by": "operator",
                           "action": "check the server log for the traceback",
                           "why_not_the_agent": ("the tool threw before "
                                                 "producing a result; no "
                                                 "arguments change that")}])}],
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
