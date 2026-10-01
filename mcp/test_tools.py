#!/usr/bin/env python
"""Every code tool, every state it can be in, against a stated contract.

WHAT CHANGED 2026-09-29 (docs/REMOVED.md; the way back is commit e360d37)

The code-intelligence tools (find_*, read_file_range, describe_index,
summarize_text, run_check) and the work log's record_step / read_rings are no
longer run by the proxy: the second brain that used them is gone, and so is
the proxy's no-repository package fallback. They stay on the MCP tools API
(:1235, mcp/tools_api.py -> code_search.handle), so this suite now calls
`code_search.handle` directly -- the one code path both the stdio and HTTP
MCP servers share. The proxy's own server tools (yama_generate_image,
yama_describe_image, yama_recall_craft, the MCP host's package lookups) are
called through `proxy.run_our_tool(name, args, state)`, and a name the proxy
no longer runs is answered NOT_A_SERVER_TOOL.

WHY THIS GATES THE BENCHMARKS

A night was spent measuring a model that was being handed empty strings. The
tool loop, the hop budget, the repeat tracker and the retry logic were all
tuned against a failure that was ours: with no repository bound -- the normal
condition for a remote service -- `run_our_tool` never executed anything and
returned "". Twelve identical searches over 842 seconds, scored as the model
failing to answer a question it was never given the means to answer.

None of that is visible from a benchmark. It is visible in one second here.

So: no benchmark runs until this passes. A benchmark measures a model through
the tools; if the tools are broken the number describes the tools, and it does
so without saying that is what it is measuring.

THE CONTRACT

  1. NEVER EMPTY. An empty result is indistinguishable from a crashed tool, a
     broken transport, or a tool that does not exist. It is the least
     informative thing a tool can say.
  2. NEVER A RAW EXCEPTION. "OperationalError: no such table: roots" is our
     bug leaking into someone's conversation as though it were an answer.
  3. THE SITUATION, NAMED. A missing index, a failed retriever and a query
     that matched nothing are three different facts and must read differently.
  4. `retryable` IS A FACT. False means no arguments can ever succeed here.
     An agent that retries a false is one we lied to.
  5. A REMEDY, WITH AN OWNER. What would change it, and whether the agent, the
     user or the operator is the one who can do it.
  6. NO CROSS-SESSION DATA. The work log is per-conversation. It once keyed on
     the SERVER's working directory, so every caller shared one log.
  7. NO GPU. These run in a second, so they run every time.

WHAT THIS FILE COVERS, AND WHAT IT DELIBERATELY DOES NOT

Every tool is exercised in every state it can reach without a GPU:

    no index bound          the normal condition for a remote caller
    index bound, HIT        the documented result format
    index bound, MISS       a real negative, distinguishable from a failure
    retrievers all down     an outage, which is NOT a zero-match result
    retrievers partly down  a negative that must say it is incomplete
    bad arguments           missing, wrong-typed and absurd values

`summarize_text` calls the model, so only its argument handling and an
unreachable model are checked here -- validation short-circuits before the
HTTP request, which is exactly why it can be. Its live behaviour lives in
`test_tools_live.py`, which is opt-in because it needs the card.

Findings and their fixes are written up in `docs/TOOLS.md`. A measurement that
only ever appeared in a conversation is a measurement that was not made.
"""
from __future__ import annotations

import json
import os
import sqlite3
import struct
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Never the real corpus. Two of this session's own test runs wrote synthetic
# turns into the file that trains the decision model before anyone noticed.
os.environ["YAMADORI_CORPUS_DB"] = os.path.join(
    tempfile.gettempdir(), "yamadori_test_corpus.sqlite3")
os.environ["RINGS_DB"] = os.path.join(
    tempfile.gettempdir(), "yamadori_test_rings.sqlite3")
# Nor the live ledger or jobs/skills store: proxy (imported below) opens
# nebari, and the knowledge-base test reads the skills store.
_DB_TMP = tempfile.mkdtemp(prefix="yamadori_test_tools_db_")
os.environ["YAMADORI_NEBARI_DB"] = os.path.join(_DB_TMP, "nebari.sqlite3")
os.environ["YAMADORI_JOBS_DB"] = os.path.join(_DB_TMP, "jobs.sqlite3")
os.environ["YAMADORI_SKILLS_DIR"] = os.path.join(_DB_TMP, "skills")
# Never the real package store either (what describes and screens a held
# package reads it).
PKG_STORE = tempfile.mkdtemp(prefix="yamadori_test_pkgs_")
os.environ["YAMADORI_PKG_DIR"] = PKG_STORE
# No image server, whatever the operator's shell has: yama_generate_image is offered
# only when one is configured, and this suite asserts both sides of that gate
# itself (test_generate_image_is_gated_and_says_why). The media store and URL
# key are temp files, never index/media.
os.environ.pop("YAMADORI_IMAGEGEN_URL", None)
os.environ.pop("YAMADORI_VISION", None)     # yama_describe_image on, as shipped
_MEDIA_TMP = tempfile.mkdtemp(prefix="yamadori_test_media_")
os.environ["YAMADORI_MEDIA_DIR"] = os.path.join(_MEDIA_TMP, "media")
os.environ["YAMADORI_MEDIA_SECRET_FILE"] = os.path.join(_MEDIA_TMP, "url.key")

# ---------------------------------------------------------------------------
# THE FIXTURE INDEX
#
# A throwaway repository and a throwaway index over it, built the way
# `code_search._db()` builds one, with contents chosen so that every branch
# below has a known right answer:
#
#   sizeKvPool       declared, and referenced -- exercises the hit paths
#   packGlyphAtlas   declared, and referenced NOWHERE -- exercises the dead
#                    code answer, which must not read like a failed lookup
#   src/pool.ts      has a sibling basename under lib/ so that a wrong-
#                    directory read exercises the "did you mean" branch
#
# CODE_INDEX_DB is pointed at it BEFORE code_search is imported, because that
# module reads the path at import time. The real index/code.sqlite3 holds 7,742
# chunks of this codebase; a test that read it would both be non-deterministic
# and answer its own questions with our source.
# ---------------------------------------------------------------------------
FIXTURE_FILES = {
    "src/pool.ts": (
        "export function sizeKvPool(slots: number): number {\n"
        "  // the kv pool is divided evenly among slots\n"
        "  return TOTAL_KV / slots;\n"
        "}\n"),
    "src/atlas.ts": (
        "export function packGlyphAtlas(glyphs: Glyph[]): Atlas {\n"
        "  return layout(glyphs);\n"
        "}\n"),
    "src/main.ts": (
        "import { sizeKvPool } from './pool';\n"
        "\n"
        "export function boot(): void {\n"
        "  const n = sizeKvPool(4);\n"
        "}\n"),
}

FIXTURE_ROOT = tempfile.mkdtemp(prefix="yamadori_test_repo_")
FIXTURE_DB = os.path.join(FIXTURE_ROOT, "code.sqlite3")
# A second database with chunks but NO roots table -- what `_db()` produced
# before the schema fix. See test_roots_table_is_part_of_the_schema.
PARTIAL_DB = os.path.join(FIXTURE_ROOT, "partial.sqlite3")
# An index that does not exist yet: the "nothing is indexed" state (code_search
# creates it empty on first use, inside this temp directory).
EMPTY_DB = os.path.join(FIXTURE_ROOT, "empty.sqlite3")

os.environ["CODE_INDEX_DB"] = FIXTURE_DB

# Every other store the code under test can write (the jobs database
# behind skill selection, the concept seed, ...), BEFORE any mcp import:
# 2026-09-27 this suite wrote the live index/ (the offline guard).
import offline_stores  # noqa: E402
offline_stores.isolate("yamadori_test_tools_stores_")

import code_search as cs  # noqa: E402
import fusion  # noqa: E402
import proxy  # noqa: E402
import served_fixture  # noqa: E402
# The served model's /props, pinned (mcp/served_fixture.py): budget and
# tiers would otherwise ask the live stack (llama-swap reloads `bonsai`).
served_fixture.pin()
# No offline test may reach the live model server (2026-09-24: a fixture
# that faked only proxy._post let the turn engine's _post_events reach
# :11434). The proxy's door and internal generation's (mcp/model.py) go to a
# port that refuses, unless a test points them at its own fake.
import model as _model  # noqa: E402
_model.UPSTREAM = "http://127.0.0.1:9"
proxy.UPSTREAM = "http://127.0.0.1:9"
# A pinned concept seed: no 420 MB matrix load in an offline suite.
proxy._draw_seed = lambda prompt=None: {"word": "cedar", "token_id": 1,
                                        "u32": 2}

# Tools that call the model. Excluded from the execute-for-real paths: this
# suite must not need a GPU, and an audit that hangs for ten minutes is an
# audit nobody runs.
NEEDS_MODEL = {"summarize_text", "judge"}

# The code tools that read an index (the MCP surface's search and read tools).
INDEX_TOOLS = {"find_by_meaning", "find_definition_opt", "find_references",
               "find_by_pattern", "read_file_range", "describe_index"}

# The real index, which nothing here may touch.
REAL_INDEX = os.path.abspath(os.path.join(HERE, "..", "index", "code.sqlite3"))


def build_fixture() -> None:
    for rel, body in FIXTURE_FILES.items():
        path = os.path.join(FIXTURE_ROOT, rel.replace("/", os.sep))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(body)

    con = sqlite3.connect(FIXTURE_DB)
    con.executescript("""
        CREATE TABLE IF NOT EXISTS chunks(
            id INTEGER PRIMARY KEY, path TEXT, start INT, end INT,
            text TEXT, vec BLOB);
        CREATE TABLE IF NOT EXISTS defs(
            name TEXT, kind TEXT, path TEXT, start INT, end INT, line TEXT);
        CREATE TABLE IF NOT EXISTS refs(
            name TEXT, kind TEXT, path TEXT, line_no INT, line TEXT);
        CREATE TABLE IF NOT EXISTS roots(path TEXT PRIMARY KEY);
    """)
    con.execute("INSERT OR IGNORE INTO roots VALUES(?)",
                (os.path.abspath(FIXTURE_ROOT),))
    # Every vector must be the same width: load_index takes the dimension from
    # the first row and reshapes the concatenated bytes over all of them. The
    # values are irrelevant -- the semantic retriever is off by configuration,
    # which is itself one of the facts asserted below.
    vec = struct.pack("<4f", 1.0, 0.0, 0.0, 0.0)
    for i, (rel, body) in enumerate(FIXTURE_FILES.items(), 1):
        con.execute("INSERT INTO chunks(id,path,start,end,text,vec) "
                    "VALUES(?,?,?,?,?,?)",
                    (i, rel, 1, body.count("\n"), body, vec))
    con.executemany(
        "INSERT INTO defs VALUES(?,?,?,?,?,?)",
        [("sizeKvPool", "function", "src/pool.ts", 1, 4,
          "export function sizeKvPool(slots: number): number {"),
         ("packGlyphAtlas", "function", "src/atlas.ts", 1, 3,
          "export function packGlyphAtlas(glyphs: Glyph[]): Atlas {")])
    con.executemany(
        "INSERT INTO refs VALUES(?,?,?,?,?)",
        [("sizeKvPool", "call", "src/main.ts", 4, "  const n = sizeKvPool(4);"),
         ("sizeKvPool", "ref", "src/main.ts", 1,
          "import { sizeKvPool } from './pool';")])
    con.commit()
    con.close()

    part = sqlite3.connect(PARTIAL_DB)
    part.execute("CREATE TABLE IF NOT EXISTS chunks(id INTEGER PRIMARY KEY, "
                 "path TEXT, start INT, end INT, text TEXT, vec BLOB)")
    part.execute("INSERT INTO chunks(path,start,end,text,vec) VALUES(?,?,?,?,?)",
                 ("src/pool.ts", 1, 4, FIXTURE_FILES["src/pool.ts"], vec))
    part.commit()
    part.close()


build_fixture()

_results: list[tuple[bool, str, str]] = []
# Which tools this run actually exercised. Populated by call(), so it cannot
# drift from what really happened -- see test_every_tool_has_a_test.
_exercised: set[str] = set()


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


# Which of the proxy's own server tools this run exercised (ours()).
_exercised_ours: set[str] = set()
WORK_LOG_TOOLS = {"record_step", "read_rings"}


def call(tool: str, args: dict, db=None, state=None) -> str:
    """A code tool through code_search.handle -- the MCP tools API's own path
    -- against `db` (default: an index that holds nothing). The work log is
    per conversation: its tools get the conversation's key as `_session`, as
    the proxy passed it when it ran them."""
    _exercised.add(tool)
    a = dict(args)
    if tool in WORK_LOG_TOOLS:
        a["_session"] = (state if state is not None
                         else {"_key": "test-session"}).get("_key", "")
    resp = cs.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                      "params": {"name": tool, "arguments": a}},
                     db=db or EMPTY_DB)
    if "result" not in resp:
        return json.dumps(resp)
    return resp["result"]["content"][0]["text"]


def indexed(tool: str, args: dict, **kw) -> str:
    """A call against the fixture index."""
    kw.setdefault("db", FIXTURE_DB)
    return call(tool, args, **kw)


def ours(tool: str, args: dict, state=None) -> str:
    """One of the proxy's own server tools, as main's hidden hop runs it."""
    _exercised_ours.add(tool)
    return proxy.run_our_tool(tool, args,
                              state if state is not None else {"_key": "test-session"})


def as_json(text: str):
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return None


def leaked_exception(text: str) -> list[str]:
    """Contract 2, as a predicate.

    These substrings are the shapes a raw Python or sqlite error takes when it
    escapes into a tool result. `ProgrammingError` is on the list because a
    list passed where a string belonged came back as "ProgrammingError: Error
    binding parameter 1: type 'list' is not supported" -- the storage layer
    naming itself in someone's conversation.
    """
    low = (text or "").lower()
    return [m for m in ("traceback", "operationalerror", "no such table",
                        "programmingerror", "attributeerror", "keyerror",
                        "valueerror", "typeerror:")
            if m in low]


# ---------------------------------------------------------------------------
# Contract 1 + 2, applied to every tool in the failure state that matters
# ---------------------------------------------------------------------------
def test_never_empty_never_raw():
    """Nothing indexed. Through the proxy this was the NORMAL case (no
    repository bound), and it returned ""."""
    cases = [
        ("find_by_meaning", {"query": "how is the KV pool sized"}),
        ("find_by_pattern", {"pattern": "minimumIncrements"}),
        ("find_definition_opt", {"symbol": "NoSuchSymbolAnywhere"}),
        ("find_references", {"symbol": "NoSuchSymbolAnywhere"}),
        ("read_file_range", {"path": "src/nope.ts", "start": 1, "end": 20}),
        ("describe_index", {}),
        ("run_check", {"label": "lint"}),
        ("record_step", {"kind": "did", "summary": "tried the median approach"}),
        ("read_rings", {}),
        ("yama_generate_image", {"prompt": "a lighthouse at dusk"}),
    ]
    for tool, args in cases:
        if tool in NEEDS_MODEL:
            continue
        try:
            out = (ours if tool in proxy.OUR_NAMES else call)(tool, args)
        except Exception as e:                                   # noqa: BLE001
            check(False, f"{tool} does not raise", f"{type(e).__name__}: {e}")
            continue
        check(bool((out or "").strip()), f"{tool} returns something",
              repr(out)[:60])
        low = (out or "").lower()
        leaked = [m for m in ("traceback", "operationalerror", "no such table",
                              "attributeerror", "keyerror", "typeerror:")
                  if m in low]
        check(not leaked, f"{tool} leaks no raw exception", str(leaked))


# ---------------------------------------------------------------------------
# Contract 3 + 4 + 5
# ---------------------------------------------------------------------------
def test_index_tools_name_the_situation():
    # describe_index is not in this list: with nothing indexed it states the
    # counts ("0 chunks across 0 files"), a fact rather than a failure. (The
    # proxy used to answer it NO_INDEX, or with the held packages, before it
    # stopped running the code tools, 2026-09-29.)
    for tool, args in (("find_by_meaning", {"query": "x"}),
                       ("find_by_pattern", {"pattern": "x"}),
                       ("find_definition_opt", {"symbol": "x"})):
        d = as_json(call(tool, args))
        if not check(d is not None, f"{tool} answers with structure"):
            continue
        check(d.get("ok") is False, f"{tool} reports failure, not emptiness",
              str(d.get("ok")))
        check(d.get("error") == "NO_INDEX", f"{tool} names the situation",
              str(d.get("error")))
        check(d.get("retryable") is False, f"{tool} says retrying cannot help",
              str(d.get("retryable")))
        rem = d.get("remedies") or []
        check(bool(rem), f"{tool} offers a remedy")
        check(all(r.get("fixable_by") in ("agent", "user", "operator")
                  for r in rem), f"{tool} says WHO can fix it",
              json.dumps([r.get("fixable_by") for r in rem]))


def test_non_index_tools_work_without_an_index():
    """These never needed a corpus and were gated behind one anyway."""
    out = call("record_step", {"kind": "did", "summary": "a recorded step"})
    check("recorded" in out.lower(), "record_step works with no index", out[:80])

    out = call("read_rings", {})
    check(bool(out.strip()) and "error" not in out[:20].lower(),
          "read_rings works with no index", out[:80])


# ---------------------------------------------------------------------------
# Contract 6 -- the disclosure this suite was written after finding
# ---------------------------------------------------------------------------
def test_work_log_is_per_session():
    call("record_step", {"kind": "learned", "summary": "SECRET-ALPHA detail"},
         state={"_key": "session-alpha"})
    other = call("read_rings", {}, state={"_key": "session-beta"})
    check("SECRET-ALPHA" not in other,
          "one session cannot read another's work log", other[:120])

    mine = call("read_rings", {}, state={"_key": "session-alpha"})
    check("SECRET-ALPHA" in mine, "a session CAN read its own log", mine[:120])

    unscoped = call("read_rings", {}, state={})
    check("SECRET-ALPHA" not in unscoped,
          "an unscoped call leaks nothing", unscoped[:120])
    check("ERROR" in unscoped, "an unscoped call says why", unscoped[:120])


# ---------------------------------------------------------------------------
# Adversarial and malformed input
# ---------------------------------------------------------------------------
def test_bad_input_is_answered_not_crashed():
    cases = [
        ("find_by_meaning", {}, "missing required argument"),
        ("find_by_pattern", {"pattern": "[unclosed"}, "invalid regex"),
        ("read_file_range", {"path": "../../../etc/passwd", "start": 1, "end": 5},
         "path traversal"),
        ("run_check", {"label": "rm -rf /"}, "command injection via label"),
        ("record_step", {"kind": "nonsense", "summary": "x"}, "unknown enum"),
        ("record_step", {"kind": "did"}, "missing summary"),
    ]
    for tool, args, why in cases:
        try:
            out = call(tool, args)
        except Exception as e:                                   # noqa: BLE001
            check(False, f"{tool} survives {why}", f"{type(e).__name__}: {e}")
            continue
        check(bool((out or "").strip()), f"{tool} answers on {why}", repr(out)[:60])
        check("traceback" not in (out or "").lower(),
              f"{tool} does not leak a traceback on {why}")
        if tool == "read_file_range":
            # With nothing indexed code_search answers in prose that echoes
            # the path it could not find; what must never appear is the
            # file's content.
            check("root:x:" not in (out or "")
                  and ("not found under any indexed root" in out
                       or "ERROR" in out or '"ok": false' in out),
                  "path traversal does not return the file", (out or "")[:100])
        if tool == "run_check" and "rm -rf" in json.dumps(args):
            check("deleted" not in (out or "").lower(),
                  "an arbitrary label is not executed as a command",
                  (out or "")[:100])


# ---------------------------------------------------------------------------
# THE WITH-INDEX PATH
#
# Until now every search tool was only ever tested with no index, which proves
# the error envelope and nothing about the tool. A tool that returns a
# beautifully structured NO_INDEX for every input and garbage for a real query
# would have passed this suite.
# ---------------------------------------------------------------------------
def test_the_fixture_is_not_the_real_index():
    """The guard that makes every assertion below trustworthy.

    If CODE_INDEX_DB ever resolved to index/code.sqlite3, the counts asserted
    here would be whatever this codebase happens to contain today, the tests
    would pass for the wrong reason, and a run of this suite would answer
    questions about the caller's code with ours.
    """
    check(os.path.abspath(cs.INDEX_DB) != REAL_INDEX,
          "the suite is not pointed at the real index", cs.INDEX_DB)
    check(os.path.abspath(FIXTURE_DB).startswith(
        os.path.abspath(tempfile.gettempdir())),
        "the fixture index lives in a temp directory", FIXTURE_DB)
    # The fixture has three files. The real index has thousands. If the two
    # were ever confused this count is the thing that would say so.
    chunks = cs.index_state()["chunks"]
    check(chunks == len(FIXTURE_FILES),
          "the index in force holds exactly the fixture", str(chunks))


def test_search_hits_return_the_documented_format():
    """A hit must carry path, line range and source, or it cannot be followed.

    `find_by_meaning` documents itself as returning "file path, line range and
    source text". read_file_range's description tells the model to use that
    location instead of searching again -- so a result missing any of the three
    sends it back into the loop this suite exists to close.
    """
    out = indexed("find_by_meaning", {"query": "size pool"})
    check("src/pool.ts" in out, "find_by_meaning hit names the file", out[:120])
    check("src/pool.ts:1-4" in out,
          "find_by_meaning hit carries a line range", out[:160])
    check("TOTAL_KV / slots" in out,
          "find_by_meaning hit carries the source text", out[:200])
    check("declares sizeKvPool" in out,
          "find_by_meaning names the evidence for its top tier", out[:200])
    check("TAPROOT" in out,
          "a declaration match is tiered as exact, not scored", out[:80])
    check(as_json(out) is None,
          "a hit is prose, not an error envelope", out[:80])

    out = indexed("find_by_pattern", {"pattern": "sizeKvPool"})
    for want in ("src/pool.ts:1:", "src/main.ts:1:", "src/main.ts:4:"):
        check(want in out, f"find_by_pattern reports {want}", out[:200])

    out = indexed("find_definition_opt", {"symbol": "sizeKvPool"})
    check("src/pool.ts:1-4" in out,
          "find_definition_opt points at the declaration", out[:200])
    check("function" in out, "find_definition_opt says what kind it is", out[:200])

    out = indexed("find_references", {"symbol": "sizeKvPool"})
    check("CALL SITES:" in out and "src/main.ts:4" in out,
          "find_references separates the call site", out[:200])
    check("src/main.ts:1" in out,
          "find_references also reports the import", out[:300])

    out = indexed("find_references", {"symbol": "sizeKvPool", "calls_only": True})
    check("src/main.ts:4" in out and "import" not in out,
          "calls_only narrows to invocations", out[:200])


def test_read_file_range_returns_the_right_lines():
    """The exact lines, numbered, from the file on disk -- not from the index.

    Off-by-one here is invisible: the model gets plausible code for the wrong
    lines and reasons about it confidently. So the boundaries are asserted
    rather than the presence of "some source".
    """
    out = indexed("read_file_range", {"path": "src/pool.ts", "start": 2, "end": 3})
    check("    2  " in out and "    3  " in out,
          "the requested lines are numbered", out[:200])
    check("    1  " not in out and "    4  " not in out,
          "lines outside the range are not returned", out[:200])
    check("// the kv pool is divided evenly among slots" in out,
          "line 2 is line 2 of the real file", out[:250])
    check("(4 lines total)" in out,
          "the file's true length is reported", out[:120])

    # The usual real-world miss: a plausible but wrong directory.
    out = indexed("read_file_range", {"path": "lib/pool.ts", "start": 1, "end": 3})
    check("not found under any indexed root" in out,
          "a wrong path says so", out[:120])
    check("src/pool.ts" in out,
          "a wrong directory is answered with the right one", out[:200])


def test_describe_index_reports_the_real_counts():
    """Facts about the index, matching what is actually in it.

    describe_index is what an agent calls to decide whether searching is worth
    doing. A number that is merely plausible is worse than none.
    """
    out = indexed("describe_index", {})
    n_files = len(FIXTURE_FILES)
    check(f"{n_files} chunks across {n_files} files" in out,
          "describe_index counts the fixture exactly", out[:120])
    for rel in FIXTURE_FILES:
        check(rel in out, f"describe_index lists {rel}", out[:200])
    check(leaked_exception(out) == [], "describe_index leaks nothing", out[:120])


def test_a_real_miss_is_a_real_miss():
    """A populated index that matched nothing. THIS is the retryable case.

    It has to be distinguishable from NO_INDEX by a machine, not by tone:
    ok stays true, matches is 0, and retryable is true because a different
    query genuinely could succeed. The retriever roll-call is included so the
    agent can see that the search actually happened -- including that the
    semantic arm is off by configuration rather than broken.
    """
    d = as_json(indexed("find_by_meaning", {"query": "quantum teapot recursion"}))
    if not check(d is not None, "a miss answers with structure"):
        return
    check(d.get("ok") is True, "a miss is not an error", str(d.get("ok")))
    check(d.get("matches") == 0, "a miss reports zero matches", str(d.get("matches")))
    check(d.get("retryable") is True,
          "a miss says another query could work", str(d.get("retryable")))
    check(d.get("error") is None, "a miss carries no error code", str(d.get("error")))
    r = d.get("retrievers") or {}
    check(set(r) == {"semantic", "lexical", "symbols"},
          "every retriever is accounted for", json.dumps(sorted(r)))
    check(r.get("lexical", {}).get("ran") is True,
          "the lexical retriever is reported as having run", json.dumps(r.get("lexical")))
    check(r.get("semantic", {}).get("reason") == "disabled by configuration",
          "a retriever that never ran says it was switched off",
          json.dumps(r.get("semantic")))
    check(d.get("complete") is True,
          "a clean miss is marked complete", str(d.get("complete")))
    check(d.get("retrievers_failed") == [],
          "a clean miss names no failures", json.dumps(d.get("retrievers_failed")))
    check("Every available retriever ran" in (d.get("reason") or ""),
          "a clean miss says the search really happened",
          (d.get("reason") or "")[:140])
    check((d.get("index") or {}).get("chunks") == len(FIXTURE_FILES),
          "the miss carries what the index does contain",
          json.dumps(d.get("index")))

    d = as_json(indexed("find_by_pattern", {"pattern": "zzz_no_such_token"}))
    if check(d is not None, "find_by_pattern miss answers with structure"):
        check(d.get("ok") is True and d.get("matches") == 0
              and d.get("retryable") is True,
              "find_by_pattern miss is ok/0/retryable", json.dumps(d)[:160])
        check(d.get("files_scanned") == len(FIXTURE_FILES),
              "find_by_pattern says how many files it read",
              str(d.get("files_scanned")))

    # A symbol miss against a POPULATED index must not claim there is no index.
    out = indexed("find_definition_opt", {"symbol": "sizeKvPoolz"})
    check("NO_INDEX" not in out,
          "a symbol miss is not reported as a missing index", out[:160])
    check("among 2 indexed symbols" in out,
          "a symbol miss says how many symbols were searched", out[:160])
    check("sizeKvPool" in out, "a symbol miss offers the near match", out[:200])

    # Defined, referenced nowhere. This is an ANSWER -- "it is dead code" --
    # and must not read like a lookup that failed.
    out = indexed("find_references", {"symbol": "packGlyphAtlas"})
    check("is defined" in out and "dead code" in out,
          "an unreferenced symbol is answered, not failed", out[:200])
    check("src/atlas.ts" in out,
          "the dead-code answer still says where it is declared", out[:200])

    # A glob that excludes everything is a THIRD situation, and the agent
    # cannot fix it by rephrasing the pattern -- so it is told what is indexed.
    out = indexed("find_by_pattern", {"pattern": "pool", "glob": "*.rs"})
    check("matched 0 of 3 indexed files" in out,
          "a glob that excludes everything says so", out[:160])
    check("src" in out, "a bad glob is answered with what IS indexed", out[:250])


# ---------------------------------------------------------------------------
# THE RETRIEVER-OUTAGE PATH
#
# Both retriever handlers in search_fused are `except: pass` with a diagnostic
# written beside them. Before that diagnostic existed, a total outage returned
# an empty fusion and the tool said the query missed -- an outage wearing the
# clothes of a bad query. These two tests are the only thing keeping that
# distinction honest, because nothing else can see it.
# ---------------------------------------------------------------------------
def test_total_outage_is_an_error_not_a_miss():
    """No retriever completed, so NOTHING was searched.

    Simulated by breaking fusion.content_words, which both surviving
    retrievers call inside their own try blocks -- the same shape a real
    failure takes, and it leaves load_index working, so the index still
    exists. Zero matches would be a lie here: the correct answer is that the
    search did not happen.
    """
    original = fusion.content_words
    cs._LEX = ((), None)

    def broken(*_a, **_k):
        raise RuntimeError("simulated retriever outage")

    fusion.content_words = broken
    try:
        out = indexed("find_by_meaning", {"query": "size pool"})
    finally:
        fusion.content_words = original
        cs._LEX = ((), None)

    d = as_json(out)
    if not check(d is not None, "a total outage answers with structure", out[:120]):
        return
    check(d.get("ok") is False, "a total outage is a failure", str(d.get("ok")))
    check(d.get("error") == "RETRIEVERS_UNAVAILABLE",
          "a total outage is NOT reported as zero matches", str(d.get("error")))
    check(d.get("matches") is None,
          "a total outage does not report a match count", str(d.get("matches")))
    check(d.get("retryable") is False,
          "an agent is not invited to retry a broken service",
          str(d.get("retryable")))
    r = d.get("retrievers") or {}
    check(r.get("lexical", {}).get("ran") is False
          and r.get("symbols", {}).get("ran") is False,
          "both failures are named individually", json.dumps(r)[:200])
    check("simulated retriever outage" in json.dumps(r),
          "the underlying error is carried for the operator", json.dumps(r)[:200])
    owners = [rem.get("fixable_by") for rem in (d.get("remedies") or [])]
    check(owners and all(o == "operator" for o in owners),
          "a failing service is the operator's to fix", json.dumps(owners))


def test_partial_outage_says_the_search_was_incomplete():
    """One retriever down, the rest found nothing.

    MEASURED, before the fix: this returned

        "reason": "Every available retriever ran and no chunk scored above
                   threshold for this query."

    while shipping `"lexical": {"ran": false, "reason": "failed"}` in the same
    object. The prose contradicted the data beside it, and the prose is what a
    model reads -- so a negative produced by half the index was presented as a
    complete one. A model would conclude the code does not exist.
    """
    original = cs._lexical

    def broken(*_a, **_k):
        raise RuntimeError("simulated lexical outage")

    cs._lexical = broken
    try:
        out = indexed("find_by_meaning", {"query": "quantum teapot recursion"})
    finally:
        cs._lexical = original

    d = as_json(out)
    if not check(d is not None, "a partial outage answers with structure", out[:120]):
        return
    check(d.get("matches") == 0, "a partial outage still reports the miss",
          str(d.get("matches")))
    check(d.get("retrievers_failed") == ["lexical"],
          "the result names which retriever failed",
          json.dumps(d.get("retrievers_failed")))
    check(d.get("complete") is False,
          "the miss is marked incomplete", str(d.get("complete")))
    check("Every available retriever ran" not in (d.get("reason") or ""),
          "the prose does not contradict the data beside it",
          (d.get("reason") or "")[:140])
    check("failed" in (d.get("reason") or ""),
          "the reason states that part of the index was not searched",
          (d.get("reason") or "")[:140])
    check((d.get("retrievers") or {}).get("symbols", {}).get("ran") is True,
          "the retriever that DID run is still credited",
          json.dumps(d.get("retrievers", {}).get("symbols")))


def test_roots_table_is_part_of_the_schema():
    """The `no such table: roots` failure, through its second door.

    The docstring at the top of this file records that failure being fixed by
    gating run_check behind a bound repository. It was only half fixed:
    `_db()` created chunks, defs and refs but never `roots`, and THREE tools
    read `SELECT path FROM roots`. Against any database this module made
    rather than scripts/index_code.py -- a half-built index, a dependency
    index, or the proxy's own `_no_repository_bound.sqlite3` (until the proxy
    stopped running these tools, 2026-09-29) -- all three raised:

        {"error": "TOOL_RAISED", "reason": "OperationalError: no such table:
         roots", "retryable": false, "remedies": [{"fixable_by": "operator"}]}

    PARTIAL_DB is exactly that database. These calls used to fail; they now
    answer, because zero roots is a fact such a database can state.
    """
    for tool, args in (("find_by_pattern", {"pattern": "pool"}),
                       ("read_file_range", {"path": "src/pool.ts"}),
                       ("run_check", {})):
        out = indexed(tool, args, db=PARTIAL_DB)
        check("no such table" not in out.lower(),
              f"{tool} survives an index with no roots table", out[:160])
        check(leaked_exception(out) == [],
              f"{tool} leaks no sqlite error on a rootless index", out[:160])
        check(bool((out or "").strip()),
              f"{tool} still answers on a rootless index", repr(out)[:80])

    # And the tool that would otherwise have run a build in the SERVER's own
    # directory says it cannot, rather than doing it somewhere arbitrary.
    d = as_json(indexed("run_check", {"check": "lint"}, db=PARTIAL_DB))
    if check(d is not None, "run_check with no root answers with structure"):
        check(d.get("error") == "NO_PROJECT_ROOT",
              "run_check refuses rather than picking a directory",
              str(d.get("error")))
        check(d.get("retryable") is False,
              "no arguments make an absent project root present",
              str(d.get("retryable")))


# ---------------------------------------------------------------------------
# ARGUMENT VALIDATION, FOR EVERY TOOL
#
# Contract 4 is what this section is really about. Before the fix, EVERY
# argument mistake came back as
#
#   {"error": "TOOL_RAISED", "reason": "KeyError: 'pattern'",
#    "retryable": false, "remedies": [{"fixable_by": "operator", ...}]}
#
# which is a raw exception (contract 2), a false `retryable` (contract 4), and
# the wrong owner (contract 5), all at once, for a mistake the caller could
# have corrected in one call.
# ---------------------------------------------------------------------------
BAD_ARGUMENT_CASES = [
    # tool, args, what is wrong with it
    ("find_by_meaning", {}, "missing query"),
    ("find_by_meaning", {"query": ""}, "empty query"),
    ("find_by_meaning", {"query": 12345}, "query is a number"),
    ("find_by_meaning", {"query": "x", "top_k": "lots"}, "top_k is a word"),
    ("find_by_meaning", {"query": "x", "top_k": 0}, "top_k asks for nothing"),
    ("find_by_pattern", {}, "missing pattern"),
    ("find_by_pattern", {"pattern": ""}, "empty pattern"),
    ("find_by_pattern", {"pattern": 42}, "pattern is a number"),
    ("find_by_pattern", {"pattern": "pool", "glob": 42}, "glob is a number"),
    ("find_by_pattern", {"pattern": "pool", "max_results": -1},
     "negative max_results"),
    ("find_definition_opt", {}, "missing symbol"),
    ("find_definition_opt", {"symbol": ""}, "empty symbol"),
    ("find_definition_opt", {"symbol": ["a"]}, "symbol is a list"),
    ("find_references", {}, "missing symbol"),
    ("find_references", {"symbol": None}, "null symbol"),
    # bool("false") is True, and so is bool("no") and bool("0"). A caller that
    # asked to widen the search got it narrowed, silently.
    ("find_references", {"symbol": "sizeKvPool", "calls_only": "false"},
     "calls_only is the STRING false"),
    ("read_file_range", {}, "missing path"),
    ("read_file_range", {"path": 7}, "path is a number"),
    ("read_file_range", {"path": "src/pool.ts", "start": "x"},
     "start is a word"),
    ("summarize_text", {}, "missing text"),
    ("summarize_text", {"text": ["a"]}, "text is a list"),
    ("summarize_text", {"text": "x", "max_words": 0}, "max_words asks for nothing"),
    ("record_step", {"kind": "did", "summary": 5}, "summary is a number"),
    ("describe_index", {"nonsense": True}, "an argument it does not take"),
    ("read_rings", {"limit": "sixty"}, "limit is a word"),
    ("read_rings", {"limit": 0}, "limit asks for nothing"),
    ("run_check", {"check": 5}, "check is a number"),
]


def test_bad_arguments_blame_the_caller_and_stay_retryable():
    for tool, args, why in BAD_ARGUMENT_CASES:
        try:
            out = indexed(tool, args)
        except Exception as e:                                   # noqa: BLE001
            check(False, f"{tool} survives {why}", f"{type(e).__name__}: {e}")
            continue
        if not check(bool((out or "").strip()), f"{tool} answers on {why}",
                     repr(out)[:60]):
            continue
        check(leaked_exception(out) == [],
              f"{tool} leaks no raw exception on {why}",
              str(leaked_exception(out)) + " " + out[:100])
        d = as_json(out)
        if d is None or d.get("ok") is not False:
            # Tools that answer argument mistakes in prose -- record_step's
            # "summary is required" -- are
            # acceptable, provided the prose says what to do. What is NOT
            # acceptable is silence or a stack trace, both checked above.
            check(len(out.strip()) > 20,
                  f"{tool} explains {why} rather than just refusing",
                  out[:100])
            continue
        check(d.get("error") != "TOOL_RAISED",
              f"{tool} does not report {why} as the server breaking",
              str(d.get("error")))
        check(d.get("retryable") is True,
              f"{tool} says {why} can be corrected and retried",
              str(d.get("retryable")))
        owners = [r.get("fixable_by") for r in (d.get("remedies") or [])]
        check(owners and all(o == "agent" for o in owners),
              f"{tool} blames the caller for {why}, not the operator",
              json.dumps(owners))


def test_absurd_but_legal_values_are_answered():
    """Enormous is not the same as invalid, and must not be refused.

    top_k, max_results and limit are slice and LIMIT bounds. A billion is
    harmless there, and rejecting it would invent a failure. Only the values
    that change the ANSWER -- zero and negative, which made find_by_pattern
    report a false negative -- are refused.
    """
    huge = 10 ** 9
    for tool, args in (("find_by_meaning", {"query": "size pool", "top_k": huge}),
                       ("find_by_pattern", {"pattern": "pool", "max_results": huge}),
                       ("read_rings", {"limit": huge})):
        out = indexed(tool, args)
        check(bool((out or "").strip()), f"{tool} answers an enormous limit",
              repr(out)[:60])
        check(leaked_exception(out) == [],
              f"{tool} does not crash on an enormous limit", out[:120])
    check("src/pool.ts" in indexed("find_by_meaning",
                                   {"query": "size pool", "top_k": huge}),
          "an enormous top_k still returns the right file")

    # The false negative this suite was written to catch. `pool` matches three
    # lines; with max_results below one, `len(hits) >= limit` was true before
    # the first file was read and the tool reported matches: 0.
    # A 20,000-character work-log entry. The log is read back in full after a
    # compaction, so an entry nobody capped is an entry that can swamp the
    # thing it exists to preserve. Recorded rather than refused today; what is
    # asserted is that it does not crash and does not silently vanish.
    out = indexed("record_step", {"kind": "did", "summary": "x" * 20000})
    check("recorded" in out.lower(), "an enormous work-log entry is recorded",
          out[:120])
    check(len(out) < 500, "and is acknowledged briefly, not echoed back",
          str(len(out)))

    for limit in (0, -1):
        d = as_json(indexed("find_by_pattern",
                            {"pattern": "pool", "max_results": limit}))
        check(d is not None and d.get("error") == "BAD_ARGUMENTS",
              f"max_results={limit} is refused, not answered with a false negative",
              json.dumps(d)[:160] if d else "")
        check(not (d and d.get("matches") == 0),
              f"max_results={limit} never reports zero matches for a pattern "
              f"that matches three lines", json.dumps(d)[:160] if d else "")

    # bool("false") is True. A caller asking to WIDEN the result set got it
    # narrowed, and nothing said so. There is no safe coercion here, so the
    # only honest answer is to refuse -- which means the result must not be
    # the narrowed one either.
    out = indexed("find_references",
                  {"symbol": "sizeKvPool", "calls_only": "false"})
    d = as_json(out)
    check(d is not None and d.get("error") == "BAD_ARGUMENTS",
          "a non-boolean calls_only is refused rather than coerced",
          out[:160])
    check("CALL SITES:" not in out,
          "and the flag is not silently read as true",
          out[:160])

    # An empty or inverted line range selects nothing. That is a fact about
    # the request, and it has to be said rather than rendered as an empty
    # code fence, which is contract 1 evaded by punctuation.
    for args, why in (({"path": "src/pool.ts", "start": 3, "end": 1}, "inverted"),
                      ({"path": "src/pool.ts", "start": 10 ** 9}, "past the end")):
        d = as_json(indexed("read_file_range", args))
        check(d is not None and d.get("error") == "EMPTY_RANGE",
              f"read_file_range names a range that is {why}",
              json.dumps(d)[:160] if d else "")
        if d:
            check(d.get("lines_in_file") == 4,
                  f"the {why} range answer says how long the file is",
                  str(d.get("lines_in_file")))
            check(d.get("retryable") is True,
                  f"a {why} range can be corrected by asking again",
                  str(d.get("retryable")))


def test_run_check_lists_and_refuses_without_executing_anything():
    """Every run_check state EXCEPT actually running a check.

    Running one is deliberately out of scope for the gate: it shells out to
    `npm run lint-core` with a 900s timeout, which would make this suite
    depend on node, a network and somebody's package.json. What is checked is
    everything around it -- and the thing that matters most, which is that the
    agent picks a LABEL and never a command line.
    """
    out = indexed("run_check", {})
    check("lint" in out and "test" in out and "build" in out,
          "run_check lists what this project offers", out[:200])
    check(FIXTURE_ROOT.rsplit(os.sep, 1)[0] in out or "project root" in out,
          "the listing says where checks would run", out[-160:])

    out = indexed("run_check", {"check": "typecheck"})
    check("unknown check" in out.lower(),
          "an unknown check is named as unknown", out[:160])
    check("lint" in out,
          "and the available ones are listed instead of guessed at", out[:160])

    # The allow-list is the security boundary: VERIFY_CHECKS maps a label to a
    # fixed argv, so text arriving from a source file, a search result or a
    # model cannot reach a shell. Asserted here with an index bound, where the
    # dispatch is genuinely reached.
    for hostile in ("lint; rm -rf /", "lint && curl evil.example",
                    "lint`whoami`", "../../bin/sh"):
        out = indexed("run_check", {"check": hostile})
        check("unknown check" in out.lower(),
              "a check name that is a command line is refused", out[:120])
        check("PASSED" not in out and "FAILED" not in out and "exit " not in out,
              "and nothing was executed", out[:120])


def test_summarize_text_names_an_unreachable_model():
    """The one summarize_text state reachable without a GPU, and it was broken.

    Pointing STACK at a closed port is exactly what the agent sees when the
    stack is down. MEASURED, before the fix, the tool returned the bare string

        compaction failed: URLError: <urlopen error [WinError 10061] No
        connection could be made because the target machine actively refused it>

    which is contract 2 in one line: a raw exception presented as an answer,
    with no statement of whether retrying could help. It can -- the text was
    never sent, so nothing about the request is wrong. An agent reading that
    string has no way to know its own context is intact.
    """
    # summarize_text reaches the model through the one door, mcp/model.py, so
    # that is what is pointed at the closed port. Patching cs.STACK alone
    # stopped isolating this test once the door existed: it reached the REAL
    # model and got a real summary.
    import model
    original = (cs.STACK, model.UPSTREAM)
    cs.STACK = model.UPSTREAM = "http://127.0.0.1:1"   # reserved; refuses
    try:
        out = indexed("summarize_text",
                      {"text": "a long log " * 40, "focus": "errors"})
    finally:
        cs.STACK, model.UPSTREAM = original

    check(not out.lower().startswith("compaction failed"),
          "an unreachable model is not reported as a bare exception string",
          out[:140])
    check(leaked_exception(out.split('"detail"')[0]) == [],
          "no exception name appears outside the operator's detail field",
          out[:140])
    d = as_json(out)
    if not check(d is not None, "an unreachable model answers with structure",
                 out[:140]):
        return
    check(d.get("error") == "MODEL_UNAVAILABLE",
          "the situation is named", str(d.get("error")))
    check(d.get("retryable") is True,
          "an unreachable service is retryable -- the request was never sent",
          str(d.get("retryable")))
    check("unchanged" in (d.get("reason") or ""),
          "the agent is told its own text survived", (d.get("reason") or "")[:120])
    owners = {r.get("fixable_by") for r in (d.get("remedies") or [])}
    check(owners == {"agent", "operator"},
          "both the workaround and the real fix are named, with owners",
          json.dumps(sorted(owners)))
    check(leaked_exception(d.get("reason") or "") == [],
          "the reason line is prose, not an exception",
          (d.get("reason") or "")[:120])
    check("URLError" in json.dumps(d) or "Error" in (d.get("detail") or ""),
          "the underlying error is kept, in a field, for the operator",
          (d.get("detail") or "")[:120])


def test_path_traversal_is_refused_with_an_index_present():
    """The traversal check, repeated where it can actually do damage.

    With nothing indexed there are no roots to join against, so the check
    above proves little about the resolver. With roots present, the join and
    the prefix test are really exercised.
    """
    for rel in ("../../../../Windows/win.ini", "..\\..\\..\\etc\\passwd",
                os.path.abspath(__file__)):
        out = indexed("read_file_range", {"path": rel, "start": 1, "end": 5})
        check("not found under any indexed root" in out or '"ok": false' in out,
              "a path outside the indexed roots is refused", out[:120])
        check("[extensions]" not in out and "root:x:" not in out
              and "THE CONTRACT" not in out,
              "no file outside the roots is returned", out[:160])


# ---------------------------------------------------------------------------
# THE COVERAGE ASSERTION
# ---------------------------------------------------------------------------
def test_generate_image_is_gated_and_says_why():
    """yama_generate_image with no image server: offered nowhere, and a call that
    arrives anyway (a stale transcript, a replay) is answered with the
    situation, a false `retryable`, and remedies with owners. Its behaviour
    against a server lives in mcp/test_images.py."""
    names = {t["function"]["name"] for t in proxy.main_tools(None)[0]}
    check("yama_generate_image" not in names,
          "unconfigured: yama_generate_image is not offered", str(sorted(names)))
    d = as_json(ours("yama_generate_image", {"prompt": "a lighthouse at dusk"}))
    check(d is not None and d.get("ok") is False
          and d.get("error") == "IMAGEGEN_NOT_CONFIGURED",
          "a call names the situation", json.dumps(d)[:160])
    check(d is not None and d.get("retryable") is False,
          "and says retrying cannot help", str((d or {}).get("retryable")))
    rem = (d or {}).get("remedies") or []
    check(rem and all(r.get("fixable_by") in ("agent", "user", "operator")
                      for r in rem), "with remedies that name an owner",
          json.dumps(rem)[:160])
    os.environ["YAMADORI_IMAGEGEN_URL"] = "http://127.0.0.1:9"
    try:
        names = {t["function"]["name"] for t in proxy.main_tools(None)[0]}
        check("yama_generate_image" in names,
              "configured: yama_generate_image is offered", str(sorted(names)))
        check("yama_describe_image" in names,
              "yama_describe_image goes wherever yama_generate_image goes",
              str(sorted(names)))
    finally:
        os.environ.pop("YAMADORI_IMAGEGEN_URL", None)


def test_descriptions_say_remote_read_only():
    """The code tools describe a REMOTE, READ-ONLY library-source service, and
    send the user's files to the client's own tools; so do main's image tools.

    Operator report, 2026-09-23: in Hermes, "start this project in
    ~/Develop..." was spent on these tools instead of the harness's file
    tools. (The second brain's copy of this set, and the addendum on main that
    named its fold-back phrases, were removed 2026-09-29.)
    """
    by_name = {t["name"]: t["description"] for t in cs.ALL_TOOLS}
    os.environ["YAMADORI_IMAGEGEN_URL"] = "http://127.0.0.1:9"
    try:
        main = {t["function"]["name"]: t["function"]["description"]
                for t in proxy.image_tools()}
    finally:
        os.environ.pop("YAMADORI_IMAGEGEN_URL", None)
    for want in ("find_by_meaning", "find_definition_opt", "find_references",
                 "find_by_pattern", "read_file_range", "describe_index",
                 "summarize_text", "run_check", "record_step", "read_rings"):
        check(want in by_name, f"{want} is on the tools API's surface")
    for want in ("yama_generate_image", "yama_describe_image"):
        check(want in main, f"{want} is main's image tool")
    for name, text in by_name.items():
        low = text.lower()
        check("this server" not in low and "proxy" not in low,
              f"{name}: no 'this server', no 'proxy'", text[:120])
    for name in sorted(INDEX_TOOLS):
        d = by_name[name]
        check(cs.REMOTE_READ_ONLY in d and "read-only" in d.lower(),
              f"{name}: says remote and read-only", d[-160:])
        check("client's own file tools" in d,
              f"{name}: names the client's own tools for the user's files",
              d[-160:])
        check(d.startswith("Answers '"),
              f"{name}: leads with the question it answers", d[:60])
    for name, d in (("run_check", by_name["run_check"]),
                    ("summarize_text", by_name["summarize_text"]),
                    ("record_step", by_name["record_step"]),
                    ("yama_generate_image", main.get("yama_generate_image", "")),
                    ("yama_describe_image", main.get("yama_describe_image", ""))):
        check("client's own" in d,
              f"{name}: sends the user's files to the client's own tools",
              d[-200:])


def test_describe_image_names_what_it_can_see():
    """yama_describe_image with nothing to look at: a call that arrives anyway is
    answered with the situation, whether retrying helps, and a remedy -- and
    a path is never read. Its behaviour against a vision server lives in
    mcp/test_vision.py."""
    names = {t["function"]["name"] for t in proxy.main_tools(None)[0]}
    check("yama_describe_image" not in names,
          "no image server and nothing attached: yama_describe_image is not offered",
          str(sorted(names)))
    d = as_json(ours("yama_describe_image", {"image": "image-0123456789",
                                        "question": "what is in it?"}))
    check(d is not None and d.get("ok") is False
          and d.get("error") == "UNKNOWN_IMAGE" and d.get("retryable") is True,
          "an id the conversation does not hold: UNKNOWN_IMAGE", json.dumps(d)[:200])
    check(d is not None and "available" in d and d.get("remedies"),
          "it names what IS available, with a remedy", json.dumps(d)[:200])
    d = as_json(ours("yama_describe_image", {"image": "C:/Windows/win.ini",
                                        "question": "read it"}))
    check(d is not None and d.get("error") == "UNKNOWN_IMAGE",
          "a filesystem path is not an image id", json.dumps(d)[:200])


def test_concurrent_calls_never_swap_indexes():
    """Pre-deploy review, 2026-09-24 (BLOCKS DEPLOY #2). The proxy's package
    and repository paths saved, mutated and restored the process-wide
    cs.INDEX_DB / CODE_INDEX_DB with no lock; two requests interleaved and
    one got the other's repository source. The database is now PASSED
    (code_search.handle(db=)) and bound per thread. Here two calls are forced
    to interleave: each blocks inside its tool until the other is inside its
    own."""
    import threading
    tmp = tempfile.mkdtemp(prefix="yamadori_test_swap_")
    vec = struct.pack("<4f", 1.0, 0.0, 0.0, 0.0)
    dbs = {}
    for who in ("a", "b"):
        db = os.path.join(tmp, f"account_{who}.sqlite3")
        con = sqlite3.connect(db)
        con.execute("CREATE TABLE chunks(id INTEGER PRIMARY KEY, path TEXT, "
                    "start INT, end INT, text TEXT, vec BLOB)")
        con.execute("INSERT INTO chunks(path,start,end,text,vec) "
                    "VALUES(?,?,?,?,?)",
                    (f"account_{who}/secret.ts", 1, 1, f"const {who} = 1;",
                     vec))
        con.commit()
        con.close()
        dbs[who] = db
    prev_db, prev_env = cs.INDEX_DB, os.environ.get("CODE_INDEX_DB")
    barrier = threading.Barrier(2, timeout=5)
    waited = threading.local()
    orig = cs.load_index

    def interleaved():
        if not getattr(waited, "done", False):
            waited.done = True
            try:
                barrier.wait()
            except threading.BrokenBarrierError:
                pass
        return orig()

    out: dict = {}

    def run_a():
        out["a"] = call("describe_index", {}, db=dbs["a"])

    def run_b():
        out["b"] = call("describe_index", {}, db=dbs["b"])

    cs.load_index = interleaved
    try:
        ts = [threading.Thread(target=run_a), threading.Thread(target=run_b)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(15)
    finally:
        cs.load_index = orig
    a, b = out.get("a") or "", out.get("b") or ""
    check("account_a/secret.ts" in a and "account_b" not in a,
          "interleaved: the first call reads ITS index, not the other "
          "request's", a[:200])
    check("account_b/secret.ts" in b and "account_a" not in b,
          "interleaved: the second call reads ITS index, not the other "
          "request's", b[:200])
    check(cs.INDEX_DB == prev_db
          and os.environ.get("CODE_INDEX_DB") == prev_env,
          "the process-wide index and CODE_INDEX_DB are never mutated",
          f"{cs.INDEX_DB} / {os.environ.get('CODE_INDEX_DB')}")


def test_the_proxys_other_server_tools_answer_or_say_why():
    """The proxy's other server tools through its own dispatch
    (proxy.run_our_tool): each answers, or fails with the situation, whether
    a retry helps, and a remedy. And a name it no longer runs -- a code tool,
    a removed tool of ours -- is NOT_A_SERVER_TOOL, not silence."""
    out = ours("yama_recall_craft", {"name_or_topic": "no such craft here"})
    d = as_json(out)
    check(d and d.get("error") == "NO_SUCH_CRAFT"
          and d.get("retryable") is True and d.get("remedies")
          and "skill" not in out.lower(),
          "yama_recall_craft with nothing matching: the situation, "
          "retryable, a remedy -- and the model reads 'craft'",
          (out or "")[:200])
    # The MCP host's package lookups (mcp/mcp_host.py): this process never
    # enabled the host (server.py does), so each says so -- the situation,
    # not retryable, a remedy with its owner. Their own suite, against a real
    # stdio server, is mcp/test_mcp_host.py.
    for name, args in (
            ("yama_find_package", {"query": "a date library",
                                   "ecosystem": "npm"}),
            ("yama_list_package_versions", {"package": "left-pad",
                                            "ecosystem": "npm"}),
            ("yama_read_package_readme", {"package": "left-pad",
                                          "ecosystem": "npm"}),
            ("yama_resolve_packages", {"packages": ["left-pad"],
                                       "ecosystem": "npm"})):
        d = as_json(ours(name, args))
        check(d and d.get("error") == "MCP_SERVER_OFF"
              and d.get("retryable") is False
              and any(r.get("fixable_by") == "operator"
                      for r in d.get("remedies") or []),
              f"{name} with the MCP host off: the situation, not "
              f"retryable, the operator's remedy", json.dumps(d)[:200])
    for name, args in (("find_by_meaning", {"query": "size pool"}),
                       ("record_step", {"kind": "did", "summary": "x"}),
                       ("yama_think_deeply", {"question": "why?"}),
                       ("yama_plan", {"task": "build a voxel pagoda"}),
                       ("delegate_investigation", {"question": "how is it"}),
                       ("search_web", {"query": "three.js"})):
        out = proxy.run_our_tool(name, args, {"_key": "test-session"})
        d = as_json(out)
        check(d and d.get("ok") is False
              and d.get("error") == "NOT_A_SERVER_TOOL"
              and d.get("retryable") is False and d.get("remedies")
              and "Nothing was run" in (d.get("reason") or ""),
              f"{name}: the proxy no longer runs it, and says so -- nothing "
              f"was run", (out or "")[:200])


def test_every_tool_has_a_test():
    """A tool added later cannot go quietly untested.

    `_exercised` and `_exercised_ours` are filled by call() and ours()
    themselves rather than by a hand-maintained list, so they record what
    this run actually did. The failure mode being prevented is the ordinary
    one: somebody adds a tool, the suite still says all green, and it says so
    about the tools it had before.
    """
    surface = {t["name"] for t in cs.ALL_TOOLS}
    missing = sorted(surface - _exercised)
    check(not missing, "every tool on the tools API (code_search.ALL_TOOLS) "
          "has at least one test case", "untested: " + ", ".join(missing))
    stray = sorted(_exercised - surface)
    check(not stray, "no test exercises a code tool the tools API does not "
          "offer", "unknown: " + ", ".join(stray))
    missing = sorted(proxy.OUR_NAMES - _exercised_ours)
    check(not missing, "every tool in proxy.OUR_NAMES has at least one test "
          "case", "untested: " + ", ".join(missing))
    stray = sorted(_exercised_ours - proxy.OUR_NAMES)
    check(not stray, "no test calls a server tool the proxy does not run",
          "unknown: " + ", ".join(stray))
    # 23 until 2026-09-29, when the code tools, the work-log tools, the
    # research tools, yama_think_deeply and yama_plan left the proxy
    # (docs/REMOVED.md): what it runs now is the two image tools,
    # yama_recall_craft and the MCP host's four package lookups
    # (yama_find_package, yama_list_package_versions,
    # yama_read_package_readme, yama_resolve_packages; the old names are
    # legacy, not OUR_NAMES).
    check(len(proxy.OUR_NAMES) == 7
          and {"yama_generate_image", "yama_describe_image",
               "yama_recall_craft", "yama_find_package",
               "yama_resolve_packages"} <= proxy.OUR_NAMES
          and not surface & proxy.OUR_NAMES
          and not {"yama_think_deeply", "yama_plan", "yama_package_versions",
                   "find_in_knowledge_base"} & proxy.OUR_NAMES
          and not set(proxy.LEGACY_TOOL_NAMES) & proxy.OUR_NAMES,
          "the tool count is what the audit was written against",
          str(sorted(proxy.OUR_NAMES)))

    # `judge` is implemented in code_search.handle and wired to Laya, and it
    # appears in NO tool list: not TOOLS, not INTERNAL_TOOLS, not OUR_NAMES.
    # That is deliberate -- eval_judge.py scored it 6/10 against a 5/10 coin
    # flip.
    #
    # It USED to dispatch anyway: the boundary was a comment, and `tools/call`
    # with name "judge" walked straight past it. `handle` now checks the name
    # against ALL_TOOLS before dispatching, so a name nobody is offered is not
    # callable. Asserted here so that re-exposing it is a decision rather than
    # an accident -- and note this check replaced one whose NAME said "judge is
    # nonetheless reachable" while its assertion only tested that an argspec
    # entry existed. It passed before and after the behaviour changed, which is
    # the coverage-shaped-hole this suite exists to prevent.
    listed = {t["name"] for t in cs.ALL_TOOLS}
    check("judge" not in listed and "judge" not in proxy.OUR_NAMES,
          "judge is advertised nowhere", str("judge" in listed))
    unlisted = cs.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                          "params": {"name": "judge", "arguments": {}}})
    check("error" in unlisted and unlisted["error"].get("code") == -32601,
          "an unadvertised name does not dispatch",
          json.dumps(unlisted)[:140])
    check("result" not in unlisted,
          "and no result is produced for it", json.dumps(unlisted)[:140])


def main() -> int:
    for fn in (test_never_empty_never_raw,
               test_index_tools_name_the_situation,
               test_non_index_tools_work_without_an_index,
               test_work_log_is_per_session,
               test_bad_input_is_answered_not_crashed,
               test_the_fixture_is_not_the_real_index,
               test_search_hits_return_the_documented_format,
               test_read_file_range_returns_the_right_lines,
               test_describe_index_reports_the_real_counts,
               test_a_real_miss_is_a_real_miss,
               test_total_outage_is_an_error_not_a_miss,
               test_partial_outage_says_the_search_was_incomplete,
               test_roots_table_is_part_of_the_schema,
               test_bad_arguments_blame_the_caller_and_stay_retryable,
               test_absurd_but_legal_values_are_answered,
               test_run_check_lists_and_refuses_without_executing_anything,
               test_summarize_text_names_an_unreachable_model,
               test_path_traversal_is_refused_with_an_index_present,
               test_generate_image_is_gated_and_says_why,
               test_describe_image_names_what_it_can_see,
               test_descriptions_say_remote_read_only,
               test_concurrent_calls_never_swap_indexes,
               test_the_proxys_other_server_tools_answer_or_say_why,
               test_every_tool_has_a_test):
        print(f"\n--- {fn.__name__} ---")
        n0 = len(_results)
        try:
            fn()
        except Exception:                                        # noqa: BLE001
            check(False, f"{fn.__name__} itself raised",
                  traceback.format_exc().strip().split("\n")[-1])
        for ok, name, detail in _results[n0:]:
            print(("  pass  " if ok else "  FAIL  ") + name
                  + (f"   <- {detail}" if not ok and detail else ""))

    passed = sum(1 for ok, _, _ in _results if ok)
    total = len(_results)
    print(f"\n{'=' * 70}\n  {passed}/{total} checks passed")
    if passed < total:
        print("  BENCHMARKS ARE GATED ON THIS. Fix the tools, not the harness.")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
