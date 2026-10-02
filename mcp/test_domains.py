#!/usr/bin/env python
"""The tool-admission gate, asserted. No GPU, no model, no real session store.

WHAT THIS IS GATING

`domains.tool_admission` decides whether any library source this server
holds can bear on a request. It once decided whether a request carried the
capability block and our twelve tool definitions; since 2026-09-24 no tool
of ours rides on main for it, and since 2026-09-29 the injections that read
it are removed (docs/REMOVED.md) -- what reads it now is the route
(mcp/route.py: a library_question needs it; proxy.prepare records the
gate's situation in `_route.signals.gate`). Both directions of error were
measured harms, which is why each one has checks here:

  offered wrongly    bench/lcb_after.jsonl: a LiveCodeBench puzzle carried
                     3,145 prompt tokens against `minimal`'s 416 -- 2,729 of
                     them (87%) the block and tool list -- and called no tool.
                     Two siblings ran 849s and 889s. (n=3.)
  withheld wrongly   a three.js question loses the one thing this stack exists
                     to add: the library's real source.

WHY THE OLD GATE IS ALSO TESTED AGAINST

The previous gate was `root or code_search.has_index()`. `has_index()` read
index/code.sqlite3 -- the server's own source -- which a caller with no
repository is never searched against. It was true on every deployment, so the
gate never fired. `test_the_gate_does_not_read_the_servers_own_index` pins the
repair: the decision must not move when `cs.INDEX_DB` does.

REAL INPUT (PROTOCOL rule 7)

Every rule is first checked on a fixture package store in a temp directory.
Then `test_real_prompts_against_the_real_store` replays prompts captured from
real producers -- 342 LiveCodeBench prompts built by bench/livecodebench.py's
own templates, 26 hand-written three.js questions -- against the real package
store, opened read-only. If any of those files is missing that test says
SKIPPED, loudly, and does not count as a pass.

NOTHING HERE WRITES OUTSIDE A TEMP DIRECTORY. `YAMADORI_NEBARI_DB`,
`CODE_INDEX_DB`, `RINGS_DB` and every store in offline_stores.STORES (the
jobs database above all: skill selection's records) are pointed at temp
files BEFORE the proxy is imported, because those modules read their paths
at import time. (Until 2026-09-27 the jobs database was not, and this file
wrote the live one.)
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

_TMP = tempfile.mkdtemp(prefix="yamadori_test_domains_")
os.environ["YAMADORI_NEBARI_DB"] = os.path.join(_TMP, "nebari.sqlite3")
os.environ["CODE_INDEX_DB"] = os.path.join(_TMP, "code.sqlite3")
os.environ["RINGS_DB"] = os.path.join(_TMP, "rings.sqlite3")
# And every other store proxy.prepare can write (2026-09-27: without
# YAMADORI_JOBS_DB, prepare's skill selection wrote 21 fallback records --
# this file's "where is Object3D defined?" -- into the live
# index/jobs.sqlite3, and the worker learned from them).
import offline_stores  # noqa: E402
offline_stores.isolate("yamadori_test_domains_stores_")

import domains  # noqa: E402
import nebari  # noqa: E402
import proxy  # noqa: E402
import served_fixture  # noqa: E402
# The served model's /props, pinned (mcp/served_fixture.py): budget and
# tiers would otherwise ask the live stack (llama-swap reloads `bonsai`).
served_fixture.pin()
served_fixture.no_embedder()   # skill selection's embedder: an outage

REAL_STORE = os.path.join(REPO, "index", "packages")
REAL_NEBARI = os.path.abspath(os.path.join(REPO, "index", "nebari.sqlite3"))

# Skills off: skill selection would call the live embeddings model; this
# suite must not need one. (The retrieval / fan-out / deep-thinking flags it
# once set were removed with their features, 2026-09-29.)
FEATURES = '{"skills": false}'

# Built by bench/livecodebench.py's PROMPT_STDIN, verbatim template, on a real
# AtCoder-shaped question. The template is the part that matters: every one of
# the 342 real prompts carries "competitive programming" and "standard input".
LCB_PROMPT = """You will be given a competitive programming question.
Write a complete Python 3 program that reads from standard input and writes to
standard output.

Return only the program, in a single ```python code block.

### Question
You are given a simple undirected graph with N vertices and M edges. For each
vertex, print the number of vertices reachable from it.

Input

The input is given from Standard Input in the following format:
N M
u_1 v_1

Constraints

- 1 <= N <= 2 x 10^5
"""

_results: list[tuple[bool, str, str]] = []
_skipped: list[str] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


def make_package(store: str, name: str, version: str, chunks: int = 1,
                 defs: tuple = ()) -> str:
    """A package index in the shape scripts/index_code.py writes."""
    import deps
    db = os.path.join(store, f"{deps.slug(name, version)}.sqlite3")
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE chunks(id INTEGER PRIMARY KEY, path TEXT, "
                "start INT, end INT, text TEXT, vec BLOB)")
    con.execute("CREATE TABLE defs(name TEXT, kind TEXT, path TEXT, "
                "start INT, end INT, line TEXT)")
    for i in range(chunks):
        con.execute("INSERT INTO chunks(path, start, end, text) VALUES(?,?,?,?)",
                    (f"src/f{i}.js", 1, 2, "x"))
    # A real index has definitions; held_sources() now requires them
    # (deps.index_health). `defs=None` builds the README-only shape on purpose.
    if defs is None:
        defs = ()
    elif not defs and chunks:
        defs = ("fixtureDefinition",)
    for d in defs:
        con.execute("INSERT INTO defs VALUES(?,?,?,?,?,?)",
                    (d, "function", "src/f0.js", 1, 2, f"function {d}()"))
    con.commit()
    con.close()
    return db


def store_with(*packages) -> str:
    s = tempfile.mkdtemp(prefix="store_", dir=_TMP)
    for p in packages:
        make_package(s, *p)
    return s


# The deployed shape: three and typegpu, both live, with a few real symbols.
HELD = store_with(("three", "0.185.1", 3,
                   ("Object3D", "painterSortStable", "DEFAULT_UP", "sort")),
                  ("typegpu", "0.12.5", 2, ("OmitBuiltins", "run")))
EMPTY = tempfile.mkdtemp(prefix="empty_", dir=_TMP)


def decide(content, store=HELD, root=None, system=None, discovered=None,
           offered_before=False, extra=None) -> dict:
    msgs = ([{"role": "system", "content": system}] if system else [])
    msgs.append({"role": "user", "content": content})
    msgs += extra or []
    return domains.tool_admission(msgs, root, discovered=discovered,
                                  offered_before=offered_before, store=store)


def user(text: str) -> list[dict]:
    return [{"role": "user", "content": text}]


def body(messages: list[dict]) -> dict:
    return {"model": "yamadori", "reasoning_effort": "medium",
            "messages": messages, "_client_ip": "127.0.0.1",
            "_features": FEATURES}


def prepare_gated(b: dict) -> tuple[dict, dict | None]:
    """proxy.prepare, and the gate decision it made (proxy.tool_gate's
    return). The decision is no longer carried on the payload (the
    `_tools_gate` record went with x_yamadori.tools_gate, 2026-09-29); the
    route's signals carry its situation only for a question."""
    seen: list = []
    real = proxy.tool_gate

    def spy(*a, **k):
        seen.append(real(*a, **k))
        return seen[-1]
    proxy.tool_gate = spy
    try:
        out = proxy.prepare(b)
    finally:
        proxy.tool_gate = real
    return out, (seen[-1] if seen else None)


def our_tool_names(payload: dict) -> set[str]:
    return {t["function"]["name"] for t in payload.get("tools") or []
            if t.get("function", {}).get("name") in proxy.OUR_NAMES}


# ---------------------------------------------------------------------------


def test_the_fixture_is_not_the_real_store():
    check(os.path.abspath(nebari.DB).startswith(os.path.abspath(_TMP)),
          "session memory is a temp file", nebari.DB)
    check(os.path.abspath(nebari.DB) != REAL_NEBARI,
          "and is not index/nebari.sqlite3")
    check(domains.PACKAGE_STORE is None,
          "the module default is untouched; stores are passed per call")
    check(os.path.abspath(HELD).startswith(os.path.abspath(_TMP)),
          "the fixture package store is a temp directory", HELD)


def test_held_means_alive_not_present():
    s = store_with(("three", "0.185.1", 5), ("dead", "1.0.0", 0),
                   ("@react-three/fiber", "9.1.0", 1))
    with open(os.path.join(s, "garbage@1.0.0.sqlite3"), "wb") as f:
        f.write(b"not a database")
    held = domains.held_sources(s)
    check("three" in held, "a package with chunks is held", json.dumps(list(held)))
    check("dead" not in held,
          "a package index with ZERO chunks is not held (PROTOCOL rule 1)")
    check("garbage" not in held, "an unreadable file is not held, and does not raise")
    check("@react-three/fiber" in held,
          "a scoped package's slug is read back as its install name",
          json.dumps(list(held)))
    check(domains.held_sources(os.path.join(_TMP, "no_such_dir")) == {},
          "a missing store is an empty store, not an exception")
    readme = store_with(("fiber-readme-only", "10.0.0", 3, None))
    check("fiber-readme-only" not in domains.held_sources(readme),
          "an index with chunks but NO definitions is not held -- the "
          "dist-only README index that offered tools which found nothing")

    make_package(s, "typegpu", "0.12.5", 1)
    check("typegpu" in domains.held_sources(s),
          "a package indexed after the first call is seen on the next, "
          "with no restart")


def test_a_bound_repository_is_never_second_guessed():
    d = decide(LCB_PROMPT, root=os.path.join("C:", "some", "repo"))
    check(d["offer"] and d["situation"] == "REPOSITORY_BOUND",
          "a bound repository is offered even for a puzzle", d["situation"])
    d = decide(LCB_PROMPT, store=EMPTY, root=os.path.join("C:", "some", "repo"))
    check(d["offer"], "and even when no package is held", d["situation"])


def test_nothing_held_is_withheld_and_not_retryable():
    d = decide("how does the KV pool get sized?", store=EMPTY)
    check(d["offer"] is False and d["situation"] == "NOTHING_HELD",
          "no repository and no live package index: withheld", d["situation"])
    check(d.get("retryable") is False,
          "and it is NOT retryable by the caller -- no wording changes it")
    owners = {r.get("fixable_by") for r in d["remedies"]}
    check("operator" in owners, "a remedy names the operator", json.dumps(owners, default=list))
    check(all(r.get("action") and r.get("effect") for r in d["remedies"]),
          "every remedy says what to do and what it changes")


def test_each_route_to_offering():
    d = decide("why is this slow?", discovered=["three"])
    check(d["offer"] and d["situation"] == "IMPORTS_HELD_SOURCE",
          "the conversation imports a held package", d["situation"])

    for text in ("How do I use three.js with WebGPU?",
                 "In TSL, what replaced label()?",
                 "import tgpu from 'typegpu' -- how do buffers work?"):
        d = decide(text)
        check(d["offer"] and d["situation"] in ("NAMES_HELD_SOURCE",
                                                "IMPORTS_HELD_SOURCE"),
              f"names a held package: {text[:32]!r}", d["situation"])

    d = decide("What is the default value of Object3D.DEFAULT_UP?")
    check(d["offer"] and d["situation"] == "DEFINES_MENTIONED_SYMBOL",
          "a code-shaped name the held package DEFINES", d["situation"])
    check("Object3D" in json.dumps(d["evidence"]),
          "and the evidence names the symbol", json.dumps(d["evidence"])[:160])

    d = decide("Where is painterSortStable called from?")
    check(d["offer"], "camelCase symbol from the index", d["situation"])

    d = decide("Why does my fragment shader band?")
    check(d["offer"] and d["situation"] == "DOMAIN_MATCHES_HELD_SOURCE",
          "a gpu word meets a held package's domain", d["situation"])

    d = decide("hi")
    check(d["offer"] and d["situation"] == "NO_DOMAIN_EVIDENCE",
          "no evidence at all is offered: nothing is inferred from an absence",
          d["situation"])

    d = decide(LCB_PROMPT, offered_before=True)
    check(d["offer"] and d["situation"] == "OFFERED_EARLIER_THIS_SESSION",
          "a conversation that already had the tools keeps them", d["situation"])


def test_plain_words_are_not_symbols_and_numerals_are_not_names():
    # `sort` is defined in the fixture three, as it is in the real one.
    d = decide("Write a program to sort the input and print the graph.")
    check(d["offer"] is False,
          "a plain word the package happens to define is not evidence",
          d["situation"] + " " + json.dumps(d["evidence"])[:120])
    d = decide("Explain the graph algorithm in three paragraphs.")
    check(d["offer"] is False and "names" not in d["evidence"],
          "the numeral 'three' is not three.js (a real logged request)",
          d["situation"])


def test_english_words_are_not_symbols_even_where_defined():
    """Live 2026-09-23: `held=three:MOUSE;wgpu-matrix:API` on a pasted game
    spec. A package that DEFINES a plain English word is not named by it."""
    s = store_with(("three", "0.185.1", 2,
                    ("MOUSE", "Event", "POINT", "Object3D", "WeakMap")),
                   ("wgpu-matrix", "3.4.2", 1, ("API",)))
    text = ("MOUSE CONTROLS: the ship follows the MOUSE. Web Audio API sounds. "
            "Each Event is read once; every POINT popup fades. A WeakMap "
            "caches sprites.")
    got = domains._symbols(text, domains.held_sources(s))
    check(got == {}, "API, MOUSE, POINT, Event as prose, and a platform name "
          "(WeakMap), match nothing", json.dumps(got))
    got = domains._symbols("Where is Object3D.DEFAULT_UP set?",
                           domains.held_sources(s))
    check(got.get("three") == ["Object3D"],
          "a code-shaped name still does", json.dumps(got))
    names = domains.code_context_names(
        "THREE.MOUSE.LEFT, Loop(), class A extends Pipelines, the Web Audio "
        "API (see API docs), Node.js, Three.js, Array.from(xs), new Event('x')")
    check({"THREE", "MOUSE", "LEFT", "Loop", "Pipelines"} <= set(names)
          and not {"API", "Node", "Three", "Array", "Event"} & set(names),
          "code_context_names: identifier syntax yes; prose, product "
          "spellings and platform names no", json.dumps(names))


def test_a_package_is_not_named_by_an_ordinary_word():
    s = store_with(("postprocessing", "6.39.5", 1), ("three", "0.185.1", 1))
    held = domains.held_sources(s)
    check(domains._named("add a bloom postprocessing pass to my shader", held)
          == [], "the word 'postprocessing' does not name the pmndrs package",
          json.dumps(domains._named("a postprocessing pass", held)))
    for text in ("import { EffectComposer } from 'postprocessing'",
                 "pmndrs/postprocessing v6", "postprocessing@6.39.5"):
        check("postprocessing" in domains._named(text, held),
              f"but a module spelling does: {text!r}")
    # pmndrs/math (held 2026-09-27): the word, and Python's stdlib, do not
    # name it; it has a hand-mapped domain, since it imports nothing to
    # derive one from (an unmapped held package reopens the gate).
    s = store_with(("math", "0.1.0", 1))
    held = domains.held_sources(s)
    check(domains._named("Print the answer modulo 998244353 using math. "
                         "import math", held) == [],
          "the word 'math' does not name the pmndrs package")
    check(all("math" in domains._named(t, held) for t in
              ("pmndrs/math", "from 'math/noise'", "math@0.1.0")),
          "but its repo, a subpath and a version pin do")
    check(domains.PACKAGE_DOMAINS.get("math") == {"gpu", "web-frontend"},
          "and it is mapped by hand to skill_classify's pmndrs_math domains")


def test_a_word_counts_only_in_its_domain_sense():
    """bench/domain/tasks/rs12, a Rust C-ABI point parser, was classified
    `visual-design` on "after trimming ASCII whitespace"."""
    rs12 = os.path.join(REPO, "bench", "domain", "tasks", "rs12", "task.json")
    if os.path.exists(rs12):
        with open(rs12, encoding="utf-8") as fh:
            prompt = json.load(fh)["prompt"]
        got = domains.detect(user(prompt))
        check("visual-design" not in got,
              "the real rs12 prompt is not visual-design", json.dumps(sorted(got)))
    else:
        _skipped.append("rs12 replay: " + rs12 + " missing")
    for text, want in (("trim leading whitespace from the token", False),
                       ("in contrast to the old parser, this one streams", False),
                       ("flatten the class hierarchy into one struct", False),
                       ("the card needs more whitespace between rows", True),
                       ("the visual hierarchy of the page is flat", True),
                       ("the text contrast is too low in dark mode", True),
                       ("pick a palette for the charts", True)):
        got = "visual-design" in domains.detect(user(text))
        check(got is want, f"{text!r}: visual-design is {want}",
              json.dumps(sorted(domains.detect(user(text)))))


def test_prepare_hands_the_clients_tools_to_selection():
    """A Hermes-shaped request through proxy.prepare at max: the client's
    own tools reach selection and the route, and main gets the client's
    tools untouched and first. (The deep-thinking / fan-out decision this
    once checked was removed 2026-09-29, with yama_think_deeply and
    yama_plan.)"""
    prev = domains.PACKAGE_STORE
    domains.PACKAGE_STORE = HELD
    try:
        b = {"model": "yamadori", "reasoning_effort": "max",
             "_client_ip": "127.0.0.1",
             "_features": '{"skills": false}',
             "messages": [
                 {"role": "system", "content": "You are Hermes Agent."},
                 {"role": "user", "content":
                  "@file:`.hermes/attachments/Pasted content (8.8 KB)`\n\n"
                  "I want to start this project in ~/Developer/octopus-invaders"
                  "\n\n--- Attached Context ---\n\n```\nbuild a three.js "
                  "shooter. MOUSE CONTROLS: Object3D per enemy.\n```"}],
             "tools": [{"type": "function", "function": {
                 "name": n, "description": "x", "parameters": {}}}
                 for n in ("write_file", "terminal", "find_by_meaning")]}
        out = proxy.prepare(b)
        sel = out["_selection"]
        check(sel["signals"]["client_tools"] == 2,
              "the client's tools reach selection, a name that was once ours "
              "(find_by_meaning, re-sent) excluded",
              json.dumps(sel["signals"].get("client_tools")))
        check(out["_route"]["class"] == "agent_step",
              "act locally with the client's tools: the client's agent loop",
              json.dumps(out["_route"])[:240])
        names = [t["function"]["name"] for t in out["tools"]]
        check(names[:3] == ["write_file", "terminal", "find_by_meaning"]
              and not {"yama_think_deeply", "yama_plan"} & set(names)
              and set(names[3:]) == set(out["_ours"]),
              "main gets the client's tools untouched and first -- its own "
              "find_by_meaning included -- and of ours only what is offered "
              "after them (no yama_think_deeply or yama_plan: removed)",
              json.dumps({"tools": names, "ours": out["_ours"]}))
    finally:
        domains.PACKAGE_STORE = prev


def test_an_unmapped_held_package_turns_domain_off():
    s = store_with(("three", "0.185.1", 1), ("zod", "3.23.8", 1))
    d = decide(LCB_PROMPT, store=s)
    check(d["offer"] and d["situation"] == "HELD_SOURCE_UNMAPPED",
          "a held package with no domain mapping cannot be ruled out",
          d["situation"])
    check("zod" in d["because"], "and the reason names it", d["because"])


def _with_sources(store: str, name: str, version: str,
                  files: dict[str, str]) -> None:
    """Give a fixture package its indexed file list and source files, in the
    store's _src cache, as deps.index_package leaves them."""
    import deps
    stem = deps.slug(name, version)
    db = os.path.join(store, f"{stem}.sqlite3")
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE IF NOT EXISTS package_files(path TEXT PRIMARY KEY)")
    con.executemany("INSERT OR IGNORE INTO package_files VALUES(?)",
                    [(p,) for p in files])
    con.commit()
    con.close()
    for p, text in files.items():
        full = os.path.join(store, "_src", stem, p)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(text)


def test_a_new_package_is_mapped_from_its_own_imports():
    """#20 (docs/SELF-IMPROVEMENT-LOG.md): the Octopus pilot indexed
    @pmndrs/glyph and three-flatland, neither in PACKAGE_DOMAINS, and the
    gate offered library help to every request again (HELD_SOURCE_UNMAPPED,
    all 342 LiveCodeBench prompts). A held package's domains are now derived
    from its own imports: the hand-mapped domains of what it imports in at
    least DERIVE_MIN_FILES files."""
    s = store_with(("three", "0.185.1", 1), ("@acme/glyphs", "0.1.0", 1),
                   ("lonely", "1.0.0", 1))
    _with_sources(s, "@acme/glyphs", "0.1.0", {
        "src/text.ts": "import { Mesh } from 'three';\nexport class Text {}\n",
        "src/atlas.ts": "import * as THREE from 'three/webgpu';\n"
                        "import { local } from './text';\nexport const a = 1;\n",
        "src/util.ts": "import { thing } from 'some-unmapped-lib';\n"
                       "import { other } from 'some-unmapped-lib';\n"})
    _with_sources(s, "lonely", "1.0.0", {
        "index.js": "const t = require('three');\nmodule.exports = t;\n"})
    held = domains.held_sources(s)
    g = domains.derived_domains("@acme/glyphs", held["@acme/glyphs"][0][1])
    check(g["domains"] == ["gpu", "web-frontend"]
          and g["imports"] == {"three": 2},
          "a package importing three in 2 files derives three's domains; an "
          "unmapped import and a relative one add nothing", json.dumps(g))
    lone = domains.derived_domains("lonely", held["lonely"][0][1])
    check(lone["domains"] == [],
          f"one importing file is below DERIVE_MIN_FILES "
          f"({domains.DERIVE_MIN_FILES}): no domain", json.dumps(lone))
    d = decide(LCB_PROMPT, store=s)
    check(d["offer"] and d["situation"] == "HELD_SOURCE_UNMAPPED"
          and "lonely" in d["because"] and "@acme/glyphs" not in d["because"],
          "the gate counts the derived package as mapped; only the one that "
          "derives nothing is still unmapped", d["because"])
    s2 = store_with(("three", "0.185.1", 1), ("@acme/glyphs", "0.1.0", 1))
    _with_sources(s2, "@acme/glyphs", "0.1.0", {
        "src/text.ts": "import { Mesh } from 'three';\n",
        "src/atlas.ts": "import { Scene } from 'three';\n"})
    d2 = decide(LCB_PROMPT, store=s2)
    check(not d2["offer"] and d2["situation"] == "DOMAIN_OUTSIDE_HELD_SOURCES"
          and (d2.get("evidence") or {}).get("derived", {}).get(
              "@acme/glyphs", {}).get("domains") == ["gpu", "web-frontend"],
          "with every held package mapped or derived, a puzzle is withheld "
          "again, and the record shows the derivation",
          json.dumps({"situation": d2["situation"],
                      "derived": (d2.get("evidence") or {}).get("derived")}))


def test_a_puzzle_is_withheld_with_a_structured_reason():
    d = decide(LCB_PROMPT)
    check(d["offer"] is False and d["situation"] == "DOMAIN_OUTSIDE_HELD_SOURCES",
          "a competitive-programming prompt is withheld", d["situation"])
    check("algorithms" in d["evidence"].get("domains", []),
          "for the right reason: it is classified as algorithms",
          json.dumps(d["evidence"].get("domains")))
    check("gpu" not in d["evidence"].get("domains", []),
          "and 'vertices' in a graph problem is not GPU evidence")
    check(d.get("retryable") is True,
          "retryable is a fact: a later turn naming a held library changes it")
    owners = {r.get("fixable_by") for r in d["remedies"]}
    check(owners == {"user", "operator"}, "remedies carry owners",
          json.dumps(sorted(owners)))
    text = json.dumps(d).lower()
    check("different tool" not in text and "try " not in text,
          "and nothing in it says 'try a different tool'")


def test_boilerplate_cannot_withhold_but_can_offer():
    harness = ("You are an agent. Use the server tools. Read the input, check "
               "the database schema, and keep the index up to date.")
    d = decide("hi", system=harness)
    check(d["offer"] and d["situation"] == "NO_DOMAIN_EVIDENCE",
          "a harness prompt's words do not withhold a task with no evidence",
          d["situation"] + " " + json.dumps(d["evidence"].get("domains")))
    d = decide("what does this do?", system="This is a three.js project.")
    check(d["offer"] and d["situation"] == "NAMES_HELD_SOURCE",
          "a harness prompt naming a held library does offer", d["situation"])


def test_the_word_stems_now_match_their_words():
    for word, dom in (("animation", "motion"), ("typography", "visual-design"),
                      ("accessibility", "accessibility"), ("algorithms", "algorithms")):
        got = domains.detect(user(f"a question about {word}"))
        check(dom in got, f"{word!r} is {dom}", json.dumps(sorted(got)))
    got = domains.detect(user("count the vertices of the graph"))
    check("gpu" not in got, "bare 'vertices' is graph theory, not GPU",
          json.dumps(sorted(got)))
    got = domains.detect(user("my vertex shader is slow"))
    check("gpu" in got, "'vertex shader' is GPU", json.dumps(sorted(got)))


# GitHub issue #4 (2026-10-02): `@types/react` gave `ui-component` to the gate,
# and the bare words "input", "component" and "button" then matched
# LiveCodeBench puzzles. UI questions with no import line, and puzzle / graph /
# shader sentences that carry the same words. (The positives were written
# beside the word list: in-sample.)
UI_PROSE = [
    "The modal doesn't close when I press Escape, how do I fix it?",
    "Make this button feel nicer when it is hovered",
    "My dropdown menu is cut off by its parent container",
    "How do I add a tooltip to an icon?",
    "The navbar collapses wrongly on mobile",
    "How should I validate a text input as the user types?",
    "The input field loses focus after every keystroke",
    "Write a reusable card component with a header and a footer",
    "Why does this component re-render on every keystroke?",
    "Add a submit button that is disabled while the form is saving",
    "my <button onClick={save}> never fires",
    "How do I align a checkbox with its label?",
    "Build a sidebar with collapsible sections",
    "Create a form field with an inline error message",
    "The parent component passes props down but the child component never "
    "updates",
    "A popover that closes when you click outside of it",
    "My disabled button still shows a hover state",
    "Write a search input with a debounce",
    "I need an accessible accordion component",
    "Open the modal dialog from the settings page",
    "Style the primary button so it has rounded corners",
    "Show a toast when the save succeeds",
    "The button looks blurry on retina screens",
    "Center the modal window vertically",
]
NOT_UI_PROSE = [
    "Takahashi will press the button six times, each press gives a candy",
    "Count the number of connected components in the graph",
    "The input is given from Standard Input in the following format",
    "Multiply the 3-component vector component-wise",
    "Find the modal value of the array, the most frequent element",
    "Find the number of strongly connected components",
    "The first component of each pair is the key",
    "Each test case has an input string S of length N",
    "Return the index of the button that took the longest time to push",
    "Print the number of input lines that are palindromes",
    "A calculator displays a string, and you press a button b",
    "Components of the vector are summed; find the maximum component",
]


def test_ui_component_needs_a_ui_context():
    missed = [s for s in UI_PROSE
              if "ui-component" not in domains.detect(user(s))]
    check(not missed, f"all {len(UI_PROSE)} prose UI questions with no import "
          "are ui-component", "; ".join(missed[:3]))
    wrong = [s for s in NOT_UI_PROSE
             if "ui-component" in domains.detect(user(s))]
    check(not wrong, f"none of {len(NOT_UI_PROSE)} puzzle / graph / shader "
          "sentences with the same words is ui-component", "; ".join(wrong[:3]))
    got = domains.detect(user(LCB_PROMPT))
    check("ui-component" not in got and "algorithms" in got,
          "the LiveCodeBench template and a graph question: algorithms, not UI",
          json.dumps(sorted(got)))


def test_the_gate_does_not_read_the_servers_own_index():
    import code_search as cs
    prev = cs.INDEX_DB
    try:
        results = []
        for db in (os.path.join(REPO, "index", "code.sqlite3"),
                   os.path.join(_TMP, "missing.sqlite3")):
            cs.INDEX_DB = db
            results.append(decide(LCB_PROMPT)["situation"])
        check(results[0] == results[1] == "DOMAIN_OUTSIDE_HELD_SOURCES",
              "the decision does not move with cs.INDEX_DB -- the old gate's "
              "bug was reading it", json.dumps(results))
    finally:
        cs.INDEX_DB = prev


def test_prepare_withholds_offers_and_keeps():
    prev = domains.PACKAGE_STORE
    domains.PACKAGE_STORE = HELD
    try:
        out, gate = prepare_gated(body(user(LCB_PROMPT)))
        # CHANGED 2026-09-24: no tool of ours reaches main for the gate;
        # it decides whether any held source can bear on the request, and
        # the route reads it (its signals record the situation).
        check(our_tool_names(out) == set(),
              "prepare(): a puzzle gets none of our tools on main",
              json.dumps(sorted(our_tool_names(out))))
        check(not [m for m in out["messages"] if m.get("role") == "system"],
              "and no system text added at medium")
        check((gate or {}).get("situation") == "DOMAIN_OUTSIDE_HELD_SOURCES"
              and not gate["offer"],
              "and prepare's gate withholds it",
              json.dumps(gate)[:200])
        client_tool = {"type": "function", "function": {
            "name": "read_file", "description": "x", "parameters": {}}}
        b = body(user(LCB_PROMPT))
        b["tools"] = [client_tool]
        out = proxy.prepare(b)
        check([t["function"]["name"] for t in out["tools"]]
              == ["read_file"],
              "a client's own tools pass through untouched",
              json.dumps([t["function"]["name"] for t in out["tools"]]))

        convo = user("In three.js, where is Object3D defined?")
        out, gate = prepare_gated(body(convo))
        names = our_tool_names(out)
        check(gate["offer"] and gate["situation"] == "NAMES_HELD_SOURCE"
              and out["_route"]["signals"].get("gate") == "NAMES_HELD_SOURCE"
              and not names,
              "prepare(): a three.js question: the gate offers, and no tool "
              "of ours goes to main for it", json.dumps(sorted(names)))
        check(out["_route"]["class"] == "library_question",
              "and it is routed as a library question",
              json.dumps(out["_route"])[:200])

        # The flip this rule exists for: a first turn with no evidence is
        # offered (nothing inferred from an absence), and a later turn adds
        # evidence that on its own would withhold. Withdrawing the tools then
        # would change the prefix mid-session. The session key is derived
        # from the first two messages, so this is the same session -- WITH a
        # system message. Without one, `nebari.key_of` hashes (user) on turn 1
        # and (user, assistant) on turn 2, so the key changes once; that is a
        # property of nebari's key, found here, and noted in _remember_offered.
        opening = [{"role": "system", "content": "You are a helpful assistant."}]
        opening += user("hello, I have a question coming")
        out, gate = prepare_gated(body(opening))
        check(gate["situation"] == "NO_DOMAIN_EVIDENCE" and gate["offer"],
              "an opening with no evidence is offered", gate["situation"])
        # The opening is a new conversation (#41); its answer made no tool
        # call, so the proxy records the id with it when it delivers it
        # (_run_turn, THE ANSWER RECORD) -- done here by hand, since
        # prepare() alone delivers nothing.
        proxy._record_answer_session(out, opening, "Go ahead.", [])
        later = opening + [{"role": "assistant", "content": "Go ahead."},
                           {"role": "user", "content": LCB_PROMPT}]
        check(not domains.tool_admission(later, None, store=HELD)["offer"],
              "(the later conversation, judged cold, would be withheld)")
        out, gate = prepare_gated(body(later))
        check(gate["offer"]
              and gate["situation"] == "OFFERED_EARLIER_THIS_SESSION",
              "a session the gate offered keeps the offer on a later turn",
              (gate or {}).get("situation", ""))

        # A client's side call is not gated at all. (The retrieval-off tier
        # this once checked went with the `retrieval` flag, 2026-09-29.)
        side = user("<command>ls</command>\n\nRespond with exactly one word: "
                    "APPROVE, DENY, or ESCALATE")
        out, gate = prepare_gated(body(side))
        check(out["_route"]["class"] == "utility" and gate is None
              and not our_tool_names(out),
              "a side call never reaches the gate",
              json.dumps(out["_route"])[:200])
    finally:
        domains.PACKAGE_STORE = prev


def test_a_long_transcript_decides_quickly():
    # ~250 KB of agent transcript, no evidence of a held package, many
    # code-shaped tokens: the symbol probe is capped, the regexes are linear.
    chunk = ("def handle_request_42(ctx):\n    fooBar = parse_input(ctx)\n"
             "    return sortedItems(fooBar)\n") * 3000
    msgs = [{"role": "user", "content": chunk}]
    t0 = time.time()
    d = domains.tool_admission(msgs, None, store=HELD)
    ms = (time.time() - t0) * 1000
    print(f"    ({len(chunk):,} chars decided in {ms:.0f} ms: {d['situation']})")
    check(ms < 5000, "a 250 KB transcript decides in under 5 s", f"{ms:.0f} ms")


def test_real_prompts_against_the_real_store():
    """PROTOCOL rule 7: input captured from the real producers."""
    lcb = [os.path.join(REPO, "bench", "data", f) for f in ("test6.jsonl", "test5.jsonl")]
    ce = os.path.join(REPO, "bench", "context_economy_tasks.jsonl")
    missing = [p for p in lcb + [ce, REAL_STORE] if not os.path.exists(p)]
    if missing or not domains.held_sources(REAL_STORE):
        _skipped.append("test_real_prompts_against_the_real_store: missing "
                        + ", ".join(missing or [REAL_STORE + " (no live package)"]))
        return
    sys.path.insert(0, os.path.join(REPO, "bench"))
    import discover
    from livecodebench import PROMPT_FUNCTIONAL, PROMPT_STDIN

    def run(text):
        msgs = user(text)
        return domains.tool_admission(msgs, None, store=REAL_STORE,
                                      discovered=discover.scan(msgs)["packages"])

    n = offered = 0
    wrong = []
    for path in lcb:
        for line in open(path, encoding="utf-8"):
            row = json.loads(line)
            fn = bool((row.get("starter_code") or "").strip())
            p = (PROMPT_FUNCTIONAL.format(question=row["question_content"],
                                          starter=row["starter_code"])
                 if fn else PROMPT_STDIN.format(question=row["question_content"]))
            d = run(p)
            n += 1
            if d["offer"]:
                offered += 1
                wrong.append(f"{row['question_id']}:{d['situation']}")
    check(n > 0 and offered == 0,
          f"all {n} LiveCodeBench prompts are withheld", ", ".join(wrong[:6]))

    n = withheld = 0
    wrong = []
    for line in open(ce, encoding="utf-8"):
        row = json.loads(line)
        d = run(row["question"])
        n += 1
        if not d["offer"]:
            withheld += 1
            wrong.append(f"{row['id']}:{d['because'][:60]}")
    check(n > 0 and withheld == 0,
          f"all {n} hand-written three.js questions are offered", "; ".join(wrong[:4]))


def main() -> int:
    for fn in (test_the_fixture_is_not_the_real_store,
               test_held_means_alive_not_present,
               test_a_bound_repository_is_never_second_guessed,
               test_nothing_held_is_withheld_and_not_retryable,
               test_each_route_to_offering,
               test_plain_words_are_not_symbols_and_numerals_are_not_names,
               test_english_words_are_not_symbols_even_where_defined,
               test_a_package_is_not_named_by_an_ordinary_word,
               test_a_word_counts_only_in_its_domain_sense,
               test_prepare_hands_the_clients_tools_to_selection,
               test_an_unmapped_held_package_turns_domain_off,
               test_a_new_package_is_mapped_from_its_own_imports,
               test_a_puzzle_is_withheld_with_a_structured_reason,
               test_boilerplate_cannot_withhold_but_can_offer,
               test_the_word_stems_now_match_their_words,
               test_ui_component_needs_a_ui_context,
               test_the_gate_does_not_read_the_servers_own_index,
               test_prepare_withholds_offers_and_keeps,
               test_a_long_transcript_decides_quickly,
               test_real_prompts_against_the_real_store):
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
    for s in _skipped:
        print(f"  SKIPPED, NOT PASSED: {s}")
    print(f"  temp directory: {_TMP}")
    if passed < total:
        print("  The gate is wrong, not the harness.")
    return 0 if passed == total and not _skipped else 1


if __name__ == "__main__":
    sys.exit(main())
