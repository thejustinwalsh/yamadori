#!/usr/bin/env python
"""The selection engine (mcp/selection.py), against real prompts. No GPU.

WHAT THIS IS GATING -- docs/SELECTION-BUILD.md section 4, steps 3-5

  1. ONE COPY OF THE REGEX. `rule_baseline` moved out of
     bench/laya_calibration.py; its 89 predictions, captured before the move
     (bench/data/rule_baseline_golden.json), are reproduced exactly, and no
     second definition exists anywhere.
  2. REAL INPUT (PROTOCOL rule 7). Deep thinking is withheld on all 342
     LiveCodeBench prompts and fires on all 26 hand-written three.js
     questions in bench/context_economy_tasks.jsonl -- decided against the
     real package store, the way the proxy decides.
  3. TWO SIGNALS, DISAGREEMENT ESCALATES. The 2x2 of rule x Laya, through a
     stub /route over real HTTP; Laya down is None, never a guess.
  4. THE TIER BOUNDS, THE HEADER FORCES. Nothing fires above the tier; a
     flag in X-Yamadori-Features is forced on or off.

It also REPORTS (asserting only that it ran) the regex on the 120 held-out
package-domain labels in bench/laya_routing_heldout_packages.jsonl. Those
labels were written independently of the regex and are evaluation only: the
regex must not be tuned on them.
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "bench"))

_TMP = tempfile.mkdtemp(prefix="yamadori_test_selection_")
os.environ["YAMADORI_CORPUS_DB"] = os.path.join(_TMP, "corpus.sqlite3")
os.environ["LLAMA_STACK_URL"] = "http://127.0.0.1:1"      # reserved; refuses

import domains  # noqa: E402
import selection  # noqa: E402
import tiers  # noqa: E402

# The real store, read-only: the prompts are real, so the index they are
# judged against must be too. Nothing here writes to it.
import deps  # noqa: E402

REAL_STORE = deps.STORE
GOLDEN = os.path.join(REPO, "bench", "data", "rule_baseline_golden.json")
HELD_OUT = os.path.join(REPO, "bench", "laya_routing_heldout_packages.jsonl")
CE_TASKS = os.path.join(REPO, "bench", "context_economy_tasks.jsonl")
LCB = [os.path.join(REPO, "bench", "data", f) for f in ("test6.jsonl", "test5.jsonl")]

_results: list[tuple[bool, str, str]] = []
_skipped: list[str] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


def tier(effort: str, header: str | None = None) -> dict:
    return tiers.resolve({"reasoning_effort": effort}, tiers.from_header(header))


def user(text: str) -> list[dict]:
    return [{"role": "user", "content": text}]


OPEN = {"offer": True, "situation": "TEST", "because": "stub"}
# A bound repository: the one gate that says there is source to read for a
# question that names no held symbol ("our proxy's retry policy").
BOUND = {"offer": True, "situation": "REPOSITORY_BOUND", "because": "stub"}

# ---------------------------------------------------------------------------
# A stub Laya: a real HTTP server, so laya_signal's client runs unmodified.
# ---------------------------------------------------------------------------
ROUTE: dict[str, dict | int] = {}      # question -> reply, or an HTTP status
_route_seen: list[dict] = []


class _Laya(BaseHTTPRequestHandler):
    def do_POST(self):                                           # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _route_seen.append({"path": self.path, **body})
        r = ROUTE.get(body.get("question"), 500)
        code, data = (r, {"error": "stub"}) if isinstance(r, int) else (200, r)
        raw = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *a):                                   # noqa: D102
        pass


_srv = ThreadingHTTPServer(("127.0.0.1", 0), _Laya)
threading.Thread(target=_srv.serve_forever, daemon=True).start()
LAYA = f"http://127.0.0.1:{_srv.server_address[1]}"


def head(choice: str, margin: float = 0.6, abstain: bool = False,
         engine: str = "trained") -> dict:
    return {"type": "choice", "choice": choice, "margin": margin,
            "abstain": abstain, "engine": engine, "gate": 0.1,
            "probabilities": {choice: 0.8}, "task": "route_in"}


# ---------------------------------------------------------------------------


def test_the_held_out_labels_stay_out_of_training():
    """train_laya.py trains on every file matching its LABEL_GLOBS. The
    held-out package labels were first written as
    laya_routing_labels_packages.jsonl -- inside the glob -- so the next
    retrain would have trained on the set used to judge it."""
    import fnmatch
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    src = open(os.path.join(REPO, "scripts", "train_laya.py"),
               encoding="utf-8").read()
    import re as _re
    globs = _re.findall(r'"(laya_[a-z_]*\*\.jsonl)"', src)
    name = os.path.basename(HELD_OUT)
    check(globs and not any(fnmatch.fnmatch(name, g) for g in globs),
          "the held-out file matches no training glob", f"{name} vs {globs}")


def test_one_copy_of_the_regex():
    with open(GOLDEN, encoding="utf-8") as fh:
        gold = json.load(fh)
    items = gold["items"]
    check(len(items) == 89, "the golden file holds the 89 labelled questions",
          str(len(items)))
    wrong = [i["question"][:50] for i in items
             if selection.rule_baseline(i) != i["rule"]]
    check(not wrong, f"selection.rule_baseline reproduces all {len(items)} "
          "pre-move predictions", "; ".join(wrong[:3]))
    keys = ("domains", "libraries", "symbols", "code_block", "words",
            "context_words")
    drift = [i["question"][:50] for i in items
             if {k: selection.signals(i)[k] for k in keys}
             != {k: i["signals"][k] for k in keys}]
    check(not drift, "and the same cheap signals", "; ".join(drift[:3]))

    import laya_calibration as L
    check(L.rule_baseline is selection.rule_baseline
          and L.signals is selection.signals
          and L.index_facts is selection.index_facts,
          "bench/laya_calibration.py re-exports the one copy")
    with open(os.path.join(REPO, "scripts", "train_laya.py"),
              encoding="utf-8") as fh:
        src = fh.read()
    check("from selection import rule_baseline" in src,
          "scripts/train_laya.py scores the one copy")
    defs = []
    for sub in ("mcp", "bench", "scripts"):
        for dirpath, _dirs, files in os.walk(os.path.join(REPO, sub)):
            if "__pycache__" in dirpath or "_work" in dirpath:
                continue
            for fn in files:
                if fn.endswith(".py"):
                    p = os.path.join(dirpath, fn)
                    with open(p, encoding="utf-8", errors="replace") as fh:
                        if re.search(r"^def rule_baseline\(", fh.read(), re.M):
                            defs.append(os.path.relpath(p, REPO))
    check(defs == [os.path.join("mcp", "selection.py")],
          "exactly one `def rule_baseline` in the repo", str(defs))


def _real_store_ready() -> bool:
    missing = [p for p in LCB + [CE_TASKS] if not os.path.exists(p)]
    if missing or not domains.held_sources(REAL_STORE):
        _skipped.append("real-prompt replay: missing "
                        + ", ".join(missing or [REAL_STORE + " (no live package)"]))
        return False
    return True


def test_real_prompts():
    """PROTOCOL rule 7: the producers' own prompts, the real store."""
    if not _real_store_ready():
        return
    import discover
    from livecodebench import PROMPT_FUNCTIONAL, PROMPT_STDIN

    t = tier("max")
    dbs = selection.symbol_dbs()
    n = fired = fanned = 0
    wrong = []
    for path in LCB:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                fn = bool((row.get("starter_code") or "").strip())
                p = (PROMPT_FUNCTIONAL.format(question=row["question_content"],
                                              starter=row["starter_code"])
                     if fn else PROMPT_STDIN.format(question=row["question_content"]))
                msgs = user(p)
                gate = domains.tool_admission(
                    msgs, None, discovered=discover.scan(msgs)["packages"])
                d = selection.decide(msgs, t, gate, dbs=dbs,
                                     laya_status="not consulted: offline test")
                n += 1
                fired += d["investigate"]
                fanned += d["fanout_n"] > 1
                if d["investigate"]:
                    wrong.append(row["question_id"])
    check(n == 342 and fired == 0,
          f"deep thinking withheld on all {n} LiveCodeBench prompts at max",
          ", ".join(wrong[:6]))
    check(fanned == 0, "and none of them is fanned out", str(fanned))

    n = fired = 0
    wrong = []
    with open(CE_TASKS, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            msgs = user(row["question"])
            gate = domains.tool_admission(msgs, None)
            d = selection.decide(msgs, t, gate, dbs=dbs,
                                 laya_status="not consulted: offline test")
            n += 1
            fired += d["investigate"]
            if not d["investigate"]:
                wrong.append(f"{row['id']}: {d['because']['investigate'][:60]}")
    check(n == 26 and fired == 26,
          f"deep thinking fires on all {n} context-economy three.js questions "
          f"(rule alone, Laya absent)", "; ".join(wrong[:3]))


def test_backticked_words_count_against_the_named_package_only():
    """A backticked plain word is a real variable (`velocity`), so it counts --
    but only against a package the question names. Language keywords never
    count. Live case: tg01 backticked `import * as d from 'typegpu/data'` and
    `import`/`from` matched definitions while `data`/`velocity` matched
    three.js and wgpu-matrix, which the question never mentioned."""
    import sqlite3 as _sq
    store = tempfile.mkdtemp(prefix="yamadori_test_sel_pkgs_", dir=_TMP)

    def pkg(name: str, defs: list[str]) -> str:
        db = os.path.join(store, name.replace("/", "__") + ".sqlite3")
        con = _sq.connect(db)
        con.execute("CREATE TABLE defs(name TEXT, kind TEXT, path TEXT, "
                    "start INT, end INT, line TEXT)")
        for d in defs:
            con.execute("INSERT INTO defs VALUES(?,?,?,?,?,?)",
                        (d, "const", "src/x.js", 1, 1, d))
        con.commit()
        con.close()
        return db

    dbs = {"typegpu": pkg("typegpu", ["tgpu", "velocity", "import", "f32"]),
           "three": pkg("three", ["velocity", "data", "from"])}
    q = ("Using typegpu (`import * as d from 'typegpu/data'`), give the "
         "struct a `velocity: vec3f` field.")
    got = selection.defined_symbols(q, dbs)
    check("velocity" in got.get("typegpu", []),
          "a backticked plain word counts against the package the question "
          "names", json.dumps(got))
    check("three" not in got,
          "and never against a package it does not name", json.dumps(got))
    check("import" not in got.get("typegpu", [])
          and "from" not in json.dumps(got),
          "language keywords never count, even where a package defines them",
          json.dumps(got))
    got2 = selection.defined_symbols("give it a `velocity` field", dbs)
    check(got2 == {"typegpu": ["velocity"], "three": ["velocity"]},
          "a backticked NAME on its own still counts everywhere",
          json.dumps(got2))


def test_the_symbol_lookup_is_the_hard_slice_check():
    if not _real_store_ready():
        return
    t = tier("max")
    dbs = selection.symbol_dbs()
    q = "What is the default value of `Object3D.DEFAULT_UP`?"
    d = selection.decide(user(q), t, OPEN, dbs=dbs)
    check(d["signals"]["rule"] == "answer_directly" and d["investigate"],
          "the regex says answer_directly; a held definition upgrades it",
          d["because"]["investigate"])
    d = selection.decide(user(q), t, OPEN, dbs={})
    check(not d["investigate"],
          "with no symbol table to consult, the regex alone decides",
          d["because"]["investigate"])
    q = "Which is faster, instancedArray or the other one?"
    d = selection.decide(user(q), t, OPEN, dbs=dbs)
    check(d["signals"]["rule"] == "clarify" and not d["investigate"],
          "clarify is never upgraded: a class name does not specify a question",
          d["because"]["investigate"])
    code, words = selection.split_probe_tokens(
        "Where is it. For each item the renderer's Pipelines module and "
        "`label()` use REVISION and PMREMGenerator, into a Zero Array")
    check({"label", "PMREMGenerator"} <= set(code)
          and {"Pipelines", "REVISION"} <= set(words),
          "probe: backticks and internal capitals are code-shaped; a "
          "capitalised code noun and ALLCAPS are English-shaped",
          f"{code} / {words}")
    check(not {"Where", "For", "Zero", "Array"} & set(code + words),
          "sentence-initial words and a capitalised word that is not used "
          "as a code noun are not probed", f"{code} / {words}")
    q = "Return true if nums can become a Zero Array after the Group step"
    d = selection.decide(user(q), t, OPEN, dbs=dbs)
    check(not d["signals"]["held_symbols"],
          "prose capitals from a puzzle match nothing (typegpu defines Array, "
          "three defines Group)", str(d["signals"]["held_symbols"]))


RULE_YES = "Which retry policy does our proxy apply when the upstream returns a 503?"
RULE_NO = "What does the SOLID acronym stand for in object oriented design?"


def test_two_signals_and_disagreement_escalates():
    prev = domains.PACKAGE_STORE
    domains.PACKAGE_STORE = tempfile.mkdtemp(prefix="yamadori_sel_empty_")
    try:
        t = tier("max")
        check(selection.decide(user(RULE_YES), t, BOUND, dbs={})["signals"]["rule"]
              == "investigate"
              and selection.decide(user(RULE_NO), t, BOUND, dbs={})["signals"]["rule"]
              == "answer_directly",
              "the two fixture questions sit on opposite sides of the rule")
        matrix = [(RULE_YES, "investigate", True, "agree: investigate"),
                  (RULE_YES, "answer_directly", True, "rule yes, Laya no"),
                  (RULE_NO, "investigate", True, "rule no, Laya yes"),
                  (RULE_NO, "answer_directly", False, "agree: do not")]
        for q, laya_choice, want, label in matrix:
            ROUTE.clear()
            ROUTE[q] = head(laya_choice)
            _route_seen.clear()
            d = selection.select(user(q), t, BOUND, laya_url=LAYA)
            check(d["investigate"] is want,
                  f"2x2 {label} -> {'investigate' if want else 'continue'}",
                  d["because"]["investigate"])
            sig = d["signals"]
            check(sig["laya"] and sig["laya"]["choice"] == laya_choice
                  and sig["rule"] is not None,
                  f"   both signals recorded ({label})", json.dumps(sig)[:160])
        seen = _route_seen[-1] if _route_seen else {}
        check(seen.get("path") == "/route" and seen.get("task") == "route_in"
              and seen.get("engine") == "trained" and seen.get("question") == RULE_NO,
              "the call is POST /route, task route_in, engine trained",
              json.dumps(seen)[:200])
        line = selection.log_line(d)
        check("rule=answer_directly" in line and "laya=answer_directly" in line,
              "the one log line shows both signals", line)

        ROUTE.clear()
        ROUTE[RULE_NO] = head("answer_directly", margin=0.02, abstain=True)
        d = selection.select(user(RULE_NO), t, BOUND, laya_url=LAYA)
        check(d["investigate"] is True,
              "the head abstains: undecided is not agreement -> investigate",
              d["because"]["investigate"])

        for label, reply in (("HTTP 500", 500),
                             ("a zero-shot answer", head("answer_directly",
                                                         engine="zero_shot"))):
            ROUTE.clear()
            ROUTE[RULE_YES] = reply
            d = selection.select(user(RULE_YES), t, BOUND, laya_url=LAYA)
            check(d["signals"]["laya"] is None and d["investigate"] is True,
                  f"Laya returns {label}: signal None, the rule decides",
                  str(d["signals"]["laya_status"])[:160])
        d = selection.select(user(RULE_NO), t, BOUND,
                             laya_url="http://127.0.0.1:1")
        check(d["signals"]["laya"] is None
              and d["signals"]["laya_status"].startswith("down")
              and d["investigate"] is False,
              "Laya down: None, recorded as down, never a guess",
              d["signals"]["laya_status"][:120])
    finally:
        domains.PACKAGE_STORE = prev


def test_the_tier_bounds_and_the_header_forces():
    prev = domains.PACKAGE_STORE
    domains.PACKAGE_STORE = tempfile.mkdtemp(prefix="yamadori_sel_empty_")
    try:
        ROUTE.clear()
        _route_seen.clear()
        for effort in ("minimal", "low", "medium", "high"):
            d = selection.select(user(RULE_YES), tier(effort), BOUND, laya_url=LAYA)
            check(d["investigate"] is False,
                  f"{effort}: the rule says investigate, the tier does not allow it",
                  d["because"]["investigate"])
        check(not _route_seen, "and Laya is not even asked",
              str(len(_route_seen)))

        d = selection.select(user(RULE_YES), tier("max"),
                             {"offer": False, "situation": "DOMAIN_OUTSIDE_HELD_SOURCES"},
                             laya_url=LAYA)
        check(d["investigate"] is False and "withheld" in d["because"]["investigate"]
              and not _route_seen,
              "max with the tools withheld: no deep thinking, no Laya call",
              d["because"]["investigate"])

        d = selection.decide(user(RULE_YES), tier("max", '{"investigate": false}'), BOUND)
        check(d["investigate"] is False and "forced off" in d["because"]["investigate"],
              "the header forces it off at max", d["because"]["investigate"])
        d = selection.decide(user(RULE_NO), tier("minimal", '{"investigate": true}'), None)
        check(d["investigate"] is True and "forced on" in d["because"]["investigate"],
              "and on at minimal, with no gate at all (the experiment arm)",
              d["because"]["investigate"])

        convo = user(RULE_YES) + [
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
            {"role": "tool", "tool_call_id": "c1", "content": "file contents"}]
        d = selection.select(convo, tier("max"), BOUND, laya_url=LAYA)
        check(d["investigate"] is False and not _route_seen,
              "a turn that continues the client's own tool loop is not a new "
              "question", d["because"]["investigate"])

        design = "How should I structure the state for this editor, and why?"
        lookup = "Where is sizeKvPool defined?"
        check(selection.decide(user(design), tier("high"), OPEN)["fanout_n"] == 3,
              "high + a design question: fanned out to the tier's 3")
        check(selection.decide(user(lookup), tier("high"), OPEN)["fanout_n"] == 1,
              "high + a lookup: one answer")
        check(selection.decide(user(design), tier("medium"), OPEN)["fanout_n"] == 1,
              "medium + a design question: the tier allows one")
        check(selection.decide(user(lookup), tier("high", '{"fanout": 3}'),
                               OPEN)["fanout_n"] == 3,
              "the header forces fan-out on a lookup")
        check(selection.decide(user(design), tier("max", '{"fanout": 1}'),
                               OPEN)["fanout_n"] == 1,
              "and off on a design question")

        check(selection.decide(user(lookup), tier("low"), OPEN)["hints"] is False
              and selection.decide(user(lookup), tier("medium"), OPEN)["hints"] is True
              and selection.decide(user(lookup), tier("medium", '{"hints": false}'),
                                   OPEN)["hints"] is False,
              "hints: allowed from medium, forced off by the header")

        d = selection.select(user(RULE_YES), tier("max"), OPEN, laya_url=LAYA)
        check(d["investigate"] is False and "nothing to read" in
              d["because"]["investigate"] and not _route_seen,
              "tools offered for a reason unrelated to the question (no repo, "
              "no held symbol): nothing to read, no deep thinking, no Laya call",
              d["because"]["investigate"])

        worst = []
        for effort in tiers.ORDER:
            t = tier(effort)
            for q in (design, lookup, RULE_YES, RULE_NO):
                d = selection.decide(user(q), t, OPEN)
                if (d["investigate"] and not t["investigate"]) \
                        or d["fanout_n"] > t["fanout"] \
                        or (d["hints"] and not t["hints"]):
                    worst.append(f"{effort}:{q[:20]}")
        check(not worst, "no decision exceeds its tier, any tier, any question",
              ", ".join(worst))
    finally:
        domains.PACKAGE_STORE = prev


def test_held_out_package_labels_are_reported_not_tuned():
    """Evaluation only. The regex was not changed after reading these."""
    if not os.path.exists(HELD_OUT):
        _skipped.append("held-out labels: " + HELD_OUT + " missing")
        return
    rows = [json.loads(x) for x in open(HELD_OUT, encoding="utf-8") if x.strip()]
    hard = [r for r in rows if r.get("hard")]

    def acc(rs, pred):
        return sum(pred(r) for r in rs), len(rs)

    three = acc(rows, lambda r: selection.rule_baseline(r) == r["label"])
    three_h = acc(hard, lambda r: selection.rule_baseline(r) == r["label"])
    binary = acc(rows, lambda r: (selection.rule_baseline(r) == "investigate")
                 == (r["label"] == "investigate"))
    print(f"    regex, 3-way:  {three[0]}/{three[1]} overall, "
          f"{three_h[0]}/{three_h[1]} on the hard slice")
    print(f"    regex, investigate vs not:  {binary[0]}/{binary[1]}")
    if domains.held_sources(REAL_STORE):
        t = tier("max")
        dbs = selection.symbol_dbs()

        def sel(r):
            msgs = (user(r["context"]) if r.get("context") else []) + user(r["question"])
            return selection.decide(msgs, t, domains.tool_admission(msgs, None),
                                    dbs=dbs)["investigate"]
        eng = acc(rows, lambda r: sel(r) == (r["label"] == "investigate"))
        eng_h = acc(hard, lambda r: sel(r) == (r["label"] == "investigate"))
        print(f"    selection (rule + symbol lookup, Laya absent), investigate "
              f"vs not:  {eng[0]}/{eng[1]} overall, {eng_h[0]}/{eng_h[1]} hard")
    check(len(rows) == 120 and three[1] == 120,
          "the 120 held-out package labels were scored (reported above)",
          str(len(rows)))


def main() -> int:
    for fn in (test_the_held_out_labels_stay_out_of_training,
               test_one_copy_of_the_regex,
               test_real_prompts,
               test_backticked_words_count_against_the_named_package_only,
               test_the_symbol_lookup_is_the_hard_slice_check,
               test_two_signals_and_disagreement_escalates,
               test_the_tier_bounds_and_the_header_forces,
               test_held_out_package_labels_are_reported_not_tuned):
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
    _srv.shutdown()
    for s in _skipped:
        print(f"  SKIPPED {s}")
    passed = sum(1 for ok, _, _ in _results if ok)
    total = len(_results)
    print(f"\n{'=' * 70}\n  {passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
