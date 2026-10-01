#!/usr/bin/env python
"""The selection engine (mcp/selection.py), against real prompts. No GPU.

WHAT THIS IS GATING -- docs/SELECTION-BUILD.md section 4, steps 3-5

  1. ONE COPY OF THE REGEX. `rule_baseline` moved out of
     bench/laya_calibration.py (removed with Laya, 2026-09-29); its 89
     predictions, captured before the move
     (bench/data/rule_baseline_golden.json), are reproduced exactly, and no
     second definition exists anywhere. E1 is compared against it
     (mcp/e1.py).
  2. THE SYMBOL LOOKUP (defined_symbols, which mcp/route.py's
     library_question reads): code-shaped names against every held
     package, plain words only against a package the text names, and a
     pasted spec's English (API, MOUSE, POINT, Event) is never a held
     symbol -- against the real package store.
  3. AN AGENT HARNESS ACTING LOCALLY (2026-09-23): acts_locally reads the
     user's instruction, instruction_of splits it from Hermes's attachment.
  4. THE TIER BOUNDS, THE HEADER FORCES: skills are off at every tier
     (operator, 2026-09-29), a flag in X-Yamadori-Features forces them on or
     off, and nothing decided exceeds the tier. decide() returns {skills,
     because, signals} and records the route it was given.

REMOVED 2026-09-29 (docs/REMOVED.md; the way back is commit e360d37): the
deep-thinking and fan-out decisions, the LEGACY rule + Laya two-signal path
(its stub /route server and 2x2), and the trigger path -- their checks went
with them.

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
import traceback

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

_results: list[tuple[bool, str, str]] = []
_skipped: list[str] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


def tier(effort: str, header: str | None = None) -> dict:
    return tiers.resolve({"reasoning_effort": effort}, tiers.from_header(header))


def user(text: str) -> list[dict]:
    return [{"role": "user", "content": text}]


def test_the_held_out_labels_stay_out_of_training():
    """The route_in trainer (scripts/train_laya.py until 2026-09-29, now
    e1.gold_route_rows) reads every bench/laya_routing_labels*.jsonl. The
    held-out package labels were first written as
    laya_routing_labels_packages.jsonl -- inside the glob -- so the next
    retrain would have trained on the set used to judge it."""
    import e1
    name = os.path.basename(HELD_OUT)
    srcs = sorted({r["source"] for r in e1.gold_route_rows()})
    check(srcs and name not in srcs,
          "the held-out file is not a route_in training file",
          f"{name} vs {srcs}")


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

    # bench/laya_calibration.py and scripts/train_laya.py, which re-exported
    # and scored this copy, were removed with Laya (2026-09-29; commit
    # e360d37 has them).
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
    if not domains.held_sources(REAL_STORE):
        _skipped.append("real-store checks: " + REAL_STORE
                        + " (no live package)")
        return False
    return True


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
    dbs = selection.symbol_dbs()
    q = "What is the default value of `Object3D.DEFAULT_UP`?"
    found = selection.defined_symbols(q, dbs)
    check(selection.rule_baseline({"question": q}) == "answer_directly"
          and "Object3D" in found.get("three", []),
          "the regex says answer_directly; the symbol lookup finds the held "
          "definition the question is about", json.dumps(found)[:200])
    check(selection.defined_symbols(q, {}) == {},
          "with no symbol table to consult, nothing is found")
    code, words = selection.split_probe_tokens(
        "Where is it. For each item the renderer's Pipelines module and "
        "`label()` use REVISION and PMREMGenerator, into a Zero Array")
    check({"label", "PMREMGenerator"} <= set(code)
          and {"Pipelines"} <= set(words),
          "probe: backticks and internal capitals are code-shaped; a "
          "capitalised code noun is English-shaped",
          f"{code} / {words}")
    # CHANGED 2026-09-23: a bare ALLCAPS word is no longer probed. It was,
    # and "MOUSE" / "API" in a pasted spec matched three and wgpu-matrix.
    check(not {"Where", "For", "Zero", "Array", "REVISION"} & set(code + words),
          "sentence-initial words, a capitalised word not used as a code "
          "noun, and a bare ALLCAPS word are not probed", f"{code} / {words}")
    code, words = selection.split_probe_tokens(
        "What is `REVISION`, and what does THREE.MOUSE.LEFT map to?")
    plain = selection.backticked_plain_words(
        "What is `REVISION`, and what does THREE.MOUSE.LEFT map to?")
    check("REVISION" in code and {"MOUSE", "LEFT"} <= set(plain),
          "ALLCAPS in code context still counts: backticked (every table), "
          "in identifier syntax (the named package only)",
          f"{code} / {words} / {plain}")
    q = "Return true if nums can become a Zero Array after the Group step"
    found = selection.defined_symbols(q, dbs)
    check(not found,
          "prose capitals from a puzzle match nothing (typegpu defines Array, "
          "three defines Group)", json.dumps(found))


# ---------------------------------------------------------------------------
# THE HARNESS REPLAY (2026-09-23). Built from what Hermes actually sent:
# index/corpus.sqlite3 events 3396 and 3398 (system-prompt head, tool list,
# the first 2,000 characters of the user turn). The logged turn is cut at
# 2,000 characters, so the spec below is the captured head plus a short
# tail written in the same style carrying the words the live log matched
# (MOUSE, API) and two more of the same kind (POINT, Event).
# ---------------------------------------------------------------------------
HERMES_SYSTEM = (
    "You are Hermes Agent, built by Nous Research. Be direct: match the "
    "length of your reply to the weight of the ask. ... The `hermes-agent` "
    "skill has the actual commands and proven workflows -- load it with "
    "skill_view(name='hermes-agent') before configuring, modifying, or ...")
# Hermes's own tools from the logged list (ours removed, as proxy.prepare
# removes them before selection sees the list).
HERMES_TOOLS = ["browser_exec", "clarify", "delegate_task", "execute_code",
                "memory", "patch", "read_file", "search_files", "skill_view",
                "skills_list", "terminal", "web_extract", "web_search",
                "write_file"]
HERMES_SPEC = (
    "```\n"
    "build a space shooter game with vanilla JavaScript and canvas. no "
    "libraries. no frameworks. multi-file project structure.\n\n"
    "PROJECT STRUCTURE:\nspace-shooter/\n"
    "  index.html          -- entry point, canvas setup, script imports\n"
    "  js/\n    config.js         -- color palette, speeds, enemy stats\n"
    "    game.js           -- main game loop, state machine, collision "
    "detection, screen shake\n"
    "    player.js         -- ship rendering, mouse tracking with lerp\n"
    "    background.js     -- 4-layer parallax, a shader glow on the "
    "nebula layer, render targets for bloom\n"
    "    audio.js          -- Web Audio API procedural sounds (laser, "
    "explosions, boss music)\n\n"
    "CRITICAL IMPLEMENTATION DETAILS (do not skip these):\n\n"
    "CANVAS SETUP:\n- in game.js init(), explicitly set canvas.width = "
    "window.innerWidth\n- set ctx.imageSmoothingEnabled = false\n\n"
    "MOUSE CONTROLS:\n- ship follows the MOUSE with lerp; left click fires; "
    "each Event is read once per frame\n\n"
    "SCORING:\n- every POINT popup floats up and fades\n```")


def hermes_turn(instruction: str, attachment: str = HERMES_SPEC,
                label: str = "Pasted content (8.8 KB)") -> list[dict]:
    """A user turn in the exact layout event 3396 logged."""
    head = f"@file:`.hermes/attachments/{label}`\n\n{instruction}"
    body = (f"{head}\n\n--- Attached Context ---\n\n"
            f"\U0001F4C4 @file:`.hermes/attachments/{label}` (2246 tokens)\n"
            f"{attachment}")
    return [{"role": "system", "content": HERMES_SYSTEM},
            {"role": "user", "content": body}]


def test_an_agent_harness_acting_locally():
    import discover
    cases = [
        ("I want to start this project in ~/Developer/octopus-invaders", True),
        ("Let's create octopus-invaders here in this local documents folder "
         "in a subfolder called octopus-invaders.", True),
        ("set up a vite + three.js app in ./demo", True),
        ("Please scaffold a new Rust workspace", True),
        ("install dependencies in ./demo and run the dev server", True),
        ("How do I set up a vite + three.js app?", False),
        ("In ~/work/scene/main.ts, why does renderAsync warn?", False),
        ("Write a function that reads a file and returns its lines", False),
        ("Why does `npm run build` fail in ./demo?", False),
        ("Can you explain how to create a project with typegpu?", False),
    ]
    wrong = [f"{t!r} -> {selection.acts_locally(t)!r}" for t, want in cases
             if bool(selection.acts_locally(t)) is not want]
    check(not wrong, f"acts_locally: {len(cases)} requests and questions "
          "classified", "; ".join(wrong))

    whole = hermes_turn("I want to start this project in "
                        "~/Developer/octopus-invaders")[1]["content"]
    inst, att = selection.instruction_of(whole)
    check(inst.rstrip().endswith("octopus-invaders") and "MOUSE" in att
          and "MOUSE" not in inst,
          "instruction_of splits the user's words from Hermes's attachment",
          repr(inst[-60:]))
    check(selection.instruction_of("plain question")[1] == "",
          "a turn with no delimiter is read whole")

    if not _real_store_ready():
        return
    dbs = selection.symbol_dbs()

    def gate_of(msgs):
        return domains.tool_admission(
            msgs, None, discovered=discover.scan(msgs)["packages"])

    # THE LIVE CASE: the user's instruction acts locally, and no plain
    # English word from the paste is a held symbol.
    msgs = hermes_turn("I want to start this project in "
                       "~/Developer/octopus-invaders")
    inst = selection.instruction_of(msgs[1]["content"])[0]
    check(selection.acts_locally(inst),
          "the live Hermes request acts on the user's machine")
    # The gate's situation is mcp/test_domains.py's to gate: it moves with
    # the live store (an unmapped package held reads HELD_SOURCE_UNMAPPED).
    gate = gate_of(msgs)
    check(gate["offer"], "the gate offers (read-only, remote)",
          gate["situation"])
    held = json.dumps(selection.defined_symbols(msgs[1]["content"], dbs))
    check(not any(w in held for w in ("MOUSE", "API", "POINT", "Event")),
          "no plain English word from the paste is a held symbol", held)

    # The whole spec as the instruction (no delimiter): the paste's words
    # are read, and still none is a symbol, because none is in code context.
    code, words = selection.split_probe_tokens(HERMES_SPEC)
    plain = selection.backticked_plain_words(HERMES_SPEC)
    probed = set(code) | set(words) | set(plain)
    check(not {"MOUSE", "API", "POINT", "Event", "CANVAS", "SCORING"} & probed,
          "a pasted spec's ALLCAPS headings and English words are not probed",
          ", ".join(sorted(probed))[:200])
    gate_syms = json.dumps(domains._symbols(
        HERMES_SPEC, domains.held_sources(REAL_STORE)))
    check(not any(w in gate_syms for w in ("MOUSE", "API", "POINT", "Event")),
          "the tool gate's symbol probe does not match them either", gate_syms)

    # A scaffold request that NAMES a held library, no attachment at all.
    msgs = [{"role": "system", "content": HERMES_SYSTEM},
            {"role": "user", "content": "set up a vite + three.js app in ./demo"}]
    gate = gate_of(msgs)
    check(gate["offer"] and gate["situation"] == "NAMES_HELD_SOURCE",
          "'set up a vite + three.js app in ./demo': the gate reads the named "
          "library", gate["situation"])


LOOKUP = "Where is sizeKvPool defined?"


def test_the_tier_bounds_and_the_header_forces():
    d = selection.decide(user(LOOKUP), tier("max"),
                         route={"class": "library_question"})
    check(set(d) == {"skills", "because", "signals"}
          and set(d["because"]) == {"skills"},
          "decide returns {skills, because, signals}: skills is the one "
          "decision left", json.dumps(sorted(d)))
    check(d["signals"]["route"] == "library_question"
          and d["signals"]["allowed"] == {"skills": False},
          "the signals record the route and what the tier allows",
          json.dumps(d["signals"]))
    # skills are off at every tier (operator, 2026-09-29: "Stop skills
    # until we have a good skill injector."); a header forces them on
    check(all(selection.decide(user(LOOKUP), tier(e))["skills"] is False
              for e in tiers.ORDER)
          and selection.decide(user(LOOKUP),
                               tier("medium", '{"skills": true}'))["skills"]
          is True
          and selection.decide(user(LOOKUP),
                               tier("medium", '{"skills": false}'))["skills"]
          is False,
          "skills: off at every tier since 2026-09-29; the header forces them "
          "on (or off)")
    d = selection.decide(user(LOOKUP), tier("minimal", '{"skills": true}'))
    check(d["skills"] and "forced on" in d["because"]["skills"]
          and d["signals"]["forced"] == ["skills"],
          "a forced flag says so, even at minimal (the experiment arm)",
          d["because"]["skills"])
    d = selection.decide(user(LOOKUP), tier("max", '{"investigate": true, '
                                                   '"fanout": 3}'))
    check(set(d) == {"skills", "because", "signals"} and not d["skills"],
          "an old header naming removed features decides nothing more",
          json.dumps(d["because"]))
    line = selection.log_line(d)
    check(line.startswith("selection: skills=no"),
          "the one log line", line)

    worst = []
    for effort in tiers.ORDER:
        t = tier(effort)
        for q in ("How should I structure the state for this editor?", LOOKUP):
            d = selection.decide(user(q), t)
            if d["skills"] and not t["skills"]:
                worst.append(f"{effort}:{q[:20]}")
    check(not worst, "no decision exceeds its tier, any tier, any question",
          ", ".join(worst))


def test_question_of_reads_the_last_user_turn():
    msgs = [{"role": "system", "content": "sys"},
            {"role": "user", "content": "we are on three 0.185.0"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "what does fov set?"}]
    q, ctx, speaking = selection.question_of(msgs)
    check(q == "what does fov set?" and "0.185.0" in ctx and speaking,
          "question_of: the last user turn, earlier user text as context, "
          "and the user is speaking", repr((q, ctx, speaking)))
    convo = msgs + [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "tool_call_id": "c1", "content": "file contents"}]
    check(selection.question_of(convo)[2] is False,
          "a turn that continues the client's own tool loop is not the user "
          "speaking")


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
    check(len(rows) == 120 and three[1] == 120,
          "the 120 held-out package labels were scored (reported above)",
          str(len(rows)))


def main() -> int:
    for fn in (test_the_held_out_labels_stay_out_of_training,
               test_one_copy_of_the_regex,
               test_backticked_words_count_against_the_named_package_only,
               test_the_symbol_lookup_is_the_hard_slice_check,
               test_an_agent_harness_acting_locally,
               test_the_tier_bounds_and_the_header_forces,
               test_question_of_reads_the_last_user_turn,
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
    for s in _skipped:
        print(f"  SKIPPED {s}")
    passed = sum(1 for ok, _, _ in _results if ok)
    total = len(_results)
    print(f"\n{'=' * 70}\n  {passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
