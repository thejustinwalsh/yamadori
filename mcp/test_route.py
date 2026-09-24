#!/usr/bin/env python
"""The code-work router (mcp/route.py). No GPU, no service.

WHAT THIS IS GATING (operator, 2026-09-24)

  1. One class per request, from deterministic signals, each class reached
     by the shape the real producer sends (PROTOCOL rule 7: the fixtures
     below keep the captured requests' load-bearing words).
  2. utility and agent_step NEVER route to the code pipelines -- on the
     fixtures and on the labelled corpus set (bench/route/labels.jsonl),
     where the misroute count must be 0.
  3. The benchmark prompt sets (LiveCodeBench, bench/domain) route to
     code_generation / code_edit, every prompt.
  4. The features READ the class: fan-out on the code classes only, repair
     on the code classes only (proxy.prepare) -- and a header that forces
     one still forces it. Deep thinking no longer reads it (Phase 0.6,
     docs/SELF-IMPROVEMENT-PLAN.md, operator 2026-09-24): it runs when a
     trigger fires (mcp/deep.py), on any class; mcp/test_deep.py gates the
     triggers themselves.

The labelled-set and benchmark checks replay files that may be absent on a
fresh checkout; each says so when skipped. The labelled-set thresholds are
REGRESSION GUARDS on an in-sample measurement (bench/route/eval_route.py
explains why it is in-sample), not claims of accuracy.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "bench"))
sys.path.insert(0, os.path.join(REPO, "bench", "route"))

_TMP = tempfile.mkdtemp(prefix="yamadori_test_route_")
os.environ["YAMADORI_CORPUS_DB"] = os.path.join(_TMP, "corpus.sqlite3")
os.environ["YAMADORI_NEBARI_DB"] = os.path.join(_TMP, "nebari.sqlite3")
os.environ["RINGS_DB"] = os.path.join(_TMP, "rings.sqlite3")
os.environ["CODE_INDEX_DB"] = os.path.join(_TMP, "code.sqlite3")

import route  # noqa: E402
import selection  # noqa: E402
import tiers  # noqa: E402

tiers._accepted = ("low", "medium", "xhigh")

_results: list[tuple[bool, str, str]] = []
_skipped: list[str] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


HERMES_SYSTEM = ("You are Hermes Agent, built by Nous Research. Be direct: "
                 "match the length of your reply to the weight of the ask.")
HERMES_TOOLS = ["browser_exec", "clarify", "delegate_task", "execute_code",
                "memory", "patch", "read_file", "search_files", "skill_view",
                "skills_list", "terminal", "web_extract", "web_search",
                "write_file"]
REVIEWER = ("You are a security reviewer for an AI coding agent. You assess "
            "whether a flagged command is safe to run.")
OFFER_NONE = {"offer": True, "situation": "NO_DOMAIN_EVIDENCE"}
OFFER_THREE = {"offer": True, "situation": "NAMES_HELD_SOURCE"}
BOUND = {"offer": True, "situation": "REPOSITORY_BOUND"}


def user(text: str, system: str | None = None) -> list[dict]:
    return ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": text}]


def after_tool(text: str) -> list[dict]:
    """A Hermes turn mid-loop: the client ran a tool and sent its result."""
    return [{"role": "system", "content": HERMES_SYSTEM},
            {"role": "user", "content": text},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "terminal",
                              "arguments": '{"command": "ls"}'}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "index.html\n"}]


def mid_conversation(text: str) -> list[dict]:
    return [{"role": "system", "content": HERMES_SYSTEM},
            {"role": "user", "content": "build the game"},
            {"role": "assistant", "content": "Working on it."},
            {"role": "user", "content": text}]


def cls(msgs, client_tools=None, gate=None, response_format=None) -> dict:
    util = selection.utility_call(msgs, client_tools, response_format)
    return route.classify(msgs, client_tools=client_tools, util=util,
                          gate=gate, dbs={})


# --------------------------------------------------------------------------

CASES = [
    # (label, messages, client tools, gate, expected class)
    ("security reviewer (Hermes side call)",
     user("<command>ls -la</command>\n\nRespond with exactly one word: "
          "APPROVE, DENY, or ESCALATE", REVIEWER), None, None, "utility"),
    ("compaction summariser",
     user("You are a summarization agent creating a context checkpoint. Treat "
          "the conversation turns below as source material. Summarize the "
          "conversation."), None, None, "utility"),
    ("ends on a client tool result",
     after_tool("Let's create octopus-invaders here"), HERMES_TOOLS, None,
     "agent_step"),
    ("ends on a tool result, even asking for code",
     after_tool("Write a function that parses the level file"), HERMES_TOOLS,
     None, "agent_step"),
    ("harness notice: resume after a network cut",
     mid_conversation("[System: The previous response was cut off by a network "
                      "error mid-stream. Continue exactly where you left off.]"),
     HERMES_TOOLS, None, "agent_step"),
    ("a bare 'continue'", mid_conversation("continue"), HERMES_TOOLS, None,
     "agent_step"),
    ("act locally: start a project in ~/",
     user("I want to start this project in ~/Developer/octopus-invaders",
          HERMES_SYSTEM), HERMES_TOOLS, None, "agent_step"),
    ("act locally: push with gh (corpus 3653)",
     mid_conversation("lets push this to a github repo, I shoul be authed on "
                      "this machine and you can use the gh command to push "
                      "the code up to a new public repo. Does this work yet?"),
     HERMES_TOOLS, None, "agent_step"),
    ("SWE-agent task statement",
     user("<pr_description>\nCannot override get_FOO_display().\n"
          "</pr_description>\n\nYou're a software engineer interacting "
          "continuously with a computer by submitting commands.\nYour task is "
          "specifically to make changes to non-test files in the current "
          "directory in order to fix the issue described in the PR "
          "description.", "You are a helpful assistant that can interact "
          "with a computer."), ["bash"], None, "agent_step"),
    ("fenced code + fix",
     user("Fix the off-by-one in this:\n```ts\nfor (let i = 0; i <= xs.length; "
          "i++) { total += xs[i]; }\n```"), None, None, "code_edit"),
    ("starter code to complete (LiveCodeBench functional)",
     user("You will be given a competitive programming question.\nComplete "
          "the given function. Do not read from standard input.\n\n### "
          "Question\nGiven nums, return the sum.\n\n### Starter code\n"
          "```python\nclass Solution:\n    def total(self, nums: List[int]) -> "
          "int:\n        \n```"), None, None, "code_edit"),
    ("port given C to Rust",
     user("Port this C tagged value to Rust so both sides can share memory:\n\n"
          "```c\nstruct Value { uint32_t tag; double f; };\n```"), None, None,
     "code_edit"),
    ("write a function",
     user("Write a Python function that reverses a string. Code only."), None,
     None, "code_generation"),
    ("LiveCodeBench stdin",
     user("You will be given a competitive programming question.\nWrite a "
          "complete Python 3 program that reads from standard input and "
          "writes to\nstandard output.\n\n### Question\nPrint A+B."), None,
     None, "code_generation"),
    ("LiveBench placeholder fence is not code",
     user("### Instructions: You will be given a question and will generate a "
          "correct Python program.\n### Question: add two numbers\n### "
          "Format: Read the inputs from stdin.\n```python\n# YOUR CODE HERE\n"
          "```"), None, None, "code_generation"),
    ("export `name` (bench/domain typegpu)",
     user("Using typegpu 0.12.5 (`import tgpu from 'typegpu'`), export "
          "`GAUSS_WEIGHTS`: a WGSL module-scope constant holding five f32 "
          "weights."), None, None, "code_generation"),
    ("write + a language name",
     user("Write a Rust string interner for wasm32-unknown-unknown."), None,
     None, "code_generation"),
    ("where is X defined, repository bound",
     user("Where is the Trait type defined? Answer with the file path."), None,
     BOUND, "library_question"),
    ("a three.js question, three named",
     user("In one sentence, what does a three.js PerspectiveCamera's fov "
          "set?"), None, OFFER_THREE, "library_question"),
    ("a question ABOUT code is not a code request (context-economy ce12)",
     user("Where is the class `WebGPUBackend` defined, which class does it "
          "extend, and in which file is that base class defined?"), None,
     OFFER_THREE, "library_question"),
    ("'node code, the method' is not a code request (ce01)",
     user("In the TSL node code, the method `label()` was deprecated. What "
          "should be used instead?"), None, OFFER_THREE, "library_question"),
    ("a general question, gate offered for no reason about it",
     user("Explain quicksort in three paragraphs."), None, OFFER_NONE, "prose"),
    ("rewrite prose", user("Rewrite in simple technical English, return only "
                           "the rewrite: the buffer should be checked."),
     None, None, "prose"),
    ("a pasted spec in Hermes' attachment fence is prose, not code",
     user("What do you think of this plan?\n\n--- Attached Context ---\n\n"
          "```\nbuild a space shooter game with vanilla JavaScript and canvas."
          " no libraries.\nPLAYER: moves with arrow keys, fires with space.\n"
          "ENEMIES: descend in waves.\n```"), None, None, "prose"),
    ("'how do I write' is a question, not a request",
     user("How do I write a function that debounces calls?"), None, OFFER_NONE,
     "prose"),
    ("'can you write ...?' is a request",
     user("Can you write a function that debounces calls?"), None, None,
     "code_generation"),
    ("'make changes to non-test files' is not a request for tests",
     user("Please make changes to non-test files so the build passes, and "
          "tell me what you changed."), None, None, "prose"),
    # #14 (docs/SELF-IMPROVEMENT-LOG.md): the live gate's agent-loop task,
    # verbatim, with the client's own tools. It routed prose, both runs.
    ("a write verb + a source file is a code request (#14, live gate)",
     user("The project in the current directory has README.md, stats.py and "
          "test_stats.py. Read README.md and test_stats.py, then implement "
          "the task by writing the whole of stats.py with write_file. Work "
          "only through the tools. When stats.py is written, reply with one "
          "line saying it is done."), ["read_file", "write_file"], None,
     "code_generation"),
    ("a question about a source file is not a code request",
     user("What does utils.py do?"), None, None, "prose"),
    ("a write verb + a file that is not code is not a code request",
     user("Update README.md with the new install steps."), None, None,
     "prose"),
]


def test_each_class_on_real_shapes():
    for label, msgs, tools, gate, want in CASES:
        got = cls(msgs, tools, gate)
        check(got["class"] == want, f"{label}: {want}",
              f"got {got['class']}: {got['because'][:160]}")
    r = cls(after_tool("Write a function"), HERMES_TOOLS)
    check(set(r) == {"class", "because", "signals"} and
          r["signals"]["ends_on"] == "tool",
          "the record is {class, because, signals}, ends_on in signals",
          json.dumps(r)[:200])
    check(json.loads(json.dumps(r)) == r, "and it is JSON-safe (x_yamadori)")


def test_utility_and_agent_step_never_route_to_code():
    """Code-shaped words in every place they could leak from."""
    code_words = ("Write a Python function that parses JSON.\n```python\n"
                  "def f(:\n    pass\n```")
    bad = []
    for label, msgs, tools in (
            # the reviewer's real shape: the flagged command, then the
            # closed-form contract (corpus, every reviewer row)
            ("reviewer quoting code", user(
                "<command>\n" + code_words + "\n</command>\n\nRespond with "
                "exactly one word: APPROVE, DENY, or ESCALATE", REVIEWER),
             None),
            ("summariser over code", user(
                "You are a summarization agent creating a context "
                "checkpoint. Summarize the conversation.\n" + code_words),
             None),
            ("tool result after a code request", after_tool(code_words),
             HERMES_TOOLS),
            ("notice after code", mid_conversation(
                "[System: The previous response was cut off. Continue.]"),
             HERMES_TOOLS),
            ("act locally with a code noun", user(
                "Create a Python script in ~/tmp/x.py that prints hi",
                HERMES_SYSTEM), HERMES_TOOLS)):
        c = cls(msgs, tools)["class"]
        if c in route.CODE_CLASSES or c not in ("utility", "agent_step"):
            bad.append(f"{label} -> {c}")
    check(not bad, "utility / agent_step shapes never reach a code class",
          "; ".join(bad))


def test_the_signals_parse_rather_than_match():
    check(route.prompt_code("```\ndef f(x):\n    return x + 1\n```") == ["?"],
          "an untagged fence that parses as Python is code")
    check(route.prompt_code("```\nbuild a space shooter with canvas.\nno "
                            "libraries, no frameworks.\n```") == [],
          "an untagged fence of English is not")
    check(route.prompt_code("```python\n# YOUR CODE HERE\n```") == [],
          "a comment-only placeholder is not code")
    check(route.prompt_code("```text\nfoo(bar)\n```") == [],
          "a block tagged text is not code, whatever it contains")
    check(route.prompt_code("```wgsl\n@compute fn main() {}\n```") == ["wgsl"],
          "a language code_check cannot parse still counts by its tag")
    check(route.harness_notice("continue") == "continue"
          and route.harness_notice("[CONTEXT COMPACTION — REFERENCE ONLY] x")
          == "notice"
          and route.harness_notice("continue with the refactor, but why "
                                   "does it fail?") is None,
          "harness notices: continue, the bracketed Hermes notices; a "
          "question is not one")
    check(route.ends_on([{"role": "user"}, {"role": "function"}]) == "tool",
          "the legacy `function` role is a tool result")


def test_the_features_read_the_class():
    t = tiers.resolve({"reasoning_effort": "max"})
    msgs = user("Write a Python function that reverses a string.")
    code = cls(msgs)
    d = selection.decide(msgs, t, OFFER_NONE, dbs={}, route=code)
    check(d["fanout_n"] == 3 and "route code_generation" in d["because"]["fanout"],
          "code_generation fans out at max", d["because"]["fanout"])
    # Phase 0.6: no trigger, no deep thinking -- and the reason is no longer
    # the class ("library questions only" is gone).
    check(not d["investigate"] and "library questions only"
          not in d["because"]["investigate"]
          and "trigger" in d["because"]["investigate"],
          "and deep thinking stays out without a trigger, saying why",
          d["because"]["investigate"])
    design = user("How should we design the retry policy? Trade-offs?")
    d = selection.decide(design, t, OFFER_NONE, dbs={},
                         route=cls(design, gate=OFFER_NONE))
    check(d["fanout_n"] == 1 and "code_generation / code_edit only"
          in d["because"]["fanout"],
          "a design question (prose) no longer fans out", d["because"]["fanout"])
    d0 = selection.decide(design, t, OFFER_NONE, dbs={})
    check(d0["fanout_n"] == 3, "without a route the word rule is unchanged "
          "(every older caller)", str(d0["fanout_n"]))
    lib = user("Which retry policy does our proxy apply when the upstream "
               "returns a 503?")
    d = selection.decide(lib, t, BOUND, dbs={}, route=cls(lib, gate=BOUND))
    check(not d["investigate"] and d["signals"]["route"] == "library_question"
          and d["signals"].get("rule") is None,
          "library_question alone no longer runs deep thinking, and the "
          "rule + Laya path is not consulted (Phase 0.6)",
          d["because"]["investigate"])
    area = {"fire": True, "kind": "area", "job": "investigate",
            "question": "q", "because": "known-hard area: pkg@1.0.0-alpha.1"}
    d = selection.decide(lib, t, BOUND, dbs={}, route=cls(lib, gate=BOUND),
                         trigger=area)
    check(d["investigate"] and d["because"]["investigate"].startswith(
              "known-hard area") and d["signals"]["trigger"] == "area",
          "with a trigger that fired, it runs, and the reason is the "
          "trigger's", d["because"]["investigate"])
    step = after_tool("Write a function")
    d = selection.decide(step, t, OFFER_NONE, dbs={}, client_tools=HERMES_TOOLS,
                         route=cls(step, HERMES_TOOLS))
    check(d["fanout_n"] == 1 and not d["investigate"],
          "agent_step: no fan-out, no deep thinking without a trigger")
    struggle = {"fire": True, "kind": "struggle", "job": "investigate",
                "question": "q", "because": "struggle: 3 signals"}
    d = selection.decide(step, t, OFFER_NONE, dbs={}, client_tools=HERMES_TOOLS,
                         route=cls(step, HERMES_TOOLS), trigger=struggle)
    check(d["fanout_n"] == 1 and d["investigate"],
          "agent_step with a struggle trigger: deep thinking runs on the "
          "agent path (the gate that kept it to library questions is gone)",
          d["because"]["investigate"])
    off = tiers.resolve({"reasoning_effort": "max"},
                        tiers.from_header('{"investigate": false}'))
    d = selection.decide(step, off, OFFER_NONE, dbs={},
                         client_tools=HERMES_TOOLS,
                         route=cls(step, HERMES_TOOLS), trigger=struggle)
    check(not d["investigate"] and "forced off" in d["because"]["investigate"],
          "a header that forces deep thinking off beats a trigger",
          d["because"]["investigate"])
    high = tiers.resolve({"reasoning_effort": "high"})
    d = selection.decide(step, high, OFFER_NONE, dbs={},
                         client_tools=HERMES_TOOLS,
                         route=cls(step, HERMES_TOOLS), trigger=struggle)
    check(not d["investigate"], "and a tier that does not allow it (high) "
          "never runs it", d["because"]["investigate"])
    forced = tiers.resolve({"reasoning_effort": "max"},
                           tiers.from_header('{"investigate": true, '
                                             '"fanout": 3}'))
    d = selection.decide(msgs, forced, OFFER_NONE, dbs={}, route=code)
    check(d["investigate"] and d["fanout_n"] == 3,
          "a header still forces deep thinking on a code request")


def test_prepare_records_the_route_and_gates_repair():
    import proxy
    proxy.PREAMBLE = False

    def prep(text, effort="high", header=None, tools=None, msgs=None):
        body = {"model": "yamadori", "reasoning_effort": effort,
                "messages": msgs or user(text), "_client_ip": "127.0.0.1",
                "_features": json.dumps(header or {"hints": False,
                                                   "investigate": False})}
        if tools:
            body["tools"] = tools
        return proxy.prepare(body)
    out = prep("Write a Python function that reverses a string.")
    check(out["_route"]["class"] == "code_generation" and out["_repair"],
          "prepare: a code request at high gets the route and repair",
          json.dumps(out["_route"])[:160])
    out = prep("Explain quicksort in three paragraphs.")
    check(out["_route"]["class"] == "prose" and out["_repair"] is False,
          "a prose request at high: no repair pass", str(out["_repair"]))
    out = prep("Explain quicksort in three paragraphs.",
               header={"hints": False, "investigate": False, "repair": True})
    check(out["_repair"] is True, "a header that forces repair still forces it")
    out = prep("Write a function.", effort="medium")
    check(out["_repair"] is False and out["_fixup"] is False
          and out["_tool_code"] is True,
          "medium: client writes are checked and noted, nothing is repaired "
          "(operator, 2026-09-24)")
    out = prep("Write a Python function that reverses a string.")
    check(out["_fixup"] is True and out["_tool_code"] is True,
          "high: client writes are checked and repaired")
    wf = [{"type": "function", "function": {
        "name": "write_file", "parameters": {"type": "object"}}}]
    out = prep(None, msgs=after_tool("go"), tools=wf)
    check(out["_route"]["class"] == "agent_step" and out["_repair"] is False
          and out["_tool_code"] is True,
          "agent_step at high: no final-answer repair, the tool-call check on",
          f"{out['_route']['class']} {out['_repair']} {out['_tool_code']}")
    side = user("<command>ls</command>\n\nRespond with exactly one word: "
                "APPROVE, DENY, or ESCALATE", REVIEWER)
    out = prep(None, msgs=side)
    check(out["_route"]["class"] == "utility" and not out["_repair"]
          and not out["_tool_code"],
          "utility: no repair, no tool-call check")
    x = proxy._x_yamadori(out, hops=1, fan=None, think=None)
    check(x.get("route", {}).get("class") == "utility"
          and "tool_code" in x and x["tool_code"] is None,
          "x_yamadori carries route and tool_code", json.dumps(x.get("route"))[:120])


def test_the_labelled_corpus_set():
    labels = os.path.join(REPO, "bench", "route", "labels.jsonl")
    corpus = os.path.join(REPO, "index", "corpus.sqlite3")
    if not (os.path.exists(labels) and os.path.exists(corpus)):
        _skipped.append("labelled set: bench/route/labels.jsonl or "
                        "index/corpus.sqlite3 absent")
        return
    import eval_route
    rows = eval_route.replay()
    res = eval_route.matrix(rows)
    n_labels = sum(1 for _ in open(labels, encoding="utf-8"))
    check(res["n"] >= 1000 and n_labels >= 1000,
          f"the labelled set replays ({res['n']} of {n_labels} rows)")
    check(res["misroutes"] == 0,
          f"MISROUTES: utility / agent_step sent to the code pipelines: "
          f"{res['misroutes']}", str(res["misroute_ids"]))
    cp = res["code_pipelines"]
    check(cp["fn"] == 0 and cp["fp"] == 0,
          f"code vs not: {cp['tp']} code turns routed to code, {cp['fn']} "
          f"missed, {cp['fp']} non-code sent there")
    pc = res["per_class"]
    check(pc["utility"]["recall"] == 1.0 and pc["utility"]["precision"] == 1.0,
          "utility: every side call, nothing else", json.dumps(pc["utility"]))
    check(res["accuracy"] >= 0.95,
          f"six-way accuracy {res['accuracy']} (regression guard at 0.95, "
          f"in-sample)", json.dumps(pc))
    hermes = eval_route.matrix([r for r in rows if r["producer"] == "hermes"])
    check(hermes["misroutes"] == 0 and hermes["n"] >= 190,
          f"Hermes turns ({hermes['n']}): no misroute")
    sp = eval_route.second_pass(rows)
    check(sp["turns"] >= 200 and sp["agree"] / sp["turns"] >= 0.95,
          f"the blind second pass agrees on {sp['agree']}/{sp['turns']} turns")
    for c in route.CLASSES:
        p = pc[c]
        print(f"    {c:<17} n={p['n']:>4}  recall {p['recall']}  precision "
              f"{p['precision']}")


def test_the_benchmark_sets_route_to_code():
    import eval_route
    sets = eval_route.benchmark_prompts()
    if not sets:
        _skipped.append("benchmark prompt sets absent")
        return
    for name, b in eval_route.benchmarks().items():
        check(b["to_code"] == b["n"] and b["n"] > 0,
              f"{name}: {b['to_code']}/{b['n']} route to code",
              json.dumps(b["classes"]))


def main() -> int:
    for fn in (test_each_class_on_real_shapes,
               test_utility_and_agent_step_never_route_to_code,
               test_the_signals_parse_rather_than_match,
               test_the_features_read_the_class,
               test_prepare_records_the_route_and_gates_repair,
               test_the_labelled_corpus_set,
               test_the_benchmark_sets_route_to_code):
        print(f"\n--- {fn.__name__} ---")
        n0 = len(_results)
        try:
            fn()
        except Exception:                                        # noqa: BLE001
            check(False, f"{fn.__name__} itself raised",
                  traceback.format_exc().strip().split("\n")[-1])
            traceback.print_exc()
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
