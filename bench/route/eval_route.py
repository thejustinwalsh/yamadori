#!/usr/bin/env python
"""Score the code-work router (mcp/route.py) on the labelled corpus set and
the benchmark prompt sets. Offline: no model, no service; it reads
index/corpus.sqlite3 and the package store read-only.

    python bench/route/eval_route.py            # print the tables
    python bench/route/eval_route.py --write    # and bench/route/results.json

THE LABELLED SET: bench/route/labels.jsonl, one row per corpus turn (event
id, turn id, producer, whether the conversation ended on a client tool
result, the label, why, and the basis). No request text: the text stays in
the corpus, and this script joins on the event id. Written 2026-09-24,
BEFORE mcp/route.py existed, by reading every unique user-speaking request
against the rubric below.

  - 189 unique user-speaking requests were read and labelled by hand; the
    templated benchmark prompts (LiveCodeBench, LiveBench) by their template
    after reading examples of each ("rubric: template family").
  - 341 turns END ON A CLIENT TOOL RESULT and are agent_step by definition.
    The corpus does not record the last message's role, so that is
    RECONSTRUCTED: a turn whose system head and last user message repeat an
    earlier turn's, with more messages, is the client's loop coming back
    with a tool result (the reconstruction test_utility.py's empty-answer
    replay relies on; rows with a recorded `finish` confirm it -- the prior
    answer ended in client tool_calls). Scoring the router on those rows
    measures the reconstruction as much as the router: they are reported
    separately ("user-speaking only") as well as in the whole.
  - one turn is unlabelled: the corpus cut its request at 2,000 characters
    before the instruction.

SECOND PASS: a second labeller, blind to these labels and to the router,
labelled a seeded random 20% of the labelled turns (203 turns, 90 unique
requests) from the rubric alone: bench/route/second_pass.jsonl. Agreement
is printed below; the one disagreement is inside the code classes.

NOT 300 HERMES TURNS. The operator asked for at least 300 Hermes turns. The
corpus held 196 turns Hermes produced (its main loop, its approval
reviewer, title namer and compaction summariser) when this was labelled,
and all are labelled; the rest of the 1,016 is SWE-agent (200), benchmark
and probe traffic (612), and other agent harnesses (8). Per-producer
results are printed. None of the 196 is a code request at the request level:
Hermes wrote its code through write_file / patch calls on agent_step turns,
which is what mcp/tool_code.py checks.

IN-SAMPLE. The router's rules were adjusted while reading this set's misses
(the SWE-agent task lead, the hyphenated "non-test", comment-only
placeholder fences, the held-symbol lookup) and the benchmark sets' misses
(export `name`, "write a Rust ...", questions about code). Every number
here is therefore a training-set number. The out-of-sample check is the
corpus rows logged from the restart on, which record the route and the
role the conversation ends on (corpus.log_turn): label a sample of those.

THE RUBRIC. One class per request. Apply the rules in order; the first that
fits wins. Read the request as a person would: what is the model being asked
to do on THIS turn?

1. utility -- the client's own side call, not a task: an approval classifier
   ("You are a security reviewer ..."), a session-title namer ("You name chat
   sessions ..."), a conversation compaction / summarisation job ("You are a
   summarization agent creating a context checkpoint ..."). A benchmark's or a
   person's short probe ("Reply with the word ready", "say ok") is NOT
   utility: it is prose. (The same ground truth as mcp/test_utility.py.)

2. agent_step -- the turn belongs to the CLIENT's own agent loop, and the
   client sent tools of its own:
     a. the conversation ends on a client tool result (the model is mid-way
        through its own tool loop);
     b. the latest user turn carries no new task: a harness notice (a resume
        after a network cut, an interrupted-response notice, a compaction
        hand-off) or a bare "continue";
     c. the instruction asks the agent to act on, or inspect, the user's
        machine through its tools: create or scaffold a project, folder or
        file; run, install, build, push, commit; check what is committed or
        running. An agent-harness task framed as "issue bash commands to
        change the files in this directory" counts.

3. code_edit -- the prompt CARRIES code (fenced or pasted) and the task is to
   change, fix, port, complete or continue that code. Starter code to
   complete counts, and so does "write the remaining lines".

4. code_generation -- the task is to write NEW code in the answer (a
   function, program, module, type, struct, tests ...) from a description.
   Code in the prompt that is only an interface to implement against (a C
   header the Rust must match) still makes it code_generation.

5. library_question -- a question or a lookup, NOT a request to write code,
   about a specific library's or codebase's API, source or behaviour, or
   where something is defined, whose answer lives in source that is NOT in
   the prompt ("Where is the Trait type defined?", "what replaced label() in
   three r185?").

6. prose -- everything else: general knowledge, explanations not tied to a
   specific codebase, writing or rewriting text, arithmetic, chit-chat,
   image generation, readiness probes, and questions about code pasted in the
   prompt itself (the answer is in the prompt; nothing needs reading).

THE CODE PIPELINES are fan-out and the repair pass: they run on
code_generation / code_edit. A MISROUTE is a utility or agent_step turn the
router sends there. The operator's bar is zero.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "mcp"))
sys.path.insert(0, os.path.join(ROOT, "bench"))

LABELS = os.path.join(HERE, "labels.jsonl")
SECOND = os.path.join(HERE, "second_pass.jsonl")
CORPUS = os.path.join(ROOT, "index", "corpus.sqlite3")
RESULTS = os.path.join(HERE, "results.json")

CLASSES = ("utility", "agent_step", "code_edit", "code_generation",
           "library_question", "prose")
CODE = {"code_edit", "code_generation"}
NEVER_CODE = {"utility", "agent_step"}

# Our tool names, as the corpus logged them merged into tools_offered.
# proxy.OUR_NAMES is the source of truth; importing proxy here would start
# its module-level machinery, so the list is taken from it lazily.


def _our_names() -> set[str]:
    # Plus the names the proxy injected before 2026-09-24
    # (bind_project_context, check_code), which the corpus recorded too.
    import proxy
    return set(proxy.OUR_NAMES) | set(proxy._LEGACY_NAMES)


def load_labels() -> list[dict]:
    with open(LABELS, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def load_second() -> dict[int, dict]:
    if not os.path.exists(SECOND):
        return {}
    with open(SECOND, encoding="utf-8") as f:
        return {r["id"]: r for r in map(json.loads, f) if r}


def corpus_payloads(ids: list[int]) -> dict[int, dict]:
    con = sqlite3.connect(f"file:{CORPUS}?mode=ro", uri=True)
    try:
        out = {}
        for i, repo, p in con.execute(
                "SELECT id, repo, payload FROM events WHERE kind='turn'"):
            if i in ids:
                d = json.loads(p)
                d["_repo"] = repo
                out[i] = d
        return out
    finally:
        con.close()


def messages_of(p: dict, ends_on: str) -> list[dict]:
    """The request as the corpus can reconstruct it: the system head, a
    placeholder earlier exchange when it was not the first turn, the last
    user message, and -- when the turn ended on a client tool result -- the
    assistant's call and the result."""
    msgs = []
    if p.get("system_head"):
        msgs.append({"role": "system", "content": p["system_head"]})
    if not p.get("first_turn"):
        msgs += [{"role": "user", "content": "(earlier request)"},
                 {"role": "assistant", "content": "(earlier answer)"}]
    msgs.append({"role": "user", "content": p.get("request") or ""})
    if ends_on == "tool":
        msgs += [{"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "terminal", "arguments": "{}"}}]},
                 {"role": "tool", "tool_call_id": "c1", "content": "(result)"}]
    return msgs


def route_of(msgs: list[dict], client_tools: list[str], repo: str | None,
             response_format=None) -> dict:
    """What proxy.prepare computes: utility_call, the tool gate (the real
    package store, read-only), then the router."""
    import discover
    import domains
    import route
    import selection
    util = selection.utility_call(msgs, client_tools, response_format)
    gate = None
    if not util["utility"]:
        gate = domains.tool_admission(
            msgs, repo, discovered=discover.scan(msgs)["packages"])
    return route.classify(msgs, client_tools=client_tools, util=util,
                          gate=gate)


def replay() -> list[dict]:
    labels = [r for r in load_labels() if r.get("label")]
    ours = _our_names()
    pays = corpus_payloads([r["id"] for r in labels])
    rows = []
    for r in labels:
        p = pays.get(r["id"])
        if p is None:
            continue
        client = [n for n in p.get("tools_offered") or [] if n not in ours]
        rt = route_of(messages_of(p, r["ends_on"]), client, p.get("_repo"))
        rows.append(dict(r, pred=rt["class"], because=rt["because"]))
    return rows


def matrix(rows: list[dict]) -> dict:
    m = {t: Counter() for t in CLASSES}
    for r in rows:
        m[r["label"]][r["pred"]] += 1
    per = {}
    for c in CLASSES:
        tp = m[c][c]
        fn = sum(m[c].values()) - tp
        fp = sum(m[t][c] for t in CLASSES if t != c)
        per[c] = {"n": tp + fn, "predicted": tp + fp, "tp": tp,
                  "precision": round(tp / (tp + fp), 3) if tp + fp else None,
                  "recall": round(tp / (tp + fn), 3) if tp + fn else None}
    code_tp = sum(1 for r in rows if r["label"] in CODE and r["pred"] in CODE)
    code_fn = sum(1 for r in rows if r["label"] in CODE and r["pred"] not in CODE)
    code_fp = sum(1 for r in rows if r["label"] not in CODE and r["pred"] in CODE)
    misroutes = [r for r in rows if r["label"] in NEVER_CODE
                 and r["pred"] in CODE]
    return {"n": len(rows), "matrix": {t: dict(m[t]) for t in CLASSES},
            "per_class": per,
            "accuracy": round(sum(m[c][c] for c in CLASSES) / len(rows), 3)
            if rows else None,
            "code_pipelines": {"tp": code_tp, "fn": code_fn, "fp": code_fp},
            "misroutes": len(misroutes),
            "misroute_ids": [r["id"] for r in misroutes][:20]}


def print_matrix(title: str, res: dict) -> None:
    print(f"\n{title}  (n={res['n']}, accuracy {res['accuracy']})")
    short = {"utility": "util", "agent_step": "agent", "code_edit": "c_edit",
             "code_generation": "c_gen", "library_question": "libq",
             "prose": "prose"}
    print("  truth \\ routed   " + "".join(f"{short[c]:>8}" for c in CLASSES)
          + "   recall  precision")
    for t in CLASSES:
        row = res["matrix"][t]
        pc = res["per_class"][t]
        if not pc["n"] and not pc["predicted"]:
            continue
        print(f"  {t:<17}" + "".join(f"{row.get(c, 0):>8}" for c in CLASSES)
              + f"   {pc['recall'] if pc['recall'] is not None else '-':>6}"
              f"  {pc['precision'] if pc['precision'] is not None else '-':>9}")
    cp = res["code_pipelines"]
    print(f"  code pipelines: {cp['tp']} code turns routed there, {cp['fn']} "
          f"code turns missed, {cp['fp']} non-code turns sent there; "
          f"MISROUTES (utility/agent_step -> code): {res['misroutes']}")


def second_pass(rows: list[dict]) -> dict:
    sec = load_second()
    by_id = {r["id"]: r for r in rows}
    agree = dis = 0
    cases = []
    for i, s in sec.items():
        r = by_id.get(i)
        if not r:
            continue
        if s["second_label"] == r["label"]:
            agree += 1
        else:
            dis += 1
            cases.append({"id": i, "first": r["label"],
                          "second": s["second_label"]})
    return {"turns": agree + dis, "agree": agree, "disagree": dis,
            "cases": cases[:10]}

# ------------------------------------------------------ benchmark sets ----


def _single(prompt: str) -> list[dict]:
    return [{"role": "user", "content": prompt}]


def benchmark_prompts() -> dict[str, list[str]]:
    sets: dict[str, list[str]] = {}
    lcb = [os.path.join(ROOT, "bench", "data", f)
           for f in ("test6.jsonl", "test5.jsonl")]
    if all(os.path.exists(p) for p in lcb):
        from livecodebench import PROMPT_FUNCTIONAL, PROMPT_STDIN
        out = []
        for p in lcb:
            with open(p, encoding="utf-8") as fh:
                for line in fh:
                    row = json.loads(line)
                    fn = bool((row.get("starter_code") or "").strip())
                    out.append(PROMPT_FUNCTIONAL.format(
                        question=row["question_content"],
                        starter=row["starter_code"]) if fn else
                        PROMPT_STDIN.format(question=row["question_content"]))
        sets["livecodebench (test5+test6)"] = out
    for name in ("tasks.jsonl", "tasks_react.jsonl",
                 "tasks_type_challenges.jsonl"):
        p = os.path.join(ROOT, "bench", "domain", name)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                sets[f"bench/domain/{name}"] = [json.loads(x)["prompt"]
                                                for x in fh if x.strip()]
    return sets


def benchmarks() -> dict:
    out = {}
    for name, prompts in benchmark_prompts().items():
        got = Counter()
        wrong = []
        for i, pr in enumerate(prompts):
            c = route_of(_single(pr), [], None)["class"]
            got[c] += 1
            if c not in CODE:
                wrong.append(i)
        out[name] = {"n": len(prompts), "classes": dict(got),
                     "to_code": sum(v for k, v in got.items() if k in CODE),
                     "not_code_index": wrong[:10]}
    return out


def deep_thinking_effect() -> dict:
    """What `deep thinking on library_question only` changes, on the sets
    test_selection.py measures deep thinking on, at tier max, Laya absent:
    selection.decide without the route (the word rules) against with it."""
    import discover
    import domains
    import selection
    import tiers
    t = tiers.resolve({"reasoning_effort": "max"})
    dbs = selection.symbol_dbs()
    out = {}

    def both(msgs):
        gate = domains.tool_admission(
            msgs, None, discovered=discover.scan(msgs)["packages"])
        rt = route_of(msgs, [], None)
        a = selection.decide(msgs, t, gate, dbs=dbs)
        b = selection.decide(msgs, t, gate, dbs=dbs, route=rt)
        return a, b, rt
    ce = os.path.join(ROOT, "bench", "context_economy_tasks.jsonl")
    if os.path.exists(ce):
        on0 = on1 = 0
        for line in open(ce, encoding="utf-8"):
            a, b, _ = both(_single(json.loads(line)["question"]))
            on0 += a["investigate"]
            on1 += b["investigate"]
        out["context_economy (26 three.js questions)"] = {
            "investigate_without_route": on0, "investigate_with_route": on1}
    held = os.path.join(ROOT, "bench", "laya_routing_heldout_packages.jsonl")
    if os.path.exists(held):
        rows = [json.loads(x) for x in open(held, encoding="utf-8") if x.strip()]
        ok0 = ok1 = 0
        disc = Counter()
        for r in rows:
            msgs = ((_single(r["context"]) if r.get("context") else [])
                    + _single(r["question"]))
            a, b, rt = both(msgs)
            want = r["label"] == "investigate"
            c0, c1 = a["investigate"] == want, b["investigate"] == want
            ok0 += c0
            ok1 += c1
            if c0 != c1:
                disc[("route_right" if c1 else "route_wrong", rt["class"])] += 1
        out["held-out package labels (120)"] = {
            "correct_without_route": ok0, "correct_with_route": ok1,
            "discordant": {f"{k[0]}:{k[1]}": v for k, v in disc.items()}}
    for name, prompts in benchmark_prompts().items():
        if not name.startswith("bench/domain"):
            continue
        on0 = on1 = 0
        for pr in prompts:
            a, b, _ = both(_single(pr))
            on0 += a["investigate"]
            on1 += b["investigate"]
        out[name] = {"investigate_without_route": on0,
                     "investigate_with_route": on1}
    return out


def run(write: bool = False, verbose: bool = False) -> dict:
    rows = replay()
    res = {"all": matrix(rows),
           "user_speaking": matrix([r for r in rows if r["ends_on"] == "user"]),
           "hermes": matrix([r for r in rows if r["producer"] == "hermes"]),
           "unique_user_speaking": matrix(list({(r["key"], r["ends_on"]): r
                                                for r in rows
                                                if r["ends_on"] == "user"}
                                               .values())),
           "producers": dict(Counter(r["producer"] for r in rows)),
           "second_pass": second_pass(rows),
           "benchmarks": benchmarks()}
    if verbose:
        for r in rows:
            if r["pred"] != r["label"]:
                print(f"  miss id={r['id']} {r['label']} -> {r['pred']}: "
                      f"{r['because'][:140]}")
    if write:
        res["deep_thinking"] = deep_thinking_effect()
        with open(RESULTS, "w", encoding="utf-8", newline="\n") as f:
            json.dump(res, f, indent=1)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="also measure the deep-thinking effect and write "
                         "bench/route/results.json")
    ap.add_argument("-v", action="store_true", help="print every miss")
    a = ap.parse_args()
    if not os.path.exists(CORPUS):
        print("index/corpus.sqlite3 absent: nothing to replay")
        return 1
    res = run(write=a.write, verbose=a.v)
    print_matrix("ALL LABELLED TURNS", res["all"])
    print_matrix("USER-SPEAKING TURNS ONLY (the router reads text)",
                 res["user_speaking"])
    print_matrix("UNIQUE USER-SPEAKING REQUESTS", res["unique_user_speaking"])
    print_matrix("HERMES TURNS", res["hermes"])
    print(f"\nproducers: {res['producers']}")
    sp = res["second_pass"]
    print(f"second pass: {sp['agree']}/{sp['turns']} turns agree; "
          f"disagreements {sp['cases']}")
    print("\nbenchmark prompt sets (must route to code_generation / code_edit):")
    for name, b in res["benchmarks"].items():
        print(f"  {name}: {b['to_code']}/{b['n']} to code  {b['classes']}"
              + (f"  not code: {b['not_code_index']}" if b["not_code_index"]
                 else ""))
    if "deep_thinking" in res:
        print("\ndeep thinking, word rules vs route (tier max, Laya absent):")
        for k, v in res["deep_thinking"].items():
            print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
