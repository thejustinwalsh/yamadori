#!/usr/bin/env python
"""Read Atomic Agent's published GAIA L1 traces (the release bundle
`gaia-l1-eval.tar.gz`, tag gaia-l1-eval-2026-06-11) and count how often each
of its mechanisms fired, what the model did right after, and how firing sits
against solved vs failed tasks. Read-only: it opens the extracted NDJSON and
matrix files and runs nothing inside the bundle. No network, no GPU.

    python bench/atomic/gaia_traces.py BUNDLE_DIR [--out FILE.json]

BUNDLE_DIR is the extracted `gaia-l1-eval/` (reports/<run>/{environment.json,
matrix.jsonl, traces/<task>/<session>.ndjson}). docs/research/ATOMIC-AGENT.md
section 9 reports the output.

WHAT IS READ, per trace event type (Atomic's own trace schema):
  loop_detected     the loop detector: level (warn|critical|breaker) and
                    detector (generic_repeat|no_progress|wandering), tool,
                    count. Version 0.1.36 logs neither level nor detector
                    (reported as "unlabelled").
  tool_invocation   tool, args, status, summary (what the model was shown),
                    toolTruncated (the compressor cut the summary),
                    batchIndex/batchSize (the call was one of a parallel
                    batch); a vetoed call's details.deniedReason is
                    "tool-loop".
  prompt_captured   the prompt tail: a `### recalled` section is a memory
                    recall shown to the model.
  query_rewriter, reflection  the memory fabric's side calls.
  turn_finished     reason (reply | ...).

"WHAT THE MODEL DID NEXT" after a loop event: the tool calls of the next
step, compared with the flagged call (the event's tool in the event's step,
by tool + canonical arguments): `repeated` (the same call again), `changed`
(other calls), `ended` (no next step: the turn finished).
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import sys

RUNS = ("atomic-agent-L1", "atomic-agent-qwen3.5-9b-L1",
        "atomic-agent-gemma4-12b-L1")


def _jsonl(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except ValueError:
                    continue


def canon(tool, args) -> str:
    return tool + ":" + json.dumps(args, sort_keys=True, ensure_ascii=False)


def fisher(a, b, c, d) -> float:
    """Two-sided Fisher exact p for [[a, b], [c, d]]."""
    n = a + b + c + d
    r1, c1 = a + b, a + c

    def p(x):
        return (math.comb(r1, x) * math.comb(n - r1, c1 - x)
                / math.comb(n, c1))
    p0 = p(a)
    lo, hi = max(0, c1 - (n - r1)), min(r1, c1)
    return min(1.0, sum(p(x) for x in range(lo, hi + 1)
                        if p(x) <= p0 * (1 + 1e-9)))


def table(tasks: dict, flag) -> dict:
    """Solved/failed among tasks where `flag(task)` holds, and among the
    rest; Fisher's p on the 2x2."""
    a = sum(1 for t in tasks.values() if flag(t) and t["correct"])
    b = sum(1 for t in tasks.values() if flag(t) and not t["correct"])
    c = sum(1 for t in tasks.values() if not flag(t) and t["correct"])
    d = sum(1 for t in tasks.values() if not flag(t) and not t["correct"])
    return {"fired_solved": a, "fired_failed": b, "not_solved": c,
            "not_failed": d, "fisher_p": round(fisher(a, b, c, d), 3)}


def load_run(run_dir: str) -> dict:
    env = json.load(open(os.path.join(run_dir, "environment.json"),
                         encoding="utf-8"))
    matrix = {}
    for row in _jsonl(os.path.join(run_dir, "matrix.jsonl")):
        r = row.get("result") or {}
        matrix[r.get("taskId") or row["row"]["task_id"]] = r
    tasks = {}
    for tdir in sorted(glob.glob(os.path.join(run_dir, "traces", "*"))):
        tid = os.path.basename(tdir)
        evs = []
        for f in sorted(glob.glob(os.path.join(tdir, "*.ndjson"))):
            if os.path.basename(f).startswith("._"):
                continue                     # macOS resource forks
            evs.extend(_jsonl(f))
        evs.sort(key=lambda e: (e.get("sessionId", ""), e.get("seq", 0)))
        r = matrix.get(tid) or {}
        tasks[tid] = {"events": evs, "correct": bool(r.get("correct")),
                      "empty": not (r.get("extractedAnswer") or "").strip(),
                      "steps": (r.get("metrics") or {}).get("stepCount")}
    return {"env": env, "tasks": tasks, "matrix_n": len(matrix),
            "matrix_correct": sum(1 for r in matrix.values()
                                  if r.get("correct"))}


def analyse(run: dict) -> dict:
    tasks = run["tasks"]
    out: dict = {"tasks": len(tasks),
                 "correct": sum(1 for t in tasks.values() if t["correct"]),
                 "matrix": [run["matrix_correct"], run["matrix_n"]]}
    loops = collections.Counter()
    nxt = collections.defaultdict(collections.Counter)
    next_solved = collections.defaultdict(collections.Counter)
    trunc = collections.Counter()
    trunc_len = collections.defaultdict(list)
    after_trunc = collections.Counter()
    batch_steps = 0
    batch_sizes = collections.Counter()
    memory_tool_calls = 0
    recalled_prompts = 0
    qrw = collections.Counter()
    refl = collections.Counter()
    finished = collections.Counter()
    vetoed_calls = 0
    parse_retries = 0
    errors = 0
    calls = 0
    for tid, t in tasks.items():
        evs = t["events"]
        by_step = collections.defaultdict(list)
        steps_seen = set()
        t["fired"] = collections.Counter()
        t["trunc"] = 0
        t["batch"] = 0
        t["recall"] = 0
        for e in evs:
            ty = e.get("type")
            if ty == "tool_invocation":
                calls += 1
                by_step[e.get("stepIndex")].append(e)
                if (e.get("details") or {}).get("deniedReason") == \
                        "tool-loop":
                    vetoed_calls += 1
                if e.get("toolTruncated"):
                    trunc[e["tool"]] += 1
                    trunc_len[e["tool"]].append(len(e.get("summary") or ""))
                    t["trunc"] += 1
                if (e.get("batchSize") or 1) > 1 and \
                        e.get("batchIndex") == 0:
                    batch_steps += 1
                    batch_sizes[e["batchSize"]] += 1
                    t["batch"] += 1
                if str(e.get("tool", "")).startswith("memory."):
                    memory_tool_calls += 1
                    t["recall"] += 1
            elif ty == "step_started":
                steps_seen.add(e.get("stepIndex"))
            elif ty == "prompt_captured":
                if "### recalled" in (e.get("tail") or ""):
                    recalled_prompts += 1
                    t["recall"] += 1
            elif ty == "query_rewriter":
                qrw[e.get("outcome")] += 1
            elif ty == "reflection":
                refl[e.get("outcome")] += 1
            elif ty == "turn_finished":
                finished[e.get("reason")] += 1
            elif ty == "parse_retry":
                parse_retries += 1
            elif ty == "error":
                errors += 1
        # After a cut shell result: did the next step run the shell again?
        for s_i, cs in by_step.items():
            for c in cs:
                if c["tool"] == "os.shell.run" and c.get("toolTruncated"):
                    later = by_step.get(s_i + 1) or []
                    after_trunc["ended" if not later else
                                "shell_again" if any(
                                    x["tool"] == "os.shell.run"
                                    for x in later) else "other"] += 1
        for e in evs:
            if e.get("type") != "loop_detected":
                continue
            key = (e.get("level") or "unlabelled",
                   e.get("detector") or "unlabelled")
            loops[key] += 1
            t["fired"][key] += 1
            s = e.get("stepIndex")
            flagged = [c for c in by_step.get(s, []) if c["tool"] == e["tool"]]
            fsig = {canon(c["tool"], c.get("args")) for c in flagged}
            later = by_step.get(s + 1) if s is not None else None
            if not later:
                what = "ended"
            elif e["tool"] == "<batch>":
                what = "changed"
            elif any(canon(c["tool"], c.get("args")) in fsig for c in later):
                what = "repeated"
            else:
                what = "changed"
            nxt[key][what] += 1
            next_solved[key]["solved" if t["correct"] else "failed"] += 1
    out["calls"] = calls
    out["loop_events"] = {f"{a}/{b}": n for (a, b), n in loops.items()}
    out["after_loop_event"] = {f"{a}/{b}": dict(c) for (a, b), c in
                               nxt.items()}
    out["vetoed_calls"] = vetoed_calls
    kinds = sorted(loops)
    out["tasks_with"] = {}
    for k in kinds:
        out["tasks_with"][f"{k[0]}/{k[1]}"] = table(
            tasks, lambda t, k=k: t["fired"][k] > 0)
    out["tasks_with"]["any_loop_event"] = table(
        tasks, lambda t: sum(t["fired"].values()) > 0)
    out["truncated_results"] = {"n": sum(trunc.values()),
                                "by_tool": dict(trunc.most_common()),
                                "summary_chars": {
                                    k: {"n": len(v), "min": min(v),
                                        "median": sorted(v)[len(v) // 2],
                                        "max": max(v)}
                                    for k, v in trunc_len.items()},
                                "after_cut_shell": dict(after_trunc),
                                "tasks": table(tasks,
                                               lambda t: t["trunc"] > 0)}
    out["parallel_batches"] = {"steps": batch_steps,
                               "sizes": dict(sorted(batch_sizes.items())),
                               "tasks": table(tasks,
                                              lambda t: t["batch"] > 0)}
    out["memory"] = {"memory_tool_calls": memory_tool_calls,
                     "prompts_with_recalled": recalled_prompts,
                     "query_rewriter": dict(qrw),
                     "reflection": dict(refl)}
    out["turn_finished"] = dict(finished)
    out["parse_retries"] = parse_retries
    out["errors"] = errors
    out["empty_answers"] = sum(1 for t in tasks.values() if t["empty"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--out")
    a = ap.parse_args()
    res = {}
    for name in RUNS:
        d = os.path.join(a.bundle, "reports", name)
        if not os.path.isdir(d):
            continue
        run = load_run(d)
        env = run["env"]
        r = analyse(run)
        r["model"] = env["llama"]["chatModelId"]
        r["agent_version"] = env.get("agentVersion")
        r["git"] = env["git"]["shortSha"]
        r["sampling"] = env.get("sampling")
        res[name] = r
    h = os.path.join(a.bundle, "reports", "hermes-L1", "matrix.jsonl")
    if os.path.exists(h):
        rows = [row.get("result") or {} for row in _jsonl(h)]
        res["hermes-L1"] = {"tasks": len(rows),
                            "correct": sum(1 for r in rows if r.get("correct"))}
    print(json.dumps(res, indent=1))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
