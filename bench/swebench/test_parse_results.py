#!/usr/bin/env python
"""Proof that parse_results.py keeps every outcome apart, on a built fixture.

The fixture is a run directory with two arms laid out exactly as
mini-swe-agent 2.1.0 and the swebench 4.1.0 harness write them:

  bonsai/   three instances -- one resolved, one unresolved after a format
            error and a `length` finish, one whose agent run raised (empty
            patch, never evaluated by the harness)
  yamadori/ one instance, resolved, with x_yamadori on every response
            (hints, deep thinking that searched, a fan-out, internal hops)

    python bench/swebench/test_parse_results.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import arms  # noqa: E402
import parse_results as pr  # noqa: E402

CHECKS: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    CHECKS.append((name, bool(ok)))
    print(f"  {'ok  ' if ok else 'FAIL'} {name}")


def _asst(pt, ct, finish="tool_calls", x=None, hops=1):
    resp = {"choices": [{"finish_reason": finish, "message": {}}],
            "usage": {"prompt_tokens": pt, "completion_tokens": ct,
                      "hops": hops}}
    if x is not None:
        resp["x_yamadori"] = x
    return {"role": "assistant", "content": "THOUGHT", "extra": {"response": resp}}


def _traj(iid, exit_status, api_calls, msgs, **info):
    return {"instance_id": iid, "trajectory_format": "mini-swe-agent-1.1",
            "info": {"exit_status": exit_status, "submission": "",
                     "model_stats": {"instance_cost": 0.0, "api_calls": api_calls},
                     **info},
            "messages": [{"role": "system", "content": "s"},
                         {"role": "user", "content": "u"}] + msgs}


def _write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


def build(root: str) -> None:
    # ---- bonsai
    b = os.path.join(root, "bonsai")
    _write(os.path.join(b, "preds.json"), {
        "a__a-1": {"instance_id": "a__a-1", "model_name_or_path": "openai/yamadori",
                   "model_patch": "diff --git a/x b/x\n"},
        "b__b-2": {"instance_id": "b__b-2", "model_name_or_path": "openai/yamadori",
                   "model_patch": "diff --git a/y b/y\n"},
        "c__c-3": {"instance_id": "c__c-3", "model_name_or_path": "openai/yamadori",
                   "model_patch": ""}})
    _write(os.path.join(b, "a__a-1", "a__a-1.traj.json"),
           _traj("a__a-1", "Submitted", 2, [_asst(100, 10), _asst(200, 20, "stop")]))
    _write(os.path.join(b, "b__b-2", "b__b-2.traj.json"),
           _traj("b__b-2", "LimitsExceeded", 3, [
               _asst(100, 5000, "length"),
               {"role": "user", "content": "Tool call error",
                "extra": {"interrupt_type": "FormatError"}},
               _asst(300, 30), _asst(400, 40)]))
    _write(os.path.join(b, "c__c-3", "c__c-3.traj.json"),
           _traj("c__c-3", "APIConnectionError", 0, [],
                 exception_str="Connection refused"))
    with open(os.path.join(b, "timings.jsonl"), "w", encoding="utf-8") as f:
        f.write(json.dumps({"instance": "a__a-1", "seconds": 50.0}) + "\n")
        f.write(json.dumps({"instance": "b__b-2", "seconds": 1.0}) + "\n")
        f.write(json.dumps({"instance": "b__b-2", "seconds": 90.5}) + "\n")  # re-run wins
        f.write(json.dumps({"instance": "c__c-3", "seconds": 3.0}) + "\n")
    with open(os.path.join(b, "mini_stdout.log"), "w", encoding="utf-8") as f:
        f.write("=== a__a-1 2026-09-22\nall fine\n"
                "=== c__c-3 2026-09-22\nRetrying in 4s: APIConnectionError\n"
                "litellm.InternalServerError: 502\n")
    _write(os.path.join(b, "openai__yamadori.pilot.json"), {
        "total_instances": 3, "submitted_instances": 3,
        "resolved_ids": ["a__a-1"], "unresolved_ids": ["b__b-2"],
        "empty_patch_ids": ["c__c-3"], "error_ids": []})
    # ---- yamadori: no run report, only the per-instance report.json
    y = os.path.join(root, "yamadori")
    _write(os.path.join(y, "preds.json"), {
        "a__a-1": {"instance_id": "a__a-1", "model_name_or_path": "openai/yamadori",
                   "model_patch": "diff --git a/x b/x\n"}})
    x = {"tier": "max", "hints": [{"score": 0.7}], "hops": 3,
         "tools_gate": {"offer": True, "why": "held: numpy"},
         "investigate": {"ran": True, "hops": 2, "injected": True},
         "fanout": {"n": 3, "asked": 3},
         "selection": {"because": {"hints": "forced on by X-Yamadori-Features"}}}
    _write(os.path.join(y, "a__a-1", "a__a-1.traj.json"),
           _traj("a__a-1", "Submitted", 1, [_asst(500, 50, x=x, hops=3)]))
    _write(os.path.join(y, "logs", "run_evaluation", "pilot", "openai__yamadori",
                        "a__a-1", "report.json"), {"a__a-1": {"resolved": True}})
    with open(os.path.join(y, "timings.jsonl"), "w", encoding="utf-8") as f:
        f.write(json.dumps({"instance": "a__a-1", "seconds": 400.0}) + "\n")


def main() -> int:
    root = tempfile.mkdtemp(prefix="swebench_parse_")
    run = os.path.join(root, "pilot")
    try:
        build(run)
        rows = pr.parse_run(run)
        by = {(r["arm"], r["instance"]): r for r in rows}
        check("one row per (arm, instance): 4", len(rows) == 4)
        a = by[("bonsai", "a__a-1")]
        check("resolved from the run report", a["resolved"] is True
              and a["eval_status"] == "resolved")
        check("tokens summed over assistant turns",
              a["prompt_tokens"] == 300 and a["completion_tokens"] == 30)
        check("steps = api_calls", a["steps"] == 2)
        check("seconds from timings.jsonl", a["seconds"] == 50.0)
        b = by[("bonsai", "b__b-2")]
        check("unresolved kept as unresolved", b["resolved"] is False
              and b["eval_status"] == "unresolved")
        check("format errors counted", b["format_errors"] == 1)
        check("a length finish is a budget event, counted",
              b["length_events"] == 1 and b["finish_reasons"].get("length") == 1)
        check("step-limit exit carried through", b["exit_status"] == "LimitsExceeded")
        check("a re-run's timing supersedes", b["seconds"] == 90.5)
        c = by[("bonsai", "c__c-3")]
        check("agent exception is agent_error, never 'unresolved'",
              c["eval_status"] == "agent_error" and c["resolved"] is False)
        check("the exception text is kept", c["agent_exception"] == "Connection refused")
        check("proxy error lines counted per instance",
              c["proxy_error_lines"] == 2 and a["proxy_error_lines"] == 0)
        yrow = by[("yamadori", "a__a-1")]
        check("per-instance report.json used when no run report",
              yrow["eval_status"] == "resolved")
        xs = yrow["x_yamadori"]
        check("x_yamadori: hints, deep thinking, fan-out, internal hops",
              xs["hint_turns"] == 1 and xs["investigate_ran"] == 1
              and xs["investigate_injected"] == 1 and xs["fanout_turns"] == 1
              and xs["internal_tool_turns"] == 1 and xs["tools_offered_turns"] == 1)
        check("proxy hops summed", yrow["proxy_hops"] == 3)
        check("bonsai rows carry no x_yamadori turns",
              by[("bonsai", "a__a-1")]["x_yamadori"]["turns"] == 0)
        s = pr.summarize(rows)
        check("summary counts per arm", s["bonsai"]["n"] == 3
              and s["bonsai"]["resolved"] == 1 and s["yamadori"]["resolved"] == 1)
        # Rows must be JSON-serialisable as written to results.jsonl.
        check("rows serialise", all(json.loads(json.dumps(r)) for r in rows))
        # The arm table is what the runner and the doc name.
        check("arm names are the operator's",
              sorted(arms.ARMS) == ["bonsai", "yamadori", "yamadori-auto"])
        check("bonsai forces everything off",
              json.loads(arms.header("bonsai")) == {
                  "retrieval": False, "hints": False, "investigate": False,
                  "fanout": 1, "effort": "medium"})
        check("yamadori-auto forces nothing but effort",
              json.loads(arms.header("yamadori-auto")) == {"effort": "medium"})
        check("yamadori forces everything on, fan-out 3",
              json.loads(arms.header("yamadori")) == {
                  "retrieval": True, "hints": True, "investigate": True,
                  "fanout": 3, "effort": "medium"})
    finally:
        shutil.rmtree(root, ignore_errors=True)
    n = sum(ok for _, ok in CHECKS)
    print(f"{n}/{len(CHECKS)} checks passed")
    return 0 if n == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
