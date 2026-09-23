#!/usr/bin/env python
"""SWE-bench run outputs -> one results jsonl row per (instance, arm).

    python bench/swebench/parse_results.py bench/swebench/results/<run-id>
    python bench/swebench/parse_results.py <run-dir> --print     # + a table

Reads, for every arm directory under the run directory:

  preds.json                          mini-swe-agent's predictions
  <instance>/<instance>.traj.json     the trajectory (messages carry the raw
                                      response, so usage and x_yamadori too)
  timings.jsonl                       wall seconds per instance (wsl_side.py)
  mini_stdout.log                     retries / proxy errors, per instance
  <model>.<run_id>.json               the swebench harness's run report
  logs/run_evaluation/<run_id>/<model>/<instance>/report.json

and writes <run-dir>/results.jsonl. Standard library only, so the dashboard
(or anything else) can import `parse_run`.

OUTCOMES ARE KEPT APART (PROTOCOL rule 3). `eval_status` is one of
  resolved | unresolved | empty_patch | agent_error | eval_error | not_evaluated
and `exit_status` is the agent's own ending (Submitted, LimitsExceeded,
FormatError, ContextWindowExceededError, ...). A row whose agent run raised
is an ERROR of the harness or the stack until a trajectory says otherwise --
it is never folded into "unresolved".
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import sys

ERROR_LINE = re.compile(
    r"Retrying|RateLimitError|InternalServerError|ServiceUnavailable|"
    r"APIConnectionError|APIError|Timeout|\b429\b|\b502\b|\b503\b")


def _load(path: str):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _timings(arm_dir: str) -> dict[str, dict]:
    """The LAST timing record per instance (a re-run supersedes)."""
    out: dict[str, dict] = {}
    p = os.path.join(arm_dir, "timings.jsonl")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("instance"):
                    out[r["instance"]] = r
    return out


def _log_errors(arm_dir: str) -> dict[str, int]:
    """Error-looking lines in mini_stdout.log, per `=== <instance>` section."""
    out: collections.Counter = collections.Counter()
    p = os.path.join(arm_dir, "mini_stdout.log")
    if not os.path.exists(p):
        return {}
    cur = None
    with open(p, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("=== "):
                cur = line.split()[1]
                out.setdefault(cur, 0)
            elif cur and ERROR_LINE.search(line):
                out[cur] += 1
    return dict(out)


def _eval(arm_dir: str) -> dict[str, str]:
    """instance -> eval_status, from the harness run report (and the
    per-instance reports when the run report is missing)."""
    status: dict[str, str] = {}
    reports = [p for p in glob.glob(os.path.join(arm_dir, "*.json"))
               if os.path.basename(p) not in ("preds.json", "run.json",
                                              "model_registry.json")]
    for p in reports:
        r = _load(p)
        if not isinstance(r, dict) or "resolved_ids" not in r:
            continue
        for k, v in (("empty_patch_ids", "empty_patch"),
                     ("error_ids", "eval_error"),
                     ("unresolved_ids", "unresolved"),
                     ("resolved_ids", "resolved")):
            for i in r.get(k) or []:
                status[i] = v
    for p in glob.glob(os.path.join(arm_dir, "logs", "run_evaluation", "*", "*",
                                    "*", "report.json")):
        r = _load(p) or {}
        for iid, rep in r.items():
            if iid not in status and isinstance(rep, dict):
                status[iid] = "resolved" if rep.get("resolved") else "unresolved"
    return status


def _traj_stats(traj: dict) -> dict:
    info = traj.get("info") or {}
    msgs = traj.get("messages") or []
    usage = collections.Counter()
    finish = collections.Counter()
    x = {"turns": 0, "hint_turns": 0, "hints_injected": 0,
         "investigate_ran": 0, "investigate_injected": 0,
         "fanout_turns": 0, "internal_tool_turns": 0,
         "tools_offered_turns": 0, "tiers": collections.Counter(),
         "selection_because": collections.Counter()}
    format_errors = 0
    for m in msgs:
        extra = m.get("extra") or {}
        if extra.get("interrupt_type") == "FormatError":
            format_errors += 1
        resp = extra.get("response")
        if m.get("role") != "assistant" or not isinstance(resp, dict):
            continue
        u = resp.get("usage") or {}
        for k in ("prompt_tokens", "completion_tokens", "hops"):
            usage[k] += int(u.get(k) or 0)
        ch = (resp.get("choices") or [{}])[0]
        finish[str(ch.get("finish_reason"))] += 1
        xy = resp.get("x_yamadori")
        if not isinstance(xy, dict):
            continue
        x["turns"] += 1
        x["tiers"][str(xy.get("tier"))] += 1
        if xy.get("hints"):
            x["hint_turns"] += 1
            x["hints_injected"] += len(xy["hints"])
        inv = xy.get("investigate")
        if isinstance(inv, dict):
            x["investigate_ran"] += bool(inv.get("ran"))
            x["investigate_injected"] += bool(inv.get("injected"))
        if isinstance(xy.get("fanout"), dict) and (xy["fanout"].get("n") or 0) > 1:
            x["fanout_turns"] += 1
        if int(xy.get("hops") or 0) > 1:
            x["internal_tool_turns"] += 1
        gate = xy.get("tools_gate")
        if isinstance(gate, dict) and gate.get("offer"):
            x["tools_offered_turns"] += 1
        sel = xy.get("selection") or {}
        for k, v in (sel.get("because") or {}).items():
            x["selection_because"][f"{k}: {v}"[:100]] += 1
    x["tiers"] = dict(x["tiers"])
    x["selection_because"] = dict(x["selection_because"].most_common(8))
    stats = info.get("model_stats") or {}
    return {
        "exit_status": info.get("exit_status"),
        "steps": stats.get("api_calls"),
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "proxy_hops": usage["hops"],
        "format_errors": format_errors,
        "finish_reasons": dict(finish),
        "length_events": finish.get("length", 0),
        "agent_exception": (info.get("exception_str") or "")[:300] or None,
        "x_yamadori": x,
    }


def parse_arm(arm_dir: str, run_id: str, arm: str) -> list[dict]:
    preds = _load(os.path.join(arm_dir, "preds.json")) or {}
    timings = _timings(arm_dir)
    errs = _log_errors(arm_dir)
    status = _eval(arm_dir)
    ids = sorted(set(preds) | set(timings))
    rows = []
    for iid in ids:
        traj = _load(os.path.join(arm_dir, iid, f"{iid}.traj.json")) or {}
        st = _traj_stats(traj) if traj else {"exit_status": None, "steps": None}
        patch = ((preds.get(iid) or {}).get("model_patch") or "")
        ev = status.get(iid)
        if ev is None:
            ev = "empty_patch" if iid in preds and not patch.strip() else "not_evaluated"
        if ev in ("empty_patch", "not_evaluated") and st.get("agent_exception"):
            # The agent never finished: the harness or the stack raised. The
            # leaderboard would score it unresolved; here it stays visible.
            ev = "agent_error"
        t = timings.get(iid) or {}
        rows.append({
            "run_id": run_id, "arm": arm, "instance": iid,
            "resolved": {"resolved": True, "unresolved": False,
                         "empty_patch": False, "agent_error": False}.get(ev),
            "eval_status": ev,
            "seconds": t.get("seconds"),
            "patch_chars": len(patch),
            "proxy_error_lines": errs.get(iid, 0),
            **st,
        })
    return rows


def parse_run(run_dir: str) -> list[dict]:
    run_id = os.path.basename(os.path.normpath(run_dir))
    rows: list[dict] = []
    for arm_dir in sorted(glob.glob(os.path.join(run_dir, "*"))):
        if os.path.isdir(arm_dir) and (
                os.path.exists(os.path.join(arm_dir, "preds.json"))
                or os.path.exists(os.path.join(arm_dir, "timings.jsonl"))):
            rows += parse_arm(arm_dir, run_id, os.path.basename(arm_dir))
    return rows


def summarize(rows: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in rows:
        s = out.setdefault(r["arm"], collections.Counter())
        s["n"] += 1
        s[r["eval_status"]] += 1
        s["seconds"] += r.get("seconds") or 0
        s["steps"] += r.get("steps") or 0
    return {k: dict(v) for k, v in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--print", action="store_true")
    a = ap.parse_args()
    rows = parse_run(a.run_dir)
    out = os.path.join(a.run_dir, "results.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} rows -> {out}")
    if a.print:
        for r in rows:
            print(f"  {r['arm']:<14} {r['instance']:<36} {r['eval_status']:<13} "
                  f"{str(r.get('exit_status')):<22} steps={r.get('steps')} "
                  f"s={r.get('seconds')} pt={r.get('prompt_tokens')} "
                  f"ct={r.get('completion_tokens')} fmt={r.get('format_errors')}")
        print(json.dumps(summarize(rows), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
