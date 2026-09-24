#!/usr/bin/env python
"""Per-step table of one Octopus run, from the relay's rows (x_yamadori) and
the proxy log's warm lines. Works while the run is still going.

    python bench/octopus/steps.py pilot-V0-xhigh-1 [--json]

Columns: step, start (s since the run began), seconds, completion tokens,
reasoning chars (usage carries no reasoning_tokens on this build), tool
calls, tool_code (stopped, errors before->after, rounds), cache reused /
processed, Wh (x_yamadori.energy), route, finish.

WARMS: x_yamadori.warm only says a warm was scheduled; its cost is printed
by the proxy as `warm: slot N reused R processed P`, with no timestamp. They
are read from logs/proxy.out.log from the byte offset run.py recorded when
the run started (runs.jsonl stack.proxy_log_bytes[0]; for a run still going,
the offset of its not_run-free row is unknown, so --from-byte can be given).
Valid only while the run is the proxy's sole client (the pilot's rule).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import run as runmod  # noqa: E402

PROXY_LOG = os.path.join(ROOT, "logs", "proxy.out.log")


def steps(run_id: str) -> list[dict]:
    path = os.path.join(runmod.LOGS_DIR, run_id, "relay.jsonl")
    rows = [json.loads(ln) for ln in open(path, encoding="utf-8") if ln.strip()]
    posts = [r for r in rows if r.get("method") == "POST"
             and r.get("path", "").endswith("/chat/completions")]
    t0 = posts[0]["t0"] if posts else 0
    out = []
    for i, r in enumerate(posts, 1):
        rs = r.get("response") or {}
        x = rs.get("x_yamadori") or {}
        tc = x.get("tool_code") or {}
        c = x.get("cache") or {}
        u = rs.get("usage") or {}
        out.append({
            "step": i, "start_s": round(r["t0"] - t0), "seconds": round(r.get("t_end", r["t0"]) - r["t0"]),
            "prompt_tokens": u.get("prompt_tokens"), "completion": u.get("completion_tokens"),
            "reasoning_chars": rs.get("reasoning_chars"), "content_chars": rs.get("content_chars"),
            "tool_calls": rs.get("tool_calls"),
            "tool_code": (f"{tc.get('stopped')} {tc.get('errors_before')}->{tc.get('errors_after')}"
                          f" r{tc.get('rounds')}" if tc else None),
            "files": [f.get("path", "").split("/")[-1] for f in tc.get("files") or []],
            "cache": f"{c.get('reused')}/{c.get('processed')}" if c else None,
            "wh": (x.get("energy") or {}).get("wh"),
            "route": (x.get("route") or {}).get("class"), "utility": x.get("utility"),
            "compaction": bool(x.get("compaction")), "warm": (x.get("warm") or {}).get("sent"),
            "finish": rs.get("finish_reason"), "status": r.get("status"),
            "client_gone": r.get("client_gone"),
        })
    return out


def warms(from_byte: int) -> list[dict]:
    data = open(PROXY_LOG, "rb").read()[from_byte:].decode("utf-8", "replace")
    return [{"slot": int(m.group(1)), "reused": int(m.group(2)), "processed": int(m.group(3))}
            for m in re.finditer(r"warm: slot (\d+) reused (\d+) processed (\d+)", data)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--from-byte", type=int)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    st = steps(a.run_id)
    fb = a.from_byte
    if fb is None:
        for ln in open(runmod.RUNS, encoding="utf-8"):
            r = json.loads(ln)
            if r.get("run_id") == a.run_id and r.get("stack"):
                fb = r["stack"]["proxy_log_bytes"][0]
    w = warms(fb) if fb is not None else []
    tot = {"steps": len(st), "minutes": round(sum(s["seconds"] for s in st) / 60, 1),
           "completion": sum(s["completion"] or 0 for s in st),
           "reasoning_chars": sum(s["reasoning_chars"] or 0 for s in st),
           "wh": round(sum(s["wh"] or 0 for s in st), 1),
           "repairs": sum(1 for s in st if (s["tool_code"] or "").startswith("fixed")),
           "warms": len(w), "warm_processed": sum(x["processed"] for x in w),
           "warm_reused": sum(x["reused"] for x in w),
           "peak_prompt_tokens": max((s["prompt_tokens"] or 0 for s in st), default=0),
           # wall-clock rates: a step's seconds include prompt processing,
           # the tool_code fix-up and queueing, so these are LOWER BOUNDS on
           # decode speed. "fresh" = step 1 (empty cache, shortest context).
           "tok_s_avg": round(sum(s["completion"] or 0 for s in st)
                              / max(1, sum(s["seconds"] for s in st)), 1),
           "tok_s_fresh": (round((st[0]["completion"] or 0) / max(1, st[0]["seconds"]), 1)
                           if st else None),
           "compactions": sum(1 for s in st if s["compaction"])}
    if a.json:
        print(json.dumps({"steps": st, "warms": w, "totals": tot}, indent=1))
        return 0
    cols = ["step", "start_s", "seconds", "completion", "reasoning_chars", "tool_calls",
            "tool_code", "files", "cache", "wh", "route", "finish"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for s in st:
        print("| " + " | ".join(str(s[c]) for c in cols) + " |")
    print()
    print("warms (proxy log):", w)
    print("totals:", tot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
