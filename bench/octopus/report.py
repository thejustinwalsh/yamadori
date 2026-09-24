#!/usr/bin/env python
"""The pilot / paired-run table from grade.py's rows (latest grade per run).

    python bench/octopus/report.py [--tag pilot] [--md]

One line per run: completed?, build, runtime checks, spec checks passed/total,
model calls, minutes, tokens, Wh, repairs, deep thinking, compactions, cache
reuse. Numbers only; every one traces to bench/octopus/results/grades.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GRADES = os.path.join(HERE, "results", "grades.jsonl")


# The prompt author's own run, as he reported it (x.com/sudoingX/status/
# 2102829187581317225, 2026-09-23; relayed by the coordinator, not
# re-verified here): Bonsai 2 27B + MTP via Hermes Agent on an RTX 3060 12 GB,
# ONE session. Not graded by grade.py -- a reference, not a row of ours.
REFERENCE = {
    "run": "reference: sudoingX, V0, RTX 3060 (self-reported)",
    "completed": "finished (reported)", "files": "8 js", "build": "-",
    "runtime": "-", "spec": "-", "model_calls": None, "minutes": 300,
    "tokens_completion": 328000, "gpu_wh_ledger": None, "repairs": "-",
    "deep_thinking": "-", "compactions": None, "cache_reuse_pct": None,
    "lines": 2368, "tok_s_fresh": 50, "tok_s_avg": 22, "peak_context": 125000,
}


def rows(tag: str | None) -> list[dict]:
    latest: dict[str, dict] = {}
    for ln in open(GRADES, encoding="utf-8"):
        r = json.loads(ln)
        gid = r.get("grade_id", "")
        if tag and not gid.startswith(tag + "-"):
            continue
        latest[gid] = r
    return [latest[k] for k in sorted(latest)]


def line(r: dict) -> dict:
    c = r.get("completion") or {}
    b = r.get("build") or {}
    st = r.get("stack") or {}
    led = st.get("ledger") or {}
    spec = r.get("spec") or []
    rt_checks = [x for x in spec if x.get("method") in ("runtime", "pixel", "runtime+pixel")]
    tok = {"completion": 0, "prompt_processed": 0, "prompt_cached": 0}
    for _acct, roles in (led.get("tokens") or {}).items():
        for _role, d in roles.items():
            for k in tok:
                tok[k] += int(d.get(k) or 0)

    def step(k):
        v = b.get(k)
        return v.get("rc") if isinstance(v, dict) else None
    if b.get("npm_install") is not None:
        build = f"install {step('npm_install')} / tsc {step('tsc')} " \
                f"({(b.get('tsc') or {}).get('error_ts')} TS errors) / build {step('build')}"
    else:
        build = f"serve_ok {b.get('serve_ok')}"
    tc = st.get("tool_code") or {}
    dt = st.get("deep_thinking") or {}
    cache = st.get("cache") or {}
    return {
        "run": r.get("grade_id"),
        "completed": c.get("outcome"),
        "files": f"{b.get('files_present')}/{b.get('files_expected')}",
        "build": build,
        "runtime": f"{sum(1 for x in rt_checks if x['pass'] is True)}/{len(rt_checks)}"
                   + (f" ({(r.get('runtime') or {}).get('mode')})" if r.get("runtime") else ""),
        "spec": f"{r.get('spec_passed')}/{r.get('spec_total')}",
        "model_calls": c.get("model_calls"),
        "minutes": round((c.get("wall_s") or 0) / 60, 1),
        "tokens_completion": tok["completion"] or st.get("completion_tokens"),
        "peak_context": st.get("peak_prompt_tokens"),
        "tokens_prompt_processed": tok["prompt_processed"],
        "gpu_wh_ledger": led.get("gpu_wh"),
        "wh_requests": st.get("energy_wh_requests"),
        "repairs": f"{tc.get('fixed', 0)} fixed / {tc.get('requests_checked', 0)} checked "
                   f"({tc.get('errors_before', 0)} -> {tc.get('errors_after', 0)} errors)",
        "deep_thinking": f"{dt.get('ran', 0)} ran",
        "compactions": len(st.get("compactions") or []),
        "cache_reuse_pct": cache.get("reuse_pct"),
        "routes": st.get("routes"),
        "tier": (st.get("first_request") or {}).get("tier"),
        "fps_play": ((r.get("runtime") or {}).get("fps_play") or {}).get("fps"),
        "renderer": (r.get("runtime") or {}).get("context_types"),
        "screenshots": (r.get("runtime") or {}).get("screenshots"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--md", action="store_true")
    a = ap.parse_args()
    out = [line(r) for r in rows(a.tag)] + ([REFERENCE] if a.tag == "pilot" else [])
    if not a.md:
        print(json.dumps(out, indent=1))
        return 0
    cols = ["run", "completed", "files", "build", "runtime", "spec", "model_calls",
            "minutes", "tokens_completion", "peak_context", "wh_requests", "gpu_wh_ledger", "repairs", "deep_thinking",
            "compactions", "cache_reuse_pct"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for o in out:
        print("| " + " | ".join(str(o.get(c)) for c in cols) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
