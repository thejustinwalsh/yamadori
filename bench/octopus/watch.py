#!/usr/bin/env python
"""Continuous assessment of one Octopus run WHILE it runs.

    python bench/octopus/watch.py pilot-V0-xhigh-1 [--interval 30] [--once]

Reads, every interval, from scratch (so it is idempotent and can be restarted):
  - the relay's rows, logs/<run>/relay.jsonl: per request x_yamadori (tier,
    route, cache, tool_code, compaction, warm, energy), usage, timings
  - logs/proxy.out.log from the run's first line: the `warm: slot N reused R
    processed P` lines (the only place a warm's cost is printed)
  - the Hermes session in the dogfood profile's state.db (read-only): each
    assistant message's reasoning and content, each tool result
  - logs/<run>/hermes.jsonl (stream-json): tool calls, tool results, errors
  - index/token_ledger.sqlite3: the run's account's tokens since it started

Writes:
  logs/<run>/assessment.md     rolling summary at the top, then one row per
                               step, then every defect with its step and
                               evidence (rewritten every pass)
  logs/<run>/assessment.jsonl  one row per completed step, appended once
  logs/<run>/defects.jsonl     one row per defect, appended when first seen
stdout: one line per NEW defect (run it under a monitor to be told).

DETECTORS and their thresholds (choices, not measurements -- stated so a
reader can disagree with a number rather than with a hidden rule):
  cache_miss        a step reused < 95% of the PREVIOUS step's prompt tokens
                    (the conversation only grows; anything else re-read the
                    past, not just the new tail)
  warm_reread       a warm processed more than max(4000, 1.5 x that step's
                    completion tokens): it re-read the prompt, not its delta
  repeat_call       a tool call identical (name + arguments) to an earlier one
  reread_file       read_file of a path already read, with no write between
  replan            word 8-gram overlap (Jaccard) >= 0.30 between this step's
                    reasoning and an earlier step's (both >= 200 words)
  template_marker   <think>, </think> or <|im_ in an assistant message's content
  empty_turn        a finished step with no content and no tool call
  tool_error_streak 3+ consecutive tool results that are errors (is_error, or
                    a terminal exit_code != 0)
  long_reasoning    completion > 10,000 tokens in one step
  depth_jump        prompt grew > 12,000 tokens in one step
  slow_decode       < 10 tok/s over a step with > 1,000 completion tokens
  kickoff           step 1: did deep thinking fire with trigger `kickoff` (the
                    spec is ~2,246 tokens against a 1,500 threshold)? Reported
                    either way; NOT firing is a defect
  deep_fired        any x_yamadori.deep trigger that fired (model / struggle /
                    area / kickoff / forced), with the hand-off size
                    (x_yamadori.investigate.handoff) and think_deeply calls
  planning          main's completion tokens on steps 1-3, against the last
                    V0's 13,923 / 12,175 / 13,232 (summary, not a defect)
  warm_short        x_yamadori.warm_before.short is true: the previous turn's
                    warm reused less than the prompt the slot had just generated
  template_markers  x_yamadori.template_markers present (source model / ours)
  gpu_room          x_yamadori.gpu_room entries (recorded, summarised)
  wrap_up           Hermes' run-budget wrap-up notice reached the model (a
                    request whose last message mentions the budget / wrap up);
                    what follows it is summarised
Reference pace (the prompt author's own V0 run, self-reported, relayed by the
coordinator): 5 h, 328k tokens written, 125k context peak, 50 tok/s fresh,
22 tok/s average, RTX 3060.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import run as runmod  # noqa: E402

PROXY_LOG = os.path.join(ROOT, "logs", "proxy.out.log")
STATE_DB = os.path.join(runmod.HERMES_HOME, "state.db")
REF = {"hours": 5.0, "tokens": 328_000, "context": 125_000, "tok_s_avg": 22, "tok_s_fresh": 50}
MARKERS = ("<think>", "</think>", "<|im_")
# The previous V0 on the old build (pilot-V0-xhigh-1, stopped by the operator at
# 68 min): context 9,457 -> 89,116 over 16 steps; steps 1-3 wrote 13,923 /
# 12,175 / 13,232 completion tokens; 5/8 js files; 18.7 tok/s average.
LAST_V0 = {"run": "pilot-V0-xhigh-1", "hours": 1.14, "js_files": 5, "context_peak": 89_116,
           "tok_s_avg": 18.7, "planning_1_3": [13_923, 12_175, 13_232],
           "depth_by_step": [9457, 23436, 35758, 48988, 52923, 56288, 59609, 59823,
                             65058, 68938, 76251, 86394, 87270, 87715, 88116, 89116]}


def jl(path: str) -> list[dict]:
    out = []
    try:
        for ln in open(path, encoding="utf-8", errors="replace"):
            ln = ln.strip()
            if ln:
                try:
                    out.append(json.loads(ln))
                except ValueError:
                    pass
    except OSError:
        pass
    return out


def proxy_offset(run_id: str, log_dir: str) -> int:
    """Byte offset of the run's first request in the proxy log: remembered in
    logs/<run>/proxy_offset once found (the first route line naming the
    prompt's opening words after the run started)."""
    p = os.path.join(log_dir, "proxy_offset")
    if os.path.isfile(p):
        return int(open(p).read().strip())
    head = open(os.path.join(log_dir, "prompt.md"), encoding="utf-8").read().split(".")[0][:40]
    data = open(PROXY_LOG, "rb").read()
    i = data.rfind(f"('{head[:30]}".encode())
    if i < 0:
        return 0
    off = data.rfind(b"\n", 0, i) + 1
    with open(p, "w") as f:
        f.write(str(off))
    return off


def warms(off: int) -> list[dict]:
    try:
        data = open(PROXY_LOG, "rb").read()[off:].decode("utf-8", "replace")
    except OSError:
        return []
    return [{"slot": int(m.group(1)), "reused": int(m.group(2)), "processed": int(m.group(3))}
            for m in re.finditer(r"warm: slot (\d+) reused (\d+) processed (\d+)", data)]


def session_messages(sid: str | None) -> list[dict]:
    if not sid or not os.path.isfile(STATE_DB):
        return []
    try:
        con = sqlite3.connect(f"file:{STATE_DB}?mode=ro", uri=True, timeout=5)
        con.row_factory = sqlite3.Row
        rows = [dict(r) for r in con.execute(
            "SELECT id, role, content, tool_calls, tool_name, timestamp, token_count, "
            "finish_reason, reasoning_content, reasoning FROM messages "
            "WHERE session_id=? ORDER BY id", (sid,))]
        con.close()
        return rows
    except sqlite3.Error:
        return []


def _deep_of(x: dict) -> dict | None:
    """x_yamadori.deep (the trigger decision) + investigate (the run and its
    hand-off size), flattened for the step table."""
    d = x.get("deep") or {}
    inv = x.get("investigate") or {}
    if not d and not inv:
        return None
    tt = d.get("think_tool") or {}
    ho = inv.get("handoff") or {}
    return {"fired": bool(d.get("fire")), "kind": d.get("kind"), "job": d.get("job"),
            "because": (d.get("because") or "")[:160], "ran": bool(inv.get("ran")),
            "searches": inv.get("searches"), "handoff_chars": ho.get("chars"),
            "handoff_facts": ho.get("facts"), "seconds": inv.get("seconds"),
            "think_offered": tt.get("offered"), "think_calls": len(tt.get("calls") or [])}


def ngrams(text: str, n: int = 8) -> set:
    w = re.findall(r"\w+", (text or "").lower())
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)} if len(w) >= 200 else set()


def assess(run_id: str) -> dict:
    log_dir = os.path.join(runmod.LOGS_DIR, run_id)
    run_dir = os.path.join(runmod.RUNS_DIR, run_id)
    relay = [r for r in jl(os.path.join(log_dir, "relay.jsonl"))
             if r.get("method") == "POST" and r.get("path", "").endswith("/chat/completions")]
    events = jl(os.path.join(log_dir, "hermes.jsonl"))
    sid = next((e.get("session_id") for e in events if e.get("session_id")), None)
    msgs = session_messages(sid)
    ws = warms(proxy_offset(run_id, log_dir))
    done = os.path.isfile(os.path.join(log_dir, "meta.json"))

    steps, defects = [], []

    def flag(kind, step, evidence, proxy=False):
        defects.append({"kind": kind, "step": step, "evidence": evidence,
                        "proxy_suspect": proxy})

    t_first = relay[0]["t0"] if relay else time.time()
    assistant = [m for m in msgs if m["role"] == "assistant"]
    wi = 0
    prev_prompt = None
    grams: list[tuple[int, set]] = []
    first_main = next((i for i, r in enumerate(relay, 1)
                       if not ((r.get("response") or {}).get("x_yamadori") or {}).get("utility")
                       and "t_end" in r), None)
    for i, r in enumerate(relay, 1):
        rs = r.get("response") or {}
        if "t_end" not in r:
            continue
        x = rs.get("x_yamadori") or {}
        u = rs.get("usage") or {}
        c = x.get("cache") or {}
        tc = x.get("tool_code") or {}
        util = bool(x.get("utility"))
        secs = max(0.001, r["t_end"] - r["t0"])
        comp = u.get("completion_tokens") or 0
        prompt = u.get("prompt_tokens") or c.get("prompt")
        # the assistant message this request produced: first one stamped in its window
        # Hermes stamps a message when the response completes, so it is the one
        # nearest t_end, strictly after this request started (the previous
        # message can carry the same second as this request's t0).
        cand = [m for m in assistant if r["t0"] + 0.5 <= m["timestamp"] <= r["t_end"] + 30]
        am = min(cand, key=lambda m: abs(m["timestamp"] - r["t_end"])) if cand else None
        reasoning = (am or {}).get("reasoning_content") or (am or {}).get("reasoning") or ""
        content = (am or {}).get("content") or ""
        st = {"step": i, "t_min": round((r["t0"] - t_first) / 60, 1), "seconds": round(secs),
              "prompt": prompt, "reused": c.get("reused"), "processed": c.get("processed"),
              "completion": comp, "reasoning_chars": rs.get("reasoning_chars"),
              "content_chars": rs.get("content_chars"), "tool_calls": rs.get("tool_calls"),
              "route": (x.get("route") or {}).get("class"), "tier": x.get("tier"),
              "utility": util, "repair": (f"{tc.get('stopped')} {tc.get('errors_before')}->"
                                          f"{tc.get('errors_after')}" if tc else None),
              "compaction": bool(x.get("compaction")), "wh": (x.get("energy") or {}).get("wh"),
              "tok_s": round(comp / secs, 1), "finish": rs.get("finish_reason"),
              "warm": None, "status": r.get("status"),
              "deep": _deep_of(x), "warm_before": x.get("warm_before"),
              "markers": x.get("template_markers"), "gpu_room": len(x.get("gpu_room") or [])}
        if (x.get("warm") or {}).get("sent") and wi < len(ws):
            st["warm"] = ws[wi]
            wi += 1
        steps.append(st)
        if util:
            continue
        # --- detectors
        if prev_prompt and c.get("reused") is not None and c["reused"] < 0.95 * prev_prompt:
            flag("cache_miss", i, f"reused {c['reused']} of prompt {prompt}; previous step's "
                                  f"prompt was {prev_prompt} (processed {c.get('processed')})",
                 proxy=True)
        w = st["warm"]
        if w and w["processed"] > max(4000, 1.5 * comp):
            flag("warm_reread", i, f"warm after step {i} on slot {w['slot']}: reused "
                                   f"{w['reused']} processed {w['processed']} (step wrote {comp} "
                                   f"tokens); repair: {st['repair']}", proxy=True)
        if comp > 10_000:
            flag("long_reasoning", i, f"{comp} completion tokens, {st['reasoning_chars']} "
                                      f"reasoning chars, {st['seconds']} s, for "
                                      f"{st['tool_calls']}")
        if prev_prompt and prompt and prompt - prev_prompt > 12_000:
            flag("depth_jump", i, f"prompt {prev_prompt} -> {prompt} (+{prompt - prev_prompt})")
        if comp > 1000 and st["tok_s"] < 10:
            flag("slow_decode", i, f"{st['tok_s']} tok/s at depth {prompt}")
        if st["finish"] == "stop" and not st["content_chars"] and not st["tool_calls"]:
            flag("empty_turn", i, "finish stop with no content and no tool call")
        if any(mk in content for mk in MARKERS):
            flag("template_marker", i, repr(content[:160]), proxy=True)
        tm = x.get("template_markers")
        if tm and tm.get("in_content"):
            flag("template_markers_x", i, f"x_yamadori.template_markers {json.dumps(tm)[:240]}",
                 proxy=(tm.get("source") == "ours"))
        wb = x.get("warm_before") or {}
        if wb.get("short"):
            flag("warm_short", i, f"warm_before {json.dumps(wb)[:240]}", proxy=True)
        d = st["deep"]
        if i == first_main:
            if d and d.get("fired") and d.get("kind") == "kickoff":
                flag("kickoff_fired", i, f"deep {json.dumps(d)[:300]}")
            else:
                flag("kickoff_missed", i, f"step 1 deep: {json.dumps(d)[:300]}", proxy=True)
        elif d and d.get("fired"):
            flag("deep_fired", i, f"trigger {d.get('kind')}: {json.dumps(d)[:300]}")
        if d and d.get("think_calls"):
            flag("think_deeply_call", i, f"{d['think_calls']} think_deeply call(s)")
        g = ngrams(reasoning)
        for j, gj in grams:
            if g and gj:
                jac = len(g & gj) / max(1, len(g | gj))
                if jac >= 0.30:
                    flag("replan", i, f"reasoning 8-gram Jaccard {jac:.2f} with step {j}")
                    break
        if g:
            grams.append((i, g))
        head = ((r.get("request") or {}).get("last_head") or "").lower()
        if (r.get("request") or {}).get("last_role") in ("user", "system") and i > 1 and \
                re.search(r"budget|wrap[ -]?up|time is (almost )?up|remaining", head):
            flag("wrap_up", i, f"last message: {head[:200]!r}")
        prev_prompt = prompt

    # tool calls / results from the stream
    seen_calls: dict[str, int] = {}
    read_paths: dict[str, int] = {}
    streak = 0
    k = 0
    written = set()
    for e in events:
        if e.get("type") == "tool_use":
            k += 1
            key = json.dumps([e.get("name"), e.get("input")], sort_keys=True)
            if key in seen_calls:
                flag("repeat_call", f"call {k}", f"{e.get('name')} identical to call "
                                                 f"{seen_calls[key]}: {key[:200]}")
            seen_calls.setdefault(key, k)
            inp = e.get("input") or {}
            path = inp.get("path") or inp.get("file_path")
            if e.get("name") == "read_file" and path:
                if path in read_paths:
                    flag("reread_file", f"call {k}", f"{path} read again (first at call "
                                                     f"{read_paths[path]}), no write between")
                read_paths[path] = k
            if e.get("name") in ("write_file", "patch") and path:
                read_paths.pop(path, None)
                written.add(path)
        elif e.get("type") == "tool_result":
            err = bool(e.get("is_error"))
            try:
                out = json.loads(e.get("output") or "{}")
                err = err or (isinstance(out, dict) and out.get("exit_code") not in (0, None))
            except ValueError:
                pass
            streak = streak + 1 if err else 0
            if streak == 3:
                flag("tool_error_streak", f"call {k}", "3 consecutive tool errors")

    # depth-bucketed decode speed
    buckets: dict[str, list] = {}
    for s in steps:
        if s["utility"] or not s["prompt"] or s["completion"] < 500:
            continue
        b = f"{(s['prompt'] // 32000) * 32}k-{(s['prompt'] // 32000 + 1) * 32}k"
        buckets.setdefault(b, []).append(s["tok_s"])
    by_depth = {b: round(sum(v) / len(v), 1) for b, v in sorted(buckets.items())}

    real = [s for s in steps if not s["utility"]]
    hours = (time.time() - t_first) / 3600 if relay else 0
    if done:
        try:
            m = json.load(open(os.path.join(log_dir, "meta.json")))
            hours = (m["t_end"] - m["t_start"]) / 3600
        except (OSError, ValueError, KeyError):
            pass
    comp_total = sum(s["completion"] for s in real)
    files = []
    for dp, dn, fn in os.walk(run_dir):
        dn[:] = [d for d in dn if d not in ("node_modules", "dist")]
        for f in fn:
            p = os.path.join(dp, f)
            try:
                files.append((os.path.relpath(p, run_dir).replace("\\", "/"),
                              sum(1 for _ in open(p, encoding="utf-8", errors="replace"))))
            except OSError:
                pass
    counts: dict[str, int] = {}
    for d in defects:
        counts[d["kind"]] = counts.get(d["kind"], 0) + 1
    summary = {
        "run": run_id, "state": (("stopped_by_operator" if (json.load(open(os.path.join(
            log_dir, "meta.json"))) or {}).get("stopped_by_operator") else "finished")
            if done else "running"),
        "elapsed_h": round(hours, 2), "steps": len(real),
        "tool_calls": k, "files": len(files), "lines": sum(n for _f, n in files),
        "completion_tokens": comp_total,
        "pace_tokens_per_h": round(comp_total / hours) if hours else None,
        "ref_tokens_per_h": round(REF["tokens"] / REF["hours"]),
        "context_now": real[-1]["prompt"] if real else None,
        "context_peak": max((s["prompt"] or 0 for s in real), default=0),
        "ref_context_peak": REF["context"],
        "tok_s_avg": round(comp_total / max(1, sum(s["seconds"] for s in real)), 1),
        "tok_s_by_depth": by_depth, "ref_tok_s": [REF["tok_s_fresh"], REF["tok_s_avg"]],
        "wh": round(sum(s["wh"] or 0 for s in real), 1),
        "warm_processed": sum((s["warm"] or {}).get("processed", 0) for s in steps),
        "repairs": sum(1 for s in real if (s["repair"] or "").startswith("fixed")),
        "compactions": sum(1 for s in steps if s["compaction"]),
        "routes": sorted({s["route"] for s in real if s["route"]}),
        "tiers": sorted({s["tier"] for s in real if s["tier"]}),
        "defects_by_kind": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        "planning_tokens_steps_1_3": [s["completion"] for s in real[:3]],
        "last_v0_planning_1_3": LAST_V0["planning_1_3"],
        "depth_by_step": [s["prompt"] for s in real],
        "last_v0_depth_by_step": LAST_V0["depth_by_step"],
        "depth_vs_last_v0_same_step": ([round((s["prompt"] or 0) / LAST_V0["depth_by_step"][j], 2)
                                        for j, s in enumerate(real[:len(LAST_V0["depth_by_step"])])]),
        "js_files": sum(1 for f, _n in files if f.endswith(".js")),
        "last_v0": {k: LAST_V0[k] for k in ("hours", "js_files", "context_peak", "tok_s_avg")},
        "deep": {"kickoff_step1": next((s["deep"] for s in real[:1]), None),
                 "fired": [(s["step"], (s["deep"] or {}).get("kind"),
                            (s["deep"] or {}).get("handoff_chars"))
                           for s in real if (s["deep"] or {}).get("fired")],
                 "think_deeply_calls": sum((s["deep"] or {}).get("think_calls") or 0
                                           for s in real)},
        "warm_short": sum(1 for s in real if (s["warm_before"] or {}).get("short")),
        "template_markers": sum(1 for s in real if (s["markers"] or {}).get("in_content")),
        "gpu_room_entries": sum(s["gpu_room"] for s in real),
    }
    return {"summary": summary, "steps": steps, "defects": defects, "files": files,
            "session_id": sid, "done": done}


def write(run_id: str, a: dict) -> list[dict]:
    log_dir = os.path.join(runmod.LOGS_DIR, run_id)
    s = a["summary"]
    lines = [f"# Assessment: {run_id} ({s['state']}, updated {time.strftime('%H:%M:%S')})", "",
             "## Summary", ""]
    lines += [f"- {k}: {v}" for k, v in s.items()]
    lines += ["", "## Files", ""] + [f"- {f} ({n} lines)" for f, n in a["files"]]
    cols = ["step", "t_min", "seconds", "prompt", "reused", "processed", "warm", "completion",
            "reasoning_chars", "content_chars", "tool_calls", "route", "tier", "repair",
            "compaction", "deep_col", "wb_col", "tok_s", "wh", "finish"]
    lines += ["", "## Steps", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for st in a["steps"]:
        w = st["warm"]
        d = st.get("deep") or {}
        wb = st.get("warm_before") or {}
        st = {**st, "warm": f"{w['reused']}/{w['processed']}" if w else "",
              "deep_col": (f"{d.get('kind')} ho={d.get('handoff_chars')}" if d.get("fired")
                           else ""),
              "wb_col": ("SHORT" if wb.get("short") else
                         (f"{wb.get('reused')}/{wb.get('processed')}" if wb else ""))}
        lines.append("| " + " | ".join(str(st.get(c)) for c in cols) + " |")
    lines += ["", "## Defects", ""] + [
        f"- step {d['step']}: **{d['kind']}**{' (proxy suspect)' if d['proxy_suspect'] else ''}"
        f" -- {d['evidence']}" for d in a["defects"]]
    with open(os.path.join(log_dir, "assessment.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    # append-once files
    seen_steps = {r["step"] for r in jl(os.path.join(log_dir, "assessment.jsonl"))}
    with open(os.path.join(log_dir, "assessment.jsonl"), "a", encoding="utf-8") as f:
        for st in a["steps"]:
            if st["step"] not in seen_steps:
                f.write(json.dumps({**st, "logged_at": time.time()}) + "\n")
    old = {(d["kind"], str(d["step"])) for d in jl(os.path.join(log_dir, "defects.jsonl"))}
    new = [d for d in a["defects"] if (d["kind"], str(d["step"])) not in old]
    with open(os.path.join(log_dir, "defects.jsonl"), "a", encoding="utf-8") as f:
        for d in new:
            f.write(json.dumps({**d, "seen_at": time.time()}) + "\n")
    return new


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--interval", type=float, default=30)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--wait", action="store_true",
                    help="wait for the run's first relay row before assessing")
    a = ap.parse_args()
    relay_path = os.path.join(runmod.LOGS_DIR, a.run_id, "relay.jsonl")
    while a.wait and not os.path.isfile(relay_path):
        time.sleep(5)
    print(f"[{a.run_id}] assessment started", flush=True)
    while True:
        res = assess(a.run_id)
        for d in write(a.run_id, res):
            print(f"[{a.run_id}] step {d['step']} {d['kind']}"
                  f"{' PROXY?' if d['proxy_suspect'] else ''}: {d['evidence'][:300]}", flush=True)
        if a.once or res["done"]:
            if res["done"]:
                print(f"[{a.run_id}] run finished: {json.dumps(res['summary'])[:600]}", flush=True)
            return 0
        time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())
