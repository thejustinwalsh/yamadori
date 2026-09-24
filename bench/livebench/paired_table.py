"""Per-question paired table: one arm against the bonsai baseline on the same questions.

    python paired_table.py results/<run-id> <arm> [category]    -> paired_<arm>_vs_bonsai_<category>.md

Evidence only, no verdicts (operator's standing rule): the official scores
with 95% CIs (summary.json, stratified bootstrap), the paired difference with
its CI, the discordant pairs with the exact McNemar p, one row per question
with what each mechanism did, and a stdin.buffer sensitivity row
(scan_stdin_buffer.py) beside -- never instead of -- the official score.
Run summarize.py and scan_stdin_buffer.py first.
"""
from __future__ import annotations

import json
import os
import sys


def load(run_dir, arm, cat):
    """Rows of one arm; `arm@condition` selects that arm's rows under one stack condition."""
    base, _, cond = arm.partition("@")
    p = os.path.join(run_dir, f"rows_{base}_{cat}.jsonl")
    if not os.path.exists(p):
        return {}
    rows = (json.loads(l) for l in open(p))
    return {r["id"]: r for r in rows if not cond or r.get("condition") == cond}


def main() -> int:
    run_dir, arm = sys.argv[1], sys.argv[2]
    cat = sys.argv[3] if len(sys.argv) > 3 else "coding"
    s = json.load(open(os.path.join(run_dir, "summary.json")))
    base, other = load(run_dir, "bonsai", cat), load(run_dir, arm, cat)
    order = [r["question_id"] for r in json.load(open(os.path.join(run_dir, f"order_{cat}.json")))["order"]]
    ids = [q for q in order if q in base and q in other]
    scored = {"ok", "budget_event"}
    sens = {}
    sp = os.path.join(run_dir, "stdin_buffer_sensitivity.json")
    if os.path.exists(sp):
        for x in json.load(open(sp))["rows"]:
            sens[(x["arm"], x["id"])] = x["with_stdin_buffer"]

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import drive
    eff = (drive.ARMS.get(arm.split("@")[0]) or {}).get("effort_sent")
    effort_note = ("Both arms send MEDIUM thinking effort upstream, so this pair is effort-matched: all "
                   "augmentation allowed vs all forced off." if eff == "medium" else
                   "The tier sends a different thinking effort upstream than bonsai (medium), so this pair mixes "
                   "effort and tools; it is a product comparison, not a tool ablation.")
    L = [f"# {arm} vs bonsai (bare @ medium), {cat}, run {s['run_id']}\n",
         f"Arms: `bonsai` = {s['arm_labels'].get('bonsai')}; `{arm}` = {s['arm_labels'].get(arm, arm)}. "
         f"{effort_note}\n"]
    L.append("## Official scores (LiveBench judge, unmodified)\n")
    L.append("| arm | score [95% CI] | n | per task |")
    L.append("|---|---|---|---|")
    for a in ("bonsai", arm):
        c = s["arms"][a]["categories"].get(cat)
        if c:
            tasks = ", ".join(f"{t} {v['score']:.1f} (n={v['n']})" for t, v in c["tasks"].items())
            L.append(f"| {a} | {c['score']:.1f} [{c['ci95'][0]:.1f}, {c['ci95'][1]:.1f}] | {c['n']} | {tasks} |")
    p = s.get(f"paired_{arm}_minus_bonsai")
    if p and cat in p["categories"]:
        v = p["categories"][cat]
        L.append(f"\n**Paired ({arm} minus bonsai), n={v['n_pairs']} questions both arms scored:** "
                 f"{v['diff']:+.1f} points [95% CI {v['ci95'][0]:+.1f}, {v['ci95'][1]:+.1f}]. "
                 f"Discordant pairs: {arm} right only {v['b_only_correct']}, bonsai right only {v['a_only_correct']}; "
                 f"exact McNemar p = {pfmt(v['mcnemar_p'])}.\n")
    # sensitivity
    def sens_score(a, rows):
        by = {}
        for q, r in rows.items():
            if r["status"] not in scored:
                continue
            sc = sens.get((a.split("@")[0], q), r["score"])
            by.setdefault(r["task"], []).append(float(sc or 0))
        tm = [100 * sum(v) / len(v) for v in by.values() if v]
        return sum(tm) / len(tm) if tm else None
    L.append("**stdin.buffer sensitivity** (LiveBench's LCB grader gives programs a StringIO stdin without `.buffer`; "
             "same answers re-graded with `.buffer` provided, `scan_stdin_buffer.py`): "
             + ", ".join(f"{a} {fmt(sens_score(a, rows))}" for a, rows in (("bonsai", base), (arm, other)))
             + f"; answers affected: "
               f"{sum(1 for (a, q) in sens if (a == 'bonsai' and q in base) or (a == arm.split('@')[0] and q in other))}. "
               "Official scores above stand.\n")

    L.append("## Per question\n")
    L.append("| # | id | task | bonsai | " + arm + " | seconds (bonsai / " + arm + ") | fan-out (mode, steps, stop, "
             "method, winner, replaced) | deep thinking (decided/ran) | repair | tool turns (turns/limit, hit) | status | "
             + arm + " timing |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    inv_decided = inv_ran = 0
    for i, q in enumerate(ids, 1):
        b, o = base[q], other[q]
        m = o.get("mechanisms") or {}
        fo, dt, rp = m.get("fanout") or {}, m.get("deep_thinking") or {}, m.get("repair") or {}
        inv_decided += bool(dt.get("decided")); inv_ran += bool(dt.get("ran"))
        fan = (f"{fo.get('mode')}, {fo.get('steps')}, {fo.get('stop_reason')}, {fo.get('method')}, "
               f"{fo.get('winner')}, {fo.get('replaced')}") if fo.get("ran") else f"not run (decided n={fo.get('decided_n')})"
        rep = f"{rp.get('rounds')} rounds, {rp.get('stopped')}" if rp.get("enabled") else "off"
        tt = m.get("tool_turns")
        tts = f"{tt['turns']}/{tt['limit']}, {tt['hit']}" if tt else "--"
        secs = f"{fmt(b.get('seconds'))} / {fmt(o.get('seconds'))}"
        L.append(f"| {i} | {q[:8]} | {o['task']} | {b['score']} | {o['score']} | {secs} | {fan} | "
                 f"{bool(dt.get('decided'))}/{bool(dt.get('ran'))} | {rep} | {tts} | {b['status']}/{o['status']} | "
                 f"{o.get('timing')} |")
    L.append(f"\nDeep thinking (investigate) decided on {inv_decided} of {len(ids)} questions and ran on {inv_ran}. "
             "Selection's reason when it did not is in `x_yamadori.selection.because.investigate` on each answer.\n")
    out = os.path.join(run_dir, f"paired_{arm}_vs_bonsai_{cat}.md")
    open(out, "w").write("\n".join(L))
    print("\n".join(L))
    print(f"\n-> {out}")
    return 0


def pfmt(p):
    return "--" if p is None else f"{p:.3f}"


def fmt(x):
    return "--" if x is None else f"{x:.1f}"


if __name__ == "__main__":
    sys.exit(main())
