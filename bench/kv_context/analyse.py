#!/usr/bin/env python
"""Apply the pre-registered gates to a bench/kv_context run.

    python bench/kv_context/analyse.py bench/kv_context/results/<run-id>

Writes verdict.json and verdict.md into the run directory. Evidence first,
then one line per gate; the gate THRESHOLDS come from the run's manifest.json
(written before the first request), never from this file, so they cannot be
moved after the data is seen. Standard library only; sends nothing.

Gates (docs/CONTEXT-EXPANSION.md s.5):

  G1 fit       the candidate loaded, and free VRAM at idle AND at the stress
               peak is >= floor_mib; no stress request failed; the guard did
               not trip.
  G2 speed     q4/q8 median ratio >= speed_min_ratio for decode at ~2k, decode
               at ~98k, and cold prefill at ~98k (wall clock through :1234).
  G3a accuracy on the rungs both arms ran (<= 151,552), pooled over (L, item,
               task): no significant paired drop (exact McNemar, two-sided,
               acc_alpha) AND q4 accuracy >= q8 accuracy - acc_max_drop_points.
  G3b          on the q4-only rungs: no rung significantly below q4's own 8k
               (paired by item x task, exact McNemar, Bonferroni over those
               rungs). The largest such rung is the usable context.
  G4 quality   LiveBench coding, q4 vs the fresh q8 control on the same
               questions: no significant paired drop (exact McNemar,
               quality_alpha) AND q4 total >= q8 total - quality_max_drop_questions.
               The cached lb-20260923-minp0 bonsai answers are a second,
               reported-only pairing.

Every result names its n. A gate with no data is "not run", never a pass.
"""
from __future__ import annotations

import json
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
LIVEBENCH_RESULTS = os.path.join(ROOT, "bench", "livebench", "results")
CACHED_Q8 = os.path.join(LIVEBENCH_RESULTS, "lb-20260923-minp0", "rows_bonsai_coding.jsonl")


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided McNemar on discordant counts (bench/livecodebench.py)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1))
    return min(1.0, 2.0 * tail / (2 ** n))


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


def rows(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    out = []
    for ln in open(path, encoding="utf-8"):
        ln = ln.strip()
        if ln:
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                pass
    return out


# ------------------------------------------------------------- gates -------
def gate_fit(fit: list[dict], floor: int) -> dict:
    per = {}
    for r in fit:
        if r.get("verdict") not in ("pass", "fail"):
            continue
        idle = (r.get("idle") or {}).get("min_free")
        peak = r.get("min_free_mib")
        per[r["c"]] = {"verdict": r["verdict"], "why": r.get("why"), "idle_free_mib": idle,
                       "peak_free_mib": peak, "peak_used_mib": r.get("peak_used_mib"),
                       "load_seconds": (r.get("load") or {}).get("load_seconds"),
                       "projected": r.get("projected"),
                       # re-derived from the numbers, not trusted from the row
                       "recheck": bool(idle is not None and peak is not None
                                       and idle >= floor and peak >= floor
                                       and all(s.get("ok") for s in r.get("stress") or [{}]))}
    passing = [c for c, v in per.items() if v["verdict"] == "pass" and v["recheck"]]
    return {"per_candidate": per, "floor_mib": floor,
            "adopt": max(passing) if passing else None,
            "status": "pass" if passing else ("fail" if per else "not run")}


def _median(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return (statistics.median(xs), len(xs)) if xs else (None, 0)


def gate_speed(speed: list[dict], ratio_min: float) -> dict:
    ok = [r for r in speed if r.get("outcome") == "measured"]
    cells = {}
    for arm in ("q8", "q4"):
        for L in sorted({r["L"] for r in ok}):
            sel = [r for r in ok if r["arm"] == arm and r["L"] == L]
            dec, nd = _median(r.get("decode_tps") for r in sel)
            pre, npf = _median(r.get("prefill_tps") for r in sel if r.get("cold"))
            cells[f"{arm}@{L}"] = {"decode_tps": dec, "n_decode": nd,
                                   "prefill_tps": pre, "n_prefill": npf}
    Ls = sorted({r["L"] for r in ok})
    checks = {}
    if Ls:
        short, long_ = Ls[0], Ls[-1]
        for name, L, metric in (("decode_short", short, "decode_tps"),
                                ("decode_long", long_, "decode_tps"),
                                ("prefill_long", long_, "prefill_tps")):
            a = cells.get(f"q8@{L}", {}).get(metric)
            b = cells.get(f"q4@{L}", {}).get(metric)
            r_ = (b / a) if (a and b) else None
            checks[name] = {"L": L, "q8": a, "q4": b, "ratio": r_,
                            "pass": None if r_ is None else r_ >= ratio_min}
    if not checks or any(v["pass"] is None for v in checks.values()):
        status = "not run"
    else:
        status = "pass" if all(v["pass"] for v in checks.values()) else "fail"
    return {"cells": cells, "checks": checks, "min_ratio": ratio_min, "status": status,
            "source": "wallclock_stream through :1234"}


def _scored(acc: list[dict]) -> dict:
    out = {}
    for r in acc:
        if r.get("outcome") in ("correct", "wrong"):
            out[(r["arm"], r["L"], r["item"], r["task"])] = bool(r["correct"])
    return out


def gate_accuracy(acc: list[dict], alpha: float, max_drop: float) -> dict:
    s = _scored(acc)
    per_L = {}
    for (arm, L, _i, _t), ok in s.items():
        d = per_L.setdefault(f"{arm}@{L}", [0, 0])
        d[0] += ok
        d[1] += 1
    table = {k: {"correct": v[0], "n": v[1], "pct": round(100 * v[0] / v[1], 1),
                 "wilson95": wilson(v[0], v[1])} for k, v in sorted(per_L.items())}
    # G3a: shared rungs, pooled paired
    q8_L = {L for (a, L, _, _) in s if a == "q8"}
    pairs = [(s[("q8", L, i, t)], s[("q4", L, i, t)]) for (a, L, i, t) in s
             if a == "q8" and ("q4", L, i, t) in s]
    b = sum(1 for x, y in pairs if x and not y)       # q8 right, q4 wrong
    c = sum(1 for x, y in pairs if y and not x)
    n = len(pairs)
    acc8 = 100 * sum(x for x, _ in pairs) / n if n else None
    acc4 = 100 * sum(y for _, y in pairs) / n if n else None
    p = mcnemar(b, c)
    g3a = {"n_pairs": n, "q8_pct": acc8 and round(acc8, 1), "q4_pct": acc4 and round(acc4, 1),
           "q8_only": b, "q4_only": c, "p": p,
           "status": "not run" if not n else (
               "fail" if (b > c and p < alpha) or (acc4 < acc8 - max_drop) else "pass")}
    per_rung = {}
    for L in sorted(q8_L):
        pr = [(s[("q8", L, i, t)], s[("q4", L, i, t)]) for (a, LL, i, t) in s
              if a == "q8" and LL == L and ("q4", L, i, t) in s]
        bb = sum(1 for x, y in pr if x and not y)
        cc = sum(1 for x, y in pr if y and not x)
        per_rung[L] = {"n": len(pr), "q8_only": bb, "q4_only": cc, "p": mcnemar(bb, cc)}
    g3a["per_rung"] = per_rung
    # G3b: q4-only rungs vs q4 at the smallest rung
    q4_L = sorted({L for (a, L, _, _) in s if a == "q4"})
    beyond = [L for L in q4_L if L not in q8_L]
    base = q4_L[0] if q4_L else None
    m = max(len(beyond), 1)
    g3b_rungs = {}
    usable = max(q8_L & set(q4_L), default=None)
    ok_so_far = True
    for L in beyond:
        pr = [(s[("q4", base, i, t)], s[("q4", L, i, t)]) for (a, LL, i, t) in s
              if a == "q4" and LL == base and ("q4", L, i, t) in s]
        bb = sum(1 for x, y in pr if x and not y)
        cc = sum(1 for x, y in pr if y and not x)
        pv = mcnemar(bb, cc)
        drop = bb > cc and pv < alpha / m
        g3b_rungs[L] = {"n": len(pr), "base_only": bb, "L_only": cc, "p": pv,
                        "bonferroni_alpha": alpha / m, "significant_drop": drop}
        if drop:
            ok_so_far = False
        if ok_so_far and pr:
            usable = L
    g3b = {"base_L": base, "rungs": g3b_rungs, "usable_context": usable,
           "status": "not run" if not g3b_rungs else (
               "pass" if not any(v["significant_drop"] for v in g3b_rungs.values()) else "fail")}
    # per-needle depth grid (multi task)
    grid = {}
    for r in acc:
        for pn in r.get("per_needle") or []:
            k = f"{r['arm']}@{r['L']}@{pn['depth']}"
            g = grid.setdefault(k, [0, 0])
            g[0] += bool(pn["ok"])
            g[1] += 1
    errors = sum(1 for r in acc if r.get("outcome") == "stack_error")
    budget = sum(1 for r in acc if r.get("outcome") == "budget")
    return {"by_arm_L": table, "g3a": g3a, "g3b": g3b, "depth_grid": grid,
            "stack_errors": errors, "budget_events": budget}


def _lb(path: str) -> dict:
    out = {}
    for r in rows(path):
        if r.get("status") in ("ok", "budget_event") and r.get("score") is not None:
            out[r["id"]] = float(r["score"])
    return out


def _pair(a: dict, b: dict, alpha: float, max_drop: int,
          labels: tuple[str, str] = ("q8", "q4")) -> dict:
    """a = the reference side, b = the tested side."""
    ids = sorted(set(a) & set(b))
    bb = sum(1 for i in ids if a[i] >= 1 and b[i] < 1)       # a right only
    cc = sum(1 for i in ids if b[i] >= 1 and a[i] < 1)       # b right only
    ta, tb = sum(a[i] for i in ids), sum(b[i] for i in ids)
    p = mcnemar(bb, cc)
    return {"labels": list(labels), "n": len(ids), "q8_total": ta, "q4_total": tb,
            "q8_only": bb, "q4_only": cc, "p": p, "status": "not run" if not ids else (
                "fail" if (bb > cc and p < alpha) or (tb < ta - max_drop) else "pass")}


def gate_quality(run_dir: str, alpha: float, max_drop: int) -> dict:
    lb = os.path.join(LIVEBENCH_RESULTS, os.path.basename(run_dir.rstrip("/\\")) + "-lb")
    q8 = _lb(os.path.join(lb, "rows_kv-q8_coding.jsonl"))
    q4 = _lb(os.path.join(lb, "rows_kv-q4_coding.jsonl"))
    cached = _lb(CACHED_Q8)
    prim = _pair(q8, q4, alpha, max_drop)
    sec = _pair(cached, q4, alpha, max_drop, ("q8 cached", "q4"))
    sec["status"] = "reported only (" + sec["status"] + ")"
    return {"primary_fresh_q8_vs_q4": prim, "secondary_cached_q8_vs_q4": sec,
            "q8_fresh_vs_cached": _pair(cached, q8, alpha, max_drop,
                                        ("q8 cached", "q8 fresh")),
            "status": prim["status"], "livebench_run": lb,
            "power_note": "n<=21: exact McNemar reaches p<0.05 only at 6-0, 7-0, 8-1 and "
                          "similar splits; this detects a collapse, not a small drop."}


def analyse(run_dir: str) -> dict:
    man = json.load(open(os.path.join(run_dir, "manifest.json"), encoding="utf-8"))
    g = man["gates"]
    fit = gate_fit(rows(os.path.join(run_dir, "fit.jsonl")), g["floor_mib"])
    speed = gate_speed(rows(os.path.join(run_dir, "speed.jsonl")), g["speed_min_ratio"])
    acc = gate_accuracy(rows(os.path.join(run_dir, "accuracy.jsonl")), g["acc_alpha"],
                        g["acc_max_drop_points"])
    qual = gate_quality(run_dir, g["quality_alpha"], g["quality_max_drop_questions"])
    statuses = {"G1 fit": fit["status"], "G2 speed": speed["status"],
                "G3a accuracy vs q8": acc["g3a"]["status"],
                "G3b accuracy beyond q8": acc["g3b"]["status"],
                "G4 quality": qual["status"]}
    if all(v == "pass" for v in statuses.values()):
        verdict = f"ADOPT -c {fit['adopt']} at q4_0/q4_0 + mean-centering"
    elif any(v == "fail" for v in statuses.values()):
        verdict = "DO NOT ADOPT: stay at q8_0 -c 163840 (or re-run at the smaller candidate)"
    else:
        verdict = "INCOMPLETE: at least one gate has no data; no decision"
    return {"run_dir": run_dir, "gates": g, "statuses": statuses, "verdict": verdict,
            "fit": fit, "speed": speed, "accuracy": acc, "quality": qual,
            "adopted_candidate": fit["adopt"]}


def render(v: dict) -> str:
    L = [f"# KV context trial -- {os.path.basename(v['run_dir'])}", "",
         f"**{v['verdict']}**", "", "| gate | status |", "|---|---|"]
    L += [f"| {k} | {s} |" for k, s in v["statuses"].items()]
    L += ["", f"Pre-registered thresholds (from manifest.json): `{json.dumps(v['gates'])}`", "",
          "## G1 fit", "", "| -c | verdict | idle free MiB | peak free MiB | load s | why |",
          "|---|---|---|---|---|---|"]
    for c, r in sorted(v["fit"]["per_candidate"].items(), reverse=True):
        L.append(f"| {c} | {r['verdict']} | {r['idle_free_mib']} | {r['peak_free_mib']} | "
                 f"{r['load_seconds']} | {r['why']} |")
    L += ["", "## G2 speed (wall clock through :1234)", "",
          "| check | L | q8 | q4 | ratio | pass |", "|---|---|---|---|---|---|"]
    for k, r in v["speed"]["checks"].items():
        f = (lambda x: f"{x:.1f}" if isinstance(x, (int, float)) else "--")
        L.append(f"| {k} | {r['L']} | {f(r['q8'])} | {f(r['q4'])} | {f(r['ratio'])} | {r['pass']} |")
    a = v["accuracy"]
    L += ["", "## G3 long-context accuracy (synthetic codebase, exact match)", "",
          "| arm@L | correct/n | % | Wilson 95% |", "|---|---|---|---|"]
    for k, r in a["by_arm_L"].items():
        L.append(f"| {k} | {r['correct']}/{r['n']} | {r['pct']} | {r['wilson95']} |")
    g3a = a["g3a"]
    L += ["", f"G3a pooled paired, n={g3a['n_pairs']}: q8 {g3a['q8_pct']}% vs q4 {g3a['q4_pct']}%, "
          f"q8-only {g3a['q8_only']}, q4-only {g3a['q4_only']}, exact McNemar p={g3a['p']:.4f} "
          f"-> {g3a['status']}"]
    g3b = a["g3b"]
    L += [f"G3b q4 beyond q8 vs q4@{g3b['base_L']}: " + "; ".join(
        f"{Lr}: n={r['n']} base-only {r['base_only']} L-only {r['L_only']} p={r['p']:.4f}"
        f"{' DROP' if r['significant_drop'] else ''}" for Lr, r in g3b["rungs"].items())
        + f" -> usable context {g3b['usable_context']} ({g3b['status']})",
        f"Stack errors {a['stack_errors']}, budget events {a['budget_events']} (neither is scored)."]
    q = v["quality"]
    L += ["", "## G4 quality (LiveBench coding, bare @ medium, check_code/repair off)", ""]
    for k in ("primary_fresh_q8_vs_q4", "secondary_cached_q8_vs_q4", "q8_fresh_vs_cached"):
        r = q[k]
        ra, rb = r["labels"]
        L.append(f"- {k}: n={r['n']}, {ra} {r['q8_total']:g} vs {rb} {r['q4_total']:g}, "
                 f"{ra}-only {r['q8_only']}, {rb}-only {r['q4_only']}, p={r['p']:.4f} "
                 f"-> {r['status']}")
    L += ["", f"_{q['power_note']}_"]
    return "\n".join(L) + "\n"


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    run_dir = argv[0]
    v = analyse(run_dir)
    with open(os.path.join(run_dir, "verdict.json"), "w", encoding="utf-8") as f:
        json.dump(v, f, indent=2, default=str)
    md = render(v)
    with open(os.path.join(run_dir, "verdict.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
