#!/usr/bin/env python
"""Speed and accuracy vs context length, per model/config, from run.py rows.

    python bench/longctx/analyse.py                       # every run under results/
    python bench/longctx/analyse.py results/A results/B   # compare configs
    python bench/longctx/analyse.py results/A --json      # print the JSON

Writes summary.json and summary.md into the run directory (one run) or into
results/_compare/ (several).

WHAT IS COUNTED

The LAST row per key (L, task, item, seed) is the outcome: a retry after a
stack error replaces it. Accuracy is over SCORED rows only (correct / wrong).
`stack_error` rows (the server failed) and `budget` rows (finish_reason
`length` with no answer) are counted and reported beside every cell, never
scored (PROTOCOL rule 3; AGENTS.md: a length finish is a budget event).

SPEED is reported per rung as median and range over the speed rows, with the
source named -- `server_timings` (llama-server's own prompt_per_second /
predicted_per_second) or `wallclock_stream` (the proxy; includes its
overhead). The two are never pooled.

USABLE CONTEXT -- the test, stated before the data (PROTOCOL rule 4)

Reference rung: 8k (8192) if present, else the smallest rung at or above it.
For every longer rung L and each task, items are PAIRED (the same item at both
lengths: same needles, depths and question; bench/longctx/haystack.py), and
the discordant pairs enter an exact McNemar test (livecodebench.mcnemar,
two-sided, the implementation the rest of bench/ uses). A rung is a
significant DROP when the discordance runs against the longer context (ref
right / L wrong outnumbers the reverse) and p < alpha / m, alpha = 0.05,
Bonferroni over the m longer rungs compared for that task.

    usable(task)  = the largest rung L such that neither L nor any rung
                    between the reference and L is a significant drop
    usable        = min over tasks (the headline)

The power this has is printed with it: `min_discordant` is the fewest
discordant pairs that could reach alpha/m even if all fell one way, so the
smallest drop the test can detect at n paired items is min_discordant / n --
at n=20 that is 8/20 = 40 points with m=5 longer rungs (a 163,840 server) and
9/20 = 45 points with m=9 (a 262,144 server); uncorrected (alpha 0.05) it is
6/20 = 30 points. A rung that is "not significantly below" at that n has not
been shown equal: the Wilson intervals beside it say how wide the ignorance is.
Early warnings (uncorrected p < 0.05, same direction) are listed separately.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
ROOT = os.path.dirname(BENCH)
RESULTS = os.path.join(HERE, "results")
sys.path.insert(0, BENCH)

from livecodebench import mcnemar, wilson  # noqa: E402

ALPHA = 0.05
REF_L = 8192
TASKS = ("single", "multi", "reason")
DEPTH_BINS = (0.1, 0.3, 0.5, 0.7, 0.9)
SCORED = ("correct", "wrong")


# ------------------------------------------------------------- loading -----
def load_run(run_dir: str) -> dict:
    rows, bad = [], 0
    path = os.path.join(run_dir, "rows.jsonl")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    rows.append(json.loads(ln))
                except json.JSONDecodeError:
                    bad += 1
    man = {}
    mp = os.path.join(run_dir, "manifest.json")
    if os.path.exists(mp):
        with open(mp, encoding="utf-8") as f:
            man = json.load(f)
    last: dict[str, dict] = {}
    for r in rows:
        if r.get("key"):
            last[r["key"]] = r
    label = man.get("label") or os.path.basename(run_dir.rstrip("/\\"))
    return {"dir": run_dir, "label": label, "manifest": man, "rows": rows,
            "last": last, "bad": bad, "path": path}


# ------------------------------------------------------------- stats -------
def min_discordant(alpha: float, cap: int = 400) -> int | None:
    for d in range(1, cap + 1):
        if mcnemar(d, 0) < alpha:
            return d
    return None


def _med(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return statistics.median(xs) if xs else None


def _r(x, nd=1):
    return None if x is None else round(x, nd)


def lengths(last: dict) -> list[int]:
    return sorted({r["L"] for r in last.values() if isinstance(r.get("L"), int)})


def is_speed(r: dict) -> bool:
    return str(r.get("task") or "").startswith("speed")


def is_cold(r: dict) -> bool:
    """A prefill measurement: the rep's first request (fresh nonce), and the
    server confirms it reused (almost) nothing -- the chat template's opening
    tokens are all a fresh nonce can share."""
    if r.get("cold") is False:
        return False
    cn = r.get("cache_n")
    return cn is None or cn <= 64


def _pooled(rs: list[dict]) -> dict:
    """Draft acceptance over rows: pooled (sum accepted / sum drafted) and
    the median of per-request rates. None when nothing was drafted."""
    d = [r for r in rs if r.get("draft_n")]
    if not d:
        return {"rate": None, "median": None, "n": 0, "drafted": 0}
    dn = sum(r["draft_n"] for r in d)
    da = sum(r.get("draft_n_accepted") or 0 for r in d)
    return {"rate": _r(da / dn, 3) if dn else None,
            "median": _r(_med([r.get("draft_accept_rate") for r in d]), 3),
            "n": len(d), "drafted": dn}


def speed_table(last: dict) -> list[dict]:
    out = []
    for L in lengths(last):
        rs = [r for r in last.values() if r.get("L") == L and is_speed(r)]
        ok = [r for r in rs if r.get("outcome") == "measured"]
        cold = [r for r in ok if is_cold(r)]
        srcs = sorted({r.get("speed_source") or "none" for r in ok})
        pre = [r.get("prefill_tps") for r in cold if r.get("prefill_tps") is not None]
        dec = [r.get("decode_tps") for r in ok if r.get("decode_tps") is not None]
        by_content = {}
        for c in sorted({r.get("content_type") or "prose" for r in ok}):
            cr = [r for r in ok if (r.get("content_type") or "prose") == c]
            cd = [r.get("decode_tps") for r in cr if r.get("decode_tps") is not None]
            by_content[c] = {"n": len(cr), "decode_tps": _r(_med(cd), 2),
                             "decode_min": _r(min(cd) if cd else None, 2),
                             "decode_max": _r(max(cd) if cd else None, 2),
                             "draft": _pooled(cr)}
        out.append({
            "L": L, "n": len(ok), "n_prefill": len(pre), "errors": len(rs) - len(ok),
            "L_actual": _med([r.get("prompt_tokens") for r in cold or ok]),
            "speed_source": srcs[0] if len(srcs) == 1 else "+".join(srcs),
            "prefill_tps": _r(_med(pre)), "prefill_min": _r(min(pre) if pre else None),
            "prefill_max": _r(max(pre) if pre else None),
            "decode_tps": _r(_med(dec), 2), "decode_min": _r(min(dec) if dec else None, 2),
            "decode_max": _r(max(dec) if dec else None, 2),
            "decode_n_tokens": _med([r.get("predicted_n") for r in ok]),
            "draft_accept_rate": _pooled(ok)["rate"],
            "by_content": by_content,
            "vram_peak_mib": max([r.get("vram_peak_mib") or 0 for r in rs] or [0]) or None,
            "seconds_median": _r(_med([r.get("seconds") for r in ok])),
        })
    return out


def draft_table(last: dict) -> dict:
    """Draft (MTP) acceptance by content type and context length: the speed
    rows' three content types plus the accuracy answers (`answer:<task>`).
    Every cell is null when the server did not speculate."""
    rows = [r for r in last.values()
            if r.get("outcome") in ("measured",) + SCORED + ("budget",)]
    contents = sorted({r.get("content_type") or ("prose" if is_speed(r) else
                                                  f"answer:{r.get('task')}")
                       for r in rows})
    table = []
    for L in lengths(last):
        cells = {}
        for c in contents:
            cr = [r for r in rows if r.get("L") == L and
                  (r.get("content_type") or ("prose" if is_speed(r)
                                             else f"answer:{r.get('task')}")) == c]
            cells[c] = _pooled(cr)
        table.append({"L": L, "by_content": cells,
                      "all": _pooled([r for r in rows if r.get("L") == L])})
    return {"speculating": any(r.get("draft_n") for r in rows),
            "contents": contents, "table": table,
            "note": ("pooled = accepted / drafted over the cell's requests; source "
                     "is the reply's timings, else the /metrics delta; null = no "
                     "tokens drafted (speculation off)")}


def acc_cell(last: dict, L: int, task: str) -> dict:
    rs = [r for r in last.values() if r.get("L") == L and r.get("task") == task]
    sc = [r for r in rs if r.get("outcome") in SCORED]
    k = sum(1 for r in sc if r.get("outcome") == "correct")
    n = len(sc)
    lo, hi = wilson(k, n)
    return {"L": L, "task": task, "k": k, "n": n,
            "rate": (k / n) if n else None, "lo": lo if n else None,
            "hi": hi if n else None,
            "budget": sum(1 for r in rs if r.get("outcome") == "budget"),
            "stack_errors": sum(1 for r in rs if r.get("outcome") == "stack_error"),
            "fallback_parse": sum(1 for r in sc if r.get("parse") == "fallback"),
            "L_actual": _med([r.get("prompt_tokens") for r in sc])}


def accuracy_table(last: dict, tasks=TASKS) -> dict:
    return {t: [acc_cell(last, L, t) for L in lengths(last)
                if any(r.get("task") == t and r.get("L") == L for r in last.values())]
            for t in tasks}


def depth_grid(last: dict) -> dict:
    """Recall by (L, depth bin): from every needle of the multi question
    (4 per item) and, separately, the single question."""
    grid = {"multi_per_needle": [], "single": []}
    for L in lengths(last):
        cells_m, cells_s = {}, {}
        for r in last.values():
            if r.get("L") != L or r.get("outcome") not in SCORED:
                continue
            if r.get("task") == "multi":
                for pn in r.get("per_needle") or []:
                    c = cells_m.setdefault(pn["depth"], [0, 0])
                    c[0] += 1 if pn["ok"] else 0
                    c[1] += 1
            elif r.get("task") == "single":
                d = (r.get("depths") or [None])[0]
                c = cells_s.setdefault(d, [0, 0])
                c[0] += 1 if r.get("outcome") == "correct" else 0
                c[1] += 1
        for name, cells in (("multi_per_needle", cells_m), ("single", cells_s)):
            row = {"L": L}
            for d in DEPTH_BINS:
                k, n = cells.get(d, [0, 0])
                row[f"{int(d * 100)}%"] = {"k": k, "n": n,
                                            "rate": (k / n) if n else None}
            grid[name].append(row)
    return grid


def ref_length(Ls: list[int], ref: int = REF_L) -> int | None:
    if not Ls:
        return None
    if ref in Ls:
        return ref
    above = [L for L in Ls if L >= ref]
    return above[0] if above else Ls[0]


def paired(last: dict, task: str, L1: int, L2: int, seed=None) -> tuple[int, int, int]:
    """(b, c, n_paired): b = right at L1 and wrong at L2; c = the reverse."""
    def idx(L):
        out = {}
        for r in last.values():
            if r.get("L") == L and r.get("task") == task and r.get("outcome") in SCORED:
                out[(r.get("item"), r.get("seed"))] = r.get("outcome") == "correct"
        return out
    a, z = idx(L1), idx(L2)
    both = set(a) & set(z)
    b = sum(1 for k in both if a[k] and not z[k])
    c = sum(1 for k in both if z[k] and not a[k])
    return b, c, len(both)


def usable_context(last: dict, tasks=TASKS, alpha: float = ALPHA,
                   ref: int = REF_L) -> dict:
    Ls = lengths(last)
    R = ref_length(Ls, ref)
    out = {"ref_L": R, "alpha": alpha, "test": (
        "exact McNemar (two-sided, livecodebench.mcnemar) on items paired "
        "across lengths; a drop needs ref-right/L-wrong > the reverse and "
        "p < alpha/m (Bonferroni over the m longer rungs)"),
        "by_task": {}, "usable": None, "early_warnings": []}
    if R is None:
        return out
    usable_all = []
    for t in tasks:
        longer = [L for L in Ls if L > R and paired(last, t, R, L)[2] > 0]
        m = max(len(longer), 1)
        thr = alpha / m
        dmin = min_discordant(thr)
        tests, usable, broken = [], R if paired(last, t, R, R)[2] else None, False
        for L in longer:
            b, c, n = paired(last, t, R, L)
            p = mcnemar(b, c)
            drop = b > c and p < thr
            warn = b > c and p < alpha
            tests.append({"L": L, "n_paired": n, "ref_right_L_wrong": b,
                          "ref_wrong_L_right": c, "p": p, "significant_drop": drop,
                          "uncorrected_warning": warn and not drop,
                          "min_detectable_drop": (dmin / n) if (dmin and n and dmin <= n) else None})
            if warn and not drop:
                out["early_warnings"].append({"task": t, "L": L, "p": p, "b": b, "c": c})
            if drop:
                broken = True
            if not broken:
                usable = L
        out["by_task"][t] = {"usable": usable, "m": len(longer),
                             "alpha_per_test": thr, "min_discordant": dmin,
                             "tests": tests}
        if usable is not None:
            usable_all.append(usable)
    out["usable"] = min(usable_all) if usable_all else None
    return out


def power_note(n: int, m: int, p_ref: float | None, alpha: float = ALPHA) -> dict:
    """What an n-item paired comparison can see (PROTOCOL rule 4)."""
    d_corr = min_discordant(alpha / max(m, 1))
    d_unc = min_discordant(alpha)
    half = None
    if p_ref is not None and n:
        lo, hi = wilson(round(p_ref * n), n)
        half = (hi - lo) / 2
    return {"n_paired": n, "m": m,
            "min_discordant_corrected": d_corr,
            "min_detectable_drop_corrected": (d_corr / n) if (d_corr and n and d_corr <= n) else None,
            "min_discordant_uncorrected": d_unc,
            "min_detectable_drop_uncorrected": (d_unc / n) if (d_unc and n and d_unc <= n) else None,
            "wilson_halfwidth_at_ref": half}


def compare(a: dict, b: dict, tasks=TASKS) -> list[dict]:
    """Config B vs config A at each shared rung: same seed and items, so the
    items pair; exact McNemar, Bonferroni over the cells compared."""
    out = []
    La, Lb = set(lengths(a["last"])), set(lengths(b["last"]))
    cells = []
    for L in sorted(La & Lb):
        for t in tasks:
            ia = {(r["item"], r["seed"]): r["outcome"] == "correct"
                  for r in a["last"].values()
                  if r.get("L") == L and r.get("task") == t and r.get("outcome") in SCORED}
            ib = {(r["item"], r["seed"]): r["outcome"] == "correct"
                  for r in b["last"].values()
                  if r.get("L") == L and r.get("task") == t and r.get("outcome") in SCORED}
            both = set(ia) & set(ib)
            if not both:
                continue
            a_only = sum(1 for k in both if ia[k] and not ib[k])
            b_only = sum(1 for k in both if ib[k] and not ia[k])
            cells.append((L, t, len(both), a_only, b_only))
    m = max(len(cells), 1)
    for L, t, n, ao, bo in cells:
        p = mcnemar(ao, bo)
        out.append({"L": L, "task": t, "n_paired": n, "a_only": ao, "b_only": bo,
                    "p": p, "p_bonferroni": min(1.0, p * m),
                    "family": m})
    return out


def speed_ratio(a: dict, b: dict) -> list[dict]:
    sa = {r["L"]: r for r in speed_table(a["last"])}
    sb = {r["L"]: r for r in speed_table(b["last"])}
    out = []
    for L in sorted(set(sa) & set(sb)):
        x, y = sa[L], sb[L]
        out.append({"L": L,
                    "prefill_ratio": (y["prefill_tps"] / x["prefill_tps"])
                    if x["prefill_tps"] and y["prefill_tps"] else None,
                    "decode_ratio": (y["decode_tps"] / x["decode_tps"])
                    if x["decode_tps"] and y["decode_tps"] else None,
                    "same_source": x["speed_source"] == y["speed_source"]})
    return out


# ------------------------------------------------------------- per config --
def summarise(run: dict) -> dict:
    last = run["last"]
    man = run["manifest"]
    tasks = [t for t in TASKS if any(r.get("task") == t for r in last.values())]
    spd = speed_table(last)
    acc = accuracy_table(last, tasks)
    use = usable_context(last, tasks)
    R = use["ref_L"]
    caveats = []
    srcs = {s["speed_source"] for s in spd if s["n"]}
    if any("wallclock" in s for s in srcs):
        caveats.append("speed is wall-clock through the proxy (includes proxy "
                       "overhead; decode length is the model's) -- a smoke "
                       "number, not a characterisation")
    for t, cells in acc.items():
        rates = [c["rate"] for c in cells if c["n"]]
        if len(rates) > 1 and len(set(rates)) == 1:
            caveats.append(f"{t}: identical accuracy at every length "
                           f"({rates[0]:.2f}) -- check the harness before "
                           f"believing it (PROTOCOL rule 3)")
        for c in cells:
            if c["stack_errors"]:
                caveats.append(f"{t} L={c['L']}: {c['stack_errors']} stack "
                               f"error(s), excluded from n")
            if c["budget"]:
                caveats.append(f"{t} L={c['L']}: {c['budget']} budget event(s) "
                               f"(length, no answer), excluded from n")
            if c["L_actual"] and abs(c["L_actual"] / c["L"] - 1) > 0.10:
                caveats.append(f"{t} L={c['L']}: actual prompt "
                               f"{c['L_actual']:.0f} tokens is >10% off target")
    st = str(man.get("spec_types") or "").lower()
    flags = str(man.get("flags") or "")
    if (("--spec-type" in flags) or (st and st not in ("none", "[]", "")))\
            and not any(r.get("draft_n") for r in last.values()):
        caveats.append(f"the server was launched to speculate ({man.get('spec_types')!r}"
                       f"{', --spec-type in flags' if '--spec-type' in flags else ''}) "
                       "but no row recorded a drafted token -- MTP may not be active, "
                       "or neither timings nor /metrics (--metrics) expose it")
    n_ref = {t: next((c["n"] for c in acc[t] if c["L"] == R), 0) for t in tasks}
    p_ref = {t: next((c["rate"] for c in acc[t] if c["L"] == R), None) for t in tasks}
    power = {t: power_note(n_ref[t], use["by_task"].get(t, {}).get("m", 0), p_ref[t])
             for t in tasks}
    mode = man.get("mode")
    return {
        "label": run["label"], "run_dir": os.path.relpath(run["dir"], ROOT).replace("\\", "/"),
        "model": man.get("model_id") or man.get("model_path"),
        "model_path": man.get("model_path"), "build": man.get("build"),
        "n_ctx": man.get("n_ctx"), "limit": man.get("limit"),
        "limit_why": man.get("limit_why"), "mode": mode,
        "thinking": ("proxy:" + json.dumps(man.get("features"), sort_keys=True)
                     if mode == "proxy" else man.get("thinking")),
        "flags": man.get("flags"), "skipped_lengths": man.get("skipped_lengths"),
        "calibration": man.get("calibration"),
        "rows": len(run["rows"]), "keys": len(last), "bad_lines": run["bad"],
        "speed": spd, "accuracy": acc, "depth_grid": depth_grid(last),
        "draft": draft_table(last), "spec_types": man.get("spec_types"),
        "usable_context": use, "power": power, "caveats": caveats,
    }


def dashboard_block(sums: list[dict], pairs: list[dict], out_path: str) -> dict:
    """Shaped like bench/domain/analyse.py dashboard_block (state / file /
    how / arms / progress / power / arm_table / pairs / note), with the
    long-context curves added under `curves` and `depth_grid`."""
    arms = [s["label"] for s in sums]
    table = []
    for s in sums:
        spd = [x for x in s["speed"] if x["n"]]
        first, lastr = (spd[0], spd[-1]) if spd else ({}, {})
        R = s["usable_context"]["ref_L"]
        ref_acc = {t: next((c for c in cells if c["L"] == R), None)
                   for t, cells in s["accuracy"].items()}
        top_acc = {t: (cells[-1] if cells else None) for t, cells in s["accuracy"].items()}
        table.append({
            "arm": s["label"], "model": s["model"], "build": s["build"],
            "mode": s["mode"], "thinking": s["thinking"], "n_ctx": s["n_ctx"],
            "usable_context": s["usable_context"]["usable"], "ref_L": R,
            "acc_at_ref": {t: c and {k: c[k] for k in ("k", "n", "rate", "lo", "hi")}
                           for t, c in ref_acc.items()},
            "acc_at_longest": {t: c and {k: c[k] for k in ("L", "k", "n", "rate", "lo", "hi")}
                               for t, c in top_acc.items()},
            "prefill_tps_first": first.get("prefill_tps"),
            "prefill_tps_longest": lastr.get("prefill_tps"),
            "decode_tps_first": first.get("decode_tps"),
            "decode_tps_longest": lastr.get("decode_tps"),
            "speed_source": first.get("speed_source"),
            "speculating": s["draft"]["speculating"],
            "draft_accept_first": first.get("draft_accept_rate"),
            "draft_accept_longest": lastr.get("draft_accept_rate"),
            "errors": sum(c["stack_errors"] for cells in s["accuracy"].values() for c in cells)
            + sum(x["errors"] for x in s["speed"]),
        })
    extreme = [x["arm"] for x in table
               for t, c in x["acc_at_ref"].items() if c and c["rate"] in (0.0, 1.0)]
    rel = os.path.relpath(out_path, ROOT).replace("\\", "/")
    return {
        "state": "ready" if any(s["rows"] for s in sums) else "empty",
        "file": rel, "exists": os.path.exists(out_path),
        "file_age_s": (round(time.time() - os.path.getmtime(out_path))
                       if os.path.exists(out_path) else None),
        "bad_lines": sum(s["bad_lines"] for s in sums),
        "how": "run  python bench/longctx/run.py --url URL --run-id ID; "
               "then python bench/longctx/analyse.py",
        "arms": arms,
        "suspect": {"extreme_at_ref": sorted(set(extreme))},
        "progress": {s["label"]: {"rows": s["rows"], "keys": s["keys"]} for s in sums},
        "power": {s["label"]: s["power"] for s in sums},
        "arm_table": table,
        "curves": {s["label"]: {"speed": s["speed"], "accuracy": s["accuracy"]}
                   for s in sums},
        "draft_acceptance": {s["label"]: s["draft"] for s in sums},
        "depth_grid": {s["label"]: s["depth_grid"] for s in sums},
        "usable": {s["label"]: s["usable_context"] for s in sums},
        "pairs": pairs,
        "note": ("usable context = largest rung not significantly below the 8k "
                 "rung (exact McNemar on paired items, alpha 0.05 Bonferroni "
                 "over longer rungs); speed from llama-server timings unless "
                 "labelled wallclock_stream"),
    }


# ------------------------------------------------------------- markdown ----
def _pct(r):
    return "-" if r is None else f"{100 * r:.0f}%"


def _det(r):
    return "none (n below min discordant)" if r is None else _pct(r)


def _k(L):
    return f"{L // 1024}k" if L % 1024 == 0 else str(L)


def markdown(sums: list[dict], pairs: list[dict], ratios: list[dict]) -> str:
    L = ["# Long-context benchmark", "",
         f"Analysed {time.strftime('%Y-%m-%d %H:%M')}. Accuracy counts scored rows "
         "only; stack errors and budget events are listed, never scored.", ""]
    for s in sums:
        L += [f"## {s['label']}", "",
              f"- model `{s['model']}`, build `{s['build']}`, n_ctx {s['n_ctx']}, "
              f"mode **{s['mode']}**, thinking `{s['thinking']}`",
              f"- ladder limit {s['limit']} ({s['limit_why']})"
              + (f"; not run (over the limit): {[_k(x) for x in s['skipped_lengths']]}"
                 if s.get("skipped_lengths") else ""),
              f"- flags: `{s['flags']}`" if s.get("flags") else "- flags: not recorded",
              ""]
        L += ["### Speed", "",
              "| L | L actual | n | prefill tok/s (min-max) | decode tok/s (min-max) "
              "| decode n | draft accept | VRAM peak MiB | source |",
              "|---|---|---|---|---|---|---|---|---|"]
        for x in s["speed"]:
            L.append(f"| {_k(x['L'])} | {x['L_actual'] or '-'} | {x['n']}"
                     + (f" (+{x['errors']} err)" if x["errors"] else "")
                     + f" | {x['prefill_tps']} ({x['prefill_min']}-{x['prefill_max']}, "
                     f"n={x['n_prefill']} cold)"
                     f" | {x['decode_tps']} ({x['decode_min']}-{x['decode_max']})"
                     f" | {x['decode_n_tokens']} | {x['draft_accept_rate'] if x['draft_accept_rate'] is not None else '-'}"
                     f" | {x['vram_peak_mib'] or '-'} | {x['speed_source']} |")
        contents = sorted({c for x in s["speed"] for c in x["by_content"]})
        if contents:
            L += ["", "Decode tok/s by content type (median; draft acceptance "
                  "pooled, null when not speculating):", "",
                  "| L | " + " | ".join(contents) + " |",
                  "|---|" + "---|" * len(contents)]
            for x in s["speed"]:
                cols = []
                for c in contents:
                    b = x["by_content"].get(c)
                    if not b:
                        cols.append("-")
                        continue
                    acc = b["draft"]["rate"]
                    cols.append(f"{b['decode_tps']} t/s (n={b['n']})"
                                + (f", accept {acc:.2f}" if acc is not None else ""))
                L.append(f"| {_k(x['L'])} | " + " | ".join(cols) + " |")
        dt = s["draft"]
        L += ["", "### Draft acceptance by content and context depth", ""]
        if not dt["speculating"]:
            L.append("No tokens were drafted: speculation off (every cell null).")
        else:
            L += ["| L | " + " | ".join(dt["contents"]) + " | all |",
                  "|---|" + "---|" * (len(dt["contents"]) + 1)]
            for row in dt["table"]:
                cells = [row["by_content"][c] for c in dt["contents"]] + [row["all"]]
                L.append(f"| {_k(row['L'])} | " + " | ".join(
                    "-" if c["rate"] is None else f"{c['rate']:.3f} (n={c['n']})"
                    for c in cells) + " |")
            L.append("")
            L.append(dt["note"])
        L += ["", "### Accuracy (Wilson 95%)", "",
              "| L | " + " | ".join(s["accuracy"]) + " |",
              "|---|" + "---|" * len(s["accuracy"])]
        allL = sorted({c["L"] for cells in s["accuracy"].values() for c in cells})
        for Lx in allL:
            cols = []
            for t, cells in s["accuracy"].items():
                c = next((c for c in cells if c["L"] == Lx), None)
                if not c or not c["n"]:
                    cols.append("-")
                    continue
                extra = []
                if c["stack_errors"]:
                    extra.append(f"{c['stack_errors']} err")
                if c["budget"]:
                    extra.append(f"{c['budget']} budget")
                cols.append(f"{c['k']}/{c['n']} {_pct(c['rate'])} "
                            f"[{_pct(c['lo'])}-{_pct(c['hi'])}]"
                            + (f" ({', '.join(extra)})" if extra else ""))
            L.append(f"| {_k(Lx)} | " + " | ".join(cols) + " |")
        u = s["usable_context"]
        L += ["", "### Usable context", "",
              f"Reference {_k(u['ref_L']) if u['ref_L'] else '-'}; {u['test']}; "
              f"alpha {u['alpha']}.", "",
              f"**Usable context: {_k(u['usable']) if u['usable'] else 'n/a'}** "
              "(min over tasks)", ""]
        for t, bt in u["by_task"].items():
            pw = s["power"].get(t, {})
            L.append(f"- {t}: usable {_k(bt['usable']) if bt['usable'] else 'n/a'}; "
                     f"m={bt['m']}, alpha/m={bt['alpha_per_test']:.4f}, "
                     f"min discordant {bt['min_discordant']}; at n={pw.get('n_paired')} "
                     f"the smallest detectable drop is "
                     f"{_det(pw.get('min_detectable_drop_corrected'))} corrected, "
                     f"{_det(pw.get('min_detectable_drop_uncorrected'))} uncorrected")
            for tt in bt["tests"]:
                flag = ("SIGNIFICANT DROP" if tt["significant_drop"] else
                        "uncorrected warning" if tt["uncorrected_warning"] else "")
                L.append(f"  - {_k(tt['L'])}: n={tt['n_paired']}, ref-right/L-wrong "
                         f"{tt['ref_right_L_wrong']}, reverse {tt['ref_wrong_L_right']}, "
                         f"p={tt['p']:.3g} {flag}")
        g = s["depth_grid"]["multi_per_needle"]
        if g:
            L += ["", "### Depth x length (multi, per needle)", "",
                  "| L | " + " | ".join(f"{int(d * 100)}%" for d in DEPTH_BINS) + " |",
                  "|---|" + "---|" * len(DEPTH_BINS)]
            for row in g:
                L.append(f"| {_k(row['L'])} | " + " | ".join(
                    f"{row[f'{int(d * 100)}%']['k']}/{row[f'{int(d * 100)}%']['n']}"
                    for d in DEPTH_BINS) + " |")
        if s["caveats"]:
            L += ["", "### Caveats", ""] + [f"- {c}" for c in s["caveats"]]
        L.append("")
    if pairs:
        L += ["## Config comparison (paired by item, exact McNemar)", "",
              "| a | b | L | task | n | a only | b only | p | p (Bonferroni) |",
              "|---|---|---|---|---|---|---|---|---|"]
        for p in pairs:
            L.append(f"| {p['a']} | {p['b']} | {_k(p['L'])} | {p['task']} | {p['n_paired']} "
                     f"| {p['a_only']} | {p['b_only']} | {p['p']:.3g} | {p['p_bonferroni']:.3g} |")
        L.append("")
    if ratios:
        L += ["## Speed ratios (b / a, medians)", "",
              "| a | b | L | prefill | decode | same source |", "|---|---|---|---|---|---|"]
        def _x(v):
            return "-" if v is None else f"{v:.2f}"
        for r in ratios:
            L.append(f"| {r['a']} | {r['b']} | {_k(r['L'])} | {_x(r['prefill_ratio'])} | "
                     f"{_x(r['decode_ratio'])} | {r['same_source']} |")
    return "\n".join(L) + "\n"


def analyse(run_dirs: list[str]) -> tuple[dict, str, str]:
    runs = [load_run(d) for d in run_dirs]
    sums = [summarise(r) for r in runs]
    pairs, ratios = [], []
    for i in range(1, len(runs)):
        for p in compare(runs[0], runs[i]):
            pairs.append(dict(p, a=runs[0]["label"], b=runs[i]["label"]))
        for r in speed_ratio(runs[0], runs[i]):
            ratios.append(dict(r, a=runs[0]["label"], b=runs[i]["label"]))
    out_dir = run_dirs[0] if len(run_dirs) == 1 else os.path.join(RESULTS, "_compare")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "summary.json")
    summary = {"analysed_at": time.time(), "runs": [s["run_dir"] for s in sums],
               "configs": sums, "pairs": pairs, "speed_ratios": ratios}
    summary["dashboard"] = dashboard_block(sums, pairs, out_path)
    md = markdown(sums, pairs, ratios)
    return summary, md, out_dir


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="*", help="run directories (default: all under results/)")
    ap.add_argument("--json", action="store_true", help="print summary JSON")
    args = ap.parse_args(argv)
    dirs = args.runs or sorted(d for d in glob.glob(os.path.join(RESULTS, "*"))
                               if os.path.exists(os.path.join(d, "rows.jsonl")))
    if not dirs:
        print("  no run directories with rows under bench/longctx/results/")
        return 1
    summary, md, out_dir = analyse(dirs)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(json.dumps(summary, indent=2) if args.json else md)
    print(f"  wrote {os.path.relpath(out_dir, ROOT)}/summary.json and summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
