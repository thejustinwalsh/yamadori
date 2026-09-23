#!/usr/bin/env python
"""Paired analysis of a bench/domain/run.py run: each arm against A0, and A5
against A6 (the `fixed` floor), by exact McNemar with Bonferroni.

    python bench/domain/analyse.py bench/domain/results/<run_id>
    python bench/domain/analyse.py            # the newest run under results/

Writes <run_dir>/summary.json and <run_dir>/summary.md.

WHAT IS PAIRED

The last row per (task, arm) is the outcome (a retry after a stack error
replaces it). A comparison X vs Y runs on the tasks where BOTH last rows are
pass or fail; stack errors are excluded, counted per arm, and reported, never
scored (PROTOCOL rule 3). Tasks both arms solve, or both miss, carry no
information about the difference; only the discordant pairs b (Y solved, X
missed) and c (X solved, Y missed) enter the exact test (rule 6).

WHAT IS HEADLINED

The headline comparisons are on the UNCONTAMINATED tasks only. A task is
contaminated when its row says so: three_tsl (three.js, in every training
set) and typescript_tc (type-challenges, public since 2020). They are
reported in their own block and never pooled into a headline number (rule
11). An all-domains block is also given, labelled as containing them.

There are three Bonferroni families, because they are three experiments:
`augmentation` -- each A-arm present vs A0, plus A5 vs A6 --
`thinking_cap` -- C8 and C128 vs C32 (every C arm sets its cap explicitly;
C32 is the reference) -- and `self_check` -- each S arm vs its one-shot twin
(S0 vs A0, S5 vs A5, S6 vs A6) and each S arm other than S0 vs A0. Each
family's size is printed beside every p.

WHAT ELSE IS MEASURED, PER ARM

Where each failure died (extract / compile / test), seconds (mean and
median, whole conversation), prompt and completion tokens summed over every
round, the proxy's own tool hops (x_yamadori.hops - 1 per request, i.e.
retrieval calls the model made) and deep-thinking hops, and for S arms:
check rounds, how often the answer it finally gave passed the public check,
and `fixed_by_checking` -- tasks the one-shot twin failed at compile that the
S arm passed.

Rows marked `stale_code` (run.py --mark-stale) measured a stack that has
since been fixed. They are dropped before anything is computed and only
counted, so pre-fix and post-fix rows are never mixed.
`min_discordant_for_sig` is the fewest discordant pairs that could reach the
corrected alpha even if every one fell the same way (rule 4): below it, a
comparison cannot produce a finding, and says so.

The McNemar implementation is `livecodebench.mcnemar`, imported, not
reimplemented, so this and the dashboard cannot disagree.
"""
from __future__ import annotations

import glob
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
RESULTS = os.path.join(HERE, "results")
sys.path.insert(0, BENCH)

from livecodebench import mcnemar, wilson  # noqa: E402

ALPHA = 0.05
BASE = "A0"
ARM_ORDER = ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "C8", "C32", "C128",
             "S0", "S5", "S6"]
CAP_BASE = "C32"
# Mirrors run.SELF_CHECK (test_run.py checks the two agree). Not imported:
# analysis must not depend on the runner.
SELF_CHECK_TWIN = {"S0": "A0", "S5": "A5", "S6": "A6"}
STAGES = ("extract", "compile", "test")


# -------------------------------------------------------------- loading ----
def load(run_dir: str) -> tuple[list[dict], int]:
    path = os.path.join(run_dir, "rows.jsonl")
    rows, bad = [], 0
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    return rows, bad


def split_stale(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """(current rows, stale_code rows)."""
    return ([r for r in rows if not r.get("stale_code")],
            [r for r in rows if r.get("stale_code")])


def last_by_pair(rows: list[dict]) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    for r in rows:
        if "task" in r and "arm" in r:
            out[(r["task"], r["arm"])] = r
    return out


def scored(r: dict | None) -> bool:
    return bool(r) and r.get("outcome") in ("pass", "fail")


def min_discordant(alpha: float, cap: int = 400) -> int | None:
    for d in range(1, cap + 1):
        if mcnemar(d, 0) < alpha:
            return d
    return None


def _median(xs: list) -> float | None:
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(statistics.median(xs), 1) if xs else None


def _mean(xs: list) -> float | None:
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(sum(xs) / len(xs), 1) if xs else None


# ------------------------------------------------------------ row facts ----
def stage(r: dict) -> str:
    """pass, or where a scored failure died."""
    if r.get("outcome") == "pass":
        return "pass"
    return (r.get("grade") or {}).get("stage") or "?"


def _xs(r: dict) -> list[dict]:
    """x_yamadori of every request the row made (one, or one per S round)."""
    xs = r.get("x_rounds")
    return [x or {} for x in (xs if xs else [r.get("x_yamadori")])]


def tool_hops(r: dict) -> int:
    """Tool rounds the PROXY ran for the model (retrieval): hops - 1 per request."""
    return sum(max(int(x.get("hops") or 1) - 1, 0) for x in _xs(r))


def investigate_hops(r: dict) -> int:
    return sum(int((x.get("investigate") or {}).get("hops") or 0) for x in _xs(r))


def final_compiles(r: dict) -> bool:
    """The answer given got past extract and compile (it passed, or failed only
    at the hidden tests). Comparable across every arm."""
    return stage(r) in ("pass", "test")


def arm_metrics(sc: list[dict], arm: str) -> dict:
    """Cost, stage and self-check facts over an arm's scored rows."""
    st = {s: sum(1 for r in sc if stage(r) == s) for s in ("pass",) + STAGES}
    u = [r.get("usage") or {} for r in sc]
    out = {
        "stages": st,
        "mean_s": _mean([r.get("seconds") for r in sc]),
        "median_s": _median([r.get("seconds") for r in sc]),
        "tok_in_mean": _mean([x.get("prompt_tokens") for x in u]),
        "tok_out_mean": _mean([x.get("completion_tokens") for x in u]),
        "tok_total_mean": _mean([(x.get("prompt_tokens") or 0)
                                 + (x.get("completion_tokens") or 0) for x in u
                                 if x.get("prompt_tokens") is not None]),
        "tool_hops_mean": _mean([tool_hops(r) for r in sc]),
        "tool_hops_rows": sum(1 for r in sc if tool_hops(r) > 0),
        "investigate_hops_mean": _mean([investigate_hops(r) for r in sc]),
        "final_compiles": sum(1 for r in sc if final_compiles(r)),
        "n": len(sc),
        "self_check": arm in SELF_CHECK_TWIN,
    }
    if arm in SELF_CHECK_TWIN:
        cr = [int(r.get("check_rounds") or 0) for r in sc]
        fp = [r["final_public_check"] for r in sc
              if isinstance(r.get("final_public_check"), dict)]
        last_ok = [bool((r.get("check_results") or [])[-1].get("ok"))
                   for r in sc if r.get("check_results")]
        out.update({
            "twin": SELF_CHECK_TWIN[arm],
            "check_rounds_mean": _mean(cr), "check_rounds_median": _median(cr),
            "check_rounds_max": max(cr) if cr else None,
            "checked_any": sum(1 for c in cr if c > 0),
            "last_check_ok": sum(last_ok),
            "final_public_ok": sum(1 for f in fp if f.get("ok")),
            "final_public_n": len(fp),
            "rounds_mean": _mean([r.get("rounds") for r in sc]),
            "other_tool_calls": sum(len(r.get("other_tool_calls") or []) for r in sc),
        })
    return out


# ----------------------------------------------------------- comparisons ---
def compare(last: dict, tasks: list[str], x: str, y: str) -> dict:
    """y against x on the tasks where both are scored."""
    both = [t for t in tasks if scored(last.get((t, x))) and scored(last.get((t, y)))]
    px = {t: last[(t, x)]["outcome"] == "pass" for t in both}
    py = {t: last[(t, y)]["outcome"] == "pass" for t in both}
    b = sum(1 for t in both if py[t] and not px[t])
    c = sum(1 for t in both if px[t] and not py[t])
    return {"a": x, "b": y, "n_paired": len(both),
            "a_pass": sum(px.values()), "b_pass": sum(py.values()),
            "b_only": b, "a_only": c, "discordant": b + c,
            "both_solved": sum(1 for t in both if px[t] and py[t]),
            "p": mcnemar(b, c)}


def families(arms: list[str]) -> dict[str, list[tuple[str, str]]]:
    aug = [(BASE, a) for a in arms
           if a.startswith("A") and a != BASE and BASE in arms]
    if "A5" in arms and "A6" in arms:
        aug.append(("A6", "A5"))            # A5 vs the always-everything floor
    cap = [(CAP_BASE, a) for a in arms
           if a.startswith("C") and a != CAP_BASE and CAP_BASE in arms]
    sc = [(SELF_CHECK_TWIN[a], a) for a in arms
          if a in SELF_CHECK_TWIN and SELF_CHECK_TWIN[a] in arms]
    sc += [(BASE, a) for a in arms
           if a in SELF_CHECK_TWIN and SELF_CHECK_TWIN[a] != BASE and BASE in arms]
    return {k: v for k, v in (("augmentation", aug), ("thinking_cap", cap),
                              ("self_check", sc)) if v}


def twins(last: dict, tasks: list[str], arms: list[str]) -> list[dict]:
    """Each S arm against its one-shot twin: what checking changed, by stage."""
    out = []
    for s, a in SELF_CHECK_TWIN.items():
        if s not in arms or a not in arms:
            continue
        c = compare(last, tasks, a, s)
        both = [t for t in tasks if scored(last.get((t, a))) and scored(last.get((t, s)))]

        def moved(frm: str, to: str) -> int:
            return sum(1 for t in both if stage(last[(t, a)]) == frm
                       and stage(last[(t, s)]) == to)
        out.append({
            "s": s, "one_shot": a, "n_paired": c["n_paired"],
            "s_pass": c["b_pass"], "one_shot_pass": c["a_pass"],
            "s_only": c["b_only"], "one_shot_only": c["a_only"],
            "discordant": c["discordant"], "p": c["p"],
            # the one-shot answer did not compile; the checked one passed
            "fixed_by_checking": moved("compile", "pass"),
            "compile_to_test": moved("compile", "test"),
            "extract_to_pass": moved("extract", "pass"),
            "test_to_pass": moved("test", "pass"),
            "one_shot_compile_fail": sum(1 for t in both
                                         if stage(last[(t, a)]) == "compile"),
            "s_compile_fail": sum(1 for t in both if stage(last[(t, s)]) == "compile"),
            "one_shot_compiles": sum(1 for t in both if final_compiles(last[(t, a)])),
            "s_compiles": sum(1 for t in both if final_compiles(last[(t, s)])),
        })
    return out


def block(last: dict, tasks: list[str], arms: list[str], label: str) -> dict:
    out, fam_info = [], {}
    for fam, comps in families(arms).items():
        m = len(comps)
        alpha_adj = ALPHA / m
        floor = min_discordant(alpha_adj)
        fam_info[fam] = {"size": m, "alpha_corrected": alpha_adj,
                         "min_discordant_for_sig": floor}
        for x, y in comps:
            r = compare(last, tasks, x, y)
            r["p_bonferroni"] = min(1.0, r["p"] * m)
            r["family"] = fam
            r["family_size"] = m
            r["could_reach_significance"] = bool(floor is not None
                                                  and r["discordant"] >= floor)
            verdict = None
            if r["p_bonferroni"] < ALPHA and r["b_only"] != r["a_only"]:
                verdict = (f"{y} better" if r["b_only"] > r["a_only"]
                           else f"{x} better")
            r["verdict"] = verdict
            out.append(r)
    return {"label": label, "tasks": len(tasks), "families": fam_info,
            "comparisons": out}


def per_arm(rows: list[dict], last: dict, tasks: list[str], arms: list[str]) -> dict:
    tset = set(tasks)
    out = {}
    for a in arms:
        att = [r for r in rows if r.get("arm") == a and r.get("task") in tset]
        fin = [last[(t, a)] for t in tasks if (t, a) in last]
        sc = [r for r in fin if scored(r)]
        k = sum(1 for r in sc if r["outcome"] == "pass")
        lo, hi = wilson(k, len(sc))
        kinds: dict[str, int] = {}
        for r in att:
            if r.get("outcome") == "stack_error":
                kinds[r.get("stack_error_kind") or "?"] = \
                    kinds.get(r.get("stack_error_kind") or "?", 0) + 1
        stages: dict[str, int] = {}
        for r in sc:
            if r["outcome"] == "fail":
                s = (r.get("grade") or {}).get("stage") or "?"
                stages[s] = stages.get(s, 0) + 1
        out[a] = {
            "attempted_pairs": len(fin), "scored": len(sc), "passed": k,
            "rate": (k / len(sc)) if sc else None, "lo": lo, "hi": hi,
            "stack_error_rows": sum(kinds.values()),
            "stack_error_kinds": kinds,
            "pairs_still_stack_error": sum(1 for r in fin if not scored(r)),
            "fail_stages": stages,
            "median_seconds": _median([r.get("seconds") for r in sc]),
            "median_prompt_tokens": _median([(r.get("usage") or {}).get("prompt_tokens")
                                             for r in sc]),
            "median_completion_tokens": _median(
                [(r.get("usage") or {}).get("completion_tokens") for r in sc]),
            "median_hops": _median([(r.get("usage") or {}).get("hops") for r in sc]),
            # Thinking length. The proxy's usage has no reasoning/answer split,
            # so this is reasoning_content characters, plus reasoning_tokens
            # wherever a server reported them, plus how often the cap fired.
            "median_reasoning_chars": _median([r.get("reasoning_chars") for r in sc]),
            "median_reasoning_tokens": _median([r.get("reasoning_tokens") for r in sc]),
            "hit_thinking_cap": sum(1 for r in sc if r.get("hit_thinking_cap")),
            "reasoning_cap_sent": sorted({((r.get("x_yamadori") or {}).get("budget")
                                           or {}).get("reasoning_budget_tokens")
                                          for r in sc} - {None}),
            "empty_content": sum(1 for r in sc if r.get("empty_content")),
            **arm_metrics(sc, a),
        }
    return out


def triggers(last: dict, tasks: list[str], arms: list[str]) -> dict:
    """What the stack actually did, read off x_yamadori, per arm."""
    out = {}
    for a in arms:
        xs = [last[(t, a)].get("x_yamadori") or {} for t in tasks
              if (t, a) in last and last[(t, a)].get("outcome") != "stack_error"]
        n = len(xs)
        if not n:
            out[a] = {"n": 0}
            continue
        sel = [x.get("selection") or {} for x in xs]
        inv = [x.get("investigate") or {} for x in xs]
        gate = [x.get("tools_gate") for x in xs]

        def rate(k):
            return {"k": k, "n": n, "rate": round(k / n, 3)}
        out[a] = {
            "n": n,
            "tools_offered": rate(sum(1 for g in gate if g and g.get("offer"))),
            "hints_selected": rate(sum(1 for s in sel if s.get("hints"))),
            "hints_emitted": rate(sum(1 for x in xs if x.get("hints"))),
            "investigate_chosen": rate(sum(1 for s in sel if s.get("investigate"))),
            "investigate_searched": rate(sum(1 for i in inv
                                             if i.get("ran") and (i.get("hops") or 0) > 0)),
            "investigate_injected": rate(sum(1 for i in inv if i.get("injected"))),
            "fanned_out": rate(sum(1 for s in sel if int(s.get("fanout_n") or 1) > 1)),
            "main_hops_gt1": rate(sum(1 for x in xs if (x.get("hops") or 0) > 1)),
        }
    return out


def by_domain(last: dict, tasks_by_domain: dict[str, list[str]],
              arms: list[str], dirty: set[str] | None = None) -> dict:
    """Per domain; a domain is contaminated when its task rows say so."""
    out = {}
    for d, ts in sorted(tasks_by_domain.items()):
        cells = {}
        for a in arms:
            sc = [last[(t, a)] for t in ts if scored(last.get((t, a)))]
            k = sum(1 for r in sc if r["outcome"] == "pass")
            lo, hi = wilson(k, len(sc))
            cells[a] = {"k": k, "n": len(sc), "rate": (k / len(sc)) if sc else None,
                        "lo": lo, "hi": hi,
                        "stages": {s: sum(1 for r in sc if stage(r) == s)
                                   for s in STAGES},
                        "stack_errors": sum(1 for t in ts if (t, a) in last
                                            and not scored(last[(t, a)]))}
        out[d] = {"contaminated": bool(dirty and any(t in dirty for t in ts)),
                  "tasks": len(ts), "by_arm": cells,
                  "vs_A0": [compare(last, ts, BASE, a) for a in arms
                            if a != BASE and BASE in arms],
                  "vs_twin": twins(last, ts, arms)}
    return out


def suspects(arm_stats: dict) -> list[str]:
    """PROTOCOL rule 3: identical or extreme rates are a harness bug until proven."""
    out = []
    rated = {a: s for a, s in arm_stats.items() if s["scored"]}
    ks = {s["passed"] for s in rated.values()}
    if len(rated) > 1 and len(ks) == 1:
        out.append("every arm passed the same count")
    for a, s in rated.items():
        if s["rate"] in (0.0, 1.0):
            out.append(f"{a} scored exactly {s['rate']:.0%}")
    return out


# -------------------------------------------------------------- summary ----
def analyse(run_dir: str) -> dict:
    rows, bad = load(run_dir)
    rows, stale = split_stale(rows)
    last = last_by_pair(rows)
    man = {}
    mp = os.path.join(run_dir, "manifest.json")
    if os.path.exists(mp):
        man = json.load(open(mp, encoding="utf-8"))
    seen = {r["arm"] for r in rows if "arm" in r}
    arms = [a for a in ARM_ORDER if a in seen] + sorted(seen - set(ARM_ORDER))
    meta: dict[str, dict] = {}
    for r in rows:
        if "task" in r:
            meta[r["task"]] = {"domain": r.get("domain"),
                               "contaminated": bool(r.get("contaminated"))}
    tasks_all = sorted(meta)
    clean = [t for t in tasks_all if not meta[t]["contaminated"]]
    dirty = [t for t in tasks_all if meta[t]["contaminated"]]
    tbd: dict[str, list[str]] = {}
    for t in tasks_all:
        tbd.setdefault(meta[t]["domain"] or "?", []).append(t)

    arm_stats = per_arm(rows, last, tasks_all, arms)
    overridden = sorted({n for run in man.get("runs") or []
                         for n in run.get("preflight_overridden") or []})
    caveats = []
    if stale:
        reasons = sorted({(r["stale_code"] or {}).get("reason", "?")
                          if isinstance(r["stale_code"], dict) else str(r["stale_code"])
                          for r in stale})
        caveats.append(f"{len(stale)} stale_code rows excluded (measured a stack "
                       f"since fixed): " + "; ".join(reasons)[:300])
    if overridden:
        caveats.append("preflight checks were OVERRIDDEN for part of this run: "
                       + ", ".join(overridden))
    a3 = triggers(last, tasks_all, ["A3"]).get("A3") if "A3" in arms else None
    if a3 and a3.get("n") and a3["investigate_searched"]["k"] == 0:
        caveats.append("A3 never searched in its investigation: check that deep "
                       "thinking has tools with retrieval off")
    caveats += suspects(arm_stats)
    dirty_domains = sorted({meta[t]["domain"] or "?" for t in dirty})
    dnames = ", ".join(dirty_domains) or "none in this run"
    return {
        "run_id": man.get("run_id") or os.path.basename(run_dir.rstrip("/\\")),
        "run_dir": run_dir, "analysed_at": time.time(),
        "conditions": {k: man.get(k, "core" if k == "suite" else None)
                       for k in ("effort", "max_tokens", "temperature",
                                 "body_tier", "suite")},
        "rows": len(rows), "stale_rows": len(stale), "bad_lines": bad,
        "arms": arms,
        "tasks": {"all": len(tasks_all), "uncontaminated": len(clean),
                  "contaminated": len(dirty),
                  "contaminated_domains": dirty_domains},
        "caveats": caveats,
        "headline_uncontaminated": block(last, clean, arms,
                                         "uncontaminated domains (headline)"),
        # Key name kept for existing readers; it holds every contaminated task.
        "contaminated_three_tsl": block(last, dirty, arms,
                                        f"contaminated only ({dnames}) -- "
                                        f"CONTAMINATED"),
        "all_domains": block(last, tasks_all, arms,
                             f"all domains -- includes contaminated ({dnames})"),
        "per_arm": arm_stats,
        "per_arm_uncontaminated": per_arm(rows, last, clean, arms),
        "self_check_twins_uncontaminated": twins(last, clean, arms),
        "by_domain": by_domain(last, tbd, arms, set(dirty)),
        "triggers": triggers(last, tasks_all, arms),
        "dashboard": dashboard_block(rows, last, clean, arms, run_dir, bad,
                                     excluded={"tasks": len(dirty),
                                               "domains": dirty_domains}),
    }


def dashboard_block(rows, last, tasks, arms, run_dir, bad,
                    excluded: dict | None = None) -> dict:
    """The same shape `mcp/dash_results.py lcb_summary()` returns, so the
    results page can render this run with its existing `renderLcb`. Computed
    on the UNCONTAMINATED tasks; no contaminated task reaches the page.

    Added beyond the lcb shape (web/src/api/types.ts LcbSection, optional):
    per arm_table row `stages`, `median_s`, `tok_total`, `tool_hops`,
    `final_compiles` and the S-arm fields; `twins` (each S arm vs its
    one-shot twin); per-domain Wilson bounds and stages in `difficulty`;
    `domain_pairs` (uncorrected exact p per domain vs A0 and vs twin); and
    `excluded_contaminated`."""
    path = os.path.join(run_dir, "rows.jsonl")
    rel = os.path.relpath(path, os.path.dirname(BENCH)).replace("\\", "/")
    tset = set(tasks)
    rs = [r for r in rows if r.get("task") in tset]
    errs = [r for r in rs if r.get("outcome") == "stack_error"]
    err_by_arm: dict[str, dict[str, int]] = {a: {} for a in arms}
    for r in errs:
        k = r.get("stack_error_kind") or "?"
        err_by_arm.setdefault(r["arm"], {})[k] = err_by_arm.get(r["arm"], {}).get(k, 0) + 1
    complete = [t for t in tasks if all(scored(last.get((t, a))) for a in arms)]
    n = len(complete)
    floor = min_discordant(ALPHA)
    table, raw = [], []
    for a in arms:
        vals = [last[(t, a)] for t in complete]
        k = sum(1 for r in vals if r["outcome"] == "pass")
        lo, hi = wilson(k, n)
        secs = [r.get("seconds") for r in vals if isinstance(r.get("seconds"), (int, float))]
        pin = [(r.get("usage") or {}).get("prompt_tokens") for r in vals]
        pin = [x for x in pin if x]
        pout = [(r.get("usage") or {}).get("completion_tokens") for r in vals]
        pout = [x for x in pout if x]
        m = arm_metrics(vals, a)
        extra = {"stages": m["stages"], "median_s": m["median_s"],
                 "tok_total": (round(m["tok_total_mean"])
                               if m["tok_total_mean"] is not None else None),
                 "tool_hops": m["tool_hops_mean"],
                 "investigate_hops": m["investigate_hops_mean"],
                 "final_compiles": m["final_compiles"],
                 "self_check": m["self_check"]}
        if m["self_check"]:
            extra.update({k: m[k] for k in (
                "twin", "check_rounds_mean", "check_rounds_median",
                "check_rounds_max", "checked_any", "last_check_ok",
                "final_public_ok", "final_public_n", "rounds_mean")})
        table.append({"arm": a, "k": k, "n": n, "rate": (k / n) if n else None,
                      "lo": lo, "hi": hi,
                      "mean_s": round(sum(secs) / len(secs), 1) if secs else None,
                      "tok_in": round(sum(pin) / len(pin)) if pin else None,
                      "tok_in_reported": len(pin),
                      "tok_out": round(sum(pout) / len(pout)) if pout else None,
                      "tok_out_reported": len(pout),
                      "empty": sum(1 for r in vals if r.get("empty_content")),
                      "len_cut": 0,
                      "errors": sum(err_by_arm.get(a, {}).values()),
                      **extra})
        att = [last[(t, a)] for t in tasks if scored(last.get((t, a)))]
        kk = sum(1 for r in att if r["outcome"] == "pass")
        lo2, hi2 = wilson(kk, len(att))
        raw.append({"arm": a, "attempts": len(att), "passed": kk,
                    "rate": (kk / len(att)) if att else None, "lo": lo2, "hi": hi2})
    blk = block(last, tasks, arms, "")
    pairs = [{"a": c["a"], "b": c["b"], "b_only": c["b_only"],
              "a_only": c["a_only"], "discordant": c["discordant"],
              "p": c["p_bonferroni"], "verdict": c["verdict"],
              "family": c["family"],
              "underpowered": not c["could_reach_significance"],
              "both_solved": c["both_solved"], "signs": []}
             for c in blk["comparisons"]]
    domains = {}
    for t in tasks:
        d = (last.get((t, arms[0])) or {}).get("domain") or "?"
        domains.setdefault(d, []).append(t)
    difficulty, domain_pairs = [], []
    for d, ts in sorted(domains.items()):
        g = [t for t in ts if t in set(complete)]
        if not g:
            continue
        cells = {a: {"k": sum(1 for t in g if last[(t, a)]["outcome"] == "pass"),
                     "n": len(g),
                     "stages": {s: sum(1 for t in g if stage(last[(t, a)]) == s)
                                for s in STAGES}} for a in arms}
        for c in cells.values():
            c["rate"] = c["k"] / c["n"]
            c["lo"], c["hi"] = wilson(c["k"], c["n"])
        difficulty.append({"difficulty": d, "n": len(g), "by_arm": cells})
        for x, y in ([(BASE, a) for a in arms if a != BASE and BASE in arms]
                     + [(SELF_CHECK_TWIN[a], a) for a in arms
                        if a in SELF_CHECK_TWIN and SELF_CHECK_TWIN[a] in arms
                        and SELF_CHECK_TWIN[a] != BASE]):
            c = compare(last, g, x, y)
            domain_pairs.append({"domain": d, "a": x, "b": y,
                                 "n_paired": c["n_paired"], "b_only": c["b_only"],
                                 "a_only": c["a_only"],
                                 "p_uncorrected": c["p"]})
    failures = {}
    for a in arms:
        sc = [last[(t, a)] for t in tasks if scored(last.get((t, a)))]
        buckets: dict[str, dict] = {}
        for r in sc:
            if r["outcome"] == "pass":
                continue
            s = (r.get("grade") or {}).get("stage") or "?"
            e = buckets.setdefault(s, {"bucket": f"failed at {s}", "n": 0,
                                       "scaffolding": False})
            e["n"] += 1
        failures[a] = {"passed": sum(1 for r in sc if r["outcome"] == "pass"),
                       "failed": sum(1 for r in sc if r["outcome"] == "fail"),
                       "buckets": sorted(buckets.values(), key=lambda x: -x["n"])}
    ks = {x["k"] for x in table}
    return {
        "state": "ready" if rows else "empty",
        "file": rel, "exists": os.path.exists(path),
        "file_age_s": round(time.time() - os.path.getmtime(path)) if os.path.exists(path) else None,
        "bad_lines": bad,
        "how": "run  python bench/domain/run.py --key-file KEY --run-id ID",
        "arms": arms,
        "suspect": {"identical_across_arms": bool(n and len(ks) == 1 and len(table) > 1),
                    "extreme": [x["arm"] for x in table if n and x["rate"] in (0.0, 1.0)]},
        "progress": {"rows": len(rs), "attempted": len({r["task"] for r in rs}),
                     "scored_rows": sum(1 for r in rs if scored(r)),
                     "errors": len(errs), "complete_all_arms": n,
                     "errors_by_arm": err_by_arm},
        "power": {"n_paired": n, "min_discordant_for_sig": floor,
                  "max_possible_discordant": n, "alpha": ALPHA,
                  "comparison_possible": bool(floor is not None and n >= floor)},
        "arm_table": table, "raw": raw, "pairs": pairs,
        "difficulty": difficulty, "failures": failures, "scaffolding_total": 0,
        "twins": [dict(tw, p_bonferroni=next(
            (p["p"] for p in pairs if p["a"] == tw["one_shot"] and p["b"] == tw["s"]),
            None)) for tw in twins(last, complete, arms)],
        "domain_pairs": domain_pairs,
        "excluded_contaminated": excluded or {"tasks": 0, "domains": []},
        "note": ("uncontaminated domains only; McNemar p on this page is "
                 "Bonferroni-corrected over the comparisons shown"),
    }


def dashboard_section(results_root: str = RESULTS) -> dict:
    """For mcp/dash_results.py: the newest run under results/, lcb-shaped."""
    runs = [d for d in glob.glob(os.path.join(results_root, "*"))
            if os.path.exists(os.path.join(d, "rows.jsonl"))]
    if not runs:
        return {"state": "empty", "file": "bench/domain/results/*/rows.jsonl",
                "exists": False, "file_age_s": None, "bad_lines": 0,
                "how": "run  python bench/domain/run.py --key-file KEY --run-id ID",
                "note": "no run directory holds rows yet"}
    newest = max(runs, key=lambda d: os.path.getmtime(os.path.join(d, "rows.jsonl")))
    return analyse(newest)["dashboard"]


# ------------------------------------------------------------- markdown ----
def _p(p: float) -> str:
    return "1" if p >= 0.9995 else f"{p:.3g}"


def _pct(r) -> str:
    return "-" if r is None else f"{100 * r:.1f}%"


def markdown(s: dict) -> str:
    L = [f"# Domain benchmark -- {s['run_id']}", "",
         f"Conditions: {json.dumps(s['conditions'])}. Rows: {s['rows']}"
         + (f" ({s['bad_lines']} unparseable)" if s["bad_lines"] else "")
         + f". Tasks: {s['tasks']['uncontaminated']} uncontaminated, "
         f"{s['tasks']['contaminated']} contaminated ("
         f"{', '.join(s['tasks'].get('contaminated_domains') or []) or 'none'}).",
         ""]
    for c in s["caveats"]:
        L.append(f"> **Caveat:** {c}")
    if s["caveats"]:
        L.append("")
    for key in ("headline_uncontaminated", "contaminated_three_tsl", "all_domains"):
        b = s[key]
        L += [f"## {b['label']}", "", f"{b['tasks']} tasks. " + "; ".join(
              f"family `{f}`: m={i['size']} (alpha {i['alpha_corrected']:.4f}), "
              f"at least {i['min_discordant_for_sig']} discordant pairs needed "
              f"before any comparison in it can reach significance"
              for f, i in b["families"].items()) + ".", "",
              "| comparison | family | paired n | pass (a / b) | b only | a only | "
              "discordant | exact p | Bonferroni p | verdict |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for c in b["comparisons"]:
            v = c["verdict"] or ("none (underpowered)"
                                 if not c["could_reach_significance"] else "none")
            L.append(f"| {c['b']} vs {c['a']} | {c['family']} | {c['n_paired']} | "
                     f"{c['a_pass']} / {c['b_pass']} | {c['b_only']} | "
                     f"{c['a_only']} | {c['discordant']} | {_p(c['p'])} | "
                     f"{_p(c['p_bonferroni'])} | {v} |")
        L.append("")
    L += ["## Per arm (all domains, last row per pair)", "",
          "| arm | scored | passed | rate (Wilson 95%) | stack errors (rows) | "
          "kinds | median s | median prompt tok | median completion tok | "
          "median hops | median reasoning chars | cap fired | cap sent | "
          "fail stages |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for a, x in s["per_arm"].items():
        L.append(f"| {a} | {x['scored']} | {x['passed']} | {_pct(x['rate'])} "
                 f"[{_pct(x['lo'])}-{_pct(x['hi'])}] | {x['stack_error_rows']} | "
                 f"{json.dumps(x['stack_error_kinds'])} | {x['median_seconds']} | "
                 f"{x['median_prompt_tokens']} | {x['median_completion_tokens']} | "
                 f"{x['median_hops']} | {x['median_reasoning_chars']} | "
                 f"{x['hit_thinking_cap']} | {x['reasoning_cap_sent']} | "
                 f"{json.dumps(x['fail_stages'])} |")
    L += ["", "## Where it failed, what it cost, what it checked (all domains)", "",
          "| arm | pass | extract | compile | test | final compiles | mean s | "
          "median s | mean prompt tok | mean completion tok | proxy tool hops | "
          "deep-thinking hops | check rounds (mean / max) | checked any | "
          "final answer passed public check |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for a, x in s["per_arm"].items():
        st = x.get("stages") or {}
        sc = x.get("self_check")
        L.append(f"| {a} | {st.get('pass', 0)} | {st.get('extract', 0)} | "
                 f"{st.get('compile', 0)} | {st.get('test', 0)} | "
                 f"{x.get('final_compiles')}/{x.get('n')} | {x.get('mean_s')} | "
                 f"{x.get('median_s')} | {x.get('tok_in_mean')} | "
                 f"{x.get('tok_out_mean')} | {x.get('tool_hops_mean')} | "
                 f"{x.get('investigate_hops_mean')} | "
                 + (f"{x.get('check_rounds_mean')} / {x.get('check_rounds_max')} | "
                    f"{x.get('checked_any')}/{x.get('n')} | "
                    f"{x.get('final_public_ok')}/{x.get('final_public_n')} |"
                    if sc else "- | - | - |"))
    tw = s.get("self_check_twins_uncontaminated") or []
    if tw:
        L += ["", "## Self-check vs its one-shot twin (uncontaminated)", "",
              "| S arm vs twin | paired n | pass (twin / S) | S only | twin only | "
              "exact p | fixed by checking (compile -> pass) | compile -> test | "
              "extract -> pass | test -> pass | compile fails (twin / S) |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for t in tw:
            L.append(f"| {t['s']} vs {t['one_shot']} | {t['n_paired']} | "
                     f"{t['one_shot_pass']} / {t['s_pass']} | {t['s_only']} | "
                     f"{t['one_shot_only']} | {_p(t['p'])} | "
                     f"{t['fixed_by_checking']} | {t['compile_to_test']} | "
                     f"{t['extract_to_pass']} | {t['test_to_pass']} | "
                     f"{t['one_shot_compile_fail']} / {t['s_compile_fail']} |")
    L += ["", "## By domain (scored pass / n per arm)", ""]
    arms = s["arms"]
    L += ["| domain | tasks | " + " | ".join(arms) + " |",
          "|---|---|" + "---|" * len(arms)]
    for d, x in s["by_domain"].items():
        name = d + (" (CONTAMINATED)" if x["contaminated"] else "")
        L.append(f"| {name} | {x['tasks']} | " + " | ".join(
            f"{x['by_arm'][a]['k']}/{x['by_arm'][a]['n']}"
            + (f" +{x['by_arm'][a]['stack_errors']}err"
               if x['by_arm'][a]['stack_errors'] else "") for a in arms) + " |")
    L += ["", "## What the stack did (from x_yamadori, non-stack-error rows)", "",
          "| arm | n | tools offered | hints selected | hints emitted | "
          "investigate chosen | investigation searched | injected | fanned out |",
          "|---|---|---|---|---|---|---|---|---|"]
    for a, t in s["triggers"].items():
        if not t.get("n"):
            L.append(f"| {a} | 0 | | | | | | | |")
            continue
        f = lambda k: f"{t[k]['k']}/{t['n']}"                     # noqa: E731
        L.append(f"| {a} | {t['n']} | {f('tools_offered')} | {f('hints_selected')} | "
                 f"{f('hints_emitted')} | {f('investigate_chosen')} | "
                 f"{f('investigate_searched')} | {f('investigate_injected')} | "
                 f"{f('fanned_out')} |")
    L.append("")
    return "\n".join(L)


def newest_run(root: str = RESULTS) -> str | None:
    runs = [d for d in glob.glob(os.path.join(root, "*"))
            if os.path.exists(os.path.join(d, "rows.jsonl"))]
    return max(runs, key=lambda d: os.path.getmtime(
        os.path.join(d, "rows.jsonl"))) if runs else None


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    run_dir = argv[0] if argv else newest_run()
    if not run_dir or not os.path.exists(os.path.join(run_dir, "rows.jsonl")):
        print("  no rows.jsonl found; pass a run directory")
        return 2
    s = analyse(run_dir)
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(s, f, indent=1)
    md = markdown(s)
    with open(os.path.join(run_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
