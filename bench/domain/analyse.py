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


# THE SERVER'S OWN REASONING CEILING. llama-server is launched with
# --reasoning-budget 32768 (config.yaml). The MTP build honours a SMALLER
# per-request budget gracefully (stops, injects the budget message, answers),
# but a request asking for MORE than the launch value (the proxy asks ~98k)
# is hard-stopped near 32,768 thinking tokens: finish "stop", no budget
# message, no answer. Found on selfcheck-probe ts02/A0 (32,881 completion
# tokens, empty content, stopped mid-sentence) and confirmed by a direct
# probe at reasoning_budget_tokens=300. It was first treated as a stack
# error. The transcript review (docs/TRANSCRIPT-REVIEW-2026-09-23.md) showed
# the reasoning was one paragraph repeated 95 times: a model failure (a
# greedy-decoding loop) that the server's ceiling merely cut off. So the
# outcome stands; `server_reasoning_cap` is an ANNOTATION on the row, counted
# and reported, never a reclassification.
SERVER_CAP_BAND = (32700, 33300)


def server_cap_hit(r: dict) -> bool:
    """A scored row that is really the server's hard reasoning stop."""
    if r.get("outcome") not in ("pass", "fail") or r.get("finish_reason") != "stop":
        return False
    if not (r.get("empty_content") or stage(r) == "extract"):
        return False
    if r.get("hit_thinking_cap"):
        return False           # the graceful close happened: a real budget answer
    u = (r.get("usage_rounds") or [r.get("usage") or {}])[-1] or {}
    lo, hi = SERVER_CAP_BAND
    return any(isinstance(t, int) and lo <= t <= hi
               for t in (u.get("completion_tokens"), r.get("reasoning_tokens")))


def annotate(rows: list[dict]) -> list[dict]:
    """Rows with `annotations` added: "server_reasoning_cap" where the
    server's launch ceiling cut the thinking off. The outcome is untouched."""
    out = []
    for r in rows:
        if server_cap_hit(r) and "server_reasoning_cap" not in (r.get("annotations") or []):
            r = dict(r, annotations=list(r.get("annotations") or []) + ["server_reasoning_cap"])
        out.append(r)
    return out


# ------------------------------------------------------- mechanisms -------
# Per-row EVIDENCE that each mechanism the arm allowed actually worked, read
# from x_yamadori (every request of the row) and the row's own fields. Not a
# verdict: a mechanism that ran and legitimately found nothing (0 hints above
# the floor, deep thinking not searching, every search empty) is DATA and is
# recorded as such. A mechanism that was chosen and broke is caught by
# run.verify and is a stack_error row, re-run.
MECHANISMS = ("retrieval", "hints", "deep_thinking", "fanout", "self_check",
              "check_code")
NOT_RETRIEVAL = ("check_code", "generate_image")


def _hint_id(h: dict) -> str:
    import hashlib
    return hashlib.sha1(((h.get("source") or "") + "|" + (h.get("recipe") or ""))
                        .encode("utf-8")).hexdigest()[:10]


def mechanisms(r: dict) -> dict:
    """{mechanism: {allowed, ran, data, ...evidence}} for one row."""
    xs = [x for x in _xs(r) if x]
    sels = [x.get("selection") or {} for x in xs]

    def allowed(key: str, sel_key: str) -> bool:
        """Allowed for THIS arm: a flag the header forced is what it forced;
        otherwise what the tier allowed the selection engine to choose."""
        for sl in sels:
            sig = sl.get("signals") or {}
            if key in (sig.get("forced") or []):
                v = sl.get(sel_key)
                if (int(v or 1) > 1) if key == "fanout" else bool(v):
                    return True
            else:
                v = (sig.get("allowed") or {}).get(key)
                if (int(v or 1) > 1) if key == "fanout" else bool(v):
                    return True
        return False
    out: dict = {}
    # x_yamadori.tools lists EVERY proxy-executed call; check_code and
    # generate_image are their own mechanisms, not retrieval.
    all_calls = [c for x in xs for c in (x.get("tools") or [])]
    calls = [c for c in all_calls if c.get("name") not in NOT_RETRIEVAL]
    gates = [x.get("tools_gate") for x in xs]
    names: dict[str, int] = {}
    for c in calls:
        names[c.get("name") or "?"] = names.get(c.get("name") or "?", 0) + 1
    nonempty = sum(1 for c in calls if not c.get("empty") and not c.get("error"))
    out["retrieval"] = {
        "allowed": any(g is not None for g in gates),
        "offered": any(isinstance(g, dict) and g.get("offer") for g in gates),
        "ran": bool(calls), "data": nonempty > 0,
        "calls": len(calls), "tools": names, "nonempty": nonempty,
        "empty": sum(1 for c in calls if c.get("empty") and not c.get("error")),
        "errors": sum(1 for c in calls if c.get("error")),
        "evidence": "x_yamadori.tools" if any("tools" in x for x in xs) else
                    "not recorded by this proxy build (x_yamadori.tools absent)",
        "gate_why": next((g.get("why") for g in gates if isinstance(g, dict)), None)}
    hints = [h for x in xs for h in (x.get("hints") or [])]
    supp = [h for x in xs for h in (x.get("suppressed_hints") or [])]
    out["hints"] = {
        "allowed": allowed("hints", "hints"),
        "ran": any(s.get("hints") for s in sels), "data": bool(hints),
        "injected": len(hints), "ids": sorted({_hint_id(h) for h in hints}),
        "suppressed": len(supp)}
    invs = [x.get("investigate") for x in xs if isinstance(x.get("investigate"), dict)]
    out["deep_thinking"] = {
        "allowed": allowed("investigate", "investigate"),
        "chosen": any(s.get("investigate") for s in sels),
        "ran": any(i.get("ran") for i in invs),
        "data": any(i.get("injected") for i in invs),
        "searches": sum(int(i.get("hops") or 0) for i in invs),
        "injected": any(i.get("injected") for i in invs),
        "why_not": [i.get("why") for i in invs if i.get("why")]}
    fans = [x.get("fanout") for x in xs if isinstance(x.get("fanout"), dict)]
    f = fans[-1] if fans else {}
    cands = f.get("candidates") or []
    out["fanout"] = {
        "allowed": allowed("fanout", "fanout_n"),
        "chosen_n": max([int(s.get("fanout_n") or 1) for s in sels] or [1]),
        "ran": bool(f) and int(f.get("n") or 0) > 0 and not f.get("error"),
        "data": bool(f.get("selection")),
        "n": f.get("n"), "asked": f.get("asked"), "selection": f.get("selection"),
        "winner": f.get("winner"), "replaced": f.get("replaced"),
        "delivered": bool(f.get("replaced")) or f.get("winner") in ("original", None)
                     if f.get("selection") in ("code_medoid", "code_grade")
                     else None,
        "candidates_parsed": sum(1 for c in cands if c.get("parses")),
        "candidates": len(cands), "error": f.get("error")}
    cr = r.get("check_results") or []
    out["self_check"] = {
        "allowed": r.get("arm") in SELF_CHECK_TWIN, "ran": bool(cr), "data": bool(cr),
        "rounds": r.get("rounds"), "checks": len(cr),
        "errors_per_check": [int(c.get("n_errors") or 0) for c in cr],
        "ok_per_check": [bool(c.get("ok")) for c in cr]}
    cc = [x.get("check_code") for x in xs if isinstance(x.get("check_code"), dict)]
    # The tool-turn cap (PrismML agenticMaxTurns, 2026-09-23): the main loop
    # and deep thinking land after `limit` tool turns. hit=true is DATA.
    tts = [x.get("tool_turns") for x in xs if isinstance(x.get("tool_turns"), dict)]
    out["tool_turns"] = {
        "recorded": bool(tts),
        "limit": sorted({t.get("limit") for t in tts} - {None}),
        "turns_max": max([int(t.get("turns") or 0) for t in tts] or [0]),
        "turns_sum": sum(int(t.get("turns") or 0) for t in tts),
        "hit": any(t.get("hit") for t in tts)}
    cc_calls = [c for c in all_calls if c.get("name") == "check_code"]
    out["check_code"] = {
        "tool_calls_nonempty": sum(1 for c in cc_calls if not c.get("empty")),
        "tool_calls_errors": sum(1 for c in cc_calls if c.get("error")),
        "ok_results": sum(1 for c in cc for x in (c.get("results") or [])
                          if x.get("ok")),
        "allowed": any(c.get("offered") for c in cc),
        "ran": sum(int(c.get("calls") or 0) for c in cc) > 0,
        "data": sum(int(c.get("calls") or 0) for c in cc) > 0,
        "calls": sum(int(c.get("calls") or 0) for c in cc),
        "recorded": bool(cc)}
    return out


def mechanism_health(rows: list[dict]) -> dict:
    """Per mechanism over an arm's scored rows: % allowed, % ran, % produced
    data (allowed rows are the denominator for ran/data)."""
    out = {}
    ms = [mechanisms(r) for r in rows]      # recomputed: the stored copy may predate a fix
    n = len(ms)
    tt = [x.get("tool_turns") or {} for x in ms]
    out["tool_turns"] = {"rows": n, "recorded": sum(1 for t in tt if t.get("recorded")),
                         "hit": sum(1 for t in tt if t.get("hit")),
                         "limit": sorted({lim for t in tt for lim in t.get("limit") or []}),
                         "turns_max": max([t.get("turns_max") or 0 for t in tt] or [0])}
    for m in MECHANISMS:
        al = [x[m] for x in ms if x.get(m, {}).get("allowed")]
        out[m] = {"rows": n, "allowed": len(al),
                  "tool_errors": (sum(int(x.get("errors") or 0) for x in al)
                                  if m == "retrieval" else None),
                  "allowed_pct": round(100 * len(al) / n, 1) if n else None,
                  "ran": sum(1 for x in al if x.get("ran")),
                  "ran_pct": round(100 * sum(1 for x in al if x.get("ran")) / len(al), 1)
                  if al else None,
                  "data": sum(1 for x in al if x.get("data")),
                  "data_pct": round(100 * sum(1 for x in al if x.get("data")) / len(al), 1)
                  if al else None}
    return out


def final_compiles(r: dict) -> bool:
    """The answer given got past extract and compile (it passed, or failed only
    at the hidden tests). Comparable across every arm."""
    return stage(r) in ("pass", "test")


def style_metrics(sc: list[dict]) -> dict:
    """The SEPARATE style score (grade_style.py): never folded into pass/fail."""
    sty = [r.get("style") or {} for r in sc]
    lint = [x for x in sty if isinstance(x.get("lint_errors"), int)]
    flags: dict[str, int] = {}
    credits: dict[str, int] = {}
    for x in sty:
        for f in x.get("modern_flags") or []:
            flags[f] = flags.get(f, 0) + 1
        for c in x.get("modern_credits") or []:
            credits[c] = credits.get(c, 0) + 1
    return {"lint_n": len(lint),
            "lint_errors_mean": _mean([x["lint_errors"] for x in lint]),
            "lint_warnings_mean": _mean([x.get("lint_warnings") or 0 for x in lint]),
            "lint_clean": sum(1 for x in lint if x["lint_errors"] == 0),
            "modern_flags": dict(sorted(flags.items())),
            "modern_credits": dict(sorted(credits.items())),
            "style_errors": sum(1 for x in sty if x.get("style_error"))}


def row_concurrent(r: dict, man: dict | None) -> list[str]:
    """The other GPU consumers during this ROW. Rows since 2026-09-23 10:0x
    carry `concurrent_with`; an older row takes it from the manifest run
    (invocation) it was produced in -- the latest run started before it."""
    if "concurrent_with" in r:
        return list(r.get("concurrent_with") or [])
    runs = sorted((man or {}).get("runs") or [], key=lambda x: x.get("started") or 0)
    t = r.get("started_at") or 0
    cur = None
    for run in runs:
        if (run.get("started") or 0) <= t:
            cur = run
    if cur is not None:
        return list(cur.get("concurrent_with") or [])
    return list((man or {}).get("concurrent_with") or [])


def timing_metrics(sc: list[dict], man: dict | None) -> dict:
    """Seconds split by whether the GPU was shared. Only `clean` rows are
    single-user timings; `concurrent` rows are labelled and kept apart."""
    clean = [r for r in sc if not row_concurrent(r, man)]
    conc = [r for r in sc if row_concurrent(r, man)]
    return {"timing_rows": {"clean": len(clean), "concurrent": len(conc)},
            "mean_s_clean": _mean([r.get("seconds") for r in clean]),
            "median_s_clean": _median([r.get("seconds") for r in clean]),
            "mean_s_concurrent": _mean([r.get("seconds") for r in conc]),
            "timing_label": ("clean" if not conc else
                             "concurrent" if not clean else "mixed")}


_MAN_FOR_TIMING: dict = {}


def hint_metrics(last: dict, tasks: list[str], arm: str) -> dict:
    """Hint injection rate (rows with >= 1 hint injected / scored rows) and,
    on the injected rows, the pass rate beside their paired A0."""
    sc = [last[(t, arm)] for t in tasks if scored(last.get((t, arm)))]
    inj = [r for r in sc if any((x.get("hints") or []) for x in _xs(r))]
    paired = [r for r in inj if scored(last.get((r["task"], BASE)))]
    return {"hint_rows": len(inj), "hint_rate": (len(inj) / len(sc)) if sc else None,
            "hint_rows_paired": len(paired),
            "hint_rows_pass": sum(1 for r in paired if r["outcome"] == "pass"),
            "hint_rows_a0_pass": sum(1 for r in paired
                                     if last[(r["task"], BASE)]["outcome"] == "pass")}


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
        **style_metrics(sc),
        "mechanism_health": mechanism_health(sc),
        **timing_metrics(sc, _MAN_FOR_TIMING),
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
            # the one-shot answer did not compile; the S arm's passed (as
            # specified: whether or not the S row called check_solution)
            "fixed_by_checking": moved("compile", "pass"),
            # the same, counting only S rows that called check_solution >= 1
            "fixed_with_check": sum(
                1 for t in both if stage(last[(t, a)]) == "compile"
                and stage(last[(t, s)]) == "pass"
                and int(last[(t, s)].get("check_rounds") or 0) >= 1),
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
            **hint_metrics(last, tasks, a),
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
def _manifest(d: str) -> dict:
    mp = os.path.join(d, "manifest.json")
    if os.path.exists(mp):
        with open(mp, encoding="utf-8") as f:
            return json.load(f)
    return {}


FIXED = ("effort", "max_tokens", "temperature", "body_tier", "suite")


def group_dirs(run_dir: str) -> list[str]:
    """The run dir, plus every sibling whose manifest names the same `group`
    (run.py --group: parallel runners on disjoint tasks). Sorted, run_dir's
    own group first-found; a dir without a group is analysed alone."""
    g = _manifest(run_dir).get("group")
    if not g:
        return [run_dir]
    root = os.path.dirname(os.path.normpath(run_dir))
    out = [d for d in sorted(glob.glob(os.path.join(root, "*")))
           if os.path.exists(os.path.join(d, "rows.jsonl"))
           and _manifest(d).get("group") == g]
    return out or [run_dir]


def analyse(run_dir: str, merge: bool = True) -> dict:
    dirs = group_dirs(run_dir) if merge else [run_dir]
    rows, bad = [], 0
    for d in dirs:
        r, b = load(d)
        rows += r
        bad += b
    rows, stale = split_stale(rows)
    rows = annotate(rows)
    _MAN_FOR_TIMING.clear()
    _mm = [_manifest(d) for d in dirs]
    _MAN_FOR_TIMING["runs"] = [r for m in _mm for r in m.get("runs") or []]
    _MAN_FOR_TIMING["concurrent_with"] = sorted({c for m in _mm
                                                 for c in m.get("concurrent_with") or []})
    n_cap = sum(1 for r in rows if "server_reasoning_cap" in (r.get("annotations") or []))
    last = last_by_pair(rows)
    mans = [_manifest(d) for d in dirs]
    man = dict(mans[0]) if mans else {}
    if len(dirs) > 1:
        man["runs"] = [dict(r, run_dir=os.path.basename(d))
                       for d, m in zip(dirs, mans) for r in m.get("runs") or []]
        man["concurrent_with"] = sorted({c for m in mans
                                         for c in m.get("concurrent_with") or []})
        man["run_id"] = man.get("group") or man.get("run_id")
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
    if n_cap:
        caveats.append(f"{n_cap} row(s) annotated server_reasoning_cap: no answer, "
                       f"finish stop, ~32,768 reasoning tokens, no budget message -- "
                       f"the server's launch --reasoning-budget cut the thinking off. "
                       f"Scored as the model's failure (it had looped); annotation only")
    if len(dirs) > 1:
        differ = {k: sorted({json.dumps(m.get(k)) for m in mans}) for k in FIXED
                  if len({json.dumps(m.get(k)) for m in mans}) > 1}
        if differ:
            caveats.append(f"the grouped run dirs differ in fixed conditions "
                           f"{differ}: their rows are NOT one condition set")
        both = {}
        for d in dirs:
            for r in load(d)[0]:
                both.setdefault((r.get("task"), r.get("arm")), set()).add(d)
        dup = [k for k, v in both.items() if len(v) > 1]
        if dup:
            caveats.append(f"{len(dup)} (task, arm) pairs appear in more than one "
                           f"grouped run dir; the later-loaded dir's row is used")
    epochs = sorted({e.get("at") for m in mans for e in m.get("condition_epochs") or []}
                    - {None})
    pre_epoch = 0
    if epochs:
        newest = epochs[-1]
        pre_epoch = sum(1 for (t, a), r in last.items()
                        if scored(r) and (r.get("started_at") or 0) < newest)
        if pre_epoch:
            caveats.append(
                f"{pre_epoch} scored row(s) predate the newest condition epoch "
                f"({time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(newest))}): "
                f"they ran on an earlier stack deploy")
    timing = timing_of(man)
    shas = {json.dumps((run.get("hints_corpus") or {}).get(k))
            for run in man.get("runs") or [] if run.get("hints_corpus")
            for k in ("hint_buckets_sha256",)}
    rsh = {(run.get("hints_corpus") or {}).get("recipes_sha256")
           for run in man.get("runs") or [] if run.get("hints_corpus")}
    if len(shas) > 1 or len(rsh) > 1:
        caveats.append("the hints corpus changed between invocations of this run "
                       "(manifest runs[].hints_corpus): hint rows before and after "
                       "measured different corpora")
    if not timing["clean"]:
        caveats.append(f"timing is CONCURRENT with {', '.join(timing['concurrent_with'])}"
                       f": every seconds and tokens/s figure shared the GPU with "
                       f"them and is not a clean measurement")
    dirty_domains = sorted({meta[t]["domain"] or "?" for t in dirty})
    dnames = ", ".join(dirty_domains) or "none in this run"
    return {
        "run_id": man.get("run_id") or os.path.basename(run_dir.rstrip("/\\")),
        "run_dir": run_dir, "analysed_at": time.time(),
        "merged_run_dirs": [os.path.basename(os.path.normpath(d)) for d in dirs],
        "conditions": dict({k: man.get(k, "core" if k == "suite" else None)
                            for k in ("effort", "max_tokens", "temperature",
                                      "body_tier", "suite", "server_sampling")},
                           tool_turns_limit=sorted({
                               (x.get("tool_turns") or {}).get("limit")
                               for r in rows for x in _xs(r)
                               if isinstance((x or {}).get("tool_turns"), dict)} - {None})),
        "rows": len(rows), "stale_rows": len(stale), "bad_lines": bad,
        "annotated_server_reasoning_cap": n_cap,
        "condition_epochs": epochs if len(dirs) else [],
        "rows_before_newest_epoch": pre_epoch,
        "arms": arms,
        "tasks": {"all": len(tasks_all), "uncontaminated": len(clean),
                  "contaminated": len(dirty),
                  "contaminated_domains": dirty_domains},
        "caveats": caveats,
        "timing": timing,
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
                                               "domains": dirty_domains},
                                     timing=timing),
    }


def timing_of(man: dict) -> dict:
    """Whether this run's seconds / tokens-per-second are clean. A manifest
    naming other GPU consumers (run.py --concurrent-with) makes every timing
    figure CONCURRENT; it is labelled wherever it is shown."""
    conc = sorted(set(man.get("concurrent_with") or [])
                  | {c for run in man.get("runs") or []
                     for c in run.get("concurrent_with") or []})
    return {"concurrent_with": conc, "clean": not conc,
            "label": "CONCURRENT" if conc else "clean"}


def dashboard_block(rows, last, tasks, arms, run_dir, bad,
                    excluded: dict | None = None,
                    timing: dict | None = None) -> dict:
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
                 "self_check": m["self_check"],
                 **{k: m[k] for k in ("lint_n", "lint_errors_mean",
                                      "lint_warnings_mean", "lint_clean",
                                      "modern_flags", "modern_credits",
                                      "mechanism_health", "timing_rows",
                                      "mean_s_clean", "median_s_clean",
                                      "timing_label")}}
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
        "timing": timing or {"concurrent_with": [], "clean": True, "label": "clean"},
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
    return analyse(newest)["dashboard"]      # merged with its group, if any


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
    cs = "" if (s.get("timing") or {}).get("clean", True) else " (CONCURRENT)"
    L += ["## Per arm (all domains, last row per pair)", "",
          "| arm | scored | passed | rate (Wilson 95%) | stack errors (rows) | "
          f"kinds | median s{cs} | median prompt tok | median completion tok | "
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
          f"| arm | pass | extract | compile | test | final compiles | mean s{cs} | "
          f"median s{cs} | mean prompt tok | mean completion tok | proxy tool hops | "
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
              "exact p | fixed by checking (compile -> pass) | of those, with >= 1 "
              "check | compile -> test | "
              "extract -> pass | test -> pass | compile fails (twin / S) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for t in tw:
            L.append(f"| {t['s']} vs {t['one_shot']} | {t['n_paired']} | "
                     f"{t['one_shot_pass']} / {t['s_pass']} | {t['s_only']} | "
                     f"{t['one_shot_only']} | {_p(t['p'])} | "
                     f"{t['fixed_by_checking']} | {t['fixed_with_check']} | "
                     f"{t['compile_to_test']} | "
                     f"{t['extract_to_pass']} | {t['test_to_pass']} | "
                     f"{t['one_shot_compile_fail']} / {t['s_compile_fail']} |")
    L += ["", "## Style (separate score, never part of pass/fail; TS and React "
          "answers)", "",
          "| arm | linted | lint errors (mean) | lint warnings (mean) | "
          "0 lint errors | React 19 flags | credits (called for) | scorer errors |",
          "|---|---|---|---|---|---|---|---|"]
    for a, x in s["per_arm"].items():
        L.append(f"| {a} | {x.get('lint_n', 0)} | {x.get('lint_errors_mean')} | "
                 f"{x.get('lint_warnings_mean')} | {x.get('lint_clean', 0)} | "
                 f"{json.dumps(x.get('modern_flags') or {})} | "
                 f"{json.dumps(x.get('modern_credits') or {})} | "
                 f"{x.get('style_errors', 0)} |")
    L += ["", "## Hints (rows with >= 1 hint injected; on those rows, pass vs "
          "their paired A0)", "", "| arm | injected rows | rate | paired with A0 | "
          "pass (arm) | pass (A0) |", "|---|---|---|---|---|---|"]
    for a, x in s["per_arm"].items():
        L.append(f"| {a} | {x.get('hint_rows', 0)} | {_pct(x.get('hint_rate'))} | "
                 f"{x.get('hint_rows_paired', 0)} | {x.get('hint_rows_pass', 0)} | "
                 f"{x.get('hint_rows_a0_pass', 0)} |")
    L += ["", "## Timing by GPU sharing (only `clean` rows are single-user "
          "timings)", "", "| arm | clean rows | concurrent rows | mean s clean | "
          "median s clean | mean s concurrent |", "|---|---|---|---|---|---|"]
    for a, x in s["per_arm"].items():
        tr = x.get("timing_rows") or {}
        L.append(f"| {a} | {tr.get('clean', 0)} | {tr.get('concurrent', 0)} | "
                 f"{x.get('mean_s_clean')} | {x.get('median_s_clean')} | "
                 f"{x.get('mean_s_concurrent')} |")
    L += ["", "## Mechanism health (evidence, not a verdict): % of rows where the "
          "arm allowed it; of those, % where it ran, and % where it produced data",
          "", "| arm | " + " | ".join(MECHANISMS) + " |",
          "|---|" + "---|" * len(MECHANISMS)]
    for a, x in s["per_arm"].items():
        mh = x.get("mechanism_health") or {}
        L.append(f"| {a} | " + " | ".join(
            (f"{m['allowed_pct']}% / {m['ran_pct']}% / {m['data_pct']}%"
             if m.get("allowed") else f"{m.get('allowed_pct')}% / - / -")
            for m in (mh.get(k) or {} for k in MECHANISMS)) + " |")
    L += ["", "Tool-turn cap (x_yamadori.tool_turns; hit = the loop landed at "
          "the cap -- data, not an error): " + ", ".join(
              f"{a} limit {((x.get('mechanism_health') or {}).get('tool_turns') or {}).get('limit')}"
              f" hit {((x.get('mechanism_health') or {}).get('tool_turns') or {}).get('hit')}"
              f"/{((x.get('mechanism_health') or {}).get('tool_turns') or {}).get('rows')}"
              for a, x in s["per_arm"].items()) + "."]
    L += ["", "Retrieval tool errors per arm (a tool that answered ok:false -- "
          "evidence, not a stack error): " + ", ".join(
              f"{a} {((x.get('mechanism_health') or {}).get('retrieval') or {}).get('tool_errors')}"
              for a, x in s["per_arm"].items()) + ".",
          "Known limit: x_yamadori.tools records only error: true/false, not the "
          "error code, so NO_INDEX, a broken tool and bad arguments from the model "
          "cannot be told apart here yet."]
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
