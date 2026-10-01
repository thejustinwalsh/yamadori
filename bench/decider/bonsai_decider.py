#!/usr/bin/env python
"""THE BONSAI TYPED DECIDER (mcp/decider_bonsai.py), measured on the labels
in bench/decider/decider_labels.py (first written for the CLM yes/no run,
bench/clm/yesno.py; CLM and its recorded columns were removed 2026-09-29 --
the way back is commit e360d37).

    python bench/decider/bonsai_decider.py [--only intent,phase,h4,packages,batching,load]
    python bench/decider/bonsai_decider.py --model flash-next         --base-url http://127.0.0.1:18095 --only batching,calib

ANOTHER MODEL (2026-09-29; docs/JJAVA.md "Per model"): --model / --base-url
/ --slot (bench/decider/decider_target.py) read that engine instead -- the
decider's doors patched to it, the preflight on ITS server -- and write
results/bonsai_decider.<model>.json. The stack-layout parts (load,
load_small) are refused there: they measure the served Bonsai's slots. The
per-model record the runtime reads is bench/decider/measure_model.py's.

Operator, 2026-09-27: "Seems bonsai and reading the weights is likely the
best we can do." Every question is decided by argmax over its labels'
next-token probabilities at the answer position; no threshold anywhere.

WHAT IS MEASURED (everything IN-SAMPLE to our own labels):

  intent    bench/skills/work_intent.jsonl (49 yes / 50 no), three
            wordings (decider_labels.INTENT_Q), against route.work_intent.
  phase     bench/skills/daily_eval.jsonl, the points decider_labels._daily_points
            builds. The file carries NO phase labels: the only phase
            evidence is debugging (the 5 'bug-report' rows and
            seq-phase-debug#3), so debug RECALL on those 6, and for every
            phase AGREEMENT (not accuracy) with skill_classify.request_
            signals. One 4-way choice plus four yes/no questions.
            State: the whole transcript (decider_labels.transcript_pieces), uncut --
            the largest point is ~8k tokens, which the main model reads.
  h4        pagoda-h4's Hermes export hermes.jsonl (the full 134-step tool
            stream; session.jsonl holds only the post-compaction tail),
            decider_labels.h4_labels' mechanical labels, plus one of our own:
            scratch_write -- a step that creates a file progress.is_project
            says is NOT the project's (root /workspace/pagoda): a write/patch
            path, or a shell redirection / tee target in a terminal command
            (not /dev/null). Three state variants: `step` (the step alone,
            as the CLM run asked it), `window2048` (the task and the newest
            steps within 2,048 tokens: CLM's window, kept for the line-up
            with earlier results) and `full`
            (the task and EVERY step so far: the slot is KEPT across steps
            for this replay only and released at the end of the pass --
            the one place this bench leaves decider cells between batches,
            said in the result).
  packages  bench/skills/package_detect_labels.jsonl, every row asked about
            every package the labels name ("Does this work use X?", X
            described by its README's opening sentence), against
            skill_packages.detect; can the decider veto the detector's
            false positives (the four known: useQuery on a TanStack prompt,
            WebGL and WebGPU -> three, MeshBasicMaterial on a TSL prompt,
            and `View` in a code comment -- pagoda-h4 session.jsonl request
            42) without losing a true detection?
  batching  the same questions over one state, prefix cached on one slot
            vs cache_prompt false (every question re-reads the state).
  load      a long generation on the second brain's slot through
            model.post, its decode tok/s (a) alone, (b) with decider
            batches running beside it the whole time, (c) right after a
            decider batch that was RELEASED, (d) right after one that was
            NOT (cells left); /slots cells of the decider's slot read before
            each generation. n=3 each, interleaved.
  calib     the research doc's debiasing (docs/research/SKILLS-RESEARCH.md
            Part 4 item 2) re-measured on intent, phase, packages and the
            h4 step questions: raw yes/no, content-free ("N/A")
            calibration, batch calibration, a neutral-letter choice in two
            orders and averaged, ties within decider_bonsai.TIE_BAND.
  load_small  arm (b) again with a typical small state.
  h4turns   the wall time a request spends in the turn decider before
            main (decide_turn.Turn.facts + the selector's choose rounds)
            on pagoda-h4 session.jsonl's 57 requests, user turns vs steps.
  daily_choose  the selector's picks with choose() live vs its stub on the
            daily eval's own judge (MUST / MUST-NOT).

ONE GPU CONSUMER: before each part and every CHECK_EVERY decider calls, no
hermes.exe and no main-model slot processing (except this run's own during
`load`); otherwise the run stops (exit 2) and says so. Test material (the
Octopus/pagoda specs, the h4 task) is read, never printed or stored: rows
are named by id.

Written: bench/decider/results/bonsai.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "results", "bonsai.json")
SLOTS = "http://127.0.0.1:11434/upstream/bonsai/slots"
H4_DIR = "C:/Users/jwals/octo/logs/pagoda-h4-pagoda-xhigh-1"
CHECK_EVERY = 25
sys.path.insert(0, os.path.join(ROOT, "mcp"))

import decider_bonsai as D  # noqa: E402
sys.path.insert(0, HERE)
import decider_labels as Y  # noqa: E402  (labels, states, scoring)
import decider_target as DT  # noqa: E402

# The engine this run reads (decider_target.Target); None = the one door.
TARGET = None
# Parts that measure the served stack's slot layout, not a model.
STACK_PARTS = ("load", "load_small")

# ---------------------------------------------------------- questions ------
# Fixed before the run. The intent, phase and h4 wordings are CLM's
# (decider_labels.INTENT_Q, PHASE_Q, SITU_Q); the scratch question is the brief's.
SCRATCH_Q = ("Is the agent writing a throwaway file to test how something "
             "behaves?")
PHASE_KEYS = ["plan", "implement", "debug", "verify"]
PHASE_CHOICE = ("Which phase is the agent's work in?",
                ["planning the work", "implementing the work",
                 "debugging a failure", "verifying or testing the work"])
# The packages the labels name, each described by its README's opening
# sentence (index/packages/_src/<pkg>/README.md); three/tsl has no README of
# its own: skill_packages' definition of the unit (three's TSL entry, the
# WebGPU classes and the node materials).
PACKAGES = {
    "@react-three/fiber": "react-three-fiber, a React renderer for three.js",
    "@react-three/drei": ("drei, a collection of helpers and ready-made "
                          "abstractions for @react-three/fiber"),
    "@react-three/postprocessing": ("react-postprocessing, a postprocessing "
                                    "wrapper for @react-three/fiber"),
    "koota": ("Koota, an ECS-based state management library for real-time "
              "apps and games"),
    "math": "pmndrs math, a collection of math helpers for graphics and "
            "simulations (the npm package `math`)",
    "three": "three.js, the JavaScript 3D library",
    "three/tsl": ("three.js's TSL (`three/tsl`): its node shading language, "
                  "node materials and the WebGPURenderer"),
    "typegpu": "TypeGPU, a toolkit for WebGPU with shaders in TypeScript",
    "three-mesh-bvh": ("three-mesh-bvh, a bounding volume hierarchy to speed "
                       "up raycasting against three.js meshes"),
}


def package_q(pkg: str) -> dict:
    return D.yes_no(f"Does this work use {pkg} ({PACKAGES[pkg]})?")


# ------------------------------------------------------- one GPU consumer --
Busy = DT.Busy


def preflight(mine: tuple = ()) -> dict:
    if TARGET is not None and TARGET.mode != "one door":
        return DT.preflight(TARGET, mine)
    # A READ NEVER LOADS A MODEL (2026-09-30): /upstream/bonsai/slots only
    # while llama-swap's GET /running lists bonsai ready; otherwise nothing
    # is generating on it and the read is skipped, with why.
    import gpu_room
    loaded, why = gpu_room.model_loaded(SLOTS.split("/upstream/")[0], "bonsai")
    if loaded:
        with urllib.request.urlopen(SLOTS, timeout=10) as r:
            slots = json.loads(r.read())
    elif loaded is None:
        raise Busy(f"cannot tell whether the main model's slots are busy: {why}")
    else:
        slots = []
    busy = [s.get("id") for s in slots if s.get("is_processing")
            and s.get("id") not in mine]
    tl = subprocess.run(["tasklist", "/FI", "IMAGENAME eq hermes.exe"],
                        capture_output=True, text=True).stdout
    hermes = "hermes.exe" in tl.lower()
    if busy or hermes:
        raise Busy(f"main model slots processing {busy}; hermes.exe "
                   f"running: {hermes}")
    return {"slots_busy": busy, "slots_read": why, "hermes": hermes,
            "at": time.time()}


# ------------------------------------------------------------------ ask ----
CALLS: list[dict] = []
BATCHES: list[dict] = []
_MINE: tuple = ()


_CHECKED = [-1]


def run(state: str, qs: list[dict], part: str, item: str, **kw) -> dict:
    if len(CALLS) // CHECK_EVERY != _CHECKED[0]:
        preflight(_MINE)
        _CHECKED[0] = len(CALLS) // CHECK_EVERY
    for attempt in range(3):
        try:
            out = D.decide(state, qs, **kw)
            break
        except D.DeciderUnavailable as e:
            if not e.retryable or attempt == 2:
                raise
            preflight(_MINE)
            time.sleep(1)
    for a in out["answers"]:
        CALLS.append({"part": part, "item": item, "q": a["question"][:80],
                      **{k: a[k] for k in (
                          "answer", "probs", "label_mass", "exact",
                          "labels_unread", "unread_bound", "k", "reads",
                          "prompt_tokens", "processed_tokens",
                          "cached_tokens", "prefill_ms", "total_ms")}})
    BATCHES.append({"part": part, "item": item, "n_q": len(qs),
                    "slot": out["slot"],
                    "batch_ms": out["batch_ms"], "release": out["release"]})
    return out


_NTOK: dict[str, int] = {}


def tok(text: str) -> int:
    if text not in _NTOK:
        _NTOK[text] = len(D._upstream("/tokenize", {
            "content": text, "add_special": False})["tokens"])
    return _NTOK[text]


# ------------------------------------------------------------- scoring -----
def recs_of(answers: list[dict]) -> list[dict]:
    return [{"yes": a["answer"] == "yes", "p_true": a["probs"]["yes"]}
            for a in answers]


def ece(p_true: list[float], labels: list[bool], bins: int = 10) -> dict:
    """Expected calibration error of the decision's confidence (max(p,1-p)),
    equal-width bins over [0, 1] (the usual convention; 10 bins), plus a
    reliability table of p(yes) against the positive rate."""
    conf = [max(p, 1 - p) for p in p_true]
    corr = [(p > 0.5) == y for p, y in zip(p_true, labels)]
    n = len(conf)
    e, rows = 0.0, []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(conf)
               if lo <= c < hi or (b == bins - 1 and c == 1.0)]
        if idx:
            acc = sum(corr[i] for i in idx) / len(idx)
            mc = sum(conf[i] for i in idx) / len(idx)
            e += len(idx) / n * abs(acc - mc)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, p in enumerate(p_true)
               if lo <= p < hi or (b == bins - 1 and p == 1.0)]
        if idx:
            rows.append({"p_yes": f"{lo:.1f}-{hi:.1f}", "n": len(idx),
                         "mean_p_yes": round(sum(p_true[i] for i in idx)
                                             / len(idx), 3),
                         "positive_rate": round(sum(labels[i] for i in idx)
                                                / len(idx), 3)})
    return {"ece": round(e, 4) if n else None, "n": n, "reliability": rows}


def table(recs: list[dict], labels: list[bool]) -> dict:
    return {"confusion": Y.confusion([(y, r["yes"]) for y, r in
                                      zip(labels, recs)]),
            "calibration": ece([r["p_true"] for r in recs], labels),
            **Y.pdist(recs, labels)}


# ============================================================ 1. intent ====
def part_intent() -> dict:
    import route
    rows = []
    with open(os.path.join(ROOT, "bench", "skills", "work_intent.jsonl"),
              encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln and not ln.startswith("//"):
                rows.append(json.loads(ln))
    preflight()
    labels = [r["intent"] for r in rows]
    rule = [bool(route.work_intent(r["text"])) for r in rows]
    per = {q: [] for q in Y.INTENT_Q}
    out = {"n": len(rows), "positives": sum(labels), "rows": [],
           "route.work_intent": Y.confusion(list(zip(labels, rule))),
           "wordings": {}}
    for i, r in enumerate(rows):
        res = run(r["text"], [D.yes_no(q) for q in Y.INTENT_Q], "intent",
                  str(i))
        row = {"text": r["text"], "intent": r["intent"], "rule": rule[i],
               "p_yes": {}}
        for q, a in zip(Y.INTENT_Q, res["answers"]):
            per[q].append(a)
            row["p_yes"][q] = a["probs"]["yes"]
        out["rows"].append(row)
    for q in Y.INTENT_Q:
        recs = recs_of(per[q])
        t = table(recs, labels)
        t["vs_rule"] = {
            "both_right": sum(1 for y, a, g in zip(labels, recs, rule)
                              if a["yes"] == y and g == y),
            "decider_only_right": [rows[i]["text"] for i, (y, a, g) in
                                   enumerate(zip(labels, recs, rule))
                                   if a["yes"] == y and g != y],
            "rule_only_right": [rows[i]["text"] for i, (y, a, g) in
                                enumerate(zip(labels, recs, rule))
                                if a["yes"] != y and g == y]}
        t["errors"] = [{"text": rows[i]["text"], "label": y,
                        "p_yes": a["p_true"]} for i, (y, a) in
                       enumerate(zip(labels, recs)) if a["yes"] != y]
        out["wordings"][q] = t
    return out


# ============================================================= 2. phase ====
def part_phase() -> dict:
    import skill_classify
    pts = Y._daily_points()
    preflight()
    qs = ([D.choice(*PHASE_CHOICE)]
          + [D.yes_no(Y.PHASE_Q[k]) for k in PHASE_KEYS])
    res = {"n_points": len(pts), "labels": (
        "daily_eval.jsonl carries NO phase labels. Phase evidence: debugging "
        "only (cat 'bug-report' x5 and seq-phase-debug#3). Debug RECALL on "
        "those 6; for every phase AGREEMENT with skill_classify."
        "request_signals, not accuracy."),
        "state": "decider_labels.transcript_pieces joined, uncut", "points": []}
    for p in pts:
        sig = skill_classify.request_signals(p["msgs"], p["route"],
                                             p["tools"])
        rule = set((sig.get("phases") or {}).keys())
        state = "\n\n".join(Y.transcript_pieces(p["msgs"]))
        out = run(state, qs, "phase", p["id"])
        ch, yn = out["answers"][0], out["answers"][1:]
        rec = {"id": p["id"], "cat": p["cat"], "route": p["route"],
               "source": p["source"], "rule": sorted(rule),
               "tokens": ch["prompt_tokens"],
               "choice": PHASE_KEYS[D.LETTERS.index(ch["answer"])],
               "choice_probs": {PHASE_KEYS[D.LETTERS.index(k)]: v
                                for k, v in ch["probs"].items()},
               "yes_no": {k: {"p_yes": a["probs"]["yes"],
                              "yes": a["answer"] == "yes"}
                          for k, a in zip(PHASE_KEYS, yn)}}
        res["points"].append(rec)
    P = res["points"]
    debug_ev = [r for r in P if r["cat"] == "bug-report"
                or r["id"] == "seq-phase-debug#3"]
    res["debug_evidence"] = {
        "n": len(debug_ev),
        "bonsai_yes_no_yes": sum(r["yes_no"]["debug"]["yes"]
                                 for r in debug_ev),
        "bonsai_choice_debug": sum(r["choice"] == "debug" for r in debug_ev),
        "rule_yes": sum("debug" in r["rule"] for r in debug_ev),
        "rows": [(r["id"], r["yes_no"]["debug"]["p_yes"], r["choice"],
                  "debug" in r["rule"]) for r in debug_ev]}
    res["agreement_yes_no"] = {}
    for ph in PHASE_KEYS:
        c = Y.confusion([(ph in r["rule"], r["yes_no"][ph]["yes"])
                         for r in P])
        res["agreement_yes_no"][ph] = {
            "rule_yes_bonsai_yes": c["tp"], "rule_yes_bonsai_no": c["fn"],
            "rule_no_bonsai_yes": c["fp"], "rule_no_bonsai_no": c["tn"],
            "agree": c["tp"] + c["tn"], "n": c["n"],
            "bonsai_yes": c["tp"] + c["fp"]}
    one = [r for r in P if len(set(r["rule"]) & set(PHASE_KEYS)) == 1]
    res["choice"] = {
        "distribution": _hist(r["choice"] for r in P),
        "rule_single_phase_points": len(one),
        "agree_with_rule_single_phase": sum(
            r["choice"] in r["rule"] for r in one),
        "by_rule_phase": {ph: _hist(r["choice"] for r in one
                                    if ph in r["rule"]) for ph in PHASE_KEYS},
        "rule_none_points": _hist(r["choice"] for r in P
                                  if not set(r["rule"]) & set(PHASE_KEYS))}
    res["phases_per_point"] = {
        "bonsai_yes_no": _hist(sum(v["yes"] for v in r["yes_no"].values())
                               for r in P),
        "rule": _hist(len(set(r["rule"]) & set(PHASE_KEYS)) for r in P)}
    # The only labels in this part: the 6 debugging points.
    res["debug_calibration_note"] = (
        "n=6 positives and no labelled negatives: no calibration is "
        "computed for phase.")
    return res


def _hist(xs) -> dict:
    h: dict = {}
    for x in xs:
        h[str(x)] = h.get(str(x), 0) + 1
    return dict(sorted(h.items()))


# ================================================================ 3. h4 ====
_REDIRECT = re.compile(
    r"(?:^|[^\d&>])>>?\s*(['\"]?)([^\s'\";&|<>()]+)\1"
    r"|\btee\s+(?:-a\s+)?(['\"]?)([^\s'\";&|<>()]+)\3")
H4_PROJECT = {"root": "/workspace/pagoda", "named": []}


def scratch_targets(name: str, args: dict) -> tuple[list, list]:
    """(files this step creates that are not the project's, write-tool paths
    that are not the project's -- what the stack's own rule sees)."""
    import progress
    if name in ("write_file", "patch"):
        tg = [str(args.get("path") or "")]
    elif name == "terminal":
        c = str(args.get("command") or "")
        tg = [m.group(2) or m.group(4) for m in _REDIRECT.finditer(c)]
        tg = [t for t in tg if t and t != "/dev/null"]
    else:
        tg = []
    out = [t for t in tg if t and not progress.is_project(t, H4_PROJECT)[0]]
    rule = out if name in ("write_file", "patch") else []
    return out, rule


def part_h4(variants=("full", "step", "window2048")) -> dict:
    import skill_select
    steps = Y._h4_steps()
    with open(os.path.join(H4_DIR, "prompt.md"), encoding="utf-8") as f:
        task = f.read().strip()
    msgs = [{"role": "user", "content": task}]
    for i, s in enumerate(steps):
        msgs.append({"role": "assistant", "content": s["said"],
                     "tool_calls": [{"id": f"call_{i}", "type": "function",
                                     "function": {"name": s["name"],
                                                  "arguments": json.dumps(
                                                      s["args"])}}]})
        msgs.append({"role": "tool", "tool_call_id": f"call_{i}",
                     "content": s["result"]})
    preflight()
    qkeys = ["repeat_failed", "reads_package", "scratch_write"]
    qs = [D.yes_no(Y.SITU_Q["repeat_failed"]),
          D.yes_no(Y.SITU_Q["reads_package"]), D.yes_no(SCRATCH_Q)]
    rows, prior, pieces = [], [], [f"User: {task}"]
    for i, s in enumerate(steps):
        lab = Y.h4_labels(s["name"], s["args"], prior, s["failed"], s["sig"])
        sc, sc_rule = scratch_targets(s["name"], s["args"])
        lab["scratch_write"] = bool(sc)
        call = {"type": "function", "function": {
            "name": s["name"], "arguments": json.dumps(s["args"])}}
        cur = Y.step_text(s["said"], s["name"], json.dumps(s["args"]),
                          s["result"])
        pieces.append(cur)
        rows.append({"step": i, "tool": s["name"], "failed": s["failed"],
                     "labels": lab, "rule": {
                         "probed_packages": bool(
                             skill_select.probed_packages(call)),
                         "is_project_write": bool(sc_rule)},
                     "cur": cur, "pieces_upto": len(pieces), "bonsai": {}})
        prior.append({"name": s["name"], "args": json.dumps(
            s["args"], sort_keys=True), "failed": s["failed"],
            "sig": s["sig"]})
    out = {"n_steps": len(steps), "source": "hermes.jsonl",
           "scratch_label": ("a step creating a file progress.is_project "
                             "calls not the project's (root "
                             "/workspace/pagoda): write/patch path, or a "
                             "shell redirection/tee target, /dev/null "
                             "aside"),
           "label_counts": {k: sum(r["labels"][k] for r in rows)
                            for k in rows[0]["labels"]}, "variants": {}}
    for var in variants:
        t0 = time.time()
        info = {"slot_kept_across_steps": var == "full"}
        uncached = 0
        try:
            for r in rows:
                if var == "full" and uncached >= 3:
                    info["aborted"] = (
                        "the kept slot did not reuse the history: three "
                        "steps past 8,192 prompt tokens re-read over half "
                        "of it")
                    break
                if var == "step":
                    state = r["cur"]
                elif var == "window2048":
                    state, fit = D.fit_tail(pieces[:r["pieces_upto"]], 2048,
                                            tok)
                    r.setdefault("fit2048", fit)
                else:
                    state = "\n\n".join(pieces[:r["pieces_upto"]])
                res = run(state, qs, f"h4:{var}", str(r["step"]),
                          keep_slot=(var == "full"))
                r["bonsai"][var] = {k: {"p_yes": a["probs"]["yes"],
                                        "yes": a["answer"] == "yes",
                                        "tokens": a["prompt_tokens"],
                                        "processed": a["processed_tokens"]}
                                    for k, a in zip(qkeys, res["answers"])}
                a0 = res["answers"][0]
                if var == "full" and (a0["prompt_tokens"] or 0) > 8192 and                         (a0["processed_tokens"] or 0) > a0["prompt_tokens"] / 2:
                    uncached += 1
        finally:
            if var == "full":
                info["release_at_end"] = D.release(
                    BATCHES[-1]["slot"] if BATCHES else None,
                    "h4 full-history pass end")
        info["seconds"] = round(time.time() - t0, 1)
        if var == "full" and not info.get("aborted"):
            first = [r["bonsai"]["full"]["repeat_failed"] for r in rows]
            info["processed_tokens_total"] = sum(
                (x["processed"] or 0) for r in rows
                for x in r["bonsai"]["full"].values())
            info["history_tokens_final"] = first[-1]["tokens"]
        out["variants"][var] = info
    out["tables"] = {}
    variants = [v for v in variants
                if all(v in r["bonsai"] for r in rows)]
    for key, lks in (("reads_package", ["reads_package"]),
                     ("repeat_failed", ["repeat_failed_strict",
                                        "repeat_failed_same_error"]),
                     ("scratch_write", ["scratch_write"])):
        for lk in lks:
            ys = [r["labels"][lk] for r in rows]
            for var in variants:
                recs = [{"yes": r["bonsai"][var][key]["yes"],
                         "p_true": r["bonsai"][var][key]["p_yes"]}
                        for r in rows]
                out["tables"][f"{lk}|bonsai:{var}"] = table(recs, ys)
            if key == "reads_package":
                out["tables"][f"{lk}|rule:probed_packages"] = {
                    "confusion": Y.confusion(
                        [(y, r["rule"]["probed_packages"])
                         for y, r in zip(ys, rows)])}
            elif key == "scratch_write":
                out["tables"][f"{lk}|rule:progress.is_project(write tools)"] \
                    = {"confusion": Y.confusion(
                        [(y, r["rule"]["is_project_write"])
                         for y, r in zip(ys, rows)])}
    out["positives"] = {lk: [(r["step"], r["tool"],
                              {v: r["bonsai"][v][key]["p_yes"]
                               for v in variants})
                             for r in rows if r["labels"][lk]]
                        for key, lk in (("repeat_failed",
                                         "repeat_failed_same_error"),
                                        ("scratch_write", "scratch_write"))}
    out["rows"] = [{k: v for k, v in r.items() if k != "cur"} for r in rows]
    return out


# ========================================================== 4. packages ====
def part_packages() -> dict:
    sys.path.insert(0, os.path.join(ROOT, "bench", "skills"))
    _argv = sys.argv
    sys.argv = [sys.argv[0]]
    gpu = os.environ.get("YAMADORI_GPU_ROOM")
    import eval_packages as EP  # copies the skill store to a temp dir
    sys.argv = _argv
    if gpu is None:
        os.environ.pop("YAMADORI_GPU_ROOM", None)
    import skill_packages as SP
    preflight()
    pkgs = list(PACKAGES)
    qs = [package_q(p) for p in pkgs]
    rows = EP.rows()
    out = {"n_rows": len(rows), "packages": PACKAGES, "rows": []}
    for r in rows:
        det = SP.detect(EP.user(r["text"]))
        res = run(r["text"], qs, "packages", r["id"])
        out["rows"].append({
            "id": r["id"], "near": bool(r.get("near")),
            "expect": r["expect"],
            "detected": {p: [f"{w['how']}:{w['what']}" for w in e["why"]]
                         for p, e in det.items()},
            "p_yes": {p: a["probs"]["yes"] for p, a in
                      zip(pkgs, res["answers"])}})
    R = out["rows"]
    # The decider alone, every (row, package) pair.
    pairs = [(p in r["expect"], r["p_yes"][p] > 0.5, r["p_yes"][p])
             for r in R for p in pkgs]
    out["decider_alone"] = {
        "confusion": Y.confusion([(y, g) for y, g, _ in pairs]),
        "calibration": ece([pp for _, _, pp in pairs],
                           [y for y, _, _ in pairs]),
        "near_miss_rows_with_any_yes": [
            (r["id"], [p for p in pkgs if r["p_yes"][p] > 0.5])
            for r in R if r["near"] and any(r["p_yes"][p] > 0.5
                                            for p in pkgs)],
        "expected_missed": [(r["id"], p, r["p_yes"][p]) for r in R
                            for p in r["expect"] if r["p_yes"][p] <= 0.5],
        "extra_yes_on_positive_rows": [
            (r["id"], p, r["p_yes"][p]) for r in R if not r["near"]
            for p in pkgs if p not in r["expect"] and r["p_yes"][p] > 0.5]}
    # The detector, and the detector gated by the decider (detect AND yes).
    det_pairs = [(r["id"], p, p in r["expect"], r["p_yes"].get(p))
                 for r in R for p in r["detected"]]
    out["detector"] = {"detections": len(det_pairs),
                       "true": sum(1 for x in det_pairs if x[2]),
                       "false": [(i, p) for i, p, y, _ in det_pairs
                                 if not y],
                       "missed": [(r["id"], p) for r in R
                                  for p in r["expect"]
                                  if p not in r["detected"]]}
    out["veto"] = {
        "true_detections_kept": sum(1 for x in det_pairs
                                    if x[2] and (x[3] or 0) > 0.5),
        "true_detections_lost": [(i, p, pp) for i, p, y, pp in det_pairs
                                 if y and (pp or 0) <= 0.5],
        "false_detections_vetoed": [(i, p, pp) for i, p, y, pp in det_pairs
                                    if not y and (pp or 0) <= 0.5],
        "false_detections_kept": [(i, p, pp) for i, p, y, pp in det_pairs
                                  if not y and (pp or 0) > 0.5]}
    # The fourth known false positive: `View` in a code comment, pagoda-h4
    # session.jsonl request 42 (bench/skills/eval_packages.py --h4). The
    # state is the detector's own evidence view of that request.
    msgs = EP._session_messages(os.path.join(H4_DIR, "session.jsonl"))
    points = [i for i, m in enumerate(msgs) if m["role"] == "assistant"
              and i > 0]
    upto = msgs[:points[42]]
    det = SP.detect(upto)
    ev = SP.evidence_pieces(upto)
    state = "\n\n".join(f"{w}:\n{t}" for w, t in ev)
    ask = [p for p in ("@react-three/drei", "three", "koota")]
    res = run(state, [package_q(p) for p in ask], "packages:h4-req42",
              "h4-req42")
    out["h4_req42"] = {
        "detected": {p: [f"{w['how']}:{w['what']}" for w in e["why"]]
                     for p, e in det.items()},
        "labels": {"@react-three/drei": False, "three": True,
                   "koota": True},
        "label_basis": ("drei: `View` in a code comment (the known false "
                        "positive); three, koota: an import in the "
                        "evidence (mechanical)"),
        "state_tokens": res["answers"][0]["prompt_tokens"],
        "p_yes": {p: a["probs"]["yes"] for p, a in
                  zip(ask, res["answers"])}}
    known = [("near-react-query", "koota"), ("near-webgl-uniform", "three"),
             ("imp-pin-fiber", "three"), ("imp-colornode-ts2339", "three")]
    by = {r["id"]: r for r in R}
    out["known_false_positives"] = (
        [{"row": i, "package": p, "p_yes": by[i]["p_yes"][p],
          "vetoed": by[i]["p_yes"][p] <= 0.5} for i, p in known]
        + [{"row": "h4-req42", "package": "@react-three/drei",
            "p_yes": out["h4_req42"]["p_yes"]["@react-three/drei"],
            "vetoed": out["h4_req42"]["p_yes"]["@react-three/drei"] <= 0.5}])
    return out


# ========================================================== 5. batching ====
def part_batching() -> dict:
    """The same 5 phase questions over 20 daily-eval states, prefix cached on
    one slot (the decider's normal path) vs cache_prompt false."""
    pts = Y._daily_points()
    preflight()
    step = max(len(pts) // 20, 1)
    sel = pts[::step][:20]
    qs = ([D.choice(*PHASE_CHOICE)]
          + [D.yes_no(Y.PHASE_Q[k]) for k in PHASE_KEYS])
    rows = []
    for p in sel:
        state = "\n\n".join(Y.transcript_pieces(p["msgs"]))
        a = run(state, qs, "batching:cached", p["id"])
        b = run(state, qs, "batching:uncached", p["id"], cache=False)
        rows.append({"id": p["id"],
                     "state_tokens": a["answers"][0]["prompt_tokens"],
                     "cached": {"batch_ms": a["batch_ms"],
                                "processed": sum(x["processed_tokens"] or 0
                                                 for x in a["answers"]),
                                "per_q_ms": [x["total_ms"]
                                             for x in a["answers"]]},
                     "uncached": {"batch_ms": b["batch_ms"],
                                  "processed": sum(x["processed_tokens"] or 0
                                                   for x in b["answers"]),
                                  "per_q_ms": [x["total_ms"]
                                               for x in b["answers"]]},
                     "same_answers": [x["answer"] for x in a["answers"]]
                     == [x["answer"] for x in b["answers"]],
                     "max_prob_diff": max(
                         abs(x["probs"][k] - y["probs"][k])
                         for x, y in zip(a["answers"], b["answers"])
                         for k in x["probs"])})
    return {"n_states": len(rows), "questions_per_state": len(qs),
            "rows": rows,
            "sum_batch_ms": {"cached": round(sum(r["cached"]["batch_ms"]
                                                 for r in rows)),
                             "uncached": round(sum(r["uncached"]["batch_ms"]
                                                   for r in rows))},
            "sum_processed": {"cached": sum(r["cached"]["processed"]
                                            for r in rows),
                              "uncached": sum(r["uncached"]["processed"]
                                              for r in rows)}}


# ============================================================== 6. load ====
GEN_TOKENS = 512     # long enough to overlap ~10 decider batches (~1.5 s)
GEN_PROMPT_TOKENS = 32768


def _gen_body() -> dict:
    """A long prompt (repo source, not test material) and a fixed-length,
    greedy, thinking-off generation, shaped by the one rule and then fixed:
    ignore_eos and max_tokens so every run decodes GEN_TOKENS; top_k 1 so the
    runs are the same tokens (speculative acceptance comparable)."""
    import model
    src = open(os.path.join(ROOT, "mcp", "proxy.py"), encoding="utf-8").read()
    # Cut to ~GEN_PROMPT_TOKENS by the model's own count.
    lo, hi = 0, len(src)
    while hi - lo > 2000:
        mid = (lo + hi) // 2
        if tok(src[:mid]) < GEN_PROMPT_TOKENS:
            lo = mid
        else:
            hi = mid
    text = src[:lo]
    b = model.shape({"messages": [
        {"role": "system", "content": "You are a careful code reviewer."},
        {"role": "user", "content": "Here is a Python source file.\n\n```"
         "python\n" + text + "\n```\n\nDescribe what it does, section by "
         "section, in detail."}]}, effort="minimal")
    b.update(max_tokens=GEN_TOKENS, ignore_eos=True, top_k=1)
    return b


def _gen(b: dict) -> dict:
    import model
    t0 = time.time()
    d = model.post(dict(b))
    tm = d.get("timings") or {}
    return {"decode_tps": round(tm.get("predicted_per_second") or 0, 2),
            "predicted_n": tm.get("predicted_n"),
            "prompt_n": tm.get("prompt_n"), "cache_n": tm.get("cache_n"),
            "draft_n": tm.get("draft_n"),
            "draft_accepted": tm.get("draft_n_accepted"),
            "wall_s": round(time.time() - t0, 2)}


def part_load(rounds: int = 3) -> dict:
    global _MINE
    import slots
    pts = Y._daily_points()
    preflight()
    big = max(pts, key=lambda p: len("\n\n".join(
        Y.transcript_pieces(p["msgs"]))))
    state = "\n\n".join(Y.transcript_pieces(big["msgs"]))
    qs = ([D.choice(*PHASE_CHOICE)]
          + [D.yes_no(Y.PHASE_Q[k]) for k in PHASE_KEYS])
    b = _gen_body()
    hs = slots.helper_slot()
    dslot = slots._transient_slot(slots.count())
    _MINE = (hs, dslot)
    out = {"gen": {"slot": hs, "prompt_tokens_target": GEN_PROMPT_TOKENS,
                   "tokens": GEN_TOKENS, "sampling": "greedy (top_k 1), "
                   "thinking off, ignore_eos"},
           "decider": {"slot": dslot, "state_id": big["id"],
                       "questions": len(qs)},
           "warmup": None, "runs": []}
    out["warmup"] = _gen(b)          # prefills the long prompt once
    D.release(dslot, "load: start clean")
    for r in range(rounds):
        preflight(_MINE)
        # (a) alone
        D.release(dslot, "load: (a)")
        cells = D.slot_cells(dslot)
        out["runs"].append({"round": r, "arm": "a_alone",
                            "decider_cells": cells, **_gen(b)})
        # (b) decider batches beside the generation, the whole time
        box = {}
        th = threading.Thread(target=lambda: box.update(g=_gen(b)))
        th.start()
        lat, nb = [], 0
        time.sleep(0.5)
        while th.is_alive():
            res = run(state, qs, "load:b", f"r{r}")
            nb += 1
            lat += [a["total_ms"] for a in res["answers"]]
        th.join()
        out["runs"].append({"round": r, "arm": "b_during", "batches": nb,
                            "decider_q_ms": lat, **box["g"]})
        # (d) after a batch, NOT released
        res = run(state, qs, "load:d", f"r{r}", keep_slot=True)
        cells = D.slot_cells(dslot)
        out["runs"].append({"round": r, "arm": "d_after_kept",
                            "decider_cells": cells, **_gen(b)})
        # (c) after a batch, released
        res = run(state, qs, "load:c", f"r{r}")
        cells = D.slot_cells(dslot)
        out["runs"].append({"round": r, "arm": "c_after_released",
                            "decider_cells": cells,
                            "release": res["release"], **_gen(b)})
    D.release(dslot, "load: end")
    _MINE = ()
    out["summary"] = {}
    for arm in ("a_alone", "b_during", "c_after_released", "d_after_kept"):
        xs = [x["decode_tps"] for x in out["runs"] if x["arm"] == arm]
        out["summary"][arm] = {"n": len(xs), "decode_tps": xs,
                               "mean": round(statistics.mean(xs), 2)
                               if xs else None}
    return out


# ============================================================= 7. calib ====
# docs/research/SKILLS-RESEARCH.md Part 4 item 2, applied and re-measured on
# the same labels: every yes/no question also asked as a neutral-letter
# two-option choice in both orders (A=yes/B=no and A=no/B=yes), the phase
# choice in its given and reversed order, each template also asked of the
# content-free state D.NEUTRAL_STATE, and near-ties (D.TIE_BAND) counted as
# ties (no decision). Arms, all from these reads:
#   yn_raw      the yes/no as asked
#   yn_cc       yn_raw divided by its content-free read
#   yn_bc       Batch Calibration: argmax(p - mean p over the set)
#   mc_fwd / mc_rev   the two-letter choice in one order
#   mc_avg      the two orders averaged
#   mc_avg_cc   each order calibrated by its own content-free read, averaged
def _ask_set(state: str, qs_sem: list[dict], part: str, item: str) -> list:
    """Each semantic question as: yes/no raw (yes_no only), then as_choice in
    both orders. Returns per question {raw, orders: [meaning probs]}."""
    batch, index = [], []
    for qi, q in enumerate(qs_sem):
        if q["kind"] == "yes_no":
            batch.append(q)
            index.append((qi, "raw", None))
        for o in D.orders_for(q, 2):
            c = D.as_choice(q, o)
            batch.append(c)
            index.append((qi, "order", c))
    res = run(state, batch, part, item)
    out = [{"raw": None, "orders": []} for _ in qs_sem]
    for (qi, kind, c), a in zip(index, res["answers"]):
        if kind == "raw":
            out[qi]["raw"] = dict(a["probs"])
        else:
            out[qi]["orders"].append(D.meaning_probs(a, c))
    return out


_CF: dict[str, dict] = {}


def _content_free(q: dict) -> dict:
    key = json.dumps(q, sort_keys=True)
    if key not in _CF:
        _CF[key] = _ask_set(D.NEUTRAL_STATE, [q], "calib:content_free",
                            q["question"][:40])[0]
    return _CF[key]


def _arms(reads: list[dict], cf: dict) -> dict:
    """{arm: [probability dict per row]} for one semantic question."""
    arms = {"mc_fwd": [r["orders"][0] for r in reads],
            "mc_rev": [r["orders"][1] for r in reads],
            "mc_avg": [D.average(r["orders"]) for r in reads],
            "mc_avg_cc": [D.average([D.contextual(o, c) for o, c in
                                     zip(r["orders"], cf["orders"])])
                          for r in reads]}
    if reads and reads[0]["raw"] is not None:
        arms["yn_raw"] = [r["raw"] for r in reads]
        arms["yn_cc"] = [D.contextual(r["raw"], cf["raw"]) for r in reads]
    return arms


def _bc(dists: list[dict]) -> list[dict]:
    """Batch Calibration's decision input: p minus the set's mean p (not a
    probability; used for the argmax only)."""
    keys = list(dists[0])
    mean = {k: sum(d[k] for d in dists) / len(dists) for k in keys}
    return [{k: d[k] - mean[k] for k in keys} for d in dists]


def _score_binary(dists: list[dict], labels: list[bool], prob=True) -> dict:
    dec = [D.decision(d) for d in dists]
    pairs = [(y, x["answer"] == "yes") for y, x in zip(labels, dec)
             if not x["tie"]]
    out = {"confusion": Y.confusion(pairs),
           "ties": sum(x["tie"] for x in dec)}
    if prob:
        out["ece"] = ece([d["yes"] for d in dists], labels)["ece"]
    return out


def part_calib() -> dict:
    import route
    out = {"tie_band": D.TIE_BAND, "neutral_state": D.NEUTRAL_STATE,
           "orders": 2}
    # --- intent
    rows = []
    with open(os.path.join(ROOT, "bench", "skills", "work_intent.jsonl"),
              encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln and not ln.startswith("//"):
                rows.append(json.loads(ln))
    preflight()
    labels = [r["intent"] for r in rows]
    qs = [D.yes_no(q) for q in Y.INTENT_Q]
    reads = [_ask_set(r["text"], qs, "calib:intent", str(i))
             for i, r in enumerate(rows)]
    out["intent"] = {"rule": Y.confusion(
        [(y, bool(route.work_intent(r["text"])))
         for y, r in zip(labels, rows)]), "wordings": {}}
    for qi, q in enumerate(qs):
        cf = _content_free(q)
        arms = _arms([r[qi] for r in reads], cf)
        t = {a: _score_binary(v, labels) for a, v in arms.items()}
        t["yn_bc"] = _score_binary(_bc(arms["yn_raw"]), labels, prob=False)
        t["mc_avg_bc"] = _score_binary(_bc(arms["mc_avg"]), labels,
                                       prob=False)
        t["content_free"] = cf
        out["intent"]["wordings"][q["question"]] = t
    # --- phase (agreement and debug recall; no phase labels exist)
    import skill_classify
    pts = Y._daily_points()
    preflight()
    pq = D.choice(*PHASE_CHOICE)
    yq = [D.yes_no(Y.PHASE_Q[k]) for k in PHASE_KEYS]
    prow = []
    for p in pts:
        sig = skill_classify.request_signals(p["msgs"], p["route"],
                                             p["tools"])
        rule = set((sig.get("phases") or {}).keys())
        state = "\n\n".join(Y.transcript_pieces(p["msgs"]))
        r = _ask_set(state, [pq] + yq, "calib:phase", p["id"])
        prow.append({"id": p["id"], "cat": p["cat"], "rule": rule, "r": r})
    cfs = [_content_free(q) for q in [pq] + yq]
    debug_ids = [i for i, x in enumerate(prow) if x["cat"] == "bug-report"
                 or x["id"] == "seq-phase-debug#3"]
    out["phase"] = {"n": len(prow), "debug_n": len(debug_ids),
                    "choice": {}, "yes_no": {},
                    "content_free": {"choice": cfs[0]["orders"],
                                     "yes_no": {k: c["orders"] for k, c in
                                                zip(PHASE_KEYS, cfs[1:])}}}
    carms = _arms([x["r"][0] for x in prow], cfs[0])
    letter_key = dict(zip(D.LETTERS, PHASE_KEYS))
    one = [i for i, x in enumerate(prow)
           if len(x["rule"] & set(PHASE_KEYS)) == 1]
    for a, dists in carms.items():
        dec = [D.decision(d) for d in dists]
        ch = [letter_key.get(x["answer"]) for x in dec]
        out["phase"]["choice"][a] = {
            "distribution": _hist(c for c in ch),
            "ties": sum(x["tie"] for x in dec),
            "debug_recall": sum(ch[i] == "debug" for i in debug_ids),
            "agree_rule_single_phase": sum(ch[i] in prow[i]["rule"]
                                           for i in one),
            "rule_single_phase_n": len(one)}
    for k, qi in zip(PHASE_KEYS, range(1, 5)):
        arms = _arms([x["r"][qi] for x in prow], cfs[qi])
        out["phase"]["yes_no"][k] = {}
        for a, dists in arms.items():
            dec = [D.decision(d) for d in dists]
            yes = [x["answer"] == "yes" for x in dec]
            ruled = [k in x["rule"] for x in prow]
            out["phase"]["yes_no"][k][a] = {
                "yes": sum(yes), "ties": sum(x["tie"] for x in dec),
                "agree_rule": sum(y == r for y, r in zip(yes, ruled)),
                "debug_recall": (sum(yes[i] for i in debug_ids)
                                 if k == "debug" else None)}
    # --- packages
    sys.path.insert(0, os.path.join(ROOT, "bench", "skills"))
    _argv = sys.argv
    sys.argv = [sys.argv[0]]
    gpu = os.environ.get("YAMADORI_GPU_ROOM")
    import eval_packages as EP
    sys.argv = _argv
    if gpu is None:
        os.environ.pop("YAMADORI_GPU_ROOM", None)
    import skill_packages as SP
    preflight()
    pkgs = list(PACKAGES)
    pqs = [package_q(p) for p in pkgs]
    prs = EP.rows()
    preads, dets = [], []
    for r in prs:
        preads.append(_ask_set(r["text"], pqs, "calib:packages", r["id"]))
        dets.append(SP.detect(EP.user(r["text"])))
    out["packages"] = {"arms": {}}
    pcf = [_content_free(q) for q in pqs]
    for a in ("yn_raw", "yn_cc", "mc_fwd", "mc_rev", "mc_avg", "mc_avg_cc"):
        dists, labs, det_pairs = [], [], []
        for ri, r in enumerate(prs):
            for pi, pkg in enumerate(pkgs):
                d = _arms([preads[ri][pi]], pcf[pi])[a][0]
                dists.append(d)
                labs.append(pkg in r["expect"])
                if pkg in dets[ri]:
                    det_pairs.append((r["id"], pkg, pkg in r["expect"], d))
        t = _score_binary(dists, labs)
        dec = {(i, p): D.decision(d) for i, p, _, d in det_pairs}
        t["veto"] = {
            "true_kept": sum(1 for i, p, y, d in det_pairs
                             if y and dec[(i, p)]["answer"] == "yes"),
            "true_lost": sum(1 for i, p, y, d in det_pairs
                             if y and dec[(i, p)]["answer"] == "no"),
            "false_vetoed": [(i, p) for i, p, y, d in det_pairs
                             if not y and dec[(i, p)]["answer"] == "no"],
            "false_kept": [(i, p) for i, p, y, d in det_pairs
                           if not y and dec[(i, p)]["answer"] == "yes"],
            "ties": sum(1 for v in dec.values() if v["tie"])}
        out["packages"]["arms"][a] = t
    # --- h4 (step state): reads_package on every step; scratch_write on
    # terminal steps only (the rule never sees a terminal write)
    steps = Y._h4_steps()
    preflight()
    rq, sq = D.yes_no(Y.SITU_Q["reads_package"]), D.yes_no(SCRATCH_Q)
    hr, hs, lr, ls = [], [], [], []
    prior = []
    for i, s in enumerate(steps):
        lab = Y.h4_labels(s["name"], s["args"], prior, s["failed"], s["sig"])
        sc, _ = scratch_targets(s["name"], s["args"])
        cur = Y.step_text(s["said"], s["name"], json.dumps(s["args"]),
                          s["result"])
        term = s["name"] == "terminal"
        r = _ask_set(cur, [rq, sq] if term else [rq], "calib:h4", str(i))
        hr.append(r[0])
        lr.append(lab["reads_package"])
        if term:
            hs.append(r[1])
            ls.append(bool(sc))
        prior.append({"name": s["name"], "args": json.dumps(
            s["args"], sort_keys=True), "failed": s["failed"],
            "sig": s["sig"]})
    out["h4"] = {"reads_package": {}, "scratch_write_terminal": {},
                 "n_steps": len(hr), "n_terminal": len(hs),
                 "scratch_positives": sum(ls)}
    for key, reads, labs, q in (("reads_package", hr, lr, rq),
                                ("scratch_write_terminal", hs, ls, sq)):
        arms = _arms(reads, _content_free(q))
        for a, v in arms.items():
            out["h4"][key][a] = _score_binary(v, labs)
    return out


# ========================================================= 8. load_small ===
def part_load_small(rounds: int = 3) -> dict:
    """Arm (b) again with a TYPICAL state: the daily-eval point of median
    size, the 5 phase questions, batches back to back beside the same long
    generation; (a) alone interleaved."""
    global _MINE
    import slots
    pts = Y._daily_points()
    preflight()
    sized = sorted(pts, key=lambda p: len("\n\n".join(
        Y.transcript_pieces(p["msgs"]))))
    mid = sized[len(sized) // 2]
    state = "\n\n".join(Y.transcript_pieces(mid["msgs"]))
    qs = ([D.choice(*PHASE_CHOICE)]
          + [D.yes_no(Y.PHASE_Q[k]) for k in PHASE_KEYS])
    b = _gen_body()
    hs = slots.helper_slot()
    dslot = slots._transient_slot(slots.count())
    _MINE = (hs, dslot)
    out = {"state_id": mid["id"], "state_tokens": tok(state),
           "warmup": _gen(b), "runs": []}
    for r in range(rounds):
        preflight(_MINE)
        out["runs"].append({"round": r, "arm": "a_alone", **_gen(b)})
        box = {}
        th = threading.Thread(target=lambda: box.update(g=_gen(b)))
        th.start()
        lat, nb = [], 0
        time.sleep(0.5)
        while th.is_alive():
            res = run(state, qs, "load_small:b", f"r{r}")
            nb += 1
            lat.append(res["batch_ms"])
        th.join()
        out["runs"].append({"round": r, "arm": "b_during", "batches": nb,
                            "batch_ms": lat, **box["g"]})
    _MINE = ()
    for arm in ("a_alone", "b_during"):
        xs = [x["decode_tps"] for x in out["runs"] if x["arm"] == arm]
        out[arm] = {"decode_tps": xs, "mean": round(statistics.mean(xs), 2)}
    return out


# ============================================================ 9. h4turns ===
def part_h4turns(limit: int | None = None) -> dict:
    """WHAT A USER WAITS FOR: the wall time a request spends in the turn
    decider before main -- decide_turn.Turn.facts() plus every choose()
    round the selector asks (skill_select.decide, the proxy's own path, with
    `choose` live because the Turn is open) -- on pagoda-h4's session.jsonl
    requests (the post-compaction part of the run, 57 requests), in order,
    the skill state carried from request to request as the proxy does.
    The embedding stage is NOT run (its query embedder is on the A4000,
    which this bench never touches): skill_match.rank_all answers "not
    run", so a match question's candidates are the pattern path's, and the
    option lists can be shorter than live."""
    sys.path.insert(0, os.path.join(ROOT, "bench", "skills"))
    _argv = sys.argv
    sys.argv = [sys.argv[0]]
    gpu = os.environ.get("YAMADORI_GPU_ROOM")
    import eval_packages as EP          # copies the skill store to a temp dir
    import replay_selection as R
    sys.argv = _argv
    if gpu is None:
        os.environ.pop("YAMADORI_GPU_ROOM", None)
    import decide_turn
    import route
    import skill_match
    import skill_select
    skill_match.rank_all = lambda state, pool, embed=None: (None, {
        "ok": False, "why": "bench: the embedding stage is not run (its "
        "embedder is on the A4000)"})
    decide_turn.enable()
    preflight()
    msgs = EP._session_messages(os.path.join(H4_DIR, "session.jsonl"))
    points = [i for i, m in enumerate(msgs) if m["role"] == "assistant"
              and i > 0][:limit]
    pool = R.skills.armed()
    st = None
    rows = []
    for k, i in enumerate(points):
        if k % 5 == 0:
            preflight()
        clean = [m for m in msgs[:i] if m.get("role") in (
            "system", "user", "assistant", "tool")]
        try:
            rc = route.classify(clean, client_tools=R.HERMES_TOOLS,
                                util={"utility": False}, gate=None,
                                dbs={})["class"]
        except Exception:                                        # noqa: BLE001
            rc = None
        t0 = time.time()
        with decide_turn.Turn(clean, key="h4", request=f"req{k}",
                              client_tools=R.HERMES_TOOLS, on=True) as t:
            facts = t.facts()
            t_f = time.time()
            _txt, rec, st = skill_select.decide(clean, rc, pool,
                                                R.HERMES_TOOLS, st,
                                                key=f"h4:{k}")
        wall = time.time() - t0
        tr = t.record()
        chooses = [a for a in tr["answers"]
                   if str(a.get("name", "")).startswith("choose:")]
        rows.append({
            "req": k, "kind": tr["kind"], "route": rc,
            "state_tokens": tr["state"].get("tokens"),
            "state_cut": tr["state"].get("cut"),
            "wall_ms": round(wall * 1000, 1),
            "facts_ms": round((t_f - t0) * 1000, 1),
            "select_ms": round((wall - (t_f - t0)) * 1000, 1),
            "facts": {n: {"value": v.get("value"), "source": v.get("source")}
                      for n, v in facts.items()},
            "choose_rounds": len(chooses),
            "choose_ms": round(sum(a["ms"] for a in chooses), 1),
            "choose_options": [a["options"] for a in chooses],
            "selector_decider": rec.get("decider"),
            "questions": len(rec.get("questions") or []),
            "picked": [d.get("name") for d in rec.get("decisions") or []],
            "release": (tr.get("release") or {}).get("released"),
            "failure": tr.get("failure")})

    def q(xs):
        xs = sorted(xs)
        return {"n": len(xs), "p50": round(statistics.median(xs), 1),
                "p90": round(xs[int(0.9 * (len(xs) - 1))], 1),
                "max": round(xs[-1], 1)} if xs else None
    out = {"n": len(rows), "embedding": "not run (A4000)", "rows": rows,
           "wall_ms": {}, "facts_ms": {}, "select_ms": {}}
    for kind in ("user", "step", "all"):
        rs = [r for r in rows if kind == "all" or r["kind"] == kind]
        for key in ("wall_ms", "facts_ms", "select_ms"):
            out[key][kind] = q([r[key] for r in rs])
    return out


# ======================================================= 10. daily_choose ==
def part_daily_choose() -> dict:
    """The selector's picks with decide_turn.choose live vs its stub, on the
    daily eval's MUST / MUST-NOT labels (bench/skills/replay_selection.py
    daily(), its own judge): every skill_select.decide call runs inside an
    open Turn, so skill_match asks Bonsai; then the same rows with no Turn
    (the stub). Embedding stage not run (A4000), as in the replay's default.
    IN-SAMPLE: the labels were written beside the selector."""
    sys.path.insert(0, os.path.join(ROOT, "bench", "skills"))
    _argv = sys.argv
    sys.argv = [sys.argv[0]]
    gpu = os.environ.get("YAMADORI_GPU_ROOM")
    import replay_selection as R
    sys.argv = _argv
    if gpu is None:
        os.environ.pop("YAMADORI_GPU_ROOM", None)
    import decide_turn
    import skill_match
    import skill_select
    skill_match.rank_all = lambda state, pool, embed=None: (None, {
        "ok": False, "why": "bench: the embedding stage is not run"})
    decide_turn.enable()
    preflight()
    pool = R.skills.armed()
    real = skill_select.decide
    walls: list[float] = []

    def wrapped(messages, *a, **kw):
        t0 = time.time()
        with decide_turn.Turn(messages, key="daily", on=True):
            out = real(messages, *a, **kw)
        walls.append((time.time() - t0) * 1000)
        return out
    out = {}
    for arm in ("bonsai", "stub"):
        skill_select.decide = wrapped if arm == "bonsai" else real
        try:
            res = R.daily(pool)
        finally:
            skill_select.decide = real
        checks = res["checks"]
        out[arm] = {"passed": sum(1 for ok, _ in checks if ok),
                    "total": len(checks),
                    "failed": [w for ok, w in checks if not ok],
                    "per_area": res["per_area"],
                    "requests": [{"id": r["id"], "got": [g["name"] for g in
                                                         r["got"]],
                                  "fp_areas": r["fp_areas"]}
                                 for r in res["requests"]]}
        preflight()
    xs = sorted(walls)
    out["bonsai_wall_ms"] = ({"n": len(xs),
                              "p50": round(statistics.median(xs), 1),
                              "p90": round(xs[int(0.9 * (len(xs) - 1))], 1)}
                             if xs else None)
    return out


# ================================================================= main ====
def latency_summary() -> dict:
    def q(xs):
        xs = sorted(x for x in xs if x is not None)
        return {"n": len(xs), "p50": round(statistics.median(xs), 1),
                "p90": round(xs[int(0.9 * (len(xs) - 1))], 1),
                "max": round(xs[-1], 1)} if xs else None
    alone = [c for c in CALLS if not c["part"].startswith("load:b")]
    out = {"alone": {"total_ms": q([c["total_ms"] for c in alone]),
                     "prefill_ms": q([c["prefill_ms"] for c in alone]),
                     "prompt_tokens": q([c["prompt_tokens"] for c in alone]),
                     "processed_tokens": q([c["processed_tokens"]
                                            for c in alone])},
           "during_generation": {
               "total_ms": q([c["total_ms"] for c in CALLS
                              if c["part"] == "load:b"]),
               "prefill_ms": q([c["prefill_ms"] for c in CALLS
                                if c["part"] == "load:b"])},
           "by_processed_tokens": {}}
    for lo, hi in ((0, 64), (64, 256), (256, 1024), (1024, 4096),
                   (4096, 16384), (16384, 10 ** 7)):
        xs = [c for c in alone if lo <= (c["processed_tokens"] or 0) < hi]
        out["by_processed_tokens"][f"{lo}-{hi - 1}"] = {
            "total_ms": q([c["total_ms"] for c in xs]),
            "prefill_ms": q([c["prefill_ms"] for c in xs])}
    out["reads_beyond_first"] = sum(1 for c in CALLS if c["reads"] > 1)
    out["labels_unread_after_reads"] = sum(1 for c in CALLS
                                           if c["labels_unread"])
    lm = sorted(c["label_mass"] for c in CALLS)
    out["label_mass"] = ({"min": lm[0], "p10": lm[len(lm) // 10],
                          "median": statistics.median(lm)} if lm else None)
    rel = [b["release"] for b in BATCHES]
    out["releases"] = {"batches": len(rel),
                       "released": sum(1 for r in rel if r.get("released")),
                       "kept": sum(1 for r in rel if r.get("skipped")
                                   == "keep_slot"),
                       "ms": q([r.get("ms") for r in rel
                                if r.get("released")])}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only",
                    default="intent,phase,h4,packages,batching,load")
    DT.add_args(ap)
    a = ap.parse_args(argv)
    global TARGET, OUT
    try:
        TARGET = DT.from_args(a)
    except (DT.NotRun, ValueError) as e:
        print(f"NOT RUN: {e}")
        return 3
    if TARGET.mode != "one door":
        bad = [p for p in a.only.split(",") if p in STACK_PARTS]
        if bad:
            print(f"NOT RUN: {bad} measure the served stack's slots, not a "
                  "model; run them without --model/--base-url")
            return 3
        DT.install(TARGET, decisions_dir=os.path.join(
            ROOT, "logs", "decider_measure"))
        OUT = os.path.join(HERE, "results",
                           f"bonsai_decider.{TARGET.model}.json")
    res = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "target": TARGET.describe(),
           "preflight": preflight(),
           "template": {"version": D.TEMPLATE_VERSION, "system": D.SYSTEM,
                        "state_head": D.STATE_HEAD, "yes_no": D.YES_NO,
                        "choice": D.CHOICE, "answer_lead": D.ANSWER_LEAD},
           "decision": "argmax over the question's labels; no threshold",
           "in_sample": "every label here is ours; nothing is held out",
           "status": "complete"}
    if os.path.exists(OUT):
        try:
            with open(OUT, encoding="utf-8") as f:
                old = json.load(f)
            for k in ("intent", "phase", "h4", "packages", "batching",
                      "load", "calib", "load_small", "h4turns",
                      "daily_choose"):
                if k in old and k not in a.only.split(","):
                    res[k] = old[k]
        except (OSError, ValueError):
            pass
    parts = {"intent": part_intent, "phase": part_phase, "h4": part_h4,
             "packages": part_packages, "batching": part_batching,
             "load": part_load, "calib": part_calib,
             "load_small": part_load_small, "h4turns": part_h4turns,
             "daily_choose": part_daily_choose}
    code = 0
    try:
        for name in a.only.split(","):
            t0 = time.time()
            res[name] = parts[name]()
            res[name]["seconds"] = round(time.time() - t0, 1)
            print(f"[{name}] done in {res[name]['seconds']} s, "
                  f"{len(CALLS)} calls so far", flush=True)
            _write(res)
    except Busy as e:
        res["status"] = f"STOPPED: {e}"
        print(res["status"], flush=True)
        code = 2
    except Exception as e:                                       # noqa: BLE001
        import traceback
        traceback.print_exc()
        res["status"] = f"FAILED: {type(e).__name__}: {e}"[:400]
        print(res["status"], flush=True)
        code = 1
    _write(res)
    print("wrote", OUT)
    return code


def _write(res: dict) -> None:
    res["latency"] = latency_summary()
    res["calls_n"] = len(CALLS)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    with open(OUT.replace(".json", f".calls.{res['started'][:10]}.jsonl"),
              "w", encoding="utf-8") as f:
        for c in CALLS:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    sys.exit(main())
