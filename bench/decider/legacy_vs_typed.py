#!/usr/bin/env python
"""THE ONE SIDE-BY-SIDE: the decider's LEGACY readout vs the TYPED one (Jev's
API), on the served Bonsai, before the legacy readout is deleted.

    python bench/decider/legacy_vs_typed.py            # prints the plan
    python bench/decider/legacy_vs_typed.py --run      # runs it (GPU)
    python bench/decider/legacy_vs_typed.py --run --only intent
    python bench/decider/legacy_vs_typed.py --run --model flash-next \
        --base-url http://127.0.0.1:18095      # another model (below)

PER MODEL (2026-09-29; docs/JJAVA.md "Per model"): --model / --base-url /
--slot read that engine (bench/decider/decider_target.py: the decider's
doors patched to it, the preflight on its server, its decisions logged to
logs/decider_measure/<model>.decisions.jsonl) and write
results/legacy_vs_typed.<model>.json. bench/decider/measure_model.py runs
this as its last part and files summary() in the model's profile record.

Operator, 2026-09-29: "Stop keeping old shit behind switches." The legacy
readout (decide_turn READOUT=legacy: the count-only rotation plus the
divided-out label prior in choose() / pick() / judge_stop; the 2026-09-27
FORM table) stays ONLY until this has run once; then it goes (docs/
CONSTANTS-AUDIT.md "Decider legacy readout" lists every name). This script
is written to run later -- the card was busy when it was written -- and it
has not run.

WHERE THE TWO READOUTS DIFFER, and so what is measured:
  daily_choose  the skill selector's picks with decide_turn.choose() live
                (the one place the readouts differ materially: legacy =
                one rotated read with the label prior divided out; typed =
                two orders, positional letters, no prior), scored by the
                daily eval's own judge (bench/skills/replay_selection.py
                daily(): MUST / MUST-NOT, bench/skills/daily_eval.jsonl).
                IN-SAMPLE: those labels were written beside the selector.
                The embedding stage is not run (the A4000), as in
                bonsai_decider.py daily_choose. Per arm: checks passed,
                the failures, wall ms per request.
  intent        build intent on bench/skills/work_intent.jsonl (49 yes / 50
                no, labelled by hand): per arm accuracy and the top-label
                ECE (10 bins) of p(yes). The two readouts render build
                intent BYTE FOR BYTE alike (decide_turn's typed noul is the
                measured mc_avg form; mcp/test_decide_turn.py checks it), so
                this part is a CONTROL: any difference is batch
                nondeterminism, reported as the largest |p_legacy -
                p_typed| beside decider_bonsai.TIE_BAND.
  (no judge_stop part: no labelled set of stopped steps exists.)

ONE GPU CONSUMER (AGENTS.md): bonsai_decider.preflight before each part and
every CHECK_EVERY items -- no main-model slot processing, no hermes.exe --
else the run stops (exit 2) and says so; a 429 or a DeciderUnavailable is
NOT RUN, never a result. The arms run one after the other in the order
given by --order (default legacy first); PROTOCOL's single-run caution
applies: one run is a comparison, not a verdict.

Written: bench/decider/results/legacy_vs_typed.json (with the git state of
the decider files, the template and readout versions, and the model); for
another model, legacy_vs_typed.<model>.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "results", "legacy_vs_typed.json")
INTENT = os.path.join(ROOT, "bench", "skills", "work_intent.jsonl")
DAILY = os.path.join(ROOT, "bench", "skills", "daily_eval.jsonl")
CHECK_EVERY = 25
ARMS = ("legacy", "typed")
PARTS = ("daily_choose", "intent")

sys.path.insert(0, os.path.join(ROOT, "mcp"))
sys.path.insert(0, HERE)


def out_path(model: str | None = None) -> str:
    """The result file: the default model's is legacy_vs_typed.json (as
    before), another's legacy_vs_typed.<model>.json."""
    import decider_bonsai as D
    import model as M
    if not model or D.canonical_model(model) == D.canonical_model(M.MODEL):
        return OUT
    return os.path.join(HERE, "results", f"legacy_vs_typed.{model}.json")


def plan(order=ARMS, only=PARTS, target=None) -> str:
    n_intent = sum(1 for ln in open(INTENT, encoding="utf-8")
                   if ln.strip() and not ln.startswith("//")) \
        if os.path.exists(INTENT) else "?"
    n_daily = sum(1 for ln in open(DAILY, encoding="utf-8")
                  if ln.strip() and not ln.startswith("//")) \
        if os.path.exists(DAILY) else "?"
    lines = ["THE PLAN (nothing is sent without --run):",
             (f"  target: {target.model} ({target.mode}, {target.base}, "
              f"slot {target.slot})" if target is not None else
              "  target: the stack's main model through the one door"),
             f"  arms, in order: {', '.join(order)} "
             "(decide_turn.READOUT flipped per arm; the label-prior cache "
             "cleared between arms)"]
    if "daily_choose" in only:
        lines.append(f"  daily_choose: {n_daily} daily-eval rows through "
                     "skill_select.decide inside an open Turn, per arm; "
                     "scored MUST / MUST-NOT")
    if "intent" in only:
        lines.append(f"  intent: {n_intent} work_intent rows, build intent "
                     "per arm; accuracy, ECE, max |p_legacy - p_typed| (a "
                     "control: identical renderings)")
    lines += ["  preflight: no main-model slot processing, no hermes.exe "
              "(bench/decider/bonsai_decider.py preflight; another model: "
              "ITS server's slots, decider_target.preflight)",
              "  writes: " + os.path.relpath(
                  out_path(target.model if target else None), ROOT),
              "  command: C:\\Users\\jwals\\textgen\\installer_files\\env\\"
              "python.exe bench/decider/legacy_vs_typed.py --run"
              + (f" --model {target.model}" if target is not None else "")
              + (f" --base-url {target.base}" if target is not None
                 and target.mode == "base-url" else "")]
    return "\n".join(lines)


# ------------------------------------------------------------------ parts --
def _rows(path: str) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln and not ln.startswith("//"):
                out.append(json.loads(ln))
    return out


def ece(p_yes: list[float], truth: list[bool], bins: int = 10) -> float:
    """Top-label ECE of a yes/no read (jevbench metrics.ece_top_label)."""
    acc: dict = {}
    for p, y in zip(p_yes, truth):
        c = max(p, 1 - p)
        ok = (p >= 0.5) == y
        i = min(int(c * bins), bins - 1)
        a = acc.setdefault(i, [0, 0.0, 0])
        a[0] += 1
        a[1] += c
        a[2] += int(ok)
    n = len(p_yes) or 1
    return round(sum(v[0] / n * abs(v[2] / v[0] - v[1] / v[0])
                     for v in acc.values()), 6)


def part_intent(arms, preflight, decide_turn) -> dict:
    rows = _rows(INTENT)
    per = {}
    for arm in arms:
        decide_turn.READOUT = arm
        decide_turn.TEMPLATE_CF.clear()
        preflight()
        ps, t0 = [], time.time()
        for i, r in enumerate(rows):
            if i and i % CHECK_EVERY == 0:
                preflight()
            bi = decide_turn.build_intent(
                [{"role": "user", "content": r["text"]}], on=True,
                key="legacy_vs_typed", request=f"intent:{arm}:{i}")
            if bi.get("source") == "unavailable":
                raise RuntimeError(f"NOT RUN: {bi.get('failure')}")
            ps.append(float((bi.get("p") or {}).get("yes", 0.5)))
        truth = [bool(r["intent"]) for r in rows]
        per[arm] = {"n": len(rows),
                    "accuracy": round(sum((p >= 0.5) == y for p, y in
                                          zip(ps, truth)) / len(rows), 4),
                    "ece": ece(ps, truth), "p_yes": ps,
                    "seconds": round(time.time() - t0, 1)}
    out = {"arms": per}
    if len(per) == 2:
        a, b = (per[x]["p_yes"] for x in arms)
        import decider_bonsai as D
        out["max_abs_diff"] = round(max(abs(x - y) for x, y in zip(a, b)), 6)
        out["tie_band"] = D.TIE_BAND
        out["argmax_agree"] = sum((x >= 0.5) == (y >= 0.5)
                                  for x, y in zip(a, b))
    return out


def part_daily_choose(arms, preflight, decide_turn) -> dict:
    sys.path.insert(0, os.path.join(ROOT, "bench", "skills"))
    argv = sys.argv
    sys.argv = [sys.argv[0]]
    gpu = os.environ.get("YAMADORI_GPU_ROOM")
    import replay_selection as R                     # copies the jobs store
    sys.argv = argv
    if gpu is None:
        os.environ.pop("YAMADORI_GPU_ROOM", None)
    import skill_match
    import skill_select
    skill_match.rank_all = lambda state, pool, embed=None: (None, {
        "ok": False, "why": "bench: the embedding stage is not run"})
    decide_turn.enable(prime=False)
    pool = R.skills.armed()
    real = skill_select.decide
    out = {}
    for arm in arms:
        decide_turn.READOUT = arm
        decide_turn.TEMPLATE_CF.clear()
        preflight()
        walls: list[float] = []

        def wrapped(messages, *a, **kw):
            t0 = time.time()
            with decide_turn.Turn(messages, key="legacy_vs_typed", on=True):
                res = real(messages, *a, **kw)
            walls.append((time.time() - t0) * 1000)
            return res
        skill_select.decide = wrapped
        try:
            res = R.daily(pool)
        finally:
            skill_select.decide = real
        checks = res["checks"]
        xs = sorted(walls)
        out[arm] = {"passed": sum(1 for ok, _ in checks if ok),
                    "total": len(checks),
                    "failed": [w for ok, w in checks if not ok],
                    "per_area": res["per_area"],
                    "wall_ms": ({"n": len(xs),
                                 "p50": round(statistics.median(xs), 1),
                                 "p90": round(xs[int(0.9 * (len(xs) - 1))],
                                              1)} if xs else None)}
    if len(out) == 2:
        a, b = (set(out[x]["failed"]) for x in arms)
        out["only_legacy_failed"] = sorted(a - b)
        out["only_typed_failed"] = sorted(b - a)
    return out


def _files_state() -> dict:
    out = {}
    for rel in ("mcp/decider_bonsai.py", "mcp/decide_turn.py",
                "mcp/skill_match.py", "mcp/skill_select.py"):
        p = os.path.join(ROOT, rel)
        if os.path.exists(p):
            out[rel] = hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
    return out


def summary(res: dict) -> dict:
    """What a model's profile record keeps of a result (measure_model): per
    arm accuracy / ECE / checks passed, the control's largest difference,
    the status -- no per-row distributions."""
    out = {"status": res.get("status"), "order": res.get("order"),
           "runs": res.get("runs"), "in_sample": True}
    it = res.get("intent") or {}
    if it:
        out["intent"] = {arm: {k: v.get(k) for k in ("n", "accuracy", "ece")}
                         for arm, v in (it.get("arms") or {}).items()}
        for k in ("max_abs_diff", "tie_band", "argmax_agree"):
            if k in it:
                out["intent"][k] = it[k]
    dc = res.get("daily_choose") or {}
    if dc:
        out["daily_choose"] = {arm: {k: dc[arm].get(k) for k in (
            "passed", "total", "wall_ms")} for arm in ARMS if arm in dc}
    return out


def run(order, only, target=None, preflight=None) -> tuple[dict, int]:
    """Both arms of every part in `only` on `target` (decider_target; the
    caller installs it) -> (result, exit code: 0 done, 1 failed, 2 busy, 3
    not run)."""
    import decide_turn
    import decider_bonsai as D
    import decider_target as DT
    if preflight is None:
        if target is not None and target.mode != "one door":
            def preflight():
                return DT.preflight(target)
        else:
            from bonsai_decider import preflight
    res = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "model": D.model_name(), "template": D.TEMPLATE_VERSION,
           "typed_readout": D.READOUT_VERSION, "order": order,
           "target": target.describe() if target is not None else None,
           "model_profile": D.profile_status(),
           "files": _files_state(), "status": "complete",
           "in_sample": "daily_eval and work_intent are ours; nothing is "
                        "held out", "runs": 1}
    code = 0
    try:
        for part in only:
            t0 = time.time()
            fn = {"intent": part_intent,
                  "daily_choose": part_daily_choose}[part]
            res[part] = fn(order, preflight, decide_turn)
            res[part]["seconds"] = round(time.time() - t0, 1)
            print(f"[{part}] done in {res[part]['seconds']} s", flush=True)
    except DT.Busy as e:
        res["status"] = f"STOPPED (not run): {e}"
        code = 2
    except DT.NotRun as e:
        res["status"] = f"NOT RUN: {e}"
        code = 3
    except Exception as e:                                       # noqa: BLE001
        res["status"] = f"FAILED: {type(e).__name__}: {e}"[:400]
        code = 1
    finally:
        decide_turn.READOUT = "typed"
    return res, code


def main(argv=None) -> int:
    import decider_target as DT
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", action="store_true",
                    help="run it (GPU); without it the plan is printed")
    ap.add_argument("--only", default=",".join(PARTS))
    ap.add_argument("--order", default=",".join(ARMS),
                    help="the arms in the order they run")
    DT.add_args(ap)
    a = ap.parse_args(argv)
    only = [x for x in a.only.split(",") if x]
    order = [x for x in a.order.split(",") if x]
    if set(order) - set(ARMS) or set(only) - set(PARTS):
        print(f"arms are {ARMS}, parts {PARTS}")
        return 2
    if not a.run:
        # The plan only: no engine is asked anything (resolve() would read
        # llama-swap's /running and the target's /slots).
        shown = None
        if a.model or a.base_url:
            shown = DT.Target(a.model or "(the main model)",
                              "base-url" if a.base_url else "llama-swap",
                              (a.base_url or "").rstrip("/")
                              or f"/upstream/{a.model}", a.slot)
        print(plan(order, only, shown))
        return 0
    try:
        target = DT.from_args(a)
    except (DT.NotRun, ValueError) as e:
        print(f"NOT RUN: {e}")
        return 3
    print(plan(order, only, target))
    if target.mode != "one door":
        DT.install(target, decisions_dir=os.path.join(ROOT, "logs",
                                                      "decider_measure"))
    res, code = run(order, only, target)
    path = out_path(target.model)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(res["status"], "->", path)
    return code


if __name__ == "__main__":
    sys.exit(main())
