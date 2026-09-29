#!/usr/bin/env python
"""JEVBENCH FOR JJAVA: JevBench's public items asked of our decider (the
served Bonsai, through mcp/model.py -- the one door) as Jev's typed
questions, scored by JevBench's OWN code, beside the Jev 1.13.0,
reflex-27b and SimpleJev Qwen3.8-27B rows of the same release on the SAME
items.

    python bench/decider/jevbench_run.py                     # the plan
    python bench/decider/jevbench_run.py --run               # GPU: runs it
    python bench/decider/jevbench_run.py --run --resume      # continue
    python bench/decider/jevbench_run.py --score RUN_DIR     # re-score

Operator, 2026-09-29, approved. JevBench (github.com/fstandhartinger/
jevbench, MIT) is NOT copied into this repo: --jevbench names a clone
(default $YAMADORI_JEVBENCH_DIR, else the session scratchpad the operator
cloned it into), and the run records its commit (PINNED below; a different
one is refused unless --any-commit) and each dataset file's canonical hash
(jevbench.tasks.dataset_hash; a Windows checkout's CRLF changes the raw
bytes) against the clone's datasets/manifest.json -- a mismatch is refused.
The harness's scoring is imported from
the clone (jevbench.scoring.score_task, jevbench.metrics.ece_top_label,
jevbench.composite_v13), so a decision is scored exactly as JevBench scores
it.

THE ITEMS: datasets/public/{easy, original, hard}.jsonl -- the PUBLIC half
only (231 decisions: easy 48 of 72, standard 72 of 96, hard 111 of 220;
judge 0 of 146; the rest is held out and not published). So the release's
own Intelligence and Calibration (over all 534) cannot be reproduced; the
COMPARABLE numbers are the per-tier accuracies on these 231 items, with
the three reference systems' per-item outcomes on the same items read from
the clone (results/v1.2/jevbench-v1.2-per-task.json, the v1.3.0 release:
RESULTS-v1.2.md), and a paired count per reference.

THE MAPPING (Jev's request form is JevBench's item form: {type,
instructions, criteria}):
  state   a string as is; a JSON object as json.dumps(indent=2) (Jev takes
          structured state; the decider reads text)
  noul    decider_bonsai.q_noul(instructions, criteria {true, false})
          -> probs {"yes": noul, "no": 1 - noul}  (JevBench's TypeSafe
          adapter maps Jev's noul the same way: jevbench/adapters/
          typesafe.py)
  choice  q_choice(instructions, [criteria[label] for label in labels],
          keys=labels) -> probs = the answer's `probabilities`
  score   q_score(instructions, criteria levels) -> probs = the answer's
          `probabilities`, keyed "0".."k" as JevBench's labels are
  more than 26 options -> refused before anything is sent, scored as a
          failure (counts wrong, as JevBench scores any invalid answer);
          the public items have at most 6
  Each item is ONE decide() batch (one transient slot, the state placed
  once, the two orders read after it), no state cut (Jev has none; the
  hard tier's long policies run to ~6k tokens).

THE SCORES (jevbench/composite_v13.py, the v1.3.0 release):
  tier accuracy     argmax (JevBench's, ties to the lexicographically
                    smallest label), failures wrong, per tier
  intelligence      composite_v13.intelligence (chance-corrected, weights
                    easy 14 / standard 28 / hard 30, judge missing and
                    renormalised away) with the chance of the PUBLIC items
                    (1/options per item) -- and, labelled, with the
                    release's TIER_CHANCES (all frozen items)
  calibration       composite_v13.calibration: 100 (1 - ECE/0.5) on the
                    public hard items (metrics.ece_top_label, top-label
                    probability, 10 bins) and 100 (1 - mean TVD) to the
                    exact gold distributions of the public probability
                    items, averaged
  latency           p50 / p95 of the wall time per decision -- NOT
                    comparable (our card, shared, in the same room; theirs
                    rented GPUs over the network)
  The reference rows' calibration is published over all 220 hard items
  only, so theirs is quoted, not recomputed on the public half.

ONE GPU CONSUMER: bench/decider/bonsai_decider.py preflight before the run
and every CHECK_EVERY items; a DeciderUnavailable that a retry cannot fix,
or three in a row, stops the run (the rest NOT RUN, never scored as
wrong); --resume continues. Written: bench/decider/results/jevbench/<run>/
{items.jsonl, summary.json} -- item ids and our distributions, never item
text.

ANOTHER MODEL (2026-09-29; docs/JJAVA.md "Per model"): --model / --base-url
/ --slot (bench/decider/decider_target.py) ask that engine instead, with the
preflight on ITS server; the run directory is named <model>-<time>.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "mcp"))

import decider_bonsai as D  # noqa: E402

RUN_VERSION = "jevbench-jjava/1"
PINNED_COMMIT = "bb05a335bc809e61b20c0f745d25499a82b326fc"
DEFAULT_CLONE = os.environ.get("YAMADORI_JEVBENCH_DIR") or (
    "C:/Users/jwals/AppData/Local/Temp/claude/C--Users-jwals-llama-stack/"
    "d16e1f09-4699-483b-a257-1ba77b329ce7/scratchpad/jevbench")
OUT_DIR = os.path.join(HERE, "results", "jevbench")
# file -> (manifest split name, JevBench tier)
SPLITS = {"easy.jsonl": ("easy", "easy"),
          "original.jsonl": ("original", "standard"),
          "hard.jsonl": ("hard", "hard")}
# The reference rows (the v1.3.0 release, results/v1.2): key -> display.
REFERENCES = {"jev-1.13.0": "Jev 1.13.0",
              "reflex-27b": "reflex-27b (Qwen3.8-27B)",
              "simplejev-qwen3.8-27b": "SimpleJev Qwen3.8-27B"}
CHECK_EVERY = 25
QNAME = "decision"


# ------------------------------------------------------------- the clone --
def clone_state(clone: str) -> dict:
    """The clone's commit and each public file's sha256, checked against
    its manifest."""
    commit = None
    try:
        commit = subprocess.run(["git", "-C", clone, "rev-parse", "HEAD"],
                                capture_output=True, text=True,
                                timeout=20).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        pass
    man = {}
    try:
        with open(os.path.join(clone, "datasets", "manifest.json"),
                  encoding="utf-8") as f:
            man = {s["name"]: s for s in json.load(f).get("splits", [])}
    except (OSError, ValueError):
        pass
    files = {}
    jb = None
    try:
        jb = jevbench(clone)
    except Exception:                                            # noqa: BLE001
        pass
    for fn, (split, tier) in SPLITS.items():
        p = os.path.join(clone, "datasets", "public", fn)
        sha = (hashlib.sha256(open(p, "rb").read()).hexdigest()
               if os.path.exists(p) else None)
        # The CANONICAL hash (jevbench.tasks.dataset_hash over the parsed
        # records) is what is compared: a Windows checkout's CRLF line ends
        # change the file's bytes, not its records.
        canon = None
        if jb is not None and os.path.exists(p):
            canon = jb["tasks"].dataset_hash(jb["tasks"].load_jsonl(p))
        want = (man.get(split) or {}).get("canonical_sha256")
        files[fn] = {"tier": tier, "sha256": sha,
                     "raw_matches_manifest": sha is not None
                     and sha == (man.get(split) or {}).get("sha256"),
                     "canonical_sha256": canon,
                     "matches": canon is not None and canon == want}
    return {"path": clone, "commit": commit, "pinned": PINNED_COMMIT,
            "commit_ok": commit == PINNED_COMMIT, "files": files}


def jevbench(clone: str):
    """JevBench's own modules, imported from the clone."""
    if clone not in sys.path:
        sys.path.insert(0, clone)
    from jevbench import composite_v13, metrics, scoring, tasks
    return {"tasks": tasks, "scoring": scoring, "metrics": metrics,
            "composite": composite_v13}


def load_items(clone: str, jb: dict, splits=tuple(SPLITS)) -> list:
    """[(tier, Task)] in file order, easy then original then hard."""
    out = []
    for fn in splits:
        tier = SPLITS[fn][1]
        for t in jb["tasks"].load_jsonl(
                os.path.join(clone, "datasets", "public", fn)):
            out.append((tier, t))
    return out


# ------------------------------------------------------------ the mapping --
def state_text(state) -> str:
    return state if isinstance(state, str) else json.dumps(
        state, indent=2, ensure_ascii=False)


def to_question(task) -> dict:
    """A JevBench item's question as our typed question (Jev's primitives).
    Raises DeciderUnavailable for one we cannot ask (> 26 options)."""
    q = task.question
    t, text, crit = q["type"], q.get("instructions") or "", q.get("criteria")
    if t == "noul":
        return D.q_noul(QNAME, text, criteria=crit if isinstance(crit, dict)
                        else None)
    if t == "choice":
        labels = [str(x) for x in task.labels]
        if isinstance(crit, dict):
            opts = [str(crit.get(lab, lab)) for lab in labels]
        else:
            opts = [str(x) for x in (crit or labels)]
        return D.q_choice(QNAME, text, opts, keys=labels)
    levels = [str(x) for x in (crit or task.labels)]
    if len(levels) != len(task.labels):
        raise D.DeciderUnavailable(
            "LEVELS_MISMATCH", f"{task.id}: {len(levels)} levels, "
            f"{len(task.labels)} labels", False,
            "the maintainer: check the item's criteria")
    return D.q_score(QNAME, text, levels)


def to_probs(task, answer: dict) -> dict:
    """Our answer as JevBench's exact-label distribution."""
    if task.question["type"] == "noul":
        p = float(answer["noul"])
        return {"yes": p, "no": 1.0 - p}
    return {str(k): float(v) for k, v in answer["probabilities"].items()}


def run_item(tier: str, task, jb: dict, *, post=None, upstream=None
             ) -> dict:
    """One item through decider_bonsai.decide (model.post, the one door, by
    default), scored by jevbench.scoring.score_task."""
    kw = {}
    if post is not None:
        kw["post"] = post
    if upstream is not None:
        kw["upstream"] = upstream
    t0 = time.perf_counter()
    rec = {"task_id": task.id, "tier": tier, "family": task.family,
           "type": task.question["type"], "group": task.group,
           "n_labels": len(task.labels)}
    try:
        q = to_question(task)
        out = D.decide(state_text(task.state), [q], **kw)
    except D.DeciderUnavailable as e:
        rec.update(ok=False, error=e.facts(),
                   latency_s=time.perf_counter() - t0, valid=False,
                   correct=False, predicted=None)
        return rec
    a = out["answers"][QNAME]
    probs = to_probs(task, a)
    s = jb["scoring"].score_task(probs, task)
    d = a["diagnostics"]
    rec.update(ok=True, latency_s=time.perf_counter() - t0,
               valid=s["valid"], strict_valid=s.get("strict_valid"),
               correct=s["correct"], predicted=s.get("predicted"),
               ordinal_ev=s.get("ordinal_ev"), probs=s.get("probs"),
               answer={k: v for k, v in a.items() if k != "diagnostics"},
               usage=out.get("usage"), model=out.get("model"),
               diagnostics={k: d.get(k) for k in (
                   "disagreement", "argmax_agree", "label_mass_min",
                   "tie", "reads", "prompt_tokens", "processed_tokens",
                   "ms")})
    return rec


# ---------------------------------------------------------------- scoring --
def _pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    f, c = math.floor(k), math.ceil(k)
    return xs[f] if f == c else xs[f] * (c - k) + xs[c] * (k - f)


def public_chances(items: list, jb: dict) -> dict:
    by: dict = {}
    for tier, t in items:
        n = len(t.labels)
        by.setdefault(tier, {}).setdefault(n, 0)
        by[tier][n] += 1
    return {tier: jb["composite"].chance_from_option_counts(c)
            for tier, c in by.items()}


def tiers_of(outcomes: dict, items: list) -> dict:
    """{tier: accuracy} from {task_id: correct bool} over the given items
    (an item with no outcome counts wrong: JevBench's rule)."""
    acc: dict = {}
    for tier, t in items:
        a = acc.setdefault(tier, [0, 0])
        a[0] += int(bool(outcomes.get(t.id)))
        a[1] += 1
    return {tier: c / n for tier, (c, n) in acc.items()}


def references(clone: str) -> dict:
    """The reference rows: per-item public outcomes (c/w/f/n) and the
    release's published axes."""
    out = {}
    try:
        with open(os.path.join(clone, "results", "v1.2",
                               "jevbench-v1.2-per-task.json"),
                  encoding="utf-8") as f:
            per = json.load(f)
        with open(os.path.join(clone, "results", "v1.2",
                               "jevbench-v1.2-results.json"),
                  encoding="utf-8") as f:
            res = {s["key"]: s for s in json.load(f).get("systems", [])}
    except (OSError, ValueError):
        return out
    for key, disp in REFERENCES.items():
        s = (per.get("systems") or {}).get(key) or {}
        pt = s.get("public_tasks") or {}
        if isinstance(pt, str):
            continue
        r = res.get(key) or {}
        out[key] = {"display": disp, "revision": per.get("revision"),
                    "outcomes": {tid: v[0] for tid, v in pt.items()},
                    "published": {
                        "intelligence": (r.get("axes") or {}).get(
                            "intelligence"),
                        "calibration": (r.get("axes") or {}).get(
                            "calibration"),
                        "ece_hard": (r.get("calibration") or {}).get(
                            "ece_hard"),
                        "probability_fidelity": (r.get("calibration")
                                                 or {}).get(
                                                     "probability_fidelity"),
                        "tiers_all_items": r.get("tiers")}}
    return out


def mcnemar_p(b: int, c: int) -> float | None:
    """Exact two-sided McNemar p for b vs c discordant pairs."""
    n = b + c
    if n == 0:
        return None
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def summarize(records: list, items: list, jb: dict, clone: str) -> dict:
    comp, met = jb["composite"], jb["metrics"]
    by_id = {r["task_id"]: r for r in records}
    ours = {tid: bool(r.get("correct")) for tid, r in by_id.items()}
    done = [(tier, t) for tier, t in items if t.id in by_id]
    ch = public_chances(items, jb)
    tiers = tiers_of(ours, items)
    hard = [by_id[t.id] for tier, t in done if tier == "hard"]
    pairs = [(max(r["probs"].values()), bool(r["correct"])) for r in hard
             if r.get("valid") and r.get("probs")]
    e = met.ece_top_label(pairs)["ece"] if pairs else None
    tv = []
    for tier, t in done:
        g = (t.provenance or {}).get("gold_probs")
        r = by_id[t.id]
        if g and r.get("probs"):
            tv.append(comp.tvd(r["probs"], g, t.labels))
    mean_tvd = sum(tv) / len(tv) if tv else None
    lat = [r.get("latency_s") for r in records if r.get("ok")]
    out = {"version": RUN_VERSION, "release": "JevBench v1.3.0 scoring "
           "(jevbench/composite_v13.py; reference rows results/v1.2)",
           "n_items": len(items), "n_run": len(records),
           "n_failed": sum(1 for r in records if not r.get("ok")),
           "complete": len(records) == len(items),
           "tiers": {k: round(v, 4) for k, v in tiers.items()},
           "intelligence_public": comp.intelligence(tiers, ch),
           "intelligence_release_chances": comp.intelligence(tiers),
           "public_chances": ch,
           "calibration_public_hard": comp.calibration(e, mean_tvd),
           "ece_public_hard": e, "n_ece": len(pairs),
           "mean_tvd_public_probability": mean_tvd, "n_tvd": len(tv),
           "latency_s": {"p50": _pct(lat, 0.5), "p95": _pct(lat, 0.95),
                         "note": "not comparable: our shared card vs "
                                 "their rented GPUs over the network"},
           "references": {}}
    for key, ref in references(clone).items():
        oc = {tid: v == "c" for tid, v in ref["outcomes"].items()}
        rt = tiers_of(oc, items)
        b = sum(1 for _tier, t in done if ours[t.id] and not oc.get(t.id))
        c = sum(1 for _tier, t in done if oc.get(t.id) and not ours[t.id])
        out["references"][key] = {
            "display": ref["display"],
            "tiers_public": {k: round(v, 4) for k, v in rt.items()},
            "intelligence_public": comp.intelligence(rt, ch),
            "paired_on_items_run": {"n": len(done), "only_ours_right": b,
                                    "only_theirs_right": c,
                                    "mcnemar_p": mcnemar_p(b, c)},
            "published_all_items": ref["published"]}
    return out


# -------------------------------------------------------------------- run --
def plan(clone: str, state: dict, items: list | None) -> str:
    lines = ["THE PLAN (nothing is sent without --run):",
             f"  JevBench clone: {clone}",
             f"    commit {state['commit']} (pinned {PINNED_COMMIT}: "
             f"{'OK' if state['commit_ok'] else 'DIFFERENT -- refused'})"]
    for fn, f in state["files"].items():
        lines.append(f"    {fn:<14} tier {f['tier']:<8} canonical "
                     f"{(f['canonical_sha256'] or 'missing')[:16]} "
                     f"{'= manifest' if f['matches'] else '!= manifest'}"
                     + ("" if f["raw_matches_manifest"] else
                        " (raw bytes differ: line endings)"))
    if items is not None:
        n = {}
        for tier, t in items:
            n[tier] = n.get(tier, 0) + 1
        lines.append(f"  items: {len(items)} public decisions {n}; each "
                     "one decide() batch, two reads (two orders)")
    lines += ["  model: the served Bonsai through mcp/model.py post() "
              "(decider_bonsai.decide), thinking off, one token per read",
              "  preflight: no main-model slot processing, no hermes.exe "
              f"(every {CHECK_EVERY} items)",
              f"  writes: {os.path.relpath(OUT_DIR, ROOT)}/<run>/"
              "{items.jsonl, summary.json} (ids and distributions only)",
              "  command (when the card is free):",
              "    C:\\Users\\jwals\\textgen\\installer_files\\env\\"
              "python.exe bench/decider/jevbench_run.py --run"
              + ("" if os.path.normcase(os.path.abspath(clone))
                 == os.path.normcase(os.path.abspath(DEFAULT_CLONE))
                 else f" --jevbench {clone}"),
              "  a clone elsewhere: git clone https://github.com/"
              f"fstandhartinger/jevbench && git -C jevbench checkout "
              f"{PINNED_COMMIT}, then --jevbench <dir>"]
    return "\n".join(lines)


def run(clone: str, run_dir: str, items: list, jb: dict, *, resume=False,
        post=None, upstream=None, preflight=None, log=print) -> list:
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "items.jsonl")
    done = {}
    if resume and os.path.exists(path):
        for ln in open(path, encoding="utf-8"):
            r = json.loads(ln)
            if r.get("ok"):
                done[r["task_id"]] = r
    records = list(done.values())
    fails = 0
    with open(path, "w" if not resume else "a", encoding="utf-8") as f:
        if resume:
            f.seek(0, 2)
        for i, (tier, t) in enumerate(items):
            if t.id in done:
                continue
            if preflight and i % CHECK_EVERY == 0:
                preflight()
            r = run_item(tier, t, jb, post=post, upstream=upstream)
            if not r["ok"] and (r["error"] or {}).get("retryable"):
                fails += 1
                if fails >= 3:
                    log(f"STOP: three decider failures in a row "
                        f"({r['error']}); the rest NOT RUN (--resume)")
                    break
                continue
            fails = 0
            records.append(r)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 10 == 0:
                log(f"  {i + 1}/{len(items)}")
    return records


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--jevbench", default=DEFAULT_CLONE)
    ap.add_argument("--run", action="store_true", help="run it (GPU)")
    ap.add_argument("--resume", metavar="RUN_DIR", nargs="?", const="",
                    help="continue a run (default: the newest)")
    ap.add_argument("--score", metavar="RUN_DIR",
                    help="re-score a run's items.jsonl, no model")
    ap.add_argument("--limit", type=int, help="the first N items only")
    ap.add_argument("--any-commit", action="store_true",
                    help="run on a clone at another commit (recorded)")
    sys.path.insert(0, HERE)
    import decider_target as DT
    DT.add_args(ap)
    a = ap.parse_args(argv)
    clone = os.path.abspath(a.jevbench)
    st = clone_state(clone)
    items = None
    try:
        jb = jevbench(clone)
        items = load_items(clone, jb)
    except Exception as e:                                       # noqa: BLE001
        print(f"JevBench not loadable from {clone}: {type(e).__name__}: "
              f"{e}")
        print(plan(clone, st, None))
        return 2
    if a.limit:
        items = items[:a.limit]
    if a.score:
        recs = [json.loads(ln) for ln in open(
            os.path.join(a.score, "items.jsonl"), encoding="utf-8")]
        s = summarize(recs, items, jb, clone)
        print(json.dumps(s, indent=1))
        return 0
    print(plan(clone, st, items))
    if not a.run:
        return 0
    if not st["commit_ok"] and not a.any_commit:
        print(f"REFUSED: the clone is at {st['commit']}, not {PINNED_COMMIT}"
              " (--any-commit runs it anyway, recorded)")
        return 2
    if not all(f["matches"] for f in st["files"].values()):
        print("REFUSED: a dataset file does not match the clone's manifest")
        return 2
    from bonsai_decider import Busy, preflight
    try:
        target = DT.from_args(a)
    except (DT.NotRun, ValueError) as e:
        print(f"NOT RUN: {e}")
        return 3
    prefix = ""
    if target.mode != "one door":
        DT.install(target, decisions_dir=os.path.join(ROOT, "logs",
                                                      "decider_measure"))
        prefix = f"{target.model}-"

        def preflight():
            return DT.preflight(target)
    if a.resume is not None:
        runs = sorted(r for r in (os.listdir(OUT_DIR) if os.path.isdir(
            OUT_DIR) else []) if (r.startswith(prefix) if prefix
                                  else r[:1].isdigit()))
        run_dir = a.resume or (os.path.join(OUT_DIR, runs[-1]) if runs
                               else None)
        if not run_dir:
            print("nothing to resume")
            return 2
    else:
        run_dir = os.path.join(OUT_DIR, prefix
                               + time.strftime("%Y%m%d-%H%M%S"))
    t0 = time.time()
    status = "complete"
    try:
        recs = run(clone, run_dir, items, jb, resume=a.resume is not None,
                   preflight=preflight)
    except Busy as e:
        status, recs = f"STOPPED (not run): {e}", [
            json.loads(ln) for ln in open(os.path.join(
                run_dir, "items.jsonl"), encoding="utf-8")]
    s = summarize(recs, items, jb, clone)
    s.update(status=status if s["complete"] else f"partial: {status}",
             clone=st, model=D.model_name(), target=target.describe(),
             model_profile=D.profile_status(),
             template=D.TEMPLATE_VERSION,
             readout=D.READOUT_VERSION,
             seconds=round(time.time() - t0, 1),
             started=os.path.basename(run_dir))
    with open(os.path.join(run_dir, "summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(s, f, indent=1)
    print(json.dumps({k: s[k] for k in ("status", "tiers",
                                        "intelligence_public",
                                        "calibration_public_hard")},
                     indent=1))
    print("->", run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
