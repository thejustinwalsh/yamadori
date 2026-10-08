#!/usr/bin/env python
"""THE TRAINED DECISION MODEL ON JEVBENCH'S 231 PUBLIC ITEMS, scored by JevBench's own code.

    # in the unsloth venv (GPU): records
    .venv-unsloth/Scripts/python.exe bench/decider/unsloth/eval_jevbench.py --ckpt CKPT --out RUN_DIR
    # in the stack python (offline): the same summary jjava's run has, beside Jev 1.13.0 / reflex-27b / SimpleJev
    python bench/decider/unsloth/eval_jevbench.py --summarize RUN_DIR

The items, mapping and scoring are bench/decider/jevbench_run.py's (JevBench commit bb05a33, MIT, the clone stays outside
the repo): a JevBench question IS a Clef/Jev question {type, instructions, criteria}; a choice's criteria become
{label: description}; a score's levels are the criteria list (labels "0".."k-1"); a noul is yes/no. The state goes in
as the item gives it (a JSON object stays an object). CONTAMINATION: prepare_data.py decontaminated the training mix
against these items' text (13-word shingles), and the guide's mix does not include JevBench; MMLU/ARC/BANKING77/CLINC150
train splits ARE in the mix, so items derived from those benchmarks (JevBench's public items are authored or drawn from
its own families) are not guaranteed out-of-distribution; the run records nothing about that beyond this note.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.environ.setdefault("HF_HOME", "C:/Users/jwals/textgen/user_data/models/hf-unsloth")
CLONE = os.environ.get("YAMADORI_JEVBENCH_DIR") or (
    "C:/Users/jwals/AppData/Local/Temp/claude/C--Users-jwals-llama-stack/"
    "d16e1f09-4699-483b-a257-1ba77b329ce7/scratchpad/jevbench")
QNAME = "decision"


def clef_question(task) -> dict:
    q = task.question
    t, text, crit = q["type"], q.get("instructions") or "", q.get("criteria")
    out = {"type": t, "instructions": text}
    if t == "noul":
        if isinstance(crit, dict):
            out["criteria"] = crit
    elif t == "choice":
        labels = [str(x) for x in task.labels]
        if isinstance(crit, dict):
            out["criteria"] = {lab: str(crit.get(lab, lab)) for lab in labels}
        else:
            opts = [str(x) for x in (crit or labels)]
            out["criteria"] = dict(zip(labels, opts))
    else:
        out["criteria"] = [str(x) for x in (crit or task.labels)]
    return out


def collect(args) -> int:
    sys.path.insert(0, CLONE)
    from jevbench import scoring, tasks
    from unsloth import FastDecisionModel
    splits = {"easy.jsonl": "easy", "original.jsonl": "standard", "hard.jsonl": "hard"}
    items = []
    for fn, tier in splits.items():
        for t in tasks.load_jsonl(os.path.join(CLONE, "datasets", "public", fn)):
            items.append((tier, t))
    model, tok = FastDecisionModel.from_pretrained(args.ckpt, max_seq_length=4096, load_in_4bit=True)
    FastDecisionModel.for_inference(model)
    os.makedirs(args.out, exist_ok=True)
    t00 = time.time()
    with open(os.path.join(args.out, "items.jsonl"), "w", encoding="utf-8") as f:
        for tier, task in items:
            rec = {"task_id": task.id, "tier": tier, "family": task.family, "type": task.question["type"],
                   "group": task.group, "n_labels": len(task.labels)}
            t0 = time.perf_counter()
            try:
                ans = FastDecisionModel.predict(model, tok, task.state, {QNAME: clef_question(task)})[QNAME]
                pr = ans["probabilities"]
                if task.question["type"] == "noul":
                    p = float(pr["true"])
                    probs = {"yes": p, "no": 1.0 - p}
                else:
                    probs = {str(k): float(v) for k, v in pr.items()}
                s = scoring.score_task(probs, task)
                rec.update(ok=True, latency_s=time.perf_counter() - t0, valid=s["valid"],
                           strict_valid=s.get("strict_valid"), correct=s["correct"], predicted=s.get("predicted"),
                           ordinal_ev=s.get("ordinal_ev"), probs=s.get("probs"))
            except Exception as e:  # recorded, scored wrong, never hidden
                rec.update(ok=False, error=str(e)[:300], latency_s=time.perf_counter() - t0, valid=False,
                           correct=False, predicted=None)
            f.write(json.dumps(rec) + "\n")
            f.flush()
    print("done", len(items), "items in", round(time.time() - t00, 1), "s", flush=True)
    return 0


def summarize(run_dir: str) -> int:
    sys.path.insert(0, os.path.join(ROOT, "bench", "decider"))
    sys.path.insert(0, os.path.join(ROOT, "mcp"))
    import jevbench_run as J
    jb = J.jevbench(CLONE)
    items = J.load_items(CLONE, jb)
    recs = [json.loads(l) for l in open(os.path.join(run_dir, "items.jsonl"), encoding="utf-8") if l.strip()]
    s = J.summarize(recs, items, jb, CLONE)
    s["model"] = "unsloth-clef-head (" + os.path.basename(run_dir.rstrip("/\\")) + ")"
    open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8").write(json.dumps(s, indent=1))
    keep = {k: s[k] for k in ("n_items", "n_run", "n_failed", "tiers", "intelligence_public", "calibration_public_hard",
                              "ece_public_hard", "latency_s")}
    keep["vs"] = {k: {"intelligence_public": v["intelligence_public"], "paired": v["paired_on_items_run"]}
                  for k, v in s["references"].items()}
    print(json.dumps(keep, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--out")
    ap.add_argument("--summarize")
    a = ap.parse_args()
    if a.summarize:
        return summarize(a.summarize)
    if not (a.ckpt and a.out):
        ap.error("--ckpt and --out, or --summarize RUN_DIR")
    return collect(a)


if __name__ == "__main__":
    sys.exit(main())
