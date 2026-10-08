#!/usr/bin/env python
"""THE TRAINED DECISION MODEL ON OUR SKILL-INJECTION CASES (relevance and need), in the unsloth venv.

    .venv-unsloth/Scripts/python.exe bench/decider/unsloth/eval_skills.py --ckpt index/decider_train/runs/q2b-e1/adapters \
        --items index/decider_train/skills_items.jsonl --out PRED.jsonl [--with-skill-name]

Two noul questions per (case, item), one record each (so the questions do not read each other):
  relevance  "Is the fact below about the code, libraries or task that the state is working on?"
  need       "Does the assistant need the fact below for the very next action it takes in this state?"
Output rows carry P(true) for each. Score with skills_score.py (stack python): AUROC against the rubric truth beside
jjava's recorded reads. The model was trained on the guide's mix only; nothing here is a training row.
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("HF_HOME", "C:/Users/jwals/textgen/user_data/models/hf-unsloth")

REL = "Is the fact below about the code, libraries or task that the state is working on?"
NEED = "Does the assistant need the fact below for the very next action it takes in this state?"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--with-skill-name", action="store_true")
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-seq", type=int, default=4096)
    a = ap.parse_args()
    from unsloth import FastDecisionModel
    model, tok = FastDecisionModel.from_pretrained(a.ckpt, max_seq_length=a.max_seq, load_in_4bit=not a.no_4bit)
    FastDecisionModel.for_inference(model)
    rows = [json.loads(l) for l in open(a.items, encoding="utf-8") if l.strip()]
    if a.limit:
        rows = rows[:a.limit]
    t0 = time.time()
    with open(a.out, "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            fact = r["fact"] if not a.with_skill_name else f"[{r['skill']}] {r['fact']}"
            out = {}
            for name, text in (("relevance", REL), ("need", NEED)):
                q = {name: {"type": "noul", "instructions": f"{text}\n\nFact: {fact}"}}
                ans = FastDecisionModel.predict(model, tok, r["state"], q)[name]
                out[name] = float(ans["probabilities"]["true"]) if "probabilities" in ans else float(ans["noul"])
            f.write(json.dumps({"case": r["case"], "key": r["key"], **out}) + "\n")
            if i % 100 == 0:
                print(i, len(rows), round(time.time() - t0, 1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
