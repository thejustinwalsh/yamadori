#!/usr/bin/env python
"""THE GUIDE'S DATA: Unsloth's decision-model mix and test split, built with Unsloth's own builder.

    .venv-unsloth/Scripts/python.exe bench/decider/unsloth/prepare_data.py --n-train 12000

Follows https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth (docs/research/DECIDER-RESEARCH.md
Part 5): the 12 sources "ag_news, arc, banking77, boolq, clinc150, commonsense_qa, mmlu, mnli, prompt_injections, snli,
sst5, wanli" plus typed-decisions, converted and schema-augmented by `unsloth.models.decision_datasets`
(`build_decision_mixture`, equal weight per source: the guide gives no weights and no row count), and the test split
"3,000 rows: 2000 from typed-decisions, 500 from BANKING77 and 500 from CLINC150" (the typed-decisions `test` split is
400 cases = 2,000 decisions; 500 BANKING77 `test` rows and 500 CLINC150 `plus` `test` rows sampled with the seed).

NEVER TRAIN ON A TEST ROW: the train mix is built from `train` splits only (MMLU: `auxiliary_train`), then every row
whose state shares a 13-word shingle (or, for a shorter state, the whole text) with any test state, with any state in
the extra decontamination files (--also-decontaminate: our JevBench public items), is dropped (`Decontaminator`, the
guide's "decontaminated against the training data"). CONTAMINATION NOTE for every later report: this mix trains on the
BANKING77 / CLINC150 / MMLU / ARC train splits, so a Decision Index number for those benchmarks from the result is
in-distribution.

Outputs (outside git): OUT/train.jsonl, OUT/test.jsonl, OUT/manifest.json (dataset revisions, counts, hashes).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
os.environ.setdefault("HF_HOME", "C:/Users/jwals/textgen/user_data/models/hf-unsloth")
os.environ.setdefault("PYTHONUTF8", "1")

GUIDE_SOURCES = ["ag_news", "arc", "banking77", "boolq", "clinc150", "commonsense_qa", "mmlu", "mnli",
                 "prompt_injections", "snli", "sst5", "wanli", "typed_decisions"]
SEED = 3407


def sha(rows) -> str:
    h = hashlib.sha256()
    for r in rows:
        h.update(json.dumps(r, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    return h.hexdigest()


def strip(row: dict) -> dict:
    return {k: v for k, v in row.items() if k != "_variants"}


def state_texts(rows):
    for r in rows:
        s = r["state"]
        yield s if isinstance(s, str) else json.dumps(s, ensure_ascii=False, sort_keys=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=12000, help="rows in the train mix (the guide does not state it)")
    ap.add_argument("--out", default=str(REPO / "index" / "decider_train"))
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--also-decontaminate", action="append", default=[],
                    help="a JSONL/JSON file whose strings are extra evaluation texts (our JevBench public items)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from unsloth.models import decision_datasets as dd

    rng = random.Random(a.seed)
    # ---- TEST: typed-decisions test (all 400 cases) + 500 banking77 test + 500 clinc150 (plus) test
    typed = [strip(r) for r in dd.load_source("typed_decisions", "test")]
    bank = dd.load_source("banking77", "test")
    clinc = dd.load_source("clinc150", "test")
    bank = [strip(r) for r in rng.sample(bank, 500)]
    clinc = [strip(r) for r in rng.sample(clinc, 500)]
    test = []
    for name, rows in (("typed_decisions", typed), ("banking77", bank), ("clinc150", clinc)):
        for r in rows:
            r["source"] = name
            test.append(r)
    n_dec = {n: sum(len(r["questions"]) for r in test if r["source"] == n) for n in ("typed_decisions", "banking77", "clinc150")}
    print("test rows", len(test), "decisions", n_dec, sum(n_dec.values()))

    # ---- extra evaluation texts to keep out of training
    extra = []
    for path in a.also_decontaminate:
        text = Path(path).read_text(encoding="utf-8")
        try:
            data = json.loads(text)
            items = data if isinstance(data, list) else [data]
        except ValueError:
            items = [json.loads(l) for l in text.splitlines() if l.strip()]
        for it in items:
            extra.append(it if isinstance(it, str) else json.dumps(it.get("state", it), ensure_ascii=False, sort_keys=True))
    eval_texts = list(state_texts(test)) + extra
    print("decontaminating against", len(eval_texts), "texts (", len(extra), "extra )")

    # ---- TRAIN: the guide's mix, equal weights, augmented, decontaminated
    # Over-draw before decontamination so the mix still reaches n_train.
    draw = int(a.n_train * 1.15) + 64
    mix = dd.build_decision_mixture(GUIDE_SOURCES, draw, seed=a.seed, split="train", augment=True,
                                    decontaminate_against=eval_texts)
    train = mix[:a.n_train]
    by_src = {}
    for r in train:
        by_src[r.get("source")] = by_src.get(r.get("source"), 0) + 1
    print("train rows", len(train), by_src)
    assert len(train) == a.n_train, f"decontamination left {len(train)} of {a.n_train}"

    # ---- hard check: no test state appears in train (exact text)
    test_states = set(state_texts(test))
    clash = [r for r in train if next(iter(state_texts([r]))) in test_states]
    assert not clash, f"{len(clash)} train rows equal a test state"

    with open(out / "train.jsonl", "w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "test.jsonl", "w", encoding="utf-8") as f:
        for r in test:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    import unsloth
    manifest = {
        "built": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "unsloth": getattr(unsloth, "__version__", None),
        "seed": a.seed, "n_train": len(train), "train_by_source": by_src,
        "weights": "equal per source (the guide gives none)", "augment": True,
        "test": {"rows": len(test), "decisions": n_dec, "sha256": sha(test)},
        "train_sha256": sha(train),
        "decontaminated_against": {"test_states": len(test_states), "extra_texts": len(extra), "shingle_words": 13},
        "sources": {n: {"hf_id": dd.SOURCES[n].hf_id, "config": dd.SOURCES[n].config,
                        "train_split": dd.SOURCES[n].train_split, "test_split": dd.SOURCES[n].test_split,
                        "license": dd.SOURCES[n].license} for n in GUIDE_SOURCES},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
