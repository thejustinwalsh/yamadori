#!/usr/bin/env python
"""TRAIN A DECISION MODEL THE WAY UNSLOTH'S GUIDE DOES, AND SCORE IT ON THE GUIDE'S TEST SPLIT.

    # data first: prepare_data.py
    .venv-unsloth/Scripts/python.exe bench/decider/unsloth/train_decision.py --model unsloth/Qwen3.5-2B \
        --epochs 1 --out index/decider_train/runs/q2b-e1

The recipe is Unsloth Studio's own default for "a text LLM trained as a decision model" (fresh Clef head + LoRA; read
from unsloth 2026.10.2 `studio/backend/assets/configs/model_defaults/decision/llm_decision_defaults.yaml` and
`decision_trainer.py`): LoRA r=64 / alpha=64 / dropout 0, 1 epoch, lr 1e-4, per-device batch 8 x accumulation 2, warmup 5,
weight decay 0.01, adamw_8bit, linear schedule, seed 3407, the new head at 3e-4, QLoRA (4-bit base), gradient checkpointing
"unsloth", max_seq_length 4096 in Studio / 2048 in the guide's code. This is the guide's "LoRA (r=64) for just one epoch" table.
The guide's text also says "set epochs to 2, LoRA rank to 16 and learning rate to 2e-4" to match typed-decisions; --epochs 2
--lora-r 16 --lr 2e-4 --head-lr 1e-4 is that variant. Every departure (batch size for a smaller card, max_seq_length) is
recorded in the result file.

Calibration rows are Unsloth's `split_holdout` of the TRAIN items (at most 400 decisions), never the test rows. The test
split (prepare_data.py: 2,000 typed-decisions + 500 BANKING77 + 500 CLINC150 decisions) is scored before training (the
guide's "before" column) and after.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "C:/Users/jwals/textgen/user_data/models/hf-unsloth")
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "0")


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Qwen3.5-2B")
    ap.add_argument("--data", default="C:/Users/jwals/llama-stack/index/decider_train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=None)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head-lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--optim", default="adamw_8bit")
    ap.add_argument("--scheduler", default="linear")
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--n-train", type=int, default=0, help="use only the first N train rows")
    ap.add_argument("--eval-batch", type=int, default=4)
    ap.add_argument("--test-rows", type=int, default=0, help="smoke runs: only the first N test rows per source")
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--mem-gb", type=float, default=0, help="cap this process's CUDA memory (leave room for resident servers)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import torch
    from unsloth import DecisionTrainer, FastDecisionModel, is_bfloat16_supported
    from transformers import TrainingArguments

    if a.mem_gb and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(min(1.0, a.mem_gb * 1024**3 / torch.cuda.get_device_properties(0).total_memory))
    t0 = time.time()
    res = {"args": vars(a), "device": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
    model, tok = FastDecisionModel.from_pretrained(
        a.model, decision_head="clef", max_seq_length=a.max_seq, load_in_4bit=not a.no_4bit,
        use_gradient_checkpointing="unsloth", random_state=a.seed)
    model = FastDecisionModel.get_peft_model(
        model, r=a.lora_r, lora_alpha=a.lora_alpha or a.lora_r, lora_dropout=0.0,
        use_gradient_checkpointing="unsloth", random_state=a.seed)
    res["trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)

    train_rows = read_jsonl(Path(a.data) / "train.jsonl")
    if a.n_train:
        train_rows = train_rows[:a.n_train]
    test_rows = read_jsonl(Path(a.data) / "test.jsonl")
    items, rep = FastDecisionModel.build_dataset(train_rows, tok, model)
    res["train_build"] = rep
    train_items, cal_items = FastDecisionModel.split_holdout(items, a.seed)
    sources = ("typed_decisions", "banking77", "clinc150")
    test_items = {}
    for s in sources:
        rows_s = [r for r in test_rows if r["source"] == s]
        it, r = FastDecisionModel.build_dataset(rows_s[:a.test_rows] if a.test_rows else rows_s, tok, model)
        test_items[s] = it
        res.setdefault("test_build", {})[s] = r
    dec = lambda its: sum(len(i["labels"]) for i in its)
    res["counts"] = {"train_items": len(train_items), "train_decisions": dec(train_items),
                     "calibration_decisions": dec(cal_items), "test_decisions": {s: dec(v) for s, v in test_items.items()}}
    print(json.dumps(res["counts"]), flush=True)

    def score(tag):
        r = {}
        tot_c = tot_n = 0
        for s in sources:
            m = FastDecisionModel.evaluate(model, tok, test_items[s], batch_size=a.eval_batch)
            r[s] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items() if not isinstance(v, (list, dict))}
            n = dec(test_items[s])
            tot_n += n
            tot_c += m["accuracy"] * n
        r["pooled_accuracy"] = round(tot_c / tot_n, 4)
        r["pooled_decisions"] = tot_n
        print(tag, json.dumps(r), flush=True)
        return r

    if not a.skip_baseline:
        res["test_before"] = score("before")
    batches = -(-len(train_items) // a.batch)
    steps_per_epoch = max(1, -(-batches // a.accum))
    targs = dict(
        output_dir=str(out / "trainer"), per_device_train_batch_size=a.batch, per_device_eval_batch_size=a.batch,
        gradient_accumulation_steps=a.accum, learning_rate=a.lr, weight_decay=0.01, lr_scheduler_type=a.scheduler,
        optim=a.optim, seed=a.seed, bf16=is_bfloat16_supported(), fp16=torch.cuda.is_available() and not is_bfloat16_supported(),
        logging_steps=1, report_to="none", warmup_steps=a.warmup, save_strategy="no", eval_strategy="no",
        disable_tqdm=True)
    if a.max_steps > 0:
        targs["max_steps"] = a.max_steps
    else:
        targs["num_train_epochs"] = a.epochs
    trainer = DecisionTrainer(model=model, args=TrainingArguments(**targs), train_dataset=train_items,
                              processing_class=tok, head_learning_rate=a.head_lr)
    t1 = time.time()
    trainer.train()
    res["train_seconds"] = round(time.time() - t1, 1)
    res["steps"] = trainer.state.global_step
    res["steps_per_epoch"] = steps_per_epoch
    res["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None
    res["log_history"] = [h for h in trainer.state.log_history if "loss" in h][::max(1, res["steps"] // 60)]
    (out / "result.json").write_text(json.dumps(res, indent=1), encoding="utf-8")      # partial: training is never lost
    cal = FastDecisionModel.calibrate(model, tok, cal_items, batch_size=a.eval_batch)
    res["calibration_heldout"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in cal.items() if not isinstance(v, (list, dict))}
    res["temperature"] = model.decision_config.get("temperature")
    if not a.no_save:      # SAVE BEFORE THE TEST SPLIT: the 2026-10-07 run lost 47 minutes of training to an OOM in the final eval
        model.save_pretrained(str(out / "adapters"))
        model.save_pretrained_merged(str(out / "merged"), tok)
        print("saved", out, flush=True)
    (out / "result.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    FastDecisionModel.for_inference(model)
    res["test_after"] = score("after")
    res["wall_seconds"] = round(time.time() - t0, 1)
    (out / "result.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("DONE", out / "result.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
