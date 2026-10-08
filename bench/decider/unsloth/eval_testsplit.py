#!/usr/bin/env python
"""Score a trained decision checkpoint on the guide's test split (prepare_data.py's test.jsonl: 2,000 typed-decisions,
500 BANKING77, 500 CLINC150 decisions), in the unsloth venv.

    .venv-unsloth/Scripts/python.exe bench/decider/unsloth/eval_testsplit.py --ckpt RUN/merged [--no-4bit] --out RUN/test_16bit.json

--no-4bit loads the merged checkpoint in 16-bit; the default loads it bnb NF4 (Unsloth's own 4-bit runtime: this is NOT
llama.cpp's Q4_K_M, whose accuracy needs the Clef GGUF path). Records accuracy, ECE, loss per source, latency per record
and peak VRAM.
"""
import argparse, json, os, sys, time
os.environ.setdefault("HF_HOME", "C:/Users/jwals/textgen/user_data/models/hf-unsloth")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="C:/Users/jwals/llama-stack/index/decider_train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--batch", type=int, default=4)
    a = ap.parse_args()
    import torch
    from unsloth import FastDecisionModel
    model, tok = FastDecisionModel.from_pretrained(a.ckpt, max_seq_length=4096, load_in_4bit=not a.no_4bit)
    FastDecisionModel.for_inference(model)
    rows = [json.loads(l) for l in open(os.path.join(a.data, "test.jsonl"), encoding="utf-8") if l.strip()]
    res = {"ckpt": a.ckpt, "bits": 16 if a.no_4bit else "bnb-nf4", "weights_vram_gb": round(torch.cuda.memory_allocated() / 1e9, 3)}
    tot_c = tot_n = 0
    for s in ("typed_decisions", "banking77", "clinc150"):
        its, rep = FastDecisionModel.build_dataset([r for r in rows if r["source"] == s], tok, model)
        torch.cuda.synchronize(); t0 = time.time()
        m = FastDecisionModel.evaluate(model, tok, its, batch_size=a.batch)
        torch.cuda.synchronize(); dt = time.time() - t0
        n = sum(len(i["labels"]) for i in its)
        res[s] = {**{k: round(v, 4) for k, v in m.items() if isinstance(v, float)}, "decisions": n, "records": len(its),
                  "ms_per_record_batched": round(1000 * dt / len(its), 1), "skipped": rep["skipped"]}
        tot_c += m["accuracy"] * n; tot_n += n
    res["pooled_accuracy"] = round(tot_c / tot_n, 4); res["pooled_decisions"] = tot_n
    res["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    open(a.out, "w", encoding="utf-8").write(json.dumps(res, indent=1))
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
