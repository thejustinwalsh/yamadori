# Run 1: Qwen3.5-0.8B, the guide's recipe, 1 epoch (2026-10-07) -- trained, NOT saved, test-split "after" NOT scored

What ran: `train_decision.py --model unsloth/Qwen3.5-0.8B --epochs 1 --batch 4 --accum 4 --max-seq 4096 --mem-gb 3.4 --eval-batch 2`
(`UNSLOTH_COMPILE_DISABLE=1`: with Unsloth's compile path on, a batch of 4 at length 1,187 died in an inductor stride assertion).
Data: `prepare_data.py --n-train 12000` (the 12 sources + typed-decisions, equal weight, augmented, decontaminated; data_manifest.json);
11,756 train items (19,841 decisions) after the package's own 400-decision calibration hold-out; test = 3,000 decisions.
Recipe: Studio's LLM-decision defaults (LoRA r 64 / alpha 64, lr 1e-4, head lr 3e-4, effective batch 16, linear, adamw_8bit, 4-bit
base); 70.6M trainable parameters of 731.7M.

GPU: the A4000 (RTX A4000, 16 GB), beside the LIVE bonsai-a4000 and the embedder, process capped at 3.4 GB of CUDA allocator memory.
This was a mistake: it ran during the coordinator's deploy check and contended for the card (see the coordinator's account). The
process was suspended (NtSuspendProcess) from about 23:25 to 23:40 once that was noticed and resumed when the deploy check ended.

## Numbers

Before training (random head, the untrained model; the guide's "before" column in brackets):

| split | decisions | accuracy | guide before |
|---|---|---|---|
| typed-decisions | 2,000 | 34.2% | 36% |
| BANKING77 | 500 | 8.2% | 7% |
| CLINC150 | 500 | 17.6% | 19% |
| pooled | 3,000 | 27.1% | |

Training: 735 steps, 2,841 s wall including about 15 minutes suspended; unsuspended pace was 24% of the epoch in about 9.5 minutes
(about 40 minutes for the epoch, the guide's 42). Mean training loss by fraction of the epoch: 0.99 (first 10%), 0.74, 0.64, 0.57, 0.49.
The earlier 14-step speed probe peaked at 2.46 GB allocated at batch 4 (the guide: 4 GB).

After training: NOT SCORED. The final test-split evaluation ran out of memory at the 3.4 GB cap (3.02 GB allocated, evaluating the
3,000-token CLINC150 rows) and the old script saved the adapters only after that evaluation, so the trained weights were lost.
Fixed in 4a54008 (save, then score). Nothing from this run counts as a result beyond the baseline above and the loss curve.

The rerun needs a card window: nothing else serving on it, no memory cap, and the lanes paused.
