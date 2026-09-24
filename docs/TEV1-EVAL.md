# Tev1-4B-experimental as Yamadori's decision classifier

Written 2026-09-24. The question: can Tev1 replace Laya? The evaluation
design is `docs/CLM-EVAL.md` §5, which says a candidate must beat both Laya
and a control that adds no new system.

## Verdict: not yet

**Tev1 zero-shot does not earn a slot.** It clearly beats Laya's head on the
held-out routing set, so criterion 1 passes. It does not beat the control,
so criterion 2 fails:

| arm (held-out `route_in`, investigate vs not, n=120) | correct |
|---|---|
| Laya head | 80 |
| E1: logistic head on the Qwen3-Embedding-0.6B vectors we already serve | 104 |
| Tev1, Q4_K_M, pre-registered prompt | 104 |
| Tev1, Q8_0, pre-registered prompt | 105 |

- **Tev1 against Laya.** Exact McNemar, 34 vs 10 discordant, p=0.0004
  (Q4_K_M). 33 vs 8, p=0.0001 (Q8_0).
- **Tev1 against E1.** 13 vs 13, p=1 (Q4_K_M). 14 vs 13, p=1 (Q8_0).
  This result is "cannot tell". It does not show the two are equal.
- **Tev1 depends on option order.** Over the six orders of the same three
  options it scores 79 to 107 (Q4_K_M). Its worst order is Laya's level.

**What the data points to is the CLM-EVAL "fewer systems" outcome: replace
Laya with E1.** That head runs on the embedder retrieval already keeps loaded.

- E1 against Laya: 104 vs 80, 31 vs 7 discordant, p=0.0001.
- The result repeats over 8 CV seeds: 99–106/120, and every seed gives
  p < 0.01 against Laya.
- In-set CV accuracy is 0.795–0.823. Laya's is 0.728.
- Combined with `selection.decide` in place of Laya, E1 scores 99/120 against
  92/120, p=0.039.

E1 has not met the rest of the CLM-EVAL bar. That needs a second held-out set,
the live service reproducing the offline result, and an operator decision. So
the finding is "E1 beats Laya, repeatably, on this set". It is not yet "ship
E1".

**Tev1 could still earn a slot after fine-tuning (§4), or on jobs E1 cannot
do.** E1 is a head trained per task. Tev1 answers any closed question with
no training, which a new Phase 0.6 decision would need on day one. But the
labels for those decisions do not exist yet (§3c), so that advantage is
unmeasured.

**On the router, Tev1 is a clear no.** It scored 781/1,024 against the
router's 981, with 43 misroutes into the code pipelines against 0.

**Weights licence.** The model card says: *"The release license for these
fine-tuned weights is being finalized before public conversion."* The base
Qwen3.5-4B is Apache-2.0, and the tev1 code is MIT. Everything here is
**evaluation only**.

---

## 0. What was run, and where

- **Hardware.** Everything ran on the A4000
  (`GPU-43e37d0c-4104-9056-2552-6109d4d3382c`), in our own llama-server on
  `127.0.0.1:10091` with `CUDA_VISIBLE_DEVICES` set to that UUID.
- **Things this eval did not touch.** The 5060 Ti, llama-swap, the proxy,
  Laya, and anything in `config.yaml`.
- **Build.** The official PrismML fork, `config.yaml` macro `server`,
  `prism-b10709-9a9394a`. It loads and serves `qwen35`. The sudoingX build
  was not needed. MTP is its only reason to exist, and the GGUF was
  converted without MTP (§1).
- **Room on the card.**
  - The harness (`bench/tev1/eval_tev1.py`) reads nvidia-smi by UUID before
    every 25 requests. It waits while free memory is under 1,331 MiB or
    while an `sd-server` is on the card. It never unloads anything.
  - Free memory never fell below 2,707 MiB with Tev1 loaded (Q4_K_M, while
    another agent's image generation ran). With Q8_0 loaded and idle, 5
    reads gave 6,574 MiB free. Memory was not sampled during the Q8_0
    held-out run, and the harness recorded no waits in either run.
  - The Q8_0 server was **stopped when the coordinator asked for the card
    back** (an image-fix agent was blocked). The Q8_0 latency and peak-VRAM
    measurements are therefore **not run**.
- **The E1 control** made 409 short query embeddings through the resident
  `embeddings` model on :11434. The calls went through `code_search.embed`,
  under a `gpu_room` lease, and loaded nothing.
- **Files.**
  - Code: `bench/tev1/adapters.py` (every prompt adapter, one module) and
    `bench/tev1/eval_tev1.py` (client, arms, statistics).
  - Results: `bench/tev1/results/{Q4_K_M,Q8_0}/*.json`,
    `bench/tev1/results/control_e1*.json` and
    `bench/tev1/results/report.txt`.
  - Staged entry: `config.tev1.yaml.snippet`.
  - Model files: `C:\Users\jwals\textgen\user_data\models\tev1-4b\`, holding
    `hf/`, `tev1-repo/`, the GGUFs, and the conversion and quantisation logs.
    Server logs are in `runs/`.

## 1. Download, verification, conversion

**Source.** HF `togethercomputer/Tev1-4B-experimental`, revision
`0b7becf017daa0e5eb222f8ce7483c8c8259c52f` (last modified 2026-09-23
23:09 UTC). There were 13 files, 9,342,811,993 bytes in total, and the
download took 52 s. The code came from a shallow clone of
`github.com/togethercomputer/tev1`.

**Verification against the HF API** (`model_info(files_metadata=True)`):

- The two safetensors shards and `tokenizer.json` match their LFS SHA-256:
  - shard 1: `f0d45353…2514`, 5,329,398,688 B
  - shard 2: `f7a11c87…941b`, 3,990,429,408 B
- The other 10 files match their git blob ids (`git hash-object`).
- **All 13 match.**

**Conversion.** `convert_hf_to_gguf.py` from the PrismML repo:
`--outtype bf16 --no-mtp`.

- **Text model only.** The config's architecture is
  `Qwen3_5ForConditionalGeneration`. Without `--mmproj`, the converter's
  `Qwen3_5TextModel` writes the language model only, so the vision tower (24
  layers) and the preprocessors are not converted.
- **No MTP layer.** `--no-mtp` drops the 1-layer MTP head. A one-letter
  answer gains nothing from drafting.
- **Output.** 426 tensors, `blk.0`–`blk.31`, no `blk.32` and no `visual.*`.
- **Chat template.** Embedded from `chat_template.jinja`.
- **Warnings.** Exactly one: `WARNING:hf-to-gguf:Unknown RoPE type:
  default`. The converter still writes the mrope sections, which
  `_Qwen35MRopeMixin` defaults to `[11,11,10]`. The config's
  `mrope_section` is also `[11,11,10]`.

**Quantisation.** `llama-quantize` from the same build. There were no
warnings. In Q4_K_M, `token_embd` becomes q6_K.

| file | bytes | llama-quantize size |
|---|---|---|
| BF16 | 8,424,393,280 | 8,023.67 MiB (16.00 BPW) |
| Q8_0 | 4,482,402,880 | 4,264.29 MiB (8.51 BPW) |
| Q4_K_M | 2,708,804,160 | 2,572.86 MiB (5.13 BPW) |

**Chat-template sanity check** (`results/*/sanity.json`, both quants):

- `/apply-template` with `enable_thinking: false` renders
  `<|im_start|>assistant\n<think>\n\n</think>\n\n`, which is the model's
  non-thinking form.
- The recommended request gave **exactly one letter** on all 4 examples in
  the tev1 repo, with `completion_tokens` = 2 (the letter and EOS) and
  `finish` = stop. The model card's own example also gives one letter,
  although its JSON is malformed (the closing `}` is missing).
- **`enable_thinking: false` is load-bearing.** Without it the template
  opens a think block, the 8 tokens end inside it, and `content` is `""`
  with `finish` = length.

## 2. Cost: VRAM and latency

**VRAM**, measured with nvidia-smi on the A4000.

- **Baseline.** 4,704 MiB with retrieval embeddings and Laya resident. It
  was stable over repeated reads through the afternoon.
- **Q4_K_M.** A dedicated baseline → load → idle sequence
  (`runs/vram-Q4_K_M.txt`): idle **+3,197 MiB** (n=5 reads). The peak
  during 20 decisions with 2,000-token states was **+3,219 MiB** (0.25 s
  sampling, `results/Q4_K_M/latency_run1.json`).
- **Q8_0.** The in-script baseline was polluted: another agent's sd-server
  started and stopped during it. The quiet 4,704 MiB baseline is used
  instead, giving idle **+4,889 MiB** (n=5 reads). Q8_0 peak: **not
  measured**.

llama-server's own breakdown, `-c 4096 -np 1 -ub 2048` (MiB):

| component | Q4_K_M | Q8_0 |
|---|---|---|
| model | 2,572.86 | 4,264.29 |
| KV (8 full-attention layers, f16) | 128.00 | 128.00 |
| recurrent state (24 DeltaNet layers, f32) | 50.25 | 50.25 |
| compute buffer | 272.06 | 272.06 |
| **sum** | **3,023** | **4,714** |

The CUDA context accounts for the rest. `token_embd` (497 MiB in Q4_K_M,
644 MiB in Q8_0) stays in host RAM.

**Fit with image generation.** Image generation's measured peak is
+6,389 MiB. The four cases:

- **Q4_K_M:** 11,463 − 3,219 − 6,389 = **1,855 MiB free**, which clears the
  1,331 MiB headroom. Q4_K_M **coexists** with image generation.
- **Q8_0:** leaves about 185 MiB, so it **does not coexist**.
- **Tev1 replacing Laya:** Laya's share comes back as well. That share has
  never been measured separately: it is somewhere between "~1 GiB" and
  ≥2,445 MiB (gpu_room `FIXED`).
- **Another agent's direct sd-server** (the VAE-variant runs): the same
  arithmetic applies, but `gpu_room` never sees it.

**Latency.** Wall-clock HTTP round trip per decision, over n=20 distinct
states per size. The states are seeded random word samples, checked
distinct: the prompt cache reused only the 37-token system prefix. One
un-counted warm-up request came first.

| state tokens (prompt tokens) | Q4_K_M, quiet card: p50 / p95 ms | Q4_K_M, image generation running: p50 / p95 ms | Q8_0 |
|---|---|---|---|
| 200 (435–464) | **313 / 328** | 634 / 656 | not run |
| 600 (834–847) | **463 / 690** | 859 / 945 | not run |
| 2,000 (2,237–2,249) | **898 / 919** | 2,049 / 2,268 | not run |

- **Short prompts.** A held-out `route_in` decision is about 250 prompt
  tokens. Its round trip was p50 289 / p95 450 ms on Q4_K_M and p50 273 /
  p95 305 ms on Q8_0 (n=120 each, `laya` arm).
- **Against Laya.** Laya's round trip is 66 ms median and 86 ms p90
  (`live_route_check.txt`), so **Tev1 is 5–14× slower**.
- **Contention.** When image generation shares the card, Tev1's latency
  roughly doubles. The effect runs both ways (AGENTS.md "one GPU consumer at
  a time").
- **A harness bug, found and fixed** (PROTOCOL rule 3). The first latency
  run built every state from the same file offset, so all 20 states were
  identical and "2,000 tokens" came back in 75 ms from the cache. That run
  is discarded, and the fix is commented in `cmd_latency`.

## 3. Zero-shot evaluation on our data

**Prompt adapters.** All of them are in `bench/tev1/adapters.py`.

- The state is always the first JSON key. It is **structured**: labelled
  fields, one per line, with long text capped and cut from the end with a
  visible marker. It is never raw chat history.
- The request is the card's interface exactly: its system prompt, the tev1
  `json.dumps` rendering, temperature 0, `max_tokens` 8 and thinking off.
- **Scoring.** The raw reply must be exactly one listed letter, otherwise
  the row is wrong and is counted as invalid. The best listed letter from
  the top-20 logprobs, which is what the card's regex constraint would
  pick, is recorded beside it as `constrained`. On `route_in` the two were
  identical everywhere.

### 3a. Held-out `route_in`: investigate vs not (the Laya job)

**Data.** `bench/laya_routing_heldout_packages.jsonl`: 120 rows, 45 labelled
investigate, a hard slice of 28. No training file matches it. All arms run
on the same rows and are compared with exact two-sided McNemar.

**Power, stated first** (CLM-EVAL §5). With 20–40 discordant pairs, p <
0.05 needs a net gap of 10–14 rows. A smaller gap reads as "cannot tell".

**Pre-registered primary arm.** Written before any output was seen:

- the question and option descriptions Laya's head uses
  (`mcp/laya_head.TASKS`)
- in Laya's option order
- state = `question:` / `context:` (Laya's `render_state` fields)

**Exploratory arms:**

- **`held`**: a wording written for this eval, also before any output. It
  says "library versions it holds" rather than "this repository", because
  the held-out set is about packages.
- **All six option orders** for both wordings.

| arm | investigate-vs-not | 3-way | hard (28) | missed / unneeded |
|---|---|---|---|---|
| Laya head (argmax) | 80 [0.575–0.750] | 66 | 11 | – |
| bare regex | 73 | 61 | 18 | – |
| `selection.decide`, Laya absent, **today** | 91 | – | 19 | – |
| `selection.decide`, recorded 2026-09-22 | 89 | – | 18 | 18 / 13 |
| **E1** embeddings head (control) | **104** [0.793–0.922] | 100 | 22 | 10 / 6 |
| **Tev1 Q4_K_M, primary** | **104** [0.793–0.922] | 104 | 21 | 3 / 13 |
| **Tev1 Q8_0, primary** | **105** [0.802–0.928] | 105 | 21 | 3 / 12 |
| Tev1 Q4_K_M, `held` (exploratory) | 108 | 106 | 18 | 0 / 12 |
| Tev1 Q8_0, `held` (exploratory) | 110 | 106 | 20 | 0 / 10 |

The bracketed ranges are 95% exact CIs. Today's `selection.decide` scores 91
where 89 was recorded on 2026-09-22 (6 rows moved). The package store and
`domains.derived_domains` have changed since. Every combined arm below uses
today's code, so all of them are paired.

**Paired, exact McNemar** (the reference arm's own wins against Tev1's own
wins):

| comparison | Q4_K_M | Q8_0 |
|---|---|---|
| Laya head vs Tev1 primary | 10 vs 34, **p=0.0004** | 8 vs 33, **p=0.0001** |
| regex vs Tev1 primary | 13 vs 44, p=5e-5 | 12 vs 44, p=2e-5 |
| selection vs Tev1 primary | 12 vs 25, p=0.047 | 12 vs 26, p=0.034 |
| **E1 vs Tev1 primary** | **13 vs 13, p=1** | **13 vs 14, p=1** |
| E1 vs Tev1 `held` | 10 vs 14, p=0.54 | 8 vs 14, p=0.29 |
| Laya head vs E1 | 7 vs 31, **p=0.0001** | (same) |

**As the second signal in `selection.decide`.** This is the live
combination: disagreement escalates. Laya's abstain gate is rebuilt from the
cached features, and the rebuilt choices equal the logged ones on 120/120.

| combination | correct | missed | unneeded |
|---|---|---|---|
| selection + Laya | 92 | 10 | 18 |
| selection + E1 | 99 (vs +Laya: 1 vs 8, **p=0.039**) | 8 | 13 |
| selection + Tev1 Q4_K_M primary | 96 (vs +Laya: 4 vs 8, p=0.39) | 8 | 16 |
| selection + Tev1 Q8_0 primary | 96 (vs +Laya: 3 vs 7, p=0.34) | 8 | 16 |
| selection + E1 vs selection + Tev1 Q4_K_M | 4 vs 1, p=0.375 | | |

Every head scores better **alone** than combined: E1 104 against 99, and
Tev1 104 against 96. The rule "disagreement escalates" costs accuracy on
this set. That is a finding about `selection.decide`, not about either model.

**Option order: the headline depends on it.** Tev1 reads options as letters,
and six orders of the same three options give:

| wording, quant | per order (Laya's order first) | min / mean / max | all 6 orders agree | majority of 6 |
|---|---|---|---|---|
| laya, Q4_K_M | 104, 102, 107, **79**, 94, 90 | 79 / 96.0 / 107 | 83/120 | 99 |
| laya, Q8_0 | 105, 100, 105, **79**, 92, 83 | 79 / 94.0 / 105 | 82/120 | 98 |
| held, Q4_K_M | 108, 104, 113, **82**, 102, 86 | 82 / 99.2 / 113 | 84/120 | 105 |
| held, Q8_0 | 110, 108, 115, **83**, 100, 84 | 83 / 100.0 / 115 | 82/120 | 105 |

- **Worst order.** answer_directly, clarify, investigate (investigate last)
  scores 79. Against Laya that is p=1 (30 vs 29).
- **Reversed order.** Tev1 over-predicts investigate: 73–82 of 120.
- **Majority vote over six orders.** 99/120 beats Laya (14 vs 33, p=0.008)
  and ties E1 (18 vs 13, p=0.47), at 6× the latency.
- **The canonical order was a fixed choice, not a selected one.** Still,
  mean-over-orders is the honest single number: **96.0 (Q4_K_M) and 94.0
  (Q8_0)**.
- **Do not cite the best cell** (115, held order 2). It is the maximum of 12
  cells chosen on the test set.

**Quant fidelity.** Q4_K_M and Q8_0 gave the same key on 111–119 of 120 rows
per arm, and 115/120 on the primary arm. The accuracy differences between
the quants are within noise.

### 3b. Route classes: `bench/route/labels.jsonl` (1,024 labelled rows)

**The data is in-sample for the router's rules, not for Tev1.**

- The state carries the facts the router is given: ends on a client tool
  result or a user message, first turn, the client's own tools, the system
  prompt head (1,200 characters) and the latest user message (2,000
  characters).
- Arm `facts` also adds the index fact the router computes: the one
  `route.classify` reports as `signals.readable`. It is only computed when
  the router's question detector fires, so it partly carries the router's
  own signal.
- Q4_K_M only: CLM-EVAL §3 said not to test this job, and the coordinator
  asked for Q8_0 to be loaded for the paired comparison only.

| | correct | invalid | misroutes into code (bar: 0) | McNemar vs router |
|---|---|---|---|---|
| router (`mcp/route.py`) | 981/1,024 (0.958) | – | 0 | – |
| Tev1 `text` | 722 (0.705) | 4 | **105** | 271 vs 12, p=6e-65 |
| Tev1 `facts` | 781 (0.763) | 8 | **43** | 212 vs 12, p=2e-48 |

**Recall and precision per class** (`facts` arm):

| class | recall | precision |
|---|---|---|
| utility | 4/43 | 4/4 |
| agent_step | 338/377 | 338/371 |
| code_edit | 11/160 | 11/39 |
| code_generation | 223/225 | 223/390 |
| library_question | 57/60 | 57/57 |
| prose | 148/159 | 148/155 |

Two failures stand out:

- **It collapses code_edit into code_generation.** Recall on code_edit is
  11/160.
- **It almost never recognises a client side call.** Recall on utility is
  4/43.

**Prompt injection, observed.** The invalid rows are benchmark prompts with
formatting instructions: IFEval-style "separate with `******`", "SECTION 1",
"P.P.S". Tev1 **followed the instructions inside the state**, for example
replying `F\n******\nF` or `SECTION 1\nDebra Simons`, despite the system
line telling it to treat the state as data. Constrained decoding would force
a letter, but the model's attention had already been taken.

**Verdict for the router: no.** The router decides on facts it parses
(PROTOCOL rule 8), and Tev1 cannot approach it.

### 3c. The four Phase 0.6 decisions

**No labels exist for any of the four decisions.**
`SELF-IMPROVEMENT-PLAN.md` Phase 0.6 records each deep-thinking decision as
trigger ∈ {model, struggle, area, none} and derives labels from what
happened next (a missed escalation, a wasted one). That loop is not built,
no escalation has run on the agent path (LOG #19), and the Octopus V0 pilot
was stopped at 68 minutes with no outcome labels.

| decision | adapter | labelled evidence today | result |
|---|---|---|---|
| **area** (known-hard: library or version question) | `route_in` | the 120 held-out labels, the same job as 3a | Tev1 104–105/120 primary; E1 104 |
| **struggle** (escalate now?) | `struggle` | **none**. V0 has 16 tool calls, 1 tool error (the patch at event 56, recovered the next step), and 2 files written twice. Under the plan's own threshold of 3 signals, no V0 step qualifies: there is no positive example | smoke run only: "continue" on 16/16 steps. At the failed patch the margin narrowed from about 4 nats to 0.54 (B −0.46, A −1.00) |
| **model** (should `think_deeply` have been called?) | `model_trigger` | **none** | smoke run: "fine" on 15/16; "should_have" only at the failed patch, at 0.69 vs 0.70 nats (a coin flip) |
| **none** (outcome after a non-escalation: missed / fine) | `outcome` | **none** | smoke run at the failed patch: "unclear" (C −0.88, B −1.07) |

The smoke outputs are in `results/Q4_K_M/phase06.json`. **They are not
scored.** They show only that the adapters render real harness signals and
that the model's margins move in the plausible direction.

**What would create labels:**

1. The Phase 0.6 record: trigger, signals, hand-off, and the outcome over
   the following turns.
2. The V1–V3 Octopus runs, graded.
3. Hand labels on a sample of the struggle states: blind, rubric first, a
   second pass on 20%, as `bench/route` did.

## 4. Fine-tuning plan (not executed)

### Our label sources

| task | file(s) | rows | note |
|---|---|---|---|
| route_in (investigate / answer / clarify) | `bench/laya_routing_labels*.jsonl` (59 + 30 + 200) | 289 | train; the 120 held-out rows are **never** trained on |
| route classes (6-way) | `bench/route/labels.jsonl` (+ `second_pass.jsonl`, 203) | 1,024 | the rules already win (3b); train only if a head is wanted as a second opinion |
| grounded_excerpt / grounded | `bench/laya_grounded_excerpt_labels.jsonl`, `bench/laya_grounded_labels.jsonl` | 60 / 48 | grounded is a documented negative (LAYA.md) |
| distil (finding quality) | `bench/laya_distil_labels.jsonl` | 29 | small |
| guardrail (injection in retrieved text) | `bench/guardrail_labels.jsonl` | 156 | embeddings AUC 0.886 today |
| skill / hint selection | `bench/hint_buckets.jsonl` (19 buckets) + `hint_probes.jsonl` | 89 probes | the embedding arm is 64/89 today |
| Phase 0.6 struggle / outcome | – | **0** | see 3c |

**The total is about 1,700 labelled decisions**, and only about 290 of them
are for the job Tev1 would take over.

### The Tev1 recipe

From the repository: `docs/DATASET.md`, `docs/TRAINING.md` and
`runs/new-v1/README.md`.

The data is "new v1": 37,840 training rows and 4,568 dev rows,
16,929,529 training tokens, with the longest sequence 1,526 tokens.

| training source | rows |
|---|---|
| MultiNLI | 5,000 |
| BoolQ | 3,000 |
| Banking77 | 3,000 |
| AG News | 1,500 |
| SST-5 | 2,000 |
| original synthetic policies | 1,500 |
| additional synthetic policies | 12,000 |
| priority routing | 6,000 |
| synthetic research classification | 3,840 |

**Rendering.**

- Categorical options are **permuted per example**. The order spread in 3a
  suggests that was not enough.
- The prompt is rendered with the **Qwen3.5-2B** tokenizer's non-thinking
  template.
- The completion is the letter plus EOS, with completion-only loss.

**Proposed settings** (the repo says these are "not a verified export of
the historical job"):

| setting | value |
|---|---|
| LoRA | all-linear, r=8, α=16, dropout 0 |
| epochs / batch | 1 / 8 |
| learning rate | 5e-5, cosine, warmup 0.03 |
| sequence | 2,048, packing on |
| seed | 42 |

The run's own record says the uploaded-file hashes, job id and actual
hyperparameters "could not be retrieved" (HTTP 403).

### Option A: QLoRA on the A4000

**What exists** in the stack interpreter: `peft` 0.18.1, `bitsandbytes`
0.49.2, `accelerate` 1.13.0, `fla` 0.4.2 (the Gated-DeltaNet kernels) and
`triton` 3.5.1 (Windows). Missing: `causal_conv1d` (transformers falls back
to a torch conv) and `trl` (not needed: a plain Trainer with a
completion-only collator does it).

**Memory: an estimate, not measured.**

- NF4 weights: about 2.6 GB
- LoRA r=8 on all-linear, with Adam state: well under 0.5 GB
- activations at sequence ≤ 1,024 with gradient checkpointing, batch 8:
  about 3–5 GB
- **total: about 6–8 GB**

That fits beside retrieval and Laya (11.4 GB free), but **not beside image
generation**. It is a GPU consumer for its whole run. It needs a
maintenance window, queued through `bench/queue_runner.py`, with the A4000
yielded exactly as in this eval.

**Time: an estimate, not measured.**

- Data: our about 1,700 rows (about 0.6M tokens), plus a replay sample of
  about 4k rows of new-v1 (about 1.8M tokens) so that the general ability
  is not lost.
- Throughput: at a guessed 600–1,000 tokens/s for 4-bit QLoRA of a 4B on
  the A4000, that is **about 40–70 minutes per epoch**.
- `fla`'s Triton kernels on Windows are the first thing to prove (PROTOCOL
  rule 1). If they fail, the torch fallback is several times slower.

**Output.** A LoRA adapter, merged, then converted with
`convert_hf_to_gguf.py` and quantised as in §1.

**Evaluation after training.**

- The 120 held-out rows stay held out.
- This eval has now looked at them across many arms, so a **second held-out
  set (120 or more rows, labelled blind)** must be written and
  pre-registered before any fine-tuned model is scored (CLM-EVAL §5).
- The fine-tuned Tev1 must beat E1 with p < 0.05, on both sets.
- Option-order robustness should be a reported metric: the minimum over
  orders, not only the maximum.

### Option B: Together's fine-tuning service

This is the repo's own path: `examples/train_together.py --launch` uploads
`train.jsonl` and `dev.jsonl` and starts a billed job. The blog title says
the vendor's run cost "$17". The blog itself was not read, and the repo says
the actual price and runtime "have not been verified".

**It sends our prompts off the machine.** Those prompts include corpus
requests from real Hermes and SWE-agent sessions and our held package
questions. **That is the operator's decision, not ours.** Hosting and
inference are billed separately.

Whether the trained weights can be downloaded for local serving was **not
checked**. If they cannot, the result is a hosted classifier on the
per-request path, which is a network dependency the stack does not have
today.

### Recommendation

1. **E1 first.** It costs nothing new.
2. **Then a local QLoRA arm** of Tev1 against E1 on held-out set 2, in a
   maintenance window.
3. **Together** only if the operator accepts that the data leaves the
   machine.

## 5. Staged llama-swap entry and gpu_room row

`config.tev1.yaml.snippet` is at the repo root. **It is not in
`config.yaml`.** It holds:

- a `tev1` model on the A4000: Q4_K_M, `-c 4096 -np 1 -ub 2048`,
  `--cache-ram 0`, `ttl: 600`
- its own non-exclusive `decision` group. An ungrouped model would land in
  llama-swap's exclusive default group and evict retrieval.
- the proposed `gpu_room.SIZES` row: `Size(3219, 3197, True, …)` with the
  measurements above
- the weights-licence note

## Numbers not to cite

Each item below is labelled as it should be:

- **Tev1's best cells** (113–115/120). The maximum of 12 wording-by-order
  cells, chosen on the test set.
- **The `held` wording's lead over the primary.** Exploratory, and it was
  not repeated on a second set.
- **The QLoRA memory and time figures.** Arithmetic, not measurements.
- **Q8_0 latency and peak VRAM.** Not run: the card was yielded.
- **The Phase 0.6 smoke outputs.** Unlabelled.
- **E1's latency.** Not measured here. It is one query embedding against
  the resident embedder plus a 1,024-by-3 affine map.

## Sources

**Read 2026-09-24.**

- **The model.**
  - Card: <https://huggingface.co/togethercomputer/Tev1-4B-experimental>
    (`README.md` at the revision above; its licence section is quoted in the
    verdict)
  - File metadata: `HfApi().model_info(..., files_metadata=True)`
- **The Tev1 code.** <https://github.com/togethercomputer/tev1>: `README.md`,
  `docs/DATASET.md`, `docs/TRAINING.md`, `runs/new-v1/README.md`,
  `examples/decide.py`, `build_dataset.py` (`SYSTEM`, `messages`, option
  permutation) and `LICENSE` (MIT).
- **Ours.**
  - `AGENTS.md` ("Laya", "Claims carry their evidence", "one GPU consumer at
    a time")
  - `docs/PROTOCOL.md`: rules 1, 3, 4, 6, 10 and 11
  - `docs/CLM-EVAL.md` §5
  - `bench/eval_route_heldout.py`: the statistics, imported, not copied
  - `bench/laya_factcheck/heldout_rows.jsonl` and `live_route_check.txt`
  - `index/laya/route_in.json` and
    `index/laya_staging_20260922_191425/features_heldout.json`
  - `bench/route/eval_route.py` and `labels.jsonl`
  - `scripts/train_laya.py`: `_fit_linear` and `kfold`, reused for E1
  - `mcp/selection.py` (`decide`) and `mcp/gpu_room.py` (`SIZES`, `FIXED`)
  - `docs/SELF-IMPROVEMENT-PLAN.md` Phase 0.6 and
    `docs/SELF-IMPROVEMENT-LOG.md`
  - `C:\Users\jwals\octo\logs\pilot-V0-xhigh-1\` (`hermes.jsonl`,
    `assessment.md`)
