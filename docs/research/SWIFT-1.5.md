# Swift 1.5 (Qwen3.8-27B, GSQ-RCO GGUF): what it is, and whether it gets a sibling slot

Research note, 2026-09-26. Read-only. No model was downloaded; the only bytes read from a weight file were the
first 256 KiB of one GGUF, by HTTP range request, to read its header. Nothing was sent to `:1234` or `:11434`.
Sources: the Hugging Face API and raw files at pinned revisions, the HF discussion API, the Hacker News Algolia API,
two web pages, and this repo. Reddit refused every fetch (blocked), so the Reddit threads the vendor cites were not
read.

**Operator's report:** "this one is now making the rounds: ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF".
**Operator's standing position:** our bet is Bonsai plus our stack. A stronger model gets tested as a sibling only
after we have bug-free runs and the overthinking is tuned. It is tested as separate arms: Bonsai+stack, the other
model bare, and the other model with our stack.

## Verdict, in brief

- **Same base as our model, a different compression, and a heavier file.** This is UkisAI's own RL/OPD fine-tune of
  **Qwen3.8-27B**, the same dense 27B that Bonsai 2 is built on. It is quantised to 2.5-3.5 bpw by reusing
  ISTA-DASLab's per-tensor allocation (verified identical below). Bonsai 2 is ternary, 1.77 bpw. The architecture,
  chat template and tokenizer are **byte-identical** to what we serve, so the proxy needs nothing changed. Our fork
  should load the file. That is inferred from the source; the load has not been run.
- **What it claims is shorter thinking, not more capability.** The headline "58.5% fewer thinking tokens, 0.35%
  higher" is ONE benchmark (GPQA-Diamond): its median token reduction and a +0.31 pp score change. The agentic
  evidence is one self-reported row (Terminal-Bench 2.1, 69.21 vs 72.13). There is **no SWE-bench number** and no
  independent evaluation with n > 1. The only result for this exact quant is KLD.
- **On our card it costs speed and context.** The smallest usable file is 1.4-1.6x Bonsai's bytes. Expect roughly
  0.6-0.7x Bonsai's decode rate (a bandwidth estimate, unmeasured). At the same KV settings, main's window drops
  from 132k to about 58-77k. A 40% cut in thinking at 0.7x the speed is roughly a wash in wall time.
- **Found on the way, and closer to our bet:** `ukisai/Swift-Bonsai-2-GGUF`. It is the same "Swift" treatment
  applied to **Bonsai 2 itself**: PTQ1_0 at 5.95 GB, Apache-2.0, run on the PrismML fork. Its own card says its
  file-level evaluations show little token saving, and that tool use and agent tasks "remain uneven" (section 6).
- **Recommendation:** do nothing now. When the sibling slot opens, run the paired plan in section 7. Put
  Swift-Bonsai-2 PTQ1_0 in the queue beside it, not instead of it.

## How to read the citations

| prefix | source |
|---|---|
| `G:` | `https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF/blob/d74895b/`: the repo the operator named (HEAD `d74895bbe5db…`, created 2026-09-24 13:13Z) |
| `W:` | `https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b/blob/bc7a1e1/`: the BF16 fine-tune, the original |
| `D:` | `https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF/blob/d562806/`: ISTA's quant of the Qwen base |
| `Q:` | `https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f/`: the base |
| `SB:` | `https://huggingface.co/ukisai/Swift-Bonsai-2-GGUF/blob/a3bdac0/`: the Swift treatment of Bonsai 2 |
| `API` | `https://huggingface.co/api/models/<repo>?blobs=true`, fetched 2026-09-26 |
| `HD:<repo>#n` | an HF community discussion, `https://huggingface.co/api/models/<repo>/discussions/<n>` |

Line numbers are those of the raw file at that revision. `STRATA.md` means `docs/research/STRATA.md` in this repo.

---

## 1. What it is

### 1a. Base model, and the fine-tune's lineage

| layer | repo | evidence |
|---|---|---|
| base | `Qwen/Qwen3.8-27B`, Apache-2.0 | `Q:README.md#L3`; `API` `license: apache-2.0` |
| Swift 1.0 | `ukisai/Swift-Qwen3.8-27b` (created 2026-09-08) | `API` model list for `author=ukisai` |
| **Swift 1.5 BF16** | `ukisai/Swift-1.5-Qwen3.8-27b` (created 2026-09-16, last modified 09-24) | `W:README.md#L15-L16` `base_model: Qwen/Qwen3.8-27B`, `base_model_relation: finetune` |
| **Swift 1.5 GSQ-RCO GGUF** | `ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF` | `G:README.md#L7-L8` `base_model: ukisai/Swift-1.5-Qwen3.8-27b`, `quantized` |

- **Dense 27B, the same base as Bonsai 2.** Bonsai 2 is PrismML's ternary requant of Qwen3.8-27B. We serve
  BoldingBuilds' abliterated PTQ1_0 file with Qwen's MTP head grafted on (`models/manifest.yaml`, id
  `bonsai-2-27b-abliterated`, "base_model Qwen/Qwen3.8-27B via prism-ml's PTQ1_0").
- **Only the weights changed.** `W:config.json`, `W:chat_template.jinja` and `W:tokenizer_config.json` have the
  same SHA-256 as `Q:`'s (`191e0af2…`, `c3cf9e34…`, `b11349aa…`, hashed here). The shard sizes are identical.
  `W:NOTICE` says the same: those files "are retained from the parent model".
- **The GGUF header** (256 KiB range read of `Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ2_XS-mtp.gguf`):
  - `general.architecture = qwen35`, `general.name = Swift15-GSQfix-V1MIX-ISTAalloc-IQ2_XS MTP`
  - `qwen35.block_count = 65`, `qwen35.nextn_predict_layers = 1`
  - `head_count_kv = 4`, `key_length`/`value_length = 256`, `full_attention_interval = 4`
  - `context_length = 262144`, `tokenizer.ggml.pre = qwen35`

  These are the same keys and values as our served GGUF (`docs/research/SUDOINGX-VISION-CU.md` §5b).

### 1b. What "Swift 1.5" training is

- **Swift 1.0** (quoted from `W:README.md#L52`): "figuring out which tokens were linked to pathological overthinking
  and penalizing them without 'attacking' the reasoning length directly then regained the accuracy with RL and
  OPD". On HN the author cites a Meta paper aimed at overthinking in PTQ models, arXiv 2606.00206 (HN item
  49727511; the paper was not read here).
- **Swift 1.5** "scaled up the post-training (RL and OPD)", "this time with the main focus on long-horizon, agentic,
  and coding tasks" (`W:README.md#L36`, `#L53-L54`).
- **Data:** `ukisai/Qwen3.8-27B-multi-turn-agent-sft`, Apache-2.0, about 15,200 Terminus-2 agent traces generated
  by Qwen3.8-27B FP16 from OpenThoughts-Agent-v1-SFT tasks (its card). It is "not used out of the box, but rather
  re-sampled, turned into proper RL environments" (`W:README.md#L54`). The RL environments are not published.
- **Purpose:** fewer thinking tokens at base-level accuracy, with agentic coding named. The org calls itself "a small
  research lab from Europe" (dataset card), with offices in Belgrade, Eindhoven and Wilmington (ukisai.com).

### 1c. Its relation to Strata's "Swift 1.5" and to Flash-Next

**STRATA.md covers a different model with the same series name.** Strata's alternative weights are
`ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF` (`b22d729`), a fine-tune of the 125B MoE Flash-Next. Its
`base_model` is `ukisai/Swift-Qwen3.8-Flash-Next` (`API`). Its "63.4% fewer thinking tokens" figure (STRATA.md M1)
is that model's claim. The repo the operator named is the **27B dense** sibling.

Both come from the same org, released the same day with the same quant recipe (ISTA allocation + Swift refinement).
Strata cannot run the 27B: its kernels are qwen4exp-only (STRATA.md §4). The 27B is an ordinary `qwen35` GGUF.

### 1d. Uploader and provenance

- **Official, not a re-upload.** `ukisai` is the fine-tune's author. The BF16 original, the standard GGUFs, this
  GSQ-RCO repo and the Flash-Next and Bonsai variants are all under the same account (`API` list: 23 models). The
  GSQ-RCO card links back to the BF16 card (`G:README.md#L23`) and vice versa (`W:README.md#L25`, `#L178`).
- **Not an ISTA release.** ISTA published only the Qwen-base quants (`D:`). UkisAI reused ISTA's allocation and
  says so: "This release reuses ISTA's allocation search results; it does not claim a new RCO search"
  (`G:README.md#L94`).
- **Verified here:** the per-tensor type assignment in `G:tensor-allocation/…IQ2_XS-mtp.rco-allocation.txt` is
  **identical** to `D:tensor-allocation/Qwen3.8-27B-GSQ-RCO-IQ2_XS-mtp.rco-allocation.txt`. All 866 tensors, same
  types. The file sizes differ by 160 bytes: 8,771,311,616 vs 8,771,311,680.
- **Integrity:** `G:SHA256SUMS` and `G:release-manifest.json` list all 8 files. Each sum equals the HF LFS sha256
  from `API`, checked for all 8.
- **Not reproducible from what is published.** The importance matrix is published (`imatrix-swift15-v1mix.gguf`),
  and so is the allocation. The refinement step, `refine_gguf_fast.py` "(Astra IQ1 stage_objective patch)", is not
  (`G:RECIPE.txt#L10-L11`, `#L19`). We would pin by hash, as we do for every model.
- **Stale or contradictory metadata.**
  - `G:README.md#L71` says "Authenticate with an account granted access while this repository is private", but
    `API` reports `private: false, gated: false`.
  - `W:README.md#L7` says `gated: true`; `API` says `gated: false`.
  - The GSQ card says "9.18× speed-up" (`G:README.md#L32`) where the BF16 card says "1.95×" (`W:README.md#L34`).
    9.18 is 104.6 / 11.39 min, the single planet-game demo (`W:README.md#L48`).
- **Traction:** 22,455 downloads and 90 likes in two days (`API`). Swift 1.0's GGUF had 189,413 downloads.

### 1e. Licence

- **Base:** Qwen3.8-27B is **Apache-2.0** (`Q:README.md#L3`; the `Q:LICENSE` file is 11,544 bytes, the same size as
  `G:LICENSE-APACHE-2.0`).
- **Fine-tune: "Swift Open License v1.0"** (`G:LICENSE`, the same size as `W:LICENSE`). It is Apache-2.0 in shape,
  with one material addition, section 5 (quoted from `G:LICENSE#L158-L162`): "(a) The rights granted under this
  License for Commercial Use are conditioned upon You or Your Legal Entity not exceeding the Threshold. (b) Any
  Commercial Use of the Work or a Derivative Work by a Legal Entity that exceeds the Threshold is not licensed
  under this License."
  - The Threshold is "gross revenue of one million United States dollars (US$1,000,000) or more, measured over the
    most recently completed fiscal year", affiliates included (`#L87-L90`).
  - Quantised weights count as the Work in "Object" form (`#L26-L29`).
  - Section 12 (`#L221-L225`): on any breach the licence ends, and you "must cease all use of the Swift Contribution
    ... and delete all copies".
  - Section 6 (`#L171-L177`): Qwen's Apache rights in the base are untouched.
- **For us:** personal and internal use is fine, and so is commercial use under US$1M revenue. An entity above that
  needs a "Swift Enterprise License". Compare the licences in the same line:
  - Bonsai 2 (ours): Apache-2.0 (`models/manifest.yaml`).
  - Qwen Flash-Next: its own licence, with the MaaS / coding-assistant clause (STRATA.md M2).
  - **Swift-Bonsai-2: Apache-2.0** (`SB:README.md#L2`, `SB:LICENSE`).

## 2. The quantisation

### 2a. Files

| file | bytes | bpw (excl. head) | dev KLD vs Swift BF16 |
|---|---:|---:|---:|
| `…-IQ2_XS.gguf` / `-mtp` | 8,422,841,632 / 8,771,311,616 | 2.505 | 0.189979 |
| `…-IQ2_S.gguf` / `-mtp` | 9,259,511,072 / 9,607,981,056 | 2.754 | 0.134751 |
| `…-IQ3_XXS.gguf` / `-mtp` | 10,094,357,792 / 10,442,827,776 | 3.002 | 0.097774 |
| `…-IQ3_S.gguf` / `-mtp` | 11,771,546,912 / 12,120,016,896 | 3.501 | 0.051265 |

- bpw is bytes × 8 / 26,895,998,464 parameters (`API` `gguf.total`). It matches ISTA's stated 2.50/2.75/3.00/3.50
  (`D:README.md#L76-L81`); the IQ2_XS file with its head is 2.5652 (`D:` allocation header).
- For comparison, **Bonsai's trunk is 1.77 bpw** (5.95 GB), and our served file with its head is 6,297,658,880
  bytes (`models/manifest.yaml`).
- **No mmproj in this repo.** "A Swift 27B vision projector has not been verified for this release"
  (`G:README.md#L85`). One user ran Qwen's BF16 mmproj with IQ3_S-mtp and said it "worked normal", on a small
  sample (`HD:ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF#1`).

### 2b. Tensor types

**The tier name is a budget, not a type.** From the IQ2_XS-mtp dump (`G:tensor-allocation/…IQ2_XS-mtp…`, line 5),
the histogram is:

> BF16 96, F32 360, IQ1_M 31, IQ1_S 36, IQ2_S 60, IQ2_XS 53, IQ2_XXS 70, IQ3_S 47, IQ3_XXS 37, IQ4_XS 11, Q2_K 53,
> Q4_K 4, Q6_K 8

- `token_embd` is IQ1_M and `output` is IQ4_XS.
- The DeltaNet `ssm_alpha`/`ssm_beta` tensors are BF16.
- The 15 MTP-head tensors (`blk.64.*`) are Q6_K/F32.

### 2c. Engine support

- **Upstream llama.cpp:** every type above is a long-standing ggml type, and ISTA says the files "run unmodified in
  `llama.cpp`, Ollama, and LM Studio" (`D:README.md#L57`). UkisAI quantised with llama.cpp `fee39dd`
  (`G:RECIPE.txt#L7`).
- **Our fork** (`llama-bonsai2`, sudoingX `bonsai2` @ `285542d98`, rebased on PrismML b10709; `docs/ENGINES.md`):
  - **Expected to load it. Inferred, not run.**
  - The arch is `qwen35`, which the fork serves (`src/models/qwen35.cpp` in our build tree).
  - The i-quant CUDA kernels are present (`ggml/src/ggml-cuda/mmvq.cu` handles `IQ2_XS`).
  - The fork's MTP Hadamard fix applies only when the file declares Hadamard-rotated tensors (`qwen35.cpp:642`,
    `if (hadamard_inverses)` … `find(tok_embd_w)`). A standard GGUF declares none, so that path stays inert.
  - Our reasoning-budget nudge patch (0001) is in the sampler and is model-agnostic.
  - **No new engine is needed**, unlike Flash-Next, which needs upstream (STRATA.md M6). Step 0 (section 7) is the
    proof.
- **MTP head:** present in the `-mtp` files. It is Qwen's base head: "The published weights include the base
  model's MTP head" (`W:README.md#L344`). It was added by `add_mtp15.py` as "blk.64 Q6_K/F32" (`G:RECIPE.txt#L15`).
  - It was not retrained on Swift's trunk. Its "decoding speed and quality have not been separately evaluated"
    (`G:README.md#L45`; `G:release-manifest.json` `"evaluated": false`).
  - We are in the same position with our graft: a stock head on an abliterated trunk, draft acceptance 0.80-0.84 on
    code (`config.yaml`, `bonsai` cmd comment).
  - ISTA measured the Qwen-base quants at a 54.2% mean acceptance with 3 draft tokens (`D:assets/plots/…mtp…png`).
    That is a different metric from our n-max 1.

### 2d. Chat template, efforts, markers, tokenizer

- **The template is identical to ours.** The template embedded in the GGUF (`API` `gguf.chat_template`) is
  byte-identical to `Q:chat_template.jinja`. Our fixture `mcp/fixtures/bonsai_chat_template.jinja` is also
  byte-identical to it (CRLF normalised). Consequences:
  - Efforts: `xhigh` (the default), `medium`, `low`. `tiers.accepted_efforts()` reads them unchanged.
  - The same `<think>`, `</think>`, `<|im_start|>` and `<|im_end|>` markers (`STRAY_MARKERS`, `END_OF_TURN`).
  - The same tool-call format and the same `preserve_thinking` default.
  - Offline gates that render the served template (`test_ledger`, `test_stream`, `test_sessions`) exercise the same
    bytes.
- **The tokenizer is identical to Qwen's** (hashes in 1a; GGUF `bos` `<|endoftext|>`, `eos` `<|im_end|>`). Our
  served GGUF uses the same `qwen35` pre-tokenizer.
  - **Concept seeds need nothing for a test.** A seed is a word, drawn from `index/token_embd.npz`, and that word
    tokenises identically.
  - Re-extracting from Swift's own `token_embd` would fail as the code stands:
    `scripts/extract_token_embd.py:161-163` refuses anything but PTQ1_0, and this `token_embd` is IQ1_M.

## 3. Fit on the RTX 5060 Ti 16 GB

**Same architecture, so the same KV per token.**
- 16 full-attention layers × 4 KV heads × 256 × (K+V) = 32,768 values per token, or **34 KiB at q8_0** in theory.
  The stack budgets **44 KiB/token**, measured (`mcp/budget.py:126`; SUDOINGX-VISION-CU.md §5b explains the gap:
  draft cache and buffer growth).
- DeltaNet state is fixed per sequence: 48 layers × 48 heads × 128 × 128 × fp32 ≈ 151 MB.
- Both are the same for Bonsai and Swift. **Only the weights differ.**

Today `-c 181248` q8_0 gives main 132,096 + helper 49,152 at a 600 MiB free-at-peak target (`config.yaml`,
`bonsai` block). Holding everything else fixed, each extra MiB of weights costs 1024/44 ≈ 23.3 tokens of pool.
Computed, not measured:

| file | Δ weights vs our served 6,006 MiB | pool `-c` (rounded to 1,024) | **main share** (helper fixed 49,152) |
|---|---:|---:|---:|
| Bonsai (served) | 0 | 181,248 | **132,096** |
| Swift IQ2_XS-mtp | +2,359 MiB | ~125,952 | **~76,800** |
| Swift IQ2_XS (no MTP) | +2,027 MiB | ~133,120 | ~83,968 |
| Swift IQ2_S-mtp | +3,157 MiB | ~107,520 | **~58,368** |
| Swift IQ3_XXS-mtp | +3,953 MiB | ~89,088 | ~39,936 |
| Swift IQ3_S-mtp | +5,553 MiB | ~51,200 | ~2,048: **does not fit** with the helper as sized |
| Swift-Bonsai-2 PTQ1_0 (no head yet) | −335 MiB | ~188,416 | ~139,264 |

- **Context is a real cost for agentic runs.** Octopus v0b-V0 peaked at 104,907 prompt tokens, and the reference
  run finished at 125k (`config.yaml` 2026-09-25 comment). Only IQ2_XS keeps main above 75k at q8_0. At IQ2_S, Hermes
  would compact roughly twice as often. q4_0 K/V would buy the context back, but it is a second variable whose
  quality is unmeasured on this model (docs/CONTEXT-EXPANSION.md). Keep it out of the first test.
- **Compute buffers may differ** with the i-quant MMQ path. That is unmeasured, so measure the peak (Step 0).

**Speed class (an estimate, not a measurement).**
- Decode is bandwidth-bound, and each token reads every weight once. The 5060 Ti's spec bandwidth is 448 GB/s.
- Scaling Bonsai's 47-61 tok/s at 8k (operator's figure; 55.33 in `docs/KNOWN-ISSUES.md:42`) by weight bytes:
  - IQ2_XS ≈ **33-43 tok/s** (0.71×)
  - IQ2_S ≈ 30-39 (0.64×)
  - IQ3_XXS ≈ 28-36 (0.59×)
- Two reasons it could land lower: i-quant dequantisation is table-driven, and PTQ1_0 has its own tuned mat-vec
  kernel (PR #218). The MTP acceptance of an untuned head on a changed trunk is unknown.
- **Published tok/s:** none from UkisAI for the GSQ files. ISTA's MTP plot shows about 100-118 tok/s decode for the
  Qwen-base quants with 3 draft tokens, **hardware not stated** (`D:assets/plots/…mtp_speculative_decoding.png`).
  Community figures are for other quants and cards: Swift 1.0 Q4_K_M at ~44-55 t/s with MTP, card not stated
  (`HD:ukisai/Swift-Qwen3.8-27B-GGUF#3`); Swift 1.5 Q4_K_M at 44 tok/s at 100k context, card not stated
  (`HD:ukisai/Swift-1.5-Qwen3.8-27B-GGUF#3`).
- **The arithmetic that matters.** If Swift really cut thinking tokens by 40% on our traffic, then at 0.7× decode an
  answer takes about 0.6 / 0.7 ≈ 0.86× the time. That is roughly a wash, and on this card the speed argument for
  it mostly disappears. Only its accuracy could justify it.

## 4. Published evaluations

**All of them are self-reported. None is independent at n > 1 for Swift 1.5.**

### 4a. Swift 1.5 BF16 vs Qwen3.8-27B BF16 (UkisAI's runs)

Source: `W:README.md#L108-L130`. vLLM 0.27.1, thinking xhigh, 5 seeds; Terminal-Bench is 5 trials per task on
Harbor at 131,072 context.

| benchmark | Qwen3.8-27B | Swift 1.5 | mean thinking tokens |
|---|---:|---:|---:|
| GPQA-Diamond | 88.28 | 88.59 | 15,014 → 8,717 (−41.9%; median −58.5%) |
| IFBench | 73.53 | 72.07 | −38.5% |
| AIME 2026 | 98.67 | 96.00 | −40.0% |
| HMMT Nov 2025 | 99.33 | 97.33 | −32.1% |
| LiveCodeBench v6 (output cap 32,768) | 76.76 | **81.71** | −24.5% |
| Terminal-Bench 2.1 | 69.21 | **72.13** | 52,265 → 43,733 (−16.3%; median −0.1%) |

- **Effort sweep, GPQA only** (`W:README.md#L165-L167`):

  | effort | Qwen | Swift | thinking |
  |---|---:|---:|---:|
  | xhigh | 88.28 | 88.59 | −41.9% |
  | medium | **84.14** | **82.22** | −24.8% |
  | low | 84.04 | 84.85 | −28.7% |

  **Medium is the effort our stack sends at every thinking tier from `low` to `xhigh`; only `max` sends xhigh** (the
  tier table in AGENTS.md). It is the one effort where Swift scores below the base.
- **The base numbers are not Qwen's.** Qwen reports its own 27B at LCB v6 **90.3** and Terminal-Bench 2.1 **73.0**
  (`Q:README.md`, benchmark table). ISTA measured BF16 LCB v6 at **85.71** (`D:README.md#L98`). Swift's base run
  gives 76.76. Three harnesses give three LCB scores for one model.
  - Swift's 32,768-token output cap truncates a longer thinker more often. So part of a short-thinking model's LCB
    "gain" may be fitting inside the cap. That is an inference, not tested.
- **The base drifts between UkisAI's own releases.** For Swift 1.0 the author gave Terminal-Bench 2.1 as "Base
  66.74% vs Swift 65.84%" (HN item 49727511, 2026-09-16). The Swift 1.5 card gives the base as 69.21.
- **No SWE-bench, SWE-bench Pro, DeepSWE or Toolathlon rows exist for Swift.** Qwen's table has them for the base
  (SWE-bench Pro 61.7, DeepSWE 42.2), and so does STRATA.md M3.

### 4b. What quantisation costs (the quantiser's own numbers)

- **For Swift's GSQ files: KLD only.** `G:evaluation/heldout-kld.tsv` measures each quant against Swift BF16 at a
  512-token context. For IQ2_XS on code (CodeParrot), Swift's KLD is 0.1209 against ISTA's 0.1267 (each against its
  own BF16). The card says so itself: these are "not task accuracy" and "do not establish quality at 32K or longer
  contexts" (`G:README.md#L53`, `#L67`).
- **Task cost, ISTA's runs on the Qwen base** (`D:README.md#L96-L102`):

  | variant | AIME25 | GPQA-D | LCB v6 |
  |---|---:|---:|---:|
  | BF16 | 100.00 | 89.90 | 85.71 |
  | IQ2_XS | 96.67 | 84.85 | **76.57 (−9.1)** |
  | IQ2_S | 100.00 | 86.36 | 82.29 (−3.4) |
  | IQ3_XXS | 100.00 | 88.89 | 84.57 (−1.1) |

  The KLD levels are similar, so Swift's IQ2_XS probably loses a similar amount. That is an inference. **At IQ2_XS,
  the quant's LCB loss (−9) exceeds Swift's claimed LCB gain over base (+5).**
- **UkisAI's INT4 exports** (`W:README.md#L200-L253`) are single-seed, served on vLLM, and not GGUF. They are not
  evidence for this repo.

### 4c. Against Bonsai 2's own numbers

- PrismML reports Bonsai 2 at AIME25 95.00 and LCB 90.07, against the 27B FP16's 96.67 and 90.05 (STRATA.md M4, `B:`
  card). By PrismML's measure, ternary costs essentially nothing on those two benchmarks.
- No harness is shared with any Swift table, so no subtraction is valid. Qualitatively: Bonsai at 1.77 bpw reports
  keeping LCB, while ISTA's 2.5 bpw quant of the same base loses 9 points in ISTA's harness.
- Our own measurements of Bonsai are on no benchmark UkisAI reports (STRATA.md M4: Octopus V0 19/27 at n=1,
  LiveBench n=21, SWE-bench Verified 2/2).

## 5. Community reports ("making the rounds")

Where: HF discussions on the five UkisAI repos (read), a Show HN for Swift 1.0 (2026-09-16, 33 points, 17
comments), and a 2026-09-24 HN story for the series (1 point). The r/LocalLLaMA thread the vendor cites as
"independent evals" (`/r/LocalLLaMA/comments/1wg7dd5`) could not be fetched. One blog, MindStudio (2026-09-19).
None is reproducible as published: no logs, no data files, n = 1 task or a handful.

| report | model | claim | weight |
|---|---|---|---|
| `HD:…Swift-1.5-Qwen3.8-27b#4` (09-26) | 1.5 IQ3_XXS vs 1.0, llama-server, effort `low` | 1.5 makes "20%–44% more `<think>` tokens" and "frequently hits the 8k budget limit" | anecdote; the **opposite** of the claim, on the quant family in question |
| `HD:…Swift-1.5-Qwen3.8-27b#3` (09-26) | 1.5 Q8, 256k | after 12-14 h of editing, "the number of errors it made was simply insane", worse than Qwen 3.6 | anecdote, agentic use |
| `HD:…Swift-1.5-Qwen3.8-27B-GGUF#3` (09-26) | 1.5 Q4_K_M, 100k context | it claimed the user had confirmed a fix they never confirmed | one transcript |
| `HD:…Swift-Qwen3.8-27B-GGUF#3` (09-15) | **1.0** Q4_K_M vs Unsloth UD Q4_K_M, 3 runs × (100 GSM8K + 164 HumanEval) | 20-38% shorter completions, GSM8K 97.0 vs 95-100, HumanEval 97.6 vs 98.8 | the most systematic report; Swift 1.0, not 1.5 |
| `HD:…Swift-Qwen3.8-27b#7` (09-16) | **1.0** BF16 vs Qwen FP8, AIME 2026, SGLang, temp 0.6 | +14% reasoning tokens, 26/30 vs 27/30 | the vendor replied that the settings were wrong, and admitted a training bug "penalising a math related token" |
| MindStudio blog (09-19) | version unstated, vLLM, 80 GB | one hard coding prompt; token and tool-call counts differ widely between the two models | n=1 task |
| Show HN (09-16) | 1.0 | the author: loops occur "very rarely (as opposed to the Base ...)" | vendor |

**Net:** the shorter-thinking effect reproduces on short tasks for 1.0. For 1.5 on agentic or long work there are
two negative anecdotes and no positive measurement.

## 6. Found on the way: Swift-Bonsai-2 (our base, their treatment)

`ukisai/Swift-Bonsai-2-GGUF` (created 2026-09-22; 1,651 downloads).

- **What it is:** "UkisAI's reasoning-efficient derivative of Prism ML's Ternary Bonsai 2 27B" (`SB:README.md#L34`).
- **Format:** `Swift-Bonsai-2-PTQ1_0.gguf`, 5,946,648,960 B, and `…-PQ2_0.gguf`, 7,206,168,928 B. Each is a
  "plain Bonsai 2 pack": no adapter and the same size as the base (`#L65`). Hashes are in `SB:model_info.json` and
  match `API`.
- **Runtime:** the PrismML fork, "tested at revision `1a07bfa5f`" (`#L193`). The template is byte-identical to
  Qwen's (hashed). **Apache-2.0.**
- **Its evidence is weak, and the vendor says so.**
  - The GPQA and C-Eval rows (median thinking −39.8% on GPQA) compare the base with "the historical **Swift
    runtime correction**", not these files (`#L76`, `#L177`).
  - The file-level rows are for **PQ2_0 only**: IFBench thinking −1.7% mean; AIME 2025 completion **+4.3%**, with
    a 95% CI of −4.0..+8.0 pp (`SB:benchmark_results.json`).
  - "PTQ1_0 remains the earlier Swift release" and has no evaluation of its own (`#L65`, `#L177`).
  - The card's own caveat (`#L185`): "benchmark scores did not consistently translate into reliable general-purpose
    behavior. Instruction following, tool use, and open-ended coding or agent tasks remain uneven."
- **Differences from ours:**
  - It is **not abliterated**; ours is.
  - It has **no MTP head**. It would need the same graft we did (docs/MODELS.md).
  - It keeps our speed and our window.
- **Why it matters here:** it attacks overthinking at the weights of the very model we bet on. On this card it is the
  only Swift variant that costs no speed and no context.

## 7. What a sibling test would take (described, not run)

**Precondition (operator):** bug-free runs and overthinking tuned first; fix → test → deploy → run. Nothing below is
started before that. The download (~9-10 GB) is an operator decision.

### 7a. Changes, all additive, none in `mcp/`

| where | change |
|---|---|
| `models/manifest.yaml` | new entry, status `trial`: repo `ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF`, revision `d74895bbe5db4bec1e0024e7cc87d59c02d7631a`, the chosen file, its size and sha256 (from `G:SHA256SUMS`, which equals LFS), licence `swift-open-license-1.0 (+ Apache-2.0 for Qwen components)`. `verify_artifacts.py` then accepts it. |
| `engines/manifest.yaml` | **nothing**, if Step 0 loads the file on the shipped `llama-bonsai2` build. `build_engine.py --verify-only` stays green. |
| `config.yaml` | a `swift-iq2s` entry. It is `bonsai`'s cmd with `-m` swapped, the same `-dev CUDA0`, UUID pin, `--spec-type draft-mtp --spec-draft-n-max 1`, q8_0 K/V (q4_0 draft), `--jinja --reasoning-format deepseek`, `--reasoning-budget` + message, `--no-cache-idle-slots`, `--no-context-shift`, `${sampling}` (Swift's card uses the same 1.0/0.95/20/0, `W:README.md#L129`), and `-c` from the table in §3 re-measured in Step 0. Also a trial group like `context-trial` (swap, non-exclusive), and a **profile** `model-trial-swift` pinning `bonsai`/`bonsai-agent` to it, exactly as the `context-trial-*` profiles do. With a profile active, the proxy, deep thinking and the worker all get Swift with **no code change**, and deactivating it restores Bonsai. `GGML_CUDA_BATCH_INVARIANT` is PTQ1_0-specific; leave it set so the env matches. |
| `mcp/budget.py` | nothing. The pool is read from `/props`, and 44 KiB/token holds (same architecture). Re-check the peak. |
| proxy, tiers, markers, ledger, sessions | nothing. The template, efforts and markers are byte-identical (§2d). |
| concept seeds | nothing (§2d). |
| vision | nothing. `describe_image` uses `bonsai-vision` on the A4000. |
| run records | record the active llama-swap profile (`GET :11434/api/profiles/active`) and the model sha in every run row. `x_yamadori` does not say which weights answered. |

### 7b. File choice

- **IQ2_S-mtp is the quality arm.** In ISTA's run on the Qwen base it lost 3.4 on LCB, where IQ2_XS lost 9.1. It
  leaves main ≈ 58k.
- **IQ2_XS-mtp is the fallback** if Step 0 shows IQ2_S too slow or the window too small: main ≈ 77k, 0.71×.
- IQ3_* does not leave a useful window at q8_0.

### 7c. Arms

These are the operator's three, plus one control so the window is not confounded.

| arm | model | path | tier |
|---|---|---|---|
| A | Bonsai + stack (as deployed) | `:1234`, no profile | `xhigh` (the Octopus arm) |
| A-w | Bonsai + stack, **window-matched** (`-c` = Swift's) | `:1234`, a `context-trial`-style profile | `xhigh` |
| B | Swift **bare** | `:1234` with the profile | `low` ("the model as it ships, the benchmark baseline", AGENTS.md tier table) |
| C | Swift + stack | `:1234` with the profile | `xhigh` |

"Bare" goes through `:1234` at tier `low`, never straight to `:11434` (AGENTS.md "Never test around the stack"). A
Bonsai-bare arm (A at `low`) is cheap to add on the LiveBench step.

### 7d. Steps, in order, each a gate

**Step 0: does it run here** (about 1 h, maintenance window, one GPU consumer).
1. Load on the shipped fork build.
2. Corruption checks as in docs/MTP-STAGING.md, **on the 5060 Ti**:
   - character-run generations 0/16
   - tool calls 10/10 with MTP on
   - draft acceptance by category
3. Peak VRAM under the stress load (`bench/kv_context/vram.py`); set `-c`.
4. Decode and prefill at 4k, 32k and 64k, n=3.
5. The STEP 0 cache probes (prefill, warm, diverge) read from `x_yamadori.cache`.
6. `scripts/run_tests.py` offline, and the live suite through `:1234` with the profile active.

**Stop if:**
- decode is below ~25 tok/s at 32k
- the extend-the-slot request reprocesses the prompt
- corruption appears
- or peak free VRAM is under 600 MiB at the chosen `-c`

**Step 1: the claim itself, cheapest layer** (PROTOCOL 5). Replay a fixed set of our own `agent_step` requests at
effort `medium` (what our tiers send) and `xhigh`, with Bonsai and Swift, n=3 each. Take them from the Octopus v0e
transcripts that the overthinking scripts already index.
- **Metric:** reasoning tokens per request and wall time per request, paired by request.
- If Swift's thinking reduction on OUR traffic, net of its slower decode, gives no wall-time saving, stop here. Its
  capability case then rests on evidence that does not exist.

**Step 2: Octopus V0, paired** (the decision run). Same prompt (`prompt_sha256 d30c0264…`), `--iterative 6`, arms
A, A-w and C, **n ≥ 2 each, interleaved** (PROTOCOL 6, 10). Grade the reference fixture in the same session to
calibrate the grader spread.
- **Record:**
  - `grade.py` spec_passed/27
  - wall time, and time to final grade (`snaps.py`)
  - post-answer time (docs/research/OVERTHINKING.md's metric)
  - reasoning and completion tokens from relay `usage`
  - deep-thinking runs, compactions, tool calls
- **Decision rule, set now:** C replaces nothing unless its mean grade beats A's by more than the reference spread
  AND its wall time is no worse than A's. Swift's selling point is time.
  - A-w vs A tells how much of any difference is the window.
  - C vs A-w is the model effect at equal window.

**Step 3: corroboration.**
- `bench/livebench` bare arms (A-bare vs B), paired n=21.
- `bench/domain`.
- A 10-20 instance SWE-bench Verified slice through mini-swe-agent, the same instances for each arm. This would be
  the first SWE-bench number anyone has for Swift.

**The same plan fits Swift-Bonsai-2 PTQ1_0 with fewer steps.**
- It needs the MTP graft (`graft/tools/merge.py`, as in docs/MODELS.md). It keeps `-c 181248`.
- Step 0 shrinks to the corruption checks and the cache probes.
- Step 1 is the whole question: does it cut thinking on our traffic without losing tool-call validity? Its own card
  doubts the latter.
- One difference to record: it is not abliterated.

### 7e. What each outcome means

- **C beats A and A-w in Step 2, and holds in Step 3:** plan a migration. Before any commercial use by an entity
  above US$1M revenue, a licence review is due (§1e).
- **C ties or loses:** either the Swift gain does not survive 2.5-2.75 bpw, or it is eaten by 0.64-0.71× decode and
  half the window on this card. Bonsai stays, and the result is recorded in docs/SELF-IMPROVEMENT-LOG.md style.
- **Step 1 shows no thinking saving at `medium`:** that is consistent with UkisAI's own effort table (§4a) and the
  `HD:…#4` report. Stop.

## 8. Recommendation

**Respect the sequencing: nothing runs now.** This is not a stronger model on the evidence. It is the same base
with a token-efficiency fine-tune:
- Its capability claims are one self-reported agentic row and one LCB row under a 32k output cap.
- Its only quant evidence is KLD.
- Its community reports on agentic work are negative or mixed.
- On our card it runs at ~0.6-0.7× speed with about half the context.

Its one real advantage is integration. The same template, tokenizer, architecture and (expected) engine mean a
sibling test costs a download, a config entry and a profile, with no code.

When the sibling slot opens, rank the two candidates:
1. **Swift-Bonsai-2 PTQ1_0, first.** It is a direct probe of the overthinking question on our own base, at no cost in
   speed, context or licence. Step 1 alone decides it.
2. **Swift 1.5 IQ2_S-mtp** (IQ2_XS-mtp as fallback). Run the full A / A-w / B / C plan above.

Qwen3.8-Flash-Next (STRATA.md M7) remains the only candidate whose own evidence says "more capable". It needs an
upstream engine and a RAM/CPU-offload feasibility test first.

## Reproduce

- Metadata: `curl -s "https://huggingface.co/api/models/<repo>?blobs=true"` for the repos in the prefix table.
  Raw files come from `https://huggingface.co/<repo>/raw/<sha>/<path>`.
- Allocation identity: fetch the two `…IQ2_XS-mtp.rco-allocation.txt` files, normalise `name: TYPE` to `name=TYPE`,
  sort, then `diff`. The output is empty (866 lines each).
- Header: `curl -r 0-262143 https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF/resolve/d74895bbe5db4bec1e0024e7cc87d59c02d7631a/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ2_XS-mtp.gguf`,
  then parse the GGUF v3 KV table. The first 41 of 49 KVs fit before the tokenizer arrays.
- Template identity: `sha256sum` of `Q:chat_template.jinja` and `W:chat_template.jinja`; `diff` against
  `mcp/fixtures/bonsai_chat_template.jinja` and against the GGUF's `tokenizer.chat_template` with CR stripped.
- Discussions: `https://huggingface.co/api/models/<repo>/discussions[/<n>]`. HN:
  `https://hn.algolia.com/api/v1/items/49727511`.
