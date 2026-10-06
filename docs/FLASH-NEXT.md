# Flash-Next: the max tier's model (PHASE 1 done, PHASE 2 waits for "GPU free")

Operator, 2026-09-27: "The next test for our proxy would be offering Qwen 3.8 Flash Next ... This is our Max mode ...
The proxy will swap in this model when using max mode." Follow-ups the same day: max swaps the model out ("you choose
Max you get this model, that is it"); ONE big model loaded at a time; at max EVERY call of the conversation goes to
Flash-Next; a non-max request while Flash-Next serves max is refused ("model at capacity"); vision folds into the main
model (Flash-Next loads with its own mmproj; bonsai-vision on the A4000 retires); embeddings and imagegen stay on the
A4000.

PHASE 1 (this document's evidence) touched no GPU and sent nothing to :1234 or :11434: sources read, engines built on
the CPU, files downloaded and hashed. **Nothing below is measured on our card.** Every speed number is someone else's,
labelled as such. Background: docs/research/STRATA.md (section M, and its 2026-09-27 update).

## 0. Update 2026-09-28: Strata's "2x", and porting it (operator: PORT, not serve on Strata)

**What the 2x is.** Strata `d551edf4` (21 commits after `b742ff99`): engine 0.1.13's PROMPT reading, against
Strata's own 0.1.12 -- Q2_0 32K prompt 572 -> 1,290 tok/s, IQ3_S 383 -> 1,208, one prompt per step, RTX 5070 12 GB
(`bench/results/2026-09-28-prefill-speed`); "output speed is unchanged". 0.1.14's matrix: IQ2_XS prompts
461-1,238 tok/s, output 74.4 / 71.5 / 64.3 / 59.8 / 52.8 tok/s at 1K / 32K / 64K / 128K / 262K with MTP and ~3,600-4,400
experts resident in VRAM (their GPU expert cache). No llama.cpp baseline is published.

**The two paths, sized** (the coordinator's question, before the operator decided):

| | (1) port into our llama.cpp engine | (2) serve max mode on Strata's engine |
|---|---|---|
| what the proxy gets | everything llama-server gives Bonsai: slots + `id_slot`, `/slots` release/erase, `/apply-template`, `/tokenize`, `logprobs`/`top_logprobs` (the decider), a prompt cache per slot plus `--cache-ram`, the reasoning budget + our nudge, parallel requests, tool-call grammar, `--mmproj` | `/v1/chat/completions`, `/v1/messages`, `/v1/models`, `/health`, `/status`, `/metrics` only (`serve/server.py:916-996`): no slots, tokenize, apply-template, logprobs, budget or nudge; ONE request at a time and ONE cached conversation (a helper run or side call evicts main's cache: a full re-read, ~80 s at 100K and 1,200 tok/s); `stop`, `tool_choice`, `response_format` ignored; its own model pack format (another ~40 GB artifact); no licence; release binary unpinned |
| closing the gaps | patches in the engine (this section); proxy plumbing as planned (section 5) | a proxy adapter that turns off slot pinning, warms, release, the decider, window checks by /tokenize (or re-implements them in Python), serialises every max call; Strata patches for logprobs, the budget force-close and several cached conversations (its checkpoints are ~118 MB each and positional: a multi-conversation cache is a redesign) |
| risk | draft upstream PRs rebased; new code paths gated below | the proxy's whole cache/slot design does not hold at max; an engine with one author |

**Operator, 2026-09-28: path (1)** -- "the whole [point] is that we get some of the improvements with other models
we run too. So it is worth faithfully porting in to keep our own version."

**Built (CPU only): `llama-upstream-moe`** (`C:/Users/jwals/engines/llama-upstream-moe-2c8dbf5f`, docs/ENGINES.md
"Strata's MoE work, ported"): our nudge; the expert streaming ring as upstream PR #28414 (rebased); the GPU expert
cache as PR #27861's exact split (rebased) with Strata's expert-profile selection, its adaptive promotion tier and
every CPU-computed batch (0004); pinned experts under mmap and Strata's Windows large-page arena (0005). MMQ experts
and the shared scratch are llama.cpp's already; the prefill chunk size is `-ub`. Strata's expert profile is
`models/manifest.yaml` `flash-next-expert-profile-strata`; `scripts/moe_profile.py` builds one for any MoE model.
Not yet: the PLE row prefetch thread (qwen4exp/gemma4), the AVX2 multi-token CPU kernels, the GPU's PCIe share, the
KV-placement span (0038) on this base (measured first: the gate's `nounified` arm).

**The gate:** `bench/flashnext_gate.py` -- kernels (test-backend-ops), fit, exact (slot-state hash + greedy text,
Strata's bit-identical rule), kl (teacher-forced, Strata's same-engine yardstick), needles (Strata's 5/5 per depth),
corrupt (engine_corruption.py), speed. Arms: `base`, `moe-off`, `prefetch`, `pinned`, `largepages`, `eager`,
`cache-lru`, `cache-strata`, `all`, `nounified`, `ub2048`, `ub4096`, `mtp`. Runs only with `--window` on "GPU free".

## 1. Engine: upstream llama.cpp, not Strata, not our forks

| question | answer | evidence |
|---|---|---|
| Can `llama-bonsai2-ada` (PrismML `adfffbe`) or `llama-bonsai2` load it? | **No.** PrismML's `prism` branched from upstream on 2026-08-25 (merge base `5ea87dda`); qwen4exp landed two days later. `src/models/qwen4exp.cpp` is absent at `adfffbe` and at prism's head `84445367`. Upstream is 607 commits ahead of that merge base. | `gh api .../compare` |
| Can upstream load Bonsai? | **No.** PTQ1_0 (type 143) is PrismML's; upstream's gguf-py rejects the Bonsai file (`143 is not a valid GGMLQuantizationType`); upstream closed the PQ2_0/PTQ1_0 PR (#29077) unmerged. So two engines, one per main model. | read on the file |
| Upstream support for the architecture | `qwen4exp` = 12 x (3 Gated DeltaNet + 1 Qwen Sparse Attention), 512 experts (10 routed + 1 shared), hyper-connections, the PLE n-gram table. Added by #27742 (`6c84c7d5d`, 2026-08-27); fixes #27880, #28023, #28123 (recurrent-state rollback), #27941 (seq_cp, block keying -- closes #27994, the unified-KV "amnesia"), #28068 (GDN norm), #28896, #28901 (hc ops), **#28770 `3cf03257f` "CUDA: enable sparse fa for qwen4"** (Strata's own pin). No qwen4exp change on master since. | `git log -- src/models/qwen4exp.cpp` |
| The n-gram table on disk | `-lm mmap --lazy-mode on` (#27794 TENSOR_READ_LAZY, #27837, #27969 renamed to `--lazy-mode`/`-lzm`); `auto` = on for tensors > 4 GiB. ISTA's card recommends exactly this. | upstream `common/arg.cpp:2688-2717` |
| The quant types (GSQ-RCO) | Standard GGUF types, "run unmodified in llama.cpp" (ISTA card). IQ2_XS shard 1: BF16 484, F32 292, IQ4_XS 181, **IQ2_S 68**, Q2_0 53, **IQ3_S 44**, IQ4_NL 36, Q6_K 29, IQ2_XXS 22, Q8_0 7, **IQ1_M 6**, F16 1. Q2_0 is upstream's own (#24448 CPU, #25707 CUDA). | read on the file |
| MTP | **Not on master.** PR #28243 (open, `6fcaa16f`, "1.3 to 2x faster") on top of #27836; draft-only GGUF via its converter (`--mtp --mtp-shared-embd`). | PR |
| Expert cache (Strata's VRAM cache of hot experts) | Not upstream: #27861 (draft, `--moe-expert-cache`, +31% decode on 2x3090, n=1) and #28414 (draft prefetch). We use the static split `--n-cpu-moe N` instead. | PRs |

**Built (CPU only), `scripts/build_engine.py`, pinned in `engines/manifest.yaml`:**

| engine | base | patches | `llama-server.exe` sha256 | build |
|---|---|---|---|---|
| `llama-upstream` | `4da6337767f973e2b4d0797e5b323d77d8565e4a` (master 2026-09-27, build 11223) | `0001-reasoning-budget-nudge.patch` (= llama-bonsai2's 0001, same bytes; applies with offsets) | `5aba8a56...8e9c` | `C:/Users/jwals/engines/llama-upstream-1b446647` |
| `llama-upstream-mtp` | same | 0001 + `0002-qwen4exp-mtp-pr28243.patch` (PR #28243's net diff) | `b6806ed2...e6ee` | `C:/Users/jwals/engines/llama-upstream-mtp-3534183c` |

Both: sm_86 + sm_120, MSVC 14.36 / CUDA 12.8.93 (`msvc-cuda128`), no web UI, OpenSSL off, targets `llama-server`,
`llama-bench`, `llama-quantize`, `test-backend-ops`; `test-reasoning-budget` + `test-chat` 2/2 in the static CPU tree
(the nudge on the new base). `--version` (no GPU visible): `0.5.0-dev (build 11223, commit 4da633776)`.
`mcp/test_engines.py` 173/173. MTP is a separate engine because #28243 rewrites the TRUNK's graph file
(`qwen4exp.cpp` +290/-48): the no-MTP gate runs untouched by it.

**Not taken, and why:** #27902 (Blackwell IQ packed loads; open): its failures are on CUDA 13.2; the issue it fixes
(#27763) and #21289 were closed as CUDA 13.2 toolkit faults (13.0 and 13.3 coherent). We build with 12.8. The file
carries exactly the affected types, so PHASE 2 step 1 is `test-backend-ops` on them; a failure brings the patch in (it
applies cleanly). #29166 (QSA per-block bias with several sequences in a unified cache; draft, n=4): avoided by
configuration or taken, decided by PHASE 2's concurrency check (section 4). #29030 (lazy rows by direct reads, +65-121%
prefill on Strix Halo; open): a candidate if the PLE reads dominate prefill here.

**Open upstream hazards that touch our design** (from the tracker, 2026-09-27): #28734 decode slows with depth on CUDA
(open; a third party's IQ3_XXS on one A100 slice: 46.5 tok/s at 32k, 29.3 at 128k, 21.1 at 224k); #28286 MTP with
`--parallel > 1` mixes content across slots; #28019 multi-seq split replay of the recurrent state; #28497 the QSA
indexer's top-k picks a different cell set run to run on CUDA (ties); #27840 `mmap`/lazy reads cause constant disk
WRITES on Windows; #28194 `/slots` restore gives no KV reuse on hybrid models.

## 2. The files (downloaded on the operator's yes; `models/manifest.yaml` `flash-next-*`)

Ownership checked on the HF API: `Qwen/Qwen3.8-Flash-Next` (author Qwen), `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`
(author ISTA-DASLab, commits by helcig / anm2211). Each GGUF equals HF's LFS sha256 AND the pinned resolve URL's
`X-Linked-ETag`. Disk: 843 GB free before, ~82 GB used.

| file | size | sha256 | from |
|---|---:|---|---|
| `flash-next/IQ2_XS/...-IQ2_XS-00001-of-00002.gguf` | 39,225,954,592 | `92cee27a...49d7` | ISTA @ `2a55d759` |
| `flash-next/IQ2_XS/...-IQ2_XS-00002-of-00002.gguf` (n-gram, shared by all four quants) | 28,800,138,432 | `316b46f3...161e113` | ISTA @ `2a55d759` |
| `flash-next/mmproj-Qwen3.8-Flash-Next-BF16.gguf` | 907,543,008 | `b1a82259...49bd0` | ISTA @ `2a55d759` |
| `flash-next/mtp-bf16/model-mtp-bf16.safetensors` (31 `mtp.*` tensors, range-fetched from 28 of 131 shards) | 5,214,305,424 | `f346d471...a8212` | Qwen @ `de4b8e4d`, `scripts/fetch_hf_tensors.py` |
| `flash-next/mtp-Qwen3.8-Flash-Next-BF16-shared-embd.gguf` (blk.48, 32 tensors) | 5,227,963,360 | `37f3c110...8fffe` | PR #28243's converter |
| `flash-next/mtp-Qwen3.8-Flash-Next-Q8_0-shared-embd.gguf` | 2,786,568,160 | `79abc660...3149c` | `llama-quantize` Q8_0, re-run byte-identical |

The MTP graft, our way: Strata's `tools/mtp_fetch.py` idea (read each shard's header, then only the `mtp.*` byte ranges)
reimplemented as `scripts/fetch_hf_tensors.py` (Strata has no licence; nothing copied), writing ONE safetensors file
that upstream's converter reads; then the PR's `convert_hf_to_gguf.py --mtp --mtp-shared-embd` (the draft borrows the
target's embeddings and LM head). A range fetch cannot be checked against a shard's whole-file sha256; every response
named the pinned commit and each payload's sha256 is recorded (`mtp-bf16/fetch-manifest.json`).

**Licence: Qwen Community License 1.0** (all files; ISTA's YAML says apache-2.0, its own text says the weights inherit
the base licence). Condition 2, verbatim: "If the licensee or any of its affiliates conducts a Model as a Service or AI
Work Assistant business, the licensee shall obtain a separate license from Qwen before Using the Software or its
derivative works for any commercial purpose. The foregoing requirement shall not apply to the licensee's internal Use of
the Software, provided that such Use does not make the Software, its outputs, or its underlying model capabilities
available to any third party." "AI Work Assistant" names AI-assisted coding products. Internal use is fine; offering
Yamadori to third parties commercially needs Qwen's licence (Bonsai's base is Apache-2.0).

**Facts read from the files:** the ISTA GGUF's embedded chat template is byte-identical to Strata's
`serve/chat_template.jinja` (sha256 `12827f24...`): Unsloth's fix of Qwen's (merges leading system/developer messages,
accepts `high` as `xhigh`, tool-call arguments must be a mapping; diff in the scratch `tpl.diff`, summarised in
section 3). **The tokenizer is identical to Bonsai's** (tokens, merges and token types hash-equal; pre `qwen35`; eos
248046, bos/pad 248044): the decider's label tokens are the same ids on both models. Header sampling: temp 1.0, top_p
0.95, top_k 20. Context 262,144. Experts 35,454,976,000 B of shard 1; everything else 3,759,953,920 B.

## 3. How Strata steers the model, against our max tier

Strata @ `b742ff9` steers very little: the GGUF's embedded template, no system text, no prefill, no thinking budget,
effort defaulting to the template's `xhigh` (`serve/frontend.py:61-77`), past `reasoning_content` relayed. Its API
default is GREEDY (no `sampling` block written by setup, `setup.py:965-967`; greedy until `012992f`). Its measured work
is speed and KV precision, not output quality.

| aspect | Strata | Qwen card | ours at max | verdict |
|---|---|---|---|---|
| template | the GGUF's (`serve/server.py:1278-1280`) | Qwen's raw | would be the GGUF's (`--jinja`) | **adopt**: serve the embedded template, never Qwen's raw one via `--chat-template-file` (it raises on `developer`, on `high`, on no user query) |
| effort | `xhigh` default; high/max -> xhigh | xhigh default | `xhigh` at max (`tiers.py:213-219`) | same already |
| thinking budget | none; `max_new` = the rest of the context | reasoning up to 262,144, answer up to 131,072 | user turn 20,480, agent step 6,144, jobs 3,072-12,288, nudge at 0.6, breaker 32,768 -- all derived on **Bonsai** | **conflict; operator decision** (section 6, Q6) |
| sampling | greedy unless the client sends values | 1.0 / 0.95 / 20 / min_p 0 / presence 0 / rep 1.0 | the same values, enforced (`VENDOR_SAMPLING`) | keep ours (= the card; Strata's greedy has no evidence) |
| KV | int8 above 8K; q4_0 optional | no YaRN at <= 262,144 | q8_0 | **adopt** q8_0 (their bench: q4_0 perplexity +8% at 1K, +12% at 8K; int8 ~fp16) |
| MTP | `--spec 4 --spec-min-p 0.5` + prompt lookup | one MTP layer | none yet | **adopt as the MTP arm's starting point**: `--spec-draft-n-max 3 --spec-draft-p-min 0.5` (speed only; one prompt per length, n=1) |
| stop | ids 248044 / 248046 | - | the GGUF's EOG | same |
| preserved thinking | relays the client's reasoning; `preserve_thinking` undefined = on | on (helps consistency and KV reuse) | `restore_reasoning` on | same |
| experimental speed projection | a refusal-direction ablation, off by default; code perplexity +15% | - | - | **not taken** |
| tool calls | its own XML parser, no grammar | - | llama-server's parser + our tool_code | nothing to take; check a tool-call render on the served template |

## 4. The swap: llama-swap entry (PROPOSED, not in config.yaml)

```yaml
macros:
  server_upstream: "C:/Users/jwals/engines/llama-upstream-1b446647/src/build/bin/llama-server.exe"

models:
  "flash-next":
    env:
      - "CUDA_VISIBLE_DEVICES=<the 5060 Ti's UUID, as bonsai>"
    cmd: |
      ${server_upstream} --port ${PORT}
      -m ${models}/flash-next/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf
      --mmproj ${models}/flash-next/mmproj-Qwen3.8-Flash-Next-BF16.gguf
      -dev CUDA0 -ngl 999
      --n-cpu-moe <N from PHASE 2 step 0; estimate 42 of 48>
      -lm mmap --lazy-mode on            # shard 2 stays on disk (ISTA's card)
      -c 262144 --cache-type-k q8_0 --cache-type-v q8_0 -fa on
      <-np 4 unified | --no-kv-unified: PHASE 2 step 3 decides>
      -b 2048 -ub <512 | 2048 | 4096: PHASE 2 sweep>   -t <8 | 16: sweep>
      --jinja --reasoning-format deepseek --no-context-shift --no-cache-idle-slots
      --reasoning-budget 32768 --reasoning-budget-message "<bonsai's, unchanged>"
      --cache-ram <sized to the RAM left; default 8192 MiB is too much here>
      ${sampling}
    ttl: 0          # the proxy decides when it goes (section 5)
groups:
  primary:          # 'canopy' in the naming table; the rename stays the operator's
    persistent: true
    swap: true      # was false: bonsai and flash-next now swap each other INSIDE the group
    exclusive: false
    members: [bonsai, flash-next]
```

MTP arm: `${server_upstream_mtp}` + `--spec-type draft-mtp --spec-draft-model .../mtp-Qwen3.8-Flash-Next-Q8_0-shared-embd.gguf
-ngld 999 --spec-draft-n-cpu-moe 1 --spec-draft-n-max 3 --spec-draft-p-min 0.5` (draft experts on CPU like the trunk's).
Whether swap-inside-a-persistent-group behaves this way in llama-swap v256 is read from its docs, not verified: PHASE 2
checks it with `/running` before anything else.

**VRAM on the 5060 Ti (16,311 MiB). Arithmetic from the file and config.json, not a measurement:**

| item | MiB | basis |
|---|---:|---|
| non-expert weights (attention, DeltaNet, router, shared experts, hc, head) | 3,586 | 3,759,953,920 B, read from shard 1 |
| mmproj + its compute buffer | 865 + ~1,000 | file; the buffer is Strata's ~1.2 GB at 1,024 image tokens (their note), unmeasured here |
| K/V q8_0, 262,144 cells | 3,264 | 12 attention layers x 2 KV heads x 256 x (K+V) = 12,288 values x 1.0625 B |
| QSA indexer keys, recurrent state (4 slots), compute buffers, CUDA context | ~2,500 | estimates; STEP 0 measures |
| demotion margin | 1,000 | bonsai-ada-surgery's soak (docs/ENGINES.md), headless card |
| **left for experts** | **~4,100** | one layer's experts = 738.6 MB (35.45 GB / 48): **~5-6 layers -> `--n-cpu-moe` ~42** |

**RAM (63.7 GB).** Read 2026-09-27 23:20 with Bonsai loaded: 27.3 GB available, commit 87.8 of 103.7 GB; Bonsai's
llama-server holds 26.6 GB private (pinned KV tail, prompt cache), sd-server 10.4 GB, embeddings + reranker 6.7 GB,
WSL 4.7 GB. Flash-Next needs ~31 GB of CPU-side experts at `--n-cpu-moe 42` (file-backed pages under mmap), plus host
buffers, the prompt cache (`--cache-ram`) and context checkpoints (~118 MB each per Strata PR #8). Bonsai's 26.6 GB goes
when it unloads. **Tight but plausible; the risk is silent eviction of expert pages under mmap** (re-read from NVMe,
decode collapses, no error). Mitigations to test: a smaller `--cache-ram`, `--ctx-checkpoints`, imagegen's ttl; an
`mmap+mlock` variant only if lazy tensors are excluded from the lock. The PLE table (28.8 GB) lives in the page cache
and pays NVMe reads when evicted (#27840: watch disk writes).

**Load time:** not measured. Bounded below by reading ~39 GB (plus the lazily paged table) from NVMe; Bonsai's reload
was 4.3 s on the old engine (T0). Swap cost both ways is PHASE 2 step 6. Until measured, the streamed path must send
heartbeats while the swap runs (Hermes' stale detector 900 s, OpenCode's chunkTimeout 300 s, Pi's idle 300 s).

**Speed expectation (not a prediction):** Strata's IQ2_XS on a 5070 12 GB + AVX-512 Ryzen: 82 tok/s at 1K, 48 at
262K, prompt 332-495 tok/s on ITS engine. ISTA's llama.cpp IQ2_XS: prompt 108, decode 70 tok/s (hardware not stated);
Q2_0 3.4x the prompt rate. Strata issue #6 (13th-gen i7, 64 GB): 9-10 tok/s default, 35 after tuning. Our CPU is AVX2
only. **Prefill is the agentic risk**: with experts in RAM, llama.cpp copies them over PCIe (5.0 x8) per ubatch, so
`-ub` is the lever, and Q2_0 (shard 1 only, 37.6 GB; shard 2 is the same file) is the speed arm if IQ2_XS prefill is
too slow for agent steps.

## 5. The proxy side (BUILT 2026-09-28, offline; OFF until the deploy sets YAMADORI_MAX_MODEL)

**What was built** (`mcp/max_mode.py`, and small hooks; `mcp/test_max_mode.py` 57/57):

| rule | where |
|---|---|
| tier max -> `flash-next`, else `bonsai`; a side call while flash-next is loaded -> flash-next, never refused | `max_mode.decide`, called in `server._serve_turn` before admission; `prepare` routes on `body["_upstream_model"]` |
| every internal call follows the conversation's model | the model is bound to the request's cancel token (`max_mode.set_current` in `proxy._run_turn`; job threads re-bind the token); `model.shape` defaults to it (the second brain, summaries, the skill fallback), `shomen._model()`, `vision._vision_model()` and `main_sees`, `decider_bonsai._upstream` / `ask_one` |
| 503 `model_at_capacity` + Retry-After, in `x_yamadori.capacity` | `api_errors.at_capacity`; refused in `_serve_turn` (blocking and streamed, before any byte); the one door's backstop (`max_mode.ModelAtCapacity`) maps to the same error. Retry-After: the rest of the operator's idle period when known, else 30 s (api_errors' existing 503 value: the end of max work in flight cannot be known) |
| a swap-in waits for bonsai's work, never cancels | `max_mode.Lease` (in-flight counts; a max lease while bonsai works marks the switch, and new bonsai work is refused), `max_mode.wait_ready` in `_run_turn` (honours the client's hang-up). Before the first byte on a stream: no heartbeat during the wait yet (see open items) |
| the return to bonsai | the next non-max request once no max request is in flight; `YAMADORI_MAX_IDLE_S` (unset: the operator has not given it) adds an idle period, with Retry-After = its remainder |
| nothing reloads bonsai while flash-next serves | `max_mode.blocks` / `guard` (llama-swap `/running`, which never loads, plus the proxy's state file for other processes): `model.post`, `model.props`, `model.release_slot`, `proxy._upstream_json` (tokenize, apply-template, warms), `proxy._warm_now`, `tiers.accepted_efforts` (per model), `vision.main_sees`, the decider prime's `_idle`, `idle.stack_idle` (checked first), the worker (a `ModelAtCapacity` defers the job without an attempt), `code_search.summarize_text` (`MODEL_AT_CAPACITY`, retryable), `skill_match --check-labels --model`, `skill_questions_bank.slots_busy`, `deploy_check.slots_busy` (asks `/running` first) |
| the decider per model | label spellings (`spelling_ids`, each model's own `/tokenize`: the per-model label check) and the label priors (`TEMPLATE_CF`) keyed by model; `decide_turn.prime_for(flash-next)` re-primes on the first max request |
| thinking at max follows the card | `tiers.budget`: at max no `USER_TURN_THINKING` / `AGENT_STEP_THINKING` / `JOB_THINKING` / `HELPER_THINKING` -- the whole room is the budget (Qwen card @ `de4b8e4d`, README "Best Practices" 2: reasoning up to 262,144 tokens, answer 131,072); a benchmark's `reasoning_cap` still wins; bonsai unchanged |
| the llama-swap entry, prepared, not enabled | `config.flash-next.fragment.yaml` (llama-upstream-moe, its mmproj on the 5060 Ti, the gate's values as placeholders) |
| the deploy | `bench/deploy_flash_next.py --gate DIR [--preview / --dry]` (in deploy_kv_rank's style; `--selftest` 15/15): refuses unless the gate passed; adds the entry, `primary` swap:true with both members, `YAMADORI_MAX_MODEL` for proxy/tools API/worker, manifests; restarts through the Scheduled Task |

**Open items** (not built): heartbeats while a streamed max request waits for bonsai to drain and flash-next to load
(before the first byte the client sees nothing; Hermes' stale detector is 900 s, OpenCode's and Pi's 300 s); the pool
and window for flash-next (`budget.pool_size` reads bonsai's `/props` once: both run `-c 262144`, and bonsai's main
cap is the smaller, so flash-next is advertised less than it holds -- re-derived per model after the gate); per-model
slot state (a swap leaves bonsai's pins; the new server simply starts cold); a live group in `mcp/test_live_stack.py`.
`gpu_room` needs no change: it acts only on A4000 models in `SIZES`, and flash-next is not one.

The routing rule (operator): tier `max` = `flash-next`; every call of a max conversation goes to it -- main, the
second brain (plan, investigate, fixup, alternative, tiebreak), summaries, compaction, its side calls, the decider's
questions. Nothing may make llama-swap load Bonsai while Flash-Next serves.

**Where the model is chosen today** (read 2026-09-27; every row is a change): `catalog.py:61-103` maps every public
name to `"bonsai"` and `:314-318` `_is_chat == "bonsai"`; `proxy.py:3116` sets `out["model"]` from the catalog;
`model.py:54-60` `MODEL`/`VISION_MODEL` = `bonsai`; `shomen.py:154` `MODEL`; `vision.py:1152`; `decider_bonsai.py:276-288,
364` `/upstream/{model.MODEL}`; `skill_select.py:848-855` fallback ask; `tiers.py:328-346` `accepted_efforts` cached
once per process; `budget.py:103, 128-166` one pool from :10001; `slots.py` global slot state; `gpu_room.py:121` NEVER
= {bonsai, bonsai-agent}; `compaction.py:98-123` stores no model.

**Things that would LOAD Bonsai under Flash-Next today** (any `/upstream/<model>/...` loads it): `vision.main_sees` ->
`model.props` on every request (300 s cache, `proxy.py:2586-2599`); `tiers.accepted_efforts` on first use; the
decider prime at startup (`decide_turn.py:105-204`, reads `/upstream/bonsai/slots`); `proxy._warm_now` for a Bonsai
conversation; `code_search.summarize_text`; worker extract/assist and the skill pipeline; `idle.stack_idle` counts
"not loaded" as idle (`idle.py:111-112`); `skill_match --check-labels` (hard-coded bonsai URL);
`scripts/deploy_check.py`.

**Design:**
1. **One table: tier -> upstream model** in `tiers.py` (`max` -> `flash-next`, the rest -> `YAMADORI_MODEL`), plus a
   per-model profile (sampling, template fixture for `accepted_efforts`, pool/slot counts, concept-seed matrix,
   decider constants). `prepare` sets `out["model"]` from the REQUESTED tier; the catalog keeps names and tier hints.
2. **Internal callers inherit the conversation's model**: `server._serve_turn` decides it before admission and binds it
   to the request's cancel token (the registry `slots.bind_request` already re-binds on job threads); `model.shape/post`,
   `shomen`, `vision`, the skill fallback and the decider read it. `compaction.record` stores the model. Callers with
   no conversation (worker, tools API, prime) never load a model that is not serving: they defer (worker `Deferred`)
   or return a failure with the next step (`MODEL_AT_CAPACITY`: retryable, when, remedy).
3. **Decider per model**: `Turn(model=)`; spell ids cached per (model, label) -- identical tokenizers make the check
   pass, and it stays cheap; `TEMPLATE_CF` priors keyed by model and primed per serving model after each swap;
   `skill_match --check-labels --model`. `TIE_BAND` (0.0034) and `FORM` were measured on Bonsai: whether the decider
   runs at max before Flash-Next is measured is the operator's (Q5).
4. **Who knows what is loaded**: a proxy-owned state file under `gpu_room`'s lock ({serving, switching_to,
   last_max_at, leases}), recovered from llama-swap's `/running` (which never loads). One door: `model.post`,
   `proxy._upstream_json` and `_post_events_raw` take a lease for their model and refuse a model that is not serving.
5. **At capacity** (operator: "we reject it"): while Flash-Next is serving or being switched to, a non-max request is
   refused at once in `server._serve_turn`, before admission, through `api_errors`: HTTP **503**,
   `type: service_unavailable_error`, `code: model_at_capacity`, `Retry-After` computed from `last_max_at` + the idle
   threshold, message "The stack is serving max mode (Flash-Next); non-max requests are refused until it frees, about
   N s." Recorded in `x_yamadori.capacity` and one log line. On `/v1/responses` it maps to `server_is_overloaded`
   (`responses_api.py:929-947`). **503 vs 429:** the repo's evidence -- Hermes honours `Retry-After` on 429 with 3
   retries, then exits (SELF-IMPROVEMENT-LOG #44); Pi retries 5xx 3 times at 2/4/8 s (HARNESS-PI.md); Codex retries
   both `rate_limit_exceeded` and `server_is_overloaded`; OpenCode: no evidence; OpenAI SDKs retry both, honouring
   Retry-After. 503 says what it is (the service, not the caller's rate); our benchmarks treat 429 as NOT RUN and 503
   as a failure, so the grader must learn the new code either way. Recommendation 503; the operator picks (Q1).
6. **Swap in** (a max request while Bonsai serves): set `switching_to` (new Bonsai work is refused from then on), wait
   for Bonsai's leases to drain (in-flight turns, helper runs, warms), then `GET /upstream/flash-next/health` loads it
   and the group swap unloads Bonsai; Bonsai's slot namespace is invalidated. Heartbeats on the streamed path meanwhile.
7. **Return to Bonsai**: after max has been idle for a threshold, on the next non-max request (or proactively). The
   threshold is **the operator's** (Q3); `slots.IDLE_CLEAR_S` (600 s) is the nearest measured-in-context constant
   and is not proposed as the value.
8. **Helper slot at max**: the second brain runs on Flash-Next too, so Flash-Next needs >= 3 slots (`helper_slot` =
   n-2) -- another reason step 3 must make 4 slots correct. `budget.HELPER_TOKENS` (49,152) is an operator choice
   for Bonsai's pool; Flash-Next's pool is cheaper per token (12.75 KiB vs Bonsai's 34 KiB target K/V), so the same
   VRAM holds more -- re-derived from its `/props`, not assumed.
9. **gpu_room**: `NEVER` gains `flash-next` (the main card, never evicted by gpu_room); `bonsai-vision`'s SIZES row and
   the `ondemand` group go when it retires; no A4000 row for Flash-Next (it never loads there).

**What our features need from the engine** (source-level, upstream `4da63377`): `/slots` (+ `?action=erase`, 501
without `--slot-save-path` -> our one-token fallback), `/apply-template`, `/tokenize` (`with_pieces`), `/props`
(`chat_template`), `/completion` with `id_slot`, chat `logprobs`/`top_logprobs`, `timings.cache_n`/`prompt_n`,
`reasoning_budget_tokens` + `reasoning_budget_message` (upstream, `server-schema.cpp:387-430`), the nudge fields (our
0001, applied), `--cache-ram` and `--no-cache-idle-slots` (present), `--mmproj` with `image_url` parts: **all present
in the source; each is a PHASE 2 live check**, and #28194 (no KV reuse on `/slots` restore for hybrid models) says the
idle-clear `restored` path may re-prefill here.

### 5b. One model per effort tier (2026-09-29; BUILT offline, NOT deployed)

The operator: "we should make mirai xhigh flash-next max, and bonsai everything else ... capability is now the model
swap." Max mode became one row of a table.

| what | where |
|---|---|
| the table, and each model's ONE token profile (effort, thinking cap per route, nudge, answer allowance, turn floor, force-close, sampling; every value with its class and source) | `mcp/tier_models.yaml` (data), `mcp/tier_models.py` (loader); read only when `YAMADORI_TIER_MODELS` names it (else the 2026-09-28 `YAMADORI_MAX_MODEL` = a one-row table; neither = one model) |
| routing, swaps, 503 | `mcp/max_mode.py` (name kept): tiers are ordered; a higher tier's model waits for a lower one's work and takes the card, a lower tier is refused 503 `model_at_capacity` while a higher one holds it (the operator's max rule, generalised; confirmed 2026-09-29: "swap order is fine"); a waiter overtaken before it starts is refused; side calls go to the model on the card. The swap is an explicit, timed load, then `/running` must show no other main model (`x_yamadori.capacity.swap.left_loaded`). Only the proxy (`enable_swaps`, server.py) or `YAMADORI_ALLOW_SWAP=1` may load; never under the offline guard |
| no borrowed caps (operator 2026-09-29: "they need to be based on what makes them perform the best and how much memory we have") | mirai-s and flash-next think within their own window only (thinking = main_cap - prompt - answer) until `bench/tier_effort.py`, run uncapped, measures the curve; bonsai stays on bonsai-ada-surgery's tune |
| the profile applied | `tiers.apply` / `tiers.budget` / `tiers.rebudget` (`resolve_profile`); a client's `max_tokens` no longer sizes its turn; `x_yamadori.sampling.profile` records the profile and what the client asked |
| the window per model | `budget.budgets(model=)` (a table row's `window`), `catalog.tier_window`, the card's `x_yamadori.reasoning_effort.tokens_by_value` |
| vision | a table row's `vision: false` sends that model's images to `bonsai-vision` |
| deploy | `bench/deploy_tier_models.py --gate DIR` (Mirai S's `bench/mirai_s_gate.py`, `--no-mmproj` by default, after layout v2) |
| measure | `bench/tier_swap_probe.py` (clean swaps both ways, VRAM per landing), `bench/tier_effort.py` (best single effort per model, n=6 paired tasks), live group `tier_models` |
| tests | `mcp/test_max_mode.py` (93 checks) |

## 6. PHASE 2 (on "GPU free"; one GPU consumer; production Bonsai unloaded through engine_corruption.py's window)

0. **Fit**: launch at `--n-cpu-moe 44`, warm, read lowest free VRAM and RAM, step N down with the 1,000 MiB margin;
   record VRAM, RAM (private, file-backed, commit), load time.
1. **Kernels**: `test-backend-ops -o MUL_MAT` and `-o MUL_MAT_ID` for IQ2_S, IQ3_S, IQ1_M, IQ2_XXS, IQ4_XS, IQ4_NL,
   Q2_0, Q6_K, Q8_0 on CUDA0 (sm_120) and CUDA1 (sm_86). Any failure: take #27902 and rebuild.
2. **Corruption**: greedy rep-to-rep identity (3 prompts x 3), 16 sampled runs for symptoms, 10 tool requests; a
   tool-call turn, a `developer` message and effort `high` rendered through `/apply-template`; both EOG ids.
3. **Concurrency**: the #27994/#29166 shape -- two slots decoding different long conversations at once, each asked
   about its own earlier content -- on `-np 4` unified; if it fails, `--no-kv-unified` (or #29166) and re-run.
4. **Speed** (llama-server timings, n=3): decode and prefill at 4K/32K/64K/128K; `-ub` 512/2048/4096; `-t` 8/16;
   IQ2_XS vs Q2_0 only if prefill is the blocker (needs the Q2_0 shard-1 download, 37.6 GB, the operator's yes).
5. **MTP arm** (`llama-upstream-mtp`): acceptance and tok/s with 1 slot and with 4 (the #28286 check).
6. **Swap cost**: Bonsai -> Flash-Next -> Bonsai, wall time and first-token time, n=3.
7. **Proxy**: build section 5 behind the max tier with offline tests (`test_tiers`, `test_catalog`, `test_budget`,
   `test_slots`, `test_gpu_room`, `test_openai_conformance`, `test_responses_api`, `test_stream`, `test_model`,
   `test_decider_bonsai`, `test_decide_turn`, `test_utility`, `test_harness_decisions` (a `model` column),
   `test_ledger`, the worker/idle tests), a new live group in `mcp/test_live_stack.py`, then `deploy_check.py`.
8. **Then** the operator's comparison: a pagoda run at xhigh (Bonsai) and at max (Flash-Next).

## 7. Open questions for the operator

1. 503 (recommended) or 429 for "model at capacity"; should benchmarks count it NOT RUN?
2. Side calls that carry no effort (Hermes' title/classifier/flattened compaction; every OpenCode request): while
   Flash-Next serves, a max conversation's side calls go to Flash-Next -- but a side call names no conversation. Serve
   every side call on Flash-Next while it is loaded, or refuse them?
3. The idle threshold before returning to Bonsai, and whether to preload Bonsai then or wait for the next request.
4. How long a swap-in waits for Bonsai's in-flight work (a 40-minute generation is on record, #44), and whether a
   long Bonsai generation is cancelled instead.
5. Whether the Bonsai decider runs at max before its constants are measured on Flash-Next.
6. The thinking caps at max: keep Bonsai's (20,480 / 6,144 / jobs, nudge 0.6) or follow the card (no cap below the
   window's room). No number is proposed here.
7. A Bonsai conversation mid-run is refused while Flash-Next is loaded, and Hermes exits after 3 retries: accepted?
8. The Q2_0 shard 1 (37.6 GB) as a prefill-speed arm.

## 8. Why 18 tok/s (2026-09-29; the evidence, cause by cause)

The deployed max arm (`llama-upstream` 1b446647, IQ2_XS, `--n-cpu-moe 42`, `-ub 512`, 16 threads, no MTP) decodes at
17.0-18.4 tok/s (55-58 ms/token) at 1-9K and 14.7-15.0 at 36K, and reads prompts at 104-196 tok/s
(`C:/Users/jwals/octo/flashnext-gate-20260928b/base.server.log`, llama-server's own `print_timing`, the gate's
corrupt/exact/needle requests, n=5-9 per shape). The "17-46 tok/s prompt" figures in gate.json are 4-token
continuations and 1K chat prompts whose timings are dominated by fixed costs; they are not the prompt rate.

**Measured offline on the CPU alone** (`C:/Users/jwals/engines/dev-flash/tools/moe_cpu_bench.cpp`, built against
the patched ggml; the CPU half of one host-resident MoE layer at Flash-Next's exact shapes -- n_embd 2560,
n_ff_exp 640, 512 experts, 10 used, gate/up IQ2_S, down Q2_0 -- over 6 rotating layers of random weights (4.6 GB,
so reads come from DRAM), median of 300 steps, the live stack running beside it, so each number is an upper bound):

| case | ms per layer | x 42 CPU layers |
|---|---:|---:|
| 1 token, `-t 8` | 1.26 | 53 |
| 1 token, `-t 16` (the deployed default) | 0.83-0.91 | 35-38 |
| 1 token, `-t 24` | 1.07 | 45 |
| 1 token, `-t 16` pinned to the P-cores (`0xFFFF`, strict) | 0.80 | 34 |
| 1 token, `-t 8`, one thread per P-core (`0x5555`) | 0.84 | 35 |
| gate + up only (IQ2_S, 2 of the 3 matrices), `-t 16` | 0.30 | 13 |
| down only (Q2_0, 1 of 3), `-t 16` | 0.57 | 24 |
| 2 / 3 / 4 tokens (an MTP verify window, independent routing), `-t 16` | 1.63 / 2.42 / 3.20 | 68 / 102 / 134 |

Causes, in order of size:

1. **The CPU computes the routed experts of 42 of 48 layers, and that is ~35-38 ms of the 55 ms.** It is COMPUTE
   bound, not bandwidth bound: one layer reads 15.1 MB of experts (10 x 1.51 MB), 0.91 ms is 16.6 GB/s against
   DDR5-4800 dual channel's 76.8 GB/s peak (Strata `df6980d` measures ~30 GB/s as the practical pool rate).
2. **Q2_0 has no x86 SIMD kernel in ggml.** `ggml/src/ggml-cpu/arch-fallback.h` (x86 block) maps
   `ggml_vec_dot_q2_0_q8_0` to the scalar generic; every GSQ-RCO file keeps its down experts in Q2_0 (48/48 layers
   in IQ2_XS, all three matrices in the Q2_0 file). The down projection -- a third of the weights -- is two thirds
   of the CPU time (0.57 of 0.87 ms). Strata says the same (`src/kernels/cpu/pool.cpp:383`: "ggml-cpu has only a
   scalar one on x86") and ships its own. **Fix: an AVX2 Q2_0 dot.** Estimate, not yet measured: ~20 ms/token.
3. **Only 6 of 48 layers' experts are in VRAM, chosen by layer, not by use** (`--n-cpu-moe 42`): 12.5% of routed
   reads hit the card. Strata's VRAM cache holds ~4,000 experts ranked by an expert profile and hits 70-92% on a
   16 GB card (their `2026-09-29-layer-split`). Our port (0003 + 0004) crashed before it ran: 0004 extended the
   cache chain to 2-31 token batches but kept 0003's one-token `ggml_get_rows(table, ids)`, which asserts
   `a->ne[2] == b->ne[1]` (ggml.c:3973) on the `-np` reserve graph. **Fix: look the ids up as one flat row**
   (`src/llama-graph.cpp`, built in the dev tree).
4. **No MTP.** Strata's output numbers are with MTP at 2.7-3.2 tokens per verify round. With the stock CPU
   kernels a 4-token window costs 3.2 ms/layer (134 ms/round), so MTP alone would LOSE here. It pays only once the
   cache takes most routed reads and the CPU kernels amortise the weight decode across the window's tokens
   (Strata `ed14227`: the multi-token AVX2 kernels, their 5700X3D + 5060 Ti).
5. **Threads**: `-t 16` unpinned is within 5% of the best placement measured (0.80 ms, P-cores only). A small
   lever, measured again on the card.
6. **Prompt reading is PCIe-bound at `-ub 512`.** For a batch of >= 32 tokens ggml-cuda copies each CPU layer's
   experts to the card (738 MB a layer, ~31 GB per ubatch at 42 layers). 512 tokens take ~2.8 s = ~11 GB/s,
   the pageable-copy rate under mmap (the loader warns: "tensor overrides to CPU are used with mmap enabled").
   Strata measured 14.1 GB/s pinned H2D on a 5060 Ti at x8 (`7a4b627`). Levers: a larger `-ub` (fewer copies per
   token), pinned experts (0005), the prefetch ring (0002), and experts that live in VRAM (3).
7. Not yet attributed (needs the card): the GPU half and the ~84 GPU<->CPU hand-offs per token under WDDM (55 ms
   minus the CPU's ~36 ms leaves ~19 ms for ~3.6 GB of dense weights at 448 GB/s, ~8 ms, plus synchronisation).

**Built for the gate (2026-09-29): `llama-upstream-flash`** (docs/ENGINES.md): the fix for 2 (0008, the AVX2 Q2_0
dot: the CPU layer 0.87 -> 0.42-0.48 ms offline), 3 (0007, the crash; the cache over all 48 layers sized by the
`fit2` step), 4 (0006 MTP + 0009's multi-token rows), 5 (the `t16p` arm), 6 (`-ub 4096` + pinned experts), and the
Q2_0 file (37.6 GB, ISTA @ `ed59f920`, sha256 checked) as the all-Q2_0 arm. Gate arms: `base` (speed, the "before"),
`t16p`, `flash-cpu`, `flash-ub4k`, `flash-cache`, `flash-mtp`, `flash-all`, `flash-all-mmcpu`, `flash-all-q2`
(`bench/flashnext_gate.py --dry-run`). Output: `C:/Users/jwals/octo/flashnext-gate-20260929` (seeded with the
2026-09-28 base logits and its KL yardstick: mean KLD 0.010993, same top-1 96.757%, batch 8 vs 16 on the base
engine, 8 x 4,096 tokens).

**Operator decision, 2026-09-29, on `flash-cpu`'s KL REPORT** (the AVX2 Q2_0 kernel alone; mean KLD 0.010793 vs the
yardstick's 0.010993, same top-1 96.464% vs 96.757%, 8 x 4,096 tokens, llama-perplexity at batch 16): "Acceptable,
continue" -- the drift is accepted (recorded in `flashnext-gate-20260929/operator-decisions.json` and gate.json).

**OPEN ENGINE BUG (2026-09-29): the GPU expert cache faults on CUDA with multi-token batches.** `llama-upstream-flash`
with `--moe-expert-cache N` (0003 #27861 + 0004 + 0007's flat-row slot lookup), the profile policy, pinned experts:
- `llama-perplexity` at `-b/-ub 16` (the KL step): `CUDA error: an illegal memory access was encountered`, surfacing in
  `launch_mul_mat_q` (`mmq.cuh:1417`, `cudaFuncSetAttribute`) on the first chunk
  (`C:/Users/jwals/octo/flashnext-gate-20260929/kl-flash-cache-b16.log`);
- `llama-server` (the speed step): the same error at the tail of a ~142K-token prompt, in
  `ggml_backend_cuda_synchronize` (`flash-cache.server.log`), after 16 clean runs (single-token decode and 512-token
  batches).
Suspect: the cache chain's `mul_mat_id` over the slot tensors with a `[n_expert_used, n_tokens]` id table for
2-31 token batches (0004 extended #27861's one-token chain to them; 0007 fixed only the graph-build assert). Not
investigated in the GPU window (the coordinator, 2026-09-29). Before it faulted, the cache at 46 slots a layer (the
VRAM left beside MTP, ~9% of experts) decoded within ~3% of `flash-cpu` (4K 26.3 vs 25.6 tok/s; ~35K 20.8/16.9/17.1
vs 19.9/16.3/16.5; ~68K 14.4/12.3 vs 13.8/11.8; n=2-3). The combined arms place whole expert layers instead.

M0, the one bounded attempt (window B, 2026-09-30; `bench/flashnext_gate.py --arms fault-cache --steps fault`, n=1
each): with `CUDA_LAUNCH_BLOCKING=1` the fault is reported in `ggml_cuda_mul_mat_q` at `mmq.cu:291` on the first
16-token batch (`fault-blocking.log`). In the shipped source that line is the check after the `mul_mat_id` path's
src1 quantization (`quantize_scatter_mmq_q8_1_cuda` / its FP4 twin: the activations are broadcast, ne11 == 1, so
the dedup scatter runs); the ids helper before it passed its own check (line 254). So the kernel that faults reads
the activations through the inverse id map built from the cache's slot ids -- a lead, not a diagnosis (0007 fixed
the graph-build `get_rows` assert only). compute-sanitizer 2025.1 (the toolchain's
`sanitizer:` row) could not instrument it: "Failed to initialize WDDM debugger interface. Please run
EnableDebuggerInterface.bat as an administrator" (`fault-memcheck.log`); that is a system setting, for the operator.
Left open.

**What a working cache would buy (decode), simulated.** `bench/moe_hit_sim.py` over a routing trace of flash-cpu's own
decode (`GGML_MOE_LOG`, arm `probe-trace`: 512 tokens at ~4K and ~35K of the gate's corpus, 48 layers, 245,370 routed
reads, 15,279 distinct (layer, expert) pairs; n=1 text; `moe_hit_sim.txt`), the share of routed reads a VRAM cache of N
experts serves:

| N experts (1.32 MiB each: 676 MiB a layer / 512) | Strata profile, static | Strata adaptive tier (0004's rule) | LRU | best static set (oracle) |
|---:|---:|---:|---:|---:|
| 2,200 (46 a layer: the VRAM beside MTP) | 18.1% | 56.4% | 57.4% | 50.9% |
| 3,000 | 23.3% | 64.7% | 65.5% | 60.3% |
| 4,000 | 30.1% | 71.2% | 77.0% | 69.6% |
| 5,000 | 36.4% | 75.0% | 83.7% | 76.8% |
| 7,000 | 47.9% | 80.9% | 89.9% | 87.2% |

Whole static layers in the same VRAM serve 4 of 48 layers' reads (8.3%). Strata's shipped profile is a poor seed for
this text (18% at 2,200); the adaptive tier recovers to within a point of LRU. The simulation charges nothing for a
swap (the adaptive tier moves at most 96 experts per 4 tokens, ~32 MiB a token over PCIe).

**Window B (2026-09-30): what held the decode, and 0013-0016.** Every row `bench/flashnext_gate.py --steps probe`: one
256-token decode (ignore_eos) at ~4.4K and at ~35K of the gate's corpus, n=1 each -- probes, not the gate's n=3 speed
step. Layout: the operator's (-np 2, -c 265,216, one pool), MTP unless marked, P-cores, pinned experts, no projector.

| arm (engine) | experts on the card | KV | 4.4K decode | ~35K decode | lowest free MiB |
|---|---|---|---:|---:|---:|
| fit-kvmap (0001-0012) | none | host-mapped | 9.3 | 7.7 | 8,845 |
| fit-kvmap-nomtp (0001-0014) | none | host-mapped, no MTP | 4.9 | 0.9 | 10,001 |
| fit-kvmap-nomtp (0001-0016) | none | host-mapped, no MTP | 19.9 | 18.2 | 10,001 |
| probe-q8-cpu (0001-0016) | none | card | 19.3 | 26.1 | 5,023 |
| cache-fit (0001-0016, first 0016) | 63-slot adaptive cache a layer (3,024) | card | 29.8 | 35.0 | 767 |

Three engine causes, each found from these rows and fixed:
- **0015**: with a q8_0 cache, ggml-cuda sent a 1-2 query decode to the VECTOR flash-attention kernel, which reads
  every cell; only an f16 cache was tested for the sparse (top-k) kernel. On the card that is a context-linear cost;
  host-mapped, a latency-bound walk over PCIe (0.9 tok/s at 35K). Fixed: a quantized cache takes the sparse test too.
- **0014**: the sparse MMA kernel converted a q8_0 K/V to f16 WHOLE on every call (MTP's 4-query verify took it).
  The op profile (0010, `probe-opt-all`, -lv 4) put the flash-attention node at 0.087 ms a graph at ~35K before and
  0.046 after (n=1 each). Fixed: only the selected cells.
- **0013**: the expert cache's CUDA fault (M0 above). Fixed: the cache chain only on MMVQ-served batches.
With them the adaptive expert cache (0003/0004, Strata's policy) is usable again and is the first arm past 30 tok/s.
The profile lines are GPU time with CUDA graphs off and an event per node, so they rank ops, they do not add up to
a token's time. A `GET_ROWS q8_0` over every cell per QSA layer remains (the indexer pools every block's key every
token, `qwen4exp.cpp build_qsa_top_k`): Strata keeps pooled block keys; not ported (M2).

**The first 0014 lost needles, and the fix.** cache-all on that build: needles 7/15 (1K 5/5; 32K 2/5; 128K 0/5, the
misses empty: 4,096 tokens of thinking and no answer). Bisected at 32K, 5 needles an arm, env switches in one binary:
0015 off 3/5, no cache 2/5, 0014 and 0015 off (cache on) 5/5 -- 0014. The sparse MMA kernel pads a tile past a list's
count by gathering ROW 0 (fattn-mma-f16.cuh, the cp.async load) and relies on the -inf mask; 0014 left row 0 stale,
and a masked score times a non-finite row is NaN. 0014 now converts row 0 too. test-backend-ops passed the broken
version (its buffers held finite leftovers); `GGML_CUDA_FA_SPARSE_ROWS_POISON=1` (tests only) fills the f16 buffers
with NaN first: broken, 6 q8_0 sparse cases fail (369/375); fixed, all 4,004 FLASH_ATTN_EXT cases pass. New eval
cases at the model's own sizes (top-k 2,051, the permuted cache view, 1 query past 4,096 cells, 2 and 4 past
16,384). The gate's kernels step runs them poisoned.

**The fixed build's gate (cache-all on `llama-upstream-flash-74746ec1`, 0001-0016, run B12, 2026-09-30).** Kernels
pass on both cards (the poisoned sparse-FA cases 26/26 each); KL 0.011094 / 96.532% (identical to the first build:
batch 4 never reaches the sparse path); needles **15/15**, every answer the exact key; corrupt clean (0 symptom rows
of 35, tools 10/10). Decode seen in the needle runs (server log, n=1 each, not the speed step): ~35K 31-36 tok/s,
~142K 18-25 tok/s (the broken build: 9.6). The window stopped at 10:20; contract, lane and speed ran in window C
(below); the base arm's ~35K row and jjava ran 10:22-11:41 -- after the card was declared released, because the
chain's shell survived its stop (no client request reached the stack meanwhile; reported). No deploy. The arm:
`-np 2 -c 265216`, q8_0 K/V on the card, every layer's experts in RAM (pinned), a 63-slot adaptive cache a layer
(Strata's profile + adaptive tier, 4,236.8 MiB), MTP, P-cores, `-b 2048 -ub 512`, no projector.

**Window C (2026-09-30, the rest of the gate on 74746ec1's cache-all; `bench/flashnext_gate.py`, runs C1-C4).**
Contract PASS (every item; the slot-count check read ">= 3", the four-slot layout's, until it was fixed to the
arm's `-np`). Speed, n=3 each, 256 tokens, `-np 2` (lane idle), with the cache's hit rate over each row:

| context | decode tok/s | prompt tok/s | VRAM hit rate | MTP accepted/drafted |
|---|---|---|---:|---|
| 4.4K | 28.2 / 32.9 / 29.3 | 50 (first, cold) / 141 / 150 | 59.7% | 127/167, 158/195, 138/179 |
| ~35K | 27.5 / 38.4 / 42.2 | 131 / 162 / 162 | 62.4% | 142/177, 180/205, 181/198 |
| ~68K | 22.3 / 27.1 / 31.4 | 130 / 149 / 148 | 61.5% | 130/169, 168/220, 176/201 |
| ~140K | 17.1 / 20.5 / 21.5 | 128 / 136 / 137 | 57.5% | 125/155, 163/215, 158/219 |

Against, same card: the stock engine (`base`, -np 4, no MTP) 16.8-17.4 at 4.4K and 15.0 / 12.5 / 12.7 at ~35K (B13,
n=3); `flash-all` (-np 4, 4 static GPU expert layers) 23.9 / 22.7 / 21.5 at ~35K and 15.4 / 17.5 / 14.4 at ~142K.
Strata's own claim (5070 12 GB + 7600, IQ2_XS): 82 -> 48 tok/s from 1K to 262K; their 5060 Ti Q2_0 estimate
~80-87 at 1-4K. We are at about 35-45% of it.

The lane step (slot 0 decodes, n=3, while a thread sends jjava-shaped reads to slot 1: ~1.5K-token state, 1 token,
top_logprobs 5; 325 reads, median 3.5 s each): idle 26.5-32.0 / 27.4-42.6 / 22.4-36.7 tok/s at 4K / 35K / 68K;
ACTIVE 1.7-2.2 / 2.0-2.8 / 1.5-2.4 -- the lane takes the main conversation to ~2 tok/s. The operator, 2026-09-30:
"no jjava slowing down this highly tuned masterpiece, it stays locked in once it is swapped" -- Flash-Next at
`-np 1`; jjava and side calls go to a Bonsai on the A4000. The `-np 1` fit (`cache-fit-np1`, the same arm with one
slot and `-c 262144`, n=1 probe): lowest free 1,289 MiB against `-np 2`'s 767 -- 522 MiB more, 7 more slot rows of
66.2 MiB (63 -> 70 a layer at the same margin); decode 26.9 / 28.5 tok/s at 4.4K / ~35K (hit 64.8% / 61.4%).

jjava on Flash-Next's own server (measurement only, B14; `bench/decider/results/models/flash-next.json`,
`bench/decider/results/jevbench/flash-next-20260930-111001`): the letter prior leans hard on A (0.92-0.97 at 2-5
options, typed/2), tie band 0.052, build_intent label bias 0.107 (noul); legacy vs typed on intent 0.909 each
(n=99, in-sample); JevBench public 83.16 (easy 1.00, standard 0.972, hard 0.757; calibration on hard 79.96).

**Window D (2026-09-30/10-01, build `llama-upstream-flash-a9be47e4`, 0001-0018; kernels pass on both cards, the
poisoned sparse-FA cases 37/37 each).** All at `-np 1 -c 262144`, the operator's layout. n=3 each, 256 tokens:

| arm | 4.4K | ~35K | ~68K | ~140K | VRAM hit | verdict |
|---|---|---|---|---|---|---|
| cache-all-np1 (70 slots, q8_0 K/V, MTP 3) | 29.3 / 29.6 / 30.0 | 29.4 / 40.2 / 45.6 | 19.2 / 33.4 / 34.9 | 22.3 / 38.0 / 28.7 | 63-64% | **winner** |
| cache-all-np1-q4kv (93 slots, q4_0 K/V) | 28.3 / 31.3 / 32.6 | 32.6 / 43.2 / 44.5 | 23.3 / 38.5 / 35.2 | 19.2 / 29.3 / 29.7 | 66-70% | FAILS: KL 0.026247 / 94.516% (2.4x the yardstick), needles 13/15 (two refusals without the key); corrupt clean |
| cache-all-np1-draft5 (MTP draft 5) | 27.5 / 33.5 / 28.7 | 31.0 / 40.5 / 47.3 | 23.9 / 32.5 / 32.5 | 18.5 / 27.3 / 24.8 | 59-65% | no gain: keep 3 |

Against `-np 2` at 63 slots (window C): ~140K went 17.1 / 20.5 / 21.5 -> 22.3 / 38.0 / 28.7 and the hit rate 57.5% ->
63.9%; the first run of each row is the cold one. The 0017 profile on this card (D1, ~128K, one MTP verify step of
~4 tokens): 84.3 ms = 41.0 waiting on the GPU + 15.6 in launches + 22.9 CPU experts + 4.6 copies. The launches read
as CUDA graphs that never replay: MTP's drafts stop early (p-min 0.5), so the verify batch changes step to step and
ggml-cuda re-warms (`ggml_cuda_graph_update_required`). D5 (`--spec-draft-p-min 0`, n=1 probe): 30.2 / 29.6 / 21.6
tok/s at 4.4K / ~35K / ~128K against D1's 28.8 / 27.9 / 21.9, with acceptance down to 0.44-0.52 -- no clear net
effect, and no decode-only profile line came out of it (fewer than 100 verify graphs a context). The engine-side
candidate stays open: keep one llama graph (and so one CUDA graph) per verify-batch size instead of rebuilding on
every size change. Not built.

**Window E1 (2026-10-01, `llama-upstream-flash-cand0020`, 0001-0020; kernels pass on both cards).** 0020's graph cache
(`LLAMA_GRAPH_CACHE=8`) on cache-all-np1, speed n=3: 4.4K 27.1 / 33.0 / 33.2 (mean 31.1 vs 29.6 without), ~35K 27.7 /
41.3 / 44.9 (38.0 vs 38.4), ~68K 24.3 / 35.8 / 39.0 (33.0 vs 29.2), ~140K 20.6 / 26.2 / 29.5 (25.4 vs 29.7):
inconclusive. The 0017 profile (768-token decodes, ~128K): a verify step 83.5 ms = 47.4 waiting on the GPU + 27.2
CPU + 3.6 launches + 5.2 copies, against D1's 84.3 = 41.0 + 22.9 + 15.6 + 4.6 -- the launches went, the wait on the
GPU grew by as much: the one-by-one launches had overlapped GPU execution, and the step is GPU-bound. Greedy
identity cannot be judged on this card: the same arm run twice also differs (the ~32K slot state; the greedy text
at chars 435 / 169), as the graph cache's run does against it (248 / 10). Not shipped; 0020 stays a pinned patch,
off.

**The 1,024-token crash (2026-10-01; 0022).** The deployed entry (a9be47e4, cache-all, 70 slots, MTP, `-np 1`) died on
its first request twice in the coordinator's deploy check (06:42:57, 06:59:51; llama-swap: "upstream process exited
unexpectedly"; the driver logged Xid 13 "MMU NACK" at both instants). Reproduced outside llama-swap with the entry's
exact command and env: a FRESH server whose first prompt is exactly 1,024 tokens -- "Reply with exactly: ok" + the
concept seed + the two image tools -- faults 5 of 5 times ("an illegal memory access" in MUL_MAT_ID, blk.0's
`ffn_moe_gate`, src0 the op-offloaded iq2_s experts, a 508-token ubatch: llama-server ran the prompt as 512 + 508 + 4).
Not the cause: the expert cache (`--moe-expert-cache 0` still faults), MTP (off still faults), host memory. It passes
with 1,025 tokens, on a warm server, and with `--no-op-offload`. The bug is upstream's, at our base: `mul_mat_id`'s MMQ
padded its packed src1 by `ggml_cuda_mmq_get_J_max(..., ne11)` columns -- 0 when the activations are broadcast (ne11
= 1) -- while the tile width it picks (up to 128) follows the tokens, so the last expert's last tile read up to J-1
columns past the buffer; on a fresh process that ran off the CUDA VMM pool's mapped end. test-backend-ops reproduces it
with the model's shape (iq2_s, 2560 -> 640, 512 experts, top-10, broadcast): n = 508 faults, and it is the only n in
440..600 that does (one process per n, before the fix). 0022 pads by the widest tile the switch can pick (128 columns,
18 KiB a call): the shape passes at 500..516, MUL_MAT_ID 932/932 on the 5060 Ti, and
`llama-upstream-flash-cand0022` (0001-0018 + 0022) serves the fresh 1,024-token request (n=1).

**Prefill, cand0022, deployed entry, a warm server** (`C:/Users/jwals/octo/fn-crash/prefill.py`: a fresh prompt each
run, `cache_prompt` false, the server's own `prompt_per_second`; n=3, tok/s):

| prompt | op offload on (as deployed) | `--no-op-offload` |
|---|---|---|
| ~1K (1,038) | 101 / 236 / 260 (the first right after a fresh server's first request) | 97 / 99 / 100 |
| ~8K (8,024) | 247 / 313 / 318 | 92 / 101 / 101 |
| ~32K (32,032) | 231 / 234 / 233 | 95 / 96 / 96 |

The tens of tok/s on a first request are cold: 12-30 tok/s on each fresh server's first ~1K prompt (the deployed run's
21), 40 for `--no-op-offload`'s warm-up, and the second request of the same size runs at ~240. Op offload is 2.4x
faster warm, and it stays.

**Operator, 2026-09-30, on cache-all's KL** (mean KLD 0.011094, same top-1 96.532%, batch 4, against the yardstick's
0.010993 / 96.757%): "Yes accept and deploy" -- the deploy passes `--accept-kl`, conditional on the rest of the
gate passing on the fixed build (`C:/Users/jwals/octo/flashnext-gate-20260929/operator-decisions.json`).

At 03:29:50 a Python-urllib client asked llama-swap for `GET /upstream/bonsai/slots` (10 s timeout, `logs/stack.log`),
which started `bonsai` on the card during the window; the gate's guard killed that arm (kvcache-fit, first attempt).

## 9. Every Strata speed mechanism, against `llama-upstream-flash` (2026-09-29)

Strata's claims are theirs (RTX 5070 12 GB + Ryzen 5 7600 AVX-512 unless stated; n=1 per cell); ours name the script
and n. "Ported" = in `engines/patches/llama-upstream-flash/`.

| mechanism | Strata's claim (commit) | status here |
|---|---|---|
| MTP draft layer, 3 drafts, verify window | 2.7-3.2 tokens per round, acceptance 0.63-0.86 (speed-0114) | PORTED as upstream PR #28243 (0006); the draft's experts on the CPU (`--spec-draft-n-cpu-moe 49`); costs ~2.4 GB VRAM beside the trunk (RS 450 -> 1,801 MiB at `-np 4`, draft ~1 GB: fit2). Gate: `flash-all`, `flash-mtp` |
| Profile-ranked VRAM expert cache + adaptive swaps | 70-92% of routed reads from VRAM on 16 GB (layer-split README), ~4,000 experts on 12 GB | PORTED (0003 #27861 + 0004 Strata + 0007); the CUDA fault fixed by 0013 (2026-09-30), hit rate logged by 0016. The deploy candidate `cache-all`: 63 slots a layer (3,024 experts, 4,236.8 MiB beside the KV and MTP), hit rate 61-67% (cache-fit probe, n=1); every layer's experts in RAM otherwise |
| CPU expert kernels: AVX2 multi-token i-quant rows | round 50.9 -> 48.5 ms (-4.7%) on a 5700X3D + 5060 Ti (`ed14227`) | PORTED (0009, Strata's iq_avx2.cpp @ 3ce2523c, MIT) + our Q2_0 rows; ours measured within noise to ~10% on 2-4 token batches (moe_cpu_bench, n=300) |
| Q2_0 CPU kernel (ggml has only scalar on x86) | their own AVX-512/AVX2 Q2_0 rows (`pool.cpp:383`) | OURS (0008): down layer 0.575 -> 0.118 ms; whole-arm decode 17 -> 26 tok/s at 4.4K (`flash-cpu` vs `base`, n=3) |
| E-2: 2 KB row prefetch ahead of the AVX-512 decode | 37.1 -> 36.5 ms/round (`df6980d`) | not ported (AVX-512 path; ~2%) |
| Pinned expert arena + helper copy threads | part of 0.1.13's 2x prompt (`928b0e07`) | PORTED (0005 `LLAMA_PIN_EXPERTS`) |
| Windows large-page arena | part of 0.1.13 (`0bf3216e`, PR #42) | PORTED (0005 `GGML_CUDA_HOST_LARGE_PAGES`); needs "Lock pages in memory" (a system policy we do not change) -- not run |
| Expert streaming ring (prefill: next layer's experts over PCIe during attention) | part of 0.1.13: Q2_0 32K prompt 572 -> 1,290 (`2026-09-28-prefill-speed`) | PORTED as upstream PR #28414 (0002 `--prefetch-experts-slots`); not in the combined arms yet |
| MMQ experts in the prompt path | part of 0.1.13 | llama.cpp's own (MMQ is the default for these types) |
| Larger prompt chunks (8K) | part of 0.1.13 | `-ub`: 4096 needs a 14.5 GB compute buffer on this model, 1024 3.8 GB (vram-*.log); the `flash-ub1k` arm |
| Batched PLE block; embedding in one gather; PLE rows by 4 threads | Q2_0 4K PLE 591 -> 291 ms (`5b7e316`) | not ported (llama.cpp's lazy PLE gather; upstream PR #29030 is the candidate) |
| D-1 QSA prompt attention on tensor cores | 32K prompt +18.8% (`dce4598`) | not ported (prompt side) |
| QSA select on tensor cores + register top-k | 128K prompt +14.6%, "decode uses it too" (`731899f`) | the DECODE half addressed our own way: 0011 (upstream's radix select for GGML_OP_TOP_K; ggml sorted the whole KV row 12x a token); 0014/0015 (ours): the sparse attention reads only the selected q8_0 cells, and a 1-2 query decode takes it |
| C-1/C-2 select grid on active blocks; chunked indexer appends | Coder 4K +18%, 20K +13% (`758eb1a`) | not ported (prompt side) |
| D-2 GDN recurrence split; C-3 short conv tiled | 32K prompt 1,258 -> 1,308 (`66f4341`, `2575cb1`) | not ported (prompt side) |
| D-4/D-5 queued refills; stream issuer thread | 162 -> 96 ms/prompt; IQ3_S 32K 1,143 -> 1,213 (`581765a`, `cf68b00`) | not applicable as is (their cache slots and stager) |
| E-6 device plan: a layer whose experts are all resident skips the host | not quantified (`efddd74`) | not ported (needs the cache) |
| Batched verify-window kernels | not quantified (`1e4515c`, PR #109) | llama.cpp batches a verify window natively |
| PCIe share: the GPU computes some missed experts by copying them | 5060 Ti x8 probes 14.1 GB/s -> `pcie_frac` 0.29 (`7a4b627`) | not ported |
| KV streaming `--kv-resident` (KV in RAM, the read window in VRAM) | Q2_0 262K 50.9 -> 62.6 tok/s; ~+6% at 128K | PART: 0012 puts the attention K/V in host-mapped memory (frees ~3.8 GB); 0014/0015 make the decode read only the selected cells. Without Strata's VRAM page window it is still slower than the KV on the card (kvcache-fit 16.2 / 17.9 tok/s at 4K / 35K vs cache-fit 24.7 / 37.9, n=1): the page window is M3, not ported |
| q4_0 KV + Hadamard | ~4% at 128K, perplexity +8-12% | not taken (q8_0 kept) |
| Prompt-lookup (suffix) drafter beside MTP | code edits 6-11% faster | not ported (llama.cpp has `ngram-*` speculative types; combining with MTP untested) |
| Conversation cache: pinned shared prefix, a checkpoint at the system prompt's end | re-reads only what follows the system prompt (`6fb2085`, `c1e9033`) | llama-server's slots + `--cache-ram` + context checkpoints already do this for us |
| Layer split over several GPUs | Coder 5080+3090: prompt +18-20%, decode +0-7% | the `flash-a4000` arm: experts of m layers on the A4000 via `-ot` (static), measured in this window |
| Bulk expert-arena reads on MSVC | load time (`5edb9d6`) | not applicable (llama.cpp mmap) |
| Experimental speed projection (a refusal-direction ablation) | top-1 changes at 10% of positions | NOT TAKEN (a behaviour change, not a speed one) |

## 10. Port plans for the unported Strata pieces, and the lost slot cache (2026-10-02; plans, nothing built)

Strata paths are in `Niko1221/Strata` @ `d551edf` (the tree this section was read from); ours are in the vendored
`engines/src/llama-upstream-flash`. Strata's numbers are theirs (n=1 each). Nothing here is measured by us yet: the
order of the ports is decided by the profiles of step (b) (`bench/flashnext_gate.py` arms `probe-opt-np1`,
`probe-sched-np1`, `prefill-*-np1`).

### 10.1 M3: the VRAM page window for the attention K/V (`--kv-resident`)

- **Strata**: `src/kernels/cuda/kv_stream.cu` (`resolve_kernel` :82, `copy_kernel` :161, `kv_stream_resolve` :204),
  `include/strata/kernels/kv_stream.hpp` (`KvStreamMap`), `src/core/layer.cpp` (`kv_plan` :491, `qsa_kv_resolve` :691).
  The K/V lives in host-mapped pinned memory; VRAM holds `n_slots = ceil(max(N, 20480)/4)` pages of 4 cells (one
  indexer block); `page_table[block]` = slot or -1. After top-k, one kernel marks the selected blocks' pages (CLOCK
  second chance, never a page stamped this step) and a copy kernel reads the missing pages from the host alias over
  PCIe into their slots, on the compute stream. Readers address `page_table[cell/4]`; writers write the host copy and
  the slot if resident. Claim: Q2_0 262K 50.9 -> 62.6 tok/s (1,589 -> 3,872 experts in VRAM), ~+6% at 128K.
- **Ours today**: 0012/0021 put the QSA layers' K/V in host-mapped memory (`src/llama-kv-cache.cpp`,
  `ggml-cuda.cu`'s mapped buffer type); 0014/0015/0018 make the sparse attention convert only the selected cells to
  f16 (`ggml/src/ggml-cuda/fattn.cu`, `flash_attn_mask_to_sparse_indices` and the conversion before the MMA kernel).
  With the K/V mapped, that conversion reads its ~2,051 cells per query per QSA layer over PCIe EVERY step.
- **Port**: make that converted-cell buffer persistent. Per mapped K/V tensor, a page pool on the card (pages of
  `r` = 4 cells, K and V), a device page table, and a resolve + copy pair in front of the sparse kernel; the gather
  then indexes through the page table. 0019's block list (512 blocks + the tail) is the natural page key, so M3
  builds on M2b. Writers: `set_rows` / `cpy` into a mapped tensor invalidate the pages at and above the lowest block
  written (the tail block and MTP's rejected drafts live there). State in `ggml_backend_cuda_context`, keyed by the
  tensor's data pointer; size from `LLAMA_KV_RESIDENT=N` cells (Strata's floor is 20,480).
- **Tests**: `test-backend-ops` FLASH_ATTN_EXT sparse cases with the window on must equal the window off bit for bit
  (it is a cache), with the non-resident host cells poisoned after the copy; the gate's kernels, exact, needles.
- **When**: only if E2 shows the mapped K/V (0021 at ~118 slots) decoding slower than the K/V on the card at the
  same context. If 0021 alone already wins, M3 is the remaining PCIe read per step, sized by the profile.

### 10.2 `pcie_frac`: a share of a step's missed experts computed on the GPU

- **Strata**: `src/program/generate.cpp` (`probe_pcie_h2d_gbps` :848: 4 x 256 MiB pinned copies between two events;
  the rule :1320-1335: `bw >= 20 ? base : bw < 4 ? 0 : min(base, max(0.05, base*bw/26))`, base 0.55),
  `src/core/expert_source.cpp` (`expert_pool_dispatch_multi` :282, the split :309-392), `src/core/verify.cpp`
  (`fetch_dma` :1143, `publish_plan` :1155; 16 staging blobs). The last `m = misses * frac` missed experts of a
  layer go to staging slots (a copy kernel from the pinned arena's device alias, or `cudaMemcpyAsync` on a copy
  stream) and the GPU computes them while the CPU pool computes the rest; the graph waits on doorbell flags and
  merges by destination row. Their 5060 Ti x8 probe: 14.1 GB/s -> 0.29.
- **Ours today**: a miss is computed on the CPU (0003's chain: the cached slots by MMVQ on the card, the uncached
  ids by the CPU `mul_mat_id`, one scheduler split per layer each way). ggml already has the upload half:
  `ggml-backend.cpp` copies only the USED experts of an op-offloaded `MUL_MAT_ID` to the card (`copy_experts`,
  around line 1985), which is what prefill uses.
- **Port**: split the miss branch in `src/llama-moecache.cpp`'s graph into two `MUL_MAT_ID` nodes over the same
  host weights -- one kept on the CPU, one offloaded -- with the missed ids partitioned by a small CPU node (the
  last `m` in routing order to the GPU, the rest to the CPU), so the existing used-expert copy uploads exactly those
  (1.38 MiB an expert-layer). `LLAMA_MOE_PCIE_FRAC`, default from a one-off H2D probe at load (Strata's rule).
  No new kernel.
- **Open, for the profile to answer first**: whether the CPU's miss computation overlaps GPU work at all today
  (0017: a verify step is 41-47 ms waiting on the GPU + 23-27 ms of CPU, added, not overlapped). If they are serial,
  the larger gain is running the two branches concurrently, and `pcie_frac` is the way to balance them.
- **Tests**: `test-backend-ops` MUL_MAT_ID with a partitioned id set against the single node (exact); the gate's KL
  (batch 4), needles, corrupt, speed.

- **BUILT (2026-10-02, offline; patch 0024, `LLAMA_MOE_PCIE_FRAC`, default off, NOT pinned, NOT measured on a GPU)**.
  Strata's rule and plan are ported (`expert_source.cpp` @ d551edf :282-392: the last m = (missed * pcie_num) >> 8 of
  the layer's distinct missed experts in routing order; `generate.cpp`'s probe and rule are at d9ab843 :950 and
  :1715-1735 -- the plan's `generate.cpp:848` / `:1320-1335` are not in d551edf, which has only a fixed per-pack
  default, 0.55 native / 0.2 Q2_0). It is NOT built as two MUL_MAT_ID nodes with one op-offloaded, for three reasons found
  in the source: an offloaded node cannot skip ids, so the experts it does not own are either computed from slots
  nothing copied (a NaN, times a zero weight, is NaN) or one dummy expert is copied on every layer where the share is
  empty (fewer than 4 misses at a 0.29 share: 49% of layer-steps at 36% misses of 10); ggml's used-expert copy is
  synchronous inside the scheduler, so nothing would overlap; and the scheduler allocates the node's full 512-expert tensor as its input. Built
  instead: a small POOL of expert slots on the device (one per expert shape, shared by all layers: 21 slots x 1.38 MiB =
  ~29 MiB at a 0.29 share and a 7-token batch limit), a PLAN node (a ggml custom op on the CPU backend, first in the layer's
  CPU split) that picks the share, writes the table the CPU `mul_mat_id` nodes skip by and the pool's slot ids, and
  enqueues the share's copies from the pinned experts on a second CUDA stream; a WAIT node after the CPU chain that blocks
  on the copies' event; and the same `mul_mat_id` + activation chain the cache uses, over the pool, after it. No new
  kernel, no new scheduler split. `LLAMA_MOE_PCIE_FRAC=<f>` or `auto` (the probe, Strata's rule, `LLAMA_MOE_PCIE_BASE`
  default 0.55 -- THEIR measurement on THEIR CPU kernels, so `auto` is a place to start a sweep of explicit shares, not a
  result). **Tested** (`tests/test-moe-pcie-split.cpp`, ctest `--quick`; CPU backend, no GPU): the rule and the env
  parse; 5,000 random partitions against a brute force; the model's shape (iq2_s 2560 x 640 gate/up, q2_0 down, top-10,
  1-7 tokens, duplicates across tokens, random residency, shares 0 / 8 / 74 / 141 / 256 of 256) with CPU + cache chain +
  pool chain equal to the unsplit chain and, EXACTLY, to the cache chain over a cache that also holds the pool's experts (the
  same kernels on the same rows in one chain; on a GPU the CPU reference differs by q8_1 noise, this one cannot) -- bit-exact on the
  CPU backend (60 trials, 25,687 checks, the pool took 371 experts over 37 steps),
  the pool poisoned before every step and the copies landing only at the wait node, and a control that drops the wait
  and must fail (it does). With `--device NAME` on the A4000 (the real second stream and event, scheduled over {GPU, CPU}; 60 trials, 25,567 checks, 0
  failed, exact against the enlarged cache, 1.6% from the CPU reference = kernel noise): the first device run failed with NaN, and
  the cause was the test, not the engine -- a graph whose inputs are all CPU graph inputs ran entirely on the CPU, copying the
  device pool to the host at the start of the split, before the plan had enqueued its copies; the engine's router and next-layer
  work are on the GPU, which is what places the chains there, and the test now anchors them the same way. In the engine (a tiny
  generated olmoe with Q4_0 experts, `--moe-expert-cache 1`): plan and wait nodes on the CPU, every miss through the pool,
  next-token logprobs within 3.5e-3 of the GPU cache path; switch unset: the GGML_SCHED_DEBUG graph (94 splits, 1,360 nodes) is
  identical to the 0001-0022 and int0028 builds. One more thing the device runs showed: on this card (WDDM) the compute stream
  waits for work queued on a second stream (a step took ~0.9 s behind 600 x 256 MiB of device-to-device copies on the second
  stream), so the second stream's copies may not overlap compute at all -- the first thing a speed gate should look at. **Not measured, and to be measured first**: (1) the arithmetic says little is on the table.
  Section 8 puts the CPU at 0.042-0.048 ms per expert after 0008 and a 1.38 MiB expert over a 14 GB/s link at ~0.1 ms, so
  the balance point is about 0.3 only if the copy and the CPU overlap; at a 64% hit rate a layer misses 3.6 of 10 and
  Strata's floor rule gives m = 0 for fewer than 4 (49% of layer-steps), 0.54 expert per layer-step on average at 0.29,
  and each shared expert costs 3 copy submissions plus the pool chain's 5 GPU nodes. (inferred, not measured) (2) The
  copies overlap the CPU chain; the GPU's own chains do not (they still follow it: the CPU split's input copy
  synchronises the compute stream, so a GPU chain launched before it is waited for). Running the CPU and GPU branches
  concurrently -- the plan's "larger gain" -- needs the CPU chain's inputs copied to the host in an earlier split, so that
  the GPU chains can be launched between that copy and the CPU compute; not built. (3) the gate: graph dump with the switch
  unset against cand0022 (`GGML_SCHED_DEBUG=2`: byte-identical, the claim for "off"), KL, needles, corrupt, then speed n=3
  at a sweep of explicit shares.

### 10.3 The stager: queued refills and the stream-issuer thread

- **Strata**: `src/prefill/prefill.cpp` (`struct Stager` :122-239: a ring of 16 pinned buffers and 2-4 memcpy
  workers for UNPINNED blobs; the issuer thread :945-1003 calls `cudaMemcpyAsync` on the copy stream, throttled by
  ring occupancy, with `copied[]` / `used[]` events), `src/core/expert_cache.cpp` (`fill_slot_queued` :268,
  `sync_queued` :285), `generate.cpp` (`adapt` / `apply_pending` :3302-3370: a swap marks the victim uncached first,
  copies on `adapt_stream`, and publishes the new residency only once its event has completed). Claims: refills
  162 -> 96 ms a prompt; IQ3_S 32K 1,143 -> 1,213 tok/s.
- **Ours today**: 0004's adaptive swaps run in `llama_moe_cache_step` (`src/llama-moecache.cpp` :528) with
  `ggml_backend_tensor_set` -- a synchronous copy on the compute stream, up to 96 swaps every 4 steps. 0002 already
  creates a second backend instance on the card for uploads that overlap compute (`sched->prefetch_backend`,
  `ggml_backend_tensor_set_async` + events).
- **Port (D-4 only)**: swaps issued through a second backend instance with an event; the slot table changes at the
  next step whose event has completed (victim -> the dummy slot at once, so the CPU serves it meanwhile). The
  issuer thread and the pageable-blob stager are NOT ported: our experts are pinned (0005), so `cudaMemcpyAsync`
  returns at once and there is nothing for a thread to hide. **Superseded 2026-10-06 (13.5): the experts have been
  unpinned since 2026-10-05, so the copy from pageable memory does block its thread; the stager is patch 0030.**
- **Tests**: the cache's unit test of the table (a swap never leaves a slot whose rows are half-written visible:
  poison the slot before the copy, the gate's kernels + needles); speed n=3.
- **When**: if the decode profile shows copies or waits attributable to swaps (0017's `copy` is 4-5 ms of ~84).

- **BUILT (2026-10-02, offline; patch 0025, `LLAMA_MOE_CACHE_QUEUED_REFILL=1`, default off, NOT pinned, NOT measured on a GPU)**.
  The plan's premise is not what the source does: 0004's swaps never ran on the compute stream in `llama_moe_cache_step`.
  Since 0003 (#27861) the slices are copied by a worker thread (`ggml_backend_tensor_set`, one copy and one stream
  synchronise per matrix) and published at a later step; what ran on the decode thread every adaptive step was the table
  writes, one synchronous device copy per evicted expert and per newcomer, up to 2 x 96. Built: no worker thread; a
  step's evictions go to the host mirrors, each changed layer's tables are written once, whole (one device copy a layer);
  the newcomers' three matrices are enqueued on a second backend instance's stream and closed into a batch with an event;
  later steps poll the events WITHOUT blocking (`cudaEventQuery`, offered as the registry proc address
  `ggml_backend_event_query` so no ggml header changes and no CUDA object rebuilds) and publish only the jobs of a batch that
  has landed, in order; the LRU policy does not queue an expert whose copy is in flight. Strata's issuer thread and
  pageable-blob stager are not ported (our experts are pinned, 0005; with pageable ones the new path is not used and the
  upload thread stays, with a warning). `LLAMA_MOE_REFILL_POISON=1` (tests, both paths) fills a slot with 0xFF before its
  copy. **Tested** (`tests/test-moe-refill.cpp`, ctest; no GPU): the real `queue` / `enqueue_job` / `publish` /
  `flush_tables` over the cache's bookkeeping, random swaps, a fake engine whose copies land only when told to; after every
  operation every published expert sits in a slot holding exactly its bytes (846 checks, 0 failed; 1,221 refills over 400
  steps, 399 steps found a batch in flight, steps to land avg 3.58 max 13), and a harness that publishes at push time is
  caught in 40 of 40 steps. With `--device NAME`: the real stream and event, made slow with 512 MiB of copies queued ahead,
  and the event query itself: run on the A4000, 1,411 checks, 0 failed, including a held batch (a 4 GiB backlog ahead of it: the first poll returns at once with nothing, 8.8 million polls over 647 ms until it lands, then published and checked); between steps every batch has landed by the next on that card, since any device-synchronising call also waits for the second stream. **Not measured**: whether any of this moves the decode (the table writes it
  removes are ~48-100 small synchronous copies per adaptive step against ~85 ms a verify step, inferred); the engine-side
  gate is the n=3 speed step plus the cache's new "refills:" line (steps to land). 0017's `copy` 4-5 ms is the scheduler's
  input copies, not the refills.

### 10.4 Prompt side, for a 30K agent prompt

- **What the prompt costs today** (cand0022, warm, n=3): 231-318 tok/s with op offload, 95-101 without. With op
  offload every 512-token ubatch uploads the experts it uses, per layer: up to 33.1 GiB a ubatch if all 512 x 48 are
  used (66.2 MiB a slot row x 512). At Strata's 14.1 GB/s for this card on x8 that alone would cap prefill near 200
  tok/s -- we measure more, so either fewer experts are used per ubatch or the link is faster; the prefill profile
  (sched `copy` / `other`, the op table) and an H2D probe say which. The four candidates, in the order the profile
  will rank:
  1. **Larger prompt ubatch** (`-ub 1024`: half the uploads per token; 1,898 MiB more compute buffer = 29 cache
     rows; the `cache-all-ub1k` arm exists). 0021 frees ~3.2 GB that can pay for it.
  2. **0002's streaming ring** (`--prefetch-experts-slots`, Strata's `prefill.cpp` :70-86, :889-1003: the next
     layer's experts upload on a copy stream while the current layer's attention runs). Built, never measured in
     the combined arm: `prefill-ring-np1` (2 staging slots, 62 cache slots).
  3. **CPU and GPU experts together in prefill** (10.2's split applied to prompt batches: ~100 tok/s of CPU
     experts are idle while the GPU's uploads are the limit).
  4. **D-1, QSA prompt attention on tensor cores** (`src/kernels/cuda/qsa_prompt_attn.cu`: `prompt_attn_i8_kernel`
     :358, int8 codes straight into `mma.sync` with the scales applied in FP32, one block per (query, KV head), a
     whole chunk per launch; select: `qsa_select.cu` `block_scores_tc_kernel` :189, C-1's active-block grid, C-2's
     chunked indexer appends). Claim: attention 5,216 -> 1,318 ms of a 32K prompt, +18.8%. Ours: the sparse MMA
     kernel of `fattn.cu` after 0014's f16 conversion. Worth porting only if the op table puts FLASH_ATTN_EXT,
     TOP_K and the indexer above the expert uploads.

### 10.5 "Reused 0 of 30,262 tokens" two hours later (diagnosis; `logs/proxy.log`, `index/corpus.sqlite3`)

- **What the records say** (2026-10-01): 19:29 `first prompt 30262 reused 0` on flash-next, slot 0, no swap since
  17:2x. It was a NEW VS Copilot conversation (corpus: `n_messages` 2, `first_turn` true, 73 tools), not the 16:52
  one continued; the requests after it in the same conversation reused 30,833 of 30,884 and 30,992 of 31,066. So
  the slot was not emptied by idleness, a release or the prompt cache: the new prompt diverged from the slot's
  tokens EARLY, and the server could not keep the common prefix.
- **Why a common prefix is lost on this model**: Flash-Next's memory is hybrid (DeltaNet recurrences), which cannot
  be truncated. On a divergence at position p, llama-server needs a context checkpoint at or before p
  (`tools/server/server-context.cpp` :3300-3400), else "forcing full prompt re-processing due to lack of cache
  data". Checkpoints are only made at a user message's start, near the prompt's end, and at least 8,192 tokens
  apart (:3596-3646) -- never inside a 26K system + tools block. Two candidates, which the records cannot tell
  apart (the corpus keeps 1,200 characters of the system text; both conversations' heads hash alike): (i) the
  prompts differ INSIDE the system + tools block (VS Copilot's own environment text, the tool list -- 73 tools,
  110 at 20:25 --, anything of ours rendered there), where no checkpoint can exist; (ii) they differ only from the
  first user turn on (it opens with `<current_datetime>`), and the user-start checkpoint that should cover that
  was not there -- the 16:52 conversation's first read was cancelled at 10,240 tokens and resumed, which is the
  one unusual thing about how its slot was built. The prompts were 26,500 and 30,262 tokens, both two messages.
- **Not involved** (read from the code): `--cache-ram` (only used when f_keep < 0.5, and a failed load logs "failed
  to load prompt from cache"), the proxy's releases (logged "kept": locked), the MTP draft (its state rides in the
  same checkpoints: `load_dft`, `common_speculative_set_state`).
- **To confirm on the live server** (a config change, the coordinator's): on the flash-next entry add
  `--log-file <logs>/flash-next.server.log -lv 4` and env `LLAMA_SERVER_SLOTS_DEBUG=1`,
  `LLAMA_SERVER_SLOTS_N_DIFF=16`; open two VS Copilot conversations a few minutes apart. The lines that decide it:
  `old: ... | ...` / `new: ... | ...` (the tokens at the mismatch), `restored context checkpoint (pos_min = ..)` or
  `forcing full prompt re-processing`, `erased invalidated context checkpoint`, and for the cache-ram path
  `updating prompt cache` / ` - saving prompt with length` / `failed to load prompt from cache`.
- **The engine-side fix if confirmed** (a patch, not built): periodic checkpoints inside a long prompt -- one every
  `checkpoint_min_step` tokens whatever the message boundaries (Strata does the same: `--prompt-cache-every`
  16,384 and a pinned root at the system prompt's end, `generate.cpp` :3241, :3906-3921) -- so a changed system
  block re-reads from the last 8K boundary before the change, not from 0. Its cost is host RAM per checkpoint (the
  size is in the `created context checkpoint` line; unmeasured here).
- **0023, built 2026-10-02** (operator: "We should build this"; `0023-server-checkpoint-every.patch`, on 0022,
  independent of 0019-0021; candidate `llama-upstream-flash-cand0023`): `--checkpoint-every N` (env
  `LLAMA_ARG_CHECKPOINT_EVERY_NT`, default 0 = today's behaviour) ends a prompt batch N tokens after the slot's last
  checkpoint and makes one in front of the batch that starts there. N = 16,384 is Strata's
  (`src/program/generate.cpp:283` `prompt_cache_every = 16384`). The MTP draft's state rides in it like in every
  checkpoint (`create_checkpoint`: `update_dft`, `common_speculative_get_state`). The server's two existing knobs:
  `--checkpoint-min-step` (8,192) does not apply to a periodic checkpoint when it is made, and applies to it like
  to any other once the list is full (other requests' checkpoints closer than the min step to an earlier one are
  dropped first, then the oldest) -- so N >= 8,192 is kept, a smaller N is not on a full list; `--ctx-checkpoints`
  (32) caps the list as before.
- **What a checkpoint costs on this model** (the server's own `created context checkpoint ... size` lines,
  `cache-all-np1.server.log`, 24 checkpoints at 3.9K-140K tokens): 112.6 MiB + 2.02 KiB per token of position --
  120.3 MiB at 3,916, 390.2 MiB at 140,483, so 630.6 MiB at 262,144 (extrapolated). The periodic set alone: 1
  checkpoint (145 MiB) for a 30K prompt, 7 (1.7 GiB) for 130K, 15 (5.4 GiB) for a full 262K window. The list's cap
  is unchanged: 32 x the size at the slot's length (20 GiB at 262K, already true today). Host RAM is the constraint:
  with flash-next loaded the box had ~1.4 GB of commit left (2026-10-01, 109.0 of 110.5 GB; the pagefile grows on
  demand). The time to make one is in 0023's log line; measured with the live check.
- **The check** (`bench/flashnext_gate.py` step `ckpt`, arms `ckpt-np1` and `ckpt-off-np1`): A (~30K tokens), B =
  A's first ~20K + other text, C = A's first ~8K + other text, greedy, 48 tokens. With the flag B must reuse exactly
  floor(lcp/N) x N tokens and C 0; without it B reuses 0. B's text must equal a cold run of B, judged only if two
  cold runs equal each other (this card's own variation, window E1); the first token's top-5 probabilities are
  recorded for each. **Not yet run** (the card).
- **Bonsai and Mirai S**: the same limit. Their engines carry the same server code (the vendored
  `llama-bonsai2-ada` and `llama-mirai-s` trees: the patch applies to both with offsets, no rejects), their memory
  is recurrent too (Qwen3.8-27B's Gated DeltaNet), and `logs/proxy.log` has it on bonsai: `first prompt 30164
  reused 0`. Not ported.
- **Also seen there**: the 16:52 request's swap-in took 180 s and the client hung up; its prefill was cancelled at
  10,240 tokens, and the retry reused exactly those 10,240.

## 11. Stability of a multi-turn session (2026-10-02; operator: "Stable multi-turn flash-next from a harness is my main concern right now")

### 11.1 Where the host memory goes (`C:/Users/jwals/octo/fn-crash/mem_probe.py`, cand0022, the deployed entry, n=1 each)

The limit the box hits is COMMIT (RAM + pagefile), and with the experts pinned, physical RAM too. bonsai-a4000 and
embeddings were loaded on the A4000 throughout (12.2 + 2.5 GB of private bytes); nothing else on the 5060 Ti.

| | before the load | loaded, pinned (deployed) | + a 33K prompt | loaded, `LLAMA_PIN_EXPERTS=0` | + a 33K prompt |
|---|---|---|---|---|---|
| system commit in use / limit, GB | 36.7 / 114.7 | 91.5 / 114.7 | 92.3 / 114.7 | 54.9 / 114.7 | 55.8 / 114.7 |
| free physical RAM, GB | 47.0 | 0.8 | 0.8 | 43.1 | 12.2 |
| flash-next private bytes / working set, GB | - | 54.8 / 47.2 | 55.8 / 52.8 | 18.4 / 10.6 | 19.4 / 41.6 |
| bonsai-a4000 working set, GB (private 12.2) | 6.1 | 0.14 | 0.05 | 0.05 | 0.05 |
| load to /health, s | - | 88.3 | - | 9.5 - 12.6 | - |
| the first 33K prompt, tok/s; decode, tok/s | - | - | 106.8; 27.7 | - | 157.4; 28.0 |

What the 54.8 GB is (the server's own lines at `-lv 4`):
- **36.4 GB: the pinned expert arena** -- `CPU model buffer size = 33,812.50 MiB` (the RAM experts, pinned by 0005 /
  `LLAMA_PIN_EXPERTS=1`) + `CUDA_Host model buffer size = 2,550.00 MiB` (the MTP draft's). Pinned memory is committed
  AND locked in RAM. Unpinned, the same experts are the mmapped file (`CPU_Mapped model buffer size = 37,034 MiB`):
  no commit, and the OS may evict and re-read them.
- **~15.1 GB: the card's allocations, charged to the process's commit** (WDDM): with the pin off the process still
  has 18.4 GB of private bytes at 15,104 MiB of VRAM and no large host buffer of its own. Commit tracks VRAM about
  one for one, for every llama-server on either card.
- **~3 GB**: host compute buffers (402 + 281 + 281 MiB), the CUDA runtime, the heap.
- **+1.0 GB after a 33K prompt**: two checkpoints (177 + 178 MiB) and the prompt's buffers. `--cache-ram` (8,192 MiB)
  is not touched until a different conversation takes the slot.
- The mmapped n-gram table (27.5 GB) and model file are not commit.

Pinning costs 36.4 GB of commit, all the free RAM (0.8 GB left: the other servers' working sets are paged out to
~50 MB), and 76-79 s of load. Without it the load is 9.5-12.6 s and the first 33K prompt ran faster (157 vs 107
tok/s) with the same decode (28.0 vs 27.7), n=1. Not yet measured unpinned: warm prefill and decode n=3, and the
expert cache's refills from pageable memory. **The first thing to try for stability is `LLAMA_PIN_EXPERTS=0`** (one
env line in the entry; `bench/deploy_flash_next.py` without `--pin`).

**Pinned against unpinned, measured (2026-10-02; `C:/Users/jwals/octo/fn-crash/pin_compare.py`, cand0023, the deployed
entry + `--checkpoint-every 16384`, n=3 each; bonsai-a4000 and embeddings loaded):**

| | pinned (deployed) | `LLAMA_PIN_EXPERTS=0` |
|---|---|---|
| load to /health, s | 87.5 (file cache warm: 42.9 GB standby) / 76.3 / 79.2 (25-28 GB standby) | 9.1 / 7.6 / 7.6 (warm, 44 GB standby); 12.6 right after a pinned server exited (9 GB standby) |
| warm prefill, ~8.8K fresh prompt, tok/s | 106.2 / 157.7 / 151.4 | 149.0 / 150.8 / 164.1 |
| warm prefill, ~35K fresh prompt, tok/s | 131.2 / 155.5 / 154.1 | 153.6 / 155.1 / 157.0 |
| decode ~4K, tok/s (hit rate) | 29.2 / 29.1 / 31.0 (65.5%) | 28.2 / 30.6 / 29.2 (67.5%) |
| decode ~35K | 35.2 / 35.8 / 32.6 (63.7%) | 39.0 / 33.1 / 37.4 (69.4%) |
| decode ~140K | not run here (window D, another procedure: 22.3 / 38.0 / 28.7) | 16.8 / 26.9 / 23.9 (62.6%) |
| 600 streamed tokens at ~35K: tok/s; gap between tokens p50 / p95 / p99 / max, ms | 40.8; 0 / 96.5 / 108.4 / 122.1 | 35.4; 0 / 94.2 / 107.0 / 118.8 |
| gaps over 500 ms | 0 | 0 |
| commit in use / limit, GB; free RAM, GB (loaded) | 91.6 / 114.7; 1.8 | 57.0 / 114.7; 41.5 |
| the same at the end of the run | 99.0; 0.8 (after ~35K) | 65.1; 2.4 (after ~140K: the mapped experts fill RAM as evictable file cache) |

Unpinned output: needles 5/5 at 32,768, the corrupt check 35 generations with no symptom and every tool call well
formed (`cache-all-np1-nopin`, gate.json). The expert cache's refill count is not in this build's log line (0025
adds it); its hit rate and the token gaps are the evidence that refills from pageable memory do not stall. With
imagegen-turbo registered but idle (its models load on the first picture) the unpinned load was 12.6 s to 55.4 GB of
commit, 43.4 GB of RAM free; sd-server's own 10.6 GB (measured 2026-10-01 after a picture) comes on top:
~66 GB unpinned against ~103 GB pinned, of 114.7. bonsai-vision was not loaded (it does not fit beside
bonsai-a4000 on the A4000).

### 11.2 Ways a multi-turn agent session can fail or stall (from the code; "test" = what the soak should show)

| # | situation | what the server / stack does | test |
|---|---|---|---|
| 1 | Swap-in under memory pressure | The load commits 54.8 GB and pins 36.4 GB. If commit is short when the process starts -- the previous model's process still exiting, imagegen (10.6 GB) or bonsai-vision loaded -- it dies at start (seen: exit 0xC0000142 right after another flash server was killed) or loads by paging (149-272 s loads seen; 9.5-88 s alone). llama-swap answers the waiting request with an error, or the client gives up first (16:52: the client hung up at 180 s). | Swap in with imagegen-turbo and bonsai-vision loaded; record commit before/after and the load time. Repeat with the pin off. |
| 2 | llama-swap's health wait | `healthCheckTimeout: 900` s in config.yaml and the proxy's `max_mode.LOAD_TIMEOUT_S` = 900 s: neither cuts a 150-270 s load. The CLIENT's own timeout does; the proxy's heartbeats (253641c) must keep it alive through the swap and the prefill. | A cold swap-in from the harness: bytes on the wire every few seconds until the first token. |
| 3 | A cold 26-30K prompt | 135-321 s before the first token (94-120 tok/s cold; ~230 warm). With the pin on the first prompt after a load is the slowest (the experts page in while RAM is exhausted). | Time to first token on the first request after a swap, pinned vs not. |
| 4 | A new conversation, or a changed system/tools block, on the slot | Hybrid memory cannot be truncated: without a checkpoint at or before the first differing token the prompt is re-read from 0 (10.5). 0023 (`--checkpoint-every 16384`) resumes from the last 16K boundary. A difference inside the first 16,384 tokens still costs everything. | Second conversation with the same system + tools: reuse >= 16,384; a changed tool list: reuse floor(lcp/16384) x 16384. |
| 5 | The re-rendered previous turn differs from what was generated | Falls back to the checkpoint at the previous prompt's end and re-reads the turn (the mirai-s 891-vs-276 case). Not covered by 0023. On flash-next a turn can be thousands of tokens of reasoning. | Every agent step: `cache_n` == previous prompt + previous completion (minus a few tokens). LLAMA_SERVER_SLOTS_DEBUG=1 + _N_DIFF=16 names the differing tokens. |
| 6 | Cancelled generation, then a new request | The cancel is taken between decode calls: up to one batch (2,048 tokens of prefill, ~9-20 s) later; the slot keeps what was processed (seen: a read cancelled at 10,240, the retry reused 10,240). With `-np 1` the new request waits for that. A cancel during an MTP verify window leaves the slot at the last accepted token. If the new prompt diverges before the slot's end, rows 4/5 apply. | Abort mid-prefill and mid-generation, then retry the same request and a different one: no hang, reuse as rows 4-5 predict, the answer intact. |
| 7 | A crash of the server mid-request | llama-swap returns 200 with an EMPTY body ("recovered from upstream disconnection during streaming", 2026-10-01); the proxy logged `finish=stop, 0 chars` -- an outage delivered as an empty answer, which a harness takes as the turn's end. The next request reloads (150-270 s) with an empty slot. | Kill the flash-next process mid-generation: the client must get an error (5xx / an SSE error event), not an empty 200. |
| 8 | A compaction | It stays on the card (operator, 2026-09-30). Mapped onto the stored prompt it extends the slot. Sent as is (no match) it is a different prompt: f_keep < 0.5, so the server first SAVES the whole slot to `--cache-ram` (the full state: 12.45 KiB a token of K/V + 450 MiB of recurrent state + the draft's; ~0.9 GiB at 30K, ~2.4 GiB at 140K, derived from the buffer sizes; copied off the card), then reads the transcript from 0. The continuation is a third prompt: another save, another read. | A harness compaction at ~100K: time, `cache_n`, commit before/after, and that the continuation reuses the summary turn. |
| 9 | `--cache-ram` 8,192 MiB (the default; not set in the entry) | Up to 8 GiB more commit, filled only when conversations alternate on the slot. On a box with 22 GB of commit left after the pinned load it fits; with imagegen or vision loaded it may not. An entry larger than the limit is skipped (logged), the oldest is evicted at the limit. | Two alternating conversations: commit growth, and whether the returning one is restored (`found better prompt`) or re-read. Consider an explicit `--cache-ram`. |
| 10 | 32 checkpoints | Each is 112.6 MiB + 2.02 KiB a token (145 MiB at 16K, 178 at 33K, 390 at 140K). Every request adds up to two near its end, exempt from the 8,192 spacing; the list is thinned only when full. Worst case 32 x the size at the slot's length: 5.6 GiB at 33K, 12.2 GiB at 140K, 19.7 GiB at 262K -- host commit. An allocation that fails is an uncaught bad_alloc: the server dies. | 40+ agent steps at ~100K: the count, the bytes (the `created context checkpoint` lines), commit. Consider `--ctx-checkpoints 16`. |
| 11 | A context near 262,144 | The proxy refuses a prompt that leaves less than its floor (400 context_length_exceeded -> the harness compacts). The server has `--no-context-shift`: a generation that reaches the window ends with `length`. MTP drafts up to 3 tokens ahead: the verify batch needs room past the last token. | Fill to within 4K of the window, then generate: a clean `length`, no fault. |
| 12 | The MTP draft on a divergence or restore | The draft context's state and the speculative state are saved in every checkpoint and prompt-cache entry and restored with it (`load_dft`, `common_speculative_set_state`); a restore whose size does not match logs "failed to restore state" and the slot is cleared (a full re-read, not a fault). | After rows 4, 6 and 9: draft acceptance stays in its usual range (0.6-0.9), not ~0. |
| 13 | A runaway turn | At max the proxy sends the whole window as the budget (`max_tokens` 261K, reasoning budget 259K, the nudge at 0.6 of it = 155K tokens). A model that keeps thinking gets the nudge after ~1.5 h and the hard stop after ~2.6 h at 28 tok/s; heartbeats keep the client connected, so it looks like a stall. | The longest reasoning in the soak; decide a per-turn ceiling for flash-next (an operator number). |
| 14 | A second conversation / a side call while flash-next holds the card | No `other_card` for flash-next: a second conversation gets 503 `conversation_at_capacity` with Retry-After (the owner's 60 s hold). Side calls and jjava go to bonsai-a4000 -- whose working set was paged out to ~50 MB by the pinned load, so its first answer pages back in. | The side call's latency right after a flash-next load, pinned vs not; the second conversation's 503 and its retry. |
| 15 | An idle gap | `ttl: 0`: never unloaded; the slot keeps its cells. The watchdog polls `/health` (served by the HTTP thread during a decode). Nothing expires -- the "reused 0 after 2 h" was row 4. | A 30-minute gap, then the next step: `cache_n` as in row 5. |
| 16 | The 1,024-token class of fault (0022) | Fixed for the MMQ src1 padding. Other first-use faults would show the same way: a 200 with 0 bytes (row 7) and an Xid 13 in the system log. | The soak's server log: no "CUDA error"; the system log: no nvlddmkm 13/153. |

## 12. Layer-major prefill (design; 2026-10-02; prototype built offline as patch 0029, NOT pinned, NOT measured on the real model)

The prompt path runs at 140-160 tok/s (6.3-6.5 ms a token; the engine profile of 2026-10-02, deployed setting, ~25K-token
prompt, warm). This section reads where that time goes in the code, weighs the ways to cut it, and records the smallest
prototype that tests the recommendation. Every number is labelled: measured (script / log), Strata's, llama.cpp's own, or
derived (arithmetic shown). Line numbers are the vendored `llama-upstream-flash-int0028` tree, i.e. 0001-0028, before 0029.

### 12.1 Where the time goes, from the code

- **Who decides the upload.** `ggml_backend_sched_backend_id_from_cur` (`ggml/src/ggml-backend.cpp` :1046, cause "1.off") sends an
  op whose weight is in host memory to the GPU when `ggml_backend_cuda_device_offload_op` says so: for MUL_MAT_ID that is
  `op->ne[2] >= 32` tokens (`ggml-cuda.cu` :5831-5850; `GGML_OP_OFFLOAD_MIN_BATCH`, default 32, :6079). A ubatch of 512
  tokens always qualifies; a batch's last ubatch of < 32 tokens does not (its experts run on the CPU, as today).
- **One split per offloaded weight.** The split loop starts a new split at a node whose weight is on another backend when the
  running split already has inputs (`need_new_split`, :1399-1418), and the previous offloaded weight is such an input: so each of
  a layer's MUL_MAT_IDs (gate, up, down) is the FIRST node of its own split. A generated qwen4exp with separate gate/up
  confirms it (below): in the normal path every expert weight takes the used-experts copy, none the whole-tensor one.
- **The copy.** `ggml_backend_sched_compute_splits` (:1807), input loop :1921, the block at :1982-2070: for a split whose first node is
  a MUL_MAT_ID with a host weight it reads the ids back (`ggml_backend_tensor_get_async` + `ggml_backend_synchronize`, :2014-2017: a
  host-waits-for-GPU round trip per MUL_MAT_ID, 3 x 48 = 144 a ubatch), marks the used experts, and copies runs of consecutive
  used experts (`copy_experts`, :2034-2046, + 512 bytes of padding for MMQ). **So it copies only the used experts** -- but with 512 tokens
  x top-10 of 512 experts the chance a given expert is unused is (1 - 10/512)^512 = 4.1e-5 (0.02 experts a layer under uniform
  routing; real routing is skewed, so some), i.e. nearly all. A weight that is not the first node of its split would be copied
  whole and synchronously (:2071-2083); none of the experts is.
- **The bytes.** `blk.N.ffn_{gate,up,down}_exps` of the IQ2_XS file (read from both shards): 35,454,976,000 B = 33.02 GiB = 35.45 GB
  over 48 layers, 704.4 MiB a layer on average, 737.5 MiB the largest (gate 256.25 IQ2_S, up 256.25, down 225.0 Q2_0); the
  largest single tensor is 268,697,600 B (256.2 MiB). The loader's "CPU model buffer size = 33,812.50 MiB" (section 11.1) is the same number.
- **Arithmetic against the profile.** 2,348 ms of a 3,124 ms ubatch are outside the timed nodes (measured, engine profile);
  35.45 GB / 2.348 s = 15.1 GB/s, against Strata's 14.1 GB/s H2D probe on this card at x8 (section 10.2: 35.45 / 14.1 = 2.51 s). The remaining
  776 ms is the GPU's own work (1.6 ms a token x 512 = 819 ms, same profile). Taken together: a ubatch costs about C + U with C = 776 ms of
  compute and U = 2,348 ms of expert upload (and its 144 syncs), and U does not depend on how many tokens the ubatch holds -- it is
  paid per ubatch. (Section 8 reports 231-318 tok/s for cand0022 and section 11.1 131-158 for cand0023 from other scripts; the
  model below is calibrated on the profile: 512 / 3.124 s = 164 tok/s.)
- **What a ubatch needs from the memory.** `decode` (`llama-context.cpp` :1765) takes the batch's ubatches from `memory->init_batch` and runs
  `process_ubatch` (:1403) on each: `mctx->apply()` (the KV cells: `llama_kv_cache::apply_ubatch` `llama-kv-cache.cpp` :1119; the
  recurrent head/state-copy ids: `llama_memory_recurrent::find_slot` `llama-memory-recurrent.cpp` :505, `s_copy` :1346, which also
  consumes a sequence's rollback index), then builds the WHOLE model's graph (qwen4exp.cpp :402-521, layer loop :443) over those
  cells and runs it. The Gated DeltaNet state and the K/V are per layer, so a layer needs only its own earlier ubatches to have run.

### 12.2 What Strata does (`Niko1221/Strata` @ d551edf, `C:/Users/jwals/octo/strata-src`)

`src/prefill/prefill.cpp`: the prompt is cut into chunks of `m.T` tokens (8,192 in the claim; :752 `for (c0 ...; c0 += m.T)`); per chunk the
embeddings and PLE rows of the WHOLE chunk are made first (:759-790), then **layers are the outer loop** (:855) and each layer runs
its attention (GDN or QSA) and its MoE over the whole chunk with the experts of that layer streamed through a ring of expert
slots (:798-826 builds the chunk's stream in layer-then-expert order; :854 primes the ring before layer 0; the copy of entry k+ring is
issued as entry k is consumed, :1284-1312). The wide residual stays on the device between layers (one buffer set, ~680 KB of scratch a token
of which the attention and MoE halves share ~260 KB, :355-359: 5.6 GB at 8,192 tokens, derived, their VRAM budget). Their measurement (Strata's, n=1):
Q2_0, 8,192-token chunks, ring of 96 slots 1,153 tok/s, 384 slots 1,294 (:73-76: "the next layer's experts arrive during its attention half");
the ring only engages from `STREAM_ALL_MIN` = 2,048 tokens (:71), because below that most experts are not routed. Each expert
crosses PCIe once per chunk: that is the whole mechanism, and it is option (a) with the chunk as one batch per layer.

### 12.3 The options

Quantities used below, all from 12.1: U = 2,348 ms, C = 776 ms (per 512-token ubatch), N = ubatches per batch.
Per token: no overlap `(N*C + U) / (512*N)`; the upload fully hidden behind compute `max(N*C, U) / (512*N)`.

| N | no overlap: ms/token, tok/s | upload hidden: ms/token, tok/s |
|---:|---|---|
| 1 (today) | 6.10, 164 | 4.59, 218 |
| 2 | 3.81, 263 | 2.29, 436 |
| 4 (`-b 2048`, the deployed batch) | 2.66, 376 | 1.52, 660 |
| 8 | 2.09, 479 | 1.52, 660 |
| 16 (`-b 8192`) | 1.80, 555 | 1.52, 660 |
| infinity | 1.52, 660 | 1.52, 660 |

(The engine agent's ~0.3 ms a token of upload at 8K chunks is the N = 16 column: 2,348 / 8,192 = 0.29 ms. Not measured.)

**(a) Layer-major execution.** *Changes:* the batch's ubatches go through the layers layer by layer: layer 0 for every ubatch, then layer 1, ...;
a layer's experts are uploaded once and read by all its ubatch graphs. *VRAM:* the residual between layers, 2 x 40 KiB a token (hc 4 x n_embd
2,560 x 4 B, ping-pong by layer parity): 160 MiB at `-b 2048`, 640 MiB at 8,192 (derived); the expert slots, 3 x 256.2 = 769 MiB for one
layer, 6 x = 1,537 MiB with the next layer's upload in flight (derived from the tensor sizes); the compute buffer stays the ubatch's (no growth).
*Host:* none beyond the `-b`-sized buffers a larger batch implies (the MTP target's unmasked hidden-state rows: 40 KiB a token, 320 MiB at 8,192).
*Correctness:* below. *Estimate:* the table, N = 4 to 16: 376-555 tok/s without overlap, up to 660 with it (2.3-4x today).
**(b) A bigger single ubatch.** `-ub 1024/2048/4096` has the same arithmetic (N = ub/512) and no new code. Its compute buffer
grows 3.708 MiB a token: 1,871.61 MiB at 512 and 3,770.23 MiB at 1,024 (`C:/Users/jwals/octo/flashnext-gate-20260929/vram-cache48.log`,
`vram-cache48-ub1024.log`: 3,770.23 - 1,871.61 = 1,898.62 MiB a 512 tokens, and the intercept is -27 MiB: it is all proportional) -- 14.8 bytes a
cell a token at the reserved n_kv = 262,144, which is the worst-case `build_qsa_top_k` / `build_attn_qsa` graph (qwen4exp.cpp :952-968, :1020-1055):
`expanded` [n_kv, n_tokens] f32 and its permuted copy, the f32 cast of the mask, the mask sum, `kq_mask_all` and `kq_mask_top_k`, ~3.7 f32 tensors
live at the peak (derived). It is a reservation at n_ctx, not what a 25K prompt uses. **It can be avoided without changing results**: queries are
independent rows, so the n_kv-wide part can be built in token slices that reuse one buffer (a 128-token slice is 475 MiB at any ub);
`LLAMA_QSA_BLOCK_TOPK=1` removes `expanded` but not the masks. What remains is the rest of the layer's per-token scratch, which is not
measured (the graph's MoE and DeltaNet tensors add up to roughly 0.5 MiB a token, derived, order of magnitude: ~1.1 GiB at 2,048, ~2.2 GiB at
4,096). At the deployed 262K context `-ub 2048` is +5.6 GiB, `-ub 4096` +13.0 GiB (a 14.8 GiB compute buffer; section 9's "14.5 GB"): impossible as it is, and
the card's lowest free is 1,289 MiB. *Estimate:* the table, N = 4 or 8, no overlap (the upload cannot overlap its own ubatch's compute).
**(c) Make the 0002 ring hide the upload.** The ring (`--prefetch-experts-slots`, upstream #28414, `ggml-backend.cpp` :1837-1883) uploads the
next weights while the current split computes. A layer's compute is 776 / 48 = 16.2 ms, its upload 2,348 / 48 = 48.9 ms: the ring can hide the
compute behind the upload, not the upload behind the compute, so it is bounded by max(C, U) = 2,348 ms a ubatch = 512 / 2.348 = **218 tok/s
(+33%)**, and it copies whole tensors (no saving where all experts are used). Built, never measured in the combined arm (section 10.4).
**(d) Anything simpler in Strata.** No: Strata's order IS (a) (12.2); its extra is the ~5.6 GB of per-chunk scratch that lets the whole chunk be one batch
per layer. Our version keeps the ubatch at 512 and pays for the order instead with per-layer graphs (0.04 ms to build, 0.04 to split and allocate,
A4000, tiny model, n=1 each: ~0.1 ms a graph, 768 graphs an 8K batch = ~84 ms, under 1%).

### 12.4 Recommendation

**(a), at `-b 2048` first (N = 4) and `-b 8192` (N = 16) second, with the 0002 ring left off.** It is the only option that amortises the upload at the
deployed 262K context without growing the compute buffer, its cost is a fixed ~0.9 GiB (3 slots + 160 MiB hidden) to ~1.7 GiB (6 slots, overlap)
that comes out of the expert cache (66.2 MiB a slot row: 14 to 26 of its 70 rows; section 8's simulation puts 46 rows at a 56.4% hit rate against the measured 63-64% at 70: a decode trade for the operator, derived), and N
is a flag (`-b`). (b) is the smaller change but needs the QSA slicing AND ~1-2 GiB of per-token scratch before it can leave 512; (c) is capped at +33%.
**The smallest first step that proves the gain** needs no engine code: a prefill at `-ub 512/1024/2048 -c 16384` (the QSA reservation is then
475 MiB at 2,048 tokens), n=3 warm, on the 5060 Ti. The model predicts 164 / 263 / 376 tok/s (N = 1, 2, 4, no overlap): if it holds, the
upload is the cost and (a) removes it at any context. Then the prototype below at the deployed layout.

### 12.5 What was built: patch 0029 (`engines/patches/llama-upstream-flash/0029-llama-layer-major-prefill.patch`, sha256 dfe4ed040c90eb0b01803f653bbd4b1d7892abbccd55dcd960a728be2c8e9a46; not in the manifest)

Dev tree `C:/Users/jwals/engines/dev-layermajor` (int0028 + the patch; the base and the patch are two commits), built on the E-cores. The patch applies with
`git apply --cached` on the repo's 0001-0028 (the result differs from int0028's tree only in `tests/test-moe-pcie-split.cpp` and
`tests/test-moe-refill.cpp`, which the patches have moved on from). **Switch:** `LLAMA_LAYER_MAJOR` unset or 0 = today's code (the
`GGML_SCHED_DEBUG=2` dump of `llama-bench -p 192 -n 4 -ub 64` on the A4000 with a generated qwen4exp is identical to int0028's: 454 splits, 18,560
split/node lines after masking sizes); 1 = on where it gains (a GPU, experts in host memory, op offload, no `--prefetch-experts-slots`);
2 = wherever it can run (the CPU test). `LLAMA_LAYER_MAJOR_SLOTS` (6; 3 = no overlap), `_MIN_UBATCHES` (2), `_DEBUG=1` (a line a batch: ms, MiB uploaded,
hits, prefetched, host time a layer graph). qwen4exp only; the MTP context type, embeddings/pooling, backend samplers and layer-input taps take the old path.
- **Execution.** `llama_context::decode_layer_major`: every ubatch is applied to the memory once, in order, with `lm_record()` keeping what its
  graph reads (the K/V and indexer cell range; the recurrent head, rs_z and every `s_copy` id, incl. the one-shot rollback index);
  then for each layer, for each ubatch, `lm_seek(i)` puts that back and `process_ubatch_range` builds, allocates, feeds and runs the graph of ONE layer
  for it (never reused). Layer L of ubatch i therefore sees the cells and states the ubatch-by-ubatch path saw; ubatches after i may already be
  in the cell array and are masked by position. `finish_ubatch` (logits, embeddings, `h_nextn`) is the old code after the last layer.
- **Graph.** `llm_graph_params` carries a layer range; `qwen4exp::graph` starts a range past layer 0 from rows of a persistent device buffer and ends it by
  copying the residual into the other of two ping-pong buffers; inputs a range does not read are not created.
- **Scheduler.** `ggml-backend-lm.h`: weight-stationary slots. A split whose MUL_MAT_ID reads a host weight finds the tensor whole in a slot or uploads
  it whole on a second stream (event to the compute stream); the driver names the next layer's weights after the first graph of a layer so the upload overlaps the
  rest; a slot of the current layer is never reused. The ids readback and its sync are gone on a hit. Counters for the used-experts and whole-weight copies are always kept.
- **Tests** (`tests/test-layer-major.cpp`, ctest `test-layer-major`; a generated qwen4exp: 8-12 layers, Gated DeltaNet + QSA, hyper connections, PLE, 16-64
  experts, `tests/split_gate_up.py` makes the separate gate/up of the real files). The same batches in two contexts, the second with the switch on: a prompt
  batch with an output on every token, a server-style batch (the last token only), an optional rollback of K positions with `n_rs_seq` snapshots
  and a batch after it, generated tokens; logits, the serialized sequence state (K/V, indexer cells, recurrent state) and the MTP hidden states compared.
  **CPU: 48 runs (6 seeds x 8 shapes: ubatch 20-128, 1-3 sequences, top-k 24-131,072, f16/q8_0 K/V, `n_rs_seq` 3-4, a batch cancelled by the abort callback) all
  bit-exact, state byte-identical; A4000 (CUDA0, experts in host memory, f32 and Q4_0, fused and separate gate/up, 3-6 slots): 19 runs, also bit-exact** (NMSE 0; the
  limit of test-llama-archs is 1e-4). A control the comparison must fail -- the ubatches of every layer past the first taken back to front -- fails it in
  every run. On the A4000 a 401-token batch (6 full ubatches) copied 288.0 MiB of experts ubatch by ubatch and 48.0 MiB layer-major (24 uploads, 21 prefetched, 141 hits):
  one ubatch's worth, as N = 6 predicts. These are correctness evidence on a small model: **no speed number from the real model or the 5060 Ti exists.** The
  A4000 lane was already held by the coordinator; the tests ran for under a second each on `CUDA_VISIBLE_DEVICES` = the A4000's UUID, nothing was unloaded.

### 12.6 What the GPU gate must measure

1. **Off is off:** `GGML_SCHED_DEBUG=2` dump on the real model with the switch unset against cand0022/0023's (the claim for "off"), then KL (batch 4) and needles / corrupt with `-b 2048 LLAMA_LAYER_MAJOR=1`
   (`llama-perplexity` at `-b 2048` goes through decode_layer_major): the real model's QSA/indexer path at 262K is tested only at 2^14 cells here.
2. **Prefill speed, n=3 warm and cold** at ~1K / 8K / 32K, switch off vs on at `-b 2048` and `-b 8192`, deployed layout and flags, against 164 tok/s (the profile) and the 231-318 / 131-158 of sections 8 and 11.1; decode unchanged (1-token batches never take the new path). The pass rule from the model: >= 1.8x (>= 300 tok/s) at `-b 2048`
   proves the upload amortises; between 1.2x and 1.8x, profile before concluding.
3. **Does the upload overlap?** `LLAMA_LAYER_MAJOR_SLOTS=3` (none) against 6; the 0017 profile (sched `copy` / `other` / GPU wait) with the switch on. 0024's A4000 run found that on WDDM the compute stream waits for work queued on a second stream (a step took
   ~0.9 s behind 600 x 256 MiB device copies): the first thing to look at. With `LLAMA_PIN_EXPERTS=0` the prefetch copies read pageable memory and block the calling thread for the copy (~25 ms a 256 MiB tensor at 11 GB/s): expect the no-overlap column. **Confirmed by the gate (13.3):** 6 slots with prefetch read 433.1 / 443.4 / 434.6 tok/s, 3 slots 434.1 / 433.1 / 438.7 (n=3 each); and 0024's observation was device-to-device copies, not pageable memory (13.3).
4. **VRAM:** lowest free MiB with 3 and 6 slots at `-b 2048` / `8192` (the log line `layer-major prefill: hidden-state buffers ... expert slots ...`), and the cache slot rows that still fit; decode tok/s and hit rate at that row count.
5. **MTP and checkpoints:** draft acceptance after a layer-major prompt (the `h_nextn` rows) in its usual 0.6-0.9; `--checkpoint-every` (0023) checkpoints taken after a layer-major batch restore bit-identically; a client hang-up mid-batch (the cancel is taken between decode calls: ~15 s of prefill at 8,192 tokens and ~550 tok/s) and the retry.
6. **The tail:** the last ubatch of < 32 tokens runs its experts on the CPU (as today) and the last layer's FFN sees only the output rows (one token in a server batch): the chunk's last ~1/48 layer and the tail cost, in ms.

### 12.7 Risks

- **The gain is an estimate**; the model has one calibration point (N = 1 -> 164 tok/s) and assumes U is pure copy and fully paid per ubatch. If routing is more skewed than assumed, U is smaller and the gain too.
- **VRAM**: +0.9 to +1.7 GiB against 1,289 MiB free means fewer expert-cache rows, a decode cost the operator decides. The compute buffer still reserves the scheduler's own copy of each offloaded weight (up to 737 MiB of the 1,871 MiB, derived) although a slot-backed split never touches it: a follow-up could give those copies no space.
- **Overlap is unproven** (WDDM, pageable memory, above). Without it the numbers are the left column: still 2.3x at `-b 2048` and 3.4x at 8,192 over today.
- **A larger `-b` is a larger unit of work**: a cancelled batch costs a whole chunk, and a cancel in the middle of a chunk that did not start at position 0 leaves recurrent state that cannot be rolled back (as today within a ubatch).
- **Maintenance**: the range graph is qwen4exp-only code in `qwen4exp.cpp` and `llama-graph`, and three new virtuals on the memory context; an upstream change to that graph or to the hybrid memory has to keep `lm_record` / `lm_seek` and the range builder in step (the test catches a divergence, on the CPU, exactly).
- **PLE** (0028): the gather of ubatch i's rows now happens at layer 1 after layer 0 of the whole batch; the prefetch thread's "later ubatches first" order is unchanged, the first ubatch's cold rows are still on the critical path.

## 13. Port status 2026-10-06: the ledger, the stager (0030), the overlap question, and the next lever

Operator, 2026-09-29: "We must get strata speed or this model is of no value to me. And strata proves it can work." The work
stalled on 2026-10-02 when the priority moved to session stability and is resumed. Labels as everywhere: **measured** (script +
n), **Strata's** (theirs, n=1, RTX 5070 12 GB), **derived** (arithmetic shown). Strata's source is `Niko1221/Strata` @ d551edf
(`C:/Users/jwals/octo/strata-src`, MIT, `engines/patches/llama-upstream-flash/LICENSE.strata`). Raw results are in
`C:/Users/jwals/octo/flashnext-gate-20260929/` (`gate.json` and the per-arm `*.server.log`) and `C:/Users/jwals/octo/fn-crash/`.

### 13.1 Section 9's rows, one by one

Status words: **SHIPPED** (in the manifest's series and in the running entry), **MEASURED** (built, run on the card, a number
below), **BUILT** (compiled and tested offline or on the A4000's unit tests only, never gated), **NOT STARTED**, **N/A**.
The deployed entry (config.yaml, 2026-10-05) is `llama-upstream-flash-cand0029` = 0001-0018 + 0022 + 0023 + 0029, `-np 1 -c 262144`,
`-b 8192 -ub 512`, `--moe-expert-cache 50`, `LLAMA_PIN_EXPERTS=0`, `LLAMA_LAYER_MAJOR=1`, `LLAMA_LAYER_MAJOR_SLOTS=3`, MTP 3 drafts.

| # | Strata mechanism (section 9) | status | evidence |
|---|---|---|---|
| 1 | MTP draft layer, 3 drafts | **SHIPPED** (0006) | window C/D speed tables (section 8): acceptance 0.7-0.85, decode 28-31 tok/s deployed (`pin_compare.py`, n=3: 28.2-31.0 at ~4K) |
| 2 | Profile-ranked VRAM expert cache + adaptive tier | **SHIPPED** (0003/0004/0007/0013/0016) | window D `cache-all-np1` hit rate 63-64% at 70 rows (n=3); at the entry's 50 rows (`pf-lm3-b8k` speed probe, n=1) 53.9% at ~4K |
| 3 | CPU AVX2 multi-token i-quant rows | **SHIPPED** (0009) | `moe_cpu_bench`, n=300; the operator's KL decision 2026-09-29 |
| 4 | Q2_0 CPU kernel | **SHIPPED** (0008) | down layer 0.575 -> 0.118 ms (n=300) |
| 5 | E-2 AVX-512 row prefetch | **N/A** | the CPU is AVX2 only; ~2% in Strata's own claim |
| 6 | Pinned expert arena + helper copy threads | **SHIPPED as code, OFF in the entry** (0005) | `LLAMA_PIN_EXPERTS=0` since 2026-10-05: pinning cost 36.4 GB of commit and ~78 s of load (11.1, `pin_compare.py` n=3) |
| 7 | Windows large-page arena | **BUILT**, never run | needs "Lock pages in memory" (a system policy we do not change) |
| 8 | Expert streaming ring (0002, upstream #28414) | **MEASURED, no gain, not used** | `pf-ring` (2 staging slots, 62 rows) 149.4 / 141.9 / 147.5 tok/s against `pf-dep` 138.4 / 159.7 / 158.7 and `pf-int-off` 142.2 / 145.8 / 148.0 (n=3 each, ~25.6K prompt, `gate.json`); `--prefetch-experts-slots 0` in the entry |
| 9 | MMQ experts in the prompt path | llama.cpp's own | `GGML_CUDA_OP_TIMING` table, `pf-dep-opt` (23k prompt, n=1 run): MUL_MAT_ID iq2_s 26.5% + q2_0 20.0% + iq1_m 10.1% + iq2_xxs 6.0% = 62.6% of GPU op time at 512-token ubatches: the lever of 13.6 |
| 10 | Larger prompt chunks (8K) | **SHIPPED** as `-b 8192` over 0029 | `pf-lm3-b8k-c29`; `-ub 1024` alone (`pf-ub1k`, 41 rows): 254.2 / 257.8 / 260.7 tok/s against 142-148 at `-ub 512`, needles 4/5 (one answer a refusal; the arm's PASS false), lowest free VRAM 397 MiB: not taken |
| 11 | Batched PLE gather | **BUILT, no gain measured** (0028) | `pf-ple` 157.9 / 157.3 / 156.3 against `pf-dep` 138-160 (n=3, warm; the cold-row case it was written for is not what that arm measures) |
| 12 | D-1 QSA prompt attention on tensor cores | **BUILT, no gain measured** (0027) | `pf-qsa1` 155.4 / 154.8 / 153.8, `pf-qsa2` 150.2 / 155.9 / 156.2 (n=3) -- the upload dominated then; FLASH_ATTN_EXT is 6.6% of GPU op time at 23k, so 6.6% is the ceiling; test-backend-ops FLASH_ATTN_EXT on the A4000 with the switch on (1, 2, poisoned) passes (`fn-crash/a4000-fa-on1-poison.log`, `-on2-poison.log`); needles/corrupt on the real model not run |
| 13 | QSA select (decode half) | **SHIPPED** the decode half (0011, 0014, 0015, 0018) | window B/D; 0019 (13.2) is the n_kv-sized remainder |
| 14 | C-1/C-2 select grid, chunked indexer appends | **NOT STARTED** | prompt side; the profile does not rank it |
| 15 | D-2 GDN recurrence split, C-3 short conv | **NOT STARTED** | GATED_DELTA_NET is 6.2% of GPU op time at 23k (`pf-dep-opt`); Strata's own gain is 1,258 -> 1,308 tok/s (4%) |
| 16 | D-4/D-5 queued refills, stream issuer | **BUILT** refills (0025); the stager is now 0030 | 0025: unit tests only (13.2); 0030: 13.5 |
| 17 | E-6 device plan (a layer whose experts are all resident skips the host) | **NOT STARTED** | needs a cache that holds whole layers |
| 18 | Batched verify-window kernels | llama.cpp's own | MTP verify batches of 4: window D profile |
| 19 | PCIe share (`pcie_frac`) | **BUILT** (0024) | unit tests only (13.2); the arithmetic says little is on the table (10.2) |
| 20 | KV streaming `--kv-resident` | **PART**: 0012 SHIPPED as code (off: `LLAMA_KV_HOST_MAPPED=0`); 0021 **MEASURED, slower**; 0026 **BUILT** | 0021, `kvcache-fit-np1` at 118 rows: **13.0 / 15.1 / 18.0 tok/s** at 4.4K / 32K / 128K against `cache-fit-np1`'s 26.9 / 28.5 at 4.4K / 32K (n=1 probes, `gate.json`): the mapped K/V halves the decode, which is 10.1's trigger for the page window (0026, never run on the real model) |
| 21 | q4_0 KV + Hadamard | **MEASURED, refused** | `cache-all-np1-q4kv`: KL 0.026247 against 0.0110, needles 13/15 (section 8, window D) |
| 22 | Prompt-lookup drafter beside MTP | **NOT STARTED** | |
| 23 | Conversation cache | **SHIPPED** (server slots, `--cache-ram`, 0023) | `ckpt-np1`; `pf-lm3-b8k` ckpt: B reused the 16,384-token boundary |
| 24 | Layer split over several GPUs (`flash-a4000`) | arm **defined, never run** | no entry in `gate.json` |
| 25 | Bulk expert-arena reads on MSVC | **N/A** | llama.cpp mmap |
| 26 | Refusal-direction speed projection | **NOT TAKEN** | a behaviour change |

What section 9 did not list and 0029 added: **layer-major prefill** (Strata's `prefill.cpp:752/:855` order) -- **SHIPPED**, the one big win:
~25.6K prompts **433.1 / 443.4 / 434.6** tok/s (`pf-lm6-b8k`) and **434.1 / 433.1 / 438.7** (`pf-lm3-b8k`, the entry's setting) against
154-160 without it (n=3 each, `gate.json`); KL 0.011379 / 96.458% same top (batch 8192, the yardstick 0.010993 / 96.757%), needles 5/5
at 32K, corrupt clean, `--checkpoint-every` restore exact, cancel mid-prompt clean. After the deploy, warm and unpinned
(`bench/results/fn_first_prompt/20261006-a/summary.txt`, n=3, the proxy's own 24.6K prompts): 8,192-token batches take 16.0-16.9 s
(485-512 tok/s) as a prompt's first batch, 17.5-18.5 s as its second; the whole prompt 58.4-61.3 s (400-422 tok/s).

### 13.2 Patches 0019-0030, one by one

| patch | what | status | evidence |
|---|---|---|---|
| 0019 | `LLAMA_QSA_BLOCK_TOPK=1`: QSA budget as whole blocks (M2b) | **BUILT, half-gated**, not in the series | KL **0.010651 / 96.373%** (`cache-all-np1-blk`, batch 4, against 0.011094 / 96.532%; `kl-cache-all-np1-blk-b4.log`); the needles/corrupt/speed run of window E2c was killed by the gate's guard (llama-swap started `bonsai` on the card, `runE2c.out`): **needles, corrupt and speed never ran**. A4000 numbers in the patch text only |
| 0020 | `LLAMA_GRAPH_CACHE=N` | **MEASURED, inconclusive**, not in the series, off | window E1 (section 8): 31.1 vs 29.6 tok/s at 4.4K, 38.0 vs 38.4 at 35K, 33.0 vs 29.2 at 68K, 25.4 vs 29.7 at 140K (n=3 each); the profile's launches fell 15.6 -> 3.6 ms and the GPU wait grew by as much; coordinator 2026-10-01 "no graph cache" |
| 0021 | `LLAMA_KV_HOST_MAPPED=1` maps only the sparse layers | **MEASURED, slower**, not in the series | row 20 (13.0 / 15.1 / 18.0 tok/s, 118 rows, n=1) |
| 0022 | MMQ src1 padding (the 1,024-token crash) | **SHIPPED** | test-backend-ops MUL_MAT_ID 932/932 on the 5060 Ti; the fresh 1,024-token request serves |
| 0023 | `--checkpoint-every` | **SHIPPED** | `ckpt-np1`: a prompt sharing 22,344 tokens reused 16,384 (0 without) |
| 0024 | `LLAMA_MOE_PCIE_FRAC` | **BUILT**, unit tests only, not in the series | CPU `test-moe-pcie-split` 60 trials bit-exact (patch text). The only saved A4000 log (`fn-crash/a4000-test-moe-pcie-split_--device_CUDA0.log`) shows 74 of 25,567 checks failing on a NaN; 10.2 and the patch state a later run with 0 failed (the failure was the test's, not the engine's) but **no log of that run is in the tree**. Speed never measured |
| 0025 | `LLAMA_MOE_CACHE_QUEUED_REFILL=1` | **BUILT**, unit tests only, not in the series | CPU `test-moe-refill` 846 checks, 0 failed (`a4000-test-moe-refill.log`, which says "no CUDA-capable device": the CPU run). The only saved `--device` log (`..._--device_CUDA0.log`) shows 1,252 checks with 1 failed ("no poll ever found a batch still in flight"); the patch text's 1,411 checks, 0 failed has **no kept log**. No GPU gate (`refill-np1` is not in `gate.json`) |
| 0026 | `LLAMA_KV_PAGE_WINDOW` (Strata's M3) | **BUILT**, unit tests only, not in the series | `test-kv-page-window` on the A4000: 60 checks, 0 failed (`a4000-test-kv-page-window.log`); CPU model test 12,000 calls, 0 failed. `kvpage-fit-np1` / `cache-all-np1-kvpage` never ran; the patch text says the window "has not been measured" |
| 0027 | `GGML_CUDA_QSA_PROMPT_ATTN` (D-1) | **BUILT, no gain measured**, not in the series | row 12 |
| 0028 | `LLAMA_PLE_PREFETCH` | **BUILT, no gain measured**, not in the series | row 11 |
| 0029 | layer-major prefill | **SHIPPED** (cand0029, deployed 2026-10-05) | 13.1 |
| 0030 | the staging ring for pageable expert uploads (`LLAMA_STAGER=1`) | **BUILT, CPU-tested, NOT in the series, NOT measured** | 13.5 |

The vendored tree (`engines/src/llama-upstream-flash`, 21 patches) and the manifest series are 0001-0018, 0022, 0023, 0029. 0019-0021 and
0024-0028 are in `engines/patches/` and were built into `llama-upstream-flash-int0028` (0001-0028, `C:/Users/jwals/engines/`) for the
offline tests and the `pf-*` arms above; **no shipped build contains them**. 0030 is written against the shipped series, not against
them.

### 13.3 What the data say about the upload (derived; the model behind 13.4-13.6)

**The prefetch of 0029 did nothing, and the reason is in the data.** `pf-lm6-b8k` (6 slots: the next layer's three tensors are queued after the
layer's first graph, **141 of 144 uploads prefetched**) and `pf-lm3-b8k` (3 slots: every upload is a demand upload) read the same, **433.1 /
443.4 / 434.6** against **434.1 / 433.1 / 438.7** tok/s (n=3 each); per 8,192-token batch, first batch of a prompt, `LLAMA_LAYER_MAJOR_DEBUG`
lines: lm6 15,229 / 14,876 / 15,023 ms (the first batch of each of three prompts), lm3 14,655-15,772 ms (the first batches of ten prompts) (`pf-lm*-b8k.server.log`). What did move: the host time of
the layer graph's "compute call", 19.1 ms (lm3) -> 16.1 ms (lm6), because the blocking copy left the compute call and went to the prefetch call. **With the
experts unpinned the copy is a `cudaMemcpyAsync` from pageable memory; the driver stages it and the calling thread is blocked for the whole
copy** (CUDA's documented behaviour for pageable sources), so after the first graph is launched the host sits in the prefetch call for the layer's whole upload and launches nothing, and the GPU
idles behind its one running graph. A pinned source would not block; that arm (`LLAMA_PIN_EXPERTS=1` with 6 slots) was never run: every layer-major arm used the
unpinned `dep` setting.

**Calibration (derived from the first batch of each prompt, which has no context, so attention does not grow the numbers).** A layer-major batch of N ubatches of 512 costs
T(N) = N x C + U, with U the whole-batch expert upload (35.45 GB = 33.02 GiB, 12.1) that nothing hides.
lm3 (3 slots): N = 4 (`pf-lm3`, the first batch of each of three prompts) 5,605 / 5,778 / 6,012 ms and N = 16 (`pf-lm3-b8k`) 14,655-15,772 ms (median ~15,100), so C = (15,100 - 5,800) / 12 = **~780 ms** per
512-token ubatch (1.52 ms a token; the engine profile's C was 776 ms) and U = 5,800 - 4 x 780 = **~2,600 ms** (2.57-2.70 s depending on the pairing) = 35.45 GB / 2.6 s = **~13.6 GB/s**, the rate of
Strata's pinned-copy probe of this card (14.1 GB/s at x8, 10.2): the pageable path through the driver is not slower than pinned DMA, it is **blocking**.
(A two-point fit from 3-10 first batches each: +-5%.) Over the 25.6K prompt the batches grow with context: 15.1 / 16.2 / 17.5 s (`pf-lm3-b8k`, the three batches
of every prompt), the extra being attention.

**What hiding the upload is worth** (a batch of 16 ubatches, no context): 16 x 780 ms = 12.5 s of compute, plus 2.6 s of upload = 15.1 s now (8,192 / 15.1 = 542 tok/s); with the upload
fully hidden ~12.5-12.6 s: **~652 tok/s, +20%** per batch (derived). The end-to-end prompt rate is lower than the batch rate
by the tail: the server runs a 24.6K prompt as 8,192 + 8,192 + 7,706 + **409 + 103**, and the last two take 4.0-4.5 s and 1.8-2.4 s
(`fn_first_prompt` per-batch lines, n=3): 6.0-6.9 s of 58.4-61.3 s, ~11%, because a batch under two ubatches does not take the layer-major
path and pays the whole upload (U ~ 2.6 s) for 409 or 103 tokens (a 103-token batch is above `GGML_OP_OFFLOAD_MIN_BATCH` = 32, so its experts go to the card). Not fixed here; the options
are an offload minimum near 200-300 tokens (the CPU computes prompt experts at ~87 tok/s, `pf-dep-nooffload`) for the 103-token piece, and not splitting the prompt there.

**The first prompt after a load** (`fn_first_prompt`, n=3 per arm): with the file cache cold the first 8,192-token batch is 75.0 / 76.0 / 87.3 s against 16.0-16.9 s warm: 31.3
GB read from disk at **~0.44-0.54 GB/s** (demand paging; derived: 31.3 GB / (75.0 - 16.5) s and / (87.3 - 16.5) s). The same disk reads sequentially at 2.25-2.39 GB/s (the arm's cached sequential pre-read, 39.23 GB in 17.0-17.4 s,
and its eviction pass, 66.42 GB in 27.6-29.0 s). Parallel reads ahead of need are therefore worth up to ~58-71 s on a cold first prompt, which the proxy-side pre-read
(built separately) and the stager's read-ahead (13.5) both address.

**Was "on WDDM the compute stream waits for work on a second stream" (10.2, 0024) about pageable memory?** No. That observation (a step took ~0.9 s behind
600 x 256 MiB **device-to-device** copies on the second stream, A4000) involved no host memory at all, and the same notes say "any device-synchronising call also
waits for the second stream", so the 0.9 s may be the step's own synchronise rather than a stalled compute stream; and a same-device cudaMemcpyAsync is
commonly executed by SM copy kernels, which would compete with compute on any OS (an assumption, not checked here). It says nothing about the **host-to-device DMA engine**, which is what a prefetch uses. So the
question is open, and it is the probe's first line (13.4): pinned H2D against a compute kernel, on this card, under WDDM.

### 13.4 The probe: does a copy on a second stream overlap compute here? (written and built; NOT RUN: the GPU is shared)

`bench/fn_probe/h2d_overlap.cu` (a standalone CUDA program), `build.bat`, `run_probe.py`. It copies 0029's structure: per "layer", `graphs` kernels of ~16.2 ms on the
main stream (the host syncs after each, as the engine's compute call does), and right after the first one the next layer's 3 x 256 MiB tensors are uploaded into the other slot
set on a second stream; the next layer's first kernel waits for the upload's event. The kernel is an ALU loop on all SMs, calibrated to the target time with events.
Per mode it reports T_c (compute alone), T_u (the upload alone), T_o (both), the host thread's time inside the upload call, copy GB/s, and the overlap efficiency
**E = (T_c + T_u - T_o) / (T_c + T_u - max(T_c, T_u))**: 1 = hidden, 0 = added. Modes: `pin` (a: pinned source, `cudaMemcpyAsync`), `pageable` (b: malloc),
`mmap` (b, the experts' real source: a mapped temp file), `stage-heap` / `stage-mmap` (c: the **engine's own `ggml-stager.cpp`**, 0030, compiled into the probe, with CUDA hooks written in the probe),
`register` (e: `cudaHostRegister` on the mapped range per layer, then the async copy -- an alternative to the ring if registering is cheap and works on a file view) and `d2d` (the 0024 observation).
A sweep over ring threads {1,2,3,4,6} x chunk {4,16,64} MiB x ring {8,16,32} gives the copy-only throughput the ring can feed. Three runs: `engine`, `nosync` (without the per-graph host sync) and `b2048` (4 graphs a layer, where the upload dominates).

How to read the answer: **E(pin) >= 0.8** says the card overlaps H2D DMA with compute and the unlock is "make the source async": **E(stage-mmap)** against it then says whether the ring achieves it
(and the sweep how many threads it costs); `register` is the cheaper way if it works. **E(pin) < 0.5** says this card and driver do not overlap at all: the stager is not the unlock, 13.5 stays as built but off, and the whole
remaining gain is 13.6's compute side. `d2d` explains 10.2.

Correctness, offline: `python bench/fn_probe/run_probe.py --selftest` runs the whole harness on a simulated device (threads and spin loops standing in for streams, a pageable copy that blocks its caller) and
checks that pinned and ring sources score E >= 0.8 and pageable and mapped ones < 0.5: **PASS, 5 checks** (pin 1.00, stage-heap 1.00, stage-mmap 1.00, pageable 0.22, mmap 0.22). The CUDA build compiles
(`nvcc`, sm_86 + sm_120). **Run time on the 5060 Ti: ~6 minutes for the three runs (~2.5 + ~2 + ~1; derived: 16 graphs x 16.2 ms x 9 layers x 5 reps = 11.7 s per mode, 8 modes, plus the copy-only passes and the sweep), 2.5 GiB of
VRAM, a 768 MiB temp file.** `run_probe.py --go` refuses unless the 5060 Ti has >= 4,096 MiB free (the stack's model unloaded through the operator's window; the script never touches a process, :1234, :11434 or config.yaml).

### 13.5 Patch 0030: the stager, ported (Strata `prefill.cpp:122-239`)

`engines/patches/llama-upstream-flash/0030-ggml-staging-ring-for-pageable-expert-uploads.patch` (sha256 `2b4c44414ba96b24ccf497ffad79b4eeee1409d58eb4a7013f1c7d1039f16844`; on the shipped
series; **not in the manifest's series, not pinned, not shipped**; candidate `C:/Users/jwals/engines/llama-upstream-flash-cand0030`, built by `build_engine.py build` from a re-vendored tree
(a `--manifest` copy that lists the patch, `--src-root` outside the repo) so nothing in `engines/src` or the manifest's vendor record changed). **Off unless `LLAMA_STAGER=1`**; with it unset the slot upload makes the calls 0029 makes, in the same order.
- **Design, Strata's:** a ring of pinned buffers, memcpy worker threads that fill them in launch order from the pageable source ahead of the DMA, a buffer reused only after an event says the DMA that read it has landed,
  in-order claims. **Ours:** the unit is a 16 MiB chunk of a tensor (a 256 MiB expert tensor streams through a 16 x 16 MiB ring = 256 MiB pinned), not an expert blob; a separate **issuer thread** queues the DMA (Strata issues
  from the launching thread and waits for the memcpy there; here the thread that launches compute never blocks on a copy); a request names the event to record behind its last chunk and an event the copy stream waits on first
  (the last reader of the destination slot). Because that event is recorded by the issuer, a launch that reads a staged slot first calls `ggml_stager_wait_enqueued()` -- queued, not landed.
- **Used by:** 0029's slot upload (`ggml_backend_sched_resident_upload`) for any host weight that is not in the device's own pinned host buffer (a pinned one is already asynchronous and takes the direct path).
  Defaults: `LLAMA_STAGER_THREADS` = Strata's max(2, min(4, hw / 4)) = 4 here, `LLAMA_STAGER_RING` = 16 (Strata's `kRing`), `LLAMA_STAGER_CHUNK_MIB` = 16 (ours: 1.2 ms of a 14 GB/s link, so a copy call is < 2% of it) -- **starting values for the probe's sweep, not results**.
- **Read-ahead:** the workers copy up to a ring's worth ahead, and `ggml_stager_will_need()` asks the OS to read the pages of the layer after next (Windows `PrefetchVirtualMemory`, else `madvise(WILLNEED)`, on a hint thread) and
  of layers 0-1 when a batch starts (`LLAMA_STAGER_WILLNEED=0` turns it off), so a cold mapping is read by queued I/O and parallel faults instead of one fault at a time (the 0.44-0.54 GB/s of 13.3).
- **Not done: the refills (0025).** 0025 is not in the shipped series, so a patch on the series cannot call it; and in the series the cache's refill already runs on a worker thread, so there is nothing for a stager to hide. With 0025's
  queued path and unpinned experts (it falls back to the thread then, by its own text) the request API fits: one `ggml_stager_submit` per matrix with a done event. That is a follow-up that needs 0025 in the series.
- **Cost:** 4 + 1 + 1 threads, 256 MiB pinned (commit), and **the 6 slots that overlap needs (13.6): +769 MiB over the entry's 3 = 12 of the cache's 66.2 MiB rows (50 -> 38; section 12.4's trade)**. With 3 slots the stager still replaces the blocking copy by a parallel one, but nothing overlaps.
- **Tests** (`tests/test-stager.cpp`, in the patch; CPU, no GPU): a fake device whose DMA lands each chunk after a random delay, in stream order, with events; destinations and ring buffers poisoned and guarded;
  random requests (0 bytes, chunk +-1, several chunks), 1-4 workers, chunks 4-64 KiB, rings of 2-9: 200 trials, 0 wrong destinations; three further seeds x 1,500 trials, ~71,000 checks each, 0 failed; a request that names a wait event lands nothing until it completes; `wait_enqueued` returns with the done event recorded and the
  done event completes after the last byte; `free()` drains. **Controls that must fail do:** workers that skip the DMA wait are caught in 50 of 50 trials; two mutations of the stager (no stream wait for the wait event; the done event recorded before the copies) fail 391 and 1,608 checks.
  `test-layer-major` (0029, CPU, 48 runs) passes unchanged. **The candidate itself** (`cand0030`, MSVC + CUDA 12.8, the shipped flags, network fenced; `llama-server.exe` sha256 `ceafc357bbad75fc1260e4479e5c43cae6d36f675e5ca8fef3a3c45212cb0d33`, `ggml-base.dll` `d2d5c9cd...`, `ggml-cuda.dll` `d14416bf...`; engine tests 2/2): its own CUDA-built `test-stager.exe` and `test-layer-major.exe` were run with `CUDA_VISIBLE_DEVICES=-1` (no GPU visible): 9,570 checks 0 failed, and 71,752 at another seed, test-layer-major all passed. **Not run: `test-stager --device CUDA0`** (a pageable source through the real pinned ring to the card, read back; written, in the CUDA build only), and **no speed number exists**.

### 13.6 The bigger lever: the MoE over the whole chunk (a plan; no code)

**What Strata's prompt path does with the experts** (`src/prefill/moe_mmq.hpp`, `prefill.cpp:1132-1243`): the experts of a layer run through **llama.cpp's own MMQ kernels** (int8 tensor cores, quantized weights) over the
rows of the **whole 8,192-token chunk grouped by expert**: `mmq::quantize` of all T x K rows, then one launch per product with per-expert row bounds. That is the kernel family `MUL_MAT_ID` uses here; the
difference is N: 8,192 x 10 / 512 = **160 rows an expert** against 512 x 10 / 512 = **10** in our 512-token ubatch. (Their header says the FP16-dequantize + cuBLAS route it replaced wrote ~10 MB of FP16 per expert while MMQ reads the 1.4-2 MB expert once, and that IQ1_M is not covered by their MMQ.)

**Where our compute goes (measured):** at 512-token ubatches the experts are **62.6%** of GPU op time (`pf-dep-opt`, 23k prompt, n=1 run: iq2_s 26.5, q2_0 20.0, iq1_m 10.1, iq2_xxs 6.0); the weights of a layer are read once per ubatch (704 MiB x 48 = 33 GiB / 448 GB/s = 79 ms of a ~780 ms ubatch, derived), so
the time is not weight bandwidth: it is MMQ running tiles that are mostly empty (10 columns in a 16-128 column tile) and unpacking each expert's weights per ubatch. **Evidence that the per-token cost falls with N:** non-layer-major `-ub 1024` (`pf-ub1k`) took 3.98 s per 1,024-token ubatch against 3.41 s per 512 (`pf-dep`, ~150 tok/s);
with U ~ 2.6 s in both (13.3; 3.41 s minus the layer-major C of ~0.86 s averaged over the prompt = 2.55 s) C(1,024) ~ 1.43 s against C(512) ~ 0.86 s: **per token 1.40 ms against 1.67 ms, -16%** (derived; two points, assumes U equal).
Fitting c(ub) = c0 + k / ub through them gives **c0 = 1.12 ms, k = 283 ms-tokens**, so c(2,048) = 1.26, c(4,096) = 1.19, c(8,192) = 1.155 ms a token. **This extrapolates two derived points by 16x; it is the thing to measure first (13.7 step 4).**

**The design:** keep 0029's layer-major order and split each layer into two graph ranges. (1) The attention half (DeltaNet or QSA + the residual mix) runs per 512-token ubatch exactly as now. (2) The **MoE half** (FFN norm, router, top-10, the three `MUL_MAT_ID`, the shared expert, the hyper-connection post) runs as ONE graph over G tokens =
all the ubatches of the layer's chunk (G = 2,048 at `-b 2048`, up to 8,192), reading and writing rows of the persistent hidden buffer 0029 already has (2 x 40 KiB a token: 160 MiB at 2,048, 640 MiB at 8,192, **already allocated** at `-b 8192`). The MoE is per-token independent, so the only numeric change is MMQ's tile shape (the stream-k fixup order): KL and needles, not bit equality.
The last layer's FFN sees only the output rows, as today. MTP's `h_nextn` taps are the last layer's output rows and are unchanged.

**VRAM on the 16 GB card (derived).** The MoE half's scratch is ~0.3 MiB a token live at once (gate, up and swiglu [640 x 10] f32 = 25.6 KB each, the down output and its weighted copy [2560 x 10] f32 = 102 KB each, the quantized activations 3 KB, ids; the ggml
allocator reuses what is dead): **0.6 GiB at G = 2,048, 1.2 at 4,096, 2.4 at 8,192**. The attention half's graph is reserved at its worst case already (the compute buffer is **1,871.6 MiB** at `-ub 512`, 3.7 MiB a token, dominated by the QSA's n_kv-wide scratch at the reserved 262,144 cells, 12.3(b)), and the two halves run one after the other and can share it:
**the MoE half costs nothing extra while 0.3 x G stays under ~1.8 GiB, i.e. up to G ~ 6,000**; at G = 8,192 it is ~+0.5 GiB. The deployed card has **~750-780 MiB free at its lowest** (`pf-lm3-b8k` 763, `pf-lm6-b8k` 781 MiB), so anything beyond that comes out of the cache's 66.2 MiB rows (the decode cost of a row is
small, section 8's hit-rate simulation, derived): 100 MiB = 1.5 rows. Reclaimable room, in order of cost: (a) the scheduler's own copy of every offloaded weight inside the compute buffer, up to **737 MiB** of the 1,871 (12.7, derived; a slot-backed split never touches it) = 11 rows free of any decode cost; (b) 0019's block top-k removes the `expanded` [n_kv, n_tokens] f32
(1 MiB a token at 262,144 cells) from the attention half (its needles and speed gate never ran, 13.2); (c) 0029's 3 slots (769 MiB) become 6 for the overlap (+769 MiB). Net at G = 4,096 with (a) and (b): **no row lost**; with the 6 slots ~ -1 row. (Derived; the arm's `lowest free` line decides.)

**Expected speed (derived from 13.3's fit; compute-only ceilings at 25.6K-prompt averages, tok/s = 1000 / c(ub) with the batch's upload added or hidden).** Now: 8,192 x 1.672 ms = 13.7 s + U 2.6 s = **16.3 s a batch, 503 tok/s** (the measured end-to-end 433-439 is lower by the 409 + 103 tail, 13.3).

| | per batch | tok/s (batch) |
|---|---:|---:|
| now (ub 512 MoE, upload blocking) | 16.3 s | 503 |
| + the stager, upload hidden (6 slots) | 13.7 s | 598 |
| + MoE over 2,048 tokens, upload hidden | 10.3 s | 794 |
| + MoE over 4,096, upload hidden | 9.7 s | 840 |
| + MoE over 8,192, upload hidden | 9.5 s | 866 |
| the asymptote (c0 = 1.12 ms) | 9.2 s | 893 |

So the stager is **+19%** and the whole-chunk MoE a further **+33-45%**: **~800-870 tok/s on the batch, ~1.6-1.7x the current batch rate**, against Strata's **1,153-1,294** (theirs: Q2_0, RTX 5070, 32K, n=1): **~65-70% of theirs**. The remainder is not in this lever: the non-MoE share
(0.66 ms a token at 512: DeltaNet 6.2%, attention 6.6% at 23K and growing with context, the dense MUL_MATs, norms, hyper-connection ops: ~37% of C), the iq1_m layers (10.1% of op time; Strata's MMQ does not cover IQ1_M either), the tail batches, and the card (the 5070's memory bandwidth is ~1.5x ours: spec sheets, 672 vs 448 GB/s). **Risks:** the fit is two points;
MMQ's per-tile cost may flatten earlier than c0 says; the QSA scratch and the MoE half may not share the buffer as cleanly as the allocator's liveness suggests; an 8,192-token MoE graph is a longer uninterruptible unit (a cancel costs a chunk, 12.7). **Decode is untouched** (1-token batches never take the path).

### 13.7 GPU steps, in order (each needs the operator's "GPU go" and a free card; none was run)

1. **The probe** (`python bench/fn_probe/run_probe.py --go`; ~6 minutes; flash-next unloaded; 2.5 GiB of VRAM). Answers 13.4. If E(pin) < 0.5 the stager is not the unlock: report, and go to step 4.
2. **cand0030 on the card, unit level** (seconds): `test-stager --device CUDA0` (the pinned ring and the real events); the sweep's best threads / chunk / ring into the entry's env.
3. **The stager in the engine** (`bench/flashnext_gate.py` arms next to `pf-lm3-b8k` / `pf-lm6-b8k`; n=3 each, ~25.6K prompts, warm, the entry's flags): `LLAMA_STAGER=0` (the control: must read 433-439: "off is off"), `LLAMA_STAGER=1` with 3 slots (no overlap: the parallel-copy effect alone), with 6 slots (38 rows: the overlap), and `LLAMA_PIN_EXPERTS=1` with 6 slots (the pinned
   ceiling: how much of the gain the pin, at 36.4 GB of commit, would buy); `LLAMA_LAYER_MAJOR_DEBUG=1` for the stager's line (memcpy / ring-wait / launching-thread-block ms); then KL (batch 8192), needles 5/5 at 32K, corrupt, the `--checkpoint-every` restore, a cancel mid-prompt, the lowest free VRAM, and the decode at 38 rows (4.4K / 35K, n=3). And `fn_first_prompt` cold-cache with `LLAMA_STAGER=1` (the read-ahead's own effect; its pre-read arm as the comparison).
4. **The compute curve (no new code):** layer-major at `-c 16384` (the QSA reservation is then small) with `-ub` 512 / 1,024 / 2,048 / 4,096 (and `-b` 8,192), n=3 warm, the upload hidden by step 3's best setting if there is one: C(ub) per token. The table in 13.6 rests on it: if c(2,048) is not ~1.3 ms a token or lower, the plan is wrong and the lever is smaller.
5. Only then the MoE-half graph (13.6), gated by KL / needles / corrupt, the lowest free VRAM, and a 6-slot + G sweep.
