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
  returns at once and there is nothing for a thread to hide.
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
