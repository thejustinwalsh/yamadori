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
