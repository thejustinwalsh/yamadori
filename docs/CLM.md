# CLM: serving, fidelity, and the skill decider (Tier 0 and Tier 1)

Built and measured 2026-09-27. The assessment that preceded it is
`docs/CLM-EVAL.md`. What CLM is: a frozen Qwen3-8B used as an embedder (the
last token's final hidden state, L2-normalised) plus two small MLP heads.
A state and each option are embedded separately. The answer is a softmax over
`scale * cos(state_head(state), action_head(option))`, and it is relative to
the option set it was given.

**Verdict (Tier 0 and Tier 1).** The released CLM, used as the selector's
decider on the daily eval, scored **445/464** checks. The deterministic
selector scored **464/464** on the same cases, but that selector was tuned on
this eval, so the comparison is against an in-sample baseline.

- CLM never broke a MUST-NOT (385/385) or a NOTHING (19/19).
- Every failure is a missed injection: MUST was 41/54. The misses cluster on
  two skills and one area.
- A light fine-tune of the heads on the activation tests also scored 445/464.
  It fixed 2 MUST misses and caused 2 others.

Both runs are n=1 on one eval. See "Tier 0" below.

## What is served, and where

| piece | what | where |
|---|---|---|
| encoder | our GGUF of `Qwen/Qwen3-8B` @ `b968826d`, **Q8_0** (8,709,518,688 B, sha256 `1633fc3e…`) | llama-swap `clm-encoder` (config.yaml and config.template.yaml, group `decision`), on the A4000 by UUID |
| server flags | `--embeddings --pooling last -c 2048 -np 1 -b 2048 -ub 2048 -ngl 999 -dev CUDA0` | the llama-prism server (`macros.server`, pinned in engines/manifest.yaml) |
| heads | `CLM_v0.1-8B.pt`, converted to a deterministic `.npz`, run in numpy on the CPU | `index/clm/heads/CLM_v0.1-8B.npz` (`mcp/clm_heads.py`) |
| fine-tuned heads (Tier 1, not the default) | the same heads after `finetune.py --task choice` on the skills' activation tests | `index/clm/heads/CLM_v0.1-8B_skills-v1.npz` (`YAMADORI_CLM_HEADS`) |
| client | `decide`, `decide_many`, `decide_detail[_many]`, `warm_options`, `status` | `mcp/clm.py` |
| decider | `skill_deciders.ClmDecider` (the selector agent's), `YAMADORI_SKILL_DECIDER=clm` | `mcp/skill_deciders.py` |

**The encoder is not live yet.** The running llama-swap was started without
`-watch-config`, so `clm-encoder` is served only after llama-swap restarts,
and that restarts the main model too. Every number below comes from
`bench/clm/standalone.py`, which runs the same binary with the entry's own
parsed arguments and card pin on a scratch port, holding gpu_room's room
lock. The live check through llama-swap is written but has not been run:
`mcp/test_live_stack.py --only clm` (LIVE-COVERAGE row 89).

**On-demand swap** (operator, 2026-09-27):
- llama-swap loads the encoder on the first decision.
- `mcp/gpu_room.py` makes room for it. Its SIZES row is measured: 9,379 MiB
  idle and 9,415 MiB at peak.
- gpu_room evicts it (least recently used first) when vision or a draw needs
  the card.
- `ttl: 300` frees the card after five idle minutes. That value is a choice,
  the same as bonsai-vision's.
- It does **not** co-reside with image generation. The A4000 had 11.3 GB
  free beside retrieval, and the encoder plus a draw's +6.4 GB peak does not
  fit.

## The client, and what the heads are fed

The client reproduces the official path (`Contrastive-LM/CLM` @
`bb42c6c5bf914fd449bed2f6ca65be80602cb1f7`, Apache-2.0, read only):

- **State text** is `state + "\n\n" + instructions`, question last
  (`schema.state_text`). Each option is embedded as its own text, with
  nothing prefixed.
- **Tokens** come from the pinned `tokenizer.json`, with
  `add_special_tokens=False`: no BOS or EOS (`embed_utils.Recipe`). They are
  sent to the server as token ids, so the truncation is exact.
  - G2 check: llama-server's own `/tokenize` gave the same ids on all 185
    fidelity texts.
  - The GGUF has `add_bos_token false` and no `add_eos`.
- **Window.** At most 2,047 tokens (`Recipe.cap = max_len - 1`).
  - A longer state is cut without reordering the caller's evidence. Whole
    pieces go first, from the front (`keep="tail"`, CLM's own left
    truncation) or from the back (`keep="head"`). Then the boundary piece
    is cut.
  - The instructions are never cut.
- **Pooling.** llama-server's `--pooling last` output comes after the final
  norm, the same as HF `last_hidden_state[:, -1]` and vLLM's LAST pooler.
  `/v1/embeddings` L2-normalises it, and the client normalises again, as
  `embedder.py` does.
- **Heads.**
  - `mcp/clm_heads.py` is `make_head` in numpy: exact erf GELU, LayerNorm
    eps 1e-5, width 1536, depth 3, projection 512.
  - `scale = min(exp(4.6132), 100) = 100`.
  - Parity with the official torch modules on 64 random unit vectors: max
    |logit| difference 7.6e-6, projections 2.4e-7.
  - The `.npz` re-extracts byte-identically.
- **Caching.** Option vectors go to `index/clm_actions.npz`, keyed by the
  sha256 of the option text under the encoder's sha256. A skill's revision
  changes its trigger text, and so its key. The state is encoded once per
  call for any number of questions (`decide_many`).
- **One door and lanes.**
  - Requests go to llama-swap's `/v1/embeddings` inside
    `gpu_room.use("clm-encoder")`, the way code_search's embeddings do.
  - They never go through `:1234` and take no admission lane.
  - Each process allows one encoder request at a time (`_LANE`). A busy
    lane, no room on the card or an unreachable server raises
    `ClmUnavailable`, which carries a retryable fact and a remedy. The
    decider then abstains.
- **Prefix cache.** llama-server reuses a slot's cached prefix even for
  embeddings: the task's `cache_prompt` is always on, and the request field
  is ignored.
  - The same input twice gives cosine 0.99998 (max |diff| 5.3e-4 at 54
    tokens), because only the last token is recomputed.
  - An input after a different one is recomputed whole and is bit-identical
    (3 passes, max |diff| 0.0).
  - The probe is `bench/clm/standalone.py`, `prefix_cache`.

## Building the encoder (reproduced byte for byte)

The tools are the llama-prism checkout at `9a9394a8`: `convert_hf_to_gguf.py`
(sha256 `21b70f59…`, gguf-py 0.19.0) and `build/bin/llama-quantize.exe`
(`fc7c5579…`).

```
python convert_hf_to_gguf.py <models>/Qwen3-8B --outtype bf16 --outfile <models>/Qwen3-8B-BF16.gguf
llama-quantize.exe <models>/Qwen3-8B-BF16.gguf <models>/Qwen3-8B-Q8_0.gguf Q8_0
```

- The whole recipe was re-run into a scratch directory with the GPUs hidden.
  The BF16 intermediate (`9992dae5…`), the Q8_0 and the Q6_K came out
  byte-identical.
- The five source shards match the HF LFS sha256s at the pinned revision.
- `models/manifest.yaml` has the entries:
  - `qwen3-8b-clm-q8` and `qwen3-8b-clm-bf16`
  - `qwen3-8b-hf-shard-1..5` and `qwen3-8b-tokenizer`
  - `clm-v0.1-8b-pt` (Contrastive-LM/CLM-v0.1-8B @ `e939398d`), `clm-heads-v0.1`
    and `clm-heads-skills-v1`
- `scripts/verify_artifacts.py`: 0 errors, 0 warnings.

## Fidelity

`bench/clm/fidelity.py` runs 40 (state, options) skill-selection questions:
22 daily-eval user rows, 4 agent-step sequences, and 7 activation `should`
cases plus their 7 near misses. That is 36 distinct states and 149 option
texts.

- **Reference.** HF transformers' own Qwen3 modules in **fp32 on the CPU**,
  streamed one layer's weights at a time.
  - The 16.4 GB model does not fit this host's free commit next to the
    stack. The streaming equals `Qwen3Model.forward` on a tiny random config
    (max |diff| 4e-7).
  - The run took 607 s for 6,653 tokens.
- **Served.** `mcp/clm.py`'s own encoder request.
- **Decisions.** The release heads, zero-shot.

| encoder (A4000 unless noted) | state cos median / min | option cos median / min | argmax | none-vs-pick | max prob diff (median / max) | VRAM idle / peak | kept |
|---|---|---|---|---|---|---|---|
| **Q8_0** | 0.99990 / 0.99965 | 0.99994 / 0.99989 | **40/40** | **40/40** | 0.005 / 0.095 | 9,379 / 9,415 MiB | **yes** |
| Q8_0, `-ngl 0` (CPU weights, op offload) | 0.99989 / 0.99969 | 0.99994 / 0.99984 | 40/40 | 40/40 | 0.009 / 0.062 | 1,995 / 2,075 MiB | the mode is optional |
| Q6_K | 0.99943 / 0.99863 | 0.99964 / 0.99933 | 39/40 | 39/40 | 0.010 / 0.204 | 7,631 / 7,669 MiB | no (deleted) |
| Q4_K_M | 0.99583 / 0.98798 | 0.99803 / 0.99569 | 37/40 | 39/40 | 0.029 / 0.466 | 6,165 / 6,201 MiB | no (deleted) |

Determinism: identical vectors on 3 passes over the 36 states, for every arm.

## Serving cost (Q8_0, `bench/clm/swap.py` and `standalone.py`)

| mode | VRAM held | swap-in | per call at 40 / 300 / 800 / 1,500 / 2,047 tokens |
|---|---|---|---|
| full GPU, on demand (chosen) | 9,379 MiB idle, 9,415 peak | **3.7 s warm**, **6.3 s cold** (the copy was made with `robocopy /J`, so none of it was in the page cache). The first call adds about 55 ms | 65 / 180 / 415 / 716 / 985 ms |
| `-ngl 0`, op offload (weights in RAM) | 1,995 MiB idle, 2,075 peak | 2.8 s | 1,365 / 1,473 / 1,759 / 2,081 ms |

The per-call cost on real selector states (Tier 0, 75 calls, 87 questions):

- state encode p50 49 ms, p90 100 ms, max 467 ms
- state tokens p50 44, max 481
- whole-call p50 142 ms when the options are seen for the first time (147
  option texts encoded). After that the options come from the cache.

## Tier 0: the released CLM as the skill decider

`bench/clm/tier0.py` runs the selector exactly as
`bench/skills/replay_selection.py --daily` does, offline, with no embedding
stage and no fallback model. The selector's rounds build the questions: the
area, each surviving skill's trigger text, and NONE. It runs twice: once with
the stub first, once with `clm` first. The encoder is Q8_0 on a standalone
llama-server under the room lock. The results are in
`bench/clm/results/tier0_zero_shot.json`.

| decider | MUST | MUST-NOT | NOTHING | form/recall | all |
|---|---|---|---|---|---|
| stub (deterministic; **tuned on this eval, in-sample**) | 54/54 | 385/385 | 19/19 | 6/6 | 464/464 |
| CLM, released heads, zero-shot | 41/54 | 385/385 | 19/19 | 0/6 | 445/464 |
| CLM, skills-v1 heads (Tier 1) | 41/54 | 385/385 | 19/19 | 0/6 | 445/464 |

CLM answered all 87 questions, with no abstention.

**Where it is wrong, and why.** The probabilities below are CLM's, over the
question's own options.

1. **It prefers NONE on short step and evidence questions about koota.** 6
   MUST misses and all 6 form/recall checks.
   - Example: `seq-koota-body-recall#1`, step question, area koota: options
     `koota-queries-and-systems 0.000, none 1.000`.
   - The deterministic evidence (the tool result imports `koota`) settles
     these. CLM reads only the text, and a one-option question over a
     step's digest scores NONE.
2. **It prefers NONE on long, specific build requests.** The Octopus and
   pagoda specs and `stack-particles`.
   - Example: `op-sentence-octopus`, asked r3f: `r3f-v10-setup-21 0.006,
     r3f-background-6 0.032, none 0.950`.
   - Every asked area went to NONE (r3f 0.95, koota 0.94, pmndrs_math 0.96).
3. **It picks the wrong skill inside a named area.**
   - Example: `op-sentence-pagoda`, asked r3f: `r3f-background-6 0.491,
     r3f-v10-setup-21 0.256`. The request names "r3f (react-three-fiber)
     v10".
4. **`typegpu-pipeline`.** The web-gpu skills all score under 0.02 against
   `none 0.983`.

It never picked a skill where none belonged. The same shape showed on the
fidelity set: the reference picked NONE on 26 of 40 questions, 4 of 7 should
cases were right, and 7 of 7 near misses were right.

**Tier 1: the light fine-tune.** `bench/clm/finetune_data.py` builds the
data; the command is below.

- Data: 2,202 train rows: 1,098 `should`, 1,103 `should_not` and 1
  client-traffic fallback. Test: 18 daily-eval rows, held out.
- Train rows that share any 8-word shingle with a daily-eval text (the test
  material included) are dropped: 2 were.
- Validation accuracy went from 0.627 to 0.850. Validation is the same
  skills' other cases, so it is in-distribution.
- On the held-out daily test it went from 15/18 to 16/18: one question.
- The fine-tune fixed `op-sentence-pagoda` and `bug-koota-throw`, and broke
  `stack-r3f-v9` and `tsl-material`. The net change is 0.
- It is reproducible: re-run on the CPU, `best_head.pt` came out
  byte-identical, and it took 35 s.

```
git clone https://github.com/Contrastive-LM/CLM CLM && git -C CLM checkout bb42c6c5bf914fd449bed2f6ca65be80602cb1f7
CUDA_VISIBLE_DEVICES=-1 python CLM/train/finetune.py --task choice --data index/clm/finetune/data --workflow skills \
  --init-ckpt <models>/CLM-v0.1-8B/CLM_v0.1-8B.pt --embed-cache index/clm/finetune/embeddings \
  --embed-model Qwen/Qwen3-8B --max-len 2048 --loss infonce --targets soft --batch 256 --epochs 20 \
  --patience 5 --val-frac 0.1 --seed 1234 --out-dir index/clm/finetune/runs/skills-v1
```

The embeddings come from **our served Q8_0 encoder**, not vLLM bf16:
`finetune_data.py embed` fills finetune.py's own `TextCache`, so finetune.py
never builds an encoder.

## Later options (paused or shelved by the operator, 2026-09-27; not built)

- **Bonsai as the one A4000 base.** `bonsai-vision`'s process would serve
  vision and CLM decisions.
  - Header facts: `qwen35` hybrid (Gated DeltaNet plus full attention every
    4th layer), hidden **5,120**, 65 blocks (64 + MTP), no pooling key, and
    Prism Hadamard-rotated weights. The rotation is orthogonal, so it does
    not matter to heads trained on its vectors.
  - In our fork `--embeddings` restricts the whole server, and it also sets
    `server_output_limits` to `{n_batch, 1}` (server-context.cpp:41), which
    turns off MTP speculative decoding. So the main server cannot simply add
    it.
  - It would need engine **patch 0004**: per-request last-token hidden state
    from a generation server, through the existing per-slot `need_embd`
    path.
  - New 5,120-d heads would have to be trained by distillation from the 8B
    teacher. For scale: CLM's own heads saw about 60M plus 30M plus 1M pairs.
    We hold 2,213 activation cases and 8 client selection labels. The unit
    that multiplies is the question: states times area questions share one
    state encoding. Even so, tens of thousands of teacher-labelled questions
    would come from a few thousand states.
- **Distillation to smaller students.** Candidates:
  - Qwen3-Embedding-0.6B, already resident
  - Qwen3.5-0.8B, on disk
  - an early exit of Qwen3-8B
  - a low-bit Qwen3-8B
  - Qwen3-1.7B or 4B, which would need a download
- **Low-bit and ternary encoders.** The prism `llama-quantize` lists
  `PTQ1_0`, `TQ1_0` and `TQ2_0`. None was measured. Q4_K_M already moves 3 of
  40 decisions.
- **Stitching** Bonsai vectors into Qwen3-8B space with a linear map.

## Files

- `mcp/clm.py`: the client
- `mcp/clm_heads.py`: the heads, extract and parity
- `mcp/test_clm.py`: offline, 31 checks
- `bench/clm/fidelity.py` and `fidelity_questions.jsonl`
- `bench/clm/standalone.py`: measurement server and `serving()`
- `bench/clm/swap.py`
- `bench/clm/tier0.py`
- `bench/clm/finetune_data.py`
- `bench/clm/results/*.json`: fidelity, standalone, swap, tier0
- `mcp/test_live_stack.py`: `clm`
- `config.yaml` and `config.template.yaml`: `clm-encoder`, group `decision`
- `mcp/gpu_room.py`: SIZES `clm-encoder`, measured
- `models/manifest.yaml`

The vectors, heads, fine-tune data and runs live under `index/clm/`
(gitignored).
