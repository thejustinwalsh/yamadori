# jjava batched reads: every question of a call in one engine pass (2026-10-07)

Operator, 2026-10-07: "we cant do that with ours too, this is a model hack somehow to make it faster?" (on Cloudflare's
Clef reading every question of a call in one forward pass).

**STATUS: BUILT OFFLINE, NOT RUN ON A GPU.** Engine patch `engines/patches/llama-bonsai2-ada/0042-server-decide-batch.patch`
(recorded UNSHIPPED in `engines/manifest.yaml`; `shipped` is still `80d2c60d`, `config.yaml` does not pass
`--decide-seqs`), Python path `mcp/decider_batch.py` behind `YAMADORI_DECIDER_BATCH` (default OFF). Every speed figure
below is arithmetic from a fit, not a measurement of the batched path; the measurement is section 8 and waits for "GPU go".

## 1. What Clef does, and why it is not a trick we can borrow

Read from the merged PR (ggml-org/llama.cpp#29831, merge `99b9548`, `gh pr diff`, fetched read-only 2026-10-07):

- Clef is its OWN ARCHITECTURE (`LLM_ARCH_CLEF`, `src/models/clef.cpp`, 616 lines, a converter `conversion/clef.py`): a
  Qwen-family backbone with a **trained joint head**. All the questions of a call are laid out in **one prompt, one
  sequence** ("Known limitations: ... Single seq per batch"); `llama_batch_ext_set_decision_order` marks which tokens are
  question text (kinds 1/2/3: noul/choice/score) and option text (kind 4); the head mean-pools those spans and emits one
  score per option from the span embeddings. One task, one forward pass, no answer token generated, no vocabulary
  softmax (`server-decision.cpp` `fill_task_joint`; the server answers from `result->scores`, `is_joint()` = Clef only).
- The questions DO see each other (one causal pass over one sequence). That is fine for a head trained for it; it is not for
  jjava, whose readout is the base model's next-token letter, measured one question at a time.

So the speed of Clef is three things: (1) one task instead of 2K, (2) each question's text appears once (no second option
order: the trained head needs none), (3) no per-question logits row. Only (1) transfers to us, and (2) in part: our two
orders of a question share all of the question text, and the engine can decode that head once.

What upstream does for its OTHER decision models (OPENJEV, LEV, KEV, NIMBLE: a next-token read per question variant, our
shape) is `server_decision_group_tasks` + `server_slot::copy_prompt_to`: a group of `n_parallel` tasks whose prompts share a
token prefix; the first slot decodes the shared prefix, the others continue from it with `llama_memory_seq_cp`
(`can_share_prompt()`). That is the mechanism this patch uses, outside the decision-model GGUF metadata (our Bonsai is not a
decision model) and with more blocks than slots.

## 2. Is there an existing server feature that already shares the prefix? No (checked in our tree, `engines/src/llama-bonsai2-ada`)

| feature | what it does | why it does not do this |
|---|---|---|
| `n_cmpl` / parent-child tasks (`add_child`, `copy_state_to`) | children are clones of the parent's task (`tokens.clone()`), started after the WHOLE prompt is processed | children cannot carry a different continuation: only the sampling differs. (Upstream's newer `n_tokens_shared` / `copy_prompt_to` for children with their own prompt is not in this tree: it arrived with the decision-model PRs.) |
| several concurrent requests on `-np K` slots, `cache_prompt` | slot selection picks the slot with the best prefix similarity (`get_available_slot`); an idle slot re-prefills | every other slot re-prefills the state or restores it from the host prompt cache (`--cache-ram`): K copies of 150 MiB recurrent state + the state's KV through PCIe, and K slots = K recurrent cells anyway. No live KV sharing between slots |
| the host prompt cache / context checkpoints | what the sequential path already uses (a checkpoint at the last user message restores the state for the next question) | it is the per-read cost we want to remove: a restore (host to GPU) per read |
| `/slots/<id>?action=save|restore` | state to disk | file I/O per read |
| `llama_batch_ext_set_decision_order` | Clef's span marks | needs the Clef head |

A server patch is needed. It is small because the engine already has every primitive (`llama_memory_seq_cp`, equal-split
ubatches for recurrent models, the unified pool).

## 3. The wire contract: `POST /decide-batch` (llama-server; through llama-swap `/upstream/<model>/decide-batch`)

Request (JSON):

```
{
  "prefix":        "<text>" | [token ids / strings],   # the shared state, rendered by the SERVED chat template up to and
                                          #   including the state message's end ("...<|im_end|>\n"); text is tokenized
                                          #   add_special=true, parse_special=true (what /v1/chat/completions does)
  "groups":        [ ["<block>", ...], ...],   # each block = the rest of one prompt after the prefix, starting at a
                                          #   special token ("<|im_start|>user\n...Answer:"); add_special=false,
                                          #   parse_special=true. A group is a HINT that its blocks probably share a
                                          #   head (the two orders of one question): the engine decodes the longest
                                          #   common TOKEN prefix of a group once and forks it, so sharing can never
                                          #   change an answer
  "token_ids":    [int, ...],             # label token ids whose exact log-probability is returned for every block
                                          #   (union over all questions; <= 256)
  "top_logprobs": 20,                     # also the top-N entries of each block's next-token distribution (<= 100)
  "keep_prefix":  false,                  # true: leave the prefix's state resident for the next call; false: free all
  "verify_tokenization": false,           # true: per block, tokenize prefix+block whole and report whether it equals
                                          #   tokenize(prefix)+tokenize(block) (the boundary check; text only)
  "release":      false                   # true (nothing else): free the resident prefix
}
```

Response 200:

```
{ "prefix":  {"tokens": N, "reused": R, "processed": N-R, "resident": bool},
  "groups":  [ [ {"tokens": n_block, "shared": s, "processed": own,
                  "top_logprobs": [{"id", "token", "logprob"}, ...],
                  "logprobs": {"<id>": float, ...},       # exactly the requested token_ids
                  "tokenization_ok": true|false|null }, ... ], ... ],   # same nesting and order as the request
  "timings": {"total_ms", "prefix_ms", "shared_ms", "blocks_ms", "waves", "decode_calls",
              "processed_tokens", "shared_tokens_saved"},
  "seqs":    {"prefix": id, "first": id, "forks": W} }
```

`logprob` is the natural-log softmax over the WHOLE vocabulary at the block's last token: the quantity
`/v1/chat/completions` returns as `logprobs.content[0].top_logprobs[].logprob` for `max_tokens: 1`. `processed` of a
group's first block includes the shared head (counted once).

Errors (llama-server's `{"error": {"code": <http status>, "message", "type"}}`): **501** `not_supported_error` when the
server was not started with `--decide-seqs N` (N >= 1) and `--kv-unified`; **400** `invalid_request_error` for a malformed
body or a token id outside the vocabulary; **503** `unavailable_error` when the
unified pool has no free cells (nothing is left behind: every scratch sequence is freed and the resident prefix dropped).
`/props` reports `decide_batch: {seqs, prefix_seq}`. A caller treats 404, 501 and 503 as "use the one-request-per-read
path".

## 4. The mechanism, as built (`common/decide-batch.{h,cpp}`, `tools/server/server-context.cpp`)

- **Sequences.** `--decide-seqs N` makes the context `n_seq_max = n_parallel + 1 + N`: slots take ids `[0, n_parallel)`,
  the prefix `n_parallel`, the forks the next N. `n_outputs_max` is raised to at least `n_seq_max` (the context asserts
  it; the server had set it to the slot count) so N rows can leave one decode.
- **The prefix** is decoded once on its own sequence P (chunks of `n_batch`, no outputs) and kept between calls only when
  `keep_prefix`; a later call reuses it when it is the resident prefix or extends it (the recurrent state cannot roll
  back, so any other prefix resets P). A prefix that was cleared out from under it (`llama_memory_clear`, another
  caller's `seq_rm(-1)`) is noticed by `seq_pos_max(P)` and reset.
- **Blocks.** Each wave takes groups until their leaves fill N forks. A group's blocks are decoded from their longest
  common token prefix once (a group with more blocks than forks is cut into chunks of N, each with its own head; stage A:
  one head sequence per chunk, `seq_cp(P -> head)`), then each leaf forks from the head
  (`seq_cp(head -> leaf)`; groups with no shared head fork from P) and decodes its own tokens (stage B), the last token of
  each flagged as an output. Every stage goes through `llama_decode` in lock-step slices (`n_batch / n_active` tokens of
  every unfinished sequence per call), sequence ids ascending and longest first, so the finished sequences of the
  staircase are a suffix and the recurrent equal-split (`split_equal(sequential = true)`) keeps consecutive ids together.
- **Why it is exact.** A block's sequence holds the prefix's attention cells (shared: in the unified pool `seq_cp` only adds
  the sequence id to the cells) and the prefix's recurrent state (the first decode of a fork copies it into the fork's own
  cell, `find_slot`'s copy-on-write); nothing the fork writes is visible to another sequence. The shared head is the
  longest common prefix of the blocks' own TOKEN lists, every block keeping at least its last token, so what is skipped is
  identical by construction.
- **The loop.** The request is a server task (`SERVER_TASK_TYPE_DECIDE_BATCH`) processed on the loop thread between
  decodes (declined while a decode yields), so nothing else touches the context; the server stalls for the call's
  duration, as it does for any decode.
- **Cleanup.** The scratch ids a wave used are `seq_rm`'d after the wave and every fork id again at the start of the next
  call (a call that died half way cannot leave cells a fork would inherit); any failure, or a row reader that throws, frees
  them all and drops the resident prefix. The pool keeps only the prefix between calls (`llama_memory_cells_used_max_p1` is 0 after a call without
  `keep_prefix`).

### 4.1 What it costs: memory

Recurrent state is 149.63 MiB per cell on this model (the 27B: 64 layers, R 5.63 + S 144 MiB; measured:
`bench/results/kv_placement/*/kvplace.log`, "2244.38 MiB (3 cells, 64 layers, 3 seqs 4 rs_seq)" = 15 rows). bonsai-a4000 runs
`-np 2` (2 cells, 299 MiB, no rollback rows). Each fork and the prefix take a cell:

| W (`--decide-seqs`) | extra cells | extra MiB | the same VRAM in KV cells (35,840 B/cell, `bench/a4000_fit.py`) | `-c` that keeps today's headroom |
|---|---|---|---|---|
| 1 | 2 | 299 | 8.8k | 132,557 |
| 2 | 3 | 449 | 13.1k | 128,179 |
| 4 | 5 | 748 | 21.9k | 119,424 |
| 8 | 9 | 1,347 | 39.4k | 101,914 |

(`bench/decider/batch_arith.py`; `-c` is today's 141,312 less the cells; re-fit with `bench/a4000_fit.py` before choosing: the
fit's peak includes compute buffers that grow slightly with the output rows. Deciding W and `-c` is the operator's.) The
KV cells for a call are transient and come out of the same unified pool: a call needs `prefix + sum of the wave's own
tokens` cells, e.g. 8k state + 4 blocks of ~100 = 8.4k of 141k; slot 0's cells are never touched, and a pool that cannot
hold it answers 503 and the caller falls back.

## 5. The engine's tests (CPU, offline; what exists and what it shows)

`tests/test-decide-batch.cpp` (static CPU tree; registered in the manifest entry's `tests`): a tiny random hybrid **qwen35**
(4 layers: linear, attention, linear, attention; norm weights ~1 and projections of std 0.1, because test-llama-archs'
1e-2 everywhere makes every row ignore its context: the first run of this test passed with a row identical with and without
its prefix, which is why section 0 below exists). 140 checks:

- **sensitivity (0):** rows differ between blocks by 6.96, with no prefix by 0.76, with another prefix by 0.87 (log-prob);
  tolerance 5e-4, so a wrong state is ~1,500x the tolerance and the test can see it.
- **equality:** every batched row equals a cold single-sequence decode of prefix + block: largest log-softmax difference
  **2.365e-4** over 16 batch-vs-cold comparisons (15 of them between 1.3e-4 and 2.4e-4, one exactly 0) (float32 order-of-sum noise: the reference decodes 512-token ubatches, the batch 2-16
  tokens a sequence per ubatch). Flat groups, groups sharing a head, waves (W=3 over groups of 1/2/3), W=1 (alone and on a pair), a group larger than the
  forks (split in chunks), duplicated blocks,
  blocks differing only in the last token, identical blocks, a reversed group order (difference 0), the resident prefix
  reused whole, extended, replaced, shortened, cleared under it, released.
- **isolation:** two live slots (20 and 12 tokens) beside the batch: their positions are unchanged and slot 0's next row
  equals the cold decode (1.5e-5).
- **cleanup:** scratch sequences and the pool's cells free after every call; refusals (an empty block; an empty prefix;
  forks beyond `n_seq_max`) and a full pool (`no_cells`) leave nothing behind and the context still works.
- **mutation check:** with the fork `seq_cp` of a no-head block deleted the test fails 18 of 140 checks (worst difference
  1.27); with the head-fork replaced by a fork from the prefix, 6.

`bench/decide_batch_server_smoke.py` runs the real `llama-server` (CPU, private loopback ports, never the stack) on the
same model saved to GGUF (`test-decide-batch --save`): 53 checks. 501 without `--decide-seqs`; `/props`; 8 blocks over a
shared prefix compared with the server's own `/completion` (one token, `n_probs` 20) on the same tokens: largest
|log-prob| difference **4.2e-6** at `-b 64 -ub 16` and **7.3e-7** at the stack's `-b 1024 -ub 512`, same argmax for every block; every requested id returned; resident prefix reused whole /
extended / freed; a group larger than the forks (split, rows equal); the refusals (400/501); malformed JSON; and **a slot generating 48 tokens while 3-5 `/decide-batch` calls
run beside it produces exactly the tokens of the quiet run** (a hybrid slot's recurrent state is not disturbed).

NOT tested, and why: anything on CUDA (no GPU rule), the real tokenizer's behaviour at the prefix/block boundary (no model
with a vocabulary here; checked per call by `verify_tokenization` and once per structure by the Python side), the real
Bonsai's chat template (the Python side derives the wrappers from the served `/apply-template`), batch-invariance of the CUDA
kernels across ubatch shapes (section 8 counts flips).

## 6. The Python side (`mcp/decider_batch.py`, edits in `mcp/decider_bonsai.py`, `mcp/jev_api.py`, `mcp/decide_turn.py`)

Built against section 3 with a FAKE server only (`mcp/test_decider_batch.py`, 143 checks: a Qwen-like template, a toy
tokenizer, the sequential chat route and `/decide-batch` computing the same deterministic distribution from the full prompt
text); it has never spoken to the real engine endpoint.

- **Switch.** `YAMADORI_DECIDER_BATCH` ("1"/"on"), unset = off: requests and answers are byte-identical to today (tested).
- **Rendering without per-question `/apply-template`.** The wrapper pieces around each message's content are derived ONCE
  per (model, role structure, template flags) by rendering with sentinel contents through the served `/apply-template`;
  the prefix/block split is at the message boundary (the first k shared messages rendered alone must be a prefix of the whole
  rendering). Verified once per structure by a second, different message set (substitution must equal the server's rendering
  byte for byte) and a `/tokenize` boundary check; the engine's `verify_tokenization` repeats the boundary check on the first
  call. Any mismatch disables batching for that structure (cached) and the call falls back to the per-read path. A custom
  `render=` goes through the same derivation; `prior_for` questions fall back per question.
- **`read_many`** (decider_bonsai) returns exactly `[read(state, q) ...]`: `read()` was split into `_ask_result` /
  `_order_record` / `_assemble` so both paths share the temperature, exclusion, averaging, tie band and diagnostics code.
  The labels' ids are requested (`token_ids`), so no K re-read loop: `exact` is true (the difference from the sequential path
  is the sub-top-20 spelling mass it leaves unread, bounded by `unread_bound`, tested). Diagnostics differ by design:
  `read_path: "batch"`, equal-share `ms`, no `reads`/`k`.
- **Usage** (jev_api.usage_of): `prompt_tokens` = prefix + block, `cached_tokens` = the prefix's resident tokens,
  `processed` = the block's own (the engine charges a group's shared head to its first block) plus the prefix's on the call's
  first read: `input_tokens` equals the sequential path's on a cold and a warm prefix; `batch.accounted_processed` equals
  `timings.processed_tokens`.
- **Callers.** `decider_bonsai._decide_typed`, `jev_api._run` (single plans in one batch; two-stage `rounds`: stage 1 chunks in
  one batch, the final read a second batch on the kept prefix, then released), `decide_turn.Turn.decide` (a burst: the prefix
  kept between decides, released in `close()`).
- **Fallback.** 404/501/503, timeouts, 500 and garbage JSON: the answer comes from the per-read path for that call, and a
  negative is cached for 60 s (an injectable clock in the tests).
- **Scope of the resident prefix (engine side).** There is ONE resident prefix per server. Every call carries its full
  prefix and the engine compares tokens, so a concurrent burst with another state, or a `release` from another burst, can only
  cost a re-decode of the prefix, never a wrong row. `keep_prefix` is a hint, `release` frees whatever is resident.
- **Gaps.** A client cancel does not abort an in-flight `/decide-batch` (it goes through `_upstream`, not `model.post`; the
  call is bounded by its own timeout and short). `bench/decider/batch_ab.py` dry-runs (231 JevBench items, K = 1/2/5/10/26, 201
  rubric cases); its `--run --gpu-go` path is untested.

## 7. Expected speedup: arithmetic, with the fit it comes from

Two fits exist for a sequential read; both are used because they disagree.

- **Supplied** (the Decision Index run, 2026-10-06, 86 rows; from the request, not re-derived here): ~0.26 s per read +
  ~2.3 ms per processed token.
- **Computed here** from the real bonsai-a4000 JevBench run (`bench/decider/results/jevbench/bonsai-a4000-20261006-193326/
  items.jsonl`, 231 items, 2 reads each, state cold, `diagnostics.ms` against `diagnostics.processed_tokens`, ordinary least
  squares, `bench/decider/batch_arith.py`): **ms = 690 + 1.452 x processed_tokens, R^2 0.966** = 345 ms per read + 1.45 ms per
  token (the 64 items over 600 tokens alone: 711 + 1.445 x tokens). The slope is irreducible by any batching: every token
  still goes through the weights. The intercept is what batching can remove, and only the part of it outside the decode
  (HTTP, template rendering, tokenization of the whole prompt, slot bookkeeping, a checkpoint restore, the vocabulary
  softmax and JSON); what part of 345 ms that is, is NOT known. The engine's own floor per decode call is the weights read
  once: 5,946,648,928 B / 448 GB/s (RTX A4000 datasheet) = 13.3 ms.

**Upper bound on the JevBench public items** (K = 1 question per call = 2 reads; one request instead of two removes at most
one read's intercept): 231 x 345 ms = 79.7 s of the 407.1 s the run spent in reads = **x1.24 overall at best**, by size:

| processed tokens per call | items | median ms | bound |
|---|---|---|---|
| < 200 | 88 | 934 | x1.57 |
| 200-400 | 58 | 1,066 | x1.47 |
| 400-1,000 | 40 | 1,539 | x1.29 |
| >= 1,000 | 45 | 4,500 | x1.08 |

JevBench K = 1 is the WORST case for batching: the state's own prefill (identical in both paths, and what Clef pays too) is
most of a long item. The gain grows with the questions per call (decide_turn's sets, the Jev API's multi-question calls, the
skills window's 1,338 items over 193 cases = ~7 per case): sequential pays 2K intercepts, batched one (plus the engine's
floor per decode call). With the supplied fit (a = 0.26 s, b = 2.3 ms), block tokens from the template's structure (a block
= 40 + q tokens for a question of q tokens; shared head 12 + q; own tail 28; ESTIMATES until `processed` says otherwise) and
W = 4 (two questions a wave, two decode calls a wave, 13.3 ms each):

```
sequential = 2K a + b (S + 2K block)          batched = a + b (S + K (head + 2 tail)) + 13.3 ms x calls
```

| K | state S = 500 | S = 2,000 | S = 8,000 | S already resident |
|---|---|---|---|---|
| 1 | 1.99 -> 1.68 s (x1.19) | 5.44 -> 5.17 (x1.05) | 19.2 -> 19.1 (x1.01) | 0.84 -> 0.51 (x1.64) |
| 2 | 2.83 -> 1.90 (x1.49) | 6.28 -> 5.39 (x1.17) | 20.1 -> 19.4 (x1.04) | 1.68 -> 0.74 (x2.28) |
| 5 | 5.36 -> 2.63 (x2.04) | 8.81 -> 6.12 (x1.44) | 22.6 -> 20.1 (x1.13) | 4.21 -> 1.47 (x2.87) |
| 10 | 9.57 -> 3.81 (x2.51) | 13.0 -> 7.30 (x1.78) | 26.8 -> 21.3 (x1.26) | 8.42 -> 2.65 (x3.18) |
| 26 | 23.0 -> 7.63 (x3.02) | 26.5 -> 11.1 (x2.38) | 40.3 -> 25.1 (x1.61) | 21.9 -> 6.47 (x3.39) |

(q = 30; q = 60 changes the ratios by at most 0.1-0.2: `bench/decider/batch_arith.py` prints both.) Reading: **up to ~3x for a
many-question call on a small or resident state, ~1.2x for a lone question on a state of a few thousand tokens**, and no
better than the prefill of the state allows. The sharing of the question head is worth ~30% of a question's tokens
(70 -> 98 for the pair instead of 140). Clef-like single-pass speed for ONE long state is not available to a per-question
readout: the state's tokens are the floor.

What the table assumes and the measurement must check: that the intercept is mostly outside the decode (if it is mostly
`GGML_CUDA_BATCH_INVARIANT=1`'s slow small-batch kernels per decode call, the batched path pays it per wave stage and the
bound moves toward x1.5 instead of x3), that the 1.45-2.3 ms/token slope holds for 50-150-token ubatches (small ubatches
may run below tensor-core efficiency), and that the log-softmax (a 250k-vocabulary pass per block, ~2-5 ms) is not larger
than the per-read overhead it replaces.

## 8. The measurement plan (NOT RUN; "GPU go" first)

Prerequisites (operator): the candidate build (`C:/Users/jwals/engines/llama-bonsai2-ada-cand0042`, section 9) deployed to
bonsai-a4000 ONLY, with `--decide-seqs W` and `-c` lowered per section 4.1 (rerun `bench/a4000_fit.py` for the chosen W;
the main card and `config.yaml` stay as they are until the result is read). One GPU consumer at a time (AGENTS.md "Before you
claim anything works" 4): the A4000 is the skills agent's today.

Driver: `bench/decider/batch_ab.py` (dry-run by default; `--run --gpu-go` runs). Per item it runs the sequential path and
the batched path alternately (A,B / B,A across items), records per-call and per-question wall time and the engine's
`timings`, and compares answers.

1. **Wire and boundary (minutes).** `verify_tokenization` on every block of every item below: count `tokenization_ok` false
   (expected 0; any is a stop and a fix before anything else). Server log: the recurrent buffer size after the restart
   (expected `(n_parallel + 1 + W)` cells x 149.63 MiB) and `nvidia-smi` free memory at idle.
2. **Identity (the batch-invariance floor).** The 231 JevBench public items (K = 1) and the 193 labelled skill cases of the
   2026-10-06 window (decide_turn question sets, ~7 questions each, variants as that window ran them), sequential vs batched on
   the SAME reader, same session: (a) items whose argmax differs, (b) max |p| difference over every label of every order
   (compare with `TIE_BAND` 0.0034, which was measured as exactly this: cached vs `cache_prompt: false` on the main card;
   expected: the batched numbers sit inside it; CUDA with `GGML_CUDA_BATCH_INVARIANT=1` may be bit-equal), (c) tie flips (a
   decision that is a tie in one path and not the other), (d) the Jev outputs (probabilities, confidence) differ by at most (b).
   The sequential path's own repeatability (it reads to SD 0.002, docs/JJAVA.md 9) is the floor the batched difference is judged
   against: run the sequential path twice and report its self-difference beside the A/B difference.
3. **Speed by K.** K = 1, 2, 5, 10, 26 questions per call: each JevBench item's state with its own question plus the questions
   of the next K - 1 items (semantically odd, valid for speed and identity; capped by the 64k request limit), and the skills
   cases' native sets. Wall per call and per question, median and p90, sequential vs batched, W in {1, 2, 4, 8} (one restart
   each; W = 1 is the sequential-in-the-engine arm: one block a wave, no sharing, only the per-read HTTP and bookkeeping gone), with and without the shared head (`groups` of 2 vs 1). Each cell n >= 20 calls
   after 2 warm-ups, alternating; report the ratio with its bootstrap interval. Compare with section 7's table and say which
   assumption failed where they differ.
4. **Decomposition (what the 345 ms is).** From the response `timings`: prefix_ms / shared_ms / blocks_ms / decode_calls, and
   the request's wall time minus `total_ms` (HTTP + JSON + tokenize); against the sequential `prompt_ms` + `latency` per read.
   One 8k-state call at K = 10 with the server's own `--verbose` timing lines for a decode call. This is what decides whether
   W > 2 pays.
5. **Isolation under load.** The conversation's decode rate on slot 0 (the other card's conversation lives there) with and
   without a batched burst on the lane's behalf, n = 3 each as `bench/kv_rank.py --layout v3` measured the lane arm (a jjava
   burst on a lane cut decode ~30%, AGENTS.md): the batched path stalls the loop for the call's duration; report the stall and
   the decode rate over the burst.
6. **Memory.** `nvidia-smi` peak during a K = 26 call at W = 8 (the worst wave) against the idle figure; the pool's cells
   (`/slots`, `/props`) during and after; a deliberately oversized call (a state that fits but not the blocks) must answer 503
   and leave the pool clean.

Pass: step 1 zero mismatches; step 2 argmax flips no more than the sequential path's own repeat flips and every |p| difference
inside `TIE_BAND`; step 3 batched faster at every K >= 2 with the state cached or short; step 5 no corruption of slot 0
(its next 48 tokens at temperature 0 equal a quiet run, the check the smoke test makes on the CPU). Then, and only then, the
switch `YAMADORI_DECIDER_BATCH` defaults ON for the model whose profile records the measurement (`read_regime`'s pattern:
a measured field, never borrowed).

## 9. The candidate build and where things are

- Source of truth: `engines/patches/llama-bonsai2-ada/0042-server-decide-batch.patch` (sha256 in the manifest; applies after
  0041; `python scripts/build_engine.py vendor llama-bonsai2-ada` re-derived `engines/src/llama-bonsai2-ada` from base +
  42 patches, `check --derive` ok; the vendored tree equals the dev tree the tests ran on, `diff -r` empty).
- Candidate: `python scripts/build_engine.py build llama-bonsai2-ada --jobs 8 --out C:/Users/jwals/engines/llama-bonsai2-ada-cand0042`
  (the shipped flags, CUDA 86;120, no UI; its static CPU test tree builds and runs `test-decide-batch` beside
  `test-reasoning-budget` and `test-chat`). Not deployed, not in `config.yaml`, not in `shipped`.
  BUILT 2026-10-07 02:26 from the vendored tree (patch sha256 in the manifest) into that directory: all 34 build checks passed,
  CMakeCache identical to the original's (`cache_diff_vs_original` empty), and its static CPU test tree ran 3/3 under ctest
  (`test-reasoning-budget`, `test-chat`, `test-decide-batch`). `llama-server.exe` sha256 937c6b62..., `ggml-cuda.dll` 9794c9ef...
  (`build-report.json`). The CUDA binary has NOT been run: no GPU.
- To deploy later (operator decisions): `--decide-seqs W` and the lower `-c` on `bonsai-a4000`'s argv, then
  `build_engine.py --describe` into `shipped`, `deploy_check.py`.
- The engine's dev tree and the CPU test binaries (`C:/Users/jwals/engines/dev-decide/`) are scratch.

## 10. Open decisions and known gaps

- W and `-c` (section 4.1): memory against how many questions share a wave. W = 2 runs a question a wave and already has one request instead of 2K;
  W = 4 halves the waves for ~300 MiB more.
- **The resident prefix is not time-limited.** A call with `keep_prefix` leaves the state's cells in the pool until a later
  call or `release`; the Python side releases in `finally`, a crashed caller would leave it until the next call (which
  reuses or resets it). No timeout is added because none can be derived; if the operator wants one it is a number to choose.
- It replaces the lane for jjava's reads: the batched path uses the engine's own scratch sequences, not slot 1. The lane slot
  is still acquired by the callers (harmless) and still serves the sequential fallback.
- Scores with temperature, tie bands, the content-free prior and every readout rule are unchanged: only where the
  next-token distribution comes from changes. `exact` is now true for every label (the labels' ids are requested), where the
  sequential path leaves spellings outside its top-K unread within `unread_bound`.
- The shared-head sharing assumes the two orders' blocks begin alike; when they do not (a question whose options come first)
  the engine finds no shared head and decodes both whole, correctly.
- Not looked at: the main card (`bonsai`, `flash-next`, `mirai-s`: other engines; the patch is ours for `llama-bonsai2-ada`
  only; jjava's helper `bonsai-a4000` is the only reader this is built for).

## 11. Deploy to bonsai-a4000 (2026-10-07, operator: "make sure one pass read is working, turned on and deployed with everything else")

**Verified before any GPU run (CPU, the REAL model).** `bench/decider/batch_cpu_real.py` against a private CPU `llama-server`
(this patch's build, `Ternary-Bonsai-2-27B-PTQ1_0.gguf`, `-ngl 0 -np 2 --kv-unified --decide-seqs 2 --jinja`): the render
derivation (the served template through `/apply-template`, byte-for-byte on a second message set), the real tokenizer's boundary
check (`/tokenize` and the engine's `verify_tokenization`) all passed; three questions (6 orders) answered in ONE
`/decide-batch` request against 6 per-read requests: **0 of 3 argmax flips, largest |p| difference 4.9e-4** (TIE_BAND 0.0034),
prefix 80 tokens, 303 tokens decoded, 51 saved by the shared question heads, 7 decode calls in 3 waves (89 s against 106 s
sequential: CPU, not a speed figure). Also offline: the structure derives and verifies against the repo's real Bonsai template
fixture (`mcp/fixtures/bonsai_chat_template.jinja`), splitting at `<|im_end|>\n | <|im_start|>user`.

**W = 2.** One question (its two orders) per wave: the minimum that has one request per call and shares each question's
head. Memory: prefix + 2 forks = 3 cells x 149.63 MiB = 448.9 MiB = 13,134 KV cells at 35,840 B (`bench/a4000_fit.py`, fit.json
`bytes_per_cell`), rounded up to whole 1,024s = 13,312 cells: `-c` 141,312 -> **128,000**; the net VRAM change is +448.9 - 455.0 =
-6 MiB, so the fit's 1,331 MiB headroom is kept (today: 12,854 MiB used on the A4000 beside embeddings, 3,313 free). W = 4 would
cost 22.5k more cells of window to halve the waves (arithmetic: ~0.35 s of an 11 s K = 26 call); the first measurement
(`timings.waves`, `decode_calls`, `blocks_ms`) says whether it pays.

**The exact changes (the operator's restart; nothing here edited by the builder).**

1. `config.yaml`: a new macro beside `server_nudge` (the main card and everything else keep the 80d2c60d binary):
   `server_decide: "C:/Users/jwals/engines/llama-bonsai2-ada-cand0042/src/build/bin/llama-server.exe"`.
2. `config.yaml` `bonsai-a4000` `cmd`: first line `${server_nudge}` -> `${server_decide}`; `-c 141312` -> `-c 128000`; add
   `--decide-seqs 2` (after `--kv-unified`). Everything else unchanged (`-np 2 --kv-unified -b 1024 -ub 512`, the batch-invariant
   env).
3. `mcp/tier_models.yaml` `bonsai-a4000.window`: `ctx: 128000`, `main_cap: 124928` (= ctx - the 3,072-cell lane, as 141,312 ->
   138,240), so the proxy budgets against what the server serves.
4. `scripts/start-stack.bat` (the proxy and worker environment): `set "YAMADORI_DECIDER_BATCH=1"` -- ONLY after step 3 of the
   measurement below passes (the code path is a no-op while unset and falls back per call if the endpoint is absent).
5. `engines/manifest.yaml` is already updated (this commit): `shipped` = cand0042, 80d2c60d moved to `previous` (the main card and
   the rest still run it); `build_engine.py --verify-only`: 7/7 match.

**After the restart (the operator's "GPU go" on the A4000, the main card untouched).** The reader must be LOADED first (a request
through the proxy loads it; never GET /upstream/<model> for an unloaded one). Then:

    python bench/decider/batch_ab.py --run --gpu-go --sets jevbench          # identity + speed, K = 1, 231 items
    python bench/decider/batch_ab.py --run --gpu-go --sets k --ks 2,5,10     # speed by K
    python bench/decider/batch_ab.py --run --gpu-go --sets skill             # the labelled skill cases

and the checks of section 8 steps 1, 2, 5, 6. PASS = zero `tokenization_ok` false; argmax flips no more than the sequential path's
own repeat flips and every |p| difference inside TIE_BAND; batched faster at every K >= 2; slot 0's generation unchanged beside a
burst. If identity holds, step 4 above.
