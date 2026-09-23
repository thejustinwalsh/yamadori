# bench/longctx -- speed and accuracy against context length

The operator's question, in their words: *"For 256k context, if it drifts in
long context, we keep a smaller context, we will not quant kv cache lower, and
I want to make sure we benchmark our current model and the new model,
including base metrics like tokens per second over context length, accuracy
etc."*

So this measures, for ONE served model/config at a time, at context lengths
from 2k up to whatever the server was launched with:

| what | how | source |
|---|---|---|
| prefill tok/s | a prompt of L tokens with a fresh nonce as its first text, so nothing is reused (`cold` rows; the server's `cache_n` confirms it) | llama-server `timings.prompt_per_second` |
| decode tok/s | a fixed 256 tokens (`ignore_eos`), temperature 0, thinking off, for three content types (prose, code, json) off the same cached haystack | `timings.predicted_per_second` |
| MTP draft acceptance | per request, by content type and context length; null when speculation is off | `timings.draft_n_accepted / draft_n`, else the `/metrics` `spec_decode_*` counter delta around the request |
| VRAM | card used before, peak during (1 s poll), after | `nvidia-smi -i <card>` |
| accuracy (drift) | needle retrieval and a reasoning step over real source code | exact-match grader |
| usable context | the largest L not significantly below the 8k accuracy | exact McNemar, paired |

All rows record the model path, n_ctx and build (`GET /props`), the launch
flags (detected from the process, or `--flags`), the actual prompt tokens the
server counted, `finish_reason`, and how thinking was set.

Files: `haystack.py` (probe generation and grading), `run.py` (the driver),
`analyse.py` (curves, usable context, dashboard JSON, markdown),
`test_longctx.py` (offline proofs; `scripts/run_tests.py` runs it).

## The accuracy probe

A haystack is a concatenation of **real package source** --
`index/packages/_src/` (typegpu, three-mesh-bvh, koota, wgpu-matrix,
postprocessing, drei, fiber, ..., three.js last), 13.1 M characters after
dropping minified files, `.d.ts`, `.cjs` and exact duplicates, which covers a
256k-token haystack about four times over. Into it go, per item:

- **4 needles**: `export function velKorthaBrin() { return 4821; }`, each at
  one of the depth bins 10/30/50/70/90% (four of the five, chosen per item),
  inserted at a line boundary;
- **4 decoys**: the same name with the last syllable changed
  (`velKorthaDask`) and a different value, at random depths. A reader that
  matches "roughly that name" gets the decoy's number.

Three questions per haystack, answered on an `ANSWER:` line and graded by
exact match (`haystack.grade`):

| task | question | correct when |
|---|---|---|
| `single` | what does `A` return? (the needle in bin item mod 5) | the integer is exact |
| `multi` | what does each of `A`,`B`,`C`,`D` return? | all four exact (per-needle results kept for the depth grid) |
| `reason` | x = `A`'s value, y = `B`'s; x+y or \|x-y\| | the integer is exact |

The three questions share one haystack and are sent with `cache_prompt:true`,
so a haystack is prefilled once, not three times; only the question differs.

**Items are paired across lengths.** Names, values, decoys, depth bins and
questions depend only on (seed, item); only the amount of filler changes with
L. So "item 7 at 8k" and "item 7 at 128k" are the same question, and a drop is
tested within item. Everything is deterministic from `--seed`: a resumed or
repeated run rebuilds byte-identical prompts (the chars-per-token calibration
is stored in `manifest.json` and reused).

**Outcomes** (`outcome` on each row): `correct` / `wrong` are scored.
`budget` (finish `length` with no ANSWER line) is a budget event, never an
answer, and is not scored. `stack_error` (HTTP error, timeout, 429, context
overflow) is never scored as wrong and is retried on the next run of the same
run id. Three consecutive stack errors stop the run as an outage (PROTOCOL
rule 2). A context-overflow error ends the ladder at that rung.

## Speed rows and draft acceptance

Each speed rep at a rung is one haystack of real source and three requests:

| content type | asks for | why |
|---|---|---|
| `prose` | a file-by-file summary of the code | ordinary explanation; sent first with a fresh nonce, so it is the **cold prefill** measurement |
| `code` | a new TypeScript module using the APIs above | the thing this stack mostly generates |
| `json` | every exported function as a JSON array | structured, repetitive text -- the easy case for a draft head |

`code` and `json` reuse the cached haystack (`cache_prompt:true`, same
nonce), so they cost decode only; prefill statistics use cold rows alone.
Every row, speed or accuracy, records `draft_n`, `draft_n_accepted`,
`draft_accept_rate` and `draft_source`: the reply's own `timings` when the
build exposes `draft_n` there (the PrismML fork does, in
`server_slot_stats::to_json`), else the difference of
`llamacpp:spec_decode_num_draft_tokens_total` / `..._accepted_tokens_total`
on `/metrics` before and after the request (the server must be started with
`--metrics`; exact only with `-np 1` and nothing else running), else null.
Nothing drafted means null, never "0% acceptance". `analyse.py` reports
acceptance next to decode tok/s per content type, plus a **draft acceptance
by content and context depth** table (content types: the three above and the
accuracy answers, `answer:single|multi|reason`), pooled as accepted / drafted
over each cell.

## Statistics, stated before any data (PROTOCOL rule 4)

- Every accuracy cell has a Wilson 95% interval.
- **Usable context** = the largest rung L such that neither L nor any rung
  between 8k and L is a significant drop from the 8k rung. A drop is: on the
  items answered at both lengths, *right at 8k and wrong at L* outnumbers the
  reverse, and exact McNemar (two-sided, `livecodebench.mcnemar`) gives
  p < 0.05 / m, Bonferroni over the m longer rungs tested. Computed per task;
  the headline is the minimum over tasks. Uncorrected p < 0.05 drops are
  listed as early warnings.
- **Power at the default n = 20 items per (L, task).** With m = 5 longer rungs
  (a 163,840 server) a drop is only detectable if at least 8 items flip one
  way and none the other: **the minimum detectable drop is 40 points**
  (30 points uncorrected). With m = 9 (a 262,144 server: 16k ... 240k) it
  is 9 flips, 45 points. The Wilson interval on a 90% cell at n = 20 is roughly +/-13
  points. n = 20 therefore finds a *collapse*, not a gentle slope. `--items 40`
  halves the detectable drop (9/40 = 23 points at m = 9) at twice the cost;
  the multi task's per-needle depth grid has 4x the observations per rung and
  is the finer (but within-item correlated) view.
- A config comparison (A vs B, `analyse.py results/A results/B`) pairs the
  same items at the same rung (same `--seed`) and reports exact McNemar per
  (L, task), Bonferroni over the cells, plus prefill/decode speed ratios.

## Why the real sweep needs a dedicated llama-server

Through the proxy on `:1234` a single request can hold at most the **main
share** of the KV pool: 5/8 of it (`mcp/budget.py`), 102,400 tokens of today's
163,840, and the proxy reserves at least 3,072 of those for thinking + answer
(`tiers.A_MIN` + `MIN_THINKING`). That is by design -- the other 3/8 belongs to
deep thinking -- so the ladder through the proxy stops at 64k. The proxy also
re-assembles the upstream stream and drops llama-server's `timings`, forces
thinking on (every tier thinks), and raises `max_tokens` to at least 2,048, so
through it speed can only be wall clock (time to first streamed token for
prefill, inter-token time for decode), with proxy overhead in it and a decode
length the model chooses. `run.py` supports that path, labels it
`wallclock_stream`, and analysis never pools it with `server_timings`.

The characterisation -- 256k, both models, MTP on and off -- therefore runs
against a **dedicated llama-server with `-np 1`** (one slot owns the whole
pool) in a maintenance window, with nothing else on CUDA0. The production
server on `:10001` runs with the default slot count over a unified pool; a
request there would compete with live traffic and the worker.

### Maintenance window: what the operator does first

Stop the production stack's use of CUDA0 (watchdog, then llama-swap), so the
card is empty. `run.py` never starts or stops anything itself; it refuses to
start while the card is busy (>30% utilisation over 10 s), while another
process holds the worker's gpu-lane pause, or while a known benchmark process
is running.

Set `CUDA_DEVICE_ORDER=PCI_BUS_ID` in the shell first (as
`scripts/start-stack.bat` does), so `-dev CUDA0` is the 5060 Ti and
`--gpu-index 0` in nvidia-smi is the same card.

### Model A -- current production weights and build, KV q8_0

The production flags from `config.yaml`'s `bonsai` block, with `-np 1`, a
private port and the largest `-c` that loads. 262,144 is the model's native
context; `mcp/budget.py` puts q8_0 KV at ~44 KiB/token (11 GiB at 262,144,
which on paper does not fit beside 5.95 GB of weights on 16.3 GB -- the
config.yaml comment's 34 KiB/token figure says it might). Try it; if the load
or the first long prefill fails, step down (229376, then 196608) and the
ladder stops at whatever n_ctx the server reports. **Never lower the KV type
below q8_0.**

```
set CUDA_DEVICE_ORDER=PCI_BUS_ID
C:/Users/jwals/llamacpp-prism-official/build/bin/llama-server.exe ^
  --host 127.0.0.1 --port 18080 ^
  -m C:/Users/jwals/textgen/user_data/models/Ternary-Bonsai-2-27B-Abliterated-PTQ1_0.gguf ^
  -dev CUDA0 -ngl 999 -c 262144 -np 1 ^
  --cache-type-k q8_0 --cache-type-v q8_0 -fa on --cont-batching -b 1024 -ub 512 --jinja ^
  --reasoning-budget 32768 ^
  --reasoning-budget-message "Thinking budget reached. I will stop deliberating and write the final answer now." ^
  --reasoning-format deepseek --no-context-shift --metrics ^
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.05 --presence-penalty 0 --repeat-penalty 1.0

C:/Users/jwals/textgen/installer_files/env/python.exe bench/longctx/run.py ^
  --url http://127.0.0.1:18080 --label A-prism-q8 --run-id A-prism-q8 --no-pause-lane
```

(`--no-pause-lane` because the production stack, and so the worker's model,
is down in the window; leave it off if the worker is still running.)

### Model B -- the new build, MTP off and on

The build staged at `C:/Users/jwals/llamacpp-sudoingx-bonsai2`; its findings
land in `docs/MTP-STAGING.md`, and the flag spellings below are the ones the
staging brief gave -- check that document before the window. Same flags as A
except the binary and model; one run per config, each under its own run id
and the SAME `--seed` (default 1) so every config answers the same items.

```
rem B, MTP OFF
C:/Users/jwals/llamacpp-sudoingx-bonsai2/build/bin/llama-server.exe ^
  --host 127.0.0.1 --port 18080 ^
  -m C:/Users/jwals/textgen/user_data/models/Ternary-Bonsai-2-27B-PTQ1_0-mtp-lean.gguf ^
  -dev CUDA0 -ngl 999 -c 262144 -np 1 ^
  --cache-type-k q8_0 --cache-type-v q8_0 -fa on --cont-batching -b 1024 -ub 512 --jinja ^
  --reasoning-budget 32768 ^
  --reasoning-budget-message "Thinking budget reached. I will stop deliberating and write the final answer now." ^
  --reasoning-format deepseek --no-context-shift --metrics ^
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.05 --presence-penalty 0 --repeat-penalty 1.0

C:/Users/jwals/textgen/installer_files/env/python.exe bench/longctx/run.py ^
  --url http://127.0.0.1:18080 --label B-mtp-off --run-id B-mtp-off --no-pause-lane

rem B, MTP ON: the same command line plus
  --spec-type draft-mtp --spec-draft-n-max 1

C:/Users/jwals/textgen/installer_files/env/python.exe bench/longctx/run.py ^
  --url http://127.0.0.1:18080 --label B-mtp-on --run-id B-mtp-on --no-pause-lane
```

For the abliterated MTP graft, swap `-m` for
`Ternary-Bonsai-2-27B-Abliterated-PTQ1_0-mtp-lean.gguf` and use its own label
and run id. Note that the MTP draft context takes VRAM of its own, so B with
MTP on may load a smaller `-c` than B with it off; the manifest records the
n_ctx each run actually had, and the ladder stops there.

Two cautions carried from `config.yaml`: a sudoingX build has been
numerically broken before (`pr-ptq1-mmv`: "generation collapses into runs of
'/'"), which this probe catches as wrong answers at every length, including
2k -- look at the 2k/8k accuracy before reading any speed number. And MTP
acceptance depends on the text being generated, which is why it is reported
per content type (prose, code, json, and the short accuracy answers) rather
than as one number; none of them is a counting task that would inflate it.

Then compare:

```
C:/Users/jwals/textgen/installer_files/env/python.exe bench/longctx/analyse.py ^
  bench/longctx/results/A-prism-q8 bench/longctx/results/B-mtp-off bench/longctx/results/B-mtp-on
```

It writes `results/_compare/summary.json` (with a `dashboard` block shaped
like `bench/domain/analyse.py dashboard_block`: state / file / how / arms /
progress / power / arm_table / pairs, plus `curves`, `depth_grid` and
`usable`) and `summary.md`.

Useful switches: `--plan` prints the rungs that fit and the request count and
sends nothing; `--max-minutes` stops cleanly (resume with the same run id);
`--thinking budget:N` measures accuracy with thinking on under a fixed budget
(recorded on every row; default is `off`); `--ladder`, `--items`,
`--speed-reps`, `--decode-tokens`, `--seed`, `--max-ctx`.

## Validation run (2026-09-22)

VALIDATION_PLACEHOLDER

## Expected wall time of a full sweep

ESTIMATE_PLACEHOLDER
