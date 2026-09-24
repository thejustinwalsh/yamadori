# LiveBench on Yamadori

LiveBench (https://livebench.ai, harness https://github.com/LiveBench/LiveBench)
run against the local model through the door users use, the proxy on `:1234`,
in two arms:

| arm | display name in LiveBench files | `X-Yamadori-Features` |
|---|---|---|
| `bonsai` (bare model) | `yamadori-bonsai-arm` | `{"retrieval": false, "hints": false, "investigate": false, "fanout": 1, "effort": "medium"}` |
| `yamadori` (full stack) | `yamadori-full-arm` | `{"retrieval": true, "hints": true, "investigate": true, "fanout": 3, "effort": "medium"}` |

The arm-to-header table is in one place, `drive.py:ARMS`. The proxy silently
drops a header it cannot parse, so every answer's `api_info.x_yamadori` is
checked against the arm, and a mismatch stops the run.

Results, numbers and caveats for a given night are in `results/<run-id>/`
(`summary.json`, `comparison.md`, `RESULTS.md`).

## Which questions, and why not the current leaderboard's

- The current leaderboard release is **2026-06-25** (site bundle, fetched
  2026-09-23). Its questions are **not public**. LiveBench withholds recent
  questions to limit contamination, and the HF datasets `livebench/<category>`
  were last updated 2025-04-07.
- The latest release with **every public question** is **2024-11-25**, which
  is also what the harness README tells third parties to use. It has 1000
  questions:

  | category | tasks (questions) |
  |---|---|
  | coding (128) | LCB_generation 78, coding_completion 50 |
  | reasoning (150) | spatial 50, web_of_lies_v2 50, zebra_puzzle 50 |
  | instruction_following (200) | paraphrase, simplify, story_generation, summarize, 50 each |
  | math (232) | AMPS_Hard 100, math_comp 96, olympiad 36 |
  | data_analysis (150) | cta, tablejoin, tablereformat, 50 each |
  | language (140) | connections 50, plot_unscrambling 40, typos 50 |

  (`make_sample.py` recomputes these counts from HF with LiveBench's own
  release/removal filter.)
- Consequences, stated before any number: (1) our scores are comparable
  with LiveBench's **2024-11-25** table and per-question judgments. They are
  **not** comparable with the 2026-06-25 leaderboard, which uses different
  questions and a different task mix per category. (2) These questions have
  been public since April 2025, and our base, Qwen3.8-27B, was released
  2026-08-05. Treat every score as **possibly contaminated**.

## Model lineage (verified on the HF cards, 2026-09-23)

`Qwen/Qwen3.8-27B` (created 2026-08-05) -> `prism-ml/Ternary-Bonsai-2-27B-gguf`
(`base_model: Qwen/Qwen3.8-27B`) -> `BoldingBuilds/Ternary-Bonsai-2-27B-Abliterated-PTQ1_0-GGUF`
(`base_model: prism-ml/...`). The served file,
`Ternary-Bonsai-2-27B-Abliterated-PTQ1_0-mtp-lean.gguf`, is that
abliterated PTQ1_0 file with Qwen's MTP head grafted on (`config.yaml`).
GGUF header: `general.architecture = qwen35`, `size_label = 27B`, and
sampling defaults temp 1.0 / top_p 0.95 / top_k 20.

## Setup (WSL Ubuntu, not the stack interpreter)

The harness needs Linux tooling for code execution, so it runs in WSL. From
WSL, `localhost:1234` does not reach Windows. The Windows host on the WSL
switch does: `http://172.29.32.1:1234/v1`.

```bash
mkdir -p ~/livebench-run && cd ~/livebench-run
git clone https://github.com/LiveBench/LiveBench.git     # HEAD 0baa4aa7 (2026-09-23)
uv venv -p 3.11 .venv && . .venv/bin/activate
cd LiveBench && uv pip install -e . && uv pip install -r livebench/code_runner/requirements_eval.txt
uv pip install pip && python -m spacy download en_core_web_sm   # IF judge needs it; uv venvs have no pip
git apply /mnt/c/Users/jwals/llama-stack/bench/livebench/patches/livebench-yamadori.patch
cp /mnt/c/Users/jwals/llama-stack/bench/livebench/patches/yamadori_local.yml livebench/model/model_configs/
```

Versions: Python 3.11.15, livebench 0.0.4 (HEAD 0baa4aa7), openai 2.54.0,
httpx 0.28.1, datasets 5.0.1, nltk 3.10.3, spacy 3.8.16.

## Patches to LiveBench (`patches/livebench-yamadori.patch`)

All changes are marked `[yamadori patch]`. None changes a scoring rule.

1. `model/completions.py`, `chat_completion_openai` (the path for an
   OpenAI-compatible `--api-base`):
   - `default_headers` taken from env `LB_EXTRA_HEADERS` (a JSON object).
     This is how `X-Yamadori-Features` reaches the proxy. Unset means
     upstream behaviour.
   - Records `finish_reason` and the proxy's `x_yamadori` decision record in
     the answer's `api_info`. Upstream records neither.
   - When `content` is empty, the error names the `finish_reason`, so a
     `length` stop (a budget event) can be told apart from a transport
     failure.
   - `TIMEOUT` can be raised (never lowered) through env `LB_TIMEOUT`. We
     use 7200 s: one local GPU, long thinking.
   - **429 handling (added mid-run, see below).** On `RateLimitError` ("all N
     main lanes busy") the call retries in-process after 1–2 s of jitter,
     for up to `LB_429_RETRY_S` (default 3600 s), then surfaces the 429. The
     SDK's own retries are turned off (`max_retries=0`) when the proxy
     headers are set, so this loop is the only retry policy. The count is
     recorded per answer as `api_info.retries_429`. Checked offline against
     a stub that returns 429, 429, 200 (5/5: three requests, 1–2 s gaps,
     count recorded).
2. `gen_ground_truth_judgment.py`, old-format instruction following
   (releases before 2025-11-25): `list(set([m.question ...]))` raised
   `TypeError: unhashable type: 'dict'`, which killed the whole judgment
   run for any batch containing IF. The fix de-duplicates by `question_id`
   and scores each IF task separately, so result rows carry their own
   `task` (upstream stamped the first question's task on all of them).
   `validate_scorer.py` found it.
3. `model/model_configs/yamadori_local.yml` (new file): the two display
   names, `api_name: {local: yamadori}`, and sampling temp 1.0 / top_p 0.95.
   This mirrors LiveBench's own `qwen3.8-27b` entry (`qwen.yml`), which is
   also Qwen's thinking-mode card and the served default.
   **`max_tokens` is the harness default, 4096.** The proxy treats it as the
   *answer* allowance and adds the thinking budget on top (about 97k tokens,
   shown per answer in `x_yamadori.budget`). LiveBench's own Qwen3.8-27B row
   used `max_tokens: 32768` total at 32k context.

`content` is what gets scored. `reasoning_content` is stored separately
under `choices[0].reasoning`, never in `turns`, and the judge also strips any
`<think>` block.

## Running

The key never appears on a command line, in a file or in output. `drive.py`
reads it from `--key-file` into the child's `LIVEBENCH_API_KEY`.

```bash
# fix the question order once (stratified: shuffled per task, interleaved)
python make_sample.py --release 2024-11-25 --out results/<run-id> coding reasoning instruction_following math data_analysis language

# one arm x category, ONE request in flight (gen_api_answer --parallel 1),
# stopping cleanly at a deadline; re-running resumes and retries transport errors
KEY_FILE=/path/to/key bash run_arm.sh bonsai   coding <run-id> 0 <deadline-epoch>
KEY_FILE=/path/to/key bash run_arm.sh yamadori coding <run-id> <n> <deadline-epoch>

# official judge on what was answered, then per-question rows
python score.py --run-dir results/<run-id> --arm bonsai --category coding

# dashboard JSON (bootstrap 95% CIs) and the leaderboard comparison
python summarize.py results/<run-id>
python compare.py results/<run-id> reference/leaderboard_2026-09-23.json reference/table_2024_11_25.csv
```

Checks:

- `python test_summarize.py`: parser/aggregation test, offline; prints `N/M checks passed`.
- `python validate_scorer.py reasoning instruction_following math data_analysis language`:
  runs the official judge on ground-truth-shaped right and wrong answers
  (33 checks, no model call).
- `compare.py`'s reproduction table recomputes each reference model from
  LiveBench's per-question judgments over the full release, and diffs it
  against the published `table_2024_11_25.csv`.

## Results so far (2026-09-23 13:45 EDT), run `lb-20260923-minp0`, coding

Questions: the same 21 coding questions for every arm, a stratified random
sample of LiveBench release 2024-11-25 (11 LCB_generation, 10
coding_completion; `make_sample.py`, seed 20260923). Local ternary Bonsai 2
27B (derived from Qwen3.8-27B) on one RTX 5060 Ti, server min_p 0.0.

| arm | official score [95% CI] | LCB_generation | coding_completion | `.buffer` sensitivity |
|---|---|---|---|---|
| `bonsai`: bare @ medium (all augmentation forced off) | 70.9 [51.4, 90.0] | 9/11 | 6/10 | 75.5 |
| `yamadori-xhigh`: every augmentation allowed, medium effort | 72.3 [54.1, 90.5] | 6/11 | 9/10 | 90.5 |

- Paired (xhigh minus bonsai, n=21): +1.4 points [95% CI -17.7, +20.5].
  Discordant pairs 3 and 3, exact McNemar p = 1.000. This n cannot
  separate the arms (PROTOCOL rule 4).
- `.buffer` sensitivity: LiveBench's LCB grader gives stdin programs
  StringIO stdin/stdout without `.buffer`. Five answers that pass on a real
  stdin score 0 (1 bonsai, 4 xhigh). With `.buffer` provided, xhigh's 3
  LCB losses disappear and the discordant pairs become 3 and 0. The official
  scores are what we report, since every leaderboard model faced the same
  grader. The sensitivity column is shown beside them, never instead
  (`scan_stdin_buffer.py`, `results/lb-20260923-minp0/sanity.md`).
- Mechanisms on xhigh (from `x_yamadori`, all 21 answers): fan-out ran on
  21/21, sequential: 3 steps on 16, 2 steps on 5. Selection was code_medoid
  12, code_grade 5, fallback 4, and a different candidate's code was
  delivered (`replaced`) on 11. Repair ran 1 round on 3/21 (stopped clean
  20, repeated_answer 1), turning one parse error into a correct answer
  (7878539e). Deep thinking was decided on
  0/21: LiveBench's domains are outside the held package sources, so the
  code tools, and with them deep thinking, are withheld. **Deep thinking was
  not exercised by this benchmark.** The tool-turn cap was never hit.
  Stack errors 0.
- Reference, same 21 questions, from LiveBench's published per-question
  judgments: Gemini 2.5 Pro 90.0, Claude 3.7 Sonnet thinking 85.9, o3-mini
  high 81.4, QwQ-32B 75.5, Qwen2.5-Max 71.8, GPT-4o 67.3, Qwen2.5-Coder-32B
  62.3, Gemma 3 27B 47.3 (`comparison.md` table A). The current leaderboard
  (2026-06-25, including Qwen3.8 27B) uses unpublished questions and is
  context only (table C).
- Partial and set-aside arms: tier max (2 rows, dropped by the operator),
  bonsai+check (2 rows, paused), instruction_following (not yet run under
  this condition). None of them is reported as an arm result.

## Runs

| run | server min_p | what it holds | status |
|---|---|---|---|
| `lb-20260923` | 0.05 (llama.cpp default, never measured) | bonsai coding, n=21 | **record only**, frozen 07:20 EDT. Its IF answers (8 complete, plus 2 from the stopped chunk) are in `discarded_if_minp005/` and are not reported |
| `lb-20260923-minp0` | 0.0 (vendor-published; operator decision ~07:10 EDT; server restart ~07:16, checked in llama-swap `/running`) | bonsai and yamadori coding on the same 21 questions (paired), then bonsai instruction_following from question 1 | current |

Both runs use the same question order files (same seed), so the 21 coding
questions are identical across runs and arms. From `lb-20260923-minp0` on,
2 requests are in flight (two streams of one each). SWE-bench is stopped
and the domain suite still shares the GPU. `condition.json` in each run
dir records the sampling and sharing.

## Changes during run lb-20260923

- **03:59:29 EDT, 429 retry.** At the coordinator's request, and the same
  fix SWE-bench got. Before this, a 429 failed the question: the SDK retried
  twice, then the driver swept the error row, backed off 90 s and relaunched
  the harness, about 2 minutes lost per 429 and the freed lane given to
  whichever client retried sooner. From 03:59:29 each harness process
  retries the 429 itself after 1–2 s of jitter. It took effect at `drive.py`'s
  next chunk (every chunk is a new `gen_api_answer.py` process, so the
  in-flight question was not interrupted). The change is harness-only and
  scores are unaffected: a 429 was never scored before or after, only
  retried. Answers from 03:59:29 on carry `api_info.retries_429`; earlier
  ones do not.
- **~02:40 EDT, plan change.** Bonsai arm only, coding then
  instruction_following, time-boxed (`plan_lb-20260923.sh` explains why).
  The two yamadori smoke answers were moved to
  `results/lb-20260923/smoke_yamadori_arm/` and are not in any summary.
- **~07:30 EDT, arms redefined as product tiers** (operator). `yamadori` =
  body `reasoning_effort: "max"`, no header; `minimal` = `"minimal"`, no
  header; `bonsai` = "bare @ medium" (header forces everything off). The
  tier also raises the thinking effort sent upstream (minimal=low ...
  max=xhigh), so tier arms against bare @ medium mix effort and tools. The
  arm check verifies tier, `effort_sent`, nothing forced, and what the tier
  allowed. Header-forced yamadori answers from before the fan-out winner fix
  are in `results/lb-20260923-minp0/prefix_yamadori_features_arm/`, labelled
  pre-fix and never reported.
- **~08:15 EDT, per-answer mechanism evidence** (`mechanisms.py`,
  `test_mechanisms.py`). Every row carries a `mechanisms` record from
  `x_yamadori`, `summary.json` has `mechanism_health` per arm, and
  `comparison.md` has the table. A broken mechanism is a `stack_error`: not
  scored, re-asked, with the full row kept in `stack_errors.jsonl`. Per-call
  retrieval results come from `x_yamadori.tools` once the proxy emits it;
  answers from before that show retrieval "produced" as `--`.
- **~08:20 EDT, `X-Yamadori-Session`.** A fresh nonce per answer (per call;
  LiveBench questions are single-turn), recorded as
  `api_info.session_nonce`, so no two questions or arms share a proxy
  session. It takes effect for each driver at its next chunk, and in the
  proxy from the deploy that introduced the header; before that the header
  was ignored. `test_harness_patch.py` checks it offline with the 429 loop
  (8/8, WSL venv).
- **08:34:28 EDT deploy** (enforced vendor sampling, fan-out with the
  original as a candidate and the code-medoid winner delivered,
  `x_yamadori.tools`, `X-Yamadori-Session`, check_code, repair at high/max).
  Bonsai answers produced before it, or in a harness process without the
  session nonce, are in `results/lb-20260923-minp0/predeploy_bonsai/` and
  were re-asked. The feature freeze applies from here: only fixes to broken
  machinery, each with a re-run of the affected rows.
- **09:02:36 EDT deploy: code-task fan-out at high/max.** Tier-max answers
  from before it are in `prechange_tier_max/`. Tier max passed its n=1 gate
  at 09:03:45 (fanout decided_n=3, code_medoid, delivery and candidate
  parses recorded).
- **~09:40 EDT, two operator decisions.** (1) Fan-out is redesigned as
  sequential candidates through the second brain. Tier-max answers since GO
  are in `pre_sequential_fanout_tier_max/`, and tier max re-runs from
  question 1 after that deploy and a new gate. (2) One request in flight in
  total across all benchmarks. LiveBench runs a single lane from 09:43:23
  (`plan_lb-20260923-minp0-single.sh`). Reordered ~09:52 at the
  coordinator's call (the paired coding comparison is the priority): bonsai
  coding, then tier max coding (as soon as `GO_TIER_MAX`: n=1 gate, then
  `GATE_OK`, then 21), then bonsai+check coding, then minimal coding, then
  bonsai IF (200). Later arms run in slices, so tier max is slotted in at the
  first slice boundary after GO. Each row's `timing` says
  which concurrency its request started under (`condition.json`
  `timing_epochs`). Correctness is unaffected, but `seconds` is only
  comparable within one label.
- **~07:16 EDT, condition change: min_p 0.05 -> 0.0.** The run was stopped
  and frozen as the min_p 0.05 record, and `lb-20260923-minp0` started
  (`plan_lb-20260923-minp0.sh`). A run dir containing `FROZEN` is never
  re-scored, because the LiveBench answer paths are reused by the next run.

## How errors are counted

| status | meaning | in the score? |
|---|---|---|
| `ok` | answered and judged | yes |
| `budget_event` | `finish_reason=length` with no answer | yes, as 0 (LiveBench does the same) |
| `not_run` | transport failure after 12 driver attempts with 90 s backoff (restarts, resets; and, before 03:59:29, 429s) | no, counted separately |
| `eval_error` | the judge raised | no, counted separately |

The `finish_reason` distribution is in `summary.json` per arm and category.

## Files

- `make_sample.py`: fixed, stratified question order (`results/<run-id>/order_<category>.json`)
- `drive.py`, `run_arm.sh`: run one arm x category through the harness
- `score.py`: official judge, then `rows_<arm>_<category>.jsonl`
- `summarize.py` (+ `test_summarize.py`): `summary.json`, dashboard-ready
- `compare.py`: `comparison.json` / `comparison.md`
- `validate_scorer.py`: judge self-test
- `reference/`: leaderboard snapshots used (fetched 2026-09-23)
- `patches/`: the LiveBench patch and model config
