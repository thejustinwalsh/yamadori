# SWE-bench: where this model stands against the others

The operator asked for "the exact SWE benchmark all models run when being
compared to each other online". This file covers five things:

- what that benchmark is today, with sources
- how it is set up here, and every deviation from the published setup
- the pilot and what it showed
- how long a full run would take
- the published scores we compare against, and which of them are like for like

Runner and results: [bench/swebench/](../bench/swebench/README.md).

## 1. The standard, as of 2026-09-22

**Dataset.** SWE-bench Verified is 500 human-validated Python tasks from 12
repositories: django 231, sympy 75, sphinx 44, matplotlib 34, scikit-learn 32,
astropy 22, xarray 22, pytest 19, pylint 10, requests 8, seaborn 2 and flask 1.

- `princeton-nlp/SWE-bench_Verified`, test split, 500 rows. The id is still
  live, and mini-swe-agent's `--subset verified` maps to it.
- `SWE-bench/SWE-bench_Verified` is the maintained copy (last modified
  2026-08-16). It has the same 500 instance ids and adds image and eval-script
  columns.

**Scaffold: mini-swe-agent, "bash only".** The swebench.com leaderboard ranks
models under one fixed agent. That agent is mini-swe-agent: the model gets one
tool (`bash`) in the task's Docker image.

- That board was its own tab until 2026-09-01. Commit `193160a` "Drop bash-only
  board; show mini icon alongside model logo" merged it into the main Verified
  list, where these entries now carry a mini icon.
- Source: `data/leaderboards.json` in
  https://github.com/SWE-bench/swe-bench.github.io (the site renders it with
  JS). Commit history: `api.github.com/repos/SWE-bench/swe-bench.github.io/commits`.

**The newest leaderboard entries are dated 2026-02-17 to 2026-02-26.** Every one
is tagged `mini-swe-agent_version: 2.0.0`. The trajectories they published
(`s3://swe-bench-submissions/bash-only/<entry>/trajs/`, readable over HTTPS)
all say `mini_version: 2.1.0`. I checked all 13 v2 entries on
`django__django-15277`.

| setting | value | source |
|---|---|---|
| config | `src/minisweagent/config/benchmarks/swebench.yaml`, byte-identical at tags v2.0.0 and v2.1.0 (sha256 `4f0492bb…7778c`) | github.com/SWE-agent/mini-swe-agent |
| actions | native tool calling, one tool `bash {"command"}`, `parallel_tool_calls: true` | same yaml, `models/utils/actions_toolcall.py` |
| step limit | 250 model calls | yaml `agent.step_limit`; every v2 trajectory's `info.config` |
| cost limit | $3.00 per instance | yaml `agent.cost_limit`; every v2 trajectory |
| command timeout | 60 s, cwd `/testbed` | yaml `environment` |
| temperature | **the yaml says 0.0, but all 13 v2 submissions ran with `temperature: None`** (the provider default) plus a reasoning effort, mostly `high` | `info.config.model.model_kwargs` in each trajectory |
| prompt | the yaml's `system_template` and `instance_template`, unchanged in every submission | diff of trajectory config against the yaml |
| attempts | 1 (pass@1) | leaderboard tag `System: Attempts - 1` |

**Grading.** Each patch is applied in the instance's Docker image and scored
with the SWE-bench harness. An instance is resolved only if every FAIL_TO_PASS
test now passes and every PASS_TO_PASS test still passes.

- The mini-swe-agent docs point to either `sb-cli` (cloud) or the local
  harness: `python -m swebench.harness.run_evaluation`.
- We use the local harness at **swebench 4.1.0** (2025-09-11), the release in
  force when the v2 entries were graded. 5.0.x shipped on 2026-08-17.

**SWE-bench Multilingual** is 300 tasks in 9 languages
(`SWE-bench/SWE-bench_Multilingual`):

- JS/TS 43, including preact, axios, babel, docusaurus, vuejs/core, three.js
  and immutable-js
- Rust 43, including tokio, bat, ruff, axum, nushell, coreutils and ripgrep
- plus C, C++, Go, Java, PHP and Ruby

It is our target domain, not Verified's. The leaderboard has a Multilingual
board with 13 mini-swe-agent entries (v2.0.0a0/2.0.0, 2026-02-13 to 02-20).
`arms.DATASETS["multilingual"]` is wired, but it has not been run here (see
section 6).

**The vendors have partly moved on.**

- OpenAI stopped reporting Verified on 2026-02-23 ("why we no longer evaluate
  SWE-bench Verified": flawed tests, contamination) and recommends SWE-bench
  Pro.
- Anthropic's newest system cards (Opus 5.5, Fable 5.1) report Pro and
  Multilingual but not Verified.
- Qwen's own card for Qwen3.8-27B reports SWE-bench Pro, not Verified.

Verified with mini-swe-agent is still the only board where the same scaffold
has been run across Anthropic, OpenAI, Google and the open models. That is why
it is the one used here.

## 2. The model's lineage

| layer | id | source |
|---|---|---|
| base | `Qwen/Qwen3.8-27B`: 27.8B, hybrid Gated DeltaNet + attention, Apache-2.0, created 2026-08-05 | huggingface.co/Qwen/Qwen3.8-27B |
| ternary | `prism-ml/Ternary-Bonsai-2-27B-gguf` (`base_model: Qwen/Qwen3.8-27B`, "architecture unchanged"), created 2026-09-16 | its HF card |
| served | `BoldingBuilds/Ternary-Bonsai-2-27B-Abliterated-PTQ1_0-GGUF` (`base_model: prism-ml/…`), 1.75 bpw, created 2026-09-18 | `config.template.yaml`, its HF card |

No layer of this lineage has a SWE-bench Verified number.

- Qwen3.8-27B's card reports SWE-bench Pro 61.7, Terminal Bench 2.1 73.0 and
  DeepSWE 42.2, all with a Claude Code harness.
- The Bonsai cards report neither SWE-bench nor Terminal-Bench.

## 3. Setup here

**Versions.**

| component | version |
|---|---|
| mini-swe-agent | 2.1.0 |
| swebench | 4.1.0 |
| litellm | 1.102.0 |
| datasets | 5.0.1 |
| Python | 3.11, uv venv `~/swebench-yamadori/.venv` in WSL Ubuntu |
| Docker | 28.1.1 (Docker Desktop, WSL2 backend) |

Installed with `uv pip install "mini-swe-agent==2.1.0" "swebench==4.1.0"`.
`wsl_side.py check` prints the versions and the config hash.

**The model.** litellm's OpenAI-compatible provider, `openai/yamadori`, is
pointed at the proxy.

- **Address.** From WSL (NAT networking) that is the Windows host at the
  default gateway, `http://172.29.32.1:1234/v1`. `127.0.0.1` does not reach it.
  `wsl_side.resolve_api_base` tries localhost first and then the gateway.
- **Key.** Read from the key file into `OPENAI_API_KEY` in the mini-extra
  child's environment only. It never appears on a command line, in a config
  file, in a trajectory or in a log. After the pilot, a grep of every result
  file for the key found nothing.

**Deviations from the leaderboard config.** Every one is set in a single
overlay (`config_overlay.yaml`, merged after the unmodified package yaml) and
recorded in each arm's `run.json`.

| key | value | why |
|---|---|---|
| `model.model_name` | `openai/yamadori` | our model |
| `model.model_kwargs.api_base` | `http://<windows host>:1234/v1` | our endpoint |
| `model.model_kwargs.extra_headers` | `X-Yamadori-Features: <arm>` | selects the arm |
| `model.model_kwargs.extra_body` | `{"reasoning_effort": "max"}` | So the tier *allows* everything and the header alone decides. This is `bench/domain/run.py`'s convention. The header pins `effort: medium`. |
| `model.model_kwargs.timeout` | 3600 | litellm's 600 s default is shorter than one deep-thinking turn here, and a timed-out call is retried, which doubles GPU load |
| `model.model_kwargs.temperature` | `null` (was `0.0`) | Matches what all 13 v2 submissions actually ran. The server's own sampling then applies (`--temp 1.0 --top-p 0.95 --top-k 20`, Qwen's recommendation). |
| `model.litellm_model_registry` | cost 0 per token | a free local model |
| `model.cost_tracking` | `ignore_errors` | mini raises on a cost of 0. **Consequence: the $3 cost limit never binds for us; only the 250-step limit does.** |

**Unchanged:** `step_limit: 250`, `cost_limit: 3`, the system prompt, the
instance prompt, the observation and format-error templates, the 60 s command
timeout, and `parallel_tool_calls: true`.

**Step limit vs. hop caps.** The operator removed hop caps inside our own stack
(`proxy.complete`: "THERE IS NO HOP BUDGET"). mini-swe-agent's 250-step limit is
a different thing. It is part of the benchmark's definition, applied to every
model on the board, and **it stays** so the numbers are comparable.

- A proxy-internal tool round (retrieval in the `yamadori` arm) happens inside
  one mini-swe-agent step and does not count against the 250.
- It does count in `proxy_hops` in the results.

**Effort.** The leaderboard's thinking models mostly ran at effort `high`. Our
chat template accepts `low`, `medium` and `xhigh` (AGENTS.md). Every arm pins
`medium`, the operator's choice, so effort is held fixed across arms.

**The arms** (`bench/swebench/arms.py`, header = `X-Yamadori-Features`):

| arm | header | what it is |
|---|---|---|
| `bonsai` | `{"retrieval": false, "hints": false, "investigate": false, "fanout": 1, "effort": "medium"}` | the bare model (bench/domain A0) |
| `yamadori-auto` | `{"effort": "medium"}` | everything allowed; the selection engine decides (A5) |
| `yamadori` | `{"retrieval": true, "hints": true, "investigate": true, "fanout": 3, "effort": "medium"}` | the full stack, forced on (A6). **The headline row.** |

**What `yamadori` does and does not get.** It is run the way Hermes would be.

- The task repository is **not** staged, indexed or bound on the host.
- The model sees the repository only through its own bash calls in the
  container. `proxy.resolve_repo` returns None by contract.
- Nothing in the prompt names a Windows path; `/testbed` does not exist on
  Windows.
- The server adds only what it holds: package indexes (matched from imports
  the model has seen, `proxy.session_context`), hints, deep thinking and
  fan-out.
- Verified is Python, and the held package indexes are mostly JS/TS/Rust, so
  retrieval is expected to have little to serve. Section 4 reports what fired.

**GPU discipline.** `run.py` does four things:

1. It waits until the worker's gpu lane is not paused by someone else and no
   other known benchmark process is alive (`OTHER_CONSUMERS`: bench/domain,
   mtpbench, livecodebench, queue_runner and others).
2. It pauses the gpu lane as `bench/swebench/run.py` with a 30-minute TTL,
   refreshed every 10 minutes.
3. It checks again before every instance, and stops if another consumer has
   appeared.
4. It resumes the lane in a `finally`. It never removes a pause that someone
   else holds.

Grading is CPU and Docker only, and runs after the lane is released.

## 4. Commands

Run these from the repo root on Windows with the stack interpreter. `K` is the
path of the key file.

```powershell
$PY = "C:\Users\jwals\textgen\installer_files\env\python.exe"
# one arm, one instance
& $PY bench/swebench/run.py --run-id smoke --arms bonsai --instances django__django-15277 --key-file K
# the pilot: 5 seeded instances x 3 arms
& $PY bench/swebench/run.py --run-id pilot-20260922 --arms bonsai,yamadori-auto,yamadori --pilot --key-file K
# Verified Mini (50), one arm per run id
& $PY bench/swebench/run.py --run-id mini50-bonsai --arms bonsai --subset verified-mini --all --key-file K
# full Verified (500)
& $PY bench/swebench/run.py --run-id verified500-yamadori --arms yamadori --all --key-file K
# regrade / re-parse without the model
& $PY bench/swebench/run.py --run-id pilot-20260922 --arms bonsai --evaluate-only
& $PY bench/swebench/parse_results.py bench/swebench/results/pilot-20260922 --print
```

Or queue it behind other GPU work:

```powershell
& $PY bench/queue_runner.py add swebench/run.py --args "--run-id … --key-file K" --needs proxy
```

**Results.** `bench/swebench/results/<run-id>/<arm>/` holds `preds.json`,
trajectories (every raw response, with `usage` and `x_yamadori`),
`timings.jsonl`, `run.json`, the harness report and `logs/`.
`results/<run-id>/results.jsonl` has one row per (instance, arm):

- `resolved`, `eval_status` (resolved / unresolved / empty_patch / agent_error
  / eval_error / not_evaluated) and `exit_status`
- `steps`, `seconds`, `prompt_tokens`, `completion_tokens` and `proxy_hops`
- `format_errors`, `finish_reasons`, `length_events` and `proxy_error_lines`
- an `x_yamadori` summary: hint turns, deep-thinking runs and injections,
  fan-out turns, internal tool turns, and the selection reasons

`bench/swebench/test_parse_results.py` proves the parser keeps these apart. It
runs in `scripts/run_tests.py`.

PILOT_AND_PROJECTION_PLACEHOLDER

## 7. Published scores

Every leaderboard number below is SWE-bench Verified, % resolved of 500, from
`data/leaderboards.json` at
https://github.com/SWE-bench/swe-bench.github.io (fetched 2026-09-22). Each
entry's trajectories are at
`https://swe-bench-submissions.s3.amazonaws.com/bash-only/<folder>/trajs/`.

**A. Same scaffold: mini-swe-agent v2 (runtime 2.1.0), bash only, pass@1. These
are like for like with a run here.**

| model | open weights | Verified | effort | date |
|---|---|---|---|---|
| Claude Opus 4.5 | no | 76.8 | high | 2026-02-17 |
| Gemini 3 Flash | no | 75.8 | high | 2026-02-17 |
| MiniMax M2.5 | yes | 75.8 | high | 2026-02-17 |
| Claude Opus 4.6 | no | 75.6 | default (adaptive) | 2026-02-17 |
| GLM-5 | yes | 72.8 | high | 2026-02-17 |
| GPT-5.2 | no | 72.8 | high | 2026-02-17 |
| GPT-5.2-Codex | no | 72.8 | high | 2026-02-19 |
| Claude Sonnet 4.5 | no | 71.4 | high | 2026-02-17 |
| Kimi K2.5 | yes | 70.8 | high | 2026-02-17 |
| DeepSeek V3.2 | yes | 70.0 | high | 2026-02-17 |
| Gemini 3 Pro | no | 69.6 | high | 2026-02-26 |
| Claude Haiku 4.5 | no | 66.6 | high | 2026-02-17 |
| GPT-5 mini | no | 56.2 | medium | 2026-02-17 |

**B. Same agent family, older version (mini v1.x, text-based actions rather
than tool calls). Close, but not identical.** These are the only small open
models on the board.

| model | size | Verified | mini | date |
|---|---|---|---|---|
| Devstral Small 2 (2512) | 24B dense | 56.4 | 1.17.2 | 2025-12-09 |
| Qwen3-Coder 480B-A35B | 480B MoE | 55.4 | 1.0.0 | 2025-08-02 |
| gpt-oss-120b | 117B MoE | 26.0 | 1.7.0 | 2025-08-07 |
| Qwen2.5-Coder 32B | 32B dense | 9.0 | 1.0.0 | 2025-08-03 |

The board has no Qwen3.x entry of any size.

**C. SWE-bench Multilingual, mini-swe-agent v2.0.0a0/2.0.0, bash only (like for
like with a Multilingual run here).**

| model | score |
|---|---|
| Gemini 3 Flash | 72.7 |
| Claude Opus 4.6 | 72.0 |
| Claude Opus 4.5 | 70.7 |
| GLM-5 | 69.7 |
| Gemini 3 Pro | 68.7 |
| MiniMax M2.5 | 68.3 |
| Kimi K2.5 | 67.3 |
| Claude Sonnet 4.5 | 67.0 |
| GPT-5.2 (high) | 66.7 |
| GPT-5.2-Codex | 66.3 |
| Claude Haiku 4.5 | 64.7 |
| DeepSeek V3.2 | 59.0 |
| GPT-5 mini | 39.7 |

All entries are dated 2026-02-13 to 02-20.

**D. Vendor-reported, each with its own scaffold. NOT comparable 1:1.** These
come from each vendor's own harness, which often adds file-edit tools and
averages over many trials. They are typically 5 to 12 points above the same
model under mini-swe-agent (compare Opus 4.5, Haiku 4.5, GPT-5.2 and Devstral
Small 2 across tables A, B and D).

| model | Verified | scaffold / notes | source | date |
|---|---|---|---|---|
| Claude Opus 5 | 96.0 | average of 5 trials, adaptive thinking at max effort; scaffold not described | Opus 5 system card §8.2, www-cdn.anthropic.com/c5fbac3f…/Claude Opus 5 System Card.pdf | 2026-07-24 |
| Claude Sonnet 5 | 85.2 | average of 5 trials, max effort, 1M context | Sonnet 5 system card §8.2 | 2026-06-30 |
| Claude Opus 5.5 | not reported | Pro 89.9 and Multilingual 93.9 instead | Opus 5.5 system card §8.2 | 2026-09-22 |
| Claude Haiku 4.5 | 73.3 | bash plus a string-replace edit tool, 50 trials, 128K thinking | anthropic.com/news/claude-haiku-4-5 | 2025-10-15 |
| GPT-5.2 Thinking | 80.0 | own harness | openai.com/index/introducing-gpt-5-2/ | 2025-12-11 |
| GPT-5.1-Codex-Max (xhigh) | 77.9 | own harness, n=500 | openai.com/index/gpt-5-1-codex-max/ | 2025-11-19 |
| GPT-5.5 / 5.6 / 6 | not reported | OpenAI stopped reporting Verified | openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/ | 2026-02-23 |
| Qwen3.8-27B (our base) | not reported | card: SWE-bench Pro 61.7 (Claude Code harness, temp 1.0) | huggingface.co/Qwen/Qwen3.8-27B | 2026-08-05 |
| Qwen3.6-27B | 77.2 | Qwen's internal scaffold (bash + file edit), temp 1.0, 200K context | huggingface.co/Qwen/Qwen3.6-27B | 2026-04-21 |
| Qwen3.6-35B-A3B | 73.4 | same | huggingface.co/Qwen/Qwen3.6-35B-A3B | 2026-04-15 |
| Qwen3.5-27B | 72.4 (card) / 75.0 (as re-scored in the Qwen3.6 card) | not stated | huggingface.co/Qwen/Qwen3.5-27B | 2026-02-24 |
| Devstral Small 2 (24B) | 68.0 | not stated; the same model scored 56.4 on mini-swe-agent (table B) | huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512 | 2025-11-28 |
| GLM-4.7-Flash (31B MoE) | 59.2 | not stated | huggingface.co/zai-org/GLM-4.7-Flash | 2026-01-19 |
| gpt-oss-20b (high) | 60.7 | internal harness, n=477 | arxiv.org/pdf/2508.10925 table 3 | 2025-08-05 |
| Qwen3-Coder-30B-A3B | 51.6 | OpenHands, 500 turns | huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct | 2025-07-31 |

**What a comparison can claim.**

- A Verified run here, with the arm named, sits in table A.
- The same model's distance to its own base is not measurable on Verified:
  Qwen never published a Verified number for Qwen3.8-27B. The nearest same-size
  Qwen number is Qwen3.6-27B's 77.2, and it comes from Qwen's own scaffold, so
  it belongs to table D.
- A fair base-model comparison would mean running Qwen3.8-27B itself (BF16 or
  FP8) through this same runner. That is possible: the runner only needs an
  OpenAI endpoint. But the full-precision model does not fit on this card.
