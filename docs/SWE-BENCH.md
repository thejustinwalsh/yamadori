# SWE-bench: where this model stands against the others

The operator asked for "the exact SWE benchmark all models run when being
compared to each other online". This file covers five things:

- what that benchmark is today, with sources
- how it is set up here, and every deviation from the published setup
- the overnight run and what it showed
- how long a full run would take
- the published scores we compare against, and which of them are like for like

**The result so far (2026-09-23).** The harness runs on the official SWE-bench
Verified Mini with the bash-only leaderboard scaffold (mini-swe-agent 2.1.0,
its unmodified `swebench.yaml`).

- **bonsai (everything off) resolved 2 of 2 graded instances.** The 95% Wilson
  interval is [34.2, 100]%. That is a working pipeline, not a leaderboard
  position.
- **Cost:** about 100 min per instance. A real run needs the GPU to itself: 3.5
  days per arm for the Mini (50) on one card.
- **Not run:** the full-stack `yamadori` arm (section 5 says why).

Details are in section 6.

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
| litellm | 1.102.1 (1.102.0 in the WSL venv) |
| datasets | 5.0.1 |
| Python | 3.13 venv `C:\Users\jwals\swebench-yamadori\venv-win` (what the overnight run uses); 3.11 uv venv `~/swebench-yamadori/.venv` in WSL Ubuntu (the original setup) |
| Docker | 28.1.1 (Docker Desktop 4.41.2, WSL2 backend) |

Both venvs were installed with `pip install "mini-swe-agent==2.1.0"
"swebench==4.1.0"`. `wsl_side.py check` prints the versions and the config
hash. On both, the hash is `4f0492bb…7778c`, the leaderboard yaml.

**Why the overnight run is Windows-native (2026-09-23).** The machine crashed
at 23:42. When it came back, Docker Desktop had two faults:

- **Its WSL integration for Ubuntu had failed.** There is no `docker` and no
  `/var/run/docker.sock` in the distro. Its dialog offers "Restart the WSL
  integration" and is waiting for a human.
- **Its API cache serves stale container reads.** In
  `com.docker.backend.exe.apicache`, `GET /containers/<id>/json` returns 404
  for a running container, and `GET /containers/json` returns `[]`. Every write
  works: create, start, exec, put_archive, stop and remove.

`docker exec` and the SDK's `containers.create()` both inspect first, so both
fail. Restarting Docker Desktop was ruled out (operator constraint: restart
nothing). So `bench/swebench/dockerfix.py` runs both tools natively on
Windows:

- **mini-swe-agent.** Its `docker` environment is swapped for
  `LowLevelDockerEnvironment`, which uses the Engine API and never inspects.
  It keeps everything else the same: the container, `bash -c <command>`, the
  env vars, cwd `/testbed`, stdout and stderr merged, and the 60 s timeout with
  the same `TimeoutExpired` text.
- **The harness.** `containers.create()` and `images.list()` build their
  models from the write responses, not from the stale cache.
- **Line endings.** Windows `Path.write_text` would turn `\n` into `\r\n` in
  `patch.diff` and `eval.sh`, which the harness then copies into the Linux
  container. `write_text` is pinned to `\n` in that process.

**Proof, before scoring anything:**

- `dockerfix.py selftest` passes 12/12 against the real image: quotes,
  backslashes, heredocs, unicode, merged stderr, exit codes, cwd, no CRLF, the
  timeout text and the submit signal.
- The gold patches for `sympy__sympy-17655` and `django__django-11999`,
  graded by the patched harness, come back resolved 2/2.
- A mock-model patch (a README edit), graded in WSL on 2026-09-22, came back
  unresolved.

The clean fix is to click "Restart the WSL integration" in Docker Desktop.
After that, `SWEBENCH_SIDE=wsl` restores the WSL path.

**The model.** litellm's OpenAI-compatible provider, `openai/yamadori`, is
pointed at the proxy.

- **Address.** Natively on Windows this is `http://127.0.0.1:1234/v1`. From WSL
  (NAT networking) it is the Windows host at the default gateway,
  `http://172.29.32.1:1234/v1`. `wsl_side.resolve_api_base` tries loopback
  first, then the gateway.
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

**A known quirk of the leaderboard config, kept on purpose.** The config uses
`interpreter: ["bash", "-c"]`, which is not a login shell, so the image's
`~/.bashrc` (`conda activate testbed`) never runs. Every command runs in the
base conda env, not the task's `testbed` env.

- mini-swe-agent fixed this after the leaderboard runs: v2.4.6 adds
  `BASH_ENV: /root/.bashrc` with that comment.
- The v2 leaderboard entries all ran 2.1.0 without the fix, so every model on
  that board faced the same environment. It stays here for comparability.
- It shows in trajectories. On `sphinx-doc__sphinx-8269` our model wrote stub
  modules (`/tmp/stub/roman.py`) to import sphinx.

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

## 5. The overnight run (2026-09-23): Verified Mini, two arms

**Subset: SWE-bench Verified Mini** (`MariusHobbhahn/swe-bench-verified-mini`,
50 test instances). It was checked here against Verified before use:

- All 50 ids are in `princeton-nlp/SWE-bench_Verified`.
- Every row is identical to Verified on `base_commit`, `patch`,
  `FAIL_TO_PASS` and `PASS_TO_PASS`, so it is graded against Verified itself
  (`arms.EVAL_DATASETS`).

**How it was chosen.** Marius Hobbhahn built it
(https://github.com/mariushobbhahn/SWEBench-verified-mini; MIT in the repo; no
licence field on the HF card; created 2025-01-08):

1. k-means over per-instance pass rates of 16 models' full-Verified runs.
2. A linear program keeps the cluster proportions while minimising Docker
   storage (130 GB → 5 GB).

**It is NOT matched on repository.** It holds django 25 and sphinx 25. Its
difficulty mix is close to Verified's:

| difficulty | Mini | Verified (scaled to 50) |
|---|---|---|
| <15 min | 19 | 19.4 |
| 15 min–1 h | 23 | 26.1 |
| 1–4 h | 7 | 4.2 |
| >4 h | 1 | 0.3 |

So the Mini is slightly harder than Verified, and it covers two repositories
only.

**Order.** The 50 ids, sorted, were shuffled with `random.Random(20260923)`
(`results/mini50-20260923/instances.json` records the order and the seed), so
any prefix is a random sample. The schedule is in `schedule.json`:

1. Arm `bonsai` on instances 1–20.
2. Arm `yamadori` on the same 20.
3. From instance 21 on, the arms alternate per instance (bonsai #21, yamadori
   #21, …) until the run is stopped.

Two agent runs are in flight at once (`--workers 2`). The model is shared on
purpose with two other benchmarks (LiveBench and the domain suite), each at one
request. **Wall times from this run are measured under that sharing; they are
not the stack's unloaded speed.**

The `yamadori` arm's fan-out (N=3) makes more than one upstream generation per
agent step, inside the proxy.

Each instance is graded the moment its patch exists (its own harness run id,
`<run>.<arm>.<instance>`). `results.jsonl` is rewritten after every agent run
and every grade.

```powershell
& $PY bench/swebench/run.py --run-id mini50-20260923 --arms bonsai,yamadori `
    --subset verified-mini --all --seed 20260923 --shared --workers 2 --block 20 `
    --key-file K
```

`--shared` means the runner does not wait for the card and does not own the
gpu lane. It checks that `jobs.paused('gpu')` is set, and sets it as
`swebench overnight` (TTL 3600 s, refreshed every 20 min) only if nobody holds
it. At the end it resumes only its own pause.

Transient failures:

- An agent run that ends on a connection or server error
  (`APIConnectionError`, `InternalServerError`, a 502 from the proxy restart…)
  is re-run once at the end of the queue. It is never scored.
- A context-window or step-limit ending is a real outcome and is kept.

**What was actually run, and why the plan changed (2026-09-23).**

1. **01:26: two workers, bonsai then yamadori.** Both workers started on the
   schedule above.
2. **About 01:30: the workers starved.** The proxy admitted 2 main lanes
   against 4 clients, and mini's log filled with 429s ("all 2 main lanes busy
   after 20s"). The operator raised `YAMADORI_MAIN_LANES` to 4, one per
   llama-server slot.
   - A 429 is admission, not a model result. The request never ran, and its
     retries happen inside one model query, so they add no step.
   - mini's retry ceiling was raised from 10 to 200 attempts
     (`MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT`) so that a queue cannot fail an
     instance.
3. **03:10: the first instance finished.** bonsai took 103 min on
   `django__django-11999` (80 steps). All four llama-server slots were busy
   with three benchmarks, and nvidia-smi showed 99%.
   - At that rate the 20-instance bonsai block would end in the evening.
   - The yamadori block would not start before the presentation.
4. **Operator decision, 03:12: no yamadori arm tonight.** The GPU is needed for
   the domain suite, which is tonight's statistical backbone, and five pairs at
   103 min each would be anecdote. The two bonsai agents already running
   finished; then **one** worker ran bonsai sequentially (#5, #6, …) until
   about 08:00, grading each instance as it finished.
   - The `yamadori` and `yamadori-auto` arms are wired and tested but have
     **no SWE-bench results**.
   - Also untested: whether forced deep thinking makes each agent step
     minutes long. The operator's fallback, if it exceeds 3× bonsai's time per
     step, is to run the stack as `yamadori-auto`.
     `results/<run>/arm_override.json` makes that switch without a restart.

5. **About 04:00: the run starved again.** Even with 4 lanes, the other two
   benchmarks held all four llama-server slots with generations of 18k–48k
   tokens. A lane came free only every several minutes. Operator decision
   (b), 04:05: retry a 429 after 1–2 s with jitter instead of mini's
   exponential backoff (`dockerfix._patch_retry`), so SWE-bench queues as one
   fair client. The two starved in-flight runs, #3 and #4, were discarded
   and restarted.
6. **04:37: stopped by the operator.** From 04:06 to 04:37 bonsai #3 made 1
   action, so no further instance could finish before the presentation.
   **Why no more instances were run: one card shared by three benchmarks.**
   The GPU went to the domain suite. Every graded result was kept; the
   unfinished #3 was discarded. The full timeline is in
   `results/mini50-20260923/run.log`.

**What a real run needs.** It needs the GPU to itself, at about 100 min per
instance:

- **Verified Mini (50):** about 83 h, or **3.5 days per arm on one card**.
- **Verified (500):** about 35 days per arm.

The 100 minutes were measured on a shared card, so they overstate the cost for
a card running only SWE-bench. How much they overstate it has not been
measured. The first instance's median step gap was 20 s; its long gaps were
waits for a lane.

A two-arm comparison doubles these figures. The `yamadori` arm will cost more
per step if forced deep thinking fires on every step; that has not been
measured either.

## 6. Results

Run `mini50-20260923`, arm `bonsai` (everything off). The subset is Verified
Mini in seeded order; grading is by the swebench 4.1.0 harness against
`princeton-nlp/SWE-bench_Verified`. Regenerate this table with
`parse_results.py bench/swebench/results/mini50-20260923 --markdown`.

| arm | instance | verdict | exit | steps | wall s | prompt tok (sum over steps) | completion tok | format err |
|---|---|---|---|---|---|---|---|---|
| bonsai | django__django-11999 | **resolved** | Submitted | 80 | 6,174 | 1,097,012 | 40,103 | 0 |
| bonsai | sphinx-doc__sphinx-8269 | **resolved** | Submitted | 66 | 6,264 | 1,414,691 | 27,695 | 0 |

| arm | resolved / scored | % | 95% Wilson |
|---|---|---|---|
| bonsai | 2 / 2 | 100 | [34.2, 100] |

- **Two instances are not a rate.** The interval is [34.2, 100]%, so all it
  excludes is a model below about a third. **It says nothing about where
  bonsai sits among the leaderboard models** (56–77% in table A below).
- **Harness checks.** The patched Windows harness was verified in both
  directions before these verdicts:
  - Gold patches: 2/2 resolved.
  - A comment-only patch on django-11999: unresolved.
- **Plumbing.** Neither trajectory had a format error or a `length` event.
  Every model turn ended in a parsed `bash` tool call, so the model's thinking
  output and the tool-call format work with mini-swe-agent as they are.

**The trajectory to show: `django__django-11999`.** The task: Django 2.2 stopped
letting a model override `get_FOO_display()`.

1. **Steps 1–16 (5 min).** The agent explores the repo and the git history.
2. **Steps 17–40.** It builds a standalone reproduction settings module and
   confirms the bug.
3. **Step 41 (48 min).** It verifies a fix, then runs `tests/runtests.py` for
   `model_regress`, `admin`, `forms_tests` and `admin_widgets`.
4. **Steps 65–73.** It reads the metaclass to explain why 2.1 worked.
5. **Step 80 (103 min).** It submits a 5-line patch to
   `Field.contribute_to_class`: only install the default `get_%s_display` if
   the class does not already define one. That is the upstream fix, and the
   harness confirms it: FAIL_TO_PASS passes, PASS_TO_PASS holds.

About 30 of the 103 minutes were spent waiting for a lane on the shared card,
not generating.

**Width of a 50-instance interval.** Even the full Mini is not the 500:

| resolved of 50 | % | 95% Wilson | width |
|---|---|---|---|
| 25 | 50 | [36.6, 63.4] | ±13.4 pts |
| 30 | 60 | [46.2, 72.4] | ±13 pts |
| 35 | 70 | [56.2, 80.9] | ±12 pts |

The same rates on 500 instances give about ±4 points. So a Mini score can
place a model in a band of roughly 25 points. That separates GPT-5 mini (56.2)
from Claude Opus 4.5 (76.8), but not neighbours 2–3 points apart. The Mini is
also django and sphinx only, and slightly harder than Verified (section 5), so
it is an estimate of Verified, not a substitute for it.

**Not run tonight.** The `yamadori` arm (the full stack) and `yamadori-auto`.
Both are wired and unit-tested but have no SWE-bench numbers. Section 5 gives
the reason.

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
