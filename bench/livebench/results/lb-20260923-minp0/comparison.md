# LiveBench comparison -- run lb-20260923-minp0

**Read this first.** Our numbers are a local, ternary-quantised (1.75 bpw) derivative of Qwen3.8-27B served on one consumer GPU, plus our tool stack in the `yamadori` arm. LiveBench's leaderboard numbers come from the providers' own APIs at full precision. The question set is the latest PUBLIC release, 2024-11-25; the current leaderboard (2026-06-25) uses newer questions that LiveBench has not published, so tier C below is context, not a comparison. The 2024-11-25 questions have been public since April 2025 and may be in any 2026 model's training data, ours included: treat every number here as possibly contaminated.

## A. Same questions (apples to apples)

Reference models re-scored from LiveBench's published per-question judgments on exactly the questions our arm scored; same aggregation (task mean -> category mean of tasks -> overall mean of categories). 95% interval: bootstrap over questions, ours only (reference rows are fixed published judgments on the same sample).

### arm `bonsai` (n per category: coding 21)

| model | overall | Coding |
|---|---|---|
| **ours: bonsai** | **70.9 [51.4, 90.0]** | 70.9 [51.4, 90.0] |
| Gemini 2.5 Pro exp | 90.0 | 90.0 |
| Claude 3.7 Sonnet thinking (64k) | 85.9 | 85.9 |
| o3-mini high | 81.4 | 81.4 |
| GPT-4.5 preview | 81.4 | 81.4 |
| o1 high | 77.3 | 77.3 |
| Claude 3.7 Sonnet | 76.8 | 76.8 |
| DeepSeek R1 | 75.5 | 75.5 |
| QwQ-32B | 75.5 | 75.5 |
| Qwen2.5-Max | 71.8 | 71.8 |
| Claude 3.5 Sonnet (2024-10-22) | 67.3 | 67.3 |
| Claude 3.5 Haiku | 67.3 | 67.3 |
| GPT-4o (2024-11-20) | 67.3 | 67.3 |
| GPT-4o mini | 67.3 | 67.3 |
| Qwen2.5-72B Instruct | 67.3 | 67.3 |
| Qwen2.5-Coder-32B Instruct | 62.3 | 62.3 |
| Mistral Small 3.1 (2503) | 47.7 | 47.7 |
| Gemma 3 27B | 47.3 | 47.3 |
| R1-Distill-Qwen-32B | 31.8 | 31.8 |

`--` = LiveBench publishes no per-question judgments for that category on this release (the HF set covers coding, IF/paraphrase and language only); use table B for it.


### arm `bonsai+check` (n per category: coding 2)

| model | overall | Coding |
|---|---|---|
| **ours: bonsai+check** | **100.0 [100.0, 100.0]** | 100.0 [100.0, 100.0] |
| Claude 3.7 Sonnet thinking (64k) | 100.0 | 100.0 |
| Claude 3.5 Sonnet (2024-10-22) | 100.0 | 100.0 |
| Claude 3.5 Haiku | 100.0 | 100.0 |
| o3-mini high | 100.0 | 100.0 |
| o1 high | 100.0 | 100.0 |
| GPT-4.5 preview | 100.0 | 100.0 |
| GPT-4o (2024-11-20) | 100.0 | 100.0 |
| DeepSeek R1 | 100.0 | 100.0 |
| QwQ-32B | 100.0 | 100.0 |
| Qwen2.5-Max | 100.0 | 100.0 |
| Qwen2.5-72B Instruct | 100.0 | 100.0 |
| Qwen2.5-Coder-32B Instruct | 100.0 | 100.0 |
| Claude 3.7 Sonnet | 50.0 | 50.0 |
| GPT-4o mini | 50.0 | 50.0 |
| Gemini 2.5 Pro exp | 50.0 | 50.0 |
| R1-Distill-Qwen-32B | 50.0 | 50.0 |
| Gemma 3 27B | 50.0 | 50.0 |
| Mistral Small 3.1 (2503) | 50.0 | 50.0 |

`--` = LiveBench publishes no per-question judgments for that category on this release (the HF set covers coding, IF/paraphrase and language only); use table B for it.


### arm `yamadori` (n per category: coding 2)

| model | overall | Coding |
|---|---|---|
| **ours: yamadori** | **100.0 [100.0, 100.0]** | 100.0 [100.0, 100.0] |
| Claude 3.7 Sonnet thinking (64k) | 100.0 | 100.0 |
| Claude 3.7 Sonnet | 100.0 | 100.0 |
| o3-mini high | 100.0 | 100.0 |
| o1 high | 100.0 | 100.0 |
| Gemini 2.5 Pro exp | 100.0 | 100.0 |
| DeepSeek R1 | 100.0 | 100.0 |
| QwQ-32B | 100.0 | 100.0 |
| Qwen2.5-Max | 100.0 | 100.0 |
| Claude 3.5 Sonnet (2024-10-22) | 50.0 | 50.0 |
| Claude 3.5 Haiku | 50.0 | 50.0 |
| GPT-4.5 preview | 50.0 | 50.0 |
| GPT-4o (2024-11-20) | 50.0 | 50.0 |
| GPT-4o mini | 50.0 | 50.0 |
| Qwen2.5-72B Instruct | 50.0 | 50.0 |
| Qwen2.5-Coder-32B Instruct | 50.0 | 50.0 |
| R1-Distill-Qwen-32B | 50.0 | 50.0 |
| Gemma 3 27B | 50.0 | 50.0 |
| Mistral Small 3.1 (2503) | 50.0 | 50.0 |

`--` = LiveBench publishes no per-question judgments for that category on this release (the HF set covers coding, IF/paraphrase and language only); use table B for it.


### arm `yamadori-xhigh` (n per category: coding 21)

| model | overall | Coding |
|---|---|---|
| **ours: yamadori-xhigh** | **72.3 [54.1, 90.5]** | 72.3 [54.1, 90.5] |
| Gemini 2.5 Pro exp | 90.0 | 90.0 |
| Claude 3.7 Sonnet thinking (64k) | 85.9 | 85.9 |
| o3-mini high | 81.4 | 81.4 |
| GPT-4.5 preview | 81.4 | 81.4 |
| o1 high | 77.3 | 77.3 |
| Claude 3.7 Sonnet | 76.8 | 76.8 |
| DeepSeek R1 | 75.5 | 75.5 |
| QwQ-32B | 75.5 | 75.5 |
| Qwen2.5-Max | 71.8 | 71.8 |
| Claude 3.5 Sonnet (2024-10-22) | 67.3 | 67.3 |
| Claude 3.5 Haiku | 67.3 | 67.3 |
| GPT-4o (2024-11-20) | 67.3 | 67.3 |
| GPT-4o mini | 67.3 | 67.3 |
| Qwen2.5-72B Instruct | 67.3 | 67.3 |
| Qwen2.5-Coder-32B Instruct | 62.3 | 62.3 |
| Mistral Small 3.1 (2503) | 47.7 | 47.7 |
| Gemma 3 27B | 47.3 | 47.3 |
| R1-Distill-Qwen-32B | 31.8 | 31.8 |

`--` = LiveBench publishes no per-question judgments for that category on this release (the HF set covers coding, IF/paraphrase and language only); use table B for it.


### Arms

- `bonsai+check`: bonsai+check
- `bonsai`: bare @ medium (all augmentation forced off, effort medium)
- `yamadori-xhigh`: tier xhigh (reasoning_effort xhigh -> MEDIUM upstream; every augmentation allowed: retrieval, hints, fan-out 3, deep thinking, check_code, repair, 10-turn tool cap) -- the all-on arm, effort-matched to bonsai
- `yamadori`: tier max (reasoning_effort max -> xhigh upstream; selection decides fan-out/deep thinking)

Tier `xhigh` sends MEDIUM effort upstream, so `yamadori-xhigh` vs `bonsai` (bare @ medium) is effort-matched: all augmentation allowed vs all forced off. Tier `max` sends xhigh effort, so `yamadori` (tier max, a partial arm dropped by the operator) vs `bonsai` mixes effort and tools.

### Mechanism health per arm

Counts over each arm's answers, read from `x_yamadori` on every response (bench/livebench/mechanisms.py). allowed = the tier or header permitted it; decided = selection chose it for that request; ran = it executed; produced = it returned data. A broken mechanism is a stack_error: that answer is not scored and is re-asked. Retrieval is read from `x_yamadori.tools`; for answers from before the proxy emitted it only `hops` is known, those count under `answers_without_tool_record`, and `produced` is `--` when no answer had the record.

| arm | answers | stack errors | mechanism | allowed | decided | ran | produced | detail |
|---|---|---|---|---|---|---|---|---|
| bonsai | 21 | 0 | retrieval | 0 | -- | 0 | 0 | calls_total 0, nonempty_total 0 |
|  |  |  | hints | 0 | 0 | 0 | 0 | injected_total 0 |
|  |  |  | deep_thinking | 0 | 0 | 0 | 0 |  |
|  |  |  | fanout | 0 | 0 | 0 | 0 | winner_replaced 0 |
|  |  |  | check_code | 0 | -- | 0 | 0 | calls_total 0 |
|  |  |  | repair | 0 | -- | 0 | 0 |  |
| bonsai+check | 2 | 0 | retrieval | 0 | -- | 0 | 0 | calls_total 0, nonempty_total 0 |
|  |  |  | hints | 0 | 0 | 0 | 0 | injected_total 0 |
|  |  |  | deep_thinking | 0 | 0 | 0 | 0 |  |
|  |  |  | fanout | 0 | 0 | 0 | 0 | winner_replaced 0 |
|  |  |  | check_code | 2 | -- | 0 | 0 | calls_total 0 |
|  |  |  | repair | 2 | -- | 0 | 0 | stopped:clean 2 |
| yamadori | 2 | 0 | retrieval | 0 | -- | 0 | 0 | calls_total 0, nonempty_total 0 |
|  |  |  | hints | 2 | 2 | 2 | 0 | injected_total 0 |
|  |  |  | deep_thinking | 2 | 0 | 0 | 0 |  |
|  |  |  | fanout | 2 | 2 | 2 | 2 | winner_replaced 1, method:code_medoid 2, mode:sequential 2, steps:3 2 |
|  |  |  | check_code | 2 | -- | 0 | 0 | calls_total 0 |
|  |  |  | repair | 2 | -- | 0 | 0 | stopped:clean 2 |
| yamadori-xhigh | 21 | 0 | retrieval | 0 | -- | 4 | 4 | calls_total 6, nonempty_total 6 |
|  |  |  | hints | 21 | 21 | 21 | 0 | injected_total 0 |
|  |  |  | deep_thinking | 21 | 0 | 0 | 0 |  |
|  |  |  | fanout | 21 | 21 | 21 | 21 | winner_replaced 11, method:code_medoid 12, mode:sequential 21, steps:3 16, method:fallback 4, method:code_grade 5, steps:2 5 |
|  |  |  | check_code | 21 | -- | 4 | 4 | calls_total 6 |
|  |  |  | repair | 21 | -- | 3 | 3 | stopped:clean 20, stopped:repeated_answer 1 |

### Paired difference, bonsai+check minus bonsai (same questions)

| category | n pairs | bonsai | bonsai+check | diff [95% CI] | exact McNemar p |
|---|---|---|---|---|---|
| Coding | 2 | 100.0 | 100.0 | +0.0 [+0.0, +0.0] | -- |
| overall | 2 | | | +0.0 [+0.0, +0.0] | |

### Paired difference, yamadori-xhigh minus bonsai (same questions)

| category | n pairs | bonsai | yamadori-xhigh | diff [95% CI] | exact McNemar p |
|---|---|---|---|---|---|
| Coding | 21 | 70.9 | 72.3 | +1.4 [-17.7, +20.5] | 1.000 |
| overall | 21 | | | +1.4 [-17.7, +20.5] | |

### Paired difference, yamadori minus bonsai (same questions)

| category | n pairs | bonsai | yamadori | diff [95% CI] | exact McNemar p |
|---|---|---|---|---|---|
| Coding | 2 | 50.0 | 100.0 | +50.0 [+0.0, +100.0] | 1.000 |
| overall | 2 | | | +50.0 [+0.0, +100.0] | |

### Reproduction check

Each reference model recomputed on the FULL release from per-question judgments, against the published table_2024_11_25.csv. Max absolute per-task difference (points):

| model | n | max abs task diff |
|---|---|---|
| Claude 3.7 Sonnet thinking (64k) | 318 | 0.0 |
| Claude 3.7 Sonnet | 318 | 0.0 |
| Claude 3.5 Sonnet (2024-10-22) | 318 | 0.0 |
| Claude 3.5 Haiku | 318 | 0.0 |
| o3-mini high | 318 | 1.282 |
| o1 high | 318 | 0.0 |
| GPT-4.5 preview | 318 | 0.0 |
| GPT-4o (2024-11-20) | 318 | 0.0 |
| GPT-4o mini | 318 | 0.0 |
| Gemini 2.5 Pro exp | 318 | 0.0 |
| DeepSeek R1 | 318 | 3.167 |
| QwQ-32B | 318 | 4.333 |
| Qwen2.5-Max | 318 | 0.0 |
| Qwen2.5-72B Instruct | 318 | 0.0 |
| Qwen2.5-Coder-32B Instruct | 318 | 0.0 |
| R1-Distill-Qwen-32B | 318 | 17.4 |
| Gemma 3 27B | 318 | 0.0 |
| Mistral Small 3.1 (2503) | 318 | 0.0 |

## B. Published 2024-11-25 leaderboard (full release, all questions)

Same release and question pool; ours is a stratified random sample of it (95% CI from the bootstrap), theirs is every question. `mean of ours` averages only the categories we ran (Coding) so it lines up with our overall.

| model | mean of ours | global (6 cat.) | Reasoning | Coding | IF | Mathematics | Data Analysis | Language |
|---|---|---|---|---|---|---|---|---|
| **ours: bonsai** | **70.9 [51.4, 90.0]** | -- | -- | 70.9 [51.4, 90.0] | -- | -- | -- | -- |
| **ours: bonsai+check** | **100.0 [100.0, 100.0]** | -- | -- | 100.0 [100.0, 100.0] | -- | -- | -- | -- |
| **ours: yamadori** | **100.0 [100.0, 100.0]** | -- | -- | 100.0 [100.0, 100.0] | -- | -- | -- | -- |
| **ours: yamadori-xhigh** | **72.3 [54.1, 90.5]** | -- | -- | 72.3 [54.1, 90.5] | -- | -- | -- | -- |
| Gemini 2.5 Pro exp | 85.9 | 82.4 | 89.8 | 85.9 | 80.6 | 90.2 | 79.9 | 67.8 |
| Claude 3.7 Sonnet thinking (64k) | 74.5 | 76.1 | 87.8 | 74.5 | 81.3 | 79.0 | 74.1 | 59.9 |
| o3-mini high | 82.7 | 75.9 | 89.6 | 82.7 | 84.4 | 77.3 | 70.6 | 50.7 |
| o1 high | 69.7 | 75.7 | 91.6 | 69.7 | 81.5 | 80.3 | 65.5 | 65.4 |
| QwQ-32B | 72.2 | 72.0 | 83.5 | 72.2 | 81.8 | 77.8 | 65.0 | 51.4 |
| DeepSeek R1 | 66.7 | 71.6 | 83.2 | 66.7 | 80.5 | 80.7 | 69.8 | 48.5 |
| GPT-4.5 preview | 75.2 | 68.9 | 71.1 | 75.2 | 72.3 | 69.3 | 64.3 | 61.4 |
| Claude 3.7 Sonnet | 67.5 | 65.6 | 66.0 | 67.5 | 76.5 | 63.3 | 63.4 | 56.8 |
| Qwen2.5-Max | 64.4 | 62.3 | 51.4 | 64.4 | 75.3 | 58.3 | 67.9 | 56.3 |
| Claude 3.5 Sonnet (2024-10-22) | 67.1 | 59.0 | 56.7 | 67.1 | 69.3 | 52.3 | 55.0 | 53.8 |
| GPT-4o (2024-11-20) | 46.1 | 52.2 | 55.8 | 46.1 | 64.9 | 42.9 | 56.2 | 47.4 |
| Qwen2.5-72B Instruct | 57.6 | 51.4 | 45.4 | 57.6 | 64.4 | 54.3 | 51.9 | 35.0 |
| Gemma 3 27B | 39.9 | 50.0 | 43.8 | 39.9 | 74.9 | 55.4 | 51.4 | 34.6 |
| Qwen2.5-Coder-32B Instruct | 56.8 | 46.2 | 42.1 | 56.8 | 58.7 | 46.6 | 49.9 | 23.2 |
| R1-Distill-Qwen-32B | 33.7 | 45.6 | 52.2 | 33.7 | 55.7 | 59.4 | 45.4 | 26.8 |
| Mistral Small 3.1 (2503) | 36.2 | 44.0 | 44.8 | 36.2 | 63.7 | 39.4 | 50.5 | 29.1 |
| Claude 3.5 Haiku | 51.4 | 43.4 | 28.1 | 51.4 | 61.9 | 35.5 | 48.5 | 35.4 |
| GPT-4o mini | 43.2 | 41.3 | 32.8 | 43.2 | 56.8 | 36.3 | 50.0 | 28.6 |

## C. Current leaderboard, release 2026-06-25 -- NOT comparable

Different (unpublished) questions, different tasks per category (Coding = code_generation + code_completion of 2025-04-25; Reasoning adds theory_of_mind and logic_with_navigation; IF was rebuilt 2025-11-25), and an Agentic Coding category we did not run. Shown so the audience can place the base model: **Qwen3.8 27B** is the model ours is quantised from.

| model (release) | global | Reasoning | Coding | IF | Agentic Coding | Mathematics | Data Analysis | Language |
|---|---|---|---|---|---|---|---|---|
| Claude Fable 5.1 Max Effort [2026-06-25] | 83.4 | 91.7 | 86.4 | 73.0 | 66.1 | 97.0 | 80.3 | 89.5 |
| claude-opus-5-5-max-effort [2026-06-25] | 83.2 | 92.2 | 89.2 | 65.7 | 71.7 | 97.1 | 80.3 | 86.3 |
| GPT-6 Astra Max Effort [2026-06-25] | 82.2 | 92.7 | 80.4 | 75.6 | 57.3 | 96.8 | 83.0 | 89.4 |
| GPT-6 Sol Max Effort [2026-06-25] | 79.2 | 88.7 | 81.8 | 68.6 | 52.9 | 96.4 | 81.2 | 85.3 |
| Qwen 3.8 Max [2026-06-25] | 78.5 | 88.2 | 72.9 | 74.1 | 64.7 | 91.3 | 78.4 | 79.7 |
| Smaug Mini [2026-06-25] | 76.9 | 82.8 | 75.4 | 73.9 | 60.8 | 89.7 | 78.8 | 77.0 |
| Qwen 3.8 Flash Next [2026-06-25] | 76.2 | 87.4 | 72.5 | 77.1 | 61.6 | 85.8 | 74.2 | 74.6 |
| Claude Sonnet 5 xHigh Effort [2026-06-25] | 76.0 | 88.7 | 80.7 | 63.9 | 59.4 | 92.9 | 71.7 | 75.0 |
| Qwen3.8 27B [2026-06-25] | 75.3 | 80.0 | 75.7 | 72.7 | 61.4 | 86.2 | 76.6 | 74.3 |
| GPT-6 Luna Max Effort [2026-06-25] | 72.0 | 81.8 | 79.0 | 55.9 | 51.2 | 89.1 | 73.4 | 73.8 |
| GLM-5.3 Flash [2026-06-25] | 71.6 | 77.6 | 79.0 | 52.8 | 56.8 | 81.2 | 76.4 | 77.3 |
| QwQ 32B [2025-04-25] | 69.5 | 76.7 | 61.4 | 81.8 | -- | 76.1 | 69.5 | 51.5 |
| gpt-5.4-mini-xhigh [2026-06-25] | 66.4 | 71.3 | 71.6 | 59.8 | 41.7 | 78.5 | 70.8 | 71.0 |
| Qwen 3.6 27B [2026-06-25] | 64.0 | 70.3 | 71.8 | 53.2 | 39.3 | 79.9 | 70.4 | 63.3 |
| Gemma 4 31B [2026-01-08] | 61.6 | 59.4 | 60.3 | 67.6 | 40.0 | 73.9 | 58.8 | 71.3 |

Sources: https://livebench.ai/, https://livebench.ai/static/js/main.20d28c99.js, https://livebench.ai/table_2026_06_25.csv, https://livebench.ai/table_2026_01_08.csv, https://livebench.ai/table_2025_12_23.csv, https://livebench.ai/table_2025_11_25.csv, https://livebench.ai/table_2025_05_30.csv, https://livebench.ai/table_2025_04_25.csv, https://livebench.ai/table_2025_04_02.csv, https://livebench.ai/table_2024_11_25.csv, https://livebench.ai/categories_2026_06_25.json, https://livebench.ai/categories_2026_01_08.json, https://livebench.ai/categories_2025_12_23.json, https://livebench.ai/categories_2025_11_25.json, https://livebench.ai/categories_2025_05_30.json, https://livebench.ai/categories_2025_04_25.json, https://livebench.ai/categories_2025_04_02.json, https://livebench.ai/categories_2024_11_25.json; https://livebench.ai/table_2024_11_25.csv; HF dataset livebench/model_judgment (split leaderboard).
