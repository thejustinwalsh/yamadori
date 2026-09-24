# Domain benchmark -- overnight-0923-t1

Conditions: {"effort": "medium", "max_tokens": 4096, "temperature": 1.0, "body_tier": "max", "suite": "core,react", "server_sampling": {"model": "Ternary-Bonsai-2-27B-Abliterated-PTQ1_0-mtp-lean.gguf", "spec_type": "draft-mtp", "reasoning_budget": "32768", "temp": "1.0", "top_p": "0.95", "top_k": "20", "min_p": "0.0", "presence_penalty": "0", "repeat_penalty": "1.0"}, "tool_turns_limit": [20]}. Rows: 6. Tasks: 6 uncontaminated, 0 contaminated (none).

> **Caveat:** 12 stale_code rows excluded (measured a stack since fixed): condition epoch 2026-09-23 15:43:17: ran on an earlier stack (no generate_image/describe_image, no tool-turn cap); redone under one condition (operator); pre-sequential-fanout (A6 made under parallel fan-out, 2026-09-23)
> **Caveat:** preflight checks were OVERRIDDEN for part of this run: live_stack, proxy_fresh, run_tests
> **Caveat:** the hints corpus changed between invocations of this run (manifest runs[].hints_corpus): hint rows before and after measured different corpora
> **Caveat:** timing is CONCURRENT with livebench, swebench: every seconds and tokens/s figure shared the GPU with them and is not a clean measurement

## uncontaminated domains (headline)

6 tasks. .

| comparison | family | paired n | pass (a / b) | b only | a only | discordant | exact p | Bonferroni p | verdict |
|---|---|---|---|---|---|---|---|---|---|

## contaminated only (none in this run) -- CONTAMINATED

0 tasks. .

| comparison | family | paired n | pass (a / b) | b only | a only | discordant | exact p | Bonferroni p | verdict |
|---|---|---|---|---|---|---|---|---|---|

## all domains -- includes contaminated (none in this run)

6 tasks. .

| comparison | family | paired n | pass (a / b) | b only | a only | discordant | exact p | Bonferroni p | verdict |
|---|---|---|---|---|---|---|---|---|---|

## Per arm (all domains, last row per pair)

| arm | scored | passed | rate (Wilson 95%) | stack errors (rows) | kinds | median s (CONCURRENT) | median prompt tok | median completion tok | median hops | median reasoning chars | cap fired | cap sent | fail stages |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A6 | 5 | 3 | 60.0% [23.1%-88.2%] | 1 | {"http": 1} | 181.0 | 13910 | 2861 | 3 | 516 | 0 | [90598, 91261, 91973, 92323, 92648] | {"compile": 2} |

## Where it failed, what it cost, what it checked (all domains)

| arm | pass | extract | compile | test | final compiles | mean s (CONCURRENT) | median s (CONCURRENT) | mean prompt tok | mean completion tok | proxy tool hops | deep-thinking hops | check rounds (mean / max) | checked any | final answer passed public check |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A6 | 3 | 0 | 2 | 0 | 3/5 | 370.5 | 181.0 | 13564.8 | 5493.2 | 1.8 | 2.2 | - | - | - |

## Style (separate score, never part of pass/fail; TS and React answers)

| arm | linted | lint errors (mean) | lint warnings (mean) | 0 lint errors | React 19 flags | credits (called for) | scorer errors |
|---|---|---|---|---|---|---|---|
| A6 | 0 | None | None | 0 | {} | {} | 0 |

## Hints (rows with >= 1 hint injected; on those rows, pass vs their paired A0)

| arm | injected rows | rate | paired with A0 | pass (arm) | pass (A0) |
|---|---|---|---|---|---|
| A6 | 1 | 20.0% | 0 | 0 | 0 |

## Timing by GPU sharing (only `clean` rows are single-user timings)

| arm | clean rows | concurrent rows | mean s clean | median s clean | mean s concurrent |
|---|---|---|---|---|---|
| A6 | 5 | 0 | 370.5 | 181.0 | None |

## Mechanism health (evidence, not a verdict): % of rows where the arm allowed it; of those, % where it ran, and % where it produced data

| arm | retrieval | hints | deep_thinking | fanout | self_check | check_code |
|---|---|---|---|---|---|---|
| A6 | 100.0% / 60.0% / 40.0% | 100.0% / 100.0% / 20.0% | 100.0% / 100.0% / 40.0% | 100.0% / 100.0% / 100.0% | 0.0% / - / - | 100.0% / 80.0% / 80.0% |

Tool-turn cap (x_yamadori.tool_turns; hit = the loop landed at the cap -- data, not an error): A6 limit [20] hit 0/5.

Retrieval tool errors per arm (a tool that answered ok:false -- evidence, not a stack error): A6 0.
Known limit: x_yamadori.tools records only error: true/false, not the error code, so NO_INDEX, a broken tool and bad arguments from the model cannot be told apart here yet.

## By domain (scored pass / n per arm)

| domain | tasks | A6 |
|---|---|---|
| rust_wasm | 6 | 3/5 +1err |

## What the stack did (from x_yamadori, non-stack-error rows)

| arm | n | tools offered | hints selected | hints emitted | investigate chosen | investigation searched | injected | fanned out |
|---|---|---|---|---|---|---|---|---|
| A6 | 5 | 5/5 | 5/5 | 1/5 | 5/5 | 2/5 | 2/5 | 5/5 |
