# Domain benchmark -- selfcheck-probe

Conditions: {"effort": "medium", "max_tokens": 4096, "temperature": 0.0, "body_tier": "max", "suite": "core"}. Rows: 3. Tasks: 1 uncontaminated, 0 contaminated (none).

> **Caveat:** preflight checks were OVERRIDDEN for part of this run: live_stack
> **Caveat:** 1 row(s) reclassified stack_error/server_reasoning_cap: no answer, finish stop, ~32,768 reasoning tokens and no budget message -- the server's launch --reasoning-budget cut the model off (resume regenerates them)
> **Caveat:** timing is CONCURRENT with livebench, swebench: every seconds and tokens/s figure shared the GPU with them and is not a clean measurement

## uncontaminated domains (headline)

1 tasks. family `self_check`: m=1 (alpha 0.0500), at least 6 discordant pairs needed before any comparison in it can reach significance.

| comparison | family | paired n | pass (a / b) | b only | a only | discordant | exact p | Bonferroni p | verdict |
|---|---|---|---|---|---|---|---|---|---|
| S0 vs A0 | self_check | 0 | 0 / 0 | 0 | 0 | 0 | 1 | 1 | none (underpowered) |

## contaminated only (none in this run) -- CONTAMINATED

0 tasks. family `self_check`: m=1 (alpha 0.0500), at least 6 discordant pairs needed before any comparison in it can reach significance.

| comparison | family | paired n | pass (a / b) | b only | a only | discordant | exact p | Bonferroni p | verdict |
|---|---|---|---|---|---|---|---|---|---|
| S0 vs A0 | self_check | 0 | 0 / 0 | 0 | 0 | 0 | 1 | 1 | none (underpowered) |

## all domains -- includes contaminated (none in this run)

1 tasks. family `self_check`: m=1 (alpha 0.0500), at least 6 discordant pairs needed before any comparison in it can reach significance.

| comparison | family | paired n | pass (a / b) | b only | a only | discordant | exact p | Bonferroni p | verdict |
|---|---|---|---|---|---|---|---|---|---|
| S0 vs A0 | self_check | 0 | 0 / 0 | 0 | 0 | 0 | 1 | 1 | none (underpowered) |

## Per arm (all domains, last row per pair)

| arm | scored | passed | rate (Wilson 95%) | stack errors (rows) | kinds | median s (CONCURRENT) | median prompt tok | median completion tok | median hops | median reasoning chars | cap fired | cap sent | fail stages |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 | 0 | 0 | - [0.0%-0.0%] | 1 | {"server_reasoning_cap": 1} | None | None | None | None | None | 0 | [] | {} |
| S0 | 0 | 0 | - [0.0%-0.0%] | 2 | {"exception": 2} | None | None | None | None | None | 0 | [] | {} |

## Where it failed, what it cost, what it checked (all domains)

| arm | pass | extract | compile | test | final compiles | mean s (CONCURRENT) | median s (CONCURRENT) | mean prompt tok | mean completion tok | proxy tool hops | deep-thinking hops | check rounds (mean / max) | checked any | final answer passed public check |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 | 0 | 0 | 0 | 0 | 0/0 | None | None | None | None | None | None | - | - | - |
| S0 | 0 | 0 | 0 | 0 | 0/0 | None | None | None | None | None | None | None / None | 0/0 | 0/0 |

## Self-check vs its one-shot twin (uncontaminated)

| S arm vs twin | paired n | pass (twin / S) | S only | twin only | exact p | fixed by checking (compile -> pass) | compile -> test | extract -> pass | test -> pass | compile fails (twin / S) |
|---|---|---|---|---|---|---|---|---|---|---|
| S0 vs A0 | 0 | 0 / 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 / 0 |

## Style (separate score, never part of pass/fail; TS and React answers)

| arm | linted | lint errors (mean) | lint warnings (mean) | 0 lint errors | React 19 flags | credits (called for) | scorer errors |
|---|---|---|---|---|---|---|---|
| A0 | 0 | None | None | 0 | {} | {} | 0 |
| S0 | 0 | None | None | 0 | {} | {} | 0 |

## By domain (scored pass / n per arm)

| domain | tasks | A0 | S0 |
|---|---|---|---|
| typescript | 1 | 0/0 +1err | 0/0 +1err |

## What the stack did (from x_yamadori, non-stack-error rows)

| arm | n | tools offered | hints selected | hints emitted | investigate chosen | investigation searched | injected | fanned out |
|---|---|---|---|---|---|---|---|---|
| A0 | 0 | | | | | | | |
| S0 | 0 | | | | | | | |
