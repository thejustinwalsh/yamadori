# Domain benchmark -- overnight-0923

Conditions: {"effort": "medium", "max_tokens": 4096, "temperature": 0.0, "body_tier": "max", "suite": "core,react"}. Rows: 9. Tasks: 4 uncontaminated, 0 contaminated (none).

> **Caveat:** preflight checks were OVERRIDDEN for part of this run: live_stack, run_tests
> **Caveat:** A6 scored exactly 100%
> **Caveat:** timing is CONCURRENT with livebench, swebench: every seconds and tokens/s figure shared the GPU with them and is not a clean measurement

## uncontaminated domains (headline)

4 tasks. family `augmentation`: m=1 (alpha 0.0500), at least 6 discordant pairs needed before any comparison in it can reach significance; family `self_check`: m=1 (alpha 0.0500), at least 6 discordant pairs needed before any comparison in it can reach significance.

| comparison | family | paired n | pass (a / b) | b only | a only | discordant | exact p | Bonferroni p | verdict |
|---|---|---|---|---|---|---|---|---|---|
| A6 vs A0 | augmentation | 2 | 2 / 2 | 0 | 0 | 0 | 1 | 1 | none (underpowered) |
| S0 vs A0 | self_check | 3 | 2 / 2 | 0 | 0 | 0 | 1 | 1 | none (underpowered) |

## contaminated only (none in this run) -- CONTAMINATED

0 tasks. family `augmentation`: m=1 (alpha 0.0500), at least 6 discordant pairs needed before any comparison in it can reach significance; family `self_check`: m=1 (alpha 0.0500), at least 6 discordant pairs needed before any comparison in it can reach significance.

| comparison | family | paired n | pass (a / b) | b only | a only | discordant | exact p | Bonferroni p | verdict |
|---|---|---|---|---|---|---|---|---|---|
| A6 vs A0 | augmentation | 0 | 0 / 0 | 0 | 0 | 0 | 1 | 1 | none (underpowered) |
| S0 vs A0 | self_check | 0 | 0 / 0 | 0 | 0 | 0 | 1 | 1 | none (underpowered) |

## all domains -- includes contaminated (none in this run)

4 tasks. family `augmentation`: m=1 (alpha 0.0500), at least 6 discordant pairs needed before any comparison in it can reach significance; family `self_check`: m=1 (alpha 0.0500), at least 6 discordant pairs needed before any comparison in it can reach significance.

| comparison | family | paired n | pass (a / b) | b only | a only | discordant | exact p | Bonferroni p | verdict |
|---|---|---|---|---|---|---|---|---|---|
| A6 vs A0 | augmentation | 2 | 2 / 2 | 0 | 0 | 0 | 1 | 1 | none (underpowered) |
| S0 vs A0 | self_check | 3 | 2 / 2 | 0 | 0 | 0 | 1 | 1 | none (underpowered) |

## Per arm (all domains, last row per pair)

| arm | scored | passed | rate (Wilson 95%) | stack errors (rows) | kinds | median s (CONCURRENT) | median prompt tok | median completion tok | median hops | median reasoning chars | cap fired | cap sent | fail stages |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 | 4 | 3 | 75.0% [30.1%-95.4%] | 0 | {} | 289.8 | 168.0 | 1373.0 | 1.0 | 3770.0 | 0 | [97934, 98057, 98115, 98131] | {"compile": 1} |
| A6 | 2 | 2 | 100.0% [34.2%-100.0%] | 0 | {} | 854.0 | 8015.5 | 1467.5 | 2.0 | 2151.0 | 0 | [91914, 93947] | {} |
| S0 | 3 | 2 | 66.7% [20.8%-93.9%] | 0 | {} | 1529.6 | 1285 | 1192 | 2 | 3618 | 0 | [97175, 97565, 97843] | {"test": 1} |

## Where it failed, what it cost, what it checked (all domains)

| arm | pass | extract | compile | test | final compiles | mean s (CONCURRENT) | median s (CONCURRENT) | mean prompt tok | mean completion tok | proxy tool hops | deep-thinking hops | check rounds (mean / max) | checked any | final answer passed public check |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 | 3 | 0 | 1 | 0 | 3/4 | 382.1 | 289.8 | 208.0 | 2224.0 | 0.0 | 0.0 | - | - | - |
| A6 | 2 | 0 | 0 | 0 | 2/2 | 854.0 | 854.0 | 8015.5 | 1467.5 | 1.0 | 2.0 | - | - | - |
| S0 | 2 | 0 | 0 | 1 | 3/3 | 1825.4 | 1529.6 | 1399.7 | 7108.7 | 0.0 | 0.0 | 1.0 / 2 | 2/3 | 3/3 |

## Self-check vs its one-shot twin (uncontaminated)

| S arm vs twin | paired n | pass (twin / S) | S only | twin only | exact p | fixed by checking (compile -> pass) | compile -> test | extract -> pass | test -> pass | compile fails (twin / S) |
|---|---|---|---|---|---|---|---|---|---|---|
| S0 vs A0 | 3 | 2 / 2 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 1 / 0 |

## Style (separate score, never part of pass/fail; TS and React answers)

| arm | linted | lint errors (mean) | lint warnings (mean) | 0 lint errors | React 19 flags | credits (called for) | scorer errors |
|---|---|---|---|---|---|---|---|
| A0 | 1 | 1.0 | 0.0 | 0 | {} | {} | 0 |
| A6 | 0 | None | None | 0 | {} | {} | 0 |
| S0 | 1 | 0.0 | 0.0 | 1 | {} | {} | 0 |

## By domain (scored pass / n per arm)

| domain | tasks | A0 | A6 | S0 |
|---|---|---|---|---|
| rust_wasm | 3 | 3/3 | 2/2 | 2/2 |
| typescript | 1 | 0/1 | 0/0 | 0/1 |

## What the stack did (from x_yamadori, non-stack-error rows)

| arm | n | tools offered | hints selected | hints emitted | investigate chosen | investigation searched | injected | fanned out |
|---|---|---|---|---|---|---|---|---|
| A0 | 4 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 |
| A6 | 2 | 2/2 | 2/2 | 0/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| S0 | 3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 |
