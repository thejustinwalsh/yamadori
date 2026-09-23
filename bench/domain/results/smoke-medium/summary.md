# Domain benchmark -- smoke-medium

Conditions: {"effort": "medium", "max_tokens": 4096, "temperature": 0.0, "body_tier": "max"}. Rows: 0. Tasks: 0 uncontaminated, 0 contaminated (three_tsl).

> **Caveat:** 10 stale_code rows excluded (measured a stack since fixed): pre-fix stack (proxy pid 4500 started 19:15): package-named glob/path matched nothing in find_by_pattern/read_file_range; MAX_TOOL_HOPS breaker since removed; selection symbol lookup matched keywords/unrelated packages
> **Caveat:** preflight checks were OVERRIDDEN for part of this run: proxy_fresh

## uncontaminated domains (headline)

0 tasks. .

| comparison | family | paired n | pass (a / b) | b only | a only | discordant | exact p | Bonferroni p | verdict |
|---|---|---|---|---|---|---|---|---|---|

## three_tsl only -- CONTAMINATED

0 tasks. .

| comparison | family | paired n | pass (a / b) | b only | a only | discordant | exact p | Bonferroni p | verdict |
|---|---|---|---|---|---|---|---|---|---|

## all domains -- includes contaminated three_tsl

0 tasks. .

| comparison | family | paired n | pass (a / b) | b only | a only | discordant | exact p | Bonferroni p | verdict |
|---|---|---|---|---|---|---|---|---|---|

## Per arm (all domains, last row per pair)

| arm | scored | passed | rate (Wilson 95%) | stack errors (rows) | kinds | median s | median prompt tok | median completion tok | median hops | median reasoning chars | cap fired | cap sent | fail stages |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

## By domain (scored pass / n per arm)

| domain | tasks |  |
|---|---|

## What the stack did (from x_yamadori, non-stack-error rows)

| arm | n | tools offered | hints selected | hints emitted | investigate chosen | investigation searched | injected | fanned out |
|---|---|---|---|---|---|---|---|---|
