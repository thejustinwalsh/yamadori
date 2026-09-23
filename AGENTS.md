# Repository guidance

**Yamadori** serves a local 27B (Bonsai 2, ternary) with a code-intelligence
tool layer, for SWE work in TypeScript, Rust/WASM across C ABIs, and
three.js TSL / TypeGPU.

Clients reach it at `:1234`, the proxy (`mcp/server.py`, logic in
`mcp/proxy.py`). The proxy advertises one model, `yamadori`, and any
unknown name resolves to it (`mcp/catalog.py`). It serves `/v1` and the
dashboard at `/` (a browser asking for HTML gets the React
SPA; any other client gets the JSON service descriptor, and old `/dash`
links redirect). llama-swap sits behind it on loopback `:11434`, where
`bonsai` and `bonsai-agent` are the same process: the alias exists only for
old clients. The tools API is `:1235` and Laya is `:1237`. The watchdog
supervises five services: llama-swap, proxy, tools-api, laya and worker.

## Naming

**Products, routes, groups and classifications take bonsai terms.** They are
the names a person reads.

| name | what it is | in code today |
|---|---|---|
| `canopy` | serving group for the main model | no. `config.yaml` says `primary` |
| `rootstock` | embeddings + reranker, always resident | no. `config.yaml` says `retrieval` |
| `graft` | on-demand models, swapped in and evicted | no. `config.yaml` says `ondemand` |
| `taproot` | result tier: a declaration with this name is here | yes, `find_by_meaning` output |
| `branch` | result tier: both retrievers agree | yes |
| `shoot` | result tier: one retriever only, unproven | yes |
| `rings` | the durable work log that survives compaction | yes, `mcp/rings.py` |
| `shomen` | deep thinking, the second context | yes, `mcp/shomen.py` |

Whether the group names or `config.yaml` should change is an open operator
decision. Do not "fix" either side on your own.

In anything a user sees, the second context is called **deep thinking**.

**Tool and MCP call names take Jane Street conventions.** They are the names a
machine reads, and they are part of the prompt.

- `snake_case`, verb first, spelled out; no abbreviations
- `_opt` when the answer may legitimately be absent (`find_definition_opt`)
- `_exn` only if it raises, which none of these do
- `of_` / `to_` for conversions

**The surface** (`proxy.our_tools()`):

| who sees it | tools |
|---|---|
| MCP clients (`code_search.TOOLS`) | `find_by_meaning`, `find_definition_opt`, `find_references`, `find_by_pattern`, `read_file_range`, `summarize_text`, `describe_index`, `run_check` |
| the model only (proxy executes them) | the eight above, plus `record_step`, `read_rings` (`INTERNAL_TOOLS`, scoped to a conversation key only the proxy has) and `bind_project_context` |
| off by default | `delegate_investigation`. Only with `YAMADORI_DELEGATE_TOOL=1` or a tier's `delegate` flag. It is a benchmark arm. |
| the model, only when `YAMADORI_IMAGEGEN_URL` is set | `generate_image` (`mcp/images.py`). Tier `low` and up, even when the code tools are withheld; never to deep thinking. See `docs/IMAGEGEN.md`. |

Deep thinking and fan-out are **not tools**. `mcp/selection.py` turns them on
per request, and the tier (`mcp/tiers.py`) only says what is *allowed*.
Selection uses two signals for deep thinking. The first is the regex
(`rule_baseline`) plus a symbol lookup in the held packages. The second is
the trained Laya `route_in` head. **If the two disagree, deep thinking
runs.** If Laya is down, its signal is recorded as `None` and the rule
decides alone. Every decision rides on the response as `x_yamadori`.

## Tool descriptions are prompts

Rewriting one description moved first-call routing from 10.7/14 to 11.7/14.
`find_references` failed in every prompt variant while its description said
what the tool *did*. It won once the description did three things: led with
the question it answers, contrasted itself with the tool it was losing to,
and listed the phrasings that should trigger it. **These numbers are at
risk.** The eval sent `max_tokens: 400` to a thinking model and did not
record `finish_reason`. See `docs/CONSTRAINTS.md` #31, and re-run before
citing them.

Write descriptions as trigger conditions. When a tool is mis-selected, fix its
description before touching the system prompt.

## Failure returns carry the next step

A tool that fails without saying why causes retry loops. `find_by_pattern`
returned "no matches in 0 files" for an unindexed directory, and that
produced 14 consecutive retries with permuted arguments. Every failure path
returns three things: the situation, whether it is retryable (as a fact),
and a remedy with an owner. It names what IS available: near-miss symbols,
indexed roots, top-level directories.

Never add a prompt rule to suppress a loop a tool return is causing.

## One door to the model, one budget rule

- **Client requests** are shaped by `tiers.apply()` in `proxy.prepare`.
  **Internal generation** (`summarize_text`, shomen, the worker) goes through
  `mcp/model.py`, which calls the same `tiers.apply()`. Nothing else talks to
  llama-swap. Internal callers do not call `:1234`, because that would
  re-enter admission, pollute the corpus and need a key. `model.py` explains
  why.
- **Tokens** come from `tiers.budget()`. The client's `max_tokens` is an
  *answer* allowance, `answer = max(client, A_MIN)` with `A_MIN=2048`.
  Thinking is derived from the request's share of the KV pool
  (`mcp/budget.py`: main 5/8, one helper 3/8 -- at `-c 163840`, main
  102,400 and helper 61,440; at 147,456, 92,160 and 55,296):
  `thinking = max(share[role] // share_n - prompt - answer, 1024)`.
  Upstream gets `max_tokens = thinking + answer`,
  `reasoning_budget_tokens = thinking` and a `reasoning_budget_message`.
  A fan-out of n samples shares main's 5/8 n ways; deep thinking
  (`shomen`) draws from the helper's 3/8, one investigation at a time
  (`admission.HELPER_LANES = 1`). The split is the operator's decision of
  2026-09-22, not a measurement: it replaced a 1/2 + 2 x 1/4 split the same
  day (a second concurrent helper needed two conversations investigating at
  once), which had replaced a fixed `R_CAP=8192` breaker and a 60/25/15 split
  that had no measurement behind it (`docs/CONSTRAINTS.md` item 19). Natural thinking ran 682–2,826 tokens
  at n=7 (§1), so no measured result says a shorter thought is better.
  `config.yaml` launches with `--reasoning-budget 32768` and the same
  message, the server default for a request that sends no budget of its own.
- **A `finish_reason: length` is a budget event, never an answer.**
  `model.BudgetEvent`.

## Prompting this model

- A decision-router table beats prose. Measured 10.7 vs 10.0.
- Prohibitions degrade monotonically: 0 `never` (10.7) > 2 (10.0) > 6 (9.3).
  A negative instruction fires attention on the thing it forbids. Keep at most
  two, each naming a specific observed failure. (Same eval, same caveat as
  above.)
- `reasoning_effort` values are **read from the served chat template**
  (`tiers.accepted_efforts()`). Today they are `low`, `medium` and `xhigh`.
  Any other value makes the template return an HTTP 500. `safe_effort` rounds
  up: `high` becomes `xhigh`, `minimal` becomes `low`, and `max` becomes
  `xhigh`. The default is `medium` (`config.yaml`).
- Tier ladder (`tiers.TIERS`), where each tier sets what is *allowed*:
  - `minimal` is the model as it ships
  - `low` adds retrieval
  - `medium` adds hints
  - `high` adds fan-out
  - `max` adds deep thinking
- Effort does not order thinking length at the measured n
  (`docs/CONSTRAINTS.md` §1b). Do not derive token numbers from effort.

## Claims carry their evidence

Every threshold, default and cut names the script that justifies it and the n
it was measured at. A number from `n=11` is labelled as such. Three.js results
are labelled contaminated: it is in every training set.

If a measurement does not survive a repeat, it is not a result. This repo has a
history of single-run conclusions that reversed.

**`docs/PROTOCOL.md` enforces this.** It has sixteen rules, and each one is
attached to a specific failure in this repo. Read it before changing a
default, cutting a component, or reporting a result.

## Before you claim anything works

1. `python scripts/run_tests.py`. This runs every offline suite plus
   `ruff --select=E9,F`. Use the stack interpreter,
   `C:\Users\jwals\textgen\installer_files\env\python.exe`. A suite that
   prints no count is a failure.
2. Then run the live suite **through `:1234`**:
   `YAMADORI_TEST_KEY=... python scripts/run_tests.py --live`. This covers
   `mcp/test_live_stack.py`, `mcp/test_tools_live.py` and
   `bench/test_laya_head.py --serve`. It judges the model's actual output
   through the door users use.
3. **Never test around the stack.** Stubs and direct calls to `:11434`
   have declared whole systems dead when the real cause was a setting (a
   budget, a hop cap, a timeout).
4. **One GPU consumer at a time.** Two consumers on one card degrade each
   other into 429s and 502s, and the loser looks like the one with the bug.
   Queue benchmark runs through `bench/queue_runner.py`. Treat a 429 as
   "not run", never as a failure.
5. The watchdog (`scripts/watchdog.ps1`) restarts a service only on the
   **second consecutive** failed `/health`, 5 minutes apart. Before it
   restarts llama-swap, it reads the slot counters twice, 10 s apart. If
   they are moving, the stack is generating and is not restarted. A slow
   answer is not an outage.

## What has been cut, and why

- `judge` (Laya as a truth judge) scored 6/10 against a 5/10 coin flip, with
  systematic false positives on negative cases. It scores what a passage is
  *about*, not whether a proposition holds. `scripts/eval_judge.py` is the
  gate to restore it.
- `apply_edit` was cut because the harness owns writing. There is one write
  path per repo.

**The reranker is not cut, not trusted, and not used.** `docs/FINDINGS.md`
#20 found that `/v1/rerank` scores depend on batch composition: the same 89
documents scored 15–16/89 batched against 68/89 one at a time. That voids
every reranker number in the repo, in both directions:

- the old "no gain above `top_k=2`" cut (11 queries, against an index of
  zero vectors)
- `bench/retrieval_results.jsonl` (n=356, rerank hit@1 48/356 vs embedding
  264/356)

The resolution is in `docs/PLAN.md` under "Cut criteria". Do not cut or
defend the reranker until the rank path is fixed and re-measured.

## Laya

**Use Laya only where the state is FIXED and the options vary**, as a
`choice` over a small closed set. Its scores are not comparable across
different passages:

- Holding a passage fixed and varying the query separates real from nonsense
  by 0.497.
- Varying both collapses the gap to zero.

Never threshold a Laya score across queries.

It also silently drops the tail of a long `state`. A query placed after ~120
lines of preamble is discarded, and every input then scores identically. Put
the thing being judged first.

**`route_in` head.** It was retrained on 2026-09-22 on 289 labels
(`index/laya/route_in.json`; the previous head is in `index/laya/_backup_*`).
It was scored by `bench/eval_route_heldout.py` on 120 held-out
package-domain labels, which no training file matches:

| condition | investigate-vs-not |
|---|---|
| new head | 80/120 |
| old head | 64/120 |
| `selection.decide` | 89/120 |

The head still does not beat the rule, so it never decides alone.

## Where things live

- **Datasets.** `mcp/datasets.py` holds the stages and `mcp/jobs.py` is the
  durable queue. The lane limit is global across processes
  (`{gpu: 1, cpu: 4, net: 4}`). `mcp/worker.py` claims and runs the jobs.
  There is no review stage: review is optional and happens after the fact on
  the dashboard. The `clarify` stage is model-assisted. A licence is filled only
  from a verified verbatim quote, and a guessed licence is never proposed.
- **Packages.** `mcp/deps.py index name@version [--embed]` indexes a
  package, and `mcp/deps.py health name@version` checks one (it reports
  `zero_vectors`: an index built without `--embed` is all zeros by design). They are stored in
  `index/packages/*.sqlite3`. `describe_index` does *not* list them; it
  covers only the bound repository. Version is part of identity.
- **Concept seeds.** `mcp/concept_seed.py` draws from the 27B's own token
  embeddings, `index/token_embd.npz`. Extract them once with
  `scripts/extract_token_embd.py`.
- **Dashboard.** The React app is in `web/`, and the committed `web/dist` is
  served at `/dash`. The Python pages are at `/dash/classic*`. `/dash/api/*`
  requires a key.
- **Design.** `design/` holds the design source. `docs/SELECTION-BUILD.md`
  is the selection plan, and `docs/HANDOFF.md` is the latest state.

## Checks

`run_check` executes project commands by LABEL from a whitelist
(`code_search.VERIFY_CHECKS`), never a command line, so text arriving from a
source file or a model cannot reach a shell. Adding a project means adding a
row.
