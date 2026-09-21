# Repository guidance

`bonsai-stack` serves a local 27B (Bonsai 2, ternary) with a code-intelligence
tool layer, for SWE work in TypeScript, Rust/WASM across C ABIs, and
three.js TSL / TypeGPU. The model is reached at `:1234` over the OpenAI API.

## Naming

**Products, routes, groups and classifications take bonsai terms.** They are
the names a person reads.

| name | what it is |
|---|---|
| `canopy` | the serving group holding the main model |
| `rootstock` | the group holding embeddings and reranker -- always resident |
| `graft` | on-demand models, swapped in and evicted |
| `taproot` | result tier: a declaration with this name is here |
| `branch` | result tier: both rankers agree |
| `shoot` | result tier: one ranker only, unproven |
| `rings` | the durable work log that survives compaction |

**Tool and MCP call names take Jane Street conventions.** They are the names a
machine reads, and they are part of the prompt.

- `snake_case`, verb first, spelled out; no abbreviations
- `_opt` when the answer may legitimately be absent (`find_definition_opt`)
- `_exn` only if it raises, which none of these do
- `of_` / `to_` for conversions

Current surface: `find_by_meaning`, `find_definition_opt`, `find_references`,
`find_by_pattern`, `read_file_range`, `summarize_text`, `describe_index`,
`run_check`.

## Tool descriptions are prompts

Measured, not assumed: rewriting one description moved first-call routing from
10.7/14 to 11.7/14. `find_references` failed in every prompt variant while its
description said what the tool *did*; it wins once the description leads with
the question it answers, contrasts itself against the tool it was losing to,
and lists the phrasings that should trigger it.

Write descriptions as trigger conditions. When a tool is mis-selected, fix its
description before touching the system prompt.

## Failure returns carry the next step

A tool that fails without saying why causes retry loops -- `find_by_pattern`
returning "no matches in 0 files" for an unindexed directory produced 14
consecutive retries with permuted arguments. Every failure path names what IS
available: near-miss symbol names, indexed roots, top-level directories.

Never add a prompt rule to suppress a loop a tool return is causing.

## Prompting this model

- A decision-router table beats prose. Measured 10.7 vs 10.0.
- Prohibitions degrade monotonically: 0 `never` (10.7) > 2 (10.0) > 6 (9.3).
  A negative instruction fires attention on the thing it forbids. Keep at most
  two, each naming a specific observed failure.
- `reasoning_effort: medium` is the default for agentic work. `xhigh` is
  useful as a stress test of tool ergonomics -- it exposes bad tool returns by
  persisting against them where `medium` gives up and answers.
- Point agents at `bonsai-agent`, never `bonsai`.

## Claims carry their evidence

Every threshold, default and cut in this repo names the script that justifies
it and the n it was measured at. A number from `n=11` is labelled as such.
Three.js results are labelled contaminated -- it is in every training set.

If a measurement does not survive a repeat, it is not a result. This repo has a
history of single-run conclusions that reversed.

## What has been cut, and why

- `judge` (Laya as a truth judge): 6/10 against a 5/10 coin flip, with
  systematic false positives on negative cases. It scores what a passage is
  *about*, not whether a proposition holds. Independently replicated by at
  least four third parties. `scripts/eval_judge.py` is the gate to restore it.
- `apply_edit`: the harness owns writing. One write path per repo.
- Reranking above `top_k=2`: no measurable gain at the shipped cutoff, ~1s per
  query. Kept below it as a latency decision only.

## Laya

Usable only where the state is FIXED and the options vary -- `choice` over a
small closed set. Its scores are not comparable across different passages:
holding a passage fixed and varying the query separates real from nonsense by
0.497, but varying both collapses the gap to zero. Never threshold a Laya
score across queries.

It also silently drops the tail of a long `state`; a query placed after ~120
lines of preamble is discarded outright and every input scores identically.
Put the thing being judged first.

## Checks

`run_check` executes project commands by LABEL from a whitelist, never a
command line, so text arriving from a source file or a model cannot reach a
shell. Adding a project means adding a row.
