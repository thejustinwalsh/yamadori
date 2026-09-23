<!-- Written 2026-09-22 by a read-only planning pass over the docs AND the code.
     Line numbers are as of that date. Verify before relying on any of them. -->

# Selection engine — build plan

The selection engine decides, per request, WHEN each system runs: the code
tools, recipe hints, fan-out (with concept seeds) and deep thinking (the second
context, `mcp/shomen.py`). The routing between the conversation and deep
thinking is the "corpus callosum".

## 1. What the design says the engine is

| # | decision | input it sees | decider (designed) | source |
|---|---|---|---|---|
| D0 | which systems are *allowed* | `reasoning_effort` / model name / `X-Yamadori-Features` | tier table | `mcp/tiers.py` TIERS, `from_header` |
| D1 | offer the code tools? | messages, bound root, held package store | deterministic fact check, no model | `mcp/domains.py` `tool_admission` |
| D2 | emit hints? | last user turn | domain pre-filter, embedding cosine >= 0.55, bucket collapse | `mcp/hints.py`; SELECTION §3 |
| D3 | deep thinking (route in)? | the question | regex decides; trained Laya head is a second signal; **disagreement escalates to investigate**; binary with an explicit `continue`; lean toward investigating when unsure | SELECTION §4 patterns 1 and 5; ROADMAP 2.2; LAYA.md Part II verdict |
| D4 | fan out, and how wide? | task class | rule: lookups N=1, design/refactor/"how should I" N=4; Laya's fan-out node failed (0.13 / 0.07) | `mcp/fanout.py` docstring; SELECTION §4 |
| D5 | route out (trust the finding?) | finding + retrieved paths | deterministic citation check; Laya `distil` removed (AUC 0.667) | `mcp/shomen.py` `_finish` |

The docs contradict each other on Laya's role: HANDOFF says "Laya is the
required router in and out", FINDINGS #22 says "Nothing gates on it", and
shomen.py says autonomous delegation is deliberately left out yet Laya is not
optional. SELECTION §4 pattern 5 reconciles them: Laya is in the loop but never
decides alone.

## 2. What is on the request path today

**Wired:**
- D0 in `proxy.prepare`: a tier hint from the model name applies only when the
  client sent no `reasoning_effort`.
- D1 via `domains.tool_admission` (package fallback now works).
- D2 via `hints.attach`: embedding floor plus bucket collapse only.
  `domains.eligible` and `filter_recipes` have no callers.
- D3 fires whenever `tier["investigate"]` is set, i.e. at `max`. No classifier.
- D4 non-streamed only; seeds via `fanout._seeds`.
- `delegate_investigation` is still offered to the model as a TOOL, which
  breaks the rule that deep thinking is not a tool.

**Built, not wired:** `shomen.route_in` and `distil`; `laya_router.py`;
`session.triage`; `fanout.break_tie`; `dialectic.py`; `judge`; Laya `/route`.
**Laya is called zero times on the request path.** The regex that beats it
lives only in `bench/laya_calibration.py` (`rule_baseline`, `signals`,
`index_facts`).

**Does not exist:** a selection module, a per-request trigger for D3 or D4, and
per-decision logging.

**Harms in wired code:**
1. At `max`, deep thinking runs on almost everything: the gate is
   `root is not None or cs.has_index()`, and `has_index()` is always true. When
   tools are withheld the investigation gets no tools, and an uncited
   general-knowledge answer is injected as "Findings from a separate
   investigation of the indexed source".
2. Fan-out: an answer with no file paths gets agreement 0.0, and `dissent_note`
   appends "only 0% agreed on the same file… unsettled" to every such answer at
   `high`/`max`. The winning variant is also thrown away (FINDINGS #18).
3. Streamed and non-streamed are different systems: `stream_body` has no
   fan-out and no final landing hop, and generates the answer twice.
   (Being fixed separately, 2026-09-22.)
4. Tiers shift effort: `low` sends `medium`, and `minimal` sends `low`. A
   same-effort comparison must set `effort` in the header.
5. Deep thinking's tools are taken from the main request's `payload["tools"]`,
   so a "deep thinking alone, retrieval off" arm has none.

## 3. Training data and the skills corpus

- **route_in** (`index/laya/route_in.json`), logistic head on `[CLS]`: held-out
  accuracy 0.726 ±0.045, against zero-shot 0.529, regex 0.841 and majority
  0.393; `beats_rule_baseline: false`. 89 labels (33 / 35 / 21, 10 hard), almost
  all about THIS repo rather than the target package domains. Needs 40 or more
  held-out labels on the binary + `continue` framing, and several hundred
  before a full fine-tune. Format: `bench/laya_routing_labels*.jsonl`
  `{question, context, label, hard?}`. Retrain with
  `scripts/train_laya.py --task all`; the floors in `bench/laya_baseline.json`
  must still pass.
- **grounded_excerpt**: 60 labels, 0.625 ±0.099. Not shippable.
- **Skills corpus**: 938 recipes. 19 buckets, 89 members, so 9.5% bucketed.
  Production probe score 67/89. The floor is uncalibrated for open retrieval;
  ROADMAP 1.5 (hints vs always-inject-the-best) has never run. "Stable" means:
  every pair of recipes that co-occur above the floor on the task set is either
  in one bucket or not mutually exclusive. Measured arguing rate 0, and the
  domain filter on.
- **Indexed source**: `three@0.185.1` and `typegpu@0.12.5` only (plus the koota
  and glyph repos). No Rust, WASM or C-ABI source is indexed.

## 4. Build plan

Each step: files, offline test, then a LIVE check on real model output through
:1234.

1. **Fix the harms.**
   - `fanout`: no dissent note when no path votes were cast.
   - Deep-thinking gate: require `payload["_tools_gate"]["offer"]`.
   - Build the deep-thinking tool list from `our_tools()`, not
     `payload["tools"]`.
   - Offline: new `mcp/test_fanout.py`, plus a `test_tools` case asserting
     `shomen.investigate` is not called for a LiveCodeBench prompt.
   - Live: `test_live_stack --only tiers`. The primes answer at `max` has no
     "unsettled" text, and its wall clock is within 1.5x of `minimal`.
2. **One path for streamed and non-streamed.** Share the post-loop code (final
   hop, fan-out). Live: the same prompt streamed and non-streamed gives the same
   hop count.
3. **`mcp/selection.py`.** A pure
   `decide(messages, tier, gate, state) -> {hints, investigate, fanout_n,
   because, signals}`. Move `rule_baseline` and `signals` into it, and have
   `bench/laya_calibration.py` and `train_laya.py` import them, so there is one
   copy. Add the symbol lookup as the hard-slice check. Offline: reproduce the
   regex predictions on all 89 labels; withhold deep thinking on all 342
   LiveCodeBench prompts; fire it on all 26 `context_economy_tasks`.
4. **Laya as the second signal.** Call `/route` with `engine: "trained"`. If
   Laya is down, record the signal as `None`; never guess. Disagreement means
   investigate. Log each decision as `_selection`. Offline: a 2x2 stub. Live:
   one proxy request whose log shows both signals.
5. **Effort bounds, it does not force.** The tier flag means "allowed";
   selection decides whether a system fires; the header still forces on or off,
   for experiments only; the engine never exceeds what the client asked for.
   Live: a TSL deprecation question at `max` investigates and cites a real
   `three@0.185.1` path; the primes prompt at `max` does not investigate.
6. **Domain filter for hints.** Apply `domains.eligible(detect(messages))`
   inside `hints.select`. The probe score must stay at 67/89 or better.
7. **Fan-out trigger.** Label about 40 prompts as lookup vs design. For code
   answers, compute agreement over import/symbol sets (`api_truth`), not over
   file paths.
8. **Keep `delegate_investigation`, behind a flag**, as a benchmark arm.
9. **Grow route_in labels** in the package domains. Promote the head only if it
   beats the regex.

## 5. Benchmark

**Arms.** All non-streamed, at fixed effort E (medium and xhigh), same
`max_tokens`, set through `X-Yamadori-Features`:

| arm | setting |
|---|---|
| A0 | everything off: the model as it ships |
| A1–A4 | one system on alone: retrieval / hints / deep thinking / fan-out |
| A5 | all on, fired by the selection engine |
| A6 | all forced on (today's `max`) |

A5 vs A6 is the `fixed` floor: selection must beat "always run everything".

**Tasks.**
- typegpu 0.12.5 — least contaminated.
- TypeScript, verified with `tsc --noEmit`.
- Rust→WASM across C ABIs, verified with `cargo test` / `wasm-bindgen` — needs
  source indexed first.
- three.js TSL: version-marker questions only, reported separately because it
  is contaminated.

**Graders:** execution, `bench/api_truth.py`, and the fact grader in
`bench/context_economy.py`.

**Preflight — the run aborts unless all pass:**
- `scripts/run_tests.py` offline suites.
- `test_live_stack --live` and `test_laya_head --serve`.
- `/heads` shows route_in loaded.
- `hints.npz` and package vectors have non-zero norms.
- Exactly one process on :1234.
- Runs go through `queue_runner.py`, one at a time.

Rows that end with `finish_reason != stop`, or with upstream errors, are
counted per arm as STACK ERRORS and excluded, never scored as model failures.

**Metrics:**
- Main: paired discordant pairs, exact McNemar per arm vs A0.
- Also: peak main-context window, wall clock, trigger rates, and precision of
  emitted hints.

**n needed:** at about 25% discordance, a 15-point gain needs about 85 paired
tasks; about 130 with Bonferroni correction across 5 comparisons; about 190
for a 10-point gain. That is roughly 150 GPU-hours, so run it in stages.

**Reuse:** `bench/livecodebench.py`, `bench/context_economy.py`,
`bench/mechanisms/selectors.py`, `bench/hint_collapse.py`,
`bench/recipe_oracle.py`, `scripts/eval_fanout.py`, `bench/quality.py`,
`tiers.from_header`.
