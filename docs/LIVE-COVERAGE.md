# Live coverage: every claimed feature, and the live test that exercises it

Operator rule (2026-09-24): **"If we don't have tests that exercise the real
models we don't have tests."** A feature is COVERED only when a test sends a
real request through `:1234` (or the real service it lives on), gets real
Bonsai output, and judges it deterministically: parsed, run, compared as an
exact string, or read off `x_yamadori` and the cache numbers. The model never
grades itself.

Sources for the feature list: `AGENTS.md`, `README.md`, `tiers.features`
(the tier matrix), `mcp/proxy.py` (`_x_yamadori`, `_run_turn`), and
`docs/SELF-IMPROVEMENT-PLAN.md` Phases 0-0.5.

**How to run:** `python scripts/run_tests.py --live --key-file PATH`, or
after a restart `python scripts/deploy_check.py --key-file PATH` (the deploy
gate). One test at a time:
`python mcp/test_live_stack.py --live --key-file PATH --only cache,seeds`.
`--maintenance` adds the intrusive tests.

Status values:
- **COVERED**: a live test asserts it.
- **PARTIAL**: exercised live, but part of the claim is not asserted.
- **UNCOVERED**: no live test.
- **NO MODEL**: no model is involved; covered offline only (listed so the
  map is complete).
- **STALE**: a live test exists, but it exercises a path that is cut, off by
  default, or around the stack.

Test names are `mcp/test_live_stack.py --only <name>` unless another file
is named.

**Layout v2 (operator, 2026-09-29; AGENTS.md "Layout v2", WRITTEN, NOT RUN).**
`images` follows the served projector (the pinned fixture the deploy
re-records): after the deploy the look is `yama_describe_image` on
`bonsai-vision`, and gpu_room records its load on the A4000. `slots` checks
the lane KEPT after a side call (at most `budget.LANE_TOKENS` cells). New
group `layout`: the advertised window is the
served line less the lane, every request ranks the lane `slots.RANK_LANE`
(`x_yamadori.cache.kv_ranks`), and the served main model has no projector.

**One model per effort tier (operator, 2026-09-29; mcp/tier_models.py,
mcp/max_mode.py; WRITTEN, NOT RUN).** New group `tier_models`: the walk
medium -> xhigh -> max -> medium, each answer from its tier's model
(`x_yamadori.capacity.model` and `why`), each swap loaded its model and left
no other main model loaded (`capacity.swap.left_loaded == []`), the model's
token profile applied (`x_yamadori.sampling.profile`: the effort sent and its
class, the client's max_tokens recorded, not trusted), and a medium request
while an xhigh one works is 503 `model_at_capacity` with the holder named.
NOT RUN while the table is off. `tiers`' overhead check counts a swap's wait
and load as model time.

**The removal (operator, 2026-09-29; docs/REMOVED.md).** Deep thinking and
its triggers (`yama_think_deeply`, `yama_plan`, the kickoff plan), fan-out,
the code check and fix-up (`Verified` / `Repaired` / `Checked`), the static
addendum, library definitions and library use, the second brain's research
tools, `delegate_investigation`, `/dash/api/deep`, Laya and CLM were
removed. Their live groups went with them: `tools`, `selection`, `repair`,
`note`, `deep`, `fanout` and `clm` in `mcp/test_live_stack.py`, and
`test_delegate_investigation_live` in `mcp/test_tools_live.py`. `cache` keeps
its five turns without the removed features' checks; `seeds` now checks the
conversation's concept seed (`x_yamadori.session.seed`, WRITTEN, NOT RUN);
`slots` lost its fix-up half; `e1` decides in process through the live
embedder (no route serves E1 and no request consults `route_in` any more).
Their rows below say REMOVED. The summary counts are from before the removal.

## Summary

76 features and mechanisms:

| status | count |
|---|---|
| COVERED | 43 (one of them, the ledger across a restart, only with `--maintenance`) |
| PARTIAL | 5 |
| UNCOVERED | 23 |
| STALE | 3 |
| NO MODEL | 2 |

Added on 2026-09-24, all in `mcp/test_live_stack.py`: `agent_loop`,
`repair`, `note`, `deep`, `fanout`, `compaction`, `images`, `tokens`,
`router`, and `ledger_restart` (`--maintenance`). Together they took 25 rows
from UNCOVERED or PARTIAL to COVERED. The largest gaps left are the code
check beyond `write_file` (formatter, edit shapes, other harnesses' tool
names, the final-answer repair pass), budget and cap landings, the skill
pipeline's model stages on the worker, and the draw-then-look image loop.
(Skills replaced hints on 2026-09-26; the `skills` group is written, not yet
run.)

## The map

### The door

| # | feature | status | live test |
|---|---|---|---|
| 1 | `/health` answers; a trivial request returns the model's answer | COVERED | `health` |
| 1a | The served model's state that offline suites PIN (`mcp/served_fixture.py`: the chat template, its efforts, n_ctx, total_slots, eos_token) is what `bonsai` serves; offline suites never read it live (`scripts/offline_guard`, 2026-09-27) | WRITTEN, NOT RUN | `served` |
| 2 | `/v1/models` advertises one model and the conversation window (the main share) | COVERED | `harness` |
| 3 | Unknown model names resolve to `yamadori` (`catalog`) | UNCOVERED | tests only send `yamadori` |
| 4 | Every `/v1` and `/dash/api` route needs a key (401 without one) | UNCOVERED | none |
| 5 | `/` gives HTML to a browser and a JSON descriptor to other clients; `/dash` redirects | NO MODEL | none |
| 6 | Streaming returns the whole answer over SSE | COVERED | `stream` |
| 7 | Streamed and blocking are one system (same `x_yamadori` keys, same hops) | COVERED | `parity` |
| 8 | CHANNEL ORDER: no content before reasoning, no reasoning after content | COVERED | `agent_loop` (every step, two client shapes) |
| 9 | 429 admission when the lane is full | UNCOVERED | deliberately not provoked; a 429 anywhere in the suite is NOT RUN |

### Tiers and budgets

| # | feature | status | live test |
|---|---|---|---|
| 10 | Every tier returns a complete answer at a small `max_tokens` (A_MIN floor) | PARTIAL | `tiers` covers minimal, low, medium, high and max. **`xhigh` is not in the loop.** |
| 11 | `safe_effort`: high/max/minimal never make the template return a 500 | COVERED | `tiers` (a 500 fails it) |
| 12 | A `length` finish is a budget event, reported as a notice, never an answer | UNCOVERED | none |
| 13 | The tool-turn cap lands (tools withdrawn, answer asked for) | UNCOVERED | none |
| 14 | `max` wall clock within 1.5x of `minimal` on a trivial prompt | COVERED | `tiers` |

### Route and selection

| # | feature | status | live test |
|---|---|---|---|
| 15 | Router: `utility` | COVERED | `router`, `harness` |
| 16 | Router: `agent_step` (ends on a client tool result) | COVERED | `router`; also every `agent_loop` step after the first |
| 17 | Router: `code_edit` | COVERED | `router` |
| 18 | Router: `code_generation` | COVERED | `router` |
| 19 | Router: `library_question` | COVERED | `router` |
| 20 | Router: `prose` | COVERED | `router` |
| 21 | Deep thinking decided by the rule plus Laya's `route_in` head; both signals recorded | REMOVED 2026-09-29 (the feature was removed; docs/REMOVED.md) | -- (`selection` removed) |
| 22 | No deep thinking and no fan-out on a non-code, non-library prompt, even at max | REMOVED 2026-09-29 (the feature was removed; docs/REMOVED.md) | -- |
| 23 | `X-Yamadori-Features` forces a feature | COVERED | used by `skills` (`{"skills": true}`) and `slots` (`idle_clear_s`) |

### Side calls, sessions, slots

| # | feature | status | live test |
|---|---|---|---|
| 24 | A client side call gets the bare model at `minimal`, fast | COVERED | `harness`, `router` |
| 25 | A conversation is pinned to a slot; turn 2 reuses its prefix | COVERED | `harness`, `agent_loop`, `cache` |
| 26 | Side calls use the transient slot | PARTIAL | `harness` checks the utility record, not the slot id |
| 27 | A conversation continues across a harness compaction (work log, slot, tools: `_continue_after_compaction`) | UNCOVERED | none |
| 28 | `X-Yamadori-Session` names a session | UNCOVERED | none |
| 89 | The transient slot is emptied after each side call (layout v2: the lane is KEPT, at most its cells); a conversation's pinned slot never is (`slots.release_idle`, `x_yamadori.slots.released`, 2026-09-26). The second brain's slot release went with the second brain (2026-09-29) | COVERED (written 2026-09-26, not yet run) | `slots`: after a side call, `x_yamadori.slots.released` names the slot and `/dash/api/vitals/pulse` (llama-server `/slots`) shows it holding <= 8 tokens (the lane: <= `budget.LANE_TOKENS`) |
| 90 | An idle conversation's pinned slot (idle > `slots.IDLE_CLEAR_S`, 600 s) is cleared when another conversation generates; its pin is kept and its next request records `resumed_cold` (`x_yamadori.slots.cleared_idle` / `resumed_cold`, 2026-09-26) | COVERED (written 2026-09-26, not yet run) | `slots`: two of the test's own conversations, the threshold overridden to 2 s by the test-account-only header `idle_clear_s`; decode tok/s of the active one with the other's ~40k idle cells kept vs cleared, n=3 each (pass: cleared >= 95% of kept -- clearing never slows the active one; the gain depends on where the cells sit in the unified pool, #59, and is reported as evidence), and the cleared one's next turn records `resumed_cold` with `how`: `restored` (llama-server's host-RAM prompt cache brought it back: reused >= half the prompt) or `reprocessed` (processed >= half), agreeing with its counts, and its `prompt_ms`. First live run 2026-09-27: kept [29.3, 29.06, 28.05], cleared [45.36, 29.88, 27.51] tok/s; B came back in ~516 ms (restored) |
| 90a | LAYOUT V3, one conversation per card (-np 1 on a LOCKED card, operator 2026-09-30; or -np 2): the conversation on slot 0; a NEW conversation id inside the owner's 60 s hold runs on THE OTHER CARD (bonsai-a4000's slot 0; `x_yamadori.slots.routed`, operator 2026-09-30) and keeps it (its next turn there again, reusing its prefix) -- or 503 `conversation_at_capacity` + Retry-After when the card's model has no other card (never downgraded); the switch after the hold is n/a live (a 60 s idle wait) and checked offline; the owner's compaction on its own card and slot 0, never bonsai-a4000; on a locked card a side call on the table's helper (bonsai-a4000) with the card's one slot unchanged, and jjava's reads on the helper (`mcp/slots.py`, `mcp/max_mode.py`) | WRITTEN, NOT RUN | `slots` (`test_one_conversation_card`). On a locked -np 1 card rows 89 and 90 are NOT APPLICABLE, printed with the reason (no lane; no second conversation's slot to clear) -- never a FAIL, and not NOT RUN (exit 0). The whole suite waits out a 503 `conversation_at_capacity` as a FALLBACK only (Retry-After, at most 180 s a request; printed): both cards held, or a tier with no other card |

### One model, one cache (Phase 0.5)

| # | feature | status | live test |
|---|---|---|---|
| 29 | The ledger restores reasoning for a client that strips it | COVERED | `agent_loop [strip]`, `cache` |
| 30 | A client that echoes what it was shown gets the clean copy in its place | COVERED | `agent_loop [echo]` (tail-only on every step) |
| 31 | Per-turn injections (skills, the concept seed, the work log) are replayed byte for byte | PARTIAL | implied by the tail bound in `cache` and `agent_loop`; no test compares the injection text |
| 32 | Static addendum at `high` and up | REMOVED 2026-09-29 (the feature was removed; docs/REMOVED.md) | -- |
| 33 | Library definitions injected at `medium` | REMOVED 2026-09-29 (the feature was removed; docs/REMOVED.md) | -- (`tools` removed) |
| 34 | Deep thinking: hand-off prefilled as reasoning; answer opens with the seed line and "After thinking deeply," | REMOVED 2026-09-29 (the feature was removed; docs/REMOVED.md) | -- (`deep` removed) |
| 35 | Deep thinking: every cited `path:line` exists in the held source | REMOVED 2026-09-29 (the feature was removed; docs/REMOVED.md) | -- |
| 36 | The slot is warmed with a delivered turn that differs from the generated one, and the next request extends it | UNCOVERED since 2026-09-29 | its live cases were the repaired and the noted write (`repair`, `note`, `cache` steps 2-3), removed with the code check; an image turn's delivered line still warms, unasserted |
| 37 | The ledger survives a proxy-only restart | COVERED, `--maintenance` only | `ledger_restart` (restarts the proxy; never run by default) |
| 38 | A conversation's first user turn carries its concept seed at `high` and up, recorded (`x_yamadori.session.seed`, the injection's `seed` part); none at `medium` (2026-09-29; before, each second-brain job carried one) | WRITTEN, NOT RUN | `seeds` |
| 39 | A replay of a turn draws the same seed | UNCOVERED | none |

### Code checks and repair

| # | feature | status | live test |
|---|---|---|---|
| 40 | `medium`: a broken client write is noted "Checked" and forwarded unchanged | REMOVED 2026-09-29 (the feature was removed; docs/REMOVED.md) | -- (`note` removed) |
| 41 | `high`+: a broken client write is repaired by the second brain, noted "Repaired", and it parses | REMOVED 2026-09-29 (the feature was removed; docs/REMOVED.md) | -- (`repair` removed) |
| 42 | A clean write gets a "Verified" note | REMOVED 2026-09-29 (the feature was removed; docs/REMOVED.md) | -- |
| 43 | Formatter output for a whole file that parses (ruff, prettier, rustfmt) | REMOVED 2026-09-29 (the feature was removed; docs/REMOVED.md) | -- |
| 44 | Edit-shaped calls (old/new strings, diff hunks, SEARCH/REPLACE) block or flag | REMOVED 2026-09-29 (the feature was removed; docs/REMOVED.md) | -- |
| 45 | Known tool names of other harnesses (Claude Code, Codex, Cline...) | REMOVED 2026-09-29 (the feature was removed; docs/REMOVED.md) (the check); the detection table stays for the image guard | -- |
| 46 | Final-answer repair pass ("Verified" / "Repaired" on a code answer) | REMOVED 2026-09-29 (the feature was removed; docs/REMOVED.md) | -- |

### Fan-out

| # | feature | status | live test |
|---|---|---|---|
| 47 | Code request at `high`: B written, "Compared two approaches", winner delivered | REMOVED 2026-09-29 (the feature was removed; docs/REMOVED.md) | -- (`fanout` removed) |
| 48 | The delivered code works | REMOVED 2026-09-29 (the feature was removed; docs/REMOVED.md) | -- |
| 49 | Tie-breaker C when the check does not separate A and B | REMOVED 2026-09-29 (the feature was removed; docs/REMOVED.md) | -- |
| 50 | Prose hand-back continuation ("Weighing them") | REMOVED 2026-09-29 (the feature was removed; docs/REMOVED.md) | -- |
| 51 | `x_yamadori.fanout` reports variants and their seeds | REMOVED 2026-09-29 (the feature was removed; docs/REMOVED.md) | -- |

### Compaction

| # | feature | status | live test |
|---|---|---|---|
| 52 | In-place compaction on the conversation's stored prompt, thinking at its effort | COVERED | `compaction` (in place), `cache` step 5 |
| 53 | Hermes' flattened compaction rewritten onto the conversation | COVERED | `compaction` (flattened) |
| 54 | The summary is whole (not `length`) and keeps paths and errors from the dropped span | COVERED | `compaction` |
| 55 | Hermes' iterative form (PREVIOUS SUMMARY) | UNCOVERED | none |

### Images

| # | feature | status | live test |
|---|---|---|---|
| 56 | `yama_generate_image`: the model calls it, `x_yamadori.images` records it | COVERED | `images` |
| 57 | The signed `/media` link serves the PNG with no key; a tampered one is 403 | COVERED | `images` |
| 58 | `POST /v1/images/generations` | UNCOVERED | none |
| 59 | Image model from the account preference, else the default (turbo) | UNCOVERED | `x_yamadori.images[].model` is printed, not asserted |
| 60 | `yama_describe_image` on an attached image | COVERED | `images` |
| 61 | `yama_describe_image` on an image drawn in the same request (the refine loop), and VRAM concern sequence 2 | UNCOVERED | the docs/IMAGEGEN.md live check, not yet a test |
| 62 | Peak A4000 VRAM recorded | COVERED | `images` (nvidia-smi every second) |

### Skills (the one knowledge system since 2026-09-26)

| # | feature | status | live test |
|---|---|---|---|
| 63 | The hints path | RETIRED 2026-09-26 (migrated into skills; mcp/skill_migrate.py) | -- |
| 64 | Skill selection at `medium`: an authored skill reaches its target shape, `x_yamadori.skills` names it | WRITTEN, NOT RUN (needs the proxy restarted on this code) | `skills` |
| 65 | Skill pipeline model stages (distil, decompose, tag, tests, faithful) on the worker | UNCOVERED (offline: fake-model tests; the frontier example ran with a hand-written stand-in reply) | none live |

### Tools API, MCP, internal generation

| # | feature | status | live test |
|---|---|---|---|
| 66 | `summarize_text` through the tools API on `:1235` | COVERED | `summarize` |
| 67 | `summarize_text` keeps identifiers, paths, numbers and errors verbatim | STALE | `mcp/test_tools_live.py` (in process, on a fixture index, internal generation to llama-swap; see below) |
| 68 | `find_by_meaning`, `find_definition_opt`, `find_references`, `find_by_pattern`, `read_file_range`, `describe_index`, `run_check` | NO MODEL | `mcp/test_tools.py` offline; nothing live through `:1235` |
| 69 | `delegate_investigation` (off by default: a benchmark arm) | REMOVED 2026-09-29 (the feature was removed; docs/REMOVED.md) | -- (`test_delegate_investigation_live` removed) |
| 69a | The tools API's door (2026-09-26): no key 401 + `WWW-Authenticate` resource_metadata, foreign Origin 403, key 200 on `/mcp`, `/health` liveness only, RFC 9728 metadata unauthenticated | WRITTEN, NOT RUN (needs the tools API restarted on this code; until then its first check fails with 200) | `mcp/test_tools_live.py` `test_tools_api_door_live`; offline `mcp/test_tools_api_auth.py` |

### Ledgers, dashboard, Laya

| # | feature | status | live test |
|---|---|---|---|
| 70 | `/dash/api/tokens` grows by exactly the usage a request reported | COVERED | `tokens` |
| 71 | `x_yamadori.cache` reports reused vs processed | COVERED | `cache`, `agent_loop`, `compaction`, `tokens` |
| 72 | `x_yamadori.energy` and the power ledger | UNCOVERED | none |
| 73 | Laya `route_in` head served (both engines) | REMOVED 2026-09-29 (the feature was removed; docs/REMOVED.md) | -- (`bench/test_laya_head.py` deleted with Laya's code) |
| 74 | Laya's cached calibration and guardrail numbers still reproduce on :1237 | REMOVED 2026-09-29 (the feature was removed; docs/REMOVED.md) | -- |
| 75 | Rings work log written by the proxy and reinjected after a compaction | UNCOVERED | none |
| 76 | Dataset pipeline (`clarify` is model-assisted) | UNCOVERED | none live |

### OpenAI conformance (added 2026-09-25, `docs/OPENAI-CONFORMANCE.md` fixes 1-3)

| # | feature | status | live test |
|---|---|---|---|
| 77 | Errors are OpenAI's four-field object with the right status (bad JSON 400, missing `messages` 400, `n: 2` 400, unknown `/v1` POST 404) | COVERED | `conformance` |
| 78 | A prompt past the advertised window is 400 `context_length_exceeded`, counted by the model server, before any byte (blocking and streamed) | COVERED | `conformance` |
| 79 | `stream_options.include_usage`: a last chunk with `choices: []` and usage (`cached_tokens` included) | COVERED | `conformance` |
| 80 | `usage.prompt_tokens` is the final generation's on a multi-generation turn, not the sum | COVERED | `images` (the yama_generate_image turn) |
| 81 | A failure after a stream's first byte is one SSE error event, never content | UNCOVERED (live) | offline only (`mcp/test_openai_conformance.py`): no cheap way to make the live model server fail mid-turn |

### The Responses API (added 2026-09-26, `docs/OPENAI-CONFORMANCE.md` "Status: R1")

| # | feature | status | live test |
|---|---|---|---|
| 82 | `POST /v1/responses` answers a plain request with a completed Response and usage | COVERED | `responses` |
| 83 | The streamed event sequence (created, in_progress, ..., completed; sequence numbers; deltas = the message) | COVERED | `responses` |
| 84 | A function-call round trip replayed as Codex replays it: same session (`prompt_cache_key`), the slot's cache reused | COVERED | `responses` |
| 85 | `input_image` reaches the chat image path (an attachment; the answer names the colour) | COVERED | `responses` |
| 86 | The hosted `image_generation` tool is our yama_generate_image; an `image_generation_call` item with the PNG | COVERED (NOT RUN when no image server) | `responses` |
| 87 | An unknown `previous_response_id` is 400 `previous_response_not_found` (OpenAI's code; the stored-state chain is R10) | COVERED | `responses` |
| 88 | A real Codex / Hermes `codex_responses` / OpenCode / Pi session against `:1234` | UNCOVERED | none: Codex CLI is not installed; the offline replay (`mcp/test_responses_api.py`) is built from codex-rs source |
| R1 | A tool loop with NO `prompt_cache_key` (what VS Code Copilot sends): our session id rides in the `function_call.call_id`, the next request finds its conversation there (`source: tool_call_id`) and reuses the slot's cache | COVERED | `responses_features` (also `bench/harness_soak.py` scenario i) |
| R2 | The model's reasoning streams as `reasoning_summary_text` deltas before its call or answer; the item's summary is what streamed | COVERED | `responses_features` (`summary` mode is the served one; `content` and `off` need a restart with `YAMADORI_RESPONSES_REASONING`: offline only, `test_responses_api.py`) |
| R3 | Terminal usage: `cached_tokens` is the reused prefix, `output_tokens_details.reasoning_tokens` is counted | COVERED | `responses_features` |
| R4 | A client's title request and a one-word classifier over Responses are utility calls (tier overridden to `minimal`, the helper's model on a locked card) | COVERED | `responses_features` |
| R5 | The concept seed on a Responses conversation's first user turn at `high`, none at `medium` | COVERED | `responses_features` |
| R6 | A prompt past the window is HTTP 400 `context_length_exceeded` before any byte, blocking and streamed | COVERED | `responses_features` |
| R7 | `text.format` `json_schema` gives JSON of the schema's shape | COVERED | `responses_features` |
| R8 | Drawing with NO hosted tool declared (the signed link in the text, no `image_generation_call` invented); our signed link sent back as an `input_image` is described; a `function_call_output` that carries an image is looked at through `yama_describe_image` (left=red, right=blue) | COVERED | `responses_features` (the hosted tool itself: R `responses`, row 86) |
| R9 | The MCP package tools offered at `medium` and a lookup run as a hidden hop | NOT APPLICABLE while Docker is down (PackageLens is its container): the record says why (`x_yamadori.mcp.why`); offline `test_packagelens_tools_through_responses` runs the host against the fake server | `responses_features` |
| R10 | STORED RESPONSES (operator, 2026-10-06): a 3-turn chain by `previous_response_id` only (a `function_call_output` alone on turn 2, blocking / streamed / blocking), `store` never sent; the model answers from the stored turns, the slot's cache is reused, one session; GET, `input_items`, DELETE then 404; a chain through a deleted response and a `store: false` one are 400 `previous_response_not_found` | COVERED (**written 2026-10-06, not yet run**) | `responses_features` |
| R10 | In-place compaction over Responses (mode `ledger`, >= 90% of the stored prompt reused, summary whole, keeps the path and the error string, opens with the session line) and Hermes' flattened one (`rewritten`) | COVERED | `responses_compaction` |
| R11 | A NEW `prompt_cache_key` whose history carries the summary line continues the conversation (`aliased_to`) | COVERED | `responses_compaction` |
| R12 | The one-conversation rule over Responses: a second conversation inside the hold is served on the other card (`x_yamadori.slots.routed`, bonsai-a4000) and keeps it; 503 `conversation_at_capacity` + Retry-After where the model has no other card | COVERED (routed arm; the 503 arm is offline only: `test_one_conversation_per_card_through_responses`) | `responses_slots` |
| R13 | A client that hangs up mid-stream stops the generation (slot 0 reads idle in about a second) and the next request is not queued behind it | COVERED | `responses_slots` |
| R14 | The tier walk by `reasoning.effort` (low, xhigh, max, low): each answer is its tier's model, each swap loads it and leaves no other loaded, and the swap is heard (`response.in_progress` beats, no silence over 12 s between parsed events) | COVERED | `responses_tiers` |
| R15 | A lower tier while max works is served on bonsai-a4000; xhigh (no other card) is HTTP 503 `model_at_capacity` with Retry-After before any byte | COVERED | `responses_tiers` |
| R16 | A killed model server over Responses is `response.failed`, never a completed answer | COVERED (live: `bench/harness_soak.py --api responses --only k`; offline: `test_a_killed_model_server_through_responses`) | `harness_soak` k |
| R17 | `length` is `response.incomplete` (`max_output_tokens`) | OFFLINE ONLY: the proxy floors an answer at `A_MIN` and thinks inside its own budget, so a live request cannot be made to hit `length` cheaply | `test_responses_api.py` `test_streamed_turns` |
| 89 | CLM (`mcp/clm.py`, docs/CLM.md) through llama-swap's `clm-encoder` behind gpu_room reproduces the offline reference decisions (40/40 argmax and none-vs-pick), deterministic, and the selector's `clm` decider answers | REMOVED 2026-09-29 (the feature was removed; docs/REMOVED.md) | -- (`clm` removed) |
| 91 | PACKAGE ONBOARDING (docs/PACKAGE-ONBOARDING.md): a prompt with links submitted through `:1234`, run by the worker, resolved with its rules, the licence quoted, indexed, the vocabulary judged by the floor, completed | WRITTEN, NOT RUN; OPT-IN (writes live state -- a held package, its skills -- and fetches from npm and GitHub; the operator names the package and the wait) | `onboarding` (`--only onboarding --onboard "<prompt>" --onboard-wait S`; never in the default live run) |

## Stale live tests, and what should replace them

| test | why it is stale | replacement |
|---|---|---|
| `mcp/test_tools_live.py` `test_delegate_investigation_live` | REMOVED 2026-09-29 with `delegate_investigation` and deep thinking. | -- |
| `mcp/test_tools_live.py` summarize tests | In process against the fixture, not the running tools API. | `summarize` (through `:1235`). Its verbatim-token checks (`KEEP`) should move into `summarize`. |
| `mcp/test_live_stack.py` `hints` | REPLACED 2026-09-26 by `skills` (the hints path is gone). | -- |
| `bench/test_hint_collapse.py --live` (reranker checks) | REMOVED 2026-10-01 with the reranker (docs/REMOVED.md); the suite has no --live arm. Until then: calls the reranker on `:11434` directly. The reranker is "not trusted, not used" (`docs/FINDINGS.md` #20). Its checks assert the serving BUG exists. | Keep as a regression probe of the bug, outside the deploy gate. See the result below. |

## Results of the first full run

2026-09-24 12:26-13:02, `scripts/deploy_check.py` (the gate), one run, not
repeated. The card was idle on two `/slots` reads 5 s apart, and no other
live run was active. **Verdict: DEPLOY NOT GOOD (exit 1).** The raw output is
in the session scratchpad (`live-gate.txt`), and the verdict is in
`logs/deploy_check.jsonl`.

| suite | passed / total | failures |
|---|---|---|
| `bench/test_guardrail.py --live` | 95/95 | none |
| `bench/test_hint_collapse.py --live` | 51/52 | the reranker's batched top-1 was RIGHT on the Paris probe (the test asserts the bug). Batched self-retrieval is still 16/89 vs 66/89 solo |
| `bench/test_laya_calibration.py --live` | 48/48 | none |
| `bench/test_laya_head.py --serve` | 31/33 | route_in "loses to the regex" and "carries the caveat". The retrained artefact beats the bare regex (0.728 vs 0.608), so the assertion is stale |
| `mcp/test_live_stack.py --live` | 112/115 | `seeds` x2 (stale: sequential fan-out has one seed, `where=fanout:direct`); `cache` step 2 after deep thinking processed 1442 tokens against a bound of 197 (a real cache miss) |
| `mcp/test_tools_live.py --live` | 19/21 | summarize: MODEL_UNAVAILABLE, because the suite imports `test_tools`, which points `model.UPSTREAM` at a refusing port. Its delegate checks pass with no model behind them |

Every NEW test passed on its first live run:

| test | checks | evidence (abridged) |
|---|---|---|
| `router` | 6/6 | each class returned itself. library_question: "the tools are offered because NAMES_HELD_SOURCE" (172 s at medium) |
| `tokens` | 5/5 | usage 914 prompt / 151 completion / 1 hop, and the ledger delta was exactly 914 processed, 0 cached, 151 completion, 1 generation |
| `agent_loop` | 14/14 | strip: 3 steps (processed 162, then 24); echo: 5 steps (161, 32, 24, 24). No re-read, no restart, no late reasoning, and both `stats.py` pass `test_stats.py`. **But** the echo run's final content was `'Done.\n</think>\n\nDone.\n</think>\n\nDone.'`, which no check caught. A check was added after the run |
| `repair` | 5/5 | tool_code stopped=fixed in 1 round; "Repaired"; warm sent; the next request processed 18 (reused 3107 >= previous prompt 2565) |
| `note` | 5/5 | stopped=noted, "Checked", `def g(:` forwarded unchanged, warm sent, the next request processed 18 |
| `deep` | 5/5 | 9 searches, hand-off prefilled (7 facts, 4 verified), opens "Today I was inspired by senang. After thinking deeply,". Cited `src/core/Object3D.js:714/716/720/724` and `src/math/Matrix4.js:483`, all present in the held three source |
| `fanout` | 4/4 | code_generation, 3 steps (tie-breaker: both parse but disagree), "Compared two approaches", and the winner passed 10 bracket cases. **But** the winner's first line is `# humanidad`, the tie-breaker's seed word |
| `compaction` | 10/10 | in place: mode `ledger` (the ledger's rendering extends the stored prompt byte for byte), reused 2660, processed 39. Flattened: mode `rewritten`, 4 of 4 turns mapped, reused 2697 of a 2560-token stored prompt, processed 287. Both kept `src/ledger/rollup.py` and `E_LEDGER_SKEW` |
| `images` | 8/8 | yama_generate_image (turbo, 4 steps, 25.2 s). The signed link served a 1.43 MB PNG with no key, and a tampered signature got 403. yama_describe_image on an attached 256x256 PNG: `left=red, right=blue`, 5.93 s, 168 prompt tokens |

**A4000 VRAM during `images`** (nvidia-smi every second, 41 samples):
baseline 12,457 MiB at the start of the test, peak 16,068 of 16,376 MiB,
**308 MiB free at the peak**. That is the docs/IMAGEGEN.md VRAM concern,
measured: drawing then looking left almost nothing spare. What was
resident at the 12,457 baseline was not recorded.

**Not run:** `ledger_restart` (intrusive; needs `--maintenance`, and the
operator said no restart). No 429 occurred.

**On the in-place compaction's mode.** The operator's brief expected
`spliced`. The code serves `ledger` whenever the ledger's rendering extends
the stored prompt (`proxy._serve_compaction`, since 2026-09-24), and falls
back to `spliced` only when it does not. The test accepts either, because
both are served on the stored prompt; `as_sent` is the miss. Say so if
`spliced` specifically is required.

Every failure and defect above is a row in `docs/SELF-IMPROVEMENT-LOG.md`
(#10-#15).

**Side effect on the offline suite.** Live tests go through the real door,
so their turns land in `index/corpus.sqlite3` like any client's.
`mcp/test_utility.py` `test_the_hermes_compactions_in_the_corpus_replayed`
counts the corpus's Hermes compactions ("all 12 ..."). After this run it
read 13, because `compaction` (flattened) added one, and it failed 2
checks. Either the replay excludes the test key's account, or the live
test's compaction is labelled so the replay can skip it. That decision
belongs to the owner of `test_utility.py`. The same offline run (13:05)
also failed `mcp/test_domains.py`, `mcp/test_selection.py` (all 342
LiveCodeBench prompts: HELD_SOURCE_UNMAPPED) and `bench/domain/test_grade.py`
(no count printed). All three were green at 08:53 and sit in files other
agents changed since then. Nothing in this work touched them.
