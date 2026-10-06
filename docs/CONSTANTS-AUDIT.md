# Constants audit (2026-09-27)

On 2026-09-27 the operator said the "CHOICE, unmeasured" numbers and rules
in this stack "are probably all fabricated hallucinations you put together
to boost the performance on this task ... remove that part ... start doing
honest work." At 13:31 he said of MAX_FINDING_CHARS: "You made that limit
up, unfounded."

This file is the inventory that followed. It lists every tunable number,
threshold, cap, window, timeout, rate limit and hand-written behavioural rule
or injected text in `mcp/`, plus the bench/octopus run-time settings the
stack reads. Each one is classified, and the file says what was done about
it.

## The rule

A number or rule may exist only if it is one of these:

| class | meaning | action |
|---|---|---|
| **A** | **OPERATOR DECISION**: the operator's own words, with a date, about *that* thing | keep; cite |
| **A-flag** | the operator set the direction, but the number or wording is ours; or a code comment claims an operator decision and no matching operator message exists | keep; **the operator should confirm the value** |
| **B** | **DERIVED FROM A REAL CONSTRAINT**: VRAM, the KV pool or context window, an engine's, protocol's or harness's own limit, security, correctness | keep; the derivation is written beside it |
| **B-value-arbitrary** | the *existence* of a bound is forced (a network timeout must exist, a tool result must fit a window), but the *value* was picked | keep; derive the value |
| **C** | **MEASURED**: a script and an n in this repo support it. C rows note "in-sample" where rules were tuned on the set that scores them | keep; cite |
| **D** | **INVENTED**: none of the above; usually picked to steer one benchmark run (Octopus, pagoda) or because it "felt right" | **remove**, and let the underlying behaviour stand: the engine's or harness's default, no cap, no nudge, no injected text. If the feature cannot exist without the number, remove the feature |

**Sources for A.** The operator's words come from two places:

1. Every message the operator typed in the session transcript, 367 of them:
   `~/.claude/projects/C--Users-jwals-llama-stack/<session>.jsonl`, user
   messages.
2. The operator's AskUserQuestion answers in the same transcript, 9 of them
   ("Your questions have been answered: ...").

Compaction summaries that *quote* the operator count only as "summary-quoted".
Memory files, AGENTS.md and docs/SELF-IMPROVEMENT-LOG.md count only where
they quote him. A comment reading "(operator, DATE)" beside a number counts
only if one of those sources holds the decision about that thing.

**Method.** Seven read-only inventory passes (proxy.py; the second brain;
budgets and slots; fan-out and routing; skills and CLM; web, images and
server; infrastructure and bench/octopus) each read their files in full.
They classified every item against the sources above, with the evidence
quoted and the file and line given. The consolidation checked the
consequential calls against both operator sources.

Line numbers are as of reading on 2026-09-27. proxy.py, shomen.py, deep.py,
skill_* and clm.py were being edited by other agents at the time, so some
lines have moved.

## Counts

Main class per row. Composite labels are counted under their first class:
"A + B" as A, "C (in-sample)" as C.

| group | A | A-flag | B | B-value-arbitrary | C | D | total |
|---|---|---|---|---|---|---|---|
| mcp/proxy.py | 16 | 23 | 16 | 30 | 0 | 22 | 108 |
| Second brain and triggers | 12 | 36 | 22 | 19 | 2 | 31 | 122 |
| Budgets, slots, ledger | 34 | 12 | 52 | 43 | 7 | 15 | 163 |
| Fan-out, repair, routing | 9 | 12 | 48 | 30 | 16 | 51 | 166 |
| Skills and CLM | 5 | 23 | 24 | 39 | 3 | 91 | 185 |
| Web, images, vision, server | 9 | 6 | 39 | 65 | 9 | 34 | 162 |
| Infrastructure and bench/octopus | 9 | 6 | 26 | 27 | 5 | 28 | 101 |
| **all** | **94** | **118** | **227** | **253** | **42** | **272** | **1007** |

**The counts are not a quality score.**
- Most B-value-arbitrary rows are timeouts, output bounds and byte caps. The
  bound itself is required; only the number is ours.
- The 272 D rows are the removal list. 39 were done in the first pass
  (below). The second pass (2026-09-27, "Removals done, second pass")
  finished the proxy.py, shomen.py and deep.py owners' lists and three
  unassigned rows: 44 more (83 in all), 2 LEFT with the reason written
  (proxy's `answer record` and `progress.learn`). The skills owner's 88
  were applied the same day: 60 DONE, 21 KEPT (each with the check its
  removal fails), 7 for the OPERATOR (prompt wording and taxonomy axes;
  "skills owner" below). Pending: clm.py's 1 and the 98 unassigned rows
  left, the operator's call.
  The pending ones are grouped by owner below, with exact instructions.
- Nothing in proxy.py reaches C. The only measurements behind it (the STEP 0
  cache probes, the 7/40 stray-marker replay) live in the scratchpad, not
  the repo.
- The skill selector's numbers are D or C-in-sample: they were tuned on
  `bench/skills/test_daily_eval.py`'s own rows.

## Removals done (2026-09-27)

Every change below was made and its suites re-run. The results:

| suite | result |
|---|---|
| test_deep | 182/182 |
| test_stream | 241/241 |
| test_ledger | 199/199 |
| test_tools | 478/478 |
| test_tiers | 178/178 |
| test_fanout | 96/96 |
| test_fanout_delivery | 31/31 |
| test_rings | 38/38 |
| test_repeats | 12/12 |
| test_web_access | 62/62 |
| test_images (run_tests) | 228/228 |
| test_image_emit | 69/69 |
| test_vision (run_tests) | 273/273 |
| test_package_index | 72/72 |
| test_progress | 30/30 |
| test_e1 | 54/54 |
| test_selection | 78/78 |
| test_deep_review | 53/53 |
| test_image_guard | 67/67 |
| test_tool_code | 128/128 |
| test_utility | 195/195 |
| test_harness_forms | 36/36 |
| test_session | 28/28 |
| test_sessions | 49/49 |
| test_slots | 85/85 |
| ruff E9,F | clean |

bench/octopus/test_grade_checks.py is at 138/139. The one failure is the
pagoda prompt byte-for-byte check against bench/voxel, which none of these
changes touch.

**How the colliding files were handled.** Where a D item lives in a file
another agent was editing (proxy.py, shomen.py), or is read from there, it
was switched OFF through `tiers.OFF_BY_DEFAULT`, a one-line hunk in
tiers.py. That lets the underlying behaviour stand now. Deleting the dead
code is left to the file's owner (see "Pending", below).

- `tiers.OFF_BY_DEFAULT` held `step_thinking`, `plan_tools`,
  `plan_budget` and `fixup_project_only`, beside the existing
  `deep_tool_hop`. In the second pass the four were DELETED with the code
  they switched; `OFF_BY_DEFAULT` is `{deep_tool_hop}` again.

- `turn.guidance appended to tool results` (mcp/proxy.py:1230,1405,1473-1480): DONE 2026-09-27: the three proxy call sites deleted, and repeats.Turn.guidance itself.
- `by-result step caps (call site)` (mcp/proxy.py:2980-2984): DONE 2026-09-27: branch deleted: every agent step thinks at most AGENT_STEP_THINKING. tiers.AGENT_STEP_THINKING_READ / _ERROR, tiers.step_thinking(), the step_thinking switch, progress.step_kind and SEARCH_NAMES deleted. test_ledger [step], test_progress updated.
- `fix-up scope project-only (call site)` (mcp/proxy.py:7296-7300): DONE 2026-09-27: call site, tool_code.scope, its note branch and scope_skipped deleted; switch fixup_project_only gone. test_ledger [fixup] rewritten. progress._TEMP / SCRATCH_NAMES / dot-file rule stay: label rule 2 reads them (unassigned).
- `QUESTION_CHARS` (mcp/deep.py:150 (used 1136,1155,1324,1382,1395)): DONE 2026-09-27: REMOVED: the whole spec/question goes to the second brain (test_deep test_a_long_spec_reaches_the_plan_job_whole: a 20k spec whole). E1's input keeps its training cut as E1_QUESTION_CHARS 2000 (B: bench/e1/build_heldout2.py:193).
- `CONTEXT_CHARS + task[:1500]` (mcp/deep.py:151,1112,1141,1382,1395): DONE 2026-09-27: REMOVED from the second brain's input (ctx, task[:1500] in struggle/area/route questions). E1 keeps E1_CONTEXT_CHARS 1500 (B: the label rendering, build_heldout2.py:194). shomen.py:1252 context[:1500] PENDING (shomen.py owner).
- `struggle question excerpts` (mcp/deep.py:1116-1136): DONE 2026-09-27: REMOVED: every failing output of the struggle's span, whole; every event, path and phrase.
- `MAX_FINDING_CHARS` (mcp/shomen.py:185,2060-2069): DONE by the shomen.py owner (verified absent 2026-09-27; operator 13:31 'You made that limit up, unfounded.').
- `PLAN_TOOL_SITUATIONS / plan_tools gate` (mcp/shomen.py:419-462): DONE 2026-09-27: DELETED: the plan gets our tools, less run_check with no repository (B). Switch gone from tiers.
- `plan time/turn cap` (mcp/shomen.py:1262-1304 (tiers.py:549-550)): DONE 2026-09-27: DELETED (shomen branch, tiers.PLAN_TOOL_TURNS / PLAN_SECONDS, switch plan_budget).
- `MAX_ITEMS` (mcp/shomen.py:1646,2046-2057): DONE by the shomen.py owner (verified absent 2026-09-27).
- `PLAN_MAX_ITEMS` (mcp/shomen.py:2387,2453-2467): DONE by the shomen.py owner (verified absent 2026-09-27).
- `AGENT_STEP_THINKING_READ` (mcp/tiers.py:498): DONE 2026-09-27: DELETED with its call site.
- `AGENT_STEP_THINKING_ERROR` (mcp/tiers.py:500): DONE 2026-09-27: DELETED with its call site.
- `step_thinking() by result kind` (mcp/tiers.py:504): DONE 2026-09-27: DELETED with its call site and the switch.
- `JOB_THINKING plan` (mcp/tiers.py:539): DONE 2026-09-27: REVERTED to 12,288 (the 1.5x value, A-flag). test_deep updated.
- `PLAN_TOOL_TURNS` (mcp/tiers.py:549): DONE 2026-09-27: DELETED with shomen's budget branch.
- `PLAN_SECONDS` (mcp/tiers.py:550): DONE 2026-09-27: DELETED with shomen's budget branch.
- `BEHAVIOURS: fixup_project_only` (mcp/tiers.py:1090): DONE 2026-09-27: DELETED (switch and code).
- `BEHAVIOURS: plan_tools / plan_prompt` (mcp/tiers.py:1093): DONE 2026-09-27: plan_tools DELETED (switch and code); plan_prompt kept, its style sentences removed.
- `MIN_MAPPED` (mcp/compaction.py:87): DONE 2026-09-27: REMOVED: every record must map (1.0); otherwise the compaction goes up as sent. test_utility 195/195, test_harness_forms 36/36.
- `AGREE` (mcp/fanout.py:340): DONE 2026-09-27: REMOVED: both parsing -> the tie-breaker C runs (the operator's design: C when the check does not separate A and B). test_fanout, test_fanout_delivery updated.
- `dissent_note threshold` (mcp/fanout.py:1337): DONE 2026-09-27: fanout.dissent_note DELETED and proxy's call; _fan_out returns (record, winner). test_fanout updated.
- `dissent_note text` (mcp/fanout.py:1340): DONE 2026-09-27: DELETED with the function.
- `ALT_MAX_POINTS` (mcp/fanout.py:1369): DONE 2026-09-27: REMOVED: every differing point crosses. test_fanout updated.
- `ALT_MAX_CHARS` (mcp/fanout.py:1370): DONE 2026-09-27: REMOVED (same class as MAX_FINDING_CHARS).
- `CODE_NOTE_NAMES` (mcp/fanout.py:1371): DONE 2026-09-27: REMOVED: every differing name is listed.
- `FIX-UP SCOPE (scope)` (mcp/tool_code.py:773): DONE 2026-09-27: DELETED (tool_code.scope and its call).
- `guidance trigger` (mcp/repeats.py:102): DONE 2026-09-27: Turn.guidance DELETED with proxy's three calls.
- `guidance text` (mcp/repeats.py:105): DONE 2026-09-27: REMOVED.
- `_forward_hint texts` (mcp/repeats.py:115): DONE 2026-09-27: REMOVED (the function is gone).
- `closing line` (mcp/rings.py:147): DONE 2026-09-27: REMOVED ('Work listed above is DONE. Do not redo it.'). test_rings updated.
- `empty-log text` (mcp/rings.py:117): DONE 2026-09-27: REMOVED the 'record it now' instruction; the situation stays.
- `CHECKS RUN header` (mcp/rings.py:128): DONE 2026-09-27: REMOVED the claim; plain heading 'CHECKS RUN:'.
- `HOST_BACKOFF_MAX_S` (mcp/research_tools.py:100): DONE 2026-09-27: REMOVED: a server's Retry-After is honoured as sent (RFC 9110; floor 1 s). test_web_access 62/62.
- `safesearch=0` (mcp/research_tools.py:1228): DONE 2026-09-27: REMOVED: SearXNG's own default applies.
- `generate_image TOOL description` (mcp/images.py:755-799): DONE 2026-09-27: the stale timing claims ('about 2 minutes', '3-4 minutes') removed; the rest of the description PENDING (A-flag wording).
- `VISION_LOADING remedy text` (mcp/vision.py:454): DONE 2026-09-27: FIXED: the false '15 minutes (ttl 900)' claim removed (config.yaml bonsai-vision ttl is 300).
- `package fallback text caps` (mcp/packages.py:85,334): DONE 2026-09-27: REMOVED the silent [:2500]/[:3000]; the tool loop's marked breaker (repeats.cap_tool_result) bounds it.
- `--max-turns` (bench/octopus/run.py:638): DONE 2026-09-27: REMOVED the 1000 default: Hermes' own default applies unless --max-turns is given.

## Removals done, second pass (2026-09-27)

The proxy.py, shomen.py and deep.py owners' lists (below, each line marked),
the dead call sites the first pass's switches left, `COMPACTION_THINKING`
and the nebari truncation bug. Per item, what replaces it was checked first:

- **Replacements, not bare removals.** A refused `yama_think_deeply` call
  still gets a structured `ALREADY_THOUGHT` result (situation, retryable,
  remedy with an owner); only the steering went. The library-use caps gave
  way to one B bound, the room the window leaves (`proxy._injection_room`).
  Compaction thinking is what its own window leaves (the one budget rule).
  A `length` finish is `finish_reason: "length"` and nothing else.
- **Correctness paths checked.** Removing the stray-marker-adjacent filters
  (`_ToolMarkup`, `_ImitatedNotes`, `dedup_opening`) makes the client's copy
  closer to the slot's text, so the ledger's #10 rendering is unaffected
  (test_ledger 198/198, test_stray_markers 138/138). Our tool note still
  bypasses the marker filter.
- **LEFT, with the reason on its line:** the `answer record` (removing it
  breaks "one model, one cache" for every client without a key or header)
  and `progress.learn` / project inference (label rule 2 reads it; it goes
  with `helped_needs_change`, the operator's call).
- **Deviations from the written instruction, each on its line:**
  `ECHO_WINDOW` (a number-free set of observed habits instead of either
  option the row offered: both break a tested path) and
  `DEFER_REQUESTS` (the standard helper-lane wait instead of
  `lane_timeout 0`, which would skip the kickoff plan the operator said
  always runs).
- **A consequence the operator should see:** with `_ERR_TEXT` gone, the
  struggle trigger reads failures only from structured fields, so Codex,
  OpenCode and Pi (plain-text tool results) give it no signal. Reading each
  harness's own exit-status line is an open decision.

Tests changed: test_stream (length finish, landing, repeated opening,
fold_back_answer, ALREADY_THOUGHT, machine hand-off), test_ledger ([step],
[fixup], library use, echo habit, compaction thinking, [struggle]),
test_utility (compaction continuity x3, empty answer, compaction thinking),
test_fanout, test_progress, test_tools (x_yamadori key set),
test_stray_markers (section 7), test_harness_forms, test_nebari (new
check), test_vision (the image-paragraph slice), test_live_stack
(fold_back_answer checks; not run: GPU), test_deep, test_deep_review,
test_tool_code, test_e1, test_sessions, test_image_input,
test_harness_decisions, bench/test_laya_calibration.py. RETIRED:
mcp/test_imitated_notes.py (deleted; the feature is gone).

Results, `scripts/run_tests.py` (offline), after the pass: every suite
these changes touch passes -- test_stream 229/229, test_ledger 198/198,
test_utility 192/192, test_tools 478/478, test_deep 181/181,
test_deep_review 53/53, test_sessions 49/49, test_session 28/28,
test_stray_markers 138/138, test_harness_decisions 352/352 (3 KNOWN),
test_harness_forms 36/36, test_responses_api 102/102, test_fanout 96/96,
test_fanout_delivery 31/31, test_tool_code 129/129, test_tiers 178/178,
test_progress 29/29, test_nebari 29/29, test_e1 54/54, test_selection
78/78, test_vision 262/262. The suites still failing belong to the skills
work running beside this pass (test_skills, test_skill_factory,
test_skill_questions, test_skill_turns, test_clm, bench/skills
test_daily_eval, test_gpu_room's skill-selection check, test_web_access's
screen check, ruff's one finding in mcp/skill_packages.py, and the bench
suites charged with index/skills/questions files a worker job wrote).

## Kept on the operator's decision, reclassified from our notes

- `tiers.AGENT_STEP_NUDGE_MESSAGE` is **A**: the AskUserQuestion answer of
  2026-09-26T20:48, "Approve as written (Recommended)".
- `tiers.HELPER_NUDGE_MESSAGE` is **A**: the same answer, "Action-naming
  variant (Recommended)".
- The tool-turn cap of 10 for main and for deep thinking is **A**: the
  answer of 2026-09-23T14:55, "Both, 10 each".
- Sequential fan-out through the second brain, repair at high and max,
  fan-out on code, and the completion-aware "General rule" are **A**: the
  answers of 2026-09-23.
- "One GPU consumer at a time" / one benchmark stream is **A**: "One at a
  time (Recommended)", 2026-09-23.
- `deep.E1_QUESTION_CHARS` 2000 and `E1_CONTEXT_CHARS` 1500 are **B**. They
  are the one cut left on what deep.py sends anywhere: E1's heads were
  trained on states cut exactly so (bench/e1/build_heldout2.py:193-194), and
  e1.render caps the state inside the embedder's `-c 8192`. A longer state
  is off-distribution.

## A-flags the operator should confirm

The direction is the operator's; the number or wording is ours.

- **Thinking caps.** `AGENT_STEP_THINKING` 6144, `USER_TURN_THINKING`
  12288, `HELPER_THINKING` 6144, and `JOB_THINKING` fixup 3072,
  investigate 6144, plan 12288, alternative/tiebreak 6144.
  - The operator's words: "this model overthinks pretty quickly"
    (2026-09-25), and "1.5 those numbers above and dial it in" (summary-quoted).
  - The base values they multiplied (4096 / 8192 / 2048) were ours.
- **`NUDGE_AT` 0.6.** The operator typed "close to 80% of budget"
  (2026-09-25 12:29). 0.6 appears only in a summary quote: "One more round
  with 60% nudge".
- **`slots.IDLE_CLEAR_S` 600 s.** The feature is the operator's: "If we get
  big token boost then clear the idle, no brainer for me" (2026-09-27
  02:05). The code's "the floor the operator set for thin data" has no
  matching message. Either derive the value from the gap data (482 gaps,
  max 38.2 s, one harness), or drop it.
- **`gpu_room.HEADROOM_MIB` 1331.** The operator's 1.3 GB target was for the
  5060 Ti's llama-server, not the A4000; he has since said "600mb free,
  dial it in" (2026-09-25).
- **Ledger eviction** (256 MB per account, 2 GB total, 30 days). The code
  claims "(operator, 2026-09-24)"; no quote was found.
- **Struggle trigger.** `STRUGGLE_THRESHOLD` 3: the operator said "several
  turns". The learner's `BOUNDS` 2-6, `LEARN_MIN_N` 5, `OUTCOME_REQUESTS` 5
  and `HELPED_WINDOW` 20 are ours.
- **`tool_code.REPAIR_ROUNDS` 3.** The operator asked for repair "until it
  compiles"; the 3 is ours.
- **Wording of our own features.** The `seed_frame` research seed line,
  `proxy.ADDENDUM` and its think row, `FOLD_BACK_TAIL` (two prohibitions)
  and `LANDING_PROMPT` are ours, as is the wording of the nudges on top of
  the operator's voice. Memory records that fixes to our own machinery's
  prompts are fine; the operator should see the texts.
- **Operator decisions claimed in code with no matching message.**
  - fanout `AGREE` 0.80 (now removed)
  - utility call -> tier minimal
  - `_ImitatedNotes`
  - LIBRARY USE
  - the `answer_record` session source (AGENTS.md itself says it awaits
    confirmation)
  - skill_select's IMPLIES rows other than r3f v10 -> TSL (the only
    verbatim quote is 08:44, "(TSL is implicit)")
  - `MAX_OPTIONS` "~6"
  - "flag, do not rewrite silently"
  - the yama_* rename, and THINK_DESCRIPTION's server-tool sentence (both
    claim 2026-09-27)
  - make_profile's "SEPARATE profile"
  - the browser-as-default-loadout (quoted only in memory)

## Pending removals, by owner

Each line is the item, where it is, and the exact removal: what remains,
and which tests and docs to update. Owners:

- **proxy.py owner** and **shomen.py owner**: those files were under
  active edit.
- **deep.py owner**: the struggle heuristics. QUESTION_CHARS and
  CONTEXT_CHARS were removed here, at the coordinator's request.
- **skills owner**: skill_*, skills.py, dash_skills.
- **clm.py owner**.
- **unassigned**: a file no agent was editing that this pass did not
  finish. Most of these are features that cannot exist without their
  number, such as the domains gate's word, alias and domain tables and the
  E1 learner's minimums. Removing them removes the feature, which is the
  operator's call. The rest are code-tool output defaults whose bound is
  forced: derive them from the window.

Three items could be deleted outright, because the tiers switches made
them dead (second pass, 2026-09-27):
- `progress.step_kind` and `SEARCH_NAMES`: DELETED. `_INSPECT`,
  `inspect_only`, `_WRITE_OPS` / `_REDIRECT` / `_SEGMENTS` and
  `tool_code.READ_KNOWN` / `read_target` were NOT dead: `skill_select`
  reads them (skills owner's), so they stay.
- `progress.SCRATCH_NAMES`, `_TEMP` and the dot-file rule: dead for the
  fix-up (its scope is DELETED). They are still read by
  `helped_needs_change`, so they go with it (unassigned).
- `shomen.plan_tools` gate and the plan budget branch: DELETED (the plan
  keeps the B rule: no `run_check` without a repository).


#### proxy.py owner (20)

Second pass, 2026-09-27: 18 DONE, 2 LEFT (reasons on their lines).

- `PREAMBLE / preamble_for` (mcp/proxy.py:98,943-993,8208-8217,8339-8347): DONE 2026-09-27: REMOVED: PREAMBLE, preamble_for, available_checks, both proxy call sites, the server.py call and every test's `proxy.PREAMBLE = False` (10 files). Behaviour unchanged (it never wrote anything).
- `COMPACTION_LINK_SECONDS` (mcp/proxy.py:1855,1911,1975): DONE 2026-09-27: REMOVED with the recency guess: `_note_compaction` records only a MAPPED compaction; `_continue_after_compaction` links only a continuation that CARRIES the summary (explicit ids resolve before it). An unmapped compaction's continuation is a new conversation. test_utility updated (3 tests).
- `USE_MAX_CHARS` (mcp/proxy.py:3497,3702): DONE 2026-09-27: REMOVED. Replaced by a B bound: `proxy._injection_room` (main share - the request's high estimate - A_MIN - MIN_THINKING, x3 chars/token, less what else was decided for the message). A definition that does not fit is left uncovered (`did_not_fit`), tried again later.
- `USE_MAX_NAMES` (mcp/proxy.py:3498,3660): DONE 2026-09-27: REMOVED: every imported name of a held package is looked up; the window room is the one bound.
- `USE_MAX_CHARS_EACH` (mcp/proxy.py:3499,3681): DONE 2026-09-27: REMOVED: each definition whole.
- `USE_OVERVIEW_ITEMS + overview filters` (mcp/proxy.py:3500,3552-3571): DONE 2026-09-27: REMOVED with the package OVERVIEW itself (it cannot exist without a picked number): a package used with no names yet gets nothing until a name is used. test_ledger updated.
- `USE_CONVERSATION_MAX_CHARS` (mcp/proxy.py:3501,3643,3702): DONE 2026-09-27: REMOVED: each name is injected once (the covered set), bounded by the window room.
- `answer record` (mcp/proxy.py:1737-1794): LEFT 2026-09-27 (correctness; operator decides): not a number but the only carrier a text-only conversation has. Removing it makes every request of a client that sends no key or header a NEW conversation -- a new slot pin, a new chain salt so turn 1's recorded injection is not replayed, the yama_think_deeply offer re-decided -- which breaks "one model, one cache" (A, operator 2026-09-24). AGENTS.md session paragraph says so.
- `progress.learn / project inference` (mcp/proxy.py:2886-2904,3069-3082): LEFT 2026-09-27: its #56 consumer (the fix-up scope) is removed, but label rule 2 (`helped_needs_change`, unassigned: the operator's call) still reads the project; it goes with that switch.
- `FOLD_BACK_MIN_CHARS` (mcp/proxy.py:4236-4240,7011-7017): DONE 2026-09-27: REMOVED: x_yamadori.fold_back_answer is {chars_after_opening, opened}. test_stream updated; test_live_stack still reads fold_back_answer (the key stays).
- `_ABOVE_REF / _HIDDEN_REF` (mcp/proxy.py:4241-4255,4288-4302): DONE 2026-09-27: REMOVED with refers_to_hidden / refers_context. test_stream, test_tools updated.
- `dedup_opening` (mcp/proxy.py:4258-4285,6671,6717,6883): DONE 2026-09-27: REMOVED (all three call sites and `_opening_repeats_removed`): the answer is delivered as written. test_stream updated.
- `already_thought text` (mcp/proxy.py:4561-4632): DONE 2026-09-27: REPLACED by a factual return (AGENTS "Failure returns carry the next step"): situation (ran once, when, concluded or not, searches, where its hand-off/plan is), retryable false, remedy {agent: use the result already in this reply}. Removed: "answer now" / "make the next call now", "cite by path:line", labels[:6], reads[:6]/[:12]. test_stream updated.
- `_ToolMarkup + tool_markup_notice` (mcp/proxy.py:5812-5979,6917-6937): DONE 2026-09-27: REMOVED (class, notice, _Out filter, x_yamadori.tool_markup): the landed answer is delivered as written. test_stray_markers section 7 rewritten (138/138); test_tools key set.
- `_ImitatedNotes + NOTE_HEAD_MAX` (mcp/proxy.py:5982-6181,6508-6513,7045-7064): DONE 2026-09-27: REMOVED (class, note_heads, _notes_record, _Out filter, x_yamadori.imitated_notes). mcp/test_imitated_notes.py RETIRED (deleted; it was untracked); test_stray_markers and test_tools updated. Our own note still bypasses the marker filter.
- `withheld-call defect notices` (mcp/proxy.py:6899-6911): DONE 2026-09-27: REMOVED: a withheld call of ours leaves empty content, finish_reason stop. test_stream's landing check updated.
- `_budget_notice / _budget_note` (mcp/proxy.py:3895-3927,6945-6950): DONE 2026-09-27: REMOVED: a `length` finish is finish_reason length with the content generated; no fan-out / code check on it. test_stream updated.
- `_empty_notice` (mcp/proxy.py:5264-5286,6951-6957): DONE 2026-09-27: REMOVED: blank content, finish_reason stop. test_utility updated.
- `ECHO_WINDOW` (mcp/proxy.py:7728-7758): DONE 2026-09-27: REMOVED the 8-observation window. DEVIATION from the doc's two options, both of which break a tested path ("strips" breaks an echo-only account's first-turn warm, test_ledger [fixup echo] RENDER; "last observation" reproduces the mixed-account incident): the account keeps the SET of habits it was seen with, and a first turn guesses "echoes" only if it was never seen stripping. No number. test_ledger updated.
- `PREAMBLE on first turn (blocking)` (mcp/server.py:500-511; mcp/proxy.py:98): DONE 2026-09-27: REMOVED with PREAMBLE (server.py block deleted).

#### shomen.py owner (13)

Second pass, 2026-09-27: all 13 DONE.

- `HANDOFF_TARGET_WORDS` (mcp/shomen.py:220,238): DONE 2026-09-27: REMOVED with its "Aim for under N words" sentence.
- `investigate SYSTEM: plain-English style block` (mcp/shomen.py:268-283): DONE 2026-09-27: REMOVED; the routing table and the image paragraph stay.
- `investigate temperature` (mcp/shomen.py:1317): DONE 2026-09-27: REMOVED (tiers.enforce_sampling set it anyway).
- `context[:1500] in investigate` (mcp/shomen.py:1244): DONE 2026-09-27: REMOVED: the context is sent whole.
- `PLAN_TARGET_WORDS` (mcp/shomen.py:306,322): DONE 2026-09-27: REMOVED with v1's word-target sentence.
- `PLAN_SYSTEM_V2: style steering` (mcp/shomen.py:375-376): DONE 2026-09-27: REMOVED the two sentences. test_deep updated.
- `T_INVESTIGATE / T_ANSWERED` (mcp/shomen.py:569-570): DONE 2026-09-27: REMOVED.
- `LAYA_PERMUTATIONS / MARGIN_GATE` (mcp/shomen.py:611-612): DONE 2026-09-27: REMOVED with route_in/distil, _choice_averaged, _laya, LAYA_URL, LAYA_TIMEOUT. bench/test_laya_calibration.py and test_e1 updated.
- `route_in() / distil() prompts and cuts` (mcp/shomen.py:587-762): DONE 2026-09-27: REMOVED (no production caller).
- `FIXUP_MIN_KEEP` (mcp/shomen.py:1048,1069-1071): DONE 2026-09-27: REMOVED (and _nonblank): fix_rejection keeps closed fence / not cut / no errors. A short parsing repair is now written back. test_tool_code updated.
- `fixup cuts` (mcp/shomen.py:986,1000): DONE 2026-09-27: REMOVED: all errors and the request whole. test_tool_code checks it.
- `MACHINE EVIDENCE scoring` (mcp/shomen.py:2209-2362): DONE 2026-09-27: REMOVED: the machine hand-off lists the paths retrieved and the search log. proxy.FOLD_BACK_TAIL_MACHINE and deep.THINK_REASONING_MACHINE no longer promise source lines. test_stream updated.
- `machine hand-off cuts` (mcp/shomen.py:2162,2185,2189): DONE 2026-09-27: REMOVED. test_stream test_nothing_cuts_a_hand_off_or_a_plan checks 20 paths, a long question and reason whole.

#### deep.py owner (10)

Second pass, 2026-09-27: all 10 DONE (two deviations, on their lines).

- `STRUGGLE_WINDOW` (mcp/deep.py:112): DONE 2026-09-27: REMOVED with its env var: signals count from the episode boundary (last run, or a compaction).
- `COOLDOWN_REQUESTS` (mcp/deep.py:115): DONE 2026-09-27: REMOVED with its env var and the in_cooldown label; x_yamadori.deep.last_run {requests_since_run, boundary} replaces rec.cooldown (the DB column stays for old rows).
- `same_pattern suppression` (mcp/deep.py:1276-1287): DONE 2026-09-27: REMOVED with missed_same_pattern; a run is still judged by its pattern's recurrence (_recurrence, not_helped).
- `DEFER_REQUESTS + one lane-wait per episode` (mcp/deep.py:142,1219-1227): DONE 2026-09-27: REMOVED (mark_deferred, waited_episode, the proxy call sites and lane_timeout plumbing). DEVIATION from the doc's "lane_timeout 0": a trigger waits the standard helper-lane wait (admission.WAIT_SECONDS, B) like every helper job, then is skipped for that request only -- lane_timeout 0 would skip the kickoff plan whenever another conversation holds the lane, against the operator's "always give it a planning turn" (A, 2026-09-27).
- `unseen: any prerelease` (mcp/deep.py:847-849,899-900): DONE 2026-09-27: REMOVED. The registry history stores stable releases only, so a prerelease's own date is unknown: a prerelease is judged by the same rules as any version. @react-three/fiber@10.0.0-alpha.5 and @react-three/drei@11.0.0-alpha.7 stay unseen, as new majors.
- `_ERR_TEXT word list` (mcp/deep.py:272-285): DONE 2026-09-27: REMOVED: is_error reads only structured fields (exit status, exit_code_meaning, ok/success, error field). CONSEQUENCE: Codex, OpenCode and Pi send plain-text tool results, so struggle gets no failure signal from them (test_harness_decisions marks their struggle fixtures KNOWN). Reading Codex's "Process exited with code N" / Pi's "Command exited with code N" (each harness's own exit-status line) is an OPEN operator decision.
- `STILL_BROKEN phrase regex` (mcp/deep.py:292-300): DONE 2026-09-27: REMOVED with the user_still_broken signal and the "still broken" labels (deep_learn.propose_descriptions inert for new rows). e1.HEADS escalate still lists the feature (always 0) -- changing it changes the head's dimensions.
- `_NOT_APPLIED phrases` (mcp/deep.py:397-400): DONE 2026-09-27: REMOVED; the harness's structured no_change flag stays.
- `error_signature / normalise_error_line` (mcp/deep.py:439-510): DONE 2026-09-27: REMOVED the normaliser: the raw first error line (exact); records keep a sha1 digest of it, never the text.
- `ENVIRONMENT table` (mcp/deep.py:529-556): DONE 2026-09-27: REMOVED with environment_class and the environment record: every failure counts.

#### skills owner (88) -- applied 2026-09-27 (DONE / KEPT with the check that fails / OPERATOR)

- `MAX_TITLE_CHARS` (mcp/skill_limits.py:83): Drop; title length is covered by body token cap. No test. **DONE**: removed; the body token cap bounds a skill (skill_builder has no title check).
- `DESCRIPTION_AIM` (mcp/skill_limits.py:91): Use the 1,024 spec bound only. skill_compile.py:292 truncates at it. **DONE**: removed; compile bounds a description at the spec's DESCRIPTION_CHARS (1,024), the prompts read that.
- `MAX_TAGS / MAX_TOPICS` (mcp/skill_limits.py:92-93): Drop caps or derive from description size. No test. **DONE**: removed; tags_of, gates and extract_topics keep every topic.
- `TURN_TOKENS_AIM / TURN_TOKENS_HARD` (mcp/skill_limits.py:102-103): Delete; remove the meter from web/src/screens/Skills.tsx. Test: test_skill_factory.py. **DONE**: removed; the meter is gone from web/src (Skills.tsx, api/skills.ts, Skills.test.ts; tsc and vitest pass). web/dist is NOT rebuilt.
- `STEP_MAX_BODIES` (mcp/skill_limits.py:107): Remove; ceiling (if any) applies. No test. **DONE**: removed; one recall per area per turn, the sanity ceiling (max_skills_per_turn) applies.
- `FADE_TOKENS` (mcp/skill_limits.py:114): Recall on evidence events only (asked/error/phase) without a fade. test_skill_turns.py references it. **DONE**: removed with the faded state and GROW (skill_chart); a given skill is recalled only on its own ASKED / ERROR / PHASE event. daily_eval seq-koota-fade's last turn now expects nothing.
- `RECALL_EVERY_STEPS` (mcp/skill_limits.py:117): Drop cooldown; keep "never the identical line twice". test_skill_turns.py; skill_chart cooling guard. **DONE**: removed with the chart's cooling guard; never the identical recall line twice in a row stays (test_skill_turns, test_skill_questions).
- `MAX_TRIGGERS / TRIGGER_CHARS` (mcp/skill_limits.py:128-129): Keep all description sentences (<=1,024 chars bounds them). No test. **DONE**: removed; triggers_of keeps the whole description as one trigger plus every when-to-use bullet (the 1,024 bound bounds them).
- `MAX_ACTIVATION_TESTS / MAX_BEHAVIOUR_CHECKS` (mcp/skill_limits.py:135-136): Drop caps. No test. **DONE**: removed with skill_tests' case caps.
- `EMB_HIGH` (mcp/skill_select.py:127): Use cosine only for ranking inside a question (set-relative, like the deciders). Dashboard label "unmeasured starting points" (dash_skills.py:151). **DONE**: removed; a cosine only ranks inside a question. A round the patterns leave silent admits nothing on a cosine (embedding-only rows are no candidates), so the CATEGORY question went too. dash_skills reports thresholds {}.
- `EMB_LOW` (mcp/skill_select.py:128): Same as EMB_HIGH. Referenced in mcp/test_skills.py. **DONE**: removed with EMB_HIGH.
- `MAX_ASK` (mcp/skill_select.py:129): Bound by question options instead. No test. **DONE**: removed; a question is bounded by its own options.
- `REQUEST_BUILD_MAX` (mcp/skill_select.py:241): Build vectors at arm/startup always (refresh_triggers exists). test_skills.py references it. **DONE**: removed; the request path never builds trigger vectors (trigger_index(request=True)); they are built at arm / refresh_triggers.
- `fallback sampling` (mcp/skill_select.py:786): Use tier defaults (vendor sampling); operator decides internal temperature. **DONE**: removed; the fallback call sends no temperature or max_tokens (tier defaults, vendor sampling).
- `FALLBACK_SYSTEM decision table` (mcp/skill_select.py:152): Keep DATA framing; table wording needs measurement or operator sign-off. **OPERATOR**: kept as written, DATA framing included; its wording needs a measurement or the operator's sign-off.
- `HOSTED` (mcp/skill_select.py:381): Remove host ordering; decider ranks. **DONE**: the extras' host ordering is removed; the HOSTED table still names host areas for the relevance floor below, which is kept.
- `confidence()` (mcp/skill_select.py:397): Order by decider probability only. **DONE**: no weights: a lexicographic rank -- strength ordinal (fact > phrase > word), then strong-topic count, then cosine.
- `_REL_STOP stopword list` (mcp/skill_select.py:411): Use IDF alone (common words already weigh ~0). **KEPT**: IDF alone moves the koota asked pick traits -> queries (daily_eval seq-koota-body-recall#1, 2 checks).
- `stems() rules` (mcp/skill_select.py:433): Use the embedder tokenizer or a standard stemmer. **KEPT**: no standard stemmer is installed offline, and the embedder's tokenizer needs the stack.
- `REL_WORD_MIN / REL_FLOOR` (mcp/skill_select.py:429-430): Remove the floor; let the decider see host-area options (set-relative). Also read_craft topic threshold (:2287). No test. **KEPT**: with no floor every host pick is eligible and daily_eval three-shadow-acne's MUST-NOT fails (a TSL craft on 'shadow'); the structural alternative (a host pick needs a candidate row) fails rust-repr-c and emscripten-embind MUSTs. The read_craft topic threshold is DONE (read_craft reads by name only).
- `IMPLIES r3f->react refs skill` (mcp/skill_select.py:527): Delete row; asked r3f still gets its own area skill. daily_eval rows expect it. **KEPT**: for the operator: deleting it fails 5 daily_eval MUST area:react checks (the operator's r3f prompts).
- `IMPLIES webgpu+threejs -> TSL skill` (mcp/skill_select.py:544): Delete row. **DONE**: row deleted.
- `IMPLIES koota+react / koota+r3f` (mcp/skill_select.py:548,551): Delete rows; asked koota gets its area pick. **DONE**: rows deleted; asked koota gets its area pick.
- `IMPLIES pmndrs_math+threejs/r3f` (mcp/skill_select.py:556,560): Delete rows. **DONE**: rows deleted.
- `IMPLIES typegpu+threejs` (mcp/skill_select.py:563): Delete row. **DONE**: row deleted.
- `_BUILD_ASK + BUILD_HINT` (mcp/skill_select.py:574-577): Remove; ranking by relevance/decider. daily_eval exact-prompt rows depend on it. **KEPT**: removing it loses r3f-v10-setup-21 on op-sentence-pagoda, op-pagoda-r3f-stack-exact, stack-particles and seq-unrelated-after-stack#0 (4 MUST checks; r3f-tsl-hooks-18 takes the asked r3f slot). For the operator.
- `older-major penalty` (mcp/skill_select.py:715): Remove. **DONE**: replaced by a structural filter: a skill whose NAME carries vN other than the asked version is skipped.
- `generality ordering` (mcp/skill_select.py:721): Remove. **KEPT**: removing it moves the koota asked pick traits -> queries (seq-koota-body-recall#1, seq-koota-fade#1: 4 checks).
- `QUESTION_TEXT` (mcp/skill_select.py:1086): Measure with CLM fidelity set or operator approves. **OPERATOR**: kept; measure with the CLM fidelity set, or the operator approves.
- `_strong()` (mcp/skill_select.py:1141): Replace with decider pick vs NONE only. **DONE**: removed; the tier is the verdict (a decider's pick vs NONE).
- `index candidates` (mcp/skill_select.py:1628-1630): Index = chosen + asked areas' remaining, capped by INDEX_MAX only. **DONE**: only INDEX_MAX bounds the index.
- `_PHASE_ORDER` (mcp/skill_select.py:1685): Keep all phases in state; no priority. No test. **DONE**: removed; skill_select.phases_of keeps every phase, and a change is a new phase entered.
- `_item_score weights` (mcp/skill_select.py:1809-1812): Recall the skill's first DO + first DO NOT item, or decider picks. **DONE**: removed; a recall line is the skill's first DO/WHEN item plus its first DO NOT.
- `recall candidates` (mcp/skill_select.py:1842): Remove the limit. **DONE**: limit removed with _item_score.
- `step conf bonus` (mcp/skill_select.py:1895): Remove; decider ranks. **DONE**: removed.
- `compaction detection` (mcp/skill_select.py:1995): Use the proxy's own compaction signal (_serve_compaction / session) instead of a ratio. **DONE**: the 0.7 shrink ratio is gone; the proxy passes its own compaction count (progress 'compactions', proxy._compactions on the skills paths) and a moved count resets what was given.
- `read_craft matching` (mcp/skill_select.py:2262-2302): Name match only, else return list of names (near) without a threshold. **DONE**: name only (craft/4); an unknown name returns NO_SUCH_CRAFT with every name sharing a word as `near`, no threshold.
- `craft offer only if index non-empty` (mcp/skill_select.py:2233): Operator decides. **OPERATOR**: kept as is.
- `docstring stale budget` (mcp/skill_select.py:50-52): Delete the stale lines. **DONE**: stale lines deleted.
- `LexicalStub SCALE / IDF smoothing` (mcp/skill_deciders.py:164,180): Delete lexical decider or leave as test-only. **DONE**: the lexical decider is test-only: skill_deciders.SERVING = (stub, clm); YAMADORI_SKILL_DECIDER cannot select it.
- `_STOP (deciders)` (mcp/skill_deciders.py:139): Goes with the lexical decider. **DONE**: test-only with the lexical decider.
- `PHASE_TOP mapping` (mcp/skill_chart.py:84): Keep fine phases only. **DONE**: removed; the chart's top states are the fine phases (implement, debug, ...).
- `MIN_HITS / SHARE` (mcp/skill_classify.py:71-72): Declared metadata or model tag stage decides; drop counts. **KEPT**: with declared terms only, an undeclared source is filed by nothing offline: test_skills' fixture skills quarantine ('names no language'), test_pmndrs_stack's pinned sources fail.
- `PHASES taxonomy` (mcp/skill_classify.py:274): Operator confirms axis. **OPERATOR**: kept; the operator confirms the axis.
- `_PHASE_RX keyword tables` (mcp/skill_classify.py:278-309): Phase from structure only (tool error in last result, route class) or model tag. **KEPT**: phase from structure only fails 5 daily_eval checks (seq-koota-body-recall#1/#2, seq-koota-fade#1) and the phase-gated crafts' own activation tests (module-exports-match-callers, koota-queries-and-systems, koota-traits-and-entities: quarantine on re-arm).
- `SITUATIONS set` (mcp/skill_classify.py:312): Keep only if authored skills stay; else remove. **OPERATOR**: kept, because the authored skills stay; the operator confirms.
- `_CALL_MISMATCH` (mcp/skill_classify.py:333): Remove with the situation. **OPERATOR**: kept with the situation.
- `_CODE_SHAPED / code_shaped` (mcp/skill_classify.py:351,415): Treat topics as confirming only; decider decides. **KEPT**: topics as confirm-only fails 9 checks (bug-koota-throw, seq-koota-body-recall#1/#2, seq-koota-fade#1, seq-phase-debug#3: the koota-queries craft on updateEach needs the code-shaped topic as a fact).
- `_PROSE_ABBREV / _PACKAGE_NAME / _FILE_NAME exclusions` (mcp/skill_classify.py:358-366): Goes with code_shaped. **KEPT**: with code_shaped.
- `COMMON_API` (mcp/skill_classify.py:379): Remove; rely on decider. AGENTS.md skills bullet mentions it. **KEPT**: removing it makes 'localStorage' alone a FACT for an auth-token craft (test_skill_turns [precision], 2 checks) and moves test_pmndrs_stack's React pick.
- `HOST_FRAMEWORKS` (mcp/skill_classify.py:412): Remove. **KEPT**: removing it makes 'React' in prose an asked FACT, which moves every React mention off the ask path (10 test_skills cascade checks).
- `VERBS + phrase window` (mcp/skill_classify.py:443,452): Nouns only (word strength). **KEPT**: nouns-only makes 'make me a slide deck' a WORD, which the language round does not open: no slides/docs craft is selectable (test_skills slides check).
- `_KEYWORDS` (mcp/skill_classify.py:619): Use IDF over the store. **KEPT**: an IDF cut over the store needs a threshold, and none is sourced.
- `extract_topics limits` (mcp/skill_classify.py:631-663): Drop limits (MAX_TOPICS already). **DONE**: removed (limit=None, no length bounds).
- `triggers_of rules` (mcp/skill_classify.py:711,727,729): Keep whole description as one trigger. **DONE**: the whole description is one trigger, plus every when-to-use bullet; _sentences removed.
- `non-code artifact prominence` (mcp/skill_classify.py:825): Model tag stage decides. **KEPT**: with MIN_HITS / SHARE.
- `WORKED_MESSAGES / WORKED_CHARS` (mcp/skill_classify.py:1062-1063): Use messages since last user turn (a structural boundary). **DONE**: replaced by a structural boundary, skill_classify.window_start: since the last user turn (the previous one when the request ends on a user turn).
- `EMBED_SUBSTANTIVE_CHARS / DIGEST_SHORT / DIGEST_LONG` (mcp/skill_classify.py:1127-1129): Fixed digest or none; measure. **DONE**: a fixed digest: every line; the whole query is cut at EMBED_QUERY_CHARS, the user text first.
- `DIGEST_ERRORS/IMPORTS/NAMES/FILES` (mcp/skill_classify.py:1130-1133): Goes with the digest. **DONE**: gone with the digest counts.
- `_PLAIN_WORDS` (mcp/skill_classify.py:1143): Use IDF. **KEPT**: it feeds _REL_STOP, which is kept.
- `_evidence window` (mcp/skill_classify.py:1166-1178): Since-last-user-turn boundary. **DONE**: window_start (above).
- `_about_evidence join rule` (mcp/skill_classify.py:1209): Always or never join; measure. **DONE**: always join.
- `_word_stems 4+ letters` (mcp/skill_classify.py:1200): Goes with the join rule. **DONE**: gone with the join rule.
- `implement default phase` (mcp/skill_classify.py:1455): Remove with phase keyword tables. **KEPT**: with _PHASE_RX.
- `_NEGATION + window` (mcp/skill_classify.py:1597,1629): Remove; decider sees the sentence. **DONE**: the pattern is KEPT (removing it fails stack-without's MUST-NOT); the 24/60-character windows are replaced by the clause (_clause_before).
- `_CONTEXT_MENTION` (mcp/skill_classify.py:1603,1636): Remove. **DONE**: the pattern is KEPT (removing it fails three-shadow-acne's MUST-NOT); the 24-character window is replaced by the clause.
- `alias version window` (mcp/skill_classify.py:1654): Remove or generalise to a parser. **DONE**: the 60/40-character windows are removed; the alias-version structure is KEPT (removing it fails 12 checks: v10 on the operator's sentence).
- `plain topics need two` (mcp/skill_classify.py:1719): Plain topics only confirm; decider decides. **KEPT**: removing it fails react-dev-blog-react-19-2-conditional-rendering's own activation near miss (quarantine on re-arm).
- `tools-keyed base score` (mcp/skill_classify.py:1732): Remove (ordering only). **DONE**: the score is the strength ordinal.
- `other-language gate` (mcp/skill_classify.py:1746): Remove; language round filters. **KEPT**: removing it makes the pmndrs math and koota crafts fail their own activation near misses (test_pmndrs_stack: 7 of 10 quarantined on re-arm).
- `match score weights` (mcp/skill_classify.py:1766-1774): Order by decider only. **DONE**: the score is the strength ordinal (fact 3 / phrase 2 / word 1); strong_topics is reported beside it.
- `_primary base weights` (mcp/skill_classify.py:1790-1806): Keep strength (ordinal); drop score magnitudes. **DONE**: strength ordinal only.
- `STRIP_MAX_FRACTION` (mcp/skill_screen.py:1053): Strip only (never drop whole) or operator sets. No test pins 0.25. **DONE**: removed; the screen strips, and drops a text only structurally: a finding with no line to cut, what is left still failing, or nothing but headings left (test_deep's all-injection page; test_deep_review's knowledge-base check updated).
- `SCREEN_SYSTEM prompt` (mcp/skill_screen.py:944): Measure on bench/skills/fixtures or operator approves. **OPERATOR**: kept; measure on bench/skills/fixtures, or the operator approves.
- `MAX_CHUNKS / MAX_SOURCE_CHARS` (mcp/skill_pipeline.py:86,537): Raise to worker's bound or operator picks. bench/skills/eval_builder.py. **DONE**: the pipeline reads the worker's own bound (YAMADORI_EXTRACT_MAX_CHUNKS, 60); MAX_SOURCE_CHARS = CHUNK_CHARS x that. The 60 is the worker's (unassigned).
- `pipeline sampling` (mcp/skill_pipeline.py:364,584,627,663,759,838,863): Use vendor sampling or operator decides. **DONE**: no temperature on any pipeline call (vendor sampling).
- `decompose allowance` (mcp/skill_pipeline.py:663): Use A_MIN or measure reply sizes. **DONE**: DISTIL_MAX_TOKENS (= A_MIN).
- `MAX_CHILDREN` (mcp/skill_pipeline.py:643): Drop cap. **DONE**: removed.
- `generated test templates` (mcp/skill_tests.py:66-98): Model-proposed or authored tests only. **KEPT**: without them every compiled skill needs a model call to get tests, and quarantines offline (migration, authored, dataset rows).
- `case caps` (mcp/skill_tests.py:215,397): Drop. **DONE**: removed (clean_behaviour, merge, generate).
- `MIN_QUOTE_CHARS` (mcp/skill_builder.py:65): Align with prompt or operator sets. **DONE**: aligned: one constant, skill_limits.MIN_QUOTE_CHARS, which skill_builder and the prompts read.
- `ABSOLUTE regex / DO NOT rule` (mcp/skill_builder.py:68): Remove; MAX_PROHIBITIONS governs. **DONE**: removed; MAX_PROHIBITIONS governs (the golden fixture carries its DO NOT).
- `trailing-item drop` (mcp/skill_builder.py:270): Fail with reason instead (if the flag-not-rewrite rule is confirmed). **KEPT**: failing with a reason instead depends on the flag-not-rewrite rule, which is unconfirmed: the operator's.
- `compile group-size` (mcp/skill_compile.py:213): Operator confirms grouping; else pack by file only. **DONE**: one unit per topic (the >=3 pooling removed); the operator confirms the grouping.
- `compile rule caps` (mcp/skill_compile.py:275-283): Drop caps. **DONE**: removed.
- `compile description` (mcp/skill_compile.py:292): Use 1,024 spec bound. **DONE**: bounded by DESCRIPTION_CHARS (1,024).
- `prompt size numbers` (mcp/skill_prompts.py:71-77,196,252,287-289): Operator approves or derive from skill_limits. **DONE**: derived from skill_limits (NAME_CHARS, DESCRIPTION_CHARS, MIN_ITEMS, MAX_ITEMS, MIN_QUOTE_CHARS, MAX_ITEM_CHARS); versions distil/3, decompose/2, tag/4, tests/2; pins regenerated.
- `DECOMPOSE "at most three skills"` (mcp/skill_prompts.py:171): Fix wording with version bump. **DONE**: wording fixed (decompose/2).

#### clm.py owner (1)

- `INSTRUCTIONS / NONE_OPTION (CLM)` (mcp/clm.py:111-112): Measure on bench/clm/fidelity.py.

#### unassigned (101)

Second pass, 2026-09-27: `COMPACTION_THINKING` (the caller named it),
`SEARCH_NAMES` and `step_kind` (dead) DONE; `_INSPECT`,
`_WRITE_OPS/_REDIRECT/_SEGMENTS` and `READ_KNOWN` KEPT (skill_select reads
them). The rest are the operator's call.

- `COMPACTION_THINKING` (mcp/tiers.py:765): DONE 2026-09-27: REMOVED: tiers.compaction_budget returns `thinking_tokens` = its window - prompt - answer, never below MIN_THINKING (the one budget rule); compaction.prefix_fields takes it. test_utility, test_ledger, test_harness_forms updated.
- `AB_ARMS / ab()` (mcp/tiers.py:982): a benchmark factorial helper; the operator rejected on/off arms (2026-09-27). Delete ab()/AB_ARMS and test_tiers' check.
- `BEHAVIOURS: helped_needs_change` (mcp/tiers.py:1089): a labelling rule for the learner, from #52 (n=1). Remove the switch and LABEL_RULE 2, or measure.
- `IDLE_CLEAR_S` (mcp/slots.py:883): mechanism needs a threshold: derive it from the measured gap distribution with the rule written, or drop idle_clear; tests test_slots.py, test_utility.py; AGENTS.md:159-188
- `RESTORED_MIN_TOKENS` (mcp/slots.py:1099): report reused/processed raw; no tests
- `TTL_SECONDS (nebari)` (mcp/nebari.py:57): remove -> state keyed by explicit session id (#41) lives until ledger's age bound; no tests
- `VARIANTS evidence/skeptical/terse` (mcp/fanout.py:149): Delete the 3 unused rows; tests test_fanout.py, test_pmndrs_stack.py reference VARIANTS
- `_ASKED_LANG / with_language` (mcp/fanout.py:413): Remove: untagged answers take the prose path (recorded only). Tests: test_fanout.py
- `_CODE_WORD + >=2 lines` (mcp/fanout.py:417): Goes with with_language
- `code vs prose split` (mcp/fanout.py:1151): Use A's language only (already computed in run()); tests test_fanout.py
- `SIM_TIE` (mcp/fanout.py:877): Remove: exact ties only, then prefer C. No test references it
- `medoid tie-break order` (mcp/fanout.py:1214): Prefer C on tie, else A. Tests test_fanout.py
- `break_tie margin` (mcp/fanout.py:1312): Delete break_tie/_choice_averaged (no callers)
- `_choice_averaged caps` (mcp/fanout.py:1271): Delete
- `ALT_MIN_WORDS` (mcp/fanout.py:1367): Hand back B's text whole (or its differing paths/symbols only). No test reference
- `ALT_NOVELTY` (mcp/fanout.py:1368): same
- `_STOPWORDS (fanout)` (mcp/fanout.py:1375): Goes with ALT_NOVELTY
- `UNKNOWN_MIN_CHARS/LINES` (mcp/tool_code.py:101): Log every unrecognised client call. No test reference
- `READ_KNOWN` (mcp/tool_code.py:211): KEPT 2026-09-27: skill_select reads tool_code.read_target (skills owner's).
- `IMAGE_REGENERATIONS` (mcp/tool_code.py:1109): Land on first stop, or rely on tiers.tool_turn_limit. Tests test_image_guard.py; AGENTS.md image guard para
- `CODE_LANGS` (mcp/code_check.py:117): Low impact; keep or measure
- `infer_width min lines` (mcp/code_check.py:334): Low impact
- `review_answer feedback text` (mcp/code_check.py:1866): Delete feedback text; tests test_code_check.py
- `_UNTAGGED_ERROR_SHARE` (mcp/route.py:189): Sweep on bench/route/labels.jsonl or require 0 error lines; tests test_route.py
- `_UNTAGGED_GRAMMARS` (mcp/route.py:190): Try every CODE_LANG; test_route.py
- `MIN_QUESTION_CHARS` (mcp/selection.py:149): Remove (legacy path) ; no test reference
- `CONTEXT_CHARS` (mcp/selection.py:154): Remove with legacy path; no test reference
- `_DESIGN` (mcp/selection.py:491): Delete; proxy passes a route. No test reference
- `_TEMP` (mcp/progress.py:63): Remove with scope; tests test_progress.py
- `SCRATCH_NAMES` (mcp/progress.py:67): Delete table. Tests test_progress.py; AGENTS.md:625
- `dot-file rule` (mcp/progress.py:255): Remove with scope
- `project inference (named_paths, tree_paths, stated_root, _CWD_WORDS, _infer_root, _learn_…` (mcp/progress.py:81): Remove with those features; tests test_progress.py
- `_INSPECT` (mcp/progress.py:360): KEPT 2026-09-27: skill_select reads progress.inspect_only (skills owner's).
- `_WRITE_OPS/_REDIRECT/_SEGMENTS` (mcp/progress.py:348): KEPT 2026-09-27: skill_select reads _REDIRECT, _SEGMENTS and inspect_only (skills owner's).
- `SEARCH_NAMES` (mcp/progress.py:383): DONE 2026-09-27: DELETED with step_kind.
- `step_kind` (mcp/progress.py:412): DONE 2026-09-27: DELETED (and the step_thinking switch). test_progress: test_step_kind replaced by test_inspect_only.
- `removed situations still replay` (mcp/progress.py:15): Accept (cache) or purge ledger rows holding them
- `STOP` (mcp/fusion.py:41): Use no stoplist (BM25 idf discounts common words)
- `content_words min length` (mcp/fusion.py:61): Low impact
- `vote-count-first ordering` (mcp/fusion.py:138): Plain RRF order
- `detect_repo weights` (mcp/detect.py:110): Delete module and the two scripts
- `away_from median ceiling` (mcp/concept_seed.py:181): Uniform draw
- `centroid word regex` (mcp/concept_seed.py:150): Goes with away_from
- `record_step / read_rings descriptions` (mcp/rings.py:162): Plain description; test_rings.py
- `FANOUT_HOLD_S / FANOUT_DECAY_S` (mcp/recent_turns.py:28-29): Harmless display; keep or operator picks. test_tree_sources.py.
- `FOLIAGE_WINDOW_S / FOLIAGE_TURNS / FOLIAGE_FULL` (mcp/recent_turns.py:34-36): Display only. test_tree_sources.py.
- `KB match floor` (mcp/research_tools.py:309): Rank by words matched with no floor (or by embedding). test_deep.py / test_skill_factory.py call find_in_knowledge_base (behaviour, not the constant).
- `tool descriptions (find_in_knowledge_base, read_web_page, search_web) + result wording` (mcp/research_tools.py:122-193,:1319): Minimal factual descriptions (what it returns, args). test_deep.py checks DATA_NOTE/names. AGENTS.md:57.
- `LINKS_LISTED` (mcp/research_tools.py:665): Return no link list; seen-link provenance still works from page parse. test_web_access.py test_links_are_listed_followed_and_seen. AGENTS.md:57 'up to 20 of the page's visible links'.
- `search_web code default + categories` (mcp/research_tools.py:188,:1229): Default code=false or let the model choose explicitly. No test names it.
- `MIN_EDGE` (mcp/images.py:130): Accept any multiple of 32. No test names it.
- `MAX_N` (mcp/images.py:134,:657): n=1 only (as dall-e-3) until measured. No test names it.
- `_oom smaller-size remedy` (mcp/images.py:310): Remedy says 'use a smaller size' without a number. No test.
- `_TRIM_SEGMENTS` (mcp/image_input.py:287): Try only the whole run (cut -> IMAGE_INCOMPLETE). No test names it.
- `EFFORT` (mcp/vision.py:97): Use model.chat default or the request's effort. No test names vision.EFFORT.
- `vision SYSTEM prompt` (mcp/vision.py:134-139): Send the question alone. No test pins it.
- `MIN_TRAIN_N` (mcp/e1.py:85): Serve any head that fits with >=2 labels, or keep only the promotion test as gate. test_e1.py references MIN_TRAIN_N. AGENTS.md ~1227 (E1 section).
- `MIN_PER_LABEL` (mcp/e1.py:86): Drop; rely on McNemar promotion. No test names it.
- `LEARN_MIN_NEW` (mcp/e1.py:87): Retrain each idle pass; promotion test decides. No test names it.
- `MIN_EVAL_N` (mcp/e1.py:88): Let the exact test decide at any n. No test names it.
- `EVAL_FRAC` (mcp/e1.py:89): Any split; document it. No test.
- `_deep_label outcome->label map` (mcp/e1.py:725-764): Train only on gold/hand labels; untrained heads fall back to rules. No test calls _deep_label.
- `escalate head extras` (mcp/e1.py:117-124): Remove with the escalate head (untrained). No test named.
- `DOMAINS vocabulary + STRICT_FILES` (mcp/domains.py:62-69): Delete with the recipe filter. test_domains.py.
- `PACKAGE_DOMAINS` (mcp/domains.py:73-116): Gate on held/imported/named only; drop domain rule. test_domains.py.
- `EXTENSION_DOMAINS` (mcp/domains.py:118-127): Same as above. test_domains.py.
- `WORD_DOMAINS` (mcp/domains.py:165-191): Remove word evidence; gate no longer withholds on words. test_domains.py test_a_word_counts_only_in_its_domain_sense, test_the_word_stems_now_match_their_words.
- `DERIVE_MIN_FILES` (mcp/domains.py:229): Any import counts, or drop derivation with the domain rule. test_domains.py references it.
- `HELD_ALIASES` (mcp/domains.py:431-460): Match install name/import only. test_domains.py test_a_package_is_not_named_by_an_ordinary_word.
- `symbol code-shape rule` (mcp/domains.py:468-470,:655-704): Probe all identifier tokens. test_domains.py test_plain_words_are_not_symbols...
- `PLATFORM_NAMES / _CONTEXT / _NOT_MEMBER` (mcp/domains.py:512-558): Drop; symbol evidence only from code-shaped tokens. test_domains.py test_english_words_are_not_symbols_even_where_defined.
- `tool_admission DOMAIN_OUTSIDE rule` (mcp/domains.py:877-907): Offer whenever something is held and no fact rule withholds. test_domains.py test_a_puzzle_is_withheld..., test_real_prompts_against_the_real_store.
- `worth_indexing` (mcp/deps.py:132): min_imports=1, no max; CLI only. No test.
- `REUSE_GAP_DAYS` (mcp/deps.py:681): Treat any version run with a known different repository as another project. No test names the constant (test_deep.py covers name_reused).
- `schema_fill score levels / range` (mcp/schema_fill.py:55-57,:91): Delete module. No test.
- `CANDIDATES` (mcp/code_search.py:85): Dormant (semantic off). If semantic returns, use the full pool or measure. No tests reference it.
- `DEFAULT_TOP_K` (mcp/code_search.py:89): Leave to caller; if a default is needed, cite a measurement. No tests reference it.
- `search_fused wide` (mcp/code_search.py:647): Take all ranker hits (fusion is cheap) or measure recall vs depth.
- `search_fused output cap` (mcp/code_search.py:746): Return top_k fused rows, or measure.
- `symbol-only hits surfaced` (mcp/code_search.py:754): Surface all declaration hits (exact by construction) or cap by top_k.
- `WEAK_SIM` (mcp/code_search.py:1108): Delete (unused). No tests reference it.
- `find_references truncation` (mcp/code_search.py:1372-1375): Return all with a count, or derive from a token budget.
- `find_by_pattern max_results default` (mcp/code_search.py:1414): Derive from a token budget or measure.
- `read_file_range default span` (mcp/code_search.py:1491): Require end, or derive from a token budget.
- `near-miss path suggestions` (mcp/code_search.py:1511): Low impact; show all or keep with a count.
- `summarize_text max_words default` (mcp/code_search.py:1561): Require max_words or measure.
- `run_check output trim` (mcp/code_search.py:1700-1701): Derive from a token budget. No tests reference it.
- `max_attempts` (mcp/jobs.py:153,172): Keep retries explicit per handler (Permanent vs transient). test_worker/test_assist reference max_attempts.
- `MAX_CHUNKS` (mcp/worker.py:95): Extract all chunks (the gpu lane already serialises), or record the truncation.
- `EXTRACT_MAX_TOKENS` (mcp/worker.py:96): Use tiers.budget like other internal callers. No tests reference it.
- `MIN_EVIDENCE_CHARS` (mcp/worker.py:99,371): Enforce the prompt's stated floor or measure false accepts. No tests reference it.
- `EXTRACT_SYSTEM prompt` (mcp/worker.py:309-322): Move into skill_prompts.py (versioned, pinned) and justify or drop the bounds.
- `extraction temperature` (mcp/worker.py:343): Omit temperature (vendor sampling). No tests reference it.
- `assist temperature / ASSIST_MAX_TOKENS` (mcp/worker.py:592,745): Omit temperature; budget via tiers.budget.
- `ASSIST_HEAD/TAIL/KEYWORD_LINES, LICENCE_FILE_CHARS` (mcp/worker.py:593-596): Derive from a token budget or measure licence recall.
- `corpus truncation` (mcp/corpus.py:88,198,206): Store full text within the 'caller code is hashed' rule, or derive the caps.
- `MAX_PACKAGES` (mcp/packages.py:42): Consult every imported+indexed package, or measure. No tests reference it.
- `Laya zero-shot gate` (mcp/laya_service.py:206): Delete with the Laya retirement (docs/E1.md s9).
- `dialectic/session margin floor` (mcp/dialectic.py:105; mcp/session.py:94): Delete (no importers).
- `--ignore-rules` (bench/octopus/run.py:450): Write a justification (hygiene: no stray rules or memory) or drop it; the operator's goal is daily-work realism.
- `browser-cdp toolset off` (bench/octopus/toolset_arms.py:169): Decide on the tools' own merit; if kept off, record why.

## Added 2026-09-27: server-tool recall and the work's own evidence

No number was added (docs/SKILL-FACTORY.md "The work's own evidence",
"Server-tool recall"). The rules and texts, classified:

| what | where | class | basis |
|---|---|---|---|
| once per key per conversation (think:<pkg>, scratch:<dir>, piece:<dir>, plan_done:<plan>, implement:<n>) | skill_select.tool_recall | A | the operator's task, 2026-09-27: "Each fires once per distinct trigger key per conversation (pattern key: kind + package or directory)" |
| TOOL_RECALL_* wording | skill_prompts (craft/3) | A-flag | the direction is the operator's ("pattern matching and injection"; "Remember (server tool ...)"); the sentences are ours, unmeasured |
| a package.json VERSION grep is not a probe | skill_select.probed_packages | A-flag | ours: the operator listed package.json among the probes; a version-only read checks what is installed, it does not learn how the package works |
| dependency paths (node_modules/, site-packages/, dist-packages/), temp/tmp-named files, manifest/config file names | skill_select `_DEP_SEG`, `_TMP_NAME`, `_CONFIG_FILE` | B | each ecosystem's own layout and naming (correctness: what is and is not the project's) |
| BUILT_ON (typegpu -> webgpu; r3f -> react, threejs) | skill_select | B | each row quotes the package's readme |
| route.work_intent lexicons (the pro-verbs, openers, information verbs, the desire-noun exclusions) | mcp/route.py | C (in-sample) | bench/skills/work_intent.jsonl, n=99, 99/99 after one adjustment (first pass 93/94); the verb sets are the router's own (`_WRITE`, selection's act verbs less `run`, `_EDIT_VERB`) |

## Decider legacy readout (removal list, 2026-09-29)

Operator, 2026-09-29: "look to the Jev api for the answer, they got it
right" and "Stop keeping old shit behind switches". The decider
(mcp/decider_bonsai.py, mcp/decide_turn.py) now answers in Jev's shape
(docs/JJAVA.md).

**Removed 2026-09-29 (DONE):** the decider's built-in ABSTAIN
(`abstain`, `abstain_why`, `thresholds` in every answer and log row),
`decider_bonsai.params_for`, `PARAMS` / `index/decider/params.json` /
`YAMADORI_DECIDER_PARAMS`, `PARAM_NAMES`, the params cache, and the
`values` of `q_score` (Jev numbers levels 0..k). Callers own thresholds:
`decide_turn.THRESHOLDS`, EMPTY until an owner accepts a row that
bench/decider/tune.py proposes from labelled decisions.

**Pending removal, after ONE GPU side-by-side** (bench/decider/
legacy_vs_typed.py; the only reason these still exist):
- `YAMADORI_DECIDER_READOUT=legacy` and `decide_turn.READOUT` /
  `typed_readout()` -- the switch itself.
- `decide_turn.FORM`, `Turn.ask`, `Turn._prime`, `Turn._cf`,
  `Turn._restricted_cf`, `Turn._id_prior`, `Turn.prime_choose`,
  `_rotation`, `prime_priors` / `prime_priors_background` / `prime_for` /
  `PRIME_COUNTS` / `PRIME_POLL_S`, the legacy branches of `facts()`,
  `confirm_packages()`, `pick()`, `choose()`, `build_intent()`, and
  `CONTEXTUAL`'s legacy uses (the typed package veto keeps its content-free
  read).
- `decider_bonsai.yes_no`, `choice` (as a public question form; `read`
  still renders through it), `as_choice`, `orders_for`, `average`, and the
  `{kind: yes_no | choice}` path of `decide()`; bench/decider/
  bonsai_decider.py's parts that call them are re-pointed at the typed API
  or retired with them.

| what | where | class | basis |
|---|---|---|---|
| confidence = clamp((n x p_max - 1)/(n - 1), 0, 1) | decider_bonsai.confidence | B | the vendor's own definition: docs.typesafe.ai /confidence (the three-option form in the prose, the n-option form in its demo code), checked against 17 response examples in the docs (mcp/test_decider_bonsai.py [confidence]); two /primitives/score widget examples do not fit and are recorded |
| score = sum of level number x probability, levels 0..k | decider_bonsai.read | B | docs.typesafe.ai /primitives/score |
| TIE_BAND 0.0034 (a diagnostic now: `diagnostics.tie`) | decider_bonsai | C | bench/decider/bonsai_decider.py `batching`, n=100 questions, one run |
| THRESHOLDS (empty) | decide_turn | A | operator, 2026-09-29: callers own thresholds, none ship untuned |
| the tuner's significance (Wilson lower bound at ALPHA 0.05) | bench/decider/tune.py | A | package_eval.ALPHA (operator, 2026-09-27: keep ours) |
| the tuner's 10 equal-width bins | bench/decider/tune.py | B | JevBench metrics.ece_top_label (the benchmark the field reports) |

## Other findings (not constants, found on the way)

**Bugs and stale code**
- **Truncated state is silently lost.** FIXED 2026-09-27 (the state is
  saved whole; test_nebari `test_a_large_state_is_saved_whole`). In `nebari.py:170`,
  `json.dumps(state)[:200000]` truncates the JSON text, so `load()` fails to
  parse it and silently returns `{}`.
- **config.template.yaml is stale.** It has `-c 262144`, a q4_0 cache,
  `--kv-mean-center` and no `${server_nudge}`; the live config.yaml runs
  `-c 181248` with q8_0.
- **Stale fallbacks.** `budget.pool_size` falls back to 131072 and
  `tiers._shares` to 163840, both from before 181,248.
  `shomen._helper_budget` falls back to 61440 when budget.py raises, but to
  49152 when it returns 0.
- **`admission.MAIN_LANES` 2 oversubscribes.** The module's own arithmetic
  gives one main request at full budget.

**Out-of-date documentation**
- AGENTS.md says `YAMADORI_E1` is off by default; scripts/watchdog.ps1 sets
  it to 1.
- FIXED 2026-09-27 (deep.py docstring; AGENTS.md's kickoff row). deep.py:41-42 and AGENTS.md said a user turn after a finished answer is a
  kickoff; the code plans only the initial prompt (operator, 2026-09-27).

**Rules broken or unsupported**
- **Private media used as data.** `images.REAL_ALPHA_*` thresholds were
  chosen from `index/media` (n=55), the operator's private media. The
  private-media rule says never to use it as test material.
- **The repeat breaker's evidence is not this model's.** Its guidance
  wording (now removed) cited a margin of 0.288 vs 0.493, which Laya
  measured, not the 27B.
- **Temperatures other than the vendor's 1.0.** The skill_select fallback
  (0.0), the skill pipeline stages (0.0/0.2), worker extraction (0.2), the
  worker assist (0.0) and the investigate hop (0.2, overridden to 1.0 by
  tiers.enforce_sampling) all depart from it. The operator's "keep temp at
  1.0" (2026-09-25) was about main; whether it covers internal calls is his
  to say.

**Benchmark and watchdog**
- **GPU gap in Octopus runs.** The runner checks that the slots are idle
  only at start and never calls `jobs.pause`, so worker GPU-lane jobs can
  start in the middle of a 7 h run.
- **Harness rules switched off.** `bench/octopus/run.py` passes
  `--ignore-rules`, which stops Hermes from loading AGENTS.md, SOUL.md,
  memory and preloaded skills. No rationale is written anywhere. It may
  guard against the answer-key leak (#42); the operator should decide,
  given the daily-work goal.
- **A budget number reaches the model.** `--run-budget 25200` (7 h) was the
  coordinator's number. Hermes sends a wrap-up notice at 80% of it.
- **Watchdog.** The 5-minute interval, 10-minute cooldown and 15 s timeout
  are B-value-arbitrary. The two-strike rule and the 10 s slot-progress read
  are B, each traced to a logged incident.

**Dead code**
- `proxy.PREAMBLE`/`preamble_for`: `resolve_repo` always returns None.
  DELETED 2026-09-27.
- `fanout.break_tie`/`_choice_averaged`, and the unused `fanout.VARIANTS`
  rows.
- `tool_shim.py`, `schema_fill.py`, `detect.py`,
  `streaming.stream_hop`/`stream_upstream`, `domains.strong_evidence`.
- `laya_router`, `dialectic`, `session`.
- (`shomen.T_INVESTIGATE`/`T_ANSWERED` and `route_in`/`distil`/`MARGIN_GATE`
  were DELETED 2026-09-27.) `code_search.WEAK_SIM`, `shomen.T_INVESTIGATE`/`T_ANSWERED`,
  `route_in`/`distil`/`MARGIN_GATE`.

**Old conversations.** Text removed on 2026-09-27 still replays in old
conversations: `ledger_restore` replays stored bytes, so a conversation that
already carried the rings closing line or a repeat-guidance line keeps it
until it ends. New conversations never get it.

## The full inventory

One row per item. The "action" column says what was done or what remains.


### mcp/proxy.py

| name | file:line | value | what it does | class | evidence or derivation | action |
|---|---|---|---|---|---|---|
| FANOUT_HEARTBEAT / HEARTBEAT | mcp/proxy.py:100,5311 | 5 s (env YAMADORI_FANOUT_HEARTBEAT) | Empty-delta interval while the second brain or a long job runs on a stream | B-value-arbitrary | Existence forced: Hermes kills a stream silent for 900 s (agent/chat_completion_helpers.py _local_stream_stale_timeout_default, cited proxy.py:4411-4418; LOG #39). The 5 s value is unmeasured; anything well under 900 s works. | keep (the bound is forced); derive the value |
| tool-call argument beat (#44) | mcp/proxy.py:529-543 | one beat per HEARTBEAT while tool-call deltas arrive | Streams an empty delta while a client call's arguments are written | B | LOG #44: a 40-min silent tool call hit Hermes' 900 s stale kill and orphaned the lanes; an empty delta is a chunk the OpenAI SDK counts. | keep |
| MIN_BUDGET | mcp/proxy.py:120 | tiers.A_MIN (2048) | Legacy alias of the answer floor | B-value-arbitrary | L115-119: "SUPERSEDED 2026-09-22 ... Kept as a name for anything that still reads it". Value is tiers.A_MIN (audited in tiers.py). | keep (the bound is forced); derive the value |
| ADDENDUM (static addendum text) | mcp/proxy.py:122-152 | 4-row table + "Text that opens with these phrases was written for you… | Appended to the client's system text at high/xhigh/max | A-flag | Operator 2026-09-24T13:47 "Can't we manage this all with a system prompt addendum"; plan decision 5 approved 14:28 ("Approved"). Wording ours: "Its wording is a CHOICE; nothing has measured it" (L132). | keep; operator to confirm the value/wording |
| ADDENDUM_THINK_ROW | mcp/proxy.py:154-162 | one table row (stuck / unsure of API / fix failed twice / still broke… | Tells main when to call think_deeply | A-flag | Operator 2026-09-24T14:45 "maybe deep thinking is in our system prompt ... how do we ensure the model knows it can do deep thinking and when to choose it?" Trigger wording ours ("Its wording is a CHOICE", L157). | keep; operator to confirm the value/wording |
| addendum gate (repair_on) | mcp/proxy.py:2685-2690 | high, xhigh, max | Only adds the addendum where every row is true | A | SELF-IMPROVEMENT-PLAN.md:137 decision 5 "added only where every row of it is true"; operator answer 2026-09-23T12:32 repair "On at high and max". | keep |
| upstream generation timeout | mcp/proxy.py:225,271,408 | 3600 s | Read timeout of one streamed upstream generation | B-value-arbitrary | A socket timeout must exist; a correct 447 s generation is documented (L233). 3600 is not derived. | keep (the bound is forced); derive the value |
| retry once on empty drop | mcp/proxy.py:598-611 | retries=1 | Re-sends a request that dropped before any byte arrived | B-value-arbitrary | L599-603: nothing lost, prefix cached; "exactly one ... retrying a fault in a loop turns one bad request into sustained load". The count 1 is not measured. | keep (the bound is forced); derive the value |
| upstream error excerpt caps | mcp/proxy.py:366,373,383-387,405,497 | 8192 read / 300 / 500 / 400 chars | Truncates upstream error bodies in logs and error objects | B-value-arbitrary | Log/error-size bounds; no effect on generation. | keep (the bound is forced); derive the value |
| dropped-connection marker | mcp/proxy.py:617-621,6938-6944 | "[the connection to the model dropped after Ns ...]" + finish_reason… | Returns the partial answer with a text marker instead of a 502 | B-value-arbitrary | L250-256: partial beats a 502 (LiveCodeBench lost ~60% to reaper drops; no script cited). Wording ours; "incomplete" is not an OpenAI finish_reason. | keep (the bound is forced); derive the value |
| TOOL_OVERLAPS | mcp/proxy.py:713-746 | 3 rows (vision_analyze; image_generation etc.; task/multi_agent_v1) | Withholds our tool when the harness has one answering the same question | A | Operator 2026-09-27T08:55 "Make sure proxy tools do not conflict with harness tools". Rows cite harness sources. | keep |
| _norm_tool | mcp/proxy.py:749-751 | case, -/_ to _, trailing s dropped if len>3 | Same-name detection for the conflict rule | A-flag | Serves the 2026-09-27 conflict decision; the normalisation rules are ours. | keep; operator to confirm the value/wording |
| PREAMBLE / preamble_for | mcp/proxy.py:98,943-993,8208-8217,8339-8347 | env YAMADORI_PREAMBLE=1 | First-answer status line (indexing state, available checks) prepended to content | D | No evidence for the text. DEAD: resolve_repo always returns (None, False, "none") since 2026-09-22 (L1548), so preamble_for returns "" every time. | DONE 2026-09-27: REMOVED: PREAMBLE, preamble_for, available_checks, both proxy call sites, the server.py call and every test's `proxy.PREAMBLE = False` (10 files). Behaviour unchanged (it never wrote anything). |
| _EMPTY_AGAIN | mcp/proxy.py:1017-1023,1227-1230 | fixed "No results -- this exact search was already run..." | An identical empty search is not re-run; the cached miss is returned | B | Idempotence: the index does not change within a turn (L1219-1223; "cost about 47 seconds each of the eleven times", observation, no script). Wording ours. test_repeats.py. | keep |
| turn.guidance appended to tool results | mcp/proxy.py:1230,1405,1473-1480 | text from repeats.Turn.guidance | Adds "what has NOT been tried" to second-brain tool results | D | Rationale cites "margin 0.288 ... 0.493" of a decision model (L1473-1477); no script or n cited here. Text aimed at the model's retry behaviour. Text lives in repeats.py (audit there). | DONE 2026-09-27: the three proxy call sites deleted, and repeats.Turn.guidance itself. |
| repeats.cap_tool_result (call site) | mcp/proxy.py:6827 | repeats.py value | Caps our tool results on main's hidden hops | see repeats.py | Call site only; audit the cap in repeats.py. test_stream.py. | keep |
| package-search result truncations | mcp/proxy.py:1157,1185,1194 | 1500 / 3000 chars | Bounds library-search text returned to the second brain | B-value-arbitrary | Tool output must fit the helper share (HELPER_TOKENS 49,152); the specific sizes are not derived. | keep (the bound is forced); derive the value |
| package-miss return text | mcp/proxy.py:1186-1197 | "None matched in any library ... pass its name as glob" | What a search matching nothing in any held library returns | B-value-arbitrary | A tool must return something (L1482-1497: empty results caused 12 repeats over 842 s). "Failure returns carry the next step" is AGENTS' rule, not an operator quote. Wording ours. | keep (the bound is forced); derive the value |
| _route_package_glob | mcp/proxy.py:1026-1054 | rule | A glob naming a held package searches only that package | B | Correctness: package indexes store paths as src/..., so glob "typegpu@0.12.5/data" matched nothing in any index (typegpu task, 12-hop loop, L1030-1037). | keep |
| question minimum length | mcp/proxy.py:1313,4351,4645 | 8 chars (selection.MIN_QUESTION_CHARS) | Refuses delegate / think_deeply / a deep run on a near-empty question | B-value-arbitrary | Validate before committing the one helper lane (L1302-1311: a test launched an investigation into ""). 8 is not derived. | keep (the bound is forced); derive the value |
| delegate_investigation cost trailer | mcp/proxy.py:1363-1370 | "[thought about deeply: N tool calls, T tokens ... Trace handle H.]" | Appends the run's cost to the delegate tool result | B-value-arbitrary | Benchmark arm, off by default (YAMADORI_DELEGATE_TOOL=0, L681); reports cost for context-economy measurement (L1363-1365). | keep (the bound is forced); derive the value |
| COMPACTION_LINK_SECONDS | mcp/proxy.py:1855,1911,1975 | 1800 s (env YAMADORI_COMPACTION_LINK_S) | Links a new-key request to the account's last conversation by recency | D | A recency guess; LOG #38: a 2,903 s turn defeated it. Operator 2026-09-25T15:14: "we can't assume a new session is a resumable one". Explicit ids, mapped lineage and summary probes exist beside it. | DONE 2026-09-27: REMOVED with the recency guess: `_note_compaction` records only a MAPPED compaction; `_continue_after_compaction` links only a continuation that CARRIES the summary (explicit ids resolve before it). An unmapped compaction's continuation is a new conversation. test_utility updated (3 tests). |
| summary probes | mcp/proxy.py:1862-1896 | PROBE_CHARS 96, PROBE_MIN_SUMMARY 200, windows at 0.25/0.5/0.75 | Recognises a continuation that carries the compaction summary | B-value-arbitrary | Content matching needs some window; sizes neither derived nor measured. No tests reference them. | keep (the bound is forced); derive the value |
| utility call -> tier minimal | mcp/proxy.py:2588-2596 | tier minimal (thinking off, instruct sampling) | Client side calls get the bare model | A-flag | Comment claims "(operator, 2026-09-23)"; no matching operator message found (2026-09-23T15:27 defines the minimal tier, not side-call routing). Evidence: one probe 1.97 s vs 409 s, n=1, no script. | keep; operator to confirm the value/wording |
| utility max_tokens room | mcp/proxy.py:3089-3100 | main share - prompt - MIN_THINKING - 1, floor A_MIN | Answer allowance for a side call that set none | B | Derived from the KV main share (budget.py) so a thinking-off summary is not cut at A_MIN. | keep |
| thinking cap on every non-utility turn (call site) | mcp/proxy.py:2963-2995 | tiers.USER_TURN_THINKING 12288 / AGENT_STEP_THINKING 6144 | step_cap passed to tiers.apply | A-flag | Operator 2026-09-25T11:48 "we may need to limit thinking a bit more ... this model overthinks pretty quickly". Numbers ours (tiers.py). Contrast 2026-09-23T10:51 "Why do we keep tightening them down?" | keep; operator to confirm the value/wording |
| by-result step caps (call site) | mcp/proxy.py:2980-2984 | tiers AGENT_STEP_THINKING_READ 2048 / _ERROR 6144 | Caps thinking by the kind of result an agent step answers | D | LOG #53 (Octopus v0e, n=1 run), "CHOICES"; coordinator: stays D. | DONE 2026-09-27: branch deleted: every agent step thinks at most AGENT_STEP_THINKING. tiers.AGENT_STEP_THINKING_READ / _ERROR, tiers.step_thinking(), the step_thinking switch, progress.step_kind and SEARCH_NAMES deleted. test_ledger [step], test_progress updated. |
| agent-step nudge (call site) | mcp/proxy.py:2985-2986 | tiers.AGENT_STEP_NUDGE_MESSAGE | Action-naming reasoning nudge on agent steps | A | Operator answer 2026-09-26T20:48: "Approve as written (Recommended)". | keep |
| DEFINITIONS_MAX_NAMES | mcp/proxy.py:3425,3457 | 3 | Names per library-definitions injection | A-flag | Feature: PLAN decision 1 (2026-09-24, approved 14:28) "An UNMEASURED choice". Cap ours: "The caps are choices" (L3424). AGENTS.md:430. | keep; operator to confirm the value/wording |
| DEFINITIONS_MAX_CHARS_EACH | mcp/proxy.py:3426,3471 | 1200 chars | Per-definition char cap | A-flag | Same as above: feature approved, value ours. | keep; operator to confirm the value/wording |
| DEFINITIONS_MAX_CHARS | mcp/proxy.py:3427,3474 | 3000 chars | Total definitions injection cap | A-flag | AGENTS.md:430 "capped at 3 names and 3,000 characters (choices)". | keep; operator to confirm the value/wording |
| DEFINITIONS_HEAD | mcp/proxy.py:3428-3430 | header text "(cite them as path:line)" | Header of the definitions tail on the user turn | A-flag | Feature approved (decision 1); wording ours. test_ledger.py. | keep; operator to confirm the value/wording |
| LIBRARY USE feature | mcp/proxy.py:2831-2842,2881-2951,3477-3496 | on where the tier has retrieval and deep thinking did not run | Injects held-package definitions/overviews on user turns and tool results | A-flag | Claimed "coordinator/operator, 2026-09-24; an UNMEASURED choice"; no operator message found. Origin LOG #19 (Octopus pilot: three-flatland, @pmndrs/glyph). | keep; operator to confirm the value/wording |
| USE_MAX_CHARS | mcp/proxy.py:3497,3702 | 3000 chars | One library-use injection cap | D | "Every cap is a choice" (L3495-3496). | DONE 2026-09-27: REMOVED. Replaced by a B bound: `proxy._injection_room` (main share - the request's high estimate - A_MIN - MIN_THINKING, x3 chars/token, less what else was decided for the message). A definition that does not fit is left uncovered (`did_not_fit`), tried again later. |
| USE_MAX_NAMES | mcp/proxy.py:3498,3660 | 4 | Definitions per library-use injection | D | "Every cap is a choice" (L3495-3496). | DONE 2026-09-27: REMOVED: every imported name of a held package is looked up; the window room is the one bound. |
| USE_MAX_CHARS_EACH | mcp/proxy.py:3499,3681 | 1000 chars | Per-definition cap in library use | D | "Every cap is a choice" (L3495-3496). | DONE 2026-09-27: REMOVED: each definition whole. |
| USE_OVERVIEW_ITEMS + overview filters | mcp/proxy.py:3500,3552-3571 | 20 exports; lines cut at 140; test/spec/internal paths skipped | Package overview when no names are used yet | D | "Every cap is a choice" (L3495-3496); filters hand-written. | DONE 2026-09-27: REMOVED with the package OVERVIEW itself (it cannot exist without a picked number): a package used with no names yet gets nothing until a name is used. test_ledger updated. |
| USE_CONVERSATION_MAX_CHARS | mcp/proxy.py:3501,3643,3702 | 12000 chars | Library-use budget per conversation | D | "Every cap is a choice" (L3495-3496). | DONE 2026-09-27: REMOVED: each name is injected once (the covered set), bounded by the window room. |
| USE_SCAN_CHARS | mcp/proxy.py:3504,3526,3542,3585 | 20000 chars | Only the head of each message/file is parsed for imports | B-value-arbitrary | Performance bound: "a harness can return a megabyte bundle" and imports sit at the top (L3502-3503). Value not derived. | keep (the bound is forced); derive the value |
| USE_HEAD | mcp/proxy.py:3505-3506 | header text | Header of the library-use injection | A-flag | Wording of an (unconfirmed) feature; ours. | keep; operator to confirm the value/wording |
| held_version fallback | mcp/proxy.py:3591-3614 | exact, then same major.minor, else newest (labelled) | Picks which held index serves a stated version | B-value-arbitrary | Correctness: serve the conversation's own version (pre-deploy review 2026-09-24); the major.minor fallback order is ours. | keep (the bound is forced); derive the value |
| _work_log_block | mcp/proxy.py:3711-3725 | rings.read(limit=40) + header "Work log of this conversation before i… | Work log injected on the first user turn after a compaction | A-flag | AGENTS claims "(operator, 2026-09-24)" for the proxy writing the work log; no operator message found. limit=40 not derived. | keep; operator to confirm the value/wording |
| _log_turn contents | mcp/proxy.py:8047-8089 | up to 8 calls, 160 chars, kinds did/check/learned/decided | What the proxy writes to the work log | A-flag | L8052: "What is logged is a CHOICE". Feature claim as above. | keep; operator to confirm the value/wording |
| work-log reinject after same-key compaction | mcp/proxy.py:2749-2759 | switch work_log_reinject | Re-injects the work log after a compaction that kept the key | A-flag | LOG #54: kept 2026-09-27 as "a mechanic of the work log, not steering" (agent's classification, no operator quote). | keep; operator to confirm the value/wording |
| retryable injection decisions | mcp/proxy.py:2762-2780,3232-3247 | rule | A part left empty because something was unavailable is decided again | B | Correctness: live gate 2026-09-24, a decision made while embeddings were unloaded "was replayed, forever, as final". | keep |
| per-turn injection decided once, replayed (ledger) | mcp/proxy.py:2039-2099,2726-2880 | rule | Injections recorded and re-added byte for byte | A | Operator 2026-09-24T14:05 "ensure data is folded in and memory and cache are coherent"; Phase 0.5 approved 14:28. STEP 0 probes (scratchpad, n=2, not in repo). | keep |
| reasoning pass-through | mcp/proxy.py:2061-2078,2281-2289 | rule | Past reasoning is not restored; the client's echo passes as sent | A | Operator 2026-09-24T14:05 "Then why strip reasoning, this is a harness decision, not our decision". | keep |
| sort_call_arguments | mcp/proxy.py:2112-2160 | sorted keys | Renders tool-call arguments with sorted keys | B | Correctness: Hermes re-serialises with sort_keys (conversation_loop.py); the cache diverged (v0e step 6, reused 16187 of 25487). | keep |
| session id carried in tool-call ids | mcp/proxy.py:1797-1824 | call_<id>_<8 hex> | Carries our conversation id through the client | A | Operator 2026-09-25T19:12 "tool-call-id session carrier = yes". | keep |
| answer record | mcp/proxy.py:1737-1794 | rule | Resolves a conversation from our own earlier answer text | D | AGENTS: "a deviation from the brief, for the operator to confirm"; operator 2026-09-25T15:14 "we can't assume a new session is a resumable one". Not confirmed. | LEFT 2026-09-27 (correctness; operator decides): not a number but the only carrier a text-only conversation has. Removing it makes every request of a client that sends no key or header a NEW conversation -- a new slot pin, a new chain salt so turn 1's recorded injection is not replayed, the yama_think_deeply offer re-decided -- which breaks "one model, one cache" (A, operator 2026-09-24). AGENTS.md session paragraph says so. |
| session line on compaction summaries only | mcp/proxy.py:3048-3057,3834-3845 | "yamadori session <id>" | Only a compaction summary we write opens with the id line | A-flag | Claimed "(operator, 2026-09-25: no visible line in answers)"; the removal from answers is operator-driven, the summary line itself is ours. | keep; operator to confirm the value/wording |
| craft offer + recall_craft on main | mcp/proxy.py:2691-2724 | kept per conversation | Craft index at the end of the system text and the recall tool | A | Operator 2026-09-27T08:51 "progress disclosure is fine"; 08:55 "it is not a skill it is a craft". | keep |
| step skills on tool results | mcp/proxy.py:2905-2951 | skill_select.attach_step | Skills appended to the tool result an agent step ends on | A | Operator 2026-09-27T08:52 "memory recall skill markers, and it works"; 08:46 on injecting "at the highest attention when it matters most". | keep |
| fix-up scope project-only (call site) | mcp/proxy.py:7296-7300 | switch fixup_project_only | No repair round for /tmp and scratch files | D | LOG #56, Octopus v0b/v0e observations; scratch-name table; no operator quote. | DONE 2026-09-27: call site, tool_code.scope, its note branch and scope_skipped deleted; switch fixup_project_only gone. test_ledger [fixup] rewritten. progress._TEMP / SCRATCH_NAMES / dot-file rule stay: label rule 2 reads them (unassigned). |
| progress.learn / project inference | mcp/proxy.py:2886-2904,3069-3082 | state | Infers project root/named files for fix-up scope and deep's label rule | D | LOG #52/#56 (Octopus-derived); no operator quote. | LEFT 2026-09-27: its #56 consumer (the fix-up scope) is removed, but label rule 2 (`helped_needs_change`, unassigned: the operator's call) still reads the project; it goes with that switch. |
| COUNT_TIMEOUT | mcp/proxy.py:3969 | 30 s (env YAMADORI_COUNT_TIMEOUT) | Timeout of the apply-template + tokenize count | B-value-arbitrary | A timeout must exist; on failure the request is sent and llama-server's own limit decides (L4043-4049). | keep (the bound is forced); derive the value |
| generation_floor | mcp/proxy.py:3937-3975 | A_MIN + MIN_THINKING = 3072 (2048 without thinking) | A client prompt leaving less than this is refused with 400 | B-value-arbitrary | Refusing an over-long prompt is the OpenAI contract (context_length_exceeded; C1 approved 2026-09-25T21:02 "Before v0 fixes should land."). Floor "A CHOICE" (L3956). AGENTS.md:1010. | keep (the bound is forced); derive the value |
| window_limit | mcp/proxy.py:3978-3982 | main share (or compaction window) | The enforced context limit | B | budget.budgets()["main"] from -c and HELPER_TOKENS; the same as /v1/models context_length. | keep |
| high_estimate | mcp/proxy.py:3985-4000 | chars/3 + half the extra UTF-8 bytes | Decides whether the exact token count is needed | B-value-arbitrary | Only gates the exact count; fallback is llama-server's exceed_context_size_error mapped to 400 (L3966-3968). chars/3 is not measured here. | keep (the bound is forced); derive the value |
| fit_window | mcp/proxy.py:4059-4075 | thinking = min(budget, max(room-answer, min(MIN_THINKING, room//2))) | Cuts hop 0's max_tokens to the window | B-value-arbitrary | max_tokens must not exceed the window (B); the room//2 split is ours. | keep (the bound is forced); derive the value |
| context_full | mcp/proxy.py:4078-4110 | prompt + answer + MIN_THINKING >= share | Lands our tool loop when the share is full | B | KV share per role (budget.py); OUR hops only, the client's request is never landed (C1). | keep |
| tool-turn cap (call site) | mcp/proxy.py:4113-4127,6604-6607 | tiers.tool_turn_limit 10 (20 at max) | Lands our tool loop after N tool turns | A | Operator answer 2026-09-23T14:55 "Both, 10 each"; 14:54 "We should for sure follow the agenticMaxTurns then"; 15:24 "let it do 20 max instead maybe". | keep |
| LANDING_PROMPT | mcp/proxy.py:3930-3934,4171 | user message "Stop searching and answer now ... that is expected here… | Appended user turn when our loop lands | A-flag | Landing follows the operator's cap; wording ours and steering ("answer from your own knowledge -- that is expected here"), unmeasured. AGENTS.md:816. | keep; operator to confirm the value/wording |
| HOLD_CONTENT_CHARS | mcp/proxy.py:4130-4133,6669 | 400 chars | Content held before a hop is treated as the answer and streamed | B-value-arbitrary | CHANNEL ORDER needs a point where a hop's content counts as the answer (a preface vs an answer); 400 is not measured. test_stream.py. | keep (the bound is forced); derive the value |
| FINDINGS_HEAD / REASONING_HEAD | mcp/proxy.py:4179-4193 | two header texts | Label on the hand-off prefilled as main's reasoning | A-flag | Hand-off as reasoning: PLAN decision 2 (2026-09-24, approved). "(operator, 2026-09-23)" for the no-search label is claimed; no message found. Wording ours. test_stream, test_tools. | keep; operator to confirm the value/wording |
| FOLD_BACK_TAIL | mcp/proxy.py:4194-4214 | text with two prohibitions | Tail of the prefilled hand-off: the answer must state findings, never mention the hand-off | A-flag | Mechanism: PLAN decision 2. Wording "a choice, unmeasured beyond the live check" (L4201), from live gate 2026-09-24 (two runs). Own-machinery prompt. AGENTS.md:914. test_stream.py. | keep; operator to confirm the value/wording |
| FOLD_BACK_TAIL_MACHINE | mcp/proxy.py:4215-4232 | text | Tail for a machine-built hand-off | A-flag | "UNMEASURED WORDING" (L4222); deploy check 2026-09-26 handle d28941fb (n=1). AGENTS.md:933. test_stream.py. | keep; operator to confirm the value/wording |
| PLAN_TAIL | mcp/proxy.py:4233-4235 | text | Tail of the prefilled kickoff plan | A-flag | Plan job: operator 2026-09-27T12:48 "we should always give it a planning turn"; wording ours. | keep; operator to confirm the value/wording |
| FOLD_BACK_MIN_CHARS | mcp/proxy.py:4236-4240,7011-7017 | 200 chars | Logs a fold-back answer shorter than this | D | "200 is a choice: the failing answer had 72" (L4238-4239). Log/record only; the live suite asserts on it. | DONE 2026-09-27: REMOVED: x_yamadori.fold_back_answer is {chars_after_opening, opened}. test_stream updated; test_live_stack still reads fold_back_answer (the key stays). |
| _ABOVE_REF / _HIDDEN_REF | mcp/proxy.py:4241-4255,4288-4302 | regex (hand-off, investigation, "as noted above", ...) | Flags an answer that points at text the user never saw | D | Hand-written from two live-gate answers (L4241-4245); record only (x_yamadori.fold_back_answer). | DONE 2026-09-27: REMOVED with refers_to_hidden / refers_context. test_stream, test_tools updated. |
| dedup_opening | mcp/proxy.py:4258-4285,6671,6717,6883 | regex | Removes later copies of "After thinking deeply," from the model's answer | D | One live-gate run (2026-09-24, second run, L4270-4272); alters model output; not operator-requested. | DONE 2026-09-27: REMOVED (all three call sites and `_opening_repeats_removed`): the answer is delivered as written. test_stream updated. |
| THINK_CALLS_PER_REQUEST | mcp/proxy.py:4558,4669-4673 | 1 | At most one think_deeply run per request | B | One helper lane (HELPER_LANES=1); a double-run/recursion guard; AGENTS Phase 0.6 table "THINK_CALLS_PER_REQUEST = 1 per request". | keep |
| already_thought text | mcp/proxy.py:4561-4632 | refusal text; labels[:6], reads[:6]/[:12] | What a second think_deeply call is told | D | "UNMEASURED WORDING" (L4572); built from deploy check 2026-09-26 (n=1) to steer main ("make the next call now" / "answer now"). | DONE 2026-09-27: REPLACED by a factual return (AGENTS "Failure returns carry the next step"): situation (ran once, when, concluded or not, searches, where its hand-off/plan is), retryable false, remedy {agent: use the result already in this reply}. Removed: "answer now" / "make the next call now", "cite by path:line", labels[:6], reads[:6]/[:12]. test_stream updated. |
| think_deeply `tried` cap | mcp/proxy.py:4674-4678 | 2000 chars + "Already tried, and how it failed:" | Appends the call's `tried` argument to the question | B-value-arbitrary | The tool's own schema (AGENTS: "the question, and what was tried"); cap not derived. test_ledger.py. | keep (the bound is forced); derive the value |
| deep-thinking thread join | mcp/proxy.py:4437 | 5 s | Wait for the deep-thinking thread after its queue ends | B-value-arbitrary | Any short bound; the thread has already signalled completion. | keep (the bound is forced); derive the value |
| _research_context caps | mcp/proxy.py:4872-4898 | user_urls last 100; conversation last 200,000 chars; own_text last 40… | Context for read_web_page's leak check and seen-URL rule | B-value-arbitrary | Security (operator 2026-09-26T00:08 on query args) needs the conversation text; the bounds are memory caps, unmeasured. test_web_access.py. | keep (the bound is forced); derive the value |
| x_yamadori record caps | mcp/proxy.py:4913-4920,4982,5979,6176 | 20 / 60 / 20 / 8 lines / 200 chars | Sizes of telemetry records | B-value-arbitrary | Telemetry bounds; no effect on the model. | keep (the bound is forced); derive the value |
| fan-out delivery rule | mcp/proxy.py:4814-4833 | DELIVERED_SELECTIONS code_grade, code_medoid | Only a graded code winner replaces the answer; prose keeps the original | A | Operator answer 2026-09-23T12:46 "Yes, fan out on code ... best parsing/consensus answer delivered"; prose: "no measured basis for swapping prose" (L4829). | keep |
| fan-out fold-back text | mcp/proxy.py:7377-7435 | "Compared two approaches: <why>." / prose "... makes these points tha… | Folds B/C back into main's turn | A-flag | Phrases: operator 2026-09-24T14:05 key phrases; weigh turn -> prefill "(operator yes)" (PLAN:70). Sentence wording ours. | keep; operator to confirm the value/wording |
| Verified / Repaired / Checked answer notes | mcp/proxy.py:7359-7365,7480-7507 | fixed note texts | Notes after an answer's code check / repair | A | PLAN decision 3 (2026-09-24, approved); operator 2026-09-24T14:05 "We can have our output say ... 'verified' 'repaired'". | keep |
| fold-back opening prefill | mcp/proxy.py:4466-4471,6773-6786 | seed line + "After thinking deeply," | Prefilled opening of main's answer after deep thinking | A | Operator 2026-09-24T14:28 "Today I was inspire by seed-word"; 14:05 "After thinking deeply, I decided ...". | keep |
| TEMPLATE_MARKERS scrub (our path) | mcp/proxy.py:5348-5410 | <think>, </think>, <\|im_start\|>, <\|im_end\|>, <tool_call>, </tool_… | Strips template tokens from text the proxy injects | B | Correctness: a </think> inside injected reasoning closes the served template's think block early (LOG #12). | keep |
| _StrayMarkers | mcp/proxy.py:5587-5809 | 4 markers; repeat-drop rule; markers in code kept | Removes stray template markers (and the repeat after them) from the model's answer | A | Operator 2026-09-25T19:05 "why not clean it up in the proxy?"; rate 7/40 and 3/40 (scratchpad replay_final.py, n=40, not in repo). test_stray_markers.py. | keep |
| REASONING_BEAT_S | mcp/proxy.py:5467 | 1.0 s | Throttle of heartbeats replacing reasoning after content began | B-value-arbitrary | Channel order forces reasoning suppression after content; 1 s not derived. | keep (the bound is forced); derive the value |
| image emitted on creation | mcp/proxy.py:5449-5466,6792-6810 | rule | Image markdown streamed as content the moment the tool returns | A | Operator 2026-09-25T16:05 "emit the image to the harness when they make it". test_image_emit.py. | keep |
| _ImageDedup | mcp/proxy.py:5472-5584 | alt text <=400 chars, 440-char look-back | Removes the model's own copy of an image already shown | A-flag | Follows from immediate emission (no double image); the rule and windows are ours. test_image_emit, test_stray_markers. | keep; operator to confirm the value/wording |
| DESCRIBE_LINE_CHARS | mcp/proxy.py:5469,6184-6219 | 100 chars | One reasoning line after describe_image | A-flag | Operator 2026-09-25T16:08 asked how describe shows; AGENTS claims "(operator 2026-09-25)" for the line; no quote for 100. test_image_emit.py. | keep; operator to confirm the value/wording |
| _describe_line screen | mcp/proxy.py:6199-6216 | visible limit 4000; tag regex 200 | Screens the excerpt shown to the user | B | Security: tool text shown to the user is screened (skill_screen); bounds not derived. | keep |
| CHANNEL ORDER (_Out) | mcp/proxy.py:6222-6350 | rule | No content before reasoning ends; later reasoning becomes heartbeats | B | Clients close the thinking block at the first content delta (live SSE diagnostic 2026-09-24; SELF-IMPROVEMENT-PLAN.md:23-26). | keep |
| _ToolMarkup + tool_markup_notice | mcp/proxy.py:5812-5979,6917-6937 | markup filter + "[no answer: the model tried to call a tool ... defec… | Removes written-out tool calls from a landed answer | D | One run, bench/voxel pagoda-r3f-1, 2026-09-27 (L5812-5815); no operator request. | DONE 2026-09-27: REMOVED (class, notice, _Out filter, x_yamadori.tool_markup): the landed answer is delivered as written. test_stray_markers section 7 rewritten (138/138); test_tools key set. |
| _ImitatedNotes + NOTE_HEAD_MAX | mcp/proxy.py:5982-6181,6508-6513,7045-7064 | note-head regex; 600 chars ("A choice") | Removes note-shaped lines the model writes in its own content | D | Claims "operator 2026-09-26: fixed before the next run" (L5983); no operator message found. Octopus v0b/v0e/v0f counts. Alters model output; the model imitates OUR notes. | DONE 2026-09-27: REMOVED (class, note_heads, _notes_record, _Out filter, x_yamadori.imitated_notes). mcp/test_imitated_notes.py RETIRED (deleted; it was untracked); test_stray_markers and test_tools updated. Our own note still bypasses the marker filter. |
| TurnRefused Retry-After | mcp/proxy.py:6366 | 60 s | Retry-After on a retryable refusal | B-value-arbitrary | A retryable 429 should carry one; 60 not derived. | keep (the bound is forced); derive the value |
| image guard gate (call site) | mcp/proxy.py:6627-6634 | off at minimal and low | Stops image data in tool arguments on main | A | Operator 2026-09-26T13:04 "we can't just harden the tool calls around image generation?"; minimal/low carry nothing of ours (2026-09-23T15:27). IMAGE_REGENERATIONS audited in tool_code.py. | keep |
| withheld-call defect notices | mcp/proxy.py:6899-6911 | "[no answer: the tool loop reached its breaker ...]" / "[... carried… | Content when a withheld call left nothing | D | No evidence; wording ours. An empty answer with finish_reason stop is protocol-valid. | DONE 2026-09-27: REMOVED: a withheld call of ours leaves empty content, finish_reason stop. test_stream's landing check updated. |
| _budget_notice / _budget_note | mcp/proxy.py:3895-3927,6945-6950 | "[no answer: ... token limit ...]" / "[answer cut off at the token li… | Text appended to content on finish_reason length | D | AGENTS.md:1000 rule is ours ("a budget event, never an answer"); no operator quote. finish_reason=length already carries the fact; the text enters the client's stored history. | DONE 2026-09-27: REMOVED: a `length` finish is finish_reason length with the content generated; no fan-out / code check on it. test_stream updated. |
| _empty_notice | mcp/proxy.py:5264-5286,6951-6957 | "[no answer: the model stopped (finish_reason=stop) ...]" | Text when the model wrote nothing | D | 2026-09-23 corpus observation; the rule is ours. | DONE 2026-09-27: REMOVED: blank content, finish_reason stop. test_utility updated. |
| REASONING_COUNT_TIMEOUT | mcp/proxy.py:7226 | 10 s | Tokenize timeout for usage.reasoning_tokens | B-value-arbitrary | Field absent on failure, never guessed. | keep (the bound is forced); derive the value |
| usage = final generation | mcp/proxy.py:7196-7282 | rule | usage reports the last main generation | B | OpenAI spec; Hermes reads usage.prompt_tokens as context size (context_compressor.py). | keep |
| WARM / END_OF_TURN | mcp/proxy.py:7563-7581 | on (env YAMADORI_WARM); "<\|im_end\|>" (env YAMADORI_END_OF_TURN) | Warms the slot with the delivered turn | A | Phase 0.5 (operator 2026-09-24, approved); STEP 0 probe (b) 1278/23 vs 1208/93 (scratchpad, n=2, not in repo). END_OF_TURN = the template's eos. | keep |
| _WARMS_DONE cap | mcp/proxy.py:7654 | 512 | Bound on warm records kept | B-value-arbitrary | Memory bound. | keep (the bound is forced); derive the value |
| warm request timeouts | mcp/proxy.py:7662,7990,7998,8010 | 120 / 120 / 600 / 30 s, each clipped to WARM_WAIT | Bounds of the warm's upstream calls | B-value-arbitrary | Must not outlive the waiter (L7972-7975); values not derived. | keep (the bound is forced); derive the value |
| WARM_WAIT | mcp/proxy.py:7710-7720 | 180 s (env YAMADORI_WARM_WAIT) | Next request waits this long for its pending warm | B-value-arbitrary | The race (#11) needs a bound; "a CHOICE: a warm of a 70k prompt re-read from zero is ~20-40 s" (L7716-7717, no script). test_ledger.py. | keep (the bound is forced); derive the value |
| WARM_HOLD_S | mcp/proxy.py:7721-7724 | 30 s (env YAMADORI_WARM_HOLD) | Abandons a warm hold nothing took over | B-value-arbitrary | "A breaker, not a measurement" (L7723). | keep (the bound is forced); derive the value |
| ECHO_WINDOW | mcp/proxy.py:7728-7758 | 8 observations | First-turn guess of whether the client echoes reasoning | D | "A choice: 8" (L7738); from one live-gate incident with two test clients on one key. | DONE 2026-09-27: REMOVED the 8-observation window. DEVIATION from the doc's two options, both of which break a tested path ("strips" breaks an echo-only account's first-turn warm, test_ledger [fixup echo] RENDER; "last observation" reproduces the mixed-account incident): the account keeps the SET of habits it was seen with, and a first turn guesses "echoes" only if it was never seen stripping. No number. test_ledger updated. |
| WARM_CHECKPOINT_SLACK | mcp/proxy.py:7761-7766 | 8 tokens | Tolerance before a warm is flagged SHORT | B-value-arbitrary | 4 from llama-server checkpoint_offsets {4+n_ubatch, 4} (server-context.cpp); the extra 4 is "a choice". Telemetry only. test_live_stack.py. | keep (the bound is forced); derive the value |
| CHECKPOINT_MIN_STEP | mcp/proxy.py:7821-7856 | 8192 (env YAMADORI_CHECKPOINT_MIN_STEP) | Expected warm reuse bound after a prefill | B | llama.cpp common.h default checkpoint_min_step; config.yaml does not set it (L7829-7831). Telemetry. test_ledger.py. | keep |
| idle-clear override only for test accounts | mcp/proxy.py:3119-3126 | rule | A header's idle threshold applies only to a test account's own conversations | B | Security: a request header must not clear other users' slots. | keep |
| compaction tool_choice none | mcp/proxy.py:3879-3885 | "none" | A compaction cannot call a tool | B | llama-server renders tools regardless of tool_choice, so the cached prefix is unchanged (compaction.py cites the lines). | keep |

### Second brain and triggers: shomen.py, deep.py, deep_learn.py, dash_deep.py

| name | file:line | value | what it does | class | evidence or derivation | action |
|---|---|---|---|---|---|---|
| struggle trigger (existence) | mcp/deep.py:18-34,1272 | on at xhigh/max | Run investigate before main when failures repeat in the conversation | A | operator 2026-09-24T14:45 (operator_messages.txt:1747): "deep thinking should have a trigger, like if an agent fails to get something write after several turns" | keep |
| STRUGGLE_THRESHOLD_DEFAULT | mcp/deep.py:107 | 3 | Signals in an episode needed to fire struggle | A-flag | Operator said "several turns" (1747); 3 is ours: SELF-IMPROVEMENT-PLAN.md:232 "Threshold 3 signals (a choice, set from dogfood data)" -- never set from data; deep.py:66 "NONE MEASURED". Number ours. | keep; operator to confirm the value/wording |
| BOUNDS struggle_threshold | mcp/deep.py:108 | (2, 6) | Range the idle learner may move the threshold within | A-flag | Operator required a self-improvement loop (1749 "so long as it also has self improvement loop"); "arm automatically within bounds" is our plan text (SELF-IMPROVEMENT-PLAN.md:245); bounds ours. | keep; operator to confirm the value/wording |
| STRUGGLE_WINDOW | mcp/deep.py:112 | 40 messages | How far back struggle signals are counted within an episode | D | No evidence; comment only. Not operator, not measured. | DONE 2026-09-27: REMOVED with its env var: signals count from the episode boundary (last run, or a compaction). |
| COOLDOWN_REQUESTS | mcp/deep.py:115 | 3 requests | Requests after any run before struggle may fire again | D | deep.py:66 "EVERY NUMBER HERE IS A CHOICE"; no quote, no script. | DONE 2026-09-27: REMOVED with its env var and the in_cooldown label; x_yamadori.deep.last_run {requests_since_run, boundary} replaces rec.cooldown (the DB column stays for old rows). |
| same_pattern suppression | mcp/deep.py:1276-1287 | subset rule | Hold back a struggle whose signals all repeat the last run's pattern | D | #45(d), built from Octopus v0e p2 replay (n=1 run); "one incident, one run" is our rule, not operator or measured. | DONE 2026-09-27: REMOVED with missed_same_pattern; a run is still judged by its pattern's recurrence (_recurrence, not_helped). |
| OUTCOME_REQUESTS | mcp/deep.py:117 | 5 | Later requests observed before a non-run row is labelled | A-flag | Loop is operator-required (1749); window ours ("a CHOICE"). | keep; operator to confirm the value/wording |
| HELPED_WINDOW | mcp/deep.py:123 | 20 | Observations without recurrence before a run is labelled helped | A-flag | Loop operator-required; 20 ours: deep.py:119 "(#45 (e); a CHOICE)"; motivated by 5 making helped automatic (v0e p2). | keep; operator to confirm the value/wording |
| LABEL_RULE 2 (helped needs a project write) | mcp/deep.py:136,1700-1717 | rule 2 | A run is helped only if a project file changed after it | A-flag | Label semantics ours (#52, OVERTHINKING.md remedy 7, n=1 run set); loop itself operator-required. | keep; operator to confirm the value/wording |
| DEFER_REQUESTS + one lane-wait per episode | mcp/deep.py:142,1219-1227 | 3 requests | After a busy-lane skip, no trigger fires for N requests | D | "pre-deploy review, 2026-09-24" (ours); value unmeasured. | DONE 2026-09-27: REMOVED (mark_deferred, waited_episode, the proxy call sites and lane_timeout plumbing). DEVIATION from the doc's "lane_timeout 0": a trigger waits the standard helper-lane wait (admission.WAIT_SECONDS, B) like every helper job, then is skipped for that request only -- lane_timeout 0 would skip the kickoff plan whenever another conversation holds the lane, against the operator's "always give it a planning turn" (A, 2026-09-27). |
| CHARS_PER_TOKEN | mcp/deep.py:144 | 4 | Estimate spec size in tokens (recorded only since #65) | B-value-arbitrary | Decides nothing now: kickoff size threshold removed 2026-09-27 (#65); only feeds the `tokens` record. | keep (the bound is forced); derive the value |
| MODEL_CUTOFF | mcp/deep.py:149 | 2025-12-31 | Training cutoff for the unseen-package area rule | B-value-arbitrary | Existence forced (a package first published after training cannot be known); value guessed: deep.py:146 "The model card states none ... a CHOICE". | keep (the bound is forced); derive the value |
| unseen: first published after cutoff | mcp/deep.py:896-898 | date compare | Area trigger for a package that did not exist at training time | B | Derivation: first publish > cutoff means no training data could contain it (given the cutoff). | keep |
| unseen: any prerelease | mcp/deep.py:847-849,899-900 | -alpha/-beta/-rc/-next/... | Flags every prerelease version as unseen regardless of its date | D | "THE RULE (a CHOICE; the coordinator's revision of 2026-09-24)" deep.py:872 -- not operator, not measured. | DONE 2026-09-27: REMOVED. The registry history stores stable releases only, so a prerelease's own date is unknown: a prerelease is judged by the same rules as any version. @react-three/fiber@10.0.0-alpha.5 and @react-three/drei@11.0.0-alpha.7 stay unseen, as new majors. |
| unseen: new major vs last pre-cutoff release | mcp/deep.py:901-909 | major compare | Flags a major above every major released before the cutoff | B | Derivation: a major higher than every pre-cutoff release was released after the cutoff. | keep |
| YAMADORI_UNSEEN/SEEN_PACKAGES | mcp/deep.py:890-893 | empty | Operator override list for the area rule | B | Plumbing hook; no values set. | keep |
| known-hard area trigger (existence) | mcp/deep.py:35-40,1329-1340 | once per package | Investigate an unseen held package the conversation uses | A | Operator 1747: "or if we know this is an area the agent could use an entire second context to solve the problem". | keep |
| escalate skills (frontmatter) | mcp/deep.py:1398-1425 | once per skill | A skill declaring escalate fires the area trigger | A-flag | Operator 1747 "this is like the skill trigger for deep thinking"; the frontmatter flag mechanism is ours. | keep; operator to confirm the value/wording |
| kickoff: every new task planned | mcp/deep.py:41-50,788-843 | no size threshold | Plan job before main on a new task | A | Operator 2026-09-27T12:48 (operator_messages.txt:2858): "Doesn't matter the length of the prompt we should always give it a planning turn." | keep |
| kickoff: only the initial prompt | mcp/deep.py:835-841 | follow-up after answer not planned | A follow-up after a finished answer gets no plan | A-flag | Operator 2026-09-27T13:22 (2874) asked "I thought it was only the initial prompt that got planning?" (a question, taken as decision). NOTE deep.py:41-42 docstring and AGENTS.md:524 still say "or a user turn after a finished answer" -- contradicts the code. | keep; operator to confirm the value/wording |
| kickoff exclusions (notice, compaction, tool-media) | mcp/deep.py:799-833 | rules | Harness notices, compaction continuations, tool-media turns are not new tasks | B | Correctness: these turns continue the task in flight (pre-deploy review 2026-09-24). | keep |
| priority order + one run per request | mcp/deep.py:52-54,1294-1350 | off>on>struggle>kickoff>area | Which trigger wins when several apply | A-flag | One run per request is B (one helper lane; operator 2026-09-23T13:42 "Sequential via 2nd brain", operator_answers.txt). The ORDER itself has no operator quote. | keep; operator to confirm the value/wording |
| QUESTION_CHARS | mcp/deep.py:150 (used 1136,1155,1324,1382,1395) | 6000 chars | Silent cut of the task/spec sent to the second brain | D | No evidence. Kickoff: ("Plan this task.\n\nTASK:\n"+spec)[:6000] with no marker; index/octopus/prompts/V0.md is 8,984 chars, so the planner sees ~66% of the Octopus spec. Same class as MAX_FINDING_CHARS. | DONE 2026-09-27: REMOVED: the whole spec/question goes to the second brain (test_deep test_a_long_spec_reaches_the_plan_job_whole: a 20k spec whole). E1's input keeps its training cut as E1_QUESTION_CHARS 2000 (B: bench/e1/build_heldout2.py:193). |
| CONTEXT_CHARS + task[:1500] | mcp/deep.py:151,1112,1141,1382,1395 | 1500 chars | Silent cut of earlier user text / task in trigger questions | D | No evidence (selection.py:154 has the same 1500, also unexplained). | DONE 2026-09-27: REMOVED from the second brain's input (ctx, task[:1500] in struggle/area/route questions). E1 keeps E1_CONTEXT_CHARS 1500 (B: the label rendering, build_heldout2.py:194). shomen.py:1252 context[:1500] PENDING (shomen.py owner). |
| struggle question excerpts | mcp/deep.py:1116-1136 | 3 errors, 300+400 chars, 8 events, 8 paths, 3 phrases | What of the failing output the struggle investigation sees | D | No evidence; hand-picked excerpt sizes. | DONE 2026-09-27: REMOVED: every failing output of the struggle's span, whole; every event, path and phrase. |
| yama_think_deeply tool (existence) | mcp/deep.py:158-190 | tool on main at xhigh/max | Model-chosen deep-thinking trigger | A | Operator 2026-09-24T15:02 (1749): "that sounds accurate for think_deeply, so long as it also has self improvement loop." | keep |
| TOOL_NAME yama_* rename | mcp/deep.py:153-159 | yama_think_deeply | Our tool names on main | A-flag | Claimed "(operator, 2026-09-27)"; no quote found in operator_messages.txt (ends 13:33). | keep; operator to confirm the value/wording |
| THINK_DESCRIPTION | mcp/deep.py:166-178 | text | Tool description = trigger list prompt for think_deeply | A-flag | Wording ours ("Its wording is a CHOICE"); "server tool" sentence claimed operator 2026-09-27, no quote found. Routing-by-description rule is AGENTS "Tool descriptions are prompts" (numbers at risk, CONSTRAINTS #31). | keep; operator to confirm the value/wording |
| yama_plan tool + PLAN_DESCRIPTION | mcp/deep.py:192-221 | text | Plan job as a model-callable server tool | A-flag | Claimed "(operator, 2026-09-27; pagoda-h2)", no quote found. Description steers: "before a significantly long implementation task ... a one-file fix or a question needs no plan" -- wording a CHOICE. | keep; operator to confirm the value/wording |
| KICKOFF_PLAN_ARGS | mcp/deep.py:226 | "The task in the user's message above." | Pointer arg of the inserted kickoff plan call | B-value-arbitrary | Derivation: a copy of the task would sit in main's cached context all conversation; wording arbitrary. Not yet referenced by proxy. | keep (the bound is forced); derive the value |
| THINK_REASONING / THINK_REASONING_MACHINE | mcp/deep.py:232-252 | text | Prefilled reasoning after a think_deeply result | A-flag | Prefill of the hand-off as reasoning is operator decision 2 (SELF-IMPROVEMENT-PLAN.md:126); wording ours, "UNMEASURED WORDING", from live gate n=1 / handle d28941fb. | keep; operator to confirm the value/wording |
| PLAN_HEAD | mcp/deep.py:262 | text | Header prefilled before the plan in main's reasoning | A-flag | Plan prefilled as reasoning follows operator decision 2; wording ours. | keep; operator to confirm the value/wording |
| is_error: exit status / exit_code_meaning | mcp/deep.py:323-359 | exit 0 ok; exit 1 + no output ok | Structured decision of whether a tool result failed | B | POSIX status semantics (grep/diff exit 1 = no match); Hermes' own exit_code_meaning "(not an error)" field. | keep |
| _ERR_TEXT word list | mcp/deep.py:272-285 | regex | Plain-text failure detection (error:, Traceback, npm ERR!, N failed...) | D | deep.py:270 "A CHOICE, checked on the fixtures in mcp/test_deep.py; not measured on harness traffic". | DONE 2026-09-27: REMOVED: is_error reads only structured fields (exit status, exit_code_meaning, ok/success, error field). CONSEQUENCE: Codex, OpenCode and Pi send plain-text tool results, so struggle gets no failure signal from them (test_harness_decisions marks their struggle fixtures KNOWN). Reading Codex's "Process exited with code N" / Pi's "Command exited with code N" (each harness's own exit-status line) is an OPEN operator decision. |
| is_error head/tail scan | mcp/deep.py:358-359,416 | 3000+3000 chars | Bound on text scanned per tool result | B-value-arbitrary | Cost bound; value arbitrary. | keep (the bound is forced); derive the value |
| STILL_BROKEN phrase regex | mcp/deep.py:292-300 | regex | Detects the user saying the fix did not work | D | Phrase list ours; no measurement. Used for the user_still_broken signal, the kickoff exclusion and labels. | DONE 2026-09-27: REMOVED with the user_still_broken signal and the "still broken" labels (deep_learn.propose_descriptions inert for new rows). e1.HEADS escalate still lists the feature (always 0) -- changing it changes the head's dimensions. |
| FIXUP_CAPPED | mcp/deep.py:303 | regex | Reads our own fix-up note "still there after N rounds" | B | Correctness: matches the text tool_code.finish writes. | keep |
| file_rewritten rule (after_failure / same_edit) | mcp/deep.py:590-715 | rule | A rewrite counts only after that file's failed write or a repeated edit | A-flag | #33 says "operator rule"; only a compaction summary lists it (operator_messages.txt:2185), no typed message found. Replay n=1 run (bench/octopus/replay_struggle.py). | keep; operator to confirm the value/wording |
| _NOT_APPLIED phrases | mcp/deep.py:397-400 | regex | Detects a write/patch that changed nothing | D | Hermes' JSON no_change flag is B (the harness's field); the phrase list is "A CHOICE, from the Octopus v0b-V0 run's tool results" (deep.py:395). | DONE 2026-09-27: REMOVED; the harness's structured no_change flag stays. |
| error_signature / normalise_error_line | mcp/deep.py:439-510 | first error line, 160 chars, normalised | Decides whether two failures are "the same error" | D | deep.py:437 "A CHOICE, tested on the Octopus error lines"; approval only in a compaction summary quote (2685: "Struggle rule change yes if it is causing issues"). | DONE 2026-09-27: REMOVED the normaliser: the raw first error line (exact); records keep a sha1 digest of it, never the text. |
| ENVIRONMENT table | mcp/deep.py:529-556 | 4 rows | Harness refusals / tool input errors never count as struggle | D | deep.py:527 "THE TABLE IS A CHOICE; each row names where it was seen" (Octopus v0e/v0b). #45 claims operator's (b); only summary text. | DONE 2026-09-27: REMOVED with environment_class and the environment record: every failure counts. |
| grep syntax errors count | mcp/deep.py:550-555 | search_syntax row removed | A search tool's own syntax error counts as struggle | A-flag | Claimed "(operator, 2026-09-26)"; only compaction summaries (operator_messages.txt:2437, 2484), no typed message. | keep; operator to confirm the value/wording |
| one event, one signal | mcp/deep.py:680-693 | rule | A re-run failing again is one signal, not two | B | Correctness: no double counting of one event. | keep |
| still-broken only after an answer | mcp/deep.py:693-713 | rule | A first-turn bug report is the task, not a struggle | B | Correctness. | keep |
| think_tool_offered (sticky) | mcp/deep.py:998-1037 | decided on first request, kept | Keep main's tool list fixed for the conversation | B | Prompt-cache correctness: a tool list change changes the cached system block (AGENTS "One model, one cache", operator 2026-09-24). | keep |
| state_lock cache cap | mcp/deep.py:973 | 4096 | Clears the per-conversation lock map past N | B-value-arbitrary | Memory bound. | keep (the bound is forced); derive the value |
| kickoffs kept | mcp/deep.py:1072 | last 20 | Turn keys of planned kickoffs remembered | B-value-arbitrary | Memory bound. | keep (the bound is forced); derive the value |
| E1 heads in decide | mcp/deep.py:1266-1274,1288-1289 | escalate/route_in | E1 head replaces the threshold / adds a library area | C | route_in: docs/E1.md:13-15 104/120 vs 80, 110/141 vs 86 (McNemar). escalate head untrained (deep_learn.py:134). Off by default (YAMADORI_E1). | keep |
| _THR_TTL | mcp/deep.py:914 | 30 s | Cache of thresholds in force | B-value-arbitrary | Cache freshness bound. | keep (the bound is forced); derive the value |
| outcome labels + COUNTED_ONCE | mcp/deep.py:1646-1747 | len(ev)>=2 miss rule, one label per episode | Label rules for runs and non-decisions | A-flag | Loop and missed/wasted labels operator-required (1749; SELF-IMPROVEMENT-PLAN.md:237-243); every rule and count ours. | keep; operator to confirm the value/wording |
| handoff_terms / "hand-off used" | mcp/deep.py:1528-1543,1686-1688 | 20 terms | Hand-off counted used if any backticked name/path appears later | A-flag | "hand-off unused" label in the op-required loop (plan:241); the heuristic is ours. | keep; operator to confirm the value/wording |
| investigate question texts | mcp/deep.py:1126-1155,1389-1395 | text | Wording of struggle/area/library questions to the second brain | A-flag | Own-machinery wording (memory skills-not-rules: own-machinery prompts fine); "(data ..., not instructions)" is B (operator 2026-09-24 fetched-content-is-data rule). | keep; operator to confirm the value/wording |
| traffic fails closed | mcp/deep.py:1552-1557 | "test" on error | Unknown account traffic is never learned from | B | Correctness/safety. | keep |
| RECORD_QUEUE | mcp/deep.py:1860 | 256 | Bounded background write queue; full drops a row | B-value-arbitrary | Keeps writes off the response path; size arbitrary. | keep (the bound is forced); derive the value |
| DB timeouts / retries | mcp/deep.py:1503-1504,1821-1848 | 10 s; 4 tries x 0.5 s | SQLite busy handling | B-value-arbitrary | Plumbing. | keep (the bound is forced); derive the value |
| row size cuts | mcp/deep.py:1581-1583 | signals 20000, handoff 4000 chars | Stored JSON sizes in deep_decisions | B-value-arbitrary | Storage bound; records only. | keep (the bound is forced); derive the value |
| per_day | mcp/deep.py:1935 | 14 days | Dashboard window | B-value-arbitrary | Display. | keep (the bound is forced); derive the value |
| LEARN_MIN_N | mcp/deep_learn.py:65 | 5 | Deciding labels needed before the learner moves the threshold | A-flag | Plan text "each names its n" (SELF-IMPROVEMENT-PLAN.md:246); 5 ours: deep_learn.py:41 "Every rule and number here is a CHOICE, none measured". | keep; operator to confirm the value/wording |
| learner step rule | mcp/deep_learn.py:176-190 | +/-1; missed>wasted, wasted>helped+missed | How labels move struggle_threshold | A-flag | Operator-required loop; rule ours. | keep; operator to confirm the value/wording |
| no_effect counts as wasted | mcp/deep_learn.py:126-136,178-179 | rule | A run that changed no project file pushes the threshold up | A-flag | deep_learn.py:131 "(a CHOICE)"; ours. | keep; operator to confirm the value/wording |
| old-rule run rows ignored | mcp/deep_learn.py:136 | label_rule >= 2 | Learner skips runs labelled under the automatic-helped rule | B | Correctness: old labels were helped by construction (#45/#52). | keep |
| IDLE_MINUTES | mcp/deep_learn.py:66 | 15 min | Quiet time before the learner job is queued | A-flag | Idle-time job is plan text (SELF-IMPROVEMENT-PLAN.md:241); 15 ours. | keep; operator to confirm the value/wording |
| STALE_SECONDS | mcp/deep_learn.py:69 | 24 h | Open rows older than this closed as unobserved | A-flag | Ours; AGENTS.md:592. | keep; operator to confirm the value/wording |
| propose_descriptions | mcp/deep_learn.py:198-230 | top 5 phrases, never armed | Proposes think_deeply description variants from missed rows | A-flag | Plan text "think_deeply description variants" under operator loop; never armed (PROTOCOL). | keep; operator to confirm the value/wording |
| overview note text | mcp/deep_learn.py:345 | "every threshold, rule and bound is a choice, unmeasured" | Dashboard caveat string | B | Honest label; no behaviour. | keep |
| dash_deep routes | mcp/dash_deep.py:1-93 | none | GET /dash/api/deep, revert, E1 revert/decide | B | Plumbing behind accounts.identify; no tunables. | keep |
| MAX_FINDING_CHARS | mcp/shomen.py:185,2060-2069 | 6000 chars (+excerpt chars) | Cuts the hand-off/plan and appends "[hand-off cut ...]" | D | Operator 2026-09-27T13:31 "You made that limit up, unfounded." (cut a plan, pagoda-h2; memory no-invented-numbers.md). CONSTRAINTS.md #9 measured only that 1400 cut 2/26; 6000 was never derived. Being removed concurrently. | DONE by the shomen.py owner (verified absent 2026-09-27; operator 13:31 'You made that limit up, unfounded.'). |
| MAX_TRACES | mcp/shomen.py:190 | 32 | In-process trace store size | B-value-arbitrary | Memory bound. | keep (the bound is forced); derive the value |
| HANDOFF_TARGET_WORDS | mcp/shomen.py:220,238 | 250 words | "Aim for under 250 words" in the investigate prompt | D | shomen.py:218 "The word target is a CHOICE, not a measurement". #60 found the plan's word target caused 85 word-count passages. | DONE 2026-09-27: REMOVED with its "Aim for under N words" sentence. |
| investigate SYSTEM: sections + routing table | mcp/shomen.py:228-259 | 4 sections | Hand-off format FACTS/SEARCHED/OPEN/NEXT | A-flag | Operator 2026-09-23 (1395): "sending back the distillation and facts"; the four sections are ours (code says "operator decision, 2026-09-23" -- the quote asks for distillation+facts only). | keep; operator to confirm the value/wording |
| investigate SYSTEM: plain-English style block | mcp/shomen.py:268-283 | text | Style rules "may be read by a classifier" | D | Rationale stale: the classifier was Laya distil, removed on measurement (FINDINGS #17). No evidence the rules help. | DONE 2026-09-27: REMOVED; the routing table and the image paragraph stay. |
| investigate SYSTEM: image paragraph | mcp/shomen.py:261-266 | text | Draw mockups with generate_image, check with describe_image | A | Operator 2026-09-23T15:37 (840): "Deep thinking should be able to generate images for mockups and designs". | keep |
| LANDING / cap landing | mcp/shomen.py:289-293,1261,1286-1314 | tool_turn_limit 10 (20 at max) | At the cap tools are withdrawn and the hand-off is asked for | A | operator_answers.txt 2026-09-23T14:55 "Both, 10 each" ("then answer from what they have"); 830: "let it do 20 max instead"; 804 "follow the agenticMaxTurns". | keep |
| helper KV window stop | mcp/shomen.py:1368-1384 | window >= helper share | Lands the run when its context reaches the helper share | B | Derived from budget.py HELPER_TOKENS 49,152 (AGENTS: operator 2026-09-25 "~48k second brain"); CONSTRAINTS #7 peak-window fix. | keep |
| _helper_budget fallbacks | mcp/shomen.py:574-584 | 49152 / 61440 | Helper share if budget.py fails or returns 0 | B-value-arbitrary | 61440 is the stale 3/8 of 163,840; inconsistent with 49152. | keep (the bound is forced); derive the value |
| investigate hop max_tokens | mcp/shomen.py:1317 | 1500 | Answer allowance per helper hop | B-value-arbitrary | CONSTRAINTS.md §3: "a tool call, or a finding of <=200 words, with headroom"; effectively dead: tiers.apply raises it to A_MIN 2048. | keep (the bound is forced); derive the value |
| investigate temperature | mcp/shomen.py:1317 | 0.2 | Requested sampling temperature | D | Dead: tiers.enforce_sampling overrides with vendor 1.0 (operator 2026-09-23 "any vendor settings, the proxy needs to enforce", tiers.py:806). | DONE 2026-09-27: REMOVED (tiers.enforce_sampling set it anyway). |
| context[:1500] in investigate | mcp/shomen.py:1244 | 1500 chars | Silent cut of conversation context given to the second brain | D | No evidence; stacks on deep.CONTEXT_CHARS. | DONE 2026-09-27: REMOVED: the context is sent whole. |
| _post timeout | mcp/shomen.py:498 | 3600 s | Transport timeout per helper generation | B-value-arbitrary | CONSTRAINTS.md §23-30 "Timeouts -- KEEP" as breakers. | keep (the bound is forced); derive the value |
| helper nudge | mcp/shomen.py:521-524 | tiers.helper_nudge | Action-naming nudge on research hops | A | operator_answers.txt 2026-09-26T20:48: "Action-naming variant (Recommended)". | keep |
| per-job thinking cap | mcp/shomen.py:519-521 (tiers.py:538) | investigate 6144, plan 4096, fixup 3072 | Caps each helper hop's thinking | A-flag | Direction operator 2026-09-25 "this model overthinks" (AGENTS); values ours, plan n=1 (#60 "A CHOICE, n=1 run"). | keep; operator to confirm the value/wording |
| plan job (existence) | mcp/shomen.py:295-305 | plan job | Second brain writes FILES/ORDER/KEY DECISIONS/RISKS for a new task | A | Operator 2026-09-27 (2858) "always give it a planning turn"; 2361 "planning up front and auto fixup kept this model on track". | keep |
| PLAN_TARGET_WORDS | mcp/shomen.py:306,322 | 300 words | Word target in PLAN_SYSTEM v1 | D | "a word target that is a CHOICE" (shomen.py:302); #60: 85 word-count passages; V2 (default) already drops it. | DONE 2026-09-27: REMOVED with v1's word-target sentence. |
| PLAN_SYSTEM (v1) | mcp/shomen.py:311-340 | text | Plan job's system prompt (routing table) | A-flag | Own-machinery prompt of an operator-approved job; wording ours. Confirm-cwd/entry-first rows already removed 2026-09-27 (operator 12:41 "realtime target corrections"). | keep; operator to confirm the value/wording |
| PLAN_SYSTEM_V2: truthful task/tools lines | mcp/shomen.py:342-372,406-416 | switch plan_prompt on | Says the engineer has the task; tools line matches what is offered | B | Correctness: the prompt must describe the system truthfully (v1 said "nothing else"; main does have the task). | keep |
| PLAN_SYSTEM_V2: style steering | mcp/shomen.py:375-376 | "One short line per item ... referred to, not copied" | Shapes plan items to not copy the task's numbers | D | From one run (v0f, n=1), "UNMEASURED WORDING". | DONE 2026-09-27: REMOVED the two sentences. test_deep updated. |
| PLAN_TOOL_SITUATIONS / plan_tools gate | mcp/shomen.py:419-462 | 4 situations | Plan gets our tools only when the task names held source | D | shomen.py:432 "A CHOICE, n=1 run" (v0f). | DONE 2026-09-27: DELETED: the plan gets our tools, less run_check with no repository (B). Switch gone from tiers. |
| run_check only with a root | mcp/shomen.py:435,456-459 | rule | Drop run_check when no repository is bound | B | Correctness: run_check needs a repository. | keep |
| PLAN_LANDING | mcp/shomen.py:465 | text | Landing prompt for the plan job | A-flag | Landing mechanism A (cap answer); plan wording ours. | keep; operator to confirm the value/wording |
| plan time/turn cap | mcp/shomen.py:1262-1304 (tiers.py:549-550) | 2 tool turns, 150 s | Lands the plan job early | D | tiers.py:547 "Both CHOICES, unmeasured"; v0f n=1. | DONE 2026-09-27: DELETED (shomen branch, tiers.PLAN_TOOL_TURNS / PLAN_SECONDS, switch plan_budget). |
| LAYA_TIMEOUT | mcp/shomen.py:566 | 20 s | Laya call timeout | B-value-arbitrary | Dead path: route_in/distil have no production caller; E1 retires Laya. | DONE 2026-09-27: deleted with shomen._laya (dead path). |
| T_INVESTIGATE / T_ANSWERED | mcp/shomen.py:569-570 | 0.5 | "UNCALIBRATED" Laya thresholds | D | Referenced nowhere (grep). | DONE 2026-09-27: REMOVED. |
| LAYA_PERMUTATIONS / MARGIN_GATE | mcp/shomen.py:611-612 | 2 / 0.3 | Laya ordering count and decided-margin gate | D | bench/mechanisms/selectors.py:106 "PROVENANCE OF 0.3: UNVERIFIABLE. Nothing in this repo reproduces it." Dead path in production. | DONE 2026-09-27: REMOVED with route_in/distil, _choice_averaged, _laya, LAYA_URL, LAYA_TIMEOUT. bench/test_laya_calibration.py and test_e1 updated. |
| route_in() / distil() prompts and cuts | mcp/shomen.py:587-762 | state[:6000], context[:2000], finding[:4000] | Laya closed-choice router and finding judge | D | No production caller; distil measured non-discriminating (FINDINGS.md:561-567, n=29, 29/29 from_the_files). | DONE 2026-09-27: REMOVED (no production caller). |
| PHRASES (fold-back vocabulary) | mcp/shomen.py:797-811 | After thinking deeply, / Verified / Repaired / ... | Fixed words that mark second-brain work in main's turn | A | Operator 2026-09-24 (1725): "'thought deeply' or 'verified' 'repaired' ... 'After thinking deeply, I decided'"; (1735) "Today I was inspire by seed-word." | keep |
| prefill ends on a letter | mcp/shomen.py:792-796 | no trailing space | Prefilled phrase must not end on a space token | C | STEP 0 probe run twice: 976/995 reused; trailing space splits the token (SELF-IMPROVEMENT-PLAN.md:87-88). | keep |
| seed on every second-brain job | mcp/shomen.py:866-881,1245-1247,1096-1098 | seed in USER message | Concept seed word in each job's user turn | A | Operator (499): "Every second brain run gets a seed word always"; (1735) "in the user message ... same see word". | keep |
| SEED_LINE wording | mcp/shomen.py:846-848 | text | Says the word shapes approach and stays out of the output | A-flag | #13 fix (seed leaked into code, n=1); memory skills-not-rules: own-machinery seed wording fixes are fine. Wording ours. | keep; operator to confirm the value/wording |
| SEED_LINE_RESEARCH | mcp/shomen.py:849-853,857-863 | text, switch seed_frame on | Says the word is random, not a clue or anagram | A-flag | #52: 4 of 6 runs decoded the seed (v0e p2); "UNMEASURED WORDING"; own-machinery fix. | keep; operator to confirm the value/wording |
| fixup job (existence) | mcp/shomen.py:963-1150 | fixup at high/max | Second brain repairs code that fails to parse | A | operator_answers.txt 2026-09-23T12:32: repair "On at high and max (Recommended)". | keep |
| FIXUP_SYSTEM | mcp/shomen.py:963-969 | text | Change only the lines the errors require | A-flag | #24 rule (2026-09-24) "change what the model wrote only when it is broken"; wording ours. | keep; operator to confirm the value/wording |
| FIXUP_MIN_KEEP | mcp/shomen.py:1048,1069-1071 | 0.5 | Rejects a repaired version under half the original's length | D | shomen.py:1043 "A CHOICE, not a measurement ... No fix-up length distribution exists in this repo yet." | DONE 2026-09-27: REMOVED (and _nonblank): fix_rejection keeps closed fence / not cut / no errors. A short parsing repair is now written back. test_tool_code updated. |
| fixup max_tokens | mcp/shomen.py:1103-1104 | max(A_MIN, chars//2 + 512) | Answer allowance for a repaired file | B-value-arbitrary | Reply must hold the whole code (~4 chars/token, so chars/2 is ~2x headroom); +512 arbitrary. | keep (the bound is forced); derive the value |
| fixup retry message | mcp/shomen.py:1132-1136 | text | Feeds the parser's remaining errors back | B | Mechanical: carries the checker's output. | keep |
| fixup cuts | mcp/shomen.py:986,1000 | errors[:8], request[:2000] | Silent cuts of errors and request in the fixup prompt | D | No evidence. | DONE 2026-09-27: REMOVED: all errors and the request whole. test_tool_code checks it. |
| fix_rejection closed-fence / not-cut / no-errors | mcp/shomen.py:1029-1039,1055-1068 | rules | Only a complete parsing repair replaces the model's code | B | Correctness: an unclosed fence or finish_reason length is a cut reply (model.BudgetEvent). | keep |
| MAX_ITEMS | mcp/shomen.py:1646,2046-2057 | facts 12, searched 8, open 5, next 3 | Items past the cap replaced by "(+N more; trace handle)" | D | shomen.py:1642 "They are CHOICES, not measurements". Main cannot open a trace handle. | DONE by the shomen.py owner (verified absent 2026-09-27). |
| PLAN_MAX_ITEMS | mcp/shomen.py:2387,2453-2467 | files 15, order 12, decisions 8, risks 6 | Plan items past the cap are dropped with a handle marker | D | No evidence. A plan with >12 ORDER steps loses the rest silently for main (handle unreadable) -- same class as MAX_FINDING_CHARS. | DONE by the shomen.py owner (verified absent 2026-09-27). |
| THE EVIDENCE (inline excerpts) | mcp/shomen.py:1725-1742,1971-1991 | inline lines per verified fact | Verified facts carry the cited source lines | A | Operator (1890): "hand-off must inline the src code snippets that are relevant the second brain can not hand it server file paths from tool use." | keep |
| EXCERPT_MAX_LINES / EXCERPT_CONTEXT / EXCERPT_TOTAL_CHARS | mcp/shomen.py:1743-1745 | 12 lines / 3 lines / 3600 chars | Size of each excerpt and of all excerpts | A-flag | Feature operator; shomen.py:1742 "The caps are CHOICES, unmeasured." | keep; operator to confirm the value/wording |
| unreadable citation removed | mcp/shomen.py:1907-1945,2005-2034 | rule | A citation the verifier cannot read is removed; fact labelled reasoning | B | Correctness (fabrication check; live gate 2026-09-24 invented path). Citation check replaced Laya on measurement (FINDINGS #17). | keep |
| REASONING_LABEL / WEB_LABEL | mcp/shomen.py:221-222,2023-2026 | "(reasoning, not checked against source)" / "(web)" | Labels unverified and web facts | A-flag | (web) label claimed "(operator, 2026-09-24)" (shomen.py:2023); no typed quote found. Reasoning label is B (honest labelling). | keep; operator to confirm the value/wording |
| machine-built hand-off (existence) | mcp/shomen.py:2136-2196 | never empty | Proxy builds a hand-off from the trace when none written | A-flag | Operator 1395: "never sending back the distillation and facts" (work must cross); "An empty hand-off is impossible (operator, 2026-09-23)" is our paraphrase. | keep; operator to confirm the value/wording |
| machine-built NEXT STEP texts | mcp/shomen.py:2170-2180 | text | Tells main how to use the machine hand-off | A-flag | Own-machinery wording. | keep; operator to confirm the value/wording |
| MACHINE EVIDENCE scoring | mcp/shomen.py:2209-2362 | 4 excerpts, 400 hits/step, >=2 terms, code x3, idf, stoplist | Picks source lines matching the question for a machine hand-off | D | shomen.py:2208 "All CHOICES, unmeasured"; from one deploy-check handle d28941fb (n=1). | DONE 2026-09-27: REMOVED: the machine hand-off lists the paths retrieved and the search log. proxy.FOLD_BACK_TAIL_MACHINE and deep.THINK_REASONING_MACHINE no longer promise source lines. test_stream updated. |
| machine hand-off cuts | mcp/shomen.py:2162,2185,2189 | paths[:12], asked[:200], why[:360] | Short cuts in the machine hand-off | D | No evidence. | DONE 2026-09-27: REMOVED. test_stream test_nothing_cuts_a_hand_off_or_a_plan checks 20 paths, a long question and reason whole. |
| delegate_investigation TOOL | mcp/shomen.py:2487-2526 | off by default | Benchmark-arm tool on main | A-flag | AGENTS: "a benchmark arm", only with YAMADORI_DELEGATE_TOOL=1; description wording ours (one prohibition). | keep; operator to confirm the value/wording |
| as_thinking display | mcp/shomen.py:2529-2555 | 12 steps, 48-char args | Renders a trace for a thinking block | B-value-arbitrary | Display only; no caller found (dead). | keep (the bound is forced); derive the value |
| _LINES_CACHE | mcp/shomen.py:1901 | 64 files | Source-line cache size | B-value-arbitrary | Memory bound. | keep (the bound is forced); derive the value |
| _PATH / _FILE_CITE / section heading patterns | mcp/shomen.py:482-484,1648-1662,1746-1751 | regex | Parse citations and section headings | B | Correctness of parsing (json-before-js bug, dotfile bug documented at shomen.py:468-481, 539-546). | keep |
| plan claims = KEY DECISIONS only | mcp/shomen.py:2397-2413 | rule | Files to create are not counted as citations | B | Correctness (#60: "unsupported 9" were files to create). | keep |
| helper lane wait in run() | mcp/shomen.py:915-921 | admission.WAIT_SECONDS | Wait for the one helper lane, else skipped | B | Lane exclusivity: HELPER_LANES = 1 (operator 2026-09-23 sequential second brain); wait value lives in admission.py. | keep |

### Budgets, slots, ledger: tiers.py, budget.py, admission.py, slots.py, nebari.py, compaction.py, model.py, catalog.py, gpu_room.py, config.template.yaml

| name | file:line | value | what it does | class | evidence or derivation | action |
|---|---|---|---|---|---|---|
| ALIASES | mcp/tiers.py:80 | none/off->minimal, med->medium, ultra->max ... | maps client effort spellings onto the six tiers | B | OpenAI schema minimal\|low\|medium\|high plus spellings clients send (xhigh, none, off) (tiers.py:77-79); a lookup table, no tuning. | keep |
| TIERS ladder (minimal..max flags) | mcp/tiers.py:147 | 6 tiers, flags thinks/effort/retrieval/skills/fanout/investigate/chec… | what each reasoning_effort allows | A | operator 2026-09-23T15:08 "xHigh -> medium thinking, all augmentations / Max -> xHigh thinking"; 15:23 "high -> medium in your table ... everything else looks good"; 15:27 "minimal should be thinking off ... low should be base model". | keep |
| TIERS repair at high/xhigh/max | mcp/tiers.py:192 | repair=True | second brain repairs non-parsing code at high and up | A | AskUserQuestion 2026-09-23T12:32: "On at high and max (Recommended)". | keep |
| TIERS fanout | mcp/tiers.py:191 | 3 at high/xhigh/max, 1 below | max fan-out candidates | A | AskUserQuestion 2026-09-23T12:46 "Yes, fan out on code (Recommended)" (3 candidates); 13:42 "Sequential via 2nd brain (Recommended)". | keep |
| TIERS xhigh effort=medium, max effort=xhigh | mcp/tiers.py:206 | medium / xhigh | effort-matched pair | A | operator 2026-09-23T15:08 tier mapping; AskUserQuestion 2026-09-24T01:38 "Keep it, relabel only". | keep |
| images_offered (every tier) | mcp/tiers.py:394 | True | generate_image/describe_image offered on every tier | A | operator 2026-09-23T15:34 "image gen should just be everywhere it is a capability not a gate, keep it on everywhere". | keep |
| TOOL_TURNS | mcp/tiers.py:273 | 10 (YAMADORI_MAX_TOOL_TURNS) | tool-turn cap per context (main loop, each deep-thinking run) | A | operator 2026-09-23T14:54 "We should for sure follow the agenticMaxTurns then"; AskUserQuestion 14:55 "Both, 10 each (Recommended)"; PrismML agenticMaxTurns = 10. | keep |
| TOOL_TURNS_MAX | mcp/tiers.py:274 | 20 (YAMADORI_MAX_TOOL_TURNS_MAX) | tool-turn cap at tier max | A | operator 2026-09-23T15:24 "That 10 max could be bummped on Max mode, let it do 20 max instead maybe...". | keep |
| CEILING | mcp/tiers.py:285 | max (YAMADORI_TIER_CEILING) | deployment cap on the tier a caller may pick | B | default is no cap (highest tier); a deployment switch, not a tuned number. | keep |
| DEFAULT tier | mcp/tiers.py:286 | medium (YAMADORI_TIER_DEFAULT) | tier when the client sends no effort | B-value-arbitrary | a default must exist; medium is OpenAI's own reasoning_effort default and the config filter's default. No operator quote or measurement found. | keep (the bound is forced); derive the value |
| FALLBACK_EFFORTS | mcp/tiers.py:315 | (low, medium, xhigh) | efforts assumed if the served template cannot be read | B | served template guard: "Supported types are xhigh (default), medium, and low" (tiers.py:302-303); parsed live from /props otherwise. | keep |
| safe_effort rounding | mcp/tiers.py:349 | high->xhigh, minimal->low, max->xhigh, ""->xhigh | maps unsupported efforts to accepted ones | B-value-arbitrary | a mapping is forced (template raises HTTP 500 on anything else, tiers.py:305-308); rounding UP rather than down is our choice, unmeasured. | keep (the bound is forced); derive the value |
| accepted_efforts timeout | mcp/tiers.py:340 | 5 s | wait for /props when reading the template | B-value-arbitrary | a network timeout must exist; value not derived. | keep (the bound is forced); derive the value |
| A_MIN | mcp/tiers.py:464 | 2048 (YAMADORI_ANSWER_MIN) | minimum answer allowance; added on top of thinking | C (weak) | docs/CONSTRAINTS.md:205 "Measured answers ran 89-597 tokens ... LCB completions ... run to 4,595"; existence from empty-reply bug (SELF-IMPROVEMENT-LOG #8). Small n (7 natural runs, 3 prompts). | keep |
| MIN_THINKING | mcp/tiers.py:465 | 1024 | floor on derived thinking room and on every cap | B-value-arbitrary | a positive floor is forced (window-prompt-answer can go <=0); 1024 echoes the retired server budget of 1000 that CONSTRAINTS s.1b showed was below natural length. No measurement. AGENTS.md:1010 uses A_MIN+MIN_THINKING=3,072 as the refusal floor ("a CHOICE"). | keep (the bound is forced); derive the value |
| AGENT_STEP_THINKING | mcp/tiers.py:482 | 6144 (YAMADORI_AGENT_STEP_THINKING) | thinking cap on agent_step requests | A-flag (number ours) | operator 2026-09-25T11:48 "we may need to limit thinking a bit more ... this model overthinks pretty quickly"; "1.5 those numbers above and dial it in" (quoted in the session summary; typed original not in extract). Base 4,096 ours (Octopus v0b, n=1 run). | keep; operator to confirm the value/wording |
| AGENT_STEP_THINKING_READ | mcp/tiers.py:498 | 2048 (YAMADORI_AGENT_STEP_THINKING_READ) | cap for a step answering a read/search/inspect result | D | #53: "CHOICES", status "a PROPOSAL for the operator"; the 2026-09-26 AskUserQuestion approved only the nudge wording. n=1 run per prompt (Octopus v0e). Steers one benchmark's deliberation. | DONE 2026-09-27: DELETED with its call site. |
| AGENT_STEP_THINKING_ERROR | mcp/tiers.py:500 | 6144 (YAMADORI_AGENT_STEP_THINKING_ERROR) | cap for a step answering an error | D | #53, same as above; equals AGENT_STEP_THINKING, so it only labels. | DONE 2026-09-27: DELETED with its call site. |
| step_thinking() by result kind | mcp/tiers.py:504 | read\|error\|other | chooses the agent-step cap by the previous tool result | D | #53 behaviour built to steer Octopus v0e ("25 of 170 agent steps reached the nudge"); "CHOICES, unmeasured for quality". | DONE 2026-09-27: DELETED with its call site and the switch. |
| USER_TURN_THINKING | mcp/tiers.py:520 | 12288 (YAMADORI_USER_TURN_THINKING) | thinking cap on every other main turn | A-flag (number ours) | tiers.py:516-520: 8,192 = ~3x longest natural finish 2,826 (CONSTRAINTS 1b, n=7), x1.5 per operator "1.5 those numbers" (summary-quoted); code itself says "A CHOICE". | keep; operator to confirm the value/wording |
| HELPER_THINKING | mcp/tiers.py:524 | 6144 (YAMADORI_HELPER_THINKING) | cap on helper requests with no job | A-flag (number ours) | operator 2026-09-25 (summary-quoted) "Can you put a tighter thinking cap on second brain"; "1.5 those numbers"; base 4,096 ours. | keep; operator to confirm the value/wording |
| JOB_THINKING fixup | mcp/tiers.py:538 | 3072 | thinking cap per fix-up hop | A-flag (number ours) | session summary lists "Deep-thinking caps per job" among operator decisions; base 2,048 ours x1.5 operator. | keep; operator to confirm the value/wording |
| JOB_THINKING investigate | mcp/tiers.py:538 | 6144 | thinking cap per investigate hop | A-flag (number ours) | as above; base 4,096 ours x1.5. | keep; operator to confirm the value/wording |
| JOB_THINKING alternative/tiebreak | mcp/tiers.py:540 | 6144 | thinking cap per fan-out candidate hop | A-flag (number ours) | as above; base 4,096 ours x1.5. | keep; operator to confirm the value/wording |
| JOB_THINKING plan | mcp/tiers.py:539 | 4096 (YAMADORI_PLAN_THINKING) | thinking cap per plan hop | D | #60 "A CHOICE, n=1 run, unmeasured for plan quality"; cut 12,288->4,096 to shorten one Octopus v0f kickoff. No operator quote. | DONE 2026-09-27: REVERTED to 12,288 (the 1.5x value, A-flag). test_deep updated. |
| PLAN_TOOL_TURNS | mcp/tiers.py:549 | 2 (YAMADORI_PLAN_TOOL_TURNS) | tool turns for the plan job | D | #60 "Both CHOICES, unmeasured"; v0f n=1. | DONE 2026-09-27: DELETED with shomen's budget branch. |
| PLAN_SECONDS | mcp/tiers.py:550 | 150 (YAMADORI_PLAN_SECONDS) | lands the plan when elapsed+slowest hop would pass it | D | #60, same; "(time cap)" landing. | DONE 2026-09-27: DELETED with shomen's budget branch. |
| reasoning_budget_nudge mechanism | mcp/tiers.py:551 | on (YAMADORI_THINKING_NUDGE) | forces a wrap-up line into reasoning mid-budget | A | operator 2026-09-25T12:29 "would be nice to have a trigger that fired when they get close to 80% of budget"; 12:35 "Let's try nudge". | keep |
| NUDGE_MESSAGE | mcp/tiers.py:578 | "The user is waiting for a response. Let me go with the best answer..… | text injected at NUDGE_AT | A | operator 2026-09-25T14:17 proposed "The user is awaiting a response, I should consider my existing solutions as the answer ..."; 14:20 "Sounds good, lets send it". Wording unmeasured. | keep |
| NUDGE_AT | mcp/tiers.py:581 | 0.6 (YAMADORI_NUDGE_AT) | fraction of budget where the nudge fires | A-flag | operator asked "80% of budget" (2026-09-25T12:29); 0.6 claimed "operator 2026-09-25 (was 0.8)"; summary lists "nudge at 60%" as operator's and quotes "Change the 60% message then". No typed 0.6 decision found. live n=1. | keep; operator to confirm the value/wording |
| AGENT_STEP_NUDGE_MESSAGE | mcp/tiers.py:592 | "... Let me make the call I've already worked out -- the edit itself… | nudge text for agent steps | A | A: AskUserQuestion answer 2026-09-26T20:48 'Approve as written (Recommended)'. AskUserQuestion 2026-09-26T20:48 "Approve as written (Recommended)". Wording ours, approved; unmeasured. | keep |
| HELPER_NUDGE_MESSAGE | mcp/tiers.py:605 | "... Let me make the search I've already worked out, or write the han… | nudge text for investigate/plan hops | A | A: AskUserQuestion answer 2026-09-26T20:48 'Action-naming variant (Recommended)'. AskUserQuestion 2026-09-26T20:48 "Action-naming variant (Recommended)". | keep |
| HELPER_NUDGE_JOBS | mcp/tiers.py:609 | (investigate, plan) | which jobs get the helper nudge | A | same answer: research hops end in a search or hand-off (the question's own framing). | keep |
| BUDGET_MESSAGE | mcp/tiers.py:626 | "Thinking budget reached. I will stop deliberating and write the fina… | hard-stop text at the thinking budget | A + C | operator kept the original: "I told you to use the word stop here ... it was the default it used to work" (summary-quoted, 2026-09-25); CONSTRAINTS 1b: with a message 2 of 2 runs closed cleanly (n=2). | keep |
| _shares() fallback | mcp/tiers.py:662 | main 114688 / helper 49152 / pool 163840 | shares used if the budget module raises | B (stale) | only on an import/probe error; pool 163,840 predates live -c 181,248. Should be derived or fail loudly. | keep |
| estimate_prompt_tokens | mcp/tiers.py:665 | chars // 3 | high estimate of prompt tokens for budgeting | B-value-arbitrary | over-estimating is forced for correctness (never overflow the share); divisor 3 unmeasured (#60 measured ~4.0 chars/token on reasoning). | keep (the bound is forced); derive the value |
| budget(): thinking = share - prompt - answer | mcp/tiers.py:684 | derived | thinking room from the request's KV share | B | unified KV pool split by role (budget.py; CONSTRAINTS s.3). | keep |
| caps limit thinking only (max_tokens = room + answer) | mcp/tiers.py:711 | rule | a cap never squeezes the answer | B | correctness: capped total of 6,144 would cut a ~7K-token whole-file write (Octopus game.js, 709 lines) (tiers.py:704-710). | keep |
| COMPACTION_BUDGET | mcp/tiers.py:764 | 5120 (YAMADORI_COMPACTION_BUDGET) | minimum answer allowance for a compaction | A | operator 2026-09-24T13:00 "4-5k was compaction budget"; 5,120 = top of range. Not measured sufficient (7 of 12 Hermes summaries > 5,120 tokens, tiers.py:741-745). | keep |
| COMPACTION_THINKING | mcp/tiers.py:765 | 2048 (YAMADORI_COMPACTION_THINKING) | thinking budget for a compaction | D | operator asked that compaction think (2026-09-24T14:14 "isn't thinking about compaction really important") but gave no number; 2,048 from CONSTRAINTS runs "NOT a compaction measurement"; SELF-IMPROVEMENT-PLAN:193 calls it a choice. | DONE 2026-09-27: REMOVED: tiers.compaction_budget returns `thinking_tokens` = its window - prompt - answer, never below MIN_THINKING (the one budget rule); compaction.prefix_fields takes it. test_utility, test_ledger, test_harness_forms updated. |
| compaction window = pool less active helper | mcp/tiers.py:787 | derived | compaction may draw on the idle helper share | A + B | operator 2026-09-24T13:00 "we already have extra KV available at any time"; unified pool arithmetic. | keep |
| compaction client target honoured, no fixed ceiling | mcp/tiers.py:777 | cap = pool | answer = client target bounded by pool | B | Hermes states "Target ~N tokens" and discards a cut summary (context_compressor.py), tiers.py:747-754. | keep |
| VENDOR_SAMPLING | mcp/tiers.py:817 | temp 1.0, top_p 0.95, top_k 20, min_p 0, presence 0, repeat 1.0 | sampling enforced on every thinking request | A + B | PrismML card / GGUF header; operator 2026-09-23 "Any vendor settings we have the proxy need to enforce"; 2026-09-25T13:02 "keep temp at 1.0". | keep |
| VENDOR_SAMPLING_INSTRUCT | mcp/tiers.py:822 | temp 0.7, top_p 0.80, top_k 20, presence 1.5 | sampling for thinking-off (minimal) | A + B | PrismML card instruct values (tiers.py:150-153); operator 2026-09-23T15:27 "minimal should be thinking off and the special thinking off tune". | keep |
| client sampling overridden | mcp/tiers.py:827 | rule | client temperature etc. replaced by vendor values | A | operator (summary-quoted 2026-09-23) "We don't let people change our defaults here because this is a fine tune". | keep |
| AB_ARMS / ab() | mcp/tiers.py:982 | aug_off / aug_on | factorial benchmark arms | D | no caller outside mcp/test_tiers.py:490; operator 2026-09-27 "I am not letting you A/B on off ... again". | PENDING (low): a benchmark factorial helper; the operator rejected on/off arms (2026-09-27). Delete ab()/AB_ARMS and test_tiers' check. |
| BEHAVIOURS: seed_frame | mcp/tiers.py:1088 | on (YAMADORI_SEED_FRAME) | seed line says the word is random, not a clue | A-flag | #52 "UNMEASURED WORDING"; built from 4 of 6 Octopus v0e p2 runs decoding the seed. | KEPT: a fix to our own machinery's prompt (the seed word read as a clue, #52). Memory skills-not-rules (2026-09-27) records 'Fixes to our OWN machinery's prompts (seed wording, fold-back tail) are fine'. The wording is ours, unmeasured; operator to confirm. |
| BEHAVIOURS: helped_needs_change | mcp/tiers.py:1089 | on | a deep-thinking run is 'helped' only if a project file changed | D | #52 remedy, no operator quote; labelling rule built from v0b/v0e n=1. | PENDING (deep.py owner): a labelling rule for the learner, from #52 (n=1). Remove the switch and LABEL_RULE 2, or measure. |
| BEHAVIOURS: fixup_project_only | mcp/tiers.py:1090 | on | fix-up repairs project files only | D | #56 "built, not yet live"; scratch-name table approximation; v0b/v0e n=1. | DONE 2026-09-27: DELETED (switch and code). |
| BEHAVIOURS: plan_tools / plan_prompt | mcp/tiers.py:1093 | on | plan gets tools only on evidence; PLAN_SYSTEM_V2 wording | D | #60 "each a CHOICE with its switch", "UNMEASURED WORDING"; v0f n=1. | DONE 2026-09-27: plan_tools DELETED (switch and code); plan_prompt kept, its style sentences removed. |
| MAIN_SHARE / HELPER_SHARE | mcp/budget.py:112 | 0.70 / 0.30 (YAMADORI_MAIN_SHARE, _HELPER_SHARE) | fraction split, used only when HELPER_TOKENS=0 | A-flag (numbers ours) | direction operator (summary-quoted 2026-09-25): "lower the deep thinking second brain split, and weigh the first context larger again"; 0.70/0.30 ours, "A CHOICE". Inactive while HELPER_TOKENS>0. | keep; operator to confirm the value/wording |
| HELPER_TOKENS | mcp/budget.py:119 | 49152 (YAMADORI_HELPER_TOKENS) | fixed second-brain KV share; main gets the rest | A | operator 2026-09-25T12:47 "Make sure the main model gets the share of the new tokens, ~48k second brain is the constraint"; summary "Push second brain down to the 48k range". 49,152 = 48 Ki. | keep |
| HELPERS | mcp/budget.py:120 | 1 (YAMADORI_HELPERS) | number of concurrent second brains budgeted | A | AskUserQuestion 2026-09-23T13:42 "Sequential via 2nd brain (Recommended)" ("Never more than two live contexts"); supersedes 09-23T00:48 "1/4 for up to two second Brian's". | keep |
| MAIN_FLOOR | mcp/budget.py:124 | 0.50 (YAMADORI_MAIN_FLOOR) | main never below half the pool | B-value-arbitrary | a floor protecting main is a correctness guard (helper must not starve its own conversation); 0.5 echoes the operator's superseded 1/2 split (09-23T00:48). Not binding at 49,152/181,248. | keep (the bound is forced); derive the value |
| KV_KIB_PER_TOKEN | mcp/budget.py:126 | 44 (YAMADORI_KV_KIB) | KV cost per token for GiB reports and -c arithmetic | C | docs/MTP-STAGING.md:345 "q8_0 main KV is 34.0 KiB/token, exactly"; :374 "44 KiB with the head on". Used in config.yaml's 181,248 derivation. | keep |
| pool_size fallback | mcp/budget.py:165 | 131072 | pool assumed if /props unreachable | B (stale) | code flags it stale itself (budget.py:141-147): live pool 181,248. Should fail loudly or read config. | keep |
| what_if overhead | mcp/budget.py:213 | 5.95 GB weights + 0.5 GB compute; card 16.3 GB | display of what a bigger pool costs | B | weights file size; 0.5 compute "roughly" (display only). | keep |
| MAIN_LANES | mcp/admission.py:61 | 2 (YAMADORI_MAIN_LANES) | concurrent main requests admitted | B-value-arbitrary | a cap must exist (three benchmarks degraded each other into 502s, admission.py:5-12); the file's own arithmetic gives 1 main at full budget; 2 is stated oversubscription, unmeasured. | keep (the bound is forced); derive the value |
| HELPER_LANES | mcp/admission.py:65 | 1 (YAMADORI_HELPER_LANES) | concurrent second-brain jobs | A + B | AskUserQuestion 2026-09-23T13:42 "Sequential via 2nd brain"; one helper share in the pool. | keep |
| WAIT_SECONDS | mcp/admission.py:68 | 20 s (YAMADORI_ADMIT_WAIT) | queue wait before 429; default helper-lane wait | B-value-arbitrary | a bound is forced (unbounded queue = timeouts); 20 unmeasured. Consequence seen in #39: "helper lane busy after 20s; investigate skipped". | keep (the bound is forced); derive the value |
| IMAGE_LANES | mcp/admission.py:226 | 1 (YAMADORI_IMAGE_LANES) | one image generation at a time | B | VRAM: two draws = 2 x +6.3 GB on a card with 8.6 GB free (config.template.yaml:742-743). | keep |
| IMAGE_WAIT_SECONDS | mcp/admission.py:227 | 30 s (YAMADORI_IMAGE_WAIT) | wait for the image lane | B-value-arbitrary | a wait bound is forced; 30 unmeasured (draws take 19-220 s). | keep (the bound is forced); derive the value |
| 429 + Retry-After when full | mcp/admission.py:45 | rule | refuse instead of unbounded queue | B | HTTP semantics; 429 treated as NOT RUN by the test harness. | keep |
| ENABLED (slot pinning) | mcp/slots.py:100 | on (YAMADORI_SLOT_PINNING) | pin each conversation to one slot | A + C | operator 2026-09-25T11:33 "Why can't we keep a conversation on a specific slot, why does it jump around? This should be fixed"; live reuse 37,791 of 40,080 (slots.py:10-11). | keep |
| FALLBACK_SLOTS | mcp/slots.py:102 | 4 | slot count if /props has none | B | llama-server auto n_parallel = 4 (server.cpp:152-155). | keep |
| transient slot = n-1 (never pinned) | mcp/slots.py:192 | rule | side calls use a slot no conversation holds | B | correctness: a side call on LRU slot evicted a conversation's cache (slots.py:15-19). | keep |
| helper_slot = n-2 (reserved) | mcp/slots.py:197 | rule | second brain's fixed reserved slot | B | correctness: cross-process second brain overwrote a conversation's slot (live gate 2026-09-24, #10/#11). | keep |
| AFFINITY_MIN_TOKENS | mcp/slots.py:247 | 1024 (YAMADORI_COMPACTION_AFFINITY_MIN) | min shared prefix to take a pinned slot (affinity, adoption) | B-value-arbitrary | a threshold is forced for affinity (a shared harness head would otherwise overwrite a conversation's slot); code calls 1,024 "a choice" (slots.py:346). Adoption's real guard is "through an answer". | keep (the bound is forced); derive the value |
| chars/3 in _adopt/_affinity | mcp/slots.py:363 | chars // 3 | token estimate of a shared prefix | B-value-arbitrary | same estimator as tiers.estimate_prompt_tokens; unmeasured. | keep (the bound is forced); derive the value |
| ADOPTION rule | mcp/slots.py:348 | through an assistant turn, or extends whole prompt | unpinned key takes the slot it continues | A + B | operator 2026-09-25 "why does it jump around? This should be fixed" (#38); correctness: only model-generated text identifies a conversation. | keep |
| pins persisted across restart | mcp/slots.py:386 | rule | slot table survives a proxy restart | B | correctness (#38): every deploy scattered live conversations. | keep |
| LRU pin eviction | mcp/slots.py:514 | least recently used | which conversation loses its pin | B | standard policy; forced when conversations exceed pinnable slots. | keep |
| RELEASE (helper + transient slots after use) | mcp/slots.py:657 | on (YAMADORI_SLOT_RELEASE) | empty second-brain and side-call slots after use | C (script not in repo) | #58 kv_share_probe n=2 per cell: 8k decode 60.7/47.3 vs 20.1/18.2 tok/s; script lives in the session scratchpad, not the repo. | keep |
| RELEASE_WAIT_S | mcp/slots.py:706 | 15 s (YAMADORI_SLOT_RELEASE_WAIT) | wait for an in-flight release before sending | B-value-arbitrary | breaker; unmeasured. | keep (the bound is forced); derive the value |
| IDLE CLEAR mechanism | mcp/slots.py:843 | on (YAMADORI_IDLE_CLEAR) | clear other idle conversations' slots before a generation | A (conditional) | operator 2026-09-27T02:05 "If we get big token boost then clear the idle, no brainer for me." Live 2026-09-27 n=3: kept [29.3, 29.06, 28.05] vs cleared [45.36, 29.88, 27.51] -- boost on rep 0 only; the condition is not clearly met. | keep |
| IDLE_CLEAR_S | mcp/slots.py:883 | 600 s (YAMADORI_IDLE_CLEAR_S) | idle time before a conversation's slot is cleared | D (value); mechanism A | code claims "the floor the operator set for thin data" -- no operator quote found; data: 482 gaps, max 38.2 s, one harness (#59). | PENDING (unassigned): mechanism needs a threshold: derive it from the measured gap distribution with the rule written, or drop idle_clear; tests test_slots.py, test_utility.py; AGENTS.md:159-188 |
| RESTORED_MIN_TOKENS | mcp/slots.py:1099 | 64 and 0.5 of prompt | labels a resumed slot 'restored' vs 'reprocessed' | D | reporting threshold, no derivation. | PENDING (unassigned): report reused/processed raw; no tests |
| _released_recent bound | mcp/slots.py:1002 | 32 | recent release records kept for vitals | B-value-arbitrary | display bound. | keep (the bound is forced); derive the value |
| TTL_SECONDS (nebari) | mcp/nebari.py:57 | 36 h (YAMADORI_NEBARI_TTL) | session state expires after this | D | rationale only "long enough to span a working session, short enough that a stale pin cannot follow someone into next week's project"; no quote or data. | PENDING (unassigned): remove -> state keyed by explicit session id (#41) lives until ledger's age bound; no tests |
| session token pattern | mcp/nebari.py:113 | [A-Za-z0-9_-]{1,64} | validates X-Yamadori-Session | B | security: input validation of a client header; 64 arbitrary. | keep |
| key_of first 2 messages, 4000 chars | mcp/nebari.py:143 | 2 msgs, 4000 chars | legacy conversation key | B (legacy) | AGENTS.md: "no longer reached" since explicit session ids (#41). | keep |
| state blob cap | mcp/nebari.py:170 | 200000 chars | truncates saved session JSON | B-value-arbitrary | a size bound is reasonable, but slicing JSON text yields invalid JSON; load() then returns {} silently (a latent bug). | FIXED 2026-09-27: the state is saved whole (`json.dumps(state)`); SQLite's own TEXT limit is the only bound. test_nebari test_a_large_state_is_saved_whole (29/29). |
| LEDGER_ACCOUNT_BYTES | mcp/nebari.py:296 | 256 MB (YAMADORI_LEDGER_ACCOUNT_MB) | per-account ledger size before LRU eviction | B-value-arbitrary | a disk bound is forced; code claims "CHOICES (operator, 2026-09-24)" -- no quote found; SELF-IMPROVEMENT-PLAN:193 calls ledger caps choices. | keep (the bound is forced); derive the value |
| LEDGER_TOTAL_BYTES | mcp/nebari.py:298 | 2048 MB (YAMADORI_LEDGER_TOTAL_MB) | global ledger size cap | B-value-arbitrary | as above. | keep (the bound is forced); derive the value |
| LEDGER_MAX_AGE | mcp/nebari.py:300 | 30 days (YAMADORI_LEDGER_MAX_DAYS) | privacy age bound on ledger rows | B-value-arbitrary | a privacy bound is reasonable; value claimed operator's, no quote found. | keep (the bound is forced); derive the value |
| LEDGER_MEM_BYTES | mcp/nebari.py:303 | 64 MB (YAMADORI_LEDGER_MEM_MB) | in-process LRU size | B-value-arbitrary | "A choice"; memory bound. | keep (the bound is forced); derive the value |
| LEDGER_PRUNE_EVERY | mcp/nebari.py:305 | 60 s | prune interval | B-value-arbitrary | housekeeping interval. | keep (the bound is forced); derive the value |
| LEDGER_MISSES_MAX | mcp/nebari.py:319 | 100000 | known-miss cache bound | B-value-arbitrary | "A bound, not a measurement". | keep (the bound is forced); derive the value |
| LEDGER_PERSIST_REASONING | mcp/nebari.py:294 | on | persist image-tool hops across restarts | B | cache-stability design (Phase 0.5, approved 2026-09-24T14:28 "Approved"); name predates pass-through. | keep |
| sqlite connect timeout | mcp/nebari.py:77 | 30 s | lock wait on the nebari DB | B-value-arbitrary | breaker. | keep (the bound is forced); derive the value |
| compaction served on the conversation's slot/prompt | mcp/compaction.py:5 | rule | compaction reuses the cached conversation | A | operator 2026-09-24T13:35 "I want compaction, with cache, the normal way". | keep |
| compaction thinks at conversation's effort | mcp/compaction.py:596 | rule | effort line and thinking flag kept | A + B | operator 2026-09-24T14:14 (compaction should think); B: the effort line is part of the rendered prefix. | keep |
| tool_choice none on compaction | mcp/compaction.py:58 | none | model cannot call a tool while summarising | B | correctness; does not change the render (common/chat.cpp). | keep |
| KEEP_PER_ACCOUNT | mcp/compaction.py:78 | 4 (YAMADORI_COMPACTION_KEEP) | stored prompts per account in memory | B-value-arbitrary | memory bound; no rationale. | keep (the bound is forced); derive the value |
| MATCH_CHARS | mcp/compaction.py:82 | 200 | chars compared when mapping flattened records | B-value-arbitrary | must fit Hermes' verbatim 4,000-char head (compaction.py:80-81); 200 unmeasured. | keep (the bound is forced); derive the value |
| MIN_MAPPED | mcp/compaction.py:87 | 0.8 (YAMADORI_COMPACTION_MIN_MAPPED) | share of records that must map to trust the rewrite | D | code: "NOT MEASURED: a choice, stated." | DONE 2026-09-27: REMOVED: every record must map (1.0); otherwise the compaction goes up as sent. test_utility 195/195, test_harness_forms 36/36. |
| HEAD_CHARS | mcp/compaction.py:190 | 1000 | head scanned for summarise phrasing | B-value-arbitrary | phrases sit at the head of the last turn; 1000 unmeasured. | keep (the bound is forced); derive the value |
| _IN_PLACE phrases | mcp/compaction.py:186 | regex | detects in-place compaction requests | B | harness source strings (Codex prompt.md); Claude Code wording "not verified from source". | keep |
| harness_of 'summar' window | mcp/compaction.py:388 | 2000 chars | detect Hermes summariser | B-value-arbitrary | heuristic window. | keep (the bound is forced); derive the value |
| find_previous min head | mcp/compaction.py:406 | 40 chars | minimum text to match a previous summary | B-value-arbitrary | avoid spurious matches; unmeasured. | keep (the bound is forced); derive the value |
| TIMEOUT (model.py) | mcp/model.py:60 | 3600 s (YAMADORI_MODEL_TIMEOUT) | internal generation HTTP timeout | B-value-arbitrary | breaker; unmeasured. | keep (the bound is forced); derive the value |
| RELEASE_TIMEOUT | mcp/model.py:163 | 10 s (YAMADORI_SLOT_RELEASE_TIMEOUT) | timeout for slot release calls | B-value-arbitrary | breaker. | keep (the bound is forced); derive the value |
| SHRINK_PROMPT | mcp/model.py:167 | "x" | one-token prompt that clears a slot | B | engine: diverges at position 0 from every rendered prompt; erase answers 501 without --slot-save-path. | keep |
| BudgetEvent on finish_reason length | mcp/model.py:63 | rule | a length stop is a budget event, never an answer | B | correctness (summarize_text at 900 returned nothing). | keep |
| context_window = main share | mcp/catalog.py:129 | budget.main | advertised context_length = enforced limit | B | advertised window must equal the enforced one (docs/OPENAI-CONFORMANCE.md C1). | keep |
| OUTPUT_FRACTION | mcp/catalog.py:178 | 5 (window/5) | advertised max_completion_tokens | B | Roo-Code's own fallback `Math.ceil(context_length * 0.2)` (openrouter.ts L121-123); code says "NOT MEASURED"; operator asked 09-24T03:49 "should max output be higher". | keep |
| CONTEXT_FIELDS / OUTPUT_FIELDS | mcp/catalog.py:149 | context_length, max_model_len, context_window / max_completion_tokens… | field names clients read | B | client sources (Hermes model_metadata.py L859-870). | keep |
| HEADROOM_MIB | mcp/gpu_room.py:101 | 1331 MiB (YAMADORI_A4000_HEADROOM_MIB) | free VRAM kept on the A4000 | A-flag | operator "262k is floor, 1.3gb target" (2026-09-24T03:52) was the 5060 Ti's target; applied to the A4000 by us (#16); GiB reading ours. Operator later lowered the 5060 Ti target to 600 MB. | keep; operator to confirm the value/wording |
| A4000 rule (fit with headroom else LRU-unload) | mcp/gpu_room.py:4 | rule | coordinate loads on the A4000 | A | operator 2026-09-24 "If it fits with headroom fine, if it doesn't drop them and load in what you need on use" (#16, config.template.yaml:710-711). | keep |
| ROOM_WAIT_S | mcp/gpu_room.py:106 | 300 s (YAMADORI_GPU_ROOM_WAIT) | wait for the A4000 room lock | B | covers the longest measured draw, 220.5 s at 1344x1344 (docs/IMAGEGEN.md); margin arbitrary. | keep |
| STATE_WAIT_S | mcp/gpu_room.py:108 | 30 s | state-file lock wait | B-value-arbitrary | "this is a breaker". | keep (the bound is forced); derive the value |
| SETTLE_S | mcp/gpu_room.py:111 | 15 s (YAMADORI_GPU_ROOM_SETTLE) | wait for memory to free after unload | B-value-arbitrary | "Not measured". | keep (the bound is forced); derive the value |
| WAIT_POLL_S | mcp/gpu_room.py:115 | 0.5 s | re-check interval while a model is leased | B-value-arbitrary | poll interval. | keep (the bound is forced); derive the value |
| LEASE_MAX_S | mcp/gpu_room.py:117 | 3600 s | stale lease age | B | = model.TIMEOUT, the longest request timeout. | keep |
| evicting-record staleness | mcp/gpu_room.py:405 | 120 s | forget an in-progress eviction marker | B-value-arbitrary | unmeasured. | keep (the bound is forced); derive the value |
| LOG_LOADED_EVERY_S | mcp/gpu_room.py:517 | 60 s | rate-limit 'loaded' log lines | B-value-arbitrary | logging only. | keep (the bound is forced); derive the value |
| FAIL_FAST_WAIT_S | mcp/gpu_room.py:610 | 2 s (YAMADORI_GPU_ROOM_FAST_WAIT) | chat-path wait for the A4000 | B-value-arbitrary | a zero wait skipped embedder loads after restart (live gate 2026-09-24); code: "2 s is a choice". | keep (the bound is forced); derive the value |
| refuse when card unreadable | mcp/gpu_room.py:127 | rule | no on-demand load into unknown VRAM | B | safety (never load into an OOM); pre-deploy review 2026-09-24. | keep |
| eviction breaker | mcp/gpu_room.py:776 | len(evicted) > len(SIZES) | stop an eviction loop | B | correctness bound derived from the table size. | keep |
| SIZES embeddings | mcp/gpu_room.py:161 | 2100 MiB | A4000 need of the embedder | B | arithmetic: weights 610 + f16 KV 896 + ~600 assumed; measured only together (7,565 with reranker + Laya). | keep |
| SIZES reranker | mcp/gpu_room.py:169 | 3000 MiB | A4000 need of the reranker | B | arithmetic: 610 + 1,792 + ~600 assumed. | REMOVED 2026-10-01 with the reranker (docs/REMOVED.md) |
| SIZES bonsai-vision | mcp/gpu_room.py:175 | 9449 MiB | A4000 need of the vision model | B | upper end of config's 8,265-9,449 MiB estimate; never measured alone. | keep |
| SIZES imagegen | mcp/gpu_room.py:181 | 6389 peak / 319 idle | A4000 need of image generation | C | docs/IMAGEGEN.md: +6,389 at 1344x1344 (n=1), +6,281-6,371 at 1024x1024 (n=13), +319 idle (n=1). | keep |
| SIZES imagegen-turbo | mcp/gpu_room.py:186 | 6389 / 319 | A4000 need of turbo | B | borrowed from imagegen (same server/budget); own peak read once 5,527 (n=1). | keep |
| SIZES clm-encoder | mcp/gpu_room.py:192 | 9400 MiB | A4000 need of the CLM encoder | B | arithmetic: 8,300 weights + 288 KV + ~800 assumed. | keep |
| SIZES critic-disabled | mcp/gpu_room.py:198 | 20173 MiB | marks the critic as not fitting | B | arithmetic ~19.7 GiB. | keep |
| cancel on client disconnect | mcp/cancel.py:1 | rule | shuts upstream sockets when the client leaves | B | correctness (#39: orphaned second brain held the helper lane for ~2,000 s). | keep |
| WEEK_START / LAST_DAYS | mcp/token_ledger.py:100 | Monday 00:00 local / 30 days | dashboard token periods | B-value-arbitrary | reporting windows only. | keep (the bound is forced); derive the value |
| token_ledger busy_timeout | mcp/token_ledger.py:136 | 30000 ms | sqlite lock wait | B-value-arbitrary | breaker. | keep (the bound is forced); derive the value |
| PRICED_ROLES excludes warm | mcp/token_ledger.py:94 | main, second_brain, side_call, internal | which generations are priced | B | a hosted API manages its own cache; warm is our prefill. | keep |
| TEMPLATE STALENESS | config.template.yaml:240 | template -c 262144 q4_0 + --kv-mean-center; live config.yaml -c 18124… | template no longer matches the running config | B (stale) | config.yaml:149-162: "2026-09-25 (operator): ROLLED BACK to -c 163840 q8_0" then 181,248. Template also lacks the nudge binary the proxy relies on. | keep |
| healthCheckTimeout | config.template.yaml:25 | 900 s | llama-swap model load timeout | B-value-arbitrary | CONSTRAINTS s.2 row 29 "KEEP" (model load); value unmeasured. | keep (the bound is forced); derive the value |
| startPort | config.template.yaml:27 | 10001 | first upstream port | B | budget.DIRECT reads :10001; a port assignment. | keep |
| sampling macro | config.template.yaml:67 | --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence 0 --repeat… | server-side sampling defaults | A + B | PrismML/Qwen card; operator 2026-09-25T13:02 "keep temp at 1.0"; min_p 0.05 was llama.cpp default, corrected. | keep |
| bonsai -c (live) | config.yaml:162 | 181248 | KV pool size | A + B | operator 2026-09-25T12:44 "600mb free, dial it in"; derivation config.yaml:155-161: 1,579 free - 194 stress - 600 = 785 MiB / 44 KiB = ~18K tokens -> 181,248. | keep |
| bonsai -c (template) | config.template.yaml:240 | 262144 | stale pool | B (stale) | operator 2026-09-24T03:52 "262k is floor, 1.3gb target", rolled back 2026-09-25. | keep |
| --cache-type-k/v | config.template.yaml:241 | q4_0 (template) / q8_0 (live) | main KV cache precision | A | operator 2026-09-25 "How is Q8 going" / summary "Q8 KV cache"; template stale. | keep |
| --cache-type-k/v-draft | config.template.yaml:248 | q4_0 | MTP draft cache precision | A-flag | claimed "2026-09-24 (operator)"; mirrors the reference run's -ctkd q4_0; speed "measured in the G2 run, not assumed" (not found). | keep; operator to confirm the value/wording |
| --spec-type draft-mtp, --spec-draft-n-max 1 | config.template.yaml:171 | MTP, 1 draft token | speculative decoding | A + C | AskUserQuestion 2026-09-23T04:34 "first one" (MTP); acceptance 0.80-0.84 code, 0.61 prose (n=3 prompts). n-max 1 not separately justified. | keep |
| -fa on / --cont-batching / --jinja | config.template.yaml:250 | on | engine features | B | engine requirements (template, flash attention). | keep |
| -b 1024 -ub 512 | config.template.yaml:252 | 1024 / 512 | batch / micro-batch for the main model | B-value-arbitrary | no rationale in file; -ub is the engine default, -b halves the default 2048. | keep (the bound is forced); derive the value |
| --no-cache-idle-slots | config.template.yaml:259 | set | keep idle slots from being cleared on every task | C + B | live gate 2026-09-24: warm after repair reused 0 / processed 3,085; probe n=1 reused 1,561/1,678 without the clear (config.yaml:175-192). | keep |
| -np (absent: auto 4, unified KV) | config.template.yaml:250 | engine auto | four slots over one pool | B | engine default; -np 4 --no-kv-unified measured and rejected: 45,312 cells per slot (#59). | keep |
| --cache-ram (absent: default 8192 MiB) | config.template.yaml:259 | engine default | host-RAM prompt cache used to restore cleared slots | B | engine default; relied on by IDLE CLEAR restore (slots.py:1088-1098, live 516 ms, n=3). | keep |
| --reasoning-budget | config.template.yaml:278 | 32768 | server default thinking breaker | B-value-arbitrary | breaker existence measured (ts_code ran to 20k twice, CONSTRAINTS 1b, n=7); 32,768 "~11x the longest natural completion" arbitrary. Operator 2026-09-25T15:46: "the llama server sets no default thinking budget". | keep (the bound is forced); derive the value |
| --reasoning-budget-message | config.template.yaml:279 | "Thinking budget reached. I will stop ..." | server-side hard-stop text | A + C | same as BUDGET_MESSAGE. | keep |
| --reasoning-format deepseek | config.template.yaml:283 | deepseek | parse <think> delimiters | B | Qwen3-family <think> tags; auto-detect gated on --jinja. | keep |
| --no-context-shift | config.template.yaml:288 | set | fail loudly when the window fills | B | correctness: silent context rotation is data loss. | keep |
| filter reasoning_effort?: medium | config.template.yaml:149 | medium | default effort for direct callers | B-value-arbitrary | mirrors tiers.DEFAULT; the template accepts only low/medium/xhigh. | keep (the bound is forced); derive the value |
| GGML_CUDA_BATCH_INVARIANT=1 | config.template.yaml:162 | set | batch-invariant PTQ1_0 mat-vec | C (weak) | MTP speeds (54 vs 32 tok/s, A4000) were measured with it set; no run without it. | keep |
| bonsai ttl 0 / primary persistent | config.template.yaml:290 | 0 / persistent | main model never unloaded | B | correctness: an exclusive on-demand load would tear down the chat model (config.template.yaml:677-680). | keep |
| trial ttl | config.template.yaml:367 | 1800 s | frees the card if a trial is left loaded | B-value-arbitrary | maintenance-only entries; unmeasured. | keep (the bound is forced); derive the value |
| imagegen / imagegen-turbo ttl | config.template.yaml:466 | 600 s | unload image server after idle | A-flag | operator 2026-09-23T19:42 "I would let image have at least 5 minutes if you tighten it"; 600 ours. | keep; operator to confirm the value/wording |
| imagegen --max-vram 6 | config.template.yaml:459 | 6 GiB | caps sd.cpp managed VRAM | C | without it CUDA1 fell to 124 MiB free; peaks measured n=13/n=1; vae variants n=39 pairs (bench/imagegen). | keep |
| imagegen --steps 20 --cfg-scale 6.0; turbo --steps 4 --cfg 1.0 shifts | config.template.yaml:461 | 20/6.0; 4/1.0/base_shift 0.5 max_shift 0.69355 | default diffusion sampling | B | model's sampling contract (turbo trained on 4 steps, final sigma 0.40; config.template.yaml:475-479). | keep |
| image -t 8 | config.template.yaml:464 | 8 threads | CPU threads for sd.cpp | B-value-arbitrary | unexplained. | keep (the bound is forced); derive the value |
| bonsai-vision ttl | config.template.yaml:531 | 300 s | unload vision after idle | A | operator 2026-09-23T19:42 "I would let image have at least 5 minutes if you tighten it"; code "(operator, 2026-09-23; was 900)". | keep |
| bonsai-vision -c | config.template.yaml:521 | 16384 | vision context | B-value-arbitrary | VRAM fit (9,449 MiB estimate); value unmeasured for need. | keep (the bound is forced); derive the value |
| embeddings -c 8192, -b 2048 -ub 512 | config.template.yaml:551 | 8192 / 2048 / 512 | embedder window and batches | B | VRAM: -b 8192 -ub 2048 left GPU1 at 13.9 of 16.4 GB (config.template.yaml:554-557). | keep |
| reranker -c 16384 | config.template.yaml:585 | 16384 | cross-encoder window | B | correctness: overflow returns 0.0 for every document (config.template.yaml:571-574). | REMOVED 2026-10-01 with the reranker (docs/REMOVED.md) |
| clm-encoder -c 2048 -np 1 -b/-ub 2048 | config.template.yaml:625 | 2048 / 1 / 2048 | CLM encoder window | B | CLM recipe max-model-len 2048; one micro-batch per embedding. | keep |
| llama-swap groups (ondemand exclusive, imagegen swap) | config.template.yaml:728 | swap/exclusive flags | backstop for A4000 co-residence | A + B | operator 2026-09-24 A4000 rule; VRAM arithmetic (vision 8,265-9,449 vs 8,605 free beside retrieval). | keep |
| hooks preload | config.template.yaml:791 | bonsai, embeddings (reranker until 2026-10-01) | models loaded at startup | B | resident models by design. | keep; the reranker's line REMOVED 2026-10-01 with the reranker (docs/REMOVED.md) |

### Fan-out, repair, routing: fanout.py, tool_code.py, code_check.py, route.py, selection.py, progress.py, repeats.py, fusion.py, rings.py, concept_seed.py, session_id.py, ...

| name | file:line | value | what it does | class | evidence or derivation | action |
|---|---|---|---|---|---|---|
| fan-out design A/B/grade/C, sequential | mcp/fanout.py:6 | A + B, grade after two, C tie-breaker | sequential candidates on the helper lane | A | operator_messages.txt 2026-09-23T13:41 '3x fan out ... first brain using second brain ... sequentially'; operator_answers.txt 2026-09-23T13:42 'Sequential via 2nd brain'; 12:46 'Yes, fan out on code' | keep |
| MAX_STEPS | mcp/fanout.py:344 | 3 | most candidates incl. A | A-flag (count ambiguous) | operator_messages.txt 13:41 '3x fan out'; but operator_answers.txt 12:46 option text says '3 candidates + the original' (=4). Code does A+B+C=3 | keep; operator to confirm the value/wording |
| AGREE | mcp/fanout.py:340 | 0.80 | A-B similarity that skips tie-breaker C | D | comment 'UNMEASURED choice (operator, 2026-09-23)': no operator message names 0.80 (claimed, no quote); summary at operator_messages:549 'AGREE=0.80 (an unmeasured constant)' | DONE 2026-09-27: REMOVED: both parsing -> the tie-breaker C runs (the operator's design: C when the check does not separate A and B). test_fanout, test_fanout_delivery updated. |
| LANE_WAIT | mcp/fanout.py:350 | 1200 s (YAMADORI_FANOUT_LANE_WAIT) | wait for helper lane before skipping fan-out | B-value-arbitrary | a wait must be bounded; value: 'Generous ... A choice.' no measurement | keep (the bound is forced); derive the value |
| run timeout | mcp/fanout.py:583 | 3600 s between tokens | inter-token timeout for B/C | B | 'matching every other generation timeout in the stack and llama-server's own -to' | keep |
| VARIANTS evidence/skeptical/terse | mcp/fanout.py:149 | 3 system nudges | alternative-candidate nudges, never run | D | only VARIANTS[0] (empty nudge) is generated; others 'kept, named, for a benchmark arm'; no measurement | PENDING (unassigned): Delete the 3 unused rows; tests test_fanout.py, test_pmndrs_stack.py reference VARIANTS |
| TIEBREAK_PROMPT | mcp/fanout.py:355 | prompt text | C's instruction: both candidates + check results | A-flag (wording ours) | operator_messages.txt:517 quotes operator 'save the third run with details of the first two to improve a third tie breaker run'; wording unmeasured | keep; operator to confirm the value/wording |
| seed on B and C | mcp/fanout.py:649 | one seed per job | concept seed per second-brain run | A | operator_messages.txt 2026-09-23T13:46 'Every second brain run gets a seed word always too'; 2026-09-24T14:28 seed in user message, same on replay | keep |
| strip_seed_echo | mcp/fanout.py:252 | leading comment == seed word dropped | removes our seed word leaked into code | B | correctness: our injected word must not reach delivered code; LOG #13 live gate '# humanidad' first line | keep |
| _ASKED_LANG / with_language | mcp/fanout.py:413 | regex; fence bare code that parses | read untagged answer as the asked language | D | LOG/comment: one live-gate cache test 'Write a Python function fib(n)', n=1; not measured | PENDING (unassigned): Remove: untagged answers take the prose path (recorded only). Tests: test_fanout.py |
| _CODE_WORD + >=2 lines | mcp/fanout.py:417 | regex, 2 lines | bare-code detection for with_language | D | same n=1 origin as with_language | PENDING (unassigned): Goes with with_language |
| largest-block rule | mcp/fanout.py:502 | largest fenced block, ties to later | which block is a candidate's code | B | mirrors our own grader bench/domain/grade.py (consistency with how answers are graded) | keep |
| tiebreak error excerpt | mcp/fanout.py:1000 | 5 errors, 160 chars each | errors shown to C | B-value-arbitrary | a bound on prompt size; values ours | keep (the bound is forced); derive the value |
| code vs prose split | mcp/fanout.py:1151 | > half of answers carry code | decides code medoid vs path vote | D | no evidence given | PENDING (unassigned): Use A's language only (already computed in run()); tests test_fanout.py |
| SIM_TIE | mcp/fanout.py:877 | 0.01 | medoid tie band | D | 'A choice, not a measurement' | PENDING (unassigned): Remove: exact ties only, then prefer C. No test references it |
| NGRAM | mcp/fanout.py:880 | 3 | token n-gram width for similarity | B-value-arbitrary | a width must exist; 'Also a choice' | keep (the bound is forced); derive the value |
| MAX_TOKENS (fanout) | mcp/fanout.py:882 | 20000 | token bound per candidate for similarity | B-value-arbitrary | bounded work; value ours | keep (the bound is forced); derive the value |
| medoid tie-break order | mcp/fanout.py:1214 | warnings, truncated, \|len-median\| | breaks similarity ties | D | no measurement; 'never the shortest' from 5/6 records decided by length (a count, not a grade) | PENDING (unassigned): Prefer C on tie, else A. Tests test_fanout.py |
| prose fan-out: B recorded, A delivered | mcp/fanout.py:659 | - | prose B never replaces A | A-flag | operator_messages.txt 2026-09-24T02:08 'it can't throw that away' (hand-back); delivery rule ours | keep; operator to confirm the value/wording |
| dissent_note threshold | mcp/fanout.py:1337 | agreement < 0.75 | appends 'unsettled' note to answer | D | no evidence; user-visible injected text (proxy.py fanout.dissent_note) | DONE 2026-09-27: fanout.dissent_note DELETED and proxy's call; _fan_out returns (record, winner). test_fanout updated. |
| dissent_note text | mcp/fanout.py:1340 | 'Answered N ways; only X% agreed ... Treat this as unsettled.' | text added to delivered answer | D | wording ours, unmeasured | DONE 2026-09-27: DELETED with the function. |
| break_tie margin | mcp/fanout.py:1312 | 0.15 | Laya pick floor for a tie | D (dead code) | docs/SELECTION.md:230 LAYA_MARGIN_FLOOR; flip band 0.012-0.184 measured contains 0.15; no caller in repo; Laya off under YAMADORI_E1 | PENDING (unassigned): Delete break_tie/_choice_averaged (no callers) |
| _choice_averaged caps | mcp/fanout.py:1271 | state[:2000], timeout 30 | Laya /decide call | D (dead code) | no caller | PENDING (unassigned): Delete |
| ALT_MIN_WORDS | mcp/fanout.py:1367 | 4 | min content words for a prose 'point' | D | 'Every threshold here is a CHOICE, not a measurement' | PENDING (unassigned): Hand back B's text whole (or its differing paths/symbols only). No test reference |
| ALT_NOVELTY | mcp/fanout.py:1368 | 0.5 | share of new words to count as a point | D | same comment | PENDING (unassigned): same |
| ALT_MAX_POINTS | mcp/fanout.py:1369 | 6 | cap on handed-back points | D | same comment; a cut with marker like MAX_FINDING_CHARS | DONE 2026-09-27: REMOVED: every differing point crosses. test_fanout updated. |
| ALT_MAX_CHARS | mcp/fanout.py:1370 | 1200 | cap on handed-back chars | D | same comment | DONE 2026-09-27: REMOVED (same class as MAX_FINDING_CHARS). |
| CODE_NOTE_NAMES | mcp/fanout.py:1371 | 5 | APIs named per loser in code note | D | same comment | DONE 2026-09-27: REMOVED: every differing name is listed. |
| code hand-back note text | mcp/fanout.py:1513 | 'The other candidates differed: ... uses none of these.' | note after delivered code | A-flag (wording ours) | operator_messages.txt 2026-09-24T02:08 second brain's work must hand back; wording unmeasured | keep; operator to confirm the value/wording |
| _STOPWORDS (fanout) | mcp/fanout.py:1375 | word list | content words for novelty | D | no evidence | PENDING (unassigned): Goes with ALT_NOVELTY |
| UPSTREAM | mcp/fanout.py:134 | 127.0.0.1:11434 | model server address | B | llama-swap loopback port (AGENTS.md) | keep |
| check/repair by tier | mcp/tool_code.py:25 | repair at high+, note at medium | tool-call code check and fix-up | A | operator_answers.txt 2026-09-23T12:32 'On at high and max'; operator_messages.txt 2026-09-24T12:04 'auto repair of code the agent is creating' | keep |
| REPAIR_ROUNDS | mcp/tool_code.py:97 | 3 (YAMADORI_REPAIR_ROUNDS) | fix-up rounds cap | B-value-arbitrary | a bound is needed; operator 2026-09-24T12:04 'auto looping to repair ... until it compiles'; value: 'Not measured ... n=1' | keep (the bound is forced); derive the value |
| UNKNOWN_MIN_CHARS/LINES | mcp/tool_code.py:101 | 200 chars / 3 lines | log unrecognised code-sized args | D (log only) | no evidence; affects only a log line | PENDING (unassigned): Log every unrecognised client call. No test reference |
| KNOWN table | mcp/tool_code.py:134 | harness write/edit shapes | detect file-writing calls | B | harness schemas; VERIFIED rows cite source files, UNVERIFIED rows flagged | keep |
| shape key regexes | mcp/tool_code.py:255 | key-name lists | fallback detection by argument shape | B | common harness argument names; detection only enables a check | keep |
| path-under-unusual-key length | mcp/tool_code.py:304 | < 400 chars | short one-line value = path | B-value-arbitrary | value ours | keep (the bound is forced); derive the value |
| READ_KNOWN | mcp/tool_code.py:211 | 4 read tools | read detection for progress.step_kind | D (feature) | built for #54 unchanged-read line (removed 2026-09-27) and #53 step caps (Octopus v0e, 'CHOICES'); rows themselves from harness schemas | KEPT 2026-09-27: skill_select reads tool_code.read_target (skills owner's). |
| edit block rule | mcp/tool_code.py:667 | block only if old parses and new does not | when an edit blocks | B | correctness: a fragment cannot be judged out of context | keep |
| formatter never applied | mcp/tool_code.py:993 | record only | prettier/ruff output not written | B | correctness, LOG #24: patch failed because disk text differed from model's (Octopus V0 events 55-57) | keep |
| note phrases Verified/Repaired/Checked | mcp/tool_code.py:1007 | note text before tool calls | one-line mechanical note | A (phrases); wording ours | operator_messages.txt 2026-09-24T14:05 'We can have our output say ... "verified" "repaired"'; sentence wording revised by LOG #34 | keep |
| note error excerpt | mcp/tool_code.py:1009 | 80 chars | first error message cut | B-value-arbitrary | value ours | keep (the bound is forced); derive the value |
| no note for fragments | mcp/tool_code.py:1030 | - | fragment edits get no note | B | truthfulness: no problem was found; LOG #36 model imitated such notes | keep |
| FIX-UP SCOPE (scope) | mcp/tool_code.py:773 | project files only (fixup_project_only, YAMADORI_FIXUP_PROJECT_ONLY) | scratch/temp writes noted, not repaired | D | LOG #56: Octopus v0b test_full.js 2,241 s, v0e /tmp/harness.js; 'built, not yet live'; no operator quote | DONE 2026-09-27: DELETED (tool_code.scope and its call). |
| image guard, image tools only | mcp/tool_code.py:1104 | YAMADORI_IMAGE_GUARD=1 | stop image data in image-tool args | A | operator_messages.txt 2026-09-26T13:04 'harden the tool calls around image generation'; need B: a model cannot write image bytes (LOG #44: 23+40 min generations) | keep |
| IMAGE_REGENERATIONS | mcp/tool_code.py:1109 | 2 (YAMADORI_IMAGE_REGENERATIONS) | rewrites before the turn lands | D | 'A CHOICE'; no evidence | PENDING (unassigned): Land on first stop, or rely on tiers.tool_turn_limit. Tests test_image_guard.py; AGENTS.md image guard para |
| _MAGIC | mcp/tool_code.py:1137 | PNG/JPEG/GIF/WebP base64 heads | recognise image data | B | file-format magic bytes, base64-encoded | keep |
| _MAGIC_AT | mcp/tool_code.py:1139 | 24 | base64 chars needed before judging | B-value-arbitrary | a run length is needed; 24 ours | keep (the bound is forced); derive the value |
| _DATA_WAIT | mcp/tool_code.py:1142 | 80 | chars to wait for a data: header | B-value-arbitrary | value ours | keep (the bound is forced); derive the value |
| _Scan key buffer | mcp/tool_code.py:1266 | 256 | key buffer cap | B-value-arbitrary | memory bound | keep (the bound is forced); derive the value |
| image param shape regexes | mcp/tool_code.py:1123 | _IMAGE_NAME/_IMAGE_WORDS/_LOCATOR_NAME | which undeclared args take images | A-flag (rule ours) | direction operator_messages.txt 2026-09-26T13:04; name lists unmeasured | keep; operator to confirm the value/wording |
| image_result text | mcp/tool_code.py:1386 | 'NOT EXECUTED ... don't paste image data into a tool call.' | tool result for a stopped call | A-flag (wording ours) | failure return shape (AGENTS.md rule); comment: 'The wording is a prompt and a CHOICE' | keep; operator to confirm the value/wording |
| JS_SCRIPT_TYPES / inline scripts | mcp/tool_code.py:529 | JS MIME types | check inline <script> in .html writes | B | HTML spec script types; docs/HARNESS-PI.md gap 4 | keep |
| FORMAT_TIMEOUT | mcp/code_check.py:81 | 20 s (YAMADORI_FORMAT_TIMEOUT) | formatter/linter subprocess timeout | B-value-arbitrary | subprocess must be bounded; value ours | keep (the bound is forced); derive the value |
| MAX_CHARS (check_code) | mcp/code_check.py:85 | 200,000 | check_code tool input cap | B-value-arbitrary | 'Bigger than any single file ...' no measurement; tool no longer on main | keep (the bound is forced); derive the value |
| MAX_ERRORS | mcp/code_check.py:86 | 8 | syntax/lint errors kept per check | B-value-arbitrary | silent cut of errors sent to fix-up; value ours | keep (the bound is forced); derive the value |
| CODE_LANGS | mcp/code_check.py:117 | 8 languages | repair looks only at code, not JSON/YAML | D | rationale 'illustration with ...' unmeasured | PENDING (unassigned): Low impact; keep or measure |
| STD_WIDTHS | mcp/code_check.py:287 | 80, 88, 100, 120 | snap inferred line width | B | formatter defaults: prettier/clang 80, ruff/black 88, rustfmt 100 | keep |
| infer_width min lines | mcp/code_check.py:334 | 10 | lines needed to infer width | D | no evidence; info only (format never applied) | PENDING (unassigned): Low impact |
| indent deltas | mcp/code_check.py:326 | 2,3,4,8 | usual indent units | B | conventional indent widths | keep |
| DEFAULTS | mcp/code_check.py:567 | per-language style | formatter defaults | B | documented defaults of ruff, prettier, rustfmt, clang-format LLVM | keep |
| LINT_SELECT | mcp/code_check.py:775 | E9,F63,F7,F82 | ruff errors that block a Python write | B | flake8's fatal set (its docs / GitHub starter workflow): errors, not style | keep |
| formatter parse error = syntax | mcp/code_check.py:816 | - | prettier/rustfmt parser rejections block | B | stricter reference parsers than tree-sitter | keep |
| loops_that_cannot_end | mcp/code_check.py:1063 | provable non-terminating while | defect warning | B (flag: origin LiveBench n=5) | a loop that cannot end is a bug; 'deliberately narrow'; false-positive rate unmeasured; counts in review_answer when the question has no code | keep |
| continuation defects | mcp/code_check.py:1156 | redeclares_signature, body_not_indented, unclosed string | check_code continuation facts | B (flag: LiveBench origin) | parser facts; categories from 'the operator's reading of those five' LiveBench misses | keep |
| check_against_prompt general rule | mcp/code_check.py:1695 | parses alone OR appended | completion-aware check | A | operator_answers.txt 2026-09-23T20:37 'General rule (Recommended)' | keep |
| review_answer feedback text | mcp/code_check.py:1866 | 'Your answer's code was checked mechanically ... Write your complete… | repair feedback prompt | D (dead text) | no reader of `feedback` in proxy.py/shomen.py; fix-up job replaced the main-side repair | PENDING (unassigned): Delete feedback text; tests test_code_check.py |
| fenced_blocks | mcp/code_check.py:1607 | 3+ fence, indent <= 3 | markdown fences | B | CommonMark spec | keep |
| formatter isolation | mcp/code_check.py:33 | --isolated, temp HOME, no config | sandbox formatters | B | security: server never reads user disk or config | keep |
| router classes and order | mcp/route.py:12 | 6 classes, first rule wins | one class per request | A + C | operator_messages.txt 2026-09-24T12:04 'really good code detection and routing so we don't route tool calls or random requests through the code ... pipelines'; bench/route/eval_route.py n=1,024, 0 misroutes (in-sample) | keep |
| _NOTICE | mcp/route.py:152 | Hermes resume heads | harness notice = agent_step | B | captured producer words, corpus turns 3409-3699 | keep |
| _CONTINUE | mcp/route.py:154 | bare continue, <= 60 trailing chars | continue turn = agent_step | C (in-sample) | eval_route.py agent_step recall 0.995; 60 unmeasured | keep |
| _CODE_TAGS/_TEXT_TAGS | mcp/route.py:173 | tag lists | fence tag says code or not | B | language names | keep |
| _UNTAGGED_ERROR_SHARE | mcp/route.py:189 | 0.25 | untagged fence counts as code | D | 'a choice, checked on the fixtures in mcp/test_route.py, not a measurement'; not swept by eval_route.py | PENDING (unassigned): Sweep on bench/route/labels.jsonl or require 0 error lines; tests test_route.py |
| _UNTAGGED_GRAMMARS | mcp/route.py:190 | python, typescript, rust | grammars tried on untagged fences | D | no evidence | PENDING (unassigned): Try every CODE_LANG; test_route.py |
| placeholder_only | mcp/route.py:225 | comments-only block | starter placeholder is not code | C (in-sample; LiveBench template) | LiveBench '# YOUR CODE HERE'; eval_route.py | keep |
| _WRITE/_NOUN/_CODE_REQUEST | mcp/route.py:268 | verb + noun within 80 chars | code_generation signal | C (in-sample) | eval_route.py code_generation recall 0.978 precision 0.88; AGENTS.md: 'In-sample: the rules were adjusted while reading these misses'; 80 unmeasured | keep |
| _EXPORT_NAME/_WRITE_IN_LANG/_NEED_CODE | mcp/route.py:294 | 3 regexes | extra code-request shapes | C (in-sample, benchmark-fitted) | 'each from the benchmark prompts the proxy is measured on (bench/domain/tasks*.jsonl)' | keep |
| _WRITE_FILE | mcp/route.py:319 | write verb + source-file name | code_generation for 'write stats.py' | C (in-sample) | LOG #14: eval_route old vs new, code turns missed 2 -> 0; written for the case | keep |
| _EDIT_VERB | mcp/route.py:324 | verb list | code_edit with code present | C (in-sample) | eval_route.py code_edit recall 0.812 precision 0.963 | keep |
| _QUESTION_LEAD/_REQUEST_LEAD | mcp/route.py:328 | regexes | question vs request | C (in-sample) | eval_route.py prose/library_question rows | keep |
| library_question gate | mcp/route.py:457 | READABLE_GATE or held symbol | library_question vs prose | C (weak) | held-out 88->91/120, p=0.375 not significant (selection.py:94) | keep |
| LAYA_ROUTE_TIMEOUT | mcp/selection.py:145 | 10 s (YAMADORI_LAYA_ROUTE_TIMEOUT) | Laya /route call timeout | B-value-arbitrary | '~50x the measured cost'; Laya off under YAMADORI_E1 | keep (the bound is forced); derive the value |
| MIN_QUESTION_CHARS | mcp/selection.py:149 | 8 | shortest question deep thinking takes | D | no evidence; on the trigger path a fired trigger bypasses it, so it gates the legacy path only | PENDING (unassigned): Remove (legacy path) ; no test reference |
| CONTEXT_CHARS | mcp/selection.py:154 | 1500 | earlier user text sent as context | D | no evidence; legacy path + Laya/E1 context | PENDING (unassigned): Remove with legacy path; no test reference |
| rule_baseline regex | mcp/selection.py:258 | 6-line regex | legacy investigate/clarify/answer | C | bench/eval_route_heldout.py regex 73/120; bench/data/rule_baseline_golden.json 89 preds; legacy path only | keep |
| named_symbols cap | mcp/selection.py:187 | 8 | symbols kept | B-value-arbitrary | legacy | keep (the bound is forced); derive the value |
| MAX_PROBES | mcp/selection.py:316 | 400 | names probed per request | B-value-arbitrary | bounds SQL work | keep (the bound is forced); derive the value |
| SQL IN chunk | mcp/selection.py:470 | 200 | IN-list size | B | SQLite variable-number limit (999 in older builds) | keep |
| probe shape rules (_MID_CAPITAL, _NOT_API, no bare ALLCAPS) | mcp/selection.py:304 | regexes | which names are probed | C (in-sample) | test_selection: 0/342 LCB fire, 25/26 ce; held-out 88->91/120 p=0.375; ALLCAPS rule from a Hermes Octopus spec incident | keep |
| READABLE_GATE | mcp/selection.py:942 | 4 gate situations | evidence the request is about held source | C | 'Measured necessary 2026-09-22': 342 LCB prompts opened the gate, 65 would investigate | keep |
| _DESIGN | mcp/selection.py:491 | regex | legacy fan-out on design questions | D | 'n=0 labels'; only used when route is None (legacy) | PENDING (unassigned): Delete; proxy passes a route. No test reference |
| _CODE_TASK | mcp/selection.py:504 | regex | legacy fan-out on code tasks | A-flag (regex ours) | operator_answers.txt 2026-09-23T12:46 'Yes, fan out on code'; legacy only | keep; operator to confirm the value/wording |
| ATTACHMENT_DELIMITERS | mcp/selection.py:541 | '--- Attached Context ---' | split instruction from attachment | B | captured Hermes format, corpus event 3396 | keep |
| acts_locally regex | mcp/selection.py:578 | lead+verb+object/place (100-char window, 3 modifiers) | agent_step: act on user's machine | C (in-sample) | eval_route.py agent_step; origin: Hermes octopus-invaders start, 26,660 prompt tokens of deep thinking; windows unmeasured | keep |
| utility_call rule | mcp/selection.py:755 | no tools + one exchange + contract | client side call -> bare model | C | mcp/test_utility.py: 41/41 side calls, 0/954 task turns, 0/4,317 benchmark prompts (in-sample corpus) | keep |
| _CLOSED_FORMS windows | mcp/selection.py:668 | 40/30/60/20 chars | reply-contract regexes | C (in-sample) | test_utility.py replay; windows unmeasured | keep |
| SUMMARY_HEAD_CHARS | mcp/selection.py:689 | 1000 | head scanned for summarise/title | B-value-arbitrary | 'holds the Hermes compaction's whole instruction (first noun at ~110)' | keep (the bound is forced); derive the value |
| _TITLE | mcp/selection.py:712 | harness title words | title side calls | B | Hermes corpus 510e1d; OpenCode 1.18.32 source | keep |
| image request never utility | mcp/selection.py:764 | - | image requests go to vision path | B | a text model cannot see (pre-deploy review 2026-09-24) | keep |
| title/summary image exception | mcp/selection.py:770 | - | title/summary with image stays utility | A-flag (claimed, no quote found) | comment '(operator, 2026-09-26)'; no matching message in operator_messages.txt | keep; operator to confirm the value/wording |
| legacy disagree -> investigate | mcp/selection.py:1104 | - | Laya/rule disagreement escalates | C (weak) | 91/120 vs 89/120, p=0.79 not significant (AGENTS.md Laya table); legacy only | keep |
| _MAX_NAMED | mcp/progress.py:56 | 60 | task-named paths kept | B-value-arbitrary | value ours | keep (the bound is forced); derive the value |
| _TEMP | mcp/progress.py:63 | temp-dir regex | temp paths are not project | D (feature) | exists for #56 scope and #52 label rule 2 (Octopus) | PENDING (unassigned): Remove with scope; tests test_progress.py |
| SCRATCH_NAMES | mcp/progress.py:67 | test_full, harness, diag*, dbg* | scratch-harness file names | D | 'a CHOICE; each row names where it was seen' -- every row an Octopus v0b/v0e run | PENDING (unassigned): Delete table. Tests test_progress.py; AGENTS.md:625 |
| dot-file rule | mcp/progress.py:255 | basename starts with '.' | dot-files are not project | D (feature) | #56 (.runtime_test.js, .dbg*.js Octopus) | PENDING (unassigned): Remove with scope |
| project inference (named_paths, tree_paths, stated_root, _CWD_WORDS, _infer_roo… | mcp/progress.py:81 | regexes + widening rule | infer working dir and project files | D (feature) | built from the v0f-V0 Octopus spec; consumers are #56 scope and #52 'helped needs a project write' | PENDING (unassigned): Remove with those features; tests test_progress.py |
| _INSPECT | mcp/progress.py:360 | command list | inspect-only terminal step | D | 'a CHOICE; bench/octopus/overthinking classify.py' -- Octopus | KEPT 2026-09-27: skill_select reads progress.inspect_only (skills owner's). |
| _WRITE_OPS/_REDIRECT/_SEGMENTS | mcp/progress.py:348 | regexes | does a shell command write | D (feature) | support for _INSPECT | KEPT 2026-09-27: skill_select reads _REDIRECT, _SEGMENTS and inspect_only (skills owner's). |
| SEARCH_NAMES | mcp/progress.py:383 | search tool names | search results count as reads | D (feature) | rows from harness fixtures, feature is #53 | DONE 2026-09-27: DELETED with step_kind. |
| step_kind | mcp/progress.py:412 | read / error / other | picks agent-step thinking cap (tiers.AGENT_STEP_THINKING_READ 2048 / _ERROR 6144) | D | LOG #53 Octopus v0e; caps 'CHOICES'; 'built, not yet live' | DONE 2026-09-27: DELETED (and the step_thinking switch). test_progress: test_step_kind replaced by test_inspect_only. |
| _LOCKS clear | mcp/progress.py:447 | 4096 | lock table bound | B-value-arbitrary | memory bound | keep (the bound is forced); derive the value |
| compaction reinject counters | mcp/progress.py:534 | compactions > reinjected | re-inject work log after compaction | A-flag | LOG #54 says it 'stays -- a mechanic of the work log' (our words); operator 2026-09-24T14:05 'ensure compaction works' (direction) | keep; operator to confirm the value/wording |
| removed situations still replay | mcp/progress.py:15 | ledger replays stored lines | old PROGRESS/UNCHANGED lines persist in old conversations | D (residue) | removed after operator_messages.txt 2026-09-27T12:41 'realtime target corrections'; 'Lines already recorded in the ledger replay byte for byte' | PENDING (unassigned): Accept (cache) or purge ledger rows holding them |
| normalise key rules | mcp/repeats.py:45 | ignore paging keys; strip */\. from glob/path | detect same search again | B | observed 14 permuted retries on an unindexed dir (AGENTS.md, LOG #3); a key for 'same query' | keep |
| cached_empty | mcp/repeats.py:78 | reuse empty result | skip re-running an identical empty search | B | determinism: same query over an unchanged index; corpus 12 repeats over 842 s | keep |
| guidance trigger | mcp/repeats.py:102 | seen >= 2 and empty | when to append repeat guidance | D | no evidence for 2 | DONE 2026-09-27: Turn.guidance DELETED with proxy's three calls. |
| guidance text | mcp/repeats.py:105 | '[This request has now been made N times ...] Still available this tu… | appended to our tool results | D | 'measured' margins 0.288/0.493 are Laya's decision model, not the 27B; wording unmeasured | DONE 2026-09-27: REMOVED. |
| _forward_hint texts | mcp/repeats.py:115 | 4 hint sentences | next-step hint per tool | D | wording ours, unmeasured | DONE 2026-09-27: REMOVED (the function is gone). |
| _empty head | mcp/repeats.py:160 | 140 chars + phrases | detect empty results | B | phrases are our tools' own wording; 140 arbitrary | keep |
| RESULT_CAP | mcp/repeats.py:180 | 6000 chars | cut our tool results with a marker | B-value-arbitrary (D-flag) | '6000 stays, as a BREAKER'; 3 of 651 corpus results hit it; same kind of invented 6000 cut as MAX_FINDING_CHARS the operator rejected 2026-09-27T13:31 | keep (the bound is forced); derive the value |
| cut at newline > limit/2 | mcp/repeats.py:202 | 1/2 | line-boundary cut | B-value-arbitrary | value ours | keep (the bound is forced); derive the value |
| STOP | mcp/fusion.py:41 | stopword list | lexical stopwords | D | 'Deliberately small'; no measurement | PENDING (unassigned): Use no stoplist (BM25 idf discounts common words) |
| content_words min length | mcp/fusion.py:61 | > 2 chars | drop short parts | D | no evidence | PENDING (unassigned): Low impact |
| BM25 k1, b | mcp/fusion.py:88 | 1.5, 0.75 | lexical scoring | B | standard BM25 parameters (b=0.75 canonical, k1 in 1.2-2.0) | keep |
| RRF_K | mcp/fusion.py:107 | 60 | rank fusion constant | B | Cormack et al. RRF original value; 'no reason to tune it on 11 queries' | keep |
| vote-count-first ordering | mcp/fusion.py:138 | found_by count, then RRF | result order | D | 'deliberately'; evidence n=11 queries | PENDING (unassigned): Plain RRF order |
| tier rule | mcp/fusion.py:150 | symbol / both rankers / one | taproot/branch/shoot tiers | C (weak) + A names | n=11 (3/11 rank-1, 7/11 top-5); names from AGENTS.md naming table | keep |
| RANKERS exclude symbol | mcp/fusion.py:147 | semantic, lexical | symbol is a tag, not a rank | B | correctness: symbol rows are table-ordered | keep |
| detect_repo weights | mcp/detect.py:110 | cwd x10, recency 1+2i/n | vote for repo root | D (dead on request path) | no evidence; proxy.resolve_repo returns no root (guard.py:26); only scripts/test_detect.py, scripts/test_hermes_detect.py call it | PENDING (unassigned): Delete module and the two scripts |
| _root_of depth | mcp/detect.py:64 | 40 levels | walk-up bound | B-value-arbitrary | loop bound | keep (the bound is forced); derive the value |
| _NEVER/MARKERS | mcp/detect.py:33 | dir and marker lists | project root markers | B | ecosystem manifest names | keep |
| SECRET_NAMES/WORDS/CONTENT | mcp/guard.py:50 | regexes | never index credentials | B | security; incidents: secrets.json, credentials.toml, deploy.ps1 were indexed; vendor key formats | keep |
| MAX_SNIFF | mcp/guard.py:81 | 65536 | bytes scanned for secrets | B-value-arbitrary | bound; value ours | keep (the bound is forced); derive the value |
| operator-only root approval | mcp/guard.py:155 | - | requests cannot choose a directory | B | security hole closed 2026-09-22 (commit 7f8f062) | keep |
| identity sources and order | mcp/session_id.py:13 | pck, header, tool_call_id, summary line, minted | explicit session id | A | operator_messages.txt 2026-09-25T15:14 'we can't assume a new session is a resumable one'; 19:10 'session id and tool call id is sound, lets do it' | keep |
| no visible line in answers | mcp/session_id.py:28 | - | id rides in tool-call ids | A-flag (claimed, no quote) + B | comment '(operator, 2026-09-25: no visible line)' has no matching message; B: 'Reply with exactly: ok' broke live | keep; operator to confirm the value/wording |
| ID_HEX | mcp/session_id.py:73 | 12 | session id length | B | must stay under Hermes' 16-hex credential redaction (agent/redact.py) and be unique | keep |
| CALL_SUFFIX_HEX | mcp/session_id.py:78 | 8 (26-char ids) | per-call uniqueness | B-value-arbitrary | uniqueness required (ledger keys call turns by id); 32 bits ours | keep (the bound is forced); derive the value |
| CALL_ID _d<n> | mcp/session_id.py:79 | regex | tolerate Hermes duplicate suffix | B | Hermes uniquify_tool_call_ids | keep |
| MAX_CACHE_KEY | mcp/session_id.py:81 | 1024 | prompt_cache_key length bound | B-value-arbitrary | bound; value ours | keep (the bound is forced); derive the value |
| answer_record source | mcp/session_id.py:84 | - | text-only answers keep the session | A-flag (pending) | AGENTS.md: 'a deviation from the brief, for the operator to confirm' | keep; operator to confirm the value/wording |
| summary line format | mcp/session_id.py:58 | 'yamadori session <12 hex>' | id in compaction summaries | B + A-flag | survives Hermes redaction (no =/:, < 16 hex); design ours | keep |
| hash truncations | mcp/session_id.py:213 | 20 / 32 / 8 hex | key lengths | B-value-arbitrary | collision bound; values ours | keep (the bound is forced); derive the value |
| key-change alias / fork rules | mcp/session_id.py:216 | - | join client keys across compaction | A-flag (coordinator, not operator) | 'coordinator's decision, 2026-09-26' | keep; operator to confirm the value/wording |
| promptCacheKey alias | mcp/session_id.py:192 | camelCase key | OpenCode cache key | B | Vercel AI SDK provider (docs/HARNESS-OPENCODE.md) | keep |
| seed in user message, every second-brain run | mcp/concept_seed.py:254 | - | concept seed injection point | A | operator_messages.txt 2026-09-24T14:28 'seed word concept implemented in the user message ... same rules of cache replay'; 2026-09-22T21:00 original idea | keep |
| phrase template | mcp/concept_seed.py:287 | 'Inspiration word: X' | seed text | B | the original project's template (ClancyDennis/concept-seed); shomen.seed_phrase adds wording (out of scope) | keep |
| away_from median ceiling | mcp/concept_seed.py:181 | median similarity | draw only words far from prompt | D | 'Whether far beats merely unrelated is not yet measured' | PENDING (unassigned): Uniform draw |
| orthogonal frame | mcp/concept_seed.py:170 | QR frame per fan-out | distinct seeds | B | math: guarantees distinct directions; 20,000-draw figure cited without a script | keep |
| redraw loop | mcp/concept_seed.py:187 | 32 | redraw attempts | B-value-arbitrary | loop bound | keep (the bound is forced); derive the value |
| centroid word regex | mcp/concept_seed.py:150 | [a-z]{4,14} | prompt words for centroid | D | no evidence | PENDING (unassigned): Goes with away_from |
| FNV-1a u32 | mcp/concept_seed.py:136 | 32-bit FNV-1a | seed number for dashboard | B | must match the dashboard's JS | keep |
| read limit default | mcp/rings.py:86 | 60 (proxy uses 40) | entries read back | B-value-arbitrary | bound; values ours; proxy.py rings.read(limit=40) on post-compaction re-injection | keep (the bound is forced); derive the value |
| detail excerpt | mcp/rings.py:143 | 140 chars | detail line cut | B-value-arbitrary | value ours | keep (the bound is forced); derive the value |
| closing line | mcp/rings.py:147 | 'Work listed above is DONE. Do not redo it. ...' | instruction in the re-injected work log (main) | D | prohibition with no measured failure ('Observed on frontier models', no evidence); injected into main after compaction (proxy.py rings.read) | DONE 2026-09-27: REMOVED ('Work listed above is DONE. Do not redo it.'). test_rings updated. |
| empty-log text | mcp/rings.py:117 | '... record it now.' | instruction on empty log | D | wording ours; record_step not on main | DONE 2026-09-27: REMOVED the 'record it now' instruction; the situation stays. |
| CHECKS RUN header | mcp/rings.py:128 | 'this is the evidence work actually landed' | section heading | D | wording ours | DONE 2026-09-27: REMOVED the claim; plain heading 'CHECKS RUN:'. |
| record_step / read_rings descriptions | mcp/rings.py:162 | tool descriptions | second brain's work-log tools | D | rewritten 2026-09-23 from one Hermes session (record_step once in 132 turns); unmeasured | PENDING (unassigned): Plain description; test_rings.py |
| KINDS | mcp/rings.py:60 | did/learned/check/decided | entry kinds | B | schema of our own log | keep |
| one_system rule | mcp/system_roles.py:44 | merge leading system/developer; later -> user | one system block | B | served template raises 'System message must be at the beginning.' | keep |
| TEXT_PARTS | mcp/message_text.py:28 | text, input_text, output_text | text part types | B | Chat Completions / Responses protocol | keep |
| HARNESS_CONTEXT_TURNS | mcp/message_text.py:68 | codex <environment_context> | harness context is not the user | B | captured @openai/codex 0.157.1 (docs/HARNESS-CODEX.md) | keep |
| plain RULES | mcp/plain.py:39 | 8 regex rules | controlled-English lint of our text | B (offline only) | ASD-STE100; no request-path caller (test_plain.py only); '3/6->5/6' was Laya's margin | keep |
| MAX_WORDS | mcp/plain.py:67 | 20 | sentence length limit | B | ASD-STE100 procedural limit 20 words | keep |
| report limit | mcp/plain.py:96 | 12 | violations listed | B-value-arbitrary | display bound | keep (the bound is forced); derive the value |

### Skills and CLM: skill_*.py, skills.py, clm.py, clm_heads.py, dash_skills.py, recent_turns.py

| name | file:line | value | what it does | class | evidence or derivation | action |
|---|---|---|---|---|---|---|
| SKILL_TOKENS_HARD | mcp/skill_limits.py:77 | 450 tokens | Validator fails/drops trailing items of a skill body over this | A-flag (number ours) | Operator 2026-09-24 12:58 "skills need a cap of length per skill ... set fuzzy targets"; 2026-09-26 12:36 "compact"; 2026-09-27 08:46 "the input skills need budgeted". 450 itself: skill_limits.py:4 "THESE ARE CHOICES, NOT MEASUREMENTS". Agent Skills spec bound is <5k tokens (DECISION-TREES §362). | keep; operator to confirm the value/wording |
| SKILL_TOKENS_AIM | mcp/skill_limits.py:76 | (100, 300) tokens | Soft aim told to prompts; validator notes a body over 300 | A-flag (number ours) | Same operator direction ("fuzzy targets"); 300 cited to nothing measured. Store audit: 49% of 535 armed skills exceed it (SKILL-FACTORY.md:384). | keep; operator to confirm the value/wording |
| MAX_ITEMS | mcp/skill_limits.py:78 | 6 | Max guidance items per skill; extra items dropped | A-flag (number ours) | Operator "compact"; "a handful of moves" is our phrasing (SKILL-FACTORY.md:375). No measurement. | keep; operator to confirm the value/wording |
| MIN_ITEMS | mcp/skill_limits.py:79 | 1 | A skill with zero surviving items fails | B | Correctness: a skill with no items injects only a title. | keep |
| MAX_ITEM_CHARS | mcp/skill_limits.py:82 | 300 chars | Item longer than this dropped (distil) or row excluded (compile) | C-weak | skill_limits.py:80 "recipe corpus runs p50 166 / p90 233 characters (2,575 rows, 2026-09-26), so 300 keeps ~95%". Describes the corpus, no script cited, not a quality measure. | keep |
| MAX_TITLE_CHARS | mcp/skill_limits.py:83 | 80 | Title over this fails validation | D | No source. | DONE 2026-09-27: removed; the body token cap bounds a skill (skill_builder has no title check). |
| MAX_PROHIBITIONS | mcp/skill_limits.py:85 | 2 | Skill with >2 prohibiting items FAILS validation | A-flag (number ours) | Operator 2026-09-26 12:39 "responds to do not / never ... should be used more sparingly" (verbatim in OM). "2" from AGENTS.md "at most two" (prompt eval 10.7/10.0/9.3, flagged "at risk", max_tokens 400 bug). | keep; operator to confirm the value/wording |
| PROHIBITION regex | mcp/skill_limits.py:139 | never\|do not\|don't\|must not... | Decides which items count as prohibitions | B | Definition of the operator's "do not / never" category (2026-09-26 12:39); lexical, not tunable. | keep |
| NAME_CHARS | mcp/skill_limits.py:89 | 64 | SKILL.md name length cap | B | Agent Skills spec: name <=64 chars (DECISION-TREES-AND-SKILLS.md:362, S23); Hermes MAX_NAME_LENGTH. | keep |
| DESCRIPTION_CHARS | mcp/skill_limits.py:90 | 1024 | SKILL.md description cap | B | Agent Skills spec: description <=1,024 chars (DECISION-TREES-AND-SKILLS.md:362). | keep |
| DESCRIPTION_AIM | mcp/skill_limits.py:91 | 300 | Compiled descriptions truncated to this; prompts aim at it | D | "one or two sentences" -- no source. | DONE 2026-09-27: removed; compile bounds a description at the spec's DESCRIPTION_CHARS (1,024), the prompts read that. |
| MAX_TAGS / MAX_TOPICS | mcp/skill_limits.py:92-93 | 12 / 12 | Caps tags and topics kept per rule | D | No source. Topics gate selection, so the cap changes matching. | DONE 2026-09-27: removed; tags_of, gates and extract_topics keep every topic. |
| MAX_SKILLS_PER_TURN | mcp/skill_limits.py:98 | 8 (env YAMADORI_SKILL_CEILING) | Ceiling on bodies injected per decision | A-flag (number ours) | Operator 2026-09-27 08:44 "You said only 3 skills per turn, I just asked for 4 and maybe 5 apply"; 08:46 "I don't think a token budget on skill selection is useful". 8 is ours ("a CHOICE"). | keep; operator to confirm the value/wording |
| TURN_TOKENS_AIM / TURN_TOKENS_HARD | mcp/skill_limits.py:102-103 | 900 / 1500 | Reported only, on the dashboard meter | D | Operator 2026-09-27 08:46 rejected a per-selection token budget. Not enforced (skill_limits.py:99). | DONE 2026-09-27: removed; the meter is gone from web/src (Skills.tsx, api/skills.ts, Skills.test.ts; tsc and vitest pass). web/dist is NOT rebuilt. |
| STEP_MAX_BODIES | mcp/skill_limits.py:107 | 2 | Max new bodies on an agent step | D | "the evidence of one step names one or two areas (a CHOICE)". | DONE 2026-09-27: removed; one recall per area per turn, the sanity ceiling (max_skills_per_turn) applies. |
| FADE_TOKENS | mcp/skill_limits.py:114 | 8000 tokens | Given skill becomes recall-eligible after this much conversation | D | Liu et al. 2023 span "rounded up"; skill_limits.py:112 "A CHOICE, unmeasured on this model". Literature inspiration, not a constraint. | DONE 2026-09-27: removed with the faded state and GROW (skill_chart); a given skill is recalled only on its own ASKED / ERROR / PHASE event. daily_eval seq-koota-fade's last turn now expects nothing. |
| RECALL_EVERY_STEPS | mcp/skill_limits.py:117 | 5 requests | Cooldown between recall lines per area | D | "(a CHOICE)". No source. | DONE 2026-09-27: removed with the chart's cooling guard; never the identical recall line twice in a row stays (test_skill_turns, test_skill_questions). |
| RECALL_ITEMS | mcp/skill_limits.py:120 | 2 | Items per recall line (one DO + one DO NOT) | A-flag | Operator 2026-09-27 08:52 "remember we don't do x, we do y ... it works" -- the do/not pair shape. Number follows the phrase. | keep; operator to confirm the value/wording |
| INDEX_MAX / INDEX_LINE_CHARS | mcp/skill_limits.py:123-124 | 12 / 140 | Craft index size in system text | A-flag (numbers ours) | Operator 2026-09-27 08:51 "we can lead with you have these skills available if you need more with a sound way for the model to ask". 12/140 "CHOICES". | keep; operator to confirm the value/wording |
| MAX_TRIGGERS / TRIGGER_CHARS | mcp/skill_limits.py:128-129 | 8 / 200 | Trigger lines kept per skill (embedding targets) | D | No source. | DONE 2026-09-27: removed; triggers_of keeps the whole description as one trigger plus every when-to-use bullet (the 1,024 bound bounds them). |
| MAX_LEARNED_TRIGGERS | mcp/skill_limits.py:130 | 40 | Cap on fallback-learned triggers per skill | B-value-arbitrary | A bound is needed (learned triggers grow unbounded); 40 unjustified. | keep (the bound is forced); derive the value |
| MIN_SHOULD / MIN_SHOULD_NOT | mcp/skill_limits.py:133-134 | 2 / 2 | Min activation cases per skill | A-flag (numbers ours) | Operator 2026-09-26 12:36 "process that shapes tests and distills the skill for activation and selection". Counts ours. | keep; operator to confirm the value/wording |
| MAX_ACTIVATION_TESTS / MAX_BEHAVIOUR_CHECKS | mcp/skill_limits.py:135-136 | 12 / 4 | Caps stored tests | D | No source; storage bound. | DONE 2026-09-27: removed with skill_tests' case caps. |
| CHARS_PER_TOKEN | mcp/skill_limits.py:72 | 3.0 | Token estimate for every skill cap | B-value-arbitrary | Deliberately high estimate (tiers.estimate_prompt_tokens' rate). Exact counts available via /tokenize; rate not measured on this tokenizer here. | keep (the bound is forced); derive the value |
| EMB_HIGH | mcp/skill_select.py:127 | 0.60 | Cosine above which a word match injects / no-evidence asks | D | skill_select.py:57 "thresholds are UNMEASURED starting points"; :70 "placeholders". Absolute cosine threshold across queries. | DONE 2026-09-27: removed; a cosine only ranks inside a question. A round the patterns leave silent admits nothing on a cosine (embedding-only rows are no candidates), so the CATEGORY question went too. dash_skills reports thresholds {}. |
| EMB_LOW | mcp/skill_select.py:128 | 0.45 | Cosine below which fact/phrase rows only ask | D | Same: "placeholders". | DONE 2026-09-27: removed with EMB_HIGH. |
| verdict decision table | mcp/skill_select.py:365 | fact/phrase/word x cosine | Turns strength+cosine into inject/ask/none | A-flag (thresholds D) | Operator 2026-09-24 12:58: "use a cheaper faster classification pipeline and worst case fall back to a full agent". Cascade shape is operator's; table cells ours. | keep; operator to confirm the value/wording |
| MAX_ASK | mcp/skill_select.py:129 | 4 | Max unconfirmed candidates sent to E1/Laya/fallback | D | No source. | DONE 2026-09-27: removed; a question is bounded by its own options. |
| TRIGGER_INSTRUCT | mcp/skill_select.py:140 | instruction string | Query-side retrieval instruction for trigger embedding | C-partial | STACK-TUNING-2026-09-23.md:199-205: instruction mismatch shifted hints cosines; recipe-shaped instruction raised median 0.549->0.610 (hints, n=120). Not re-measured on skill triggers. | keep |
| FALLBACK_TIMEOUT | mcp/skill_select.py:143 | 90 s | Timeout of the fallback model call | B-value-arbitrary | A timeout is required on the request path; 90 unjustified. | keep (the bound is forced); derive the value |
| LAYA_TIMEOUT | mcp/skill_select.py:145 | 10 s | Laya call timeout (off by default) | B-value-arbitrary | Liveness bound; value arbitrary. | keep (the bound is forced); derive the value |
| REQUEST_BUILD_MAX | mcp/skill_select.py:241 | 128 texts | Past it trigger vectors build in background, request runs without embedding | D | "A CHOICE". Effect: first request after restart silently skips stage 2. | DONE 2026-09-27: removed; the request path never builds trigger vectors (trigger_index(request=True)); they are built at arm / refresh_triggers. |
| embed batch size | mcp/skill_select.py:298 | 64 | Trigger texts per embedding call | B-value-arbitrary | Embedder batch/-ub limit; value not derived. | keep (the bound is forced); derive the value |
| zero-vector norm check | mcp/skill_select.py:301,345 | norm < 0.5 | Rejects zero embeddings | B | PROTOCOL rule 1 (zero vectors once passed for an index); correctness. Normalised vectors have norm 1. | keep |
| QCACHE_MAX / STICKY_MAX / _EQ_CACHE_MAX | mcp/skill_select.py:325,828; skill_classify.py:1136 | 64 / 512 / 64 | In-memory cache bounds | B-value-arbitrary | Memory bound needed; sizes arbitrary; no behaviour effect beyond recompute. | keep (the bound is forced); derive the value |
| fallback query/field cuts | mcp/skill_select.py:338,791-805 | 2000/1500/80/100/240 chars, 3 triggers | Truncation of text sent to embedder and fallback model | B-value-arbitrary | Embedder -c 8192 and model window bound them; chosen values arbitrary. | keep (the bound is forced); derive the value |
| fallback sampling | mcp/skill_select.py:786 | effort minimal, max_tokens 512, temperature 0.0 | The fallback model call's settings | D | No source; temperature 0.0 departs from operator 2026-09-25 "keep temp at 1.0" (memory temperature-decision; scope: main). | DONE 2026-09-27: removed; the fallback call sends no temperature or max_tokens (tier defaults, vendor sampling). |
| FALLBACK_SYSTEM decision table | mcp/skill_select.py:152 | prompt text | Rules the fallback model applies to pick skills | D | Prompt wording, unmeasured (skill_prompts.py:18 "EVERY TEMPLATE IS A CHOICE"). Data-not-instructions framing: C-partial (INJECTION.md F1 1/310 vs 13/350, p=0.0021, on answering). | OPERATOR 2026-09-27: kept as written, DATA framing included; its wording needs a measurement or the operator's sign-off. |
| HOSTED | mcp/skill_select.py:381 | react->r3f,koota; ts/js->web fws; python->pytest; rust->wasm_bindgen,… | Host areas rank after areas they host; relevance floor | D | "the V4 stack replay, 2026-09-26: React 19 upgrade notes took a slot from TSL" (skill_select.py:1584) -- in-sample, n=1. | DONE 2026-09-27: the extras' host ordering is removed; the HOSTED table still names host areas for the relevance floor below, which is kept. |
| confidence() | mcp/skill_select.py:397 | score + 2*cosine | Ordering score for candidates | D | No source for weight 2. | DONE 2026-09-27: no weights: a lexicographic rank -- strength ordinal (fact > phrase > word), then strong-topic count, then cosine. |
| _REL_STOP stopword list | mcp/skill_select.py:411 | ~130 words | Words ignored by subject relevance | D | Hand list, no source. | KEPT 2026-09-27: IDF alone moves the koota asked pick traits -> queries (daily_eval seq-koota-body-recall#1, 2 checks). |
| stems() rules | mcp/skill_select.py:433 | len<3 dropped; plural/ies stripping | Tokenisation for relevance | D | Hand rules. | KEPT 2026-09-27: no standard stemmer is installed offline, and the embedder's tokenizer needs the stack. |
| REL_WORD_MIN / REL_FLOOR | mcp/skill_select.py:429-430 | 3.5 / 7.0 (IDF) | Host-area pick needs 2 rare shared words summing to 7 | D | "a CHOICE"; tuned on dark-mode and shadow-acne replays (docstring :491-499), in-sample. | KEPT 2026-09-27: with no floor every host pick is eligible and daily_eval three-shadow-acne's MUST-NOT fails (a TSL craft on 'shadow'); the structural alternative (a host pick needs a candidate row) fails rust-repr-c and emscripten-embind MUSTs. The read_craft topic threshold is DONE (read_craft reads by name only). |
| subject_topics rule | mcp/skill_select.py:616 | topic must appear in skill's trigger text | Only topics a skill is ABOUT count on steps/host areas | A-flag | Code claims operator 2026-09-27 "a skill must be ABOUT the request's intent, not share a word" -- claimed, no quote found. Nearest: 08:40 "ensure the engine isn't just injecting 'use this shit' blindly". | keep; operator to confirm the value/wording |
| IMPLIES r3f->react refs skill | mcp/skill_select.py:527 | react-dev-...-refs-vs-state-2 | Asking r3f injects a named React refs skill | D | Code claims operator 2026-09-27 "r3f v10 -> react, threejs, tsl; koota+r3f -> ..." -- claimed, no quote found. Skill choice ours, justified by readme reading. | KEPT 2026-09-27: for the operator: deleting it fails 5 daily_eval MUST area:react checks (the operator's r3f prompts). |
| IMPLIES r3f v10+ -> r3f-tsl-hooks-18 | mcp/skill_select.py:534 | min_version 10 | Asking r3f v10 injects TSL hooks skill | A-flag | Operator 2026-09-27 08:44 "(TSL is implicit)" verbatim in OM. Which skill is ours. | keep; operator to confirm the value/wording |
| IMPLIES r3f v10+ -> threejs-llms-full-tsl-e-g | mcp/skill_select.py:540 | min_version 10 | Asking r3f v10 injects three.js TSL skill | A-flag | Same "TSL is implicit" (08:44). | keep; operator to confirm the value/wording |
| IMPLIES webgpu+threejs -> TSL skill | mcp/skill_select.py:544 | row | Asking webgpu+threejs injects TSL skill | D | No operator quote; readme reasoning. | DONE 2026-09-27: row deleted. |
| IMPLIES koota+react / koota+r3f | mcp/skill_select.py:548,551 | 2 rows | Named koota integration skills injected | D | Operator asked for a Koota skill (02:31 "we want a skill for Koota"), not an implication. Claimed quote not found. | DONE 2026-09-27: rows deleted; asked koota gets its area pick. |
| IMPLIES pmndrs_math+threejs/r3f | mcp/skill_select.py:556,560 | 2 rows | math-with-three-js injected | D | No quote. | DONE 2026-09-27: rows deleted. |
| IMPLIES typegpu+threejs | mcp/skill_select.py:563 | row | typegpu three integration injected | D | No quote. | DONE 2026-09-27: row deleted. |
| _BUILD_ASK + BUILD_HINT | mcp/skill_select.py:574-577 | regex window {0,40}; {setup,setting,install,configure} | "Build it with X" boosts X's setup skill | D | Comment: "picked R3F's version-pinning skill over its v10 setup skill on the Octopus spec's words" -- steers one benchmark prompt. | KEPT 2026-09-27: removing it loses r3f-v10-setup-21 on op-sentence-pagoda, op-pagoda-r3f-stack-exact, stack-particles and seq-unrelated-after-stack#0 (4 MUST checks; r3f-tsl-hooks-18 takes the asked r3f slot). For the operator. |
| older-major penalty | mcp/skill_select.py:715 | v<N words subtract | Skill naming older major ranks lower | D | In-sample (v9 beside v10 on Octopus/pagoda prompts). | DONE 2026-09-27: replaced by a structural filter: a skill whose NAME carries vN other than the asked version is skipped. |
| generality ordering | mcp/skill_select.py:721 | -count(gates) | Fewer gates ranks higher among area skills | D | No source. | KEPT 2026-09-27: removing it moves the koota asked pick traits -> queries (seq-koota-body-recall#1, seq-koota-fade#1: 4 checks). |
| laya_pick state cut / two orders | mcp/skill_select.py:751 | state[:1500]; 2 orders | Laya input cut and order averaging | B | Laya drops tail past ~512 tokens (AGENTS.md Laya; tail_probe.py); order-averaging removes position bias. 1500 value approx. | keep |
| option_text cut | mcp/skill_select.py:1083 | 600 chars | Option text shown to deciders | B-value-arbitrary | Bounded by description 1,024; 600 arbitrary. | keep (the bound is forced); derive the value |
| QUESTION_TEXT | mcp/skill_select.py:1086 | 4 templates | Question wording for deciders | D | Wording unmeasured. | OPERATOR 2026-09-27: kept; measure with the CLM fidelity set, or the operator approves. |
| _strong() | mcp/skill_select.py:1141 | fact/phrase or model-confirmed; host prose needs relevance | Which evidence questions may inject | D | "the old confidence slot's rule, unchanged"; in-sample. | DONE 2026-09-27: removed; the tier is the verdict (a decider's pick vs NONE). |
| slots: asked | mcp/skill_select.py:1400 | one question per asked area | Every explicitly asked area gets its best skill | A | Operator 2026-09-27 08:40 "it's more about I asked for it, reinforce it" (verbatim OM). Claimed longer quote "one slot per EXPLICITLY asked area..." -- not found. | keep |
| slots: confidence extras | mcp/skill_select.py:1472 | one per uncovered area | Strongly supported extra skills added | A-flag | Operator direction "a really smart skill selector ... knows a skill would really reinforce here at this turn" (08:51). One-per-area rule ours. | keep; operator to confirm the value/wording |
| index candidates | mcp/skill_select.py:1628-1630 | top 3 per asked area, 4 ask rows | What the craft index lists | D | No source. | DONE 2026-09-27: only INDEX_MAX bounds the index. |
| _PHASE_ORDER | mcp/skill_select.py:1685 | debug>verify>refactor>review>implement>plan | Main phase when several match | D | No source. | DONE 2026-09-27: removed; skill_select.phases_of keeps every phase, and a change is a new phase entered. |
| state lock clear | mcp/skill_select.py:1698 | 4096 conversations | Lock table bound | B-value-arbitrary | Memory bound. | keep (the bound is forced); derive the value |
| _item_score weights | mcp/skill_select.py:1809-1812 | 2.0 per API name; 0.25*IDF words | Which item a recall line states | D | No source. | DONE 2026-09-27: removed; a recall line is the skill's first DO/WHEN item plus its first DO NOT. |
| recall candidates | mcp/skill_select.py:1842 | pos[:3] | Tries 3 best items to avoid repeating last line | D | No source. | DONE 2026-09-27: limit removed with _item_score. |
| step evidence cuts | mcp/skill_select.py:1878,2025-2035 | [-12000:] text, [-6000:] args | Evidence window for steps/recall | B-value-arbitrary | Bounded window needed; values arbitrary. | keep (the bound is forced); derive the value |
| step conf bonus | mcp/skill_select.py:1895 | +0.5 on error, +len(subject) | Ranking of step rows | D | No source. | DONE 2026-09-27: removed. |
| compaction detection | mcp/skill_select.py:1995 | chars < 0.7 * previous | Treats a 30% shrink as a compaction; forgets given skills | D | No source. | DONE 2026-09-27: the 0.7 shrink ratio is gone; the proxy passes its own compaction count (progress 'compactions', proxy._compactions on the skills paths) and a moved count resets what was given. |
| read_craft matching | mcp/skill_select.py:2262-2302 | q[:200]; name hit weight 2.0; >= REL_WORD_MIN; near 5; also 3 | recall_craft tool topic lookup | D | No source. | DONE 2026-09-27: name only (craft/4); an unknown name returns NO_SUCH_CRAFT with every name sharing a word as `near`, no threshold. |
| craft offer only if index non-empty | mcp/skill_select.py:2233 | rule | No tool/index when nothing matched | D | "a CHOICE: Shi et al. 2023" -- literature, unmeasured. | OPERATOR 2026-09-27: kept as is. |
| docstring stale budget | mcp/skill_select.py:50-52 | "aim 1,500, hard 2,500" | Stale text: says a per-turn budget selects | D | Contradicts code (no budget since 2026-09-27). | DONE 2026-09-27: stale lines deleted. |
| MAX_OPTIONS | mcp/skill_deciders.py:52 | 6 incl. NONE | Options per decider question | A-flag (number ours) | Operator 2026-09-27 13:05 "we can have a handful of options"; 2026-09-24 (Laya article) "if you give it too many options it quickly degrades". "~6" in comment: claimed, no quote found. | keep; operator to confirm the value/wording |
| EvidenceStub NONE logit | mcp/skill_deciders.py:134 | min(eligible)-1 | Stub picks best eligible else NONE | B | Deterministic encoding of the stub's own rule (argmax when anything eligible); no tunable effect. | keep |
| LexicalStub SCALE / IDF smoothing | mcp/skill_deciders.py:164,180 | 4.0; +0.5 | Lexical decider's logit scale | D | No source; offline stand-in ("it is not the gate"). | DONE 2026-09-27: the lexical decider is test-only: skill_deciders.SERVING = (stub, clm); YAMADORI_SKILL_DECIDER cannot select it. |
| _STOP (deciders) | mcp/skill_deciders.py:139 | stopword list | Lexical decider stopwords | D | Hand list. | DONE 2026-09-27: test-only with the lexical decider. |
| CLM_TIMEOUT | mcp/skill_deciders.py:54 | 5 s | CLM decider HTTP timeout | B-value-arbitrary | Request-path liveness; value arbitrary. | keep (the bound is forced); derive the value |
| CLM_STATE_CHARS | mcp/skill_deciders.py:55 | 6000 chars | Left-cut of state sent to CLM | B-value-arbitrary | CLM window 2,048 tokens (CLM-EVAL.md:105, vLLM --max-model-len 2048). 6000 chars approximates it; clm.py:100 cuts exactly by tokens, so this char cut is redundant. | keep (the bound is forced); derive the value |
| pick() rule | mcp/skill_deciders.py:98 | argmax when > NONE | No absolute threshold on probabilities | B | CLM card "relative to provided candidate sets" (CLM-EVAL.md §3); Laya rule AGENTS.md "Never threshold a Laya score across queries". | keep |
| statechart structure | mcp/skill_chart.py:82-120 | 4 top states, 4 area states, guards | Which skill questions are legal per turn | A-flag | Operator 2026-09-27 13:03 "skill selection might be XState or something like it with this model" (OM). States/guards ours. | keep; operator to confirm the value/wording |
| PHASE_TOP mapping | mcp/skill_chart.py:84 | implement,refactor->building; verify,review->verifying | Collapses six phases to four top states | D | No source. | DONE 2026-09-27: removed; the chart's top states are the fine phases (implement, debug, ...). |
| chart transition log | mcp/skill_chart.py:360 | last 12 | Record size | B-value-arbitrary | Record bound. | keep (the bound is forced); derive the value |
| MIN_HITS / SHARE | mcp/skill_classify.py:71-72 | 2 / 0.25 | A source term needs 2 hits and 25% of top count | D | No source. | KEPT 2026-09-27: with declared terms only, an undeclared source is filed by nothing offline: test_skills' fixture skills quarantine ('names no language'), test_pmndrs_stack's pinned sources fail. |
| MAX_TERMS | mcp/skill_classify.py:73 | 2 per kind | Caps languages/frameworks per rule | B-value-arbitrary | Security intent: "a rule cannot be stuffed into matching everything" (:52). Bound justified; 2 arbitrary. | keep (the bound is forced); derive the value |
| STRENGTH | mcp/skill_classify.py:74 | fact 3, phrase 2, word 1 | Ordinal of evidence strength | B | Ordinal only (max()); values carry no magnitude. | keep |
| VOCAB table | mcp/skill_classify.py:90-183 | 26 terms with regexes | Languages/frameworks recognised in requests and sources | A-flag | Stack scope (AGENTS.md: TS, Rust/WASM, C ABIs, three.js TSL/TypeGPU); koota and pmndrs math: operator 2026-09-27 02:31 "we want a skill for Koota ... npm math". Regex spellings ours. | keep; operator to confirm the value/wording |
| ARTIFACTS noun regexes | mcp/skill_classify.py:196-250 | 8 artifacts | Artifact detection by nouns/paths | A-flag | Operator 2026-09-24 12:58 "there are artifacts and those artifacts have categories" (OM). Noun lists ours; ui_design "layout" exclusion from migration dry run. | keep; operator to confirm the value/wording |
| PHASES taxonomy | mcp/skill_classify.py:274 | plan..review (6) | Phase axis of the taxonomy | D | Operator asked "well tagged categorized" (09-26); phase axis ours. | OPERATOR 2026-09-27: kept; the operator confirms the axis. |
| _PHASE_RX keyword tables | mcp/skill_classify.py:278-309 | 5 regexes | Decides request phase (gates skills) | D | Hand keyword lists; debug list tuned on "replay of the Octopus V0 spec, 2026-09-26". | KEPT 2026-09-27: phase from structure only fails 5 daily_eval checks (seq-koota-body-recall#1/#2, seq-koota-fade#1) and the phase-gated crafts' own activation tests (module-exports-match-callers, koota-queries-and-systems, koota-traits-and-entities: quarantine on re-arm). |
| SITUATIONS set | mcp/skill_classify.py:312 | 5 situations | Conversation facts a skill can gate on | D | Introduced for authored skills from Octopus failures (browser-app-entry-point etc.). | OPERATOR 2026-09-27: kept, because the authored skills stay; the operator confirms. |
| _ERROR_OUT | mcp/skill_classify.py:322 | regex | Detects error output (situation, debug phase) | B-value-arbitrary | Patterns are real tool output formats (tsc TS2339, rustc E0425, ESLint, Vite). Using it to set debug phase is ours. test_skills.py. | keep (the bound is forced); derive the value |
| _CALL_MISMATCH | mcp/skill_classify.py:333 | regex | Detects call/export mismatch errors | D | Built for module-exports-match-callers authored skill; SKILL-FACTORY.md:352 "a broader regex than the skill's subject". | OPERATOR 2026-09-27: kept with the situation. |
| _FILE_LINE / _SOURCE_PATH / multi_file>=2 | mcp/skill_classify.py:339,344,1484 | regexes; >=2 files | file_line and multi_file situations | B | Definitions (path:line shape; two or more files is "multi"). | keep |
| _CODE_SHAPED / code_shaped | mcp/skill_classify.py:351,415 | regex; len>=3 | Code-shaped topic is a fact | D | Design choice; "does not appear in prose by accident" (asserted). | KEPT 2026-09-27: topics as confirm-only fails 9 checks (bug-koota-throw, seq-koota-body-recall#1/#2, seq-koota-fade#1, seq-phase-debug#3: the koota-queries craft on updateEach needs the code-shaped topic as a fact). |
| _PROSE_ABBREV / _PACKAGE_NAME / _FILE_NAME exclusions | mcp/skill_classify.py:358-366 | lists | Such topics only confirm, never facts | D | "replay of the V4 stack, 2026-09-26" in-sample. | KEPT 2026-09-27: with code_shaped. |
| COMMON_API | mcp/skill_classify.py:379 | ~70 identifiers | Platform APIs are plain words outside specific frameworks | D | "A CHOICE"; from one dark-mode replay (SKILL-FACTORY.md:165), in-sample. | KEPT 2026-09-27: removing it makes 'localStorage' alone a FACT for an auth-token craft (test_skill_turns [precision], 2 checks) and moves test_pmndrs_stack's React pick. |
| HOST_FRAMEWORKS | mcp/skill_classify.py:412 | {react} | React treated like a language for evidence | D | "half the store names them" -- observation, no threshold source. | KEPT 2026-09-27: removing it makes 'React' in prose an asked FACT, which moves every React mention off the ask path (10 test_skills cascade checks). |
| VERBS + phrase window | mcp/skill_classify.py:443,452 | verb list; {0,5} words | Artifact verb+noun = phrase strength | D | No source. | KEPT 2026-09-27: nouns-only makes 'make me a slide deck' a WORD, which the language round does not open: no slides/docs craft is selectable (test_skills slides check). |
| _KEYWORDS | mcp/skill_classify.py:619 | ~150 words | Excluded from topics and evidence words | D | Hand list. | KEPT 2026-09-27: an IDF cut over the store needs a threshold, and none is sourced. |
| extract_topics limits | mcp/skill_classify.py:631-663 | limit 8; len 3..48; span 2..60 | Topic extraction | D | No source. | DONE 2026-09-27: removed (limit=None, no length bounds). |
| triggers_of rules | mcp/skill_classify.py:711,727,729 | sentence>=12; block 2000; bullet>=8 | Trigger line extraction | D | No source. | DONE 2026-09-27: the whole description is one trigger, plus every when-to-use bullet; _sentences removed. |
| non-code artifact prominence | mcp/skill_classify.py:825 | artifact count >= top term count | Drops weak artifact tags on code sources | D | "migration dry run classified 20 of 67 code groups as UI design" -- observation, in-sample. | KEPT 2026-09-27: with MIN_HITS / SHARE. |
| escalate frontmatter | mcp/skill_classify.py:860 | escalate: true | Skill can trigger deep thinking | A-flag | AGENTS.md Phase 0.6 triggers "(operator 2026-09-24)"; skill escalate specifics ours. | keep; operator to confirm the value/wording |
| _tool_call_paths cuts | mcp/skill_classify.py:1024,1036 | <400 chars; last 50 | Which tool args count as paths | B-value-arbitrary | Bound needed; values arbitrary. | keep (the bound is forced); derive the value |
| WORKED_MESSAGES / WORKED_CHARS | mcp/skill_classify.py:1062-1063 | 8 / 40000 | Recent window read for code evidence | D | "(a CHOICE)". | DONE 2026-09-27: replaced by a structural boundary, skill_classify.window_start: since the last user turn (the previous one when the request ends on a user turn). |
| EMBED_QUERY_CHARS | mcp/skill_classify.py:1126 | 2000 | Embedding query cap | B-value-arbitrary | Embedder -c 8192 (CLM-EVAL.md:160); 2000 "the old cut". | keep (the bound is forced); derive the value |
| EMBED_SUBSTANTIVE_CHARS / DIGEST_SHORT / DIGEST_LONG | mcp/skill_classify.py:1127-1129 | 400 / 800 / 300 | Digest size beside user text | D | skill_classify.py:1125 "Every number is a CHOICE"; one replay (0.34->0.55). test_skills.py. | DONE 2026-09-27: a fixed digest: every line; the whole query is cut at EMBED_QUERY_CHARS, the user text first. |
| DIGEST_ERRORS/IMPORTS/NAMES/FILES | mcp/skill_classify.py:1130-1133 | 3/8/10/4 | Digest content caps | D | "CHOICE". | DONE 2026-09-27: gone with the digest counts. |
| _PLAIN_WORDS | mcp/skill_classify.py:1143 | ~90 words | Words that do not tie a user turn to evidence | D | Hand list. | KEPT 2026-09-27: it feeds _REL_STOP, which is kept. |
| _evidence window | mcp/skill_classify.py:1166-1178 | 2 tool results, 2 writes, 8000 chars each | Newest evidence definition | D | No source. | DONE 2026-09-27: window_start (above). |
| _about_evidence join rule | mcp/skill_classify.py:1209 | step/debug/shared word | Whether digest joins the embedding query | D | One replay ("make the background dark blue", 0.34->0.55), in-sample. | DONE 2026-09-27: always join. |
| _word_stems 4+ letters | mcp/skill_classify.py:1200 | len>=4 | Shared-word test | D | No source. | DONE 2026-09-27: gone with the join rule. |
| request_signals windows | mcp/skill_classify.py:1325,1419,1424,1447,1483,1376 | last[:8000]; last 8 msgs; 20000; last 4 tool msgs 8000; blob[:40000];… | Where signals are read | B-value-arbitrary | Bounded scan needed; values arbitrary. | keep (the bound is forced); derive the value |
| how list cap | mcp/skill_classify.py:1333 | 3 | Evidence reasons kept per term | B-value-arbitrary | Record bound. | keep (the bound is forced); derive the value |
| asked specific framework = fact | mcp/skill_classify.py:1394 | specific fw fact; language/React word | Explicit ask strength | A-flag | Operator 08:40 "I asked for it, reinforce it". Language/React demotion ours (in-sample dark-mode case). | keep; operator to confirm the value/wording |
| implement default phase | mcp/skill_classify.py:1455 | code & no debug/verify/refactor/review -> implement | Default phase | D | "replay of the V4 stack 2026-09-26: the Octopus spec read as plan only". | KEPT 2026-09-27: with _PHASE_RX. |
| _NEGATION + window | mcp/skill_classify.py:1597,1629 | word list; 60/24 chars | "without React" asks for nothing | D | No source; ours. | DONE 2026-09-27: the pattern is KEPT (removing it fails stack-without's MUST-NOT); the 24/60-character windows are replaced by the clause (_clause_before). |
| _CONTEXT_MENTION | mcp/skill_classify.py:1603,1636 | my/our/this...; 24 chars | "my React page" is context not choice | D | In-sample daily eval row. | DONE 2026-09-27: the pattern is KEPT (removing it fails three-shadow-acne's MUST-NOT); the 24-character window is replaced by the clause. |
| alias version window | mcp/skill_classify.py:1654 | (..{1,40}) within 60 | "r3f (react-three-fiber) v10" version | D | Fitted to one operator sentence, 2026-09-27 08:43 "r3f (react-three-fiber) v10" -- one prompt. | DONE 2026-09-27: the 60/40-character windows are removed; the alias-version structure is KEPT (removing it fails 12 checks: v10 on the operator's sentence). |
| plain topics need two | mcp/skill_classify.py:1719 | min(2, len(plain)) | One plain topic word not evidence | D | "replay of 2026-09-26: a single plain word put an algorithms skill on a game spec" (in-sample). | KEPT 2026-09-27: removing it fails react-dev-blog-react-19-2-conditional-rendering's own activation near miss (quarantine on re-arm). |
| tools-keyed base score | mcp/skill_classify.py:1732 | 1.0 + 0.5*n | Score of a harness-tool-keyed rule | D | No source. | DONE 2026-09-27: the score is the strength ordinal. |
| other-language gate | mcp/skill_classify.py:1746 | rule | Code topic alone fails if request names other languages | D | In-sample (TS typing skill on vanilla-JS game). | KEPT 2026-09-27: removing it makes the pmndrs math and koota crafts fail their own activation near misses (test_pmndrs_stack: 7 of 10 quarantined on re-arm). |
| match score weights | mcp/skill_classify.py:1766-1774 | 1.0/0.25 topics; +0.5 phases/situations/all_of; +0.5 fresh | Candidate ordering | D | "CHOICES; replay of the V4 stack". | DONE 2026-09-27: the score is the strength ordinal (fact 3 / phrase 2 / word 1); strong_topics is reported beside it. |
| _primary base weights | mcp/skill_classify.py:1790-1806 | any 1.5, artifact 1.5, fw 2.0, lang 1.0, code 0.75, domain 0.5, secon… | Primary-key match score | D | "(the pre-taxonomy rule, unchanged)" -- no source. | DONE 2026-09-27: strength ordinal only. |
| skill screen existence + quarantine | mcp/skill_screen.py:7 | quarantine on failed screen | Every skill screened before arming | A | Operator 2026-09-24 22:46 "Anything we fetch must be scanned and stripped of malicious intent ... should not encode skills that ask our model to exfiltrate data or run untrusted scripts" (OM). Quarantine action ours. | keep |
| TOOL_NAMES | mcp/skill_screen.py:102 | our tool names | Source naming our tools is flagged | B | Security; checked against proxy.OUR_NAMES by test_skills.py. | keep |
| _INVISIBLE chars | mcp/skill_screen.py:119 | zero-width, bidi, filler | ASCII smuggling guard | B | Security (named attack: invisible/bidi smuggling). | keep |
| _AI_DIRECTED patterns | mcp/skill_screen.py:344 | 10 regexes | Prompt-injection shapes quarantine | B-value-arbitrary | Security; test_deep_review.py 17/17 malicious fixtures caught, 0/5 clean touched (bench/skills/fixtures verified: 17 + 5 files). Pattern set hand-written. | keep (the bound is forced); derive the value |
| exfiltration / shell / credentials / remote_load patterns | mcp/skill_screen.py:421-526,649,696 | regex tables | Named-attack shapes quarantine | B-value-arbitrary | Same security basis and fixtures; "lean strict" by design (skill_screen.py:65). | keep (the bound is forced); derive the value |
| _B64 run length | mcp/skill_screen.py:732 | 200 chars | Base64 run outside code quarantines | B-value-arbitrary | Encoded payload attack; 200 arbitrary. | keep (the bound is forced); derive the value |
| screen_item drops (url/command/tool) | mcp/skill_screen.py:806-818 | rules | Distilled items carrying URLs/commands dropped | B | Security; operator 22:46 "ensure we don't accident leak commands". | keep |
| _UNRELATED patterns | mcp/skill_screen.py:838 | 5 regexes | Drop items telling model to act outside the work | A | Operator 2026-09-24 22:46 "not to instruct it to do unrelated actions" (verbatim). Patterns "CHOICES"; AGENTS.md: 0 of 2,575 recipe rows flagged. | keep |
| STRIP_MAX_FRACTION | mcp/skill_screen.py:1053 | 0.25 | Fetched text >25% stripped is dropped whole | D | "The threshold is a CHOICE" (:1049). Operator asked to strip, not for a drop cutoff. AGENTS.md ~L493. | DONE 2026-09-27: removed; the screen strips, and drops a text only structurally: a finding with no line to cut, what is left still failing, or nothing but headings left (test_deep's all-injection page; test_deep_review's knowledge-base check updated). |
| WEB_UNVERIFIED labelling | mcp/skill_screen.py:1055 | label | Web-sourced commands cross only as labelled facts | A-flag | Operator 22:46 "ensure we don't accident leak commands"; label mechanism ours. | keep; operator to confirm the value/wording |
| handoff instruction sections | mcp/skill_screen.py:1182 | NEXT STEP(S), ORDER, STEPS | Where web commands are removed | B | Names of the hand-off's own fixed sections (shomen format). | keep |
| SCREEN_SYSTEM prompt | mcp/skill_screen.py:944 | prompt | Model-assisted screen wording | D | Unmeasured template. | OPERATOR 2026-09-27: kept; measure on bench/skills/fixtures, or the operator approves. |
| UA / robots obey | mcp/skill_pipeline.py:76,129-160 | robots.txt, 401/403 rules | Skill fetches obey robots | B | RFC 9309 conventions (skill_pipeline.py:43-45); politeness/legal. | keep |
| FETCH_TIMEOUT / robots timeout | mcp/skill_pipeline.py:78,144 | 30 s / 15 s | Fetch timeouts | B-value-arbitrary | Liveness; values arbitrary. | keep (the bound is forced); derive the value |
| FETCH_MAX_BYTES / ROBOTS_MAX_BYTES | mcp/skill_pipeline.py:79,81 | 1 MiB / 512 KiB | Fetch size caps | B-value-arbitrary | Security/resource bound; values arbitrary. test_deep_review.py, test_skills.py. | keep (the bound is forced); derive the value |
| ROBOTS_TTL | mcp/skill_pipeline.py:82 | 3600 s | robots cache lifetime | B-value-arbitrary | Cache bound. | keep (the bound is forced); derive the value |
| REFUSED_EXT / ALLOWED_TYPES | mcp/skill_pipeline.py:104-112 | lists | Never fetch scripts/binaries | B | Security: "Scripts are data ... never fetched" (SKILL-FACTORY.md:427); operator 22:46 no untrusted scripts. | keep |
| binary sniff | mcp/skill_pipeline.py:231 | NUL in first 8192 bytes | Refuse binary content | B | Correctness. | keep |
| CHUNK_CHARS | mcp/skill_pipeline.py:85 | 12000 | Source chars per distil call | B-value-arbitrary | worker.py:91 "~3k tokens of source leaves the thinking model room" (empty-reply bug fdc9067). Existence forced by budget; value not derived. | keep (the bound is forced); derive the value |
| MAX_CHUNKS / MAX_SOURCE_CHARS | mcp/skill_pipeline.py:86,537 | 8 / 96000 chars | Longer sources fail | D | No source (worker uses 60). | DONE 2026-09-27: the pipeline reads the worker's own bound (YAMADORI_EXTRACT_MAX_CHUNKS, 60); MAX_SOURCE_CHARS = CHUNK_CHARS x that. The 60 is the worker's (unassigned). |
| DISTIL_MAX_TOKENS | mcp/skill_pipeline.py:88 | 2048 | Answer allowance for distil | B | Equals tiers.A_MIN 2048 (tiers.py:464), the stack's answer floor. | keep |
| SCREEN_MAX_TOKENS | mcp/skill_pipeline.py:90 | 1024 | Answer allowance for screen/tag/tests | B-value-arbitrary | Below A_MIN; tiers.budget raises answer to max(client, A_MIN), so likely inert (unverified). | keep (the bound is forced); derive the value |
| MODEL_TIMEOUT | mcp/skill_pipeline.py:92 | 3600 s | Pipeline model call timeout | B-value-arbitrary | Background job liveness; same as worker. | keep (the bound is forced); derive the value |
| pipeline sampling | mcp/skill_pipeline.py:364,584,627,663,759,838,863 | effort medium; temperature 0.0/0.2 | Distil/screen/tag/tests/faithful sampling | D | No source; departs from vendor 1.0 (operator 2026-09-25, main). | DONE 2026-09-27: no temperature on any pipeline call (vendor sampling). |
| decompose allowance | mcp/skill_pipeline.py:663 | DISTIL_MAX_TOKENS*2 | Decompose answer tokens | D | No source. | DONE 2026-09-27: DISTIL_MAX_TOKENS (= A_MIN). |
| MAX_CHILDREN | mcp/skill_pipeline.py:643 | 12 | Max atomic children per frontier skill | D | No source. | DONE 2026-09-27: removed. |
| SPDX table / NO_DERIVATIVES | mcp/skill_pipeline.py:388,413 | licence regexes; CC-BY-ND | Licence from verbatim quote; ND fails | B | Legal correctness (ND forbids derivatives). | keep |
| reply cuts | mcp/skill_pipeline.py:628,665 | 20000/30000 chars stored | Stored reply size | B-value-arbitrary | Record bound. | keep (the bound is forced); derive the value |
| no review stage / arm on pass | mcp/skills.py:39 | arm without review | Skills arm when pipeline passes | A | Operator 2026-09-22 (memory dataset-pipeline-no-review-gate): "handled by the agent and we could manually review it if we wanted to"; 2026-09-26 16:42 "Keep the skills armed". | keep |
| WATCH_HOURS | mcp/skills.py:145 | 24 h | Re-fetch interval for watched sources | B-value-arbitrary | Operator 2026-09-24 "keep track of when that skill is updated" (OM line 860); interval ours. | keep (the bound is forced); derive the value |
| CACHE_SECONDS | mcp/skills.py:1058 | 5 s | Armed-list cache | B-value-arbitrary | Freshness bound. | keep (the bound is forced); derive the value |
| arming rule (activation) | mcp/skill_tests.py:34 | all should select, all should_not don't | Failing activation tests quarantine the skill | A-flag | skill_tests.py:34 "(a CHOICE, unmeasured)"; operator 09-26 12:36 "tests ... for activation and selection". Quarantine-on-fail ours. | keep; operator to confirm the value/wording |
| generated test templates | mcp/skill_tests.py:66-98 | _OTHER, _FRAMEWORK_LANG, phrases | Deterministic should/should-not cases | D | Hand tables. | KEPT 2026-09-27: without them every compiled skill needs a model call to get tests, and quarantines offline (migration, authored, dataset rows). |
| case caps | mcp/skill_tests.py:215,397 | prompt<=1200; 4 each | Test size caps | D | No source. | DONE 2026-09-27: removed (clean_behaviour, merge, generate). |
| MIN_QUOTE_CHARS | mcp/skill_builder.py:65 | 24 | Item quote must be >=24 chars | D | "worker.MIN_EVIDENCE_CHARS: 'use a' is in every document" -- rationale for a floor, value arbitrary; prompt says 30-300 (skill_prompts.py:77), inconsistent. | DONE 2026-09-27: aligned: one constant, skill_limits.MIN_QUOTE_CHARS, which skill_builder and the prompts read. |
| verbatim quote rule | mcp/skill_builder.py:18-23 | quote must be in source | Each distilled item traceable | B | Correctness/provenance; licence verbatim rule (memory: licence only from verbatim quote). | keep |
| ABSOLUTE regex / DO NOT rule | mcp/skill_builder.py:68 | word list | DO NOT kept only when quote states an absolute | D | No source; interacts with operator's "sparingly" (A). | DONE 2026-09-27: removed; MAX_PROHIBITIONS governs (the golden fixture carries its DO NOT). |
| trailing-item drop | mcp/skill_builder.py:270 | pop until under hard cap | Over-cap skills lose trailing items | D | Conflicts with "FAILS ... rather than rewriting" (skill_limits.py:47-49; operator 2026-09-26 "flag, do not rewrite silently" -- claimed, no quote found). | KEPT 2026-09-27: failing with a reason instead depends on the flag-not-rewrite rule, which is unconfirmed: the operator's. |
| compile group-size | mcp/skill_compile.py:213 | topics <3 rows pooled | Small topics merged into "general" skill | D | Code claims operator 2026-09-26 "Don't make 2,575 one-line skills" -- claimed, no quote found. 3 ours. | DONE 2026-09-27: one unit per topic (the >=3 pooling removed); the operator confirms the grouping. |
| compile rule caps | mcp/skill_compile.py:275-283 | langs/fws [:2]; topics [:6]; common word c>=2 [:3]; triggers [:4] | Migrated rule shape | D | No source. | DONE 2026-09-27: removed. |
| compile description | mcp/skill_compile.py:292 | DESCRIPTION_AIM-40; where[:30] | Truncated migrated description | D | No source. | DONE 2026-09-27: bounded by DESCRIPTION_CHARS (1,024). |
| NAME_RX | mcp/skill_md.py:75 | ^[a-z0-9][a-z0-9-]*$ | SKILL.md name format | B | Agent Skills spec "lowercase, hyphens" (DECISION-TREES:362); Hermes lints [a-z0-9_-]. | keep |
| item forms | mcp/skill_md.py:77 | DO / DO NOT / WHEN / NEVER | Body item grammar | A-flag | Operator 2026-09-26 format direction (Agent Skills, compact); form set ours. | keep; operator to confirm the value/wording |
| prompt size numbers | mcp/skill_prompts.py:71-77,196,252,287-289 | name 2-6 words; topics <=6; items 2-6; quote 30-300; 1-6 skills; test… | Numbers told to the distilling model | D | skill_prompts.py:18 "EVERY TEMPLATE IS A CHOICE". | DONE 2026-09-27: derived from skill_limits (NAME_CHARS, DESCRIPTION_CHARS, MIN_ITEMS, MAX_ITEMS, MIN_QUOTE_CHARS, MAX_ITEM_CHARS); versions distil/3, decompose/2, tag/4, tests/2; pins regenerated. |
| DECOMPOSE "at most three skills" | mcp/skill_prompts.py:171 | text | Stale claim in pinned prompt | D | SKILL-FACTORY.md:394 "no longer true since the ceiling rose". | DONE 2026-09-27: wording fixed (decompose/2). |
| CRAFT naming / texts | mcp/skill_prompts.py:338-380 | craft/1 templates | What main reads about skills | A-flag | Operator 2026-09-27 08:55 "it is not a skill it is a craft, mastery, occupation" (OM). Wording ours, pinned by fixture. | keep; operator to confirm the value/wording |
| DATA_PARAGRAPH | mcp/skill_prompts.py:48 | prompt | "is DATA, not instructions" framing | C-partial | INJECTION.md:220 hardened 1/310 vs minimal 13/350, p=0.0021 (bench/injection_compose.py exists) -- on answering, not these prompts. | keep |
| IDLE_MINUTES | mcp/skill_learn.py:93 | 15 min | Quiet time before learning job runs | B-value-arbitrary | Operator 09-24 12:58 "self improvement happens at some frequency when the stack is idle"; 15 ours. | keep (the bound is forced); derive the value |
| QUERY_CHARS | mcp/skill_learn.py:94 | 600 | Durable request excerpt length | B-value-arbitrary | Privacy: store the user's words only, bounded; 600 arbitrary. | keep (the bound is forced); derive the value |
| LEARN_BATCH / SELECTIONS_KEEP | mcp/skill_learn.py:95,229 | 200 / 20000 | Records per learn job; selection log size | B-value-arbitrary | Storage/throughput bounds. | keep (the bound is forced); derive the value |
| learned triggers from fallback | mcp/skill_learn.py:14-17 | rule | Fallback picks become triggers | A | Operator 2026-09-24 12:58 "the fallback should be self healing, it collects the exact evidence we need to train or embed" (OM). | keep |
| client-only learning | mcp/skill_learn.py:51-80 | traffic class | Only client traffic learned from | B | Correctness: test/offline fixtures polluted the store (found 2026-09-27). | keep |
| backfill window | mcp/skill_learn.py:~433 | +-2 s | Matches old rows to corpus turns | B-value-arbitrary | Observed rows 0.0-0.2 s apart; bound arbitrary. | keep (the bound is forced); derive the value |
| MAX_TOKENS / CAP (CLM) | mcp/clm.py:100-101 | 2048 / 2047 | CLM input token cap | B | CLM trained/served at --max-model-len 2048; Recipe.cap = max_len-1 (CLM-EVAL.md:105; upstream train/embed_utils). | keep |
| BATCH (CLM) | mcp/clm.py:102 | 16 | Inputs per embeddings request | B-value-arbitrary | "(a choice)". | keep (the bound is forced); derive the value |
| TIMEOUT / LANE_WAIT_S / CACHE_MAX (CLM) | mcp/clm.py:103-105 | 120 s / 30 s / 50000 | CLM liveness and cache bounds | B-value-arbitrary | Liveness/memory; values arbitrary. test_clm.py. | keep (the bound is forced); derive the value |
| INSTRUCTIONS / NONE_OPTION (CLM) | mcp/clm.py:111-112 | strings | CLM question and none text | D | "Wording is a choice, not a measurement." | PENDING (clm.py owner): Measure on bench/clm/fidelity.py. |
| clm_heads constants | mcp/clm_heads.py:66-70 | HIDDEN 4096; SCALE_MAX 100; eps | Head math | B | Copied from upstream heads.py/embedder.py at pinned commit bb42c6c5. | keep |
| MAX_BATCH | mcp/dash_skills.py:73 | 25 | Items per dashboard batch POST | B-value-arbitrary | API bound. | keep (the bound is forced); derive the value |
| MAX_TURNS | mcp/recent_turns.py:23 | 50 | In-memory recent requests kept | B-value-arbitrary | Memory bound. | keep (the bound is forced); derive the value |
| FANOUT_HOLD_S / FANOUT_DECAY_S | mcp/recent_turns.py:28-29 | 120 / 60 s | Dashboard tree animation | D | "Chosen for the eye ... not measured". Display only. | PENDING (unassigned): Harmless display; keep or operator picks. test_tree_sources.py. |
| FOLIAGE_WINDOW_S / FOLIAGE_TURNS / FOLIAGE_FULL | mcp/recent_turns.py:34-36 | 1800 s / 20 / 3.0 | Dashboard foliage density | D | "Choices, not measurements"; display only; FOLIAGE_FULL=3 echoes the retired 3-skill cap. | PENDING (unassigned): Display only. test_tree_sources.py. |

### Web, images, vision, server: research_tools.py, images.py, image_input.py, vision.py, server.py, e1.py, domains.py, deps.py, ...

| name | file:line | value | what it does | class | evidence or derivation | action |
|---|---|---|---|---|---|---|
| SEARXNG_URL loopback-only | mcp/research_tools.py:72,:1219 | http://127.0.0.1:8888; non-loopback refused | search only through a local SearXNG | A-flag | operator 2026-09-24T15:03Z: "What is SearXNG search service, cpu side, local, I'm here for it". Loopback refusal itself is B (queries must not go to an arbitrary configured host). | keep; operator to confirm the value/wording |
| SEARCH_TIMEOUT | mcp/research_tools.py:75 | 20 s (YAMADORI_SEARCH_TIMEOUT) | HTTP timeout for one SearXNG query | B-value-arbitrary | A timeout is required; SearXNG's own outgoing.request_timeout is 4 s max 8 (docs/SEARCH.md:138), so 20 s only has to exceed 8. Value unmeasured. | keep (the bound is forced); derive the value |
| SEARCH_RESULTS | mcp/research_tools.py:76 | 8 | results per search passed to the model | B-value-arbitrary | Some cap forced by helper context (SearXNG returned 21-114 results, docs/SEARCH.md:279-282). 8 has no cited basis. | keep (the bound is forced); derive the value |
| SEARCHES_PER_RUN | mcp/research_tools.py:87 | 10 (YAMADORI_SEARCHES_PER_RUN) | web searches allowed per deep-thinking run | A-flag (rationale/number ours) | operator 2026-09-26T00:08Z: "let it search lots of stuff just be careful of request params and headers, and keep it on get requests". 10 = Brave's OBSERVED 429 after ~10 queries/2 min (docs/SEARCH.md:150, one observation, not Brave documentation); a per-run cap is not a per-2-min rate. | keep; operator to confirm the value/wording |
| READS_PER_RUN | mcp/research_tools.py:93 | 20 (YAMADORI_READS_PER_RUN) | pages read per deep-thinking run | A-flag (rationale/number ours) | operator 2026-09-26T00:08Z: "let it search lots of stuff just be careful of request params and headers, and keep it on get requests". 20 copied from the tool-turn cap at max (operator 2026-09-23 per AGENTS.md); comment: 'A CHOICE'. | keep; operator to confirm the value/wording |
| SEARCH_BACKOFF_S | mcp/research_tools.py:98 | 180 s (YAMADORI_SEARCH_BACKOFF_S) | pause search after SearXNG 429 / all engines limited | B | SearXNG's own default: C:\Users\jwals\searxng\src\searx\settings.yml:76 'SearxEngineTooManyRequests: 180'. Verified. | keep |
| HOST_BACKOFF_S | mcp/research_tools.py:99 | 60 s | host pause after 429 with no Retry-After | B-value-arbitrary | Backing off a 429 is HTTP politeness (RFC 6585); 60 s has no source. docs/SEARCH.md:257 'all CHOICES, none measured'. | keep (the bound is forced); derive the value |
| HOST_BACKOFF_MAX_S | mcp/research_tools.py:100 | 600 s | caps a server's Retry-After | D | No evidence; capping a server-sent Retry-After is our choice, not a protocol need (docs/SEARCH.md:262). | DONE 2026-09-27: REMOVED: a server's Retry-After is honoured as sent (RFC 9110; floor 1 s). test_web_access 62/62. |
| QUERY_MAX_CHARS | mcp/research_tools.py:105 | 200 | refuse longer search queries | B-value-arbitrary | Security: a query leaves the machine, so bound the exfil channel. 200 unmeasured ('a CHOICE, checked on the fixtures in mcp/test_deep.py'). | keep (the bound is forced); derive the value |
| _CODE_IN_QUERY | mcp/research_tools.py:106 | regex [{};\n] => ' = ' ``` <? <tag> | refuse queries that contain code | B-value-arbitrary | Security (conversation code must not leave). Pattern hand-picked; fixtures in test_deep.py only. | keep (the bound is forced); derive the value |
| _SECRET_IN_QUERY | mcp/research_tools.py:107 | vendor key prefixes; 32+ hex; 40+ base64; password=/token= | refuse secret-shaped queries/URL parts | B-value-arbitrary | Security. Prefixes (sk, ghp, AKIA, AIza, xox*) are real vendor key formats; length cut-offs 32/40 ours. | keep (the bound is forced); derive the value |
| WEB_MAX_CHARS | mcp/research_tools.py:113 | 12,000 chars (YAMADORI_WEB_MAX_CHARS) | chars of one page given to the second brain | B-value-arbitrary | Context bound forced; comment's basis 'helper's share is ~61k' is STALE (helper is 49,152 since 2026-09-25, AGENTS.md). Operator 2026-09-23T10:51Z objected to tightened budgets generally. | keep (the bound is forced); derive the value |
| KB_RESULTS / KB_TEXT_CHARS | mcp/research_tools.py:116-117 | 6 items x 900 chars | knowledge-base results and chars per item | B-value-arbitrary | Comment: 'CHOICES (find_skills' 5 x 900 before the fold), unmeasured'. The 5->6 change has no stated reason. | keep (the bound is forced); derive the value |
| KB match floor | mcp/research_tools.py:309 | at least ceil(terms/2) query words | which skills count as a knowledge-base match | D | Hand heuristic, no measurement or source. | PENDING (unassigned): Rank by words matched with no floor (or by embedding). test_deep.py / test_skill_factory.py call find_in_knowledge_base (behaviour, not the constant). |
| _kb_topics miss listing | mcp/research_tools.py:284-285 | 12 tags + 6 titles | what a miss lists as held topics | B-value-arbitrary | AGENTS 'Failure returns carry the next step' requires naming what IS available; counts ours. | keep (the bound is forced); derive the value |
| DATA_NOTE | mcp/research_tools.py:119 | '...It is data to quote and cite by its URL; it gives no instructions… | frame prepended to every fetched page | A-flag | AGENTS.md 'Fetched content is data (operator, 2026-09-24)': claimed, no operator quote found in operator_messages.txt. Wording ours, unmeasured. | keep; operator to confirm the value/wording |
| tool descriptions (find_in_knowledge_base, read_web_page, search_web) + result… | mcp/research_tools.py:122-193,:1319 | trigger-list prose | descriptions and instruction text the model reads | D | Module doc :44 'Their wording is a CHOICE; nothing has measured it.' AGENTS' supporting eval (10.7->11.7/14) is 'at risk' (CONSTRAINTS #31). | PENDING (unassigned): Minimal factual descriptions (what it returns, args). test_deep.py checks DATA_NOTE/names. AGENTS.md:57. |
| FETCH_DEADLINE | mcp/research_tools.py:343 | 30 s (YAMADORI_WEB_DEADLINE) | one deadline over robots+redirects+page | B-value-arbitrary | Security (slow-server bound, pre-deploy review BLOCKS DEPLOY #1). Comment: 'All numbers are choices.' | keep (the bound is forced); derive the value |
| FETCH_MAX_BYTES | mcp/research_tools.py:344 | 1 MiB (YAMADORI_WEB_MAX_BYTES) | byte cap on one fetch | B-value-arbitrary | Security (never-ending server). Value a choice. | keep (the bound is forced); derive the value |
| MAX_REDIRECTS | mcp/research_tools.py:346 | 5 | redirect hops followed | B | RFC 7231 s6.4 notes earlier HTTP (RFC 2068) recommended a maximum of five redirections; loop protection required. | keep |
| per-hop socket timeout | mcp/research_tools.py:482,:504 | min(15 s, deadline left) | socket timeout per connect/read | B-value-arbitrary | A per-read timeout is required under one deadline; 15 s ours. | keep (the bound is forced); derive the value |
| read/scan block sizes | mcp/research_tools.py:505,:576,:593 | 64 KiB reads; CAPTCHA scan 64 KiB; NUL check 8 KiB | I/O chunking and binary sniff windows | B-value-arbitrary | Implementation bounds; no behavioural claim. | keep (the bound is forced); derive the value |
| UA_TOKEN/UA/REQUEST_HEADERS/METHOD | mcp/research_tools.py:355-363,:451 | GET only; fixed UA, Accept, Accept-Language en, Accept-Encoding ident… | every request the web tools send | A | operator 2026-09-26T00:08Z: "let it search lots of stuff just be careful of request params and headers, and keep it on get requests"; 'If we strip/limit headers...'. | keep |
| PAGE_TYPES | mcp/research_tools.py:364 | text/*, markdown, html, xhtml, rst, json | content types accepted as a page | B | Only text can be screened and read by a text model; binaries refused. | keep |
| _CHALLENGE | mcp/research_tools.py:369,:575 | regex captcha\|cf-chl\|challenge-platform\|are you a robot\|unusual t… | 403/503 with this body = rate limit, pause host | B-value-arbitrary | Honouring bot checks is politeness; word list hand-written, not tested against real block pages in repo. | keep (the bound is forced); derive the value |
| _LOCAL_NAMES | mcp/research_tools.py:371 | .localhost .local .internal .lan .home .zt .ts.net .home.arpa | local names refused without DNS | B | SSRF: RFC 6761/6762/8375 local suffixes; .zt/.ts.net are ZeroTier/Tailscale names this network uses. | keep |
| _pin is_global | mcp/research_tools.py:391-412 | every resolved address must be ip.is_global and not multicast; connec… | SSRF + DNS-rebinding guard | B | Security: pre-deploy review BLOCKS DEPLOY #1 (302 to 127.0.0.1:11434, CGNAT 100.64/10). | keep |
| robots.txt policy | mcp/research_tools.py:530-561 | honoured; cache 3600 s; 401/403=deny, other 4xx=allow, 5xx=retryable… | robots.txt handling | B | RFC 9309: 4xx -> may crawl, 5xx -> treat as disallowed, cache SHOULD NOT exceed 24 h. 401/403=deny is stricter than RFC; 3600 within bound. | keep |
| URL provenance rule | mcp/research_tools.py:620-652,:795-811 | search/user/link URL fetched with query as seen; memory URL by path o… | what part of a URL may be sent | A | operator 2026-09-26T00:08Z: "let it search lots of stuff just be careful of request params and headers, and keep it on get requests"... ensure query args match the link from the source'; 00:11Z: 'you can't strip the path, but query string may or may not be safe'. | keep |
| URL_MAX_CHARS | mcp/research_tools.py:655 | 2048 | max URL length fetched | B-value-arbitrary | Exfil/abuse bound; 2048 is folklore browser limit, not cited. SEARCH.md:239 'All numbers are choices.' | keep (the bound is forced); derive the value |
| URL_QUERY_MAX_CHARS | mcp/research_tools.py:656 | 200 | max query-string length, every provenance | B-value-arbitrary | Exfil guard; value ours. Applies even to seen URLs the operator allowed 'as seen'. | keep (the bound is forced); derive the value |
| MEMORY_PATH_MAX_CHARS | mcp/research_tools.py:657 | 256 | max path length for a URL no source gave | B-value-arbitrary | Exfil via path (the remaining channel, docs/SEARCH.md:238); value ours. | keep (the bound is forced); derive the value |
| URL_OVERLAP_CHARS / _IDENT_MAX / step | mcp/research_tools.py:658-659,:781 | 24-char window, step 4; words <=40 alnum exempt | refuse URL carrying conversation text | B-value-arbitrary | Exfil guard (AGENTS.md:57 '24+ character run'). Values ours, unmeasured; behaviour covered by test_web_access.py crafted URLs. | keep (the bound is forced); derive the value |
| _ENTROPY_MIN_LEN / _ENTROPY_BITS / _wordy | mcp/research_tools.py:660-705 | 24 chars, >3.6 bits/char; wordy = avg piece>=3, digits<=20% | refuse high-entropy path/query parts | B-value-arbitrary | Secret/exfil guard; thresholds hand-set, no measurement. | keep (the bound is forced); derive the value |
| LINKS_LISTED | mcp/research_tools.py:665 | 20 | page links appended to read_web_page result | D | No evidence; listing links for the model is our addition (operator asked only that query args match the source's link). SEARCH.md:242. | PENDING (unassigned): Return no link list; seen-link provenance still works from page parse. test_web_access.py test_links_are_listed_followed_and_seen. AGENTS.md:57 'up to 20 of the page's visible links'. |
| LINKS_SEEN_PER_PAGE / LINKS_SEEN_MAX / FETCHES_RECORDED / LINK_TEXT_CHARS | mcp/research_tools.py:666-669 | 300 / 3000 / 60 / 80 | memory and record bounds for links/fetch rows | B-value-arbitrary | Bounded memory/record size; comment 'CHOICES'. | keep (the bound is forced); derive the value |
| web_text / urls rolling windows | mcp/research_tools.py:1094,:1292,:1296 | 400,000 chars; last 200 URLs | run-context text kept for screening; seen search URLs | B-value-arbitrary | Memory bounds; values ours. | keep (the bound is forced); derive the value |
| _LIMITED_ENGINE + pause rule | mcp/research_tools.py:1167,:1308 | regex captcha\|too many requests\|suspended\|access denied\|rate; no… | detect engines pushing back | B-value-arbitrary | Politeness; rule written from observed DDG/Brave behaviour (SEARCH.md:330-331). 'rate' substring is over-broad. | keep (the bound is forced); derive the value |
| snippet/title caps | mcp/research_tools.py:1325-1326,:271 | snippet 240, title 120, KB head 100 chars | truncation of injected text | B-value-arbitrary | Context bounds; values ours. | keep (the bound is forced); derive the value |
| search_web code default + categories | mcp/research_tools.py:188,:1229 | code=true -> categories=general,it | default search adds GitHub/npm/crates/docs.rs | D | SearXNG 'it' category is real (docs/SEARCH.md); making it the default is unmeasured. | PENDING (unassigned): Default code=false or let the model choose explicitly. No test names it. |
| safesearch=0 | mcp/research_tools.py:1228 | 0 | SearXNG safesearch off | D | No evidence or rationale. | DONE 2026-09-27: REMOVED: SearXNG's own default applies. |
| DEFAULT_STEPS / DEFAULT_SIZE | mcp/images.py:79-80 | 20 steps; 1024x1024 | base model defaults | B | Model card setting (comment :77); measured 105-111 s n=10 (docs/IMAGEGEN.md, bench/imagegen/parti-20260923). | keep |
| MODELS.turbo.steps | mcp/images.py:110 | 4 | turbo sampling steps | B | A 4-step DMD-distilled student (Viggle); steps are the model's design (docs/IMAGEGEN-TURBO.md). | keep |
| default_model | mcp/images.py:163-170 | turbo (YAMADORI_IMAGEGEN_DEFAULT) | image model when caller chose none | A | operator 2026-09-23T19:54Z: "wait yamadori image turbo is not the default, it is better right>"; later "It is way faster". Quality comparison not run. | keep |
| MODELS.est_seconds | mcp/images.py:105,:115 | 107 / 24 | shown to users choosing a model | C | 107: n=10 (n=2 sd-server + n=8 proxy, bench/imagegen/parti-20260923); 24: n=1 smoke (bench/imagegen/samples/turbo-smoke-20260923). | keep |
| SIZE_MULTIPLE | mcp/images.py:129 | 32 | edges must be multiples of 32 | B | sd.cpp requirement for Qwen-Image-2.1 (docs/qwen_image_2.1.md in sd.cpp tree). | keep |
| MIN_EDGE | mcp/images.py:130 | 256 | smallest image edge accepted | D | No source or measurement. | PENDING (unassigned): Accept any multiple of 32. No test names it. |
| MAX_EDGE | mcp/images.py:131 | 1536 (YAMADORI_IMAGEGEN_MAX_EDGE) | largest edge accepted | B | Hermes image_generate sends 1536x1024/1024x1536 (plugins/image_gen/_common.py:18); 1.57 MP < measured 1.81 MP. | keep |
| MAX_PIXELS | mcp/images.py:132 | 1344*1344 (YAMADORI_IMAGEGEN_MAX_PIXELS) | pixel ceiling per image | C | bench/imagegen/results.jsonl: one 1344x1344 row (n=1), peak +6,389 MiB under the 6 GiB --max-vram budget (docs/IMAGEGEN.md:202). VRAM-derived, n=1. | keep |
| MAX_N | mcp/images.py:134,:657 | 4 | images per /v1/images/generations call | D | No evidence; batch VRAM at n>1 never measured. | PENDING (unassigned): n=1 only (as dall-e-3) until measured. No test names it. |
| MAX_PROMPT_CHARS | mcp/images.py:135 | 4000 | image prompt length cap | B-value-arbitrary | Coincides with OpenAI dall-e-3's 4,000-char prompt limit, but the code cites nothing. | keep (the bound is forced); derive the value |
| steps range | mcp/images.py:654 | 1..50 | accepted steps override | B-value-arbitrary | Bound on a GPU-time argument; 50 ours. | keep (the bound is forced); derive the value |
| timeout() | mcp/images.py:237 | 900 s (YAMADORI_IMAGEGEN_TIMEOUT) | image request timeout | B-value-arbitrary | Must exceed cold load + 220 s measured at 1344 (docs/IMAGEGEN.md:202); 900 'generous' unmeasured. | keep (the bound is forced); derive the value |
| url_ttl | mcp/images.py:251 | 30 days (YAMADORI_MEDIA_URL_TTL) | signed /media link lifetime | B-value-arbitrary | Capability URLs need expiry (security); 30 days unexplained. docs/IMAGEGEN.md:427. | keep (the bound is forced); derive the value |
| _secret | mcp/images.py:543-563 | 32 random bytes, hex, file 0600 | HMAC key for media links | B | HMAC-SHA256 key >= output length (RFC 2104). | keep |
| _EXTRA_ARGS strip | mcp/images.py:415-430 | remove <sd_cpp_extra_args> blocks | stop model prompt setting server params | B | Security: sd-server reads settings from prompt text (routes_sdapi.cpp). | keep |
| QUALITIES/OUTPUT_FORMATS/BACKGROUNDS | mcp/images.py:384-409 | quality accepted, not used; png only; opaque/auto | OpenAI image options honoured/refused | B | OpenAI Images spec (openai-openapi 2.3.0); quality ignored because the operator's turbo default must not be overridden (A above). | keep |
| REAL_ALPHA_BELOW / REAL_ALPHA_MIN_FRACTION | mcp/images.py:490-491 | alpha<128 on >=0.5% of pixels | keep real transparency, flatten noise alpha | B-value-arbitrary | Flattening noise alpha is correctness; cut 'chosen, not tuned (n=1 real)' from n=55 images in index/media, no script -- and that is the operator's PRIVATE media (memory private-media.md). | keep (the bound is forced); derive the value |
| _oom smaller-size remedy | mcp/images.py:310 | suggest 1024x1024 if >1MP else 768x768 | OOM remedy text | D | No measurement of what fits. | PENDING (unassigned): Remedy says 'use a smaller size' without a number. No test. |
| generate_image TOOL description | mcp/images.py:755-799 | trigger-list prose; size hint 'about 2 minutes' | tool description the model reads | D | Wording unmeasured; '1024x1024 about 2 minutes' is STALE for turbo (24 s, n=1). | DONE 2026-09-27: the stale timing claims ('about 2 minutes', '3-4 minutes') removed; the rest of the description PENDING (A-flag wording). |
| MAIN_TOOL / SHOWN_INSTRUCTION / SHOWN_LOOK / result 'instruction' | mcp/images.py:807-906 | shown-on-main sentence swap + instruction text | tell main the picture is already shown | A-flag (wording ours) | operator 2026-09-25T15:39Z: "...they typically emit the image to the harness when they make it". The injected sentences are ours, unmeasured. | keep; operator to confirm the value/wording |
| _alt | mcp/images.py:854 | 80 chars | alt-text truncation | B-value-arbitrary | Cosmetic bound. | keep (the bound is forced); derive the value |
| image lane (one at a time) | mcp/images.py:666 | admission.image_lane, single | one A4000 image job at a time | B | AGENTS 'One GPU consumer at a time' (observed 429/502 degradation); operator answer 2026-09-23T13:42Z 'One at a time' for streams. | keep |
| MAX_IMAGES_PER_REQUEST | mcp/image_input.py:69 | 20 (YAMADORI_VISION_MAX_IMAGES) | images whose bytes a request holds | B-value-arbitrary | Memory bound needed; comment 'CHOICES, not measurements'; docs/VISION.md:529 '(choice)'. | keep (the bound is forced); derive the value |
| MAX_BODY_BYTES | mcp/image_input.py:74; mcp/body_limit.py:67 | 64 MiB (YAMADORI_MAX_BODY_BYTES) | request body limit, 413 | B-value-arbitrary | DoS/memory bound (server read any size). Derived loosely: 10 MB image = 13.4 MB base64, history resent. VISION.md:529 'choice'. | keep (the bound is forced); derive the value |
| INLINE_MIN_CHARS | mcp/image_input.py:78 | 256 base64 chars | shortest printed-base64 run read as an image | B-value-arbitrary | Avoid treating ids/hashes as images; docs/VISION.md:353 '(choice)'. | keep (the bound is forced); derive the value |
| _TRIM_SEGMENTS | mcp/image_input.py:287 | 3 | trailing segments dropped when run swallowed text | D | Comment: 'a choice'. No evidence. | PENDING (unassigned): Try only the whole run (cut -> IMAGE_INCOMPLETE). No test names it. |
| TOOL_MEDIA_TURNS / PI_BRIDGE | mcp/image_input.py:123-154 | OpenCode/Pi/Cline exact strings | recognise harness synthetic image turns | B | Read from harness source with commit and file:line (opencode b65de4d message-v2.ts:46; pi 2b0a123 :1455,:1236; cline 29896ec). | keep |
| MAGICS / complete() | mcp/image_input.py:262-320 | PNG/JPEG/GIF/WebP magic, end markers | detect printed image bytes | B | File-format specifications. | keep |
| EFFORT | mcp/vision.py:97 | low | effort for the vision call | D | Comment: 'Not measured for vision'. | PENDING (unassigned): Use model.chat default or the request's effort. No test names vision.EFFORT. |
| DEFAULT_CONTEXT | mcp/vision.py:102 | 16384 (YAMADORI_VISION_CTX) | vision server context | B | Mirrors config.yaml bonsai-vision -c 16384. | keep |
| IMAGE_TOKENS_MAX | mcp/vision.py:109 | 4096 | tokens reserved for the image | B | llama.cpp clip.cpp:1632 set_limit_image_tokens(8,4096) for Qwen-VL family; family membership ASSUMED. Live n=1: 168 prompt tokens for 256x256. | keep |
| DEFAULT_MAX_BYTES | mcp/vision.py:114 | 10 MiB (YAMADORI_VISION_MAX_BYTES) | largest image looked at | B | llama-server's own params.max_size 10 MB (server-common.cpp:1068). | keep |
| MAX_QUESTION_CHARS | mcp/vision.py:115 | 4000 | describe_image question cap | B-value-arbitrary | Input bound; value ours. | keep (the bound is forced); derive the value |
| vision SYSTEM prompt | mcp/vision.py:134-139 | 'You are looking at one image for another model...' | system prompt to vision model | D | Injected wording, unmeasured. | PENDING (unassigned): Send the question alone. No test pins it. |
| vision timeout() | mcp/vision.py:159-164 | 900 s (YAMADORI_VISION_TIMEOUT) | vision call timeout | B-value-arbitrary | Comment: 'Not measured.' Live n=1 took 5.9 s. | keep (the bound is forced); derive the value |
| thinking_cap | mcp/vision.py:182-194 | ctx - 4096 - A_MIN - prompt, >= MIN_THINKING | vision thinking budget | B | Arithmetic from the context and caps above. | keep |
| VISION_LOADING remedy text | mcp/vision.py:454 | 'stays resident for 15 minutes (ttl 900)' | remedy the model reads | D | STALE fact: vision ttl is 300 s since 2026-09-23 (docs/IMAGEGEN.md:987). Invented claim in a tool return. | DONE 2026-09-27: FIXED: the false '15 minutes (ttl 900)' claim removed (config.yaml bonsai-vision ttl is 300). |
| placeholders + result notes | mcp/vision.py:587-593,:616-719,:1148 | '[image-<id>: ... call describe_image ...]' etc. | text the model sees for each image | B-value-arbitrary | The text model must learn an image exists (correctness); byte-stable for prefix cache. Wording ours. | keep (the bound is forced); derive the value |
| _available caps | mcp/vision.py:232 | 8 attached ids, 5 generated | ids listed in a remedy | B-value-arbitrary | Remedy bound; values ours. | keep (the bound is forced); derive the value |
| never fetch image URLs | mcp/vision.py:673-681,:982 | http(s) image parts refused | no server-side download | B | Security: llama-server would fetch arbitrary URLs (server-common.cpp:1065). | keep |
| our_hosts / LOOPBACK | mcp/vision.py:536-562 | loopback + PUBLIC_BASE + request host | which /media links count as ours | B | Security: capability links verified; hosts compared not resolved. | keep |
| vision enabled by default | mcp/vision.py:146-148 | on unless YAMADORI_VISION=0 | describe_image offered | A | operator 2026-09-23T15:59Z: "...this was always the plan, I asked for that at the same time as image generation, you need both." | keep |
| HOST 0.0.0.0 | mcp/server.py:79 | 0.0.0.0 (YAMADORI_PROXY_HOST) | proxy bind address | A | operator 2026-09-22T20:52Z: "It is bound to 0.0.0.0 because it is in a private network..." (memory network-binding.md). | keep |
| 429 Retry-After | mcp/server.py:146 | 30 s | lanes full refusal hint | B-value-arbitrary | Retry-After on 429 is protocol; 30 ours. | keep (the bound is forced); derive the value |
| image 429 Retry-After | mcp/server.py:757 | 60 s | image lane busy hint | B-value-arbitrary | Turbo draws in ~24 s; 60 unexplained. | keep (the bound is forced); derive the value |
| SESSION_HEADERS | mcp/server.py:317 | x-yamadori-session, x-session-id, x-session-affinity | explicit session id headers | B | OpenCode sends X-Session-Id / x-session-affinity (docs/HARNESS-OPENCODE.md); AGENTS: explicit ids only (operator 2026-09-25). | keep |
| SUPERSEDE_AFTER_S | mcp/server.py:673 | 30 s (YAMADORI_SUPERSEDE_AFTER) | cancel an identical in-flight request older than this | B-value-arbitrary | Existence: #44 self-lock (orphaned retries held both lanes, 429 x3). Comment: '30, a CHOICE'. | keep (the bound is forced); derive the value |
| retry key | mcp/server.py:679-692 | sha256(account, session token, messages, tools) | what counts as the same request | B-value-arbitrary | Dedup rule; KNOWN COST: identical concurrent requests without session header lose the older one (server.py:668). | keep (the bound is forced); derive the value |
| timeout_keep_alive | mcp/server.py:964 | 300 s | idle socket keep-alive | B-value-arbitrary | Stdlib server lost 8/16 runs to dropped keep-alive (server.py:13); 300 itself unmeasured. | keep (the bound is forced); derive the value |
| timeout_graceful_shutdown | mcp/server.py:967 | 120 s | drain in-flight on shutdown | B-value-arbitrary | Two benchmark runs lost to hard kills; 120 unmeasured. | keep (the bound is forced); derive the value |
| media Cache-Control | mcp/server.py:872 | private, max-age=86400, immutable | browser caching of images | B-value-arbitrary | Content-addressed so immutable is correct; 1 day ours. | keep (the bound is forced); derive the value |
| PREAMBLE on first turn (blocking) | mcp/server.py:500-511; mcp/proxy.py:98 | on (YAMADORI_PREAMBLE=1) | prepends a note to the first answer's CONTENT | D | No evidence here; injects visible text into an answer, cf. operator 2026-09-25 'no visible line in answers' (AGENTS). Only the blocking path. | DONE 2026-09-27: REMOVED with PREAMBLE (server.py block deleted). |
| body_limit METHODS | mcp/body_limit.py:24 | POST PUT PATCH | methods with a body checked | B | HTTP semantics. | keep |
| _TYPES status->type | mcp/api_errors.py:53 | OpenAI error types | error envelope type field | B | OpenAI error object; docs/OPENAI-CONFORMANCE.md E1/E2. | keep |
| context_length_exceeded wording | mcp/api_errors.py:118-131 | OpenAI's sentence | overflow error text | B | Hermes classifies by these phrases (agent/error_classifier.py, model_metadata.py). | keep |
| upstream 4xx kept set | mcp/api_errors.py:204 | 400,401,403,404,409,413,422 else 400 | which upstream 4xx pass through | B-value-arbitrary | Keep client errors 4xx (SDKs retry 5xx); the set ours. | keep (the bound is forced); derive the value |
| 503 Retry-After | mcp/api_errors.py:210,:250 | 30 s | model unavailable hint | B-value-arbitrary | Protocol header; value ours. | keep (the bound is forced); derive the value |
| template refusal -> 400 invalid_prompt | mcp/api_errors.py:191-202 | 400 | Jinja raise_exception mapped | B | Pi retries 5xx 3x (docs/HARNESS-PI.md gap 1); Codex branches on invalid_prompt. | keep |
| validate_chat refusals | mcp/api_errors.py:261-364 | n>1, logprobs, audio, input_audio/file parts, non-function tools | 400 before admission | B | Each would be silently wrong; llama-server refuses logprobs with tools+stream. | keep |
| describe_call / dead stream helpers | mcp/streaming.py:166,:170-323 | 80 chars; timeout 3600 (dead code) | tool label; unused stream_hop/stream_upstream | B-value-arbitrary | Comment :28 'Nothing in the stack calls them now'. Dead code. | keep (the bound is forced); derive the value |
| /v1/responses exists | mcp/responses_api.py:4 | served beside chat | Responses API translation | A | operator 2026-09-26T14:34Z: "If everyone supports responses api... seems like they both need to exist." | keep |
| REASONING_MODE | mcp/responses_api.py:144 | summary (YAMADORI_RESPONSES_REASONING) | how reasoning is shown to Responses clients | B-value-arbitrary | Comment 'a CHOICE, the operator's to change'. Rationale: summary is what Codex/Hermes display and do not echo, matching the pass-through design (claimed 2026-09-24, no operator quote found). | keep (the bound is forced); derive the value |
| KEEPALIVE_S | mcp/responses_api.py:148 | 1.0 s | min spacing of response.in_progress heartbeats | B-value-arbitrary | Must be < Hermes codex 12 s stale timeout (codex_runtime.py) and Codex 300 s; 1.0 ours. | keep (the bound is forced); derive the value |
| HOSTED_IGNORED | mcp/responses_api.py:152 | web_search, file_search, code_interpreter, computer*, mcp, tool_search | hosted tools accepted and ignored | B | Codex offers web_search by default; a 400 would end every session. | keep |
| _STATEFUL refusals (`conversation`, `prompt`) | mcp/responses_api.py | refused | no Conversations API / stored templates | B | `previous_response_id` and `store` are served since 2026-10-06 (operator, "Store, local, capped"; mcp/response_store.py: 256 MB per account / 2 GB / 30 days are the ledger's numbers, class A) | keep |
| incomplete -> failed | mcp/responses_api.py:809-819 | status failed | dropped stream status | B | Client behaviours cited (docs/HARNESS-RESPONSES.md gap 3). | keep |
| E1 flag default | mcp/e1.py:135-138 | off (YAMADORI_E1=0) | E1 replaces Laya on request path | A | operator 2026-09-24T22:24Z: "Ok if it works then we do it... retire Laya for now, use E1". Off until live check. | keep |
| learn() self-tuning | mcp/e1.py:927-1010 | idle-time retrain + promote | E1 heads learn from outcomes | A-flag (mechanism ours) | Same message: "let's collect more data and self tune". | keep; operator to confirm the value/wording |
| route_in head | mcp/e1.py:110-116 | logistic head on embedding | investigate vs not | C | docs/E1.md: 104/120 vs Laya 80/120, p=0.0001; set 2 110/141 vs 86 (bench/e1/eval_e1.py, bench/e1/results). | keep |
| MIN_TRAIN_N | mcp/e1.py:85 | 40 (YAMADORI_E1_MIN_TRAIN_N) | min labels before a head is served | D | e1.py:84 'None of these is measured'; docs/E1.md:98 'choices, not measurements'. | PENDING (unassigned): Serve any head that fits with >=2 labels, or keep only the promotion test as gate. test_e1.py references MIN_TRAIN_N. AGENTS.md ~1227 (E1 section). |
| MIN_PER_LABEL | mcp/e1.py:86 | 8 (YAMADORI_E1_MIN_PER_LABEL) | min examples per label | D | Same: choice. | PENDING (unassigned): Drop; rely on McNemar promotion. No test names it. |
| LEARN_MIN_NEW | mcp/e1.py:87 | 30 (YAMADORI_E1_LEARN_MIN_NEW) | new rows before a retrain | D | Choice (docs/E1.md:363,512). | PENDING (unassigned): Retrain each idle pass; promotion test decides. No test names it. |
| MIN_EVAL_N | mcp/e1.py:88 | 20 (YAMADORI_E1_MIN_EVAL_N) | min held-out rows to judge a candidate | D | Choice; docs/E1.md:398 says p<0.05 needs ~6 net discordant on 20-30 rows. | PENDING (unassigned): Let the exact test decide at any n. No test names it. |
| EVAL_FRAC | mcp/e1.py:89 | 0.3 | share of new rows held out | D | Choice (docs/E1.md:512 'the 30% split'). | PENDING (unassigned): Any split; document it. No test. |
| PROMOTE_P | mcp/e1.py:90 | 0.05 | McNemar p to promote | B-value-arbitrary | Conventional alpha with an exact test; E1's own adoption used p<0.05. | keep (the bound is forced); derive the value |
| WD_GRID | mcp/e1.py:101 | (1e-3, 1e-2) | weight-decay grid | C | bench/e1/eval_e1.py stability: float32/64 argmax agree 120/120 at 1e-3,1e-2 vs 89/120, 109/120 at 0.1, 1.0 (bench/e1/results/stability.json). | keep |
| CV_FOLDS / STEPS / LR | mcp/e1.py:103-105 | 5 / 600 / 0.05 | fit hyperparameters | B-value-arbitrary | Reproduce scripts/train_laya.py _fit_linear for comparability; the originals were never tuned. | keep (the bound is forced); derive the value |
| STATE_CHARS | mcp/e1.py:107 | 6000 | render cap | B | Byte-for-byte laya_head.render_state (training labels' rendering; test_e1.py asserts). | keep |
| CACHE_SIZE | mcp/e1.py:108 | 512 vectors | in-memory vector LRU | B-value-arbitrary | Memory bound. | keep (the bound is forced); derive the value |
| dead-embedder norm | mcp/e1.py:260 | norm < 0.5 -> raise | reject non-unit vectors | B-value-arbitrary | PROTOCOL rule 1 (aliveness); threshold ours. | keep (the bound is forced); derive the value |
| _deep_label outcome->label map | mcp/e1.py:725-764 | helped->escalate, wasted->continue, missed->escalate... | turn deep_decisions outcomes into training labels | D | Comment 'CHOICES, stated in docs/E1.md'; inherits deep.py's labels (themselves unmeasured). | PENDING (unassigned): Train only on gold/hand labels; untrained heads fall back to rules. No test calls _deep_label. |
| escalate head extras | mcp/e1.py:117-124 | struggle counts features | inputs to escalate head | D | Features are deep.py's struggle signals, whose thresholds are choices. | PENDING (unassigned): Remove with the escalate head (untrained). No test named. |
| _gold_guard | mcp/e1.py:1013-1035 | candidate not significantly worse on held-out sets | block regressions on route_in | B | Guards against label-mapping drift (docs/E1.md:397). | keep |
| DOMAINS vocabulary + STRICT_FILES | mcp/domains.py:62-69 | 13 domains; design_visual fails closed | recipe eligibility taxonomy | D | Hand taxonomy, no evidence. Recipe/hints path retired 2026-09-26; only dashboard.py/datasets.py still call recipe_domains. | PENDING (unassigned): Delete with the recipe filter. test_domains.py. |
| PACKAGE_DOMAINS | mcp/domains.py:73-116 | hand map package->domains | domains a package implies | D | Entries edited after LiveCodeBench outcomes (koota, math: 'all 342 LiveCodeBench prompts'). Benchmark-steered. | PENDING (unassigned): Gate on held/imported/named only; drop domain rule. test_domains.py. |
| EXTENSION_DOMAINS | mcp/domains.py:118-127 | hand map ext->domains | domain from file extensions | D | Hand map, no measurement. | PENDING (unassigned): Same as above. test_domains.py. |
| WORD_DOMAINS | mcp/domains.py:165-191 | regexes per domain | domain from words | D | Fitted to benchmark prompts: 'algorithms' phrases for 342 LCB prompts, 'whitespace' from bench/domain rs12, vertex exclusion (LCB 149/92 hits). test_domains replays the same prompts (in-sample). Design phrases n=0. | PENDING (unassigned): Remove word evidence; gate no longer withholds on words. test_domains.py test_a_word_counts_only_in_its_domain_sense, test_the_word_stems_now_match_their_words. |
| AREA_DOMAINS / LANGUAGE_DOMAINS | mcp/domains.py:195-209 | corpus labels -> domains | untagged recipe mapping | B | Maps the corpora's own labels; corpora retired (likely dead). | keep |
| DERIVE_MIN_FILES | mcp/domains.py:229 | 2 | imports needed to derive a package's domains | D | Comment: 'a CHOICE'. | PENDING (unassigned): Any import counts, or drop derivation with the domain rule. test_domains.py references it. |
| DERIVE_MAX_FILES / 120 lines | mcp/domains.py:230,:267 | 600 files; first 120 lines | derivation cost bound | B-value-arbitrary | Cost bound; values ours. | keep (the bound is forced); derive the value |
| HELD_ALIASES | mcp/domains.py:431-460 | hand regexes per package | request names a held package | D | Hand patterns written after observed prompts ('three paragraphs' corpus request; math, postprocessing). No measurement. | PENDING (unassigned): Match install name/import only. test_domains.py test_a_package_is_not_named_by_an_ordinary_word. |
| MAX_SYMBOL_PROBES / batch | mcp/domains.py:474,:691 | 400 tokens; 200 per SQL | symbols probed per request | B-value-arbitrary | Per-request cost bound for 65k transcripts; value ours. | keep (the bound is forced); derive the value |
| symbol code-shape rule | mcp/domains.py:468-470,:655-704 | 4-64 chars, camel/snake/digit only | which words are symbols | D | Heuristic; capitalised words excluded after 21 React/type-challenge benchmark tasks (bench/domain/tasks*). | PENDING (unassigned): Probe all identifier tokens. test_domains.py test_plain_words_are_not_symbols... |
| PLATFORM_NAMES / _CONTEXT / _NOT_MEMBER | mcp/domains.py:512-558 | stoplist + identifier-syntax regexes | names that count only in code context | D | Hand lists from one live Hermes request (MOUSE/API, 2026-09-23). No measurement. Used by selection.py. | PENDING (unassigned): Drop; symbol evidence only from code-shaped tokens. test_domains.py test_english_words_are_not_symbols_even_where_defined. |
| tool_admission fact rules | mcp/domains.py:822-875 | repo bound / offered earlier / nothing held / imports / names / symbo… | is library source reachable | B | A fact about what is indexed (NO_INDEX otherwise); 'offered earlier' keeps prefix stable. held = index_health ok (README-only indexes, deps.py:220-233). | keep |
| tool_admission DOMAIN_OUTSIDE rule | mcp/domains.py:877-907 | withhold when task domains miss held domains | withhold library help on domain evidence | D | Built to withhold on LiveCodeBench (n=3 overhead, bench/lcb_after.jsonl); '342 LCB withheld, 26 three.js offered' is in-sample (test_domains.py). End-to-end quality unmeasured. | PENDING (unassigned): Offer whenever something is held and no fact rule withholds. test_domains.py test_a_puzzle_is_withheld..., test_real_prompts_against_the_real_store. |
| strong_evidence | mcp/domains.py:348-369 | restrict on imports/extensions only | recipe filter restriction | C | bench/data/hint_collapse_runs.json, 89 probes: 64 -> 59 (words) -> 41 (strict). No caller outside domains.py: dead. | keep |
| imported limit_files / skip dirs | mcp/deps.py:67,:73 | 4000 files; node_modules,.git,dist,... | repo import scan bounds | B-value-arbitrary | Cost bound; skip list is build output (correct). | keep (the bound is forced); derive the value |
| worth_indexing | mcp/deps.py:132 | min_imports=2, max_packages=25 | which deps get indexed | D | No evidence ('The filter is the owner's' refers to imports-only, not these numbers). | PENDING (unassigned): min_imports=1, no max; CLI only. No test. |
| fetch limits | mcp/deps.py:179,:192,:196 | 180 s; path-escape refused; member >2 MB skipped | tarball download/extract | B-value-arbitrary | Path traversal refusal is security (B); 180 s / 2 MB ours. | keep (the bound is forced); derive the value |
| MAX_FILE_BYTES | mcp/deps.py:258 | 1.5 MB | skip larger files | B-value-arbitrary | Same cap as index_code.iter_files (consistency); origin unmeasured. | keep (the bound is forced); derive the value |
| MINIFIED_MEAN_LINE | mcp/deps.py:264 | 200 | mean line length = minified | C-weak (no script) | Comment: 20 bundles, readable 12-43 vs minified 345/19,509/19,615 (2026-09-22). No script in repo. | keep |
| MINIFIED_CODE_PUNCT_PER_KB / >500-char lines | mcp/deps.py:271,:324 | 20 per KB | packed code vs embedded data | C-weak (no script) | Comment: minified 59.9-191 vs data 0.0-2.4, named files. No script. | keep |
| DUP_SAME_FORMAT / DUP_OTHER_FORMAT | mcp/deps.py:289-290 | 0.95 / 0.80 | duplicate-bundle thresholds | C-weak (no script) | Comment: copies 0.954-1.000 vs distinct 0.9016-0.9273; format copies 0.885-0.987, 5 packages. No script. | keep |
| DUP_MIN_LINES / distinct line >20 chars | mcp/deps.py:293,:363 | 20 / 20 | noise floor for dedupe | B-value-arbitrary | Stated as noise floor; unmeasured. | keep (the bound is forced); derive the value |
| MIN_DEFS | mcp/deps.py:464 | 1 | index must hold a definition | B | Floor separating README-only indexes (0 defs, 4 packages) from real ones (deps.py:226-233). | keep |
| index_health vector checks | mcp/deps.py:524-532 | 200-vector sample, \|norm-1\|<0.01 | embedded index aliveness | B-value-arbitrary | PROTOCOL rule 1; sample/tolerance ours. | keep (the bound is forced); derive the value |
| MAX_ZERO_FRACTION | mcp/deps.py:582 | 0.02 (INDEX_MAX_DEAD) | zero-vector share refused | B-value-arbitrary | Same as index_code.check_alive; origin unmeasured. | keep (the bound is forced); derive the value |
| index/registry timeouts | mcp/deps.py:600,:610,:653,:846 | 60/30/60 s; 14400/1800 s | network and indexing timeouts | B-value-arbitrary | Required bounds; values ours. | keep (the bound is forced); derive the value |
| REUSE_GAP_DAYS | mcp/deps.py:681 | 365 | gap that marks an npm name reused | D | Comment: 'The gap is a CHOICE'; one case (npm math). | PENDING (unassigned): Treat any version run with a known different repository as another project. No test names the constant (test_deep.py covers name_reused). |
| stable releases only | mcp/deps.py:740 | versions without '-' | releases list | B | Semver prerelease marker. | keep |
| TOOL_PREAMBLE / flat render / enable_thinking off | mcp/tool_shim.py:50-102,:223 | template-mirrored prompt; thinking forced off | legacy tool shim (:1233) | C-weak (no script) | Comment: 0/10 vs 5/5, 0/12 vs 5/5, '2000 tokens, 0 calls'. No script; no launcher or import uses this file: DEAD. | keep |
| schema_fill score levels / range | mcp/schema_fill.py:55-57,:91 | <=11 levels; default 0..10; noul>=0.5 | Laya structured output | D | 'keep it small', no evidence. Laya retired; no importer: DEAD. | PENDING (unassigned): Delete module. No test. |
| tools API bind 0.0.0.0 | mcp/tools_api.py:77 | 0.0.0.0 | bind address | A | operator 2026-09-22T20:52Z private ZeroTier binding (memory network-binding.md). | keep |
| tools API bearer auth (+mcp_ws gate) | mcp/tools_api.py:303-310; mcp/mcp_ws.py:55-76 | account key required; no accounts -> closed | the door on :1235/:1236 | A | operator 2026-09-26T16:41Z: "Key for MCP seams reasonable if we can configure it... If so key it up." Closed-without-accounts is ours (B, security). | keep |
| Origin check | mcp/tools_api.py:259-265 | present Origin must be allow-listed; default empty | 403 foreign browser origins | B | MCP spec streamable-http: 'If the Origin header is present and invalid, servers MUST respond with HTTP 403'. | keep |
| MAX_BODY / ws max_size | mcp/tools_api.py:221; mcp/mcp_ws.py:81; mcp/mcp_bridge.py:1… | 32 MiB | largest request/frame | B-value-arbitrary | Bound needed; justification is circular (each cites the other). | keep (the bound is forced); derive the value |
| DRAIN_LIMIT | mcp/tools_api.py:225 | 1 MiB | drain refused bodies up to | B-value-arbitrary | Avoid TCP reset losing the 401; value ours. | keep (the bound is forced); derive the value |
| CORS Max-Age | mcp/tools_api.py:494 | 600 s | preflight cache | B-value-arbitrary | Header value ours. | keep (the bound is forced); derive the value |
| redact / _HOST_OK | mcp/tools_api.py:228,:313-325 | strip query, bearer, ym- keys; validate Host | log/header hygiene | B | Security. | keep |
| bridge timeouts | mcp/mcp_bridge.py:84,:159-161 | 3600 s; drain deadline 3600 s, poll 0.05 | stdio bridge waits | B-value-arbitrary | Required bounds; values ours. | keep (the bound is forced); derive the value |

### Infrastructure and bench/octopus: code_search.py, worker.py, datasets.py, jobs.py, corpus.py, packages.py, dash_*, laya_*, watchdog, bench/octopus

| name | file:line | value | what it does | class | evidence or derivation | action |
|---|---|---|---|---|---|---|
| CODE_SEARCH_SEMANTIC default off | mcp/code_search.py:679 | 0 (semantic retriever off) | find_by_meaning uses lexical+symbol only; embeddings not consulted | C | scripts/eval_retrieval.py: 'semantic 77/120 against keyword 92/120, McNemar p=0.0041' (gauntlet, 120 gold queries). Caveat in code: gold queries derived from symbol names flatter lexical (655-657) | keep |
| CANDIDATES | mcp/code_search.py:85 | 40 | embedding candidate pool before rerank | D | No measurement or derivation; comment says only 'retrieve wide, rerank narrow'. Dormant while semantic is off. | PENDING (unassigned): Dormant (semantic off). If semantic returns, use the full pool or measure. No tests reference it. |
| RERANK_DOC_CHARS | mcp/code_search.py:88 | 1200 | per-doc truncation sent to cross-encoder | B-value-arbitrary | Existence: keeps query+docs inside the cross-encoder context. 1200 not derived from the reranker's ctx. Reranker 'not trusted, not used' (AGENTS.md). | REMOVED 2026-10-01 with the reranker (docs/REMOVED.md) |
| DEFAULT_TOP_K | mcp/code_search.py:89 | 5 | default snippets from find_by_meaning; stated in model-visible schema | D | No evidence. | PENDING (unassigned): Leave to caller; if a default is needed, cite a measurement. No tests reference it. |
| RERANK_MAX_K | mcp/code_search.py:123 | 2 (no rerank above k=2) | reranking skipped for top_k>2 (latency) | C | scripts/eval_rerank.py, n=11: correct in top-5 7/11 vs 7/11; ~1100 ms -> 96 ms. Comment: provisional, three.js contaminated; FINDINGS #20 voids rerank numbers. | REMOVED 2026-10-01 with the reranker (docs/REMOVED.md) |
| QUERY_INSTRUCT prefix | mcp/code_search.py:158 | 'Instruct: Given a question about a codebase, retrieve...' | query-side instruction for Qwen3-Embedding | B | Qwen3-Embedding is asymmetric: queries need an instruction prefix; observed score 0.002 without it (152-157). Task sentence wording ours. | keep |
| search_fused wide | mcp/code_search.py:647 | max(top_k*3, 10) | per-retriever depth before fusion | D | No evidence for 3x or floor 10. | PENDING (unassigned): Take all ranker hits (fusion is cheap) or measure recall vs depth. |
| search_fused output cap | mcp/code_search.py:746 | rows[:max(top_k*2, 8)] | fused rows kept before tiering | D | No evidence. | PENDING (unassigned): Return top_k fused rows, or measure. |
| symbol-only hits surfaced | mcp/code_search.py:754 | [:2] | declaration hits missed by rankers inserted first | D | No evidence for 2. | PENDING (unassigned): Surface all declaration hits (exact by construction) or cap by top_k. |
| WEAK_SIM | mcp/code_search.py:1108 | 0.40 | cosine junk floor | D (dead) | Defined, never referenced. Comment: worst genuine 0.476, best nonsense 0.445 (docs/PLAN.md:54); 0.40 picked below both. | PENDING (unassigned): Delete (unused). No tests reference it. |
| REMOTE_READ_ONLY sentence | mcp/code_search.py:885 | 'Remote and read-only: it searches indexed library source...' | appended to every code tool description (model-visible) | B | Correctness: tools search held library source, not the user's files; observed misuse (find_by_pattern 'Cargo.toml', 871-878). 'Operator report 2026-09-23': claimed, no quote found. Pinned by test_tools. | keep |
| Tool descriptions (8 code tools) | mcp/code_search.py:889-1058 | trigger-phrase descriptions | what the second brain reads to choose tools | C-at-risk | AGENTS.md: routing 10.7->11.7/14 after rewrite, but 'These numbers are at risk' (max_tokens 400, no finish_reason; CONSTRAINTS #31). Re-run before citing. | keep |
| find_references ordering (_rank_refs) | mcp/code_search.py:1130-1160 | library source first, then reference density | orders refs the model sees before truncation | B | Correctness: walk order is arbitrary; positionLocal led with examples, buried NodeMaterial.js (1139-1143). Peripheral-dir regex ours. | keep |
| find_references truncation | mcp/code_search.py:1372-1375 | 30 calls + 30 other | caps references shown to the model | D | No evidence for 30. | PENDING (unassigned): Return all with a count, or derive from a token budget. |
| find_by_pattern max_results default | mcp/code_search.py:1414 | 40 | default hit cap (schema: 'Default 40') | D | No evidence. | PENDING (unassigned): Derive from a token budget or measure. |
| read_file_range default span | mcp/code_search.py:1491 | start+120 lines | lines returned when end omitted (schema says so) | D | No evidence. | PENDING (unassigned): Require end, or derive from a token budget. |
| near-miss path suggestions | mcp/code_search.py:1511 | [:8] | same-basename files offered on a miss | D | No evidence (cap in a failure return). | PENDING (unassigned): Low impact; show all or keep with a count. |
| summarize_text max_words default | mcp/code_search.py:1561 | 300 | default summary length (in schema) | D | No evidence. | PENDING (unassigned): Require max_words or measure. |
| summarize_text answer allowance | mcp/code_search.py:1579 | max_words*1.6+64 tokens; effort=low; timeout 3600 | answer budget for summaries | B-value-arbitrary | ~1.6 tokens/word for identifiers; replaced max_tokens=900 that measured a 0-char reply (2026-09-22). 1.6, 64 and effort low not measured. | keep (the bound is forced); derive the value |
| VERIFY_CHECKS whitelist | mcp/code_search.py:1064 | lint/test/build npm rows | run_check runs labels only, never command lines | B | Security: source/model text cannot reach a shell (AGENTS.md 'Checks'). | keep |
| VERIFY_TIMEOUT | mcp/code_search.py:1069 | 900 s | run_check subprocess timeout | B-value-arbitrary | Existence: a hung check must end. 900 not derived. | keep (the bound is forced); derive the value |
| run_check output trim | mcp/code_search.py:1700-1701 | >6000 chars -> head 1500 + tail 4000 | what the model sees of a check's output | D | Rationale (errors at the end) sound; numbers not derived. | PENDING (unassigned): Derive from a token budget. No tests reference it. |
| jobs.LANES gpu | mcp/jobs.py:74 | gpu: 1 | one GPU job at a time | B | AGENTS.md 'One GPU consumer at a time' (429/502 degradation); operator_answers 2026-09-23: 'One at a time (Recommended)' for benchmark streams. | keep |
| jobs.LANES cpu/net | mcp/jobs.py:74 | cpu: 4, net: 4 | parallel CPU/network jobs | A-flag (number ours) | Operator's pasted brief 2026-09-22T20:40 says 'honouring LANES {gpu:1, cpu:4, net:4}', but the values already existed in agent-written jobs.py. Not derived from core count. | keep; operator to confirm the value/wording |
| STALE_SECONDS | mcp/jobs.py:68 | 1200 s (YAMADORI_JOB_STALE) | heartbeat age after which a running job is reclaimed | B-value-arbitrary | Existence: dead-worker detection; must exceed an observed 450 s generation (64-67). 1200 not derived; with a 60 s heartbeat, a smaller value would also work. | keep (the bound is forced); derive the value |
| max_attempts | mcp/jobs.py:153,172 | 3 | retries before a job is errored | D | No evidence. | PENDING (unassigned): Keep retries explicit per handler (Permanent vs transient). test_worker/test_assist reference max_attempts. |
| pause ttl_seconds | mcp/jobs.py:90 | 7200 s | a paused lane auto-resumes after 2 h | B-value-arbitrary | Existence: a crashed benchmark must not pause the worker forever. 2 h not derived. | keep (the bound is forced); derive the value |
| BEAT_SECONDS | mcp/worker.py:84 | 60 s | heartbeat cadence | B | Derived: 'well inside jobs.STALE_SECONDS (1200)' without hammering one sqlite file (82-83). | keep |
| POLL_SECONDS | mcp/worker.py:85 | 2 s | queue poll interval | B-value-arbitrary | Latency vs sqlite load; value not derived. No model effect. | keep (the bound is forced); derive the value |
| RECLAIM_EVERY (idle scheduler tick) | mcp/worker.py:86,1214 | 60 s | reclaim + schedules skill watch/learn, deep.learn, prices | B-value-arbitrary | Existence: jobs.py has no cron. 60 s not derived. The learners it schedules move trigger thresholds (their files are out of this scope). | keep (the bound is forced); derive the value |
| FETCH_TIMEOUT / MAX_FETCH_BYTES | mcp/worker.py:88-89 | 60 s / 25 MiB | dataset source fetch limits | B-value-arbitrary | Security/resource: bounded fetch. Values not derived. | keep (the bound is forced); derive the value |
| CHUNK_CHARS | mcp/worker.py:94 | 12000 chars | source chunk per extraction call | B-value-arbitrary | Comment: ~3k tokens leaves room to think; fixed an observed empty reply (fdc9067). Not measured. | keep (the bound is forced); derive the value |
| MAX_CHUNKS | mcp/worker.py:95 | 60 | chunks extracted per source | D | No evidence; silently drops the rest of a long source. | PENDING (unassigned): Extract all chunks (the gpu lane already serialises), or record the truncation. |
| EXTRACT_MAX_TOKENS | mcp/worker.py:96 | 4096 | answer allowance for recipe extraction | D | No evidence. | PENDING (unassigned): Use tiers.budget like other internal callers. No tests reference it. |
| MODEL_TIMEOUT | mcp/worker.py:97 | 3600 s | extraction/assist model call timeout | B-value-arbitrary | Must exceed long generations (450 s observed). 1 h not derived. | keep (the bound is forced); derive the value |
| MIN_EVIDENCE_CHARS | mcp/worker.py:99,371 | 24 | shorter quote rejects an extracted row | D | '"use a" is in every document': no measurement. The extraction prompt itself asks for 30-300 chars (317). | PENDING (unassigned): Enforce the prompt's stated floor or measure false accepts. No tests reference it. |
| EXTRACT_SYSTEM prompt | mcp/worker.py:309-322 | recipe table; evidence 30-300 chars | prompt whose rows become skills the model later sees | D | Hand-written prompt; 30-300 not derived. Dataset rows compile into skills (AGENTS.md skills pipeline). | PENDING (unassigned): Move into skill_prompts.py (versioned, pinned) and justify or drop the bounds. |
| extraction temperature | mcp/worker.py:343 | 0.2 | sampling for recipe extraction | D | No evidence. Memory temperature-decision: vendor 1.0 kept; operator_answers 2026-09-23 chose temp 1.0 over greedy for benchmarks. | PENDING (unassigned): Omit temperature (vendor sampling). No tests reference it. |
| assist temperature / ASSIST_MAX_TOKENS | mcp/worker.py:592,745 | 0.0 / 1024 | licence/field assist sampling and allowance | D | No evidence for greedy or for 1024. | PENDING (unassigned): Omit temperature; budget via tiers.budget. |
| ASSIST_HEAD/TAIL/KEYWORD_LINES, LICENCE_FILE_CHARS | mcp/worker.py:593-596 | 8000 / 4000 / 40 / 6000 | what of a source the assist model reads | D | No evidence. | PENDING (unassigned): Derive from a token budget or measure licence recall. |
| LICENCE_FETCH_BYTES / TIMEOUT | mcp/worker.py:597-598 | 512 KiB / 20 s | licence file fetch limits | B-value-arbitrary | Bounded fetch (security); values not derived. | keep (the bound is forced); derive the value |
| MIN_LICENCE_QUOTE_CHARS | mcp/worker.py:603 | 8 | shortest licence quote accepted | B | Derived: 'MIT License' is 11 chars and a complete statement (601-602). | keep |
| HOLD_RESTRICTED | mcp/worker.py:600 | on | restricted licence holds a dataset in clarify | B | Legal correctness: serving AGPL/no-redistribution rows is a decision nobody made (585-590). Tension with memory 'No review gate' (operator 2026-09-22). | keep |
| NON_ANSWERS | mcp/datasets.py:213 | {'unknown','n/a','tbd',...} | rejects non-answers in provenance fields | B | Correctness: 'unknown' in a licence field is the failure the module prevents. | keep |
| RESTRICTED licence table | mcp/datasets.py:221-238 | AGPL, NC, ND, proprietary, ... | flags licences on datasets | B | Licence terms (legal correctness). | keep |
| ASSIST_PRIORITY | mcp/datasets.py:128 | 10 | assist claimed before extract (priority 0) | B | Ordering only: any value > 0 works. | keep |
| HUMAN_STAGES / no review stage | mcp/datasets.py:106 | ('clarify',) | only clarify waits for a human | A | Memory dataset-pipeline-no-review-gate, user 2026-09-22: 'This was all handled by the agent and we could manually review it if we wanted to.' | keep |
| TEST_ACCOUNT_LABELS; traffic fails closed | mcp/corpus.py:105 | ('live-test','claude-dogfood') | test traffic excluded from learning | B | Correctness: live-suite traffic was counted as producer evidence (95-104). Attributed to 'coordinator, 2026-09-24', not operator. | keep |
| corpus truncation | mcp/corpus.py:88,198,206 | payload 20000, last_user 2000, system_head 1200 chars | what the corpus keeps for replays and learning | D | No evidence. AGENTS.md: the 2,000-char cut blocks compaction replay and causes code_edit/code_generation confusion in route evals. | PENDING (unassigned): Store full text within the 'caller code is hashed' rule, or derive the caps. |
| USAGE_FLUSH_SECONDS / _REPLACE_TRIES | mcp/accounts.py:88-89 | 60 s / 6 | usage count flush; replace retries | B-value-arbitrary | Existence: per-request writes gave WinError 5 HTTP 500s (80-87). Values not derived. | keep (the bound is forced); derive the value |
| single-user mode without accounts.json | mcp/accounts.py:165-166 | anonymous admitted | proxy open without a registry; unreadable registry refuses all | B | Security design: unreadable registry fails closed. :1234 exposure intentional (memory network-binding, operator 2026-09-22). | keep |
| MAX_PACKAGES | mcp/packages.py:42 | 3 | packages consulted per fallback search | D | 'More than a couple ... is a sign the question was not really about a dependency': no measurement. | PENDING (unassigned): Consult every imported+indexed package, or measure. No tests reference it. |
| package fallback text caps | mcp/packages.py:85,334 | 2500 / 3000 chars | truncates fallback search results the model sees | D | No evidence (same kind as MAX_FINDING_CHARS, which cut a plan and broke pagoda-h2). | DONE 2026-09-27: REMOVED the silent [:2500]/[:3000]; the tool loop's marked breaker (repeats.cap_tool_result) bounds it. |
| discover builtin module tables | mcp/discover.py:87-98 | JS/Py/Rust stdlib names | excludes stdlib imports from library-use detection | B | Correctness: language stdlib names; the per-language split fixed a dropped 'math' import (82). | keep |
| repos indexing timeouts | mcp/repos.py:108,145,190,251 | 10 / 30 / 1800 / 7200 s | git and index_code subprocess limits | B-value-arbitrary | Existence: bounded subprocesses; values not derived. | keep (the bound is forced); derive the value |
| Laya zero-shot gate | mcp/laya_service.py:206 | LAYA_ZS_GATE 0.3 | abstain threshold for zero-shot route | D (dead) | No evidence. Laya retired: operator 2026-09-24T22:24 'retire Laya for now, use E1'; watchdog no longer starts it (watchdog.ps1:107). | PENDING (unassigned): Delete with the Laya retirement (docs/E1.md s9). |
| WS_QUEUE_DEPTH | mcp/laya_service.py:276 | 64 | Laya websocket backpressure bound | B-value-arbitrary (dead) | Existence: bounded queue. Service retired. | keep (the bound is forced); derive the value |
| laya_router THRESHOLDS | mcp/laya_router.py:82-88 | 0.25-0.50 per feature | binary-feature routing thresholds | C (dead) | Measured on 14 labelled requests, 43% -> 86% (docstring 9-24); n=14. Module has no importers. | keep |
| dialectic/session margin floor | mcp/dialectic.py:105; mcp/session.py:94 | 0.15 | 'decided' margin for Laya choices | D (dead) | No evidence; neither module has importers. | PENDING (unassigned): Delete (no importers). |
| vitals MAIN_FREE_TARGET_MIB | mcp/vitals.py:56 | 600 MiB | dashboard 'tight' floor for the main card | A-flag (display only) | Comment claims operator 2026-09-25 'the dashboard should be aware of live membudgets': no quote found. 600 = config -c sizing target. Display only. | keep; operator to confirm the value/wording |
| vitals SLOTS_TIMEOUT / TOOL_WINDOW | mcp/vitals.py:288-290 | 1.0 s / 600 s | dashboard pulse timeout/window | B-value-arbitrary (display) | Display only; no stack behaviour. | keep (the bound is forced); derive the value |
| power sampler constants | mcp/power.py:102-111 | 1 s sample, 10 s gap, 6 h keep, 60 s flush, 400 d | energy accounting (x_yamadori.energy) | B-value-arbitrary (accounting) | Operator 2026-09-23T14:40 'calculate the watt draw at my average electricity rate'; cadences ours. No model effect. | keep (the bound is forced); derive the value |
| power MEASURED_GENERATING | mcp/power.py:94 | 138.5 W | estimate where no sample exists | C (weak) | 594 one-second samples, n=1 run, script not in repo (91-93). | keep |
| prices refresh | mcp/prices.py:63-68 | daily; keep 30; timeout 60 s; MIN_MODELS 50 | OpenRouter snapshot job scheduled by the worker | A-flag (number ours) | Operator 2026-09-24T14:42: 'cost savings vs how much those tokens would have cost us through openrouter'. Cadence and sanity floor ours. | keep; operator to confirm the value/wording |
| watchdog interval | scripts/install-autostart.ps1:73 | every 5 min | liveness check cadence | B-value-arbitrary | Existence: operator 2026-09-24T15:06 'needs a watchdog to ensure it is running'. 5 min not derived. | keep (the bound is forced); derive the value |
| two-strike restart | scripts/watchdog.ps1:244-252 | 2 consecutive failures; strike mark valid 12 min | restart only if also unhealthy on the previous run | B | From a measured failure: timer restarts cost a LiveCodeBench run (~60% rows errored) (header 20-30). 12 min = two 5-min intervals + slack; 2 is the minimum that tolerates one slow probe. | keep |
| slot-progress check | scripts/watchdog.ps1:229-236 | 2 reads, 10 s apart | no llama-swap restart while tokens are moving | B | A 120 s probe killed 450 s generations; verified live under a 20k-token run (header 45-51). 10 s not derived. | keep |
| RestartCooldownMin | scripts/watchdog.ps1:63 | 10 min | per-service restart cooldown | B-value-arbitrary | Existence: prevents restart thrash. 10 not derived. | keep (the bound is forced); derive the value |
| TimeoutSec | scripts/watchdog.ps1:64 | 15 s | health probe timeout; only a timeout counts as down | B-value-arbitrary | Existence: liveness only ('Only a TIMEOUT counts as down', 154). 15 s not derived. | keep (the bound is forced); derive the value |
| on-demand model unload rule | scripts/watchdog.ps1:181 | ttl>0 -> unload that model, never restart the stack | wedged on-demand model handled alone | B | 2026-09-23: a vision wedge restarted the stack and killed a 1,429 s row (171-180). | keep |
| autostart task restart policy | scripts/install-autostart.ps1:46,77 | 3 restarts / 1 min; watchdog run cap 10 min | Scheduled Task retry and watchdog execution limit | B-value-arbitrary | Windows task settings; values not derived. | keep (the bound is forced); derive the value |
| watchdog proxy env YAMADORI_E1=1 | scripts/watchdog.ps1:102 | 1 | E1 replaces Laya on the request path | A | Operator 2026-09-24T22:24: 'retire Laya for now, use E1 and let's collect more data and self tune.' AGENTS.md still says off by default: stale. | keep |
| watchdog proxy env IMAGEGEN_DEFAULT=turbo | scripts/watchdog.ps1:98 | turbo | default image model | A | Operator 2026-09-24T20:41: 'I want turbo to be the default'. | keep |
| Hermes --reasoning effort (arm) | bench/octopus/run.py:448,635 | xhigh (or low) | reasoning_effort Hermes sends; selects the proxy tier | A | Operator 2026-09-27T02:20: 'our xHigh thinking is all I care to test anymore. Until it actually works.' | keep |
| --max-turns | bench/octopus/run.py:638 | 1000 | Hermes tool iterations per turn | D | Comment: 'the last V0 used 16 in 68 min'. Hermes' documented default is 500 (hermes_cli/_parser.py:274). Nothing derives 1000. | DONE 2026-09-27: REMOVED the 1000 default: Hermes' own default applies unless --max-turns is given. |
| --run-budget | bench/octopus/run.py:641 | 25200 s (7 h) | Hermes wall-clock budget; at 80% Hermes injects a wrap-up notice | B-value-arbitrary | 'coordinator, 2026-09-24' (not operator): the author's V0 took 5 h on a 3060. Hermes help: 'At 80% elapsed the agent gets a one-time wrap-up notice' (model-visible at 5.6 h). | keep (the bound is forced); derive the value |
| Hermes process kill | bench/octopus/run.py:468 | budget + 900 s | runner kills Hermes 15 min past its budget | B-value-arbitrary | Safety net past Hermes' own budget; 900 not derived. | keep (the bound is forced); derive the value |
| --ignore-rules | bench/octopus/run.py:450 | on | Hermes skips AGENTS.md/SOUL.md/memory/preloaded skills | D | No rationale anywhere. Hermes help: 'Skip auto-injection of AGENTS.md, SOUL.md, .cursorrules, memory, and preloaded skills'. Changes what the model sees vs a normal user. | PENDING (unassigned): Write a justification (hygiene: no stray rules or memory) or drop it; the operator's goal is daily-work realism. |
| compression.enabled true | bench/octopus/make_profile.py:106 | false -> true | Hermes compacts the conversation | A | Operator 2026-09-24T13:35 'I want compaction, with cach[e]...'; 02:22 'I want more context'. The 'SEPARATE profile' (2026-09-24) is claimed; no quote found. | keep |
| terminal docker backend + mounts | bench/octopus/make_profile.py:107-116 | docker; only the run folder at /workspace; not persistent | the model's shell sandbox | B | Security: commands are model-chosen; no host fallback, never --yolo (run.py header 20-33). | keep |
| docker_extra_args --network none / sandbox network | bench/octopus/make_profile.py:115 | --network none; run.py uses the internal network | model container network isolation | B | Security: SELF-IMPROVEMENT-LOG #48 probe reached host services through host.docker.internal. | keep |
| egress gate ALLOWED_PORTS | bench/sandbox/egress_gate.py:49 | (80, 443) | ports the model's sandbox may reach | B-value-arbitrary | Labelled 'A CHOICE (2026-09-26)': npm/pip/git/CDNs need 80/443. The global-only address rule is security (B). | keep (the bound is forced); derive the value |
| terminal.lifetime_seconds | bench/octopus/make_profile.py:117; run.py:439 | 300 -> 21600 | idle reaper for the terminal container | B-value-arbitrary | Existence: the model thinks for minutes between calls and the reaper must not recycle. 6 h not derived (run budget is 7 h). | keep (the bound is forced); derive the value |
| creation_nudge_interval | bench/octopus/make_profile.py:121 | N -> 0 | no 'save a skill' reminders to the model | B | Benchmark hygiene: Hermes' nudge text enters the conversation and writes state across runs. No operator quote. | keep |
| auxiliary.background_review.enabled | bench/octopus/make_profile.py:150-153 | false | no post-turn skill/memory review | B | Observed: 601 s of GPU after v0f prompt 1, writing into the profile, so state carries across runs. Operator 2026-09-27T02:27 'fix all the issues we have identified' (general). | keep |
| compression.no_progress_timeout (responses wire) | bench/octopus/make_profile.py:155 | 300 s | Hermes side-call stall timeout during compaction | B-value-arbitrary | Labelled 'a CHOICE (docs/HARNESS-RESPONSES.md)'. Existence: Hermes' 60 s default kills a thinking compaction. 300 not derived. | keep (the bound is forced); derive the value |
| default arm = browser toolset | bench/octopus/toolset_arms.py:139 | browser (sandbox host) | Hermes offers browser_* tools to the model | A | Memory harness-tools-direction: operator 2026-09-26 'All of our harnesses should have usable browser tools attached'. Quote in memory only; not in operator_messages.txt. | keep |
| browser.inactivity_timeout | bench/octopus/toolset_arms.py:172 | 120 -> 21600 | browser session idle close | B-value-arbitrary | Existence: the model thinks for minutes. 6 h not derived. | keep (the bound is forced); derive the value |
| browser-cdp toolset off | bench/octopus/toolset_arms.py:169 | disabled | hides browser_cdp/browser_dialog tools | D | Stated reason: keep the sandbox and host arms equal (15 tools). That is A/B parity; operator 2026-09-27 rejected on/off A/B (memory no-onoff-ab). | PENDING (unassigned): Decide on the tools' own merit; if kept off, record why. |
| allow_private_urls / restrict_evaluate | bench/octopus/toolset_arms.py:175,180 | true / true | browser may open the localhost game; no fetch/XHR in evaluated JS | B | Correctness: the game is on localhost:3001, refused as private otherwise. Security: restrict_evaluate limits evaluated JS to GETs. | keep |
| tool_search off | bench/octopus/toolset_arms.py:191 | enabled: off | tools offered directly, no search bridges | A-flag (choice ours) | docs/HARNESSES.md 'Default loadout' (1,741 chars saved); operator asked for 'a generally useful normal tool loadout' (memory). The specific choice is ours. | keep; operator to confirm the value/wording |
| vault sources off | bench/octopus/toolset_arms.py:197 | onepassword/bitwarden disabled | password managers unreachable from model tools | B | Security: keep credentials out of model reach. | keep |
| run-skills create_dir | bench/octopus/toolset_arms.py:207 | run-skills (emptied per run) | skills the model saves do not carry to later runs | B | Benchmark correctness: skill_manage cannot be disabled alone, and state must not leak between runs. | keep |
| pinned type checkers volume | bench/octopus/toolset_arms.py:142-146 | tsc 5.9.3 + pyright 1.1.414 | type checkers mounted in the model's sandbox | A | Operator 2026-09-27T02:34: 'LSP should probably be in the default loadout for everyone.' | keep |
| hermes skill: type-check | bench/octopus/hermes_skills/type-check/SKILL.md | command table + 'Fix the first error, then run the check again' | installed in Hermes' skills folder (model-visible) | A-flag (text ours) | Operator 2026-09-27 LSP quote; 2026-09-26 'harness specific skills ... job of the harness' (L2760). The 'fix the first error' advice is ours, unmeasured. | keep; operator to confirm the value/wording |
| hermes skill: look-at-a-screenshot | bench/octopus/hermes_skills/look-at-a-screenshot/SKILL.md | shared path + 'ask one concrete question' | tells the model where vision_analyze can read files | B (path) / D (advice) | The path is a real constraint (only the mounted cache folder is readable; #46 image guard). 'Ask one concrete question' is ours, no evidence. | keep |
| variants V1-V3 stack text | bench/octopus/variants.py:167-297,306-364 | pins, project structure, R3F/flatland setup lines | rewrites the task prompt the model gets | A-flag (text ours) | Operator: Octopus is 'a specific prompt I asked for' (2026-09-24T22:41). The stacks and wording are ours; the V4 docstring calls this style 'wrote the model's architecture for it'. | keep; operator to confirm the value/wording |
| variant V4 tech line | bench/octopus/variants.py:299 | the operator's sentence | the only change to the original prompt | A | Operator 2026-09-27 (quoted at variants.py:61-63): 'build ... with r3f (react-three-fiber) v10 and Koota and pmndrs math'. | keep |
| --iterative graded follow-ups | bench/octopus/run.py:570-583,717; followup.py:44-45 | refused unless --allow-graded-followups; caps 12 items / 6000 chars | grader output never enters the conversation by default | A | Operator 2026-09-26 L2800: 'that is cheating and not valuable for us to do at all.' The followup.py caps are D but unused. | keep |
| relay upstream timeout | bench/octopus/relay.py:313 | 7200 s | a request longer than 2 h fails at the relay | B-value-arbitrary | Pass-through recorder; must exceed the longest generation (1,429 s observed). 2 h not derived. | keep (the bound is forced); derive the value |
| preflight slot-idle check | bench/octopus/run.py:190-192 | 2 reads 5 s apart | run refuses to start unless the GPU is idle | B | AGENTS.md 'One GPU consumer at a time'. GAP: run.py never calls jobs.pause, so worker gpu-lane jobs can start mid-run. | keep |
