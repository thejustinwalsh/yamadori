# Overthinking in the Octopus runs: where the hours go after the answer

Research note, 2026-09-26. Offline and read-only: the relay rows, Hermes' stream and session exports, the
corpus (`deep_decisions`) and the token ledger. No request went to `:1234` or `:11434`. The only thing run was
`bench/octopus/grade.py`, in Docker, on project snapshots rebuilt from the transcripts. Every run below is
**n=1**. The numbers describe these runs. They are not rates, and PROTOCOL rule 10 applies to every remedy.
V0 is vanilla JS and canvas, so three.js contamination does not arise here.

> **Note added 2026-09-26 (after the operator's decision that day).** Every prompt-2 result below, and every
> trap measured on prompt 2 (v0e p2 in T1-T5 and T8, the "eight located defects", the 5.43 h with no product
> change), was measured under **graded follow-ups**: prompt 2 was built by `bench/octopus/followup.py` from
> the grader's failed checks. That leaks the answer key, and some of its items were **false** -- grader
> defects, not game defects: v0e's prompt 2 carried the case-sensitive `particles_draw_ctx` failure and the
> 0/1-only `pixel_grids_ge_4` failure (items 3 and 5 of 8; SELF-IMPROVEMENT-LOG #57), and runtime checks that
> flipped on unchanged code could send non-defects too (#62). The operator has ruled graded follow-ups out:
> the grader is a measurement instrument only, prompt 1 (single-shot, no feedback) is the headline, `run.py`
> refuses `--iterative` without `--allow-graded-followups`, and prompt-n >= 2 grades are labelled
> `assisted: graded follow-up`. Read the prompt-2 figures as behaviour under a partly false, answer-key
> prompt, not as the protocol's results. The prompt-1 figures (v0b, v0e p1) are unaffected by this.

The operator's observation (2026-09-26): runs "balloon into 8hr runs after the question was answered in about
3hrs". The data supports it, and it is worse than that in one case:

- **v0b-V0-xhigh-1** reached its final grade (18/27) **1 h 09 min** into a 7.25 h run. Its last change to a
  product file came at 1 h 51 min. For the remaining 5 h 24 min it wrote only test harnesses, and the grade
  never moved.
- **v0e-V0-xhigh-1 prompt 1** reached its final grade (19/27) at **3 h 50 min** of 3.97 h. The ballooning in
  this run happened *before* the answer. The JavaScript already graded 16/27 at 1 h 27 min. Then 2 h 24 min went
  into finding one load-time error, `state is not defined`, and two neighbours. A browser shows that error the
  moment the page opens, but the run had no page yet and no browser.
- **v0e-V0-xhigh-1 prompt 2** was handed eight located defects (`PLAYER.hit is not a function`, among others).
  It ran 5.43 h and **did not change a single product file**. The grade was the same before and after.
  From 01:03 onward its own reasoning stated the root cause of the first defect eight times, and it never
  applied the fix.

Across the two runs, 11.5 h of wall clock produced no grade change (v0b 6.1 h after the answer, v0e prompt 2
5.4 h). A further 2.4 h in v0e prompt 1 was spent hunting errors that a page check would have shown at once.

---

## Method

**Tools.** The analysis is `bench/octopus/overthinking/` (the scripts and their order are in its `common.py`).
Its outputs go to `out/`, which is gitignored.

1. **Per request** (`req.py`, `timeline.py`). Built from the relay's POST rows: wall time, `usage`,
   `reasoning_chars` and `reasoning_tokens` (v0e reports them. v0b predates that, so its tokens are chars / 3.88,
   the ratio measured on v0e p1: 601,362 chars over 154,970 tokens), `x_yamadori` (route, deep, investigate,
   tool_code/fixup, compaction, cache `model_ms`, budget).
2. **What each step did** (`classify.py`, `table.py`). Each Hermes `tool_use` is assigned to the request that
   produced it, then classed as one of: product edit, harness build (a write outside the product: `/tmp/*`,
   dot-files, `test_full.js`), harness run, product read, search, inspect/verify (`ls`, `node --check`,
   `curl`, `sed -n`), browser environment (`playwright`, `apt-get`, `base64`, screenshots) or vision. A
   request's time goes to its first class in that precedence. Deep thinking's `investigate.seconds` and a
   fix-up's time outside `model_ms` are split out.
3. **Effectively answered** (`recon.py`, `snaps.py`). Each run's project is rebuilt from its `write_file` and
   `patch` calls. When a patch's `old_string` no longer matches exactly, the diff Hermes returned is applied
   instead. **Validation:** the rebuilt final state matches the real files byte for byte, 15/15 files for v0b
   (`root_mirror`) and 11/11 for v0e after both prompts. The v0b state after 00:27:37 also equals the
   `v0b-snap-0048` fixture. Snapshots were graded with `grade.py`, runtime included, 11 grades. **Definition:**
   the earliest snapshot whose grade is within 1 of the run's final grade.
   v0e p1 wrote `index.html` only at 22:10, so its earlier snapshots are **counterfactuals**: the JS as it was
   at time T, plus the `index.html`, `css/styles.css` and `README.md` the model wrote at the end. They are
   labelled `cf`.

| snapshot | grade | page errors |
|---|---|---|
| v0b after 00:11:20 (first complete product, +1 h 09 min) | **18** | none |
| v0b 00:13:52 / 00:17:50 / 00:22:27 | 18 / 18 / 18 | none |
| v0b 00:27:37 (`v0b-snap-0048`, graded 2026-09-25) / final | 18 / 18 | none |
| v0e p1 cf 19:58:51 (game.js written, +1 h 27 min) | 16 | `state is not defined` x2 |
| v0e p1 cf 20:11:24 / 20:43:52 / 21:40:30 | 16 / 16 / 16 | `state is not defined` x2 |
| v0e p1 cf 22:07:06 (the free `state` fixed) | 17 | none |
| v0e p1 actual 22:16:26 (index.html, no css yet) / 22:17:08 | 14 / 16 | 404 on css / none |
| v0e p1 final (after game.js patches 22:17:45 to 22:22:31, graded `@p1`) | **19** | `PLAYER.hit is not a function` at 33 s |
| v0e p2 final (no product change) | 19 | the same, at 25.6 s |

---

## 1. The runs

| | v0b p1 | v0e p1 | v0e p2 |
|---|---|---|---|
| wall | 7.25 h (budget, then an orphan, #39) | 3.97 h (finished) | 5.43 h (429 x3, exit 1, #44) |
| requests | 91 | 177 | 141 (137 x 200, 3 x 429, 1 x 502) |
| thinking caps | none (pre-2026-09-25) | `agent_step` 6,144, nudge at 0.6 | the same |
| main completion tokens | 156,196 | 258,132 | 137,445 |
| of which reasoning | ~95,900 | 154,970 | ~96,400 |
| tool-call arguments (approx.) | ~50,000 | ~78,600 | ~20,000 |
| deep thinking | kickoff 157 s + struggle 756 / 1,267 / 2,658 s, plus the orphaned fourth (#39) | kickoff 235 s | struggle x6: 6,536 s (1.8 h) |
| second-brain tokens | not separable per run | not separable | at most 91,750 (the ledger's 09-26 `second_brain`, which also holds daytime tests) |
| compactions | 1 | 5 | 3 |
| product files changed | last at 00:52:55 (+1 h 51 min) | through 22:25 | **none** |
| final grade | 18/27 | 19/27 | 19/27 |
| answered at | **+1 h 09 min (16% of wall)** | +3 h 50 min | **before it started** |

Decode ran at a median 10.8 tok/s in v0b (deep contexts, no compaction for 4 h), 18.0 in v0e p1 and 15.6 in v0e
p2. Tool execution time is negligible: every gap between requests is 2 s or less, except v0e p2's 2,311 s
stale stream.

### Where the time went after the answer

**v0b, after 00:11:21** (6.20 h of request time; the requests overlap at the end):

| class | min | main tokens (reasoning) |
|---|---|---|
| orphaned deep thinking and the main request beside it (rows 90-91, #39) | 70.3 | 9,884 reasoning streamed, no answer |
| deep thinking (runs at 00:27 and 01:19) | 65.4 | second brain |
| search (includes row 68: 2,903 s and 20,203 reasoning tokens to choose one `tool_search`) | 64.3 | 33,456 (27,005) |
| building the test harness (`.runtime_test.js` x7, `.dbg*.js`, `test_full.js`) | 45.2 | 26,143 (9,706) |
| fix-up on harness files (`test_full.js` 2,241 s, rejected; `.runtime_test.js` 191 s) | 40.5 | second brain |
| re-reading product files | 30.5 | 13,261 (11,473) |
| running the harness | 18.6 | 9,087 (7,206) |
| inspect/verify | 14.8 | 5,587 |
| product edits (4 patches, none moved the grade) | 8.7 | 5,881 |
| compaction | 7.8 | 4,299 |

**v0e p2, the whole prompt** (4.78 h in requests, plus a 38.5 min stale gap):

| class | min | main tokens (reasoning) |
|---|---|---|
| deep thinking, 6 struggle runs | 108.9 | second brain |
| browser environment: `npm i playwright`, `apt-get` for Chromium libraries, `base64` of screenshots (67 requests) | 58.8 | 49,913 (35,980) |
| product reads (20 of 31 re-read an unchanged file) | 41.1 | 41,586 (37,320) |
| a silent tool-call stream, stale-killed (#44) | 21.0 | none recorded |
| compaction x3 | 16.8 | 15,065 |
| search | 11.8 | 10,054 |
| `/tmp/pw/test-game.js`, `/tmp/diag*.js` | 9.6 | 8,240 |
| `vision_analyze`, 13 calls, all failed (#45, #46) | 6.5 | 6,531 |
| 429 / 502 | 4.8 | none |
| product edits | **0** | none |

**v0e p1, from JS-complete (16/27, 19:58) to 19/27 (22:22)** (2.38 h in requests): inspect/verify 36.1 min,
product edits 30.2, product reads 28.1, compaction x4 15.1, harness build 12.2 and run 9.8 (`/tmp/harness.js`, a
Node `vm` sandbox with a fake canvas, then debugging the harness itself: "`vm.contextify` was a mistake"),
search 6.4, fix-ups 2.8.

---

## 2. The traps

Ranked by the time they cost in these runs. Every signal named is something the proxy already sees: the
client's tool calls and their arguments (`tool_code` parses them), the tool results in the next request,
`x_yamadori`, and reasoning token counts.

### T1. Diagnosed, never applied (verify before fix)

**Pattern.** The model has a located defect, often with the root cause written out in its own reasoning. It
does not apply the fix. It reads the files again, or tries to reproduce the defect through a tool it does not
have, then loses the diagnosis to a compaction or a deep-thinking run and diagnoses it again.

**Evidence.** v0e p2 opened with eight located defects. The first is an exact error string, and `grep hit
js/player.js` answers it. Reasoning excerpts from `session.p2.jsonl`:
- 01:03 (deep-thinking hand-off): "no `hit()` export"
- 01:10: "My test was too short to reproduce the real bug ... Let me extend the test."
- 01:14: "Fix: add a `hit()` method to PLAYER export", then "Let me run the existing test script first to see
  the ACTUAL current state."
- 02:30: "I'll present it directly now and skip the extra run ... Let me apply the fixes now", then "let me read
  the init region", and the next call was `vision_analyze`.

The root cause is stated in 8 assistant turns between 01:03 and 02:30. The earlier turns were compacted out of
the export, so this is a lower bound. **Zero product writes in 141 requests.** The same shape appears in v0e
p1: at 20:10 the model wrote "missing `PLAYER.takeDamage`/`PLAYER.hit`", then added `takeDamage` only. That
`PLAYER.hit` is the page error in both final grades.

**Frequency.** 1 of 1 follow-up prompts; partially in 1 of 2 first prompts.

**Cost.** v0e p2, all of it: 5.43 h, 137k main tokens and up to 92k second-brain tokens, with no grade change.

**Live signal.** The current user turn carries located defects (an error string, a named function or file, a
`path:line`), and N consecutive agent steps write nothing under the project's own paths. Writes to `/tmp`,
dot-files and new test files do not count as progress. From 23:03 on, p2 also sent calls to `apt-get` and
`playwright` and `base64` of screenshots, none of which touch the project.

### T2. Verifying through a stand-in instead of the page

**Pattern.** With no way to open the page, the model builds one: a Node `vm` harness with a fake window and
canvas (v0b, v0e p1), or a headless Chromium it installs into the sandbox (v0e p2). Then it debugs the stand-in.
The stand-in differs from the page in the one way that matters. v0b's harness "runs init() and drives many
frames" itself, so it passed while the page stayed blank because nothing on the page called `Game.init()`
(#35, #40). v0e p1's harness took 44 min to agree with itself ("the harness is missing browser globals that
game.js legitimately uses").

**Evidence and cost.**
- v0b after the answer: harness build 45.2 min, harness run 18.6 min, fix-ups on harness files 40.5 min.
  **104 min**, plus the 5 h search that the harness misdirected.
- v0e p1: `state is not defined` sat in every snapshot from 19:58 to 21:40 (the counterfactual grades above).
  It was fixed at 22:07, after 2 h 08 min. A page check reports it at load.
- v0e p2: 58.8 min of browser environment, 9.6 min of harness scripts and 6.5 min of `vision_analyze`
  (**75 min**). Beyond that, the six deep-thinking runs (T3) fired on this environment's errors.

**Live signal.** Write or patch calls whose path is outside the project (`/tmp/*`, `.runtime_test.js`,
`.dbg*.js`, `test_full.js`, `harness.js`). Terminal commands naming `vm.createContext`, `node -e`,
`playwright`, `chromium` or `apt-get`. `toolset_arms.py measure` already counts 37 such terminal calls in v0e.

### T3. Deep thinking that changes nothing, and is labelled "helped"

**Pattern.** A struggle run fires on errors that the task's code cannot cause (#45), or it lands a hypothesis
that main then chases. Afterwards the product either does not change or changes without effect. The outcome
label calls the run `helped`.

**Evidence.**
- v0b run 2 (00:27, 1,267 s): its hand-off put a `dt` scope bug in main's reasoning ("The second model flagged
  a scope bug -- let me verify", #40). Main added `currentDt` and fixed `enemiesList`, with no grade change.
  It then spent 5 h on `dt` / `currentDt` / `dt || 0` (search, harnesses, deep run 3 at 01:19 for 2,658 s)
  and never looked at what the page runs on load.
- v0e p2: six runs, 6,536 s, and **0 product writes after any of them**. `deep_decisions` labels 5 of these
  6 `helped`, and all 4 of v0b's runs `helped`: 10 of 11 client runs in these two runs. `helped` was
  automatic at the time (#45 (e)).
- The seed word derails the second brain. In 4 of the 6 p2 runs the reasoning decodes the "inspiration word"
  as a clue: "'irres' is an anagram", "the inspiration word in these benchmarks is usually an anagram of a key
  word from the answer", "'encab' -- probably encode/cab... encoding, base64 encoding?". The encab run's hand-off
  preceded the base64 data-URL attempts (#46). That link is inferred, not shown.

**Cost.** v0b 78 min (3 struggle runs), v0e p2 109 min, plus the misdirection that followed.

**Live signal.** Whether a product file changes within K requests after a run, and whether the same pattern
recurs. The proxy sees both. `deep._recurrence` (#45 (e), built) covers recurrence. Product change is not
yet part of the label.

### T4. Deliberating to the nudge after a read

**Pattern.** After a `read_file` or inspect result, the model reasons until the 60% nudge (~3,700 tokens) and
then asks to read something else. With the caps on (v0e), no step hits the hard stop. The nudge has become the
effective limit.

**Evidence.**
- v0e p1: 25 of 170 agent steps reached the nudge (at least 3,686 reasoning tokens; max 4,186). They took
  **99.7 min, 46% of the prompt's agent-step time**. 17 of the 25 chose a read, inspect or harness call next,
  not an edit. After a product read, 12 of 32 steps reached the nudge and 7 of 32 then edited.
- v0e p2: after a product read the median was 2,864 reasoning tokens, and 0 of 15 edited.
- Typical content: the p2 reasoning at 02:25 to 02:33 re-derives the same eight-bug mapping three times.
- Uncapped (v0b), two steps took 60 min: row 12, a `read_file` after 11,036 reasoning tokens (684 s), and row
  68, one `tool_search` after 20,203 tokens (2,903 s). The 2026-09-25 caps removed that tail (v0e max 4,718).

**Cost.** Reasoning above 2,048 tokens on steps that answer a read/search/inspect result: v0e p1 32,632 tokens,
about **30 min** at 18 tok/s. v0e p2 18,944 tokens, about **20 min**.

**Live signal.** The request ends on a tool result from `read_file`, `search_files` or an inspect-type
terminal command (the proxy sees the tool name and the arguments that produced it), and
`completion_tokens_details.reasoning_tokens` is known after every step.

### T5. Compaction churn fed by re-reading unchanged files

**Pattern.** The model re-reads whole files it has already read (game.js is ~34k characters). The context
refills, Hermes compacts at 88-99k prompt tokens, and after the compaction the model re-orients by reading the
same files again.

**Evidence.**
- v0e p1: 5 compactions in 3.97 h (19:16, 19:58, 20:17, 20:44, 21:54). They were 19 to 70 min apart and took
  16.4 min of requests. 18 of 38 product reads returned a file unchanged since the model last read it, 13 of
  them the same range and 8 across a compaction, 79k result characters in all.
- v0e p2: the first compaction came **13 min** into the prompt. Its first 3 requests read 6 files, taking the
  prompt from 77.5k to 99k. 20 of 31 product reads were unchanged re-reads (88k characters). No product edit followed
  that compaction or either of the two after it.

**Cost.** Compaction requests: 16.4 min (p1) and 16.8 min (p2). Requests that issued unchanged re-reads: 27 min
(p1) and 22 min (p2). Each re-read also shortens the time to the next compaction.

**Live signal.** A `read_file` result byte-identical to an earlier result in the conversation, with no write
or patch to that path in between. The proxy sees both the calls and the results.

### T6. The entry point comes last

v0e p1 wrote the 8 JS modules first and `index.html` at 22:10, **2 h 12 min after game.js existed** (the kickoff
plan's ORDER, 16 steps, is not recorded verbatim; the prompt lists `index.html` first). No page could be
opened during the whole debugging stretch, which is what made T2 necessary. v0b wrote `index.html` first but
never wired `Game.init()` (#35). The two runs fail in opposite directions on the same thing: the load path.

**Live signal.** Product JS files written, but no write of the HTML entry file named in the task's structure.

### T7. Fix-up on throwaway files

The repair pass ran on harness files: v0b row 89, `test_full.js`, **2,241 s** in one round, then rejected
("no closed fenced block"). Also v0b `.runtime_test.js` 191 s and v0e p1 `/tmp/harness.js` x2, 165 s. **41 min
in v0b** repaired nothing the grader reads. **Signal:** the path, which `tool_code` already has.

### T8. Lanes held by the dead (known, fixed offline)

- v0b's end: 70 min (#39). The orphaned deep-thinking run, then the retry beside it, then idle to the budget.
- v0e p2's end: a tool call streamed silently for 21 min until Hermes killed it, then a 502, then at 03:17 a
  second silent call. The run's last 42 min were three 429s (#44).

Both are built and **not deployed**.

### T9. Rewriting working code while building (minor, weak evidence)

v0e p1, 18:44 to 19:36: 8 requests to rewrite the comet and planet drawing in a background.js that already
parsed ("rewrite it cleanly", "cleaner and faster"). Then 18 patches to enemies.js and two grid-generator
scripts in 38 min. Some of the fixes were real (a recursive `spawnBoss`). The grids that the regeneration was
about still fail the grader's `pixel_grids_ge_4`. It cost roughly 50 min, but how much of it was waste is not
separable, so this one is not ranked.

---

## 3. Remedies, ranked by expected time saved

All of these use our own means. There are no harness patches. The one wording change that names a behaviour
names a situation, not a ban. Every saving below is an **upper bound read off one run each**. **Measure each
one with paired V0 xhigh runs, control vs arm, n of at least 2 per arm** (PROTOCOL 4 and 6). At n=2, only an
effect of hours is detectable, which is the size claimed here. **Metric:** wall time to the final grade, with
`snaps.py` finding the answer point on each run, plus the post-answer time. Grade must hold within 1 point.
Each remedy also names its own counter.

| # | remedy | trap | means | expected saving (these runs) | confirm by |
|---|---|---|---|---|---|
| 1 | **Deploy what is built**: #45 struggle rule (environment classes, one signature, product-agnostic `helped`), #44 heartbeat and supersede, #39 cancel | T3, T8 | trigger, ours | v0e p2: 6,536 s of deep thinking (replay: the new rule fires nowhere) + 42 min tail. v0b: 70 min tail. **About 3.7 h** | replay already done (#45); live: `deep` runs per prompt, `environment` counts, 429 count, time from the last 200 to exit |
| 2 | **A progress situation on the tool result.** Once K agent steps (a CHOICE, start 8) pass with no write to a project path, append one line to the tool result's injection (the ledger's tool-result slot, like library use): "No project file has changed in the last K steps (M min); the last change was <path> at <hh:mm>." It is a fact, with no instruction, and it is recorded once and replayed. Pair with the armed skill **fix-located-defect-first** | T1, T2 | tool-return situation + skill | v0e p2: up to 5.4 h, if the edits then happen. The root cause was in the reasoning by 01:03 at the latest | a paired **follow-up prompt** (the v0e p2 prompt on the v0e p1 state): time to first product write, product writes per hour, grade. Check first that `skill_select` picks the skill on that user turn: not replayed here, because it needs the embedder |
| 3 | **Open the page the way the user does**: the `--tools browser` sandbox arm (#47, built), with the skill `browser-app-entry-point` (armed) | T2, T6 | harness config (ours) + skill | v0e p1: most of the 2.4 h hunt (`state is not defined` at load). v0b: the #35 blank page is visible at once | `toolset_arms.py measure`: `first_open_s`, `errors_seen_by_model`; snapshots: time from JS complete to final grade |
| 4 | **Plan ORDER opens with a runnable skeleton**: one row in `shomen.PLAN_SYSTEM`'s table: "a browser app \| ORDER \| the entry file and the call that starts the app, before the modules it loads". It is a routing row, not a prohibition | T6 | plan prompt | v0e p1: the 2 h 12 min with no page | the ORDER of `x_yamadori.investigate.handoff.plan`; minutes from start to the first gradeable snapshot |
| 5 | **Think by what the step answers.** After a `read_file`, `search_files` or inspect-type result, `agent_step` thinking is capped at 2,048 (nudge at ~1,230). After an error result or a failing check it keeps 6,144. The nudge text for agent steps names the action: "The user is waiting. Let me take the next step I've already worked out." (UNMEASURED WORDING) | T4 | thinking caps / nudge wording | about 30 min (v0e p1), 20 min (p2) | per-step reasoning tokens by previous tool; steps from a read to the next edit; grade held |
| 6 | **Unchanged-read situation.** When a `read_file` result equals an earlier one with no write between them, the tool result's injection gets "Unchanged since your read at step N (hh:mm)." After a compaction, the work log that is re-injected (#38) lists the files read and their exports | T5 | tool-return situation, work log | 22-27 min of re-read steps per prompt, plus fewer compactions (5 in 4 h) | unchanged re-reads per prompt (`rereads.py`), compactions per hour, prompt tokens at compaction |
| 7 | **Label deep thinking by product change.** `helped` needs a write to a project path within `HELPED_WINDOW` and no recurrence; a run followed by no product write is `no_effect`. For the `investigate` job, change the seed line to "Inspiration word: X -- for variety only; it is not a clue about the task" | T3 | deep labels, seed wording | indirect: stops the learner from reinforcing runs like p2's six. Seed decoding in 4 of 6 runs | label distribution vs product writes; the share of second-brain runs whose reasoning decodes the seed |
| 8 | **Fix-up scope.** Note only, no repair round, for files outside the project (`/tmp/*`) and for scratch files the project does not load (not referenced from the entry HTML or any product import) | T7 | check | v0b: 41 min | `x_yamadori.tool_code.fixup` seconds by target class |

Not proposed: a hard cap on wall time or on tool turns. PROTOCOL 15 applies: a budget does not fix a loop.
Each trap above has a situation the model could be told, or a trigger that is currently wrong.

---

## 4. What this does not show

- n=1 per run. v0b ran without thinking caps, v0e with them, on a different day and build. The two are not a
  paired comparison of the caps.
- Prompt 2 (v0e p2) ran under a graded follow-up, some of whose items were false (note at the top; #57,
  #62). Its traps are real behaviour, but on a prompt the protocol no longer uses.
- The v0e p1 counterfactual grades add entry files written with knowledge of later fixes. They bound when the
  JS was ready, not what the run would have scored had it written the page first.
- Tool-argument tokens are approximate (completion − reasoning − content/3.88). Content includes our own notes.
- Second-brain tokens are not recorded per run (`x_yamadori.investigate` has no token count). The ledger is
  daily. Adding `investigate.tokens` would close this.
- T1 counts "root cause stated" by pattern-matching the reasoning in the session export. Turns compacted out
  of the export are missing, so 8 is a lower bound.
- The skill `fix-located-defect-first` and the browser arm are unmeasured. Remedies 2 and 3 are bets that
  they bite. That is what the paired runs are for.

## Reproduce

```
cd bench/octopus/overthinking
python req.py && python timeline.py && python classify.py && python rereads.py
python table.py v0e2
python recon.py        # must print MATCH for every file
python snaps.py        # ~4-5 min per snapshot in Docker; rows in out/grades_snap.jsonl
```
