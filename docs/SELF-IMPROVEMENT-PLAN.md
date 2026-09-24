# Self-improvement plan (2026-09-24)

The stack builds and repairs itself: real agent harnesses (Hermes first) run
real backlog tasks against `yamadori`; every attempt is graded; winners are
reviewed and integrated; failures become evidence that trains the cheap paths.
Every fix is harness-agnostic -- if it only works for Hermes, it is not a fix.

## Status of the blocking bug

**Revised 2026-09-24 (afternoon), on evidence; the morning's diagnosis is
superseded.** The symptom: the model restarted every agent step "as if the
last turn didn't exist". The morning's fix restored the model's reasoning on
every past turn (`proxy.scope_reasoning`, then the ledger's `reasoning`
kind), on the theory that `strip_thinking` deleting it caused the restarts.

That theory cannot be the cause. Hermes itself strips `reasoning_content`
from every replayed assistant turn for a provider that does not require the
echo -- ours is a custom provider (`agent/message_sanitization.py`
`apply_reasoning_content_policy`; `agent/agent_init.py`: the echo flag is
opt-in) -- and the reference run (sudoingX: plain llama-server + Hermes +
Bonsai 2, 5 h, 328k tokens written, finished at 125k context) ran with NO
past reasoning in context and "does not lose the thread". The loop was most
likely OURS in two other ways: the CHANNEL-ORDER bug (tool-activity lines and
the real answer streamed as reasoning, so the harness dropped the answer with
the reasoning and the model next saw an empty turn -- fixed the same day in
`stream_body`) and our internal tool loop rebuilding turns. Not measured
apart: nobody replayed the morning's loop with only one of the two fixed.

**Now (design change, coordinator/operator):** past reasoning PASSES THROUGH.
What a client sends is what the model sees -- a stripping client's turns
render with empty think blocks, an echoing client's with its echo -- and the
ledger records and restores no reasoning. Reasoning lives within one request
(its own hidden hops; a deep-thinking hand-off prefilled as reasoning, whose
conclusion the visible opening carries forward). Restoring it had cost 6-10k
tokens of context per step (V0 pilot, ledger rows of 25-41k chars). The
cost to measure live: each request now diverges at the previous turn's think
block and reuses up to the checkpoint at the previous prompt's end.

Related, fixed the same day: streamed tool-activity lines went out as content
between reasoning hops, so clients rendered the real answer as thinking
(CHANNEL ORDER in `stream_body`).

## Phase 0 -- in flight (agents, offline, one deploy)

| # | workstream | done when |
|---|---|---|
| A | router (utility / agent step / code gen / code edit / library question / prose) + tool-call code check & repair (known tool-name table, shape fallback; syntax/lint/format; turn cap, full effort, last version sent, agent told what changed) | per-class precision/recall on >=300 labelled Hermes turns; utility and agent steps never routed to code pipelines |
| B | skills replace hints: ingest -> screen -> classify by artifact -> distil for Bonsai -> arm (no review gate) -> watch; cheap classifier first, self-healing fallback queue, idle-time improvement; length caps (fuzzy: ~300-500 tokens/skill, <=~1.5K/turn) | malicious fixtures quarantined; hints corpus migrated; fallback rate on the dashboard |
| C | compaction on the conversation's pinned slot (prefix reuse), summary drawn from spare KV (~5K budget), no delay, full window advertised | live compaction reuses most of its prefix from cache |

## Phase 0.5 -- one model, one cache (operator, 2026-09-24)

The client must see one standard model. Everything the proxy adds is kept in
a per-session ledger and re-rendered byte-identically on every request, so
the pinned slot's prefix always matches. Nothing is stripped: what a harness
stores is its decision; what the model is shown is what it saw last time.

| what the proxy adds | where it lives | why it cannot miss the cache |
|---|---|---|
| static addendum (how this model works: writes are checked, library questions are researched, the phrase vocabulary) | end of the client's system text, same every turn | never changes |
| per-turn context (skills, retrieval) | tail of the user turn it served, re-added from the ledger | was in the prompt when the slot processed it |
| the model's reasoning | NOT restored (superseded 2026-09-24): past reasoning passes through as the client sends it; it lives within one request (its hidden hops, a prefilled hand-off) | it cannot: the next request diverges at the previous turn's think block and reuses up to the checkpoint at the previous prompt's end -- cost to be measured live (see Status above) |
| second-brain results (deep thinking, B/C, fix-up) | distilled, then PREFILLED into the main turn: the conclusion opens the visible answer in fixed phrases ("After thinking deeply, ...", "Verified ...", "Repaired ...") | the slot processed exactly those tokens; conclusions sit in content, so they survive a harness's own compaction |
| a fixed tool call | the client stores the fixed call; the proxy warms the pinned slot with it while the harness runs the tool | the next request finds it cached |

Our tools leave main (except `generate_image`, `describe_image`); the second
brain holds them all and needs no client access -- what the harness read is in
the conversation, and a missing file comes back as the hand-off's NEXT STEP
for main to read with the harness's own tool. Fix-up and repair run on the
second brain; the weigh turn becomes proxy-side distillation (operator yes).

To verify before building (llama-server mechanics, measured by
`x_yamadori.cache` on the NEXT request): assistant prefill with reasoning
renders the same tokens the template renders later; a zero-token warm request
on a pinned slot caches the fixed call; the ledger survives a proxy restart
(persist it, or a restart costs one re-read per session).

**Verified 2026-09-24** (build b10738-285542d9, model `bonsai` on llama-swap
:11434, transient slot 3 only, `/slots` idle on two reads 5 s apart before
each probe, temperature 0, every probe run twice with identical numbers; a
~940-token system prompt so the numbers are readable; scripts kept in the
session scratchpad, not the repo -- they are mechanism probes, not a
benchmark):

| probe | request | next request: `cache_n` / `prompt_n` (of prompt) | verdict |
|---|---|---|---|
| (a) prefill, phrase `After thinking deeply,` | last message `assistant {reasoning_content, content: phrase}`, thinking on, medium | 976 / 19 (of 995); the 19 are `<\|im_end\|>` + the new user turn + generation prompt | **passes**: the model continues the phrase; the completed turn re-renders as a strict extension |
| (a) same, phrase with trailing space `After thinking deeply, ` | same | 973 / 18 (of 991) | passes here only because the continuation began with a digit; `/tokenize` splits `, ` into `,` + ` ` while `, the` is `,` + ` the`, so a prefill ending in a space breaks the token boundary for most words. **Prefill text never ends in a space** |
| (b) warm, `n_predict` 0 | `/completion` with the prompt rendered by `/apply-template` for `[..., assistant(fixed call), tool]`, cut before `<\|im_start\|>user\n<tool_response>` | 1278 / 23 (of 1301); 23 = the tail after the cut, exactly | **passes** (the server still predicts 1 token, content `""`) |
| (b) warm, `n_predict` 1 | same | 1278 / 23 | passes, same |
| (b) control, no warm | -- | 1208 / 93 | the whole fixed assistant turn is re-read |
| (c) medium, thinking on then off | same messages | 943 / 6 (of 947) | no effort line either way (the template writes one only for `xhigh` and `low`); only the generation prompt differs |
| (c) xhigh, thinking on then off | same messages | 448 / 502 (of 950) | the effort line disappears with thinking off, so the rendering shares 19 characters (`<\|im_start\|>system\n`) |

What the probes also showed:

- The prefill response (streamed and not) carries the prefilled reasoning
  (plus a trailing `\n`, which the template's `|trim` removes) and the
  phrase itself as its first content delta. The client therefore receives
  the phrase from llama-server; the proxy emits nothing extra.
- Thinking on or off makes no difference to a prefill's rendering at
  medium: the prefilled turn renders `<think>\n{reasoning}\n</think>\n\n`
  itself, and the generation prompt (where the two differ) is not used. At
  `xhigh`/`low` a prefill with thinking OFF loses the effort line and misses
  the whole cache (the (c) row). **A prefill or warm is sent with exactly the
  thinking and effort fields of the conversation's own turns.**
- A trailing `assistant` message that carries `tool_calls` is NOT a usable
  warm on the chat endpoint: prefill mode drops the calls
  (`/apply-template` renders it as `<think>\n{reasoning}` and stops). The
  warm is a raw `/completion` of the rendered prefix, as in (b).
- `cache_n` 448 recurs whenever a prompt diverges anywhere before its end,
  even 19 characters in, and even when llama-server's host-side prompt cache
  held a prompt sharing ~940 tokens. That is consistent with this build
  restoring only at context checkpoints (hybrid/recurrent memory), not at an
  arbitrary token; this server's log verbosity does not print checkpoint
  lines, so the mechanism is **inferred, not confirmed**. What it means for
  the design is not in doubt: a request reuses its full prefix only when it
  EXTENDS the slot's last sequence; any edit in the middle can cost far more
  than the edited tail.
**Operator decisions, 2026-09-24** (taken on the probe results above):

1. Library help at `medium` and `high`, where deep thinking is not allowed:
   a capped tail injection of held-symbol definitions (the router's
   held-symbol lookup, then `find_definition_opt`), recorded in the ledger
   and replayed identically. An UNMEASURED choice.
2. Deep thinking's distilled hand-off is prefilled as main's
   `reasoning_content`; the visible content opens with the fold-back phrase.
   The second brain's thinking becomes main's thinking, the ledger restores
   it on later turns, and the user sees only the conclusion.
3. "Verified" costs no generation: the proxy writes it as a one-line note.
   "Repaired" on a final answer: non-streamed, the repaired answer is
   delivered IN PLACE of the broken one; streamed, the repaired block is
   appended (the original has gone out). Either way the slot is warmed with
   the delivered turn, and the ledger records it.
4. Prefilled text ends on a letter (or an ending measured safe), never a
   space.
5. The static addendum is added only where every row of it is true: where
   the tool-call check is on (`high`, `xhigh`, `max`).
6. `bind_project_context` is deleted. The `check_code` tool leaves main. At
   `medium` the proxy checks client writes and adds an error NOTE without
   fixing (no generation; the slot is warmed when the note changes the
   stored turn). At `high` and up: check plus fix-up.
7. (New requirement) Every second-brain job (investigate, alternative,
   tiebreak, fixup) carries a concept seed (mcp/concept_seed.py) in its USER
   message. A fresh seed is drawn per job and recorded in the ledger keyed by
   (conversation turn, job); every replay of that turn or job uses the same
   seed. The seed also opens the visible fold-back ("Today I was inspired by
   <seed>. After thinking deeply, ..."), one seed line per fold-back, naming
   both when B and C ran. It is prefilled, so rule 4 applies.

- The first `/upstream/bonsai/...` call of the session found bonsai not
  loaded (llama-server's log starts ~3 minutes before the first probe, and
  only slot 3 ever ran a task), so llama-swap loaded it. No operator task ran
  on any slot during the probes.

**Status, 2026-09-24: built and offline-green; NOT yet run live.** What
exists now (AGENTS.md "One model, one cache" and "The second brain" are the
reference):

- STEP 1, the ledger: `proxy.ledger_restore` / `ledger_record_turn` /
  `ledger_seed` over nebari's `additions` table (per-user-turn and
  per-tool-result injections decided once, a call turn's delivered content,
  hidden internal hops with their reasoning emptied, concept seeds; NO
  reasoning since the afternoon's design change -- see Status), memory in
  front, pruned by size and age.
  `mcp/test_ledger.py` renders every upstream request with the served
  template (jinja2, `mcp/fixtures/bonsai_chat_template.jinja`) across a
  stripping client's session at tiers xhigh and max and asserts each one
  reuses everything before the previous assistant turn (the processed tail
  is that turn + the tool result + the new part); with the ledger's
  injection or hop replay switched off it fails at that turn. The
  known 87% bug (a hint missing from the resent copy) is now a 100% share
  in `mcp/test_utility.py`.
- STEP 2: main gets the client's tools plus the image tools; the static
  addendum at `high` and up; library definitions at `medium`/`high`; the
  proxy writes the work log; `bind_project_context` deleted; the check_code
  tool gone from main.
- STEP 3: `shomen.run` is the one second-brain runner (investigate,
  alternative, tiebreak, fixup), every job seeded; the hand-off is prefilled
  as main's reasoning; fix-ups never touch main; the weigh turn is gone
  (prose alternatives are prefilled after main's answer and main continues);
  the slot is warmed whenever the delivered turn differs from what was
  generated. `proxy._run_turn` is the one turn both paths run.
- STEP 4: an in-place compaction is served on the ledger's rendering,
  checked against the stored prompt; compactions think at the conversation's
  effort with 2,048 of thinking; the answer cap follows the client's target.

Open, and why: the warm, the prefill continuation after an answer, and the
checkpoint behaviour have been exercised only by the STEP 0 probes and the
offline template gate -- the live check (mcp/test_live_stack.py `cache`, at
xhigh, plus a medium pass) is what proves them on the real slot. The
definitions injection, the addendum's wording, the fold-back texts, the
ledger caps and the 2,048 compaction thinking budget are choices. Bytes a
turn adds to the ledger on the offline fixture: 10-370 (its reasoning is a
few words; real turns will be larger -- read `x_yamadori.ledger` and
`nebari.ledger_stats()` live before setting the caps).

**Incident during the build (reported to the operator).** Mid-refactor, one
offline suite (`mcp/test_tools.py`) faked `proxy._post` but not the new
turn engine's `_post_events`, so its requests went to the live llama-swap on
:11434 -- twice, each for up to the suite timeout (300 s and 900 s), the
second time from a `run_tests.py` started by another agent. Fixed: every
suite that runs a turn now points `proxy.UPSTREAM` and `model.UPSTREAM` at
its fake or at a refusing port, and `test_ledger.py` sets
`LLAMA_STACK_URL` to a refusing port before import.

## Phase 0.6 -- deep thinking with triggers (operator, 2026-09-24; after 0.5 lands)

Precondition (operator): internal tool hops are replayed exactly -- the ledger
records the hidden assistant call + tool result a proxy-executed tool produced
and re-expands them on the next request (in 0.5, generic "internal hops").

Deep thinking = a second context that takes a hard problem off main, works it
with server-side tools only, and hands back a distilled result with sources.

| tool (second brain only) | source |
|---|---|
| library source | existing indexes |
| skills + knowledge base (skill store, docs, work log) | thin wrappers over existing stores |
| web fetch | the skills pipeline fetcher + `skill_screen` (fetched text is data; hand-off cites the URL, labelled `(web)`) |
| web search | SearXNG, self-hosted, loopback only, JSON output, developer-leaning engines (operator, 2026-09-24). Native Windows first, in its own venv (Windows is not an upstream-supported target: try it, report what fails); WSL as today's fallback. Either way the watchdog supervises it like the other services, so it is up when deep thinking needs it; a search that finds it down returns the situation, retryable, and the remedy, per "Failure returns carry the next step". Queries still reach the upstream engines, unattributed. Long term: a launcher that also runs it on a macOS/Linux host (operator) |

Triggers, all at `xhigh`/`max`, one helper lane:

1. **Model-chosen:** `think_deeply` -- the one non-image tool on main; its
   description is a trigger list (tried twice and it still fails; unsure of an
   API/version; needs many files or docs; user says still broken -- instead of
   guessing or repeating a failing edit), plus one addendum row. Proxy-executed,
   replayed as an internal hop.
2. **Proxy-detected struggle:** from what the harness sends back -- the same
   tool erroring, the same file rewritten, a failing command re-run, fix-up at
   its cap, "still broken". Threshold 3 signals (a choice, set from dogfood
   data). Runs before main generates; result prefilled as reasoning.
3. **Known-hard area:** library/version question, or a skill that declares
   "escalate".

**Self-improvement loop (operator requirement).** Every deep-thinking decision
and every NON-decision is a record: trigger (model / struggle / area / none),
the signals, the hand-off, and the outcome over the following turns (struggle
stopped? check passed? same file rewritten again? user said still broken?).
Missed escalations (struggle that ended badly with no deep thinking) and wasted
ones (hand-off unused, task already fine) become labels. The idle-time job
(same queue as `skill_learn`) turns them into: struggle-threshold adjustments,
`think_deeply` description variants, route/Laya labels, skill escalate
triggers. Adjustments arm automatically within bounds and are recorded and
reversible on the dashboard; each names its n. Dashboard: escalations per day
by trigger, outcome rates, misses.

**Status, 2026-09-24: built and offline-green; NOT yet run live** (live runs
frozen until this deploys). The operator widened it the same day: a fourth
trigger, TASK KICKOFF (a new task whose spec is at least 1,500 tokens sends
its planning to the second brain -- evidence: V0 steps 1-3 planned on main at
12-14k reasoning tokens each, #18), the known-hard area as "a held package
the model cannot have seen", and Laya NOT consulted. AGENTS.md "Deep
thinking's triggers" is the reference. What exists:

- `mcp/deep.py`: the four triggers, pure; the per-conversation state in the
  ledger (episode boundary, cooldown, areas/kickoffs done, the sticky
  `think_deeply` offer); the records and the outcome labels in the corpus
  database (`deep_decisions`). `selection.decide` runs deep thinking exactly
  when a trigger fired, on any route class -- the `library_question` gate is
  gone -- and keeps the old rule + Laya path only as its legacy path (no
  route, no trigger) for the offline evaluators.
- `think_deeply` on main at `xhigh`/`max` plus one addendum row; executed by
  `proxy._think_deeply` through `shomen.run("investigate")` as a hidden hop
  the ledger replays; the next hop is prefilled with the fold-back opening.
- Struggle, area and kickoff run BEFORE main (`proxy._deep_thinking`); the
  kickoff is shomen's new `plan` job (FILES / ORDER / KEY DECISIONS / RISKS).
- The second brain's sources (`mcp/research_tools.py`): `find_skills`,
  `find_in_knowledge_base`, `read_web_page` (skills fetcher + `skill_screen`,
  private addresses refused, cited by URL with `(web)`), `search_web`
  (SearXNG on loopback, `YAMADORI_SEARCH_URL`; at most 3 searches per run;
  queries carrying code or secrets refused; down -> situation, retryable,
  remedy). Skills may declare `escalate: true` (frontmatter).
- The learner (`mcp/deep_learn.py`, `deep.learn` on the cpu lane when
  idle): `struggle_threshold` +-1 within [2, 6], `kickoff_tokens` +-25%
  within [500, 8000], each at n >= 5 since that parameter's last change,
  recorded and reversible; `think_deeply` description variants PROPOSED,
  never armed. `x_yamadori.deep`; `GET /dash/api/deep`,
  `POST /dash/api/deep/revert`. The dashboard panel is to follow.
- Unseen packages (revised the same day, coordinator: the version-date rule
  flagged every three.js conversation): the PACKAGE first published after
  the cutoff, or a prerelease / new major against the last release before
  it, or the operator's list; a minor or patch of a long-lived package is
  not. The registry history is stored per package
  (`index/packages/registry_history.json`, `deps.record_history`), backfilled
  for all held packages: glyph (first 2026-08-15), three-flatland
  (2026-02-25), r3f 10.0.0-alpha.5 and drei 11.0.0-alpha.7 are unseen;
  three 0.185.1 and koota 0.6.6 are not.

- Pre-deploy review (coordinator, 2026-09-24), fixed offline, each with a
  test that failed before (`mcp/test_deep_review.py`): read_web_page SSRF
  (pinned, re-validated hops incl. robots.txt, `is_global` only, deadline,
  byte cap); URL exfiltration (only search-result or user URLs, a guard,
  refusals recorded); `is_error` false positives ("0 failed", exit 0, empty
  grep); a busy lane defers the trigger, one wait per episode; a first-turn
  bug report is a task; one incident = one label, `in_cooldown`, traffic
  fails closed (`hermes-dogfood` = client); recording off the response path;
  compaction epochs, the continuation guard and a per-conversation lock;
  the knowledge base keeps network details out. And the operator's rule
  FETCHED CONTENT IS DATA, in one place (`skill_screen`): fetched text
  stripped, the hand-off screened on the way to main, skill items about
  unrelated actions dropped.

Every threshold, window, cooldown, bound, step and the model cutoff
(2025-12-31) is a CHOICE. Tests: `mcp/test_deep.py`, `mcp/test_ledger.py`
`[think]` / `[struggle]` / `[kickoff]`, updated `test_route`,
`test_selection`, `test_tools`. Owed live: a Hermes session where a struggle
fires and the next request's `x_yamadori.cache` shows the prefix held; a
kickoff on the Octopus spec (planning tokens on main before vs after); the
first labelled rows and the learner's first adjustment.

## Phase 0.7 -- the skills pass: one pipeline, one surface (operator, 2026-09-24)

Starts after 0.5/0.6 are proven live in Hermes, OpenCode and/or Pi.

Today there are two ingestion pipelines on one job queue (`mcp/jobs.py`):

| pipeline | surface | stages |
|---|---|---|
| datasets (`mcp/datasets.py`) | NAEDOKO, `/data` | submitted -> clarify (licence only from a verbatim quote) -> extract (the model reads a source into recipe rows) -> index (`hints.npz`) -> label -> train (Laya), the last two optional |
| skills (`mcp/skill_pipeline.py`) | SKILLS tab | fetch -> screen -> screen_model -> classify -> distil -> validate -> arm, plus watch |

Target: ONE pipeline and ONE dashboard surface, the skill approach
throughout. A source (URL, repo, package docs, paste) goes fetch -> screen
(exploits) -> clarify (licence from a verbatim quote: kept) -> classify
(applies-when) -> distil (title + DO/WHEN items, each with a verbatim source
quote) -> validate -> arm (no review; edit any time) -> watch. Training
(label -> train, for Laya/router heads) is an optional branch off the same
rows, not a second pipeline. The recipes/hints corpus (2,575 rows,
`skill_migrate.py`) becomes skills; `hints.npz`, the hints path and
`YAMADORI_RECALL` are deleted once skills beat or match hints on paired,
repeated live runs (Octopus variants, LiveBench). NAEDOKO (the seedbed)
is the natural name for the one surface; the SKILLS tab folds into it.
Nothing is renamed or deleted before that measurement.

## Phase 1 -- dogfood harness

1. Hermes installed headless in WSL (Ubuntu), pointed at `yamadori` on :1234,
   working in a **git worktree** of this repo -- never the live checkout.
   The API key is set by the operator.
2. Task bank: small backlog tasks with acceptance tests written first, e.g.
   `run_tests.py` names failing checks; `run_check` not offered when it can
   only fail; stale `budget.py` constants for q4 KV; deterministic
   `test_fanout`; stale numbers in `hints.py`; the Laya binary-router probe.
3. Grading (by the coordinating agent): tests pass, lint clean, diff scope,
   behaviour (loops, re-reads, router class, repair, skills, cache reuse,
   turns, minutes, electricity).
4. Winners reviewed and integrated; losers fixed or redone with the reason
   recorded.
5. Every attempt is data: transcript + per-turn `x_yamadori` + grade +
   failure category -> the skills fallback queue and the router's labelled
   set; the idle-time job turns them into triggers and training pairs.

## Phase 2 -- integrate and prove

Review every diff (agents' and Hermes'), full offline suite, one restart,
live acceptance through :1234 including a real Hermes session. Scoreboard on
the dashboard: Hermes task pass rate, repair saves, misroutes, skills
fallback rate, electricity per task.

Deferred until A and C land (same streaming code): cancel-on-disconnect.
