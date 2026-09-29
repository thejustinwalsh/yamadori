# Atomic Agent: what it does for small models, what we can take

Study of **Atomic Agent** (github.com/AtomicBot-ai/atomic-agent, MIT,
TypeScript) for the operator's bet that the proxy can snipe any harness's
wins so the choice of harness stays open (operator, 2026-09-29: "Do it, fold
in the actionable things from atomic agent").

- Source read at commit **52f90e55** (2026-09-29, `main`), web GET only:
  the GitHub API tree and `raw.githubusercontent.com` files. Nothing cloned,
  installed or run. The GAIA artifact bundle (`gaia-l1-eval.tar.gz`,
  10.8 MB, on the `gaia-l1-eval-2026-06-11` release) was **not** downloaded
  (it needs the operator's yes); mining its NDJSON traces is the one way to
  learn how often each of their mechanisms fired on their benchmark.
- Tags: **[their evidence]** = a measurement or a trace observation their
  repo records; **[claim only]** = stated, no measurement behind it in the
  repo; **[ours, n=...]** = counted here from our own logs by `bench/atomic/replay_logs.py` (section 4).

## 1. The headline evidence, and what it can carry

`eval-agents/docs/GAIA-L1-EXPERIMENT.md`: GAIA 2023 validation **Level 1**,
53 tasks, both agents on the same local `Qwen3.6-35B-A3B-UD-Q4_K_XL`
(llama-server, 262k ctx, Apple M4 Max), 40 steps / 15 min per task,
deterministic scoring (the official scorer, no LLM judge).

| | Atomic Agent 0.1.36 | Hermes |
|---|---|---|
| accuracy | **37/53 = 69.8%** | 31/53 = 58.5% |
| empty answers | 4 | 0 |
| wall / task | ~217 s | ~351 s |
| both right / atomic only / Hermes only / both wrong | 22 / 15 / 9 / 7 | |

What this can and cannot support **[their evidence, n=1 run per arm]**:

- **Not significant.** 15 vs 9 discordant tasks: exact McNemar (binomial,
  n=24) two-sided **p = 0.31** (computed here). Their own caveat: "Single
  run per configuration. No multi-seed averaging."
- **Confounded.** Atomic ran at temperature 0.2 (its default); Hermes at
  its own default (not configurable, their caveat). Atomic had its memory
  fabric and embedding daemon on; Hermes did not. The browser/fetch tools
  differ (GAIA L1 is web-heavy; their caveat).
- **No ablation of any technique.** Nothing in the repo turns one mechanism
  off and re-runs. The smaller-model rows (Qwen3.5-9B 28/53, Gemma-4-12B
  24/53) span Atomic versions 0.1.36-0.1.47 (their caveat: "not a
  controlled single-version sweep") and have no Hermes arm.
- The README headline ("The only variable is the agent loop") is **[claim
  only]** given the temperature and memory differences.
- **Their traces (section 9)**: the 37/53 run is `Qwen3.6-35B-A3B-UD-Q4_K_XL`
  on Atomic **0.1.36** (git cc5c7d4, dirty: 5 files), temperature 0.2,
  seed 10947100. In it the loop detector fired **3 times** (warnings, no
  veto), no memory recall reached the model, and 40 results were cut by the
  compressor.

So every per-technique entry below carries at most anecdotal evidence
(a trace observation written in a code comment). Where we build, the
evidence that decides is ours: how often it would have fired on our runs
(section 4).

## 2. Technique by technique

Each: the mechanics (file, numbers, wording), the evidence, where it maps
for us, and the decision.

### 2.1 Stable prefix and slot reuse

- **Mechanics.** `src/prompt/stable-prefix.ts`: persona, rules, skill
  catalog, tools, capabilities, instructions byte-stable within a session;
  a variable tail after it (`### conversation`, `### notice`,
  `### respond`), ordered "by what can change within a turn" so a change
  lands behind the transcript. `src/llm/slot-manager.ts`: first request
  `id_slot: -1` + `cache_prompt: true` (llama-server picks by prefix
  similarity), then the session is pinned to the slot the server named; a
  reserved slot for side calls (reflection) so "main agent KV-cache is
  never invalidated". `packConversation` holds its cut between overflows
  (`conversationLowWater` 0.65) so "between cuts the prompt only grows at
  its end".
- **Evidence.** [their evidence, anecdotal] "A model whose attention cannot
  roll back (Gemma 4's sliding window) then re-read the whole prompt each
  step: 40 of one turn's 79 minutes went to prompt evaluation"
  (`conversation-turn.ts`, `ConversationPackStart`).
- **Map: already ours.** "One model, one cache" (AGENTS.md): slot pinning
  (`mcp/slots.py`), the child slot for side calls and the second brain
  (`slots.child_slot()`), the ledger that replays every addition byte for
  byte (`proxy.ledger_restore`), past reasoning restored so each request
  extends the slot. Nothing to take.

### 2.2 GBNF-constrained tool-call arrays

- **Mechanics.** `grammars/tool-call.gbnf`: `root ::= tool-call-array`,
  1-16 calls `[{"tool": ..., "args": {...}}]`, tool names enumerated; the
  grammar travels with each request (not in the cached prefix) and is
  narrowed per step (`build-grammar.ts`); whitespace bounded `{0,64}`
  ("an unbounded ws gives a stuck sampler an infinite legal move"); the
  reasoning prelude bounded to `reasoningBudgetTokens x 4` characters (F49).
- **Evidence.** [their evidence, anecdotal] Gemma 4 26B "used to slide into
  a whitespace-only tail after a long reasoning block"; a model "that would
  think for 22 minutes is made to close the block".
- **Map: skip.** It is Atomic's own wire format (their harness parses the
  array). Our harnesses speak OpenAI tool calls; llama-server renders and
  parses them through the served template. We already bound thinking per
  request (`reasoning_budget_tokens`, `tiers.budget`). Malformed calls are
  not our observed failure: **0 of 1,570** harness calls in our logs carry
  a template marker in their arguments [ours, n=1,570; same replay].

### 2.3 Parallel read batching and single-inference batching

- **Mechanics.** One inference = one JSON array of independent calls;
  `src/agent/tool-resource-class.ts` groups them (`pure_read`, `fs_write`,
  `browser`, ..., `approval_gated` forbidden in a batch, `terminal`
  (`reply`/`finish`) only last); groups run concurrently. `[store, reply]`
  collapses a write and the answer into one inference.
  `agent.maxParallelToolCalls` 8 (grammar cap 16).
- **Evidence.** [their evidence, anecdotal] "Bumped from 4 -> 8 after
  production traces showed qwen-3.5 routinely emits 5-7 reads when the user
  requests >=N files".
- **Map: harness-config.** Executing calls is the harness's job; the proxy
  cannot run a harness tool. Multi-call turns already happen through us:
  Hermes **52 of 575** assistant turns, Pi **30 of 510** [ours: Hermes
  session exports, Pi `turn_end`]. The proxy passes `parallel_tool_calls` as
  the client sends it (`responses_api.py`); Codex and Hermes' Responses wire
  send `true` (bench/harness_shapes), Hermes chat, Pi and OpenCode send
  nothing. Whether llama-server's default admits several calls for our
  template was not checked here. Nothing to build.

### 2.4 Compact ARIA browser snapshots

- **Mechanics.** `src/tools/browser/aria-compressor.ts`: Playwright AI-mode
  ARIA tree, noise roles dropped (`generic`, `group`, `none`,
  `presentation`, `paragraph` with no name or text), then a character
  budget `DEFAULT_ARIA_MAX_CHARS` 24,000 and 800 lines.
- **Evidence.** [their evidence, anecdotal] "The previous 300-line cap was
  measured to truncate GitHub READMEs before the model ever saw them".
- **Map: harness-config.** A browser tool's output shape is the harness's
  (Hermes' browser tools, OpenCode/Pi MCP browsers). Our runs verify in a
  browser through harness tools (`verify_moment.py`); the proxy could
  compress a snapshot result the way it compresses shell output, but no run
  of ours shows browser results as a cost. Skip until a run does.

### 2.5 Externalized state

- **Mechanics.** Session state in SQLite (`src/session/session-state.ts`),
  `### world` (the browser's current snapshot) rendered fresh, not stacked
  in history; memory in its own SQLite; the loop owns stop conditions.
- **Map: already ours / harness.** The harness owns its transcript; the
  proxy keeps its own state outside the model (ledger, rings work log,
  deep state). Nothing to take.

### 2.6 The "compress" step (result summarization)

- **Mechanics** (the three places that decide what the model reads of a
  result):
  1. `src/compressor/result-compressor.ts` `compressToolResult`: blank lines
     dropped, the last `maxTailLines` kept, then the character cap, cut
     from the head (`overflow: "head"`) or the tail (`"tail"`); for a
     failure the first line matching `ERROR_MARKERS` (`error:`, `failed:`,
     `traceback (most recent call last)`, `assertionerror`, `exception:`)
     is lifted on top as `key: <line[:180]>`. Defaults 400 chars / 12 lines.
     **No model call**: it is deterministic truncation, not an LLM summary
     (`log-summarizer.ts` counts PASS/FAIL lines, also no model).
  2. Shell ingestion (`src/tools/os/shell-result.ts`): 16,000 chars / 500
     lines, `overflow: "tail"` ("an `exit:` line or a test verdict is the
     reason the command was run").
  3. Render (`src/session/conversation-turn.ts`): a shell result is shown
     **tail-first**, at most `TOOL_RESULT_RENDER_CAP_CHARS` **8,000** on the
     inference that consumes it, and **400** (`TOOL_RESULT_HISTORY_CAP_CHARS`)
     once its macro-turn (a user message .. the reply) closes; a file read is
     cut on a line boundary with the exact `offset` of the rest; a batch
     shares `batchToolResultCharCap` 32,000 by water-filling
     (`batch-summary-cap.ts`, floor 600 each).
- **Evidence.** [their evidence, anecdotal] "a fusion reviewer re-read a
  4.5 KB `main.js` five times and never saw its last 486 chars" (the reason
  for line-boundary cuts with the next offset); "one argument-free
  `os.proc.list` would take 53% of" an 8k window (the reason for the 4,000
  listing cap). No measurement of task success with vs without compression.
- **Map: proxy-snipable for EVERY harness** (the proxy sees every tool
  result before the model does), **with one limit**: Atomic's aging
  (8,000 fresh -> 400 once the task is answered) rewrites a result the model
  already read, which changes the prefix the slot holds. The proxy can only
  decide once, when a result first arrives, and replay that decision.
  Within one task Atomic itself never ages (all our pagoda runs are one
  task), so the comparable rule is: **a shell result over 8,000 characters
  is cut to its tail, once**.
- **Built:** `mcp/result_compress.py` (section 5). What it would have done
  on our runs is in section 4: almost nothing.

### 2.7 Memory with on-demand recall

- **Mechanics.** `memory.notes.store/recall/forget`, profile facts, lessons,
  procedures, all SQLite (FTS5 + optional embeddings, `hybrid-recall.ts`);
  an end-of-turn reflection on a reserved slot writes facts; a recall tail
  (`### recalled`) is pre-fetched per turn.
- **Evidence.** `eval-memory/` holds LoCoMo / LongMemEval harnesses; no
  result in the repo ties memory to GAIA accuracy [claim only for the
  agent benchmark].
- **Map: skip / harness.** Cross-session user memory is a harness product
  feature. Our in-task equivalent is the work log (`rings`) re-injected
  after a compaction; our knowledge recall is the skills system. Nothing to
  take.

### 2.8 Approval gates

- **Mechanics.** `src/approval/approval-gate.ts`, levels, a dangerous-tool
  list, a shell command guard (`rules-hardline.ts`, `rules-dangerous.ts`).
- **Map: harness-config.** Each harness has its own approvals (Hermes
  refuses flagged commands: `chat__environment-refusals.json`). The proxy is
  not the place to approve a user's shell. Skip.

### 2.9 The no-progress guard (`ToolLoopTracker`)

`src/agent/loop-detector.ts`, "Ported from OpenClaw 2026.6.5" (their
AGENTS.md). OpenClaw ships its equivalent **off by default** and
recommends it "For smaller models" (docs/tools/loop-detection.md) [claim
only]. One tracker per turn (a user message, its steps, its reply).

| detector | identity | fires | action | Atomic default |
|---|---|---|---|---|
| `generic_repeat` | tool + canonical JSON args (keys sorted, values exact), counted in the window, interleaving tolerated | prior count >= `LOOP_WARNING_THRESHOLD` (the **4th** identical call) | a `### notice` in the next prompt; the call runs; once per bucket of 10 | 3 |
| `no_progress` | args AND result hash; a changed result breaks the streak; vetoes skipped | streak >= `LOOP_CRITICAL_THRESHOLD` (the **6th** call, after 5 identical results) | **veto**: the call never runs; a synthetic `status: error` result (`deniedReason: "tool-loop"`) | 5 |
| breaker | consecutive vetoes of one signature | >= `LOOP_BREAKER_VETO_STREAK` | the turn ends with a synthetic reply ("stopped: ... Here is my best answer ...") | 3 |
| `wandering` | distinct args on web/http/browser tools only ("scanning many files is legitimate work") | spread >= 6 / >= 12 | redirect notice / breaker | 6 / 12 |
| `outcome_repeat` | tool + status + first 200 chars of the result, whitespace collapsed, whatever the args; reset by a successful write | 3rd time | notice only | 3 |
| `read_repeat` | same file, same content hash, no new line returned | 2nd consecutive | notice only | 2 |
| `test_repeat` | a recognised test command against an unchanged workspace fingerprint | 2nd run | notice only | 2 |

Window `LOOP_HISTORY_SIZE` 30. A result's volatile JSON keys are dropped
before hashing (`VOLATILE_RESULT_KEYS`: `timestamp`, `ts`, `date`, `time`,
`timeTotal(Seconds)`, `durationMs`, `sizeDownload`, `requestId`,
`request_id`, `id`, `traceId`, `trace_id`, `sentAt`, `createdAt`,
`deliveredAt`); a shell result hashes as exit code + summary.

**Wording shown to the model** (verbatim, which we do NOT reuse): the veto
opens "BLOCKED: `<tool>` — N consecutive calls returned the same
no-progress outcome." then "Change strategy BEFORE calling any tool
again:" and "- Do NOT repeat this exact call."; the wandering redirect says
"This is a wandering loop. STOP probing more URLs/pages". Capitals and
prohibitions, both against our rules (AGENTS.md "Prompting this model":
prohibitions degrade; "Failure returns carry the next step").

**Evidence.** [their evidence, anecdotal, from code comments] "run 02 ran
one failing check chain five times with one warning, and run 04 spent six
steps re-globbing and re-grepping with slightly different arguments and
identical answers" (the reason for `outcome_repeat`); "GAIA traces show
small models burn an entire step budget re-formulating ~35 distinct search
queries" (wandering); "21 identical search POSTs" that never registered
until volatile keys were stripped. No count of how often the veto fired in
the GAIA run, and no accuracy with vs without it.

**Map: proxy-snipable for every harness.** The proxy holds a generation's
client calls until the generation ends (tool_code checks them there), so it
can decide on each call before the harness ever sees it, reading the
earlier calls and results from the messages the client sent. The veto uses
the image guard's mechanism (tool_code IMAGE GUARD: a hidden hop with a NOT
EXECUTED result, the model writes its turn again, the ledger replays the
hop). **Built:** `mcp/progress_guard.py` (section 5). One difference from
Atomic, forced by that mechanism: Atomic runs a vetoed call's siblings; the
proxy cannot forward half a generation, so every call of a stopped
generation gets a result (the vetoed one the veto, the others "not run")
and the model writes the turn again -- exactly the image guard's rule.

### 2.10 Task-unit history bounds

- **Mechanics.** `agent.conversationMaxPairs`: history counted in tasks
  (macro-turns), code default **200** (`config-schema.ts`; the README says
  "1-100, default 20" -- the README is stale against the code [claim vs
  code]), plus a token ceiling; when either overflows the transcript drops
  whole tasks to `conversationLowWater` 0.65 of the limit and holds the cut
  (`ConversationPackStart`), with a one-line recap of what was dropped.
- **Map: harness-config / skip.** The harness owns its history and its
  compaction (Hermes, Pi and OpenCode each compact on their own); the proxy
  already serves compactions as part of the conversation
  (`_serve_compaction`) and refuses a request past the window with
  `context_length_exceeded` so the harness compacts. Dropping history
  ourselves would break the harness's own view and the slot's prefix.
  Nothing to build.

### 2.11 Smaller things seen on the way

`control-marker-guard.ts` (template markers inside tool arguments: 0 of
1,570 calls in our logs, skip); `truncation-recovery.ts`,
`empty-completion-recovery.ts`, `parse-failure-recovery.ts` (their own
wire's failures; ours are `finish_reason` events the harness sees);
`read-coverage.ts` (our #54 unchanged-read line did this and was removed
2026-09-27 with the other tool-result situations).

## 3. The map, in one table

| technique | where it lands | why | built |
|---|---|---|---|
| stable prefix / slot reuse | already ours | slots, ledger, restored reasoning | -- |
| GBNF tool-call arrays | skip | Atomic's own wire; ours is the template's tool calls; 0/1,570 malformed-marker calls | -- |
| parallel read batching | harness-config | the harness runs calls; multi-call turns already occur (52/575 Hermes, 30/510 Pi) | -- |
| single-inference batching | harness-config | same | -- |
| ARIA snapshots | harness-config | the browser tool's output is the harness's; no cost seen in our runs | -- |
| externalized state | already ours / harness | ledger, rings | -- |
| result compression (shell, tail, once) | **proxy** | the proxy sees every result first | `mcp/result_compress.py` |
| result aging (8,000 -> 400) | skip | rewrites what the slot holds | -- |
| memory + recall | harness / skip | cross-session memory is a harness feature; our in-task memory is rings + skills | -- |
| approval gates | harness-config | the harness approves its own shell | -- |
| no-progress guard: repeat warn, veto, breaker | **proxy** | the proxy holds a generation's calls before the harness runs them | `mcp/progress_guard.py` |
| outcome-repeat warn | **proxy** | same; warn only | `mcp/progress_guard.py` |
| wandering (web/browser) | proxy, not built | our harness runs rarely use web tools | -- |
| wandering over dependency files | operator decision | Atomic excludes file reads on purpose; it is our largest measured pattern (section 4) | counted only |
| task-unit history bounds | harness-config / skip | the harness compacts; the proxy serves compactions | -- |

## 4. What each would have done on our runs [ours]

`python bench/atomic/replay_logs.py` replays every run under
`C:\Users\jwals\octo\logs` (Pi: `pi.jsonl` tool executions; Hermes:
`hermes.jsonl` tool calls for the guard, the session exports for result
sizes, since `hermes.jsonl` cuts results at 5,000 characters -- 138 of 1,012
results are exactly 5,003 long). Tasks restart at a user turn and at each
compaction. The guard is replayed on the calls as they happened (a veto
would have changed what came next, so "would veto" counts calls that met
the rule). Snapshot 2026-09-29 10:25; **pagoda-p4 was still running** and
its counts grew between replays.

| run | harness | calls | max identical | warn | outcome | veto | dependency probes | probe chars / all result chars | probe spread >=6 / >=12 | shell results | compressed | chars saved |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pagoda-h1 | hermes | 102 | 1 | 0 | 0 | 0 | 57 | 93,741 / 120,510 | 1 / 1 | 82 | 2 | 9,077 |
| pagoda-h4 | hermes | 134 | 2 | 0 | 1 | 0 | 66 | 143,298 / 263,050 | 3 / 3 | 32 | 0 | 0 |
| pagoda-h5 | hermes | 26 | 1 | 0 | 1 | 0 | 9 | 21,054 / 28,022 | 1 / 0 | 23 | 0 | 0 |
| pagoda-h6 | hermes | 76 | 1 | 0 | 0 | 0 | 46 | 105,430 / 127,622 | 2 / 1 | 42 | 1 | 1,028 |
| pagoda-p1 (max) | pi | 38 | 1 | 0 | 0 | 0 | 23 | 37,202 / 45,980 | 1 / 1 | 38 | 0 | 0 |
| pagoda-p1 (xhigh) | pi | 29 | 1 | 0 | 0 | 0 | 5 | 6,945 / 20,364 | 0 / 0 | 28 | 0 | 0 |
| pagoda-p2 | pi | 13 | 1 | 0 | 0 | 0 | 7 | 23,695 / 28,260 | 1 / 0 | 8 | 0 | 0 |
| pagoda-p3 | pi | 288 | 4 | 0 | 2 | 0 | 112 | 186,680 / 339,855 | 2 / 2 | 202 | 1 | 678 |
| pagoda-p4 (running) | pi | 145 | 4 | 1 | 0 | 0 | 37 | 124,020 / 250,836 | 3 / 1 | 70 | 2 | 41,208 |
| v0b | hermes | 103 | 2 | 0 | 0 | 0 | 0 | 0 / 140,836 | 0 / 0 | 11 | 0 | 0 |
| v0e | hermes | 336 | 4 | 1 | 4 | 0 | 6 | 2,638 / 550,594 | 0 / 0 | 53 | 1 | 49,735 |
| v0f | hermes | 201 | 3 | 0 | 0 | 0 | 0 | 0 / 232,448 | 0 / 0 | 82 | 0 | 0 |

(Runs with fewer than 10 calls -- h2, h3, pilot, probes, smokes, v0c, v0d
-- fired nothing.) **Totals, 1,564 calls in 29 run directories:**

- **The veto (Atomic's 5-streak) would have fired 0 times.** No call was
  ever made 6 times with identical results in a task; the most any call
  was repeated in one task is **4** (a `tsc` re-run after edits in p3; a
  re-read of `world-data.ts` in p4; a re-read of `enemies.js` in v0e).
  The breaker, 0.
- **The repeat warning (4th identical call): 2** (p4 `read
  pagoda/src/world-data.ts` x4, v0e `read_file .../enemies.js` x4).
- **The outcome repeat (3rd identical result, any arguments): 8**, all
  genuine: v0e `search_files` returning `{"total_count": 0}` 9 times and a
  grep regex error (`Unmatched ( or \(`) 7 times, `vision_analyze` failing
  identically 5 and 4 times (the #44/#46 image-data episode); p3 two
  identical `grep` results and a port check; h4 one `tsc`; h5 one `grep`.
- **Compression at Atomic's 8,000: 7 of 702 shell results**, 101,726
  characters saved (~26,800 tokens at 3.8 characters a token) of 1.74M
  result characters (5.8%). 2 more were protected (a `tsc --listFiles` that
  lists the model's own files; a syntax check naming `js/game.js`). Tool
  results are small: of 1,529 results (Pi whole; Hermes from hermes.jsonl,
  which cuts at 5,000), 1,177 are <= 2,000 characters, 159 are 2-4k, 185
  are 4-8k, 9 are over 8k (Hermes' session exports hold 29 over 8k: 21
  `read_file`, never compressed; 5 `terminal`; 2 `write_file`; 1 `patch`). The p3 `sed -n` dumps of
  `node_modules` the operator named are 5-8.7k each: **under** the cap.
- **What does cost: reading installed dependencies.** 370 calls (24% of all
  calls) read inside `node_modules/` or `site-packages/`, returning 744,887
  characters -- **34% of all result characters**, and 78% (h1), 54% (h4),
  83% (h6), 81% (p1-max), 55% (p3) of their runs. These are DISTINCT
  reads, which Atomic's argument- and result-keyed detectors never see.
  Atomic's WANDERING rule, extended to dependency reads as one tool class
  (6 distinct in the 30-call window: redirect; 12: end the turn), would
  have redirected in **14** task episodes and ended the turn in **9**
  (188 calls past 12) -- every pagoda run of both harnesses. Atomic
  deliberately does NOT apply wandering to file reads, and ending the turn
  ends a harness's run; so this is counted, not built.

**Reading.** On our runs Atomic's two headline mechanisms would barely have
acted: the model does not loop on identical calls, and our harnesses already
cap shell output (Pi's bash, Hermes' terminal) below Atomic's render cap.
What they would catch is real but small (8 outcome repeats, 2 repeats, 7
compressions, 0 vetoes). The measured cost is dependency probing, which is
a different pattern (distinct reads) that neither mechanism was designed
for.

## 5. What was built

Two pure modules (no I/O, no proxy import, no state; everything read from
the messages passed), their offline suites, and the replay script. No change
to `mcp/proxy.py` or any file under rewrite.

### `mcp/progress_guard.py` -- 64/64 (`mcp/test_progress_guard.py`)

- `decide(messages, name, arguments)` -> `{level: ok|warn|veto|breaker,
  detector, count, key, tool, target}`: Atomic's `check()` against the
  current task (after the last user turn, last 30 calls).
- `gate(messages, tool_calls, is_client)` -> `(stop, verdicts)`: one
  generation's client calls in order (a duplicate inside a batch counts).
- `hand_back(tool_calls, stop)` -> the tool messages for a stopped
  generation (`veto_result` for the vetoed call, `other_result` for the
  rest).
- `notice_for_result(messages, call_id)` / `notices_for_trailing(messages)`
  -> the warning line for a result (repeat on the 4th identical call, once
  per bucket of 10; outcome repeat on the 3rd identical result since the
  last successful write).
- `record(verdict, delivered)` -> the `x_yamadori.progress_guard` row: level,
  detector, count, tool, command word, key hash, thresholds and their
  source; never arguments or result text.
- Numbers: Atomic's `ENV_DEFAULTS` (warn 3, veto 5, breaker 3, window 30,
  bucket 10, outcome 3 / 200 chars), each pinned by an env variable
  (`YAMADORI_GUARD_WARN_REPEATS`, `_VETO_STREAK`, `_BREAKER`, `_WINDOW`).
  **Not measured on this model; for the operator to confirm.**
- Identity: exact canonical JSON arguments; results by text, with Atomic's
  volatile JSON keys and Codex's `Chunk ID:` / `Wall time:` lines dropped;
  failure read from structured fields only (`exit_code`, `ok`/`success`,
  `error`), so a write resets the outcome count unless it failed.
- What the model reads (no capitals, no doubt, and ONE prohibition, in the
  warning line only -- operator, 2026-09-29: "do not repeat is a good
  exemption to add a rule for that helps steer a session"; the tests scan
  for all of it):
  - veto: "NOT EXECUTED (no progress): `terminal` (`npx`) with exactly these
    arguments already ran 5 times in this task and returned the same result
    each time; that result is in the conversation above, and running the
    call again returns it again. Retryable: no, with these arguments. Next
    step (yours): act on that result -- change the arguments or the command,
    edit the file the result points at, or answer with what you have."
  - warning (DELIVERED on the repeated call's result, operator
    2026-09-29): "This is the 4th time this task has run `read` with exactly
    these arguments and returned the same result each time. Do not repeat
    this identical call. Instead, act on the result above: change the
    arguments or the command, edit the file it points at, or answer with
    what you have."
  - breaker (3 vetoes in a row): the veto's result plus "It has also been
    written 3 more times since then and not run." The guard NEVER ends the
    harness's turn (operator, 2026-09-29); `level` "breaker" is for the
    record, `action` is "veto" for both.
  - outcome: "`search_files` has returned this same result 3 times since the
    last file write. Next: act on it -- edit or write the file it points at,
    run a different command, or answer with what you found."

### `mcp/result_compress.py` -- 41/41 (`mcp/test_result_compress.py`)

- `compress(name, arguments, text, written, cap_chars)` -> `(text, record)`.
  Only `SHELL_TOOLS` (`terminal`, `bash`, `exec_command`: each verified from
  bench/harness_shapes), only over `CAP_CHARS` 8,000 (Atomic
  `TOOL_RESULT_RENDER_CAP_CHARS`); blank lines dropped, the last 500 lines
  (`SHELL_TOOL_RESULT_TAIL_LINES`), then the tail that fits, cut on line
  boundaries; a structured failure keeps its first error line on top
  (Atomic `extractSignature`); one line says what was left out and how to
  see it: "[N earlier lines of this output (M characters) are not shown; the
  last K lines follow. To see the part not shown, run the command again
  narrowed: a line range, a pattern, or the first lines.]". Hermes' JSON
  stays valid (only `output` changes), Codex's header stays, Pi's exit line
  is the tail.
- **Never the model's code**: a result whose command or output names a path
  the model wrote (a write/edit call's path via `tool_code.KNOWN`; a shell
  redirect target outside heredoc bodies; matched on the last two path
  components) passes whole (`why: protected`).
- `compress_at(messages, i, cap)` judges a result on the messages up to it,
  so the ledger can store only the DECISION (the cap) and re-derive the
  same bytes on every later request -- nothing of the caller's text is
  stored (nebari's rule).
- `compress_last(messages, cap)`: only the results the request ends on.
- Record for `x_yamadori.result_compress`: tool, before/after characters,
  lines before/kept, key line, cap, why (`not_shell`, `under_cap`, `fits`,
  `protected`, `no_gain`, `compressed`).

### `bench/atomic/replay_logs.py`

The counts in section 4, reproducible: `python bench/atomic/replay_logs.py
[--logs DIR] [--out FILE.json]`. Read-only.

## 6. The hooks, for the operator to place (proxy.py is being rewritten)

Line numbers are proxy.py as of 2026-09-29 10:00; the anchors are named so
they survive a rewrite.

**0. Keep the client's messages.** In `prepare`, where `raw` (the client's
messages as sent -- what `_skills_tail` reads) is final:

```python
payload["_raw_messages"] = raw
```

**1. The veto** -- in `_run_turn`, right after
`img = d.get("_image_arg")` (~7121), before `calls` is computed:

```python
nog, nog_v = (progress_guard.gate(payload["_raw_messages"] + hops_added, msg.get("tool_calls"), is_client=lambda n: not is_ours(n, ours)) if guard_on and img is None and not last else (None, []))
```

and treat `nog` exactly like `img`: `calls = []` when `nog` is set; after
the image guard's hand-back block (~7277, "THE IMAGE GUARD'S HAND-BACK"):

```python
if nog is not None:
    for tmsg in progress_guard.hand_back(hop_msg["tool_calls"], nog):
        convo.append(tmsg); hops_added.append(tmsg)
    guard_rec.append(progress_guard.record(nog, "vetoed"))   # veto AND breaker: never _land
```

`guard_on`: not at `minimal`/`low` (the model as it ships, as the image
guard), and a `tiers.BEHAVIOURS` switch `progress_guard`
(`YAMADORI_PROGRESS_GUARD`). `x_yamadori.progress_guard = guard_rec`.
**Open point for the hook (the breaker decision):** nothing ends the turn,
so within ONE request a model that keeps writing the vetoed call is bounded
only by the existing tool-turn cap (`tiers.tool_turn_limit`, which counts
hidden hops like the image guard's) -- and at that cap the proxy LANDS, which
is the turn-ending the operator ruled out. The hook's author must choose how
the guard's hops meet that cap (e.g. not counted toward it, with some other
bound); the module does not decide it.
Offline gate to add then: `mcp/test_ledger.py` `[guard]` -- five identical
Hermes calls, the sixth stopped, the hidden hop replayed, the next request
extends the slot (the image guard's `[image]` checks are the template).

**2. The warning and outcome lines** -- in `prepare`'s `ends_on_tool`
block (~3070, the tool-result injection decided once under `keys[-1]`), as
one more part beside `library_use` and `skills`:

```python
nts = progress_guard.notices_for_trailing(raw) if notice_on else []
if nts: parts.append("progress_notice"); use += "\n" + "\n".join(n["text"] for n in nts)
```

(recorded by the same `nebari.ledger_decide(... "inject" ...)`, replayed by
`ledger_restore`). DELIVERED, not record-only (operator, 2026-09-29). Switch
`progress_notice` (`YAMADORI_PROGRESS_NOTICE`).

**3. Compression** -- two lines:

- decide, in `prepare`'s `ends_on_tool` block, for each trailing tool
  message `i` of `raw` (before the injections are appended):
  ```python
  new, rec = result_compress.compress_at(raw, i); nebari.ledger_decide(account, lineage, key_of_result_i, "compress", "", {"cap": rec["cap"], "why": rec["why"]})
  ```
- replay, in `ledger_restore`, for a tool message with a stored `compress`
  decision, BEFORE its injections are appended:
  ```python
  m = dict(m, content=result_compress.compress_at(messages, i, meta["cap"])[0])
  ```

A result the model already read whole is never shortened later (only the
request that ends on it decides), so no slot's prefix changes. Switch
`result_compress` (`YAMADORI_RESULT_COMPRESS`); `x_yamadori.result_compress`
= the records.

## 7. How to measure it on our own runs

Offline, done: section 4 (`bench/atomic/replay_logs.py`).

Live, after the hooks land (not before: fix -> test -> deploy -> run), on
the pagoda prompt with Hermes and Pi, as before (headline: prompt-1
single-shot; no graded follow-ups; the operator judges the output -- no
on/off arms):

1. **Did it fire, and was it right?** From relay.jsonl's `x_yamadori`:
   `progress_guard` rows (level, detector, count) and `result_compress`
   records per request. Expected from section 4: a handful of outcome
   lines, 0-2 repeat lines, vetoes rare; 0-2 compressions per run.
2. **What the model did next.** Extend `replay_logs.py` to read the new
   records: after a veto, did the next generation make a different call (and
   did a project write follow within N steps); after a warning, did the key
   repeat again; after a compression, did the model re-run the same command
   narrowed (a re-fetch: same leading command and target plus
   `head`/`sed -n`/`grep`) -- a re-fetch rate near 0 means the tail was
   enough.
3. **What it cost or saved.** Per run: result characters in (before/after
   from the records), `x_yamadori.cache` processed tokens per request,
   compactions per run, wall time -- beside h1-h6 and p1-p4.

## 8. For the operator

1. **Build status.** Both modules and their suites are in; no hook is
   placed. The evidence from our runs says their effect will be small
   (0 vetoes, 2 repeat warnings, 8 outcome warnings, 7 compressions in 1,564
   calls). Place them anyway (cheap, bounded, recorded), or wait for the
   rewrite to settle?
2. **Thresholds.** Atomic's (3/5/3/30; 8,000/500) with their sources; not
   measured on Bonsai. Keep, or set.
3. **The breaker** -- DECIDED 2026-09-29: never end the harness's turn;
   the NOT EXECUTED result, and the run continues. Open for the hook: what
   bounds repeated rewrites of the vetoed call inside one request, since
   the tool-turn cap lands (section 6, hook 1).
4. **The warning lines** -- DECIDED 2026-09-29: delivered, and they say "Do
   not repeat this identical call" then what to do instead.
5. **Dependency probing** is the pattern that costs (34% of result
   characters; wandering >= 12 in every pagoda run). Atomic's answer
   (distinct-argument wandering) is built for web tools and ends the turn.
   The retired `probe` auto-trigger answered it with deep thinking, which is
   being removed. What should answer it next (a skill, a notice, nothing)?
6. **The GAIA traces** -- DONE 2026-09-29 (operator approved the
   download; section 9).
7. **A bound for the guard's hops that does not end the turn** (hook 1;
   follows from decision 3). Inside ONE request, a vetoed call goes back to
   the model as a hidden hop and the model writes its turn again; if it
   writes the same call again it is vetoed again. Today the only bound on
   hidden hops is the tool-turn cap (`tiers.tool_turn_limit`: 10, 20 at
   `max`), and at the cap the proxy LANDS -- the turn-ending ruled out. In
   Atomic's GAIA traces the model wrote the vetoed call again at the next
   step 4 times in 16 vetoes (Gemma-4-12B) and 0 in 1 (Qwen3.5-9B)
   (section 9), so a repeat after a veto does happen. Options:
   - **A. Forward after N.** After N guard hand-backs in one request
     (N = `BREAKER_VETOES` = 3, Atomic's), deliver the model's next
     generation to the harness as written, the repeated call included: the
     harness runs it and the run continues. The next request gates afresh
     (its streak still stands, so a further repeat is vetoed again). Bounded
     at N+1 generations per request; the cost is one real run of the
     repeated call per request cycle.
   - **B. Deliver without the call.** After N hand-backs, deliver the
     generation with the vetoed call removed (its other calls forwarded).
     When it was the only call, the harness receives text and no call --
     which Hermes, Pi and OpenCode read as an answer and END THE RUN (the
     pagoda-h3 "planning without action" stop). Keeps the call from ever
     running; ends some runs.
   - **C. Guard hops outside the cap, no other bound.** The model decides
     when to stop; a model that repeats forever holds the request (and the
     slot) until the harness's own timeout. Unbounded.
   - **D. Count guard hops in the tool-turn cap, forward at the cap.** As A,
     with the existing cap as N (10 / 20), and the landing at the cap
     replaced by forwarding for guard hops only.
   The module supports all four unchanged (`gate`, `hand_back`, the verdict's
   `vetoed` count); the choice is the hook's.

## 9. What fired in Atomic's own GAIA traces

The release bundle `gaia-l1-eval.tar.gz` (tag `gaia-l1-eval-2026-06-11`;
10,812,639 bytes = the release listing; sha256
`030a058196b9624f2edf708588e30bafb6d3c8728d40587d8e6ad2b52bbecb0e`),
downloaded by the coordinator on the operator's approval (2026-09-29),
extracted to the scratchpad, read only (nothing inside was run).
`python bench/atomic/gaia_traces.py <bundle>/gaia-l1-eval` produces every
number below from Atomic's own trace events (`loop_detected`,
`tool_invocation` with `toolTruncated` / `batchSize` / `details.deniedReason`,
`prompt_captured`, `query_rewriter`, `reflection`, `turn_finished`) and each
run's `matrix.jsonl` (the scored answers). The Hermes arm ships only its
matrix (31/53), no traces: nothing below can be compared per mechanism
against Hermes.

**The three Atomic runs are different agent versions** (their own caveat),
all at temperature 0.2, top_p 0.95, top_k 40, seed 10947100:

| run | model | agent | git | solved | calls | empty answers |
|---|---|---|---|---|---|---|
| atomic-agent-L1 (**the 37/53**) | Qwen3.6-35B-A3B-UD-Q4_K_XL | 0.1.36 | cc5c7d4 (dirty) | 37/53 | 458 | 4 |
| qwen3.5-9b | Qwen3.5-9B-Q4_K_M | 0.1.47 | 6e78ec0 | 28/53 | 394 | 2 |
| gemma4-12b | gemma-4-12B-it-qat-UD-Q4_K_XL | 0.1.41 | 0ef9a1b | 24/53 | 501 | 23 |

### 9.1 The loop detector

Events (`loop_detected`, level/detector as logged; 0.1.36 logs neither --
all three of its events have count 3 and the notice text of the repeat
warning: "You called `os.web.fetch` with the same arguments 3 times in a
row and neither the result nor the world snapshot changed. This is a
no-progress loop. Change strategy BEFORE calling any tool again: ..."):

| run | repeat warn | wandering warn | veto (critical) | breaker | calls refused |
|---|---|---|---|---|---|
| 35B (0.1.36) | 3 (unlabelled: `os.fs.read` 1, `os.web.fetch` 2) | -- | 0 | 0 | 0 |
| 9B (0.1.47) | 8 | 15 | 1 (`os.web.fetch`) | 5 (wandering, all `os.web.search` at 12 distinct) | 6 |
| 12B (0.1.41) | 22 | 9 | 16 | 1 (no_progress, `os.web.fetch`) | 17 |

What the model did at the next step (its calls vs the flagged call, by
tool + exact arguments):

| run | event | changed | repeated the same call | turn ended |
|---|---|---|---|---|
| 35B | warn (3) | 3 | 0 | 0 |
| 9B | repeat warn (8) | 6 | 2 | 0 |
| 9B | wandering warn (15) | 14 | 1 | 0 |
| 9B | veto (1) | 1 | 0 | 0 |
| 9B | breaker (5) | -- | -- | 5 |
| 12B | repeat warn (22) | 16 | 4 | 2 |
| 12B | wandering warn (9) | 7 | 2 | 0 |
| 12B | veto (16) | 12 | 4 | 0 |
| 12B | breaker (1) | -- | -- | 1 |

Tasks with at least one loop event, solved vs failed (Fisher exact,
two-sided, on the 2x2 against the tasks without one):

| run | with an event: solved / failed | without: solved / failed | p |
|---|---|---|---|
| 35B | 1 / 2 | 36 / 14 | 0.21 |
| 9B | 5 / 12 | 23 / 13 | 0.037 |
| 12B | 2 / 15 | 22 / 14 | 0.001 |

By kind: every task with a veto failed (12B: 0 / 7; 9B: 0 / 1), and so did
every task with a breaker (9B wandering: 0 / 5; 12B: 0 / 1). What the data
cannot say: whether the events hurt, helped or only mark hard tasks -- there
is no run with the detector off, and the tasks it fired on differ from the
rest.

### 9.2 The compressor (`toolTruncated`)

| run | results cut | by tool | cut shell results: size shown | next step after a cut shell result |
|---|---|---|---|---|
| 35B | 40 | vision.describe 18, os.shell.run 11, reply 10, os.web.fetch 1 | median 399 chars (max 399) | shell again 7, other 4 |
| 9B | 54 | os.web.search 33, os.shell.run 10, vision.describe 5, reply 3, os.web.fetch 2, os.fs.list 1 | median 399 | shell again 6, other 4 |
| 12B | 76 | os.shell.run 40, browser.click 10, os.web.fetch 10, browser.navigate 5, browser.read_aria 5, other 6 | median 399 | shell again 34, other 4, ended 2 |

In all three versions a cut shell result was the compressor's old 400-char
default (median 399 characters) -- the "before `shellToolResultCharCap`
existed" behaviour their AGENTS.md describes; the 16,000/8,000 caps our
`result_compress.py` copies from commit 52f90e55 are LATER than every GAIA
run. Tasks with a cut result: 35B 8 solved / 8 failed vs 29 / 8 without
(p 0.054); 9B 4 / 14 vs 24 / 11 (p 0.003); 12B 5 / 15 vs 19 / 14 (p 0.026).
The same limit as 9.1: no run without the compressor.

### 9.3 Parallel batches

| run | steps with >1 call | batch sizes | tasks with a batch: solved / failed | without | p |
|---|---|---|---|---|---|
| 35B | 24 | 2: 21, 3: 2, 4: 1 | 4 / 6 | 33 / 10 | 0.05 |
| 9B | 12 | 2: 10, 3: 1, 6: 1 | 2 / 5 | 26 / 20 | 0.23 |
| 12B | 42 | 2: 36, 3: 2, 4: 4 | 2 / 10 | 22 / 19 | 0.045 |

### 9.4 Memory

In all three runs the memory fabric delivered **nothing** to the model:
0 `### recalled` sections in 1,285 captured prompts, 1 `memory.*` tool call
(12B), the query rewriter `skipped_no_history` on every step (1,286), and
every end-of-turn reflection `aborted` (129). Each task ran in a fresh state
directory (their harness), so there was nothing to recall. The "memory
fabric ON" difference from Hermes in section 1 was, on this evidence,
inert in the scored runs.

### 9.5 Other counts

Parse retries (their grammar's array failed to parse) 4 / 7 / 9 and
errors 4 / 4 / 16 (35B / 9B / 12B); `turn_finished` reasons: 35B reply 49,
failed 2 (2 tasks without one); 9B reply 50, failed 2, max_steps 1; 12B
reply 30, failed 8 (15 without one).

### 9.6 What this adds to section 1

For the 37/53 run specifically: the loop detector warned 3 times and never
refused a call; the memory fabric recalled nothing; the compressor cut 40
results (11 of them shell output, to ~400 characters); 24 steps ran 2-4
calls at once. None of these was switched off in any run, so none of them
can be credited or blamed for the 37 vs 31 from these traces. The veto and
the wandering breaker fired only in the smaller-model runs, and only on
tasks that were failed.
