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
old clients. The tools API is `:1235`, Laya is `:1237` and SearXNG (web search,
`docs/SEARCH.md`) is loopback `:8888`. The watchdog supervises six services:
llama-swap, proxy, tools-api, laya, worker and searxng.

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

**The surface** (one model, one cache: operator, 2026-09-24):

| who sees it | tools |
|---|---|
| MCP clients (`code_search.TOOLS`, `:1235`, unchanged) | `find_by_meaning`, `find_definition_opt`, `find_references`, `find_by_pattern`, `read_file_range`, `summarize_text`, `describe_index`, `run_check` |
| main -- the model the client talks to (`proxy.main_tools`) | the CLIENT's tools, untouched and first, plus only `generate_image` and `describe_image` where offered (rows below), and `think_deeply` at `xhigh` and `max` (below). No code tool, no work-log tool, no `check_code`. |
| main, at tiers `xhigh` and `max` (`deep.think_tool_offered`; Phase 0.6) | `think_deeply` (`deep.THINK_TOOL`): the one non-image tool of ours on main, the model-chosen deep-thinking trigger. Arguments: the question, and what was tried. The proxy runs it as a hidden hop (below, "Deep thinking's triggers"). Offered by the tier's own allowance and not forced off, decided on a conversation's first request and KEPT for the conversation (a tool list that changes between turns changes the cached system block; a first request is always decided afresh, so benchmark arms that share opening messages do not inherit each other's list); a call on a request that forbids deep thinking gets `DEEP_THINKING_OFF`. Never on a header-forced arm at a lower tier. KNOWN and kept (pre-deploy review, 2026-09-24): the offer flips from no to yes when a conversation's tier rises to `xhigh`/`max` mid-way, which changes the system block once and costs that request its prefix. |
| the second brain (`proxy.deep_thinking_tools`; `shomen.run`) | the eight above, plus `record_step` and `read_rings` (`INTERNAL_TOOLS`, scoped to the conversation's lineage), `generate_image` and `describe_image`, and (Phase 0.6, `mcp/research_tools.py`) `find_skills` (the skill store; cite `skill:<id>`), `find_in_knowledge_base` (docs/, AGENTS.md, README.md and the conversation's work log; cite `path:line`), `read_web_page` (a pinned fetch: every hop -- robots.txt and each redirect -- resolved once and connected to at that address, refused unless `ip.is_global`, one deadline and a byte cap; only a URL a `search_web` result returned in the same run or the user gave, through a guard against long, high-entropy or secret-shaped values and conversation text; refusals recorded; cite the URL, labelled `(web)`) and `search_web` (the local SearXNG, `YAMADORI_SEARCH_URL`, loopback only; `categories=it` for code; at most `SEARCHES_PER_RUN` = 3 per run, a choice from the engines' rate limits; a query carrying code or a secret is refused before it leaves; down -> `SEARCH_UNAVAILABLE`, retryable, the remedy). Never `think_deeply` (it would recurse). Never the client's tools: what the harness read is in the conversation already, and a file it would need becomes the hand-off's NEXT STEP, for main to read with the harness's own tool. |
| off by default | `delegate_investigation`. Only with `YAMADORI_DELEGATE_TOOL=1` or a tier's `delegate` flag. It is a benchmark arm; main runs it as the runner's `investigate` job. |
| main and the second brain, only when `YAMADORI_IMAGEGEN_URL` is set | `generate_image` (`mcp/images.py`). Every tier -- a capability, not a gate (operator, 2026-09-23) -- even when the code tools are withheld, and to deep thinking for mockups and designs (its markdown crosses back in the finding; `describe_image` below is how the text-only model looks at what it drew). Two image models, `base` (20 steps) and `turbo` (Viggle 4-step): the caller's account preference picks (dashboard SETTINGS, `PUT /dash/api/settings/image`), else `YAMADORI_IMAGEGEN_DEFAULT`, else `turbo` -- the built-in default since 2026-09-24 (operator; the launch scripts also set it; 23.6 s vs ~108 s; quality vs base not yet compared by `bench/imagegen/compare_turbo.py`). The model has no say; `x_yamadori.images` records the model, steps and where the choice came from. See `docs/IMAGEGEN.md`. |
| main and the second brain, wherever `generate_image` is offered (deep thinking included), and on every tier when the request carries an attached image; off with `YAMADORI_VISION=0` | `describe_image` (`mcp/vision.py`): sends one image and a question to `bonsai-vision` (the 27B + mmproj on the A4000) through `mcp/model.py`, in the image lane, and returns the answer as text. It reads only this request's attached images (by id `image-<10 hex>`) and our media store (by sha, with a verified signed link or an image made in this request). It never fetches a URL and never opens a path. `proxy.prepare` replaces each attached image part with a placeholder naming its id; an image part carrying OUR signed /media link (signature and expiry verified, host `YAMADORI_PUBLIC_BASE`'s, the request's own, or loopback) is read from the media store as an attachment -- how Hermes' `vision_analyze` looks at what we drew (2026-09-24; offline tests only, the live check in `mcp/test_live_stack.py` images group has not run). `x_yamadori.vision` and `x_yamadori.attachments` record calls and attachments. **Run live once, on an attached image** (2026-09-24, `mcp/test_live_stack.py --only images`, n=1: a correct answer, 5.9 s, 168 prompt tokens for a 256x256 PNG). The draw-then-look check in `docs/IMAGEGEN.md` "Seeing: `describe_image`" has not run. The A4000 VRAM concern there was measured in the same test: 308 MiB free at the peak. |

Deleted 2026-09-24: `bind_project_context` (it pinned versions for tools main
no longer has) and the `check_code` TOOL (the proxy checks code itself --
`mcp/tool_code.py` for client writes, `code_check.review_answer` for an
answer's code). The proxy writes the work log itself (`proxy._log_turn`: the
client calls the model made, what the checks found, what deep thinking and
fan-out handed back) and re-injects it on the first user turn after a
compaction. `proxy.OUR_NAMES` is what the proxy executes itself (18 names
since Phase 0.6).

Fan-out is **not a tool**: `mcp/selection.py` turns it on per request, and
the tier (`mcp/tiers.py`) only says what is *allowed*. **Deep thinking runs
on four triggers** (Phase 0.6, operator 2026-09-24; section "Deep thinking's
triggers" below), at `xhigh` and `max`: the model's own `think_deeply` call,
struggle the proxy detects, a known-hard area, a large task kickoff. It is no
longer limited to `library_question`, and Laya is NOT consulted for it (a
separate evaluation decides Laya vs Tev1). The regex + symbol lookup + Laya
path survives only as selection's LEGACY path (no route, no trigger), for
the offline evaluators that replay it. Every decision -- and every
non-decision -- rides on the response as `x_yamadori` (`selection`, `deep`).

**A client's own side calls get the bare model** (`selection.utility_call`):
no client tools, one exchange, and a reply fixed by a contract (a closed
one-word/one-of/JSON/yes-no form addressed to the reply, `response_format`
JSON, or "summarise this conversation"). The proxy overrides the tier to
`minimal` and adds nothing; the call has no session (`x_yamadori.utility`,
`tier_overridden`). Replayed by `mcp/test_utility.py`: 41/41 Hermes side calls
in the corpus, 0 of 954 task turns and 0 of 4,317 benchmark prompts
misclassified. A header `{"utility": true|false}` forces it -- except that a request carrying an image is never a utility call (the bare text model cannot see; 2026-09-24). Each conversation
is pinned to a llama-server slot (`mcp/slots.py`, `id_slot`), side calls use
the one slot never pinned, the second brain (deep thinking, fan-out's B/C,
fix-ups, summaries) always uses its own RESERVED slot, `slots.helper_slot()`
= n-2, the same number in every process, never a conversation's (2026-09-24,
#10/#11 in `docs/SELF-IMPROVEMENT-LOG.md`; so two conversations hold pins
of four slots), and `x_yamadori.cache` reports prompt tokens reused vs
processed; `x_yamadori.warm_before` reports the previous turn's warm. A conversation that a harness compacts keeps its work log,
tools and slot (`proxy._continue_after_compaction`).

## One model, one cache (operator, 2026-09-24)

A client -- Hermes, OpenCode, Pi, anything -- sees ONE standard model. What
the proxy adds never breaks the pinned slot's prompt cache; the second brain
does the heavy work and folds a short result back; main's context stays
small. The mechanics this rests on were measured before anything was built
(`docs/SELF-IMPROVEMENT-PLAN.md` Phase 0.5, STEP 0; mechanism probes on the
transient slot, each run twice): after an assistant PREFILL the next request
reused 976 of 995 prompt tokens; after a zero-token WARM it processed exactly
its 23-token tail (93 without the warm); and a request that diverges
anywhere before the end of the slot's sequence falls back to a context
checkpoint (448 tokens in, wherever the edit was -- inferred as
checkpoint restore, not confirmed from the log). So a request must EXTEND
the slot's last sequence -- with one deliberate exception since 2026-09-24:
past reasoning passes through (see "The ledger" below), so each request
diverges at the previous assistant turn's think block and relies on the
checkpoint at the previous prompt's end; its live cost is not measured yet.

**What main's context holds:** the client's messages, what the LEDGER puts
back, the static addendum, and main's own generations. Nothing else -- no
repair turn, no tool-call round, no weigh turn, no hand-off as a user turn.
`proxy._run_turn` is the one turn implementation both paths run
(`complete()` drains it; `stream_body()` streams its events).

**The ledger** (`proxy.ledger_restore` / `ledger_record_turn` /
`ledger_seed`; storage in `nebari.py`, table `additions` in
`index/nebari.sqlite3`). Everything the proxy added, per message, re-added
byte for byte on every request:

| addition | keyed by | when decided |
|---|---|---|
| a user turn's injection: skills / hints, library definitions, the work log after a compaction | a hash of the conversation up to and including that user turn | ONCE, on the request whose last message is that turn (the first request after a compaction may add the work log); replayed ever after |
| a tool result's injection: library use (the held packages the conversation uses, #19) | a hash of the conversation up to and including that tool result | ONCE, on the request that ends on it |
| a call turn's delivered content (the check note) | its tool-call ids | when delivered |
| hidden internal hops (image tools and `think_deeply` the proxy ran inside the turn, the hand-off as the tool result), their reasoning emptied | the final visible turn, keyed by the content the client STORES (every byte streamed, #10) | when delivered; expanded back in place on replay |
| a second-brain job's concept seed | the request (hash of its last message) and the job | on the job's first run; a retry or re-render reuses it |

**Past reasoning is NOT in the ledger: it passes through** (design change,
coordinator/operator 2026-09-24). What a client sends is what the model sees:
a client that drops past reasoning gets none back, one that echoes it gets its
echo, unchanged (a template marker inside an echo is counted in
`x_yamadori.ledger.restored`, not scrubbed). Evidence: Hermes strips
`reasoning_content` from every replayed assistant turn for a provider that
does not require the echo (`agent/message_sanitization.py`
`apply_reasoning_content_policy`; the echo flag is opt-in), and the reference
run -- plain llama-server + Hermes + Bonsai 2, 5 h, 328k tokens written,
finished at 125k context -- ran that way and kept the thread; restoring it
cost 6-10k tokens of context per step (V0 pilot, ledger rows of 25-41k
chars). Reasoning lives within one request only: its own hidden hops and a
deep-thinking hand-off prefilled as reasoning (the visible opening carries
the conclusion forward). **The cache cost is NOT measured yet:** the slot
generated the previous turn's reasoning and the client sends none, so every
request diverges at the previous assistant turn's think block; the processed
tail is that turn (as the client sends it) + the tool result + the new part,
and everything before it rests on the hybrid model's checkpoint at the
previous prompt's end. `mcp/test_ledger.py` gates the prefix offline; the
live `cache` and `agent_loop` tests measure the tail.

Only what the proxy or the model produced is stored; the caller's messages
are only hashed (nebari's standing rule: "Not kept: the caller's code").
Every query names the account. Memory is a read-through LRU in front of the
table (`YAMADORI_LEDGER_MEM_MB`, 64). Hops follow
`YAMADORI_LEDGER_PERSIST_REASONING` (default on; the name predates the
pass-through). Eviction is by SIZE, not
nebari's 36 h TTL: whole sessions, least recently seen first, past
`YAMADORI_LEDGER_ACCOUNT_MB` (256) per account, then
`YAMADORI_LEDGER_TOTAL_MB` (2048) in all, and anything older than
`YAMADORI_LEDGER_MAX_DAYS` (30) -- all three CHOICES (operator, 2026-09-24),
not measurements; pruning runs in a background thread at most once a minute.
`mcp/test_ledger.py` prints the bytes each turn adds (10-370 bytes a turn on
its fixture, whose reasoning is a few words; real reasoning is larger) and
gates prefix stability with the served template itself.

**The static addendum** (`proxy.ADDENDUM`): one short, fixed decision table
at the end of the client's system text -- what the second model does and the
phrases it uses -- added only where every row is true (where the fixup runs:
`high`, `xhigh`, `max`). Where main has `think_deeply` it carries one more
row (`proxy.ADDENDUM_THINK_ROW`, `proxy.addendum_text`), kept for the
conversation like the tool. No prohibition. Its wording is a choice.

**Library definitions** (`proxy._library_definitions`; operator decision
2026-09-24, an UNMEASURED choice): a user turn classified
`library_question`, at a tier with library help, where the gate says
something held can answer and deep thinking does not run, gets the
definitions of the names it uses that a held source defines (the router's
own lookup, then `find_definition_opt`), capped at 3 names and 3,000
characters (choices), as a tail injection recorded in the ledger.

## The second brain

ONE runner, `shomen.run(job, ...)`, on the one helper lane
(`admission.helper_lane`, `HELPER_LANES = 1`). It replaced three paths:
`shomen.investigate`, fan-out's B/C generation, and the repair loops that
ran on main. Our tools live only here.

| job | what it gets | what comes back |
|---|---|---|
| `investigate` | the question and context, our tools | the four-section hand-off (FACTS / SEARCHED, FOUND NOTHING / OPEN QUESTIONS / NEXT STEP) |
| `plan` (Phase 0.6, task kickoff) | the task's spec and earlier user turns, our tools, `shomen.PLAN_SYSTEM` | a four-section plan (FILES / ORDER / KEY DECISIONS / RISKS; `shomen.plan_handoff`), a KEY DECISION's citation checked like a fact |
| `alternative` | the task (fan-out's candidate B) | an answer, graded by the code check (`fanout.analyse`) |
| `tiebreak` | the task, both candidates and their check results (C) | an answer, graded the same way |
| `fixup` | ONLY the code, its exact errors and the user's request -- a client write's file or edit, or an answer's fenced block | the repaired code only when `shomen.fix_rejection` accepts it (else the original, and why), up to `tool_code.REPAIR_ROUNDS` (3) rounds at the request's own effort |

Every job carries a **concept seed** (`mcp/concept_seed.py`) in its USER
message, drawn once per (request, job) and recorded in the ledger, so a
retry, re-render, warm or compaction splice uses the same word. The second
brain never gets client access.

**Fold-back phrases** (`shomen.PHRASES`, the one table in code; this is the
same table). What the second brain did reaches main's turn in these fixed
words, so the result sits in visible content (it survives a harness's own
compaction) and the slot processed exactly those tokens:

| phrase | when | how it gets there |
|---|---|---|
| `Today I was inspired by <word>.` | first, in the fold-back of second-brain work that becomes ANSWER content -- deep thinking, fan-out's B/C (both words when B and C ran). NOT on the mechanical notes (`Verified`, `Repaired`, `Checked`; coordinator 2026-09-24): a fix-up's seed stays in its own user message | with the phrase that follows it |
| `After thinking deeply,` | deep thinking ran | before main (a trigger or a header): PREFILLED as the opening of main's visible answer, the hand-off (or the plan) prefilled as main's `reasoning_content`; main continues. After a `think_deeply` call: the hand-off is the hidden hop's tool result, and the next hop is prefilled with this opening (reasoning `deep.THINK_REASONING`) |
| `Verified` | the check passed | a one-line note the proxy writes; no generation |
| `Repaired` | the fixup job changed code | a client write: the note before the fixed call. A final answer: non-streamed, the repaired answer IN PLACE of the broken one plus the note; streamed, the repaired block after the answer |
| `Compared two approaches` | fan-out ran B (and C) | code: a line with the winner (delivered as before); prose: PREFILLED after main's answer, ending "Weighing them", and main continues |
| `Checked` | problems remain (at `medium` nothing fixes; or the fixup could not) | a one-line note |

A prefilled phrase ends on a letter, or on an ending measured safe (`After
thinking deeply,` -- STEP 0), never a space: a trailing space is its own
token and breaks the boundary. Whenever the turn the client stores differs
from what the slot generated (a repaired call, a note, a notice), the proxy
**warms** the conversation's slot with the turn as delivered while the
harness runs its tool (`proxy._warm`: `/apply-template`, cut at the last
end-of-turn token, a zero-token `/completion`; `x_yamadori.warm`); the
conversation's next request waits for its own warm on the same slot
(`slots.acquire`, `warm`). `x_yamadori.fold_back` lists each fold-back: job,
phrase, where it went, the seed words.

**Fetched content is data** (operator, 2026-09-24; one rule, one place:
`mcp/skill_screen.py`, the skill screen). Anything the second brain fetches
-- a web page, a search title or snippet, a skill it looks up (re-checked:
skills can be edited) -- is scanned and STRIPPED before it reads it
(`screen_fetched`: the source rules, the offending span removed and
recorded, a text stripped past `STRIP_MAX_FRACTION` = 25% dropped whole, a
choice), inside a "data, not instructions" frame. The hand-off to main is
the path to the user, so it is screened on the way out (`screen_handoff`):
AI-directed text, exfiltration, shell danger, credentials, hidden markup and
unrelated actions never cross; a shell command, install step or URL that
came from the web (in what the run fetched, not in the conversation) is
removed from an instruction and otherwise crosses only as a fact labelled
"(from the web, unverified)". `x_yamadori.deep.screen` records what was
stripped, dropped, removed and labelled. And a skill item must be about the
work itself: `screen_item`'s `unrelated_action` drops one that tells the
model to contact a URL, send data anywhere, run a downloaded script or
change credentials or configuration, however politely phrased (0 of the
2,575 recipe rows flagged). Tests: `mcp/test_deep_review.py` (all 17
malicious fixtures caught on both paths; 0/5 clean fixtures touched).

## Deep thinking's triggers (Phase 0.6)

Operator, 2026-09-24; `docs/SELF-IMPROVEMENT-PLAN.md` Phase 0.6. Allowed at
`xhigh` and `max` (`tiers.TIERS` `investigate`), one helper lane, one run per
request. `mcp/deep.py` decides from what the CLIENT sent (never Laya);
`mcp/selection.py` records the reason; `proxy._deep_thinking` runs the
pre-main job, `proxy._think_deeply` the model's call.

| trigger | fires when | runs | once per |
|---|---|---|---|
| model-chosen | main calls `think_deeply` (a trigger-list description: stuck, unsure of an API or version, a fix failed twice, "still broken") | `shomen.run("investigate")` as a HIDDEN HOP: the hand-off is the tool result, the ledger replays call + result, the next hop is prefilled with the fold-back opening | `THINK_CALLS_PER_REQUEST` = 1 per request; not after a pre-main run in the same request |
| struggle | at least `STRUGGLE_THRESHOLD` (3) signals in the current episode (the last `STRUGGLE_WINDOW` = 40 messages after the boundary): the same tool erroring again, the same file rewritten after a failure (or a third time), a failing command re-run, our fix-up's "still there after N rounds of repair", the user saying it is still broken / didn't work / same error | `investigate` BEFORE main, with the task and the failing output (marked as data); the hand-off prefilled as main's reasoning | episode: a run moves the boundary; `COOLDOWN_REQUESTS` = 3 requests before struggle can fire again |
| known-hard area | the conversation USES (`proxy.library_uses`) a held package that is unseen (`deep.unseen`: the PACKAGE was first published after `YAMADORI_MODEL_CUTOFF` = 2025-12-31; or the used version is a prerelease, or a new major against the last release before the cutoff; or `YAMADORI_UNSEEN_PACKAGES`) -- a minor or patch of a long-lived package is not; or an injected skill declares `escalate: true` in its frontmatter | `investigate` before main, on the package's API as the task uses it | package (or skill) per conversation |
| task kickoff | a new task -- the first user turn, or a user turn after a finished answer; not a harness notice, not "still broken" -- whose spec is at least `KICKOFF_TOKENS` (1,500 at ~4 chars a token; the Octopus V0 spec was ~2,246) | the `plan` job before main; the plan prefilled as main's reasoning (`deep.PLAN_HEAD`); main starts acting | user turn |

Priority: a header forcing it off > forcing it on > struggle > kickoff > area.
A trigger that finds the helper lane BUSY is deferred: no trigger fires for
`DEFER_REQUESTS` = 3 requests, and main waits for the lane at most once per
episode (after that a trigger only takes a free lane). A compaction starts a
new EPOCH of message indexes and keeps the episode, cooldown and coverage; a
compaction continuation is never a kickoff; "still broken" counts only on a
user turn that follows an answer (a bug report as the first turn is the
task). A tool result with an exit status decides by it (0 is success; exit
1 with no output is grep's "no match"), and "failed" counts only with a
non-zero count. The state is changed under a per-conversation lock.
**Every number above is a CHOICE, unmeasured.** Unseen packages
(coordinator's revision, 2026-09-24: a version's own date flagged every
three.js conversation): each package's registry history -- its first
publish and its stable releases' dates -- is stored PER PACKAGE in
`index/packages/registry_history.json` (`deps.record_history`;
`deps.index_package` records it, `deps.py published name@version` backfills).
Backfilled 2026-09-24 for every held package: unseen are `@pmndrs/glyph`
(first published 2026-08-15), `three-flatland` (2026-02-25) and the
prereleases `@react-three/fiber@10.0.0-alpha.5` and `@react-three/drei@11.0.0-alpha.7`;
`three@0.185.1` (0.182.0 before the cutoff), `koota@0.6.6` (first published
2024-10-15) and the rest are not. A 0.x minor is not a new major. The per-conversation state (episode boundary, cooldown,
areas and kickoffs done, the `think_deeply` offer) is in the ledger
(`deep:<lineage>`), so a compaction-linked continuation shares it.

**The self-improvement loop.** Every request of a conversation leaves ONE
row in `deep_decisions` (the corpus database): the trigger or `none`, the
signals (names and counts, never text), the thresholds in force and their
source, whether deep thinking ran, the hand-off's size and the names it gave
main. Each later request of the conversation OBSERVES the open rows against
what the client sent back (did the struggle stop, did the check pass, was a
file rewritten again, "still broken"?, was the hand-off used) and, after
`OUTCOME_REQUESTS` = 5 observations or at once on a clear outcome, labels
them: `helped` / `not_helped` / `wasted` (a hand-off none of whose names main
then used) for a run; `fine` / `missed` (a struggle under way, no deep
thinking, then "still broken" or more signals) / `escalated_later` for a
non-decision; `skipped` when the helper lane was busy. ONE INCIDENT, ONE
LABEL: a non-decision label counts once per episode (the other rows of the
episode are `same_episode`), and a row inside a run's cooldown is
`in_cooldown`, not `missed`. Test traffic is recorded and never learned
from; `corpus.account_traffic` fails CLOSED (an unreadable registry means
test), and `hermes-dogfood` is client traffic. The writes (observe +
record, one transaction) run on one background thread through a bounded
queue, never on the response path. The idle-time job `deep.learn`
(`mcp/deep_learn.py`, cpu lane, scheduled by the worker like `skill.learn`)
moves `struggle_threshold` by 1 (bounds 2-6) and `kickoff_tokens` by 25%
(bounds 500-8,000) when at least `LEARN_MIN_N` = 5 of the deciding label
accumulated since that parameter's last change; each adjustment records its
old and new value and its n, and is reversible (`POST
/dash/api/deep/revert {id}`); an environment variable pins a parameter.
It PROPOSES `think_deeply` description variants from missed rows' phrasings
and never arms one (a description is a prompt; PROTOCOL). `x_yamadori.deep`
carries the decision, the thresholds (default / env / learned with n), the
cooldown, `think_deeply`'s calls and the row id; `GET /dash/api/deep` the
thresholds, adjustments, proposals, runs per day by trigger and the labels.
The dashboard panel is to follow. Tests: `mcp/test_deep.py` (the triggers,
the records, the learner, the tools), `mcp/test_ledger.py` `[think]`,
`[struggle]`, `[kickoff]` (end to end through the served template: each
request after one extends the slot). **Not yet run live.**

## The code-work router and tool-call repair

**One class per request** (`mcp/route.py`), decided once in `proxy.prepare`
and recorded as `x_yamadori.route {class, because, signals}`. First rule that
fits wins: `utility` (`selection.utility_call`) → `agent_step` (ends on a
client tool result; a harness notice or bare "continue"; `acts_locally`) →
`code_edit` (the prompt carries code that PARSES, and the instruction asks
for work on it) → `code_generation` (a write verb + code noun, "export
`name`", "write a Rust ...") → `library_question` (a question, and the gate
offered for a reason about it, or a held source defines a name it uses) →
`prose`. Features read the class: **fan-out and the repair pass on the code
classes only; the definitions injection on `library_question` only.** Deep
thinking no longer reads it (Phase 0.6: triggers, on any class). A header
that forces one still forces it.

Measured by `bench/route/eval_route.py` on `bench/route/labels.jsonl`:
1,016 corpus turns (196 Hermes, 200 SWE-agent, 612 benchmark/probe, 8
other), 189 unique user-speaking requests labelled by hand against the
rubric in that script, 341 agent steps whose tool-result ending is
**reconstructed** (the corpus did not record it; rows from 2026-09-24 do). A
blind second pass on a seeded 20% (203 turns) agreed on 202. **In-sample**:
the rules were adjusted while reading these misses.

| truth (n) | recall | precision |
|---|---|---|
| utility (43) | 1.000 | 1.000 |
| agent_step (371) | 0.995 | 1.000 |
| code_edit (160) | 0.812 | 0.963 |
| code_generation (223) | 0.978 | 0.879 |
| library_question (60) | 0.900 | 1.000 |
| prose (159) | 1.000 | 0.952 |

**Misroutes (utility/agent_step → a code class): 0.** Code vs not: 383/383
code turns routed to code, 0 non-code turns sent there. The code_edit ↔
code_generation confusion is mostly LiveBench starter code the corpus cut
off at 2,000 characters; both classes run the same pipelines. Benchmark
sets: LiveCodeBench 342/342, `bench/domain` 100/100, react 40/40, type
challenges 183/183 route to code. **What changed because of it** (tier max,
Laya absent): deep thinking unforced on the domain bench went from 80/100,
9/40 and 166/183 tasks to 0 (the header arms that force it are unchanged);
context-economy 25/26 and held-out 91/120 are unchanged; design questions
(prose) no longer fan out. None of the 196 Hermes turns is a code request:
Hermes writes code through tool calls, which is what the next paragraph is
for.

**Tool-call repair** (`mcp/tool_code.py`; never for a utility call). When a
generation ends with CLIENT calls that write or patch a file, the code is
checked before anything is forwarded: whole files by the parser, ruff
`E9,F63,F7,F82` for Python, and the formatter's own parser; edits (old/new
strings, diff hunks, SEARCH/REPLACE) block only when the text they replace
parses and the replacement does not, otherwise they are flagged. Detection: a
known-names table (Hermes, MCP filesystem, Anthropic's editor, Claude Code,
Gemini CLI, Cline/Roo, OpenCode, Codex, Continue; the table marks which
entries were verified from source), then argument shape; an unrecognised
call with a code-sized string is logged in `x_yamadori.tool_code.unknown`.
At `medium` (`tiers.check_code_offered`) what the check found is only
NOTED. At `high` and up (`tiers.repair_on`) each blocking unit goes to the
second brain's **fixup** job with ONLY its code, its errors and the user's
request, at the request's own effort, up to `REPAIR_ROUNDS = 3` (not
measured; the final-answer repair is the same job). The repaired version is
written back into the call ONLY when it came from a closed fence, the reply
was not cut off (`finish_reason: length`), it has no errors left and it is
plausibly complete (at least `shomen.FIXUP_MIN_KEEP` = 1/2 of the
original's non-blank characters -- a choice, not measured); otherwise the
model's own content goes untouched and the note says the repair was not
used, and why (`shomen.fix_rejection`). Main generates the call once and never sees a
failed attempt. A unit inside a diff or a multi-file patch has nowhere to
write a fix and is only noted. **Change what the model wrote only when it
is broken** (2026-09-24, #24 in `docs/SELF-IMPROVEMENT-LOG.md`): the client
receives the calls exactly as written, except the lines a repair of real
errors required (the fixup is told to change only those and keep every other
line, formatting included). Formatter output is never applied -- it rewrote
every whole-file write (5-65 lines each, V0 pilot), the model then patched
with its own pre-format text and the patch failed -- and appears in the note
only as information. The note before the calls is mechanical, in the
fold-back phrases and with NO concept-seed line (the seed stays in the fixup
job's own user message): `Repaired js/game.js (javascript): 2 syntax errors
fixed in 1 round.`, `Verified app.py (python): parses, lint clean; ruff
would change 3 lines; sent as written.`, "Checked" when problems remain.
Nothing is changed after the client has the call, and the note claims syntax
and lint of each file on its own, nothing more. The proxy then warms the slot with the turn as delivered.
`mcp/test_tool_code.py`, `mcp/test_stream.py` and `mcp/test_route.py` hold
the checks. **Not yet run live.**

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

Each tool loop is capped per context at the vendor's 10 tool turns (PrismML
Bonsai-demo `agenticMaxTurns`), 20 at tier `max` (`tiers.tool_turn_limit`,
used by `proxy.py` and `shomen.py`; operator decision 2026-09-23). The main
loop -- which since 2026-09-24 runs only for our tools on main (the image
tools, the delegate arm) -- and every deep-thinking run each count their own;
fan-out candidates and fixup rounds make no tool calls. Neither number is measured. At the cap the loop **lands**:
tools withdrawn, answer asked for, never a tool request returned as the
answer.
`x_yamadori.tool_turns` records `{limit, turns, hit}`. A hit alongside repeated
empty or error calls is a tool defect to fix, not a budget spent.

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
  Deep thinking (`shomen`) and fan-out both draw from the helper's 3/8, one
  second-brain job at a time (`admission.HELPER_LANES = 1`; in one request
  they run in sequence). Fan-out is sequential (`mcp/fanout.py`, operator
  decision 2026-09-23): the original answer is candidate A, the second brain
  writes B with the helper's whole 3/8, the code check grades the two, and
  only when that does not separate them does it write a tie-breaker C from
  both candidates and their check results. At most two contexts are live.
  Every second-brain run carries a fresh concept seed in its user turn, and
  its work always crosses back (next bullet). The split is the operator's decision of
  2026-09-22, not a measurement: it replaced a 1/2 + 2 x 1/4 split the same
  day (a second concurrent helper needed two conversations investigating at
  once), which had replaced a fixed `R_CAP=8192` breaker and a 60/25/15 split
  that had no measurement behind it (`docs/CONSTRAINTS.md` item 19). Natural thinking ran 682–2,826 tokens
  at n=7 (§1), so no measured result says a shorter thought is better.
  `config.yaml` launches with `--reasoning-budget 32768` and the same
  message, the server default for a request that sends no budget of its own.
- **The second brain's work is never thrown away** (operator decisions
  2026-09-23 and 2026-09-24). "Only the conclusion crosses" still holds for
  summarization and lookup. Work it DID crosses back as a distillation plus
  facts, folded into main's own turn (see "The second brain" above):
  - *Deep thinking* always hands off, in four fixed sections (FACTS, each
    ending `path:line` or labelled `(reasoning, not checked against
    source)`; SEARCHED, FOUND NOTHING; OPEN QUESTIONS; NEXT STEP),
    PREFILLED as main's `reasoning_content` for this answer; the visible
    answer opens with the seed line and "After thinking deeply,". Citations
    are still checked against what was retrieved; an unchecked fact is
    labelled, not dropped. No search: it crosses under "reasoning, no
    sources checked". Turn cap or helper-budget landing: the landing prompt
    requires the hand-off. Nothing written: the proxy builds one from the
    trace, labelled machine-built. `MAX_FINDING_CHARS` (6000) is the breaker
    and says when it cuts. `x_yamadori.investigate.handoff` has the counts.
  - *Fan-out, prose*: B's differing points are PREFILLED after main's own
    answer, in its own turn ("Compared two approaches: ... Weighing them"),
    and main continues -- it weighs them itself. The weigh turn (a user
    turn in main's context the client never saw) is gone (operator,
    2026-09-24). A continuation that fails leaves the points as a note.
    *Code*: the winner is delivered (in place on the blocking path, after
    the answer on a stream) with one line: why, and the APIs the losers used
    that it does not. `x_yamadori.fanout.handback`.
  The context-economy numbers in `mcp/shomen.py` (main context 8,326 ->
  2,410 tokens, n=26) were measured under the old conclusion-only rule and
  **must be re-measured** (`bench/context_economy.py`) before they are cited.
  The "material difference" thresholds in `fanout.handback` are choices, not
  measurements.
- **A client's compaction is part of the conversation it summarises**
  (`mcp/compaction.py`, `proxy._serve_compaction`; `x_yamadori.utility_kind`
  and `x_yamadori.compaction`). Two shapes. *In place* (the history resent
  plus a summarise turn, as Claude Code and Codex send it: not a utility
  call, so it keeps its session, tools and slot) is served on the LEDGER's
  rendering of the resent history, which puts back everything the proxy
  added, and is checked against the stored prompt (the last prompt the proxy
  sent for that conversation, and the turn it delivered); only when they
  differ is the resent history spliced onto the stored prompt (mode
  `ledger` / `spliced`). *Flattened* (Hermes' one user message, "TURNS TO
  SUMMARIZE: [USER]: ...") has its records mapped onto the stored prompt by
  text and its transcript replaced by a reference to that span (the
  iterative form's previous summary too). No match means the request goes up
  as sent, with the reason recorded; a flattened one then falls back to the
  transient slot by prefix affinity (`slots.acquire` `prefix`). A compaction
  THINKS at the conversation's own effort -- at every effort, medium
  included -- with the conversation's own sampling; thinking is off only
  where the conversation itself runs with it off, so the effort line is
  always the conversation's (interim defaults, operator 2026-09-24, until
  `docs/COMPACTION-RESEARCH.md`'s eval runs). Its thinking budget is
  `COMPACTION_THINKING` = 2,048: from `docs/CONSTRAINTS.md`, 5 of the 7
  self-finished thinking runs fit in 2,048, on coding prompts, n=1 each --
  not a compaction measurement. `tool_choice: "none"`, which llama-server
  does not render. The answer allowance is at least `COMPACTION_BUDGET`
  (5,120, the operator's figure; `YAMADORI_COMPACTION_BUDGET`) and otherwise
  the client's own target -- `max_tokens`, or Hermes' "Target ~N tokens" (up
  to ~12,192 at this pool; a cut summary makes Hermes discard it and compact
  again) -- bounded only by the pool (`tiers.compaction_budget`; the fixed 2x
  ceiling is gone). It never waits for the helper lane: its window is the
  pool less a running second brain's share (`admission.helper_active`),
  falling back to the main share when it does not fit. The advertised window
  stays the main share. **Not yet run live.** The mapping cannot be replayed
  from the corpus, which keeps 2,000 characters of each request (11 of its 12
  Hermes compactions are the iterative form, all preamble in that head).
- **A `finish_reason: length` is a budget event, never an answer.**
  `model.BudgetEvent`.

## The A4000's room

`mcp/gpu_room.py` decides what is loaded on the A4000 (operator,
2026-09-24: "if it fits with headroom fine, if it doesn't drop them and load
in what you need on use"). Every caller about to make llama-swap load an
A4000 model wraps that request in `gpu_room.use(model, upstream=...)`:
`images.generate`, `model.post` (vision), `code_search._post` (embeddings,
reranker), `scripts/index_code.py`. It reads `/running` and nvidia-smi by
UUID, sizes the model from `SIZES` (each row names its source; only
`imagegen` is measured), and if free − need < 1,331 MiB it unloads other
A4000 models least recently used first through `/api/models/unload`, or
refuses with `A4000_NO_ROOM` / `A4000_BUSY`. The main model and Laya are
never touched; a model in use (a lease) is never unloaded. A file lock
serialises it across the proxy, tools API and worker. `x_yamadori.gpu_room`
records every decision. llama-swap's groups remain the backstop. A new A4000
model needs a `SIZES` row (`mcp/test_gpu_room.py` checks the table against
config.yaml). Offline suites run with `YAMADORI_GPU_ROOM=0`. **Not yet run
live** (SELF-IMPROVEMENT-LOG #16).

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
- Tier ladder (`tiers.TIERS`): the client's `reasoning_effort` picks a tier,
  and each tier sets what is *allowed* (the selection engine still decides per
  request whether fan-out and deep thinking actually run). The same table is
  in `README.md` under "Effort tiers". Both are generated from
  `tiers.features` (the one feature matrix, which the dashboard's tier
  ladder also reads); `mcp/test_tier_docs.py` fails when a copy disagrees.

  | `reasoning_effort` | thinking sent | library help | skills | code check | fan-out | deep thinking | addendum | images | concept seed | adds |
  |---|---|---|---|---|---|---|---|---|---|---|
  | `minimal` | off | – | – | – | 1 | – | – | yes | – | thinking off, the vendor's instruct sampling, nothing of ours (fastest; least injection-resistant) |
  | `low` | medium | – | – | – | 1 | – | – | yes | – | nothing: the model as it ships, the benchmark baseline |
  | `medium` | medium | definitions | yes | note | 1 | – | – | yes | – | library definitions for a library question, skills (formerly hints; armed without review, screened), a note when a client write does not parse |
  | `high` | medium | definitions | yes | repair | up to 3 | – | yes | yes | yes | the second brain: code that does not parse is repaired, a second approach is compared, and the addendum says so |
  | `xhigh` | medium | definitions | yes | repair | up to 3 | allowed | yes | yes | yes | everything `max` has, at medium thinking: the effort-matched pair to `max` |
  | `max` | xhigh | definitions | yes | repair | up to 3 | allowed | yes | yes | yes | everything, at xhigh thinking |
  `code check`: syntax and lint; the model's code is changed only where a
  repair of real errors needs it -- formatter output is never applied, only
  reported (#24, 2026-09-24). `library help` also covers, on every route
  class from `medium` up, the held packages a conversation USES (imports in
  tool results and written files, manifests): LIBRARY USE (#19), an
  unmeasured choice.
  `images`: `generate_image` / `describe_image` are offered on every tier
  where `YAMADORI_IMAGEGEN_URL` is set. `library help` is the definitions
  injection (a choice, unmeasured); where deep thinking runs, it does the
  reading instead. `concept seed`: every second-brain job carries one. Because the tiers also raise thinking effort, a `minimal` vs `max`
  comparison mixes effort with tools; benchmarks report an effort-matched pair
  (everything forced off vs on at the same effort) beside the ladder.
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
   `python scripts/run_tests.py --live --key-file PATH` (or
   `YAMADORI_TEST_KEY=...`). This covers `mcp/test_live_stack.py` (every
   claimed feature; the map is `docs/LIVE-COVERAGE.md`),
   `mcp/test_tools_live.py`, `bench/test_laya_head.py --serve` and the
   bench suites that take `--live`. It judges the model's actual output
   through the door users use, prints every check's evidence, and treats a
   429 as NOT RUN: exit 1 on any failure, 3 when nothing failed but
   something did not run, 0 only when everything ran and passed. **If we
   don't have tests that exercise the real models, we don't have tests**
   (operator, 2026-09-24): a feature with no live test is unproven, however
   green the offline suites are.
   **A deploy is not good until `python scripts/deploy_check.py --key-file
   PATH` exits 0 after the restart.** It waits for every service and an idle
   card, runs the live suites (`run_tests.py --live --live-only`), and
   appends the verdict to `logs/deploy_check.jsonl`. Exit 3 (a 429) is not
   good either: run it again on an idle stack. The intrusive live tests
   (they restart the proxy) run only with `--maintenance`.
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

**Being replaced by E1** (`mcp/e1.py`, `docs/E1.md`): logistic heads on the
resident embedder's vector of the request. route_in v1 scores 104/120 on
held-out set 1 vs Laya's 80 (p=0.0001) and 110/141 on the new set 2 vs 86
(p=0.003). No Laya combination beat E1 alone, so the verdict is to retire
Laya. `YAMADORI_E1=1` puts E1 in Laya's place and takes Laya off the request
path. It is **off by default** until `mcp/test_live_stack.py --only e1`
passes; the retirement checklist is docs/E1.md §9. Until then, the rules
below still hold for Laya.

**Use Laya only where the state is FIXED and the options vary**, as a
`choice` over a small closed set. Its scores are not comparable across
different passages:

- Holding a passage fixed and varying the query separates real from nonsense
  by 0.497. **Unverified:** no script in the repo produces this number
  (`docs/LAYA-FACTCHECK.md`).
- Varying both collapses the gap to zero.

Never threshold a Laya score across queries.

It also silently drops the tail of a long `state`. The window is about 512
tokens, cut from the end: a question after ~620 tokens of preamble is
discarded, and every input then scores identically
(`bench/laya_factcheck/tail_probe.py`). Put the thing being judged first.

**`route_in` head.** It was retrained on 2026-09-22 on 289 labels
(`index/laya/route_in.json`; the previous head is in `index/laya/_backup_*`).
It was scored by `bench/eval_route_heldout.py` on 120 held-out
package-domain labels, which no training file matches:

| condition | investigate-vs-not |
|---|---|
| new head | 80/120 |
| old head | 64/120 |
| `selection.decide` | 89/120 |
| bare regex, no symbol lookup | 73/120 |
| rule + head live, disagreement escalates | 91/120 (p=0.79 vs 89, not significant) |

The head's features are Laya's CLS vector plus its three option logits. It
beats the bare regex (80 vs 73, not significant) but not the rule with its
symbol lookup, so it never decides alone. The live row is
`bench/laya_factcheck/live_route_check.py`; old vs new head is p=0.011.
Since Phase 0.6 (2026-09-24) the proxy's deep-thinking decision does not
consult it at all (triggers, above); these rows describe selection's legacy
path, which the evaluators still replay.
Deep thinking still injects findings that carry no citation
(`docs/LAYA-FACTCHECK.md`); since 2026-09-23 each such fact crosses labelled
as unchecked reasoning rather than passing as read.

## Where things live

- **Datasets.** `mcp/datasets.py` holds the stages and `mcp/jobs.py` is the
  durable queue. The lane limit is global across processes
  (`{gpu: 1, cpu: 4, net: 4}`). `mcp/worker.py` claims and runs the jobs.
  There is no review stage: review is optional and happens after the fact on
  the dashboard. The `clarify` stage is model-assisted. A licence is filled only
  from a verified verbatim quote, and a guessed licence is never proposed.
- **Skills** (replacing hints; operator, 2026-09-24). `mcp/skills.py` is the
  store and states, `mcp/skill_pipeline.py` the jobs (fetch → screen →
  screen_model → classify → distil → validate → arm, plus a scheduled
  watch), `mcp/skill_screen.py` the exploit screen, `mcp/skill_builder.py`
  the builder prompt and validator, `mcp/skill_classify.py` the applies-when
  vocabulary (artifacts, languages, frameworks, triggers),
  `mcp/skill_select.py` request-time selection, `mcp/skill_learn.py` the
  fallback records and idle-time learning, `mcp/skill_limits.py` the size
  targets (fuzzy, not measured), `mcp/skill_migrate.py` the recipe
  migration. A skill ARMS without review; a failed screen QUARANTINES it.
  `YAMADORI_RECALL=hints` (the default) keeps the legacy hints path;
  `=skills` switches. `x_yamadori.skills` is the record; `x_yamadori.hints`
  is a deprecated alias for one release. The tier flag is still called
  `hints` in code and in X-Yamadori-Features.
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
