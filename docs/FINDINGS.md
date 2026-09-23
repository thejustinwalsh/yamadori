# Findings and solutions

Every defect found, what it cost, and the change that fixed it. Written down
because the recurring failure in this project is not a lack of findings — it
is findings that were measured, reported in conversation, and never landed.

A finding with no solution beside it is half a result. Where something is
still open, it says so and says what would close it.

Dates are the session of 2026-09-21/22.

---

## 1. The watchdog killed a healthy stack every ten minutes

**Finding.** `scripts/watchdog.ps1` asked the PROXY on :1234 for a model named
`bonsai`. The proxy advertises only `yamadori` — internal names are
implementation detail, a decision made later and in a different file. So every
five-minute pass concluded the stack was dead and, on each cooldown expiry,
killed llama-swap and every llama-server under it.

```
23:51:02  UNHEALTHY: model 'bonsai' missing  -> restarting
00:06:02  UNHEALTHY: model 'bonsai' missing  -> restarting
00:16:02  UNHEALTHY: model 'bonsai' missing  -> restarting
```

**Cost.** A LiveCodeBench run: 33 of 50 rows recorded as generation errors.
They were spread evenly through the file because the kills were on a timer,
which made them look like a transport fault. Hours were spent blaming the
model, then llama-swap, then the connection. It also ran a generation probe
with a 120-second timeout against a model whose answers legitimately take 450
seconds, so a busy stack failed its own health check.

**Solution, applied.** Rewritten to three states and nothing else: running →
do nothing; down or missing → restart THAT service. It checks `/health` routes
only — no model names, no model status, no generation probe. Per-upstream
checks take their names from llama-swap's own `/running`, never from the
script, so a rename cannot resurrect this.

**Verified.** Killed the tools API: the watchdog restarted only it, llama-swap
and all three llama-server PIDs untouched. Then 417 minutes of continuous
uptime with zero restarts, against a stack that had been dying every ten
minutes.

**Rule.** A liveness check may know one thing: whether the process answers.
The moment it knows a model name it has an opinion the system can invalidate,
and its response to being wrong is to destroy what it watches.

---

## 2. Every tool returned an empty string when no repository was bound

**Finding.** `run_our_tool` opened with `text = ""` and only filled it inside
`if db:`. With no repository — **the normal case for a remote service** — no
tool executed at all. Not the searches, and not `record_step`, `read_rings` or
`summarize_text`, which never needed an index.

**Cost.** This is the root of the benchmark failures. The corpus recorded
`chars: 0` on the first tool results of the run that then issued the identical
search twelve times over 842 seconds and never wrote an answer. The model was
not ignoring guidance; there was no guidance, because nothing ran. Every
carefully worded tool result in the codebase was unreachable code.

**Solution, applied.** Two decisions were tangled in one `if`: whether a corpus
exists, and whether this tool reads one.

- `INDEX_TOOLS` — need an index; with none, return the structured `NO_INDEX`.
- `ROOT_TOOLS` — need a repository on disk (`run_check`); return `NO_REPOSITORY`.
- Everything else runs regardless.
- A final guard: **a tool never returns nothing.** An empty result is
  indistinguishable from a crashed tool, a broken transport, or a tool that
  does not exist.
- With no repository the index is pointed at a path that cannot exist, so a
  caller's search can never fall through to the server's own 7,742-chunk index
  of this codebase. That would be both wrong and a disclosure.

---

## 3. Tool results told the agent to retry something that could not work

**Finding.** `find_by_meaning` answered two different situations with one
sentence:

> "No results. Nothing is indexed for this repository yet, **or** the query
> matched nothing. **Try** find_by_pattern with a literal you expect…"

It merges two facts behind an "or", never says the call failed, and ends by
recommending a tool that fails identically. `search_fused` returns `[]` both
when no index exists and when a query misses.

**Cost.** The twelve-identical-searches loop. The model did what it was told.

**Solution, applied.** Tool results are now structured facts with no advice in
them. Three situations, three answers:

| situation | `error` | `retryable` |
|---|---|---|
| nothing bound to this conversation | `NO_INDEX` | false |
| index fine, every retriever threw | `RETRIEVERS_UNAVAILABLE` | false |
| index fine, retrievers ran, nothing scored | `ok: true, matches: 0` | **true** |

`retryable` is a fact, not a suggestion: false means no arguments can ever
succeed here. Each error carries `remedies` with `fixable_by` set to agent,
user or operator, and `why_not_the_agent` where the agent cannot act — e.g.
*"this server cannot see the user's filesystem; no tool call can reach it."*
That is a fact about the architecture, which is what makes "don't retry"
self-evident rather than an order.

Errors deliberately do NOT list what the server has indexed: we do not know
which source the caller wants, and a list invites picking from it instead of
asking about the code in front of them.

**Note on the protocol.** OpenAI tool results are
`{"role": "tool", "tool_call_id", "content"}` — content is a plain string and
there is **no error field** (Anthropic's API has `is_error`; OpenAI's does
not). So structure has to live inside content, as JSON.

---

## 4. Retriever outages were silent and looked like bad queries

**Finding.** `search_fused` ran three retrievers with `except Exception: pass`
on two of them. If both failed, the fusion returned nothing and the tool
reported "the index is populated, your query missed". A total outage wore the
clothes of a bad query. The module's own docstring warns about the embeddings
version of exactly this — *"a benchmark run scored 0.0% across every condition
before anyone noticed the index was never consulted"* — and the same shape was
still present one layer down.

**Solution, applied.** Failures are recorded rather than swallowed. `diag`
reports which retrievers ran, which failed and with what error, and which are
off by CONFIGURATION (`semantic` is disabled by `CODE_SEARCH_SEMANTIC=0`). If
none ran, the tool returns `RETRIEVERS_UNAVAILABLE` instead of zero matches.

**Still open.** `CODE_SEARCH_SEMANTIC` is off because embeddings measurably
lost to BM25 on the gauntlet index (recall@5 77/120 vs 92/120, McNemar
p=0.0041), and fusion did not rescue it (87/120, fused vs keyword 3–8,
p=0.2266). The recorded caveat is that the gold queries derive from symbol
names, which flatters lexical matching, and **the conceptual query set that
would favour embeddings was never run.** That test is still owed.

---

## 5. The work log was shared by every caller on the server

**Finding.** `rings.session_key()` fell back to the basename of the SERVER's
working directory — the string `llama-stack` for every caller there will ever
be. One shared work log: every remote user wrote into it and read everyone
else's entries back.

**Evidence.** In the tool audit, a caller with no repository was handed a note
about `EntityIndex` in `packages/core` from an unrelated session.

**Solution, applied.** The proxy passes the real conversation key as
`_session`, per call rather than through an environment variable, because the
server answers several requests at once and a process-global would race. A
missing `_session` is not quietly defaulted — returning another session's log
is worse than returning none, so an unscoped call says so.

**Verified.** Session A cannot read session B's entries; A can read its own;
an unscoped call leaks nothing and explains why.

---

## 6. The two tool surfaces were merged, and one of them could not work

**Finding.** `code_search.py` did `TOOLS = TOOLS + rings.TOOLS`, putting the
work log on both the MCP surface (outside clients, over stdio) and the
proxy-injected surface. But `record_step` needs a conversation key that only
the proxy has, so over MCP it either errors for a legitimate caller or writes
into a shared log — finding 5.

**Solution, applied.** Split: `TOOLS` is the MCP surface, `INTERNAL_TOOLS` is
what only this stack invokes, `ALL_TOOLS` is what the proxy offers the model.
Nothing was removed from what the model can call.

**The rule.** A tool offered on both surfaces must take everything it needs
from its arguments and its configuration, and assume nothing about who called
it. Search qualifies — it searches whichever index it is pointed at, and the
caller decides which. The work log does not.

**Still open.** `code_search.py` retains a standalone MCP stdio server. For a
zero-config remote product that path arguably should not exist; it is
currently the only way to use the stack without the proxy.

---

## 7. `tracker` was undefined in the streamed tool loop

**Finding.** `stream_body` called `run_our_tool(..., tracker, ...)` with a
`tracker` that is a local of `complete()`. **Every streamed request that
touched any of our twelve tools died with `NameError`** — that is the editor
path with augmentation on. It survived because the loop only runs when the
model actually calls a tool.

**Found by** `ruff check --select=E9,F`, in under a second. Along with
`difflib` undefined in `code_search.py:631`, which crashed the "did you mean…"
helper.

**Solution, applied.** Both fixed and covered by a test that drives the
streamed tool loop end to end.

**Rule.** A linter is not style enforcement; it is the cheapest test that
exists, and it found two live outages instantly.

---

## 8. The loop ran out and returned a tool call as the answer

**Finding.** `complete()` looped to `MAX_TOOL_HOPS`, then called the model once
more **with the tools still attached and nothing asking for an answer**. The
model, having requested a search twelve times, requested it a thirteenth, and
that response was returned to the caller AS the answer. Every such row carried
`finish_reason: "tool_calls"` and no code, and was scored as the model failing
to answer a question it was never asked.

**Solution, applied.** The final hop withdraws the tools and asks for the
answer — which `shomen.py` had been doing correctly all along. Verified
with a model that would otherwise search forever: `finish_reason` becomes
`stop` and content appears.

**Related, and the more important half.** A hop cap does not fix a loop; it
bounds how long you burn. The loop was caused by finding 2 and finding 3. The
budget is therefore ADVISORY (stated in the prompt so the model can plan and
front-load) and `MAX_TOOL_HOPS` is a runaway guard that should never end a
healthy request.

**Counter-evidence, recorded because it matters.** The model ignored both
forms of suggestion: the repeat-guidance told it eleven times that it was
repeating, and it repeated; the stated budget said six turns and it took
twelve. Only the hard cap stopped it. This was on a pathological case — an
empty index where no search could succeed — so it is not evidence that
suggestion never works, but it is the only direct evidence we have.

---

## 9. Multi-hop requests reported 8% of their token cost

**Finding.** `complete()` returned the LAST hop's response, so `usage` was the
cost of the twelfth generation rather than all twelve. Two benchmark rows spent
842s and 734s; at the stack's measured 16.0 tok/s that is ~13,500 and ~11,700
tokens. They reported 1,039 and 1,031.

**Why worse than noise.** The bias has a direction: the arm that hops most is
under-reported most, so the cost column flattered augmentation precisely where
it was most expensive.

**Solution, applied.** Usage accumulates across hops and reports a `hops`
count. Verified: a 4-hop request now reports 2,500 completion tokens instead of
the last hop's 400.

---

## 10. The harness scored its own output as the model's failure

Two instances, both fixed:

**The preamble was compiled as Python.** A reply that hit the token cap never
closed its code fence, so `extract_code` fell through to its bare-code path,
matched `class Solution:` further down, and returned the WHOLE reply —
including the proxy's status line:

```
sol.py", line 1
    `yamadori` · no repository detected in this conversation · ...
SyntaxError: invalid character '·' (U+00B7)
```

`extract_code`'s own docstring already promised the preamble must not end up in
the program. Fixed: it now drops leading lines until the first that looks like
code. And the preamble is now silent unless it has something actionable to say
— the no-repository case is the NORMAL state for a remote service, and its
remedy is the model's job, not the reader's.

**The typing guard was a substring test.** `"from typing import" in code` — so
a reply importing only `Optional` passed it and then died on `List` at line 2,
before any model logic ran. Scored as a model failure. Fixed with an AST check
of what the code actually loads versus binds, with the preamble's 35 names
derived by executing it so it cannot go stale. 8/8 including the real failure.

---

## 11. The launcher started the wrong server

**Finding.** `scripts/start-stack.bat` launched `mcp/proxy.py` — the stdlib
`BaseHTTPRequestHandler` — not `mcp/server.py`, the ASGI one. So an unattended
start had no HEAD (health checkers got 501), no real keep-alive and no graceful
drain. Every run that worked did so because someone had started `server.py` by
hand.

**Solution, applied.** Fixed, with the reason recorded in the file.

---

## 12. A self-test that crashed looked like coverage

**Finding.** `python mcp/tiers.py` raised `KeyError: 'budget'` on every run —
the key had been renamed to `floor` and the self-test was never updated. Nobody
noticed, because nobody ran it, because it was assumed to pass.

Related: `mcp/fanout.py` documents a gating policy in its docstring (lookups
N=1, design questions N=4) that **is not implemented anywhere** and the module
is called from the proxy zero times. A rule written in prose beside code that
does not follow it reads exactly like a rule that is enforced.

**Solution, applied.** Self-test fixed. The fanout gap is recorded in §14.

---

## 13. The hemisphere was not a singleton and exceeded its budget

**Finding.** `HELPER_LANES = 1` declared one second brain and nothing enforced
it: `shomen.investigate()` runs in a worker thread and called the model
directly, never touching the lane. And its first live run spent 51,206 tokens
against a helper allocation of 36,864 — 39% over, taken from the pool the main
conversation is entitled to.

**Solution, applied.**
- A THREADING semaphore (the existing lane is an `asyncio.Semaphore`, useless
  from a worker thread) taken at the call site. Verified: three concurrent
  investigations → one ran, two refused with a structured `HELPER_BUSY`.
- The KV allocation is read from `budget.py` and enforced: at budget the
  investigation writes up what it has rather than being cut mid-thought.
- A refusal is REPORTED. An investigation that silently did not happen looks to
  the model exactly like one that found nothing, and it will reason from an
  absence we manufactured.

---

## 14. Built, measured, never wired

Recorded as a standing list because this is the pattern that costs most.

| component | state |
|---|---|
| `mcp/fanout.py` | called from the proxy **0 times**; its gating policy exists only in a docstring |
| `mcp/concept_seed.py` | **0 calls** |
| `mcp/domains.py` | **0 calls** — gates nothing, though it exists to stop exactly the "tools offered where they cannot help" failure |
| `judge` (the only Laya caller) | implemented, wired to Laya, **declared in no tool list** — unreachable |
| `tier["hints"]`, `tier["fanout"]`, `tier["investigate"]` | never read outside `tiers.py` |

**Consequence, measured.** `tier["retrieval"]` is the ONLY augmentation flag
the proxy reads — `grep 'tier\["' mcp/proxy.py` returns exactly one line — and
it is `True` for `low`, `medium`, `high` and `max` alike. So every
augmentation those four tiers differ by is inert, and `tiers.describe()`
advertises "search, hints, fan-out x3, second context" while only "search"
executes.

**Stated precisely, because the looser version of this sentence is wrong.**
The four tiers do NOT produce identical requests: `tiers.apply()` still writes
a different `max_tokens` floor (1200 / 1500 / 3000 / 6000) and a different
`reasoning_effort` (medium / medium / high / xhigh) for each, and
`tier["hops"]` (4 / 6 / 8 / 12) reached the model as a stated tool budget
through `augment_messages`, while the *enforced* cap stayed
`MAX_TOOL_HOPS = 12` for every tier — so a `low` request was told "at most 4
tool-calling turns" and actually had twelve. **That dial is now deleted
outright; see #27.** What is identical is the **augmentation** — the four features
`describe()` names. A reader who repeats "the tiers produce identical
requests" will conclude the tier system does nothing, and then be surprised
when raising the tier changes cost and output length.

**Solution, applied in part.** All five flags — `retrieval`, `hints`,
`fanout`, `investigate` and `hops` — were wired to the proxy, after which
`hops` was deleted as a dial entirely (#27). Verified by grep: `hints` 2
reads, `fanout` 2, `investigate` 3, `retrieval` 1.

`thinks` is read by nothing, which is correct and deliberate rather than an
oversight: it is `True` on every tier, because thinking is the model's own
feature and no tier turns it off. It is a declared invariant, not a switch.

**Still not applied:** `domains.py` gating tool injection, the
highest-value one remaining. On the sample, `aug_on` spent 2,729 extra prompt
tokens (87% of its prompt) on a problem where it called **zero** tools, and
looped on the two where it did.

---

## 15. The benchmark measured the wrong thing

**Finding.** LiveCodeBench is self-contained algorithm puzzles with no
repository and nothing indexed. The entire augmentation is code search over an
indexed corpus. So the benchmark asked "do code-search tools help on problems
with no code to look up?" — and the answer is no, and worse than no.

**Measured**, after the transport was fixed (n=3, no statistical power, both
runs agreeing):

```
condition   pass@1      mean s   tok in   tok out
minimal     3/3 100%    270.4      557     4363
aug_off     3/3 100%    110.7      515     1770
aug_on      1/3  33%    603.0     5068     1073*
```

\* last-hop-only; see §9.

`aug_off` — thinking at medium, no tools, no capability block — wins on every
axis at once. Thinking mode was also CHEAPER here: 1,770 completion tokens
against `minimal`'s 4,363 for the same 3/3, which is the opposite of the usual
assumption and deserves a real measurement.

**Solution.** The benchmark for the actual product claim does not exist.
`bench/api_truth.py` and `bench/semantic_tasks.py` are present with no result
files. The claim is about three.js/TSL, react-three-fiber, TypeGPU, WGSL, Rust
and hard TypeScript types, where an indexed corpus exists and retrieval has
something to retrieve.

**Blocker found while writing this.** Only TWO **packages** are indexed:
`index/packages/` holds `three@0.185.1` and `typegpu@0.12.5` and nothing else.
Not drei, not r3f v9 or v10-alpha. Most of the target domains have no
dependency source to retrieve from today.

**Correction: koota is not one of the gaps.** It is indexed, as a *repo* rather
than a package — `index/repos/koota-6955afb4.sqlite3` — and it carried 116 of
the 356 rows in `bench/retrieval_results.jsonl`. The package corpus is the
thing that is thin; the repo corpus is glyph and koota.

---

## 16. Laya: System 1, running with no callers

**Finding.** The architecture is settled — Laya is System 1 (fast, structured,
always on), the second model context is System 2 (slow, deliberate), and Laya
is the required router in and out of the hemisphere. In the code, Laya had
**zero live callers**: its one tool was unadvertised (§14), and the hemisphere
summarised itself with the big model.

**Measured today, zero-shot.** Using the primitive that
`bench/mechanisms/selectors.py` already documents as the good one —
permutation-averaged closed choice with a margin gate at 0.3, abstaining below
it:

```
route_in, raw prose state:
  "what is 2 + 2?"                                 DECIDED  margin 0.404  answer_directly
  "how does the KV pool get sized in this stack?"  ABSTAIN  margin 0.005
  "struct-of-arrays or array-of-structs for ECS?"  ABSTAIN  margin 0.098
  "fix it"                                         ABSTAIN  margin 0.232
  "why does my TSL shader render black in r185?"   ABSTAIN  margin 0.070

distil, on a finding that cites config.yaml and budget.py:
  grounded ("is this prose about those files")     DECIDED  margin 0.925
  verdict  ("does it answer the question")         ABSTAIN  margin 0.005
distil, on a vague finding with no citations:
  grounded                                         ABSTAIN  margin 0.070
```

**Two results, and they are different.** The GROUNDING question separates
decisively — 0.925 against 0.070 — and that is the question that matters most
for the callosum, because it is the fabrication check. The ROUTING question
abstains on 4 of 5.

**My error, recorded.** The first version asked a bare `noul` once and read
0.162 as "no". It was not a no — it was the undecided band, and
`selectors.py` already records that reordering-sensitive cases land at
0.012–0.184 while a decided case sits at 0.800. An abstention read as a
decision silently disables the feature it is meant to gate.

**Solution, partly applied.**
- `distil()` is now called on every finding — Laya distils on the way out,
  which is mandatory. The deterministic citation check remains the authority on
  WHICH paths were read; Laya judges whether the prose is about them.
  **Reversed by §17 and no longer true of the code.** `distil()` is kept and
  tested as the reference implementation of the call pattern, but it gates
  nothing and is not called on the finish path; `shomen.py` says so in place.
  `docs/ROADMAP.md` §2.1 records it as **removed** (29/29 constant, AUC 0.667).
- `route_in()` exists and abstains honestly (`run: None`, never `False`).
- **Not yet solved:** routing needs structured input, not prose. The hints
  system already demonstrates the pattern — retrieval narrows to a small
  candidate set, then Laya picks from a closed set with the state fixed. The
  narrowing signals exist and are unwired: `domains.detect()`,
  `discover.imports()`, retrieval score spread, whether any index exists.
- **And Laya needs training**, not only threshold-fitting. Prior work measured
  calibration on ~50 labels halving ECE and moving accuracy 64.6% → 75.1%, for
  a different question set, never productionised.

In flight: labelled routing set and a prose-vs-structured paired comparison; a
training pipeline with held-out evaluation and a loadable artefact; completion
of the tool test suite. See `docs/LAYA.md` and `docs/TOOLS.md`.

### 16a. The routing question was the wrong SHAPE, not beyond Laya

**My error, twice over.**

First: the 0.3 margin gate was measured on the nodes in
`selectors.EXAMPLE_TREE`, three of whose four nodes are BINARY. With two
options, uniform is 0.5/0.5 and a 0.3 margin means 0.65 vs 0.35. With the
three-way choice I wrote, uniform is 0.333 and the same gate demands roughly
0.52 vs 0.22 -- a far harder bar for identical underlying confidence. I
imported a threshold tuned for one arity, applied it to another, and reported
the resulting abstentions as "Laya cannot route".

Second: I asked one multi-way "which technique?" question. That is the worst
shape available.

**Measured, same six-to-eight questions, same model, same gate:**

| framing | decided | correct |
|---|---|---|
| 4-way "pick a technique" | 1/6 | 1/6 |
| binary, two ACTIVE options | 3/6 | 3/6 |
| binary cascade, `continue` an explicit option | 8/8 on the investigate node | 6/8 overall |

**The change that did it: `continue` / do-nothing is a first-class OPTION, not
the fallback.** Then "nothing is needed here" -- the most common correct answer
-- is a positive decision rather than an abstention, and the node only abstains
on genuine uncertainty between real alternatives.

**The investigate gate, node-level, 8/8:**

```
how does the KV pool get sized?            0.46  FIRE      correct
why does my TSL shader render black?       0.54  FIRE      correct
where is sizeKvPool defined?               0.36  FIRE      correct
what is 2 + 2?                             0.13  continue  correct
what does SOLID stand for?                 0.12  continue  correct
rename this variable to userCount          0.06  continue  correct
struct-of-arrays or array-of-structs?      0.10  continue  correct
cleanest way to model this state machine?  0.11  continue  correct
```

**The fan-out node does NOT work.** It stays quiet correctly on trivial input
(`rename` decided `continue` at 0.32) but fails to fire on either genuine
fan-out case (0.13, 0.07). The System 2 trigger and the multiple-answers
trigger need separate designs, not one shared one.

**Failure mode is the safe one.** Uncertain falls to `continue`, which is the
cheap path, and the asymmetry supports it: a false fire costs a twelve-hop loop
or 3.2x wall clock; a false quiet costs a slightly worse answer.

**Caveat, stated because this is where overclaiming has happened before.** n=8,
and the examples were written by the same person who chose the phrasing. The
labelled set of >=40 is the real test. This is a promising signal, not a
validated gate.

**Solution, and it is the design for every technique gate:**

1. Free structural gates first, no model at all -- nothing indexed means
   retrieval cannot help; no imports and no code means design recipes are
   inadmissible. Facts, not predictions, and they cost nothing.
2. Then a cascade of BINARY Laya questions, one per technique, each with
   `continue` as an explicit option. Never one multi-way choice.
3. Abstain means do not apply, because of the cost asymmetry.
4. Prefer reactive signals where they exist -- retrieval score spread after one
   cheap search, tool-hop count as revealed uncertainty -- over predicting.
5. Measure the ceiling before building any gate: run with and without, count
   where outcomes DIFFER. Fan-out on lookups is already measured at 7/8 vs 7/8,
   zero discordant pairs, 3.2x cost. No gate can recover value that is absent.

---

## 17. §16's Laya result did not survive being measured properly

**Finding.** §16 above reports two spot-checks and draws opposite conclusions
from them: `route_in` "abstains on 4 of 5", and the `distil` GROUNDING
question "separates decisively -- 0.925 against 0.070". The first held up.
**The second did not.** Both were five and two data points respectively.

Re-measured on labelled sets (`bench/laya_routing_labels.jsonl`, n=59;
`bench/laya_distil_labels.jsonl`, n=29 including 12 deliberately fabricated
findings that cite files they do not describe):

```
route_in, three-way, prose, gate 0.3 :  5/59 decided (91.5% abstain), 50.8% argmax
                                        regex floor on the same labels: 79.7%
distil GROUNDING                     :  chose from_the_files on 29/29, INCLUDING
                                        all 12 fabrications; AUC 0.667; mean
                                        probability gap +0.029; mean margin 0.867
distil VERDICT                       :  chose answers_it on 27/29; 44.8% argmax
```

The 79.7% is that regex scored once over all 59 of these labels. `docs/LAYA.md`
Part II quotes **the same six lines at 0.841 ±0.023** — 89 labels (these 59
plus 30 more), mean over 8 group-aware held-out splits. Different label set and
different protocol, both correct, and the 0.841 is the only one comparable to
the trained head's 0.726.

The 0.925-vs-0.070 pair was real but on the wrong axis. A state-sensitivity
probe shows the question Laya actually answers is "is this text about code and
files" (real finding 0.95, unrelated prose 0.39, empty 0.69) -- not "does this
text describe THESE files". §16's vague uncited finding differed on the first
axis. Every finding a real investigation produces does not.

**Cost.** §16's recommendation was to keep the grounding question and treat the
deterministic citation check as secondary. That is backwards, and shipping it
would have put a fabrication detector in the callosum that passes 100% of
fabrications at a mean margin of 0.867.

**The general defect, which is bigger than Laya.** A margin gate was being read
as confidence. Three separate questions return large stable margins while
carrying zero information, and the one question that does work has margins that
INVERT exactly where it fails. Before trusting any new Laya question, run the
degeneracy check -- what fraction of items got the same answer --
`python -X utf8 bench/laya_calibration.py --binary` prints it.

**Solution.** In `docs/LAYA.md`, with the numbers, the method, the negative
results and a stated fix for each. In short: drop the three-way question and
the two distil questions; keep ONE binary question ("does answering this need
our source?") on a PROSE state at a **margin gate of 0.30** -- 95.5% accurate
on the 47.8% it decides; handle underspecification with a deterministic guard
(12/13 vs Laya's 2/13); make the existing `cited_paths` check the authority for
grounding. Still open: an overlap test between a finding and the retrieved
chunk text, which needs `distil` to receive the trace handle --
`mcp/shomen.py` is owned by another workstream.

**Guarded by** `bench/test_laya_calibration.py` -- 46 runtime checks (44 assertion sites), no GPU
needed, one per claim in `docs/LAYA.md`. If a future change makes structured
state win, or makes the grounding question work, those tests fail and the doc
gets rewritten rather than quietly re-litigated.

---

## 18. Fan-out computes the antithesis and then throws it away

**Finding.** `mcp/fanout.py` opens: *"Answer a question several ways at once and
keep what they agree on."* `run()` returns a consensus, an agreement ratio and
a winner. `dissent_note()` surfaces disagreement -- **to the user, as a
caveat**, and only when agreement falls below 0.75.

So the competing positions are generated, footnoted, and discarded. **The first
brain never sees them.**

**Why that matters to the thesis.** The stated purpose of the second hemisphere
is to let the first consider alternatives and distil something dialectic.
Consensus is the opposite operation: it discards precisely the material a
dialectic runs on. Thesis and antithesis both have to reach the reasoning step
for there to be a synthesis; right now we compute the antithesis and mention it
in a footnote to somebody who is not doing the reasoning.

**Solution, not yet applied.** Feeding disagreement BACK to the first brain is a
different product from fan-out voting, and it is a design decision rather than a
tuning one. Record it as the open question it is. Note also that fan-out is
called from the proxy zero times, and its one measurement is a null result on
lookups (7/8 vs 7/8, zero discordant pairs, 3.2x cost) -- a task class with no
headroom for alternatives to exploit, which is the wrong place to have tested a
creativity mechanism.

---

## 19. The load-bearing claim has never been measured

**Finding.** The hemisphere's entire justification, from its own docstring, is
context economy: *"six thousand tokens of searching becomes a two hundred token
finding."* Every other claim in the construct rests on it -- that you can buy
better thinking without paying for it in the window whose recall degrades.

It has never been measured. Not once.

**Why this is the cheapest experiment available.** It needs no Laya, no buckets,
no routing gate, no trained head. Same question, same model, tools in the main
context versus tools in a second context; measure answer quality AND
main-context tokens consumed. Everything else in this project was chased while
the foundation went untested.

**Solution.** `docs/ROADMAP.md` should carry this as step 0. If a second context
does not buy context economy, the selection machinery is decorating something
that does not hold. If it does, every later measurement has a foundation to sit
on.


## 20. The reranker is producing garbage, live, and every reranker number is void

**Finding.** `/v1/rerank` fails two ways at once, on the exact query
`config.yaml` uses as its own worked example.

Reproduced directly, query "how do I parse a tool call from the model output":

```
ONE AT A TIME
  sourdough needs a long cold ferment...      7.03e-08   <- WINS
  def parse_tool_call(answer: str) -> dict    6.93e-08
  the weather in paris is mild...             1.53e-08
  class Renderer: def draw(self)              3.31e-10

BATCHED, same four documents, one request
  class Renderer: def draw(self)              3.31e-10   <- WINS
  the weather in paris is mild...             2.96e-11
  def parse_tool_call(answer: str) -> dict    1.27e-13
  sourdough needs a long cold ferment...      3.26e-23
```

Two independent defects:

1. **A document's score depends on what else is in the request.** Batching
   changes the winner. Measured at scale by the hint-collapse work: the same
   89 documents, each queried with its own verbatim text, score **15-16/89
   batched against 68/89 one at a time**. Embeddings score 89/89.
2. **It is wrong even one at a time.** Sourdough beats the correct function.

**The model is not the known-broken one.** llama-swap and the process list both
confirm `Qwen3-Reranker-0.6B-Q8_0.gguf` on port 10005. `config.yaml` documents
the 4B GGUF failing exactly this way -- "1.3e-19 ranked FIRST, 8.8e-24 ranked
LAST" -- and states plainly that **"The 0.6B scores the same case at 0.9992."**
It now scores 6.93e-08. Either that claim was always wrong, or the rank path
regressed; the repo updated llama.cpp recently, which is the first thing to
check.

**What this voids.** Every reranker measurement in this repo was taken through
this path:
- `scripts/eval_rerank.py`'s "rerank 3/11 vs embed 1/11" -- void
- the 13.5% hit@1 in `bench/retrieval_results.jsonl` -- void. **The "void twice
  over, since 199 of 356 rows never ran the reranker at all" half of this
  sentence is wrong and is corrected here rather than deleted**: that arm is
  driven at `rerank_cutoff=999` (`bench/retrieval.py:106`, `:302`), every one of
  the 356 rows carries a non-zero `rerank_ms` (median 1,330 ms, min 466 ms), and
  a `rerank_rank` of `0` means "gold not in the top 5" — a miss, not a skipped
  call. It is void once, through the batching path described above. Same
  correction as `docs/ROADMAP.md` §1.3 and `AGENTS.md`
- `RERANK_MAX_K = 2` is justified in `code_search.py` as a LATENCY decision
  resting on "the reranker permutes within the top 5 without changing which
  files reach the caller". That was measured through the corrupt path and
  cannot be relied on either.

**Why the damage is limited, and why that is luck.** `RERANK_MAX_K = 2` against
`DEFAULT_TOP_K = 5` means reranking rarely runs. A latency optimisation is the
only thing standing between this and a silently corrupted search path --
`search_code` batches up to 40 candidates, which is the worse of the two
failure modes.

**Solution, recommended not applied** (it is a live-config decision):
1. Disable reranking until the rank path is verified. `config.yaml` already
   states the principle for the 4B: the failure is silent, search "returns a
   plausible but wrong function, with no error anywhere". That is now true of
   the 0.6B.
2. Add a STARTUP ASSERTION on the worked example from `config.yaml`: if the
   0.6B does not score that pair as documented, refuse to serve reranking. A
   retriever that fails silently must be made to fail loudly.
3. Only then re-run the owed measurement (ROADMAP 1.3) with `RERANK_MAX_K`
   raised, since no accuracy claim about the reranker is currently admissible
   in either direction.

---

## 21. Laya loses in the one regime its architecture predicted it would win

**Finding.** Laya has now been measured in all three regimes and fails all
three. The third was the one I argued for on structural grounds, with a
prediction recorded in advance: *"Laya wins on contrastive buckets, ties or
loses on topically distinct ones."*

**Refuted.** 19 hand-curated buckets of mutually exclusive hints, 89 members,
89 probes, chance 21.3%:

| arm | overall | contrastive | topical |
|---|---|---|---|
| embedding | **71.9%** | **57.4%** | **88.1%** |
| lexical floor (no model) | 51.7% | 46.8% | 57.1% |
| laya, condition-only options | 44.9% | 36.2% | 54.8% |
| laya, joint permutation-averaged | 33.7% | **29.8%** | 38.1% |
| rerank | 22.5% | 27.7% | 16.7% |
| random / fixed | 21.3% | | |

It is WORST on contrastive buckets -- the slice the mechanism said was its
home -- at 29.8% against embeddings' 57.4%, barely above the 21.3% floor.
Paired McNemar on that slice: 5 Laya-only wins against 18 embedding-only,
p=0.011. Sweeping thresholds to match coverage, there is no point on any slice
where Laya's precision reaches embeddings'; the smallest gap anywhere is 27.7
points. Four Laya configurations were swept before concluding.

**The cause, and it is the same one already in the record.** Laya returns the
same member for **75% of a bucket's probes** against a 1/k floor of 21%, and is
a literal constant on 5 of 19 buckets. The option WORDINGS decide, not the
problem. That is the degeneracy `docs/LAYA.md` findings 3, 5 and 7 describe,
now confirmed in a third regime.

**And its margin still is not confidence.** Discarding the 80% of probes it is
least sure about buys 5 points. Embeddings' margin is real: 71.9% -> 86.8% at
60% coverage, and 100% on topical buckets from 75% coverage down.

**Solution.** Laya has no job in this stack. The structural argument for it was
sound and the model does not deliver on it, which is the only kind of negative
result worth anything. Routing uses the regex; hint collapse uses embeddings
with their margin as the abstention signal, which is the "never harm" property
we needed and it is already calibrated. Keep `laya_service.py`, the trained
head and the harnesses -- they cost nothing idle and they are how this gets
re-tested if the model is retrained -- but nothing gates on them.


## 22. Laya fails the fourth regime too, and this one was my hypothesis

**Finding.** I argued guardrails were the job where all four of Laya's stated
design conditions finally held: categories far apart, evidence in the input,
high volume, calibrated threshold. Measured on 156 labelled items, paired
bootstrap of AUC against a pre-registered regex floor, 2000 resamples:

| framing | AUC | dAUC vs regex | paired 95% CI |
|---|---|---|---|
| laya_what | 0.711 | +0.021 | [-0.081, +0.124] **includes 0** |
| laya_addressee | 0.720 | +0.031 | [-0.074, +0.141] **includes 0** |
| laya_behaviour | 0.725 | +0.035 | [-0.069, +0.139] **includes 0** |
| regex (pre-registered floor) | 0.690 | -- | |
| **embeddings, nearest centroid** | **0.886** | **+0.196** | **[+0.116, +0.279]** |

Three framings were run specifically so a null could not be blamed on wording.
All three land together, all three CIs include zero. Recall at an FPR the
stack could live with (<=5%): Laya 25.9%, regex 0%, **embeddings 55.6%**.

**Four regimes, four failures** -- routing, grounding, comparative collapse,
and now guardrails. I predicted a win in two of them from the architecture.
The mechanism argument was right every time about what the model *should* be
good at, and wrong every time about what it *is* good at. That is the lesson,
and it is about me rather than about Laya: a structural argument is a
hypothesis, and four in a row is enough to stop advancing them.

**Three results worth keeping regardless of the verdict:**

1. **The regex is WORSE THAN CHANCE on adversarial-benign input** -- AUC 0.405,
   ranking security code above actual attacks, and flagging 16 of 26 such items
   including `config.yaml`'s own paragraph about prompt injection. "ignore all
   previous instructions" in a denylist and in an attack are the same tokens;
   what separates them is provenance and position, which is not in the string.
2. **Laya's single statistically real claim** is exactly there: dAUC +0.220 on
   that slice, CI [+0.068, +0.371], surviving Bonferroni x3. Embeddings beat it
   there too (+0.322).
3. **The margin inversion is worse than anywhere yet.** On that slice all three
   framings invert, the best carrying margins **2.43x larger when WRONG than
   when right** (0.631 vs 0.260). No gate can be set: raising the threshold
   preferentially retains the false positives.

**And a finding that undercuts the product claim itself:** calibration is a
property of the PROMPT, not the checkpoint. ECE moves 0.101 -> 0.356 by
rewording the two options alone, and the best-calibrated framing is the worst
discriminating one. "Calibrated probabilities from strictly proper scoring
rules" does not survive being asked a different question.

**Solution.** Close the Laya question. Nothing gates on it. Keep the service,
the trained head and the four harnesses -- they cost nothing idle and they are
how this gets retested if the model is retrained -- but no decision in this
stack waits on them.

**What to do about injection instead, and it is not a classifier.**
Provenance-based handling: mark `index/packages/` content untrusted
STRUCTURALLY, because where a string came from is a fact we hold and no model
has to infer it. Then channel separation, then tree-sitter structural
screening. The embedding screen (55.6% recall at 4.9% FPR, using vectors
already sitting in `chunks.vec`) is worth having as ANNOTATION, never as a gate
on a tool call -- the test suite's own summary is blunt about why: "NO policy
tested is usable: the best still blocks 20+ of 26 adversarial-benign items".

---

## 23. The combined system has never been measured, because it does not exist

**Finding.** Every "combined system" number reported in this project measured
roughly one third of the stack, through a broken harness. Checked directly:

```
tier["retrieval"]    read by the proxy: True
tier["hints"]        read by the proxy: False   -> no hints are ever injected
tier["fanout"]       read by the proxy: False   -> no fan-out ever happens
tier["investigate"]  read by the proxy: False   -> only if the model calls the tool
```

So at tier `max`, a request gets retrieval and tools. `tiers.describe()`
advertises "search, hints, fan-out x3, second context" to callers, and one of
those four executes.

Meanwhile every measurement that did run was corrupted: the watchdog killing
the stack every ten minutes (33/50 rows), tools returning the empty string
(twelve-hop loops), the preamble compiling as Python, token accounting
reporting 8% of actual, tools offered where nothing was indexed, and a baseline
with the model's own thinking switched off.

**Solution.** Build the three unwired systems before any combined benchmark,
and treat every prior combined result as void rather than as a weak signal.
`mcp/hints.py` is the first of them.


## 24. The load-bearing claim HOLDS -- context economy is real and large

**Finding.** The hemisphere's whole justification is context economy, and it
had never been measured. It now has: n=26 paired tasks, 78 generations, zero
errors.

| | main context | second context |
|---|---|---|
| main-context tokens, median | 8,326 | **2,410 (0.29x)** |
| main-context peak window, median | 3,670 | **1,498 (0.41x)** |
| peak window, worst row | 8,790 | 2,622 |
| correct (path + facts) | 25/26 | 26/26 |
| correct on facts alone | 26/26 | 26/26 |
| total tokens across both | 303,531 | 1.76x-2.06x -- *a meaningless sum, see below* |
| wall clock | 737s | 1,423-1,752s |

Fewer main-context tokens on **26 of 26 rows**, sign test p=2.98e-08, and the
main-context figures replicate to within 3%. This is roadmap outcome 1: saves
tokens at equal quality. The construct is sound and the work above it has a
foundation.

**Caveats, none of them buried.** Only a LARGE quality difference could have
been detected at n=26; on facts alone there are zero discordant pairs. The one
discordant row partly flatters the second-context arm, because the hemisphere
ENFORCES citation and the grader rewards that. And none of this went through
the proxy's tier gate, so it is a property of the module, not yet of the
product.

**The docstring's own claim was wrong in both magnitudes.** It said "six
thousand tokens of searching becomes a two hundred token finding". Measured:
10,764-15,798 tokens of searching become **531** crossing back -- better
compression than advertised (21-30x) from a finding 2.6x larger than
advertised. The finding size is the stable number, so there is no sampling
excuse for it. Replaced with the measured figures.

**THE TOTAL-TOKEN FIGURE IS THE WRONG UNIT.** Tokens are counted PER
CONTEXT, because contexts are separate windows and the helper's is discarded
once the finding crosses. Summing them measures a bill nobody pays -- the
compute is local and free, and the helper's 18k never competes with the main
conversation's 2.4k. The main context is the scarce resource because it
persists and its recall degrades. The real cost is WALL CLOCK: 737s ->
1,423-1,752s, which the caller waits for.

**And that wall clock is inflated by a FIXABLE DEFECT.**
The main model rewrites the question at the delegation boundary and invents
context. Captured verbatim: "the **pymrem Python library**", "TSL
(**TypeScript Style Linter**)", "likely a file named **animation.py**",
"**@deprecated Javadoc**". Where the rewrite is faithful, delegating is
CHEAPER -- 0.31x, 0.75x, 0.79x. So 2x is an upper bound on a bug.

**Solution.** Stop the model guessing what is indexed: put the index roots in
the `delegate_investigation` tool description, or force `describe_index` on
hop 0. `mcp/domains.py` exists for this shape of problem and still has zero
callers.

---

## 25. Three bugs in the ONLY remaining fabrication guard

**Finding.** With Laya's grounding check removed on measurement (#17 — the
29/29-constant, AUC 0.667 result; #21 is the separate hint-collapse regime),
`shomen._cited()` is the sole thing standing between a distilled finding
and an undetectable fabrication. It had three defects, all confirmed live:

1. **`lstrip("./")` is a CHARACTER-SET strip, not a prefix strip.** Every
   leading dot and slash is removed, so `.eslintrc.js` became `eslintrc.js`
   and matched nothing that was actually retrieved. Any finding citing a
   dotfile counted as citing nothing.
2. **`yaml|yml` were absent**, and `config.yaml` is where a large share of the
   answers in this repo live.
3. **`.json` was recorded as `.js`.** Regex alternation is leftmost-first and
   `js` preceded `json`, so `package.json` matched `.js` and left `on`
   unconsumed. Confirmed: `.npmrc.json` -> `.npmrc.js`. Every JSON citation
   was mangled into a path that could never match.

**Solution, applied.** Prefix-strip by loop rather than `lstrip`; extensions
ordered longest-first AND guarded by a negative lookahead, because order alone
is fragile -- the next person appending an extension reintroduces it. 8/8 on a
case set including the fragile ones.

**Worth noting how these were found.** Not by a test. By another workstream
instrumenting the module to measure something else entirely. A guard nobody
exercises is a guard nobody knows is broken -- and this one is now
load-bearing precisely because the redundant check was removed.

---

## 26. The index is three.js only

**Finding.** `index/code.sqlite3` was described in several places -- including
by me, to the project owner -- as holding koota, three and typegpu. It does
not. Top-level directories are `renderers`, `nodes`, `math`, `core`,
`materials`, `extras`; the root is a single three.js checkout; every chunk is
one tree at `REVISION='187dev'`.

koota and typegpu live in separate index files under `index/repos/` and
`index/packages/`.

**Why it matters more than a documentation slip.** three.js is in every
training set, so contamination is the central threat to any retrieval
measurement taken on this index -- a model can answer from memory and look
like retrieval worked. The context-economy task set handles this by building
half its questions from `@deprecated rNNN` version markers, which are specific
enough that memorisation does not supply them.

---

## 27. The hop budget was a fossil of a broken tool surface

**Finding.** Two mechanisms existed to stop a model calling tools too many
times: a budget stated in the prompt (`budget_line()` in the proxy, and a
`BUDGET: at most N searching turns` paragraph in `shomen.py`'s system prompt),
and a cap in the loop. Both were built in response to one observation — a
benchmark arm that "spent twelve generations on a problem the other arms
answered in one", and a second-brain run that spent 51,206 tokens over eight
hops and returned 48 characters.

**Neither observation was about overspending.** Both were taken on a stack
whose tools returned **empty strings** on failure. A model that receives `""`
cannot distinguish "nothing matched" from "this call is broken", so it calls
again. It was not exploring out of appetite; it was retrying a broken tool.
That root cause was fixed separately — tool results now state the situation,
whether it is retryable as a fact, and a remedy with an owner — and the budget
machinery was never revisited afterwards.

**The stated budget was also false.** Both tool loops in `mcp/proxy.py` read
`MAX_TOOL_HOPS` and never the tier, so a `low`-tier request was told "at most
4 tool-calling turns" while its real ceiling was 12. The number the model was
given was checked by nothing. `budget_line()`'s own docstring argued that "a
budget in the PROMPT and a cap in the LOOP are different instruments and
neither substitutes for the other" — a good argument for two instruments that
agree, and these did not.

**Why a number is the wrong instrument anyway.** A hop budget answers "how
many times may you try" when the real question is "has anything changed since
the last try". With truthful tool results the loop ends because the model has
an answer or an error it can believe. A loop that continues past that point
means a tool is lying, and the repair is to fix the tool. Capping it hides the
defect and returns a truncated answer in its place — which is how one
benchmark row spent 842 seconds and came back with no code at all.

It is also the wrong instrument for a *second* brain specifically. The entire
reason the deep-thinking context exists is to spend effort somewhere the main
context does not have to. Telling it to hurry buys a shallower answer and
nothing else.

**Solution, applied.**

- `budget_line()` deleted from `mcp/proxy.py`; `augment_messages()` lost its
  `hops` parameter.
- The `BUDGET:` paragraph deleted from `shomen.py`'s system prompt, along with
  the "one searching turn left" warning that existed only to soften a
  scheduled landing. The behavioural guidance stays — *"Stop as soon as you
  can answer. Do not explore further out of interest."* — because that is
  direction, not a leash.
- The `hops` key deleted from every tier in `mcp/tiers.py`. The tier ladder is
  now purely cognitive effort: `minimal` → `low` (search) → `medium`
  (+hints) → `high` (+fan-out) → `max` (+deep thinking).
- `MAX_TOOL_HOPS` (proxy) and `MAX_HOPS` (shomen, raised 8 → 16) **stay**, and
  are reframed in code from working limits to **unadvertised runaway
  breakers**. Nothing tells the model about them, a healthy request never
  reaches one, and they exist only because these loops hold one of two GPU
  lanes and an unbounded loop would hold a card forever. `shomen`'s failure
  message now names a tripped breaker as *a tool returning results the model
  cannot act on*, rather than reporting a spent budget.
- Hop **counting** stays everywhere it was: `usage.hops`, `corpus.log_answer`,
  `res["hops"]`. Counting is measurement; gating was the mistake.

**The distinction worth keeping.** A gate shapes normal operation and is
advertised to the thing it constrains. A breaker is silent, is never reached
in a healthy run, and tripping it is a defect report. The same integer can be
either one; what decides it is whether you tell the model about it and whether
you expect to hit it.

**Verified:** 693 checks across six suites, `ruff check mcp bench scripts
--select=E9,F` clean.

---

## Standing rules that came out of this

1. A liveness check may know only whether the process answers (§1).
2. A tool never returns nothing (§2).
3. Report the situation as a fact; `retryable` is a fact, not a suggestion (§3).
4. Never swallow a failure into a result that means something else (§4).
5. A tool on two surfaces assumes nothing about who called it (§6).
6. Run the linter; it is the cheapest test there is (§7).
7. Do not instruct one way and constrain the other (§8).
8. Attribute the measurement before acting on it, and check which axis a number
   is on (§9, and see `docs/PROTOCOL.md` rule 13).
9. A self-test that crashes, or a policy that lives only in a docstring, looks
   exactly like coverage (§12).
10. An abstention is not a decision (§16).
11. A margin is not a confidence. A constant classifier returns a large, stable
    margin; check for degeneracy before reading any threshold (§17).
12. Two data points are not a measurement, and the one that looks best is the
    one most likely to be an artefact of how it was picked (§17).
