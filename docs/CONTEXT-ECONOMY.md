# Context economy: measuring the claim the hemisphere rests on

`mcp/shomen.py`'s own docstring states the justification for the entire
dual-hemisphere construct:

> Six thousand tokens of searching becomes a two hundred token finding.

`docs/ROADMAP.md` Step 0 says it has never been measured. This is that
measurement, and the answer is:

**The second context buys real, large, consistent main-context economy — 26/26
paired, a 71% cut in main-context tokens and a 59% cut in peak window occupancy
— with no detectable quality cost at this n, and it costs 2.06x the total
tokens and 2.76x the wall clock to get it.**

The saving is real. The bill is real. Both belong in the same sentence.

| | arm A: main-context | arm B: second-context |
|---|---|---|
| correct (path + every fact) | 25/26 | 26/26 |
| correct on **facts alone** | 26/26 | 26/26 |
| **main-context tokens, median** | **8,326** | **2,410** (0.29x) |
| **main-context peak window, median** | **3,670** | **1,498** (0.41x) |
| main-context peak window, worst row | 8,790 | 2,622 |
| total tokens, both contexts | 303,531 | 623,779 (2.06x) |
| wall clock, total | 737 s | 1,752 s (2.38x) |
| errors | 0 | 0 |

Arm B was run twice (52 + 26 rows, 78 generations, 0 errors). **The
main-context saving replicates tightly and the total-token price does not** —
see "Replication" below. Where the two runs differ the range is given, not the
prettier number.

---

## Method

### Two arms, one difference

| | |
|---|---|
| **A — main-context** | the model runs the search tools itself. Every tool result and all the scaffolding stays in the context that also has to hold the answer. |
| **B — second-context** | a main context holding only the question calls `delegate_investigation`; the searching happens in the hemisphere and only the finding crosses back. |

**Arm A is `shomen.investigate()` itself.** The obvious way to build arm A
is to write a second tool loop with a main-assistant system prompt. That adds a
prompt difference, a loop difference and a stopping-rule difference on top of
the one difference under test, and any of them could produce the result. So arm
A calls `investigate()` — identical loop, identical system prompt, identical
tools, identical 8-hop budget — and counts its spend as **main**-context
tokens. That is exactly the counterfactual the docstring names: *"Done in the
user's own context, all of it stays there."*

> **Since this run, the ADVERTISED hop budget has been deleted**
> (`docs/FINDINGS.md` #27). Stated precisely, because the loose version of
> this sentence is wrong: what was removed is the budget the model was *told*
> about — `shomen.py`'s `BUDGET:` paragraph and the tier's `hops` dial. A
> ceiling still exists, `MAX_HOPS`, raised 8 → 16 and no longer named in any
> prompt, kept as a silent runaway breaker so an unbounded loop cannot hold a
> GPU lane forever.
>
> The comparison above stays valid — both arms carried the *same* budget, so
> it cannot explain the difference between them — but the **magnitude is
> re-measurable**, not the direction. With nothing telling it to stop early,
> the deep-thinking context may search longer on questions that warrant it,
> which moves `helper_tokens` and `ctx_peak` without touching what lands in
> the main window. Re-run before quoting the ratio as current.

Arm B runs the same `investigate()` and adds a main context on top. The arms
are the same work differing only in who pays for it.

**Arm B's main model is offered only `delegate_investigation`.** With the search
tools as well it could decline to delegate, and the arm would silently become
arm A on some fraction of rows — self-selection destroying the pairing. Forcing
the channel is a stated design choice: this measures what delegation costs and
buys, not how often a model picks it.

**Modules are driven directly, not through the proxy.** `bench/context_economy.py`
imports `shomen` and `code_search` and posts to the upstream on `:11434`.
The proxy's preamble, tier gate and tool merge are not under test and would land
on the two arms unevenly. `bench/livecodebench.py:api_key()` is still used, in
the preflight, to prove `:1234` is up and keyed.

### What is counted, and on which axis

Two token numbers get conflated constantly (PROTOCOL rule 13), so all three are
reported separately:

| name | definition | what it answers |
|---|---|---|
| `tokens_cum` | Σ `usage.total_tokens` over every hop | what the work **costs**. In a re-prefilled tool loop hop *k* re-pays for the whole prefix. This is the figure `helper_tokens` reports. |
| `ctx_peak` | max(`prompt_tokens` + `completion_tokens`) over hops | how much **window** the work occupies at its widest — the axis the context-economy argument is actually about, "the window whose recall degrades". |
| `callosum` | growth of the main context's `prompt_tokens` between the hop that issued the delegation and the hop that read the finding | what actually crossed back, counted by the server's own tokenizer. |

`investigate()` returns only a summed `helper_tokens`, so per-hop usage is
captured by instrumenting `shomen._post` from the harness. Nothing under
`mcp/` is edited; it is a probe. **Harness check: the probe's sum equals the
module's own accumulated figure on every one of 52 rows, drift min 0 max 0** —
which is also the check that today's accumulation fix is really in force.

### Grading is deterministic

Each task carries required file paths and required facts with explicit accepted
alternatives, hand-written against the indexed source. A row is **correct** when
it cites a required path **and** states every required fact. No model judges
anything, so grading is a pure function of the stored answer text and re-runs
over an existing results file without spending a token (PROTOCOL rule 5).

### The task set

`bench/context_economy_tasks.jsonl`, **26 questions, hand-written** — as
expected and as stated. Three shapes: `defs_refs` (6), `module` (10), `version`
(10). Ground truth was verified against both the index and the on-disk tree
before any generation ran.

### Power, stated before the result (PROTOCOL rule 4)

n=26 paired rows. McNemar's exact test reaches p<0.05 only at 6+ discordant
pairs with a lopsided split. **At this n the quality comparison can detect a
large effect and nothing smaller.** The token comparison is a different matter:
it is near-deterministic per row, every one of 26 rows points the same way, and
it does not need n to be readable.

---

## Results

### 1. Main-context tokens — the claim itself

| measure | arm A | arm B | B/A |
|---|---|---|---|
| cumulative tokens, median | 8,326 | 2,410 | **0.29** |
| cumulative tokens, mean | 11,674 | 2,662 | 0.23 |
| peak window occupied, median | 3,670 | 1,498 | **0.41** |
| peak window occupied, worst row | 8,790 | 2,622 | 0.30 |

**The second context used fewer main-context tokens on 26 of 26 tasks. Zero
exceptions. Exact sign test p = 2.98e-08.** Median saving 5,777 tokens per
task; range 1,891 to 43,401. Peak-window saving: median 2,100, range 649 to
7,066.

The peak figure is the one the architecture argument is about, and it is the
more conservative of the two: arm B's main context never exceeded 2,622 tokens
on any row, while arm A's reached 8,790.

Part of arm B's main-context saving is not the finding at all — it is that arm
B's main context never carries the six search-tool schemas. That is a genuine
property of the design, not a confound, but it means the saving is not
attributable to compression alone.

### 2. Answer quality — no detectable cost, and the one discordant pair is not about content

| | arm A | arm B |
|---|---|---|
| correct (path + facts) | 25/26 (96.2%, CI 81.1–99.3) | 26/26 (100%, CI 87.1–100) |
| cited a required path | 25 | 26 |
| stated every required fact | 26 | 26 |

Discordant pairs: b = 0 (arm A only), c = 1 (arm B only). **McNemar exact
p = 1.0000.**

**On facts alone both arms score 26/26 and there are ZERO discordant pairs.**
The single discordant row, `ce17` (ChainMap), is a citation-format difference:
arm A's answer is factually correct and quotes the right line numbers but never
writes the file path, so the deterministic grader fails it. Arm B names
`renderers/common/ChainMap.js` because the hemisphere refuses uncited findings
and hands the path forward.

That is a real property of the architecture — delegation structurally enforces
citation — but it means **arm B's 26/26 is partly the grader rewarding the
mechanism's own citation discipline, not better retrieval.** Reported here
rather than banked.

### 3. The total-token price

| arm | main ctx | second ctx | total |
|---|---|---|---|
| main-context | 303,531 | 0 | **303,531** |
| second-context (run 1) | 69,205 | 554,574 | **623,779** (2.06x) |
| second-context (run 2) | 68,142 | 465,102 | **533,244** (1.76x) |

**Delegation costs 1.76x–2.06x the total tokens.** Arm B was dearer in total on
23 of 26 rows in both runs, cheaper on 3; sign test p = 8.8e-05.

The spread between the runs is itself the finding: the main-context half is
stable to 1.5% and the investigator half swings 16%, because the investigator's
workload is set by a question the main model writes fresh each time (N1).

### 4. "Six thousand tokens of searching becomes a two hundred token finding"

Measured directly:

| | run 1 median | run 2 median |
|---|---|---|
| investigation spent | **15,798** tokens | 10,764 |
| investigation peak window | 5,344 tokens | — |
| what actually crossed back | **531** tokens (range 292–1,162) | 516 |
| compression across the callosum | **29.8x** | 20.9x |
| crossing-back as a share of the investigation | 3.2% | 4.8% |

**The sentence is directionally right and wrong in both magnitudes.** The
searching is 1.8x–2.6x larger than "six thousand" and the finding is 2.6x
larger than "two hundred" — and the finding size is the stable number of the
two, so it is the one the docstring had least excuse to miss. The ratio the
sentence implies, 30:1, happens to land inside the measured 21–30x only because
both of its errors point the same way.

`MAX_FINDING_CHARS = 1400` caps a finding at roughly 350 tokens, so the 531
median is mostly *not* the prose: it includes the assistant's tool-call message
and the proxy's cost note, which are part of what the main context pays and are
counted here on purpose.

### 5. Wall clock

| arm | median | mean | min | max | total |
|---|---|---|---|---|---|
| main-context | 21.4 s | 28.3 s | 11.7 s | 112.4 s | 737 s |
| second-context | 59.0 s | 67.4 s | 22.5 s | 183.5 s | 1,752 s |

**Arm B takes 2.76x the wall clock at the median.** Nothing here ran anywhere
near the 450 s that PROTOCOL rule 13 warns about; the longest single row was
183.5 s.

### 6. By task kind

| kind | n | A correct | B correct | A main tok (median) | B main tok (median) |
|---|---|---|---|---|---|
| defs_refs | 6 | 6 | 6 | 7,706 | 2,108 |
| module | 10 | 9 | 10 | 12,414 | 2,501 |
| version | 10 | 10 | 10 | 7,646 | 2,449 |

The saving is uniform across shapes. It is largest on `module` questions, which
are the ones that need the most searching — the direction the construct
predicts.

### 7. Replication (PROTOCOL rule 10: one experiment is a data point)

Arm B was run a second time over the same 26 tasks, capturing every delegated
question. Arm A was not re-run, so arm A is the fixed comparator in both.

| | run 1 | run 2 | stable? |
|---|---|---|---|
| main-context tokens, median | 2,410 | 2,382 | **yes** |
| main-context peak, median | 1,498 | 1,542 | **yes** |
| callosum, median | 531 | 516 | **yes** |
| fewer main-context tokens than arm A | 26/26 | 26/26 | **yes** |
| correct | 26/26 | 26/26 | **yes** |
| investigator tokens, median | 15,798 | 10,764 | **no, −32%** |
| total-token price | 2.06x | **1.76x** | **no** |
| wall-clock price | 2.38x | 1.93x | no |
| compression across callosum | 29.8x | 20.9x | no |

**The primary claim is the stable half.** Everything that describes the main
context reproduces to within 3%. Everything that describes the investigator
swings by a third, because the investigator's effort is driven by a question
the main model writes fresh each time (N1).

So the honest statement of the price is a **range, 1.76x–2.06x total tokens**,
not a point estimate — and a point estimate taken from one run would have been
up to 17% wrong.

**One grader false negative, found by the replication and disclosed as
post-hoc.** Run 2's `ce12` answer — *"It extends the abstract base class
`Backend`, which is defined in `renderers/common/Backend.js` at line 20"* — is
completely correct and was scored wrong because the task's accepted-phrasing
list held `extends backend` and five siblings but not `extends the abstract
base class Backend`. The fact group was widened to require the base-class file
plus the word `extends`, which cannot be satisfied without answering the
question. **This was changed after seeing the data**, which is exactly the move
PROTOCOL rule 3 is suspicious of, so: it does not move run 1 at all (both arms
already passed `ce12`), and it moves run 2 from 25/26 to 26/26. Every other
task is untouched. Grading is a pure function of the stored text, so the
re-grade cost nothing and can be repeated by anyone.

---

## Negative results, all of them

Every one of these is a place the construct is worse than its own description.

### N1. Delegation does not move the work — it nearly doubles it

The investigator, running the identical loop on the identical index, spends
**1.39x (run 1) / 1.15x (run 2)** what arm A spends at the median, makes 6.85
tool calls against arm A's 3.92, and hits the 36,864-token helper budget on
**4 of 26 rows against arm A's 1**. The delegated question is *harder to answer*
than the one a human wrote.

**Mechanism, named before it is acted on (PROTOCOL rule 13): the main model
rewrites the question, and the rewrite can invent context that does not exist.**
Captured verbatim on `ce09`:

> task asked: *"PMREMGenerator has several deprecated `*Async` methods…"*
> model asked: *"**In the pymrem Python library**, PMREMGenerator has several
> deprecated methods ending in Async (e.g. runAsync, generateAsync, etc.)…"*

There is no `pymrem` Python library. The main model was never told what the
index contains, so when the tool description tells it *"the investigator cannot
see this conversation, so name the symbols, files or versions it needs"*, it
complies by inventing them. That row cost 64,460 tokens and 18 tool calls — six
times what arm A spent on the same question — and still landed on the right
answer, so the damage is invisible in the quality table and shows up only in the
bill.

It is not a one-off. All 26 delegated questions were captured in the
replication run, and the model invents an ecosystem repeatedly:

| task | what the model told the investigator | truth |
|---|---|---|
| ce02 | "In the TSL (**TypeScript Style Linter / ts-lint**) codebase" | TSL is three.js Shading Language |
| ce05 | "In the TSL (**TypeScript Language**) codebase" | same |
| ce10 | "In the TSL (**Trellis Shader Language**) … find the math node exports file (e.g. **math.ts**)" | same, and the file is `.js` |
| ce09 | "deprecation warnings (e.g. **warnings.warn** or a deprecation decorator)" | Python idiom; this is JavaScript |
| ce25 | "the Animation module … likely a file named **animation.py**" | `renderers/common/Animation.js` |
| ce06 | "from a **@deprecated Javadoc annotation** or similar" | JavaScript comment |
| ce19 | "likely a file like **Info.ts** or Info.js" | `.js` |

**But the mechanical version of the hypothesis is only partly supported, and
it is a post-hoc split, so it is reported as exploratory rather than as a
result.** Classifying rows by whether the delegated question contains a
non-JavaScript marker (`.py`, `.ts`, `python`, `typescript`, `ts-lint`,
`warnings.warn`, `javadoc`):

| | n | median investigator cost vs arm A | median tool calls |
|---|---|---|---|
| names a wrong ecosystem | 7 | **1.92x** | 8.0 |
| does not | 19 | **1.15x** | 5.0 |

The direction is right, but the two most expensive rows in the whole set —
`ce03` at 4.61x and `ce04` at 4.14x — carry **no** wrong-ecosystem marker.
What they do carry is speculative *search instructions*: *"search for 'class
Matrix3' and any file named matrix3"*, *"Look for …"*. So the broader and
better-supported statement is that **the rewrite adds guessed context of every
kind — ecosystem, filename, artefact — and the investigator dutifully chases
all of it.** Naming only the ecosystem error would be attributing the cost to
the part that happens to be quotable.

Where the rewrite is faithful and merely adds precision, delegation is
*cheaper* than arm A: `ce12` 0.31x, `ce26` 0.75x, `ce11` 0.79x.

**Solution.** The delegation tool must state what is indexed before the main
model writes the question. The `describe_index` tool in `mcp/code_search.py`
already produces exactly that text and `mcp/domains.py` exists to gate on it (ROADMAP Part 3 records it at
zero calls). Injecting the index description — roots, languages, package
versions — into `delegate_investigation`'s description, or requiring the
investigator to call `describe_index` on hop 0, would stop the main model
inventing a domain. Second, cheaper: the hemisphere could pass the caller's
original question alongside the rewrite, so an invented framing is correctable
from inside.

### N2. The advertised finding size is 2.6x optimistic, and that number is stable

"A two hundred token finding" is 531 at the median in run 1 and 516 in run 2,
and reaches 1,162. A caller budgeting on the docstring will under-provision by
roughly 3x. Unlike the cost figures, this one replicates, so there is no
sampling excuse for it.

**Solution.** Quote the measured number. The tool description already tells the
model "a few hundred tokens instead of several thousand", which is accurate;
the module docstring is the one that is wrong, and it is the sentence the
roadmap treats as the claim.

### N3. The quality result is under-powered and cannot be banked

n=26 with one discordant pair cannot distinguish "no quality cost" from "a
quality cost this experiment could not see" (rule 4). What it does establish is
that the callosum is **not** dropping something catastrophic: on facts alone,
26/26 both arms, zero discordant pairs.

**Solution.** The task set is mechanically extensible — `defs`/`refs` in the
index generate `where is X defined / what calls it` at any n. 200 rows costs
about 4 hours on one card at the measured 21 s/59 s per row and would detect a
10-point quality difference.

### N4. The whole result is contaminated, by construction

The live index is three.js, which is in every training set. Half the task set is
built from `@deprecated rNNN` markers in a `187dev` working tree specifically to
blunt this, and the contamination watch shows **zero rows answered with zero
tool calls** — every row did retrieve. But a model that already knows the answer
searches differently from one that does not, and that affects both arms in ways
this design cannot separate.

**Solution.** Re-run on `index/repos/koota-6955afb4.sqlite3` or the `glyph`
index, neither of which is public. The harness takes `CONTEXT_ECONOMY_INDEX`
and the task set is the only thing that needs rewriting.

### N5. The brief's description of the index is wrong

`index/code.sqlite3` is **not** "koota, three and typegpu". The counts are right
— 7,742 chunks / 5,874 defs / 163,903 refs — but every chunk comes from one
root, `C:\Users\jwals\tj-test\src`, a three.js tree at `REVISION = '187dev'`.
koota is in `index/repos/koota-6955afb4.sqlite3` and inside
`index/gauntlet.sqlite3` (1,351 chunks); typegpu is in
`index/packages/typegpu@0.12.5.sqlite3`. Neither is in the live index.

**Solution.** `describe_index` reports chunk counts and file names but not
roots. Adding the root list to its output would have made this visible from
inside the stack instead of from a sqlite prompt — and it is the same fix N1
needs.

### N6. Bugs found in `mcp/`, reported not fixed (read-only, as instructed)

1. **`shomen._cited()` strips characters, not a prefix.**
   `p.lstrip("./")` is a character-set strip, so `.eslintrc.js` becomes
   `eslintrc.js` and any path whose first characters are `.` or `/` is
   silently mangled. Confirmed live. Low impact on this index (all paths are
   ordinary `.js`), latent everywhere else. `removeprefix("./")` is the fix.
   **FIXED SINCE.** `shomen._cited()` now prefix-strips by loop and calls
   `p.lstrip("/")` — leading separators only, never dots. Recorded as
   `docs/FINDINGS.md` #25.1.

2. **`shomen._PATH` still does not match `yaml`/`yml`.** Already on the
   roadmap (Part 3); confirmed live — `_cited("config.yaml and a.py")` returns
   `{'a.py'}`. `config.yaml` is where half this stack's answers live, so the
   citation check under-credits every finding that cites one.
   **FIXED SINCE.** `shomen._PATH` now carries `yaml|yml`, with the extensions
   ordered longest-first and guarded by a negative lookahead. Recorded as
   `docs/FINDINGS.md` #25.2.

3. **`investigate()`'s returned `hops` is a tool-call count, not a hop count.**
   `_finish` sets `"hops": len(trace)` and `trace` gets one entry per tool
   call. On a row where the model made 1 tool call across 2 model turns, the
   result says `hops: 1`. The proxy's own text says "*N* tool calls", so the
   production string is right and the key name is the hazard — anything reading
   `res["hops"]` as model turns is wrong. Measured here: arm A median 3 tool
   calls against 3 model hops, but they diverge per row.

4. **`investigate()` exposes only a summed `helper_tokens`, never per-hop
   usage.** Peak window occupancy — the quantity the module exists to reduce —
   is not observable from its return value. This experiment had to instrument
   `_post` to see it. A `peak_tokens` field alongside `helper_tokens` would
   make the module's own headline claim self-measuring.

### N7. The deterministic grader produced a false negative

Run 2's `ce12` — *"It extends the abstract base class `Backend`, which is
defined in `renderers/common/Backend.js` at line 20"* — is a perfect answer and
was marked wrong, because the accepted-phrasing list held `extends backend` and
five siblings but not `extends the abstract base class Backend`. Deterministic
grading buys reproducibility and pays for it in brittleness, and one row in 78
generations is the measured rate of that.

**Solution.** Fact groups should require *identifiers* (`Backend.js`,
`extends`, `r185`, `setName`) and never a sentence shape, because a sentence
shape is a guess about phrasing rather than a statement about the code. The
task was corrected that way and both files re-graded; the change is post-hoc
and its effect is stated in full under "Replication" (run 1 unmoved, run 2
25/26 → 26/26).

### N8. Nothing here tested the mechanism as the proxy actually ships it

Both arms drive `shomen.investigate()` directly, which is the right call
for attribution and is why the result is clean. It also means **the proxy's
tier gate was never exercised**: `investigate` is on only at tier `max`
(`mcp/tiers.py`), and ROADMAP Part 3 records that `tier["retrieval"]` is the
only flag the proxy reads, so `low`/`medium`/`high`/`max` currently produce
identical requests. A saving measured on the module says nothing about whether
the shipped path can reach it.

**Solution.** Re-run one arm through `:1234` with `X-Yamadori-Features` set,
comparing against these module-level numbers. Any gap is proxy overhead, and it
is measured rather than assumed. That is a small run — 26 rows, one arm — and it
is the last step before this result can be quoted as a property of the product
rather than of the module.

---

## Answering the question directly

**Does a second context buy main-context economy?**
Yes, decisively. 26/26 paired, p = 2.98e-08. Main-context tokens fall to 0.29x
and peak window occupancy to 0.41x. The worst arm-B row occupied less window
than the median arm-A row.

**At what quality cost?**
None that this experiment can see, and it could only have seen a large one. On
facts, both arms are 26/26 with zero discordant pairs. The single discordant row
is a citation the architecture enforces and arm A omitted, so if anything the
grader flatters arm B.

**What is the total-token price?**
**1.76x–2.06x total tokens across two runs**, and 1.93x–2.76x wall clock. It is
quoted as a range because it does not replicate to a point estimate, while the
main-context saving does.

And the price is not the irreducible cost of a second context. The investigator
runs the identical loop on the identical index and still spends 1.15x–1.92x
what arm A spends, because **it is sent a question the main model rewrote,
often with invented context** (N1). Where the rewrite is faithful, delegation is
*cheaper* than doing it inline. That is a fixable defect in how the delegation
is framed, not a property of having two contexts — which means the measured
2x is an upper bound on the real price, not the floor.

**So which of the three roadmap outcomes is this?**
"Saves tokens at equal quality" — the construct is sound and the work above it
has a foundation. With the caveat that equal-quality is established only against
large differences, and with N1 outstanding: the economy claim holds, the
efficiency claim does not yet.

---

## Reproducing

```
python -X utf8 bench/context_economy.py                 # full run, resumable
python -X utf8 bench/context_economy.py --report-only   # re-print every table
python -X utf8 bench/context_economy.py --only ce01,ce02
python -X utf8 bench/context_economy.py --arms second_context
```

Rows append as they finish. A row carrying `error` is **not** a completion and
is retried on the next run — a previous benchmark was poisoned by a resume path
that counted errored rows as done. `YAMADORI_CORPUS_DB` is redirected to a temp
path at import, before `code_search` loads, so no run can touch
`index/corpus.sqlite3`.

Files: `bench/context_economy_tasks.jsonl` (26 tasks),
`bench/context_economy.py` (harness), `bench/context_economy_results.jsonl`
(52 rows, 0 errors), `bench/context_economy_rewrite.jsonl` (arm-B replication
capturing every delegated question, for N1).
