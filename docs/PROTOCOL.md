# How work gets done here

This file exists because the same mistakes kept happening, in one day, in this
repo. Every rule below has a named failure behind it. None of them is a
principle someone liked the sound of.

Read this before changing a default, cutting a component, or reporting a
result.

---

## 1. Bring the whole system up before judging any part of it

Both package indexes -- 18,750 chunks of three.js and typegpu -- contained
nothing but zero vectors. The indexer pointed at port 1234, which is the
proxy and does not serve `/v1/embeddings`. Every request 404'd, a per-chunk
fallback substituted `np.zeros`, and the run finished with a progress bar
reading `indexed 15021 chunks`.

Semantic search over dependencies returned whatever `argpartition` happened to
surface from an all-zero similarity array. It had never worked once.

Retrieval quality was then assessed on that index, and the reranker was cut on
the result.

**The rule.** Before measuring anything that spans components, prove each
component is alive, individually, with a call that would fail loudly if it
were not. An index gets checked for non-zero vectors. A server gets asked for
its model list and then asked to do the actual job. "It returned 200" is not
aliveness; "it returned a unit-norm vector" is.

## 2. A fallback that hides an outage is a bug, not resilience

The zero-vector fallback was written so one oversized chunk could not kill a
long run. That is a reasonable thing to want. What made it dangerous is that
it had no ceiling: it behaved identically whether one chunk failed or all
15,021 did.

**The rule.** Every fallback states the rate at which it stops being a
fallback and becomes an outage, and fails the run when it crosses it. The
degenerate case is never allowed to look like the healthy one.

## 3. Verify the harness before believing its numbers

The first benchmark run printed `0.0%` for every condition, every source, and
every metric. That is not a finding about retrieval, it is a broken harness --
the search call was raising and an `except` was turning it into an empty
result list.

**The rule.** A result that is identical across all conditions, or exactly 0
or 100 percent, is a harness bug until proven otherwise. Benchmarks count
errors as a separate outcome from failures and report them. Never let an
exception become a wrong answer.

## 4. State the power before running the experiment

The reranker was cut on 11 queries with p ~= 0.6. Eleven paired queries cannot
detect anything short of an enormous effect. The honest report was "this
experiment could not have answered the question", and instead a decision was
made and a default was shipped.

**The rule.** Before running a comparison, say what effect size the planned n
can detect. If the answer is "only a huge one", either get more n or do not
present the outcome as a finding. Where ground truth can be generated
mechanically, n is a choice, so choose a real one -- 628 tasks came out of the
existing indexes in under a minute.

## 5. Measure the cheapest layer that can answer the question

"Does the reranker help?" is a question about ordering, and ordering is settled
before the model runs. Asking it end-to-end costs a generation per task, which
is why the first attempt could only afford 11. Asking it at the retrieval
layer costs milliseconds, so the same question runs at full n.

**The rule.** Push each question down to the lowest layer that can answer it.
Reserve end-to-end runs for questions that are genuinely end-to-end, and
stratify and sample those.

## 6. Pair the comparison and use the paired test

Tasks differ enormously in difficulty. Comparing two conditions on two
different samples measures the samples. Run both conditions on the same task,
compare within-task, and test the discordant pairs -- McNemar exactly, since
the data is paired and binary. Tasks both conditions get right carry no
information about the difference and must not inflate n.

## 7. Test against what the client actually sends

Repository detection passed 8/8 on synthetic prompts and 4/4 on a documented
format, then returned `None` on the first real Hermes run. The prompts it was
validated against were written by the same person who wrote the detector.
Hermes emits its working directory only from a container-backend probe, so a
local run states no location at all.

**The rule.** A parser, detector or router is only tested by input captured
from the real producer. Log what actually arrives -- `corpus.log_turn` records
the system-prompt head for exactly this reason -- and build fixtures from
that.

## 8. Parse the language, do not pattern-match it

Import extraction used regular expressions and was wrong on its first real
sample: a Python rule matched the JavaScript line `import tgpu from 'typegpu'`
and invented a package called `tgpu`. A lookahead fixed that case. It would
have been wrong again on the next one, because a regex cannot tell that a line
is inside a string or a comment.

Tree-sitter can. `import_statement` is a node type; a package name inside a
string is a `string` node and never matches. Both cases pass now and could not
have been made to pass reliably any other way.

**The rule.** Anything that reads the structure of a language uses a parser.
Regular expressions are for flat text with no grammar -- version strings
pasted out of half a dozen manifest formats, path shapes, log lines.

## 9. Do not delete what you have not measured

The reranker was removed from the default path on evidence that could not
support the decision, and it turned out the index under it was dead. A
component that has not been measured on a working system has not been
measured.

**The rule.** Switch it off behind a flag, keep it runnable, and put it in the
benchmark as a condition. Removal comes after a result, at an n that could
have detected the effect, on a system proven alive. Until then it stays.

> **SUPERSEDED 2026-09-22 — the conflict below was resolved after this note was
> written, and the resolution lives in `docs/PLAN.md` under "Cut criteria",
> backed by `docs/FINDINGS.md` #20.** The reranker is *corrupt as deployed*:
> `/v1/rerank` scores depend on what else is in the same request, so every
> reranker number in this repo — including the n=356 row this note leans on — is
> void, and neither document's instruction can be honoured on void data.
> Nothing has been cut; `RERANK_MAX_K = 2` still gates it. The note is kept
> below unedited because the reasoning is the useful part.
>
> **CONFLICT — unresolved, and both sides are defensible.** `docs/PLAN.md`,
> under "Cut criteria, written before the data", pre-registers the opposite
> instruction: *"the reranker is cut entirely unless it beats embeddings on the
> uncontaminated index."* A pre-registered cut rule exists precisely so a
> decision is not re-litigated after seeing the numbers, and this rule exists
> precisely because that once happened on a dead index. They cannot both be
> followed here.
>
> The reranker **has** since been measured on a live index —
> `bench/retrieval_results.jsonl`, n=356 paired rows, rerank hit@1 13.5%
> against embedding order's 74.2%, 226 discordant losses to 10 wins, and it
> genuinely ran (`bench/retrieval.py` forces `rerank_cutoff=999` for that arm;
> a `rerank_rank` of `0` is a miss, not a skipped call). So the "never
> measured" protection this rule grants it has expired. What has *not* been
> measured is the case it exists for: every query in that run is a bare symbol
> name, which the symbol table already answers at 100% in 0.5 ms.
>
> **What settles it:** the same paired comparison on conceptual,
> natural-language queries with independent gold paths, McNemar on discordant
> pairs, n fixed in advance (rule 4). `scripts/eval_retrieval.py:GOLD_EXTRA` is
> the stub for that query set and is currently declared and never read.
>
> **This note does not rule between the two documents**, and nothing has been
> cut: the reranker is still loaded and still gated at `RERANK_MAX_K = 2`. See
> `docs/PLAN.md` "Cut criteria" and `docs/ROADMAP.md` §1.2 / §1.3 — §1.3's
> explanation for the 199 zero ranks *was* wrong and has since been corrected in
> place; it now says the same thing this note does.

## 10. One experiment is a data point, not a verdict

Rule 9 said do not delete what you have not measured. It is not enough, because
it is satisfied by measuring once and then deleting. That happened the same day
it was written: a recipe-injection experiment was designed with a single
configuration -- one hint form, one injection point, one domain -- and
described as a test that would decide whether the whole idea worked.

It would have decided nothing of the sort. A null result there means that one
cell of a large grid did not move the metric. The hint could have been in the
wrong place, phrased the wrong way, too long, too short, selected badly, or
measured on a substrate where the effect cannot appear. Collapsing all of that
into one bit and calling the idea dead is how a promising mechanism gets thrown
away by someone who felt rigorous while doing it.

Flight was not abandoned after one aircraft failed to leave the ground.

**The rule.** Before running an experiment on a mechanism, write down the
variables that could plausibly determine the outcome, and make each one its own
arm so a null result is attributable to a variable rather than to the idea. A
mechanism is only set aside after a SEARCH -- several points in that space,
ranked, with the best of them still failing -- and the write-up then says which
region was explored and which was not.

Include the bounds. A random selector is the floor and an oracle is the
ceiling: without them a middle result cannot be interpreted, because nobody can
tell whether the mechanism is weak or the selection is. Include a placebo of
the same shape and size as the real intervention, or a gain from adding text
will be read as a gain from adding information.

## 11. Report what happened, not what was hoped for

Where a run failed, say so with the output. Where a step was skipped, say it
was skipped. Where a number is contaminated -- three.js is in every training
set -- label it. Where a conclusion reversed, leave the reversal in the
comment, because the next person needs to know the ground is soft there.

## 12. A monitor that understands the application will eventually kill it

`watchdog.ps1` asked the proxy for a model named `bonsai`. The proxy advertises
`yamadori`, because internal names are implementation detail -- a decision made
weeks later, in a different file, for a good reason. The watchdog found the
model "missing", concluded the stack was dead, and killed llama-swap and every
llama-server with it. Every ten minutes. For hours.

    23:51:02  UNHEALTHY: model 'bonsai' missing  -> restarting
    00:06:02  UNHEALTHY: model 'bonsai' missing  -> restarting
    00:16:02  UNHEALTHY: model 'bonsai' missing  -> restarting

It cost a LiveCodeBench run -- 33 of 50 rows recorded as generation errors --
and the errors were spread evenly through the file because the kills were on a
timer, which is what made them look like a transport fault. Hours went into
blaming the model, then llama-swap, then the connection.

A liveness check may know one thing: whether the process answers. The moment it
knows a model name, a schema, or an expected value, it has acquired an opinion
the system is free to invalidate -- and its response to being wrong is to
destroy the thing it is watching. Restarting cannot fix a wrong answer, and
trying is how a monitor becomes the outage.

It also ran a generation probe with a 120-second timeout against a model whose
answers legitimately take 450 seconds. Busy is not broken.

## 13. Attribute the measurement before acting on it

Twice in one session the same mistake, in opposite directions.

A long request died at ~104s. Streaming survived 388s; non-streaming did not.
The conclusion -- "an idle reaper is cutting non-streamed connections" -- fit
every number. It was wrong: the streamed run happened to fall inside a watchdog
cooldown window, and the next streamed run died at 101s. The correlation was
real and the cause was somewhere else entirely.

Later, a tool result logged `ms: 0.1` alongside 47-second gaps between hops. A
fix was justified as saving "517 seconds". The 47 seconds was the model
generating the next call; the tool itself ran in 0.1 milliseconds. All 24 calls
in that turn totalled 0.7ms. The saving was six orders of magnitude smaller
than claimed, and the real cost -- 524 seconds of generation -- was untouched.

Before a fix, name the mechanism and say what it predicts. Before quoting a
number, check which axis it is on. A measurement that merely *fits* the story
is not evidence for it.

## 14. A self-test that crashes looks like coverage

`python mcp/tiers.py` raised `KeyError: 'budget'` on every run -- the key had
been renamed to `floor` and the self-test was never updated. Nobody noticed,
because nobody ran it, because it was assumed to pass.

`mcp/fanout.py` documents its gating policy in a docstring: lookups get N=1,
design questions get N=4. That policy is not implemented anywhere and the
module is not wired into the proxy. A rule written in prose next to code that
does not follow it reads exactly like a rule that is enforced.

Ruff's `E9,F` on this repo found two undefined names in shipped paths: a
`tracker` that made every streamed request crash the moment it touched one of
our twelve tools, and a `difflib` that crashed the "did you mean" helper. Both
had been there for days. A linter is not style enforcement; it is the cheapest
test that exists, and it found real outages in under a second.

## 15. Do not instruct and constrain in opposite directions

The capability block told the model the tools "have no quota". The loop cut it
off at twelve hops. So it was steered to explore without limit, then
guillotined mid-plan, and the caller received whatever happened to be on the
table -- in one case an answer consisting solely of the proxy's own preamble,
after 842 seconds.

The first fix was to make instruction and enforcement quote the same number.
That was wrong, and the corollary below is why.

**The corollary won the argument, and the budget is now gone entirely.** A
budget does not fix a loop. Given an empty index, the model issued the
identical pair of searches twelve times, and from the second one onward every
result told it so in plain text. It repeated ten more times. A cap would only
have bounded how long it burned.

Plain text was not enough; what ended those loops was a tool result that
states the situation, whether it is retryable **as a fact**, and a remedy with
an owner — plus de-duplication of a repeated empty search. Once tool results
are believable, a loop that continues means a tool is lying, and the repair is
to fix the tool.

**The rule, restated.** Prefer removing the instruction to reconciling the
numbers. Where a ceiling must exist for a physical reason — these loops hold
one of two GPU lanes — make it an unadvertised **breaker**: tell the model
nothing, expect never to reach it, and treat tripping it as a defect report
rather than a budget being spent. A gate shapes normal operation; a breaker
reports a bug. The same integer can be either, and what decides it is whether
you tell the model about it. See `docs/FINDINGS.md` #27.

---

## The short version

Prove it is alive. Make failures loud. Check the harness. Size the experiment
before running it. Measure at the cheapest honest layer. Pair the comparison.
Use real inputs. Parse, do not guess. Keep what you have not measured. Search
the space before calling it dead. Report what happened. Let the monitor know
nothing. Name the mechanism before trusting the correlation. Run the self-test.
Never instruct one way and constrain the other.

## 16. Retrieve the whole answer before narrowing it

Three wrong claims in one session, all the same shape, all about published
package versions:

1. A `dist-tags` probe written as `next or beta or alpha`. It matched `beta`
   first, short-circuited, and discarded the `alpha` key **that was already in
   the response**. Conclusion published: "there is no r3f v10." There is;
   `10.0.0-alpha.5`, with canaries two days old.
2. A version list filtered with `major in ('3','4','5')` — hardcoded from what
   was expected to be there. Anything newer was dropped before it could be
   seen.
3. A query that asked only for `dist-tags.latest`, never listing versions.
   Conclusion published: `postprocessing` is at 6.39.5. It is, but `7.0.0-beta.16`
   and `7.0.0-alpha.4` also exist and were never looked at.

Each time the data was fetched correctly and destroyed on the way to being
read. Each time the operator knew better from daily use and had to say so.

**The rule.** When querying a source you do not control, fetch the whole
answer and narrow it while reading, never while fetching. A filter written
from what you expect to find cannot report that your expectation was wrong —
it returns a clean, confident, empty result, which is indistinguishable from
the thing not existing.

This is the same failure as PROTOCOL rule 1 (an index of all-zero vectors
assessed as if it were alive) and rule 14 (a self-test that crashes looks like
coverage): **a check that cannot fail is not a check.** A narrowing applied
before inspection is exactly that check.

The corollary for this session: "I could not find it" is a claim about the
search, not about the world, and must be stated that way.

