# Roadmap: close the open doors, then tune

`PLAN.md` holds the strategic bet — which domains, why the seam exists, what
winning looks like. This is the execution layer under it: what is unfinished,
what measurement closes each item, and in what order.

## The gating principle

**Benchmarks are last, not first.**

A benchmark measures a model *through* the systems around it. When those
systems are broken the number describes the systems, and it does so without
saying that is what it is measuring. This is not a theory — it is what the
2026-09-21/22 session cost:

- 33 of 50 rows recorded as generation errors, caused by a watchdog killing the
  stack every ten minutes
- twelve identical searches over 842 seconds, caused by tools returning the
  empty string
- multi-hop requests reporting 8% of their token cost, biased toward the arm
  that hopped most

Every one of those was invisible from the benchmark and visible in seconds from
a unit test. So: **no benchmark run until the system under it passes its own
tests.** `mcp/test_tools.py` is that gate for the tool layer and currently
passes 356/356.

## Floors before mechanisms

Every mechanism below is measured against two floors and one ceiling, the
discipline already encoded in `bench/mechanisms/selectors.py`:

| | |
|---|---|
| `random` | the floor. Does not beat a coin, the idea dies here. |
| `fixed` | **the harder floor, and the one that matters.** Always apply the single best option, no selector at all. A mechanism that does not beat this is paying for nothing. |
| `oracle` | the ceiling. A cheat; raises rather than degrades when its map is absent. |

A regex is a legitimate mechanism and has twice been the thing to beat
(0.841 against a trained Laya head's 0.726 on routing).

---

## Step 0 — Measure the claim everything else rests on

**The hemisphere's whole justification is context economy**, in its own words:
*"six thousand tokens of searching becomes a two hundred token finding."* The
thesis -- a second brain stimulating the first into better output, considering
alternatives, distilling something dialectic -- is a CONTEXT-SAVING construct.
If the saving is not real, nothing above it holds.

**It has never been measured.**

This experiment needs none of the unproven machinery: no Laya, no buckets, no
routing gate, no trained head. Same question, same model, same tools. One arm
runs the tools in the main context; the other runs them in a second context and
returns only the finding.

Measured on both arms:
- answer quality (paired, discordant pairs -- not a pass rate)
- **main-context tokens consumed**, which is the claim itself
- wall clock and total tokens across both contexts, because the saving is in
  the main window and the cost is in the total

**Possible outcomes, all useful.** If the second context saves main-context
tokens at equal quality, the construct is sound and every later measurement has
a foundation. If it saves tokens and LOSES quality, the callosum is dropping
something the first brain needed, and that is a design target. If it saves
nothing, the selection work is decorating a claim that does not hold and the
roadmap changes entirely.

Do this first. It is one experiment against the current stack, it is waiting on
nothing, and it was skipped for a full session while the machinery above it was
chased.

---

## Part 1 — Open doors

Measurements that are owed. Each states what closes it.

### 1.1 Does Laya beat ranking at a comparative pick? — IN FLIGHT

The only place the mechanism predicts Laya helps: a mutually exclusive bucket
where the discriminator is contrastive. Everything measured so far tested it on
jobs it structurally cannot do (routing, grounding — the deciding information
was not in the input).

**Closes when:** `bench/hint_collapse.py` reports precision at matched coverage
for embeddings, reranker and Laya on the same buckets, split by contrastive vs
topically-distinct. **Prediction on record:** Laya wins on contrastive buckets,
ties or loses on distinct ones. If it loses on both it has no job in this stack.

### 1.2 Embeddings vs BM25 on conceptual queries — NOT RUN

`CODE_SEARCH_SEMANTIC=0` because BM25 beat embeddings on the gauntlet index
(recall@5 92/120 vs 77/120, McNemar p=0.0041; fusion did not rescue it at
87/120, p=0.2266). The cut rule fired honestly.

**But the caveat that keeps it provisional has never been discharged:** the gold
queries derive from symbol names, which flatters lexical matching. The
conceptual query set that would favour embeddings does not exist.

**Closes when:** a hand-written conceptual query set ("how does this handle
back-pressure", "where is the retry policy") is run over the same index and
BM25 vs embeddings vs fusion are compared on it. Until then, embeddings are
"lost on one biased test", not "lost".

### 1.3 The reranker is measured BAD as deployed — and the deployment is corrupt

**Correction.** An earlier version of this section claimed the 199 zero ranks
were `RERANK_MAX_K = 2` skipping the call, and that the number meant "did not
run". That was wrong, and it was wrong the same way three other things were
wrong in the same session: a mechanism was inferred because it fitted the
number, without reading the code that produced it. See PROTOCOL rule 13.

What the code actually does:

```
bench/retrieval.py:106   cs.RERANK_MAX_K = rerank_cutoff   # that arm passes 999
rank_of()                returns 0 for "gold not in the top-5" = a MISS
rerank_ms                n=356, median 1330ms, min 466ms, rows skipped: 0
```

It ran every time. So the reranker IS measured, paired, at n=356:

| | hit@1 | discordant |
|---|---|---|
| embedding order | 74.2% | 226 wins |
| rerank order | 13.5% | 10 wins |

and it holds on both uncontaminated corpora alone (koota 83-2, typegpu 73-2).

**But that run went through a corrupted path.** `/v1/rerank` scores depend on
what else is in the same request -- reproduced directly, and at scale in the
hint-collapse work: the same 89 documents each queried with their own verbatim
text score **15-16/89 batched against 68/89 one at a time**. `search_code`
batches up to 40 candidates, so every one of those 356 rows was reranked in a
batch. See FINDINGS #20.

**So what is actually established:** the reranker *as deployed* is far worse
than embedding order, decisively, at adequate n. What is NOT established is
whether the model is bad, because the deployment scrambles its scores before
they are read.

**Two separate things now close this:**
1. **Fix or disable the batching.** Until a document's score is independent of
   its neighbours, no reranker measurement means anything. A startup assertion
   on `config.yaml`'s own worked example -- which the 0.6B is documented to
   score 0.9992 and currently scores 6.93e-08 -- turns a silent retriever
   failure into a loud one.
2. **Then re-measure on the right query shape.** Every query in the n=356 run
   is a bare symbol name: the worst possible input for a cross-encoder, and a
   task the symbol table already answers at 100% in 0.5ms. The case the
   reranker exists for is conceptual natural-language queries, and
   `scripts/eval_retrieval.py:GOLD_EXTRA` -- the stub for that set -- is
   declared and never read, which is also why 1.2 is still open.

**Note the ordering.** Rule 9 says do not delete what you have not measured;
`PLAN.md` pre-registers cutting the reranker unless it beats embeddings. Both
documents now carry a mutual cross-reference. Neither can be honoured on
evidence taken through a corrupt path.

### 1.4 The symbol arm's 100% is a tautology — NEEDS INDEPENDENT GROUND TRUTH

```
symbol   hit@1 100.0%   0.5ms
embed    hit@1  74.2%   276ms
```

`bench/tasks.py` builds ground truth from the `defs` table, which is exactly
what the symbol arm queries. It is scored against its own source. We know it is
fast (550x) and structurally sound (163,903 *resolved* references, not string
matches); we do **not** know it is accurate.

**Closes when:** ground truth is derived independently — hand-labelled, or from
a different parser — and the symbol arm is re-scored against it.

### 1.5 Hint selection vs the floors — HARNESS BUILT, NEVER RUN

938 recipes, seven selectors (`random`, `fixed`, `embedding`, `rerank`, `laya`,
`tree`, `oracle`), and `bench/recipe_results.jsonl` does not exist.

**Closes when:** `recipe_oracle.py` establishes the ceiling and every selector
runs on identical inputs. **The decisive comparison is against `fixed`**: if
"always inject the best hint" wins, selection is not worth building and the
corpus should ship unconditionally.

### 1.6 Does shaping the data close the gap? — THE HIGHEST-LEVERAGE UNKNOWN

The corpus states discriminators *implicitly*:

```
"Static array, many range-sum queries: prefix sums…"
"Point updates interleaved with range sums: Fenwick tree…"
```

The deciding fact — *are there updates* — is buried in prose and only becomes a
discriminator relative to the sibling. Every selector is asked to re-derive, per
query, something fixed the moment the bucket was formed.

**Hypothesis:** if System 2 shapes the corpus offline — emitting the bucket, the
question it answers, and each member's discriminating condition **explicitly** —
the runtime choice stops being inference and becomes matching, and *every*
mechanism improves, not just Laya.

**Closes when:** 1.1 is re-run on a shaped version of the same buckets. The
as-is baseline must exist first, or the shaping cannot be priced.

---

## Part 2 — The three systems

In dependency order. Each must pass its own tests before the next is switched
on, and none is tuned until all three work.

### 2.1 Hemisphere (System 2)

The second context that investigates, so the first does not have to.

| | state |
|---|---|
| singleton enforced | **done** — threading semaphore at the call site; 3 concurrent → 1 runs, 2 get `HELPER_BUSY` |
| KV budget enforced | **done** — reads `budget.py` (the helper's 3/8: 55,296 at a 147,456 pool, 61,440 at 163,840); at budget it writes up rather than being cut |
| budget stated to the model | **done** — and one turn's notice before tools are withdrawn |
| argument gate | **done** — a malformed call no longer starts a GPU run |
| tools it calls tell the truth | **done** — 356/356; it loops for the same reasons the main model did |
| Laya distilling the output | **removed** — the grounding question is undecidable without file contents (29/29 constant, AUC 0.667). The deterministic citation check is the authority. |
| **streamed as reasoning** | **TODO** — its output should stream as `reasoning_content` until it collapses back into the primary answer. `as_thinking()` exists; nothing calls it. |
| **run by decision, not by tool call** | **TODO** — depends on 2.2 |

### 2.2 Routing (when System 2 fires)

**Shipped recommendation, from measurement:** regex decides, the Laya head is a
second independent signal, and **disagreement escalates to System 2**.

```
route_in          zero-shot   trained head   regex    majority
accuracy            0.529        0.726       0.841     0.393
abstention          96.2%         2.4%         —         —
adversarial         0.333        0.271       0.000       —
```

| | state |
|---|---|
| trained head + artefact | **done** — `index/laya/route_in.json`, regression floors in tests |
| `/route` endpoint | **done in code**, service needs a restart to expose it |
| regex baseline | **done** — and it is the thing to beat |
| **two-signal gate wired into the proxy** | **TODO** |
| **binary + explicit `continue`** | **TODO** — the framing that worked (8/8 on the investigate node at n=8; 4-way scored 1/6) |
| **adversarial robustness** | **OPEN, and neither mechanism has it.** Regex 0.000, head 0.271. Escalating disagreement is a mitigation, not a fix. |

### 2.3 Hint injection (attention boosters at code-writing time)

The product thesis: a corpus of simple skills, emit only what applies, **and
never harm**.

| | state |
|---|---|
| corpus | 938 recipes, 8 files |
| domain gate | **built, 0 calls** — `domains.py` gates nothing today |
| buckets | **TODO** — and the bucket key is *"what question does this answer"*, not a topic label. `category` (547) and `area` (35) are topic taxonomies and demonstrably wrong: `area=problem_shape_to_data_structure` holds both true alternatives and a constraint that applies alongside one of them. |
| offline shaping | **TODO** — see 1.6 |
| collapse mechanism | **TODO** — decided by 1.1 |
| scoring metric | **precision at matched coverage**, not accuracy. Emitting nothing is a good outcome; a wrong attention-booster is an active harm. |

---

## Part 3 — Remaining systems to test and implement

Built, or half-built, and not wired. Each needs a floor comparison before it
earns a place.

| system | state | what it needs |
|---|---|---|
| `domains.py` | 0 calls from the proxy | wire into **tool injection**, so search tools are not offered where nothing is indexed. Highest value: `aug_on` spent 2,729 extra prompt tokens (87% of its prompt) on a problem where it called zero tools. |
| `fanout.py` | 0 calls; gating policy lives only in a docstring | measured null on lookups (7/8 vs 7/8, **zero discordant pairs**, 3.2x cost). Needs a task class with headroom before any trigger work. |
| `concept_seed.py` | 0 calls | diversity *within* a fan-out, not a trigger. Only matters once fan-out has a job. Seed belongs in the **user** message, one per sub-agent run. |
| tier flags | `hints`, `fanout`, `investigate` never read outside `tiers.py` | `tier["retrieval"]` is the only flag the proxy reads, so `low`/`medium`/`high`/`max` produce identical requests while `describe()` advertises four features. Either wire them or stop advertising them. |
| `judge` | unreachable since the surface gate | **remove.** Misapplication on three counts: inverts the architecture (System 1 as a tool System 2 calls), sends bare labels with empty criteria (the discriminator is not in the input), and scored 6/10 against a 5/10 coin flip. Its own `eval_judge.py` verdict branch says "do NOT expose". |
| `code_search.py` MCP stdio surface | live, unmaintained | decide whether a zero-config remote product keeps a second entry point that nothing tests. |
| package corpus | **only `three@0.185.1` and `typegpu@0.12.5`** | not drei, not r3f v9/v10-alpha, not koota. Most target domains have no source to retrieve. This blocks any domain benchmark. |
| live-model tool tests | written, never run | `mcp/test_tools_live.py`, opt-in, needs the card |
| `_PATH` citation regex | **CLOSED** — `shomen._PATH` now carries `yaml|yml`, ordered longest-first behind a negative lookahead | was: `config.yaml` is where half the answers live, so the citation check silently under-credited findings. See `docs/FINDINGS.md` #25 |

---

## Part 4 — Sequence

1. **Close 1.1** (in flight). It decides whether Laya has a job at all.
2. **Close 1.6** — shape a sample of the corpus, re-run 1.1 on it. This decides
   whether hint selection is a corpus problem or a selector problem, and it is
   the difference between tuning a mechanism and fixing the data it reads.
3. **Wire `domains.py` into tool injection.** Cheapest real win available; stops
   tools being offered where they cannot help.
4. **Build buckets** for a slice of the corpus, keyed on the question answered.
   Hand-authored if needed — 938 is small, and System 2 can propose groupings
   for review offline.
5. **Wire the two-signal routing gate** (regex + head, disagreement escalates),
   binary with an explicit `continue`.
6. **Stream the hemisphere as reasoning**, so its work is visible without
   entering the primary KV.
7. **Close 1.5** — hint selection against `fixed`. If `fixed` wins, ship the
   corpus unconditionally and delete the selector.
8. **Close 1.2, 1.3, 1.4** — the three owed retrieval measurements. Independent
   of the above, can run whenever the card is free.
9. **Then, and only then, benchmarks** — with all three systems on, tuned
   against a task set inside the target domains rather than LiveCodeBench, which
   has no repository and nothing to retrieve.

## What benchmarks are for, when we get there

Not to produce a number. To find the **discordant pairs** — the problems where
one arm succeeds and another fails. That set is the entire signal: it says where
a mechanism earns its cost, and if it is empty the mechanism is free to delete.
Pass rates without discordant pairs describe the task set, not the system.
