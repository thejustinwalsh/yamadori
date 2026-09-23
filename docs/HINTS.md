# Collapsing a mutually exclusive bucket of hints: does Laya's joint scoring win?

Measured 2026-09-22 against the live services: Laya on `http://127.0.0.1:1237`
(`convaiinnovations/laya`, 421M ModernBERT-large, A4000), and Qwen3-Embedding
and Qwen3-Reranker-0.6B behind llama-swap on `http://127.0.0.1:11434`. Laya is
deterministic, so every Laya number here is reproducible rather than sampled.

Regenerate everything with:

```
python -X utf8 bench/hint_collapse.py               # live; writes the cache
python -X utf8 bench/hint_collapse.py --replay      # every table, no GPU
python -X utf8 bench/test_hint_collapse.py          # 47 assertions, no GPU
python -X utf8 bench/test_hint_collapse.py --live   # 52, incl. the serving bug
```

| artefact | what it is |
| --- | --- |
| `bench/hint_buckets.jsonl` | 19 hand-curated buckets of mutually exclusive hints, 89 members, drawn verbatim from the 938 recipes in `bench/recipes/*.jsonl`. Each carries `why_exclusive`. |
| `bench/hint_probes.jsonl` | 89 developer problem statements, one per member, ground truth by construction. |
| `bench/hint_collapse.py` | the harness. Prints every table below. |
| `bench/test_hint_collapse.py` | every claim below as an assertion that fails when it stops being true. |
| `bench/data/hint_collapse_runs.json` | raw per-ordering Laya probabilities, raw cosines, raw rerank scores. ~390 KB, replays with no GPU. |

---

## THE QUESTION

Laya is an option-scoring cross-encoder. Its input is

```
[CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]
```

with one scalar per option marker and a **softmax across the markers in a
single forward pass**. The options are therefore normalised *against each
other*. `code_search.embed()` scores every candidate independently and compares
afterwards; `code_search.rerank()` cross-encodes `(query, doc)` one document at
a time. Joint normalisation over a set where exactly one member can be right is
the one property nothing else in this stack has, so it is the one place the
mechanism predicts Laya should win.

**The prediction under test, stated in advance:** *Laya beats both on
CONTRASTIVE buckets (alternatives separated by one condition), and ties or
loses on TOPICALLY DISTINCT ones.*

---

## VERDICT

**REFUTED, and not narrowly.** On the contrastive buckets — the half where the
mechanism predicted a win — Laya scores **29.8%** against embeddings' **57.4%**,
paired McNemar p=0.011 against it (5 items Laya alone got right, 18 embeddings
alone got right). Laya does not reach embeddings' precision at **any** coverage
point on any slice. It also loses to a six-line token-overlap floor (33.7% vs
51.7% overall), which is the same shape as `docs/LAYA.md` finding 11.

**The best Laya configuration found does not change the answer.** Shortening
each option from the whole recipe to its applicability condition takes Laya from
33.7% to 44.9% overall and 29.8% to 36.2% on contrastive buckets — a real
improvement, still a clear loss to embeddings, still degenerate.

**Laya has no job in hint selection.** Use `code_search.embed()` with the
asymmetric query prefix, at a cosine-margin gate. It is more accurate, it is
already deployed, and its margin is an honest confidence where Laya's is not.

**The obvious data-side fix for the contrastive gap also fails.** Embedding each
hint's applicability condition instead of the whole recipe — the change that
buys Laya 11 points — costs embeddings 14.6 points overall and 12.7 on the
contrastive slice it was meant to rescue. Cosine needs the whole passage; Laya
is hurt by it. See Finding 5.

**A separate and unrelated finding fell out of the control arm:** the
`/v1/rerank` endpoint **contaminates scores across a batch**. Sent four
documents at once it retrieves a document from its own verbatim text 15–16 times
out of 89; sent one document per call it gets 68/89. Everything in this stack
that batches a rerank call is currently getting a corrupted ranking. See
Finding 4 — that is the most actionable thing in this document.

---

## THE DATA, AND WHY IT IS BUILT THIS WAY

938 recipes across eight files. The brief's warning is correct and was verified:
**`area` and `category` are not the exclusion key.** `area=problem_shape_to_data_structure`
contains genuine alternatives (prefix sums / Fenwick / lazy segment tree) *and*
`complexity:24`, "Segment trees need the combining operation to be associative",
which applies **alongside** one of them rather than competing with it. A bucket
keyed on `area` would score a mechanism for failing to choose between a choice
and a constraint on that choice.

So the key is **"what question does this bucket answer"**, curated by hand.
`complexity:24` and `complexity:26` are explicitly excluded from
`range_structure`, and `bench/test_hint_collapse.py` asserts that exclusion so
the trap cannot be re-introduced.

| | buckets | members | probes | bucket size |
| --- | --- | --- | --- | --- |
| **contrastive** — alternatives separated by ONE condition | 11 | 47 | 47 | 3–6 |
| **topical** — alternatives that are obviously different subjects | 8 | 42 | 42 | 5–6 |
| **total** | **19** | **89** | **89** | mean 4.7 |

Chance, averaged over probes as 1/bucket-size, is **21.3%**.

Contrastive buckets: `range_structure`, `shortest_path`, `string_search`,
`input_size_budget`, `associative_container`, `zig_allocation_strategy`,
`ts_inference_control`, `ts_type_testing_tool`, `r3f_repeated_objects`,
`sorting_decision`, `rust_allocation_fix`.

Topical buckets: `accessibility_finding`, `craft_finish_failure`, `motion_rule`,
`component_behaviour_bug`, `api_convention`, `colour_critique`,
`r3f_core_mechanism`, `complexity_surprise`.

`input_size_budget` is the sharpest contrastive test in the file: every member
is the same sentence about the same subject with a different number in it
(n≤10 → n!, n≤20 → 2ⁿ, 25–40 → meet in the middle, n≤5000 → n², n≈10⁵ → n log n,
values to 10¹⁸ → log n). Nothing topical separates them at all.

**Probes deliberately avoid the hint's own vocabulary.** "Point updates
interleaved with range sums" becomes "individual players' numbers are revised a
few thousand times a second, one at a time". The maximum Jaccard overlap between
any probe and its own correct hint is 0.22, asserted in the tests, and the
`lexical` floor below is the quantitative version of the same check.

---

## Finding 1 — Laya loses on the contrastive buckets, which is where it was supposed to win

Accuracy at full coverage, argmax, nobody abstains. n=89 / 47 / 42.

| arm | overall | contrastive | topical | contr − top |
| --- | --- | --- | --- | --- |
| random | 21.3% | 27.7% | 14.3% | +13.4pp |
| fixed (always the bucket's first member) | 21.3% | 23.4% | 19.0% | +4.4pp |
| **lexical** (token overlap, no model, ~1 ms) | 51.7% | 46.8% | 57.1% | −10.3pp |
| **embedding** (Qwen3-Embedding cosine) | **71.9%** | **57.4%** | **88.1%** | −30.6pp |
| embedding (condition-only documents — Finding 5) | 57.3% | 44.7% | 71.4% | −26.7pp |
| rerank (one doc per call) | 22.5% | 27.7% | 16.7% | +11.0pp |
| rerank (batched — see Finding 4) | 21.3% | 25.5% | 16.7% | +8.9pp |
| **laya** (joint, permutation-averaged) | 33.7% | 29.8% | 38.1% | −8.3pp |
| **laya** (condition-only options) | 44.9% | 36.2% | 54.8% | −18.6pp |
| chance | 21.3% | | | |

Paired McNemar, exact, two-sided:

| pair | slice | a only | b only | discordant | p |
| --- | --- | --- | --- | --- | --- |
| laya vs embedding | contrastive | 5 | 18 | 23 | **0.011** |
| laya vs embedding | topical | 2 | 23 | 25 | **0.000** |
| laya vs rerank | contrastive | 11 | 10 | 21 | 1.000 |
| laya vs rerank | topical | 13 | 4 | 17 | 0.049 |
| laya vs lexical | contrastive | 10 | 18 | 28 | 0.185 |
| laya_cond vs laya | contrastive | 9 | 6 | 15 | 0.607 |
| embedding_cond vs embedding | all | 6 | 19 | 25 | **0.015** |
| embedding vs rerank | contrastive | 22 | 8 | 30 | 0.016 |

**What can and cannot be concluded at this n.** Laya losing to embeddings is
significant on both slices and the direction is unambiguous. Laya *beating*
rerank is **not** established on contrastive buckets — 11 vs 10 discordant, p=1.000,
which is a coin. Laya beating rerank on topical buckets (13 vs 4, p=0.049) is on
the edge and should not be leaned on. The condition-only variant beating the
full-text variant is **not** significant (p=0.607 contrastive, p=0.110 overall)
even though the raw accuracy moves 11 points; that is what n=89 buys.

**Solution.** Select hints with `code_search.embed()`, asymmetric, query prefix
on the problem and no prefix on the hint. Do not call Laya for this. Concretely:
in `bench/mechanisms/selectors.py`, `select_embedding` is the arm to keep;
`select_laya` should stay in the registry as the measured negative control it
already documents itself as, and `select_tree` should not be built on the
assumption that a per-node Laya choice over recipe text is reliable — every node
of such a tree is this experiment, and this experiment says 29.8%.

---

## Finding 2 — Laya is degenerate on these buckets: the option wordings decide, not the problem

`docs/LAYA.md`'s cross-cutting lesson says a margin gate is only meaningful
after the question has been shown not to be degenerate. Here is that check.
**Modal share** is the fraction of a bucket's probes that received the *same*
answer. The floor is 1/k; a bucket where every probe returns one member is a
constant.

| bucket | 1/k | lexical | embedding | emb_cond | rerank | laya | laya_cond |
| --- | --- | --- | --- | --- | --- | --- | --- |
| range_structure | 0.20 | 0.40 | 0.40 | 0.40 | 1.00 | 0.60 | 0.40 |
| shortest_path | 0.25 | 0.75 | 0.75 | 0.50 | 0.75 | 0.75 | 0.50 |
| input_size_budget | 0.17 | 0.83 | 0.50 | 0.50 | 0.50 | 0.50 | **1.00** |
| associative_container | 0.25 | 0.75 | 0.75 | 0.50 | 0.75 | **1.00** | 0.50 |
| zig_allocation_strategy | 0.33 | 0.33 | 0.33 | 0.67 | 0.67 | **1.00** | 0.33 |
| r3f_repeated_objects | 0.25 | 0.50 | 0.50 | 0.50 | 0.75 | **1.00** | **1.00** |
| rust_allocation_fix | 0.25 | 0.50 | 0.25 | 0.50 | **1.00** | **1.00** | 0.75 |
| motion_rule | 0.20 | 0.20 | 0.40 | 0.40 | **1.00** | **1.00** | 0.40 |
| accessibility_finding | 0.17 | 0.50 | **0.17** | 0.50 | 0.83 | 0.50 | **1.00** |
| api_convention | 0.17 | 0.50 | **0.17** | 0.33 | 0.67 | 0.83 | 0.50 |
| **all, probe-weighted** | **0.21** | 0.52 | **0.39** | 0.47 | 0.84 | **0.75** | 0.66 |

(The full 19-row table is printed by the harness as TABLE 7.)

Laya returns the same member for **75%** of a bucket's probes against a 21%
floor, and returns a *constant* on five of nineteen buckets. Embeddings sit at
0.39 and hit the floor exactly on the two six-member topical buckets they score
6/6 on. This is the same failure `docs/LAYA.md` finding 3 named: a large margin
means the option wordings separate cleanly in Laya's embedding, not that the
state selected between them.

**Why.** `docs/LAYA.md` finding 8 and `bench/mechanisms/selectors.py` both say
it outright: Laya's measured weak regime is *options that are different
passages*. A recipe is a 150–230 character passage. Fixing the passage and
varying the query separates real from nonsense by 0.497; varying both collapses
the gap to ~0. Hint selection is by construction the second case — every option
is a different passage and the state changes on every call.

**Solution.** Two parts.

- *Do not add a Laya question over passage-shaped options without running the
  degeneracy check first.* It is one table, it is in `bench/hint_collapse.py`
  (`TABLE 7`), and it takes a second on cached data.
- *If a Laya choice is wanted anyway, shrink the options to labels.* The
  condition-only variant drops modal share from 0.75 to 0.66 and adds 11 points
  of accuracy. The logical endpoint of that trend is options that are two or
  three words — which is the configuration `docs/LAYA.md` finding 4 found usable,
  and is not a configuration that can carry a recipe. **Blocker:** a two-word
  label per recipe does not exist in the corpus; `category` is the closest field
  and it is not written to be discriminative (547 distinct values, many singleton).

---

## Finding 3 — Laya's margin is not a confidence here, and embeddings' is

This is the "hints must never harm" table. Each arm's own confidence signal is
sorted descending and truncated so **every arm commits on the same fraction of
probes**; precision is compared there. An arm that abstains more gets no credit
for it. Margins are top-1 minus top-2 after sum-normalising within the bucket,
so they are comparable across buckets of different size.

**ALL PROBES (n=89)** — precision %, at each coverage:

| arm | 100% | 90% | 75% | 60% | 50% | 40% | 30% | 20% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random | 21.3 | 21.3 | 21.3 | 21.3 | 21.3 | 21.3 | 21.3 | 21.3 |
| fixed | 21.3 | 21.3 | 21.3 | 21.3 | 21.3 | 21.3 | 21.3 | 21.3 |
| lexical | 51.7 | 55.0 | 62.7 | 67.9 | 70.5 | 77.8 | 85.2 | 83.3 |
| **embedding** | **71.9** | **77.5** | **85.1** | **86.8** | 84.1 | 83.3 | 85.2 | 88.9 |
| embedding_cond | 57.3 | 61.3 | 64.2 | 66.0 | 70.5 | 72.2 | 66.7 | 55.6 |
| rerank | 22.5 | 23.8 | 19.4 | 22.6 | 25.0 | 27.8 | 29.6 | 33.3 |
| **laya** | 33.7 | 32.5 | 29.9 | 35.8 | 36.4 | 38.9 | 33.3 | 38.9 |
| laya_cond | 44.9 | 45.0 | 46.3 | 47.2 | 43.2 | 47.2 | 44.4 | 44.4 |
| *committed* | 89 | 80 | 67 | 53 | 44 | 36 | 27 | 18 |

**CONTRASTIVE (n=47)**:

| arm | 100% | 90% | 75% | 60% | 50% | 40% | 30% | 20% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lexical | 46.8 | 47.6 | 57.1 | 53.6 | 54.2 | 57.9 | 64.3 | 66.7 |
| **embedding** | **57.4** | **61.9** | **68.6** | **75.0** | **70.8** | **68.4** | **71.4** | **77.8** |
| embedding_cond | 44.7 | 50.0 | 45.7 | 46.4 | 50.0 | 52.6 | 42.9 | 33.3 |
| rerank | 27.7 | 26.2 | 28.6 | 32.1 | 29.2 | 31.6 | 35.7 | 33.3 |
| **laya** | 29.8 | 28.6 | 28.6 | 25.0 | 29.2 | 31.6 | 35.7 | 33.3 |
| laya_cond | 36.2 | 38.1 | 37.1 | 35.7 | 29.2 | 36.8 | 42.9 | 44.4 |
| *committed* | 47 | 42 | 35 | 28 | 24 | 19 | 14 | 9 |

**TOPICAL (n=42)**:

| arm | 100% | 90% | 75% | 60% | 50% | 40% | 30% | 20% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lexical | 57.1 | 63.2 | 71.9 | 80.0 | 90.5 | 94.1 | 92.3 | 87.5 |
| **embedding** | **88.1** | **94.7** | **100.0** | **100.0** | **100.0** | **100.0** | **100.0** | **100.0** |
| embedding_cond | 71.4 | 73.7 | 84.4 | 88.0 | 90.5 | 94.1 | 92.3 | 100.0 |
| rerank | 16.7 | 15.8 | 15.6 | 12.0 | 14.3 | 11.8 | 7.7 | 0.0 |
| laya | 38.1 | 36.8 | 37.5 | 44.0 | 38.1 | 41.2 | 30.8 | 37.5 |
| laya_cond | 54.8 | 55.3 | 56.2 | 56.0 | 52.4 | 52.9 | 38.5 | 37.5 |
| *committed* | 42 | 38 | 32 | 25 | 21 | 17 | 13 | 8 |

**There is no coverage point on any slice where Laya's precision reaches
embeddings'.** The smallest gap anywhere is 27.7 points, on the contrastive
slice. This is asserted in the tests.

The shape matters as much as the level. Embeddings' curve *rises* as it
abstains — 71.9% → 86.8% at 60% coverage, and **100% on the topical slice from
75% coverage down**. Laya's curve is flat: 33.7% at full coverage, 38.9% at 20%
coverage. Discarding the four fifths of probes Laya is least sure about buys
five points. **Its margin is not measuring whether it is right.** That is the
third time this has been observed on Laya (`docs/LAYA.md` findings 3, 5, 7) and
the first time in a regime where the options are hints rather than routes.

Each arm at its own native gate, for completeness:

| arm | gate | fired | coverage | precision | cov-acc |
| --- | --- | --- | --- | --- | --- |
| lexical | 0.05 | 58/89 | 65.2% | 65.5% | 42.7% |
| embedding | 0.05 | 78/89 | 87.6% | 79.5% | 69.7% |
| embedding_cond | 0.05 | 64/89 | 71.9% | 64.1% | 46.1% |
| rerank | 0.05 | 72/89 | 80.9% | 20.8% | 16.9% |
| laya | 0.15 (`LAYA_MARGIN_FLOOR`) | 38/89 | 42.7% | 36.8% | 15.7% |
| laya_cond | 0.15 | 57/89 | 64.0% | 47.4% | 30.3% |
| laya | 0.30 (`TREE_MARGIN_GATE`) | 21/89 | 23.6% | 33.3% | 7.9% |

`TREE_MARGIN_GATE` makes it worse in every column, which is the identical shape
of `docs/LAYA.md`'s finding 1 gate sweep.

**Solution.** Gate hint injection on the **cosine margin**, not on a Laya
margin. From the curve, a gate that commits on ~60% of problems runs at ~87%
precision overall and 100% on topically separable buckets; that is a defensible
"never harm" operating point. Record the gate next to the curve it came from, as
`docs/LAYA.md` finding 4 does, and re-derive it if the hint corpus changes —
this curve is fit on the same 89 probes it reports, so it is an upper bound and
is labelled as one.

---

## Finding 4 — the `/v1/rerank` endpoint contaminates scores across a batch

This is a serving bug, it was found by accident, and it is the most immediately
actionable result in this document.

The rerank arm first scored **22.5%**, at chance, which is not a plausible
number for a cross-encoder. Three checks:

1. **Instruction prefix and document order make no difference.** 20/89 plain,
   20/89 with a query instruction prefix, 19/89 with the documents reversed.
   Not a prompt-format problem.
2. **The endpoint cannot retrieve a document from its own verbatim text.**
   Query each bucket member's *exact recipe text* against its own bucket:
   batched, **15–16 / 89**. Embeddings get **89/89** on the same test. A
   cross-encoder that cannot match a document to itself is not ranking.
3. **Scoring one document per call fixes most of it.** Same identity query,
   one document per call: **68/89**.

The mechanism, on a four-document toy set and the query *"what is the capital of
France"*:

| document | score alone | score inside the batch |
| --- | --- | --- |
| "The capital of France is Paris." | **7.3e-08** (highest) | 7.3e-08 (**last**) |
| "A red panda is …" | 1.16e-08 | 1.90e-01 |
| "To reverse a linked list …" | 4.47e-08 | 8.6e-08 |
| "Sourdough bread …" | 1.63e-09 | **4.43e-01** (**first**) |

Alone, the correct document wins. Inside the batch, the document that scores
*lowest* alone wins by two orders of magnitude. Four identical batched calls in
one process return identical numbers, so it is not sampling noise; the same call
in a later session picked a different winner, so it is server state leaking
between slots in one rerank request.

**The numbers in this document are therefore not a measurement of
Qwen3-Reranker's ability.** They are a measurement of the endpoint as served
today. Both forms are reported (`rerank` = one per call, `rerank_batched` =
batched) so a fix will show up as this arm's score moving.

**Solution, in priority order.**

1. *Stop batching rerank calls until the server is fixed.* `mcp/code_search.py`
   calls `rerank(query, docs, top_k)` with up to 40 candidate documents in one
   request at `search_code`'s rerank step. Every one of those rankings is
   currently corrupted. Because `RERANK_MAX_K = 2` the reranker is skipped at
   the shipped `top_k`, so production is mostly not hitting it — but
   `scripts/eval_rerank.py`'s "rerank 3/11 vs embed 1/11" numbers were measured
   through the batched path and should be regarded as void. **Blocker:**
   `mcp/code_search.py` is owned by another workstream; this needs a one-line
   change from one batched call to a loop, at 40× the requests.
2. *Fix it properly in the server.* The signature — per-document scores that
   depend on batch composition, deterministic within a process — points at KV
   cache or sequence-slot reuse in llama.cpp's rerank path rather than at the
   model. Reproduce with the four-document table above; it is four lines.
3. *Assert it.* `bench/test_hint_collapse.py --live` asserts both the solo and
   the batched behaviour, so the day the server is fixed, that test fails and
   tells you why.

---

## Finding 5 — the topical/contrastive split is real, and it moves everything except Laya

Contrastive minus topical accuracy, from Finding 1:

| arm | contr − top |
| --- | --- |
| embedding | **−30.6pp** |
| laya_cond | −18.6pp |
| lexical | −10.3pp |
| laya | −8.3pp |
| rerank | +11.0pp |
| random | +13.4pp |

Embeddings fall 30.6 points going from topical to contrastive buckets, and
88.1% → 57.4% is a real and interpretable failure: cosine similarity is a
topical instrument, and the contrastive buckets were built so topic is held
constant and only a condition varies. `input_size_budget` — six members that
differ only by a number — is 3/6 for embeddings, and `ts_type_testing_tool` is
1/6, its worst bucket.

**This is the gap the brief predicted Laya would fill, and Laya does not fill
it.** Laya is 29.8% there, below embeddings' 57.4% and below the lexical floor's
46.8%. The gap is real; Laya is not the thing that closes it.

Where Laya is alone in being right (6 probes) it is mostly on contrastive
buckets — `range_structure::sqrt_decomposition`, `shortest_path::bellman_ford`,
`ts_inference_control::as_const`, `ts_type_testing_tool::expect_type` — which is
a faint trace of the predicted effect. It is exactly cancelled by the 6 probes
where Laya is alone in being wrong, three of them at margins above 0.25
(`zig::caller_supplied` at 0.507, `api_convention::casing` at 0.325). Six versus
six is not an effect.

**The obvious fix was tried and it does not work.** The hypothesis was that the
contrastive gap is a *data* problem: **430 of the 938** recipes carry an
explicit `trigger_condition` that states the discriminating clause in isolation
(the harness falls back to the clause before the first colon for the rest), and
embedding *that* instead of the whole recipe should put the discriminator where
cosine weighs it instead of diluting it in 200 characters of advice. It is the
same change that helped Laya by 11 points. Measured as the `embedding_cond` arm:

| | overall | contrastive | topical |
| --- | --- | --- | --- |
| embedding, whole recipe | **71.9%** | **57.4%** | **88.1%** |
| embedding, condition only | 57.3% | 44.7% | 71.4% |

It is **worse everywhere**: -14.6pp overall (paired, 6 vs 19 discordant,
p=0.015), and -12.7pp on exactly the contrastive slice it was supposed to rescue
(5 vs 11 discordant, p=0.210 - not significant, but the direction is wrong and
it is not close). Embedding and Laya want opposite things from the same text:
Laya is hurt by long passage-shaped options and helped by shortening them, while
cosine needs the whole passage because the surrounding advice is what anchors
the topic. A bare condition such as "Is the input size at most about 20?" is too
short and too generic to land anywhere useful in the space.

**Solution.** Leave the embedded document as the whole recipe. The contrastive
gap is real and none of the seven arms here closes it - the best contrastive
number in the file is embeddings' 57.4%, against 88.1% on topical buckets. Two
routes remain untried and both are RETRIEVAL routes, not Laya routes:

- *Concatenate rather than replace.* Embed `trigger_condition` followed by the
  recipe, so the discriminator is repeated and up-weighted without losing the
  topical anchor. One line, measurable on this harness, and it is the version of
  the hypothesis the result above does not rule out.
- *Two stages.* Use embeddings to narrow to a bucket, which they are excellent
  at - 100% precision on topical buckets from 75% coverage down - then a
  System-2 generation, not Laya, to pick within it. The within-bucket choice is
  where every mechanism measured here is weak, and it is one short generation
  over at most six candidates.

---

## What did NOT work — the negative results, kept on purpose

| tried | result | why |
| --- | --- | --- |
| Laya joint choice, full recipe as option text | 33.7% overall, 29.8% contrastive | Options are passages. Laya's documented weak regime. Finding 2. |
| Laya with `TREE_MARGIN_GATE` = 0.30 | 23.6% coverage at 33.3% precision | The gate makes every column worse, exactly as in `docs/LAYA.md` finding 1. |
| Laya with `LAYA_MARGIN_FLOOR` = 0.15 | 42.7% coverage at 36.8% precision | Abstaining on 57% of problems does not buy precision. The margin is not a confidence. |
| Laya instruction reworded ("Which of these applies…") | 31/89 vs 30/89 | Prompt wording is not the bottleneck. Four configurations were swept; the spread is 30–40 / 89. |
| Laya with condition-only options | 44.9% overall | Real improvement (+11pp), not significant at n=89 (p=0.110), still loses to embeddings by 27pp, still 0.66 modal share. |
| **Embedding the condition instead of the recipe** | 57.3% vs 71.9% | Worse everywhere, p=0.015. The fix that helps Laya hurts cosine. Finding 5. |
| Rerank, batched | 22.5% — and 15/89 on identity queries | Serving bug, not model ability. Finding 4. |
| Rerank with a query instruction prefix | 20/89, unchanged | Not a prompt-format problem. |
| Rerank with documents reversed | 19/89 vs 20/89 | Not a document-order problem. |
| Keying buckets on `area` or `category` | rejected before measuring | `area=problem_shape_to_data_structure` mixes alternatives with a constraint on one of them. The brief's warning, verified. |
| Un-normalised rerank margin as a confidence signal | discarded | Rerank scores here live around 1e-06 to 1e-01 with no stable scale, so a raw gap is not comparable across buckets. Sum-normalising is a choice and it is the most favourable honest one. |
| Putting `bench/mechanisms` on `sys.path` | reverted | It shadows the stdlib `selectors` module for the whole process. Loaded by path under a private name instead; asserted in the tests. |

Two methodology notes worth calling out because they produced numbers that
looked fine:

- **The first rerank arm batched its documents**, which is the obvious way to
  write it and the way `mcp/code_search.py` already does it. It produced a
  plausible-looking 22.5% that would have been reported as "the reranker cannot
  do this task". The identity-query sanity check — *can it retrieve a document
  from its own exact text* — is what caught it, costs four lines, and should be
  the first thing run against any new retrieval endpoint.
- **A single Laya call is not a reading.** Every Laya decision here is averaged
  over as many orderings as the bucket has members, through
  `selectors._laya_choice`, which also enforces that candidates never appear in
  `state`. The tests assert the ordering count per probe equals the bucket size.

---

## What is NOT claimed here

- **That Laya is a bad model.** It is answering in ~23 ms and it is being asked
  to compare content across passages, which `mcp/shomen.py` and
  `docs/LAYA.md` both document it as weak at. This is the third independent
  confirmation of a known limit, not a new discovery about the checkpoint.
- **That Laya beats the reranker.** On contrastive buckets it is 11 vs 10
  discordant, p=1.000. That is a coin. And the reranker arm is degraded by a
  serving bug, so the comparison is not about model quality in either direction.
- **That the condition-only Laya variant is worse than it looks, or better.**
  +11pp raw, p=0.110 paired. n=89 cannot separate those.
- **That 89 probes are representative.** They were written by one agent against
  one corpus, in one sitting, with ground truth known by construction. Probes
  written by someone who had not just read the hints would move these numbers,
  probably downward for every arm.
- **That the embedding coverage curve is a calibrated gate.** It is fit on the
  same 89 probes it reports. It is an upper bound and is labelled as one; a
  held-out gate needs a second probe set.
- **That the rerank numbers describe Qwen3-Reranker-0.6B.** They describe the
  endpoint as currently served. Re-run after Finding 4 is fixed.
