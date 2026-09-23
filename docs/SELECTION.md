# Choosing a selection mechanism

This stack has five ways to pick one thing out of several: Laya, embeddings,
BM25/lexical, a cross-encoder reranker, and regular expressions. They are not
interchangeable, they fail in different directions, and the wrong one has been
reached for here more than once at real cost.

This document is the decision guide. Every number in it was measured in this
repository, on labelled data, with its sample size attached. Where something is
untested it says so and says what would test it. Nothing here is a principle
someone liked the sound of; each rule has a measurement behind it and names the
file that produced the measurement.

Sources of truth, which this document summarises and must not contradict:
`docs/LAYA.md` (both parts), `docs/FINDINGS.md` §16/§16a/§17,
`bench/mechanisms/selectors.py`, `mcp/code_search.py`,
`.venv-laya/Lib/site-packages/laya/common.py`.

---

## 1. The fact that organises everything: what Laya actually is

Read from `.venv-laya/Lib/site-packages/laya/common.py`, not inferred from
behaviour. `build_sequence` lays the input out as:

```
[CLS] <type> question: <instructions> [SEP]
[MASK] opt0 [MASK] opt1 ... [SEP]
<state> [SEP]
```

`DecisionModel.forward` gathers the hidden state at each `[MASK]` marker, runs
`scorer` (LayerNorm → Linear(d,d) → GELU → Linear(d,1)) to get **one scalar per
option**, and takes a **softmax across the markers**. That is the whole answer.

**Laya is an option-scoring cross-encoder.** Not a sentence encoder, not a
fixed-class classifier. Four consequences follow, and they decide every use of
it in this stack:

1. **Options are rendered INTO the input as text.** Laya can only judge what is
   in the input. If the deciding information is not in the sequence, no
   prompting, gating or training recovers it — see §5, anti-pattern A.
2. **Scores are jointly normalised over the option set.** Nothing else in this
   stack has that property. `code_search.embed()` scores every candidate
   independently; `code_search.rerank()` cross-encodes one (query, doc) pair at
   a time. Joint normalisation is the only thing Laya offers that the
   alternatives cannot, so it is the only reason to reach for it.
3. **The label space is open by construction.** That is why arbitrary zero-shot
   questions work at all, and equally why the checkpoint has no particular
   competence at *our* boundary. Nothing in it has seen our labels.
4. **`act_probability` is not an independent abstention signal.** `act_head`
   consumes the pooled `[CLS]` concatenated with four scalars derived from the
   answer distribution itself — top-1 probability, top-1 minus top-2,
   normalised entropy, `k/255`. It is a function of the same distribution the
   margin gate already reads, not a second opinion. Do not treat agreement
   between the margin and `act_probability` as corroboration.

Laya is a 421M ModernBERT-large answering in ~23 ms, and it is **deterministic**:
three identical calls returned identical probabilities to six decimal places
(`docs/LAYA.md`, Part I preamble). A single call is reproducible. It is still not
a reading — see §4, pattern 2.

### The mechanism table

| mechanism | what it scores | normalisation | cost | shipped here |
| --- | --- | --- | --- | --- |
| **Laya** | all options in ONE forward pass | **joint** — softmax across option markers | ~23 ms per call × permutations | `/decide`, `/route` on port 1237 |
| **embeddings** (Qwen3-Embedding-0.6B) | `cos(query, doc_i)` | **independent** per doc, absolute relevance | ~ms, GPU round trip | OFF by default (`CODE_SEARCH_SEMANTIC=0`) |
| **BM25 / lexical** | term overlap | **independent** per doc | microseconds, no GPU | the shipped retrieval path |
| **reranker** (Qwen3-Reranker-0.6B) | cross-encodes (query, doc) **jointly** | but still **independent across docs** — one doc at a time | ~1 s per query at top-5 | only at `k ≤ RERANK_MAX_K = 2` |
| **symbol table** | "is there a declaration named this here" | not a ranker at all — a yes/no tag | sqlite lookup | always on, never enters the fusion |
| **regex / rules** | whatever you write | deterministic | ~1 ms, no GPU, no service | repeatedly the thing to beat |

The reranker is the one that gets misfiled. It *is* a cross-encoder, like Laya,
but it reads one document at a time, so its outputs are no more comparable
across documents than cosines are. Only Laya normalises the candidates against
each other.

---

## 2. The decision table

Two questions predict success here, and almost nothing else does.

**(a) Is the deciding information present in the input?**
**(b) Is the judgement ABSOLUTE (how relevant is this one thing) or COMPARATIVE
(which one of these mutually exclusive things)?**

| (a) info in input? | (b) judgement | reach for | evidence |
| --- | --- | --- | --- |
| no | either | **nothing — fix the input, or use a deterministic check over data you already hold** | `docs/LAYA.md` F12: grounding without the file in the state is unlearnable, ≤ chance on 24 positives |
| yes, and it is a surface cue | either | **regex/rules first**, always | 0.841 ±0.023 vs the trained head's 0.726 ±0.045 on the same splits, `docs/LAYA.md` F11 |
| yes | absolute: "how relevant is this document" | **BM25 first, embeddings only if measured to help on YOUR queries** | recall@5 92/120 vs 77/120, McNemar p=0.0041, `mcp/code_search.py:search_fused` |
| yes | absolute, and ordering within a handful of already-retrieved docs matters | **reranker, at small k only** | top-5 identical 7/11; skipping it took search ~1100 ms → 96 ms |
| yes | comparative: exactly one of a small closed set applies | **binary Laya with an explicit `continue`, gated, as a SECOND signal next to a rule** | 8/8 on the investigate node at n=8, §4 pattern 1 — small n, see the caveat |
| yes | comparative over a whole pool of passages | **do not use Laya** | varying state AND options collapses the real-vs-nonsense gap from 0.497 to ~0 |
| yes | comparative, mutually exclusive, non-passage options | **OPEN — being measured now** | `bench/hint_collapse.py`; see §7 |

The default answer is the boring one. On the only decision measured end to end
in this repo, a six-line regex beat both zero-shot Laya and a trained head, and
it is free. Start there and make the model earn its place against it.

---

## 3. Floors before mechanisms

This comes before the patterns because without it none of the patterns can be
read. `bench/mechanisms/selectors.py` implements the bounds as selectors,
deliberately, so they run through the same interface on the same inputs.

- **`random`** — the floor. A mechanism that does not beat a coin is not a
  mechanism. Seeded on `(seed, problem)` rather than on a shared stream, so a
  rerun draws the same arms even if the problem order changes; an
  unreproducible floor cannot be compared against. It carries no confidence,
  deliberately — attaching one would make abstention analysis meaningless.
- **`fixed`** — the harder floor, and the one that actually matters. "Always
  apply the single best option, no selector at all." **A mechanism that does
  not beat `fixed` is paying for nothing.** Most of the value in a selection
  problem often sits in the best single arm; `fixed` is what tells you whether
  *selection* is doing work or the *content* is.
- **`oracle`** — the ceiling. A cheat: it reads the answer from a precomputed
  map, is marked `deployable = False` in the registry, and **raises** when the
  map is missing or does not cover the problem. It never degrades to a guess.
  A ceiling that silently becomes a guess is worse than no ceiling, because
  everything else is then compared against it (PROTOCOL rule 2).

Without both bounds a middle result is uninterpretable: a selector four points
above random tells you nothing until you know whether the headroom was five
points or forty.

### The metric depends on the job

There are two regimes and they want different headline numbers.

**Routing — you must decide every time.** Score **accuracy over all cases**,
counting an abstention as a miss. This is what caught the worst near-miss in
the project: zero-shot `route_in` scores "1.000 accuracy when it answers" and
answers 3.8% of questions (`docs/LAYA.md` F9). A gate that abstains 96% of the
time has not made a decision, it has deferred one — which is precisely the
System-2 call the gate existed to avoid. Under coverage-accuracy the same
configuration reads 3.4% at gate 0.30, which is the honest number.

**"Never harm" — emitting nothing is a fine outcome.** Score **precision at
matched coverage**: sort each arm by its own confidence signal, truncate so
every arm commits on the same fraction of cases, and compare precision there.
An arm is only allowed to look better if it is better at the same workload.
This is the discipline in `bench/hint_collapse.py`, written specifically
because accuracy-when-it-answers rewards abstaining and nearly shipped the
91.5%-abstention router.

`random` and `fixed` have no confidence signal and are reported flat in a
coverage comparison. Giving them a synthetic one flatters them.

---

## 4. The patterns that worked

### Pattern 1 — binary choice with an explicit `continue`, never a multi-way "pick a technique"

Same questions, same model, same gate (`docs/FINDINGS.md` §16a):

| framing | decided | correct |
| --- | --- | --- |
| 4-way "pick a technique" | 1/6 | 1/6 |
| binary, two ACTIVE options | 3/6 | 3/6 |
| binary cascade, `continue` an explicit option | 8/8 on the investigate node | 6/8 overall |

The change that did it: **`continue` / do-nothing is a first-class OPTION, not
the fallback.** Then "nothing is needed here" — the most common correct answer —
becomes a positive decision rather than an abstention, and the node abstains
only on genuine uncertainty between real alternatives.

The investigate node, node-level, 8/8 (margins shown):

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

**n=8, and the examples were written by the same person who chose the
phrasing.** That is a promising signal, not a validated gate. The labelled set
of ≥40 is the real test and it has not been run on this framing.

The same structure does **not** transfer for free: the fan-out node stays quiet
correctly on trivial input (`rename` → `continue` at 0.32) and **fails to fire
on either genuine fan-out case** (0.13, 0.07). The System-2 trigger and the
multiple-answers trigger need separate designs.

### Pattern 2 — permutation averaging, because one call is not a reading

Laya is deterministic but **not permutation invariant**. Every choice is
averaged over option orderings (`selectors._laya_choice`, mirroring
`fanout._choice_averaged`). For two options, forward + reversed is a *complete*
de-bias: each option occupies each position exactly once. Beyond two options,
`_orders()` adds rotations, which spreads a first-position boost evenly rather
than merely symmetrising the ends.

`LAYA_PERMUTATIONS = 2` is the shipped default. Going from 2 orderings to 6
does **not** change the three-way result (50.8% either way, n=59), so the
zero-shot routing failure is not a position-bias artefact — but that is a
statement about *that* result, not a licence to make single calls.

Two guards that go with it, both in `_laya_choice`:

- **Candidates go in `criteria` ONLY.** Repeating them inside `state` produced
  option "a" eight times at margins up to 0.919 — a pure positional artefact.
  `_check_laya_contract` raises on it, because appending "the options are ..."
  to a prompt is a natural thing to write and silently poisons every number
  downstream.
- **An order whose probabilities sum to ~0 is dropped, not averaged in.**
  Averaging a degenerate response quietly halves the margin, making an outage
  look like a healthy close call.

Also: the tail of a long `state` is silently dropped (`LAYA_STATE_CHARS = 2000`).
The part you want read goes FIRST and nothing is appended after it.

### Pattern 3 — a margin gate with ABSTENTION, and its arity dependence

Abstention is the right shape: below the gate, return "undecided", not "no".
`route_in` models this correctly as `run=None`, never `False`.

`TREE_MARGIN_GATE = 0.3` sits in an empty band: reordering-sensitive cases
averaged down to margins of 0.012–0.184 while a genuinely decided case stayed
at 0.800. `LAYA_MARGIN_FLOOR = 0.15` is the looser "do not report a pick you did
not make" floor used by `fanout.break_tie`; the tree walk uses 0.3 because its
errors compound with depth.

**The gate is arity-dependent, and forgetting that caused a wrong verdict
(`docs/FINDINGS.md` §16a).** The 0.3 gate was measured on `selectors.EXAMPLE_TREE`,
three of whose four nodes are BINARY:

| arity | uniform | what margin 0.3 demands |
| --- | --- | --- |
| 2 options | 0.500 / 0.500 | 0.65 vs 0.35 |
| 3 options | 0.333 each | roughly 0.52 vs 0.22 |

The same gate demands a far larger separation from a 3-way choice for
*identical underlying confidence*. A threshold tuned at one arity was applied
at another and the resulting abstentions were reported as "Laya cannot route".
**State the arity with every gate value you quote.**

The one gate with a proper calibration behind it is `RECOMMENDED_GATE = 0.30`
for the **binary** "does answering this need our source?" question on a prose
state, n=46 (23 investigate / 23 answer_directly, so chance is 50%):

| gate | decided | abstention | acc \| decided | cov-acc |
| --- | --- | --- | --- | --- |
| 0.00 | 46/46 | 0% | 82.6% | 82.6% |
| 0.10 | 37/46 | 19.6% | 86.5% | 69.6% |
| 0.20 | 32/46 | 30.4% | 87.5% | 60.9% |
| **0.30** | **22/46** | **52.2%** | **95.5%** | 45.7% |
| 0.40 | 12/46 | 73.9% | 100.0% | 26.1% |

That gate is for that question at that arity and nothing else. It is not a gate
for the three-way question (no value works — the sweep is monotone downward and
every fold of a five-fold hold-out picks gate 0.0) and not for the `distil`
questions (they are constants — §5).

### Pattern 4 — regex as a first-class mechanism, and as the floor to beat

`rule_baseline()` in `bench/laya_calibration.py` is six lines and needs no GPU,
no service and no labels. Scored on the **same held-out splits** as the trained
head — 89 labels, 8 repeated group-aware holdouts, everything selected by
cross-validation inside the training split:

| route_in, held out | accuracy | adversarial slice |
| --- | --- | --- |
| zero-shot Laya | 0.529 ±0.053 | 0.333 |
| trained Laya head | 0.726 ±0.045 | 0.271 |
| **regex, no model, ~1 ms** | **0.841 ±0.023** | **0.000** |
| majority class | 0.393 | — |

Training buys +0.197 over zero-shot and still loses by 0.115 to the regex. Both
numbers are true; the second is the one that decides shipping.

The adversarial slice (~3 items per split, `"hard": true` — written to defeat
surface cues) is where nothing works. The regex scores 0.000 there by
construction. The trained head scores 0.271, worse than zero-shot's 0.333, and
at n≈3 per split all three are within noise. **No configuration tested handles
the adversarial slice**, and claiming it as the head's justification would be
reading noise.

This is enforced rather than remembered: `scripts/train_laya.py` scores the
regex on every split and records the comparison inside the artefact;
`mcp/laya_service.py:beats_rule_baseline()` reads it back and `GET /heads`
reports `"beats_rule_baseline": false` for `route_in`. A retrain that overtakes
the regex flips the flag automatically.

### Pattern 5 — regex + Laya as TWO INDEPENDENT SIGNALS, escalating disagreement

This is the shipped recommendation. Route with the regex. Keep the trained head
served alongside as a second, independently-derived signal — and treat "they
disagree" as a reason to **escalate to System 2**, not as a reason to trust
either one.

The justification is that the two agree where they are both right and fail
together where they both fail: on the cue-honest slice (n=37) prose Laya scores
36/37 and the regex 36/37; on the adversarial slice (n=9) Laya scores 2/9 and
the regex 0/9, failing the same nine items in the same direction. That is the
cleanest available evidence that Laya is doing lexical cue-matching on this
question rather than reading intent. (n=9 — a direction, not a rate.)

So Laya here is worth having as a cheap confirmation, not as a judgement, and
the disagreement set is small enough that escalating it is affordable.

Where a rule and a model disagree and escalation is not available, bound the
damage with the cost asymmetry instead. For routing: an `investigate` misfire
costs one wasted search loop, an `answer_directly` misfire costs a wrong answer
with no receipts, so abstain toward `investigate`.

### Pattern 6 — training the head, for what it does and does not buy

89 labels, 8 repeated group-aware holdout splits, candidate and L2 and gate all
chosen by cross-validation *inside* each training split so the test side never
picks anything:

| route_in, held out | zero-shot | trained | majority |
| --- | --- | --- | --- |
| accuracy | 0.529 ±0.053 | **0.726 ±0.045** | 0.393 |
| abstention rate | 0.962 | **0.024** | — |
| accuracy when it answers | 1.000 (on 3.8% of items) | 0.74 (on 97.6%) | — |
| ECE | 0.123 | 0.240 | — |
| median top-two margin | 0.058 | 0.893 | — |

**What training buys is the abstention collapse, not primarily the accuracy.**
96% → 2% abstention is the real product: a head that decides 97.6% of questions
at 74% is a usable object, where a gate that declines 96% of them is not.

The winning configuration is `logistic_cls` — plain multinomial logistic
regression on the frozen pooled `[CLS]`, cross-validated at 0.808. ECE gets
*worse* (0.123 → 0.240) and that is not a bug: the zero-shot model is well
calibrated about being uncertain because it is uniformly uncertain.
`vector_scale` reaches ECE 0.052 at lower accuracy if calibrated probabilities
matter more than decisions.

**Temperature scaling cannot do this job and never could.** It is monotonic:
dividing every logit by one scalar cannot reorder them, so the argmax is
identical before and after and accuracy is mathematically unchanged.
`index/calibration.json` records `temperature 9.95, ece 0.279 → 0.032, accuracy
0.5066` — an honest measurement of a technique that cannot move accuracy. The
fitted temperature hitting 9.95, essentially the top of the search range, is the
model saying "my confidence is meaningless", not "I have been fixed". Any head
with more than one parameter can reorder; `vector_scale` (k²+k parameters on the
marker logits) reaches 0.673 where temperature reaches 0.515, i.e. chance.

Two methodology faults worth importing wholesale into any future training run,
because they produced *wrong numbers that looked fine*:

- **Pair leakage.** Where two examples share a passage and differ only in the
  claim, splitting them across train and test means a head that memorised one
  has effectively seen the other. Splits are group-aware now, keyed on `pair`.
- **fp16 batch non-determinism.** Padding a batch changes the fp16 reduction
  order, perturbing logits and hidden states. Harmless for a zero-shot argmax;
  not harmless for a linear head fitted on one version and served the other.
  Features are extracted at **batch size 1** in both trainer and server.

---

## 5. The anti-patterns, each with the measurement that killed it

### A. Asking Laya to judge something whose deciding information is not in the input

The `distil` grounding question — "is this finding grounded in the files it
cites?" — with the filenames in the state and the file contents nowhere. n=29,
12 of them fabricated and written to be *code-shaped* rather than vague:

| | value |
| --- | --- |
| chose `from_the_files` | **29/29 = 100%**, including all 12 fabrications |
| accuracy | 17/29 = 58.6% — exactly the base rate |
| mean margin | 0.867 (range 0.53–0.97) |
| separation, grounded vs fabricated | **+0.029** |
| AUC | **0.667** |

A state-sensitivity probe shows the model *is* reading the input — real
grounded 0.954, real fabricated 0.949, empty string 0.693, unrelated prose
0.391, random characters 0.386. The question it answers is **"is this text about
code and files?"**, for which it works well, not "does this text describe *these*
files?". Every finding a real investigation produces sits on the same side of the
first axis.

**The task is undecidable, not the model weak.** `"MAX_TRACES is 256"` and
`"MAX_TRACES is 1024"` are the same sentence to any classifier that cannot see
`shomen.py`. Training confirms it: grounded vs *generic advice* reaches
0.814 against a 0.667 majority (surface form is detectable); grounded vs
*fabricated specifics* reaches 0.645 against the same 0.667 majority — at or
below chance. Any accuracy above chance there would have been the model
exploiting a labelling tic.

Putting the file in the state makes it decidable but does not fix it on its own
— see anti-pattern H. The shipped answer is the deterministic one: check the
identifiers, line numbers and symbols in the finding against the text of the
retrieved chunks. That is a set intersection over data the investigation already
holds, costs nothing, and cannot hallucinate.

**Cost of getting this wrong:** §16's recommendation was to keep the grounding
question and treat the deterministic citation check as secondary. That is
backwards, and shipping it would have put a fabrication detector in the callosum
that passes 100% of fabrications at a mean margin of 0.867.

### B. Varying BOTH the state and the options

Holding the passage fixed and varying the query separates a real query from a
nonsense one by **0.497**. Varying both collapses the gap to **~0**.

This is why `select_laya` — ranking a whole recipe pool as one closed-set
choice — is Laya's weakest regime and is in the harness only because the
question "is the tree worth its complexity" needs that arm answered. It is also
why every question in `mcp/shomen.py` holds the state constant (one request,
or one finding) and varies only the options, and why none of them asks Laya to
rank passages against each other.

The fatal version, which nothing here does, is scoring each passage in a
*separate* call and thresholding across them. Laya's scores are only comparable
*within* one forward pass. Never threshold a Laya score across calls; only
margins within a call, and only against a gate.

### C. A question whose answer is constant

Asked as a dedicated binary question — "Is this request specific enough to act
on?" — Laya returned `underspecified` on **59/59 items at mean margin 0.652**
(0.707 structured, 0.695 hybrid). Zero information at high confidence.

In the three-way form, `clarify` is chosen for only 2/13 clarify items (15.4%).
The deterministic guard — no concrete symbol, no code block, no file path, under
~8 words, a referring pronoun with no referent — gets **12/13**.

**Run the degeneracy check before trusting any new Laya question.** What
fraction of items got the same answer? `python -X utf8 bench/laya_calibration.py
--binary` prints it and it takes one line. A margin gate is only meaningful
after you have shown, on labels, that the question is not degenerate.

### D. Treating a low margin as "no" rather than as "undecided"

Recorded in `docs/FINDINGS.md` §16 as an error made in this repo: a bare `noul`
was asked once and 0.162 was read as "no". It was not a no — it was the
undecided band. `selectors.py` already records that reordering-sensitive cases
land at 0.012–0.184 while a decided case sits at 0.800.

An abstention read as a decision silently disables the feature it is meant to
gate. Standing rule: **an abstention is not a decision.**

### E. Trusting the margin as confidence

On the cue-honest slice (n=37) the margin genuinely separates: mean 0.338 when
right, **0.017** when wrong. On the adversarial slice (n=9) it **inverts**:
mean 0.185 when right, **0.208 when wrong** — wrong answers carry larger
margins. At gate 0.20 the adversarial slice is 5/9 decided at 20% accuracy:
confidently, unabstainingly wrong.

Three separate questions (grounding, verdict, clarify) return large stable
margins while carrying literally zero information. The margin reflects how
cleanly the *option wordings* separate in the embedding, plus how strongly the
state's *surface features* match one of them. It reflects correctness only when
those two happen to coincide.

**Margin is not confidence.** Neither is `act_probability` — §1, consequence 4.

### F. Structured state instead of prose, for routing

The hypothesis under test was: *Laya abstains because it is handed raw prose,
and becomes decisive when handed structured signals.* Three state forms, the
same 59 items, paired:

| | prose | structured | hybrid |
| --- | --- | --- | --- |
| 3-way argmax, 2 orders | **50.8%** | 42.4% | 47.5% |
| 3-way argmax, 6 orders | 50.8% | 39.0% | 57.6% |
| abstention at gate 0.3 | 91.5% | **100.0%** | 88.1% |
| binary "needs codebase?" | **82.6%** | 50.0% (chance) | 73.9% |

**The hypothesis is refuted, and structured was WORSE.** McNemar exact, paired,
at gate 0.0: prose vs structured p=0.21–0.27, prose vs hybrid p=0.21–0.81 —
nothing significant at n=59 — but the direction is consistently against, and the
binary result is unambiguous: structured state turns the working classifier into
a constant (`needs_the_codebase` on 59/59).

Laya was trained to judge natural-language state. A `key: value` block is out of
distribution: it flattens toward a single answer and its probability mass stops
moving with the input. The signals are not worthless — the regex floor built
from the *same* signals scores 79.7% — but they are worth more to `if`
statements than to a transformer. **Branch on signals in Python; pass prose to
Laya.**

### G. Bare option labels with empty criteria

The `judge` construct — Laya as a truth judge, options with no criteria text —
scored **6/10 against a 5/10 coin flip** (`scripts/eval_judge.py`, unambiguous
yes/no engineering questions). The errors were systematic rather than noisy:
three of four misses were false positives on the NEGATIVE cases, all landing at
0.67–0.69.

```
printf(ptr) then free(ptr)   "uses memory after free"   want F, got 0.69
f(): number { return 42 }    "return type is wrong"     want F, got 0.68
borrow-legal Rust            "violates borrow rules"    want F, got 0.67
```

It is scoring what the passage is ABOUT, not whether the proposition holds — the
same failure as anti-pattern A. `judge` is implemented and reachable over the
Laya service but **deliberately not in `TOOLS`**. A judge a model trusts and
that is wrong turns an open question into a confident wrong answer, which is
worse than having no judge.

The lesson for construction: options carry their discriminating criteria as
text, because the criteria text is literally what gets scored. `render_options`
in `laya/common.py` emits `"<key>"` alone when the criterion is `None` or `""`,
and `"<key>: <criterion>"` otherwise. An empty criterion means the model is
separating your label *names*, not your *concepts*.

### H. Related: an excerpt that swamps the claim

When the file *was* put in the state, the first trained run scored 0.544 — no
better than zero-shot. Measured on standardised pooled `[CLS]` vectors:

| cosine between | value |
| --- | --- |
| the two members of a pair (same excerpt, opposite label) | **0.727** |
| two unrelated examples | −0.015 |

The ~1100-character excerpt dominates the representation; the ~20-word finding
that determines the label barely moves it. Subtracting a forward pass over the
excerpt alone (`delta = cls(finding + excerpt) − cls(excerpt alone)`) cancels the
shared component and recovers real signal — 0.625 ±0.099 against a 0.500
majority on 60 labels — and that is still **not shippable**: about one standard
deviation above chance, with the winning configuration changing from seed to
seed, which is a sign selection is fitting noise. A grounding gate that is wrong
a third of the time is worse than no gate, because it launders fabrications as
verified.

(Raw `[CLS]` cosine is 0.9998 within a pair *and* 0.9993 between unrelated
examples, because the raw space is strongly anisotropic. The raw number looks
alarming and means nothing. Centre before computing any cosine diagnostic.)

---

## 6. Retrieval: embeddings, lexical, and the reranker

### BM25 beat embeddings, and embeddings are off by default

Measured on the gauntlet index — 60k chunks across four private repos, 120 gold
queries (`mcp/code_search.py:search_fused`):

| arm | recall@5 | vs keyword |
| --- | --- | --- |
| lexical (BM25) | **92/120** | — |
| semantic (embeddings) | 77/120 | McNemar **p=0.0041** |
| fused | 87/120 | 3 wins to 8, **p=0.2266** — indistinguishable |

The pre-registered rule was "cut unless fusion beats BM25 significantly", so it
is cut: `CODE_SEARCH_SEMANTIC=0`. Warm query latency fell from 1284 ms to 729 ms
as a side effect. Being off *by configuration* is reported in the diagnostics —
an agent told "no match" deserves to know the semantic retriever was never
consulted.

**The caveat that keeps this provisional, stated plainly: the gold queries
derive from symbol names, which flatters lexical matching.** A conceptual query
set — the kind of question where the asker does not know the identifier, which
is precisely where embeddings should win — **has never been run.** This is an
open question, not a settled loss for embeddings. Set `CODE_SEARCH_SEMANTIC=1`
and re-measure on a conceptual set before concluding anything about embeddings
in general.

A second reason not to over-read the cut: on this corpus, cosine similarity
separates genuine from nonsense queries by only **0.03**, so a cosine score is
a poor confidence signal here regardless of its recall. The shipped fusion uses
*agreement between retrievers with different failure modes* as the confidence
instead — embeddings miss on vocabulary mismatch, lexical misses when the query
words are everywhere, symbol lookup misses anything that is not a declaration,
so a file all three surface is supported by three kinds of evidence.

The symbol table is **not a ranker**. It answers a yes/no question and is
collected as a tag; letting its sqlite row order act as a rank made both the RRF
contribution and the confidence tier depend on insertion order.

### The reranker: a latency decision, not an accuracy one

> **Cross-reference, added by audit — this section and `docs/FINDINGS.md` #20
> disagree and neither is deleted.** #20 (2026-09-22) reproduced `/v1/rerank`
> scoring a document differently depending on what else is in the same request,
> and `search_code` batches up to 40 candidates. On that reading every number
> below is void, *including* the "permutes within the top 5 without changing
> which files reach the caller" claim the `RERANK_MAX_K = 2` latency decision
> rests on. `docs/PLAN.md` "Cut criteria" and `docs/ROADMAP.md` §1.3 carry the
> same correction; `docs/HINTS.md` finding 4 is the measurement. The latency
> figures are unaffected — a corrupt ranking still costs ~1 s.

`scripts/eval_rerank.py`, 11 queries with a known correct file:

| | embed | rerank |
| --- | --- | --- |
| correct at #1 | 1/11 | 3/11 |
| correct in top-5 | 7/11 | 7/11 |
| mean rank | 5.36 | 5.36 |

**The 1/11 vs 3/11 is Fisher exact p ≈ 0.6. It is NOT a result and must not be
cited as one.** Eleven paired queries cannot detect anything short of an
enormous effect (PROTOCOL rule 4), and the corpus was three.js, which is in
every training set.

What the numbers *do* support: the reranker permutes within the top 5 and does
not change which files reach the caller at the shipped cutoff. Since all five
snippets land in context and get read, paying ~1 s to reorder them buys nothing
measurable. **Skipping it takes `search_code` from ~1100 ms to 96 ms.** Hence
`RERANK_MAX_K = 2` against `DEFAULT_TOP_K = 5`: reranking runs only when the
caller asks for so few results that order determines content.

That is a latency decision. Restating it as "the reranker does not help" is the
mistake PROTOCOL rules 4 and 9 exist to prevent. The external literature points
the other way — BM25 + cross-encoder is reported at +11% nDCG@10 over BM25 on 16
of 18 datasets, the strongest single retrieval lever there is — while also
reporting that 53.3% of reranking experiments came out *worse* than no reranker
as K grew, with pointwise cross-encoders improving at small K then inverting.
Rerank the top 10–50, never the top 500. `RERANK_MAX_K = 2` is on the right side
of that and may be too conservative; an eval at an honest n would say.

### Two silent failures that will cost you a day each

**1. The Qwen3-Reranker-4B GGUF is numerically broken through llama.cpp's rank
path.** mradermacher's 4B GGUF returns inverted near-zero scores through
`--reranking` / `--pooling rank`. Measured on "how do I parse a tool call":

```
"the weather in paris is mild"      1.3e-19   ranked FIRST
"def parse_tool_call(answer, ...)"  8.8e-24   ranked LAST
```

The 0.6B scores the same case at 0.9992. **The failure is silent**:
`search_code` falls back to embedding order and returns a plausible but wrong
function, with no error anywhere. **Use the 0.6B.** (`README.md`'s model table
lists `Qwen3-Reranker-0.6B` and carries its own note that the 4B is NOT used —
an earlier version of this line said otherwise and pointed at a report that was
never written into this file.)

Related: a cross-encoder reads query and document *together*, so N large chunks
overflow its window, and when that happens it returns 0.0000 for every document
and silently destroys a perfectly good embedding ranking. Hence 16k context,
`RERANK_DOC_CHARS = 1200`, `RERANK_QUERY_CHARS = 2000`.

Also: correctly-ordered rerank scores here come back around **1e-13**. An
earlier guard rejected any result where every score was below 1e-6, on the
assumption that scores are probabilities. They are not, and that guard was
discarding good rankings. The real degenerate case is **every score identical**,
which is what is detected now. Rerank scores are ORDINAL only; never threshold
them, and never compare them between queries.

**2. Qwen3-Embedding is ASYMMETRIC.** Queries must carry the instruction prefix
(`QUERY_INSTRUCT`), documents must not. Embedding a bare query puts it in a
different region of the space than the corpus and retrieval silently returns
near-random results — observed directly: a question about tool-call parsing
matched a pydantic response model at score 0.002. `embed(texts, is_query=True)`
applies the prefix; getting the flag wrong does not error.

`sim` (cosine against the query embedding) is calibrated and is the only
retrieval score a threshold may be applied to. `score` (the cross-encoder's) is
for ORDER only.

---

## 7. What is untested, and what would test it

Stated explicitly so nobody reads silence as evidence.

| open question | status | what would settle it |
| --- | --- | --- |
| **Laya on a COMPARATIVE mutually-exclusive pick** — a bucket of alternatives of which exactly one applies, options that are not passages | **ANSWERED 2026-09-22, and the answer is no.** `bench/hint_collapse.py` ran: 19 buckets, 89 members, 89 probes, chance 21.3%. Laya 33.7% overall / 29.8% contrastive against embeddings' 71.9% / 57.4%, paired McNemar p=0.011 on the contrastive slice, and it also loses to a token-overlap floor at 51.7%. `docs/HINTS.md` is the write-up; `docs/FINDINGS.md` §21 is the verdict. This row previously read "being measured right now — do not assume an answer". | closed by that run |
| Embeddings on conceptual queries | **never run.** The 92/120 vs 77/120 result used symbol-derived gold queries, which flatter lexical matching | a hand-written conceptual query set with `CODE_SEARCH_SEMANTIC=1`, paired, McNemar |
| The reranker's accuracy effect | **untested at usable n, and the one run at usable n is void.** 11 queries at p≈0.6 on a contaminated corpus; separately n=356 paired rows in `bench/retrieval_results.jsonl`, which ran on every row but through the batched path `docs/FINDINGS.md` #20 shows is corrupt | fix or disable the batching first (`docs/ROADMAP.md` §1.3 step 1), then the same gauntlet-style harness that produced the 120-query BM25 result, on conceptual queries and an uncontaminated index |
| The binary-with-`continue` cascade | **n=8, self-authored.** A promising signal, not a validated gate | the ≥40-item labelled set, on this framing, with the arity-correct gate |
| The adversarial slice | **nothing handles it.** regex 0.000, zero-shot 0.333, trained 0.271, at n≈3 per split | a corpus lookup: the adversarial `investigate` items ("the retry helper", "the investigation loop") are resolvable against the 7,742-chunk index. A one-shot search for the question's noun phrases with "did anything match?" as a boolean would separate them without a model |
| Full fine-tune of Laya | **untried.** `DecisionModel.forward` takes `detach_encoder` and the package ships the training utilities, so it is architecturally supported. 89 labels would memorise | several hundred labels, then the full fine-tune |
| Deterministic identifier-overlap grounding check | **designed, measured as the right answer twice, not wired.** Blocked on `distil` receiving the trace handle so it can see retrieved chunk text | a one-argument change in `mcp/shomen.py` |

---

## 8. Quick reference

**Before adding any new Laya question:**

1. Is the deciding information in the input? If not, stop. Nothing recovers it.
2. Write the rule-based version first and score it. It is the floor that
   matters.
3. Run the degeneracy check — what fraction of items got the same answer?
   `bench/laya_calibration.py --binary`.
4. Make it binary with `continue` as an explicit option, not multi-way.
5. Average over orderings. Candidates in `criteria` only, never in `state`.
6. State the arity next to any gate value you quote.
7. Score `random`, `fixed` and the rule baseline on the *same splits*. If it
   does not beat `fixed`, it is paying for nothing.
8. Pick the metric for the job: coverage-accuracy for routing, precision at
   matched coverage for "never harm".

**Where the numbers live:**

| | |
| --- | --- |
| `docs/LAYA.md` Part I | zero-shot calibration, 59 routing + 29 distil labels, findings 1–7 |
| `docs/LAYA.md` Part II | training, 89 + 60 labels, findings 8–13, the runbook |
| `docs/FINDINGS.md` §16, §16a, §17 | the reversals, the arity error, the standing rules |
| `bench/mechanisms/selectors.py` | floors, ceiling, `_laya_choice`, the Laya contract guard |
| `bench/laya_calibration.py` | the regex baseline, the degeneracy check, `RECOMMENDED_GATE` |
| `bench/test_laya_calibration.py` | 44 assertions, no GPU — each claim in LAYA.md Part I as a test |
| `bench/test_laya_head.py` | 21 assertions against `bench/laya_baseline.json` regression floors; 33 with `--serve`, which adds the HTTP checks on an alternate port |
| `bench/hint_collapse.py` | the comparative-pick experiment, precision at matched coverage |
| `mcp/code_search.py` | `RERANK_MAX_K`, `CODE_SEARCH_SEMANTIC`, `QUERY_INSTRUCT` |
| `config.yaml` | why the reranker is the 0.6B and not the 4B |
| `mcp/laya_head.py`, `scripts/train_laya.py` | task prompts, feature specs, the trained-head path |
