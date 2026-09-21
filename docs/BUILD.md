# What to build, in order, with the evidence for each

Ranked by evidence strength times effect size. Every item names what justifies
it. Items at the bottom are things to **not** build, with the reason.

---

## 0. Parallel sampling and speculative decoding CONFLICT

Measured on this box, concurrent completions against the one model instance:

    N=1   5.5s   36.6 tok/s
    N=2   8.8s   45.3 tok/s
    N=4  14.9s   53.8 tok/s    <- best throughput
    N=8  32.5s   49.2 tok/s    <- batch saturates, regresses

Four candidates cost 2.7x the wall clock of one, not 4x, because batching
recovers memory bandwidth that a single stream leaves idle.

That is the same idle bandwidth speculative decoding exploits, and the
published numbers show it collapsing as batch size grows -- 1.96x at batch 1
falling to 0.7x at batch 48, and stock EAGLE in vLLM going 1.3x at batch 2 to
0.7x at batch 48. **They cannot both win.**

Given best-of-N also produces something speculative decoding does not -- a set
of candidates to select from, and the labelled data that trains the selector --
parallel sampling is the better use of the same headroom. Speculative decoding
drops to a measurement worth doing for the SINGLE-sample path (interactive
chat, where N=1 and latency is what the user feels), not for the verified-edit
path.

## 1. Speculative decoding -- the speed lever (now scoped to N=1 only)

**Problem it solves:** a complex task taking days. Speed is the binding
constraint; cache economics are not, because the cache costs us nothing.

**Why it is first:** 2-5x, lossless by construction, and we sit in the single
best regime for it. Speculative decoding degrades as batch size grows --
measured 1.96x at batch 1 falling to 1.21x at batch 128, and stock EAGLE in
vLLM going 1.3x at batch 2 to **0.7x at batch 48**. We are permanently at
batch 1 and memory-bandwidth-bound, which is exactly where it pays most.

Agentic workloads accept drafts better than prose: 3.5-4.45x on tool-calling
benchmarks, up to 5.3x on SWE-Bench-style traces.

**Build notes:**
- llama.cpp community measurement on a comparable card: a 14B coder model with
  a **0.5B** draft gives 2.36x on coding, 1.67x on reasoning.
- **0.5B is the sweet spot**; 1.5B is questionable and 3B is not useful.
- **Lineage matters more than size.** A Gemma2-9B/Gemma2-2B pair gave *no*
  speedup because the two were not distillation-related. The draft must come
  from Bonsai's own family.
- Use **standard rejection sampling**, not Medusa-style typical acceptance.
  Lossy verification is now quantified: MATH 76.88 -> 69.84, and
  **BFCL 86.60 -> 81.40** -- tool calling is what we care about most.

**Risk:** finding a draft model of the right lineage for a ternary Bonsai. If
none exists this item dies, which is why it gets checked first.

---

## 2. `rings` — a durable work log

**Problem it solves:** the observed failure where a long session reaches ~75%
context, compaction summarises, the summary loses "I already did X and it went
green", and the model redoes the work. Confirmed in practice on frontier
models; Bonsai will be worse, not better.

**Why a tool and not better compaction:** compaction is lossy in exactly the
dimension that prevents repetition. The fix is that the record of work must
live **outside the context window** and be re-readable at fixed small cost
after any compaction.

**Design:** append-only log in sqlite, `record_step` / `read_rings`.
`run_check` results land automatically, so "these four checks were green as of
step 12" is a fact on disk, not a memory. One line in the system prompt:
read the rings before planning. MCP, so Hermes and opencode get it identically.

**Evidence status: weakest item here, and stated as such.** The published
material on context compaction is blog-grade -- claims like "compaction loses
77% of named entities" circulate with no reproducible methodology. This is
built on an observed failure, not a paper. It is cheap, so measure it: run a
long task, force compaction, count repeated work with and without.

---

## 3. Verifier-gated best-of-N

**Problem it solves:** a 27B losing to frontier models one-shot.

**The arithmetic that decides the design.** Suppose Bonsai is wrong 10% of the
time and a gate achieves 90% recall at 90% specificity -- better than most
published classifiers. Per 100 outputs: 9 true flags, **9 false flags**.
Precision 50%. At a 5% error rate, 32%. **Half of what you block is fine.**

That is why the pattern that works is *selection among N candidates*, which
has no false-positive cost, and not *gating one output*, which does.

**Sound verifiers dwarf learned ones.** With a sound external verifier:
Game of 24 5% -> 38%, Graph Coloring 16% -> 37%, Blocksworld 40% -> 87%.
Verifiers synthesised as code (Lean, z3) took a 30B from **32% -> 87%** on a
tool-use benchmark with no finetuning. Self-critique, by contrast, made GPT-4
*worse* on two of three domains, and external reward-model best-of-N **lost to
plain majority voting** on MATH500.

So: `tsc --noEmit` with `expectTypeOf`, `cargo test` plus a node harness across
the WASM boundary, `naga`/`tint` on shaders. Executable checks, never a model's
opinion, wherever one exists.

**Where it wins:** verification gains peak at *weak-to-medium* generators and
shrink as generators improve -- strong models hide errors in coherent
reasoning. A 27B sits near that peak. This is a real structural advantage.

**Where it does not:** small verifiers over large N eat themselves. A 1B
process reward model on MATH improves to best-of-16 then **drops at
best-of-64** from reward hacking. Keep N modest -- 8 to 16 -- unless the
verifier is sound, in which case N is bounded only by time.

---

## 4. A linear probe on Bonsai's own hidden states

**The most valuable finding in the research, and it does not involve Laya.**

L2-regularised logistic regression on a **single mid-layer** of the generator's
own activations reaches **0.952 AUROC** for detecting its unreliable outputs,
against semantic entropy ~0.53, DoLa 0.580, and asking the model itself 0.660.
MLP probes beat linear by **<0.01** -- the signal is a linear mean shift, so a
logistic regression is the right tool, not a network.

Optimal layer is model-specific and mid-stack (layer 14 of 32 for an 8B Llama,
18 for Qwen2.5-7B). We run Bonsai locally, so we already have the activations.
**This is cheaper than calling Laya and reads the generator's internal state
rather than guessing from its text.**

It converges with two independent results: cascade deferral gains coming
specifically from "intermediate layers of the larger model", and probes plus
recall calibration cutting agent token use **54.9-60.2% at 90% target recall**.

**The caveat is severe and must be designed around:** cross-domain transfer
collapses to **AUROC 0.477 -- chance** -- because the mean-shift directions are
near-orthogonal between domains (cosine ~0.12). It needs ~500 labelled examples
*per domain*. For us that means one probe per task class (type-level TS,
Rust/WASM, TSL), not one probe overall.

**Prerequisite:** llama.cpp must expose mid-layer activations. Check before
committing.

---

## 5. A typed decision API over Laya — Jev-shaped, calibrated, narrowly scoped

**The API is the part worth taking.** Jev's contribution is
"unstructured state in, typed probabilistic decisions out": several typed
questions batched into one call, each answered with a full probability
distribution, in one forward pass.

```
{ "state": ..., "questions": { "<name>": { "type": "choice"|"score"|"noul",
                                           "instructions": ..., "criteria": ... } } }
-> { "model": ..., "answers": { "<name>": {...} }, "usage": {...} }
```

`choice` returns `choice` + `confidence` + `probabilities`; `score` adds a
`legend`; `noul` returns a probability and notably **no confidence field**.

**Their own rationale names our exact problem:** a model right 95% of the time
that does not signal uncertainty "can't automate that task". Laya is
confidently wrong -- mis-routes at 0.99, and one evaluation found **54% of
answers with confidence >= 0.90 were wrong**. Convai's card admits raw
ECE 0.466 before temperature refitting.

**So calibration is the build, not the wrapper.** Fit temperature per question
type on our own labelled data. A useful number: logistic regression on
**50-100 labelled examples cut ECE by half** versus naive Platt scaling.

**Scope it to what measures well:**
- `choice` over a **small, well-separated, closed** option set. Routing is the
  only primitive that beat baseline in independent evaluation, by ~10 points.
- State FIRST in the payload. Laya silently drops the tail of a long state --
  a query placed after ~120 lines of preamble is discarded and every input
  then scores identically. Reproduced here.
- **Never threshold a score across different passages.** Holding a passage
  fixed and varying the query separates real from nonsense by 0.497; varying
  both collapses it to 0.000.
- Above 11 options, confidence saturates to 1.00 including on wrong answers.
  Keep option sets small.

**The honest upside, and it is real:** the best-supported use of a model in
exactly Laya's class is as a *selector over N candidates*. A ModernBERT-Large
cross-encoder -- 396M, the same backbone family -- distilled from a verifier
ensemble **retained 98.2% of the ensemble's performance at 99.97% fewer
FLOPs**, beating Pass@1 by 19.3 points at K=100. That is item 3's selector,
and it is the strongest argument for keeping Laya in the stack. But note what
it required: **distillation on the task distribution.** Convai says the same
thing about their own model -- "a fast base to specialise, not a zero-shot
decision engine." Zero-shot it is not that; fine-tuned it could be.

---

## 6. Retrieval — finish and stop

Fusion is built and the bugs are fixed. Run `eval_retrieval.py` once on the
gauntlet index, apply the cut rule, stop iterating.

Two findings worth holding it to:
- BM25 + cross-encoder is **+11% nDCG@10 over BM25 on 16 of 18 datasets** --
  the strongest single retrieval lever there is, and evidence our reranker
  should work *if used correctly*.
- But **53.3% of reranking experiments came out worse than no reranker** as K
  grew, with pointwise cross-encoders improving at small K then inverting.
  Rerank the top 10-50, never the top 500. Our `RERANK_MAX_K = 2` is on the
  right side of this; it may be too conservative, which the eval will say.

---

## Do NOT build

**Laya as a correctness gate.** The base-rate arithmetic in item 3 kills it
independent of any measurement, and our measurements agree (6/10 against a
5/10 coin flip). Already cut.

**Laya as a relevance scorer or compactor.** Compaction is inherently a
cross-passage comparison, which is the exact comparison its scores do not
survive. The one shipped implementation reports relevance ratings that
**invert as history grows**.

**A learned router.** Most-documented, most-debunked pattern in the space.
Across 400k queries and 10 routing methods, most fail to beat a simple
baseline; the top 15 routers on one benchmark differ by **0.23 points**; kNN
on frozen embeddings matches state of the art. We have one model and nothing
to route to.

**Semantic caching.** Precision **0.52** measured -- an embedding keying on
topic rather than proposition, which is the same failure we already cut Laya
for.

**GBNF grammar constraints during reasoning.** llama.cpp's implementation is
the weak one: a character-level backtracking parser in CPU sampling with no
GPU overlap, measured at **3.6-8.2x latency overhead**, one report showing
sampling time up 51x and GPU utilisation falling 70% -> 10%. And it would not
buy what we want anyway -- **92% of the format tax comes from the format
prompt alone**, only ~1.6 points from the grammar. Constrained decoding buys
parseability, not correctness: wrong-*value* errors are **96-100% of imperfect
structured outputs at every model size**.

If tool-call format errors turn out to be material, the fix is
**"reason free, then serialise"** -- which recovers 79-87% of the loss -- plus
a generous token budget. Small models are hit far harder by format constraints
than large ones (-36.2 points vs -0.6 on the same task), and ~87% of that is
explained by **token truncation**, not by the constraint itself.

---

## Sequence

1. Check a same-lineage 0.5B draft exists -> speculative decoding. (speed)
2. Check llama.cpp exposes mid-layer activations. (gates item 4)
3. Tool-use competence probe, including the partial-input baseline below.
4. Best-of-N go/no-go: pass@1 vs pass@8 with `tsc`/`cargo` on 15 tasks.
5. `rings`, measured against a forced compaction.
6. `eval_retrieval.py` once; apply the cut rule.
7. Typed decision API + calibration, scoped as above.
8. Linear probe, one per task class, ~500 labels each.

## The one experiment to run this week

**The partial-input baseline.** Run the classifier on inputs with the
proposition *removed* -- topic and context only. If it scores near its full
accuracy, it is keying on topic, and validation accuracy is structurally
incapable of revealing that. This is a canonical diagnostic: a classifier
reading only the hypothesis and never the premise scores **~67% on SNLI
against a 33% baseline**, and a claim-only model that never reads the evidence
gets **61.7% on fact verification** against 69.7% for the real system.

We have already seen the shape of this result informally. Doing it properly
takes an afternoon and it is the highest-information experiment available.

Also: stratify every eval by lexical overlap and report TPR/TNR **separately**.
Balanced accuracy hides >20-point TPR/TNR gaps on half the datasets where it
has been checked, and the collapse to chance lives specifically in the
high-overlap bin.
