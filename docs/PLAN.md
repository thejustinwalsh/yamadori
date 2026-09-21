# Plan: make Bonsai competitive with frontier models in one narrow domain

## The goal, stated so it can fail

Bonsai (27B, ternary, local, free) will not beat frontier models generally and
we are not going to pretend otherwise. The bet is narrower and winnable:

> On SWE tasks inside Justin's actual domains -- front-end UI, WASM modules,
> Rust/native-to-WASM across a C ABI, deep TypeScript type-level work, three.js
> TSL and TypeGPU -- Bonsai plus this tool stack beats a frontier model with
> only generic tools, on the same repos, at zero marginal cost.

The frontier model's advantages are raw reasoning and breadth of memorised
code. Neither helps on a private monorepo it has never seen, with local
conventions it cannot infer. That is the seam.

Three things exploit that seam, in descending order of confidence:

1. **Retrieval it doesn't have.** An index of *your* repos. Frontier models
   get grep over a working tree; that is genuinely good (see "what we're up
   against") but it is not the same as symbol tables plus embeddings.
2. **Verification it won't do.** `verify` turns one-shot generation into
   iterate-to-green. Fabel's read is that this dwarfs retrieval in end-task
   impact, and it is the cheapest thing on this list.
3. **Knowledge baked in.** Your `AGENTS.md` conventions, TSL/TypeGPU
   specifics, and the shader-export trick, as retrievable context rather than
   as prompt bloat.

## What we're up against (research, hype removed)

- Claude Code **moved off vector RAG to agentic grep**: no index to sync, no
  chunking to tune, always current. This is the strongest argument against our
  entire index, and it is not a weak one.
- Cursor runs **both** and reports **+12.5%** from combining exact + semantic
  over either alone. That is the shape we should copy.
- Off-the-shelf **cross-encoders degrade NDCG 0.3-3.1% on technical corpora**
  while adding 560-2100ms -- they are trained on web-search distributions.
  Matches our own measurement exactly.
- Long context does **not** remove the need for retrieval: models "often fail
  to locate relevant code reliably" even when it is in the window, and
  effective recall degrades well before the advertised limit.

Conclusion: an index is defensible only as a *complement* to exact search, and
only if it is measured against exact search rather than against nothing.

## What we already measured (and what it cost us)

| finding | status |
|---|---|
| Reranking: 3/11 vs 1/11 at rank 1 | **p≈0.6, not a result.** Kept only as a latency decision |
| Reranking at top-5: 7/11 vs 7/11 | identical; rerank skipped above k=2, 1100ms → 96ms |
| `judge` (Laya as truth-judge): 6/10 vs 5/10 coin flip | **cut from the tool surface** |
| Laya as relevance scorer | **failed**: nonsense query scored higher than every real one |
| Cosine separation, good vs junk queries | 0.476 vs 0.445 — too narrow to threshold |
| `grep` returning "0 files" with no context | caused a 14-call retry loop; fixed |
| three.js as eval corpus | **contaminated**, in every training set. All numbers above are suspect |

The through-line: **our precision is poor and our recall is fine.** That is the
single most important fact and it determines the design below.

## The design change that follows from it

Stop returning one confident answer. Return a **structured, weighted set**:

```
EXACT        supported by all retrievers, top ranks
CLOSE        supported by two, or rank 1 in one
ALTERNATIVES surfaced by one retriever only
```

Confidence comes from **agreement between independent retrievers**, not from a
score. Embeddings, BM25, and the symbol table fail in different ways, so a file
all three surface has three kinds of evidence behind it. Standard reciprocal
rank fusion; no GPU, no calibration, no threshold fitted to noise.

This is the opposite of overfitting on exactness, and it is justified by our
own numbers: the right answer is usually *in* the set and usually *not* at
rank 1. Handing the model a ranked set with honest confidence plays to what it
is good at (disambiguation) instead of what our scores are bad at (precision).

Implemented in `mcp/fusion.py` and `search_fused()`. Not yet evaluated.

## Where Laya belongs

Laya is measurably bad at judging whether a proposition is true -- it scores
what a passage is *about*. Two failures, same root cause. So:

- **Not** a relevance scorer. **Not** a correctness judge. Both cut.
- **Yes** as a typed-structure engine (`schema_fill.py`): JSON Schema → typed
  questions → conforming object. This is its actual strength and is untested.
- **Yes**, possibly, as a cheap router over a *closed* option set, where it
  previously scored 85%. Decisions with named options are its shape.

"Laya-style structured responses" is a good idea that does not require Laya to
produce them. We adopt the format; we use Laya only where it measures well.

## Evaluation ladder (cheap and exact first)

Fabel's ordering, because the expensive noisy eval should run last and only on
arms that survive:

1. **Retrieval-only** (`eval_retrieval.py`, minutes, no model): fused vs
   semantic vs BM25 vs symbol, on the uncontaminated gauntlet index.
   Scored by **McNemar on discordant pairs** -- "won 12, lost 4, tied 34",
   not "62% vs 55%".
2. **Tool-use competence**: can Bonsai use tools at all? Call-format error
   rate, calls per task, fraction of tool results never referenced again. If
   this is bad, everything downstream measures agentic competence, not
   retrieval.
3. **End-to-end gauntlet**, paired, in **Hermes** -- the real harness with
   skills -- not a toy loop. Arms:
   - Bonsai, no tools (contamination filter, not a fair control)
   - Bonsai, **grep + read only** ← the arm that actually threatens us
   - Bonsai, full stack
   - Bonsai, **oracle context** (hand-picked files) ← the retrieval ceiling
   - Frontier model, generic tools ← the benchmark to beat

   Without the oracle arm we cannot distinguish "retrieval is the bottleneck"
   from "the model is the bottleneck", and would tune retrieval for a model
   that cannot use it.

**Grading is mechanical, never an LLM judge:**
- type-level TS → hidden `expectTypeOf` / `@ts-expect-error` under `tsc`
- Rust→WASM → `cargo test` plus a node harness asserting across the boundary
- convention tasks → hidden test catching the missed registry/barrel entry
- location tasks → set overlap against gold `file:symbol` sets
- plus trace metrics (did it ever open the gold file?) and token/wall cost

**Pre-registration.** Task list, metric, arms and cut rule written down before
anything runs. This project has a track record of overclaiming from n=1 and the
antidote is not more n, it is committing to the decision rule in advance.

SUPERSEDED: the ~50 figure below was the statistical ideal and was cut to
15-20 later in the same review, on the grounds that each hidden-test task
costs about an hour to author and only a large effect would change what gets
reached for anyway. The reasoning is kept because the tradeoff is real.

Target ~50 paired tasks for a 20-point effect. Under 30 detects only effects so
large we would not need statistics.

## Self-tuning, bounded

The obvious version is a trap. "Model called another tool right after" reads as
rejection but is exactly what **success** looks like (search → read_file on the
hit). And tuning thresholds to minimise rejection is Goodhart: the easiest way
to cut rejections is to return nothing, so the model falls back to grep, and
retrieval dies silently.

The sound version:

- **Log for counterfactual replay**, not for online updates: every query with
  its full top-50 candidates and raw scores, plus which files were later read
  or edited, plus whether the session ended with `verify` green. Files edited
  in a green session are weak positives -- a far better label than "called
  another tool next".
- **Tune offline against a fixed gold set.** Deterministic, model-free, runs in
  seconds. Grid-search cutoffs, candidate pool, chunk size against Recall@5 and
  MRR, per repo (a 0.7 in three.js is not a 0.7 in a Rust crate).
- **Emit a report, not a mutation.** A human flips the switch. If it is ever
  closed-loop, gate it: apply only if gold-set Recall@5 does not drop.

## Domain knowledge to build in

- **Conventions from `AGENTS.md`**, especially glyph's validation rule
  (classify by who authors the value; never justify a runtime guard with a test
  that forges an internal value). Non-local by construction -- unlearnable from
  any single file, which makes it both valuable and the right thing to test.
- **TSL / TypeGPU.** Models struggle here. Justin's own workaround is to export
  the generated shader and compare. That belongs in the stack as a tool
  (`compile_shader` → raw WGSL/GLSL), not as an instruction. **Highest-value
  untested idea on this list**, because it targets a known, specific weakness
  in the exact domain we are trying to win.

## The lever this plan originally missed: verifier-gated best-of-N

A local model is not mainly cheap per token. It is free **per sample**. A
frontier model at API prices gets used once, maybe twice. Bonsai can be sampled
8 or 16 times per task at zero marginal cost.

On its own that is worthless, because you cannot pick the winning sample. But
three of these domains have *mechanical* verifiers:

  type-level TS     `tsc` with `expectTypeOf` / `@ts-expect-error`
  Rust -> WASM      `cargo test` plus a node harness across the boundary
  shaders           `naga` / `tint` validation, or a diff of emitted WGSL

With a verifier, best-of-N becomes "sample until green, return what passed".
That changes the contest from *27B vs frontier, one-shot each* -- hopeless --
to *27B with sixteen tries and a compiler vs frontier one-shot*, which is
winnable on the verifiable subset. It is not a separate lever from `verify`;
it is the version of `verify` that actually exploits being local.

This reorders everything. The original plan spent most of its effort on
retrieval, which moves the location step from eight calls to three but does not
make type-level code compile. The gauntlet grades generation with a retrieval
prerequisite, not retrieval.

## Where this can actually win, stated narrowly

**Winnable**
- Location and non-local-convention tasks in private repos. Real but modest:
  fewer calls, not more correctness, since frontier-with-grep is good here too.
- Narrow, mechanically-verified edit tasks with best-of-N: "make this
  `expectTypeOf` pass", "make this cross-boundary test green", "port this
  shader without changing emitted WGSL". **The only class where the win can be
  large.**

**Hopeless regardless of tooling** -- do not chase these
- Type-level TS *design* from scratch. Huge search space and `tsc`'s
  type-level errors are too unhelpful to verify against.
- Rust/WASM *FFI design* -- ownership and layout across the boundary.
- Front-end UI: no mechanical verifier short of screenshot diffing, and the
  frontier model has memorised an enormous UI corpus.
- Anything needing more than ~20k tokens of live context. Ternary degrades
  faster in long context than the competition does.
- Novel TSL/TypeGPU API usage: knowledge-bound, and a 27B's is thin.

Before authoring tasks: **log a week of real work and classify it.** If the
winnable classes are under a third of it, narrow the goal again or stop.

## Sequence (revised after review)

1. **Tool-use competence probe.** Cheapest gate, highest risk, and it was
   buried at step 3. If call-format error rate is material, GBNF
   grammar-constrained decoding in llama.cpp removes it at zero cost. Folded
   in: the three-shape experiment -- top-1 vs flat top-5 vs tiered -- on trace
   metrics (did it open the gold file, after how many calls). That settles
   whether tiers help a 27B or just distract it, which is currently an
   assumption in the design.
2. **Best-of-N go/no-go.** pass@1 vs pass@8-with-verifier, 15 tasks in the
   verifiable classes. If Bonsai with eight verified attempts cannot get its
   own type errors green, the bet is off and we know in a day.
3. **`compile_shader` validation without building it.** 10 TSL/TypeGPU tasks,
   run twice, once with the exported shader pasted in by hand. If that does
   not move the pass rate, the tool will not either.
4. Fix fusion (done: dedupe, symbol-as-tag, stable tier semantics), run
   `eval_retrieval.py` **once**, apply the cut rule, stop iterating.
   DONE. It fired: semantic 77/120 against keyword 92/120, McNemar p=0.0041;
   fused 87/120 against keyword, p=0.2266, indistinguishable. Embeddings are
   off by default and warm latency fell from 1284ms to 729ms.
5. Hermes + pre-registered gauntlet, **15-20 tasks** in the classes that
   survived 2 and 3 -- not 50. Each hidden-test task is ~an hour to author, and
   only large effects change what gets reached for anyway.
6. Everything else only if step 5 shows a large effect: telemetry,
   counterfactual replay, threshold tuning, `compact`.

## Cut now, not later

Premature until there is traffic or an uncontaminated baseline to tune against:
self-tuning, counterfactual replay, threshold and chunk-size grid search, and
`compact`. All designed, none on the critical path.

## Arms, precisely defined

The frontier arm must be specified or the comparison is meaningless: **does it
have a shell?** If it can run `tsc` and `cargo test`, it iterates too and
best-of-N stops being a differentiator. Pre-register this.

Add a **frontier model with the same stack** arm. If the stack helps the
frontier model more than it helps Bonsai -- which is likely -- that is a
finding: the product is the stack, not the local model.

The oracle arm is a **ceiling, not a target**. Hand-picked files leak task
structure; do not tune toward it.

## On "knowledge baked in"

The plan said "retrievable context rather than prompt bloat". That is RAG, not
baking, and the phrasing was implying something that is not happening. Actual
baking is a LoRA over the repos, `AGENTS.md` and TSL/TypeGPU docs -- and
whether that is even possible on ternary-quantised weights needs checking,
since they do not take standard LoRA without the master weights. If it is
feasible it is the only lever that improves the model rather than its
surroundings.

## Cut criteria, written before the data

- `search_code` is cut unless **fused** retrieval beats BM25 by a significant
  margin on discordant pairs. Losing to grep means it is costing a GPU and a
  stale index to do what ripgrep does free.
- The reranker is cut entirely unless it beats embeddings on the
  uncontaminated index.
- `compact` is cut unless it measurably saves tokens without losing task
  accuracy.
- Any tool not measurably better than its generic equivalent gets removed. Tool
  count is not the constraint; unmeasured tools are.
