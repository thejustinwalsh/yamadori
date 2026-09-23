# What we are building, technically

## One sentence

**A verifier-driven sampling engine with a learned selector, wrapped around a
local model.** Not a chatbot, not a RAG app. The model is a component; the
engine is the product.

## Are we training a model?

**We are not training the 27B.** Bonsai 2 is a post-training ternary
quantisation of someone else's weights. We do not have the master weights, and
ternary PTQ weights do not take standard LoRA, so finetuning the generator is
off the table. Anyone saying "knowledge baked in" about the 27B means RAG.

**We are training three small things**, and they are the parts that make this
more than a wrapper:

| what | size | trained on | job |
|---|---|---|---|
| selector | ~400M (Laya/ModernBERT class) | (candidate, passed/failed) pairs the loop harvests | pick the winner among N samples |
| probe | a logistic regression | ~500 labelled examples per task class | read Bonsai's own mid-layer activations to predict unreliability |
| calibration | one scalar per question type | 50-100 labelled examples | make the selector's confidence mean something |

All three are cheap, all three train on data the system produces while it runs,
and none of them requires touching the generator.

## The core idea

A frontier model costs money per sample, so it is used once. **Bonsai is free
per sample.** That is the only structural advantage a local model has, and
everything here is built to exploit it.

On its own it is worthless — sampling 16 times is useless if you cannot tell
which one is right. But the domains we care about have **executable
verifiers**:

    type-level TS     tsc --noEmit with expectTypeOf / @ts-expect-error
    Rust -> WASM      cargo test + a node harness asserting across the boundary
    shaders           naga / tint validation, or a diff of emitted WGSL

With a verifier, best-of-N becomes "sample until green". The contest changes
from *27B vs frontier, one shot each* -- which we lose -- to *27B with sixteen
tries and a compiler vs frontier with one*, which is winnable.

This is not a guess. With a **sound** external verifier: Game of 24 5% -> 38%,
Graph Coloring 16% -> 37%, Blocksworld 40% -> 87%. Verifiers synthesised as
code took a 30B from 32% -> 87% on tool use. Meanwhile a model critiquing
itself made GPT-4 *worse* on two of three domains. The verifier must be
executable, never an opinion.

## The flywheel

This is the self-compiling-compiler part, and it is the reason the system gets
better by being used.

    generate N candidates
        -> run the executable verifier on each        (ground truth, free)
        -> some pass, some fail
        -> that IS a labelled dataset
        -> train the selector on it
        -> better selector picks winners when several pass or none do
        -> fewer samples needed next time

Every run produces training data as a by-product, labelled by a compiler rather
than by a model's opinion. Nothing else in the stack has that property.

## Where the executable verifier runs out

Sometimes several candidates compile and only one is *right*; sometimes none
compile and we need the closest. That is where the learned selector earns its
place, and where Laya actually fits.

## How Laya fits, after all the measurements

Cut, with evidence:
- **as a truth judge** -- 6/10 against a 5/10 coin flip, systematic false
  positives on negative cases. Scores what a passage is *about*.
- **as a relevance scorer** -- a nonsense query outscored every genuine one.
- **as a compactor** -- compaction is a cross-passage comparison, and its
  scores do not survive that. Holding the passage fixed and varying the query
  separates by 0.497; varying both collapses to 0.000.

Kept, for two things:

**1. The trainable backbone of the selector.** Laya is ModernBERT-large plus a
decision head. The strongest result for exactly that model class is a 396M
ModernBERT cross-encoder distilled from a verifier ensemble, which **retained
98.2% of the ensemble at 99.97% fewer FLOPs** and beat Pass@1 by 19.3 points at
K=100. Convai describe their own checkpoint as "a fast base to specialise, not
a zero-shot decision engine" -- which is exactly how we use it. Zero-shot it
fails; finetuned on verifier-labelled pairs from our own loop, it is the right
starting checkpoint.

**2. The response envelope.** The typed-decision API shape -- several typed
questions batched into one call, each answered with a full probability
distribution -- is the good idea, and it is the interface convention for every
structured response in the stack, not just Laya's.

## The layers

```
  Hermes / opencode            any MCP client
        |
        v
  MCP surface        find_by_meaning  find_definition_opt  find_references
  (stdio/HTTP/WS)    find_by_pattern  read_file_range      describe_index
                     run_check        summarize_text
        |            EIGHT tools -- `code_search.TOOLS`. `record_step` and
        |            `read_rings` are NOT here: they are INTERNAL_TOOLS,
        |            injected by the proxy only, because the work log is
        |            scoped to a conversation key an outside MCP client
        |            cannot supply. With bind_project_context and
        |            delegate_investigation the model sees twelve
        |            (`proxy.OUR_NAMES`, asserted at 12 in test_tools.py).
        v
  ENGINE
    sampling loop      N candidates -> executable verifier -> selector
    rings              durable work log, outside the context window
    probe              mid-layer activations -> "is this one worth keeping"
        |
        v
  RETRIEVAL           tree-sitter symbol tables + BM25. Embeddings are built
                      and OFF by default (CODE_SEARCH_SEMANTIC=0): keyword
                      beat semantic 92/120 vs 77/120, McNemar p=0.0041, and
                      fusion did not rescue it at 87/120, p=0.2266.
        |
        v
  SERVING             llama-swap -> llama.cpp (PrismML fork), OpenAI API on
                      127.0.0.1:11434; the proxy owns :1234 (`scripts/start-stack.bat`)
    canopy    GPU0     Bonsai 2 27B ternary, 147,456 ctx, q8 KV
    rootstock GPU1     embeddings + reranker + Laya, always resident --
                       see "Retrieval" below: the cut rule fired on both and
                       neither has been removed
    graft     GPU1     vision, on demand
    draft     TBD      Qwen3.5-0.8B, vocab-matched, if it measures positive --
                       the consequent is not stated here and nothing in this
                       repo decides it. `docs/BUILD.md` §1 and Sequence step 1
                       own the condition: find a same-lineage draft first, or
                       the item dies. Not measured, not loaded.
```

## What each layer is for, honestly

**Serving** is solved and boring. It works.

**Retrieval** is the layer most likely to shrink. Measured on our own repos,
BM25 beats embeddings significantly: recall@5 92/120 against 77/120, McNemar
p=0.0041.

**The pre-registered condition fired, and the consequent did not happen.** This
document used to say "if fusion does not beat BM25 either, the embedding model
and the reranker come out, which frees ~6 GB and removes an index that goes
stale on every commit." Fusion did not beat BM25 — **87/120 against keyword's
92/120, 3 discordant wins to 8, McNemar p=0.2266, indistinguishable**
(`scripts/eval_retrieval.py`, gauntlet index, n=120). The reranker, measured
separately at n=356 paired rows in `bench/retrieval_results.jsonl`, lost to
plain embedding order 226 discordant to 10.

**That reranker row has since been declared VOID and this page predates the
declaration.** `docs/FINDINGS.md` #20 (2026-09-22) reproduced `/v1/rerank`
scoring a document differently depending on what else is in the same request,
and `search_code` batches up to 40 candidates, so all 356 rows went through the
corrupted path. `docs/PLAN.md` "Cut criteria" records the resolution: nothing
is cut, because the data supports neither cutting nor keeping.

**Both models are nevertheless still resident on GPU1**, and this is
UNRESOLVED rather than an oversight. Cutting them is a product decision, not a
reading of the table, because two things the cut rule assumed have turned out
not to hold:

- The gold queries in both runs derive from **symbol names**, which flatters
  lexical matching and is the single worst query shape for a cross-encoder.
  The conceptual query set that would test the other direction does not exist.
  → `docs/ROADMAP.md` §1.2 states what closes it.
- The reranker's number has been **misread once already** in the opposite
  direction. That misreading — "the 199 zero ranks mean the call was skipped" —
  has since been corrected in place: `docs/ROADMAP.md` §1.3 now opens with the
  correction, and `bench/retrieval.py:106` drives that arm at
  `rerank_cutoff=999`, so a rank of `0` is a miss and not a skipped call.
  → `docs/ROADMAP.md` §1.3, and the note in `docs/PLAN.md` under
  "Cut criteria" for what the row actually shows.

Until those two measurements exist, the resident VRAM is being spent on
components whose cut rule has technically fired. **Somebody has to decide
that deliberately.** Symbol tables stay regardless — they are sqlite and cost
nothing.

**The engine is the product.** Sampling loop, verifiers, selector, rings. This
is the part nobody else has configured for these domains, and it is the only
part that can make a 27B beat a frontier model at anything.

**`rings`** exists because of a specific observed failure: a long session hits
~75% context, compaction summarises, the summary loses "I already did X and it
went green", and the work gets redone. Compaction is lossy in exactly the
dimension that prevents repetition, so the record has to live outside the
window.

## What this is NOT

- Not a general assistant. SWE only, in five named domains.
- Not a frontier competitor in general. It loses at design and synthesis --
  type-level TS *design*, FFI *design*, UI -- and we do not chase those.
- Not a training run. We train heads and probes, never the generator.
- Not a RAG product. Retrieval is a subsystem, and possibly a shrinking one.

## The honest claim

On narrow, mechanically-verifiable edit tasks in your repos, a free local 27B
with sixteen tries and a compiler should beat a frontier model with one try.
On location and non-local-convention tasks it should roughly match, with fewer
calls. On everything else it loses, and the system should be built to know the
difference and say so.
