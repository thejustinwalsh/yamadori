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
                     run_check        record_step/read_rings
        |
        v
  ENGINE
    sampling loop      N candidates -> executable verifier -> selector
    rings              durable work log, outside the context window
    probe              mid-layer activations -> "is this one worth keeping"
        |
        v
  RETRIEVAL           tree-sitter symbol tables + BM25 (+ embeddings, pending
                      the cut rule -- keyword currently beats semantic
                      92/120 vs 77/120, p=0.0041)
        |
        v
  SERVING             llama-swap -> llama.cpp (PrismML fork), OpenAI API :1234
    canopy    GPU0     Bonsai 2 27B ternary, 208k ctx, q8 KV
    rootstock GPU1     embeddings + reranker + Laya, always resident
    graft     GPU1     vision, on demand
    draft     TBD      Qwen3.5-0.8B, vocab-matched, if it measures positive
```

## What each layer is for, honestly

**Serving** is solved and boring. It works.

**Retrieval** is the layer most likely to shrink. Measured on our own repos,
BM25 beats embeddings significantly. If fusion does not beat BM25 either, the
embedding model and the reranker come out, which frees ~6 GB and removes an
index that goes stale on every commit. Symbol tables stay regardless -- they
are sqlite and cost nothing.

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
