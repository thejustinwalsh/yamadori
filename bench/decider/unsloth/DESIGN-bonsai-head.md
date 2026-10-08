# A Clef-style head on a FROZEN Bonsai (phase 4 design; operator, 2026-10-07)

The ask: "can we do this with one of our models ... a model we already have in our stack, but for someone with tighter
memory it would be nice to fit say bonsai and a smaller decider on the same card." Bonsai is a ternary PTQ GGUF
(PTQ1_0, Hadamard-rotated, 64 layers, hidden 5120, vocab 248,320): LoRA is impossible, so the only trainable part is a
head on its frozen hidden states. `bonsai_head.py` is the feasibility test; this note is the design and what is and is
not established.

## What the head reads, and why a frozen backbone can carry it

- Cloudflare's head (`joint_schema_model.JointSchemaHead`, Unsloth's subclass is name- and shape-identical) takes the
  backbone's FINAL hidden states for every token (LayerNorm'd inside the head), mean-pools them over each question's and each
  option's token span, routes evidence from the whole sequence into each option with 2 cross-attention layers (width 1024), decodes
  the fields jointly with 4 more, and scores each option by cosine plus a small residual scorer. Its `lexical` input is the
  model's OWN output-embedding rows for the option tokens.
- Unsloth's from-LM recipe has a head-only stage (`FastDecisionModel.freeze_backbone`, "Stage A ... train only the decision
  head on frozen backbone features") before the LoRA stage; Cloudflare's reference stack is LoRA rank 256 + head
  ([H] blog summary). So "head only on a frozen model" is a documented first stage, not a novelty. Whether it is enough is
  the open question this test measures.
- llama.cpp can hand back those hidden states: `llama-server --embeddings --pooling none` returns one vector per token of the
  final-norm output (`/embedding`, the token array as `content`). The engine's `result_embd` is the post-`output_norm` state,
  the same tensor HF's `last_hidden_state` is. Requires `-ub >= n_tokens`.

## The experiment (small scale, `bonsai_head.py`)

| step | what | size |
|---|---|---|
| train set | the guide's mix (prepare_data.py; equal weight per source, schema-augmented), first 3,000 rows | 5,037 decisions, 1.16M tokens, 11.9 GB fp16 of states |
| held-out | typed-decisions test (400 cases = 2,000 decisions) + 150 BANKING77 + 60 CLINC150 | 610 records, 2,210 decisions, 0.81M tokens, 8.3 GB |
| our sets | JevBench's 231 public items (167k tokens, 1.7 GB); skill cases, relevance + need as two noul questions per item, one record per case (183 of 201 fit 4,096 tokens as one record, the rest are split by items) | 460k tokens, 4.7 GB |
| head | width 1024, 2 + 4 layers, 16 heads, ff 4096: 128,056,324 parameters (selftest) | 3 epochs, lr 3e-4 (Studio's fresh-head rate), batch 4 |
| baseline | jjava's recorded reads on the same JevBench items (Intelligence 80.46; tiers .. / .944 / .739) and skill beliefs (relevance AUROC 0.902, need-vs-rest 0.693, need-vs-AREA 0.638) | recorded, not re-run |

What would count as a result: on the SAME held-out decisions, the head's accuracy and ECE against the 2B LoRA+head run (phase 2)
and, on JevBench and the skill cases, against jjava's letter read. A head trained on 3,000 generic rows that does not beat
jjava on JevBench's hard tier is the expected outcome (reflex found its adapters worse on long ambiguous inputs;
DECIDER-RESEARCH 3.2/4.3); what the test can show is whether the frozen ternary states carry the decision at all (the
guide's table starts at chance and reaches 73-78% with a LoRA; a head alone on Bonsai could land anywhere between).
Contamination: the mix includes the BANKING77 / CLINC150 / MMLU / ARC TRAIN splits; the held-out rows are test splits, decontaminated.

## Cost, measured later (nothing here is a measurement yet)

- Extraction is a prefill of every record on the 27B (about 2.6M tokens for the whole set above) on a card with room for the 5.9 GB
  GGUF plus an 8,192-token embeddings context.
- At serve time the head costs one pass over the same hidden states plus ~128M parameters (256 MB bf16, ~136 MB at Q8), against
  jjava's 2K next-token reads per call; the head replaces `/decide-batch` for typed calls, since a joint head scores ALL questions
  of a call in ONE sequence (state + every question and option), which is Clef's own layout.

## How `/decide-batch` (patch 0042) would apply the head -- two routes

1. **In the engine (the real route).** Graft the head into the Bonsai GGUF the way the MTP head was grafted (`graft/tools/merge.py`
   for `blk.64.*`): `joint_head.*` tensors as extra GGUF tensors, plus the three learned scalars folded as the converter does. Port
   `src/models/clef.cpp` (616 lines) and the `llama_batch_ext_set_decision_order` plumbing (24 files in #29831, on top of #29818's
   decision-model infrastructure) onto the prism base, which has none of it. `/decide-batch` stays the route for next-token reads;
   a head call is a different endpoint (Jev-shaped `/v1/systemone`, which `mcp/jev_api.py` already speaks) answered from
   `result->scores`. The head runs inside the same llama-server as Bonsai: no second model in VRAM, ~+2% weights.
2. **Outside the engine (the cheap test of the idea).** `/embedding` returns the states; the head runs in PyTorch on the A4000.
   JSON transport of 5,120 floats per token makes it a measurement tool, not a serving path.

## Open decisions

- Where the extraction server runs (a free card, started by us, never through llama-swap).
- Whether a positive result is worth the port; the port is the cost, not the head.
