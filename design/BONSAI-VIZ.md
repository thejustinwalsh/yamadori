# The Tokonoma tree — a procedural bonsai driven by model state

A single living 3D bonsai that *is* the readout. Not a decoration beside the
telemetry: the telemetry, rendered as a tree. It grows, thickens, bleaches and
blooms as the stack works, and it moves slowly enough to be watched rather
than read.

The alcove a bonsai is displayed in is the **tokonoma**, which is already the
first panel in the mockup. This is what goes in it.

---

## 1. The rule this whole document obeys

**Every visual channel is wired to a signal we already collect, or it does not
ship.** No channel invents a number, and no channel is driven by a constant
dressed up as data.

This is not aesthetic purity. A dashboard that animates plausible-looking
values teaches you to trust it, and this repo has already lost a day to a
figure read off the wrong axis, and 33 benchmark rows to a health check that
knew too much. A tree that looks busy when nothing is happening is that same
bug with better lighting.

Where a signal does not exist yet the channel stays **inert and obviously
inert** — not randomised, not idled with noise. A dead channel should look
dead.

---

## 2. The mapping: signal to form

Everything in the left column is real today, with its source named.

| tree element | driven by | source |
|---|---|---|
| **Nebari** (root flare spread) | corpus breadth — chunks, defs, roots | `index/code.sqlite3` (7,742 / 5,874 / 1 root) |
| **Root depth and taper** | index tier occupancy | `index/*.sqlite3` row counts |
| **Trunk girth** | accumulated history | `index/nebari.sqlite3` 60 sessions, `corpus` 1,831 events |
| **Nenrin** (growth rings at a cut) | epochs, training cycles | dataset stage completions, `jobs` table |
| **The trunk's two main limbs** | **main vs helper KV split** | `vitals.context` main 88,473 / helper 36,864 |
| **Limb thickness, live** | KV occupancy per lane | `vitals.context`, live `prompt_tokens` |
| **The fork itself (corpus callosum)** | is deep thinking engaged | `shomen` active / idle |
| **Branch extension** | tool calls actually taken | `usage.hops`, `shomen` trace length |
| **Foliage density** | hints above the floor | `hints.select()` count and `_score` |
| **Foliage hue** | hint confidence | cosine margin vs `FLOOR = 0.55` |
| **Bioluminescent arcs** | token throughput, live | streaming rate, tok/s |
| **Arc colour** | decision confidence | Laya top-1 margin, entropy, `act_probability` |
| **Shari** (bleached deadwood) | culled fan-out samples, errored jobs | fan-out losers, `jobs.state='errored'` |
| **Sentei** (fresh pruning cuts) | abstentions, where we chose silence | hints below floor, Laya gate abstain |
| **Crimson caps** | breaker trips, wedged upstreams | `MAX_HOPS` trip, `endpoints[].ok == false` |
| **Moss** | index freshness | index mtime vs corpus mtime |
| **Slab tilt, thermal haze** | GPU pressure | `vitals.gpus[].pct`, `.util`, `.tight` |
| **Ground circuit traces** | per-service health | `vitals.endpoints[]`, the four routes |
| **Sway amplitude** | request concurrency | `admission` semaphore occupancy |

Two deliberate absences.

**"Mood" and "emotion" have no source.** There is no measured affect in this
stack and inventing one would be the exact failure named in §1. What the tree
*can* honestly express is **confidence** (Laya margin and entropy), **effort**
(tier, fan-out width, tool calls) and **strain** (GPU pressure, breaker trips).
Read together those produce something that reads as mood without anything
being fabricated: a low-margin, high-entropy, high-strain tree looks agitated
because each of those three is separately true.

**Attention and KV internals** — per-head activity, layer sparsity — are in
the mockup and we do not collect them. llama.cpp does not surface them on this
path. Inert until it does. See Phase 5.

---

## 3. Fan-out is the tree's branching, and concept_seed is its genome

`mcp/concept_seed.py` draws a concept word by sampling a **random direction in
the model's embedding vector space** and taking the nearest real vocabulary
word. It exists to give **one word per fan-out sample**, in that sample's user
message, so the samples diverge in content rather than only in temperature.
It has **zero callers today**.

This is not a decorative tie-in. It is the correct structural mapping, and it
makes the visualisation and the architecture the same shape:

- A fan-out of N samples is **N limbs from one fork**. `high` and `max` fan out
  3, so three limbs.
- **Each limb's geometry is seeded by its own concept word.** Different words
  produce visibly different limbs: branching angle, node count, taper, lean.
  The divergence you are looking at is the actual divergence in the prompts.
- **The chosen sample's limb thrives.** It extends, thickens, takes foliage.
- **The culled samples bleach to `shari`.** They stay on the tree as deadwood.
  This is the honest visual and it mirrors a decision already made in code:
  `errored` is terminal and distinct from `done`, and losing work is never
  silently dropped. You can see what was considered and rejected.
- **A fan-out of 1** (`minimal`, `low`, `medium`) is a single trunk with no
  fork. The tier is legible at a glance from silhouette alone.

The word is hashed to a `u32` driving a deterministic PRNG (`mulberry32`), so
**the same word always grows the same limb** — across reloads, machines and
sessions. Two people looking at the same run see the same tree. That is a real
property and it is cheaply testable: same seed in, byte-identical skeleton
out, no GPU required. It is also the reason to use an explicit PRNG and never
`Math.random()`.

Words are shown in the tokonoma caption, the way a displayed bonsai is
labelled.

---

## 4. The hard problem: interpolating between states

> *"it lerps slowly between different samples"*

This decides whether the feature is beautiful or janky, so it gets designed
first.

**The naive approach fails.** Generate an L-system per state and cross-fade.
Two L-systems seeded differently have different branch counts and different
topology, so there is nothing to interpolate — branches pop in and out and the
whole thing flickers. Morph targets do not rescue it; they need matching
vertex counts.

**The design: fixed superset topology, animated parameters.**

1. **Grow once.** From the seed words, deterministically build a *superset*
   skeleton: the maximum tree these seeds will ever produce. Every branch that
   could ever exist exists now. Runs once, never again while the seeds hold.
2. **Every branch carries a parameter vector**, not a presence flag:
   `{ length, radius, pitch, yaw, alive, foliage, emissive, bleach }`.
3. **State maps to that vector** per branch, by a pure function.
4. **Interpolation is vector lerp** over a fixed structure. Branches never
   appear or vanish. They grow from zero length, thicken, bleach toward bone,
   retract. Nothing pops because nothing is added or removed.
5. **One instanced draw.** Branches are instanced tapered cylinders, per
   instance matrices and colours straight from the parameter vector. Foliage
   is a second instanced pass.

Three consequences worth stating because they are load-bearing:

- **Death is retraction, not deletion.** A culled fan-out limb shrinks and
  bleaches rather than disappearing, which is both the better visual and the
  truthful one.
- **Springs, not linear tweens.** Critically damped, per channel, with
  different stiffness: throughput arcs snap, trunk girth crawls. A burst of
  work is felt as a lean.
- **Never lerp topology.** New fan-out, new words, new tree. Cross-dissolve
  two rendered trees over about two seconds, then discard the old skeleton.

---

## 5. The look

Target is the reference render: obsidian void, slate slab, volumetric shafts
from above, cyan emissive arcs threading the branches, moss, crimson caps,
circuit traces bleeding neon beneath.

- **Renderer.** `ACESFilmicToneMapping`, sRGB output, physically correct
  lights, tuned exposure constant.
- **Lighting: three lights, no more.** Warm key at high angle for the shafts,
  cold cyan rim behind-left to separate bark from the void, very dim fill.
  Everything else is emissive plus bloom.
- **Shadows.** PCSS soft shadows from the key only, one map, tight ortho
  frustum around the slab. Contact hardening is what sells the slab.
- **Bloom must be selective** — emissive arcs and caps only, layer-masked.
  Full-scene bloom washes the bark out and is the single fastest way to make
  this look cheap.
- **God rays.** Screen-space radial blur from the key with an occlusion pass.
  Against a black void with one light this is indistinguishable from raymarched
  volumetrics at a fraction of the cost.
- **Bark.** Triplanar mapped so taper never shows UV stretch. Bleach toward
  `shari` by raising roughness and pulling albedo to bone, never by tinting —
  tinting reads as plastic.
- **Moss.** Shell texturing, six to eight shells, alpha cut, on nebari and
  slab. Instanced grass costs more and looks worse at this scale.
- **Palette** comes from `design/DESIGN.md`'s **frontmatter**, which is the
  authoritative one. See `design/README.md`: the prose section is a second,
  stale palette and seven of its eight colours appear in no mockup.

---

## 6. The constraint nobody will think of until it bites

**This renders on the same two GPUs that run the model.** A 60fps showpiece
beside a 27B inference job is not free, and this stack's thesis is measured in
tok/s.

- **Render on demand.** r3f `frameloop="demand"`, or a manual invalidate loop.
  The tree is static at rest and state arrives every few seconds: render only
  while a spring is settling, then stop. At rest the cost is **zero draw
  calls**, not a cheap frame.
- Cap `devicePixelRatio` at 1.5 and expose it.
- Pause on `visibilitychange` and on `IntersectionObserver` miss.
- A hard off switch degrading to the 2D panels.
- **Budget it, then measure it.** Frame time and GPU delta recorded, and a
  before/after on tok/s with the tab open. If the showpiece costs measurable
  throughput, that number gets published next to it rather than hidden.

---

## 7. What is testable without a GPU

Deliberately almost all of it, matching how everything else here is tested.

| test | asserts |
|---|---|
| seed determinism | same word produces an identical skeleton, over 100 words |
| mapping purity | `state -> params` has no clock, no RNG, no I/O |
| mapping range | every channel stays in domain for adversarial state: empty index, all endpoints down, pool exhausted |
| inert channels | an unwired channel emits its inert value, never noise |
| no fabrication | every channel names a key that exists in `vitals.snapshot()` |
| interpolation | lerping any two valid vectors stays in range at all `t` |
| topology stability | branch count constant across every state for fixed seeds |
| fan-out arity | N seeds produce exactly N limbs; culled limbs bleach, never vanish |

Only lighting and perf need the card.

---

## 8. Phases

1. **Skeleton and determinism.** PRNG, superset topology, instanced branches,
   flat lighting. Prove same-seed-same-tree. No state wiring at all.
2. **The mapping layer.** Pure `state -> params`, fully tested, driven by a
   recorded `vitals` fixture so it replays history offline.
3. **Wire concept_seed into fan-out.** This is owed anyway — the module has
   zero callers and the seed belongs in each sub-agent's user message. The
   visualisation is the forcing function, not the justification.
4. **Live wiring.** `/dash/api/vitals` polling or SSE, spring interpolation,
   cross-dissolve on new fan-out.
5. **The showpiece pass.** PCSS, selective bloom, god rays, triplanar bark,
   shell moss, slab. This is where the hours go; everything before it is cheap.
6. **Open doors.** Per-head and per-layer channels *if* llama.cpp can be made
   to surface them. Inert and labelled until then.

Phases 1 through 3 need no GPU and no dashboard rewrite, so they can land
independently of the React decision.

---

## 9. Open questions for the operator

1. **Per-session tree or per-stack tree?** A tree of *this conversation* and a
   tree of *the whole node* are different instruments. The fan-out mapping in
   §3 is per-request, which argues for per-session with the stack-level
   signals (GPU, endpoints, index) as the slab and roots rather than the
   canopy.
2. **How long do culled limbs stay?** `shari` is permanent on a real tree.
   Keeping every rejected fan-out sample forever eventually makes a tree that
   is mostly deadwood — which may be the honest picture, or may be noise.
   Decay over a session, or keep?
3. **Ship behind a flag by default?** Given §6, recommend yes.

---

## 10. The latent-space axis, and exactly how much of it is real

> *"a view into the live state of the 4d latent space of the model"*

Most of this is reachable today. The rest needs a patch, and the difference
matters, so here is what was probed against the running server rather than
assumed.

### Available right now, no changes

**Per-token top-k logprobs.** Verified live: the model returns `logprobs` with
`top_logprobs: 5`, each carrying token id, text, bytes and logprob — and it
returns them for `reasoning_content` as well as content. That is a real,
per-token window into the distribution, on the thinking stream, at zero extra
cost because it rides the response we already stream.

Four honest scalars fall straight out of it, per token:

| quantity | from | reads as |
|---|---|---|
| **entropy** over top-k | the 5 logprobs | uncertainty, how many futures are live |
| **margin** top1 - top2 | two logprobs | decisiveness |
| **surprise** -logprob(chosen) | chosen token | did it take an unlikely path |
| **branch factor** exp(entropy) | derived | effective live candidates |

Measured on the probe above: the first token sat at margin 0.42 nats with five
plausible continuations, the third at 0.23 with the top choice dominant. The
model's confidence visibly changes token to token, and that is the signal.

### The 3D basis, and why it must be frozen

Token ids are the bridge to actual latent space. The GGUF carries
`token_embd.weight`, the model's own input embedding matrix. Extract it
**once, offline**, fit PCA to three components, and **freeze that basis
forever**.

Freezing is not an optimisation, it is the whole trick. Refitting a projection
per frame — the classic t-SNE and UMAP mistake — makes the space itself jump
around, so motion on screen stops meaning motion in the model. A frozen basis
means a point that moves, moved. It is the same principle as §4's fixed
superset topology: **hold the structure still so the change is legible.**

Then every generated token is a real point in the model's own space, at zero
runtime cost — one table lookup.

### The fourth dimension is time, and it is the branch

Three spatial axes plus token order is the 4D. And it maps onto the tree
without forcing:

**The branch grows along the generation's latent trajectory.** The path the
model walks through its own embedding space *is* the shape of the wood.

- **Path** = the sequence of token points. Smoothed with a Catmull-Rom spline,
  swept to a tube.
- **Local thickness** = inverse entropy. Confident passages are thick, dense
  wood; uncertain ones thin and whippy.
- **Knots** = surprise spikes. A token the model did not expect leaves a knot
  in the grain, permanently.
- **Curvature** = how far consecutive tokens sit apart in latent space. A
  topic shift is a visible bend.
- **Emissive intensity** along the tube = live throughput, already in §2.

Fan-out becomes literal: N samples are N trajectories from a shared prefix, so
they share a trunk and diverge exactly where their token paths diverge. The
fork is not a metaphor — it is where the distributions parted.

### What this does NOT give, stated plainly

**This is the input embedding manifold, not the residual stream.** Thinking
happens in the residual stream at layer N, and `llama-server` does not expose
it. What the trajectory shows is which tokens the model chose and how sure it
was — real, live, and a genuine projection of its state — but it is not the
deep latent space, and calling it that would be the §1 failure with better
marketing.

Two routes to the real thing, both costed honestly:

1. **`--embeddings` on bonsai** exposes final-layer pooled hidden states.
   Probed: `/embedding` currently returns **501**, not enabled. Turning it on
   is not free — in llama.cpp it conflicts with generation on the same slot,
   so it means a second 27B instance, and there is no card for one. Cost: a
   GPU we do not have.
2. **Patch the PrismML fork** to emit per-layer residuals on a side channel.
   This is the only route to the actual 4D latent state. Real work, and the
   fork is already ours, so it is possible rather than hypothetical.

### Chaos, and the answer to it

The worry is right — a raw latent trajectory at 16 tok/s is a twitching mess.
Three things tame it without lying:

1. **The tree is slow, the trajectory is fast.** Token-rate detail goes into
   *grain* — thickness, knots, curvature baked into geometry as it grows —
   never into anything that moves. Growth is spring-damped per §4. You read
   the history in the wood, you do not watch it vibrate.
2. **One axis, as suggested.** Latent trajectory drives branch *path* only.
   Girth, roots, foliage and slab stay on the §2 signals. If the trajectory is
   noisy, one channel is noisy, and the readout survives.
3. **Grain is permanent, position is not.** A knot stays. That turns
   token-rate chaos into a record instead of a flicker, which is also the more
   honest object: the tree remembers what it was uncertain about.

### Phase placement

This is **Phase 2.5**, between the pure mapping layer and live wiring. It
needs the offline PCA artefact (one script, one `.npy`, no GPU) and a
logprobs-capable streaming path in the proxy. Both are testable headless:
fixed token-id sequence in, identical trajectory out.
