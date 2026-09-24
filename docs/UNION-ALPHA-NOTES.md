# Union Alpha, and what a multi-model gateway could teach Yamadori

Written 2026-09-24 for the operator, as a design memo. **Nothing in it is
built.** No code or config changed, no GPU job ran, and nothing was sent to
`:1234`, `:11434` or `:1237`. External pages were read on 2026-09-24 and
treated as data. None of them contained text addressed to an agent.

Labels:

| label | meaning |
|---|---|
| **[V]** | verified on an official page (the maker, OpenRouter or Cloudflare). This verifies that the claim was *published*. It does not verify that the claim is true. |
| **[R]** | reported by press, blogs or aggregators, not checked against a primary source |
| **[S]** | speculation, including unsourced "analysis" |
| **[M]** | measured in this repo: names its artefact and its n |
| **[P]** | proposed here; untested |

---

## 1. What Union Alpha is

**It is no longer anonymous.** Union Alpha was Unbiased's **Pareto 26.9**.
Unbiased is the AI platform of Circuit & Chisel.

| fact | label | source |
|---|---|---|
| Listed as `stealth/union-alpha` on OpenRouter on 2026-09-16, and also served by OpenCode and Cloudflare AI Gateway. It was free during the preview. | [V] OpenRouter's and Cloudflare's own X posts (read through the search index; X was not fetched) | [1] [2] |
| "A multimodal model designed for research, coding, and agentic workflows", "frontier-level" | [V] | [3] |
| Context 262,144 tokens, output up to 131,072 tokens. Input is text plus image, output is text. Tool calling is supported. | [V] for 256K and tool calling [1]; [R] for the exact figures [5] [6] | |
| Revealed at about 23:24 UTC on 2026-09-17: OpenRouter posted "Union Alpha is revealed on OpenRouter as @TheUnbiasedCo Pareto", crediting community "detective work". The stealth period lasted about 33 hours of a planned week. | [V] OpenRouter X post [4]; timings [R] [5] | |
| Cloudflare's `unbiased/pareto` page shows `"model": "union-alpha"` in its example responses. | [V] | [7] |
| Price after the reveal: $2.50 per M input tokens, $0.25 cached, $7.50 output. | [V] maker [8] [9] | |
| The free preview ended early after demand reached "billions of tokens per minute". | [R] | [5] [10] [11] |
| Official launch set for 2026-10-10. | [R] | [11] |
| `openrouter.ai/stealth/union-alpha` now returns 404. | observed 2026-09-24 | |

**Benchmarks come from the vendor only** [V, published]. The model card
reports DeepSWE 74, Terminal-Bench 4.0 51, MMMU-Pro 78, HLE 49 and ArXivMath
88, against Fable 5.1, GPT 6 Astra and DeepSeek 4.1 Flash [9]. The card has
its own caveats. Headline scores and task costs come from *separate
slates*. The one cost figure (70.0% DeepSWE at $0.29 a task) is "transcribed
from the supplied September 21, 2026 screenshot, not independently
validated". The card says "Do not combine these scores with the preliminary
task costs". The widely repeated "~73% at ~$0.65 a task" [12] is [R], and
that article itself calls the costs "estimated". No independent benchmark was
found.

## 2. The architecture claim, weighed

The operator's paragraph describes a "parallel-mixture or multi-tier gateway
that routes requests". The maker says the first half and explicitly denies
the second.

| claim | label | evidence |
|---|---|---|
| Several models run **in parallel on every request**, and one answer is synthesised | **[V]**, the maker's own statement | "Multiple LLMs are engaged in parallel on every request, and their results are synthesized dynamically based on the task" [8]. Cloudflare's page repeats it [7]. The site: "runs a mix of frontier and open source models against each other on every request, checks which one earns the answer, and keeps the best result" [13]. |
| It is **not a router** | **[V]**, the maker | "not a 'model router'", and it "works the same way on easy prompts and hard ones" [8]. The stated reason: routers switch models, and "the cache misses erase the savings". Pareto "never switches models mid-conversation, so your prompt cache ... stay[s] intact" [13]. |
| The blend includes "frontier and open models", and **its composition can change** between releases | [V] [8]; the terms allow change "if functionality is not materially reduced" [R] [14] | The component models are **not disclosed** anywhere [9]. |
| How the answer is chosen (judge, verifier, vote) | **unknown** | "checks which one earns the answer" [13] is the only statement. The mechanism is not published. |
| Latency | **unknown** | "Latency varies by workload. Current-release latency comparisons have not been published" [8]. |
| It "escalates difficult tasks" to a stronger model | [R] third-party paraphrase [14] [15] | This contradicts "the same way on easy prompts and hard ones" [8]. Treat multi-tier escalation as unconfirmed. |
| "Speculative parallel MoA": 3–4 specialist agents (AST diff, test fixtures, logic audit), then consensus. "Cloudflare telemetry ... confirmed" it. TTFT 680 ms / 2,850 ms, 68 tok/s. | **[S]** | [16] asserts this with no link to any Cloudflare report. Cloudflare's own docs say nothing about the architecture [3] [7]. |
| Evidence from outside that it is multi-model | [R], circumstantial | Early testers "struggled to identify a single parent model from style, refusals or benchmark behavior" [14]. No published fingerprint study, latency distribution or per-request style analysis was found. The one hands-on test found no architectural evidence either way [12]. |

**Bottom line.** Union Alpha/Pareto is a **per-request ensemble with
selection or synthesis**, in the MoA / best-of-n family. By its maker's own
account it is not a per-request router and not a cascade. What it contributes
to design is a single argument: *keep one model per conversation so the
prompt cache survives, and buy quality with parallel candidates.* That
argument transfers directly to our slot pinning (§5). The quality claims are
all vendor-run and cannot be audited.

## 3. The public techniques such a gateway would use

These are [R]: headline results as their authors report them.

| technique | mechanism | reported result | source |
|---|---|---|---|
| **FrugalGPT** | A cascade: try a cheap model first, score the answer, and escalate if the score is low | matches GPT-4 at up to 98% lower cost | [17] |
| **RouteLLM** | A router learned from preference data that picks strong or weak before generation | more than 2× cost reduction "without sacrificing response quality" | [18] |
| **AutoMix** | The small model answers and verifies itself; a meta-verifier decides whether to escalate | better cost-quality than fixed routing | [19] |
| **Hybrid LLM** | A difficulty-predicting router with a tunable quality threshold | up to 40% fewer large-model calls with no quality drop | [20] |
| **Not Diamond** | A trained router. Per turn it considers session state, token counts and **KV-cache state**, and models future cost and reward | vendor; no independent n | [21] |
| **Martian** | A router built on "model mapping" | vendor: 20–97% cost cut | [22] |
| **OpenRouter Auto Router** | A lightweight classifier assigns one of about 30 task types. Models are ranked by 7-day community spend share within a cost band (`low`…`max`), with fallbacks | no quality number; ranking is by "the wisdom of the market" | [23] |
| **Arch-Router** | A 1.5B generative router keyed to user-defined *domain* × *action* labels | beats proprietary LLMs on preference-matched routing | [24] |
| **Mixture-of-Agents (Together)** | Layers of proposers, then an aggregator that synthesises | 65.1% on AlpacaEval 2.0 with open models only | [25] [26] |
| **Self-MoA** | The same MoA, but every proposer is the **single best model**, sampled repeatedly | **beats mixed MoA** by 6.6% on AlpacaEval 2.0 and 3.8% on average. "MoA is highly sensitive to proposer quality": mixing weaker models lowers the average. | [27] |
| **Speculative cascades** | A cascade whose deferral rule runs as speculative execution, so the small model drafts and the large one verifies with a lossy acceptance rule | better cost-quality than either cascades or speculative decoding (Gemma, T5) | [28] |
| **Verifier / best-of-n** | Sample n answers and let a verifier pick. Repeated sampling raises *coverage* roughly log-linearly, but without an automatic verifier, voting and reward models plateau. | [29] [30] [31] | |
| **Router reality check** | Four open-source routers under one common interface, on 290 frozen tasks | three gave near-constant tier assignments. **"Always-Mid" matched or nearly matched a router.** Gains tracked *which tier* was picked, not task-aware routing. | [32] |

Two results matter most here. **Self-MoA** says our same-weights fan-out is
the *right* shape, not a poor substitute for a multi-lab mixture. The
**router reality check** says any router we build must beat "always the same
model" before it is credited with anything.

## 4. What fits a single-GPU, memory-bound local stack

The constraints [M, `config.yaml`]:

- The 5060 Ti holds Bonsai (5.95 GB of weights) plus a 262,144-token q4_0 KV
  pool, with a 1.3 GB free-VRAM target. Nothing else fits on that card.
- The A4000 holds retrieval plus Laya: 7,565 MiB resident, **8,605 MiB free**.
  Image generation peaks at +6,389 MiB. The vision copy needs 8,265–9,449 MiB
  and is `exclusive` (it evicts retrieval).
- A different-weights 27B critic (Qwen3.8-27B Q4_K_M, about 19.7 GiB) is
  disabled because **it does not fit**.

| mechanism | fits? | why / the local analogue |
|---|---|---|
| Parallel mixture of different frontier models on every request (Pareto) | **no** | It needs several concurrent large models. We have one 27B that fits and `HELPER_LANES = 1`. |
| Sequential same-model ensemble: A, then B, grade, then tie-breaker C | **already built** (`mcp/fanout.py`) | This is Self-MoA-Seq in shape [27]. Our grader is the code check, not an LLM aggregator. |
| **Cross-card parallel B** (a second Bonsai process on the A4000 writes B while A runs on the 5060 Ti) | **maybe; measure it** | This is the only true parallelism available. Bonsai decodes 54.2–54.9 tok/s on the A4000 with MTP [M, `MTP-STAGING.md`, staging build], so the card is not slow. It is the vision copy's footprint without the mmproj, which is marginal against 8,605 MiB free (the config's own analysis). It would free the helper's 3/8 on the 5060 Ti. Risk: evicting retrieval breaks B's tool calls. |
| Routing **across hot-swapped models** by class (llama-swap) | **only on the A4000, only for models ≤ ~6 GiB all-in, and only per conversation** | The primary is `persistent` and never swaps. A swap costs a load (~11 s, the config's critic comment, not a measurement) **plus a full re-prefill of the conversation** (§5). |
| Per-turn switching of models inside one conversation | **no** | It throws away the pinned prefix. This is the maker's own argument [13], and our numbers make it concrete (§5). |
| Cascade (a cheap attempt, verify, escalate on failure) | **yes, in form**. The cheap tier is Bonsai at lower effort or without fan-out, not a smaller model. | It is bounded by the verifier's recall (§6). |
| Separate small draft model for speculative decoding | **not worth it** | Bonsai already runs an MTP draft head (`--spec-type draft-mtp`, +68% decode on the A4000 [M, staging]). A draft on the other card crosses PCIe on every step and needs a matching vocabulary. It would have to beat MTP, and nothing suggests it would. |
| Speculative cascades [28] | **no** | It needs two models verifying at token level in one server process. Nothing in our serving path does that. |
| Market-ranked auto-routing (OpenRouter) | **no market** | Our "market" is our own paired benchmark rows (§5). |
| Classifier routing by task type | **already built** (`mcp/route.py`, 6 deterministic classes) | Today the class selects *pipelines*. The registry below would let it select *models* as well. |

## 5. A model registry and capability profile

**Principle.** Bonsai is the default for every class. Another model takes a
class only on paired evidence, and only per conversation. This is the
local reading of both Pareto's "never switch mid-conversation" and the finding
that "Always-Mid" matches routers.

### 5.1 The switching cost, which decides where routing can pay [M]

| quantity | value | source |
|---|---|---|
| Prefix reused on a live Hermes turn when it returns to its pinned slot | 37,791 of 40,080 prompt tokens | `mcp/slots.py` docstring, n=1 |
| Prefill, 5060 Ti, short prompt | about 490 tok/s | `KNOWN-ISSUES.md` |
| Decode, 5060 Ti (official prism, no MTP) | 46.06 tok/s | `KNOWN-ISSUES.md` |
| Model load through llama-swap | about 11 s | a `config.yaml` comment; **not measured** |

Moving a 40k-token conversation to another model costs about 80 s of
prefill at 490 tok/s, plus the load. The 490 tok/s is for short prompts;
long-prompt prefill is slower and unmeasured here. That is **longer than a
typical whole answer** (about 44 tok/s bounds rumination at 190–205 s,
`CONSTRAINTS.md`). So:

- **Per-conversation routing only.** The class of a conversation's *first*
  code turn picks its model, and the pick sticks.
- **Stateless side calls are the cheap place to route.** These are the
  `utility` class, minus compactions. A compaction must stay on the pinned
  slot for prefix reuse (Phase 0 workstream C, "compaction affinity").

### 5.2 The registry [P]

One row per servable model, kept beside `config.yaml` (for example
`index/models/registry.json`). Every number carries its run id and n
(PROTOCOL rules 10 and 13).

| field | how it is filled |
|---|---|
| `id`, `card`, `group`, `vram_mib_peak`, `load_s` | measured cold and warm, n ≥ 5, with the other card's residents loaded |
| `prefill_tps@{8k,32k,64k}`, `decode_tps` | measured on its own card, not borrowed from a sibling card |
| `efforts`, `template_quirks`, `tool_call_ok` | read from the served template (as `tiers.accepted_efforts()` does), plus a 10-call native tool check like the one in `KNOWN-ISSUES` (9/9) |
| `wh_per_task[class]` | `mcp/power.py` board watts integrated over each benchmark row, **both cards** |
| `pass[class]` | paired rows from the domain suite (uncontaminated 120) and LiveBench categories, mapped to `route.py` classes, with a Wilson interval |
| `n[class]`, `run_ids[class]`, `contaminated` | carried with every number. A three.js or type-challenges row is never a headline. |

**Which classes can have a profile at all:**

| class | data we have or will have | profile possible? |
|---|---|---|
| `code_generation`, `code_edit` | domain suite, LiveBench coding, LCB | **yes**. Most benchmark rows are these two. |
| `utility` | Hermes logs: titles, approval checks | **by agreement with Bonsai only**. There are no ground-truth labels. |
| `agent_step` | Hermes task bank (Phase 1) | later: needs pass rates per *task*, not per turn |
| `library_question` | context-economy set, held-package questions | small n; one direction only |
| `prose` | none | **no.** Stay on Bonsai. |

### 5.3 Routing policy [P]

For class `c` and a candidate model `m`, `m` replaces Bonsai for `c` only if
both hold:

1. **Quality:** `pass[m,c]` is not worse than Bonsai's, by a paired
   non-inferiority test at a margin set *before* the run. Or it is better at
   p < 0.05 exact McNemar. The result must **repeat on a fresh run**
   (PROTOCOL rule 10: at temperature 1.0 the bare LiveBench arm disagreed
   with itself on 8 of 21 questions [M, `STACK-TUNING-2026-09-23.md`]).
2. **Cost:** lower `wh_per_task` or lower wall time, **including**
   `load_s` and the re-prefill of §5.1, amortised over the measured
   conversation length for that class.

A route decision is recorded in `x_yamadori` like every selection decision.
A failed load or an unavailable model **must be reported** in
`x_yamadori`, not silently served by Bonsai (PROTOCOL rule 2: a fallback that
hides an outage is a bug).

### 5.4 What the A4000 could host when idle [P]

The A4000 has 8,605 MiB free while image generation is unloaded. When it
loads, only 2,213 MiB are left, and the two must not coexist.

| candidate | fits? | what it would be for | first measurement |
|---|---|---|---|
| A **Bonsai text copy** at small context (the B writer of §4) | marginal; same arithmetic as the vision copy | parallel fan-out B, or a second deep-thinking lane | VRAM peak with retrieval and Laya resident; tokens/s; fan-out wall time |
| A **small model (≤ ~4 GB in weights)** for `utility` side calls | yes | titles and short classifications, keeping them off the main slots | agreement with Bonsai on logged utility calls, in shadow |
| A **different-weights 27B critic** at ~12 GiB (IQ4_XS / Q3_K_XL, per the config) | only with the card exclusive (evicting retrieval). Room beside Laya is unmeasured. | a heterogeneous reviewer. Self-MoA [27] predicts that a *weaker* proposer lowers quality. | first, is it at least Bonsai's equal on the code classes? |

**Swap policy.** Image generation and any specialist share one llama-swap
group (`swap: true`), so they never coexist. The group must stay
`exclusive: false` so retrieval is not evicted, and vision stays `exclusive`
and locked. Any card-sharing change is a maintenance-window restart, after
which the whole live suite runs again (PROTOCOL rule 1).

### 5.5 KV reuse across roles [P]

Fan-out B in `mcp/fanout.py` uses the **"direct" variant**, which has no
system nudge. It copies the whole conversation and appends the concept seed
to the *last* user turn. So B's prompt shares A's prefix up to that turn, but
B runs on the HELPER slot and **re-prefills the whole conversation**. On a
40k-token agent context that is about 80 s of avoidable prefill per fan-out.

llama-server can save and restore a slot's state, but only when
`--slot-save-path` is set, which our config does not do. Whether this fork
restores the Bonsai 2 hybrid (recurrent) state correctly, and how large the
file is, are **unmeasured**. The test: restore A's slot onto the helper
slot before B, then compare B's prompt-processing seconds with and without
the restore, paired on ≥ 20 fan-out requests. As a correctness check, a
fixed seed must produce identical B output. The same move would serve the
tie-breaker C, whose prompt also starts with the conversation.

## 6. Cascade and escalation

We already have the pieces: effort tiers, the code check (a verifier that
parses and formats but never runs code), the repair pass, and fan-out. The
Union-Alpha-style question is whether to run the expensive path *always*
(Pareto) or *only on failure* (FrugalGPT / AutoMix).

**The cascade is bounded by the verifier's recall** [P, arithmetic]. When the
verifier triggers on a fraction *r* of the answers that actually fail, and
escalation fixes a fraction *f* of those, the most the pass rate can rise
is *r × f × (fail rate)*. The evidence says *r* is small for a parse-only
check. On LiveBench, 18 of 21 bare programs compiled and all three failures
were format errors, and a compiled answer can still fail the tests
[M, `STACK-TUNING-2026-09-23.md`, n=21].

| step | what it does | evidence needed before building or keeping it |
|---|---|---|
| 1. A cheap first attempt | Bonsai at `medium`, no fan-out | its pass rate per class, with the noise floor measured by a repeat |
| 2. Verify | `check_code` (parse, lint, format); `run_check` where a project whitelists one | **a confusion matrix of verifier-fail against hidden-test-fail** on existing rows: recall, precision, n. This is CPU-only and uses rows we already have. |
| 3a. Escalate by repair | feed the error back to the model | the repair save rate: rows where repair turned a fail into a pass, paired with no repair |
| 3b. Escalate by fan-out | B, then C, in the helper role | fan-out pass rate *conditioned on the verifier firing*, against always-on fan-out at matched effort |
| 3c. Escalate to a different model | a registry model | §5.3's gate, on the rows where the verifier fired |
| Always-on alternative (Pareto-like) | fan-out on every code request at `high`/`max`, as today | always-on against verifier-gated fan-out: same pass rate, and how much less GPU time and energy? |

If *r* comes out small, the right conclusion is that **verifier-gated
escalation cannot pay** until the verifier is stronger (tests, or `run_check`
labels). Always-on fan-out at the higher tiers would then remain the better
design, which is the choice Pareto made.

## 7. Risks and gates

| risk | evidence | gate |
|---|---|---|
| A router that credits itself with gains that come from tier choice | routers were near-constant, and "Always-Mid" matched them [32] [R] | every routing evaluation includes **always-Bonsai-medium**, **always-max** and **random with matched tier composition** as arms, compared pairwise by exact McNemar (PROTOCOL rule 6) |
| Underpowered comparisons | about 85 paired tasks for a 15-point gain at ~25% discordance, about 130 with Bonferroni correction [M, arithmetic, `SELECTION-BUILD.md` §5]; LiveBench n=21 self-discordance 38% [M] | state the power before the run (rule 4); per-class profiles use the uncontaminated 120, never n=21 alone |
| Latency variance from swaps and escalation | a swap plus re-prefill is ~90 s (§5.1); a slow answer is not an outage (AGENTS.md) | report p50/p90 wall time per class and per arm; never swap mid-conversation |
| A silent fallback hiding a dead model | PROTOCOL rule 2 | the route's outcome goes in `x_yamadori`, including `unavailable` |
| Two consumers on one card | AGENTS.md "one GPU consumer at a time" | profiling runs are queued through `bench/queue_runner.py`; the A4000's specialist is not benchmarked while image generation runs |
| Mixing slates | Pareto's own card warns against combining scores with costs from another slate [9] | a profile cell holds numbers from **one** run id; costs and scores come from the same rows |
| Contamination | three.js and type-challenges are marked contaminated | never headline; the leak preflight runs on any new model's prompts too |
| Model drift | Pareto's composition can change silently [8] | version is part of identity, as for packages: a new GGUF sha256 is a new registry row, and it serves in shadow until re-profiled |
| Routing overhead | `route.py` is pure and deterministic; Laya `/route` has a 66 ms median [M, n=120] | no LLM in the routing path; the Laya head stays a second signal only |

## 8. Prioritised next steps

Each step has its measurement and its stop condition. Steps 1 and 2 need no
GPU.

1. **Verifier recall audit (CPU, existing rows).** For every domain-suite
   and LiveBench row that has a hidden-test verdict, re-run `check_code` on
   the delivered answer and tabulate verifier-fail against test-fail.
   **Measure** recall and precision, with Wilson intervals, per class and
   language, and compute the §6 ceiling *r × f × fail rate*. **Stop** building
   verifier-gated escalation if the ceiling is under the paired-test minimum
   detectable effect at n=120.
2. **Registry v0 from existing data (CPU).** Fill Bonsai's row: per-class
   pass rate from the uncontaminated domain rows and the LiveBench
   categories, decode and prefill tokens/s, and Wh per task from the
   `power.py` samples kept for those runs. Every cell names its run id and n.
   **Measure** the noise floor by repeating one bare arm: a result that does
   not survive the repeat is not entered. This is the baseline any other
   model must beat.
3. **Switching and reuse costs (GPU, one consumer, maintenance window).**
   (a) Cold and warm llama-swap load time of an A4000 model, n ≥ 5. (b)
   Prefill tokens/s at 8k, 32k and 64k on both cards. (c) The slot-restore
   experiment of §5.5: B's prompt seconds with and without restoring A's
   prefix, paired on ≥ 20 fan-out requests, with fixed-seed identical output
   as the correctness check. **Decides** whether per-conversation routing and
   prefix-sharing fan-out are worth code.
4. **Router baselines as a standing gate.** Add the arms
   always-Bonsai-medium, always-max and random-matched to the evaluation of
   the Phase 0 router (`route.py`) and of the tier ladder. **Measure** exact
   McNemar of the routed arm against each, on the uncontaminated 120.
   **Stop** crediting routing with any gain that "always-max" matches.
5. **Cross-card parallel B, feasibility first (A4000, image generation
   unloaded).** Load a Bonsai text copy with retrieval and Laya resident, at
   the smallest context a B candidate needs. **Measure** peak VRAM under a
   long prefill (the ≥ 1 GB free floor), decode tokens/s, and fan-out wall
   time for parallel against sequential on ≥ 20 code requests. Also measure
   whether B's tool calls survive (it must not evict retrieval). **Stop** if
   free VRAM at peak drops below the floor. A marginal fit is not shipped, as
   the vision entry already records.

Deferred until 1–3 report: any second chat model in the registry, the
`utility` side-call model, and the different-weights critic.

---

## Sources

Read 2026-09-24. X posts were read through search-index snippets, not
fetched.

1. OpenRouter on X, launch post: https://x.com/OpenRouter/status/2100235351575191751
2. Cloudflare Developers on X, "our first ever stealth model": https://x.com/CloudflareDev/status/2100333813385646534
3. Cloudflare AI docs, Union Alpha (stealth): https://developers.cloudflare.com/ai/models/stealth/union-alpha/
4. OpenRouter on X, reveal post: https://x.com/OpenRouter/status/2100727672074551676 ; Union Alpha on X, "Union Alpha is Pareto 26.9": https://x.com/unionalphaai/status/2100722366200557603 ; Alex Atallah on X, DeepSWE post: https://x.com/alexatallah/status/2100243109401469062
5. CellCog, "Union Alpha Revealed": https://cellcog.ai/blog/what-is-union-alpha/
6. Kingy AI, specs: https://kingy.ai/news/union-alpha-ai-model/
7. Cloudflare AI docs, Pareto (unbiased): https://developers.cloudflare.com/ai/models/unbiased/pareto/
8. Unbiased, "How a Pareto answer gets made": https://unbiased.ai/how/
9. Unbiased, Pareto 26.9 model card: https://unbiased.ai/model-card/
10. Superpower Daily, preview goes paid: https://superpowerdaily.com/posts/openrouter-s-anonymous-union-alpha-model-goes-paid-after-free-preview-overwhelms-capacity
11. GIGAZINE: https://gigazine.net/gsc_news/en/20260918-union-alpha/
12. MindStudio, hands-on tests: https://www.mindstudio.ai/blog/union-alpha-stealth-model-benchmarks
13. Unbiased home page: https://unbiased.ai/
14. LLM Rumors, "Union Alpha Was Pareto 26.9": https://www.llmrumors.com/news/union-alpha-after-the-mystery-evaluation
15. OrcaRouter, "Unbiased's Pareto": https://www.orcarouter.ai/blog/unbiased-pareto-release
16. AI×AI (unsourced architecture claims): https://aiintoai.vercel.app/article/union-alpha-stealth-ai-model-launches-coding-benchmarks-deep-report
17. Chen, Zaharia & Zou, FrugalGPT: https://arxiv.org/abs/2305.05176
18. Ong et al., RouteLLM: https://arxiv.org/abs/2406.18665
19. Aggarwal et al., AutoMix: https://arxiv.org/abs/2310.12963
20. Ding et al., Hybrid LLM: https://arxiv.org/abs/2404.14618
21. Not Diamond: https://www.notdiamond.ai/ and https://docs.notdiamond.ai/docs/what-is-model-routing
22. Martian model router: https://route.withmartian.com/ ; TechCrunch: https://techcrunch.com/2023/11/15/martians-tool-automatically-switches-between-llms-to-reduce-costs/
23. OpenRouter Auto Router docs: https://openrouter.ai/docs/guides/routing/routers/auto-router
24. Arch-Router: https://arxiv.org/abs/2506.16655
25. Wang et al., Mixture-of-Agents: https://arxiv.org/abs/2406.04692
26. Together AI, Together MoA: https://www.together.ai/blog/together-moa
27. Li et al., "Rethinking Mixture-of-Agents" (Self-MoA): https://arxiv.org/abs/2502.00674
28. Narasimhan et al., "Faster Cascades via Speculative Decoding" (ICLR 2025): https://arxiv.org/abs/2405.19261
29. Cobbe et al., training verifiers: https://arxiv.org/abs/2110.14168
30. Lightman et al., "Let's Verify Step by Step": https://arxiv.org/abs/2305.20050
31. Brown et al., "Large Language Monkeys": https://arxiv.org/abs/2407.21787
32. "Task- and Session-Level Model Routing: A Common-Interface Hybrid Evaluation of Four Open-Source Routers": https://arxiv.org/abs/2608.14641
33. Survey, dynamic routing and cascading: https://arxiv.org/abs/2603.04445

In-repo: `AGENTS.md`, `config.yaml` (card inventory, VRAM notes, groups),
`mcp/fanout.py`, `mcp/slots.py`, `mcp/route.py`, `mcp/budget.py`,
`mcp/admission.py`, `mcp/power.py`, `docs/MTP-STAGING.md`,
`docs/KNOWN-ISSUES.md`, `docs/CONSTRAINTS.md`,
`docs/STACK-TUNING-2026-09-23.md`, `docs/DECISION-TREES-AND-SKILLS.md`,
`docs/SELF-IMPROVEMENT-PLAN.md`, `docs/PROTOCOL.md`.
