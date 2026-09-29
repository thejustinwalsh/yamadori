# Skills, retrieval, placement, deciders and thinking budgets: what the research says

Research note, 2026-09-27. This is web research only. No code changed, no GPU ran, and nothing was sent to `:1234` or `:11434`. Two local artefacts were read for comparison: `bench/decider/results/bonsai.json` and the skill code.

The operator asked for "the research on skills and models and qwen + skills, and not just make it all up ... not just Anthropic". arXiv is the primary source. Vendor docs, model cards and blogs are secondary and labelled **[V]**.

## How to read the citations

Every source was opened during this pass. Nothing is cited from memory. Each citation carries a label for how much of the source was read:

| label | what was read |
|---|---|
| **[P]** | Full paper. The arXiv PDF text was extracted locally (PyMuPDF), and the tables and sections named were read from that text. |
| **[H]** | The arXiv HTML full text, read through the fetch tool's summarising model. Numbers came back through that model. They were not checked character by character, and the summariser invented one claim that was caught (see Discrepancies). |
| **[A]** | The arXiv abstract page only. |
| **[V]** | Vendor doc, model card, blog or README. Secondary. |
| **[S]** | Seen only in a search snippet. **Not verified.** Never relied on below. |

Sources marked **(re-checked)** were fetched a second time for this document. That second fetch confirmed the abstract, the version history and the specific numbers quoted:

- SkillsBench 2602.12670
- SRA 2604.24594
- Skill Shadowing 2605.24050
- Skills in the Wild 2604.04323
- Hybrid recall 2609.04434
- Cuadron 2502.08235 (the function-calling quote)
- Vercel's blog
- The Qwen3.6-27B and Qwen3.8-27B cards
- The Bonsai 2 card and PrismML's Ternary Bonsai page

"Inference" marks a conclusion this note draws that no single source states.

---

## The biggest takeaways (one screen)

**1. Few, compact, curated skills help; more and longer skills help less or hurt.**
- SkillsBench (2602.12670v4 [P], 87 tasks, 18 model-harness configurations, 3 trials): curated skills raise the pass rate from 33.9% to 50.5%.
  - By number of skills: 1 skill +18.0 pp, 2-3 skills +19.0 pp, 4 or more +10.1 pp.
  - By skill length: compact +19.0, standard +21.5, detailed +14.5, "comprehensive" +0.7 pp.
- ReasoningBank: one retrieved item is best (49.7), and four fall to 44.4.
- Skill Shadowing: the pass rate falls 0.21 at a 202-skill library, and the cause is picking the WRONG skill, not context length.

**2. Self-written skills hurt unless they are verified.**
- SkillsBench: self-generated skills cost −8.1 to −11.5 pp.
- ASI: verification alone added +4.2.
- Dynamic Cheatsheet: small models fill memory with flawed strategies.

**3. For open and smaller models, pushed context beats model-pulled context.**
- SRA (2604.24594v3 [P]; Qwen3-4B/32B/235B, Llama, Mistral): models load skills at about the same rate whether or not the gold skill was retrieved, and whether or not the task needs one. On Qwen3-32B, progressive disclosure scores 55.3, against 62.4 for LLM selection from the top 50 and 67.2 with the oracle skill.
- Skills in the Wild (2604.04323 [P]), Qwen3.5-397B: force-loaded 41.2, agent-chosen 31.6.
- Vercel [V]: the skill was never invoked in 56% of cases.
- MemTool: non-reasoning models manage their own tools poorly; workflow mode does above 90%.

**4. Put the live instruction last; restating helps.**
- Lost in the Middle: a U-shaped curve, and the worst position scores below the closed-book baseline.
- Laban et al.: restating ("Recap") recovers 50.4 → 66.5.
- Li et al.: attention to the system prompt decays across turns.
- Prompt repetition [H]: 47/70 wins for non-reasoning models, but neutral with reasoning (5 wins, 1 loss, 22 neutral).
- No position study exists for Gated-DeltaNet hybrids. In Qwen3.5-27B, exact recall runs through the attention layers, and style/mode through the recurrent state (2609.04434 [H]).

**5. Raw label-token probabilities are biased, and our CLM run showed it.**
- Calibrate Before Use: majority, recency and common-token biases, and up to +30 points from content-free calibration.
- PriDe: LLMs prefer some option IDs (llama-30B picks A 34.6%, D 15.8%).
- A 2026 yes/no study: the bias follows the last-printed option and the word "no".
- Quantization moves confidence most on uncertain items, and calibration does not transfer across precisions.
- Our own run: CLM said "yes" on 99-119 of 124 phase points. Bonsai's p(yes) for "Go ahead." ranged 0.18-0.42 across three wordings of one question (`bench/decider/results/bonsai.json`).

**6. Thinking: past ~6-8k tokens the gains stop and flips turn negative; overthinking hurts agents.**
- R1-32B-class math: accuracy saturates around 8-12k thinking tokens; answer flips turn net-negative from about 7k; stopping at about 6k costs about 6% accuracy for about 50% of the compute (2604.10739 [H]).
- Agentic SWE: overthinking correlates negatively with resolution, and smaller models overthink more (2502.08235 [H]).
- Aggressive low-bit quantization lengthens outputs (2504.04823 [H]).
- Qwen's own cards recommend 32,768+ output tokens, but for benchmark maxima, not agent steps.

**7. Our regime (27B, Qwen3.8 hybrid, ternary PTQ) is almost unmeasured in public.**
- The Bonsai 2 card [V] claims 98.2% of FP16 across 14 thinking benchmarks.
- Academic ternary PTQ on Qwen3 4-14B loses 16-30 points on math and code (2608.01078 [H]).
- 4-bit models keep single tool calls (−1 to −3%) but lose 10-15% on real agent tasks (2505.19433 [H]).
- 4-bit models lose up to 23% at 128K context (2505.20276 [H]).
- **No public study measures ternary models on agent, tool or long-context work. Our own measurements are the only evidence for our regime.**

**8. Qwen's own base model keeps its thinking across turns by default.**
- The Qwen3.8-27B card [V] (Bonsai 2's base per its card): `preserve_thinking` is "enabled by default", citing consistency and "improved KV cache utilization".
- Qwen3.6-27B [V]: keeps only the latest user message's thinking by default, and was "additionally trained" to use historical thinking when asked.
- This bore on our "past reasoning passes through" design (below). The operator reversed it on 2026-09-27: past reasoning is restored (recommendation 9).

---

# Part 1. Our regime: ~27B, Qwen-family, low-bit, hybrid

The operator asked for this section first. Where it conflicts with frontier-model findings, it wins in "What this means for Yamadori".

## 1.1 What Bonsai 2 is

- **The Bonsai 2 card** [V] (huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf; the repo `models/manifest.yaml` pins it at `6ed5e12b`; re-checked).
  - Derived from "Qwen3.8-27B, a 27B hybrid-attention causal language model".
  - Weights: "Ternary g128: {−1, 0, +1} weights with FP16 group-wise scaling". PTQ1_0 packs "dense trits at 1.75 bits/weight".
  - Architecture: "Hybrid attention (~75% linear / ~25% full attention)", 262K context.
  - Claims "84.78 average across 14 thinking-mode benchmarks", "98.2% of FP16 intelligence retained".
  - Recommended thinking sampling: T=1.0, top_p 0.95, top_k 20, min_p 0.05.
  - **Vendor claim, method not disclosed.** The card does not say whether this is plain PTQ or distillation/QAT. PrismML's April 2026 Ternary Bonsai page (8B/4B/1.7B) does not disclose it either.
  - **No PrismML arXiv paper was found.**
- **The Qwen3.8-27B card** [V] (re-checked).
  - 64 layers: "16 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))", so 16 attention layers.
  - Context: 262,144 native, extensible to 1M.
  - `reasoning_effort`: xhigh (default), medium, low. That is exactly the set AGENTS.md says the served template accepts.
  - Thinking sampling: T=1.0, top_p 0.95, top_k 20, min_p 0, presence_penalty 0.
  - `preserve_thinking`: "enabled by default for all workloads".

## 1.2 Low-bit quantization: what degrades first

| source | setup | finding |
|---|---|---|
| **Quantization Hurts Reasoning?** Liu et al., COLM 2025, arXiv 2504.04823v2 [H] | R1-Distill-Qwen 1.5-32B, QwQ-32B, Qwen3-8B. GPTQ/AWQ, KV and weight-activation quantization. 5 reasoning benchmarks, 3 seeds. | W4G128 loses 2.1% (1.5B) and 0.4% (32B). W3G128 loses over 7% and 3%. KV8 with W8A8 loses less than 1 point. Harder tasks lose more (32B W4A4KV4: AIME −3.9, GSM8K 0.0). "aggressive low-bit quantization can lead to increased output lengths, particularly in smaller models". |
| **ACBench**, "Can Compressed LLMs Truly Act?" Dong et al., ICML 2025, 2505.19433v2 [H] | 15 models of 2-32B (Qwen2.5 among them). 4/8-bit GPTQ/AWQ/SmoothQuant, pruning. Temperature 0, no seeds. | "4-bit quantization preserves workflow generation and tool use (1%–3% drop) but degrades real-world application accuracy by 10%–15%". JSON-format output degrades more than string format. Needle retrieval at 40K degrades. Nothing below 4 bits. |
| **Long-context under quantization**, Mekala et al., EMNLP 2025, 2505.20276v3 [H] | Llama-3.1 8B/70B, Qwen2.5 7B/32B/72B. 9.7K examples, 8K-128K context. | 8-bit loses about 0-0.8%. 4-bit loses 1.8-6.9% on average, "up to 23% across models at 128K". Degradation grows with length. |
| **Quantization trade-offs**, Lee et al., IJCAI 2025, 2409.11055 (v5 [H]) | GPTQ/AWQ/SmoothQuant/FP8, 1B-405B, 13 benchmarks. | Quantized models "often struggle with instruction-following (IFEval) and hallucination detection". IFEval for Llama-3.1-8B: 50.09 → 47.95 (GPTQ). |
| **An Empirical Study of Qwen3 Quantization**, Zheng et al., 2505.02214v1 [H] | Qwen3 at 1-8 bits. | "Qwen3 exhibits more pronounced performance degradation under low-bit quantization (3 bits or fewer)" than LLaMA3. At 2-3 bits, MMLU often lands in the mid-20s. |
| **Low-bit favors undertrained LLMs**, Ouyang et al., 2411.17691v2 [A] | 1,500+ checkpoints. | Quantization hurts fully trained models more. Heavily trained Qwen weights are the hard case (inference). |
| **ScaleQ-1.58**, "Attend to Your Own Thoughts ... 1.58-Bit", Wang et al., 2608.01078 [H] | W1.58A16 PTQ on Qwen3 1.7B-235B, calibrated on the model's own chain-of-thought. | Average of Math-500, GSM8K, HumanEval+ and MBPP+: Qwen3-4B 52.27 vs 82.59 FP16; 8B 58.58 vs 84.00; 14B 70.61 vs 86.56. The gap shrinks with size. Calibrating on its own reasoning is the lever. No output-length analysis. |
| **PTQTP (trit-planes)**, Xiao et al., 2509.16989 (v2 [H]) | LLaMA3.x and Qwen3, 0.6-70B. | Naive 2-3-bit GPTQ and BiLLM reach "0% accuracy on Math-500". Math degrades about 19× more than linguistic tasks. |
| **Post-training ternarization to Qwen3-8B**, Malik et al., 2609.09240v1 [H] | About 1.64 bits/weight, 8 zero-shot tasks, n=500 each, 1 seed. | Retention 78.5%. Knowledge suffers most (MMLU 71.2% retention). Context-reading holds best (LAMBADA 90.7%, BoolQ 89.7%). |
| **BitNet b1.58 2B4T**, Ma et al., 2504.12285v2 [H] | Native ternary 2B vs Qwen2.5-1.5B under int4 PTQ. | int4 PTQ cost Qwen2.5-1.5B IFEval 50.12 → 45.44 and GSM8K 56.79 → 50.57. "Standard PTQ techniques lead to a noticeable degradation". |
| **When Quantization Affects Confidence**, Proskurina et al., Findings NAACL 2024, 2405.00632v1 [H] | GPTQ 4-bit on Mistral-7B, LLaMA-7B and others, 6 multiple-choice tasks. | Confidence in true labels decreases, concentrated on samples where full precision was already unsure. ECE worsens by up to about 3 pp for some models. |
| **Quantisation Reshapes the Metacognitive Geometry**, Cacioli, 2604.08976v1 [A] | Llama-3-8B, Q5_K_M vs f16, 3,000 questions. | Domain-level confidence profiles do not correlate across precisions (rho = 0.00). Type-2 AUROC profiles are stable (rho = 1.00). Ranking survives; calibrated levels do not. |

**What this implies for our regime (inference).**
- Every public ternary-PTQ result on Qwen3 loses the most on math, code and multi-step reasoning, less on context-reading. At 4 bits, instruction following, long-context retrieval and multi-step agent work degrade before single tool calls do.
- The vendor's 98.2% retention for Bonsai 2 is not replicated anywhere public.
- No public study measures a ternary model on agent, tool, long-context or skill-use tasks.

## 1.3 The hybrid architecture (Gated DeltaNet + attention)

| source | finding |
|---|---|
| **What Attention Recalls and Recurrence Controls in Hybrid Language Models**, Afendulev et al., Findings EMNLP 2026, 2609.04434v1 [H] (abstract re-checked) | Tested on **Qwen3.5-27B** (1 seed at 27B), 4B/9B and Falcon-H1. "Exact retrieval survives only through attention (64-98% of full accuracy) and collapses to zero through recurrence." On Qwen3.5-27B: KV-retrieve 1.00 full, 0.00 recurrent-only, 0.99 KV-only. Language-following 0.99 full, 0.83 recurrent-only, 0.08 KV-only. In a state swap, "the answer takes its value from the KV side and its language from the recurrent side". **Position was not studied.** |
| **Why Gated DeltaNet Survives 4-Bit Quantization**, Kozyrev and Maiboroda, 2609.04098v1 [H] | On "Qwen3.8-27B", NVFP4 W4A4 on all 496 linear layers, FP8 KV, 4 seeds. RULER 32K/64K 100% vs 100%. Average gap −0.52, within noise. The recurrent-state error plateaus (12.96% at token 256, 12.31% at 32,768): "the delta rule overwrites the state along the current key direction". |
| **Gated Delta Networks**, Yang et al., 2412.06464v3 [H] | At 1.3B/100B tokens, pure GDN recalls well below Transformer++ (average of six recall tasks 30.6 vs 37.0). Hybrids reach 39.0-40.1. |
| **An Empirical Study of Mamba-based LMs**, Waleffe et al. (NVIDIA), 2406.07887v1 [H] | 8B hybrid: phonebook recall is perfect to 128K, but multi-document QA is lower than the Transformer: "SSM layer states are sometimes confused by documents irrelevant to the question". |
| **Jamba**, Lieber et al., 2403.19887v2 [H] | Pure Mamba "often does not follow the correct format" in in-context learning (IMDB 48.8 vs attention 84.1). The hybrid reaches 90.9. |
| **Repeat After Me**, Jelassi et al., 2402.01032v2 [A]; **Zoology**, Arora et al., 2312.04927v1 [A] | A fixed-size state bounds copying and associative recall. Attention closes the gap. |
| **Stuffed Mamba**, Chen et al., COLM 2025, 2410.07145v4 [H] | Recurrent models with oversized states fail to forget. "high retention of the first token" after 8K. |

**What this implies (inference).**
- A skill's exact facts (API names, signatures) are recalled through the 16 attention layers.
- Mode and style instructions ride the recurrent state, which is recency-weighted by decay. That plausibly favours putting procedural text late.
- Irrelevant injected text is the specific risk for the recurrent half.
- **No study measures lost-in-the-middle or needle depth on a GDN hybrid.**

## 1.4 The KV cache at q8_0

- **KV8 evidence.**
  - 2504.04823v2 [H]: W8A8 with KV8 loses less than 1 point, 3 seeds.
  - 2609.04098v1 [H]: FP8 KV on the hybrid 27B holds RULER at 100%.
  - "KV Cache Compression, But What Must We Give in Return?", Yuan et al., 2407.01527v2 [H]: KIVI-4bit on Llama-3-8B scores LongBench 45.3 vs 45.2 baseline.
  - KIVI 2402.02750v2 [A] and KVQuant 2401.18079v6 [A]: near-lossless at 2-3 bits by their measures.
- **No study of llama.cpp q8_0 specifically was found.** Inference: q8_0 KV is very likely near-free, since 4-bit KV already is.

## 1.5 MTP / speculative decoding

- Leviathan et al., ICML 2023, 2211.17192v2 [A]: speculative decoding makes "no changes to the outputs"; the output distribution is preserved.
- "Lossless but Not Free", Chordiya, 2607.17283v1 [H]:
  - Distributions were equivalent (χ² p = 0.976).
  - But "quantized-Metal logits are not batch-invariant". Near-tied greedy tokens can flip between batch sizes.
  - Inference for us: the MTP verify batch can flip near-ties, so the decider's argmax on near-equal options is not bit-reproducible across batch shapes.

## 1.6 Open 7-32B models with skills, memory and tools (where they differ from frontier)

| source | open-model evidence |
|---|---|
| **SRA**, "Skill Retrieval Augmentation for Agentic AI", Su et al., 2604.24594v3 [P] (re-checked) | Table 2 average accuracy (direct / oracle / top-1 full injection / LLM selection from top-50 / progressive disclosure): **Qwen3-4B** 38.8 / 61.6 / 45.5 / 53.3 / 43.2; **Llama-3.1-8B** 29.8 / 44.5 / 32.7 / 37.0 / 36.3; **Qwen3-32B** 50.8 / 67.2 / 54.3 / 62.4 / 55.3; **Qwen3-235B** 53.3 / 67.5 / 56.2 / 62.8 / 59.9. Abstract: agents "load skills at similar rates, regardless of whether a gold skill is retrieved or whether the task actually requires external capabilities". Distractors hurt, and full injection is the most brittle. 5,400 instances, BM25 retriever, temperature 0.7. |
| **Skills in the Wild**, Liu et al., 2604.04323v1 [P] (abstract re-checked) | 84 SkillsBench tasks × 3 runs. **Qwen3.5-397B-A17B** in Qwen-Code: curated force-loaded 41.2; curated, agent chooses 31.6; with distractors 33.7; retrieved with curated in the pool 26.7; retrieved with curated removed 19.7; no skills 20.5. Kimi K2.5 follows the same shape (19.8 vs 21.8 baseline). "weaker models are more likely to be hurt by low-quality retrieved skills." |
| **Memp**, Fang et al., 2508.06433v4 [P] | Qwen2.5-72B on ALFWorld: 41.25 → 77.19 with procedural memory. GPT-4o's memory handed to Qwen2.5-14B raised TravelPlanner completion 5%. Too many retrieved memories declines. Single runs. |
| **ReasoningBank**, Ouyang et al., 2509.25140v2 [P] | Gemma-3-12B on Shopping: none 17.1, AWM 21.4, ReasoningBank 24.1. |
| **Less is More (edge function calling)**, Paramanayakam et al., DATE 2025, 2411.15399v1 [H] | Qwen2-7B, Llama3.1-8B, Hermes2-Pro-8B, etc., at q4-q8, 230 queries each. Offering fewer tools raised success "up to 71%" and cut time up to 80%. |
| **Tool Calling is Linearly Readable**, 2605.07990v1 [H] | Qwen3/2.5 0.6-14B, Gemma 3, Llama 3.1. Gemma 3 4B's tool representations blur at K=750 tools with descriptions (about 16K tokens), but not at K=1,500 without them. Prompt length, not count, drives the collapse. |
| **Procedural Memory Under Change**, Cao, 2609.09774v1 [P] | qwen3:8b locally: mismatched procedures did not interfere when the task's own evidence was explicit. n is tiny; very weak. |
| **SkillOpt** 2605.23904v2 [A], **SKILLER** 2608.10538v2 [A] | Abstracts claim the largest relative gains for small models. The per-Qwen numbers from the bodies were read inconsistently or only in snippets: **NOT VERIFIED**. |

**Where open models differ (inference across the rows above).**
- Open models gain from good skills as much as frontier models (SRA oracle +14 to +23).
- They are worse at deciding when to load one (SRA).
- They are hurt more by bad or distractor skills (Wild).
- They degrade with long tool and skill prompts (edge, linear-readability).

---

# Part 2. The research by topic

## 2.1 Skill and procedural-memory libraries

**The classics**

- **Voyager**, Wang et al., 2305.16291v2 [H], TMLR 2023.
  - Skills are executable code, indexed by an embedding (ada-002) of their description. Top-5 retrieval by the embedding of the plan plus feedback.
  - Minecraft, GPT-4, 3 trials: 63 unique items (3.3× baselines). Without the skill library, progress plateaus late.
  - Zero-shot transfer: AutoGPT plus Voyager's library solved tasks AutoGPT alone never did.
  - Weak to moderate evidence: 3 trials, one game.
- **ExpeL**, Zhao et al., 2308.10144v3 [H].
  - Insights with ADD/EDIT/UPVOTE/DOWNVOTE, plus top-k retrieval of successful trajectories. GPT-3.5 actor, GPT-4 extractor.
  - ALFWorld: ReAct 40, insights only 50, retrieval only 55, both 59 (read from a figure).
  - Retrieval by task similarity 59.0, by reasoning similarity 48.5, random 42.5.
  - Insight source: GPT-4-written 39.0, hand-crafted 32.0.
- **Agent Workflow Memory (AWM)**, Wang et al., 2409.07429v1 [H].
  - Natural-language workflows induced from LM-judged successes and added to memory.
  - WebArena: 35.5% vs BrowserGym 23.5%.
  - Rule-based and LM induction score about the same (35.6 vs 35.5). Text vs code format differ by 0.3-0.6 points.
  - No variance reported.
- **SkillWeaver**, Zheng et al., 2504.07079v1 [P].
  - Synthesised Playwright APIs. WebArena: GPT-4o 22.6 → 29.8; GPT-4o-mini with GPT-4o's skills 9.2 → 14.1.
  - Picking the wrong API and passing wrong parameters are the failures, "even greater for weaker LLMs".
  - An API-selection module filters to the relevant APIs and their preconditions.
- **ASI**, "Inducing Programmatic Skills for Agentic Tasks", Wang et al., COLM 2025, 2504.06821v2 [P].
  - A skill is admitted only if re-execution passes: the task is judged solved, a new skill was called, and every call changed the environment.
  - WebArena with Claude-3.5: vanilla 32.7, AWM 36.3, ASI 40.4.
  - Verification adds +4.2 (unverified text 32.6 vs verified text 39.0 in memory).
  - Only 15.6% of induction turns passed verification.
- **CRADLE** 2403.03186v3 [H] and **JARVIS-1** 2311.05997v3 [H]: code skills or key→plan memories retrieved by embedding similarity. Summariser-only; weak.

**The SKILL.md era (2026)**

- **SkillsBench**, Li et al., 2602.12670v4 [P] (re-checked). The strongest source.
  - Setup: 87 tasks in 8 domains, deterministic verifiers, curated skills written independently of the benchmark, progressive disclosure through the harness's `skills/` directory. 18 configurations, 3 trials, Wald CIs.
  - Main result: 33.9% → 50.5%. Every configuration gains, from +4.1 (Gemini 3.1 Flash Lite) to +25.7.
  - Software engineering gains the least: +11.6 on 16 tasks.
  - Number of skills: 1 skill +18.0 (23 tasks), 2-3 skills +19.0 (43), 4+ +10.1 (21). Different task sets per bucket, so confounded.
  - Length: compact +19.0, standard +21.5, detailed +14.5, comprehensive +0.7 (5 tasks).
  - Self-generated skills: −8.1 (Claude Code + Opus 4.7), −11.3 (Codex + GPT-5.5), −11.5 (Gemini CLI + 3.1 Pro).
    - In 10 of 12 audited runs the solver never read the packs.
    - Packs that were read carried "confidently wrong" content.
  - 13 of 87 tasks got worse. The causes: a heavyweight pipeline displacing a simple path, a skill displacing a stronger default, a tool the agent cannot debug. The authors recommend applicability boundaries and fallbacks.
  - The median public SKILL.md is about 1.2k tokens.
  - No small open model is in the main table.
- **More Skills, Worse Agents? Skill Shadowing**, Song and Wei, 2605.24050v2 [P] (abstract re-checked).
  - Oracle-only libraries vs libraries of 52/102/202 skills, 2,545 trajectories.
  - The pass rate drops 0.08, 0.14 and 0.21.
  - Shadowing (wrong selection) accounts for 0.14; context overhead is "indistinguishable from zero".
  - The oracle-skill invocation share falls from 88.0% to 52.6%.
- **The Scaling Laws of Skills in LLM Agent Systems**, Chen et al., 2605.16508v1 [A plus the introduction].
  - 15 LLMs, 1,141 skills. Routing accuracy falls as a − b·ln N (R² > 0.97).
  - Generic "black-hole" skills capture requests.
  - Fixing the library lifted routing from 71.3% to 91.7%, mainly by boundary rewriting (+12.8) and removing abstract skills (+4.9).
- **When Single-Agent with Skills Replace Multi-Agent Systems**, Li, 2601.04748v2 [P].
  - Selection stays above 90% up to about 20 skills and degrades past about 30.
  - Confusability matters more than size: one similar competitor per skill costs 7-30%.
  - GPT-4o/4o-mini, synthetic, 3 seeds.
- **SRA** and **Skills in the Wild**: see 1.6.
  - Wild also finds query-specific refinement recovers some loss (Opus 40.1 → 48.2), but not when the good skill is absent.
- **What Keeps Agent Skills from Being Reusable?** 2608.08453v1 [A]: 91.8% of 138K public SKILL.md files have at least one defect. Valid routing metadata improves retrieval from descriptions.

**Procedural memory and context engineering**

- **Memp** 2508.06433v4 [P]: see 1.6. On TravelPlanner, GPT-4o's hard-constraint score fell with memory (12.88 → 5.50).
- **ReasoningBank**, Ouyang et al., 2509.25140v2 [P].
  - Memory items are injected into the system instruction.
  - WebArena, Gemini-2.5-flash: 40.5 → 48.8.
  - By number of items retrieved: 0 → 39.0, 1 → 49.7, 2 → 46.0, 3 → 45.5, 4 → 44.4.
  - AWM falls below no-memory with Claude (40.8 vs 41.7).
- **ACE**, "Agentic Context Engineering", Zhang et al., ICLR 2026, 2510.04618v3 [P].
  - Incremental bullet deltas instead of rewrites.
  - "Context collapse": a Dynamic Cheatsheet rewrite from 18,282 to 122 tokens fell below no-context (57.1 vs 63.7).
  - AppWorld: 42.4 → 59.4.
  - Without reliable feedback, the playbook "can be polluted".
- **Dynamic Cheatsheet**, Suzgun et al., 2504.07952v1 [P].
  - Large models gain a lot.
  - GPT-4o-mini is flat or lower (AIME 16.7 → 13.3): small models fill memory "with flawed or incomplete strategies".

## 2.2 Tool, skill and document retrieval

**Classic tool retrieval**

- **ToolLLM**, Qin et al., 2307.16789v2 [H].
  - A trained Sentence-BERT retriever: NDCG@5 84.9 vs ada 45.4 vs BM25 17.0.
  - Top-5 retrieved APIs gave a pass rate of 67.3% vs 66.7% with the oracle APIs. In-domain only.
- **Gorilla**, Patil et al., 2305.15334v1 [P].
  - "augmenting a LLM with retrieval, does not always lead to improved performance, and can at-times hurt".
  - Gorilla on HuggingFace: 71.68 zero-shot, 17.03 with BM25 documents, 91.26 with the oracle document.
  - "when a good retriever is not available, zero-shot finetuning might be the preferred choice".
- **AnyTool**, Du et al., 2402.04253v1 [H]: hierarchical LLM selection (category → tool → API) with no embeddings. 58.2% vs ToolLLM 22.9%.
- **ToolRet**, "Retrieval Models Aren't Tool-Savvy", Shi et al., Findings ACL 2025, 2503.01763v2 [P].
  - 7,615 tasks, 43,215 tools.
  - "all the IR model achieves better performance when an additional instruction is paired with the query"; instruction-tuned embedders gain most.
  - Reranking gives "limited and even negative improvements": MonoT5 over NV-Embed dropped 33.83 → 28.92.
  - Retrieved tools cut GPT-3.5's pass rate by 11.40 vs the oracle.

**2026 skill retrieval**

- **SkillRet**, Cho et al., 2605.05726v1 [P].
  - 17,810 skills, nDCG@10.
  - Scores: BM25 48.86; **Qwen3-Embedding-0.6B 58.35**; Qwen3-Embedding-8B 59.98; fine-tuned 0.6B 78.03.
  - They replaced Qwen3's default web-search instruction with an authored skill-retrieval instruction.
  - **Indexing name + description + BODY beats name + description by 1.5-11.4 points** (8B: 48.6 → 60.0).
  - Rerankers: Qwen3-Reranker helps an off-the-shelf first stage (59.98 → 64.0-68.9 as k grows) and hurts a fine-tuned one (78.03 → 75.8-76.2).
- **Skill Is Not Document (R3-Skill)**, Wang et al., 2606.03565v4 [P].
  - 10,246 skills. Hit@1: Qwen3-Emb-0.6B 0.431, 8B 0.623, fine-tuned 0.721.
  - An off-the-shelf Qwen3-Reranker lowered Hit@1 over the fine-tuned embedder.
- **Right Family, Wrong Skill**, Ding et al., 2606.10388v3 [P, abstract and conclusion].
  - "95.0–95.7% of helpful top-three hits also contain the risky sibling."

**Query and document expansion**

- **Doc2query**, Nogueira et al., 1904.08375v2 [P]: 10 predicted queries per document with BM25 lift MS MARCO MRR@10 from 18.4 to 21.5.
- **Doc2Query--**, Gospodinov et al., ECIR 2023, 2301.03266v3 [A]: filtering hallucinated queries gives "up to 16%" better effectiveness.
- **Re-Invoke**, Chen et al. (Google), 2408.01875v2 [H+A]: 10 synthetic queries per tool plus intent extraction give "20% relative improvement in nDCG@5" single-tool and 39% multi-tool, with no training.
- **Tool2Vec**, Moon et al., 2409.02141v1 [H]: a tool embedding is the mean of its usage-query embeddings. Recall@3 91.84 vs 79.97.
- **Tool-DE**, Lu et al., 2510.22670v1 [P].
  - LLM-expanded tool docs (description, tags, when_to_use, limitations, example_usage).
  - Off-the-shelf Qwen3-Embedding gains about 0 (0.6B 43.13 → 43.30).
  - The "example usage" field "provides the smallest (often negative) gains".
  - Gains come only after training on the expanded docs.

**Instruction-tuned asymmetric embedders** [V, model cards]

- **Qwen3-Embedding-0.6B.**
  - Query: `Instruct: {task}\nQuery:{query}` (no space after "Query:"). Documents get no instruction. Last-token pooling.
  - "not using an instruct on the query side can lead to a drop in retrieval performance by approximately 1% to 5%". Write instructions in English.
  - Tech report 2506.05176v3 [H]: the embedding is the last layer's hidden state at [EOS].
- **Qwen3-Reranker.** A yes/no judge: system "Judge whether the Document meets the requirements ... can only be "yes" or "no"", with the score taken as the softmax over the yes/no logits.
- **E5** (`query: `/`passage: `), **multilingual-e5-large-instruct** and **gte-Qwen2-instruct** (`Instruct: {task}\nQuery: {query}`): instructions on queries only.

## 2.3 Qwen's own guidance

- **Qwen3 Technical Report**, 2505.09388v1 [H].
  - A thinking budget: when it is reached, the model is fed "Considering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>.\n\n". The ability "emerges naturally" from thinking-mode fusion.
  - Qwen3-235B improves "scalable and smooth[ly]" with budgets from 1K to 32K on math, code and STEM.
  - For non-thinking mode the model keeps an empty think block.
  - Stage-4 RL trains tool invocation over "complete multi-turn interaction cycles with real environment execution feedback".
- **Qwen3-32B card and template** [V].
  - Thinking sampling: T 0.6, top_p 0.95, top_k 20. "DO NOT use greedy decoding".
  - Output 32,768 tokens, or 38,912 for complex problems.
  - "Historical model output should only include the final output part". The template nonetheless keeps `<think>` for turns after the last real user query, so within a tool loop.
  - Hermes-style tools: `<tools>` in the system message, `<tool_call>{json}</tool_call>`.
- **Qwen docs, function calling** [V].
  - "We recommend using Hermes-style tool use for Qwen3".
  - "it is not recommended to use tool call template based on stopwords, such as ReAct, because the model may output stopwords in the thought section".
- **Qwen-Agent** [V].
  - The nous template is the default and recommended for Qwen3. MCP is supported.
  - Its long-context RAG uses 512-token chunks, BM25 with query decomposition, and chunk-by-chunk relevance reading. It claims its 4K agent "consistently surpasses the 32k-Model and the 4k-RAG" (blog, 2024-06-06; per-benchmark numbers not extracted).
- **Qwen3.6-27B card and template** [V] (re-checked).
  - Same hybrid layout.
  - Tools: "You have access to the following functions:\n\n<tools>". XML calls: `<tool_call><function=...><parameter=...>`. Parser `qwen3_coder`.
  - By default only "the thinking blocks generated in handling the latest user message" are kept. The model was "additionally trained to preserve and leverage thinking traces from historical messages" (`preserve_thinking`).
  - Output 32,768, up to 81,920.
- **Qwen3.8-27B card** [V] (re-checked): see 1.1. `preserve_thinking` is on by default, for consistency and "improved KV cache utilization".
- **Thinking mode can swallow tool calls on Qwen.**
  - GitHub issue QwenLM/Qwen3#1817 [V], Qwen3-32B-AWQ on vLLM, n=5: with thinking on, 2/5 succeeded and 2/5 fabricated results; with `/no_think`, 5/5 succeeded.
  - "Fidelity Is Not Enough", 2608.28439v1 [H] (appendix only): the same interaction on Qwen3.6-27B/vLLM (passes 25/23/19 with reasoning, 25/24/24 without), called a deployment-stack artefact.
  - Anecdotal. It resembles pagoda-h2's "finish=stop, no tool call" and helper d28941fb stopping with no call (inference).

## 2.4 Where injected context goes

**Position and recency**

- **Lost in the Middle**, Liu et al., TACL 2024, 2307.03172v3 [H].
  - NaturalQuestions-Open, 2,655 queries, 10/20/30 documents.
  - U-shaped curve. GPT-3.5's worst position at 20-30 documents falls below closed-book (56.1%).
  - "the 7B Llama-2 models are solely recency biased".
  - Query-aware placement (the query before and after) fixes key-value retrieval but "minimally affects" multi-document QA.
- **Follow-ups** [A]:
  - Found in the Middle 2406.16008v2: a U-shaped attention bias; calibrating it gives up to +15 pp.
  - RULER 2404.06654v3: "only half" of 17 models hold up at 32K.
  - NoLiMa 2502.05167v3: at 32K, 11 of 13 models fall below 50% of their short baseline when the needle shares no words with the question.
- **Context Rot** [V], Chroma, 2025: "even a single distractor reduces performance".
- **LLMs Get Lost In Multi-Turn Conversation**, Laban et al., 2505.06120v1 [H].
  - 15 LLMs, including Llama3.1-8B and OLMo-2-13B, 200k+ simulated conversations.
  - An average drop of 39%, with unreliability up 112%.
  - Recap (restate earlier user turns at the end): GPT-4o-mini sharded 50.4 → 66.5 (full 86.8).
  - "loss-of-middle-turns".

**Distraction**

- **Shi et al.**, ICML 2023, 2302.00093v3 [P].
  - GSM-IC, 58,052 examples. "fewer than 30% of the base problems are consistently solved after adding distractors".
  - Lexical overlap with the problem is what hurts.
  - "Feel free to ignore irrelevant information" and distractor-bearing exemplars help.
- **Long-Context LLMs Meet RAG**, Jin et al., 2410.05983v1 [H].
  - Gemma-2-9B, Mistral-Nemo-12B, Gemini-1.5-Pro.
  - More passages help, then hurt (an inverted U), and hurt more with the stronger retriever's hard negatives.
  - Putting the best passages at the start and end helps.

**Drift, repetition and reminders**

- **Instruction (In)Stability**, Li et al., COLM 2024, 2402.10962v4 [H]: "significant instruction drift within eight rounds". Attention to system-prompt tokens decreases across turns. Re-injecting the system prompt helps "moderately" at a large context cost.
- **Prompt Repetition Improves Non-Reasoning LLMs**, Leviathan et al. (Google), 2512.14982v1 [H]: repeating the prompt `<QUERY><QUERY>` "wins 47 out of 70 tests, with 0 losses". With reasoning: 5 wins, 1 loss, 22 neutral.
- **OpenDev**, 2603.05344v2 [H]: "event-driven system reminders ... injecting targeted guidance at the point of decision". A design rationale; **no evaluation**.
- **Anthropic prompting docs** [V]: long material at the top, the query at the end; "Queries at the end can improve response quality by up to 30 percent in tests". The test is not published.
- **Memory-Induced Tool-Drift**, Dabas et al., 2605.24941v1 [A]: biased memories pushed into context shift tool calls by up to +3.6 on a 1-5 scale. Filters "reduce drift but do not eliminate it".

**Compression and context engineering**

- **LLMLingua** 2310.05736v2 [A]: up to 20× compression.
- **LongLLMLingua** 2310.06839v2 [H]: up to +21.4% at about 4× fewer tokens, partly by reordering documents by importance.
- **ACON**, ICML 2026, 2510.00615v3 [A]: peak tokens −26 to −54%; smaller long-horizon agents gain up to 46%.
- **What Does Context Compression Cost an Agent?** 2608.16370 [A]: at 5× compression, completion is unchanged, but re-fetch calls rise from 21.0 to 63.9.
- **Survey of Context Engineering**, Mei et al., 2507.13334v2 [A]: covers 1,411 papers.
- **Manus** [V] (2025-07-18):
  - "KV-cache hit rate is the single most important metric".
  - Keep the prefix stable and the context append-only.
  - "Mask, Don't Remove" tools, because changing tool definitions invalidates the cache and confuses the model.
  - Recite the todo list "into the end of the context".
  - No controlled numbers.
- **Anthropic, "Effective context engineering"** [V] (2025-09-29): "the smallest possible set of high-signal tokens"; just-in-time retrieval. No numbers.

## 2.5 An LLM as a classifier and router

**Calibration and label bias**

- **Calibrate Before Use**, Zhao et al., 2102.09690v2 [P].
  - Majority-label, recency and common-token biases; example order alone moves SST-2 from 54% to 93%.
  - Contextual calibration with a content-free input ("N/A"), W = diag(p_cf)⁻¹, gives "up to 30.0% absolute".
- **Batch Calibration**, Zhou et al., ICLR 2024, 2309.17249v3 [P]: subtract the mean prediction over the test batch. +8% (PaLM 2-S) and +6% (PaLM 2-L) over plain in-context learning; contextual calibration "displays more failure cases".
- **Mitigating Label Biases**, Fei et al., ACL 2023, 2305.19148v3 [A]: random in-domain words as the content-free input give "up to 37%" Macro-F1.

**Option and yes/no bias**

- **PriDe**, "LLMs Are Not Robust Multiple Choice Selectors", Zheng et al., ICLR 2024, 2309.03882v4 [P].
  - Moving the gold answer to A lifts llama-30B from 53.1 to 68.2; to D, it falls to 41.2.
  - On balanced items the model picks A/B/C/D 34.6/27.3/22.3/15.8%.
  - The cause is **token bias toward option IDs**, not position as such.
  - The prior is estimated by permuting option contents on about 5% of samples.
- **Permutation Self-Consistency**, Tang et al., NAACL 2024, 2310.07712v2 [A]: shuffle and aggregate: "+7-18% for GPT-3.5".
- **RankGPT**, 2304.09542v3 [H]: listwise reranking from a random start order falls to about 25-26 nDCG@10 (BM25 order 65.8, a single summariser read).
- **The yes-no bias ... reflects answer order and wording**, Huang, 2607.05552v1 [A]: the bias follows the **last-printed option**, plus a lexical pull toward "no". With arbitrary labels it is about 0. Remedy: cross the frames.
- **Acquiescence Bias in LLMs**, Braun, Findings EMNLP 2025, 2509.08480v1 [A]: LLMs lean to "no", unlike humans.

**Routers**

- **RouteLLM**, Ong et al., ICLR 2025, 2406.18665v4 [P].
  - Routers: matrix factorisation, BERT, a Llama-3-8B causal classifier, SW ranking.
  - BERT and the causal LLM "perform close to random when trained on the Arena dataset". Matrix factorisation was best with augmented data.
  - "high capacity approaches performing worse in a low-data regime".
- **Not Diamond** [V]: marketing only, with no method or evaluation.
- **JevBench** [V] (github.com/fstandhartinger/jevbench): exists, as a one-person, non-peer-reviewed benchmark. State plus rubric → a typed answer with per-option probabilities. 534 decisions, scored on chance-corrected accuracy, ECE/Brier, speed and cost. The top entries are 4B specialists.

**Our own numbers, for comparison** (`bench/decider/results/bonsai.json`, 2026-09-27, in-sample; measured here, not literature).
- On 124 phase points, CLM said "yes" on 114 (plan), 115 (implement), 99 (debug) and 119 (verify). The rule said yes on 4, 64, 15 and 8.
- Bonsai's yes/no agreed with the rule on 114, 70, 95 and 113 of 124.
- Bonsai's p(yes) on one row moved with the question's wording. "Go ahead." scored 0.18 / 0.28 / 0.42 across three phrasings.
- These match the literature's label-bias and wording-bias failure modes.

## 2.6 Thinking budgets and overthinking

- **s1**, Muennighoff et al., 2501.19393v3 [H].
  - Budget forcing on Qwen2.5-32B: append the end-of-thinking delimiter to stop, or "Wait" to extend.
  - AIME24 50.0 → 56.7, then "flattens out". Suppressing the end too often causes loops.
  - Rejection sampling for length shows inverse scaling. AIME has n=30.
- **Do NOT Think That Much for 2+3=?**, Chen et al., 2412.21187v2 [H].
  - QwQ-32B used 1,953% more tokens than conventional models on 2+3.
  - In over 92% of cases the first solution round is already right.
  - Trained efficiency on MATH500: tokens 2,407.9 → 1,330.7 at accuracy 93.0 → 92.8.
- **The Danger of Overthinking (agentic)**, Cuadron et al., 2502.08235v1 [H] (re-checked).
  - 19 models, including Qwen2.5 1.5-32B, QwQ-32B and R1-Distill-Qwen, on SWE-bench Verified with OpenHands, 4,018 trajectories.
  - Overthinking regresses negatively on resolution (β = −7.894 for reasoning models).
  - "Both reasoning and non-reasoning models show higher overthinking scores as their size decreases".
  - For o1 at high effort, native function calling raised performance "from 29.1% to 47.7%, while simultaneously reducing the average overthinking score from 2.43 to 1.05".
  - The judge is an LLM; single scaffold.
- **When More Thinking Hurts**, Zhou et al., 2604.10739v1 [H].
  - R1-Distill-32B and s1-32B, budgets 0.5-16K.
  - AIME marginal gain per 500 tokens: +3.2% at 0.5-2K, +0.1% at 8-12K, −0.3% at 12-16K.
  - Negative flips exceed positive ones from about 7K.
  - "stopping at ∼6K tokens yields ∼50% compute reduction with only ∼6% accuracy loss".
- **When More is Less (CoT length)**, Wu et al., 2502.07266v3 [H]: on Qwen2.5 1.5-72B, accuracy is an inverted U in chain length. Larger models peak shorter; harder tasks peak longer.
- **Don't Overthink It**, Hassid et al., 2505.17813v2 [H]: on R1-Distill-Qwen-32B, the shortest of k chains scores 60.0 vs the longest 37.8; on QwQ-32B, 71.1 vs 56.7.
- **Mirage of Test-Time Scaling**, Ghosal et al., NeurIPS 2025, 2506.04210v3 [H]: R1-Distill-Qwen-1.5B on GSM8K goes 82.2 → 87.3 (385 → 1,100 tokens), then falls to 70.3 at 19,980 tokens.
- **OptimalThinkingBench**, Aggarwal et al. (Meta), 2508.13141v2 [H].
  - Qwen3 1.7B-235B on simple queries: 750-950 thinking tokens, flat accuracy.
  - A "Don't Overthink" prompt cut Qwen3's thinking 23% with no loss.
- **Inverse Scaling in Test-Time Compute**, Gema et al., TMLR, 2507.14417v2 [H]: includes Qwen3-32B and QwQ at budgets 0-4,096. Mixed for Qwen3: it did not clearly inverse-scale. Synthetic tasks.
- **Qwen3 report** (above): smooth gains to 32K on 235B, on math, code and STEM.
- **Survey**, Sui et al., TMLR 2025, 2503.16419v4 [A]: a taxonomy of efficient reasoning.
- **[A] only**: AdaptThink 2505.13417; Budget Guidance 2506.13752; ThinkBrake 2510.00546v5; Between Underthinking and Overthinking 2505.00127.
- **No source gives an official per-task budget for agent steps.**

## 2.7 Progressive disclosure: model-pulled vs server-pushed

**Vendors and standards**

- **Anthropic, Agent Skills** [V]: blog of 2025-10-16 plus docs. Level 1 is name + description (about 100 tokens each), always loaded. Level 2 is SKILL.md (under 5k tokens), loaded when triggered. Level 3 is files. "The description ... must say both what the Skill does and when to use it". No measurements.
- **agentskills.io** [V]: an open standard, with Codex, Gemini CLI, OpenCode, pi, Hermes Agent, Cursor, Letta and others listed as adopters. SKILL.md under 500 lines.
- **Gemini CLI** [V]: the model calls `activate_skill`, which needs user consent. GEMINI.md holds always-present context.
- **OpenAI Codex/ChatGPT skills** [V]: the skill list is budgeted at "at most 2% of the model's context window, or 8,000 characters".
- **OpenAI function calling** [V]: "fewer than 20 functions" at a time. Tool search defers the rest, and deferred tools load "at the end of the model's context window" to keep the cache.
- **Anthropic advanced tool use** [V] (2025-11-24): tool search raised internal MCP evals from 49% to 74% (Opus 4) and 79.5% to 88.1% (Opus 4.5). "The most common failures are wrong tool selection and incorrect parameters, especially when tools have similar names." n not given.
- **Google ADK** [V]: ships both `load_memory` (pulled, "when your agent decides") and `preload_memory` (pushed, every turn).
- **LangGraph** [V]: procedural memory is the system prompt; memory is written "in the hot path" or "in the background".
- **LlamaIndex** [V]: memory blocks are inserted into the system message or the latest user message.
- **Letta** [V]: archival memory is only queried on demand through tools.

**Measurements**

- **MemGPT**, Packer et al., 2310.08560v2 [H]: self-directed memory paging. Deep memory retrieval: GPT-4 32.1 → 92.5. Frontier models only.
- **MemTool**, Lumer et al., 2507.21428v1 [H]: in autonomous mode, reasoning models remove 90-94% of stale tools, while "medium-sized models" manage 0-60% (LLaMA 3 70B 24.4%). Workflow (pushed) mode exceeds 90% for all models.
- **RAG-MCP**, Gan and Sun, 2505.03275 [H]: retrieving tools first gives 43.13% vs 13.62% with all tools in the prompt, on qwen-max. Not peer-reviewed.
- **Vercel** [V] (2026-01-27, re-checked), Next.js 16 APIs:
  - No docs 53%.
  - Skills by default 53%: "In 56% of eval cases, the skill was never invoked."
  - Skills with an explicit instruction 79%.
  - An always-present compressed docs index in AGENTS.md 100%.
  - The n and the model are not stated.
- **Self-RAG**, Asai et al., 2310.11511v1 [P, §5.2]: the 7B model's own hard retrieve decision scored 28.3 on PopQA, vs 45.5 with the adaptive threshold and 41.8 when it always retrieved. It under-retrieves.
- **When Do LLMs Need Retrieval Augmentation?** 2402.11457v2 [A]: LLMs "have difficulty knowing they do not possess certain knowledge".
- **Search-R1** 2503.09516v5 [A]: pulling well needed RL training (Qwen2.5-3B/7B).
- **SMART** 2502.11435v2 [A]: models also over-use tools; training cut tool use 24%.
- **SkillJuror** 2606.11543v1 [A]: a progressive-disclosure layout raised resources touched from 1.18 to 3.85 and pass rate +4.1%.
- **SkillsBench**: for frontier harnesses, discovery is "usually not the bottleneck" (Appendix K rates not read).

---

# Part 3. What this means for Yamadori

Each of our choices is checked against the evidence above. Our regime (Part 1) comes first where it conflicts with frontier results.

| our choice | verdict | the evidence |
|---|---|---|
| **Package detection → skill** (a pin, an import, an explicit ask or an error opens an area; the server pushes the skill) | **Supported** | For open models, pushing beats letting the model decide. SRA: load rates are insensitive to need or gold on Qwen3-4B/32B. Wild: Qwen3.5-397B force-loaded 41.2 vs agent-chosen 31.6. MemTool: pushed workflow above 90% vs 0-60% for mid-size models. Vercel [V]: never invoked in 56% of cases. Also: SkillsBench's gains need curated skills, and our IMPLIES rows cite the package's own docs. |
| **8 bodies at once** (`MAX_SKILLS_PER_TURN` = 8; the stack sentence injects 8, about 2,290 tokens) | **Contradicted in direction** | SkillsBench: 4+ skills +10.1 vs +18-19 for 1-3 (buckets confounded). ReasoningBank: best at 1 item, falling to 44.4 at 4. Memp: declines when too many are retrieved. SRA: full injection is the most brittle under distractors. Skill Shadowing: the loss grows with candidates. Hybrid evidence: irrelevant documents confuse SSM states (2406.07887). Our context cost (1.7% of the window) is not the issue: Shadowing finds context overhead about 0. Confusion and dilution are. |
| **Compact atomic skills** (body aim 100-300 tokens, hard cap 450, ≤6 items; distil/decompose) | **Supported** | SkillsBench: compact +19.0 and standard +21.5 vs detailed +14.5 and comprehensive +0.7. The public median is about 1.2k tokens, so ours are already small. Agent Skills/agentskills.io [V] allow under 5k; the research favours much less. |
| **Recall lines at the END of the tool result**, on an event (error, phase, asked) | **Supported (indirect)**; no study tests this exact mechanism | Recency favours the end: Lost in the Middle's U-shape, Anthropic's query-at-the-end [V], Laban's Recap 50.4 → 66.5, Manus's recitation [V], OpenDev's event-driven reminders (no evaluation). Li et al.: system-prompt attention decays across turns, so a skill given once early fades. Hybrid: mode/style rides the recency-weighted recurrent state (2609.04434; its position effect is not measured). **Caveat:** Prompt Repetition is neutral for reasoning models (5/1/22), and Memory-Induced Tool-Drift shows a wrong pushed memory biases tool calls. A recall on weak evidence has a cost. |
| **Byte-stable, append-only injection** (ledger replay; the tool list decided once per conversation) | **Supported** [V] | Manus: KV-cache hit rate, append-only context, "Mask, Don't Remove". OpenAI tool search loads deferred tools at the END to keep the cache. |
| **`yama_recall_craft` (model-pulled) beside the pushed skills** | **Supported as a supplement only; expect low use** | SRA: progressive disclosure on Qwen3-32B scores 55.3, below LLM selection 62.4 and about equal to top-1 injection 54.3. Models load at need-insensitive rates. Vercel [V]: 53%, the same as no docs. Self-RAG: under-retrieval. Search-R1: good pulling needed RL. The research does not justify leaning on pull for this model. The always-present **index** fits the "compressed index in context" result (Vercel [V]: 100%) and Codex's 2%-of-window budget [V]. |
| **Skills pipeline**: sources with verbatim provenance quotes, model-distilled | **Supported, with one gap** | Curated beats self-generated (SkillsBench −8 to −11.5 for self-generated; Dynamic Cheatsheet on small models). Our skills are distilled from cited sources, not written from the model's memory. That is closer to "curated", but a model still writes them. **Gap:** the faithfulness check is recorded SKIPPED for the pmndrs skills (`mcp/skill_offline.py`). SkillsBench's self-generated packs failed by carrying "confidently wrong" content. A ternary distiller loses most on code and reasoning (Part 1.2). |
| **Activation tests with near misses; quarantine a skill that fires on a near miss** | **Supported** | Confusability is the main selection failure: Single-Agent Skills (one competitor costs 7-30%), Right Family Wrong Skill (95% of helpful hits carry the risky sibling), Scaling Laws (boundary rewriting +12.8 routing). ASI and SkillOpt admit a skill or an edit only after it passes a check. |
| **Descriptions as trigger conditions** | **Supported, could go further** | Anthropic docs [V]: what and when. Scaling Laws: boundary rewriting and removing abstract "black-hole" skills. SkillsBench: "applicability boundaries and fallbacks". Our descriptions say when to use, but rarely when NOT to. |
| **Embedding stage**: Qwen3-Embedding-0.6B, query instruction `TRIGGER_INSTRUCT`, trigger texts only (name, title, description, when-lines, topics; "never its body") | **Instruction: supported. Index text: contradicted.** | The query-side instruction follows the card (1-5%) and ToolRet (instructions help most for instruction-tuned embedders). SkillRet likewise authored a skill-retrieval instruction. But SkillRet App. E found indexing the **body** as well beats name + description by 1.5-11.4 nDCG@10. Tool-DE warns that LLM-written example queries can hurt. Absolute ceiling: off-the-shelf Qwen3-Emb-0.6B reaches 58.35 nDCG@10 (SkillRet) and 0.431 Hit@1 (R3-Skill), so the pattern rounds should stay primary. |
| **No absolute cosine cut; rank only** | **Supported** | ToolRet, SkillRet and the calibration papers (a score depends on the query and the batch). Cacioli: calibrated levels do not transfer, rankings do. |
| **Reranker "not trusted, not used"** | **Consistent** | Off-the-shelf rerankers help a weak off-the-shelf first stage (SkillRet +4 to +9 over Qwen3-Emb-8B) but give little or negative gain on tool corpora (ToolRet: MonoT5 −4.9) and hurt fine-tuned embedders. The local batch-composition bug (FINDINGS #20) must be fixed before any of that applies. |
| **The Bonsai typed decider**: argmax over label-token probabilities, lettered options, "None of these" last, no threshold (`mcp/skill_match.py`, `bench/decider/bonsai_decider.py`) | **Contradicted as raw argmax; supported once calibrated** | Our regime: quantization shifts confidence most on uncertain items, and calibration does not transfer across precision (2405.00632, 2604.08976). MTP/quantized kernels can flip near-tied argmaxes (2607.17283). Literature: option-ID token bias (PriDe), last-printed-option bias (2607.05552), majority/recency/common-token bias with content-free calibration up to +30 (Calibrate Before Use), Batch Calibration +6-8%. An 8B causal-LM router was near random on low data (RouteLLM). Ours: CLM said yes on 99-119/124; Bonsai's p(yes) moves 0.18-0.42 with wording. **"None of these" printed last is exactly the position the yes/no study finds favoured.** Comparing only within a set is right (PriDe/Calibrate address set-level priors). |
| **Thinking caps**: agent step 6,144; HELPER_THINKING 6,144; investigate 6,144; plan 12,288 | **Supported (indirect)**; no agent-step budget study for Qwen3 exists | Our regime: aggressive low-bit lengthens outputs (2504.04823), and small models overthink more (Cuadron). Literature: on 32B reasoners gains saturate at 8-12K, negative flips dominate past about 7K, and a 6K stop costs about 6% for 50% compute (2604.10739, math only). Agentic overthinking lowers SWE resolution (Cuadron). Qwen3 simple queries use 750-1,600 tokens for no gain (OptimalThinkingBench). **Against:** Qwen3-235B improves smoothly to 32K on math and code (Qwen3 report), and Qwen's cards recommend 32K+ outputs. Those are benchmark maxima, not agent steps. Our plan job at 12,288 sits past the flip point measured on math. That is not contradicted for planning, which no study measures. |
| **Budget nudge wording** (`AGENT_STEP_NUDGE_MESSAGE`, `HELPER_NUDGE_MESSAGE`) | **No evidence either way** | Qwen3's report documents the family's own stop phrase ("Considering the limited time by the user, ...\n</think>.") and that budget-following "emerges naturally" with it. Whether Qwen3.8/Bonsai respond better to it than to our operator-approved wording is unmeasured. |
| **Past reasoning passes through** (the client's echo or its absence; Hermes strips it) | **In tension with the base model's guidance** | Qwen3.8-27B [V]: `preserve_thinking` is on by default, for decision consistency and "improved KV cache utilization". Qwen3.6 [V] was "additionally trained" to use historical thinking. The Qwen3/3.5/3.6 templates keep thinking within the tool loop after the last user query. Our Octopus measurement (every request diverges at the previous think block and rests on a checkpoint) is the cost side. Not a skills question, but the base model's authors state a direction. |
| **Code check and repair of client writes** | **Supported by the regime evidence** | Ternary PTQ loses most on code and math (2608.01078, 2509.16989). ACBench: JSON-format output degrades more than string format. |
| **The addendum and craft index at the END of the system text; ≤2 prohibitions** | **Consistent** | The index at the end suits recency (Lost in the Middle). Shi et al.: an "ignore irrelevant information" line helps; no study tests prohibitions per se. Our own 10.7/10.0/9.3 measurement is the only evidence on prohibitions (and is at risk, #31). |

---

# Part 4. Design changes the research recommends

Each change names its sources. None is measured on Bonsai. Per PROTOCOL, each needs our own live measurement before a claim, and the operator decides.

1. **Cap the skill BODIES delivered together at 3, and send the rest as index lines** (the name plus its when-line) that `yama_recall_craft` can fetch.
   - Sources: SkillsBench (1-3 skills +18-19 vs 4+ +10.1), ReasoningBank (k=1 best), Memp, SRA (full injection brittle), Skill Shadowing, and the hybrid distractor evidence (2406.07887).
   - Order: the asked areas first, then implied ones. Today's 8-body stack case would deliver the 3 asked bodies and index the implied 5.
   - The "3" is SkillsBench's bucket boundary, not a tuned number, and its buckets are confounded.
2. **Debias the Bonsai decider before it replaces the stub.**
   - (a) Contextual calibration: score each question template once with a content-free state ("N/A") and divide it out (Calibrate Before Use; Fei et al.). Or subtract the mean over a turn's batch of questions (Batch Calibration).
   - (b) Permute option order, and do not always print NONE last. Average over at least two orders (PriDe; Permutation Self-Consistency; 2607.05552's last-option bias).
   - (c) Prefer multiple-choice with neutral labels over yes/no. The yes/no bias vanishes with arbitrary labels (2607.05552), and CLM's yes/no collapsed locally.
   - (d) Re-calibrate on the served checkpoint, never import thresholds from another precision (2405.00632, 2604.08976).
   - (e) Treat near-ties as ties, since the quantized MTP batch can flip them (2607.17283).
3. **Index each skill's body with its trigger text for the embedding stage.** Keep the skill-retrieval `Instruct:` on the query side and no instruction on documents.
   - Sources: SkillRet App. E (+1.5 to +11.4 nDCG@10); the Qwen3-Embedding card.
   - Do not add LLM-written example queries (Tool-DE: often negative), or filter them if added (Doc2Query--).
4. **Write "NOT when" boundaries into every description**, especially between siblings (r3f v9/v10, koota-react vs koota-r3f, the 26 TSL skills). Retire generic "black-hole" skills that match everything.
   - Sources: Scaling Laws of Skills (boundary rewriting +12.8, removing abstract skills +4.9), SkillsBench's applicability boundaries, Right Family Wrong Skill, Single-Agent Skills (confusability).
   - The activation-test near misses are the place to enforce it.
5. **Make the faithfulness check mandatory before a model-distilled skill arms.** Do not record it SKIPPED.
   - Sources: SkillsBench (self-generated packs with "confidently wrong" content, −8 to −11.5), ASI (verification +4.2), Dynamic Cheatsheet (small models write flawed memory), ACE (a rewrite can collapse context), and the regime evidence that ternary models lose most on code.
   - When the model rewrites a skill, prefer delta edits over whole rewrites (ACE).
6. **Keep push primary; measure pull.** Record `yama_recall_craft`'s call rate against the turns where the index listed a craft the pushed set lacked, before relying on it.
   - Sources: SRA's need-insensitive load rates, Vercel [V] (56% never invoked), MemTool.
   - If the rate is low, the always-present index is the useful part (Vercel's compressed index [V]).
7. **Keep recall lines, but only on evidence**, as today. Do not add periodic reminders.
   - Sources: prompt repetition is neutral for reasoning models (2512.14982), and wrong pushed memories cause tool drift (2605.24941). Recency (Lost in the Middle, Laban's Recap) is why the end of the tool result is the right place.
   - Our "no fade, no cooldown; recall on an event" rule is consistent with this.
8. **Put thinking budgets on a measured footing for our regime.**
   - The literature brackets 6-8K as where marginal gains vanish on 32B reasoners (2604.10739) and shows overthinking hurts agent work (Cuadron). It does not measure agent steps on a ternary 27B.
   - Log per-step thinking length against the step's outcome (the #53 data already exists) to check the 6,144 cap in our regime. Measure the plan's 12,288 the same way.
   - Evaluate Qwen3's own stop phrase beside our nudge only as an operator decision on wording.
9. **Take the Qwen3.8 `preserve_thinking` default to the operator** as a question: does restoring the tool-loop reasoning (the Qwen3.x template default: thinking after the last user query) buy consistency and cache reuse that outweigh the 6-10k tokens per step measured in the V0 pilot?
   - Sources: the Qwen3.8 and Qwen3.6 cards [V], the Qwen3-32B template.
   - **Decided 2026-09-27 (operator): restore it.** "Keeping thinking across turns seems useful, fuck Hermes, Hermes can do whatever it wants." The served template (the GGUF's own) already renders every past turn's think block (`preserve_thinking` undefined = on), so the proxy restores the slot's reasoning from the ledger and passes nothing (AGENTS.md "Past reasoning is restored"; switch `restore_reasoning`).
10. **Watch for thinking-mode tool-call swallowing.** "Planned the call in reasoning, ended with finish=stop and no call" is a documented Qwen failure (Qwen3#1817 [V], 2608.28439).
    - The proxy already records `finish_reason` and reasoning-held calls for helper hops. Doing the same for main's agent steps would show whether pagoda-h2's stop is this pattern.
    - Evidence is anecdotal: n=5 plus one appendix.

---

## Discrepancies and cautions

- **SkillsBench versions.** v1 said 86 tasks, 11 domains, +16.2 pp, 16 of 84 negative. v4 says 87 tasks, 8 domains, +16.6 pp, 13 of 87 negative. v4 is cited.
- **SkillWeaver.** The v1 text says "39.8% ... 54.3%", which is the live-website figure. Its Table 1 gives WebArena gains of 32% and 45%. The table is cited.
- **SkillReranker (2607.06283).** The summariser claimed an LLM listwise position-bias finding. The phrase does not occur in the paper's extracted text, so it is not cited.
- **ToolRet Table 5.** The instruction-condition values for e5-mistral and gte-Qwen2 were read two ways (38.97/45.96 vs 40.02/41.27 as reprinted by Tool-DE). Neither is relied on.
- **Qwen3.5 thinking presence_penalty.** 1.5 and 0.0 were both seen (HF discussion "Conflicting sampling parameters", not opened). Qwen3.8's card says 0.0.
- **The Bonsai 2 card** says base Qwen3.8-27B. The PrismML news page (April 2026) covers only the 8B/4B/1.7B Ternary Bonsai and names no base or method.

## Not verified (not relied on above)

- SkillOpt's per-Qwen numbers (inconsistent reads); SKILLER's body; SkillsInjector 2605.29794 (abstract only, models from a snippet).
- ThinkBrake's BFCL/Qwen3-4B setup; Budget Guidance's models.
- The ACON ">95%" figure; the Chroma per-model numbers; the n and model of the Vercel eval.
- SkillsBench Appendix K (invocation rates).
- "Hiding the skill body lowers routing accuracy 31-44 pp" (SkillRouter, cited by R3-Skill; not opened).
- Every venue not shown on the page read (ToolLLM ICLR, Gorilla NeurIPS, Calibrate Before Use ICML, ExpeL AAAI).
- Not opened at all:
  - Skills and memory: LEGOMem, Agent KB, Memento, SkillRL, EvolveR, SkillReducer 2603.29919, the Agent Skills survey 2602.12430.
  - Quantization and hybrids: Tequila 2509.23809, "Give Me BF16 or Give Me Death" 2411.02355, DAMP 2608.27513 (abstract only: quantizing the GDN recurrent STATE harms reasoning, which is relevant only if the state is ever quantized).
  - Tool retrieval: ToolDreamer 2510.19791, "Tool Retrievers Are Underestimated" 2609.08327.
- **Not found at all:**
  - A ternary model evaluated on agent, tool, long-context or skill-use tasks.
  - A needle-position or lost-in-the-middle study on Gated-DeltaNet hybrids.
  - Label-token calibration under 1-2-bit quantization.
  - A llama.cpp q8_0 KV long-context measurement.
  - A PrismML paper.
