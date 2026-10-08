# The decider: what Jev is, how an LLM makes calibrated closed-set decisions without an embedder, and who the open competitors are

Research note, 2026-09-29. Web research plus a read-only look at our own logs. No code changed, no GPU ran, nothing was sent to `:1234`, `:11434` or `:1235`, and nothing was downloaded except public README/results text files (GitHub raw, read into the scratchpad).

The operator (2026-09-29): "I'd prefer to go principled on decider and really deeply research how Jev works and how our decider can improve without embeddings before we dive deeper, send out the agent to find Jev, research, and the open and free top tier competitors."

The subject is our decider: `mcp/decider_bonsai.py` (one forward pass on the served Bonsai 2 27B, thinking off, the next-token distribution read at an `Answer:` prefill, labels A..Z, renormalised over the labels) and `mcp/decide_turn.py` (the turn's questions: build intent, phase, weak package detections, the selector's `choose()`, `judge_stop`; two option orders for the fact questions, a count-dependent rotation plus a divided-out label prior for `choose()`, a head+tail state cut at 2,048 tokens, a 0.0034 tie band).

## How to read the citations

Two labels per source: how it was read, and who is speaking.

| reading label | what was read |
|---|---|
| **[R]** | Raw text read directly (a README or results file fetched as text and read here, no summariser). Numbers are as printed. |
| **[H]** | A page read through the fetch tool's summarising model. Numbers came back through that model and were not checked character by character. |
| **[A]** | An arXiv abstract page only. |
| **[S]** | A search-result snippet only. **Not verified.** Not relied on. |
| **[P]/[H]/[A] (SR)** | Already read and labelled in `docs/research/SKILLS-RESEARCH.md`; not re-read here. |

| provenance label | who is speaking |
|---|---|
| **vendor** | The maker of the thing measured (TypeSafe about Jev; a model's authors about their model). |
| **self-reported** | An open-source author measuring their own system, usually on a development set they consulted. |
| **independent** | Someone other than the maker (a third-party benchmark, a peer-reviewed paper about other people's models). |
| **ours** | Our own measurement, with its script and n. In-sample unless said otherwise. |

"Inference" marks a conclusion this note draws that no single source states. Every number carries its conditions; where none is given, none was found.

---

## The biggest takeaways (one screen)

**1. Jev is a closed, hosted "System One" decision model from TypeSafe AI, and it is not an acronym.**
- State plus typed questions in (`choice` up to 255 options, `score` on a rubric, `noul` true/false); a probability for every option out; no text generated. `jev-1.13.0`, text only, 64k tokens per request (32k for state plus the longest question), $0.042 per million input tokens, output free [H, vendor].
- Architecture, base model, size and training data are **not disclosed**. The vendor says "non-autoregressive" with a "parallel sampler" and "Reinforcement Learning for Calibrated Decisions (RLCD)" [H, vendor]. An independent black-box study infers a causal decoder with a shared state prefix and isolated question branches, read out from a small head after prefill [H, independent, partly speculative].
- No weights, no code. Early access only.

**2. JevBench is the only independent scoreboard, and on the two axes we care about, a FROZEN Qwen3.8-27B read at its option-letter logits ties or beats Jev.**
- JevBench v1.2/v1.3 [R, independent; one maintainer, not peer-reviewed; 534 decisions]: Jev 1.13.0 Intelligence 85.7 / Calibration 82.7; **reflex-27b (frozen Qwen3.8-27B, two option orders averaged, T=1, no training) 85.8 / 86.2; SimpleJev Qwen3.8-27B 84.7 / 81.1**; LitJev Qwen3.8-27B 82.4 / 83.5; a stock Qwen3-32B with one fitted temperature 79.3 / 79.0.
- Hard tier (long multi-condition policies, traps, multi-hop, numbers): Jev 74.1%, reflex-27b 75.9%, SimpleJev 27B 75.0%. The 4B rows JevBench ran in that release land at 56.8-63.2%.
- Jev and the 4B specialists win the composite on **speed and cost**, which the composite weights 50%. For us (Bonsai already resident, no per-token price) those axes do not transfer.
- **Our base model is exactly the one that scores best open.** What we do not know is how much the ternary PTQ costs that readout. That is the first thing to measure (Part 4, item 2).

**3. The Jev-clone field converged on one recipe, and it is ours, with two differences.**
- The recipe [R, self-reported, several independent authors]: state first and shared once; each question isolated after it; options as letters; the answer read from the letter logits in one pass; thinking off; **two distinct option orders averaged**; at most one fitted temperature, fitted on your own workload.
- The differences: our `choose()` and `judge_stop` read ONE count-dependent rotation and divide out a content-free label prior. The best-documented ablation (reflex) found the fitted/content-free prior **hurt** (hard ECE 0.086 to 0.124, accuracy down) and two orders **helped** (27B hard 0.703 to 0.766, ECE 0.088 to 0.061) [R, self-reported, dev suite].
- Our own calibration run agrees: content-free calibration hurt every yes/no-type question (intent 0.869 to 0.818; reads_package 0.851 to 0.582) [ours, `bench/decider/results/bonsai.json` `calib`, in-sample].

**4. Calibration does not travel; fit it on our own labels or not at all.**
- "A temperature is a property of a distribution, not of a model" [R, reflex]: a temperature fitted on one mix made external ECE worse on 3 of 4 sets. jqv: an MMLU-fitted temperature "transfers partially to JevBench hard and is harmful on a reading-type task" [R].
- Quantization moves calibrated levels but keeps rankings (Cacioli [A] (SR)); NVFP4 serving of Qwen3.8-27B raised the letter-readout's ECE 0.061 to 0.087 [R, reflex, self-reported].
- So: every threshold and temperature is fitted on the served Bonsai, per question, on held-out labels from our traffic.

**5. Thinking before the label buys accuracy on hard items and costs seconds; the evidence says "cascade on low confidence", not "always think".**
- For: reasoning variants lift JevBench hard (djev 69.5% to 77.7%; OpenJev thinking 78.2%) [R]; reasoning judges beat non-thinking Qwen3 0.6-4B by ~10 points on RewardBench [A]; reasoning models are better calibrated in 33 of 36 settings [A].
- Against: CoT helps "mainly on math and symbolic reasoning" [A]; Granite Guardian's think mode scores at or below no-think (harm F1 0.79 vs 0.81) [H, vendor]; reflex's disagreement-gated reasoning did not improve the hard tier and put p95 at 42 s [R].
- A cheap, well-supported gate exists: the two orders' disagreement. reflex, external sets (n=1,200): disagreement < 0.05 gave accuracy 0.908; > 0.30 gave 0.250 [R, self-reported].

**6. Fine-tuning and prompt search overfit; the frozen model plus the right readout generalised best.**
- reflex: every LoRA mix (4 of them) and a 27B-teacher distillation "won on data shaped like its training data and lost general judgement on long, ambiguous inputs"; GEPA prompt optimisation did the same "in miniature" [R].
- jqv: "Accuracy is set mainly by backbone scale"; a head + LoRA on 4,800 examples helps at 1.7B "and not at 14B/32B" [R].
- Two small prompt changes did transfer in reflex's controls: a **lettered yes/no pair** (we already do this) and **"# Evidence / # Criterion" headings** (toxic-chat ECE 0.104 to 0.046 with both) [R].

**7. Our decider's measurements are in-sample and our live log cannot be judged yet.**
- `logs/decider_disagreements.jsonl`: 260 disagreement rows when read (2026-09-27 to 09-29; the file grows live). 115 of them are one pattern: the decider says **verify**, the rule says **implement**, mostly on agent steps. No truth labels exist for any row.
- The log's request ids join to 0 of the corpus's turn ids (checked: 0 of 222), so the text behind a disagreement cannot be recovered for labelling. Fixing that join is the cheapest high-value change.

---

# Part 1. Jev, exactly

## 1.1 Who, what, when

| fact | value | source |
|---|---|---|
| maker | TypeSafe AI | TypeSafe blog [H, vendor] |
| founder named on the blog | Diogo Almeida ("previously at OpenAI") | TypeSafe blog [H, vendor] |
| announced | 2026-09-15, early access | search snippets (Vercel, MarkTechPost) [S]; the repo's `docs/CLM-EVAL.md` §0 cites the same date from snippets |
| funding | "$40 million seed round led by DCVC" | search snippet [S], **not verified** |
| name | "Jev". No expansion appears on any TypeSafe page read. "System One" borrows Kahneman's System 1 / System 2. | TypeSafe docs and blog [H, vendor]; Simple Jev README [R] |
| model id | `jev-1.13.0`; `jev-latest` and `jev-preview` both point to it | docs.typesafe.ai/models [H, vendor] |
| endpoint | `POST /v1/systemone` | same |
| weights / code | **None released.** A Python "System One LLM adapter" repo exists (typesafe-ai/system-one-adapter-python); not read. | TypeSafe blog [H, vendor] |

**"JevBench" is not TypeSafe's.** It is Benchmark Heaven's benchmark by Florian Standhartinger, "not affiliated with or endorsed by TypeSafe AI" [R]. "Jev clones" are the dozens of open reproductions it ranks (Part 3).

## 1.2 What it decides, and the API contract [H, vendor, docs.typesafe.ai]

- **Three primitives.**
  - `choice`: "Choose an option from a list". Returns `choice`, `probabilities` (sum 1) and `confidence`. "A Choice question accepts up to 255 options." The docs advise the full list of options and an `other` / `none of the above` option when the list might not be exhaustive. Option order is not discussed.
  - `score`: rate the state on a rubric of described levels. Returns `score`, `probabilities`, `confidence`.
  - `noul`: "Is this statement true?" Returns one number in 0-1 and no confidence.
- **Fields.** `state` (text, JSON or an array of text), and per question `instructions` (the question) and `criteria` (the options or levels, each with a description). "All questions are evaluated in parallel and in isolation against the same state."
- **Confidence is not a second model.** It is a function of the distribution's shape: "concentrated on one outcome means a confident answer, spread out means an uncertain one"; for three options, approximately "(3 × largest probability − 1) / 2". Recommended use: act above ~0.9 for high stakes, human review below 0.5, "Start with conservative thresholds, test with your own data".
- **Limits.** 64k tokens per request, "32k tokens for `state` plus the longest question". Text only. English best. No per-account fine-tuning: behaviour is shaped only by `state`, `instructions` and `criteria`.
- **Published "jaggedness" of 1.13** (the vendor's own limitations list; each is a design hint for us):
  1. Literal reading: "Jev answers the question you wrote, not the one you meant."
  2. No arithmetic or counting; 3. numeric representations read poorly; 4. score interpolation is weakly calibrated; 5. dates are text, not ordered quantities.
  6. Indirection and multi-hop questions lose accuracy.
  7. **"Large Irrelevant State"**: accuracy falls as unrelated detail acts as a distractor; filter in code first.
  8. Adversarial content in the state can move the answer.
  9. Contradictory instructions and criteria confuse it.
  10. **No structural invariants**: "Don't assume P(noul) + P(not noul) = 1"; don't transfer thresholds between primitive types.
  11. It is not a generator.

## 1.3 How it works mechanically

**What the vendor says** [H, vendor, TypeSafe blog and ML primer]:
- "Non-autoregressive with parallel sampling", a custom architecture, a parallel sampler, and RLCD training that returns "decisions and calibrated probabilities instead of generated text".
- Calibration defined as usual ("Outcomes assigned a probability of 0.2 should occur about 20% of the time").
- Base model, size and training data: not disclosed.

**What an independent black-box study observed** (Archer Hume, "Jev's architecture, unmasked" [H, independent]):
- Observed: token accounting is strictly additive across questions; latency is nearly flat up to ~100 questions. A declaration placed in a sibling question does not reach another question, while the same declaration in the state does (0.90-0.92). So questions are isolated from each other and share the state.
- Observed: a 1,500-question request took ~610 ms median; a 30k-token state ~160 ms. The returned numbers do not change the cost ("an answer of 0.0 costs the same as 0.01").
- Inferred by the author: a causal decoder (84.6% MMLU-Pro "requires frontier-scale pretraining"), a shared prefix KV with isolated causal suffixes, and a small readout `z = Wh + b` with a softmax over options. Speculated: a sparse MoE with ~10B active parameters. The tokenizer matches no public one exactly but is closer to Qwen's.
- Observed calibration: MMLU ECE 0.0313 on 1,200 items.

**What TypeSafe's own cookbook shows about determinism** [H, vendor, consistency_choice_cookbook]: repeating one 8-question moderation request 15 times with only a throwaway `uid` changed gave 90.8% top-label agreement (Haiku at temperature 0: 100%) and a mean per-label standard deviation of 0.0098. **Jev is not deterministic across calls** (inference: consistent with the "parallel sampler").

**Inference for us.** Everything observable about Jev's serving shape (shared state, isolated branches, a label readout, no text) is what our decider already does on llama-server, sequentially rather than in one pass. What we cannot copy is RLCD, the training on proper scoring rules. The open clones show how far a frozen model gets without it (Part 3).

## 1.4 Speed and cost

| claim | number | label |
|---|---|---|
| end-to-end latency | "70ms-500ms", "40x-200x faster than frontier LLMs" | vendor [H]; the blog says the speed tests ran "on our laptops on the West Coast" |
| measured latency | p50 0.65 s, p95 0.72 s from Germany | JevBench v1.2 [R, independent] |
| price | $0.042 per million input tokens, output free; JevBench computes $0.0399 per 1,000 of its decisions | vendor [H]; JevBench [R] |

## 1.5 How good it is

**Vendor claims.** "Cannot hallucinate" (by schema construction, not measured), "193.6x faster, 444.6x cheaper" on workflow evals the vendor built itself ("individuals on our model capabilities team") [H, vendor].

**JevBench v1.2/v1.3** [R, independent]. Jev 1.13.0: easy 100%, standard 99.0%, judge 94.5%, hard 74.1%; Intelligence 85.7, Calibration 82.7; composite #1 at 74.4 of 48 ranked systems.
- Later releases added sealed items (v1.4: 308; v1.5: 720) and changed the scoring. Jev's composite is 63.29 in v1.4.2.2 (#4) [R]. On the v1.4 sealed items the jqv author reports Jev 0.367 against a field median 0.292 [R, self-reported]. The benchmarkheaven page shows v1.5.1 with Jev at Intelligence 72, Calibration 88, composite #3 [H]. **The absolute numbers move with every release; only same-release comparisons are meaningful.**

**Two independent academic evaluations** (both September 2026, arXiv, abstract only):
- "Same Scores, Different Decisions" (2609.27678 [A]): Jev vs nine LMs on ContractNLI. "Rankings by baseline accuracy differ from rankings by correctness across every condition and repeat"; aggregate accuracy "can conceal changes in the individual decisions" when the request's format or ordering changes.
- "Replacing LLMs with Jev Decision Models for Low-Latency Edge Service Orchestration" (2609.22753 [A]): 8,280 verified requests. Jev cut median decision latency 22.7-64.5% against the fastest LLM and fees per correct decision 59.7-80.9%, "at a cost of a few exact-match points"; "Wide contracts mark the limit of the substitution" (it worked for four to eight intent fields).

**The one cookbook that matches our job** (skill selection) [H, vendor, docs.typesafe.ai/cookbooks/skill_suggestion]:
- 182 skills from Nous Research's Hermes catalog in 33 categories; 488 requests (315 with one covering skill, 173 with none).
- Two stages: one `choice` over all 182 with ~54-character descriptions, then the top 3 re-ranked with full descriptions and the SKILL.md opening. A gate of three `noul` questions ("is any action needed?") and a per-candidate `noul` fit check, each with a 0.30 threshold.
- Against Haiku 4.5 choosing its own skills: wrong loads 16.8% to 7.3%, needless loads 9.8% to 4.0% (oracle 2.5% / 1.2%). Two calls, ~0.31 s + ~0.09 s.
- Vendor-run, one dataset, thresholds not said to be held out. The shape (a gate question, a coarse choice, a fine re-rank over 3, an explicit "none") is the part worth copying, and it agrees with our pushed-skills design (SKILLS-RESEARCH 2.7).

---

# Part 2. Deciding with an LLM's own probabilities, by topic

## 2.1 The label-logprob readout and its biases

**Why a letter readout works at all.**
- Multiple-choice prompting (question and lettered options together, answer with the letter) beats cloze scoring for models with "multiple choice symbol binding" ability, which "varies greatly by model" (Robinson et al., ICLR 2023, 2210.12353 [A]).
- Larger models are well calibrated on lettered multiple-choice when the format is right, and a "None of the above" option changes calibration (Kadavath et al. 2207.05221 [A]).

**Where it goes wrong: the first token is not always the answer.**
- For instruction-tuned models the first-token distribution and the text the model would write "are severely misaligned on all dimensions, reaching mismatch rates over 60%", worst for models trained heavily on chat and safety data ("My Answer is C", ACL Findings 2024, 2402.14499 [A]).
- Where the mismatch exceeds 50%, the text answer is more robust to option order than even PriDe-debiased token probabilities (COLM 2024, 2404.08382 [A]).
- Our mitigation is the `Answer:` prefill with thinking off, so the next token should be the label. `decider_bonsai` already reports `label_mass`, the labels' share of the whole distribution. **Inference:** `label_mass` is the direct measure of this failure on our model and should be logged per decision.

**Biases of the label distribution** (SKILLS-RESEARCH 2.5 has the full entries):
- Majority, recency and common-token bias; content-free calibration gives "up to 30.0% absolute" (Calibrate Before Use 2102.09690 [P] (SR)).
- Option-ID token bias: llama-30B picks A/B/C/D 34.6/27.3/22.3/15.8% on balanced items (PriDe 2309.03882 [P] (SR)).
- The yes/no bias follows the last-printed option and the word "no"; it is about 0 with arbitrary labels (2607.05552 [A] (SR)).
- Surface-form competition: synonyms split the mass; PMI_DC reweights by each option's prior (Holtzman et al. 2104.08315 [A]).

**The fixes, and what happened when people measured them on current Qwen models.**

| fix | literature | measured on current open models |
|---|---|---|
| **contextual (content-free) calibration**: divide by the "N/A" read | Zhao et al. [P] (SR); domain-context version with random in-domain words, "up to 37%" Macro-F1 (Fei et al. 2305.19148 [A] (SR)) | **Hurt.** reflex, Qwen3.5-4B: the content-free prior is "large and ragged" (0.053 on one option, 0.422 on another) because with no evidence the model reads the option TEXTS, which differ per question. Division cost accuracy (external sets 0.808 to 0.790) [R, self-reported]. **Ours:** it hurt every yes/no-type question (the N/A read leans "no" 0.78-0.90) and helped only as a veto on package detections [ours, `calib`, in-sample]. |
| **PriDe-style fitted position prior**, divided out of one reading | PriDe [P] (SR) | **Rejected.** reflex 4B: outside the yes/no pair the letter prior is near uniform (`choice:12` every position within half a point of uniform). Dividing it out lost a hard item and raised hard ECE 0.086 to 0.124; a borrowed prior was a no-op. The 27B was not measured [R]. |
| **averaging over option orders** | Permutation Self-Consistency, "+7-18% for GPT-3.5" (2310.07712 [A] (SR)); Balanced Position Calibration for judges (2305.17926 [A]) | **Helped, repeatedly.** reflex, frozen Qwen3.8-27B, 1 vs 2 distinct orders: standard 0.917 to 0.958, hard 0.703 to 0.766, hard ECE 0.088 to 0.061 [R, self-reported, dev suite]. jqv: `perm_avg` "a training-free +3 points at 1.7B/14B and cancels the position and letter priors" [R]. **Ours:** build intent 0.848 to 0.869, phase agreement with the rule 47 to 57 of 79 [ours, in-sample]. |
| **more than two orders** | | reflex 27B, four orders: hard 0.712, ECE 0.095 (on code that was later found buggy; not rerun) [R]. No evidence that more than two pays. |
| **batch calibration**: subtract the mean prediction over a batch | +8% / +6% on PaLM 2 over plain ICL; contextual calibration "displays more failure cases" (2309.17249 [P] (SR)) | Not measured by any Jev clone read. **Ours:** on build intent, `mc_avg` with batch calibration scored 0.887 (2 ties) against 0.869 without; the yes/no read stayed at 0.848. In-sample twice over: the batch mean is the 99 scored rows' own [ours, `calib`]. Not adopted; a streaming version (the running mean of the last N decisions of one question) is testable on the item-1 labels (4.2). |
| **prototypical calibration**: a GMM over output probabilities, clusters matched to labels | robust to templates, permutations and class imbalance (Han et al. 2205.10183 [A]) | Not measured by anyone read. It needs a batch of unlabelled decisions per question, which our log could supply. |

**Why the two fixes diverge** (reflex's own diagnosis, [R]): a fitted prior removes the AVERAGE position effect over a corpus; two orders remove THIS question's effect. The permuted and content-free estimators "disagree about everything except `noul:2`", so the effect is not constant across questions, and "the second order is mostly buying an ensemble".

## 2.2 Yes/no vs lettered choice

- reflex's controls, frozen Qwen3.5-4B, toxic-chat (external, yes/no): reading Yes/No tokens 0.787 / ECE 0.104; a lettered pair ("A. yes: ...", "B. no: ...") 0.797 / 0.083; with Evidence/Criterion headings as well 0.843 / 0.046. "The yes/no token readout carried a 'say Yes' prior that the lettered pair removes" [R, self-reported].
- A wording ensemble that mixed Yes/No-token readouts was worse on both models; "averaging pulls the answer toward" the yes prior [R].
- reflex's `noul:2` prior leans "yes" by 17 points, and removing it cost accuracy on toxic-chat: "closer to a correct base rate" than a habit [R].
- **Ours:** `FORM` already asks build intent and phase as neutral-letter choices in two orders (`mc_avg`). `reads_package` and `scratch_write` still use the raw yes/no read, which scored best on the h4 steps (0.888) [ours]. The literature and our data agree on letters for semantic questions. The step questions are rules now anyway.

## 2.3 Verbalized confidence vs token probabilities; P(True)

- For RLHF models, "verbalized confidences emitted as output tokens are typically better-calibrated than the model's conditional probabilities", ~50% relative ECE reduction (Tian et al., EMNLP 2023, 2305.14975 [A]; ChatGPT, GPT-4, Claude).
- Against: verbalized confidence "tend[s] to be overconfident"; white-box methods do better, "0.522 to 0.605 in AUROC" (Xiong et al., ICLR 2024, 2306.13063 [A]).
- JevBench scores verbalized systems (JSON schema, the model writes probabilities) and native readouts separately. DeepSeek V4.1 Flash, verbalized, reached Calibration 96.7 [R]; that is a frontier API with its own reasoning.
- P(True) self-evaluation scales and calibrates well on larger models, and seeing several samples first helps (Kadavath et al. [A]). Jev's docs warn that a separate true/false question does not obey probability identities with the choice it checks (jaggedness #10) [H, vendor].
- **Inference for us:** verbalized confidence costs generated tokens. Our single-token readout is the white-box side of Xiong's comparison. Nothing read suggests switching.

## 2.4 Prompt format, demonstrations, and state

- **Format sensitivity is large and does not go away with scale.** Up to 76 accuracy points on LLaMA-2-13B from formatting alone; it "remains even when increasing model size, the number of few-shot examples, or performing instruction tuning" (Sclar et al., ICLR 2024, 2310.11324 [A]). reflex: wording is worth "±8 points" on the frozen 4B [R].
- **Prompt search overfits.** GEPA on reflex's own mix: validation 0.628 to 0.649, then a 2-5 point loss on every external set [R, self-reported]. The same lesson as fine-tuning.
- **Demonstrations.**
  - Correct labels in demonstrations matter little; the label space, the input distribution and the format matter (Min et al., EMNLP 2022, 2202.12837 [A]).
  - Demonstration order can move results "between near state-of-the-art and random guess performance", at every size (Lu et al., ACL 2022, 2104.08786 [A]).
  - Many-shot ICL keeps improving "with thousands of demonstrations" for large label spaces; "grouping of same-label examples negatively impacts performance" (Bertsch et al., NAACL 2025, 2405.00200 [A]).
  - SimpleJev's development search picked, for Qwen3.8-27B, a format with "strict decision rules, worked examples" and a fixed three-line `[thinking]` prefill on choice branches: 445 of 477 on its development set. The other formats' scores for that model were not printed, and it is a selection set, not held out [R, self-reported].
- **Channel vs direct scoring.** Channel models (score the input given each label) are more stable in few-shot, with imbalanced labels and for unseen labels (Min et al., ACL 2022, 2108.04106 [A]). Each option then needs its own pass over the whole state. **Inference:** at our 2,048-token state that is k passes over the state per question. No Jev clone read uses it.
- **Where the question goes.** jqv at 32B: question before and after the state gave +3/111 on the hard tier (p=0.55) at 7.5-18x the cost. Repeating the question after the state added +1.9 MMLU points and nothing on the hard tier [R, self-reported]. State first, question last is not measurably worse.
- **State size.** Jev's jaggedness #7 ("Large Irrelevant State") [H, vendor]. Ours: `reads_package` 0.888 on the step alone, 0.784 on 2,048 tokens of history, 0.410 on the whole history [ours, pagoda-h4, `decide_turn.py` docstring].

## 2.5 Constrained decoding

- Format restrictions (JSON mode and similar) cause "a significant decline in LLMs reasoning abilities", and stricter constraints degrade more (Tam et al. 2408.02442 [A]).
- Our readout reads the pre-sampling distribution and renormalises over the labels (`decider_bonsai` docstring: a grammar or `logit_bias` acts after the distribution that is read). That is equivalent to masking to the labels, as Cygnet does with vLLM ("logits masked to the option letters") [H, self-reported]. Nothing is generated, so the reasoning-degradation result does not apply.
- **Inference:** a grammar would add nothing to a one-token read. `label_mass` is the diagnostic that a constraint would hide.

## 2.6 Reasoning before the answer: does thinking help a decider at this size?

| evidence | finding | label |
|---|---|---|
| JevBench, same systems with and without thinking | djev hard 69.5% to 77.7% with thinking; calibration axis 65.4 to 92.7; but 72 of 534 requests "exhausted the output budget without a parseable distribution". OpenJev thinking (think=512) Intelligence 88.0, hard 78.2% | [R, independent] |
| JevBench reasoning APIs | GPT-5.6 Luna (low effort) hard 94.5%, DeepSeek V4.1 Flash 95.0%, against 74-76% for every no-reasoning readout, Jev's included | [R, independent] |
| reflex teachers | Qwen3.5-4B thinking 768 tokens: standard 72/72, judge_hard 0.94 vs 0.65, but long_policy 0.53 vs 0.58 and 20-40 s per item | [R, self-reported] |
| reflex disagreement-gated reasoning | 34 escalations on 231 items; standard 0.917 to 0.958, hard 0.685 to 0.658; p95 42 s. "The disagreement gate works as a gate; what it routes to is not yet a better judge on hard items." | [R, self-reported] |
| Explicit Reasoning Makes Better Judges (2509.13332, NeurIPS 2025 workshop) | Qwen3 0.6B/1.7B/4B on RewardBench: thinking "approximately 10% points higher accuracy with little overhead (under 2x)"; ICL, rubrics, references and n-best did not close the gap | [A] |
| Reasoning Models Better Express Their Confidence (2505.14489, NeurIPS 2025) | "strictly better confidence calibration than their non-reasoning counterparts in 33 out of the 36 settings"; removing slow thinking drops calibration | [A] |
| To CoT or not to CoT (2409.12183, ICLR 2025) | CoT helps "primarily on tasks involving math or logic"; on MMLU, direct answers match CoT unless an equals sign is involved | [A] |
| Granite Guardian 3.3 8B card | no_think vs think: harm F1 0.81 vs 0.79; RAG hallucination 0.761 vs 0.765; function-calling hallucination 0.74 vs 0.71 | [H, vendor] |
| Calibration Drift Under Reasoning (2606.11211) | ECE falls then rises with reasoning budget on Llama-3.1-8B; 47 questions; 70B inconclusive | [A], weak |
| SKILLS-RESEARCH 2.6 | past ~6-8k tokens gains stop and flips turn negative (math); small models overthink more | [H] (SR) |

**Inference for our decider.**
- The questions we ask (does the user want something built, which phase, did the stopped turn only state a next step, which skill fits) are judgement-shaped, not arithmetic-shaped. Thinking helps judgement-shaped items (reflex standard tier, RewardBench, JevBench judge_hard) and not long-document or numeric ones.
- Cost on our card: at the decode rates AGENTS.md records for this engine (roughly 23-62 tok/s depending on context and neighbours), a 512-token thought is roughly 8-22 s (a derivation, not a measurement). That is not affordable per request, but it is affordable for the few decisions whose two orders disagree.
- The Jev clones deliberately keep reasoning off the serving path (reflex's VISION) because of the p95.

## 2.7 Reading the weights: linear probes on hidden states

- A decision is often linearly readable before it is generated. "Which tool it needs" is predictable from hidden states before generation, on eight instruction-tuned models from 4B to 27B. Error detection AUROC is 0.61-0.78, and it beats a first-token confidence baseline on three of four models (2605.07990 [A]).
- Probes generalise poorly across datasets: error detectors "fail to generalize across datasets"; truthfulness encoding is "not universal but rather multifaceted" (Orgad et al. 2410.02707 [A]).
- Probes can calibrate judges: Brier-trained linear probes on reasoning judges' hidden states give "superior calibration", ~10x less compute, "generalize robustly to unseen evaluation domains", but are "conservative" on easy data (Radharapu et al. 2512.22245 [A]).
- A probe decides whether to retrieve (Probing-RAG, NAACL Findings 2025, 2410.13339 [A]): fewer redundant retrievals on five QA sets. No numbers read.
- A mid-layer probe "recovers most decisions of a strong guard model" at negligible cost (2606.10487 [A]).
- Ours, for scale: E1 (logistic heads on Qwen3-Embedding-0.6B vectors, 289 labels) beat Laya's head 104 vs 80 of 120 held-out (p=0.0001) and repeated on a second set (110 vs 86 of 141) [ours, `docs/E1.md`]. That is an embedder, which the operator has asked to avoid here.
- **What it would take on Bonsai:** engine patch 0004 (per-request last-token hidden state from a generation server; `docs/CLM.md` "Later options"). Today `--embeddings` restricts the whole server and turns off MTP.
- **Inference:** the single-token readout already IS a fixed linear probe, the LM head's rows for the label tokens. A logistic calibrator over those few logits (plus the two orders' disagreement and `label_mass`) is the cheapest "trained readout". It needs no engine change and tens to hundreds of labels, not thousands.

## 2.8 Reward models, verifiers, LLM-as-judge: what transfers

- **Position bias is the constant.** An evaluator could be made to prefer Vicuna-13B over ChatGPT "on 66 over 80 tested queries" by order alone; Balanced Position Calibration (aggregate over orders) mitigates it (Wang et al. 2305.17926 [A]). GPT-4 judges reach ">80% agreement" with humans, with position, verbosity and self-enhancement biases (Zheng et al., NeurIPS 2023 D&B, 2306.05685 [A]). **Transfers directly:** average over orders.
- **Yes-token probability as the score.** GenRM scores with next-token prediction, and majority voting over CoT verifications adds test-time compute; GSM8K Best-of-N 73% to 93.4% (Zhang et al., ICLR 2025, 2408.15240 [A]). Granite Guardian's risk score comes from yes/no token logprobs [H, vendor]. Qwen3-Reranker is a yes/no judge (SKILLS-RESEARCH 2.2). On JevBench, per-option yes/no rerankers as deciders scored low: Qwen3-Reranker-4B Intelligence 64.0; bge-reranker-v2-m3 6.3 [R].
- **Committees and cascades.** JevBench's combination study [R, independent]: a calibration-weighted committee (Jev + SemIf + djev) improved calibration (87.12) but not the composite; majority voting was "materially worse" than probability averaging. A confidence cascade (classifier.dev Fast then Jev at a 0.42 threshold) escalated 2.97% held out and kept 99.61% of Jev's accuracy.
- **Inference:** average probabilities, never majority-vote; escalate on low confidence. Both are about orders and gates, not new models.

## 2.9 Quantization and calibration

- Quantization lowers confidence in true labels, most on items where full precision was already unsure; ECE up to ~3 pp worse (Proskurina et al. 2405.00632 [H] (SR)).
- Domain-level confidence profiles do not correlate across precisions (rho = 0.00) while ranking quality does (rho = 1.00) (Cacioli 2604.08976 [A] (SR)).
- The direct datum for our base model: Qwen3.8-27B's letter readout served NVFP4 on SGLang, "~208 ms warm but costs calibration (ECE 0.061 → 0.087)" [R, reflex results index, self-reported].
- Quantized logits are not batch-invariant; near-ties can flip (2607.17283 [H] (SR)). Our `TIE_BAND` = 0.0034 is the measured size of that effect on our stack (`bench/decider/bonsai_decider.py` `batching`, n=100, one run) [ours].
- **No public study measures label-logprob calibration under 1-2-bit or ternary quantization** (still true; searched again).
- **Inference:** Bonsai's ranking of options should survive; its probability LEVELS should not be trusted until calibrated on its own outputs. Any threshold imported from a bf16 result (reflex's disagreement bands, Jev's 0.9 / 0.5) is a starting point to re-fit, not a number to use.

---

# Part 3. The open competitors

## 3.1 The Jev-class field on JevBench v1.2/v1.3 (534 decisions; one release, one scoring)

All [R] from `RESULTS-v1.2.md` (titled "JevBench v1.3.0 — results"). Intelligence = chance-corrected accuracy over tiers (hard 30%, easy 14%, standard 28%, judge 28%). Calibration = hard tier only: mean of 100×(1−ECE/0.5) and fidelity to exact gold distributions on 20 items. Latency is raw p50 on the benchmark's own hardware and network (Germany to rented GPUs); self-hosted rows carry a ×2 + 0.15 s "assumption, not a measurement" in the Speed axis.

| system | what it is | Intel. | Calib. | hard | raw p50 | licence / weights |
|---|---|---|---|---|---|---|
| Jev 1.13.0 | closed API | 85.7 | 82.7 | 74.1% | 0.65 s | closed |
| **reflex-27b** | frozen **Qwen3.8-27B**, 2 orders averaged, T=1, no training | **85.8** | **86.2** | **75.9%** | 1.89 s (H100 NVL) | MIT code; Qwen weights Apache-2.0 |
| **SimpleJev Qwen3.8-27B** | frozen Qwen3.8-27B, answer-token logits, public demo | 84.7 | 81.1 | 75.0% | 1.01 s | open code (featherless-ai/simple-jev; licence line not found in README) |
| **LitJev Qwen3.8-27B** | frozen Qwen3.8-27B, "no training and no calibration file" | 82.4 | 83.5 | 73.2% | 2.03 s | open |
| djev (Maisa) | DiffusionGemma-26B-A4B, one-step readout, no new weights | 82.7 | 65.4 | 69.5% | 0.24 s | Apache-2.0 runtime and weights |
| Winnow-12B Q8 | Gemma-3-12B-class, private training corpus | 82.0 | 72.0 | 70.9% | 0.23 s | weights submitted as GGUF; corpus private |
| decider-35b-a3b (Mapika) | Qwen3.6-35B-A3B, trained readout, FP8 | 79.6 | 71.5 | 65.5% | 0.29 s | published weights |
| jqv | stock Qwen3-32B, letter logits, one temperature (3.02, from 400 MMLU items) | 79.3 | 79.0 | 64.5% | 0.75 s | Apache-2.0 |
| SemIf | frozen Qwen3.5-4B, compact prompt, uppercase-letter logits | 79.0 | 72.6 | 59.5% | 0.20 s | open |
| reflex 4B | Qwen3.5-4B + LoRA (mix3) + per-primitive calibration | 80.1 | 75.2 | 63.2% | 1.80 s | MIT; Apache-2.0 weights |
| Bespoke Nimble 9B | Qwen3.5-9B + LoRA | 77.9 | 65.3 | 65.5% | 0.39 s | open |
| jev-local | frozen Qwen3.5-9B, mean log-probability per option (one pass per option) | 70.8 | 68.7 | 59.1% | 1.05 s | open |
| system-one (Goedecke) | Qwen3-8B | 70.3 | 36.8 | n/r | 0.17 s | open |
| Qwen3-Reranker-4B | per-option yes/no reranker | 64.0 | 67.0 | 50.0% | 0.13 s | Apache-2.0 |
| Laya (421M) | ModernBERT typed decisions | 45.8 | 62.5 | n/r | 0.79 s (CPU) | open |
| GLiNER2 large (Fastino) | schema classifier | 40.1 | 24.3 | n/r | 1.10 s (CPU) | open |
| GPT-5.6 Luna (low) | reasoning API, verbalized under a JSON schema | 95.3 | 89.8 | 94.5% | 0.97 s | closed |
| DeepSeek V4.1 Flash | API, verbalized | 94.3 | 96.7 | 95.0% | 1.42 s | API |

(n/r: not read in the rows consulted.) Also on the board: a partial run of "Qwen3.8 27B" as a general model through a hosted TEE provider, JSON-verbalized, Intelligence 67.4 on the tiers it finished (partial: fewer than 95% of decisions). A generative, verbalized 27B did far worse than the same weights read at the letter logits (84.7-85.8). **Inference:** the readout, not the weights, is the difference.

**Later releases** [R]: v1.4.2.2 top five by composite are Imajev-4B 67.37, Plumb-4B (JevK5 v0.2 + LoRA) 65.84, decider-4b v2 64.13, Jev 63.29, JevK5 v0.2.0 62.04. The v1.5.1 page [H] puts Cygnet (frozen Gemma-4-12B-it) first at 73.7 (Intelligence 71, Calibration 87). Composite ranks reward speed and cost; our use does not.

**Decision Index** (apolinario/decision-index; 132,422 requests, 37 benchmarks, 5 areas) [H, independent kit]: Reflex 27B scored 56.23 "above the kit's Jev reference at 55.74" (reflex results index [R, self-reported]). The live leaderboard was not read.

## 3.2 The clones' recipes, and ours

| | reflex | jqv | SimpleJev | Cygnet | SemIf | **ours** (`decider_bonsai` / `decide_turn`) |
|---|---|---|---|---|---|---|
| backbone | frozen Qwen3.5-4B / Qwen3.8-27B | frozen Qwen3-32B | frozen Qwen3.8-27B etc. | frozen Gemma-4-12B-it | frozen Qwen3.5-4B | Bonsai 2 27B (ternary PTQ of Qwen3.8-27B) |
| state | encoded once, shared | shared prefill, isolated branches (block mask) | common prefix KV reused | | | state message once per turn, questions in their own message after the checkpoint |
| labels | letters; yes/no as a lettered pair | option letters | letters | letters, logits masked | uppercase letters | letters (A..Z); yes/no as lettered pair for `mc_avg` |
| orders | **2 distinct, averaged** | `perm_avg` optional | not mentioned | 1 | reverse-order stability test | 2 for facts; **1 rotation for `choose()` / `judge_stop`** |
| prior correction | tried, **rejected** | temperature only | none | one temperature | none | **content-free label prior divided out in `choose()`** (`rotated_idprior`) |
| temperature | none shipped; fit on own workload | one, fitted on MMLU | none | one | none | none |
| thinking | off (offline tool only) | off | off; fixed `[thinking]` prefill lines for choice | off | off | off |
| training | none (all adapters rejected) | optional targeted LoRA | optional (RFDT) | none | none | none |

## 3.3 Distilled 4B specialists (could run on the A4000)

- **JevK5** (ngrok-adhoc/jevk5; Apache-2.0): Qwen3.5-4B + a LoRA "Distilled from Qwen3.6-27B with thinking", 3,272 teacher questions plus 3,272 human-labelled items (MMLU-Pro, WANLI, MultiNLI, BoolQ, banking77, ARC, CommonsenseQA). Public-item hard tier 0.739, ECE 0.066. H100 p50 13.5 ms easy/standard, 30 ms hard. GGUF not mentioned [H, self-reported]. It sits on the board under JevK5 v0.2.0 / v0.3 and as the base of Plumb-4B.
- **SemIf / OpenJev** (TheoLeeCJ): frozen Qwen3.5-4B; "a softmax only to the logits of fixed uppercase answer tokens", zero generated tokens [H, self-reported].
- **decider-4b v2, decider-2b** (Mapika): Gemma/Qwen bases with a trained one-pass readout; published weights [R, board rows].
- **Hardware (derivation, not measured).** A 4B at 8 bits is about 4.0e9 × ~8.5 bits ≈ 4.3 GB of weights plus KV. It would fit the A4000 beside the 0.6B embedder, but it would compete with image generation and `bonsai-vision` under gpu_room (`docs/CLM.md`: the A4000 had 11.3 GB free beside retrieval). A 12B at Q8 (Cygnet, Winnow) is about 12-13 GB and would not co-reside with a draw.
- **What they would buy:** latency off the main card and parallelism with main. **What they would cost:** accuracy. In the same benchmark release the frozen 27B readouts lead every 4B row on Intelligence (82.4-85.8 vs at most 80.1) and on hard-tier accuracy (73.2-75.9% vs 56.8-63.2%) [R]. JevK5's self-reported public-item hard tier (0.739) is a different measurement and is not comparable.

## 3.4 Other open decision models (not Jev-class)

| model | size / base | licence | what it decides | reported | fit for us |
|---|---|---|---|---|---|
| **Arch-Router-1.5B** (Katanemo; 2506.16655) | Qwen2.5-1.5B-Instruct | "Katanemo license" (the arXiv paper is CC BY 4.0; the MODEL is not) | a route from user-defined domain/action descriptions; generates `{"route": ...}` or `{"route": "other"}` | "SOTA ... outperforming top proprietary models" on its conversational sets; no numbers in the abstract or card read | generative (no distribution) unless re-read at logits; licence to check [H, vendor] |
| **RouteLLM** (ICLR 2025) | MF, BERT, Llama-3-8B classifier routers | Apache-2.0 (code) | strong-vs-weak model routing | BERT and causal-LLM routers "perform close to random" on Arena data alone; MF best with augmentation | wrong decision (model choice, trained on preference data) [P] (SR) |
| **Granite Guardian 3.3 8B** (IBM; 2412.07724) | Granite 3.3 8B | Apache-2.0 | risk yes/no from token logprobs; **custom criteria** ("bring their own criteria") | harm F1 0.81 no_think / 0.79 think; RAG hallucination 0.761/0.765; AUC 0.871 / 0.854 (paper) | the closest open "custom yes/no judge" with a logprob score; 8B would contend on the A4000 [H, vendor] |
| **Qwen3Guard-Gen** 0.6B/4B/8B | Qwen3 | Apache-2.0 | Safe/Controversial/Unsafe plus categories; generated | "state-of-the-art" on safety sets | fixed policy; no custom criteria mentioned [H, vendor] |
| **Prometheus 2** 7B / 8x7B | Mistral | Apache-2.0 | 1-5 rubric score or A/B preference, after written feedback | "highest correlation and agreement with humans ... among all tested open evaluator LMs" (EMNLP 2024) | writes feedback first (generative); a rubric grader, not a router [A]/[H] |
| **Laya** (421M, ModernBERT) | | open | typed decisions | JevBench Intelligence 45.8 | retired by E1 [ours, `docs/E1.md`] |
| **CLM-v0.1-8B** | Qwen3-8B embedder + heads | Apache-2.0 | state/option cosine choice | ours: 445/464 on the daily eval vs the in-sample stub 464/464 | embedder-based; outside this brief [ours, `docs/CLM.md`] |

## 3.5 Would any of them beat Bonsai's own logprobs? (honest verdict)

- **On accuracy and calibration, probably not, if the ternary tax is small.** The best open results on the only independent decision benchmark come from our base model read exactly the way we read it (Qwen3.8-27B, letters, two orders, thinking off): Intelligence 85.8, Calibration 86.2, hard 75.9% [R]. No 4B-12B specialist beats that on those two axes in the same release.
- **The unknown is the quantization.** Bonsai is a ternary PTQ. Public ternary-PTQ results on Qwen3 lose most on reasoning and least on context-reading (SKILLS-RESEARCH 1.2). The only direct datum, NVFP4 on this base, cost ECE 0.061 to 0.087 [R]. Until measured (Part 4, item 2), the claim "Bonsai's readout ≈ reflex-27b's" is inference.
- **On latency, a 4B on the A4000 would win.** Our turn decider costs a request p50 2.1 s, p90 4.1 s before main (pagoda-h4 replay, 57 requests; facts p50 1.4 s) [ours, `bonsai.json` `h4turns`]. The one live `judge_stop` in the relay logs took 5.3 s and processed 2,156 tokens (pagoda-p1-pi-xhigh) [ours]. Published 4B clones run 0.2 s p50 on datacentre GPUs [R]; on an A4000 unmeasured.
- **Inference:** the principled comparison is one held-out set of OUR decisions run through Bonsai's readout and one 4B clone, compared on accuracy, Brier/ECE and wall time. Swap only if the 4B is not significantly less accurate. Nothing read predicts that it would be.

---

# Part 4. What this means for our decider

## 4.1 Where our design stands against the evidence

| our choice | verdict | evidence |
|---|---|---|
| one forward pass, label logits at an `Answer:` prefill, thinking off, renormalised over labels | **Supported**; it is the field's consensus recipe | every top open row in 3.1; Robinson (MCSB); jqv finding 1 ("reproducible on an ordinary open decoder") |
| state first and shared, question in its own message, prefix cached on one slot | **Supported** | jqv finding 2 (the speed-up "comes from sharing the state, not from 'not generating'", 53-73x at long states); jqv finding 9 (question-first +3/111 n.s. at 7.5-18x cost). Ours: 100 questions cached 50.5 s vs uncached 116.8 s; processed tokens 16,432 vs 70,640 [ours, `batching`] |
| the current turn/step as state, head+tail cut at 2,048 | **Supported** | Jev jaggedness #7; ours: reads_package 0.888 step vs 0.410 whole history |
| neutral-letter choice in 2 orders for build intent and phase (`mc_avg`) | **Supported** | reflex (2 orders, lettered yes/no), jqv (`perm_avg`), PriDe/PSC; ours: intent 0.848 to 0.869 |
| **`choose()` and `judge_stop`: one count-dependent rotation plus a divided-out content-free label prior** (`rotated_idprior`) | **Contradicted** | reflex position-prior (rejected: hard ECE 0.086 to 0.124; content-free prior "large and ragged"); reflex order-averaging (27B hard 0.703 to 0.766 with 2 orders); ours: cc hurt every yes/no question. The one live `judge_stop` record shows the prior flipping the decision: raw pick `next_step`, calibrated pick `other` (0.637 vs 0.355), so the continuation did not fire [ours, octo relay pagoda-p1-pagoda-pi-xhigh-1]. Truth unknown for that row. |
| content-free (`CONTEXTUAL`) division as a package-detection veto (`mc_avg_cc`) | **Keep, as measured** | the one place it helped (kept 33/33 true detections, vetoed 2 of 4 false) [ours, in-sample]; asymmetric vetoes are where a "yes"-leaning correction is wanted |
| `TIE_BAND` = 0.0034 as the only abstention | **Supported as a floor, insufficient as a gate** | it is the batch nondeterminism floor (measured). `calib` recorded 0 ties on intent/phase. reflex's disagreement bands show where accuracy actually drops (< 0.05: 0.908; > 0.30: 0.250) |
| no temperature, no threshold | **Supported for now** | reflex, jqv: temperatures do not transfer; fit only on own labels when raw ECE says so. Our build-intent ECE is 0.055-0.081 for the letter forms (`mc_avg` 0.070; in-sample, n=99) |
| a disagreement log with ids and labels, never text | **Supported in intent, broken in practice** | it cannot be joined to the corpus (0 of 222 request ids match); only disagreements are logged, so agreement accuracy is invisible |

## 4.2 Design changes, ranked by expected value

Each change names its evidence and how to measure it on our data. None is measured on Bonsai. Per PROTOCOL, each needs a measurement that survives a repeat before any claim, and the operator decides. "Held-out" below means labels not used to choose the change.

**1. Make the decider's decisions labellable: log every decision with a corpus join key, then hand-label a stratified sample.** (Highest EV: every other change is judged on it.)
- What: record every decision (not only disagreements) with the corpus turn id (ids only, as now), the question, both orders' distributions, `label_mass`, the margin and the rule's answer. Then draw a stratified sample for the operator to label: disagreements first (the 115 verify-vs-implement rows), then agreements at low margin, then high margin.
- Why: all our decider numbers are in-sample (`bonsai.json` says so). reflex's and jqv's central lesson is that tuning on your development set does not transfer (GEPA, LoRA, temperatures), and Sclar shows format effects dominate single-format results. 2609.27678 [A] shows aggregate accuracy hides decision flips.
- Measure: labelled n per question; accuracy, Brier and ECE of decider vs rule on the labels, McNemar on the discordant pairs; label client traffic (`d427…`, `bb96…`) separately from test and replay rows.

**2. Measure the ternary tax against the published bf16 readout of the same base model.**
- What: run the JevBench public items (the harness and the 72 original decisions are MIT; check the hard tier's file licence before use) through `decider_bonsai` in its `mc_avg` form, and compare with reflex-27b's published public-item numbers on the same items (easy 1.000, standard 0.958, hard 0.766, hard ECE 0.061) [R].
- Why: it is the only external, not-in-sample yardstick available, and it isolates the one unknown in 3.5. Needs a data download and GPU time, so the coordinator decides.
- Measure: per-tier accuracy and hard ECE, the same metric definitions as JevBench; run twice (the batch-invariance floor says small differences are noise).

**3. Read `choose()` and `judge_stop` in two distinct orders and average; stop dividing by the content-free label prior.**
- What: replace `rotated_idprior` with the `mc_avg` form the fact questions already use. Two distinct orders, "None of these" never last in either. The id prior goes behind a switch, off.
- Evidence: reflex order-averaging and position-prior (2.1); jqv `perm_avg`; Balanced Position Calibration (2305.17926); our own `calib`; the live `judge_stop` flip.
- Cost: the options live in their own user message, so a second order is a second options message. The state prefix stays cached, but each order re-processes its options block; measure the added tokens and ms per question.
- Measure: first, without labels, estimate Bonsai's letter prior with PriDe's permutation estimator on logged states (no labels needed) per option count. If it is near uniform, as reflex found for the 4B outside yes/no, the prior division has nothing to remove. Then, on the labelled set from item 1 (the daily eval's MUST / MUST-NOT for `choose()`, labelled stopped steps for `judge_stop`), count flips vs the current form and their correctness.

**4. Add an abstain band from the two orders' disagreement, and treat low `label_mass` as abstain.**
- What: total-variation distance between the two orders' distributions above a threshold means the decider abstains and the rule (or a cascade, item 6) decides. Likewise a `label_mass` below a floor, meaning the model wanted to say something other than a label ("My Answer is C").
- Evidence: reflex's disagreement table (n=1,200 external: < 0.05 accuracy 0.908, 0.15-0.30 0.691, > 0.30 0.250); JevBench's cascade (escalating ~3% kept 99.6% of accuracy); Jev's own confidence-routing guidance (vendor).
- Measure: accuracy of the decisions kept vs abstained, and the abstain rate, on the labelled set. The threshold is FITTED there (reflex's bands are bf16 4B numbers and do not transfer, 2.9), and reported with its n.

**5. Calibrate per question on our own labels, only where raw ECE says so.**
- What: one temperature, or a 3-5 parameter logistic over [log p of the pick, margin, two-order disagreement, `label_mass`], fitted with a proper scoring rule (log loss or Brier) per question on held-out labels. Refit whenever the model, template or precision changes.
- Evidence: Guo et al. (ICML 2017, 1706.04599 [A]): temperature scaling is "surprisingly effective"; reflex and jqv: fit on your own workload, not on MMLU; quantization (2.9): levels do not transfer across precision; Radharapu (Brier-trained probes calibrate judges).
- Measure: cross-validated ECE and Brier before/after on the item-1 labels; keep only if held-out ECE improves AND accuracy does not fall. Our build-intent letter forms are already at ECE 0.055-0.081 in-sample (`mc_avg` 0.070), so this may be a no-op for intent; the h4 step questions were worse (reads_package `mc_avg` 0.134, scratch_write 0.158) [ours, `calib`].

**6. A cascade, not always-on thinking: re-ask with thinking only when the fast read abstains (item 4).**
- What: for abstained decisions only, re-ask the same question with thinking on and a bounded budget, reading the label after `</think>`. Else fall back to the rule.
- Evidence: for (JevBench thinking rows; 2509.13332 +10 points; 2505.14489 33/36); against (reflex's gated reasoning did not help the hard tier and p95 42 s; Granite Guardian think ≤ no_think; Sprague: CoT mainly math). Our questions are judgement-shaped, where thinking helped most.
- Cost: seconds per escalation on the main card (2.6 derivation), and the child slot is shared with the second brain.
- Measure: on the abstained subset of the labelled set only, accuracy with thinking vs the rule vs the fast read, and seconds per decision. Adopt only if the lift per second beats the rule fallback. The budget is set from the measured thinking lengths, not chosen.

**7. Two prompt changes that transferred elsewhere: "Evidence / Criterion" headings, and worked examples in the cached system prefix.**
- What: (a) replace `MATERIAL:` / `QUESTION:` with evidence/criterion framing; (b) put a few worked examples per question type in the system message, which is byte-stable and so cached across calls (label space, input distribution and format matter more than label correctness: Min et al. 2202.12837; don't group same-label examples: Bertsch).
- Evidence: reflex's controls ((a) transferred to external sets: toxic-chat ECE 0.104 to 0.059 alone); SimpleJev's development pick for Qwen3.8-27B ("strict decision rules, worked examples"; dev set only). Against: format effects are high-variance (Sclar, reflex ±8), and GEPA-style search overfits.
- Measure: only on labels NOT used to pick the wording; accept a change only if it holds on two disjoint labelled sets (reflex's rule: "external sets first"). Low-to-medium EV because of the variance.

**8. Idle-time silver labels from Bonsai with thinking, for monitoring and for fitting item 5. Never for training weights.**
- What: at idle (the worker's idle gate), re-ask logged decisions with thinking on as a teacher. Use its distributions as silver labels to monitor drift and enlarge item 5's fitting set. Estimate the teacher's own accuracy first on the operator's hand labels.
- Evidence: JevK5's teacher was Qwen3.6-27B with thinking; reflex chose the 27B as teacher and found the thinking 4B "near-perfect on judgement items". Both then TRAINED students, which reflex found did not transfer; we would only calibrate and monitor.
- Measure: teacher vs hand labels (accuracy, n). If the teacher is not clearly better than the fast read on the hand labels, drop it.

**9. Probes on Bonsai's hidden states (later; needs engine patch 0004).**
- Evidence: 2.7. Decisions are linearly readable (2605.07990), probes calibrate judges (2512.22245), but they generalise poorly across datasets (2410.02707). Our own E1 shows logistic heads work with a few hundred labels, on an embedder.
- EV: lower than items 1-5 until the label set exists. It needs an engine patch, and its generalisation risk is exactly our situation (few, narrow labels).
- Measure: only after item 1, as a head on the same labels, against the calibrated readout, with McNemar on held-out.

**10. A 4B clone on the A4000, only if latency on the main card becomes the binding constraint.**
- Evidence: 3.3 and 3.5. The 4B clones trail the frozen 27B readouts on Intelligence by 4.6-5.7 points (against reflex-27b and SimpleJev 27B) and on the hard tier by 10+ points in the same release.
- Measure: the item-1 held-out set through JevK5/SemIf (Apache-2.0) vs Bonsai: accuracy, Brier, wall time on our A4000 under gpu_room. Swap only on non-inferiority.

## 4.3 What NOT to do (each tried by someone and measured to fail)

- Divide out a content-free or fitted letter prior when the prior is small (reflex position-prior; our `calib`).
- Import a temperature or threshold from another dataset or precision (reflex §4; jqv finding 4; Cacioli).
- Ensemble wordings that mix Yes/No-token readouts with letters (reflex: pulls toward "yes").
- More than two orders without a measurement (reflex: no gain seen).
- Majority vote instead of averaging probabilities (JevBench combinations).
- Always-on thinking in the request path (reflex p95 42 s; djev thinking 72/534 unparseable).
- Fine-tune on public classification mixes (reflex: four adapters rejected; jqv: no help at 14B/32B).

---

# Part 5. Trained decision heads: Unsloth's guide, Cloudflare Clef, Liquid d1 (added 2026-10-07)

Operator, 2026-10-07 (verbatim): "We don't need clef, too big, I'd rather have the smaller model, download the unsloth 2b, but my point was can we do this with one of our models. I think a part of jjava for me was we can use a model we already have in our stack, but for someone with tighter memory it would be nice to fit say bonsai and a smaller decider on the same card." And: "Add this to the knowledge base and ensure we are following this training guide or close to it."

Approved downloads (same message): Unsloth (pip, in its own pinned venv, not the `irm | iex` installer; versions in `locks/unsloth.lock.txt`), the Qwen3.5-2B base model, and the datasets the guide uses. Anything else: ask. Licences never block (operator, 2026-10-07).

Reading labels as in "How to read the citations" above. [R] here also covers files read from a pip-installed package (`unsloth 2026.10.2`) and raw model/dataset cards fetched as text. Results of OUR runs of this recipe are in `bench/decider/unsloth/` (RESULT files); this part records what the sources say.

## 5.1 The Unsloth guide, read raw [R, vendor]

Page: <https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth> (raw HTML text read 2026-10-07; the page says "Last updated 7 hours ago", so it moves). Announcement: x.com/UnslothAI/status/2107868866361930236 (script-rendered; not read). The fetch tool's summary of this page left out the per-source columns and the "r=64, one epoch" statement and gave only the code's r=16 / 2-epoch recipe; the figures below are from the raw text.

**Method.** A Clef head: "Unsloth puts your input in one prompt, followed by every question and its options. The LLM reads it once. A small head, the same design as Cloudflare's Clef, looks at the LLM's output over each question and option and scores every option, deciding all the questions together. It never writes text." The backbone is LoRA-tuned; the head is new (random init) for a plain LLM.

**Reported results** (the page's two tables; "Test sets were decontaminated against the training data"):

| model | typed-decisions | BANKING77 | CLINC150 | holdout acc (3,000 rows) | VRAM | time |
|---|---|---|---|---|---|---|
| Qwen3.5-0.8B | 36% -> 73% | 7% -> 74% | 19% -> 76% | 78% | 4 GB | 42 min |
| Qwen3.5-2B | 33% -> 78% | 1% -> 58% | 1% -> 62% | 81% | 8 GB | 40 min |
| Llama 3.2 3B | | | | 79% | 4.1 GB | 30 min |
| Gemma 4 E4B | | | | 77% | 14.4 GB | 49 min |
| Laya (fine-tuned) | | | | 77% | 2.5 GB | 10 min |

The operator's summary matches the first table. The page does not say which GPU produced the times (it names an L4 for a 60-step run of Qwen3.5-4B: 76% in 10 minutes), the training-set size, or the number of rows per source.

**Data.** "We used a mix of 12 sources, plus typed-decisions: ag_news, arc, banking77, boolq, clinc150, commonsense_qa, mmlu, mnli, prompt_injections, snli, sst5 and wanli. The test set contained 3,000 rows: 2000 from typed-decisions, 500 from BANKING77 and 500 from CLINC150." The mixture builder ships in the package: `unsloth.models.decision_datasets` (`SOURCES`, `build_decision_mixture`, `augment_row`, `Decontaminator`) [R, unsloth 2026.10.2]. It converts each source to `{state, questions, gold}` (intent -> a `choice`, NLI -> a 3-way `choice`, boolq and prompt_injections -> a `noul`, sst5 -> a 5-level `score`, MCQ -> a `choice`), then AUGMENTS the schema per row (question ids renamed 30%, options renamed to letters/codes 30%, instructions paraphrased 40% or dropped 10%, a derived yes/no question added 50%, the state flattened to text 30%, fields shuffled, at most 24 options). MMLU trains on `auxiliary_train` only; Decision Index benchmarks (`bfcl`, `when2call`) are refused as training sources; `xlam` is in `SOURCES` but is not one of the guide's 12. The mix's row count is not stated.

**typed-decisions** (LocalLLaMA/typed-decisions, Apache-2.0) [R]: synthetic, four workflows (agent-trace review, customer service, invoice processing, security incidents), each case = one state + FIVE typed questions; train 1,200 cases (300 per workflow), test 400 cases = 2,000 decisions. Gold is the MEAN of three samples from a teacher "of roughly 4B-class capability", so a score measures agreement with that teacher: the dataset card's references are Prior (label frequencies, input ignored) 0.470, perfect factor recovery 0.704, **teacher self-agreement 0.735**, uniform 0.308. Models fine-tuned on `train` self-report 0.77-0.80 there; "scores well above 0.735 mean a model is learning the teacher's quirks". Inference: the guide's "typed-decisions 33% -> 78%" starts below the Prior (0.47) and ends 4 points above the teacher's own consistency, so most of the gain on that column is learning the dataset's label frequencies and the teacher's habits; it is not evidence of general decision skill.

**Recipe, and where the page disagrees with itself.**
- Prose: "fine-tuned LLMs with a Clef head using LoRA (r=64) for just one epoch".
- Studio steps: "To match our results, set epochs to 2, LoRA rank to 16 and learning rate to 2e-4."
- Code: `r = 16, lora_alpha = 16, lora_dropout = 0`, `load_in_4bit = True`, `max_seq_length = 2048`, batch 8 x accumulation 4, 2 epochs, lr 2e-4 cosine, warmup 10, weight decay 0.01, seed 3407, `FastDecisionModel.split_holdout` for calibration rows. Head lr 1e-4 by default (`head_learning_rate`).
- In the package: `FastDecisionModel.get_peft_model` defaults to r = 64, alpha = 64 (the table's r=64 is the library default); Studio passes a head lr of 3e-4 for a plain LLM (`FRESH_HEAD_LEARNING_RATE`, "the from-LM recipe's head rate") and the library's own default is 1e-4 [R, `studio/backend/core/training/decision_trainer.py`]. Our run therefore follows the TABLE (r=64, alpha=64, 1 epoch, then 2) and states each departure.
- Head: `default_head_config`: width 1024 when the backbone's hidden size >= 3072, else 512; 2 routing layers + 4 decoder layers, heads = width/64, feedforward 4 x width. Cloudflare's released heads are width 1024 / 2 + 4 / 16 heads / ff 4096 for hidden 5120 (Clef) and 4096 (Clef-Flash) [R, `joint_head_config.json`].
- Loss: soft cross-entropy over each question's options, optional label smoothing / Brier / ordinal terms (off by default for a plain-LLM head); calibration afterwards = per-type temperatures fitted on the held-out rows (`FastDecisionModel.calibrate`).
- Export: `model.save_pretrained(dir)` (LoRA adapters + head), `model.save_pretrained_merged(dir)` (16-bit merged), `push_to_hub`; GGUF through `unsloth.models.decision_gguf` (quantisations q8_0, f16, bf16, q6_k, q5_k_m, q4_k_m) which calls llama.cpp's Clef converter at tag `b11443` and then writes the calibration temperatures into the GGUF [R, unsloth 2026.10.2].
- Serve: Unsloth Studio, `UNSLOTH_SYSTEMONE_MODEL=/path/to/merged unsloth studio -H 0.0.0.0 -p 8888`, then Settings -> API -> Decision API; requests naming `laya`, `default` or `jev-latest` go to the model. "The model needs a GPU and loads on the first request".
- Inference reads up to 16,384 tokens; training cuts to `max_seq_length` keeping the questions and options.

## 5.2 Cloudflare Clef [R model cards and converter; H blog summary, vendor]

- **Clef** (27B, post-trained from Qwen3.8-27B, 12 safetensors shards, 27.36 GB listed by the Hub API) and **Clef-Flash** (9B, from Qwen3.5-9B, 9.41 GB); Apache-2.0; created on the Hub 2026-09-30; both multimodal (text, JSON, images, video); announcement blog.cloudflare.com/clef-decision-models [R model cards].
- Mechanism [R model card]: the backbone's final hidden states -> a "joint schema head" (a small transformer; routing layers that pull evidence from the state to each question, then decoder layers) -> one logit per allowed option per question, softmax per question, ONE forward pass, no text generated; API-compatible with Jev/SystemOne.
- Training, from the blog through the fetch tool's summariser [H]: label-smoothed cross-entropy plus a Brier loss; rank-256 low-rank adapters; "Reinforcement Learning for Calibrated Decisions (RLCD)" as a secondary objective with partial credit for adjacent ordinal choices; the summary also said the backbone was "frozen", which contradicts the adapters it names and is not relied on. Latency (median) 38.8 ms Clef-Flash, 209.3 ms Clef, 524.1 ms Jev [H, vendor].
- Cloudflare's 27B Clef shares its base (Qwen3.8-27B) with our Bonsai 2 27B. Clef itself is 27B at 16-bit (about 54 GB); the operator ruled it out ("too big").
- **llama.cpp**: PR ggml-org/llama.cpp#29831 "model: add support for clef decision model (text-only)", merged 2026-10-03 as 99b9548; it follows #29818 "add /v1/systemone API (models: laya, julia-1, lev, openjev, kev)", merged 2026-10-02. 24 files: `conversion/clef.py` (a `ClefModel(Qwen3_5TextModel)` converter that reads `joint_head_config.json` and `joint_head.safetensors`), a Clef graph (`src/models/clef.cpp`, 616 lines), a new `llama_batch_ext_set_decision_order(batch, idx, order)` call marking question (kinds 1/2/3 = noul/choice/score) and option (4) token spans, and `tools/server/server-decision.cpp`. Known limits from the PR: no vision (waits for #29622), single sequence per batch. Pre-quantised: `ggml-org/Clef-GGUF`, `ggml-org/Clef-Flash-GGUF`.
- Decision Index 0.2.1 context: the Decision Index harness's board has no Clef entrant in the file our 860-row run used (`bench/decider/decision_index/RESULT-strat860.md`); Cloudflare publishes its own board (clef-evals.workers-ai-mle.workers.dev, not read).

## 5.3 Liquid AI d1-omni-600M [R, model card, vendor; read 2026-10-07; not downloaded]

https://huggingface.co/LiquidAI/d1-omni-600M. Created on the Hub 2026-10-05 (Hub API `createdAt`); licence `lfm1.0` (card: `license: other`). The coordinator's note gave "released ~Nov 2025"; the card does not support it: its citation is dated 2026 ("Open d1: Edge decision models for text, vision, and audio") and Nov 2025 is the LFM2 technical report (arXiv 2511.23404) it also cites.

| item | card |
|---|---|
| size | 587M: a 381M shared trunk + decision head, a 94M vision encoder (SigLIP2 tower from LFM2.5-VL-450M), a 112M audio encoder (17-layer FastConformer); base LFM2.5-Encoder-350M |
| input | text/JSON, images (tiled, several), up to 30 s of 16 kHz speech; one modality set per request (images or audio, not both) |
| output | zero output tokens; `noul`, `choice`, `score` (2-10 levels); `system_one()`, `system_one_batch()`, `probabilities()` |
| context | 16,384 tokens (text, image and audio positions together); with images the state and question text is cut to 896 tokens, as trained |
| precision | trained fp32; fp16 gave the same top answer on every text (243), image (214) and audio (416) row checked; bf16 changed 0.8% of text and 1.7% of audio rows |
| calibration | text answers use per-type temperatures stored in `config.json`; image and audio answers are the raw softmax |
| format | `model.safetensors` + `trust_remote_code` Python (`modeling_d1.py`, `encoder.py`, `vision.py`, `audio.py`, `prompt.py`); no GGUF, nothing in llama.cpp |
| speed | "We don't report inference numbers ... early research release" |
| Decision Index 0.2.1 | **15.95** (scored by Liquid with the official scorer, "not leaderboard submissions"); Knowledge 8.3, Language 12.9, Retrieval 35.0, Tools 15.1, Arts 6.8; sibling **d1-3B 48.57** (Knowledge 23.8, Language 56.4, Retrieval 52.8, Tools 74.5, Arts 36.3) |
| Liquid's own suite (decisions from public benchmarks) | SQuAD 2.0 74.0, Civil Comments 95.8, MASSIVE intent 86.1, PubMedQA 61.3, BoolQ 77.7, XNLI 74.7, PAWS-X 79.5; mean 78.4 (d1-3B 82.9, Decider 4B 81.1, Decider 2B 77.1); Fast Decisions dev 76.9. HelpSteer2 left out for possible training overlap |

Reading: a small ENCODER decider with multimodal decisions in one pass (images and speech are decisions jjava cannot make, since jjava reads text through a text-only engine path, though `bonsai-vision` can describe an image first), weak on the general index (15.95 against 40-50 for the 4B-12B text deciders and 28.97 for Decider 2B), with d1-3B the stronger sibling; not servable by llama.cpp today. On the typed-decisions card a different Liquid model, `d1:free` through Liquid's API, scores 0.742 zero-shot (rank 2, behind meraGPT Decider 1's 0.768 and ahead of Jev 1.13.0's 0.727) [R, dataset card, self-reported by the dataset's authors with Liquid's API]; that is not the open d1-omni-600M.

## 5.4 OUR approach, for contrast

| | jjava (ours) | a trained Clef-style head (Unsloth, Cloudflare) |
|---|---|---|
| read | the next-token probabilities of the letters A..Z of an UNMODIFIED served model, thinking off, on a cached state (`mcp/decider_bonsai.py`) | one forward pass of the backbone, then a small trained head scores every option of every question jointly from hidden states pooled over the question and option spans |
| training | none | LoRA on the backbone (optional) + a new head, on thousands to hundreds of thousands of labelled decisions |
| calibration | per-model letter bias / temperature, two option orders averaged, thresholds tuned on our labels (`bench/decider/tune.py`) | learned (soft-label cross-entropy, Brier) and a fitted temperature per question type on held-out decisions |
| extra memory | none: the model that serves the conversation reads | a second model (or a head on the first) |
| any served model | yes: Bonsai, Flash-Next, Mirai S, `bonsai-a4000` | one trained checkpoint per backbone; the head is tied to that backbone's hidden states |
| batching | `/decide-batch` (engine patch 0042: every question of a call in one pass; built, unshipped) | the schema's questions are one sequence by design |
| measured here | JevBench public items (231), `bonsai-a4000`: Intelligence 80.46 (tiers easy 1.000 / standard 0.944 / hard 0.739), hard-tier ECE 0.056, against Jev 1.13.0's 82.25 on the same items, McNemar p 0.84 (`bench/decider/results/jevbench/bonsai-a4000-20261006-193326/summary.json`); Decision Index 0.2.1 860-row subset (not the board's index): mean skill over 37 benchmarks 41.0 against Jev 55.9 (`bench/decider/decision_index/RESULT-strat860.md`; knowledge/reasoning benchmarks far behind, tools 72); skill injection (JJAVA section 9): NEEDED against OFF AUROC 0.924, but NEEDED against AREA 0.631 and the top-belief item is the needed one 30% of the time (a random item 31%), stage 2 AUROC 0.687 | not yet measured on our sets (Part 5.5 and `bench/decider/unsloth/`) |

**What a trained head adds** (stated plainly): learned, calibrated scoring of the options as one distribution per question; one pass for all questions; the ability to learn a task the base model's letter read cannot express (here: NEED against RELEVANCE, if labelled data separates them); a small model can do it (0.8B-2B). **What jjava keeps**: no training and no labelled-data requirement to start; no extra memory; any served model, so the model that already holds the conversation decides; reads the model's whole-context understanding at 27B scale. **The evidence on whether training helps a strong frozen reader** is mixed and already in Part 4: reflex rejected four LoRA mixes and jqv found a head plus LoRA helped at 1.7B and "not at 14B/32B" (3.2 and 4.3). The operator's question for Bonsai (a head on a FROZEN ternary Bonsai; no LoRA possible) is Phase 4 of the work recorded in `bench/decider/unsloth/`.

## 5.5 Where our own runs are recorded

`bench/decider/unsloth/README.md` indexes the scripts and RESULT files: the guide's recipe on Qwen3.5-2B and its test split; our sets (JevBench's 231 public items; the skill-injection labels, relevance and need questions); serving feasibility in OUR engine (llama-bonsai2-ada) and the VRAM beside Bonsai; and the frozen-Bonsai head feasibility test. CONTAMINATION, binding on every number from those runs: the guide's mix trains on the train splits of BANKING77, CLINC150, MMLU (`auxiliary_train`), ARC and others, so Decision Index results for those benchmarks from a model trained on that mix are in-distribution and are never reported without saying so, and no test row is ever trained on.

---

## Our data, as it stands (read-only, 2026-09-29)

- `logs/decider_disagreements.jsonl`: 260 rows when last read, 2026-09-27 to 09-29 (it grows while the stack serves).
  - By account: replay (pagoda-h4, no account) 72; test traffic `f694…` 76; client traffic `d427…` 67 and `bb96…` 45.
  - Patterns: phase verify-vs-implement 115 (client 65, test 24, replay 26); build_intent decider-yes/rule-no 47 (test traffic and 2 replay rows; median top probability 0.94 at the first read, n=46); phase verify-vs-debug 28; phase debug-vs-implement 23; phase plan-vs-implement 23; phase implement-vs-debug 12; reads_package 10 (replay).
  - Phase disagreements' top probability: p10 0.48, median 0.69, p90 0.95 (n=203).
  - No row carries a truth label. Request ids match 0 of the corpus's `turn` ids (222 distinct ids checked against `events` kind `turn`).
- `index/corpus.sqlite3` (read-only): `events` 11,062 rows (4,261 turns). Each turn's `request` holds the head of the last user message, with its account and traffic class. It is the text source for labelling once a join key exists. `deep_decisions` 2,883 rows (the deep-thinking triggers, not this decider).
- Relay logs under `C:\Users\jwals\octo\logs\`: only one `x_yamadori.continued` record with a decider judgement (pagoda-p1-pagoda-pi-xhigh-1). The pagoda-h1..h6 relay rows carry no decider records.
- `bench/decider/results/bonsai.json` (2026-09-27, in-sample): the numbers quoted above (`intent`, `calib`, `h4`, `packages`, `batching`, `h4turns`).

## Discrepancies and cautions

- **reflex-27b's Intelligence.** The JevBench v1.2/v1.3 table prints 85.8 (GPT-5.6 Luna 95.3 and DeepSeek V4.1 Flash 94.3 are higher). reflex's README says "the highest Intelligence (90.5) and Calibration (86.2) on the board". The two likely come from different scoring releases or cohorts; the table is cited.
- **JevBench version drift.** The same systems' composites and ranks change between v1.2, v1.4 and v1.5 as sealed items and scoring rules change (Jev: 74.4 #1, then 63.29 #4, then 72.1 #3 on the v1.5.1 page as summarised). Only same-release comparisons are made above.
- **The v1.5.1 page summary** (benchmarkheaven, [H]) reported a "Qwen3.8 27B ... 96.8 capability" line. It was not confirmed in any raw file and is not used.
- **reflex's external-set calibration claim for two orders** was corrected by its own author: single-order pooled ECE re-measured 0.028, not ~0.055, so on external sets two orders bought ~2 accuracy points on two sets and no pooled calibration. The calibration gain rests on the public items [R].
- **JevBench's own caveats** [R]: one maintainer; latency adjustment ×2 + 0.15 s is "an assumption, not a measurement"; the hard tier was written by two frontier models; the held-out split is sent to the evaluated services ("Not public is not the same as not seen"); the adequacy cohort's majority floor is 82%.
- **Clone results on public items** (reflex, jqv, JevK5, SimpleJev) are self-reported on a development suite their authors consulted, except the rows JevBench ran itself.
- **Arch-Router licence.** The arXiv page's CC BY 4.0 is the paper's licence; the model card says "Katanemo license".

## Not verified (not relied on above)

- TypeSafe's funding and the DCVC lead (snippet only). The Vercel, MindStudio, DataCamp, LangChain and flaviocopes explainers (snippets only).
- The Decision Index live leaderboard (not rendered); Laya 16.4 vs Jev 59.5 there (repo note from an earlier pass).
- Arch-Router's, Prometheus 2's and Qwen3Guard's numbers (none in what was read); RouteLLM's details beyond SKILLS-RESEARCH.
- G-Eval's probability-weighted scoring (the abstract does not state it).
- Probing-RAG's numbers and training-set size (abstract only).
- SimpleJev's per-format scores for Qwen3.8-27B other than the selected one; its repository licence.
- Winnow-12B's and decider-4b's licences and training data.
- **Not found at all:**
  - A label-logprob calibration study under ternary or 1-2-bit quantization.
  - An official expansion of "Jev".
  - Any Jev clone measured on agent-loop decisions (all benchmarks read are single-document classification and judging).

## Sources

**Jev and TypeSafe (vendor):** <https://typesafe.ai/blog/introducing-system-one-models-and-jev>; <https://docs.typesafe.ai/llms.txt>; <https://docs.typesafe.ai/models.md>; <https://docs.typesafe.ai/confidence.md>; <https://docs.typesafe.ai/primitives/choice.md>; <https://docs.typesafe.ai/model-jaggedness/jev-1.13.md>; <https://docs.typesafe.ai/introduction/machine-learning-primer.md>; <https://docs.typesafe.ai/introduction/coding-agents.md>; <https://docs.typesafe.ai/cookbooks/skill_suggestion.md>; <https://docs.typesafe.ai/cookbooks/consistency_choice_cookbook.md>.

**Independent on Jev:** Archer Hume, <https://archerhume.com/posts/jevs-architecture-unmasked/>; Simon Willison, <https://simonwillison.net/2026/Sep/21/jev/> (one informal experiment); arXiv 2609.27678, 2609.22753.

**JevBench (independent; raw):** <https://github.com/fstandhartinger/jevbench> (README.md, RESULTS-v1.2.md, RESULTS-COMBINATIONS.md); <https://benchmarkheaven.com/jev-models>.

**Clones (self-reported; raw where marked [R]):** reflex <https://github.com/kshetrajna12/reflex> (README, docs/results/: README, order-averaging, frozen-vs-trained, position-prior, weight-classes, teachers-27b-and-4b-think); jqv <https://github.com/Octalab-Inc/jqv>; Simple Jev <https://github.com/featherless-ai/simple-jev> (README, hf-server/README) and <https://featherless.ai/blog/jev-llm-classifier-qwen3-simple-jev>; Cygnet <https://github.com/blockbrain-ai/cygnet-recipe> (snippet-level); SemIf <https://github.com/TheoLeeCJ/SemIf-OpenJev/blob/master/docs/METHOD.md>; JevK5 <https://github.com/ngrok-adhoc/jevk5>; Decision Index <https://github.com/apolinario/decision-index>.

**Papers (arXiv abstract pages unless marked):** 2108.04106 (noisy channel); 2205.10183 (prototypical calibration); 2402.14499 ("My Answer is C"); 2404.08382 ("Look at the Text"); 2207.05221 (Kadavath); 2305.14975 (Tian); 2306.13063 (Xiong); 2310.11324 (Sclar); 2104.08786 (Lu); 2202.12837 (Min, demonstrations); 2405.00200 (Bertsch); 1706.04599 (Guo); 2104.08315 (Holtzman); 2210.12353 (Robinson); 2408.02442 (Tam); 2409.12183 (Sprague); 2509.13332 (explicit reasoning judges); 2505.14489 (reasoning models' confidence); 2606.11211 (calibration drift); 2410.02707 (Orgad); 2410.13339 (Probing-RAG); 2512.22245 (probes for judges); 2605.07990 (tool calling linearly readable); 2606.10487 (streaming probes); 2305.17926 (unfair evaluators); 2306.05685 (MT-Bench judge); 2408.15240 (GenRM); 2303.16634 (G-Eval); 2405.01535 (Prometheus 2); 2506.16655 (Arch-Router); 2412.07724 (Granite Guardian). Carried from SKILLS-RESEARCH: 2102.09690, 2309.03882, 2309.17249, 2305.19148, 2310.07712, 2607.05552, 2405.00632, 2604.08976, 2607.17283, 2406.18665.

**Model cards (vendor):** <https://huggingface.co/katanemo/Arch-Router-1.5B>; <https://huggingface.co/ibm-granite/granite-guardian-3.3-8b>; <https://huggingface.co/prometheus-eval/prometheus-7b-v2.0>; <https://huggingface.co/Qwen/Qwen3Guard-Gen-8B>.

**Ours:** `mcp/decider_bonsai.py`; `mcp/decide_turn.py`; `bench/decider/results/bonsai.json`; `logs/decider_disagreements.jsonl`; `index/corpus.sqlite3` (read-only); `C:\Users\jwals\octo\logs\*\relay.jsonl`; `docs/E1.md`; `docs/CLM.md`; `docs/CLM-EVAL.md`; `docs/research/SKILLS-RESEARCH.md`.
