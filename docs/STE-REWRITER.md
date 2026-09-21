# Single-Pass, Classifier-Speed Rewriting of English into ASD-STE100 Controlled English: A Feasibility Study

**Status:** Pre-implementation feasibility research. Not peer reviewed.
**Date:** September 2026

## Abstract

We assess the feasibility of building a model that rewrites arbitrary English into
ASD-STE100 ("Simplified Technical English," STE)-compliant English in a single forward
pass, at classifier-like latency (tens of milliseconds), reliably enough to sit inline in a
request path. We find no evidence that such a system has been built or shipped: the
standard's own maintainers state plainly that no existing tool converts non-STE text into
STE, commercial STE products (HyperSTE, Congree, Acrolinx, TechScribe) are check-and-flag
tools that leave rewriting to a human, and the only "rewriters" we could find are 2025-2026
hobbyist LLM-prompting scripts (multi-second latency, unverified meaning preservation, no
reported accuracy). The task is best understood as a constrained monolingual sequence
transduction problem structurally identical to grammatical error correction (GEC) and
sentence simplification, both mature research areas with standard metrics (SARI, F0.5,
GLEU) and, critically, an architecture family — encoder-only sequence tagging
(GECToR; LaserTagger; Seq2Edits; and directly relevant, "Text Simplification by Tagging")
— that already achieves near state-of-the-art quality at 10-100x the inference speed of
sequence-to-sequence rewriting, because it performs exactly one non-autoregressive forward
pass, identical in cost profile to a token classifier. We argue this tagging family is the
only architectural class that structurally satisfies a "tens of milliseconds, single pass"
latency budget; non-autoregressive translation (NAT) approaches close most of the quality
gap to autoregressive seq2seq but still require either a separate decoder stack or several
refinement iterations, and small autoregressive seq2seq models (T5-small/base, BART-base)
do not meet the latency bar at all because decoding cost scales with output length. We
estimate, from a structural reading of ASD-STE100's 53 writing rules, that a majority of
required edits (controlled-vocabulary substitution, article/pronoun insertion, verb-form
correction, punctuation) are local token edits reachable by tagging, but a non-trivial
minority (passive-to-active voice conversion, splitting long or compound sentences into
one-instruction sentences) require structural reordering/splitting that pure KEEP/DELETE/
REPLACE tagging handles poorly without extension (bounded phrase insertion, a
split-and-duplicate operation, or a small secondary realization step). No paper we found
measures this split directly for STE; this is a reasoned estimate, not a citation, and we
flag it as the central open empirical question a prototype must answer. We propose a
rejection-sampling / self-distillation data pipeline (generate candidate rewrites with a
larger LLM, keep only rewrites a deterministic ASD-STE100 rule checker accepts, train a
small tagger on the accepted set) that mirrors established practice (STaR / rejection
sampling fine-tuning, GECToR's own synthetic-then-real training curriculum, and
Grammarly's BART-to-tagger distillation for simplification) but note that the deterministic
gate can verify *compliance* (this is checkable, since compliance is exactly what existing
STE checkers already compute) while it cannot verify *meaning preservation* — the harder,
unresolved problem that both the academic tagging literature and the existing STE-rewriting
hobbyist tools explicitly leave open. We conclude the goal is plausible as an encoder-only
tagging model in the 100-400M parameter range, but "high reliability, not best-effort" will
most likely require the tagger to be paired with a fast deterministic post-hoc compliance
check (reject/flag on failure) rather than trusting the model's single pass unconditionally,
because no published tagging system reports the near-100% compliance rate an inline,
unsupervised request path would need.

## 1. Introduction

ASD-STE100 (Simplified Technical English, "STE") is a controlled natural language
originally developed in the late 1970s for European aircraft-maintenance documentation and
now an ASD (Aerospace, Security and Defence Industries Association of Europe)
international standard, currently at Issue 9 (January 2025) [ASD-STE100]. It consists of
53 writing rules across 9 sections (word choice, grammar, sentence structure, style) plus a
controlled dictionary of roughly 900 approved words, each restricted to one meaning and one
part of speech, and roughly 1,200 words to avoid with suggested replacements
[ASD-About; TCWorld-Issue9]. STE is mandatory or recommended across civil and military
aviation maintenance documentation (EASA, FAA, CAAC; EDSTAR/EDA) and has spread into
broader technical writing and translation workflows, in part because controlled source text
measurably improves downstream machine translation quality [ResearchGate-STE-MT].

The question motivating this paper is narrow and operational: can arbitrary English be
rewritten into STE-compliant English in one model pass, at latency low enough (tens of
milliseconds) to sit inline in a request path — e.g., rewriting an agent's tool
description, an error message, or a generated instruction before it is shown to a
downstream reader or another agent — with reliability high enough that the system can be
trusted without a human in the loop for most traffic. This is a "build or don't build"
question, so we treat a well-argued negative as equally valuable as a positive: if no
architecture plausibly hits both the speed and the reliability bar, that is the useful
answer.

We proceed by (1) establishing what already exists (Section 2), (2) arguing that the task
is a special case of well-studied monolingual sequence transduction and importing its
metrics and baselines (Section 2-3), (3) reviewing every architecture family that has been
used to make sequence transduction fast, with measured numbers where they exist
(Section 2.3), (4) proposing a concrete approach and data pipeline (Sections 4-5), and (5)
laying out an evaluation plan and the risks that could sink the approach in production
(Sections 6-7).

## 2. Background and Related Work

### 2.1 Prior art: does an ASD-STE100 rewriter already exist?

**No, not as a fast, validated, ship-quality system, and the standard's own maintainers say
so explicitly.** The official ASD-STE100 "Tools for STE" page states: *"No language
checking tools will write STE text for you, nor can they convert non-STE text into STE,"*
and separately that *"ASD and STEMG do not endorse, certify, or authorize any software
tools, including AI-based ones"* [ASD-Tools]. This is a direct, current (Issue 9-era)
statement from the standard body that no accepted rewriter exists.

Consistent with that, the commercial STE tool landscape is entirely **check, not rewrite**:

- **HyperSTE**, marketed as "the leading checker tool to comply with Simplified Technical
  English" [HyperSTE].
- **Congree**'s STE Checker "finds instances of non-compliance while writing" and
  integrates into authoring tools as a linter [Congree].
- **Acrolinx** offers STE guidance/checking within its content-quality platform
  [Acrolinx-Docs].
- **TechScribe**'s term checker flags non-compliant terms against the ASD-STE100 dictionary
  [TechScribe].

All four surface violations for a human to fix; none claims to output a compliant rewrite
automatically. This matches the general academic controlled-language literature: controlled
natural languages (CNLs) are described as *writing disciplines enforced by checkers*, not
generation targets in their own right [Wikipedia-CNL; Kittredge-StyleGuides]. We found one
2025 academic paper specifically addressing STE and machine translation relevance
[ResearchGate-STE-MT-Overview], but it discusses STE as a *source-side authoring practice
that improves downstream MT quality*, not as a rewriting target for a generative model — it
does not propose or evaluate an automatic STE rewriter.

The only artifacts we found that attempt automatic rewriting are a handful of
September-2025-to-2026 open-source "Claude Code skill" / "agent skill" repositories
(e.g., `asd-ste100-skill`, `SimpleEnglish`) that combine a deterministic structural linter
(sentence length, passive-voice detection, banned constructs) with an LLM (Claude, or
Llama 3 in at least one "NLP Compliance Tool" variant) prompted to rewrite flagged
sentences [GH-asd-ste100-skill; GH-SimpleEnglish]. These are explicit about their limits.
One README states: *"The linter checks structural patterns only. It does not compare an
original text with a rewrite, verify that requirement strength stayed the same, or prove
that the rewrite preserved meaning"* [GH-asd-ste100-skill-SKILL], and none reproduces ASD's
actual ~900-word licensed dictionary, instead approximating it. None reports a compliance
rate, a meaning-preservation rate, or a latency figure — these are agent-loop tools
designed for offline document authoring assistance, invoked through a multi-second LLM
call per sentence or paragraph, not an inline low-latency service. We treat this as the
honest current state of the art for *rewriting*: a wrapper around a large general-purpose
LLM, unvalidated, and far outside the latency budget this paper is evaluating.

**Conclusion for prior art:** the checking problem is solved and commoditized; the
rewriting problem is, as of September 2026, open. Nobody has shipped a fast, validated
STE rewriter. This is a genuine gap, not a "reinventing the wheel" risk — but it is a gap
precisely because reliable rewriting is harder than checking, and the field has apparently
judged (or simply not attempted) closing it with a purpose-built model.

### 2.2 Is this just (monolingual) translation?

Yes, structurally. Rewriting ordinary English into STE is a same-language, meaning-
preserving sequence transduction under a target-side constraint set — the same shape as:

- **Text simplification** (rewrite into simpler, more readable English),
- **Grammatical error correction, GEC** (rewrite into grammatically correct English),
- **Formality/style transfer** (rewrite into a target register).

**Text simplification** is the closest semantic analogue (STE explicitly aims at
comprehensibility for non-native readers) and has mature datasets and metrics:

- **WikiLarge** (Zhang & Lapata 2017): ~296K automatically aligned Wikipedia
  simplification pairs, the standard large-scale training set [WikiLarge].
- **TurkCorpus** (Xu et al. 2016): 2,359 source sentences, 8 human references each,
  historically the standard tuning/eval benchmark, biased toward lexical paraphrase
  [TurkCorpus].
- **ASSET** (Alva-Manchego et al. 2020): same 2,359 sentences, 10 references each, written
  to cover a *wider variety* of rewrite operations (splitting, deletion, reordering, not
  just paraphrase) — closer in spirit to STE's mix of lexical and structural rules
  [ASSET].
- **Newsela**: professionally simplified news articles at 5 reading levels; higher-quality
  but license-restricted [Newsela via NLP-progress].
- **SARI** (Xu et al. 2016) is the standard metric: the mean of F-scores for words
  correctly **added**, **deleted**, and **kept**, computed against the source and multiple
  references — designed specifically to reward simplification operations rather than
  n-gram overlap alone, unlike BLEU [SARI-orig; SARI-critique].

Reported system quality: an LLM-guided (planning + summarization) pipeline evaluated at
CLEF 2025's SimpleText shared task scored SARI 42.3-43.0 on scientific-text simplification
test sets [CLEF2025-SimpleText] — useful as a general sense of where a strong 2025-era
system lands on a *harder*, domain-shifted simplification task; general-domain
Wikipedia-style simplification SOTA has historically been reported in the mid-40s SARI by
strong seq2seq/paraphrase-mining systems such as MUSS [MUSS]. We did not find a single
authoritative current SOTA number that would be safe to state as *the* benchmark to beat;
readers building an eval plan should treat 40-45 SARI as the right order of magnitude for
"strong system," not a hard target, and should establish their own STE-specific baseline
(Section 6), since STE compliance is a different, narrower objective than general
readability.

**GEC is the more load-bearing analogue**, because — like STE compliance — it targets a
**checkable rule set**, not an ill-defined "more readable" objective. Standard data: **Lang-8**
(1.04M sentence pairs of naturally occurring learner corrections), **NUCLE**, **FCE**,
**W&I+LOCNESS** (the BEA-2019 shared task corpora), and **cLang-8** (2.37M pairs, produced
by *re-labeling* raw Lang-8 with the output of a large multilingual GEC model — i.e.,
model-distilled data, discussed further in Section 2.4 and Section 5) [cLang8;
BEA2019-Task]. Standard metrics: **F0.5** over ERRANT-extracted edits (precision-weighted,
because over-correction is penalized more than under-correction) on CoNLL-2014 and
BEA-2019 test, and **GLEU** [Napoles-GLEU], a BLEU variant that scores against both source
and reference so it rewards *changes that should happen* and penalizes *source errors left
uncorrected* — a much better fit than BLEU for a "rewrite under constraints" task, and one
we recommend adapting for STE (Section 6). Representative strong results: GECToR's
best models reach F0.5 = 65.3 (single) / 66.5 (ensemble) on CoNLL-2014 and 72.4 / 73.6 on
BEA-2019 [GECToR]. These are not "solved" numbers — F0.5 in the 65-74 range means real,
frequent residual errors — which is itself an important calibration for what "reliable"
realistically looks like from this model family at comparable scale, addressed in
Section 7.

**Readability metrics** relevant to STE's own goals (short sentences, simple words) include
**FKGL** (Flesch-Kincaid Grade Level, a closed-form function of words/sentence and
syllables/word) [FKGL], though FKGL has been directly criticized as a poor
*simplification*-quality metric on its own because it can be gamed by trivial edits that
don't improve real comprehensibility [FKGL-critique] — relevant because STE compliance
checking must not be allowed to degenerate into FKGL-gaming either; compliance has to be
checked against the actual rule set and dictionary, not a readability proxy.

### 2.3 How to make it fast: architecture survey

This is the technical crux of the feasibility question, so we cover it in depth.

**Autoregressive (AR) baseline.** A standard encoder-decoder (T5-base, BART-base) decodes
one token at a time, each step attending to all previous steps; inference latency scales
with output length and cannot be fully parallelized at generation time regardless of model
size. This rules AR seq2seq out of a "tens of milliseconds, classifier-like" budget by
construction, independent of accuracy: even heavily optimized T5 serving stacks (ONNX
Runtime, TensorRT) report only **3-21x** latency reduction over unoptimized baselines
[T5-TensorRT; T5-ONNX-fastT5], and T5-base is reported roughly 3x faster than BART-base at
comparable settings [BART-vs-T5-issue] — useful relative numbers, but we could not find a
citable absolute millisecond figure for unbatched, single-sentence T5-small/BART-base
inference; any such number should be benchmarked directly rather than assumed, and even the
optimized numbers remain decode-length-dependent, unlike a single fixed-cost forward pass.

**Non-autoregressive translation (NAT).** NAT removes the sequential decoding dependency
by predicting all target tokens in parallel (usually still via a separate decoder that
cross-attends to the encoder).

- **Original NAT** (Gu et al. 2017, arXiv:1711.02281): predicts a latent "fertility" per
  source token then generates the target in parallel; reports **2-15x** speedup over a
  comparable AR Transformer depending on the number of fertility samples used, at a
  non-trivial BLEU cost relative to AR (recovered partially by rescoring with an AR model,
  which reduces the effective speedup) [NAT-orig].
- **Mask-Predict / CMLM** (Ghazvininejad et al. 2019): trains a conditional masked
  language model and decodes by iteratively remasking and repredicting the
  lowest-confidence tokens over a small fixed number of iterations; reported to close
  most of the gap, landing **within about 1 BLEU point** of a standard left-to-right
  Transformer while decoding significantly faster, and improving over prior
  fully-parallel NAT by **4+ BLEU on average** [MaskPredict]. Note this is *iterative*
  (multiple forward passes), not single-pass.
- **Levenshtein Transformer** (Gu, Wang, Zhao 2019, NeurIPS): frames generation as
  learned insertion/deletion edit operations; "comparable performance with much-improved
  efficiency" versus standard seq2seq, and — notably for this paper — a model trained for
  translation transfers directly to automatic post-editing without modification, i.e., it
  is natively an *edit* model, not just a faster translator [LevT].
- **DisCo** (Kasai et al., ICML 2020): attention-masking model that predicts each output
  token from an arbitrary subset of the others, with a "parallel easy-first" inference
  algorithm that adaptively reduces the number of iterations needed; competitive with or
  better than prior NAT while reducing average decoding time [DisCo].
- **GLAT / Glancing Transformer** (Qian et al., ACL 2021): a *training* method (Glancing
  Language Model) rather than an architecture change, that makes **single-pass** parallel
  decoding competitive; reports **8-15x speedup** with the BLEU gap to AR Transformer
  reduced to **0.25-0.9 points** — the best fully single-pass NAT result we found [GLAT].
- **CTC-based NAT**: repurposes Connectionist Temporal Classification (built for ASR's
  monotonic alignment) for translation by mapping source positions to target tokens with a
  blank symbol; a natural fit for single-pass parallel decoding, but the literature is
  explicit that translation's alignment is *not* monotonic the way speech is, which is a
  structural mismatch requiring extra machinery, and results are decoder-architecture- and
  task-dependent rather than uniformly strong [CTC-NAT; CTC-SpeechTrans].
- **DA-Transformer** (Huang et al., ICML 2022): represents hidden states as a directed
  acyclic graph so a single non-autoregressive pass can implicitly represent multiple
  candidate outputs; reports **7-14x** latency reduction and **~20x** throughput increase,
  landing competitively with AR Transformers **without needing knowledge distillation**
  (most earlier NAT work depended on distilling from an AR teacher to simplify the
  target distribution) — the strongest "close the quality gap" NAT result in our search
  [DA-Transformer].

**Verdict on NAT for this task:** NAT substantially closes the AR quality gap and,
in single-pass variants (GLAT, DA-Transformer), gets close to AR quality at 8-20x AR
speed. But every NAT system above is still an **encoder-decoder** architecture — it runs
a decoder stack (whether single-pass or iterative) in addition to the encoder, and the
best-quality variants (Mask-Predict, DisCo) still take multiple forward passes. None of
this literature reports numbers in the "tens of milliseconds, encoder-only classifier"
regime; it reports large *relative* speedups over an AR baseline, which is a different
thing from an absolute latency floor low enough for an inline gate. NAT is the right
answer if the task genuinely requires open-ended generation (new word order, new words not
present in the source). It is very plausibly the *wrong* answer if most of the required
transformation is expressible as edits to the existing token sequence — which brings us to
the approach we think is the actual fit.

**Edit-tagging: encoder-only, single pass, literally classifier speed.** This family
reformulates rewriting as **per-token tag prediction** — KEEP / DELETE / REPLACE-with-X /
INSERT-X — using only an encoder (e.g., BERT/RoBERTa) with lightweight linear/FFN heads on
top, exactly the architecture and exactly the inference cost profile of a token
classification model (NER, POS tagging). There is no decoder and no autoregressive loop.

- **LaserTagger** (Malmi et al., Google, EMNLP 2019): casts generation as tagging with
  three edit operations (KEEP, DELETE, ADD-phrase-before-token, where added phrases come
  from a small fixed vocabulary mined from training data), using a BERT encoder with a
  small autoregressive "realization" component only for phrase insertion. Reported to
  compute predictions **up to 100x faster** than a comparable seq2seq baseline, reaches
  new SOTA on 3 of 4 evaluated text-editing tasks, and — importantly for a
  low-resource-data plan — "produces reasonable outputs even when trained using only a few
  hundred or a few thousand examples," beating seq2seq baselines specifically in the
  low-data regime [LaserTagger].
- **GECToR** (Omelianchuk et al., Grammarly, BEA-2020): a GEC tagger over a Transformer
  encoder (best results with RoBERTa-large) with ~5,000 custom token-level
  transformation tags (including linguistically aware ones like verb-form change, not
  just KEEP/DELETE/REPLACE), trained with a synthetic-pretrain → errorful-finetune →
  errorful+clean-finetune curriculum. F0.5 65.3/66.5 (CoNLL-2014) and 72.4/73.6
  (BEA-2019) as noted above; inference "up to 10x as fast as" a Transformer-NMT seq2seq
  GEC system, with one reported comparison giving **0.20-0.40s vs 0.71-4.35s** for the
  tagger versus seq2seq baselines respectively (batch-level, not confirmed per-sentence in
  the sources we accessed — flagged as a precision gap, not a claim we independently
  verified) [GECToR; GECToR-speed-secondary]. GECToR is open-sourced, including a later
  paper explicitly titled **"Text Simplification by Tagging"** using the same codebase
  [GECToR-repo].
- **Text Simplification by Tagging (TST)** (Omelianchuk et al., Grammarly, BEA-2021) is
  the most directly relevant prior result to this paper's goal: it applies the GECToR-style
  tagging recipe to *simplification* rather than error correction — RoBERTa-base encoder
  plus two feed-forward heads (edit-detection, edit-classification) — trained on ~384K
  WikiLarge sentences augmented by back-translation (3x) and **ensemble distillation from a
  larger seq2seq model (2x)**, reaching **within about 1 SARI point of the BART-based
  state of the art** while running **11.75x faster than pure BART** [TST].
  This is essentially a smaller-scale rehearsal of the exact recipe this paper's target
  task needs: take a task that's "simplify/rewrite under constraints," distill from a
  larger generator, and land a tagger within roughly a point of seq2seq SOTA at an
  order of magnitude (or more) lower latency.
- **Seq2Edits** (Stahlberg & Kumar, Google, EMNLP 2020): represents transduction as
  span-level edit tuples (error tag, span end, replacement) predicted with a modified
  Transformer; **inference cost scales with the number of edits, not the number of output
  tokens**, giving up to **5.2x** speedup over full seq2seq on GEC specifically, and it is
  evaluated across five tasks including text normalization, sentence fusion/splitting,
  simplification, and GEC — i.e., it was explicitly validated on the "split a sentence"
  operation that pure KEEP/DELETE/REPLACE tagging struggles with [Seq2Edits]. (Seq2Edits
  keeps a small autoregressive component for the edit sequence, so it is not
  strictly single-pass encoder-only, but its cost model is edit-count-bound rather than
  token-count-bound, which is a relevant middle ground.)

**Does an encoder-only tagger plausibly handle ASD-STE100? What fraction of edits are
local vs. structural?** No paper we found measures this split for STE specifically; what
follows is our own structural analysis of the 53 rules as summarized by ASD and secondary
sources [ASD-About; ClickHelp-STE-rules], offered as a hypothesis to validate empirically
(Section 6), not a citable result:

- **Clearly local, tagger-friendly (majority of rules by count):** controlled-vocabulary
  substitution (an unapproved word → its approved synonym — pure REPLACE), removing banned
  words/phrases (DELETE), inserting dropped articles or relative pronouns ("which"/"that")
  that STE requires to remain unambiguous (local INSERT), correcting non-approved verb
  forms and gerunds used as nouns (local REPLACE, possibly with a small closed
  target vocabulary — exactly LaserTagger's/GECToR's design point), and most punctuation
  and capitalization rules.
- **Structurally harder, poorly served by pure per-token tagging:** (a) **passive → active
  voice conversion**, which requires promoting an object to subject position and often
  inserting an explicit agent (e.g., an imperative "you"), i.e., reordering, not just
  substitution; (b) **splitting long or compound/complex sentences** to satisfy STE's
  ~20-word (procedural) / ~25-word (descriptive) sentence-length limits and "one
  instruction per sentence" rule — a SPLIT operation, well studied in the simplification
  literature (this is exactly what ASSET was built to evaluate, and what Seq2Edits
  explicitly targets with "sentence fusion/splitting") but not a KEEP/DELETE/REPLACE
  primitive; (c) converting nominalizations back to verb phrases, which can require
  local reordering around the clause.
- **Net assessment:** we expect the *majority* of required token positions in a typical
  input sentence to be correctly handled by pure tagging (this matches GECToR's and TST's
  general experience that most GEC/simplification edits are local), but a **meaningful
  minority of sentences** — specifically ones that are passive-voiced or over the length
  limit, which in real technical writing is common, not rare — will need capability beyond
  KEEP/DELETE/REPLACE. The literature offers three compatible extensions, all still
  encoder-based or edit-count-bound rather than full seq2seq: LaserTagger's bounded phrase
  insertion vocabulary, a dedicated SPLIT/reorder tag with position-pointer targets (as
  used in sentence-fusion tagging work), or a Seq2Edits-style span-replace with a very
  short autoregressive tail invoked only on flagged spans. **This split (what % of real STE
  rewrite instances are pure local edits vs. need restructuring) is the single most
  important number this paper cannot supply from the literature and that a prototype must
  measure directly against a labeled STE corpus before committing to a pure-tagging
  architecture.**

### 2.4 Training data and the synthetic-generation-with-gate approach

Available *direct* ASD-STE100 (ordinary, STE) parallel data is effectively **zero** in the
public domain: the dictionary itself is licensed/copyrighted by ASD, no public parallel
corpus of "before/after STE rewrite" pairs surfaced in our search, and the existing
checker vendors (HyperSTE, Congree, Acrolinx) are commercial products whose training/rule
data is not published. This means the project is a **cold-start data problem**, not a
fine-tuning problem, and synthetic generation is not a shortcut — it is the only viable
path.

The proposed approach — generate (ordinary, STE) pairs with a local LLM, keep only pairs a
deterministic rule checker accepts — is a specific instance of a well-established pattern
that goes by different names in different subfields, and is not novel as a *method*, only
in its application here:

- **Rejection-sampling fine-tuning / STaR** (Zelikman et al. and the broader RFT/RAFT
  literature): generate candidate outputs, keep only those that pass an external,
  verifiable check, fine-tune on the accepted set, optionally iterate
  [STaR-RFT-summary]. This is exactly "generate with an LLM, gate with a deterministic
  checker, train on survivors," applied elsewhere to reasoning traces rather than
  controlled-language rewrites; the mechanism transfers directly because STE compliance,
  like a math answer, is a **binary, automatically checkable** property (this is precisely
  what HyperSTE/Congree/Acrolinx already compute).
- **GEC's own synthetic-data tradition** independently validates the "corrupt/generate,
  then train" half of the pipeline: PIE's ~9M synthetic sentence pairs generated by
  rule-based corruption [PIE], and Google's **C4_200M "tagged corruption model"**, which
  deliberately generates synthetic GEC errors *conditioned on error-type tags* so the
  synthetic error distribution can be steered to match real error statistics
  [C4-200M-tagged-corruption] — the closest published precedent for *distribution-aware*
  synthetic generation of an edit task, though it corrupts clean text to make noisy input
  (the reverse direction from what an STE pipeline needs, which is closer to TST's
  ensemble-distillation step, below).
- **cLang-8** is a directly relevant precedent for **model-distilled labels**: rather than
  use noisy human-written Lang-8 corrections directly, Google relabeled the raw Lang-8
  source sentences with the output of a large multilingual GEC model and got a *better*
  training set than the human-authored one [cLang8]. This validates using a large model's
  output as training supervision for a smaller model, which is structurally the same move
  proposed here (LLM generates candidate STE rewrites; a smaller tagger learns from the
  accepted subset).
- **TST's own recipe** already contains a small-scale version of exactly this plan:
  starting from 384K sentence pairs, TST used **ensemble distillation** (predictions from
  an ensemble of larger seq2seq models, added as extra training signal) to double its
  effective training data, on top of back-translation augmentation [TST].
- **ChatLang-8** (2024) is a direct, named precedent for using an LLM specifically to
  synthesize GEC training pairs at scale, an explicit "LLM as data generator for a
  smaller specialized model" pipeline in the GEC literature [ChatLang8].

**How much data is plausibly needed?** There is no single number, but the surveyed systems
bracket a realistic range: LaserTagger gets usable results from a **few hundred to a few
thousand** examples in low-resource settings [LaserTagger]; TST's simplification tagger was
trained on **384K base pairs, augmented to roughly 1-1.5M effective examples**
[TST]; GECToR's curriculum uses **~9M synthetic pairs for pretraining**, then fine-tunes on
**~1.2M genuine error-correction pairs** (Lang-8 1.04M + NUCLE 57K + FCE 34K), with
**cLang-8's 2.37M distilled pairs** as a stronger alternative to raw Lang-8 [PIE; cLang8;
GECToR]. For an STE tagger, we would target the TST-scale regime (hundreds of thousands to
low millions of gated pairs) as a first checkpoint, informed by the fact that STE's rule
set (53 rules, ~900-word dictionary) is *narrower and more literally rule-based* than
"general grammaticality," so we would expect useful compliance behavior to emerge from
less data than full open-domain GEC needs — but this is a hypothesis to test, not a
result we can cite.

**The gate is necessarily partial.** A deterministic ASD-STE100 checker (which this
project would need to build or license regardless, as the same component the commercial
checkers already sell) can verify **compliance** with high precision — that is definitionally
what a rule/dictionary checker does. It **cannot** verify that the rewrite preserved the
original instruction's meaning, scope, or requirement strength (e.g., "must" vs. "should"
vs. "may"). This is the same limitation the hobbyist LLM-skill READMEs flag explicitly
about themselves [GH-asd-ste100-skill-SKILL]. Any synthetic pipeline built this way filters
for *form*, not *fidelity*, and fidelity will need a separate signal (a second gate — e.g.,
round-trip back-translation consistency, NLI-style entailment checking between source and
rewrite, or sampled human/LLM-judge review) discussed further in Sections 5-7.

### 2.5 Model size

Realistic candidates and what evidence supports each, given the "encoder-only tagging is
the architecture that structurally fits the latency budget" conclusion of Section 2.3:

| Family | Example sizes | Fit for this task |
|---|---|---|
| Encoder-only tagger | BERT-base 110M, RoBERTa-base 125M, DeBERTa-v3-base ~184M, RoBERTa-large 355M | Best fit. TST used RoBERTa-base (125M) and got within ~1 SARI of BART SOTA at 11.75x speed [TST]; GECToR's best numbers use RoBERTa-large/XLNet-large (~340-355M) [GECToR]. Single forward pass, no decoder — cost profile identical to a production NER/classification model. |
| Small AR seq2seq | T5-small 60M, T5-base 220M, BART-base 140M | Wrong architecture class for the latency target regardless of parameter count: decoding is sequential and cost scales with output length; best reported optimized speedups (TensorRT/ONNX) are still only 3-21x over an unoptimized baseline, not a fixed low-ms floor [T5-TensorRT; T5-ONNX-fastT5]. |
| NAT (single-pass, e.g. GLAT/DA-Transformer) | Comparable to base Transformer (~60-200M) plus a decoder stack | Closes most of the AR quality gap (0.25-0.9 BLEU for GLAT; DA-Transformer competitive without distillation) at 7-20x AR speed [GLAT; DA-Transformer], but still runs a decoder, and the higher-quality iterative variants (Mask-Predict, DisCo) need multiple forward passes — a worse fit than pure tagging for a hard tens-of-ms floor. |

For latency, we found credible production evidence that small/optimized encoder-only
classifiers reach **10-20ms median latency** in real deployments — e.g., a quantized
BERT-tiny classifier at Roblox reported <20ms median on CPU after optimization
[BERT-latency-roblox-etc]. This is the right ballpark for the "tens of milliseconds"
target, but it is evidence from unrelated classification tasks (not a rewriting/tagging
model at 300M+ parameters), and we did not find a directly reported single-sentence,
unbatched latency figure for GECToR or TST themselves — both papers report *relative*
speedup over a seq2seq baseline, not an absolute serving latency. **This is a gap a
prototype must close with its own benchmark before committing to a production latency
SLA.**

## 3. Problem Formulation

Given input sentence (or short passage) $x$ in ordinary English, produce $\hat{y}$ such
that:

1. $\hat{y}$ is ASD-STE100-compliant: every word is in the ~900-word approved dictionary
   (or an approved technical/proper noun per STE's naming rules), every applicable
   grammar/structure rule (of the 53) is satisfied, and sentence length limits are met.
2. $\hat{y}$ preserves the meaning, scope, and requirement strength of $x$ (no silently
   dropped conditions, no changed modality).
3. Inference for a single sentence completes in tens of milliseconds on realistic serving
   hardware (single forward pass, no autoregressive decode loop of unbounded length).
4. The system is reliable enough for unsupervised inline use — i.e., near-100%,
   not "usually," compliance on in-distribution input, with a well-characterized failure
   mode (flag/reject) for out-of-distribution input rather than silent miscompliance.

We treat (1) as automatically checkable (a deterministic function of $\hat{y}$, given a
rule/dictionary checker — buildable or licensable, since this is what the commercial
checker vendors already sell). We treat (2) as **not** automatically checkable with
existing tools, and therefore as the primary reliability risk. (3) is an architecture
constraint, argued in Section 2.3 to select encoder-only tagging over seq2seq/NAT. (4) is
a product requirement that changes the acceptance bar relative to the published
academic baselines in Section 2.2 (F0.5 in the 65-74 range, SARI in the 40s), which are
*not* "reliable enough for unsupervised inline use" bars — they are "good research
result" bars. Closing that gap is the central unresolved problem this paper surfaces
(Section 7).

## 4. Proposed Approach

1. **Architecture:** an encoder-only sequence tagger in the 100-400M parameter range
   (starting point: RoBERTa-base/DeBERTa-v3-base class, ~125-184M, matching TST's proven
   operating point [TST]), predicting per-token edit tags (KEEP / DELETE / REPLACE-with-X
   from a closed target vocabulary / local INSERT-X), following the GECToR/LaserTagger/TST
   design pattern [GECToR; LaserTagger; TST].
2. **Structural-edit extension:** add a bounded mechanism for the non-local operations
   identified in Section 2.3 as poorly served by pure tagging — at minimum a SPLIT tag
   (with a pointer to the split point, in the spirit of sentence-fusion tagging and
   Seq2Edits' span operations [Seq2Edits]) and a small closed-vocabulary phrase-insertion
   mechanism for passive→active repair (LaserTagger's approach [LaserTagger]), rather than
   assuming pure KEEP/DELETE/REPLACE suffices — this must be validated empirically
   (Section 6), not assumed.
3. **Deterministic compliance gate:** build (or license) a rule/dictionary checker
   implementing the 53 ASD-STE100 rules and the approved-word dictionary — the same
   category of artifact HyperSTE/Congree/Acrolinx already sell — and use it two ways:
   (a) as the training-data filter (Section 5), and (b) optionally as a fast post-hoc
   runtime check on the tagger's own output, since the checker itself is fast
   (dictionary lookup + regex/POS-pattern rules), making a "generate, then verify, flag on
   failure" runtime architecture compatible with the latency budget as long as the checker
   is itself sub-tens-of-ms.
4. **Fidelity gate (secondary, weaker):** because compliance-checking cannot verify meaning
   preservation, add a second, independent signal during data curation — e.g., round-trip
   consistency (does an unconstrained paraphrase-back-to-ordinary-English model recover
   the original meaning?) or an NLI/entailment check between $x$ and $\hat{y}$ — understood
   as risk-reduction, not proof, consistent with the honest limitation the hobbyist STE
   skills already flag about themselves.

## 5. Data

**Pipeline:**

1. Source ordinary-English technical sentences (real technical documentation the
   organization already has, or general technical-domain corpora) as $x$.
2. Generate candidate STE rewrites $\hat{y}$ with a larger LLM prompted with the actual
   ASD-STE100 rules and dictionary (this mirrors ChatLang-8's use of an LLM as a synthetic
   pair generator for a rule-governed monolingual rewrite task [ChatLang8]).
3. **Gate**: run the deterministic compliance checker (Section 4.3) on every $\hat{y}$;
   discard non-compliant candidates. This is a direct application of the rejection-sampling
   / STaR pattern [STaR-RFT-summary] — generate, verify against an automatic checker, keep
   only survivors — and is methodologically the same move as cLang-8's use of a larger
   model's output as training supervision for a smaller model [cLang8].
4. **Secondary fidelity filter** (weaker, higher cost): sample-check accepted pairs for
   meaning preservation (Section 4.4); down-weight or discard pairs that fail.
5. Optionally **augment** via back-translation and/or ensemble distillation from more than
   one generator LLM, following TST's demonstrated approach of using both techniques to
   roughly triple/double effective training data from a smaller seed set [TST].
6. Train the tagger (Section 4) on the surviving, gated pairs.

**Scale target:** given TST reached within ~1 SARI of seq2seq SOTA from ~384K seed pairs
augmented to roughly 1-1.5M effective examples [TST], and given STE's rule set is narrower
than open-domain grammaticality, we would target an initial gated corpus in the
**hundreds of thousands** of pairs as a first checkpoint, scaling toward GECToR/cLang-8's
**1-2M+** range [GECToR; cLang8] if quality plateaus early. This is a plan to validate
empirically, not a number derived from any STE-specific precedent, because none exists.

**Known data risks:** (a) LLM-generated candidates will reflect whatever the generator LLM
already knows about STE from pretraining, which is uncontrolled and unverified — the gate
controls for rule compliance but not for representativeness of real technical-writing
inputs; (b) the ASD-STE100 dictionary itself is licensed, so a from-scratch checker
implementation needs either a license or a carefully-scoped independent reproduction (the
hobbyist tools explicitly avoid reproducing the real dictionary for this reason
[GH-asd-ste100-skill-SKILL]) — this is a legal/licensing dependency, not just an
engineering one, and should be resolved before data generation begins at scale.

## 6. Evaluation Plan

**Compliance metrics (primary, since this is the checkable half of the objective):**
- **Rule-level compliance rate**: % of output sentences with zero rule violations per the
  deterministic checker (directly analogous to what HyperSTE/Congree/Acrolinx report today
  for human-written text, giving an apples-to-apples "how much better than a human writer"
  comparison).
- **Edit-level precision/recall/F0.5** against gold or LLM-generated-and-human-vetted
  rewrites, following GEC's ERRANT-based evaluation convention [GECToR; BEA2019-Task],
  since STE rewriting is edit-shaped in the same way GEC is.

**Fidelity metrics (secondary, since no existing tool solves this):**
- **SARI** against references, to capture whether the *kind* of edit made
  (add/delete/keep) resembles reference simplification behavior [SARI-orig].
- **GLEU**, adapted to reward STE-required changes and penalize uncorrected
  non-compliant spans, following its GEC usage [Napoles-GLEU].
- **Semantic equivalence / entailment** between $x$ and $\hat{y}$ (bidirectional NLI or a
  larger LLM-as-judge score focused specifically on requirement strength and dropped
  conditions), reported separately from compliance, since these two axes can trade off
  against each other (a rewrite can be maximally compliant and still lose meaning).
- **FKGL**, reported only as a secondary descriptive statistic, given its documented
  weakness as a standalone simplification-quality metric [FKGL-critique].

**Latency:** single-sentence, unbatched, p50/p95/p99 latency on the target serving
hardware, benchmarked directly rather than assumed from the relative-speedup numbers in
Section 2.3/2.5, since none of the surveyed papers report this figure for a comparable
model.

**Baselines to report against:**
- The **do-nothing baseline**: compliance rate of un-rewritten input (establishes how much
  of the "problem" is already-compliant text).
- **A commercial checker's flag rate** as a compliance-only reference point (not a
  rewriter, but the closest existing measured artifact) [HyperSTE; Congree; Acrolinx].
- **Zero-shot/few-shot prompting of a large general LLM** (the closest thing to today's de
  facto approach, per the hobbyist skills in Section 2.1), measured for both compliance
  rate and latency, to make the speed/reliability trade-off this paper is investigating
  concrete and quantified rather than assumed.
- **A small AR seq2seq baseline** (T5-small/BART-base fine-tuned the same way), to
  quantify the tagging-vs-seq2seq quality gap directly on this task, mirroring how
  GECToR/TST report against seq2seq baselines [GECToR; TST].

**Held-out test design:** should include both (a) in-distribution technical sentences from
the same domain as training data, and (b) out-of-distribution/adversarial sentences
(unusually long, heavily passive, dense nominalization) specifically to measure the
structural-edit failure mode flagged in Section 2.3 — this split is where we expect the
pure-tagging hypothesis to be falsified or confirmed.

## 7. Limitations and Risks

- **The central open question is empirical, not architectural**: what fraction of real STE
  rewrite instances require restructuring (split/reorder) beyond local tagging. We could
  not find literature that measures this for STE, and our own estimate in Section 2.3 is a
  structural reading of the rules, not a measurement. If the true fraction is high, pure
  tagging quality will fall well short of the "reliable" bar, and the system will need a
  heavier structural-edit mechanism (Section 4.2) that could erode the latency advantage
  that motivates this entire approach.
- **"Reliable, not best-effort" is a much higher bar than the literature's reported
  numbers.** GECToR's F0.5 65-74 and TST's "within ~1 SARI point" are strong *research*
  results, not evidence of near-100% compliance suitable for unsupervised inline
  deployment. Nothing in the surveyed literature reports a compliance rate anywhere near
  what "sits inline in a request path" implies is needed; closing this gap likely requires
  the runtime compliance gate proposed in Section 4.3 (generate, verify, flag-on-failure)
  rather than trusting a single forward pass unconditionally — which in turn means the
  system's *effective* latency and reliability depend on how the checker's rejections are
  handled (retry? fall back to a slower model? pass through unmodified with a flag?), a
  product decision this paper does not resolve.
- **Meaning preservation is unverified by construction.** The compliance gate, by design,
  cannot check fidelity; this is the same gap every existing STE-adjacent tool (commercial
  checkers and the hobbyist LLM skills alike) leaves open. A synthetic pipeline trained
  purely against a compliance gate risks learning to satisfy the rule checker via
  degenerate rewrites (over-deleting content, flattening meaning) that a naive SARI/GLEU
  score might not catch either, since both metrics reward *some* deletion.
  Human or strong-LLM-judge spot-checking of fidelity is not optional.
- **Licensing.** The ASD-STE100 dictionary and rule text are ASD intellectual property; ASD
  explicitly does not endorse third-party tools [ASD-Tools]. Any checker or training
  pipeline built against the real dictionary needs its licensing question resolved
  independent of the ML approach; the hobbyist tools' choice to *not* reproduce the
  dictionary is itself evidence this is a live constraint, not a formality.
  This is a legal/business risk to flag early, not an ML risk.
  We are not qualified to and do not offer licensing advice here.
- **Domain shift in training data.** Because there is no existing STE parallel corpus, all
  training data is either LLM-generated (Section 5) or comes from whatever real technical
  documentation the organization can source; both are unverified with respect to how
  representative they are of the true deployment input distribution (arbitrary English in
  a request path, which may include non-technical, conversational, or malformed text STE's
  rules were never designed to be applied to).
- **NAT and small-seq2seq were not empirically re-benchmarked here.** All numbers in
  Section 2.3/2.5 are as reported in their original papers on their original tasks
  (general-domain MT, GEC, Wikipedia simplification), not measured on STE data. Relative
  rankings (tagging < NAT-single-pass < NAT-iterative < AR seq2seq, in speed; roughly the
  reverse in raw generative flexibility) are a reasonable prior but not a guarantee they
  transfer unchanged to this specific, narrower task.

## 8. Conclusion

No validated, fast ASD-STE100 rewriter exists today; the standard's own maintainers say
explicitly that no tool converts non-STE text into STE, the commercial ecosystem is
check-only, and the only rewriting attempts we found are slow, unverified LLM-prompting
scripts. The task is a same-language, rule-constrained sequence transduction problem
structurally identical to GEC and sentence simplification, both of which have an
architecture family — encoder-only edit tagging (GECToR, LaserTagger, Seq2Edits, and most
directly, Text Simplification by Tagging) — that is the only family in the literature
whose cost profile (one non-autoregressive forward pass, no decoder) structurally matches
a "tens of milliseconds" target, and which has already been shown, on the closely related
task of sentence simplification, to land within about a SARI point of seq2seq state of the
art at roughly 12x the speed. This makes the goal **plausible, not proven**: the
open question the literature cannot answer, and that must be settled empirically before
committing to a pure-tagging architecture, is what fraction of real ASD-STE100 rewrites
are local token edits versus sentence-level restructuring (passive-voice conversion,
long-sentence splitting) that tagging alone handles poorly. The proposed synthetic
data pipeline — LLM generation gated by a deterministic compliance checker — is a
direct, well-precedented application of rejection-sampling/self-distillation methodology
(STaR/RFT, cLang-8, TST's own ensemble distillation), but it verifies compliance, not
fidelity, and meaning preservation remains the unresolved problem every adjacent existing
tool, commercial and hobbyist alike, also leaves open. We recommend prototyping the
tagging architecture against a small hand-built or LLM-generated-and-human-vetted STE
evaluation set before any large-scale data investment, specifically to measure the
local-vs-structural edit split identified in Section 2.3/7 — that single number is the
most decision-relevant unknown this paper surfaces.

## References

- [ASD-STE100] ASD-STE100 Home Page. https://www.asd-ste100.org/
- [ASD-About] ASD-STE100, "About STE." https://www.asd-ste100.org/about_STE.html
- [ASD-Tools] ASD-STE100, "Tools for STE." https://www.asd-ste100.org/STEsoftware.html
- [TCWorld-Issue9] tcworld magazine, "ASD-STE100 Issue 9: Setting a standard for technical documentation." https://www.tcworld.info/e-magazine/technical-writing/asd-ste100-issue-9-setting-a-standard-for-technical-documentation
- [ASD-Europe-Issue9] ASD Europe, "A milestone in Simplified Technical English: ASD-STE100 Issue 9." https://www.asd-europe.org/news-media/news-events/news/simplified-technical-english-asd-ste100-issue-9/
- [ResearchGate-STE-MT] / [ResearchGate-STE-MT-Overview] "Simplified Technical English: An Overview of the Current Standard and Its Relevance for Machine and Automated Translation." https://www.researchgate.net/publication/399459173
- [Wikipedia-STE] Wikipedia, "Simplified Technical English." https://en.wikipedia.org/wiki/Simplified_Technical_English
- [Wikipedia-CNL] Wikipedia, "Controlled natural language." https://en.wikipedia.org/wiki/Controlled_natural_language
- [ClickHelp-STE-rules] ClickHelp, "Simplified Technical English (STE): Rules and Examples." https://clickhelp.com/clickhelp-technical-writing-blog/what-is-simplified-technical-english/
- [HyperSTE] HyperSTE, "Simplified Technical English - The STE Checker." https://hyperste.ai/ste-simplified-technical-english-checker-hyperste/
- [Congree] Congree, "Simplified Technical English - the STE Checker." https://www.congree.com/en/ste-simplified-technical-english
- [Acrolinx-Docs] Acrolinx Platform Docs, "Simplified Technical English." https://docs.acrolinx.com/acrolinxplatform/latest/en/guidance/simplified-technical-english
- [TechScribe] TechScribe, "ASD-STE100 Simplified Technical English." https://www.techscribe.co.uk/techw/asd-simplified-technical-english.htm
- [GH-asd-ste100-skill] danyuchn/asd-ste100-skill (GitHub). https://github.com/danyuchn/asd-ste100-skill
- [GH-asd-ste100-skill-SKILL] danyuchn/asd-ste100-skill, SKILL.md. https://github.com/danyuchn/asd-ste100-skill/blob/master/SKILL.md
- [GH-SimpleEnglish] AminBlg/SimpleEnglish (GitHub). https://github.com/AminBlg/SimpleEnglish
- [WikiLarge] Zhang & Lapata, "Sentence Simplification with Deep Reinforcement Learning," EMNLP 2017 (WikiLarge corpus). https://github.com/XingxingZhang/dress
- [TurkCorpus] Xu et al., "Optimizing Statistical Machine Translation for Text Simplification," TACL 2016.
- [ASSET] Alva-Manchego et al., "ASSET: A Dataset for Tuning and Evaluation of Sentence Simplification Models with Multiple Rewriting Transformations," ACL 2020. https://arxiv.org/abs/2005.00481
- [MUSS] Martin et al., "MUSS: Multilingual Unsupervised Sentence Simplification by Mining Paraphrases." https://arxiv.org/abs/2005.00352
- [SARI-orig] Xu, Napoles, Pavlick, Chen, Callison-Burch, "Optimizing Statistical Machine Translation for Text Simplification," TACL 2016 (SARI metric).
- [SARI-critique] "Rethinking Automatic Evaluation in Sentence Simplification." https://arxiv.org/abs/2104.07560
- [CLEF2025-SimpleText] "LLM-Guided Planning and Summary-Based Scientific Text Simplification: DS@GT at CLEF 2025 SimpleText." https://arxiv.org/abs/2508.11816
- [FKGL] "The Flesch-Kincaid Grade Level for Scoring Digital Content." https://readabilityformulas.com/the-flesch-kincaid-grade-level-for-digital-content/
- [FKGL-critique] "Flesch-Kincaid is Not a Text Simplification Evaluation Metric," GEM Workshop 2021. https://aclanthology.org/2021.gem-1.1/
- [Napoles-GLEU] Napoles, Sakaguchi, Post, Tetreault, "GLEU Without Tuning." https://arxiv.org/abs/1605.02592
- [cLang8] Rothe et al., "A Simple Recipe for Multilingual Grammatical Error Correction" (cLang-8). https://arxiv.org/abs/2106.03830
- [BEA2019-Task] Bryant et al., "The BEA-2019 Shared Task on Grammatical Error Correction."
- [GECToR] Omelianchuk, Atrasevych, Chernodub, Skurzhanskyi, "GECToR -- Grammatical Error Correction: Tag, Not Rewrite," BEA 2020. https://arxiv.org/abs/2005.12592
- [GECToR-speed-secondary] Grammarly Engineering Blog, "Experimenting with GECToR: Research into Ensembling and Knowledge Distillation for Large Sequence Taggers." https://www.grammarly.com/blog/engineering/experimenting-with-gector/
- [GECToR-repo] grammarly/gector (GitHub), covering GECToR (BEA-20) and Text Simplification by Tagging (BEA-21). https://github.com/grammarly/gector
- [TST] Omelianchuk, Raheja, Skurzhanskyi, "Text Simplification by Tagging," BEA 2021. https://aclanthology.org/2021.bea-1.2/ ; Grammarly Engineering Blog summary, "When Less Is More: Text Simplification by Tagging." https://www.grammarly.com/blog/engineering/text-simplification-by-tagging/
- [LaserTagger] Malmi, Krause, Rothe, Mirylenka, Severyn, "Encode, Tag, Realize: High-Precision Text Editing," EMNLP 2019. https://arxiv.org/abs/1909.01187 ; Google Research Blog summary. https://research.google/blog/encode-tag-and-realize-a-controllable-and-efficient-approach-for-text-generation/
- [Seq2Edits] Stahlberg & Kumar, "Seq2Edits: Sequence Transduction Using Span-level Edit Operations," EMNLP 2020. https://arxiv.org/abs/2009.11136
- [PIE] Awasthi et al., "Parallel Iterative Edit Models for Local Sequence Transduction," EMNLP 2019. https://arxiv.org/abs/1910.02893
- [C4-200M-tagged-corruption] Stahlberg & Kumar, "Synthetic Data Generation for Grammatical Error Correction with Tagged Corruption Models," BEA 2021. https://arxiv.org/abs/2105.13318 ; Google Research Blog, "The C4_200M Synthetic Dataset for Grammatical Error Correction." https://blog.research.google/2021/08/the-c4200m-synthetic-dataset-for.html
- [ChatLang8] "ChatLang-8: An LLM-Based Synthetic Data Generation Framework for Grammatical Error Correction," 2024. https://arxiv.org/abs/2406.03202
- [STaR-RFT-summary] Zelikman et al., "STaR: Bootstrapping Reasoning With Reasoning"; general RFT/RAFT rejection-sampling fine-tuning literature (surveyed via secondary sources for this report).
- [NAT-orig] Gu, Bradbury, Xiong, Li, Socher, "Non-Autoregressive Neural Machine Translation," ICLR 2018. https://arxiv.org/abs/1711.02281
- [MaskPredict] Ghazvininejad, Levy, Liu, Zettlemoyer, "Mask-Predict: Parallel Decoding of Conditional Masked Language Models," EMNLP 2019. https://arxiv.org/abs/1904.09324
- [LevT] Gu, Wang, Zhao, "Levenshtein Transformer," NeurIPS 2019. https://arxiv.org/abs/1905.11006
- [DisCo] Kasai, Cross, Ghazvininejad, Gu, "Non-Autoregressive Machine Translation with Disentangled Context Transformer," ICML 2020. https://arxiv.org/abs/2001.05136
- [GLAT] Qian et al., "Glancing Transformer for Non-Autoregressive Neural Machine Translation," ACL 2021. https://arxiv.org/abs/2008.07905
- [CTC-NAT] Libovický & Helcl, "End-to-End Non-Autoregressive Neural Machine Translation with Connectionist Temporal Classification," EMNLP 2018. https://arxiv.org/abs/1811.04719
- [CTC-SpeechTrans] "CTC-based Non-autoregressive Speech Translation," ACL 2023. https://arxiv.org/abs/2305.17358
- [DA-Transformer] Huang, Zhou, Jiao, Liu, Xu, "Directed Acyclic Transformer for Non-Autoregressive Machine Translation," ICML 2022. https://arxiv.org/abs/2205.07459
- [T5-TensorRT] NVIDIA Developer Blog, "Optimizing T5 and GPT-2 for Real-Time Inference with NVIDIA TensorRT." https://developer.nvidia.com/blog/optimizing-t5-and-gpt-2-for-real-time-inference-with-tensorrt/
- [T5-ONNX-fastT5] Ki6an/fastT5 (GitHub), "boost inference speed of T5 models by 5x & reduce the model size by 3x." https://github.com/Ki6an/fastT5
- [BART-vs-T5-issue] huggingface/transformers GitHub issue #27290, "Bart vs T5 inference time." https://github.com/huggingface/transformers/issues/27290
- [BERT-latency-roblox-etc] Secondary sources on production BERT-tiny/quantized classifier latency (~10-20ms median, CPU), surveyed via web search for this report; treat as illustrative, not a primary citation.

---

*Author's note on evidentiary standard: numbers in this document are drawn from the
cited papers/pages as accessed in September 2026 via web search and automated page
fetching; several figures (notably GECToR's absolute per-sentence latency, and any
single authoritative current SARI/F0.5 SOTA number) could not be independently confirmed
beyond secondary summaries and are flagged inline as such. Any of these should be
re-verified against the primary PDF tables before being used to justify a go/no-go
decision.*
