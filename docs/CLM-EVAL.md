# CLM-v0.1-8B: what it is, which claims hold, and the one test that decides it

Written 2026-09-24, the day after the model was released. A colleague
recommended it as something that "beats Jev and Laya".

What was done:

- the model card, the repo README and source, both benchmark charts and the
  secondary coverage were read
- nothing was downloaded and nothing ran on either GPU
- one read-only `nvidia-smi` query was made
- the running stack was not touched

Status key, used on every claim below:

- **PUBLISHED**: stated in a primary source (card, repo README, repo code
  or chart) and read today. This means the source says it. It does not mean
  it is true.
- **VENDOR-CLAIMED**: a result the authors report that nobody else has
  reproduced.
- **UNSUPPORTED**: no source states it.
- **INFERENCE**: our own reading of the sources, labelled as ours.
- **OURS**: a repo artefact, with its script and n.

## Summary

- **What it is.** CLM is **a frozen Qwen3-8B used as an embedder, with two
  small MLP projection heads on top.** Together the heads are about 20M
  parameters (a 75.6 MB `.pt`). The state and each candidate are embedded
  **separately**, and the answer is a softmax over the scaled cosines. It
  cannot generate. The "8B" is the stock Qwen3-8B, and the trained part is
  the head. Everything is Apache 2.0.
- **"Beats Jev": not established.**
  - Zero-shot, CLM never beats Jev on success. It ties on two 5-run games,
    and it loses on BFCL (95.2 vs 99.2%) and on WikiRacing (26/30 vs 30/30).
  - The only wins are for a **fine-tuned** CLM verifier. On DeepSWE that is
    31/38 against 27/38. At that n even a perfect 4–0 split of discordant
    tasks gives exact p = 0.125.
  - In both verifier charts, Jev scores **below** the plain Pass@1 line.
  - Every number comes from the vendor. No independent result exists.
- **"Beats Laya": unsupported.** Neither the card, the repo nor the coverage
  mentions Laya. The claim can only be transitive: CLM ≈ Jev, and Jev > Laya
  in third-party reports. It does not follow.
- **The one job it could plausibly take is Laya's.** That is a fixed-state
  `choice` over a small closed set, and in particular the `route_in` second
  signal in `selection.decide`. It would replace Laya one-for-one. That is
  not *fewer* systems, and it costs about 5 GiB more on the A4000 than Laya
  does, which ends image generation's co-residence there.
- **The eval that decides it** is `bench/eval_route_heldout.py` on the 120
  held-out labels, run with two new arms:
  - a CLM head
  - **a control head on the Qwen3-Embedding-0.6B vectors we already run**

  If the control matches Laya, Laya can be removed and nothing added. That is
  the fewer-systems outcome, and CLM does not come in at all. CLM earns a slot
  only by beating **both** Laya and the control (§5).

---

## 0. Who "Jev" and "Laya" are

**Jev is not a typo.** It is TypeSafe AI's closed, hosted "System One"
decision model, released 2026-09-15. You send it a state plus typed questions
(`choice`, `score`, `noul`) and it returns probabilities. It is the model
Laya set out to reproduce in the open. The repo already refers to it
throughout:

- `docs/BUILD.md` §5 ("A typed decision API over Laya — Jev-shaped")
- `docs/DECISION-TREES-AND-SKILLS.md` §1.1 (Jev vs Laya table, source S1)
- `docs/LAYA-FACTCHECK.md` "Community reports": #6 Laya 0.80–0.88 against
  Jev 0.89–0.99 on 310 decisions; #10 JevBench 54.4 against 74.4; #9 catalog
  pick 23–30 against Jev's 92. All are reported, **not reproduced by us**.

The CLM card compares against Jev and nothing else, so the colleague's "Jev"
is that model.

**Laya** is `convaiinnovations/laya`, run by `mcp/laya_service.py` on
`:1237` and pinned to the A4000:

- ModernBERT-large, 421M
- non-autoregressive
- typed `choice`, `score` and `noul` in one forward pass
- a 512-token window

Our live use is the trained `route_in` head: CLS plus 3 option logits,
logistic, 289 labels (`index/laya/route_in.json`). In `selection.decide` it
can add a deep-thinking run and never cancel one. It is also stage 4 of skill
selection, which is off by default (`YAMADORI_SKILL_LAYA`). Its record is in
AGENTS.md "Laya", `docs/LAYA-FACTCHECK.md` and FINDINGS #16, #17, #21 and #22.

## 1. What it is

| item | finding | status |
|---|---|---|
| Architecture | "two small projection heads (a state head and an action head) on top of a frozen Qwen3-8B encoder trained with a bidirectional InfoNCE loss" (card). Heads are MLPs, 4096 → width → 512, GELU, default depth 2 (`src/clm/heads.py`). Score = `exp(logit_scale) · cos(state_head(s), action_head(c))`, with the scale clamped at 100. | PUBLISHED |
| Parameter count | README: "20M-parameter trainable projection head". The checkpoint `CLM_v0.1-8B.pt` is 75.6 MB. That matches ~18.9M fp32 parameters, and 2 heads at width 2048 gives 18.88M. So "20M" is the **total** for both heads, not 20M each, as one secondary article says. The encoder is the unmodified 8B Qwen3-8B. | size PUBLISHED; width and dtype INFERENCE |
| "Contrastive LM" | A dual-encoder (bi-encoder) scorer. The state and each candidate go through the encoder **independently**. InfoNCE pulls the matching (state, action) pair together and pushes negatives apart. Training: pre-training on ~60M Nemotron Q&A pairs, mid-training on ~30M synthetic hard negatives (Gemini 2.5 Flash-Lite), post-training on ~1M agent trajectories. It is **not** a cross-encoder: the state never sees the candidates. | PUBLISHED; "never sees" is INFERENCE from `engine.py` and `schema.py` |
| Inputs | State text, plus typed questions. The state-head text is `"{state}\n\n{instructions}"`, "context first, question last — the layout the heads were trained on". Each option is embedded as its description, or as its key when the description is empty (`schema.py`, README). | PUBLISHED |
| Outputs | Scores and probabilities only. `choice` returns `{choice, confidence, probabilities}`. `score` returns an expected level index. `noul` is a **2-way choice** between the candidate texts "Yes. This is true: {instructions}" and "No. This is false: {instructions}", reported as p(true). Confidence is "top probability minus the mean of the rest". `/v1/rank` ranks free-form candidates. **No generation**: "CLM only scores the candidates you give it". | PUBLISHED |
| Calibration | Nothing is claimed. There is no ECE and no temperature refit. (Jev and Laya both claim calibration.) | UNSUPPORTED |
| Licence | Apache 2.0 for the weights and the code. The base Qwen3-8B is Apache 2.0. | PUBLISHED |
| Formats | The HF repo holds only `CLM_v0.1-8B.pt` (75.6 MB), `LICENSE`, `README.md` and `.gitattributes`. There are no safetensors, no GGUF and no `config.json`. The `.pt` is a pickle: load it with `torch.load(weights_only=True)` after inspecting it. The encoder is the stock `Qwen/Qwen3-8B`. Official GGUFs are Q4_K_M 5.03 GB, Q5_K_M 5.85, Q6_K 6.73, Q8_0 8.71. | PUBLISHED |
| Reference serving | `vllm serve Qwen/Qwen3-8B --runner pooling --enable-prefix-caching --max-model-len 2048 …` on `:8090`, then `clm-serve --emb-url http://127.0.0.1:8090/v1/embeddings` on `:8700`. Endpoints: `POST /v1/systemone`, `POST /v1/rank`, `GET /v1/models`, `GET /health`, and a playground at `/`. Needs "Python 3.10+, Linux, NVIDIA GPU". | PUBLISHED |
| llama.cpp / llama-server | **The encoder half, probably yes; the CLM half, no.** llama-server can serve a Qwen3-8B GGUF at `/v1/embeddings` with `--embeddings --pooling last`, which is how we already serve Qwen3-Embedding (same architecture). The heads are a PyTorch MLP that no llama.cpp endpoint runs: not `/v1/rerank` and not completion. They would run in Python, pointed at llama-server through `--emb-url`, or inside a service like `laya_service.py`. vLLM does not run natively on this Windows host. | INFERENCE, **unverified**; four things need checking (§5, stage 0) |
| Window and truncation | Trained and served at `--max-model-len 2048`. `Embedder` sends `truncate_prompt_tokens=2048`, and vLLM's documented meaning is "only the last k tokens", which is **left** truncation. So the question at the end survives and the *start* of a long state is dropped. | PUBLISHED (code and vLLM docs) |
| Pooling | "the heads require Qwen3-8B last-token-pooled embeddings", L2-normalised (`embedder.py`). | PUBLISHED |
| Paper | The README cites Kwok, Kang, Suresh, Saad-Falcon, Pavone, Ré and Mirhoseini (2026), "Contrastive Language Models: A System One Model for Fast and Generalizable Decision-Making", url `contrastive-lm.notion.site`. That is a Notion blog, and it did not render for our fetcher. No arXiv or peer-reviewed version was found. GitHub has 1 commit and 519 stars. | PUBLISHED; paper content **not read** |

## 2. Its claims

The authors ran every benchmark themselves. Two claims:

| claim | what the source shows | status |
|---|---|---|
| Zero-shot "on par with Jev … up to 9× lower latency" | Chart `assets/zero-shot.png`. Success: T-Rex 5/5 vs 5/5, BFCL v4 95.2% vs 99.2%, WikiRacing 26/30 vs 30/30, Super Mario 5/5 vs 5/5. Latency: 16.5/76.8/79.8/33.5 ms vs 149.8/125.5/225/132.6 ms. **CLM beats Jev on success nowhere and trails on 2 of 4.** Two of the four tasks have n=5. BFCL's n is not given. The chart does not state the hardware or whether Jev's latency includes a network round trip to a hosted API; a comparison against a remote service would be expected to favour a local one. | VENDOR-CLAIMED; "on par" is generous |
| Fine-tuned verifier "SOTA on DeepSWE (81.6%) and Terminal-Bench 2.1 (87.6%), 4.1–5.7× faster than Jev" | Chart `assets/agentic.png`. For each task, candidate solutions are sampled from **Opus 5** (DeepSWE, Bo4) or **Fable 5** (TB 2.1, Bo5), and the verifier picks one. Held out: 38 DeepSWE tasks and 30 TB 2.1 tasks; latency on an H100. DeepSWE: CLM fine-tuned 81.6% (31/38), Jev 71.1% (27/38), **Pass@1 73.7% (28/38)**. TB 2.1: 87.6%, Jev 83.1%, **Pass@1 84.0%**. The task counts in brackets are our arithmetic from the percentages. TB percentages are not multiples of 1/30, so some unstated averaging was done. | VENDOR-CLAIMED |

**What the verifier chart actually says (INFERENCE):**

- The success rate is mostly the policy model's. CLM adds **+3 tasks over
  Pass@1** on DeepSWE (28 → 31) and about +1 on TB.
- Jev picks *worse* than taking one sample, on both benchmarks.
- No pass@N oracle ceiling and no random-pick floor are shown, so what the
  verifier contributes cannot be bounded (PROTOCOL rule 10).
- CLM is fine-tuned for these tasks. Jev, a closed model, presumably is not.
- At n=38 the CLM-vs-Jev gap is 4 tasks. Even if all discordant pairs went
  CLM's way (4–0), exact McNemar p = 0.125. **The headline comparison cannot
  be significant at the published n.**

**Secondary coverage.** MarkTechPost (2026-09-23) reports DeepSWE as "CLM
fine-tuned 73.7%, Jev 71.1%". That 73.7% is the **Pass@1** dashed line, not
CLM, so the article mislabels the chart. The HF discussions have one thread,
a third-party integration announcement ("JevEmbed now supports it"). **No
independent reproduction exists** one day after release.

**"Beats Laya": UNSUPPORTED.** Neither the card, the README nor the
MarkTechPost article mentions Laya. No benchmark measures both models. The
transitive argument ("CLM ≈ Jev, Jev > Laya") fails three ways:

1. CLM does not match Jev on the zero-shot chart.
2. The Jev > Laya numbers are third-party and unreproduced
   (LAYA-FACTCHECK "Community reports").
3. The tasks differ: agent actions and verification there, typed text
   classification for Laya.

Also stated by the vendor: pre-training alone reaches 52.1% top-1, mid-training
lifts it to 69.2%, and training with hard negatives from the start peaks at
62.4%. The benchmark behind these numbers is not identified. There is also a
power-law scaling fit and "~310 tokens per parameter" for the optimal head
size. All of this is VENDOR-CLAIMED and irrelevant to our decision.

## 3. Fit to our jobs

One row per job. "Today" is what we run and have measured.

| job | today (OURS) | CLM fit | verdict |
|---|---|---|---|
| **Laya `choice` over a small closed set, and the `route_in` head** | `route_in` held out, investigate-vs-not (`bench/eval_route_heldout.py`, n=120): Laya head alone 80/120, bare regex 73, `selection.decide` 89, decide + live Laya 91 (p=0.79 vs 89). In-set CV 0.728 (8 splits of 86). 66 ms median round trip (`bench/laya_factcheck/live_route_check.py`). | The job's shape is the one CLM is built for: fixed state, a few typed options, a `choice`. CLM also has a documented cheap fine-tune (only the heads train), which is what made Laya useful to us (zero-shot 0.478 → trained 0.728). Our route_in questions are three.js/TSL/TypeGPU version questions, far from Nemotron Q&A and agent traces, so zero-shot transfer is unknown. | **The one plausible candidate.** It would replace Laya one-for-one (§5). |
| **Router** (`mcp/route.py`, rules) | `bench/route/eval_route.py`, n=1,016: utility and agent_step recall 1.000 and 0.995, **0 misroutes** into a code class, 383/383 code turns to code. | The classes are defined by **facts**: the conversation ends on a client tool result, the prompt carries code that *parses*, a utility contract is present. A similarity scorer can only approximate a fact the parser already knows (PROTOCOL rule 8). There is no headroom where it matters, and the in-sample weak spot (code_edit ↔ code_generation) runs the same pipelines. | **No.** Do not test it. |
| **Embeddings for retrieval** (rootstock: Qwen3-Embedding-0.6B Q8_0, `--pooling last`, `-c 8192`, `config.yaml` "embeddings") | Embedding hit@1 264/356 (`bench/retrieval_results.jsonl`, `bench/retrieval.py`). Also the winner in hint collapse (71.9%, n=89) and guardrails (AUC 0.886, n=156). | The CLM heads are an asymmetric Q&A retriever in effect, but nothing is claimed for retrieval, let alone code. Replacing the embedder means re-indexing every `index/packages/*.sqlite3` with an 8B encoder, at 2,048 tokens against today's 8,192. The raw Qwen3-8B last token is a language model's state, not a trained retrieval embedding. | **No.** It is 13× the parameters for an untested job, and the rootstock model must stay resident for retrieval anyway. |
| **Reranker** (Qwen3-Reranker-0.6B; **not trusted**, FINDINGS #20: 15–16/89 batched vs 68/89 one at a time) | Loaded, gated at `RERANK_MAX_K = 2`; `hints.py` does not call it. | CLM is not a cross-encoder, so it is not a reranker in the same sense. By construction, though, each candidate's score is independent of what else is in the request (a separate embedding and a cosine), so it **cannot** have #20's batch-composition defect. The softmax only rescales. It would be a second bi-encoder stage over the first bi-encoder. | Possible **second** job, only after the route_in stage passes (§5 stage 2). Replacing both Laya and the reranker is what makes "fewer" true: two systems become one. |
| **Skill selection, embedding stage** (`mcp/skill_select.py` stage 2: best trigger cosine, Qwen3-Embedding; thresholds EMB_HIGH 0.60 and EMB_LOW 0.45, **unmeasured**) | Nearest measured proxy: `bench/hint_collapse.py`, 19 buckets and 89 probes. Embedding 71.9%, Laya joint 33.7%, rerank 22.5%, chance 21.3% (FINDINGS #21). | Stage 2 shares its embedder with retrieval, so CLM cannot remove it. CLM could take **stage 4** (Laya over the 2–4 `ask` candidates plus "none", off by default), and possibly the stage-5 model fallback. Being order-invariant by construction removes Laya's #9 failure (reversing candidate order changed 65–76 of 100 picks). | Only as part of replacing Laya. Measured with a CLM arm in `hint_collapse.py` (§5 stage 2). |

**Laya's known flaws, and whether CLM has them:**

| flaw | Laya (OURS) | CLM |
|---|---|---|
| The window cuts the tail | 512 tokens, **tail** cut. A question after ~420–620 state tokens is lost, and every input then scores identically (`bench/laya_factcheck/tail_probe.py`). | 2,048 tokens, and vLLM cuts the **head** (keeps the last 2,048). The question is placed last, so it survives: the opposite failure, four times later. **On llama-server this does not carry over automatically.** We would have to truncate from the left ourselves, in the client, before sending (stage 0, G3). PUBLISHED and INFERENCE. |
| Scores not comparable across passages | A margin is not confidence (LAYA.md F3/F5/F7; FINDINGS #21, #22). Never threshold across queries. | **Same rule applies.** Probabilities are a softmax over *this* candidate set ("relative to provided candidate sets", card), and cosine levels differ by state. The raw per-pair logit is at least independent of the other candidates. INFERENCE. |
| Order sensitivity over options | Community #9: reversing the order changed 65–76 of 100 picks. | None, by construction: each option is embedded alone. INFERENCE, testable with one shuffle (stage 1). |
| Judges "aboutness", not truth (`judge` 6/10 vs a 5/10 coin flip) | Cut. | Probably worse. `noul` is the cosine of the state against "Yes. This is true: X" and against "No. This is false: X", which differ by a few tokens. Bi-encoders are known to be weak on negation. INFERENCE; do not use it as a judge. |
| Over-confident out of the box (card ECE 0.466) | Refit per deployment. | Nothing claimed. UNSUPPORTED either way. |

## 4. Hardware

Live, read-only, 2026-09-24 (`nvidia-smi`):

- 5060 Ti: 14,466 MiB used, 1,585 free. It stays full; nothing goes there.
- A4000: 7,567 MiB used, **8,600 free**. That matches `config.yaml`'s
  measured 7,565 MiB for embeddings + reranker + Laya.

Also resident or on demand on the A4000:

- **image generation**, co-resident: peak +6,389 MiB, leaving 2,213 free
  (`config.yaml`)
- **bonsai-vision**, exclusive, evicts retrieval: 8,265–9,449 MiB

Laya's own share is "~1 GiB" (`mcp/laya_service.py` comment). It was not
measured separately, because Windows reports per-process usage as N/A.

CLM encoder on the A4000 (**estimates, not measurements**):

- KV at `-c 2048` f16: 36 layers × 8 KV heads × 128 × 2 × 2 B = 288 MiB
- compute buffer: ~0.5–1 GiB at `-ub 2048`
- CUDA context: ~0.3 GiB

| quant | weights | est. total | A4000 free, CLM *replacing* Laya | co-resides with image gen? |
|---|---|---|---|---|
| Q4_K_M | 4.68 GiB | ~5.8–6.3 GiB | ~3.3–3.8 GiB | **no** (needs 6.2 GiB) |
| Q6_K | 6.27 GiB | ~7.4–7.9 GiB | ~1.7–2.2 GiB | no |
| Q8_0 | 8.11 GiB | ~9.2–9.7 GiB | does not fit next to retrieval | no |

- **Q4_K_M fits.** Any quant ends image generation's co-residence. The
  operator would have to accept either of these:
  - image generation evicts CLM, and `decide` falls back to the rule alone,
    which is its designed Laya-down path
  - CLM joins a non-persistent group
- **Quant fidelity is the unknown that matters.** The heads were trained on
  bf16 vLLM embeddings. A Q4 encoder feeds them shifted inputs, by an amount
  nobody has measured. That is gate G1 in §5.
- **Latency on the A4000 is unmeasured.** The vendor's 16–80 ms figures are on
  an H100 and an RTX 4090, in bf16, under vLLM. Laya's round trip here is
  66 ms median and 86 ms p90.
- **CPU is not an option** for a per-request signal. An 8B forward pass per
  question would take seconds (INFERENCE).

## 5. Evaluation plan

**Principle.** The operator wants fewer systems. CLM can only *replace*
Laya. Every stage therefore carries a **zero-new-system control**: the same
head trained on the Qwen3-Embedding-0.6B vectors we already serve. If that
control is as good as Laya, the right move is to remove Laya and add nothing,
and CLM never enters. CLM is adopted only if it beats both the incumbent and
the control.

Rules that apply:

- PROTOCOL 4: state the power first
- PROTOCOL 6: paired data, exact McNemar
- PROTOCOL 10: floors and ceilings, a search rather than one cell
- AGENTS.md: one GPU consumer at a time, queued through
  `bench/queue_runner.py`; a 429 means "not run"

**Downloads need operator approval before anything starts:**

- a Qwen3-8B GGUF, 5.0–8.7 GB
- the 75.6 MB head (a pickle)
- for G1, a bf16 reference: `Qwen/Qwen3-8B`, ~16 GB, run on CPU

### Stage 0: can we even run it faithfully? (feasibility gates, no accuracy claims)

- **G1, encoder fidelity.** Take 200 texts: 120 held-out questions plus 80
  from the training labels. Embed each with a llama-server Qwen3-8B GGUF
  (`--embeddings --pooling last -c 2048 -ub 2048`) and with the bf16
  reference. Pass: median cosine ≥ 0.99, and the zero-shot CLM route_in
  `choice` argmax agrees on ≥ 196/200. If Q4_K_M fails, try Q6_K, and
  re-read the §4 VRAM table.
- **G2, tokenisation.** The pooled token must be the last *text* token in
  both paths, with no BOS or EOS appended by one side only. Compare the token
  ids. The failure would be silent, as the asymmetric-prefix bug was
  (`config.yaml` "embeddings" comment).
- **G3, truncation and wire format.** Check four things:
  - what llama-server does with inputs longer than `-ub`: an error, or
    truncation, and from which end
  - whether it accepts `encoding_format: base64`
  - that it ignores `truncate_prompt_tokens`
  - then implement client-side **left** truncation to 2,048 tokens, as in
    training
- **G4, cost.** In a maintenance window, measure resident MiB, p50 and p90
  latency for a route_in question over HTTP, and determinism: 3 identical
  calls must match to 6 decimals, as Laya's did.

### Stage 1: the route_in job (decides whether CLM replaces Laya)

Data:

- training labels, 289 rows: `bench/laya_routing_labels.jsonl` (59),
  `_ext` (30) and `_packages_train` (200), the same as
  `index/laya/route_in.json` `trained_on`
- held out, 120 rows: `bench/laya_routing_heldout_packages.jsonl`, which no
  training glob matches

All arms run on the same 120 rows, paired, through `bench/eval_route_heldout.py`:

| arm | what | today |
|---|---|---|
| L | Laya route_in head, argmax (cached features) | 80/120 |
| R | bare regex `selection.rule_baseline` | 73/120 |
| S | `selection.decide`, Laya absent | 89/120 |
| S+L | `decide` + live Laya, disagreement escalates (`live_route_check.py`) | 91/120 |
| **C0** | CLM zero-shot `choice` with route_in's three labels (the same option descriptions Laya uses), argmax | new |
| **C1** | logistic head on CLM features, fitted with `scripts/train_laya.py`'s own procedure (same weight-decay search, gate by CV, 8 repeats, seed 0). Features: the 512-d state-head projection of `"{question}\n\n{route instruction}"` plus the 3 option logits, mirroring Laya's `cls_logits` spec. | new |
| **E1 (control)** | the same logistic head on Qwen3-Embedding-0.6B vectors from `code_search.embed` (query side, instruction prefix applied), 1,024-d | new; **no new system** |
| S+C1, S+E1 | `decide` with C1 or E1 as the second signal, same escalation rule | new |

Code change needed: one feature extractor per new arm in
`scripts/train_laya.py:get_features` (a `feature_spec` value). The eval script
gains the new heads as `--new` artefacts. Nothing is trained on the held-out
file.

Also report the three things the script already reports: 3-way accuracy, the
hard slice (n=28), and missed vs unneeded investigations. And one order check
on C0: shuffle the option order, and expect exactly identical output.

**Power, stated first.** Exact two-sided McNemar at α = 0.05 needs a net
discordant gap of:

| discordant pairs | net gap needed |
|---|---|
| ~20 | ≥ 10 |
| ~30 | ≥ 12 |
| ~40 | ≥ 14 |

On 120 rows that is **a 10–12 point difference**. Smaller real effects are
invisible here. The last comparison of this kind, decide ± Laya, had 14
discordant pairs and p = 0.79. If C1 and L land within ~10 points, the result
is "cannot tell", not "equal". Before anything is run, label **another 120+
package-domain questions**, blind to every arm, and pre-register them as
held-out set 2.

**Pass bar for CLM to replace Laya.** All of the following must hold:

1. C1, or C0, beats L with exact McNemar p < 0.05 on the held-out 120.
2. C1 beats **E1** with p < 0.05. If E1 ≥ L instead, the recommendation is
   to **remove Laya and use E1**, which leaves one fewer system, and CLM
   stops here.
3. S+C1 ≥ 91/120, with missed investigations ≤ 10 and unneeded
   investigations ≤ 19 (the S+L figures).
4. **It repeats.** Every one of these must hold:
   - in-set: C1 beats L in mean CV accuracy and in ≥ 6 of the 8 seeded
     splits
   - held-out set 2 gives the same sign
   - the live service reproduces the offline argmax 120/120
5. Stage 0 passed, and the operator has accepted the §4 VRAM trade
   explicitly (image generation no longer co-resides).

### Stage 2 (only if stage 1 passes): earn "fewer"

Stage 1 alone swaps one system for one. To come out ahead, CLM must also
absorb a second job:

- **Skill-select stage 4, and possibly the stage-5 fallback.** Add a CLM
  `choice` arm to `bench/hint_collapse.py`: 19 buckets, 89 probes, chance
  21.3%; embedding 64/89 (71.9%), Laya 30/89. Pass: beats the embedding arm
  with p < 0.05 overall, and on the contrastive slice (57.4%). Otherwise
  stage 4 is simply deleted along with Laya, and the embedding margin keeps
  the job (FINDINGS #21).
- **The reranker's slot.** Add a CLM second stage to `bench/retrieval.py` at
  full N (628 tasks): CLM `/v1/rank`-style scoring over the embedding top-40,
  against embedding-only order. Pass: hit@1 beats embedding-only with
  McNemar p < 0.05, and hit@5 is not worse. If it passes, the reranker can go
  (operator decision, PLAN "Cut criteria"), and the count becomes Laya +
  reranker → CLM, two systems to one. If it fails, the reranker question
  stays where FINDINGS #20 left it.

**Not evaluated, and why.** The router is decided by facts it already
parses. The retrieval embedder cannot be removed, because stages 2 of
skill-select and all of retrieval run on it, and swapping it means
re-indexing with an 8B model that claims nothing for retrieval. The `judge`
role (`scripts/eval_judge.py`, 10 items) could be tried in one call, but
`noul` is built on similarity, and that is the mechanism that failed there.

## Sources

Read 2026-09-24.

- Model card: <https://huggingface.co/Contrastive-LM/CLM-v0.1-8B> (raw
  README and YAML: `license: apache-2.0`, `base_model: Qwen/Qwen3-8B`,
  `pipeline_tag: text-ranking`)
- Model files: <https://huggingface.co/Contrastive-LM/CLM-v0.1-8B/tree/main>
- Model discussions: <https://huggingface.co/Contrastive-LM/CLM-v0.1-8B/discussions>
- Repo: <https://github.com/Contrastive-LM/CLM>, including `README.md`,
  `src/clm/{embedder,heads,engine,schema}.py` and `evaluation/bon_eval.py`
  (listed, not read)
- Charts: <https://raw.githubusercontent.com/Contrastive-LM/CLM/main/assets/zero-shot.png>
  and <https://raw.githubusercontent.com/Contrastive-LM/CLM/main/assets/agentic.png>
  (both read as images)
- Paper and blog: <https://contrastive-lm.notion.site>. It did not render,
  so it was **not read**.
- vLLM `truncate_prompt_tokens` ("use only the last k tokens … left
  truncation"): <https://docs.vllm.ai/en/v0.8.4/api/inference_params.html>
- Qwen3-8B GGUF sizes: <https://huggingface.co/Qwen/Qwen3-8B-GGUF/tree/main>
- Secondary, checked against the charts: MarkTechPost,
  <https://www.marktechpost.com/2026/09/23/contrastive-lm-releases-clm-8b-an-open-system-one-model-that-scores-agent-actions-up-to-9x-faster-than-jev/>
  (mislabels Pass@1 as CLM; does not mention Laya)
- Jev background: the repo sources in §0. The vendor (TypeSafe AI), the
  release date (2026-09-15) and the question types come from web search
  snippets, not from pages read in full: e.g.
  <https://simonwillison.net/2026/Sep/21/jev/> and
  <https://www.marktechpost.com/2026/09/23/a-coding-guide-to-typesafe-ai-jev/>.
- Repo, ours: `AGENTS.md` ("Laya", "Claims carry their evidence");
  `docs/LAYA-FACTCHECK.md`; `docs/FINDINGS.md` #20, #21, #22;
  `docs/PROTOCOL.md` rules 4, 6, 8, 10; `docs/PLAN.md` "Cut criteria";
  `config.yaml` (A4000 residents and measured MiB); `mcp/laya_service.py`;
  `mcp/skill_select.py`; `mcp/route.py`; `bench/eval_route_heldout.py`;
  `bench/route/eval_route.py`; `bench/hint_collapse.py`; `bench/retrieval.py`;
  `index/laya/route_in.json`; `scripts/train_laya.py`
