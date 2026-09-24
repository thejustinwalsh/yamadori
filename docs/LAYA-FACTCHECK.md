# Laya and the two hemispheres: fact-check for the meetup deck

Written 2026-09-23 for the six deck slides `hemispheres`, `callosum`,
`laya-use`, `laya-training`, `laya-findings` and `laya-public`. Every number
on those slides is listed below with its evidence: the script, the data and
the n. The table also says whether it was re-verified today, and how.

Nothing here ran on GPU 0. The re-runs were of three kinds:

- offline, on cached features, with `CUDA_VISIBLE_DEVICES=""`
- calls to the live Laya service on `:1237`, which is pinned to the A4000
- replays of recorded runs

No service was restarted and nothing was committed.

Status key:

- **RE-RUN** means re-executed today, and the result is given.
- **READ** means read today from the recorded artefact or code, and it matched.
- **DOC** means taken from a repo doc whose script and data exist, but not
  re-executed today.
- **UNVERIFIED** means no script plus data could be found. These claims are
  kept off the slides.

## Today's re-runs, and where their output is

The logs were written to an ephemeral session folder, so the output that
matters is reproduced below. The two new probes are kept in the repo so the
numbers stay re-runnable:

- `bench/laya_factcheck/live_route_check.py`
- `bench/laya_factcheck/tail_probe.py`

1. **`bench/eval_route_heldout.py`.** Run in `.venv-laya` on the cached
   held-out features, copied out of `index/laya_staging_20260922_191425/`
   ("feature cache hit for all 120 held-out rows"). The run used:
   - `--old index/laya/_backup_20260922_191425/route_in.json`
   - `--new` set to a copy of `index/laya/route_in.json`

   Output was written to the scratchpad so that nothing landed in
   `index/laya/`. The result was **identical** to the recorded
   `heldout_log.txt`. The shipped head is byte-identical to the staged one
   (same sha256, `0390aa73…`).

   ```
   investigate-vs-not: old 64/120  new 80/120  rule 73/120  selection 89/120
   3-way:              old 59/120  new 66/120  rule 61/120
   hard slice (n=28), investigate-vs-not: old 11  new 11  rule 18  selection 18
   McNemar: old vs new p=0.0113 (10 vs 26); rule vs new p=0.401 (22 vs 29);
            selection vs new p=0.1996 (24 vs 15)
   new head coverage at its own gate 0.15: 105/120, 76/105 right on answered
   ```

2. **`live_route_check.py`.** This sends the same 120 held-out rows through
   the **live** `/route` (`engine: trained`). It then runs
   `selection.decide` with that live signal handed in, which is the
   combination production runs.

   ```
   live /route vs offline new-head argmax: 120/120 identical choices; all engine=trained
   live abstain 15/120 (gate 0.15); live head investigate-vs-not 80/120
   round trip: median 66 ms, p90 86 ms (HTTP included)
   selection.decide, Laya absent:              89/120
   selection.decide, live Laya signal:         91/120
     hard slice (n=28): alone 18, with Laya 14
     decisions Laya flipped to investigate: 14 (8 right, 6 wrong)
     missed investigations: 18 -> 10; unneeded investigations: 13 -> 19
   ```

   Exact McNemar for 91 vs 89 is on 8 vs 6 discordant rows: **p = 0.79**.
   Laya and `decide` are both deterministic, so one run is the result, not
   a sample of it.

3. **`scripts/eval_judge.py`**, run against the live `/decide`. It
   reproduces the cut exactly:

   ```
   DISCRIMINATION (passage fixed, question varied): gaps +0.53, +0.04, +0.08 -> PASS
   DECISIONS: 6/10 correct (coin flip is 5/10) -> do NOT expose `judge`
     misses: 3 of 5 false statements accepted (p=0.69, 0.68, 0.67), 1 of 5 true rejected (p=0.30)
   ```

4. **`tail_probe.py`**, run against the live `/decide`. It asks one binary
   `choice`, "Which language is the code in the state written in?"
   (rust/typescript). The Rust snippet and the TypeScript snippet each go
   **after** n lines of log preamble, at 10 tokens a line:

   | preamble | state tokens | rust p(rust) | ts p(rust) | identical |
   |---|---|---|---|---|
   | 0 | 21 / 19 | 0.8001 | 0.5288 | no |
   | 40 | 421 / 419 | 0.7526 | 0.6006 | no |
   | 60 | 621 / 619 | 0.5832 | 0.5832 | **yes** |
   | 80, 120 | 821+, 1221 | 0.5832 | 0.5832 | **yes** |
   | snippet FIRST, 120 lines after | | 0.6983 | 0.5441 | no |

   The mechanism is in the code. `laya/common.py:build_sequence` keeps
   `state[:room]`, where `room = max_len - len(prompt+options) - 1`, and
   `max_len` is 512. The state is cut from the tail. How much of it survives
   depends on how long the question and its options are. The card says
   "~320 tokens for state" with the full 192-token option budget; with this
   short question the cut fell between 421 and 621 state tokens.

   A side observation: Laya put the TypeScript snippet at p(rust) = 0.53
   even with no preamble. That is one more zero-shot miss, and it is not
   used on any slide.

5. **Offline replays**, no GPU:
   - `bench/test_laya_calibration.py`: "all 46 checks passed -- docs/LAYA.md
     still describes the system"
   - `bench/test_hint_collapse.py`: 47 passed, 0 failed
   - `bench/data/guardrail_summary.json`, read directly

## Claims on the slides

### `hemispheres`

| claim | evidence | status |
|---|---|---|
| Main is 5/8 and helper is 3/8 of one KV pool | `mcp/budget.py` `MAIN_SHARE=0.625`, `HELPER_SHARE=0.375` | READ |
| Pool 163,840, so main 102,400 and helper 61,440 | `config.yaml` `-c 163840`; 163,840 × 5/8 and × 3/8 | READ |
| The split is an operator decision of 2026-09-22, not a measurement | `mcp/budget.py` docstring; AGENTS.md | READ |
| Main runs 2 lanes; the helper runs 1 job at a time, and a second is refused as HELPER_BUSY | `mcp/admission.py` `MAIN_LANES=2`, `HELPER_LANES=1`; `proxy.py` HELPER_BUSY | READ |
| Deep thinking sees the question, up to 1,500 chars of earlier user text, and read-only tools, but not the conversation | `selection.CONTEXT_CHARS=1500`; `shomen._investigate` (`context[:1500]`, SYSTEM plus question only); `proxy.deep_thinking_tools()` (no `bind_project_context`, no delegate, no image tool) | READ |
| Fan-out reads the conversation without tools and writes a second candidate after the answer | `proxy._fan_out`: `fanout.run(dict(payload, tools=[]))`; `fanout._helper_body` sets role helper, `share_n=1` | READ |
| Fan-out is **being built** | `mcp/fanout.py` (+954 lines) and `mcp/proxy.py` are uncommitted in the working tree. `docs/HANDOFF.md`: "Next deploy … fan-out code-aware selection". No measured result. | READ |

### `callosum`

| claim | evidence | status |
|---|---|---|
| Going in: the question plus up to 1,500 chars of earlier user messages | as above | READ |
| A seed word rides along (**being built**) | `shomen.investigate` calls `concept_seed.seed_for` and `phrase`. Uncommitted diff in `mcp/shomen.py`. | READ |
| Coming back: the finding. Cited paths are checked against what was retrieved, and an uncited finding is marked unverified | `shomen._finish`, which **annotates** and does not refuse (see discrepancies); `proxy._deep_thinking` injects under `FINDINGS_HEAD` | READ |
| Fan-out's winning code replaces the delivered code answer, and prose keeps the original (**being built**) | `proxy._fan_out` docstring, `DELIVERED_SELECTIONS`, `FANOUT_MARK` | READ |
| The searches stay behind under a trace handle | `shomen._TRACES`, `get_trace(handle)` | READ |
| Main-context tokens, median, went 8,326 → 2,410; n=26 paired | `bench/context_economy.py`, `bench/context_economy_results.jsonl`, `docs/CONTEXT-ECONOMY.md`. The deep-thinking arm was run twice (2,410 / 2,382); the in-context arm once. | DOC |
| 531 tokens cross back, median; 516 on the repeat | `docs/CONTEXT-ECONOMY.md` replication table | DOC |
| Total tokens 1.8–2.1× (1.76× and 2.06×); facts right 26/26 in both arms | same | DOC |
| Measured under a 1,400-char finding cap, which is now 6,000 | `docs/CONTEXT-ECONOMY.md` line 208; `shomen.MAX_FINDING_CHARS=6000` | READ |

The speaker notes add one known gap. Today a finding is injected whenever
`hops > 0`, even with 0 citations. See `docs/TRANSCRIPT-REVIEW-2026-09-23.md`
improvement 5: rows rs01 and rs03 A6 had `injected: true, cited: 0`.

### `laya-use`

| claim | evidence | status |
|---|---|---|
| The rule is a six-line regex plus a symbol lookup in held packages | `selection.rule_baseline`, `selection.defined_symbols`; docstring "THE HARD-SLICE CHECK" | READ |
| Laya `route_in` is a trained head that answers investigate, answer_directly or clarify | `index/laya/route_in.json` labels; `selection.laya_signal` (`engine: trained`) | READ |
| agree → as they say; disagree → runs; abstain → runs; down → the rule alone | `selection.decide`, lines 598–614 | READ |
| "Laya can add a deep-thinking run, never cancel one" | This follows from `decide`: `investigate = rule_says` when Laya agrees, and True on disagree or abstain. Laya never turns a rule `True` into `False`. | READ (derived from code) |
| 89 → 91 /120; missed investigations 18 → 10; unneeded 13 → 19; n=120; p=0.79 | `live_route_check.py` (re-run 2, above); held-out set `bench/laya_routing_heldout_packages.jsonl` | **RE-RUN (new measurement, not yet in the repo docs)** |
| Hard slice 18 → 14 (notes) | same | RE-RUN |
| 66 ms median round trip, n=120 (notes and `laya-public`) | same | RE-RUN |

### `laya-training`

| claim | evidence | status |
|---|---|---|
| The encoder is frozen; features come from one pass per question at batch size 1 | `docs/LAYA.md` F8 and "fp16 batch non-determinism"; `mcp/laya_head.py` | DOC and READ |
| Features are pooled [CLS] (1,024) plus the 3 option logits | `route_in.json` `kind: logistic_cls_logits`, `feature_spec: cls_logits`, W is 3×1027 | READ |
| A 3-class logistic regression, with weight decay and gate picked by CV inside the training split | `scripts/train_laya.py`; `train_log.txt` candidate table | READ |
| 289 labels: 113 / 115 / 61 | `route_in.json` `trained_on`. The files count 59 + 30 + 200 = 289 lines: `laya_routing_labels.jsonl`, `_ext`, `_packages_train`. | READ |
| The artefact records its regex comparison, and a retrain below the floors fails the tests | `route_in.json` `metrics.rule_baseline`; `bench/laya_baseline.json`; `bench/test_laya_head.py` | READ |
| Held-out scores: first head 64/120, regex 73/120, new head 80/120, regex+lookup 89/120; hard slice 11 vs 18 of 28 | `bench/eval_route_heldout.py` | **RE-RUN, identical** |
| New vs first head p=0.011; new head vs the rule p=0.20 | same | RE-RUN |
| Notes: in-set, zero-shot 0.478 (95% abstain) → trained 0.728 ± 0.037 (13% abstain); 8 splits of 86 | `route_in.json` `metrics`; `train_log.txt` | READ |
| Notes: the first head lost to the regex, 0.726 against 0.841 (n=89) | `index/laya/_backup_20260922_191425/route_in.json` metrics; `docs/LAYA.md` F11 | READ |

### `laya-findings`

| claim | evidence | status |
|---|---|---|
| Routing with a trained head: 80/120 held out | above | RE-RUN |
| Deterministic: live choices matched those from features cached a day earlier, 120/120 | re-run 2; `docs/LAYA.md` header (3 identical calls, 6 decimals) | RE-RUN |
| Binary "needs our code?": 36/37 on plainly worded questions, 2/9 on adversarial ones | `docs/LAYA.md` F5; `bench/data/laya_routing_runs_binary.json`; asserted by `bench/test_laya_calibration.py` | DOC plus test RE-RUN (46/46) |
| The window is 512 tokens and the state is cut from the end | `laya/common.py:build_sequence`; tail probe | **RE-RUN** |
| Truth judge 6/10 against a 5/10 coin flip; accepted 3 of 5 false statements | `scripts/eval_judge.py` | **RE-RUN, identical** |
| Fabrication check: "from the files" 29/29, AUC 0.667 | `docs/LAYA.md` F6; `bench/laya_distil_labels.jsonl`, `bench/data/laya_distil_runs.json` | DOC plus test RE-RUN |
| Picking one hint among near-twins: 33.7% against 71.9% for embeddings (n=89) | `docs/FINDINGS.md` #21; `bench/hint_collapse.py`, `bench/data/hint_collapse_runs.json` | DOC plus `test_hint_collapse.py` RE-RUN (47/47) |
| "Specific enough?": the same answer 59/59 times, at mean margin 0.65 | `docs/LAYA.md` F3 (0.652); `bench/data/laya_routing_runs*.json` | DOC plus test RE-RUN |
| Notes: guardrail AUC 0.711–0.725 against the regex's 0.690, every CI including 0 (n=156) | `bench/data/guardrail_summary.json` | READ (the dAUC CIs match FINDINGS #22) |

### `laya-public`

| claim | evidence | status |
|---|---|---|
| "a fast base to specialise, not a zero-shot decision engine" | model card, Honest Limits | READ, live card 2026-09-23 |
| Base 0.362 on typed-decisions, 400 cases / 2,000 decisions; majority 0.461; fine-tuned 0.766 | model card table | READ |
| Ships over-confident: ECE 0.466 → 0.081 after temperature refit | model card, Honest Limits. **The card gives no n for the ECE.** | READ |
| English state budget about 320 tokens | model card, Honest Limits | READ |
| MASSIVE English 0.783, from a run of 17,416 questions | model card, "Why Route" | READ |
| Ours: zero-shot 0.478 → trained 0.728 (8 splits of 86) | `route_in.json` | READ |
| We reproduced the cut | tail probe | RE-RUN |
| 66 ms median (n=120) | re-run 2 | RE-RUN |

## UNVERIFIED, and kept off the slides

| claim (where it appears) | why it is unverified |
|---|---|
| "Holding a passage fixed and varying the query separates real from nonsense by **0.497**; varying both collapses the gap to zero" (AGENTS.md; `docs/BUILD.md`; a `mcp/shomen.py` comment) | No script and no data in the repo produce it. It enters in commit 2500bdf, and `docs/BUILD.md` presents it alongside external-research claims. The nearest recorded evidence is `eval_judge.py`'s discrimination test. That test is fixed passage with the question varied, and its gaps were +0.53, +0.04 and +0.08 today. The *direction* of the rule is supported by `docs/LAYA.md` F3, F5 and F6 and FINDINGS #21, which show margins are not comparable across inputs. The number is not supported. |
| "A query placed after **~120 lines** of preamble is discarded" (AGENTS.md; `docs/BUILD.md`, "Reproduced here") | No artefact records it. The mechanism is verified, since the state is truncated from the tail to fit a 512-token window. But a line count is not a stable unit. At 10 tokens a line, the cut came after 40 lines, somewhere below 60. State it in tokens, not lines. |
| "Calibration on ~50 labels halved ECE and moved accuracy 64.6% → 75.1%" (a `shomen.py` comment; FINDINGS #16) | `docs/FINDINGS.md` calls it prior work "for a different question set, never productionised". No artefact was found. |

## Discrepancies found in repo text

These are reported only. None were edited.

1. **"The head still does not beat the rule"** (AGENTS.md, HANDOFF.md,
   `selection.py`). This is true against `selection.decide`, which is the
   regex **plus** symbol lookup: 80 vs 89 of 120, p=0.20, not significant.
   Against the bare six-line regex, the new head **wins**:
   - in-set: 0.728 vs 0.608, and `train_log.txt` prints "trained head WINS"
   - held-out: 80 vs 73, p=0.40, not significant

   The live `GET /heads` now returns `"beats_rule_baseline": true` for
   `route_in`, meaning against the bare regex. The docstring of
   `mcp/laya_service.py:beats_rule_baseline` still says "For route_in it does
   NOT: 0.841 vs 0.726". That describes the 89-label head and is now stale.
   The slides say "does not beat regex + symbol lookup".
2. **The route_in head is not "a logistic head on CLS features".** The
   shipped head is `logistic_cls_logits`: CLS plus the 3 option logits, 1,027
   features. The CLS-only head was the 89-label one.
3. **`shomen._finish` does not refuse an uncited finding.** Its docstring
   says "refuse a finding that has none", and the module docstring says "a
   finding is refused unless it cites paths". The code appends an
   "unverified" note and returns `ok: True`. `proxy._deep_thinking` then
   injects it if `hops > 0`, which is transcript-review improvement 5. The
   slide says "marked unverified", which matches the code.
4. **`shomen.py`'s docstring says autonomous delegation "is deliberately not
   here".** This is stale. `selection.select` now triggers deep thinking per
   request, via `proxy._deep_thinking`.
5. **FINDINGS #21 and #22 say "Laya has no job in this stack … nothing gates
   on it".** This was superseded on 2026-09-22 by the retrained head and
   selection's two-signal design (HANDOFF "Stale claims" table). Laya's
   signal is live on the request path, but it can only add investigations.
   `logs/proxy.log` has 5 selection lines where Laya answered; the rest were
   forced by benchmark headers.
6. **Latency.** `docs/LAYA.md` says ~23 ms per call, which is presumably the
   model's forward pass. Today's measurement is the HTTP round trip through
   `/route` with the trained head: 66 ms median.

## What is said publicly

Sources were read on 2026-09-23. The model card was fetched live and also
matched against the cached snapshot `1c5edc17…`. The live card adds a
"What's new in 0.3.7" section and says "The checkpoints themselves are
unchanged".

**What it is, in its authors' words.** The card is at
<https://huggingface.co/convaiinnovations/laya>. It calls Laya a
"non-autoregressive System 1 decision model". You give it a state and typed
questions (`choice`, `score`, `noul`), and it returns answers with
probabilities in one forward pass.

- English root: ModernBERT-large, 421M, a 512-token window.
- Multilingual: mmBERT-base, 322M.
- typed-decisions: fine-tuned, 421M.
- Trained with RL against proper scoring rules ("RLCD"). Apache 2.0.

Their stated limits, from the "Honest Limits" section:

- Base checkpoints are near chance zero-shot on typed-decisions: 0.362
  against a 0.461 majority baseline.
- The model is "a fast base to specialise, not a zero-shot decision engine".
- It ships over-confident: ECE 0.466 → 0.081 after refitting temperature.
- The state gets about 320 tokens.
- Choices with many options degrade: Banking77 0.425.
- `score` is the weakest primitive.

Their headline results are:

- 0.766 on typed-decisions (fine-tuned checkpoint, 2,000 decisions)
- AG News 0.950
- MASSIVE-en 0.783
- XNLI-en 0.860
- 32.8–39.5 ms on a T4

Other official sources:

- Repo: <https://github.com/NandhaKishorM/laya>
- PyPI: <https://pypi.org/project/laya/>
- Write-up: <https://dev.to/nandakishor_m_6cc0adfde9f/i-built-non-autoregressive-decision-models-a-year-ago-then-a-frontier-lab-called-it-a-18me>

**Where public reports and ours agree.**

- The authors themselves say zero-shot is weak and fine-tuning is where the
  accuracy comes from. We found the same: zero-shot `route_in` scored 0.478
  at 95% abstention; our trained head scored 0.728.
- The truncation is documented on the card (about 320 state tokens, with the
  tail cut). A community thread reports it independently: HF discussion #15,
  titled "…blind past 320–768 tokens". We reproduced it today.
- The card says probabilities need per-deployment temperature fitting. We
  found that margin is not confidence (LAYA.md F3, F5), and that temperature
  cannot change accuracy (F10).

**Where they differ, and why that can be legitimate.** Positive reports mostly
use Laya the way the card intends:

- short input
- a fixed state
- a handful of clearly different options
- tasks such as email triage, intent, and routing between models

One example is <https://astgl.com/p/local-laya-vs-hosted-jev-typed-decisions>
(a coding agent choosing a model: 37/40 against 33/40 for a hand rule, as
reported). That matches where it holds up for us.

Where it failed for us, we had asked it a different kind of question:

- whether a claim is true (the judge)
- whether prose describes files it cannot see (grounding)
- which of several near-identical hints fits

None of those is what the card advertises. We are not saying it is a bad
model. Our measured scope is this: it works as a fixed-state, few-option
routing signal once a head is trained on our labels, and it was not a good
truth-judge for us, at 6/10 against a 5/10 coin flip.

**Community reports.** These came from a research pass; **the numbers were
not reproduced by us**. The thread titles and numbers were confirmed on
<https://huggingface.co/convaiinnovations/laya/discussions> on 2026-09-23.

- **#15, zero-shot agent step guard, as reported: 61 transcripts, 244
  decisions.** It agreed when agents said "done" over failing tests, and
  escalated 0 of 8 destructive commands. The state was cut at about 320
  tokens on English.
- **#7.** The title calls Laya "an embedding/similarity matcher, not a
  context-understanding model". Reported: Chinese voice commands scored 10/20,
  against 19/20 for a regex.
- **#6.** An independent Laya vs Jev head-to-head on 310 decisions. Reported:
  Laya 0.80–0.88 against Jev 0.89–0.99. The harness is
  <https://github.com/instax-dutta/sysone-bench>.
- **#8.** K-12 email triage on 6 emails: "what worked, what didn't".
- **#9.** Catalog selection, reported low. Laya scored 23–30/100, and simply
  taking the first retrieved candidate scored 86/100.
- **#14.** MASSIVE was reproduced at 78.33%, but own-data results were weak.
- **#3.** The multilingual model on 100 scenarios: 40.3%, ECE 0.313.
- **#2.** The shipped temperature for 11+ options saturates confidence to
  1.00, including on wrong answers.
- **#10.** Third-party leaderboards, reported:
  - JevBench (<https://benchmarkheaven.com/jev-models>): Laya 54.4 against
    Jev's 74.4
  - Decision Index
    (<https://huggingface.co/spaces/multimodalart/jev-decision-index>): Laya
    16.4 against Jev's 59.5

Also reported, not checked by us:

- a Hacker News thread, <https://news.ycombinator.com/item?id=49765348>
- a r/LocalLLaMA post, which could not be fetched
- repo star counts
- figures from the dev.to write-up that conflict with the card: 83.8%
  in-task and 65.1% zero-shot, against the card's 0.362 zero-shot

Treat all of these as leads, not results.
