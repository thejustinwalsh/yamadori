# jjava: asking our decider well

**jjava** is the operator's name for our Jev: the Bonsai decider
(`mcp/decider_bonsai.py`, with the question sets in `mcp/decide_turn.py`).
It reads the served Bonsai 2 27B at one token per option order and answers
typed questions in **Jev's API shape** (operator, 2026-09-29: "look to the
Jev api for the answer, they got it right"). This page is how to ask it
well: the API as built, who owns thresholds, the rules for writing a
question and its state, and what each of our question sets should look
like. Every rule names its source (list at the end). Where a source is
TypeSafe's own documentation it is marked **[TS]**; the community guide
"Jev in the Agent Loop" (x.com/N01ennn/status/2103542021071978601; the
post is script-rendered and was read through the coordinator's summary,
not first-hand) is **[AL]**; our own measurements are **[ours]** with their
script and n.

---

## 1. What jjava is, and is not

- **A decision layer beside the generation model, never in its context.**
  [AL] Main (the model the client talks to) generates; jjava answers
  narrow, typed questions about main's situation, and CODE acts on the
  answers. Its output never enters main's prompt as text; what reaches main
  is what the code decides to do (a skill injected, a plan run, a fallback
  taken). TypeSafe says the same about Jev: it "does not generate text,
  write code, or hold a conversation" and is not "a drop-in replacement for
  the LLM behind" a coding agent [TS coding-agents].
- **Typed questions over one state.** "Each request evaluates one state
  against one or more questions. All questions see the same state and are
  evaluated independently" [TS state]. jjava places the state once on its
  slot (the lane) and appends each question after it; a question never
  sees another question.
- **Not a calculator, a clock or a generator** [TS jaggedness 2, 3, 9]:
  arithmetic, counting, date comparison and extraction belong in code (an
  extraction becomes a Choice over candidates code found).

## 2. The API, as implemented

```python
import decider_bonsai as D            # or decide_turn.decide(...) in serving code
out = D.decide(state, [
    D.q_noul("done", "Every deliverable named in the goal now exists."),
    D.q_choice("phase", "Which phase is the agent's work in?",
               ["planning the work", "implementing the work",
                "debugging a failure", "verifying or testing the work"],
               keys=["plan", "implement", "debug", "verify"]),
    D.q_score("severity", "How severe is the failure?",
              ["cosmetic", "degraded, a workaround exists", "blocking"]),
])
out["answers"]["done"]      -> {"type": "noul", "name": "done", "noul": 0.93, "diagnostics": {...}}
out["answers"]["phase"]     -> {"type": "choice", "choice": "implement",
                                "probabilities": {...}, "confidence": 0.81, ...}
out["answers"]["severity"]  -> {"type": "score", "score": 1.24, "probabilities": {"0": ..},
                                "confidence": 0.64, "legend": {"0": "cosmetic", ..}, ...}
out["usage"]                -> {"input_tokens": ..., "output_tokens": ...}
```

Jev's request form is accepted as is: `{"type", "name", "instructions",
"criteria"}` with a Choice's criteria a `{key: description}` map, a Score's
a list of levels (lowest first), a Noul's an optional `{"true", "false"}`
pair (`decider_bonsai.typed`).

| primitive | answer (Jev's fields) | how it is computed | source |
|---|---|---|---|
| noul | `noul`: the probability the answer is yes | the averaged probability of the "yes" option (below) | [TS noul]: "a single number representing the probability that the answer is yes"; "There is no separate `confidence` value for a Noul" |
| choice | `choice`, `probabilities` (per key, sum 1), `confidence` | `choice` = the highest probability; confidence below | [TS choice]: "`choice`: The option with the highest probability"; "The sum of all values is 1" |
| score | `score`, `probabilities` (keyed "0".."k"), `confidence`, `legend` | `score` = sum of level number x probability; levels numbered 0..k in the order given | [TS score]: "each level number multiplied by its probability, added up"; "`legend`: Each level number mapped back to its description" |

**Confidence** (Choice and Score): `clamp((n x p_max - 1) / (n - 1), 0, 1)`
over the n options the distribution is spread over (`decider_bonsai.
confidence`). [TS confidence] gives "(3 x largest probability - 1) / 2" for
three options in its prose, and its own demo code computes `(count x peak -
1) / (count - 1)` clamped to [0, 1] -- the n-option form -- while calling
the three-option form an approximation. Checked: every one of the 17
distinct API-response examples in the docs (2-5 options, Choice and Score)
agrees with it within the rounding of the two decimals the docs print
(`mcp/test_decider_bonsai.py` [confidence]). Two illustrative widget
examples on the Score page do not fit (5 levels `[0, .14, .86, 0, 0]` shown
0.89, the form gives 0.825; 4 levels `[0, 0, .48, .52]` shown 0.52, the form
gives 0.36) and the Score page states no formula of its own. So the
n-option form is **derived** from the docs' demo code and response
examples, not quoted as TypeSafe's definition; `confidence()` is the one
place to change if TypeSafe publishes another.

**What jjava adds, kept out of Jev's fields** (`diagnostics`, logged with
every decision, never acted on by the decider): each of the two option
orders' distributions, their total-variation `disagreement`, whether their
argmaxes agree, `label_mass_min` (how much of the whole next-token
distribution landed on the option letters), the margin, and `tie` (the top
two within the measured `TIE_BAND` = 0.0034, the batch-nondeterminism floor:
`bench/decider/bonsai_decider.py` `batching`, n=100 questions, one run --
Bonsai's; another model reads its own, section 7).

**The readout underneath** (`READOUT_VERSION` typed/2; DECIDER-RESEARCH.md
4.2 item 3): options printed with positional letters, a noul as the lettered
pair "yes" / "no" (with criteria, "yes: <true>" / "no: <false>", unmeasured
on Bonsai), read in the given order and its reverse, "none" in the middle of
both; the two distributions AVERAGED are the answer's probabilities, and
the confidence is computed on the average. No label prior is divided out,
except the one question it was measured to help (the weak-package veto).

**Where jjava differs from Jev, and why:**

| | Jev | jjava | why |
|---|---|---|---|
| options per Choice | 255 | 26 | one single-token letter per option; the served Bonsai does not answer double letters (skill_match LABELS, 2026-09-27). Larger sets go in rounds (verify_moment CHUNK) or two stages (5.1) |
| Score levels | "the API accepts up to 10" [TS score] | 2-26 | the same letter limit |
| questions | "evaluated in parallel" [TS fan-out] | read one after another on one slot, the state cached | llama-server; each question costs two one-token reads of its own suffix |
| determinism | not deterministic across calls: 90.8% plurality agreement over 15 repeats [TS consistency choice] | batch nondeterminism below 0.0034 [ours] | quantized logits are not batch-invariant (DECIDER-RESEARCH 2.9) |
| state | 32k tokens for the state plus the longest question [TS models] | cut head+tail to 2,048 tokens by the question sets (`decide_turn.STATE_TOKENS`, operator 2026-09-27) | the configuration every comparison measured; see 3.2 |

## 3. Thresholds belong to the question set

The decider returns belief (probabilities, the noul) and certainty (the
confidence). **It never thresholds** -- its built-in abstain and per-model
params file were removed 2026-09-29. The QUESTION SET's code owns what an
answer means for the action it gates (`decide_turn.THRESHOLDS`):

- **Three tiers** [TS confidence, "Three paths for using confidence in your
  code"]: high -- act; medium -- proceed with caution (the source says
  "(medium)"; a cautious action where one exists); low -- do not act, fall
  back (the rule for a fact, no pick for a selection, keep for a package).
- **Per question set, per model.** "A confidence threshold is not one
  number. Different actions within the same system should be gated at
  different levels depending on the consequences of getting it wrong" [TS
  confidence]. A probability level is a property of the model reading it
  (DECIDER-RESEARCH 2.9: levels do not transfer across precision), so
  Bonsai, Flash-Next and Mirai S each get their own rows.
- **A safe default destination, and confidence required to leave it.**
  [AL] Every question set names what happens with no confident answer (its
  rule, no injection, keep) and the answer must earn its way off it. The
  community post's 0.85, like TypeSafe's own 0.85 / 0.6 in the voice-banking
  example [TS confidence-routing] and 0.30 in the skill cookbook [TS
  skill_suggestion], are **placeholders to tune, not numbers to adopt**:
  "The correct threshold values depend on your domain and the performance of
  the model for your use case" [TS confidence].
- **A noul near 0.5 means the state does not separate the cases.** [AL]
  [TS noul: "A value near 0.5 means the model gives yes and no similar
  probability"; "Values in the middle can go to a person"]. It is an
  escalation trigger (the rule, a deep-thinking run, the user), never a
  coin to flip. A noul's tiers are bands on the noul itself, each side
  tuned apart (`act_yes`, `act_no`, `caution_yes`, `caution_no`): raise
  the yes bar where a false yes is expensive [TS noul].
- **No structural invariants** [TS jaggedness 8]: a noul threshold does not
  carry to a Choice of the same question; P(q) + P(not q) need not be 1.
  Tune each question as asked.
- **None ship untuned** (operator). `THRESHOLDS` is empty; every tier reads
  "untuned" and each question set acts on the argmax as it did.

**The tuning loop:**

```
python bench/decider/label.py --question phase --n 40        # the operator labels decisions
python bench/decider/tune.py --stats                         # labelled n per model / question set
python bench/decider/tune.py --question phase --model bonsai \
    --high-accuracy 0.95 --medium-accuracy 0.85              # the OWNER's targets
```

`tune.py` shows accuracy against confidence (or the noul) in 10
equal-width bins (JevBench's ECE convention), the top-label ECE, and a
PROPOSED row: the lowest confidence whose decisions at or above it meet the
target at the Wilson lower bound (at the repo's ALPHA, 0.05), with its n
and each tier's coverage computed through `decide_turn.tier` itself. It
writes nothing. The owner pastes an accepted row into `THRESHOLDS` with
its n and date, or does not. [TS confidence]: "Start with conservative
thresholds, test with your own data, and adjust as you observe results."

## 4. Writing a question and its state

### 4.1 The question

- **Say exactly what you mean.** "`jev-1.13` answers the question you
  wrote, not the one you meant" [TS jaggedness 1]. Put boundary cases in
  the criteria; when explaining a wrong answer, that explanation is the
  missing half of the question.
- **Write a noul as a statement to judge.** [AL] "Every deliverable named
  in the goal now exists." rather than "Are we done?" [TS noul]: "The
  yes/no question the model answers, or a statement for it to judge."
- **One judgment per question.** Two judgments in one question can only be
  answered by guessing how to combine them; split them and combine in code
  [TS jaggedness 1; TS score: "Keep each Score question to one
  dimension"].
- **Score levels describe situations, not degrees.** "Broken or degraded
  feature, but workaround exists" gives the model something to match;
  "moderately severe" does not [TS score]. Their example: one report
  scored 0.55 at confidence 0.33 against levels "0", "1", "2", and 0.0 at
  confidence 1.0 against the three described levels.
- **Criteria extend the instruction; they never contradict it** [TS
  jaggedness 7]: a noul whose `true` means "no" performs worse.
- **No indirection.** A property of a property, a double negative or a
  multi-hop question costs accuracy; name the relevant part of the state
  directly [TS jaggedness 4].

### 4.2 The state

- **Only what the question needs.** "Accuracy falls as the state grows with
  content unrelated to the decision" -- filter in code first, or use a
  Noul to filter for relevance [TS jaggedness 5, "Large state full of
  irrelevant detail"]. [ours] measured the same on Bonsai: pagoda-h4's
  reads_package question scored 0.888 on the step alone, 0.784 on 2,048
  tokens of history, 0.410 on the whole history (`bench/decider/results/
  bonsai.json`, in-sample). Every question set's state is therefore the
  current turn or step, cut head+tail to 2,048 tokens.
- **Name the parts.** "Use an object for most requests so each part of the
  state has a descriptive name" [TS state]. A step's state names who did
  what ("Assistant called terminal: ...", "Result of terminal: ...").
- **The state is data** [TS jaggedness 6]: adversarial content in it can
  move the answer. Fetched text reaching a question is screened first
  (`mcp/skill_screen.py`), as everywhere.
- **Arithmetic, counts and dates are computed, then stated** [TS
  jaggedness 2, 3]: pass "3 of 12 files written", not the twelve paths and
  a question about how many.

### 4.3 The options

- **Build the option list live from what is available now.** [AL] The
  client's own tools this request (verify_moment), the skills the evidence
  admitted this turn (skill_select), never a fixed table.
- **Say "none" when the list may not be exhaustive** [DECIDER-RESEARCH 1.2,
  from TS choice]. jjava prints it in the middle of both orders.
- **At most 26 per question.** More: two stages (5.1) or rounds.

### 4.4 Many questions, one state

"Putting all of the questions your system needs in a single request, and
then using code to decide what is relevant after the fact" [TS fan-out],
speculative ones included. [AL] makes the same point. On jjava the state is
placed once and each question is a short suffix after it, so a second
question costs two one-token reads of its own text, not a re-read of the
state (the decider's latency table: 64-255 processed tokens p50 496 ms, p90
577 ms; `mcp/decide_turn.py`, from bench/decider/bonsai_decider.py) -- cheap,
but not free as on Jev.

## 5. Patterns, mapped to our question sets

| pattern | source | our question set | state it needs | today | what to measure |
|---|---|---|---|---|---|
| **5.1 Two stages for a large candidate set**: filter in code, Score the survivors, Choice among a shortlist, a "fits" noul per candidate as the gate | [AL]; [TS skill_suggestion]: a Choice over all 182 skills, the top 3 re-read with full descriptions, "fits" nouls under 0.30 dropped; wrong loads 16.8% -> 7.3%, needless 9.8% -> 4.0% | the skills injector (`skill_select` filter rounds -> `skill_match.ask_rounds` -> `decide_turn.choose`) | the turn or step (2,048 tokens); each candidate's description, then its full description for the shortlist | filter by evidence in code, then one Choice per area with "none", repeated without the pick; no Score stage, no per-candidate noul | bench/skills/test_daily_eval.py MUST / MUST-NOT; labelled `choose:*` decisions (label.py); wrong vs needless injections, the cookbook's two error rates |
| **5.2 Query-aware compaction**: Score each chunk drop / one-line summary / keep in full, instead of rewriting | [AL]; [TS classifying_rag_passages]: per-passage nouls (relevant, usable, contradicts, injection) routed in code | tool-result relevance (not built) | the task's goal plus ONE chunk (a tool result), never the whole history (4.2) | nothing: the proxy never rewrites a client's history (the ledger keeps every request extending the slot) | offline first: labelled chunks from relay logs, accuracy by level; any use is an operator decision |
| **5.3 Stuck detection**: two nouls every N steps -- "The last steps moved measurably closer to the goal." and "The agent is repeating an action that already failed." | [AL] | struggle | the goal plus the last N steps, compact | a RULE (`deep.struggle_scan`): 5/5 on pagoda-h4 with no false alarm; the decider found 0-2 of 5 at every context size [ours, bonsai.json h4] | only as an additive signal beside the rule, on the h4 and v0e-p2 replays (bench/octopus/replay_struggle.py), against the rule's hits |
| **5.4 Completion = a noul plus a code check** | [AL] | verify / plan_done | the goal (the plan's FILES) and the files written | mechanical (`verify_moment`: every file of the plan written, the entry written); the decider picks the TOOL that runs it (a Choice over the client's tools) | labelled verify decisions; whether the named line was followed |
| **5.5 Intent** | [TS intent-routing]; [AL] | build intent (noul) | the user turn plus the previous assistant turn's last 255 tokens | the decider, the rule (`route.work_intent`) only when unavailable | work_intent.jsonl (n=99, in-sample); labelled traffic; its tiers (a noul near 0.5 falls back) |
| **5.6 Phase** | [AL] ("which phase") | phase (Choice, 4 keys) | the turn or step | the decider; the rule compared, disagreements logged | the first label batch: 115 legacy "decider verify, rule implement" rows (label.py --pattern phase:verify/implement) |
| **5.7 A stop with no call** | ours | judge_stop (Choice, "none" in the middle) | the reasoning's tail (255 tokens) and the message | the decider continues a step that only stated its next action | no labelled set yet: label the `choose:stop` decisions |
| **5.8 A veto of a weak detection** | ours | package:<p> (noul, content-free prior) | the turn or step | kept 33/33 true detections as a veto (in-sample) | labelled package decisions |

A noul near 0.5 on any of these is the escalation signal of 3: the rule,
the user, or deep thinking -- never the argmax.

## 6. Measuring jjava

- **Our decisions**: every one is logged with its confidence and join keys
  (`logs/decider_decisions.jsonl`); label with `bench/decider/label.py`,
  tune with `bench/decider/tune.py` (3).
- **The legacy readout, once**: `bench/decider/legacy_vs_typed.py --run`
  (GPU) compares the 2026-09-27 readout with this one on the daily eval's
  skill picks and on build intent; then the legacy readout is deleted
  (docs/CONSTANTS-AUDIT.md "Decider legacy readout").
- **Against the field**: `bench/decider/jevbench_run.py --run` (GPU) asks
  JevBench's 231 public decisions (github.com/fstandhartinger/jevbench at
  bb05a33, MIT; the dataset stays in its clone) as typed questions through
  mcp/model.py and scores them with JevBench's own code, beside the Jev
  1.13.0, reflex-27b and SimpleJev Qwen3.8-27B rows of the v1.3.0 release on
  the same items. Only same-release, same-item numbers compare
  (DECIDER-RESEARCH 1.5).

## 7. Per model (2026-09-29)

jjava reads **whichever model serves the conversation** (`decider_bonsai.
model_name()`: `bonsai` by default, `flash-next` at tier max, `mirai-s` at
xhigh when it lands). A label's probability is a property of the model that
reads it (DECIDER-RESEARCH 2.9), so nothing measured on Bonsai is assumed
for another model.

**What was measured on Bonsai only, and how it is keyed now**
(`decider_bonsai` THE MODEL PROFILE):

| constant | where | Bonsai's evidence | per model |
|---|---|---|---|
| `TIE_BAND` 0.0034 | `decider_bonsai` | `bonsai_decider.py batching`, n=100, one run | **used** through `tie_band(model)`: the model's measured record, else Bonsai's value **recorded as unmeasured for that model** (`tie_band_of`: `measured: false`, the source says so) -- what every model got before the keying; `TIE_BAND` stays Bonsai's (skill_match reads it) |
| label tokens (A..Z, yes/no spellings; single letters only) | `LETTERS`, `LABEL_SPELLINGS`, `spelling_ids` | `skill_match --check-labels` + the double-letter replay, 2026-09-27 | ids resolved per model at runtime (unchanged); the profile's `labels` records the model's own check |
| typed/2's choices (two orders, no prior division, the package veto's prior) | `READOUT_VERSION`, `read` | `bonsai_decider.py calib`, in-sample | unchanged for every model; `readout` listed **unmeasured** for any model but Bonsai |
| `FORM` (legacy) | `decide_turn` | `calib`, in-sample | unchanged; `legacy_form` unmeasured for other models |
| label priors (`TEMPLATE_CF`) | `decide_turn` | read live per process | already keyed by model under max mode (`_cf_key`); the measured `letter_prior` / `label_bias` are recorded per model |
| `THRESHOLDS` | `decide_turn` | none tuned | keyed by model; an alias (`bonsai-agent`) reads its model's row, another model's row is never borrowed |

Not keyed (not a per-model measurement, or not the decider's): `STATE_TOKENS`
(operator), `PREV_TOKENS` (Bonsai's latency table: a latency bound, re-derive
if Flash-Next's latency matters), `FIRST_K` (llama-server's own default),
`NEUTRAL_STATE` (the paper's input). `skill_match._tie_band` reads
`decider_bonsai.TIE_BAND` (Bonsai's) for every model -- `skill_match.py` is
not the decider's file; it should call `decider_bonsai.tie_band()`.

**Reported:** every typed answer's `diagnostics` carry `model` and
`tie_band {value, measured}`; every decision row the same; a typed batch and
`Turn.record()` carry `model_profile` (`profile_status()`: measured fields,
unmeasured fields, the note "unmeasured for <model>: ...").

**The record** the runtime reads: `bench/decider/results/models/<model>.json`
(`decider-model/1`; `YAMADORI_DECIDER_MODELS_DIR` moves it), one field per
part, each with its date, script, n and the engine (mode, base URL, the
server's model file and build). A field without script and date, a band
outside [0, 0.5), a record naming another model or another version is
rejected and the status says why. A measured field replaces the built-in
one -- for Bonsai too, once someone runs it on Bonsai.

**The one command** (the engine owner, in a GPU window; one GPU consumer):

```
C:\Users\jwals\textgen\installer_files\env\python.exe bench/decider/measure_model.py --run --model flash-next --base-url http://127.0.0.1:18095
```

Without `--run` it prints the plan and sends nothing. `--base-url` is a
bare llama-server on a test port (the ISTA GGUF, `--jinja` so the GGUF's
template renders, `/slots` on, at least ~12k context on the slot it uses --
one tie-band state is 11,412 tokens). Without `--base-url` it reads
llama-swap's `/upstream/<model>`, and only if `/running` already lists the
model (nothing here loads a model or swaps the card). `--slot N` picks the
server's slot (default: its highest). Parts, in order (`--only` picks):
`labels` (64 `/tokenize` calls, no generation), `letter_prior` (50 one-token
reads: option counts 2..26, both renderings, every option "N/A"),
`label_bias` (6 reads: build_intent, phase, choose:stop on the content-free
state), `tie_band` (`bonsai_decider.py batching` unchanged: 200 reads, ~16k
tokens processed cached and ~71k uncached), `legacy_vs_typed` (intent: 99
rows x 2 arms x 2 reads; daily_choose: 102 rows x 2 arms, each the facts and
the selector's rounds). On Bonsai the tie-band part took 169 s
(`bonsai.json`); the rest is not timed on Bonsai as a whole, and Flash-Next's
prefill is unmeasured -- the short parts are minutes, the whole run likely
one to two hours on Flash-Next (an estimate from ISTA's ~108 tok/s prefill,
not a measurement). The record is written after every part; a busy card
stops the run (exit 2, NOT RUN), an unreachable or unloaded target is exit
3. Decisions go to `logs/decider_measure/<model>.decisions.jsonl`, never
the live log. The same `--model / --base-url / --slot` work on
`legacy_vs_typed.py`, `bonsai_decider.py` (not its stack-layout parts) and
`jevbench_run.py`.

Every number it writes is one run: repeat before citing a difference
(docs/PROTOCOL.md).

## Sources

**TypeSafe documentation** (docs.typesafe.ai, read 2026-09-29; vendor):
[confidence](https://docs.typesafe.ai/confidence.md) (TS confidence),
[primitives/choice](https://docs.typesafe.ai/primitives/choice.md) (TS choice),
[primitives/score](https://docs.typesafe.ai/primitives/score.md) (TS score),
[primitives/noul](https://docs.typesafe.ai/primitives/noul.md) (TS noul),
[concepts/state](https://docs.typesafe.ai/concepts/state.md) (TS state),
[introduction/coding-agents](https://docs.typesafe.ai/introduction/coding-agents.md) (TS coding-agents),
[models](https://docs.typesafe.ai/models.md) (TS models),
[model-jaggedness/jev-1.13](https://docs.typesafe.ai/model-jaggedness/jev-1.13.md) (TS jaggedness, numbered as its table),
[patterns/fan-out](https://docs.typesafe.ai/patterns/fan-out.md) (TS fan-out),
[patterns/confidence-routing](https://docs.typesafe.ai/patterns/confidence-routing.md) (TS confidence-routing),
[patterns/intent-routing](https://docs.typesafe.ai/patterns/intent-routing.md) (TS intent-routing),
[cookbooks/skill_suggestion](https://docs.typesafe.ai/cookbooks/skill_suggestion.md) (TS skill_suggestion),
[cookbooks/consistency_noul_cookbook](https://docs.typesafe.ai/cookbooks/consistency_noul_cookbook.md) and
[cookbooks/consistency_choice_cookbook](https://docs.typesafe.ai/cookbooks/consistency_choice_cookbook.md) (TS consistency),
[cookbooks/classifying_rag_passages](https://docs.typesafe.ai/cookbooks/classifying_rag_passages.md) (TS classifying_rag_passages),
[llms.txt](https://docs.typesafe.ai/llms.txt) and llms-full.txt (the response examples the confidence form was checked on).

**Community**: "Jev in the Agent Loop", x.com/N01ennn/status/2103542021071978601
(AL; read through the coordinator's summary: the page is script-rendered).

**Benchmark**: JevBench, github.com/fstandhartinger/jevbench, commit
bb05a335bc809e61b20c0f745d25499a82b326fc (MIT): `jevbench/scoring.py`,
`metrics.py`, `composite_v13.py`, RESULTS-v1.2.md (the v1.3.0 release).

**Ours**: docs/research/DECIDER-RESEARCH.md; bench/decider/results/
bonsai.json (bench/decider/bonsai_decider.py; every number in-sample);
mcp/test_decider_bonsai.py, mcp/test_decide_turn.py,
mcp/test_decider_tune.py, mcp/test_jevbench_run.py.
