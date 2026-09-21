# Ship plan: what to build so it can be used by hand

Ordered by what unblocks hands-on testing, not by what is most interesting.
Every item states how it will be known to work.

---

## 1. Make the proxy the front door  (blocks everything else)

Right now llama-swap is on `0.0.0.0:1234` with no authentication, so anything
that can reach the machine bypasses the proxy entirely -- no key, no tools, no
account isolation, no logging. Verified: a direct request to `:1234` with no
key returned 200 while the proxy correctly returned 401.

`scripts/start-stack.bat` is already changed: llama-swap moves to
`127.0.0.1:11434` and the proxy takes `1234`. It needs a stack restart to take
effect, which costs a model reload.

**Done when:** `curl :1234` without a key returns 401, with a key returns a
completion, and nothing in Hermes had to be reconfigured -- it is already
pointed at 1234.

---

## 2. Reach the dependency indexes from a normal request

`three@0.185.1` is indexed -- 15,021 chunks, fetched in 0.8s and built in 32s
with no GPU -- and the model still cannot read a line of it, because nothing
in the proxy consults the package store. This is the strongest thing built and
it is currently unreachable.

Two parts:
- `deps.plan(root)` on repo detection, indexing the packages the repo actually
  imports (401 in the lockfile, 52 imported, 23 both) in the background
- a search path that falls back to the package index when the repo index has
  no answer, clearly labelled so an answer about `three` internals is never
  mistaken for the user's own code

**Done when:** asking "how does three decide to recompile a shader program"
in a repo that imports three returns a file from three's own source.

---

## 3. The reframing layer

Measured repeatedly today, and it is the finding with the most support:

    tool description rewritten as trigger conditions   10.7 -> 11.7
    candidates moved from `state` into `criteria`      0.500 -> 8/12
    six "never" rules removed                          9.3  -> 10.7
    negative framing -> positive framing               1/2  -> 2/2,
                                                       margin 0.06 -> 0.493

None of those touched a weight. So the proxy should restructure a request to
be answerable, under two rules that are not negotiable:

- **Augment, never replace.** The user's words stay. Structure is added around
  them. Rewriting "why is this slow" into a retrieval query answers a
  different question well, which is worse than answering the real one badly.
- **Stream what was changed.** A hidden reframe that goes wrong looks like a
  stupid model; a visible one can be contradicted. The machinery exists --
  streaming already narrates tool calls.

**Done when:** a reframed turn shows the reframe in the stream, and the
reframing can be turned off with an env var so its effect is measurable rather
than assumed.

---

## 4. A decision API whose shape prevents today's mistakes

Docs get skipped; a schema that rejects the broken shape does not. Every rule
below was learned by getting it wrong today.

    goal      required, string       what is being decided
    options   required, 2..8 items   the things being compared
    evidence  optional, per option   computed facts, stated positively

- `goal` and `options` are **separate required fields**, so candidates cannot
  be duplicated into both. That mistake produced eight consecutive wrong picks
  at margins up to 0.919 -- confident and content-blind.
- **Permutation averaging is on by default.** The model is not permutation
  invariant: reproduced at a 27-33% flip rate, including one flip at margin
  0.441, far above any gate worth setting.
- **A margin below 0.15 returns `undecided`, not a pick.** Confidence is not
  usable -- it saturates to 1.00 above eleven options -- so the margin between
  the top two is the only honest signal.
- **Options cap at 8.** The checkpoint ships separate temperatures bucketed at
  2 / 3-5 / 6-10 / 11+, and each option is truncated to
  `max(4, 176/n_options)` tokens, so more options means less of each option is
  even read.
- **Instructions must be positive.** Negation is a documented weakness: "tried
  five times, zero results" reads as *topic: retrying* and it picks retry.
  Exhausted branches are removed mechanically before asking, never described.

**Done when:** the API refuses a request with candidates in `goal`, returns
`undecided` below the margin floor, and the luminance-style metric test passes
through the public surface.

---

## 5. Bootstrap

The README promises a bootstrap script and there is none. Detect GPUs, fetch
the models, write `config.yaml` from the template, create the venvs, start the
stack.

**Done when:** a clone plus one command serves a completion.

---

## What is deliberately NOT in this plan

- **Laya fine-tuning.** Calibration is done and honest -- ECE 0.279 -> 0.032 --
  and accuracy stayed at 0.507 against a 0.500 baseline, because temperature
  cannot create discrimination. But the failure was mine: "is this code
  correct" is an LLM question with no metric and no computed evidence. Given
  structured options with real numbers the same checkpoint scores 8/12
  untuned. Fine-tuning is not the unlock; putting an oracle in front of it is,
  and the oracles already exist.
- **Speculative decoding.** Rescoped to the single-sample path only, because
  parallel sampling and speculation compete for the same memory bandwidth and
  sampling also produces candidates and training labels.
- **The head-to-head.** Still the eval that settles the headline claim. It is
  runnable now via subagents with grep and read as the frontier arm, but it
  needs the gauntlet tasks authored first and it is not what blocks hands-on
  use.
