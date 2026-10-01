#!/usr/bin/env python
"""A TYPED DECIDER ON THE MAIN MODEL: Bonsai 2 27B read at one token.

    import decider_bonsai as D
    out = D.decide(state_text, [D.yes_no("Is the user asking for something
                                          to be built, made or changed?"),
                                D.choice("Which phase is the agent in?",
                                         ["planning", "implementing",
                                          "debugging", "verifying"])])
    out["answers"][0]["answer"]      -> "yes" | "no"
    out["answers"][0]["probs"]       -> {"yes": 0.97, "no": 0.03}

    THE TYPED API (2026-09-29, Jev's API exactly; "THE TYPED API" below):
    out = D.decide(state_text, [D.q_noul("build_intent", "Is the user ..."),
                                D.q_choice("phase", "Which phase ...?",
                                           [...], keys=[...]),
                                D.q_score("done", "How finished ...?",
                                          ["not at all", ..., "completely"])])
    out["answers"]["build_intent"]["noul"]            -> 0.97
    out["answers"]["phase"]["choice"], ["probabilities"], ["confidence"]
    out["answers"]["done"]["score"], ["probabilities"], ["confidence"],
                          ["legend"]
    (every answer also carries `diagnostics`: the two orders, their
    disagreement, the label mass -- logged, never a threshold)

    The {kind: yes_no | choice} questions above (yes_no(), choice(),
    ask_one() on its own) are the LEGACY readout's forms: they stay only
    until bench/decider/legacy_vs_typed.py has run once on the GPU (the
    removal list: docs/CONSTANTS-AUDIT.md "Decider legacy readout").

Operator, 2026-09-27: "Seems bonsai and reading the weights is likely the best
we can do." A Jev-style decider (typed questions in, a distribution over the
question's labels out), built on the model already resident on the 5060 and
reached through the one door (mcp/model.py post()). Nothing is generated: one
forward pass per question, the next-token distribution read at the answer
position.

THE RENDERING (TEMPLATE_VERSION; the text is pinned by mcp/test_decider_bonsai.py)

  The SERVED chat template (llama-server renders /v1/chat/completions itself,
  the same template /apply-template and the proxy use), thinking OFF by the
  template's own switch (chat_template_kwargs.enable_thinking = false, what
  tiers.apply writes at tier `minimal`), four messages:

    system     SYSTEM                                  fixed
    user       STATE_HEAD + the state                  the shared prefix
    user       YES_NO or CHOICE, the question last     one per question
    assistant  ANSWER_LEAD, continued (a prefill)      "Answer:"

  so the prompt ends `<think>\\n\\n</think>\\n\\nAnswer:` and the next token is
  the label (" yes", " A" ...). THE THINK-BLOCK GUARD (below; docs/JJAVA.md
  8) refuses any body that could render otherwise before it is sent. The question sits in its OWN user message on
  purpose: this is a hybrid model, whose recurrent state cannot be rolled
  back token by token; llama-server keeps a context checkpoint at the start
  of the LAST user message (server-context.cpp "break at the last user
  message"), so the next question on the same slot restores the state there
  and processes only its own question. With the question inside the state's
  message that checkpoint would sit before the state and every question
  would re-read it.

THE READ (exact, and which)

  llama-server's `logprobs`/`top_logprobs` (-> n_probs) on the chat route
  returns, for the one generated position, the top-K tokens of the
  PRE-SAMPLING distribution (post_sampling_probs false: populate_token_probs
  -> get_token_probabilities, a softmax over every logit of the vocabulary,
  server-common.cpp). So each probability read is exact; the only question
  is whether a label is inside the top K. Each label is the set of its
  single-token spellings (LABEL_SPELLINGS, resolved once through /tokenize;
  " yes", "yes", " Yes", "Yes" ... all count as yes), summed. When a label
  has NO spelling in the top K, or what is unread could change the argmax,
  K is raised tenfold (FIRST_K = the server's own default of 20) -- a
  re-ask costs one token of prefill, the prompt is cached. A spelling still
  unread is bounded by the smallest probability returned (`unread_bound`,
  per spelling); `exact` says whether every spelling was read.
  NOT used: a grammar or logit_bias restriction. Both act in the sampler
  chain, after the distribution this reads -- they would change the token
  sampled, never the probabilities reported (and post_sampling_probs with a
  lazy grammar reports the unconstrained candidates whenever the first draw
  already fits). The renormalisation over the labels is done here instead,
  and `label_mass` (the labels' share of the whole distribution) is
  reported so a question the model wanted to answer with something else is
  visible.

  The decision is argmax; the distribution returned is renormalised over the
  question's labels. No threshold.

THE SLOT, AND KEEPING THE CONTEXT CLEAN (operator, 2026-09-27)

  The transient side-call slot (slots.acquire(None, transient=True): never a
  conversation's pinned slot), cache_prompt on, every question of a batch in
  sequence on that one slot -- the state prefix is processed once. Until
  layout v2 the slot was RELEASED as soon as the batch ended (slots.
  release_idle in the proxy, where releases are enabled; model.release_slot
  elsewhere). LAYOUT V2 (operator, 2026-09-29): that slot is THE LANE --
  budget.LANE_TOKENS cells kept in VRAM at a rank above the primary
  (slots THE LANE) -- and it is KEPT between batches (slots.lane_kept()).
  `release` in the result carries
  cells_before and the method, or why it was kept; keep_slot=True (a caller
  that asks again on the same state at once) skips it and says so. A slot
  the CALLER placed (benchmarks) is still released unless keep_slot.

FAILURES (AGENTS.md "Failure returns carry the next step"): DeciderUnavailable
carries the situation, whether a retry can help, and a remedy with an owner.
A question with no labels, or a choice with more options than single-token
letters, is refused before anything is sent (retryable: false).

Non-generation calls (/tokenize, /slots) go to the model server directly, as
proxy.count_prompt_tokens does; the generation goes through model.post.
Both doors (_upstream, _post_default) are looked up at CALL time when a
caller passes none, so a bench can point every read at one engine
(bench/decider/decider_target.py).

PER MODEL (2026-09-29; "THE MODEL PROFILE" below; docs/JJAVA.md "Per
model"): the decider reads whichever model serves the conversation
(model_name()). What was measured on Bonsai -- the tie band, the label
tokens, the readout's evidence -- is keyed by the serving model's name:
that model's measured record (bench/decider/measure_model.py) when one
exists, else Bonsai's value, and every result says which
(profile_status(), the answer's diagnostics `tie_band` {value, measured}).
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ------------------------------------------------------------ the template --
TEMPLATE_VERSION = "bonsai-decider/1"
SYSTEM = ("You read the material the user gives you and answer one question "
          "about it. Reply with the answer's label only.")
STATE_HEAD = "MATERIAL:\n\n"
YES_NO = "QUESTION: {question}\n\nAnswer yes or no."
CHOICE = ("QUESTION: {question}\n\nOPTIONS:\n{options}\n\n"
          "Answer with the letter of one option.")
OPTION_LINE = "{label}. {description}"
ANSWER_LEAD = "Answer:"

YES_NO_LABELS = ("yes", "no")
LETTERS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
# Every spelling that means the label at the answer position. Only the ones
# the tokenizer makes ONE token are read (resolved through /tokenize).
LABEL_SPELLINGS = {
    "yes": ("yes", " yes", "Yes", " Yes", "YES", " YES"),
    "no": ("no", " no", "No", " No", "NO", " NO"),
}
# The server's own default for top_logprobs (server-common.cpp:
# json_value(body, "top_logprobs", 20)); raised tenfold per re-read.
FIRST_K = 20
GROWTH = 10


class DeciderUnavailable(RuntimeError):
    """The decider could not answer. The situation, whether a retry can help
    (a fact), and a remedy with an owner."""

    def __init__(self, code: str, situation: str, retryable: bool,
                 remedy: str):
        super().__init__(f"{code}: {situation}")
        self.code, self.situation = code, situation
        self.retryable, self.remedy = retryable, remedy

    def facts(self) -> dict:
        return {"code": self.code, "situation": self.situation,
                "retryable": self.retryable, "remedy": self.remedy}


# ------------------------------------------------------------ questions ----
def yes_no(question: str) -> dict:
    return {"kind": "yes_no", "question": question.strip(),
            "labels": list(YES_NO_LABELS)}


def choice(question: str, options: list[str]) -> dict:
    """A choice over `options` (short descriptions), labelled A, B, C ... in
    the order given."""
    opts = [str(o).strip() for o in options]
    if not opts:
        raise DeciderUnavailable(
            "NO_OPTIONS", "a choice question was given no options", False,
            "the caller: pass at least two option descriptions")
    if len(opts) > len(LETTERS):
        raise DeciderUnavailable(
            "TOO_MANY_OPTIONS",
            f"{len(opts)} options; single-token labels run A..Z "
            f"({len(LETTERS)})", False,
            "the caller: split the choice into smaller ones")
    return {"kind": "choice", "question": question.strip(),
            "labels": list(LETTERS[:len(opts)]), "options": opts}


# ------------------------------------------------------------ debiasing ----
# docs/research/SKILLS-RESEARCH.md 2.5 and Part 4 item 2 (each source there):
#   (a) contextual calibration -- the same question asked of a CONTENT-FREE
#       state, NEUTRAL_STATE ("N/A", Calibrate Before Use, Zhao et al.
#       2102.09690), and the answer divided by it: W = diag(p_cf)^-1;
#   (b) permuted option order, averaged (PriDe 2309.03882; Permutation
#       Self-Consistency 2310.07712; the last-printed-option bias of
#       2607.05552): every question is asked in `orders`;
#   (c) neutral labels: a yes/no question is asked as a two-option CHOICE
#       (letters, "yes"/"no" as the options' text) in both orders, so no
#       label token is "yes" or "no" and neither answer is always last;
#   (e) near-ties are ties: TIE_BAND, below.
NEUTRAL_STATE = "N/A"
# TIE_BAND: two labels whose probabilities differ by no more than this are a
# TIE, and a tie is no decision (the caller's fallback decides). The number
# is OURS, MEASURED, not chosen: the largest difference in any label's
# probability between two reads of the same question on the same state that
# differ only in batch shape -- prefix-cached vs cache_prompt false -- over
# 20 states x 5 questions (bench/decider/bonsai_decider.py `batching`,
# 2026-09-27, max_prob_diff 0.0034; n=100 questions, one run). The doc names
# the cause (2607.17283: quantized logits are not batch-invariant, near-tied
# argmaxes flip) and gives no number.
# MEASURED ON BONSAI ONLY. This constant is BONSAI'S value (the module
# attribute other code reads: skill_match._tie_band); what the decider uses
# is tie_band(model) -- THE MODEL PROFILE, below -- the serving model's own
# measured band, or this one, recorded as unmeasured for that model.
TIE_BAND = 0.0034


def as_choice(q: dict, order: list[int] | None = None) -> dict:
    """The question as a CHOICE whose letters are neutral: a yes/no becomes
    options ["yes", "no"]; `order` permutes the options (indices into the
    original list). The result carries `meaning`: letter -> the original
    option (a yes/no's "yes"/"no", a choice's original letter)."""
    if q["kind"] == "yes_no":
        texts, keys = ["yes", "no"], ["yes", "no"]
    else:
        texts, keys = list(q["options"]), list(q["labels"])
    order = list(range(len(texts))) if order is None else list(order)
    c = choice(q["question"], [texts[i] for i in order])
    c["meaning"] = {lab: keys[i] for lab, i in zip(c["labels"], order)}
    return c


def orders_for(q: dict, n: int = 2) -> list[list[int]]:
    """`n` option orders: the given order, then its reverse (so the last-
    printed option changes), then the other cyclic rotations."""
    k = len(q["options"]) if q["kind"] == "choice" else 2
    base = list(range(k))
    out = [base, base[::-1]]
    for r in range(1, k):
        rot = base[r:] + base[:r]
        if rot not in out:
            out.append(rot)
    return out[:max(1, n)]


def meaning_probs(answer: dict, c: dict) -> dict[str, float]:
    """An answer to as_choice(q) mapped back to the original options."""
    return {c["meaning"][lab]: p for lab, p in answer["probs"].items()}


def contextual(p: dict[str, float], p_cf: dict[str, float]) -> dict:
    """Calibrate Before Use: p_i / p_cf_i, renormalised."""
    q = {k: v / max(p_cf.get(k, 0.0), 1e-12) for k, v in p.items()}
    z = sum(q.values()) or 1.0
    return {k: v / z for k, v in q.items()}


def average(dists: list[dict[str, float]]) -> dict[str, float]:
    keys = dists[0].keys()
    return {k: sum(d[k] for d in dists) / len(dists) for k in keys}


def decision(p: dict[str, float], band: float | None = None) -> dict:
    """{answer, tie}: argmax, or tie=True (answer None) when the two largest
    are within `band` (default: the serving model's tie_band())."""
    band = tie_band() if band is None else band
    s = sorted(p.items(), key=lambda kv: -kv[1])
    if len(s) > 1 and s[0][1] - s[1][1] <= band:
        return {"answer": None, "tie": True,
                "tied": [s[0][0], s[1][0]]}
    return {"answer": s[0][0], "tie": False}


def question_text(q: dict) -> str:
    if q["kind"] == "yes_no":
        return YES_NO.format(question=q["question"])
    lines = "\n".join(OPTION_LINE.format(label=lab, description=d)
                      for lab, d in zip(q["labels"], q["options"]))
    tpl = SCORE_DIGITS if q.get("label_kind") == "digits" else CHOICE
    return tpl.format(question=q["question"], options=lines)


def messages(state: str, q: dict) -> list[dict]:
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": STATE_HEAD + (state or "").strip()},
            {"role": "user", "content": question_text(q)},
            {"role": "assistant", "content": ANSWER_LEAD}]


def body(state: str, q: dict, slot: int | None, k: int,
         model_name: str, cache: bool = True,
         msgs: list[dict] | None = None) -> dict:
    """The chat body for one question. Sent as is (model.post), never
    re-shaped: tiers.apply would give it a thinking budget and an answer
    allowance of A_MIN; this generates one token with thinking off."""
    b = {"model": model_name, "messages": msgs or messages(state, q),
         "max_tokens": 1, "logprobs": True, "top_logprobs": int(k),
         "chat_template_kwargs": {"enable_thinking": False},
         "enable_thinking": False, "cache_prompt": bool(cache),
         "stream": False}
    if slot is not None:
        b["id_slot"] = int(slot)
    return b


# ------------------------------------------------ THE THINK-BLOCK GUARD ----
# SGLang's decision-model server (docs.sglang.io/docs/supported-models/
# decision_models, "How answers are computed", read 2026-09-30; validated
# there on Qwen3.8-27B, the base Bonsai 2's card names): it "turns thinking
# off for every question", refuses a request that re-enables it, and
# refuses a prompt that leaves a reasoning block open -- a label read inside
# an open think block is a reasoning token, not an answer.
#
# HOW OUR SERVER RENDERS A QUESTION (every engine tree in engines/src, the
# same code in each; checked 2026-09-30): a final assistant message is a
# CONTINUATION (tools/server/server-common.cpp: prefill_assistant ->
# continue_final_message AUTO, add_generation_prompt false; two assistant
# messages at the end are refused). common/chat.cpp resolves AUTO to
# REASONING when the message has reasoning_content and NO content, else to
# CONTENT; common/chat-auto-parser-generator.cpp then renders messages[:-1],
# the template's generation prompt cut at the reasoning start, the start,
# the reasoning, and -- for CONTENT only -- the reasoning END, then the
# content. So the block is CLOSED exactly when the prefill has content and
# no reasoning: "<think>\n\n</think>\n\nAnswer:" on Bonsai
# (mcp/test_decider_bonsai.py [guard], through the served template), and a
# reasoning-only prefill (proxy.directive_prefill's shape) leaves it OPEN.
# Thinking off matters too: the template's enable_thinking also decides its
# system text (the reasoning-effort lines, bonsai_chat_template.jinja 46).
#
# guard_body() checks both on EVERY body before it is sent (no network);
# prompt_guard() checks a RENDERED prompt (the served template offline; the
# live /apply-template in a window: template_check()). A failure is a
# DeciderUnavailable that is not retryable: it is this module's bug or a
# caller's rendering, never a model's answer.
THINK_MARKERS = (("<think>", "</think>"),)
THINK_START, THINK_END = THINK_MARKERS[0]


def _guard_fail(code: str, situation: str) -> DeciderUnavailable:
    return DeciderUnavailable(
        code, situation, False,
        "the maintainer: jjava's body must carry enable_thinking false (the "
        "body and chat_template_kwargs) and end on ONE assistant message "
        "whose content is the ANSWER_LEAD prefill and which carries no "
        "reasoning (docs/JJAVA.md 8)")


def guard_body(b: dict) -> dict:
    """Refuse a body whose rendered question could leave a think block open
    or run with thinking on. Returns {ok: True, continuation: "content"}."""
    kw = b.get("chat_template_kwargs") or {}
    if kw.get("enable_thinking") is not False \
            or b.get("enable_thinking") is not False:
        raise _guard_fail("THINKING_ON", "enable_thinking is not false in "
                          "both the body and chat_template_kwargs")
    if b.get("reasoning_effort") not in (None, "none"):
        raise _guard_fail("THINKING_ON", "reasoning_effort "
                          f"{b.get('reasoning_effort')!r} re-enables thinking")
    msgs = b.get("messages") or []
    last = msgs[-1] if msgs else {}
    content = last.get("content") if isinstance(last, dict) else None
    if isinstance(last, dict) and last.get("role") == "assistant" and any(
            last.get(k) for k in ("reasoning_content", "reasoning",
                                  "thinking")):
        raise _guard_fail("THINK_BLOCK_OPEN", "the prefill carries "
                          "reasoning: the server continues inside the think "
                          "block")
    if not isinstance(last, dict) or last.get("role") != "assistant" \
            or not isinstance(content, str) or not content.strip():
        raise _guard_fail("NO_ANSWER_PREFILL", "the last message is not an "
                          "assistant prefill with text: the server would "
                          "render a generation prompt (thinking's own) or a "
                          "reasoning continuation")
    if not content.endswith(ANSWER_LEAD):
        raise _guard_fail("NO_ANSWER_PREFILL", "the prefill does not end on "
                          f"{ANSWER_LEAD!r}")
    for s, e in THINK_MARKERS:
        if content.rfind(s) > content.rfind(e):
            raise _guard_fail("THINK_BLOCK_OPEN", "the prefill's text opens "
                              f"a {s} block it does not close")
    if len(msgs) >= 2 and isinstance(msgs[-2], dict) \
            and msgs[-2].get("role") == "assistant":
        raise _guard_fail("NO_ANSWER_PREFILL", "two assistant messages at "
                          "the end (llama-server refuses it)")
    return {"ok": True, "continuation": "content"}


def prompt_guard(prompt: str, after: str | None = None) -> dict:
    """A RENDERED prompt checked: {ok, open, ends_with_lead, why}. `after`:
    only the text after its last occurrence is inspected (the question's
    last line), so a think marker quoted inside the state does not count."""
    p = prompt or ""
    tail = p[p.rfind(after):] if after and after in p else p
    open_ = any(tail.rfind(s) > tail.rfind(e) for s, e in THINK_MARKERS)
    ends = p.endswith(ANSWER_LEAD)
    why = ("a think block is open at the answer position" if open_ else
           f"the prompt does not end on {ANSWER_LEAD!r}" if not ends else "")
    return {"ok": not open_ and ends, "open": open_, "ends_with_lead": ends,
            "why": why}


def template_check(upstream=None) -> dict:
    """LIVE, inside a GPU window only (the /upstream door starts a model that
    is not loaded): the serving model's own rendering of jjava's canonical
    question -- /apply-template with the body jjava sends -- through
    prompt_guard. {model, checked, ok, open, ends_with_lead, why, tail}."""
    q = as_choice(yes_no("Is the material empty?"))
    b = body(NEUTRAL_STATE, q, None, 1, model_name())
    guard_body(b)
    payload = {"messages": b["messages"],
               "chat_template_kwargs": b["chat_template_kwargs"],
               "enable_thinking": False}
    out = {"model": model_name(), "checked": False}
    try:
        d = (upstream or _upstream)("/apply-template", payload)
    except Exception as e:                                       # noqa: BLE001
        out["why"] = f"/apply-template: {type(e).__name__}: {e}"[:200]
        return out
    prompt = (d or {}).get("prompt") if isinstance(d, dict) else None
    if not isinstance(prompt, str):
        out["why"] = "/apply-template returned no prompt"
        return out
    g = prompt_guard(prompt, after=question_text(q).splitlines()[-1])
    out.update(checked=True, **g, tail=prompt[-80:])
    return out


# ------------------------------------------------------------ the server ---
def _upstream(path: str, payload: dict | None = None,
              timeout: float = 30) -> object:
    import model
    import max_mode
    # MAX MODE (mcp/max_mode.py): the model jjava reads for the request -- its own, or the table's helper
    # (`bonsai-a4000` while a LOCKED model holds the card: max_mode.decider_model); never one that is off the card
    name = max_mode.decider_model(model.MODEL)
    max_mode.guard(name)
    # a worker job claimed for the A4000 (mcp/jobs.py, the gpu lane per card) never reads the main card
    max_mode.check_scope(name)
    url = f"{model.UPSTREAM}/upstream/{name}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={
        "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8") or "null")


_SPELL_IDS: dict[str, dict[str, int]] = {}


def spelling_ids(label: str, upstream=None) -> dict[str, int]:
    """{spelling: token id} for the label's single-token spellings -- per
    main model under max mode (its own /tokenize: the per-model label check)."""
    import max_mode
    upstream = upstream or _upstream
    key = f"{max_mode.decider_model()}:{label}" if max_mode.ENABLED else label
    if key in _SPELL_IDS:
        return _SPELL_IDS[key]
    spell = LABEL_SPELLINGS.get(label, (label, " " + label))
    out = {}
    for s in spell:
        try:
            toks = upstream("/tokenize", {"content": s,
                                          "add_special": False})["tokens"]
        except Exception as e:                                   # noqa: BLE001
            raise DeciderUnavailable(
                "TOKENIZE_FAILED", f"/tokenize of {s!r}: {type(e).__name__}:"
                f" {e}"[:200], True,
                "retry; if it persists the operator checks that llama-swap "
                "serves the main model (GET /running)") from e
        if isinstance(toks, list) and len(toks) == 1:
            t = toks[0]
            out[s] = int(t["id"] if isinstance(t, dict) else t)
    if not out:
        raise DeciderUnavailable(
            "LABEL_NOT_ONE_TOKEN", f"no spelling of label {label!r} is a "
            "single token", False,
            "the caller: use labels the tokenizer keeps whole (A..Z, yes/no)")
    _SPELL_IDS[key] = out
    return out


def read_labels(top: list[dict], ids: dict[str, dict[str, int]]) -> dict:
    """From one position's top_logprobs, each label's probability (its
    spellings summed), what is unread, and the bound on it."""
    import math
    by_id = {}
    for t in top or []:
        if "id" in t and "logprob" in t:
            by_id[int(t["id"])] = math.exp(float(t["logprob"]))
    p_min = min(by_id.values()) if by_id else 1.0
    raw, missing, primary_missing = {}, [], []
    for lab, spell in ids.items():
        s = 0.0
        for sp, tid in spell.items():
            if tid in by_id:
                s += by_id[tid]
            else:
                missing.append(sp)
        raw[lab] = s
        if not any(tid in by_id for tid in spell.values()):
            primary_missing.append(lab)
    return {"raw": raw, "missing": missing, "labels_unread": primary_missing,
            "p_min": p_min, "unread_bound": round(p_min * len(missing), 9),
            "top": max(by_id.items(), key=lambda kv: kv[1])[0]
            if by_id else None}


def case_variants(top: list[dict], ids: dict[str, dict[str, int]],
                  words=()) -> dict:
    """DIAGNOSTIC, never acted on (SGLang's decision models, "How answers
    are computed": their "label_mass counts only the lowercase yes and no
    tokens", and the model puts mass on the capitalised ones too, so a
    lower-case-only label mass reads low while the probabilities are
    unaffected). Our readout already SUMS every single-token spelling of a
    label (LABEL_SPELLINGS: yes / Yes / YES, with and without the space;
    a letter as "A" / " A"); this says where the rest of the top K went:
      by_spelling  each READ spelling's own mass (Yes vs yes, A vs " A")
      variant      per label: mass on top-K tokens that spell it in another
                   case or spacing but are not read ("a" beside "A")
      word         per option WORD (a lettered noul's "yes" / "no"): mass on
                   the word itself, every case and spacing -- the model
                   answering the text instead of the letter
    {available: False} when the server's top entries carry no token text."""
    import math
    if not any(isinstance(t, dict) and "token" in t for t in top or []):
        return {"available": False}
    read = {}
    for lab, spell in ids.items():
        for sp, tid in spell.items():
            read[tid] = sp
    labs = {str(lab).strip().casefold(): lab for lab in ids}
    wds = {str(w).strip().casefold(): w for w in words or ()}
    by_sp: dict = {}
    var = {lab: 0.0 for lab in ids}
    word = {w: 0.0 for w in words or ()}
    for t in top:
        if not isinstance(t, dict) or "id" not in t or "logprob" not in t:
            continue
        p = math.exp(float(t["logprob"]))
        tid = int(t["id"])
        if tid in read:
            by_sp[read[tid]] = by_sp.get(read[tid], 0.0) + p
            continue
        key = str(t.get("token", "")).strip().casefold()
        if key in labs:
            var[labs[key]] += p
        elif key in wds:
            word[wds[key]] += p
    r6 = lambda d: {k: round(v, 6) for k, v in d.items()}   # noqa: E731
    return {"available": True, "by_spelling": r6(by_sp),
            "variant": r6(var), "variant_mass": round(sum(var.values()), 6),
            "word": r6(word), "word_mass": round(sum(word.values()), 6)}


# ------------------------------------------------------------- decide ------
def _post_default(b: dict, timeout: float) -> dict:
    import model
    return model.post(b, timeout=int(timeout))


def ask_one(state: str, q: dict, *, slot: int | None, post=None,
            upstream=None, timeout: float = 120,
            n_vocab: int | None = None, cache: bool | None = None,
            msgs: list[dict] | None = None) -> dict:
    """One question, one forward pass (more only to read a label outside the
    top K). The slot is the caller's. `msgs`: a caller's own rendering (it
    must end on the ANSWER_LEAD prefill); q still names the labels.
    post / upstream: None -> this module's doors, looked up at call time
    (_post_default, _upstream), so a bench can point them at one engine."""
    import model
    post, upstream = post or _post_default, upstream or _upstream
    if cache is None:              # THE READ REGIME: the model's measured one
        cache = read_cache()
    ids = {lab: spelling_ids(lab, upstream) for lab in q["labels"]}
    band = tie_band()
    k, reads, t0 = FIRST_K, [], time.time()
    cap = n_vocab or 10 ** 6
    while True:
        t1 = time.time()
        import max_mode
        b = body(state, q, slot, k, max_mode.decider_model(model.MODEL), cache,
                 msgs)
        guard_body(b)                      # THE THINK-BLOCK GUARD (above)
        try:
            d = post(b, timeout)
        except DeciderUnavailable:
            raise
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", "replace")[:200]
            except Exception:                                    # noqa: BLE001
                pass
            raise DeciderUnavailable(
                "MODEL_HTTP_%d" % e.code, f"the model server answered "
                f"{e.code}: {detail}", e.code >= 500 or e.code == 429,
                "retry when the main model is idle (a 503 is a load in "
                "progress); a 400 is this module's body -- the maintainer "
                "reads `situation`") from e
        except Exception as e:                                   # noqa: BLE001
            raise DeciderUnavailable(
                "MODEL_UNREACHABLE", f"{type(e).__name__}: {e}"[:200], True,
                "retry; the operator checks llama-swap on :11434 "
                "(scripts/watchdog.ps1 restarts it on two failed "
                "health checks)") from e
        ch = (d.get("choices") or [{}])[0]
        content = ((ch.get("logprobs") or {}).get("content") or [])
        if not content:
            raise DeciderUnavailable(
                "NO_LOGPROBS", "the response carried no logprobs for the "
                "answer position", False,
                "the maintainer: this server build must honour logprobs/"
                "top_logprobs on /v1/chat/completions")
        tm = d.get("timings") or {}
        r = read_labels(content[0].get("top_logprobs") or [], ids)
        reads.append({"k": k, "ms": round((time.time() - t1) * 1000, 1),
                      "prompt_n": tm.get("prompt_n"),
                      "cache_n": tm.get("cache_n"),
                      "prompt_ms": tm.get("prompt_ms")})
        # Enough is read when every label has a spelling in the top K, or
        # K has reached the vocabulary (the server clamps n_probs to it, so
        # the 10**6 cap is the whole vocabulary). Unread SPELLINGS of a
        # label already read ("YES" beside " yes") are bounded
        # (`unread_bound`), not chased: where that bound could flip a
        # decision the two labels are inside the tie band anyway. (Chasing
        # them read a flat content-free distribution six times, K up to
        # the cap -- found by mcp/test_decide_turn.py, 2026-09-27.)
        # A LABEL still unread is below the smallest probability returned;
        # all of them together are below p_min x their spellings. Stop when
        # that cannot overturn the argmax (it is below the top two's gap),
        # or the top two are a tie anyway (the model's tie_band()). Without this a
        # 39-option question chased K to the vocabulary for its rarely
        # chosen labels, seconds per read (pagoda-h4 replay, 2026-09-27).
        vals = sorted(r["raw"].values(), reverse=True)
        gap = vals[0] - vals[1] if len(vals) > 1 else 1.0
        lab_bound = r["p_min"] * sum(len(ids[lab])
                                     for lab in r["labels_unread"])
        if not r["labels_unread"] or lab_bound < gap or gap <= band \
                or k >= cap:
            break
        k = min(k * GROWTH, cap)
    total = sum(r["raw"].values())
    probs = ({lab: v / total for lab, v in r["raw"].items()} if total > 0
             else {lab: 1.0 / len(r["raw"]) for lab in r["raw"]})
    ans = max(probs.items(), key=lambda kv: kv[1])[0]
    first = reads[0]
    usage = d.get("usage") or {}
    return {"question": q["question"], "kind": q["kind"],
            "labels": q["labels"], "answer": ans,
            "probs": {k_: round(v, 6) for k_, v in probs.items()},
            "label_mass": round(total, 6),
            "exact": not r["missing"], "unread": r["missing"],
            "labels_unread": r["labels_unread"],
            "unread_bound": r["unread_bound"], "k": k, "reads": len(reads),
            "case_variants": case_variants(
                content[0].get("top_logprobs") or [], ids,
                q.get("words") or ()),
            # every HTTP read of this question (a K re-read included): what
            # the Jev API's usage counts (mcp/jev_api.py USAGE)
            "read_log": [{"k": x["k"], "prompt_n": x.get("prompt_n"),
                          "cache_n": x.get("cache_n")} for x in reads],
            "prompt_tokens": usage.get("prompt_tokens")
            or ((first.get("prompt_n") or 0) + (first.get("cache_n") or 0)),
            "processed_tokens": first.get("prompt_n"),
            "cached_tokens": first.get("cache_n"),
            "prefill_ms": first.get("prompt_ms"),
            "total_ms": round((time.time() - t0) * 1000, 1)}


def slot_cells(slot: int, upstream=None) -> int | None:
    """What llama-server says the slot holds (n_prompt_tokens), or None.
    Through the one door only while llama-swap's /running lists the model
    ready (a read never loads a model, 2026-09-30)."""
    try:
        if upstream is None:
            import gpu_room
            import max_mode
            import model
            if not gpu_room.model_loaded(model.UPSTREAM,
                                         max_mode.decider_model(model.MODEL))[0]:
                return None
        table = (upstream or _upstream)("/slots")
        row = next((s for s in table if isinstance(s, dict)
                    and s.get("id") == slot), None)
        return int(row.get("n_prompt_tokens") or 0) if row else None
    except Exception:                                            # noqa: BLE001
        return None


def release(slot: int | None, why: str = "decider batch") -> dict:
    """Empty the decider's slot now: the proxy's release rule where this
    process releases (slots.release_idle), else the one door's release
    (model.release_slot). Never raises."""
    if slot is None:
        return {"released": False, "skipped": "no slot was granted"}
    try:
        import slots
        import max_mode
        # the decider's own server (max_mode.decider_model: `bonsai-a4000` while a LOCKED model holds the card),
        # never the conversation's
        dm = max_mode.decider_model()
        if slots.release_enabled():
            rec = slots.release_idle(slot, why, model=dm)
            if rec is not None:
                return dict(rec, via="slots.release_idle")
        import model
        res = model.release_slot(slot, model=dm)
        return {"slot": slot, "why": why, "via": "model.release_slot",
                "released": bool(res.get("ok")) and not res.get("skipped"),
                **{k: res.get(k) for k in ("cells_before", "ms", "method",
                                           "skipped", "error",
                                           "erase_refused") if res.get(k)
                   is not None}}
    except Exception as e:                                       # noqa: BLE001
        return {"slot": slot, "released": False,
                "error": f"{type(e).__name__}: {e}"[:200]}


def decide(state: str, questions: list[dict], *, keep_slot: bool = False,
           post=None, upstream=None, timeout: float = 120,
           slot: int | None = None, cache: bool | None = None) -> dict:
    """Every question over one state, in order, on one transient slot, then
    the slot released. Returns {template, slot, how, answers[], release,
    batch_ms}. `slot` given: the caller placed it (benchmarks) and it is not
    acquired here; it is still released unless keep_slot. cache=False
    (measurement only) sends cache_prompt false: every question re-reads
    the state."""
    post, upstream = post or _post_default, upstream or _upstream
    if not questions:
        raise DeciderUnavailable("NO_QUESTIONS", "nothing to decide", False,
                                 "the caller: pass at least one question")
    if all(isinstance(q, dict) and "type" in q for q in questions):
        # THE TYPED API (below): {type: choice | score | noul, name, text,
        # ...} -> Jev's response. The old {kind: yes_no | choice} questions
        # keep this path until the legacy readout is removed (REMOVAL LIST:
        # after bench/decider/legacy_vs_typed.py's one GPU run).
        return _decide_typed(state, questions, keep_slot=keep_slot,
                             post=post, upstream=upstream, timeout=timeout,
                             slot=slot, cache=cache)
    for q in questions:
        if not q.get("labels"):
            raise DeciderUnavailable(
                "NO_LABELS", f"question {q.get('question')!r} has no labels",
                False, "the caller: build it with yes_no() or choice()")
    import slots
    t0 = time.time()
    grant = None
    how = "given by the caller"
    if slot is None:
        grant = slots.acquire(None, transient=True)
        slot, how = grant.get("slot"), grant.get("how")
    answers = []
    try:
        for q in questions:
            answers.append(ask_one(state, q, slot=slot, post=post,
                                   upstream=upstream, timeout=timeout,
                                   cache=cache))
    finally:
        if grant is not None:
            slots.release(grant)
        rel = ({"released": False, "skipped": "keep_slot"} if keep_slot
               else slots.lane_kept_note(slot, "decider batch", log=[])
               if grant is not None and slots.lane_kept()
               else release(slot))
    return {"template": TEMPLATE_VERSION, "slot": slot, "how": how,
            "answers": answers, "release": rel,
            "batch_ms": round((time.time() - t0) * 1000, 1)}


def fit_tail(pieces: list[str], budget: int, count) -> tuple[str, dict]:
    """The newest pieces whose token counts (count(piece)) fit `budget`,
    oldest first, joined by blank lines; the boundary piece is cut from its
    front (by characters, in proportion) when not even the newest fits.
    Linear: each piece is counted once."""
    kept, used = [], 0
    for p in reversed(pieces):
        n = count(p) + 2
        if used + n > budget:
            if not kept:
                frac = budget / max(n, 1)
                kept.append(p[-max(int(len(p) * frac), 1):])
                used = budget
            break
        kept.append(p)
        used += n
    kept.reverse()
    return "\n\n".join(kept), {"pieces": len(pieces), "kept": len(kept),
                               "tokens": used,
                               "dropped": len(pieces) - len(kept)}




# ============================================================ THE TYPED API ==
# Operator, 2026-09-29: "look to the Jev api for the answer, they got it
# right." Jev (TypeSafe; docs/research/DECIDER-RESEARCH.md 1.2; docs/JJAVA.md,
# "jjava" being our Jev) types every question, and the answer has Jev's shape
# EXACTLY (docs.typesafe.ai /primitives/choice, /primitives/score,
# /primitives/noul, /confidence, read 2026-09-29):
#
#   choice  {type, choice, probabilities, confidence}
#             choice: "The option with the highest probability."
#             probabilities: per option key, "The sum of all values is 1."
#   score   {type, score, probabilities, confidence, legend}
#             levels numbered 0..k in the order given, lowest first;
#             score: "each level number multiplied by its probability,
#             added up"; probabilities keyed by level number as a string;
#             legend: "Each level number mapped back to its description."
#   noul    {type, noul}
#             "a single number representing the probability that the answer
#             is yes"; "There is no separate confidence value for a Noul."
#
#   q_noul(name, text, criteria=)          criteria {true, false}: what a yes
#                                          and a no mean (Jev's optional
#                                          noul criteria)
#   q_choice(name, text, options, keys=, none=)
#   q_score(name, text, levels)            levels lowest first
#   decide(state, [q, ...])                -> {model, answers: {name: answer},
#                                          usage, ...}: one shared state
#                                          prefix on one transient slot
#   read(state, q, slot=...)               one question on a slot the caller
#                                          holds (decide_turn.Turn.decide)
#
# CONFIDENCE (confidence()): Jev's docs say it is "computed from how
# `probabilities` is spread. All of it on one option gives 1.0; the more
# evenly it spreads, the lower the confidence" and give "(3 x largest
# probability - 1) / 2" for three options (/confidence). The page's own demo
# code computes, for `count` options, clamp((count x peak - 1) / (count - 1),
# 0, 1) -- the n-option form -- and the prose calls it an approximation
# ("to approximate confidence for three options"). CHECKED against every
# API-response example in the docs (llms-full.txt, 2026-09-29): 17 distinct
# Choice and Score answers, 2 to 5 options, all agree within the rounding of
# the 2 decimals the docs print (e.g. choice [0.34, 0.40, 0.02, 0.24] ->
# 0.20; score [0, 0.57, 0.43] -> 0.355, shown 0.35; choice [0.88, 0.12, 0]
# -> 0.82, shown 0.81: p_max itself is rounded; mcp/test_decider_bonsai.py
# [confidence]). Two illustrative widget examples on /primitives/score do
# NOT fit it (5 levels [0, .14, .86, 0, 0] shown 0.89 where the form gives
# 0.825; 4 levels [0, 0, .48, .52] shown 0.52 where it gives 0.36), and the
# Score page gives no formula of its own. So: the form is DERIVED from the
# docs' demo code and their response examples, and it is what this module
# implements for Choice and Score alike; if TypeSafe publishes a different
# Score formula, confidence() is the one place to change.
#
# NO THRESHOLD HERE. The decider returns belief (the probabilities, the noul)
# and certainty (the confidence) only; the QUESTION SET's code owns its
# thresholds (decide_turn THRESHOLDS: high -> act, medium -> proceed with
# caution, low -> fall back), tuned per question set per model on labelled
# decisions (bench/decider/tune.py proposes, the owner accepts). The
# built-in abstain and its per-model params file (index/decider/params.json,
# params_for) were REMOVED 2026-09-29 (operator: callers own thresholds).
#
# N is len(LETTERS) = 26, not Jev's 255: an option's label must be ONE token
# the model answers, and the served Bonsai does not answer the double letters
# (skill_match LABELS, measured 2026-09-27). A Score takes at most 26 levels
# (Jev: "the API accepts up to 10"). Measured on Bonsai only; the single
# letters are the bound for every model (a model's own one-token check is
# its profile's `labels`: bench/decider/measure_model.py).
#
# THE READOUT (READOUT_VERSION; INTERNAL -- none of it is in the answer's
# Jev fields; each rule cites DECIDER-RESEARCH.md):
#   - LETTERS for every option, a noul included: a LETTERED PAIR ("A. yes" /
#     "B. no"), never the bare yes/no tokens. reflex's controls on toxic-chat
#     (frozen Qwen3.5-4B, self-reported [R]): Yes/No tokens 0.787 / ECE
#     0.104, the lettered pair 0.797 / 0.083 (2.2). The pair's printed texts
#     are "yes" / "no", byte for byte the rendering decide_turn's build
#     intent was measured in (`mc_avg`, intent 0.869, bench/decider/results/
#     bonsai.json `calib`, in-sample). A noul WITH criteria prints
#     "yes: <true>" / "no: <false>" (Jev's noul criteria; that rendering is
#     UNMEASURED on Bonsai).
#   - TWO ORDERS, averaged: the given order and its reverse; a choice's
#     "none" option sits in the middle of both where the option count
#     allows (k >= 3). Letters are POSITIONAL. Evidence (4.2 item 3; 2.1):
#     reflex, frozen Qwen3.8-27B, 1 vs 2 orders: hard 0.703 -> 0.766, hard
#     ECE 0.088 -> 0.061 (dev suite, self-reported); ours: build intent
#     0.848 -> 0.869 (in-sample). No more than two (4.3). The AVERAGED
#     distribution is the answer's `probabilities`, and the confidence is
#     computed on it.
#   - NO LABEL-PRIOR DIVISION by default (2.1, 4.1, 4.3: it raised reflex's
#     hard ECE 0.086 -> 0.124 and cost our build intent 0.869 -> 0.818). The
#     ONE exception is a question built with prior="content_free" --
#     decide_turn's weak package detections, the one place it was measured
#     to help (as a VETO: mc_avg_cc kept 33/33 true detections, mc_avg lost
#     17; 4.1).
#   - DIAGNOSTICS, logged, never acted on here (`diagnostics`): each order's
#     distribution, `disagreement` (the total-variation distance between the
#     two orders' distributions: reflex's error signal, 4.2 item 4), whether
#     the orders' argmaxes agree, `label_mass_min` (the smallest share of the
#     whole next-token distribution on the labels, the "My Answer is C"
#     failure, 2.1), the margin, and `tie`: the top two within the measured
#     TIE_BAND (batch nondeterminism; a caller may treat it as no decision).
READOUT_VERSION = "typed/2"
TYPES = ("choice", "score", "noul")
NOUL_KEYS = ("true", "false")
NOUL_TEXTS = ("yes", "no")
PRIORS = (None, "content_free")


def model_name() -> str:
    """The model this process's decider reads (max mode's current, else the
    main model), or "" when that cannot be said."""
    try:
        import model
        import max_mode
        return str(max_mode.decider_model(model.MODEL) or "")
    except Exception:                                            # noqa: BLE001
        return ""


# ====================================================== THE MODEL PROFILE ==
# (2026-09-29.) jjava reads WHICHEVER model serves the conversation
# (model_name(): max_mode.current() -- `bonsai` by default, `flash-next` at
# tier max, `mirai-s` at xhigh when it lands). A label's probability is a
# property of the model reading it (DECIDER-RESEARCH 2.9), so everything that
# was MEASURED on one model is keyed by the serving model's name here, never
# assumed to carry over (docs/JJAVA.md "Per model", the audit):
#
#   tie_band       USED: the batch-nondeterminism floor (decision(), read(),
#                  ask_one's re-read stop). bench/decider/bonsai_decider.py
#                  `batching`.
#   labels         REPORTED: which spellings of A..Z and yes/no are one token
#                  (the runtime resolves its own per model: spelling_ids).
#   letter_prior   REPORTED: the content-free distribution over the option
#                  letters per option count (the letter bias the typed
#                  readout's two orders are there to cancel).
#   label_bias     REPORTED: each question set's content-free answer.
#   readout        REPORTED: the evidence for typed/2's choices (two orders
#                  averaged, no label-prior division, the package veto's
#                  prior) -- bonsai_decider.py `calib`.
#   legacy_form    REPORTED: the evidence for decide_turn.FORM (legacy).
#   legacy_vs_typed REPORTED: bench/decider/legacy_vs_typed.py's summary.
#   temperature    USED: the per-question-set temperature over the label
#                  probabilities (THE TEMPERATURE, below), fitted by
#                  bench/decider/fit_temperature.py held out by run. None
#                  fitted: T = 1 (the readout as it was); never borrowed
#                  from another model (FALLBACK_MODEL's is not applied).
#   read_regime    USED: whether a question is read with the prefix cached
#                  or fully cold (cache_prompt false) -- the cheapest regime
#                  that REPEATS EXACTLY on this model (bench/decider/
#                  determinism.py; operator, 2026-09-30: "20s is not too
#                  slow ... we would normally be paying 10s of minutes or
#                  more without jjava"). None measured: cached, as before,
#                  recorded as unmeasured; never borrowed.
#
# WHERE A PROFILE COMES FROM: BUILTIN_PROFILES (Bonsai's 2026-09-27 numbers,
# each with its script and n, exactly the values this module used before),
# then the per-model record bench/decider/measure_model.py writes --
# PROFILES_DIR/<model>.json (PROFILE_VERSION; each field with its script,
# date and n). A record's field replaces the built-in one (a newer
# measurement). A model with NO measured value for a USED field gets
# FALLBACK_MODEL's -- the model every built-in value was measured on, i.e.
# exactly what the decider did for every model before this keying -- and the
# answer says so: tie_band {measured: False, source: "unmeasured for <model>:
# ..."}; profile_status() lists every unmeasured field. No value is invented
# for any other model.
PROFILE_VERSION = "decider-model/1"
FALLBACK_MODEL = "bonsai"
# AGENTS.md: "`bonsai` and `bonsai-agent` are the same process: the alias
# exists only for old clients".
MODEL_ALIASES = {"bonsai-agent": "bonsai"}
PROFILES_DIR = os.environ.get("YAMADORI_DECIDER_MODELS_DIR") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench",
    "decider", "results", "models")
PROFILE_FIELDS = ("tie_band", "labels", "letter_prior", "label_bias",
                  "readout", "legacy_form", "legacy_vs_typed", "temperature",
                  "read_regime")
BUILTIN_PROFILES: dict = {
    "bonsai": {
        "tie_band": {
            "value": TIE_BAND, "n": 100, "runs": 1, "date": "2026-09-27",
            "script": "bench/decider/bonsai_decider.py --only batching",
            "how": "max |p_label| difference between a prefix-cached and a "
                   "cache_prompt-false read of the same question, 20 "
                   "states x 5 questions (max_prob_diff 0.003436)"},
        "labels": {
            "date": "2026-09-27",
            "script": "mcp/skill_match.py --check-labels; the double-letter "
                      "replay in skill_match LABELS",
            "single_letters_one_token": True,
            "double_letters_answered": False},
        "readout": {
            "date": "2026-09-27", "in_sample": True,
            "script": "bench/decider/bonsai_decider.py --only calib",
            "what": "two orders averaged (intent 0.848 -> 0.869), no "
                    "label-prior division (intent 0.869 -> 0.818 with it), "
                    "the content-free prior kept only as the weak-package "
                    "veto (33/33 true detections kept)"},
        "legacy_form": {
            "date": "2026-09-27", "in_sample": True,
            "script": "bench/decider/bonsai_decider.py --only calib",
            "what": "decide_turn.FORM"},
    },
}
_PROFILE_CACHE: dict = {}


def canonical_model(name: str | None = None) -> str:
    """The profile's key for a model name (an alias resolved); the serving
    model when `name` is None."""
    name = model_name() if name is None else str(name or "")
    return MODEL_ALIASES.get(name, name)


def profile_path(name: str | None = None) -> str:
    return os.path.join(PROFILES_DIR, f"{canonical_model(name)}.json")


def _valid_field(field: str, v) -> str | None:
    """Why a record's field is not usable, or None."""
    if not isinstance(v, dict):
        return "not an object"
    if not v.get("script") or not v.get("date"):
        return "no script or date (AGENTS.md: claims carry their evidence)"
    if field == "tie_band":
        x = v.get("value")
        if not isinstance(x, (int, float)) or isinstance(x, bool) \
                or not 0.0 <= float(x) < 0.5:
            return f"value {x!r} is not a probability difference in [0, 0.5)"
        if not isinstance(v.get("n"), int) or v["n"] < 1:
            return "no n"
    if field == "temperature":
        sets = v.get("sets")
        if not isinstance(sets, dict) or not sets:
            return "no `sets` {question set: {value, n, heldout_ok}}"
        for name, row in sets.items():
            x = row.get("value") if isinstance(row, dict) else None
            if not isinstance(x, (int, float)) or isinstance(x, bool) \
                    or not 0.0 < float(x) < float("inf"):
                return f"set {name!r}: value {x!r} is not a positive number"
            if not isinstance(row.get("n"), int) or row["n"] < 1:
                return f"set {name!r}: no n"
            if row.get("heldout_ok") is not True:
                return (f"set {name!r}: not shown to hold out by run "
                        "(heldout_ok)")
    if field == "read_regime":
        if v.get("mode") not in READ_MODES:
            return f"mode {v.get('mode')!r} (one of {READ_MODES})"
        if v.get("repeats_exactly") is not True:
            return ("not shown to repeat exactly (repeats_exactly): only "
                    "such a regime is chosen")
        if not isinstance(v.get("n"), int) or v["n"] < 1:
            return "no n"
    return None


# THE READ REGIME (profile field `read_regime`, above).
READ_MODES = ("cold", "cached")


def read_regime_of(name: str | None = None) -> dict:
    """{model, mode, measured, source}: the model's measured regime, else
    `cached` (the path before 2026-09-30), recorded as unmeasured."""
    p = profile(name)
    rr = p["fields"].get("read_regime")
    if rr is not None:
        return {"model": p["model"], "mode": rr["mode"], "measured": True,
                "source": rr.get("source"), "n": rr.get("n")}
    return {"model": p["model"], "mode": "cached", "measured": False,
            "source": f"unmeasured for {p['model'] or '(model unknown)'}: "
                      "cached, the path before 2026-09-30 (bench/decider/"
                      "determinism.py measures it)"}


def read_cache(name: str | None = None) -> bool:
    """cache_prompt for a jjava read on the serving model: False when its
    measured regime is `cold`."""
    return read_regime_of(name)["mode"] != "cold"


def load_record(name: str | None = None) -> dict:
    """The measured record of model `name` (PROFILES_DIR/<model>.json):
    {path, fields, rejected: {field: why}, error?}. Cached by mtime; never
    raises."""
    key = canonical_model(name)
    path = profile_path(key)
    try:
        st = os.stat(path)
    except OSError:
        return {"path": None, "fields": {}, "rejected": {}}
    hit = _PROFILE_CACHE.get(path)
    if hit and hit[0] == (st.st_mtime_ns, st.st_size):
        return hit[1]
    out: dict = {"path": path, "fields": {}, "rejected": {}}
    try:
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)
        if not isinstance(rec, dict) or rec.get("version") != PROFILE_VERSION:
            out["error"] = (f"not a {PROFILE_VERSION} record (version "
                            f"{rec.get('version') if isinstance(rec, dict) else None!r})")
        elif canonical_model(rec.get("model")) != key:
            out["error"] = (f"the record names model {rec.get('model')!r}, "
                            f"not {key!r}")
        else:
            for field in PROFILE_FIELDS:
                if field not in rec:
                    continue
                why = _valid_field(field, rec[field])
                if why:
                    out["rejected"][field] = why
                else:
                    out["fields"][field] = rec[field]
    except (OSError, ValueError) as e:
        out["error"] = f"{type(e).__name__}: {e}"[:200]
    _PROFILE_CACHE[path] = ((st.st_mtime_ns, st.st_size), out)
    return out


def profile(name: str | None = None) -> dict:
    """The model's profile: {model, record, fields: {field: value + source},
    unmeasured: [field], rejected, error?}. Built-in values first, the
    measured record's over them."""
    key = canonical_model(name)
    rec = load_record(key)
    fields: dict = {}
    for field, v in (BUILTIN_PROFILES.get(key) or {}).items():
        fields[field] = dict(v, source="built-in")
    for field, v in rec["fields"].items():
        fields[field] = dict(v, source=rec["path"])
    out = {"model": key, "record": rec["path"], "fields": fields,
           "unmeasured": [f for f in PROFILE_FIELDS if f not in fields],
           "rejected": dict(rec["rejected"])}
    if rec.get("error"):
        out["error"] = rec["error"]
    return out


def tie_band_of(name: str | None = None) -> dict:
    """{model, value, measured, source}: the model's measured tie band, else
    FALLBACK_MODEL's, recorded as unmeasured for this model."""
    p = profile(name)
    tb = p["fields"].get("tie_band")
    if tb is not None:
        return {"model": p["model"], "value": float(tb["value"]),
                "measured": True, "source": tb["source"],
                "n": tb.get("n"), "script": tb.get("script")}
    fb = profile(FALLBACK_MODEL)["fields"]["tie_band"]
    return {"model": p["model"], "value": float(fb["value"]),
            "measured": False, "n": fb.get("n"), "script": fb.get("script"),
            "source": (f"unmeasured for {p['model'] or '(model unknown)'}: "
                       f"{FALLBACK_MODEL}'s measured value applied, as "
                       "before per-model keying (bench/decider/"
                       "measure_model.py --model "
                       f"{p['model'] or '<name>'} measures it)")}


def tie_band(name: str | None = None) -> float:
    """The serving model's tie band (tie_band_of's value)."""
    return tie_band_of(name)["value"]


def profile_status(name: str | None = None) -> dict:
    """What a result reports about the model it read: {model, record,
    measured: [field], unmeasured: [field], tie_band: {value, measured,
    source}, rejected?, error?, note?}."""
    p = profile(name)
    out = {"model": p["model"], "record": p["record"],
           "measured": sorted(p["fields"]), "unmeasured": p["unmeasured"],
           "tie_band": {k: v for k, v in tie_band_of(p["model"]).items()
                        if k in ("value", "measured", "source")}}
    if p["rejected"]:
        out["rejected"] = p["rejected"]
    if p.get("error"):
        out["error"] = p["error"]
    if p["unmeasured"]:
        out["note"] = (f"unmeasured for {p['model'] or '(model unknown)'}: "
                       + ", ".join(p["unmeasured"]))
    return out


# ======================================================= THE TEMPERATURE ==
# SGLang's decision models ("How answers are computed", read 2026-09-30):
# a yes/no probability is exp(lp_yes / T) / (exp(lp_yes / T) + exp(lp_no /
# T)) -- a temperature over the LABEL logits, the one calibration knob;
# the vocabulary's normaliser cancels (every label shares it), and
# label_mass = exp(lp_yes) + exp(lp_no) does not depend on T. Their values
# are "not a calibrated probability that the decision is correct. Validate
# any threshold on labeled data".
#
# HERE: T is per MODEL and per QUESTION SET (a set's full name, else its
# family: the THRESHOLDS keys), from the model's profile field
# `temperature` -- written only by bench/decider/fit_temperature.py, fitted
# on labelled decisions and shown to hold out by run (`heldout_ok`); never
# borrowed from another model. None fitted: T = 1, the readout exactly as
# it was. APPLIED in read() to each order's label distribution as read
# (p_i^(1/T), renormalised: the same thing as T over the label logits),
# BEFORE the content-free prior and the exclusion, then the orders are
# averaged as always. A single read's argmax never moves; the two orders'
# average can, and every confidence and noul does -- so THRESHOLDS are
# tuned AFTER a temperature is fitted, on tempered values (readout_of()
# recomputes them from the per-order `raw` the decisions carry).
# YAMADORI_JJAVA_TEMPERATURE=0 reads with T = 1 whatever is fitted.
TEMPERATURE_ON = os.environ.get("YAMADORI_JJAVA_TEMPERATURE", "1") != "0"


def temperature_of(question: str, name: str | None = None) -> dict:
    """{value, fitted, set, source}: the question set's temperature on the
    model `name` (default: the serving model)."""
    key = canonical_model(name)
    q = str(question or "")
    fam = q.split(":", 1)[0]
    if not TEMPERATURE_ON:
        return {"value": 1.0, "fitted": False, "set": None,
                "source": "off (YAMADORI_JJAVA_TEMPERATURE=0)"}
    field = profile(key)["fields"].get("temperature") or {}
    sets = field.get("sets") or {}
    for s in (q, fam):
        if s in sets:
            return {"value": float(sets[s]["value"]), "fitted": True,
                    "set": s, "source": field.get("source"),
                    "n": sets[s].get("n")}
    return {"value": 1.0, "fitted": False, "set": None,
            "source": f"none fitted for {fam!r} on {key or '(model unknown)'}"
                      ": T = 1"}


def temper(p: dict, t: float) -> dict:
    """A label distribution at temperature t: p_i^(1/t), renormalised (t = 1
    returns it unchanged; a zero stays zero)."""
    t = float(t)
    if t == 1.0:
        return dict(p)
    import math
    lg = {k: (math.log(v) / t if v > 0 else None) for k, v in p.items()}
    m = max((x for x in lg.values() if x is not None), default=None)
    if m is None:
        return dict(p)
    e = {k: (math.exp(x - m) if x is not None else 0.0)
         for k, x in lg.items()}
    z = sum(e.values()) or 1.0
    return {k: v / z for k, v in e.items()}


def readout_of(orders_raw: list[dict], t: float = 1.0,
               allowed=None) -> dict:
    """read()'s averaged distribution rebuilt from its per-order `raw`
    distributions (diagnostics.orders[].raw, the decisions log's
    orders[].raw) at temperature t, the excluded keys dropped: how a bench
    re-scores logged decisions under a fitted T. (The content-free prior is
    not re-applied: only the package veto's questions carry one.)"""
    keys = list(orders_raw[0])
    allowed = [k for k in keys if allowed is None or k in allowed]
    per = []
    for raw in orders_raw:
        p = temper(raw, t)
        z = sum(p.get(k, 0.0) for k in allowed)
        per.append({k: ((p.get(k, 0.0) / z if z > 0 else 1.0 / len(allowed))
                        if k in allowed else 0.0) for k in keys})
    return {k: sum(o[k] for o in per) / len(per) for k in keys}


# ====================================================== THE SCORE LABELS ==
# SGLang labels a score's LEVELS 0-9 (single tokens) where options take
# A-Z. Ours label every option, levels included, with positional letters
# (typed/2). The digit labels are here BEHIND A SWITCH, default unchanged
# ("letters") until the window measures them (bench/skills/inject_decide.py
# variant `e`): YAMADORI_JJAVA_SCORE_LABELS=digits, or per question
# q_score(..., label_kind="digits"). With digits a level is printed with its
# OWN number in both orders ("2. <level 2>"), so the label carries the
# level's meaning; the two orders still move each level's position. The
# instruction line SCORE_DIGITS is our wording, UNMEASURED. More than 10
# levels cannot take single digits: those stay lettered, and the answer
# says so (diagnostics.score_labels).
LABEL_KINDS = ("letters", "digits")
SCORE_LABELS = os.environ.get("YAMADORI_JJAVA_SCORE_LABELS", "letters")
DIGITS = tuple("0123456789")
SCORE_DIGITS = ("QUESTION: {question}\n\nLEVELS:\n{options}\n\n"
                "Answer with the number of one level.")


def score_label_kind(q: dict) -> str:
    """The label kind a typed score is printed with: "digits" | "letters"
    (a choice or noul: always letters)."""
    if q.get("type") != "score":
        return "letters"
    want = q.get("label_kind") or SCORE_LABELS
    if want == "digits" and len(q["keys"]) <= len(DIGITS):
        return "digits"
    return "letters"


def confidence(probs: dict) -> float:
    """Jev's confidence of a Choice or Score distribution: clamp((n x p_max -
    1) / (n - 1), 0, 1) over the n options it is spread over (see "THE TYPED
    API" for the source and what was checked). 1.0 with one option."""
    vals = [float(v) for v in (probs or {}).values()]
    n = len(vals)
    if n < 2:
        return 1.0
    return max(0.0, min(1.0, (n * max(vals) - 1.0) / (n - 1)))


def _refuse(code: str, situation: str) -> DeciderUnavailable:
    return DeciderUnavailable(code, situation, False,
                              "the caller: build the question with q_noul, "
                              "q_choice or q_score")


def _keys(keys, n: int, default) -> list[str]:
    out = [str(k) for k in (keys if keys is not None else default)]
    if len(out) != n or len(set(out)) != n:
        raise _refuse("BAD_KEYS", f"{n} options need {n} distinct keys; got "
                      f"{len(out)} ({len(set(out))} distinct)")
    return out


def _options(options, what: str) -> list[str]:
    opts = [str(o).strip() for o in options or []]
    if len(opts) < 2:
        raise DeciderUnavailable(
            "NO_OPTIONS", f"a {what} needs at least two options; got "
            f"{len(opts)}", False, "the caller: pass two or more")
    if len(opts) > len(LETTERS):
        raise DeciderUnavailable(
            "TOO_MANY_OPTIONS",
            f"{len(opts)} options; single-token labels run A..Z "
            f"({len(LETTERS)})", False,
            "the caller: split the choice into rounds (verify_moment CHUNK), "
            "or filter in code and ask about a shortlist (docs/JJAVA.md)")
    return opts


def q_noul(name: str, text: str, *, criteria: dict | None = None,
           prior: str | None = None) -> dict:
    """A yes/no question (or a statement to judge): answered as the
    probability that it is true. `criteria` {true, false}: what a yes and a
    no mean (Jev's noul criteria)."""
    if prior not in PRIORS:
        raise _refuse("BAD_PRIOR", f"prior {prior!r} (one of {PRIORS})")
    opts = list(NOUL_TEXTS)
    if criteria:
        if not isinstance(criteria, dict) or set(criteria) - {"true",
                                                              "false"}:
            raise _refuse("BAD_CRITERIA", "a noul's criteria are {true, "
                          "false} descriptions")
        opts = [f"{t}: {str(criteria[k]).strip()}" if criteria.get(k)
                else t for t, k in zip(NOUL_TEXTS, NOUL_KEYS)]
    q = {"type": "noul", "name": str(name), "text": str(text).strip(),
         "options": opts, "keys": list(NOUL_KEYS), "none": None}
    if criteria:
        q["criteria"] = {k: str(v) for k, v in criteria.items()}
    if prior:
        q["prior"] = prior
    return q


def q_choice(name: str, text: str, options, *, keys=None,
             none: int | None = None, prior: str | None = None) -> dict:
    """One of `options` (short descriptions). `keys` name them in the answer
    (default A, B, C ... in the order given); `none` is the index of the
    "none of these" option, if there is one."""
    opts = _options(options, "choice")
    if none is not None and not 0 <= int(none) < len(opts):
        raise _refuse("BAD_NONE", f"none={none} outside 0..{len(opts) - 1}")
    if prior not in PRIORS:
        raise _refuse("BAD_PRIOR", f"prior {prior!r} (one of {PRIORS})")
    q = {"type": "choice", "name": str(name), "text": str(text).strip(),
         "options": opts, "keys": _keys(keys, len(opts), LETTERS[:len(opts)]),
         "none": None if none is None else int(none)}
    if prior:
        q["prior"] = prior
    return q


def q_score(name: str, text: str, levels, *,
            label_kind: str | None = None) -> dict:
    """A level of a rubric: `levels` are the levels' descriptions, lowest
    first, numbered 0..k as Jev numbers them (the answer's `probabilities`
    and `legend` are keyed "0", "1", ...). `label_kind` "digits" prints each
    level with its own number (THE SCORE LABELS); None: the switch."""
    opts = _options(levels, "score")
    if label_kind is not None and label_kind not in LABEL_KINDS:
        raise _refuse("BAD_LABEL_KIND", f"label_kind {label_kind!r} (one of "
                      f"{LABEL_KINDS})")
    q = {"type": "score", "name": str(name), "text": str(text).strip(),
         "options": opts, "none": None,
         "keys": [str(i) for i in range(len(opts))]}
    if label_kind:
        q["label_kind"] = label_kind
    return q


def typed(q: dict) -> dict:
    """A plain question dict, validated and completed by its constructor (a
    question built by one passes as is). Jev's request field names are read
    too: `instructions` for the text, `criteria` for a choice's {key:
    description} map, a score's level list, or a noul's {true, false}."""
    if not isinstance(q, dict) or q.get("type") not in TYPES:
        got = q.get("type") if isinstance(q, dict) else q
        raise _refuse("BAD_TYPE", f"type {got!r} (one of {TYPES})")
    t, name = q["type"], q.get("name") or ""
    text = q.get("text") or q.get("instructions") or ""
    crit = q.get("criteria")
    if t == "noul":
        return q_noul(name, text, criteria=crit if isinstance(crit, dict)
                      else None, prior=q.get("prior"))
    if t == "choice":
        if isinstance(crit, dict) and not q.get("options"):
            return q_choice(name, text, list(crit.values()),
                            keys=list(crit.keys()), none=q.get("none"),
                            prior=q.get("prior"))
        return q_choice(name, text, q.get("options") or crit,
                        keys=q.get("keys"), none=q.get("none"),
                        prior=q.get("prior"))
    return q_score(name, text, q.get("levels") or q.get("options") or crit,
                   label_kind=q.get("label_kind"))


def two_orders(k: int, none: int | None = None) -> list[list[int]]:
    """The two option orders: `base` and its reverse. `base` is the given
    order with the "none" option (if any) moved to the middle, so neither
    order prints it last (k >= 3)."""
    base = list(range(k))
    if none is not None and k >= 3:
        others = [i for i in base if i != none]
        base = others[:k // 2] + [none] + others[k // 2:]
    return [base, base[::-1]]


def rendered_orders(q: dict) -> list[dict]:
    """The question as it is printed in each order: a positional-letter
    choice (the CHOICE template) whose `meaning` maps each letter back to
    the question's key, and whose `order` is the permutation."""
    q = typed(q)
    out = []
    digits = score_label_kind(q) == "digits"
    for order in two_orders(len(q["keys"]), q.get("none")):
        c = choice(q["text"], [q["options"][i] for i in order])
        if digits:                  # each level printed with its own number
            c["labels"] = [q["keys"][i] for i in order]
            c["label_kind"] = "digits"
        c["meaning"] = {lab: q["keys"][i] for lab, i in zip(c["labels"],
                                                            order)}
        c["order"] = list(order)
        if q["type"] == "noul":     # the option words (case_variants)
            c["words"] = list(NOUL_TEXTS)
        out.append(c)
    return out


def tv_distance(p: dict, q: dict) -> float:
    """Total-variation distance, 0 (identical) .. 1 (disjoint)."""
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def answer_probs(a: dict) -> dict:
    """An answer's distribution over its keys: a noul as {true, false}, a
    choice or score as its `probabilities`."""
    if a.get("type") == "noul":
        p = float(a["noul"])
        return {"true": p, "false": 1.0 - p}
    return dict(a.get("probabilities") or {})


def read(state: str, q: dict, *, slot: int | None, post=None,
         upstream=None, timeout: float = 120, cache: bool | None = None,
         render=None, exclude=(), prior_for=None) -> dict:
    """ONE typed question over `state` on the caller's slot, in two orders.

    render(c) -> messages: a caller's own rendering of the order `c`
      (rendered_orders; decide_turn.choose puts its options in their own
      message), ending on the ANSWER_LEAD prefill; default messages().
    exclude: keys that are not answers this time (read, then dropped and
      the rest renormalised, per order; they read 0 in `probabilities`, and
      the confidence is over the options left).
    prior_for(c) -> {key: p} | None: the content-free read to divide out,
      only for a question built with prior="content_free".

    Returns Jev's answer -- {type, name, choice, probabilities, confidence}
    | {type, name, score, probabilities, confidence, legend} | {type, name,
    noul} -- plus `diagnostics` {keys, argmax, tie, tied?, margin, orders,
    disagreement, argmax_agree, label_mass_min, excluded, prior, readout,
    template, ms, prompt_tokens, processed_tokens, reads}. Raises
    DeciderUnavailable."""
    post, upstream = post or _post_default, upstream or _upstream
    q = typed(q)
    t0 = time.time()
    keys = q["keys"]
    ex = sorted({str(x) for x in exclude or ()} & set(keys))
    allowed = [k for k in keys if k not in ex]
    if not allowed:
        raise DeciderUnavailable(
            "ALL_EXCLUDED", f"every option of {q['name']!r} is excluded",
            False, "the caller: exclude fewer options")
    use_prior = q.get("prior") == "content_free" and prior_for is not None
    per = []
    if cache is None:              # THE READ REGIME: the model's measured one
        cache = read_cache()
    temp = temperature_of(q["name"])
    for c in rendered_orders(q):
        a = ask_one(state, c, slot=slot, post=post, upstream=upstream,
                    timeout=timeout, cache=cache,
                    msgs=render(c) if render else None)
        raw = meaning_probs(a, c)
        p = temper(raw, temp["value"])            # THE TEMPERATURE (T = 1:
        prior = prior_for(c) if use_prior else None   # unchanged)
        if prior:
            p = contextual(p, prior)
        z = sum(p.get(k, 0.0) for k in allowed)
        p = {k: ((p.get(k, 0.0) / z if z > 0 else 1.0 / len(allowed))
                 if k in allowed else 0.0) for k in keys}
        cv = a.get("case_variants") or {}
        wm = cv.get("word") or {}
        per.append({"printed": [keys[i] for i in c["order"]], "probs": p,
                    # the order's label distribution as read, before the
                    # temperature, the prior and the exclusion (readout_of)
                    "raw": {k: round(float(raw.get(k, 0.0)), 6)
                            for k in keys},
                    "label_mass": a.get("label_mass"),
                    # case_variants: where the rest of the top K went
                    "variant_mass": cv.get("variant_mass"),
                    "word_mass": ({"true" if w == NOUL_TEXTS[0] else
                                   "false": v for w, v in wm.items()}
                                  if q["type"] == "noul" and cv.get(
                                      "available") else None),
                    "by_spelling": cv.get("by_spelling"),
                    "exact": a.get("exact"),
                    "prompt_tokens": a.get("prompt_tokens"),
                    "processed_tokens": a.get("processed_tokens"),
                    "cached_tokens": a.get("cached_tokens"),
                    # all of this order's HTTP reads (K re-reads included)
                    "http_reads": a.get("reads"),
                    "processed_all": (
                        sum(int(x.get("prompt_n") or 0)
                            for x in a.get("read_log") or [])
                        if all(x.get("prompt_n") is not None
                               for x in a.get("read_log") or [])
                        else None),
                    "ms": a.get("total_ms")})
    avg = {k: sum(o["probs"][k] for o in per) / len(per) for k in keys}
    band = tie_band_of()
    dec = decision({k: avg[k] for k in allowed}, band["value"])
    s = sorted((avg[k] for k in allowed), reverse=True)
    top = max(allowed, key=lambda k: avg[k])     # first in the given order
    tops = [max(allowed, key=lambda k, o=o: o["probs"][k]) for o in per]
    masses = [o["label_mass"] for o in per if o["label_mass"] is not None]
    lm = min(masses) if masses else None
    diag = {"keys": keys, "argmax": top, "tie": dec["tie"],
            **({"tied": dec["tied"]} if dec["tie"] else {}),
            "margin": round(s[0] - s[1], 6) if len(s) > 1 else 1.0,
            "orders": [dict(o, probs={k: round(v, 6)
                                      for k, v in o["probs"].items()})
                       for o in per],
            "disagreement": round(tv_distance(per[0]["probs"],
                                              per[-1]["probs"]), 6),
            "argmax_agree": len(set(tops)) == 1,
            "label_mass_min": None if lm is None else round(lm, 6),
            # DIAGNOSTICS (case_variants), never acted on: the most any
            # order put on unread case/spacing variants of its labels, and
            # (a noul) on the option WORDS instead of the letters
            "variant_mass_max": max((o["variant_mass"] for o in per
                                     if o["variant_mass"] is not None),
                                    default=None),
            "word_mass_max": max((sum(o["word_mass"].values()) for o in per
                                  if o["word_mass"] is not None),
                                 default=None),
            "temperature": temp,
            "read_regime": "cached" if cache else "cold",
            "score_labels": (score_label_kind(q) if q["type"] == "score"
                             else None),
            "excluded": ex, "prior": q.get("prior") if use_prior else None,
            "model": band["model"],
            "tie_band": {k: band[k] for k in ("value", "measured")},
            "readout": READOUT_VERSION + (
                "+digits" if q["type"] == "score"
                and score_label_kind(q) == "digits" else ""),
            "template": TEMPLATE_VERSION,
            "prompt_tokens": sum(o.get("prompt_tokens") or 0 for o in per),
            "processed_tokens": sum(o.get("processed_tokens") or 0
                                    for o in per),
            "reads": len(per), "ms": round((time.time() - t0) * 1000, 1)}
    out: dict = {"type": q["type"], "name": q["name"]}
    probs = {k: round(avg[k], 6) for k in keys}
    if q["type"] == "noul":
        out["noul"] = probs["true"]
    else:
        conf = round(confidence({k: avg[k] for k in allowed}), 6)
        if q["type"] == "choice":
            out.update(choice=top, probabilities=probs, confidence=conf)
        else:
            out.update(score=round(sum(int(k) * avg[k] for k in keys), 6),
                       probabilities=probs, confidence=conf,
                       legend=dict(zip(keys, q["options"])))
    out["diagnostics"] = diag
    return out


def _decide_typed(state: str, questions: list[dict], *, keep_slot: bool,
                  post, upstream, timeout: float, slot: int | None,
                  cache: bool) -> dict:
    """decide() for typed questions, Jev's response shape: {model, answers:
    {name: answer}, usage: {input_tokens, output_tokens}} plus where it ran
    (template, readout, slot, how, release, batch_ms). Validated first
    (nothing is sent for a malformed batch or a repeated name), then read in
    order on one slot, released after."""
    qs = [typed(q) for q in questions]
    names = [q["name"] for q in qs]
    if len(set(names)) != len(names):
        raise _refuse("DUPLICATE_NAME", "every question in a batch needs its "
                      "own name (the answers are keyed by it)")
    import slots
    t0 = time.time()
    grant = None
    how = "given by the caller"
    if slot is None:
        grant = slots.acquire(None, transient=True)
        slot, how = grant.get("slot"), grant.get("how")
    answers: dict = {}
    try:
        for q in qs:
            answers[q["name"]] = read(state, q, slot=slot, post=post,
                                      upstream=upstream, timeout=timeout,
                                      cache=cache)
    finally:
        if grant is not None:
            slots.release(grant)
        rel = ({"released": False, "skipped": "keep_slot"} if keep_slot
               else slots.lane_kept_note(slot, "decider batch", log=[])
               if grant is not None and slots.lane_kept()
               else release(slot))
    d = [a["diagnostics"] for a in answers.values()]
    return {"model": model_name(), "answers": answers,
            "usage": {"input_tokens": sum(x["prompt_tokens"] for x in d),
                      "output_tokens": sum(x["reads"] for x in d)},
            "model_profile": profile_status(),
            "template": TEMPLATE_VERSION, "readout": READOUT_VERSION,
            "slot": slot, "how": how, "release": rel,
            "batch_ms": round((time.time() - t0) * 1000, 1)}
