#!/usr/bin/env python
"""The Bonsai typed decider (mcp/decider_bonsai.py). No GPU, no network: the
model server is a fake passed in (post=, upstream=).

    python mcp/test_decider_bonsai.py      -> "N/M checks passed"

WHAT THIS IS GATING:

  1. THE TEMPLATE: the wording is pinned (a change is a new TEMPLATE_VERSION
     and a re-measurement: bench/decider/bonsai_decider.py); the messages
     are system / state / question / assistant prefill, the question in
     its OWN user message (the hybrid model's checkpoint at the last user
     message keeps the state reusable), thinking off by the template's
     switch, one token, logprobs on, never re-shaped by tiers.apply.
  2. THE READ: each label is the sum of its single-token spellings;
     the distribution is renormalised over the labels; the decision is
     argmax; a label missing from the top K raises K tenfold, and an unread
     spelling is bounded, never guessed.
  3. THE SLOT: one transient slot for the whole batch, released after it
     (unless keep_slot); failures carry situation, retryable, remedy.
  4. PER MODEL ([per-model]): the tie band and the profile keyed by the
     serving model's name; Bonsai's built-in values unchanged; a model with
     no measured record gets Bonsai's value RECORDED as unmeasured; a
     malformed record is rejected with its reason.
  5. THE BENCH TARGET ([bench]): bench/decider/decider_target.py resolves
     --model / --base-url (the one door unchanged; a llama-swap model that
     is not loaded refused), patches every door to the target and back out;
     bench/decider/measure_model.py's parts on a fake engine, and the record
     it writes is the one the runtime reads.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import offline_stores  # noqa: E402

_TMP = offline_stores.isolate("yamadori_test_decider_bonsai_")
os.environ["YAMADORI_SLOTS"] = "4"
# THE MODEL PROFILE's records: a temp dir, so a committed record under
# bench/decider/results/models/ never changes what this suite sees.
PROFILES = os.path.join(_TMP, "decider_models")
os.environ["YAMADORI_DECIDER_MODELS_DIR"] = PROFILES
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "bench", "decider"))

import decider_bonsai as D  # noqa: E402

CHECKS: list[tuple[str, bool]] = []


def check(name: str, ok: bool, detail=None) -> None:
    CHECKS.append((name, bool(ok)))
    if not ok:
        print("FAIL", name, "" if detail is None else str(detail)[:400])


# The pinned wording. Changing any of it changes what is measured.
PINNED = ("bonsai-decider/1"
          "|You read the material the user gives you and answer one question "
          "about it. Reply with the answer's label only."
          "|MATERIAL:\n\n"
          "|QUESTION: {question}\n\nAnswer yes or no."
          "|QUESTION: {question}\n\nOPTIONS:\n{options}\n\nAnswer with the "
          "letter of one option."
          "|{label}. {description}"
          "|Answer:")
PINNED_SHA = hashlib.sha256(PINNED.encode()).hexdigest()


def current() -> str:
    return "|".join([D.TEMPLATE_VERSION, D.SYSTEM, D.STATE_HEAD, D.YES_NO,
                     D.CHOICE, D.OPTION_LINE, D.ANSWER_LEAD])


# ---------------------------------------------------------------- fakes ----
IDS = {"yes": 9405, " yes": 9542, "Yes": 9175, " Yes": 7179,
       "no": 2083, " no": 874, "No": 2665, " No": 2233,
       "A": 32, " A": 357, "B": 33, " B": 417, "C": 34, " C": 356}


def upstream(path, payload=None, timeout=30):
    if path == "/tokenize":
        s = payload["content"]
        return {"tokens": [IDS[s]] if s in IDS else [1, 2]}
    if path == "/slots":
        return [{"id": 3, "n_prompt_tokens": 1}]
    raise AssertionError(path)


class FakeServer:
    """top_logprobs from a fixed distribution; only the first k entries."""

    def __init__(self, dist: dict[int, float]):
        self.dist = sorted(dist.items(), key=lambda kv: -kv[1])
        self.bodies: list[dict] = []

    def __call__(self, body, timeout):
        self.bodies.append(body)
        k = body["top_logprobs"]
        top = [{"id": i, "token": "?", "logprob": math.log(p)}
               for i, p in self.dist[:k]]
        return {"choices": [{"logprobs": {"content": [
            {"id": top[0]["id"], "logprob": top[0]["logprob"],
             "top_logprobs": top}]}}],
                "usage": {"prompt_tokens": 80},
                "timings": {"prompt_n": 30, "cache_n": 50,
                            "prompt_ms": 12.0}}


def _write_record(model: str, rec: dict) -> str:
    os.makedirs(PROFILES, exist_ok=True)
    path = os.path.join(PROFILES, f"{model}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rec, f)
    # a same-second rewrite must not hit the mtime cache
    D._PROFILE_CACHE.clear()
    return path


def _clear_records() -> None:
    if os.path.isdir(PROFILES):
        for n in os.listdir(PROFILES):
            os.remove(os.path.join(PROFILES, n))
    D._PROFILE_CACHE.clear()


def per_model() -> None:
    """THE MODEL PROFILE: every per-model constant keyed by the serving
    model's name; Bonsai's built-in values unchanged; another model without
    a measured record gets Bonsai's value RECORDED as unmeasured; a record
    measured for it replaces the value for it only."""
    import model as M
    _clear_records()
    check("[per-model] the records dir is the suite's temp dir",
          os.path.abspath(D.PROFILES_DIR) == os.path.abspath(PROFILES))
    check("[per-model] bonsai-agent is bonsai's profile (AGENTS.md: the "
          "same process)", D.canonical_model("bonsai-agent") == "bonsai"
          and D.canonical_model("flash-next") == "flash-next")
    b = D.tie_band_of("bonsai")
    check("[per-model] bonsai: the built-in, measured 0.0034 (bonsai_decider"
          " batching, n=100), byte for byte TIE_BAND",
          b["value"] == D.TIE_BAND == 0.0034 and b["measured"]
          and b["source"] == "built-in" and b["n"] == 100, b)
    check("[per-model] the serving model decides: model.MODEL bonsai -> "
          "tie_band() is Bonsai's", M.MODEL == "bonsai"
          and D.tie_band() == 0.0034 and D.model_name() == "bonsai")
    f = D.tie_band_of("flash-next")
    check("[per-model] flash-next unmeasured: Bonsai's value applied (as "
          "before the keying) and SAID so, never an invented number",
          f["value"] == 0.0034 and f["measured"] is False
          and f["source"].startswith("unmeasured for flash-next")
          and "measure_model.py --model flash-next" in f["source"], f)
    st = D.profile_status("flash-next")
    check("[per-model] status: every field unmeasured for flash-next, "
          "listed", st["unmeasured"] == list(D.PROFILE_FIELDS)
          and st["measured"] == [] and st["record"] is None
          and st["note"].startswith("unmeasured for flash-next")
          and st["tie_band"]["measured"] is False, st)
    sb = D.profile_status("bonsai-agent")
    check("[per-model] status: bonsai's built-in fields measured, the rest "
          "(letter_prior, label_bias, legacy_vs_typed) unmeasured",
          sb["model"] == "bonsai" and set(sb["measured"]) == {
              "tie_band", "labels", "readout", "legacy_form"}
          and set(sb["unmeasured"]) == {"letter_prior", "label_bias",
                                        "legacy_vs_typed", "temperature",
                                        "read_regime"},
          sb)
    # a measured record for flash-next
    _write_record("flash-next", {
        "version": D.PROFILE_VERSION, "model": "flash-next",
        "tie_band": {"value": 0.012, "n": 100, "date": "2026-10-01",
                     "script": "bench/decider/measure_model.py --only "
                               "tie_band"},
        "letter_prior": {"date": "2026-10-01", "script": "x", "n": 50}})
    f = D.tie_band_of("flash-next")
    check("[per-model] a measured record: flash-next's own band, measured, "
          "sourced to the record", f["value"] == 0.012 and f["measured"]
          and f["source"].endswith("flash-next.json"), f)
    check("[per-model] ... and Bonsai's is untouched",
          D.tie_band("bonsai") == 0.0034)
    st = D.profile_status("flash-next")
    check("[per-model] status: tie_band and letter_prior measured, the rest"
          " not", st["measured"] == ["letter_prior", "tie_band"]
          and "label_bias" in st["unmeasured"], st)
    # decision() and read() at the serving model's band
    p = {"A": 0.505, "B": 0.495}
    old = M.MODEL
    try:
        M.MODEL = "flash-next"
        check("[per-model] decision() uses the SERVING model's band: a 0.01"
              " gap is a tie on flash-next (0.012)",
              D.decision(p)["tie"] is True and D.model_name() == "flash-next")
        srv = FakeServer({357: 0.502, 417: 0.498})
        D._SPELL_IDS.clear()
        a = D.read("s", D.q_choice("c", "Which?", ["x", "y"]), slot=3,
                   post=srv, upstream=upstream)
        d = a["diagnostics"]
        check("[per-model] read() diagnostics name the model and its band "
              "(value, measured)", d["model"] == "flash-next"
              and d["tie_band"] == {"value": 0.012, "measured": True}
              and d["tie"] is True, d)
    finally:
        M.MODEL = old
    check("[per-model] ... on bonsai the same gap decides",
          D.decision(p) == {"answer": "A", "tie": False})
    # malformed records are rejected, and the fallback says so
    _write_record("flash-next", {
        "version": D.PROFILE_VERSION, "model": "flash-next",
        "tie_band": {"value": 0.9, "n": 5, "date": "d", "script": "s"}})
    f = D.tie_band_of("flash-next")
    st = D.profile_status("flash-next")
    check("[per-model] an out-of-range band is REJECTED (not used), with "
          "the reason", f["measured"] is False and f["value"] == 0.0034
          and "tie_band" in st.get("rejected", {}), st)
    _write_record("flash-next", {
        "version": D.PROFILE_VERSION, "model": "flash-next",
        "tie_band": {"value": 0.01, "n": 5}})
    check("[per-model] a field without its script and date is rejected "
          "(claims carry their evidence)",
          D.tie_band_of("flash-next")["measured"] is False
          and "script" in D.profile_status("flash-next")["rejected"][
              "tie_band"])
    _write_record("flash-next", {"version": D.PROFILE_VERSION,
                                 "model": "bonsai",
                                 "tie_band": {"value": 0.01, "n": 5,
                                              "date": "d", "script": "s"}})
    check("[per-model] a record naming another model is refused whole",
          D.tie_band_of("flash-next")["measured"] is False
          and "names model 'bonsai'" in D.profile_status(
              "flash-next").get("error", ""))
    _write_record("flash-next", {"version": "other/9", "model": "flash-next"})
    check("[per-model] another record version is refused",
          "not a decider-model/1 record" in D.profile_status(
              "flash-next").get("error", ""))
    # a batch reports the profile
    _clear_records()
    srv = FakeServer({357: 0.7, 417: 0.3})
    D._SPELL_IDS.clear()
    out = D.decide("s", [D.q_noul("n", "Is it?")], post=srv,
                   upstream=upstream, slot=3, keep_slot=True)
    mp = out.get("model_profile") or {}
    check("[per-model] a typed batch reports its model's profile (bonsai: "
          "the tie band measured, what is not)", mp.get("model") == "bonsai"
          and mp["tie_band"]["measured"] is True
          and "letter_prior" in mp["unmeasured"], mp)
    # the doors resolve at call time (a bench points them at one engine)
    seen = []
    saved = D._post_default, D._upstream
    try:
        D._post_default = lambda b, timeout: (seen.append("post"),
                                              srv(b, timeout))[1]
        D._upstream = lambda path, payload=None, timeout=30: (
            seen.append(path), upstream(path, payload, timeout))[1]
        D._SPELL_IDS.clear()
        D.ask_one("s", D.choice("Which?", ["x", "y"]), slot=1)
        check("[per-model] ask_one with no post/upstream uses the module's "
              "doors as they are NOW (not bound at import)",
              "post" in seen and "/tokenize" in seen, seen)
    finally:
        D._post_default, D._upstream = saved
        D._SPELL_IDS.clear()


class FakeEngine:
    """A bare llama-server for decider_target / measure_model: /tokenize,
    /slots, /props and /v1/chat/completions (FakeServer's answer)."""

    def __init__(self, dist=None, busy=()):
        self.chat = FakeServer(dist or {357: 0.5, 417: 0.3, 356: 0.2})
        self.urls: list[str] = []
        self.busy = set(busy)

    def __call__(self, url, payload=None, timeout=30):
        self.urls.append(url)
        if url.endswith("/v1/chat/completions"):
            return self.chat(payload, timeout)
        if url.endswith("/tokenize"):
            return upstream("/tokenize", payload)
        if url.endswith("/slots"):
            return [{"id": i, "is_processing": i in self.busy}
                    for i in range(2)]
        if url.endswith("/props"):
            return {"model_path": "C:/models/x/flash.gguf",
                    "build_info": "b1", "total_slots": 2}
        raise AssertionError(url)


# ============================================= HARDENING (SGLang, 2026-09-30)
def _served_template():
    import jinja2
    env = jinja2.Environment()
    env.filters["tojson"] = lambda v, **k: json.dumps(v, ensure_ascii=False)
    return env.from_string(open(os.path.join(
        HERE, "fixtures", "bonsai_chat_template.jinja"),
        encoding="utf-8").read())


def served_continuation(tpl, msgs: list[dict], **kw) -> str:
    """llama-server's rendering of a body that ends on an assistant message
    (every engine tree: server-common.cpp prefill_assistant -> continue_
    final_message AUTO; chat.cpp AUTO -> REASONING when the message has
    reasoning and no content, else CONTENT; chat-auto-parser-generator.cpp:
    messages[:-1], the generation prompt cut at the reasoning start, the
    start, the reasoning, the END only for CONTENT, then the content). The
    reasoning markers are found the way the server's differential analysis
    finds them: from the template's own rendering of a past turn."""
    head = tpl.render(messages=msgs[:-1], add_generation_prompt=False, **kw)
    gen = tpl.render(messages=msgs[:-1], add_generation_prompt=True,
                     **kw)[len(head):]
    probe = tpl.render(messages=[{"role": "user", "content": "u"},
                                 {"role": "assistant", "content": "CCC",
                                  "reasoning_content": "RRR"}],
                       add_generation_prompt=False)
    i = probe.rindex("<think>")
    start = probe[i:probe.index("RRR", i)]
    end = probe[probe.index("RRR", i) + 3:probe.index("CCC", i)]
    last = msgs[-1]
    reasoning = last.get("reasoning_content") or ""
    content = last.get("content") or ""
    mode = "reasoning" if reasoning and not content else "content"
    out = head + gen[:gen.find(D.THINK_START)] + start + reasoning
    if mode == "content":
        out += end + content
    return out


def hardening() -> None:
    """THE SGLANG FINDINGS (docs/JJAVA.md 8): the think-block guard, the
    case-variant diagnostics, the digit score labels, the temperature."""
    _clear_records()
    tpl = _served_template()

    # --- [guard] every question form, through the served template
    forms = [D.yes_no("Is it?"), D.choice("Which?", ["one", "two", "three"])]
    for q in (D.q_noul("n", "Is it?"),
              D.q_noul("nc", "Is it?", criteria={"true": "it is",
                                                 "false": "it is not"}),
              D.q_choice("c", "Which?", ["one", "two", "none"], none=2),
              D.q_score("s", "How much?", ["none", "some", "all"]),
              D.q_score("sd", "How much?", ["none", "some", "all"],
                        label_kind="digits")):
        forms += D.rendered_orders(q)
    bodies = [D.body("state with a stray <think> in it", f, 3, 20, "bonsai")
              for f in forms]
    checks = [D.guard_body(b) for b in bodies]
    rendered = [served_continuation(tpl, b["messages"], enable_thinking=False)
                for b in bodies]
    gs = [D.prompt_guard(r, after=D.question_text(f).splitlines()[-1])
          for r, f in zip(rendered, forms)]
    check("[guard] every question form (yes/no, choice, noul, criteria, "
          "none-choice, score, digit score; both orders) passes guard_body "
          "and renders with the think block CLOSED, ending on the answer "
          "lead (served template, llama-server's continuation)",
          all(c["ok"] for c in checks) and all(g["ok"] for g in gs)
          and all(r.endswith("<|im_start|>assistant\n<think>\n\n</think>"
                             "\n\nAnswer:") for r in rendered),
          [g for g in gs if not g["ok"]][:2] or rendered[0][-80:])
    # the guard DISCRIMINATES: a reasoning-only prefill (directive_prefill's
    # shape) leaves the block open, and prompt_guard sees it
    bad = bodies[0]["messages"][:-1] + [{"role": "assistant", "content": "",
                                        "reasoning_content": "Answer:"}]
    r_bad = served_continuation(tpl, bad, enable_thinking=False)
    g_bad = D.prompt_guard(r_bad)
    check("[guard] a reasoning-only prefill renders an OPEN think block and "
          "prompt_guard refuses it", g_bad["open"] and not g_bad["ok"],
          r_bad[-60:])
    check("[guard] a state that quotes <think> does not trip the rendered "
          "check (only the text after the question is read)",
          "stray <think>" in rendered[0] and gs[0]["ok"])
    # thinking on renders the template's thinking generation prompt
    on = tpl.render(messages=bodies[0]["messages"][:-1],
                    add_generation_prompt=True, enable_thinking=True)
    check("[guard] with thinking ON the template's generation prompt ends "
          "inside an open block (why thinking must be off)",
          D.prompt_guard(on + "Answer:")["open"], on[-30:])

    def refused(b, code):
        try:
            D.guard_body(b)
            return False
        except D.DeciderUnavailable as e:
            return e.code == code and e.retryable is False

    b0 = bodies[0]
    variants = {
        "THINKING_ON": [dict(b0, chat_template_kwargs={"enable_thinking":
                                                       True}),
                        dict(b0, chat_template_kwargs={}),
                        dict(b0, enable_thinking=True),
                        dict(b0, reasoning_effort="high")],
        "THINK_BLOCK_OPEN": [
            dict(b0, messages=b0["messages"][:-1] + [
                {"role": "assistant", "content": "Answer:",
                 "reasoning_content": "hmm"}]),
            dict(b0, messages=b0["messages"][:-1] + [
                {"role": "assistant", "content": "<think>Answer:"}])],
        "NO_ANSWER_PREFILL": [
            dict(b0, messages=b0["messages"][:-1]),
            dict(b0, messages=b0["messages"][:-1] + [
                {"role": "assistant", "content": ""}]),
            dict(b0, messages=b0["messages"][:-1] + [
                {"role": "assistant", "content": "Sure"}]),
            dict(b0, messages=b0["messages"] + [
                {"role": "assistant", "content": "Answer:"}])]}
    check("[guard] refused, not retryable: thinking on (either switch, "
          "reasoning_effort), reasoning in the prefill or an unclosed "
          "<think> in it, no prefill / an empty one / one not ending on the "
          "lead / two assistant messages at the end",
          all(refused(b, code) for code, bs in variants.items()
              for b in bs))
    srv = FakeServer({32: 0.9, 33: 0.1})
    try:
        D.ask_one("s", D.as_choice(D.yes_no("x")), slot=3, post=srv,
                  upstream=upstream, msgs=bad)
        check("[guard] ask_one refuses a caller's open rendering", False)
    except D.DeciderUnavailable as e:
        check("[guard] ask_one refuses a caller's rendering that would leave "
              "the block open BEFORE anything is sent",
              e.code == "THINK_BLOCK_OPEN" and srv.bodies == [])

    def up_tpl(prompt):
        def f(path, payload=None, timeout=30):
            if path == "/apply-template":
                f.seen = payload
                return {"prompt": prompt(payload)}
            return upstream(path, payload, timeout)
        return f
    ok_up = up_tpl(lambda pl: served_continuation(
        tpl, pl["messages"], enable_thinking=False))
    tc = D.template_check(ok_up)
    check("[guard] template_check (the window's live check): the body "
          "jjava sends, thinking off, read back closed",
          tc["checked"] and tc["ok"] and not tc["open"]
          and ok_up.seen["chat_template_kwargs"] == {"enable_thinking":
                                                     False}, tc)
    tc2 = D.template_check(up_tpl(lambda pl: "x<think>\nAnswer:"))
    tc3 = D.template_check(lambda *a, **k: (_ for _ in ()).throw(
        OSError("down")))
    check("[guard] template_check reports an open rendering, and an "
          "unanswered check as unchecked with why",
          tc2["checked"] and tc2["open"] and not tc2["ok"]
          and tc3["checked"] is False and "down" in tc3["why"], [tc2, tc3])

    # --- [variants] case-variant label mass (diagnostic only)
    ids = {"yes": {"yes": 9405, " yes": 9542, "Yes": 9175, " Yes": 7179},
           "no": {"no": 2083, " no": 874}}
    top = [{"id": 9175, "token": "Yes", "logprob": math.log(0.4)},
           {"id": 9405, "token": "yes", "logprob": math.log(0.3)},
           {"id": 874, "token": " no", "logprob": math.log(0.2)},
           {"id": 555, "token": " YES!", "logprob": math.log(0.05)},
           {"id": 556, "token": "NO", "logprob": math.log(0.03)}]
    cv = D.case_variants(top, ids)
    check("[variants] raw yes/no: each read spelling's own mass (Yes 0.4 "
          "beside yes 0.3), an unread spelling in another case (NO) counted "
          "as that label's variant",
          cv["available"] and abs(cv["by_spelling"]["Yes"] - 0.4) < 1e-6
          and abs(cv["by_spelling"]["yes"] - 0.3) < 1e-6
          and abs(cv["variant"]["no"] - 0.03) < 1e-6
          and cv["variant"]["yes"] == 0.0, cv)
    check("[variants] no token text from the server -> not available",
          D.case_variants([{"id": 1, "logprob": -1.0}], ids)
          == {"available": False})

    class WordServer(FakeServer):
        def __call__(self, body, timeout):
            d = super().__call__(body, timeout)
            text = {32: "A", 33: "B", 700: "Yes", 701: "a", 702: "no"}
            for t in d["choices"][0]["logprobs"]["content"][0][
                    "top_logprobs"]:
                t["token"] = text.get(t["id"], "?")
            return d
    ws = WordServer({32: 0.5, 33: 0.2, 700: 0.15, 701: 0.03, 702: 0.02})
    a = D.read("s", D.q_noul("w", "Is it?"), slot=3, post=ws,
               upstream=upstream)
    dg = a["diagnostics"]
    check("[variants] a lettered noul: the mass on the WORDS yes/no "
          "(every case) and on an unread letter case is reported per order "
          "and as maxima; the answer's probabilities are the letters' alone",
          abs(dg["word_mass_max"] - 0.17) < 1e-6
          and abs(dg["variant_mass_max"] - 0.03) < 1e-6
          and all(abs(o["word_mass"]["true"] - 0.15) < 1e-6
                  and abs(o["word_mass"]["false"] - 0.02) < 1e-6
                  for o in dg["orders"])
          and abs(a["noul"] - 0.5) < 1e-6, dg)

    # --- [digits] the score labels behind a switch
    q3 = D.q_score("sd", "How much?", ["none", "some", "all"],
                   label_kind="digits")
    o3 = D.rendered_orders(q3)
    check("[digits] each level printed with its OWN number in both orders; "
          "meaning is the level; the instruction asks for a number",
          [c["labels"] for c in o3] == [["0", "1", "2"], ["2", "1", "0"]]
          and all(c["meaning"] == {"0": "0", "1": "1", "2": "2"} for c in o3)
          and "2. all\n1. some\n0. none" in D.question_text(o3[1])
          and D.question_text(o3[0]).endswith(
              "Answer with the number of one level."),
          [D.question_text(c) for c in o3])
    lt = D.rendered_orders(D.q_score("s", "How much?", ["none", "some",
                                                         "all"]))
    check("[digits] the default is unchanged: lettered, the CHOICE line",
          D.SCORE_LABELS == "letters"
          and [c["labels"] for c in lt] == [["A", "B", "C"]] * 2
          and "label_kind" not in lt[0]
          and D.question_text(lt[0]).endswith(
              "Answer with the letter of one option."))
    big = D.rendered_orders(D.q_score("b", "?", [str(i) for i in range(11)],
                                      label_kind="digits"))
    check("[digits] more than 10 levels stay lettered (no single digit)",
          big[0]["labels"][:2] == ["A", "B"])
    check("[digits] typed() keeps label_kind; an unknown kind is refused",
          D.typed(dict(q3))["label_kind"] == "digits")
    try:
        D.q_score("x", "?", ["a", "b"], label_kind="roman")
        check("[digits] bad kind refused", False)
    except D.DeciderUnavailable as e:
        check("[digits] an unknown label kind is refused",
              e.code == "BAD_LABEL_KIND")
    IDS.update({"0": 15, " 0": 220, "1": 16, " 1": 221, "2": 17, " 2": 222})
    ds = FakeServer({17: 0.6, 16: 0.3, 15: 0.1})
    ad = D.read("s", q3, slot=3, post=ds, upstream=upstream)
    check("[digits] read: probabilities keyed by level (the digit IS the "
          "level, so both orders agree); readout typed/2+digits",
          abs(ad["probabilities"]["2"] - 0.6) < 1e-6
          and ad["diagnostics"]["disagreement"] == 0
          and ad["diagnostics"]["readout"] == D.READOUT_VERSION + "+digits"
          and ad["diagnostics"]["score_labels"] == "digits"
          and "Answer with the number" in ds.bodies[0]["messages"][2][
              "content"], ad)
    saved = D.SCORE_LABELS
    D.SCORE_LABELS = "digits"
    try:
        sw = D.rendered_orders(D.q_score("s", "?", ["a", "b"]))
    finally:
        D.SCORE_LABELS = saved
    check("[digits] the switch (YAMADORI_JJAVA_SCORE_LABELS) turns it on for "
          "every score", sw[0]["labels"] == ["0", "1"])

    # --- [temperature]
    t2 = D.temper({"A": 0.8, "B": 0.2, "C": 0.0}, 2.0)
    check("[temperature] T over the label logits: p^(1/T) renormalised "
          "(0.8/0.2 at T=2 -> 2/3), a zero stays zero, T=1 unchanged",
          abs(t2["A"] - 2 / 3) < 1e-9 and t2["C"] == 0.0
          and D.temper({"A": 0.8, "B": 0.2}, 1.0) == {"A": 0.8, "B": 0.2})
    t0 = D.temperature_of("skill_item:x#1")
    check("[temperature] none fitted -> T = 1, said so",
          t0["value"] == 1.0 and t0["fitted"] is False, t0)
    rec = {"version": D.PROFILE_VERSION, "model": "bonsai",
           "temperature": {"script": "bench/decider/fit_temperature.py",
                           "date": "2026-09-30", "sets": {
                               "skill_item": {"value": 2.0, "n": 40,
                                              "heldout_ok": True},
                               "skill_inject": {"value": 0.5, "n": 40}}}}
    _write_record("bonsai", rec)
    st = D.profile("bonsai")
    check("[temperature] a set not shown to hold out by run is rejected "
          "with its reason (the whole field)",
          "temperature" in st["rejected"]
          and "heldout_ok" in st["rejected"]["temperature"]
          and D.temperature_of("skill_item:x")["value"] == 1.0, st)
    rec["temperature"]["sets"]["skill_inject"]["heldout_ok"] = True
    _write_record("bonsai", rec)
    tf = D.temperature_of("skill_item:x#1")
    check("[temperature] a fitted set applies to its family (the name's "
          "part before ':'), never to another model",
          tf["value"] == 2.0 and tf["fitted"] and tf["set"] == "skill_item"
          and D.temperature_of("skill_item:x", "flash-next")["value"] == 1.0
          and D.temperature_of("phase")["value"] == 1.0, tf)
    ts = FakeServer({32: 0.8, 33: 0.2})
    at = D.read("s", D.q_noul("skill_item:x#1", "Is it?"), slot=3, post=ts,
                upstream=upstream)
    raws = [o["raw"] for o in at["diagnostics"]["orders"]]
    want = D.readout_of(raws, 2.0)
    check("[temperature] read() tempers each order's raw distribution; "
          "readout_of rebuilds the answer from the logged raw; the raw is "
          "the read as it was",
          abs(at["noul"] - want["true"]) < 1e-6
          and abs(raws[0]["true"] - 0.8) < 1e-6
          and at["diagnostics"]["temperature"]["value"] == 2.0
          and abs(at["noul"] - 0.5) < 1e-6, at["diagnostics"])
    ts2 = FakeServer({32: 0.9, 33: 0.1})
    saved_on = D.TEMPERATURE_ON
    D.TEMPERATURE_ON = False
    try:
        a1 = D.read("s", D.q_choice("skill_item:y", "?", ["p", "q"]),
                    slot=3, post=ts2, upstream=upstream)
    finally:
        D.TEMPERATURE_ON = saved_on
    check("[temperature] switched off (YAMADORI_JJAVA_TEMPERATURE=0): read "
          "at T = 1", a1["diagnostics"]["temperature"]["value"] == 1.0)
    _clear_records()

    # --- [regime] the read regime: the model's measured one
    r0 = D.read_regime_of("bonsai")
    s0 = FakeServer({32: 0.9, 33: 0.1})
    a0 = D.read("s", D.q_noul("n", "?"), slot=3, post=s0, upstream=upstream)
    check("[regime] none measured: cached (the path before), said so; a "
          "read sends cache_prompt true",
          r0["mode"] == "cached" and r0["measured"] is False
          and all(b["cache_prompt"] is True for b in s0.bodies)
          and a0["diagnostics"]["read_regime"] == "cached", r0)
    _write_record("bonsai", {"version": D.PROFILE_VERSION, "model": "bonsai",
                             "read_regime": {
                                 "mode": "cold", "n": 300,
                                 "script": "bench/decider/determinism.py",
                                 "date": "2026-09-30"}})
    check("[regime] a mode not shown to repeat exactly is rejected (stays "
          "cached)", D.read_cache("bonsai") is True
          and "read_regime" in D.profile("bonsai")["rejected"])
    _write_record("bonsai", {"version": D.PROFILE_VERSION, "model": "bonsai",
                             "read_regime": {
                                 "mode": "cold", "n": 300,
                                 "repeats_exactly": True,
                                 "script": "bench/decider/determinism.py",
                                 "date": "2026-09-30"}})
    s1 = FakeServer({32: 0.9, 33: 0.1})
    a1 = D.read("s", D.q_noul("n", "?"), slot=3, post=s1, upstream=upstream)
    s2 = FakeServer({32: 0.9, 33: 0.1})
    D.read("s", D.q_noul("n", "?"), slot=3, post=s2, upstream=upstream,
           cache=True)
    check("[regime] measured cold: every read is fully cold (cache_prompt "
          "false), never for another model; an explicit cache= (a bench) "
          "still wins",
          D.read_regime_of("bonsai")["measured"]
          and all(b["cache_prompt"] is False for b in s1.bodies)
          and a1["diagnostics"]["read_regime"] == "cold"
          and D.read_cache("flash-next") is True
          and all(b["cache_prompt"] is True for b in s2.bodies))
    _clear_records()


def bench_target() -> None:
    """bench/decider/decider_target.py and measure_model.py, offline: a
    target by name or base URL, patched in and back out; the parts on a
    fake engine; the record they write is the one the runtime reads."""
    import decider_target as DT
    import measure_model as MM
    import model as M
    import slots
    _clear_records()
    t = DT.resolve(model=None, running=lambda: (_ for _ in ()).throw(
        AssertionError("the one door must not read /running")))
    check("[bench] no --model / --base-url: the one door, unchanged",
          t.mode == "one door" and t.model == "bonsai" and t.slot is None)
    check("[bench] bonsai-agent is the one door too",
          DT.resolve(model="bonsai-agent",
                     running=lambda: set()).mode == "one door")
    try:
        DT.resolve(model="flash-next", running=lambda: {"bonsai"})
        check("[bench] a llama-swap model that is not loaded is refused",
              False)
    except DT.NotRun as e:
        check("[bench] a llama-swap model not in /running is NOT RUN (a "
              "read would load it and swap the card)",
              "not loaded" in str(e))
    eng = FakeEngine()
    t = DT.resolve(model="flash-next", running=lambda: {"flash-next"},
                   http=eng)
    check("[bench] loaded in llama-swap: its /upstream/<model> path, the "
          "highest slot", t.mode == "llama-swap"
          and t.base.endswith("/upstream/flash-next") and t.slot == 1, t.base)
    t = DT.resolve(model="flash-next",
                   base_url="http://127.0.0.1:18095/v1/", http=eng)
    check("[bench] --base-url: normalised (no /v1, no slash), the given "
          "name kept", t.mode == "base-url"
          and t.base == "http://127.0.0.1:18095" and t.model == "flash-next")
    try:
        DT.resolve(model="x", base_url="127.0.0.1:18095", http=eng)
        check("[bench] a base URL without a scheme refused", False)
    except ValueError:
        check("[bench] a base URL without a scheme is refused", True)
    d = t.describe()
    check("[bench] the engine recorded: mode, base, the server's model file "
          "and build", d["model_path"] == "flash.gguf" and d["build"] == "b1"
          and d["mode"] == "base-url")
    # preflight on THE TARGET's server
    check("[bench] preflight: an idle target and no hermes -> ok",
          DT.preflight(t, tasklist=lambda: "")["slots_busy"] == [])
    busy = DT.resolve(model="flash-next", base_url="http://h:1", slot=0,
                      http=FakeEngine(busy={1}))
    for what, tl, tt in (("a slot processing", "", busy),
                         ("hermes.exe running", "hermes.exe 123", t)):
        try:
            DT.preflight(tt, tasklist=lambda tl=tl: tl)
            check(f"[bench] preflight stops on {what}", False)
        except DT.Busy:
            check(f"[bench] preflight stops on {what} (NOT RUN)", True)
    # install / restore
    saved = (M.MODEL, M.post, D._upstream, D._post_default, slots.acquire)
    import decide_turn as T
    dec_before = T.DECISIONS
    done = DT.install(t, decisions_dir=os.path.join(_TMP, "measure"))
    try:
        check("[bench] installed: the serving model is the target's name",
              D.model_name() == "flash-next"
              and D.tie_band_of()["measured"] is False)
        g = slots.acquire(None, transient=True)
        check("[bench] installed: every decider batch on the target's slot",
              g["slot"] == 1)
        n0 = len(eng.urls)
        D._SPELL_IDS.clear()
        out = D.decide("s", [D.q_noul("n", "Is it?")])
        check("[bench] installed: a decide() with no doors given reaches ONLY"
              " the target (chat and tokenize), on its slot",
              len(eng.urls) > n0 and all(u.startswith(
                  "http://127.0.0.1:18095/") for u in eng.urls[n0:])
              and {b.get("id_slot") for b in eng.chat.bodies[-2:]} == {1}
              and out["model"] == "flash-next"
              and out["release"].get("released") is False, eng.urls[n0:])
        M.post({"model": "bonsai", "messages": [], "top_logprobs": 1})
        check("[bench] installed: the one door (model.post) reaches the "
              "target, never llama-swap's bonsai",
              eng.urls[-1] == "http://127.0.0.1:18095/v1/chat/completions")
        check("[bench] installed: decisions go to the run's own log, not the"
              " live one", T.DECISIONS.startswith(os.path.join(_TMP,
                                                               "measure")))
    finally:
        done["restore"]()
    check("[bench] restore puts every patch back",
          (M.MODEL, M.post, D._upstream, D._post_default,
           slots.acquire) == saved and T.DECISIONS == dec_before)
    # the parts, on the fake engine (doors passed explicitly)
    lab = MM.part_labels(t)
    check("[bench] labels: one-token spellings found per label, the rest "
          "listed (the fake keeps only A-E and yes/no whole)",
          lab["ids"]["A"] == {"A": 32, " A": 357}
          and lab["single_letters_one_token"] is False
          and lab["yes_no_one_token"] is True
          and "Z" in lab["not_one_token"] and lab["n"] == 2 * 26 + 12
          and lab["script"].endswith("--only labels") and lab["date"], lab)
    eng.chat = FakeServer({357: 0.6, 417: 0.3, 356: 0.1})
    D._SPELL_IDS.clear()
    lp = MM.part_letter_prior(t, counts=(2, 3))
    r2, r3 = lp["renderings"]["typed"]["2"], lp["renderings"]["typed"]["3"]
    check("[bench] letter_prior: per count, both renderings, the letter "
          "distribution, its top and deviation from uniform",
          lp["n"] == 4 and set(lp["renderings"]) == {"typed", "choose"}
          and r2["top"] == "A" and abs(r2["p"]["A"] - 2 / 3) < 1e-6
          and abs(r3["max_dev"] - (0.6 - 1 / 3)) < 1e-6
          and lp["summary"]["typed"]["top_is_first"] == 2, lp["summary"])
    bodies = eng.chat.bodies[-4:]
    check("[bench] letter_prior: content-free -- the neutral state and "
          "every option N/A; the choose rendering puts the options in "
          "their own message", all(D.NEUTRAL_STATE in b["messages"][1][
              "content"] for b in bodies)
          and "A. N/A\nB. N/A" in bodies[0]["messages"][2]["content"]
          and bodies[1]["messages"][2]["content"].startswith("OPTIONS:\nA) "
                                                             "N/A"),
          [b["messages"][2]["content"][:40] for b in bodies])
    eng.chat = FakeServer({357: 0.55, 417: 0.25, 356: 0.1, 358: 0.1})
    D._SPELL_IDS.clear()
    lb = None
    try:
        lb = MM.part_label_bias(t)
    except D.DeciderUnavailable as e:
        check("[bench] label_bias ran", False, e.facts())
    if lb:
        qs = lb["questions"]
        check("[bench] label_bias: build_intent (noul), phase, choose:stop, "
              "two orders each", set(qs) == {"build_intent", "phase",
                                             "choose:stop"}
              and "noul" in qs["build_intent"]
              and len(qs["phase"]["orders"]) == 2 and lb["n"] == 6, lb)
    # the record: written by the bench, read by the runtime
    tb = MM.tie_band_of_batching({"n_states": 2, "questions_per_state": 5,
                                  "rows": [{"max_prob_diff": 0.0021,
                                            "same_answers": True},
                                           {"max_prob_diff": 0.0047,
                                            "same_answers": False}]})
    check("[bench] tie_band: the largest difference, n = states x "
          "questions", tb["value"] == 0.0047 and tb["n"] == 10
          and tb["same_answers"] == 1)
    fld = MM._field(t, "tie_band", tb.pop("n"), **tb)
    path = MM.write_field("flash-next", "tie_band", fld, out_dir=PROFILES,
                          log={"part": "tie_band"})
    MM.write_field("flash-next", "letter_prior", lp, out_dir=PROFILES)
    D._PROFILE_CACHE.clear()
    rec = json.load(open(path, encoding="utf-8"))
    f = D.tie_band_of("flash-next")
    check("[bench] the record the bench writes is the one the runtime reads:"
          " flash-next's band measured, 0.0047",
          f["measured"] and f["value"] == 0.0047
          and rec["version"] == D.PROFILE_VERSION
          and rec["model"] == "flash-next" and rec["log"][0]["part"]
          == "tie_band" and set(D.profile_status("flash-next")["measured"])
          == {"tie_band", "letter_prior"}, f)
    check("[bench] merged: a second part keeps the first",
          "tie_band" in rec and "letter_prior" in rec)
    _clear_records()


def main() -> int:
    # 1. the template
    check("template wording pinned (sha256)",
          hashlib.sha256(current().encode()).hexdigest() == PINNED_SHA)
    q = D.yes_no("Is it raining?")
    m = D.messages("the state", q)
    check("four messages: system, state, question, prefill",
          [x["role"] for x in m] == ["system", "user", "user", "assistant"])
    check("state first, question last in its own user turn",
          m[1]["content"] == "MATERIAL:\n\nthe state"
          and m[2]["content"].startswith("QUESTION: Is it raining?")
          and m[2]["content"].endswith("Answer yes or no."))
    check("prefill is the answer lead, no trailing space",
          m[3]["content"] == "Answer:")
    c = D.choice("Which?", ["one", "two", "three"])
    check("choice labels A, B, C in order", c["labels"] == ["A", "B", "C"])
    check("choice renders its options",
          "A. one\nB. two\nC. three" in D.question_text(c))
    b = D.body("s", q, 3, 20, "bonsai")
    check("one token, logprobs, thinking off, cached, on the slot",
          b["max_tokens"] == 1 and b["logprobs"] is True
          and b["top_logprobs"] == 20
          and b["chat_template_kwargs"] == {"enable_thinking": False}
          and b["cache_prompt"] is True and b["id_slot"] == 3
          and "reasoning_budget_tokens" not in b)
    try:
        D.choice("x", [])
        check("empty choice refused", False)
    except D.DeciderUnavailable as e:
        check("empty choice refused, not retryable, with a remedy",
              e.code == "NO_OPTIONS" and not e.retryable and e.remedy)
    try:
        D.choice("x", [str(i) for i in range(27)])
        check("27 options refused", False)
    except D.DeciderUnavailable as e:
        check("27 options refused (A..Z)", e.code == "TOO_MANY_OPTIONS")

    # 2. the read
    D._SPELL_IDS.clear()
    srv = FakeServer({9542: 0.6, 7179: 0.1, 874: 0.2, 11: 0.05, 12: 0.05})
    a = D.ask_one("s", q, slot=3, post=srv, upstream=upstream)
    check("spellings summed: yes = ' yes' + ' Yes'",
          abs(a["probs"]["yes"] - 0.7 / 0.9) < 1e-6)
    check("renormalised over the labels", abs(sum(a["probs"].values()) - 1)
          < 1e-9 and abs(a["label_mass"] - 0.9) < 1e-6)
    check("argmax", a["answer"] == "yes")
    check("one read when the labels are in the top K", a["reads"] == 1)
    check("timings carried", a["processed_tokens"] == 30
          and a["cached_tokens"] == 50 and a["prompt_tokens"] == 80)
    # 'no' outside the top 20 and able to change the answer (its bound,
    # p_min x its spellings, exceeds the gap): K is raised to 200.
    filler = {1000 + i: 0.045 for i in range(19)}
    srv = FakeServer({9542: 0.05, **filler, 874: 0.044})
    a = D.ask_one("s", q, slot=3, post=srv, upstream=upstream)
    check("a label outside the top K that could change the answer raises "
          "K tenfold", [x["top_logprobs"] for x in srv.bodies] == [20, 200]
          and a["reads"] == 2 and a["k"] == 200)
    check("the raised read finds it", a["probs"]["no"] > 0
          and not a["labels_unread"])
    # 'no' outside the top 20 but unable to change the answer: bounded, not
    # chased (a 39-option question chased K to the vocabulary before).
    filler = {1000 + i: 0.004 for i in range(30)}
    srv = FakeServer({9542: 0.8, **filler, 874: 0.0005})
    a = D.ask_one("s", q, slot=3, post=srv, upstream=upstream)
    check("an unread label that cannot change the argmax is not chased",
          a["reads"] == 1 and a["labels_unread"] == ["no"]
          and a["answer"] == "yes")
    # A label never found: its probability is 0, the bound is reported.
    srv = FakeServer({9542: 0.9, 11: 0.1})
    a = D.ask_one("s", q, slot=3, post=srv, upstream=upstream, n_vocab=200)
    check("an unread label stays unread and bounded, never guessed",
          a["labels_unread"] == ["no"] and a["probs"]["yes"] == 1.0
          and a["unread_bound"] > 0 and not a["exact"])
    srv = FakeServer({357: 0.1, 417: 0.7, 356: 0.1, 11: 0.1})
    a = D.ask_one("s", c, slot=3, post=srv, upstream=upstream)
    check("choice: argmax letter", a["answer"] == "B")

    # 3. the slot and the batch
    released = []
    D.release = lambda slot, why="": released.append(slot) or {  # noqa: E731
        "released": True, "slot": slot}
    srv = FakeServer({9542: 0.6, 874: 0.4})
    out = D.decide("state", [q, D.yes_no("Another?")], post=srv,
                   upstream=upstream)
    check("one transient slot for the batch",
          out["slot"] == 3 and {x.get("id_slot") for x in srv.bodies} == {3})
    check("the same state prefix for every question",
          len({x["messages"][1]["content"] for x in srv.bodies}) == 1)
    import slots as _slots
    if _slots.lane_kept():
        check("layout v2: the lane is KEPT after the batch (no release), and "
              "the result says so", released == []
              and out["release"]["released"] is False
              and "kept" in str(out["release"].get("skipped")), out["release"])
    else:
        check("released after the batch", released == [3]
              and out["release"]["released"])
    released.clear()
    out = D.decide("state", [q], post=srv, upstream=upstream, keep_slot=True)
    check("keep_slot skips the release and says so",
          released == [] and out["release"]["skipped"] == "keep_slot")

    def down(body, timeout):
        raise ConnectionRefusedError("refused")
    try:
        D.decide("s", [q], post=down, upstream=upstream)
        check("unreachable raises", False)
    except D.DeciderUnavailable as e:
        check("unreachable: retryable, with a remedy",
              e.code == "MODEL_UNREACHABLE" and e.retryable and e.remedy)
    try:
        D.decide("s", [], post=srv, upstream=upstream)
        check("no questions refused", False)
    except D.DeciderUnavailable as e:
        check("no questions refused", e.code == "NO_QUESTIONS")

    # 4. THE TYPED API (operator, 2026-09-29: "Jev has 3 modes to gate these
    # behind, noul is the yes/no")
    IDS.update({"D": 35, " D": 422, "E": 36, " E": 468})
    D._SPELL_IDS.clear()
    n = D.q_noul("intent", "Is it asked?")
    check("[typed] noul: keys true/false, printed as the pair yes/no",
          n["type"] == "noul" and n["keys"] == ["true", "false"]
          and n["options"] == ["yes", "no"])
    ro = D.rendered_orders(n)
    check("[typed] a noul's two orders render byte for byte as the measured "
          "mc_avg form (as_choice of the yes/no, given then reversed)",
          [D.question_text(c) for c in ro] ==
          [D.question_text(D.as_choice(D.yes_no("Is it asked?"), o))
           for o in ([0, 1], [1, 0])]
          and ro[1]["meaning"] == {"A": "false", "B": "true"})
    ok_orders = True
    for k in range(2, 9):
        for nn in [None] + list(range(k)):
            o1, o2 = D.two_orders(k, nn)
            ok_orders &= sorted(o1) == list(range(k)) and o2 == o1[::-1]
            if nn is not None and k >= 3:
                ok_orders &= o1[-1] != nn and o2[-1] != nn
    check("[typed] two orders: a permutation and its reverse; 'none' never "
          "last in either when there are 3+ options", ok_orders)
    check("[typed] with no none option the orders are the legacy ones",
          D.two_orders(4) == D.orders_for(D.choice("q", list("abcd")), 2))
    sc = D.q_score("done", "How done?", ["no", "some", "all"])
    check("[typed] score: levels lowest first, numbered 0..k as Jev numbers "
          "them", sc["keys"] == ["0", "1", "2"] and "values" not in sc)
    bad = []
    for fn in (lambda: D.typed({"type": "vote", "name": "x", "text": "?"}),
               lambda: D.q_choice("x", "?", ["only one"]),
               lambda: D.q_choice("x", "?", [str(i) for i in range(27)]),
               lambda: D.q_choice("x", "?", ["a", "b"], keys=["k", "k"]),
               lambda: D.q_noul("x", "?", criteria={"maybe": "?"}),
               lambda: D.q_noul("x", "?", prior="fitted")):
        try:
            fn()
            bad.append(None)
        except D.DeciderUnavailable as e:
            bad.append(e.code if not e.retryable else "retryable?")
    check("[typed] malformed questions are refused, not retryable",
          bad == ["BAD_TYPE", "NO_OPTIONS", "TOO_MANY_OPTIONS", "BAD_KEYS",
                  "BAD_CRITERIA", "BAD_PRIOR"], bad)
    # Jev's request form is read too: instructions + criteria.
    jc = D.typed({"type": "choice", "name": "dept", "instructions": "Which?",
                  "criteria": {"shipping": "Where is it", "returns": "Send "
                               "it back", "billing": "Charged twice"}})
    jn = D.typed({"type": "noul", "name": "esc", "instructions": "A human?",
                  "criteria": {"true": "asks for a person", "false":
                               "does not"}})
    js = D.typed({"type": "score", "name": "sev", "instructions": "How bad?",
                  "criteria": ["cosmetic", "workaround", "blocking"]})
    check("[typed] Jev's request form: criteria as a choice's {key: "
          "description}, a noul's {true, false}, a score's level list",
          jc["keys"] == ["shipping", "returns", "billing"]
          and jc["options"][2] == "Charged twice" and jc["text"] == "Which?"
          and jn["options"] == ["yes: asks for a person", "no: does not"]
          and js["keys"] == ["0", "1", "2"], (jc, jn, js))

    # CONFIDENCE: Jev's form, checked on the docs' own response examples
    # (docs.typesafe.ai llms-full.txt, 2026-09-29): (n x p_max - 1)/(n - 1).
    docs = [([0.88, 0.12, 0.0], 0.81), ([0.0, 0.95, 0.05], 0.92),
            ([0.85, 0.0, 0.15], 0.78), ([0.04, 0.35, 0.61], 0.42),
            ([0.0, 0.26, 0.0, 0.0, 0.74], 0.67), ([0.34, 0.4, 0.02, 0.24],
                                                   0.2),
            ([0.84, 0.16, 0.0], 0.76), ([0.0, 0.57, 0.43], 0.35),
            ([0.0, 0.76, 0.24], 0.64), ([0.0, 0.72, 0.28], 0.58),
            ([0.0, 0.91, 0.09], 0.87), ([0.0, 1.0], 1.0),
            ([0.0, 0.0, 0.0, 1.0], 1.0)]
    # The docs print probabilities and confidence to 2 decimals, so the form
    # can only agree within that rounding: +-0.005 on p_max moves it by
    # 0.005 n/(n-1), and the shown confidence is itself +-0.005.
    off = [(round(D.confidence(dict(enumerate(p))), 4), c)
           for p, c in docs
           if abs(D.confidence(dict(enumerate(p))) - c)
           > 0.005 * len(p) / (len(p) - 1) + 0.005 + 1e-9]
    check("[confidence] Jev's n-option form agrees with every docs example "
          "within the rounding of what the docs print", off == [], off)
    widget = [([0.0, 0.14, 0.86, 0.0, 0.0], 0.89), ([0.0, 0.0, 0.48, 0.52],
                                                    0.52)]
    check("[confidence] the two /primitives/score widget examples do NOT "
          "fit it (recorded in decider_bonsai THE TYPED API, not hidden)",
          all(abs(D.confidence(dict(enumerate(p))) - c) > 0.05
              for p, c in widget))
    check("[confidence] all mass on one option is 1.0, uniform is 0, "
          "never outside [0, 1]", D.confidence({"a": 1.0, "b": 0.0}) == 1.0
          and abs(D.confidence({"a": 0.25, "b": 0.25, "c": 0.25,
                                "d": 0.25})) < 1e-12
          and D.confidence({"a": 0.5}) == 1.0)

    # A model that prefers the letter A whatever it says (0.7 vs 0.2): the
    # two orders cancel it, and their disagreement shows it.
    pos = FakeServer({357: 0.7, 417: 0.2, 11: 0.1})
    r = D.read("s", n, slot=3, post=pos, upstream=upstream)
    dg = r["diagnostics"]
    check("[typed] noul: Jev's shape -- {type, name, noul} and no "
          "confidence (Jev: a noul carries none)",
          set(r) == {"type", "name", "noul", "diagnostics"}
          and r["type"] == "noul", sorted(r))
    check("[typed] read: two reads, averaged; a letter bias cancels to "
          "noul 0.5 (a tie in the diagnostics) and the orders' TV distance "
          "reports it",
          len(pos.bodies) == 2 and dg["tie"] and abs(r["noul"] - 0.5) < 1e-6
          and abs(dg["disagreement"] - (0.7 / 0.9 - 0.2 / 0.9)) < 1e-6
          and not dg["argmax_agree"] and abs(dg["label_mass_min"] - 0.9)
          < 1e-6 and dg["reads"] == 2, r)
    check("[typed] no threshold anywhere in the decider: no abstain, no "
          "params", not hasattr(D, "params_for") and not hasattr(D, "PARAMS")
          and "abstain" not in json.dumps(r))
    ch = D.q_choice("c", "Which?", ["x", "y", "z"], keys=["kx", "ky", "kz"])
    srv = FakeServer({357: 0.1, 417: 0.1, 356: 0.1, 11: 0.7})
    r = D.read("s", ch, slot=3, post=srv, upstream=upstream,
               exclude=["ky"])
    check("[typed] exclude: the key reads 0, the rest renormalise, and the "
          "confidence is over the options left",
          r["probabilities"]["ky"] == 0
          and abs(sum(r["probabilities"].values()) - 1) < 1e-6
          and r["diagnostics"]["excluded"] == ["ky"]
          and abs(r["confidence"] - D.confidence(
              {"kx": r["probabilities"]["kx"],
               "kz": r["probabilities"]["kz"]})) < 1e-5)
    srv = FakeServer({357: 0.8, 417: 0.1, 356: 0.1})
    r = D.read("s", ch, slot=3, post=srv, upstream=upstream)
    p = r["probabilities"]
    check("[typed] choice: Jev's shape -- {type, name, choice, "
          "probabilities (sum 1), confidence}; choice is the highest "
          "probability; confidence is computed on the AVERAGED distribution",
          set(r) == {"type", "name", "choice", "probabilities", "confidence",
                     "diagnostics"}
          and r["choice"] == max(p, key=p.get)
          and abs(sum(p.values()) - 1) < 1e-6
          and abs(r["confidence"] - D.confidence(p)) < 1e-5, r)
    srv = FakeServer({357: 0.1, 417: 0.1, 356: 0.6, 422: 0.1, 11: 0.1})
    s4 = D.q_score("done", "How done?", ["a", "b", "c", "d"])
    r = D.read("s", s4, slot=3, post=srv, upstream=upstream)
    # order 1 (0,1,2,3): C -> level 2; order 2 (3,2,1,0): C -> level 1
    lv = {0: (1 + 1) / 18, 1: (1 + 6) / 18, 2: (6 + 1) / 18, 3: (1 + 1) / 18}
    exp = sum(k * v for k, v in lv.items())
    check("[typed] score: Jev's shape -- {type, name, score, probabilities "
          "keyed '0'..'k', confidence, legend}; score = sum of level number "
          "x probability",
          set(r) == {"type", "name", "score", "probabilities", "confidence",
                     "legend", "diagnostics"}
          and abs(r["score"] - exp) < 1e-6
          and set(r["probabilities"]) == {"0", "1", "2", "3"}
          and r["legend"] == {"0": "a", "1": "b", "2": "c", "3": "d"}
          and abs(r["confidence"] - D.confidence(r["probabilities"])) < 1e-5,
          r)
    released.clear()
    srv = FakeServer({357: 0.5, 417: 0.3, 356: 0.2})
    out = D.decide("state", [n, ch], post=srv, upstream=upstream)
    check("[typed] decide(): Jev's response -- {model, answers keyed by "
          "name, usage}; one slot, released after; two reads each",
          out["readout"] == D.READOUT_VERSION
          and set(out["answers"]) == {"intent", "c"}
          and out["answers"]["c"]["type"] == "choice"
          and "noul" in out["answers"]["intent"]
          and out["usage"]["output_tokens"] == 4
          and out["usage"]["input_tokens"] == 4 * 80
          and "model" in out and len(srv.bodies) == 4
          and released == ([] if __import__("slots").lane_kept() else [3])
          and {x.get("id_slot") for x in srv.bodies} == {3},
          (out.get("readout"), sorted(out["answers"]), len(srv.bodies),
           released))
    try:
        D.decide("s", [n, D.q_noul("intent", "again?")], post=srv,
                 upstream=upstream)
        check("[typed] a repeated name refused", False)
    except D.DeciderUnavailable as e:
        check("[typed] a repeated name is refused before anything is sent "
              "(answers are keyed by it)", e.code == "DUPLICATE_NAME")
    try:
        D.decide("s", [n, q], post=srv, upstream=upstream)
        check("[typed] a mixed batch refused", False)
    except D.DeciderUnavailable as e:
        check("[typed] a batch mixing typed and old questions is refused",
              e.code == "NO_LABELS")

    # fit_tail: newest pieces kept, oldest first, linear.
    s, info = D.fit_tail(["a" * 40, "b" * 40, "c" * 40], 25,
                         lambda p: len(p) // 4)
    check("fit_tail keeps the newest that fit, in order",
          s == "b" * 40 + "\n\n" + "c" * 40 and info["dropped"] == 1)
    per_model()
    hardening()
    bench_target()
    ok = sum(1 for _, o in CHECKS if o)
    print(f"{ok}/{len(CHECKS)} checks passed")
    return 0 if ok == len(CHECKS) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:                                            # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
