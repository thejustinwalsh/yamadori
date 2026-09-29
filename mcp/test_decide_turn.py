#!/usr/bin/env python
"""decide_turn: a turn's judgment questions in one batch (mcp/decide_turn.py).
No GPU, no network: the model server is a fake.

    python mcp/test_decide_turn.py      -> "N/M checks passed"

GATES: the state is the current turn or step only, cut head+tail; which
questions run on which request; every question is a neutral-letter choice
asked in two orders and calibrated by its content-free read, made BEFORE
the state is placed; a near-tie falls back to the rule; a failure falls back
to the rules and says so; disagreements are logged with ids and labels,
never text; the slot is held for the turn and released once; pick() repeats
without the picked option until none wins, and leaves "none" out on hard
evidence. The rotation-and-label-prior checks run under the LEGACY readout
(YAMADORI_DECIDER_READOUT=legacy, kept for comparison).

THE TYPED READOUT (test_typed; operator 2026-09-29, Jev's API exactly):
build intent a noul read as a lettered pair in two orders, byte for byte
the measured mc_avg rendering; no content-free read; a letter-biased
model's two orders cancel and their disagreement is reported; the door
answers in Jev's response shape (answers keyed by name; noul a number;
choice + probabilities + confidence; score + probabilities + confidence +
legend over levels 0..k); no abstain anywhere -- the QUESTION SET's
THRESHOLDS decide tiers (untuned by default; a PER-MODEL fixture row here:
a low tier falls back, a medium one acts and says so), the label mass is a
diagnostic only; choice's "none" never printed last; choose() in two
orders with
positional letters, rounds extending each order's slot sequence (served
template) and the question's letter references rewritten per order;
judge_stop typed; every decision logged with its join keys, join_corpus
naming them, a later decision carrying corpus_turn; the wrappers
(build_intent, confirm_packages with the kept content-free veto, pick);
the legacy switch logged on the same log; no prime thread in typed mode.
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import offline_stores  # noqa: E402

_TMP = offline_stores.isolate("yamadori_test_decide_turn_")
os.environ["YAMADORI_SLOTS"] = "4"
LOG = os.path.join(tempfile.mkdtemp(prefix="decide_turn_log_"), "d.jsonl")
os.environ["YAMADORI_DECIDER_LOG"] = LOG
# THE MODEL PROFILE's records (decider_bonsai): a temp dir, so a committed
# record never changes what this suite sees.
PROFILES = os.path.join(_TMP, "decider_models")
os.environ["YAMADORI_DECIDER_MODELS_DIR"] = PROFILES

import decider_bonsai as D  # noqa: E402
import decide_turn as T  # noqa: E402

CHECKS: list[tuple[str, bool]] = []


def check(name: str, ok: bool, detail=None) -> None:
    CHECKS.append((name, bool(ok)))
    if not ok:
        print("FAIL", name, "" if detail is None else str(detail)[:400])


IDS = {"A": 32, " A": 357, "B": 33, " B": 417, "C": 34, " C": 356,
       "D": 35, " D": 422, "E": 36, " E": 468, " yes": 9542, " no": 874}


def upstream(path, payload=None, timeout=30):
    if path == "/tokenize":
        s = payload["content"]
        return {"tokens": [IDS[s]] if s in IDS else [1, 2]}
    raise AssertionError(path)


class Fake:
    """Answers by a rule over the rendered question: `want(state, options)`
    returns the option TEXT to favour (0.9) or None (uniform)."""

    def __init__(self, want):
        self.want, self.bodies = want, []

    def __call__(self, body, timeout):
        self.bodies.append(body)
        msgs = body["messages"]
        state = msgs[1]["content"][len(D.STATE_HEAD):]
        qtext = msgs[2]["content"]
        if "OPTIONS:" in qtext:
            opts = [ln.split(". ", 1) for ln in qtext.split("OPTIONS:\n")[1]
                    .split("\n\n")[0].splitlines()]
        else:                               # a plain yes/no (FORM "yn")
            opts = [["yes", "yes"], ["no", "no"]]
        # The content-free state carries no preference (a real model's
        # label prior is what calibration divides out).
        fav = (None if state == D.NEUTRAL_STATE
               else self.want(state, [o[1] for o in opts]))
        n = len(opts)
        top = []
        for lab, text in opts:
            p = (0.9 if text == fav else 0.1 / max(n - 1, 1)) if fav \
                else 1.0 / n
            top.append({"id": IDS[" " + lab], "logprob": math.log(p)})
        top.sort(key=lambda t: -t["logprob"])
        return {"choices": [{"logprobs": {"content": [
            {"id": top[0]["id"], "logprob": top[0]["logprob"],
             "top_logprobs": top}]}}], "usage": {"prompt_tokens": 50},
            "timings": {"prompt_n": 10, "cache_n": 40, "prompt_ms": 5.0}}


def down_fn():
    def down(body, timeout):
        raise ConnectionRefusedError("refused")
    return down


def user_turn(text):
    return [{"role": "system", "content": "sys"},
            {"role": "user", "content": text}]


def step(cmd, result):
    return [{"role": "user", "content": "build the game"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function", "function": {
                    "name": "terminal",
                    "arguments": json.dumps({"command": cmd})}}]},
            {"role": "tool", "tool_call_id": "c1", "content": result}]



# ======================================================= THE TYPED READOUT ==
# Operator, 2026-09-29: "Jev has 3 modes to gate these behind, noul is the
# yes/no." Every letter A..Z answers; a fake whose next token is decided by
# the printed option TEXT (want), by a printed LETTER (a position/letter
# bias), and whose labels carry only `mass` of the whole distribution.
for _i, _L in enumerate(D.LETTERS):
    IDS.setdefault(_L, 600 + _i)
    IDS.setdefault(" " + _L, 700 + _i)
FILLER = 99999


def printed_options(msgs):
    """[(letter, text)] of the message that holds the options (". " in the
    CHOICE template, ") " in choose()'s own options message)."""
    om = next(x["content"] for x in msgs if x["role"] == "user"
              and "OPTIONS:\n" in x["content"])
    out = []
    for ln in om.split("OPTIONS:\n")[1].split("\n\n")[0].splitlines():
        sep = ". " if ln[1:3] == ". " else ") "
        lab, text = ln.split(sep, 1)
        out.append((lab, text))
    return out


class TFake:
    def __init__(self, want=None, letter=None, mass=1.0, p=0.9):
        self.want, self.letter, self.mass, self.p = want, letter, mass, p
        self.bodies = []

    def __call__(self, body, timeout):
        self.bodies.append(body)
        m = body["messages"]
        state = m[1]["content"][len(D.STATE_HEAD):]
        opts = printed_options(m)
        n = len(opts)
        fav = self.want(state, [t for _l, t in opts]) if self.want else None
        top = []
        for lab, text in opts:
            if fav is not None:
                p = self.p if text == fav else (1 - self.p) / max(n - 1, 1)
            elif self.letter is not None:
                p = 0.8 if lab == self.letter else 0.2 / max(n - 1, 1)
            else:
                p = 1.0 / n
            top.append({"id": IDS[" " + lab],
                        "logprob": math.log(p * self.mass)})
        if self.mass < 1:
            top.append({"id": FILLER, "logprob": math.log(1 - self.mass)})
        top.sort(key=lambda t: -t["logprob"])
        return {"choices": [{"logprobs": {"content": [
            {"id": top[0]["id"], "logprob": top[0]["logprob"],
             "top_logprobs": top}]}}], "usage": {"prompt_tokens": 60},
            "timings": {"prompt_n": 15, "cache_n": 45, "prompt_ms": 5.0}}


def decisions_rows():
    path = T.DECISIONS
    if not os.path.exists(path):
        return []
    return [json.loads(x) for x in open(path, encoding="utf-8")]


def test_typed(released) -> None:
    T.READOUT = "typed"
    T.TEMPLATE_CF.clear()
    if os.path.exists(T.DECISIONS):
        os.remove(T.DECISIONS)
    check("[typed] the decision log is the offline temp store, never logs/",
          os.path.abspath(T.DECISIONS).startswith(os.path.abspath(_TMP)))

    # --- facts: build intent a NOUL (lettered pair, two orders), phase a
    # CHOICE; no content-free read; every decision logged with join keys
    def want(state, opts):
        if "yes" in opts:
            return "yes"
        if "implementing the work" in opts:
            return "implementing the work"
        return None
    f = TFake(want)
    released.clear()
    with T.Turn(user_turn("Add a pause menu to the game."), post=f,
                upstream=upstream, count=len, on=True, key="convT",
                request="reqT1", account="acct1") as t:
        facts = t.facts()
        rec = t.record()
    bi = [b for b in f.bodies if "built, made or changed" in
          b["messages"][2]["content"]]
    pairs = [[x for x in printed_options(b["messages"])] for b in bi]
    check("[typed] build intent is a noul read as a LETTERED PAIR in two "
          "orders: A=yes/B=no, then A=no/B=yes",
          pairs == [[("A", "yes"), ("B", "no")], [("A", "no"),
                                                 ("B", "yes")]], pairs)
    check("[typed] the noul's rendering is byte for byte the measured "
          "mc_avg form (decider_bonsai.as_choice)",
          [b["messages"][2]["content"] for b in bi] ==
          [D.question_text(D.as_choice(D.yes_no(T.BUILD_INTENT_Q), o))
           for o in ([0, 1], [1, 0])])
    check("[typed] no content-free read (no label-prior division)",
          all(b["messages"][1]["content"] != D.STATE_HEAD + D.NEUTRAL_STATE
              for b in f.bodies))
    check("[typed] facts: the decider answers, p(yes) and the decision id "
          "recorded", facts["build_intent"]["value"] is True
          and facts["build_intent"]["source"] == "decider"
          and facts["build_intent"]["p"]["yes"] > 0.85
          and facts["phase"]["value"] == "implement"
          and facts["build_intent"]["decision_id"]
          in rec["decisions"], facts)
    check("[typed] the record names the readout and every mode",
          rec["readout"] == D.READOUT_VERSION and rec["form"] == T.MODES
          and len(rec["decisions"]) == 2)
    import slots as _slots
    want_rel = [] if _slots.lane_kept() else [3]
    check("[typed] one slot for the turn, released once (none when the "
          "lane is kept: slots LAYOUT V2)", released == want_rel
          and {b.get("id_slot") for b in f.bodies} == {3}, released)
    rows = decisions_rows()
    r0 = next((r for r in rows if r["question"]["name"] == "build_intent"),
              {})
    check("[typed] every decision is logged -- agreements too -- with its "
          "join keys", len(rows) == 2 and r0.get("conversation") == "convT"
          and r0.get("request") == "reqT1" and r0.get("account") == "acct1"
          and r0["question"]["type"] == "noul"
          and len(r0["orders"]) == 2 and r0["pick"] == "true"
          and r0["rule"] is True and r0["state"]["sha"]
          and r0["readout"] == D.READOUT_VERSION, r0)
    check("[typed] each order's distribution and printed order are logged",
          r0["orders"][0]["printed"] == ["true", "false"]
          and r0["orders"][1]["printed"] == ["false", "true"]
          and abs(r0["orders"][0]["p"]["true"] - 0.9) < 1e-6)
    check("[typed] the log holds ids and labels, never the text",
          all("pause menu" not in json.dumps(r) for r in rows))

    # --- a letter-biased model: the two orders cancel it and DISAGREE
    fb = TFake(letter="A")
    out = T.decide("some state", [D.q_noul("n1", "Is it so?")],
                   post=fb, upstream=upstream, on=True, request="reqB")
    a = out["answers"]["n1"]
    dg = a["diagnostics"]
    check("[noul] a model that always says A: the orders cancel (noul 0.5, "
          "a tie in the diagnostics) and disagree (TV 0.6)",
          dg["tie"] and abs(a["noul"] - 0.5) < 1e-6
          and abs(dg["disagreement"] - 0.6) < 1e-6
          and dg["argmax_agree"] is False, a)
    check("[api] the door answers in Jev's response shape: answers keyed by "
          "name, the noul a single number, usage and model",
          set(out["answers"]) == {"n1"} and a["type"] == "noul"
          and "confidence" not in a and out["usage"]["output_tokens"] == 2
          and "model" in out, out)
    check("[tiers] nothing tuned: the tier is 'untuned' and nothing in the "
          "answer is an abstain", a["tier"]["tier"] == "untuned"
          and "abstain" not in json.dumps(a) and T.THRESHOLDS == {},
          a.get("tier"))

    # --- THE QUESTION SET'S TIERS (a FIXTURE here: nothing is tuned in the
    # repo; the owner accepts a row from bench/decider/tune.py)
    model = D.model_name()
    T.THRESHOLDS = {model: {"n1": {"act_yes": 0.9, "act_no": 0.1,
                                   "caution_yes": 0.7, "caution_no": 0.3},
                            "c": {"high": 0.8, "medium": 0.5}},
                    "other-model": {"n3": {"act_yes": 0.0}}}
    try:
        fa = TFake(want=lambda s, o: "yes", p=0.95)
        fm = TFake(want=lambda s, o: "yes", p=0.8)
        out = T.decide("s", [D.q_noul("n1", "Q?"), D.q_noul("n3", "Q?")],
                       post=fb, upstream=upstream, on=True)
        a1, a3 = out["answers"]["n1"], out["answers"]["n3"]
        hi = T.decide("s", [D.q_noul("n1", "Q?")], post=fa,
                      upstream=upstream, on=True)["answers"]["n1"]
        md = T.decide("s", [D.q_noul("n1", "Q?")], post=fm,
                      upstream=upstream, on=True)["answers"]["n1"]
        check("[tiers] a noul's tier is read on the noul itself: 0.5 is "
              "low, 0.95 high, 0.8 medium (this model's row)",
              a1["tier"]["tier"] == "low" and hi["tier"]["tier"] == "high"
              and md["tier"]["tier"] == "medium"
              and a1["tier"]["question_set"] == "n1",
              (a1["tier"], hi["tier"], md["tier"]))
        check("[tiers] thresholds are per MODEL: another model's row does not "
              "apply", a3["tier"]["tier"] == "untuned", a3["tier"])
        fc3 = TFake(want=lambda s, o: "y", p=0.6)
        c = T.decide("s", [D.q_choice("c", "Which?", ["x", "y", "z"])],
                     post=fc3, upstream=upstream, on=True)["answers"]["c"]
        check("[tiers] a choice's tier is read on Jev's confidence (0.6 of 3 "
              "-> 0.4: low)", abs(c["confidence"] - 0.4) < 1e-6
              and c["tier"]["tier"] == "low", c)
        rows = [r for r in decisions_rows() if r.get("id")
                in (a1["decision_id"], c["decision_id"])]
        check("[tiers] a low tier is logged as no pick, with the tier and "
              "the confidence", len(rows) == 2 and all(
                  r["pick"] is None and r["tier"] == "low" for r in rows)
              and any(abs((r.get("confidence") or 0) - 0.4) < 1e-6
                      for r in rows), rows)
        with T.Turn(user_turn("Add a pause menu to the game."),
                    post=TFake(want=lambda s, o: "yes" if "yes" in o
                               else None), upstream=upstream, count=len,
                    on=True) as t:
            T.THRESHOLDS[model]["build_intent"] = {
                "act_yes": 0.99, "act_no": 0.01, "caution_yes": 0.95,
                "caution_no": 0.05}
            T.THRESHOLDS[model]["phase"] = {"high": 0.9, "medium": 0.95}
            ff = t.facts()
        check("[tiers] facts: a LOW tier falls back to the rule and says "
              "so; the tier is recorded",
              ff["build_intent"]["source"] == "low->rule"
              and ff["build_intent"]["tier"] == "low"
              and ff["phase"]["source"] in ("low->rule", "tie->rule"), ff)
        bi = T.build_intent(user_turn("Please build a todo app."),
                            post=TFake(want=lambda s, o: "yes", p=0.9),
                            upstream=upstream, count=len, on=True)
        check("[tiers] build_intent: a low tier returns no value (the "
              "caller's rule answers)", bi["value"] is None
              and bi["source"] == "low" and bi["tier"] == "low", bi)
    finally:
        T.THRESHOLDS = {}
    fm = TFake(want=lambda s, o: "yes", mass=0.3)
    out = T.decide("s", [D.q_noul("n2", "Q?")], post=fm, upstream=upstream,
                   on=True)
    a2 = out["answers"]["n2"]
    check("[diagnostics] label mass: the labels' share is reported "
          "(label_mass_min 0.3), never acted on by the decider",
          abs(a2["diagnostics"]["label_mass_min"] - 0.3) < 1e-6
          and a2["tier"]["tier"] == "untuned", a2)
    fa = TFake(want=lambda s, o: "yes", p=0.7)
    out = T.decide("s", [D.q_noul("n9", "Q?")], post=fa, upstream=upstream,
                   on=True)
    check("[noul] agreeing orders: noul 0.7, TV 0",
          abs(out["answers"]["n9"]["noul"] - 0.7) < 1e-6
          and out["answers"]["n9"]["diagnostics"]["disagreement"] == 0)

    # --- score: a distribution over the levels 0..k and its score
    levels = ["not at all", "partly", "mostly", "completely"]
    fs = TFake(want=lambda s, o: "mostly", p=0.7)
    out = T.decide("s", [{"type": "score", "name": "done", "instructions":
                          "How finished is the work?", "criteria": levels}],
                   post=fs, upstream=upstream, on=True)
    sc = out["answers"]["done"]
    exp = 2 * 0.7 + (0 + 1 + 3) * 0.1
    check("[score] Jev's form in, Jev's shape out: levels read ascending "
          "then descending; score = sum of level number x probability; "
          "legend; confidence",
          sc["type"] == "score" and set(sc["probabilities"]) ==
          {"0", "1", "2", "3"} and abs(sc["score"] - exp) < 1e-6
          and sc["legend"]["2"] == "mostly"
          and abs(sc["confidence"] - (4 * 0.7 - 1) / 3) < 1e-6
          and [o["printed"] for o in sc["diagnostics"]["orders"]] ==
          [["0", "1", "2", "3"], ["3", "2", "1", "0"]], sc)
    # --- choice with keys and a none option: none in the MIDDLE of both
    fc = TFake(want=lambda s, o: "bash: runs a command")
    out = T.decide("s", [D.q_choice(
        "tool", "Which tool runs it?", ["read: reads", "bash: runs a "
                                        "command", "write: writes",
                                        "None of these."],
        keys=["read", "bash", "write", "none"], none=3)], post=fc,
        upstream=upstream, on=True)
    ch = out["answers"]["tool"]
    check("[choice] choice is the winning key, a probability for every "
          "option, the confidence from their spread; none never printed "
          "last in either order", ch["choice"] == "bash"
          and set(ch["probabilities"]) == {"read", "bash", "write", "none"}
          and abs(ch["confidence"] - (4 * 0.9 - 1) / 3) < 1e-5
          and all(o["printed"][-1] != "none"
                  for o in ch["diagnostics"]["orders"]), ch)
    n_before = len(fc.bodies)
    out = T.decide("s", [{"type": "vote", "name": "x", "text": "?"}],
                   post=fc, upstream=upstream, on=True)
    check("[api] a malformed question is refused before anything is sent, "
          "with the failure's facts", out["answers"] == {}
          and out["failure"]["code"] == "BAD_TYPE"
          and len(fc.bodies) == n_before, out.get("failure"))
    out = T.decide("s", [D.q_noul("x", "Q?"), D.q_noul("x", "R?")],
                   post=fc, upstream=upstream, on=True)
    check("[api] a repeated question name is refused (answers are keyed by "
          "it)", out["answers"] == {}
          and out["failure"]["code"] == "DUPLICATE_NAME"
          and len(fc.bodies) == n_before, out.get("failure"))
    out = T.decide("s", [D.q_noul("x", "Q?")], post=fc, upstream=upstream)
    check("[api] off unless the serving process enabled it", not out["on"]
          and out["answers"] == {} and "off" in out["why"])

    # --- choose(): two orders, positional letters, NO label prior; the
    # letter bias that the legacy prior was built for cancels instead
    opts2 = [{"letter": "A", "id": "s1", "name": "s1", "text": "wrote it"},
             {"letter": "B", "id": "s2", "name": "s2", "text": "other"},
             {"letter": "C", "id": None, "name": None,
              "text": "None of these"}]
    fb = TFake(letter="A")
    with T.Turn(step("cat > /tmp/x.js", "ok"), post=fb, upstream=upstream,
                count=len, on=True) as t:
        lab, dist = T.choose("any", opts2, {"kind": "match", "exclude": [],
                                            "question": T.FIT_Q})
        rec = [x for x in t.answers if x.get("name") == "choose:any"][0]
    check("[choose] no label-prior read (no blanked option set)",
          len(fb.bodies) == 2 and all(
              not b["messages"][2]["content"].endswith(D.NEUTRAL_STATE)
              for b in fb.bodies))
    check("[choose] a letter-A bias is cancelled by positional re-lettering "
          "in two orders: s1 and s2 tie, no label",
          lab is None and rec["tie"] and abs(dist["A"] - dist["B"]) < 1e-6
          and rec["calibration"].startswith("none"), (lab, dist, rec))
    check("[choose] 'None of these' is printed in the middle of both orders",
          [printed_options(b["messages"])[1][1] for b in fb.bodies]
          == ["None of these"] * 2)

    # --- choose() rounds EXTEND the slot per order, the question's letter
    # references rewritten into each order's letters
    opts = [{"letter": L, "id": i, "name": i, "text": x} for L, i, x in (
        ("A", "s1", "Koota traits"), ("B", "s2", "Koota with React"),
        ("C", "s3", "noise"))] + [{"letter": "D", "id": None, "name": None,
                                   "text": "None of these"}]
    order_pref = ["Koota with React", "Koota traits", "None of these"]

    def w(state, texts):
        m = fr.bodies[-1]["messages"]
        taken = [x["content"] for x in m if x["role"] == "assistant"][:-1]
        return order_pref[len(taken)]
    fr = TFake(want=w)
    with T.Turn(user_turn("wire koota into react"), post=fr,
                upstream=upstream, count=len, on=True) as t:
        l1, d1 = T.choose("koota", opts, {"kind": "package", "exclude": [],
                                          "question": T.FIT_Q})
        q2 = T.FIT_Q + " Already chosen: B; choose another, or D (none of " \
            "these)."
        l2, d2 = T.choose("koota", opts, {"kind": "package",
                                          "exclude": ["B"], "question": q2})
    check("[choose] round 1 picks B, round 2 (B excluded) picks A; the "
          "distribution is in the caller's letters, B 0 in round 2",
          l1 == "B" and l2 == "A" and d2["B"] == 0
          and set(d2) == {"A", "B", "C", "D"}, (l1, l2, d2))
    b1 = [b["messages"] for b in fr.bodies[:2]]
    b2 = [b["messages"] for b in fr.bodies[2:4]]
    same_order = all(b2[i][:3] == b1[i][:3] for i in range(2))
    lab_of = []
    for i in range(2):
        m = dict((x, lab) for lab, x in printed_options(b1[i]))
        lab_of.append(m)
    check("[choose] per order, round 2 carries round 1's question and that "
          "order's letter for the answer taken",
          same_order and all(
              b2[i][4]["content"] == D.ANSWER_LEAD + " "
              + lab_of[i]["Koota with React"] for i in range(2)), b2[0][4])
    check("[choose] the round's letter references are rewritten into each "
          "order's letters",
          all(b2[i][5]["content"] == "QUESTION: " + T.FIT_Q
              + f" Already chosen: {lab_of[i]['Koota with React']}; choose "
              f"another, or {lab_of[i]['None of these']} (none of these)."
              + T.CHOOSE_TAIL for i in range(2)),
          [b2[i][5]["content"] for i in range(2)])
    import jinja2
    env = jinja2.Environment()
    env.filters["tojson"] = lambda v, **k: json.dumps(v, ensure_ascii=False)
    tpl = env.from_string(open(os.path.join(
        HERE, "fixtures", "bonsai_chat_template.jinja"),
        encoding="utf-8").read())

    def served(msgs):
        o = tpl.render(messages=msgs, add_generation_prompt=False,
                       enable_thinking=False)
        tail = "<|im_end|>\n"
        return o[:-len(tail)] if o.endswith(tail) else o
    check("[choose] each order's round 2 extends its round 1 and the "
          "answer taken (served template)",
          all(served(b2[i]).startswith(served(b1[i]) + " "
                                       + lab_of[i]["Koota with React"])
              for i in range(2)))

    # --- judge_stop: the typed choice, two reads, no prior read
    T.TEMPLATE_CF.clear()
    released.clear()
    fj = TFake(want=lambda s, o: (T.STOP_OPTIONS[1][1]
                                  if "let me" in s.lower()
                                  else T.STOP_OPTIONS[0][1]))
    j = T.judge_stop("Now let me understand the task.", "thinking",
                     post=fj, upstream=upstream, count=len, on=True,
                     request="reqJ", key="convJ")
    check("[judge_stop] typed: two reads, no label-prior read, picks "
          "next_step, with its decision id and diagnostics",
          j["judged"] and j["pick"] == "next_step" and len(fj.bodies) == 2
          and j["readout"] == D.READOUT_VERSION and j["decision_id"]
          and j["disagreement"] == 0 and j["tier"] == "untuned"
          and abs(j["confidence"] - (4 * 0.9 - 1) / 3) < 1e-5
          and set(j["distribution"]) == {"finished", "next_step", "asking",
                                         "other"}
          and released == ([] if _slots.lane_kept() else [3]), j)
    check("[judge_stop] 'other' (none) is in the middle of both orders",
          all(o[-1] != "other" and o[0] != "other" for o in j["orders"]),
          j.get("orders"))

    # --- the corpus join
    rows = [r for r in decisions_rows() if r.get("request") == "reqJ"]
    res = T.join_corpus("reqJ", "corpusTurn01", account="acctJ",
                        conversation="convJ")
    joins = [r for r in decisions_rows() if r.get("row") == "join"]
    check("[join] join_corpus names every decision of the request made "
          "before it", res["joined"] == 1 and joins
          and joins[-1]["corpus_turn"] == "corpusTurn01"
          and joins[-1]["decisions"] == [rows[0]["id"]], joins)
    j2 = T.judge_stop("Done.", "", post=fj, upstream=upstream, count=len,
                      on=True, request="reqJ")
    late = [r for r in decisions_rows() if r.get("id") == j2["decision_id"]]
    check("[join] a decision made after the join carries corpus_turn itself",
          late and late[0]["corpus_turn"] == "corpusTurn01")
    check("[join] no request key or no corpus turn: nothing written",
          T.join_corpus(None, "x")["joined"] == 0
          and T.join_corpus("reqNone", None)["joined"] == 0)
    check("[join] the choose question's options are logged as ids",
          late and late[0]["option_ids"] == ["finished", "next_step",
                                             "asking", "none"])

    # --- the wrappers
    fw = TFake(want=lambda s, o: "yes")
    bi = T.build_intent(user_turn("Please build a todo app."), post=fw,
                        upstream=upstream, count=len, on=True,
                        request="reqW")
    check("[wrapper] build_intent: the noul, value and decision id",
          bi["value"] is True and bi["source"] == "decider"
          and bi["decision_id"] and len(fw.bodies) == 2, bi)
    wrow = [r for r in decisions_rows() if r.get("id") == bi["decision_id"]]
    check("[wrapper] build_intent logs the rule's answer beside it",
          wrow and wrow[0]["rule"] is True)
    fp = TFake(want=lambda s, o: None if s == D.NEUTRAL_STATE
               else ("no" if "no" in o else None))
    with T.Turn(user_turn("useQuery from @tanstack/react-query refetches"),
                post=fp, upstream=upstream, count=len, on=True) as t:
        t._facts = {}
        cp = t.confirm_packages(
            {"koota": {"why": [{"how": "symbol", "what": "useQuery",
                                "where": "user"}]}},
            {"koota": "an ECS library"})
    na = [i for i, b in enumerate(fp.bodies) if b["messages"][1]["content"]
          == D.STATE_HEAD + D.NEUTRAL_STATE]
    check("[wrapper] a weak package detection: a noul WITH the content-"
          "free prior (the measured veto), read before the state",
          cp["koota"]["source"] == "decider" and cp["koota"]["keep"] is False
          and na == [0, 1] and len(fp.bodies) == 4, (cp, na))
    cands = [{"name": "koota-traits", "description": "Koota traits"},
             {"name": "koota-react", "description": "Koota with React"},
             {"name": "math-noise", "description": "noise"}]
    order = ["Koota with React", "Koota traits"]

    def fit(state, texts):
        return next((o for o in order if o in texts), T.NONE_OPTION)
    with T.Turn(user_turn("wire koota into react"), post=TFake(fit),
                upstream=upstream, count=len, on=True) as t:
        r = t.pick("koota", cands)
        r2 = t.pick("koota", cands[:2], hard_evidence=True)
    check("[wrapper] pick(): ordered picks until none wins; hard evidence "
          "leaves none out", r["picks"] == ["koota-react", "koota-traits"]
          and r["rounds"][-1]["pick"] == "none"
          and "none" not in r2["rounds"][0]["options"]
          and r2["picks"] == ["koota-react", "koota-traits"], (r, r2))

    # --- the legacy switch: the old forms, logged on the same log
    T.READOUT = "legacy"
    T.TEMPLATE_CF.clear()
    fl = TFake(want=lambda s, o: "Koota with React")
    with T.Turn(user_turn("wire koota into react"), post=fl,
                upstream=upstream, count=len, on=True) as t:
        T.choose("koota", opts, {"kind": "package", "exclude": [],
                                 "question": T.FIT_Q})
        rec = t.record()
    blank = [b for b in fl.bodies
             if b["messages"][2]["content"].endswith(D.NEUTRAL_STATE)]
    lrow = decisions_rows()[-1]
    check("[legacy] YAMADORI_DECIDER_READOUT=legacy: one rotation plus the "
          "label-prior read, as before", len(fl.bodies) == 2
          and len(blank) == 1 and rec["readout"] == "legacy"
          and rec["form"] == T.FORM)
    check("[legacy] its decisions go to the same log, marked legacy",
          lrow["readout"] == "legacy/rotated_idprior"
          and lrow["question"]["name"] == "choose:koota")
    with T.Turn(user_turn("Add a pause menu."), post=TFake(want),
                upstream=upstream, count=len, on=True) as t:
        t.facts()
    lr = decisions_rows()[-2:]
    check("[legacy] legacy facts are logged in the typed keys (true/false, "
          "phase names), so the two readouts compare on one log",
          [r["readout"] for r in lr] == ["legacy/mc_avg"] * 2
          and lr[0]["pick"] == "true" and set(lr[1]["p"]) == set(T.PHASES)
          and lr[1]["pick"] == "implement", lr)
    T.READOUT = "typed"
    th = T._PRIME_THREAD[:]
    T.enable(True, prime=True)
    started = T._PRIME_THREAD[:] != th
    pf = T.prime_for("bonsai")
    T.enable(False)
    check("[typed] nothing to prime: enable() and prime_for start no "
          "label-prior thread", not started and pf is None)


def _phase_fake(p):
    """The phase choice favours "implementing the work" at p (the other
    three share the rest); build intent says yes."""
    def want(state, opts):
        if "yes" in opts:
            return "yes"
        if "implementing the work" in opts:
            return "implementing the work"
        return None
    return TFake(want, p=p)


def test_per_model() -> None:
    """PER MODEL (decider_bonsai THE MODEL PROFILE): the Turn reads the
    SERVING model's tie band and thresholds and reports its profile; Bonsai
    unchanged; flash-next without a record says it is unmeasured; a record
    measured for flash-next applies to flash-next only."""
    import model as M
    T.READOUT = "typed"
    D._PROFILE_CACHE.clear()
    if os.path.exists(T.DECISIONS):
        os.remove(T.DECISIONS)

    def turn(fake):
        with T.Turn(user_turn("Add a pause menu to the game."), post=fake,
                    upstream=upstream, count=len, on=True,
                    request="reqPM") as t:
            facts = t.facts()
            rec = t.record()
        return facts, rec
    facts, rec = turn(_phase_fake(0.4))
    mp = rec["model_profile"]
    check("[per-model] bonsai: the record's tie band is Bonsai's 0.0034, "
          "measured, and the profile names what is not measured",
          rec["tie_band"] == 0.0034 and mp["model"] == "bonsai"
          and mp["tie_band"]["measured"] is True
          and "letter_prior" in mp["unmeasured"], mp)
    check("[per-model] bonsai: a 0.2 gap decides (unchanged)",
          facts["phase"]["source"] == "decider"
          and facts["phase"]["value"] == "implement", facts["phase"])
    rows = [r for r in decisions_rows() if r.get("request") == "reqPM"]
    check("[per-model] every decision row carries its model and the band it "
          "was read at", rows and all(
              r["model"] == "bonsai" and r["tie_band"] == {
                  "value": 0.0034, "measured": True} for r in rows), rows)
    old = M.MODEL
    try:
        M.MODEL = "flash-next"
        facts, rec = turn(_phase_fake(0.4))
        mp = rec["model_profile"]
        check("[per-model] flash-next, no record: Bonsai's band applied and "
              "REPORTED as unmeasured for flash-next (nothing invented)",
              rec["tie_band"] == 0.0034 and mp["model"] == "flash-next"
              and mp["tie_band"]["measured"] is False
              and mp["tie_band"]["source"].startswith(
                  "unmeasured for flash-next")
              and mp["unmeasured"] == list(D.PROFILE_FIELDS), mp)
        rows = [r for r in decisions_rows() if r.get("request") == "reqPM"
                and r["model"] == "flash-next"]
        check("[per-model] flash-next's decision rows say the band is "
              "unmeasured", rows and all(r["tie_band"]["measured"] is False
                                         for r in rows), rows)
        os.makedirs(PROFILES, exist_ok=True)
        with open(os.path.join(PROFILES, "flash-next.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"version": D.PROFILE_VERSION, "model": "flash-next",
                       "tie_band": {"value": 0.25, "n": 100,
                                    "date": "2026-10-01",
                                    "script": "bench/decider/"
                                              "measure_model.py"}}, f)
        D._PROFILE_CACHE.clear()
        facts, rec = turn(_phase_fake(0.4))
        check("[per-model] flash-next WITH a measured band (a fixture: "
              "0.25): the same 0.2 gap is a tie there -> the rule, and the "
              "record says measured", facts["phase"]["source"] == "tie->rule"
              and rec["tie_band"] == 0.25
              and rec["model_profile"]["tie_band"]["measured"] is True,
              (facts["phase"], rec["tie_band"]))
        T.THRESHOLDS = {"bonsai": {"n1": {"act_yes": 0.99, "act_no": 0.01,
                                          "caution_yes": 0.98,
                                          "caution_no": 0.02}}}
        a = T.decide("s", [D.q_noul("n1", "Q?")],
                     post=TFake(want=lambda s, o: "yes", p=0.9),
                     upstream=upstream, on=True)["answers"]["n1"]
        check("[per-model] thresholds: bonsai's row never applies to "
              "flash-next (untuned there)", a["tier"]["tier"] == "untuned"
              and a["tier"]["model"] == "flash-next", a["tier"])
        M.MODEL = "bonsai-agent"
        a = T.decide("s", [D.q_noul("n1", "Q?")],
                     post=TFake(want=lambda s, o: "yes", p=0.9),
                     upstream=upstream, on=True)["answers"]["n1"]
        check("[per-model] thresholds: bonsai-agent (the same process) reads "
              "bonsai's row", a["tier"]["tier"] == "low", a["tier"])
    finally:
        M.MODEL = old
        T.THRESHOLDS = {}
        for n in (os.listdir(PROFILES) if os.path.isdir(PROFILES) else []):
            os.remove(os.path.join(PROFILES, n))
        D._PROFILE_CACHE.clear()


def main() -> int:
    released = []
    D.release = lambda slot, why="": released.append(slot) or {  # noqa: E731
        "released": True, "slot": slot}
    T.TEMPLATE_CF.clear()

    # --- the state
    s, info = T.state_of(user_turn("x" * 30000), count=lambda t: len(t) // 3)
    check("a long user turn is cut head+tail to STATE_TOKENS",
          info["cut"] and s.startswith("x") and "middle left out" in s
          and len(s) < 30000)
    s, info = T.state_of(step("ls node_modules/koota", "dist\nreact"))
    check("an agent step's state is the call and its result only",
          info["kind"] == "step" and "terminal" in s and "dist" in s
          and "build the game" not in s)

    # --- a user turn: build intent + phase; decider says yes/implementing
    def want(state, opts):
        if "yes" in opts:
            return "yes"
        if "implementing the work" in opts:
            return "implementing the work"
        return None
    f = Fake(want)
    with T.Turn(user_turn("Add a pause menu."), post=f, upstream=upstream,
                count=len, on=True) as t:
        facts = t.facts()
    check("user turn asks build_intent and phase",
          set(facts) == {"build_intent", "phase"})
    check("decider answers", facts["build_intent"]["value"] is True
          and facts["build_intent"]["source"] == "decider"
          and facts["phase"]["value"] == "implement"
          and facts["phase"]["source"] == "decider")
    na = [b for b in f.bodies if b["messages"][1]["content"]
          == D.STATE_HEAD + D.NEUTRAL_STATE]
    first_state = next(i for i, b in enumerate(f.bodies)
                       if b["messages"][1]["content"]
                       != D.STATE_HEAD + D.NEUTRAL_STATE)
    check("build_intent and phase read without content-free calibration "
          "(FORM: it hurt them)", not na and first_state == 0)
    q_orders = {}
    for b in f.bodies:
        q = b["messages"][2]["content"]
        q_orders.setdefault(q.split("\n")[0], set()).add(q)
    check("every question is asked in two option orders",
          all(len(v) == 2 for v in q_orders.values()))
    check("yes/no asked as a neutral-letter choice",
          all("OPTIONS:" in b["messages"][2]["content"] for b in f.bodies))
    import slots as _slots
    check("one slot for the turn, released once (none when the lane is kept: "
          "slots LAYOUT V2)",
          released == ([] if _slots.lane_kept() else [3])
          and {b.get("id_slot") for b in f.bodies} == {3}, released)
    n_before = len(f.bodies)
    with T.Turn(user_turn("Add a pause menu."), post=f, upstream=upstream,
                count=len, on=True) as t:
        t.facts()
    check("content-free reads are made once per template per process",
          all(b["messages"][1]["content"] != D.STATE_HEAD + D.NEUTRAL_STATE
              for b in f.bodies[n_before:]))

    # --- disagreement logged, ids and labels only
    if os.path.exists(LOG):
        os.remove(LOG)
    with T.Turn(user_turn("Explain the collision code you wrote."), post=f,
                upstream=upstream, count=len, on=True, key="conv1",
                request="req9") as t:
        facts = t.facts()
    rows = [json.loads(x) for x in open(LOG, encoding="utf-8")] \
        if os.path.exists(LOG) else []
    check("a decider-vs-rule disagreement is logged",
          any(r["question"] == "build_intent" and r["decider"] is True
              and r["rule"] is False for r in rows))
    check("the log carries ids and labels, never the text",
          rows and all("collision" not in json.dumps(r) for r in rows)
          and rows[0]["conversation"] == "conv1")

    # --- a tie: build intent stays the decider's (operator: the rule is
    # the fallback only when the decider is unavailable); phase falls back
    with T.Turn(user_turn("Add a pause menu."), post=Fake(lambda s, o: None),
                upstream=upstream, count=len, on=True) as t:
        facts = t.facts()
    check("a build-intent tie stays the decider's, recorded as a tie",
          facts["build_intent"]["source"] == "decider (tie)")
    check("a phase tie falls back to the rule",
          facts["phase"]["source"] == "tie->rule")

    # --- the previous assistant turn's tail on a user turn's state
    conv = [{"role": "system", "content": "sys"},
            {"role": "user", "content": "make a game"},
            {"role": "assistant", "content": "x" * 3000 + " Shall I add "
             "the pause menu next?"},
            {"role": "user", "content": "Go ahead."}]
    st, info = T.state_of(conv, count=lambda t: len(t) // 3)
    check("user state = previous assistant tail (bounded) + the user turn",
          st.startswith("Assistant (previous turn): ...")
          and st.endswith("User: Go ahead.")
          and "pause menu next?" in st
          and info["previous_assistant"]["cut"]
          and info["previous_assistant"]["tokens"] == T.PREV_TOKENS)
    seen_rule = []
    import route
    orig = route.work_intent
    route.work_intent = lambda t: seen_rule.append(t) or orig(t)
    try:
        with T.Turn(conv, post=down_fn(), upstream=upstream, count=len,
                    on=True) as t:
            facts = t.facts()
    finally:
        route.work_intent = orig
    check("unavailable: the rule answers build intent from the user's text "
          "alone", facts["build_intent"]["source"] == "rule"
          and seen_rule == ["Go ahead."])

    # --- unavailable: every question falls back
    def down(body, timeout):
        raise ConnectionRefusedError("refused")
    with T.Turn(user_turn("Add a pause menu."), post=down, upstream=upstream,
                count=len, on=True) as t:
        T.TEMPLATE_CF.clear()
        facts = t.facts()
        rec = t.record()
    check("unavailable: rules answer, the failure is recorded",
          facts["build_intent"]["source"] == "rule"
          and rec["failure"]["code"] == "MODEL_UNREACHABLE"
          and rec["failure"]["retryable"])

    # --- an agent step: phase, reads_package, scratch (terminal)
    f2 = Fake(lambda s, o: "yes" if "yes" in o else (
        "verifying or testing the work"
        if "verifying or testing the work" in o else None))
    with T.Turn(step("cat > /tmp/t.ts << 'EOF'\nx\nEOF\nnode /tmp/t.ts",
                     "ok"), post=f2, upstream=upstream, count=len, on=True) as t:
        facts = t.facts()
    check("a terminal step asks phase only; reads_package is the rule's "
          "path fact; scratch_write is not asked (the selector's rule)",
          set(facts) == {"phase", "reads_package"}
          and facts["reads_package"]["source"] == "rule (a path fact)"
          and facts["phase"]["source"] == "decider")

    # --- pick: ordered, none ends it; hard evidence leaves none out
    cands = [{"name": "koota-traits", "description": "Koota traits"},
             {"name": "koota-react", "description": "Koota with React"},
             {"name": "math-noise", "description": "noise"}]
    order = ["Koota with React", "Koota traits"]

    def fit(state, opts):
        for o in order:
            if o in opts:
                return o
        return T.NONE_OPTION if T.NONE_OPTION in opts else None
    T.TEMPLATE_CF.clear()
    with T.Turn(user_turn("wire koota into react"), post=Fake(fit),
                upstream=upstream, count=len, on=True) as t:
        r = t.pick("koota", cands)
        r2 = t.pick("koota", cands[:2], hard_evidence=True)
    check("pick: ordered picks until none wins",
          r["picks"] == ["koota-react", "koota-traits"]
          and r["rounds"][-1]["pick"] == "none")
    check("pick: none is an option unless hard evidence",
          "none" in r["rounds"][0]["options"]
          and "none" not in r2["rounds"][0]["options"]
          and r2["picks"] == ["koota-react", "koota-traits"])
    # --- choose(): skill_match's contract, through the module attribute.
    # THE LEGACY READOUT from here to judge_stop (YAMADORI_DECIDER_READOUT=
    # legacy): the rotation and the divided-out label prior, kept for
    # comparison. The typed readout's choose() is tested in test_typed().
    T.READOUT = "legacy"
    IDS.update({"AA": 1001, " AA": 1002, "BB": 1003, " BB": 1004})
    opts = [{"letter": L, "id": i, "name": i, "text": t} for L, i, t in (
        ("A", "s1", "Koota traits"), ("B", "s2", "Koota with React"),
        ("C", "s3", "noise"))] + [{"letter": "D", "id": None, "name": None,
                                   "text": "None of these"}]
    seen = []

    def cfake(body, timeout):
        seen.append(body)
        m = body["messages"]
        state = m[1]["content"][len(D.STATE_HEAD):]
        lines = m[2]["content"].split("OPTIONS:\n")[1].splitlines()
        labs = [ln.split(") ", 1) for ln in lines]
        fav = None if state == D.NEUTRAL_STATE else "Koota with React"
        top = [{"id": IDS[" " + lab], "logprob": math.log(
            (0.7 if t == fav else 0.1) if fav else 1 / len(labs))}
            for lab, t in labs]
        top.sort(key=lambda t: -t["logprob"])
        return {"choices": [{"logprobs": {"content": [
            {"id": top[0]["id"], "logprob": top[0]["logprob"],
             "top_logprobs": top}]}}], "usage": {"prompt_tokens": 90},
            "timings": {"prompt_n": 20, "cache_n": 70, "prompt_ms": 5.0}}
    T.TEMPLATE_CF.clear()
    check("no open turn: decide_turn has no `choose` (the selector's stub "
          "answers)", not hasattr(T, "choose"))
    with T.Turn(user_turn("wire koota into react"), post=cfake,
                upstream=upstream, count=len, on=True) as t:
        ch = getattr(T, "choose", None)
        lab, dist = ch("koota", opts, {"kind": "package", "exclude": [],
                                       "question": T.FIT_Q})
        lab2, dist2 = ch("koota", opts, {"kind": "package",
                                         "exclude": ["B"],
                                         "question": T.FIT_Q + " Already "
                                         "chosen: B; choose another, or D "
                                         "(none of these)."})
    check("choose returns (label, {label: p}) over every option",
          lab == "B" and set(dist) == {"A", "B", "C", "D"}
          and abs(sum(dist.values()) - 1) < 1e-6)
    check("excluded labels get 0 and are never answered",
          dist2["B"] == 0 and lab2 != "B")
    real = [b for b in seen if b["messages"][1]["content"]
            != D.STATE_HEAD + D.NEUTRAL_STATE]
    check("the options sit in their own user message, identical across "
          "rounds; the question comes after them",
          len({b["messages"][2]["content"] for b in real}) == 1
          and [x["role"] for x in real[0]["messages"]] ==
          ["system", "user", "user", "user", "assistant"]
          and real[0]["messages"][3]["content"].startswith("QUESTION: "))
    check("round 2 carries round 1's question and answer before its own",
          [x["role"] for x in real[1]["messages"]] ==
          ["system", "user", "user", "user", "assistant", "user",
           "assistant"]
          and real[1]["messages"][4]["content"] == D.ANSWER_LEAD + " B"
          and "Already chosen: B" in real[1]["messages"][5]["content"])
    prior_reads = [b for b in seen if b not in real]
    check("one label-prior read, option texts blanked to N/A",
          len(prior_reads) == 1 and all(
              ln.endswith(") " + D.NEUTRAL_STATE) for ln in
              prior_reads[0]["messages"][2]["content"].splitlines()[1:]))
    printed = [ln.split(")")[0] for ln in
               real[0]["messages"][2]["content"].splitlines()[1:]]
    check("'None of these' is not printed last", printed[-1] != "D"
          and sorted(printed) == ["A", "B", "C", "D"])
    check("the turn closed: `choose` is gone again", not hasattr(T, "choose"))

    # --- ROUNDS EXTEND THE SLOT: two rounds rendered through the SERVED
    # template (mcp/fixtures/bonsai_chat_template.jinja; a final assistant
    # message is a prefill: llama-server drops its end-of-turn, as the live
    # /apply-template showed). Round 2's prompt must extend round 1's plus
    # the label; what it processes is only the new question line.
    import jinja2
    env = jinja2.Environment()
    env.filters["tojson"] = lambda v, **k: json.dumps(v, ensure_ascii=False)
    tpl = env.from_string(open(os.path.join(
        HERE, "fixtures", "bonsai_chat_template.jinja"),
        encoding="utf-8").read())

    def served(msgs):
        out = tpl.render(messages=msgs, add_generation_prompt=False,
                         enable_thinking=False)
        tail = "<|im_end|>\n"
        return out[:-len(tail)] if out.endswith(tail) else out
    r1, r2 = served(real[0]["messages"]), served(real[1]["messages"])
    held = r1 + " B"                       # the slot: prompt + its answer
    common = next((i for i, (x, y) in enumerate(zip(held, r2)) if x != y),
                  min(len(held), len(r2)))
    new_line = real[1]["messages"][5]["content"]
    processed = len(r2) - common
    check("round 2's prompt extends round 1's and its answer (served "
          "template)", r2.startswith(held) and r1.endswith("Answer:"))
    turn_markers = ("<|im_end|>\n<|im_start|>user\n", "<|im_end|>\n"
                    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
                    + D.ANSWER_LEAD)
    check("round 2 processes exactly: end of the answer turn, its question "
          "line, the prefill (%d chars, the line %d)" % (processed,
                                                         len(new_line)),
          r2[common:] == turn_markers[0] + new_line + turn_markers[1])
    # The old rendering (each round REPLACING the last user message)
    # diverged before the question: everything after the options' end.
    old2 = served(real[0]["messages"][:3] + [real[1]["messages"][5],
                                              real[1]["messages"][6]])
    old_common = next(i for i, (x, y) in enumerate(zip(r1, old2)) if x != y)
    check("the old rendering diverged inside the last user message (the "
          "live re-read)", old_common < len(r1) - 4)

    # --- LABEL-PRIOR CALIBRATION: a flat read that favours "A" by its ID
    # (A 0.28 vs None 0.24, the diagnosis's agent step) picks None once the
    # content-free label prior (A favoured) is divided out.
    T.TEMPLATE_CF.clear()
    opts2 = [{"letter": "A", "id": "s1", "name": "s1", "text": "wrote it"},
             {"letter": "B", "id": "s2", "name": "s2", "text": "other"},
             {"letter": "C", "id": None, "name": None,
              "text": "None of these"}]

    def bias(body, timeout):
        m = body["messages"]
        blank = m[2]["content"].endswith(") " + D.NEUTRAL_STATE)
        p = ({"A": 0.5, "B": 0.25, "C": 0.25} if blank
             else {"A": 0.28, "B": 0.20, "C": 0.24})
        top = sorted(({"id": IDS[" " + k], "logprob": math.log(v)}
                      for k, v in p.items()), key=lambda t: -t["logprob"])
        return {"choices": [{"logprobs": {"content": [
            {"id": top[0]["id"], "logprob": top[0]["logprob"],
             "top_logprobs": top}]}}], "usage": {"prompt_tokens": 90},
            "timings": {"prompt_n": 20, "cache_n": 70, "prompt_ms": 5.0}}
    with T.Turn(step("cat > /tmp/x.js", "ok"), post=bias, upstream=upstream,
                count=len, on=True) as t:
        lab3, dist3 = T.choose("any", opts2, {"kind": "match",
                                              "exclude": [],
                                              "question": T.FIT_Q})
        rec3 = [a for a in t.answers if a.get("name") == "choose:any"][0]
    check("the label prior turns a flat A-over-None read into None",
          rec3["raw_pick"] == "A" and lab3 == "C"
          and rec3["calibration"] == "rotated_idprior")

    # --- a package confirmation: mc_avg_cc, content-free reads first
    T.TEMPLATE_CF.clear()
    fp = Fake(lambda s, o: "no" if "no" in o else None)
    with T.Turn(user_turn("useQuery from @tanstack/react-query refetches"),
                post=fp, upstream=upstream, count=len, on=True) as t:
        t._facts = {}
        cp = t.confirm_packages(
            {"koota": {"why": [{"how": "symbol", "what": "useQuery",
                                "where": "user"}]}},
            {"koota": "an ECS library"})
    na = [i for i, b in enumerate(fp.bodies) if b["messages"][1]["content"]
          == D.STATE_HEAD + D.NEUTRAL_STATE]
    check("a weak detection is confirmed (here vetoed) by the decider, "
          "content-free reads first", cp["koota"]["source"] == "decider"
          and cp["koota"]["keep"] is False and na == [0, 1])

    # --- the label-prior prime: idle-gated, one count per Turn, logged
    T.TEMPLATE_CF.clear()
    released.clear()
    pb, polls, lines = [], [], []

    def ppost(body, timeout):
        pb.append(body)
        labs = [ln.split(")")[0] for ln in
                body["messages"][2]["content"].splitlines()[1:]]
        top = [{"id": IDS[" " + lab], "logprob": math.log(1 / len(labs))}
               for lab in labs]
        return {"choices": [{"logprobs": {"content": [
            {"id": top[0]["id"], "logprob": top[0]["logprob"],
             "top_logprobs": top}]}}], "usage": {"prompt_tokens": 30},
            "timings": {"prompt_n": 30, "cache_n": 0, "prompt_ms": 5.0}}
    busy = iter([False, False, True, True, True])   # 2 busy polls, then idle

    def idle():
        polls.append(1)
        return next(busy, True)
    st = T.prime_priors((2, 3), idle=idle, sleep=lambda s: None, post=ppost,
                        upstream=upstream,
                        log=lambda m, **k: lines.append(m))
    import slots as _slots
    check("prime: waits for an idle model, then reads each count once, "
          "one Turn per count (a slot release each, none when the lane is "
          "kept: slots LAYOUT V2)",
          st["state"] == "done" and st["done"] == [2, 3]
          and st["waited_s"] == 2 * T.PRIME_POLL_S and len(pb) == 2
          and released == ([] if _slots.lane_kept() else [3, 3]), released)
    check("prime: the priors are the blanked, rotated option sets",
          all(ln.endswith(") " + D.NEUTRAL_STATE) for b in pb for ln in
              b["messages"][2]["content"].splitlines()[1:])
          and len([k for k in T.TEMPLATE_CF if k.startswith("idprior:")])
          == 2)
    check("prime: one log line with the counts and seconds",
          len(lines) == 1 and "read [2, 3]" in lines[0]
          and " s (waited 2.0 s" in lines[0])
    pb.clear()
    with T.Turn(step("ls", "ok"), post=ppost, upstream=upstream, count=len,
                on=True) as t:
        T.choose("any", [{"letter": "A", "id": "s", "name": "s",
                          "text": "x"},
                         {"letter": "B", "id": None, "name": None,
                          "text": "None of these"}],
                 {"kind": "match", "exclude": [], "question": T.FIT_Q})
    check("a primed count costs choose() no prior read",
          len(pb) == 1 and not pb[0]["messages"][2]["content"].endswith(
              D.NEUTRAL_STATE))
    import threading as _th
    ev = _th.Event()
    ev.set()
    st = T.prime_priors((4,), idle=lambda: False, sleep=lambda s: None,
                        post=ppost, upstream=upstream, stop=ev,
                        log=lambda m, **k: lines.append(m))
    check("prime: stopped while waiting, the lazy read remains (logged)",
          st["state"] == "stopped" and "read lazily" in lines[-1])
    st = T.prime_priors((5,), idle=lambda: True, sleep=lambda s: None,
                        post=down_fn(), upstream=upstream,
                        log=lambda m, **k: lines.append(m))
    check("prime: a failure stops it and says why",
          st["state"] == "failed" and "MODEL_UNREACHABLE" in st["error"])

    # --- judge_stop (proxy CONTINUE A STATED STEP): one rotated, prior-
    # calibrated choice over the stopped turn's text and reasoning tail
    T.TEMPLATE_CF.clear()
    released.clear()
    sb = []

    def spost(body, timeout):
        sb.append(body)
        m = body["messages"]
        state = m[1]["content"][len(D.STATE_HEAD):]
        labs = [ln.split(") ", 1) for ln in
                m[2]["content"].split("OPTIONS:\n")[1].splitlines()]
        fav = None
        if state != D.NEUTRAL_STATE and "N/A" not in m[2]["content"]:
            fav = (T.STOP_OPTIONS[1][1] if "let me" in state.lower()
                   else T.STOP_OPTIONS[0][1])
        top = [{"id": IDS[" " + lab], "logprob": math.log(
            (0.7 if t == fav else 0.1) if fav else 1 / len(labs))}
            for lab, t in labs]
        top.sort(key=lambda t: -t["logprob"])
        return {"choices": [{"logprobs": {"content": [
            {"id": top[0]["id"], "logprob": top[0]["logprob"],
             "top_logprobs": top}]}}], "usage": {"prompt_tokens": 90},
            "timings": {"prompt_n": 20, "cache_n": 70, "prompt_ms": 5.0}}
    j = T.judge_stop("Now let me understand the task.", "r" * 5000,
                     post=spost, upstream=upstream, count=len, on=True)
    real = [b for b in sb if "N/A" not in b["messages"][2]["content"]]
    st = real[0]["messages"][1]["content"] if real else ""
    printed = [ln.split(")")[0] for ln in
               real[0]["messages"][2]["content"].splitlines()[1:]] \
        if real else []
    check("judge_stop: one choice, the label prior read once, picks "
          "next_step", j["judged"] and j["pick"] == "next_step"
          and j["raw_pick"] == "next_step" and len(real) == 1
          and len(sb) == 2 and set(j["distribution"]) ==
          {"finished", "next_step", "asking", "other"})
    check("judge_stop: the state is the reasoning's tail (cut to "
          "PREV_TOKENS, from its front) then the message",
          "Assistant (the end of its thinking): ..." in st
          and st.rstrip().endswith("Assistant (its message): Now let me "
                                   "understand the task.")
          and j["state"]["reasoning"]["cut"]
          and st.count("r") <= T.PREV_TOKENS + 20)
    import slots as _slots
    check("judge_stop: 'None of these' is not printed last; the slot is "
          "released (kept, layout v2: the lane)",
          printed and printed[-1] != "D"
          and released == ([] if _slots.lane_kept() else [3]), released)
    j2 = T.judge_stop("Done. The camera is fixed.", "", post=spost,
                      upstream=upstream, count=len, on=True)
    check("judge_stop: a report of finished work picks finished",
          j2["pick"] == "finished")
    j3 = T.judge_stop("x", "", post=down_fn(), upstream=upstream, count=len,
                      on=True)
    check("judge_stop: unavailable -> judged False with the failure's facts",
          j3["judged"] is False and j3["failure"]["code"]
          == "MODEL_UNREACHABLE")
    n_before = len(sb)
    j4 = T.judge_stop("x", "", post=spost, upstream=upstream, count=len)
    check("judge_stop: off unless enabled -- nothing sent, says why",
          j4["judged"] is False and "off" in j4["why"]
          and len(sb) == n_before)

    T.READOUT = "typed"

    # --- not enabled (no serving process called enable()): nothing sent
    sent = []
    with T.Turn(user_turn("Add a pause menu."),
                post=lambda b, t: sent.append(b), upstream=upstream) as t:
        facts = t.facts()
        r = t.pick("koota", cands)
    check("off unless the serving process enabled it: no request, rules "
          "answer", not sent and facts["build_intent"]["source"] == "off"
          and r["source"] == "off")
    # PRIME_COUNTS stays inside the labels: "None of these" takes a letter
    # too, so the largest count is len(LETTERS) (2..27 raised IndexError at
    # every proxy start until 2026-09-28).
    check("[prime] every PRIME_COUNTS size has a letter for each option, "
          "None included",
          max(T.PRIME_COUNTS) <= len(T.D.LETTERS)
          and min(T.PRIME_COUNTS) == 2)
    test_typed(released)
    test_per_model()
    ok = sum(1 for _, o in CHECKS if o)
    print(f"{ok}/{len(CHECKS)} checks passed")
    return 0 if ok == len(CHECKS) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:                                            # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
