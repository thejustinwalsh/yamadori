#!/usr/bin/env python
"""skill_inject: one source, a jjava gate, a composer, a renderer per model
(mcp/skill_inject.py). No GPU, no network: the decider reads a fake model
server through the REAL typed readout (decider_bonsai.read via
decide_turn.Turn.decide).

    python mcp/test_skill_inject.py      -> "N/M checks passed"

GATES: items come from the skill's own lines, in its words, never a doubt-
bearing one; a repeat (same text or same code spans) is asked once; at most
MAX_SKILLS skills; an item given before comes back only with an event;
stage 2 is a jjava SCORE per item over the turn's state (levels lowest
first, the fact named in the question), passing on the top level when
untuned and on THRESHOLDS tiers when tuned; stage 3 a NOUL over the
shortlist that fits the lane's room; medium stage-3 keeps only the high
items; no decider or a failure injects nothing; the renderer adds nothing to
an item's words, in each format and voice; a profile per model family;
skill_select.decide under a Turn returns the injector's text, records the
items given and replays a retry byte for byte.
"""
from __future__ import annotations

import json
import math
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import offline_stores  # noqa: E402

_TMP = offline_stores.isolate("yamadori_test_skill_inject_")
os.environ["YAMADORI_SLOTS"] = "4"
os.environ.setdefault("YAMADORI_GPU_ROOM", "0")

import decider_bonsai as D  # noqa: E402
import decide_turn as T  # noqa: E402
import skill_inject as I  # noqa: E402

CHECKS: list[tuple[str, bool]] = []


def check(name: str, ok: bool, detail=None) -> None:
    CHECKS.append((name, bool(ok)))
    if not ok:
        print("FAIL", name, "" if detail is None else str(detail)[:500])


IDS = {}
for _i, _L in enumerate(D.LETTERS):
    IDS[_L] = 600 + _i
    IDS[" " + _L] = 700 + _i


def upstream(path, payload=None, timeout=30):
    if path == "/tokenize":
        s = payload["content"]
        return {"tokens": [IDS[s]] if s in IDS else [1, 2]}
    raise AssertionError(path)


def printed(msgs):
    q = msgs[2]["content"]
    out = []
    for ln in q.split("OPTIONS:\n")[1].split("\n\n")[0].splitlines():
        lab, text = ln.split(". ", 1)
        out.append((lab, text))
    return q, out


class Fake:
    """A model server whose answer is decided by `want(state, question,
    option texts)` -> the option text to favour (p) or None (uniform)."""

    def __init__(self, want, p=0.9):
        self.want, self.p, self.bodies = want, p, []

    def __call__(self, body, timeout):
        self.bodies.append(body)
        m = body["messages"]
        state = m[1]["content"][len(D.STATE_HEAD):]
        q, opts = printed(m)
        fav = self.want(state, q, [t for _l, t in opts])
        n = len(opts)
        top = []
        for lab, text in opts:
            pr = (self.p if text == fav else (1 - self.p) / max(n - 1, 1)) \
                if fav is not None else 1.0 / n
            top.append({"id": IDS[" " + lab], "logprob": math.log(pr)})
        top.sort(key=lambda t: -t["logprob"])
        return {"choices": [{"logprobs": {"content": [
            {"id": top[0]["id"], "logprob": top[0]["logprob"],
             "top_logprobs": top}]}}], "usage": {"prompt_tokens": 60},
            "timings": {"prompt_n": 15, "cache_n": 45, "prompt_ms": 5.0}}


def skill(sid, name, items, area="koota"):
    return {"id": sid, "name": name, "version": 1,
            "rule": {"frameworks": [area]}, "items": items}


KOOTA = skill("k1", "koota-queries", [
    {"form": "DO", "text": "Query entities with `world.query(Position, "
     "Velocity)` and iterate with `forEach`."},
    {"form": "DO NOT", "text": "Store entity references across frames; "
     "hold the `Entity` id instead."},
    {"form": "DO", "text": "Verify the API in the README before relying on "
     "it."},                                             # doubt: never in
])
R3F = skill("r1", "r3f-frame-loop", [
    {"form": "DO", "text": "Mutate refs inside `useFrame` instead of "
     "setting React state every frame."},
    {"form": "WHEN", "situation": "an animation stutters",
     "text": "move the per-frame work into `useFrame`."},   # same spans
], area="r3f")
MATH = skill("m1", "math-noise", [
    {"form": "DO", "text": "Seed noise with `createNoise2D(seed)` for "
     "repeatable terrain."}], area="pmndrs_math")
EXTRA = skill("x1", "extra", [{"form": "DO", "text": "Use `extraThing()`."}])


def user(text):
    return [{"role": "system", "content": "sys"},
            {"role": "user", "content": text}]


def level_for(fact_words):
    """A fake that answers the item Score by the fact's words and the
    stage-3 noul by `inject`."""
    def want(state, q, opts):
        if q.startswith("QUESTION: How does this fact"):
            fact = q.split("FACT: ", 1)[1].split("\n")[0]
            for w, lvl in fact_words.items():
                if w in fact:
                    return I.ITEM_LEVELS[lvl]
            return I.ITEM_LEVELS[0]
        if "more correct with these facts" in q:
            return "yes" if "yes" in opts else None
        return None
    return want


def test_items() -> None:
    it = I.skill_items(KOOTA)
    check("items: one per line of the skill, keyed <skill id>#<index>",
          [x["key"] for x in it] == ["k1#0", "k1#1", "k1#2"], it)
    check("items: a doubt-bearing line is marked (the ASSURED VOICE rule)",
          it[2]["doubt"] and not it[0]["doubt"], it[2])
    check("items: code spans are read from the item's own backticks",
          "world.query(Position, Velocity)" in it[0]["spans"])
    keep, left = I.candidates([KOOTA, R3F, MATH, EXTRA])
    keys = [x["key"] for x in keep]
    check("candidates: the doubt line never goes to a question",
          "k1#2" not in keys and any(x.get("key") == "k1#2" and
                                     x["why"].startswith("doubt")
                                     for x in left), left)
    check("candidates: at most MAX_SKILLS skills (SkillsBench), in stage "
          "1's order", I.MAX_SKILLS == 3 and "x1#0" not in keys
          and any(x.get("skill") == "x1" for x in left), keys)
    check("candidates: an item whose code spans repeat an earlier item's is "
          "asked once", "r1#0" in keys and "r1#1" not in keys, keys)
    keep2, left2 = I.candidates([KOOTA], given={"k1#0": 3})
    check("candidates: an item given before is left out with no event",
          [x["key"] for x in keep2] == ["k1#1"])
    keep3, _ = I.candidates([KOOTA], given={"k1#0": 3},
                            events={"k1": "error"})
    check("candidates: an error brings a given item back",
          [x["key"] for x in keep3] == ["k1#0", "k1#1"])


def run_turn(messages, fake, skills, **kw):
    with T.Turn(messages, post=fake, upstream=upstream, count=len,
                on=True, key="conv", request="req1") as t:
        return I.inject(skills, T.turn_kind(messages), turn=t,
                        model="bonsai", **kw), t.record()


def test_gate() -> None:
    I.THRESHOLDS.clear()
    fake = Fake(level_for({"world.query": 3, "Entity": 1, "useFrame": 3,
                           "createNoise2D": 2}))
    (text, rec), trec = run_turn(
        user("Make the enemies chase the player with a koota query in "
             "useFrame."), fake, [KOOTA, R3F, MATH])
    g = rec["gate"]
    names = [b["messages"][2]["content"] for b in fake.bodies]
    check("stage 2: one SCORE per candidate item, the fact named in the "
          "question and the levels printed as options",
          sum("FACT: " in q for q in names) == 2 * 4
          and all(lvl in names[0] for lvl in [I.ITEM_LEVELS[1]]), names[:1])
    check("stage 2: read in two orders over the turn's state (the typed "
          "readout)", all(len(b["messages"]) == 4 for b in fake.bodies)
          and fake.bodies[0]["messages"][1]["content"].startswith(
              D.STATE_HEAD + "Make the enemies"))
    passed = [r["key"] for r in g["items"] if r["pass"]]
    check("stage 2 untuned: an item passes on the top level (argmax), "
          "others do not", passed == ["k1#0", "r1#0"], g["items"])
    check("stage 2: the belief `need` is P(top level)",
          all(abs(r["need"] - (0.9 if r["pass"] else 0.1 / 3)) < 1e-4
              for r in g["items"]), g["items"])
    check("stage 3: one NOUL over the shortlist; yes -> all go in",
          g["inject"]["act"] == "all" and g["chosen"] == ["k1#0", "r1#0"]
          and rec["chosen"] == ["k1#0", "r1#0"], g)
    check("render: the note voice at the tail, the craft header, the "
          "items' own lines", text.startswith("\n---\n")
          and "- DO: Query entities with `world.query(Position, Velocity)`"
          in text and "useFrame" in text, text)
    check("the record carries ids and numbers, never the conversation's "
          "text", "Make the enemies" not in json.dumps(rec), rec)
    check("every decision is logged by the Turn (decide_turn DECISIONS)",
          len(trec["decisions"]) == 5, trec["decisions"])
    # stage 3 says no -> nothing goes in (the safe default)
    fake2 = Fake(lambda s, q, o: ("no" if "more correct" in q else
                                  I.ITEM_LEVELS[3]))
    (text2, rec2), _ = run_turn(user("koota query"), fake2, [KOOTA])
    check("stage 3 no -> nothing goes in", text2 == ""
          and rec2["gate"]["inject"]["act"] == "none", rec2["gate"])
    # no item at the top -> no stage-3 question at all
    fake3 = Fake(lambda s, q, o: I.ITEM_LEVELS[1])
    (text3, rec3), _ = run_turn(user("koota query"), fake3, [KOOTA])
    check("no item passes -> no stage-3 question, nothing goes in",
          text3 == "" and rec3["gate"]["inject"] is None
          and not any("more correct" in b["messages"][2]["content"]
                      for b in fake3.bodies))


def test_tiers() -> None:
    I.THRESHOLDS.clear()
    I.THRESHOLDS["bonsai"] = {
        I.QSET_ITEM: {"act_yes": 0.85, "caution_yes": 0.6},
        I.QSET_INJECT: {"act_yes": 0.95, "caution_yes": 0.7}}
    try:
        # item world.query at 0.9 (high), useFrame at 0.7 (medium)
        def want(state, q, opts):
            return None
        calls = {"n": 0}

        class F(Fake):
            def __call__(self, body, timeout):
                q, opts = printed(body["messages"])
                if "FACT: " in q:
                    fact = q.split("FACT: ", 1)[1].split("\n")[0]
                    self.p = (0.9 if "world.query" in fact else 0.7
                              if "useFrame" in fact else 0.05)
                    self.want = (lambda s, qq, o: I.ITEM_LEVELS[3]) \
                        if self.p > 0.5 else (lambda s, qq, o:
                                              I.ITEM_LEVELS[0])
                else:
                    calls["n"] += 1
                    self.p = 0.8
                    self.want = lambda s, qq, o: "yes"
                return super().__call__(body, timeout)
        (text, rec), _ = run_turn(user("x"), F(want), [KOOTA, R3F])
        items = {r["key"]: r for r in rec["gate"]["items"]}
        check("tuned stage 2: high and medium tiers from THRESHOLDS on the "
              "belief", items["k1#0"]["tier"] == "high"
              and items["r1#0"]["tier"] == "medium"
              and items["k1#1"]["tier"] == "low", items)
        check("tuned stage 3 medium yes -> only the HIGH items go in",
              rec["gate"]["inject"]["tier"] == "medium"
              and rec["gate"]["inject"]["act"] == "sure_only"
              and rec["chosen"] == ["k1#0"], rec["gate"])
        check("the high item ranks before the caution item in the shortlist",
              rec["gate"]["shortlist"] == ["k1#0", "r1#0"])
    finally:
        I.THRESHOLDS.clear()


def test_room_and_render() -> None:
    long_items = [{"key": f"s#{i}", "skill": "s", "form": "DO",
                   "situation": "", "text": "word " * 120, "spans": []}
                  for i in range(8)]
    q, listed = I.inject_question(long_items, room=500)
    check("stage 3: facts past the lane's room are dropped, lowest first",
          0 < len(listed) < 8 and listed[0]["key"] == "s#0"
          and I.L.tokens(q["text"]) <= 500 or len(listed) == 1, len(listed))
    check("the lane's room is derived from budget.LANE_TOKENS - "
          "decide_turn.STATE_TOKENS less the template",
          0 < I.question_room() < 1024)
    its = I.skill_items(KOOTA)[:2] + I.skill_items(R3F)[1:]
    t_list = I.render(its, {"format": "list", "voice": "note"})
    t_tab = I.render(its, {"format": "table", "voice": "note"})
    t_fp = I.render(its, {"format": "list", "voice": "first_person"})
    for it in its:
        check(f"render keeps {it['key']}'s words in every format",
              it["text"].rstrip(" .") in t_list["text"]
              and it["text"].rstrip(" .").replace("|", "\\|")
              in t_tab["text"])
    check("render table: one row per item, never / always / the situation",
          t_tab["text"].count("\n| ") == 4 and "| never |" in t_tab["text"]
          and "| an animation stutters |" in t_tab["text"], t_tab["text"])
    check("render first person: injected TEXT at the tail in the model's "
          "voice (no prefill), each item's words kept",
          t_fp["placement"] == "tail" and "prefill" not in t_fp
          and I.FIRST_PERSON_HEAD in t_fp["text"]
          and "- I'll query entities with `world.query(Position, Velocity)`"
          in t_fp["text"] and "- I won't store entity references" in
          t_fp["text"] and "- When an animation stutters, I'll move"
          in t_fp["text"], t_fp)
    t_pf = I.render(its, {"format": "list",
                          "voice": "first_person_prefill"})
    check("render prefill voice: one reasoning line for the proxy, no text, "
          "ending on a letter (the prefill rule)",
          t_pf["text"] == "" and t_pf["placement"] == "reasoning_prefill"
          and t_pf["prefill"].startswith(I.PREFILL_HEAD)
          and t_pf["prefill"][-1].isalnum()
          and "I'll query entities with `world.query(Position, Velocity)`"
          in t_pf["prefill"] and "; when an animation stutters, I'll "
          in t_pf["prefill"], t_pf)
    check("compose: at most the profile's max_items, in the gate's order",
          [x["key"] for x in I.compose(its, ["r1#1", "k1#0", "k1#1"],
                                       {"max_items": 2})] == ["r1#1", "k1#0"])


def test_profiles() -> None:
    check("profiles: one per model family",
          I.family_of("bonsai") == "bonsai"
          and I.family_of("flash-next") == "flash-next"
          and I.family_of("mirai-s-q2") == "mirai-s"
          and I.family_of("something") == "default")
    p = I.profile_for("bonsai")
    check("a profile names the evidence of every choice",
          set(p["evidence"]) == {"format", "voice", "max_items"}
          and all(v["kind"] in ("default", "measured")
                  for v in p["evidence"].values()), p)


def test_profile_follows_the_serving_model_not_the_reader() -> None:
    """A LOCKED model's jjava reads on bonsai-a4000 (max_mode.decider_model)
    whoever holds the card: the PROFILE is the model that receives the text
    (flash-next's), THRESHOLDS stay the reader's."""
    I.THRESHOLDS.clear()
    saved = dict(I.PROFILES)
    try:
        I.PROFILES["flash-next"] = dict(I.PROFILES["flash-next"],
                                        max_items=1, format="table")
        fake = Fake(level_for({"world.query": 3}))
        with T.Turn(user("koota query"), post=fake, upstream=upstream,
                    count=len, on=True, key="conv", request="req1") as t:
            text, rec = I.inject([KOOTA], "user", turn=t,
                                 model="bonsai-a4000", serving="flash-next")
        check("the record names the reader and the serving model, the "
              "profile is the serving model's",
              rec["model"] == "bonsai-a4000"
              and rec["serving"] == "flash-next"
              and rec["profile"] == "flash-next"
              and rec["rendered"]["format"] == "table", rec)
        fake = Fake(level_for({"world.query": 3}))
        with T.Turn(user("koota query"), post=fake, upstream=upstream,
                    count=len, on=True, key="conv", request="req1") as t:
            _t2, rec2 = I.inject([KOOTA], "user", turn=t,
                                 model="bonsai-a4000")
        check("no serving model given and none bound to a request: the "
              "reader's family, as before", rec2["serving"] ==
              "bonsai-a4000" and rec2["profile"] == "bonsai", rec2)
        import cancel
        import max_mode
        fake = Fake(level_for({"world.query": 3}))
        with cancel.bound(cancel.Token()):
            max_mode.set_current("mirai-s")
            with T.Turn(user("koota query"), post=fake, upstream=upstream,
                        count=len, on=True, key="conv", request="req1") as t:
                _t3, rec3 = I.inject([KOOTA], "user", turn=t,
                                     model="bonsai-a4000")
        check("a request bound to mirai-s: its profile, whoever reads",
              rec3["serving"] == "mirai-s" and rec3["profile"] == "mirai-s",
              rec3)
    finally:
        I.PROFILES.clear()
        I.PROFILES.update(saved)


def test_no_decider() -> None:
    text, rec = I.inject([KOOTA], "user", turn=None, model="bonsai")
    check("no decider -> nothing goes in, and the record says why",
          text == "" and "nothing goes in" in rec["why"]
          and rec["failure"]["code"] == "NO_DECIDER", rec)

    class Down:
        on = True

        def decide(self, qs):
            raise D.DeciderUnavailable("MODEL_UNREACHABLE", "down", True, "x")
    text, rec = I.inject([KOOTA], "user", turn=Down(), model="bonsai")
    check("a decider failure -> nothing goes in, the failure recorded",
          text == "" and rec["gate"]["failure"]["code"]
          == "MODEL_UNREACHABLE", rec)


def test_select_wiring() -> None:
    import skill_select as S
    fake = Fake(level_for({"world.query": 3, "Entity": 3}))
    msgs = user("Use a koota query to move the enemies.")
    orig = S.select

    def sel(ev, rc, pool, tools, **kw):
        return [dict(KOOTA, _slot="asked")], {"matched": [
            {"id": "k1", "why": ["asked"]}]}
    import skill_match
    orig_plan = skill_match.plan_turn

    def plan_turn(**kw):
        return kw["old"], {}, kw.get("pkg_state")
    skill_match.plan_turn = plan_turn
    try:
        with T.Turn(msgs, post=fake, upstream=upstream, count=len, on=True,
                    key="conv", request="req9"):
            text, rec, st = S.decide(msgs, "code_generation", [KOOTA],
                                     [], None, key="kk", select_fn=sel)
            text2, rec2, _st = S.decide(msgs, "code_generation", [KOOTA],
                                        [], st, key="kk", select_fn=sel)
        check("skill_select.decide under a Turn: the injector's items",
              "world.query" in text and "Entity" in text
              and rec["decisions"][0]["form"] == "items"
              and rec["inject"]["chosen"] == ["k1#0", "k1#1"], rec)
        check("the state remembers each item given",
              set(st.get("items") or {}) == {"k1#0", "k1#1"}, st)
        check("a retry of the same request replays the same text",
              text2 == text and rec2.get("replayed"))
        text3, rec3, _ = S.decide(msgs, "code_generation", [KOOTA], [],
                                  None, key="k3", select_fn=sel)
        check("no Turn in scope: NOTHING goes in -- no path picks skills "
              "without jjava (operator, 2026-09-29) -- and the record says "
              "why", text3 == "" and rec3["inject"]["failure"]["code"]
              == "NO_DECIDER" and rec3["decisions"] == [], rec3)
        import decider_stub
        with decider_stub.installed(want=decider_stub.evidence):
            with T.Turn(msgs, on=None, key="conv", request="req10") as t4:
                text4, rec4, _ = S.decide(msgs, "code_generation", [KOOTA],
                                          [], None, key="k4", select_fn=sel)
        check("the offline stub decider (mcp/decider_stub.py): the real "
              "Turn, on, answering by its evidence rule",
              t4.on and "world.query" in text4
              and rec4["inject"]["chosen"], rec4.get("inject"))
    finally:
        S.select = orig
        skill_match.plan_turn = orig_plan


# ----------------------------------------------- the 2026-10-06 variants ---
VITEMS = [
    {"key": "k1#0", "name": "koota-queries", "fact": "Query entities with "
     "`world.query(Position, Velocity)` and iterate with `forEach`."},
    {"key": "k1#1", "name": "koota-queries", "fact": "Do not: Store entity "
     "references across frames; hold the `Entity` id instead."},
    {"key": "r1#0", "name": "r3f-frame-loop", "fact": "Mutate refs inside "
     "`useFrame` instead of setting React state every frame."},
    {"key": "m1#0", "name": "math-noise", "fact": "Seed noise with "
     "`createNoise2D(seed)` for repeatable terrain."},
]


def vturn(fake, state="STATE", kind="step"):
    return T.Turn([], state=state, state_info={"kind": kind}, post=fake,
                  upstream=upstream, count=len, on=True, key="conv",
                  request="req1")


def fact_of_q(q):
    return q.split("FACT: ", 1)[1].split("\n")[0]


def vwant(rules):
    """A fake's want(): `rules` {statement: words that make it true}; a
    noul is answered yes when a word is in the question's fact."""
    def want(state, q, opts):
        for stmt, words in rules.items():
            if q.startswith("QUESTION: " + stmt):
                f = fact_of_q(q)
                return "yes" if any(w in f for w in words) else "no"
        return None
    return want


def test_served_set_is_untouched() -> None:
    check("the served question set is variant a: ITEM_Q / ITEM_LEVELS, "
          "version inject-q/1", I.SERVED_VARIANT == "a"
          and I.QUESTION_VERSION == "inject-q/1"
          and I.ITEM_Q.startswith("How does this fact bear on")
          and len(I.ITEM_LEVELS) == 4 and I.NEED_LEVEL == "3")
    I.THRESHOLDS.clear()
    fake = Fake(level_for({"world.query": 3}))
    run_turn(user("koota query"), fake, [KOOTA])
    qs = [b["messages"][2]["content"] for b in fake.bodies]
    new = (I.NEXT_Q, I.DONE_Q, I.FIT_Q, I.PICK_Q)
    check("gate() asks only the served Score and the stage-3 noul: none of "
          "the new variants' statements is ever sent",
          qs and not any(n in q for n in new for q in qs), qs[:1])


def test_framed_state() -> None:
    goal = "Build  a pagoda\nscene with r3f."
    st = "Assistant called terminal: ls\nResult of terminal: a.ts"
    f = I.framed_state(goal, st, {"kind": "step", "cut": False})
    check("framed state, a step: the goal first, then the latest step, the "
          "parts named, the state kept whole",
          f == (I.GOAL_HEAD + "\nBuild a pagoda scene with r3f.\n\n"
                + I.STEP_HEAD + "\n" + st), f)
    f1 = I.framed_state(goal, "Build a pagoda scene with r3f.",
                        {"kind": "user", "previous_assistant": None})
    check("framed state, a first user turn: the message is the goal and is "
          "said ONCE (variant d said it twice)",
          f1.count("pagoda") == 1 and f1.startswith(I.OPENING_HEAD)
          and I.GOAL_HEAD not in f1, f1)
    f2 = I.framed_state(goal, "Assistant (previous turn): ok\n\nUser: next",
                        {"kind": "user", "previous_assistant": {"tokens": 3}})
    check("framed state, a later user turn: the goal, then the newest message",
          f2.startswith(I.GOAL_HEAD) and I.USER_HEAD in f2
          and f2.endswith("User: next"), f2)
    check("framed state: no goal, no goal block",
          I.GOAL_HEAD not in I.framed_state("", st, {"kind": "step"}))
    long_goal = "g" * 5000
    check("framed state: the goal is cut to GOAL_CHARS",
          len(I.framed_state(long_goal, st, {"kind": "step"}))
          <= I.GOAL_CHARS + len(st) + 200)
    cut = ("H" * 3000) + "\n\n[... middle left out ...]\n\n" + ("T" * 3000)
    fc = I.framed_state(goal, cut, {"kind": "step", "cut": True})
    check("framed state of a CUT state is never longer than the cut state: "
          "the frame's room comes off the head, the middle marker and the "
          "tail stay", len(fc) <= len(cut) and fc.endswith("T" * 3000)
          and "[... middle left out ...]" in fc and fc.startswith(
              I.GOAL_HEAD), (len(fc), len(cut)))
    pre = len(I.framed_state(goal, "", {"kind": "step"}))
    fm = I.framed_state(goal, st, {"kind": "step"}, max_chars=pre + 20)
    check("framed state: max_chars is respected (the head of the state "
          "gives way, the frame itself is never cut)",
          len(fm) == pre + 20 and fm.endswith("a.ts")
          and fm.startswith(I.GOAL_HEAD), fm)
    check("goal_of: the first user message, folded and cut",
          I.goal_of([{"role": "system", "content": "s"},
                     {"role": "user", "content": "do  this\nthing"},
                     {"role": "user", "content": "later"}]) == "do this thing"
          and I.goal_of([]) == "")


def test_variant_questions() -> None:
    it = VITEMS[0]
    n, d, g = I.q_next(it), I.q_done(it), I.q_fit(it)
    check("f/g questions are NOULS naming the craft and the fact, one "
          "statement each", all(q["type"] == "noul" for q in (n, d, g))
          and n["text"].startswith(I.NEXT_Q) and d["text"].startswith(
              I.DONE_Q) and g["text"].startswith(I.FIT_Q)
          and all("CRAFT: koota-queries\nFACT: " + it["fact"] in q["text"]
                  for q in (n, d, g)), [n, d, g])
    check("each variant's questions are their own question sets "
          "(thresholds and temperatures are keyed by the name's family)",
          n["name"].split(":")[0] == "skill_item_f_next"
          and d["name"].split(":")[0] == "skill_item_f_done"
          and g["name"].split(":")[0] == "skill_item_g"
          and I.q_fit(it, "h")["name"].split(":")[0] == "skill_item_h"
          and n["name"].endswith(":k1#0"))
    check("the statements are positive single judgments: no negation "
          "(TS jaggedness 4; variant c's 'does not yet' read worst)",
          all(" not " not in " " + s.lower() + " "
              and " no " not in " " + s.lower() + " "
              for s in (I.NEXT_Q, I.DONE_Q, I.FIT_Q)))
    ni = {"key": "x#0", "fact": "Bare fact."}
    check("an item with no craft name is asked without the CRAFT line",
          "CRAFT" not in I.q_next(ni)["text"]
          and I.q_next(ni)["text"].endswith("FACT: Bare fact."))
    q, listed = I.q_pick(VITEMS)
    check("pick: one CHOICE over the facts keyed by item key, 'none' last "
          "in the list and in the MIDDLE of both printed orders",
          q["type"] == "choice" and q["keys"][-1] == I.NONE_KEY
          and q["keys"][:-1] == [x["key"] for x in VITEMS]
          and q["options"][-1] == I.PICK_NONE and q["none"] == 4
          and q["options"][0].startswith("[koota-queries] Query")
          and [x["key"] for x in listed] == [x["key"] for x in VITEMS], q)
    orders = D.two_orders(len(q["keys"]), q["none"])
    check("pick: neither order prints 'none' first or last",
          all(0 < o.index(4) < len(o) - 1 for o in orders), orders)
    many = [{"key": f"s#{i}", "name": "s", "fact": f"fact {i}"}
            for i in range(30)]
    q30, l30 = I.q_pick(many)
    check("pick: at most PICK_MAX facts are listed (a letter each, none "
          "included: 26 options)", len(l30) == I.PICK_MAX == 25
          and len(q30["options"]) == 26 and q30["keys"][-1] == "none")
    check("planned questions: f two an item, g one, h a pick and PICK_TOP "
          "fits",
          len(I.planned_questions("f", VITEMS)) == 8
          and len(I.planned_questions("g", VITEMS)) == 4
          and len(I.planned_questions("h", VITEMS)) == 1 + I.PICK_TOP
          and len(I.planned_questions("h", VITEMS[:2])) == 3)
    try:
        I.planned_questions("z", VITEMS)
        bad = False
    except ValueError:
        bad = True
    check("an unknown variant is refused", bad)


def test_item_beliefs() -> None:
    I.THRESHOLDS.clear()
    # f: NEXT yes for world.query and useFrame; DONE yes for useFrame
    fake = Fake(vwant({I.NEXT_Q: ("world.query", "useFrame"),
                       I.DONE_Q: ("useFrame",)}))
    with vturn(fake) as t:
        rows, meta = I.item_beliefs("f", VITEMS, t)
    b = {r["key"]: r for r in rows}
    check("f: need = P(next) x (1 - P(done)) in code, two nouls an item "
          "(2 reads each)", len(fake.bodies) == 2 * 8
          and abs(b["k1#0"]["belief"] - 0.81) < 1e-4
          and abs(b["r1#0"]["belief"] - 0.09) < 1e-4
          and abs(b["m1#0"]["belief"] - 0.09) < 1e-4, b)
    check("f: passes only when NEXT is yes and DONE is no",
          [r["pass"] for r in rows] == [True, False, False, False]
          and b["k1#0"]["parts"]["next"]["noul"] > 0.5
          and b["k1#0"]["parts"]["done"]["noul"] < 0.5
          and all(r["decision_id"] for r in rows), rows[0])
    qs = [x["messages"][2]["content"] for x in fake.bodies]
    check("f: the same state is read for every question (placed once), the "
          "questions after it", len({x["messages"][1]["content"]
                                     for x in fake.bodies}) == 1
          and sum(I.NEXT_Q in q for q in qs) == 8
          and sum(I.DONE_Q in q for q in qs) == 8)
    # g
    fake = Fake(vwant({I.FIT_Q: ("world.query", "createNoise2D")}))
    with vturn(fake) as t:
        rows, _ = I.item_beliefs("g", VITEMS, t)
    check("g: one noul an item; belief = P(true); passes on yes",
          len(fake.bodies) == 2 * 4
          and [round(r["belief"], 2) for r in rows] == [0.9, 0.1, 0.1, 0.9]
          and [r["pass"] for r in rows] == [True, False, False, True], rows)

    # h: the pick favours r1#0; fits yes for useFrame and createNoise2D
    def hwant(state, q, opts):
        if q.startswith("QUESTION: " + I.PICK_Q):
            return next(o for o in opts if "useFrame" in o)
        if q.startswith("QUESTION: " + I.FIT_Q):
            f = fact_of_q(q)
            return "yes" if "useFrame" in f or "createNoise2D" in f else "no"
        return None
    fake = Fake(hwant)
    with vturn(fake) as t:
        rows, meta = I.item_beliefs("h", VITEMS, t)
    b = {r["key"]: r for r in rows}
    check("h: ONE pick over all facts, then fits for the PICK_TOP it "
          "ranked first (2 reads each)", len(fake.bodies) == 2 * (1 + 3)
          and meta["pick"]["top"] == ["r1#0", "k1#0", "k1#1"], meta)
    check("h: need = P(fit) x (1 - P(none)); the fact the pick left out of "
          "its top goes in with belief 0 however a fit would read",
          abs(b["r1#0"]["belief"] - 0.9 * (1 - meta["pick"]["none"])) < 1e-4
          and b["m1#0"]["belief"] == 0.0 and not b["m1#0"]["pass"]
          and b["m1#0"]["parts"]["pick"]["in_top"] is False
          and b["r1#0"]["pass"] and not b["k1#0"]["pass"], b)
    check("h: every row names a decision of its case (inject_tune.loro "
          "groups rows by it)", all(r["decision_id"] for r in rows)
          and b["m1#0"]["decision_id"] == meta["pick"]["decision_id"])
    pq = [x["messages"][2]["content"] for x in fake.bodies][:2]
    for q in pq:
        opts = [ln.split(". ", 1)[1] for ln in q.split(
            "OPTIONS:\n")[1].split("\n\n")[0].splitlines()]
        check("h: the pick prints 'none' in the middle of the options in "
              "each order", opts[2] == I.PICK_NONE and len(opts) == 5, opts)
    # none favoured: every belief scaled down by the none mass
    fake = Fake(lambda s, q, o: (I.PICK_NONE if q.startswith(
        "QUESTION: " + I.PICK_Q) else "yes" if q.startswith(
            "QUESTION: " + I.FIT_Q) else None))
    with vturn(fake) as t:
        rows, meta = I.item_beliefs("h", VITEMS, t)
    check("h: when the pick says none, a yes fit is scaled by 1 - P(none): "
          "the case-level 'nothing here is needed'",
          meta["pick"]["none"] > 0.85
          and all(r["belief"] < 0.1 for r in rows), rows)
    fake = Fake(lambda s, q, o: None)
    with vturn(fake) as t:
        r0, m0 = I.item_beliefs("h", [], t)
        r1, _ = I.item_beliefs("g", [], t)
        r2, _ = I.item_beliefs("f", [], t)
    check("no items -> no question is sent", r0 == [] and r1 == []
          and r2 == [] and not fake.bodies)
    try:
        with vturn(Fake(lambda s, q, o: None)) as t:
            I.item_beliefs("z", VITEMS, t)
        bad = False
    except ValueError:
        bad = True
    check("an unknown variant is refused by item_beliefs", bad)


def main() -> int:
    D.release = lambda slot, why="": {"released": True, "slot": slot}
    for fn in (test_items, test_gate, test_tiers, test_room_and_render,
               test_profiles, test_profile_follows_the_serving_model_not_the_reader,
               test_no_decider, test_select_wiring,
               test_served_set_is_untouched, test_framed_state,
               test_variant_questions, test_item_beliefs):
        try:
            fn()
        except Exception:                                        # noqa: BLE001
            traceback.print_exc()
            check(f"{fn.__name__} raised", False)
    ok = sum(1 for _n, v in CHECKS if v)
    print(f"{ok}/{len(CHECKS)} checks passed")
    return 0 if ok == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
