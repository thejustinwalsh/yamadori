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


def test_no_decider() -> None:
    text, rec = I.inject([KOOTA], "user", turn=None, model="bonsai")
    check("no decider -> nothing goes in, and the record says why",
          text == "" and "safe default" in rec["why"],
          rec)

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
        check("no Turn in scope (an offline caller): the per-skill path",
              "inject" not in rec3 and text3, rec3.get("decisions"))
    finally:
        S.select = orig
        skill_match.plan_turn = orig_plan


def main() -> int:
    D.release = lambda slot, why="": {"released": True, "slot": slot}
    for fn in (test_items, test_gate, test_tiers, test_room_and_render,
               test_profiles, test_no_decider, test_select_wiring):
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
