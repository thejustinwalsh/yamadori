#!/usr/bin/env python
"""The skill selector as ROUNDS, QUESTIONS, a DECIDER and a STATECHART
(operator, 2026-09-27). No GPU, no network beyond loopback fakes.

    python mcp/test_skill_questions.py      -> "N/M checks passed"

WHAT IS GATED

  1. THE ROUNDS narrow the armed pool in order -- language, framework,
     phase, situation, not_given_or_faded -- each logging its survivors;
     vector search is consulted in a round ONLY when that round's patterns
     find nothing; a name behind "without" places the request nowhere.
  2. THE QUESTIONS: one per area, at most 6 options with an explicit NONE
     last, each option a survivor's trigger condition; a CATEGORY question
     first when no pattern places the turn; the record carries each
     question's options, probabilities, pick and decider -- never the
     request's text.
  3. THE DECIDERS: decide(state, options) -> probabilities; the pick is the
     argmax when it beats NONE; every decider is order-invariant; the stub
     abstains on options only a model can settle (and the last resort, the
     fallback, is asked ONCE). (The CLM decider's checks went with
     mcp/clm.py, removed 2026-09-29; the way back is commit e360d37.)
  4. THE STATECHART: the transition table (a guard returning None is an
     illegal event), legality in front of the questions (a cooling area's
     question is never asked), the chart persisted in the skill state and
     reset by a compaction.
  5. THROUGH THE SERVED TEMPLATE (test_ledger's harness): a conversation
     whose skills go in as a body, then a recall on an error step, extends
     the slot on every request, and the chart carried by the ledger row
     moves given -> recall_eligible -> given.
"""
from __future__ import annotations

import json
import os
import random
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

_TMP = tempfile.mkdtemp(prefix="yamadori_test_skill_questions_")
for _k, _v in (("YAMADORI_CORPUS_DB", "corpus.sqlite3"),
               ("YAMADORI_NEBARI_DB", "nebari.sqlite3"),
               ("RINGS_DB", "rings.sqlite3"),
               ("CODE_INDEX_DB", "code.sqlite3"),
               ("YAMADORI_SLOTS_STATE", "slots_state.json"),
               ("CONCEPT_SEED_LAST", "seed_last.json")):
    os.environ[_k] = os.path.join(_TMP, _v)
os.environ.setdefault("YAMADORI_GPU_ROOM", "0")
os.environ["YAMADORI_SKILL_DECIDER"] = "stub"

import test_ledger as T  # noqa: E402  (its own temp paths + fake upstream)
import proxy  # noqa: E402
import skill_chart as CH  # noqa: E402
import skill_classify as C  # noqa: E402
import skill_deciders as D  # noqa: E402
import skill_select as S  # noqa: E402
import skills  # noqa: E402

_results: list[tuple[bool, str, str]] = []


def check(ok, name: str, detail="") -> bool:
    _results.append((bool(ok), name, str(detail)[:400]))
    return bool(ok)


# ------------------------------------------------------------- the pool --
def mk(sid: str, name: str, desc: str, applies: dict, items=None,
       **gates) -> dict:
    items = items or [{"form": "DO", "situation": "", "text": f"do {name}."}]
    rule = C.rule_from_metadata(dict(applies, **gates))
    rule = C.with_gates(rule, description=desc, **{
        k: v for k, v in gates.items() if k in ("phases", "situations",
                                                "all_of", "topics")})
    title = name.replace("-", " ").title()
    body = "\n".join([title] + [
        f"- {it['form']}: {it['text']}" if it["form"] != "WHEN" else
        f"- WHEN {it['situation']}: {it['text']}" for it in items])
    return {"id": sid, "version": "1", "name": name, "title": title,
            "description": desc, "rule": rule, "items": items,
            "body": body, "text": body}


KOOTA_Q = mk(
    "k_queries", "koota-queries-and-systems",
    "Use when querying a koota world and writing its systems in "
    "TypeScript: world.query with updateEach or readEach, queryFirst.",
    {"frameworks": ["koota"]},
    [{"form": "DO", "situation": "", "text": "batch-update entities with "
      "world.query(Position, Velocity).updateEach(([pos, vel]) => ...) "
      "rather than for...of with entity.get()."},
     {"form": "WHEN", "situation": "a query mixes tags or Not() with data "
      "traits", "text": "destructure only the data-bearing traits in "
      "updateEach / readEach; tags are left out of the tuple."},
     {"form": "DO NOT", "situation": "", "text": "destructure a tag trait "
      "in updateEach"}],
    phases=["implement", "debug", "refactor"],
    topics=["world.query", "updateEach", "readEach", "queryFirst"])
KOOTA_T = mk(
    "k_traits", "koota-traits-and-entities",
    "Use when defining koota traits or spawning koota entities in "
    "TypeScript: trait(), world.spawn, entity.set.",
    {"frameworks": ["koota"]},
    [{"form": "DO", "situation": "", "text": "define data with trait({ x: 0 "
      "}) and spawn with world.spawn(Position)."}],
    topics=["world.spawn", "entity.set", "trait()"])
REACT = mk("r_react", "react-auth-token-persistence",
           "Use when persisting a JWT or session token in a React SPA.",
           {"frameworks": ["react"], "languages": ["typescript"]},
           topics=["localStorage", "sessionStorage"])
RUST = mk("s_rust", "rust-memory-safety",
          "Use when writing memory safe systems code in Rust.",
          {"languages": ["rust"]})
SLIDES = mk("s_slides", "slide-decks", "Use when making a slide deck.",
            {"artifacts": ["slides"]})
TS_PLAIN = mk("t_plain", "typescript-plain",
              "Use when writing TypeScript.", {"languages": ["typescript"]})
REACT_ANY = mk("r_any", "react-render-lists",
               "Use when rendering lists of React components.",
               {"frameworks": ["react"]})
TS_DEBUG = mk("t_debug", "typescript-debugging",
              "Use when debugging TypeScript errors.",
              {"languages": ["typescript"]}, phases=["debug"])
POOL = [KOOTA_Q, KOOTA_T, REACT, RUST, SLIDES]

FALLBACK = {"calls": 0, "asked": [], "use": None}


def fake_fallback(system, user):
    import re
    FALLBACK["calls"] += 1
    ids = re.findall(r"^\| (\w+) \|", user, re.M)
    ids = [i for i in ids if i != "id"]
    FALLBACK["asked"].append(ids)
    return json.dumps({"use": ids if FALLBACK["use"] is None
                       else FALLBACK["use"], "why": "fake"})


def offline(cos=None):
    S._STICKY.clear()
    FALLBACK.update(calls=0, asked=[], use=None)
    S.ask_fallback = fake_fallback
    if cos is None:
        S.best_cosines = lambda q, pool: ({}, {"ok": False,
                                               "why": "offline"})
    else:
        S.best_cosines = lambda q, pool: (dict(cos), {"ok": True,
                                                      "why": "fake"})


def U(text: str) -> list[dict]:
    return [{"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": text}]


def step(msgs, name, args, result, i):
    cid = f"c{i}"
    return msgs + [{"role": "assistant", "content": "", "tool_calls": [
        {"id": cid, "type": "function", "function": {
            "name": name, "arguments": json.dumps(args)}}]},
        {"role": "tool", "tool_call_id": cid, "content": result}]


def rounds_of(rec):
    return {r["name"]: r for r in rec.get("rounds") or []}


# ============================================================ 1. rounds ==
def test_the_rounds_narrow_in_order():
    offline()
    _c, rec = S.select(U("Start a small shooter in TypeScript and use koota "
                         "for the ECS."), "code_generation",
                       POOL + [TS_PLAIN, TS_DEBUG, REACT_ANY])
    names = [r["name"] for r in rec["rounds"]]
    check(names == ["armed"] + list(S.ROUNDS),
          "[rounds] armed -> language -> framework -> phase -> situation -> "
          "not_given_or_faded, each logged", names)
    n = [r["survivors"] for r in rec["rounds"]]
    check(all(a >= b for a, b in zip(n, n[1:])),
          "[rounds] survivors never grow from one round to the next", n)
    r = rounds_of(rec)
    # language: rust and slides are not this request's; framework: React
    # is not open (neither asked nor shown by code), so both React skills
    # fall -- the one filed under TypeScript too (a HOST framework's skill
    # does not ride a TypeScript ask); phase: the debug-only TypeScript
    # skill is not now.
    check(r["language"]["survivors"] == 6 and r["language"]["how"] ==
          "pattern", "[rounds] language (pattern): the Rust and slides "
          "skills fall", r["language"])
    check(r["framework"]["survivors"] == 4,
          "[rounds] framework (pattern): both React skills fall (koota is "
          "asked, React is not open)", r["framework"])
    check(r["phase"]["survivors"] == 3,
          "[rounds] phase (pattern): the debug-only skill falls while "
          "implementing", r["phase"])
    check(r["not_given_or_faded"]["survivors"] == r["situation"][
        "survivors"] and "no conversation state" in r[
        "not_given_or_faded"]["why"],
          "[rounds] with no conversation state, nothing is given yet",
          r["not_given_or_faded"])


R3F_A = mk("r3f_a", "r3f-canvas-setup", "Use when setting up a React Three "
           "Fiber canvas.", {"frameworks": ["r3f"]})
REACT_REFS = mk("react_refs",
                "react-dev-learn-referencing-values-with-refs-ref-vs-state-2",
                "Use when choosing between a ref and state in React.",
                {"frameworks": ["react"]})
THREE_A = mk("three_a", "three-generic-lights", "Use when lighting a "
             "three.js scene.", {"frameworks": ["threejs"]})


def test_only_open_areas_pass_the_framework_round():
    """Coordinator, 2026-09-27: an area is open only when ASKED, shown by
    code (a fence, an import, a pin, a file), or named in an error line; an
    IMPLIED area contributes only the skills its implication row names."""
    offline()
    pool = [R3F_A, REACT_REFS, THREE_A, REACT_ANY]
    chosen, rec = S.select(U("Build it with react-three-fiber."),
                           "code_generation", pool)
    r = rounds_of(rec)
    ids = [c["id"] for c in chosen]
    check(r["framework"]["survivors"] == 2 and "react(implied)" in r[
        "framework"]["why"] and "threejs" not in r["framework"]["why"].split(
        "closed")[0],
          "[open] r3f asked: React and three.js are IMPLIED, not open -- "
          "only the named React skill passes; the generic React and three.js "
          "skills fall", r["framework"])
    check(ids == ["r3f_a", "react_refs"],
          "[open] ... and the named implication is still injected", ids)
    msgs = [{"role": "system", "content": "You write React and Python."},
            {"role": "user", "content": "Build it with react-three-fiber."}]
    _c, rec = S.select(msgs, "code_generation", pool)
    closed = rounds_of(rec)["framework"]["why"].split("closed")[-1]
    check("react(word)" in closed and "python(word)" in closed
          and rounds_of(rec)["framework"]["survivors"] == 2,
          "[open] a framework WORD in the system text opens nothing",
          rounds_of(rec)["framework"]["why"])
    _c, rec = S.select(U("```ts\nimport { Canvas } from '@react-three/fiber'"
                         "\nimport * as THREE from 'three'\n```\nWhy is "
                         "it black?"), "code_edit", pool)
    check("threejs(import)" in rounds_of(rec)["framework"]["why"]
          and rounds_of(rec)["framework"]["survivors"] >= 2,
          "[open] an IMPORT opens three.js", rounds_of(rec)["framework"])
    sig = C.request_signals(U("TypeError: THREE.Mesh is not a constructor "
                              "(three.js)"), "prose")
    opened, _closed = S.open_areas(sig)
    check("threejs" in opened, "[open] a name in an error line (or asked) "
          "opens its area", opened)


def test_vector_search_only_where_patterns_are_silent():
    """No absolute cosine cut since 2026-09-27 (EMB_HIGH): a silent round
    keeps only what carries its own proof, and the vector search admits
    nothing -- no CATEGORY question from embedding-only candidates."""
    offline(cos={"s_rust": 0.9, "k_queries": 0.2})
    chosen, rec = S.select(U("memory safe systems guidance"), "prose", POOL)
    r = rounds_of(rec)
    q = rec["questions"]
    check(r["language"]["how"] == "silent" and r["language"][
        "survivors"] == 0,
          "[vector] no language, framework or artifact in the patterns: "
          "the silent round admits nothing on a cosine", r["language"])
    check(not q and not chosen and not FALLBACK["asked"],
          "[category] an embedding-only match is no candidate: no category "
          "question, no fallback call", (q, FALLBACK["asked"]))
    offline(cos={"s_rust": 0.9})
    _c, rec = S.select(U("```ts\nconst x: number = 1\n```\nAdd a field."),
                       "code_edit", POOL + [TS_PLAIN])
    r = rounds_of(rec)
    check(r["language"]["how"] == "pattern"
          and not any(x["area"] == "rust" for x in rec["questions"])
          and FALLBACK["calls"] == 0,
          "[vector] patterns found (a ts fence): the vector search is not "
          "consulted and the high-cosine Rust skill is not a candidate",
          (r["language"], rec["questions"]))


def test_a_negated_name_places_nothing():
    offline()
    msgs = U("Write a tiny canvas game in vanilla JavaScript without React: "
             "a bouncing ball.")
    sig = C.request_signals(msgs, "code_generation")
    check("react" in sig["negated"] and "react" not in sig["asked"],
          "[negation] 'without React' is negated, not asked", sig["negated"])
    chosen, rec = S.select(msgs, "code_generation", POOL)
    check(not chosen and not any(q["area"] == "react"
                                 for q in rec["questions"])
          and FALLBACK["calls"] == 0,
          "[negation] ... so no React question and no fallback about it",
          rec["questions"])


# ========================================================= 2. questions ==
def test_one_question_per_area_capped_with_none():
    offline()
    many = [mk(f"k{i}", f"koota-skill-{i}", f"Use when koota thing {i} is "
               "needed in an ECS world.", {"frameworks": ["koota"]})
            for i in range(8)]
    chosen, rec = S.select(U("Build the game with koota for the ECS."),
                           "code_generation", many)
    qs = rec["questions"]
    areas = [q["area"] for q in qs]
    check(len(areas) == len(set(areas)), "[questions] one question per "
          "area", areas)
    q = next((x for x in qs if x["area"] == "koota"), None)
    check(q and len(q["options"]) == D.MAX_OPTIONS
          and q["options"][-1] == D.NONE and q["kind"] == "asked",
          f"[questions] 8 survivors: the question shows the best "
          f"{D.MAX_OPTIONS - 1} (pre-ranked) and NONE last", q)
    check(q and abs(sum(q["probabilities"].values()) - 1) < 1e-3
          and q["pick"] == chosen[0]["id"] and q["decider"] == "stub"
          and rec["decider"] == "stub",
          "[questions] probabilities over the set sum to 1; the pick is "
          "what was injected; the decider is recorded", q)
    blob = json.dumps(rec["questions"])
    check("Build the game" not in blob and all(
        isinstance(x.get("state_chars"), int) for x in qs),
          "[questions] the record keeps ids, numbers and the state's LENGTH "
          "-- never the request's text", blob[:200])


def test_an_asked_area_the_evidence_cannot_pick_is_one_question():
    offline()
    # A host area (TypeScript) named in passing: its skills are ranked but
    # none is about the request -- they join the area's one question.
    chosen, rec = S.select(U("Start a shooter in TypeScript and use koota "
                             "for the ECS."), "code_generation",
                           POOL + [TS_PLAIN])
    areas = [q["area"] for q in rec["questions"]]
    check(areas.count("typescript") <= 1 and areas.count("koota") == 1
          and "typescript" in (rec.get("asked_unfilled") or []),
          "[questions] the unpicked host ask is asked at most once, and "
          "recorded unfilled", (areas, rec.get("asked_unfilled")))


# ========================================================== 3. deciders ==
def _opts(spec):
    return [D.Option(i, t, dict(ev)) for i, t, ev in spec]


def test_the_deciders():
    stub = D.EvidenceStub()
    o = _opts([("a", "alpha", {"tier": "inject", "prior": 3.0}),
               ("b", "beta", {"tier": "inject", "prior": 5.0}),
               ("c", "gamma", {"tier": "no", "prior": 9.0})])
    p = stub.decide("x", o)
    check(D.pick(p) == "b" and p["c"] == 0.0 and p[D.NONE] < p["b"],
          "[stub] the best ELIGIBLE option wins; an ineligible one never "
          "does, however it ranks", p)
    p = stub.decide("x", _opts([("c", "gamma", {"tier": "no"})]))
    check(D.pick(p) is None and p[D.NONE] == 1.0,
          "[stub] nothing eligible: NONE", p)
    check(stub.decide("x", _opts([("c", "g", {"tier": "ask"})])) is None,
          "[stub] only unconfirmed ask-tier options: it ABSTAINS (a model "
          "must settle them)")
    lex = D.LexicalStub()
    st = "the koota world query with updateEach throws"
    o = _opts([("q", "querying a koota world with updateEach", {}),
               ("t", "defining traits and spawning entities", {}),
               ("s", "making slide decks", {})])
    p = lex.decide(st, o)
    check(D.pick(p) == "q", "[lexical] text only: the option whose words "
          "the state carries wins", p)
    p2 = lex.decide("the weather is nice", o)
    check(D.pick(p2) is None, "[lexical] no option's words in the state: "
          "NONE", p2)
    rnd = random.Random(7)
    for dec, state, opts in ((stub, "x", _opts([
            ("a", "alpha", {"tier": "inject", "prior": 3.0}),
            ("b", "beta", {"tier": "inject", "prior": 5.0}),
            ("c", "gamma", {"tier": "inject", "prior": 5.0})])),
            (lex, st, o)):
        base = dec.decide(state, opts)
        same = True
        for _ in range(5):
            sh = list(opts)
            rnd.shuffle(sh)
            got = dec.decide(state, sh)
            same = same and got == base and D.pick(got) == D.pick(base)
        check(same, f"[{dec.name}] order-invariant: shuffling the options "
              "changes no probability and not the pick", base)
    check(D.pick({"a": 0.4, D.NONE: 0.4, "b": 0.2}) is None,
          "[pick] a tie with NONE is not a pick")


# ======================================================= 4. statechart ==
def test_the_transition_table():
    t = CH.transition
    check(t("unseen", "BODY") == "given" and t("unseen", "RECALLED") is None
          and t("unseen", "ERROR") is None,
          "[chart] unseen: only BODY (and COMPACTED) is legal")
    check(t("given", "ERROR") == "recall_eligible"
          and t("given", "ERROR", {"cooling": True}) == "recall_eligible"
          and t("given", "EVIDENCE") is None,
          "[chart] given: an error makes a recall eligible (no cooldown "
          "since 2026-09-27); evidence alone does not bring it back")
    check("faded" not in CH.AREA_STATES and t("given", "GROW") is None,
          "[chart] no fade since 2026-09-27: no faded state, no GROW")
    check(t("recall_eligible", "RECALLED") == "given"
          and t("recall_eligible", "SETTLE", {"before": "given"}) == "given"
          and all(t(s, "COMPACTED") == "unseen" for s in CH.AREA_STATES),
          "[chart] RECALLED -> given; SETTLE returns; COMPACTED -> unseen")
    check(CH.legal_events("unseen") == ["BODY", "COMPACTED"]
          and "RECALLED" not in CH.legal_events("given"),
          "[chart] legal_events lists what a decide could choose among",
          CH.legal_events("given"))


def test_the_chart_gates_the_questions():
    offline()
    st = None
    m = U("Start a small shooter in TypeScript and use koota for the ECS.")
    _t, r0, st = S.decide(m, "code_generation", POOL, [], st, key="q0")
    ch = r0["chart"]
    check(ch["state"] == "implement" and ch["areas"].get("koota") ==
          "given" and any(x["event"] == "BODY" and x["area"] == "koota"
                          for x in ch["transitions"]),
          "[chart] turn 0: implement; koota unseen --BODY--> given", ch)
    m = step(m, "write_file", {"path": "src/systems/bullets.ts", "content":
             "import { trait } from 'koota'\nexport function move(world, d)"
             " {\n  world.query(IsBullet, Position, Velocity).updateEach(("
             "[b, pos, vel]) => { pos.y -= vel.y * d })\n}\n"},
             '{"bytes_written": 200}', 1)
    _t, r1, st = S.decide(m, "agent_step", POOL, [], st, key="q1")
    err = ("Uncaught TypeError: Cannot read properties of undefined "
           "(reading 'y')\n    at updateEach (koota.js:1:5210)\n"
           "world.query(IsBullet, Position, Velocity).updateEach(([b, pos,"
           " vel]) => ...)")
    m = step(m, "terminal", {"command": "npm run dev"}, err, 2)
    t2, r2, st = S.decide(m, "agent_step", POOL, [], st, key="q2")
    tr = [(x["from"], x["event"], x["to"]) for x in r2["chart"][
        "transitions"] if x.get("area") == "koota"]
    check(("given", "ERROR", "recall_eligible") in tr
          and ("recall_eligible", "RECALLED", "given") in tr
          and "debug" in str(r2["chart"]["state"])
          and "updateEach" in t2 and "Remember (craft" not in t2,
          "[chart] an error step: koota given --ERROR--> recall_eligible "
          "--RECALLED--> given; the top state is debug (every phase the "
          "evidence raises)", (tr, r2["chart"]["state"]))
    m = step(m, "terminal", {"command": "npm run dev"}, err, 3)
    t3, r3, st = S.decide(m, "agent_step", POOL, [], st, key="q3")
    check(not t3 and any("same item" in x["why"] for x in r3["skipped"]),
          "[chart] the same error again: legal (no cooldown), but the "
          "identical items are never sent twice in a row (the injector)",
          (r3["questions"], r3["skipped"]))
    check(json.loads(json.dumps(st))["chart"]["areas"].get("koota") ==
          "given" and st["chart"]["skill_area"].get("k_queries") == "koota",
          "[chart] the chart is part of the conversation's skill state (the "
          "ledger row), JSON through and through", st.get("chart"))
    short = U("Continue the shooter; use koota for the ECS.")
    _t, r4, st = S.decide(short, "code_generation", POOL, [], st, key="q4",
                          compactions=1)
    check(st.get("compacted") and any(x["event"] == "BODY"
                                      for x in r4["chart"]["transitions"])
          and any(d["form"] == "body" for d in r4["stage1"]),
          "[chart] a compaction: every area back to unseen, the next need "
          "is a body again", r4["chart"])
    view = CH.View(st, chars=10, req=99, phase_now="implement",
                   phase_changed=False, first=False,
                   area_of=lambda s: S.area_of(s.get("rule") or {}))
    sid = next(iter(st["given"]))
    s0 = next(x for x in POOL if x["id"] == sid)
    trig, form, why = view.option(s0, "evidence", fresh_hit=False)
    check(trig is None and form is None,
          "[chart] a given skill with no new evidence: not a legal option",
          why)


# =================================================== 5. served template ==
def test_through_the_served_template():
    saved = (skills.armed, proxy._skills_tail, S.best_cosines,
             S.ask_fallback)
    skills.armed = lambda *a, **k: list(POOL)
    proxy._skills_tail = proxy._skills_tail_real
    S.best_cosines = lambda q, pool: ({}, {"ok": False, "why": "offline"})
    S.ask_fallback = lambda s, u: json.dumps({"use": [], "why": "offline"})
    S._STICKY.clear()
    try:
        T.slots.reset(n=4)
        T.compaction.reset()
        c = T.Client("medium")
        c.msgs[0]["content"] += " [skill questions]"
        w = {"path": "src/systems/bullets.ts", "content":
             "import { trait } from 'koota'\nexport function move(world, d)"
             " {\n  world.query(IsBullet, Position, Velocity).updateEach(("
             "[b, pos, vel]) => { pos.y -= vel.y * d })\n}\n"}
        t1 = c.turn([T.reply("", calls=[T.call("write_file", w, "w1")])],
                    user="Start a small shooter in TypeScript and use koota "
                         "for the ECS.")
        x1 = (t1["d"]["x_yamadori"].get("skills") or {})
        check([r["name"] for r in x1.get("rounds") or []] ==
              ["armed"] + list(S.ROUNDS)
              and any(q["area"] == "koota" for q in x1.get("questions") or [])
              and (x1.get("chart") or {}).get("areas", {}).get("koota")
              == "given" and x1.get("decider") == "stub",
              "[served] x_yamadori.skills: rounds, questions, decider, chart",
              json.dumps({k: x1.get(k) for k in ("rounds", "questions",
                                                 "chart")})[:300])
        c.tool_result("w1", '{"bytes_written": 200}')
        w2 = {"path": "README.md", "content": "# Shooter\n"}
        t2 = c.turn([T.reply("", calls=[T.call("write_file", w2, "w2")])])
        check(T._extends(t1, t2, "[served] step 1"),
              "[served] the step after the write extends the slot")
        c.tool_result("w2", "Uncaught TypeError: Cannot read properties of "
                      "undefined (reading 'y')\n    at updateEach (koota.js:"
                      "1:5210)\nworld.query(IsBullet, Position, Velocity)."
                      "updateEach(([b, pos, vel]) => ...)")
        t3 = c.turn([T.reply("Fixed the bullets query.")])
        check(T._extends(t2, t3, "[served] the error step"),
              "[served] the error step extends the slot")
        x3 = (t3["d"]["x_yamadori"].get("skills") or {})
        req3 = t3["gens"][0]["request"]["messages"]
        last_tool = [m for m in req3 if m.get("role") == "tool"][-1]
        tr = [(x["from"], x["event"], x["to"]) for x in (x3.get("chart") or {})
              .get("transitions") or [] if x.get("area") == "koota"]
        check("updateEach" in (last_tool.get("content") or "")
              and "Remember (craft" not in (last_tool.get("content") or "")
              and ("given", "ERROR", "recall_eligible") in tr
              and ("recall_eligible", "RECALLED", "given") in tr,
              "[served] the recall is appended to the error's tool result, "
              "and the chart the LEDGER carried moved given -> "
              "recall_eligible -> given", (tr, (last_tool.get("content")
                                                or "")[-200:]))
        t4 = c.turn([T.reply("Glad it works.")], user="Thanks, that works.")
        check(T._extends(t3, t4, "[served] the next user turn"),
              "[served] the next user turn extends the slot (the recall is "
              "replayed byte for byte)")
    finally:
        (skills.armed, proxy._skills_tail, S.best_cosines,
         S.ask_fallback) = saved


def main() -> int:
    for fn in (test_the_rounds_narrow_in_order,
               test_only_open_areas_pass_the_framework_round,
               test_vector_search_only_where_patterns_are_silent,
               test_a_negated_name_places_nothing,
               test_one_question_per_area_capped_with_none,
               test_an_asked_area_the_evidence_cannot_pick_is_one_question,
               test_the_deciders,
               test_the_transition_table, test_the_chart_gates_the_questions,
               test_through_the_served_template):
        print(f"\n--- {fn.__name__} ---")
        try:
            fn()
        except Exception as e:                                   # noqa: BLE001
            traceback.print_exc()
            check(False, f"{fn.__name__} itself raised",
                  f"{type(e).__name__}: {e}")
    fails = 0
    for ok, name, detail in _results:
        print(f"  {'pass' if ok else 'FAIL'}  {name}"
              + ("" if ok else f"   <- {detail}"))
        fails += not ok
    n = len(_results)
    print(f"\n  {n - fails}/{n} checks passed")
    return 1 if fails else 0


if __name__ == "__main__":
    # jjava is always in scope where skills serve (operator, 2026-09-29):
    # offline, the STUBBED decider answers (mcp/decider_stub.py).
    import decider_stub
    with decider_stub.installed():
        sys.exit(main())
