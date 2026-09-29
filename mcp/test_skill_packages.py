#!/usr/bin/env python
"""The package rule (mcp/skill_packages.py). No GPU, no network; the held
package sources and the TypeScript lib files are read, nothing is written.

    python mcp/test_skill_packages.py      -> "N/M checks passed"

  1. EXTRACTION: export statements (a chunk's `a as b` aliases resolved to
     the declared name), members only inside `{ }` bodies (a tuple's labels
     and private members are not members).
  2. THE VOCABULARY on the held sources: a symbol detects a package only
     when exactly one package exports it and it is neither a platform API
     nor a plain English word.
  3. DETECTION: names (with the version and the R3F v10 -> TSL
     implication), negation, imports, pins, symbols in code-like context, a
     file extension is not a member, a config file's keys are no symbols,
     our own craft block is not evidence.
  4. THE RULE: the body the first time, nothing again without an event, a
     recall on ASKED or ERROR; the canonical skill by major version.
  5. THE MAPPING: every canonical skill is armed in (a copy of) the live
     store and within the body cap.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import offline_stores  # noqa: E402
_TMP = offline_stores.isolate("yamadori_test_skill_packages_")
_LIVE_JOBS = os.path.join(HERE, "..", "index", "jobs.sqlite3")
if os.path.exists(_LIVE_JOBS):
    shutil.copyfile(_LIVE_JOBS, os.environ["YAMADORI_JOBS_DB"])
os.environ["YAMADORI_SKILLS_DIR"] = os.path.join(HERE, "..", "index",
                                                 "skills")
os.environ.setdefault("YAMADORI_GPU_ROOM", "0")

import skill_packages as SP  # noqa: E402

_results: list[tuple[bool, str, str]] = []


def check(ok, name, detail=""):
    _results.append((bool(ok), name, str(detail)[:300]))


def user(text):
    return [{"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": text}]


def ev(text):
    return {p: sorted(f"{w['how']}:{w['what']}" for w in e["why"])
            for p, e in SP.detect(user(text)).items()}


# ======================================================== 1. extraction ==
def test_extraction():
    chunk = ("declare function f(): void;\ntype TraitValue = number;\n"
             "export { f as a, type TraitValue as P, g as useThing };")
    got = SP.exports_of(chunk)
    check({"f", "TraitValue", "useThing"} <= got and not {"a", "P"} & got,
          "[extract] a chunk's short aliases resolve to its declared names; "
          "an alias of a name declared elsewhere is the public name", got)
    got = SP.exports_of("export declare function createWorld(): World;\n"
                        "export const trait = 1;\nexport * as ns from './x'")
    check({"createWorld", "trait", "ns"} <= got, "[extract] export "
          "declarations and namespace re-exports", got)
    dts = ("export type Mat4 = [\n    e1: number,\n    e2: number\n];\n"
           "export interface Query {\n    updateEach(cb: Fn): void;\n"
           "    private secret: number;\n    readonly size: number;\n}\n")
    got = SP.members_of(dts)
    check(got == {"updateEach", "size"}, "[extract] members come from `{ }` "
          "bodies only: no tuple labels, no private members", got)


# ======================================================== 2. vocabulary ==
def test_vocabulary():
    v = SP.vocabulary()
    s = v["symbols"]
    want = {"updateEach": "koota", "readEach": "koota",
            "createWorld": "koota", "useFrame": "@react-three/fiber",
            "useThree": "@react-three/fiber", "colorNode": SP.TSL,
            "MeshStandardNodeMaterial": SP.TSL, "WebGPURenderer": SP.TSL,
            "scaleAndAdd": "math", "mulberry32": "math",
            "useGLTF": "@react-three/drei"}
    for n, p in want.items():
        check(s.get(n) == p, f"[vocab] {n} detects {p}", s.get(n))
    for n in ("world", "query", "get", "set", "vec3", "uniform"):
        check(n not in s, f"[vocab] {n} is shared or plain: no detector",
              s.get(n))
    for n in ("Position", "texture", "position", "Canvas"):
        check(n not in s, f"[vocab] {n} is a plain English word: no "
              "detector", s.get(n))
    for n in ("addEventListener", "requestAnimationFrame", "Float32Array"):
        check(n not in s, f"[vocab] {n} is a platform API: no detector",
              s.get(n))
    check("@webgpu/types" not in v["packages"], "[vocab] a platform type "
          "package is never in play", v["packages"])
    check(not any(p.startswith("(ref) ") for p in s.values()),
          "[vocab] a reference library (React's types) never owns a symbol")


# ========================================================= 3. detection ==
def test_detection():
    got = SP.detect(user("Build it with r3f (react-three-fiber) v10 and "
                         "Koota and pmndrs math."))
    check(set(got) == {"@react-three/fiber", "koota", "math", SP.TSL},
          "[detect] the operator's sentence: fiber, koota, math, and TSL "
          "by the v10 implication", sorted(got))
    check(got.get("@react-three/fiber", {}).get("version") == "10",
          "[detect] the version written after the name",
          got.get("@react-three/fiber"))
    check(got.get(SP.TSL, {}).get("why", [{}])[0].get("how") == "implied",
          "[detect] TSL is in play by implication", got.get(SP.TSL))
    check(ev("Rewrite it without koota, using plain arrays.") == {},
          "[detect] a negated name puts nothing in play",
          ev("Rewrite it without koota, using plain arrays."))
    check(ev("the particles stutter in useFrame") ==
          {"@react-three/fiber": ["symbol:useFrame"]},
          "[detect] a code-shaped name in prose is code-like context",
          ev("the particles stutter in useFrame"))
    got = ev("```ts\nimport { uv } from 'three/tsl'\nimport { fbm } from "
             "'math/noise'\n```")
    check(got.get(SP.TSL) == ["import:three/tsl"] and "import:math/noise"
          in got.get("math", []), "[detect] imports, subpaths included",
          got)
    got = ev('my dependencies:\n```json\n{"dependencies": {"koota": '
             '"^0.6.6"}}\n```')
    check(got.get("koota") == ["pin:koota@0.6.6"], "[detect] a dependency "
          "pin", got)
    check(ev("My <Canvas> goes black") == {"@react-three/fiber":
                                           ["symbol:Canvas"]},
          "[detect] a word-shaped API name in code position (`<Canvas`) "
          "counts", ev("My <Canvas> goes black"))
    check(ev("draw it on a canvas with a bloom look") == {},
          "[detect] the same words in prose do not",
          ev("draw it on a canvas with a bloom look"))
    got = ev("```ts\nconst tree = 1 // the main feature\nfor (const inner "
             "of rows) {}\n```")
    check(got == {}, "[detect] a word-shaped local name or a comment in "
          "code is no use of a package", got)
    check(ev("HUD: score (top left), health bar (bottom)") == {},
          "[detect] a parenthesis after a word is not a call",
          ev("HUD: score (top left), health bar (bottom)"))
    check(ev("Chart.js tooltip shows undefined") == {},
          "[detect] `.js` ending a word is a file extension, not a member",
          ev("Chart.js tooltip shows undefined"))
    msgs = user("go on") + [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {
                "name": "read_file",
                "arguments": '{"path": "tsconfig.json"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content":
         '{"compilerOptions": {"moduleResolution": "bundler", '
         '"noEmit": true, "skipLibCheck": true}}'}]
    got = SP.detect(msgs)
    check("type-fest" not in got, "[detect] a config file's keys are no "
          "symbols (tsconfig's moduleResolution is not type-fest's)", got)
    ours = ("The bug again.\n\n---\n" + _craft_header()
            + "\n\nworld.query(A).updateEach(([a]) => a)")
    got = SP.detect(user(ours))
    check("koota" not in got, "[detect] our own craft block is not "
          "evidence of the work", got)
    step = user("fix the build") + [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c2", "type": "function", "function": {
                "name": "terminal", "arguments": '{"command": "npm run '
                                                 'build"}'}}]},
        {"role": "tool", "tool_call_id": "c2", "content":
         "src/m.ts:14:5 - error TS2339: Property 'colorNode' does not exist "
         "on type 'MeshBasicMaterial'."}]
    got = SP.detect(step)
    check(SP.TSL in got and "ERROR" in got[SP.TSL]["events"],
          "[detect] a unique symbol in an error line, with the ERROR event",
          got.get(SP.TSL))


def _craft_header():
    try:
        import skill_prompts
        return skill_prompts.CRAFT_HEADER
    except Exception:                                            # noqa: BLE001
        return "Craft from this service's library that fits this work."


# ============================================================== 4. rule ==
def test_rule():
    first = SP.detect(user("Build it with koota."))
    dec, st = SP.decide(first, None)
    check([(d["skill"], d["form"]) for d in dec] ==
          [("koota-traits-and-entities", "body")], "[rule] the body the "
          "first time a package is in play", dec)
    step = SP.detect(user("```ts\nworld.query(A).updateEach(([a]) => a)\n"
                          "```"))
    for p in step.values():
        p["events"] = []
    dec, st = SP.decide(step, st)
    check([d["form"] for d in dec] == ["none"], "[rule] nothing again "
          "without a recall event", dec)
    dec, st = SP.decide(SP.detect(user("the koota world is empty again")), st)
    check([d["form"] for d in dec] == ["recall"], "[rule] a RECALL when "
          "the user names it again (ASKED)", dec)
    dec2, _ = SP.decide({"koota": {"why": [{"how": "symbol", "what": "x",
                                            "where": "error"}],
                                   "version": None, "events": ["ERROR"]}},
                        st)
    check([d["form"] for d in dec2] == ["recall"], "[rule] a RECALL on an "
          "error line naming it (ERROR)", dec2)
    dec3, _ = SP.decide({"koota": {"why": [], "version": None,
                                   "events": []}}, st, phase_changed=True)
    check([d["form"] for d in dec3] == ["recall"], "[rule] a RECALL on a "
          "phase change (PHASE)", dec3)
    check(SP.canonical_skill("@react-three/fiber", "9.8.0") ==
          "r3f-v9-setup-22" and SP.canonical_skill("@react-three/fiber",
                                                    None) ==
          "r3f-v10-setup-21", "[rule] the canonical skill of the major in "
          "play; the newest held with no version")
    dec, _ = SP.decide({"three-mesh-bvh": {"why": [], "version": None,
                                           "events": []}}, None)
    check(dec and dec[0]["form"] == "none" and dec[0]["skill"] is None,
          "[rule] a package with no canonical skill injects nothing", dec)


# =========================================================== 5. mapping ==
def test_mapping():
    import skill_limits as L
    import skills
    pool = skills.armed()
    if not pool:
        check(True, "[mapping] no store copy here: not checked")
        return
    by = {s["name"]: s for s in pool}
    cap = getattr(L, "SKILL_TOKENS_HARD", None)
    for pkg, row in SP.CANONICAL.items():
        for name in [row["skill"]] + list((row.get("by_major") or {})
                                          .values()):
            s = by.get(name)
            check(s is not None, f"[mapping] {pkg}: {name} is armed")
            if s is not None and cap:
                check(L.tokens(s["body"]) <= cap, f"[mapping] {name} is "
                      f"within the body cap ({cap})", L.tokens(s["body"]))


# ========================================================= 6. the plan ==
def _sk(sid, name, fw, desc="Use when x.", lead=None):
    return {"id": sid, "name": name, "version": 1, "description": desc,
            "title": name, "body": desc, "items": [],
            "rule": {"applies_to": {"frameworks": [fw] if fw else [],
                                    "languages": ["typescript"],
                                    "artifacts": ["code"]}},
            "lead_for": lead}


def _area(rule):
    a = (rule or {}).get("applies_to") or {}
    return (a.get("frameworks") or a.get("languages") or ["code"])[0]


def test_plan():
    import skill_match as M
    pool = [_sk("k1", "koota-lead", "koota", lead="koota"),
            _sk("k2", "koota-queries", "koota"),
            _sk("k3", "koota-react", "koota"),
            _sk("m1", "math-lead", "pmndrs_math", lead="math"),
            _sk("r1", "r3f-lead", "r3f", lead="@react-three/fiber"),
            _sk("t1", "tsl-lead", "threejs", lead="three/tsl"),
            _sk("g1", "generic-ts", None)]
    by = {s["id"]: s for s in pool}
    leads = {"koota": "koota-lead", "math": "math-lead",
             "@react-three/fiber": "r3f-lead", "three/tsl": "tsl-lead"}

    def play(*pk, events=("ASKED",)):
        return {p: {"why": [{"how": "name", "what": p, "where": "user"}],
                    "version": None, "events": list(events),
                    "strength": "strong"} for p in pk}

    def run(pp, old=(), state=None, given=(), cos=None):
        return M.plan_turn(pool=pool, old=list(old), pkg_play=pp,
                           pkg_state=state, given=set(given), cos=cos,
                           legal_whys={}, phase_changed=False,
                           leads_for=leads.get, area_of=_area,
                           subject_areas=lambda s: set())
    out, rec, st = run(play("koota", "math", "@react-three/fiber",
                            "three/tsl"))
    forms = [(p["skill"]["name"], p["form"]) for p in out]
    check([f for n, f in forms].count("body") == M.BODIES_PER_DECISION
          and ("tsl-lead", "index") in forms, "[plan] at most "
          f"{M.BODIES_PER_DECISION} bodies per decision (SkillsBench "
          "2602.12670v4); the rest go as index lines", forms)
    q = next(q for q in rec["questions"] if q["area"] == "koota")
    none = q["options"][-1].split(")")[0]
    check(q["hard"] and none in q["rounds"][0]["excluded"]
          and q["picks"][0] == "k1",
          "[plan] a named package is HARD: its first round excludes 'none', "
          "and its lead goes first on its first appearance", q["rounds"])
    import skill_match as M2
    cands = [{"id": f"s{i}", "name": f"s{i}", "description": f"d{i}"}
             for i in range(4)]
    seen = []

    def ask(area, options, turn):
        seen.append((turn["prompt"]["prefix"], turn["question"],
                     list(turn["exclude"])))
        dist = {o["letter"]: 1.0 / len(options) for o in options}
        dist["A"] = 0.9      # the first option, already chosen after round 1
        dist["B"] = 0.05 if len(seen) == 1 else 0.3
        return "A", dist
    res = M2.ask_rounds("any", cands, {}, False, ask, max_picks=2)
    check(len(seen) == 2 and seen[0][0] == seen[1][0]
          and "Already chosen: A" in seen[1][1] and seen[1][2] == ["A"]
          and res["picks"] == ["s0", "s1"]
          and res["rounds"][1]["tokens"]["reused"] ==
          res["tokens"]["prefix"] > 0,
          "[plan] one option list for every round (the prefix is reused); "
          "the question names what is chosen; chosen labels are ignored and "
          "the rest renormalised", (seen, res["rounds"]))
    # Deploy check 2026-09-27 (h5): the doubles AA..ZZ are single tokens
    # but the served model answers them with their single letter ("None
    # of these" as ZZ read as " Z" 0.371, an unrelated skill, vs " ZZ"
    # 0.037): every request injected three unrelated bodies. A-Z only, so
    # no label is another label's prefix.
    check(M2.LABELS == tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
          and not any(a != b and b.startswith(a) for a in M2.LABELS
                      for b in M2.LABELS),
          "[plan] single-letter labels only (A-Z): no label is another's "
          "prefix", M2.LABELS)
    check(M2.QUESTION_CANDIDATES == len(M2.LABELS) - 1 == 25
          and M2.QUESTION_CANDIDATES <= M2.CANDIDATES_PER_QUESTION,
          "[plan] a question carries at most 25 candidates (the labels' "
          "bound under SRA's 50) plus 'none'", M2.QUESTION_CANDIDATES)
    big = [_sk(f"x{i:02d}", f"extra-{i:02d}", None) for i in range(60)]
    seen_opts = []

    def ask_none(area, options, turn):
        seen_opts.append([o["letter"] for o in options])
        none_l = next(o["letter"] for o in options if o["id"] is None)
        return none_l, {o["letter"]: (1.0 if o["letter"] == none_l else 0.0)
                        for o in options}
    res_big = M2.ask_rounds("any", [M2._cand(s_, None) for s_ in
                                    big[:M2.QUESTION_CANDIDATES]],
                            {}, False, ask_none, max_picks=3)
    check(res_big["picks"] == [] and seen_opts
          and seen_opts[0][-1] == "Z" and len(seen_opts[0]) == 26
          and all(len(x) == 1 for x in seen_opts[0]),
          "[plan] a full question: 25 candidates A-Y, 'None of these' is Z, "
          "and a 'none' answer injects nothing", (seen_opts, res_big))
    pool_big = pool + big
    out_b, rec_b, _ = M2.plan_turn(
        pool=pool_big, old=[], pkg_play={}, pkg_state=None, given=set(),
        cos={s_["id"]: 1.0 - k / 1000 for k, s_ in enumerate(pool_big)},
        legal_whys={}, phase_changed=False, leads_for=leads.get,
        area_of=_area, subject_areas=lambda s_: set())
    mq_b = [q_ for q_ in rec_b["questions"] if q_["kind"] == "match"]
    check(mq_b and len(mq_b[0]["candidates"]) == 25
          and rec_b["bounds"]["question_candidates"] == 25,
          "[plan] retrieval over 60+ skills asks about the top 25 only",
          [len(q_["candidates"]) for q_ in mq_b])
    old = [{"skill": by["k2"], "form": "body", "trigger": "new_area"}]
    out2, rec2, st2 = run(play("koota", events=()), old=old, state=st,
                          given={"k1"})
    names = [p["skill"]["name"] for p in out2]
    q2 = rec2["questions"][0]
    check(names == ["koota-queries"] and "koota-lead" not in
          [c["name"] for c in q2["candidates"]], "[plan] later, the turn's "
          "evidence picks among the package's skills; the lead is not a "
          "candidate again", (names, q2["candidates"]))
    cos = {s["id"]: 0.5 for s in pool}
    out3, rec3, _ = run({}, cos=cos)
    mq = [q for q in rec3["questions"] if q["kind"] == "match"]
    cands = [c["name"] for q in mq for c in q["candidates"]]
    check("generic-ts" in cands and not any(n.endswith("-lead")
                                            for n in cands),
          "[plan] retrieval ranks every non-package skill; a lead skill is "
          "never a general candidate", cands)
    check(rec3["bounds"]["candidates_per_question"] == 50
          and "2604.24594" in rec3["bounds"]["source"],
          "[plan] the candidate bound is sourced (SRA's top 50)",
          rec3["bounds"])


def test_detection_strength():
    got = SP.detect(user("the particles stutter in useFrame"))
    check(got["@react-three/fiber"]["strength"] == "weak",
          "[strength] a symbol read only in prose is WEAK (the decider "
          "confirms it)", got)
    got = SP.detect(user("```ts\nuseFrame(() => {})\n```"))
    check(got["@react-three/fiber"]["strength"] == "strong",
          "[strength] the same symbol in code is strong", got)
    msgs = user("go on") + [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "w", "type": "function", "function": {
                "name": "write_file", "arguments": '{"path": "src/r.ts", '
                '"content": "export function rand() { return 4 }"}'}}]},
        {"role": "tool", "tool_call_id": "w", "content": "ok"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "x", "type": "function", "function": {
                "name": "write_file", "arguments": '{"path": "src/m.ts", '
                '"content": "const v = rand()"}'}}]},
        {"role": "tool", "tool_call_id": "x", "content": "ok"}]
    check(SP.TSL not in SP.detect(msgs), "[strength] a name the project "
          "declares itself (its own rand()) is not a package's symbol",
          SP.detect(msgs))


def test_index_upkeep():
    import numpy as np
    import code_search
    import skill_match as M
    pool = [_sk("a1", "alpha", None, "Use when alpha."),
            _sk("b1", "beta", None, "Use when beta.")]
    for s in pool:
        s["rule"]["triggers"] = [{"text": s["description"]}]
    saved = code_search.embed
    code_search.embed = lambda texts, is_query=False: np.ones(
        (len(texts), 4), dtype=np.float32) / 2.0
    try:
        M._DOC.update(sig=None, rows=[], mat=None)
        st = M.index_state(pool)
        check(not st["fresh"], "[index] no file: not fresh", st)
        M.doc_index(pool, build=True)
        st = M.index_state(pool)
        check(st["fresh"] and st["documents"] == 6, "[index] built: fresh, "
              "each skill's trigger text, title and body (2 x 3 documents)",
              st)
        st = M.index_state(pool + [_sk("c1", "gamma", None)])
        check(not st["fresh"] and "stale" in st["why"], "[index] a skill "
              "armed since: stale", st)
    finally:
        code_search.embed = saved
    import worker
    check(worker.HANDLERS.get(M.QUEUE) is M.handle_build and M.LANE ==
          "gpu", "[index] the worker runs the rebuild on the gpu lane")
    jid = M.schedule()
    again = M.schedule()
    check(jid and again is None, "[index] a stale index enqueues ONE "
          "rebuild; none while one is queued", (jid, again))


def test_imports_decide():
    fence = "```"
    got = ev(fence + 'ts\nimport { createWorld } from "bitecs"\n'
             'const world = createWorld()\n' + fence)
    check(got == {}, "[imports] a name imported from ANOTHER module (bitecs"
          "'s createWorld) never counts for a held package (koota)", got)
    got = ev(fence + 'ts\nimport * as ecs from "bitecs"\n'
             'const w = ecs.createWorld()\n' + fence)
    check(got == {}, "[imports] a member used through another module's "
          "namespace import belongs to that module", got)
    got = ev(fence + 'js\nconst { createWorld } = require("bitecs")\n'
             'createWorld()\n' + fence)
    check(got == {}, "[imports] a require binds the same way", got)
    got = ev(fence + 'ts\nimport { createWorld } from "koota"\n'
             'const world = createWorld()\n' + fence)
    check(sorted(got.get("koota") or []) == ["import:koota",
                                             "symbol:createWorld"],
          "[imports] the same name imported from its own package counts",
          got)
    b = SP.imported_names([{"role": "user", "content":
                            'import d, { a, b as c } from "m"; import * as '
                            'ns from "n"; ns.q(); const { f, g: h } = '
                            'require("r")'}])
    check(b == {"a": "m", "c": "m", "d": "m", "q": "n", "f": "r",
                "h": "r"}, "[imports] named, aliased, default, namespace "
          "members and require bindings", b)



def test_blockers():
    import skill_match as M
    pool = [_sk("k1", "koota-lead", "koota", lead="koota"),
            _sk("k2", "koota-queries", "koota"),
            _sk("m1", "math-lead", "pmndrs_math", lead="math"),
            _sk("r1", "r3f-lead", "r3f", lead="@react-three/fiber"),
            _sk("t1", "tsl-lead", "threejs", lead="three/tsl")]
    leads = {"koota": "koota-lead", "math": "math-lead",
             "@react-three/fiber": "r3f-lead", "three/tsl": "tsl-lead"}
    play = {p: {"why": [{"how": "name", "what": p, "where": "user"}],
                "version": None, "events": ["ASKED"], "strength": "strong"}
            for p in ("koota", "math", "@react-three/fiber", "three/tsl")}
    out, rec, _ = M.plan_turn(pool=pool, old=[], pkg_play=play,
                              pkg_state=None, given=set(), cos=None,
                              legal_whys={}, phase_changed=False,
                              leads_for=leads.get, area_of=_area,
                              subject_areas=lambda s: set())
    rounds = sum(len(q["rounds"]) for q in rec["questions"])
    bodies = [p["skill"]["name"] for p in out if p["form"] == "body"]
    later = [q for q in rec["questions"] if q.get("room") == 0]
    check(len(bodies) == 3 and later and all(not q["rounds"] for q in later)
          and ("tsl-lead", "index") in [(p["skill"]["name"], p["form"])
                                        for p in out],
          "[blockers] the 3-body room is shared: once it is full a package "
          "question asks nothing, and its lead goes as an index line",
          (bodies, rounds, [(q["area"], q.get("room"), len(q["rounds"]))
                            for q in rec["questions"]]))
    k = next(q for q in rec["questions"] if q["area"] == "koota")
    check(len(k["rounds"]) <= M.BODIES_PER_DECISION
          and all(r["tokens"]["reused"] == k["tokens"]["prefix"]
                  for r in k["rounds"][1:])
          and len(k["rounds"]) > 1
          and all(r["tokens"]["processed"] == M._tokens(r["question"])
                  for r in k["rounds"][1:]),
          "[blockers] a package question's rounds share one option list: "
          "every round after the first reuses the prefix", k["rounds"])
    cands = [{"id": f"s{i}", "name": f"s{i}", "description": f"d{i}"}
             for i in range(3)]

    def tie_ask(area, options, turn):
        dist = {o["letter"]: 0.0 for o in options}
        dist["B"] = dist["C"] = 0.5       # B and C tie; A ranks first
        return None, dist
    res = M.ask_rounds("any", cands, {}, False, tie_ask, max_picks=1)
    r0 = res["rounds"][0]
    check(res["picks"] == ["s1"] and r0["tie"]
          and r0["tie"]["tied"][:2] == ["B", "C"],
          "[blockers] a tie (the decider returns no label) is broken by the "
          "retrieval rank -- the option order -- and recorded", r0)


def test_scratch_writes():
    import skill_select as S
    msgs = user("build it in /workspace/app") + [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "t1", "type": "function", "function": {
                "name": "terminal", "arguments": json.dumps({"command":
                    "cat <<'EOF' > /tmp/probe.ts\nconsole.log(1)\nEOF"})}},
            {"id": "t2", "type": "function", "function": {
                "name": "terminal", "arguments": json.dumps({"command":
                    "echo hi | tee -a /workspace/app/.scratch.log"})}},
            {"id": "t3", "type": "function", "function": {
                "name": "terminal", "arguments": json.dumps({"command":
                    "node gen.js > /workspace/app/src/data.json"})}}]}]
    got = {f["path"]: f["why"] for f in S.scratch_writes(msgs)}
    check(got.get("/tmp/probe.ts") == "a temp directory"
          and got.get("/workspace/app/.scratch.log") == "a dot-file"
          and "/workspace/app/src/data.json" not in got,
          "[scratch] a terminal's heredoc and tee targets are write "
          "targets; the temp and dot-file ones are scratch, a project "
          "file is not", got)


def main() -> int:
    for fn in (test_extraction, test_vocabulary, test_detection, test_rule,
               test_mapping, test_plan, test_detection_strength,
               test_index_upkeep, test_imports_decide, test_blockers,
               test_scratch_writes):
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
    shutil.rmtree(_TMP, ignore_errors=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
