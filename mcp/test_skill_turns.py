#!/usr/bin/env python
"""Per-turn, evidence-triggered skill injection, the craft index and
recall_craft, and the tool-conflict rule. No GPU, no network.

    python mcp/test_skill_turns.py      -> "N/M checks passed"

WHAT IS GATED (operator, 2026-09-27)

  1. THE WORDING. Everything the model reads calls them CRAFT, never
     "skill" (skill_prompts CRAFT_*), and each text is pinned to its
     version (mcp/fixtures/craft_prompt_pins.json).
  2. THE EXPLICIT ASK. "Build it with r3f (react-three-fiber) v10 and Koota
     and pmndrs math" asks for R3F (v10), koota and math, in every form the
     operator writes; a name in a pasted manifest or behind "without" asks
     for nothing; "my React page" is context.
  3. PRECISION. A platform API (localStorage) is no fact for a skill about
     something else; a host area's skill must be about the request.
  4. THE ENGINE (skill_select.decide) on a synthetic pool: the body the
     first time, a RECALL line (the skill's own matching item) when an
     error brings the area back, nothing on an unrelated step, the fade
     brings it back at the point of need, one recall per area per K
     requests, never the same line twice in a row, idempotent per message
     key, a compaction forgets what was given.
  5. recall_craft: by name, by topic, and an unknown name's near misses.
  6. NO CONFLICTS: a tool of ours is withheld when the client has the same
     name (normalised) or a declared overlap; a mutant that adds ours
     regardless is caught.
  7. THROUGH THE SERVED TEMPLATE (test_ledger's harness): the craft index
     at the end of the system text is decided once and kept, skills
     injected at an agent step are appended to its tool result and the
     next request extends the slot, and a recall_craft hop is hidden and
     replayed from the ledger.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

_TMP = tempfile.mkdtemp(prefix="yamadori_test_skill_turns_")
for _k, _v in (("YAMADORI_CORPUS_DB", "corpus.sqlite3"),
               ("YAMADORI_NEBARI_DB", "nebari.sqlite3"),
               ("RINGS_DB", "rings.sqlite3"),
               ("CODE_INDEX_DB", "code.sqlite3"),
               ("YAMADORI_SLOTS_STATE", "slots_state.json"),
               ("CONCEPT_SEED_LAST", "seed_last.json")):
    os.environ[_k] = os.path.join(_TMP, _v)
os.environ.setdefault("YAMADORI_GPU_ROOM", "0")

import test_ledger as T  # noqa: E402  (its own temp paths + fake upstream)
import nebari  # noqa: E402
import proxy  # noqa: E402
import skill_classify as C  # noqa: E402
import skill_prompts as P  # noqa: E402
import skill_select as S  # noqa: E402
import skills  # noqa: E402

_tmp_root = os.path.abspath(tempfile.gettempdir())
assert os.path.abspath(nebari.DB).startswith(_tmp_root), nebari.DB

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, str(detail)[:400]))
    return bool(ok)


# ------------------------------------------------------------- the pool --
def mk(sid: str, name: str, desc: str, applies: dict, items: list[dict],
       **gates) -> dict:
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
AUTH = mk(
    "r_auth", "react-auth-token-persistence",
    "Use when persisting a JWT or session token in a React SPA.",
    {"frameworks": ["react"], "languages": ["typescript"]},
    [{"form": "DO", "situation": "", "text": "store the token in an "
      "HttpOnly cookie."}],
    topics=["localStorage", "sessionStorage"])
CSS = mk(
    "c_css", "css-sticky-header",
    "Use when making a sticky header in CSS.",
    {"artifacts": ["ui_design"], "languages": ["css"]},
    [{"form": "DO", "situation": "", "text": "use position: sticky."}])
POOL = [KOOTA_Q, KOOTA_T, AUTH, CSS]


def U(text: str) -> list[dict]:
    return [{"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": text}]


def step(msgs: list[dict], name: str, args: dict, result: str,
         i: int) -> list[dict]:
    cid = f"c{i}"
    return msgs + [{"role": "assistant", "content": "", "tool_calls": [
        {"id": cid, "type": "function", "function": {
            "name": name, "arguments": json.dumps(args)}}]},
        {"role": "tool", "tool_call_id": cid, "content": result}]


# ============================================================ 1. wording ==
def test_the_model_reads_craft_and_the_wording_is_pinned():
    reg = P.craft_registry()
    with open(os.path.join(HERE, "fixtures", "craft_prompt_pins.json"),
              encoding="utf-8") as f:
        pins = json.load(f)
    for r in reg:
        check("skill" not in r["text"].lower(),
              f"[craft] {r['name']}: the model reads 'craft', never 'skill'",
              r["text"][:120])
        pin = pins.get(r["name"]) or {}
        check(pin.get("version") == r["version"]
              and pin.get("sha256") == r["sha256"],
              f"[craft] {r['name']} is pinned to {r['version']}: change the "
              "text, bump CRAFT_VERSION and mcp/fixtures/"
              "craft_prompt_pins.json", (pin, r["sha256"]))
    check(S.READ_TOOL["function"]["name"] == P.CRAFT_TOOL_NAME
          == "yama_recall_craft"
          and P.CRAFT_TOOL_NAME in proxy.OUR_NAMES
          and proxy.canonical_tool_name("recall_craft")
          == P.CRAFT_TOOL_NAME,
          "[craft] the tool's name is the one constant, and the proxy runs "
          "it", S.READ_TOOL["function"]["name"])


# =========================================================== 2. the ask ==
def test_the_explicit_ask():
    got = C.asked_terms("Build it with r3f (react-three-fiber) v10 and Koota "
                        "and pmndrs math.")
    check(set(got) == {"r3f", "koota", "pmndrs_math"}
          and got["r3f"]["version"] == "10"
          and all(v["mode"] == "choice" for v in got.values()),
          "[ask] the operator's sentence asks for R3F v10, koota and pmndrs "
          "math (a choice)", got)
    got = C.asked_terms("Set it up with react-three-fiber and three.js")
    check({"r3f", "threejs"} <= set(got),
          "[ask] 'react-three-fiber' and 'three.js' are asks", got)
    got = C.asked_terms("Write it in vanilla JavaScript without React or "
                        "three.js.")
    check("react" not in got and "threejs" not in got,
          "[ask] a name behind 'without' is not an ask", got)
    got = C.asked_terms('Why does the build fail?\n```json\n{"dependencies":'
                        ' {"koota": "0.6.6"}}\n```')
    check(not got, "[ask] a name in a pasted manifest is not an ask", got)
    got = C.asked_terms('deps:\n  "koota": "0.6.6",\n  "three": "0.185.1"\n'
                        'Why is the page blank?')
    check(not got, "[ask] manifest-shaped lines outside a fence are not "
          "asks", got)
    got = C.asked_terms("Add a dark-mode toggle to my React settings page.")
    check(got.get("react", {}).get("mode") == "context",
          "[ask] 'my React page' is CONTEXT, not a stack choice", got)
    sig = C.request_signals(U("Build it with koota and React."))
    check(sig["terms"]["koota"]["strength"] == "fact"
          and sig["terms"]["react"]["strength"] == "word",
          "[ask] a specific framework asked is a FACT; a host (React) "
          "stays a word", {k: v["strength"] for k, v in sig["terms"].items()})


# ======================================================= 3. precision ==
def test_a_platform_api_is_no_subject():
    sig = C.request_signals(U("Add a dark-mode toggle to my React settings "
                              "page that remembers the choice in "
                              "localStorage."))
    det = C.match(AUTH["rule"], sig)
    check(det["strength"] != "fact",
          "[precision] localStorage alone is no FACT for an auth-token "
          "skill (a platform API is a plain word)", det)
    chosen, _rec = S.select(U("Add a dark-mode toggle to my React settings "
                              "page that remembers the choice in "
                              "localStorage."), "prose", POOL)
    check(AUTH["id"] not in [c["id"] for c in chosen],
          "[precision] ... and it is not injected", [c["id"] for c in chosen])
    chosen, rec = S.select(U("Start a shooter in TypeScript and use koota "
                             "for the ECS."), "code_generation", POOL)
    check([c["id"] for c in chosen][:1] in (["k_queries"], ["k_traits"])
          and rec["matched"][0]["slot"] == "asked",
          "[ask] an asked specific area gets its best skill though none of "
          "its API names appear yet (slot: asked)", rec.get("matched"))


# ========================================================== 4. engine ==
_STATES: list = []


def _sequence():
    st = None
    out = []
    _STATES.clear()
    m = U("Start a small shooter in TypeScript and use koota for the ECS.")
    t, r, st = S.decide(m, "code_generation", POOL, [], st, key="k0")
    out.append((t, r))
    _STATES.append(st)
    m = step(m, "write_file", {"path": "src/systems/bullets.ts", "content":
             "import { trait } from 'koota'\nexport function move(world, d)"
             " {\n  world.query(IsBullet, Position, Velocity).updateEach(("
             "[b, pos, vel]) => { pos.y -= vel.y * d })\n}\n"},
             '{"bytes_written": 200}', 1)
    t, r, st = S.decide(m, "agent_step", POOL, [], st, key="k1")
    out.append((t, r))
    _STATES.append(st)
    m = step(m, "terminal", {"command": "npm run dev"},
             "Uncaught TypeError: Cannot read properties of undefined "
             "(reading 'y')\n    at updateEach (koota.js:1:5210)\n"
             "world.query(IsBullet, Position, Velocity).updateEach(([b, pos,"
             " vel]) => ...)", 2)
    t, r, st = S.decide(m, "agent_step", POOL, [], st, key="k2")
    out.append((t, r))
    _STATES.append(st)
    m = step(m, "write_file", {"path": "README.md", "content": "# Shooter\n"
                                                               "Run it.\n"},
             '{"bytes_written": 20}', 3)
    t, r, st = S.decide(m, "agent_step", POOL, [], st, key="k3")
    out.append((t, r))
    return out, m, st


def _forms(rec: dict) -> dict:
    """STAGE 1's forms (x_yamadori.skills.stage1): body -- new to the
    conversation; recall -- an event brings a given skill back. The
    injector then asks jjava about the skill's ITEMS (the stub decider
    offline, mcp/decider_stub.py)."""
    return {d["name"]: d["form"] for d in rec.get("stage1") or []}


def test_the_engine_body_then_recall_then_nothing():
    out, m, st = _sequence()
    (t0, r0), (t1, r1), (t2, r2), (t3, r3) = out
    check(r0["stage1"] and r0["stage1"][0]["trigger"] == "asked"
          and r0["stage1"][0]["form"] == "body"
          and P.CRAFT_HEADER in t0 and r0["decisions"],
          "[engine] turn 0: stage 1 offers the asked area's skill as new "
          "(body); its items go in under the craft header",
          r0.get("stage1"))
    f1 = _forms(r1)
    check(f1.get("koota-queries-and-systems") in ("body", None)
          and ("koota-queries-and-systems" in f1
               or r0["stage1"][0]["name"] == "koota-queries-and-systems"),
          "[engine] step 1: the first koota query written brings the "
          "queries skill (a body, unless the ask already gave it)", f1)
    f2 = _forms(r2)
    d2 = [d for d in r2["stage1"] if d["name"] ==
          "koota-queries-and-systems"]
    check(d2 and d2[0]["form"] == "recall" and d2[0]["trigger"] == "error"
          and "updateEach" in t2 and "Remember (craft" not in t2,
          "[engine] step 2: a koota error brings the given skill back "
          "(stage 1: recall, trigger error) and its items, updateEach "
          "among them, go in again -- no recall LINE (retired with the "
          "per-skill path)", (f2, t2[:300]))
    check(not r3["decisions"] and not t3,
          "[engine] step 3: an unrelated step (a README) injects nothing",
          r3.get("decisions"))
    # Idempotent per key: the same request again, the same text, no change.
    # (the state after step 2 is replayed from the sequence's saved copy)
    st_after2 = _STATES[2]
    t2b, r2b, st2 = S.decide(m[:-2], "agent_step", POOL, [], st_after2,
                             key="k2")
    check(t2b == t2 and r2b.get("replayed") and st2 == st_after2,
          "[engine] a retry of the same request (same key) gets the same "
          "text and changes nothing", (t2b[:80], r2b.get("replayed")))


def test_the_fade_rate_limit_and_compaction():
    """No fade and no cooldown since 2026-09-27 (docs/CONSTANTS-AUDIT.md
    "FADE_TOKENS", "RECALL_EVERY_STEPS"): a recall comes on an EVENT only,
    never the identical line twice in a row; a compaction is the proxy's own
    count (progress), not a shrink ratio."""
    _out, m, st = _sequence()
    n = 3
    # Unrelated steps that grow the conversation: nothing, and after them
    # the koota query written again brings nothing either (no event).
    for i in range(4):
        n += 1
        m = step(m, "terminal", {"command": f"npm ci #{i}"},
                 "npm warn deprecated inflight@1.0.6\n" * 250, n)
        t, r, st = S.decide(m, "agent_step", POOL, [], st, key=f"f{i}")
        check(not t, f"[no fade] unrelated step {i} injects nothing",
              r.get("decisions"))
    n += 1
    m = step(m, "write_file", {"path": "src/systems/wind.ts", "content":
             "import { trait } from 'koota'\nexport function wind(world, d)"
             " {\n  world.query(Falling, Velocity).updateEach(([f, vel]) =>"
             " { vel.x += d })\n}\n"}, '{"bytes_written": 150}', n)
    t, r, st = S.decide(m, "agent_step", POOL, [], st, key="fade")
    check(not any(x["name"] == "koota-queries-and-systems"
                  for x in r["decisions"]) and not t,
          "[no fade] the koota query written again after unrelated output: "
          "nothing (given, and no event makes it matter)",
          (r.get("stage1"), r.get("skipped")))
    # An error is an event: stage 1 brings the skill back at once (no
    # cooldown), and its items go in unless they were the LAST injection
    # on an error too -- never the identical injection twice in a row.
    n += 1
    m = step(m, "terminal", {"command": "npm test"},
             "TypeError: x is undefined at updateEach (koota.js:1:9)\n"
             "world.query(Falling, Velocity).updateEach(...)", n)
    t, r, st = S.decide(m, "agent_step", POOL, [], st, key="rate")
    check(any(x["form"] == "recall" for x in r["stage1"])
          and any("same item" in s0["why"] for s0 in r["skipped"])
          and not t,
          "[no cooldown] an error brings the skill back at once (stage 1: "
          "recall), and the items the LAST error already brought are not "
          "sent again (never the identical injection twice in a row)",
          (r.get("stage1"), r.get("skipped")))
    # A compaction: the proxy's count moved, what was given is gone.
    short = U("Continue the shooter; use koota for the ECS.")
    t, r, st = S.decide(short, "code_generation", POOL, [], st, key="cmp",
                        compactions=1)
    check(st.get("compacted") and any(x["form"] == "body"
                                      for x in r["stage1"]) and t,
          "[compaction] the proxy's compaction count moved: what was given "
          "is forgotten and the next need gets the body again",
          r.get("decisions"))
    t, r, st = S.decide(short + [{"role": "assistant", "content": "ok"},
                                 {"role": "user", "content": "go on"}],
                        "code_generation", POOL, [], st, key="cmp2",
                        compactions=1)
    check(st.get("compacted") == 1, "[compaction] the same count again "
          "resets nothing", st.get("compacted"))


def test_a_phase_change_on_a_user_turn():
    st = None
    m = U("Use koota for the ECS: move enemies each frame.")
    _t, _r, st = S.decide(m, "code_generation", POOL, [], st, key="p0")
    m = step(m, "write_file", {"path": "src/waves.ts", "content":
             "import { trait } from 'koota'\nworld.query(Enemy, Position)"
             ".updateEach(([e, pos]) => { pos.y += 1 })\n"},
             '{"bytes_written": 90}', 1)
    _t, _r, st = S.decide(m, "agent_step", POOL, [], st, key="p1")
    m = m + [{"role": "assistant", "content": "Done."},
             {"role": "user", "content": "Still broken: the enemies never "
              "move and the console shows TypeError in waves.ts."}]
    t, r, st = S.decide(m, "prose", POOL, [], st, key="p2")
    check(r.get("phase_changed") and any(
              x["name"] == "koota-queries-and-systems" and x["trigger"]
              in ("phase", "error") for x in r["stage1"]),
          "[phase] implement -> debug on a user turn brings the koota "
          "queries skill back (a recall: it was given)",
          (r.get("phase"), r.get("decisions")))


# ====================================================== 5. recall_craft ==
def test_recall_craft():
    text, rec = S.read_craft({P.CRAFT_TOOL_ARG: "koota-queries-and-systems"},
                             POOL)
    check(rec["found"] == "k_queries" and rec["how"] == "name"
          and text.startswith(P.CRAFT_RESULT_HEAD.format(
              name="koota-queries-and-systems")) and "updateEach" in text,
          "[recall_craft] by name: the craft in full", rec)
    # By NAME only since 2026-09-27 (craft/4): a topic lists the names that
    # match it, closest first, to call again by name.
    text, rec = S.read_craft({P.CRAFT_TOOL_ARG: "koota queries"}, POOL)
    check(rec["found"] is None and rec.get("near", [None])[0]
          == "koota-queries-and-systems",
          "[recall_craft] a topic: NO_SUCH_CRAFT with the matching names, "
          "closest first", rec)
    text, rec = S.read_craft({P.CRAFT_TOOL_ARG: "kubernetes helm charts"},
                             POOL)
    d = json.loads(text)
    check(d["ok"] is False and d["error"] == "NO_SUCH_CRAFT"
          and d["retryable"] is True and d.get("remedies")
          and "skill" not in text.lower(),
          "[recall_craft] an unknown name: the situation, retryable, a "
          "remedy with an owner, and what IS there", d)
    text, rec = S.read_craft({P.CRAFT_TOOL_ARG: "koota-querys-and-system"},
                             POOL)
    check(rec.get("found") is None and "koota-queries-and-systems" in text,
          "[recall_craft] a near-miss name names it (among `near`)",
          text[:200])


# ======================================================= 6. conflicts ==
HERMES = [{"type": "function", "function": {"name": n, "parameters": {}}}
          for n in ("terminal", "read_file", "write_file", "patch",
                    "search_files", "skills_list", "skill_view",
                    "vision_analyze")]


def _no_conflicts(tools: list[dict]) -> tuple[bool, str]:
    names = [(t.get("function") or {}).get("name") for t in tools]
    norm = [proxy._norm_tool(n) for n in names]
    dup = sorted({n for n in norm if norm.count(n) > 1})
    over = [(o, c) for o, cl, _w in proxy.TOOL_OVERLAPS for c in cl
            if o in names and c in names]
    return not dup and not over, f"dup={dup} overlap={over}"


def test_tools_of_ours_never_conflict_with_the_clients():
    os.environ["YAMADORI_IMAGEGEN_URL"] = "http://127.0.0.1:9"
    try:
        held: list = []
        tools, ours = proxy.main_tools(HERMES, None, craft=True,
                                       withheld=held)
        ok, why = _no_conflicts(tools)
        check(ok and "yama_describe_image" not in ours
              and {"ours": "yama_describe_image", "client_tool":
                   "vision_analyze"}.items() <= held[0].items()
              and "yama_recall_craft" in ours,
              "[conflicts] Hermes: yama_describe_image is withheld beside "
              "vision_analyze (a declared overlap), recorded; "
              "yama_recall_craft stays",
              (sorted(ours), held, why))
        check(all(n.startswith("yama_") for n in ours),
              "[conflicts] every tool of ours on main is yama_* (operator, "
              "2026-09-27)", sorted(ours))
        client = HERMES + [{"type": "function", "function": {
            "name": "Yama-Generate-Images", "parameters": {}}}]
        held = []
        tools, ours = proxy.main_tools(client, None, withheld=held)
        check("yama_generate_image" not in ours and any(
                  w["because"] == "same name" for w in held),
              "[conflicts] a normalised name match (case, '-', plural) "
              "withholds ours", held)
        client = HERMES + [{"type": "function", "function": {
            "name": "generate_image", "parameters": {}}}]
        held = []
        tools, ours = proxy.main_tools(client, None, withheld=held)
        check("yama_generate_image" not in ours and any(
                  w["client_tool"] == "generate_image"
                  and w["because"] != "same name" for w in held),
              "[conflicts] a client tool under our OLD name answers the same "
              "question: a declared overlap withholds ours", held)
        held = []
        tools, ours = proxy.main_tools(HERMES, None, craft=True,
                                       withheld=held,
                                       keep_withheld=["recall_craft"])
        check("yama_recall_craft" not in ours,
              "[conflicts] a tool withheld earlier in the conversation stays "
              "out (the list is stable) -- a name stored before the yama_* "
              "rename is read as its new name", sorted(ours))
        saved = proxy.tool_conflicts
        proxy.tool_conflicts = lambda o, c: (list(o), [])
        try:
            tools, _o = proxy.main_tools(HERMES, None, craft=True)
            ok, why = _no_conflicts(tools)
        finally:
            proxy.tool_conflicts = saved
        check(not ok, "[conflicts] a MUTANT that adds ours regardless is "
              "caught by the check", why)
    finally:
        os.environ.pop("YAMADORI_IMAGEGEN_URL", None)


# ============================================ 7. the served template ====
def test_through_the_served_template():
    """The craft index is decided once and kept; skills at a step go on the
    tool result and the next request extends the slot; a recall_craft hop
    is hidden and replayed."""
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
        c.msgs[0]["content"] += " [skill turns]"
        w = {"path": "src/systems/bullets.ts", "content":
             "import { trait } from 'koota'\nexport function move(world, d)"
             " {\n  world.query(IsBullet, Position, Velocity).updateEach(("
             "[b, pos, vel]) => { pos.y -= vel.y * d })\n}\n"}
        t1 = c.turn([T.reply("", calls=[T.call("write_file", w, "w1")])],
                    user="Start a small shooter in TypeScript and use koota "
                         "for the ECS.")
        req1 = t1["gens"][0]["request"]
        sys1 = req1["messages"][0]["content"]
        names1 = [t["function"]["name"] for t in req1.get("tools") or []]
        x1 = t1["d"]["x_yamadori"]
        check("## Craft you can recall" in sys1
              and "koota-" in sys1 and P.CRAFT_TOOL_NAME in names1
              and (x1.get("craft") or {}).get("tool") is True,
              "[served] the first request: the craft index at the end of "
              "the system text, recall_craft on main, x_yamadori.craft",
              (names1, x1.get("craft")))
        check(any(P.CRAFT_HEADER in (m.get("content") or "")
                  for m in req1["messages"] if m.get("role") == "user"),
              "[served] the asked koota skill is a body on the user turn",
              json.dumps(x1.get("skills"))[:300])
        c.tool_result("w1", '{"bytes_written": 200}')
        t2 = c.turn([T.reply("", calls=[T.call(
                        P.CRAFT_TOOL_NAME,
                        {P.CRAFT_TOOL_ARG: "koota-traits-and-entities"},
                        "rc1")]),
                     T.reply("Wrote the bullets system.")])
        req2 = t2["gens"][0]["request"]
        T._extends(t1, t2, "[served] the step after a write")
        check(req2["messages"][0]["content"] == sys1
              and [t["function"]["name"] for t in req2.get("tools") or []]
              == names1,
              "[served] the system block and the tool list are the same on "
              "the next request (decided once, kept)")
        last_tool = [m for m in req2["messages"] if m.get("role") == "tool"]
        x2 = t2["d"]["x_yamadori"]
        sk2 = x2.get("skills") or {}
        check(last_tool and (sk2.get("kind") == "step"),
              "[served] the step was decided (x_yamadori.skills.kind "
              "step)", json.dumps(sk2)[:300])
        hop = t2["gens"][1]["request"]["messages"]
        check(any(m.get("role") == "tool" and m.get("tool_call_id") == "rc1"
                  and "Craft koota-traits-and-entities" in (m.get("content")
                                                            or "")
                  for m in hop)
              and not (t2["d"]["choices"][0]["message"].get("tool_calls"))
              and (((x2.get("craft") or {}).get("reads") or [{}])[0].get(
                  "found") == "k_traits"
                   # the write just before it imported koota: the package
                   # trigger already gave this craft's body, so the recall is
                   # a REPEAT (2026-10-07): the do-not-repeat line
                   or ((x2.get("craft") or {}).get("repeat") or [{}])[0].get(
                       "craft") == "koota-traits-and-entities"),
              "[served] recall_craft ran as a hidden hop: its result is the "
              "craft (or, when the conversation already has it, the "
              "do-not-repeat line), the client never sees the call, "
              "x_yamadori.craft records it", json.dumps(x2.get("craft"))[:300])
        t3 = c.turn([T.reply("Thanks.")], user="Thanks, that works.")
        T._extends(t2, t3, "[served] the request after the recall_craft hop")
        msgs3 = t3["gens"][0]["request"]["messages"]
        check(any(m.get("role") == "tool" and m.get("tool_call_id") == "rc1"
                  for m in msgs3)
              and msgs3[0]["content"] == sys1,
              "[served] the hidden hop is replayed from the ledger; the "
              "system block is still the first request's")
    finally:
        (skills.armed, proxy._skills_tail, S.best_cosines,
         S.ask_fallback) = saved


def main() -> int:
    for fn in (test_the_model_reads_craft_and_the_wording_is_pinned,
               test_the_explicit_ask, test_a_platform_api_is_no_subject,
               test_the_engine_body_then_recall_then_nothing,
               test_the_fade_rate_limit_and_compaction,
               test_a_phase_change_on_a_user_turn, test_recall_craft,
               test_tools_of_ours_never_conflict_with_the_clients,
               test_through_the_served_template):
        print(f"\n--- {fn.__name__} ---")
        try:
            fn()
        except Exception as e:                                   # noqa: BLE001
            traceback.print_exc()
            check(False, f"{fn.__name__} itself raised",
                  f"{type(e).__name__}: {e}")
        for ok, name, detail in _results[-50:]:
            pass
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
