#!/usr/bin/env python
"""The package skills channel (mcp/package_skills.py) and the question path of
yama_recall_craft (mcp/craft_query.py). No GPU, no network, no model server:
a fake armed pool, a fake MCP server (mcp/fixtures/fake_mcp_server.py), a
fake jjava, test_ledger's served template.

    python mcp/test_package_skills.py      -> "N/M checks passed"

GATED HERE (operator, 2026-10-06: "If the model uses some of our other tools
or mcps this may be a good point to skill up"):
  1. THE SECTION (inject): a package tool's result carries the package's LEAD
     skill once, then the strongest items by PROVE verdict and pitfall
     evidence, the craft names and the pointer; a recall line the next time
     the package comes up, never the identical line twice; the major version
     picks the skills (a resolved version, an asked one, the registry's); a
     quarantined (PROVE worse) skill never rides; the sizes (items, tokens);
     a doubt-bearing line never rides; nothing where the package has no armed
     skill.
  2. THE ROUTER and BOTH renderings: the table's shape, caps, only armed and
     proven crafts, no body in `router`, the lead's body in `both`, the inject
     form where yama_recall_craft is not on main.
  3. THE SWITCH `package_skills` (off by default; env, header) and its mode.
  4. THE TRIGGERS between turns: an install line (a command, or an echoed
     result line), an import in written code, an error naming the package;
     NOT a package.json the model only read; nothing without an armed skill;
     body first, recall later, once per trigger.
  5. THROUGH THE SERVED TEMPLATE, chat and Responses: the section rides in the
     hidden hop, the ledger replays it byte for byte, the next request EXTENDS
     the slot; the craft tool is offered with the `skills` flag off; the
     records (x_yamadori.skills.package); a recall after a router is recorded.
  6. THE QUESTION PATH of yama_recall_craft: a name is unchanged; a question
     is narrowed by the embedder (a fake), chosen by ONE jjava choice with NO
     "none" option and checked by ONE noul relevance gate; untuned: the best
     craft is returned and the record says so; a tuned gate that is confident
     "no" returns the honest NO_SUCH_CRAFT answer with the next steps; jjava
     down: the same; the durable log row (traffic class, never the key);
     the ledger replays the hop byte for byte.

Every database and store is a temp path set BEFORE proxy is imported (the
test_mcp_host / test_ledger harness does it).
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

_TMP = tempfile.mkdtemp(prefix="yamadori_test_package_skills_")
os.environ["YAMADORI_PITFALLS_DIR"] = os.path.join(_TMP, "pitfalls")
os.makedirs(os.environ["YAMADORI_PITFALLS_DIR"], exist_ok=True)

# the MCP servers' configuration, before anything imports mcp_config
os.environ["YAMADORI_MCP_SERVERS"] = os.path.join(_TMP, "mcp", "servers.json")
os.environ["YAMADORI_MCP_ALLOW_LOCAL"] = "1"
os.environ.pop("YAMADORI_MCP_TOOLS", None)
os.environ.pop("YAMADORI_MCP_HOST", None)
os.environ.setdefault("YAMADORI_GPU_ROOM", "0")

# The Responses suite first: it points the accounts store, the media store and
# the databases at its own temp paths before they are imported.
import test_responses_api as RT  # noqa: E402
import test_mcp_host as H  # noqa: E402  (test_ledger's harness + the fake MCP server)
import test_ledger as T  # noqa: E402
import craft_query  # noqa: E402
import decider_bonsai as D  # noqa: E402
import mcp_host  # noqa: E402
import package_skills as PS  # noqa: E402
import skill_inject  # noqa: E402
import skill_learn  # noqa: E402
import skill_limits as L  # noqa: E402
import skill_packages  # noqa: E402
import skill_prompts as P  # noqa: E402
import skill_select  # noqa: E402
import skills  # noqa: E402
import tiers  # noqa: E402

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


# ------------------------------------------------------------- the library --
def sk(name: str, area: str, items: list, *, desc: str | None = None,
       lead_for=None, package: str | None = None) -> dict:
    its = [{"form": f, "text": t, "situation": sit, "quote": "q" * 30}
           for f, t, sit in items]
    body = f"# {name}\n" + "\n".join(
        f"- {f}: {t}" if f != "WHEN" else f"- WHEN {sit}: {t}"
        for f, t, sit in items)
    return {"id": "id-" + name, "name": name, "version": 1,
            "description": desc or f"Use when writing {name} code. Not for "
                                   f"other things.",
            "title": name, "items": its, "body": body, "text": "",
            "rule": {"applies_to": {"frameworks": [area]}, "topics": []},
            "lead_for": lead_for, "package": package, "tags": []}


KOOTA_LEAD = sk("koota-traits-and-entities", "koota", [
    ("DO", "Spawn entities with world.spawn(Trait(...)).", ""),
    ("WHEN", "mutate an array trait in place: call entity.changed(Trait).",
     "you mutate an array trait in place"),
    ("DO NOT", "read a trait by destructuring the entity; use entity.get.", "")])
KOOTA_Q = sk("koota-queries-and-systems", "koota", [
    ("DO", "Query with world.query(A, B) and update with updateEach.", "")],
    desc="Use when iterating koota entities each frame in a system.")
KOOTA_R = sk("koota-react-integration", "koota", [
    ("DO", "Read traits in React with useTrait(entity, Trait).", ""),
    ("DO", "Mount a world with WorldProvider.", "")])
KOOTA_BAD = sk("koota-quarantined", "koota", [
    ("DO", "Use the quarantined koota pattern.", "")])
KOOTA_DOUBT = sk("koota-doubtful", "koota", [
    ("DO", "Verify the koota API before relying on it.", "")])
MATH_LEAD = sk("math-data-oriented-functions", "pmndrs_math", [
    ("DO", "Call vec3.add(out, a, b): the out argument comes first.", "")])
MATH_PIT = sk("math-hot-path-pitfalls", "pmndrs_math", [
    ("DO NOT", "allocate vectors inside updateEach; reuse scratch ones.", "")])
R3F10 = sk("r3f-v10-setup-21", "r3f", [
    ("DO", "Import Canvas from @react-three/fiber/webgpu in v10.", "")])
R3F9 = sk("r3f-v9-setup-22", "r3f", [
    ("DO", "Import Canvas from @react-three/fiber in v9.", "")])
R3F_ANY = sk("r3f-useframe-1", "r3f", [
    ("DO", "Do per-frame work in useFrame, not in React state.", "")])
DREI = sk("r3f-drei-pairing-8", "r3f", [
    ("DO", "Pair drei 11 with fiber 10.", "")])
UNRELATED = sk("typescript-strict-1", "typescript", [
    ("DO", "Enable strict.", "")])

POOL = [KOOTA_LEAD, KOOTA_Q, KOOTA_R, KOOTA_BAD, KOOTA_DOUBT, MATH_LEAD,
        MATH_PIT, R3F10, R3F9, R3F_ANY, DREI, UNRELATED]
VERDICT = {"koota-quarantined": "worse", "koota-queries-and-systems": "better",
           "math-hot-path-pitfalls": "tie"}
skills.armed = lambda *a, **k: list(POOL)
PS.PROVE_OF = lambda s: VERDICT.get(s["name"])
KOOTA_LEAD_ITEM = "Spawn entities with world.spawn(Trait(...))."

# pitfall evidence: the react-integration craft answers a recorded pitfall, its
# first item (useTrait) is the evidence
with open(os.path.join(os.environ["YAMADORI_PITFALLS_DIR"], "koota.jsonl"),
          "w", encoding="utf-8") as f:
    f.write("// test fixture\n")
    f.write(json.dumps({"id": "c1", "skills": [
        {"name": "koota-react-integration",
         "match": ["WorldProvider"]}]}) + "\n")


def ctx(**kw) -> dict:
    base = {"mode": "inject", "serving": "bonsai", "craft_tool": True,
            "tool": "yama_find_package", "asked": {}, "key": None}
    base.update(kw)
    return base


KOOTA = [{"name": "koota", "version": "0.6.6"}]


# =================================================== 1. the section (inject) ==
def test_the_inject_section():
    text, recs, st = PS.decide({}, KOOTA, ctx(), POOL)
    r = recs[0]
    names = [x["name"] for x in r["skills"]]
    check(text.startswith("\n---\n" + P.CRAFT_HEADER)
          and "- DO: " + KOOTA_LEAD_ITEM in text,
          "the section opens under the craft header with the LEAD skill's "
          "items, delimited from the tool result", text[:200])
    check(names[0] == "koota-traits-and-entities"
          and all(x["form"] == "body" for x in r["skills"]),
          "the lead skill is first and each is a body the first time",
          json.dumps(r["skills"]))
    check("koota-quarantined" not in text and "koota-doubtful" not in text
          and "koota-quarantined" not in names
          and "koota-doubtful" not in names,
          "a skill PROVE called worse never rides (and a skill whose every "
          "item is doubt never rides)", json.dumps(r.get("left_out")))
    check("Crafts for koota 0: koota-traits-and-entities" in text
          and "yama_recall_craft returns any of them in full." in text
          and "koota-queries-and-systems" in text.split("Crafts for")[-1],
          "the craft names are listed with the pointer to yama_recall_craft",
          text[-300:])
    check(r["package"] == "koota" and r["version"] == "0.6.6"
          and r["major"] == 0 and r["tool"] == "yama_find_package"
          and r["mode"] == "inject" and r["chars"] == len(text)
          and r["why"], "the record: tool, package, version, major, skills, "
          "chars, why", json.dumps(r)[:300])
    check(skill_inject.MAX_SKILLS >= len(names) and len(
        text.split("\n- ")) - 1 <= tiers_items(),
          "at most MAX_SKILLS skills and the profile's max_items items",
          json.dumps(names))
    # the strongest items of the OTHER skills: PROVE better first
    idx = lambda s: text.find(s)                                   # noqa: E731
    check(0 <= idx("world.query(A, B)") < idx("useTrait"),
          "after the lead, the other skills' items go by PROVE verdict "
          "(better before no record)")
    # again: a recall line, not the body
    t2, r2, st = PS.decide(st, KOOTA, ctx(tool="yama_list_package_versions"),
                           POOL)
    check(t2.startswith("\n---\nRemember (craft koota-traits-and-entities): ")
          and "world.spawn" in t2 and "; not " in t2
          and r2[0]["skills"][0]["form"] == "recall"
          and "- DO:" not in t2,
          "the package again later: a recall line (the craft's first DO and "
          "first DO NOT), not the body", t2)
    t3, r3, st = PS.decide(st, KOOTA, ctx(tool="yama_read_package_readme"),
                           POOL)
    check(t3 == "" and "last one given" in r3[0]["why"],
          "never the identical line twice in a row", json.dumps(r3))
    # nothing where the package has no armed skill
    t4, r4, _ = PS.decide({}, [{"name": "zustand", "version": "5.0.0"}],
                          ctx(), POOL)
    check(t4 == "" and r4 == [], "a package with no skill area: nothing, "
          "nothing recorded", json.dumps(r4))
    pool_wo = [s for s in POOL if not s["name"].startswith("koota")]
    t5, r5, _ = PS.decide({}, KOOTA, ctx(), pool_wo)
    check(t5 == "" and "no armed" in (r5[0].get("why") or ""),
          "koota with no armed koota skill: nothing, and why", json.dumps(r5))


def tiers_items() -> int:
    return skill_inject.profile_for("bonsai")["max_items"]


def test_the_major_picks_the_skills():
    fiber = "@react-three/fiber"
    t10, r10, _ = PS.decide({}, [{"name": fiber, "version": "10.0.0-alpha.5"}],
                            ctx(), POOL)
    t9, r9, _ = PS.decide({}, [{"name": fiber, "version": "9.8.1"}], ctx(),
                          POOL)
    check("fiber/webgpu" in t10 and "r3f-v9-setup-22" not in t10
          and r10[0]["major"] == 10 and r10[0]["major_from"] == "registry",
          "fiber 10: the v10 lead, never the v9 skill", t10[:240])
    check("Import Canvas from @react-three/fiber in v9." in t9
          and "r3f-v10-setup-21" not in t9 and r9[0]["major"] == 9,
          "fiber 9: the v9 lead, never the v10 skill", t9[:240])
    ta, ra, _ = PS.decide({}, [{"name": fiber, "version": "9.8.1"}],
                          ctx(asked={fiber: "10.0.0"}), POOL)
    check(ra[0]["major"] == 10 and ra[0]["major_from"] == "asked"
          and "fiber/webgpu" in ta,
          "the version the conversation asked for beats the registry's "
          "latest (a find_package result says 9.8.1; the task says v10)",
          json.dumps(ra)[:200])
    tr, rr, _ = PS.decide({}, [{"name": fiber, "version": "10.0.0-alpha.5"}],
                          ctx(tool="yama_resolve_packages",
                              asked={fiber: "9.0.0"}), POOL)
    check(rr[0]["major"] == 10 and rr[0]["major_from"] == "resolved",
          "a resolved version (what installs) beats what was asked",
          json.dumps(rr)[:200])
    td, rd, _ = PS.decide({}, [{"name": "@react-three/drei",
                                "version": "11.0.0"}], ctx(), POOL)
    check("Pair drei 11 with fiber 10." in td
          and "r3f-useframe-1" not in td,
          "drei (which does not name the r3f area): its own canonical craft "
          "only, not fiber's whole area", td[:240])


def test_the_sizes():
    big = sk("koota-big", "koota", [
        ("DO", "Alpha " + "word " * 55, "") for _ in range(6)],
        desc="Use when writing big koota code.")
    big["id"] = "id-koota-big"
    lead_big = sk("koota-traits-and-entities", "koota", [
        ("DO", f"Move {i} " + "word " * 50, "") for i in range(6)])
    text, recs, _ = PS.decide({}, KOOTA, ctx(), [lead_big, big])
    body = text.split("\n\nCrafts for")[0]
    check(L.tokens(body) <= L.SKILL_TOKENS_HARD,
          "the block is under skill_limits.SKILL_TOKENS_HARD tokens",
          str(L.tokens(body)))
    check(len(re.findall(r"^- ", body, re.M)) <= L.MAX_ITEMS,
          "at most max_items items", str(len(re.findall(r"^- ", body,
                                                        re.M))))
    many = [sk(f"koota-x{i}", "koota", [("DO", f"x{i}", "")],
               desc=f"Use when doing x{i}.") for i in range(40)]
    tbl, used = PS.router_table("koota", 0, many)
    check(len(used) <= L.INDEX_MAX and L.tokens(tbl) <= L.SKILL_TOKENS_HARD,
          "the router table: at most INDEX_MAX rows under the token cap",
          str(len(used)))
    long_desc = sk("koota-long", "koota", [("DO", "y", "")],
                   desc="Use when " + "very " * 80 + "long. Not for z.")
    check(len(PS.trigger_text(long_desc)) <= L.INDEX_LINE_CHARS
          and "Not for" not in PS.trigger_text(long_desc),
          "a row's trigger text is cut at INDEX_LINE_CHARS and drops the "
          "'Not for' sentence", PS.trigger_text(long_desc))


# ===================================================== 2. router and both ====
def _rows(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if ln.startswith("| ")
            and not ln.startswith("| when") and not ln.startswith("|---")]


def test_the_router_rendering():
    text, recs, st = PS.decide({}, KOOTA, ctx(mode="router"), POOL)
    rows = _rows(text)
    names = [json.loads(re.search(r"\{.*\}", r).group(0))["name_or_topic"]
             for r in rows]
    check(text.startswith("\n---\nCraft for koota 0 from this service's "
                          "library: when the work reaches one of these, call "
                          "yama_recall_craft with the craft's name to read it "
                          "in full; one or two crafts are usually enough for "
                          "a step.\n\n| when you are about to | call "
                          "yama_recall_craft with |\n|---|---|\n"),
          "the router: one positive head naming the tool and the moment, "
          "then the table", text[:260])
    check(all(re.fullmatch(r'\| .+ \| \{"name_or_topic": "[\w-]+"\} \|', r)
              for r in rows) and names[0] == "koota-traits-and-entities"
          and set(names) == {"koota-traits-and-entities",
                             "koota-queries-and-systems",
                             "koota-react-integration"},
          "one row per armed, proven craft, lead first; the quarantined and "
          "the doubtful crafts have none", json.dumps(names))
    check("world.spawn" not in text and "- DO" not in text
          and "Remember" not in text and "never" not in text.lower()
          and "do not" not in text.lower(),
          "NO skill body in the router, and no prohibition")
    check(L.tokens(text) <= L.SKILL_TOKENS_HARD and len(rows) <= L.INDEX_MAX
          and recs[0]["mode"] == "router"
          and all(x["form"] == "router" for x in recs[0]["skills"]),
          "within the caps; the record says router", json.dumps(recs[0])[:240])
    t2, r2, st = PS.decide(st, KOOTA, ctx(mode="router"), POOL)
    check(t2 == "" and "router table was given" in r2[0]["why"],
          "the router table is not repeated for the package")
    tn, rn, _ = PS.decide({}, KOOTA, ctx(mode="router", craft_tool=False),
                          POOL)
    check("- DO: " + KOOTA_LEAD_ITEM in tn and rn[0]["mode"] == "inject"
          and rn[0]["mode_requested"] == "router"
          and "not on main" in rn[0]["mode_note"],
          "router without yama_recall_craft on main: the inject form, said "
          "in the record", json.dumps(rn[0])[:300])
    # follow-up: a call after the router is recorded
    rec, st = PS.note_craft_call(st, "koota-queries-and-systems", True)
    check(rec and rec["event"] == "recall_after_section"
          and rec["listed_by_router"] is True and rec["mode"] == "router"
          and rec["steps_later"] >= 0 and rec["craft"]
          == "koota-queries-and-systems" and rec["package"] == "koota",
          "a yama_recall_craft call after a router is recorded: the craft, "
          "whether the router listed it, the mode, steps later",
          json.dumps(rec))
    st2 = PS.begin_request(st, "r2")                  # one more request
    st2 = PS.begin_request(st, "r3")
    rec2, _ = PS.note_craft_call(st, "koota-react-integration", True)
    check(rec2["steps_later"] == rec["steps_later"] + 2,
          "steps later counts the requests since the section",
          f"{rec2['steps_later']} vs {rec['steps_later']}")
    del st2


def test_the_both_rendering():
    text, recs, _ = PS.decide({}, KOOTA, ctx(mode="both"), POOL)
    rows = _rows(text)
    check("- DO: " + KOOTA_LEAD_ITEM in text
          and "Craft for koota 0 from this service's library" in text
          and "world.query(A, B)" not in text
          and len(rows) == 2
          and "koota-traits-and-entities" not in "".join(rows),
          "both: the lead's body, then the router table for the REST (not "
          "the lead, no other body)", text[-420:])
    forms = {x["name"]: x["form"] for x in recs[0]["skills"]}
    check(forms["koota-traits-and-entities"] == "body"
          and forms["koota-queries-and-systems"] == "router",
          "both: the records say body for the lead and router for the rest",
          json.dumps(forms))


# ============================================================== 3. switch ====
def test_the_switch():
    t = tiers.resolve({"reasoning_effort": "medium"})
    sw = PS.switch(t)
    check(sw == {"on": True, "source": "default", "mode": "both",
                 "mode_source": "default"},
          "ON by default (operator 2026-10-06), mode both",
          json.dumps(sw))
    os.environ["YAMADORI_PACKAGE_SKILLS"] = "1"
    os.environ["YAMADORI_PACKAGE_SKILLS_MODE"] = "router"
    try:
        sw = PS.switch(tiers.resolve({"reasoning_effort": "medium"}))
        low = PS.switch(tiers.resolve({"reasoning_effort": "low"}))
    finally:
        os.environ.pop("YAMADORI_PACKAGE_SKILLS", None)
        os.environ.pop("YAMADORI_PACKAGE_SKILLS_MODE", None)
    check(sw["on"] and sw["mode"] == "router" and sw["mode_source"] == "env"
          and not low["on"] and low["source"] == "tier",
          "YAMADORI_PACKAGE_SKILLS=1 and its mode turn it on at medium; "
          "never at low (no MCP tools there)", json.dumps([sw, low]))
    feats = tiers.from_header('{"package_skills": true, '
                              '"package_skills_mode": "both"}')
    sw = PS.switch(tiers.resolve({"reasoning_effort": "medium"},
                                 overrides=feats))
    check(sw["on"] and sw["source"] == "header" and sw["mode"] == "both"
          and sw["mode_source"] == "header",
          "the header forces it and picks the mode", json.dumps(sw))
    bad = PS.switch(tiers.resolve(
        {"reasoning_effort": "medium"},
        overrides=tiers.from_header('{"package_skills_mode": "nonsense"}')))
    check(bad["mode"] == PS.DEFAULT_MODE and "unknown mode" in bad["mode_source"],
          "an unknown mode is ignored, and says so", json.dumps(bad))
    check("package_skills" in tiers.BEHAVIOURS
          and "package_skills" not in tiers.OFF_BY_DEFAULT
          and tiers.behaviours(tiers.resolve({"reasoning_effort": "high"}))[
              "package_skills"] == {"on": True, "source": "default"},
          "listed with the other switches (x_yamadori.progress), on")


# ============================================================ 4. triggers ====
_REAL_SPEC = skill_packages.package_of_specifier
_REAL_DETECT = skill_packages.detect
KNOWN = {"koota": "koota", "math": "math", "@react-three/fiber":
         "@react-three/fiber", "zustand": None}


def _fake_spec(spec: str):
    base = "/".join(spec.split("/")[:2]) if spec.startswith("@") else \
        spec.split("/")[0]
    return KNOWN.get(base)


def _fake_detect(msgs):
    """Canned for a write that imports koota; else nothing (the real detector
    is gated by test_skill_packages)."""
    for m in msgs:
        for c in m.get("tool_calls") or []:
            a = (c.get("function") or {}).get("arguments") or ""
            if "from 'koota'" in a:
                return {"koota": {"why": [{"how": "import", "what": "koota",
                                           "where": "code",
                                           "strength": "strong"}],
                                  "version": None, "events": []}}
            if "createWorld" in a:
                return {"koota": {"why": [{"how": "symbol",
                                           "what": "createWorld",
                                           "where": "code",
                                           "strength": "strong"}],
                                  "version": None, "events": []}}
    return {}


def _call(cid: str, name: str, args: dict) -> dict:
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def _step(call: dict, result: str, user: str = "Build a game with koota."
          ) -> list[dict]:
    return [{"role": "user", "content": user},
            {"role": "assistant", "content": "", "tool_calls": [call]},
            {"role": "tool", "tool_call_id": call["id"], "content": result}]


def test_the_triggers():
    skill_packages.package_of_specifier = _fake_spec
    skill_packages.detect = _fake_detect
    try:
        trig = PS.step_triggers(_step(
            _call("c1", "terminal", {"command": "npm install koota@0.6.6"}),
            "added 3 packages"), POOL)
        check([(t["trigger"], t["package"], t["version"]) for t in trig]
              == [("install", "koota", "0.6.6")]
              and "install line" in trig[0]["evidence"],
              "an install line the model ran (its command) fires install",
              json.dumps(trig))
        trig = PS.step_triggers(_step(
            _call("c2", "terminal", {"command": "pnpm i"}),
            "$ npm install koota math\nadded 4 packages"), POOL)
        check({(t["trigger"], t["package"]) for t in trig}
              == {("install", "koota"), ("install", "math")},
              "an install line a result echoes (a shell prompt first) fires "
              "install", json.dumps(trig))
        trig = PS.step_triggers(_step(
            _call("c3", "terminal", {"command": "cat README.md"}),
            "## Install\nnpm install koota\n"), POOL)
        check(trig == [], "an install line in a README the model only read "
              "is not an action", json.dumps(trig))
        pj = '{"dependencies": {"koota": "^0.6.6"}}'
        trig = PS.step_triggers(_step(
            _call("c4", "read_file", {"path": "package.json"}), pj), POOL)
        check(trig == [], "a package.json the model READ fires nothing "
              "(evidence_view's set-asides stand)", json.dumps(trig))
        trig = PS.step_triggers(_step(
            _call("c5", "terminal", {"command": "cat package.json"}), pj),
            POOL)
        check(trig == [], "nor does `cat package.json`", json.dumps(trig))
        trig = PS.step_triggers(_step(
            _call("c6", "write_file", {"path": "src/a.ts", "content":
                  "import { createWorld } from 'koota'\n"}), "wrote"), POOL)
        check([(t["trigger"], t["package"]) for t in trig]
              == [("import", "koota")],
              "an import of the package in code the model wrote fires "
              "import (one event per package)", json.dumps(trig))
        trig = PS.step_triggers(_step(
            _call("c7", "write_file", {"path": "src/b.ts", "content":
                  "const w = createWorld()\n"}), "wrote"), POOL)
        check([(t["trigger"], t["package"]) for t in trig]
              == [("new_area", "koota")],
              "a first write that uses a package's own symbol fires "
              "new_area", json.dumps(trig))
        trig = PS.step_triggers(_step(
            _call("c8", "terminal", {"command": "npx tsc"}),
            "Error: Cannot find module 'koota'\n    at x.ts:1"), POOL)
        check(("error", "koota") in {(t["trigger"], t["package"])
                                     for t in trig},
              "an error whose own lines name the package fires error",
              json.dumps(trig))
        trig = PS.step_triggers(_step(
            _call("c9", "terminal", {"command": "npx tsc"}),
            "Error: Cannot find module 'left-pad'\n"), POOL)
        check(trig == [], "an error naming a package the registry does not "
              "know fires nothing", json.dumps(trig))
        trig = PS.step_triggers([{"role": "user", "content": "use koota"}],
                                POOL)
        check(trig == [], "a user turn (no step) fires nothing")
        # delivery: body first, recall later, once per trigger, none without
        # an armed skill
        st = {}
        c = ctx(tool=None, trigger="install", evidence="install line: koota",
                key="K1|trigger:install:koota")
        text, recs, st = PS.decide(st, KOOTA, c, POOL)
        check("- DO: " + KOOTA_LEAD_ITEM in text and recs[0]["trigger"]
              == "install" and recs[0]["evidence"].startswith("install line")
              and recs[0]["skills"][0]["form"] == "body",
              "the trigger delivers the lead's body the first time, with "
              "the trigger and its evidence in the record",
              json.dumps(recs)[:300])
        again, ra, st = PS.decide(st, KOOTA, dict(c), POOL)
        check(again == text and ra[0].get("replayed"),
              "the same request again (a retry) gets the same text, "
              "decided once")
        t2, r2, st = PS.decide(st, KOOTA, ctx(
            tool=None, trigger="error", evidence="error names module",
            key="K2|trigger:error:koota"), POOL)
        check(t2.startswith("\n---\nRemember (craft koota-traits"),
              "an error later brings the recall line", t2[:120])
        t3, r3, st = PS.decide(st, KOOTA, ctx(
            tool=None, trigger="import", evidence="x",
            key="K3|trigger:import:koota"), POOL)
        check(t3 == "" and "only an error" in r3[0]["why"],
              "a package already given: an install, an import or a first "
              "write brings nothing back; only an error does (the chart's "
              "recall events)", json.dumps(r3))
        t3b, r3b, st = PS.decide(st, KOOTA, ctx(
            tool=None, trigger="error", evidence="x",
            key="K3b|trigger:error:koota"), POOL)
        check(t3b == "" and "already fired" in r3b[0]["why"],
              "…and each trigger fires once per package and major",
              json.dumps(r3b))
        t4, r4, _ = PS.decide({}, [{"name": "koota", "version": "0.6.6"}],
                              ctx(tool=None, trigger="install", evidence="x",
                                  key="K4|t"), [UNRELATED])
        check(t4 == "" and "no armed" in r4[0]["why"],
              "nothing when the package has no armed skill", json.dumps(r4))
    finally:
        skill_packages.package_of_specifier = _REAL_SPEC
        skill_packages.detect = _REAL_DETECT


# ================================================ 5. the served template ======
class PConv(H.Conv):
    def __init__(self, tag, mode="inject", **kw):
        f = {"skills": False, "package_skills": True,
             "package_skills_mode": mode}
        f.update(kw.pop("features", None) or {})
        super().__init__(tag, features=f, **kw)


def _hop(t: dict, which: int, cid: str) -> str:
    msgs = t["gens"][which]["request"]["messages"]
    return next((m["content"] for m in msgs if m.get("role") == "tool"
                 and m.get("tool_call_id") == cid), "")


def _session(mode: str, tag: str, streamed: bool = False) -> dict:
    T.slots.reset(n=4)
    H.fresh_host()
    c = PConv(tag, mode)
    t1 = c.turn([T.reply("", reasoning="Which package is pmndrs math?",
                         calls=[T.call("yama_find_package", {
                             "query": "pmndrs math", "ecosystem": "npm"},
                             "m1")]),
                 T.reply("", reasoning="It is `math`.",
                         calls=[T.call("write_file", H.PKG, "w1")])],
                user="Build a voxel scene with r3f and pmndrs math.",
                streamed=streamed)
    c.tool_result("w1", "wrote package.json")
    t2 = c.turn([T.reply("Done.")], streamed=streamed)
    return {"c": c, "t1": t1, "t2": t2}


def test_through_the_served_template():
    s = _session("inject", "inj")
    t1, t2 = s["t1"], s["t2"]
    hop = _hop(t1, 1, "m1")
    check(hop.startswith("SOURCE: the npm registry")
          and "- math 0.1.0" in hop and "\n---\n" + P.CRAFT_HEADER in hop
          and "Call vec3.add(out, a, b)" in hop
          and hop.index("- math 0.1.0") < hop.index(P.CRAFT_HEADER),
          "the section rides at the END of the find_package result, inside "
          "the hidden hop", hop[-420:])
    names = [t["function"]["name"] for t in t1["gens"][0]["request"]["tools"]]
    check("yama_recall_craft" in names,
          "yama_recall_craft is offered with the MCP tools although the "
          "`skills` flag is off", json.dumps(names))
    sk_rec = t1["x"].get("skills") or {}
    pk = sk_rec.get("package") or []
    check(sk_rec.get("package_switch", {}).get("on") is True
          and pk and pk[0]["tool"] == "yama_find_package"
          and pk[0]["package"] == "math" and pk[0]["mode"] == "inject"
          and pk[0]["skills"][0]["name"] == "math-data-oriented-functions"
          and pk[0]["chars"] > 0 and all(p["package"] != "mathjs"
                                         for p in pk),
          "x_yamadori.skills.package: tool, package, version, major, skills, "
          "chars, why (mathjs, which has no skills, is not there)",
          json.dumps(sk_rec)[:400])
    g2 = t2["gens"][0]["request"]["messages"]
    check(T._extends(t1, t2, "[package] the request after the section"),
          "the request after the hidden hop with a section EXTENDS what the "
          "slot holds (the served template, the ledger's replay)")
    check(next((m["content"] for m in g2 if m.get("role") == "tool"
                and m.get("tool_call_id") == "m1"), None) == hop,
          "the section is replayed byte for byte with the hop")
    # the model wrote math into package.json: the install trigger (the REAL
    # detector) finds the package already given by the lookup and adds
    # nothing -- only an error brings it back
    tool_msg = next((m["content"] for m in g2 if m.get("role") == "tool"
                     and m.get("tool_call_id") != "m1"), "")
    check(tool_msg == "wrote package.json",
          "a package already given by its lookup: the install trigger adds "
          "nothing to the next tool result", tool_msg[-200:])
    # a second lookup of the same package: nothing identical twice
    c = s["c"]
    c.tool_result("w1", "ok")
    t3 = c.turn([T.reply("", calls=[T.call("yama_list_package_versions", {
        "package": "math", "ecosystem": "npm"}, "m2")]),
        T.reply("Done again.")], user="Which versions of math exist?")
    hop2 = _hop(t3, 1, "m2")
    pk3 = (t3["x"].get("skills") or {}).get("package") or []
    check("Remember (craft math-data-oriented-functions)" in hop2
          and "Call vec3.add" not in hop2.split("Remember")[-1]
          or (pk3 and "last one given" in pk3[0].get("why", "")),
          "the same package in a later lookup: a recall line, not the body "
          "(or nothing when the recall was the last line given)",
          json.dumps(pk3)[:300])
    check(T._extends(t2, t3, "[package] the request after the second "
                             "lookup"),
          "…and that request extends the slot too")
    # the switch off: nothing rides
    T.slots.reset(n=4)
    off = H.Conv("off", features={"skills": False, "package_skills": False})
    t = off.turn([T.reply("", calls=[T.call("yama_find_package", {
        "query": "pmndrs math", "ecosystem": "npm"}, "o1")]),
        T.reply("ok")], user="Build with pmndrs math.")
    ho = _hop(t, 1, "o1")
    names_off = [x["function"]["name"] for x in
                 t["gens"][0]["request"]["tools"]]
    check(P.CRAFT_HEADER not in ho and "yama_recall_craft" not in names_off
          and "package" not in (t["x"].get("skills") or {}),
          "the switch off: no section, no craft tool, no "
          "record")


def test_the_router_through_the_template_and_a_recall_after_it():
    s = _session("router", "rt")
    t1, t2 = s["t1"], s["t2"]
    hop = _hop(t1, 1, "m1")
    rows = _rows(hop)
    check("Craft for math 0 from this service's library" in hop
          and len(rows) == 2 and "Call vec3.add" not in hop,
          "router: the table rides in the hidden hop, no body", hop[-400:])
    check(T._extends(t1, t2, "[router] the request after the table"),
          "router: the next request extends the slot")
    c = s["c"]
    c.tool_result("w1", "ok")
    t3 = c.turn([T.reply("", reasoning="Read that craft.", calls=[T.call(
        "yama_recall_craft", {"name_or_topic": "math-hot-path-pitfalls"},
        "k1")]), T.reply("Done.")], user="Now make it fast.")
    got = _hop(t3, 1, "k1")
    check(got.startswith("Craft math-hot-path-pitfalls") and "allocate "
          "vectors" in got,
          "a name the router listed: the craft in full (the name path is "
          "unchanged)", got[:120])
    pk = (t3["x"].get("skills") or {}).get("package") or []
    fol = [r for r in pk if r.get("event") == "recall_after_section"]
    check(fol and fol[0]["craft"] == "math-hot-path-pitfalls"
          and fol[0]["listed_by_router"] and fol[0]["mode"] == "router"
          and fol[0]["steps_later"] >= 1,
          "x_yamadori.skills.package records the recall after the router: "
          "which craft, listed by the router, how many steps later",
          json.dumps(pk)[:300])


def test_streamed_is_blocking_with_a_section():
    a = _session("inject", "sb-a")
    b = _session("inject", "sb-b", streamed=True)
    check(_hop(a["t1"], 1, "m1") == _hop(b["t1"], 1, "m1")
          and a["t1"]["m"].get("content") == b["t1"]["m"].get("content"),
          "streamed == blocking: the same section")
    check(T._extends(b["t1"], b["t2"], "[package] streamed"),
          "streamed: the next request extends the slot")


# ----------------------------------------------------------- Responses wire --
def test_the_section_over_responses():
    import mcp_config

    class PR(RT.RClient):
        def chat_of(self, body):
            chat, ctx_ = RT.R.to_chat(body)
            chat.update(_account=self.account, _client_ip="127.0.0.1",
                        _public_base=RT.BASE, _session_token="",
                        _features=json.dumps({
                            "skills": False, "package_skills": True,
                            "package_skills_mode": "both"}))
            return chat, ctx_
    spec = json.loads(json.dumps(mcp_config.PACKAGELENS))
    spec.update(id="packagelens", runtime="local",
                command=[sys.executable, H.FAKE], call_timeout_s=10.0,
                call_timeout_why="test fixture: the fake answers at once")
    spec.pop("image", None)
    H.fresh_host(spec)
    for stream in (False, True):
        T.slots.reset(n=4)
        T.compaction.reset()
        c = PR(f"pk-{stream}", stream=stream, cache_key=f"pk-{stream}")
        t1 = c.turn([T.reply("", reasoning="Which package?", calls=[T.call(
            "yama_find_package", {"query": "pmndrs math",
                                  "ecosystem": "npm"}, "pr1")]),
            T.reply("", calls=[T.call("write_file", H.PKG, "pr2")])],
            user="Build a voxel scene with pmndrs math.")
        label = "streamed" if stream else "blocking"
        hop = next((m["content"] for m in t1["gens"][1]["request"]["messages"]
                    if m.get("role") == "tool"
                    and m.get("tool_call_id") == "pr1"), "")
        check("Call vec3.add(out, a, b)" in hop
              and "Craft for math 0 from this service's library" in hop,
              f"responses {label}: both -- the lead's body and the router "
              f"table ride in the hidden hop", hop[-420:])
        pk = ((t1["x"].get("skills") or {}).get("package")) or []
        check(pk and pk[0]["mode"] == "both" and pk[0]["package"] == "math",
              f"responses {label}: x_yamadori.skills.package",
              json.dumps(pk)[:200])
        c.tool_output("pr2", "wrote package.json")
        t2 = c.turn([T.reply("Done.")])
        T._extends(t1, t2, f"responses {label}: after the section")
        check(next((m["content"] for m in t2["gens"][0]["request"]["messages"]
                    if m.get("role") == "tool"
                    and m.get("tool_call_id") == "pr1"), None) == hop,
              f"responses {label}: the section replays byte for byte")


# =========================================== 5b. triggers through the turn ====
def test_a_trigger_rides_the_tool_result_and_replays():
    skill_packages.package_of_specifier = _fake_spec
    skill_packages.detect = _fake_detect
    try:
        T.slots.reset(n=4)
        H.fresh_host()
        c = PConv("trig", "inject", tools=[T.WRITE, {
            "type": "function", "function": {
                "name": "terminal", "description": "run",
                "parameters": {"type": "object", "properties": {
                    "command": {"type": "string"}}}}}])
        c.turn([T.reply("", reasoning="Install it.", calls=[T.call(
            "terminal", {"command": "npm install koota@0.6.6"}, "i1")])],
            user="Build a game with koota.")
        c.tool_result("i1", "added 3 packages")
        t2 = c.turn([T.reply("", calls=[T.call("write_file", {
            "path": "a.ts", "content": "x"}, "i2")])])
        sent = next((m["content"] for m in
                     t2["gens"][0]["request"]["messages"]
                     if m.get("role") == "tool"), "")
        check(sent.startswith("added 3 packages\n---\n" + P.CRAFT_HEADER)
              or ("added 3 packages" in sent and P.CRAFT_HEADER in sent
                  and "world.spawn" in sent),
              "the install line fires koota's lead skill, appended to the "
              "tool result the request ends on", sent[-300:])
        pk = (t2["x"].get("skills") or {}).get("package") or []
        check(pk and pk[0]["trigger"] == "install"
              and pk[0]["package"] == "koota"
              and pk[0]["evidence"].startswith("install line"),
              "x_yamadori.skills.package: trigger, evidence, skills",
              json.dumps(pk)[:300])
        c.tool_result("i2", "wrote a.ts")
        t3 = c.turn([T.reply("Done.")])
        again = next((m["content"] for m in
                      t3["gens"][0]["request"]["messages"]
                      if m.get("role") == "tool"
                      and m.get("tool_call_id") not in ("w1",)
                      and "added 3 packages" in (m.get("content") or "")),
                     "")
        check(again == sent, "the next request replays the trigger's text "
              "from the ledger byte for byte", again[-200:])
        check(T._extends(t2, t3, "[trigger] the request after a trigger"),
              "…and extends the slot")
    finally:
        skill_packages.package_of_specifier = _REAL_SPEC
        skill_packages.detect = _REAL_DETECT


# ============================================= 6. the question path ==========
class FakeTurn:
    """A fake jjava: records the typed questions it is asked and answers by
    the scripted `choice` / `gate` (decider_bonsai's answer shapes)."""
    log: list = []
    choice: dict | None = None        # {craft id: p}
    gate: float = 0.9
    down: bool = False
    on = True

    def __init__(self, msgs, state, info, ctx_):
        self.state, self.info = state, info
        self.answers = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def decide(self, qs):
        if FakeTurn.down:
            raise D.DeciderUnavailable("MODEL_DOWN", "the model server is "
                                       "down", True, "retry")
        out = []
        for q in qs:
            FakeTurn.log.append({"q": q, "state": self.state})
            if q["type"] == "choice":
                ch = FakeTurn.choice or {}
                probs = {k: ch.get(k, 0.01) for k in q["keys"]}
                z = sum(probs.values())
                probs = {k: v / z for k, v in probs.items()}
                out.append({"probabilities": probs, "decision_id": "d-c",
                            "diagnostics": {"tie": False,
                                            "disagreement": 0.02,
                                            "argmax_agree": True}})
            else:
                out.append({"noul": FakeTurn.gate, "decision_id": "d-g",
                            "diagnostics": {"tie": False,
                                            "disagreement": 0.05}})
        return out


def _qctx(**kw):
    base = {"messages": [{"role": "user", "content": "Build a game with "
                          "koota and make it fast."},
                         {"role": "assistant", "content": "",
                          "tool_calls": [_call("z1", "terminal",
                                               {"command": "ls"})]},
                         {"role": "tool", "tool_call_id": "z1",
                          "content": "src/ a.ts"}],
            "account": "acct", "lineage": "lin", "package": "koota",
            "major": 0, "turn_factory": FakeTurn,
            "embed": lambda texts: [[1.0, 0.0] for _ in texts]}
    base.update(kw)
    return base


def _embed_pool(monkey: bool = True):
    """skill_match.rank_all over a fake doc index: rank by a word overlap so
    the shortlist is deterministic and needs no embedder."""
    import skill_match

    def rank_all(state, pool, embed=None):
        w = set(re.findall(r"[a-z0-9]+", state.lower()))
        best = {}
        for s in pool:
            sw = set(re.findall(r"[a-z0-9]+", (s["name"] + " " +
                                               s["description"]).lower()))
            best[s["id"]] = len(w & sw) / 10.0
        return best, {"ok": True}
    skill_match.rank_all = rank_all


def test_the_question_path():
    _embed_pool()
    # a name: unchanged (no jjava, no question)
    FakeTurn.log = []
    text, rec = skill_select.read_craft(
        {"name_or_topic": "koota-react-integration"}, POOL, ctx=_qctx())
    check(text.startswith("Craft koota-react-integration")
          and rec["how"] == "name" and "query" not in rec
          and FakeTurn.log == [],
          "a craft's name: that craft, as today, and jjava is not asked",
          text[:80])
    text, rec = skill_select.read_craft(
        {"name_or_topic": "r3f-useframe"}, POOL, ctx=_qctx())
    check(text.startswith("Craft r3f-useframe-1") and rec["how"] == "name"
          and FakeTurn.log == [],
          "a name less its numeric id is a near-exact name (one craft)")
    # a question: narrowed, chosen with no "none", gated, untuned
    FakeTurn.choice = {"id-koota-queries-and-systems": 0.8}
    FakeTurn.gate = 0.9
    text, rec = skill_select.read_craft(
        {"name_or_topic": "how do I iterate koota entities each frame in a "
                          "system?"}, POOL, ctx=_qctx())
    qs = [x["q"] for x in FakeTurn.log]
    q = rec["query"]
    check(text.startswith("Craft koota-queries-and-systems")
          and rec["how"] == "question"
          and rec["found"] == "id-koota-queries-and-systems",
          "a question: the craft jjava picked, in full", text[:100])
    check([x["type"] for x in qs] == ["choice", "noul"]
          and "none" not in qs[0]["keys"]
          and not any("None of these" in o for o in qs[0]["options"])
          and qs[0]["none"] is None,
          "ONE choice over the shortlist WITHOUT a 'none' option, then ONE "
          "noul relevance gate", json.dumps([x["type"] for x in qs]))
    check(qs[1]["text"].startswith("The craft koota-queries-and-systems "
                                   "answers the question: how do I iterate")
          and all("[koota-" in o or "[math-" in o or "[r3f-" in o
                  or "[typescript-" in o for o in qs[0]["options"]),
          "the gate asks about the craft by name; the choice shows each "
          "craft under its name", qs[1]["text"][:120])
    st = FakeTurn.log[0]["state"]
    check(st.startswith(craft_query.QUESTION_HEAD + "\nhow do I iterate")
          and skill_inject.GOAL_HEAD not in st
          or "OPENING REQUEST" in st or "NEWEST" in st or "LATEST STEP" in st,
          "the state is framed: the question, then the goal and the step's "
          "evidence like skill_inject.framed_state", st[:260])
    check(q["untuned"] is True and q["tier"] == "untuned"
          and q["chosen"] == "id-koota-queries-and-systems"
          and 0.7 < q["p"] < 0.9 and abs(q["gate_p"] - 0.9) < 1e-6
          and q["choice"]["probs"] and q["gate"]["noul"]
          and q["question"].startswith("how do I iterate")
          and len(q["question"]) <= craft_query.QUESTION_CHARS
          and q["shortlist"] and q["package"] == "koota"
          and q["retrieval"] == "embedding" and q["ms"] >= 0
          and "question_full" in q,
          "the record: question, shortlist, chosen, the choice's reads, the "
          "gate's p, untuned, package", json.dumps(q)[:500])
    check("koota-quarantined" not in json.dumps(q["shortlist"])
          and "id-koota-quarantined" not in q["shortlist"]
          and q["shortlist"][0].startswith("id-koota"),
          "only armed, proven crafts are shortlisted, the package's first",
          json.dumps(q["shortlist"]))
    # untuned: a LOW gate still returns the best craft
    FakeTurn.gate = 0.05
    text, rec = skill_select.read_craft(
        {"name_or_topic": "how to do something about koota"}, POOL,
        ctx=_qctx())
    check(text.startswith("Craft ") and rec["query"]["untuned"]
          and rec["query"]["gate_p"] == 0.05,
          "untuned: the best craft is returned even when the gate says no "
          "(no cut is invented); the record says untuned")
    # tuned: a confident "no" is the honest answer with next steps
    skill_inject.THRESHOLDS["testmodel"] = {craft_query.QSET_GATE: {
        "act_yes": 0.9, "act_no": 0.1, "caution_yes": 0.7, "caution_no": 0.3}}
    try:
        FakeTurn.gate = 0.04
        text, rec = skill_select.read_craft(
            {"name_or_topic": "how do I deploy to kubernetes?"}, POOL,
            ctx=_qctx(model="testmodel"))
        d = json.loads(text)
        check(d["error"] == "NO_SUCH_CRAFT" and d["retryable"] is True
              and d["near"] and any("yama_read_package_readme" in r["action"]
                                    and "koota" in r["action"]
                                    for r in d["remedies"])
              and any("one of `near`" in r["action"] for r in d["remedies"])
              and rec["query"]["answered"] is False and not rec["found"]
              and rec["query"]["untuned"] is False
              and rec["query"]["tier"] == "high",
              "tuned and confident 'no': NO_SUCH_CRAFT with the nearest "
              "crafts and the README as the next step", text[:300])
        FakeTurn.gate = 0.95
        text, rec = skill_select.read_craft(
            {"name_or_topic": "how do I iterate koota entities each frame?"},
            POOL, ctx=_qctx(model="testmodel"))
        check(text.startswith("Craft ") and not rec["query"]["untuned"],
              "tuned and the gate says yes: the craft")
    finally:
        skill_inject.THRESHOLDS.pop("testmodel", None)
    # jjava down: the honest answer, naming the nearest
    FakeTurn.down = True
    try:
        text, rec = skill_select.read_craft(
            {"name_or_topic": "how do I do koota things?"}, POOL,
            ctx=_qctx())
    finally:
        FakeTurn.down = False
    d = json.loads(text)
    check(d["error"] == "NO_SUCH_CRAFT" and "jjava could not be asked"
          in d["reason"] and d["near"]
          and rec["query"]["failure"]["code"] == "MODEL_DOWN",
          "jjava unavailable: the same honest answer, the failure recorded",
          text[:240])
    # an empty library: honest, with the next step
    text, rec = skill_select.read_craft(
        {"name_or_topic": "anything at all?"}, [], ctx=_qctx())
    d = json.loads(text)
    check(d["error"] == "NO_SUCH_CRAFT" and d["near"] == []
          and rec["query"]["answered"] is False,
          "no armed craft: NO_SUCH_CRAFT, nothing to name")
    # without a ctx (a caller that is not the proxy): the old behaviour
    text, rec = skill_select.read_craft({"name_or_topic": "koota queries"},
                                        POOL)
    check(json.loads(text)["error"] == "NO_SUCH_CRAFT"
          and "query" not in rec and rec["how"] == "unknown",
          "read_craft with no ctx keeps its old topic listing")


def test_the_question_through_the_proxy():
    """The hop, the log, the record and the ledger replay."""
    _embed_pool()
    craft_query.make_turn = lambda m, s, i, c: FakeTurn(m, s, i, c)
    FakeTurn.choice = {"id-math-hot-path-pitfalls": 0.9}
    FakeTurn.gate = 0.9
    FakeTurn.log = []
    T.slots.reset(n=4)
    H.fresh_host()
    # a test account: traffic class `test` is logged and marked
    c = PConv("q1", "router")
    # (arguments in sorted order: a text-final turn has no warm, and the
    # replay renders a hidden call's arguments sorted -- a harness fact
    # test_ledger gates, not this channel's)
    t1 = c.turn([T.reply("", calls=[T.call("yama_find_package", {
        "ecosystem": "npm", "query": "pmndrs math"}, "q0")]),
        T.reply("", reasoning="Ask how to stop allocating.", calls=[T.call(
            "yama_recall_craft", {"name_or_topic": "how do I avoid "
                                  "allocating vectors in updateEach in "
                                  "math?"}, "q1")]),
        T.reply("Done.")],
        user="Make the math voxel scene fast.")
    hop = _hop(t1, 2, "q1")
    check(hop.startswith("Craft math-hot-path-pitfalls (this service's "
                         "library):") and "allocate vectors" in hop,
          "the model's question over the served path: the craft jjava "
          "picked, as a hidden hop", hop[:120])
    cq = (t1["x"].get("craft") or {}).get("query") or []
    check(len(cq) == 1 and cq[0]["chosen"] == "id-math-hot-path-pitfalls"
          and cq[0]["package"] == "math" and cq[0]["untuned"]
          and "question_full" not in cq[0]
          and cq[0]["shortlist"][0].startswith("id-math"),
          "x_yamadori.craft.query, boosted to the conversation's latest "
          "package (math), no full text in the record", json.dumps(cq)[:400])
    reads = (t1["x"].get("craft") or {}).get("reads") or []
    check(reads and reads[0]["how"] == "question",
          "x_yamadori.craft.reads says how it was found: question")
    fol = [r for r in (t1["x"].get("skills") or {}).get("package") or []
           if r.get("event") == "recall_after_section"]
    check(fol and fol[0]["question"] is True and fol[0]["mode"] == "router",
          "a question after a router section is recorded as a recall after "
          "it, marked as a question", json.dumps(fol))
    con = skill_learn._db()
    try:
        rows = [dict(r) for r in con.execute("SELECT * FROM craft_queries")]
    finally:
        con.close()
    check(len(rows) == 1 and rows[0]["traffic"] in ("test", "client")
          and rows[0]["question"].startswith("how do I avoid allocating")
          and rows[0]["chosen"] == "id-math-hot-path-pitfalls"
          and rows[0]["answered"] == 1 and rows[0]["untuned"] == 1
          and rows[0]["package"] == "math"
          and json.loads(rows[0]["reads"])["gate"]["noul"] == 0.9
          and "acct" not in json.dumps(rows[0])
          and T.ACCOUNT not in json.dumps(rows[0]),
          "the durable log row: the question, the shortlist, the choice's "
          "and the gate's reads, the traffic class -- never the account or "
          "key", json.dumps(rows)[:400])
    check(skill_learn.traffic_of(c.account) in ("test", "client"),
          "the traffic class comes from corpus.account_traffic (a test "
          "account is `test`, never learned from)")
    c.tool_result("q1", "ok")
    t2 = c.turn([T.reply("Fine.")], user="Thanks.")
    check(_hop(t2, 0, "q1") == hop and T._extends(t1, t2, "[question] the "
                                                  "request after the hop"),
          "the ledger replays the question's hop byte for byte and the slot "
          "extends")


# ====================================== 7. the per-request craft cap ======
CRAFT_NAMES = ["math-data-oriented-functions", "math-hot-path-pitfalls",
               "koota-traits-and-entities", "koota-queries-and-systems"]


def _craft_calls(n: int, tag: str) -> list[dict]:
    return [T.call("yama_recall_craft",
                   {"name_or_topic": CRAFT_NAMES[i % len(CRAFT_NAMES)]},
                   f"{tag}{i}") for i in range(n)]


def _tool_msgs(gen: dict, tag: str) -> list[str]:
    return [m["content"] for m in gen["request"]["messages"]
            if m.get("role") == "tool"
            and re.fullmatch(re.escape(tag) + r"\d+",
                             str(m.get("tool_call_id", "")))]


def test_the_craft_cap():
    import skill_match
    limit = skill_match.BODIES_PER_DECISION
    T.slots.reset(n=4)
    H.fresh_host()
    c = PConv("cap1", "both")
    # request 1: a package lookup, TWELVE craft calls in one generation, then
    # the model's own write -- the probe's failure (4 of 15 skill-arm trials)
    t1 = c.turn([T.reply("", reasoning="Look it up.", calls=[T.call(
        "yama_find_package", {"ecosystem": "npm", "query": "pmndrs math"},
        "p0")]),
        T.reply("", reasoning="Read everything.", calls=_craft_calls(12, "c")),
        T.reply("", reasoning="Now write it.", calls=[T.call(
            "write_file", H.PKG, "w1")])],
        user="Build a voxel scene with pmndrs math.")
    last = t1["gens"][-1]
    got = _tool_msgs(last, "c")
    ran = [g for g in got if g.startswith("Craft ")]
    capped = [g for g in got if '"CRAFT_CAP"' in g]
    check(len(got) == 12 and len(ran) == limit and len(capped) == 12 - limit,
          f"12 craft calls in one request: the first {limit} run, the other "
          f"{12 - limit} are capped (skill_match.BODIES_PER_DECISION)",
          f"{len(ran)} ran, {len(capped)} capped")
    d = json.loads(capped[0])
    check(d["ok"] is False and d["retryable"] is True
          and f"already has {limit} crafts" in d["reason"]
          and "continue with the task" in d["remedies"][0]["action"]
          and "later step" in d["remedies"][0]["action"]
          and d["remedies"][0]["fixable_by"] == "agent",
          "a capped call: the situation, retryable as a fact, and the next "
          "step (Failure returns carry the next step)", capped[0][:300])
    tt = t1["x"].get("tool_turns") or {}
    check(tt.get("turns") == 1 and tt.get("hit") is False
          and (t1["x"].get("craft") or {}).get("capped") == 12 - limit,
          "craft calls do not count toward tool_turns (only the package "
          "lookup did); x_yamadori.craft.capped says how many were not run",
          json.dumps([tt, (t1["x"].get("craft") or {}).get("capped")]))
    calls = t1["m"].get("tool_calls") or []
    check([x["function"]["name"] for x in calls] == ["write_file"],
          "the client's own tool call still flows after the cap", json.dumps(
              calls)[:200])
    c.tool_result("w1", "wrote package.json")
    t2 = c.turn([T.reply("Done.")])
    check(T._extends(t1, t2, "[cap] the request after a capped turn"),
          "the request after the capped turn EXTENDS what the slot holds")
    replay = _tool_msgs(t2["gens"][0], "c")
    check(replay == got,
          "the ledger replays the crafts and the capped results byte for "
          "byte", f"{len(replay)} vs {len(got)}")
    # the cap is per REQUEST: the next request may read crafts again
    c.tool_result("w1", "ok")
    t3 = c.turn([T.reply("", calls=[T.call(
        "yama_recall_craft", {"name_or_topic": CRAFT_NAMES[0]}, "n1")]),
        T.reply("Fine.")], user="One more thing about math.")
    g = _tool_msgs(t3["gens"][1], "n")
    check(len(g) == 1 and g[0].startswith("Craft ")
          and (t3["x"].get("craft") or {}).get("capped") == 0,
          "a later request reads a craft again (the cap is per request)",
          json.dumps(g)[:120])
    # twelve SINGLE-call hops: still three run, nine capped, no landing
    T.slots.reset(n=4)
    H.fresh_host()
    c2 = PConv("cap2", "both")
    script = [T.reply("", calls=[T.call("yama_find_package", {
        "ecosystem": "npm", "query": "pmndrs math"}, "p0")])]
    script += [T.reply("", calls=[x]) for x in _craft_calls(12, "c")]
    script += [T.reply("", calls=[T.call("write_file", H.PKG, "w1")])]
    t = c2.turn(script, user="Build a voxel scene with pmndrs math.")
    g = _tool_msgs(t["gens"][-1], "c")
    tt = t["x"].get("tool_turns") or {}
    check(len([x for x in g if x.startswith("Craft ")]) == limit
          and len([x for x in g if '"CRAFT_CAP"' in x]) == 12 - limit
          and tt.get("turns") == 1 and tt.get("hit") is False
          and [x["function"]["name"] for x in
               t["m"].get("tool_calls") or []] == ["write_file"],
          "twelve single-call craft hops: three run, nine capped, the tool "
          "turns are not exhausted and the client's write flows",
          json.dumps(tt))
    # a flood past the tool-turn limit of capped calls LANDS (a loop of its own)
    T.slots.reset(n=4)
    H.fresh_host()
    c3 = PConv("cap3", "both")
    lim = tiers.tool_turn_limit(tiers.resolve({"reasoning_effort": "medium"}))
    script = [T.reply("", calls=[T.call("yama_find_package", {
        "ecosystem": "npm", "query": "pmndrs math"}, "p0")])]
    script += [T.reply("", calls=[x]) for x in
               _craft_calls(lim + limit + 2, "c")]
    script += [T.reply("Landed.")] * 3
    t = c3.turn(script, user="Build a voxel scene with pmndrs math.")
    tt = t["x"].get("tool_turns") or {}
    check(tt.get("hit") is True and (t["x"].get("craft") or {}).get(
        "capped") >= lim and not t["m"].get("tool_calls"),
          f"{lim} capped calls end the loop: it LANDS (tools withdrawn, an "
          "answer), recorded in tool_turns.hit", json.dumps(tt))


def test_the_craft_cap_over_responses():
    import mcp_config

    class PR(RT.RClient):
        def chat_of(self, body):
            chat, ctx_ = RT.R.to_chat(body)
            chat.update(_account=self.account, _client_ip="127.0.0.1",
                        _public_base=RT.BASE, _session_token="",
                        _features=json.dumps({
                            "skills": False, "package_skills": True,
                            "package_skills_mode": "both"}))
            return chat, ctx_
    import skill_match
    limit = skill_match.BODIES_PER_DECISION
    spec = json.loads(json.dumps(mcp_config.PACKAGELENS))
    spec.update(id="packagelens", runtime="local",
                command=[sys.executable, H.FAKE], call_timeout_s=10.0,
                call_timeout_why="test fixture: the fake answers at once")
    spec.pop("image", None)
    H.fresh_host(spec)
    for stream in (False, True):
        T.slots.reset(n=4)
        T.compaction.reset()
        c = PR(f"cap-{stream}", stream=stream, cache_key=f"cap-{stream}")
        t1 = c.turn([T.reply("", calls=[T.call("yama_find_package", {
            "ecosystem": "npm", "query": "pmndrs math"}, "p0")]),
            T.reply("", calls=_craft_calls(12, "c")),
            T.reply("", calls=[T.call("write_file", H.PKG, "pr2")])],
            user="Build a voxel scene with pmndrs math.")
        label = "streamed" if stream else "blocking"
        got = _tool_msgs(t1["gens"][-1], "c")
        tt = t1["x"].get("tool_turns") or {}
        names = ([x["name"] for x in RT._items(t1["resp"], "function_call")]
                 if not stream else
                 [e["item"]["name"] for e in t1["events"]
                  if e["type"] == "response.output_item.done"
                  and e["item"].get("type") == "function_call"])
        check(len([g for g in got if g.startswith("Craft ")]) == limit
              and len([g for g in got if '"CRAFT_CAP"' in g]) == 12 - limit
              and tt.get("turns") == 1 and tt.get("hit") is False
              and names == ["write_file"]
              and (t1["x"].get("craft") or {}).get("capped") == 12 - limit,
              f"responses {label}: 12 craft calls run 3 and cap 9, the tool "
              f"turns are not exhausted, the client's write flows",
              json.dumps(tt))
        c.tool_output("pr2", "wrote package.json")
        t2 = c.turn([T.reply("Done.")])
        T._extends(t1, t2, f"responses {label}: after the capped turn")
        check(_tool_msgs(t2["gens"][0], "c") == got,
              f"responses {label}: the capped turn replays byte for byte")


# ================================================================== main ====
def main() -> int:
    for fn in (test_the_inject_section, test_the_major_picks_the_skills,
               test_the_sizes, test_the_router_rendering,
               test_the_both_rendering, test_the_switch, test_the_triggers,
               test_through_the_served_template,
               test_the_router_through_the_template_and_a_recall_after_it,
               test_streamed_is_blocking_with_a_section,
               test_the_section_over_responses,
               test_a_trigger_rides_the_tool_result_and_replays,
               test_the_question_path, test_the_question_through_the_proxy,
               test_the_craft_cap, test_the_craft_cap_over_responses):
        print(f"\n--- {fn.__name__} ---")
        n0 = len(_results)
        try:
            fn()
        except Exception:                                        # noqa: BLE001
            check(False, f"{fn.__name__} itself raised",
                  traceback.format_exc().strip().split("\n")[-1])
            traceback.print_exc()
        for ok, name, detail in _results[n0:]:
            print(("  pass  " if ok else "  FAIL  ") + name
                  + (f"   <- {detail}" if not ok and detail else ""))
    mcp_host.stop_all()
    T._srv.shutdown()
    passed = sum(1 for ok, _, _ in _results if ok)
    total = len(_results)
    print(f"\n{'=' * 70}\n  {passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
