#!/usr/bin/env python
"""The skill pipeline's 2026-09-27 fixes, asserted. No GPU, no network, no
live store: the model is `model.post`, replaced by a scripted server that
answers by the stage's system prompt and records every body it was sent.

    python mcp/test_skill_pipeline_fixes.py      -> "N/M checks passed"

  1. THE THINKING CAP. Every pipeline model stage (screen_model, distil,
     decompose, tag, tests, faithful) runs as the named helper job
     `skill.<purpose>` -- role helper, the job on the thread -- so
     tiers.budget caps its thinking at HELPER_THINKING (no pipeline job has a
     JOB_THINKING row). Before, the calls ran as role main with no cap.
  2. THE LEAD PLUMBING (the template's ask is HELD for design review; these
     checks drive a scripted decompose reply that carries a `lead:` block).
     A block marked `lead:` becomes ONE lead skill for its package, quote-
     and faithful-checked like any child, whose SKILL.md says
     `metadata.yamadori.lead_for: <package>` and whose served row carries
     `lead_for`; a lead in a later part, or naming a package the source
     never names, is not a lead.
  3. THE SCREEN. Zero-width typography is stripped and recorded (three.js
     r185 llms-full.txt line 457 passes); Unicode tag characters and bidi
     overrides still quarantine, zero-width characters beside them or not.
  4. MULTI-DOMAIN FILING (tag/5). `all_of` from the tag stage is kept only
     for a vocabulary term the text names, leaves the OR list, and is
     refused when nothing would remain to key on; koota-react-integration
     arms on the pipeline's own filing ("koota, with React").
  5. THE WATCH PATH. A changed frontier SKILL.md goes back through
     decompose: a re-proposed child becomes a new version of itself, a
     dropped section's child is archived, the lead stays the lead.
"""
from __future__ import annotations

import json
import os
import re
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import offline_stores  # noqa: E402

_TMP = offline_stores.isolate("yamadori_test_skill_fixes_")
os.environ["YAMADORI_SKILL_CACHE_SECONDS"] = "0"
os.environ.setdefault("YAMADORI_GPU_ROOM", "0")

import budget  # noqa: E402
import model  # noqa: E402
import skill_classify as C  # noqa: E402
import skill_offline  # noqa: E402
import skill_pipeline as P  # noqa: E402
import skill_prompts  # noqa: E402
import skill_screen  # noqa: E402
import skill_select  # noqa: E402
import skills  # noqa: E402
import tiers  # noqa: E402


# THE PROVE STAGE (skill_prove, 2026-09-28) through its injectables: a fixed
# answer on both sides of every pair (a tie arms), no probe proposals. Its
# decisions are tested in mcp/test_skill_prove.py.
import skill_prove  # noqa: E402

skill_prove.CHAT = lambda body: {"choices": [{"message": {
    "content": "```ts\nexport const x: number = 1;\n```"},
    "finish_reason": "stop"}]}
skill_prove.ASK = lambda system, user, *, max_tokens, purpose="": (
    '{"pass": true, "why": "fake"}' if system == skill_prove.JUDGE_SYSTEM
    else '{"probes": []}')

# The pool size is asked of the model server (budget.pool_size); an offline
# suite states it (config.yaml's -c 181248, as AGENTS.md "One door" says).
budget._POOL = 181248
# The efforts the served template accepts (tiers.accepted_efforts reads
# /props): the set AGENTS.md "Prompting this model" names.
tiers._accepted = ("low", "medium", "xhigh")
import skill_packages  # noqa: E402


def _fake_detect(messages):
    """skill_packages.detect's shape, from the text alone: a package is in
    play when it is imported or named (the real detector reads the held
    packages' sources; a lead's tests only need in-play or not)."""
    text = "\n".join(str(m.get("content") or "") for m in messages)
    return {pkg: {"why": [], "version": None, "events": []}
            for pkg in ("koota", "zustand", "react", "three", "math")
            if re.search(r"from ['\"]" + pkg + r"(?:/[\w/]*)?['\"]|\b"
                         + pkg + r"\b", text, re.I)}


skill_packages.detect = _fake_detect
skill_select.best_cosines = lambda q, pool: ({}, {
    "ok": False, "why": "test: no embedder"})
skill_select.ask_fallback = lambda s, u: (_ for _ in ()).throw(
    RuntimeError("test: no fallback model"))

_results: list[tuple[bool, str, str]] = []
FIX = os.path.join(ROOT, "bench", "skills", "fixtures")
PMNDRS = os.path.join(FIX, "pmndrs")
SPEC = os.path.join(ROOT, "skills", "ingested", "pmndrs", "spec.json")
KOOTA = ("https://raw.githubusercontent.com/pmndrs/koota/"
         "7d1329aa82e313e715f6b7afbb8350e028c313b5/skills/koota/SKILL.md")


def check(ok, name: str, detail="") -> bool:
    _results.append((bool(ok), name, str(detail)[:400]))
    return bool(ok)


# ---------------------------------------------------------------------------
# The scripted model: model.post replaced.
# ---------------------------------------------------------------------------
LEAD = """=== SKILL ===
lead: koota
section: Koota ECS / Glossary
name: koota-ecs-overview
description: Use when building application state with koota, the ECS library: a world of entities composed from traits, changed by systems through queries. Not for wiring koota into React components (koota-react-integration).
phases: plan, implement
topics: createWorld, trait, world.spawn, world.query
# Koota ECS
- DO: model state as entities with composable traits held in one world.
  source: "Koota manages state using entities with composable traits."
- DO: spawn entities from the world; an entity is only an identifier for the traits it holds.
  source: "**Entity** - A unique identifier pointing to data defined by traits. Spawned from a world."
- DO: batch-update state through queries that fetch the entities matching an archetype.
  source: "**Query** - Fetches entities matching an archetype. The primary way to batch update state."
"""


class Scripted:
    """Answers each stage by its system prompt; records (job, body)."""

    def __init__(self, decompose: str, spec: dict, extra_tag=None,
                 faithful: str = "all", extra_tests=None, repair=None):
        self.decompose = decompose
        self.extra_tests = extra_tests or {}
        self.repair = repair              # {"i": quote} or None
        self.faithful = faithful          # all | error | silent_first
        self.spec = spec
        self.extra_tag = extra_tag or {}
        self.bodies: list[dict] = []
        self.descs: dict[str, str] = {}
        for b in skill_prompts.parse_decompose(decompose):
            self.descs[b["name"]] = b["description"]

    def who(self, user: str) -> str | None:
        hits = [n for n, d in self.descs.items() if d and d in user]
        return max(hits, key=lambda n: len(self.descs[n])) if hits else None

    def reply(self, system: str, user: str) -> str:
        if system == skill_screen.SCREEN_SYSTEM:
            return '{"verdict": "clean", "findings": []}'
        if system == skill_prompts.DECOMPOSE_SYSTEM:
            m = re.search(r"(?m)^PART: (\d+)", user)
            return self.decompose if not m or m.group(1) == "1" else ""
        if system == skill_prompts.DISTIL_SYSTEM:
            return "name: x\n# X\n"
        if system == skill_prompts.tag_system():
            n = self.who(user)
            if n in self.extra_tag:
                return json.dumps(self.extra_tag[n])
            if n in (self.spec.get("tag") or {}):
                return json.dumps(self.spec["tag"][n])
            return json.dumps({"artifacts": ["code"], "frameworks": ["koota"],
                               "phases": [], "topics": []})
        if system == skill_prompts.REVIEW_SYSTEM:
            return '{"items": []}'
        if system == skill_prompts.QUOTE_REPAIR_SYSTEM:
            return json.dumps({"items": [{"i": int(k), "quote": q} for k, q
                                         in (self.repair or {}).items()]})
        if system == skill_prompts.TESTS_SYSTEM:
            n = self.who(user)
            if n in self.extra_tests:
                return json.dumps(self.extra_tests[n])
            if n in (self.spec.get("tests") or {}):
                return json.dumps(self.spec["tests"][n])
            return json.dumps({
                "should": [{"text": "Set up the game's state with koota.\n\n"
                            "```ts\nimport { createWorld, trait } from "
                            "'koota'\nconst world = createWorld()\n```"}],
                "should_not": [{"text": "Set up the game's state with "
                                "zustand.\n\n```ts\nimport { create } from "
                                "'zustand'\n```"}], "behaviour": []})
        if system == skill_prompts.FAITHFUL_SYSTEM:
            if self.faithful == "error":
                return "I cannot judge these."
            n = len(re.findall(r"(?m)^(\d+)\. ", user))
            first = 2 if self.faithful == "silent_first" else 1
            return json.dumps({"items": [
                {"i": i, "faithful": True, "why": "the quote says it"}
                for i in range(first, n + 1)]})
        raise AssertionError("an unscripted model call")

    def post(self, body: dict, timeout: int = 0) -> dict:
        msgs = body.get("messages") or []
        sysm = next((m["content"] for m in msgs if m["role"] == "system"), "")
        user = next((m["content"] for m in msgs if m["role"] == "user"), "")
        self.bodies.append({"job": getattr(P._CURRENT, "job", None),
                            "body": body})
        return {"choices": [{"message": {"content": self.reply(sysm, user)},
                             "finish_reason": "stop"}], "usage": {}}


def _run(scripted: Scripted, url: str, *, package: str | None = "koota",
         fetch=None, validate: bool = True) -> dict:
    saved = (model.post, P.fetch_source, P.MODEL_SCREEN, P.MODEL_VALIDATE,
             P.MODEL_TAG, P.MODEL_TESTS)
    model.post = scripted.post
    P.fetch_source = fetch or skill_offline.fixture_fetcher(PMNDRS)
    P.MODEL_SCREEN = P.MODEL_VALIDATE = P.MODEL_TAG = P.MODEL_TESTS = True
    P.MODEL_VALIDATE = validate
    try:
        s = skills.create(url=url, name="pmndrs-koota-skill",
                          goal="atomic skills for koota 0.6.6", frontier=True,
                          meta={"package": package} if package else {},
                          enqueue_first=False, watch_hours=24)
        _STATE.setdefault("stages", {})[s["id"]] = P.run_inline(s["id"], 1)
        return skills.get(s["id"])
    finally:
        (model.post, P.fetch_source, P.MODEL_SCREEN, P.MODEL_VALIDATE,
         P.MODEL_TAG, P.MODEL_TESTS) = saved


def _spec() -> dict:
    return skill_offline.load_spec(SPEC)


def _koota_reply() -> str:
    with open(os.path.join(ROOT, "skills", "ingested", "pmndrs",
                           "koota.decompose.txt"), encoding="utf-8") as f:
        return f.read()


# The pipeline's own filing of koota-react-integration under tag/5: koota,
# with React (the operator's rule edit of 2026-09-26, now the tag stage's).
REACT_TAG = {"koota-react-integration": {
    "artifacts": ["code"], "languages": [], "frameworks": ["koota"],
    "all_of": ["react"], "phases": ["implement", "debug"], "situations": [],
    "topics": ["useTrait", "useQuery", "WorldProvider"], "description": ""}}
REACT_TESTS = {"koota-react-integration": {
    "should": [
        "The HUD uses useTrait from koota/react and does not rerender when "
        "the score changes.\n\n```tsx\nimport { useTrait } from "
        "'koota/react'\nimport { createRoot } from 'react-dom/client'\n```",
        "Wire the koota world into the React app and read the score with "
        "useTrait.\n\n```tsx\nimport { useTrait } from 'koota/react'\n"
        "import React from 'react'\n```"],
    "should_not": [
        "Why does this refetch?\n\n```tsx\nimport { useQuery } from "
        "'@tanstack/react-query'\nconst q = useQuery({ queryKey: ['todos'] "
        "})\n```",
        "Use koota's useTrait in a Node CLI with no UI.\n\n```ts\nimport "
        "{ createWorld } from 'koota'\n```"],
    "behaviour": []}}
_STATE: dict = {}


# ================================================================ 1 cap ===
def test_every_stage_runs_as_a_capped_helper_job():
    cap = tiers.HELPER_THINKING
    for purpose in ("screen", "distil", "decompose", "tag", "tests",
                    "faithful"):
        job = P.job_name(purpose)
        b = P.shaped_body("s", "u", max_tokens=1024, purpose=purpose)
        want = min(tiers.JOB_THINKING.get(job) or cap, cap) \
            if job in tiers.JOB_THINKING else cap
        check(b.get("reasoning_budget_tokens") == want
              and b.get("reasoning_effort") == "medium"
              and "_share" not in b,
              f"[cap] {purpose}: reasoning_budget_tokens = {want} "
              "(the helper cap), effort medium, accounted as internal",
              {k: b.get(k) for k in ("reasoning_budget_tokens",
                                     "reasoning_effort", "max_tokens")})
    # Through the real pipeline: every body the model server is sent.
    sc = Scripted(LEAD + _koota_reply(), _spec(), REACT_TAG,
                  extra_tests=REACT_TESTS)
    parent = _run(sc, KOOTA)
    _STATE["parent"], _STATE["scripted"] = parent, sc
    jobs_seen = sorted({x["job"] for x in sc.bodies})
    check({"skill.screen", "skill.decompose", "skill.tag", "skill.tests",
           "skill.faithful"} <= set(jobs_seen),
          "[cap] screen_model, decompose, tag, tests and faithful each ran "
          "as a named job on the thread", jobs_seen)
    bad = [(x["job"], x["body"].get("reasoning_budget_tokens"))
           for x in sc.bodies
           if x["body"].get("reasoning_budget_tokens") != cap]
    check(sc.bodies and not bad, f"[cap] all {len(sc.bodies)} model calls "
          f"carried reasoning_budget_tokens = {cap}", bad[:4])
    check(getattr(P._CURRENT, "job", None) is None,
          "[cap] the job name is cleared from the thread after the call")


# =============================================================== 2 lead ===
def _served(sid: str) -> dict:
    s = skills.get(sid) or {}
    return skills.version(sid, s.get("served_version")
                          or s.get("latest_version")) or {}


def test_the_lead_skill():
    import skill_md
    parent = _STATE.get("parent")
    if not check(parent is not None, "[lead] the koota source ran"):
        return
    kids = skills.children(parent["id"])
    by = {k["name"]: k for k in kids}
    lead = by.get("koota-ecs-overview")
    check(lead and lead["status"] == "armed", "[lead] the lead skill arms",
          {k["name"]: (k["status"], k.get("reason")) for k in kids})
    if not lead:
        return
    sk = skill_md.parse(_served(lead["id"])["text"])
    check((sk.get("yamadori") or {}).get("lead_for") == "koota",
          "[lead] its SKILL.md says metadata.yamadori.lead_for: koota",
          sk.get("yamadori"))
    row = next((r for r in skills.armed() if r["id"] == lead["id"]), {})
    check(row.get("lead_for") == "koota",
          "[lead] the served row carries lead_for (skill_packages reads it)",
          row.get("lead_for"))
    others = [r for r in skills.armed() if r["id"] != lead["id"]
              and r.get("lead_for")]
    check(not others, "[lead] no other child is a lead",
          [r["name"] for r in others])
    val = _served(lead["id"]).get("validate") or {}
    check(val.get("faithful", {}).get("prompt") == skill_prompts.
          FAITHFUL_VERSION and len(val.get("items") or []) == 3,
          "[lead] its items were quote-checked and faithful-checked",
          (val.get("faithful"), len(val.get("items") or [])))
    rec = (skills.version(parent["id"], 1).get("distil") or {}).get("lead")
    check(rec and rec.get("package") == "koota"
          and rec.get("declared") == "koota",
          "[lead] the source's version records the lead decision", rec)
    # Unit: a lead in part 2, and a lead naming what the source never names.
    p1 = [{"name": "a", "lead": "koota", "items": [1]}]
    p2 = [{"name": "b", "lead": "koota", "items": [1]}]
    r = P._choose_lead([p1, p2], "Koota manages state.", "")
    check(p1[0]["lead"] == "koota" and p2[0]["lead"] == ""
          and r.get("cleared"), "[lead] only the first part's lead is the "
          "lead; a later part's mark is cleared", r)
    p1 = [{"name": "a", "lead": "zustand", "items": [1]}]
    r = P._choose_lead([p1], "Koota manages state.", "")
    check(p1[0]["lead"] == "" and r.get("package") is None and r.get("why"),
          "[lead] a lead naming a package the source never names is an "
          "ordinary skill", r)
    check(not P._names_package("mathematics is fun", "math")
          and P._names_package("use the npm `math` package", "math")
          and P._names_package("import from @react-three/fiber",
                               "@react-three/fiber"),
          "[lead] the package must be named as a word of its own")


# ============================================================= 3 screen ===
def test_the_screen_strips_typography_and_quarantines_smuggling():
    def screen(name):
        with open(os.path.join(FIX, "screen", name), encoding="utf-8") as f:
            raw = f.read()
        return raw, skill_screen.screen(raw, raw, "markdown")
    raw, r = screen("three_llms_full_line457.md")
    check("\u200b\u200b" in raw and r["ok"]
          and any(n.get("stripped") == 2 for n in r["notes"]),
          "[screen] three.js r185 llms-full.txt line 457 passes, its two "
          "U+200B stripped and recorded", skill_screen.summary(r)
          or r["notes"])
    t, f = skill_screen.strip_typography(raw)
    check("\u200b" not in t and f and f["line"] == 9,
          "[screen] the stripped text has none left; the note names the line",
          f)
    raw, r = screen("tag_injection_with_zero_width.md")
    check(not r["ok"] and any("Unicode tag" in q["what"]
                              for q in r["quarantine"]),
          "[screen] Unicode tag characters beside zero-width ones still "
          "QUARANTINE", skill_screen.summary(r))
    raw, r = screen("bidi_with_zero_width.md")
    check(not r["ok"] and any("bidi" in q["what"] for q in r["quarantine"]),
          "[screen] bidi overrides and isolates still QUARANTINE",
          skill_screen.summary(r))
    for name, rule in (("unicode_tags.md", "Unicode tag"),
                       ("bidi.md", "bidi")):
        with open(os.path.join(FIX, "malicious", name),
                  encoding="utf-8") as fh:
            m = fh.read()
        r = skill_screen.screen(m, m)
        check(not r["ok"] and any(rule in q["what"] for q in r["quarantine"]),
              f"[screen] the malicious fixture {name} still quarantines")
    check(skill_screen.strip_typography("\ufeffabc") == ("abc", None),
          "[screen] a leading byte-order mark is removed and not recorded")
    # What the model reads is the stripped text (source_texts).
    zw = "Keep\u200bstate\u200blocal when you can, and lift it up only then."
    s = skills.create(text=zw, name="zw", enqueue_first=False)
    _raw, text, _k = P.source_texts(s["id"], 1)
    check("\u200b" not in text and "Keepstatelocal" in text,
          "[screen] the text the model reads and quotes are checked against "
          "has the typography stripped", text)


# ============================================================ 4 all_of ===
def test_multi_domain_filing():
    blob = "Use koota inside React with useTrait from koota/react."
    all_of, a, rec = C.verified_all_of(
        ["react"], {"artifacts": ["code"], "languages": [],
                    "frameworks": ["koota", "react"]}, blob)
    check(all_of == ["react"] and a["frameworks"] == ["koota"],
          "[all_of] a named term is kept and leaves the OR list "
          "(koota or React -> koota, with React)", (all_of, a, rec))
    all_of, a, rec = C.verified_all_of(
        ["threejs", "nosuch"], {"artifacts": ["code"], "languages": [],
                                "frameworks": ["koota"]}, blob)
    check(all_of == [] and {d["term"] for d in rec["dropped"]}
          == {"threejs", "nosuch"},
          "[all_of] a term the text does not name, or outside the "
          "vocabulary, is dropped with the reason", rec)
    all_of, a, rec = C.verified_all_of(
        ["react"], {"artifacts": ["code"], "languages": [],
                    "frameworks": ["react"]}, blob)
    check(all_of == [] and a["frameworks"] == ["react"],
          "[all_of] refused when it would leave nothing to key the skill "
          "on", rec)
    check("all_of" in skill_prompts.tag_system()
          and skill_prompts.TAG_VERSION == "tag/6",
          "[all_of] the tag template (tag/5 on) asks for it")
    parent = _STATE.get("parent")
    if not parent:
        return
    kid = next((k for k in skills.children(parent["id"])
                if k["name"] == "koota-react-integration"), None)
    v = _served(kid["id"]) if kid else {}
    rule = v.get("classify") or {}
    check(kid and kid["status"] == "armed"
          and C.gates(rule)["all_of"] == ["react"]
          and C.applies_to(rule)["frameworks"] == ["koota"]
          and "with React" in (rule.get("text") or ""),
          "[all_of] koota-react-integration ARMS on the pipeline's own "
          "filing (koota, with React) with no operator rule edit",
          (kid and kid.get("reason"), rule.get("text")))
    act = (v.get("validate") or {}).get("activation") or {}
    check(act.get("passed"), "[all_of] its near misses (React Query's "
          "useQuery, koota with no UI) are not selected",
          act.get("failures"))


# ============================================================== 5 watch ===
def test_a_changed_frontier_source_is_decomposed_again():
    parent = _STATE.get("parent")
    if not parent:
        check(False, "[watch] the koota source ran")
        return
    before = {k["name"]: k for k in skills.children(parent["id"])}
    fetch0 = skill_offline.fixture_fetcher(PMNDRS)

    def fetch(url, beat=None):
        raw, meta = fetch0(url, beat)
        if url.endswith("/SKILL.md"):
            raw += b"\n\n## Changelog\n\nA new section.\n"
        return raw, meta
    # v2's decomposition: the lead and two children, React dropped.
    blocks = _koota_reply().split("=== SKILL ===")
    reply = LEAD + "=== SKILL ===".join(
        b for b in blocks if "koota-react-integration" not in b)
    sc = Scripted(reply, _spec(), REACT_TAG, extra_tests=REACT_TESTS)
    saved = (model.post, P.fetch_source, P.MODEL_SCREEN, P.MODEL_VALIDATE,
             P.MODEL_TAG, P.MODEL_TESTS)
    model.post, P.fetch_source = sc.post, fetch
    P.MODEL_SCREEN = P.MODEL_VALIDATE = P.MODEL_TAG = P.MODEL_TESTS = True
    try:
        got = P.handle_watch({"payload": {"skill": parent["id"]}},
                             P._InlineCtx())
        check(got.get("changed") and got.get("version") == 2,
              "[watch] the changed source is version 2", got)
        ver = skills.version(parent["id"], 2) or {}
        check(ver.get("origin") == "watch_frontier"
              and "decompose" in (ver.get("path") or [])
              and "distil" not in (ver.get("path") or []),
              "[watch] a frontier source's new version walks decompose, "
              "not distil", ver.get("path"))
        out = P.run_inline(parent["id"], 2)
    finally:
        (model.post, P.fetch_source, P.MODEL_SCREEN, P.MODEL_VALIDATE,
         P.MODEL_TAG, P.MODEL_TESTS) = saved
    dec = out.get("decompose") or {}
    after = {k["name"]: k for k in skills.children(parent["id"])}
    kept = ("koota-ecs-overview", "koota-traits-and-entities",
            "koota-queries-and-systems")
    check(all(after.get(n) and after[n]["id"] == before[n]["id"]
              and after[n]["served_version"] == 2
              and after[n]["status"] == "armed" for n in kept),
          "[watch] each re-proposed child (the lead included) is a NEW "
          "VERSION of itself, armed, superseding version 1",
          {n: (after.get(n) or {}).get("served_version") for n in kept})
    old = [skills.version(before[n]["id"], 1)["state"] for n in kept]
    check(old == ["superseded"] * 3, "[watch] their version 1 is superseded",
          old)
    gone = after.get("koota-react-integration") or {}
    check(gone.get("status") == "archived"
          and gone["id"] in (dec.get("archived") or []),
          "[watch] the child whose section is gone is ARCHIVED, not deleted",
          (gone.get("status"), gone.get("reason")))
    check(len(after) == len(before),
          "[watch] no duplicate children", sorted(after))
    lead = after.get("koota-ecs-overview")
    row = next((r for r in skills.armed() if lead and r["id"] == lead["id"]),
               {})
    check(row.get("lead_for") == "koota", "[watch] the lead is still the "
          "lead after the re-decomposition", row.get("lead_for"))
    par = (skills.get(lead["id"]).get("meta") or {}).get("parent") or {}
    check(par.get("version") == 2, "[watch] a re-versioned child's quotes "
          "are checked against the new source (meta.parent repointed)", par)


# ======================================================== 6 faithful ===
def _one_block() -> str:
    blocks = _koota_reply().split("=== SKILL ===")
    return "=== SKILL ===" + next(b for b in blocks
                                  if "koota-traits-and-entities" in b)


def _kid(parent: dict) -> dict:
    return skills.children(parent["id"])[0]


def test_faithfulness_is_mandatory():
    check("lead:" in skill_prompts.DECOMPOSE_SYSTEM
          and "compact" in skill_prompts.DECOMPOSE_SYSTEM
          and skill_prompts.DECOMPOSE_VERSION == "decompose/5",
          "[lead] decompose/3 asks for ONE compact lead skill first")
    check('"Not for"' in skill_prompts.DISTIL_SYSTEM
          and '"Not for"' in skill_prompts.DECOMPOSE_SYSTEM
          and skill_prompts.DISTIL_VERSION == "distil/6",
          "[boundary] distil/4 and decompose/3 ask for a \"Not for\" "
          "boundary sentence")
    # Switched off: nothing a model wrote arms.
    parent = _run(Scripted(_one_block(), _spec()), KOOTA, validate=False)
    k = _kid(parent)
    v = skills.version(k["id"], 1) or {}
    check(k["status"] == "failed" and "faithfulness check is required"
          in (k.get("reason") or "") and (v.get("validate") or {})
          .get("faithful", {}).get("skipped"),
          "[faithful] switched off (YAMADORI_SKILL_MODEL_VALIDATE=0), a "
          "model-written skill FAILS with the remedy, never arms",
          (k["status"], k.get("reason")))
    # The check errors: retryable, the child stays at validate, unarmed.
    parent = _run(Scripted(_one_block(), _spec(), faithful="error"), KOOTA)
    k = _kid(parent)
    v = skills.version(k["id"], 1) or {}
    dec = (_STATE["stages"].get(parent["id"]) or {}).get("decompose") or {}
    check(k["status"] == "pipeline" and v.get("state") == "running"
          and v.get("stage") == "validate" and dec.get("child_errors"),
          "[faithful] a check that could not run leaves the child RUNNING at "
          "validate (retryable), unarmed; the source still decomposes",
          (k["status"], v.get("stage"), dec.get("child_errors")))
    # Silence on an item is not a verdict.
    parent = _run(Scripted(_one_block(), _spec(), faithful="silent_first"),
                  KOOTA)
    k = _kid(parent)
    val = _served(k["id"]).get("validate") or {}
    f = val.get("faithful") or {}
    check(k["status"] == "armed" and f.get("no_verdict") == 1
          and any("no verdict" in d["why"] for d in val.get("dropped") or [])
          and len(val.get("items") or []) == 5,
          "[faithful] an item the check is silent on is DROPPED; the rest "
          "arm", (k["status"], f, len(val.get("items") or [])))
    check("ASI 2504.06821" in (f.get("judge") or ""),
          "[faithful] the caveat is recorded with every verdict: the judge "
          "is the model that wrote the items", f.get("judge"))
    # The offline path: no verdict in the spec, nothing arms.
    spec = _spec()
    spec["sources"] = [x for x in spec["sources"] if x["kind"] == "frontier"
                       and "koota" in x["url"]]
    spec.pop("faithful", None)
    got = skill_offline.ingest(spec, fetcher=skill_offline.fixture_fetcher(
        PMNDRS))
    kids = [x for r in got for x in r["skills"]]
    check(kids and not [x for x in kids if x["status"] == "armed"],
          "[faithful] skill_offline with no verdict in its spec arms nothing",
          [(x["name"], x["status"]) for x in kids])


# ======================================================= 7 boundaries ===
def test_boundaries_are_recorded():
    import skill_boundaries as B
    parent = _STATE.get("parent")
    if not check(parent is not None, "[boundary] the koota source ran"):
        return
    by = {k["name"]: k for k in skills.children(parent["id"])}
    lead = _served(by["koota-ecs-overview"]["id"])
    val = lead.get("validate") or {}
    check((val.get("boundary") or {}).get("stated")
          and val["boundary"]["text"].startswith("Not for"),
          "[boundary] validate records the description's \"Not for\" "
          "boundary", val.get("boundary"))
    trig = [t.get("text") for t in (lead.get("classify") or {})
            .get("triggers") or [] if t.get("origin") == "description"]
    check(trig and "Not for" not in trig[0] and "React" not in trig[0],
          "[boundary] the boundary sentence stays out of the trigger text "
          "(its words would pull the embedding toward what it excludes)",
          trig)
    react = _served(by["koota-react-integration"]["id"]).get("validate") or {}
    sib = react.get("siblings") or {}
    check(sib.get("siblings", 0) >= 1 and sib.get("cases", 0) >= 1
          and "fired" in sib,
          "[boundary] validate runs the siblings' own should-cases against "
          "the new skill and records which select it too (not gating)",
          {k: sib.get(k) for k in ("siblings", "cases", "fires")})
    check(B.boundary_of("Use when math is used, not for routine arithmetic.")
          ["stated"] is False and C.strip_boundary(
              "Use when X, not for Y.") == "Use when X, not for Y.",
          "[boundary] only a sentence that OPENS with Not for / Not when is "
          "a boundary")
    # Black-hole candidates: a catch-all skill captures other areas' cases.
    generic = C.rule_from_metadata({"artifacts": ["code"]})
    rust = C.rule_from_metadata({"languages": ["rust"]})
    py = C.rule_from_metadata({"languages": ["python"]})
    pool = [{"id": "g", "name": "catch-all", "rule": generic, "served": 1},
            {"id": "r", "name": "rust-one", "rule": rust, "served": 1},
            {"id": "p", "name": "python-one", "rule": py, "served": 1}]
    cases = {"r": [{"text": "Fix the borrow error.\n\n```rust\nfn main() "
                    "{ let v = vec![1]; }\n```"}],
             "p": [{"text": "Fix the import.\n\n```python\nimport os\n"
                    "print(os.getcwd())\n```"}]}
    saved = B._should_cases
    B._should_cases = lambda sid, served: cases.get(sid, [])
    try:
        rows = B.black_holes(pool)
    finally:
        B._should_cases = saved
    top = rows[0] if rows else {}
    check(top.get("name") == "catch-all" and len(top.get("areas") or []) == 2,
          "[boundary] a catch-all skill is listed first as a black-hole "
          "candidate (cases from 2 other areas); listing only, never retired",
          rows)


# ============================================== 8 the koota-run fixes ===
def test_the_koota_run_fixes():
    import skill_builder
    # 1. The tests stage reads the skill's gates.
    sc = _STATE.get("scripted")
    tests_users = [m["content"] for x in (sc.bodies if sc else [])
                   for m in x["body"]["messages"] if m["role"] == "user"
                   and x["job"] == "skill.tests"]
    check(tests_users and all("<gates>" in u for u in tests_users)
          and any("topics (name at least one exactly as written)" in u
                  for u in tests_users)
          and skill_prompts.TESTS_VERSION == "tests/3",
          "[tests/3] every tests call carries the skill's own gates",
          tests_users[:1])
    rule = C.with_gates(C.rule_from_metadata({"frameworks": ["koota"]}),
                        phases=["plan"], topics=["updateEach"])
    tests = {"activation": {"should": [
        {"text": "Add a system that moves bullets with updateEach.\n\n```ts\n"
                 "import { createWorld } from 'koota'\n```"}]}}
    r2, gone = P._reconcile_gates(rule, tests)
    check(gone and gone[0]["gate"] == "phases" and not C.gates(r2)["phases"]
          and C.gates(r2)["topics"] == ["updateEach"],
          "[tests/3] a phase gate the skill's own should-case contradicts is "
          "DROPPED and recorded; the other gates stay", gone)
    r3, gone3 = P._reconcile_gates(C.with_gates(rule, phases=["implement"],
                                                topics=["updateEach"]), tests)
    check(not gone3 and C.gates(r3)["phases"] == ["implement"],
          "[tests/3] a phase gate the should-cases satisfy stays")
    # 2. Topics are code-shaped and in the source.
    dropped: list = []
    got = P._verified_topics(
        ["world.spawn", "query", "useQuery", "updateEach"],
        "world.spawn query useQuery updateEach",
        source="world.spawn(Position) ... query ... updateEach(",
        dropped=dropped)
    check(got == ["world.spawn", "updateEach"]
          and {d["topic"]: d["why"][:14] for d in dropped}
          == {"query": "not code-shape", "useQuery": "the source doe"},
          "[topics] a plain word and a symbol the source never names are "
          "dropped, each with the reason", (got, dropped))
    # 3. One quote-repair round.
    src = ("# Koota\n\nKoota manages state using entities with composable "
           "traits.\n\n## Queries\n\nPrefer updateEach over for...of with "
           "entity.get() for data-bearing queries.\n")
    parsed = {"items": [
        {"form": "DO", "text": "batch-update with updateEach",
         "quote": "updateEach"},
        {"form": "DO", "text": "use traits", "quote": "Koota manages state "
         "using entities with composable traits."},
        {"form": "DO", "text": "invent", "quote": "never in the source at "
         "all, not one word"}]}
    saved = P.ask_model
    seen = {}

    def fake(system, user, *, max_tokens, purpose=""):
        seen["user"] = user
        return json.dumps({"items": [
            {"i": 1, "quote": "Prefer updateEach over for...of with "
             "entity.get() for data-bearing queries."},
            {"i": 2, "quote": "a quote that is still not there at all"}]})
    P.ask_model = fake
    try:
        out, rec = P._repair_quotes(parsed, src)
    finally:
        P.ask_model = saved
    check(rec and rec["asked"] == 2 and rec["repaired"] == 1
          and out["items"][0]["quote"].startswith("Prefer updateEach")
          and out["items"][2]["quote"].startswith("never in the source")
          and 'failing quote: "updateEach"' in seen.get("user", "")
          and "## Queries" in seen.get("user", ""),
          "[quotes] one repair round: the exact failing quote and the source "
          "section go back; a verified new quote replaces it, an unverified "
          "one leaves the item to be dropped", rec)
    res = skill_builder.validate(out, source=src)
    check(len(res["items"]) == 2 and len(res["dropped"]) == 1,
          "[quotes] after the repair, validate keeps the repaired item",
          res["dropped"])
    big = ("# A\n\n" + "alpha words here. " * 900 + "\n\n# Queries\n\n"
           "updateEach lives here.\n\n# C\n\n" + "gamma words here. " * 900)
    sec = P._section_of(big, "Design Principles / Queries", ["updateEach"])
    check(sec.startswith("# Queries") and "alpha" not in sec,
          "[quotes] a large source's repair reads the section the child "
          "came from", sec[:60])
    # 4. The lead is tested by the detector that pushes it.
    parent = _STATE.get("parent")
    lead = next((k for k in skills.children(parent["id"])
                 if k["name"] == "koota-ecs-overview"), None) if parent \
        else None
    val = (_served(lead["id"]).get("validate") or {}) if lead else {}
    act = val.get("activation") or {}
    check(lead and lead["status"] == "armed"
          and "skill_packages.detect" in (act.get("by") or ""),
          "[lead] its activation is tested by skill_packages' detection, "
          "not the matcher", (lead and lead["status"], act.get("failures")))
    bad = P._lead_activation({"activation": {
        "should": [{"text": "Set up state with koota."},
                   {"text": "Spawn koota entities."}],
        "should_not": [{"text": "Use zustand for the store."},
                       {"text": "Why is koota slow?"}]}}, "koota")
    check(not bad["passed"] and any("should_not" in f
                                    for f in bad["failures"]),
          "[lead] a near miss that puts the package in play fails it",
          bad["failures"])


def main() -> int:
    for fn in (test_every_stage_runs_as_a_capped_helper_job,
               test_the_lead_skill,
               test_the_screen_strips_typography_and_quarantines_smuggling,
               test_multi_domain_filing,
               test_a_changed_frontier_source_is_decomposed_again,
               test_faithfulness_is_mandatory,
               test_boundaries_are_recorded,
               test_the_koota_run_fixes):
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
    passed = sum(1 for ok, _, _ in _results if ok)
    print(f"\n{'=' * 70}\n  {passed}/{len(_results)} checks passed")
    return 0 if passed == len(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
