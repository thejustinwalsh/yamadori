#!/usr/bin/env python
"""The PROVE stage (mcp/skill_prove.py): a paired A/B per skill, that it
does not make a task worse. No GPU, no network, no live model.

    python mcp/test_skill_prove.py      -> "N/M checks passed"

Operator, 2026-09-28: "each skill needs proof that it works ... a mini A/B
test to ensure it doesn't make a task worse, generated in the skill
pipeline." What this gates:

  1. PROBE DERIVATION: the deterministic floor from the skill's own
     should-cases and items, with no model; the model's proposals verified
     by code (a bad regex, a pattern the skill does not name and a probe
     that quotes an item are refused; a good task and check are kept).
  2. CHECKS by code: parse (a syntax error fails), a DO's pattern present,
     a DO NOT's pattern absent; types "not run" without a checker, never a
     pass; an injected checker decides.
  3. THE PAIR: the same seed and the same sampling and budget fields on
     both sides; WITHOUT's user turn is the probe, WITH's is the probe plus
     exactly the selector's craft block (skill_select._render_turn).
  4. THE DECISION: worse -> quarantined with the evidence; tie -> armed,
     "no measurable gain"; better -> armed; a budget event -> undecided,
     retried ONCE with the job's full answer room and a second seed, and
     still undecided -> UNPROVEN, quarantined, never armed (operator,
     2026-09-28).
  5. THE LAST RESORT: no code check can decide -> one model judge check,
     labelled judge: model.
  6. INLINE: a migration / authored install records "not run: inline
     install" and arms.
  7. THE BACKLOG: newest-updated first, a skill proved at its served
     version skipped, a live job skipped, jobs idle-gated on the gpu lane;
     a backlog proof that finds WORSE disarms the skill.
  9. THE REPEAT RULE (operator, 2026-10-06): a flagged check (passed
     without, failed with) is run again on fresh seeds, the same on both
     sides, and quarantines only if worse in a strict majority of its runs;
     only a probe with a flag is repeated.
 10. THE RE-PROVE: a skill PROVE quarantined under the one-sample rule is
     listed, enqueued idle-gated on the gpu lane, and either served again
     (the worse result did not repeat) or kept quarantined (it did); the
     activation-test quarantines are not touched.

Every store is a temp path set BEFORE any mcp module is imported
(mcp/offline_stores.py).
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import offline_stores  # noqa: E402

_TMP = offline_stores.isolate("yamadori_test_skill_prove_")
os.environ["YAMADORI_SKILL_CACHE_SECONDS"] = "0"
os.environ.setdefault("YAMADORI_GPU_ROOM", "0")

import jobs  # noqa: E402
import skill_classify as C  # noqa: E402
import skill_md  # noqa: E402
import skill_pipeline  # noqa: E402
import skill_prompts  # noqa: E402
import skill_prove as P  # noqa: E402
import skill_select  # noqa: E402
import skills  # noqa: E402
import budget  # noqa: E402
import tiers  # noqa: E402

# The pool size is asked of the model server (budget.pool_size); an offline
# suite states it (config.yaml's -c 181248, as AGENTS.md "One door" says),
# as mcp/test_skill_pipeline_fixes.py does.
budget._POOL = 181248
# The efforts the served template accepts (tiers.accepted_efforts reads
# /props): the set AGENTS.md "Prompting this model" names.
tiers._accepted = ("low", "medium", "xhigh")

RESULTS: list[tuple[bool, str, object]] = []


def check(ok: bool, what: str, detail=None) -> None:
    RESULTS.append((bool(ok), what, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {what}"
          + ("" if ok or detail is None else f"\n        {detail!r}"[:600]),
          flush=True)


def reset_store() -> None:
    con = skills._db()
    try:
        con.execute("DELETE FROM skills")
        con.execute("DELETE FROM skill_versions")
        con.execute("DELETE FROM jobs")
    finally:
        con.close()
    skills._invalidate()


# ---------------------------------------------------------------------------
# A skill, placed straight at a stage (no pipeline run needed).
# ---------------------------------------------------------------------------
ITEMS = [
    {"form": "DO", "situation": "", "quote": "", "ref": "",
     "text": "iterate the matching entities with `updateEach` and write "
             "each store in place."},
    {"form": "DO NOT", "situation": "", "quote": "", "ref": "",
     "text": "cast a trait's store with `as any`; type it with its schema."},
]
SHOULD = [
    {"text": "Write a TypeScript system that moves each entity by its "
             "velocity every frame.\n\n```ts\n// systems/move.ts\n```"},
    {"text": "Add a gravity step to the physics module.",
     "files": ["src/physics.ts"]},
    {"text": "The update loop is slow; rewrite it.\n\n```ts\n"
             "for (const e of world.query(Pos)) { }\n```"},
    {"text": "A fourth case the operator's 3-probe figure leaves out."
             "\n\n```ts\n// x\n```"},
]


def make_skill(name: str, *, items=None, should=None, rule=None,
               stage: str = "prove", arm: bool = False,
               updated: float | None = None) -> str:
    sid = f"{name[:8]}{abs(hash(name)) % 10**4:04d}"
    items = ITEMS if items is None else items
    rule = rule or C.rule_from_metadata({"languages": ["typescript"]})
    sk = {"name": name, "description": f"Use when {name} work is done.",
          "title": name.replace("-", " ").title(), "items": items,
          "yamadori": {"id": sid}}
    text = skill_md.render(sk)
    skills._insert(sid, name, "authored", "authored",
                   skills.PATHS["authored"], url=None, author="test",
                   meta={}, watch=None)
    skills.update_version(
        sid, 1, stage=stage, text=text, text_sha256=skill_md.sha256(text),
        classify=rule,
        validate={"ok": True, "title": sk["title"], "items": items},
        tests={"activation": {"should": SHOULD if should is None
                              else should, "should_not": []},
               "behaviour": []})
    if arm:
        skills.arm(sid, 1)
    if updated is not None:
        con = skills._db()
        try:
            con.execute("UPDATE skill_versions SET updated=? WHERE skill=?",
                        (updated, sid))
        finally:
            con.close()
    skills._invalidate()
    return sid


# ---------------------------------------------------------------------------
# Fakes: the model server (CHAT) and the pipeline's ask (ASK).
# ---------------------------------------------------------------------------
GOOD_TS = ("```ts\nexport function move(world: World) {\n"
           "  world.query(Pos, Vel).updateEach(([p, v]) => {\n"
           "    p.x += v.x;\n  });\n}\n```\nDone.")
CAST_TS = ("```ts\nexport function move(world: World) {\n"
           "  const s = world.store as any;\n  s.x += 1;\n}\n```")
BROKEN_TS = "```ts\nexport function move(world: World {\n  let = ;\n```"


class FakeChat:
    def __init__(self, without: str, with_: str, finish=("stop", "stop")):
        self.without, self.with_ = without, with_
        self.finish = finish
        self.bodies: list[dict] = []

    def __call__(self, body: dict) -> dict:
        self.bodies.append(json.loads(json.dumps(body)))
        user = body["messages"][-1]["content"]
        on = skill_prompts.CRAFT_HEADER in user
        return {"choices": [{"message": {
            "content": self.with_ if on else self.without},
            "finish_reason": self.finish[1] if on else self.finish[0]}],
            "usage": {"completion_tokens": 10}}


def no_model(system, user, *, max_tokens, purpose=""):
    raise RuntimeError("no model in this test")


class Ctx:
    def beat(self, progress=None):
        pass


def job_for(sid: str, v: int = 1, **extra) -> dict:
    return {"id": "t", "queue": P.JOB[0],
            "payload": dict({"skill": sid, "version": v, "stage": "prove"},
                            **extra)}


def with_fakes(chat, ask=no_model):
    class _W:
        def __enter__(self):
            self.saved = (P.CHAT, P.ASK, P.TYPE_CHECKER)
            P.CHAT, P.ASK = chat, ask
            return chat

        def __exit__(self, *a):
            P.CHAT, P.ASK, P.TYPE_CHECKER = self.saved
    return _W()


# ===========================================================================
# 1. probe derivation
# ===========================================================================
def test_floor_is_deterministic_and_needs_no_model():
    reset_store()
    sid = make_skill("koota-update-each")
    ver = skills.version(sid, 1)
    with with_fakes(None):
        d1 = P.derive(sid, ver)
        d2 = P.derive(sid, ver)
    check(d1 == d2, "[derive] the floor is deterministic (two runs, same "
          "probes and checks)")
    probes = d1["probes"]
    check(len(probes) == P.PROBES and [p["case"] for p in probes] == [1, 2, 3],
          "[derive] up to the operator's 3 probes, from the skill's own "
          "should-cases, in order", [p["case"] for p in probes])
    check(any(x.get("case") == 4 and "past the operator" in x["why"]
              for x in d1["dropped"]),
          "[derive] the 4th should-case is recorded as past the cap",
          d1["dropped"])
    check(all(p["language"] == "typescript" for p in probes),
          "[derive] each probe's language: its fence, its file, or the "
          "skill's language", [p["language"] for p in probes])
    kinds = {c["kind"] for c in probes[0]["checks"]}
    check({"parse", "types", "present", "absent"} <= kinds,
          "[derive] parse, types, a DO's present check and a DO NOT's "
          "absent check", sorted(kinds))
    absent = [c for c in probes[0]["checks"] if c["kind"] == "absent"]
    check(absent and absent[0]["literal"] == "as any"
          and re.search(absent[0]["pattern"], "x as  any")
          and not re.search(absent[0]["pattern"], "has anything"),
          "[derive] the DO NOT's `as any` becomes a bounded absent pattern",
          absent)
    check(d1["model"].get("error", "").startswith("RuntimeError"),
          "[derive] the model half failing leaves the floor in place, the "
          "error recorded", d1["model"])


def test_a_leaking_should_case_and_an_echoed_check_are_dropped():
    reset_store()
    leak = {"text": "Please iterate the matching entities with updateEach and "
                    "write each store in place, in TypeScript."}
    named = {"text": "Use updateEach to move entities.\n\n```ts\n//\n```"}
    sid = make_skill("leaky", should=[leak, named, SHOULD[0]])
    with with_fakes(None):
        d = P.derive(sid, skills.version(sid, 1), use_model=False)
    check(any(x.get("case") == 1 and "quotes item 1" in x["why"]
              for x in d["dropped"]) and all(p["case"] != 1
                                             for p in d["probes"]),
          "[derive] a should-case sharing an 8-word run with an item is "
          "skipped (it leaks the answer)", d["dropped"])
    p_named = next(p for p in d["probes"] if p["case"] == 2)
    check(not any(c["kind"] == "present" for c in p_named["checks"])
          and any("measure echo" in x.get("why", "") for x in d["dropped"]),
          "[derive] a present check whose literal the probe names is "
          "dropped (echo, not the skill)", p_named["checks"])


def test_model_proposals_are_verified_by_code():
    reset_store()
    sid = make_skill("koota-proposals")
    ver = skills.version(sid, 1)
    seen = {}

    def ask(system, user, *, max_tokens, purpose=""):
        seen.update(system=system, user=user, purpose=purpose)
        return json.dumps({"probes": [
            {"case": 1, "task": "Write a TypeScript function that advances "
                                "every moving body by its velocity.",
             "language": "typescript", "checks": [
                 {"kind": "present", "pattern": r"\bupdateEach\b",
                  "item": 1},
                 {"kind": "absent", "pattern": "([", "item": 2},
                 {"kind": "present", "pattern": "fooBarBaz", "item": 1},
                 {"kind": "present", "pattern": ".*", "item": 1}]},
            {"case": 2, "task": "iterate the matching entities with "
                                "updateEach and write each store in place",
             "language": "typescript", "checks": []},
            {"case": 3, "task": "Write it in Python instead.",
             "language": "python", "checks": []},
            {"case": 9, "task": "x", "checks": []}]})
    with with_fakes(None, ask):
        d = P.derive(sid, ver, use_model=True)
    check(seen.get("system") == P.PROBE_SYSTEM and seen.get("purpose") ==
          "prove" and "<skill>" in seen.get("user", ""),
          "[model] the probe template goes through the pipeline's ask as "
          "job skill.prove, the skill inside its data block")
    ref = d["model"]["refused"]
    why = " | ".join(r["why"] for r in ref)
    check("does not compile" in why and "do not contain it" in why
          and "matches the empty text" in why,
          "[model] a bad regex, a pattern the skill does not name and a "
          "match-everything pattern are refused", ref)
    check("leaks the answer" in why and "not the case's typescript" in why
          and "no such case" in why,
          "[model] a task quoting an item, one changing the case's language "
          "and an unknown case are refused", ref)
    p1 = d["probes"][0]
    check(p1["task"].startswith("Write a TypeScript function that advances")
          and p1["source"] == "should[1]+model"
          and any(c.get("from") == "model" and c["kind"] == "present"
                  for c in p1["checks"]),
          "[model] a verified task replaces its case's text and its "
          "verified present check is kept", p1)
    check(d["probes"][1]["task"] == P._case_task(SHOULD[1]),
          "[model] a refused task leaves the case's own text",
          d["probes"][1]["task"])


# ===========================================================================
# 2. checks by code
# ===========================================================================
def test_checks():
    parse = {"kind": "parse"}
    ok, why = P.run_check(parse, GOOD_TS, "typescript", [])
    check(ok is True, "[check] parse passes a well-formed TypeScript block",
          why)
    ok, why = P.run_check(parse, BROKEN_TS, "typescript", [])
    check(ok is False and "syntax error" in why,
          "[check] parse fails a syntax error, with the parser's message",
          why)
    ok, why = P.run_check(parse, "```python\ndef f(:\n  pass\n```",
                          "python", [])
    check(ok is False, "[check] parse fails broken Python (the reference "
          "parser)", why)
    ok, why = P.run_check(parse, "I would use a loop.", "typescript", [])
    check(ok is False and "no fenced code block" in why,
          "[check] no code block is a parse failure", why)
    pres = {"kind": "present", "pattern": P.literal_pattern("updateEach")}
    absent = {"kind": "absent", "pattern": P.literal_pattern("as any")}
    check(P.run_check(pres, GOOD_TS, "typescript", [])[0] is True
          and P.run_check(pres, CAST_TS, "typescript", [])[0] is False,
          "[check] a DO's pattern: present in the code passes, missing fails")
    check(P.run_check(absent, GOOD_TS, "typescript", [])[0] is True
          and P.run_check(absent, CAST_TS, "typescript", [])[0] is False,
          "[check] a DO NOT's pattern: absent passes, present fails")
    prose = GOOD_TS + "\nI avoided `as any` on purpose."
    check(P.run_check(absent, prose, "typescript", [])[0] is True,
          "[check] a pattern is looked for in the answer's code, not its "
          "prose")
    saved = P.TYPE_CHECKER
    try:
        P.TYPE_CHECKER = None
        ok, why = P.run_check({"kind": "types"}, GOOD_TS, "typescript", [])
        check(ok is None and why.startswith("not run"),
              "[check] no type checker: types is 'not run', never a pass",
              why)
        calls = []
        P.TYPE_CHECKER = lambda code, lang, pk: (calls.append((lang, pk))
                                                 or {"ran": True,
                                                     "ok": False,
                                                     "errors": ["TS2339"]})
        ok, why = P.run_check({"kind": "types"}, GOOD_TS, "typescript",
                              ["koota"])
        check(ok is False and "TS2339" in why and calls == [("typescript",
                                                              ["koota"])],
              "[check] an injected type checker decides, given the held "
              "packages", (ok, why, calls))
    finally:
        P.TYPE_CHECKER = saved


# ===========================================================================
# 3. the pair
# ===========================================================================
def test_pair_bodies_same_seed_and_selector_form():
    reset_store()
    sid = make_skill("koota-pair")
    fake = FakeChat(GOOD_TS, GOOD_TS)
    with with_fakes(fake):
        out = P.handle_prove(job_for(sid), Ctx())
    b = fake.bodies
    check(len(b) == 2 * P.PROBES, "[pair] two generations per probe",
          len(b))
    row = P.row_for(sid, skills.version(sid, 1))
    pairs = list(zip(b[0::2], b[1::2]))
    same = all(x["seed"] == y["seed"] and isinstance(x["seed"], int)
               for x, y in pairs)
    check(same, "[pair] the same seed on both sides of every pair",
          [(x.get("seed"), y.get("seed")) for x, y in pairs])
    keys = [k for k in P._SAME_FIELDS if k in pairs[0][0]]
    check(keys and all(all(x.get(k) == y.get(k) for k in P._SAME_FIELDS)
                       for x, y in pairs),
          "[pair] every sampling and budget field equal on both sides",
          keys)
    x, y = pairs[0]
    probe = x["messages"][-1]["content"]
    expect = probe + skill_select._render_turn([row], [])
    check(y["messages"][-1]["content"] == expect
          and skill_prompts.CRAFT_HEADER not in probe
          and x["messages"][:-1] == y["messages"][:-1],
          "[pair] WITHOUT is the probe; WITH is the probe plus exactly the "
          "selector's craft block (skill_select._render_turn); the system "
          "text is the same", y["messages"][-1]["content"][-300:])
    check(x["messages"][0]["content"] == P.ANSWER_SYSTEM
          and x.get("model") and "reasoning_budget_tokens" in x,
          "[pair] shaped through the one door (skill_pipeline.shaped_body: "
          "the answer template, a thinking budget)", sorted(x)[:20])
    rec = P.record_of(skills.version(sid, 1))
    check(rec.get("probes") and all(
        len(p["probe_sha256"]) == 64 and set(p["seconds"]) ==
        {"without", "with"} and p["seed"] for p in rec["probes"]),
          "[pair] the record keeps each probe's text hash, seed and the "
          "seconds of each side", rec.get("probes", [{}])[0])
    check(out.get("prove") == "tie", "[pair] identical answers: a tie", out)


# ===========================================================================
# 4. the decision
# ===========================================================================
def test_worse_quarantines_with_evidence():
    reset_store()
    sid = make_skill("koota-worse")
    with with_fakes(FakeChat(GOOD_TS, CAST_TS)):
        out = P.handle_prove(job_for(sid), Ctx())
    s = skills.get(sid)
    ver = skills.version(sid, 1)
    rec = P.record_of(ver)
    check(out.get("prove") == "worse" and out.get("quarantined")
          and s["status"] == "quarantined" and ver["state"] == "quarantined",
          "[decide] a check that passed without and failed with the skill "
          "QUARANTINES the version", (out, s["status"]))
    check("prove:" in (s["reason"] or "") and "as" in s["reason"]
          and "retryable: no" in s["reason"] and "remedy" in s["reason"],
          "[decide] the reason carries the evidence, retryable and the "
          "remedy", s["reason"])
    check(rec.get("verdict") == "worse" and rec["worse"]
          and all(w["kind"] in ("absent", "present", "parse", "types")
                  for w in rec["worse"]),
          "[decide] the record lists every worse check (probe, kind, why "
          "on each side)", rec.get("worse"))
    check("prove" in skills._JSON_V and ver.get("prove", {}).get("verdict")
          == "worse" or (ver.get("validate") or {}).get("prove"),
          "[decide] the record is stored on the version (the prove column, "
          "or validate.prove)")


def test_parse_worse_quarantines():
    reset_store()
    sid = make_skill("koota-parse-worse")
    with with_fakes(FakeChat(GOOD_TS, BROKEN_TS)):
        out = P.handle_prove(job_for(sid), Ctx())
    check(out.get("prove") == "worse" and any(
        w["kind"] == "parse" for w in P.record_of(
            skills.version(sid, 1)).get("worse") or []),
          "[decide] code that parsed without the skill and does not with it "
          "is worse", out)


def _advance_and_arm(sid: str) -> dict:
    nxt = skills.advance(sid, 1, "prove", enqueue_next=False)
    if nxt == "arm":
        skill_pipeline.handle_arm({"payload": {"skill": sid, "version": 1,
                                               "stage": "arm"}}, Ctx(),
                                  inline=True)
    return skills.get(sid)


def test_tie_and_better_arm():
    reset_store()
    sid = make_skill("koota-tie")
    with with_fakes(FakeChat(GOOD_TS, GOOD_TS)):
        out = P.handle_prove(job_for(sid), Ctx())
    s = _advance_and_arm(sid)
    rec = P.record_of(skills.version(sid, 1))
    check(out.get("prove") == "tie" and rec.get("gain") ==
          "no measurable gain" and s["status"] == "armed",
          "[decide] a tie records 'no measurable gain' and the skill ARMS",
          (out, s["status"]))
    sid2 = make_skill("koota-better")
    with with_fakes(FakeChat(CAST_TS, GOOD_TS)):
        out2 = P.handle_prove(job_for(sid2), Ctx())
    s2 = _advance_and_arm(sid2)
    rec2 = P.record_of(skills.version(sid2, 1))
    check(out2.get("prove") == "better" and rec2.get("better")
          and not rec2.get("worse") and s2["status"] == "armed",
          "[decide] better (failed without, passed with; none worse) ARMS",
          (out2, s2["status"]))


class LengthThen(FakeChat):
    """Every generation ends on the token limit until `n` have been made."""

    def __init__(self, without, with_, n):
        super().__init__(without, with_)
        self.n = n

    def __call__(self, body):
        d = super().__call__(body)
        if len(self.bodies) <= self.n:
            d["choices"][0]["finish_reason"] = "length"
        return d


def test_budget_event_is_retried_once_then_unproven():
    reset_store()
    sid = make_skill("koota-budget")
    chat = FakeChat(GOOD_TS, CAST_TS, finish=("stop", "length"))
    with with_fakes(chat):
        out = P.handle_prove(job_for(sid), Ctx())
    ver = skills.version(sid, 1)
    rec = P.record_of(ver)
    first = [b for b in chat.bodies[:len(chat.bodies) // 2]]
    again = [b for b in chat.bodies[len(chat.bodies) // 2:]]
    check(out.get("prove") == "unproven" and out.get("quarantined")
          and ver["state"] == "quarantined"
          and "unproven" in (ver["reason"] or "")
          and all(not p["decided"] and "budget event" in p["why"]
                  for p in rec["probes"]),
          "[decide] every probe on the token limit, and again on the retry: "
          "UNPROVEN, quarantined, never armed", (out, ver["state"]))
    check(rec.get("retry") and len(again) == len(first) > 0
          and all(b["seed"] != a["seed"] for a, b in zip(first, again))
          and all(b["max_tokens"] >= a["max_tokens"]
                  for a, b in zip(first, again))
          and all(p.get("attempt") == 2 for p in rec["probes"]),
          "[decide] the retry runs each probe once more, with the job's full "
          "answer room and a second seed (the same on both sides)",
          [(a.get("max_tokens"), b.get("max_tokens")) for a, b in
           zip(first, again)])
    reset_store()
    sid = make_skill("koota-budget-2")
    chat = None
    with with_fakes(FakeChat(GOOD_TS, GOOD_TS)) as c0:
        n_probes = len(P.derive(sid, skills.version(sid, 1),
                                use_model=False)["probes"])
    chat = LengthThen(GOOD_TS, GOOD_TS, 2 * n_probes)
    with with_fakes(chat):
        out = P.handle_prove(job_for(sid), Ctx())
    check(out.get("prove") == "tie"
          and skills.version(sid, 1)["state"] == "running"
          and P.record_of(skills.version(sid, 1)).get("retry"),
          "[decide] a probe the retry decides counts: the skill goes on to "
          "arm", (out, c0 is not None))


def test_no_probe_is_unproven():
    reset_store()
    sid = make_skill("koota-no-probe", should=[])
    with with_fakes(FakeChat(GOOD_TS, GOOD_TS)):
        out = P.handle_prove(job_for(sid), Ctx())
    ver = skills.version(sid, 1)
    check(out.get("prove") == "unproven" and ver["state"] == "quarantined",
          "[decide] a skill with no probe to prove it with is UNPROVEN, "
          "quarantined", (out, ver["state"], ver["reason"]))


def test_the_stack_type_checker_is_the_default():
    """TYPE_CHECKER None is the stack's checker (mcp/typecheck.py); offline
    it starts no container and says so -- 'not run', never a pass."""
    import typecheck
    ok, why = typecheck.available()
    check(not ok and "offline" in why,
          "[types] an offline suite never starts a container", why)
    saved = P.TYPE_CHECKER
    try:
        P.TYPE_CHECKER = None
        ok, why = P.run_check({"kind": "types"}, GOOD_TS, "typescript",
                              ["koota"])
        check(ok is None and why.startswith("not run") and "offline" in why,
              "[types] the default checker is the stack's: not run offline, "
              "with its reason", why)
    finally:
        P.TYPE_CHECKER = saved
    check(typecheck.specs_for(["koota"]) == ["koota@0.6.6"]
          and "three@0.185.1" in typecheck.specs_for(["three"], version=
                                                     "0.185.1")
          and typecheck.pinned("@react-three/fiber", major=10)
          == "10.0.0-alpha.5"
          and typecheck.pinned("@react-three/fiber", major=9) == "9.8.0",
          "[types] the HELD version is installed: the skill's own, else the "
          "major it names, else the newest held",
          typecheck.specs_for(["three"]))
    sid = make_skill("r3f-v10-canvas")
    pin = P.pin_of(sid, {"text": "# R3F v10 Canvas"}, ["@react-three/fiber"])
    check(pin["majors"] == {"@react-three/fiber": 10},
          "[types] a skill that names v10 pins major 10", pin)


def test_transport_failure_raises_retryable():
    reset_store()
    sid = make_skill("koota-down")

    def down(body):
        raise OSError("connection refused")
    try:
        with with_fakes(down):
            P.handle_prove(job_for(sid), Ctx())
        ok, msg = False, "no raise"
    except RuntimeError as e:
        ok, msg = "retryable: yes" in str(e) and "remedy" in str(e), str(e)
    check(ok and not P.record_of(skills.version(sid, 1)),
          "[decide] a model that does not answer RAISES retryable (the "
          "worker retries); nothing is recorded from half the pairs", msg)


# ===========================================================================
# 5. the judge, last resort
# ===========================================================================
def test_judge_is_the_last_resort():
    reset_store()
    rule = C.rule_from_metadata({"artifacts": ["docs"]})
    items = [{"form": "DO", "situation": "", "quote": "", "ref": "",
              "text": "open the README with what the project does."}]
    sid = make_skill("readme-opening", items=items, rule=rule, should=[
        {"text": "Write a README for this project."},
        {"text": "Write the README's first section."}])
    asked = []

    def ask(system, user, *, max_tokens, purpose=""):
        asked.append(system)
        if system == P.JUDGE_SYSTEM:
            return json.dumps({"pass": "What it does" in user,
                               "why": "w"})
        return json.dumps({"probes": []})
    with with_fakes(FakeChat("# Tool\nWhat it does: x.", "# Tool\nHi."),
                    ask):
        out = P.handle_prove(job_for(sid), Ctx())
    rec = P.record_of(skills.version(sid, 1))
    js = [c for p in rec.get("probes") or [] for c in p["checks"]]
    check(js and all(c["kind"] == "judge" and c["judge"] == "model"
                     and c.get("caveat") for c in js)
          and asked.count(P.JUDGE_SYSTEM) == 2 * len(rec["probes"]) * (
              1 + P.REPEATS),
          "[judge] no code check can decide a prose probe: one model judge "
          "per answer, labelled judge: model with its caveat", js[:1])
    check(out.get("prove") == "worse",
          "[judge] the judge's verdicts are paired like any check", out)


# ===========================================================================
# 6. inline
# ===========================================================================
def test_inline_install_is_not_run_and_arms():
    reset_store()
    # The wiring the coordinator adds (skill_pipeline.HANDLERS), emulated in
    # this process when it is not there yet.
    added = []
    q = skills.JOBS.get("prove", P.JOB)[0]
    if q not in skill_pipeline.HANDLERS:
        skill_pipeline.HANDLERS[q] = P.handle_prove
        added.append(q)
    if "review" in skills.JOBS and skills.JOBS["review"][0] not in \
            skill_pipeline.HANDLERS:
        rq = skills.JOBS["review"][0]
        skill_pipeline.HANDLERS[rq] = lambda job, ctx: {
            "skipped": "review is not wired in this process"}
        added.append(rq)
    try:
        import skill_tests
        rule = C.rule_from_metadata({"languages": ["typescript"]})
        tests = skill_tests.generate(rule, description="Use when writing "
                                     "TypeScript update loops.")
        sk = {"name": "inline-proof", "title": "Inline Proof",
              "description": "Use when writing TypeScript update loops.",
              "items": ITEMS}
        fake = FakeChat(GOOD_TS, CAST_TS)
        with with_fakes(fake):
            s = skills.create_compiled(skill=sk, rule=rule, tests=tests,
                                       source="# Inline Proof\n- DO: x",
                                       origin="authored", author="test")
        ver = skills.version(s["id"], 1)
        rec = P.record_of(ver)
        in_path = "prove" in (ver.get("path") or [])
        check(not in_path or (rec.get("verdict") == "not_run"
                              and "inline install" in rec.get("why", "")),
              "[inline] an authored install records 'not run: inline "
              "install'", rec or ver.get("path"))
        check(s["status"] == "armed" and not fake.bodies,
              "[inline] it arms, and no model was called",
              (s["status"], s.get("reason"), len(fake.bodies)))
        # handle_prove(inline=True) says the same without the flag.
        sid = make_skill("inline-direct")
        with with_fakes(fake):
            out = P.handle_prove(job_for(sid), Ctx(), inline=True)
        check(out.get("prove") == "not_run" and not fake.bodies,
              "[inline] handle_prove(inline=True): not run, no model", out)
    finally:
        for q2 in added:
            skill_pipeline.HANDLERS.pop(q2, None)


# ===========================================================================
# 7. the backlog
# ===========================================================================
def test_backlog_order_skip_and_idle():
    reset_store()
    now = time.time()
    old = make_skill("backlog-old", stage="validate", arm=True,
                     updated=now - 3000)
    new = make_skill("backlog-new", stage="validate", arm=True,
                     updated=now - 10)
    mid = make_skill("backlog-mid", stage="validate", arm=True,
                     updated=now - 500)
    done = make_skill("backlog-done", stage="validate", arm=True,
                      updated=now)
    P.store(done, 1, {"verdict": "tie", "skill_version": 1, "seconds": 42.0})
    busy = make_skill("backlog-busy", stage="validate", arm=True,
                      updated=now - 1)
    jobs.add(P.JOB[0], {"skill": busy, "version": 1, "stage": "prove"},
             lane="gpu", dataset=f"skill:{busy}", stage="prove")
    rows, skipped = P.backlog()
    check([r["skill"] for r in rows] == [new, mid, old],
          "[backlog] newest-updated first; the proved and the busy skill "
          "left out", [r["name"] for r in rows])
    check(skipped == {"proved": 1, "live_job": 1},
          "[backlog] skipped counts: proved at its version, a live job",
          skipped)
    got = P.enqueue_backlog(limit=2)
    js = {j["id"]: j for j in jobs.listing(limit=50)}
    enq = [js[e["job"]] for e in got["enqueued"]]

    def pl(j):
        x = j["payload"]
        return x if isinstance(x, dict) else json.loads(x or "{}")
    check([e["skill"] for e in got["enqueued"]] == [new, mid]
          and all(j["lane"] == "gpu" and j["queue"] == P.JOB[0]
                  and pl(j).get("idle") is True
                  and pl(j).get("backlog") is True
                  for j in enq),
          "[backlog] enqueue_backlog(limit=2): the two newest, gpu lane, "
          "payload idle (worker.run_one defers them while busy)",
          [(j["lane"], j["payload"]) for j in enq])
    est = got["estimate"]
    check(est["measured"] == 1 and est["mean_seconds"] == 42.0
          and est["remaining"] == 1 and est["remaining_seconds"] == 42.0,
          "[backlog] the estimate is the MEASURED seconds per skill times "
          "what remains", est)
    check(P.schedule_backlog() is None,
          "[backlog] schedule_backlog adds nothing while a backlog job is "
          "live (one at a time)")
    # A proof at an OLD version does not count at the served one.
    P.store(old, 1, {"verdict": "tie", "skill_version": 0})
    rows2, _ = P.backlog()
    check(old in [r["skill"] for r in rows2],
          "[backlog] a proof of another version is not a proof of the "
          "served one")


def test_backlog_proof_that_finds_worse_disarms():
    reset_store()
    sid = make_skill("backlog-worse", stage="validate", arm=True)
    check(skills.get(sid)["status"] == "armed", "[backlog] the skill is "
          "armed before its proof")
    with with_fakes(FakeChat(GOOD_TS, CAST_TS)):
        out = P.handle_prove(job_for(sid, backlog=True, idle=True), Ctx())
    s = skills.get(sid)
    check(out.get("prove") == "worse" and s["status"] == "quarantined"
          and s["served_version"] is None and not any(
              x["id"] == sid for x in skills.armed()),
          "[backlog] a backlog proof that finds WORSE quarantines the "
          "armed skill: it is no longer served", (out, s["status"]))
    sid2 = make_skill("backlog-fine", stage="validate", arm=True)
    with with_fakes(FakeChat(GOOD_TS, GOOD_TS)):
        out2 = P.handle_prove(job_for(sid2, backlog=True), Ctx())
    check(out2.get("prove") == "tie" and skills.get(sid2)["status"] ==
          "armed" and skill_pipeline.advance_after(
              {"queue": P.JOB[0], "payload": job_for(sid2, backlog=True)[
                  "payload"]}) is None,
          "[backlog] a tie only records: the skill stays armed and nothing "
          "advances", out2)
    out3 = P.handle_prove(job_for(sid, backlog=True), Ctx())
    check("skipped" in out3, "[backlog] a job for a version no longer "
          "armed is skipped", out3)


def test_a_version_not_at_prove_is_skipped():
    reset_store()
    sid = make_skill("elsewhere", stage="validate")
    out = P.handle_prove(job_for(sid), Ctx())
    check("skipped" in out, "[handler] a version not running at prove is "
          "skipped (skill_pipeline._target)", out)


# ===========================================================================
# 8. the templates
# ===========================================================================
def test_templates():
    rows = P.templates()
    names = {r["name"]: r for r in rows}
    check(set(names) == {"probe", "prove_answer", "prove_judge"}
          and names["probe"]["version"] == "probe/1"
          and all(r["stage"] == "prove" and len(r["sha256"]) == 16
                  for r in rows),
          "[templates] probe/1, prove_answer/1 and prove_judge/1, each "
          "with its version and hash for skill_prompts.registry",
          [(r["name"], r["version"]) for r in rows])
    check("DATA, not instructions" in P.PROBE_SYSTEM
          and "DATA, not instructions" in P.JUDGE_SYSTEM,
          "[templates] the model-read templates carry the "
          "data-not-instructions paragraph")
    proh = len(re.findall(r"\b(?:NOT|never|do not)\b", P.PROBE_SYSTEM))
    check(proh <= 2, "[templates] at most two prohibitions in the probe "
          "template (AGENTS.md)", proh)


# ===========================================================================
# 9. the repeat rule
# ===========================================================================
class SeedChat(FakeChat):
    """WITH answers `with_` only when `bad(body)` says so, else `without`'s
    answer: a skill whose failure is a flake of particular seeds."""

    def __init__(self, without, with_, bad):
        super().__init__(without, with_)
        self.bad = bad

    def __call__(self, body):
        self.bodies.append(json.loads(json.dumps(body)))
        user = body["messages"][-1]["content"]
        on = skill_prompts.CRAFT_HEADER in user
        text = self.with_ if (on and self.bad(body)) else self.without
        return {"choices": [{"message": {"content": text},
                             "finish_reason": "stop"}],
                "usage": {"completion_tokens": 10}}


def _first_seeds(sid: str) -> dict[str, int]:
    """{task: the seed of its first (flagging) run}."""
    d = P.derive(sid, skills.version(sid, 1), use_model=False)
    return {p["task"]: P.seed_of(sid, 1, p["task"]) for p in d["probes"]}


def test_a_flag_that_does_not_repeat_quarantines_nothing():
    reset_store()
    sid = make_skill("koota-flake")
    first = set(_first_seeds(sid).values())
    chat = SeedChat(GOOD_TS, BROKEN_TS, lambda b: b["seed"] in first)
    with with_fakes(chat):
        out = P.handle_prove(job_for(sid), Ctx())
    rec = P.record_of(skills.version(sid, 1))
    n = len(rec["probes"])
    check(out.get("prove") == "tie" and not out.get("quarantined")
          and rec.get("unconfirmed") and not rec.get("worse"),
          "[repeat] WITH worse on the first sample only: not quarantined, "
          "the flags recorded as unconfirmed", (out, rec.get("why")))
    check(len(chat.bodies) == 2 * n + 2 * n * P.REPEATS,
          "[repeat] every flagged probe is run again: both sides, REPEATS "
          "more times", (len(chat.bodies), n))
    pairs = list(zip(chat.bodies[0::2], chat.bodies[1::2]))
    seeds = [x["seed"] for x, _y in pairs]
    check(all(x["seed"] == y["seed"] for x, y in pairs)
          and len(set(seeds)) == len(seeds),
          "[repeat] each repeat uses a NEW seed, the same on both sides "
          "(the old seed would give the old text back)", seeds)
    u = rec["unconfirmed"][0]
    check(u["runs"]["worse"] == 1 and u["runs"]["of"] == 1 + P.REPEATS
          and rec["rule"] == P.RULE and rec["repeats"] == P.REPEATS,
          "[repeat] the record carries the runs of a flag, the rule and "
          "the repeat count", (u.get("runs"), rec.get("rule")))
    check(all(len(p.get("repeats") or []) == P.REPEATS and
              all(r["seed"] != p["seed"] for r in p["repeats"])
              for p in rec["probes"]),
          "[repeat] each repeated probe keeps its repeat runs' seeds and "
          "checks")
    check(skills.version(sid, 1)["state"] == "running",
          "[repeat] the version goes on (the pipeline arms it)")


def test_a_flag_that_repeats_quarantines_with_the_runs():
    reset_store()
    sid = make_skill("koota-repeats")
    chat = SeedChat(GOOD_TS, BROKEN_TS, lambda b: True)
    with with_fakes(chat):
        out = P.handle_prove(job_for(sid), Ctx())
    rec = P.record_of(skills.version(sid, 1))
    s = skills.get(sid)
    check(out.get("prove") == "worse" and out.get("quarantined")
          and rec["worse"] and all(
              w["runs"]["worse"] == w["runs"]["of"] == 1 + P.REPEATS
              for w in rec["worse"]),
          "[repeat] a worse result that repeats QUARANTINES, each worse "
          "check with its runs", (out, rec["worse"][:1]))
    check("worse in 2 of 2 runs" in (s["reason"] or ""),
          "[repeat] the quarantine reason says how often it was worse",
          s["reason"])


def test_a_majority_of_the_runs_decides():
    """REPEATS = 2: three runs, worse in a strict majority (2 of 3)."""
    reset_store()
    sid = make_skill("koota-majority")
    first = _first_seeds(sid)
    saved = P.REPEATS
    try:
        P.REPEATS = 2
        # worse in the flagging run only: 1 of 3 -> unconfirmed
        a = set(first.values())
        with with_fakes(SeedChat(GOOD_TS, BROKEN_TS,
                                 lambda b: b["seed"] in a)):
            out1 = P.handle_prove(job_for(sid), Ctx())
        # worse in the flagging run and the first repeat: 2 of 3 -> worse
        reset_store()
        sid = make_skill("koota-majority")
        first = _first_seeds(sid)
        b_seeds = set(first.values()) | {
            P.seed_of(sid, 1, f"{t}#r1") for t in first}
        with with_fakes(SeedChat(GOOD_TS, BROKEN_TS,
                                 lambda b: b["seed"] in b_seeds)):
            out2 = P.handle_prove(job_for(sid), Ctx())
        rec2 = P.record_of(skills.version(sid, 1))
    finally:
        P.REPEATS = saved
    check(out1.get("prove") == "tie",
          "[repeat] 1 worse of 3 runs is not a majority", out1)
    check(out2.get("prove") == "worse" and all(
        w["runs"]["worse"] == 2 and w["runs"]["of"] == 3
        for w in rec2["worse"]),
          "[repeat] 2 worse of 3 runs is a majority", rec2.get("worse"))


def test_only_a_probe_with_a_flag_is_repeated():
    reset_store()
    sid = make_skill("koota-one-flag")
    tasks = list(_first_seeds(sid))
    only = tasks[0]
    chat = SeedChat(GOOD_TS, BROKEN_TS, lambda b: only in
                    (b["messages"][-1]["content"] or ""))
    with with_fakes(chat):
        out = P.handle_prove(job_for(sid), Ctx())
    n = len(tasks)
    check(len(chat.bodies) == 2 * n + 2 * P.REPEATS
          and out.get("prove") == "worse",
          "[repeat] one probe flagged: only that probe is run again",
          (len(chat.bodies), n))


# ===========================================================================
# 10. the re-prove
# ===========================================================================
def _quarantined_by_one_sample(name: str) -> str:
    """A skill the backlog's proof quarantined as worse, as the library's
    107 were: its record has no `rule`."""
    sid = make_skill(name, stage="validate", arm=True)
    saved = P.REPEATS
    try:
        P.REPEATS = 0           # the old one-sample rule
        with with_fakes(FakeChat(GOOD_TS, BROKEN_TS)):
            P.handle_prove(job_for(sid, backlog=True), Ctx())
    finally:
        P.REPEATS = saved
    rec = P.record_of(skills.version(sid, 1))
    assert skills.get(sid)["status"] == "quarantined" and rec["verdict"] == \
        "worse", (rec.get("verdict"), rec.get("rule"))
    # The library's records were written before the rule existed.
    rec.pop("rule", None)
    rec.pop("repeats", None)
    P.store(sid, 1, rec)
    return sid


def test_reprove_lists_and_enqueues_exactly_the_prove_quarantines():
    reset_store()
    flake = _quarantined_by_one_sample("reprove-flake")
    real = _quarantined_by_one_sample("reprove-real")
    act = make_skill("reprove-activation", stage="validate", arm=True)
    skills.quarantine(act, 1, "activation tests: should-case 2 did not "
                      "select it | retryable: no")
    fine = make_skill("reprove-armed", stage="validate", arm=True)
    rows, skipped = P.reprove_targets()
    check(sorted(r["skill"] for r in rows) == sorted([flake, real])
          and skipped["not_target"] >= 1,
          "[reprove] the targets are the skills PROVE quarantined as worse "
          "under the old rule: not an activation-test quarantine, not an "
          "armed skill", ([r["name"] for r in rows], skipped))
    dry = P.reprove_quarantined(dry_run=True)
    check(len(dry["would_enqueue"]) == 2 and not jobs.listing(limit=50),
          "[reprove] a dry run lists and enqueues nothing", list(dry))
    got = P.reprove_quarantined()
    js = {j["id"]: j for j in jobs.listing(limit=50)}

    def payload(e):
        x = js[e["job"]]["payload"]
        return x if isinstance(x, dict) else json.loads(x or "{}")
    pl = [payload(e) for e in got["enqueued"]]
    check(len(got["enqueued"]) == 2 and all(
        js[e["job"]]["lane"] == "gpu" and js[e["job"]]["queue"] == P.JOB[0]
        for e in got["enqueued"]) and all(
        x.get("reprove") and x.get("idle") and not x.get("backlog")
        for x in pl),
          "[reprove] one idle-gated gpu-lane job per target, payload "
          "reprove", pl)
    again = P.reprove_quarantined()
    check(not again["enqueued"] and again["skipped"]["live_job"] == 2,
          "[reprove] running it again enqueues nothing while the jobs live",
          again["skipped"])
    check(skills.get(fine)["status"] == "armed",
          "[reprove] enqueueing touches no skill")
    try:
        skills.rearm(act, 1)
        refused = False
    except ValueError:
        refused = True
    check(refused and skills.get(act)["status"] == "quarantined",
          "[reprove] skills.rearm refuses a version a prove did not "
          "quarantine (an activation-test quarantine stays)")
    import dash_skills
    code, _ct, raw = dash_skills.handle_post("/dash/api/skill/reprove",
                                             {"dry_run": True})
    body = json.loads(raw)
    check(code == 200 and body.get("ok") and not body.get("enqueued")
          and body["skipped"]["live_job"] == 2,
          "[reprove] POST /dash/api/skill/reprove is the same call (dry "
          "run: nothing enqueued)", body.get("skipped"))


def test_reprove_serves_a_flake_again_and_keeps_a_real_one_out():
    reset_store()
    # Serving a skill again rebuilds the trigger index (the embedder): not
    # offline.
    skill_select.refresh_triggers = lambda *a, **k: {"offline": True}
    flake = _quarantined_by_one_sample("reprove-flake")
    real = _quarantined_by_one_sample("reprove-real")
    # `flake`'s WITH answers fail only on the seeds of the first sample (the
    # old record's); `real`'s fail on every seed.
    seeds_flake = set(_first_seeds(flake).values())
    chat_flake = SeedChat(GOOD_TS, BROKEN_TS, lambda b: b["seed"] in
                          seeds_flake)
    with with_fakes(chat_flake):
        out = P.handle_prove(job_for(flake, reprove=True, idle=True), Ctx())
    s = skills.get(flake)
    rec = P.record_of(skills.version(flake, 1))
    check(out.get("rearmed") and s["status"] == "armed"
          and s["served_version"] == 1 and rec["rule"] == P.RULE
          and rec["mode"] == "reprove" and rec["reproved"]["from"]["verdict"]
          == "worse" and skills.version(flake, 1)["state"] == "armed"
          and any(x["id"] == flake for x in skills.armed()),
          "[reprove] the worse result did not repeat: the skill is SERVED "
          "again, the record stamped with the rule and the old verdict",
          (out, s["status"], rec.get("verdict")))
    with with_fakes(SeedChat(GOOD_TS, BROKEN_TS, lambda b: True)):
        out2 = P.handle_prove(job_for(real, reprove=True, idle=True), Ctx())
    s2 = skills.get(real)
    rec2 = P.record_of(skills.version(real, 1))
    check(out2.get("prove") == "worse" and out2.get("quarantined")
          and s2["status"] == "quarantined" and rec2["rule"] == P.RULE
          and "2 of 2 runs" in (s2["reason"] or "")
          and not any(x["id"] == real for x in skills.armed()),
          "[reprove] the worse result repeated: it stays quarantined, with "
          "the runs in the reason", (out2, s2["reason"]))
    rows, _ = P.reprove_targets()
    check(not rows, "[reprove] once decided under the rule a skill is not "
          "a target again", [r["name"] for r in rows])
    out3 = P.handle_prove(job_for(flake, reprove=True), Ctx())
    check("skipped" in out3,
          "[reprove] a job for a version that is no longer a target is "
          "skipped, not re-run", out3)


def test_reprove_leaves_an_unproven_one_quarantined():
    reset_store()
    sid = _quarantined_by_one_sample("reprove-budget")
    with with_fakes(FakeChat(GOOD_TS, CAST_TS, finish=("stop", "length"))):
        out = P.handle_prove(job_for(sid, reprove=True, idle=True), Ctx())
    s = skills.get(sid)
    check(out.get("prove") == "unproven" and s["status"] == "quarantined"
          and "unproven" in (s["reason"] or ""),
          "[reprove] a re-proof no check decided leaves it quarantined "
          "(never armed)", (out, s["status"]))


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        print(f"\n{name}", flush=True)
        try:
            fn()
        except Exception:                                        # noqa: BLE001
            check(False, f"{name} raised", traceback.format_exc()[-1500:])
    n_ok = sum(1 for ok, _w, _d in RESULTS if ok)
    print(f"\n{n_ok}/{len(RESULTS)} checks passed", flush=True)
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
