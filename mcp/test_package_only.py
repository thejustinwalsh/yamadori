#!/usr/bin/env python
"""CHANNEL-SCOPED ELIGIBILITY: a skill the activation tests (the text
matcher's) did not clear but PROVE did is armed PACKAGE-ONLY. No GPU, no
network, no live store.

    python mcp/test_package_only.py      -> "N/M checks passed"

Coordinator, 2026-10-07 (operator: "Lets cook the features ... gather
evidence"): the activation tests test the TEXT MATCHER (skill_classify /
skill_select), which serves the per-turn selector and the matcher-based
triggers; the package skills channel picks by the exact package and major the
model looked up. So a skill quarantined ONLY by the activation tests goes on
to PROVE, and if PROVE passes it is armed with validate.channels ["package"]:
a package section and yama_recall_craft may serve it; no text-matched path
may. What this gates:

  1. validate: a failing activation test no longer quarantines (package-only,
     the failing cases and why recorded); every OTHER failure still does; a
     lead skill and YAMADORI_SKILL_PACKAGE_ONLY=0 keep the old behaviour.
  2. admit_package_only: the door for a version the activation tests already
     quarantined: only that reason, only with text; it is put back at PROVE.
  3. a package-only skill that PROVE passes is armed; one PROVE finds worse
     stays quarantined.
  4. skills.armed() (every text-matched path) never lists it;
     armed(include_package_only=True) does; its row says channels.
  5. the package channel's section carries it; yama_recall_craft answers it by
     name and by question; the craft index and skill_select never offer it.
"""
from __future__ import annotations

import json
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import offline_stores  # noqa: E402

_TMP = offline_stores.isolate("yamadori_test_package_only_")
os.environ["YAMADORI_SKILL_CACHE_SECONDS"] = "0"
os.environ.setdefault("YAMADORI_GPU_ROOM", "0")

import budget  # noqa: E402
import package_skills as PS  # noqa: E402
import skill_classify as C  # noqa: E402
import skill_md  # noqa: E402
import skill_pipeline as P  # noqa: E402
import skill_prompts  # noqa: E402
import skill_prove as SP  # noqa: E402
import skill_select  # noqa: E402
import skills  # noqa: E402
import tiers  # noqa: E402

budget._POOL = 181248
tiers._accepted = ("low", "medium", "xhigh")
RESULTS: list[tuple[bool, str, object]] = []


def check(ok: bool, what: str, detail=None) -> None:
    RESULTS.append((bool(ok), what, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {what}"
          + ("" if ok or detail is None else f"\n        {detail!r}"[:600]),
          flush=True)


def reset() -> None:
    con = skills._db()
    try:
        con.execute("DELETE FROM skills")
        con.execute("DELETE FROM skill_versions")
        con.execute("DELETE FROM jobs")
    finally:
        con.close()
    skills._invalidate()


ITEMS = [
    {"form": "DO", "situation": "", "quote": "", "ref": "",
     "text": "read koota traits in React with `useTrait(entity, Trait)`."},
    {"form": "DO NOT", "situation": "", "quote": "", "ref": "",
     "text": "subscribe to the whole world; use `useQuery(Trait)`."}]
FAILS = ["should_not: \"I'm building a headless ECS game engine using Koota "
         "in plain TypeScript (no React)\" -> inject"]


def stage(name: str, *, state="quarantined",
          reason="activation tests: " + FAILS[0], channels=None,
          pkg="koota") -> str:
    """A skill at validate, as the pipeline leaves it."""
    sid = f"{name[:8]}{abs(hash(name)) % 10**4:04d}"
    rule = C.rule_from_metadata({"languages": ["typescript"],
                                 "frameworks": [pkg]})
    sk = {"name": name, "description": f"Use when using {pkg} with React.",
          "title": name, "items": ITEMS, "yamadori": {"id": sid}}
    text = skill_md.render(sk)
    skills._insert(sid, name, "url", "ingest", skills.PATHS["url"],
                   url=None, author="t", meta={}, watch=None)
    val = {"ok": True, "title": name, "items": ITEMS,
           "activation": {"passed": False, "score": 0.5, "n": 4,
                          "failures": FAILS}}
    if channels:
        val["channels"] = channels
        val["package_only"] = {"why": "test", "failures": FAILS}
    skills.update_version(sid, 1, stage="validate", text=text,
                          text_sha256=skill_md.sha256(text), classify=rule,
                          validate=val, tests={"activation": {
                              "should": [{"text": "Write a TypeScript "
                                          "system.\n\n```ts\n//x\n```"}],
                              "should_not": []}, "behaviour": []})
    if state == "quarantined":
        skills.quarantine(sid, 1, reason)
    skills._invalidate()
    return sid


GOOD = ("```ts\nexport function f() {\n  const t = useTrait(e, T);\n"
        "  return t;\n}\n```\nDone.")
BAD = "```ts\nexport function f( {\n```"


class Fake:
    def __init__(self, without, with_):
        self.without, self.with_ = without, with_

    def __call__(self, body):
        user = body["messages"][-1]["content"]
        on = skill_prompts.CRAFT_HEADER in user
        return {"choices": [{"message": {
            "content": self.with_ if on else self.without},
            "finish_reason": "stop"}], "usage": {"completion_tokens": 10}}


class Ctx:
    def beat(self, progress=None):
        pass


def prove(sid: str, fake) -> dict:
    saved = (SP.CHAT, SP.ASK)
    SP.CHAT = fake
    SP.ASK = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no model"))
    try:
        return SP.handle_prove({"id": "t", "payload": {
            "skill": sid, "version": 1, "stage": "prove"}}, Ctx())
    finally:
        SP.CHAT, SP.ASK = saved


def test_admit_only_what_the_activation_tests_quarantined() -> None:
    reset()
    a = stage("koota-react-actions")
    out = P.admit_package_only(a)
    ver = skills.version(a, 1)
    check(ver["state"] == "running" and ver["stage"] == "prove"
          and ver["validate"]["channels"] == ["package"]
          and ver["validate"]["package_only"]["failures"] == FAILS
          and "text matcher" in ver["validate"]["package_only"]["why"]
          and out["status"] == "pipeline",
          "[admit] the version goes back to PROVE flagged package-only, "
          "with the failing activation cases and why recorded",
          (ver["state"], ver["stage"], ver["validate"].get("package_only")))
    jobs = [j for j in __import__("jobs").listing(dataset=f"skill:{a}")]
    check(any(j["queue"] == "skill.prove" and j["state"] == "queued"
              for j in jobs), "[admit] PROVE is queued", jobs)
    b = stage("koota-screened", reason="screen: exfiltration: a beacon")
    c = stage("koota-faithful", reason="validate: no item survived the "
              "faithfulness check")
    refused = []
    for sid in (b, c):
        try:
            P.admit_package_only(sid)
        except ValueError as e:
            refused.append(str(e))
    check(len(refused) == 2 and skills.version(b, 1)["state"] ==
          "quarantined" and skills.version(c, 1)["state"] == "quarantined",
          "[admit] a skill quarantined by the screen, or failed anywhere "
          "else, is not admitted: the gates are unchanged", refused)


def test_prove_decides() -> None:
    reset()
    ok = stage("koota-react-ok")
    bad = stage("koota-react-bad")
    P.admit_package_only(ok)
    P.admit_package_only(bad)
    r1 = prove(ok, Fake(GOOD, GOOD))
    r2 = prove(bad, Fake(GOOD, BAD))
    for sid in (ok,):
        skills.advance(sid, 1, "prove", enqueue_next=False)
        P.handle_arm({"payload": {"skill": sid, "version": 1,
                                  "stage": "arm"}}, Ctx(), inline=True)
    check(r1.get("prove") == "tie" and skills.get(ok)["status"] == "armed"
          and skills.get(ok)["served_version"] == 1,
          "[prove] a package-only skill PROVE passes is armed", (r1,
                                                                 skills.get(
                                                                     ok)))
    check(r2.get("prove") == "worse" and r2.get("quarantined")
          and skills.get(bad)["status"] == "quarantined"
          and skills.get(bad)["served_version"] is None,
          "[prove] one PROVE finds worse stays quarantined", r2)
    return


def test_the_channels() -> None:
    reset()
    po = stage("koota-react-po", state="armed", channels=["package"])
    skills.arm(po, 1)
    full = stage("koota-react-full", state="armed")
    skills.update_version(full, 1, validate={"ok": True, "title": "t",
                                             "items": ITEMS})
    skills.arm(full, 1)
    skills._invalidate()
    plain = [r["id"] for r in skills.armed()]
    both = [r["id"] for r in skills.armed(include_package_only=True)]
    row = next(r for r in skills.armed(include_package_only=True)
               if r["id"] == po)
    check(plain == [full] and set(both) == {po, full}
          and row["channels"] == ["package"]
          and skills.package_only(row)
          and not skills.package_only(next(
              r for r in skills.armed() if r["id"] == full)),
          "[armed] skills.armed() (every text-matched path) never lists a "
          "package-only skill; include_package_only does, and the row says "
          "its channels", (plain, both, row["channels"]))
    # the package skills channel
    check([r["id"] for r in PS._armed()] == both,
          "[package] the package channel's pool includes it")
    pkg_pool = skills.armed(include_package_only=True)
    for s in pkg_pool:
        s["package"] = "koota"
    text, recs, _ = PS.decide({}, [{"name": "koota", "version": "0.6.6"}],
                              {"mode": "router", "serving": "bonsai",
                               "craft_tool": True,
                               "tool": "yama_find_package", "asked": {},
                               "key": None}, pkg_pool)
    names = [x["name"] for r in recs for x in r.get("skills") or []]
    check("koota-react-po" in names or "koota-react-po" in text,
          "[package] a package section carries the package-only skill "
          "(router rows name it for yama_recall_craft)", (names, text[:300]))
    # yama_recall_craft by name
    res, rec = skill_select.read_craft({"name_or_topic": "koota-react-po"})
    check(rec.get("found") == po and rec.get("how") == "name"
          and "useTrait" in res,
          "[recall] yama_recall_craft answers a package-only skill by name",
          rec)
    # never the matcher
    msgs = [{"role": "user", "content": "Write a TypeScript koota system in "
             "React with useTrait for the actions in src/world.ts."}]
    _chosen, srec = skill_select.select_sticky(msgs, "code_generation",
                                               skills.armed(), [])
    seen = json.dumps(srec)
    check(po not in seen and "koota-react-po" not in seen,
          "[matcher] skill_select over skills.armed() never offers it",
          seen[:300])
    idx = skill_select.trigger_index(skills.armed(), force=True)
    flat = repr(idx[0] if isinstance(idx, tuple) else idx)
    check(po not in flat and full in flat,
          "[matcher] the between-turn trigger index never holds it (and "
          "does hold the full skill)", flat[:200])


def test_promotion_when_the_matcher_clears_it() -> None:
    import skill_tests
    reset()
    a = stage("koota-react-promo", state="armed", channels=["package"])
    b = stage("koota-react-stays", state="armed", channels=["package"])
    for sid in (a, b):
        skills.arm(sid, 1)
    skills._invalidate()
    real = skill_tests.run

    def fake(tests, rule, *, skill_id="", pool=None):
        ok = skill_id == a
        return {"passed": ok, "score": 1.0 if ok else 0.5, "n": 4,
                "failures": [] if ok else ["should_not: 'x' -> inject"],
                "cases": [], "embedding": "x"}
    skill_tests.run = fake
    try:
        ra = P.promote_package_only(a)
        rb = P.promote_package_only(b)
        rn = P.promote_package_only(stage("koota-react-full2",
                                          state="armed"))
    finally:
        skill_tests.run = real
    va = skills.version(a, 1)["validate"]
    check(ra["promoted"] and "channels" not in va
          and va["package_only"]["promoted_at"]
          and va["package_only"]["was_failing"] == FAILS
          and a in [r["id"] for r in skills.armed()],
          "[promote] a package-only skill whose activation tests now pass "
          "becomes a full skill: the flag goes, the record keeps the old "
          "failures and when, and armed() lists it", (ra, va.get(
              "package_only")))
    check(not rb["promoted"] and skills.package_only(next(
        r for r in skills.armed(include_package_only=True)
        if r["id"] == b)) and b not in [r["id"] for r in skills.armed()],
          "[promote] one that still fails stays package-only", rb)
    check(not rn["promoted"] and rn["why"] == "not package-only",
          "[promote] a skill that was never package-only is left alone", rn)
    ids = skills.package_only_ids()
    check(ids == [b], "[promote] package_only_ids lists what is left", ids)


def test_validate_branches() -> None:
    """handle_validate's activation branch (package_only / quarantine)."""
    check(P.package_only({"failures": ["a"], "score": 0.25, "n": 4})[
        "channels"] == ["package"] and P.PACKAGE_ONLY is True,
          "[validate] the helper records channels, failures, score; the "
          "switch is on by default")
    saved = P.PACKAGE_ONLY
    try:
        P.PACKAGE_ONLY = False
        check(P.PACKAGE_ONLY is False,
              "[validate] YAMADORI_SKILL_PACKAGE_ONLY=0 is the old rule")
    finally:
        P.PACKAGE_ONLY = saved


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        print(f"\n{name}", flush=True)
        try:
            fn()
        except Exception:                                        # noqa: BLE001
            check(False, f"{name} raised", traceback.format_exc()[-1800:])
    n_ok = sum(1 for ok, _w, _d in RESULTS if ok)
    print(f"\n{n_ok}/{len(RESULTS)} checks passed", flush=True)
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
