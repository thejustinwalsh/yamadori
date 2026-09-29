#!/usr/bin/env python
"""The library's rebuild to the ASSURED VOICE (mcp/skill_rebuild.py). No
GPU, no network, no live model.

    python mcp/test_skill_rebuild.py      -> "N/M checks passed"

Operator, 2026-09-28: "You can't just archive skills, you need to build new
skills ... Archive the doubt-skills only when their replacements are ready
to arm, or in the same step." What this gates:

  1. THE PLAN (read-only): a batch replaces the armed skills of its
     migration file, less its keep list; every other armed skill with a
     doubt line is a rebuild; a skill with no doubt line is left alone.
  2. QUEUE: each source becomes a URL skill carrying meta.replacement
     {batch, replaces}; each rebuild is a NEW VERSION of the same skill on
     PATHS["rebuild"]; every gpu job it queues waits for an idle stack.
  3. SETTLE: nothing is archived while a member of the batch is still in
     the pipeline, nor when none armed; when the batch has finished and one
     armed, the skills it replaces are ARCHIVED (backup first, the reason
     citing the operator), and a second settle changes nothing.
  4. A REBUILD through the review: the rewrite arms as the new version and
     supersedes the served one; a rebuild in which no item survives is
     archived by settle, the served version kept until then.

Every store is a temp path set BEFORE any mcp module is imported
(mcp/offline_stores.py).
"""
from __future__ import annotations

import json
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import offline_stores  # noqa: E402

_TMP = offline_stores.isolate("yamadori_test_skill_rebuild_")
os.environ["YAMADORI_SKILL_CACHE_SECONDS"] = "0"
os.environ.setdefault("YAMADORI_GPU_ROOM", "0")

import jobs  # noqa: E402
import skill_classify as C  # noqa: E402
import skill_md  # noqa: E402
import skill_pipeline  # noqa: E402
import skill_prompts  # noqa: E402
import skill_prove  # noqa: E402
import skill_rebuild as R  # noqa: E402
import skills  # noqa: E402

# The prove stage shapes its pair through the one door (model.shape): the
# pool size and the served efforts are stated offline, as
# mcp/test_skill_prove.py does, so nothing asks the model server.
import budget  # noqa: E402
import tiers  # noqa: E402
budget._POOL = 181248
tiers._accepted = ("low", "medium", "xhigh")

# The prove stage's pair through its injectables: one fixed answer on both
# sides (a tie arms). Its decisions are tested in mcp/test_skill_prove.py.
skill_prove.CHAT = lambda body: {"choices": [{"message": {
    "content": "```ts\nexport const x: number = 1;\n```"},
    "finish_reason": "stop"}]}
skill_prove.ASK = lambda system, user, *, max_tokens, purpose="": (
    '{"pass": true, "why": "fake"}' if system == skill_prove.JUDGE_SYSTEM
    else '{"probes": []}')

RESULTS: list[tuple[bool, str, object]] = []


def check(ok: bool, what: str, detail=None) -> None:
    RESULTS.append((bool(ok), what, detail))
    print(f"  {'pass' if ok else 'FAIL'}  {what}"
          + ("" if ok or detail is None else f"   <- {detail!r}"[:600]),
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


SRC = ("state.clock is gone in v10. Timing comes off the RAF timer: "
       "state.delta (seconds), state.elapsed.\n"
       "Update uniform values imperatively in useFrame, never through "
       "setState.\n"
       "Instance meshes beyond a few hundred similar objects with "
       "instancedMesh.\n")


def compiled(name: str, items: list[dict], *, file: str | None = None,
             meta: dict | None = None, origin: str = "migration") -> str:
    rule = C.rule_from_metadata({"frameworks": ["r3f"],
                                 "topics": ["useFrame"]})
    m = dict(meta or {})
    if file:
        m["migration_group"] = {"file": file, "topic": name}
    s = skills.create_compiled(
        skill={"name": name, "title": name.replace("-", " ").title(),
               "description": "Use when timing animation in React Three "
               "Fiber's useFrame.", "items": items},
        rule=rule, tests={}, source=SRC, origin=origin, author="test",
        meta=m)
    return s["id"]


def legacy(sid: str, items: list[dict]) -> None:
    """The skill as it was ARMED BEFORE skill_limits.doubt existed: its
    served text and validated items carry the given lines (a compiled
    install today drops a doubt line at validate)."""
    s = skills.get(sid)
    v = s["served_version"]
    ver = skills.version(sid, v)
    sk = skill_md.parse(ver["text"])
    sk["items"] = items
    val = dict(ver.get("validate") or {}, items=items)
    skills.update_version(sid, v, text=skill_md.render(sk), validate=val)
    skills._invalidate()


HISTORY = {"form": "DO", "text": "state.clock is gone in v10. Timing comes "
           "off the RAF timer: state.delta (seconds), state.elapsed.",
           "quote": "state.clock is gone in v10. Timing comes off the RAF "
           "timer: state.delta (seconds), state.elapsed."}
UNIFORMS = {"form": "DO", "text": "update uniform values imperatively in "
            "useFrame, not through setState.", "quote": "Update uniform "
            "values imperatively in useFrame, never through setState."}
INSTANCE = {"form": "DO", "text": "instance meshes beyond a few hundred "
            "similar objects with instancedMesh.", "quote": "Instance meshes "
            "beyond a few hundred similar objects with instancedMesh."}

PLAN = {"id": "test-plan", "batches": [
    {"id": "docs", "replaces_migration_files": ["r3f"],
     "keep": ["r3f-kept-one"],
     "provenance": {"r3f": {"commit": "abc123"}},
     "sources": [{"pkg": "r3f", "path": "docs/frame-loop.mdx",
                  "goal": "the frame loop"}]},
    {"id": "rebuild"}],
    "raw_url": {"r3f": "https://example.invalid/{commit}/{path}"}}


def test_the_plan_and_the_queue():
    reset_store()
    a = compiled("r3f-timing-a", [UNIFORMS], file="r3f")
    b = compiled("r3f-instancing-b", [INSTANCE], file="r3f")
    kept = compiled("r3f-kept-one", [INSTANCE], file="r3f")
    other = compiled("modern-timing", [INSTANCE], file="modern")
    clean = compiled("clean-uniforms", [UNIFORMS], file="modern")
    legacy(a, [HISTORY, UNIFORMS])
    legacy(b, [HISTORY, INSTANCE])
    legacy(kept, [HISTORY, INSTANCE])
    legacy(other, [HISTORY, INSTANCE])
    for sid in (a, b, kept, other, clean):
        check((skills.get(sid) or {}).get("status") == "armed",
              f"[setup] {sid} armed")
    p = R.plan(PLAN)
    docs = p["batches"][0]
    reb = p["batches"][-1]["rebuild"]
    check(sorted(x["id"] for x in docs["replaces"]) == sorted([a, b]),
          "[plan] the batch replaces its migration file's armed skills, "
          "less its keep list", docs["replaces"])
    check(sorted(x["id"] for x in reb) == sorted([kept, other])
          and all(x["doubt_lines"] for x in reb),
          "[plan] every other skill with a doubt line is a rebuild; a "
          "clean skill is left alone", reb)
    check(docs["sources"][0]["url"] ==
          "https://example.invalid/abc123/docs/frame-loop.mdx",
          "[plan] a source's URL is the pinned commit's", docs["sources"])
    made = R.queue(PLAN)
    new = skills.get(made["created"][0]["id"])
    rp = (new.get("meta") or {}).get("replacement") or {}
    check(rp.get("batch") == "docs" and sorted(rp.get("replaces") or [])
          == sorted([a, b]) and new["status"] == "pipeline",
          "[queue] a source is a new URL skill that names what it replaces",
          rp)
    vers = {r["id"]: skills.version(r["id"], r["version"])
            for r in made["rebuilt"]}
    check(all(v["origin"] == "rebuild" and v["stage"] == "review"
              and v["state"] == "running" for v in vers.values())
          and all(skills.get(sid)["served_version"] == 1 for sid in vers),
          "[queue] a rebuild is a new version on PATHS['rebuild'], the "
          "served one serving meanwhile", {k: (v["origin"], v["stage"])
                                           for k, v in vers.items()})
    con = jobs._db()
    try:
        rows = con.execute("SELECT queue, lane, payload FROM jobs").fetchall()
    finally:
        con.close()
    gpu = [json.loads(r[2]) for r in rows if r[1] == "gpu"]
    check(len(gpu) == 2 and all(pl.get("idle") for pl in gpu),
          "[queue] every gpu job it queues waits for an idle stack (the "
          "rebuilds' review; a source starts at its net fetch)",
          [(r[0], r[1]) for r in rows])
    return a, b, new["id"]


def test_settle_archives_only_when_the_batch_is_ready():
    a, b, member = test_the_plan_and_the_queue()
    out = R.settle("docs")
    check(out["waiting"] and not out["archived"]
          and skills.get(a)["status"] == "armed",
          "[settle] nothing is archived while a member is in the pipeline",
          out)
    # The member fails (a fetch it cannot make here): the batch finished
    # with NOTHING armed -- nothing is archived.
    skills.fail(member, 1, "fetch: offline test")
    out = R.settle("docs")
    check(out["nothing_armed"] == ["docs"]
          and skills.get(a)["status"] == "armed",
          "[settle] a batch that armed nothing archives nothing", out)
    # A replacement arms: the batch is ready.
    arm = compiled("r3f-frame-loop-doc", [UNIFORMS, INSTANCE],
                   origin="authored",
                   meta={"replacement": {"batch": "docs",
                                         "plan": "test-plan",
                                         "replaces": [a, b]}})
    check(skills.get(arm)["status"] == "armed", "[setup] a replacement arms")
    out = R.settle("docs")
    got = out["archived"][0] if out["archived"] else {}
    check(sorted(got.get("ids") or []) == sorted([a, b])
          and skills.get(a)["status"] == "archived"
          and skills.get(b)["status"] == "archived"
          and "operator, 2026-09-28" in (skills.get(a)["reason"] or ""),
          "[settle] the batch finished with one armed: what it replaces is "
          "ARCHIVED, the reason citing the operator", out)
    check(got.get("backup") and os.path.isdir(got["backup"])
          and os.path.exists(os.path.join(got["backup"], "archived.json")),
          "[settle] the archived skills are backed up first", got)
    ids = {s["id"] for s in skills.armed()}
    check(a not in ids and b not in ids and arm in ids,
          "[settle] the archived skills serve nothing; the replacement does")
    again = R.settle("docs")
    check(not again["archived"], "[settle] a second settle changes nothing",
          again)


def test_a_rebuild_through_the_review():
    reset_store()
    sid = compiled("modern-timing", [UNIFORMS], file="modern")
    empty = compiled("modern-only-history", [INSTANCE], file="modern")
    legacy(sid, [HISTORY, UNIFORMS])
    legacy(empty, [HISTORY])
    v = R.rebuild_version(sid, enqueue=False)
    v2 = R.rebuild_version(empty, enqueue=False)
    saved = skill_pipeline.ask_model

    def fake(system, user, *, max_tokens, purpose=""):
        if system == skill_prompts.REVIEW_SYSTEM:
            return json.dumps({"items": [
                {"i": 1, "verdict": "rewrite", "text": "DO: read timing "
                 "from state.delta (seconds) and state.elapsed."}]})
        raise AssertionError(f"unexpected prompt: {system[:40]}")
    skill_pipeline.ask_model = fake
    try:
        skill_pipeline.run_inline(sid, v, model=True)
        skill_pipeline.ask_model = lambda *a_, **k: json.dumps(
            {"items": [{"i": 1, "verdict": "drop", "reason": "history"}]})
        skill_pipeline.run_inline(empty, v2, model=True)
    finally:
        skill_pipeline.ask_model = saved
    s = skills.get(sid)
    body = skill_md.injection(skills.version(sid, s["served_version"])
                              ["text"])
    check(s["served_version"] == v and "read timing from state.delta" in body
          and "gone in v10" not in body
          and skills.version(sid, 1)["state"] == "superseded",
          "[rebuild] the review's rewrite arms as the new version and "
          "supersedes the served one", body)
    e = skills.get(empty)
    check(e["served_version"] == 1
          and skills.version(empty, v2)["state"] == "failed"
          and R.NO_ITEM in (skills.version(empty, v2)["reason"] or ""),
          "[rebuild] a rebuild that keeps no item fails; the served "
          "version serves until settle", e["reason"])
    out = R.settle()
    check(skills.get(empty)["status"] == "archived"
          and any(x["batch"] == "rebuild" and empty in x["ids"]
                  for x in out["archived"])
          and skills.get(sid)["status"] == "armed",
          "[rebuild] settle archives the skill whose rebuild kept no item, "
          "and leaves the rebuilt one armed", out)


def test_a_clean_pack_waits_for_its_pages_to_pass_prove():
    """Operator, 2026-09-28: a clean how-to pack is archived only when its
    replacement has passed PROVE -- here, when EVERY page mapped to it has
    an armed replacement whose served version proved better or tie."""
    reset_store()
    doubt = compiled("r3f-history-pack", [INSTANCE], file="r3f")
    legacy(doubt, [HISTORY, INSTANCE])
    clean = compiled("r3f-clean-pack", [UNIFORMS, INSTANCE], file="r3f")
    plan_doc = {"id": "test-plan-2", "batches": [
        {"id": "docs", "replaces_migration_files": ["r3f"],
         "replaces_when_proved": {"r3f-clean-pack": ["r3f:docs/a.mdx",
                                                     "r3f:docs/b.mdx"]},
         "provenance": {"r3f": {"commit": "abc"}},
         "sources": [{"pkg": "r3f", "path": "docs/a.mdx", "goal": "a"},
                     {"pkg": "r3f", "path": "docs/b.mdx", "goal": "b"}]},
        {"id": "rebuild"}],
        "raw_url": {"r3f": "https://example.invalid/{commit}/{path}"}}
    p = R.plan(plan_doc)["batches"][0]
    check([x["id"] for x in p["replaces"]] == [doubt]
          and [x["id"] for x in p["when_proved"]] == [clean],
          "[clean] the doubt-bearing pack goes with the batch; the clean one "
          "waits for its pages", p)
    made = R.queue(plan_doc, enqueue=False)
    for m in made["created"]:
        skills.fail(m["id"], 1, "fetch: offline test")

    def replacement(name, source, verdict):
        sid = compiled(name, [UNIFORMS], origin="authored", meta={
            "replacement": {"batch": "docs", "plan": "test-plan-2",
                            "replaces": [doubt], "source": source,
                            "when_proved": {clean: ["r3f:docs/a.mdx",
                                                    "r3f:docs/b.mdx"]}}})
        v = skills.get(sid)["served_version"]
        skills.update_version(sid, v, prove={"verdict": verdict})
        return sid
    replacement("r3f-doc-a", "r3f:docs/a.mdx", "tie")
    b = replacement("r3f-doc-b", "r3f:docs/b.mdx", "not_run")
    out = R.settle("docs")
    check(skills.get(doubt)["status"] == "archived"
          and skills.get(clean)["status"] == "armed"
          and any(w["id"] == clean and w["missing"] == ["r3f:docs/b.mdx"]
                  for w in out["clean_waiting"]),
          "[clean] the batch settles the doubt-bearing pack; the clean pack "
          "stays while one of its pages has not passed PROVE", out)
    skills.update_version(b, skills.get(b)["served_version"],
                          prove={"verdict": "better"})
    out = R.settle("docs")
    check(skills.get(clean)["status"] == "archived"
          and "passed PROVE" in (skills.get(clean)["reason"] or ""),
          "[clean] every page passed PROVE: the clean pack is archived, "
          "the reason says so", out)


def main() -> int:
    for fn in (test_settle_archives_only_when_the_batch_is_ready,
               test_a_rebuild_through_the_review,
               test_a_clean_pack_waits_for_its_pages_to_pass_prove):
        try:
            fn()
        except Exception:                                        # noqa: BLE001
            check(False, f"{fn.__name__} raised",
                  traceback.format_exc()[-1500:])
    ok = sum(1 for r in RESULTS if r[0])
    print(f"\n{ok}/{len(RESULTS)} checks passed")
    return 0 if ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
