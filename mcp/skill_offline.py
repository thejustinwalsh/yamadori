#!/usr/bin/env python
"""The skill pipeline with the model stages' replies written by hand.

    python mcp/skill_offline.py skills/ingested/pmndrs/spec.json
    python mcp/skill_offline.py SPEC --temp --fixtures DIR   # offline, a temp store
    python mcp/skill_offline.py SPEC --dry-run               # a temp store, live fetch

WHY (operator, 2026-09-26): "NO requests to :1234 and no model runs" while
fixes are in flight -- and the maintainers of koota and pmndrs/math ship
their own Agent Skills, which should become OUR atomic skills through the
ONE pipeline (fetch -> screen -> licence -> decompose | distil -> classify
-> tests -> validate -> arm), not be authored around it. The model stages
PROPOSE and the code verifies (skill_pipeline's docstring); here the
proposals are files a person (or an agent) wrote by reading the pinned
source -- the same stand-in bench/skills/frontier_example.py uses -- and
every check the code makes still runs: robots and the size cap on the
fetch, the deterministic screen, the licence from a verbatim quote, every
item's quote found in the source, the taxonomy, the caps, the activation
tests, arming.

WHAT IS SKIPPED, AND SAID SO
  screen_model  the model screen (YAMADORI_SKILL_MODEL_SCREEN=0; the
                version records {"skipped": ...}); the deterministic screen
                runs
  faithful      NOT SKIPPABLE since 2026-09-27 (the operator's plan from
                docs/research/SKILLS-RESEARCH.md Part 4.5): the check is
                mandatory before a model-written skill arms. A spec with no
                "faithful" key arms NOTHING (each skill stops at validate);
                a spec's "faithful": {"by": ...} is a verdict written by
                hand for every item, recorded with who wrote it. Before
                2026-09-27 it was skipped (YAMADORI_SKILL_MODEL_VALIDATE=0);
                the quotes are still
                verified character for character

THE SPEC (JSON)
  {"proposer": who wrote the replies,
   "sources": [{"url", "kind": "frontier" | "url", "name", "goal",
                "reply": "<file beside the spec>", "provenance": {...}}],
   "tag":   {"<skill name>": the tag template's reply},
   "tests": {"<skill name>": the tests template's reply}}
A reply is found by the child's description (tag, tests) or by the stage
(decompose, distil: the source's `reply`, sent for the first chunk; later
chunks get an empty reply). A stage with no written reply RAISES: nothing
here ever calls a model.

NO QUEUE: skills.create(..., enqueue_first=False) and run_inline, so the
worker never claims a stage (it would run the model ones).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


class NoReply(RuntimeError):
    """A model stage this spec wrote no reply for."""


def load_spec(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        spec = json.load(f)
    base = os.path.dirname(os.path.abspath(path))
    for src in spec.get("sources") or []:
        with open(os.path.join(base, src["reply"]), encoding="utf-8") as f:
            src["_reply"] = f.read()
    spec["_dir"] = base
    return spec


def _descriptions(spec: dict) -> dict[str, str]:
    """{skill name: description} of every skill the replies propose."""
    import skill_prompts
    out = {}
    for src in spec.get("sources") or []:
        blocks = (skill_prompts.parse_decompose(src["_reply"])
                  if src.get("kind") == "frontier"
                  else [skill_prompts.parse_skill_reply(src["_reply"])])
        for b in blocks:
            if b.get("name"):
                out[b["name"]] = b.get("description") or ""
    return out


def responder(spec: dict):
    """An ask_model replacement that serves the spec's written replies."""
    import skill_prompts
    import skill_screen
    by_url = {src["url"]: src for src in spec.get("sources") or []}
    descs = _descriptions(spec)
    state = {"source": None}

    def who(user: str) -> str | None:
        hits = [n for n, d in descs.items() if d and d in user]
        return max(hits, key=lambda n: len(descs[n])) if hits else None

    def ask(system, user, *, max_tokens, temperature=0.2, purpose=""):
        src = state["source"]
        if system == skill_prompts.DECOMPOSE_SYSTEM:
            if src is None or src.get("kind") != "frontier":
                raise NoReply("decompose: no frontier reply for this source")
            return src["_reply"]
        if system == skill_prompts.DISTIL_SYSTEM:
            if src is None or src.get("kind") == "frontier":
                raise NoReply("distil: no distil reply for this source")
            m = re.search(r"(?m)^PART: (\d+) of (\d+)$", user)
            return src["_reply"] if not m or m.group(1) == "1" else \
                "name:\n# \n"
        if system == skill_prompts.tag_system():
            n = who(user)
            if n and n in (spec.get("tag") or {}):
                return json.dumps(spec["tag"][n])
            raise NoReply(f"tag: no written reply for {n or 'this skill'}")
        if system == skill_prompts.TESTS_SYSTEM:
            n = who(user)
            if n and n in (spec.get("tests") or {}):
                return json.dumps(spec["tests"][n])
            raise NoReply(f"tests: no written reply for {n or 'this skill'}")
        if system == skill_screen.SCREEN_SYSTEM:
            raise NoReply("screen_model is switched off here")
        if system == skill_prompts.FAITHFUL_SYSTEM:
            by = (spec.get("faithful") or {}).get("by")
            if not by:
                raise NoReply("faithful: the check is mandatory and this "
                              "spec gives no verdict (a \"faithful\": "
                              "{\"by\": ...} key)")
            n = len(re.findall(r"(?m)^(\d+)\. ", user))
            return json.dumps({"items": [
                {"i": i, "faithful": True, "why": f"verdict by {by}"}
                for i in range(1, n + 1)]})
        raise NoReply(f"{purpose or 'a model stage'}: no written reply")

    ask.state = state
    ask.by_url = by_url
    return ask


def fixture_fetcher(fixtures: str):
    """A fetch_source replacement reading DIR/<host>/<path>: the pinned
    sources, offline."""
    import time
    import urllib.parse
    import skill_pipeline

    def fetch(url, beat=None):
        p = urllib.parse.urlsplit(url)
        path = os.path.join(fixtures, p.hostname or "", *p.path.split("/"))
        if not os.path.isfile(path):
            raise skill_pipeline.Refused(f"GET {url} answered HTTP 404 "
                                         "(no fixture)")
        with open(path, "rb") as f:
            raw = f.read()
        return raw, {"url": url, "final_url": url, "status": 200,
                     "content_type": "text/plain", "charset": "utf-8",
                     "fetched_at": time.time(),
                     "robots": {"allowed": True, "why": "fixture"}}
    return fetch


def ingest(spec: dict, *, fetcher=None) -> list[dict]:
    """Run every source of the spec through the pipeline, inline. Returns
    one report per source: its stages, and each skill it produced."""
    import skill_pipeline
    import skills
    os.environ["YAMADORI_SKILL_MODEL_SCREEN"] = "0"
    saved = (skill_pipeline.ask_model, skill_pipeline.fetch_source,
             skill_pipeline.MODEL_SCREEN, skill_pipeline.MODEL_VALIDATE,
             skill_pipeline.MODEL_TAG, skill_pipeline.MODEL_TESTS)
    ask = responder(spec)
    skill_pipeline.ask_model = ask
    skill_pipeline.MODEL_SCREEN = False
    skill_pipeline.MODEL_VALIDATE = True
    skill_pipeline.MODEL_TAG = True
    skill_pipeline.MODEL_TESTS = True
    if fetcher is not None:
        skill_pipeline.fetch_source = fetcher
    out = []
    try:
        for src in spec.get("sources") or []:
            ask.state["source"] = src
            meta = {"provenance": dict(src.get("provenance") or {}),
                    "offline": {"spec": os.path.relpath(
                        spec["_dir"], ROOT).replace("\\", "/"),
                        "reply": src["reply"],
                        "proposer": spec.get("proposer"),
                        "skipped": ["screen_model"],
                        "faithful_by": (spec.get("faithful") or {}).get(
                            "by")}}
            s = skills.create(url=src["url"], name=src.get("name"),
                              goal=src.get("goal") or "",
                              frontier=src.get("kind") == "frontier",
                              author=spec.get("author") or "operator:offline",
                              meta=meta, enqueue_first=False, watch_hours=0)
            rec = {"source": src["url"], "id": s["id"], "name": s["name"]}
            try:
                # model=False: no written reply stands in for the review
                # or the paired probes; both record "not run".
                rec["stages"] = skill_pipeline.run_inline(s["id"], 1,
                                                          model=False)
            except (NoReply, RuntimeError) as e:
                rec["error"] = str(e)
            got = skills.get(s["id"]) or {}
            rec["status"], rec["reason"] = got.get("status"), got.get("reason")
            kids = skills.children(s["id"]) if src.get("kind") == "frontier" \
                else [got]
            rec["rules"] = [r for r in (keyed_rule(spec, k["id"])
                                        for k in kids) if r]
            rec["skills"] = [_outcome(k["id"]) for k in kids]
            out.append(rec)
    finally:
        (skill_pipeline.ask_model, skill_pipeline.fetch_source,
         skill_pipeline.MODEL_SCREEN, skill_pipeline.MODEL_VALIDATE,
         skill_pipeline.MODEL_TAG, skill_pipeline.MODEL_TESTS) = saved
    return out


def keyed_rule(spec: dict, sid: str) -> dict | None:
    """The OPERATOR'S RULE for a skill the pipeline filed (spec "rules"):
    where the classified applies-when differs from it, an edit -- a new
    version through the edit path (screen -> classify -> tests -> validate
    -> arm), inline -- keys the skill on it. The pipeline's own filing stays
    on version 1. Why: a decomposed or distilled skill is filed from its
    text, and the tag stage adds at most two terms; a skill that must hold
    only WITH another library (koota inside React Three Fiber) needs an
    all_of gate, which only a stated rule carries."""
    import skill_classify as C
    import skill_md
    import skill_pipeline
    import skills
    s = skills.get(sid) or {}
    want = (spec.get("rules") or {}).get(s.get("name") or "")
    if not want:
        return None
    v = skills.version(sid, s.get("latest_version") or 1) or {}
    if not v.get("text"):
        return {"name": s.get("name"), "edited": False,
                "why": "no validated text to edit"}
    have = C.metadata_of_rule(v.get("classify") or {})
    keys = ("artifacts", "languages", "frameworks", "phases", "situations",
            "all_of", "topics")
    target = C.metadata_of_rule(C.rule_from_metadata(want))
    if all((have.get(k) or []) == (target.get(k) or []) for k in keys):
        return {"name": s["name"], "edited": False, "why": "already so"}
    sk = skill_md.parse(v["text"])
    ours = dict(sk.get("yamadori") or {})
    ours["applies_when"] = dict(want)
    sk["yamadori"] = ours
    # The tests are the spec's own; version 1's GENERATED cases were derived
    # from the rule this edit replaces (skills.edit carries tests over only
    # while the rule is unchanged, for that reason).
    import skill_tests
    written = (spec.get("tests") or {}).get(s["name"]) or {}
    tests = skill_tests.from_model(written) if written else None
    s2 = skills.edit(sid, skill_md.render(sk), author="operator:offline",
                     tests=tests, enqueue_first=False)
    ver = s2["latest_version"]
    skill_pipeline.run_inline(sid, ver, model=False)
    return {"name": s["name"], "edited": True, "version": ver,
            "from": {k: have.get(k) for k in keys if have.get(k)},
            "to": {k: target.get(k) for k in keys if target.get(k)}}


def _licence_of_text(text: str | None) -> str | None:
    """The SKILL.md's `license:` (a child's and an edit's come from the
    source they were made from, not from a stage of their own)."""
    if not text:
        return None
    import skill_md
    return skill_md.parse(text).get("license")


def _outcome(sid: str) -> dict:
    import skill_classify as C
    import skills
    s = skills.get(sid) or {}
    v = skills.version(sid, s.get("latest_version") or 1) or {}
    rule = v.get("classify") or {}
    val = v.get("validate") or {}
    act = val.get("activation") or {}
    return {"id": sid, "name": s.get("name"), "status": s.get("status"),
            "reason": s.get("reason"), "applies_when": rule.get("text"),
            "category": C.category(rule), "topics": C.gates(rule)["topics"],
            "licence": _licence_of_text(v.get("text")),
            "items_kept": len(val.get("items") or []),
            "dropped": [d.get("why") for d in val.get("dropped") or []],
            "activation": {k: act.get(k) for k in ("passed", "score", "n",
                                                   "failures")}}


def retag(path: str) -> list[dict]:
    """Operator edits of armed skills' filing and trigger text (a retag
    spec: {"edits": {name: {frameworks?, languages?, topics_remove?,
    topics_add?, description?, title?}}}), each through the edit path
    inline (no queue, no model: the screen_model stage records its skip).
    The items are never touched. A skill whose edited version does not arm
    is RESTORED: its previous version serves again, and the report says
    why the edit failed."""
    import skill_md
    import skill_pipeline
    import skills
    with open(path, encoding="utf-8") as f:
        spec = json.load(f)
    os.environ["YAMADORI_SKILL_MODEL_SCREEN"] = "0"
    saved = skill_pipeline.MODEL_SCREEN
    skill_pipeline.MODEL_SCREEN = False
    out = []
    try:
        for name, ch in (spec.get("edits") or {}).items():
            s = skills.find(name)
            if s is None or s.get("status") != "armed":
                out.append({"name": name, "edited": False,
                            "why": "not armed" if s else "no such skill"})
                continue
            served = s["served_version"]
            v = skills.version(s["id"], served) or {}
            sk = skill_md.parse(v.get("text") or "")
            ours = dict(sk.get("yamadori") or {})
            aw = dict(ours.get("applies_when") or {})
            aw.pop("text", None)
            for k in ("frameworks", "languages"):
                if k in ch:
                    aw[k] = list(ch[k])
            topics = [t for t in aw.get("topics") or []
                      if t not in set(ch.get("topics_remove") or [])]
            topics += [t for t in ch.get("topics_add") or [] if t not in topics]
            aw["topics"] = topics
            ours["applies_when"] = aw
            sk["yamadori"] = ours
            if ch.get("description"):
                sk["description"] = ch["description"]
                sk["when"] = ch["description"]
            if ch.get("title"):
                sk["title"] = ch["title"]
            s2 = skills.edit(s["id"], skill_md.render(sk),
                             author=spec.get("author") or "operator:retag",
                             enqueue_first=False)
            nv = s2["latest_version"]
            skill_pipeline.run_inline(s["id"], nv, model=False)
            got = skills.get(s["id"]) or {}
            rec = {"name": name, "edited": True, "version": nv,
                   "status": got.get("status")}
            if got.get("status") != "armed":
                rec["why"] = got.get("reason")
                skills.reinstate(s["id"], served)
                rec["restored"] = served
                rec["status"] = (skills.get(s["id"]) or {}).get("status")
            out.append(rec)
    finally:
        skill_pipeline.MODEL_SCREEN = saved
    return out


def _live_paths() -> tuple[str, str]:
    import jobs
    import skills
    return os.path.abspath(jobs.DB), os.path.abspath(skills.STORE)


def _temp_store() -> str:
    import jobs
    import skills
    tmp = tempfile.mkdtemp(prefix="yamadori_skill_offline_")
    jobs.DB = os.path.join(tmp, "jobs.sqlite3")
    skills.STORE = os.path.join(tmp, "skills")
    skills._ENSURED.clear()
    skills._invalidate()
    return tmp


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("spec", nargs="?", default="")
    ap.add_argument("--retag", default="",
                    help="a retag spec: operator edits of armed skills")
    ap.add_argument("--copy-live", action="store_true",
                    help="with --temp: start from a copy of the live store")
    ap.add_argument("--temp", action="store_true",
                    help="a temp store, not the live one")
    ap.add_argument("--dry-run", action="store_true", help="= --temp")
    ap.add_argument("--fixtures", default="",
                    help="read the sources from DIR/<host>/<path>, offline")
    ap.add_argument("--report", default="")
    a = ap.parse_args(argv)
    os.environ.setdefault("YAMADORI_GPU_ROOM", "0")
    if a.temp or a.dry_run:
        live_db, live_store = _live_paths()
        tmp = _temp_store()
        if a.copy_live:
            import shutil
            import jobs
            import skills
            shutil.copyfile(live_db, jobs.DB)
            shutil.copytree(os.path.join(live_store, "library"),
                            os.path.join(skills.STORE, "library"))
            skills._ENSURED.clear()
            skills._invalidate()
        print(f"  temp store: {tmp}" + (" (a copy of the live one)"
                                        if a.copy_live else ""))
    if a.retag:
        got = retag(a.retag)
        bad = 0
        for r in got:
            print(f"  {r['name']}: " + (f"v{r['version']} {r['status']}"
                                       if r.get("edited") else "not edited")
                  + (f" -- {r['why']}" if r.get("why") else "")
                  + (f" (v{r['restored']} restored)" if r.get("restored")
                     else ""))
            bad += bool(r.get("why") and r.get("edited"))
        if a.report:
            with open(a.report, "w", encoding="utf-8") as f:
                json.dump(got, f, indent=1, default=str)
        print(f"  {sum(1 for r in got if r.get('edited'))} edited, {bad} "
              "restored after a failed edit")
        if not a.spec:
            return 1 if bad else 0
    spec = load_spec(a.spec)
    got = ingest(spec, fetcher=fixture_fetcher(a.fixtures) if a.fixtures
                 else None)
    bad = 0
    for r in got:
        print(f"\n  {r['name']} ({r['status']}): {r['source']}")
        if r.get("error"):
            print(f"    ERROR {r['error']}")
            bad += 1
        for stage, res in (r.get("stages") or {}).items():
            print(f"    {stage:12s} {json.dumps(res, default=str)[:160]}")
        for k in r["skills"]:
            print(f"    -> {k['name']}: {k['status']} | {k['applies_when']} | "
                  f"topics {k['topics']} | items {k['items_kept']} | "
                  f"activation {k['activation'].get('score')}"
                  + (f" | {k['reason']}" if k.get("reason") else ""))
            bad += k["status"] != "armed"
    if a.report:
        with open(a.report, "w", encoding="utf-8") as f:
            json.dump(got, f, indent=1, default=str)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
