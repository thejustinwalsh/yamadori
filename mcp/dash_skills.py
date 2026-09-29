#!/usr/bin/env python
"""NAEDOKO's skill factory: the API behind the dashboard. The contract, with
request and response examples, is docs/SKILL-FACTORY.md; mcp/
test_skill_factory.py holds it to that.

Every path here is under /dash/api, which `server.py` gates with
`accounts.identify` before dispatching -- an unauthenticated request never
reaches this module. The caller's account id (a hash prefix, never the key)
is recorded as the author of a submission, an edit or a state change.

    GET  /dash/api/skills                    every skill (summary), counts by
                                             status, the taxonomy with counts,
                                             stages and paths, limits,
                                             thresholds, prompt templates,
                                             learning, the queue
    GET  /dash/api/skills/<id>               one skill in full: SKILL.md,
                                             provenance and licence quote,
                                             versions with every stage's
                                             output, jobs, tests and their
                                             last result, recent selections,
                                             children of a decomposed source
    GET  /dash/api/skills/<id>/skill.md      the SKILL.md as text/markdown
    GET  /dash/api/skill-factory/prompts     the templates, with their text
    GET  /dash/api/skill-factory/selections  recent selections, all skills
    GET  /dash/api/skill-factory/recent      the last requests' x_yamadori.
                                             skills (in memory), with the
                                             per-turn caps
    (every skill row, summary and detail, carries `size`: the injected
    body's tokens, items, prohibitions against skill_limits)
    POST /dash/api/skill                     submit: {url} | {text} | {urls:
                                             [...]}, optional name, goal,
                                             frontier, watch_hours
    POST /dash/api/skill/edit                {id, text, tests?}: a new
                                             version, re-screened and
                                             re-tested before it re-arms
    POST /dash/api/skill/tests               {id, tests}: new tests, as an
                                             edit of the current text
    POST /dash/api/skill/activation          {id}: run the activation tests
                                             now against the armed pool; no
                                             state change
    POST /dash/api/skill/disable|enable|archive|quarantine  {id, reason?}
    POST /dash/api/skill/rerun               {id, stage}
    POST /dash/api/skill/licence             {id, licence, quote}
    POST /dash/api/skill/watch               {id, hours}: 0 or null stops
    POST /dash/api/skill/refetch             {id}: enqueue a watch now

  PACKAGE ONBOARDING (mcp/onboarding.py; docs/PACKAGE-ONBOARDING.md 8.1):
    POST /dash/api/skill                     {prompt, links?, aliases?,
                                             replaces?: replace|alongside}:
                                             a PROMPT WITH LINKS -> one
                                             onboarding (a dataset of kind
                                             package); {ok, onboarding: {id,
                                             stage, packages: []}}
    GET  /dash/api/skill-factory/onboarding  every onboarding: {id, group,
                                             package, version, stage, state,
                                             updated, counts}
    GET  /dash/api/skill-factory/onboarding/<id>  one in full: job rows,
                                             blockers, warnings, resolution,
                                             licence, index, vocab, examples,
                                             knn, skills, retire, eval, reviews
    POST /dash/api/skill-factory/onboarding/review   {id, stage, note}
    POST /dash/api/skill-factory/onboarding/promote  {id}: force a HELD
                                             vocabulary, recorded with the
                                             author
    POST /dash/api/skill-factory/onboarding/tier3    {id}: ingest the
                                             recorded llms.txt page list
    (the licence answer and a re-run of an errored stage job are the
    datasets API's: POST /dash/api/dataset/answer, /dash/api/dataset/rerun)

Nothing here runs a model: every write records a row and enqueues jobs for
`mcp/worker.py`, exactly as datasets.py does. No response carries a
filesystem path (skills.public). Errors are {ok: false, error} with 400
(bad input) or 404 (no such skill).
"""
from __future__ import annotations

import json
import re
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jobs  # noqa: E402
import skill_classify  # noqa: E402
import skill_learn  # noqa: E402
import skill_limits  # noqa: E402
import skill_select  # noqa: E402
import skills  # noqa: E402

POSTS = ("/dash/api/skill", "/dash/api/skill/edit", "/dash/api/skill/tests",
         "/dash/api/skill/activation", "/dash/api/skill/disable",
         "/dash/api/skill/enable", "/dash/api/skill/archive",
         "/dash/api/skill/quarantine", "/dash/api/skill/rerun",
         "/dash/api/skill/licence", "/dash/api/skill/watch",
         "/dash/api/skill/refetch",
         "/dash/api/skill-factory/onboarding/review",
         "/dash/api/skill-factory/onboarding/promote",
         "/dash/api/skill-factory/onboarding/tier3")
MAX_BATCH = 25


def _json(code: int, payload: dict):
    return code, "application/json", json.dumps(payload, default=str).encode()


def taxonomy_counts(rows: list[dict]) -> dict:
    """For each axis value, how many skills (all states) carry it."""
    out: dict = {}
    for s in rows:
        for axis, vals in (s.get("category") or {}).items():
            for v in vals or []:
                out.setdefault(axis, {}).setdefault(v, 0)
                out[axis][v] += 1
    return out


def size_of(text: str | None = None, *, body: str | None = None,
            items: list | None = None, description: str | None = None
            ) -> dict | None:
    """What a skill costs against the caps (skill_limits): the INJECTED body
    (title + items, never the frontmatter), its items, its prohibitions,
    its longest item and its description. From a served row's parts, or
    parsed from a SKILL.md; None when there is no text yet."""
    import skill_md
    if body is None:
        if not text:
            return None
        sk = skill_md.parse(text) if text.startswith("---") else {}
        body = skill_md.injection(sk) if sk else text
        items = sk.get("items") or []
        description = sk.get("description") or ""
    items = [it for it in items or [] if isinstance(it, dict)]
    return {"tokens": skill_limits.tokens(body), "chars": len(body),
            "items": len(items),
            "prohibitions": sum(1 for it in items
                                if skill_limits.is_prohibition(it)),
            "longest_item_chars": max(
                (len(str(it.get("text") or "")) + len(str(
                    it.get("situation") or "")) for it in items), default=0),
            "description_chars": len(description or "")}


def _sizes(rows: list[dict]) -> None:
    """Attach `size` to each summary row: from the served cache for an
    armed skill, else parsed from its newest text (a few rows)."""
    served = {r["id"]: r for r in skills.armed()}
    for row in rows:
        r = served.get(row["id"])
        if r is not None and r.get("version") == row.get("text_version"):
            row["size"] = size_of(body=r.get("body") or "",
                                  items=r.get("items"),
                                  description=r.get("description"))
        else:
            tv = row.get("text_version")
            ver = (skills.version(row["id"], tv) if tv else None) or {}
            row["size"] = size_of(ver.get("text"))


def overview() -> dict:
    import skill_prompts
    rows = [skills.public(s) for s in skills.listing()]
    _sizes(rows)
    return {
        "skills": rows,
        "counts": skills.counts(),
        # Kept for the dashboard's type: skills are the one knowledge
        # system (2026-09-26), so this is always "skills".
        "recall": "skills",
        "skip_classes": list(skill_select.SKIP_CLASSES),
        "stages": list(skills.STAGES),
        "paths": {k: list(v) for k, v in skills.PATHS.items()},
        "taxonomy": skill_classify.taxonomy(),
        "taxonomy_counts": taxonomy_counts(rows),
        "limits": skill_limits.summary(),
        # No cosine thresholds since 2026-09-27: a cosine ranks inside a
        # question only (docs/CONSTANTS-AUDIT.md).
        "thresholds": {},
        "prompts": [{k: v for k, v in r.items() if k != "text"}
                    for r in skill_prompts.registry()],
        # The fallback rate per day, pending records, and whether the stack
        # is idle enough to learn from them (skill_learn).
        "learning": skill_learn.overview(),
        "queue": {k: v for k, v in jobs.snapshot().items() if k != "db"},
    }


def handle_get(path: str):
    """(status, content_type, body) or None if this path is not ours."""
    p = path.rstrip("/")
    if p == "/dash/api/skills":
        try:
            return _json(200, overview())
        except Exception as e:                                   # noqa: BLE001
            return _json(500, {"error": f"skills overview raised "
                                        f"{type(e).__name__}: {e}"})
    if p == "/dash/api/skill-factory/prompts":
        import skill_prompts
        return _json(200, {"prompts": skill_prompts.registry()})
    if p == "/dash/api/skill-factory/selections":
        return _json(200, {"selections": skill_learn.selections(limit=200)})
    if p == "/dash/api/skill-factory/recent":
        # The last requests' x_yamadori.skills, kept in this process's
        # memory (recent_turns: gone on a restart), with the per-turn caps.
        import recent_turns
        return _json(200, {"requests": recent_turns.skills_recent(),
                           "keep": recent_turns.MAX_TURNS,
                           "limits": skill_limits.summary()})
    if p == "/dash/api/skill-factory/onboarding":
        import onboarding
        return _json(200, {"onboardings": onboarding.listing()})
    if p.startswith("/dash/api/skill-factory/onboarding/"):
        import onboarding
        did = p[len("/dash/api/skill-factory/onboarding/"):]
        d = onboarding.detail(did)
        if d is None:
            return _json(404, {"error": f"no such onboarding: {did}"})
        return _json(200, {"onboarding": d})
    if p.startswith("/dash/api/skills/"):
        rest = p[len("/dash/api/skills/"):]
        sid, _, sub = rest.partition("/")
        s = skills.get(sid)
        if s is None:
            return _json(404, {"error": f"no such skill: {sid}"})
        if sub == "skill.md":
            pub = skills.public(s, detail=True)
            if not pub.get("skill_md"):
                return _json(404, {"error": f"{sid} has no SKILL.md yet "
                                            f"({s['status']})"})
            return 200, "text/markdown; charset=utf-8", \
                pub["skill_md"].encode("utf-8")
        if sub:
            return _json(404, {"error": f"no such view: {sub}"})
        pub = skills.public(s, detail=True)
        pub["size"] = size_of(pub.get("skill_md"))
        return _json(200, {"skill": pub})
    return None


def _as_list(v) -> list[str]:
    if isinstance(v, str):
        v = re.split(r"[,\n]", v)
    return [str(x).strip() for x in (v or []) if str(x).strip()]


def _submit(body: dict, author: str) -> dict:
    if isinstance(body.get("prompt"), str) and body["prompt"].strip():
        # A PROMPT WITH LINKS: a package onboarding (mcp/onboarding.py).
        import onboarding
        ds = onboarding.submit(body["prompt"], links=_as_list(
            body.get("links")), aliases=_as_list(body.get("aliases")),
            replaces=body.get("replaces") or None, author=author)
        return {"ok": True, "onboarding": {"id": ds["id"],
                                           "stage": ds["stage"],
                                           "packages": []}}
    goal = str(body.get("goal") or "")
    frontier = body.get("frontier")
    frontier = bool(frontier) if frontier is not None else None
    urls = body.get("urls")
    if isinstance(urls, list):
        if not urls or len(urls) > MAX_BATCH:
            raise ValueError(f"urls: give 1 to {MAX_BATCH} URLs")
        made = [skills.create(url=str(u), author=author, goal=goal,
                              frontier=frontier,
                              watch_hours=body.get("watch_hours"))
                for u in urls]
        return {"ok": True, "skills": [skills.public(s) for s in made]}
    url = str(body.get("url") or "").strip() or None
    text = body.get("text")
    text = str(text) if isinstance(text, str) and text.strip() else None
    s = skills.create(url=url, text=text, name=body.get("name"),
                      author=author, goal=goal, frontier=frontier,
                      watch_hours=body.get("watch_hours"))
    return {"ok": True, "skill": skills.public(s)}


def _activation(sid: str) -> dict:
    import skill_tests
    s = skills.get(sid)
    if s is None:
        raise KeyError(f"no such skill: {sid}")
    ver = (skills.version(sid, s.get("served_version"))
           if s.get("served_version") else None) or skills.version(
        sid, s["latest_version"]) or {}
    rule = ver.get("classify") or {}
    pool = [x for x in skills.armed() if x["id"] != sid] + [
        {"id": sid, "rule": rule}]
    return skill_tests.run(ver.get("tests") or {}, rule, skill_id=sid,
                           pool=pool)


def handle_post(path: str, body: dict, who: str = "operator"):
    p = path.rstrip("/")
    if p not in POSTS:
        return None
    body = body or {}
    author = f"operator:{who}" if who else "operator"
    try:
        if p == "/dash/api/skill":
            return _json(200, _submit(body, author))
        if p.startswith("/dash/api/skill-factory/onboarding/"):
            import onboarding
            did = str(body.get("id") or "")
            if onboarding.detail(did) is None:
                raise KeyError(f"no such onboarding: {did}")
            if p.endswith("/review"):
                got = onboarding.review(did, str(body.get("stage") or ""),
                                        str(body.get("note") or ""),
                                        author=author)
            elif p.endswith("/promote"):
                got = {"vocab": onboarding.promote(did, author=author)}
            else:
                got = onboarding.tier3(did, author=author)
            return _json(200, dict(got, ok=True, id=did))
        sid = str(body.get("id") or "")
        if p == "/dash/api/skill/activation":
            return _json(200, {"ok": True, "id": sid,
                               "activation": _activation(sid)})
        reason = str(body.get("reason") or "")
        if p == "/dash/api/skill/edit":
            tests = body.get("tests") if isinstance(body.get("tests"),
                                                    dict) else None
            s = skills.edit(sid, str(body.get("text") or ""), author=author,
                            tests=tests)
        elif p == "/dash/api/skill/tests":
            tests = body.get("tests")
            if not isinstance(tests, dict) or not isinstance(
                    tests.get("activation"), dict):
                raise ValueError("tests must be {activation: {should: [...],"
                                 " should_not: [...]}, behaviour: [...]}")
            cur = skills.get(sid)
            if cur is None:
                raise KeyError(f"no such skill: {sid}")
            ver = skills.latest_with_text(sid)
            if not ver:
                raise ValueError("the skill has no text yet to re-test")
            s = skills.edit(sid, ver["text"], author=author, tests=tests)
        elif p == "/dash/api/skill/disable":
            s = skills.disable(sid, reason=reason, author=author)
        elif p == "/dash/api/skill/enable":
            s = skills.enable(sid, author=author)
        elif p == "/dash/api/skill/archive":
            s = skills.archive(sid, reason=reason, author=author)
        elif p == "/dash/api/skill/quarantine":
            s = skills.quarantine_skill(sid, reason=reason, author=author)
        elif p == "/dash/api/skill/rerun":
            s = skills.rerun(sid, str(body.get("stage") or ""),
                             author=author)
        elif p == "/dash/api/skill/licence":
            s = skills.set_licence(sid, str(body.get("licence") or ""),
                                   str(body.get("quote") or ""),
                                   author=author)
        elif p == "/dash/api/skill/watch":
            h = body.get("hours")
            s = skills.set_watch(sid, float(h) if h not in (None, "") else None)
        else:
            s = skills.get(sid)
            if s is None:
                raise KeyError(f"no such skill: {sid}")
            if not s.get("source_url"):
                raise ValueError("only a skill with a source URL can be "
                                 "re-fetched")
            jid = jobs.add(skills.WATCH[0], {"skill": sid},
                           lane=skills.WATCH[1], dataset=f"skill:{sid}",
                           stage="watch")
            return _json(200, {"ok": True, "job": jid,
                               "skill": skills.public(s)})
        return _json(200, {"ok": True, "skill": skills.public(s)})
    except KeyError as e:
        return _json(404, {"ok": False, "error": str(e).strip("'\"")})
    except Exception as e:                                       # noqa: BLE001
        return _json(400, {"ok": False, "error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    o = overview()
    print(f"  {len(o['skills'])} skill(s); {o['counts']}")
