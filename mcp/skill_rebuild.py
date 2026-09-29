#!/usr/bin/env python
"""Rebuild the skill library to the ASSURED VOICE, and retire what it
replaces only when the replacement is ready.

    python mcp/skill_rebuild.py --plan          what would be queued, and
                                                what each batch replaces
                                                (read-only)
    python mcp/skill_rebuild.py --queue         queue it for the worker:
                                                GPU work, idle-gated like an
                                                onboarding's (skills.enqueue)
    python mcp/skill_rebuild.py --settle        archive the replaced skills
                                                of every finished batch (the
                                                worker also settles after
                                                each skill job)
    --plan-file PATH                            default skills/replacements/
                                                2026-09-28-assured-voice.json

WHY (operator, 2026-09-28, after pagoda-h6): "those are all terrible skills
and plans don't sow doubt, I have been building with all of those APIs in
alpha state with none of those concerns." "All of our skills increase
confidence and improve correctness; if it can't, then the line doesn't need
to exist." And: "You can't just archive skills, you need to build new
skills, and have an auto review agent in a second pass ... Archive the
doubt-skills only when their replacements are ready to arm, or in the same
step."

THE PLAN FILE names three kinds of work:

  sources             NEW skills distilled from a pinned upstream page
                      (skills.create(url=...)): distil/6 -> review/1 ->
                      classify -> tests -> validate -> prove/1 -> arm.
                      Each carries meta.replacement {batch, replaces}.
  rerun               an existing source skill (a frontier SKILL.md, a
                      reference page) walks its path again from the stored
                      source as a NEW VERSION (the watch paths): the fixed
                      templates, the served version serving meanwhile.
  rebuild             every other armed skill with a line skill_limits.doubt
                      rejects gets a new version of ITSELF through review
                      (PATHS["rebuild"]): its items as the draft, its quotes
                      and source kept.

A CLEAN skill of a replaced file -- no line skill_limits.doubt rejects --
is not archived with its batch (operator, 2026-09-28: "archive each only
when its replacement has passed PROVE"): the plan's `replaces_when_proved`
maps it to the pages that replace it, and it is archived once EVERY one of
those pages has a replacement skill that is armed with a prove verdict of
`better` or `tie`. A clean skill the map does not list is kept.

SETTLING a batch (settle): when no skill of the batch has a version still
running and at least one armed, the doubt-bearing skills it replaces are
ARCHIVED
(skills.archive: nothing is deleted; `enable` brings one back), after their
SKILL.md folders are copied under index/_archive/, with a reason citing the
operator. A batch with nothing armed archives nothing. A rebuilt skill whose
new version kept no item is archived the same way; a rebuilt version that
failed for any other reason leaves the served one serving.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
PLAN_FILE = os.path.join(ROOT, "skills", "replacements",
                         "2026-09-28-assured-voice.json")
AUTHOR = "operator:assured-voice"
REASON = ("replaced by batch {batch} ({plan}): operator, 2026-09-28, "
          "\"All of our skills increase confidence and improve correctness; "
          "if it can't, then the line doesn't need to exist.\"")
NO_ITEM = "no item survived validation"


def load_plan(path: str | None = None) -> dict:
    with open(path or PLAN_FILE, encoding="utf-8") as f:
        return json.load(f)


def _meta(s: dict) -> dict:
    return s.get("meta") or {}


def doubt_lines(row: dict) -> list[dict]:
    """The lines of a served skill that skill_limits.doubt rejects, read
    from its STORED items (not the serving-time filtered ones)."""
    import skill_limits
    import skill_md
    sk = skill_md.parse(row.get("text") or "") if (row.get("text") or "") \
        .startswith("---") else {}
    out = []
    for it in sk.get("items") or row.get("items") or []:
        line = skill_md.item_line(it)
        why = skill_limits.doubt(line)
        if why:
            out.append({"item": line[:200], "why": why})
    return out


def _source_url(plan: dict, src: dict) -> str:
    pkg = src["pkg"]
    b = next(b for b in plan["batches"] if pkg in (b.get("provenance")
                                                    or {}))
    prov = b["provenance"][pkg]
    return plan["raw_url"][pkg].format(commit=prov["commit"],
                                       path=src["path"])


PROVED_VERDICTS = ("better", "tie")


def source_key(src: dict) -> str:
    return f"{src['pkg']}:{src['path']}"


def plan(plan_doc: dict | None = None) -> dict:
    """What --queue would do, read-only: the new sources per batch, the
    source skills re-run, the rebuilds, and what each batch replaces."""
    import skills
    doc = plan_doc or load_plan()
    rows = skills.listing()
    served = {r["id"]: r for r in skills.armed_unfiltered()}
    out = {"plan": doc["id"], "batches": []}
    replaced: set[str] = set()
    rerun_ids: set[str] = set()
    for b in doc["batches"]:
        if b["id"] == "rebuild":
            continue
        keep = set(b.get("keep") or [])
        mapped = {k: v for k, v in (b.get("replaces_when_proved")
                                    or {}).items() if not k.startswith("_")}
        rep, when_proved, kept_clean = [], [], []
        for r in rows:
            mg = _meta(r).get("migration_group") or {}
            if (mg.get("file") in (b.get("replaces_migration_files") or [])
                    and r["status"] == "armed" and r["name"] not in keep
                    and skills_md_name(r) not in keep):
                n = len(doubt_lines(served.get(r["id"]) or {}))
                x = {"id": r["id"], "name": r["name"], "doubt_lines": n}
                name = skills_md_name(r)
                if n:
                    rep.append(x)
                elif name in mapped:
                    when_proved.append(dict(x, pages=list(mapped[name])))
                else:
                    kept_clean.append(x)
        replaced |= {x["id"] for x in rep} | {x["id"] for x in when_proved}
        reruns = []
        for pre in b.get("rerun_source_urls_starting") or []:
            newest: dict[str, dict] = {}
            for r in rows:
                u = r.get("source_url") or ""
                if u.startswith(pre) and r["status"] in ("armed",
                                                         "decomposed"):
                    if u not in newest or r["updated"] > newest[u]["updated"]:
                        newest[u] = r
            for u, r in sorted(newest.items()):
                reruns.append({"id": r["id"], "name": r["name"],
                               "origin": ("watch_frontier"
                                          if r["source_kind"] == "frontier"
                                          else "watch"), "url": u})
        rerun_ids |= {x["id"] for x in reruns}
        out["batches"].append({
            "id": b["id"],
            "sources": [{"url": _source_url(doc, s), "goal": s["goal"],
                         "name": _source_name(s), "key": source_key(s)}
                        for s in b.get("sources") or []],
            "rerun": reruns, "replaces": rep, "when_proved": when_proved,
            "kept_clean": kept_clean})
    rebuild = []
    for sid, row in sorted(served.items()):
        if sid in replaced or sid in rerun_ids:
            continue
        # A child of a re-run source is superseded by its parent's new
        # decompose, not rebuilt on its own.
        par = (_meta(skills.get(sid) or {}).get("parent") or {}).get("skill")
        if par and par in rerun_ids:
            continue
        lines = doubt_lines(row)
        if lines:
            rebuild.append({"id": sid, "name": row["name"],
                            "doubt_lines": lines})
    out["batches"].append({"id": "rebuild", "rebuild": rebuild})
    return out


def skills_md_name(r: dict) -> str:
    return str(r.get("name") or "").replace("_", "-")


def _source_name(src: dict) -> str:
    stem = src["path"].rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return f"{src['pkg']}-{stem}"


def rebuild_version(sid: str, *, batch: str = "rebuild",
                    enqueue: bool = True) -> int:
    """A new version of skill `sid` through review: its served items (with
    their quotes) as the draft, its rule and tests carried, its stored
    source the one the quotes verify against. The served version keeps
    serving until this one arms."""
    import skills
    import skill_md
    s = skills.get(sid)
    if not s or not s.get("served_version"):
        raise ValueError(f"{sid} has no served version to rebuild")
    ver = skills.version(sid, s["served_version"]) or {}
    sk = skill_md.parse(ver.get("text") or "") if (ver.get("text") or "") \
        .startswith("---") else {}
    items = ((ver.get("validate") or {}).get("items")
             or sk.get("items") or [])
    draft = {"name": sk.get("name") or s["name"],
             "title": (ver.get("validate") or {}).get("title")
             or sk.get("title") or s.get("title") or "",
             "description": sk.get("description") or "",
             "items": [dict(it, quote=it.get("quote") or "") for it in items]}
    v = skills.new_version(
        sid, "rebuild", author=AUTHOR,
        meta={"rebuild": {"batch": batch, "of": s["served_version"],
                          "doubt_lines": doubt_lines(
                              {"text": ver.get("text"), "items": items})}},
        distil={"compiled": True, "skill": draft,
                "rebuild_of": s["served_version"]},
        classify=ver.get("classify") or {}, tests=ver.get("tests") or {})
    skills.update_meta(sid, rebuild={"batch": batch, "version": v,
                                     "at": time.time()})
    if enqueue:
        skills.enqueue(sid, v, skills.PATHS["rebuild"][0])
    return v


def rerun_source(sid: str, origin: str, *, batch: str,
                 enqueue: bool = True) -> int:
    """A source skill walks its path again from its stored source as a new
    version (the watch paths), through the fixed templates."""
    import skills
    v = skills.new_version(sid, origin, author=AUTHOR,
                           meta={"rebuild": {"batch": batch,
                                             "rerun": True}})
    skills.update_meta(sid, rebuild={"batch": batch, "version": v,
                                     "at": time.time()})
    if enqueue:
        skills.enqueue(sid, v, skills.PATHS[origin][0])
    return v


def queue(plan_doc: dict | None = None, *, enqueue: bool = True) -> dict:
    """Create and queue everything plan() lists. Returns what was made."""
    import skills
    doc = plan_doc or load_plan()
    p = plan(doc)
    made = {"plan": doc["id"], "created": [], "rerun": [], "rebuilt": []}
    for b in p["batches"]:
        if b["id"] == "rebuild":
            for r in b["rebuild"]:
                made["rebuilt"].append({"id": r["id"], "version":
                                        rebuild_version(r["id"],
                                                        enqueue=enqueue)})
            continue
        rep = [x["id"] for x in b["replaces"]]
        wp = {x["id"]: x["pages"] for x in b.get("when_proved") or []}
        for src in b["sources"]:
            s = skills.create(url=src["url"], name=src["name"],
                              goal=src["goal"], frontier=False,
                              author=AUTHOR, watch_hours=0,
                              meta={"replacement": {"batch": b["id"],
                                                    "plan": doc["id"],
                                                    "replaces": rep,
                                                    "source": src["key"],
                                                    "when_proved": wp},
                                    "rebuild": {"batch": b["id"]}},
                              enqueue_first=enqueue)
            made["created"].append({"id": s["id"], "name": s["name"],
                                    "batch": b["id"]})
        for r in b["rerun"]:
            made["rerun"].append({"id": r["id"], "version": rerun_source(
                r["id"], r["origin"], batch=b["id"], enqueue=enqueue)})
    return made


def _backup(ids: list[str], label: str) -> str:
    import skills
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dest = os.path.join(os.path.dirname(os.path.abspath(skills.STORE)),
                        "_archive", f"skills-before-{label}-{stamp}")
    os.makedirs(dest, exist_ok=True)
    lib = skills.library_dir()
    for sid in ids:
        s = skills.get(sid) or {}
        for nm in {skills_md_name(s), str(s.get("name") or "")}:
            src = os.path.join(lib, nm)
            if nm and os.path.isdir(src):
                shutil.copytree(src, os.path.join(dest, nm),
                                dirs_exist_ok=True)
    with open(os.path.join(dest, "archived.json"), "w",
              encoding="utf-8") as f:
        json.dump({"ids": ids, "label": label, "at": time.time()}, f,
                  indent=1)
    return dest


def settle(batch: str | None = None) -> dict:
    """Archive what each FINISHED batch replaces (see the module docstring).
    Idempotent: an archived skill is skipped."""
    import skills
    rows = skills.listing()
    by_batch: dict[str, dict] = {}
    for r in rows:
        rp = _meta(r).get("replacement") or {}
        if not rp.get("batch") or (batch and rp["batch"] != batch):
            continue
        b = by_batch.setdefault(rp["batch"], {"members": [], "replaces":
                                              set(), "plan": rp.get("plan"),
                                              "when_proved": {}})
        b["members"].append(r)
        b["replaces"] |= set(rp.get("replaces") or [])
        b["when_proved"].update(rp.get("when_proved") or {})
    # A decomposed source's children are members of its batch too.
    for r in rows:
        par = (_meta(r).get("parent") or {}).get("skill")
        for b in by_batch.values():
            if par and any(m["id"] == par for m in b["members"]):
                b["members"].append(r)
    out = {"archived": [], "waiting": [], "nothing_armed": [],
           "clean_waiting": []}
    for name, b in sorted(by_batch.items()):
        # THE CLEAN PACKS, one by one: every page proved.
        proved = set()
        for m in b["members"]:
            src = (_meta(m).get("replacement") or {}).get("source")
            if src and m["status"] == "armed" and _proved(m):
                proved.add(src)
        ready = []
        for sid, pages in sorted(b["when_proved"].items()):
            if (skills.get(sid) or {}).get("status") in ("archived", None):
                continue
            missing = [p for p in pages if p not in proved]
            if missing:
                out["clean_waiting"].append({"id": sid, "missing": missing})
            else:
                ready.append(sid)
        if ready:
            where = _backup(ready, f"assured-voice-{name}-proved")
            for sid in ready:
                skills.archive(sid, reason=REASON.format(
                    batch=name, plan=b.get("plan") or "")
                    + " Every page that replaces it passed PROVE. backup: "
                    + where, author=AUTHOR)
            out["archived"].append({"batch": name, "ids": ready,
                                    "backup": where, "why": "proved"})
        running = [m["id"] for m in b["members"] if m["status"] == "pipeline"]
        armed = [m["id"] for m in b["members"] if m["status"] == "armed"]
        if running:
            out["waiting"].append({"batch": name, "running": len(running)})
            continue
        if not armed:
            out["nothing_armed"].append(name)
            continue
        todo = [sid for sid in sorted(b["replaces"])
                if (skills.get(sid) or {}).get("status") not in
                ("archived", None)]
        if not todo:
            continue
        where = _backup(todo, f"assured-voice-{name}")
        for sid in todo:
            skills.archive(sid, reason=REASON.format(
                batch=name, plan=b.get("plan") or "") + f" backup: {where}",
                author=AUTHOR)
        out["archived"].append({"batch": name, "ids": todo,
                                "backup": where, "armed": len(armed)})
    # A rebuilt skill whose new version kept no item.
    empty = []
    for r in rows:
        rb = _meta(r).get("rebuild") or {}
        if not rb.get("version") or rb.get("settled") \
                or r["status"] == "archived":
            continue
        ver = skills.version(r["id"], rb["version"]) or {}
        if ver.get("state") == "failed" and NO_ITEM in (ver.get("reason")
                                                        or ""):
            empty.append(r["id"])
    if empty:
        where = _backup(empty, "assured-voice-rebuild-empty")
        for sid in empty:
            skills.archive(sid, reason="the rebuild through review/1 kept no "
                           "item: nothing in it could be acted on (operator, "
                           "2026-09-28). backup: " + where, author=AUTHOR)
            skills.update_meta(sid, rebuild=dict(
                _meta(skills.get(sid) or {}).get("rebuild") or {},
                settled="archived: no item"))
        out["archived"].append({"batch": "rebuild", "ids": empty,
                                "backup": where})
    return out


def _proved(s: dict) -> bool:
    """The served version passed PROVE (skill_prove: better or tie)."""
    import skill_prove
    import skills
    v = s.get("served_version")
    ver = skills.version(s["id"], v) if v else None
    return (skill_prove.record_of(ver).get("verdict") in PROVED_VERDICTS
            if ver else False)


def settle_after(job: dict) -> dict | None:
    """The worker's hook after a skill job: settle the job's batch when the
    skill is one of a batch's (cheap otherwise: one skills.get)."""
    import skills
    sid = (job.get("payload") or {}).get("skill")
    s = skills.get(sid) if sid else None
    if not s:
        return None
    m = _meta(s)
    par = (m.get("parent") or {}).get("skill")
    if par and not m.get("replacement"):
        m = _meta(skills.get(par) or {})
    if not (m.get("replacement") or m.get("rebuild")):
        return None
    return settle((m.get("replacement") or {}).get("batch"))


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--plan", action="store_true")
    g.add_argument("--queue", action="store_true")
    g.add_argument("--settle", action="store_true")
    ap.add_argument("--plan-file", default=PLAN_FILE)
    a = ap.parse_args()
    doc = load_plan(a.plan_file)
    if a.plan:
        p = plan(doc)
        for b in p["batches"]:
            if b["id"] == "rebuild":
                print(f"rebuild: {len(b['rebuild'])} skills through review")
                for r in b["rebuild"]:
                    print(f"  {r['name']}: {len(r['doubt_lines'])} line(s)")
                continue
            print(f"{b['id']}: {len(b['sources'])} new sources, "
                  f"{len(b['rerun'])} re-run, replaces "
                  f"{len(b['replaces'])} on settle, "
                  f"{len(b['when_proved'])} clean pack(s) each when its "
                  f"pages pass PROVE, keeps {len(b['kept_clean'])} clean")
            for x in b["replaces"]:
                print(f"  replaces {x['name']} ({x['doubt_lines']} doubt "
                      "line(s))")
            for x in b["when_proved"]:
                print(f"  replaces when proved {x['name']} <- "
                      + ", ".join(x["pages"]))
            for x in b["kept_clean"]:
                print(f"  keeps {x['name']} (clean, no replacement mapped)")
        return 0
    if a.queue:
        print(json.dumps(queue(doc), indent=1))
        return 0
    print(json.dumps(settle(), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
