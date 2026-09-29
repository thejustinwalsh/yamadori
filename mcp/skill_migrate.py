#!/usr/bin/env python
"""Retire the hints corpus into skills: the migration, and the authored
skills' install.

    python mcp/skill_migrate.py --dry-run     compile, screen, test and
                                              validate every group in a TEMP
                                              store; print the counts; write
                                              nothing live
    python mcp/skill_migrate.py --apply       the same into the live store,
                                              after backing it up and
                                              archiving the hints corpus
                                              under index/_archive/<stamp>/
    python mcp/skill_migrate.py --authored    install skills/authored/* (the
                                              skills written from our own
                                              evidence) into the live store
    python mcp/skill_migrate.py --ingest DIR  queue every SKILL.md folder
                                              under DIR (a local checkout of
                                              a published skills collection)
                                              on the FRONTIER path: the worker
                                              screens, licenses and
                                              decomposes each into atomic
                                              skills (model stages)
    --report PATH                             the counts and every skill's
                                              outcome as JSON

WHAT GOES IN (operator, 2026-09-25/26: "we are refactoring hints to become
skills"; this overrides Phase 0.7's "nothing deleted before paired runs":
the V0 baseline v0e-V0-xhigh-1 had zero hints injected across 106+ requests,
so skills are measured against a clean no-knowledge baseline instead)

Every `bench/recipes/*.jsonl` row hints SERVED: anything a person did not
REJECT (review is optional and after the fact; an unreviewed row was live).
A row the domain benchmark's leak review marked as a LEAK
(bench/domain/leak_review.json: the recipe carries a task's answer) is
excluded. The rows are compiled into atomic skills by skill_compile (group,
pack under the caps, tag, generate activation tests), and each skill walks
the one pipeline inline -- screen -> classify -> tests -> validate -> arm --
with no model call: the rows are already written and reviewed-or-served.
A skill whose screen or activation tests fail is QUARANTINED with its
reason; nothing is dropped silently.

REPRODUCIBLE: a skill's id is sha1("migration:" + name)[:12] and nothing it
writes carries a timestamp, so the same corpus makes the same store; the
store's fingerprint (skills.manifest) is written next to it and recorded in
models/manifest.yaml.

THE HINTS CORPUS IS NEVER DELETED. bench/recipes/ stays where it is (it is
the migration's source and the domain benchmark's leak preflight reads it);
--apply copies it, bench/hint_buckets.jsonl and bench/hint_probes.jsonl into
index/_archive/hints-<stamp>/ and MOVES index/hints.npz there (a derived
cache nothing reads any more).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
CORPUS = os.path.join(ROOT, "bench", "recipes")
LEAK_REVIEW = os.path.join(ROOT, "bench", "domain", "leak_review.json")
AUTHORED = os.path.join(ROOT, "skills", "authored")
ARCHIVE = os.path.join(ROOT, "index", "_archive")
REJECTED = "reject"


def row_hash(r: dict) -> str:
    import skill_compile
    return skill_compile.row_hash(r)


def leaks(path: str | None = None) -> set[str]:
    try:
        with open(path or LEAK_REVIEW, encoding="utf-8") as f:
            review = json.load(f)
    except (OSError, ValueError):
        return set()
    return {k.split("|", 1)[1] for k, v in review.items()
            if isinstance(v, dict) and v.get("verdict") == "leak" and "|" in k}


def load(corpus: str = CORPUS) -> tuple[list[dict], dict]:
    """(rows to migrate, counts). Each row gains `_file`, `_line`, `_hash`."""
    bad = leaks()
    rows, counts = [], {"files": 0, "rows": 0, "keep": 0, "edited": 0,
                        "unreviewed": 0, "reject": 0, "leak_excluded": 0,
                        "unparseable": 0, "empty": 0}
    for path in sorted(glob.glob(os.path.join(corpus, "*.jsonl"))):
        counts["files"] += 1
        stem = os.path.basename(path)[:-len(".jsonl")]
        with open(path, encoding="utf-8") as fh:
            for i, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                counts["rows"] += 1
                try:
                    r = json.loads(line)
                except ValueError:
                    counts["unparseable"] += 1
                    continue
                st = r.get("_state") or "unreviewed"
                if st == REJECTED:
                    counts["reject"] += 1
                    continue
                counts[st if st in counts else "unreviewed"] += 1
                if not str(r.get("recipe") or "").strip():
                    counts["empty"] += 1
                    continue
                h = row_hash(r)
                if h in bad:
                    counts["leak_excluded"] += 1
                    continue
                r.update(_file=stem, _line=i, _hash=h)
                rows.append(r)
    return rows, counts


def _outcomes(ids: list[str]) -> dict:
    import skills
    out = {"skills": len(ids), "armed": 0, "quarantined": 0, "failed": 0,
           "pipeline": 0, "items_armed": 0, "items_dropped": 0,
           "quarantine_reasons": {}, "fail_reasons": {}}
    for sid in ids:
        s = skills.get(sid) or {}
        st = s.get("status") or "pipeline"
        out[st] = out.get(st, 0) + 1
        ver = skills.version(sid, s.get("latest_version") or 1) or {}
        val = ver.get("validate") or {}
        out["items_dropped"] += len(val.get("dropped") or [])
        if st == "armed":
            out["items_armed"] += len(val.get("items") or [])
        reason = str(s.get("reason") or "")
        key = reason.split(":", 1)[0][:40] if reason else ""
        if st == "quarantined":
            out["quarantine_reasons"][key] = out["quarantine_reasons"].get(
                key, 0) + 1
        elif st == "failed":
            out["fail_reasons"][key] = out["fail_reasons"].get(key, 0) + 1
    return out


def migrate(rows: list[dict], *, lookup=None) -> tuple[list[str], dict]:
    """Compile and run every group into the CURRENT store (jobs.DB,
    skills.STORE). Returns (ids, per-skill outcomes)."""
    import skill_compile
    import skills
    groups, gcounts = skill_compile.group(rows)
    ids, skipped = [], 0
    per: list[dict] = []
    for g in groups:
        c = skill_compile.compile_group(g, licence_lookup=lookup)
        if skills.get(c["sid"]) is not None:
            skipped += 1
            ids.append(c["sid"])
            continue
        skills.create_compiled(skill=c["skill"], rule=c["rule"],
                               tests=c["tests"], source=c["source"],
                               origin="migration",
                               author="pipeline:migration", meta=c["meta"],
                               sid=c["sid"], run=True)
        ids.append(c["sid"])
        s = skills.get(c["sid"]) or {}
        per.append({"id": c["sid"], "name": s.get("name"),
                    "status": s.get("status"),
                    "reason": (s.get("reason") or "")[:200],
                    "items": len(c["skill"]["items"])})
    gcounts["skipped_existing"] = skipped
    return ids, {"groups": gcounts, "skills": per}


def install_authored(root: str = AUTHORED) -> list[dict]:
    """Install every folder under skills/authored/ (idempotent by name: a
    skill already in the store is left alone; edit it instead)."""
    import skill_classify
    import skill_md
    import skills
    out = []
    for folder in skill_md.folders(root):
        sk, tests = skill_md.read_folder(folder)
        with open(os.path.join(folder, "SKILL.md"), encoding="utf-8") as f:
            text = f.read()
        ours = sk.get("yamadori") or {}
        rule = skill_classify.rule_from_metadata(ours.get("applies_when")
                                                 or {})
        rule = skill_classify.with_gates(
            rule, phases=skill_classify.gates(rule)["phases"],
            situations=skill_classify.gates(rule)["situations"],
            all_of=skill_classify.gates(rule)["all_of"],
            topics=skill_classify.gates(rule)["topics"],
            description=sk["description"])
        sid = hashlib.sha1(f"authored:{sk['name']}".encode()).hexdigest()[:12]
        if skills.get(sid) is not None:
            s = skills.get(sid)
            out.append({"id": sid, "name": s["name"], "status": s["status"],
                        "skipped": "already installed"})
            continue
        skill = {"name": sk["name"], "title": sk["title"],
                 "description": sk["description"],
                 "items": [dict(it, quote=it.get("quote") or "")
                           for it in sk["items"]]}
        skills.create_compiled(
            skill=skill, rule=rule, tests=tests, source=text,
            origin="authored", author="operator:authored",
            meta={"provenance": ours.get("provenance") or {},
                  "authored_from": os.path.relpath(folder, ROOT)
                  .replace("\\", "/")}, sid=sid, run=True)
        s = skills.get(sid)
        out.append({"id": sid, "name": s["name"], "status": s["status"],
                    "reason": s.get("reason")})
    return out


def ingest_folder(root: str, *, goal: str = "") -> list[dict]:
    """Queue every SKILL.md folder under `root` on the frontier path. Its
    scripts/, references/ and assets/ are never read: scripts are data, and
    a skill is its SKILL.md. The path is recorded as provenance."""
    import skill_md
    import skills
    out = []
    for folder in skill_md.folders(root):
        with open(os.path.join(folder, "SKILL.md"), encoding="utf-8") as f:
            text = f.read()
        rel = os.path.relpath(folder, root).replace("\\", "/")
        s = skills.create(text=text, name=os.path.basename(folder),
                          author="operator:ingest", goal=goal,
                          frontier=True,
                          meta={"provenance": {"repo": os.path.basename(
                              os.path.abspath(root)), "path": rel}})
        out.append({"id": s["id"], "name": s["name"], "path": rel})
    return out


def _stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def backup_and_archive(stamp: str) -> dict:
    """Back up the live store and archive the hints corpus. Never deletes a
    corpus source file."""
    import jobs
    import skills
    dest = os.path.join(ARCHIVE, f"hints-{stamp}")
    os.makedirs(dest, exist_ok=True)
    done = {"archive": os.path.relpath(dest, ROOT)}
    shutil.copytree(CORPUS, os.path.join(dest, "recipes"))
    done["recipes_copied"] = len(glob.glob(os.path.join(CORPUS, "*.jsonl")))
    for name in ("hint_buckets.jsonl", "hint_probes.jsonl"):
        p = os.path.join(ROOT, "bench", name)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(dest, name))
    npz = os.path.join(ROOT, "index", "hints.npz")
    if os.path.exists(npz):
        shutil.move(npz, os.path.join(dest, "hints.npz"))
        done["hints_npz"] = "moved into the archive"
    store = os.path.abspath(skills.STORE)
    if os.path.exists(store):
        shutil.copytree(store, os.path.join(ARCHIVE, f"skills-{stamp}"))
        done["store_backup"] = os.path.relpath(
            os.path.join(ARCHIVE, f"skills-{stamp}"), ROOT)
    db = os.path.abspath(jobs.DB)
    if os.path.exists(db):
        import sqlite3
        bak = os.path.join(ARCHIVE, f"jobs-{stamp}.sqlite3")
        src = sqlite3.connect(db)
        dst = sqlite3.connect(bak)
        try:
            src.backup(dst)
        finally:
            src.close()
            dst.close()
        done["jobs_db_backup"] = os.path.relpath(bak, ROOT)
    h = hashlib.sha256()
    for p in sorted(glob.glob(os.path.join(dest, "recipes", "*.jsonl"))):
        with open(p, "rb") as f:
            h.update(f.read())
    done["recipes_sha256"] = h.hexdigest()
    with open(os.path.join(dest, "ARCHIVED.json"), "w",
              encoding="utf-8") as f:
        json.dump(done, f, indent=2)
    return done


def write_manifest(extra: dict | None = None) -> dict:
    import skills
    m = skills.manifest()
    m.update(extra or {})
    p = os.path.join(os.path.abspath(skills.STORE), "MANIFEST.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(m, f, indent=1, sort_keys=True)
        f.write("\n")
    return m


def _temp_store() -> str:
    import jobs
    import skills
    tmp = tempfile.mkdtemp(prefix="yamadori_migrate_dry_")
    jobs.DB = os.path.join(tmp, "jobs.sqlite3")
    skills.STORE = os.path.join(tmp, "skills")
    skills._ENSURED.clear()
    skills._invalidate()
    return tmp


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--authored", action="store_true")
    ap.add_argument("--ingest", default="")
    ap.add_argument("--goal", default="")
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--report", default="")
    a = ap.parse_args(argv)
    if a.ingest:
        got = ingest_folder(a.ingest, goal=a.goal)
        for g in got:
            print(f"  queued {g['path']} as {g['id']} (frontier path)")
        print(f"  {len(got)} skill folder(s) queued; the worker decomposes "
              "them")
        return 0
    if not (a.dry_run or a.apply or a.authored):
        ap.error("give --dry-run, --apply, --authored or --ingest DIR")
    report: dict = {}
    if a.dry_run:
        tmp = _temp_store()
        print(f"  dry run into a temp store: {tmp}")
    rows, counts = load(a.corpus)
    report["corpus"] = counts
    print(f"  corpus: {json.dumps(counts)}")
    if a.apply and not a.dry_run:
        stamp = _stamp()
        report["archive"] = backup_and_archive(stamp)
        print(f"  archived: {json.dumps(report['archive'])}")
    if a.dry_run or a.apply:
        import skill_compile
        lookup = None
        try:
            lookup = skill_compile.licence_lookup_from_datasets()
        except Exception:                                        # noqa: BLE001
            lookup = None
        t0 = time.time()
        ids, per = migrate(rows, lookup=lookup)
        report["migration"] = per["groups"]
        report["outcomes"] = _outcomes(ids)
        report["skills"] = per["skills"]
        print(f"  groups: {json.dumps(per['groups'])}")
        if a.apply and not a.dry_run:
            # The migration's own fingerprint, before anything else arms:
            # models/manifest.yaml records this file (skills-migration).
            import skills as _skills
            write_manifest({"source": "mcp/skill_migrate.py",
                            "corpus_rows": counts.get("rows")})
            store = os.path.abspath(_skills.STORE)
            shutil.copyfile(os.path.join(store, "MANIFEST.json"),
                            os.path.join(store, "MIGRATION.json"))
        print(f"  outcomes: {json.dumps(report['outcomes'])}")
        print(f"  {time.time() - t0:.1f} s")
    if a.authored or a.dry_run:
        got = install_authored()
        report["authored"] = got
        for g in got:
            print(f"  authored {g['name']}: {g['status']}"
                  + (f" -- {g.get('reason')}" if g.get("reason") else ""))
    m = write_manifest({"source": "mcp/skill_migrate.py",
                        "corpus_rows": counts.get("rows")})
    report["manifest"] = {"n": m["n"], "sha256": m["sha256"]}
    print(f"  store manifest: {m['n']} skill folders, sha256 {m['sha256']}")
    if a.report:
        with open(a.report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
