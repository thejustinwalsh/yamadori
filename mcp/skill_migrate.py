#!/usr/bin/env python
"""Convert the recipe corpus into skills, through the same pipeline.

    python mcp/skill_migrate.py --plan       the groups, counted; writes nothing
    python mcp/skill_migrate.py --screen     + the deterministic screen of
                                             every group, offline; writes
                                             nothing
    python mcp/skill_migrate.py --enqueue    create one skill per group in the
                                             store and enqueue it; the worker
                                             runs screen -> screen_model ->
                                             classify -> distil -> validate ->
                                             arm on the gpu lane

WHAT GOES IN

Every `bench/recipes/*.jsonl` row a person kept or edited (`_state` keep or
edited). Unreviewed rows stay behind: a migration is the moment to carry
forward only what was looked at. Rejected rows never come. A row the domain
benchmark's leak review marked as a LEAK (bench/domain/leak_review.json:
the recipe carries a task's answer) is excluded, because the shingle
preflight in bench/domain/run.py reads bench/recipes, not the skill store.

HOW IT IS GROUPED

By domain (domains.recipe_domains: the domain tags a row declares, or
infers from its area and language), then by source file inside a domain, so
a skill is one coherent source -- then cut into parts of at most GROUP_MAX
rows in category order, because a skill holds at most ~10 items and a
distil of 115 rows into 10 would throw most of them away unread.

Each group is rendered as a small source document -- a `language:` header
per declared language (the classify stage's declared evidence), then one
block per recipe: its text, its trigger, its verbatim evidence -- and enters
the pipeline at `screen` like any pasted source. The builder cites quotes
from that rendering, and the validator checks them against it, so every item
of a migrated skill traces to a recipe row (`trace()`).

This migration runs the model twice per group (screen_model and distil), so
the real run is GPU time on the worker's gpu lane. `--plan` says how many.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "..", "bench", "recipes")
LEAK_REVIEW = os.path.join(HERE, "..", "bench", "domain", "leak_review.json")
KEEP = ("keep", "edited")
GROUP_MAX = int(os.environ.get("YAMADORI_SKILL_MIGRATE_GROUP", "30"))
NAME_PREFIX = "migrated_"


def row_hash(r: dict) -> str:
    """bench/domain/run.py's leak key: sha1(recipe + ' ' + evidence)[:12]."""
    text = f"{r.get('recipe') or ''} {r.get('evidence') or ''}"
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


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
    rows, counts = [], {"files": 0, "rows": 0, "kept": 0, "edited": 0,
                        "unreviewed": 0, "reject": 0, "leak_excluded": 0,
                        "unparseable": 0}
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
                if st not in KEEP:
                    counts[st if st in counts else "unreviewed"] += 1
                    continue
                counts["kept" if st == "keep" else "edited"] += 1
                if not str(r.get("recipe") or "").strip():
                    continue
                h = row_hash(r)
                if h in bad:
                    counts["leak_excluded"] += 1
                    continue
                r.update(_file=stem, _line=i, _hash=h)
                rows.append(r)
    return rows, counts


def _declared(rows: list[dict], domain: str = "") -> dict[str, str]:
    """Vocabulary ids the rows' own labels name, with the header line that
    will declare each in the rendering: languages from `language`, and the
    design artifact for the visual-design corpora (their domain tag)."""
    import skill_classify
    out: dict[str, str] = {}
    for r in rows:
        for raw in re.split(r"[/,]", str(r.get("language") or "")):
            raw = raw.strip()
            for t in skill_classify.VOCAB:
                if raw and raw.lower() == t.name.lower():
                    out.setdefault(t.id, f"language: {t.name}")
    if "visual-design" in domain.split("+"):
        out.setdefault("ui_design", "artifact: UI and visual design")
    return out


def plan(rows: list[dict]) -> list[dict]:
    """Groups by domain. A file's domain is the commonest domain key among
    its rows (per-row tags would scatter one design file over 30 one-row
    groups). Within a domain, whole files are packed into groups of at most
    GROUP_MAX rows, in file order; a larger file is split on its own, in
    category order."""
    import domains
    from collections import Counter
    by_file: dict[str, list[dict]] = {}
    for r in rows:
        by_file.setdefault(r["_file"], []).append(r)
    by_dom: dict[str, list[tuple[str, list[dict]]]] = {}
    for stem, rs in sorted(by_file.items()):
        keys = Counter()
        for r in rs:
            d = domains.recipe_domains(r, stem)
            keys["+".join(sorted(d)) if d else "untagged"] += 1
        dom = sorted(keys.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        rs = sorted(rs, key=lambda r: (str(r.get("category") or ""),
                                       r["_line"]))
        by_dom.setdefault(dom, []).append((stem, rs))
    out = []
    for dom, files in sorted(by_dom.items()):
        bins: list[list[tuple[str, list[dict]]]] = []
        cur: list[tuple[str, list[dict]]] = []
        size = 0
        for stem, rs in files:
            if len(rs) > GROUP_MAX:
                if cur:
                    bins.append(cur)
                    cur, size = [], 0
                bins += [[(stem, rs[i:i + GROUP_MAX])]
                         for i in range(0, len(rs), GROUP_MAX)]
                continue
            if cur and size + len(rs) > GROUP_MAX:
                bins.append(cur)
                cur, size = [], 0
            cur.append((stem, rs))
            size += len(rs)
        if cur:
            bins.append(cur)
        slug = re.sub(r"[^a-z0-9]+", "_", dom.lower()).strip("_")
        for n, b in enumerate(bins, 1):
            part = [r for _stem, rs in b for r in rs]
            stems = [s for s, _rs in b]
            out.append({"name": f"{NAME_PREFIX}{slug}_{n:02d}", "domain": dom,
                        "files": stems, "part": n, "parts": len(bins),
                        "rows": part, "declared": _declared(part, dom)})
    return out


def render(g: dict) -> str:
    """The group as a source document the pipeline reads."""
    names = sorted({str(r.get("source_name") or "").strip()
                    for r in g["rows"] if r.get("source_name")})
    lines = [f"# Recipes: {g['domain']}"
             + (f" (group {g['part']} of {g['parts']})" if g["parts"] > 1
                else ""), "", "files: " + ", ".join(g["files"])]
    if names:
        lines.append("sources: " + "; ".join(names[:6]))
    for line in sorted(set(g["declared"].values())):
        lines.append(line)
    lines.append(f"domain: {g['domain']}")
    lines.append("")
    for r in g["rows"]:
        lines.append(f"## {r.get('category') or 'recipe'}")
        lines.append(f"recipe: {str(r['recipe']).strip()}")
        if r.get("trigger_condition"):
            lines.append(f"when: {str(r['trigger_condition']).strip()}")
        if r.get("evidence"):
            lines.append(f"evidence: {str(r['evidence']).strip()}")
        lines.append("")
    return "\n".join(lines)


def trace(items: list[dict], rows: list[dict]) -> list[dict]:
    """Each item's quote -> the recipe rows whose recipe or evidence holds
    it. An item that traces to no row is a migration defect."""
    import skill_builder
    norm = [(r, skill_builder._norm(f"{r.get('recipe') or ''}\n"
                                    f"{r.get('trigger_condition') or ''}\n"
                                    f"{r.get('evidence') or ''}"))
            for r in rows]
    out = []
    for it in items:
        q = skill_builder._norm(it.get("quote") or "")
        refs = [f"{r['_file']}:{r['_line']}" for r, n in norm if q and q in n]
        out.append({"item": skill_builder.item_line(it), "rows": refs})
    return out


def enqueue(groups: list[dict], *, force: bool = False) -> dict:
    """Create one skill per group (skipping a name that already exists
    unless `force`), each entering the pipeline at `screen`."""
    import skills
    made, skipped = [], []
    for g in groups:
        if not force and skills.find(g["name"]):
            skipped.append(g["name"])
            continue
        s = skills.create(
            text=render(g), name=g["name"], origin="migration",
            author="pipeline:migration",
            meta={"declared": g["declared"],
                  "migration_group": {"files": g["files"],
                                      "domain": g["domain"],
                                      "part": g["part"], "parts": g["parts"],
                                      "rows": [f"{r['_file']}:{r['_line']}"
                                               for r in g["rows"]]}})
        made.append(s["id"])
    return {"created": len(made), "skipped_existing": len(skipped),
            "ids": made}


def screen_all(groups: list[dict]) -> dict:
    """The deterministic screen over every group's rendering. Offline."""
    import skill_screen
    out = {"groups": len(groups), "pass": 0, "quarantine": 0, "by_rule": {},
           "quarantined": []}
    for g in groups:
        res = skill_screen.screen(render(g), kind="text")
        if res["ok"]:
            out["pass"] += 1
            continue
        out["quarantine"] += 1
        for f in res["quarantine"]:
            out["by_rule"][f["rule"]] = out["by_rule"].get(f["rule"], 0) + 1
        out["quarantined"].append({"name": g["name"],
                                   "why": skill_screen.summary(res)})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--screen", action="store_true")
    ap.add_argument("--enqueue", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="with --enqueue: create even where the name exists")
    ap.add_argument("--corpus", default=CORPUS)
    a = ap.parse_args(argv)
    rows, counts = load(a.corpus)
    groups = plan(rows)
    doms: dict[str, int] = {}
    for g in groups:
        doms[g["domain"]] = doms.get(g["domain"], 0) + 1
    print(f"  corpus: {json.dumps(counts)}")
    print(f"  {len(rows)} rows to migrate -> {len(groups)} skill group(s) of "
          f"at most {GROUP_MAX} rows; {2 * len(groups)} model calls "
          "(screen_model + distil, one chunk each)")
    for d, n in sorted(doms.items(), key=lambda x: -x[1]):
        print(f"    {n:>3}  {d}")
    if a.screen:
        res = screen_all(groups)
        print(f"  deterministic screen: {res['pass']} pass, {res['quarantine']}"
              f" quarantine; by rule {json.dumps(res['by_rule'])}")
        for q in res["quarantined"][:20]:
            print(f"    quarantine {q['name']}: {q['why'][:160]}")
    if a.enqueue:
        res = enqueue(groups, force=a.force)
        print(f"  enqueued: {res['created']} created, "
              f"{res['skipped_existing']} already existed")
        print("  the worker (mcp/worker.py) runs them; watch /skills on the "
              "dashboard or `python mcp/skills.py`")
    return 0


if __name__ == "__main__":
    sys.exit(main())
