#!/usr/bin/env python
"""Rows of advice become atomic skills: the migration of the recipe corpus
and a dataset's extract stage both compile through here.

WHY GROUP, AND HOW (operator, 2026-09-26: "Don't make 2,575 one-line
skills: group rows into coherent skills by library/topic/applies-when")

A recipe row is one piece of advice with a trigger, a category and (for
most) a verbatim evidence quote. A skill is one SITUATION with a handful of
moves. So rows are grouped:

  1. by source file (one document, one voice: react.dev's useActionState
     page, the Rust nomicon's FFI chapter);
  2. inside a file, by the row's PRIMARY TOPIC -- the first API name in its
     recipe (`useActionState`, `repr(C)`), else its category -- so rows
     about one API land together;
  3. packed in that order into skills of at most MAX_ITEMS items, at most
     MAX_PROHIBITIONS prohibition items and the per-skill token hard cap
     (skill_limits): a cap reached starts the next part.

Duplicates (the same recipe text, normalised, anywhere in the corpus) are
dropped, first occurrence kept. A row over MAX_ITEM_CHARS is dropped and
counted (4 of 2,396 served rows on 2026-09-26).

WHAT A COMPILED SKILL CARRIES

  name          <file>-<topic>[-<n>], a SKILL.md name; the id is
                sha1("migration:" + name)[:12], so a re-run makes the same
                store (models/manifest.yaml records its hash)
  description   "Use when <the rows' first trigger> (<source>)."
  rule          languages / frameworks from the rows' `language` / `area`,
                the design artifact for the design corpora, and TOPICS --
                the API names the items use (a code-shaped topic is a fact
                when it appears), else the category's words: a migrated
                skill is gated by its topics, never by its language alone
  items         DO: <recipe>, each with the row's evidence as its verbatim
                quote where it has one and `ref` <file>:<line> always
  tests         skill_tests.generate from the rule (the pipeline runs them)
  provenance    the files and rows, source names and URLs, the licence as
                the dataset records it (a verbatim quote only where one was
                verified)

The source each compiled skill is screened and quote-checked against is
`render(group)`: the rows as a plain document (skill_migrate.render's shape).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import skill_limits as L  # noqa: E402

_STOP = set("""a an the and or of to in for is it this that with on be are use
using when you your as at by from into not no if then than their there these
those which while will can should must may also only more most any all each
every other such its it's via per been being was were has have had do does
make makes made new one two value values code""".split())


def _stop() -> set:
    import skill_classify
    vocab = {t.id for t in skill_classify.VOCAB} | {
        w for t in skill_classify.VOCAB for w in t.name.lower().split()}
    return _STOP | skill_classify._KEYWORDS | vocab


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def row_hash(r: dict) -> str:
    """bench/domain/run.py's leak key: sha1(recipe + ' ' + evidence)[:12]."""
    text = f"{r.get('recipe') or ''} {r.get('evidence') or ''}"
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _terms_of(row: dict) -> dict:
    """languages / frameworks a row declares (its language and area)."""
    import skill_classify as C
    line = f"{row.get('language') or ''} / {row.get('area') or ''}"
    r = C.rule_of_condition(line)
    return C.applies_to(r)


def primary_topic(row: dict) -> str:
    import skill_classify as C
    t = C.extract_topics([row.get("recipe"), row.get("trigger_condition")],
                         limit=1)
    if t:
        return t[0]
    return str(row.get("category") or "general").strip().lower() or "general"


def _words(text: str) -> list[str]:
    stop = _stop()
    return [w for w in re.findall(r"[a-z][a-z0-9-]{3,}", (text or "").lower())
            if w not in stop]


def _trigger_clause(t: str) -> str:
    t = " ".join(str(t or "").split()).rstrip(".")
    t = re.sub(r"^(?:you are|you're|you)\s+", "", t, flags=re.I)
    t = re.sub(r"^(?:when|if)\s+", "", t, flags=re.I)
    return t[:1].lower() + t[1:] if t else t


def _title_case(s: str) -> str:
    return " ".join(w[:1].upper() + w[1:] for w in re.split(r"[-_\s]+", s)
                    if w)


def item_of(row: dict) -> dict:
    return {"form": "DO", "situation": "",
            "text": " ".join(str(row["recipe"]).split()),
            "quote": " ".join(str(row.get("evidence") or "").split()),
            "ref": f"{row.get('_file')}:{row.get('_line')}"}


def pack(rows: list[dict]) -> list[list[dict]]:
    """Rows in topic order into parts under every per-skill cap."""
    parts: list[list[dict]] = []
    cur: list[dict] = []
    chars = proh = 0
    for r in rows:
        it = item_of(r)
        n = len(it["text"]) + 8
        p = int(L.is_prohibition(it))
        if cur and (len(cur) >= L.MAX_ITEMS or proh + p > L.MAX_PROHIBITIONS
                    or L.tokens("x" * (chars + n + 60)) > L.SKILL_TOKENS_HARD):
            parts.append(cur)
            cur, chars, proh = [], 0, 0
        cur.append(r)
        chars += n
        proh += p
    if cur:
        parts.append(cur)
    return parts


def render(g: dict) -> str:
    """A group as the plain document its skill is screened and checked
    against (the rows as they are, evidence included)."""
    names = sorted({str(r.get("source_name") or "").strip()
                    for r in g["rows"] if r.get("source_name")})
    lines = [f"# Recipes: {g['title']}", "", "files: " + ", ".join(
        sorted({r['_file'] for r in g['rows']}))]
    if names:
        lines.append("sources: " + "; ".join(names[:6]))
    lines.append("")
    for r in g["rows"]:
        lines.append(f"## {r.get('category') or 'recipe'} "
                     f"({r['_file']}:{r['_line']})")
        lines.append(f"recipe: {str(r['recipe']).strip()}")
        if r.get("trigger_condition"):
            lines.append(f"when: {str(r['trigger_condition']).strip()}")
        if r.get("evidence"):
            lines.append(f"evidence: {str(r['evidence']).strip()}")
        lines.append("")
    return "\n".join(lines)


def _licence(rows: list[dict], lookup=None) -> dict:
    """The rows' licence as recorded: a dataset's verified quote when one
    exists, else the recorded value, labelled as recorded."""
    vals = [str(r.get("license_if_known") or "").strip() for r in rows]
    # Deterministic: the commonest value, ties by first appearance.
    val = (max(dict.fromkeys(v for v in vals if v), key=vals.count)
           if any(vals) else "")
    for r in rows:
        if lookup and r.get("_dataset"):
            lic = lookup(r["_dataset"])
            if lic and lic.get("quote"):
                return {"spdx": val or lic.get("value"),
                        "quote": lic["quote"],
                        "where": lic.get("found_in") or "dataset licence"}
    if not val:
        return {}
    return {"spdx": val, "where": "recorded on the recipe rows "
                                  "(license_if_known; no verbatim quote)"}


def group(rows: list[dict], *, prefix: str = "") -> tuple[list[dict], dict]:
    """(groups, counts). Each group: {name, title, file, topic, rows}."""
    import skill_md
    counts = {"rows_in": len(rows), "duplicates": 0, "over_item_cap": 0}
    seen: set[str] = set()
    by_file: dict[str, list[dict]] = {}
    for r in rows:
        k = _norm(r.get("recipe"))
        if not k:
            continue
        if k in seen:
            counts["duplicates"] += 1
            continue
        seen.add(k)
        if len(" ".join(str(r["recipe"]).split())) > L.MAX_ITEM_CHARS:
            counts["over_item_cap"] += 1
            continue
        by_file.setdefault(r["_file"], []).append(r)
    groups = []
    for stem, rs in sorted(by_file.items()):
        by_topic: dict[str, list[dict]] = {}
        for r in sorted(rs, key=lambda r: r["_line"]):
            by_topic.setdefault(primary_topic(r), []).append(r)
        # One unit per topic (the operator's "group rows into coherent
        # skills by library/topic/applies-when", 2026-09-26). The pooling
        # of topics of fewer than 3 rows into a "general" unit was ours and
        # is gone (docs/CONSTANTS-AUDIT.md "compile group-size"; listed for
        # the operator to confirm).
        units = list(by_topic.items())
        for topic, trs in units:
            parts = pack(trs)
            for n, part in enumerate(parts, 1):
                label = topic if topic != "general" else (
                    str(part[0].get("category") or "general"))
                base = f"{prefix}{stem}-{label}"
                name = skill_md.to_name(base if len(parts) == 1
                                        else f"{base}-{n}")
                groups.append({"name": name, "file": stem, "topic": topic,
                               "part": n, "parts": len(parts), "rows": part})
    # Names are unique: a clash (two stems that normalise alike) is suffixed.
    used: dict[str, int] = {}
    for g in groups:
        k = g["name"]
        if k in used:
            used[k] += 1
            g["name"] = skill_md.to_name(f"{k[:58]}-{used[k]}")
        else:
            used[k] = 1
    counts["groups"] = len(groups)
    counts["rows_grouped"] = sum(len(g["rows"]) for g in groups)
    return groups, counts


def compile_group(g: dict, *, origin: str = "migration",
                  licence_lookup=None) -> dict:
    """{skill, rule, tests, source, meta, sid} for one group."""
    import skill_classify as C
    rows = g["rows"]
    langs, fws, arts = [], [], []
    for r in rows:
        a = _terms_of(r)
        for x in a["languages"]:
            if x not in langs:
                langs.append(x)
        for x in a["frameworks"]:
            if x not in fws:
                fws.append(x)
    doms = sorted({d for r in rows for d in (r.get("domains") or [])
                   if isinstance(d, str)})
    if "visual-design" in doms or g["file"].startswith("design_"):
        arts.append("ui_design")
    if not (langs or fws or arts):
        # Language-agnostic advice (algorithms, design principles) applies
        # to code work in any language.
        arts.append("code")
    items = [item_of(r) for r in rows]
    topics = C.extract_topics([it["text"] for it in items]
                              + [r.get("trigger_condition") for r in rows])
    if not any(C.code_shaped(t) for t in topics):
        cat_words = []
        for r in rows:
            for w in _words(str(r.get("category") or "").replace("-", " ")):
                if w not in cat_words:
                    cat_words.append(w)
        from collections import Counter
        common = [w for w, c in Counter(
            w for it in items for w in dict.fromkeys(_words(it["text"])))
            .most_common()
            if c >= 2 and w not in cat_words]
        # No caps on what a compiled rule keeps (removed 2026-09-27,
        # docs/CONSTANTS-AUDIT.md "compile rule caps"): every category word,
        # every word two or more items share, every language, framework and
        # trigger.
        topics = list(dict.fromkeys(topics + cat_words + common))
    aw = {"artifacts": arts, "languages": langs, "frameworks": fws,
          "topics": topics, "domains": doms,
          "triggers": [_trigger_clause(r.get("trigger_condition"))
                       for r in rows if r.get("trigger_condition")]}
    rule = C.rule_from_metadata(aw)
    first = next((r.get("trigger_condition") for r in rows
                  if r.get("trigger_condition")), "")
    src_names = sorted({str(r.get("source_name") or "").strip()
                        for r in rows if r.get("source_name")})
    where = src_names[0] if src_names else g["file"]
    what = _trigger_clause(first) or f"working with {g['topic']}"
    desc = f"Use when {what} ({where})."
    # The Agent Skills / Hermes bound (skill_limits.DESCRIPTION_CHARS).
    if len(desc) > L.DESCRIPTION_CHARS:
        desc = f"Use when {what[:L.DESCRIPTION_CHARS - 40].rstrip()}... " \
               f"({where[:30]})."
    label = g["topic"] if g["topic"] != "general" else _title_case(
        str(rows[0].get("category") or g["file"]))
    area = str(rows[0].get("area") or rows[0].get("language") or "").strip()
    title = f"{label} ({area})" if area and area.lower() not in \
        label.lower() else label
    if g["parts"] > 1:
        title = f"{title} {g['part']}/{g['parts']}"
    urls = sorted({str(r.get("source_url") or "") for r in rows
                   if r.get("source_url")})
    prov = {"source": "; ".join(src_names[:3]) or g["file"],
            "url": urls[0] if urls else None,
            "files": sorted({r["_file"] for r in rows}),
            "rows": [f"{r['_file']}:{r['_line']}" for r in rows],
            "licence": _licence(rows, licence_lookup)}
    if origin == "dataset":
        prov["dataset"] = rows[0].get("_dataset")
    skill = {"name": g["name"], "title": title, "description": desc,
             "items": items}
    import skill_tests
    tests = skill_tests.generate(rule, description=desc, title=title)
    sid = hashlib.sha1(f"{origin}:{g['name']}".encode()).hexdigest()[:12]
    return {"sid": sid, "skill": skill, "rule": rule, "tests": tests,
            "source": render(dict(g, title=title)),
            "meta": {"provenance": prov,
                     "migration_group": {"file": g["file"],
                                         "topic": g["topic"],
                                         "part": g["part"],
                                         "parts": g["parts"],
                                         "rows": prov["rows"]}}}


def licence_lookup_from_datasets(db: str | None = None):
    """A lookup from a dataset id to its licence field (datasets.py). `db`
    reads another jobs database READ-ONLY (a reproduction into a scratch
    store reads the live datasets' licence records, an input of the
    migration)."""
    cache: dict = {}
    if db:
        import sqlite3
        con = sqlite3.connect(f"file:{os.path.abspath(db)}?mode=ro", uri=True)
        try:
            for dsid, assist in con.execute("SELECT id, assist FROM datasets"):
                try:
                    a = json.loads(assist or "{}")
                except ValueError:
                    a = {}
                cache[dsid] = (a.get("fields") or {}).get("licence")
        finally:
            con.close()
        return lambda dsid: cache.get(dsid)
    import datasets

    def get(dsid: str):
        if dsid not in cache:
            d = datasets.get(dsid) or {}
            cache[dsid] = ((d.get("assist") or {}).get("fields") or {}).get(
                "licence")
        return cache[dsid]
    return get
