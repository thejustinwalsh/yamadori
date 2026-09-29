#!/usr/bin/env python
"""A skill on disk: an Agent Skills folder with a SKILL.md, read and written.

THE FORMAT (operator, 2026-09-26: "skills are proven in the greater
ecosystem")

    <name>/
      SKILL.md       YAML frontmatter + a short markdown body
      tests.json     the skill's activation tests and behaviour checks
                     (skill_tests.py); a harness ignores it

    ---
    name: browser-app-entry-point          # [a-z0-9-], <= 64, = folder name
    description: Use when ...              # THE TRIGGER CONDITION, <= 1,024
    version: 1.0.0                         # semver, as Hermes expects
    author: Yamadori
    license: MIT                           # SPDX, from a verbatim quote
    metadata:
      hermes:                              # what Hermes' linter looks for
        tags: [JavaScript, HTML, debug]
        related_skills: []
      yamadori:                            # ours, namespaced
        id: 3f2a9c0e1b4d                   # the store's id
        revision: 1                        # the store's version number
        state: armed                       # armed | quarantined | draft | ...
        category: {artifact: [...], language: [...], framework: [...],
                   phase: [...], domain: [...]}
        applies_when: {artifacts, languages, frameworks, phases,
                       situations, all_of, topics, triggers, text}
        escalate: false
        provenance: {kind, source, url, repo, path, section,
                     licence: {spdx, quote, where}, fetched_at, sha256}
        items: [{line, quote, ref}]        # each item's verbatim source
        tests: {activation: {should, should_not, passed, score}}
    ---
    # Browser App Entry Point

    ## When to use
    <the description again, for a harness that shows the body only>

    ## Guidance
    - WHEN <situation>: <what to do>
    - DO: <practice>
    - DO NOT: <one observed failure>      # at most MAX_PROHIBITIONS

Hermes (hermes-agent agent/skill_utils.py parse_frontmatter, tools/
skill_linter.py) reads exactly this: `name` must match the folder and be
[a-z0-9_-]; `description` is one sentence (its index shows 60 characters);
`version`, `author`, `license` and `metadata.hermes.{tags, related_skills}`
are what every bundled skill carries; a "When to Use" section is expected;
README.md / .env / install.sh are forbidden in a skill folder. So a skill
folder copied from index/skills/ into ~/.hermes/skills/<category>/ loads as
it is. `metadata.yamadori` is ignored there.

WHAT REACHES THE MODEL at request time is `injection()`: the title line and
the instruction items -- never the frontmatter, never "When to use" (the
selector already decided it applies).

Nothing here decides whether a skill is valid (skill_builder.validate does);
this module is the format, and `parse(render(x)) == x` for every field it
writes (mcp/test_skill_factory.py).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import skill_limits as L  # noqa: E402

NAME_RX = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_FM = re.compile(r"\A﻿?---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.S)
_ITEM = re.compile(r"^\s*[-*]\s+(DO NOT|DON'T|DO|WHEN|NEVER)\b(.*)$", re.I)
AUTHOR = "Yamadori"
SECTION_WHEN = "When to use"
SECTION_ITEMS = "Guidance"


def to_name(s: str) -> str:
    """A folder-safe skill name: lowercase, hyphens, <= 64 characters."""
    n = re.sub(r"[^a-z0-9]+", "-", (s or "").strip().lower()).strip("-")
    return (n[:L.NAME_CHARS].rstrip("-")) or "skill"


def split(text: str) -> tuple[dict, str]:
    """(frontmatter, body). PyYAML is required (it is in the stack env)."""
    import yaml
    text = (text or "").replace("\r\n", "\n")
    m = _FM.match(text)
    if not m:
        return {}, text
    try:
        # The C loader when PyYAML has it: selection parses every armed
        # SKILL.md once per process (525 in 3.4 s with the pure-Python one).
        loader = getattr(yaml, "CSafeLoader", None) or yaml.SafeLoader
        fm = yaml.load(m.group(1), Loader=loader) or {}
    except yaml.YAMLError:
        fm = {}
    return (fm if isinstance(fm, dict) else {}), text[m.end():]


def item_line(it: dict) -> str:
    form = str(it.get("form") or "DO").upper()
    if form == "WHEN":
        return f"- WHEN {it.get('situation', '').strip()}: {it['text'].strip()}"
    return f"- {form}: {it['text'].strip()}"


def parse_items(body: str) -> list[dict]:
    """The instruction items in a body: DO / WHEN / DO NOT lines (a NEVER or
    DON'T line is read as DO NOT). Lines under other headings that are not
    items are ignored."""
    out = []
    for line in (body or "").split("\n"):
        m = _ITEM.match(line)
        if not m:
            continue
        form = m.group(1).upper().replace("DON'T", "DO NOT").replace(
            "NEVER", "DO NOT")
        rest = m.group(2)
        situation = ""
        if form == "WHEN":
            mm = re.match(r"^\s*([^:]+?)\s*:\s*(.+)$", rest)
            if not mm:
                continue
            situation, text = mm.group(1), mm.group(2)
        else:
            text = re.sub(r"^\s*:\s*", "", rest).strip()
            if m.group(1).upper() == "NEVER":
                text = "never " + text if not text.lower().startswith(
                    "never") else text
        if text.strip():
            out.append({"form": form, "situation": situation.strip(),
                        "text": text.strip()})
    return out


def _title_of(body: str) -> str:
    for line in (body or "").split("\n"):
        m = re.match(r"^\s*#\s+(.+?)\s*$", line)
        if m:
            return m.group(1).strip()
    return ""


def _section(body: str, name: str) -> str:
    m = re.search(r"(?im)^\s*##\s+" + re.escape(name) + r"\s*$", body or "")
    if not m:
        return ""
    rest = body[m.end():]
    stop = re.search(r"(?m)^\s*##?\s+", rest)
    return rest[:stop.start() if stop else len(rest)].strip()


def parse(text: str) -> dict:
    """A SKILL.md as a dict:
    {name, description, version, author, license, tags, related_skills,
     title, when, items: [{form, situation, text}], yamadori: {...},
     frontmatter, body}. Never raises on shape; validate() reports."""
    fm, body = split(text)
    meta = fm.get("metadata") if isinstance(fm.get("metadata"), dict) else {}
    hermes = meta.get("hermes") if isinstance(meta.get("hermes"), dict) else {}
    ours = meta.get("yamadori") if isinstance(meta.get("yamadori"), dict) \
        else {}
    items = parse_items(_section(body, SECTION_ITEMS) or body)
    quotes = {str(x.get("line") or ""): x for x in ours.get("items") or []
              if isinstance(x, dict)}
    for it in items:
        q = quotes.get(item_line(it)) or {}
        it["quote"] = str(q.get("quote") or "")
        if q.get("ref"):
            it["ref"] = str(q["ref"])
    return {"name": str(fm.get("name") or ""),
            "description": " ".join(str(fm.get("description") or "").split()),
            "version": str(fm.get("version") or ""),
            "author": str(fm.get("author") or ""),
            "license": str(fm.get("license") or ""),
            "tags": [str(t) for t in hermes.get("tags") or []],
            "related_skills": [str(t) for t in hermes.get("related_skills")
                               or []],
            "title": _title_of(body),
            "when": " ".join(_section(body, SECTION_WHEN).split()),
            "items": items, "yamadori": ours, "frontmatter": fm,
            "body": body}


def render(sk: dict) -> str:
    """The SKILL.md text for a skill dict (parse()'s shape). Deterministic:
    the same dict always renders the same bytes (sorted nowhere it would
    reorder the author's items; yaml keys in a fixed order)."""
    import yaml

    class _D(yaml.SafeDumper):
        pass

    def _str(dumper, data):
        style = ">" if len(data) > 100 and "\n" not in data else None
        return dumper.represent_scalar("tag:yaml.org,2002:str", data,
                                       style=style)
    _D.add_representer(str, _str)
    ours = dict(sk.get("yamadori") or {})
    items = sk.get("items") or []
    ours["items"] = [{k: v for k, v in (("line", item_line(it)),
                                        ("quote", it.get("quote") or ""),
                                        ("ref", it.get("ref") or ""))
                      if v or k == "line"} for it in items]
    fm = {"name": sk["name"],
          "description": " ".join(str(sk.get("description") or "").split()),
          "version": sk.get("version") or "1.0.0",
          "author": sk.get("author") or AUTHOR,
          "license": sk.get("license") or "unspecified",
          "metadata": {"hermes": {"tags": list(sk.get("tags") or []),
                                  "related_skills": list(
                                      sk.get("related_skills") or [])},
                       "yamadori": ours}}
    head = yaml.dump(fm, Dumper=_D, sort_keys=False, allow_unicode=True,
                     width=78, default_flow_style=False)
    body = [f"# {sk.get('title') or sk['name']}", "",
            f"## {SECTION_WHEN}", "",
            " ".join(str(sk.get("when") or sk.get("description") or "")
                     .split()), "", f"## {SECTION_ITEMS}", ""]
    body += [item_line(it) for it in items]
    return "---\n" + head + "---\n" + "\n".join(body) + "\n"


def injection(sk_or_text) -> str:
    """What reaches the model: the title and the items, nothing else."""
    sk = parse(sk_or_text) if isinstance(sk_or_text, str) else sk_or_text
    lines = [sk.get("title") or sk.get("name") or ""]
    lines += [item_line(it) for it in sk.get("items") or []]
    return "\n".join(x for x in lines if x).strip()


def format_problems(sk: dict, folder: str | None = None) -> list[str]:
    """The ecosystem contract (Agent Skills + Hermes' linter errors), as
    reasons. Empty when the SKILL.md would load in a harness."""
    out = []
    name = sk.get("name") or ""
    if not NAME_RX.match(name):
        out.append(f"name {name!r} must be lowercase letters, digits and "
                   "hyphens")
    if len(name) > L.NAME_CHARS:
        out.append(f"name is {len(name)} characters, over {L.NAME_CHARS}")
    if folder is not None and os.path.basename(folder.rstrip("/\\")) != name:
        out.append(f"name {name!r} does not match its folder")
    d = sk.get("description") or ""
    if not d.strip():
        out.append("there is no description (the trigger condition)")
    elif len(d) > L.DESCRIPTION_CHARS:
        out.append(f"description is {len(d)} characters, over "
                   f"{L.DESCRIPTION_CHARS}")
    if not (sk.get("title") or "").strip():
        out.append("the body has no '# Title' line")
    if not sk.get("items"):
        out.append("the body has no instruction items")
    return out


# ---------------------------------------------------------------------------
# Folders.
# ---------------------------------------------------------------------------
def _write_atomic(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def write_folder(root: str, sk: dict, tests: dict | None = None) -> str:
    """Write <root>/<name>/SKILL.md (and tests.json). Returns the folder."""
    folder = os.path.join(root, sk["name"])
    _write_atomic(os.path.join(folder, "SKILL.md"),
                  render(sk).encode("utf-8"))
    if tests is not None:
        _write_atomic(os.path.join(folder, "tests.json"),
                      (json.dumps(tests, indent=2, sort_keys=True,
                                  ensure_ascii=False) + "\n").encode("utf-8"))
    return folder


def read_folder(folder: str) -> tuple[dict, dict]:
    """(parsed SKILL.md, tests) of one folder."""
    with open(os.path.join(folder, "SKILL.md"), encoding="utf-8") as f:
        sk = parse(f.read())
    tests = {}
    p = os.path.join(folder, "tests.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            tests = json.load(f)
    return sk, tests


def folders(root: str) -> list[str]:
    """Every skill folder under root (a SKILL.md directly inside), sorted."""
    out = []
    if not os.path.isdir(root):
        return out
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith((".",
                                                                      "_")))
        if "SKILL.md" in filenames:
            out.append(dirpath)
            dirnames[:] = []           # support dirs are not skills
    return sorted(out)


def sha256(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()
