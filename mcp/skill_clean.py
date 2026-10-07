#!/usr/bin/env python
"""CLEAN A PINNED SOURCE BEFORE IT IS SCREENED AND DISTILLED: badge images and
HTML comments out, recorded as a step of the fetch.

Coordinator, 2026-10-07: three koota skills were quarantined by the screen for
"a markdown image whose URL carries a query string (a beacon)" -- a README's
shields.io badge -- and a TypeScript handbook page for an HTML comment of 14
words (an editor's TODO). Both are the SOURCE's decoration, not the skill's
content, and the fix is in the fetch ("the screen stays as strict as it is --
never weaken it").

WHAT IT DOES: outside fenced code and inline code spans, removes
  markdown_image   `![alt](url)` and a badge link `[![alt](img)](href)`
  html_image       `<img ...>`
  html_comment     `<!-- ... -->` (a tool directive such as `prettier-ignore`
                   or `toc` stays: the screen ignores those; an unterminated
                   `<!--` stays too, and the screen quarantines it)
A line the removal leaves empty goes with it. Code is shown verbatim when a
page is rendered, so it is never touched.

WHY IT CANNOT WEAKEN THE SCREEN (the design, which is the point):
  * OPT-IN PER SKILL, AND BOUND TO ONE FILE'S BYTES. It runs only for a skill
    whose meta carries `fetch_clean: {"pinned_sha256": <hex>}` and only when
    the fetched bytes hash to exactly that value -- the sha256 an operator
    pinned and verified in the source's MANIFEST.json (pitfall gap entries:
    bench/skills/pitfall_gap_fill.py). A page that changed since the pin, or
    any source nobody pinned (every URL a user submits), is NOT cleaned and
    is screened as it arrived, with every rule at its full strength.
  * The screen runs on the CLEANED text, which is the text the distil stage
    and the model read. Text that is removed is never read by the model: a
    stripped comment cannot instruct it, a stripped image cannot beacon.
  * The record keeps what was removed: counts per rule, the image hosts, the
    comment word counts, and the raw and cleaned sha256 and sizes
    (`fetched["clean"]`, the fetch job's result), so the step is auditable
    and the raw bytes can be re-fetched and compared.

Pure functions, no I/O: mcp/test_skill_clean.py.
"""
from __future__ import annotations

import hashlib
import re
from urllib.parse import urlsplit

VERSION = "clean/1"
RULES = ("markdown_image", "html_image", "html_comment")

MARK = "\x01"
_FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
_CODE_SPAN = re.compile(r"(`+)(?:(?!\1).)+?\1")
# [![alt](img)](href): a badge that is a link. Alt text may hold brackets only
# as escaped characters; a title after the URL is allowed.
_IMG_BODY = r"\(\s*(?:<[^>\n]*>|[^)\s]*)(?:\s+(?:\"[^\"\n]*\"|'[^'\n]*'))?\s*\)"
_LINKED_IMAGE = re.compile(r"\[\s*!\[[^\]\n]*\]" + _IMG_BODY + r"\s*\]"
                           r"(?:\([^)\n]*\)|\[[^\]\n]*\])")
_IMAGE = re.compile(r"!\[[^\]\n]*\]" + _IMG_BODY)
_HTML_IMAGE = re.compile(r"<img\b[^>]*>", re.I | re.S)
_COMMENT = re.compile(r"<!--(.*?)-->", re.S)
# The directives the screen ignores in a comment (mcp/skill_screen._DIRECTIVE).
_DIRECTIVE = re.compile(
    r"^\s*(prettier-ignore(?:-start|-end)?|markdownlint-\S+.*|eslint-\S+.*"
    r"|toc|/?toc|end ?toc|vale \S+.*|cspell:.*|spell-?checker:.*"
    r"|textlint-\S+.*|lint-\S+.*|#region.*|#endregion.*|more|truncate"
    r"|stackedit_data:.*|omit in toc)\s*$", re.I)
_URL = re.compile(r"\]\(\s*<?\s*([^)\s>]+)")
_SRC = re.compile(r"""\bsrc\s*=\s*["']?([^"'\s>]+)""", re.I)


def _host(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "")[:80]
    except ValueError:
        return ""


def _segments(text: str) -> list[tuple[bool, str]]:
    """[(is_code, text)] over the lines: a fenced block (fences included) is
    code, every run of other lines is prose."""
    out: list[tuple[bool, list[str]]] = []
    fence = None
    for line in text.split("\n"):
        m = _FENCE.match(line)
        if fence is None:
            if m:
                fence = (m.group(1)[0], len(m.group(1)))
                out.append((True, [line]))
                continue
            if out and not out[-1][0]:
                out[-1][1].append(line)
            else:
                out.append((False, [line]))
        else:
            out[-1][1].append(line)
            if m and m.group(1)[0] == fence[0] and len(m.group(1)) >= fence[1] \
                    and line.strip().strip(fence[0]) == "":
                fence = None
    return [(c, "\n".join(ls)) for c, ls in out]


def _clean_prose(text: str, rec: dict) -> str:
    spans: list[str] = []

    def keep(m):
        spans.append(m.group(0))
        return f"\x00{len(spans) - 1}\x00"
    t = _CODE_SPAN.sub(keep, text)

    def image(rule):
        def sub(m):
            rec["removed"][rule] = rec["removed"].get(rule, 0) + 1
            u = (_URL.search(m.group(0)) or _SRC.search(m.group(0)))
            h = _host(u.group(1)) if u else ""
            if h and h not in rec["hosts"]:
                rec["hosts"].append(h)
            return MARK
        return sub

    def comment(m):
        if not m.group(1).split() or _DIRECTIVE.match(m.group(1)):
            return m.group(0)
        rec["removed"]["html_comment"] = rec["removed"].get(
            "html_comment", 0) + 1
        rec["comment_words"].append(len(m.group(1).split()))
        return MARK
    t = _LINKED_IMAGE.sub(image("markdown_image"), t)
    t = _IMAGE.sub(image("markdown_image"), t)
    t = _HTML_IMAGE.sub(image("html_image"), t)
    t = _COMMENT.sub(comment, t)
    if MARK in t:
        # a line the removal left empty goes with it; the rest lose the mark
        t = "\n".join(ln.replace(MARK, "") for ln in t.split("\n")
                      if ln.replace(MARK, "").strip() or MARK not in ln)
    return re.sub(r"\x00(\d+)\x00", lambda m: spans[int(m.group(1))], t)


def clean(text: str) -> tuple[str, dict]:
    """(cleaned text, record). The record's `removed` is empty when nothing
    was removed."""
    rec = {"v": VERSION, "rules": list(RULES), "removed": {}, "hosts": [],
           "comment_words": []}
    out = []
    for is_code, seg in _segments(text or ""):
        out.append(seg if is_code else _clean_prose(seg, rec))
    cleaned = "\n".join(out)
    rec["hosts"] = sorted(rec["hosts"])
    return cleaned, rec


def apply(raw: bytes, meta: dict, cfg) -> tuple[bytes, dict]:
    """(bytes to store, the `clean` record for the fetched record). `cfg` is
    the skill's meta `fetch_clean`: {"pinned_sha256": hex, "lines": [first,
    last]?} (`lines`: keep only that section of the pinned file). Cleans only bytes
    that hash to the pin; anything else is returned untouched with the reason
    (the screen then sees the source exactly as it arrived)."""
    sha = hashlib.sha256(raw).hexdigest()
    pin = (cfg or {}).get("pinned_sha256") if isinstance(cfg, dict) else None
    if not pin:
        return raw, {"v": VERSION, "applied": False,
                     "why": "the skill names no pinned_sha256"}
    if sha != str(pin).lower():
        return raw, {"v": VERSION, "applied": False, "raw_sha256": sha,
                     "why": f"the fetched bytes do not match the pin "
                            f"{str(pin)[:12]}: the source changed since it "
                            "was pinned, so it is screened as it arrived"}
    text = raw.decode((meta or {}).get("charset") or "utf-8",
                      errors="replace")
    section = None
    lines = cfg.get("lines") if isinstance(cfg, dict) else None
    if lines:
        # A PINNED SECTION (2026-10-07): the skill is about one part of a
        # large page, and the rest holds text the model screen reads as
        # directed at an AI (tsl_compute_shaders: three.js's TSL Guide.md,
        # 296,257 bytes, whose "> IA: ..." notes the model screen
        # quarantined; the compute section, lines 2364-2640, has none).
        # Same pin, same screen: `lines` [first, last] (1-based, inclusive)
        # of the file whose sha256 is pinned, so the narrowing is the same
        # bytes every time; the screen and the distil read only the section.
        parts = text.split("\n")
        try:
            a, b = int(lines[0]), int(lines[1])
        except (TypeError, ValueError, IndexError):
            return raw, {"v": VERSION, "applied": False, "raw_sha256": sha,
                         "why": "fetch_clean.lines must be [first, last]"}
        if not 1 <= a <= b <= len(parts):
            return raw, {"v": VERSION, "applied": False, "raw_sha256": sha,
                         "why": f"fetch_clean.lines {a}-{b} is outside the "
                                f"pinned file's {len(parts)} lines"}
        text = "\n".join(parts[a - 1:b])
        section = {"lines": [a, b], "of": len(parts)}
    cleaned, rec = clean(text)
    out = cleaned.encode("utf-8")
    rec.update(applied=True, raw_sha256=sha, raw_bytes=len(raw),
               cleaned_sha256=hashlib.sha256(out).hexdigest(),
               cleaned_bytes=len(out), pinned_sha256=str(pin).lower())
    if section:
        rec["section"] = section
    return out, rec


if __name__ == "__main__":
    import json
    import sys
    t, r = clean(open(sys.argv[1], encoding="utf-8").read())
    print(json.dumps(r, indent=1))
