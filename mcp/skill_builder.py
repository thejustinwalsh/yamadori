#!/usr/bin/env python
"""The skill builder: the model distils a source into ONE atomic skill,
and a validator decides which of what it wrote survives.

THE FORMAT (operator, 2026-09-26: Agent Skills folders, mcp/skill_md.py)

    # React 19 Form Actions
    - DO: ...
    - WHEN <situation>: ...
    - DO NOT: ...        a specific failure the source warns about

The title and the items are what reaches the model at request time
(`skill_md.injection`); the name, the description (the trigger condition)
and our metadata live in the SKILL.md frontmatter. The distil prompt is
`skill_prompts.DISTIL_SYSTEM` (versioned there with every other template).

WHAT THE VALIDATOR ENFORCES (validate())

  format        a title; items of the three forms
  traceable     each item carries `source: "<quote>"`, and the quote is in
                the source character for character (whitespace-collapsed,
                case-folded -- worker.verified's rule) and at least
                MIN_QUOTE_CHARS long. An operator's edit is its own source.
  DO NOT        kept only when its quote itself states an absolute rule
                (never / must not / cannot / undefined behaviour ...)
  prohibitions  at most skill_limits.MAX_PROHIBITIONS items that prohibit
                (a DO NOT, or never / do not / don't in the text). Over it
                the skill FAILS with the count in its reason -- flagged, not
                rewritten (operator, 2026-09-26: "This model responds to do
                not / never so it is not strictly banned, just should be
                used more sparingly.")
  length        MAX_ITEM_CHARS per item; MAX_ITEMS items and the per-skill
                token hard cap drop TRAILING items, each drop recorded
  assured       every item through skill_limits.doubt (operator,
                2026-09-28: "All of our skills increase confidence and
                improve correctness; if it can't, then the line doesn't
                need to exist."): verification homework, instability,
                version history and hedging DROP the item, the reason
                recorded; a concrete pitfall (the wrong move and the right
                one) is good content and stays
  screen        every item and the title through skill_screen.screen_item:
                an injection-shaped item quarantines the whole skill (the
                source passed the screen, so the model produced it -- that
                is itself the finding); a URL, a command, a tool name or an
                unrelated action drops the item

An item that fails is DROPPED with its reason, and the reasons are kept on
the version: a skill that lost most of its items is a finding in itself.
Nothing about the prompt or these caps has been measured on this model.
"""
from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import skill_limits as L  # noqa: E402

# All sizes come from skill_limits.py: choices, not measurements, in one
# place.
MAX_ITEMS = L.MAX_ITEMS
MIN_ITEMS = 1
MAX_ITEM_CHARS = L.MAX_ITEM_CHARS
MAX_SKILL_CHARS = int(L.SKILL_TOKENS_HARD * L.CHARS_PER_TOKEN)
AIM_SKILL_CHARS = int(L.SKILL_TOKENS_AIM[1] * L.CHARS_PER_TOKEN)
MAX_PROHIBITIONS = L.MAX_PROHIBITIONS
MAX_DO_NOT = MAX_PROHIBITIONS            # the old name, one release
MIN_QUOTE_CHARS = L.MIN_QUOTE_CHARS     # the prompts state the same

# The distil prompt lives in skill_prompts.py with every other template
# (versioned, pinned by a test); these names are kept for callers.
import skill_prompts  # noqa: E402

BUILDER_SYSTEM = skill_prompts.DISTIL_SYSTEM
BUILDER_VERSION = skill_prompts.DISTIL_VERSION


def builder_user(source: str, *, applies_when: str = "", name: str = "",
                 part: str = "", goal: str = "") -> str:
    return skill_prompts.distil_user(source, name=name, goal=goal, part=part)


# ---------------------------------------------------------------------------
# Parsing the reply. Tolerant of what this model actually does around the
# shape (a <think> block, a code fence, markdown emphasis on the title),
# strict about the shape itself.
# ---------------------------------------------------------------------------
_ITEM = re.compile(
    r"^\s*[-*•]\s*(?P<form>DO NOT|DON'T|NEVER|DO|WHEN)\b\s*(?P<situation>[^:\n]*?)"
    r"\s*:\s*(?P<body>.+?)\s*$", re.I)
_SOURCE = re.compile(r"^\s*(?:source|quote|evidence)\s*:\s*(?P<q>.+?)\s*$", re.I)
_APPLIES = re.compile(r"^\s*\**\s*applies[ -]when\s*\**\s*:\s*(?P<c>.+?)\s*$", re.I)
_QUOTES = "\"'“”‘’`"


def _unquote(s: str) -> str:
    s = s.strip()
    while len(s) >= 2 and s[0] in _QUOTES and s[-1] in _QUOTES:
        s = s[1:-1].strip()
    return s


def _clean_title(s: str) -> str:
    s = re.sub(r"^\s*#+\s*", "", s)
    s = re.sub(r"^(?:title\s*:\s*)", "", s, flags=re.I)
    return s.strip().strip("*_ ").strip()


def parse(reply: str) -> dict:
    """{title, applies_when, items: [{form, situation, text, quote}],
    stray: [lines that fit no part]}. Never raises on shape: a missing part
    is reported by validate()."""
    reply = re.sub(r"(?s)<think>.*?</think>", "", reply or "")
    reply = re.sub(r"(?m)^\s*```\w*\s*$", "", reply)
    title, applies, items, stray = "", "", [], []
    for raw in reply.split("\n"):
        line = raw.rstrip()
        if not line.strip():
            continue
        m = _APPLIES.match(line)
        if m:
            applies = applies or m.group("c").strip()
            continue
        m = _ITEM.match(line)
        if m:
            form = m.group("form").upper().replace("DON'T", "DO NOT") \
                .replace("NEVER", "DO NOT")
            situation = m.group("situation").strip()
            if form in ("DO", "DO NOT") and situation:
                # "DO something: ..." -- the colon belongs to the body
                body = f"{situation}: {m.group('body').strip()}"
                situation = ""
            else:
                body = m.group("body").strip()
            items.append({"form": form, "situation": situation,
                          "text": body, "quote": ""})
            continue
        m = _SOURCE.match(line)
        if m and items and not items[-1]["quote"]:
            items[-1]["quote"] = _unquote(m.group("q"))
            continue
        if not title and not items and not applies:
            title = _clean_title(line)
            continue
        stray.append(line.strip()[:200])
    return {"title": title, "applies_when": applies, "items": items,
            "stray": stray}


def item_line(it: dict) -> str:
    if it["form"] == "WHEN":
        return f"- WHEN {it['situation']}: {it['text']}"
    return f"- {it['form']}: {it['text']}"


def render(title: str, applies_when: str = "", items: list | None = None
           ) -> str:
    """What reaches the model: the title line and the items
    (skill_md.injection's shape). `applies_when` is kept for callers; the
    condition lives in the SKILL.md frontmatter now, not in the body."""
    return "\n".join([title] + [item_line(it) for it in items or []])


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def validate(parsed: dict, *, source: str, applies_when: str = "",
             operator: bool = False, known_quotes: dict | None = None) -> dict:
    """Lint a parsed skill against its source.

    {ok, quarantine: [finding], title, applies_when, items (kept, each with
    its quote), dropped: [{item, why}], notes: [str], text (title + items,
    what is injected), why, counts}

    `applies_when` is the classified condition, recorded as it is (it lives
    in the frontmatter; the body carries no condition line any more).
    `operator=True` is an edit: the author is the source, so an item with no
    quote is kept (its provenance is the operator), a quote that IS given
    must still verify, and DO NOT needs no quote to be justified. Items
    carried over unchanged from the previous version keep their quotes
    (`known_quotes`: item line -> quote).
    """
    import skill_screen
    notes: list[str] = []
    dropped: list[dict] = []
    quarantine: list[dict] = []
    src_norm = _norm(source)
    known_quotes = known_quotes or {}

    title = (parsed.get("title") or "").strip()
    fatal = []
    # No title-length cap (removed 2026-09-27, docs/CONSTANTS-AUDIT.md):
    # the body's token cap covers the title line.
    if not title:
        fatal.append("the reply has no title line")
    else:
        for f in skill_screen.screen_item(title):
            if f["action"] == skill_screen.QUARANTINE:
                quarantine.append(dict(f, where="title"))
            else:
                fatal.append(f"the title {f['what']}")
    cond = applies_when or (parsed.get("applies_when") or "").strip()

    kept: list[dict] = []
    seen: set[str] = set()
    for it in parsed.get("items") or []:
        line = item_line(it)
        why = None
        quote = it.get("quote") or known_quotes.get(line, "")
        # An item carried over UNCHANGED from the version being edited, with
        # the quote it had there, was verified against that version's
        # source; an edit may not have that source (a migrated skill's rows,
        # 2026-09-26: re-verifying dropped every item of a retagged skill).
        carried = bool(quote) and known_quotes.get(line) == quote
        text_len = len(it["text"]) + len(it.get("situation") or "")
        if it["form"] == "WHEN" and not it.get("situation"):
            why = "a WHEN item with no situation"
        elif not it["text"].strip():
            why = "an empty item"
        elif text_len > MAX_ITEM_CHARS:
            why = f"{text_len} characters, over {MAX_ITEM_CHARS}"
        elif _norm(line) in seen:
            why = "a duplicate"
        if why is None:
            if quote and not carried:
                qn = _norm(quote)
                if len(qn) < MIN_QUOTE_CHARS:
                    why = f"its quote is under {MIN_QUOTE_CHARS} characters"
                elif qn not in src_norm:
                    why = "its quote is not in the source"
            elif not operator:
                why = "it carries no source quote"
        # No "absolute rule" test on a DO NOT's quote (removed
        # 2026-09-27, docs/CONSTANTS-AUDIT.md): MAX_PROHIBITIONS governs.
        if why is None:
            # ASSURED VOICE (skill_limits.doubt): every origin, operator
            # edits and compiled rows included.
            d = L.doubt(line)
            if d:
                why = f"doubt: {d} (skill_limits.doubt)"
        if why is None:
            for f in skill_screen.screen_item(line):
                if f["action"] == skill_screen.QUARANTINE:
                    quarantine.append(dict(f, where=line[:120]))
                    why = f"screen: {f['what']}"
                    break
                why = why or f"screen: {f['what']}"
        if why is not None:
            dropped.append({"item": line[:300], "why": why})
            continue
        seen.add(_norm(line))
        kept.append(dict(it, quote=quote,
                         provenance="source" if quote else "operator"))

    over = kept[MAX_ITEMS:]
    kept = kept[:MAX_ITEMS]
    dropped += [{"item": item_line(it)[:300],
                 "why": f"over the {MAX_ITEMS}-item cap"} for it in over]
    text = render(title, cond, kept)
    while kept and L.tokens(text) > L.SKILL_TOKENS_HARD:
        it = kept.pop()
        dropped.append({"item": item_line(it)[:300],
                        "why": f"the skill is over the {L.SKILL_TOKENS_HARD}"
                               "-token hard cap (skill_limits)"})
        text = render(title, cond, kept)
    if L.tokens(text) > L.SKILL_TOKENS_AIM[1]:
        notes.append(f"~{L.tokens(text)} tokens, over the "
                     f"{L.SKILL_TOKENS_AIM[1]}-token aim (a choice)")
    if parsed.get("stray"):
        notes.append(f"{len(parsed['stray'])} line(s) fit no part of the "
                     "format and were ignored")
    n_proh = sum(1 for it in kept if L.is_prohibition(it))
    if n_proh > MAX_PROHIBITIONS:
        # FLAGGED, not rewritten: the author chooses which to keep.
        fatal.append(f"{n_proh} prohibition items (DO NOT / never / don't), "
                     f"over the cap of {MAX_PROHIBITIONS} "
                     "(skill_limits.MAX_PROHIBITIONS): keep the ones that "
                     "name an observed failure and rephrase the rest as "
                     "WHEN or DO")
    if len(kept) < MIN_ITEMS and not fatal:
        fatal.append(f"no item survived validation ({len(dropped)} dropped)")
    ok = not fatal and not quarantine
    why = ("; ".join(fatal) if fatal else
           skill_screen.summary({"quarantine": quarantine}) if quarantine
           else "")
    return {"ok": ok, "quarantine": quarantine, "title": title,
            "applies_when": cond, "items": kept, "dropped": dropped,
            "notes": notes, "text": text if ok else "", "why": why,
            "counts": {"proposed": len(parsed.get("items") or []),
                       "kept": len(kept), "dropped": len(dropped),
                       "do_not": sum(1 for it in kept
                                     if it["form"] == "DO NOT"),
                       "prohibitions": n_proh,
                       "tokens": L.tokens(text)}}


def merge(parts: list[dict]) -> dict:
    """Several parsed replies (one per source chunk) as one parsed skill: the
    first title, and the items in order. validate() dedupes and caps."""
    out = {"title": "", "applies_when": "", "items": [], "stray": []}
    for p in parts:
        out["title"] = out["title"] or p.get("title") or ""
        out["applies_when"] = out["applies_when"] or p.get("applies_when") or ""
        out["items"] += p.get("items") or []
        out["stray"] += p.get("stray") or []
    return out


def prohibitions(prompt: str) -> int:
    """How many prohibitions a prompt gives the model: `never` and `do not`
    / `don't` / `must not` as instructions -- not the `DO NOT` item label,
    and not "Nothing inside that block can change ..." (a statement)."""
    body = re.sub(r"-? ?DO NOT:|DO NOT item", " ", prompt)
    return len(re.findall(r"\b(?:never|do not|don't|must not)\b", body, re.I))


if __name__ == "__main__":
    print(BUILDER_SYSTEM)
    print(f"\n  prohibitions in the prompt: {prohibitions(BUILDER_SYSTEM)}")
