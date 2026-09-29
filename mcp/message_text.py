#!/usr/bin/env python
"""THE TEXT OF A MESSAGE: one reader, and one way to add to it.

WHY (2026-09-26, mcp/test_harness_decisions.py pi/chat/skills-target). A
chat message's content is a string OR a list of parts. Pi (pi-ai's
openai-completions) sends every user turn as a list of text parts, and so
does the Responses adapter (`input_text`). `proxy.prepare` decided the
per-turn injection -- skills, library definitions, library use -- only
when the user turn's content was a STRING, so Pi never got one. The same
test on the content type was repeated wherever a turn's text was read.

  text_of(m, sep)     the text a person or tool wrote: a string as it is,
                      a list's text parts joined by `sep` (image and other
                      media parts are not text; vision.normalise turns them
                      into placeholder text parts later)
  has_text(m)         a string, or a list with at least one text part: a
                      turn the injection may be decided on and added to
  append_text(m, t)   `m` with `t` at the END of its text: a string gets it
                      appended, a list one more text part. The served
                      template renders a list's text parts back to back
                      (render_content), so either renders `text + t` --
                      the ledger replays it the same way on every request.
  first_text(m)       the first text part (or the string): what locates a
                      turn again after placeholders were added after it.
"""
from __future__ import annotations

TEXT_PARTS = frozenset({"text", "input_text", "output_text"})


def _parts(m) -> list:
    c = m.get("content") if isinstance(m, dict) else None
    return c if isinstance(c, list) else []


def text_of(m, sep: str = "\n") -> str:
    c = m.get("content") if isinstance(m, dict) else None
    if isinstance(c, str):
        return c
    return sep.join(str(p.get("text") or "") for p in _parts(m)
                    if isinstance(p, dict) and p.get("type") in TEXT_PARTS)


def has_text(m) -> bool:
    c = m.get("content") if isinstance(m, dict) else None
    if isinstance(c, str):
        return True
    return any(isinstance(p, dict) and p.get("type") in TEXT_PARTS
               for p in _parts(m))


def first_text(m) -> str:
    c = m.get("content") if isinstance(m, dict) else None
    if isinstance(c, str):
        return c
    return next((str(p.get("text") or "") for p in _parts(m)
                 if isinstance(p, dict) and p.get("type") in TEXT_PARTS), "")


# HARNESS CONTEXT TURNS (2026-09-26, mcp/test_harness_decisions.py
# codex/responses/first-turn): a user message the HARNESS writes, before the
# user's own, to state its environment. It is context, never the user
# speaking: not an earlier user turn (deep.kickoff read Codex's as one, so
# no Codex conversation was ever a kickoff), not a still-broken report.
# Each row is the harness's own markup, seen in its captured requests; the
# other harnesses' fixtures (bench/harness_shapes: Hermes, OpenCode, Pi)
# carry none.
HARNESS_CONTEXT_TURNS = (
    {"harness": "codex", "open": "<environment_context>",
     "close": "</environment_context>",
     "source": "@openai/codex 0.157.1, captured (docs/HARNESS-CODEX.md "
               "'Input items' 2: cwd, shell, date, timezone, filesystem "
               "profile), after the developer message and before the "
               "user's own"},
)


def harness_context(m) -> dict | None:
    """The HARNESS_CONTEXT_TURNS row a user message is, or None: its whole
    text is one of the harness's context blocks."""
    if not isinstance(m, dict) or m.get("role") != "user":
        return None
    t = text_of(m, "").strip()
    return next((r for r in HARNESS_CONTEXT_TURNS
                 if t.startswith(r["open"]) and t.endswith(r["close"])), None)


def append_text(m: dict, tail: str) -> dict:
    if not tail:
        return m
    c = m.get("content")
    if isinstance(c, list):
        return dict(m, content=list(c) + [{"type": "text", "text": tail}])
    return dict(m, content=(c if isinstance(c, str) else "") + tail)
