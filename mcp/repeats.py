#!/usr/bin/env python
"""Notice when a turn is asking the same thing again, and say what is left.

THE FAILURE THIS EXISTS FOR

Measured: an agent at high reasoning effort ran the same grep for
`LineDashedMaterial` fourteen times, permuting only the glob -- `docs`,
`docs/`, `docs/**`, `*.md`, `**/*.md` -- against a directory that was not
indexed at all. Thirty-six tool calls, no answer, budget exhausted. A milder
version showed up in a live trace today: the same pattern searched twice with
a describe_index in between.

Those are five spellings of one exhausted idea. Catching that is a set lookup
on normalised arguments, which is deterministic, exact and costs microseconds.
No model is needed to notice a repeat, and using one would be slower and less
reliable than `in`.

WHY IT DOES NOT BLOCK

Refusing the call would be a negation -- "you already tried that" -- and this
stack has measured that negation is read as topic rather than as a constraint,
so it invites another retry. Measured directly: describing what failed made
the decision model pick retry at margin 0.288, while asking which action
progresses gave the right answer at 0.493.

So a repeat still runs, and the result carries what has NOT been tried yet,
phrased as available options. The agent is given somewhere to go rather than
told where it has been.
"""
from __future__ import annotations

import json
import re


def normalise(name: str, args: dict) -> str:
    """A key that collapses cosmetic variations of the same request.

    The point is that `docs`, `docs/`, `docs/**` and `**/*.md` are one idea
    wearing different clothes. Comparing raw arguments would see four distinct
    calls and notice nothing.
    """
    parts = [name]
    for key in sorted(args):
        if key in ("max_results", "top_k", "limit", "start", "end"):
            continue        # paging and caps are not a different question
        v = args[key]
        if isinstance(v, str):
            v = v.strip().lower()
            if key in ("glob", "path"):
                # Strip directory-separator and wildcard decoration, which is
                # exactly the axis the observed failure permuted along.
                v = re.sub(r"[*/\\.]+", "", v)
            else:
                v = re.sub(r"\s+", " ", v)
        parts.append(f"{key}={v}")
    return "|".join(parts)


class Turn:
    """Tracks what has been asked during one client request."""

    def __init__(self) -> None:
        self.seen: dict[str, int] = {}
        self.tools_used: set[str] = set()
        self.empty: set[str] = set()

    def record(self, name: str, args: dict, result: str) -> None:
        key = normalise(name, args)
        self.seen[key] = self.seen.get(key, 0) + 1
        self.tools_used.add(name)
        if _empty(result):
            self.empty.add(key)

    def repeat_count(self, name: str, args: dict) -> int:
        return self.seen.get(normalise(name, args), 0)

    def cached_empty(self, name: str, args: dict) -> bool:
        """Have we already run this exact call and got nothing?

        NOT a refusal. The call still returns a result and the model still
        decides what to do -- this only says the SEARCH need not be executed
        again, because a deterministic query over an unchanged index cannot
        return something different the second time.

        MEASURED, from the corpus: one request issued the identical pair of
        searches twelve times over 842 seconds against an index that contained
        nothing relevant. Each round trip cost about 47 seconds to re-derive an
        answer we already had. Re-running them bought nothing; it only meant
        the model spent a quarter of an hour to reach the same dead end.

        The guidance still goes back, which is the part the model acts on.
        """
        return normalise(name, args) in self.empty

    def guidance(self, name: str, args: dict, all_tools: set[str]) -> str:
        """What to append when a call repeats something already exhausted.

        Positive throughout: it names what is available, never what failed.
        """
        key = normalise(name, args)
        if key not in self.empty or self.seen.get(key, 0) < 2:
            return ""
        untried = sorted(all_tools - self.tools_used)
        lines = [f"[This request has now been made {self.seen[key]} times with "
                 f"the same effect.]"]
        if untried:
            lines.append("Still available this turn: " + ", ".join(untried) + ".")
        hint = _forward_hint(name, untried)
        if hint:
            lines.append(hint)
        return "\n\n" + "\n".join(lines)


def _forward_hint(name: str, untried: list[str]) -> str:
    """One concrete next step, phrased as an action that progresses."""
    if name == "find_by_pattern":
        return ("Searching without a glob covers every indexed file; "
                "describe_index lists which directories exist.")
    if name == "find_definition_opt":
        return ("find_by_pattern locates the name as a literal, including "
                "places it is imported or re-exported rather than declared.")
    if name == "find_by_meaning":
        return ("find_by_pattern matches an exact string the code would "
                "contain; describe_index shows what is covered.")
    if name == "read_file_range":
        return "describe_index lists the paths that exist."
    return ""


def _empty(text: str) -> bool:
    head = (text or "")[:140].lower()
    return (not text or "no matches" in head or "not found" in head
            or "no definition" in head or "no references" in head
            or "matched 0 of" in head or "no results" in head
            or "has no index" in head)


# THE TOOL-RESULT BREAKER.
#
# Every tool loop in the stack cut results with a bare `out[:6000]`: four
# places, no marker (docs/CONSTRAINTS.md item 12). read_file_range returns 120
# lines by default, which at ~50 chars plus a 7-char gutter can pass 6000, so
# the model received a range whose header promised lines the body did not
# contain -- and nothing told it so. Measured rare, 3 of 651 corpus results.
#
# 6000 stays, as a BREAKER: it bounds what one result can add to a context
# that holds a GPU lane. What changes is that tripping it is stated, with the
# size of what was withheld and the one call that fetches the rest -- a
# failure return carries the next step (AGENTS.md).
RESULT_CAP = 6000

_RANGE_HEAD = re.compile(r"^(?P<path>\S.*?):(?P<start>\d+)-(?P<end>\d+)\s+"
                         r"\((?P<total>\d+) lines total\)")
_GUTTER = re.compile(r"^\s*(\d+)  ", re.M)


def cap_tool_result(text: str, name: str = "", args: dict | None = None,
                    limit: int = RESULT_CAP) -> str:
    """`text` if it fits, else its first `limit` chars plus a marker.

    The cut is made at the last line break before `limit`, so no line arrives
    half-written, and the marker says where to get the rest: for
    read_file_range the exact start/end of the lines not shown, for anything
    else how to narrow the request.
    """
    text = text or ""
    n = len(text)
    if n <= limit:
        return text
    kept = text[:limit]
    nl = kept.rfind("\n")
    if nl > limit // 2:
        kept = kept[:nl]
    step = _next_step(kept, text, name, args if isinstance(args, dict) else {})
    return (kept.rstrip() + f"\n\n[truncated at {limit} of {n} chars; "
            f"{step}]")


def _next_step(kept: str, full: str, name: str, args: dict) -> str:
    """The concrete call that fetches what the cut withheld."""
    head = _RANGE_HEAD.match(full)
    if name == "read_file_range" or head:
        shown = [int(x) for x in _GUTTER.findall(kept)]
        if head and shown:
            path = args.get("path") or head.group("path")
            last = shown[-1]
            end = int(head.group("end"))
            if last < end:
                return (f"lines {int(head.group('start'))}-{last} are shown; "
                        f"request lines {last + 1}-{end} with read_file_range "
                        f"(path={json.dumps(path)}, start={last + 1}, "
                        f"end={end})")
        return ("request a smaller range with read_file_range (start/end) "
                "to see the rest")
    if name == "find_by_pattern":
        return ("the matches after this point were not shown; narrow the "
                "pattern or add a glob, or lower max_results")
    if name == "find_by_meaning":
        return ("the results after this point were not shown; lower top_k "
                "or read one of the paths above with read_file_range")
    return ("the rest was not shown; narrow the request, or read one of the "
            "paths above with read_file_range")
