#!/usr/bin/env python
"""Bounded compression of a harness's SHELL results, before the model reads
them.

WHERE IT COMES FROM. Atomic Agent's "compress" step
(github.com/AtomicBot-ai/atomic-agent, MIT, commit 52f90e55):
src/compressor/result-compressor.ts (`compressToolResult`),
src/tools/os/shell-result.ts (`renderShellResult`: `overflow: "tail"`),
src/session/conversation-turn.ts (`renderToolResultBody`: a shell result is
shown TAIL-FIRST, at most TOOL_RESULT_RENDER_CAP_CHARS on the inference that
consumes it). docs/research/ATOMIC-AGENT.md has the study and the evidence
(none isolated for this step: see there) and what it would have done on our
own runs.

WHAT IT DOES, and only this:
  - Only a SHELL tool's result (SHELL_TOOLS, each name verified from a
    captured request in bench/harness_shapes). A read, a search, a write
    or an edit result is never touched, and neither is any other tool's.
  - Only when the output is over CAP_CHARS. Under it, not a byte changes.
  - Never a result that shows code the model wrote: when the command or
    the output names a path the model wrote in this conversation (a write
    or edit call's path, a shell redirect's target), the result passes
    whole (`protected`).
  - Otherwise: blank lines dropped, the last TAIL_LINES lines kept, then the
    tail that fits CAP_CHARS, cut on a line boundary (Atomic: the end of a
    command's output is what the call was made for -- the exit line, the
    verdict, the last error). A result the harness marked as a FAILURE in
    a structured field keeps its first error line on top (Atomic's
    `extractSignature`, its ERROR_MARKERS). One line says what was left
    out and how to see it: the situation and the next action.
  - The harness's wrapper is kept: Hermes' JSON (`output` compressed,
    `exit_code` and `error` untouched), Codex's header up to "Output:",
    Pi's trailing "Command exited with code N" (it is the tail).

EVERY NUMBER IS ATOMIC'S, for the operator to confirm (not measured on this
model): CAP_CHARS = TOOL_RESULT_RENDER_CAP_CHARS (8,000, what Atomic's model
sees of a fresh shell result), TAIL_LINES = SHELL_TOOL_RESULT_TAIL_LINES
(500). Atomic's own derivation of 8,000: "~1000 tokens, which covers 3-4
PDF pages or a short code file" (conversation-turn.ts); at our ~3.8
characters a token (AGENTS.md) 8,000 characters are ~2,100 tokens. The
caller passes `cap_chars` no larger than what the window leaves.

MEASURABLE: `compress` returns a record (tool, before/after characters and
lines, why) for x_yamadori.result_compress; never the text.

DECIDE ONCE. The hook compresses a result on the request that ENDS on it
and the ledger replays the same bytes after; a result the model already
read whole is never shortened later (that would change a prefix the slot
holds). With a fixed cap the function is also deterministic.

PURE: no I/O; tool_code (the one table of write tools) is imported lazily.
"""
from __future__ import annotations

import json
import os
import re

# ---------------------------------------------------------------- numbers --
CAP_CHARS = int(os.environ.get("YAMADORI_RESULT_CAP_CHARS", "8000"))
#   Atomic TOOL_RESULT_RENDER_CAP_CHARS (src/session/conversation-turn.ts).
TAIL_LINES = int(os.environ.get("YAMADORI_RESULT_TAIL_LINES", "500"))
#   Atomic SHELL_TOOL_RESULT_TAIL_LINES (src/config/config-schema.ts).

# The shell tool of each harness, VERIFIED from the captured requests in
# bench/harness_shapes (the `tools` each one offers): Hermes `terminal`,
# Pi and OpenCode `bash`, Codex `exec_command`.
SHELL_TOOLS = frozenset({"terminal", "bash", "exec_command"})

# Atomic's ERROR_MARKERS (src/compressor/result-compressor.ts), verbatim.
ERROR_MARKERS = (re.compile(r"error:", re.I), re.compile(r"failed:", re.I),
                 re.compile(r"traceback \(most recent call last\)", re.I),
                 re.compile(r"assertionerror", re.I),
                 re.compile(r"exception:", re.I))
KEY_LINE_CHARS = 180
#   Atomic `extractSignature`: `key: ${line.trim().slice(0, 180)}`.

_CODEX_OUTPUT = re.compile(r"^Output:\n", re.M)
# A shell redirect's target: `> f`, `>> f`, `tee [-a] f`; a path-shaped
# token only (it has a `.` or a `/`). `=>`, `->` and `2>&1` are not
# redirects to a file.
_REDIRECT = re.compile(r"(?:(?<![=\-<>&0-9])>>?|\btee(?:\s+-a)?)\s*['\"]?"
                       r"([\w@+./-]*[./][\w@+./-]*)")
_HEREDOC = re.compile(r"<<-?\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?")


def _outside_heredocs(cmd: str) -> str:
    """The command's own lines: a heredoc's body is a file's CONTENT (code,
    where `>` is an operator), not the command."""
    out, end = [], None
    for ln in cmd.split("\n"):
        if end is not None:
            if ln.strip() == end:
                end = None
            continue
        out.append(ln)
        m = _HEREDOC.search(ln)
        if m:
            end = m.group(1)
    return "\n".join(out)


# ---------------------------------------------------- what the model wrote --
def _norm(path: str) -> str:
    p = path.strip().strip("'\"").replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    return p


def written_paths(messages: list[dict]) -> set[str]:
    """Every path the model wrote in this conversation: a write or edit
    call's path (tool_code's table, the one list) and a shell command's
    redirect target. Each as its last two path components -- how the same
    file is named from another working directory (`/work/pagoda/src/main.tsx`
    and `src/main.tsx` after a `cd`)."""
    try:
        import tool_code
    except Exception:                                   # noqa: BLE001
        tool_code = None
    out: set[str] = set()
    for m in messages:
        if (m or {}).get("role") != "assistant":
            continue
        for c in m.get("tool_calls") or []:
            fn = (c or {}).get("function") or {}
            name = fn.get("name") or ""
            if tool_code is not None:
                for u in tool_code.detect(c, whole_pages=True)["units"]:
                    if u.get("path"):
                        out.add(_tail(_norm(u["path"])))
            if name in SHELL_TOOLS:
                cmd = _command(fn.get("arguments"))
                for t in _REDIRECT.findall(_outside_heredocs(cmd or "")):
                    if not t.startswith("/dev/") and t.strip("./"):
                        out.add(_tail(_norm(t)))
    out.discard("")
    return out


def _tail(p: str) -> str:
    parts = [x for x in p.split("/") if x]
    return "/".join(parts[-2:])


def _command(arguments) -> str | None:
    try:
        a = json.loads(arguments) if isinstance(arguments, str) else arguments
    except ValueError:
        return arguments if isinstance(arguments, str) else None
    if not isinstance(a, dict):
        return None
    cmd = a.get("command") or a.get("cmd")
    if isinstance(cmd, list):
        cmd = " ".join(str(x) for x in cmd)
    return cmd if isinstance(cmd, str) else None


def names_written(text: str, paths: set[str]) -> str | None:
    """The first written path `text` names, at a path-component boundary."""
    for p in sorted(paths):
        for m in re.finditer(re.escape(p), text):
            before = text[m.start() - 1] if m.start() else ""
            after = text[m.end()] if m.end() < len(text) else ""
            if (not before or not (before.isalnum() or before in "_-.")) \
                    and (not after or not (after.isalnum() or after in "_-")):
                return p
    return None


# ----------------------------------------------------------------- wrapper --
def _split(text: str) -> tuple[str, str, str, dict | None]:
    """(head, body, tail, json_obj): the harness's wrapper around the
    command's output. Hermes: a JSON object whose `output` is the body.
    Codex: the header through "Output:". Pi and OpenCode: plain text."""
    s = text.lstrip()
    if s.startswith("{"):
        try:
            d = json.loads(s)
        except ValueError:
            d = None
        if isinstance(d, dict) and isinstance(d.get("output"), str):
            return "", d["output"], "", d
    m = _CODEX_OUTPUT.search(text)
    if m and text[:m.start()].count("\n") <= 8 and \
            re.search(r"^Process exited with code", text[:m.start()], re.M):
        return text[:m.end()], text[m.end():], "", None
    return "", text, "", None


def _failed(text: str, obj: dict | None) -> bool:
    """A failure from a STRUCTURED field only (AGENTS.md): Hermes' exit_code
    or error."""
    if obj is None:
        return False
    if isinstance(obj.get("exit_code"), int):
        return obj["exit_code"] != 0
    return bool(obj.get("error"))


def _key_line(lines: list[str]) -> str | None:
    for ln in lines:
        if any(rx.search(ln) for rx in ERROR_MARKERS):
            return ln.strip()[:KEY_LINE_CHARS]
    return None


def omitted_line(n_lines: int, n_chars: int, kept: int) -> str:
    """What was left out and how to see it: a fact and the next action."""
    return (f"[{n_lines} earlier lines of this output ({n_chars} characters) "
            f"are not shown; the last {kept} lines follow. To see the part "
            f"not shown, run the command again narrowed: a line range, a "
            f"pattern, or the first lines.]")


# ---------------------------------------------------------------- compress --
def compress(name: str, arguments, text: str, written: set[str] | None = None,
             cap_chars: int | None = None) -> tuple[str, dict]:
    """(text the model reads, record). `written`: written_paths() of the
    conversation. `cap_chars`: at most CAP_CHARS; the caller lowers it to
    what the window leaves."""
    cap = CAP_CHARS if cap_chars is None else min(CAP_CHARS, int(cap_chars))
    rec = {"tool": name, "before_chars": len(text or ""),
           "after_chars": len(text or ""), "cap": cap}
    if name not in SHELL_TOOLS:
        return text, dict(rec, why="not_shell")
    if not isinstance(text, str) or len(text) <= cap:
        return text, dict(rec, why="under_cap")
    hit = names_written(_command(arguments) or "", written or set()) or \
        names_written(text, written or set())
    if hit:
        return text, dict(rec, why="protected")
    head, body, tail, obj = _split(text)
    failed = _failed(text, obj)
    lines = [ln for ln in body.replace("\r\n", "\n").split("\n")
             if ln.strip()]
    keep = lines[-TAIL_LINES:]
    # The wrapper and the marker come out of the same cap; the marker's
    # numbers are at most as long as these.
    overhead = len(head) + len(tail) + len(omitted_line(
        len(lines), len(body), len(lines))) + 2
    if obj is not None:
        shell = dict(obj, output="")
        overhead += len(json.dumps(shell, ensure_ascii=False))
    # Atomic `extractSignature`: the first error line of the whole output,
    # and only for a failure.
    key = _key_line(lines) if failed else None
    if key:
        overhead += len("key: ") + len(key) + 1
    room = max(0, cap - overhead)
    kept: list[str] = []
    used = 0
    for ln in reversed(keep):
        if used + len(ln) + 1 > room:
            break
        kept.append(ln)
        used += len(ln) + 1
    kept.reverse()
    if not kept and keep:
        # One line longer than the cap: its end (Atomic `keepSummaryTail`).
        kept = [keep[-1][-room:]] if room else []
    ascii_only = not any(ord(ch) > 127 for ch in text)

    def build(kept: list[str]) -> str:
        dropped = len(lines) - len(kept)
        parts = []
        if key and key not in kept:
            parts.append(f"key: {key}")
        parts.append(omitted_line(
            dropped, sum(len(ln) + 1 for ln in lines[:dropped]), len(kept)))
        parts.extend(kept)
        new_body = "\n".join(parts)
        if obj is not None:
            return json.dumps(dict(obj, output=new_body),
                              ensure_ascii=ascii_only)
        return head + new_body + tail

    if len(lines) - len(kept) <= 0:
        return text, dict(rec, why="fits")
    out = build(kept)
    # JSON escaping (a newline is two characters there) can carry the
    # result past the cap: drop the oldest kept line until it fits.
    while len(out) > cap and kept:
        kept = kept[1:]
        out = build(kept)
    if len(out) >= len(text):
        return text, dict(rec, why="no_gain")
    return out, dict(rec, after_chars=len(out), why="compressed",
                     lines_before=len(lines), lines_kept=len(kept),
                     key_line=bool(key))


def compress_at(messages: list[dict], i: int, cap_chars: int | None = None
                ) -> tuple[object, dict]:
    """(content, record) for the tool message messages[i], judged on the
    messages up to it only -- so the ledger's replay of a decision (the
    cap it was made with) gives the same bytes on every later request.
    A content that is not a string (parts, an image) passes."""
    m = messages[i] or {}
    content = m.get("content")
    name, args = "", None
    for prev in reversed(messages[:i]):
        if (prev or {}).get("role") != "assistant":
            continue
        for c in prev.get("tool_calls") or []:
            if (c or {}).get("id") == m.get("tool_call_id"):
                fn = c.get("function") or {}
                name, args = fn.get("name") or "", fn.get("arguments")
                break
        if name:
            break
    if not isinstance(content, str):
        return content, {"tool": name, "why": "not_text"}
    return compress(name, args, content, written_paths(messages[:i]),
                    cap_chars)


def compress_last(messages: list[dict], cap_chars: int | None = None
                  ) -> tuple[list[dict], list[dict]]:
    """Compress the tool results the request ENDS on (the trailing run of
    tool messages): the ones the model has not read yet. Earlier results
    are what the model already read and are left as sent (the ledger
    replays our decision for them). Returns (messages, records)."""
    start = len(messages)
    while start and (messages[start - 1] or {}).get("role") == "tool":
        start -= 1
    if start == len(messages):
        return messages, []
    out = list(messages)
    recs = []
    for i in range(start, len(messages)):
        new, rec = compress_at(messages, i, cap_chars)
        recs.append(rec)
        if new is not messages[i].get("content"):
            out[i] = dict(messages[i], content=new)
    return out, recs
