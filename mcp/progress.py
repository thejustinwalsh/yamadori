#!/usr/bin/env python
"""Which files are the project's, and when a compaction happened. State
only: nothing here writes text the model reads.

  PROJECT PATHS  `project_of` / `is_project`: which written paths are the
                 project's (read by skill_select, which judges a write or a
                 read as the work's own evidence). Not a temp directory
                 (/tmp, /var/tmp, a Windows Temp), not a dot-file, not a
                 scratch-harness name (SCRATCH_NAMES, each row seen in a
                 run), not outside the working directory the conversation
                 states -- unless the task itself names the file. The
                 task's files: relative paths in its prose, and a project
                 TREE's files qualified by their folders (tree_paths:
                 space-shooter/ js/ config.js -> js/config.js).
  COMMANDS       `command_of` / `inspect_only`: a shell call's command, and
                 whether it only looks (skill_select).
  COMPACTIONS    `note_compaction` / `reinject_due` / `mark_reinjected`: the
                 work log is re-injected on the first request after one
                 (#41/#54, switch work_log_reinject).

REMOVED 2026-09-29 with deep thinking (docs/REMOVED.md): `learn` /
`project_hint`, which read each step's successful writes into the
conversation's project -- the named files and a working directory INFERRED
from the first write of a named file -- for deep thinking's label rule 2
(#52: a run is `helped` only if a project file changed after it). Earlier
removals: the fix-up's scope (#56) and the agent step's thinking cap by
result (#53, step_kind), 2026-09-27; the tool-result SITUATIONS (#50's
progress line, #54's unchanged-read line), 2026-09-27 (operator: "keep our
system prompts clean; fixes go through skills"). Lines already recorded in
the ledger replay byte for byte.

The per-conversation state lives in the ledger (`work:<lineage>`, kind
"work"), so it survives a compaction and a restart, and it is pruned with
the conversation. The scratch-name table and the inspect table are CHOICES.
"""
from __future__ import annotations

import json
import re
import threading

_MAX_NAMED = 60

# ============================================================ project ======
# A temp directory: POSIX, and Windows' %TEMP% (AppData\Local\Temp).
_TEMP = re.compile(r"(?i)^(?:/tmp|/var/tmp|/dev/shm|/private/tmp)(?:/|$)"
                   r"|/appdata/local/temp/|^[a-z]:/(?:windows/)?te?mp/")
# SCRATCH-HARNESS NAMES (a CHOICE; each row names where it was seen). A file
# the task names is never scratch, whatever its name.
SCRATCH_NAMES = (
    # v0b row 89: test_full.js, 2,241 s of fix-up (OVERTHINKING.md T7)
    (r"test_full", "Octopus v0b-V0 test_full.js"),
    # v0e p1 /tmp/harness.js (a Node vm sandbox, T2)
    (r"(?:.*[-_.])?harness(?:[-_.].*)?", "Octopus v0e-V0 p1 harness.js"),
    # v0e p2 /tmp/diag*.js
    (r"diag\w*", "Octopus v0e-V0 p2 /tmp/diag*.js"),
    # v0b .dbg*.js (also a dot-file)
    (r"\.?dbg\w*", "Octopus v0b-V0 .dbg*.js"),
)
_SCRATCH = re.compile(r"(?i)^(?:" + "|".join(p for p, _ in SCRATCH_NAMES)
                      + r")\.[\w]+$")
# A path the task names: a relative path with a file extension, in a user
# turn. Flat text, so a regex (PROTOCOL 8: paths are shapes, not grammar).
_NAMED = re.compile(r"(?<![\w@:/.-])((?:[\w.-]+/)*[\w-][\w.-]*\."
                    r"(?:html?|css|m?js|cjs|jsx|tsx?|json|py|rs|go|wgsl|glsl"
                    r"|toml|ya?ml|md|txt|c|h|cc|cpp|hpp|java|kt|rb|php|sh"
                    r"|vue|svelte|sql))(?![\w/])")
_CWD_WORDS = re.compile(
    r"(?i)\b(?:working\s+dir(?:ectory)?|cwd|workdir|current\s+directory"
    r"|project\s+root|repo(?:sitory)?\s+root)\b")
_ABS = re.compile(r"(?<![\w.~:-])(/(?:[\w.@+-]+/)*[\w.@+-]+/?"
                  r"|[A-Za-z]:[\\/](?:[\w.@+-]+[\\/])*[\w.@+-]*)")


def _norm(path: str) -> str:
    p = str(path or "").strip().strip("'\"`").replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    return p


def _is_abs(p: str) -> bool:
    return p.startswith("/") or bool(re.match(r"^[A-Za-z]:/", p))


def _text(m: dict) -> str:
    c = (m or {}).get("content")
    if isinstance(c, list):
        c = "\n".join(p.get("text") or "" for p in c
                      if isinstance(p, dict) and p.get("type") == "text")
    return c if isinstance(c, str) else ""


# A PROJECT TREE in the task (v0f-V0's spec: "space-shooter/" then
# "  index.html  -- ...", "  js/", "    config.js  -- ..."). One entry per
# line: tree glyphs or spaces, ONE path-shaped token, then nothing or a
# comment. Flat text, so a regex per line (PROTOCOL 8); the nesting is the
# indentation. Without it the task named "config.js", the first write
# /workspace/space-shooter/js/config.js made the working directory
# /workspace/space-shooter/js, and a project file outside js/ that the task
# does not name counted as outside the project.
_TREE_LINE = re.compile(
    r"^(?P<lead>[ \t│├└─┬┼┆┊|`+\\]*(?:--\s|-\s|\*\s)?[ \t]*)"
    r"(?P<name>(?:[\w.@+-]+/)*[\w.@+-]+/?)"
    r"(?:\s+(?:--|#|//|—|–|-|\(|:).*|\s*)$")


def _tree_entry(line: str) -> tuple[int, str] | None:
    m = _TREE_LINE.match(line.expandtabs(4))
    if not m or not line.strip():
        return None
    return len(m.group("lead")), m.group("name")


def tree_paths(text: str) -> tuple[list[str], set[int]]:
    """([file paths], {line indexes used}) from the project trees in `text`.
    A tree is a run of entry lines that holds a directory (`name/`) with a
    deeper entry under it. Paths are relative to the tree's ONE top
    directory when it has one (the project folder: space-shooter/js/game.js
    -> js/game.js), else to the tree itself. Only files with a known
    extension (_NAMED's) are returned."""
    lines = (text or "").splitlines()
    out: list[str] = []
    used: set[int] = set()
    i = 0
    while i < len(lines):
        block = []
        j = i
        while j < len(lines):
            e = _tree_entry(lines[j])
            if e is None:
                break
            block.append((j, e[0], e[1]))
            j += 1
        has_dir = any(n.endswith("/") and k + 1 < len(block)
                      and block[k + 1][1] > ind
                      for k, (_, ind, n) in enumerate(block))
        if len(block) < 2 or not has_dir:
            i = max(j, i + 1)
            continue
        low = min(ind for _, ind, _ in block)
        tops = [(k, n) for k, (_, ind, n) in enumerate(block) if ind == low]
        one_top = len(tops) == 1 and tops[0][1].endswith("/") and tops[0][0] == 0
        # stack of (indent, base path) -- the one top folder is the base "".
        stack: list[tuple[int, str]] = [(low, "")] if one_top else [(-1, "")]
        for k, (li, ind, name) in enumerate(block):
            used.add(li)
            if one_top and k == 0:
                continue
            while len(stack) > 1 and stack[-1][0] >= ind:
                stack.pop()
            base = stack[-1][1]
            full = (base + "/" if base else "") + name.rstrip("/")
            if name.endswith("/"):
                stack.append((ind, full))
            elif _NAMED.fullmatch(full) and full not in out:
                out.append(full)
        i = j
    return out, used


def named_paths(messages: list[dict]) -> list[str]:
    """The relative file paths the USER's turns name (the task's files). A
    project tree's files come qualified by their folders (tree_paths); a
    bare name elsewhere that one of them already qualifies is not repeated
    (js/game.js, not also game.js)."""
    out: list[str] = []
    for m in messages or []:
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        text = _text(m)
        tree, used = tree_paths(text)
        rest = "\n".join(ln for k, ln in enumerate(text.splitlines())
                         if k not in used)
        for x in tree + [_norm(y) for y in _NAMED.findall(rest)]:
            if (x and not _is_abs(x) and x not in out
                    and not any(o.endswith("/" + x) for o in out)):
                out.append(x)
            if len(out) >= _MAX_NAMED:
                return out
    # A bare name listed before the tree that qualifies it.
    return [x for x in out if not any(o != x and o.endswith("/" + x)
                                      for o in out)]


def stated_root(messages: list[dict]) -> str | None:
    """A working directory the conversation states: an absolute path on a
    line that says 'working directory' / cwd / project root (system,
    developer or user text)."""
    for m in messages or []:
        if not isinstance(m, dict) or m.get("role") not in (
                "system", "developer", "user"):
            continue
        for ln in _text(m).splitlines():
            if not _CWD_WORDS.search(ln):
                continue
            for a in _ABS.findall(ln):
                a = _norm(a).rstrip(".,;:").rstrip("/")
                if len(a) > 1 and not _TEMP.search(a + "/"):
                    return a
    return None


def project_of(messages: list[dict], st: dict | None = None) -> dict:
    """{root, root_by, named}: the project as far as the conversation says.
    `st` (the work state) adds what earlier requests learned -- the task's
    named files survive a compaction there, and an inferred root."""
    st = st or {}
    named = list(st.get("named") or [])
    for x in named_paths(messages):
        if x not in named:
            named.append(x)
    # A bare name an earlier request kept that a qualified one now covers.
    named = [x for x in named if not any(o != x and o.endswith("/" + x)
                                         for o in named)]
    root = st.get("root")
    root_by = st.get("root_by") or ("inferred" if root else None)
    if not root or root_by != "stated":
        stated = stated_root(messages)
        if stated:
            root, root_by = stated, "stated"
    return {"root": root, "root_by": root_by, "named": named[:_MAX_NAMED]}


def is_project(path: str, project: dict | None) -> tuple[bool, str]:
    """(is this written path the project's, why). See the module
    docstring, PROJECT PATHS."""
    p = _norm(path)
    if not p:
        return False, "no path"
    project = project or {}
    for n in project.get("named") or []:
        if p == n or p.endswith("/" + n):
            return True, "named by the task"
    if _TEMP.search(p):
        return False, "a temp directory"
    base = p.rstrip("/").rsplit("/", 1)[-1]
    if base.startswith(".") and base not in (".", ".."):
        return False, "a dot-file"
    if _SCRATCH.match(base):
        return False, "a scratch-harness name"
    root = project.get("root")
    if root and _is_abs(p) and not (p == root or p.startswith(root + "/")):
        return False, "outside the working directory"
    return True, "a project path"


# ============================================================ tool calls ===


def _args(call: dict) -> dict:
    raw = ((call or {}).get("function") or {}).get("arguments")
    try:
        a = json.loads(raw) if isinstance(raw, str) else (raw or {})
    except ValueError:
        return {}
    return a if isinstance(a, dict) else {}


_COMMAND_KEYS = ("command", "cmd", "script", "commands")


def command_of(call: dict) -> str | None:
    a = _args(call)
    for k in _COMMAND_KEYS:
        v = a.get(k)
        if isinstance(v, list):
            v = " && ".join(str(x) for x in v)
        if isinstance(v, str) and v.strip():
            return v
    return None


# A shell command's segments, and the operators that make it write.
_SEGMENTS = re.compile(r"&&|\|\||;|\n|\|")
_WRITE_OPS = re.compile(
    r"(?:^|\s)(?:sed\s+(?:-\w*i|--in-place)|perl\s+-\w*i|tee\b|cp\b|mv\b"
    r"|rm\b|touch\b|mkdir\b|ln\b|truncate\b|dd\b|patch\b"
    r"|git\s+(?:apply|checkout|restore|reset|stash|am|merge|rebase|pull)\b"
    r"|npx\s+prettier\s+.*--write|prettier\s+.*--write"
    r"|eslint\s+.*--fix)")
_REDIRECT = re.compile(r"(?<![0-9&])>{1,2}\s*([^\s&|;<>]+)")


# INSPECT COMMANDS (a CHOICE; bench/octopus/overthinking classify.py's
# inspect_verify class, less its writers): a terminal step that only looks.
_INSPECT = re.compile(
    r"^(?:ls|ll|cat|head|tail|less|more|wc|pwd|whoami|tree|stat|file|du|df"
    r"|which|type|echo|printf|grep|egrep|rg|ag|find|fd|sed\s+-n|awk"
    r"|diff|cmp|md5sum|sha\w*sum|node\s+--check|node\s+-c|python3?\s+-m\s+"
    r"py_compile|git\s+(?:status|diff|log|show|ls-files|blame)|curl|wget\s+-q?O-"
    r"|jq|xxd|od|sort|uniq|cut|column|nl)\b")


def inspect_only(cmd: str) -> bool:
    if not cmd or _REDIRECT.search(cmd):
        return False
    segs = [s.strip() for s in _SEGMENTS.split(cmd) if s.strip()]
    for s in segs:
        s = re.sub(r"^(?:cd\s+\S+|sudo)\s*", "", s).strip()
        if not s:
            continue
        if not _INSPECT.match(s) or _WRITE_OPS.search(" " + s):
            return False
    return bool(segs)


# ============================================================== steps ======


# step_kind (what the step a request answers was, for #53's thinking cap by
# result) and SEARCH_NAMES were REMOVED 2026-09-27 with that cap
# (docs/CONSTANTS-AUDIT.md). inspect_only stays: skill_select reads it.

# ============================================================== state ======


_LOCKS: dict[tuple[str, str], threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock(account: str, lineage: str) -> threading.Lock:
    with _LOCKS_GUARD:
        if len(_LOCKS) > 4096:
            _LOCKS.clear()
        return _LOCKS.setdefault((account or "", lineage or ""),
                                 threading.Lock())


def _key(lineage: str) -> str:
    return "work:" + (lineage or "")


def load(account: str, lineage: str) -> dict:
    if not lineage:
        return {}
    import nebari
    try:
        return json.loads(nebari.ledger_get(account or "", _key(lineage),
                                            "work") or "{}")
    except ValueError:
        return {}


def save(account: str, lineage: str, st: dict) -> None:
    if not lineage:
        return
    import nebari
    nebari.ledger_put(account or "", lineage, _key(lineage), "work",
                      json.dumps(st, sort_keys=True))


# ======================================================= compaction ========


def note_compaction(account: str, lineage: str) -> None:
    """A compaction of this conversation finished: its next request carries
    the work log (proxy.prepare), whatever its session key -- with an
    explicit session id (#41) the key does not change, so the old
    new-key link (proxy._continue_after_compaction) never fires."""
    if not lineage:
        return
    with _lock(account, lineage):
        st = load(account, lineage)
        st["compactions"] = int(st.get("compactions") or 0) + 1
        save(account, lineage, st)


def reinject_due(account: str, lineage: str) -> bool:
    if not lineage:
        return False
    st = load(account, lineage)
    return int(st.get("compactions") or 0) > int(st.get("reinjected") or 0)


def mark_reinjected(account: str, lineage: str) -> None:
    if not lineage:
        return
    with _lock(account, lineage):
        st = load(account, lineage)
        st["reinjected"] = int(st.get("compactions") or 0)
        save(account, lineage, st)
