#!/usr/bin/env python
"""Which files are the project's, and when a compaction happened. State
only: nothing here writes text the model reads.

WHY. Two features need to know the model's OWN work from what the proxy
already sees (the client's tool calls, which tool_code parses, and the tool
results in the next request): deep thinking's label rule 2 (#52,
deep._project_writes: a run is `helped` only if a PROJECT file changed after
it), and the work log's re-injection after a compaction that kept the
session key (#41/#54, switch work_log_reinject). The fix-up's scope (#56)
and the agent step's thinking cap by what the step answers (#53,
step_kind) were REMOVED 2026-09-27 (docs/CONSTANTS-AUDIT.md); the project
paths below remain for label rule 2 (switch helped_needs_change, the
operator's call).

REMOVED 2026-09-27 (operator: "keep our system prompts clean; fixes go
through skills" -- task-targeted steering in prompts; skills are the
channel): the tool-result SITUATIONS this module used to write -- the
PROGRESS line ("No project file has changed in the last N steps ...", #50,
switch progress_note), the UNCHANGED re-read line ("Unchanged since step N
...", #54, switch unchanged_read) and the files-read listing in the
post-compaction work log. Lines already recorded in the ledger replay byte
for byte (ledger_restore replays the stored text; nothing regenerates it).

  PROJECT PATHS  `project_of` / `is_project`: which written paths are the
                 project's. Not a temp directory (/tmp, /var/tmp, a Windows
                 Temp), not a dot-file, not a scratch-harness name
                 (SCRATCH_NAMES, each row seen in a run), not outside the
                 working directory when one is known -- unless the task
                 itself names the file. The task's files: relative paths
                 in its prose, and a project TREE's files qualified by
                 their folders (tree_paths: space-shooter/ js/ config.js
                 -> js/config.js). The working directory: one the
                 conversation states, else inferred from the first write
                 whose path ends in a file the task names (the most
                 qualified name decides), WIDENED to the deepest common
                 directory when a later named file lands outside it
                 (v0f-V0: root /workspace/space-shooter/js from a bare
                 config.js, and index.html outside the project). `learn`
                 reads each tool result's successful writes into it.
  COMPACTIONS    `note_compaction` / `reinject_due` / `mark_reinjected`: the
                 work log is re-injected on the first request after one.

The per-conversation state lives in the ledger (`work:<lineage>`, kind
"work"), so it survives a compaction and a restart, and it is pruned with
the conversation. The scratch-name table and the inspect table are CHOICES.
"""
from __future__ import annotations

import json
import re
import threading

_MAX_NAMED = 60
# State fields the removed situations kept (2026-09-27); dropped from a
# conversation's state the next time it is saved.
_RETIRED = ("reads", "seen", "since", "since_t", "step", "last_write")

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


def _infer_root(path: str, named: list[str]) -> str | None:
    """The working directory from an absolute write of a file the task
    names: /workspace/space-shooter/js/game.js + js/game.js ->
    /workspace/space-shooter. The most qualified name that matches decides
    (js/game.js over a bare game.js)."""
    p = _norm(path)
    if not _is_abs(p) or _TEMP.search(p):
        return None
    for n in sorted(named, key=lambda x: -x.count("/")):
        if p.endswith("/" + n):
            r = p[:-(len(n) + 1)]
            return r or None
    return None


def _common_root(a: str, b: str) -> str | None:
    """The deepest directory holding both; None when that is only the
    filesystem root (or a drive), which is no working directory."""
    pa, pb = a.rstrip("/").split("/"), b.rstrip("/").split("/")
    k = 0
    while k < min(len(pa), len(pb)) and pa[k] == pb[k]:
        k += 1
    r = "/".join(pa[:k])
    if not r or r == "/" or re.fullmatch(r"[A-Za-z]:", r):
        return None
    return r


def _learn_root(proj: dict, st: dict, path: str) -> None:
    """An inferred working directory from the write of a file the task
    names (never over a stated one). The first such write sets it; a later
    one outside it WIDENS it to the deepest directory holding both, so a
    bare name that matched in a subfolder (config.js written to js/) does
    not fence the project in (the next named file, index.html, lands one
    level up)."""
    if proj.get("root_by") == "stated":
        return
    r = _infer_root(path, proj["named"])
    if not r:
        return
    cur = proj.get("root")
    if not cur:
        new = r
    elif r == cur or r.startswith(cur + "/"):
        return
    else:
        new = _common_root(cur, r)
        if not new:
            return
    proj["root"] = st["root"] = new
    proj["root_by"] = st["root_by"] = "inferred"

# ============================================================ tool calls ===


def _args(call: dict) -> dict:
    raw = ((call or {}).get("function") or {}).get("arguments")
    try:
        a = json.loads(raw) if isinstance(raw, str) else (raw or {})
    except ValueError:
        return {}
    return a if isinstance(a, dict) else {}


def _name(call: dict) -> str:
    return str(((call or {}).get("function") or {}).get("name") or "")


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


def _step(raw: list[dict]) -> tuple[int | None, list[tuple[dict, str]]]:
    """The step this request answers: (index of the last assistant turn with
    tool calls, [(call, its result's text)]) when the request ENDS on tool
    results; (None, []) otherwise."""
    if not raw or not isinstance(raw[-1], dict) or raw[-1].get("role") != "tool":
        return None, []
    ai = next((i for i in range(len(raw) - 1, -1, -1)
               if isinstance(raw[i], dict) and raw[i].get("role") == "assistant"
               and raw[i].get("tool_calls")), None)
    if ai is None:
        return None, []
    results = {}
    for m in raw[ai + 1:]:
        if isinstance(m, dict) and m.get("role") == "tool":
            results[m.get("tool_call_id")] = _text(m)
    out = []
    for c in raw[ai].get("tool_calls") or []:
        if isinstance(c, dict):
            out.append((c, results.get(c.get("id"), "")))
    return ai, out


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


def learn(account: str, lineage: str, raw: list[dict]) -> dict:
    """Read the step a request answers into the project: the task's named
    files, and the working directory inferred from a successful write of a
    named file (_learn_root). Idempotent -- a retry of the request changes
    nothing -- and it writes no text. Returns {root, root_by, named,
    project_writes, scratch_writes}. Never raises."""
    try:
        with _lock(account, lineage):
            return _learn(account, lineage, raw)
    except Exception as e:                                       # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"[:200]}


def _learn(account, lineage, raw) -> dict:
    import deep
    import tool_code
    st = load(account, lineage)
    before = json.dumps(st, sort_keys=True)
    for k in _RETIRED:
        st.pop(k, None)
    rec: dict = {"project_writes": [], "scratch_writes": []}
    proj = project_of(raw, st)
    st["named"] = proj["named"]
    _ai, calls = _step(raw)
    for c, text in calls:
        try:
            # whole_pages: WHICH FILE the call writes -- an .html page with
            # only `src` scripts has no script unit (v0f-V0 index.html).
            units = tool_code.detect(c, whole_pages=True).get("units") or []
        except Exception:                                        # noqa: BLE001
            units = []
        if not units or deep.is_error(text) or deep.not_applied(text):
            continue
        for p in dict.fromkeys(u.get("path") for u in units if u.get("path")):
            _learn_root(proj, st, p)
            yes, why = is_project(p, proj)
            if yes:
                rec["project_writes"].append(_norm(p))
            else:
                rec["scratch_writes"].append({"path": _norm(p), "why": why})
    if json.dumps(st, sort_keys=True) != before:
        save(account, lineage, st)
    rec.update(root=proj.get("root"), root_by=proj.get("root_by"),
               named=len(proj["named"]))
    return rec


def project_hint(account: str, lineage: str, raw: list[dict]) -> dict:
    """The project for a request (for the fix-up's scope and the deep
    labels): the conversation's state plus what this request names."""
    try:
        return project_of(raw, load(account, lineage))
    except Exception:                                            # noqa: BLE001
        return project_of(raw)

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
