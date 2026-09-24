#!/usr/bin/env python
"""Tool-call code validation: check the code a model writes INTO a client
tool call before the call leaves the proxy, and let the model fix it.

WHY (operator, 2026-09-24). An agent writes most of its code through its
client's file tools -- write_file, an edit, a patch -- not in a fenced final
answer, so the final-answer repair pass never saw it. A broken file went to the client's disk, the agent ran it, read the
traceback, and spent a whole tool round-trip -- often minutes -- on a typo a
parser finds in a millisecond.

WHAT IT DOES. When a generation ends with CLIENT tool calls that carry code,
each file's code is checked before anything is forwarded:

  whole file   (write / create)  the parser (tree-sitter; Python's own
               compiler), a light lint (Python: ruff E9,F63,F7,F82 --
               code_check.LINT_SELECT), and the formatter, whose own parser
               (TypeScript's / Babel's in prettier, rustc's in rustfmt)
               counts when it rejects what tree-sitter accepted.
  edit         (old -> new strings, diff hunks)  the replacement is
               syntax-checked. It BLOCKS only when it is a complete unit: the
               text it replaces parses on its own and the replacement does
               not. A replacement that is a fragment -- the old text does not
               parse either -- is FLAGGED, never blocked.

If anything blocks, at `high` and up (tiers.repair_on) the SECOND BRAIN
repairs it before the calls are forwarded (operator, 2026-09-24): the
fixup job (shomen.run("fixup")) gets ONLY each blocking unit's code, its
exact errors and the user's request, at the request's own effort, and the
corrected code is written back into the call (`apply`). Main generates the
call once and never sees a failed attempt -- until 2026-09-24 the rounds
ran ON MAIN, as "NOT EXECUTED" tool results and a regenerated call, turns in
the conversation's context the client never received. At `medium` the
check only notes what it found.

THE CAP: REPAIR_ROUNDS = 3 fixup rounds. Not measured -- no repair-round
distribution exists in this repo yet (the LiveBench repair fixture fixes in
one round; that is n=1). The reasoning: a syntax error named by line and
column is normally fixed in one round, a second catches a fix that broke
something else, and a third is the last before the cost stops being worth
it -- each round is a generation at the request's effort. The repaired
version is sent ONLY when it came from a closed fence, the reply was not cut
off (finish_reason "length"), it has no errors left, and it is plausibly
complete (shomen.fix_rejection); otherwise the model's own content goes,
untouched, and the note says the repair was not used and why (pre-deploy
review, 2026-09-24). The cap is shared with the final-answer repair
(the same job). Override: YAMADORI_REPAIR_ROUNDS. A unit inside a diff or a
multi-file patch has no place to write a fix back and is only noted.

WHAT THE CLIENT RECEIVES. The calls exactly as the model wrote them, except
where the second brain REPAIRED code that did not parse or failed the lint
-- and then only the lines the errors required. CHANGE WHAT THE MODEL WROTE
ONLY WHEN IT IS BROKEN (#24 in docs/SELF-IMPROVEMENT-LOG.md, 2026-09-24):
until then the formatter's output replaced every whole-file write (5-65
lines changed each, V0 pilot), the model later patched with an old_string
of its OWN pre-format text, the patch failed, and it had to re-read the
file -- the note saying it was formatted did not prevent it. The formatter
still runs, as a check (its parser counts) and as INFORMATION in the note
("prettier would change 3 lines"); its output is never applied. The code is
never modified after the client has the call. A one-line note goes out as
content just before the tool calls, in the fold-back phrases
(shomen.PHRASES): "Repaired js/game.js (javascript): 2 syntax errors fixed
in 1 round." -- "Verified" when it parsed, "Checked" when problems remain --
so the agent's transcript records what was checked and changed. The note is
MECHANICAL and carries no concept-seed line: the seed stays in the fix-up
job's own user message (it steers the second brain), but the "Today I was
inspired by ..." line belongs to second-brain work that becomes ANSWER
content (deep thinking, fan-out). The proxy then warms the slot with the
turn as delivered (proxy._warm).

WHAT IT CLAIMS. Syntax and lint of each file ON ITS OWN, and whether the
formatter would change it. Not that it compiles in the project, not that
imports resolve, not that it runs.

DETECTION: the known-names table first, then argument shape. Detection only
ever ENABLES a check; a miss skips it and the call goes out as before.
A client call that neither recognises but that carries a code-sized string
is logged (proxy log, and x_yamadori.tool_code.unknown) so the table grows
from real traffic.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import code_check  # noqa: E402

REPAIR_ROUNDS = int(os.environ.get("YAMADORI_REPAIR_ROUNDS", "3"))

# A string this long, over at least this many lines, in an unrecognised call
# is "code-sized" and logged. Short strings are commands and names.
UNKNOWN_MIN_CHARS = 200
UNKNOWN_MIN_LINES = 3

# ================================================== THE KNOWN-NAMES TABLE ===
#
# name -> the shapes that name has been seen with. One name can mean
# different argument keys in different clients (`write_file` is `path` in
# the MCP filesystem server and `file_path` in Gemini CLI; `edit_file` is an
# `edits` list in MCP and an old/new pair in Roo), so each entry lists every
# shape; a call matches the first whose keys it carries.
#
# Shape kinds:
#   write    {path, content}                  a whole file
#   replace  {path, old, new}                 one replacement
#   edits    {path, list, old, new}           a list of replacements
#   diff     {path, diff}                     SEARCH/REPLACE blocks or a
#                                             unified diff for one file
#   patch    {patch}                          a multi-file patch (V4A
#                                             "*** Begin Patch" or unified)
#   editor   {command, path, ...}             Anthropic's text editor
#
# VERIFIED means read from the client's own source or schema; everything else
# is from its public documentation as remembered, not checked here.


def _w(path, content):
    return {"kind": "write", "path": path, "content": content}


def _r(path, old, new):
    return {"kind": "replace", "path": path, "old": old, "new": new}


KNOWN: dict[str, list[dict]] = {
    # Hermes Agent: the NAMES are verified (corpus tools_offered, every
    # Hermes turn since 2026-09-23 offers write_file and patch). The argument
    # KEYS are not: the corpus logs only our own tools' arguments. From the
    # hermes-agent source as remembered: write_file(path, content);
    # patch(path, old_string, new_string, replace_all) in replace mode and
    # patch(patch) in V4A patch mode. UNVERIFIED keys.
    "write_file": [_w("path", "content"),            # MCP filesystem, Hermes
                   _w("file_path", "content")],      # Gemini CLI (unverified)
    "patch": [_r("path", "old_string", "new_string"),  # Hermes (unverified)
              {"kind": "patch", "patch": "patch"}],
    # MCP filesystem server (@modelcontextprotocol/server-filesystem):
    # edit_file(path, edits[{oldText, newText}], dryRun). UNVERIFIED.
    # Roo Code: edit_file(file_path, old_string, new_string). VERIFIED
    # (src/core/prompts/tools/native-tools/edit_file.ts).
    "edit_file": [{"kind": "edits", "path": "path", "list": "edits",
                   "old": "oldText", "new": "newText"},
                  _r("file_path", "old_string", "new_string")],
    # Anthropic's text editor tool, both names: command create -> path +
    # file_text; str_replace -> path + old_str/new_str; insert -> path +
    # new_str (+ insert_line). UNVERIFIED (API documentation).
    "str_replace_based_edit_tool": [{"kind": "editor"}],
    "str_replace_editor": [{"kind": "editor"}],
    "text_editor": [{"kind": "editor"}],
    # Claude Code: Write(file_path, content), Edit(file_path, old_string,
    # new_string, replace_all) -- VERIFIED against the schema of the Claude
    # Code instance that wrote this file. MultiEdit(file_path,
    # edits[{old_string, new_string}]) -- UNVERIFIED (not in that schema).
    "Write": [_w("file_path", "content")],
    "Edit": [_r("file_path", "old_string", "new_string")],
    "MultiEdit": [{"kind": "edits", "path": "file_path", "list": "edits",
                   "old": "old_string", "new": "new_string"}],
    # Gemini CLI: write_file (above), replace(file_path, old_string,
    # new_string). UNVERIFIED.
    "replace": [_r("file_path", "old_string", "new_string")],
    # Cline / Roo Code. write_to_file(path, content) and apply_diff(path,
    # diff: SEARCH/REPLACE blocks) are VERIFIED in Roo (write_to_file.ts,
    # apply_diff.ts); Cline's replace_in_file(path, diff) is UNVERIFIED.
    # Roo's search_replace / edit (file_path, old_string, new_string) and
    # apply_patch(patch) are VERIFIED (search_replace.ts, edit.ts,
    # apply_patch.ts).
    "write_to_file": [_w("path", "content")],
    "apply_diff": [{"kind": "diff", "path": "path", "diff": "diff"}],
    "replace_in_file": [{"kind": "diff", "path": "path", "diff": "diff"}],
    "search_replace": [_r("file_path", "old_string", "new_string")],
    "edit": [_r("file_path", "old_string", "new_string"),     # Roo, VERIFIED
             _r("filePath", "oldString", "newString")],       # OpenCode
    # OpenCode: write(filePath, content), edit (above). UNVERIFIED.
    "write": [_w("filePath", "content"), _w("file_path", "content"),
              _w("path", "content")],
    # Codex CLI: apply_patch(input) -- UNVERIFIED; Roo: apply_patch(patch)
    # VERIFIED.
    "apply_patch": [{"kind": "patch", "patch": "input"},
                    {"kind": "patch", "patch": "patch"}],
    # Continue: create_new_file(filepath, contents). UNVERIFIED.
    "create_new_file": [_w("filepath", "contents")],
}

# ======================================================== SHAPE FALLBACK ===

_PATH_KEY = re.compile(r"^(?:path|file|filename|file_name|filepath|file_path|"
                       r"filePath|fileName|target|target_file|target_path|"
                       r"dest|destination|output_path|uri)$", re.I)
_CONTENT_KEY = re.compile(r"^(?:content|contents|file_text|text|code|source|"
                          r"body|new_content|file_content|fileContent)$", re.I)
_OLD_KEY = re.compile(r"^(?:old|old_str|old_string|oldString|old_text|oldText|"
                      r"old_code|search|original)$", re.I)
_NEW_KEY = re.compile(r"^(?:new|new_str|new_string|newString|new_text|newText|"
                      r"new_code|replace|replacement)$", re.I)
_PATCH_KEY = re.compile(r"^(?:patch|diff|input|changes|edits?)$", re.I)


def language_of(path: str | None) -> str | None:
    """The canonical language of a path's extension, or None."""
    if not isinstance(path, str) or not path.strip() or "\n" in path:
        return None
    ext = os.path.splitext(path.strip().rstrip("/\\"))[1]
    return code_check.normalize_language(ext) if ext else None


def _first(args: dict, rx) -> str | None:
    for k, v in args.items():
        if rx.match(k) and isinstance(v, str):
            return k
    return None


def looks_like_patch(text: str) -> bool:
    if not isinstance(text, str):
        return False
    return bool(re.search(r"^\*\*\* (?:Begin Patch|Add File:|Update File:)",
                          text, re.M)
                or (re.search(r"^--- ", text, re.M)
                    and re.search(r"^\+\+\+ ", text, re.M)
                    and re.search(r"^@@", text, re.M))
                or _SR_BLOCK.search(text))


def shape_of(args: dict) -> dict | None:
    """The shape a call's arguments have, for a name the table does not
    know: a path with a code extension plus a content string; a path plus an
    old/new pair; a list of old/new pairs; or a patch/diff string."""
    if not isinstance(args, dict):
        return None
    pk = _first(args, _PATH_KEY)
    if pk is None:
        # A path under an unusual key: any short one-line string value with
        # a code extension.
        for k, v in args.items():
            if isinstance(v, str) and len(v) < 400 and language_of(v):
                pk = k
                break
    ck = _first(args, _CONTENT_KEY)
    if pk and ck and ck != pk:
        return _w(pk, ck)
    ok, nk = _first(args, _OLD_KEY), _first(args, _NEW_KEY)
    if pk and ok and nk:
        return _r(pk, ok, nk)
    for k, v in args.items():
        if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
            o = next((kk for kk in v[0] if _OLD_KEY.match(kk)), None)
            n = next((kk for kk in v[0] if _NEW_KEY.match(kk)), None)
            if pk and o and n:
                return {"kind": "edits", "path": pk, "list": k, "old": o,
                        "new": n}
    for k, v in args.items():
        if isinstance(v, str) and looks_like_patch(v):
            return ({"kind": "diff", "path": pk, "diff": k} if pk
                    else {"kind": "patch", "patch": k})
    return None

# ======================================================== PATCH PARSING ===


_SR_BLOCK = re.compile(
    r"^(?:<<<<<<< SEARCH>?|------- SEARCH)[ \t]*\n(.*?)^=======[ \t]*\n(.*?)"
    r"^(?:>>>>>>> REPLACE|\+\+\+\+\+\+\+ REPLACE)[ \t]*$", re.M | re.S)


def search_replace_blocks(text: str) -> list[tuple[str, str]]:
    """(old, new) of each SEARCH/REPLACE block -- Roo's apply_diff (VERIFIED:
    multi-search-replace.ts, with its `:start_line:` and `-------` lines) and
    Cline's replace_in_file."""
    out = []
    for m in _SR_BLOCK.finditer(text or ""):
        old = m.group(1)
        old = re.sub(r"\A(?::start_line:\s*\d+[ \t]*\n)?(?:-------[ \t]*\n)?",
                     "", old)
        out.append((old.rstrip("\n"), m.group(2).rstrip("\n")))
    return out


def _hunks(lines: list[str]) -> list[tuple[str, str]]:
    """(old, new) per hunk from ' ', '-', '+' lines split on '@@'."""
    out, old, new = [], [], []

    def flush():
        if old or new:
            out.append(("\n".join(old), "\n".join(new)))
        old.clear()
        new.clear()
    for ln in lines:
        if ln.startswith("@@"):
            flush()
            continue
        if ln.startswith("\\"):
            continue
        tag, body = (ln[:1], ln[1:]) if ln else (" ", "")
        if tag == " ":
            old.append(body)
            new.append(body)
        elif tag == "-":
            old.append(body)
        elif tag == "+":
            new.append(body)
    flush()
    return out


def parse_patch(text: str) -> list[dict]:
    """[{path, op: add|update|delete, content (add), hunks [(old, new)]}] of
    a V4A patch ("*** Begin Patch"; Codex / Roo apply_patch) or a unified
    diff. Pure text parsing of a flat, line-oriented format."""
    text = (text or "").replace("\r\n", "\n")
    files: list[dict] = []
    if re.search(r"^\*\*\* (?:Begin Patch|Add File:|Update File:)", text, re.M):
        cur = None
        for ln in text.split("\n"):
            m = re.match(r"^\*\*\* (Add|Update|Delete) File: (.+)$", ln)
            if m:
                cur = {"path": m.group(2).strip(), "op": m.group(1).lower(),
                       "lines": []}
                files.append(cur)
                continue
            if ln.startswith("*** ") or cur is None:
                continue
            cur["lines"].append(ln)
    else:
        cur = None
        lines = text.split("\n")
        for i, ln in enumerate(lines):
            if ln.startswith("--- ") and i + 1 < len(lines) \
                    and lines[i + 1].startswith("+++ "):
                a = ln[4:].split("\t")[0].strip()
                b = lines[i + 1][4:].split("\t")[0].strip()
                op = ("add" if a == "/dev/null" else
                      "delete" if b == "/dev/null" else "update")
                path = re.sub(r"^[ab]/", "", b if op != "delete" else a)
                cur = {"path": path, "op": op, "lines": []}
                files.append(cur)
                continue
            if ln.startswith("+++ ") or cur is None:
                continue
            cur["lines"].append(ln)
    for f in files:
        lines = f.pop("lines")
        if f["op"] == "add":
            f["content"] = "\n".join(ln[1:] for ln in lines
                                     if ln.startswith("+"))
            f["hunks"] = []
        elif f["op"] == "update":
            f["hunks"] = _hunks(lines)
        else:
            f["hunks"] = []
    return files

# ========================================================== EXTRACTION ====


def _units_of(name: str, args: dict, shape: dict) -> list[dict]:
    """The code units one call carries under `shape`: whole files and
    edits, each with its path and language. [] when the keys are absent."""
    k = shape["kind"]
    units: list[dict] = []

    def s(key):
        v = args.get(key) if key else None
        return v if isinstance(v, str) else None
    if k == "editor":
        cmd, path = s("command"), s("path")
        if cmd == "create" and s("file_text") is not None:
            units.append({"kind": "file", "path": path, "code": s("file_text"),
                          "key": ("file_text",), "set": ["file_text"]})
        elif cmd == "str_replace" and s("new_str") is not None:
            units.append({"kind": "edit", "path": path, "old": s("old_str"),
                          "code": s("new_str"), "set": ["new_str"]})
        elif cmd == "insert" and s("new_str") is not None:
            units.append({"kind": "edit", "path": path, "old": None,
                          "code": s("new_str"), "set": ["new_str"]})
    elif k == "write":
        if s(shape["path"]) is not None and s(shape["content"]) is not None:
            units.append({"kind": "file", "path": s(shape["path"]),
                          "code": s(shape["content"]),
                          "key": (shape["content"],),
                          "set": [shape["content"]]})
    elif k == "replace":
        if s(shape["new"]) is not None and shape["path"] in args:
            units.append({"kind": "edit", "path": s(shape["path"]),
                          "old": s(shape["old"]), "code": s(shape["new"]),
                          "set": [shape["new"]]})
    elif k == "edits":
        lst = args.get(shape["list"])
        if isinstance(lst, list) and shape["path"] in args:
            for j, e in enumerate(lst):
                if isinstance(e, dict) and isinstance(e.get(shape["new"]), str):
                    old = e.get(shape["old"])
                    units.append({"kind": "edit", "path": s(shape["path"]),
                                  "old": old if isinstance(old, str) else None,
                                  "code": e[shape["new"]],
                                  "set": [shape["list"], j, shape["new"]]})
    elif k == "diff":
        body, path = s(shape["diff"]), s(shape.get("path"))
        if body is not None:
            pairs = search_replace_blocks(body)
            if not pairs:
                pairs = [h for f in parse_patch(body) for h in f["hunks"]]
            for old, new in pairs:
                units.append({"kind": "edit", "path": path, "old": old,
                              "code": new})
    elif k == "patch":
        body = s(shape["patch"])
        if body is not None:
            for f in parse_patch(body):
                if f["op"] == "add":
                    # Inside a patch: checked as a file, never reformatted
                    # (a formatter's output cannot be written back into a
                    # diff without re-deriving it).
                    units.append({"kind": "file", "path": f["path"],
                                  "code": f["content"], "key": None})
                for old, new in f["hunks"]:
                    units.append({"kind": "edit", "path": f["path"],
                                  "old": old, "code": new})
    for u in units:
        u["language"] = language_of(u.get("path"))
    return units


def detect(call: dict) -> dict:
    """{name, detected: table|shape|None, units, args, unknown} for one
    client tool call. `unknown` is set when nothing recognised the call but
    it carries a code-sized string."""
    fn = (call or {}).get("function") or {}
    name = fn.get("name") or ""
    raw = fn.get("arguments")
    out = {"name": name, "detected": None, "units": [], "args": None,
           "unknown": None}
    try:
        args = json.loads(raw) if isinstance(raw, str) else (raw or {})
    except ValueError:
        out["unparsed"] = True
        return out
    if not isinstance(args, dict):
        return out
    out["args"] = args
    for shape in KNOWN.get(name, []):
        units = _units_of(name, args, shape)
        if units:
            out.update(detected="table", units=units)
            return out
    shape = shape_of(args)
    if shape:
        units = _units_of(name, args, shape)
        if units:
            out.update(detected="shape", units=units)
            return out
    for k, v in args.items():
        if (isinstance(v, str) and len(v) >= UNKNOWN_MIN_CHARS
                and v.count("\n") + 1 >= UNKNOWN_MIN_LINES):
            out["unknown"] = {"name": name, "key": k, "chars": len(v)}
            break
    return out

# ============================================================= CHECKING ===


def check_file(code: str, lang: str) -> dict:
    """{errors, syntax, lint, formatted, formatter, defects} for a whole
    file. `errors` blocks; `defects` (code_check's loop heuristic) never
    does -- it is a heuristic, and a false positive would cost a round."""
    res = code_check.check(code, lang)
    syntax = list(res["syntax_errors"])
    lint: list[dict] = []
    if not syntax:
        fp = code_check.formatter_parse_error(res)
        if fp:
            syntax.append(fp)
        else:
            lint = code_check.lint_errors(code, lang)
    formatted = res.get("formatted") if not (syntax or lint) else None
    ds = res.get("diff_summary") or {}
    return {"errors": syntax + lint, "syntax": len(syntax), "lint": len(lint),
            "formatted": formatted if ds.get("changed_lines") else None,
            "changed_lines": ds.get("changed_lines") or 0,
            "formatter": (res.get("formatter") or {}).get("name"),
            "defects": len(res.get("defects") or [])}


def check_edit(old: str | None, new: str, lang: str) -> dict:
    """{status: ok|errors|fragment, errors}. See the module docstring."""
    try:
        errs = code_check.check(new, lang, run_format=False)["syntax_errors"]
    except Exception:                                            # noqa: BLE001
        return {"status": "fragment", "errors": []}
    if not errs:
        return {"status": "ok", "errors": []}
    if old is not None and old.strip():
        try:
            old_ok = not code_check.check(old, lang,
                                          run_format=False)["syntax_errors"]
        except Exception:                                        # noqa: BLE001
            old_ok = False
        if old_ok:
            return {"status": "errors", "errors": errs}
    return {"status": "fragment", "errors": errs}


def review(calls: list[dict]) -> dict:
    """Check every code unit in a generation's client calls.

    {calls: [per-call detection + results], units, errors, blocked: [call
    indexes], key, unknown: [...]}. `key` identifies the code, so the same
    calls handed back after feedback are recognised."""
    h = hashlib.sha1()
    per: list[dict] = []
    total = 0
    blocked: list[int] = []
    unknown: list[dict] = []
    for i, c in enumerate(calls or []):
        d = detect(c)
        if d.get("unknown"):
            unknown.append(d["unknown"])
        n_call = 0
        for u in d["units"]:
            if not u["language"] or not (u.get("code") or "").strip():
                u["result"] = None
                continue
            h.update(f"{u.get('path')}\x00{u['code']}\x00".encode("utf-8"))
            if u["kind"] == "file":
                r = check_file(u["code"], u["language"])
                u["result"] = r
                n_call += len(r["errors"])
            else:
                r = check_edit(u.get("old"), u["code"], u["language"])
                u["result"] = r
                if r["status"] == "errors":
                    n_call += len(r["errors"])
        if n_call:
            blocked.append(i)
        total += n_call
        per.append(d)
    checked = [u for d in per for u in d["units"] if u.get("result")]
    return {"calls": per, "units": checked, "errors": total,
            "blocked": blocked, "unknown": unknown,
            "key": h.hexdigest() if checked else None}


def _where(e: dict) -> str:
    return f"line {e.get('line', 1)}, col {e.get('col', 1)}: {e['message']}"

# ============================================================ THE FLOW ====
#
# Since 2026-09-24 (operator) main never sees a failed attempt. The rounds
# used to run ON MAIN: the model got its own calls back as "NOT EXECUTED"
# tool results and wrote them again, each round a turn in the conversation's
# context that the client never received -- so the next request's history
# no longer matched what the slot held. Now:
#
#   check    the final generation's client calls are reviewed (`check`)
#   fix      at `high` and up (tiers.repair_on), each blocking unit that can
#            be written back goes to the second brain's fixup job
#            (shomen.run("fixup")) with ONLY its code, its errors and the
#            user's request; an ACCEPTED repair is written back
#            (`apply`), anything else leaves the model's content as it was
#   note     `finish` writes the one-line note in the fold-back phrases
#            (shomen.PHRASES): "Verified ...", "Repaired ...", or at
#            `medium`, where nothing fixes, "Checked ...: N problems; sent
#            as written"
#
# The proxy then warms the conversation's slot with the turn exactly as the
# client will store it (proxy._warm), so the next request finds the note and
# the fixed call cached.


def state(check: bool, fix: bool = False) -> dict | None:
    """The x_yamadori.tool_code record, or None when the check is off.
    `fix`: the fixup job runs (high and up); off, errors are only noted."""
    if not check:
        return None
    return {"enabled": True, "fix": bool(fix), "files": [], "language": [],
            "errors_before": None, "errors_after": None, "rounds": 0,
            "formatted": False, "stopped": None, "unknown": [],
            "fixup": None, "seed": None, "_last": None, "_first": {}}


def _client_calls(msg: dict, ours: set[str]) -> list[dict]:
    return [c for c in (msg.get("tool_calls") or [])
            if ((c.get("function") or {}).get("name")) not in ours]


def check(rec: dict | None, msg: dict, finish: str | None,
          ours: set[str]) -> dict | None:
    """Review a final generation's CLIENT calls. Returns the review, or None
    when there is nothing to check; the record says what was found."""
    if rec is None:
        return None
    calls = _client_calls(msg, ours)
    if not calls or finish not in ("tool_calls", "stop"):
        return None
    rv = review(calls)
    rec["_last"] = {"msg": msg, "review": rv, "calls": calls}
    for u in rv["unknown"]:
        if u not in rec["unknown"]:
            rec["unknown"].append(u)
            print(f"  tool_code: unrecognised client tool {u['name']!r} carries "
                  f"a {u['chars']}-char string in {u['key']!r}; not checked "
                  f"(add it to tool_code.KNOWN if it writes files)", flush=True)
    for d in rv["calls"]:
        if d["detected"] == "shape" and d["units"]:
            print(f"  tool_code: {d['name']!r} recognised by argument shape, "
                  f"not by name (tool_code.KNOWN could list it)", flush=True)
    if not rv["units"]:
        rec["stopped"] = rec["stopped"] or "no_code"
        return rv
    for u in rv["units"]:
        rec["_first"].setdefault(u.get("path"), _count(u))
    rec["errors_before"] = rv["errors"]
    rec["errors_after"] = rv["errors"]
    rec["stopped"] = ("clean" if not rv["errors"] else
                      "fixup" if rec["fix"] else "noted")
    return rv


def fixable(rv: dict | None) -> list[dict]:
    """The blocking units the fixup job can correct: each with its call
    index, where its code is written back (`set`), its code and errors. A
    unit inside a diff or a multi-file patch has no write-back and is only
    noted."""
    out = []
    for i, d in enumerate((rv or {}).get("calls") or []):
        if i not in rv["blocked"]:
            continue
        for u in d["units"]:
            r = u.get("result") or {}
            errs = (r.get("errors") if u["kind"] == "file"
                    else r.get("errors") if r.get("status") == "errors"
                    else None)
            if errs and u.get("set"):
                out.append({"call": i, "set": u["set"], "path": u.get("path"),
                            "language": u["language"], "kind": u["kind"],
                            "code": u["code"], "old": u.get("old"),
                            "errors": errs})
    return out


def recheck(unit: dict, code: str) -> list[dict]:
    """The errors left in a fixed version of `unit`'s code (the fixup job's
    `check`)."""
    if unit.get("kind") == "file":
        return check_file(code, unit["language"])["errors"]
    r = check_edit(unit.get("old"), code, unit["language"])
    return r["errors"] if r["status"] == "errors" else []


def apply(rec: dict, rv: dict, calls: list[dict], fixed: list[dict],
          job: dict | None = None) -> int:
    """Write each ACCEPTED repair into its call's arguments, BEFORE the call
    is forwarded. Returns how many were written.

    A unit is written only when the fixup job accepted its version
    (shomen.fix_rejection: a closed fence, no finish_reason "length", no
    errors left, plausibly complete) and it differs from the model's. Any
    other unit keeps the model's own content, untouched, and the reason is
    recorded (pre-deploy review, 2026-09-24: a failed or truncated fix-up
    used to overwrite the model's file)."""
    n = 0
    rejected = []
    for f in fixed or []:
        if f.get("rejected"):
            rejected.append({"path": f.get("path"), "why": f["rejected"]})
        if not (f.get("changed") and f.get("accepted")
                and not f.get("errors_after")):
            continue
        d = rv["calls"][f["call"]]
        target = d["args"]
        path = f["set"]
        for k in path[:-1]:
            target = target[k]
        target[path[-1]] = f["code"]
        calls[f["call"]]["function"]["arguments"] = json.dumps(
            d["args"], ensure_ascii=False)
        n += 1
    rec["rounds"] = int((job or {}).get("rounds") or 0)
    rec["seed"] = (job or {}).get("seed")
    rec["fixup"] = {"units": len(fixed or []), "written": n,
                    "ok": bool((job or {}).get("ok")),
                    "skipped": (job or {}).get("skipped"),
                    "rejected": rejected,
                    # Lines that only repeated the job's seed word, removed
                    # from the repaired code (#13, fanout.strip_seed_echo).
                    "seed_echo_stripped": sum(
                        int(u.get("seed_echo_stripped") or 0)
                        for u in fixed or [])}
    # finish() re-reviews the calls as they now are.
    rec["_last"] = None
    return n


def _count(u: dict) -> dict:
    r = u.get("result") or {}
    if u["kind"] == "file":
        return {"syntax": r.get("syntax", 0), "lint": r.get("lint", 0)}
    return {"syntax": len(r.get("errors") or []) if r.get("status") == "errors"
            else 0, "lint": 0}


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def finish(rec: dict | None, msg: dict, ours: set[str]) -> str:
    """The one-line note to send before the final calls ("" when nothing
    was checked). The calls go as written (a fix-up's repair aside, applied
    earlier by `apply`); formatter output is reported, never applied
    (#24)."""
    if rec is None:
        return ""
    import shomen
    ph = shomen.PHRASES
    calls = _client_calls(msg, ours)
    if not calls:
        return ""
    last = rec.get("_last") or {}
    rv = last.get("review") if last.get("msg") is msg else None
    if rv is None:
        rv = review(calls)
        rec["_last"] = {"msg": msg, "review": rv, "calls": calls}
        for u in rv["unknown"]:
            if u not in rec["unknown"]:
                rec["unknown"].append(u)
        for u in rv["units"]:
            rec["_first"].setdefault(u.get("path"), _count(u))
        if rv["units"]:
            if rec["errors_before"] is None:
                rec["errors_before"] = rv["errors"]
            rec["errors_after"] = rv["errors"]
            if rec["stopped"] in (None, "fixup"):
                rec["stopped"] = ("fixed" if rec["stopped"] == "fixup"
                                  and not rv["errors"] else
                                  "clean" if not rv["errors"] else
                                  "not_fixed")
    if not rv["units"]:
        return ""
    notes: list[str] = []
    files: list[dict] = []
    langs: set[str] = set()
    for d in rv["calls"]:
        for u in d["units"]:
            r = u.get("result")
            if not r:
                continue
            langs.add(u["language"])
            path = u.get("path") or "(no path)"
            label = f"{path} ({u['language']}" + (
                ", edit" if u["kind"] == "edit" else "") + ")"
            first = rec["_first"].get(u.get("path")) or {}
            entry = {"path": path, "language": u["language"],
                     "kind": u["kind"], "detected": d["detected"],
                     "tool": d["name"], "errors_after": 0,
                     "errors_before": first.get("syntax", 0)
                     + first.get("lint", 0), "formatted": False,
                     "flagged": None}
            bits: list[str] = []
            if u["kind"] == "file":
                entry["errors_after"] = len(r["errors"])
                if not r["errors"] and r.get("formatted"):
                    # INFORMATION ONLY (#24): the content goes as written.
                    entry["format_would_change"] = r["changed_lines"]
                    bits.append(f"{r['formatter']} would change "
                                f"{_plural(r['changed_lines'], 'line')}; "
                                f"sent as written")
            elif r["status"] == "errors":
                entry["errors_after"] = len(r["errors"])
            elif r["status"] == "fragment":
                entry["flagged"] = "fragment"
            errs = r["errors"] if (u["kind"] == "file"
                                   or r["status"] == "errors") else []
            fixed_s, fixed_l = first.get("syntax", 0), first.get("lint", 0)
            if errs:
                e = errs[0]
                head = (f"{ph['checked']} {label}: "
                        f"{_plural(len(errs), 'problem')} (line "
                        f"{e.get('line')}: {e['message'][:80]})")
                tail = ("; still there after "
                        f"{_plural(rec['rounds'], 'round')} of repair"
                        if rec["fix"] and rec["rounds"] else
                        "" if rec["fix"] else
                        "; not repaired at this effort")
                # TRUE by construction: apply() never writes a version that
                # still has errors, so what goes is the model's own content.
                why = next((x["why"] for x in
                            (rec.get("fixup") or {}).get("rejected") or []
                            if x.get("path") == u.get("path")), None)
                if why and rec["fix"] and rec["rounds"]:
                    tail += f" (the repair was not used: {why})"
                note = head + tail + "; sent as written"
            elif entry["flagged"] == "fragment":
                note = (f"{ph['checked']} {label}: an edit fragment, not "
                        f"checkable on its own")
            elif fixed_s or fixed_l:
                what = " and ".join(x for x in (
                    _plural(fixed_s, "syntax error") if fixed_s else "",
                    _plural(fixed_l, "lint error") if fixed_l else "") if x)
                note = (f"{ph['repaired']} {label}: {what} fixed in "
                        f"{_plural(rec['rounds'], 'round')}")
            else:
                note = (f"{ph['verified']} {label}: parses"
                        + (", lint clean" if u["language"] == "python"
                           and u["kind"] == "file" else ""))
            if bits:
                note += "; " + "; ".join(bits)
            files.append(entry)
            notes.append(note + ".")
    rec["files"] = files
    rec["language"] = sorted(langs)
    if not notes:
        return ""
    # No seed line: the note is mechanical (see WHAT THE CLIENT RECEIVES).
    return " ".join(notes)


def public(rec: dict | None) -> dict | None:
    """The record for x_yamadori: bookkeeping stripped."""
    if rec is None:
        return None
    return {k: v for k, v in rec.items() if not k.startswith("_")}
