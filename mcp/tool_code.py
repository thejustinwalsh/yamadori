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
off (finish_reason "length") and it has no errors left
(shomen.fix_rejection); otherwise the model's own content goes,
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
(`x_yamadori.tool_code.files[].format_would_change`, never in the note: #34);
its output is never applied. The code is
never modified after the client has the call. A one-line note goes out as
content just before the tool calls, in the fold-back phrases
(shomen.PHRASES), saying plainly what was sent: "Repaired js/game.js
(javascript): 2 syntax errors fixed in 1 round (the repaired file was sent;
formatting left as written)." / "Verified js/ui.js (javascript): parses;
sent as written." / "Checked ...: N problems ...; sent as written" when
problems remain --
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
             _r("filePath", "oldString", "newString"),        # OpenCode
             # Pi: edit(path, edits[{oldText, newText}]). VERIFIED:
             # @earendil-works/pi-coding-agent 0.87.1,
             # dist/core/tools/edit.js:10-21 (replaceEditSchema, editSchema).
             # Its prepareEditArguments (:43-73) also takes `edits` as a
             # JSON string or one object, and a legacy top-level
             # oldText/newText -- the next two rows (_units_of reads the
             # string and the object).
             {"kind": "edits", "path": "path", "list": "edits",
              "old": "oldText", "new": "newText"},
             _r("path", "oldText", "newText")],               # Pi legacy
    # OpenCode: write(filePath, content), edit (above). UNVERIFIED.
    # Pi: write(path, content). VERIFIED: pi-coding-agent 0.87.1,
    # dist/core/tools/write.js:8-11 (writeSchema).
    "write": [_w("filePath", "content"), _w("file_path", "content"),
              _w("path", "content")],
    # Codex CLI: apply_patch(input) -- UNVERIFIED; Roo: apply_patch(patch)
    # VERIFIED.
    "apply_patch": [{"kind": "patch", "patch": "input"},
                    {"kind": "patch", "patch": "patch"}],
    # Continue: create_new_file(filepath, contents). UNVERIFIED.
    "create_new_file": [_w("filepath", "contents")],
}

# ============================================================ READ TOOLS ===
#
# The client tools that READ one file, for the unchanged-read situation
# (mcp/progress.py, #54 in docs/SELF-IMPROVEMENT-LOG.md): name -> (the path
# key, the range keys). Only these are read tools; a shell `cat` is not.
# VERIFIED rows are read from the harness's own tool schema in
# bench/harness_shapes (captured requests); the rest from documentation.
READ_KNOWN: dict[str, tuple[str, tuple[str, ...]]] = {
    # Hermes read_file(path, offset, limit): VERIFIED (bench/harness_shapes/
    # hermes). MCP filesystem read_file(path): UNVERIFIED, same key.
    "read_file": ("path", ("offset", "limit")),
    # OpenCode read(filePath, offset, limit) and Pi read(path, offset,
    # limit): VERIFIED (bench/harness_shapes/opencode, /pi).
    "read": ("filePath|path", ("offset", "limit")),
    # Claude Code Read(file_path, offset, limit): VERIFIED against the schema
    # of the Claude Code instance that wrote this row.
    "Read": ("file_path", ("offset", "limit")),
    # MCP filesystem read_text_file(path, head, tail): UNVERIFIED.
    "read_text_file": ("path", ("head", "tail")),
}


def read_target(call: dict) -> dict | None:
    """{path, range, tool} when `call` is a known read tool reading one file
    (READ_KNOWN); None otherwise. The path is normalised (backslashes,
    a leading ./), the range is the range keys the call set."""
    fn = (call or {}).get("function") or {}
    name = fn.get("name") or ""
    spec = READ_KNOWN.get(name)
    if spec is None:
        return None
    raw = fn.get("arguments")
    try:
        args = json.loads(raw) if isinstance(raw, str) else (raw or {})
    except ValueError:
        return None
    if not isinstance(args, dict):
        return None
    keys, rng_keys = spec
    path = next((args[k] for k in keys.split("|")
                 if isinstance(args.get(k), str) and args[k].strip()), None)
    if path is None:
        return None
    p = path.strip().replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    rng = {k: args[k] for k in rng_keys if args.get(k) not in (None, "")}
    return {"path": p, "range": rng, "tool": name}

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


def _units_of(name: str, args: dict, shape: dict,
              whole_pages: bool = False) -> list[dict]:
    """The code units one call carries under `shape`: whole files and
    edits, each with its path and language. [] when the keys are absent.
    `whole_pages`: an .html write stays ONE unit (the page) instead of its
    inline scripts' units -- for the callers that ask WHICH FILE a call
    writes (progress, deep's rewrite signals), not what code to check: a
    page with only `src` scripts has no script unit, and its write was
    invisible to them (v0f-V0: index.html's write was not progress)."""
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
        # Pi also accepts the list as a JSON string, or one object
        # (edit.js prepareEditArguments). Checked the same; a repair has no
        # place to go back into a string, so those units are only noted.
        where = True
        if isinstance(lst, str):
            try:
                lst, where = json.loads(lst), False
            except ValueError:
                lst = None
        if isinstance(lst, dict):
            lst, where = [lst], False
        if isinstance(lst, list) and shape["path"] in args:
            for j, e in enumerate(lst):
                if isinstance(e, dict) and isinstance(e.get(shape["new"]), str):
                    old = e.get(shape["old"])
                    units.append({"kind": "edit", "path": s(shape["path"]),
                                  "old": old if isinstance(old, str) else None,
                                  "code": e[shape["new"]],
                                  "set": [shape["list"], j, shape["new"]]
                                  if where else None})
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
    out: list[dict] = []
    for u in units:
        u["language"] = language_of(u.get("path"))
        if not whole_pages and u["kind"] == "file" and u[
                "language"] is None and is_html(u.get("path")):
            out += script_units(u)
        else:
            out.append(u)
    return out


# ============================================================ HTML PAGES ===
#
# An .html write carries its code in inline <script> elements -- where a
# three.js / TSL page lives (2026-09-26, docs/HARNESS-PI.md gap 4: Pi's
# canvas.html went out unchecked, `stopped: "no_code"`). Each inline script
# of a JavaScript type (none, a JavaScript MIME type, or `module`) is a unit
# of its own, checked as a whole JavaScript file with the same rules: it
# blocks only when it does not parse. Found with tree-sitter's HTML grammar
# (PROTOCOL rule 8), never a regex; a `src` script has no code here, and a
# script of another type (an import map, a shader: `x-shader/x-vertex`) is
# not JavaScript. A repair is written back into the page at the script's
# own span (`span`, character offsets into the content), nothing else of
# the page touched.
HTML_EXTS = (".html", ".htm")
JS_SCRIPT_TYPES = frozenset({"", "module", "text/javascript",
                             "application/javascript", "text/ecmascript",
                             "application/ecmascript",
                             "application/x-javascript"})


def is_html(path: str | None) -> bool:
    return isinstance(path, str) and path.strip().lower().endswith(HTML_EXTS)


def html_scripts(html: str) -> list[dict]:
    """[{code, start, end, module}] for each inline JavaScript <script> in
    `html`; `start`/`end` are CHARACTER offsets of its text. [] when the
    HTML grammar is not available."""
    try:
        tree = code_check._parser("html").parse(html.encode("utf-8"))
    except Exception:                                            # noqa: BLE001
        return []
    b = html.encode("utf-8")

    def chars(i: int) -> int:
        return len(b[:i].decode("utf-8", "replace"))

    def attrs(tag) -> dict:
        out = {}
        for a in tag.children:
            if a.type != "attribute":
                continue
            name = next((c for c in a.children if c.type == "attribute_name"),
                        None)
            val = next((c for c in a.children if c.type in (
                "attribute_value", "quoted_attribute_value")), None)
            if name is None:
                continue
            v = ""
            if val is not None:
                inner = next((c for c in val.children
                              if c.type == "attribute_value"), val)
                v = b[inner.start_byte:inner.end_byte].decode("utf-8",
                                                              "replace")
            out[b[name.start_byte:name.end_byte].decode().lower()] = v
        return out

    found = []
    stack = [tree.root_node]
    while stack:
        n = stack.pop()
        if n.type == "script_element":
            tag = next((c for c in n.children if c.type == "start_tag"), None)
            raw = next((c for c in n.children if c.type == "raw_text"), None)
            a = attrs(tag) if tag is not None else {}
            kind = a.get("type", "").strip().lower()
            if raw is not None and "src" not in a and kind in JS_SCRIPT_TYPES:
                s, e = chars(raw.start_byte), chars(raw.end_byte)
                if html[s:e].strip():
                    found.append({"code": html[s:e], "start": s, "end": e,
                                  "module": kind == "module"})
            continue
        stack.extend(reversed(n.children))
    return sorted(found, key=lambda x: x["start"])


def script_units(u: dict) -> list[dict]:
    """A whole .html file's unit as its inline scripts' units (JavaScript,
    checked as files). [] when it has none: the page carries no code this
    check reads."""
    out = []
    for i, sc in enumerate(html_scripts(u.get("code") or "")):
        out.append({"kind": "file", "path": u.get("path"),
                    "code": sc["code"], "key": None,
                    "set": u.get("set"), "span": [sc["start"], sc["end"]],
                    "script": i + 1, "module": sc["module"],
                    "language": "javascript"})
    return out


def detect(call: dict, whole_pages: bool = False) -> dict:
    """{name, detected: table|shape|None, units, args, unknown} for one
    client tool call. `unknown` is set when nothing recognised the call but
    it carries a code-sized string. `whole_pages`: see _units_of (which
    files a call writes, rather than which code to check)."""
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
        units = _units_of(name, args, shape, whole_pages)
        if units:
            out.update(detected="table", units=units)
            return out
    shape = shape_of(args)
    if shape:
        units = _units_of(name, args, shape, whole_pages)
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
            "fixup": None, "seed": None,
            "_last": None, "_first": {}}


# FIX-UP SCOPE (#56: project files only, by progress.is_project and its
# scratch-name table) was REMOVED 2026-09-27 (docs/CONSTANTS-AUDIT.md: the
# table named one Octopus run's files). Every blocking unit is repaired.


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
        rec["_first"].setdefault(_ukey(u), _count(u))
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
                            "span": u.get("span"), "errors": errs})
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
    errors left) and it differs from the model's. Any
    other unit keeps the model's own content, untouched, and the reason is
    recorded (pre-deploy review, 2026-09-24: a failed or truncated fix-up
    used to overwrite the model's file)."""
    n = 0
    rejected = []
    # A page's scripts are spliced back last-first, so an earlier script's
    # span still holds when a later one's repair changed length.
    order = sorted(fixed or [], key=lambda f: -((f.get("span") or [0])[0]))
    for f in order:
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
        if f.get("span"):
            # An inline <script> of an .html page: only its own text, the
            # line breaks around it as the page had them.
            a, b = f["span"]
            cur = target[path[-1]]
            orig = cur[a:b]
            lead = orig[:len(orig) - len(orig.lstrip("\r\n"))]
            trail = orig[len(orig.rstrip()):]
            target[path[-1]] = (cur[:a] + lead
                                + f["code"].lstrip("\r\n").rstrip()
                                + trail + cur[b:])
        else:
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


def _ukey(u: dict) -> str | None:
    """The key a unit's first counts are kept under: its path, and for an
    inline script of an .html page, which script (they share the path)."""
    p = u.get("path")
    return f"{p}#script{u['script']}" if u.get("script") else p


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
            rec["_first"].setdefault(_ukey(u), _count(u))
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
                ", edit" if u["kind"] == "edit" else "") + (
                f", script {u['script']}" if u.get("script") else "") + ")"
            first = rec["_first"].get(_ukey(u)) or {}
            entry = {"path": path, "language": u["language"],
                     "kind": u["kind"], "detected": d["detected"],
                     "tool": d["name"], "errors_after": 0,
                     "errors_before": first.get("syntax", 0)
                     + first.get("lint", 0), "formatted": False,
                     "flagged": None}
            if u["kind"] == "file":
                entry["errors_after"] = len(r["errors"])
                if not r["errors"] and r.get("formatted"):
                    # RECORD ONLY (#24): the content goes as written, and
                    # the note does not mention the formatter (#34: "fixed
                    # ...; prettier would change 69 lines; sent as written"
                    # read as a contradiction, and the model reads notes).
                    entry["format_would_change"] = r["changed_lines"]
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
                # NO NOTE (2026-09-26, docs/HARNESS-PI.md gap 4). "Checked"
                # marks problems that remain (AGENTS.md, the fold-back
                # phrases); a fragment has none found -- its replacement and
                # the text it replaces fail alike out of context (Pi's
                # `return 'howdy ' + name`). The note said "Checked ...: an
                # edit fragment, not checkable on its own" and the model
                # wrote the next line in its image ("Ran hello.py (python,
                # edit): ...", #36). Recorded in x_yamadori.tool_code.files
                # (`flagged: "fragment"`) only.
                files.append(entry)
                continue
            elif fixed_s or fixed_l:
                what = " and ".join(x for x in (
                    _plural(fixed_s, "syntax error") if fixed_s else "",
                    _plural(fixed_l, "lint error") if fixed_l else "") if x)
                # Say plainly WHAT went out (#34): the repaired version, and
                # for a file that its formatting is the model's own.
                note = (f"{ph['repaired']} {label}: {what} fixed in "
                        f"{_plural(rec['rounds'], 'round')} "
                        + ("(the repaired file was sent; formatting left "
                           "as written)" if u["kind"] == "file" else
                           "(the repaired edit was sent)"))
            else:
                note = (f"{ph['verified']} {label}: parses"
                        + (", lint clean" if u["language"] == "python"
                           and u["kind"] == "file" else "")
                        + ("; sent as written" if u["kind"] == "file"
                           else ""))
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


# ======================================================= THE IMAGE GUARD ===
#
# IMAGE DATA IN AN IMAGE TOOL'S ARGUMENT IS STOPPED AT ITS FIRST CHARACTERS
# (#46 in docs/SELF-IMPROVEMENT-LOG.md; the incident is #44). Octopus
# v0e-V0-xhigh-1 prompt 2, 03:18: the model had read a screenshot saved as a
# base64 data URL (/tmp/full_url.txt, ~57K characters) and then TRANSCRIBED
# it into vision_analyze's image_url. The two attempts ran 23 and 40 minutes
# upstream (3.65 MB and 6.64 MB of SSE) before the relay died, and none
# could have worked: a model cannot write an image's bytes, it can only copy
# or invent them (Hermes rejected every hand-typed data URL: "Incorrect
# padding", "source is not a recognized image").
#
# SO, FOR IMAGE ARGUMENTS ONLY (operator, 2026-09-26: a generic size or
# alphabet heuristic over every argument is "janky and error prone"): a
# value that opens with `data:image/` -- or any `data:...;base64,` -- or
# with the base64 of an image file's first bytes (PNG, JPEG, GIF, WebP) is
# never valid from the model. It is caught as the value's first characters
# stream in (no size threshold), the generation is cut upstream, the call is
# not forwarded, and the model gets it back NOT EXECUTED with the remedy.
#
# WHICH ARGUMENTS: the known-names table first -- verified from each tool's
# own schema -- then SHAPE, from the tool's schema in the request: a
# parameter NAMED as an image (image, image_url, image_path, screenshot,
# photo, ...), or one whose description says image / picture / screenshot /
# photo and whose name says url / path / file / source (or format "uri").
# A call to a tool the request does not declare is judged by its argument
# names alone. Every other argument -- a write_file's content, a patch -- is
# never looked at.
IMAGE_GUARD = os.environ.get("YAMADORI_IMAGE_GUARD", "1") != "0"
# How many times a request may write the turn again after a stopped call
# before it LANDS (tools withdrawn, the answer asked for). A CHOICE: a model
# that pastes image data a third time after being told twice is not
# converging.
IMAGE_REGENERATIONS = int(os.environ.get("YAMADORI_IMAGE_REGENERATIONS", "2"))

# name -> the argument keys that take an image. VERIFIED entries only.
IMAGE_TOOLS: dict[str, tuple[str, ...]] = {
    # Hermes Agent: vision_analyze(image_url, question, region), VERIFIED
    # (hermes-agent tools/vision_tools.py VISION_ANALYZE_SCHEMA; its own
    # description offers "Image URL (http/https), local file path, or data:
    # URL to load"). Its other image tools (browser_vision, computer_use)
    # take their screenshot themselves and have no image argument.
    "vision_analyze": ("image_url",),
    # Ours: yama_describe_image(image, question), VERIFIED (mcp/vision.py
    # TOOL); its name before the yama_* rename (2026-09-27) is still read.
    "yama_describe_image": ("image",),
    "describe_image": ("image",),
}

_IMAGE_NAME = re.compile(
    r"^(?:image|images|img|imgs|picture|pictures|photo|photos|screenshot|"
    r"screenshots)$|"
    r"^(?:image|img|picture|photo|screenshot)s?[_-]?(?:url|uri|urls|uris|"
    r"path|paths|file|files|src|source|data|b64|base64)$|"
    r"^(?:source|input|reference|ref|target|init)[_-]?(?:image|img)s?$",
    re.I)
_IMAGE_WORDS = re.compile(r"\b(?:image|images|picture|photo|screenshot|"
                          r"png|jpe?g|webp)\b", re.I)
_LOCATOR_NAME = re.compile(r"(?:url|uri|path|file|src|source|location)s?$",
                           re.I)
# The base64 of each image format's first bytes; _MAGIC_AT characters of
# the base64 alphabet must open the value before it counts (a path or an id
# never runs on like that).
_MAGIC = (("iVBORw0KGgo", "PNG"), ("/9j/", "JPEG"), ("R0lGODlh", "GIF"),
          ("R0lGODdh", "GIF"), ("UklGR", "WebP"))
_MAGIC_AT = 24
_B64 = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
                 "0123456789+/=")
_DATA_WAIT = 80                     # a data: URL's header, at most


def _tool_schemas(tools: list | None) -> dict[str, dict]:
    out = {}
    for t in tools or []:
        fn = t.get("function") if isinstance(t, dict) else None
        if isinstance(fn, dict) and fn.get("name"):
            p = fn.get("parameters")
            p = p if isinstance(p, dict) else {}
            props = p.get("properties")
            out[fn["name"]] = props if isinstance(props, dict) else {}
    return out


def image_param(name: str, spec: dict | None) -> bool:
    """Whether a parameter takes an image, by its SHAPE: named as one, or
    described as one with a locator's name or format "uri"."""
    if _IMAGE_NAME.match(name or ""):
        return True
    spec = spec if isinstance(spec, dict) else {}
    desc = " ".join(str(spec.get(k) or "") for k in ("description", "title"))
    return bool(_IMAGE_WORDS.search(desc)) and (
        bool(_LOCATOR_NAME.search(name or "")) or spec.get("format") == "uri")


def image_arguments(name: str, tools: list | None) -> set[str] | None:
    """The argument keys of tool `name` that take an image: the table, else
    the request's schema by shape. None: the request does not declare the
    tool, so its argument names are judged as they arrive."""
    if name in IMAGE_TOOLS:
        return set(IMAGE_TOOLS[name])
    props = _tool_schemas(tools)
    if name not in props:
        return None
    return {k for k, v in props[name].items() if image_param(k, v)}


def image_data_kind(prefix: str, closed: bool = False):
    """Judge a string value by its first characters: the kind of image data
    ("a data:image/ URL", "base64 PNG data", ...) when it is some, False
    when it is not, None while it cannot tell yet. `closed`: the string has
    ended."""
    s = prefix.lstrip()
    if not s:
        return False if closed else None
    low = s.lower()
    if low.startswith("data:"):
        if low.startswith("data:image/"):
            return "a data:image/ URL"
        if ";base64," in low:
            return "a base64 data: URL"
        return False if (closed or len(low) >= _DATA_WAIT) else None
    if "data:image/".startswith(low):
        return False if closed else None
    for magic, kind in _MAGIC:
        if s.startswith(magic):
            run = s[:_MAGIC_AT]
            if not all(c in _B64 for c in run):
                return False
            if len(run) >= _MAGIC_AT:
                return f"base64 {kind} data"
            return False if closed else None
        if magic.startswith(s):
            return False if closed else None
    return False


class _Scan:
    """One call's arguments, read as JSON while they stream: which
    top-level key a string value belongs to, and -- when that key takes an
    image -- the value's first characters, judged as they arrive."""

    def __init__(self, keys: set[str] | None):
        self.keys = keys            # None: judge keys by name (_IMAGE_NAME)
        self.stack: list[str] = []
        self.in_str = self.esc = self.is_key = False
        self.buf = ""
        self.key = None             # the current top-level key
        self.expect_key = False
        self.want = False           # this string's prefix is being judged
        self.pos = 0
        self.start = 0
        self.done = False

    def _image_key(self) -> bool:
        k = self.key or ""
        return (k in self.keys) if self.keys is not None \
            else bool(_IMAGE_NAME.match(k))

    def feed(self, piece: str) -> dict | None:
        for ch in piece:
            self.pos += 1
            if self.in_str:
                if self.esc:
                    self.esc = False
                    if self.want or self.is_key:
                        self.buf += {"n": "\n", "t": "\t", "r": "\r",
                                     "b": "\b", "f": "\f"}.get(ch, ch)
                elif ch == "\\":
                    self.esc = True
                    continue
                elif ch == '"':
                    self.in_str = False
                    if self.is_key:
                        if len(self.stack) == 1:
                            self.key = self.buf
                    elif self.want:
                        kind = image_data_kind(self.buf, closed=True)
                        if kind:
                            return self._hit(kind)
                    self.want = False
                    self.buf = ""
                    continue
                elif self.want or self.is_key:
                    self.buf += ch
                else:
                    continue
                if self.want:
                    kind = image_data_kind(self.buf)
                    if kind:
                        return self._hit(kind)
                    if kind is False:
                        self.want = False
                elif len(self.buf) > 256:
                    self.buf = self.buf[:256]
                continue
            if ch == '"':
                self.in_str = True
                self.buf = ""
                self.is_key = self.expect_key and bool(self.stack) \
                    and self.stack[-1] == "{"
                self.expect_key = False
                self.want = (not self.is_key and bool(self.stack)
                             and self._image_key())
                self.start = self.pos - 1        # the value's opening quote
            elif ch in "{[":
                self.stack.append(ch)
                self.expect_key = ch == "{"
            elif ch in "}]":
                if self.stack:
                    self.stack.pop()
                if not self.stack:
                    self.key = None
            elif ch == ",":
                self.expect_key = bool(self.stack) and self.stack[-1] == "{"
        return None

    def _hit(self, kind: str) -> dict:
        self.done = True
        return {"argument": self.key or "arguments", "kind": kind,
                "start": self.start, "at": self.pos}


class ImageArgWatch:
    """The guard's reader for one generation: `feed(index, name, piece)`
    with each argument delta as it arrives returns the hit -- {index,
    argument, kind, start, at} -- the moment an image argument's value opens
    as image data, else None. A call with no image argument (a write_file,
    a patch) is not read at all."""

    def __init__(self, tools: list | None):
        self.tools = tools
        self._scans: dict[int, _Scan | None] = {}

    def feed(self, index: int, name: str, piece: str) -> dict | None:
        if index not in self._scans:
            keys = image_arguments(name or "", self.tools)
            self._scans[index] = None if keys == set() else _Scan(keys)
        sc = self._scans[index]
        if sc is None or sc.done or not piece:
            return None
        hit = sc.feed(piece)
        if hit is not None:
            hit["index"] = index
        return hit

    def read(self, index: int) -> bool:
        """Whether call `index`'s arguments were read at all."""
        return self._scans.get(index) is not None


def _close_json(text: str) -> str:
    """A cut-off JSON text closed: an open string, then every open object
    and array. What a stopped call's arguments are rendered as."""
    stack: list[str] = []
    in_str = esc = False
    for ch in text:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]" and stack:
            stack.pop()
    if esc:
        text = text[:-1]
    if in_str:
        text += '"'
    else:
        text = text.rstrip().rstrip(",")
        if text.endswith(":"):
            text += "null"
    return text + "".join("}" if b == "{" else "]" for b in reversed(stack))


IMAGE_CUT = "[image data removed by the proxy]"


def image_stopped(name: str, arguments: str, hit: dict) -> dict:
    """What a stopped call was, for the record, the hidden hop and the NOT
    EXECUTED result: the argument, the kind of data, and the call's
    arguments with that value replaced by IMAGE_CUT (valid JSON; every key
    before it kept)."""
    start = int(hit.get("start") or 0)
    arg = hit.get("argument") or "arguments"
    try:
        args = json.loads(_close_json(arguments[:start] + json.dumps(
            IMAGE_CUT)))
        if not isinstance(args, dict):
            raise ValueError("not an object")
    except ValueError:
        args = {arg: IMAGE_CUT}
    return {"tool": name or "(unnamed)", "argument": arg,
            "kind": hit.get("kind") or "image data",
            "chars_seen": len(arguments),
            "arguments": json.dumps(args, ensure_ascii=False)}


def image_result(stop: dict, generated: bool = False) -> str:
    """The stopped call's tool result, in the shape AGENTS.md "Failure
    returns carry the next step" asks for: the situation, whether it is
    retryable (a fact), the remedy and whose it is. The wording is a prompt
    and a CHOICE: one sentence of situation, one of remedy, and a single
    "don't" naming the observed failure. `generated`: yama_generate_image
    is offered, so the remedy names the link it returns too."""
    also = (", or, for an image yama_generate_image made, the link it "
            "returned"
            if generated else "")
    return (f"NOT EXECUTED: the `{stop['tool']}` call's `{stop['argument']}` "
            f"argument was image data ({stop['kind']}), which a model cannot "
            f"write, so the proxy stopped the call before it ran. Retryable: "
            f"yes, with a path in place of the data. Remedy (yours): pass the "
            f"path of the image file, saved under the project folder{also}; "
            f"don't paste image data into a tool call.")


def image_other_result(stop: dict) -> str:
    """The result for a complete call made in the same generation, before
    the stopped one: not run either, since the turn is written again."""
    return (f"NOT EXECUTED: not run, because the `{stop['tool']}` call after "
            f"it was stopped (its result says why). Retryable: yes -- make "
            f"this call again if it is still needed.")
