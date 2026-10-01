#!/usr/bin/env python
"""Reading the code a model writes INTO a client tool call, and THE IMAGE
GUARD.

  detection     which client calls write or patch a file, and the code they
                carry (`detect`): the known-names table (KNOWN) first, then
                argument shape; which calls READ a file (`read_target`);
                patch formats (`parse_patch`). Read by the skills system
                (skill_select, skill_classify, skill_packages, skill_prove),
                the no-progress guard (progress_guard) and result
                compression (result_compress). Detection only ever ENABLES
                a reading; a miss skips it.
  image guard   a tool call whose IMAGE argument opens as image data is
                stopped while it streams (see THE IMAGE GUARD below).

REMOVED 2026-09-29 (docs/REMOVED.md; the way back is commit e360d37): the
code CHECK of client writes before they were forwarded (the parser, the
lint, the formatter as a check), the second brain's fixup job that repaired
what did not parse (REPAIR_ROUNDS), and the one-line notes it wrote as
content before the calls ("Verified ...", "Repaired ...", "Checked ...").
The model imitated those notes in its own content (#36), and nothing
measured showed the repair helped. The client receives the calls exactly as
the model wrote them.
"""
from __future__ import annotations

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import code_check  # noqa: E402

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
