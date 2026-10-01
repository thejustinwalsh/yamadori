#!/usr/bin/env python
"""Detection of code in client tool calls (mcp/tool_code.py). No GPU, no
network.

    python mcp/test_tool_code.py      -> "N/M checks passed"

WHAT THIS GATES

tool_code's CHECK of client writes and the second brain's fix-up were
removed 2026-09-29 (docs/REMOVED.md; the way back is commit e360d37): no
note, no repair, no x_yamadori.tool_code. What stays is DETECTION -- which
file a client call writes or reads and what code it carries -- read by
mcp/progress.py (the project's files, the read tools) and the proxy's work
log, and the IMAGE GUARD (gated in mcp/test_image_guard.py).

  1. Every KNOWN-NAMES table entry, and the argument-shape fallback for
     unknown names, finds the unit (kind, path, language) and the model's
     new code; an unrecognised call with a code-sized string is reported
     unknown; a short command and non-JSON arguments are neither.
  2. parse_patch reads V4A and unified diffs (add / update / delete, hunks);
     looks_like_patch and language_of.
  3. read_target: the READ_KNOWN tools, their path normalised and their
     range keys; any other call is not a read.
  4. An .html write: its inline classic and module scripts are JavaScript
     units (script_units, html_scripts); a src script and a shader are not;
     a page with no script carries no unit. Pi's edit is found by the table.
"""
from __future__ import annotations

import json
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import tool_code  # noqa: E402

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


def call(name: str, args: dict, cid: str = "call_1") -> dict:
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


# ---------------------------------------------------------------------------
# 1. Detection
# ---------------------------------------------------------------------------
BAD_PY = "def f(:\n    return 1\n"
GOOD_PY = "def f():\n    return 1\n"
BAD_JS = "function f( {\n  return 1;\n}\n"

TABLE_CASES = [
    # (label, name, args, expected (kind, path) of the first unit)
    ("Hermes / MCP write_file(path, content)", "write_file",
     {"path": "a.py", "content": BAD_PY}, ("file", "a.py")),
    ("Gemini CLI write_file(file_path, content)", "write_file",
     {"file_path": "a.py", "content": BAD_PY}, ("file", "a.py")),
    ("Hermes patch(path, old_string, new_string)", "patch",
     {"path": "a.py", "old_string": GOOD_PY, "new_string": BAD_PY},
     ("edit", "a.py")),
    ("Hermes patch(patch) in V4A form", "patch",
     {"patch": "*** Begin Patch\n*** Add File: a.py\n+def f(:\n+    pass\n"
               "*** End Patch"}, ("file", "a.py")),
    ("MCP edit_file(path, edits[{oldText,newText}])", "edit_file",
     {"path": "a.py", "edits": [{"oldText": GOOD_PY, "newText": BAD_PY}]},
     ("edit", "a.py")),
    ("Roo edit_file(file_path, old_string, new_string)", "edit_file",
     {"file_path": "a.py", "old_string": GOOD_PY, "new_string": BAD_PY},
     ("edit", "a.py")),
    ("text editor create", "str_replace_based_edit_tool",
     {"command": "create", "path": "a.py", "file_text": BAD_PY},
     ("file", "a.py")),
    ("text editor str_replace", "str_replace_editor",
     {"command": "str_replace", "path": "a.py", "old_str": GOOD_PY,
      "new_str": BAD_PY}, ("edit", "a.py")),
    ("text editor insert", "text_editor",
     {"command": "insert", "path": "a.py", "insert_line": 3,
      "new_str": BAD_PY}, ("insert", "a.py")),
    ("Claude Code Write", "Write", {"file_path": "a.py", "content": BAD_PY},
     ("file", "a.py")),
    ("Claude Code Edit", "Edit", {"file_path": "a.py", "old_string": GOOD_PY,
                                  "new_string": BAD_PY}, ("edit", "a.py")),
    ("Claude Code MultiEdit", "MultiEdit",
     {"file_path": "a.py", "edits": [{"old_string": GOOD_PY,
                                      "new_string": BAD_PY}]},
     ("edit", "a.py")),
    ("Gemini CLI replace", "replace", {"file_path": "a.py",
                                       "old_string": GOOD_PY,
                                       "new_string": BAD_PY}, ("edit", "a.py")),
    ("Cline / Roo write_to_file", "write_to_file",
     {"path": "a.js", "content": BAD_JS}, ("file", "a.js")),
    ("Roo apply_diff (SEARCH/REPLACE)", "apply_diff",
     {"path": "a.py", "diff": "<<<<<<< SEARCH\n:start_line:1\n-------\n"
                              + GOOD_PY + "=======\n" + BAD_PY
                              + ">>>>>>> REPLACE"}, ("edit", "a.py")),
    ("Cline replace_in_file (------- SEARCH)", "replace_in_file",
     {"path": "a.py", "diff": "------- SEARCH\n" + GOOD_PY + "=======\n"
                              + BAD_PY + "+++++++ REPLACE"}, ("edit", "a.py")),
    ("Roo search_replace", "search_replace",
     {"file_path": "a.py", "old_string": GOOD_PY, "new_string": BAD_PY},
     ("edit", "a.py")),
    ("Roo edit", "edit", {"file_path": "a.py", "old_string": GOOD_PY,
                          "new_string": BAD_PY}, ("edit", "a.py")),
    ("OpenCode edit(filePath, oldString, newString)", "edit",
     {"filePath": "a.py", "oldString": GOOD_PY, "newString": BAD_PY},
     ("edit", "a.py")),
    ("OpenCode write(filePath, content)", "write",
     {"filePath": "a.py", "content": BAD_PY}, ("file", "a.py")),
    # Pi 0.87.1 (docs/HARNESS-PI.md gap 4; dist/core/tools/edit.js:10-21,
    # write.js:8-11), 2026-09-26: by the TABLE, no longer the shape fallback.
    ("Pi edit(path, edits[{oldText,newText}])", "edit",
     {"path": "a.py", "edits": [{"oldText": GOOD_PY, "newText": BAD_PY}]},
     ("edit", "a.py")),
    ("Pi edit, edits as a JSON string (prepareEditArguments)", "edit",
     {"path": "a.py", "edits": json.dumps([{"oldText": GOOD_PY,
                                            "newText": BAD_PY}])},
     ("edit", "a.py")),
    ("Pi edit, legacy top-level oldText/newText", "edit",
     {"path": "a.py", "oldText": GOOD_PY, "newText": BAD_PY},
     ("edit", "a.py")),
    ("Pi write(path, content)", "write",
     {"path": "a.py", "content": BAD_PY}, ("file", "a.py")),
    ("Codex apply_patch(input)", "apply_patch",
     {"input": "*** Begin Patch\n*** Update File: a.py\n@@\n-def f():\n"
               "+def f(:\n     return 1\n*** End Patch"}, ("edit", "a.py")),
    ("Roo apply_patch(patch)", "apply_patch",
     {"patch": "*** Begin Patch\n*** Add File: b.py\n+def g(:\n+    pass\n"
               "*** End Patch"}, ("file", "b.py")),
    ("Continue create_new_file(filepath, contents)", "create_new_file",
     {"filepath": "a.py", "contents": BAD_PY}, ("file", "a.py")),
]


# The broken line each case's new code carries (BAD_PY, BAD_JS, the patches').
BROKEN_LINES = ("def f(:", "function f( {", "def g(:")


def test_every_table_entry_and_the_fallback():
    names = {c[1] for c in TABLE_CASES}
    check(names == set(tool_code.KNOWN),
          f"every KNOWN entry is exercised ({len(tool_code.KNOWN)} names)",
          str(sorted(set(tool_code.KNOWN) ^ names)))
    for label, name, args, (kind, path) in TABLE_CASES:
        d = tool_code.detect(call(name, args))
        u = d["units"][0] if d["units"] else {}
        insert = kind == "insert"
        kind = "edit" if insert else kind
        check(d["detected"] == "table" and u.get("kind") == kind
              and u.get("path") == path and u.get("language") in (
                  "python", "javascript"),
              f"table: {label}", json.dumps({k: v for k, v in d.items()
                                             if k != "args"}, default=str)[:200])
        check(any(b in (u.get("code") or "") for b in BROKEN_LINES),
              f"  and the unit carries the model's new code ({label})",
              repr(u.get("code"))[:200])
        if insert:
            check(u.get("old") is None,
                  f"  an insertion has no old text ({label})", repr(u.get("old")))
    for label, args, kind in (
            ("path + content", {"target": "src/x.ts",
                                "body": "let x: = 1\n"}, "file"),
            ("path + old/new", {"file": "x.rs", "search": "fn a() {}",
                                "replacement": "fn a( {}"}, "edit"),
            ("a unified diff string", {"changes": "--- a/x.py\n+++ b/x.py\n"
                                                  "@@ -1 +1 @@\n-def f():\n"
                                                  "+def f(:\n     pass\n"},
             "edit"),
            ("a list of old/new pairs", {"file_path": "x.py", "changes": [
                {"old": GOOD_PY, "new": BAD_PY}]}, "edit")):
        d = tool_code.detect(call("save_everything", args))
        check(d["detected"] == "shape" and d["units"]
              and d["units"][0]["kind"] == kind,
              f"shape fallback: {label}", json.dumps(d, default=str)[:200])
    d = tool_code.detect(call("terminal", {"command": "cat > x.js <<'EOF'\n"
                                                      + "let a = 1;\n" * 40
                                                      + "EOF"}))
    check(d["detected"] is None and d["unknown"]
          and d["unknown"]["name"] == "terminal"
          and d["unknown"]["key"] == "command",
          "an unrecognised call with a code-sized string is reported unknown",
          json.dumps(d["unknown"]))
    d = tool_code.detect(call("terminal", {"command": "ls -la"}))
    check(d["detected"] is None and d["unknown"] is None,
          "a short command is neither detected nor reported")
    d = tool_code.detect(call("write_file", {"path": "README.md",
                                             "content": "# hi\n\n(("}))
    check([u.get("language") for u in d["units"]] == [None],
          "a file with no code extension is a unit no parser claims",
          json.dumps(d["units"], default=str))
    d = tool_code.detect({"id": "x", "function": {"name": "write_file",
                                                  "arguments": "{broken"}})
    check(d.get("unparsed") and not d["units"],
          "arguments that are not JSON: nothing detected")


# ---------------------------------------------------------------------------
# 2. Patches and languages
# ---------------------------------------------------------------------------
def test_patches_and_languages():
    v4a = tool_code.parse_patch(
        "*** Begin Patch\n*** Update File: a.py\n@@\n-def f():\n"
        "+def f(:\n     return 1\n*** Add File: b.py\n+x = 1\n"
        "*** Delete File: c.py\n*** End Patch")
    check([(f["path"], f["op"]) for f in v4a]
          == [("a.py", "update"), ("b.py", "add"), ("c.py", "delete")],
          "V4A: every file with its operation", json.dumps(v4a))
    check(v4a[0]["hunks"] == [("def f():\n    return 1",
                               "def f(:\n    return 1")]
          and v4a[1]["content"] == "x = 1",
          "V4A: an update's hunk (old, new) with context, an add's content",
          json.dumps(v4a))
    uni = tool_code.parse_patch(
        "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n c\n"
        "--- /dev/null\n+++ b/n.py\n@@ -0,0 +1 @@\n+new\n")
    check([(f["path"], f["op"]) for f in uni]
          == [("x.py", "update"), ("n.py", "add")]
          and uni[0]["hunks"] == [("a\nc", "b\nc")]
          and uni[1]["content"] == "new",
          "unified diff: a/ b/ prefixes stripped, /dev/null is an add",
          json.dumps(uni))
    check(tool_code.parse_patch("") == [] and tool_code.parse_patch(None) == [],
          "no text: no files")
    check(tool_code.looks_like_patch("--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b")
          and not tool_code.looks_like_patch("hello"),
          "looks_like_patch: a diff yes, prose no")
    for path, want in (("a/b.tsx", "tsx"), ("x.py", "python"),
                       ("Cargo.toml", "toml"), ("x.md", None), (None, None)):
        check(tool_code.language_of(path) == want,
              f"language_of({path!r}) is {want!r}",
              repr(tool_code.language_of(path)))


# ---------------------------------------------------------------------------
# 3. Read tools
# ---------------------------------------------------------------------------
def test_read_targets():
    got = tool_code.read_target(call("read", {"filePath": ".\\src\\a.ts",
                                              "offset": 3}))
    check(got == {"path": "src/a.ts", "range": {"offset": 3}, "tool": "read"},
          "OpenCode read: backslashes and a leading ./ normalised, the range "
          "keys it set", json.dumps(got))
    got = tool_code.read_target(call("read", {"path": "a.ts"}))
    check(got and got["path"] == "a.ts" and got["range"] == {},
          "Pi read(path): the same tool name, the other path key",
          json.dumps(got))
    got = tool_code.read_target(call("Read", {"file_path": "./x.py",
                                              "limit": 20}))
    check(got == {"path": "x.py", "range": {"limit": 20}, "tool": "Read"},
          "Claude Code Read", json.dumps(got))
    check(tool_code.read_target(call("terminal", {"command": "cat a.py"}))
          is None, "a shell cat is not a read tool")
    check(tool_code.read_target(call("read_file", {"offset": 1})) is None,
          "a read with no path is not a read")
    check(tool_code.read_target({"function": {"name": "read_file",
                                              "arguments": "{nope"}}) is None,
          "arguments that are not JSON: not a read")


# ---------------------------------------------------------------------------
# 4. Pi's edit and .html pages
# ---------------------------------------------------------------------------
PAGE = ("<!DOCTYPE html>\n<html><body>\n<canvas id=\"c\" width=\"256\" "
        "height=\"256\"></canvas>\n<script src=\"lib.js\"></script>\n"
        "<script type=\"x-shader/x-vertex\">void main() { gl_Position = "
        "vec4(0.0); }</script>\n<script>\n{SCRIPT}</script>\n"
        "<script type=\"module\">\nimport * as THREE from 'three';\n"
        "const scene = new THREE.Scene();\n</script>\n</body></html>\n")
BAD_SCRIPT = ("const ctx = document.getElementById('c').getContext('2d');\n"
              "ctx.fillStyle = 'red';\nctx.fillRect(64, 64, 128, 128;\n")



def test_pi_edits_and_html_pages():
    """docs/HARNESS-PI.md gap 4 (2026-09-26): Pi's edit by the table, and an
    .html write's inline scripts as JavaScript units."""
    # Pi's own call, #16 in the harness test (relay a1): by the TABLE now.
    pi = call("edit", {"path": "hello.py", "edits": [{
        "oldText": "return 'hello ' + name",
        "newText": "return 'howdy ' + name"}]})
    d = tool_code.detect(pi)
    check(d["detected"] == "table" and d["units"][0]["kind"] == "edit",
          "Pi's edit(path, edits[{oldText,newText}]) is recognised by the "
          "known-names table, not the shape fallback", str(d["detected"]))
    # A write of an .html page: its inline scripts are units of their own.
    units = tool_code.detect(call("write", {
        "path": "canvas.html",
        "content": PAGE.replace("{SCRIPT}", BAD_SCRIPT)}))["units"]
    check([(u["language"], u.get("script"), u.get("module")) for u in units]
          == [("javascript", 1, False), ("javascript", 2, True)],
          "an .html write: its classic and module scripts are JavaScript "
          "units; the src script and the shader are not",
          json.dumps([{k: u.get(k) for k in ("language", "script", "module",
                                              "span")} for u in units]))
    page = PAGE.replace("{SCRIPT}", BAD_SCRIPT)
    check(units and page[units[0]["span"][0]:units[0]["span"][1]]
          == units[0]["code"] and units[0]["code"].strip() == BAD_SCRIPT.strip(),
          "each script unit's span is where its code sits in the page",
          json.dumps(units[0].get("span")))
    check(tool_code.detect(call("write", {
        "path": "page.htm", "content": "<p>no code here</p>"}))["units"]
          == [], "a page with no script carries no unit")
    whole = tool_code.detect(call("write", {"path": "canvas.html",
                                            "content": page}),
                             whole_pages=True)["units"]
    check(len(whole) == 1 and whole[0]["code"] == page
          and whole[0]["path"] == "canvas.html",
          "whole_pages: the page is ONE file (which file a call writes)",
          json.dumps([{k: u.get(k) for k in ("path", "language")}
                      for u in whole]))
    check([u["script"] for u in tool_code.script_units(whole[0])] == [1, 2],
          "script_units turns the whole page back into its scripts")
    check([sc["module"] for sc in tool_code.html_scripts(page)] == [False, True],
          "html_scripts: the classic script, then the module")


TESTS = [
    test_every_table_entry_and_the_fallback,
    test_patches_and_languages,
    test_read_targets,
    test_pi_edits_and_html_pages,
]


def main() -> int:
    for t in TESTS:
        try:
            t()
        except Exception as e:                                   # noqa: BLE001
            check(False, f"{t.__name__} raised",
                  f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
    passed = sum(1 for ok, _, _ in _results if ok)
    for ok, name, detail in _results:
        if not ok:
            print(f"FAIL  {name}\n      {detail[:1500]}")
    print(f"\n{passed}/{len(_results)} checks passed")
    return 0 if passed == len(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
