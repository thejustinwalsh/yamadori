#!/usr/bin/env python
"""Tool-call code validation (mcp/tool_code.py) and the repair loops'
shared cap, against a fake upstream. No GPU.

WHAT THIS IS GATING (operator, 2026-09-24)

  1. Detection: every KNOWN-NAMES table entry, and the argument-shape
     fallback for unknown names, finds the code; an unrecognised call with a
     code-sized string is logged in x_yamadori.tool_code.unknown.
  2. A broken write_file is repaired BEFORE it is forwarded -- by the second
     brain's fixup job (shomen.run), which gets only the request, the code
     and its errors. Main generates once and never sees a failed attempt
     (operator, 2026-09-24).
  3. The cap: REPAIR_ROUNDS fixup rounds; a repair is written back only
     when it is accepted (closed fence, not cut off, no errors, plausibly
     complete), else the MODEL'S content is sent as written. At `medium`
     the write is checked and only noted.
  4. No change after forwarding: what the client receives is the model's
     last call (plus formatter output, noted), and nothing follows it.
  5. The note: a short content line before the calls says what was checked
     or fixed, in the fold-back phrases ("Verified", "Repaired", "Checked"),
     after the concept seed line when the fixup ran.
  6. CHANNEL ORDER on the stream: no content before the last reasoning; the
     note goes out once, as content, just before the one tool_calls delta.
  7. agent_step and utility never enter the request-level code pipelines
     (final-answer repair, fan-out); a utility call never reaches the
     tool-call check. An agent_step that WRITES a file does -- the response
     is code_edit work (mcp/route.py, WHAT READS THE CLASS).
  8. Every round runs at the request's own effort; the final-answer repair
     is the same fixup job, with the same cap.

THE FAKE UPSTREAM is a real http.server answering SSE, so the proxy's reader
runs unmodified; each request body is recorded. Stores are temp files set
before import; nothing here talks to :1234, :11434 or :1237.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

_TMP = tempfile.mkdtemp(prefix="yamadori_test_tool_code_")
os.environ["YAMADORI_CORPUS_DB"] = os.path.join(_TMP, "corpus.sqlite3")
os.environ["YAMADORI_NEBARI_DB"] = os.path.join(_TMP, "nebari.sqlite3")
os.environ["RINGS_DB"] = os.path.join(_TMP, "rings.sqlite3")
os.environ["YAMADORI_PKG_DIR"] = os.path.join(_TMP, "pkgs")
os.makedirs(os.environ["YAMADORI_PKG_DIR"], exist_ok=True)
os.environ["CODE_INDEX_DB"] = os.path.join(_TMP, "code.sqlite3")
os.environ.pop("YAMADORI_IMAGEGEN_URL", None)
os.environ.pop("YAMADORI_DELEGATE_TOOL", None)
os.environ.pop("YAMADORI_REPAIR_ROUNDS", None)
os.environ["CONCEPT_SEED_LAST"] = os.path.join(_TMP, "seed_last.json")

import code_check  # noqa: E402
import tiers  # noqa: E402
import tool_code  # noqa: E402
import proxy  # noqa: E402
import shomen  # noqa: E402

# A pinned concept seed: no 420 MB matrix load, a predictable seed line.
proxy._draw_seed = lambda prompt=None: {"word": "cedar", "token_id": 1,
                                        "u32": 2, "hex": "0x00000002"}

tiers._accepted = ("low", "medium", "xhigh")
proxy.PREAMBLE = False

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
    ("Codex apply_patch(input)", "apply_patch",
     {"input": "*** Begin Patch\n*** Update File: a.py\n@@\n-def f():\n"
               "+def f(:\n     return 1\n*** End Patch"}, ("edit", "a.py")),
    ("Roo apply_patch(patch)", "apply_patch",
     {"patch": "*** Begin Patch\n*** Add File: b.py\n+def g(:\n+    pass\n"
               "*** End Patch"}, ("file", "b.py")),
    ("Continue create_new_file(filepath, contents)", "create_new_file",
     {"filepath": "a.py", "contents": BAD_PY}, ("file", "a.py")),
]


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
        rv = tool_code.review([call(name, args)])
        if insert:
            # An insertion has no old text to tell a unit from a fragment:
            # flagged, never blocked.
            check(rv["errors"] == 0 and rv["units"][0]["result"]["status"]
                  == "fragment", f"  and its broken code is flagged, not "
                                 f"blocked ({label})")
            continue
        check(rv["errors"] >= 1 and rv["blocked"] == [0],
              f"  and its broken code blocks ({label})",
              json.dumps(rv["units"], default=str)[:200])
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
          "a short command is neither checked nor reported")
    rv = tool_code.review([call("write_file", {"path": "README.md",
                                               "content": "# hi\n\n(("})])
    check(rv["units"] == [] and rv["errors"] == 0,
          "a file with no code extension is not checked (no parser claims it)")
    d = tool_code.detect({"id": "x", "function": {"name": "write_file",
                                                  "arguments": "{broken"}})
    check(d.get("unparsed") and not d["units"],
          "arguments that are not JSON: nothing detected, nothing blocked")


def test_the_checks():
    r = tool_code.check_file("import os\n\ndef f():\n    return y\n", "python")
    check(r["lint"] == 1 and "F821" in r["errors"][0]["message"],
          "python whole file: ruff F821 (undefined name) blocks",
          json.dumps(r["errors"]))
    r = tool_code.check_file("import os\n", "python")
    check(r["errors"] == [], "an unused import is not an error (not in "
                             f"{code_check.LINT_SELECT})")
    r = tool_code.check_file(BAD_JS, "javascript")
    check(r["syntax"] >= 1 and r["formatted"] is None,
          "javascript: a syntax error blocks, nothing is formatted")
    r = tool_code.check_file("function f(){return 1}\n", "javascript")
    fmt = code_check.formatter_status("prettier")["available"]
    # The style is inferred from the code itself: no semicolons in, none out.
    check((r["errors"] == [] and r["formatted"] == "function f() {\n  "
           "return 1\n}\n") if fmt else r["formatted"] is None,
          "javascript that parses is formatted by prettier, in its own style "
          + ("" if fmt else "(prettier absent: skipped)"), json.dumps(r))
    r = tool_code.check_file("fn main() { let x = ; }\n", "rust")
    check(r["syntax"] >= 1, "rust: a syntax error blocks")
    r = tool_code.check_file('{"a": 1,}\n', "json")
    check(r["syntax"] == 1, "json: a trailing comma blocks")
    e = tool_code.check_edit(GOOD_PY, BAD_PY, "python")
    check(e["status"] == "errors", "an edit replacing a complete unit with "
                                   "broken code blocks")
    e = tool_code.check_edit("    x = (1,\n", "    x = (2,\n", "python")
    check(e["status"] == "fragment", "a fragment replacing a fragment is "
                                     "flagged, never blocked")
    e = tool_code.check_edit(None, "def g(:\n", "python")
    check(e["status"] == "fragment", "an insertion that does not parse is "
                                     "flagged, never blocked")
    e = tool_code.check_edit(GOOD_PY, "def f():\n    return 2\n", "python")
    check(e["status"] == "ok", "a good replacement passes")


# ---------------------------------------------------------------------------
# The fake upstream
# ---------------------------------------------------------------------------
_script: list[dict] = []
_seen: list[dict] = []


def reply(content: str = "", calls: list | None = None, reasoning: str = "",
          finish: str | None = None) -> dict:
    return {"content": content, "calls": calls or [], "reasoning": reasoning,
            "finish": finish or ("tool_calls" if calls else "stop")}


def _sse(r: dict) -> bytes:
    def ev(delta=None, finish=None):
        return (b"data: " + json.dumps({
            "id": "u", "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": delta or {},
                         "finish_reason": finish}]}).encode() + b"\n\n")
    out = [ev({"role": "assistant"})]
    s = r["reasoning"]
    out += [ev({"reasoning_content": s[i:i + 9]}) for i in range(0, len(s), 9)]
    s = r["content"]
    out += [ev({"content": s[i:i + 9]}) for i in range(0, len(s), 9)]
    for i, c in enumerate(r["calls"]):
        out.append(ev({"tool_calls": [dict(c, index=i)]}))
    out.append(ev({}, r["finish"]))
    out.append(b"data: " + json.dumps({"id": "u", "choices": [], "usage": {
        "prompt_tokens": 100, "completion_tokens": 10,
        "total_tokens": 110}}).encode() + b"\n\n")
    return b"".join(out) + b"data: [DONE]\n\n"


class _Up(BaseHTTPRequestHandler):
    def do_POST(self):                                           # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path.startswith("/upstream/"):
            # proxy._warm: a render, then a zero-token /completion.
            data = json.dumps({"prompt": "".join(
                json.dumps(m) + "<|im_end|>\n"
                for m in body.get("messages") or [])}
                if self.path.endswith("apply-template") else {}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        _seen.append(body)
        if not _script:
            self.send_response(500)
            self.end_headers()
            return
        data = _sse(_script.pop(0))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):                                   # noqa: D102
        pass


_srv = ThreadingHTTPServer(("127.0.0.1", 0), _Up)
threading.Thread(target=_srv.serve_forever, daemon=True).start()
proxy.UPSTREAM = f"http://127.0.0.1:{_srv.server_address[1]}"
# The second brain's door (mcp/model.py) goes to the same fake: no
# offline test may reach the live model server.
import model as _model  # noqa: E402
_model.UPSTREAM = proxy.UPSTREAM

# `high`: repair allowed; everything else that would call a service is off.
HEADER = {"retrieval": False, "hints": False, "investigate": False,
          "fanout": 1}
WRITE_FILE = {"type": "function", "function": {
    "name": "write_file", "description": "Write a file.",
    "parameters": {"type": "object", "properties": {
        "path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"]}}}
TERMINAL = {"type": "function", "function": {
    "name": "terminal", "description": "Run a command.",
    "parameters": {"type": "object", "properties": {
        "command": {"type": "string"}}}}}
SYSTEM = "You are Hermes Agent, built by Nous Research."
ASK = [{"role": "system", "content": SYSTEM},
       {"role": "user", "content": "Add a Game class to js/game.js."}]
AFTER_TOOL = ASK + [
    {"role": "assistant", "content": "", "tool_calls": [call(
        "terminal", {"command": "ls js"}, "t0")]},
    {"role": "tool", "tool_call_id": "t0", "content": "game.js\n"}]

BROKEN = "class Game {\n  constructor() {\n    this.x = 1;\n  \n}\n"
BROKEN2 = "class Game {\n  start( {\n    this.x = 1;\n  }\n}\n"
BROKEN3 = "class Game {\n  stop() {\n    this.x = ;\n  }\n}\n"
BROKEN4 = "class Game {\n  reset() {\n    this.x = 1\n  \n}\n"
FIXED = "class Game {\n  constructor() {\n    this.x = 1;\n  }\n}\n"


def body(messages=None, tools=(WRITE_FILE, TERMINAL), header=None,
         effort="high", stream=False) -> dict:
    b = {"model": "yamadori", "reasoning_effort": effort,
         "_client_ip": "127.0.0.1",
         "_features": json.dumps(HEADER if header is None else header),
         "messages": list(messages or ASK)}
    if tools:
        b["tools"] = list(tools)
    if stream:
        b["stream"] = True
    return b


def run_complete(script: list[dict], **kw) -> dict:
    _script[:] = script
    _seen.clear()
    return proxy.complete(body(**kw))


def run_stream(script: list[dict], **kw) -> list[dict]:
    _script[:] = script
    _seen.clear()
    out = []
    for b in proxy.stream_body(body(stream=True, **kw), "yamadori"):
        if b.strip() == b"data: [DONE]":
            out.append({"_done": True})
            continue
        out.append(json.loads(b[6:].decode()))
    return out


def wf(content: str, path: str = "js/game.js", cid: str = "call_1") -> dict:
    return call("write_file", {"path": path, "content": content}, cid)


def args_of(c: dict) -> dict:
    return json.loads(c["function"]["arguments"])


def x_of(d: dict) -> dict:
    return d.get("x_yamadori") or {}


def test_fake_upstream_is_alive():
    _script[:] = [reply("pong")]
    _seen.clear()
    d = proxy._post("/v1/chat/completions", {"messages": []})
    check(d["choices"][0]["message"]["content"] == "pong",
          "fake upstream answers through the proxy's own reader")


# ---------------------------------------------------------------------------
# 2-5, 8. The fix-up, blocking path (operator, 2026-09-24): main writes the
# call ONCE; the second brain's fixup job (shomen.run, faked at
# shomen._post) gets only the request, the code and its errors; the client
# receives the fixed call and a note in the fold-back phrases.
# ---------------------------------------------------------------------------
HELPER: list[str] = []            # the fixup job's replies, popped in order
_helper_seen: list[dict] = []


def _fence(code: str, lang: str = "javascript") -> str:
    return f"```{lang}\n{code.rstrip()}\n```"


def _helper_post(path, payload, timeout=3600):
    _helper_seen.append(json.loads(json.dumps(payload)))
    text = HELPER.pop(0) if HELPER else "(no code)"
    text, fin = text if isinstance(text, tuple) else (text, "stop")
    return {"choices": [{"message": {"content": text},
                         "finish_reason": fin}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


shomen.PHRASES  # the phrase table these notes are written in
shomen._post = _helper_post


def helper(*replies: str) -> None:
    HELPER[:] = list(replies)
    _helper_seen.clear()


def test_a_broken_write_file_is_repaired_before_forwarding():
    helper(_fence(FIXED))
    d = run_complete([reply("I'll write it.", [wf(BROKEN)])])
    msg = d["choices"][0]["message"]
    check(len(_seen) == 1, "main generates ONCE: it never sees a failed "
          "attempt, and no feedback turn is added to its context",
          str(len(_seen)))
    hm = _helper_seen[0]["messages"] if _helper_seen else []
    user = hm[-1]["content"] if hm else ""
    check(len(_helper_seen) == 1 and len(hm) == 2
          and hm[0]["content"] == shomen.FIXUP_SYSTEM
          and "Add a Game class to js/game.js." in user and BROKEN in user
          and "js/game.js" in user and "line " in user
          and SYSTEM not in json.dumps(hm)
          and "Inspiration word: cedar" in user,
          "the fixup job gets ONLY the request, the file, its errors and the "
          "concept seed -- never the conversation", json.dumps(hm)[:500])
    calls = msg.get("tool_calls") or []
    check(len(calls) == 1 and args_of(calls[0])["content"] == FIXED
          and calls[0]["id"] == "call_1",
          "the client receives the model's call with the fixed code",
          json.dumps(calls)[:300])
    check(BROKEN not in json.dumps(d), "the broken version never leaves")
    tc = x_of(d).get("tool_code") or {}
    check(tc.get("rounds") == 1 and tc.get("errors_before", 0) >= 1
          and tc.get("errors_after") == 0 and tc.get("stopped") == "fixed"
          and tc.get("language") == ["javascript"]
          and tc["files"][0]["path"] == "js/game.js"
          and (tc.get("fixup") or {}).get("written") == 1,
          "x_yamadori.tool_code: files, language, errors before/after, "
          "rounds, the fixup job, stopped", json.dumps(tc))
    check(not any(k.startswith("_") for k in tc) and "class Game"
          not in json.dumps(tc), "no bookkeeping and no code in the record")
    note = msg.get("content") or ""
    # Since 2026-09-24 a note is mechanical: no seed line (coordinator).
    check(note.startswith("I'll write it.\n\nRepaired js/game.js "
                          "(javascript): ")
          and "fixed in 1 round" in note and "inspired" not in note,
          "the note: 'Repaired <file>', and no seed line", note)
    check(d["choices"][0]["finish_reason"] == "tool_calls",
          "finish stays tool_calls")
    check(_helper_seen and _helper_seen[0].get("_effort_tier") == "high",
          "the fixup runs at the request's own effort (tier high)",
          str(_helper_seen[0].get("_effort_tier") if _helper_seen else None))


def test_the_cap_and_the_last_version_is_sent():
    helper(_fence(BROKEN2), _fence(BROKEN3), _fence(BROKEN4))
    d = run_complete([reply("", [wf(BROKEN)])])
    tc = x_of(d).get("tool_code") or {}
    check(len(_seen) == 1 and len(_helper_seen) == tool_code.REPAIR_ROUNDS == 3
          and tc.get("rounds") == 3 and tc.get("stopped") == "not_fixed",
          f"REPAIR_ROUNDS={tool_code.REPAIR_ROUNDS}: three fixup rounds on "
          "the second brain, then it stops", json.dumps(tc))
    calls = d["choices"][0]["message"].get("tool_calls") or []
    # Pre-deploy review, 2026-09-24: a failed repair never overwrites the
    # model's file. What goes is what the model wrote, and "sent as written"
    # is then true.
    check(len(calls) == 1 and args_of(calls[0])["content"] == BROKEN
          and BROKEN4 not in json.dumps(d),
          "a repair that never parsed is NOT forwarded: the model's own "
          "content goes, untouched")
    note = d["choices"][0]["message"].get("content") or ""
    check(note.startswith("Checked js/game.js")
          and "still there after 3 rounds of repair" in note
          and "the repair was not used" in note
          and "sent as written" in note,
          "and the note says the problems remain and the repair was not "
          "used", note)
    helper("I cannot see the problem.")
    d = run_complete([reply("", [wf(BROKEN)])])
    tc = x_of(d).get("tool_code") or {}
    calls = d["choices"][0]["message"].get("tool_calls") or []
    check(len(_helper_seen) == 1 and tc.get("stopped") == "not_fixed"
          and args_of(calls[0])["content"] == BROKEN,
          "a reply with no code: the call goes out as written, and says so",
          json.dumps(tc))


def test_formatter_output_is_applied_and_noted():
    if not code_check.formatter_status("prettier")["available"]:
        check(True, "prettier absent: formatting test skipped")
        return
    helper()
    d = run_complete([reply("", [wf("function f(){return 1}\n",
                                    path="js/util.js")])])
    calls = d["choices"][0]["message"]["tool_calls"]
    # #24 (docs/SELF-IMPROVEMENT-LOG.md, 2026-09-24): change what the model
    # wrote ONLY when it is broken. The formatter's output used to replace
    # every whole-file write; the model then patched with its own pre-format
    # text and the patch failed. Now it is reported, never applied.
    check(len(_seen) == 1 and not _helper_seen
          and args_of(calls[0])["content"] == "function f(){return 1}\n",
          "clean code the formatter would change goes out AS WRITTEN, with "
          "no generation anywhere")
    note = d["choices"][0]["message"]["content"]
    tc = x_of(d)["tool_code"]
    check(note == "Verified js/util.js (javascript): parses; prettier would "
                  "change 3 lines; sent as written."
          and not tc["formatted"] and not tc["files"][0]["formatted"]
          and tc["files"][0].get("format_would_change") == 3,
          "the note reports the formatting as information ('Verified'), and "
          "the record says nothing was changed", note)


def test_edits_block_only_a_complete_unit():
    good_old = "function a() {\n  return 1;\n}"
    helper(_fence("function a() {\n  return 2;\n}"))
    d = run_complete([reply("", [call("patch", {
        "path": "js/a.js", "old_string": good_old,
        "new_string": "function a() {\n  return 1;\n"})])])
    tc = x_of(d)["tool_code"]
    calls = d["choices"][0]["message"]["tool_calls"]
    check(len(_seen) == 1 and tc["rounds"] == 1 and tc["stopped"] == "fixed"
          and args_of(calls[0])["new_string"]
          == "function a() {\n  return 2;\n}\n"
          and args_of(calls[0])["old_string"] == good_old,
          "an edit that breaks a complete unit is repaired: only its "
          "replacement text is rewritten", json.dumps(tc))
    helper()
    d = run_complete([reply("", [call("patch", {
        "path": "js/a.js", "old_string": "  if (x) {\n    go();",
        "new_string": "  if (x && y) {\n    go();"})])])
    tc = x_of(d)["tool_code"]
    note = d["choices"][0]["message"]["content"]
    check(len(_seen) == 1 and not _helper_seen
          and tc["files"][0]["flagged"] == "fragment" and "fragment" in note,
          "an edit fragment is flagged in the note and the record, not "
          "blocked", note)


def test_a_utility_call_and_agent_steps_stay_out_of_the_code_pipelines():
    side = [{"role": "system", "content": "You are a security reviewer for an "
                                          "AI coding agent."},
            {"role": "user", "content": "<command>ls</command>\n\nRespond "
                                        "with exactly one word: APPROVE, DENY,"
                                        " or ESCALATE"}]
    helper()
    d = run_complete([reply("APPROVE")], messages=side, tools=None)
    x = x_of(d)
    check(x.get("route", {}).get("class") == "utility"
          and x.get("tool_code") is None and x.get("repair") is None
          and len(_seen) == 1 and not _helper_seen,
          "utility: no tool-call check, no repair, one generation",
          json.dumps({k: x.get(k) for k in ("route", "tool_code", "repair")})[:300])
    broken_answer = "Done:\n```python\ndef f(:\n    pass\n```"
    d = run_complete([reply(broken_answer)], messages=AFTER_TOOL)
    x = x_of(d)
    check(x["route"]["class"] == "agent_step" and x.get("repair") is None
          and len(_seen) == 1
          and d["choices"][0]["message"]["content"] == broken_answer,
          "agent_step: a final answer with broken code is not repaired",
          json.dumps(x["route"])[:200])
    check((x.get("selection") or {}).get("fanout_n") == 1,
          "agent_step: no fan-out")
    d = run_complete([reply("", [call("terminal", {"command": "npm test"})])],
                     messages=AFTER_TOOL)
    tc = x_of(d)["tool_code"]
    check(len(_seen) == 1 and tc["stopped"] == "no_code" and tc["rounds"] == 0
          and d["choices"][0]["message"].get("content") in ("", None),
          "agent_step: a non-code tool call passes untouched, no note",
          json.dumps(tc))
    helper(_fence(FIXED))
    d = run_complete([reply("", [wf(BROKEN)])], messages=AFTER_TOOL)
    tc = x_of(d)["tool_code"]
    check(x_of(d)["route"]["class"] == "agent_step" and tc["rounds"] == 1
          and tc["stopped"] == "fixed",
          "agent_step that WRITES a file: the call is checked and repaired "
          "(the response is code_edit work)", json.dumps(tc))


def test_medium_notes_and_does_not_fix():
    helper(_fence(FIXED))
    d = run_complete([reply("", [wf(BROKEN)])], effort="medium")
    tc = x_of(d).get("tool_code") or {}
    calls = d["choices"][0]["message"]["tool_calls"]
    note = d["choices"][0]["message"]["content"] or ""
    check(not _helper_seen and len(_seen) == 1
          and args_of(calls[0])["content"] == BROKEN
          and tc.get("stopped") == "noted" and tc.get("fix") is False,
          "medium: the write is checked, nothing generates, the call goes "
          "out as written", json.dumps(tc))
    check(note.startswith("Checked js/game.js (javascript): ")
          and "not repaired at this effort" in note,
          "and the note says what the check found", note)


def test_final_answer_repair_shares_the_cap():
    code_q = [{"role": "user", "content": "Write a Python function that adds "
                                          "two numbers."}]
    helper(*[_fence(f"def add{i}(a, b:\n    return a + b\n", "python")
             for i in range(1, 4)])
    d = run_complete([reply("```python\ndef add0(a, b:\n    return a + b\n```")],
                     messages=code_q, tools=None)
    rp = x_of(d).get("repair") or {}
    check(len(_seen) == 1 and len(_helper_seen) == 3
          and rp.get("rounds") == 3 and rp.get("stopped") == "not_fixed",
          "the final-answer repair is the same fixup job, with the same cap",
          json.dumps(rp))
    check("does not parse, and the repair did not fix it"
          in d["choices"][0]["message"]["content"],
          "and says the code still does not parse")
    efforts = {s.get("_effort_tier") for s in _helper_seen}
    check(efforts == {"high"}, "at the request's effort throughout",
          str(efforts))
    prose_q = [{"role": "user", "content": "Explain recursion briefly."}]
    helper()
    d = run_complete([reply("```python\ndef f(:\n```")], messages=prose_q,
                     tools=None)
    check(x_of(d).get("repair") is None and len(_seen) == 1
          and not _helper_seen,
          "a prose request gets no repair pass, even with broken code in it")


# ---------------------------------------------------------------------------
# Pre-deploy review, 2026-09-24 (BLOCKS DEPLOY #1): a failed or truncated
# fix-up overwrote the model's file. The repaired version replaces the
# model's content ONLY when it came from a CLOSED fence, the job's reply was
# not cut off (finish_reason "length"), it parses with no errors left, and
# it is plausibly complete (shomen.FIXUP_MIN_KEEP). Otherwise the model's own
# content goes, untouched, and the note says so truthfully.
# ---------------------------------------------------------------------------
BIG_BROKEN = ("class Game {\n" + "".join(
    f"  m{i}() {{\n    return {i};\n  }}\n" for i in range(12))
    + "  broken( {\n    return 0;\n  }\n}\n")
BIG_FIXED = BIG_BROKEN.replace("broken( {", "broken() {")
# A file that itself contains a bare ``` line (markdown in a string).
TICKS_BROKEN = ('DOC = """\n```js\nlet a = 1;\n```\n"""\n\n\n'
                "def f(:\n    return DOC\n")
TICKS_FIXED = TICKS_BROKEN.replace("def f(:", "def f():")


def _one(content: str, *replies, path: str = "js/game.js"
         ) -> tuple[dict, dict, str]:
    helper(*replies)
    d = run_complete([reply("", [wf(content, path=path)])])
    calls = d["choices"][0]["message"].get("tool_calls") or []
    return ((args_of(calls[0]) if calls else {}),
            x_of(d).get("tool_code") or {},
            d["choices"][0]["message"].get("content") or "")


def test_a_failed_or_truncated_fixup_never_overwrites_the_file():
    # (b) finish_reason "length": the job's reply was cut off -- even though
    # its (closed) block would parse.
    a, tc, note = _one(BROKEN, (_fence(FIXED), "length"))
    check(a.get("content") == BROKEN
          and (tc.get("fixup") or {}).get("written") == 0
          and "cut off" in note and "sent as written" in note,
          "a fix-up reply cut off by finish_reason length is not used; the "
          "model's content goes and the note says why", note)
    # (a) an unclosed fence: the reply ran out mid-block.
    a, tc, note = _one(BIG_BROKEN, "```javascript\n"
                       + BIG_FIXED[:len(BIG_FIXED) // 3])
    check(a.get("content") == BIG_BROKEN and tc.get("stopped") == "not_fixed"
          and "no closed fenced block" in note,
          "an UNCLOSED fence is never treated as running to the end: the "
          "model's content goes", note)
    # The fence mismatch: the file holds a bare ``` line, the reply fences
    # it with ``` -- the block closes early, on a fragment.
    a, tc, note = _one(TICKS_BROKEN, "```python\n" + TICKS_FIXED + "```",
                       path="tools/doc.py")
    check(a.get("content") == TICKS_BROKEN and "not used" in note,
          "a reply whose fence the file's own ``` line closes is a fragment, "
          "never written", json.dumps(tc.get("fixup")))
    hm = _helper_seen[0]["messages"][-1]["content"] if _helper_seen else ""
    check("````python\n" + TICKS_BROKEN.rstrip() + "\n````" in hm,
          "the job is shown such a file in a fence longer than any of its "
          "own backtick lines", hm[-300:])
    # ... and in the longer fence it was shown, the same fix IS applied.
    a, tc, note = _one(TICKS_BROKEN, "````python\n" + TICKS_FIXED + "````",
                       path="tools/doc.py")
    check(a.get("content") == TICKS_FIXED and tc.get("stopped") == "fixed",
          "the same repair in a fence the file cannot close is written back",
          json.dumps(tc.get("fixup")))
    # (d) plausibly complete: a closed, parsing, drastically shorter block.
    a, tc, note = _one(BIG_BROKEN, _fence("class Game {}\n"))
    check(a.get("content") == BIG_BROKEN
          and "under 50% of the original" in note
          and "sent as written" in note,
          "a repaired version under half the original's length is a "
          "fragment: not used, and the note says so", note)
    # (c) errors left after the cap: test_the_cap_and_the_last_version_is_sent.
    # The accepted case, for contrast:
    a, tc, note = _one(BIG_BROKEN, _fence(BIG_FIXED))
    check(a.get("content") == BIG_FIXED and note.startswith("Repaired"),
          "a closed, complete, parsing repair is written back", note)
    # shomen.fixup's record directly: a rejected unit carries the original.
    helper((_fence(FIXED), "length"))
    job = shomen.fixup([{"path": "x.js", "language": "javascript",
                         "kind": "file", "code": BROKEN,
                         "errors": [{"line": 5, "message": "x"}]}],
                       "r", check=tool_code.recheck, tier="high")
    u = job["units"][0]
    check(u["code"] == BROKEN and not u["accepted"] and not u["changed"]
          and u["errors_after"] == 1 and "length" in (u["rejected"] or "")
          and job["ok"] is False,
          "fixup(): a rejected unit returns the ORIGINAL code, its errors, "
          "and why", json.dumps({k: u[k] for k in ("accepted", "changed",
                                                     "rejected",
                                                     "errors_after")}))


# ---------------------------------------------------------------------------
# 6. The stream
# ---------------------------------------------------------------------------
def _deltas(ev: list[dict]) -> list[tuple[str, object]]:
    out = []
    for e in ev:
        for ch in e.get("choices") or []:
            dl = ch.get("delta") or {}
            for k in ("reasoning_content", "content", "tool_calls"):
                if dl.get(k):
                    out.append((k, dl[k]))
    return out


def test_the_stream_repairs_and_keeps_channel_order():
    helper(_fence(FIXED))
    ev = run_stream([reply("Writing it now.", [wf(BROKEN)],
                           reasoning="I should write the class.")])
    ds = _deltas(ev)
    kinds = [k for k, _ in ds]
    first_content = kinds.index("content") if "content" in kinds else len(kinds)
    check("reasoning_content" not in kinds[first_content:],
          "CHANNEL ORDER: no reasoning after the first content delta", str(kinds))
    reasoning = "".join(v for k, v in ds if k == "reasoning_content")
    check("I should write the class." in reasoning
          and "fixing js/game.js" in reasoning,
          "main's thinking and the fix-up go out as reasoning",
          reasoning[:300])
    tcs = [v for k, v in ds if k == "tool_calls"]
    check(len(tcs) == 1 and kinds[-1] == "tool_calls",
          "exactly one tool_calls delta, and it is the last delta", str(kinds))
    fwd = tcs[0] if tcs else []
    check(len(fwd) == 1 and args_of(fwd[0])["content"] == FIXED,
          "it carries the fixed call", json.dumps(fwd)[:200])
    content = "".join(v for k, v in ds if k == "content")
    check(content.startswith("Writing it now.\n\nRepaired js/game.js")
          and kinds[-2] == "content",
          "the note goes out once, as content, just before the tool_calls "
          "delta", content)
    check(BROKEN not in json.dumps(ev), "the broken call never went out")
    tail = ev[ev.index(next(e for e in ev if any(
        (c.get("delta") or {}).get("tool_calls")
        for c in e.get("choices") or []))) + 1:]
    check(all(not (c.get("delta") or {}) for e in tail
              for c in e.get("choices") or [])
          and tail[-1].get("_done"),
          "after the calls: only the finish chunk and [DONE] -- nothing "
          "changes what was forwarded", json.dumps(tail)[:300])
    fin = [e for e in tail if e.get("x_yamadori")]
    tc = (fin[-1]["x_yamadori"].get("tool_code") if fin else None) or {}
    check(tc.get("rounds") == 1 and tc.get("stopped") == "fixed"
          and fin[-1]["choices"][0]["finish_reason"] == "tool_calls",
          "the final chunk carries x_yamadori.tool_code", json.dumps(tc))


def test_the_stream_cap_sends_the_last_version():
    helper(_fence(BROKEN2), _fence(BROKEN3), _fence(BROKEN4))
    ev = run_stream([reply("", [wf(BROKEN)])])
    ds = _deltas(ev)
    tcs = [v for k, v in ds if k == "tool_calls"]
    check(len(_seen) == 1 and len(tcs) == 1
          and args_of(tcs[0][0])["content"] == BROKEN,
          "streamed: at the cap the model's own content is sent, once")
    content = "".join(v for k, v in ds if k == "content")
    check(content.startswith("Checked js/game.js") and "still there" in
          content,
          "with the note, and no blank line before it when the model wrote "
          "nothing", content)


def main() -> int:
    for fn in (test_every_table_entry_and_the_fallback,
               test_the_checks,
               test_fake_upstream_is_alive,
               test_a_broken_write_file_is_repaired_before_forwarding,
               test_the_cap_and_the_last_version_is_sent,
               test_a_failed_or_truncated_fixup_never_overwrites_the_file,
               test_formatter_output_is_applied_and_noted,
               test_edits_block_only_a_complete_unit,
               test_a_utility_call_and_agent_steps_stay_out_of_the_code_pipelines,
               test_medium_notes_and_does_not_fix,
               test_final_answer_repair_shares_the_cap,
               test_the_stream_repairs_and_keeps_channel_order,
               test_the_stream_cap_sends_the_last_version):
        print(f"\n--- {fn.__name__} ---")
        n0 = len(_results)
        try:
            fn()
        except Exception:                                        # noqa: BLE001
            check(False, f"{fn.__name__} itself raised",
                  traceback.format_exc().strip().split("\n")[-1])
            traceback.print_exc()
        for ok, name, detail in _results[n0:]:
            print(("  pass  " if ok else "  FAIL  ") + name
                  + (f"   <- {detail}" if not ok and detail else ""))
    _srv.shutdown()
    passed = sum(1 for ok, _, _ in _results if ok)
    total = len(_results)
    print(f"\n{'=' * 70}\n  {passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
