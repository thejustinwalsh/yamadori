#!/usr/bin/env python
"""Pi's and OpenCode's shapes through the proxy, rendered by the SERVED
template. No GPU, nothing live.

    python mcp/test_harness_forms.py      -> "N/M checks passed"

WHAT THIS GATES (docs/HARNESS-PI.md, docs/HARNESS-OPENCODE.md, 2026-09-26):

  1. THE `developer` ROLE. Pi sends its instructions as `developer` when a
     model has `reasoning: true`. The addendum (high and up) used to put a
     SECOND system message in front of it; the template raised "System
     message must be at the beginning" and the proxy answered 502, which Pi
     retried three times. Now: one system message, `developer` mapped on the
     way in, every turn alike (the prefix extends); a template refusal is a
     400 `invalid_prompt`, never a 5xx.
  2. COMPACTION FORMS. Pi's `<conversation>` / `# Conversation` summaries and
     OpenCode's "Here is the conversation so far:" are recognised, mapped
     onto the stored conversation (its prompt, extended), and think at the
     conversation's effort; one that maps onto nothing thinks at the effort
     the client sent. OpenCode's title call is kind `title`.
  3. THE CACHE AFTER A HIDDEN yama_describe_image HOP. An echoing client (Pi,
     OpenCode) after a turn whose image the proxy looked at in a hidden hop:
     the next request must EXTEND what the slot holds. Pi's log: 1,829 of
     3,663 prompt tokens reused (#22 -> #23).
  4. THE SYNTHETIC TOOL-MEDIA TURN is not the user speaking
     (selection.question_of).

The fake upstream behaves as llama-server does where it matters: it maps
`developer` to `system` before rendering (Pi's #0 vs #1: the same 2,345
prompt tokens either way), renders with mcp/fixtures/bonsai_chat_template
.jinja, and answers a template refusal with a 500 whose message carries the
Jinja exception, as Pi's #2-#5 show.
"""
from __future__ import annotations

import base64
import json
import os
import struct
import sys
import tempfile
import threading
import time
import traceback
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# EVERY store in a temp dir BEFORE the proxy is imported (an earlier suite
# leaked its rows into the live databases).
_TMP = tempfile.mkdtemp(prefix="yamadori_test_harness_forms_")
os.environ["YAMADORI_CORPUS_DB"] = os.path.join(_TMP, "corpus.sqlite3")
os.environ["YAMADORI_NEBARI_DB"] = os.path.join(_TMP, "nebari.sqlite3")
os.environ["RINGS_DB"] = os.path.join(_TMP, "rings.sqlite3")
os.environ["YAMADORI_PKG_DIR"] = os.path.join(_TMP, "pkgs")
os.makedirs(os.environ["YAMADORI_PKG_DIR"], exist_ok=True)
os.environ["CODE_INDEX_DB"] = os.path.join(_TMP, "code.sqlite3")
os.environ["CONCEPT_SEED_LAST"] = os.path.join(_TMP, "seed_last.json")
os.environ["YAMADORI_SKILLS_DIR"] = os.path.join(_TMP, "skills")
os.environ["YAMADORI_JOBS_DB"] = os.path.join(_TMP, "jobs.sqlite3")
os.environ["YAMADORI_POWER_LEDGER"] = os.path.join(_TMP, "power.jsonl")
os.environ["YAMADORI_TOKEN_LEDGER"] = os.path.join(_TMP, "tokens.sqlite3")
os.environ["YAMADORI_SKILL_LABELS"] = os.path.join(_TMP, "skill_labels")
os.environ["YAMADORI_SKILL_TRIGGER_CACHE"] = os.path.join(_TMP, "trig")
os.environ.pop("YAMADORI_IMAGEGEN_URL", None)
os.environ["LLAMA_STACK_URL"] = "http://127.0.0.1:9"
os.environ["YAMADORI_MODEL_SERVER"] = "http://127.0.0.1:9"
os.environ["YAMADORI_SLOTS"] = "4"

import jinja2  # noqa: E402

import api_errors  # noqa: E402
import compaction  # noqa: E402
import proxy  # noqa: E402
import selection  # noqa: E402
import slots  # noqa: E402
import tiers  # noqa: E402
import model as _model  # noqa: E402

tiers._accepted = ("low", "medium", "xhigh")

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


# ------------------------------------------------------- the served template
_env = jinja2.Environment()
_env.filters["tojson"] = lambda v, **k: json.dumps(v, ensure_ascii=False)
_TEMPLATE = _env.from_string(open(os.path.join(
    HERE, "fixtures", "bonsai_chat_template.jinja"), encoding="utf-8").read())


class TemplateError(Exception):
    pass


def _raise(msg):
    raise TemplateError(msg)


def _as_template_input(messages: list[dict]) -> list[dict]:
    """What llama-server hands the template: `developer` as `system`,
    tool-call arguments parsed, null content as ""."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("role") == "developer":
            m["role"] = "system"
        if m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                fn = dict(c.get("function") or {})
                a = fn.get("arguments")
                if isinstance(a, str):
                    try:
                        fn["arguments"] = json.loads(a) if a.strip() else {}
                    except ValueError:
                        pass
                calls.append(dict(c, function=fn))
            m["tool_calls"] = calls
        if m.get("content") is None:
            m["content"] = ""
        out.append(m)
    return out


def render(body: dict, generation: bool = True) -> str:
    msgs = _as_template_input(body.get("messages") or [])
    kw = {}
    if body.get("reasoning_effort"):
        kw["reasoning_effort"] = body["reasoning_effort"]
    ctk = body.get("chat_template_kwargs") or {}
    if "enable_thinking" in ctk:
        kw["enable_thinking"] = ctk["enable_thinking"]
    prefill = bool(msgs) and msgs[-1].get("role") == "assistant"
    text = _TEMPLATE.render(messages=msgs, tools=body.get("tools") or None,
                            add_generation_prompt=generation and not prefill,
                            raise_exception=_raise, **kw)
    if prefill and text.endswith("<|im_end|>\n"):
        text = text[:-len("<|im_end|>\n")]
    return text


# --------------------------------------------------------- the fake upstream
_script: list[dict] = []
_gens: list[dict] = []
_refused: list[str] = []
_warms: list[str] = []            # the prompts the proxy warmed a slot with


def reply(content: str = "", reasoning: str = "", calls: list | None = None
          ) -> dict:
    return {"content": content, "reasoning": reasoning, "calls": calls or []}


def call(name: str, args: dict, cid: str) -> dict:
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def _sse(r: dict, finish: str) -> bytes:
    def ev(delta=None, fin=None):
        return (b"data: " + json.dumps({
            "id": "u", "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": delta or {},
                         "finish_reason": fin}]}).encode() + b"\n\n")
    out = [ev({"role": "assistant"})]
    if r["reasoning"]:
        out.append(ev({"reasoning_content": r["reasoning"]}))
    s = r["content"]
    out += [ev({"content": s[i:i + 40]}) for i in range(0, len(s), 40)]
    for i, c in enumerate(r["calls"]):
        out.append(ev({"tool_calls": [dict(c, index=i)]}))
    out.append(ev({}, finish))
    out.append(b"data: " + json.dumps({"id": "u", "choices": [], "usage": {
        "prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
        "timings": {"cache_n": 90, "prompt_n": 10}}).encode() + b"\n\n")
    return b"".join(out) + b"data: [DONE]\n\n"


class _Up(BaseHTTPRequestHandler):
    def _json(self, obj: dict, status: int = 200) -> None:
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):                                            # noqa: N802
        self.send_response(404)
        self.end_headers()

    def do_POST(self):                                           # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path.endswith("/apply-template"):
            try:
                return self._json({"prompt": render(body)})
            except TemplateError as e:
                return self._json({"error": {"code": 500, "message": str(e),
                                             "type": "server_error"}}, 500)
        if self.path.endswith("/tokenize"):
            return self._json({"tokens": [1] * len(body.get("content") or "")})
        if self.path.endswith("/completion"):
            _warms.append(body.get("prompt") or "")
            return self._json({"timings": {"cache_n": 100, "prompt_n": 0}})
        try:
            render(body)
        except TemplateError as e:
            # llama-server's answer to a template's raise_exception: a 500
            # server_error carrying the Jinja trace (Pi's relay #2-#5).
            _refused.append(str(e))
            return self._json({"error": {
                "code": 500, "type": "server_error", "message":
                "------------\nWhile executing CallExpression at line 106, "
                "column 32 in source:\n...{{- raise_exception('System message "
                "must be at the beginnin...\n        ^\nError: Jinja "
                f"Exception: {e}"}}, 500)
        if not _script:
            self.send_response(500)
            self.end_headers()
            return
        r = dict(_script.pop(0))
        last = (body.get("messages") or [{}])[-1]
        if last.get("role") == "assistant":
            r["content"] = (last.get("content") or "") + r["content"]
            if last.get("reasoning_content"):
                r["reasoning"] = last["reasoning_content"] + "\n" \
                    + r["reasoning"]
        _gens.append({"request": body, "reply": r})
        data = _sse(r, "tool_calls" if r["calls"] else "stop")
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
_model.UPSTREAM = "http://127.0.0.1:9"

# ------------------------------------------------------------------ stubs
proxy._draw_seed = lambda prompt=None: {"word": "cedar", "token_id": 1,
                                        "u32": 1}
proxy._skills_tail = lambda messages, sel, route, client_tools=None, account="": (
    "", {"on": False, "why": "test: no skills"})
LOOKED = "A red square centred on white, with the white text PI-TEST."
_real_run_our_tool = proxy.run_our_tool


def _run_our_tool(name, args, db, root=None, turn=None, state=None):
    if name == "yama_describe_image":
        if state is not None:
            state.setdefault("_vision", []).append({"ok": True})
        return json.dumps({"ok": True, "answer": LOOKED})
    return _real_run_our_tool(name, args, db, root, turn, state)


proxy.run_our_tool = _run_our_tool


def _png(w: int = 4, h: int = 4) -> bytes:
    def chunk(t: bytes, d: bytes) -> bytes:
        return (struct.pack(">I", len(d)) + t + d
                + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff))
    raw = b"".join(b"\x00" + b"\xff\x00\x00" * w for _ in range(h))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


DATA_URL = "data:image/png;base64," + base64.b64encode(_png()).decode()

# ------------------------------------------------------------ Pi's shapes
# pi-coding-agent 0.87.1: its default tools (docs/HARNESS-PI.md section 1)
# and the head of its system prompt (relay #0).
PI_SYSTEM = ("You are an expert coding assistant operating inside pi, a "
             "coding agent harness. You help users by reading files, "
             "executing commands, editing code, and writing new files.")
PI_TOOLS = [{"type": "function", "function": {
    "name": n, "description": f"pi {n}",
    "parameters": {"type": "object", "properties": {
        k: {"type": "string"} for k in keys}}}}
    for n, keys in (("read", ("path",)), ("bash", ("command",)),
                    ("edit", ("path",)), ("write", ("path", "content")))]
# dist/core/compaction/utils.js:139 and compaction.js:400 (the head).
PI_SUMMARY_SYSTEM = (
    "You are a context summarization assistant. Your task is to read a "
    "conversation between a user and an AI assistant, then produce a "
    "structured summary following the exact format specified.\n\nDo NOT "
    "continue the conversation. Do NOT respond to any questions in the "
    "conversation. ONLY output the structured summary.")
PI_SUMMARY_PROMPT = (
    "The messages above are a conversation to summarize. Create a structured "
    "context checkpoint summary that another LLM will use to continue the "
    "work.\n\nUse this EXACT format:\n\n## Goal\n[What is the user trying to "
    "accomplish?]")
# opencode-ai 1.18.32: the title agent and the compaction buildPrompt.
OC_TITLE_SYSTEM = (
    "You are a title generator. You output ONLY a thread title. Nothing "
    "else.\n\n<task>\nGenerate a brief title that would help the user find "
    "this conversation later.\n\nFollow all rules in <rules>\n</task>\n\n"
    "<rules>\n- you MUST use the same language as the user message you are "
    "summarizing\n- NEVER respond to questions, just generate a title for "
    "the conversation\n</rules>")
OC_SYSTEM = "You are opencode, an interactive CLI tool that helps users."
OC_TOOLS = [{"type": "function", "function": {
    "name": n, "description": f"opencode {n}",
    "parameters": {"type": "object", "properties": {
        "filePath": {"type": "string"}}}}} for n in ("read", "write", "bash")]
OC_ANCHOR = ("Create a new anchored summary from the conversation history in "
             "the <conversation> tags above so another coding agent can "
             "continue the work.")
ACCOUNT = "harness-forms"


class Client:
    """A harness that keeps what it was sent and echoes its reasoning (Pi and
    OpenCode both echo `reasoning_content` on every replayed turn)."""

    def __init__(self, effort: str | None, role: str = "developer",
                 system: str = PI_SYSTEM, tools=None, key: str | None = None):
        self.effort, self.key = effort, key
        self.tools = PI_TOOLS if tools is None else tools
        self.msgs: list[dict] = [{"role": role, "content": system}]
        self.features: dict = {}      # more X-Yamadori-Features

    def body(self, stream: bool = False) -> dict:
        b = {"model": "yamadori", "_account": getattr(self, "account",
                                                      ACCOUNT),
             "_client_ip": "127.0.0.1", "tools": self.tools,
             "messages": json.loads(json.dumps(self.msgs)),
             "_features": json.dumps(dict(
                 {"skills": False, "investigate": False, "fanout": 1,
                  "retrieval": False}, **self.features))}
        if self.effort:
            b["reasoning_effort"] = self.effort
        if self.key:
            b["prompt_cache_key"] = self.key
        if stream:
            b["stream"] = True
        return b

    def turn(self, script: list[dict], user=None, stream: bool = False
             ) -> dict:
        if user is not None:
            self.msgs.append({"role": "user", "content": user})
        _script[:] = list(script)
        n0, w0 = len(_gens), len(_warms)
        if stream:
            got = _stream(self.body(True))
            kept = {"role": "assistant", "content": got["content"],
                    "reasoning_content": got["reasoning"]}
            if got["calls"]:
                kept["tool_calls"] = got["calls"]
            d = {"x_yamadori": got["x"]}
        else:
            d = proxy.complete(self.body())
            m = d["choices"][0]["message"]
            kept = {"role": "assistant", "content": m.get("content") or "",
                    "reasoning_content": m.get("reasoning_content") or ""}
            if m.get("tool_calls"):
                kept["tool_calls"] = m["tool_calls"]
        self.msgs.append(kept)
        if ((d.get("x_yamadori") or {}).get("warm") or {}).get("sent"):
            end = time.time() + 5
            while time.time() < end and len(_warms) == w0:
                time.sleep(0.02)
        return {"d": d, "gens": _gens[n0:], "kept": kept,
                "warm": _warms[w0:]}

    def tool_result(self, text: str) -> None:
        cid = self.msgs[-1]["tool_calls"][0]["id"]
        self.msgs.append({"role": "tool", "tool_call_id": cid,
                          "content": text})


def _stream(body: dict) -> dict:
    out = {"content": "", "reasoning": "", "calls": [], "x": {}}
    for b in proxy.stream_body(dict(body, stream=True)):
        for line in b.decode("utf-8").split("\n"):
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            ev = json.loads(line[5:].strip())
            if ev.get("x_yamadori"):
                out["x"] = ev["x_yamadori"]
            for ch in ev.get("choices") or []:
                dl = ch.get("delta") or {}
                out["content"] += dl.get("content") or ""
                out["reasoning"] += dl.get("reasoning_content") or ""
                for c in dl.get("tool_calls") or []:
                    out["calls"].append({k: v for k, v in c.items()
                                         if k != "index"})
    return out


def holds_after(t: dict) -> str:
    """What the slot holds after a turn: the prompt it was warmed with, else
    the last generation's prompt and what it generated."""
    if t.get("warm"):
        return t["warm"][-1]
    g = t["gens"][-1]
    req = dict(g["request"])
    msgs = list(req.get("messages") or [])
    if msgs and msgs[-1].get("role") == "assistant":
        msgs = msgs[:-1]
    r = g["reply"]
    gen = {"role": "assistant", "content": r["content"],
           "reasoning_content": r["reasoning"]}
    if r["calls"]:
        gen["tool_calls"] = r["calls"]
    return render(dict(req, messages=msgs + [gen]), generation=False)


def _div(a: str, b: str) -> int:
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y),
                min(len(a), len(b)))


def _extends_fully(prev: dict, nxt: dict, label: str) -> bool:
    h, r = holds_after(prev), render(nxt["gens"][0]["request"])
    n = _div(h, r)
    return check(r.startswith(h),
                 f"{label}: the next request extends EVERYTHING the slot "
                 f"holds ({len(h)} chars)",
                 f"diverges at char {n} of {len(h)}: held "
                 f"{h[max(0, n - 100):n + 60]!r} vs sent "
                 f"{r[max(0, n - 100):n + 60]!r}")


def _reset() -> None:
    slots.reset(n=4)
    compaction.reset()
    _script.clear()
    _refused.clear()


# ============================================================ 1. developer
def test_the_developer_role():
    _reset()
    # The Pi request of relay #2: developer + user, tools, effort high.
    for effort in ("high", "xhigh", "max"):
        c = Client(effort)
        c.msgs.append({"role": "user",
                       "content": "What is 4+4? Answer in one word."})
        _script[:] = [reply("Eight.", reasoning="4+4=8.")]
        n0 = len(_gens)
        try:
            d = proxy.complete(c.body())
            err = None
        except Exception as e:                                   # noqa: BLE001
            d, err = None, e
        up = (_gens[n0]["request"]["messages"] if len(_gens) > n0 else [])
        roles = [m.get("role") for m in up]
        check(d is not None and not _refused and roles[:1] == ["system"]
              and roles.count("system") == 1 and "developer" not in roles
              and up[0]["content"].startswith(PI_SYSTEM)
              and "A second model works beside you" in up[0]["content"],
              f"{effort}: Pi's `developer` message is THE system message, "
              f"the addendum at its end -- one system message, rendered",
              f"err={err!r} refused={_refused} roles={roles}")
        _refused.clear()
    # The same conversation over two turns: the second extends the first.
    _reset()
    c = Client("high")
    t1 = c.turn([reply("Eight.", reasoning="4+4=8.")],
                user="What is 4+4? Answer in one word.")
    t2 = c.turn([reply("Nine.", reasoning="4+5=9.")],
                user="And 4+5?")
    _extends_fully(t1, t2, "developer role, turn 2")
    x = t2["d"].get("x_yamadori") or {}
    check((x.get("roles") or {}).get("merged") == 1,
          "x_yamadori.roles records the mapping (system_roles.one_system's "
          "record)", json.dumps(x.get("roles")))
    # medium (no addendum): still mapped, the same rendering as `system`.
    _reset()
    c = Client("medium")
    t = c.turn([reply("Eight.")], user="What is 4+4?")
    sent = t["gens"][0]["request"]
    check(sent["messages"][0]["role"] == "system"
          and render(sent) == render(dict(sent, messages=[
              dict(sent["messages"][0], role="system")]
              + sent["messages"][1:])),
          "medium: `developer` goes up as `system` (what llama-server "
          "renders anyway)")
    # A system AND a developer message first; a developer message later:
    # the shared helper (mcp/system_roles.py, both wires) as the chat path
    # applies it -- through prepare.
    out = proxy.prepare({"model": "yamadori", "_account": ACCOUNT,
                         "reasoning_effort": "low", "messages": [
                             {"role": "system", "content": "A."},
                             {"role": "developer", "content": "B."},
                             {"role": "user", "content": "hi"},
                             {"role": "assistant", "content": "hello"},
                             {"role": "developer", "content": "Be brief."},
                             {"role": "user", "content": "bye"}]})
    msgs, rec = out["messages"], out.get("_roles") or {}
    check([m["role"] for m in msgs] == ["system", "user", "assistant",
                                        "user", "user"]
          and msgs[0]["content"].startswith("A.\n\nB.")
          and rec.get("merged") == 2 and rec.get("developer_as_user") == 1,
          "the leading instructions join into one system message; a later "
          "one becomes a user turn (the Responses adapter's rule)",
          json.dumps(msgs))
    same = [{"role": "system", "content": "A."},
            {"role": "user", "content": "hi"}]
    check(proxy.system_roles.one_system(same)[0] is same,
          "a request with one system message comes back as the same list")
    # add_addendum on a developer message with list content.
    out = proxy.add_addendum([{"role": "developer", "content": [
        {"type": "text", "text": "Rules."}]},
        {"role": "user", "content": "hi"}])
    check(len(out) == 2 and out[0]["role"] == "system"
          and out[0]["content"][-1]["text"] == proxy.addendum_text(),
          "add_addendum: a developer message's list content gets the "
          "addendum as its last text part, no second system message",
          json.dumps(out)[:300])


def test_a_template_refusal_is_a_400():
    e = api_errors.of_upstream(500, {
        "code": 500, "type": "server_error", "message":
        "------------\nWhile executing CallExpression at line 106, column 32 "
        "in source:\n...{{- raise_exception('System message must be at the "
        "beginnin...\n   ^\nError: Jinja Exception: System message must be "
        "at the beginning."})
    check(e.status == 400 and e.code == "invalid_prompt"
          and e.message.endswith("System message must be at the beginning.")
          and "CallExpression" not in e.message,
          "llama-server's 500 for a template refusal is a 400 invalid_prompt "
          "with the template's own sentence (a client retries a 5xx)",
          f"{e.status} {e.code} {e.message!r}")
    e = api_errors.of_upstream(500, {"message": "CUDA error: out of memory",
                                     "type": "server_error"})
    check(e.status == 502, "any other 500 stays a 502", str(e.status))
    # Through the proxy: messages the template refuses whatever we do (no
    # user query at all).
    _reset()
    body = {"model": "yamadori", "_account": ACCOUNT, "reasoning_effort":
            "low", "messages": [{"role": "system", "content": "S."},
                                {"role": "assistant", "content": "hi"}]}
    try:
        proxy.complete(body)
        got = None
    except Exception as ex:                                      # noqa: BLE001
        got = api_errors.of_exception(ex)
    check(got is not None and got.status == 400
          and got.code == "invalid_prompt",
          "through the proxy: a refused prompt is a 400, not a 502",
          f"{getattr(got, 'status', None)} {getattr(got, 'code', None)}")


# ============================================================ 2. compaction
def _pi_conversation(effort: str = "low") -> Client:
    """A Pi session of three requests, as Pi sends them (developer role,
    reasoning echoed, its own tool-call ids)."""
    c = Client(effort, key="pid1")
    c.turn([reply("", reasoning="Write it.", calls=[call(
        "write", {"path": "hello.py",
                  "content": "def greet(name):\n    return 'hello ' + name\n"},
        "w1")])],
        user="Create hello.py containing a function greet(name) that returns "
             "the string 'hello ' + name.")
    c.tool_result("Successfully wrote to hello.py")
    c.turn([reply("Done: hello.py defines greet.", reasoning="Done.")])
    c.turn([reply("", reasoning="Run it.", calls=[call(
        "bash", {"command": "py -c \"import hello\""}, "b1")])],
        user="Now run it.")
    c.tool_result("(no output)")
    c.turn([reply("It imports cleanly.", reasoning="Fine.")])
    return c


def _pi_records(msgs: list[dict]) -> str:
    """Pi's serializeConversation (utils.js:94) over the client's copy."""
    parts = []
    for m in msgs:
        if m["role"] == "user":
            parts.append(f"[User]: {m['content']}")
        elif m["role"] == "assistant":
            if m.get("reasoning_content"):
                parts.append(f"[Assistant thinking]: {m['reasoning_content']}")
            if m.get("content"):
                parts.append(f"[Assistant]: {m['content']}")
            if m.get("tool_calls"):
                parts.append("[Assistant tool calls]: " + "; ".join(
                    c["function"]["name"] + "(" + ", ".join(
                        f"{k}={json.dumps(v)}" for k, v in json.loads(
                            c["function"]["arguments"]).items()) + ")"
                    for c in m["tool_calls"]))
        elif m["role"] == "tool":
            parts.append(f"[Tool result]: {m['content']}")
    return "\n\n".join(parts)


def test_pi_compaction_is_mapped_and_thinks():
    _reset()
    c = _pi_conversation("low")
    last = _gens[-1]
    records = _pi_records(c.msgs[1:])
    text = f"<conversation>\n{records}\n</conversation>\n\n{PI_SUMMARY_PROMPT}"
    parsed = compaction.parse_flattened(text)
    check(parsed and parsed["harness"] == "pi"
          and [r["role"] for r in parsed["records"]]
          == ["user", "assistant", "tool", "assistant", "user", "assistant",
              "tool", "assistant"]
          and parsed["records"][1].get("calls") == ["write"],
          "Pi's <conversation> transcript parses: one record per message, "
          "thinking folded in, tool calls by name",
          json.dumps(parsed["records"] if parsed else None)[:400])
    # The compaction call itself: Pi strips its session (no key, no
    # headers), sends its thinking level, and max_completion_tokens.
    _script[:] = [reply("## Goal\nhello.py with greet.", reasoning="Sum.")]
    n0 = len(_gens)
    d = proxy.complete({
        "model": "yamadori", "_account": ACCOUNT, "reasoning_effort": "low",
        "max_completion_tokens": 13107,
        "messages": [{"role": "system", "content": PI_SUMMARY_SYSTEM},
                     {"role": "user", "content": text}]})
    x = d["x_yamadori"]
    comp = x.get("compaction") or {}
    up = _gens[n0]["request"]
    check(x.get("utility_kind") == "compaction" and comp.get("harness") == "pi"
          and comp.get("mode") == "rewritten",
          "Pi's summary call: a compaction, harness pi, mapped onto the "
          "stored conversation", json.dumps(comp)[:400])
    held = render(dict(last["request"], messages=list(
        last["request"]["messages"]) + [{
            "role": "assistant", "content": last["reply"]["content"],
            "reasoning_content": last["reply"]["reasoning"]}]),
        generation=False)
    sent = render(up)
    check(sent.startswith(held),
          "it extends the conversation's last prompt and answer, byte for "
          "byte (the slot's cache)", f"diverges at {_div(held, sent)}")
    check(up.get("enable_thinking") is True
          and up.get("reasoning_budget_tokens") == comp.get("thinking_tokens")
          and comp.get("thinking", "").startswith("on"),
          "it THINKS, at the conversation's effort",
          json.dumps({k: up.get(k) for k in ("enable_thinking",
                                              "reasoning_effort")}))
    ins = up["messages"][-1]["content"]
    check(ins.startswith(PI_SUMMARY_SYSTEM) and "[User]:" not in ins
          and "The turns to summarise are in the conversation above" in ins,
          "the instruction: Pi's own system text, then a reference in place "
          "of the copied transcript", ins[:300])


def test_an_unmapped_compaction_thinks_at_the_clients_effort():
    _reset()
    text = ("<conversation>\n[User]: Plan a garden.\n\n[Assistant]: Raised "
            "beds.\n</conversation>\n\n" + PI_SUMMARY_PROMPT)
    _script[:] = [reply("## Goal\nA garden.")]
    n0 = len(_gens)
    d = proxy.complete({
        "model": "yamadori", "_account": ACCOUNT + "-other",
        "reasoning_effort": "low", "max_completion_tokens": 13107,
        "messages": [{"role": "system", "content": PI_SUMMARY_SYSTEM},
                     {"role": "user", "content": text}]})
    comp = d["x_yamadori"].get("compaction") or {}
    up = _gens[n0]["request"]
    check(comp.get("mode") == "as_sent" and comp.get("harness") == "pi"
          and up.get("enable_thinking") is True
          and up.get("reasoning_effort") == tiers.safe_effort(
              tiers.TIERS["low"]["effort"])
          and "client" in comp.get("thinking", ""),
          "Pi's compaction with nothing stored: thinks at the effort the "
          "client sent (tier low), and says whose it is -- not 'the "
          "conversation itself runs without it'",
          json.dumps({"thinking": comp.get("thinking"),
                      "enable_thinking": up.get("enable_thinking"),
                      "effort": up.get("reasoning_effort")}))
    # The split-turn form (compaction.js:751).
    t2 = ("# Conversation\n[User]: /compact\n\n[Assistant]: ok\n\n"
          "# Instructions\nThe messages above are earlier context.")
    p = compaction.parse_flattened(t2)
    check(p and p["harness"] == "pi" and len(p["records"]) == 2,
          "Pi's # Conversation form parses", json.dumps(p)[:200])


def test_opencode_title_and_compaction():
    u, kind = selection.utility_call, selection.utility_kind
    title = [{"role": "system", "content": OC_TITLE_SYSTEM},
             {"role": "user", "content": "Generate a title for this "
                                         "conversation:\n"},
             {"role": "user", "content": "\"What is the capital of France? "
                                         "Answer in one word.\""}]
    check(kind(u(title, [])) == "title",
          "OpenCode's title call is kind `title`, not compaction",
          json.dumps(u(title, [])))
    task = [{"role": "system", "content": OC_SYSTEM},
            {"role": "user", "content": "\"Create hello.py that prints hello\" "
                                        "from \"opencode, then run it\""}]
    check(not u(task, ["bash", "read", "write"])["utility"],
          "an OpenCode task turn (its tools sent) is not a utility call")
    # OpenCode's compaction: one user message, no system, no tools.
    _reset()
    c = Client(None, role="system", system=OC_SYSTEM, tools=OC_TOOLS)
    c.turn([reply("Paris.", reasoning="Capital.")],
           user="What is the capital of France?")
    c.turn([reply("Berlin.", reasoning="Capital.")],
           user="And of Germany?")
    records = "\n\n".join(
        (f"[User]: {m['content']}" if m["role"] == "user" else
         f"[Assistant reasoning]: {m['reasoning_content']}\n"
         f"[Assistant]: {m['content']}") for m in c.msgs[1:])
    text = (f"Here is the conversation so far:\n\n<conversation>\n{records}"
            f"\n</conversation>\n\n{OC_ANCHOR}\nOutput exactly the Markdown "
            f"structure shown inside <template>.")
    d0 = u([{"role": "user", "content": text}], [])
    check(kind(d0) == "compaction",
          "OpenCode's compaction (its summarise verb after the transcript) "
          "is a compaction", json.dumps(d0))
    _script[:] = [reply("## Objective\n- capitals")]
    n0 = len(_gens)
    d = proxy.complete({"model": "yamadori", "_account": ACCOUNT,
                        "messages": [{"role": "user", "content": text}]})
    comp = d["x_yamadori"].get("compaction") or {}
    up = _gens[n0]["request"]
    check(comp.get("harness") == "opencode" and comp.get("mode") == "rewritten"
          and up["messages"][0]["content"] == OC_SYSTEM,
          "OpenCode's compaction maps onto its conversation's stored prompt",
          json.dumps(comp)[:300])


# ======================================================= 3. the image hop
def test_the_cache_after_a_hidden_describe_image_hop():
    """Pi's #21-#23: read an image, the proxy looks at it in a hidden hop,
    the model writes a file; the request after the write's result must
    extend the slot -- everything, reasoning included, since Pi echoes."""
    for tag, role, system, tools, media_text, key in (
            # Pi: developer role; "Attached image(s) from tool result:"
            # (openai-completions.ts:1455).
            ("pi", "developer", PI_SYSTEM, PI_TOOLS,
             "Attached image(s) from tool result:", "piv2"),
            # OpenCode: "Attached media from tool result:"
            # (message-v2.ts:46); its 3a/4 runs showed the same drop
            # (7,438 of 9,674 reused).
            ("opencode", "system", OC_SYSTEM, OC_TOOLS,
             "Attached media from tool result:", None)):
        _reset()
        c = Client("medium", role=role, system=system, tools=tools, key=key)
        c.turn([reply("", reasoning="Read it.", calls=[call(
            "read", {"path": "pitest.png"}, "r1")])],
            user="Use the read tool to open pitest.png and tell me what it "
                 "shows. Then use the write tool to create canvas.html.",
            stream=True)
        c.tool_result("Read image file [image/png]")
        # Pi's synthetic user turn (openai-completions.ts:1455).
        c.msgs.append({"role": "user", "content": [
            {"type": "text", "text": media_text},
            {"type": "image_url", "image_url": {"url": DATA_URL}}]})
        q = selection.question_of(proxy.vision.normalise(c.msgs)[0])
        check(not q[2] and q[0].startswith("Use the read tool"),
              f"{tag}: question_of: the synthetic tool-media turn is not the user "
              "speaking; the question is the real user turn", repr(q)[:200])
        img_id = None
        t2 = c.turn([
            reply("", reasoning="Look at it first.", calls=[call(
                "yama_describe_image", {"image": "image-x", "question": "What is it?"},
                "d1")]),
            # Arguments in sorted key order, as in Pi's #22, whose warm record
            # said "the slot already holds the turn as delivered": no warm, so
            # the next request must extend the slot on its own.
            reply("It shows a red square.", reasoning="Now write the page.",
                  calls=[call("write", {"content": "<canvas></canvas>",
                                        "path": "canvas.html"}, "w1")])],
            stream=True)
        check(not t2["warm"], f"{tag}: no warm (the delivered turn is what "
                              "the slot generated)", str(len(t2["warm"])))
        for g in t2["gens"]:
            if "image-" in json.dumps(g["request"]["messages"]) and \
                    "base64," not in json.dumps(g["request"]["messages"]):
                img_id = True
        check(len(t2["gens"]) == 2 and img_id and not any(
                  (c_.get("function") or {}).get("name") == "yama_describe_image"
                  for c_ in t2["kept"].get("tool_calls") or []),
              f"{tag}: the look is a hidden hop: two generations, the client sees only "
              "its write")
        c.tool_result("Successfully wrote to canvas.html")
        t3 = c.turn([reply("Done.", reasoning="Done.")], stream=True)
        _extends_fully(t2, t3, f"{tag}: the request after the image turn (the "
                               f"client echoes its reasoning)")
        x = t3["d"].get("x_yamadori") or {}
        check(((x.get("ledger") or {}).get("restored") or {}).get("hops") == 1,
              f"{tag}: the hidden hop is put back", json.dumps((x.get("ledger") or {})
                                                       .get("restored")))
    # AN ECHOING CLIENT WHOSE HOPS RAN ON ITS FIRST REQUEST (deploy check
    # 2026-09-27, h5). On a conversation's first request the proxy can only
    # GUESS whether the client echoes (_echo_guess: the account's history;
    # a fresh account guesses "strips"), and the kickoff's inserted
    # yama_plan hop always sits on that request. The hop echo was recorded
    # only when the guess said "echoes", so the reasoning the client was
    # shown (the hops' and the answer's, run together; live: the planner's
    # 9,544 characters against the slot's 67) came back as the turn's own
    # and the next request diverged inside it (live: reused 4,131 of 7,133,
    # processed 3,002). Now the echo is recorded whenever the turn has hops.
    _reset()
    e = Client("medium", role="system", key="echo1st")
    # An account never seen before: _echo_guess says "strips".
    e.account = f"{ACCOUNT}-fresh-{time.time_ns()}"
    check(not proxy._echo_guess(e.account),
          "precondition: a fresh account's first request is guessed to strip")
    t1 = e.turn([reply("", reasoning="Look first.", calls=[call(
        "yama_describe_image", {"image": "image-x", "question": "What?"},
        "d3")]),
        reply("", reasoning="Now write the page.", calls=[call(
            "write", {"content": "<canvas></canvas>", "path": "c.html"},
            "w3")])],
        user=[{"type": "text", "text": "Look at this, then write c.html."},
              {"type": "image_url", "image_url": {"url": DATA_URL}}],
        stream=True)
    check("Look first." in (t1["kept"].get("reasoning_content") or "")
          and "Now write the page." in (t1["kept"].get("reasoning_content")
                                        or ""),
          "first-request hops: the client was shown (and echoes) the hop's "
          "reasoning and the answer's, run together",
          json.dumps(t1["kept"])[:300])
    e.tool_result("Successfully wrote to c.html")
    t2 = e.turn([reply("Done.", reasoning="Done.")], stream=True)
    _extends_fully(t1, t2, "an echoing client whose hidden hop ran on its "
                           "FIRST request (the kickoff's place): the next "
                           "request extends the slot")
    x2 = t2["d"].get("x_yamadori") or {}
    check(((x2.get("ledger") or {}).get("restored") or {})
          .get("hop_reasoning") == 1,
          "and the echo is recognised: the turn is rendered with the slot's "
          "reasoning (hop_reasoning)",
          json.dumps((x2.get("ledger") or {}).get("restored")))
    # A client that STRIPS reasoning (Hermes). PAST REASONING IS RESTORED
    # (operator, 2026-09-27): the hidden hop comes back with its own
    # reasoning and the turn with the slot's; with switch restore_reasoning
    # off (the 2026-09-24 pass-through), the hop's reasoning is emptied and
    # the turn has none.
    for key, feats in (("strip1", {}),
                       ("strip2", {"restore_reasoning": False})):
        _reset()
        s = Client("medium", role="system", key=key)
        s.features = feats
        s.turn([reply("", reasoning="Look.", calls=[call(
            "yama_describe_image", {"image": "image-x", "question": "What?"},
            "d2")]),
            reply("A red square.", reasoning="Answer.")],
            user=[{"type": "text", "text": "What is this?"},
                  {"type": "image_url", "image_url": {"url": DATA_URL}}])
        for m in s.msgs:
            m.pop("reasoning_content", None)
        s.turn([reply("You are welcome.")], user="Thanks.")
        sent = _gens[-1]["request"]["messages"]
        hop = next((m for m in sent
                    if m.get("role") == "assistant" and any(
                        (c_.get("function") or {}).get("name")
                        == "yama_describe_image"
                        for c_ in m.get("tool_calls") or [])), None)
        ans = next((m for m in sent if m.get("role") == "assistant"
                    and m.get("content") == "A red square."), {})
        if not feats:
            check(hop is not None and hop.get("reasoning_content") == "Look."
                  and ans.get("reasoning_content") == "Answer.",
                  "a stripping client, reasoning restored: the hidden hop "
                  "comes back with its own reasoning, the turn with the "
                  "slot's", json.dumps([hop, ans])[:300])
        else:
            check(hop is not None and not hop.get("reasoning_content")
                  and not ans.get("reasoning_content"),
                  "a stripping client, switch off (pass-through): the hidden "
                  "hop's reasoning emptied, the turn's none",
                  json.dumps([hop, ans])[:300])


def test_a_text_parts_user_turn_gets_its_injection():
    """Pi sends every user turn as a LIST of text parts (pi-ai
    openai-completions). The per-turn injection (here a skill) is decided on
    it, added as one more text part, and replayed the same way: the next
    request extends the slot (mcp/test_harness_decisions.py
    pi/chat/skills-target, 2026-09-26)."""
    _reset()
    hint = "\n\n---\nSkill: keep module exports and callers in step."
    saved = proxy._skills_tail
    proxy._skills_tail = lambda messages, sel, route, client_tools=None, account="": (
        hint, {"on": True, "ids": ["s1"], "names": ["s1"], "versions": [1],
               "chars": len(hint), "why": "test"})
    try:
        c = Client("medium", key="pis1")
        c.tools = PI_TOOLS
        body_feats = {"skills": True, "investigate": False, "fanout": 1,
                      "retrieval": False}
        c.body_feats = body_feats
        c.msgs.append({"role": "user", "content": [
            {"type": "text", "text": "Fix the export in js/audio.js so "
                                     "game.js can import it."}]})
        _script[:] = [reply("Done.", reasoning="Fixed.")]
        n0 = len(_gens)
        b = c.body()
        b["_features"] = json.dumps(body_feats)
        d = proxy.complete(b)
        up = _gens[n0]["request"]["messages"]
        user = next(m for m in up if m.get("role") == "user")
        check(isinstance(user["content"], list)
              and user["content"][-1] == {"type": "text", "text": hint},
              "a text-parts user turn gets the skill, as one more text part",
              json.dumps(user)[:300])
        t1 = {"gens": _gens[n0:], "warm": []}
        m = d["choices"][0]["message"]
        c.msgs.append({"role": "assistant", "content": m.get("content") or "",
                       "reasoning_content": m.get("reasoning_content") or ""})
        c.msgs.append({"role": "user", "content": [
            {"type": "text", "text": "Thanks."}]})
        _script[:] = [reply("You're welcome.")]
        n1 = len(_gens)
        b = c.body()
        b["_features"] = json.dumps(body_feats)
        proxy.complete(b)
        _extends_fully(t1, {"gens": _gens[n1:]},
                       "the next request replays it the same way")
    finally:
        proxy._skills_tail = saved


def main() -> int:
    for fn in (test_the_developer_role,
               test_a_text_parts_user_turn_gets_its_injection,
               test_a_template_refusal_is_a_400,
               test_pi_compaction_is_mapped_and_thinks,
               test_an_unmapped_compaction_thinks_at_the_clients_effort,
               test_opencode_title_and_compaction,
               test_the_cache_after_a_hidden_describe_image_hop):
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
