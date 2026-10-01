#!/usr/bin/env python
"""The Anthropic Messages API (`POST /v1/messages`, `POST
/v1/messages/count_tokens`, mcp/messages_api.py). No GPU.

    python mcp/test_messages_api.py      -> "N/M checks passed"

WHAT THIS GATES (operator, 2026-09-29: "Yes add anthropic api endpoints for
claude"): Claude Code can use Yamadori as its model, through a TRANSLATION
over the one turn, never a second pipeline.

  1. TRANSLATION. Every block Claude Code sends -- system blocks with
     cache_control, a mid-conversation system entry, text / image /
     tool_result (string and blocks, is_error, images) user blocks,
     assistant text / thinking (signed, forged) / redacted_thinking /
     tool_use, tools with input_schema, server tools, tool_choice, stop
     sequences, sampling, output_config -- becomes the chat body the one turn
     takes; what cannot be served is a 400 in Anthropic's envelope.
  2. EFFORT -> TIER, deterministically, and the model that serves each tier
     under mcp/tier_models.yaml.
  3. THE SAME TURN. Through proxy.complete / proxy.stream_body and the
     SERVED chat template (test_ledger's harness): a Claude-Code-shaped
     client that replays every block it was given, thinking echoed with its
     signature, keeps the slot's prefix across a multi-turn tool loop; the
     session is Claude Code's session header; streamed == blocking.
  4. THE STREAM, read by Claude Code's own rules (code.claude.com gateway
     protocol "Streaming": no event for a block never started or already
     stopped; message_delta carries the stop_reason; message_stop last) and
     accumulated like the SDK's; pings while the turn is silent.
  5. ERRORS before the first byte (a real HTTP status, Anthropic's
     envelope) and after it (an `error` event); the window check's refusal
     in the wording Claude Code compacts on.
  6. count_tokens (the model server's own count), auth by both headers, the
     route, and the features that touch the wire: images, a hidden hop,
     stray markers, reasoning restore, cancel on close.

Every database and store is a temp path set BEFORE proxy is imported.
"""
from __future__ import annotations

import base64
import copy
import json
import os
import struct
import sys
import tempfile
import threading
import time
import traceback
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

_TMP = tempfile.mkdtemp(prefix="yamadori_test_messages_")
for _k, _v in (("YAMADORI_CORPUS_DB", "corpus.sqlite3"),
               ("YAMADORI_NEBARI_DB", "nebari.sqlite3"),
               ("RINGS_DB", "rings.sqlite3"),
               ("CODE_INDEX_DB", "code.sqlite3"),
               ("YAMADORI_SLOTS_STATE", "slots_state.json"),
               ("CONCEPT_SEED_LAST", "seed_last.json"),
               ("YAMADORI_ACCOUNTS_DIR", "accounts"),
               ("YAMADORI_MEDIA_DIR", "media"),
               ("YAMADORI_MEDIA_SECRET_FILE", "media_url.key")):
    os.environ[_k] = os.path.join(_TMP, _v)
os.environ.setdefault("YAMADORI_GPU_ROOM", "0")
os.environ.pop("YAMADORI_TIER_MODELS", None)
os.environ.pop("YAMADORI_MAX_MODEL", None)

import test_ledger as T  # noqa: E402  (its own temp paths + fake upstream)
import accounts  # noqa: E402
import api_errors  # noqa: E402
import catalog  # noqa: E402
import images  # noqa: E402
import messages_api as M  # noqa: E402
import proxy  # noqa: E402
import session_id  # noqa: E402
import tier_models  # noqa: E402
import tiers  # noqa: E402
import vision  # noqa: E402

_tmp_root = os.path.abspath(tempfile.gettempdir())
for _k in ("YAMADORI_CORPUS_DB", "YAMADORI_NEBARI_DB", "RINGS_DB",
           "CODE_INDEX_DB", "YAMADORI_ACCOUNTS_DIR", "YAMADORI_MEDIA_DIR",
           "YAMADORI_MEDIA_SECRET_FILE"):
    assert os.path.abspath(os.environ[_k]).startswith(_tmp_root), _k

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


# ------------------------------------------------------------- fixtures ----

def tiny_png(rgb=(9, 99, 7)) -> bytes:
    raw = b"".join(b"\x00" + bytes(rgb) * 2 for _ in range(2))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))
    return (images.PNG_MAGIC
            + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


PNG_B64 = base64.b64encode(tiny_png()).decode()
IMAGE = {"type": "image", "source": {"type": "base64",
                                     "media_type": "image/png",
                                     "data": PNG_B64}}


def _describe(data, fmt, question):
    return {"answer": "A small green square.", "seconds": 0.01, "usage": {},
            "finish": "stop"}


vision.describe = _describe

# The fake image generator: a REAL PNG in the (temp) media store, so the
# signed link verifies and the proxy shows it (IMAGES REACH THE CHAT).
DRAWN: list[dict] = []


def _generate(prompt, size=None, seed=None, steps=None, n=1, model=None):
    w, h = images.parse_size(size)
    DRAWN.append({"prompt": prompt, "size": f"{w}x{h}"})
    meta = {"prompt": prompt, "seed": 7, "size": f"{w}x{h}", "steps": 4,
            "model": "imagegen-turbo", "image_model": "yamadori-image-turbo",
            "seconds": 0.01, "created": int(time.time())}
    sha = images.store(tiny_png((len(DRAWN) % 250, 40, 80)), meta)
    return [dict(meta, id=sha)]


images.generate = _generate

WRITE_TOOL = {"name": "write_file", "description": "Write a file.",
              "input_schema": T.WRITE["function"]["parameters"]}
ATTRIBUTION = "x-anthropic-billing-header: cc_version=2.1.290; cc_entrypoint=cli"
BASE = "https://ai.example.test"
ACCOUNT = f"{T.ACCOUNT}-messages"


# ------------------------------------------------------------ the reader ----
# Anthropic's SSE (event: <type> / data: <json>), read as Claude Code reads
# it (gateway protocol "Streaming") and accumulated as the SDK's
# MessageStream does (content_block_start sets the block; each delta
# appends; input_json_delta's partial JSON is parsed at the block's stop).

def sse_decode(raw: bytes) -> list[dict]:
    out, event, data = [], None, []
    for line in raw.decode("utf-8").replace("\r\n", "\n").split("\n"):
        if not line:
            if event is None and not data:
                continue
            d = json.loads("\n".join(data))
            d["_event"] = event
            out.append(d)
            event, data = None, []
            continue
        field, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if field == "event":
            event = value
        elif field == "data":
            data.append(value)
    return out


def check_stream(events: list[dict], label: str) -> dict:
    """Claude Code's rules and the SDK's accumulation; returns the
    accumulated message (with `_error` when an error event ended it)."""
    problems = []
    types = [e.get("type") for e in events]
    if not events or types[0] != "message_start":
        problems.append(f"first event {types[:1]}")
    msg = copy.deepcopy((events[0].get("message") or {}) if events else {})
    open_blocks: set = set()
    stopped: set = set()
    partial: dict = {}
    ended = False
    for i, e in enumerate(events):
        t = e.get("type")
        if e.get("_event") != t:
            problems.append(f"#{i} event: {e.get('_event')} != type {t}")
        if ended:
            problems.append(f"#{i} {t} after the end")
        if t == "content_block_start":
            idx = e["index"]
            if idx != len(msg.get("content") or []):
                problems.append(f"#{i} block {idx} out of order")
            msg.setdefault("content", []).append(copy.deepcopy(
                e["content_block"]))
            open_blocks.add(idx)
        elif t == "content_block_delta":
            idx = e["index"]
            if idx not in open_blocks:
                problems.append(f"#{i} delta for block {idx} not open "
                                f"(stopped: {idx in stopped})")
                continue
            b, d = msg["content"][idx], e["delta"]
            if d["type"] == "text_delta":
                b["text"] += d["text"]
            elif d["type"] == "thinking_delta":
                b["thinking"] += d["thinking"]
            elif d["type"] == "signature_delta":
                b["signature"] = d["signature"]
            elif d["type"] == "input_json_delta":
                partial[idx] = partial.get(idx, "") + d["partial_json"]
            else:
                problems.append(f"#{i} unknown delta {d['type']}")
        elif t == "content_block_stop":
            idx = e["index"]
            if idx not in open_blocks:
                problems.append(f"#{i} stop for block {idx} not open")
            open_blocks.discard(idx)
            stopped.add(idx)
            if idx in partial:
                msg["content"][idx]["input"] = json.loads(partial[idx])
        elif t == "message_delta":
            if open_blocks:
                problems.append(f"#{i} message_delta with blocks open "
                                f"{sorted(open_blocks)}")
            msg.update(e.get("delta") or {})
            msg["usage"] = dict(msg.get("usage") or {}, **(e.get("usage")
                                                           or {}))
            msg["x_yamadori"] = e.get("x_yamadori")
        elif t == "message_start":
            if i:
                problems.append(f"#{i} a second message_start")
        elif t == "message_stop":
            ended = True
        elif t == "error":
            msg["_error"] = e.get("error")
            ended = True
        elif t != "ping":
            problems.append(f"#{i} unknown event {t}")
    if not ended:
        problems.append("the stream never ended (message_stop or error)")
    if "_error" not in msg and not msg.get("stop_reason"):
        problems.append("no stop_reason")
    check(not problems, f"{label}: the stream reads cleanly (Claude Code's "
          f"block rules, the SDK's accumulation; {len(events)} events)",
          "; ".join(problems[:6]))
    return msg


# ------------------------------------------------------------ the client ----

class CClient:
    """A Claude-Code-shaped client: system as blocks (the attribution block
    first, cache_control on the last), tools with input_schema, adaptive
    thinking and output_config.effort, metadata.user_id, the session and
    version headers; it REPLAYS every content block it was given (thinking
    with its signature) and answers tool_use with tool_result blocks."""

    def __init__(self, tag: str, *, stream: bool = False,
                 effort: str = "medium", session: str | None = "auto",
                 echo_thinking: bool = True, tools: list | None = None):
        self.tag = tag
        self.stream = stream
        self.effort = effort
        self.account = f"{ACCOUNT}-{tag}"
        self.session = (f"cc-{tag}-{os.getpid()}" if session == "auto"
                        else session)
        self.system = [{"type": "text", "text": ATTRIBUTION},
                       {"type": "text", "text": T.SYSTEM + f" [{tag}]",
                        "cache_control": {"type": "ephemeral"}}]
        self.messages: list[dict] = []
        self.tools = tools if tools is not None else [WRITE_TOOL]
        self.echo_thinking = echo_thinking
        self.ids: dict[str, str] = {}

    def headers(self) -> dict:
        h = {"anthropic-version": "2023-06-01",
             "anthropic-beta": "claude-code-20250219,interleaved-thinking-2025-05-14"}
        if self.session:
            h["x-claude-code-session-id"] = self.session
        return h

    def body(self) -> dict:
        b = {"model": "yamadori", "max_tokens": 32000,
             "system": copy.deepcopy(self.system),
             "messages": copy.deepcopy(self.messages),
             "tools": copy.deepcopy(self.tools),
             "thinking": {"type": "adaptive"},
             "output_config": {"effort": self.effort},
             "metadata": {"user_id": json.dumps({"device_id": "d1"})},
             "stream": self.stream}
        return b

    def chat_of(self, body: dict, headers: dict):
        chat, ctx = M.to_chat(body, headers, self.account)
        chat.update(_account=self.account, _client_ip="127.0.0.1",
                    _public_base=BASE, _session_token="")
        chat["_features"] = M.merge_features(chat.get("_features"), json.dumps(
            {"skills": True}))
        return chat, ctx

    def turn(self, script: list[dict], user=None) -> dict:
        if user is not None:
            self.messages.append({"role": "user", "content": (
                [{"type": "text", "text": user}] if isinstance(user, str)
                else user)})
        T._script[:] = list(script)
        n0, w0 = len(T._gens), len(T._warms)
        chat, ctx = self.chat_of(self.body(), self.headers())
        events = None
        if self.stream:
            raw = b"".join(M.stream(proxy.stream_body(chat, ctx.model), ctx,
                                    ping_s=30))
            events = sse_decode(raw)
            msg = check_stream(events, f"{self.tag} turn")
        else:
            d = catalog.rewrite_response(proxy.complete(chat), ctx.model)
            msg = M.of_chat(d, ctx)
        gen = T._gens[-1]["reply"] if len(T._gens) > n0 else {"calls": []}
        mine = [c for c in gen["calls"]
                if c["function"]["name"] not in proxy.OUR_NAMES]
        uses = [b for b in msg.get("content") or [] if b.get("type") ==
                "tool_use"]
        for a, b in zip(mine, uses):
            self.ids[a["id"]] = b["id"]
        kept = [copy.deepcopy(b) for b in msg.get("content") or []
                if self.echo_thinking or b.get("type") != "thinking"]
        if kept:
            self.messages.append({"role": "assistant", "content": kept})
        x = msg.get("x_yamadori") or {}
        if (x.get("warm") or {}).get("sent"):
            end = time.time() + 5
            while time.time() < end and len(T._warms) == w0:
                time.sleep(0.02)
        return {"msg": msg, "events": events, "chat": chat, "x": x,
                "gens": T._gens[n0:], "warm": T._warms[w0:], "ctx": ctx}

    def tool_result(self, cid: str, content, is_error: bool = False,
                    text: str | None = None) -> None:
        blocks = [{"type": "tool_result", "tool_use_id": self.ids.get(cid, cid),
                   "content": content}]
        if is_error:
            blocks[0]["is_error"] = True
        if text:
            blocks.append({"type": "text", "text": text})
        self.messages.append({"role": "user", "content": blocks})


def _blocks(msg: dict, kind: str) -> list[dict]:
    return [b for b in msg.get("content") or [] if b.get("type") == kind]


def _text(msg: dict) -> str:
    return "".join(b.get("text", "") for b in _blocks(msg, "text"))


def _raises(fn, *a, **k):
    try:
        fn(*a, **k)
    except api_errors.ApiError as e:
        return e
    return None


# ----------------------------------------------------------------- tests ----

def test_request_translation():
    acct = f"{ACCOUNT}-translation"
    good_sig = M.sign_thinking("I will write it.", acct)
    body = {
        "model": "claude-opus-5-5", "max_tokens": 64000,
        "system": [{"type": "text", "text": ATTRIBUTION},
                   {"type": "text", "text": "You are Claude Code.",
                    "cache_control": {"type": "ephemeral"}}],
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "<system-reminder>ctx</system-reminder>\n"},
                {"type": "text", "text": "Fix a.py.",
                 "cache_control": {"type": "ephemeral"}}]},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "I will write it.",
                 "signature": good_sig},
                {"type": "redacted_thinking", "data": "opaque"},
                {"type": "text", "text": "Writing."},
                {"type": "tool_use", "id": "toolu_A", "name": "write_file",
                 "input": {"path": "a.py", "content": "x = 1"}}]},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_B", "name": "read_file",
                 "input": {"path": "b.py"}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_A",
                 "content": "wrote a.py"},
                {"type": "tool_result", "tool_use_id": "toolu_B",
                 "is_error": True,
                 "content": [{"type": "text", "text": "<tool_use_error>no "},
                             {"type": "text", "text": "such file</tool_use_error>"}]},
                {"type": "text", "text": "Also look at this."},
                IMAGE]},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "Forged reasoning.",
                 "signature": "EqQBCkgIBRABGAIiQL"},
                {"type": "server_tool_use", "id": "srv_1", "name": "web_search",
                 "input": {"query": "x"}},
                {"type": "tool_use", "id": "toolu_C", "name": "view",
                 "input": {}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_C", "content": [
                    {"type": "text", "text": "the screenshot"}, IMAGE,
                    {"type": "document", "source": {"type": "text",
                                                    "data": "x"}}]}]},
            {"role": "system", "content": "A mid-conversation reminder."},
            {"role": "user", "content": "Continue."},
        ],
        "tools": [WRITE_TOOL,
                  {"name": "read_file", "description": "Read.",
                   "input_schema": {"type": "object"}, "strict": True,
                   "cache_control": {"type": "ephemeral"}},
                  {"name": "view", "input_schema": {"type": "object"},
                   "defer_loading": True},
                  {"type": "web_search_20250305", "name": "web_search",
                   "max_uses": 5},
                  {"type": "advisor_20260301", "name": "advisor"}],
        "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "high"},
        "stop_sequences": ["</done>"], "temperature": 0.5, "top_k": 10,
        "metadata": {"user_id": "user_abc_account_def_session_"
                                "0f1e2d3c-4b5a-6978-8796-a5b4c3d2e1f0"},
        "context_management": {"edits": [{"type": "clear_tool_uses_20250919"}]},
        "stream": True,
    }
    headers = {"anthropic-version": "2023-06-01",
               "anthropic-beta": "claude-code-20250219, context-management-2025-06-27",
               "x-claude-code-session-id": "sess-1",
               "x-claude-code-agent-id": "agent-7",
               "x-claude-code-request-class": "subagent",
               "x-claude-code-agent-type": "Explore"}
    chat, ctx = M.to_chat(copy.deepcopy(body), headers, acct)
    m = chat["messages"]
    roles = [x["role"] for x in m]
    check(roles == ["system", "user", "assistant", "tool", "tool", "user",
                    "assistant", "tool", "user", "user"],
          "blocks -> chat roles: ONE system first, consecutive assistant "
          "messages merge into one turn, tool_result blocks become tool "
          "messages before the user's own text, a mid-conversation system "
          "entry becomes a user turn", json.dumps(roles))
    check(m[0]["content"] == ATTRIBUTION + "\n\nYou are Claude Code.",
          "system blocks join with a blank line (cache_control dropped)",
          json.dumps(m[0]))
    check(m[1]["content"] == "<system-reminder>ctx</system-reminder>\nFix a.py.",
          "an all-text user message is ONE string (\"\".join, as the "
          "template renders a text-part list)", json.dumps(m[1]))
    a = m[2]
    check(a.get("reasoning_content") == "I will write it."
          and a["content"] == "Writing."
          and [c["id"] for c in a["tool_calls"]] == ["toolu_A", "toolu_B"]
          and json.loads(a["tool_calls"][0]["function"]["arguments"]) ==
          {"path": "a.py", "content": "x = 1"},
          "a signed thinking block (ours) is the turn's reasoning; text and "
          "tool_use (ids KEPT) of two consecutive assistant messages merge",
          json.dumps(a)[:400])
    check(m[3] == {"role": "tool", "tool_call_id": "toolu_A",
                   "content": "wrote a.py"}
          and m[4] == {"role": "tool", "tool_call_id": "toolu_B",
                       "content": "<tool_use_error>no such file</tool_use_error>"},
          "tool_result: string content, and text blocks joined", json.dumps(
              m[3:5]))
    u5 = m[5]["content"]
    check(isinstance(u5, list) and [p["type"] for p in u5] == [
        "text", "image_url"] and u5[1]["image_url"]["url"].startswith(
            "data:image/png;base64,"),
          "an image block goes to the chat image path as a data: URL part",
          json.dumps(u5)[:200])
    check(m[6].get("reasoning_content") is None
          and [c["id"] for c in m[6]["tool_calls"]] == ["toolu_C"],
          "a thinking block whose signature is not ours is DROPPED (the "
          "ledger puts back the slot's own); server_tool_use is ignored",
          json.dumps(m[6]))
    t7 = m[7]["content"]
    check(isinstance(t7, list) and [p["type"] for p in t7] == [
        "text", "image_url", "text"] and "document" in t7[2]["text"],
          "a tool_result's image is an image_url part; a document there "
          "becomes a note (refusing would wedge the conversation)",
          json.dumps(t7)[:300])
    check(m[8]["content"] == "A mid-conversation reminder."
          and m[9]["content"] == "Continue.",
          "the later system entry reaches the model as a user turn")
    tools = chat["tools"]
    check([t["function"]["name"] for t in tools] == ["write_file",
                                                     "read_file", "view"]
          and tools[1]["function"] == {"name": "read_file",
                                       "description": "Read.",
                                       "parameters": {"type": "object"},
                                       "strict": True},
          "tools: name/description/input_schema -> function tools; strict "
          "kept; server tools (web_search, advisor) not offered",
          json.dumps(tools)[:300])
    rec = ctx.rec
    check(rec.get("tools_ignored") == ["advisor_20260301",
                                       "web_search_20250305"]
          and rec.get("deferred_tools_offered") == 1
          and rec.get("cache_control") == 3
          and rec.get("tool_errors") == 1
          and rec.get("thinking_in") == {"verified": 1,
                                         "redacted_ignored": 1,
                                         "unverified_dropped": 1}
          and rec.get("blocks_ignored") == ["server_tool_use"]
          and rec.get("fields_ignored") == ["context_management"]
          and rec.get("anthropic_version") == "2023-06-01"
          and rec.get("anthropic_beta") == ["claude-code-20250219",
                                            "context-management-2025-06-27"]
          and rec.get("system_as_user") == 1,
          "x_yamadori.messages records what was not given to the model, and "
          "the version / beta headers", json.dumps(rec)[:600])
    check(chat.get("tool_choice") == "auto"
          and chat.get("parallel_tool_calls") is False
          and chat.get("reasoning_effort") == "high"
          and chat.get("max_tokens") == 64000
          and chat.get("stop") == ["</done>"]
          and chat.get("temperature") == 0.5 and chat.get("top_k") == 10
          and chat.get("stream") is True
          and chat.get("stream_options") == {"include_usage": True}
          and chat.get("prompt_cache_key") == "claude-code:sess-1:agent-7"
          and rec.get("session_source") ==
          "x-claude-code-session-id + x-claude-code-agent-id"
          and rec.get("hints") == {"request-class": "subagent",
                                   "agent-type": "Explore",
                                   "agent-id": "present"}
          and "_features" not in chat,
          "tool_choice, disable_parallel_tool_use, effort, max_tokens, "
          "stop_sequences, sampling, stream; the session is Claude Code's "
          "session header + the subagent's id; a subagent is not a side call",
          json.dumps({k: v for k, v in chat.items() if k not in (
              "messages", "tools")}))
    again, _ = M.to_chat(copy.deepcopy(body), headers, acct)
    check(again["messages"] == chat["messages"],
          "the translation is deterministic (the ledger's chain keys)")
    other, octx = M.to_chat(copy.deepcopy(body), {}, acct)
    check(other.get("prompt_cache_key") ==
          "claude-code:0f1e2d3c-4b5a-6978-8796-a5b4c3d2e1f0"
          and octx.rec.get("session_source") == "metadata.user_id",
          "without the header, metadata.user_id's session (legacy form) "
          "names the conversation", json.dumps(other.get("prompt_cache_key")))
    mine, mctx = M.to_chat(copy.deepcopy(body), {"x-yamadori-session": "s1",
                                                 "x-claude-code-session-id":
                                                 "sess-1"}, acct)
    check("prompt_cache_key" not in mine
          and mctx.rec.get("session_source", "").startswith("x-yamadori"),
          "our own session header wins over Claude Code's")
    nosess, nctx = M.to_chat({"model": "x", "max_tokens": 5, "messages": [
        {"role": "user", "content": "hi"}], "metadata": {"user_id": "u-1"}},
        {}, acct)
    check("prompt_cache_key" not in nosess
          and nctx.rec["session_source"].startswith("none"),
          "a bare user id names a person, not a conversation: ignored (our "
          "id rides in the tool_use ids)")
    aux, actx = M.to_chat({"model": "x", "max_tokens": 50, "messages": [
        {"role": "user", "content": "Title for: fix a.py"}]},
        {"x-claude-code-request-class": "auxiliary"}, acct)
    aux_t, _ = M.to_chat({"model": "x", "max_tokens": 50, "tools": [WRITE_TOOL],
                          "messages": [{"role": "user", "content": "x"}]},
                         {"x-claude-code-request-class": "auxiliary"}, acct)
    check(json.loads(aux.get("_features") or "{}") == {"utility": True}
          and actx.rec.get("utility_from_hint") is True
          and "_features" not in aux_t
          and json.loads(M.merge_features(aux["_features"],
                                          '{"utility": false}')) ==
          {"utility": False},
          "Claude Code's auxiliary side request (no tools) is a utility call; "
          "with tools it is left to the rule; the caller's header wins",
          json.dumps([aux.get("_features"), aux_t.get("_features")]))
    ch = M.to_chat({"model": "x", "max_tokens": 5, "tools": [
        WRITE_TOOL, {"name": "read_file", "input_schema": {"type": "object"}}],
        "tool_choice": {"type": "tool", "name": "read_file"},
        "messages": [{"role": "user", "content": "x"}]}, {}, acct)[0]
    anyc = M.to_chat({"model": "x", "max_tokens": 5, "tools": [WRITE_TOOL],
                      "tool_choice": {"type": "any"},
                      "messages": [{"role": "user", "content": "x"}]},
                     {}, acct)[0]
    nonec = M.to_chat({"model": "x", "max_tokens": 5, "tools": [WRITE_TOOL],
                       "tool_choice": {"type": "none"},
                       "messages": [{"role": "user", "content": "x"}]},
                      {}, acct)[0]
    check(ch["tool_choice"] == "required"
          and [t["function"]["name"] for t in ch["tools"]] == ["read_file"]
          and anyc["tool_choice"] == "required"
          and nonec["tool_choice"] == "none",
          "tool_choice in what llama-server serves: any -> required, none -> "
          "none, a named tool -> the tools narrowed to it + required",
          json.dumps([ch.get("tool_choice"), anyc.get("tool_choice")]))
    fmt = M.to_chat({"model": "x", "max_tokens": 5, "output_config": {
        "format": {"type": "json_schema", "schema": {"type": "object"}}},
        "messages": [{"role": "user", "content": "x"}]}, {}, acct)[0]
    check(fmt.get("response_format") == {"type": "json_schema", "json_schema": {
        "name": "output", "schema": {"type": "object"}}},
          "output_config.format -> response_format", json.dumps(fmt.get(
              "response_format")))


def test_request_errors():
    ok = {"model": "x", "max_tokens": 10,
          "messages": [{"role": "user", "content": "hi"}]}
    cases = [
        ({"messages": ok["messages"], "model": "x"}, "max_tokens",
         "missing_required_parameter"),
        (dict(ok, max_tokens=0), "max_tokens", "invalid_value"),
        ({"model": "x", "max_tokens": 1}, "messages",
         "missing_required_parameter"),
        (dict(ok, messages=[]), "messages", "invalid_type"),
        (dict(ok, messages=[{"role": "tool", "content": "x"}]),
         "messages.0.role", "invalid_value"),
        (dict(ok, messages=[{"role": "user", "content": [
            {"type": "document", "source": {"type": "text", "data": "x"}}]}]),
         "messages.0.content.0.type", "invalid_content_part"),
        (dict(ok, messages=[{"role": "user", "content": [
            {"type": "tool_result", "content": "x"}]}]),
         "messages.0.content.0.tool_use_id", "missing_required_parameter"),
        (dict(ok, messages=[{"role": "assistant", "content": [
            {"type": "tool_use", "name": "x", "input": {}}]},
            {"role": "user", "content": "x"}]),
         "messages.0.content.0.id", "missing_required_parameter"),
        (dict(ok, tools=[{"name": "x"}]), "tools.0.input_schema",
         "missing_required_parameter"),
        (dict(ok, tools=[{"type": "bash_20250124", "name": "bash"}]),
         "tools.0.type", "invalid_value"),
        (dict(ok, tools=[{"type": "frobnicate", "name": "f"}]),
         "tools.0.type", "invalid_value"),
        (dict(ok, tools=[WRITE_TOOL], tool_choice={"type": "tool",
                                                   "name": "nope"}),
         "tool_choice.name", "invalid_value"),
        (dict(ok, tool_choice={"type": "sometimes"}), "tool_choice.type",
         "invalid_value"),
        (dict(ok, output_config={"effort": "ultra"}), "output_config.effort",
         "invalid_value"),
        (dict(ok, thinking={"type": "maybe"}), "thinking.type",
         "invalid_value"),
        (dict(ok, output_config={"format": {"type": "grammar"}}),
         "output_config.format", "invalid_value"),
        (dict(ok, stop_sequences="stop"), "stop_sequences", "invalid_type"),
        (dict(ok, system=[{"type": "image"}]), "system.0", "invalid_value"),
        (dict(ok, messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "s3"}}]}]),
         "messages.0.content.0.source.type", "invalid_value"),
    ]
    bad = []
    for body, param, code in cases:
        e = _raises(M.to_chat, body, {}, "a")
        st, env, _h = M.anthropic_error(e) if e else (None, {}, {})
        if not (e is not None and e.status == 400 and e.param == param
                and e.code == code and env.get("type") == "error"
                and env["error"]["type"] == "invalid_request_error"):
            bad.append((json.dumps(body)[:80], param, code,
                        e and (e.status, e.param, e.code)))
    check(not bad, f"{len(cases)} requests that cannot be served are 400 "
          f"invalid_request_error in Anthropic's envelope, naming the field",
          json.dumps(bad)[:600])
    fine = [dict(ok, service_tier="auto", container="c", speed="fast"),
            dict(ok, thinking={"type": "enabled", "budget_tokens": 31999}),
            dict(ok, thinking={"type": "disabled"}),
            dict(ok, tools=[{"type": "tool_search_tool_regex_20251119",
                             "name": "tool_search"}]),
            dict(ok, messages=[{"role": "user", "content": "x"},
                               {"role": "assistant", "content": "Prefix"}])]
    fails = [b for b in fine if _raises(M.to_chat, b, {}, "a") is not None]
    check(not fails, "service_tier, container, speed, fixed-budget and "
          "disabled thinking, a server tool, a trailing assistant (prefill) "
          "are accepted", json.dumps(fails)[:300])


def test_effort_to_tier():
    """The mapping table, deterministic, and the model each tier is served
    by under mcp/tier_models.yaml (the deployed table)."""
    base = {"model": "yamadori", "max_tokens": 10,
            "messages": [{"role": "user", "content": "x"}]}
    tier_models.reload(env={"YAMADORI_TIER_MODELS": "mcp/tier_models.yaml"})
    try:
        rows = []
        for label, extra, want_sent, want_tier, want_model in [
            ("effort low", {"output_config": {"effort": "low"}}, "low", "low", "bonsai"),
            ("effort medium", {"output_config": {"effort": "medium"}}, "medium", "medium", "bonsai"),
            ("effort high", {"output_config": {"effort": "high"}}, "high", "high", "bonsai"),
            ("effort xhigh", {"output_config": {"effort": "xhigh"}}, "xhigh", "xhigh", "mirai-s"),
            ("effort max", {"output_config": {"effort": "max"}}, "max", "max", "flash-next"),
            ("thinking disabled", {"thinking": {"type": "disabled"}}, "minimal", "minimal", "bonsai"),
            ("thinking between_tools", {"thinking": {"type": "between_tools"}}, "minimal", "minimal", "bonsai"),
            ("adaptive, no effort", {"thinking": {"type": "adaptive"}}, None, tiers.normalise(tiers.DEFAULT), "bonsai"),
            ("enabled 31999, no effort", {"thinking": {"type": "enabled", "budget_tokens": 31999}}, None,
             tiers.normalise(tiers.DEFAULT), "bonsai"),
            ("no effort, model yamadori-max", {"model": "yamadori-max"}, None, "max", "flash-next"),
            ("effort low, model yamadori-max", {"model": "yamadori-max", "output_config": {"effort": "low"}},
             "low", "low", "bonsai"),
            ("effort max + thinking disabled", {"output_config": {"effort": "max"}, "thinking": {"type": "disabled"}},
             "max", "max", "flash-next"),
            ("claude model name, effort xhigh", {"model": "claude-opus-5-5", "output_config": {"effort": "xhigh"}},
             "xhigh", "xhigh", "mirai-s"),
        ]:
            chat, ctx = M.to_chat(dict(base, **extra), {}, "a")
            e = ctx.rec.get("effort") or {}
            ok = (chat.get("reasoning_effort") == want_sent and e.get("tier") == want_tier
                  and e.get("serves") == want_model and e.get("rule"))
            rows.append((label, ok, chat.get("reasoning_effort"), e.get("tier"), e.get("serves")))
        check(all(r[1] for r in rows),
              "effort -> tier -> model: output_config.effort names the tier; thinking off is minimal; otherwise "
              "the default (or a yamadori-* name's tier); the tier table picks bonsai / mirai-s / flash-next; "
              "recorded with its rule", json.dumps([r for r in rows if not r[1]] or rows)[:700])
        chat, ctx = M.to_chat(dict(base, output_config={"effort": "high"}), {
            "x-yamadori-features": json.dumps({"utility": True})}, "a")
        check(ctx.rec["effort"]["tier"] == "high",
              "the recorded tier reads the caller's X-Yamadori-Features too")
    finally:
        tier_models.reload(env={})


def test_blocking_tool_loop_extends_the_slot():
    """A Claude-Code-shaped tool loop, blocking, thinking echoed with its
    signature: every request extends what the slot holds."""
    T.slots.reset(n=4)
    T.compaction.reset()
    c = CClient("blocking")
    t1 = c.turn([T.reply("", reasoning="Plan: write it.", calls=[T.call(
        "write_file", {"path": "a.py", "content": "x = 1"}, "mb-1")])],
        user="Write a.py with x = 1.")
    msg = t1["msg"]
    kinds = [b["type"] for b in msg["content"]]
    th = _blocks(msg, "thinking")
    tu = _blocks(msg, "tool_use")
    s1 = t1["x"].get("session") or {}
    check(kinds == ["thinking", "tool_use"]
          and th[0]["thinking"].endswith("Plan: write it.")
          and M.verify_thinking(th[0]["thinking"], th[0]["signature"],
                                c.account)
          and not M.verify_thinking(th[0]["thinking"], th[0]["signature"],
                                    "another-account")
          and tu[0]["name"] == "write_file"
          and tu[0]["input"] == {"path": "a.py", "content": "x = 1"}
          and session_id.of_call_id(tu[0]["id"]) == s1.get("carrier")
          and _text(msg) == ""
          and msg["stop_reason"] == "tool_use"
          and msg["id"].startswith("msg_") and msg["type"] == "message"
          and msg["role"] == "assistant" and msg["model"] == "yamadori",
          "a Message: thinking (signed, verifiable only for this account), "
          "tool_use (input an object, the id carrying the conversation's "
          "carrier), no text block (the check note went 2026-09-29); "
          "stop_reason tool_use",
          json.dumps(msg["content"])[:500])
    check(s1.get("source") == "prompt_cache_key"
          and isinstance(t1["x"].get("tool_turns"), dict)
          and (t1["x"].get("messages") or {}).get("effort", {}).get("tier")
          == "medium",
          "the same turn ran: the session from Claude Code's session header "
          "(prompt_cache_key), the tool-turn record, x_yamadori.messages",
          json.dumps({k: t1["x"].get(k) for k in ("session", "tool_turns")}))
    u = msg["usage"]
    check(u == {"input_tokens": 10, "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 90, "output_tokens": 10},
          "usage: input_tokens is the prompt LESS the reused cache (100 - "
          "90), cache_read_input_tokens the reused, output_tokens the "
          "generated (the final generation's)", json.dumps(u))
    c.tool_result("mb-1", "wrote a.py")
    t2 = c.turn([T.reply("", reasoning="Now b.", calls=[T.call(
        "write_file", {"path": "b.py", "content": "y = 2"}, "mb-2")])])
    T._extends(t1, t2, "messages, blocking, turn 2 (a replayed thinking + "
               "tool_use and its tool_result)")
    up2 = t2["gens"][0]["request"]["messages"]
    check([m.get("reasoning_content") for m in up2 if m["role"] ==
           "assistant"] == [th[0]["thinking"]]
          and (t2["x"].get("messages") or {}).get("thinking_in") ==
          {"verified": 1},
          "the echoed, signed thinking reaches the model as the turn's "
          "reasoning (a verified echo is kept as sent)",
          json.dumps([m.get("reasoning_content") for m in up2]))
    c.tool_result("mb-2", "wrote b.py", is_error=False,
                  text="<system-reminder>keep going</system-reminder>")
    t3 = c.turn([T.reply("Both files are written.", reasoning="Done.")])
    T._extends(t2, t3, "messages, blocking, turn 3 (a tool_result + a text "
               "block in one user message)")
    check(_text(t3["msg"]) == "Both files are written."
          and t3["msg"]["stop_reason"] == "end_turn"
          and (t3["x"].get("session") or {}).get("id") == s1.get("id"),
          "the final answer: a text block, end_turn; one session throughout",
          json.dumps(t3["msg"]["content"])[:300])


def test_a_client_that_drops_thinking_gets_it_restored():
    """Claude Code removes earlier thinking blocks after a signature
    rejection (gateway protocol "Automatic retry"): the ledger then puts
    the slot's own reasoning back, so the request still extends the slot. A
    FORGED block is dropped the same way."""
    T.slots.reset(n=4)
    T.compaction.reset()
    c = CClient("strip", echo_thinking=False)
    t1 = c.turn([T.reply("", reasoning="Plan: s.", calls=[T.call(
        "write_file", {"path": "s.py", "content": "s = 1"}, "ms-1")])],
        user="Write s.py.")
    c.tool_result("ms-1", "wrote s.py")
    t2 = c.turn([T.reply("", reasoning="Plan: t.", calls=[T.call(
        "write_file", {"path": "t.py", "content": "t = 1"}, "ms-2")])])
    T._extends(t1, t2, "messages, thinking dropped by the client: restored "
               "from the ledger")
    check(((t2["x"].get("ledger") or {}).get("restored") or {}).get(
        "reasoning") == 1,
          "x_yamadori.ledger.restored counts the restored reasoning",
          json.dumps((t2["x"].get("ledger") or {}).get("restored")))
    # a forged block on the last assistant turn: dropped, then restored
    c.messages[-1]["content"].insert(0, {"type": "thinking",
                                         "thinking": "I am someone else.",
                                         "signature": "ymt1.forged"})
    c.tool_result("ms-2", "wrote t.py")
    t3 = c.turn([T.reply("Done.")])
    T._extends(t2, t3, "messages, a forged thinking block: dropped, the "
               "slot's own restored")
    up = t3["gens"][0]["request"]["messages"]
    check("I am someone else." not in json.dumps(up)
          and (t3["x"].get("messages") or {}).get("thinking_in", {}).get(
              "unverified_dropped") == 1,
          "the forged reasoning never reaches the model", json.dumps(
              (t3["x"].get("messages") or {}).get("thinking_in")))


def test_streamed_equals_blocking():
    got = {}
    for stream in (False, True):
        T.slots.reset(n=4)
        T.compaction.reset()
        tag = "sb-" + ("s" if stream else "b")
        c = CClient(tag, stream=stream)
        t1 = c.turn([T.reply("", reasoning="I will write the file first.",
                             calls=[T.call("write_file", {
                                 "path": "q.py", "content": "q = 1"},
                                 f"{tag}-1")])], user="Write q.py.")
        c.tool_result(f"{tag}-1", "wrote q.py")
        t2 = c.turn([T.reply("Done: q.py holds q = 1. " * 3,
                             reasoning="It is written.")])
        T._extends(t1, t2, f"messages {tag}: turn 2")

        def shape(msg):
            return [(b["type"], b.get("text") or b.get("thinking")
                     or b.get("name"), b.get("input"))
                    for b in msg.get("content") or []]
        got[stream] = ([shape(t1["msg"]), shape(t2["msg"])],
                       [t1["msg"].get("stop_reason"),
                        t2["msg"].get("stop_reason")],
                       [t1["msg"].get("usage"), t2["msg"].get("usage")])
        if stream:
            ev = t1["events"]
            types = [e["type"] for e in ev]
            check(types[:3] == ["message_start", "content_block_start",
                                "content_block_delta"]
                  and ev[1]["content_block"] == {"type": "thinking",
                                                 "thinking": "",
                                                 "signature": ""}
                  and any(e.get("delta", {}).get("type") == "signature_delta"
                          for e in ev)
                  and types[-2:] == ["message_delta", "message_stop"]
                  and ev[-2]["delta"] == {"stop_reason": "tool_use",
                                          "stop_sequence": None}
                  and isinstance(ev[-2].get("x_yamadori"), dict),
                  "event order: message_start, the thinking block (deltas, "
                  "signature_delta, stop), the text, the tool_use block "
                  "(input_json_delta), message_delta (stop_reason, usage, "
                  "x_yamadori), message_stop", json.dumps(types))
            th = _blocks(t1["msg"], "thinking")[0]
            check(M.verify_thinking(th["thinking"], th["signature"],
                                    c.account),
                  "the streamed thinking block's signature verifies")
            deltas = [e["delta"]["text"] for e in t2["events"]
                      if e["type"] == "content_block_delta"
                      and e["delta"]["type"] == "text_delta"]
            check(len(deltas) >= 2 and "".join(deltas) == _text(t2["msg"]),
                  "text streams as text_delta events that add up to the "
                  "block", json.dumps(deltas)[:200])
    check(got[True] == got[False],
          "streamed == blocking: the same blocks, stop reasons and usage",
          json.dumps(got)[:700])


def _chunk(delta: dict, finish=None, **extra) -> bytes:
    return b"data: " + json.dumps(dict({"choices": [{
        "index": 0, "delta": delta, "finish_reason": finish}]}, **extra)
    ).encode() + b"\n\n"


USAGE = (b'data: {"choices": [], "usage": {"prompt_tokens": 9, '
         b'"completion_tokens": 3, "total_tokens": 12, '
         b'"prompt_tokens_details": {"cached_tokens": 4}}}\n\n')


def _ctx(body: dict | None = None, account: str = "a"):
    return M.to_chat(dict({"model": "yamadori", "max_tokens": 10,
                           "stream": True, "messages": [
                               {"role": "user", "content": "x"}]},
                          **(body or {})), {}, account)[1]


def test_stream_forms_and_keepalive():
    ctx = _ctx()
    s = M.Stream(ctx)
    ev = sse_decode(s.feed(_chunk({})))
    check([e["type"] for e in ev] == ["message_start"]
          and ev[0]["message"]["usage"]["input_tokens"] == 0,
          "a first heartbeat opens the stream with message_start",
          json.dumps(ev)[:200])
    s.last_event = 0.0
    ev = sse_decode(s.feed(_chunk({})))
    check([e["type"] for e in ev] == ["ping"],
          "a heartbeat is a `ping` event (Claude Code counts every byte; "
          "300 s of silence aborts the stream)", json.dumps(ev))
    s.last_event = time.time()
    check(sse_decode(s.feed(_chunk({}))) == [],
          f"heartbeats are throttled to one per {M.KEEPALIVE_S:g} s")
    s.feed(_chunk({"content": "Answer."}))
    s.last_event = 0.0
    ev = sse_decode(s.feed(_chunk({"reasoning_content": "late"})))
    check([e["type"] for e in ev] == ["ping"],
          "reasoning after the answer started is a ping (CHANNEL ORDER), "
          "never a thinking block after text", json.dumps(ev))
    # thinking disabled: no thinking block at all
    ctx = _ctx({"thinking": {"type": "disabled"}})
    ev = sse_decode(b"".join(M.stream(iter([
        _chunk({"reasoning_content": "hidden"}), _chunk({"content": "Hi."}),
        _chunk({}, "stop", x_yamadori={}), USAGE, b"data: [DONE]\n\n"]), ctx)))
    msg = check_stream(ev, "thinking disabled")
    check([b["type"] for b in msg["content"]] == ["text"]
          and msg["usage"] == {"input_tokens": 5,
                               "cache_creation_input_tokens": 0,
                               "cache_read_input_tokens": 4,
                               "output_tokens": 3},
          "with thinking disabled no thinking block goes out; usage from "
          "the chat usage chunk", json.dumps(msg)[:300])
    # display omitted: an empty, signed thinking block
    ctx = _ctx({"thinking": {"type": "adaptive", "display": "omitted"}})
    ev = sse_decode(b"".join(M.stream(iter([
        _chunk({"reasoning_content": "secret"}), _chunk({"content": "Hi."}),
        _chunk({}, "stop", x_yamadori={}), USAGE, b"data: [DONE]\n\n"]), ctx)))
    msg = check_stream(ev, "display omitted")
    th = msg["content"][0]
    check(th["type"] == "thinking" and th["thinking"] == ""
          and M.verify_thinking("", th["signature"], "a")
          and "secret" not in json.dumps(ev),
          "display omitted: the thinking block is empty and signed; the text "
          "never goes out", json.dumps(th))
    # a length finish, a tool call, a malformed-argument call
    ctx = _ctx({"tools": [WRITE_TOOL]})
    ev = sse_decode(b"".join(M.stream(iter([
        _chunk({"content": "cut"}), _chunk({}, "length"), USAGE,
        b"data: [DONE]\n\n"]), ctx)))
    msg = check_stream(ev, "a length finish")
    check(msg["stop_reason"] == "max_tokens", "a `length` finish is "
          "stop_reason max_tokens", json.dumps(msg)[:200])
    call = {"id": "call_X", "type": "function", "function": {
        "name": "write_file", "arguments": "{\"path\": \"a\""}}
    d = {"choices": [{"message": {"content": "", "tool_calls": [call]},
                      "finish_reason": "tool_calls"}]}
    out = M.of_chat(d, ctx)
    check(out["content"][0]["input"] == {"_unparsed_arguments":
                                         "{\"path\": \"a\""}
          and ctx.rec.get("tool_input_unparsed") == 1,
          "arguments that are not a JSON object are carried whole (the "
          "client's schema check then tells the model), and counted",
          json.dumps(out["content"]))
    # the pinger: a turn silent past ping_s gets pings between its chunks
    gate = threading.Event()

    def slow():
        yield _chunk({"content": "a"})
        gate.wait(2)
        yield _chunk({"content": "b"})
        yield _chunk({}, "stop", x_yamadori={})
        yield b"data: [DONE]\n\n"

    ctx = _ctx()
    it = M.stream(slow(), ctx, ping_s=0.05)
    raw = [next(it)]
    t0 = time.time()
    while time.time() - t0 < 0.4:
        raw.append(next(it))
    gate.set()
    raw.extend(it)
    ev = sse_decode(b"".join(raw))
    msg = check_stream(ev, "a silent turn")
    check(sum(1 for e in ev if e["type"] == "ping") >= 3
          and _text(msg) == "ab",
          "while the turn is silent a ping goes out every PING_S (the "
          "proxy's heartbeat cadence), from the pump thread",
          json.dumps([e["type"] for e in ev]))
    # E1: a failure before the first chunk RAISES (a real HTTP status)

    def refuses():
        raise api_errors.context_length_exceeded(140000, 132096, 3072)
        yield b""                                        # pragma: no cover
    e = _raises(lambda: next(M.stream(refuses(), _ctx())))
    check(e is not None and e.code == "context_length_exceeded",
          "a failure before the first chunk raises out of the stream (the "
          "server answers it with its HTTP status)")
    # closing the stream closes the turn
    closed = []

    def turn():
        try:
            yield _chunk({"content": "a"})
            while True:
                yield _chunk({})
                time.sleep(0.01)
        finally:
            closed.append(True)
    it = M.stream(turn(), _ctx(), ping_s=0.05)
    next(it)
    next(it)
    it.close()
    t0 = time.time()
    while not closed and time.time() - t0 < 2:
        time.sleep(0.01)
    check(closed == [True], "closing the Messages stream closes "
          "stream_body (whose finally cancels the turn's token)")


def test_errors():
    rows = []
    for err, want_status, want_type in [
        (api_errors.invalid("bad", param="x"), 400, "invalid_request_error"),
        (M.unauthorised("no API key supplied"), 401, "authentication_error"),
        (api_errors.ApiError(403, "no"), 403, "permission_error"),
        (api_errors.unknown_route("POST", "/v1/x"), 404, "not_found_error"),
        (api_errors.ApiError(409, "superseded", code="superseded"), 409,
         "conflict_error"),
        (api_errors.ApiError(413, "big"), 413, "request_too_large"),
        (api_errors.ApiError(429, "full", code="server_busy",
                             headers={"Retry-After": "30"}), 429,
         "overloaded_error"),
        (api_errors.ApiError(503, "capacity", code="model_at_capacity",
                             headers={"Retry-After": "30"}), 503,
         "overloaded_error"),
        (api_errors.ApiError(503, "loading", code="model_unavailable"), 503,
         "overloaded_error"),
        (api_errors.ApiError(502, "upstream", code="upstream_error"), 502,
         "api_error"),
        (api_errors.ApiError(500, "ours", code="internal_error"), 500,
         "api_error"),
        (api_errors.ApiError(504, "slow", code="upstream_timeout"), 504,
         "timeout_error"),
    ]:
        st, body, h = M.anthropic_error(err)
        ok = (st == want_status and body.get("type") == "error"
              and body["error"]["type"] == want_type
              and isinstance(body["error"]["message"], str)
              and body.get("request_id", "").startswith("req_")
              and h.get("request-id") == body["request_id"]
              and (h.get("Retry-After") == "30") == ("Retry-After" in (
                  err.headers or {})))
        rows.append((want_status, want_type, ok, body["error"]["type"]))
    check(all(r[2] for r in rows),
          "our errors in Anthropic's envelope {type: error, error: {type, "
          "message}, request_id}: 400 invalid_request_error, 401 "
          "authentication_error, 403 permission_error, 404 not_found_error, "
          "409 conflict_error, 413 request_too_large, 429/503 capacity "
          "overloaded_error with Retry-After, 5xx api_error, 504 "
          "timeout_error", json.dumps([r for r in rows if not r[2]]))
    st, body, _h = M.anthropic_error(api_errors.context_length_exceeded(
        140000, 132096, 3072))
    check(st == 400 and body["error"]["type"] == "invalid_request_error"
          and body["error"]["message"] ==
          "prompt is too long: 140000 tokens > 129024 maximum"
          and body["error"]["x_yamadori"]["code"] == "context_length_exceeded",
          "a prompt past the window: 400, in Anthropic's own wording (\"prompt "
          "is too long: N tokens > M maximum\", M = the window less what an "
          "answer needs), which Claude Code compacts on", json.dumps(body))
    st, body, _h = M.anthropic_error(api_errors.context_length_exceeded(
        131000, 132096, 3072))
    check("131000 tokens > 129024 maximum" in body["error"]["message"],
          "a prompt that fits but leaves too little to answer states the "
          "maximum it must get under", body["error"]["message"])
    ctx = _ctx()
    s = M.Stream(ctx)
    s.feed(_chunk({"content": "par"}))
    raw = s.feed(api_errors.context_length_exceeded(140000, 132096,
                                                   3072).sse())
    raw += s.feed(b"data: [DONE]\n\n")
    ev = sse_decode(raw)
    check([e["type"] for e in ev] == ["error"]
          and ev[0]["error"]["type"] == "invalid_request_error"
          and ev[0]["error"]["message"].startswith("prompt is too long:"),
          "after the first byte a failure is ONE `error` event in "
          "Anthropic's form; nothing after it", json.dumps(ev)[:300])
    ev = sse_decode(b"".join(M.stream(iter([
        _chunk({"content": "part"}), _chunk({}, "incomplete"),
        b"data: [DONE]\n\n"]), _ctx())))
    msg = check_stream(ev, "a dropped upstream")
    check((msg.get("_error") or {}).get("type") == "api_error"
          and (msg.get("_error") or {}).get("x_yamadori", {}).get("code") ==
          "upstream_dropped",
          "the model server dropping mid-answer is an `error` event "
          "(api_error), never a clean end", json.dumps(msg.get("_error")))
    e = _raises(M.of_chat, {"choices": [{"message": {"content": "p"},
                                         "finish_reason": "incomplete"}]},
                _ctx())
    check(e is not None and e.status == 502,
          "blocking: the same drop is a 502 api_error")


# ------------------------------------------------------------- the route ----
from starlette.testclient import TestClient  # noqa: E402
import server  # noqa: E402

KEY = accounts.create("messages-tests")
CLIENT = TestClient(server.app)


def post(body, path="/v1/messages?beta=true", headers=None, raw=None,
         stream=False, auth="x-api-key"):
    h = {"anthropic-version": "2023-06-01"}
    if auth == "x-api-key":
        h["x-api-key"] = KEY
    elif auth == "bearer":
        h["Authorization"] = f"Bearer {KEY}"
    h.update(headers or {})
    if raw is not None:
        r = CLIENT.post(path, content=raw, headers=dict(
            h, **{"Content-Type": "application/json"}))
        return r.status_code, r.headers, r.json(), None
    if stream:
        with CLIENT.stream("POST", path, json=body, headers=h) as r:
            data = b"".join(r.iter_bytes())
            if r.headers.get("content-type", "").startswith(
                    "text/event-stream"):
                return r.status_code, r.headers, None, sse_decode(data)
            return r.status_code, r.headers, json.loads(data), None
    r = CLIENT.post(path, json=body, headers=h)
    return r.status_code, r.headers, r.json(), None


def _is_error(d, kind=None) -> bool:
    return (isinstance(d, dict) and d.get("type") == "error"
            and isinstance(d.get("error"), dict)
            and (kind is None or d["error"].get("type") == kind))


def test_the_route_and_auth():
    T.slots.reset(n=4)
    body = {"model": "claude-opus-5-5", "max_tokens": 1024,
            "system": "Route test.", "messages": [
                {"role": "user", "content": "Say hi."}],
            "output_config": {"effort": "low"}}
    T._script[:] = [T.reply("Hi.", reasoning="Greeting.")]
    st, _h, d, _ = post(body, headers={"x-claude-code-session-id": "route-1"})
    check(st == 200 and d.get("type") == "message" and _text(d) == "Hi."
          and d.get("model") == "claude-opus-5-5"
          and d.get("stop_reason") == "end_turn"
          and (d.get("x_yamadori") or {}).get("messages", {}).get(
              "anthropic_version") == "2023-06-01",
          "POST /v1/messages?beta=true with x-api-key: 200, a Message; a "
          "claude-* model name resolves to Yamadori and is echoed",
          json.dumps(d)[:400])
    T._script[:] = [T.reply("Hello.")]
    st, _h, d, _ = post(dict(body, messages=[{"role": "user",
                                             "content": "Again."}]),
                        headers={"x-claude-code-session-id": "route-2"},
                        auth="bearer")
    check(st == 200 and _text(d) == "Hello.",
          "Authorization: Bearer (ANTHROPIC_AUTH_TOKEN) works too",
          json.dumps(d)[:200])
    T._script[:] = [T.reply("Both.")]
    st, _h, d, _ = post(dict(body, messages=[{"role": "user",
                                             "content": "Both?"}]),
                        headers={"Authorization": "Bearer ym-wrong-but-"
                                 "shaped-not-a-key-000000",
                                 "x-claude-code-session-id": "route-2b"})
    check(st == 200 and _text(d) == "Both.",
          "a wrong Bearer beside a good x-api-key: the key that names an "
          "account wins", json.dumps(d)[:200])
    st, _h, d, _ = post(body, auth=None)
    check(st == 401 and _is_error(d, "authentication_error"),
          "no key: 401 authentication_error in Anthropic's envelope",
          json.dumps(d)[:200])
    st, _h, d, _ = post(body, auth=None, headers={"x-api-key": "nope"})
    check(st == 401 and _is_error(d, "authentication_error"),
          "an unknown key: 401", json.dumps(d)[:200])
    T._script[:] = [T.reply("Streamed hello.", reasoning="Greeting.")]
    st, h, _d, ev = post(dict(body, stream=True), stream=True,
                         headers={"x-claude-code-session-id": "route-3"})
    check(st == 200 and h.get("content-type", "").startswith(
        "text/event-stream") and ev, "streamed: 200 text/event-stream",
          f"{st} {h.get('content-type')}")
    if ev:
        msg = check_stream(ev, "the route, streamed")
        check(_text(msg) == "Streamed hello."
              and msg["content"][0]["type"] == "thinking",
              "the streamed Message accumulates to the answer", json.dumps(
                  msg["content"])[:300])
    T._script[:] = []                # the fake upstream answers 500
    st, _h, d, ev = post(dict(body, stream=True), stream=True,
                         headers={"x-claude-code-session-id": "route-4"})
    check(st == 502 and ev is None and _is_error(d, "api_error"),
          "a failure before the first byte is a real HTTP status in "
          "Anthropic's envelope, not a 200 stream (E1)",
          f"{st} {json.dumps(d)[:200]}")
    st, _h, d, _ = post(dict(body, max_tokens=None))
    check(st == 400 and _is_error(d, "invalid_request_error")
          and "max_tokens" in d["error"]["message"],
          "a body we cannot serve: 400 invalid_request_error",
          json.dumps(d)[:200])
    st, _h, d, _ = post(None, raw=b"{nope")
    check(st == 400 and _is_error(d, "invalid_request_error"),
          "bad JSON: 400 in Anthropic's envelope", json.dumps(d)[:200])
    r = CLIENT.get("/v1/messages", headers={"x-api-key": KEY})
    check(r.status_code == 405 and "POST" in r.headers.get("allow", ""),
          "GET /v1/messages: 405, Allow: POST", f"{r.status_code}")
    r = CLIENT.get("/", headers={"Accept": "application/json"})
    eps = r.json().get("endpoints") or []
    check("/v1/messages" in eps and "/v1/messages/count_tokens" in eps,
          "the root descriptor lists the Messages routes", r.text[:300])
    internal, hint, known = catalog.resolve("claude-sonnet-5-5")
    r = CLIENT.get("/v1/models/claude-opus-5-5")
    check(internal == "bonsai" and hint is None and not known
          and r.status_code == 200 and r.json().get("id") == "yamadori",
          "a claude-* model id resolves to the product (no tier of its "
          "own), and GET /v1/models/<it> answers the product's card",
          r.text[:200])
    # THE WINDOW CHECK, through the route: a prompt past the window is
    # refused before anything runs, in the wording Claude Code compacts on.
    real = proxy.window_limit
    proxy.window_limit = lambda payload: 3500
    try:
        T._script[:] = [T.reply("never")]
        st, _h, d, _ = post(dict(body, messages=[{"role": "user",
                                                  "content": "x " * 900}]),
                            headers={"x-claude-code-session-id": "route-5"})
        check(st == 400 and _is_error(d, "invalid_request_error")
              and d["error"]["message"].startswith("prompt is too long: ")
              and d["error"]["message"].endswith(" maximum")
              and d["error"]["x_yamadori"]["code"] ==
              "context_length_exceeded",
              "the window check (counted by the model server) refuses a "
              "prompt past the window: 400 \"prompt is too long: N tokens > "
              "M maximum\"", json.dumps(d)[:400])
    finally:
        proxy.window_limit = real
        T._script[:] = []


def test_count_tokens():
    body = {"model": "claude-opus-5-5", "system": "Count me.",
            "messages": [{"role": "user", "content": "How many tokens?"}],
            "tools": [WRITE_TOOL]}
    n0 = len(T._tokenized)
    st, _h, d, _ = post(body, path="/v1/messages/count_tokens")
    chat, _ = M.to_chat(body, {}, "a", count_only=True)
    want = len(T.render({"messages": chat["messages"],
                         "tools": chat["tools"]}))
    check(st == 200 and d.get("input_tokens") == want
          and (d.get("x_yamadori") or {}).get("counted") is True
          and len(T._tokenized) == n0 + 1,
          "count_tokens: the model server's own count (the served template's "
          "render, tokenized), no max_tokens needed",
          json.dumps([d, want])[:300])
    st, _h, d, _ = post(body, path="/v1/messages/count_tokens", auth="bearer")
    check(st == 200 and d.get("input_tokens") == want,
          "count_tokens with Authorization: Bearer", json.dumps(d)[:200])
    st, _h, d, _ = post(body, path="/v1/messages/count_tokens", auth=None)
    check(st == 401 and _is_error(d, "authentication_error"),
          "count_tokens needs a key", json.dumps(d)[:200])
    real = proxy.count_prompt_tokens
    proxy.count_prompt_tokens = lambda payload, **k: None
    try:
        st, _h, d, _ = post(body, path="/v1/messages/count_tokens")
        check(st == 200 and d.get("input_tokens") == proxy.high_estimate(
            {"messages": chat["messages"], "tools": chat["tools"]})
              and d["x_yamadori"]["counted"] is False
              and "estimate" in d["x_yamadori"]["why"],
              "when the model server cannot count, the window check's high "
              "estimate answers and says so", json.dumps(d)[:300])
    finally:
        proxy.count_prompt_tokens = real
    st, _h, d, _ = post({"messages": []}, path="/v1/messages/count_tokens")
    check(st == 400 and _is_error(d, "invalid_request_error"),
          "a bad count_tokens body: 400", json.dumps(d)[:200])


def test_features_that_touch_the_wire():
    """Images (placeholders, never the bytes upstream), a hidden hop (our
    image tool: the client sees the picture, never our call), stray template
    markers (stripped from the client's copy)."""
    T.slots.reset(n=4)
    T.compaction.reset()
    c = CClient("wire")
    t1 = c.turn([T.reply("A green square.")],
                user=[{"type": "text", "text": "What is this?"}, IMAGE])
    up = json.dumps(t1["gens"][0]["request"]["messages"])
    check(PNG_B64[:40] not in up and "image_url" not in up and "image-" in up,
          "an image block reaches the model as a placeholder naming an id "
          "(the chat image path); no image byte goes upstream", up[:300])
    proxy.run_our_tool = T._real_run_our_tool
    try:
        n_drawn = len(DRAWN)
        t2 = c.turn([T.reply("", calls=[T.call(
            "yama_generate_image", {"prompt": "a lathe"}, "wire-g")]),
            T.reply("Here it is.")], user="Draw a lathe.")
    finally:
        proxy.run_our_tool = T._run_our_tool
    msg = t2["msg"]
    check(len(DRAWN) == n_drawn + 1 and "/media/" in _text(msg)
          and _text(msg).rstrip().endswith("Here it is.")
          and not _blocks(msg, "tool_use")
          and msg["stop_reason"] == "end_turn",
          "our image tool runs as a hidden hop: the picture is in the text, "
          "no tool_use of ours reaches Claude Code", json.dumps(
              msg["content"])[:300])
    t3 = c.turn([T.reply("Done.\n</think>\n\nDone.", reasoning="Done.")],
                user="Say done.")
    check(_text(t3["msg"]).endswith("Done.")
          and "</think>" not in _text(t3["msg"])
          and _text(t3["msg"]).count("Done.") == 1
          and (t3["x"].get("template_markers") or {}).get("stripped"),
          "a stray template marker the model wrote is stripped from the "
          "client's copy (with its repeat)", json.dumps(t3["msg"]["content"]))
    T._extends(t2, t3, "messages: after a hidden-hop turn")


def main() -> int:
    for fn in (test_request_translation, test_request_errors,
               test_effort_to_tier, test_blocking_tool_loop_extends_the_slot,
               test_a_client_that_drops_thinking_gets_it_restored,
               test_streamed_equals_blocking, test_stream_forms_and_keepalive,
               test_errors, test_the_route_and_auth, test_count_tokens,
               test_features_that_touch_the_wire):
        print(f"\n--- {fn.__name__} ---")
        n0 = len(_results)
        t0 = len(T._results)
        try:
            fn()
        except Exception:                                        # noqa: BLE001
            check(False, f"{fn.__name__} itself raised",
                  traceback.format_exc().strip().split("\n")[-1])
            traceback.print_exc()
        _results.extend(T._results[t0:])
        del T._results[t0:]
        for ok, name, detail in _results[n0:]:
            print(("  pass  " if ok else "  FAIL  ") + name
                  + (f"   <- {detail[:400]}" if not ok and detail else ""))
    T._srv.shutdown()
    passed = sum(1 for ok, _, _ in _results if ok)
    total = len(_results)
    print(f"\n{'=' * 70}\n  {passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
