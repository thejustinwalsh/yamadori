#!/usr/bin/env python
"""The Responses API (`POST /v1/responses`, mcp/responses_api.py). No GPU.

    python mcp/test_responses_api.py      -> "N/M checks passed"

WHAT THIS GATES (docs/OPENAI-CONFORMANCE.md R1 and section 10; operator
2026-09-26: "both need to exist"; Responses is the primary path):

  1. TRANSLATION. Every input item type -- messages with input_text /
     input_image / output_text parts, function_call, function_call_output as
     a string AND as an array with images (Codex's view_image), custom tool
     calls, reasoning items (content read, summary and encrypted_content
     not), developer items, hosted-call records -- becomes the chat body the
     one turn takes; tools, tool_choice, reasoning.effort, max_output_tokens,
     text.format and prompt_cache_key map; what cannot be served is a 400 in
     the error object.
  2. THE SAME TURN. Through proxy.complete / proxy.stream_body and the
     SERVED chat template (test_ledger's harness): a Codex-shaped client that
     replays every output item it was given keeps the slot's prefix across
     turns (blocking and streamed), the session comes from prompt_cache_key
     (call ids untouched) or, without one, rides in the call ids; skills,
     the tool-turn cap and x_yamadori reach the Response.
  3. THE STREAM, read the way openai-python reads it (its SSEDecoder rule,
     reproduced below) and checked against the spec's required fields per
     event (openai-openapi 2.3.0), the helper's accumulation, and Codex's
     item parser (codex-rs sse.rs / protocol models.rs): created first, one
     terminal event last, sequence numbers 0.., items added/done in order.
     Heartbeats are REAL events (Hermes' codex transport resets its 12 s
     idle timer only on a parsed event); a mid-stream failure is
     response.failed with Codex's codes; a length finish is
     response.incomplete.
  4. THE ROUTE (FastAPI TestClient): auth, bad JSON, the 400s, a failure
     before the first byte as a real HTTP status (E1), GET -> 405, the root
     descriptor lists the route.
  5. IMAGES. input_image data URLs and function_call_output images go to
     the chat image path (placeholders, never the bytes upstream); http
     URLs are not fetched. The HOSTED image_generation tool: our
     yama_generate_image runs at the client's size, and the image comes back as
     an image_generation_call item (blocking and streamed, the PNG as
     base64, our signed link as x_yamadori.url); unsupported values are
     400. /v1/images/generations against the Images API spec.

Every database and store is a temp path set BEFORE proxy is imported.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import struct
import sys
import tempfile
import time
import traceback
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

_TMP = tempfile.mkdtemp(prefix="yamadori_test_responses_")
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
os.environ.pop("YAMADORI_RESPONSES_REASONING", None)

import test_ledger as T  # noqa: E402  (its own temp paths + fake upstream)
import accounts  # noqa: E402
import api_errors  # noqa: E402
import catalog  # noqa: E402
import images  # noqa: E402
import proxy  # noqa: E402
import responses_api as R  # noqa: E402
import session_id  # noqa: E402
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
DATA_URL = "data:image/png;base64," + PNG_B64

# The vision model is never reached from here (a future eager description
# included): a fake that records what it was asked.
SEEN_BY_VISION: list[dict] = []


def _describe(data, fmt, question):
    SEEN_BY_VISION.append({"fmt": fmt, "question": question})
    return {"answer": "A small green square.", "seconds": 0.01, "usage": {},
            "finish": "stop"}


vision.describe = _describe

# The fake image generator: a REAL PNG in the (temp) media store, so the
# signed link verifies and the Response carries the real bytes.
DRAWN: list[dict] = []


def _generate(prompt, size=None, seed=None, steps=None, n=1, model=None):
    w, h = images.parse_size(size)
    DRAWN.append({"prompt": prompt, "size": f"{w}x{h}", "model": model})
    meta = {"prompt": prompt, "seed": 7, "size": f"{w}x{h}", "steps": 4,
            "model": "imagegen-turbo", "image_model": "yamadori-image-turbo",
            "seconds": 0.01, "created": int(time.time())}
    sha = images.store(tiny_png((len(DRAWN) % 250, 40, 80)), meta)
    return [dict(meta, id=sha)]


images.generate = _generate
BASE = "https://img.example.test"

FLAT_WRITE = {"type": "function", "name": "write_file",
              "description": "Write a file.",
              "parameters": T.WRITE["function"]["parameters"],
              "strict": False}


# ---------------------------------------------------- the SSE reader ----
# openai-python's rule (src/openai/_streaming.py SSEDecoder.decode and
# Stream.__stream__): lines split on CR/LF; ':' lines are comments; an
# empty line ends an event; `event:` and `data:` fields; a data payload
# with a truthy top-level "error" key RAISES APIError.

class SSEError(Exception):
    pass


def sse_decode(raw: bytes) -> list[dict]:
    out, event, data = [], None, []
    for line in raw.decode("utf-8").replace("\r\n", "\n").split("\n"):
        if not line:
            if event is None and not data:
                continue
            payload = "\n".join(data)
            event, data = event, []
            if payload.startswith("[DONE]"):
                out.append({"_event": event, "_done": True})
            else:
                d = json.loads(payload)
                if isinstance(d, dict) and d.get("error"):
                    raise SSEError(json.dumps(d["error"]))
                d["_event"] = event
                out.append(d)
            event = None
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if field == "event":
            event = value
        elif field == "data":
            data.append(value)
    return out


# Required fields per event (openai-openapi 2.3.0 components.schemas, the
# Response*Event schemas' `required`).
EVENT_REQUIRED = {
    "response.created": ["response"],
    "response.in_progress": ["response"],
    "response.output_item.added": ["item", "output_index"],
    "response.output_item.done": ["item", "output_index"],
    "response.content_part.added": ["content_index", "item_id",
                                    "output_index", "part"],
    "response.content_part.done": ["content_index", "item_id",
                                   "output_index", "part"],
    "response.output_text.delta": ["content_index", "delta", "item_id",
                                   "logprobs", "output_index"],
    "response.output_text.done": ["content_index", "item_id", "logprobs",
                                  "output_index", "text"],
    "response.reasoning_text.delta": ["content_index", "delta", "item_id",
                                      "output_index"],
    "response.reasoning_text.done": ["content_index", "item_id",
                                     "output_index", "text"],
    "response.reasoning_summary_part.added": ["item_id", "output_index",
                                              "part", "summary_index"],
    "response.reasoning_summary_part.done": ["item_id", "output_index",
                                             "part", "summary_index"],
    "response.reasoning_summary_text.delta": ["delta", "item_id",
                                              "output_index",
                                              "summary_index"],
    "response.reasoning_summary_text.done": ["item_id", "output_index",
                                             "summary_index", "text"],
    "response.function_call_arguments.delta": ["delta", "item_id",
                                               "output_index"],
    "response.function_call_arguments.done": ["arguments", "item_id",
                                              "output_index"],
    "response.custom_tool_call_input.delta": ["delta", "item_id",
                                              "output_index"],
    "response.custom_tool_call_input.done": ["input", "item_id",
                                             "output_index"],
    "response.image_generation_call.completed": ["item_id", "output_index"],
    "response.completed": ["response"],
    "response.incomplete": ["response"],
    "response.failed": ["response"],
}
RESPONSE_REQUIRED = ["created_at", "error", "id", "incomplete_details",
                     "instructions", "metadata", "model", "object", "output",
                     "parallel_tool_calls", "temperature", "tool_choice",
                     "tools", "top_p"]
TERMINAL = ("response.completed", "response.incomplete", "response.failed")


def codex_item_ok(item: dict) -> bool:
    """codex-rs protocol models.rs ResponseItem: the fields each variant it
    deserializes requires (serde; Option fields may be absent)."""
    t = item.get("type")
    if t == "message":
        return isinstance(item.get("role"), str) and all(
            isinstance(c, dict) and c.get("type") in (
                "output_text", "input_text") and isinstance(c.get("text"), str)
            for c in item.get("content") or [])
    if t == "function_call":
        return all(isinstance(item.get(k), str)
                   for k in ("name", "arguments", "call_id"))
    if t == "custom_tool_call":
        return all(isinstance(item.get(k), str)
                   for k in ("name", "input", "call_id"))
    if t == "reasoning":
        return isinstance(item.get("summary"), list)
    if t == "image_generation_call":
        return isinstance(item.get("status"), str) and \
            isinstance(item.get("result"), str)
    return False


def check_stream(events: list[dict], label: str) -> dict:
    """The spec-driven checks, the openai-python helper's accumulation and
    Codex's parse. Returns the terminal response."""
    problems = []
    types = [e.get("type") for e in events]
    if not events or types[0] != "response.created":
        problems.append(f"first event is {types[:1]}")
    terms = [i for i, t in enumerate(types) if t in TERMINAL]
    if len(terms) != 1 or terms[0] != len(events) - 1:
        problems.append(f"terminal events at {terms} of {len(events)}")
    for i, e in enumerate(events):
        if e.get("_event") != e.get("type"):
            problems.append(f"#{i} event: {e.get('_event')} != type "
                            f"{e.get('type')}")
        if e.get("sequence_number") != i:
            problems.append(f"#{i} sequence_number {e.get('sequence_number')}")
        for k in EVENT_REQUIRED.get(e.get("type"), []):
            if k not in e:
                problems.append(f"#{i} {e['type']} lacks {k}")
        if e.get("type") in ("response.created",) + TERMINAL:
            for k in RESPONSE_REQUIRED:
                if k not in (e.get("response") or {}):
                    problems.append(f"#{i} {e['type']}.response lacks {k}")
    # openai-python ResponseStreamState.accumulate_event, reduced: the
    # snapshot starts at response.created; output_item.added appends;
    # content_part.added adds a part; deltas append to that part.
    snap = None
    for e in events:
        t = e.get("type")
        if t == "response.created":
            snap = copy.deepcopy(e["response"])
            continue
        if snap is None:
            problems.append(f"{t} before response.created")
            break
        if t == "response.output_item.added":
            if e["output_index"] != len(snap["output"]):
                problems.append(f"added at {e['output_index']}, snapshot "
                                f"has {len(snap['output'])}")
            snap["output"].append(copy.deepcopy(e["item"]))
        elif t == "response.content_part.added":
            snap["output"][e["output_index"]].setdefault(
                "content", []).append(copy.deepcopy(e["part"]))
        elif t == "response.output_text.delta":
            part = snap["output"][e["output_index"]]["content"][
                e["content_index"]]
            part["text"] += e["delta"]
        elif t == "response.reasoning_summary_part.added":
            snap["output"][e["output_index"]].setdefault(
                "summary", []).append(copy.deepcopy(e["part"]))
        elif t == "response.reasoning_summary_text.delta":
            s = snap["output"][e["output_index"]]["summary"][
                e["summary_index"]]
            s["text"] += e["delta"]
        elif t == "response.output_item.done":
            item = e["item"]
            if not codex_item_ok(item):
                problems.append(f"Codex cannot parse {json.dumps(item)[:200]}")
            got = snap["output"][e["output_index"]]
            if got.get("id") != item.get("id"):
                problems.append(f"done id {item.get('id')} at "
                                f"{e['output_index']} was {got.get('id')}")
            if item.get("type") == "message":
                acc = "".join(c.get("text", "") for c in got.get("content")
                              or [])
                fin = "".join(c.get("text", "") for c in item["content"])
                if acc != fin:
                    problems.append(f"message deltas {acc!r} != done {fin!r}")
            if item.get("type") == "reasoning" and item.get("summary"):
                acc = "".join(s.get("text", "") for s in got.get("summary")
                              or [])
                if acc != item["summary"][0]["text"]:
                    problems.append("summary deltas != done")
            snap["output"][e["output_index"]] = copy.deepcopy(item)
    final = (events[-1].get("response") or {}) if events else {}
    if snap is not None and final.get("output") is not None:
        if [o.get("id") for o in snap["output"]] != \
                [o.get("id") for o in final["output"]]:
            problems.append("accumulated item ids differ from the terminal "
                            "response's output")
    u = final.get("usage")
    if final.get("status") in ("completed", "incomplete") and not (
            isinstance(u, dict) and all(isinstance(u.get(k), int) for k in (
                "input_tokens", "output_tokens", "total_tokens"))):
        problems.append(f"terminal usage {u}")
    check(not problems, f"{label}: the stream reads cleanly (openai-python's "
          f"decoder, the spec's required fields, the helper's accumulation, "
          f"Codex's item parse; {len(events)} events)",
          "; ".join(problems[:6]))
    return final


# ------------------------------------------------------------ the client ----

class RClient:
    """A Codex-shaped client: instructions + input items, flat tools,
    store false, reasoning effort + summary auto, include encrypted
    reasoning, and it REPLAYS every output item it was given (the terminal
    response's output, or the output_item.done items of a stream)."""

    def __init__(self, tag: str, *, stream: bool = False,
                 effort: str = "medium", cache_key: str | None = "auto",
                 tools: list | None = None):
        self.tag = tag
        self.stream = stream
        self.effort = effort
        self.account = f"{T.ACCOUNT}-resp-{tag}"
        self.cache_key = (f"codex-{tag}-{os.getpid()}" if cache_key == "auto"
                          else cache_key)
        self.instructions = T.SYSTEM + f" [{tag}]"
        self.input: list[dict] = []
        self.tools = tools if tools is not None else [FLAT_WRITE]
        self.ids: dict[str, str] = {}

    def body(self) -> dict:
        b = {"model": "yamadori", "instructions": self.instructions,
             "input": copy.deepcopy(self.input), "tools": self.tools,
             "tool_choice": "auto", "parallel_tool_calls": True,
             "reasoning": {"effort": self.effort, "summary": "auto"},
             "store": False, "stream": self.stream,
             "include": ["reasoning.encrypted_content"]}
        if self.cache_key:
            b["prompt_cache_key"] = self.cache_key
        return b

    def chat_of(self, body: dict):
        chat, ctx = R.to_chat(body)
        chat.update(_account=self.account, _client_ip="127.0.0.1",
                    _public_base=BASE, _session_token="",
                    _features=json.dumps({"skills": True}))
        return chat, ctx

    def turn(self, script: list[dict], user: str | list | None = None
             ) -> dict:
        if user is not None:
            self.input.append({"type": "message", "role": "user",
                               "content": ([{"type": "input_text",
                                             "text": user}]
                                           if isinstance(user, str)
                                           else user)})
        T._script[:] = list(script)
        n0, w0 = len(T._gens), len(T._warms)
        chat, ctx = self.chat_of(self.body())
        events = None
        if self.stream:
            raw = b"".join(R.stream(proxy.stream_body(chat, ctx.model), ctx))
            events = sse_decode(raw)
            resp = events[-1].get("response") or {}
            kept = [e["item"] for e in events
                    if e.get("type") == "response.output_item.done"]
        else:
            d = catalog.rewrite_response(proxy.complete(chat), ctx.model)
            resp = R.of_chat(d, ctx)
            kept = list(resp.get("output") or [])
        # The ids the proxy returned, by the script's own ids (in order).
        gen = T._gens[-1]["reply"] if len(T._gens) > n0 else {"calls": []}
        mine = [c for c in gen["calls"]
                if c["function"]["name"] not in proxy.OUR_NAMES]
        calls = [o for o in kept if o.get("type") in ("function_call",
                                                      "custom_tool_call")]
        for a, b in zip(mine, calls):
            self.ids[a["id"]] = b["call_id"]
        self.input.extend(copy.deepcopy(kept))
        x = resp.get("x_yamadori") or {}
        if (x.get("warm") or {}).get("sent"):
            end = time.time() + 5
            while time.time() < end and len(T._warms) == w0:
                time.sleep(0.02)
        return {"resp": resp, "events": events, "chat": chat, "x": x,
                "gens": T._gens[n0:], "warm": T._warms[w0:], "d": {
                    "x_yamadori": x}}

    def tool_output(self, cid: str, output) -> None:
        self.input.append({"type": "function_call_output",
                           "call_id": self.ids.get(cid, cid),
                           "output": output})


def _items(resp: dict, kind: str) -> list[dict]:
    return [o for o in resp.get("output") or [] if o.get("type") == kind]


def _text(resp: dict) -> str:
    return "".join(c.get("text", "") for m in _items(resp, "message")
                   for c in m.get("content") or [])


def _raises(fn, *a, **k):
    try:
        fn(*a, **k)
    except api_errors.ApiError as e:
        return e
    return None


# ----------------------------------------------------------------- tests ----

def test_request_translation():
    body = {
        "model": "gpt-5.1-codex", "instructions": "Be terse.",
        "input": [
            {"type": "message", "role": "developer",
             "content": [{"type": "input_text", "text": "<permissions/>"}]},
            {"role": "user", "content": "Plain string content."},
            {"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "<image name=[Image #1]>"},
                {"type": "input_image", "image_url": DATA_URL,
                 "detail": "high"},
                {"type": "input_text", "text": "</image>"}]},
            {"type": "reasoning", "id": "rs_1", "summary": [
                {"type": "summary_text", "text": "a summary"}],
             "content": [{"type": "reasoning_text", "text": "I think."}],
             "encrypted_content": "gAAAA-opaque"},
            {"type": "message", "role": "assistant", "id": "msg_1",
             "status": "completed", "phase": "commentary",
             "content": [{"type": "output_text", "text": "Writing it.",
                          "annotations": []}]},
            {"type": "function_call", "call_id": "call_A", "name":
             "write_file", "arguments": "{\"path\": \"a\"}"},
            {"type": "web_search_call", "id": "ws_1", "status": "completed"},
            {"type": "custom_tool_call", "call_id": "call_B",
             "name": "apply_patch", "input": "*** Begin Patch"},
            {"type": "function_call_output", "call_id": "call_A",
             "output": "wrote a"},
            {"type": "custom_tool_call_output", "call_id": "call_B",
             "output": [{"type": "input_text", "text": "Done"},
                        {"type": "input_text", "text": "!"}]},
            {"type": "reasoning", "summary": [
                {"type": "summary_text", "text": "only a summary"}],
             "encrypted_content": "gAAAA-2"},
            {"type": "function_call", "call_id": "call_C",
             "name": "view_image", "arguments": "{\"path\": \"s.png\"}"},
            {"type": "function_call_output", "call_id": "call_C", "output": [
                {"type": "input_text", "text": "the screenshot"},
                {"type": "input_image", "image_url": DATA_URL},
                {"type": "input_file", "filename": "log.bin",
                 "file_id": "file_1"}]},
            {"type": "image_generation_call", "id": "ig_9",
             "status": "completed", "result": "Zm9v"},
            {"type": "message", "role": "developer",
             "content": "<model_switch/>"},
            {"type": "message", "role": "user", "content": [
                {"type": "input_image", "file_id": "file_img"},
                {"type": "input_text", "text": "and this?"}]},
        ],
        "tools": [FLAT_WRITE,
                  {"type": "custom", "name": "apply_patch",
                   "description": "Patch files.",
                   "format": {"type": "grammar", "syntax": "lark",
                              "definition": "start: /.+/"}},
                  {"type": "web_search", "external_web_access": False},
                  {"type": "file_search", "vector_store_ids": ["v"]}],
        "tool_choice": {"type": "function", "name": "write_file"},
        "parallel_tool_calls": False,
        "reasoning": {"effort": "high", "summary": "auto"},
        "max_output_tokens": 900,
        "text": {"format": {"type": "json_schema", "name": "out",
                            "schema": {"type": "object"}, "strict": True},
                 "verbosity": "low"},
        "prompt_cache_key": "conv-123", "store": True,
        "metadata": {"a": "b"}, "user": "u", "service_tier": "auto",
        "include": ["reasoning.encrypted_content"], "truncation": "auto",
        "client_metadata": {"x": "y"}, "stream": True,
        "stream_options": {"reasoning_summary_delivery": "sequential_cutoff"},
    }
    chat, ctx = R.to_chat(body)
    m = chat["messages"]
    roles = [x["role"] for x in m]
    check(roles == ["system", "user", "user", "assistant", "tool", "tool",
                    "assistant", "tool", "user", "user"],
          "items -> chat roles: ONE system first (instructions + the leading "
          "developer item), a later developer item a user message, one "
          "assistant per turn", json.dumps(roles))
    check(m[0]["content"] == "Be terse.\n\n<permissions/>"
          and m[8]["content"] == "<model_switch/>"
          and ctx.rec.get("developer_as_user") == 1,
          "instructions and leading system/developer text join; the later "
          "one is counted", json.dumps([m[0], m[8], ctx.rec]))
    check(m[1]["content"] == "Plain string content.",
          "an EasyInputMessage (no type) with string content", m[1])
    c2 = m[2]["content"]
    check(isinstance(c2, list) and [p["type"] for p in c2] == [
        "text", "image_url", "text"]
          and c2[1]["image_url"] == {"url": DATA_URL, "detail": "high"},
          "input_image (data URL) -> an image_url part, detail kept; the "
          "text frame around it stays text", json.dumps(c2)[:200])
    a = m[3]
    check(a.get("reasoning_content") == "I think."
          and a["content"] == "Writing it."
          and [c["id"] for c in a["tool_calls"]] == ["call_A", "call_B"]
          and a["tool_calls"][1]["function"] == {
              "name": "apply_patch",
              "arguments": json.dumps({"input": "*** Begin Patch"})},
          "reasoning (its reasoning_text) + message + function_call + "
          "custom_tool_call of one turn -> ONE assistant message; a hosted "
          "call record in between does not split it", json.dumps(a)[:400])
    check(m[4] == {"role": "tool", "tool_call_id": "call_A",
                   "content": "wrote a"}
          and m[5] == {"role": "tool", "tool_call_id": "call_B",
                       "content": "Done!"},
          "function_call_output (string) and custom_tool_call_output (a "
          "text array: joined as the template joins it) -> tool messages",
          json.dumps(m[4:6]))
    check(m[6].get("reasoning_content") is None
          and [c["id"] for c in m[6]["tool_calls"]] == ["call_C"],
          "a reasoning item with ONLY a summary and encrypted_content gives "
          "the model nothing (display text, opaque)", json.dumps(m[6]))
    t7 = m[7]["content"]
    # Since 2026-09-26 the image stays the Responses part it is, so
    # vision.normalise records its form as responses_output (the
    # placeholder's words are the same; mcp/test_harness_decisions.py
    # codex/responses/view-image).
    check(isinstance(t7, list) and [p["type"] for p in t7] == [
        "text", "input_image", "text"] and "log.bin" in t7[2]["text"]
          and t7[1]["image_url"].startswith("data:image/png;base64,"),
          "function_call_output as an ARRAY (view_image): its image kept as "
          "an input_image part for the chat image path; an input_file there "
          "becomes a note (refusing would wedge the conversation)",
          json.dumps(t7)[:300])
    c9 = m[9]["content"]
    check(c9[0] == {"type": "input_image", "file_id": "file_img"},
          "an input_image by file_id is kept as the Responses part, which "
          "vision.extract names and explains (no files API)", json.dumps(c9))
    rec = ctx.rec
    check(rec.get("encrypted_reasoning_ignored") == 2
          and rec.get("reasoning_summaries_not_read") == 2
          and rec.get("reasoning_restored") == 1
          and rec.get("items_ignored") == ["image_generation_call",
                                           "web_search_call"]
          and rec.get("store_requested") is True,
          "x_yamadori.responses records what was not given to the model",
          json.dumps(rec))
    tools = chat["tools"]
    check([t["function"]["name"] for t in tools] == ["write_file",
                                                     "apply_patch"]
          and tools[0] == {"type": "function", "function": {
              "name": "write_file", "description": "Write a file.",
              "parameters": FLAT_WRITE["parameters"], "strict": False}}
          and tools[1]["function"]["parameters"]["required"] == ["input"]
          and "start: /.+/" in tools[1]["function"]["description"]
          and rec.get("tools_ignored") == ["web_search", "file_search"]
          and ctx.custom == {"apply_patch"},
          "flat function tools nest; a custom tool is a one-string-input "
          "function (its grammar in the description); hosted web_search / "
          "file_search are ignored and recorded", json.dumps(tools)[:400])
    check(chat.get("tool_choice") == {"type": "function",
                                      "function": {"name": "write_file"}}
          and chat.get("parallel_tool_calls") is False
          and chat.get("reasoning_effort") == "high"
          and chat.get("max_tokens") == 900
          and chat.get("response_format") == {"type": "json_schema",
                                              "json_schema": {
                                                  "name": "out",
                                                  "schema": {"type": "object"},
                                                  "strict": True}}
          and chat.get("prompt_cache_key") == "conv-123"
          and chat.get("stream") is True
          and chat.get("stream_options") == {"include_usage": True}
          and not any(k in chat for k in ("metadata", "user", "store",
                                          "include", "truncation")),
          "tool_choice, parallel_tool_calls, reasoning.effort, "
          "max_output_tokens, text.format, prompt_cache_key map; usage is "
          "asked of the chat stream; the ignored fields do not pass",
          json.dumps({k: v for k, v in chat.items() if k not in (
              "messages", "tools")}))
    chat2, _ = R.to_chat({"input": "hi"})
    check(chat2["messages"] == [{"role": "user", "content": "hi"}]
          and api_errors.validate_chat(chat2) is None,
          "a string input is one user message, and a valid chat body",
          json.dumps(chat2))
    again, _ = R.to_chat(copy.deepcopy(body))
    check(again["messages"] == chat["messages"],
          "the translation is deterministic (the ledger's chain keys)")


def test_request_errors():
    cases = [
        ({"input": "x", "previous_response_id": "resp_1"},
         "previous_response_id", "unsupported_parameter"),
        ({"input": "x", "conversation": "conv_1"}, "conversation",
         "unsupported_parameter"),
        ({"input": "x", "background": True}, "background",
         "unsupported_parameter"),
        ({"input": [{"type": "item_reference", "id": "msg_1"}]}, "input[0]",
         "unsupported_parameter"),
        ({}, "input", "missing_required_parameter"),
        ({"input": 5}, "input", "invalid_type"),
        ({"input": [{"role": "user", "content": [
            {"type": "input_file", "file_id": "f"}]}]},
         "input[0].content[0].type", "invalid_content_part"),
        ({"input": [{"role": "user", "content": [
            {"type": "input_audio", "audio": "x"}]}]},
         "input[0].content[0].type", "invalid_content_part"),
        ({"input": "x", "tools": [{"type": "local_shell"}]}, "tools[0].type",
         "invalid_value"),
        ({"input": "x", "tools": [FLAT_WRITE], "tool_choice": {
            "type": "allowed_tools", "tools": []}}, "tool_choice",
         "invalid_value"),
        ({"input": "x", "max_output_tokens": 0}, "max_output_tokens",
         "invalid_value"),
        ({"input": "x", "text": {"format": {"type": "grammar"}}},
         "text.format", "invalid_value"),
        ({"input": "x", "tools": [{"type": "image_generation",
                                   "size": "1792x1024"}]},
         "tools[0].size", "invalid_value"),
        ({"input": "x", "tools": [{"type": "image_generation",
                                   "output_format": "webp"}]},
         "tools[0].output_format", "invalid_value"),
        ({"input": "x", "tools": [{"type": "image_generation",
                                   "background": "transparent"}]},
         "tools[0].background", "invalid_value"),
        ({"input": "x", "tools": [{"type": "image_generation",
                                   "action": "edit"}]},
         "tools[0].action", "invalid_value"),
        ({"input": "x", "top_logprobs": 2}, "top_logprobs", "invalid_value"),
        ({"input": [{"type": "message", "role": "system", "content": "s"}]},
         "input", "invalid_value"),
    ]
    bad = []
    for body, param, code in cases:
        e = _raises(R.to_chat, body)
        if not (e is not None and e.status == 400 and e.param == param
                and e.code == code and e.body()["error"]["type"] ==
                "invalid_request_error"):
            bad.append((json.dumps(body)[:80], param, code,
                        e and (e.status, e.param, e.code)))
    check(not bad, f"{len(cases)} requests that cannot be served are 400 in "
          f"the error object, naming the field (previous_response_id, "
          f"item_reference, input_file, local_shell, unsupported image "
          f"tool values, ...)", json.dumps(bad)[:600])
    ok = [{"input": "x", "store": True}, {"input": "x", "metadata": {"k": 1},
                                          "user": "u", "truncation": "auto",
                                          "service_tier": "flex",
                                          "safety_identifier": "s",
                                          "max_tool_calls": 3,
                                          "previous_response_id": None},
          {"input": "x", "tools": [{"type": "image_generation",
                                    "size": "1024x1536", "quality": "high",
                                    "output_format": "png",
                                    "background": "auto",
                                    "partial_images": 2,
                                    "model": "gpt-image-1"}]}]
    fails = [b for b in ok if _raises(R.to_chat, b) is not None]
    check(not fails, "store:true, metadata, user, truncation, service_tier, "
          "safety_identifier, max_tool_calls and a servable image tool are "
          "accepted", json.dumps(fails))


def test_codex_request_replay():
    """A request as codex-rs builds it (codex-api common.rs
    ResponsesApiRequest; tools from its spec for an unknown model: a
    function shell, update_plan, view_image, and web_search), with a
    view_image result and an image the user attached (`--image`)."""
    T.slots.reset(n=4)
    shot = [{"type": "input_text", "text": "<image name=[Image #1] "
             "path=/tmp/shot.png>"},
            {"type": "input_image", "image_url": DATA_URL, "detail": "high"},
            {"type": "input_text", "text": "</image>"}]
    body = {
        "model": "yamadori",
        "instructions": "You are Codex, a coding agent. [codex replay]",
        "input": [
            {"type": "message", "role": "developer", "content": [
                {"type": "input_text", "text":
                 "<permissions instructions>sandbox: workspace-write"
                 "</permissions instructions>"}]},
            {"type": "message", "role": "user", "content": [
                {"type": "input_text", "text":
                 "<environment_context><cwd>/work</cwd></environment_context>"
                 }]},
            {"type": "message", "role": "user", "content": shot + [
                {"type": "input_text", "text": "What is wrong in this UI?"}]},
            {"type": "reasoning", "id": "rs_x", "summary": [],
             "encrypted_content": None},
            {"type": "function_call", "call_id": "call_view1",
             "name": "view_image",
             "arguments": "{\"path\":\"/tmp/shot.png\"}"},
            {"type": "function_call_output", "call_id": "call_view1",
             "output": [{"type": "input_image", "image_url": DATA_URL}]},
        ],
        "tools": [
            {"type": "function", "name": "exec_command",
             "description": "Run a command.", "strict": False,
             "parameters": {"type": "object", "properties": {
                 "cmd": {"type": "string"}}, "required": ["cmd"],
                 "additionalProperties": False}},
            {"type": "function", "name": "update_plan",
             "description": "Update the plan.", "strict": False,
             "parameters": {"type": "object", "properties": {}}},
            {"type": "function", "name": "view_image",
             "description": "View a local image.", "strict": False,
             "parameters": {"type": "object", "properties": {
                 "path": {"type": "string"}}, "required": ["path"]}},
            # Codex 0.133.0 sends this on EVERY request
            # (docs/HARNESS-RESPONSES.md, captured).
            {"type": "namespace", "name": "multi_agent_v1",
             "description": "Sub-agents.", "tools": [
                 {"type": "function", "name": "spawn_agent",
                  "description": "Start a sub-agent.", "strict": False,
                  "parameters": {"type": "object", "properties": {
                      "task": {"type": "string"}}}}]},
            {"type": "web_search", "external_web_access": False}],
        "tool_choice": "auto", "parallel_tool_calls": False,
        "reasoning": {"effort": "medium", "summary": "auto"},
        "store": False, "stream": False,
        "include": ["reasoning.encrypted_content"],
        "prompt_cache_key": "019a-codex-conv-replay",
        "client_metadata": {"x-codex-installation-id": "abc"},
    }
    chat, ctx = R.to_chat(body)
    chat.update(_account=f"{T.ACCOUNT}-codex", _client_ip="127.0.0.1",
                _public_base=BASE, _session_token="")
    T._script[:] = [T.reply("The button overlaps the header.",
                            reasoning="Looking at the placeholder.")]
    n0 = len(T._gens)
    d = catalog.rewrite_response(proxy.complete(chat), ctx.model)
    resp = R.of_chat(d, ctx)
    up = T._gens[n0]["request"]
    blob = json.dumps(up["messages"])
    tool_msg = [x for x in up["messages"] if x.get("role") == "tool"]
    check(PNG_B64[:40] not in blob and "image_url" not in blob
          and tool_msg and "image-" in json.dumps(tool_msg)
          and "image-" in json.dumps(up["messages"][1:3]),
          "the attached image and the view_image result reach the model as "
          "placeholders naming an id (the chat image path); no image byte "
          "goes upstream", blob[:400])
    check(up["messages"][0]["role"] == "system"
          and sum(1 for x in up["messages"] if x["role"] == "system") == 1
          and "permissions" in up["messages"][0]["content"],
          "one system message upstream (the template refuses a second)",
          json.dumps([x["role"] for x in up["messages"]]))
    names = [t["function"]["name"] for t in up.get("tools") or []]
    check(names[:4] == ["exec_command", "update_plan", "view_image",
                        "spawn_agent"]
          and "web_search" not in names,
          "the client's function tools go up first and untouched, its "
          "namespace flattened (no 400: Codex sends one on every request); "
          "the hosted web_search does not", json.dumps(names))
    x = resp.get("x_yamadori") or {}
    check((x.get("session") or {}).get("source") == "prompt_cache_key",
          "the session is the request's prompt_cache_key",
          json.dumps(x.get("session")))
    check(resp.get("status") == "completed"
          and _text(resp) == "The button overlaps the header."
          and resp["output"][0]["type"] == "reasoning"
          and resp["output"][0]["summary"][0]["text"].endswith(
              "Looking at the placeholder.")
          and resp["output"][0].get("encrypted_content") is None
          and all(codex_item_ok(o) for o in resp["output"])
          and all(k in resp for k in RESPONSE_REQUIRED),
          "the Response: a reasoning item (summary), the message; every "
          "item parses as Codex's ResponseItem; the required fields",
          json.dumps(resp)[:500])


def test_blocking_turns_extend_the_slot():
    T.slots.reset(n=4)
    T.compaction.reset()
    c = RClient("blocking")
    t1 = c.turn([T.reply("", reasoning="Plan: write it.", calls=[T.call(
        "write_file", {"path": "a.py", "content": "x = 1"}, "rb-1")])],
        user="Write a.py with x = 1.")
    r1 = t1["resp"]
    fc = _items(r1, "function_call")
    check(len(fc) == 1 and session_id.of_call_id(fc[0]["call_id"]) ==
          (t1["x"].get("session") or {}).get("carrier")
          and fc[0]["name"] == "write_file"
          and json.loads(fc[0]["arguments"]) == {"path": "a.py",
                                                 "content": "x = 1"}
          and fc[0]["id"].startswith("fc_")
          and _text(r1) == ""
          and [o["type"] for o in r1["output"]] == [
              "reasoning", "function_call"],
          "a client call is a function_call item: call_id is the chat "
          "tool-call id, carrying the carrier of the client's key; no "
          "message item before it (the code check's note went 2026-09-29)",
          json.dumps(r1["output"])[:400])
    x1 = t1["x"]
    check((x1.get("session") or {}).get("source") == "prompt_cache_key"
          and (x1.get("skills") or {}).get("ids") == ["s1"]
          and isinstance(x1.get("tool_turns"), dict)
          and "limit" in x1["tool_turns"]
          and isinstance(x1.get("responses"), dict)
          and x1["responses"].get("items") == {"message": 1},
          "the same turn ran: session from prompt_cache_key, the skills "
          "injection, the tool-turn cap record, plus x_yamadori.responses",
          json.dumps({k: x1.get(k) for k in ("session", "skills",
                                             "tool_turns", "responses")}))
    u = r1.get("usage") or {}
    cu = (x1.get("usage") or {})
    check(u.get("input_tokens") == 100 and u.get("output_tokens") == 10
          and u.get("total_tokens") == 110
          and u.get("input_tokens_details", {}).get("cached_tokens") == 90
          and u.get("input_tokens_details", {}).get("cache_write_tokens") == 0
          and isinstance(u.get("output_tokens_details", {}).get(
              "reasoning_tokens"), int),
          "usage: input/output/total from the final generation, cached and "
          "reasoning token details", json.dumps([u, cu.get("final")]))
    c.tool_output("rb-1", "wrote a.py")
    t2 = c.turn([T.reply("", calls=[T.call(
        "write_file", {"path": "b.py", "content": "y = 2"}, "rb-2")])])
    T._extends(t1, t2, "responses, blocking, turn 2 (a replayed "
               "function_call + function_call_output)")
    c.tool_output("rb-2", "wrote b.py")
    t3 = c.turn([T.reply("Both files are written.")])
    T._extends(t2, t3, "responses, blocking, turn 3")
    check(_text(t3["resp"]) == "Both files are written."
          and t3["resp"]["status"] == "completed"
          and (t3["x"].get("session") or {}).get("id") ==
          (x1.get("session") or {}).get("id"),
          "the final answer is a message item; one session throughout",
          json.dumps(t3["resp"]["output"])[:300])
    up = t3["gens"][0]["request"]["messages"]
    # PAST REASONING IS RESTORED (operator, 2026-09-27): the replayed
    # reasoning summaries are display text and never reach the model; the
    # LEDGER puts back what the slot generated for each turn.
    check([m.get("reasoning_content") or "" for m in up
           if m["role"] == "assistant"] == ["Plan: write it.", ""]
          and ((t3["x"].get("ledger") or {}).get("restored") or {}).get(
              "reasoning") == 1,
          "the replayed reasoning summaries are display text; the model gets "
          "the slot's own reasoning back from the ledger (restore_reasoning)",
          json.dumps([m.get("reasoning_content") for m in up]))


def test_streamed_turns():
    T.slots.reset(n=4)
    T.compaction.reset()
    c = RClient("streamed", stream=True)
    t1 = c.turn([T.reply("", reasoning="I will write the file first.",
                         calls=[T.call("write_file", {
                             "path": "s.py", "content": "s = 1"}, "rs-1")])],
                user="Write s.py.")
    final = check_stream(t1["events"], "streamed turn 1 (reasoning + a call)")
    types = [e["type"] for e in t1["events"]]
    want = ["response.created", "response.in_progress",
            "response.output_item.added",
            "response.reasoning_summary_part.added",
            "response.reasoning_summary_text.delta"]
    check(types[:5] == want
          and "response.reasoning_summary_text.done" in types
          and types.index("response.reasoning_summary_part.done") <
          types.index("response.function_call_arguments.delta")
          and types[-4:] == ["response.function_call_arguments.delta",
                             "response.function_call_arguments.done",
                             "response.output_item.done",
                             "response.completed"],
          "event order: created, in_progress, the reasoning item "
          "(summary part, deltas, done), then the call (added, arguments "
          "delta/done, done), then completed", json.dumps(types))
    fc = _items(final, "function_call")
    check(final.get("status") == "completed" and len(fc) == 1
          and session_id.of_call_id(fc[0]["call_id"])
          and (final.get("x_yamadori") or {}).get("session", {}).get(
              "source") == "prompt_cache_key"
          and (final.get("usage") or {}).get("input_tokens") == 100,
          "the terminal response carries the output, x_yamadori and usage",
          json.dumps({k: final.get(k) for k in ("status", "usage")}))
    c.tool_output("rs-1", "wrote s.py")
    t2 = c.turn([T.reply("Done: s.py holds s = 1. " * 3)])
    final2 = check_stream(t2["events"], "streamed turn 2 (text)")
    T._extends(t1, t2, "responses, streamed, turn 2")
    deltas = [e["delta"] for e in t2["events"]
              if e["type"] == "response.output_text.delta"]
    check(len(deltas) >= 2 and "".join(deltas) == _text(final2)
          == "Done: s.py holds s = 1. " * 3,
          "text streams as output_text deltas that add up to the message",
          json.dumps(deltas)[:200])
    c2 = RClient("streamed-length", stream=True)
    c2.input.append({"role": "user", "content": "go"})
    ev = sse_decode(b"".join(R.stream(iter([
        b'data: {"choices":[{"index":0,"delta":{"content":"cut"},'
        b'"finish_reason":null}]}\n\n',
        b'data: {"choices":[{"index":0,"delta":{},"finish_reason":'
        b'"length"}],"x_yamadori":{"k":1}}\n\n',
        b'data: {"choices":[],"usage":{"prompt_tokens":5,'
        b'"completion_tokens":2,"total_tokens":7}}\n\n',
        b"data: [DONE]\n\n"]), R.to_chat(c2.body())[1])))
    last = check_stream(ev, "a length finish")
    check(ev[-1]["type"] == "response.incomplete"
          and last.get("status") == "incomplete"
          and last.get("incomplete_details") == {"reason":
                                                 "max_output_tokens"}
          and _items(last, "message")[0]["status"] == "incomplete"
          and last.get("usage", {}).get("input_tokens") == 5,
          "a `length` finish is response.incomplete, reason "
          "max_output_tokens", json.dumps(last)[:300])


def test_stream_keepalive_and_failure():
    _chat, ctx = R.to_chat({"input": "x", "stream": True})
    s = R.Stream(ctx)
    out = s.feed(b'data: {"choices":[{"index":0,"delta":{},'
                 b'"finish_reason":null}]}\n\n')
    ev = sse_decode(out)
    check([e["type"] for e in ev] == ["response.created",
                                      "response.in_progress"],
          "a first heartbeat still opens the stream", json.dumps(ev)[:200])
    s.last_event = 0.0
    ev = sse_decode(s.feed(b'data: {"choices":[{"index":0,"delta":{},'
                           b'"finish_reason":null}]}\n\n'))
    check(len(ev) == 1 and ev[0]["type"] == "response.in_progress"
          and ev[0]["response"]["status"] == "in_progress"
          and "tools" not in ev[0]["response"],
          "a heartbeat is a REAL event, response.in_progress with a compact "
          "snapshot: Hermes' codex transport resets its idle timer (12 s) "
          "only on a parsed event, and SSE comments are dropped by the "
          "decoders", json.dumps(ev)[:300])
    # Just after an event (set explicitly: under load the suite's own
    # clock may pass KEEPALIVE_S between two lines).
    s.last_event = time.time()
    ev = sse_decode(s.feed(b'data: {"choices":[{"index":0,"delta":{},'
                           b'"finish_reason":null}]}\n\n'))
    check(ev == [], f"heartbeats are throttled to one per "
          f"{R.KEEPALIVE_S:g} s", json.dumps(ev))
    s.feed(b'data: {"choices":[{"index":0,"delta":{"content":"par"},'
           b'"finish_reason":null}]}\n\n')
    err = api_errors.context_length_exceeded(140000, 132096)
    raw = s.feed(err.sse()) + s.feed(b"data: [DONE]\n\n")
    try:
        ev = sse_decode(raw)
        raised = None
    except SSEError as e:
        ev, raised = [], str(e)
    f = ev[-1] if ev else {}
    check(raised is None and f.get("type") == "response.failed"
          and f["response"]["status"] == "failed"
          and f["response"]["error"]["code"] == "context_length_exceeded"
          and "maximum context length" in f["response"]["error"]["message"]
          and len(ev) == 1,
          "a mid-stream failure is ONE response.failed with Codex's code "
          "(context_length_exceeded -> compaction), not a raw chat error "
          "event (openai-python would raise on that); nothing after it",
          raised or json.dumps(ev)[:300])
    try:
        sse_decode(err.sse())
        chat_raises = False
    except SSEError:
        chat_raises = True
    check(chat_raises, "(the chat form of the same error is what "
          "openai-python raises on: the reader above is faithful)")
    codes = {R._failure_of({"type": "rate_limit_error",
                            "code": "server_busy"})["code"],
             R._failure_of({"type": "service_unavailable_error",
                            "code": "model_unavailable"})["code"],
             R._failure_of({"type": "invalid_request_error",
                            "code": "x"})["code"],
             R._failure_of({"type": "server_error",
                            "code": "upstream_error"})["code"]}
    check(codes == {"rate_limit_exceeded", "server_is_overloaded",
                    "invalid_prompt", "server_error"},
          "our errors map to the codes Codex branches on", json.dumps(
              sorted(codes)))
    u = R.usage_of({"prompt_tokens": 7, "completion_tokens": 3,
                    "total_tokens": 10})
    check(u == {"input_tokens": 7, "input_tokens_details": {
        "cached_tokens": 0, "cache_write_tokens": 0}, "output_tokens": 3,
        "total_tokens": 10},
          "usage without a reasoning count leaves output_tokens_details "
          "out (never guessed)", json.dumps(u))


def _chunk(delta: dict, finish=None, **extra) -> bytes:
    return b"data: " + json.dumps(dict({"choices": [{
        "index": 0, "delta": delta, "finish_reason": finish}]}, **extra)
    ).encode() + b"\n\n"


USAGE = (b'data: {"choices": [], "usage": {"prompt_tokens": 9, '
         b'"completion_tokens": 3, "total_tokens": 12}}\n\n')


def test_output_forms():
    """Custom tool calls come back as custom_tool_call; the `content`
    reasoning mode streams reasoning_text and its echo is read back; an image
    shown mid-message splits the message, and the replay merges it back
    into the one assistant turn the ledger keyed."""
    tools = [{"type": "custom", "name": "apply_patch"}, FLAT_WRITE]
    _c, ctx = R.to_chat({"input": "x", "tools": tools, "stream": True})
    call = {"id": "call_P", "type": "function", "function": {
        "name": "apply_patch",
        "arguments": json.dumps({"input": "*** Begin Patch\n"})}}
    d = {"choices": [{"message": {"content": "", "tool_calls": [call]},
                      "finish_reason": "tool_calls"}], "usage": {}}
    resp = R.of_chat(d, ctx)
    ct = _items(resp, "custom_tool_call")
    check(ct and ct[0]["input"] == "*** Begin Patch\n"
          and ct[0]["call_id"] == "call_P" and codex_item_ok(ct[0]),
          "a call of a custom tool is a custom_tool_call item carrying the "
          "raw input", json.dumps(resp["output"]))
    ev = sse_decode(b"".join(R.stream(iter([
        _chunk({"tool_calls": [dict(call, index=0)]}),
        _chunk({}, "tool_calls", x_yamadori={}),
        USAGE, b"data: [DONE]\n\n"]), ctx)))
    check_stream(ev, "a streamed custom tool call")
    types = [e["type"] for e in ev]
    check("response.custom_tool_call_input.delta" in types
          and "response.custom_tool_call_input.done" in types
          and "response.function_call_arguments.delta" not in types,
          "streamed: custom_tool_call_input.delta/.done (Codex reads the "
          "delta)", json.dumps(types))
    back, _ = R.to_chat({"input": [{"role": "user", "content": "x"}] + [
        e["item"] for e in ev if e["type"] == "response.output_item.done"],
        "tools": tools})
    check(back["messages"][-1]["tool_calls"][0]["function"] == call[
        "function"], "the replayed custom_tool_call renders as the call the "
          "model made", json.dumps(back["messages"][-1]))

    saved = R.REASONING_MODE
    R.REASONING_MODE = "content"
    try:
        _c, ctx = R.to_chat({"input": "x", "stream": True})
        ev = sse_decode(b"".join(R.stream(iter([
            _chunk({"reasoning_content": "step one. "}),
            _chunk({"reasoning_content": "step two."}),
            _chunk({"content": "Answer."}),
            _chunk({}, "stop", x_yamadori={}),
            USAGE, b"data: [DONE]\n\n"]), ctx)))
        check_stream(ev, "content reasoning mode")
        types = [e["type"] for e in ev]
        items = [e["item"] for e in ev
                 if e["type"] == "response.output_item.done"]
        check(types.count("response.reasoning_text.delta") == 2
              and "response.content_part.added" in types[:4]
              and items[0]["content"] == [{"type": "reasoning_text",
                                           "text": "step one. step two."}]
              and items[0]["summary"] == [],
              "YAMADORI_RESPONSES_REASONING=content: reasoning_text events "
              "and a content item", json.dumps(types))
        back, bctx = R.to_chat({"input": [{"role": "user", "content": "x"}]
                                + items + [{"role": "user",
                                            "content": "and?"}]})
        check(back["messages"][1].get("reasoning_content") ==
              "step one. step two."
              and bctx.rec.get("reasoning_restored") == 1,
              "...and a client that echoes it gives the model its reasoning "
              "back (pass-through)", json.dumps(back["messages"][1]))
    finally:
        R.REASONING_MODE = saved

    sha = images.store(tiny_png((1, 2, 3)), {"prompt": "a split test",
                                             "size": "1024x1024"})
    url = images.signed_url(sha, BASE)
    line = f"![a split test]({url})\n\n"
    _c, ctx = R.to_chat({"input": "x", "stream": True, "tools": [
        {"type": "image_generation"}]})
    ev = sse_decode(b"".join(R.stream(iter([
        _chunk({"content": "Preamble. "}),
        _chunk({"content": line}),
        _chunk({"content": "Drawn."}),
        _chunk({}, "stop", x_yamadori={}),
        USAGE, b"data: [DONE]\n\n"]), ctx)))
    final = check_stream(ev, "an image mid-message")
    kinds = [o["type"] for o in final["output"]]
    check(kinds == ["message", "image_generation_call", "message"],
          "an image shown after content closes the message, adds the image "
          "item, and opens a new message", json.dumps(kinds))
    items = [e["item"] for e in ev if e["type"] == "response.output_item.done"]
    back, _ = R.to_chat({"input": [{"role": "user", "content": "draw"}]
                         + items + [{"role": "user", "content": "thanks"}]})
    check([m["role"] for m in back["messages"]] == ["user", "assistant",
                                                     "user"]
          and back["messages"][1]["content"] == "Preamble. " + line
          + "Drawn.",
          "the replay merges message, image, message back into ONE "
          "assistant turn with the exact text the client was sent (the "
          "ledger's key)", json.dumps(back["messages"][1])[:300])
    forged = f"![x]({BASE}/media/{sha}.png?exp=1893456000&sig={'0' * 64})"
    check(R.image_of_line(forged) is None and R.image_of_line(line) == (
        sha, url),
          "only a link whose signature verifies is read as our image")


def test_harness_gaps():
    """docs/HARNESS-RESPONSES.md section 7: Codex's namespace tool, usage
    always numeric, a dropped upstream is a failure, effort values, one
    system message (the shared helper)."""
    import system_roles
    import tiers
    ns = {"type": "namespace", "name": "multi_agent_v1",
          "description": "Sub-agents.", "tools": [
              {"type": "function", "name": "spawn_agent",
               "description": "Start one.", "strict": False,
               "parameters": {"type": "object", "properties": {}}},
              {"type": "function", "name": "write_file",
               "description": "A clash.", "parameters": {"type": "object"}}]}
    body = {"input": [
        {"role": "user", "content": "go"},
        {"type": "function_call", "call_id": "c1", "name": "spawn_agent",
         "namespace": "multi_agent_v1", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": "ok"},
        {"type": "function_call", "call_id": "c2", "name": "write_file",
         "namespace": "multi_agent_v1", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c2", "output": "ok"}],
        "tools": [FLAT_WRITE, ns]}
    chat, ctx = R.to_chat(body)
    names = [t["function"]["name"] for t in chat["tools"]]
    calls = [c["function"]["name"] for m in chat["messages"]
             for c in m.get("tool_calls") or []]
    check(names == ["write_file", "spawn_agent", "multi_agent_v1__write_file"]
          and "Part of multi_agent_v1" in chat["tools"][1]["function"][
              "description"]
          and calls == ["spawn_agent", "multi_agent_v1__write_file"]
          and ctx.rec.get("namespaces") == ["multi_agent_v1"],
          "a namespace tool (Codex's multi_agent_v1, sent on every request) "
          "is flattened: its functions by name, a clash as "
          "<namespace>__<name>; replayed namespaced calls render as the "
          "flat names", json.dumps([names, calls]))
    d = {"choices": [{"message": {"content": "", "tool_calls": [
        {"id": "c3", "type": "function", "function": {
            "name": "multi_agent_v1__write_file", "arguments": "{}"}},
        {"id": "c4", "type": "function", "function": {
            "name": "write_file", "arguments": "{}"}}]},
        "finish_reason": "tool_calls"}]}
    out = _items(R.of_chat(d, ctx), "function_call")
    check(out[0].get("namespace") == "multi_agent_v1"
          and out[0]["name"] == "write_file" and "namespace" not in out[1],
          "a call of a namespaced function comes back with its namespace "
          "and its own name (Codex routes by both)", json.dumps(out))
    resp = R.of_chat({"choices": [{"message": {"content": "x"},
                                   "finish_reason": "stop"}]}, ctx)
    check(resp["usage"] == {"input_tokens": 0, "input_tokens_details": {
        "cached_tokens": 0, "cache_write_tokens": 0}, "output_tokens": 0,
        "total_tokens": 0},
          "usage is always an object of integers (zeros when unknown): the "
          "AI SDK drops a terminal event without it", json.dumps(resp["usage"]))
    resp = R.of_chat({"choices": [{"message": {"content": "part"},
                                   "finish_reason": "incomplete"}]}, ctx)
    check(resp["status"] == "failed"
          and resp["error"]["code"] == "server_error"
          and resp.get("incomplete_details") is None,
          "a dropped upstream is `failed` (server_error), not incomplete: "
          "Codex and Pi read incomplete as an error, Hermes would continue it",
          json.dumps({k: resp[k] for k in ("status", "error")}))
    _c, sctx = R.to_chat({"input": "x", "stream": True})
    ev = sse_decode(b"".join(R.stream(iter([
        _chunk({"content": "part"}), _chunk({}, "incomplete"),
        b"data: [DONE]\n\n"]), sctx)))
    check(ev[-1]["type"] == "response.failed"
          and ev[-1]["response"]["error"]["x_yamadori_code"] ==
          "upstream_dropped",
          "streamed: the same drop ends in response.failed",
          json.dumps(ev[-1])[:300])
    efforts = {}
    for e in ("none", "minimal", "low", "medium", "high", "xhigh", "max"):
        c2, _x = R.to_chat({"input": "x", "reasoning": {"effort": e}})
        try:
            efforts[e] = tiers.resolve(c2)["name"]
        except Exception as ex:                                  # noqa: BLE001
            efforts[e] = f"raised {type(ex).__name__}"
    check(all(not str(v).startswith("raised") for v in efforts.values())
          and efforts["none"] == "minimal" and efforts["max"] == "max",
          "every effort a harness sends (none..xhigh, max) picks a tier",
          json.dumps(efforts))
    same = [{"role": "system", "content": "s"}, {"role": "user",
                                                 "content": "u"}]
    m1, r1 = system_roles.one_system(same)
    m2, r2 = system_roles.one_system([
        {"role": "developer", "content": [{"type": "input_text",
                                           "text": "d"}]},
        {"role": "system", "content": "s"}, {"role": "user", "content": "u"},
        {"role": "developer", "content": "later"}])
    check(m1 is same and r1 == {}
          and m2 == [{"role": "system", "content": "d\n\ns"},
                     {"role": "user", "content": "u"},
                     {"role": "user", "content": "later"}]
          and r2 == {"merged": 2, "developer_as_user": 1},
          "system_roles.one_system (the one helper for both wires): a chat "
          "prompt with one leading system message is untouched; a leading "
          "developer/system run is ONE system message, a later one a user "
          "message", json.dumps([m2, r2]))


def test_a_changed_prompt_cache_key_continues_the_conversation():
    """A NEW prompt_cache_key joins an existing conversation ONLY on
    compaction evidence -- our summary line (coordinator, 2026-09-26).
    Hermes' Responses key hashes (session, instructions, tools), so it
    changes when Hermes rebuilds its prompt at a compaction: the summary it
    keeps carries our line, and the conversation continues as an alias. A
    Codex fork (a new thread key, the FULL history with our call ids, no
    header) is a new conversation, `forked_from` the original. Pi and
    OpenCode forks name themselves (their own key or header)."""
    T.slots.reset(n=4)
    T.compaction.reset()
    c = RClient("rekey", cache_key="pck_hermes_A")
    t1 = c.turn([T.reply("", calls=[T.call(
        "write_file", {"path": "k.py", "content": "k"}, "rk-1")])],
        user="Write k.py.")
    s1 = t1["x"].get("session") or {}
    carrier = s1.get("carrier")
    c.tool_output("rk-1", "wrote k.py")
    t2 = c.turn([T.reply("Done.")])
    check((t2["x"].get("session") or {}).get("carrier") == carrier
          and not (t2["x"].get("session") or {}).get("forked_from"),
          "the same key: the same conversation, never a fork of itself",
          json.dumps(t2["x"].get("session")))
    # (1) Codex fork: a new thread key, the whole history, no header.
    fork = RClient("rekey", cache_key="pck_codex_fork")
    fork.input = copy.deepcopy(c.input) + [{
        "role": "user", "content": "Now try another approach."}]
    fork.account = c.account
    t3 = fork.turn([T.reply("Trying.")])
    s3 = t3["x"].get("session") or {}
    check(s3.get("forked_from") == carrier and not s3.get("aliased_to")
          and s3.get("carrier") != carrier,
          "a fork (new key + our call ids, no summary line) is a NEW "
          "conversation, forked_from the original", json.dumps(s3))
    # (2) Hermes after a compaction: a new key, the summary with our line.
    hermes = RClient("rekey", cache_key="pck_hermes_B")
    hermes.account = c.account
    hermes.input = [{"role": "user", "content": session_id.line(carrier)
                     + "## Summary\nWrote k.py."},
                    {"role": "user", "content": "Continue."}]
    t4 = hermes.turn([T.reply("Continuing.")])
    s4 = t4["x"].get("session") or {}
    check(s4.get("aliased_to") == carrier and s4.get("carrier") == carrier,
          "Hermes' post-compaction shape (new key, our summary line) "
          "continues the conversation as an alias", json.dumps(s4))
    t5 = hermes.turn([T.reply("Still here.")], user="And now?")
    s5 = t5["x"].get("session") or {}
    check(s5.get("carrier") == carrier and s5.get("aliased_to"),
          "the new key stays an alias of the conversation", json.dumps(s5))
    # (3) Pi / OpenCode forks carry their own key or header: unchanged.
    acct = c.account
    msgs = [{"role": "user", "content": session_id.line(carrier) + "s"},
            {"role": "assistant", "content": "", "tool_calls": [T.call(
                "write_file", {}, session_id.call_id(carrier))]}]
    k6, s6 = proxy.session_identity(msgs, acct, header="ses_fork9",
                                    cache_key="pck_opencode_fork")
    check(not s6.get("aliased_to") and not s6.get("forked_from")
          and k6 == session_id.conversation_key(acct, "prompt_cache_key",
                                                "pck_opencode_fork"),
          "with a session header (OpenCode's X-Session-Id) a new key is its "
          "own conversation, even over a summary line", json.dumps(s6))
    k7, s7 = proxy.session_identity(msgs[:1] + [{"role": "user",
                                                 "content": "x"}], acct,
                                    cache_key="pck_pi_session_2")
    _k8, s8 = proxy.session_identity([{"role": "user", "content": "x"}],
                                     acct, cache_key="pck_pi_session_3")
    check(s7.get("aliased_to") == carrier and not s8.get("aliased_to")
          and not s8.get("forked_from"),
          "an unknown key with no carried id is a new conversation; the "
          "summary line alone is the compaction evidence",
          json.dumps([s7, s8]))


def test_a_key_named_compaction_writes_the_line():
    """The evidence must exist: an in-place compaction of a conversation
    the client named by prompt_cache_key opens with the line of its
    CARRIER, so Hermes' continuation under a new key finds it."""
    T.slots.reset(n=4)
    T.compaction.reset()
    c = T.Unforced("medium", "keyed-compaction")
    body = c.body()
    msgs = [c.msgs[0], {"role": "user", "content": "Write hi.py."}]
    T._script[:] = [T.reply("", calls=[T.call("write_file", {
        "path": "hi.py", "content": "print(1)"}, "kc1")])]
    d1 = proxy.complete(dict(body, messages=msgs, prompt_cache_key="kc-A"))
    carrier = d1["x_yamadori"]["session"]["carrier"]
    m1 = d1["choices"][0]["message"]
    msgs += [{"role": "assistant", "content": m1.get("content") or "",
              "tool_calls": m1["tool_calls"]},
             {"role": "tool", "tool_call_id": m1["tool_calls"][0]["id"],
              "content": "wrote hi.py"},
             {"role": "user", "content": "Your task is to create a detailed "
              "summary of the conversation so far."}]
    T._script[:] = [T.reply("## Goal\nWrote hi.py.")]
    d2 = proxy.complete(dict(body, messages=msgs, prompt_cache_key="kc-A"))
    summary = d2["choices"][0]["message"]["content"]
    check(d2["x_yamadori"].get("utility_kind") == "compaction"
          and summary.startswith(session_id.line(carrier)),
          "an in-place compaction of a key-named conversation opens with its "
          "carrier's line", json.dumps(summary[:120]))


APPLY_PATCH = {"type": "custom", "name": "apply_patch",
               "description": "Use the `apply_patch` tool to edit files.",
               "format": {"type": "grammar", "syntax": "lark",
                          "definition": "start: begin_patch hunk+ end_patch"}}


def test_codex_apply_patch_end_to_end():
    """Codex 0.157.1 (captured, docs/HARNESS-CODEX.md): apply_patch is a
    FREEFORM custom tool. The model sees a tool; its call comes back as a
    custom_tool_call with the raw patch, untouched; the replayed call and
    its custom_tool_call_output render as the turn the slot holds. (The code
    check that read the patch went 2026-09-29.)"""
    T.slots.reset(n=4)
    T.compaction.reset()
    c = RClient("patch", tools=[APPLY_PATCH, FLAT_WRITE])
    patch = ("*** Begin Patch\n*** Add File: p.py\n+def f(:\n+    return 1\n"
             "*** End Patch\n")
    t1 = c.turn([T.reply("", calls=[T.call("apply_patch", {"input": patch},
                                           "pa-1")])],
                user="Add p.py.")
    ct = _items(t1["resp"], "custom_tool_call")
    check(ct and ct[0]["input"] == patch and ct[0]["name"] == "apply_patch"
          and "arguments" not in ct[0] and "tool_code" not in t1["x"],
          "the model's apply_patch call is a custom_tool_call with the raw "
          "patch, as written (a syntax error in it is the harness's to meet)",
          json.dumps(ct)[:500])
    c.input.append({"type": "custom_tool_call_output",
                    "call_id": ct[0]["call_id"] if ct else "pa-1",
                    "output": "Success. Updated the following files:\nA p.py"})
    t2 = c.turn([T.reply("Added p.py.")])
    T._extends(t1, t2, "responses: after a custom apply_patch call and its "
               "custom_tool_call_output")


def test_carrier_ids_without_prompt_cache_key():
    T.slots.reset(n=4)
    T.compaction.reset()
    c = RClient("carrier", cache_key=None)
    t1 = c.turn([T.reply("", calls=[T.call(
        "write_file", {"path": "c.py", "content": "c"}, "car-1")])],
        user="Write c.py.")
    cid = _items(t1["resp"], "function_call")[0]["call_id"]
    s1 = t1["x"].get("session") or {}
    check(s1.get("source") == "minted"
          and session_id.of_call_id(cid) == s1.get("id"),
          "no prompt_cache_key: the minted session id rides in the call_id "
          "(the chat carrier)", json.dumps([cid, s1]))
    c.tool_output("car-1", "ok")
    t2 = c.turn([T.reply("Written.")])
    s2 = t2["x"].get("session") or {}
    check(s2.get("source") == "tool_call_id" and s2.get("id") == s1.get("id"),
          "the next request finds its conversation from the call_id echoed "
          "in function_call / function_call_output", json.dumps(s2))
    T._extends(t1, t2, "responses, carrier ids, turn 2")


# ------------------------------------------------------------- the route ----
from starlette.testclient import TestClient  # noqa: E402
import server  # noqa: E402

KEY = accounts.create("responses-tests")
H = {"Authorization": f"Bearer {KEY}"}
CLIENT = TestClient(server.app)


def post(body, path="/v1/responses", raw=None, headers=None, stream=False):
    h = dict(H, **(headers or {}))
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


def _is_error(d, code=None, param=None) -> bool:
    e = (d or {}).get("error")
    return (isinstance(e, dict) and all(k in e for k in (
        "message", "type", "param", "code"))
            and (code is None or e["code"] == code)
            and (param is None or e["param"] == param))


def test_the_route():
    T.slots.reset(n=4)
    body = {"model": "yamadori", "input": "Say hi.",
            "instructions": "Route test.", "prompt_cache_key": "route-1",
            "reasoning": {"effort": "low"}}
    T._script[:] = [T.reply("Hi.")]
    st, _h, d, _ = post(body)
    check(st == 200 and d.get("object") == "response"
          and d.get("id", "").startswith("resp_") and _text(d) == "Hi."
          and d.get("model") == "yamadori" and d.get("store") is False
          and d.get("instructions") == "Route test."
          and isinstance(d.get("x_yamadori"), dict),
          "POST /v1/responses: 200, a Response object", json.dumps(d)[:300])
    T._script[:] = [T.reply("Hello there.", reasoning="Greeting.")]
    st, h, _d, ev = post(dict(body, stream=True, prompt_cache_key="route-2"),
                         stream=True)
    check(st == 200 and h.get("content-type", "").startswith(
        "text/event-stream") and ev,
          "streamed: 200 text/event-stream", f"{st} {h.get('content-type')}")
    if ev:
        final = check_stream(ev, "the route, streamed")
        check(_text(final) == "Hello there." and not any(
            e.get("_done") for e in ev),
              "the terminal event ends the stream (no [DONE])",
              json.dumps([e.get("type") for e in ev]))
    T._script[:] = []                # the fake upstream answers 500
    st, h, d, ev = post(dict(body, stream=True, prompt_cache_key="route-3"),
                        stream=True)
    check(st == 502 and ev is None and _is_error(d, "upstream_error"),
          "a failure before the first byte is a real HTTP status and the "
          "error object, not a 200 stream (E1)", f"{st} {json.dumps(d)[:200]}")
    st, _h, d, _ = post(dict(body, previous_response_id="resp_x"))
    check(st == 400 and _is_error(d, "unsupported_parameter",
                                  "previous_response_id")
          and "stateless" in d["error"]["message"],
          "previous_response_id: 400 unsupported_parameter, saying why",
          json.dumps(d)[:300])
    st, _h, d, _ = post(None, raw=b"{nope")
    check(st == 400 and _is_error(d, "invalid_json"), "bad JSON: 400",
          json.dumps(d)[:200])
    r = CLIENT.post("/v1/responses", json=body)
    check(r.status_code == 401 and _is_error(r.json(), "invalid_api_key"),
          "no key: 401", r.text[:200])
    r = CLIENT.get("/v1/responses", headers=H)
    check(r.status_code == 405 and "POST" in r.headers.get("allow", ""),
          "GET /v1/responses: 405, Allow: POST", f"{r.status_code}")
    st, _h, d, _ = post({"input": "x"}, path="/v1/embeddings")
    check(st == 404 and _is_error(d, "unknown_url")
          and "/v1/responses" in d["error"]["message"],
          "other /v1 paths are still 404, and the message lists the route",
          json.dumps(d)[:300])
    r = CLIENT.get("/", headers={"Accept": "application/json"})
    check("/v1/responses" in (r.json().get("endpoints") or []),
          "the root descriptor lists /v1/responses", r.text[:300])


# -------------------------------------------------------------- images ----

def _real_tools():
    """test_ledger stubs yama_generate_image; these tests run the real tool."""
    proxy.run_our_tool = T._real_run_our_tool


def _stub_tools():
    proxy.run_our_tool = T._run_our_tool


def test_hosted_image_generation():
    _real_tools()
    try:
        _hosted_image(stream=False)
        _hosted_image(stream=True)
    finally:
        _stub_tools()
    st, _h, d, _ = post({"input": "draw", "tools": [
        {"type": "image_generation", "size": "4096x4096"}]})
    check(st == 400 and _is_error(d, "invalid_value", "tools[0].size"),
          "an unsupported image size is a 400 naming tools[0].size",
          json.dumps(d)[:300])


def _hosted_image(stream: bool):
    T.slots.reset(n=4)
    T.compaction.reset()
    tag = "img-stream" if stream else "img-block"
    c = RClient(tag, stream=stream, tools=[
        FLAT_WRITE, {"type": "image_generation", "size": "1536x1024",
                     "quality": "high", "output_format": "png"}])
    n_drawn = len(DRAWN)
    t1 = c.turn([T.reply("", calls=[T.call(
        "yama_generate_image", {"prompt": f"a lighthouse at dusk ({tag})",
                           "size": "1024x1024"}, f"{tag}-g")]),
        T.reply("Here is the lighthouse.")], user="Draw a lighthouse.")
    resp = t1["resp"]
    igs = _items(resp, "image_generation_call")
    drawn = DRAWN[n_drawn:]
    label = "streamed" if stream else "blocking"
    check(len(drawn) == 1 and drawn[0]["size"] == "1536x1024",
          f"{label}: our yama_generate_image ran once, at the CLIENT's size "
          f"(the hosted tool's 1536x1024 wins over the model's 1024x1024)",
          json.dumps(drawn))
    ok_item = False
    if igs:
        ig = igs[0]
        png = base64.b64decode(ig.get("result") or "")
        sha = hashlib.sha256(png).hexdigest()
        ok_item = (ig["status"] == "completed"
                   and png.startswith(images.PNG_MAGIC)
                   and images.png_bytes(sha) == png
                   and ig.get("revised_prompt", "").startswith(
                       "a lighthouse at dusk")
                   and ig.get("size") == "1536x1024"
                   and ig.get("output_format") == "png"
                   and f"/media/{sha}.png?" in (ig.get("x_yamadori") or {})
                   .get("url", "")
                   and codex_item_ok(ig))
    check(len(igs) == 1 and ok_item,
          f"{label}: an image_generation_call item -- the real PNG as "
          f"base64, revised_prompt, size, our signed link as "
          f"x_yamadori.url; Codex parses it",
          json.dumps([{k: (v if k != "result" else f"<{len(v or '')} b64>")
                       for k, v in i.items()} for i in igs])[:500])
    text = _text(resp)
    check("/media/" in text and text.rstrip().endswith(
        "Here is the lighthouse."),
          f"{label}: the image line stays in the message text too (what the "
          f"client stores is what the ledger keys the turn by)", text[:300])
    if stream:
        check_stream(t1["events"], "streamed image turn")
        types = [e["type"] for e in t1["events"]]
        gi = types.index("response.image_generation_call.completed")
        check(types[gi - 1] == "response.output_item.added"
              and t1["events"][gi - 1]["item"]["type"] ==
              "image_generation_call"
              and types[gi + 1] == "response.output_item.done"
              and gi < types.index("response.output_text.delta")
              and not any(t in types for t in (
                  "response.image_generation_call.partial_image",
                  "response.image_generation_call.generating")),
              "streamed: the image item (added, image_generation_call"
              ".completed, done) goes out as soon as the image exists, before "
              "the answer's text; no partial or generating events are "
              "invented", json.dumps(types))
    else:
        order = [o["type"] for o in resp["output"]]
        check(order.index("image_generation_call") < order.index("message"),
              "blocking: the image item comes before the message",
              json.dumps(order))
    c.input.append({"type": "message", "role": "user", "content": [
        {"type": "input_text", "text": "Thanks."}]})
    t2 = c.turn([T.reply("You're welcome.")])
    T._extends(t1, t2, f"responses {label}: after an image turn (the replayed "
               f"image_generation_call ignored, the message kept)")


def test_images_generations_spec():
    _real_tools()
    try:
        st, _h, d, _ = post({"prompt": "a fox", "quality": "high",
                             "style": "vivid", "moderation": "low",
                             "output_compression": 80, "user": "u",
                             "response_format": "b64_json", "model":
                             "gpt-image-1"},
                            path="/v1/images/generations")
        check(st == 200 and d.get("output_format") == "png"
              and d.get("size") == "1024x1024" and d.get("quality") == "high"
              and base64.b64decode(d["data"][0]["b64_json"]).startswith(
                  images.PNG_MAGIC),
              "Images API: quality, style, moderation, output_compression, "
              "user accepted; an unknown model ignored; output_format and "
              "size in the response", json.dumps({k: v for k, v in d.items()
                                                 if k != "data"})[:300])
        for extra, param in (({"output_format": "jpeg"}, "output_format"),
                             ({"background": "transparent"}, "background"),
                             ({"quality": "ultra"}, "quality"),
                             ({"stream": True}, "stream")):
            st, _h, d, _ = post(dict({"prompt": "a fox"}, **extra),
                                path="/v1/images/generations")
            check(st == 400 and _is_error(d, param=param),
                  f"Images API: {json.dumps(extra)} is a 400 naming {param}",
                  f"{st} {json.dumps(d)[:200]}")
        st, _h, d, _ = post(None, path="/v1/images/generations", raw=b"{x")
        check(st == 400 and _is_error(d, "invalid_json"),
              "Images API: bad JSON is the one error object", json.dumps(d))
    finally:
        _stub_tools()


def main() -> int:
    for fn in (test_request_translation, test_request_errors,
               test_codex_request_replay,
               test_blocking_turns_extend_the_slot, test_streamed_turns,
               test_stream_keepalive_and_failure, test_output_forms,
               test_harness_gaps,
               test_a_changed_prompt_cache_key_continues_the_conversation,
               test_a_key_named_compaction_writes_the_line,
               test_codex_apply_patch_end_to_end,
               test_carrier_ids_without_prompt_cache_key, test_the_route,
               test_hosted_image_generation, test_images_generations_spec):
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
