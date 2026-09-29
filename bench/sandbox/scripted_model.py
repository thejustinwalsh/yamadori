#!/usr/bin/env python
"""A SCRIPTED stand-in for the model, for checking a harness's loadout with
no model behind it (docs/HARNESS-SANDBOX.md "Default loadout"). Never the
proxy: it binds 127.0.0.1 and the harness box's one forward points at it.

    python bench/sandbox/scripted_model.py PORT OUT_DIR SCRIPT.json

Speaks both wires: POST /v1/chat/completions (OpenCode, Pi; streamed) and
POST /v1/responses (Codex; streamed). GET /health answers 200.

The script is a list of steps, each {"tool": [names...], "args": {...}}.
A request that offers tools and carries k tool results gets step k as ONE
tool call -- to the first name in `tool` the request offers (so one script
can name `mcp__playwright__browser_navigate` and `playwright_browser_navigate`)
-- and once the steps run out, the text "ok". A step whose tool is not
offered is NOT made: the stand-in answers "ok" and records it (the check
then fails on the missing result). Requests without tools (title and summary
side calls) get "ok". Every request body is saved as req<N>.json, and
steps.jsonl records which step each request got.
"""
from __future__ import annotations

import http.server
import json
import os
import sys

STEPS: list[dict] = []
OUT = "."
N = {"i": 0}


def offered(req: dict) -> list[str]:
    names = []
    for t in req.get("tools") or []:
        if t.get("type") == "namespace":                       # Responses namespaces
            for u in t.get("tools") or []:
                names.append(f"{t.get('name')}::{u.get('name')}")
        elif "function" in t:
            names.append(t["function"].get("name"))
        else:
            names.append(t.get("name"))
    return [n for n in names if n]


def results_so_far(req: dict) -> int:
    if "messages" in req:
        return sum(1 for m in req["messages"] if m.get("role") == "tool")
    return sum(1 for it in req.get("input") or []
               if isinstance(it, dict) and it.get("type") in ("function_call_output",
                                                               "custom_tool_call_output"))


def pick(req: dict) -> tuple[int, str | None, dict | None, list]:
    """(step index, tool name, args) for this request; name None = answer ok."""
    have = offered(req)
    k = results_so_far(req)
    skipped = []
    if not have:
        return k, None, None, skipped
    while k < len(STEPS):
        step = STEPS[k]
        name = next((n for n in step["tool"] if n in have), None)
        if name:
            return k, name, step["args"], skipped
        skipped.append({"step": k, "wanted": step["tool"]})
        return k, None, None, skipped               # not offered: stop here, say so
    return k, None, None, skipped


def chat_body(name: str | None, args: dict | None, i: int) -> bytes:
    base = {"id": f"c{i}", "object": "chat.completion.chunk", "created": 0, "model": "yamadori"}
    if name:
        chunks = [{**base, "choices": [{"index": 0, "delta": {"role": "assistant", "tool_calls": [
            {"index": 0, "id": f"call_step{i}", "type": "function",
             "function": {"name": name, "arguments": json.dumps(args)}}]}, "finish_reason": None}]},
                  {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}]
    else:
        chunks = [{**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "ok"},
                                        "finish_reason": None}]},
                  {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}]
    chunks.append({**base, "choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                                                    "total_tokens": 2}})
    return ("".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n").encode()


USAGE = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2,
         "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}}


def responses_body(name: str | None, args: dict | None, i: int) -> bytes:
    rid = f"resp_{i}"
    if name:
        ns, _, fn = name.rpartition("::")
        item = {"type": "function_call", "id": f"fc_{i}", "call_id": f"call_step{i}",
                "status": "completed", "name": fn, "arguments": json.dumps(args)}
        if ns:
            item["namespace"] = ns
    else:
        item = {"type": "message", "role": "assistant", "id": f"msg_{i}", "status": "completed",
                "content": [{"type": "output_text", "text": "ok", "annotations": []}]}
    ev = [{"type": "response.created", "response": {"id": rid}},
          {"type": "response.output_item.added", "output_index": 0, "item": item},
          {"type": "response.output_item.done", "output_index": 0, "item": item},
          {"type": "response.completed", "response": {"id": rid, "usage": USAGE}}]
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in ev).encode()


class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            self._send(200, json.dumps({"object": "list", "data": [
                {"id": "yamadori", "object": "model", "owned_by": "scripted"}]}).encode(),
                "application/json")
            return
        self._send(200 if self.path == "/health" else 404, b'{"ok":true,"scripted":true}',
                   "application/json")

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        N["i"] += 1
        i = N["i"]
        with open(os.path.join(OUT, f"req{i}.json"), "wb") as f:
            f.write(body)
        try:
            req = json.loads(body)
        except ValueError:
            self._send(400, b'{"error":{"message":"bad json"}}', "application/json")
            return
        k, name, args, skipped = pick(req)
        with open(os.path.join(OUT, "steps.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps({"req": i, "path": self.path, "results_so_far": k, "tool": name,
                                "skipped": skipped, "offered": offered(req)}) + "\n")
        if self.path.endswith("/responses"):
            self._send(200, responses_body(name, args, i), "text/event-stream")
        else:
            self._send(200, chat_body(name, args, i), "text/event-stream")

    def log_message(self, *a):
        pass


def main() -> None:
    global OUT
    port, OUT, script = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    STEPS[:] = json.load(open(script, encoding="utf-8"))
    os.makedirs(OUT, exist_ok=True)
    http.server.ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()


if __name__ == "__main__":
    main()
