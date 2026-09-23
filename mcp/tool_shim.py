#!/usr/bin/env python
"""OpenAI-compatible shim that makes tool calling actually work.

THE PROBLEM
-----------
llama.cpp's native `tools` parameter is unusable with this model family.
Measured, 10 runs each, identical prompt:

    ternary PTQ1_0  + llama.cpp tools param      0/10 clean
    Qwen3.8 Q4_K_M  + llama.cpp tools param      0/10 clean
    Qwen3.8 Q4_K_M  + STOCK llama.cpp build      0/10 clean
    same model, tools rendered into the prompt   5/5  clean

llama.cpp reports `Chat format: peg-native` and applies a PEG/grammar to
constrain output. That grammar fights this model's XML tool syntax, and
decoding degenerates into runs of '/' -- sometimes producing garbage
arguments, sometimes crashing the sampler outright with
"Unexpected empty grammar stack after accepting piece: /".

The model itself is faultless: given the same tools described in the prompt
and no grammar, it emits well-formed calls every time.

THE FIX
-------
Sit in front of llama-swap and do what textgen does internally:

    1. lift `tools` out of the request and render them into the prompt
    2. forward WITHOUT `tools`, so llama.cpp applies no grammar
    3. parse the model's XML back into OpenAI `tool_calls`

Requests with no `tools` are proxied untouched, streaming included.
"""
from __future__ import annotations

import json
import os
import re
import secrets
import sys
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

UPSTREAM = os.environ.get("SHIM_UPSTREAM", "http://127.0.0.1:1234")
HOST = os.environ.get("SHIM_HOST", "0.0.0.0")
PORT = int(os.environ.get("SHIM_PORT", "1233"))

# Mirrors the format in the model's own chat template. Keep it byte-similar:
# the model was trained on this shape and drifts if you paraphrase it.
TOOL_PREAMBLE = """# Tools

You have access to the following functions:

<tools>
{tools}
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>"""

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
FUNC_RE = re.compile(r"<function=([^>]+)>")
PARAM_RE = re.compile(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", re.DOTALL)


def render_tools(tools: list) -> str:
    # MEASURED, not guessed. Two renderings, same prompt, same model:
    #   full OpenAI objects + compact separators  -> degenerates to '/' from
    #                                                token 1, 0/12 usable
    #   flattened name/description/parameters,
    #   default json spacing                      -> 5/5 well formed
    # The model's own chat template emits the full wrapper object, but
    # reproducing that exactly is reliably WORSE here. Keep the flat form.
    lines = []
    for t in tools:
        fn = t.get("function", t)
        lines.append(json.dumps({
            "name": fn.get("name"),
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters", {}),
        }))
    return TOOL_PREAMBLE.format(tools="\n".join(lines))


def parse_tool_calls(text: str) -> tuple[list, str]:
    """Return (tool_calls, leading_content)."""
    calls, first = [], None
    for m in TOOL_CALL_RE.finditer(text):
        inner = m.group(1)
        fm = FUNC_RE.search(inner)
        if not fm:
            continue
        args = {}
        for pm in PARAM_RE.finditer(inner):
            key, val = pm.group(1).strip(), pm.group(2).strip()
            # numbers/bools/objects arrive as bare text; recover real types
            try:
                args[key] = json.loads(val)
            except (json.JSONDecodeError, ValueError):
                args[key] = val
        if first is None:
            first = m.start()
        calls.append({
            "id": "call_" + secrets.token_hex(8),
            "type": "function",
            "function": {"name": fm.group(1).strip(),
                         "arguments": json.dumps(args)},
        })
    content = text[:first].strip() if first is not None else text
    return calls, content


def post_upstream(path: str, payload: dict, timeout: int = 3600) -> dict:
    req = urllib.request.Request(f"{UPSTREAM}{path}", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _raw_proxy(self, method: str, path: str, body: bytes) -> None:
        """Byte-for-byte passthrough, preserving streaming."""
        req = urllib.request.Request(f"{UPSTREAM}{path}", data=body or None, method=method)
        for h in ("Content-Type", "Authorization", "Accept"):
            if self.headers.get(h):
                req.add_header(h, self.headers[h])
        try:
            with urllib.request.urlopen(req, timeout=3600) as up:
                self.send_response(up.status)
                for k, v in up.headers.items():
                    if k.lower() in ("transfer-encoding", "connection", "content-length"):
                        continue
                    self.send_header(k, v)
                self.send_header("Connection", "close")
                self.end_headers()
                while True:
                    chunk = up.read(8192)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()          # never buffer SSE
        except urllib.error.HTTPError as e:
            data = e.read()
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:                   # noqa: BLE001
            self._json(502, {"error": {"message": f"upstream: {e}", "type": "proxy_error"}})

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):                            # noqa: N802
        self._raw_proxy("GET", urlparse(self.path).path, b"")

    def do_POST(self):                           # noqa: N802
        path = urlparse(self.path).path
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""

        if path != "/v1/chat/completions":
            return self._raw_proxy("POST", path, raw)

        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            return self._json(400, {"error": {"message": "invalid JSON"}})

        tools = body.get("tools")
        if not tools:
            return self._raw_proxy("POST", path, raw)   # nothing to translate

        # --- translate -----------------------------------------------------
        body.pop("tools", None)
        # tool_choice has no meaning once the grammar is gone; the preamble
        # already instructs the model when to call.
        body.pop("tool_choice", None)
        # Tool turns must not stream: we cannot emit OpenAI tool_calls deltas
        # until the whole XML block has arrived.
        streaming = bool(body.pop("stream", False))

        preamble = render_tools(tools)
        msgs = list(body.get("messages") or [])
        if msgs and msgs[0].get("role") == "system":
            msgs[0] = {**msgs[0], "content": preamble + "\n\n" + (msgs[0].get("content") or "")}
        else:
            msgs.insert(0, {"role": "system", "content": preamble})
        body["messages"] = msgs

        # Thinking must be off: with it on the model spends its whole budget
        # reasoning and never emits a call (measured: 2000 tokens, 0 calls).
        kw = dict(body.get("chat_template_kwargs") or {})
        kw.setdefault("enable_thinking", False)
        body["chat_template_kwargs"] = kw

        try:
            up = post_upstream(path, body)
        except urllib.error.HTTPError as e:
            return self._json(e.code, json.loads(e.read() or b'{"error":{}}'))
        except Exception as e:                   # noqa: BLE001
            return self._json(502, {"error": {"message": f"upstream: {e}"}})

        try:
            choice = up["choices"][0]
            text = choice["message"].get("content") or ""
        except (KeyError, IndexError):
            return self._json(502, {"error": {"message": "malformed upstream response",
                                              "upstream": up}})

        calls, content = parse_tool_calls(text)
        if calls:
            choice["message"]["content"] = content or None
            choice["message"]["tool_calls"] = calls
            choice["finish_reason"] = "tool_calls"

        if streaming:
            # Emit the finished result as a single SSE frame so streaming
            # clients still work, without pretending to stream a tool call.
            payload = json.dumps(up).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(b"data: " + payload + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return

        self._json(200, up)

    def log_message(self, fmt, *a):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % a))


def main() -> None:
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"tool shim on http://{HOST}:{PORT} -> {UPSTREAM}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
