#!/usr/bin/env python3
"""A recording pass-through between a harness and the proxy. Stdlib only.

    python relay.py --listen 127.0.0.1:18234 --upstream http://127.0.0.1:1234 \
        --out C:\\Users\\jwals\\octo\\logs\\<id>\\relay.jsonl

WHY IT EXISTS

`x_yamadori` -- every decision the proxy took for a request (tier, route,
tool_code repairs, deep thinking, fan-out, compaction, cache, tool turns,
energy) -- rides on the RESPONSE and nowhere else: the corpus keeps no
account and no x_yamadori, and logs/proxy.out.log has no timestamps and is
truncated on restart. A harness does not keep it either. So the grader could
not tie a Hermes run to what the stack did for it. This relay sits on the
harness's side of the wire and writes one row per request.

IT CHANGES NOTHING. Bytes in are forwarded unchanged (Host is rewritten to
the upstream's, as any forward proxy must), bytes out are forwarded unchanged
as they arrive (streamed responses stay streamed; chunk boundaries may
differ, which HTTP permits). It is not "testing around the stack" (AGENTS.md):
every request still goes through :1234. It never logs headers (the key is in
one), and keeps only request SHAPE plus the last message's head, never the
conversation: the harness's own session store has the transcript.

A row: t0 / t_first_byte / t_end (epoch), method, path, status, request
{model, stream, reasoning_effort, max_tokens, n_messages, last_role,
last_chars, last_head (300 chars), n_tools, tool_names, tool_names_hash}, response
{finish_reason, usage, tool_calls: [names], content_chars, reasoning_chars,
x_yamadori (verbatim), bytes, error}.

BOTH WIRES (2026-09-26). A Responses request (POST /v1/responses: `input`
items, flat tools) is shaped the same way, `wire: "responses"`; its stream's
`x_yamadori`, usage and finish ride inside the terminal event's `response`
(response.completed / .incomplete / .failed) or the blocking Response
object, and are recorded in the chat row's terms: usage as prompt_tokens /
completion_tokens / total_tokens (+ cached and reasoning details), finish as
`tool_calls` (a function call in the output), `stop`, `length` (incomplete,
max_output_tokens) or `error` (failed), tool calls from output_item.done.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import select
import socket
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOP = {"connection", "keep-alive", "proxy-connection", "transfer-encoding", "te",
       "trailer", "upgrade", "content-length", "host"}

_lock = threading.Lock()


def _responses_messages(d: dict) -> list:
    """A Responses request's input, as the roles and texts _shape reads."""
    raw = d.get("input")
    if isinstance(raw, str):
        return [{"role": "user", "content": raw}]
    out = []
    for it in raw if isinstance(raw, list) else []:
        if not isinstance(it, dict):
            continue
        kind = it.get("type") or ("message" if "role" in it else None)
        if kind == "message":
            c = it.get("content")
            if isinstance(c, list):
                c = " ".join(x.get("text", "") for x in c
                             if isinstance(x, dict))
            out.append({"role": it.get("role"), "content": c or ""})
        elif kind in ("function_call", "custom_tool_call"):
            if out and out[-1].get("role") == "assistant":
                out[-1].setdefault("tool_calls", []).append(it)
            else:
                out.append({"role": "assistant", "content": "",
                            "tool_calls": [it]})
        elif kind in ("function_call_output", "custom_tool_call_output"):
            o = it.get("output")
            if isinstance(o, list):
                o = " ".join(x.get("text", "") for x in o
                             if isinstance(x, dict))
            out.append({"role": "tool", "content": o or ""})
    return out


def _shape(body: bytes) -> dict:
    try:
        d = json.loads(body or b"{}")
    except ValueError:
        return {"unparsed_bytes": len(body or b"")}
    wire = "responses" if "input" in d and "messages" not in d else "chat"
    msgs = d.get("messages") or [] if wire == "chat" else         _responses_messages(d)
    last = msgs[-1] if msgs else {}
    c = last.get("content")
    if isinstance(c, list):
        c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
    c = c or ""
    tools = d.get("tools") or []
    names = sorted((t.get("function") or {}).get("name", "")
                   or t.get("name") or t.get("type") or "" for t in tools
                   if isinstance(t, dict))
    effort = d.get("reasoning_effort") or (
        (d.get("reasoning") or {}).get("effort")
        if isinstance(d.get("reasoning"), dict) else None)
    return {"wire": wire, "model": d.get("model"), "stream": d.get("stream"),
            "reasoning_effort": effort,
            "max_tokens": d.get("max_tokens") or d.get("max_completion_tokens")
            or d.get("max_output_tokens"),
            "n_messages": len(msgs), "last_role": last.get("role"),
            "last_chars": len(c), "last_head": c[:300],
            "last_tool_calls": len(last.get("tool_calls") or []),
            "n_tools": len(tools), "tool_names": names,
            "tool_names_hash": hashlib.sha256("\n".join(names).encode()).hexdigest()[:12],
            "bytes": len(body or b"")}


class _Collect:
    """Reads the response as it passes: SSE events or one JSON body."""

    def __init__(self, sse: bool):
        self.sse = sse
        self.buf = b""
        self.raw = bytearray()
        self.out = {"finish_reason": None, "usage": None, "tool_calls": [],
                    "content_chars": 0, "reasoning_chars": 0, "x_yamadori": None,
                    "bytes": 0, "sse_events": 0, "error": None}

    def feed(self, data: bytes) -> None:
        self.out["bytes"] += len(data)
        if not self.sse:
            self.raw.extend(data)
            return
        self.buf += data
        while b"\n\n" in self.buf:
            ev, self.buf = self.buf.split(b"\n\n", 1)
            for line in ev.split(b"\n"):
                if line.startswith(b"data:"):
                    self._event(line[5:].strip())

    def _event(self, payload: bytes) -> None:
        if payload == b"[DONE]" or not payload:
            return
        try:
            d = json.loads(payload)
        except ValueError:
            return
        self.out["sse_events"] += 1
        self._take(d, stream=True)

    def _take(self, d: dict, stream: bool) -> None:
        if not isinstance(d, dict):
            return
        if isinstance(d.get("type"), str) or d.get("object") == "response":
            self._take_responses(d)
            return
        if d.get("x_yamadori") is not None:
            self.out["x_yamadori"] = d["x_yamadori"]
        if d.get("usage"):
            self.out["usage"] = d["usage"]
        if d.get("error"):
            self.out["error"] = d["error"]
        for ch in d.get("choices") or []:
            if ch.get("finish_reason"):
                self.out["finish_reason"] = ch["finish_reason"]
            m = ch.get("delta" if stream else "message") or {}
            self.out["content_chars"] += len(m.get("content") or "")
            self.out["reasoning_chars"] += len(m.get("reasoning_content") or "")
            for tc in m.get("tool_calls") or []:
                name = (tc.get("function") or {}).get("name")
                if name:
                    self.out["tool_calls"].append(name)

    def _take_responses(self, d: dict) -> None:
        """One Responses event, or a blocking Response object."""
        t = d.get("type") or "response.blocking"
        if t == "response.output_text.delta":
            self.out["content_chars"] += len(d.get("delta") or "")
        elif t in ("response.reasoning_summary_text.delta",
                   "response.reasoning_text.delta"):
            self.out["reasoning_chars"] += len(d.get("delta") or "")
        elif t == "error":
            self.out["error"] = {"code": d.get("code"),
                                 "message": d.get("message")}
        if t not in ("response.completed", "response.incomplete",
                     "response.failed", "response.blocking"):
            return
        r = d.get("response") if t != "response.blocking" else d
        r = r if isinstance(r, dict) else {}
        self.out["wire"] = "responses"
        if r.get("x_yamadori") is not None:
            self.out["x_yamadori"] = r["x_yamadori"]
        u = r.get("usage")
        if isinstance(u, dict):
            self.out["usage"] = {
                "prompt_tokens": u.get("input_tokens"),
                "completion_tokens": u.get("output_tokens"),
                "total_tokens": u.get("total_tokens"),
                "prompt_tokens_details": {"cached_tokens": (
                    u.get("input_tokens_details") or {}).get("cached_tokens")},
                "completion_tokens_details": {"reasoning_tokens": (
                    u.get("output_tokens_details") or {}).get(
                        "reasoning_tokens")}}
        out = [o for o in r.get("output") or [] if isinstance(o, dict)]
        calls = [o.get("name") for o in out if o.get("type") in (
            "function_call", "custom_tool_call") and o.get("name")]
        self.out["tool_calls"] = calls
        if t == "response.blocking":
            for o in out:
                if o.get("type") == "message":
                    self.out["content_chars"] += sum(
                        len(c.get("text") or "") for c in o.get("content")
                        or [] if isinstance(c, dict))
        status = r.get("status")
        if status == "failed":
            self.out["finish_reason"] = "error"
            self.out["error"] = r.get("error")
        elif status == "incomplete":
            reason = (r.get("incomplete_details") or {}).get("reason")
            self.out["finish_reason"] = "length" if reason ==                 "max_output_tokens" else (reason or "incomplete")
        else:
            self.out["finish_reason"] = "tool_calls" if calls else "stop"

    def done(self) -> dict:
        if not self.sse and self.raw:
            try:
                self._take(json.loads(bytes(self.raw)), stream=False)
            except ValueError:
                self.out["error"] = self.out["error"] or "response not JSON"
        return self.out


def make_handler(upstream: str, out_path: str):
    u = urllib.parse.urlparse(upstream)
    host, port = u.hostname, u.port or (443 if u.scheme == "https" else 80)
    conn_cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):          # quiet; the jsonl is the log
            pass

        def _relay(self):
            t0 = time.time()
            n = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(n) if n else b""
            row = {"t0": t0, "method": self.command, "path": self.path}
            if body:
                row["request"] = _shape(body)
            headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
            headers["Host"] = f"{host}:{port}"
            if body:
                headers["Content-Length"] = str(len(body))
            col = None
            up = None
            # WATCH THE CLIENT WHILE WAITING (2026-09-25, SELF-IMPROVEMENT-LOG
            # #44): the relay used to notice a hang-up only when it wrote, so
            # during a silent upstream stretch Hermes' stale-stream abort left
            # both of its attempts generating upstream for 23 and 40 min and
            # the proxy 429'd the third. A readable client socket that peeks
            # EOF means the client closed: close the upstream so the proxy's
            # disconnect cancel reaches the turn.
            stop = threading.Event()

            held = {}

            def _close_upstream():
                # shutdown() first: close() is deferred while the response's
                # reader still holds the socket, so no FIN would go out. The
                # socket is the connection's, or -- once http.client hands it
                # to a response that will close (HTTP/1.0) -- the response's.
                socks = []
                if up is not None and up.sock is not None:
                    socks.append(up.sock)
                raw = getattr(getattr(held.get("resp"), "fp", None), "raw", None)
                if getattr(raw, "_sock", None) is not None:
                    socks.append(raw._sock)
                for sk in socks:
                    try:
                        sk.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                try:
                    if up is not None:
                        up.close()
                except Exception:                            # noqa: BLE001
                    pass

            def _watch_client():
                sock = self.connection
                while not stop.is_set():
                    try:
                        readable, _, _ = select.select([sock], [], [], 1.0)
                    except (OSError, ValueError):
                        return
                    if not readable:
                        continue
                    try:
                        gone = not sock.recv(1, socket.MSG_PEEK)
                    except OSError:
                        gone = True
                    if not gone:
                        return        # the client sent data: not ours to judge
                    row["client_gone"] = "closed while waiting on the proxy"
                    _close_upstream()
                    return
            try:
                up = conn_cls(host, port, timeout=7200)
                up.request(self.command, self.path, body=body or None, headers=headers)
                threading.Thread(target=_watch_client, daemon=True).start()
                resp = up.getresponse()
                held["resp"] = resp
                row["status"] = resp.status
                ctype = resp.getheader("Content-Type", "")
                col = _Collect(sse="text/event-stream" in ctype)
                self.send_response(resp.status, resp.reason)
                for k, v in resp.getheaders():
                    if k.lower() not in HOP:
                        self.send_header(k, v)
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                first = True
                while True:
                    data = resp.read1(65536)
                    if not data:
                        break
                    if first:
                        row["t_first_byte"] = time.time()
                        first = False
                    col.feed(data)
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
                up.close()
            except (BrokenPipeError, ConnectionResetError) as e:
                row["client_gone"] = type(e).__name__
            except Exception as e:                           # noqa: BLE001
                row["relay_error"] = f"{type(e).__name__}: {e}"
                try:
                    if "status" not in row:
                        msg = json.dumps({"error": {"message": f"relay: {e}"}}).encode()
                        self.send_response(502)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(msg)))
                        self.end_headers()
                        self.wfile.write(msg)
                except Exception:                            # noqa: BLE001
                    pass
            finally:
                stop.set()
                _close_upstream()        # never leave an upstream running
            row["t_end"] = time.time()
            if col is not None:
                row["response"] = col.done()
            with _lock, open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

        do_GET = do_POST = do_PUT = do_DELETE = _relay

    return H


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", default="127.0.0.1:18234")
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    # --listen HOST:PORT[,HOST:PORT...]: the same relay on each address (the
    # WSL engine's containers reach the host on the WSL NAT address only)
    handler = make_handler(a.upstream, a.out)
    servers = []
    for one in a.listen.split(","):
        h, p = one.strip().rsplit(":", 1)
        srv = ThreadingHTTPServer((h, int(p)), handler)
        srv.daemon_threads = True
        servers.append(srv)
    for srv in servers[1:]:
        threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"relay {a.listen} -> {a.upstream}, rows to {a.out}", flush=True)
    servers[0].serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
