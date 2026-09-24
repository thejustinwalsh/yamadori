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
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOP = {"connection", "keep-alive", "proxy-connection", "transfer-encoding", "te",
       "trailer", "upgrade", "content-length", "host"}

_lock = threading.Lock()


def _shape(body: bytes) -> dict:
    try:
        d = json.loads(body or b"{}")
    except ValueError:
        return {"unparsed_bytes": len(body or b"")}
    msgs = d.get("messages") or []
    last = msgs[-1] if msgs else {}
    c = last.get("content")
    if isinstance(c, list):
        c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
    c = c or ""
    tools = d.get("tools") or []
    names = sorted((t.get("function") or {}).get("name", "") for t in tools)
    return {"model": d.get("model"), "stream": d.get("stream"),
            "reasoning_effort": d.get("reasoning_effort"),
            "max_tokens": d.get("max_tokens") or d.get("max_completion_tokens"),
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
            try:
                up = conn_cls(host, port, timeout=7200)
                up.request(self.command, self.path, body=body or None, headers=headers)
                resp = up.getresponse()
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
    h, p = a.listen.rsplit(":", 1)
    srv = ThreadingHTTPServer((h, int(p)), make_handler(a.upstream, a.out))
    srv.daemon_threads = True
    print(f"relay {a.listen} -> {a.upstream}, rows to {a.out}", flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
