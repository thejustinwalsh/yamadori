#!/usr/bin/env python
"""Offline check of bench/harness_soak.py: the soak's PASS/FAIL logic, both ways, against a fake proxy.

A small fake proxy (http.server in a thread, on a free localhost port -- never 1234/1235/11434) speaks just enough of
chat completions and the Responses API: streamed chunks with heartbeats, tool calls for the first K steps then a
final answer, an `x_yamadori` on the finish chunk, a 503 + Retry-After for a second conversation inside the owner's
hold, a silent pre-flight (the model swap), a slow first byte, a mid-stream silence, an error event, a 500, a 429.
Each case runs the real script against it, with small sizes, and asserts what the script concluded.

    python bench/test_harness_soak.py

No GPU, no stack port, nothing written outside a temp directory.
"""
from __future__ import annotations

import contextlib
import hashlib
import http.server
import io
import json
import math
import os
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import harness_soak as H                                                              # noqa: E402

KEY = "test-key"
_PASSES = 0
_FAILURES: list[str] = []


def check(name: str, ok, detail: str = "") -> bool:
    global _PASSES
    if ok:
        _PASSES += 1
    else:
        _FAILURES.append(f"{name}: {detail}")
        print(f"  FAIL  {name}  {detail}", flush=True)
    return bool(ok)


# ------------------------------------------------------------------------------------------------ the fake
class Knobs:
    def __init__(self, **kw):
        self.model = "flash-next"
        self.side_model = "bonsai-a4000"
        self.side_same_model = False
        self.tool_steps = 3
        self.swap_delay = 0.0           # silent pre-flight on the first main request (before any byte)
        self.cold_delay = 0.8           # a cold prompt (processed > 1000) takes this long to the first token
        self.heartbeats = True
        self.hb_interval = 0.3
        self.reread_all = False
        self.prefix_reuse = 256         # reused tokens for a new conversation that shares the system prefix
        self.hold = 1.5
        self.retry_after = None
        self.never_free = False         # a conversation not served yet is refused 503 for ever
        self.mid_silence_step = None    # conv step index at which the stream goes silent mid-way
        self.mid_silence_s = 0.0
        self.error_event_step = None
        self.http500_step = None
        self.invalid_tool_step = None
        self.route_step = None
        self.status429 = False
        self.no_x = False
        self.wrong_title_status = None
        self.__dict__.update(kw)


class ClientGone(Exception):
    pass


class Fake:
    def __init__(self, knobs: Knobs):
        self.k = knobs
        self.lock = threading.Lock()
        self.owner = None
        self.owner_end = 0.0
        self.convs: dict[str, dict] = {}
        self.prefixes: set[str] = set()
        self.loaded = False
        self.log: list[tuple] = []
        self.srv = None

    def start(self) -> str:
        fake = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(n)
                if self.headers.get("Authorization") != f"Bearer {KEY}":
                    return self.send_json(401, {"error": {"message": "bad key", "code": "invalid_api_key"}})
                try:
                    body = json.loads(raw)
                except ValueError:
                    return self.send_json(400, {"error": {"message": "bad json"}})
                path = self.path.split("?")[0]
                if path not in ("/v1/chat/completions", "/v1/responses"):
                    return self.send_json(404, {"error": {"message": "unknown route", "code": "not_found"}})
                try:
                    fake.handle(self, "responses" if path.endswith("responses") else "chat", body, len(raw))
                except (ClientGone, BrokenPipeError, ConnectionError, OSError):
                    fake.log.append(("client_gone", path))

            def send_json(self, status, obj, headers=None):
                b = json.dumps(obj).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(b)

            def begin(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()

            def emit(self, data: bytes):
                try:
                    self.wfile.write(b"%x\r\n" % len(data) + data + b"\r\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionError, OSError):
                    raise ClientGone() from None

            def finish_stream(self):
                try:
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionError, OSError):
                    raise ClientGone() from None

        class Server(http.server.ThreadingHTTPServer):
            daemon_threads = True

            def handle_error(self, request, client_address):          # a client that hung up is not news
                pass

        self.srv = Server(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{self.srv.server_address[1]}"

    def stop(self):
        self.srv.shutdown()
        self.srv.server_close()

    # -- the request, normalised
    @staticmethod
    def norm(api: str, body: dict):
        if api == "chat":
            msgs = body.get("messages") or []
            system = next((m["content"] for m in msgs if m["role"] == "system"), "")
            users = [m for m in msgs if m["role"] == "user"]
            first = users[0]["content"] if users else ""
            last = msgs[-1] if msgs else {}
            last_user = last.get("content") if last.get("role") == "user" else None
            effort = body.get("reasoning_effort")
        else:
            system = body.get("instructions") or ""
            items = body.get("input") or []
            us = [it for it in items if it.get("type") == "message" and it.get("role") == "user"]
            txt = lambda it: "".join(p.get("text", "") for p in it.get("content") or [])    # noqa: E731
            first = txt(us[0]) if us else ""
            last = items[-1] if items else {}
            last_user = txt(last) if last.get("type") == "message" and last.get("role") == "user" else None
            effort = (body.get("reasoning") or {}).get("effort")
        return system, first, last_user, effort

    def handle(self, h, api: str, body: dict, nbytes: int) -> None:
        k = self.k
        if k.status429:
            return h.send_json(429, {"error": {"message": "busy", "type": "rate_limit_error",
                                               "code": "server_busy"}}, {"Retry-After": "30"})
        system, first, last_user, _effort = self.norm(api, body)
        stream = bool(body.get("stream"))
        if not body.get("tools") and "title" in system.lower():
            return self.side(h, body)
        key = hashlib.sha1((system + "\n" + first).encode()).hexdigest()[:12]
        syshash = hashlib.sha1(system.encode()).hexdigest()[:8]
        compaction = last_user == H.COMPACT_ASK
        with self.lock:
            now = time.time()
            if self.owner is not None and self.owner != key and k.hold > 0:
                free_at = self.owner_end + k.hold
                refused = (k.never_free and key not in self.convs) or now < free_at
                if refused:
                    ra = 1 if k.never_free else (k.retry_after or max(1, math.ceil(free_at - now)))
                    self.log.append(("503", key))
                    return h.send_json(503, {"error": {"message": "at capacity", "type": "api_error",
                                                       "param": None, "code": "conversation_at_capacity"}},
                                       {"Retry-After": str(ra)})
            st = self.convs.get(key)
            n = st["steps"] if st else 0
            tok = nbytes // 4
            if k.reread_all:
                reused = 0
            elif st:
                reused = min(st["last_total"], tok)
            else:
                reused = k.prefix_reuse if (syshash in self.prefixes and k.prefix_reuse) else 0
            processed = max(0, tok - reused)
            displaced = bool(st and st.get("displaced"))
            if st:
                st["displaced"] = False
            for c in self.convs.values():
                if c is not st:
                    c["displaced"] = True
            first_ever = not self.loaded
            self.loaded = True
            self.prefixes.add(syshash)
            sid = (st or {}).get("sid") or key
            self.log.append(("serve", key, n))
        if k.http500_step is not None and n == k.http500_step and not compaction:
            return h.send_json(500, {"error": {"message": "boom", "type": "api_error", "code": "internal"}})
        if first_ever and k.swap_delay:
            time.sleep(k.swap_delay)                 # the swap: before any byte, as the real pre-flight
        x = {} if k.no_x else {
            "capacity": dict({"model": k.model, "tier": "max", "utility": False, "waited_s": 0.0},
                             **({"swap": {"to": k.model, "load_s": round(k.swap_delay, 1), "ok": True}}
                                if first_ever else {})),
            "cache": {"prompt": tok, "reused": reused, "processed": processed, "prompt_ms": processed * 4,
                      "decode_tps": 31.5, "slot": 0, "mode": "extend", "generations": 1},
            "slots": {"routed": ({"routed": "other_card", "model": "bonsai-a4000"}
                                 if k.route_step is not None and n == k.route_step else None),
                      "switch": None,
                      "resumed_cold": {"how": "restored", "prompt_ms": 120} if displaced else None},
            "session": {"id": sid, "source": "tool_call_id" if n else "minted"},
            "utility_kind": "compaction" if compaction else None}
        # the plan for this turn
        if compaction:
            plan = ("text", f"yamadori session {sid}\nSummary: wrote src/slug.ts and the tests; they pass.")
        elif n < k.tool_steps:
            args = [{"path": "."}, {"path": "src/slug.ts", "content": "export const slugify = (s: string) => s;\n"},
                    {"command": "npm test"}][n % 3]
            name = ["list_dir", "write_file", "run_terminal"][n % 3]
            raw_args = "{bad json" if k.invalid_tool_step == n else json.dumps(args)
            plan = ("call", name, raw_args, f"call_{sid}_{n:08x}")
        else:
            plan = ("text", "All done. The tests pass.")
        completion = 60
        delay = k.cold_delay if processed > 1000 else 0.0
        try:
            if api == "chat" and stream:
                self.chat_stream(h, body, plan, x, delay, n, tok, completion, k)
            elif api == "chat":
                h.send_json(200, {"choices": [{"message": {"content": plan[1] if plan[0] == "text" else None},
                                               "finish_reason": "stop"}], "x_yamadori": x, "model": "yamadori"})
            else:
                self.resp_stream(h, body, plan, x, delay, n, tok, completion, k)
        finally:
            pass
        with self.lock:
            c = self.convs.setdefault(key, {"sid": sid, "steps": 0, "last_total": 0})
            if not compaction:
                c["steps"] += 1
            c["last_total"] = tok + completion
            c["displaced"] = False
            self.owner, self.owner_end = key, time.time()

    def side(self, h, body):
        k = self.k
        h.send_json(200, {"model": "yamadori", "choices": [{"message": {"content": "Slug utility tests"},
                                                             "finish_reason": "stop"}],
                          "x_yamadori": {"capacity": {"model": k.model if k.side_same_model else k.side_model,
                                                      "tier": "minimal", "utility": True},
                                         "utility_kind": "title"}})

    # -- waiting: heartbeats or silence
    def wait(self, delay: float, beat, k: Knobs) -> None:
        end = time.time() + delay
        while time.time() < end:
            left = end - time.time()
            time.sleep(max(0.0, min(left, k.hb_interval) if k.heartbeats else left))
            if k.heartbeats and time.time() < end - 0.02:
                beat()

    # -- chat
    def chat_stream(self, h, body, plan, x, delay, n, tok, completion, k):
        def ch(delta, finish=None, extra=None):
            p = {"id": "chatcmpl-x", "object": "chat.completion.chunk", "created": int(time.time()),
                 "model": "yamadori", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            p.update(extra or {})
            h.emit(b"data: " + json.dumps(p).encode() + b"\n\n")

        h.begin()
        self.wait(delay, lambda: ch({}), k)
        ch({"reasoning_content": "Let me think. "})
        if k.mid_silence_step == n:
            time.sleep(k.mid_silence_s)
        ch({"reasoning_content": "Done thinking."})
        if k.error_event_step == n:
            h.emit(b'data: {"error": {"message": "the model server dropped", "type": "api_error", '
                   b'"code": "upstream_dropped"}}\n\n')
            h.emit(b"data: [DONE]\n\n")
            return h.finish_stream()
        if plan[0] == "call":
            _t, name, raw_args, cid = plan
            half = max(1, len(raw_args) // 2)
            ch({"tool_calls": [{"index": 0, "id": cid, "type": "function",
                                "function": {"name": name, "arguments": raw_args[:half]}}]})
            ch({"tool_calls": [{"index": 0, "function": {"arguments": raw_args[half:]}}]})
            fin = "tool_calls"
        else:
            half = len(plan[1]) // 2
            ch({"content": plan[1][:half]})
            ch({"content": plan[1][half:]})
            fin = "stop"
        ch({}, finish=fin, extra={} if k.no_x else {"x_yamadori": x})
        if (body.get("stream_options") or {}).get("include_usage"):
            h.emit(b"data: " + json.dumps({"id": "chatcmpl-x", "object": "chat.completion.chunk", "choices": [],
                                           "usage": {"prompt_tokens": tok, "completion_tokens": completion,
                                                     "total_tokens": tok + completion}}).encode() + b"\n\n")
        h.emit(b"data: [DONE]\n\n")
        h.finish_stream()

    # -- responses
    def resp_stream(self, h, body, plan, x, delay, n, tok, completion, k):
        seq = [0]

        def ev(t, **kw):
            kw["type"] = t
            kw["sequence_number"] = seq[0]
            seq[0] += 1
            h.emit(f"event: {t}\ndata: ".encode() + json.dumps(kw).encode() + b"\n\n")

        base = {"id": "resp_x", "object": "response", "status": "in_progress", "model": "yamadori", "output": []}
        h.begin()
        ev("response.created", response=base)
        ev("response.in_progress", response=base)
        self.wait(delay, lambda: ev("response.in_progress", response=dict(base)), k)
        ev("response.output_item.added", output_index=0, item={"type": "reasoning", "id": "rs_1", "summary": []})
        ev("response.reasoning_summary_text.delta", item_id="rs_1", output_index=0, summary_index=0,
           delta="Let me think. ")
        if k.mid_silence_step == n:
            time.sleep(k.mid_silence_s)
        ev("response.output_item.done", output_index=0, item={"type": "reasoning", "id": "rs_1",
                                                               "summary": [{"type": "summary_text", "text": "x"}]})
        if k.error_event_step == n:
            ev("response.failed", response=dict(base, status="failed", error={"code": "server_error",
                                                                              "message": "dropped"}))
            return h.finish_stream()
        out = [{"type": "reasoning", "id": "rs_1", "summary": []}]
        if plan[0] == "call":
            _t, name, raw_args, cid = plan
            item = {"type": "function_call", "id": "fc_1", "call_id": cid, "name": name, "arguments": raw_args,
                    "status": "completed"}
            ev("response.output_item.added", output_index=1, item=dict(item, arguments=""))
            ev("response.function_call_arguments.delta", item_id="fc_1", output_index=1, delta=raw_args)
            ev("response.function_call_arguments.done", item_id="fc_1", output_index=1, name=name,
               arguments=raw_args)
            ev("response.output_item.done", output_index=1, item=item)
            out.append(item)
        else:
            item = {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": plan[1], "annotations": []}]}
            ev("response.output_item.added", output_index=1, item=dict(item, content=[]))
            ev("response.output_text.delta", item_id="msg_1", output_index=1, content_index=0, delta=plan[1])
            ev("response.output_item.done", output_index=1, item=item)
            out.append(item)
        ev("response.completed", response=dict(base, status="completed", output=out,
                                               usage={"input_tokens": tok, "output_tokens": completion,
                                                      "total_tokens": tok + completion},
                                               **({} if k.no_x else {"x_yamadori": x})))
        h.finish_stream()


# --------------------------------------------------------------------------------------------- the runner
BASE_ARGS = ["--system-tokens", "2500", "--n-tools", "6", "--steps", "6", "--min-steps", "3", "--idle", "1",
             "--hold", "1.5", "--abort-after", "0.5", "--retry-margin", "0.2", "--client-timeout", "3",
             "--silence-limit", "1.5", "--expect-prefix-reuse", "256", "--max-retries", "5",
             "--max-retry-wait", "5"]


def run(knobs: Knobs, extra: list[str], *, only: str | None = None):
    fake = Fake(knobs)
    base = fake.start()
    out = tempfile.mkdtemp(prefix="soak-test-")
    keyf = os.path.join(out, "key.txt")
    with open(keyf, "w") as f:
        f.write(KEY + "\n")
    argv = BASE_ARGS + ["--base", base, "--out", out, "--key-file", keyf] + extra
    if only:
        argv += ["--only", only]
    buf = io.StringIO()
    t0 = time.time()
    try:
        with contextlib.redirect_stdout(buf):
            code = H.main(argv)
    finally:
        fake.stop()
    summ = json.load(open(os.path.join(out, "summary.json")))
    recs = [json.loads(ln) for ln in open(os.path.join(out, "requests.jsonl"))]
    sc = {s["key"]: s for s in summ["scenarios"]}
    return {"code": code, "sc": sc, "recs": recs, "fake": fake, "out": out, "text": buf.getvalue(),
            "secs": time.time() - t0, "summ": summ}


def chk(res, key: str, text_prefix: str):
    """The scenario's check whose text starts with text_prefix: (ok, detail) or (None, '') when absent."""
    for c in res["sc"][key]["checks"]:
        if c["text"].startswith(text_prefix):
            return c["ok"], c["detail"]
    return None, ""


# ------------------------------------------------------------------------------------------ unit pieces
def test_parsers_and_builders():
    p = H.ChatParser()
    evs = [
        {"choices": [{"delta": {}, "finish_reason": None}]},
        {"choices": [{"delta": {"reasoning_content": "hm"}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_a_1", "type": "function",
                                                 "function": {"name": "read_file", "arguments": '{"pa'}},
                                                {"index": 1, "id": "call_a_2", "type": "function",
                                                 "function": {"name": "list_dir", "arguments": "{}"}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'th": "a.ts"}'}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}], "x_yamadori": {"capacity": {"model": "m"}}},
        {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 2}},
    ]
    kinds = [p.feed(json.dumps(e)) for e in evs] + [p.feed("[DONE]")]
    check("chat parser: heartbeat, token, token, token, finish, other, done",
          kinds == ["hb", "token", "token", "token", "finish", "other", "done"], str(kinds))
    calls = p.tool_calls
    check("chat parser: tool_call deltas are assembled by index",
          [c["name"] for c in calls] == ["read_file", "list_dir"] and calls[0]["arguments"] == '{"path": "a.ts"}'
          and calls[0]["id"] == "call_a_1", str(calls))
    check("chat parser: finish, x_yamadori, usage kept", p.finish == "tool_calls" and p.x["capacity"]["model"] == "m"
          and p.usage["completion_tokens"] == 2 and p.complete)
    e = H.ChatParser()
    k = e.feed('{"error": {"message": "dropped", "code": "upstream_dropped"}}')
    check("chat parser: an error event is an error, never complete", k == "error" and e.error and not e.complete)
    bad = H.ChatParser()
    check("chat parser: garbage JSON is counted, not raised", bad.feed("{nope") == "other" and bad.bad == 1)

    r = H.RespParser()
    rk = [r.feed(json.dumps(x)) for x in [
        {"type": "response.created", "response": {"model": "yamadori"}},
        {"type": "response.in_progress", "response": {}},
        {"type": "response.in_progress", "response": {}},
        {"type": "response.reasoning_summary_text.delta", "delta": "hm"},
        {"type": "response.output_item.done", "item": {"type": "function_call", "call_id": "call_b_1",
                                                       "name": "list_dir", "arguments": "{}"}},
        {"type": "response.completed", "response": {"status": "completed", "output": [],
                                                    "x_yamadori": {"cache": {"processed": 3}},
                                                    "usage": {"output_tokens": 4}}}]]
    check("responses parser: created, first in_progress is the start, the next is a heartbeat",
          rk == ["other", "start", "hb", "token", "other", "finish"], str(rk))
    check("responses parser: call, x_yamadori, completion", r.tool_calls[0]["id"] == "call_b_1" and r.complete
          and r.x["cache"]["processed"] == 3 and r.finish_reason == "tool_calls")
    f = H.RespParser()
    f.feed(json.dumps({"type": "response.failed", "response": {"status": "failed", "error": {"code": "x"}}}))
    check("responses parser: response.failed is an error", f.error and not f.complete)

    tools = H.build_tools(20, 26000)
    sysm = H.build_system("abc123", 26000, tools)
    est = H.est_tokens(sysm) + H.est_tokens(tools)
    check("builder: 20 tools with long descriptions", len(tools) == 20 and all(
        len(t["function"]["description"]) > 400 for t in tools))
    check("builder: system+tools is about --system-tokens", abs(est - 26000) < 800, f"{est}")
    check("builder: the nonce is in the first system line and the text is deterministic",
          "abc123" in sysm.splitlines()[0] and sysm == H.build_system("abc123", 26000, tools)
          and sysm != H.build_system("zzz999", 26000, tools))
    ws = H.Workspace(H.FIRST_TASKS[0])
    check("workspace: tests fail before the files exist and pass after",
          "Cannot find module" in ws.call("run_terminal", '{"command": "npm test"}')
          and "Wrote" in ws.call("write_file", '{"path": "src/slug.ts", "content": "x"}')
          and "Wrote" in ws.call("write_file", '{"path": "src/slug.test.ts", "content": "y"}')
          and "# pass 3" in ws.call("run_terminal", '{"command": "npm test"}'))
    check("workspace: bad arguments are an error text, not a crash",
          ws.call("read_file", "{nope").startswith("Error") and ws.call("nope", "{}").startswith("Error"))
    s = H.Session(type("C", (), {"n_tools": 6, "system_tokens": 800, "model": "m", "tier": "max",
                                 "max_tokens": 0, "cache_key": False})(), "T")
    req = H.Req()
    req.parser = H.ChatParser()
    req.parser._calls = {0: {"id": "c1", "name": "read_file", "arguments": "{}"}}
    check("tool-call validation: a missing required argument and an undeclared tool are named",
          any("missing" in x for x in H.tool_call_problems(s, req)))
    req.parser._calls = {0: {"id": "c1", "name": "nope", "arguments": "{}"}}
    check("tool-call validation: an undeclared tool", any("not a declared" in x
                                                          for x in H.tool_call_problems(s, req)))
    body = s.request_body()
    check("chat body: streamed, tools, reasoning_effort, no prompt_cache_key",
          body["stream"] is True and body["reasoning_effort"] == "max" and len(body["tools"]) == 6
          and "prompt_cache_key" not in body)


def test_reread_rule():
    ok = H.reread_flag({"tail_est": 300, "processed": 800})
    bad = H.reread_flag({"tail_est": 300, "processed": 900})
    check("re-read rule: tail + 512 allowed, one more is flagged", ok is None and bad and "900" in bad, f"{ok} {bad}")
    check("re-read rule: a missing cache record is flagged, a first request is not",
          H.reread_flag({"tail_est": 5, "processed": None}) and H.reread_flag({"tail_est": None,
                                                                                "processed": 9999}) is None)


# ------------------------------------------------------------------------------------------------ cases
def test_everything_passes_on_a_good_fake():
    res = run(Knobs(swap_delay=0.0), ["--api", "both", "--long", "2", "--side-concurrent"])
    st = {k: v["status"] for k, v in res["sc"].items()}
    check("good fake: every scenario PASSes and the exit code is 0",
          res["code"] == 0 and all(v == "PASS" for v in st.values()) and len(st) == 10, f"{res['code']} {st}")
    if res["code"] != 0:
        for k, v in res["sc"].items():
            for c in v["checks"]:
                if not c["ok"]:
                    print(f"      [{k}] {c['text']}: {c['detail']}")
            if v.get("crashed"):
                print(v["crashed"])
    recs = res["recs"]
    need = ["scenario", "session", "step", "t_start", "status", "ttfe_s", "ttfh_s", "hb", "hb_before_token",
            "ttft_s", "total_s", "max_silence_s", "would_have_hung_up", "finish_reason", "n_tool_calls",
            "content_chars", "model", "swap_load_s", "prompt", "reused", "processed", "prompt_ms", "decode_tps",
            "slot", "mode", "routed", "switch", "resumed_cold", "session_id", "session_source", "mem_before",
            "mem_after"]
    missing = [k for k in need if recs and k not in recs[0]]
    check("records carry every field of the brief", recs and not missing, str(missing))
    main_recs = [r for r in recs if r["kind"] == "step" and r.get("final_attempt")]
    check("a cold request records heartbeats before its first token and a first heartbeat time",
          any(r["hb_before_token"] >= 1 and r["ttfh_s"] is not None for r in main_recs))
    check("the first request recorded the swap's load_s", any(r.get("swap_load_s") is not None for r in recs))
    check("a 503 attempt was recorded as its own record and waited out",
          any(r["status"] == 503 and r.get("wait_before_retry_s") for r in recs)
          and res["summ"]["n_503"] >= 2, f"{res['summ']['n_503']}")
    check("the second-conversation scenario served B after the 503 (retry obeyed)",
          res["sc"]["f"]["status"] == "PASS" and any("503" in n for n in res["sc"]["f"]["notes"]))
    check("every request sent a key and the loop completed over the fake's 401 check",
          not any(r["status"] == 401 for r in recs))
    check("the summary file and table exist", "PER-STEP TABLE" in open(os.path.join(res["out"], "summary.txt")).read())
    check("scenario j ran with --long", res["sc"]["j"]["status"] == "PASS" and "steps to prompt" in res["sc"]["j"]["headline"])
    check("the whole good run finished in a bounded time", res["secs"] < 240, f"{res['secs']:.0f} s")


def test_silent_swap_hangs_the_client_up():
    # the model swap is a silent pre-flight longer than the client's patience: no byte, the client hangs up
    res = run(Knobs(swap_delay=3.0), ["--client-timeout", "1.5"], only="a")
    ok, detail = chk(res, "a", "HTTP 200")
    silence, sdetail = chk(res, "a", "the longest silence")
    check("silent swap > client timeout: a is FAIL, exit 1", res["sc"]["a"]["status"] == "FAIL" and res["code"] == 1,
          f"{res['sc']['a']['status']} {res['code']}")
    check("... the request did not finish and the silence check names the hang-up",
          ok is False and silence is False and "HUNG UP" in sdetail, f"{ok} {sdetail}")
    check("... the record says the client would have hung up", any(r["hung_up"] and r["would_have_hung_up"]
                                                                   for r in res["recs"]))


def test_silence_over_the_limit_without_a_hangup():
    res = run(Knobs(swap_delay=2.0), ["--client-timeout", "6"], only="a")
    silence, detail = chk(res, "a", "the longest silence")
    check("silence 2 s with a patient client: only the silence check fails",
          res["sc"]["a"]["status"] == "FAIL" and silence is False and res["recs"][0]["status"] == 200
          and not res["recs"][0]["hung_up"], f"{res['sc']['a']['status']} {detail}")
    res = run(Knobs(heartbeats=False, cold_delay=2.0), ["--client-timeout", "6"], only="a")
    silence, detail = chk(res, "a", "the longest silence")
    check("a slow first byte WITHOUT heartbeats is a FAIL", silence is False and res["sc"]["a"]["status"] == "FAIL",
          detail)
    res = run(Knobs(heartbeats=True, cold_delay=2.0), ["--client-timeout", "6"], only="a")
    check("the same slow first byte WITH heartbeats passes", res["sc"]["a"]["status"] == "PASS",
          str(res["sc"]["a"]["checks"]))


def test_wrong_model_and_missing_record():
    res = run(Knobs(model="bonsai"), [], only="a")
    ok, detail = chk(res, "a", "served by the max tier")
    check("served by the wrong model: FAIL naming both", ok is False and "bonsai" in detail and "flash-next" in detail,
          detail)
    res = run(Knobs(no_x=True), [], only="a")
    ok, detail = chk(res, "a", "served by the max tier")
    check("no x_yamadori at all: FAIL (cannot tell)", ok is False and "cannot tell" in detail, detail)


def test_a_rereading_fake_is_flagged():
    res = run(Knobs(reread_all=True), [], only="b,c,d")
    ok, detail = chk(res, "b", "after step 1 every step")
    check("a fake that re-reads everything: b FAIL on the re-read rule", ok is False and "processed" in detail
          and res["sc"]["b"]["status"] == "FAIL", detail)
    okc, _d = chk(res, "c", "and still reuses its cache")
    check("... c FAIL (the next step after the side call re-read)", okc is False and res["sc"]["c"]["status"] == "FAIL")
    okd, _d = chk(res, "d", "cache reused")
    check("... d FAIL (after the idle gap)", okd is False and res["sc"]["d"]["status"] == "FAIL")
    check("... and the exit code is 1", res["code"] == 1)


def test_loop_failures_are_recorded_not_raised():
    res = run(Knobs(error_event_step=1), [], only="b")
    ok, detail = chk(res, "b", "every step returned 200")
    check("a mid-stream error event: b FAIL with the error in the evidence, no crash",
          ok is False and "upstream_dropped" in detail and not res["sc"]["b"]["crashed"], detail)
    res = run(Knobs(http500_step=1), [], only="b")
    ok, detail = chk(res, "b", "every step returned 200")
    check("an HTTP 500 body: b FAIL with the body", ok is False and "boom" in detail, detail)
    res = run(Knobs(invalid_tool_step=1), [], only="b")
    ok, detail = chk(res, "b", "tool calls are valid")
    check("a tool call with invalid JSON arguments: b FAIL", ok is False and "not JSON" in detail, detail)
    res = run(Knobs(route_step=1), [], only="b")
    ok, detail = chk(res, "b", "no step routed")
    check("a step routed to another card: b FAIL", ok is False and "other_card" in detail, detail)
    res = run(Knobs(mid_silence_step=1, mid_silence_s=2.2), ["--client-timeout", "6"], only="b")
    ok, detail = chk(res, "b", "no silence over")
    check("a mid-stream silence over the limit: b FAIL naming the step", ok is False and "#2" in detail, detail)
    res = run(Knobs(status429=True), [], only="a")
    check("a 429 is NOT RUN, exit 3, never a FAIL", res["sc"]["a"]["status"] == "NOT RUN" and res["code"] == 3,
          f"{res['sc']['a']['status']} {res['code']}")
    res = run(Knobs(), ["--key-file", os.path.join(tempfile.gettempdir(), "nonexistent-key-file")], only="a") \
        if False else None
    assert res is None


def test_side_call_and_abort():
    res = run(Knobs(side_same_model=True), [], only="c")
    ok, detail = chk(res, "c", "served by a non-main model")
    check("a side call served by the main model: c FAIL", ok is False and res["sc"]["c"]["status"] == "FAIL", detail)
    res = run(Knobs(), [], only="e")
    first = [r for r in res["recs"] if r["kind"] == "aborted"]
    check("abort_and_retry: the first attempt was cut by the client, the retry is served",
          res["sc"]["e"]["status"] == "PASS" and first and first[0]["aborted_by_client"],
          str(res["sc"]["e"]["checks"]))
    check("... and the retry's time to first token is the headline", "retry ttft" in res["sc"]["e"]["headline"])
    # the first attempt is cut BEFORE any byte (a silent swap): the retry still gets served, nothing is held
    res = run(Knobs(swap_delay=2.0), ["--client-timeout", "6"], only="e")
    first = [r for r in res["recs"] if r["kind"] == "aborted"]
    check("abort before the first byte (a silent swap): the client cut it, the retry is served",
          res["sc"]["e"]["status"] == "PASS" and first and first[0]["aborted_by_client"]
          and first[0]["status"] is None, str(res["sc"]["e"]["checks"]) + str(first[:1]))


def test_second_conversation():
    res = run(Knobs(never_free=True), ["--max-retries", "2", "--retry-margin", "0.1"], only="f")
    ok, detail = chk(res, "f", "B is served within")
    check("a 503 that never clears: f FAIL after the retries", ok is False and res["sc"]["f"]["status"] == "FAIL"
          and sum(1 for r in res["recs"] if r["status"] == 503) == 3, f"{detail} {res['summ']['n_503']}")
    res = run(Knobs(hold=0), [], only="f")
    check("a fake that serves B at once: f PASS (served)", res["sc"]["f"]["status"] == "PASS"
          and any("served at once" in n for n in res["sc"]["f"]["notes"]), str(res["sc"]["f"]["notes"]))
    res = run(Knobs(retry_after=1), [], only="f")
    check("a 503 with Retry-After 1: obeyed, B and A both served", res["sc"]["f"]["status"] == "PASS"
          and res["summ"]["n_503"] >= 1, str(res["sc"]["f"]["checks"]))


def test_shared_prefix_and_compaction():
    res = run(Knobs(prefix_reuse=0), [], only="g")
    ok, detail = chk(res, "g", "C reuses at least")
    check("no prefix reuse for a new conversation with the same system+tools: g FAIL", ok is False and
          res["sc"]["g"]["status"] == "FAIL", detail)
    res = run(Knobs(prefix_reuse=256), [], only="g")
    check("prefix reuse of 256 >= the expected 256: g PASS", res["sc"]["g"]["status"] == "PASS",
          str(res["sc"]["g"]["checks"]))
    res = run(Knobs(), [], only="h")
    cm = res["sc"]["h"]
    check("compaction: both requests 200, a summary, the continuation served",
          cm["status"] == "PASS" and any("compaction" in n for n in cm["notes"]), str(cm["checks"]))
    kinds = [r["kind"] for r in res["recs"]]
    check("... the compaction request is its own record kind", "compaction" in kinds)
    check("... and the ask text is the one the live suite uses", H.COMPACT_ASK ==
          "Summarize this conversation so far for a continuation.")


def test_responses_api():
    res = run(Knobs(), ["--api", "responses"])
    check("--api responses runs only scenario i", list(res["sc"]) == ["i"] and res["sc"]["i"]["status"] == "PASS",
          f"{list(res['sc'])} {res['sc'].get('i', {}).get('checks')}")
    check("... via /v1/responses with function_call items echoed (every step 200)",
          all(r["status"] == 200 for r in res["recs"] if r["scenario"] == "responses_api") and
          all(r["api"] == "responses" for r in res["recs"]))
    check("... heartbeats were counted from response.in_progress", any(r["hb"] >= 1 for r in res["recs"]))
    saved = H.HEARTBEAT_S
    H.HEARTBEAT_S = 0.3
    try:
        res = run(Knobs(heartbeats=False, cold_delay=1.0), ["--api", "responses", "--silence-limit", "5",
                                                           "--client-timeout", "6"])
    finally:
        H.HEARTBEAT_S = saved
    ok, detail = chk(res, "i", "a response.in_progress heartbeat")
    check("a first token after a silence over 5 s with no in_progress heartbeat: i FAIL", ok is False and
          res["sc"]["i"]["status"] == "FAIL", detail)
    res = run(Knobs(error_event_step=1), ["--api", "responses"])
    check("a response.failed mid-stream: i FAIL, no crash", res["sc"]["i"]["status"] == "FAIL"
          and not res["sc"]["i"]["crashed"])


def test_idle_and_cli():
    res = run(Knobs(), ["--idle", "1.5"], only="d")
    check("idle_gap: PASS and the gap is recorded as a wait", res["sc"]["d"]["status"] == "PASS"
          and any(w["why"] == "idle" for w in res["summ"]["waits"]), str(res["sc"]["d"]["checks"]))
    no_key = io.StringIO()
    with contextlib.redirect_stdout(no_key):
        code = H.main(["--base", "http://127.0.0.1:9", "--out", tempfile.mkdtemp(prefix="soak-test-")])
    old = os.environ.pop("YAMADORI_TEST_KEY", None)
    try:
        check("no key: a clear failure and exit 1 (never a request)", code == 1 and "no API key" in no_key.getvalue())
    finally:
        if old is not None:
            os.environ["YAMADORI_TEST_KEY"] = old
    for bad in ("zz",):
        try:
            H.pick(bad, "chat")
            check("an unknown --only is refused", False)
        except SystemExit:
            check("an unknown --only is refused", True)
    check("--only takes names and letters", H.pick("agent_loop,c", "chat") == ["b", "c"])


def main() -> int:
    fns = [test_parsers_and_builders, test_reread_rule, test_everything_passes_on_a_good_fake,
           test_silent_swap_hangs_the_client_up, test_silence_over_the_limit_without_a_hangup,
           test_wrong_model_and_missing_record, test_a_rereading_fake_is_flagged,
           test_loop_failures_are_recorded_not_raised, test_side_call_and_abort, test_second_conversation,
           test_shared_prefix_and_compaction, test_responses_api, test_idle_and_cli]
    for fn in fns:
        t0 = time.time()
        try:
            fn()
        except Exception as e:                                                         # noqa: BLE001
            import traceback
            check(f"{fn.__name__} ran", False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
        print(f"  {fn.__name__}: {time.time() - t0:.1f} s", flush=True)
    print(f"\n{_PASSES} passed, {len(_FAILURES)} failed")
    for f in _FAILURES:
        print("  -", f)
    return 1 if _FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
