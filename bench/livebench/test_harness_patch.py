"""Harness-only check of the 429 retry patch against a local stub (no model involved)."""
import json, os, threading, time
from http.server import BaseHTTPRequestHandler, HTTPServer

os.environ["LB_EXTRA_HEADERS"] = json.dumps({"X-Yamadori-Features": "{}"})
calls = []
sessions = []

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        calls.append(time.time())
        sessions.append(self.headers.get("X-Yamadori-Session"))
        self.rfile.read(int(self.headers["Content-Length"]))
        if len(calls) <= 2:
            body, code = {"error": {"message": "all 4 main lanes busy", "type": "rate_limit_error"}}, 429
        else:
            body, code = {"id": "x", "object": "chat.completion", "created": 0, "model": "yamadori",
                          "choices": [{"index": 0, "finish_reason": "stop",
                                       "message": {"role": "assistant", "content": "ok", "reasoning_content": "r"}}],
                          "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                          "x_yamadori": {"selection": {}}}, 200
        b = json.dumps(body).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

srv = HTTPServer(("127.0.0.1", 0), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
from livebench.model.completions import chat_completion_openai
out, n, meta = chat_completion_openai("yamadori", [{"role": "user", "content": "hi"}], 1.0, 16, None,
                                      {"api_key": "x", "api_base": f"http://127.0.0.1:{srv.server_port}/v1"})
gaps = [round(b - a, 2) for a, b in zip(calls, calls[1:])]
checks = [("answer returned after two 429s", out == "ok"),
          ("exactly 3 requests (no SDK retries on top)", len(calls) == 3),
          ("retries_429 recorded = 2", meta.get("retries_429") == 2),
          ("each wait is 1-2 s", all(0.9 <= g <= 2.3 for g in gaps)),
          ("finish_reason still recorded", meta.get("finish_reason") == "stop"),
          ("session header sent on every attempt, same nonce within one answer",
           all(sessions) and len(set(sessions)) == 1 and len(sessions[0]) == 32),
          ("session nonce recorded in metadata", meta.get("session_nonce") == sessions[0])]
calls.clear(); first = sessions[0]; sessions.clear()
out2, _, meta2 = chat_completion_openai("yamadori", [{"role": "user", "content": "hi"}], 1.0, 16, None,
                                        {"api_key": "x", "api_base": f"http://127.0.0.1:{srv.server_port}/v1"})
checks.append(("a second answer gets a fresh nonce", meta2.get("session_nonce") and meta2["session_nonce"] != first))
for name, ok in checks:
    print(("ok   " if ok else "FAIL ") + name, gaps if "wait" in name else "")
print(f"{sum(o for _, o in checks)}/{len(checks)} checks passed")
