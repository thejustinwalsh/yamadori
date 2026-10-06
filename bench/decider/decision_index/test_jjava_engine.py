"""Offline check of jjava_engine against a local fake /jev server (never touches the stack)."""
import json, os, sys, threading
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(__file__))
import jjava_engine
from decision_index.engines.base import Unsupported

SEQ = []


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        code, hdr, out = SEQ.pop(0)
        data = json.dumps(out(body) if callable(out) else out).encode()
        self.send_response(code)
        for k, v in hdr.items():
            self.send_header(k, v)
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


srv = HTTPServer(("127.0.0.1", 0), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
Q = {"q": {"type": "choice", "instructions": "x", "criteria": {"a": "a", "b": "b"}}}
ok = {"model": "m", "answers": {"q": {"type": "choice", "choice": "a", "probabilities": {"a": 0.7, "b": 0.3}}}}
e = jjava_engine.JjavaEngine(base_url=f"http://127.0.0.1:{srv.server_address[1]}", model="jjava-latest")
SEQ[:] = [(429, {"Retry-After": "1"}, {"detail": "busy"}), (200, {}, ok)]
resp, raw = e("s", Q)
assert raw["x_adapter"]["waited_s"] == 1.0 and resp["answers"]["q"]["choice"] == "a", raw
SEQ[:] = [(422, {}, {"detail": [{"loc": ["body", "state"], "msg": "too long", "type": "too_long"}]})]
try:
    e("s", Q); raise SystemExit("expected Unsupported")
except Unsupported:
    pass
SEQ[:] = [(422, {}, {"detail": [{"loc": ["body", "questions"], "msg": "bad", "type": "missing"}]})]
try:
    e("s", Q); raise SystemExit("expected error")
except Unsupported:
    raise SystemExit("a non-capacity 422 must stay an error")
except Exception as exc:
    assert "422" in str(exc)
print("jjava_engine: 3 checks passed")
