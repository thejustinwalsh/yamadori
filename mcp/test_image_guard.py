#!/usr/bin/env python
"""The image guard: image data in an image tool's argument is stopped at its
first characters. No GPU.

    python mcp/test_image_guard.py      -> "N/M checks passed"

THE FAILURE (Octopus v0e-V0-xhigh-1 prompt 2, 2026-09-26; #44 and #46 in
docs/SELF-IMPROVEMENT-LOG.md). The model read a screenshot saved as a base64
data URL (~57K characters) and transcribed it into vision_analyze's
image_url. The proxy assembled the call silently; the two attempts ran 23
and 40 minutes upstream before the relay died.

THE RULE (operator, 2026-09-26): for IMAGE arguments only -- Hermes'
vision_analyze, our yama_describe_image, and a parameter whose schema shape says
image -- a value that opens as image data (data:image/..., base64 image
bytes) is never valid from the model. No size threshold: it is caught at its
first characters. Every other argument is never read.

Gated here (tool_code IMAGE GUARD, proxy._post_events_raw / _run_turn):
  1. THE DETECTOR: which arguments take an image (table, shape, undeclared
     tool by name); what counts as image data at the prefix; a path, a URL,
     an attached image's id and our media link pass; a write_file -- the
     Octopus run's own space-shooter/js when present, a synthetic 30K JS
     file always, an HTML file with an inline data:image -- is not even
     read. The stopped call's arguments are valid JSON without the data.
  2. THE STREAM: a fake upstream DRIBBLES a 60K data URL into vision_analyze
     and into an undeclared-by-table tool with an image-shaped parameter;
     the proxy closes the connection within the first pieces, forwards
     nothing of the call, the second generation sees the stopped call and a
     NOT EXECUTED result with the remedy, and ITS call (a path) is forwarded
     whole. A 30K write_file dribbled the same way goes out unchanged.
  3. THE LEDGER: through the served chat template (test_ledger's harness),
     the request after a stopped call extends what the slot holds: the
     hidden hop is replayed, and the data is in no prompt.
  4. THE LANDING: a model that pastes image data IMAGE_REGENERATIONS + 1
     times gets the turn landed (tools withdrawn); no stopped call reaches
     the client.

FAIL BEFORE: `YAMADORI_IMAGE_GUARD=0 python mcp/test_image_guard.py` runs
the same checks with the guard off (the code before it): the stream, ledger
and landing checks fail (the fake sends every piece, the data URL is
forwarded).

Every database is a temp file set BEFORE proxy is imported.
"""
from __future__ import annotations

import base64
import glob
import json
import os
import random
import sys
import tempfile
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

_TMP = tempfile.mkdtemp(prefix="yamadori_test_image_guard_")
for _k, _v in (("YAMADORI_CORPUS_DB", "corpus.sqlite3"),
               ("YAMADORI_NEBARI_DB", "nebari.sqlite3"),
               ("RINGS_DB", "rings.sqlite3"),
               ("CODE_INDEX_DB", "code.sqlite3"),
               ("YAMADORI_SLOTS_STATE", "slots_state.json"),
               ("CONCEPT_SEED_LAST", "seed_last.json")):
    os.environ[_k] = os.path.join(_TMP, _v)
os.environ.setdefault("YAMADORI_GPU_ROOM", "0")

import test_ledger as T  # noqa: E402  (its own temp paths + fake upstream)
import compaction  # noqa: E402
import nebari  # noqa: E402
import proxy  # noqa: E402
import slots  # noqa: E402
import tool_code  # noqa: E402

_tmp_root = os.path.abspath(tempfile.gettempdir())
for _k in ("YAMADORI_CORPUS_DB", "YAMADORI_NEBARI_DB", "RINGS_DB",
           "CODE_INDEX_DB"):
    assert os.path.abspath(os.environ[_k]).startswith(_tmp_root), _k
assert os.path.abspath(nebari.DB).startswith(_tmp_root), nebari.DB

check = T.check
_results = T._results

# The image: 45,000 bytes after a real PNG signature, as base64 (60,000
# characters), and as the data URL the model wrote.
_rng = random.Random(46)
PNG = b"\x89PNG\r\n\x1a\n" + bytes(_rng.randrange(256) for _ in range(44992))
B64 = base64.b64encode(PNG).decode()
DATA_URL = "data:image/png;base64," + B64
VISION = {"type": "function", "function": {
    "name": "vision_analyze", "description": "Load an image into the "
    "conversation so you can see it.",
    "parameters": {"type": "object", "properties": {
        "image_url": {"type": "string", "description": "Image URL "
                      "(http/https), local file path, or data: URL to load."},
        "question": {"type": "string"}}}}}
# A tool no table knows, with an image-shaped parameter.
ANNOTATE = {"type": "function", "function": {
    "name": "annotate", "description": "Draw boxes on a picture.",
    "parameters": {"type": "object", "properties": {
        "src": {"type": "string", "description": "Location of the "
                "screenshot to annotate."},
        "boxes": {"type": "array"}}}}}
SHOT = "/workspace/shots/full.png"
OCTO_JS = r"C:\Users\jwals\octo\runs\v0e-V0-xhigh-1\space-shooter\js"


def synthetic_js(n: int = 30000) -> str:
    out = ["// " + "=" * 74, "// game.js -- the main loop", "// " + "=" * 74,
           "const ICON = 'data:image/png;base64," + B64[:400] + "';", ""]
    k = 0
    while sum(len(x) + 1 for x in out) < n:
        k += 1
        out += [f"class Enemy{k} extends Entity {{",
                "  constructor(x, y, speed = 2.5) { super(x, y); }",
                "  update(dt) {",
                "    for (let i = 0; i < this.trail.length; i++) {",
                "      this.trail[i].alpha *= 0.92;",
                "    }",
                "    this.y += this.speed * dt * 60;",
                "  }",
                "}", ""]
    return "\n".join(out)


def _watch(name: str, args: str, tools: list, piece: int = 3):
    w = tool_code.ImageArgWatch(tools)
    for i in range(0, len(args), piece):
        hit = w.feed(0, name, args[i:i + piece])
        if hit is not None:
            return hit, w
    return None, w


# --------------------------------------------------------------------------
def test_which_arguments_take_an_image():
    tools = [VISION, ANNOTATE, T.WRITE, {"type": "function", "function": {
        "name": "open_page", "parameters": {"type": "object", "properties": {
            "url": {"type": "string", "description": "The page to open."}}}}},
        {"type": "function", "function": {
            "name": "fetch_asset", "parameters": {"type": "object",
                                                  "properties": {
                "target": {"type": "string", "format": "uri",
                           "description": "A PNG or JPEG to fetch."}}}}}]
    ia = tool_code.image_arguments
    check(ia("vision_analyze", []) == {"image_url"},
          "[args] Hermes vision_analyze: image_url (the table)")
    check(ia("yama_describe_image", []) == {"image"},
          "[args] our yama_describe_image: image (the table)")
    check(ia("annotate", tools) == {"src"},
          "[args] shape: a locator described as a screenshot",
          str(ia("annotate", tools)))
    check(ia("fetch_asset", tools) == {"target"},
          "[args] shape: format uri described as an image")
    check(ia("open_page", tools) == set() and ia("write_file", tools) == set(),
          "[args] a page URL and a write_file take no image")
    check(ia("made_up_tool", tools) is None,
          "[args] a tool the request does not declare: judged by its "
          "argument names as they arrive")


def test_the_prefix_is_judged_at_once():
    args = json.dumps({"image_url": DATA_URL, "question": "what is shown?"})
    hit, _ = _watch("vision_analyze", args, [VISION])
    opened = len('{"image_url": "data:image/')
    check(hit is not None and hit["at"] <= opened + 3
          and hit["argument"] == "image_url"
          and hit["kind"] == "a data:image/ URL",
          "[prefix] a data:image/ URL is caught at its first characters, "
          "not after N KB", str(hit))
    raw = json.dumps({"question": "what is shown?", "image_url": B64})
    hit, _ = _watch("vision_analyze", raw, [VISION])
    check(hit is not None and hit["kind"] == "base64 PNG data"
          and hit["at"] <= raw.index(B64) + 24 + 3,
          "[prefix] raw base64 image bytes (PNG) are caught within 24 "
          "characters", str(hit))
    for label, v in (("a path", SHOT),
                     ("a Windows path", r"C:\Users\x\shot.png"),
                     ("an http URL", "https://example.com/a.png"),
                     ("an attached image's id", "image-3f9a1c2b7d"),
                     ("our media link", "https://yamadori.example/media/"
                                        "ab12cd34.png?exp=1&sig=ff00"),
                     ("a file named like data", "data/images/shot.png"),
                     ("a path that opens like JPEG base64", "/9j/x.png")):
        a = json.dumps({"image_url": v, "question": "what?"})
        hit, _ = _watch("vision_analyze", a, [VISION])
        check(hit is None, f"[prefix] {label} passes: {v}", str(hit))
    hit, _ = _watch("annotate", json.dumps({"boxes": [], "src": DATA_URL}),
                    [ANNOTATE])
    check(hit is not None and hit["argument"] == "src",
          "[prefix] an unknown tool's image-shaped parameter is guarded",
          str(hit))
    hit, _ = _watch("made_up_tool", json.dumps({"image": DATA_URL}),
                    [VISION])
    check(hit is not None and hit["argument"] == "image",
          "[prefix] an undeclared tool: its `image` argument, by name",
          str(hit))
    hit, _ = _watch("made_up_tool", json.dumps({"images": [SHOT, DATA_URL]}),
                    [])
    check(hit is not None and hit["argument"] == "images",
          "[prefix] a data URL inside a list of images", str(hit))
    hit, _ = _watch("vision_analyze", json.dumps(
        {"image_url": "data:application/octet-stream;base64," + B64}),
        [VISION])
    check(hit is not None and hit["kind"] == "a base64 data: URL",
          "[prefix] any base64 data: URL in an image argument", str(hit))


def test_a_write_is_not_read():
    legit = [("synthetic 30K JS (it carries an inline data:image icon)",
              synthetic_js())]
    for f in sorted(glob.glob(os.path.join(OCTO_JS, "*.js"))):
        legit.append((os.path.basename(f), open(f, encoding="utf-8").read()))
    legit.append(("index.html with an inline image",
                  f'<img src="{DATA_URL}">'))
    for label, text in legit:
        args = json.dumps({"path": "js/x.js", "content": text})
        hit, w = _watch("write_file", args, [T.WRITE, VISION], piece=64)
        check(hit is None and not w.read(0),
              f"[write] a write_file is not even read: {label} "
              f"({len(text):,} chars)")
    check(any(len(t) > 30000 for _, t in legit),
          "[write] at least one write over 30K characters was offered",
          str([(label[:20], len(t)) for label, t in legit]))


def test_the_stopped_call_and_its_result():
    args = json.dumps({"question": "what is on screen?",
                       "image_url": DATA_URL})
    hit, _ = _watch("vision_analyze", args, [VISION])
    seen = args[:hit["at"]] if hit else ""
    stop = tool_code.image_stopped("vision_analyze", seen, hit or {})
    got = json.loads(stop["arguments"])
    check(got == {"question": "what is on screen?",
                  "image_url": tool_code.IMAGE_CUT},
          "[stop] the stopped call keeps every key before the data; the "
          "value is replaced (valid JSON)", stop["arguments"])
    res = tool_code.image_result(stop, generated=True)
    check(res.startswith("NOT EXECUTED: the `vision_analyze` call's "
                         "`image_url` argument was image data")
          and "Retryable: yes" in res and "saved under the project folder"
          in res and "yama_generate_image" in res and res.count("don't") == 1,
          "[stop] the result: the situation, retryable as a fact, the "
          "remedy (a path; our link for a generated image), one \"don't\"",
          res)
    check("yama_generate_image" not in tool_code.image_result(stop),
          "[stop] the yama_generate_image remedy only where it is offered")
    for text, want in (('{"a": "xy', {"a": "xy"}),
                       ('{"a": [1, {"b": "c\\', {"a": [1, {"b": "c"}]}),
                       ('{"a": 1, ', {"a": 1}),
                       ('{"a": ', {"a": None})):
        check(json.loads(tool_code._close_json(text)) == want,
              f"[stop] a cut JSON text is closed: {text!r}",
              tool_code._close_json(text))


# --------------------------------------------------------------------------
class _Dribble(BaseHTTPRequestHandler):
    """An upstream that streams each scripted call's arguments in PIECE-
    character pieces, PAUSE apart, and records per generation how many
    pieces it managed to send before the client hung up."""
    PIECE = 256
    PAUSE = 0.01
    script: list[dict] = []
    record: list[dict] = []

    def _json(self, obj: dict) -> None:
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):                                           # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path.endswith("/apply-template"):
            return self._json({"prompt": T.render(body)})
        if self.path.endswith("/tokenize"):
            return self._json({"tokens": [1] * len(body.get("content") or "")})
        if self.path.endswith("/completion"):
            return self._json({"timings": {"cache_n": 100, "prompt_n": 20}})
        if not _Dribble.script:
            self.send_response(500)
            self.end_headers()
            return
        r = _Dribble.script.pop(0)
        a = r["args"]
        pieces = [a[i:i + _Dribble.PIECE]
                  for i in range(0, len(a), _Dribble.PIECE)]
        rec = {"body": body, "total": len(pieces), "sent": 0, "error": None,
               "t0": time.time(), "t_end": None}
        _Dribble.record.append(rec)

        def ev(delta=None, finish=None):
            return (b"data: " + json.dumps({
                "id": "up-1", "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": delta or {},
                             "finish_reason": finish}]}).encode() + b"\n\n")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        try:
            self.wfile.write(ev({"role": "assistant"}))
            self.wfile.write(ev({"reasoning_content": r["reasoning"]}))
            for k, p in enumerate(pieces):
                tc = {"index": 0, "function": {"arguments": p}}
                if k == 0:
                    tc.update(id=r["id"], type="function")
                    tc["function"]["name"] = r["name"]
                self.wfile.write(ev({"tool_calls": [tc]}))
                self.wfile.flush()
                rec["sent"] = k + 1
                time.sleep(_Dribble.PAUSE)
            self.wfile.write(ev({}, "tool_calls"))
            self.wfile.write(b"data: " + json.dumps({
                "id": "up-1", "choices": [], "usage": {
                    "prompt_tokens": 100, "completion_tokens": 10,
                    "total_tokens": 110},
                "timings": {"cache_n": 90, "prompt_n": 10}}).encode()
                + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except OSError as e:
            rec["error"] = type(e).__name__
        rec["t_end"] = time.time()

    def log_message(self, *a):                                   # noqa: D102
        pass


def _stream_raw(body: dict) -> tuple[list[bytes], dict]:
    """proxy.stream_body's bytes, and what a streaming client stores."""
    chunks = list(proxy.stream_body(dict(body, stream=True)))
    out = {"content": "", "reasoning": "", "calls": [], "x": None}
    for b in chunks:
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
                    out["calls"].append(c)
    return chunks, out


def _body(tag: str, user: str, tools: list) -> dict:
    return {"model": "yamadori", "reasoning_effort": "xhigh",
            "_account": f"{T.ACCOUNT}-img-{tag}", "_client_ip": "127.0.0.1",
            "tools": tools, "messages": [
                {"role": "system", "content": T.SYSTEM + f" [img {tag}]"},
                {"role": "user", "content": user}],
            "_features": json.dumps({"hints": True, "investigate": False,
                                     "fanout": 1})}


def _stopped_stream(tag: str, name: str, bad: str, good: str, tools: list,
                    argument: str) -> None:
    _Dribble.script[:] = [
        {"reasoning": "I will pass the screenshot.", "id": "v1",
         "name": name, "args": bad},
        {"reasoning": "Pass the path.", "id": "v2", "name": name,
         "args": good}]
    _Dribble.record.clear()
    chunks, got = _stream_raw(_body(tag, "Look at the screenshot in "
                                         "/tmp/full_url.txt", tools))
    recs = list(_Dribble.record)
    r0 = recs[0] if recs else {}
    check(len(recs) == 2,
          f"[stream {tag}] two generations: the stopped one and the one "
          f"after the NOT EXECUTED result", f"{len(recs)} generations")
    check(bool(r0) and r0["sent"] <= 3 and r0["error"] is not None,
          f"[stream {tag}] the upstream connection was closed within the "
          f"first pieces of the call (it does not run to completion)",
          f"sent {r0.get('sent')} of {r0.get('total')} pieces, error "
          f"{r0.get('error')}")
    check(not any(B64[:64].encode() in b for b in chunks),
          f"[stream {tag}] no byte of the image data reached the client")
    check(len(got["calls"]) == 1
          and got["calls"][0]["function"]["arguments"] == good,
          f"[stream {tag}] the second generation's call (a path) is "
          f"forwarded whole, the only call the client gets",
          json.dumps(got["calls"])[:300])
    req2 = recs[1]["body"]["messages"] if len(recs) > 1 else []
    hop, res = (req2[-2], req2[-1]) if len(req2) >= 2 else ({}, {})
    hc = (hop.get("tool_calls") or [{}])[0]
    hargs = (hc.get("function") or {}).get("arguments")
    hargs = json.loads(hargs) if isinstance(hargs, str) else hargs
    check(hop.get("role") == "assistant" and hc.get("id") == "v1"
          and isinstance(hargs, dict)
          and hargs.get(argument) == tool_code.IMAGE_CUT,
          f"[stream {tag}] the second generation sees the stopped call, "
          f"its image value replaced", json.dumps(hop)[:300])
    check(res.get("role") == "tool" and res.get("tool_call_id") == "v1"
          and res.get("content", "").startswith(
              f"NOT EXECUTED: the `{name}` call's `{argument}` argument was "
              f"image data")
          and "path of the image file" in res.get("content", ""),
          f"[stream {tag}] and a NOT EXECUTED result naming the argument "
          f"and the remedy", res.get("content", "")[:240])
    check(len(recs) > 1 and B64[:64] not in json.dumps(recs[1]["body"]),
          f"[stream {tag}] the data is not in the second generation's "
          f"prompt")
    check(f"stopped a {name} call" in got["reasoning"]
          and got["content"] == "",
          f"[stream {tag}] the client sees one reasoning line, no content",
          repr(got["reasoning"])[-200:])
    ig = (got["x"] or {}).get("image_guard") or {}
    st = (ig.get("stopped") or [{}])[0]
    check(ig.get("regenerated") == 1 and ig.get("landed") is False
          and st.get("tool") == name and st.get("argument") == argument
          and st.get("kind") == "a data:image/ URL"
          and 0 < (st.get("deltas") or 0) <= 3 and st.get("hop") == 0,
          f"[stream {tag}] x_yamadori.image_guard records the tool, "
          f"argument, kind, deltas read before the stop, the regeneration",
          json.dumps(ig)[:400])
    tt = (got["x"] or {}).get("tool_turns") or {}
    check(tt.get("turns") == 1 and not tt.get("hit"),
          f"[stream {tag}] the stopped call counts as a tool turn",
          json.dumps(tt))


def test_the_stream_stops_at_the_prefix():
    slots.reset(n=4)
    compaction.reset()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Dribble)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    saved = proxy.UPSTREAM
    proxy.UPSTREAM = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        q = "what is on screen?"
        _stopped_stream(
            "vision", "vision_analyze",
            json.dumps({"image_url": DATA_URL, "question": q}),
            json.dumps({"image_url": SHOT, "question": q}),
            [VISION, T.WRITE], "image_url")
        _stopped_stream(
            "shape", "annotate",
            json.dumps({"src": DATA_URL, "boxes": []}),
            json.dumps({"src": SHOT, "boxes": []}),
            [ANNOTATE, T.WRITE], "src")

        # A 30K write_file (with an inline data:image icon in its code),
        # dribbled the same way: untouched.
        js = synthetic_js()
        big = json.dumps({"path": "js/game.js", "content": js})
        _Dribble.script[:] = [{"reasoning": "Write the game loop.",
                               "id": "w1", "name": "write_file",
                               "args": big}]
        _Dribble.record.clear()
        _, got = _stream_raw(_body("legit", "Write js/game.js",
                                   [T.WRITE, VISION]))
        r = _Dribble.record[0] if _Dribble.record else {}
        check(len(_Dribble.record) == 1 and r.get("sent") == r.get("total")
              and len(got["calls"]) == 1
              and got["calls"][0]["function"]["arguments"] == big
              and (got["x"] or {}).get("image_guard") is None,
              f"[stream legit] a {len(js):,}-character write_file is not "
              f"touched: one generation, forwarded unchanged",
              f"{len(_Dribble.record)} gens, "
              f"{json.dumps((got['x'] or {}).get('image_guard'))}")
    finally:
        proxy.UPSTREAM = saved
        srv.shutdown()


# --------------------------------------------------------------------------
class VisionClient(T.Client):
    def __init__(self, tag: str):
        super().__init__("xhigh")
        self.msgs[0]["content"] += f" [img ledger {tag}]"

    def body(self, features: dict | None = None) -> dict:
        b = super().body(features)
        b["tools"] = [VISION, T.WRITE]
        return b


def test_the_next_request_extends_the_slot():
    slots.reset(n=4)
    compaction.reset()
    c = VisionClient("extends")
    bad = T.call("vision_analyze", {"image_url": DATA_URL,
                                    "question": "what is on screen?"}, "v1")
    good = T.call("vision_analyze", {"image_url": SHOT,
                                     "question": "what is on screen?"}, "v2")
    t1 = c.turn([T.reply("", reasoning="I will pass the screenshot.",
                         calls=[bad]),
                 T.reply("", reasoning="Pass the path.", calls=[good])],
                user="Look at the screenshot in /tmp/full_url.txt")
    x = t1["d"]["x_yamadori"]
    m = t1["d"]["choices"][0]["message"]
    calls = m.get("tool_calls") or []
    check(len(t1["gens"]) == 2 and len(calls) == 1
          and json.loads(calls[0]["function"]["arguments"])["image_url"]
          == SHOT,
          "[ledger] two generations; the client gets the path call only",
          json.dumps(calls)[:200])
    check((x.get("image_guard") or {}).get("regenerated") == 1,
          "[ledger] x_yamadori.image_guard on the blocking path too",
          json.dumps(x.get("image_guard"))[:200])
    check(len(t1["gens"]) > 1
          and B64[:64] not in T.render(t1["gens"][1]["request"]),
          "[ledger] the data is not in the second generation's prompt")
    c.tool_result("v2", "A title screen with a START button.")
    t2 = c.turn([T.reply("The screen shows the title.")])
    T._extends(t1, t2, "[ledger] the request after a stopped call")
    msgs2 = t2["gens"][0]["request"]["messages"]
    check(any(mm.get("role") == "tool" and mm.get("tool_call_id") == "v1"
              and mm.get("content", "").startswith("NOT EXECUTED")
              for mm in msgs2),
          "[ledger] the hidden hop (stopped call + NOT EXECUTED) is "
          "replayed from the ledger")
    check(B64[:64] not in T.render(t2["gens"][0]["request"]),
          "[ledger] and the data is in no later prompt")


def test_a_third_stop_lands_the_turn():
    slots.reset(n=4)
    compaction.reset()
    c = VisionClient("land")
    script = [T.reply("", reasoning=f"Try {k}.", calls=[T.call(
        "vision_analyze", {"image_url": DATA_URL}, f"b{k}")])
        for k in range(tool_code.IMAGE_REGENERATIONS + 1)]
    script.append(T.reply("I could not load the image; it is in "
                          "/tmp/full_url.txt."))
    t = c.turn(script, user="Look at the screenshot in /tmp/full_url.txt")
    ig = t["d"]["x_yamadori"].get("image_guard") or {}
    last = t["gens"][-1]["request"]
    m = t["d"]["choices"][0]["message"]
    check(len(t["gens"]) == tool_code.IMAGE_REGENERATIONS + 2
          and not last.get("tools")
          and last["messages"][-1].get("content") == proxy.LANDING_PROMPT,
          "[land] after IMAGE_REGENERATIONS + 1 stopped calls the turn "
          "lands (tools withdrawn, the answer asked for)",
          f"{len(t['gens'])} gens, tools {len(last.get('tools') or [])}")
    check(ig.get("landed") is True and len(ig.get("stopped") or []) ==
          tool_code.IMAGE_REGENERATIONS + 1
          and ig.get("regenerated") == tool_code.IMAGE_REGENERATIONS,
          "[land] the record says so", json.dumps(ig)[:300])
    check(not m.get("tool_calls") and "could not load" in (m.get("content")
                                                           or ""),
          "[land] no stopped call reaches the client; the answer does",
          json.dumps(m)[:200])


def main() -> int:
    print("  image guard " + ("ON" if tool_code.IMAGE_GUARD else
                              "OFF (the fail-before run)"))
    for fn in (test_which_arguments_take_an_image,
               test_the_prefix_is_judged_at_once, test_a_write_is_not_read,
               test_the_stopped_call_and_its_result,
               test_the_stream_stops_at_the_prefix,
               test_the_next_request_extends_the_slot,
               test_a_third_stop_lands_the_turn):
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
    T._srv.shutdown()
    passed = sum(1 for ok, _, _ in _results if ok)
    total = len(_results)
    print(f"\n{'=' * 70}\n  {passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
