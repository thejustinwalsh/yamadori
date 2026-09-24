#!/usr/bin/env python
"""Vision (describe_image and attached images), against fake servers. No GPU.

WHAT THIS GATES

`mcp/vision.py` and its wiring in the proxy. The promises, each asserted:

  1. The request the vision server receives: model `bonsai-vision`, the image
     as a data: URI built here (MIME sniffed from the bytes), the question as
     text, the vendor sampling tiers.apply writes, and a token budget that
     fits the vision server's -c 16384.
  2. Every failure names itself, says whether retrying helps (as a fact), and
     carries a remedy with an owner: bad arguments, vision off, unknown image,
     a link, an invalid link, a missing file, not an image, too large, busy,
     loading, unavailable, no projector, out of memory, context full, a
     rejected image, not configured, timeout, down, a budget event, empty.
  3. THE SECURITY CONTRACT. An image comes only from this request's own
     attachments or our media store, by id. No URL is ever fetched -- not a
     model-supplied one, not a client's image link -- and no model-supplied
     path is ever opened. Only the vision server is ever contacted.
  4. describe_image is offered wherever generate_image is (every tier, and
     deep thinking), and on every tier when the request carries an image,
     even with no image server. YAMADORI_VISION=0 withholds it.
  5. x_yamadori.vision records each call (ok, image, source, format, bytes,
     seconds, tokens, or the error code) and x_yamadori.attachments what the
     request carried -- never the bytes, never the question.
  6. An attached image reaches the text model as a placeholder naming its id,
     never as an image part; bad attachments get a placeholder that says why;
     nothing crashes and nothing is dropped. A text-only request is untouched,
     and an attached image does not floor the main model's thinking budget.

ISOLATION

Temp stores for everything, set BEFORE import. The chat upstream, Laya and
the model server point at a closed port except where a fake is started. The
fakes are shut down at the end; nothing is left listening.
"""
from __future__ import annotations

import base64
import builtins
import json
import os
import re
import socket
import struct
import sys
import tempfile
import threading
import time
import traceback
import urllib.request
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

_TMP = tempfile.mkdtemp(prefix="yamadori_test_vision_")


def _dead_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


DEAD = f"http://127.0.0.1:{_dead_port()}"
os.environ.update({
    "YAMADORI_ACCOUNTS_DIR": os.path.join(_TMP, "accounts"),
    "YAMADORI_CORPUS_DB": os.path.join(_TMP, "corpus.sqlite3"),
    "YAMADORI_NEBARI_DB": os.path.join(_TMP, "nebari.sqlite3"),
    "RINGS_DB": os.path.join(_TMP, "rings.sqlite3"),
    "YAMADORI_PKG_DIR": os.path.join(_TMP, "pkgs"),
    "CODE_INDEX_DB": os.path.join(_TMP, "code.sqlite3"),
    "YAMADORI_INDEX_DIR": os.path.join(_TMP, "repos"),
    "YAMADORI_MEDIA_DIR": os.path.join(_TMP, "media"),
    "YAMADORI_MEDIA_SECRET_FILE": os.path.join(_TMP, "media_url.key"),
    "LLAMA_STACK_URL": DEAD, "YAMADORI_MODEL_SERVER": DEAD, "LAYA_URL": DEAD,
    "YAMADORI_IMAGE_WAIT": "0.3",
})
for k in ("YAMADORI_IMAGEGEN_URL", "YAMADORI_PUBLIC_BASE", "YAMADORI_VISION",
          "YAMADORI_VISION_MODEL", "YAMADORI_VISION_CTX",
          "YAMADORI_VISION_MAX_BYTES", "YAMADORI_VISION_TIMEOUT",
          "YAMADORI_IMAGEGEN_DEFAULT"):
    os.environ.pop(k, None)

import admission  # noqa: E402
import budget  # noqa: E402
import images  # noqa: E402
import model  # noqa: E402
import tiers  # noqa: E402
import vision  # noqa: E402

tiers._accepted = tiers.FALLBACK_EFFORTS     # no network for the template
_real_budgets = budget.budgets
budget.budgets = lambda pool=None: _real_budgets(pool or 163840)

REAL_MEDIA = os.path.abspath(os.path.join(HERE, "..", "index", "media"))

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


def tiny_png(seed: int, pad: int = 0) -> bytes:
    r, g = seed % 256, (seed // 256) % 256
    raw = b"".join(b"\x00" + bytes([r, g, 9]) * 2 for _ in range(2))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))
    return (images.PNG_MAGIC
            + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
            + b"\x00" * pad)


def data_uri(data: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def img_part(data: bytes, mime: str = "image/png") -> dict:
    return {"type": "image_url", "image_url": {"url": data_uri(data, mime)}}


# --------------------------------------------------------------------------
# Fake servers: the vision model (non-streamed JSON, as model.post reads it),
# the image server (sd-server), and the chat model (SSE, as the proxy reads it)
# --------------------------------------------------------------------------
V_MODE = {"mode": "ok"}
V_SEEN: list[dict] = []
ANSWER = "A red fox sitting on grass, facing left. No text is visible."


class _Vision(BaseHTTPRequestHandler):
    def do_POST(self):                                           # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        V_SEEN.append({"path": self.path, "body": body})
        m = V_MODE["mode"]
        finish, content = "stop", ANSWER
        if m == "slow":
            time.sleep(3)
        replies = {
            "loading": (503, {"error": {"code": 503, "message": "Loading model",
                                        "type": "unavailable_error"}}),
            "unavailable": (502, "upstream process exited unexpectedly"),
            "no_projector": (500, {"error": {"message": "image input is not "
                                             "supported - hint: if this is unexpected, "
                                             "you may need to provide the mmproj"}}),
            "oom": (500, {"error": {"message": "CUDA error: out of memory"}}),
            "context": (400, {"error": {"message": "the request exceeds the "
                                        "available context size, try increasing it"}}),
            "reject": (400, {"error": {"message": "failed to load image"}}),
            "notfound": (404, {"error": "could not find suitable inference "
                                        "handler for bonsai-vision"}),
            "garbage": (200, "this is not json"),
        }
        if m in replies:
            code, payload = replies[m]
            data = (payload if isinstance(payload, str)
                    else json.dumps(payload)).encode()
        else:
            if m == "length":
                finish, content = "length", ""
            elif m == "empty":
                content = "   "
            code = 200
            data = json.dumps({"choices": [{"index": 0, "finish_reason": finish,
                                            "message": {"role": "assistant",
                                                        "content": content,
                                                        "reasoning_content": "looking"}}],
                               "usage": {"prompt_tokens": 1180,
                                         "completion_tokens": 311,
                                         "total_tokens": 1491}}).encode()
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except OSError:
            pass

    def log_message(self, *a):                                   # noqa: D102
        pass


I_SEEN: list[dict] = []


class _Img(BaseHTTPRequestHandler):
    def do_POST(self):                                           # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        I_SEEN.append(body)
        data = json.dumps({"images": [base64.b64encode(
            tiny_png(int(body["seed"]))).decode()]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):                                   # noqa: D102
        pass


LLM_SCRIPT: list = []
LLM_SENT: list[dict] = []


class _LLM(BaseHTTPRequestHandler):
    def do_POST(self):                                           # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        LLM_SENT.append(body)
        step = LLM_SCRIPT.pop(0) if LLM_SCRIPT else (lambda b: {"content": "?"})
        r = step(body)
        chunks = [{"choices": [{"index": 0, "delta": {"role": "assistant"}}]}]
        if r.get("content"):
            chunks.append({"choices": [{"index": 0, "delta": {"content": r["content"]}}]})
        for i, c in enumerate(r.get("calls") or []):
            chunks.append({"choices": [{"index": 0, "delta": {
                "tool_calls": [dict(c, index=i)]}}]})
        chunks.append({"choices": [{"index": 0, "delta": {}, "finish_reason":
                                    "tool_calls" if r.get("calls") else "stop"}]})
        data = b"".join(b"data: " + json.dumps(c).encode() + b"\n\n"
                        for c in chunks) + b"data: [DONE]\n\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):                                   # noqa: D102
        pass


_servers = []


def _start(handler) -> str:
    s = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    _servers.append(s)
    return f"http://127.0.0.1:{s.server_address[1]}"


VISION_URL = _start(_Vision)
IMG_URL = _start(_Img)
LLM_URL = _start(_LLM)
model.UPSTREAM = VISION_URL


class use:
    """Set env vars for the length of a block."""

    def __init__(self, **env):
        self.env = env
        self.old: dict = {}

    def __enter__(self):
        for k, v in self.env.items():
            self.old[k] = os.environ.get(k)
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return self

    def __exit__(self, *exc):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        V_MODE["mode"] = "ok"
        return False


def env_of(out: str) -> dict:
    try:
        return json.loads(out)
    except (TypeError, ValueError):
        return {}


def attached(*parts, text: str = "what is this?") -> tuple[list, dict, str]:
    """(messages after extract, register, first id) for one user turn."""
    msgs = [{"role": "user", "content": [{"type": "text", "text": text}, *parts]}]
    out, att = vision.extract(msgs)
    ids = list(att["images"])
    return out, att, (ids[0] if ids else "")


def stored_png(seed: int) -> str:
    """An image generated here, through the real generate path and store."""
    with use(YAMADORI_IMAGEGEN_URL=IMG_URL):
        return images.generate("a fox", seed=seed)[0]["id"]


# --------------------------------------------------------------------------
# 1: the request shape
# --------------------------------------------------------------------------
def test_the_fixture_is_not_the_real_store():
    check(os.path.abspath(images.media_dir()) != REAL_MEDIA,
          "media store is a temp dir, not index/media", images.media_dir())
    check(model.UPSTREAM == VISION_URL, "the vision upstream is the fake")


def test_request_shape():
    png = tiny_png(1)
    _out, att, iid = attached(img_part(png))
    V_SEEN.clear()
    rec: list = []
    d = env_of(vision.run_tool({"image": iid, "question": "What animal is this?"},
                               att, rec))
    check(d.get("ok") is True and d.get("answer") == ANSWER,
          "the vision model's answer comes back as text", json.dumps(d)[:200])
    body = V_SEEN[-1]["body"] if V_SEEN else {}
    check(V_SEEN and V_SEEN[-1]["path"] == "/v1/chat/completions",
          "POST /v1/chat/completions on the model server")
    check(body.get("model") == "bonsai-vision",
          "routed to llama-swap's `bonsai-vision`", str(body.get("model")))
    msgs = body.get("messages") or []
    user = msgs[-1] if msgs else {}
    parts = user.get("content") if isinstance(user.get("content"), list) else []
    img = [p for p in parts if p.get("type") == "image_url"]
    txt = [p for p in parts if p.get("type") == "text"]
    url = (img[0].get("image_url") or {}).get("url", "") if img else ""
    check(len(img) == 1 and url.startswith("data:image/png;base64,"),
          "one image part, as a data: URI with the sniffed MIME", url[:40])
    check(url and base64.b64decode(url.split(",", 1)[1]) == png,
          "the bytes sent are the bytes attached")
    check(len(txt) == 1 and txt[0].get("text") == "What animal is this?",
          "the question is the text part", json.dumps(txt)[:120])
    check(msgs and msgs[0].get("role") == "system",
          "a system turn tells it who it is looking for")
    for k, v in tiers.VENDOR_SAMPLING.items():
        check(body.get(k) == v, f"vendor sampling via tiers.apply: {k}={v}",
              str(body.get(k)))
    cap = vision.thinking_cap("What animal is this?")
    check(body.get("reasoning_budget_tokens") == cap
          and body.get("max_tokens") == cap + tiers.A_MIN,
          "thinking = the vision cap; max_tokens = cap + A_MIN",
          f"{body.get('reasoning_budget_tokens')} {body.get('max_tokens')} cap={cap}")
    check(cap + tiers.A_MIN + vision.IMAGE_TOKENS_MAX <= vision.DEFAULT_CONTEXT,
          "thinking + answer + the largest image fit -c 16384",
          str(cap + tiers.A_MIN + vision.IMAGE_TOKENS_MAX))
    check(body.get("reasoning_effort") in tiers.FALLBACK_EFFORTS,
          "an effort the template accepts", str(body.get("reasoning_effort")))
    check(not any(k.startswith("_") for k in body) and not body.get("tools"),
          "no underscored bookkeeping and no tools go to the vision model",
          str([k for k in body if k.startswith("_")]))
    r = rec[-1] if rec else {}
    check(r.get("ok") is True and r.get("image") == iid
          and r.get("source") == "attached" and r.get("format") == "png"
          and r.get("bytes") == len(png) and r.get("prompt_tokens") == 1180
          and r.get("completion_tokens") == 311
          and isinstance(r.get("seconds"), (int, float)),
          "the call is recorded: image, source, format, bytes, tokens, seconds",
          json.dumps(r))
    blob = json.dumps(rec)
    check("What animal" not in blob and base64.b64encode(png).decode() not in blob,
          "the record holds neither the question nor the image")

    jpg = b"\xff\xd8\xff\xe0" + b"\x00" * 64
    _o, att2, jid = attached(img_part(jpg, "image/png"))
    vision.run_tool({"image": jid, "question": "q?"}, att2, [])
    u = V_SEEN[-1]["body"]["messages"][-1]["content"][0]["image_url"]["url"]
    check(u.startswith("data:image/jpeg;base64,"),
          "the MIME sent is sniffed from the bytes, not the client's claim", u[:30])


# --------------------------------------------------------------------------
# 2: every failure carries the next step
# --------------------------------------------------------------------------
def _shape(d: dict, code: str, name: str, retryable: bool) -> None:
    if not check(d.get("ok") is False and d.get("error") == code,
                 f"{name}: {code}", json.dumps(d)[:200]):
        return
    check(d.get("tool") == "describe_image", f"{name}: names the tool")
    check(d.get("retryable") is retryable,
          f"{name}: retryable={retryable} as a fact", str(d.get("retryable")))
    rem = d.get("remedies") or []
    check(bool(rem) and all(r.get("fixable_by") in ("agent", "user", "operator")
                            and r.get("action") for r in rem),
          f"{name}: remedies name an owner and an action", json.dumps(rem)[:160])
    check("Nothing" in (d.get("reason") or "") or "not a description" in
          (d.get("reason") or ""), f"{name}: says nothing was looked at",
          (d.get("reason") or "")[:120])


def test_every_failure_names_itself_and_the_next_step():
    _o, att, iid = attached(img_part(tiny_png(2)))
    ok_args = {"image": iid, "question": "describe it"}

    def run(args=None, a=None):
        return env_of(vision.run_tool(ok_args if args is None else args,
                                      att if a is None else a, []))

    _shape(run(["not", "a", "dict"]), "BAD_ARGUMENTS", "non-object args", True)
    _shape(run({"image": iid}), "BAD_ARGUMENTS", "no question", True)
    _shape(run({"image": iid, "question": "x" * 5000}), "BAD_ARGUMENTS",
           "question too long", True)
    _shape(run({"question": "q"}), "BAD_ARGUMENTS", "no image", True)
    with use(YAMADORI_VISION="0"):
        _shape(run(), "VISION_OFF", "vision switched off", False)
    d = run({"image": "image-ffffffffff", "question": "q"})
    _shape(d, "UNKNOWN_IMAGE", "an id not in this conversation", True)
    check(iid in json.dumps(d.get("available")),
          "UNKNOWN_IMAGE names what IS available", json.dumps(d.get("available")))
    _shape(run({"image": "https://example.com/cat.png", "question": "q"}),
           "IMAGE_URL_NOT_ALLOWED", "a URL", False)

    sha = stored_png(31)
    exp = int(time.time()) + 600
    good = images.signed_url(sha, "https://ai.example.test")
    bad_sig = re.sub(r"sig=([0-9a-f])", lambda m: "sig=" + ("0" if m.group(1) != "0" else "1"), good)
    _shape(run({"image": bad_sig, "question": "q"}, vision.empty()),
           "IMAGE_LINK_INVALID", "a tampered signature", False)
    past = int(time.time()) - 5
    expired = f"https://x/media/{sha}.png?exp={past}&sig={images.sign(sha, past)}"
    d = run({"image": expired, "question": "q"}, vision.empty())
    _shape(d, "IMAGE_LINK_INVALID", "an expired link", False)
    check("expired" in d.get("reason", ""), "and says it expired", d.get("reason", "")[:100])
    missing = "d" * 64
    gone = f"https://x/media/{missing}.png?exp={exp}&sig={images.sign(missing, exp)}"
    _shape(run({"image": gone, "question": "q"}, vision.empty()),
           "IMAGE_NOT_FOUND", "a signed link to a file no longer stored", False)

    _o, att_bad, bid = attached(img_part(b"%PDF-1.7 not an image at all"))
    _shape(run({"image": bid, "question": "q"}, att_bad), "NOT_AN_IMAGE",
           "attached bytes that are not an image", False)
    _o, att_b64, b64id = attached({"type": "image_url",
                                   "image_url": {"url": "data:image/png;base64,@@@@"}})
    _shape(run({"image": b64id, "question": "q"}, att_b64), "NOT_AN_IMAGE",
           "attached data that is not base64", False)
    with use(YAMADORI_VISION_MAX_BYTES="1000"):
        _o, att_big, big = attached(img_part(tiny_png(3, pad=5000)))
        _shape(run({"image": big, "question": "q"}, att_big), "IMAGE_TOO_LARGE",
               "an attachment over the limit", False)
        check(att_big["images"][big]["b64"] is None,
              "an oversized attachment is not kept")
    with use(YAMADORI_VISION_MAX_BYTES=str(vision.DEFAULT_MAX_BYTES)):
        check(vision.max_bytes() == 10 * 1024 * 1024,
              "the default limit is llama-server's own 10 MB")

    lane = admission.image_lane(timeout=0)
    held = lane.__enter__()
    try:
        d = run()
    finally:
        lane.__exit__(None, None, None)
    check(held, "the test holds the image lane")
    _shape(d, "VISION_BUSY", "the A4000 lane is busy", True)

    for mode, code, retry, name in (
            ("loading", "VISION_LOADING", True, "the model is still loading"),
            ("unavailable", "VISION_UNAVAILABLE", False, "failed load / evicted"),
            ("no_projector", "VISION_NO_PROJECTOR", False, "started without --mmproj"),
            ("oom", "VISION_OUT_OF_MEMORY", False, "out of memory on the A4000"),
            ("context", "VISION_CONTEXT_FULL", False, "image + question over -c"),
            ("reject", "VISION_REJECTED_IMAGE", False, "the server cannot decode it"),
            ("notfound", "VISION_NOT_CONFIGURED", False, "llama-swap has no such model"),
            ("length", "VISION_BUDGET", True, "a length finish"),
            ("empty", "VISION_EMPTY", False, "an empty answer"),
            ("garbage", "VISION_FAILED", False, "an answer that is not JSON")):
        V_MODE["mode"] = mode
        _shape(run(), code, name, retry)
    V_MODE["mode"] = "ok"
    with use(YAMADORI_VISION_TIMEOUT="0.5"):
        V_MODE["mode"] = "slow"
        t0 = time.time()
        _shape(run(), "VISION_TIMEOUT", "a stuck vision server", False)
        check(time.time() - t0 < 2.5, "a stuck server does not hang the caller",
              f"{time.time() - t0:.1f}s")
    old = model.UPSTREAM
    model.UPSTREAM = DEAD
    try:
        _shape(run(), "VISION_DOWN", "llama-swap not answering", False)
    finally:
        model.UPSTREAM = old
    rec: list = []
    vision.run_tool({"image": "nope", "question": "q"}, att, rec)
    check(rec and rec[-1].get("ok") is False and rec[-1].get("error") == "UNKNOWN_IMAGE"
          and "ms" in rec[-1], "a failed call is recorded with its code",
          json.dumps(rec))


# --------------------------------------------------------------------------
# 3: the security contract
# --------------------------------------------------------------------------
class _Watch:
    """Record every URL urllib opens and every file opened, for a block."""

    def __enter__(self):
        self.urls: list[str] = []
        self.files: list[str] = []
        self._urlopen = urllib.request.urlopen
        self._open = builtins.open

        def urlopen(req, *a, **k):
            self.urls.append(req.full_url if hasattr(req, "full_url") else str(req))
            return self._urlopen(req, *a, **k)

        def opener(f, *a, **k):
            self.files.append(os.path.abspath(str(f)) if isinstance(f, (str, bytes, os.PathLike)) else str(f))
            return self._open(f, *a, **k)
        urllib.request.urlopen = urlopen
        builtins.open = opener
        return self

    def __exit__(self, *exc):
        urllib.request.urlopen = self._urlopen
        builtins.open = self._open
        return False


def test_security_contract():
    # A real PNG on disk, OUTSIDE the media store, that a path could name.
    outside = os.path.join(_TMP, "private.png")
    with open(outside, "wb") as f:
        f.write(tiny_png(99))
    sha = stored_png(41)
    stored_file = images.media_path(sha)
    key_file = images._secret_path()
    attacker = "http://127.0.0.1:9/steal.png"
    V_SEEN.clear()
    with _Watch() as w:
        for ref in (attacker, "https://example.com/x.png", "ftp://h/x.png",
                    "file:///" + outside.replace("\\", "/"), outside,
                    "../index/media_url.key", key_file, stored_file,
                    "index/media/" + sha + ".png", sha, sha + ".png",
                    "/media/" + sha + ".png", data_uri(tiny_png(5)),
                    "C:\\Windows\\win.ini", "/etc/passwd", "image-" + sha[:10]):
            d = env_of(vision.run_tool({"image": ref, "question": "q"},
                                       vision.empty(), []))
            check(d.get("ok") is False and d.get("error") in
                  ("UNKNOWN_IMAGE", "IMAGE_URL_NOT_ALLOWED"),
                  f"refused with an empty register: {ref[:48]!r}",
                  json.dumps(d)[:160])
    check(not w.urls, "no URL was fetched for any of them", str(w.urls[:3]))
    check(outside not in w.files and os.path.abspath(key_file) not in w.files
          and os.path.abspath(stored_file) not in w.files,
          "no named path was opened -- not outside, not the key, not even "
          "the store's own file without a capability",
          str([f for f in w.files if "yamadori_test_vision" in f][:4]))
    check(not V_SEEN, "the vision server was never called for a refused image")

    # A client's image LINK is not forwarded: llama-server would download it.
    V_SEEN.clear()
    with _Watch() as w:
        msgs, att, lid = attached({"type": "image_url",
                                   "image_url": {"url": attacker}})
        d = env_of(vision.run_tool({"image": lid, "question": "q"}, att, []))
    check(d.get("error") == "IMAGE_URL_NOT_ALLOWED",
          "a linked image is refused as a link", json.dumps(d)[:160])
    check(not w.urls and not V_SEEN, "and nothing fetched it: not us, not the "
                                     "vision server", str(w.urls))
    check(attacker not in json.dumps(msgs),
          "the link is not passed on to the text model either")

    # Our own images: only with a capability.
    link = images.signed_url(sha, "https://ai.example.test")
    V_SEEN.clear()
    with _Watch() as w:
        d = env_of(vision.run_tool({"image": link, "question": "q"}, vision.empty(), []))
    check(d.get("ok") is True and d.get("source") == "generated",
          "a valid signed /media link is read from the store", json.dumps(d)[:160])
    check(all(u.startswith(VISION_URL) for u in w.urls) and len(w.urls) == 1,
          "and the only URL opened is the vision server's", str(w.urls))
    check("ai.example.test" not in json.dumps(V_SEEN[-1]["body"]) if V_SEEN else False,
          "the vision server gets bytes, never our link")
    md = f"![a fox]({link})"
    d = env_of(vision.run_tool({"image": md, "question": "q"}, vision.empty(), []))
    check(d.get("ok") is True, "the markdown line generate_image returned works too")
    att = vision.empty()
    vision.note_generated(json.dumps({"ok": True, "url": link}), att)
    d = env_of(vision.run_tool({"image": sha, "question": "q"}, att, []))
    check(d.get("ok") is True, "a bare sha works once this request made it")
    _m, att2 = vision.extract([{"role": "assistant",
                                "content": f"Here it is.\n\n{md}"}])
    check(sha in att2["media"], "a signed link in the conversation makes the "
                                "sha known")
    forged = f"![x](https://x/media/{sha}.png?exp=9999999999&sig={'0' * 64})"
    _m, att3 = vision.extract([{"role": "assistant", "content": forged}])
    check(sha not in att3["media"], "a forged signature in the conversation does not")
    V_SEEN.clear()
    for ref, why in ((sha, "an unknown sha"),
                     (f"/media/{sha}.png", "a link without its signature")):
        d = env_of(vision.run_tool({"image": ref, "question": "q"}, vision.empty(), []))
        check(d.get("error") == "UNKNOWN_IMAGE",
              f"another conversation's image by {why}: refused", json.dumps(d)[:120])
    check(not V_SEEN, "and the vision server was never called for them")
    src = open(os.path.join(HERE, "vision.py"), encoding="utf-8").read()
    check("urlopen" not in src and "open(" not in src.replace("urllib.parse", ""),
          "vision.py itself contains no urlopen and no open(): every read is "
          "images.png_bytes, every request is model.chat")


# --------------------------------------------------------------------------
# 4: offered wherever generate_image is, and with an attachment
# --------------------------------------------------------------------------
def _names(out: dict) -> list[str]:
    return [t.get("function", {}).get("name") for t in out.get("tools") or []]


def test_offered_everywhere_generate_image_is():
    import proxy
    import shomen
    txt = [{"role": "user", "content": "draw me a lighthouse at dusk"}]
    pic = [{"role": "user", "content": [{"type": "text", "text": "what is this?"},
                                        img_part(tiny_png(6))]}]
    for effort in ("minimal", "low", "medium", "high", "max"):
        with use(YAMADORI_IMAGEGEN_URL=IMG_URL):
            n = _names(proxy.prepare({"model": "yamadori", "messages": txt,
                                      "reasoning_effort": effort}))
        check("generate_image" in n and "describe_image" in n,
              f"image server configured, tier {effort}: both offered", str(n))
        with use(YAMADORI_IMAGEGEN_URL=None):
            n = _names(proxy.prepare({"model": "yamadori", "messages": pic,
                                      "reasoning_effort": effort}))
        check("describe_image" in n and "generate_image" not in n,
              f"no image server, an attached image, tier {effort}: "
              f"describe_image offered", str(n))
        with use(YAMADORI_IMAGEGEN_URL=None):
            n = _names(proxy.prepare({"model": "yamadori", "messages": txt,
                                      "reasoning_effort": effort}))
        check("describe_image" not in n,
              f"nothing to look at, tier {effort}: not offered", str(n))
    with use(YAMADORI_IMAGEGEN_URL=IMG_URL, YAMADORI_VISION="0"):
        n = _names(proxy.prepare({"model": "yamadori", "messages": pic,
                                  "reasoning_effort": "low"}))
        check("describe_image" not in n and "generate_image" in n,
              "YAMADORI_VISION=0 withholds it, attachment or not", str(n))
    with use(YAMADORI_IMAGEGEN_URL=IMG_URL):
        dt = [t["function"]["name"] for t in proxy.deep_thinking_tools()]
        check("describe_image" in dt and "generate_image" in dt,
              "deep thinking gets both", str(dt))
    with use(YAMADORI_IMAGEGEN_URL=None):
        _m, att = vision.extract(pic)
        dt = [t["function"]["name"] for t in proxy.deep_thinking_tools(att)]
        check("describe_image" in dt,
              "deep thinking gets it for an attached image with no image server",
              str(dt))
    mine = {"type": "function", "function": {
        "name": "describe_image", "description": "the client's own",
        "parameters": {"type": "object", "properties": {}}}}
    with use(YAMADORI_IMAGEGEN_URL=IMG_URL):
        out = proxy.prepare({"model": "yamadori", "messages": pic,
                             "reasoning_effort": "low", "tools": [mine]})
    got = [t for t in out["tools"] if t["function"]["name"] == "describe_image"]
    check(len(got) == 1 and got[0]["function"]["description"] == "the client's own",
          "a client's own describe_image wins")
    check("describe_image" in proxy.OUR_NAMES and "describe_image" in proxy._STATEFUL,
          "the proxy runs it as its own, and never serves it from the repeat cache")

    s = shomen.SYSTEM
    para = s[s.index("generate_image"):s.index("WRITE IN PLAIN")]
    check("describe_image" in para and "draw it again" in para,
          "deep thinking is told to look at its drawing and refine it", para[:200])
    check(not re.search(r"\b(never|do not|don't|cannot)\b", para, re.I),
          "that instruction is positive: no prohibition (AGENTS.md)", para)

    desc = vision.TOOL["function"]["description"]
    check(desc.startswith("Answers a question about what is IN an image"),
          "the description leads with the question it answers", desc[:60])
    check("generate_image" in desc and "turns words into" in desc,
          "and contrasts itself with generate_image")
    for phrase in ("what is in this image", "describe this screenshot",
                   "read the text in this picture", "check the image you just drew",
                   "image-", "It sees images only",
                   "client's own file tools"):
        check(phrase in desc, f"it lists the trigger {phrase!r}")
    name = vision.TOOL["function"]["name"]
    check(re.fullmatch(r"[a-z]+(_[a-z]+)+", name) and name.split("_")[0] == "describe",
          "snake_case, verb first, spelled out", name)


# --------------------------------------------------------------------------
# 5-6: attached images through a real chat turn, and x_yamadori
# --------------------------------------------------------------------------
def _call(name: str, args: dict, cid: str = "c1") -> dict:
    return {"calls": [{"id": cid, "type": "function", "function": {
        "name": name, "arguments": json.dumps(args)}}]}


def _last_tool(body: dict) -> dict:
    for m in reversed(body.get("messages") or []):
        if m.get("role") == "tool":
            return env_of(m.get("content"))
    return {}


def _with_llm(fn):
    import proxy
    old = proxy.UPSTREAM
    proxy.UPSTREAM = LLM_URL
    try:
        return fn(proxy)
    finally:
        proxy.UPSTREAM = old


def test_placeholders():
    png = tiny_png(7)
    msgs = [{"role": "system", "content": "sys"},
            {"role": "user", "content": [
                {"type": "text", "text": "compare these"},
                img_part(png),
                {"type": "image_url", "image_url": {"url": "https://example.com/y.png"}},
                img_part(b"GIF89a" + b"\x00" * 40, "image/gif"),
                img_part(b"not an image"),
                {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                             "data": base64.b64encode(tiny_png(8)).decode()}}]}]
    out, att = vision.extract(msgs)
    parts = out[1]["content"]
    check(all(p.get("type") == "text" for p in parts),
          "every image and media part became text: no image part reaches the "
          "text model", json.dumps([p.get("type") for p in parts]))
    check(len(parts) == len(msgs[1]["content"]),
          "one placeholder per part: nothing dropped")
    ids = list(att["images"])
    ok_ids = [i for i, e in att["images"].items() if not e["error"]]
    check(len(ok_ids) == 3, "PNG, GIF and the Anthropic-shaped PNG are readable",
          json.dumps(vision.summary(att)))
    text = "\n".join(p["text"] for p in parts)
    for i in ok_ids:
        check(f'describe_image with image "{i}"' in text,
              f"the placeholder names {i} and the call to make")
    check("another site" in text and "attach the image file" in text,
          "the link's placeholder asks for the file")
    check("not an image this server can read" in text,
          "the non-image's placeholder says so")
    check("input_audio" in text and "only text and images" in text,
          "an audio part is replaced, not sent to a server that would refuse it")
    check(out[0] is msgs[0], "text messages are passed through untouched")
    check(msgs[1]["content"][1]["type"] == "image_url",
          "the client's own message list is not mutated")
    check(all(re.fullmatch(r"image-[0-9a-f]{10}", i) for i in ids),
          "ids are image-<10 hex>", str(ids))
    _o2, att_again = vision.extract(msgs)
    check(list(att_again["images"]) == ids,
          "the same attachment gets the same id on every turn (content hash)")
    s = json.dumps(vision.summary(att))
    check(base64.b64encode(png).decode() not in s and "b64" not in s,
          "the summary carries no image data", s[:200])
    json.dumps(att)
    check(True, "the register is JSON-safe (fan-out deep-copies payloads)")

    plain = [{"role": "user", "content": "hello"},
             {"role": "user", "content": [{"type": "text", "text": "a list"}]}]
    same, empty_att = vision.extract(plain)
    check(same is plain and not empty_att["images"],
          "a text-only request is the very same list: the prefix stays "
          "byte-identical")

    with use(YAMADORI_VISION="0"):
        out_off, _a = vision.extract([{"role": "user", "content": [img_part(png)]}])
    t = out_off[0]["content"][0]["text"]
    check("switched off" in t and "describe_image" not in t,
          "with vision off, the placeholder says so instead of naming the tool", t)


def test_budget_is_not_floored_by_an_image():
    import proxy
    big = tiny_png(9, pad=600_000)
    base_msgs = [{"role": "user", "content": [{"type": "text", "text": "what is this?"}]}]
    with_img = [{"role": "user", "content": [{"type": "text", "text": "what is this?"},
                                             img_part(big)]}]
    a = proxy.prepare({"model": "yamadori", "messages": base_msgs,
                       "reasoning_effort": "low"})
    b = proxy.prepare({"model": "yamadori", "messages": with_img,
                       "reasoning_effort": "low"})
    ta, tb = a.get("reasoning_budget_tokens") or 0, b.get("reasoning_budget_tokens") or 0
    check(tb > 10 * tiers.MIN_THINKING and abs(ta - tb) < 500,
          "a 600 KB attachment costs the thinking budget only its placeholder",
          f"text-only {ta}, with image {tb}")
    check(b["_seen_messages"][0]["content"][1]["type"] == "text",
          "the main model's messages carry the placeholder")
    check(a["_seen_messages"] is not None, "prepare exposes the messages it saw")


def test_a_chat_turn_that_draws_then_looks():
    """complete(): generate_image, then describe_image on its url, then an
    answer. x_yamadori records both."""
    LLM_SENT.clear()
    V_SEEN.clear()
    LLM_SCRIPT[:] = [
        lambda b: _call("generate_image", {"prompt": "a fox", "seed": 51}),
        lambda b: _call("describe_image", {"image": _last_tool(b).get("url", ""),
                                           "question": "Does this show a fox?"}, "c2"),
        lambda b: {"content": "It shows a fox."}]
    with use(YAMADORI_IMAGEGEN_URL=IMG_URL):
        d = _with_llm(lambda proxy: proxy.complete({
            "model": "yamadori", "reasoning_effort": "low",
            "messages": [{"role": "user", "content": "draw a fox and check it"}],
            "_public_base": "https://ai.example.test"}))
    first = [t["function"]["name"] for t in (LLM_SENT[0].get("tools") or [])] if LLM_SENT else []
    check("generate_image" in first and "describe_image" in first,
          "the model was offered both", str(first))
    res = _last_tool(LLM_SENT[2]) if len(LLM_SENT) > 2 else {}
    check(res.get("ok") is True and res.get("answer") == ANSWER
          and res.get("source") == "generated",
          "the proxy ran describe_image on what it drew", json.dumps(res)[:200])
    check(V_SEEN and V_SEEN[-1]["body"]["model"] == "bonsai-vision",
          "the vision call went to bonsai-vision")
    x = d.get("x_yamadori") or {}
    v = x.get("vision") or []
    check(len(v) == 1 and v[0].get("ok") and v[0].get("source") == "generated"
          and len(v[0].get("image", "")) == 16 and v[0].get("format") == "png",
          "x_yamadori.vision records the call: ok, source, sha prefix, format",
          json.dumps(v))
    check(len(x.get("images") or []) == 1 and x["images"][0].get("ok"),
          "x_yamadori.images records the drawing", json.dumps(x.get("images")))
    check(x.get("attachments") == [], "no attachments on this turn")
    check("Does this show" not in json.dumps(x),
          "the question is not in x_yamadori")
    check(d["choices"][0]["message"]["content"] == "It shows a fox.",
          "the answer comes back")


def test_a_chat_turn_with_an_attached_image():
    png = tiny_png(12)
    LLM_SENT.clear()
    V_SEEN.clear()

    def look(b):
        text = json.dumps(b.get("messages"))
        m = re.search(r"image-[0-9a-f]{10}", text)
        return _call("describe_image", {"image": m.group(0) if m else "none",
                                        "question": "What is in the picture?"})
    LLM_SCRIPT[:] = [look, lambda b: {"content": "A fox."}]
    with use(YAMADORI_IMAGEGEN_URL=None):
        d = _with_llm(lambda proxy: proxy.complete({
            "model": "yamadori", "reasoning_effort": "minimal",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "what is this?"}, img_part(png),
                img_part(b"junk")]}]}))
    sent = json.dumps(LLM_SENT[0]) if LLM_SENT else ""
    check(LLM_SENT and '"image_url"' not in sent
          and base64.b64encode(png).decode() not in sent,
          "the text model received no image part and no image bytes")
    check("describe_image" in [t["function"]["name"] for t in LLM_SENT[0].get("tools") or []],
          "describe_image was offered at tier minimal with no image server")
    res = _last_tool(LLM_SENT[1]) if len(LLM_SENT) > 1 else {}
    check(res.get("ok") is True and res.get("source") == "attached",
          "the model looked at the attachment by its id", json.dumps(res)[:200])
    x = d.get("x_yamadori") or {}
    att = x.get("attachments") or []
    check(len(att) == 2 and {a.get("error") for a in att} == {None, "NOT_AN_IMAGE"},
          "x_yamadori.attachments records both, with the bad one's code",
          json.dumps(att))
    check(all("b64" not in a for a in att) and base64.b64encode(png).decode()
          not in json.dumps(d), "no image data anywhere in the response")
    v = x.get("vision") or []
    check(len(v) == 1 and v[0].get("ok") and v[0].get("source") == "attached"
          and v[0].get("bytes") == len(png), "x_yamadori.vision records it",
          json.dumps(v))


def test_streamed_turn_with_an_attached_image():
    png = tiny_png(13)
    LLM_SENT.clear()

    def look(b):
        m = re.search(r"image-[0-9a-f]{10}", json.dumps(b.get("messages")))
        return _call("describe_image", {"image": m.group(0) if m else "none",
                                        "question": "What is it?"})
    LLM_SCRIPT[:] = [look, lambda b: {"content": "A fox."}]

    def run(proxy):
        old = proxy.PREAMBLE
        proxy.PREAMBLE = False
        try:
            return [e for e in proxy.stream_body({
                "model": "yamadori", "reasoning_effort": "low", "stream": True,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "what is this?"}, img_part(png)]}]},
                "yamadori")]
        finally:
            proxy.PREAMBLE = old
    with use(YAMADORI_IMAGEGEN_URL=None):
        events = _with_llm(run)
    last = {}
    for b in events:
        if b.startswith(b"data: {"):
            e = json.loads(b[6:].decode())
            if e.get("x_yamadori"):
                last = e["x_yamadori"]
    v = last.get("vision") or []
    check(len(v) == 1 and v[0].get("ok") and v[0].get("source") == "attached",
          "the streamed path runs it and records it", json.dumps(v))
    check(len(last.get("attachments") or []) == 1,
          "and records the attachment", json.dumps(last.get("attachments")))


def _link_part(url: str) -> dict:
    return {"type": "image_url", "image_url": {"url": url}}


def test_our_signed_link_is_an_attachment():
    """LIVE PROBLEM, 2026-09-24: Hermes' vision_analyze sends its vision
    call to the main provider -- us -- as one exchange with an image_url
    that is often OUR signed /media link. It read as a utility side call
    (the bare text model, which cannot see) and the link was only a sha
    placeholder. Now: our link (signature and expiry verified, host ours or
    loopback) is read from the media store as an ATTACHMENT, never fetched;
    a request carrying any image is never a utility call; external links
    are still never fetched, and the placeholder says so."""
    import selection
    sha = stored_png(41)
    hosts = vision.our_hosts("https://ai.example.test")
    good = images.signed_url(sha, "https://ai.example.test")
    with _Watch() as w:
        out, att = vision.extract([{"role": "user", "content": [
            {"type": "text", "text": "Describe this image."},
            _link_part(good)]}], hosts)
    ids = list(att["images"])
    e = att["images"][ids[0]] if ids else {}
    ph = out[0]["content"][1].get("text", "") if ids else ""
    check(len(ids) == 1 and e.get("error") is None and e.get("source") ==
          "media" and e.get("format") == "png"
          and base64.b64decode(e.get("b64") or "") == images.png_bytes(sha)
          and f'describe_image with image "{ids[0]}"' in ph
          and not w.urls,
          "our valid signed link: read from the media store as an attachment "
          "(the stored bytes), nothing fetched", json.dumps(
              {"summary": vision.summary(att), "urls": w.urls, "ph": ph}))
    V_SEEN.clear()
    with _Watch() as w:
        d = env_of(vision.run_tool({"image": ids[0] if ids else "none",
                                    "question": "q"}, att, []))
    check(d.get("ok") is True and all(u.startswith(VISION_URL)
                                      for u in w.urls),
          "describe_image looks at it by its attachment id; the only URL "
          "opened is the vision server's", json.dumps(d)[:200])
    loop = images.signed_url(sha, "http://127.0.0.1:1234")
    rel = loop[len("http://127.0.0.1:1234"):]
    for url, what in ((loop, "a loopback"), (rel, "a relative")):
        _o, a2 = vision.extract([{"role": "user", "content": [
            _link_part(url)]}], hosts)
        check([x.get("error") for x in a2["images"].values()] == [None],
              f"{what} signed link is ours too", json.dumps(vision.summary(a2)))
    past = int(time.time()) - 5
    expired = (f"https://ai.example.test/media/{sha}.png?exp={past}"
               f"&sig={images.sign(sha, past)}")
    tampered = re.sub(r"sig=([0-9a-f])", lambda m: "sig=" + (
        "0" if m.group(1) != "0" else "1"), good)
    for url, what, word in ((expired, "an expired", "expired"),
                            (tampered, "a tampered", "does not match")):
        with _Watch() as w:
            o3, a3 = vision.extract([{"role": "user", "content": [
                _link_part(url)]}], hosts)
        iid = next(iter(a3["images"]), "")
        e3 = a3["images"].get(iid) or {}
        ph3 = o3[0]["content"][0].get("text", "")
        d3 = env_of(vision.run_tool({"image": iid, "question": "q"}, a3, []))
        check(e3.get("error") == "IMAGE_LINK_INVALID" and not e3.get("b64")
              and word in ph3 and "not read" in ph3 and not w.urls
              and d3.get("error") == "IMAGE_LINK_INVALID",
              f"{what} signed link: not read, the placeholder and the tool "
              f"say why, nothing fetched", json.dumps({"e": e3, "ph": ph3,
                                                       "tool": d3})[:500])
    foreign = good.replace("https://ai.example.test", "https://evil.example")
    with _Watch() as w:
        o4, a4 = vision.extract([{"role": "user", "content": [
            _link_part(foreign)]}], hosts)
    check([x.get("error") for x in a4["images"].values()]
          == ["IMAGE_URL_NOT_ALLOWED"] and not w.urls,
          "our signature on ANOTHER host is an external link: not read, not "
          "fetched", json.dumps(vision.summary(a4)))
    with _Watch() as w:
        o5, a5 = vision.extract([{"role": "user", "content": [
            _link_part("https://example.com/cat.png")]}], hosts)
    ph5 = o5[0]["content"][0].get("text", "")
    check(not w.urls and "NOT downloaded" in ph5
          and "never fetches image links" in ph5
          and "could not see this image" in ph5,
          "an external image URL is never fetched, and the placeholder tells "
          "the model to say so plainly", ph5)
    png = tiny_png(42)
    _o6, a6 = vision.extract([{"role": "user", "content": [img_part(png)]}],
                             hosts)
    check([(x.get("source"), x.get("error")) for x in a6["images"].values()]
          == [("attached", None)],
          "a data: URI is an attachment, as before",
          json.dumps(vision.summary(a6)))

    # NEVER A UTILITY CALL. Hermes' shape: one exchange, no tools, and a
    # closed reply form -- which alone would read as a side call.
    hermes = [{"role": "user", "content": [
        {"type": "text", "text": "Is this image safe to show? Respond with "
                                 "exactly one word: APPROVE, DENY, or "
                                 "ESCALATE"},
        _link_part(good)]}]
    u = selection.utility_call(hermes, [])
    u_txt = selection.utility_call([{"role": "user", "content":
                                     hermes[0]["content"][0]["text"]}], [])
    check(u["utility"] is False and u["signals"].get("image") is True
          and u_txt["utility"] is True,
          "an image makes it a task, not a side call (the same words without "
          "an image are one)", json.dumps([u, u_txt])[:400])
    check(selection.IMAGE_PART_TYPES == frozenset(vision.IMAGE_PARTS),
          "selection's image part types are vision's")
    import proxy
    forced = proxy.utility_of({"messages": hermes,
                               "_features": json.dumps({"utility": True})})
    check(forced["utility"] is False,
          "not even a header forcing utility sends an image to the bare text "
          "model", json.dumps(forced)[:300])
    with use(YAMADORI_IMAGEGEN_URL=None):
        out7 = proxy.prepare({"model": "yamadori", "messages": hermes,
                              "_public_base": "https://ai.example.test"})
    names = [t["function"]["name"] for t in out7.get("tools") or []]
    sel = out7.get("_selection") or {}
    check("describe_image" in names and not sel.get("utility")
          and (out7.get("_route") or {}).get("class") != "utility",
          "through prepare: not routed utility, and describe_image is "
          "offered for our link", json.dumps({"tools": names,
                                              "route": out7.get("_route")})[:400])


def main() -> int:
    tests = (test_the_fixture_is_not_the_real_store,
             test_our_signed_link_is_an_attachment,
             test_request_shape,
             test_every_failure_names_itself_and_the_next_step,
             test_security_contract,
             test_offered_everywhere_generate_image_is,
             test_placeholders,
             test_budget_is_not_floored_by_an_image,
             test_a_chat_turn_that_draws_then_looks,
             test_a_chat_turn_with_an_attached_image,
             test_streamed_turn_with_an_attached_image)
    try:
        for fn in tests:
            print(f"\n--- {fn.__name__} ---")
            n0 = len(_results)
            try:
                fn()
            except Exception:                                    # noqa: BLE001
                check(False, f"{fn.__name__} itself raised",
                      traceback.format_exc().strip().split("\n")[-1])
            for ok, name, detail in _results[n0:]:
                print(("  pass  " if ok else "  FAIL  ") + name
                      + (f"   <- {detail}" if not ok and detail else ""))
    finally:
        for s in _servers:
            s.shutdown()
            s.server_close()
    passed = sum(1 for ok, _, _ in _results if ok)
    total = len(_results)
    print(f"\n{'=' * 70}\n  {passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
