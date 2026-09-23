#!/usr/bin/env python
"""Image generation, end to end against a fake image server. No GPU.

WHAT THIS GATES

`mcp/images.py`, the `generate_image` model tool, POST /v1/images/generations
and GET /media/<sha>.png. The promises, each asserted below:

  1. The request the image server receives is the one documented in
     docs/IMAGEGEN.md: /sdapi/v1/txt2img with model, prompt, width, height,
     steps, seed, batch_size.
  2. Every image is stored once, by the sha of its bytes, with metadata beside
     it: prompt, seed, size, steps, model, seconds.
  3. Every failure names itself and carries the next step: not configured,
     down, out of memory, busy, timeout, empty, bad arguments. None hangs,
     none raises out of the tool, none reads like an image.
  4. /v1/images/generations is authenticated like every /v1 route and answers
     the OpenAI shape, `url` by default and `b64_json` only when asked.
  5. /media is a CAPABILITY URL: a signed link works with no Authorization
     header (a browser rendering `![](url)` sends none); a tampered sig, an
     expired exp and a sig moved to another sha are each 403; `../` and
     non-hex ids never touch the filesystem.
  6. generate_image is offered only when an image server is configured, and
     at tier `low` and up even when the code tools are withheld; never at
     `minimal`, never to deep thinking.
  7. A chat turn that calls it gets a result carrying the signed URL on the
     client's public base and a markdown line, and x_yamadori records it.
  8. The markdown the tool hands the model is what a markdown renderer turns
     into an <img> whose src is that URL, and the URL's path ends in `.png`
     before the query string (docs/IMAGEGEN.md, "How Hermes shows it").

ISOLATION

Every store is a temp path set BEFORE import: accounts, corpus, nebari, rings,
packages, media and the URL-signing key. The LLM upstream, Laya and the model
server point at a closed port, so nothing here can reach the live stack. The
fake image server is shut down at the end; nothing is left listening.
"""
from __future__ import annotations

import base64
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
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

_TMP = tempfile.mkdtemp(prefix="yamadori_test_images_")


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
for k in ("YAMADORI_IMAGEGEN_URL", "YAMADORI_PUBLIC_BASE",
          "YAMADORI_IMAGEGEN_TIMEOUT"):
    os.environ.pop(k, None)

import accounts  # noqa: E402
import admission  # noqa: E402
import images  # noqa: E402

REAL_MEDIA = os.path.abspath(os.path.join(HERE, "..", "index", "media"))
REAL_KEY = os.path.abspath(os.path.join(HERE, "..", "index", "media_url.key"))

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


# --------------------------------------------------------------------------
# The fake image server: sd-server's /sdapi/v1/txt2img, with failure modes
# --------------------------------------------------------------------------
def tiny_png(seed: int) -> bytes:
    """A valid 2x2 RGB PNG whose colour depends on the seed, so different
    seeds give different bytes and different shas."""
    r, g, b = seed % 256, (seed // 256) % 256, 7
    raw = b"".join(b"\x00" + bytes([r, g, b]) * 2 for _ in range(2))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))
    return (images.PNG_MAGIC
            + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


MODE = {"mode": "ok"}
SEEN: list[dict] = []


class _Img(BaseHTTPRequestHandler):
    def do_POST(self):                                           # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN.append({"path": self.path, "body": body})
        m = MODE["mode"]
        if m == "slow":
            time.sleep(3)
        if m == "oom":
            data, code = json.dumps({"error": "server_error", "message":
                                     "CUDA error: out of memory"}).encode(), 500
        elif m == "crashed":
            data, code = b"upstream process exited", 502
        elif m == "empty":
            data, code = json.dumps({"images": []}).encode(), 200
        else:
            n = int(body.get("batch_size") or 1)
            imgs = [base64.b64encode(tiny_png(int(body["seed"]) + i)).decode()
                    for i in range(n)]
            data, code = json.dumps({"images": imgs, "parameters": body,
                                     "info": "{}"}).encode(), 200
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except OSError:
            pass                    # the client gave up (the timeout test)

    def log_message(self, *a):                                   # noqa: D102
        pass


_srv = ThreadingHTTPServer(("127.0.0.1", 0), _Img)
threading.Thread(target=_srv.serve_forever, daemon=True).start()
IMG_URL = f"http://127.0.0.1:{_srv.server_address[1]}"


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
        MODE["mode"] = "ok"
        return False


def err_of(fn, *a, **k):
    try:
        fn(*a, **k)
    except images.ImageError as e:
        return e
    return None


# --------------------------------------------------------------------------
# 1-3: the module
# --------------------------------------------------------------------------
def test_the_fixture_is_not_the_real_store():
    check(os.path.abspath(images.media_dir()) != REAL_MEDIA,
          "media store is a temp dir, not index/media", images.media_dir())
    check(os.path.abspath(images._secret_path()) != REAL_KEY,
          "the URL key is a temp file, not index/media_url.key")


def test_generate_sends_the_documented_request_and_stores_metadata():
    SEEN.clear()
    with use(YAMADORI_IMAGEGEN_URL=IMG_URL):
        recs = images.generate("a red fox, flat vector", size="1024x768",
                               seed=42)
    check(len(recs) == 1, "one image", str(len(recs)))
    req = SEEN[-1] if SEEN else {}
    b = req.get("body") or {}
    check(req.get("path") == "/sdapi/v1/txt2img", "POST /sdapi/v1/txt2img",
          str(req.get("path")))
    check(b.get("model") == "imagegen" and b.get("width") == 1024
          and b.get("height") == 768 and b.get("steps") == images.DEFAULT_STEPS
          and b.get("seed") == 42 and b.get("batch_size") == 1
          and b.get("prompt") == "a red fox, flat vector",
          "model, size, steps, seed, batch_size and prompt are sent",
          json.dumps(b)[:200])
    r = recs[0]
    p = images.media_path(r["id"])
    check(p is not None and open(p, "rb").read() == tiny_png(42),
          "the PNG is stored under its sha", str(p))
    meta = images.metadata(r["id"]) or {}
    for k, want in (("prompt", "a red fox, flat vector"), ("seed", 42),
                    ("size", "1024x768"), ("steps", 20), ("model", "imagegen")):
        check(meta.get(k) == want, f"metadata records {k}", str(meta.get(k)))
    check(isinstance(meta.get("seconds"), (int, float)),
          "metadata records seconds", str(meta.get("seconds")))
    import hashlib
    check(r["id"] == hashlib.sha256(tiny_png(42)).hexdigest(),
          "the id is the sha256 of the bytes")
    with use(YAMADORI_IMAGEGEN_URL=IMG_URL):
        again = images.generate("a red fox, flat vector", size="1024x768",
                                seed=42)
    check(again[0]["id"] == r["id"]
          and len([f for f in os.listdir(images.media_dir())
                   if f.startswith(r["id"]) and f.endswith(".png")]) == 1,
          "the same image is stored once")
    with use(YAMADORI_IMAGEGEN_URL=IMG_URL):
        two = images.generate("two", seed=100, n=2)
    check(len(two) == 2 and two[0]["id"] != two[1]["id"]
          and [x["seed"] for x in two] == [100, 101],
          "n=2 gives two images with consecutive seeds",
          json.dumps([x["seed"] for x in two]))
    with use(YAMADORI_IMAGEGEN_URL=IMG_URL):
        rnd = images.generate("no seed given")
    check(isinstance(rnd[0]["seed"], int) and SEEN[-1]["body"]["seed"]
          == rnd[0]["seed"], "an omitted seed is chosen and recorded")


def test_every_failure_names_itself_and_the_next_step():
    def shape(e, code, name, retryable=None):
        if not check(e is not None and e.code == code, f"{name}: {code}",
                     getattr(e, "code", "no error raised")):
            return
        check(e.reason and "No image" in e.reason or "Nothing" in e.reason,
              f"{name}: says nothing was made", e.reason[:120])
        if retryable is not None:
            check(e.retryable is retryable,
                  f"{name}: retryable={retryable} as a fact", str(e.retryable))
        check(bool(e.remedies) and all(r.get("fixable_by") in
                                       ("agent", "user", "operator")
                                       for r in e.remedies),
              f"{name}: remedies name an owner", json.dumps(e.remedies)[:160])

    with use(YAMADORI_IMAGEGEN_URL=None):
        shape(err_of(images.generate, "a fox"), "IMAGEGEN_NOT_CONFIGURED",
              "not configured", False)
    with use(YAMADORI_IMAGEGEN_URL=DEAD):
        e = err_of(images.generate, "a fox")
        shape(e, "IMAGEGEN_DOWN", "server down", False)
        check(e is not None and any("start" in r.get("action", "")
                                    for r in e.remedies),
              "server down: the remedy says how to start it")
    with use(YAMADORI_IMAGEGEN_URL=IMG_URL):
        MODE["mode"] = "oom"
        e = err_of(images.generate, "a fox", size="1344x1344")
        shape(e, "IMAGEGEN_OUT_OF_MEMORY", "out of memory", True)
        check(e is not None and "out of GPU memory" in e.reason
              and "1024x1024" in json.dumps(e.remedies),
              "OOM says so, and offers a smaller size", e and e.reason[:120])
        MODE["mode"] = "crashed"
        shape(err_of(images.generate, "a fox"), "IMAGEGEN_UNAVAILABLE",
              "process crashed", False)
        MODE["mode"] = "empty"
        shape(err_of(images.generate, "a fox"), "IMAGEGEN_EMPTY",
              "empty answer", False)
        MODE["mode"] = "ok"
    with use(YAMADORI_IMAGEGEN_URL=IMG_URL, YAMADORI_IMAGEGEN_TIMEOUT="0.5"):
        MODE["mode"] = "slow"
        t0 = time.time()
        e = err_of(images.generate, "a fox")
        shape(e, "IMAGEGEN_TIMEOUT", "timeout", False)
        check(time.time() - t0 < 2.5, "a stuck server does not hang the caller",
              f"{time.time() - t0:.1f}s")
    with use(YAMADORI_IMAGEGEN_URL=IMG_URL):
        lane = admission.image_lane(timeout=0)
        held = lane.__enter__()
        try:
            e = err_of(images.generate, "a fox")
        finally:
            lane.__exit__(None, None, None)
        check(held, "the test holds the image lane")
        shape(e, "IMAGEGEN_BUSY", "busy", True)
        check(e is not None and e.status == 429, "busy is a 429, not a failure",
              str(getattr(e, "status", None)))
        for bad, why in ((("",), "empty prompt"), ((5,), "prompt is a number"),
                         (("x", "1000x1000"), "size not a multiple of 32"),
                         (("x", "4096x4096"), "size over the measured ceiling"),
                         (("x", "big"), "size is a word"),
                         (("x", None, "abc"), "seed is a word")):
            e = err_of(images.generate, *bad)
            shape(e, "BAD_ARGUMENTS", why, True)


# --------------------------------------------------------------------------
# 5: signing
# --------------------------------------------------------------------------
def test_signatures():
    sha = "a" * 64
    exp = int(time.time()) + 60
    sig = images.sign(sha, exp)
    check(images.verify(sha, exp, sig) == (True, "ok"), "a fresh sig verifies")
    check(not images.verify(sha, exp, sig[:-1] + ("0" if sig[-1] != "0" else "1"))[0],
          "a tampered sig does not")
    check(not images.verify("b" * 64, exp, sig)[0],
          "a sig for another sha does not")
    check(not images.verify(sha, exp + 1, sig)[0],
          "moving the expiry breaks the sig")
    past = int(time.time()) - 5
    ok, why = images.verify(sha, past, images.sign(sha, past))
    check(not ok and "expired" in why, "an expired link is refused as expired", why)
    check(not images.verify(sha, "soon", sig)[0] and not images.verify(sha, exp, "")[0],
          "malformed exp and missing sig are refused")
    key = open(images._secret_path(), "rb").read()
    check(len(key) >= 32, "the key was created once, on first use")
    check(key.decode() not in images.signed_url(sha, "https://x"),
          "the key never appears in a URL")


def test_public_base():
    with use(YAMADORI_PUBLIC_BASE="https://ai.example.test/"):
        check(images.public_base(request_base="http://127.0.0.1:1234/")
              == "https://ai.example.test", "YAMADORI_PUBLIC_BASE wins")
    with use(YAMADORI_PUBLIC_BASE=None):
        check(images.public_base(request_base="http://host:1234/")
              == "http://host:1234", "otherwise the request's base")
        check(images.public_base(request_base="http://host/",
                                 forwarded_proto="https") == "https://host",
              "X-Forwarded-Proto is honoured")


# --------------------------------------------------------------------------
# 4-5: the HTTP routes
# --------------------------------------------------------------------------
_KEY: dict = {}


def _client():
    from starlette.testclient import TestClient
    import server
    if "key" not in _KEY:
        _KEY["key"] = accounts.create("image-tests")
    return TestClient(server.app), {"Authorization": f"Bearer {_KEY['key']}"}


def test_images_route_auth_and_shapes():
    client, auth = _client()
    with use(YAMADORI_IMAGEGEN_URL=IMG_URL):
        r = client.post("/v1/images/generations", json={"prompt": "a fox"})
        check(r.status_code == 401, "no key: 401", str(r.status_code))
        r = client.post("/v1/images/generations", json={"prompt": "a fox"},
                        headers={"Authorization": "Bearer wrong"})
        check(r.status_code == 401, "wrong key: 401", str(r.status_code))

        r = client.post("/v1/images/generations", headers=auth,
                        json={"prompt": "a fox", "size": "512x512", "seed": 3})
        d = r.json()
        check(r.status_code == 200 and isinstance(d.get("created"), int)
              and len(d.get("data") or []) == 1,
              "OpenAI shape: created + data[1]", r.text[:160])
        u = (d.get("data") or [{}])[0].get("url", "")
        check(bool(re.fullmatch(r"http://testserver/media/[0-9a-f]{64}\.png"
                                r"\?exp=\d+&sig=[0-9a-f]{64}", u)),
              "response_format defaults to url: a signed /media link", u)
        check("b64_json" not in json.dumps(d),
              "no base64 unless asked for")
        path = u.replace("http://testserver", "")
        r2 = client.get(path)
        check(r2.status_code == 200 and r2.headers["content-type"] == "image/png"
              and r2.content == tiny_png(3),
              "the signed URL serves the PNG with NO Authorization header",
              f"{r2.status_code} {r2.headers.get('content-type')}")

        r = client.post("/v1/images/generations", headers=auth,
                        json={"prompt": "a fox", "seed": 4, "n": 2,
                              "response_format": "b64_json"})
        d = r.json()
        got = [base64.b64decode(x.get("b64_json", "")) for x in d.get("data", [])]
        check(r.status_code == 200 and got == [tiny_png(4), tiny_png(5)],
              "b64_json when asked, n=2", r.text[:120])

        r = client.post("/v1/images/generations", headers=auth,
                        json={"prompt": "a fox", "response_format": "gif"})
        check(r.status_code == 400 and r.json()["error"]["code"] == "BAD_ARGUMENTS",
              "an unsupported response_format is a 400 with a remedy", r.text[:160])
        r = client.post("/v1/images/generations", headers=auth, json={})
        check(r.status_code == 400 and r.json()["error"]["remedies"],
              "a missing prompt is a 400 with a remedy", r.text[:160])

        lane = admission.image_lane(timeout=0)
        lane.__enter__()
        try:
            r = client.post("/v1/images/generations", headers=auth,
                            json={"prompt": "a fox"})
        finally:
            lane.__exit__(None, None, None)
        check(r.status_code == 429 and r.headers.get("retry-after")
              and r.json()["error"]["code"] == "IMAGEGEN_BUSY",
              "a busy lane is a 429 with Retry-After", r.text[:160])
    with use(YAMADORI_IMAGEGEN_URL=None):
        r = client.post("/v1/images/generations", headers=auth,
                        json={"prompt": "a fox"})
        check(r.status_code == 503 and r.json()["error"]["code"]
              == "IMAGEGEN_NOT_CONFIGURED",
              "unconfigured: 503 naming the situation", r.text[:160])
    with use(YAMADORI_IMAGEGEN_URL=IMG_URL,
             YAMADORI_PUBLIC_BASE="https://ai.example.test"):
        r = client.post("/v1/images/generations", headers=auth,
                        json={"prompt": "a fox", "seed": 9})
        u = r.json()["data"][0]["url"]
        check(u.startswith("https://ai.example.test/media/"),
              "the URL uses YAMADORI_PUBLIC_BASE when set", u[:60])


def test_media_route_refuses_everything_but_a_valid_signed_id():
    client, auth = _client()
    with use(YAMADORI_IMAGEGEN_URL=IMG_URL):
        sha = images.generate("media test", seed=77)[0]["id"]
    exp = int(time.time()) + 600
    sig = images.sign(sha, exp)
    ok = f"/media/{sha}.png?exp={exp}&sig={sig}"
    check(client.get(ok).status_code == 200, "valid: 200 without a key")
    check(client.head(ok).status_code == 200, "HEAD works too")

    flipped = sig[:-1] + ("0" if sig[-1] != "0" else "1")
    r = client.get(f"/media/{sha}.png?exp={exp}&sig={flipped}")
    check(r.status_code == 403, "tampered sig: 403", str(r.status_code))
    past = int(time.time()) - 10
    r = client.get(f"/media/{sha}.png?exp={past}&sig={images.sign(sha, past)}")
    check(r.status_code == 403 and "expired" in r.json()["error"]["message"],
          "expired exp: 403 saying expired", r.text[:120])
    other = "c" * 64
    r = client.get(f"/media/{other}.png?exp={exp}&sig={sig}")
    check(r.status_code == 403, "a sig for a different sha: 403", str(r.status_code))
    r = client.get(f"/media/{other}.png?exp={exp}&sig={images.sign(other, exp)}")
    check(r.status_code == 404, "a validly signed sha that does not exist: 404",
          str(r.status_code))
    r = client.get(f"/media/{sha}.png")
    check(r.status_code == 403, "no signature at all: 403", str(r.status_code))
    r = client.get(f"/media/{sha}.png", headers=auth)
    check(r.status_code == 403,
          "an API key is not a substitute for the signature", str(r.status_code))

    for bad in ("..%2F..%2Findex%2Fmedia_url.key", "..%5C..%5Csecret.png",
                "../media_url.key", "ABCDEF.png", sha.upper() + ".png",
                sha + ".json", sha, "zz" * 32 + ".png"):
        r = client.get(f"/media/{bad}?exp={exp}&sig={sig}")
        check(r.status_code == 404 and b"PNG" not in r.content[:8],
              f"{bad[:24]!r}: refused as not an id", str(r.status_code))
    for bad in ("../x", "..\\x", "x" * 64, sha + "/../" + sha, ""):
        check(images.media_path(bad) is None,
              f"media_path({bad[:20]!r}) is None")
    key = open(images._secret_path(), "rb").read().decode()
    for path in (ok, "/v1/models", "/"):
        check(key not in client.get(path).text, f"{path[:20]} never shows the key")


def test_chat_route_passes_the_public_base():
    import proxy
    client, auth = _client()
    seen: dict = {}
    real = proxy.complete

    def fake(body):
        seen.update(body)
        return {"choices": [{"message": {"role": "assistant", "content": "x"},
                             "finish_reason": "stop"}]}
    proxy.complete = fake
    try:
        with use(YAMADORI_PUBLIC_BASE="https://ai.example.test"):
            client.post("/v1/chat/completions", headers=auth,
                        json={"messages": [{"role": "assistant", "content": "a"},
                                           {"role": "user", "content": "b"}]})
    finally:
        proxy.complete = real
    check(seen.get("_public_base") == "https://ai.example.test",
          "the chat route hands the tool loop the public base",
          str(seen.get("_public_base")))


# --------------------------------------------------------------------------
# 6-8: the model tool
# --------------------------------------------------------------------------
def test_tool_is_offered_only_when_configured_and_allowed():
    import proxy
    msgs = [{"role": "user", "content": "draw me a lighthouse at dusk"}]
    for url, effort, want, why in (
            (None, "low", False, "unconfigured, tier low"),
            (IMG_URL, "low", True, "configured, tier low, code tools withheld"),
            (IMG_URL, "minimal", False, "configured, tier minimal"),
            (IMG_URL, "high", True, "configured, tier high")):
        with use(YAMADORI_IMAGEGEN_URL=url):
            out = proxy.prepare({"model": "yamadori", "messages": msgs,
                                 "reasoning_effort": effort})
        names = [t.get("function", {}).get("name") for t in out.get("tools") or []]
        check(("generate_image" in names) == want,
              f"{why}: generate_image {'offered' if want else 'not offered'}",
              str(names))
    with use(YAMADORI_IMAGEGEN_URL=IMG_URL):
        client_tool = {"type": "function", "function": {
            "name": "generate_image", "description": "the client's own",
            "parameters": {"type": "object", "properties": {}}}}
        out = proxy.prepare({"model": "yamadori", "messages": msgs,
                             "reasoning_effort": "low", "tools": [client_tool]})
        mine = [t for t in out["tools"]
                if t["function"]["name"] == "generate_image"]
        check(len(mine) == 1 and mine[0]["function"]["description"]
              == "the client's own", "a client's own generate_image wins")
    d = images.TOOL["function"]["description"]
    for phrase in ("draw", "logo", "icon", "illustration", "picture",
                   "markdown image line"):
        check(phrase in d, f"the description triggers on / says {phrase!r}")


def _markdown_img(md: str) -> list[tuple[str, str]]:
    """A markdown image parser of the kind chat UIs use (CommonMark's inline
    image: `![alt](destination)`, destination without spaces). Returns
    (alt, src) pairs -- what a renderer turns into <img alt src>."""
    return re.findall(r"!\[([^\]]*)\]\((\S+?)\)", md)


# HERMES, as its own source reads it -- NousResearch/hermes-agent at 1a90fad
# (the main commit the 2026-09-21 install cloned; docs/IMAGEGEN.md cites each).
#
# gateway/platforms/base.py:2916-2924, extract_images(): what the messaging
# gateway (Telegram, Discord, Slack, Signal...) pulls out of a reply and sends
# as a photo.
_HERMES_GW_MD = r'!\[([^\]]*)\]\((https?://[^\s\)]+)\)'
_HERMES_GW_MARKERS = ('.png', '.jpg', '.jpeg', '.gif', '.webp', 'fal.media',
                      'fal-cdn', 'replicate.delivery')


def _hermes_gateway_images(content: str) -> list[tuple[str, str]]:
    return [(m.group(2), m.group(1)) for m in re.finditer(_HERMES_GW_MD, content)
            if any(m.group(2).lower().endswith(ext) or ext in m.group(2).lower()
                   for ext in _HERMES_GW_MARKERS)]


def _hermes_desktop_inline(src: str) -> bool:
    """apps/desktop/src/lib/media.ts:75-77 isInlineMediaSrc, and :33-37
    mediaInfo: the query is cut before the extension is read."""
    ext = re.split(r"[?#]", src, maxsplit=1)[0].split(".")[-1].lower()
    return bool(re.match(r"^(?:https?|data):", src, re.I)) and ext == "png"


def test_tool_result_is_a_rendered_image():
    rec: list = []
    with use(YAMADORI_IMAGEGEN_URL=IMG_URL):
        out = images.run_tool({"prompt": "a [bracketed] lighthouse at dusk, "
                                         "the sign reads \"OPEN\"", "seed": 11},
                              "https://ai.example.test", rec)
    d = json.loads(out)
    check(d.get("ok") is True, "ok", out[:160])
    u = d.get("url", "")
    check(u.startswith("https://ai.example.test/media/") and "sig=" in u,
          "the URL is signed on the given public base", u[:80])
    imgs = _markdown_img(d.get("markdown", ""))
    check(len(imgs) == 1 and imgs[0][1] == u,
          "the markdown parses as exactly one image whose src is the URL",
          d.get("markdown", "")[:160])
    check(imgs and "[" not in imgs[0][0] and "]" not in imgs[0][0],
          "brackets in the prompt cannot break the alt text", imgs and imgs[0][0])
    path = u.split("?", 1)[0]
    check(path.endswith(".png"),
          "the URL path ends in .png before the query (extension sniffers)")
    answer = ("Here is your lighthouse.\n\n" + d.get("markdown", "")
              + "\n\nSeed 11.")
    gw = _hermes_gateway_images(answer)
    check(len(gw) == 1 and gw[0][0] == u,
          "Hermes gateway extract_images() finds it and would send it as a photo",
          json.dumps(gw)[:160])
    check(imgs and _hermes_desktop_inline(imgs[0][1]),
          "Hermes desktop renders it inline: https src, .png kind", u[:80])
    check("exactly as given" in d.get("instruction", ""),
          "the result tells the model to put the line in its answer verbatim")
    check("base64" not in out.lower() and len(out) < 2000,
          "no base64 in what goes to the model", str(len(out)))
    check(rec and rec[0].get("ok") is True and rec[0].get("seed") == 11
          and len(rec[0].get("id", "")) == 16 and "prompt" not in rec[0],
          "the call is recorded: ok, seed, id prefix, no prompt",
          json.dumps(rec))
    with use(YAMADORI_IMAGEGEN_URL=IMG_URL):
        bad = json.loads(images.run_tool(["not", "a", "dict"], "https://x", rec))
        none = json.loads(images.run_tool({"prompt": "x"}, "", rec))
    check(bad.get("error") == "BAD_ARGUMENTS", "non-object args: BAD_ARGUMENTS")
    check(none.get("error") == "NO_PUBLIC_BASE" and none.get("remedies"),
          "no public base: says so, with the operator remedy")
    check(rec[-1].get("ok") is False and rec[-1].get("error") == "NO_PUBLIC_BASE",
          "a failed call is recorded with its code")


def test_a_chat_turn_that_draws():
    """complete() with a fake LLM that calls generate_image, then answers."""
    import proxy

    replies = [
        {"calls": [{"id": "c1", "type": "function", "function": {
            "name": "generate_image",
            "arguments": json.dumps({"prompt": "a fox", "seed": 21})}}]},
        {"content": "Here it is."}]
    sent: list = []

    class _LLM(BaseHTTPRequestHandler):
        def do_POST(self):                                       # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            sent.append(body)
            r = replies.pop(0) if replies else {"content": "?"}
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

        def log_message(self, *a):                               # noqa: D102
            pass

    llm = ThreadingHTTPServer(("127.0.0.1", 0), _LLM)
    threading.Thread(target=llm.serve_forever, daemon=True).start()
    old = proxy.UPSTREAM
    proxy.UPSTREAM = f"http://127.0.0.1:{llm.server_address[1]}"
    try:
        with use(YAMADORI_IMAGEGEN_URL=IMG_URL):
            d = proxy.complete({"model": "yamadori", "reasoning_effort": "low",
                                "messages": [{"role": "user",
                                              "content": "draw me a fox"}],
                                "_public_base": "https://ai.example.test"})
    finally:
        proxy.UPSTREAM = old
        llm.shutdown()
        llm.server_close()
    first_tools = [t["function"]["name"] for t in (sent[0].get("tools") or [])] if sent else []
    check("generate_image" in first_tools, "the model was offered generate_image",
          str(first_tools))
    tool_msgs = [m for m in (sent[1]["messages"] if len(sent) > 1 else [])
                 if m.get("role") == "tool"]
    res = json.loads(tool_msgs[0]["content"]) if tool_msgs else {}
    check(res.get("ok") is True and res.get("url", "").startswith(
        "https://ai.example.test/media/"),
          "the proxy ran it and the model got the signed URL on the public base",
          json.dumps(res)[:160])
    check(not any(k.startswith("_") for k in (sent[0] if sent else {})),
          "no underscored bookkeeping (e.g. _public_base) went upstream",
          str([k for k in (sent[0] if sent else {}) if k.startswith("_")]))
    x = d.get("x_yamadori") or {}
    im = x.get("images") or []
    check(len(im) == 1 and im[0].get("ok") and im[0].get("seed") == 21,
          "x_yamadori.images records the call", json.dumps(im))
    check(d["choices"][0]["message"]["content"] == "Here it is.",
          "the answer comes back")


def main() -> int:
    tests = (test_the_fixture_is_not_the_real_store,
             test_generate_sends_the_documented_request_and_stores_metadata,
             test_every_failure_names_itself_and_the_next_step,
             test_signatures,
             test_public_base,
             test_images_route_auth_and_shapes,
             test_media_route_refuses_everything_but_a_valid_signed_id,
             test_chat_route_passes_the_public_base,
             test_tool_is_offered_only_when_configured_and_allowed,
             test_tool_result_is_a_rendered_image,
             test_a_chat_turn_that_draws)
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
        _srv.shutdown()
        _srv.server_close()
    passed = sum(1 for ok, _, _ in _results if ok)
    total = len(_results)
    print(f"\n{'=' * 70}\n  {passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
