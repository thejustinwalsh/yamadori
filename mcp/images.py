#!/usr/bin/env python
"""Image generation: the one module between the proxy and the image server.

WHAT IT TALKS TO

stable-diffusion.cpp's `sd-server`, running Qwen-Image-2.1 on CUDA1, reached at
YAMADORI_IMAGEGEN_URL. In production that is llama-swap on loopback :11434,
which starts the `imagegen` model on demand and routes on the `model` field of
the body; the proxy never needs to know whether it was already loaded. The
request is the SD-WebUI shape `/sdapi/v1/txt2img` rather than the OpenAI one,
because it takes `seed` and `steps` as fields -- the OpenAI shape would carry
them smuggled inside the prompt (`<sd_cpp_extra_args>`), and a prompt is text a
model wrote. docs/IMAGEGEN.md records the server's API and the measurements
behind every default below.

This is NOT the language model, so it does not go through mcp/model.py: that
door exists to give every LLM generation one budget rule, and an image has no
tokens. It does take a lane (`admission.image_lane`), because it is a GPU
consumer, and that lane is its own so an image never blocks a chat.

WHERE RESULTS GO

index/media/<sha256>.png, with index/media/<sha256>.json beside it holding the
prompt, seed, size, steps, model and seconds. The id is the sha of the PNG
bytes, so the same image is stored once and an id can only ever name bytes
this server wrote. index/ is gitignored.

HOW A CLIENT SEES ONE: CAPABILITY URLS

A chat client renders `![...](URL)` by having the browser fetch URL, and that
fetch carries no Authorization header. A Bearer-gated /media route would
therefore show as a broken image in every client that matters. So /media is a
capability URL instead:

    /media/<sha>.png?exp=<unix>&sig=<hex HMAC-SHA256(secret, "<sha>|<exp>")>

The secret is generated once and kept at index/media_url.key, never logged or
printed. A URL is valid until `exp` (default 30 days, YAMADORI_MEDIA_URL_TTL)
and names exactly one image; tampering with either field, or moving the sig to
another sha, fails the constant-time compare.

Base64 is never put in a chat answer: the answer is sent back to the model on
every later turn, and a 1 MB PNG as text is ~350,000 tokens of nothing.
`b64_json` exists only on /v1/images/generations, when a client asks for it.

FAILURES CARRY THE NEXT STEP

Every failure is an `ImageError` with a code, whether retrying can help (as a
fact), and remedies with an owner -- the tool envelope code_search uses, so the
model reads image failures the way it reads search failures. The four that
matter: not configured, server down, out of memory, and busy.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import random
import re
import secrets
import socket
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import admission  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX = os.path.abspath(os.path.join(HERE, "..", "index"))

# Measured defaults (docs/IMAGEGEN.md, "Measurements"). 20 steps is the model
# card's setting and the one every sample in bench/imagegen/samples used.
DEFAULT_SIZE = (1024, 1024)
DEFAULT_STEPS = 20
# sd.cpp requires both edges divisible by 32 for Qwen-Image-2.1
# (docs/qwen_image_2.1.md in the sd.cpp tree). The ceiling is what was
# measured to fit on CUDA1 beside the resident rootstock and Laya; see
# docs/IMAGEGEN.md before raising it.
SIZE_MULTIPLE = 32
MIN_EDGE = 256
MAX_EDGE = int(os.environ.get("YAMADORI_IMAGEGEN_MAX_EDGE", "1344"))
MAX_PIXELS = int(os.environ.get("YAMADORI_IMAGEGEN_MAX_PIXELS",
                                str(1344 * 1344)))
MAX_N = 4
MAX_PROMPT_CHARS = 4000
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_SHA = re.compile(r"[0-9a-f]{64}")


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def url() -> str:
    """The image server's base URL, or "" when none is configured.

    Read at call time, not import time, so a test or an operator can switch it
    without restarting anything that imported this module.
    """
    return _env("YAMADORI_IMAGEGEN_URL").rstrip("/")


def configured() -> bool:
    return bool(url())


def model_name() -> str:
    """The llama-swap model name that routes to sd-server."""
    return _env("YAMADORI_IMAGEGEN_MODEL", "imagegen")


def timeout() -> float:
    # A cold start through llama-swap loads ~10 GB of weights before the first
    # step, then 20 steps. Generous, and finite: an image server that stops
    # answering must come back as IMAGEGEN_TIMEOUT, never hang the proxy.
    return float(_env("YAMADORI_IMAGEGEN_TIMEOUT", "900"))


def media_dir() -> str:
    return os.path.abspath(_env("YAMADORI_MEDIA_DIR",
                                os.path.join(INDEX, "media")))


def _secret_path() -> str:
    return os.path.abspath(_env("YAMADORI_MEDIA_SECRET_FILE",
                                os.path.join(INDEX, "media_url.key")))


def url_ttl() -> int:
    return int(_env("YAMADORI_MEDIA_URL_TTL", str(30 * 24 * 3600)))


# --------------------------------------------------------------------- errors
class ImageError(Exception):
    """A failure that says what happened, whether retrying helps, and who can
    fix it. `status` is the HTTP code the images route answers with."""

    def __init__(self, code: str, reason: str, retryable: bool,
                 remedies: list[dict], status: int = 502, **facts):
        super().__init__(reason)
        self.code = code
        self.reason = reason
        self.retryable = retryable
        self.remedies = remedies
        self.status = status
        self.facts = facts

    def envelope(self, tool: str = "generate_image") -> dict:
        return {"tool": tool, "ok": False, "error": self.code,
                "reason": self.reason, "retryable": self.retryable,
                "remedies": self.remedies, **self.facts}


def not_configured() -> ImageError:
    return ImageError(
        "IMAGEGEN_NOT_CONFIGURED",
        "No image server is configured on this deployment, so no image was "
        "made.",
        retryable=False, status=503,
        remedies=[{"fixable_by": "operator",
                   "action": ("set YAMADORI_IMAGEGEN_URL (llama-swap, "
                              "http://127.0.0.1:11434) with the `imagegen` "
                              "model in config.yaml, and restart the proxy -- "
                              "docs/IMAGEGEN.md 'Going live'"),
                   "why_not_the_agent": "server configuration"},
                  {"fixable_by": "agent",
                   "action": "tell the user image generation is not available here",
                   "effect": "the user is not left waiting for an image"}])


def _down(detail: str) -> ImageError:
    return ImageError(
        "IMAGEGEN_DOWN",
        f"The image server at {url()} did not answer ({detail}). No image was "
        f"made.",
        retryable=False, status=503,
        remedies=[{"fixable_by": "operator",
                   "action": ("start it: llama-swap serves it as model "
                              f"`{model_name()}` (scripts/start-stack.bat), or "
                              "run sd-server by hand with the command in "
                              "docs/IMAGEGEN.md 'Running sd-server by hand'"),
                   "why_not_the_agent": "the agent cannot start processes on the server"},
                  {"fixable_by": "agent",
                   "action": "tell the user the image server is down",
                   "effect": "retrying in this turn cannot succeed"}])


def _oom(detail: str, w: int, h: int) -> ImageError:
    smaller = "1024x1024" if w * h > 1024 * 1024 else "768x768"
    return ImageError(
        "IMAGEGEN_OUT_OF_MEMORY",
        f"The image server ran out of GPU memory at {w}x{h} ({detail}). No "
        f"image was made.",
        retryable=True, status=507,
        remedies=[{"fixable_by": "agent",
                   "action": f"call generate_image again with size {smaller}",
                   "effect": "a smaller image needs less memory"},
                  {"fixable_by": "operator",
                   "action": ("check CUDA1 with nvidia-smi: something else may "
                              "be holding memory the image server needs"),
                   "why_not_the_agent": "the agent cannot see the GPU"}],
        size=f"{w}x{h}")


def busy() -> ImageError:
    return ImageError(
        "IMAGEGEN_BUSY",
        "Another image is being generated and only one runs at a time. "
        "Nothing was made for this request.",
        retryable=True, status=429,
        remedies=[{"fixable_by": "agent",
                   "action": "try again in a minute, or tell the user it is queued behind another image",
                   "effect": "the lane frees when the other image finishes"}])


def bad_args(reason: str, action: str) -> ImageError:
    return ImageError("BAD_ARGUMENTS", reason + " Nothing was generated.",
                      retryable=True, status=400,
                      remedies=[{"fixable_by": "agent", "action": action,
                                 "effect": "the image is generated"}])


# ------------------------------------------------------------------- validate
def parse_size(size) -> tuple[int, int]:
    """'WIDTHxHEIGHT' -> (w, h), both multiples of 32 inside the measured
    ceiling. None means the default."""
    if size in (None, "", "auto"):
        return DEFAULT_SIZE
    if not isinstance(size, str):
        raise bad_args(f"size must be a string like '1024x1024', got "
                       f"{type(size).__name__}.",
                       "call again with size '1024x1024' or omit it")
    m = re.fullmatch(r"\s*(\d{2,5})\s*[xX*]\s*(\d{2,5})\s*", size)
    if not m:
        raise bad_args(f"size {size!r} is not WIDTHxHEIGHT.",
                       "call again with size '1024x1024' or omit it")
    w, h = int(m[1]), int(m[2])
    if (w % SIZE_MULTIPLE or h % SIZE_MULTIPLE or min(w, h) < MIN_EDGE
            or max(w, h) > MAX_EDGE or w * h > MAX_PIXELS):
        raise bad_args(
            f"size {w}x{h} is not supported: both edges must be multiples of "
            f"{SIZE_MULTIPLE} between {MIN_EDGE} and {MAX_EDGE}, at most "
            f"{MAX_PIXELS:,} pixels.",
            "call again with one of 1024x1024, 1344x768, 768x1344, 768x768")
    return w, h


def _prompt(prompt) -> str:
    if not isinstance(prompt, str) or not prompt.strip():
        raise bad_args("prompt must be a non-empty string describing the image.",
                       "call again with a prompt describing what to draw")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise bad_args(f"prompt is {len(prompt)} characters; the limit is "
                       f"{MAX_PROMPT_CHARS}.",
                       "call again with a shorter description")
    return prompt.strip()


def _seed(seed) -> int:
    if seed is None or seed == "":
        return random.randrange(0, 2**31 - 1)
    if isinstance(seed, bool) or not isinstance(seed, (int, str)):
        raise bad_args("seed must be an integer.", "call again without seed")
    try:
        s = int(seed)
    except ValueError:
        raise bad_args(f"seed {seed!r} is not an integer.",
                       "call again without seed") from None
    if not 0 <= s < 2**31:
        raise bad_args("seed must be between 0 and 2147483647.",
                       "call again without seed")
    return s


# ---------------------------------------------------------------------- store
def media_path(sha: str) -> str | None:
    """The stored PNG for `sha`, or None. `sha` must be exactly 64 lowercase
    hex characters: it is an id, never a path, so `..`, separators and every
    other shape are refused before anything touches the filesystem."""
    if not isinstance(sha, str) or not _SHA.fullmatch(sha):
        return None
    root = media_dir()
    p = os.path.join(root, sha + ".png")
    # Containment, the same test code_search._inside makes: belt and braces,
    # since the regex already admits no separator.
    try:
        r = os.path.normcase(os.path.realpath(root))
        q = os.path.normcase(os.path.realpath(p))
        if os.path.commonpath([r, q]) != r:
            return None
    except ValueError:
        return None
    return p if os.path.isfile(p) else None


def metadata(sha: str) -> dict | None:
    p = media_path(sha)
    if not p:
        return None
    try:
        with open(p[:-4] + ".json", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def store(png: bytes, meta: dict) -> str:
    """Write the PNG and its metadata; return the sha. Idempotent."""
    if not png.startswith(PNG_MAGIC):
        raise ValueError("not a PNG")
    sha = hashlib.sha256(png).hexdigest()
    root = media_dir()
    os.makedirs(root, exist_ok=True)
    p = os.path.join(root, sha + ".png")
    if not os.path.exists(p):
        tmp = p + f".{os.getpid()}.tmp"
        with open(tmp, "wb") as f:
            f.write(png)
        os.replace(tmp, p)
    with open(os.path.join(root, sha + ".json"), "w", encoding="utf-8") as f:
        json.dump(dict(meta, id=sha, bytes=len(png)), f, indent=2)
    return sha


# -------------------------------------------------------------------- signing
def _secret() -> bytes:
    """The URL-signing key, created on first use. Never logged or printed."""
    p = _secret_path()
    try:
        with open(p, "rb") as f:
            key = f.read().strip()
        if len(key) >= 32:
            return key
    except FileNotFoundError:
        pass
    os.makedirs(os.path.dirname(p), exist_ok=True)
    key = secrets.token_hex(32).encode()
    try:
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(key)
        return key
    except FileExistsError:
        # Another process won the race; use its key.
        with open(p, "rb") as f:
            return f.read().strip()


def sign(sha: str, exp: int) -> str:
    return hmac.new(_secret(), f"{sha}|{int(exp)}".encode(),
                    hashlib.sha256).hexdigest()


def verify(sha: str, exp, sig) -> tuple[bool, str]:
    """(ok, reason). Constant-time on the signature."""
    if not isinstance(sig, str) or not re.fullmatch(r"[0-9a-f]{64}", sig or ""):
        return False, "missing or malformed signature"
    try:
        e = int(exp)
    except (TypeError, ValueError):
        return False, "missing or malformed expiry"
    if not hmac.compare_digest(sign(sha, e), sig):
        return False, "signature does not match"
    if e < time.time():
        return False, "link expired"
    return True, "ok"


def public_base(configured_base: str = "", request_base: str = "",
                forwarded_proto: str = "") -> str:
    """Where a client reaches this proxy. YAMADORI_PUBLIC_BASE when set --
    behind Caddy the request says http://127.0.0.1:1234, which no browser can
    open -- else the request's own base, with X-Forwarded-Proto honoured."""
    base = (configured_base or _env("YAMADORI_PUBLIC_BASE")).rstrip("/")
    if base:
        return base
    base = (request_base or "").rstrip("/")
    if forwarded_proto in ("http", "https") and "://" in base:
        base = forwarded_proto + base[base.index("://"):]
    return base


def signed_url(sha: str, base: str, ttl: int | None = None) -> str:
    exp = int(time.time()) + (url_ttl() if ttl is None else int(ttl))
    return f"{base.rstrip('/')}/media/{sha}.png?exp={exp}&sig={sign(sha, exp)}"


# ------------------------------------------------------------------- generate
def _classify_http(code: int, body: str, w: int, h: int) -> ImageError:
    low = body.lower()
    if any(s in low for s in ("out of memory", "cudamalloc", "cuda_error_out_of_memory",
                              "failed to allocate", "oom")):
        return _oom(body[:200].strip() or f"HTTP {code}", w, h)
    if code in (502, 503, 504):
        # llama-swap answers 502/503 when the process it starts exits or never
        # comes up -- which is what an OOM at load time looks like from here.
        return ImageError(
            "IMAGEGEN_UNAVAILABLE",
            f"The image server returned HTTP {code}: {body[:200].strip()}. It "
            f"may have failed to load or crashed; an out-of-memory at load "
            f"time looks like this. No image was made.",
            retryable=False, status=503,
            remedies=[{"fixable_by": "operator",
                       "action": ("read llama-swap's log for the imagegen "
                                  "process and nvidia-smi for CUDA1 free memory"),
                       "why_not_the_agent": "the agent cannot see the server's logs"},
                      {"fixable_by": "agent",
                       "action": "tell the user image generation failed on the server",
                       "effect": "retrying in this turn is unlikely to succeed"}])
    return ImageError(
        "IMAGEGEN_FAILED",
        f"The image server returned HTTP {code}: {body[:200].strip()}. No "
        f"image was made.",
        retryable=code >= 500, status=502,
        remedies=[{"fixable_by": "operator",
                   "action": "read the image server's log for this request",
                   "why_not_the_agent": "the agent cannot see the server's logs"}])


def generate(prompt, size=None, seed=None, steps=None, n: int = 1) -> list[dict]:
    """Generate `n` images. Returns one record per image:
    {id, prompt, seed, size, steps, model, seconds}. Raises ImageError."""
    if not configured():
        raise not_configured()
    text = _prompt(prompt)
    w, h = parse_size(size)
    s = _seed(seed)
    st = DEFAULT_STEPS if steps in (None, "") else steps
    if isinstance(st, bool) or not isinstance(st, int) or not 1 <= st <= 50:
        raise bad_args("steps must be an integer from 1 to 50.",
                       "call again without steps")
    if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= MAX_N:
        raise bad_args(f"n must be an integer from 1 to {MAX_N}.",
                       "call again with n=1")

    body = {"model": model_name(), "prompt": text, "width": w, "height": h,
            "steps": st, "seed": s, "batch_size": n}
    req = urllib.request.Request(
        url() + "/sdapi/v1/txt2img", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with admission.image_lane() as got:
        if not got:
            raise busy()
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=timeout()) as r:
                raw = r.read()
        except urllib.error.HTTPError as e:
            try:
                detail = e.read().decode("utf-8", "replace")
            except Exception:                                    # noqa: BLE001
                detail = ""
            raise _classify_http(e.code, detail, w, h) from None
        except (socket.timeout, TimeoutError):
            raise ImageError(
                "IMAGEGEN_TIMEOUT",
                f"The image server did not finish within {timeout():.0f}s. No "
                f"image was returned.",
                retryable=False, status=504,
                remedies=[{"fixable_by": "operator",
                           "action": ("check the imagegen process: a stuck "
                                      "generation holds the card"),
                           "why_not_the_agent": "the agent cannot see the server"}]) from None
        except urllib.error.URLError as e:
            reason = e.reason
            if isinstance(reason, (socket.timeout, TimeoutError)):
                raise ImageError(
                    "IMAGEGEN_TIMEOUT",
                    f"The image server did not finish within {timeout():.0f}s.",
                    retryable=False, status=504,
                    remedies=[{"fixable_by": "operator",
                               "action": "check the imagegen process",
                               "why_not_the_agent": "the agent cannot see the server"}]) from None
            raise _down(str(reason)) from None
        except (ConnectionError, OSError) as e:
            raise _down(f"{type(e).__name__}: {e}") from None
        seconds = round(time.time() - t0, 2)

    try:
        d = json.loads(raw)
        images = d.get("images") or []
    except (ValueError, AttributeError):
        images = []
    out = []
    for i, b64 in enumerate(images):
        try:
            png = base64.b64decode(b64, validate=True)
        except (ValueError, TypeError):
            continue
        if not png.startswith(PNG_MAGIC):
            continue
        meta = {"prompt": text, "seed": s + i, "size": f"{w}x{h}", "steps": st,
                "model": model_name(), "seconds": seconds,
                "created": int(time.time())}
        sha = store(png, meta)
        out.append(dict(meta, id=sha))
    if not out:
        raise ImageError(
            "IMAGEGEN_EMPTY",
            "The image server answered but returned no PNG. No image was made.",
            retryable=False, status=502,
            remedies=[{"fixable_by": "operator",
                       "action": "read the image server's log for this request",
                       "why_not_the_agent": "no arguments change a server that returns nothing"}])
    return out


def png_bytes(sha: str) -> bytes | None:
    p = media_path(sha)
    if not p:
        return None
    with open(p, "rb") as f:
        return f.read()


# ----------------------------------------------------------------- model tool
TOOL_NAME = "generate_image"

# A description is a trigger condition (AGENTS.md "Tool descriptions are
# prompts"): the question it answers first, then the phrasings that should
# fire it, then what it is NOT -- it does not describe or edit an image the
# user attached.
TOOL = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": (
            "Makes a new picture from a text description. Use it when the user "
            "asks you to draw, render, paint, sketch, generate, create or make "
            "an image, picture, photo, illustration, drawing, icon, logo, "
            "poster, diagram-style illustration, wallpaper or artwork -- "
            "'draw me a fox', 'make a logo for my app', 'generate an image of "
            "a lighthouse at dusk', 'can you show me what that would look "
            "like'. It can render short legible text inside the image, so put "
            "any words that must appear in quotes in the prompt. It does not "
            "look at or edit an image the user sent. The result gives a "
            "markdown image line; put that line in your answer exactly as "
            "given, because it is the only way the user sees the picture."),
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": (
                        "A full visual description in English: subject, "
                        "style, composition, lighting, colours, and any text "
                        "to appear in the image in quotes."),
                },
                "size": {
                    "type": "string",
                    "description": (
                        "WIDTHxHEIGHT, multiples of 32. 1024x1024 (default), "
                        "1344x768 landscape, 768x1344 portrait."),
                },
                "seed": {
                    "type": "integer",
                    "description": "Reuse a previous seed to reproduce an image.",
                },
            },
            "required": ["prompt"],
        },
    },
}


def _alt(prompt: str) -> str:
    """Markdown alt text: the prompt, flattened and cut, with the characters
    that would end the alt early removed."""
    t = re.sub(r"\s+", " ", prompt).replace("[", "(").replace("]", ")")
    return (t[:80] + "...") if len(t) > 80 else t


def run_tool(args, base: str, record: list | None = None) -> str:
    """Execute generate_image for the model. Always returns a JSON envelope;
    never raises. `record` collects one entry per call for x_yamadori."""
    t0 = time.time()
    rec: dict = {"ok": False}
    try:
        if not isinstance(args, dict):
            raise bad_args("arguments must be an object with a prompt.",
                           "call again with {\"prompt\": \"...\"}")
        if not configured():
            raise not_configured()
        if not base:
            raise ImageError(
                "NO_PUBLIC_BASE",
                "The image could not be linked: this server does not know the "
                "address clients reach it on. Nothing was generated.",
                retryable=False, status=503,
                remedies=[{"fixable_by": "operator",
                           "action": "set YAMADORI_PUBLIC_BASE, e.g. https://ai.thejustinwalsh.me",
                           "why_not_the_agent": "server configuration"}])
        recs = generate(args.get("prompt"), size=args.get("size"),
                        seed=args.get("seed"))
        r = recs[0]
        link = signed_url(r["id"], base)
        md = f"![{_alt(r['prompt'])}]({link})"
        rec = {"ok": True, "id": r["id"][:16], "size": r["size"],
               "seed": r["seed"], "steps": r["steps"], "seconds": r["seconds"]}
        return json.dumps({
            "tool": TOOL_NAME, "ok": True,
            "markdown": md, "url": link,
            "seed": r["seed"], "size": r["size"], "steps": r["steps"],
            "seconds": r["seconds"],
            "instruction": ("Put the markdown line above in your answer "
                            "exactly as given, on its own line. The user sees "
                            "the image only through that line; do not "
                            "describe it as attached, and do not alter the "
                            "URL."),
        })
    except ImageError as e:
        rec = {"ok": False, "error": e.code}
        return json.dumps(e.envelope(), default=str)
    except Exception as e:                                       # noqa: BLE001
        rec = {"ok": False, "error": "TOOL_RAISED"}
        return json.dumps({
            "tool": TOOL_NAME, "ok": False, "error": "TOOL_RAISED",
            "reason": f"{type(e).__name__}: {e}", "retryable": False,
            "remedies": [{"fixable_by": "operator",
                          "action": "check the proxy log for the traceback",
                          "why_not_the_agent": "the tool threw; no arguments change that"}]})
    finally:
        rec["ms"] = round((time.time() - t0) * 1000)
        if record is not None:
            record.append(rec)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="generate one image through the "
                                             "configured image server")
    ap.add_argument("prompt")
    ap.add_argument("--size", default=None)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    try:
        for r in generate(a.prompt, size=a.size, seed=a.seed):
            print(json.dumps(r, indent=2))
            print("  ", media_path(r["id"]))
    except ImageError as e:
        print(json.dumps(e.envelope(), indent=2))
        sys.exit(1)
