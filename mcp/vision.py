#!/usr/bin/env python
"""Vision: the one module between the proxy and the model that can see.

WHY IT EXISTS

The served model (Bonsai 2 27B on the RTX 5060 Ti) is text-only. It can draw
with generate_image, and it cannot look at what it drew. A user's attached
image was worse than invisible: the image part went upstream to a server
started without a projector, and llama-server refuses that request outright
("image input is not supported - hint: ... provide the mmproj",
tools/server/server-common.cpp:1212 in the prism build), so the turn failed.

A copy that can see exists as a llama-swap model, `bonsai-vision`: the same
27B plus Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf on the A4000, -c 16384, group
`ondemand` (exclusive), loaded on demand. This module is the only caller.

WHAT IT DOES

1. `extract()` runs on every chat request, in proxy.prepare. It takes each
   image part out of the messages, keeps it for this request under an id the
   model can pass (`image-<10 hex of its sha256>`), and puts a short text
   placeholder naming that id where the image was. The text model sees the
   placeholder; nothing is dropped silently and nothing crashes. Other media
   parts (audio, video, files) get a placeholder too, for the same crash.
2. `describe_image`, a model tool, sends one image and a question to
   `bonsai-vision` and returns its answer as text.

WHERE AN IMAGE MAY COME FROM, AND NOWHERE ELSE

This is the security contract, and it is why no argument is ever a path:

  - an image ATTACHED in this request's own messages, as base64 data, by id;
  - an image in OUR media store (index/media, by sha) that this conversation
    holds a capability for: a signed /media link that verifies (images.verify),
    a link that appeared in the conversation with a valid signature, or an
    image generate_image made during this request.

Nothing is fetched from a URL -- ours or anyone's; a /media link is resolved
to its sha and read from the store through images.media_path, which admits
only 64 hex characters. A model-written string is never opened as a file.
An http(s) image part from a client is NOT forwarded either: llama-server
would download it itself (server-common.cpp:1065, handle_media), which is a
fetch of an arbitrary URL by the server. The model is told to ask the user
for the file. The bytes sent to the vision model are always a data: URI
built here, with the MIME type sniffed from the bytes, not the one claimed.

ONE DOOR

The generation goes through mcp/model.py -- model.chat, shaped by
tiers.apply, with the vendor sampling every generation gets. The only thing
this module adds is `cap`, tiers.budget's existing one-request thinking
override, because the vision server has its own -c 16384 and a thinking room
derived from the main model's pool (~100k tokens) would not fit it. See
`thinking_cap()` for the arithmetic.

It takes the IMAGE lane (admission.image_lane): vision and image generation
are both GPU consumers on the A4000, and one at a time is the rule
(AGENTS.md, "one GPU consumer at a time").

UNTESTED LIVE. Every test is against a fake server (mcp/test_vision.py). The
first real call to bonsai-vision is the live check in docs/IMAGEGEN.md,
"Seeing: describe_image".
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
import socket
import sys
import time
import urllib.error
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import admission  # noqa: E402
import gpu_room  # noqa: E402
import images  # noqa: E402
import model  # noqa: E402
import tiers  # noqa: E402

TOOL_NAME = "describe_image"

# The effort the vision call is shaped at: model.chat's default, the same one
# summarize_text uses. Not measured for vision; the live check records
# completion tokens so it can be.
EFFORT = "low"

# THE VISION SERVER'S CONTEXT. config.yaml, `bonsai-vision`: `-c 16384`.
# Read at call time so an operator who changes -c can say so without a
# restart of anything that imported this module.
DEFAULT_CONTEXT = 16384

# THE LARGEST IMAGE THE PROJECTOR EMITS. llama.cpp clip.cpp:1632 (prism
# build) caps the Qwen-VL projector family at set_limit_image_tokens(8, 4096).
# ASSUMED, NOT MEASURED: that this mmproj is of that family (the base model
# is a Qwen3.8 derivative). The live check reads usage.prompt_tokens, which
# replaces this allowance with a number.
IMAGE_TOKENS_MAX = 4096

# llama-server's own ceiling on an image it downloads (server-common.cpp:1068,
# params.max_size = 10 MB). The same number, so an attachment is refused here
# at the size the server itself would refuse from a URL.
DEFAULT_MAX_BYTES = 10 * 1024 * 1024
MAX_QUESTION_CHARS = 4000

# Formats by magic bytes. stb_image reads PNG/JPEG/GIF/BMP
# (tools/mtmd/mtmd-helper.cpp:404); WebP goes to ffmpeg when the build has
# it (mtmd-helper.cpp:415), so it is passed on and a decode failure comes
# back as VISION_REJECTED_IMAGE rather than being guessed at here.
MIME = {"png": "image/png", "jpeg": "image/jpeg", "gif": "image/gif",
        "bmp": "image/bmp", "webp": "image/webp"}

ID_PREFIX = "image-"
_ATT_ID = re.compile(r"image-[0-9a-f]{10}")
_SHA = re.compile(r"[0-9a-f]{64}")
# A /media capability link, anywhere in a string: the sha, and the query.
_MEDIA = re.compile(r"/media/([0-9a-f]{64})\.png(?:\?([^\s)\"'<>\]]*))?")

IMAGE_PARTS = {"image_url", "input_image", "image"}
OTHER_MEDIA = {"input_audio", "input_video", "audio", "video", "file",
               "input_file", "document"}

SYSTEM = (
    "You are looking at one image for another model that cannot see it. "
    "Answer the question about the image concretely. Name the objects, their "
    "positions, colours and sizes relative to each other, and the layout. "
    "Copy any text in the image exactly as written. When something the "
    "question asks about is not visible or not legible, say so plainly.")


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def enabled() -> bool:
    """Vision is on unless YAMADORI_VISION=0. Read per call."""
    return _env("YAMADORI_VISION", "1") != "0"


def context() -> int:
    return int(_env("YAMADORI_VISION_CTX", str(DEFAULT_CONTEXT)))


def max_bytes() -> int:
    return int(_env("YAMADORI_VISION_MAX_BYTES", str(DEFAULT_MAX_BYTES)))


def timeout() -> float:
    # A cold start loads the 27B and its projector on the A4000 before the
    # first token; then up to thinking_cap() tokens. Generous and finite, as
    # for images: a vision server that stops answering must come back as
    # VISION_TIMEOUT, never hang the proxy. Not measured.
    return float(_env("YAMADORI_VISION_TIMEOUT", "900"))


def sniff(data: bytes) -> str | None:
    """The image format from its first bytes, or None if it is not one."""
    if data.startswith(images.PNG_MAGIC):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if data[:2] == b"BM" and len(data) > 26:
        return "bmp"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


def thinking_cap(question: str) -> int:
    """How much the vision model may think, so the whole request fits -c.

        context 16,384 - image 4,096 (the projector's cap) - answer 2,048
        (tiers.A_MIN) - the text prompt (tiers' deliberately high estimate)

    about 10,000 tokens for a short question. Arithmetic from the numbers
    above, not a measurement; never below tiers.MIN_THINKING."""
    text = tiers.estimate_prompt_tokens({"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question}]})
    return max(context() - IMAGE_TOKENS_MAX - tiers.A_MIN - text,
               tiers.MIN_THINKING)


# --------------------------------------------------------------------- errors
# The envelope is images.ImageError's: code, reason, retryable as a fact,
# remedies with an owner -- so the model reads a vision failure the way it
# reads an image or a search failure.
Err = images.ImageError


def _agent(action: str, effect: str) -> dict:
    return {"fixable_by": "agent", "action": action, "effect": effect}


def bad_args(reason: str, action: str) -> Err:
    return Err("BAD_ARGUMENTS", reason + " Nothing was looked at.",
               retryable=True, status=400,
               remedies=[_agent(action, "the image is looked at")])


def vision_off() -> Err:
    return Err(
        "VISION_OFF",
        "Image viewing is switched off on this server (YAMADORI_VISION=0). "
        "Nothing was looked at.",
        retryable=False, status=503,
        remedies=[{"fixable_by": "operator",
                   "action": "unset YAMADORI_VISION and restart the proxy",
                   "why_not_the_agent": "server configuration"},
                  _agent("tell the user you cannot see images on this server",
                         "the user is not left waiting for a description")])


def _available(att: dict | None) -> dict:
    """What IS available to look at, for a remedy: attached ids that are
    readable, and the images this server made that the conversation holds."""
    att = att or {}
    ok = [i for i, e in (att.get("images") or {}).items() if not e.get("error")]
    return {"attached_ids": ok[:8], "generated_ids": sorted(att.get("media") or [])[:5]}


def unknown(ref, att: dict | None, why: str) -> Err:
    have = _available(att)
    if have["attached_ids"] or have["generated_ids"]:
        action = ("call describe_image again with one of the ids listed in "
                  "`available`: an attached image's id, or the url (or sha) "
                  "generate_image returned")
    else:
        action = ("there is no image in this conversation to look at: ask the "
                  "user to attach one, or make one with generate_image first")
    shown = ref if isinstance(ref, str) else type(ref).__name__
    return Err("UNKNOWN_IMAGE",
               f"{str(shown)[:80]!r} is not an image this conversation can "
               f"show: {why} Nothing was looked at.",
               retryable=True, status=404,
               remedies=[_agent(action, "the image is looked at")],
               available=have)


def url_not_allowed() -> Err:
    return Err(
        "IMAGE_URL_NOT_ALLOWED",
        "That is a link to an image on another site. This server looks only "
        "at images attached to the conversation or made here, and it does "
        "not download from links. Nothing was looked at.",
        retryable=False, status=400,
        remedies=[_agent("ask the user to attach the image file itself in the "
                         "chat", "an attached image gets an id you can pass"),
                  {"fixable_by": "user",
                   "action": "attach the image file instead of a link",
                   "effect": "the model can look at it"}])


def link_invalid(why: str) -> Err:
    return Err(
        "IMAGE_LINK_INVALID",
        f"That /media link does not verify ({why}). Only a valid signed link "
        f"names an image here. Nothing was looked at.",
        retryable=False, status=403,
        remedies=[_agent("use the url exactly as generate_image returned it, or "
                         "draw the image again with generate_image",
                         "a fresh link verifies")])


def not_found(sha: str) -> Err:
    return Err(
        "IMAGE_NOT_FOUND",
        f"The image {sha[:16]}... was made here but is no longer in the media "
        f"store. Nothing was looked at.",
        retryable=False, status=404,
        remedies=[{"fixable_by": "operator",
                   "action": "check index/media for the file; it may have been removed",
                   "why_not_the_agent": "the agent cannot restore files"},
                  _agent("draw it again with generate_image, then look at the new one",
                         "the new image is in the store")])


def not_an_image(detail: str) -> Err:
    return Err(
        "NOT_AN_IMAGE",
        f"The attachment is not an image this server can read ({detail}). "
        f"Nothing was looked at.",
        retryable=False, status=415,
        remedies=[_agent("tell the user the attachment could not be read as an "
                         "image", "the user can send it again"),
                  {"fixable_by": "user",
                   "action": "attach the picture as a PNG or JPEG file",
                   "effect": "the model can look at it"}])


def too_large(n: int) -> Err:
    return Err(
        "IMAGE_TOO_LARGE",
        f"The image is {n:,} bytes; the limit is {max_bytes():,} "
        f"(llama-server's own limit for an image). Nothing was looked at.",
        retryable=False, status=413,
        remedies=[_agent("tell the user the image is too large to look at",
                         "the user can send a smaller one"),
                  {"fixable_by": "user",
                   "action": "attach a smaller or downscaled copy (a PNG or JPEG under 10 MB)",
                   "effect": "the model can look at it"}])


def busy() -> Err:
    return Err(
        "VISION_BUSY",
        "The A4000 is busy with another image job (drawing or looking), and "
        "only one runs at a time. Nothing was looked at.",
        retryable=True, status=429,
        remedies=[_agent("try again in a minute", "the lane frees when the other job finishes")])


def _operator(action: str, why: str = "the agent cannot see the server") -> dict:
    return {"fixable_by": "operator", "action": action, "why_not_the_agent": why}


def _down(detail: str) -> Err:
    return Err(
        "VISION_DOWN",
        f"The model server at {model.UPSTREAM} did not answer ({detail}). "
        f"Nothing was looked at.",
        retryable=False, status=503,
        remedies=[_operator("start llama-swap (scripts/start-stack.bat); "
                            f"it serves `{model.VISION_MODEL}` on demand",
                            "the agent cannot start processes on the server"),
                  _agent("tell the user the image could not be looked at",
                         "retrying in this turn cannot succeed")])


def _timeout_err() -> Err:
    return Err(
        "VISION_TIMEOUT",
        f"The vision model did not answer within {timeout():.0f}s. A cold "
        f"start loads the 27B and its projector first. Nothing came back.",
        retryable=False, status=504,
        remedies=[_operator(f"check the `{model.VISION_MODEL}` process in "
                            "llama-swap's log and the A4000 in nvidia-smi"),
                  _agent("tell the user looking at the image timed out",
                         "the user is not left waiting")])


def _classify_http(code: int, body: str) -> Err:
    low = body.lower()
    head = body[:200].strip() or f"HTTP {code}"
    if "image input is not supported" in low:
        return Err(
            "VISION_NO_PROJECTOR",
            f"The server answering as `{model.VISION_MODEL}` was started "
            f"without its vision projector ({head}). Nothing was looked at.",
            retryable=False, status=503,
            remedies=[_operator(f"launch `{model.VISION_MODEL}` with --mmproj "
                                "Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf (config.yaml)",
                                "server configuration")])
    if any(s in low for s in ("out of memory", "cudamalloc", "cuda_error_out_of_memory",
                              "failed to allocate")):
        return Err(
            "VISION_OUT_OF_MEMORY",
            f"The vision model ran out of GPU memory on the A4000 ({head}). "
            f"Nothing was looked at.",
            retryable=False, status=507,
            remedies=[_operator("check the A4000 with nvidia-smi: bonsai-vision "
                                "needs 8,265-9,449 MiB (config.yaml estimate) and "
                                "retrieval, Laya or an image generation may hold it"),
                      _agent("tell the user the image could not be looked at "
                             "right now", "retrying in this turn is unlikely to help")])
    if "exceed" in low and "context" in low:
        return Err(
            "VISION_CONTEXT_FULL",
            f"The image and question do not fit the vision model's context "
            f"({head}). Nothing was looked at.",
            retryable=False, status=413,
            remedies=[_operator("raise -c on bonsai-vision, or YAMADORI_VISION_CTX "
                                "if it was changed", "server configuration"),
                      _agent("tell the user the image is too detailed to look at here",
                             "the user can send a smaller crop")])
    if any(s in low for s in ("failed to load image", "failed to decode",
                              "failed to process image", "invalid uri",
                              "unsupported image", "webp")):
        return Err(
            "VISION_REJECTED_IMAGE",
            f"The vision server could not decode the image ({head}). Nothing "
            f"was looked at.",
            retryable=False, status=415,
            remedies=[{"fixable_by": "user",
                       "action": "attach the picture as a PNG or JPEG file",
                       "effect": "the vision server can decode it"},
                      _agent("tell the user the image format could not be read",
                             "the user can convert it")])
    if code == 404 or ("model" in low and ("not found" in low or "could not find" in low)):
        return Err(
            "VISION_NOT_CONFIGURED",
            f"llama-swap has no model `{model.VISION_MODEL}` ({head}). Nothing "
            f"was looked at.",
            retryable=False, status=503,
            remedies=[_operator("add the `bonsai-vision` entry to config.yaml, or "
                                "set YAMADORI_VISION_MODEL to its name", "server configuration"),
                      _agent("tell the user image viewing is not available here",
                             "retrying cannot succeed")])
    if code == 503 and "loading" in low:
        return Err(
            "VISION_LOADING",
            f"The vision model is still loading ({head}). Nothing was looked at.",
            retryable=True, status=503,
            remedies=[_agent("call describe_image once more in a minute",
                             "the model finishes loading and stays resident for 15 minutes (ttl 900)")])
    if code in (502, 503, 504):
        # llama-swap answers this way when the process it starts exits or never
        # comes up, and when it was stopped mid-request -- an eviction and a
        # failed load look the same from here.
        return Err(
            "VISION_UNAVAILABLE",
            f"The vision model returned HTTP {code}: {head}. It failed to load, "
            f"crashed, or was evicted. Nothing was looked at.",
            retryable=False, status=503,
            remedies=[_operator(f"read llama-swap's log for `{model.VISION_MODEL}` "
                                "and nvidia-smi for A4000 free memory"),
                      _agent("tell the user the image could not be looked at",
                             "retrying in this turn is unlikely to succeed")])
    return Err(
        "VISION_FAILED",
        f"The vision model returned HTTP {code}: {head}. Nothing was looked at.",
        retryable=code >= 500, status=502,
        remedies=[_operator(f"read llama-swap's log for `{model.VISION_MODEL}`",
                            "the agent cannot see the server's logs")])


# ---------------------------------------------------------------- attachments
def empty() -> dict:
    """An attachment register: ids -> entries, and the shas of our own images
    this conversation holds a capability for. JSON-safe on purpose: prepare()
    carries it on the payload, and fan-out deep-copies payloads through JSON."""
    return {"images": {}, "media": []}


def _know(att: dict, sha: str) -> None:
    if sha not in att["media"]:
        att["media"].append(sha)


def _note_links(text: str, att: dict) -> None:
    """Every signed /media link in `text` whose signature verifies becomes a
    known image. The link itself is the capability; it is never fetched."""
    if "/media/" not in (text or ""):
        return
    for m in _MEDIA.finditer(text):
        q = urllib.parse.parse_qs(m.group(2) or "")
        ok, _why = images.verify(m.group(1), (q.get("exp") or [""])[0],
                                 (q.get("sig") or [""])[0])
        if ok:
            _know(att, m.group(1))


def _source_of(part: dict) -> tuple[str, str | None]:
    """(kind, value) for an image part: ("data", base64) for inline data,
    ("url", url) for a link, ("other", None) for anything else. Handles the
    OpenAI chat shape, the Responses shape and the Anthropic shape."""
    t = part.get("type")
    if t == "image":
        src = part.get("source") or {}
        if isinstance(src, dict):
            if src.get("type") == "base64" and isinstance(src.get("data"), str):
                return "data", src["data"]
            if isinstance(src.get("url"), str):
                return "url", src["url"]
        return "other", None
    iu = part.get("image_url")
    url = iu.get("url") if isinstance(iu, dict) else iu
    if not isinstance(url, str):
        return "other", None
    s = url.strip()
    if s[:5].lower() == "data:":
        head, _, b64 = s.partition(",")
        if ";base64" not in head.lower():
            return "other", None
        return "data", b64
    return "url", s


def _new_id(seed: bytes) -> str:
    return ID_PREFIX + hashlib.sha256(seed).hexdigest()[:10]


LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})


def our_hosts(*bases: str) -> frozenset:
    """The hosts a /media link in an IMAGE PART may name to count as ours:
    loopback, YAMADORI_PUBLIC_BASE's host, and each base given (the address
    this request reached us on)."""
    out = set(LOOPBACK)
    for b in (images.public_base(), *bases):
        h = urllib.parse.urlsplit(b or "").hostname
        if h:
            out.add(h.lower())
    return frozenset(out)


def _is_ours(url: str, hosts: frozenset | None) -> bool:
    """A /media link names OUR store when it is relative (no host) or its
    host is one of `hosts`. Nothing is resolved: a name is compared."""
    try:
        parts = urllib.parse.urlsplit(url or "")
    except ValueError:
        return False
    if not parts.netloc:
        return not parts.scheme
    if parts.scheme not in ("http", "https"):
        return False
    return (parts.hostname or "").lower() in (hosts or our_hosts())


def _admit(part: dict, att: dict, hosts: frozenset | None = None) -> str:
    """Register one image part; return the placeholder the text model sees."""
    kind, value = _source_of(part)
    if kind == "url":
        m = _MEDIA.search(value or "")
        if m and _is_ours(value, hosts):
            # OUR SIGNED LINK AS AN ATTACHMENT (pre-deploy review,
            # 2026-09-24). A client that looks at our generated image through
            # its own vision call (Hermes' vision_analyze, which sends it to
            # the main provider -- us) sends the signed /media link back as
            # an image part. The signature and expiry are verified, the host
            # must be ours or loopback, and the bytes are read from the media
            # store (images.png_bytes: 64 hex or nothing). Never fetched.
            sha = m.group(1)
            q = urllib.parse.parse_qs(m.group(2) or "")
            ok, why = images.verify(sha, (q.get("exp") or [""])[0],
                                    (q.get("sig") or [""])[0])
            iid = _new_id(("media\x00" + sha).encode())
            if not ok:
                att["images"][iid] = {"id": iid, "source": "media",
                                      "format": None, "bytes": None,
                                      "b64": None, "error": "IMAGE_LINK_INVALID",
                                      "detail": why}
                return (f"[{iid}: a link to an image on this server whose "
                        f"signature does not verify ({why}), so it was not "
                        f"read. Tell the user the link has expired or was "
                        f"altered, and ask for the image file or a fresh "
                        f"link.]")
            _know(att, sha)
            data = images.png_bytes(sha)
            fmt = sniff(data) if data else None
            if data is None or not fmt or len(data) > max_bytes():
                err = ("IMAGE_NOT_FOUND" if data is None else
                       "IMAGE_TOO_LARGE" if fmt else "NOT_AN_IMAGE")
                att["images"][iid] = {"id": iid, "source": "media",
                                      "format": None,
                                      "bytes": len(data) if data else None,
                                      "b64": None, "error": err, "sha": sha,
                                      "detail": "the stored file is not an "
                                                "image" if data else None}
                return (f"[{iid}: a link to an image this server made, which "
                        f"could not be read from its media store ({err}). "
                        f"Tell the user.]")
            att["images"][iid] = {"id": iid, "source": "media", "format": fmt,
                                  "bytes": len(data), "sha": sha,
                                  "b64": base64.b64encode(data).decode("ascii"),
                                  "error": None}
            size = f"{fmt.upper()}, {max(1, len(data) // 1024):,} KB"
            if not enabled():
                return (f"[{iid}: an image this server made ({size}). Image "
                        f"viewing is switched off on this server, so its "
                        f"content is unavailable; tell the user you cannot "
                        f"see it.]")
            return (f"[{iid}: an image this server made ({size}), attached by "
                    f"its link. You cannot see it directly. To look at it, "
                    f"call describe_image with image \"{iid}\" and a question "
                    f"about it.]")
        iid = _new_id(("url\x00" + (value or "")).encode("utf-8", "replace"))
        att["images"][iid] = {"id": iid, "source": "link", "format": None,
                              "bytes": None, "b64": None,
                              "error": "IMAGE_URL_NOT_ALLOWED"}
        return (f"[{iid}: a link to an image on another site. It was NOT "
                f"downloaded: this server never fetches image links, and "
                f"looks only at images attached to the conversation or made "
                f"here. Tell the user plainly that you could not see this "
                f"image, and ask them to attach the image file itself.]")
    if kind != "data":
        iid = _new_id(json.dumps(part, sort_keys=True, default=str)[:4096].encode())
        att["images"][iid] = {"id": iid, "source": "attached", "format": None,
                              "bytes": None, "b64": None, "error": "NOT_AN_IMAGE",
                              "detail": "not a base64 data: URI or a link"}
        return (f"[{iid}: an attachment that could not be read as an image "
                f"(it was not image data or a link). Tell the user, and ask "
                f"for a PNG or JPEG file.]")

    b64 = re.sub(r"\s+", "", value or "")
    # Refuse by size BEFORE decoding: 4 base64 characters carry 3 bytes.
    est = len(b64) * 3 // 4
    if est > max_bytes() + 3:
        iid = _new_id(b64[:65536].encode("ascii", "replace"))
        att["images"][iid] = {"id": iid, "source": "attached", "format": None,
                              "bytes": est, "b64": None, "error": "IMAGE_TOO_LARGE"}
        return (f"[{iid}: an attached image of about {est // 1024:,} KB, over "
                f"this server's {max_bytes() // (1024 * 1024)} MB limit for "
                f"looking at images. Tell the user, and ask for a smaller copy.]")
    try:
        data = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError):
        data = None
    fmt = sniff(data) if data else None
    if not fmt:
        iid = _new_id(data if data else b64[:65536].encode("ascii", "replace"))
        att["images"][iid] = {"id": iid, "source": "attached", "format": None,
                              "bytes": len(data) if data else None, "b64": None,
                              "error": "NOT_AN_IMAGE",
                              "detail": ("the data is not valid base64" if data is None
                                         else "the bytes are not PNG, JPEG, GIF, BMP or WebP")}
        return (f"[{iid}: an attachment that is not an image this server can "
                f"read. Tell the user, and ask for a PNG or JPEG file.]")
    iid = _new_id(data)
    att["images"][iid] = {"id": iid, "source": "attached", "format": fmt,
                          "bytes": len(data), "b64": b64, "error": None}
    size = f"{fmt.upper()}, {max(1, len(data) // 1024):,} KB"
    if not enabled():
        return (f"[{iid}: an attached image ({size}). Image viewing is switched "
                f"off on this server, so its content is unavailable; tell the "
                f"user you cannot see it.]")
    return (f"[{iid}: an attached image ({size}). You cannot see it directly. "
            f"To look at it, call describe_image with image \"{iid}\" and a "
            f"question about it.]")


def extract(messages: list, hosts: frozenset | None = None
            ) -> tuple[list, dict]:
    """(the messages the text model sees, the attachment register).

    `hosts`: the hosts an image part's /media link may name to be read as
    ours (our_hosts); default loopback and YAMADORI_PUBLIC_BASE.

    Each image part becomes a text part naming its id; each other media part
    becomes a text part saying it was not passed on. Text is untouched, and
    signed /media links in it are noted as images this conversation holds.
    When nothing changes the SAME list is returned, so a text-only request
    keeps a byte-identical prefix (strip_thinking's rule).
    """
    att = empty()
    out: list = []
    changed_any = False
    for m in messages or []:
        if not isinstance(m, dict):
            out.append(m)
            continue
        c = m.get("content")
        if isinstance(c, str):
            _note_links(c, att)
            out.append(m)
            continue
        if not isinstance(c, list):
            out.append(m)
            continue
        parts: list = []
        changed = False
        for p in c:
            kind = p.get("type") if isinstance(p, dict) else None
            if kind == "text":
                _note_links(p.get("text") or "", att)
                parts.append(p)
            elif kind in IMAGE_PARTS:
                parts.append({"type": "text", "text": _admit(p, att,
                                                             hosts)})
                changed = True
            elif kind in OTHER_MEDIA:
                parts.append({"type": "text", "text": (
                    f"[an attachment of type {kind} was here. This server "
                    f"passes only text and images to the model, so its "
                    f"content is unavailable; tell the user if it matters.]")})
                changed = True
            else:
                parts.append(p)
        if changed:
            changed_any = True
            out.append(dict(m, content=parts))
        else:
            out.append(m)
    return (out if changed_any else messages), att


def summary(att: dict | None) -> list[dict]:
    """The register for x_yamadori: id, source, format, size, error. Never the
    bytes."""
    return [{k: e.get(k) for k in ("id", "source", "format", "bytes", "error")}
            for e in ((att or {}).get("images") or {}).values()]


def note_generated(result: str, att: dict | None) -> None:
    """generate_image's result in this request: the image it made may now be
    looked at by its sha or its url."""
    if att is None:
        return
    try:
        d = json.loads(result)
    except (TypeError, ValueError):
        return
    if isinstance(d, dict) and d.get("ok") and isinstance(d.get("url"), str):
        m = _MEDIA.search(d["url"])
        if m:
            _know(att, m.group(1))


# ------------------------------------------------------------------ resolving
def _from_store(sha: str) -> dict:
    data = images.png_bytes(sha)
    if data is None:
        raise not_found(sha)
    if len(data) > max_bytes():
        raise too_large(len(data))
    fmt = sniff(data)
    if not fmt:
        raise not_an_image("the stored file is not an image")
    return {"data": data, "format": fmt, "source": "generated", "label": sha[:16]}


def resolve(ref, att: dict | None) -> dict:
    """{data, format, source, label} for the image `ref` names, or ImageError.

    `ref` is TEXT A MODEL WROTE. It is matched against ids, never opened: an
    attached id is looked up in `att`; a /media link or a sha is read from the
    media store through images.png_bytes (64 hex characters or nothing), and
    only when a signature verifies or the conversation already holds it; a
    URL is refused without being fetched; anything else is unknown.
    """
    att = att if att is not None else empty()
    if not isinstance(ref, str) or not ref.strip():
        raise bad_args("`image` must be a string: an attached image's id or "
                       "the url generate_image returned.",
                       "call again with image set to an id from the conversation")
    r = ref.strip().strip("`'\"<>()[] ")
    low = r.lower()
    if _ATT_ID.fullmatch(low):
        e = (att.get("images") or {}).get(low)
        if e is None:
            raise unknown(ref, att, "no attached image has that id in this "
                                    "conversation.")
        code = e.get("error")
        if code == "IMAGE_URL_NOT_ALLOWED":
            raise url_not_allowed()
        if code == "IMAGE_LINK_INVALID":
            raise link_invalid(e.get("detail") or "it does not verify")
        if code == "IMAGE_NOT_FOUND":
            raise not_found(e.get("sha") or "?" * 16)
        if code == "IMAGE_TOO_LARGE":
            raise too_large(int(e.get("bytes") or 0))
        if code:
            raise not_an_image(e.get("detail") or "unreadable attachment")
        return {"data": base64.b64decode(e["b64"]), "format": e["format"],
                "source": ("generated" if e.get("source") == "media"
                           else "attached"), "label": e["id"]}
    m = _MEDIA.search(r)
    if m:
        sha = m.group(1)
        q = urllib.parse.parse_qs(m.group(2) or "")
        sig, exp = (q.get("sig") or [""])[0], (q.get("exp") or [""])[0]
        if sha in att.get("media", []):
            return _from_store(sha)
        if not sig:
            raise unknown(ref, att, "a /media link without its signature names "
                                    "no image this conversation holds.")
        ok, why = images.verify(sha, exp, sig)
        if not ok:
            raise link_invalid(why)
        return _from_store(sha)
    bare = low[:-4] if low.endswith(".png") else low
    if _SHA.fullmatch(bare):
        if bare in att.get("media", []):
            return _from_store(bare)
        raise unknown(ref, att, "that sha was not made in this request and no "
                                "signed link to it appears in the conversation.")
    if re.match(r"^(https?|ftp|file)://|^www\.", low):
        raise url_not_allowed()
    if low.startswith("data:"):
        raise unknown(ref, att, "pass the image's id, not its data.")
    raise unknown(ref, att, "it is not an image id. Images are named by id; "
                            "file paths and other text are not read.")


def _question(q) -> str:
    if not isinstance(q, str) or not q.strip():
        raise bad_args("`question` must be a non-empty string saying what to "
                       "find out about the image.",
                       "call again with a question, e.g. 'Describe everything "
                       "in this image'")
    if len(q) > MAX_QUESTION_CHARS:
        raise bad_args(f"`question` is {len(q)} characters; the limit is "
                       f"{MAX_QUESTION_CHARS}.", "call again with a shorter question")
    return q.strip()


# ------------------------------------------------------------------- describe
def describe(data: bytes, fmt: str, question: str) -> dict:
    """Ask the vision model `question` about one image. Returns
    {answer, seconds, usage, finish}; raises ImageError."""
    if not enabled():
        raise vision_off()
    q = _question(question)
    uri = f"data:{MIME[fmt]};base64,{base64.b64encode(data).decode('ascii')}"
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": uri}},
                    {"type": "text", "text": q}]}]
    cap = thinking_cap(q)
    with admission.image_lane() as got:
        if not got:
            raise busy()
        t0 = time.time()
        try:
            d = model.chat(messages, effort=EFFORT, max_tokens=tiers.A_MIN,
                           timeout=timeout(), cap=cap, model=model.VISION_MODEL)
        except gpu_room.NoRoom as e:
            # The A4000 coordinator could not make room (model.post holds it
            # around the request): busy (retryable) or no room at all.
            raise Err(e.code, e.reason + " Nothing was looked at.",
                      retryable=e.retryable, remedies=e.remedies,
                      status=429 if e.retryable else 507, **e.facts) from None
        except urllib.error.HTTPError as e:
            try:
                detail = e.read().decode("utf-8", "replace")
            except Exception:                                    # noqa: BLE001
                detail = ""
            raise _classify_http(e.code, detail) from None
        except (socket.timeout, TimeoutError):
            raise _timeout_err() from None
        except urllib.error.URLError as e:
            if isinstance(e.reason, (socket.timeout, TimeoutError)):
                raise _timeout_err() from None
            raise _down(str(e.reason)) from None
        except (ConnectionError, OSError) as e:
            raise _down(f"{type(e).__name__}: {e}") from None
        except ValueError as e:
            raise Err("VISION_FAILED",
                      f"The vision model's answer was not JSON ({e}). Nothing "
                      f"came back.", retryable=False, status=502,
                      remedies=[_operator(f"read llama-swap's log for `{model.VISION_MODEL}`")]) from None
        seconds = round(time.time() - t0, 2)
    finish = ((d.get("choices") or [{}])[0] or {}).get("finish_reason")
    try:
        text = model.answer(d)
    except model.BudgetEvent as e:
        # A length finish is a budget event, never an answer (AGENTS.md).
        raise Err("VISION_BUDGET",
                  f"The vision model reached its token limit before answering "
                  f"({e}). This is a budget event, not a description.",
                  retryable=True, status=502,
                  remedies=[_agent("call again with a narrower question about "
                                   "one part of the image", "a shorter answer fits"),
                            _operator("raise -c on bonsai-vision; the thinking cap "
                                      "follows YAMADORI_VISION_CTX", "server configuration")],
                  seconds=seconds) from None
    if not (text or "").strip():
        raise Err("VISION_EMPTY",
                  "The vision model finished without writing an answer. Nothing "
                  "came back.", retryable=False, status=502,
                  remedies=[_operator(f"read llama-swap's log for `{model.VISION_MODEL}`",
                                      "no arguments change a server that returns nothing"),
                            _agent("tell the user the image could not be described",
                                   "the user is not left waiting")])
    return {"answer": text.strip(), "seconds": seconds,
            "usage": d.get("usage") or {}, "finish": finish}


# ----------------------------------------------------------------- model tool
# A description is a trigger condition (AGENTS.md "Tool descriptions are
# prompts"): the question it answers first, the contrast with the tool it
# could be confused with (generate_image), then the phrasings that fire it.
TOOL = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": (
            "Answers a question about what is IN an image. Use it whenever you "
            "need to see a picture: an image the user attached (the "
            "conversation names it like image-3f9a1c2b7d) or one generate_image "
            "made (pass the url it returned). generate_image turns words into "
            "a new picture; describe_image turns an existing picture back into "
            "words. Use it for 'what is in this image', 'describe this "
            "screenshot', 'read the text in this picture', 'what does this "
            "error dialog say', 'what colour is the button', 'does the mockup "
            "match what I asked for', 'check the image you just drew'. After "
            "drawing a mockup or design, look at it with this tool and refine "
            "the prompt if it is off. It sees images only, by id or url: "
            "files in the user's project are read with your client's own "
            "file tools."),
        "parameters": {
            "type": "object",
            "properties": {
                "image": {
                    "type": "string",
                    "description": (
                        "Which image: an attached image's id as the "
                        "conversation shows it (image-...), or the url "
                        "generate_image returned."),
                },
                "question": {
                    "type": "string",
                    "description": (
                        "What to find out, specifically: 'Describe everything "
                        "in this image', 'Read every label on this form', "
                        "'Is the logo centred, and what colours does it use?'"),
                },
            },
            "required": ["image", "question"],
        },
    },
}


def run_tool(args, att: dict | None, record: list | None = None) -> str:
    """Execute describe_image for the model. Always returns a JSON envelope;
    never raises. `att` is this request's attachment register (extract()).
    `record` collects one entry per call for x_yamadori.vision: ok, which
    image (attached id or sha prefix), source, format, bytes, seconds, token
    usage -- or the error code. Never the image or the question."""
    t0 = time.time()
    rec: dict = {"ok": False}
    try:
        if not isinstance(args, dict):
            raise bad_args("arguments must be an object with image and question.",
                           "call again with {\"image\": \"...\", \"question\": \"...\"}")
        if not enabled():
            raise vision_off()
        q = _question(args.get("question"))
        src = resolve(args.get("image"), att)
        rec.update(image=src["label"], source=src["source"],
                   format=src["format"], bytes=len(src["data"]))
        r = describe(src["data"], src["format"], q)
        u = r["usage"]
        rec.update(ok=True, seconds=r["seconds"], finish=r["finish"],
                   prompt_tokens=u.get("prompt_tokens"),
                   completion_tokens=u.get("completion_tokens"),
                   answer_chars=len(r["answer"]))
        return json.dumps({
            "tool": TOOL_NAME, "ok": True, "image": src["label"],
            "source": src["source"], "answer": r["answer"],
            "seconds": r["seconds"],
            "note": ("This is what the vision model saw in the image. Use it "
                     "as the image's content; if the image is not what was "
                     "wanted, change the prompt and draw it again.")
            if src["source"] == "generated" else
            "This is what the vision model saw in the image. Use it as the image's content.",
        })
    except Err as e:
        rec.update(ok=False, error=e.code)
        return json.dumps(e.envelope(TOOL_NAME), default=str)
    except Exception as e:                                       # noqa: BLE001
        rec.update(ok=False, error="TOOL_RAISED")
        return json.dumps({
            "tool": TOOL_NAME, "ok": False, "error": "TOOL_RAISED",
            "reason": f"{type(e).__name__}: {e}", "retryable": False,
            "remedies": [_operator("check the proxy log for the traceback",
                                   "the tool threw; no arguments change that")]})
    finally:
        rec["ms"] = round((time.time() - t0) * 1000)
        if record is not None:
            record.append(rec)
