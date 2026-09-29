#!/usr/bin/env python
"""Vision: the one module between the proxy and the model that can see.

WHY IT EXISTS

The served model (Bonsai 2 27B on the RTX 5060 Ti) is text-only. It can draw
with yama_generate_image, and it cannot look at what it drew. A user's attached
image was worse than invisible: the image part went upstream to a server
started without a projector, and llama-server refuses that request outright
("image input is not supported - hint: ... provide the mmproj",
tools/server/server-common.cpp:1212 in the prism build), so the turn failed.

WHO LOOKS (LAYOUT V2, operator 2026-09-29: "Vision can go to second card and
swap in and out" -- reversing 2026-09-27's "fold vision into the main models
on the main card"). `main_sees()` reads the loaded main model's /props
(`modalities.vision`):

  - it SEES (the max-mode model; `bonsai` from 2026-09-27 until layout v2):
    a readable image in a user turn passes to it, and yama_describe_image
    asks it;
  - it does NOT (`bonsai` since layout v2: no --mmproj, the ~24.8k cells the
    projector cost went back to the KV line): every image is a placeholder
    naming its id, and yama_describe_image asks `bonsai-vision` -- the same
    27B with its projector on the A4000, loaded on demand by llama-swap
    (model.VISION_MODEL; mcp/gpu_room.py makes room and evicts, LRU), as
    before the fold. The route is the pre-fold one, restored.

Only when neither exists (YAMADORI_VISION_MODEL names the main model and it
has no projector) does yama_describe_image answer VISION_NO_PROJECTOR.

WHAT IT DOES

1. `normalise()` (= `extract()`, its old name) runs on every chat request,
   in proxy.prepare. It takes each image out of the messages, keeps it for
   this request under an id the model can pass (`image-<10 hex of its
   sha256>`), and puts a short text placeholder naming that id where the
   image was -- or, when the main model sees (`see=True`), an image part the
   main model reads directly (a data: URI built HERE from the bytes, MIME
   sniffed, never a client URL) followed by a short label naming the id.
   Only images in a USER turn pass through (a person's attachment, a
   harness's synthetic tool-media turn, our own /media link); an image in a
   `tool` message or printed as text stays a placeholder, and WebP (which
   llama-server decodes only with ffmpeg) always does. The text model sees the placeholder; nothing is dropped
   silently and nothing crashes. Other media parts (audio, video, files)
   get a placeholder too, for the same crash. EVERY FORM a harness sends
   lands in the one register (docs/VISION.md 5a; the recognisers are in
   mcp/image_input.py): an image part in any message, a harness's synthetic
   tool-media turn, a Responses `input_image` (a file id too), and image
   bytes PRINTED AS TEXT in a tool result or a user message. Each entry
   names its `form`.
2. `yama_describe_image`, a model tool, sends one image and a question to
   the model that can see (WHO LOOKS, above: `bonsai-vision` on the A4000,
   or the main model when it has a projector, on the child slot like any
   internal generation) and returns its answer as text: how an attached
   image, an image in a tool result, one yama_generate_image made, or an
   older one no longer held is looked at.

WHERE AN IMAGE MAY COME FROM, AND NOWHERE ELSE

This is the security contract, and it is why no argument is ever a path:

  - an image ATTACHED in this request's own messages, as base64 data, by id;
  - an image in OUR media store (index/media, by sha) that this conversation
    holds a capability for: a signed /media link that verifies (images.verify),
    a link that appeared in the conversation with a valid signature, or an
    image yama_generate_image made during this request.

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

It takes the IMAGE lane (admission.image_lane), one look at a time, as
before; the main server's slots do the rest.

Offline tests: mcp/test_vision.py (a fake server). The live check is the
deploy's (bench/engine_phase2.py smoke, mcp/test_live_stack.py images).
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
import image_input  # noqa: E402
import images  # noqa: E402
import model  # noqa: E402
import tiers  # noqa: E402

# `yama_describe_image` (operator, 2026-09-27; images.DESCRIBE_TOOL_NAME,
# the one constant): named apart from anything a harness offers. The old
# name, `describe_image`, is still read in stored ledger rows and records
# (proxy.LEGACY_TOOL_NAMES).
TOOL_NAME = images.DESCRIBE_TOOL_NAME
LEGACY_TOOL_NAME = "describe_image"
GENERATE_TOOL_NAME = images.TOOL_NAME

# The effort the vision call is shaped at: model.chat's default, the same one
# summarize_text uses. Not measured for vision; the live check records
# completion tokens so it can be.
EFFORT = "low"

# THE CONTEXT ONE LOOK IS SIZED FOR: the A4000 copy's `-c 16384`
# (config.yaml `bonsai-vision`), and the size of one describe request (image +
# question + thinking + answer) when the main model looks instead, which the
# thinking cap below follows. YAMADORI_VISION_CTX, read at call time.
DEFAULT_CONTEXT = 16384

# THE LARGEST IMAGE THE PROJECTOR EMITS: image_input.MAX_IMAGE_TOKENS, read
# from the served mmproj (qwen3vl_merger) and the engine's clip.cpp
# (PROJECTOR_TYPE_QWEN3VL: set_limit_image_tokens(8, 4096)); see the note
# there. The thinking cap below budgets for the largest image.
IMAGE_TOKENS_MAX = image_input.MAX_IMAGE_TOKENS

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

IMAGE_PARTS = image_input.IMAGE_PARTS
OTHER_MEDIA = image_input.OTHER_MEDIA
TEXT_PARTS = image_input.TEXT_PARTS

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


# Whether the MAIN model has a projector, from its /props (`modalities.vision`,
# llama-server). YAMADORI_MAIN_VISION=1/0 overrides it (offline tests; an
# operator who knows). NO TIMER: the projector is a launch flag (--mmproj),
# so the answer can change only when the model's server is started again,
# and a deploy restarts every service, this proxy included (docs/ENGINES.md).
# The answer is kept per model name for the process's life, and dropped
# (note_no_projector) the moment the server says it cannot take an image
# ("image input is not supported"): api_errors.of_upstream and
# _classify_http. A /props that cannot be read is not kept: asked again on
# the next request.
_sees: dict = {"value": None, "why": None, "model": None}


def separate_copy() -> bool:
    """Is there a vision copy apart from the main model (model.VISION_MODEL,
    `bonsai-vision` on the A4000 by default)?"""
    return bool(model.VISION_MODEL) and model.VISION_MODEL != model.MODEL


def _vision_model() -> str:
    """The model yama_describe_image asks (WHO LOOKS): the conversation's main model when it has a projector (MAX
    MODE, mcp/max_mode.py: a max conversation asks the max model, which loads with its own projector), else the
    separate copy (YAMADORI_VISION_MODEL, `bonsai-vision` on the A4000)."""
    import max_mode
    cur = max_mode.current(model.MODEL)
    if cur != model.MODEL:
        # another tier's model (mcp/tier_models.py): it looks itself when its
        # llama-swap entry carries a projector (the table's `vision`, which
        # the deploy writes from the entry; config.yaml `flash-next --mmproj`),
        # else the separate copy on the A4000; no /props read of it here
        import tier_models
        if tier_models.table().vision(cur) is False and separate_copy():
            return model.VISION_MODEL
        return cur
    if main_sees() or not separate_copy():
        return cur
    return model.VISION_MODEL


def main_sees(refresh: bool = False) -> bool:
    v = _env("YAMADORI_MAIN_VISION")
    if v in ("0", "1"):
        _sees.update(value=v == "1", why="YAMADORI_MAIN_VISION")
        return v == "1"
    if not enabled():
        return False
    import max_mode
    # MAX MODE (mcp/max_mode.py): the model serving this request, cached per model; never a /props read of a model
    # that is off the card (it would load it)
    name = max_mode.current(model.MODEL)
    if not refresh and _sees["value"] is not None and _sees["model"] == name:
        return _sees["value"]
    if max_mode.blocks(name) and max_mode.bound_model() != name:
        return False
    try:
        props = model.props(name)
        seen = bool((props.get("modalities") or {}).get("vision"))
        _sees.update(value=seen, model=name,
                     why="/props modalities.vision")
    except Exception as e:                                      # noqa: BLE001
        # Not known: the text path (placeholders) for THIS request, and not
        # kept -- the next request asks again.
        _sees.update(value=None, model=None,
                     why=f"/props unreadable ({type(e).__name__})")
        return False
    return bool(_sees["value"])


def note_no_projector(detail: str = "") -> None:
    """The main server refused an image ("image input is not supported"):
    what /props said is no longer true (the server was started again without
    --mmproj). Kept as NO until this process restarts."""
    _sees.update(value=False, model=model.MODEL,
                 why=f"the server refused an image{': ' + detail[:120] if detail else ''}")


def no_projector() -> Err:
    return Err(
        "VISION_NO_PROJECTOR",
        f"The loaded main model (`{model.MODEL}`) has no vision projector and "
        f"no separate vision copy is configured (YAMADORI_VISION_MODEL names "
        f"the main model), so no image can be looked at on this server now "
        f"({_sees.get('why') or 'its /props says so'}). Nothing was looked at.",
        retryable=False, status=503,
        remedies=[_operator("unset YAMADORI_VISION_MODEL (the A4000 copy, "
                            "`bonsai-vision` in config.yaml, is the default), "
                            f"or serve `{model.MODEL}` with --mmproj",
                            "server configuration"),
                  _agent("tell the user you cannot see images with the model "
                         "loaded now", "the user is not left waiting")])


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
        action = ("call yama_describe_image again with one of the ids listed in "
                  "`available`: an attached image's id, or the url (or sha) "
                  "yama_generate_image returned")
    else:
        action = ("there is no image in this conversation to look at: ask the "
                  "user to attach one, or make one with yama_generate_image first")
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
        remedies=[_agent("use the url exactly as yama_generate_image returned it, or "
                         "draw the image again with yama_generate_image",
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
                  _agent("draw it again with yama_generate_image, then look at the new one",
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


def not_held(iid: str) -> Err:
    return Err(
        "IMAGE_NOT_HELD",
        f"{iid} is an older image of this request, over this server's limit "
        f"of {image_input.max_images()} images held per request (the newest "
        f"are held). Nothing was looked at.",
        retryable=False, status=413,
        remedies=[_agent("look at one of the newest images, listed in "
                         "`available`, or have the image shown again (read "
                         "the file again, or ask the user to attach it)",
                         "an image in the newest part of the conversation is held")])


def incomplete(fmt: str | None) -> Err:
    return Err(
        "IMAGE_INCOMPLETE",
        f"The {(fmt or 'image').upper()} data printed as text was cut off "
        f"before its end (a tool's output limit), so the image is "
        f"incomplete. Nothing was looked at.",
        retryable=True, status=422,
        remedies=[_agent("print a smaller copy whose base64 fits the tool's "
                         "output limit: a JPEG at lower quality or a narrower "
                         "width", "the whole image arrives and gets an id")])


def file_id_not_supported() -> Err:
    return Err(
        "IMAGE_FILE_ID",
        "That image was sent as a file id. This server is stateless and has "
        "no files API, so a file id names nothing here. Nothing was looked at.",
        retryable=False, status=400,
        remedies=[{"fixable_by": "user",
                   "action": "attach the image itself (a data URL), not a file id",
                   "effect": "the model can look at it"}])


def busy() -> Err:
    return Err(
        "VISION_BUSY",
        "Another image job (drawing or looking) is running, and "
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
                            f"it serves `{model.VISION_MODEL}`",
                            "the agent cannot start processes on the server"),
                  _agent("tell the user the image could not be looked at",
                         "retrying in this turn cannot succeed")])


def _timeout_err() -> Err:
    return Err(
        "VISION_TIMEOUT",
        f"The vision model did not answer within {timeout():.0f}s. Nothing "
        f"came back.",
        retryable=False, status=504,
        remedies=[_operator(f"check the `{model.VISION_MODEL}` process in "
                            "llama-swap's log and its card in nvidia-smi"),
                  _agent("tell the user looking at the image timed out",
                         "the user is not left waiting")])


def _classify_http(code: int, body: str) -> Err:
    low = body.lower()
    head = body[:200].strip() or f"HTTP {code}"
    if "image input is not supported" in low:
        note_no_projector(head)
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
            f"The vision model ran out of GPU memory ({head}). "
            f"Nothing was looked at.",
            retryable=False, status=507,
            remedies=[_operator(f"check `{model.VISION_MODEL}`'s card with "
                                "nvidia-smi and its VRAM line (--kv-vram-cells, "
                                "docs/ENGINES.md)"),
                      _agent("tell the user the image could not be looked at "
                             "right now", "retrying in this turn is unlikely to help")])
    if "exceed" in low and "context" in low:
        return Err(
            "VISION_CONTEXT_FULL",
            f"The image and question do not fit the vision model's context "
            f"({head}). Nothing was looked at.",
            retryable=False, status=413,
            remedies=[_operator(f"raise -c on `{model.VISION_MODEL}`, or "
                                "YAMADORI_VISION_CTX if it was changed",
                                "server configuration"),
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
            remedies=[_operator(f"check config.yaml's `{model.VISION_MODEL}` entry, or "
                                "set YAMADORI_VISION_MODEL to the main model's name",
                                "server configuration"),
                      _agent("tell the user image viewing is not available here",
                             "retrying cannot succeed")])
    if code == 503 and "loading" in low:
        return Err(
            "VISION_LOADING",
            f"The vision model is still loading ({head}). Nothing was looked at.",
            retryable=True, status=503,
            remedies=[_agent("call yama_describe_image once more in a minute",
                             "the model is loading; it stays resident while in use")])
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
                                "and nvidia-smi for its card's free memory"),
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
    if t == "input_image" and not part.get("image_url") \
            and isinstance(part.get("file_id"), str):
        # A Responses file id: there is no files API here (docs/VISION.md 5a).
        return "file_id", part["file_id"]
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


def _put(att: dict, iid: str, entry: dict) -> None:
    """Register an entry; an id seen again moves to the NEWEST position (the
    count cap holds the newest images)."""
    att["images"].pop(iid, None)
    att["images"][iid] = entry


def _size(fmt: str, n: int) -> str:
    return f"{fmt.upper()}, {max(1, n // 1024):,} KB"


def _register(att: dict, data: bytes, fmt: str, form: str, where: str,
              b64: str | None = None) -> str:
    """Hold one readable image; return its placeholder: a pure function of
    the bytes and of `where` it came in, so the same history renders the
    same text on every request."""
    iid = _new_id(data)
    _put(att, iid, {"id": iid, "source": "attached", "form": form,
                    "format": fmt, "bytes": len(data),
                    "sha": hashlib.sha256(data).hexdigest(),
                    "b64": b64 or base64.b64encode(data).decode("ascii"),
                    "error": None})
    if not enabled():
        return (f"[{iid}: {where} ({_size(fmt, len(data))}). Image viewing is "
                f"switched off on this server, so its content is unavailable; "
                f"tell the user you cannot see it.]")
    return (f"[{iid}: {where} ({_size(fmt, len(data))}). You cannot see it "
            f"directly. To look at it, call yama_describe_image with image "
            f"\"{iid}\" and a question about it.]")


# Where an image came in, as its placeholder says it (docs/VISION.md 5b).
# "an attached image" is the wording every placeholder had before 2026-09-26,
# kept byte for byte so an existing conversation's prefix does not move.
WHERE = {"user_part": "an attached image",
         "tool_part": "an image from a tool result",
         "tool_media_turn": "an image from a tool result",
         "responses_output": "an image from a tool result"}


def _admit(part: dict, att: dict, hosts: frozenset | None = None,
           form: str = "user_part") -> str:
    """Register one image part; return the placeholder the text model sees.
    `form` says how it came (image_input.FORMS); a link or a file id records
    its own form."""
    kind, value = _source_of(part)
    if kind == "file_id":
        iid = _new_id(("file_id\x00" + value).encode("utf-8", "replace"))
        _put(att, iid, {"id": iid, "source": "attached", "form": "file_id",
                        "format": None, "bytes": None, "b64": None,
                        "error": "IMAGE_FILE_ID"})
        return (f"[{iid}: an image sent as a file id. This server is "
                f"stateless and has no files API, so a file id names nothing "
                f"here and the image was not seen. Ask for the image itself.]")
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
                _put(att, iid, {"id": iid, "source": "media",
                                "form": "media_link", "format": None,
                                "bytes": None, "b64": None,
                                "error": "IMAGE_LINK_INVALID", "detail": why})
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
                _put(att, iid, {"id": iid, "source": "media",
                                "form": "media_link", "format": None,
                                "bytes": len(data) if data else None,
                                "b64": None, "error": err, "sha": sha,
                                "detail": "the stored file is not an "
                                          "image" if data else None})
                return (f"[{iid}: a link to an image this server made, which "
                        f"could not be read from its media store ({err}). "
                        f"Tell the user.]")
            _put(att, iid, {"id": iid, "source": "media", "form": "media_link",
                            "format": fmt, "bytes": len(data), "sha": sha,
                            "b64": base64.b64encode(data).decode("ascii"),
                            "error": None})
            size = _size(fmt, len(data))
            if not enabled():
                return (f"[{iid}: an image this server made ({size}). Image "
                        f"viewing is switched off on this server, so its "
                        f"content is unavailable; tell the user you cannot "
                        f"see it.]")
            return (f"[{iid}: an image this server made ({size}), attached by "
                    f"its link. You cannot see it directly. To look at it, "
                    f"call yama_describe_image with image \"{iid}\" and a question "
                    f"about it.]")
        iid = _new_id(("url\x00" + (value or "")).encode("utf-8", "replace"))
        _put(att, iid, {"id": iid, "source": "link", "form": "url",
                        "format": None, "bytes": None, "b64": None,
                        "error": "IMAGE_URL_NOT_ALLOWED"})
        return (f"[{iid}: a link to an image on another site. It was NOT "
                f"downloaded: this server never fetches image links, and "
                f"looks only at images attached to the conversation or made "
                f"here. Tell the user plainly that you could not see this "
                f"image, and ask them to attach the image file itself.]")
    if kind != "data":
        iid = _new_id(json.dumps(part, sort_keys=True, default=str)[:4096].encode())
        _put(att, iid, {"id": iid, "source": "attached", "form": form,
                        "format": None, "bytes": None, "b64": None,
                        "error": "NOT_AN_IMAGE",
                        "detail": "not a base64 data: URI or a link"})
        return (f"[{iid}: an attachment that could not be read as an image "
                f"(it was not image data or a link). Tell the user, and ask "
                f"for a PNG or JPEG file.]")

    b64 = re.sub(r"\s+", "", value or "")
    # Refuse by size BEFORE decoding: 4 base64 characters carry 3 bytes.
    est = len(b64) * 3 // 4
    if est > max_bytes() + 3:
        iid = _new_id(b64[:65536].encode("ascii", "replace"))
        _put(att, iid, {"id": iid, "source": "attached", "form": form,
                        "format": None, "bytes": est, "b64": None,
                        "error": "IMAGE_TOO_LARGE"})
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
        _put(att, iid, {"id": iid, "source": "attached", "form": form,
                        "format": None,
                        "bytes": len(data) if data else None, "b64": None,
                        "error": "NOT_AN_IMAGE",
                        "detail": ("the data is not valid base64" if data is None
                                   else "the bytes are not PNG, JPEG, GIF, BMP or WebP")})
        return (f"[{iid}: an attachment that is not an image this server can "
                f"read. Tell the user, and ask for a PNG or JPEG file.]")
    return _register(att, data, fmt, form,
                     WHERE.get(form, WHERE["user_part"]), b64=b64)


def _inline(text: str, att: dict, in_tool: bool) -> str:
    """Image bytes PRINTED AS TEXT (image_input.find_inline) become
    placeholders in place; the rest of the text is kept. Returns `text`
    itself when there are none."""
    runs = image_input.find_inline(text)
    if not runs:
        return text
    where = ("an image printed as text in a tool result" if in_tool
             else "an image pasted as text in the message")
    what = ("image data printed as text in a tool result" if in_tool
            else "image data pasted as text in the message")
    out: list[str] = []
    pos = 0
    for r in runs:
        out.append(text[pos:r["start"]])
        pos = r["end"]
        data, fmt = r["data"], r["format"]
        if r["status"] == "complete" and len(data) <= max_bytes():
            out.append(_register(att, data, fmt, "inline_text", where))
            continue
        seed = text[r["start"]:r["end"]]
        iid = _new_id(("inline\x00" + seed[:65536]).encode("ascii", "replace"))
        if r["status"] == "complete":
            _put(att, iid, {"id": iid, "source": "attached",
                            "form": "inline_text", "format": None,
                            "bytes": len(data), "b64": None,
                            "error": "IMAGE_TOO_LARGE"})
            out.append(f"[{iid}: {what} ({_size(fmt, len(data))}), over this "
                       f"server's {max_bytes() // (1024 * 1024)} MB limit for "
                       f"looking at images; the text was replaced. To look at "
                       f"it, print a smaller copy.]")
        elif r["status"] == "cut":
            _put(att, iid, {"id": iid, "source": "attached",
                            "form": "inline_text", "format": fmt,
                            "bytes": None, "b64": None,
                            "error": "IMAGE_INCOMPLETE"})
            remedy = ("To look at it, print a smaller copy whose base64 fits "
                      "the tool's output limit, for example a JPEG at lower "
                      "quality or a narrower width." if in_tool else
                      "Ask the user to attach the image file instead.")
            out.append(f"[{iid}: {what} ({fmt.upper()}), cut off after "
                       f"{r['chars']:,} base64 characters, so the image is "
                       f"incomplete and cannot be looked at. {remedy}]")
        else:
            _put(att, iid, {"id": iid, "source": "attached",
                            "form": "inline_text", "format": None,
                            "bytes": len(data), "b64": None,
                            "error": "NOT_AN_IMAGE",
                            "detail": f"declared image/{r['declared']}"})
            out.append(f"[{iid}: {what}, declared as image/{r['declared']} "
                       f"({max(1, len(data) // 1024):,} KB), a format this "
                       f"server cannot look at; the text was replaced. A PNG "
                       f"or JPEG can be looked at.]")
    out.append(text[pos:])
    return "".join(out)


def _hold_newest(att: dict) -> None:
    """At most image_input.max_images() images keep their bytes: the newest.
    An older one keeps its id and placeholder (the text never changes) and
    yama_describe_image on it says IMAGE_NOT_HELD."""
    cap = image_input.max_images()
    held = [i for i, e in att["images"].items() if e.get("b64")]
    if len(held) <= cap:
        return
    for iid in held[:len(held) - cap]:
        e = att["images"][iid]
        e["b64"] = None
        e["error"] = "IMAGE_NOT_HELD"
    att["not_held"] = len(held) - cap


# THE FORMS THAT PASS THROUGH to a main model that sees: an image in a USER
# turn -- a person's attachment, a harness's synthetic tool-media turn, our
# own signed /media link. llama-server reads image parts in any message, but
# a `tool` message's image and base64 printed as text keep their placeholder
# (the model looks at them with yama_describe_image, as before), and so does
# a format the server cannot decode without ffmpeg.
PASS_FORMS = ("user_part", "tool_media_turn", "media_link")
PASS_FORMATS = ("png", "jpeg", "gif", "bmp")


def _seen_label(e: dict, where: str) -> str:
    """The text after an image part the model sees: its id, so the model can
    still name it (yama_describe_image, the user). A pure function of the
    bytes and where it came in, like every placeholder."""
    return (f"[{e['id']}: {where} ({_size(e['format'], e['bytes'] or 0)}), "
            f"shown to you above.]")


def _pass_through(parts: list, att: dict) -> tuple[list, int]:
    """Markers left by normalise(see=True) become [image part, label] when
    the image is still held (the newest image_input.max_images()), else its
    ordinary placeholder. Returns the parts and how many images passed."""
    out, n = [], 0
    for p in parts:
        if not (isinstance(p, dict) and "_see" in p):
            out.append(p)
            continue
        iid, where, placeholder = p["_see"]
        e = att["images"].get(iid) or {}
        if e.get("b64") and not e.get("error") and e.get("format") in PASS_FORMATS:
            uri = f"data:{MIME[e['format']]};base64,{e['b64']}"
            out.append({"type": "image_url", "image_url": {"url": uri}})
            out.append({"type": "text", "text": _seen_label(e, where)})
            e["passed"] = True
            n += 1
        else:
            out.append({"type": "text", "text": placeholder})
    return out, n


def normalise(messages: list, hosts: frozenset | None = None,
              see: bool = False) -> tuple[list, dict]:
    """THE NORMALISER: (the messages the model sees, the register).

    `hosts`: the hosts an image part's /media link may name to be read as
    ours (our_hosts); default loopback and YAMADORI_PUBLIC_BASE.

    `see`: the main model has a projector (main_sees()). Then a readable
    image in a user turn (PASS_FORMS) goes to it as an image part -- a data:
    URI built here from the held bytes, never the client's URL -- followed by
    a label naming its id; every other image keeps its placeholder. Only the
    newest images_input.max_images() pass (the held ones); an older one is a
    placeholder, exactly as without `see`.

    Each image -- a chat `image_url` part, an Anthropic `image` block, a
    Responses `input_image` (url or file id), in ANY message: user, a
    harness's synthetic tool-media turn (image_input.tool_media_turn), a
    `tool` message -- becomes a text part naming its id; each other media
    part becomes a text part saying it was not passed on. Image bytes
    printed as text in a user or tool message (image_input.find_inline)
    become a placeholder in place, the rest of the text kept. Responses text
    parts (`input_text`, `output_text`) become `text` parts. Every other
    text is untouched, and signed /media links in it are noted as images
    this conversation holds. Every placeholder is a pure function of the
    image's bytes and where it came in, so the same history renders the same
    bytes on every request. When nothing changes the SAME list is returned,
    so a text-only request keeps a byte-identical prefix (strip_thinking's
    rule). Nothing is fetched and nothing is opened.
    """
    att = empty()
    msgs = messages or []
    out: list = []
    changed_any = False
    for i, m in enumerate(msgs):
        if not isinstance(m, dict):
            out.append(m)
            continue
        role = m.get("role")
        in_tool = role in ("tool", "function")
        # Printed image bytes are looked for where a person or a tool put
        # text: never in the model's own turns (the slot generated those).
        scan = in_tool or role == "user"
        c = m.get("content")
        if isinstance(c, str):
            _note_links(c, att)
            new = _inline(c, att, in_tool) if scan else c
            if new is not c:
                changed_any = True
                out.append(dict(m, content=new))
            else:
                out.append(m)
            continue
        if not isinstance(c, list):
            out.append(m)
            continue
        media_turn = role == "user" and image_input.is_tool_media(msgs, i)
        parts: list = []
        changed = False
        for p in c:
            kind = p.get("type") if isinstance(p, dict) else None
            if kind in TEXT_PARTS:
                t = p.get("text")
                t = t if isinstance(t, str) else ""
                _note_links(t, att)
                new = _inline(t, att, in_tool) if scan else t
                if kind != "text" or new is not t:
                    parts.append({"type": "text", "text": new})
                    changed = True
                else:
                    parts.append(p)
            elif kind in IMAGE_PARTS:
                form = ("responses_output" if in_tool and kind == "input_image"
                        else "tool_part" if in_tool
                        else "tool_media_turn" if media_turn else "user_part")
                text = _admit(p, att, hosts, form)
                # _admit registers (or re-registers) exactly one entry, and
                # _put moves it to the newest position: the last key is it
                iid = next(reversed(att["images"])) if att["images"] else None
                e = att["images"].get(iid) or {}
                if see and iid and not e.get("error") and \
                        e.get("form", form) in PASS_FORMS and form in PASS_FORMS:
                    where = ("an image this server made" if e.get("form") == "media_link"
                             else WHERE.get(form, WHERE["user_part"]))
                    parts.append({"_see": (iid, where, text)})
                else:
                    parts.append({"type": "text", "text": text})
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
    _hold_newest(att)
    if see and changed_any:
        passed = 0
        for k, m in enumerate(out):
            c = m.get("content") if isinstance(m, dict) else None
            if isinstance(c, list) and any(isinstance(p, dict) and "_see" in p for p in c):
                parts, n = _pass_through(c, att)
                out[k] = dict(m, content=parts)
                passed += n
        att["passed"] = passed
    return (out if changed_any else messages), att


# The old name: proxy.prepare and the tests call it.
extract = normalise


def summary(att: dict | None) -> list[dict]:
    """The register for x_yamadori: id, source, form (image_input.FORMS),
    format, size, error. Never the bytes."""
    return [{k: e.get(k) for k in ("id", "source", "form", "format", "bytes",
                                   "error", "passed")}
            for e in ((att or {}).get("images") or {}).values()]


def note_generated(result: str, att: dict | None) -> None:
    """yama_generate_image's result in this request: the image it made may now be
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
                       "the url yama_generate_image returned.",
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
        if code == "IMAGE_NOT_HELD":
            raise not_held(e["id"])
        if code == "IMAGE_INCOMPLETE":
            raise incomplete(e.get("format"))
        if code == "IMAGE_FILE_ID":
            raise file_id_not_supported()
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
    if not main_sees() and not separate_copy():
        raise no_projector()
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
                           timeout=timeout(), cap=cap, model=_vision_model())
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
                            _operator("raise YAMADORI_VISION_CTX; the thinking cap "
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
# could be confused with (yama_generate_image), then the phrasings that fire it.
TOOL = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": (
            "Answers a question about what is IN an image. Use it whenever you "
            "need to see a picture: an image the user attached (the "
            "conversation names it like image-3f9a1c2b7d) or one yama_generate_image "
            "made (pass the url it returned). yama_generate_image turns words into "
            "a new picture; yama_describe_image turns an existing picture back into "
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
                        "yama_generate_image returned."),
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
    """Execute yama_describe_image for the model. Always returns a JSON envelope;
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
