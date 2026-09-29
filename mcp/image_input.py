#!/usr/bin/env python
"""Every image form a harness sends, recognised in one place (docs/VISION.md
Phase 1, section 5a).

WHY IT EXISTS

The main model is text-only; images reach it as placeholders naming an id
(`vision.extract`), and `yama_describe_image` looks at them. Before this module
only an image PART was recognised. Three more forms arrived as something
else:

  - OpenCode, Pi and Cline carry a tool's image (their `read` of a png) in a
    SYNTHETIC USER TURN after the tool result, because Chat Completions
    allows only text in a `tool` message. It was read as a new user turn:
    a new task, a kickoff, a route of its own.
  - a shell prints an image's bytes as base64 TEXT in a tool result
    (`base64 -w0 shot.jpg`; #44: a 57K-character data URL read back as
    text ended a run). It was 57K characters of text.
  - a Responses `input_image` with a `file_id` (no files API here).

THE API (other modules call these; the Responses adapter reuses them)

  vision.normalise(messages, hosts=None) -> (messages, register)
      THE NORMALISER. mcp/vision.py; `vision.extract` is the same function
      under its old name. Takes chat messages, returns what the text model
      sees and the request's register ({"images": {id: entry}, "media":
      [sha, ...]}). Chat parts (`image_url`), Anthropic `image` blocks and
      Responses parts (`input_image` {image_url | file_id}, `input_text`,
      `output_text`) are all accepted in any message's list content --
      so a Responses adapter may put `function_call_output`'s array into a
      `tool` message as it stands. Each register entry names its `form`
      (FORMS). A text-only request is returned as the SAME list object.
  tool_media_turn(messages, i) -> row | None
      Is messages[i] a harness's synthetic tool-media turn? The row of
      TOOL_MEDIA_TURNS that matched, or None. Works on the client's messages
      (image parts) and on the normalised ones (placeholders).
  last_is_tool_media(messages) -> row | None
      The same question for the request's last non-system message.
  find_inline(text) -> [run, ...]
      Image bytes printed as text: complete runs and cut ones.
  max_images(), max_body_bytes()
      The limits below, read per call.

WHAT IS NOT DONE HERE: nothing is fetched and nothing is opened. A URL is a
string compared, never requested (vision.py's security contract). An
attachment is never written to disk: the register lives for one request
("the conversation is the store", docs/IMAGEGEN.md); an entry keeps the full
sha256 of the bytes so a later phase can key a stored DESCRIPTION by it.

Light on purpose: route.py and deep.py import this module, so it imports
nothing of the proxy's at module level.
"""
from __future__ import annotations

import base64
import binascii
import os
import re

# ------------------------------------------------------------------ limits
# CHOICES, not measurements (docs/VISION.md 5e; PROTOCOL).
#
# MAX_IMAGES_PER_REQUEST: how many images one request's register HOLDS the
# bytes of. The newest are held. An older image over the limit keeps its
# placeholder, byte for byte (the placeholder is a pure function of the
# image's bytes, so a growing history never rewrites a past message and the
# slot's prefix holds), and yama_describe_image on it returns IMAGE_NOT_HELD with
# the remedy. Every image is still decoded to name it.
MAX_IMAGES_PER_REQUEST = 20
# MAX_BODY_BYTES: the request body server.py accepts on a POST, answered
# with 413 in OpenAI's error envelope. A 10 MB image (the per-image limit,
# vision.DEFAULT_MAX_BYTES) is ~13.4 MB of base64, and a request resends
# every image of its history.
MAX_BODY_BYTES = 64 * 1024 * 1024
# INLINE_MIN_CHARS: the shortest base64 run (after its line breaks are
# removed) read as printed image bytes. An id, a hash or a short token is
# never that long; the smallest real PNG screenshot is far longer.
INLINE_MIN_CHARS = 256


def _env_int(name: str, default: int) -> int:
    try:
        return int((os.environ.get(name) or "").strip() or default)
    except ValueError:
        return default


def max_images() -> int:
    """YAMADORI_VISION_MAX_IMAGES, default MAX_IMAGES_PER_REQUEST."""
    return max(1, _env_int("YAMADORI_VISION_MAX_IMAGES", MAX_IMAGES_PER_REQUEST))


def max_body_bytes() -> int:
    """YAMADORI_MAX_BODY_BYTES, default MAX_BODY_BYTES."""
    return max(1024, _env_int("YAMADORI_MAX_BODY_BYTES", MAX_BODY_BYTES))


# ------------------------------------------------------------------- forms
# How an image reached the register (x_yamadori.attachments[].form).
FORMS = ("user_part",          # an image part in a user (or system) message
         "tool_part",          # an image part inside a `tool` message
                               # (Hermes native: vision_analyze's result)
         "tool_media_turn",    # an image part in a harness's synthetic turn
         "responses_output",   # an `input_image` inside a `tool` message: a
                               # Responses function_call_output array as the
                               # adapter passes it on
         "inline_text",        # base64 printed as text (a tool result, or
                               # pasted in a user message)
         "media_link",         # our signed /media link as an image part
         "url",                # an http(s) link: recorded, never fetched
         "file_id")            # a Responses file id: no files API here

IMAGE_PARTS = frozenset({"image_url", "input_image", "image"})
OTHER_MEDIA = frozenset({"input_audio", "input_video", "audio", "video",
                         "file", "input_file", "document"})
TEXT_PARTS = frozenset({"text", "input_text", "output_text"})

# THE TOKENS ONE IMAGE COSTS THE MAIN MODEL (2026-09-27: the main server has
# its projector, and vision.normalise passes attached images to it as image
# parts). Computed the way the projector does it, from the SERVED mmproj and
# ENGINE, not assumed:
#   - Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf (models/manifest.yaml
#     bonsai-2-27b-mmproj-q8) says clip.projector_type = qwen3vl_merger,
#     clip.vision.patch_size = 16, clip.vision.spatial_merge_size = 2.
#   - llama-bonsai2-ada (engines/manifest.yaml), tools/mtmd/clip.cpp, case
#     PROJECTOR_TYPE_QWEN3VL: hparams.set_limit_image_tokens(8, 4096), so
#     image_min_pixels = 8 and image_max_pixels = 4096 tokens x (16 x 2)^2
#     pixels (clip-model.h set_limit_image_tokens); config.yaml passes no
#     --image-min/max-tokens.
#   - tools/mtmd/mtmd-image.cpp, mtmd_image_preprocessor_dyn_size: the image
#     is resized by calc_size_preserved_ratio (align 32 = patch x merge,
#     round each side, then scale into [min, max] pixels) and yields
#     (w / 32) x (h / 32) tokens (clip.cpp clip_n_output_tokens_x/_y).
# image_tokens() is that arithmetic: 64 tokens for a 256x256 PNG, whose whole
# request (image + its question) measured prompt_n 104 live on 2026-09-27.
# An image whose size cannot be read counts as the maximum, 4,096.
PATCH_SIZE = 16          # clip.vision.patch_size (mmproj)
SPATIAL_MERGE = 2        # clip.vision.spatial_merge_size (mmproj)
MIN_IMAGE_TOKENS = 8     # clip.cpp QWEN3VL: set_limit_image_tokens(8, 4096)
MAX_IMAGE_TOKENS = 4096
IMAGE_TOKENS = MAX_IMAGE_TOKENS   # the bound for an image of unknown size


def image_size(data: bytes) -> tuple[int, int] | None:
    """(width, height) from an image's header (PNG, GIF, BMP, JPEG), or None."""
    try:
        if data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR":
            return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
        if data[:6] in (b"GIF87a", b"GIF89a"):
            return int.from_bytes(data[6:8], "little"), int.from_bytes(data[8:10], "little")
        if data[:2] == b"BM" and len(data) >= 26:
            return (abs(int.from_bytes(data[18:22], "little", signed=True)),
                    abs(int.from_bytes(data[22:26], "little", signed=True)))
        if data[:3] == b"\xff\xd8\xff":
            i = 2
            while i + 9 < len(data):
                if data[i] != 0xFF:
                    return None
                marker = data[i + 1]
                if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                    i += 2
                    continue
                seg = int.from_bytes(data[i + 2:i + 4], "big")
                if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                    return (int.from_bytes(data[i + 7:i + 9], "big"),
                            int.from_bytes(data[i + 5:i + 7], "big"))
                i += 2 + seg
    except (IndexError, ValueError):
        return None
    return None


def image_tokens(width: int, height: int) -> int:
    """The projector's tokens for a width x height image (mtmd-image.cpp
    calc_size_preserved_ratio with align 32 and the QWEN3VL pixel limits)."""
    import math
    f = PATCH_SIZE * SPATIAL_MERGE
    if width <= 0 or height <= 0:
        return MAX_IMAGE_TOKENS
    lo, hi = MIN_IMAGE_TOKENS * f * f, MAX_IMAGE_TOKENS * f * f
    w_bar = max(f, int(round(width / f)) * f)
    h_bar = max(f, int(round(height / f)) * f)
    if h_bar * w_bar > hi:
        beta = math.sqrt(height * width / hi)
        h_bar = max(f, int(math.floor(height / beta / f)) * f)
        w_bar = max(f, int(math.floor(width / beta / f)) * f)
    elif h_bar * w_bar < lo:
        beta = math.sqrt(lo / (height * width))
        h_bar = int(math.ceil(height * beta / f)) * f
        w_bar = int(math.ceil(width * beta / f)) * f
    return (w_bar // f) * (h_bar // f)


def _part_tokens(p: dict) -> int:
    """An image part's tokens: from its data: URI's header when it can be
    read (only the first bytes are decoded), else the maximum."""
    url = ""
    if isinstance(p.get("image_url"), dict):
        url = p["image_url"].get("url") or ""
    elif isinstance(p.get("image_url"), str):
        url = p["image_url"]
    if url.startswith("data:") and ";base64," in url:
        head = url.split(";base64,", 1)[1][:87384]          # ~64 KB of bytes
        head = head[:len(head) - len(head) % 4]
        try:
            size = image_size(base64.b64decode(head))
        except (binascii.Error, ValueError):
            size = None
        if size:
            return image_tokens(*size)
    return MAX_IMAGE_TOKENS


def without_image_bytes(messages) -> tuple[list, int]:
    """(the messages with every image part's bytes replaced by a short text
    part, the TOKENS those images cost the projector). For SIZE estimates
    only: a 1 MB base64 image counted as text would be ~350,000 "tokens";
    the caller adds the returned tokens instead. The same list comes back
    when there are none."""
    if not isinstance(messages, list):
        return messages, 0
    out, n, changed = [], 0, False
    for m in messages:
        c = m.get("content") if isinstance(m, dict) else None
        if not isinstance(c, list):
            out.append(m)
            continue
        parts = []
        for p in c:
            if isinstance(p, dict) and p.get("type") in IMAGE_PARTS:
                parts.append({"type": "text", "text": "[image]"})
                n += _part_tokens(p)
            else:
                parts.append(p)
        if len(parts) != len(c) or any(a is not b for a, b in zip(parts, c)):
            changed = True
            out.append(dict(m, content=parts))
        else:
            out.append(m)
    return (out if changed else messages), n

# --------------------------------------------------- synthetic tool turns
# Each row is a harness's own words, READ FROM ITS SOURCE (the exact string
# literal, fetched with `gh api` on 2026-09-26). A row changes ROUTING only
# (the turn is a step in the client's loop, not a new task); it never
# changes what is read, so a model imitating one gains nothing.
TOOL_MEDIA_TURNS = (
    {"harness": "opencode",
     "text": "Attached media from tool result:",
     "source": "anomalyco/opencode dev b65de4d, packages/opencode/src/"
               "session/message-v2.ts:46 SYNTHETIC_ATTACHMENT_PROMPT, pushed "
               "as a user message after the assistant message whose tool "
               "results carried media (:384-401) for a provider without "
               "media in tool results (@ai-sdk/openai-compatible). The "
               "parts are `file` parts; images reach the wire as image_url.",
     "verified": True},
    {"harness": "pi",
     "text": "Attached image(s) from tool result:",
     "source": "earendil-works/pi main 2b0a123, packages/ai/src/api/"
               "openai-completions.ts:1455, a user message of that text plus "
               "image_url parts after a run of tool results (the tool "
               "message itself reads \"(see attached image)\" when the tool "
               "returned only images, :1414)",
     "verified": True},
    {"harness": "cline",
     "text": None,
     "tool_marker": "(see following user message for image)",
     "source": "cline/cline main 29896ec, sdk/packages/llms/src/providers/"
               "middleware/split-tool-images.ts IMAGE_PLACEHOLDER: each "
               "media part of a tool result is replaced by that text and a "
               "user message of the media parts ALONE (no text) follows the "
               "tool message",
     "verified": True},
)
# Pi, for a provider with compat.requiresAssistantAfterToolResult: an
# assistant message of exactly this text is put between the tool results and
# the synthetic turn (openai-completions.ts:1236 and :1446).
PI_BRIDGE = "I have processed the tool results."

# A placeholder vision.normalise wrote in place of a media part: after
# normalising, the synthetic turn is still recognised.
_PLACEHOLDER = re.compile(r"^\[(?:image-[0-9a-f]{10}: |an attachment of type )")


def _text_of(m: dict) -> str:
    c = m.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join(str(p.get("text") or "") for p in c
                         if isinstance(p, dict) and p.get("type") in TEXT_PARTS)
    return ""


def _prev(messages: list, i: int) -> int:
    """The index of the message before i, system and developer skipped."""
    j = i - 1
    while j >= 0 and (not isinstance(messages[j], dict)
                      or messages[j].get("role") in ("system", "developer")):
        j -= 1
    return j


def tool_media_turn(messages: list, i: int) -> dict | None:
    """The TOOL_MEDIA_TURNS row messages[i] matches, or None.

    A match is a USER message whose list content is media (image or other
    media parts, or the placeholders normalise wrote for them) plus, at
    most, the row's exact text -- and whose place fits the row:
      - a text row (OpenCode, Pi): the previous message is a tool result or
        an assistant message (OpenCode puts the turn after the WHOLE
        assistant message, which may end on text; Pi may bridge with
        PI_BRIDGE);
      - Cline (no text): the tool results directly before it carry Cline's
        marker.
    """
    if not (0 <= i < len(messages or [])):
        return None
    m = messages[i]
    if not isinstance(m, dict) or m.get("role") != "user":
        return None
    content = m.get("content")
    if not isinstance(content, list) or not content:
        return None
    media = 0
    texts: list[str] = []
    for p in content:
        if not isinstance(p, dict):
            return None
        kind = p.get("type")
        if kind in IMAGE_PARTS or kind in OTHER_MEDIA:
            media += 1
        elif kind in TEXT_PARTS:
            s = str(p.get("text") or "").strip()
            if not s:
                continue
            if _PLACEHOLDER.match(s):
                media += 1
            else:
                texts.append(s)
        else:
            return None
    if not media or len(texts) > 1:
        return None
    j = _prev(messages, i)
    if j < 0:
        return None
    prev = messages[j]
    role = prev.get("role")
    if texts:
        row = next((r for r in TOOL_MEDIA_TURNS if r["text"] == texts[0]), None)
        if row is None:
            return None
        return row if role in ("tool", "function", "assistant") else None
    # No text: Cline. The run of tool messages ending at j carries its
    # marker (a bridge assistant is not Cline's shape).
    row = next(r for r in TOOL_MEDIA_TURNS if r["harness"] == "cline")
    while j >= 0 and isinstance(messages[j], dict) \
            and messages[j].get("role") in ("tool", "function"):
        if row["tool_marker"] in _text_of(messages[j]):
            return row
        j = _prev(messages, j)
    return None


def last_is_tool_media(messages: list) -> dict | None:
    """The row when the request's last non-system message is a synthetic
    tool-media turn, else None."""
    msgs = messages or []
    for i in range(len(msgs) - 1, -1, -1):
        m = msgs[i]
        if not isinstance(m, dict) or m.get("role") in ("system", "developer"):
            continue
        return tool_media_turn(msgs, i)
    return None


def is_tool_media(messages: list, i: int) -> bool:
    return tool_media_turn(messages, i) is not None


# ------------------------------------------------ image bytes printed as text
# The base64 of each format's first bytes: the image guard's table
# (tool_code._MAGIC), repeated here so this module stays light. BMP's "Qk"
# is too short to open a bare run on; a data: URL may still carry a BMP.
MAGICS = (("iVBORw0KGgo", "png"), ("/9j/", "jpeg"), ("R0lGODlh", "gif"),
          ("R0lGODdh", "gif"), ("UklGR", "webp"))
_DATA_URL = re.compile(r"data:image/([A-Za-z0-9.+-]{1,24});base64,", re.I)
_BARE = re.compile(r"(?<![A-Za-z0-9+/=])(?:"
                   + "|".join(re.escape(m) for m, _f in MAGICS) + ")")
# One run: base64 segments separated by line breaks -- real ones, or the
# two-character `\n` of a tool result that is itself JSON text (a wrapped
# `base64` inside {"output": "..."}). Spaces end a run.
_SEG = r"[A-Za-z0-9+/]+={0,2}"
_SEP = r"\r?\n|\\r\\n|\\n"
_RUN = re.compile(_SEG + r"(?:(?:" + _SEP + r")" + _SEG + r")*")
_SEP_RX = re.compile(_SEP)


def _segments(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """The base64 segments of the run text[start:end], split at its line
    breaks (an escaped `\\n`'s `n` is a separator, not a base64 digit)."""
    out, pos = [], start
    for s in _SEP_RX.finditer(text, start, end):
        out.append((pos, s.start()))
        pos = s.end()
    out.append((pos, end))
    return [(a, b) for a, b in out if b > a]
# How many trailing segments may be dropped when the run swallowed a line
# of text after the image (`...AAAA\ndone`): a choice.
_TRIM_SEGMENTS = 3


def sniff(data: bytes) -> str | None:
    """vision.sniff's rule, repeated here (this module stays light)."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
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


def complete(data: bytes, fmt: str) -> bool:
    """Does `data` hold the WHOLE image: its format's end marker, or the
    length its header states? A run a harness cut at its output limit
    fails this, and so does one with a line of text glued to its end."""
    if fmt == "png":
        return data.endswith(b"IEND\xaeB`\x82")
    if fmt == "jpeg":
        return data.endswith(b"\xff\xd9")
    if fmt == "gif":
        return data.endswith(b";")
    if fmt == "webp":
        n = int.from_bytes(data[4:8], "little") + 8
        return len(data) in (n, n + 1)
    if fmt == "bmp":
        return int.from_bytes(data[2:6], "little") == len(data)
    return False


def _decode(b64: str) -> bytes | None:
    try:
        return base64.b64decode(b64 + "=" * (-len(b64) % 4), validate=True)
    except (binascii.Error, ValueError):
        return None


def _head_format(b64: str) -> str | None:
    """The format the first bytes of a run say, even when the run is cut."""
    head = _decode(b64[:32])
    return sniff(head) if head else None


def find_inline(text: str) -> list[dict]:
    """Image bytes printed as text in `text`, in order. Each run is
      {start, end, status, format, data, chars, declared}
    where [start, end) is the span to replace, `status` is "complete" (data
    holds the whole image), "cut" (the image's opening bytes, but not its
    end: a harness's output cap), or "unreadable" (a data:image/ URL whose
    bytes decode but are no format this server reads), and `declared` the
    MIME subtype a data: URL named. A run shorter than INLINE_MIN_CHARS
    base64 characters is never one. Pure: nothing is fetched or opened."""
    if not text or not isinstance(text, str):
        return []
    if "base64," not in text and not any(m in text for m, _f in MAGICS):
        return []
    found: list[dict] = []
    starts: list[tuple[int, int, str | None]] = []    # (span start, run start, declared)
    for m in _DATA_URL.finditer(text):
        starts.append((m.start(), m.end(), m.group(1).lower()))
    for m in _BARE.finditer(text):
        starts.append((m.start(), m.start(), None))
    starts.sort()
    pos = 0
    for span_start, run_start, declared in starts:
        if span_start < pos:
            continue                        # inside a run already taken
        r = _RUN.match(text, run_start)
        if not r:
            continue
        segs = _segments(text, r.start(), r.end())
        joined = "".join(text[a:b] for a, b in segs)
        if len(joined) < INLINE_MIN_CHARS:
            continue
        fmt = _head_format(joined)
        hit = None
        for k in range(len(segs), max(0, len(segs) - _TRIM_SEGMENTS), -1):
            cand = "".join(text[a:b] for a, b in segs[:k])
            data = _decode(cand)
            if data and fmt and sniff(data) == fmt and complete(data, fmt):
                hit = {"start": span_start, "end": segs[k - 1][1],
                       "status": "complete", "format": fmt, "data": data,
                       "chars": len(cand), "declared": declared}
                break
        if hit is None and fmt:
            hit = {"start": span_start, "end": r.end(), "status": "cut",
                   "format": fmt, "data": None, "chars": len(joined),
                   "declared": declared}
        if hit is None and declared is not None:
            data = _decode(joined)
            if data:
                hit = {"start": span_start, "end": r.end(),
                       "status": "unreadable", "format": None, "data": data,
                       "chars": len(joined), "declared": declared}
        if hit is None:
            continue                        # not an image: left as text
        found.append(hit)
        pos = hit["end"]
    return found
