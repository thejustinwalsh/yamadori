#!/usr/bin/env python
"""Every image form into one register (docs/VISION.md Phase 1). No GPU, no
network.

    python mcp/test_image_input.py      -> "N/M checks passed"

GATED HERE
  1. every 5a form lands in the register with its `form`: a user part, an
     image inside a `tool` message (Hermes native), the synthetic tool-media
     turns of OpenCode, Pi (and Pi's bridge) and Cline, a Responses
     `input_image` (in a user message, in a tool message, by file id), image
     bytes PRINTED AS TEXT in a tool result or a user message, an http(s)
     link (recorded, never fetched);
  2. printed runs: complete (bare, data: URL, wrapped with real or JSON-
     escaped line breaks, with a line of text glued on), cut, too short,
     not an image, a format the server cannot read;
  3. determinism: two calls on one history give byte-identical output; a
     text-only request is the same list object; the client's list is not
     mutated; through the SERVED chat template, the next request's prompt
     EXTENDS this one's;
  4. the count cap: the newest images are held, an older one keeps its
     placeholder byte for byte and yama_describe_image says IMAGE_NOT_HELD;
  5. the synthetic-turn table rows are the harnesses' own string literals
     (the source lines are quoted below, read with `gh api` on 2026-09-26);
  6. routing: a tool-media turn is agent_step, as the tool result it
     carries, and a turn with words beyond the prefix is the user speaking;
  7. the security contract: normalising the new forms opens no socket and
     no file; yama_describe_image still refuses URLs and paths;
  8. /v1/models: `modalities` beside `architecture`, text-only when vision
     is off or not configured;
  9. the request-body limit: 413 in OpenAI's envelope, declared or counted.

Temp stores for everything, set BEFORE proxy is imported (test_vision's
header). Images are drawn here; index/media is never read.
"""
from __future__ import annotations

import asyncio
import base64
import builtins
import json
import os
import socket
import struct
import sys
import tempfile
import traceback
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

_TMP = tempfile.mkdtemp(prefix="yamadori_test_image_input_")


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
    "YAMADORI_SLOTS_STATE": os.path.join(_TMP, "slots_state.json"),
    "CONCEPT_SEED_LAST": os.path.join(_TMP, "seed_last.json"),
    "LLAMA_STACK_URL": DEAD, "YAMADORI_MODEL_SERVER": DEAD, "LAYA_URL": DEAD,
    "YAMADORI_GPU_ROOM": "0",
})
for k in ("YAMADORI_IMAGEGEN_URL", "YAMADORI_PUBLIC_BASE", "YAMADORI_VISION",
          "YAMADORI_VISION_MODEL", "YAMADORI_VISION_MAX_BYTES",
          "YAMADORI_VISION_MAX_IMAGES", "YAMADORI_MAX_BODY_BYTES",
          "YAMADORI_SWAP_CONFIG"):
    os.environ.pop(k, None)
# Vision available = the MAIN model's entry carries --mmproj (2026-09-27): a
# config of our own, so the card checks do not follow the live config.yaml.
import tempfile  # noqa: E402
_SWAP_DIR = tempfile.mkdtemp(prefix="yamadori_test_image_input_swap_")
_SWAP_OK = os.path.join(_SWAP_DIR, "swap.yaml")
with open(_SWAP_OK, "w", encoding="utf-8") as _f:
    _f.write('models:\n  "bonsai":\n    cmd: |\n      server\n      --mmproj m.gguf\n')
os.environ["YAMADORI_SWAP_CONFIG"] = _SWAP_OK

import jinja2  # noqa: E402

import body_limit  # noqa: E402
import budget  # noqa: E402
import catalog  # noqa: E402
import image_input  # noqa: E402
import route  # noqa: E402
import tiers  # noqa: E402
import vision  # noqa: E402

tiers._accepted = tiers.FALLBACK_EFFORTS     # no network for the template
_real_budgets = budget.budgets
budget.budgets = lambda pool=None: _real_budgets(pool or 163840)

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


class use:
    """Environment variables for a block; None unsets."""

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


# ------------------------------------------------------------- images, drawn
def png(seed: int, side: int = 24) -> bytes:
    """A real PNG, uncompressed so its size is predictable."""
    raw = b"".join(b"\x00" + bytes([(seed + x) % 256, (seed * 7) % 256, 9])
                   * side for x in range(side))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", side, side, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 0)) + chunk(b"IEND", b""))


def jpeg(seed: int, n: int = 600) -> bytes:
    """Enough of a JPEG for sniff and completeness: SOI ... EOI."""
    return b"\xff\xd8\xff\xe0" + bytes((seed + i) % 251 for i in range(n)) \
        + b"\xff\xd9"


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def data_url(data: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64,{b64(data)}"


def img_part(data: bytes, mime: str = "image/png") -> dict:
    return {"type": "image_url", "image_url": {"url": data_url(data, mime)}}


def call(cid: str, name: str = "read", args: dict | None = None) -> dict:
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args or {})}}


def forms(att: dict) -> list:
    return [(e.get("form"), e.get("error")) for e in att["images"].values()]


def text_of(msgs: list) -> str:
    return json.dumps(msgs)


TASK = {"role": "user", "content": "Check the page renders; take a screenshot."}
ASK = {"role": "assistant", "content": "", "tool_calls": [call("c1")]}


# -------------------------------------------------------------- the harnesses
def opencode(data: bytes) -> list:
    """message-v2.ts: the tool text stripped to "Image read successfully",
    the synthetic user turn after it."""
    return [TASK, ASK,
            {"role": "tool", "tool_call_id": "c1",
             "content": "Image read successfully"},
            {"role": "user", "content": [
                {"type": "text", "text": "Attached media from tool result:"},
                img_part(data)]}]


def pi(data: bytes, bridge: bool = False) -> list:
    """openai-completions.ts: "(see attached image)", then the turn; with
    requiresAssistantAfterToolResult, the bridge between them."""
    out = [TASK, ASK, {"role": "tool", "tool_call_id": "c1",
                       "content": "(see attached image)"}]
    if bridge:
        out.append({"role": "assistant", "content": image_input.PI_BRIDGE})
    out.append({"role": "user", "content": [
        {"type": "text", "text": "Attached image(s) from tool result:"},
        img_part(data)]})
    return out


def cline(data: bytes) -> list:
    """split-tool-images.ts: the tool result's media replaced by the marker
    (the openai-compatible converter stringifies the parts), then a user
    message of the media ALONE."""
    return [TASK, ASK,
            {"role": "tool", "tool_call_id": "c1", "content": json.dumps(
                [{"type": "text", "text": "read shot.png"},
                 {"type": "text",
                  "text": "(see following user message for image)"}])},
            {"role": "user", "content": [img_part(data)]}]


def hermes_native(data: bytes) -> list:
    """vision_tools._build_native_vision_tool_result: [text, image_url] in
    the tool message itself."""
    return [TASK, {"role": "assistant", "content": "", "tool_calls": [
        call("c1", "vision_analyze", {"image_url": "/tmp/shot.png",
                                      "question": "overlap?"})]},
            {"role": "tool", "tool_call_id": "c1", "content": [
                {"type": "text", "text": "Question: overlap?"},
                img_part(data)]}]


# ============================================================== the tests ==
def test_every_form_lands_in_the_register():
    p = png(1)
    for name, msgs, want in (
            ("OpenCode's synthetic turn", opencode(p), "tool_media_turn"),
            ("Pi's synthetic turn", pi(p), "tool_media_turn"),
            ("Pi's synthetic turn after its bridge", pi(p, True),
             "tool_media_turn"),
            ("Cline's image-only turn", cline(p), "tool_media_turn"),
            ("Hermes native: an image inside the tool message",
             hermes_native(p), "tool_part")):
        out, att = vision.normalise(msgs)
        check(forms(att) == [(want, None)],
              f"{name}: registered, form {want}", json.dumps(forms(att)))
        check('"image_url"' not in text_of(out) and b64(p) not in text_of(out),
              f"{name}: no image part and no image bytes reach the text model")
        check("an image from a tool result" in text_of(out)
              and "yama_describe_image" in text_of(out),
              f"{name}: the placeholder says where it came from and the next "
              f"step")

    plain = [{"role": "user", "content": [{"type": "text", "text": "look"},
                                          img_part(p)]}]
    out, att = vision.normalise(plain)
    check(forms(att) == [("user_part", None)]
          and "an attached image (PNG," in out[0]["content"][1]["text"],
          "a user's attached image: form user_part, the placeholder worded "
          "exactly as before (an existing conversation's prefix holds)",
          out[0]["content"][1]["text"])

    # Responses parts, as the adapter passes them on (docs/VISION.md 5a).
    resp_user = [{"role": "user", "content": [
        {"type": "input_text", "text": "<image name=[Image #1] path=/w/a.png>"},
        {"type": "input_image", "image_url": data_url(p), "detail": "high"},
        {"type": "input_text", "text": "</image>"}]}]
    out, att = vision.normalise(resp_user)
    check(forms(att) == [("user_part", None)]
          and [x["type"] for x in out[0]["content"]] == ["text"] * 3
          and out[0]["content"][0]["text"].startswith("<image name=[Image #1]"),
          "Responses input_image in a message is an attachment; input_text "
          "becomes a text part, Codex's frame kept as text",
          json.dumps(out)[:300])
    resp_tool = [TASK, {"role": "assistant", "content": "",
                        "tool_calls": [call("c1", "view_image")]},
                 {"role": "tool", "tool_call_id": "c1", "content": [
                     {"type": "input_text", "text": "the screenshot"},
                     {"type": "input_image", "image_url": data_url(p),
                      "detail": "auto"}]}]
    out, att = vision.normalise(resp_tool)
    check(forms(att) == [("responses_output", None)]
          and all(x["type"] == "text" for x in out[2]["content"]),
          "a function_call_output array in a tool message: form "
          "responses_output, the tool message all text for upstream",
          json.dumps(out[2])[:300])
    fid = [{"role": "user", "content": [
        {"type": "input_image", "file_id": "file-abc123"}]}]
    out, att = vision.normalise(fid)
    check(forms(att) == [("file_id", "IMAGE_FILE_ID")]
          and "no files API" in out[0]["content"][0]["text"],
          "a Responses file id: a placeholder saying there is no files API",
          out[0]["content"][0]["text"])
    try:
        vision.resolve(next(iter(att["images"])), att)
        check(False, "yama_describe_image on a file id is refused")
    except vision.Err as e:
        check(e.code == "IMAGE_FILE_ID" and e.remedies,
              "yama_describe_image on a file id: IMAGE_FILE_ID with a remedy")

    url_tool = [TASK, ASK, {"role": "tool", "tool_call_id": "c1", "content": [
        {"type": "image_url", "image_url": {"url": "https://x.test/a.png"}}]}]
    out, att = vision.normalise(url_tool)
    check(forms(att) == [("url", "IMAGE_URL_NOT_ALLOWED")]
          and "NOT downloaded" in text_of(out),
          "an http(s) link, even from a tool: recorded, never fetched")

    s = vision.summary(vision.normalise(opencode(p))[1])
    check(s and s[0].get("form") == "tool_media_turn" and "b64" not in s[0]
          and "sha" not in s[0],
          "x_yamadori.attachments carries each image's form, never its "
          "bytes", json.dumps(s))


def test_printed_bytes():
    p, j = png(2), jpeg(3)
    # A shell's output, as a JSON tool result (Hermes' terminal).
    msgs = [TASK, {"role": "assistant", "content": "", "tool_calls": [
        call("c1", "terminal", {"command": "base64 -w0 shot.png"})]},
        {"role": "tool", "tool_call_id": "c1",
         "content": json.dumps({"output": b64(p), "exit_code": 0})}]
    out, att = vision.normalise(msgs)
    body = out[2]["content"]
    check(forms(att) == [("inline_text", None)]
          and b64(p) not in body and '"exit_code": 0' in body
          and "an image printed as text in a tool result (PNG" in body,
          "base64 printed in a tool result becomes a placeholder in place; "
          "the rest of the text is kept", body[:300])
    e = next(iter(att["images"].values()))
    check(base64.b64decode(e["b64"]) == p,
          "the register holds exactly the printed image")

    # `base64` without -w0 wraps at 76; inside JSON the breaks are `\n`.
    wrapped = "\n".join(b64(j)[i:i + 76] for i in range(0, len(b64(j)), 76))
    esc = json.dumps({"output": wrapped + "\ndone\n"})
    check("\\n" in esc, "(the fixture carries JSON-escaped line breaks)")
    out, att = vision.normalise([TASK, {"role": "tool", "tool_call_id": "x",
                                        "content": esc}])
    body = out[1]["content"]
    check(forms(att) == [("inline_text", None)]
          and base64.b64decode(next(iter(att["images"].values()))["b64"]) == j
          and "done" in body,
          "a wrapped run (JSON-escaped line breaks) is one image; the line "
          "of text after it is kept, not glued on", body[:300])
    out, att = vision.normalise([TASK, {"role": "tool", "tool_call_id": "x",
                                        "content": wrapped + "\nok"}])
    check(forms(att) == [("inline_text", None)]
          and out[1]["content"].endswith("\nok"),
          "real line breaks: the same", out[1]["content"][-120:])

    # #44: a data URL read back from a file, cut at the harness's limit.
    cut = data_url(png(4, side=60))[:3000]
    out, att = vision.normalise([TASK, {"role": "tool", "tool_call_id": "x",
                                        "content": "1|" + cut + "\n... [truncated]"}])
    body = out[1]["content"]
    check(forms(att) == [("inline_text", "IMAGE_INCOMPLETE")]
          and "cut off after" in body and "smaller copy" in body
          and "[truncated]" in body and cut[40:200] not in body,
          "a CUT run: replaced by a placeholder that says so and gives the "
          "remedy", body[:400])
    try:
        vision.resolve(next(iter(att["images"])), att)
        check(False, "yama_describe_image on a cut image is refused")
    except vision.Err as e:
        check(e.code == "IMAGE_INCOMPLETE" and e.retryable,
              "yama_describe_image on it: IMAGE_INCOMPLETE, retryable, remedy")

    short = "id iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB and sha /9j/4AAQSkZJRg=="
    same = [TASK, {"role": "tool", "tool_call_id": "x", "content": short}]
    out, att = vision.normalise(same)
    check(out is same and not att["images"],
          f"a run under {image_input.INLINE_MIN_CHARS} characters is never "
          f"touched (ids, hashes, tokens)")
    noise = base64.b64encode(bytes(range(256)) * 4).decode()
    same = [TASK, {"role": "tool", "tool_call_id": "x", "content": noise}]
    out, att = vision.normalise(same)
    check(out is same and not att["images"],
          "a long base64 run that is not an image is left as text")
    svg = data_url(b"<svg xmlns='http://www.w3.org/2000/svg'>" + b" " * 400
                   + b"</svg>", "image/svg+xml")
    out, att = vision.normalise([{"role": "user", "content": "see " + svg}])
    check(forms(att) == [("inline_text", "NOT_AN_IMAGE")]
          and "cannot look at" in out[0]["content"]
          and "PHN2Zy" not in out[0]["content"],
          "a data:image/ URL of a format the server cannot read: replaced "
          "(no flood), and says so", out[0]["content"][:300])
    out, att = vision.normalise([{"role": "user", "content": [
        {"type": "text", "text": "what is this? " + data_url(p)}]}])
    check(forms(att) == [("inline_text", None)]
          and "an image pasted as text in the message" in text_of(out),
          "a data URL pasted in a user's text part is an image too")
    asst = [TASK, {"role": "assistant", "content": "here: " + data_url(p)}]
    out, att = vision.normalise(asst)
    check(out is asst and not att["images"],
          "the model's own turns are never rewritten (the slot generated "
          "them)")


def test_determinism_and_prefix():
    p = png(5)
    msgs = pi(p)
    frozen = json.dumps(msgs)
    a, _ = vision.normalise(msgs)
    b, _ = vision.normalise(msgs)
    check(json.dumps(a) == json.dumps(b),
          "two calls on the same history: byte-identical output")
    check(json.dumps(msgs) == frozen, "the client's messages are not mutated")
    text = [{"role": "user", "content": "hello"},
            {"role": "tool", "tool_call_id": "x", "content": "plain output"}]
    same, att = vision.normalise(text)
    check(same is text and not att["images"],
          "a text-only request is the very same list object")

    # Through the SERVED chat template: the request after this one (the
    # model answered, the user asked again) renders a prompt that EXTENDS
    # this one's -- the image turns render the same bytes both times.
    env = jinja2.Environment()
    env.filters["tojson"] = lambda v, **k: json.dumps(v, ensure_ascii=False)
    tpl = env.from_string(open(os.path.join(
        HERE, "fixtures", "bonsai_chat_template.jinja"), encoding="utf-8").read())

    def render(ms, gen):
        ms = [dict(m, tool_calls=[dict(c, function=dict(
            c["function"], arguments=json.loads(c["function"]["arguments"])))
            for c in m["tool_calls"]]) if m.get("tool_calls") else dict(m)
            for m in ms]
        for m in ms:
            if m.get("content") is None:
                m["content"] = ""
        def boom(msg):
            raise ValueError(msg)
        return tpl.render(messages=ms, add_generation_prompt=gen,
                          raise_exception=boom, reasoning_effort="medium")

    for name, hist in (("Pi", pi(p)), ("OpenCode", opencode(p)),
                       ("Cline", cline(p)), ("Hermes native", hermes_native(p)),
                       ("printed bytes", [TASK, ASK, {
                           "role": "tool", "tool_call_id": "c1",
                           "content": json.dumps({"output": b64(p)})}])):
        one, _ = vision.normalise(hist)
        nxt = hist + [{"role": "assistant", "content": "The header overlaps."},
                      {"role": "user", "content": "fix it"}]
        two, _ = vision.normalise(nxt)
        r1, r2 = render(one, False), render(two, True)
        check(r2.startswith(r1) and b64(p)[:64] not in r2,
              f"{name}: through the served template, the next request's "
              f"prompt extends this one's", r1[-200:])


def test_count_cap():
    few = [{"role": "user", "content": [img_part(png(100 + i))
                                        for i in range(5)]}]
    many = [{"role": "user", "content": [img_part(png(100 + i))
                                         for i in range(25)]}]
    o5, a5 = vision.normalise(few)
    o25, a25 = vision.normalise(many)
    held = [i for i, e in a25["images"].items() if e.get("b64")]
    gone = [i for i, e in a25["images"].items()
            if e.get("error") == "IMAGE_NOT_HELD"]
    check(len(held) == image_input.MAX_IMAGES_PER_REQUEST == 20
          and len(gone) == 5 and a25.get("not_held") == 5,
          "25 images: the 20 newest held, 5 not", json.dumps(forms(a25))[:200])
    check(held == list(a25["images"])[5:],
          "the held ones are the NEWEST (the last 20 in the conversation)")
    check([x["text"] for x in o25[0]["content"][:5]]
          == [x["text"] for x in o5[0]["content"]],
          "an image over the cap keeps its placeholder byte for byte: a "
          "growing history never rewrites a past message")
    try:
        vision.resolve(gone[0], a25)
        check(False, "yama_describe_image on an image not held is refused")
    except vision.Err as e:
        check(e.code == "IMAGE_NOT_HELD" and e.remedies,
              "yama_describe_image on it: IMAGE_NOT_HELD, with the remedy")
    with use(YAMADORI_VISION_MAX_IMAGES="3"):
        _o, a3 = vision.normalise(few)
    check(sum(1 for e in a3["images"].values() if e.get("b64")) == 3,
          "YAMADORI_VISION_MAX_IMAGES sets it")
    dup = [{"role": "user", "content": [img_part(png(7))]},
           {"role": "assistant", "content": "ok"},
           {"role": "user", "content": [img_part(png(8)), img_part(png(7))]}]
    _o, ad = vision.normalise(dup)
    check(len(ad["images"]) == 2
          and list(ad["images"])[-1] == vision._new_id(png(7)),
          "an image seen again is one entry, moved to the newest place")


# The harnesses' own lines, as `gh api .../contents/...` returned them on
# 2026-09-26 (the table's `source` names each file and commit).
SOURCE_LINES = {
    "opencode": 'export const SYNTHETIC_ATTACHMENT_PROMPT = '
                '"Attached media from tool result:"',
    "pi": 'text: "Attached image(s) from tool result:",',
    "pi_bridge": 'content: "I have processed the tool results.",',
    "cline": 'const IMAGE_PLACEHOLDER = "(see following user message for '
             'image)";',
}


def test_the_table_is_the_sources_words():
    rows = {r["harness"]: r for r in image_input.TOOL_MEDIA_TURNS}
    check(f'"{rows["opencode"]["text"]}"' in SOURCE_LINES["opencode"],
          "OpenCode's row is its SYNTHETIC_ATTACHMENT_PROMPT, exactly")
    check(f'"{rows["pi"]["text"]}"' in SOURCE_LINES["pi"],
          "Pi's row is its literal, exactly")
    check(f'"{image_input.PI_BRIDGE}"' in SOURCE_LINES["pi_bridge"],
          "Pi's bridge is its literal, exactly")
    check(f'"{rows["cline"]["tool_marker"]}"' in SOURCE_LINES["cline"],
          "Cline's marker is its IMAGE_PLACEHOLDER, exactly")
    check(all(r.get("verified") and r.get("source") for r in rows.values()),
          "every row names its source and is marked verified")


def test_recognition_edges():
    p = png(9)
    ok = pi(p)
    check(image_input.last_is_tool_media(ok)["harness"] == "pi",
          "Pi's turn is recognised on the client's messages")
    out, _ = vision.normalise(ok)
    check((image_input.last_is_tool_media(out) or {}).get("harness") == "pi",
          "and on the normalised ones (placeholders in place of images)")
    typed = [TASK, {"role": "assistant", "content": "Send me the screenshot."},
             {"role": "user", "content": [
                 {"type": "text", "text": "here it is, does it overlap?"},
                 img_part(p)]}]
    check(image_input.last_is_tool_media(typed) is None,
          "a person's own words with an image are a user turn")
    first = [{"role": "user", "content": [
        {"type": "text", "text": "Attached image(s) from tool result:"},
        img_part(p)]}]
    check(image_input.last_is_tool_media(first) is None,
          "the prefix as a conversation's first message is not one")
    no_img = pi(p)[:-1] + [{"role": "user", "content": [
        {"type": "text", "text": "Attached image(s) from tool result:"}]}]
    check(image_input.last_is_tool_media(no_img) is None,
          "the prefix with no media is not one")
    bare = [TASK, ASK, {"role": "tool", "tool_call_id": "c1", "content": "ok"},
            {"role": "user", "content": [img_part(p)]}]
    check(image_input.last_is_tool_media(bare) is None,
          "an image-only turn after a tool result WITHOUT Cline's marker is "
          "not one (a person may attach an image mid-loop)")


def test_routing():
    p = png(10)
    tools = ["read", "bash", "edit"]
    for name, msgs in (("OpenCode", opencode(p)), ("Pi", pi(p)),
                       ("Pi, bridged", pi(p, True)), ("Cline", cline(p))):
        out, _ = vision.normalise(msgs)
        r = route.classify(out, client_tools=tools)
        check(route.ends_on(out) == "tool" and r["class"] == "agent_step"
              and r["signals"].get("tool_media_turn"),
              f"{name}: the synthetic turn routes agent_step, as the tool "
              f"result it carries", json.dumps(r)[:300])
        r0 = route.classify(out, client_tools=[])
        check(r0["class"] == "agent_step",
              f"{name}: agent_step without a tool list too")
    fake = [TASK, ASK, {"role": "tool", "tool_call_id": "c1", "content": "ok"},
            {"role": "user", "content": [
                {"type": "text", "text": "Attached media from tool result:"},
                {"type": "text", "text": "it is still broken"},
                img_part(p)]}]
    check(image_input.last_is_tool_media(fake) is None,
          "a turn with words beyond the prefix is the user speaking")

    import proxy
    with use(YAMADORI_IMAGEGEN_URL=None):
        out = proxy.prepare({"model": "yamadori", "messages": pi(p),
                             "tools": [{"type": "function", "function": {
                                 "name": "read", "parameters": {
                                     "type": "object", "properties": {}}}}]})
    rc = (out.get("_route") or {}).get("class")
    sent = json.dumps(out.get("messages"))
    check(rc == "agent_step" and '"image_url"' not in sent
          and b64(p) not in sent,
          "through prepare: agent_step, and no image reaches upstream",
          json.dumps({"route": out.get("_route")})[:300])
    summ = vision.summary(out.get("_attached"))
    check([a.get("form") for a in summ] == ["tool_media_turn"],
          "through prepare: the register (x_yamadori.attachments) names the "
          "form", json.dumps(summ))


def test_security_contract():
    """Normalising the new forms opens no socket and no file."""
    p = png(11)
    opened: list = []
    connects: list = []
    real_open = builtins.open
    real_conn = socket.socket.connect

    def spy_open(f, *a, **k):
        opened.append(str(f))
        return real_open(f, *a, **k)

    def spy_conn(self, addr):
        connects.append(addr)
        raise OSError("no network in this test")
    hist = (opencode(p) + [{"role": "assistant", "content": "ok"}]
            + [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "http://127.0.0.1:9/x.png"}},
                {"type": "text", "text": "/etc/passwd " + data_url(p)}]}]
            + [{"role": "tool", "tool_call_id": "q", "content": [
                {"type": "image_url", "image_url": {"url": "file:///etc/passwd"}},
                {"type": "input_image", "file_id": "file-x"}]}])
    builtins.open = spy_open
    socket.socket.connect = spy_conn
    try:
        _out, att = vision.normalise(hist)
    finally:
        builtins.open = real_open
        socket.socket.connect = real_conn
    check(not opened and not connects,
          "the new forms: no file opened, no socket connected",
          json.dumps({"opened": opened, "connects": [str(c) for c in connects]}))
    for ref, code in (("http://127.0.0.1:9/x.png", "IMAGE_URL_NOT_ALLOWED"),
                      ("/etc/passwd", "UNKNOWN_IMAGE"),
                      ("file:///etc/passwd", "IMAGE_URL_NOT_ALLOWED")):
        try:
            vision.resolve(ref, att)
            check(False, f"yama_describe_image refuses {ref}")
        except vision.Err as e:
            check(e.code == code, f"yama_describe_image still refuses {ref}: {code}",
                  e.code)


def test_model_card():
    card = catalog.public_list()["data"][0]
    one = catalog.model_card("anything")
    for name, row in (("/v1/models", card), ("/v1/models/{id}", one)):
        check(row.get("modalities") == {"input": ["text", "image"],
                                        "output": ["text"]}
              and row["architecture"]["input_modalities"] == ["text", "image"]
              and row["architecture"]["output_modalities"] == ["text"],
              f"{name}: models.dev `modalities` beside OpenRouter's "
              f"`architecture`, image in while vision is available",
              json.dumps({k: row.get(k) for k in ("modalities", "architecture")}))
    with use(YAMADORI_VISION="0"):
        off = catalog.model_card("yamadori")
    check(off.get("modalities") == {"input": ["text"], "output": ["text"]}
          and off["architecture"]["input_modalities"] == ["text"],
          "YAMADORI_VISION=0: text only, both conventions")
    cfg = os.path.join(_TMP, "swap.yaml")
    with open(cfg, "w", encoding="utf-8") as f:
        f.write('models:\n  "bonsai":\n    cmd: "server -m a.gguf"\n')
    with use(YAMADORI_SWAP_CONFIG=cfg):
        noproj = catalog.model_card("yamadori")
        check(vision.enabled() and not catalog.vision_configured(),
              "(the main entry without --mmproj is not configured)")
    check(noproj.get("modalities", {}).get("input") == ["text"],
          "the main model without its projector: text only")
    cfg2 = os.path.join(_TMP, "swap2.yaml")
    with open(cfg2, "w", encoding="utf-8") as f:
        f.write('models:\n  "bonsai":\n    cmd: |\n      server\n'
                '      --mmproj m.gguf\n')
    with use(YAMADORI_SWAP_CONFIG=cfg2):
        withp = catalog.model_card("yamadori")
    check(withp.get("modalities", {}).get("input") == ["text", "image"],
          "the main model with --mmproj: image in")
    with use(YAMADORI_SWAP_CONFIG=os.path.join(_TMP, "missing.yaml")):
        gone = catalog.model_card("yamadori")
    check(gone.get("modalities", {}).get("input") == ["text"],
          "no readable config: text only (the card never promises what "
          "cannot be checked)")


def test_body_limit():
    sent: list = []
    reached: list = []

    async def app(scope, receive, send):
        body = b""
        while True:
            m = await receive()
            body += m.get("body", b"")
            if not m.get("more_body"):
                break
        reached.append(len(body))
        await send({"type": "http.response.start", "status": 200,
                    "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    def run(headers, chunks, limit=1000, method="POST"):
        sent.clear()
        reached.clear()
        q = [{"type": "http.request", "body": c, "more_body": i < len(chunks) - 1}
             for i, c in enumerate(chunks)]

        async def receive():
            return q.pop(0) if q else {"type": "http.disconnect"}

        async def send(m):
            sent.append(m)
        mw = body_limit.BodyLimit(app, limit=limit)
        asyncio.run(mw({"type": "http", "method": method,
                        "headers": headers}, receive, send))
        status = next((m["status"] for m in sent
                       if m["type"] == "http.response.start"), None)
        body = b"".join(m.get("body", b"") for m in sent
                        if m["type"] == "http.response.body")
        return status, body

    st, body = run([(b"content-length", b"5000")], [b"x" * 5000])
    err = json.loads(body).get("error", {})
    check(st == 413 and not reached and err.get("code") == "request_too_large"
          and {"message", "type", "param", "code"} <= set(err),
          "a declared body over the limit: 413 in OpenAI's envelope, the "
          "route never runs", json.dumps(err)[:200])
    st, body = run([], [b"x" * 600, b"x" * 600])
    check(st == 413 and not reached,
          "a chunked body is counted and refused when it passes the limit")
    st, _b = run([(b"content-length", b"900")], [b"x" * 900])
    check(st == 200 and reached == [900], "a body under the limit passes")
    st, _b = run([], [b""], method="GET")
    check(st == 200, "a GET is not checked")
    check(image_input.max_body_bytes() == 64 * 1024 * 1024,
          "the default is 64 MB")
    with use(YAMADORI_MAX_BODY_BYTES="2048"):
        check(image_input.max_body_bytes() == 2048,
              "YAMADORI_MAX_BODY_BYTES sets it")
    import server
    check(any(getattr(m, "cls", None) is body_limit.BodyLimit
              for m in server.app.user_middleware),
          "server.py installs it on the app")


def main() -> int:
    tests = (test_every_form_lands_in_the_register,
             test_printed_bytes,
             test_determinism_and_prefix,
             test_count_cap,
             test_the_table_is_the_sources_words,
             test_recognition_edges,
             test_routing,
             test_security_contract,
             test_model_card,
             test_body_limit)
    for fn in tests:
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
    passed = sum(1 for ok, _, _ in _results if ok)
    total = len(_results)
    print(f"\n{'=' * 70}\n  {passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
