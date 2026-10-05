#!/usr/bin/env python
"""The streamed tool loop (`proxy.stream_body`) against a fake upstream. No GPU.

WHAT THIS IS GATING -- docs/CONSTRAINTS.md items 10, 11, 12

  1. ONE upstream generation per streamed answer. The loop used to discard
     the hop that had no tool call -- the answer -- and generate it again to
     stream it: two generations per answer, and the second could differ.
  2. The breaker LANDS: on the last hop the tools are withdrawn and the
     answer is asked for, as in `complete()`. It used to send our tools on
     the final call, so a call to one of them could reach a client that does
     not have it.
  3. The corpus gets the REAL hop count. It got MAX_TOOL_HOPS (12) for every
     streamed answer, including the ones that made no tool call.
  4. A `length` finish is finish_reason "length" with the content that was
     generated (the notice went 2026-09-27), and the reasoning never
     appears as content.
  5. A tool result over the 6000-char breaker carries a marker naming the
     size withheld and the call that fetches the rest.

  6. The pieces around the loop that stayed when the second brain went
     (2026-09-29, docs/REMOVED.md): the landing and the tool-turn cap,
     x_yamadori on both paths, the compaction store, a (directive) prefill
     across an image hop, a client that goes away, and the server's lanes.
     Deep thinking, fan-out, the fix-up and its Verified/Repaired/Checked
     notes, and the delegate arm are gone, and so are their checks.

THE FAKE UPSTREAM

A real `http.server` on 127.0.0.1:<ephemeral>, so the proxy's own reader
(`_post_events`) and its urllib transport run unmodified. Each POST pops the
next scripted reply and records the request body; a request with
`stream: true` gets SSE, anything else gets the assembled JSON. Counting the
recorded requests is what "one generation" means here.

Everything that would touch shared state is stubbed or redirected to temp
files BEFORE import: the corpus (training data for the decision model --
test runs have polluted it before), nebari, rings, and the package store.
The concept seed is pinned (proxy._draw_seed) so no test loads the embedding
matrix.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

_TMP = tempfile.mkdtemp(prefix="yamadori_test_stream_")
os.environ["YAMADORI_CORPUS_DB"] = os.path.join(_TMP, "corpus.sqlite3")
os.environ["YAMADORI_NEBARI_DB"] = os.path.join(_TMP, "nebari.sqlite3")
os.environ["RINGS_DB"] = os.path.join(_TMP, "rings.sqlite3")
os.environ["YAMADORI_PKG_DIR"] = os.path.join(_TMP, "pkgs")
os.environ["CODE_INDEX_DB"] = os.path.join(_TMP, "code.sqlite3")
os.environ["CONCEPT_SEED_LAST"] = os.path.join(_TMP, "seed_last.json")

import proxy  # noqa: E402
import repeats  # noqa: E402
import served_fixture  # noqa: E402
# The served model's /props, pinned (mcp/served_fixture.py): budget and
# tiers would otherwise ask the live stack (llama-swap reloads `bonsai`).
served_fixture.pin()

# A conversation's first user turn draws a concept seed (proxy._draw_seed,
# at tiers whose `seed` flag is on). Pinned here, so no test loads the 420 MB
# embedding matrix and every seed line is predictable.
SEED = {"word": "cedar", "token_id": 1, "u32": 2, "hex": "0x00000002"}
proxy._draw_seed = lambda prompt=None: dict(SEED)

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


# --------------------------------------------------------------------------
# The fake upstream
# --------------------------------------------------------------------------

_script: list[dict] = []          # replies, popped in order
_seen: list[dict] = []            # request bodies, in order
# Does the fake re-send a prefill's reasoning and content (llama-server does,
# STEP 0)? A test turns it off to prove the proxy does not depend on it.
ECHO_PREFILL = True


def reply(content: str = "", reasoning: str = "", calls: list | None = None,
          finish: str | None = None, usage: dict | None = None) -> dict:
    return {"content": content, "reasoning": reasoning, "calls": calls or [],
            "finish": finish or ("tool_calls" if calls else "stop"),
            "usage": usage or {"prompt_tokens": 100, "completion_tokens": 10,
                               "total_tokens": 110}}


def call(name: str, args: dict, cid: str = "call_1") -> dict:
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def _sse(r: dict) -> bytes:
    def ev(delta=None, finish=None, **extra):
        d = {"id": "up-1", "object": "chat.completion.chunk", "model": "bonsai",
             "created": 1, "choices": [{"index": 0, "delta": delta or {},
                                        "finish_reason": finish}]}
        d.update(extra)
        return b"data: " + json.dumps(d).encode() + b"\n\n"
    out = [ev({"role": "assistant"})]
    # Split in pieces so the reader really accumulates across deltas.
    for piece in _pieces(r["reasoning"]):
        out.append(ev({"reasoning_content": piece}))
    for piece in _pieces(r["content"]):
        out.append(ev({"content": piece}))
    for i, c in enumerate(r["calls"]):
        out.append(ev({"tool_calls": [dict(c, index=i)]}))
    if r.get("drop"):
        # the model server died: llama-swap answers 200 and closes the body -- no finish chunk, no usage, no [DONE]
        return b"".join(out)
    out.append(ev({}, r["finish"]))
    d = {"id": "up-1", "object": "chat.completion.chunk", "choices": [],
         "usage": r["usage"]}
    out.append(b"data: " + json.dumps(d).encode() + b"\n\n")
    out.append(b"data: [DONE]\n\n")
    return b"".join(out)


def _json(r: dict) -> bytes:
    msg = {"role": "assistant", "content": r["content"]}
    if r["reasoning"]:
        msg["reasoning_content"] = r["reasoning"]
    if r["calls"]:
        msg["tool_calls"] = r["calls"]
    return json.dumps({"id": "up-1", "object": "chat.completion",
                       "choices": [{"index": 0, "message": msg,
                                    "finish_reason": r["finish"]}],
                       "usage": r["usage"]}).encode()


def _pieces(s: str, n: int = 7) -> list[str]:
    return [s[i:i + n] for i in range(0, len(s), n)] if s else []


_warms: list[tuple[str, dict]] = []   # /upstream/... requests (proxy._warm)
_tokenized: list[dict] = []           # /upstream/.../tokenize bodies


class _Up(BaseHTTPRequestHandler):
    def do_POST(self):                                           # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path.startswith("/upstream/") and \
                self.path.endswith("/tokenize"):
            # The reasoning count (proxy._usage_of, U2): one token per
            # word. Kept out of _warms, which is the warm's record.
            _tokenized.append(body)
            data = json.dumps({"tokens": [1] * len(
                (body.get("content") or "").split())}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if self.path.startswith("/upstream/"):
            # The warm: /apply-template renders (a stand-in: the messages as
            # JSON, which shares a prefix exactly when the messages do, and
            # an end-of-turn token after each), then /completion.
            _warms.append((self.path, body))
            if self.path.endswith("/apply-template"):
                out = {"prompt": "".join(json.dumps(m, sort_keys=True)
                                         + "<|im_end|>\n"
                                         for m in body.get("messages") or [])}
            else:
                out = {"timings": {"cache_n": 10, "prompt_n": 2}}
            data = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        _seen.append(body)
        if not _script:
            self.send_response(500)
            self.end_headers()
            return
        r = _script.pop(0)
        last = (body.get("messages") or [{}])[-1]
        if last.get("role") == "assistant" and ECHO_PREFILL:
            # A PREFILL: llama-server sends the prefilled reasoning (plus a
            # newline) and content back as the first deltas, then continues
            # (STEP 0, docs/SELF-IMPROVEMENT-PLAN.md).
            r = dict(r, content=(last.get("content") or "") + r["content"],
                     reasoning=((last.get("reasoning_content") or "") + "\n"
                                + r["reasoning"])
                     if last.get("reasoning_content") else r["reasoning"])
        stream = bool(body.get("stream"))
        data = _sse(r) if stream else _json(r)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream" if stream
                         else "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):                                   # noqa: D102
        pass


_srv = ThreadingHTTPServer(("127.0.0.1", 0), _Up)
threading.Thread(target=_srv.serve_forever, daemon=True).start()
proxy.UPSTREAM = f"http://127.0.0.1:{_srv.server_address[1]}"
# The internal door (mcp/model.py) goes to the same fake: no offline test
# may reach the live model server.
import model as _model  # noqa: E402
_model.UPSTREAM = proxy.UPSTREAM

# --------------------------------------------------------------------------
# Stubs: everything around the loop that is not the loop
# --------------------------------------------------------------------------

_log: list[tuple] = []
_ran: list[tuple] = []
TOOL_OUTPUT: dict[str, str] = {}


class _Corpus:
    @staticmethod
    def new_turn():
        return "turn-1"

    @staticmethod
    def log_turn(*a, **k):
        _log.append(("turn",) + a)

    @staticmethod
    def log_tool_call(turn, root, fn, args, hop):
        _log.append(("call", fn, hop))

    @staticmethod
    def log_tool_result(turn, root, fn, out, ms):
        _log.append(("result", fn, len(out)))

    @staticmethod
    def log_answer(turn, root, content, hops, ms, **k):
        _log.append(("answer", hops, content))


# What the selection engine decided, as the stubbed prepare() reports it
# (selection.decide's shape since 2026-09-29: skills, because, signals).
SELECTION: dict = {}
# Other prepare() fields a test sets: _prefill, _slot, _ledger, _tier.
PREP: dict = {}

# The tool main runs itself in these tests: yama_generate_image, one of the
# server tools left on main (proxy.OUR_NAMES).
IMG = "yama_generate_image"


def img(prompt: str = "a lathe", cid: str = "call_1") -> dict:
    return call(IMG, {"prompt": prompt}, cid)


def _prepare(body: dict) -> dict:
    tools, ours = proxy.main_tools(body.get("tools"))
    if IMG not in {t["function"]["name"] for t in tools}:
        tools = tools + [proxy.images.TOOL]
        ours = set(ours) | {IMG}
    return dict({"model": "bonsai", "messages": list(body["messages"]),
                 "tools": tools, "max_tokens": 10240,
                 "_tier": {"name": "test"}, "_ours": sorted(ours),
                 "_selection": dict({"skills": False, "because": {},
                                     "signals": {}}, **SELECTION)},
                **json.loads(json.dumps(PREP)))


def _run_our_tool(name, args, state=None):
    _ran.append((name, args))
    return TOOL_OUTPUT.get(name, f"{name}: 1 result\nsrc/a.ts:1")


_REAL_RUN_OUR_TOOL = proxy.run_our_tool
proxy.corpus = _Corpus
proxy.prepare = _prepare
proxy.run_our_tool = _run_our_tool
proxy.resolve_repo = lambda messages, ip="": (None, False, "none")
proxy.session_context = lambda messages, account="", session="", **k: (
    "k", {"_key": "k"})


def reset(script: list[dict]) -> None:
    _script[:] = script
    _seen.clear()
    _log.clear()
    _ran.clear()
    _warms.clear()
    TOOL_OUTPUT.clear()
    SELECTION.clear()
    PREP.clear()


def run_stream(user: str = "which file defines LatheGeometry",
               tools: list | None = None,
               include_usage: bool = False) -> list[dict]:
    body = {"model": "yamadori", "stream": True,
            "messages": [{"role": "user", "content": user}]}
    if include_usage:
        body["stream_options"] = {"include_usage": True}
    if tools:
        body["tools"] = tools
    out = []
    for b in proxy.stream_body(body, "yamadori"):
        if b.strip() == b"data: [DONE]":
            out.append({"_done": True})
            continue
        assert b.startswith(b"data: ") and b.endswith(b"\n\n"), b[:60]
        out.append(json.loads(b[6:].decode()))
    return out


def deltas(events: list[dict], key: str) -> str:
    return "".join((e["choices"][0]["delta"].get(key) or "")
                   for e in events if e.get("choices"))


def finish_of(events: list[dict]) -> str | None:
    fins = [e["choices"][0]["finish_reason"] for e in events
            if e.get("choices") and e["choices"][0].get("finish_reason")]
    return fins[-1] if fins else None


def answers() -> list[tuple]:
    return [x for x in _log if x[0] == "answer"]


# --------------------------------------------------------------------------


def test_fake_upstream_is_alive():
    # PROTOCOL rule 1: prove the fixture works before judging anything on it.
    reset([reply("pong")])
    d = proxy._post("/v1/chat/completions", {"messages": []})
    check(d["choices"][0]["message"]["content"] == "pong",
          "fake upstream answers through the proxy's own reader",
          json.dumps(d)[:200])
    check(_seen and _seen[0].get("stream") is True,
          "the reader asked for a stream", json.dumps(_seen)[:200])


def test_an_answer_without_tools_is_generated_once():
    reset([reply("It is in src/geometries/LatheGeometry.js.",
                 reasoning="The user wants a file. I know this one.")])
    ev = run_stream()
    check(len(_seen) == 1, "exactly ONE upstream generation for the answer",
          f"{len(_seen)} requests")
    check(deltas(ev, "content") == "It is in src/geometries/LatheGeometry.js.",
          "the answer is the streamed generation, token by token",
          repr(deltas(ev, "content")))
    # A short answer is held until its hop ends (CHANNEL ORDER); a long one
    # streams live once it passes proxy.HOLD_CONTENT_CHARS.
    long_answer = "line of the answer. " * 60
    reset([reply(long_answer, reasoning="thinking")])
    ev2 = run_stream()
    check(deltas(ev2, "content") == long_answer
          and sum(1 for e in ev2 if e.get("choices")
                  and e["choices"][0]["delta"].get("content")) > 1,
          "a long answer streams live as several deltas, not one blob")
    check(deltas(ev, "reasoning_content")
          == "The user wants a file. I know this one.",
          "reasoning goes out as reasoning_content deltas")
    check("I know this one" not in deltas(ev, "content"),
          "reasoning never appears as content")
    check(finish_of(ev) == "stop", "finish_reason stop", str(finish_of(ev)))
    check(ev[-1] == {"_done": True}, "stream ends with [DONE]")
    check(answers() and answers()[0][1] == 0,
          "logged hop count is 0 for an answer that called no tool",
          str(answers()))


def test_a_tool_hop_then_the_answer_is_two_generations_not_three():
    reset([reply(reasoning="look it up",
                 calls=[img()]),
           reply("src/geometries/LatheGeometry.js:12", reasoning="found it",
                 usage={"prompt_tokens": 180, "completion_tokens": 10,
                        "total_tokens": 190})])
    ev = run_stream(include_usage=True)
    check(len(_seen) == 2,
          "one tool hop + one answer = 2 upstream requests (was 3)",
          f"{len(_seen)} requests")
    check(_ran == [(IMG, {"prompt": "a lathe"})],
          "our tool ran server-side", str(_ran))
    check(any(m.get("role") == "tool" for m in _seen[1]["messages"]),
          "the second request carries the tool result")
    check("src/geometries/LatheGeometry.js:12" in deltas(ev, "content"),
          "the answer is streamed")
    line = proxy.streaming.describe_call(IMG, {"prompt": "a lathe"})
    check(deltas(ev, "reasoning_content")
          == f"look it up`{line}`\nfound it",
          "reasoning from BOTH hops streamed live, with our tool activity on "
          "the thinking channel (never as content)",
          repr(deltas(ev, "reasoning_content")))
    check(not any(e.get("choices") and e["choices"][0]["delta"].get("tool_calls")
                  for e in ev),
          "our tool call is never forwarded to the client")
    check(answers() and answers()[0][1] == 1,
          "logged hop count is 1 (was MAX_TOOL_HOPS)", str(answers()))
    # U1 (2026-09-25): usage is the FINAL generation's -- the context the
    # conversation now occupies -- not the sum over hops (it was 200 + 20,
    # and Hermes compacts on prompt_tokens); the sums are in x_yamadori.
    last = [e for e in ev if e.get("usage")]
    u = last[-1]["usage"] if last else {}
    xu = (final_chunk(ev).get("x_yamadori") or {}).get("usage") or {}
    check(last and last[-1].get("choices") == []
          and u.get("prompt_tokens") == 180
          and u.get("completion_tokens") == 10
          and u.get("total_tokens") == 190 and "hops" not in u
          and xu.get("summed", {}).get("prompt_tokens") == 280
          and xu.get("generations") == 2,
          "usage is the final generation's, on its own choices:[] chunk; "
          "the per-hop sums are in x_yamadori.usage",
          json.dumps({"usage": u, "x": xu})[:400])


def test_no_reasoning_after_the_first_content():
    # Live diagnostic 2026-09-24: our tool line went out as content between
    # two hops of reasoning; clients close their thinking block at the first
    # content delta, reopened it, and rendered the real answer as thinking.
    reset([reply("Let me check that.", reasoning="plan",
                 calls=[img("fov")]),
           reply("It sets the vertical field of view.", reasoning="found")])
    ev = run_stream()
    kinds = []
    for e in ev:
        dl = (e.get("choices") or [{}])[0].get("delta") or {}
        if dl.get("reasoning_content"):
            kinds.append("R")
        if dl.get("content"):
            kinds.append("C")
    first_c = kinds.index("C") if "C" in kinds else len(kinds)
    check("R" not in kinds[first_c:],
          "no reasoning delta after the first content delta", "".join(kinds))
    check(deltas(ev, "content") == "It sets the vertical field of view.",
          "only the final answer is content; the tool hop's preface is not",
          repr(deltas(ev, "content")))
    check("Let me check that." in deltas(ev, "reasoning_content"),
          "the tool hop's preface went out as reasoning",
          repr(deltas(ev, "reasoning_content"))[:200])


def test_an_image_is_content_when_made_and_later_reasoning_a_heartbeat():
    """IMAGES REACH THE CHAT (operator, 2026-09-25): yama_generate_image's picture
    goes out as content the moment the tool returns; the reasoning of the
    hop after it goes out as a heartbeat, never as reasoning (CHANNEL
    ORDER). mcp/test_image_emit.py has the rest (ledger, dedup, blocking)."""
    url = ("https://img.test/media/" + "ab" * 32
           + ".png?exp=1893456000&sig=cd")
    md = f"![a lathe]({url})"
    reset([reply("", reasoning="draw it", calls=[img()]),
           reply("There is the lathe.", reasoning="now explain")])
    TOOL_OUTPUT[IMG] = json.dumps({"tool": IMG, "ok": True, "markdown": md,
                                   "url": url, "instruction": "copy it"})
    ev = run_stream()
    kinds = []
    for e in ev:
        ch = (e.get("choices") or [{}])[0]
        dl = ch.get("delta") or {}
        if dl.get("reasoning_content"):
            kinds.append("R")
        elif dl.get("content"):
            kinds.append("C")
        elif not dl and ch.get("finish_reason") is None and "choices" in e:
            kinds.append("H")
    first_c = kinds.index("C") if "C" in kinds else len(kinds)
    check("R" not in kinds[first_c:] and "H" in kinds[first_c:],
          "no reasoning after the image (the first content): a heartbeat "
          "stands in for it", "".join(kinds))
    check(deltas(ev, "content") == md + "\n\nThere is the lathe.",
          "the image line, then the answer",
          repr(deltas(ev, "content"))[:300])
    check("now explain" not in deltas(ev, "reasoning_content"),
          "the reasoning after the image is not forwarded")
    tool = next((m for m in _seen[1]["messages"] if m.get("role") == "tool"),
                {}) if len(_seen) > 1 else {}
    res = json.loads(tool.get("content") or "{}")
    check(res.get("shown_to_user") is True
          and "already shown" in res.get("instruction", ""),
          "the model reads that the picture is already shown",
          json.dumps(res)[:300])

def test_reasoning_continuity():
    """PAST REASONING IS RESTORED (operator, 2026-09-27, reversing the
    2026-09-24 pass-through: "Keeping thinking across turns seems useful,
    fuck Hermes, Hermes can do whatever it wants."). Hermes strips
    reasoning_content from every replayed assistant turn
    (agent/message_sanitization.py apply_reasoning_content_policy); the
    ledger records the slot's reasoning for each turn it delivers and puts
    it back, so the prompt is the slot's sequence again (Bonsai 2's base,
    Qwen3.8-27B, keeps thinking across turns by default). A client that
    echoes keeps its echo as sent. Switch `restore_reasoning` off: the
    pass-through (nothing restored)."""
    proxy.nebari.ledger_reset()
    hist = [{"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer",
             "reasoning_content": "old thought"},
            {"role": "user", "content": "build it"},
            {"role": "assistant", "content": "", "reasoning_content":
             "plan: read game.js",
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "read_file",
                                          "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "file text"}]
    # What the proxy delivered for that call turn is recorded (content,
    # hops) and, since 2026-09-27, the slot's reasoning for it.
    proxy.ledger_record_turn("acct", "s1", dict(hist[3]),
                             reasoning=hist[3]["reasoning_content"])
    out = proxy.scope_reasoning(hist, "acct")
    check(out[1].get("reasoning_content") == "old thought"
          and out[3].get("reasoning_content") == "plan: read game.js",
          "an echoing client's reasoning passes through unchanged")
    bare = [dict(m) for m in hist]
    for m in bare:
        m.pop("reasoning_content", None)
    out = proxy.scope_reasoning(bare, "acct")
    check(out[3].get("reasoning_content") == "plan: read game.js"
          and not out[1].get("reasoning_content"),
          "a client that dropped its reasoning gets the slot's back on the "
          "turn the proxy recorded (and nothing on a turn it has no record "
          "for)", json.dumps([m.get("reasoning_content") for m in out]))
    check(proxy.nebari.ledger_get("acct", "call:c1", "reasoning")
          == "plan: read game.js",
          "recorded as kind `reasoning` under the turn's key")
    off = proxy.scope_reasoning(bare, "acct",
                                features=json.dumps(
                                    {"restore_reasoning": False}))
    check(all(not m.get("reasoning_content") for m in off),
          "switch restore_reasoning off: the pass-through, nothing restored")
    shown = [dict(m) for m in hist]
    shown[3]["reasoning_content"] = ("  yama_generate_image x\ntool "
                                     "trace\nplan: read game.js")
    out = proxy.scope_reasoning(shown, "acct")
    check(out[3].get("reasoning_content") == shown[3]["reasoning_content"],
          "an echoed trace (what the client was SHOWN) is what the model sees")
    proxy.nebari.ledger_reset()


def test_internal_hops_keep_their_reasoning():
    reset([reply(reasoning="need the definition",
                 calls=[img()]),
           reply("src/geometries/LatheGeometry.js:12", reasoning="found it")])
    run_stream()
    second = _seen[1]["messages"] if len(_seen) > 1 else []
    asst = [m for m in second if m.get("role") == "assistant"]
    check(asst and asst[-1].get("reasoning_content") == "need the definition",
          "the next hop sees the previous hop's reasoning",
          json.dumps(asst[-1] if asst else None)[:200])

def test_the_breaker_lands_with_tools_withdrawn():
    # There is no hop count any more: the landing is forced the way it
    # happens for real, by the context share filling (proxy.context_full).
    prev = proxy.context_full
    n = {"calls": 0}

    def full_on_third(*a, **k):
        # context_full is asked on OUR hops only (hop 1+) since
        # 2026-09-25 (C1): the client's own prompt is measured before
        # the turn and never landed, so the Nth call is hop N.
        n["calls"] += 1
        return n["calls"] >= 2
    proxy.context_full = full_on_third
    try:
        # The model asks for our tool on EVERY hop, including the landing --
        # the worst case: nothing of ours may reach the client even then.
        loop = [reply(calls=[img("x", cid=f"c{i}")]) for i in range(3)]
        reset(loop)
        ev = run_stream()
    finally:
        proxy.context_full = prev
    check(len(_seen) == 3, "3 hops, the third landing on a full context",
          f"{len(_seen)}")
    check(all(_seen[i].get("tools") for i in (0, 1)),
          "tools offered on the hops before the landing")
    land = _seen[-1] if _seen else {}
    check(land.get("tools") in (None, []),
          "the landing request carries NO tools",
          json.dumps(land.get("tools"))[:200])
    check(any(m.get("role") == "user" and m.get("content") == proxy.LANDING_PROMPT
              for m in land.get("messages") or []),
          "the landing asks for the answer, same prompt as complete()")
    check(not any(e.get("choices") and e["choices"][0]["delta"].get("tool_calls")
                  for e in ev),
          "no call to OUR tool is forwarded to the client")
    check(len(_ran) == 2, "a call made AT the landing is not run",
          f"{len(_ran)} tool runs")
    check(finish_of(ev) != "tool_calls", "finish is not tool_calls",
          str(finish_of(ev)))
    # The "[no answer: the tool loop reached its breaker ...]" notice was
    # REMOVED 2026-09-27 (docs/CONSTANTS-AUDIT.md): an answerless landing
    # ends with empty content and finish_reason stop.
    check(deltas(ev, "content") == "" and finish_of(ev) == "stop",
          "an answerless landing: empty content, finish_reason stop",
          repr(deltas(ev, "content")))


def test_complete_lands_through_the_same_helper():
    prev = proxy.context_full
    n = {"calls": 0}

    def full_on_second(*a, **k):
        # context_full is asked on OUR hops only (hop 1+) since
        # 2026-09-25 (C1): the client's own prompt is measured before
        # the turn and never landed, so the Nth call is hop N.
        n["calls"] += 1
        return n["calls"] >= 1
    proxy.context_full = full_on_second
    try:
        reset([reply(calls=[img("x")]),
               reply("answered")])
        d = proxy.complete({"model": "yamadori",
                            "messages": [{"role": "user", "content": "q"}]})
    finally:
        proxy.context_full = prev
    check(len(_seen) == 2 and _seen[1].get("tools") in (None, []),
          "complete(): landing request carries no tools",
          json.dumps([s.get("tools") is not None for s in _seen]))
    check(d["choices"][0]["message"]["content"] == "answered",
          "complete(): the landing's answer is returned")


def test_the_tool_turn_cap_lands_and_is_recorded():
    # PrismML's agenticMaxTurns: after the tier's tool-turn limit the loop
    # LANDS (tools withdrawn, answer asked for) -- it never returns a tool
    # request as the answer -- and x_yamadori.tool_turns says it was hit.
    import tiers
    prev = tiers.TOOL_TURNS
    tiers.TOOL_TURNS = 2
    try:
        loop = [reply(calls=[img("x", cid=f"c{i}")]) for i in range(2)]
        reset(loop + [reply("answered from what I had")])
        ev = run_stream()
        seen_stream, ran_stream = list(_seen), list(_ran)
        reset(loop + [reply("answered from what I had")])
        d = proxy.complete({"model": "yamadori",
                            "messages": [{"role": "user", "content": "q"}]})
    finally:
        tiers.TOOL_TURNS = prev
    for label, seen, ran, x, content in (
            ("streamed", seen_stream, ran_stream,
             final_chunk(ev).get("x_yamadori") or {}, deltas(ev, "content")),
            ("complete()", _seen, _ran, d.get("x_yamadori") or {},
             d["choices"][0]["message"]["content"])):
        check(len(seen) == 3, f"{label}: 2 tool turns + the landing = 3 "
              "upstream requests", str(len(seen)))
        check(len(ran) == 2, f"{label}: both tool turns ran", str(len(ran)))
        land = seen[-1] if seen else {}
        check(land.get("tools") in (None, [])
              and any(m.get("content") == proxy.LANDING_PROMPT
                      for m in land.get("messages") or []),
              f"{label}: the capped hop withdraws tools and asks for the answer")
        check(x.get("tool_turns") == {"limit": 2, "turns": 2, "hit": True},
              f"{label}: x_yamadori.tool_turns records the cap",
              json.dumps(x.get("tool_turns")))
        check("answered from what I had" in (content or ""),
              f"{label}: the landing's answer is delivered", repr(content)[:200])


def test_an_uncapped_turn_records_the_cap_unhit():
    reset([reply(calls=[img("x")]),
           reply("answered")])
    d = proxy.complete({"model": "yamadori",
                        "messages": [{"role": "user", "content": "q"}]})
    check((d.get("x_yamadori") or {}).get("tool_turns")
          == {"limit": 10, "turns": 1, "hit": False},
          "one tool turn under the cap at the default tier: limit 10, hit=False",
          json.dumps((d.get("x_yamadori") or {}).get("tool_turns")))
    import tiers
    got = {n: tiers.tool_turn_limit(n) for n in tiers.ORDER}
    check(got == {"minimal": 10, "low": 10, "medium": 10, "high": 10,
                  "xhigh": 10, "max": 20},
          "limit per tier: the vendor's 10 everywhere, 20 at max", str(got))


def test_a_client_tool_call_is_forwarded_whole():
    client_tool = {"type": "function", "function": {
        "name": "read_file", "parameters": {"type": "object"}}}
    reset([reply(calls=[call("read_file", {"path": "package.json"})])])
    ev = run_stream(tools=[client_tool])
    tc = [e["choices"][0]["delta"]["tool_calls"] for e in ev
          if e.get("choices") and e["choices"][0]["delta"].get("tool_calls")]
    check(len(_seen) == 1, "one upstream request", f"{len(_seen)}")
    check(tc and tc[0][0]["function"]["name"] == "read_file"
          and tc[0][0].get("index") == 0,
          "the client's call is forwarded with an index", json.dumps(tc)[:200])
    check(finish_of(ev) == "tool_calls", "finish_reason tool_calls",
          str(finish_of(ev)))
    check(not _ran, "the client's tool is not run by the proxy", str(_ran))


def test_a_length_finish_is_finish_reason_length_only():
    """The budget notice ("[no answer: ... token limit ...]" / "[answer cut
    off ...]") was REMOVED 2026-09-27 (docs/CONSTANTS-AUDIT.md): the fact is
    finish_reason "length", and the content is what was generated."""
    usage = {"prompt_tokens": 50, "completion_tokens": 10240,
             "total_tokens": 10290}
    reset([reply("", reasoning="still deliberating about the answer",
                 finish="length", usage=usage)])
    ev = run_stream()
    content = deltas(ev, "content")
    check(content == "", "empty answer: no content, no notice",
          repr(content))
    check("deliberating" not in content,
          "reasoning is never passed off as the answer")
    check(finish_of(ev) == "length", "finish_reason stays length",
          str(finish_of(ev)))
    check(len(_seen) == 1, "no regeneration after a length finish",
          f"{len(_seen)}")

    reset([reply("partial answer", finish="length", usage=usage)])
    ev = run_stream()
    content = deltas(ev, "content")
    check(content == "partial answer" and finish_of(ev) == "length",
          "partial answer: delivered as generated, finish_reason length",
          repr(content))
    check(answers() and answers()[0][2] == "partial answer",
          "the logged answer is what was generated", str(answers()))


def test_truncation_marker_at_the_breaker():
    lines = "\n".join(f"{i:>5}  const value{i} = computeSomethingLong({i});"
                      for i in range(1, 181))
    text = f"src/big.ts:1-180  (400 lines total)\n```\n{lines}\n```"
    out = repeats.cap_tool_result(text, "read_file_range",
                                  {"path": "src/big.ts", "start": 1, "end": 180})
    check(len(text) > 6000, "fixture is over the breaker", str(len(text)))
    check(f"[truncated at 6000 of {len(text)} chars;" in out,
          "marker states the breaker and the full size", out[-300:])
    check("request lines" in out and 'path="src/big.ts"' in out
          and "end=180" in out,
          "read_file_range: marker names the exact next call", out[-300:])
    body = out.split("\n\n[truncated")[0]
    last = int(body.strip().splitlines()[-1].split()[0])
    check(f"start={last + 1}" in out, "next start follows the last line shown",
          f"last shown {last}; {out[-200:]}")
    check(len(body) <= 6000, "no more than 6000 chars of result kept",
          str(len(body)))
    check(repeats.cap_tool_result("short", "x") == "short",
          "a result under the breaker is untouched")
    other = repeats.cap_tool_result("hit\n" * 3000, "find_by_pattern",
                                    {"pattern": "x"})
    check("[truncated at 6000 of 12000 chars;" in other and "glob" in other,
          "search tools get a narrowing step", other[-200:])

    # And the streamed loop really applies it: the tool message that goes
    # upstream on the next hop carries the marker.
    reset([reply(calls=[img("big")]),
           reply("done")])
    TOOL_OUTPUT[IMG] = text
    run_stream()
    tool_msgs = [m for m in (_seen[1]["messages"] if len(_seen) > 1 else [])
                 if m.get("role") == "tool"]
    check(tool_msgs and "[truncated at 6000 of" in tool_msgs[0]["content"],
          "stream_body sends the marked result upstream",
          (tool_msgs[0]["content"][-200:] if tool_msgs else "no tool message"))


def final_chunk(events: list[dict]) -> dict:
    fins = [e for e in events if e.get("choices")
            and e["choices"][0].get("finish_reason")]
    return fins[-1] if fins else {}


def test_x_yamadori_rides_the_final_chunk_and_matches_complete():
    """The same decisions, observable on real output, on both paths."""
    script = [reply(calls=[img("X")]),
              reply("found it in src/x.ts")]
    reset(list(script))
    ev = run_stream()
    xs = final_chunk(ev).get("x_yamadori")
    check(isinstance(xs, dict), "streamed: x_yamadori is on the chunk that "
          "carries finish_reason", json.dumps(final_chunk(ev))[:200])
    check(sum(1 for e in ev if "x_yamadori" in e) == 1,
          "and on no other chunk")
    reset(list(script))
    d = proxy.complete({"model": "yamadori",
                        "messages": [{"role": "user", "content":
                                      "which file defines LatheGeometry"}]})
    xc = d.get("x_yamadori")
    check(isinstance(xc, dict) and set(xc) == set(xs or {}),
          "non-streamed: the same object, the same keys",
          str(sorted(set(xc or {}) ^ set(xs or {}))))
    check((xs or {}).get("hops") == (xc or {}).get("hops") == 2,
          "the same prompt gives the same hop count streamed and not",
          f"{(xs or {}).get('hops')} vs {(xc or {}).get('hops')}")


# --------------------------------------------------------------------------
# THE COMPACTION STORE AND THE PREFILL CHANNEL
# --------------------------------------------------------------------------

def _conversation(account: str = "acct", key: str = "conv-1") -> dict:
    return {"_slot": {"key": key, "transient": False, "account": account,
                      "record": False},
            "_ledger": {"account": account, "session": key,
                        "turn_key": "u:" + key}}


def test_the_compaction_store_keeps_the_ledgers_rendering():
    """Pre-deploy review, 2026-09-24 (FIX SOON #3). compaction.record stored
    the last generation's prompt as it went upstream: the hidden hops WITH
    their reasoning, and after a landing the LANDING_PROMPT and no tools.
    _serve_compaction's extension check then never matched the ledger's
    rendering and its splice/continue fallback reinjected that hidden
    reasoning. The store now holds the prompt the way the ledger renders
    it: hops with empty reasoning, no landing request, the turn's tools."""
    import compaction
    prev = proxy.context_full
    for landed in (False, True):
        proxy.slots.reset(4)
        proxy.nebari.ledger_reset()
        compaction.reset()
        n = {"calls": 0}

        def full_on_second(*a, **k):
            # Asked on OUR hops only (hop 1+) since 2026-09-25 (C1).
            n["calls"] += 1
            return landed and n["calls"] >= 1
        proxy.context_full = full_on_second
        try:
            reset([reply(reasoning="HIDDEN-HOP-REASONING", calls=[img()]),
                   reply("src/a.ts:1", reasoning="final thought")])
            PREP.update(_conversation())
            PREP["_slot"]["record"] = True
            run_stream()
        finally:
            proxy.context_full = prev
            PREP.clear()
        e = next(iter(compaction.entries("acct")), None)
        msgs = (e or {}).get("upstream", {}).get("messages") or []
        hops = [m for m in msgs if m.get("role") == "assistant"
                and m.get("tool_calls")]
        what = "after a landing" if landed else "a hop, then the answer"
        check(e is not None and len(_seen) == 2
              and any(m.get("role") == "user"
                      and m.get("content") == proxy.LANDING_PROMPT
                      for m in _seen[-1].get("messages") or []) == landed,
              f"{what}: the turn was recorded (the fixture ran the case)",
              json.dumps([m.get("role") for m in msgs]))
        check(hops and all(not h.get("reasoning_content") for h in hops)
              and "HIDDEN-HOP-REASONING" not in json.dumps(msgs),
              f"{what}: the stored prompt's hidden hop has EMPTY reasoning, "
              "as the ledger renders it", json.dumps(hops)[:300])
        check(not any(m.get("content") == proxy.LANDING_PROMPT for m in msgs)
              and bool(e and e["upstream"].get("tools")),
              f"{what}: no landing request stored, and the turn's own tools",
              json.dumps([m.get("content") for m in msgs])[:300])
    compaction.reset()
    proxy.nebari.ledger_reset()


def test_a_prefill_survives_an_image_hop():
    """Pre-deploy review, 2026-09-24 (MINOR): a prefill opens hop 0; hops 1+
    see hop 0 in its place. Where the upstream did not re-send the prefill,
    it was gone from every later hop of a request that also ran an image
    tool. It is carried into the hop, exactly once whether or not the
    upstream re-sends it. Since 2026-09-29 the one prefill left is the
    DIRECTIVE (proxy.directive_prefill: reasoning only, content empty);
    nothing in prepare() sets one today, so the stub's PREP carries it."""
    global ECHO_PREFILL
    line = "Next I'll draw the lathe HANDOFF-7 and then describe it"
    pre = proxy.directive_prefill(line)
    for echo in (False, True):
        ECHO_PREFILL = echo
        try:
            reset([reply(" it lathes.", reasoning="now draw", calls=[img()]),
                   reply("Here it is.")])
            PREP.update(_prefill=dict(pre))
            run_stream()
        finally:
            ECHO_PREFILL = True
            PREP.clear()
        hop = next((m for m in (_seen[1]["messages"] if len(_seen) > 1
                                else [])
                    if m.get("role") == "assistant" and m.get("tool_calls")),
                   {})
        r = hop.get("reasoning_content") or ""
        what = "upstream re-sends it" if echo else "upstream does not"
        check(r.startswith(line) and r.count("HANDOFF-7") == 1
              and "now draw" in r
              and (hop.get("content") or "") == " it lathes.",
              f"{what}: hop 1 sees the directive in hop 0's reasoning exactly "
              f"once, and hop 0's own content", json.dumps(hop)[:300])


# --------------------------------------------------------------------------
# A CLIENT THAT GOES AWAY (#39, Octopus v0b-V0-xhigh-1): a request streamed
# nothing for ~900 s, Hermes' stale detector killed the stream and retried
# -- and the first request's generation ran on beside its retry, which
# decoded at 1.2-3.4 tok/s. (The second-brain half of #39 -- deep thinking
# holding the helper lane -- went with deep thinking, 2026-09-29.)
# --------------------------------------------------------------------------

class _Slow(BaseHTTPRequestHandler):
    """An upstream that takes its time: it reads the request, then waits up
    to HOLD seconds for the client to hang up, recording when it did; a
    client still there gets an answer (non-streamed JSON, or SSE)."""
    HOLD = 4.0
    started: list = []
    gone: list = []

    def do_POST(self):                                           # noqa: N802
        import select
        import time as _t
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Slow.started.append(_t.time())
        stream = bool(body.get("stream"))
        if stream:
            # Headers now, tokens later: a prefill.
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.flush()
        t_end = _t.time() + _Slow.HOLD
        sock = self.connection
        while _t.time() < t_end:
            r, _w, _x = select.select([sock], [], [], 0.05)
            if r:
                try:
                    if sock.recv(1, socket_peek()) == b"":
                        _Slow.gone.append(_t.time())
                        return
                except OSError:
                    _Slow.gone.append(_t.time())
                    return
        try:
            if stream:
                self.wfile.write(_sse(reply("late answer")))
            else:
                data = _json(reply("late answer"))
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
        except OSError:
            _Slow.gone.append(_t.time())

    def log_message(self, *a):                                   # noqa: D102
        pass


def socket_peek() -> int:
    import socket as _s
    return _s.MSG_PEEK


def _slow_server() -> tuple:
    _Slow.started, _Slow.gone = [], []
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Slow)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def _is_heartbeat(b: bytes) -> bool:
    if not b.startswith(b"data: {"):
        return False
    e = json.loads(b[6:].decode())
    ch = (e.get("choices") or [{}])[0]
    return ch.get("delta") == {} and not ch.get("finish_reason")


def test_a_disconnect_during_a_silent_prefill_stops_main():
    """The turn is blocked in main's upstream read (a long prefill sends no
    token, so the generator never yields and cannot be closed): cancelling
    the token from the event loop shuts the socket, the read returns, and
    the stream ends at once."""
    import time as _t
    srv, url = _slow_server()
    saved = proxy.UPSTREAM
    proxy.UPSTREAM = url
    _Slow.HOLD = 4.0
    try:
        reset([])
        body = {"model": "yamadori", "stream": True, "messages": [
            {"role": "user", "content": "hello"}]}
        tok = proxy.cancel.Token()
        gen = proxy.stream_body(body, "yamadori", token=tok)
        box: dict = {"chunks": []}

        def _drive():
            try:
                for b in gen:
                    box["chunks"].append(b)
            except Exception as e:                               # noqa: BLE001
                box["err"] = e
            box["end"] = _t.time()
        th = threading.Thread(target=_drive, daemon=True)
        th.start()
        while not _Slow.started and th.is_alive():
            _t.sleep(0.01)
        _t.sleep(0.2)
        t0 = _t.time()
        tok.cancel("the client disconnected")
        th.join(timeout=2.0)
        took = (box.get("end") or 99) - t0
        check(not th.is_alive() and took < 1.0 and "err" not in box,
              "the stream ends within a second of the disconnect, quietly",
              f"{took:.2f}s, alive {th.is_alive()}, err {box.get('err')!r}")
        check(bool(_Slow.gone) and _Slow.gone[0] - t0 < 1.0,
              "main's upstream connection was closed",
              str([round(g - t0, 2) for g in _Slow.gone]))
        check(not any(b"upstream error" in c or b"late answer" in c
                      for c in box["chunks"]),
              "and nothing about it was sent as content", str(box["chunks"])[:200])
    finally:
        proxy.UPSTREAM = saved
        srv.shutdown()


def test_the_server_cancels_the_token_on_disconnect():
    """server._CancellingStream: Starlette's http.disconnect cancels the
    request's token (the part of the fix that does not wait for the turn to
    yield)."""
    import asyncio
    import server
    tok = proxy.cancel.Token()

    async def body():
        yield b""

    resp = server._CancellingStream(body(), tok,
                                    media_type="text/event-stream")

    async def receive():
        return {"type": "http.disconnect"}
    asyncio.run(resp.listen_for_disconnect(receive))
    check(tok.cancelled and "disconnect" in tok.why,
          "an http.disconnect cancels the turn's token", tok.why)

    # A client closing the connection AFTER the whole reply is an ordinary
    # close (2026-09-25: every finished stream logged "client disconnected"
    # and cancelled the token post-reply work is bound to).
    tok2 = proxy.cancel.Token()
    resp2 = server._CancellingStream(body(), tok2,
                                     media_type="text/event-stream")
    sent = []

    async def send(msg):
        sent.append(msg)

    async def finish_then_close():
        await resp2.stream_response(send)
        await resp2.listen_for_disconnect(receive)
    asyncio.run(finish_then_close())
    check(not tok2.cancelled and any(m.get("type") == "http.response.body"
                                     for m in sent),
          "a close after the full reply does not cancel the token",
          str(tok2.why))


class _Dribble(BaseHTTPRequestHandler):
    """An upstream that writes a CLIENT tool call slowly: REASONING (maybe
    empty) at once, then the call's arguments in PIECES pieces, PAUSE
    seconds apart -- the model transcribing a long argument (Octopus
    v0e-V0-xhigh-1: a screenshot's base64 into a tool call, 40 minutes of
    tool-call deltas and not one byte to the client)."""
    REASONING = "I will write the file."
    PIECES = 20
    PAUSE = 0.05
    ARGS = json.dumps({"path": "a.py", "content": "x = 1\n" * 40})

    def do_POST(self):                                           # noqa: N802
        import time as _t
        self.rfile.read(int(self.headers["Content-Length"]))

        def ev(delta=None, finish=None, **extra):
            d = {"id": "up-1", "object": "chat.completion.chunk",
                 "model": "bonsai", "created": 1,
                 "choices": [{"index": 0, "delta": delta or {},
                              "finish_reason": finish}]}
            d.update(extra)
            return b"data: " + json.dumps(d).encode() + b"\n\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        try:
            self.wfile.write(ev({"role": "assistant"}))
            if _Dribble.REASONING:
                self.wfile.write(ev({"reasoning_content": _Dribble.REASONING}))
            self.wfile.flush()
            a = _Dribble.ARGS
            n = max(1, len(a) // _Dribble.PIECES + 1)
            for k, i in enumerate(range(0, len(a), n)):
                tc = {"index": 0, "function": {"arguments": a[i:i + n]}}
                if k == 0:
                    tc.update(id="call_w", type="function")
                    tc["function"]["name"] = "write_file"
                self.wfile.write(ev({"tool_calls": [tc]}))
                self.wfile.flush()
                _t.sleep(_Dribble.PAUSE)
            self.wfile.write(ev({}, "tool_calls"))
            self.wfile.write(b"data: " + json.dumps({
                "id": "up-1", "object": "chat.completion.chunk",
                "choices": [], "usage": {"prompt_tokens": 10,
                                         "completion_tokens": 50,
                                         "total_tokens": 60}}).encode()
                + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except OSError:
            pass

    def log_message(self, *a):                                   # noqa: D102
        pass


def test_a_long_client_tool_call_keeps_the_stream_alive():
    """While the model writes a CLIENT tool call's arguments nothing of it
    can go out (the call is forwarded whole, after the code check), but the
    stream must still carry an empty delta every HEARTBEAT seconds: without
    one, Hermes' 900 s stale detector killed the request and retried, and a
    relay that only learns of a hang-up when it has bytes to write kept the
    abandoned request -- and its main lane -- alive for 40 minutes
    (Octopus v0e-V0-xhigh-1, #44). And a call with NO reasoning before it
    must not hold back the first byte until the whole call is written."""
    import time as _t
    client_tool = {"type": "function", "function": {
        "name": "write_file", "parameters": {"type": "object"}}}
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Dribble)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    saved = (proxy.UPSTREAM, proxy.HEARTBEAT)
    proxy.UPSTREAM = f"http://127.0.0.1:{srv.server_address[1]}"
    proxy.HEARTBEAT = 0.1
    try:
        for reasoning in ("I will write the file.", ""):
            _Dribble.REASONING = reasoning
            reset([])
            body = {"model": "yamadori", "stream": True, "tools": [client_tool],
                    "messages": [{"role": "user", "content": "write a.py"}]}
            t0 = _t.time()
            first = None
            chunks = []
            for b in proxy.stream_body(body, "yamadori"):
                if first is None:
                    first = _t.time() - t0
                chunks.append(b)
            ev = [json.loads(b[6:].decode()) for b in chunks
                  if b.startswith(b"data: {")]
            calls = [c for e in ev if e.get("choices")
                     for c in e["choices"][0]["delta"].get("tool_calls") or []]
            call_at = next(i for i, e in enumerate(ev) if e.get("choices")
                           and e["choices"][0]["delta"].get("tool_calls"))
            beats = sum(1 for b in chunks[:call_at] if _is_heartbeat(b))
            label = "after reasoning" if reasoning else "with no reasoning"
            check(beats >= 4,
                  f"{label}: empty deltas go out while the call's arguments "
                  f"are written (~1 s of arguments, beat 0.1 s)",
                  f"{beats} heartbeats before the call")
            check(first is not None and first < 0.5,
                  f"{label}: the first byte does not wait for the whole call",
                  f"first byte at {first}")
            check(len(calls) == 1 and calls[0]["function"]["arguments"]
                  == _Dribble.ARGS and calls[0]["function"]["name"]
                  == "write_file",
                  f"{label}: the call is still forwarded whole and unchanged",
                  json.dumps(calls)[:160])
            check(deltas(ev, "content") == ""
                  and deltas(ev, "reasoning_content") == reasoning,
                  f"{label}: the heartbeats add nothing to content or "
                  f"reasoning", repr(deltas(ev, "content"))[:80])
    finally:
        proxy.UPSTREAM, proxy.HEARTBEAT = saved
        _Dribble.REASONING = "I will write the file."
        srv.shutdown()


# --------------------------------------------------------------------------
# The server's door, driven as ASGI (server.app), with proxy.stream_body and
# the key check stubbed: what holds a main lane, and when it is given back.
# --------------------------------------------------------------------------

class _Chat:
    """One POST /v1/chat/completions against server.app, as a task. Its
    client hangs up when `hang_up` is set; `status`, `chunks`."""

    def __init__(self, app, body: dict, first_watch_blocks: bool = False):
        import asyncio
        self.app, self.body = app, body
        self.status = None
        self.chunks: list = []
        self.hang_up = asyncio.Event()
        self.started = asyncio.Event()
        self.n_receive = 0
        # _first_chunk's watch is the 2nd receive; the leak test keeps it
        # waiting so the disconnect lands on Starlette's own listener.
        self.first_watch_blocks = first_watch_blocks
        self.task = None

    async def _receive(self):
        import asyncio
        self.n_receive += 1
        if self.n_receive == 1:
            return {"type": "http.request", "more_body": False,
                    "body": json.dumps(self.body).encode()}
        if self.first_watch_blocks and self.n_receive == 2:
            await asyncio.Event().wait()             # cancelled, never set
        await self.hang_up.wait()
        return {"type": "http.disconnect"}

    async def _send(self, msg):
        import asyncio
        if msg["type"] == "http.response.start":
            self.status = msg["status"]
            self.started.set()
            await asyncio.sleep(0)
        elif msg["type"] == "http.response.body" and msg.get("body"):
            self.chunks.append(msg["body"])

    def start(self):
        import asyncio
        scope = {"type": "http", "method": "POST", "scheme": "http",
                 "path": "/v1/chat/completions", "raw_path":
                 b"/v1/chat/completions", "query_string": b"",
                 "root_path": "", "http_version": "1.1",
                 "asgi": {"version": "3.0"},
                 "client": ("127.0.0.1", 50000),
                 "server": ("127.0.0.1", 1234),
                 "headers": [(b"authorization", b"Bearer test"),
                             (b"content-type", b"application/json")]}
        self.task = asyncio.ensure_future(
            self.app(scope, self._receive, self._send))
        return self


def _fake_stream_body(record: list):
    """proxy.stream_body's stand-in: a first chunk at once, then an empty
    delta every 20 ms until the request's token is cancelled (a turn that
    runs until something stops it). `record` gets (token, closed?)."""
    def fake(body, public_name="yamadori", token=None):
        rec = {"token": token, "closed": False, "ended": False}
        record.append(rec)
        try:
            yield b"data: {\"choices\": [{\"index\": 0, \"delta\": {}}]}\n\n"
            while not token.wait(0.02):
                yield b"data: {\"choices\": [{\"index\": 0, \"delta\": {}}]}\n\n"
            rec["ended"] = True
        finally:
            rec["closed"] = True
    return fake


def _door(lanes: int = 2, wait: float = 0.6):
    """server.app with fresh lanes; returns (server, admission, restore)."""
    import admission
    import server
    saved = (server.proxy.stream_body, server.accounts.identify,
             admission.MAIN_LANES, admission.WAIT_SECONDS,
             dict(admission._lanes), dict(admission._inflight),
             getattr(server, "SUPERSEDE_AFTER_S", None))
    admission._lanes.clear()
    admission._inflight.update(main=0, helper=0)
    admission.MAIN_LANES, admission.WAIT_SECONDS = lanes, wait
    server.accounts.identify = lambda h: ("acct-test", "")

    def restore():
        (server.proxy.stream_body, server.accounts.identify,
         admission.MAIN_LANES, admission.WAIT_SECONDS) = saved[:4]
        admission._lanes.clear()
        admission._lanes.update(saved[4])
        admission._inflight.update(saved[5])
        if saved[6] is not None:
            server.SUPERSEDE_AFTER_S = saved[6]
    return server, admission, restore


_TWIN_BODY = {"model": "yamadori", "stream": True,
              "messages": [{"role": "user", "content": "fix the game"},
                           {"role": "assistant", "content": "",
                            "tool_calls": [{"id": "c1", "type": "function",
                                            "function": {"name": "read_file",
                                                         "arguments": "{}"}}]},
                           {"role": "tool", "tool_call_id": "c1",
                            "content": "1|data:image/png;base64,iVBOR"}]}


def test_a_retry_takes_the_lanes_its_abandoned_twins_hold():
    """THE SELF-LOCK (Octopus v0e-V0-xhigh-1, #44): Hermes' first attempt
    and its stale-kill retry -- the SAME request -- held both main lanes
    (their client was gone; a relay kept both connections open), and the
    third attempt was refused 429 three times, which ended the run. A
    request identical to one in flight (same account, session token,
    messages and tools) is a retry: the attempts it replaces are cancelled
    and it is admitted. A different request is never touched."""
    import asyncio
    record: list = []
    server, admission, restore = _door()
    server.proxy.stream_body = _fake_stream_body(record)

    async def go():
        a = _Chat(server.app, _TWIN_BODY).start()
        await asyncio.wait_for(a.started.wait(), 2)
        b = _Chat(server.app, _TWIN_BODY).start()
        await asyncio.wait_for(b.started.wait(), 2)
        held = admission._inflight["main"]
        other = dict(_TWIN_BODY, messages=[{"role": "user",
                                            "content": "another task"}])
        c = _Chat(server.app, _TWIN_BODY).start()
        await asyncio.wait_for(c.started.wait(), 5)
        # A different conversation arriving now waits like anyone else and
        # cancels nobody.
        d = _Chat(server.app, other).start()
        await asyncio.sleep(0.3)
        d_ok = [r["token"].cancelled for r in record]
        for x in (a, b, c, d):
            x.hang_up.set()
        await asyncio.wait_for(asyncio.gather(
            *(x.task for x in (a, b, c, d)), return_exceptions=True), 5)
        return a, b, c, d, held, d_ok

    try:
        if hasattr(server, "SUPERSEDE_AFTER_S"):
            server.SUPERSEDE_AFTER_S = 3600.0   # only the full-lanes rule
        a, b, c, d, held, d_ok = asyncio.run(go())
        check(held == 2, "two attempts of one request hold both lanes",
              str(held))
        check(c.status == 200,
              "the third attempt is admitted, not refused 429 (the Octopus "
              "run ended on three 429s)", str(c.status))
        whys = [str(r["token"].why) for r in record[:2]]
        check(all(r["token"].cancelled for r in record[:2])
              and all("superseded" in w for w in whys),
              "the two abandoned attempts were cancelled as superseded",
              str(whys))
        check(d_ok is not None and len(d_ok) >= 3 and not d_ok[2],
              "the attempt that replaced them is not cancelled by a "
              "different request", str(d_ok))
        check(admission._inflight["main"] == 0,
              "every lane is given back at the end",
              str(admission._inflight))
    finally:
        restore()


def test_a_retry_after_a_while_replaces_its_twin_at_once():
    """With SUPERSEDE_AFTER_S elapsed, a retry cancels the attempt it
    repeats on ARRIVAL, not only when the lanes are full: an orphan left
    running beside its retry halves the retry's decode speed (#39: 1.2-3.4
    tok/s)."""
    import asyncio
    record: list = []
    server, admission, restore = _door()
    server.proxy.stream_body = _fake_stream_body(record)

    async def go():
        a = _Chat(server.app, _TWIN_BODY).start()
        await asyncio.wait_for(a.started.wait(), 2)
        await asyncio.sleep(0.1)
        b = _Chat(server.app, _TWIN_BODY).start()
        await asyncio.wait_for(b.started.wait(), 2)
        await asyncio.sleep(0.2)
        n = admission._inflight["main"]
        seen = [(r["token"].cancelled, r["token"].why) for r in record]
        for x in (a, b):
            x.hang_up.set()
        await asyncio.wait_for(asyncio.gather(
            a.task, b.task, return_exceptions=True), 5)
        return n, seen

    try:
        server.SUPERSEDE_AFTER_S = 0.05
        n, seen = asyncio.run(go())
        check(len(seen) == 2 and seen[0][0] and "superseded" in seen[0][1]
              and not seen[1][0],
              "the older attempt is cancelled, the retry runs", str(seen))
        check(n == 1, "one lane held, not two", str(n))
        check(admission._inflight["main"] == 0, "and none after",
              str(admission._inflight))
    finally:
        restore()


def test_a_stream_that_never_starts_gives_its_lane_back():
    """A LANE LEAK on the streamed path: the client hangs up in the instant
    between the first chunk and Starlette's first pull of the body, so
    Starlette cancels the body before it ever runs -- and the lane, released
    only in the body generator's `finally`, was never given back (a lane
    this server never gets back: after two, it admits nobody)."""
    import asyncio
    record: list = []
    server, admission, restore = _door()
    server.proxy.stream_body = _fake_stream_body(record)

    async def go():
        a = _Chat(server.app, _TWIN_BODY, first_watch_blocks=True)
        a.hang_up.set()                 # Starlette's listener sees it at once
        a.start()
        await asyncio.wait_for(asyncio.gather(a.task,
                                              return_exceptions=True), 5)
        return a.status

    try:
        status = asyncio.run(go())
        check(status in (None, 200), "the response had started or never "
              "began (no refusal)", str(status))
        check(len(record) == 1 and record[0]["token"].cancelled,
              "the turn's token is cancelled", str(record[:1]))
        check(admission._inflight["main"] == 0,
              "and the main lane is given back although the body never ran",
              str(admission._inflight))
        check(record and record[0]["closed"],
              "and the turn's generator is closed", str(record[:1]))
    finally:
        restore()


def test_a_stream_that_ends_without_a_finish_is_a_failure():
    """A DEAD MODEL SERVER IS NOT AN EMPTY ANSWER (proxy._post_events_raw; 2026-10-01: flash-next faulted, llama-swap
    answered 200 with an empty body and the client got finish "stop" with no content). A stream that ends with no
    finish chunk and nothing in hand is retried once, then raised as an upstream failure (503); one that ended with
    text in hand is reported as a dropped connection, never as a finished answer."""
    import streaming
    import urllib.request as _ur
    calls = []

    class _Body:
        def __init__(self, lines):
            self._lines = lines

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def __iter__(self):
            return iter(self._lines)
    saved = _ur.urlopen
    try:
        def empty(req, timeout=None):
            if getattr(req, "data", None) is not None:           # the generation's POST, not llama-swap's /running
                calls.append(1)
            return _Body([])
        _ur.urlopen = empty
        try:
            got = list(proxy._post_events_raw("/v1/chat/completions", {"model": "m", "messages": []}))
            check(False, "an empty 200 stream: retried once, then an upstream failure (503)", str(got)[:200])
        except streaming.UpstreamError as e:
            check(len(calls) == 2 and getattr(e, "upstream_status", None) == 503,
                  "an empty 200 stream: retried once, then an upstream failure (503)",
                  f"calls={len(calls)} status={getattr(e, 'upstream_status', None)}")

        def partial(req, timeout=None):
            return _Body([b'data: {"choices":[{"delta":{"content":"half an ans"}}]}\n'])
        _ur.urlopen = partial
        evs = list(proxy._post_events_raw("/v1/chat/completions", {"model": "m", "messages": []}))
        done = [v for k, v in evs if k == "done"]
        check(len(done) == 1 and done[0]["choices"][0]["finish_reason"] == "incomplete"
              and done[0]["_transport"].get("dropped_after") is not None,
              "text in hand and no finish chunk: reported as a dropped connection (finish 'incomplete'), "
              "never 'stop'", str(done)[:300])

        def good(req, timeout=None):
            return _Body([b'data: {"choices":[{"delta":{"content":"ok"}}]}\n',
                          b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n', b"data: [DONE]\n"])
        _ur.urlopen = good
        evs = list(proxy._post_events_raw("/v1/chat/completions", {"model": "m", "messages": []}))
        done = [v for k, v in evs if k == "done"]
        check(len(done) == 1 and done[0]["choices"][0]["finish_reason"] == "stop"
              and done[0]["choices"][0]["message"]["content"] == "ok",
              "a stream with its finish chunk is delivered as before")
    finally:
        _ur.urlopen = saved


def test_a_dead_model_server_mid_generation_is_an_error_not_an_answer():
    """THE OPERATOR'S RULE (2026-10-05): kill flash-next mid-generation and the client gets an ERROR, never an
    answer. The stream ended with reasoning (and text) in hand and no finish chunk: the chat wire used to deliver
    the fragment as content plus a note with finish_reason "incomplete" (a value no client knows), which a harness
    read as the model's reply and ended its turn on. It is now the error path's: raised from the turn, so
    stream_body's caller (server._serve_turn) sends the HTTP 503 before the first byte or ONE error event and
    [DONE] after it; `complete` raises the same."""
    import api_errors
    for what, rep in (("reasoning only", reply("", reasoning="let me think about this for a while")),
                      ("reasoning and text", reply("half an ans", reasoning="thinking done"))):
        reset([dict(rep, drop=True), dict(rep, drop=True)])
        ev = []
        for b in proxy.stream_body({"model": "yamadori", "stream": True,
                                    "messages": [{"role": "user", "content": "write the file"}]}, "yamadori"):
            ev.append(b)
        errs = [json.loads(b[6:])["error"] for b in ev if b.startswith(b'data: {"error"')]
        check(len(errs) == 1 and errs[0].get("code") == "model_unavailable" and errs[0].get("retryable") is True
              and ev[-1].strip() == b"data: [DONE]" and ev.index(next(b for b in ev if b.startswith(b'data: {"error"')))
              == len(ev) - 2,
              f"{what}: the stream ends with ONE error event (model_unavailable, retryable) then [DONE]",
              str(ev[-2:])[:300])
        check(not any(b"\"finish_reason\": \"" in b and b"\"finish_reason\": null" not in b for b in ev)
              and not any(b"connection to the model dropped" in b and b"choices" in b for b in ev),
              f"{what}: no chunk carries a finish_reason or the fragment as a note", str(ev[-3:])[:300])
        check(not answers(), f"{what}: nothing is recorded as an answer that was delivered", str(answers()))
        reset([dict(rep, drop=True), dict(rep, drop=True)])
        try:
            proxy.complete({"model": "yamadori", "messages": [{"role": "user", "content": "write the file"}]})
            check(False, f"{what}: the blocking turn raises too")
        except Exception as e:                                   # noqa: BLE001
            err = api_errors.of_exception(e)
            check(err.status == 503 and err.code == "model_unavailable",
                  f"{what}: the blocking turn raises too (HTTP 503 model_unavailable, Retry-After)",
                  f"{type(e).__name__}: {e}")


def test_a_killed_model_server_is_waited_for_not_answered_502():
    """LIVE 2026-10-05 (soak scenario k): flash-next killed; llama-swap answered the next request 502 (dial tcp
    refused) for a moment BEFORE it noticed the process had exited, the proxy's one immediate retry met the same 502
    and the client got a 502 server_error. The retry now waits for llama-swap to notice (GET /running no longer
    lists the model as ready), then asks again -- and a 502 that survives is a 503 model_unavailable (retryable)."""
    import api_errors
    import time as _t
    state = {"n": 0, "running_polls": 0, "dead": True}

    class H(BaseHTTPRequestHandler):
        def do_GET(self):                                        # noqa: N802
            if self.path == "/running":
                state["running_polls"] += 1
                if state["running_polls"] >= 3:                  # llama-swap notices the exit on the 3rd look
                    state["dead"] = False
                rows = [{"model": "flash-next", "state": "ready"}] if state["dead"] else []
                data = json.dumps({"running": rows}).encode()
                self.send_response(200)
            else:                                                # /upstream/<m>/health: refused while dead
                data = b""
                self.send_response(502 if state["dead"] else 200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):                                       # noqa: N802
            self.rfile.read(int(self.headers["Content-Length"]))
            state["n"] += 1
            if state["dead"]:
                self.send_response(502)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            data = _sse(reply("back up"))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):                               # noqa: D102
            pass
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    saved = proxy.UPSTREAM
    proxy.UPSTREAM = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        t0 = _t.time()
        evs = list(proxy._post_events_raw("/v1/chat/completions", {"model": "flash-next", "messages": []}))
        done = [v for k, v in evs if k == "done"]
        check(len(done) == 1 and done[0]["choices"][0]["message"]["content"] == "back up",
              "a 502 with nothing in hand: the retry waits for llama-swap to notice and is then served",
              f"{state} {str(done)[:200]}")
        check(state["running_polls"] >= 3 and state["n"] == 2,
              "it polled /running until the model was no longer ready (not an immediate second try)",
              str(state))
        # the helper alone: a model that answers its health goes at once; a stopped one at once
        state.update(dead=False, running_polls=0)
        w = proxy._await_swap_notice("flash-next")
        check(w["why"] == "llama-swap no longer lists it as ready" and w["waited_s"] < 2, "a model not listed: no wait", str(w))
        # a 502 that survives both tries is a retryable 503, not a 502 server_error
        state.update(dead=True, running_polls=-10 ** 6)          # llama-swap never notices
        saved_wait = proxy._await_swap_notice
        proxy._await_swap_notice = lambda model, **k: {"waited_s": 0.0, "why": "test"}
        try:
            list(proxy._post_events_raw("/v1/chat/completions", {"model": "flash-next", "messages": []}))
            check(False, "a 502 that survives the retry is raised")
        except Exception as e:                                   # noqa: BLE001
            err = api_errors.of_exception(e)
            check(err.status == 503 and err.code == "model_unavailable"
                  and (err.headers or {}).get("Retry-After"),
                  "a 502 that survives the retry reaches the client as 503 model_unavailable with Retry-After",
                  f"{type(e).__name__}: {err.status} {err.code}")
        finally:
            proxy._await_swap_notice = saved_wait
        check(api_errors.of_upstream(500, {"message": "CUDA error", "type": "server_error"}).status == 502,
              "a real server_error (500) still stays a 502")
    finally:
        proxy.UPSTREAM = saved
        srv.shutdown()


def test_the_wait_is_heard():
    """THE WAIT IS HEARD (proxy._TurnPump; the operator's live session, 2026-10-01 ET: VS Copilot hung up at 180 s
    with no byte while Flash-Next loaded). A turn that produces nothing for HEARTBEAT seconds sends heartbeats
    (empty deltas) until its first event; a refusal inside the first HEARTBEAT seconds is still RAISED before any
    byte (E1); one after a heartbeat is the committed stream's one error event and [DONE]."""
    import api_errors
    import cancel
    import time
    saved_turn, saved_beat = proxy._run_turn, proxy.HEARTBEAT
    proxy.HEARTBEAT = 0.1

    def slow_turn(body, streamed):
        cancel.current().preflight_done = True      # as _run_turn does after the window check
        time.sleep(0.45)                     # the model load / a cold prefill
        yield ("content", "hi")
        return {"choices": [{"finish_reason": "stop"}], "x_yamadori": {}}

    def fast_refusal(body, streamed):
        raise api_errors.ApiError(503, "at capacity", code="conversation_at_capacity")
        yield                                # pragma: no cover

    def slow_refusal(body, streamed):
        time.sleep(0.35)                     # a long window count, BEFORE the pre-flight is over
        raise api_errors.ApiError(400, "too long", code="context_length_exceeded")
        yield                                # pragma: no cover

    def slow_swap(body, streamed):
        cancel.current().card_wait = True    # as max_mode.wait_ready does while the model loads
        time.sleep(0.45)
        cancel.current().card_wait = False
        time.sleep(0.35)                     # the pre-flight AFTER the swap (the decider primes): still heard
        cancel.current().preflight_done = True
        yield ("content", "hi")
        return {"choices": [{"finish_reason": "stop"}], "x_yamadori": {}}

    def late_failure(body, streamed):
        cancel.current().preflight_done = True
        time.sleep(0.35)
        raise RuntimeError("the model server went away")
        yield                                # pragma: no cover

    def deltas(chunks):
        out = []
        for c in chunks:
            t = c.decode("utf-8", "replace").strip()
            if t.startswith("data: ") and t != "data: [DONE]":
                out.append(json.loads(t[6:]))
        return out
    try:
        proxy._run_turn = slow_turn
        chunks = list(proxy.stream_body({"messages": []}))
        ds = deltas(chunks)
        first_content = next((i for i, d in enumerate(ds)
                              if ((d.get("choices") or [{}])[0].get("delta") or {}).get("content")), None)
        beats = [d for d in ds[:first_content or 0]
                 if (d.get("choices") or [{}])[0].get("delta") == {} and not (d.get("choices") or [{}])[0].get("finish_reason")]
        check(first_content is not None and len(beats) >= 2 and len(beats) == first_content,
              "a turn silent for several HEARTBEATs: heartbeats (empty deltas) go out until its first event",
              json.dumps({"before_content": first_content, "beats": len(beats)}))
        check(ds and (ds[-1].get("choices") or [{}])[0].get("finish_reason") == "stop"
              and chunks[-1].strip() == b"data: [DONE]" and "".join(
                  ((d.get("choices") or [{}])[0].get("delta") or {}).get("content") or "" for d in ds) == "hi",
              "then the turn's own events, the finish chunk and [DONE]: the content is unchanged")
        proxy._run_turn = fast_refusal
        try:
            got = list(proxy.stream_body({"messages": []}))
            check(False, "a refusal inside the first HEARTBEAT is raised before any byte (E1)", str(got)[:200])
        except api_errors.ApiError as e:
            check(e.status == 503, "a refusal inside the first HEARTBEAT is raised before any byte (E1)", str(e))
        proxy._run_turn = slow_swap
        ds = deltas(list(proxy.stream_body({"messages": []})))
        nb = sum(1 for d in ds if (d.get("choices") or [{}])[0].get("delta") == {}
                 and not (d.get("choices") or [{}])[0].get("finish_reason"))
        check(nb >= 6, "a model swap BEFORE the pre-flight (card_wait) is heard, and so is the pre-flight after it "
              "(the stream is committed): heartbeats throughout", str(nb))
        proxy._run_turn = slow_refusal
        try:
            got = list(proxy.stream_body({"messages": []}))
            check(False, "a refusal during a SLOW pre-flight (several HEARTBEATs) is still raised before any byte",
                  str(got)[:200])
        except api_errors.ApiError as e:
            check(e.status == 400 and e.code == "context_length_exceeded",
                  "a refusal during a SLOW pre-flight (several HEARTBEATs) is still raised before any byte", str(e))
        proxy._run_turn = late_failure
        chunks = list(proxy.stream_body({"messages": []}))
        text = b"".join(chunks).decode("utf-8", "replace")
        check(chunks and chunks[-1].strip() == b"data: [DONE]" and text.count('"error"') == 1
              and '"content"' not in text,
              "a failure after a heartbeat: ONE error event then [DONE], never assistant content", text[-300:])
    finally:
        proxy._run_turn, proxy.HEARTBEAT = saved_turn, saved_beat


def main() -> int:
    for fn in (test_fake_upstream_is_alive,
               test_an_answer_without_tools_is_generated_once,
               test_a_tool_hop_then_the_answer_is_two_generations_not_three,
               test_no_reasoning_after_the_first_content,
               test_an_image_is_content_when_made_and_later_reasoning_a_heartbeat,
               test_reasoning_continuity,
               test_internal_hops_keep_their_reasoning,
               test_the_breaker_lands_with_tools_withdrawn,
               test_complete_lands_through_the_same_helper,
               test_the_tool_turn_cap_lands_and_is_recorded,
               test_an_uncapped_turn_records_the_cap_unhit,
               test_a_client_tool_call_is_forwarded_whole,
               test_a_length_finish_is_finish_reason_length_only,
               test_truncation_marker_at_the_breaker,
               test_x_yamadori_rides_the_final_chunk_and_matches_complete,
               test_the_compaction_store_keeps_the_ledgers_rendering,
               test_a_prefill_survives_an_image_hop,
               test_a_disconnect_during_a_silent_prefill_stops_main,
               test_the_server_cancels_the_token_on_disconnect,
               test_a_long_client_tool_call_keeps_the_stream_alive,
               test_a_retry_takes_the_lanes_its_abandoned_twins_hold,
               test_a_retry_after_a_while_replaces_its_twin_at_once,
               test_a_stream_that_never_starts_gives_its_lane_back,
               test_a_dead_model_server_mid_generation_is_an_error_not_an_answer, test_a_killed_model_server_is_waited_for_not_answered_502, test_the_wait_is_heard,
               test_a_stream_that_ends_without_a_finish_is_a_failure):
        print(f"\n--- {fn.__name__} ---")
        n0 = len(_results)
        try:
            fn()
        except Exception:                                        # noqa: BLE001
            check(False, f"{fn.__name__} itself raised",
                  traceback.format_exc().strip().split("\n")[-1])
        for ok, name, detail in _results[n0:]:
            print(("  pass  " if ok else "  FAIL  ") + name
                  + (f"   <- {detail}" if not ok and detail else ""))
    _srv.shutdown()
    passed = sum(1 for ok, _, _ in _results if ok)
    total = len(_results)
    print(f"\n{'=' * 70}\n  {passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
