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
  4. A `length` finish ends the stream with the same notice `complete()`
     writes, as content, and the reasoning never appears as content.
  5. A tool result over the 6000-char breaker carries a marker naming the
     size withheld and the call that fetches the rest.

THE FAKE UPSTREAM

A real `http.server` on 127.0.0.1:<ephemeral>, so the proxy's own reader
(`_post_events`) and its urllib transport run unmodified. Each POST pops the
next scripted reply and records the request body; a request with
`stream: true` gets SSE, anything else gets the assembled JSON. Counting the
recorded requests is what "one generation" means here.

Everything that would touch shared state is stubbed or redirected to temp
files BEFORE import: the corpus (training data for the decision model --
test runs have polluted it before), nebari, rings, and the package store.
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
import shomen as _shomen  # noqa: E402

# THE EVIDENCE (shomen): a verified fact carries the lines it cites, read
# from the source the verifier can read. The fixtures' cited files live here.
_SRC = os.path.join(_TMP, "held_src")
for _rel in ("src/a.ts", "src/loader.ts", "lib/conf.yaml", "src/game.ts",
             "src/geometries/LatheGeometry.js"):
    os.makedirs(os.path.dirname(os.path.join(_SRC, _rel)), exist_ok=True)
    with open(os.path.join(_SRC, _rel), "w", encoding="utf-8") as _f:
        _f.write("\n".join(f"line {i} of {_rel}" for i in range(1, 81)) + "\n")
_shomen.SOURCE_RESOLVER = _shomen.directory_resolver(_SRC, "fixture@1.0.0")

# Every second-brain job draws a concept seed (proxy.ledger_seed). Pinned
# here, so no test loads the 420 MB embedding matrix and every seed line is
# predictable.
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


class _Up(BaseHTTPRequestHandler):
    def do_POST(self):                                           # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
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
# The second brain's door (mcp/model.py) goes to the same fake: no
# offline test may reach the live model server.
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


# What the selection engine decided, as the stubbed prepare() reports it.
# Tests that exercise deep thinking or fan-out set it; reset() clears it.
SELECTION: dict = {}
# Other prepare() fields a test sets: _tool_code / _fixup / _repair, _tier.
PREP: dict = {}

# The tool main runs itself in these tests: generate_image, the one kind of
# tool left on main (our code tools are the second brain's since 2026-09-24).
IMG = "generate_image"


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
                 "_tools_gate": {"offer": True, "situation": "TEST",
                                 "because": "stub"},
                 "_selection": dict({"hints": False, "investigate": False,
                                     "fanout_n": 1, "because": {},
                                     "signals": {}}, **SELECTION)},
                **json.loads(json.dumps(PREP)))


def _run_our_tool(name, args, db, root=None, turn=None, state=None):
    _ran.append((name, args))
    return TOOL_OUTPUT.get(name, f"{name}: 1 result\nsrc/a.ts:1")


_REAL_RUN_OUR_TOOL = proxy.run_our_tool
proxy.corpus = _Corpus
proxy.prepare = _prepare
proxy.run_our_tool = _run_our_tool
proxy.resolve_repo = lambda messages, ip="": (None, False, "none")
proxy.session_context = lambda messages, account="", session="", **k: (
    "k", {"_key": "k"})
proxy.PREAMBLE = False


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
               tools: list | None = None) -> list[dict]:
    body = {"model": "yamadori", "stream": True,
            "messages": [{"role": "user", "content": user}]}
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
           reply("src/geometries/LatheGeometry.js:12", reasoning="found it")])
    ev = run_stream()
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
    last = [e for e in ev if e.get("usage")]
    check(last and last[-1]["usage"].get("completion_tokens") == 20
          and last[-1]["usage"].get("hops") == 2,
          "usage is summed over both generations",
          json.dumps(last[-1]["usage"]) if last else "no usage chunk")


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

def test_reasoning_continuity():
    """PAST REASONING PASSES THROUGH (design change, coordinator/operator
    2026-09-24; it replaces the ledger's reasoning restore). Hermes strips
    reasoning_content from every replayed assistant turn for a provider that
    does not require the echo (agent/message_sanitization.py
    apply_reasoning_content_policy), and the reference run (plain
    llama-server + Hermes + Bonsai 2, 5 h, 125k context) ran that way and
    kept the thread; restoring it cost 6-10k tokens of context per step. So
    the proxy neither restores reasoning a client dropped nor strips what it
    sent: what the client sends is what the model sees."""
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
    # What the proxy delivered for that call turn is recorded (content, hops)
    # -- and no reasoning, whatever it carried.
    proxy.ledger_record_turn("acct", "s1", dict(hist[3]))
    out = proxy.scope_reasoning(hist, "acct")
    check(out[1].get("reasoning_content") == "old thought"
          and out[3].get("reasoning_content") == "plan: read game.js",
          "an echoing client's reasoning passes through unchanged")
    bare = [dict(m) for m in hist]
    for m in bare:
        m.pop("reasoning_content", None)
    out = proxy.scope_reasoning(bare, "acct")
    check(all(not m.get("reasoning_content") for m in out),
          "a client that dropped its reasoning gets none back: the ledger "
          "records and restores no reasoning")
    check(proxy.nebari.ledger_get("acct", "call:c1", "reasoning") is None,
          "and nothing of the kind was written")
    shown = [dict(m) for m in hist]
    shown[3]["reasoning_content"] = ("  find_by_pattern x\ndeep thinking "
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
        n["calls"] += 1
        return n["calls"] >= 3
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
    check("context" in deltas(ev, "content") or "breaker" in deltas(ev, "content"),
          "an answerless landing says so instead of ending empty",
          repr(deltas(ev, "content")))


def test_complete_lands_through_the_same_helper():
    prev = proxy.context_full
    n = {"calls": 0}

    def full_on_second(*a, **k):
        n["calls"] += 1
        return n["calls"] >= 2
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


def test_deep_thinking_honours_an_effort_override():
    # The domain benchmark sends body tier `max` with X-Yamadori-Features
    # {"effort": "medium"}. The main context thinks at medium; deep thinking
    # must too, not at the `max` tier name's xhigh.
    import shomen
    sent: list[dict] = []
    real = shomen.model.post if hasattr(shomen, "model") else None
    import model
    saved = model.post

    def fake(body, timeout=None):
        sent.append(body)
        return {"choices": [{"message": {"content": "It is in a.js."},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
    model.post = fake
    try:
        shomen._investigate("q", [], lambda fn, a: "", "", None, None, None,
                            tier="max", effort="medium")
        with_override = sent[-1].get("reasoning_effort") if sent else None
        shomen._investigate("q", [], lambda fn, a: "", "", None, None, None,
                            tier="max")
        without = sent[-1].get("reasoning_effort") if sent else None
    finally:
        model.post = saved
    del real
    check(with_override == "medium",
          "deep thinking under tier max with an effort override of medium "
          "sends medium", str(with_override))
    check(without == "xhigh", "and without an override, tier max sends xhigh",
          str(without))
    model.post = fake
    try:
        rec = shomen.investigate("q", [], lambda fn, a: "", tier="max",
                                 effort="medium")
        rec2 = shomen.investigate("q", [], lambda fn, a: "", tier="xhigh")
    finally:
        model.post = saved
    check(rec.get("effort") == "medium" and rec2.get("effort") == "medium"
          and shomen.investigate.__defaults__ is not None,
          "the investigate result records the effort sent (override, or the "
          "tier's own)", f"{rec.get('effort')} {rec2.get('effort')}")


def test_deep_thinking_obeys_the_tool_turn_cap():
    import shomen
    sent: list[dict] = []
    answer_at = {"n": None}

    def fake_post(path, payload, timeout=3600):
        sent.append(payload)
        if answer_at["n"] is not None and len(sent) >= answer_at["n"]:
            return {"choices": [{"message": {"content": "It is in a.js."},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
        return {"choices": [{"message": {"content": "", "tool_calls": [
                    {"id": f"t{len(sent)}", "type": "function", "function": {
                        "name": "find_by_pattern",
                        "arguments": "{\"pattern\": \"x\"}"}}]},
                    "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
    saved = shomen._post
    shomen._post = fake_post
    tool = [{"type": "function", "function": {"name": "find_by_pattern"}}]
    try:
        # Tier xhigh: 10 tool turns, then the tool-less 11th call answers.
        answer_at["n"] = 11
        res = shomen._investigate("q", tool, lambda fn, args: "nothing", "",
                                  None, None, None, tier="xhigh")
        n_ok, land_ok = len(sent), not sent[-1].get("tools")
        stored = shomen._TRACES.get(res.get("handle") or "", {})
        caps = [t for t in stored.get("trace") or []
                if t.get("tool") == "(turn cap)"]
        # Never answers, even at the landing: must still END, as a failure.
        sent.clear()
        answer_at["n"] = None
        res2 = shomen._investigate("q", tool, lambda fn, args: "nothing", "",
                                   None, None, None, tier="xhigh")
        n_fail = len(sent)
        # Tier max: its own run gets 20.
        sent.clear()
        answer_at["n"] = 21
        shomen._investigate("q", tool, lambda fn, args: "nothing", "",
                            None, None, None, tier="max")
        n_max, land_max = len(sent), not sent[-1].get("tools")
    finally:
        shomen._post = saved
    check(n_ok == 11 and land_ok,
          "deep thinking: 10 tool turns, then an 11th request with no tools",
          f"{n_ok} requests, landing tools={land_ok}")
    check(len(caps) == 1, "the trace records the cap once", str(caps))
    check(n_fail == 11 and res2.get("ok") is False
          and "cap" in str(res2.get("error") or res2),
          "a landing that still asks for a tool ends as a failure, not a hang",
          f"{n_fail} requests; {str(res2)[:200]}")
    check(n_max == 21 and land_max,
          "deep thinking at max: its own 20 tool turns, then a 21st with no "
          "tools", f"{n_max} requests, landing tools={land_max}")


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


def test_a_length_finish_streams_the_budget_notice():
    usage = {"prompt_tokens": 50, "completion_tokens": 10240,
             "total_tokens": 10290}
    reset([reply("", reasoning="still deliberating about the answer",
                 finish="length", usage=usage)])
    ev = run_stream()
    content = deltas(ev, "content")
    want = proxy._budget_notice("length", "", dict(usage, hops=1))
    check(content == want, "empty answer: the whole content IS the notice",
          repr(content))
    check("finish_reason=length" in content and "budget event" in content,
          "the notice names the event")
    check("deliberating" not in content,
          "reasoning is never passed off as the answer")
    check(finish_of(ev) == "length", "finish_reason stays length",
          str(finish_of(ev)))
    check(len(_seen) == 1, "no regeneration after a length finish",
          f"{len(_seen)}")

    reset([reply("partial answer", finish="length", usage=usage)])
    ev = run_stream()
    content = deltas(ev, "content")
    blocking = proxy._budget_note("length", "partial answer", dict(usage, hops=1))
    check(content == blocking,
          "partial answer: streamed text equals what complete() returns",
          f"{content!r} vs {blocking!r}")
    check(answers() and "cut off" in answers()[0][2],
          "the logged answer includes the notice", str(answers()))


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


_fan_calls: list[int] = []


def _fake_fanout(results: list[dict]):
    """fanout.run, faked: the second brain's candidates are `results`; the
    original is candidate A, and the real consensus() records the vote."""
    def run(payload, original=None, n=3, timeout=3600, lane_timeout=None,
            seed_for=None):
        _fan_calls.append(n)
        cands = [dict(r) for r in results]
        for r in cands:
            if seed_for is not None and not r.get("seed"):
                r["seed"] = (seed_for("alternative", "") or {}).get("word")
        v = proxy.fanout.consensus([dict(original, variant="original")]
                                   + cands)
        v.update(mode="sequential", steps=1 + len(results),
                 stop_reason="prose: recorded only")
        return v
    return run


def test_fan_out_runs_on_the_streamed_path_too():
    saved = proxy.fanout.run
    try:
        # No paths named: nothing was voted on, so nothing is "unsettled".
        proxy.fanout.run = _fake_fanout(
            [{"variant": "direct", "seed": "lantern", "content": "2 3 5 7"},
             {"variant": "terse", "seed": "harbour", "content": "2, 3, 5, 7"}])
        reset([reply("2\n3\n5\n7")])
        SELECTION["fanout_n"] = 2
        _fan_calls.clear()
        ev = run_stream()
        check(_fan_calls == [2], "the streamed answer is fanned out when the "
              "engine chose N=2", str(_fan_calls))
        check("unsettled" not in deltas(ev, "content")
              and "agreed" not in deltas(ev, "content"),
              "no path votes cast: no dissent note",
              repr(deltas(ev, "content")))
        fx = (final_chunk(ev).get("x_yamadori") or {}).get("fanout") or {}
        # CHANGED 2026-09-23: the original answer is a candidate too, listed
        # first. With no votes cast the tie goes to the shortest, and the
        # original ties with `direct` at 7 characters, so it keeps its place.
        # A prose winner is never delivered in place of the original.
        check(fx.get("agreement") is None
              and fx.get("seeds") == ["lantern", "harbour"]
              and fx.get("winner") == "original"
              and fx.get("appended") is False and fx.get("replaced") is False,
              "x_yamadori.fanout: n, seeds, agreement null, the winner kept",
              json.dumps(fx))

        # Real disagreement over named files IS reported, after the answer.
        proxy.fanout.run = _fake_fanout(
            [{"variant": "direct", "content": "see src/a.ts"},
             {"variant": "evidence", "content": "see src/b.ts"}])
        reset([reply("see src/a.ts"),
               reply(": src/b.ts only re-exports it.")])
        SELECTION["fanout_n"] = 2
        ev = run_stream()
        c = deltas(ev, "content")
        check(c.startswith("see src/a.ts") and "unsettled" in c,
              "observed disagreement follows the streamed answer as a note",
              repr(c))
        # THE FOLD-BACK replaces the weigh turn (operator, 2026-09-24): B's
        # differing point is PREFILLED after main's own answer, in main's
        # own turn, and main continues from it. No user turn is added.
        pre = (_seen[1]["messages"][-1] if len(_seen) > 1 else {})
        check(len(_seen) == 2 and pre.get("role") == "assistant"
              and pre.get("content", "").startswith("see src/a.ts\n\n"
                                                    "Today I was inspired by")
              and "Compared two approaches" in pre.get("content", "")
              and "src/b.ts" in pre.get("content", "")
              and pre.get("content", "").endswith("Weighing them")
              and not any(m.get("role") == "user" and "second answer" in
                          str(m.get("content")) for m in
                          _seen[1]["messages"]),
              "the answer, then ONE continuation of main's own turn whose "
              "prefill is the answer plus the fold-back; no weigh turn",
              json.dumps(pre)[:400])
        check("Weighing them: src/b.ts only re-exports it." in c
              and c.count("see src/a.ts") == 1,
              "the stream carries the fold-back and the continuation once, "
              "never the re-sent prefill", repr(c)[:400])
        check(not hasattr(proxy, "_weigh_alternative"),
              "the weigh turn is gone from the proxy")

        reset([reply("see src/a.ts")])
        _fan_calls.clear()
        run_stream()
        check(not _fan_calls, "N=1: no fan-out at all", str(_fan_calls))
    finally:
        proxy.fanout.run = saved


def test_a_repeated_opening_never_reaches_the_client():
    """The guard on the streamed path: main repeats the opening as a bold
    line inside its held content; the client never receives the copy, the
    delivered turn matches what it was streamed, and x_yamadori counts it."""
    saved = proxy.shomen.investigate

    def fake(question, tools, run_tool, context="", hops=8, on_think=None,
             tier="max", effort=None, seed=None, mode="investigate"):
        run_tool("find_definition_opt", {"symbol": "LatheGeometry"})
        return {"ok": True, "finding": "src/geometries/LatheGeometry.js:12",
                "hops": 1, "handle": "hy", "cited": ["x"], "unsupported": [],
                "helper_tokens": 1, "seconds": 0.1}
    try:
        proxy.shomen.investigate = fake
        reset([reply(" it is defined in one file.\n\n**After thinking "
                     "deeply,**\n\nLatheGeometry lives in "
                     "src/geometries/LatheGeometry.js.")])
        SELECTION["investigate"] = True
        ev = run_stream()
        c = deltas(ev, "content")
        x = final_chunk(ev).get("x_yamadori") or {}
        fa = x.get("fold_back_answer") or {}
        check(c.count("After thinking deeply") == 1
              and "one file.\n\nLatheGeometry lives" in c
              and fa.get("repeated_opening_removed") == 1,
              "streamed: the bold repeat of the opening is removed before it "
              "reaches the client, and counted", json.dumps(
                  {"content": c, "fold_back_answer": fa})[:500])
    finally:
        proxy.shomen.investigate = saved
        SELECTION["investigate"] = False


def test_deep_thinking_runs_on_the_streamed_path():
    saved = proxy.shomen.investigate
    got: list = []

    def fake(question, tools, run_tool, context="", hops=8, on_think=None,
             tier="max", effort=None, seed=None, mode="investigate"):
        got.append([t["function"]["name"] for t in tools])
        got.append(seed)
        run_tool("find_definition_opt", {"symbol": "LatheGeometry"})
        if on_think:
            on_think("thinking about lathes")
        return {"ok": True, "finding": "src/geometries/LatheGeometry.js:12",
                "hops": 1, "handle": "hx", "cited": ["x"], "unsupported": [],
                "helper_tokens": 1, "seconds": 0.1}
    try:
        proxy.shomen.investigate = fake
        reset([reply("It is in src/geometries/LatheGeometry.js.")])
        SELECTION["investigate"] = True
        ev = run_stream()
        r = deltas(ev, "reasoning_content")
        check(got and "delegate_investigation" not in got[0]
              and "find_by_meaning" in got[0],
              "the investigator gets deep_thinking_tools()", str(got)[:200])
        check("thinking about lathes" in r and "LatheGeometry" in r,
              "its thinking and its searches stream as reasoning", repr(r))
        check("thinking about lathes" not in deltas(ev, "content"),
              "and never as content")
        pre = _seen[0]["messages"][-1] if _seen else {}
        check(pre.get("role") == "assistant"
              and pre.get("reasoning_content", "").startswith(
                  proxy.FINDINGS_HEAD)
              and "LatheGeometry.js:12" in pre.get("reasoning_content", "")
              and pre.get("content") == "Today I was inspired by cedar. "
                                        "After thinking deeply,"
              and not any(m.get("role") == "user" and proxy.FINDINGS_HEAD
                          in str(m.get("content")) for m in
                          _seen[0]["messages"]),
              "the hand-off is PREFILLED as main's reasoning, and the "
              "visible answer opens with the seed line and the phrase",
              json.dumps(pre)[:300])
        check(got[1] == SEED, "the investigate job got the ledger's seed",
              str(got[1]))
        check(deltas(ev, "content").startswith(
                  "Today I was inspired by cedar. After thinking deeply,")
              and "LatheGeometry.js." in deltas(ev, "content"),
              "the client receives the opening and main's continuation",
              repr(deltas(ev, "content"))[:200])
        inv = (final_chunk(ev).get("x_yamadori") or {}).get("investigate") or {}
        check(inv.get("ran") and inv.get("injected") and inv.get("handle") == "hx"
              and inv.get("into") == "prefill",
              "x_yamadori.investigate on the final chunk", json.dumps(inv))
        fb = (final_chunk(ev).get("x_yamadori") or {}).get("fold_back") or []
        # `trigger` (Phase 0.6, mcp/deep.py) names what ran it; a header
        # forced this one, recorded under the selection record.
        check([{k: v for k, v in f.items() if k != "trigger"} for f in fb]
              == [{"job": "investigate", "phrase": "investigate",
                   "into": "prefill", "seeds": ["cedar"]}]
              and all("trigger" in f for f in fb),
              "x_yamadori.fold_back names the job, phrase and seed",
              json.dumps(fb))

        got.clear()
        reset([reply("answer")])
        run_stream()
        check(not got, "not selected: not run", str(got))
    finally:
        proxy.shomen.investigate = saved


# --------------------------------------------------------------------------
# THE DEEP-THINKING HAND-OFF (operator decision, 2026-09-23): the second
# brain's work always crosses back as a distillation plus facts. The helper
# is shomen._post, faked; the main model is the fake upstream above; the
# proxy's tools are the stub `_run_our_tool`, whose result names src/a.ts.
# --------------------------------------------------------------------------
Q_LATHE = "How does LatheGeometry build its points from the profile?"
HANDOFF_READ = ("FACTS\n"
                "- LatheGeometry sweeps the profile around the Y axis. "
                "src/a.ts:1\n"
                "- The segment count defaults to 12.\n"
                "SEARCHED, FOUND NOTHING\n- none\n"
                "OPEN QUESTIONS\n- Whether phiLength wraps past 2*PI.\n"
                "NEXT STEP\n- Read src/a.ts:1-40.\n")
HANDOFF_UNREAD = ("FACTS\n- A lathe sweeps a profile around an axis.\n"
                  "SEARCHED, FOUND NOTHING\n- none\n"
                  "OPEN QUESTIONS\n- none\n"
                  "NEXT STEP\n- Search for LatheGeometry.\n")
HANDOFF_KEYS = {"facts", "unverified", "searched_empty", "open", "chars",
                "machine_built"}
_helper_sent: list[dict] = []


def _helper_tool_call(n: int) -> dict:
    return {"content": "", "tool_calls": [
        {"id": f"t{n}", "type": "function", "function": {
            "name": "find_by_pattern", "arguments": "{\"pattern\": \"lathe\"}"}}]}


def _think(helper: list[dict], tier: str = "xhigh"):
    """proxy._deep_thinking with a scripted helper. Returns (record, what
    crossed -- the hand-off prefilled as main's reasoning, as a one-item
    list -- and the helper request bodies)."""
    import shomen
    sent: list[dict] = []

    def post(path, payload, timeout=3600):
        sent.append(json.loads(json.dumps(payload)))
        m = helper.pop(0) if helper else {"content": ""}
        return {"choices": [{"message": m, "finish_reason": (
                    "tool_calls" if m.get("tool_calls") else "stop")}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
    saved = shomen._post
    shomen._post = post
    try:
        payload = {"_selection": {"investigate": True}, "messages": [],
                   "_tier": {"name": tier}}
        rec = proxy._drain(proxy._deep_thinking(
            payload, [{"role": "user", "content": Q_LATHE}], None, None,
            None))
    finally:
        shomen._post = saved
    pre = payload.get("_prefill") or {}
    crossed = [pre["reasoning_content"]] if pre.get("reasoning_content") \
        else []
    check(not payload["messages"],
          "deep thinking adds no turn to main's conversation")
    return rec or {}, crossed, sent


def test_a_zero_search_hand_off_crosses_labelled():
    reset([])
    rec, crossed, sent = _think([{"content": HANDOFF_UNREAD}])
    check(len(sent) == 1 and rec.get("searches") == 0,
          "the helper answered without searching", json.dumps(rec)[:300])
    check(len(crossed) == 1 and crossed[0].startswith(proxy.REASONING_HEAD)
          and "no sources checked" in proxy.REASONING_HEAD
          and proxy.FINDINGS_HEAD not in crossed[0],
          "0 searches: the hand-off CROSSES, under 'reasoning, no sources "
          "checked' -- not dropped, and not framed as source",
          (crossed[0] if crossed else "")[:300])
    check(crossed and "A lathe sweeps a profile around an axis. "
          "(reasoning, not checked against source)" in crossed[0],
          "its fact is labelled reasoning, per line",
          (crossed[0] if crossed else "")[:400])
    h = rec.get("handoff") or {}
    check(rec.get("injected") and h.get("facts") == 1
          and h.get("unverified") == 1 and h.get("machine_built") is False,
          "x_yamadori.investigate: injected, handoff counts",
          json.dumps(rec)[:400])


def test_the_turn_cap_landing_hands_off():
    import shomen
    reset([])
    rec, crossed, sent = _think([_helper_tool_call(i) for i in range(10)]
                                + [{"content": HANDOFF_READ}])
    landing = sent[-1] if sent else {}
    last_user = next((m.get("content") for m in reversed(
        landing.get("messages") or []) if m.get("role") == "user"), "")
    check(len(sent) == 11 and not landing.get("tools")
          and last_user == shomen.LANDING,
          "10 tool turns, then a tool-less landing whose prompt REQUIRES "
          "the four-section hand-off", f"{len(sent)} {last_user[:200]}")
    check(all(s in shomen.LANDING for s in
              ("FACTS", "SEARCHED, FOUND NOTHING", "OPEN QUESTIONS",
               "NEXT STEP", "your own words")),
          "the landing names every section and asks for the helper's own "
          "words, not pasted tool output", shomen.LANDING)
    h = rec.get("handoff") or {}
    check(crossed and crossed[0].startswith(proxy.FINDINGS_HEAD)
          and rec.get("searches") == 10 and h.get("machine_built") is False,
          "it crosses under FINDINGS_HEAD: 10 searches' work is handed back",
          json.dumps(rec)[:400])
    check(h.get("facts") == 2 and h.get("verified") == 1
          and h.get("unverified") == 1 and h.get("open") == 1,
          "the fact citing a retrieved path is verified; the uncited one is "
          "labelled", json.dumps(h))
    raw = "find_by_pattern: 1 result"
    check(crossed and raw not in crossed[0],
          "no raw tool output crosses", (crossed[0] if crossed else "")[:300])


def test_a_helper_that_returns_nothing_still_hands_off():
    import shomen
    # 1. The landing asks for a tool again: the run fails, and the trace is
    #    handed back, machine-built.
    reset([])
    rec, crossed, sent = _think([_helper_tool_call(i) for i in range(11)])
    h = rec.get("handoff") or {}
    body = crossed[0] if crossed else ""
    check(len(sent) == 11 and rec.get("ok") is False
          and h.get("machine_built") is True and rec.get("injected"),
          "a landing that still asks for a tool: a machine-built hand-off "
          "crosses anyway", json.dumps(rec)[:400])
    check(body.startswith(proxy.FINDINGS_HEAD)
          and "[machine-built hand-off" in body
          and 'find_by_pattern "lathe" returned' in body
          and "naming 1 path(s)" in body and "src/a.ts" in body
          and "Not answered: " + Q_LATHE in body,
          "it lists the queries run, their hit counts, the paths retrieved "
          "and the unanswered question", body[:700])
    # 2. Empty content, no search.
    reset([])
    rec, crossed, _ = _think([{"content": ""}])
    check(crossed and "[machine-built hand-off" in crossed[0]
          and crossed[0].startswith(proxy.REASONING_HEAD)
          and (rec.get("handoff") or {}).get("machine_built") is True,
          "an empty reply with no search: machine-built, never empty",
          (crossed[0] if crossed else "")[:300])
    # 3. Reasoning but no content: the budget was spent thinking.
    reset([])
    rec, crossed, _ = _think([{"content": "", "reasoning_content": "hmm"}])
    check(crossed and "ran out of budget while reasoning" in crossed[0],
          "budget spent reasoning: machine-built, and it says why",
          (crossed[0] if crossed else "")[:300])
    # 4. The investigation itself raises: still a hand-off, and the why.
    saved = shomen.investigate
    shomen.investigate = lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("boom"))
    try:
        reset([])
        rec, crossed, _ = _think([])
    finally:
        shomen.investigate = saved
    check(crossed and "[machine-built hand-off" in crossed[0]
          and "raised: RuntimeError: boom" in crossed[0]
          and "raised" in rec.get("why", ""),
          "an investigation that raises: machine-built, the error named",
          json.dumps(rec)[:300])


def test_citations_are_verified_and_unverified_facts_labelled():
    import shomen
    text = ("## Facts\n"
            "- The loader caches the plan. src/loader.ts:42\n"
            "- The size is set in lib/conf.yaml:3\n"
            "- It probably leaks under load.\n"
            "- The cache never evicts. (reasoning, not checked against "
            "source)\n"
            "**OPEN QUESTIONS:**\n- Is the cache per process?\n"
            "NEXT STEP: read src/loader.ts:30-60\n")
    trace = [{"hop": 0, "tool": "find_by_pattern",
              "args": {"pattern": "evict"}, "chars": 12, "empty": True,
              "paths": 0}]
    out, st = shomen.handoff(text, trace, {"src/loader.ts"}, "h1")
    check("- The loader caches the plan. src/loader.ts:42\n" in out,
          "a fact citing a retrieved path stands as read", out)
    check("lib/conf.yaml:3 [cites lib/conf.yaml, which this investigation "
          "did not retrieve: unverified]" in out,
          "a fact citing a path never retrieved is kept, labelled", out)
    check("It probably leaks under load. (reasoning, not checked against "
          "source)" in out and out.count("(reasoning, not checked") == 2,
          "an uncited fact is labelled reasoning; one the helper labelled "
          "is not labelled twice", out)
    check('- find_by_pattern "evict" (search log)' in out
          and "SEARCHED, FOUND NOTHING" in out,
          "an empty search from the trace is listed, so it is not repeated",
          out)
    check(st["facts"] == 4 and st["verified"] == 1 and st["unverified"] == 3
          and st["searched_empty"] == 1 and st["open"] == 1
          and st["structured"] is True and st["chars"] == len(out),
          "the counts", json.dumps(st))
    check(out.index("FACTS") < out.index("SEARCHED") < out.index("OPEN")
          < out.index("NEXT STEP") and "read src/loader.ts:30-60" in out,
          "rendered in the one fixed order, inline items kept", out)
    # An unstructured finding is still handed back: its lines are facts.
    out2, st2 = shomen.handoff("The loader caches the plan in "
                               "src/loader.ts.\n\nNothing evicts it.",
                               [], {"src/loader.ts"})
    check(st2["structured"] is False and st2["facts"] == 2
          and st2["verified"] == 1 and "Nothing evicts it. (reasoning" in out2,
          "an unstructured finding is read as facts, each one checked",
          out2)


def test_the_fold_back_answer_carries_the_findings():
    """Live gate 2026-09-24: after a 28-search investigation the visible
    answer was "Today I was inspired by austerity. After thinking deeply,
    the deprecation chain is pinned to r179 with the hand-off above." -- the
    hand-off sat in prefilled REASONING the user never sees. The prefilled
    reasoning now ENDS by saying so and that the answer states the findings
    in full; the think_deeply line says the same; and x_yamadori records
    what followed the opening (fold_back_answer)."""
    import deep
    import shomen
    rec, crossed, sent = _think([{"content": HANDOFF_READ}])
    r = crossed[0] if crossed else ""
    check(r.endswith(proxy.FOLD_BACK_TAIL)
          and "The user sees only my answer" in proxy.FOLD_BACK_TAIL
          and "in full" in proxy.FOLD_BACK_TAIL
          and shomen.PHRASES["investigate"] in proxy.FOLD_BACK_TAIL
          and not proxy.FOLD_BACK_TAIL.endswith(" "),
          "the prefilled hand-off ends: the user sees only the answer, which "
          "states the findings in full and opens with the fold-back phrase",
          r[-400:])
    check("above" not in deep.THINK_REASONING
          and "The user sees neither" in deep.THINK_REASONING,
          "the think_deeply fold-back no longer points at a result 'above'",
          deep.THINK_REASONING)
    bad = proxy.fold_back_answer(
        "Today I was inspired by austerity. After thinking deeply, the "
        "deprecation chain is pinned to r179 with the hand-off above.")
    good = proxy.fold_back_answer(
        "Today I was inspired by cedar. After thinking deeply, `label()` "
        "was deprecated in r179 in favour of `setName()`; the warning is "
        "emitted by src/nodes/core/Node.js:412, shown here:\n```js\n"
        "label( name ) { warnOnce( 'label() is deprecated' ); }\n```\n"
        "Replace every `.label(` call with `.setName(`.")
    check(bad["refers_to_hidden"] and bad["chars_after_opening"] <
          proxy.FOLD_BACK_MIN_CHARS,
          "the gate's answer is caught: it points at hidden text and is short",
          json.dumps(bad))
    check(good["refers_to_hidden"] is None and good["chars_after_opening"]
          >= proxy.FOLD_BACK_MIN_CHARS and good["opened"],
          "an answer that states its findings passes", json.dumps(good))
    # The second live run: the opening, then a sentence about the hand-off,
    # then the opening AGAIN as a bold heading. The tail forbids both; the
    # guard removes the repeat deterministically and counts it.
    check("never mentions the hand-off, the investigation or anything "
          "\"above\"" in proxy.FOLD_BACK_TAIL
          and "never writes the opening phrase a second time"
          in proxy.FOLD_BACK_TAIL,
          "the tail names the two observed failures", proxy.FOLD_BACK_TAIL)
    live2 = ("Today I was inspired by schade. After thinking deeply, the "
             "answer below was assembled from the hand-off facts.\n\n"
             "**After thinking deeply,**\n\nThe TSL node method `label()` "
             "is deprecated in r179.\n\n## After thinking deeply\n\nDone.")
    rec: dict = {}
    out = proxy.dedup_opening(live2, rec)
    check(out.count("After thinking deeply") == 1
          and "the hand-off facts.\n\nThe TSL node" in out
          and "deprecated in r179.\n\nDone." in out
          and rec.get("_opening_repeats_removed") == 2,
          "a repeated bold or heading copy of the opening is removed, the "
          "prefilled one kept, and the removals counted", repr(out))
    check(proxy.fold_back_answer(out)["refers_to_hidden"] == "hand-off",
          "and the pointer at the hand-off is still caught (the live check "
          "stays strict)", json.dumps(proxy.fold_back_answer(out)))
    check(proxy.dedup_opening("After thinking deeply, the loop is O(n).")
          == "After thinking deeply, the loop is O(n).",
          "an answer with one opening is untouched")


def test_a_verified_fact_carries_its_source_lines():
    """THE EVIDENCE (operator, 2026-09-24). Main and the user cannot open a
    path on this server, so a verified fact carries the lines it cites, read
    by the verifier from the held source and labelled with where they came
    from; and a citation the verifier cannot read -- the live gate's
    "three/src/core/Open/Three.js:714", no such held file -- is removed from
    the fact, which crosses as reasoning (it had crossed, cited)."""
    import shomen
    seen = {"src/loader.ts", "src/core/open/three.js", "src/a.ts"}
    text = ("FACTS\n"
            "- The loader caches the plan. src/loader.ts:42\n"
            "- The camera flips the target at src/core/Open/Three.js:714.\n"
            "- The loop runs src/loader.ts:10-30\n"
            "- Past the end: src/a.ts:500\n"
            "SEARCHED, FOUND NOTHING\n- none\nOPEN QUESTIONS\n- none\n"
            "NEXT STEP\n- read src/app.ts:3 in your project\n")
    out, st = shomen.handoff(text, [], seen, "h9")
    src = open(os.path.join(_SRC, "src/loader.ts"), encoding="utf-8"
               ).read().split("\n")
    want42 = "\n".join(src[38:45])            # 39..45: 3 either side of 42
    check("source: fixture@1.0.0 src/loader.ts:39-45\n```ts\n" + want42
          + "\n```" in out,
          "a verified fact carries the cited line and 3 either side, exactly "
          "as the file holds them, labelled package@version path:start-end",
          out[:600])
    want_range = "\n".join(src[9:21])          # 10..21: capped at 12 lines
    check("source: fixture@1.0.0 src/loader.ts:10-21\n```ts\n" + want_range
          + "\n```" in out,
          "a cited range is inlined, capped at EXCERPT_MAX_LINES (12)",
          out[:900])
    check("Open/Three.js" not in out
          and "- The camera flips the target. (reasoning, not checked "
              "against source)" in out,
          "a citation of a file the verifier cannot read is REMOVED, and the "
          "fact crosses as reasoning (never passed on)", out)
    check("src/a.ts:500" not in out and "- Past the end (reasoning, not "
                                          "checked against source)" in out,
          "a line past the file's end is removed the same way", out)
    check("read src/app.ts:3 in your project" in out,
          "NEXT STEP is not checked: it may name the user's own file", out)
    check(st["verified"] == 2 and st["excerpts"] == 2
          and st["citations_removed"] == 2 and st["unverified"] == 2,
          "the counts: 2 verified with excerpts, 2 citations removed",
          json.dumps(st))
    # Live gate 2026-09-24 (second run): 3 of 5 excerpts ended on a blank
    # line the fence dropped, so each body was one line short of its label.
    # A file shaped like three's Object3D.lookAt: blank lines, tab indents,
    # a trailing tab. Every excerpt is compared the way the live test does.
    import re as _re
    blanky = os.path.join(_SRC, "src/core/Blank.js")
    os.makedirs(os.path.dirname(blanky), exist_ok=True)
    body = ["", "\tlookAt( x, y, z ) {", "", "\t\tconst parent = this.parent;",
            "", "\t\tif ( this.isCamera ) {", "", "\t\t\t_m1.lookAt( a );\t",
            "", "\t\t} else {", "", "\t\t\t_m1.lookAt( b );", "", "\t\t}", "",
            "\t}", ""]
    with open(blanky, "w", encoding="utf-8", newline="") as fh:
        fh.write("\r\n".join(body))
    t2 = ("FACTS\n- cameras flip src/core/Blank.js:7\n"
          "- the else branch src/core/Blank.js:10-15\n"
          "- the start src/core/Blank.js:2\n")
    out2, st2 = shomen.handoff(t2, [], {"src/core/blank.js"}, "h11")
    pat = _re.compile(r"source: (\S+) (\S+?):(\d+)-(\d+)\n(`{3,})[^\n]*\n"
                      r"(.*?)\n\5(?:\n|$)", _re.S)
    found = [(m.group(3), m.group(4), m.group(6)) for m in pat.finditer(out2)]
    exact = [b == "\n".join(body[int(a) - 1:int(z)]) for a, z, b in found]
    check(len(found) == 3 and all(exact)
          and all(not b.split("\n")[0].strip() == "" and
                  b.split("\n")[-1].strip() for _, _, b in found),
          "an excerpt's body is exactly the lines its label names -- no "
          "blank edge dropped by the fence, a trailing tab kept, CRLF read "
          "as the file's lines", json.dumps({"found": found,
                                              "exact": exact})[:600])
    # The budget: past EXCERPT_TOTAL_CHARS a verified fact says its excerpt
    # was not inlined.
    saved = shomen.EXCERPT_TOTAL_CHARS
    shomen.EXCERPT_TOTAL_CHARS = 300
    try:
        many = "FACTS\n" + "\n".join(f"- fact {i}. src/loader.ts:{i * 5}"
                                     for i in range(1, 5))
        out, st = shomen.handoff(many, [], seen, "h10")
    finally:
        shomen.EXCERPT_TOTAL_CHARS = saved
    check(st["verified"] == 4 and st["excerpts"] >= 1
          and st["excerpts_skipped"] >= 1
          and "[excerpt not inlined: the hand-off's excerpt budget" in out,
          "past the excerpt budget a verified fact says its lines were not "
          "inlined", json.dumps(st))


def test_the_hand_off_size_breaker():
    import shomen
    big = "- " + ("The plan is cached. " * 400) + "src/a.ts:1"
    out, st = shomen.handoff("FACTS\n" + big, [], {"src/a.ts"}, "h2")
    # The breaker is on the helper's own words; the inlined evidence rides
    # on top of it (shomen THE EVIDENCE).
    lim = shomen.MAX_FINDING_CHARS + st.get("excerpt_chars", 0)
    check(st["cut"] is True and f"[hand-off cut at {lim} characters of " in out
          and len(out) < lim + 80
          and lim <= shomen.MAX_FINDING_CHARS + shomen.EXCERPT_TOTAL_CHARS,
          f"over {lim} characters: cut, and the cut says so",
          f"{len(out)} {out[-80:]}")
    many = "FACTS\n" + "\n".join(f"- fact {i} (reasoning, not checked "
                                 f"against source)" for i in range(20))
    out, st = shomen.handoff(many, [], set(), "h3")
    shown = sum(1 for ln in out.splitlines() if ln.startswith("- fact "))
    check(shown == shomen.MAX_ITEMS["facts"]
          and f"(+{20 - shown} more; trace handle h3)" in out
          and st["facts"] == 20,
          "a section over its cap is distilled to the cap, and says how "
          "many more the trace holds", out[-200:])
    steps = [{"hop": i, "tool": "find_by_pattern",
              "args": {"pattern": f"p{i}"}, "chars": 0, "empty": True,
              "paths": 0} for i in range(30)]
    out, st = shomen.machine_handoff("q?", steps, set(), "why", "h4")
    check(len(out) < 2000 and "(+22 more; trace handle h4)" in out,
          "a machine-built hand-off is capped the same way",
          f"{len(out)} {out[-200:]}")


def test_x_yamadori_carries_the_hand_off_counts():
    import shomen
    helper = [{"content": HANDOFF_READ}]
    _helper_sent.clear()

    def post(path, payload, timeout=3600):
        _helper_sent.append(json.loads(json.dumps(payload)))
        m = helper.pop(0) if helper else {"content": ""}
        return {"choices": [{"message": m, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
    saved = shomen._post
    shomen._post = post
    try:
        reset([reply("It sweeps the profile.")])
        SELECTION["investigate"] = True
        ev = run_stream(Q_LATHE)
    finally:
        shomen._post = saved
    inv = (final_chunk(ev).get("x_yamadori") or {}).get("investigate") or {}
    h = inv.get("handoff") or {}
    check(HANDOFF_KEYS <= set(h) and inv.get("injected")
          and h.get("facts") == 2 and h.get("chars", 0) > 0,
          "x_yamadori.investigate.handoff: facts, unverified, "
          "searched_empty, open, chars, machine_built", json.dumps(inv))
    blob = json.dumps(final_chunk(ev).get("x_yamadori"))
    check("segment count" not in blob,
          "and never the hand-off's text", blob[:200])
    first = _seen[0]["messages"] if _seen else []
    pre = first[-1] if first else {}
    check(pre.get("role") == "assistant"
          and str(pre.get("reasoning_content", "")).startswith(
              proxy.REASONING_HEAD)
          and "segment count defaults to 12" in pre["reasoning_content"],
          "the answering generation reads the hand-off as its own reasoning",
          json.dumps(first)[:300])
    helper_user = [m for m in (_helper_sent[0]["messages"] if _helper_sent
                               else []) if m.get("role") == "user"]
    check(helper_user and "Inspiration word: cedar" in
          helper_user[-1]["content"],
          "the investigate job carries the concept seed in its USER message",
          json.dumps(helper_user)[:300])


# --------------------------------------------------------------------------
# FIX-UP, VERIFIED, REPAIRED, THE WARM (operator, 2026-09-24): the second
# brain repairs code main wrote; main never sees a failed attempt; the
# client gets the result in the fold-back phrases; the slot is warmed with
# the turn as delivered.
# --------------------------------------------------------------------------
BROKEN = "def f(:\n    return 1\n"
FIXED = "def f():\n    return 1\n"
WRITE_TOOL = {"type": "function", "function": {
    "name": "write_file", "parameters": {"type": "object"}}}


def _helper_fixes(sent: list[dict]):
    """shomen._post, faked: the second brain answers with FIXED."""
    def post(path, payload, timeout=3600):
        sent.append(json.loads(json.dumps(payload)))
        return {"choices": [{"message": {"content": f"```python\n{FIXED}```"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
    return post


def _conversation(account: str = "acct", key: str = "conv-1") -> dict:
    return {"_slot": {"key": key, "transient": False, "account": account,
                      "record": False},
            "_ledger": {"account": account, "session": key,
                        "turn_key": "u:" + key}}


def _wait_warm(n: int = 3, timeout: float = 5.0) -> list:
    import time as _t
    end = _t.time() + timeout
    while _t.time() < end and len(_warms) < n:
        _t.sleep(0.02)
    return list(_warms)


def _run_both(script, tools=None, user="write hi.py"):
    """(streamed events, complete() response) for the same script."""
    prep = json.loads(json.dumps(PREP))
    reset(list(script))
    PREP.update(prep)
    ev = run_stream(user, tools=tools)
    s_seen, s_warm = list(_seen), _wait_warm()
    reset(list(script))
    PREP.update(prep)
    body = {"model": "yamadori", "messages": [{"role": "user",
                                                "content": user}]}
    if tools:
        body["tools"] = tools
    d = proxy.complete(body)
    return ev, s_seen, s_warm, d


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
            n["calls"] += 1
            return landed and n["calls"] >= 2
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


def test_the_hand_off_survives_an_image_hop():
    """Pre-deploy review, 2026-09-24 (MINOR): deep thinking's hand-off is
    the hop-0 prefill; hops 1+ see hop 0 in its place. Where the upstream
    did not re-send the prefill, the hand-off was gone from every later hop
    of a request that also ran an image tool. It is carried into the hop,
    exactly once whether or not the upstream re-sends it."""
    global ECHO_PREFILL
    pre = {"role": "assistant", "content": "After thinking deeply,",
           "reasoning_content": "FACTS: HANDOFF-7 src/a.ts:1"}
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
        check(r.startswith("FACTS: HANDOFF-7") and r.count("HANDOFF-7") == 1
              and "now draw" in r
              and (hop.get("content") or "").startswith(
                  "After thinking deeply,")
              and (hop.get("content") or "").count("After thinking") == 1,
              f"{what}: hop 1 sees the hand-off in hop 0's reasoning, and the "
              f"opening phrase, exactly once", json.dumps(hop)[:300])


def test_a_broken_client_write_is_repaired_by_the_second_brain():
    import shomen
    proxy.slots.reset(4)
    proxy.nebari.ledger_reset()
    sent: list[dict] = []
    saved = shomen._post
    shomen._post = _helper_fixes(sent)
    try:
        script = [reply("Writing it.", calls=[call(
            "write_file", {"path": "hi.py", "content": BROKEN})])]
        PREP.update(_tool_code=True, _fixup=True, **_conversation())
        ev, s_seen, s_warm, d = _run_both(script, tools=[WRITE_TOOL])
    finally:
        shomen._post = saved
    tc = [e["choices"][0]["delta"]["tool_calls"] for e in ev
          if e.get("choices") and e["choices"][0]["delta"].get("tool_calls")]
    args = json.loads(tc[0][0]["function"]["arguments"]) if tc else {}
    check(len(s_seen) == 1, "main generated ONCE: it never sees the failed "
          "attempt or a feedback round", str(len(s_seen)))
    check(args.get("content") == FIXED,
          "the client receives the call with the second brain's fixed code",
          json.dumps(args)[:200])
    helper_msgs = sent[0]["messages"] if sent else []
    user = helper_msgs[-1]["content"] if helper_msgs else ""
    check(len(helper_msgs) == 2 and helper_msgs[0]["role"] == "system"
          and "write hi.py" in user and BROKEN in user and "line 1" in user
          and "hi.py" in user and "Inspiration word: cedar" in user,
          "the fixup job gets ONLY the request, the file, its errors -- and "
          "the concept seed in its user message", json.dumps(helper_msgs)[:500])
    c = deltas(ev, "content")
    # A note is mechanical since 2026-09-24: no seed line (the seed stays in
    # the fixup job's user message, checked above).
    check(c.startswith("Writing it.\n\nRepaired hi.py (python): 1 syntax "
                       "error fixed in 1 round") and "inspired" not in c,
          "the note: 'Repaired <file>', no seed line", repr(c))
    x = final_chunk(ev).get("x_yamadori") or {}
    check((x.get("tool_code") or {}).get("stopped") == "fixed"
          and (x.get("fold_back") or [{}])[0].get("phrase") == "repaired",
          "x_yamadori: tool_code fixed, fold_back repaired",
          json.dumps({k: x.get(k) for k in ("tool_code", "fold_back")})[:400])
    check((x.get("warm") or {}).get("sent") is True,
          "the delivered turn differs from the generated one: a warm is sent",
          json.dumps(x.get("warm")))
    comp = [b for p, b in s_warm if p.endswith("/completion")]
    rend = [b for p, b in s_warm if p.endswith("/apply-template")]
    last = (rend[0]["messages"][-2] if rend else {})
    check(comp and comp[0].get("n_predict") == 0
          and comp[0].get("cache_prompt") is True
          and comp[0]["prompt"].endswith("<|im_end|>\n")
          and "Repaired hi.py" in comp[0]["prompt"],
          "the warm loads the turn as delivered, cut at its end-of-turn "
          "token, generating nothing", json.dumps(comp)[:300])
    check(json.loads(last.get("tool_calls", [{}])[0]["function"]
                     ["arguments"]).get("content") == FIXED
          if last.get("tool_calls") else False,
          "and the warm renders the FIXED call, not the generated one",
          json.dumps(last)[:300])
    check(proxy.nebari.ledger_get("acct", "call:call_1", "content")
          == c, "the ledger records the delivered content for the call",
          repr(proxy.nebari.ledger_get("acct", "call:call_1", "content")))
    m = d["choices"][0]["message"]
    check(json.loads(m["tool_calls"][0]["function"]["arguments"])["content"]
          == FIXED and m["content"] == c,
          "complete(): the same fixed call and the same content", repr(m)[:300])


def test_medium_checks_a_write_and_only_notes():
    import shomen
    sent: list[dict] = []
    saved = shomen._post
    shomen._post = _helper_fixes(sent)
    try:
        script = [reply(calls=[call("write_file", {"path": "hi.py",
                                                    "content": BROKEN})])]
        PREP.update(_tool_code=True, _fixup=False)
        ev, s_seen, _w, d = _run_both(script, tools=[WRITE_TOOL])
    finally:
        shomen._post = saved
    c = deltas(ev, "content")
    tc = [e["choices"][0]["delta"]["tool_calls"] for e in ev
          if e.get("choices") and e["choices"][0]["delta"].get("tool_calls")]
    check(not sent, "medium: no second-brain generation", str(len(sent)))
    check(c.startswith("Checked hi.py (python): 1 problem (line 1")
          and "not repaired at this effort; sent as written" in c,
          "medium: the note says what the check found, and that it was not "
          "repaired", repr(c))
    check(tc and json.loads(tc[0][0]["function"]["arguments"])["content"]
          == BROKEN, "the call goes out as written")


def test_a_final_answer_is_verified_or_repaired():
    import shomen
    sent: list[dict] = []
    saved = shomen._post
    shomen._post = _helper_fixes(sent)
    try:
        PREP.update(_repair=True, _fixup=True)
        ev, s_seen, _w, d = _run_both(
            [reply(f"Here:\n\n```python\n{FIXED}```")], user="write f")
        c = deltas(ev, "content")
        check(len(s_seen) == 1 and not sent
              and c.endswith("\n\nVerified: the python code above parses.")
              and d["choices"][0]["message"]["content"] == c,
              "code that parses: a one-line 'Verified' note, no generation",
              repr(c))
        sent.clear()
        PREP.update(_repair=True, _fixup=True)
        ev, s_seen, _w, d = _run_both(
            [reply(f"Here:\n\n```python\n{BROKEN}```\nDone.")],
            user="write f")
    finally:
        shomen._post = saved
    c = deltas(ev, "content")
    blocking = d["choices"][0]["message"]["content"]
    check(len(s_seen) == 1 and len(sent) == 2,
          "main generated once on each path; the fixup job ran once on each",
          f"{len(s_seen)} {len(sent)}")
    check(BROKEN.strip() not in blocking and FIXED.strip() in blocking
          and blocking.endswith("\n\nRepaired: 1 "
                                "code block that did not parse, fixed in 1 "
                                "round; the code above is the corrected "
                                "version."),
          "complete(): the repaired answer is delivered IN PLACE, with the "
          "note", repr(blocking))
    check(c.startswith(f"Here:\n\n```python\n{BROKEN}```")
          and "\n\nRepaired: 1 code block" in c and "inspired" not in c
          and c.rstrip().endswith(f"```python\n{FIXED.rstrip()}\n```"),
          "streamed: the original went out, so the repaired block follows it",
          repr(c))
    x = d.get("x_yamadori") or {}
    check((x.get("repair") or {}).get("into") == "in_place"
          and (final_chunk(ev).get("x_yamadori") or {}).get("repair", {})
          .get("into") == "appended",
          "x_yamadori.repair says where it went", json.dumps(x.get("repair")))


def test_the_delegate_arm_runs_through_the_one_runner():
    import shomen
    calls_seen: list = []
    saved = shomen.run

    def fake_run(job, **spec):
        calls_seen.append(job)
        return {"job": job, "ok": True, "finding": "it is in a.ts",
                "hops": 0, "helper_tokens": 0, "seconds": 0, "handle": "h"}
    shomen.run = fake_run
    try:
        out = _REAL_RUN_OUR_TOOL("delegate_investigation",
                                 {"question": "where is LatheGeometry?"},
                                 None)
    finally:
        shomen.run = saved
    check(calls_seen == ["investigate"] and "it is in a.ts" in out,
          "delegate_investigation (the header arm) is the runner's "
          "investigate job", str(calls_seen))


def main() -> int:
    for fn in (test_fake_upstream_is_alive,
               test_an_answer_without_tools_is_generated_once,
               test_a_tool_hop_then_the_answer_is_two_generations_not_three,
               test_no_reasoning_after_the_first_content,
               test_reasoning_continuity,
               test_internal_hops_keep_their_reasoning,
               test_the_breaker_lands_with_tools_withdrawn,
               test_complete_lands_through_the_same_helper,
               test_the_tool_turn_cap_lands_and_is_recorded,
               test_an_uncapped_turn_records_the_cap_unhit,
               test_deep_thinking_honours_an_effort_override,
               test_deep_thinking_obeys_the_tool_turn_cap,
               test_a_client_tool_call_is_forwarded_whole,
               test_a_length_finish_streams_the_budget_notice,
               test_truncation_marker_at_the_breaker,
               test_x_yamadori_rides_the_final_chunk_and_matches_complete,
               test_fan_out_runs_on_the_streamed_path_too,
               test_deep_thinking_runs_on_the_streamed_path,
               test_a_zero_search_hand_off_crosses_labelled,
               test_the_turn_cap_landing_hands_off,
               test_a_helper_that_returns_nothing_still_hands_off,
               test_citations_are_verified_and_unverified_facts_labelled,
               test_a_verified_fact_carries_its_source_lines,
               test_the_fold_back_answer_carries_the_findings,
               test_a_repeated_opening_never_reaches_the_client,
               test_the_hand_off_size_breaker,
               test_x_yamadori_carries_the_hand_off_counts,
               test_the_compaction_store_keeps_the_ledgers_rendering,
               test_the_hand_off_survives_an_image_hop,
               test_a_broken_client_write_is_repaired_by_the_second_brain,
               test_medium_checks_a_write_and_only_notes,
               test_a_final_answer_is_verified_or_repaired,
               test_the_delegate_arm_runs_through_the_one_runner):
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
