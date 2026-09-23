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

import proxy  # noqa: E402
import repeats  # noqa: E402

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


# --------------------------------------------------------------------------
# The fake upstream
# --------------------------------------------------------------------------

_script: list[dict] = []          # replies, popped in order
_seen: list[dict] = []            # request bodies, in order


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


class _Up(BaseHTTPRequestHandler):
    def do_POST(self):                                           # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _seen.append(body)
        if not _script:
            self.send_response(500)
            self.end_headers()
            return
        r = _script.pop(0)
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
    def log_answer(turn, root, content, hops, ms):
        _log.append(("answer", hops, content))


# What the selection engine decided, as the stubbed prepare() reports it.
# Tests that exercise deep thinking or fan-out set it; reset() clears it.
SELECTION: dict = {}


def _prepare(body: dict) -> dict:
    tools, _ = proxy.merge_tools(body.get("tools"))
    return {"model": "bonsai", "messages": list(body["messages"]),
            "tools": tools, "max_tokens": 10240, "_tier": {"name": "test"},
            "_tools_gate": {"offer": True, "situation": "TEST",
                            "because": "stub"},
            "_selection": dict({"hints": False, "investigate": False,
                                "fanout_n": 1, "because": {}, "signals": {}},
                               **SELECTION)}


def _run_our_tool(name, args, db, root=None, turn=None, state=None):
    _ran.append((name, args))
    return TOOL_OUTPUT.get(name, f"{name}: 1 result\nsrc/a.ts:1")


proxy.corpus = _Corpus
proxy.prepare = _prepare
proxy.run_our_tool = _run_our_tool
proxy.resolve_repo = lambda messages, ip="": (None, False, "none")
proxy.session_context = lambda messages, account="": ("k", {"_key": "k"})
proxy.PREAMBLE = False


def reset(script: list[dict]) -> None:
    _script[:] = script
    _seen.clear()
    _log.clear()
    _ran.clear()
    TOOL_OUTPUT.clear()
    SELECTION.clear()


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
    check(sum(1 for e in ev if e.get("choices")
              and e["choices"][0]["delta"].get("content")) > 1,
          "the answer arrived as several deltas, not one blob")
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
                 calls=[call("find_definition_opt", {"symbol": "LatheGeometry"})]),
           reply("src/geometries/LatheGeometry.js:12", reasoning="found it")])
    ev = run_stream()
    check(len(_seen) == 2,
          "one tool hop + one answer = 2 upstream requests (was 3)",
          f"{len(_seen)} requests")
    check(_ran == [("find_definition_opt", {"symbol": "LatheGeometry"})],
          "our tool ran server-side", str(_ran))
    check(any(m.get("role") == "tool" for m in _seen[1]["messages"]),
          "the second request carries the tool result")
    check("src/geometries/LatheGeometry.js:12" in deltas(ev, "content"),
          "the answer is streamed")
    check(deltas(ev, "reasoning_content") == "look it upfound it",
          "reasoning from BOTH hops streamed live",
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
        loop = [reply(calls=[call("find_by_pattern", {"pattern": "x"},
                                  cid=f"c{i}")]) for i in range(3)]
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
        reset([reply(calls=[call("find_by_pattern", {"pattern": "x"})]),
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
    reset([reply(calls=[call("read_file_range",
                             {"path": "src/big.ts", "start": 1, "end": 180})]),
           reply("done")])
    TOOL_OUTPUT["read_file_range"] = text
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
    script = [reply(calls=[call("find_definition_opt", {"symbol": "X"})]),
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
    def run(payload, n=4, timeout=1800, variants=None):
        _fan_calls.append(n)
        good = [r for r in results if r.get("content")]
        votes: dict = {}
        for r in good:
            for p in proxy.fanout.claims(r["content"])[0]:
                votes[p] = votes.get(p, 0) + 1
        top = max(votes.values()) if votes else 0
        return {"n": len(good), "votes": sum(votes.values()),
                "agreement": round(top / len(good), 2) if votes else None,
                "consensus_path": max(votes, key=votes.get) if votes else None,
                "path_votes": votes, "winner": good[0], "results": results}
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
        check(fx.get("agreement") is None
              and fx.get("seeds") == ["lantern", "harbour"]
              and fx.get("winner") == "direct",
              "x_yamadori.fanout: n, seeds, agreement null, the winner kept",
              json.dumps(fx))

        # Real disagreement over named files IS reported, after the answer.
        proxy.fanout.run = _fake_fanout(
            [{"variant": "direct", "content": "see src/a.ts"},
             {"variant": "evidence", "content": "see src/b.ts"}])
        reset([reply("see src/a.ts")])
        SELECTION["fanout_n"] = 2
        ev = run_stream()
        c = deltas(ev, "content")
        check(c.startswith("see src/a.ts") and "unsettled" in c,
              "observed disagreement follows the streamed answer as a note",
              repr(c))
        check(len(_seen) == 1, "the answer itself was still generated once",
              str(len(_seen)))

        reset([reply("see src/a.ts")])
        _fan_calls.clear()
        run_stream()
        check(not _fan_calls, "N=1: no fan-out at all", str(_fan_calls))
    finally:
        proxy.fanout.run = saved


def test_deep_thinking_runs_on_the_streamed_path():
    saved = proxy.shomen.investigate
    got: list = []

    def fake(question, tools, run_tool, context="", hops=8, on_think=None):
        got.append([t["function"]["name"] for t in tools])
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
        check(bool(_seen) and any(str(m.get("content", "")).startswith(
                  proxy.FINDINGS_HEAD) for m in _seen[0]["messages"]),
              "the finding crosses into the answering generation")
        inv = (final_chunk(ev).get("x_yamadori") or {}).get("investigate") or {}
        check(inv.get("ran") and inv.get("injected") and inv.get("handle") == "hx",
              "x_yamadori.investigate on the final chunk", json.dumps(inv))

        got.clear()
        reset([reply("answer")])
        run_stream()
        check(not got, "not selected: not run", str(got))
    finally:
        proxy.shomen.investigate = saved


def main() -> int:
    for fn in (test_fake_upstream_is_alive,
               test_an_answer_without_tools_is_generated_once,
               test_a_tool_hop_then_the_answer_is_two_generations_not_three,
               test_the_breaker_lands_with_tools_withdrawn,
               test_complete_lands_through_the_same_helper,
               test_a_client_tool_call_is_forwarded_whole,
               test_a_length_finish_streams_the_budget_notice,
               test_truncation_marker_at_the_breaker,
               test_x_yamadori_rides_the_final_chunk_and_matches_complete,
               test_fan_out_runs_on_the_streamed_path_too,
               test_deep_thinking_runs_on_the_streamed_path):
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
