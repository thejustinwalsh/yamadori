#!/usr/bin/env python
"""The SSE layer, asserted against canned upstream streams. No network.

WHAT THIS IS GATING

The proxy's streamed replies are the only thing a client sees while a turn
runs. The promises `streaming.py` makes, each asserted below:

  1. Every chunk is OpenAI's `chat.completion.chunk`, carrying OUR id, so a
     client sees one completion however many upstream calls it took.
  2. Reasoning never leaks into the reply: it becomes an empty-delta
     heartbeat.
  3. Client-declared tool calls are FORWARDED (dropping them made Hermes
     retry four times, which read as the model looping).
  4. A hop holds its text back but announces each tool name the moment it
     is known, exactly once, and returns the assembled message at the end.
  5. An upstream that fails MID-STREAM is an error, not an empty answer.
     (Until 2026-09-22 a `data: {"error": ...}` event was read as an empty
     delta, and the stream ended cleanly with nothing in it.)

NO NETWORK

`urllib.request.urlopen` is replaced with a stub that replays canned SSE lines,
and the stub records every request so the tests can assert what was sent.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import streaming  # noqa: E402

CID, MODEL = "chatcmpl-ours", "yamadori"

_results: list[tuple[bool, str, str]] = []
_sent: list[dict] = []
_leaks: list[str] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


def _blocked(req, timeout=None):                                  # noqa: ARG001
    _leaks.append(getattr(req, "full_url", str(req)))
    raise OSError("test_streaming: network is blocked")


urllib.request.urlopen = _blocked


class _Resp:
    def __init__(self, lines: list[bytes]):
        self._lines = lines

    def __iter__(self):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def upstream(*events) -> None:
    """Replay these SSE events. A dict is sent as `data: <json>`; bytes as-is."""
    lines = []
    for e in events:
        if isinstance(e, (bytes, bytearray)):
            lines.append(bytes(e))
        else:
            lines.append(b"data: " + json.dumps(e).encode() + b"\n")
        lines.append(b"\n")

    def urlopen(req, timeout=None):                               # noqa: ARG001
        _sent.append({"url": req.full_url, "body": json.loads(req.data)})
        return _Resp(lines)
    urllib.request.urlopen = urlopen


def restore() -> None:
    urllib.request.urlopen = _blocked


def delta(d: dict | None = None, finish=None, uid="upstream-id-1") -> dict:
    return {"id": uid, "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": d or {}, "finish_reason": finish}]}


def parse(b: bytes) -> dict:
    assert b.startswith(b"data: ") and b.endswith(b"\n\n"), b[:40]
    return json.loads(b[6:].decode())


# ---------------------------------------------------------------------------


def test_the_fixture_is_not_a_real_server():
    check(urllib.request.urlopen is _blocked,
          "urlopen is blocked unless a test installs a canned stream")
    try:
        list(streaming.stream_upstream("http://127.0.0.1:1", {}, CID, MODEL))
        refused = False
    except OSError:
        refused = True
    check(refused and _leaks, "an unstubbed call is refused and recorded")
    _leaks.clear()


def test_chunk_is_the_openai_shape():
    c = parse(streaming.chunk(CID, MODEL, {"content": "hi"}, finish="stop"))
    check(c["id"] == CID and c["object"] == "chat.completion.chunk"
          and c["model"] == MODEL, "id, object and model are set", json.dumps(c))
    ch = c["choices"][0]
    check(ch == {"index": 0, "delta": {"content": "hi"}, "finish_reason": "stop"},
          "one choice carrying the delta and finish", json.dumps(ch))
    check(isinstance(c["created"], int), "created is an integer timestamp")
    check(parse(streaming.text_chunk(CID, MODEL, "x"))["choices"][0]["delta"]
          == {"content": "x"}, "text_chunk is a content delta")
    check(streaming.DONE == b"data: [DONE]\n\n", "DONE is the SSE terminator")
    a, b = streaming.new_id(), streaming.new_id()
    check(a.startswith("chatcmpl-") and a != b and len(a) == 33,
          "new_id is a fresh chatcmpl- id each time", a)


def test_describe_call_names_the_informative_argument():
    check(streaming.describe_call("find_by_pattern", {"pattern": "createWorld"})
          == 'find_by_pattern "createWorld"', "the pattern is quoted after the name")
    check(streaming.describe_call("t", {"limit": 3, "query": "q"}) == 't "q"',
          "a known key wins over an unknown one")
    check(streaming.describe_call("t", {"symbol": "A", "query": "B"}) == 't "A"',
          "symbol is preferred over query")
    check(streaming.describe_call("t", {"symbol": "", "path": "a.ts"})
          == 't "a.ts"', "an empty value is skipped")
    long = streaming.describe_call("t", {"query": "x" * 500})
    check(len(long) <= 2 + 80, "a long argument is cut to 80 characters",
          str(len(long)))
    check(streaming.describe_call("describe_index", {}) == "describe_index",
          "no argument is just the name")
    for bad in (["a", "list"], "a string", 42, None):
        try:
            out = streaming.describe_call("t", bad)
            check(out == "t", f"non-dict arguments ({type(bad).__name__}) give "
                  "just the name", out)
        except Exception as e:                                   # noqa: BLE001
            check(False, f"non-dict arguments ({type(bad).__name__}) do not "
                  "crash the stream", f"{type(e).__name__}: {e}")


def test_stream_upstream_rewrites_ids_and_forwards_content():
    upstream(delta({"role": "assistant"}), delta({"content": "Hel"}),
             delta({"content": "lo"}), delta({}, finish="stop"), b"data: [DONE]")
    try:
        out = [parse(b) for b in streaming.stream_upstream(
            "http://up", {"messages": [], "stream": False}, CID, MODEL)]
    finally:
        restore()
    check(_sent and _sent[-1]["body"]["stream"] is True,
          "the upstream request asks for a stream even if the payload did not")
    check(_sent and _sent[-1]["url"] == "http://up/v1/chat/completions",
          "at the chat-completions path", _sent[-1]["url"] if _sent else "")
    check(all(c["id"] == CID for c in out),
          "every chunk carries OUR id, never upstream's",
          str({c["id"] for c in out}))
    check(all(c["model"] == MODEL for c in out), "and our model name")
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in out)
    check(text == "Hello", "the content arrives intact and in order", text)
    check(out and out[0]["choices"][0]["delta"] == {"role": "assistant"},
          "the role delta is forwarded")
    check(not any(c["choices"][0]["finish_reason"] == "stop" for c in out),
          "a `stop` finish is left for the proxy to send once, at the end")


def test_stream_upstream_hides_reasoning_behind_a_heartbeat():
    upstream(delta({"reasoning_content": "secret plan"}),
             delta({"content": "answer"}), b"data: [DONE]")
    try:
        raw = list(streaming.stream_upstream("http://up", {}, CID, MODEL))
    finally:
        restore()
    check(not any(b"secret plan" in b for b in raw),
          "reasoning text never reaches the client")
    out = [parse(b) for b in raw]
    check(out and out[0]["choices"][0]["delta"] == {},
          "reasoning becomes an empty-delta heartbeat")
    check(out[-1]["choices"][0]["delta"] == {"content": "answer"},
          "and the answer follows")


def test_stream_upstream_forwards_client_tool_calls():
    tc = [{"index": 0, "id": "call_1", "type": "function",
           "function": {"name": "client_tool", "arguments": "{}"}}]
    upstream(delta({"tool_calls": tc}), delta({}, finish="tool_calls"),
             b"data: [DONE]")
    try:
        out = [parse(b) for b in streaming.stream_upstream("http://up", {}, CID, MODEL)]
    finally:
        restore()
    fwd = [c for c in out if c["choices"][0]["delta"].get("tool_calls")]
    check(len(fwd) == 1 and fwd[0]["choices"][0]["delta"]["tool_calls"] == tc,
          "a client tool call is forwarded verbatim", json.dumps(out)[:200])
    check(any(c["choices"][0]["finish_reason"] == "tool_calls" for c in out),
          "and its non-stop finish reason is forwarded")


def test_stream_upstream_skips_noise_and_stops_at_done():
    upstream(b": keep-alive comment", b"event: ping", b"data: {not json",
             delta({"content": "a"}), b"data: [DONE]", delta({"content": "AFTER"}))
    try:
        out = [parse(b) for b in streaming.stream_upstream("http://up", {}, CID, MODEL)]
    finally:
        restore()
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in out)
    check(text == "a", "comments, non-data lines and bad JSON are skipped, and "
          "nothing after [DONE] is read", text)


def test_an_error_mid_stream_is_raised_not_swallowed():
    """THE BUG THIS FILE FOUND. The error event has no `choices`, so it was
    read as an empty delta and the stream ended as if it had succeeded."""
    for label, event in (
            ("data: {error}", {"error": {"code": 500, "message": "context "
                                         "size exceeded", "type": "server_error"}}),
            ("error: line", b'error: {"message": "slot killed"}')):
        for reader in ("stream_upstream", "stream_hop"):
            upstream(delta({"content": "partial"}), event, b"data: [DONE]")
            try:
                gen = getattr(streaming, reader)("http://up", {}, CID, MODEL)
                list(gen)
                raised = None
            except streaming.UpstreamError as e:
                raised = str(e)
            except Exception as e:                               # noqa: BLE001
                raised = f"WRONG TYPE {type(e).__name__}: {e}"
            finally:
                restore()
            check(raised is not None and not raised.startswith("WRONG"),
                  f"{reader}: a `{label}` event raises UpstreamError", str(raised))
            check(raised is not None and ("exceeded" in raised
                                          or "slot killed" in raised),
                  f"{reader}: and the server's own message is carried",
                  str(raised))


def test_a_chunk_mentioning_error_in_content_is_not_an_error():
    upstream(delta({"content": "the error was fixed"}), b"data: [DONE]")
    try:
        out = [parse(b) for b in streaming.stream_upstream("http://up", {}, CID, MODEL)]
    finally:
        restore()
    check(len(out) == 1, "content that talks about an error is just content")


def test_stream_hop_holds_text_and_announces_tools_once():
    upstream(
        delta({"reasoning_content": "thinking"}),
        delta({"content": "Let me look."}),
        delta({"tool_calls": [{"index": 0, "id": "c0",
                               "function": {"name": "find_", "arguments": ""}}]}),
        delta({"tool_calls": [{"index": 0,
                               "function": {"name": "definition_opt",
                                            "arguments": '{"sym'}}]}),
        delta({"tool_calls": [{"index": 0,
                               "function": {"arguments": 'bol": "Trait"}'}}]}),
        delta({"tool_calls": [{"index": 1, "id": "c1",
                               "function": {"name": "read_file_range",
                                            "arguments": "{}"}}]}),
        delta({}, finish="tool_calls"),
        b"data: [DONE]")
    try:
        items = list(streaming.stream_hop("http://up", {"x": 1}, CID, MODEL))
    finally:
        restore()
    events = [parse(b) for k, b in items if k == "event"]
    dones = [m for k, m in items if k == "done"]
    check(len(dones) == 1 and items[-1][0] == "done",
          "exactly one `done`, and it comes last", str([k for k, _ in items]))
    said = "".join(e["choices"][0]["delta"].get("content", "") for e in events)
    check("Let me look." not in said,
          "the hop's own text is held back from the client", said)
    # Counts announcements, not names: a name split across deltas is announced
    # on its first fragment ("find_"), which is what "as soon as the name
    # appears" buys. Pinning the fragment here would pin that quirk.
    announced = [e for e in events
                 if e["choices"][0]["delta"].get("content", "").startswith("`")]
    check(len(announced) == 2 and said.count("read_file_range") == 1,
          "each tool call is announced exactly once", said)
    check(events and events[0]["choices"][0]["delta"] == {},
          "reasoning produces a heartbeat, not text")
    if dones:
        d = dones[0]
        msg = d["message"]
        check(d["finish_reason"] == "tool_calls", "the finish reason is kept",
              str(d["finish_reason"]))
        check(msg["content"] == "Let me look.", "the held text is in the message")
        check(msg.get("reasoning_content") == "thinking",
              "and so is the reasoning")
        calls = msg.get("tool_calls") or []
        check([c["id"] for c in calls] == ["c0", "c1"],
              "tool calls are assembled in index order", json.dumps(calls)[:160])
        if calls:
            f0 = calls[0]["function"]
            check(f0["name"] == "find_definition_opt",
                  "a name split across deltas is joined", f0["name"])
            check(json.loads(f0["arguments"]) == {"symbol": "Trait"},
                  "and so are its arguments", f0["arguments"])


def test_stream_hop_with_no_tool_calls_is_a_plain_message():
    upstream(delta({"content": "done"}), delta({}, finish="stop"), b"data: [DONE]")
    try:
        items = list(streaming.stream_hop("http://up", {}, CID, MODEL))
    finally:
        restore()
    check([k for k, _ in items] == ["done"], "no events, just the result")
    msg = items[-1][1]["message"]
    check(msg == {"role": "assistant", "content": "done"},
          "no tool_calls or reasoning keys when there were none", json.dumps(msg))


def main() -> int:
    for fn in (test_the_fixture_is_not_a_real_server,
               test_chunk_is_the_openai_shape,
               test_describe_call_names_the_informative_argument,
               test_stream_upstream_rewrites_ids_and_forwards_content,
               test_stream_upstream_hides_reasoning_behind_a_heartbeat,
               test_stream_upstream_forwards_client_tool_calls,
               test_stream_upstream_skips_noise_and_stops_at_done,
               test_an_error_mid_stream_is_raised_not_swallowed,
               test_a_chunk_mentioning_error_in_content_is_not_an_error,
               test_stream_hop_holds_text_and_announces_tools_once,
               test_stream_hop_with_no_tool_calls_is_a_plain_message):
        print(f"\n--- {fn.__name__} ---")
        n0 = len(_results)
        try:
            fn()
        except Exception:                                        # noqa: BLE001
            check(False, f"{fn.__name__} itself raised",
                  traceback.format_exc().strip().split("\n")[-1])
        finally:
            restore()
        for ok, name, detail in _results[n0:]:
            print(("  pass  " if ok else "  FAIL  ") + name
                  + (f"   <- {detail}" if not ok and detail else ""))

    check(not _leaks, "no test reached a real server", str(_leaks))
    passed = sum(1 for ok, _, _ in _results if ok)
    total = len(_results)
    print(f"\n{'=' * 70}\n  {passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
