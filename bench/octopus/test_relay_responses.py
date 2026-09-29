#!/usr/bin/env python
"""The Octopus relay records a Responses run as it records a chat run. No
GPU, no network.

    python bench/octopus/test_relay_responses.py      -> "N/M checks passed"

2026-09-26: V0 can run on Responses (make_profile.py --wire responses gives
the octo-relay provider `api_mode: codex_responses`). On that wire the
proxy's `x_yamadori`, the usage and the finish ride inside the terminal
event's `response`, and the relay's collector read only chat chunks, so a
Responses run would have recorded none of them. The stream fed here is the
proxy's own (mcp/responses_api.Stream over chat chunks), so a change of
event shape there fails here.
"""
from __future__ import annotations

import json
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "mcp")))

import relay  # noqa: E402
import responses_api as R  # noqa: E402

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


def _chunk(delta: dict, finish=None, **extra) -> bytes:
    return b"data: " + json.dumps(dict({"choices": [{
        "index": 0, "delta": delta, "finish_reason": finish}]}, **extra)
    ).encode() + b"\n\n"


X = {"tier": "xhigh", "route": {"class": "agent_step"}}
USAGE = {"prompt_tokens": 900, "completion_tokens": 40, "total_tokens": 940,
         "prompt_tokens_details": {"cached_tokens": 850},
         "completion_tokens_details": {"reasoning_tokens": 12}}


def _stream(chunks: list[bytes], body: dict) -> bytes:
    _c, ctx = R.to_chat(body)
    return b"".join(R.stream(iter(chunks), ctx))


def test_request_shape():
    body = {"model": "yamadori", "instructions": "sys", "input": [
        {"role": "user", "content": [{"type": "input_text",
                                      "text": "write a game"}]},
        {"type": "function_call", "call_id": "c1", "name": "write_file",
         "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1",
         "output": "wrote it"}],
        "tools": [{"type": "function", "name": "write_file"},
                  {"type": "web_search"}],
        "reasoning": {"effort": "xhigh"}, "max_output_tokens": 800,
        "stream": True}
    s = relay._shape(json.dumps(body).encode())
    check(s["wire"] == "responses" and s["n_messages"] == 3
          and s["last_role"] == "tool" and s["last_head"] == "wrote it"
          and s["reasoning_effort"] == "xhigh" and s["max_tokens"] == 800
          and s["tool_names"] == ["web_search", "write_file"],
          "a Responses request is shaped like a chat one (wire, messages, "
          "effort, tools)", json.dumps(s))
    c = relay._shape(json.dumps({"messages": [{"role": "user",
                                               "content": "hi"}]}).encode())
    check(c["wire"] == "chat" and c["n_messages"] == 1,
          "a chat request is unchanged", json.dumps(c))


def test_streamed_row():
    raw = _stream([
        _chunk({"reasoning_content": "thinking"}),
        _chunk({"content": "Writing."}),
        _chunk({"tool_calls": [{"index": 0, "id": "call_1", "type":
                                "function", "function": {
                                    "name": "write_file",
                                    "arguments": "{}"}}]}),
        _chunk({}, "tool_calls", x_yamadori=X),
        b'data: ' + json.dumps({"choices": [], "usage": USAGE}).encode()
        + b"\n\n", b"data: [DONE]\n\n"], {"input": "x", "stream": True})
    col = relay._Collect(sse=True)
    for i in range(0, len(raw), 97):           # split mid-event on purpose
        col.feed(raw[i:i + 97])
    out = col.done()
    check((out["x_yamadori"] or {}).get("tier") == "xhigh"
          and "responses" in out["x_yamadori"]
          and out["finish_reason"] == "tool_calls"
          and out["tool_calls"] == ["write_file"]
          and out["usage"]["prompt_tokens"] == 900
          and out["usage"]["prompt_tokens_details"]["cached_tokens"] == 850
          and out["usage"]["completion_tokens_details"][
              "reasoning_tokens"] == 12
          and out["content_chars"] == len("Writing.")
          and out["reasoning_chars"] == len("thinking")
          and out.get("wire") == "responses",
          "a streamed Responses turn: x_yamadori, usage (chat terms), "
          "finish tool_calls, the call's name, content and reasoning chars",
          json.dumps({k: v for k, v in out.items() if k != "x_yamadori"}))
    raw = _stream([_chunk({"content": "cut"}), _chunk({}, "length"),
                   b"data: [DONE]\n\n"], {"input": "x", "stream": True})
    col = relay._Collect(sse=True)
    col.feed(raw)
    check(col.done()["finish_reason"] == "length",
          "response.incomplete (max_output_tokens) is recorded as length")
    raw = _stream([_chunk({"content": "par"}), (
        b'data: {"error": {"message": "slot killed", "type": "server_error",'
        b' "code": "upstream_error"}}\n\n'), b"data: [DONE]\n\n"],
        {"input": "x", "stream": True})
    col = relay._Collect(sse=True)
    col.feed(raw)
    out = col.done()
    check(out["finish_reason"] == "error"
          and (out["error"] or {}).get("code") == "server_error",
          "response.failed is recorded as an error with its code",
          json.dumps(out)[:300])


def test_blocking_row():
    _c, ctx = R.to_chat({"input": "x"})
    resp = R.of_chat({"choices": [{"message": {"content": "Done."},
                                   "finish_reason": "stop"}],
                      "usage": USAGE, "x_yamadori": X}, ctx)
    col = relay._Collect(sse=False)
    col.feed(json.dumps(resp).encode())
    out = col.done()
    check(out["finish_reason"] == "stop" and out["content_chars"] == 5
          and out["usage"]["total_tokens"] == 940
          and out["x_yamadori"]["tier"] == "xhigh",
          "a blocking Response is recorded (finish, content, usage, "
          "x_yamadori)", json.dumps({k: v for k, v in out.items()
                                     if k != "x_yamadori"}))


def main() -> int:
    for fn in (test_request_shape, test_streamed_row, test_blocking_row):
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
                  + (f"   <- {detail[:400]}" if not ok and detail else ""))
    passed = sum(1 for ok, _, _ in _results if ok)
    total = len(_results)
    print(f"\n{'=' * 70}\n  {passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
