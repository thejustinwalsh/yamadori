#!/usr/bin/env python
"""The one door to the model, asserted against a fake upstream. No GPU.

What the door promises (mcp/model.py):
  1. Every internal generation is shaped by tiers.apply -- the same effort
     mapping and budget rule the proxy gives a client request.
  2. The caller's max_tokens is the ANSWER allowance; thinking is on top.
  3. A length finish raises BudgetEvent -- never returned as an answer and
     never reported as "the model returned nothing".
  4. Internal keys (leading underscore) never reach the model server.
  5. `role` picks the share thinking comes out of, and deep thinking
     (shomen._post) shapes with role="helper" -- 3/8 of the pool, never
     the conversation's 5/8 (mcp/budget.py, the split of 2026-09-22).

Expected token numbers are computed with tiers.budget rather than written as
constants, over a budget pinned to the shipped pool, so the suite needs no
server and follows the one rule rather than a copy of it.
"""
from __future__ import annotations

import http.server
import json
import os
import sys
import threading
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import budget  # noqa: E402
import model  # noqa: E402
import shomen  # noqa: E402
import tiers  # noqa: E402

tiers._accepted = tiers.FALLBACK_EFFORTS     # no network for the template
# budget.pool_size() would ask the live server for n_ctx. Pinned to the
# shipped `-c 147456`: main 92,160, the one helper 55,296.
POOL = 147456
_real_budgets = budget.budgets
budget.budgets = lambda pool=None: _real_budgets(pool or POOL)

_results: list[tuple[bool, str, str]] = []
SEEN: list[dict] = []
REPLY = {"finish": "stop", "content": "an answer"}


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


class _Upstream(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        SEEN.append(json.loads(self.rfile.read(n)))
        body = json.dumps({"choices": [{
            "finish_reason": REPLY["finish"],
            "message": {"role": "assistant", "content": REPLY["content"],
                        "reasoning_content": "thinking..."}}],
            "usage": {"completion_tokens": 42}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


_srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
threading.Thread(target=_srv.serve_forever, daemon=True).start()
model.UPSTREAM = f"http://127.0.0.1:{_srv.server_address[1]}"


def _expect(messages, answer, role="main", share_n=1):
    """What the one rule gives this request, from tiers.budget itself."""
    return tiers.budget(answer, True, role=role, share_n=share_n,
                        prompt_tokens=tiers.estimate_prompt_tokens(
                            {"messages": messages}))


def test_the_fixture_is_the_shipped_split():
    b = budget.budgets()
    check((b["pool"], b["main"], b["helper"]) == (POOL, 92160, 55296),
          "the pool is pinned, so nothing asks a server", str(b))


def test_a_helper_call_is_shaped_like_a_proxy_request():
    SEEN.clear()
    msgs = [{"role": "user", "content": "hi"}]
    out = model.ask(msgs, effort="low", max_tokens=300)
    sent = SEEN[-1]
    want = _expect(msgs, 300)
    check(out == "an answer", "ask() returns the content", repr(out))
    check(sent["max_tokens"] == want["max_tokens"]
          and sent["max_tokens"] - sent["reasoning_budget_tokens"]
          == tiers.A_MIN,
          "a 300-token answer request still gets a full answer allowance, "
          "with thinking derived from the main share on top",
          f"{sent['max_tokens']} vs {want['max_tokens']}")
    check(sent.get("reasoning_budget_tokens") == want["reasoning_budget_tokens"]
          and sent.get("reasoning_budget_message") == tiers.BUDGET_MESSAGE,
          "the derived thinking budget and its message are sent",
          str(sent.get("reasoning_budget_tokens")))
    check(sent["max_tokens"] + tiers.estimate_prompt_tokens(
              {"messages": msgs}) == budget.budgets()["main"],
          "and prompt + max_tokens is exactly the main share",
          str(sent["max_tokens"]))
    check(sent.get("reasoning_effort") in tiers.FALLBACK_EFFORTS,
          "the effort is one the template accepts", sent.get("reasoning_effort"))
    check(sent.get("model") == model.MODEL, "the internal model name is used")
    expect = tiers.apply({"messages": [{"role": "user", "content": "hi"}],
                          "max_tokens": 300},
                         tiers.resolve({"reasoning_effort": "low"}))
    check(all(sent.get(k) == expect.get(k) for k in
              ("max_tokens", "reasoning_effort", "enable_thinking",
               "reasoning_budget_tokens")),
          "and it is exactly what tiers.apply gives a client at that effort")


def test_a_large_answer_request_is_kept_whole():
    SEEN.clear()
    msgs = [{"role": "user", "content": "x"}]
    model.ask(msgs, max_tokens=16000)
    sent, want = SEEN[-1], _expect(msgs, 16000)
    check(sent["max_tokens"] == want["max_tokens"]
          and sent["max_tokens"] - sent["reasoning_budget_tokens"] == 16000,
          "16k of answer is 16k of answer, with thinking on top",
          f"{sent['max_tokens']} vs {want['max_tokens']}")


def test_shape_takes_its_thinking_from_the_role_share():
    body = {"messages": [{"role": "user", "content": "y" * 3000}],
            "max_tokens": 1000}
    main = model.shape(body, "xhigh")
    helper = model.shape(body, "xhigh", role="helper")
    for role, got in (("main", main), ("helper", helper)):
        want = _expect(body["messages"], 1000, role=role)
        check(got["max_tokens"] == want["max_tokens"]
              and got["reasoning_budget_tokens"]
              == want["reasoning_budget_tokens"],
              f"shape(role={role!r}) is tiers.budget at that role",
              f"{got['reasoning_budget_tokens']} vs "
              f"{want['reasoning_budget_tokens']}")
    check(helper["reasoning_budget_tokens"] < main["reasoning_budget_tokens"],
          "a helper thinks within its 3/8, less than main's 5/8",
          f"{helper['reasoning_budget_tokens']}/"
          f"{main['reasoning_budget_tokens']}")
    check(main == tiers.apply(body, tiers.resolve({"reasoning_effort": "xhigh"}))
          | {"model": model.MODEL},
          "and shape() is exactly tiers.apply plus the model name")
    check(body == {"messages": [{"role": "user", "content": "y" * 3000}],
                   "max_tokens": 1000}, "the caller's body is untouched")


def test_deep_thinking_posts_with_the_helper_share():
    """shomen._post is the second brain's only door to the model."""
    SEEN.clear()
    msgs = [{"role": "system", "content": "investigate"},
            {"role": "user", "content": "where is sizeKvPool defined?"}]
    d = shomen._post("/v1/chat/completions",
                     {"messages": msgs, "max_tokens": 1500})
    sent = SEEN[-1] if SEEN else {}
    want = _expect(msgs, 1500, role="helper")
    check(d["choices"][0]["message"]["content"] == "an answer",
          "shomen._post reaches the (fake) upstream through model.post")
    check(sent.get("max_tokens") == want["max_tokens"]
          and sent.get("reasoning_budget_tokens")
          == want["reasoning_budget_tokens"],
          "and shapes with role='helper': thinking from its 55,296 share",
          f"{sent.get('reasoning_budget_tokens')} vs "
          f"{want['reasoning_budget_tokens']}")
    check(sent.get("reasoning_budget_tokens")
          != _expect(msgs, 1500)["reasoning_budget_tokens"],
          "not the conversation's half")
    check(sent.get("reasoning_effort") == tiers.safe_effort("xhigh"),
          "at the xhigh effort deep thinking runs at",
          str(sent.get("reasoning_effort")))


def test_a_length_finish_is_a_budget_event():
    REPLY.update(finish="length", content="")
    try:
        raised = None
        try:
            model.ask([{"role": "user", "content": "x"}])
        except model.BudgetEvent as e:
            raised = e
        check(raised is not None, "an empty length finish raises BudgetEvent")
        check(raised is not None and "before writing any answer" in str(raised)
              and "42" in str(raised),
              "and says it stopped before answering, with the token count",
              str(raised))
        REPLY.update(content="half an ans")
        raised = None
        try:
            model.ask([{"role": "user", "content": "x"}])
        except model.BudgetEvent as e:
            raised = e
        check(raised is not None and raised.content == "half an ans",
              "a PARTIAL answer on a length finish is still an event, "
              "carrying what was written", repr(getattr(raised, "content", None)))
    finally:
        REPLY.update(finish="stop", content="an answer")


def test_internal_keys_never_reach_the_server():
    SEEN.clear()
    model.post({"model": "bonsai", "messages": [], "_hints": [1],
                "_account": "secret-account"})
    check("_hints" not in SEEN[-1] and "_account" not in SEEN[-1],
          "underscore keys are stripped before sending", json.dumps(SEEN[-1]))


def main() -> int:
    for fn in (test_the_fixture_is_the_shipped_split,
               test_a_helper_call_is_shaped_like_a_proxy_request,
               test_a_large_answer_request_is_kept_whole,
               test_shape_takes_its_thinking_from_the_role_share,
               test_deep_thinking_posts_with_the_helper_share,
               test_a_length_finish_is_a_budget_event,
               test_internal_keys_never_reach_the_server):
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
    passed = sum(1 for ok, _, _ in _results if ok)
    print(f"\n{'=' * 70}\n  {passed}/{len(_results)} checks passed")
    return 0 if passed == len(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
