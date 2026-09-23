#!/usr/bin/env python
"""The one door to the model. Every generation in this repo goes through here.

WHY THERE IS ONE DOOR

Failures that got whole mechanisms declared dead came from callers that each
set their own knobs. summarize_text sent max_tokens=900 to a model that
thinks for 1,000 tokens before answering, and returned nothing. The `high`
tier sent an effort string the chat template rejects, and every fan-out was
an instant 500. Each caller was "the model failing" to whoever read the
result, and each was a setting.

So there is one place that decides effort and budget for a generation --
`tiers.apply()`, the same function the proxy uses for every client request --
and one place that sends it, and one place that reports a length finish as a
budget event instead of an answer. Nothing else talks to the model.

WHY INTERNAL CALLERS DO NOT MAKE HTTP REQUESTS TO THE PROXY ON :1234

This was asked, and it has a concrete answer. The proxy's shaping IS this code;
the HTTP layer in front of it adds three things, and all three are wrong for a
call made from inside the stack:

  admission   summarize_text and the second context run INSIDE a client
              request that already holds one of MAIN_LANES=2. Re-entering
              :1234 asks for another. Two concurrent clients that each
              summarise hold both lanes and wait for each other -- a
              self-deadlock that ends in 429s after WAIT_SECONDS.
  corpus      every :1234 request is logged to index/corpus.sqlite3, the file
              the decision model trains on. Internal traffic would be
              synthetic turns in its training data (this has already
              happened once, from a test).
  auth        a background worker would need a stored key for its own server.

The rule is therefore: clients use :1234; code inside the stack uses
`model.chat()`, which applies exactly what :1234 applies to a request with the
same reasoning_effort, minus the augmentations (a helper call is not a
conversation and gets no tools, hints or fan-out).
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tiers  # noqa: E402

UPSTREAM = os.environ.get("LLAMA_STACK_URL", "http://127.0.0.1:11434")
MODEL = os.environ.get("YAMADORI_MODEL", "bonsai")
TIMEOUT = int(os.environ.get("YAMADORI_MODEL_TIMEOUT", "3600"))


class BudgetEvent(RuntimeError):
    """The model stopped on the token limit before writing a whole answer.

    Not "the model returned nothing" and not a transport error: a fact about
    the budget, with the numbers, so the caller can say so.
    """

    def __init__(self, content: str, usage: dict | None, reasoning_chars: int):
        self.content = content
        self.usage = usage or {}
        self.reasoning_chars = reasoning_chars
        n = self.usage.get("completion_tokens")
        super().__init__(
            f"the model reached its token limit (finish_reason=length) after "
            f"{n if n is not None else '?'} completion tokens, "
            f"{'with a partial answer' if content.strip() else 'before writing any answer'}")


def shape(body: dict, effort: str = "low", role: str = "main") -> dict:
    """What the proxy would send for this body at this effort, minus the
    augmentations: the effort string the template accepts and the token
    budget derived from the context split. The caller's max_tokens is its
    ANSWER allowance; `role` picks the share its thinking comes out of --
    "helper" for deep thinking (3/8 of the pool), "main" otherwise (5/8)."""
    tier = tiers.resolve({"reasoning_effort": effort})
    out = tiers.apply(body, tier, role=role)
    out.setdefault("model", MODEL)
    return out


def post(body: dict, timeout: int = TIMEOUT) -> dict:
    """Send an already-shaped body. For the proxy's own loops, whose payload
    came from prepare() and must not be re-shaped."""
    send = {k: v for k, v in body.items() if not k.startswith("_")}
    req = urllib.request.Request(
        f"{UPSTREAM}/v1/chat/completions", data=json.dumps(send).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def chat(messages: list[dict], *, effort: str = "low",
         max_tokens: int | None = None, timeout: int = TIMEOUT,
         **extra) -> dict:
    """One generation, shaped by the one rule. Returns the raw response."""
    body = dict(extra, messages=messages)
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    return post(shape(body, effort), timeout=timeout)


def answer(d: dict) -> str:
    """The content of a response, or BudgetEvent if it stopped on the limit.

    A length finish with a partial answer is still an event: a truncated
    summary or a half-written JSON array is not the thing that was asked for.
    """
    choice = (d.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content = msg.get("content") or ""
    if choice.get("finish_reason") == "length":
        raise BudgetEvent(content, d.get("usage"),
                          len(msg.get("reasoning_content") or ""))
    return content


def ask(messages: list[dict], **kw) -> str:
    """chat() then answer(): the common case for a helper call."""
    return answer(chat(messages, **kw))
