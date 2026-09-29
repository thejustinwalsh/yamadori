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
conversation and gets no tools, skills or fan-out).
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tiers  # noqa: E402
import cancel  # noqa: E402
import max_mode  # noqa: E402

UPSTREAM = os.environ.get("LLAMA_STACK_URL", "http://127.0.0.1:11434")
MODEL = os.environ.get("YAMADORI_MODEL", "bonsai")
# The model yama_describe_image asks when the MAIN model has no projector
# (mcp/vision.py, model=VISION_MODEL through chat(); the same tiers.apply
# shaping): `bonsai-vision`, the 27B + mmproj on the A4000, loaded on demand
# (config.yaml; mcp/gpu_room.py makes room). From 2026-09-27 the projector was
# folded into the main model and this was the main model; LAYOUT V2 (operator,
# 2026-09-29: "Vision can go to second card and swap in and out") moved it
# back. vision.main_sees() says whether the loaded main model has one (the
# max-mode model does): then it is asked instead. YAMADORI_VISION_MODEL set
# to the main model's name means no second copy (VISION_NO_PROJECTOR when
# the main model has none).
VISION_MODEL = os.environ.get("YAMADORI_VISION_MODEL", "bonsai-vision")
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


def shape(body: dict, effort: str = "low", role: str = "main",
          cap: int | None = None, step_cap: int | None = None,
          nudge: str | None = None) -> dict:
    """What the proxy would send for this body at this effort, minus the
    augmentations: the effort string the template accepts and the token
    budget derived from the context split. The caller's max_tokens is its
    ANSWER allowance; `role` picks the share its thinking comes out of --
    "helper" for deep thinking (3/8 of the pool), "main" otherwise (5/8).

    `cap` is tiers.budget's existing one-request thinking override (the
    `reasoning_cap` a benchmark sends in X-Yamadori-Features), for a caller
    whose server is not the one the shares describe: the vision model runs
    with its own -c 16384, so a thinking room derived from the main pool
    would not fit it (mcp/vision.py)."""
    tier = tiers.resolve({"reasoning_effort": effort},
                         {"reasoning_cap": int(cap)} if cap is not None else None)
    out = tiers.apply(body, tier, role=role, step_cap=step_cap, nudge=nudge)
    # MAX MODE (mcp/max_mode.py): the model of the request this thread works for -- a max conversation's second
    # brain, summaries and decider questions go to the max model; MODEL outside a request or with max mode off
    out.setdefault("model", max_mode.current(MODEL))
    if role == "helper":
        # post() counts this generation as the second brain's in the token
        # ledger (mcp/token_ledger.py). Underscored: never sent upstream.
        out["_share"] = "helper"
    return out


def post(body: dict, timeout: int = TIMEOUT) -> dict:
    """Send an already-shaped body. For the proxy's own loops, whose payload
    came from prepare() and must not be re-shaped."""
    send = {k: v for k, v in body.items() if not k.startswith("_")}
    # MAX MODE's backstop: never a request that would make llama-swap load a model max mode holds off the card
    # (raises max_mode.ModelAtCapacity: retryable; the worker defers on it, a turn answers 503 model_at_capacity)
    max_mode.guard(send.get("model", MODEL))
    # THE SECOND BRAIN'S SLOT (mcp/slots.py). Everything through here is the
    # stack's own generation -- deep thinking's hops, summarize_text -- and
    # left to itself llama-server gives it the least recently used idle slot,
    # which can be the one holding a conversation's cached prefix. Pinned to
    # slots.HELPER, an investigation's hops also reuse each other's prefix.
    # Not for another server (the vision copy has its own slots).
    grant = None
    if max_mode.is_main(send.get("model", MODEL)) and "id_slot" not in send:
        import slots
        grant = slots.acquire(slots.HELPER)
        if grant.get("slot") is not None:
            # id_slot, cache_prompt and the KV ranks (slots RANKS)
            send.update(slots.upstream_fields(grant))
            # IDLE CLEAR: a second-brain generation benefits like any other
            # (the request's switch and conversation come with its token).
            slots.clear_idle(grant)
    elif max_mode.is_main(send.get("model", MODEL)) and "kv_ranks" not in send \
            and isinstance(send.get("id_slot"), int):
        # A caller that placed the request itself (the skill decider on the
        # child slot): the slots' KV ranks as they stand now.
        import slots
        send.update(slots.rank_fields(send["id_slot"]))
    req = urllib.request.Request(
        f"{UPSTREAM}/v1/chat/completions", data=json.dumps(send).encode(),
        headers={"Content-Type": "application/json"})
    # THE A4000'S ROOM (mcp/gpu_room.py). A no-op for the main model (not on
    # the A4000); for the vision copy it makes room before llama-swap loads it
    # and holds the card until the answer is back. Raises gpu_room.NoRoom.
    import gpu_room
    try:
        # THE REQUEST'S CANCELLATION (mcp/cancel.py, #39): on a thread bound
        # to a request's token, this call's socket is registered with it, and
        # a client that goes away shuts it -- the read returns at once and
        # llama-server cancels the generation. Nothing starts once cancelled.
        cancel.check()
        with gpu_room.use(send.get("model", MODEL), upstream=UPSTREAM):
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.loads(r.read().decode("utf-8"))
    except Exception as e:
        if cancel.cancelled() and not isinstance(e, cancel.Cancelled):
            raise cancel.Cancelled(cancel.current().why) from e
        raise
    finally:
        if grant is not None:
            import slots
            slots.release(grant)
    # Every generation through the one door lands in the token ledger: deep
    # thinking (shape(role="helper")) as the second brain, everything else
    # as internal. A no-op unless this process called token_ledger.enable().
    import token_ledger
    token_ledger.record(
        "second_brain" if body.get("_share") == "helper" else "internal",
        account=str(body.get("_account") or ""),
        usage=d.get("usage") if isinstance(d, dict) else None,
        timings=d.get("timings") if isinstance(d, dict) else None)
    return d


# RELEASING A SLOT (mcp/slots.py RELEASE). Not a generation the caller reads,
# but a request to the model server all the same, so it goes through here.
RELEASE_TIMEOUT = float(os.environ.get("YAMADORI_SLOT_RELEASE_TIMEOUT", "10"))
# The shrink: a prompt that shares no token with anything a slot holds (every
# rendered prompt opens with `<|im_start|>`), so the slot's cache is cut at
# position 0 and it keeps one token.
SHRINK_PROMPT = "x"
# llama-server's erase: None until tried, False once this server refused it
# (501 without --slot-save-path; any 4xx), True once it worked.
_erase_ok: bool | None = None


def release_slot(slot: int, model: str | None = None,
                 timeout: float = RELEASE_TIMEOUT) -> dict:
    """Empty llama-server slot `slot` of `model`. Never raises.

    Reads /slots first: `n_prompt_tokens` is what the slot holds
    (`cells_before`), and a slot another process is generating on
    (`is_processing`) is left alone -- the server would defer the release
    behind that generation and this caller would wait for it. Then the
    server's erase when it allows it, else the shrink (SHRINK_PROMPT). Returns
    {ok, method: erase | shrink | None, cells_before, ms, skipped?, error?,
    erase_refused?}.

    Not tied to the request's cancellation (mcp/cancel.py): a release is
    sent for a request that has ended, and one cut half way leaves a slot
    nobody knows the state of."""
    import time as _time
    import token_ledger
    global _erase_ok
    t0 = _time.time()
    out: dict = {"ok": False, "method": None, "cells_before": None}
    model = model or max_mode.current(MODEL)
    if max_mode.blocks(model) and max_mode.bound_model() != model:
        # MAX MODE: releasing a slot of a model that is off the card would load it
        out.update(skipped=max_mode.blocked_reason(model), ms=round((_time.time() - t0) * 1000))
        return out
    base = f"{UPSTREAM}/upstream/{model}"

    def get(url):
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8") or "null")

    def send(url, body):
        req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={"Content-Type":
                                              "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8") or "{}")

    with cancel.bound(None):
        try:
            table = get(f"{base}/slots")
            row = next((s for s in table if isinstance(s, dict)
                        and s.get("id") == slot), None) \
                if isinstance(table, list) else None
            if row is None:
                out["error"] = "/slots did not list this slot"
            elif row.get("is_processing"):
                out["skipped"] = "processing: another request is on it"
            else:
                out["cells_before"] = int(row.get("n_prompt_tokens") or 0)
                if not out["cells_before"]:
                    out.update(ok=True, skipped="already empty")
        except Exception as e:                                   # noqa: BLE001
            out["error"] = f"/slots: {type(e).__name__}: {e}"[:200]
        if out.get("error") or out.get("skipped"):
            out["ms"] = round((_time.time() - t0) * 1000)
            return out
        if _erase_ok is not False:
            try:
                d = send(f"{base}/slots/{int(slot)}?action=erase", {})
                out.update(ok=True, method="erase")
                if isinstance(d, dict) and d.get("n_erased") is not None:
                    out["cells_before"] = d["n_erased"]
                _erase_ok = True
            except urllib.error.HTTPError as e:
                if 400 <= e.code < 500 or e.code == 501:
                    _erase_ok = False
                out["erase_refused"] = f"HTTP {e.code}"
            except Exception as e:                               # noqa: BLE001
                out["erase_refused"] = f"{type(e).__name__}: {e}"[:120]
        if not out["ok"]:
            try:
                d = send(f"{base}/completion",
                         {"prompt": SHRINK_PROMPT, "n_predict": 0,
                          "id_slot": int(slot), "cache_prompt": True})
                out.update(ok=True, method="shrink")
                # Our own cache management, counted like a warm (unpriced).
                token_ledger.record("warm", timings=(d or {}).get("timings"))
            except Exception as e:                               # noqa: BLE001
                out["error"] = f"shrink: {type(e).__name__}: {e}"[:200]
    out["ms"] = round((_time.time() - t0) * 1000)
    return out


def props(name: str | None = None, timeout: float = 10) -> dict:
    """llama-server's /props for a llama-swap model (default MODEL), through
    llama-swap's /upstream: `modalities` (vision.main_sees), n_ctx, the
    template. Raises on an unreachable server or a non-JSON answer."""
    name = name or max_mode.current(MODEL)
    max_mode.guard(name)
    with urllib.request.urlopen(f"{UPSTREAM}/upstream/{name}/props",
                                timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def chat(messages: list[dict], *, effort: str = "low",
         max_tokens: int | None = None, timeout: int = TIMEOUT,
         cap: int | None = None, **extra) -> dict:
    """One generation, shaped by the one rule. Returns the raw response.
    `model=` in `extra` picks the llama-swap model (default MODEL)."""
    body = dict(extra, messages=messages)
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    return post(shape(body, effort, cap=cap), timeout=timeout)


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
