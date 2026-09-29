#!/usr/bin/env python
"""The session ledger and the prefix-stability gate. No GPU.

    python mcp/test_ledger.py      -> "N/M checks passed"

WHAT THIS IS GATING (docs/SELF-IMPROVEMENT-PLAN.md Phase 0.5, operator
2026-09-24): one model, one cache.

  1. PREFIX STABILITY. A multi-turn session through the real complete(),
     with a client that strips everything the proxy added -- reasoning, the
     skills block on a user turn, image-tool hops -- EXTENDS, on every
     request, everything the slot holds: the previous request's prompt plus
     what it generated, reasoning included (PAST REASONING IS RESTORED,
     operator 2026-09-27, switch restore_reasoning), or, when the proxy
     warmed the slot (proxy._warm), the whole warmed prompt. The processed
     tail is the tool result + the new part. (Switch off, the 2026-09-24
     pass-through: the request reuses up to the previous turn's think block
     and processes that turn too; the prefilled-turn tests keep that arm.)
     The rendering is the SERVED
     chat template (mcp/fixtures/bonsai_chat_template.jinja, captured from
     llama-server /props on 2026-09-24, build b10738-285542d9), run by
     jinja2 -- STEP 0 measured that llama-server reuses a slot fully only
     when a request extends its sequence.
     The session covers: an injected skill on a user turn, a deep-thinking
     hand-off prefilled as reasoning (with the seed line), a client write
     repaired by the second brain and warmed, a yama_generate_image hop the
     client never sees (a hidden internal hop), an agent step, and an
     in-place compaction that THINKS at the conversation's effort. It runs
     at tier xhigh (medium thinking: no effort line) and max (xhigh: the
     effort line is rendered), so the compaction's cache check covers both.
  2. PERSISTENCE: a proxy restart (the memory layer and the compaction
     store emptied) costs nothing -- the next request still renders as the
     slot holds it, from sqlite.
  3. SCOPE AND SIZE: another account sees nothing; pruning drops whole
     sessions least-recently-seen first against the account cap, then the
     global cap, and anything past the max age.
  4. SEEDS: a replay of a request draws no new seed.
  5. THE BYTES: what each turn added to the ledger is printed, so the caps
     (choices) can be set from data.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

_TMP = tempfile.mkdtemp(prefix="yamadori_test_ledger_")
os.environ["YAMADORI_CORPUS_DB"] = os.path.join(_TMP, "corpus.sqlite3")
os.environ["YAMADORI_NEBARI_DB"] = os.path.join(_TMP, "nebari.sqlite3")
os.environ["RINGS_DB"] = os.path.join(_TMP, "rings.sqlite3")
os.environ["YAMADORI_PKG_DIR"] = os.path.join(_TMP, "pkgs")
os.makedirs(os.environ["YAMADORI_PKG_DIR"], exist_ok=True)
os.environ["CODE_INDEX_DB"] = os.path.join(_TMP, "code.sqlite3")
os.environ["CONCEPT_SEED_LAST"] = os.path.join(_TMP, "seed_last.json")
os.environ["YAMADORI_IMAGEGEN_URL"] = "http://127.0.0.1:9"
# Nothing here may reach the live model server: every door refuses or is
# the fake below.
os.environ["LLAMA_STACK_URL"] = "http://127.0.0.1:9"
os.environ["YAMADORI_MODEL_SERVER"] = "http://127.0.0.1:9"
os.environ["YAMADORI_SLOTS"] = "4"

import jinja2  # noqa: E402

# Every other store the code under test can write (the jobs database
# behind skill selection above all), BEFORE any mcp import: 2026-09-27 this
# suite (and test_sessions.py, which imports it) wrote the live index/.
import offline_stores  # noqa: E402
offline_stores.isolate("yamadori_test_ledger_stores_")

import compaction  # noqa: E402
import nebari  # noqa: E402
import proxy  # noqa: E402
import session_id  # noqa: E402
import shomen  # noqa: E402
import slots  # noqa: E402
import tiers  # noqa: E402
import model as _model  # noqa: E402

tiers._accepted = ("low", "medium", "xhigh")

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


# ------------------------------------------------------- the served template
_env = jinja2.Environment()
_env.filters["tojson"] = lambda v, **k: json.dumps(v, ensure_ascii=False)
_TEMPLATE = _env.from_string(open(os.path.join(
    HERE, "fixtures", "bonsai_chat_template.jinja"), encoding="utf-8").read())


def _raise(msg):
    raise ValueError(msg)


def _as_template_input(messages: list[dict]) -> list[dict]:
    """What llama-server hands the template: tool-call arguments parsed."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                fn = dict(c.get("function") or {})
                a = fn.get("arguments")
                if isinstance(a, str):
                    try:
                        fn["arguments"] = json.loads(a) if a.strip() else {}
                    except ValueError:
                        pass
                calls.append(dict(c, function=fn))
            m["tool_calls"] = calls
        if m.get("content") is None:
            m["content"] = ""
        out.append(m)
    return out


def _generated(body: dict, r: dict, p: str) -> str:
    """The slot's sequence after a generation: prompt `p` and the turn the
    fake generated (reply `r`, a prefill already folded in), as the template
    renders it; `p` + a character no prompt contains when the rendering of
    the turn does not extend `p`."""
    msgs = list(body.get("messages") or [])
    if msgs and msgs[-1].get("role") == "assistant":
        msgs = msgs[:-1]
    gen = {"role": "assistant", "content": r["content"],
           "reasoning_content": r["reasoning"]}
    if r["calls"]:
        gen["tool_calls"] = r["calls"]
    # render() leaves a trailing assistant turn open (a prefill): the
    # end-of-turn token the model generated is put back.
    full = render(dict(body, messages=msgs + [gen]), generation=False)
    if not full.startswith(p):
        return p + "\x00"
    return full + "<|im_end|>"


def render(body: dict, generation: bool = True) -> str:
    """The prompt llama-server builds for this request body. A trailing
    assistant message is a PREFILL: rendered, and left open (STEP 0 (a):
    /apply-template shows the turn without its end-of-turn token)."""
    msgs = _as_template_input(body.get("messages") or [])
    kw = {}
    if body.get("reasoning_effort"):
        kw["reasoning_effort"] = body["reasoning_effort"]
    ctk = body.get("chat_template_kwargs") or {}
    if "enable_thinking" in ctk:
        kw["enable_thinking"] = ctk["enable_thinking"]
    prefill = bool(msgs) and msgs[-1].get("role") == "assistant"
    text = _TEMPLATE.render(messages=msgs, tools=body.get("tools") or None,
                            add_generation_prompt=generation and not prefill,
                            raise_exception=_raise, **kw)
    if prefill and text.endswith("<|im_end|>\n"):
        text = text[:-len("<|im_end|>\n")]
    # A REASONING-ONLY prefill (a directive: proxy.directive_prefill) leaves
    # the think block OPEN -- the behaviour the proxy relies on, inferred
    # from STEP 0's /apply-template probe (docs/SELF-IMPROVEMENT-PLAN.md).
    last = msgs[-1] if msgs else {}
    if prefill and not (last.get("content") or "").strip() and \
            not last.get("tool_calls") and \
            (last.get("reasoning_content") or "").strip() and \
            text.endswith("\n</think>\n\n"):
        text = text[:-len("\n</think>\n\n")]
    return text


# --------------------------------------------------------- the fake upstream
# What the fake /completion (a warm) reports: 100 prompt tokens reused of a
# generation whose prompt was 100 (every fake generation reports 90 + 10).
WARM_TIMINGS = {"cache_n": 100, "prompt_n": 20}
WARM_DELAY = 0.0                  # seconds the fake /completion takes
# ONE TOKEN PER CHARACTER (the prefilled-turn warm test): a generation
# reports its rendered prompt's length as its prompt tokens, /tokenize one
# token per character, and a warm restores the server's checkpoint at the
# start of the last user message (llama.cpp server-context.cpp
# `last_user_pos`) -- or, "zero", re-reads everything (a real miss).
BY_CHARS: str | None = None       # None, "last_user", "zero" or "server"
_warm_done_at: list[float] = []   # when each fake /completion finished

# THE SERVER'S CHECKPOINTS (BY_CHARS = "server"; Octopus v0f-V0-xhigh-1 step
# 25, a hand-off prefilled mid-conversation). One token per character, and
# each slot keeps what it holds and its context checkpoints the way
# llama-server's server-context.cpp does (build b10738-285542d9, the one
# config.yaml runs): a request that diverges inside what the slot holds
# restores the latest checkpoint at or before the divergence (none: from
# zero) and erases the ones after it; processing makes a checkpoint at the
# start of a USER message (not a tool result) when it is the last one, the
# first, or more than min_step past the latest (chat requests only: a raw
# /completion carries no message delimiters), and near the prompt's end at
# E - (4 + n_ubatch) and E - 4 (or where processing starts, when that is
# already past them); and every creation first THINS: another task's
# checkpoint within min_step of the previous one kept is erased
# (create_checkpoint), then the oldest go past 32 (n_ctx_checkpoints).
SIM_MIN_STEP = 8192               # common.h checkpoint_min_step
SIM_UBATCH = 512                  # config.yaml -ub 512
SIM_MAX = 32                      # common.h n_ctx_checkpoints
# "thinned" (the server), or "old": a MISS -- it resumes at its second-oldest
# checkpoint (far back) instead of the latest usable one.
SIM_MODE = "thinned"
_sim: dict = {}
_sim_log: list[dict] = []         # one row per request the fake served


def _sim_user_starts(prompt: str) -> list[int]:
    out, i = [], prompt.find("<|im_start|>user")
    while i >= 0:
        if not prompt.startswith("<|im_start|>user\n<tool_response>", i):
            out.append(i)
        i = prompt.find("<|im_start|>user", i + 1)
    return out


def _sim_serve(slot, prompt: str, chat: bool, kind: str) -> tuple[int, int]:
    """(reused, processed) for one request on `slot`, and the slot's new
    checkpoints."""
    s = _sim.setdefault(slot, {"tokens": "", "ckpts": [], "task": 0})
    _sim["_task"] = task = _sim.get("_task", 0) + 1
    held = s["tokens"]
    n = 0
    for a, b in zip(held, prompt):
        if a != b:
            break
        n += 1
    if n < len(held):
        usable = [c for c in s["ckpts"] if c[0] <= n]
        if SIM_MODE == "old" and kind == "warm" and len(usable) > 1:
            usable = usable[:2]       # a miss: resumed far back
        n = usable[-1][0] if usable else 0
        s["ckpts"] = [c for c in s["ckpts"] if c[0] <= n]
    E = len(prompt)
    n = min(n, E - 1)

    def create(p: int) -> None:
        keep, last = [], -1
        for c in s["ckpts"]:
            if c[1] != task and last >= 0 and c[0] <= last + SIM_MIN_STEP:
                continue
            keep.append(c)
            last = c[0]
        while len(keep) >= SIM_MAX:
            keep.pop(0)
        keep.append((p, task))
        s["ckpts"] = keep

    users = _sim_user_starts(prompt) if chat else []
    for u in users:
        if n < u < E - SIM_UBATCH - 4 and (
                u == users[-1] or not s["ckpts"]
                or u > s["ckpts"][-1][0] + SIM_MIN_STEP):
            create(u)
    ends = [E - SIM_UBATCH - 4, E - 4]
    starts = [p for p in ends if p >= n]
    if len(starts) < len(ends):
        starts = [n] + starts if n < E - 4 else [n]
    for p in starts:
        create(p)
    s["tokens"] = prompt
    _sim_log.append({"kind": kind, "slot": slot, "reused": n,
                     "held": len(held), "extends": prompt.startswith(held),
                     "processed": E - n, "prompt": E,
                     "ckpts": [c[0] for c in s["ckpts"]]})
    return n, E - n
_script: list[dict] = []
_gens: list[dict] = []            # {request, reply} per generation
_warms: list[dict] = []           # /completion bodies
_tokenized: list[dict] = []       # /tokenize bodies
class _Archived(list):
    """A capture list whose clear() keeps what it held, for the final scan
    (test_no_removed_steering_reaches_any_request)."""

    def __init__(self):
        super().__init__()
        self.archive: list = []

    def clear(self):
        self.archive.extend(self)
        super().clear()


_helper: list[dict] = _Archived()  # the second brain's requests
_helper_direct: list[dict] = _Archived()  # ... through model.post itself


def reply(content: str = "", reasoning: str = "", calls: list | None = None
          ) -> dict:
    return {"content": content, "reasoning": reasoning, "calls": calls or []}


def call(name: str, args: dict, cid: str) -> dict:
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def _sse(r: dict, finish: str, timings: dict | None = None) -> bytes:
    def ev(delta=None, fin=None):
        return (b"data: " + json.dumps({
            "id": "u", "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": delta or {},
                         "finish_reason": fin}]}).encode() + b"\n\n")
    out = [ev({"role": "assistant"})]
    if r["reasoning"]:
        out.append(ev({"reasoning_content": r["reasoning"]}))
    s = r["content"]
    out += [ev({"content": s[i:i + 40]}) for i in range(0, len(s), 40)]
    for i, c in enumerate(r["calls"]):
        out.append(ev({"tool_calls": [dict(c, index=i)]}))
    out.append(ev({}, finish))
    out.append(b"data: " + json.dumps({"id": "u", "choices": [], "usage": {
        "prompt_tokens": 100, "completion_tokens": 10,
        "total_tokens": 110}, "timings": timings or {
            "cache_n": 90, "prompt_n": 10}}).encode() + b"\n\n")
    return b"".join(out) + b"data: [DONE]\n\n"


class _Up(BaseHTTPRequestHandler):
    def _json(self, obj: dict) -> None:
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):                                            # noqa: N802
        self.send_response(404)
        self.end_headers()

    def do_POST(self):                                           # noqa: N802
        arrived = time.time()
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path.endswith("/apply-template"):
            return self._json({"prompt": render(body)})
        if self.path.endswith("/tokenize"):
            _tokenized.append(body)
            return self._json({"tokens": [1] * len(body.get("content") or "")})
        if self.path.endswith("/completion"):
            _warms.append(body)
            if WARM_DELAY:
                time.sleep(WARM_DELAY)
            _warm_done_at.append(time.time())
            if BY_CHARS == "server":
                n, k = _sim_serve(body.get("id_slot"), body.get("prompt") or "",
                                  chat=False, kind="warm")
                return self._json({"timings": {"cache_n": n, "prompt_n": k}})
            if BY_CHARS:
                p = body.get("prompt") or ""
                n = p.rfind("<|im_start|>user") if BY_CHARS == "last_user" \
                    else 0
                return self._json({"timings": {"cache_n": n,
                                               "prompt_n": len(p) - n}})
            return self._json({"timings": dict(WARM_TIMINGS)})
        if self.path.endswith("/chat/completions") and not body.get("stream"):
            # The second brain through its real door (mcp/model.py posts a
            # non-streamed request): recorded with the slot it was sent to.
            _helper_direct.append(body)
            return self._json({"choices": [{"message": {
                "content": "```python\n" + FIXED + "```"},
                "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}})
        if not _script:
            self.send_response(500)
            self.end_headers()
            return
        r = dict(_script.pop(0))
        last = (body.get("messages") or [{}])[-1]
        if last.get("role") == "assistant":
            # llama-server re-sends a prefill's reasoning and content first.
            r["content"] = (last.get("content") or "") + r["content"]
            if last.get("reasoning_content"):
                r["reasoning"] = last["reasoning_content"] + "\n" \
                    + r["reasoning"]
        _gens.append({"request": body, "reply": r, "at": arrived})
        timings = None
        if BY_CHARS == "server":
            p = render(body)
            n, k = _sim_serve(body.get("id_slot"), p, chat=True, kind="gen")
            # What it generated follows the prompt: the turn as the served
            # template renders it, up to its end-of-turn token (the "\n"
            # after it is the next request's). A later prompt shares it only
            # as far as it renders the same turn -- all of it when past
            # reasoning is restored, up to the think block when it is not.
            _sim[body.get("id_slot")]["tokens"] = _generated(body, r, p)
            timings = {"cache_n": n, "prompt_n": k}
        elif BY_CHARS:
            timings = {"cache_n": 0, "prompt_n": len(render(body))}
        data = _sse(r, "tool_calls" if r["calls"] else "stop", timings)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):                                   # noqa: D102
        pass


_srv = ThreadingHTTPServer(("127.0.0.1", 0), _Up)
threading.Thread(target=_srv.serve_forever, daemon=True).start()
proxy.UPSTREAM = f"http://127.0.0.1:{_srv.server_address[1]}"
_model.UPSTREAM = "http://127.0.0.1:9"


# ------------------------------------------------------------------ stubs
SEEDS = iter(["cedar", "juniper", "larch", "alder", "rowan", "hazel",
              "maple", "willow"] * 10)
proxy._draw_seed = lambda prompt=None: {"word": next(SEEDS), "token_id": 1,
                                        "u32": 1}
HINT = "\n\n---\nSkill: prefer small diffs."


def _skills_tail(messages, sel, route, client_tools=None, account=""):
    return HINT, {"on": True, "ids": ["s1"], "names": ["s1"],
                  "chars": len(HINT), "versions": [1], "why": "test"}


proxy._skills_tail = _skills_tail
HANDOFF = ("FACTS\n- LatheGeometry sweeps a profile around the Y axis.\n"
           "SEARCHED, FOUND NOTHING\n- none\nOPEN QUESTIONS\n- none\n"
           "NEXT STEP\n- none\n")
BROKEN = "def f(:\n    return 1\n"
FIXED = "def f():\n    return 1\n"


def _helper_post(path, payload, timeout=3600):
    _helper.append(json.loads(json.dumps(payload)))
    user = payload["messages"][-1]["content"]
    text = (f"```python\n{FIXED}```" if "does not parse" in user else HANDOFF)
    return {"choices": [{"message": {"content": text},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


_REAL_SHOMEN_POST = shomen._post
shomen._post = _helper_post
_real_run_our_tool = proxy.run_our_tool


def _run_our_tool(name, args, db, root=None, turn=None, state=None):
    if name == "yama_generate_image":
        return "![a lathe](https://media.example/abc.png)"
    return _real_run_our_tool(name, args, db, root, turn, state)


proxy.run_our_tool = _run_our_tool

WRITE = {"type": "function", "function": {
    "name": "write_file", "description": "Write a file.",
    "parameters": {"type": "object", "properties": {
        "path": {"type": "string"}, "content": {"type": "string"}}}}}
SYSTEM = "You are a coding agent."
ACCOUNT = "ledger-acct"
# X-Yamadori-Features every Client request carries: {"restore_reasoning":
# False} runs a test on the 2026-09-24 pass-through arm (switch off).
EXTRA_FEATURES: dict = {}


def restoring() -> bool:
    """Is past reasoning restored for the Client requests (switch
    `restore_reasoning`, default on since 2026-09-27)?"""
    return tiers.behaviour_of_header(json.dumps(EXTRA_FEATURES),
                                     "restore_reasoning")[0]


class Client:
    """A harness that STRIPS: it keeps what it was sent as content and tool
    calls, drops reasoning, and never sees what the proxy added."""

    def __init__(self, effort: str):
        self.effort = effort
        self.msgs = [{"role": "system", "content": SYSTEM}]
        # script call id -> the id the proxy returned for it (#41).
        self.ids: dict[str, str] = {}

    def body(self, features: dict | None = None) -> dict:
        b = {"model": "yamadori", "reasoning_effort": self.effort,
             "_account": ACCOUNT, "_client_ip": "127.0.0.1",
             "tools": [WRITE], "messages": json.loads(json.dumps(self.msgs))}
        feats = {"skills": True, "investigate": False, "fanout": 1}
        feats.update(EXTRA_FEATURES)
        feats.update(features or {})
        b["_features"] = json.dumps(feats)
        return b

    def turn(self, script: list[dict], user: str | None = None,
             features: dict | None = None) -> dict:
        if user is not None:
            self.msgs.append({"role": "user", "content": user})
        _script[:] = list(script)
        n0, w0 = len(_gens), len(_warms)
        d = proxy.complete(self.body(features))
        m = d["choices"][0]["message"]
        kept = {"role": "assistant", "content": m.get("content") or ""}
        if m.get("tool_calls"):
            kept["tool_calls"] = m["tool_calls"]
            # The proxy returns ids that carry the conversation's id (#41,
            # proxy._carry_session); a test names a call by the id its
            # script gave it, so tool_result echoes the returned one.
            gen = _gens[-1]["reply"] if len(_gens) > n0 else {"calls": []}
            mine = [c for c in gen["calls"]
                    if c["function"]["name"] not in proxy.OUR_NAMES
                    and c["function"]["name"] not in proxy.LEGACY_TOOL_NAMES]
            for a, b in zip(mine, m["tool_calls"]):
                self.ids[a["id"]] = b["id"]
        self.msgs.append(kept)
        warm = (d.get("x_yamadori") or {}).get("warm") or {}
        if warm.get("sent"):
            end = time.time() + 5
            while time.time() < end and len(_warms) == w0:
                time.sleep(0.02)
        return {"d": d, "gens": _gens[n0:], "warm": _warms[w0:]}

    def tool_result(self, cid: str, text: str) -> None:
        self.msgs.append({"role": "tool",
                          "tool_call_id": self.ids.get(cid, cid),
                          "content": text})


def holds_after(t: dict) -> str:
    """What the slot holds after a turn: the warmed prompt, else the last
    generation's prompt (its prefill closed into the generated turn) and
    what it generated."""
    if t["warm"]:
        return t["warm"][-1]["prompt"]
    g = t["gens"][-1]
    req = dict(g["request"])
    msgs = list(req.get("messages") or [])
    if msgs and msgs[-1].get("role") == "assistant":
        msgs = msgs[:-1]
    r = g["reply"]
    gen = {"role": "assistant", "content": r["content"],
           "reasoning_content": r["reasoning"]}
    if r["calls"]:
        gen["tool_calls"] = r["calls"]
    return render(dict(req, messages=msgs + [gen]), generation=False)


def first_render(t: dict) -> str:
    return render(t["gens"][0]["request"])


_OPEN = "<|im_start|>assistant\n<think>\n"


def _landed(t: dict) -> bool:
    """The turn's last generation LANDED (tools withdrawn, the landing
    request appended): the slot holds a prompt the client's next request
    never renders (the warm says so and is not sent)."""
    req = (t["gens"][-1] if t.get("gens") else {}).get("request") or {}
    last = (req.get("messages") or [{}])[-1]
    first = (t["gens"][0]["request"] if t.get("gens") else {})
    return last.get("content") == proxy.LANDING_PROMPT or (
        bool(first.get("tools")) and not req.get("tools"))


def _full_extension(t: dict) -> bool:
    """The next request must extend everything the slot holds after `t`:
    reasoning restored, and the turn did not land."""
    return restoring() and not _landed(t)


def reusable_after(t: dict) -> str:
    """What the next request may reuse.

    PAST REASONING RESTORED (switch `restore_reasoning`, default since
    2026-09-27): everything the slot holds -- the warmed prompt when the
    proxy warmed the slot, else the last generation's prompt PLUS what it
    generated, reasoning included (holds_after). The next request EXTENDS
    it; the processed tail is the tool result and the new part only.

    PASS-THROUGH (switch off, the 2026-09-24 design): the warmed prompt, else
    the FIRST generation's prompt of the turn up to its opening think block
    -- the slot generated the reasoning, the client sends none (or its echo),
    so the next request diverges there and rests on the checkpoint at that
    prompt's end. A LANDED turn is judged that way too, whatever the
    switch: its last generation ran with the tools withdrawn, which no
    client request renders (_landed)."""
    if t["warm"]:
        return t["warm"][-1]["prompt"]
    if _full_extension(t):
        return holds_after(t)
    p = render(t["gens"][0]["request"])
    i = p.rfind(_OPEN)
    return p[:i + len(_OPEN)] if i >= 0 else p


def _extends(prev: dict, nxt: dict, label: str) -> bool:
    h, r = reusable_after(prev), first_render(nxt)
    n = next((i for i, (a, b) in enumerate(zip(h, r)) if a != b),
             min(len(h), len(r)))
    tail = r[len(h):] if r.startswith(h) else ""
    if _full_extension(prev) or (prev["warm"] and restoring()):
        return check(r.startswith(h),
                     f"{label}: the request EXTENDS what the slot holds "
                     f"({len(h)} chars: the previous prompt + its generation, "
                     f"reasoning restored); processed tail = the tool result "
                     f"+ the new part ({len(tail)} chars)",
                     f"diverges at char {n} of {len(h)}: held "
                     f"{h[max(0, n - 80):n + 80]!r} vs sent "
                     f"{r[max(0, n - 80):n + 80]!r}")
    # The tail must hold nothing from before the last assistant turn: it
    # opens inside that turn's think block (or, after a warm, right after
    # the warmed turn).
    before_turn = tail.split("<|im_end|>", 1)[0]
    ok = r.startswith(h) and (bool(prev["warm"]) or "<|im_start|>" not in
                              before_turn)
    return check(ok,
                 f"{label}: the request reuses everything before the last "
                 f"assistant turn ({len(h)} chars); processed tail = that "
                 f"turn + tool result + the new part ({len(tail)} chars)",
                 f"diverges at char {n} of {len(h)}: held "
                 f"{h[max(0, n - 80):n + 80]!r} vs sent "
                 f"{r[max(0, n - 80):n + 80]!r}")


def _bytes() -> int:
    return int(nebari.ledger_stats().get("bytes") or 0)


SIZES: list[tuple[str, int]] = []


def session(effort: str) -> None:
    slots.reset(n=4)
    compaction.reset()
    c = Client(effort)
    tag = f"[{effort}]"

    # 1. A library question, deep thinking forced: the hand-off is prefilled
    #    as main's reasoning; the visible answer opens with the seed line.
    b0 = _bytes()
    t1 = c.turn([reply(" it sweeps the profile around the Y axis.",
                       reasoning="")],
                user="How does LatheGeometry build its points?",
                features={"investigate": True})
    SIZES.append((f"{tag} turn 1 (skill + deep thinking)", _bytes() - b0))
    pre = t1["gens"][0]["request"]["messages"][-1]
    # A new conversation's answer carries NO session line (#41, operator
    # 2026-09-25): its id rides in tool-call ids, and this text-only
    # answer is recorded with it (proxy THE ANSWER RECORD).
    ans = t1["d"]["choices"][0]["message"]["content"]
    sid = (t1["d"]["x_yamadori"].get("session") or {}).get("id")
    check(sid and session_id.PREFIX not in ans,
          f"{tag} a new conversation's first answer has an id and no "
          f"session line", ans[:60])
    word = (t1["d"]["x_yamadori"]["investigate"].get("seed") or {}).get("word")
    check(pre.get("role") == "assistant" and "LatheGeometry sweeps"
          in pre.get("reasoning_content", "")
          and ans.startswith(f"Today I was inspired by {word}. After thinking "
                             f"deeply, it sweeps"),
          f"{tag} the hand-off is prefilled as reasoning; the answer opens "
          f"with the seed line and the phrase", ans[:120])
    check(any(shomen.SEED_LINE_RESEARCH.format(word=word)
              in m.get("content", "")
              for h in _helper for m in h["messages"]
              if m.get("role") == "user"),
          f"{tag} the investigate job's USER message names the same seed, "
          f"in the research frame (#52: random, not a clue)")
    check(HINT in t1["gens"][0]["request"]["messages"][1]["content"],
          f"{tag} the skill rides on the user turn it served")

    # 2. A client write with broken code: repaired by the second brain,
    #    delivered with the note, the slot warmed with the delivered turn.
    b0 = _bytes()
    t2 = c.turn([reply("Writing it.", calls=[call(
        "write_file", {"path": "hi.py", "content": BROKEN}, "w1")])],
        user="Write hi.py with a function f that returns 1.")
    SIZES.append((f"{tag} turn 2 (repaired write + warm)", _bytes() - b0))
    _extends(t1, t2, f"{tag} turn 2 after turn 1 (prefill, stripped "
                     f"reasoning, stripped skill)")
    check(t2["warm"] and "Repaired hi.py" in t2["warm"][-1]["prompt"],
          f"{tag} the warm loads the repaired turn")
    c.tool_result("w1", "wrote hi.py")

    # 3. The agent step after the tool: a plain answer.
    b0 = _bytes()
    t3 = c.turn([reply("Done: hi.py is written.", reasoning="It worked.")])
    SIZES.append((f"{tag} turn 3 (agent step)", _bytes() - b0))
    _extends(t2, t3, f"{tag} turn 3 after the warmed repair")

    # A PROXY RESTART: the memory layer and the compaction store are gone.
    nebari.ledger_reset()
    compaction.reset()

    # 4. An image: a hidden internal hop (yama_generate_image) the client never
    #    sees, then the answer.
    b0 = _bytes()
    t4 = c.turn([reply("", reasoning="Draw it.",
                       calls=[call("yama_generate_image",
                                   {"prompt": "a lathe"}, "g1")]),
                 reply("Here is the lathe.", reasoning="Show it.")],
                user="Draw a lathe.")
    SIZES.append((f"{tag} turn 4 (image hop)", _bytes() - b0))
    _extends(t3, t4, f"{tag} turn 4 after a proxy RESTART (from sqlite)")
    check(len(t4["gens"]) == 2 and not any(
              m.get("tool_calls") for m in c.msgs[-1:]),
          f"{tag} the client never sees the image hop")

    # 5. The next user turn: the hidden hop is put back, in place.
    b0 = _bytes()
    t5 = c.turn([reply("You are welcome.")], user="Thanks.")
    SIZES.append((f"{tag} turn 5 (plain)", _bytes() - b0))
    _extends(t4, t5, f"{tag} turn 5 after the hidden image hop")

    # 6. An in-place compaction: part of the conversation, thinking at its
    #    own effort -- the same effort line, so the prefix is reused.
    t6 = c.turn([reply("## Goal\nDrew a lathe; wrote hi.py.")],
                user="Your task is to create a detailed summary of the "
                     "conversation so far.")
    _extends(t5, t6, f"{tag} the compaction after turn 5")
    x = t6["d"]["x_yamadori"]
    req = t6["gens"][0]["request"]
    check(x["utility_kind"] == "compaction"
          and x["compaction"]["mode"] in ("ledger", "spliced")
          and req.get("enable_thinking") is True
          and req.get("reasoning_budget_tokens")
          == x["compaction"]["thinking_tokens"],
          f"{tag} the compaction thinks at the conversation's effort "
          f"({req.get('reasoning_effort')}), what its window leaves of it", json.dumps({k: req.get(k) for k in (
              "enable_thinking", "reasoning_effort",
              "reasoning_budget_tokens")}))

    # The same OPENING again (a retry, or a new conversation that opens
    # alike): a NEW conversation, always (#41) -- its own id, its own seed,
    # nothing of turn 1's replayed. (A replay WITHIN a conversation reuses
    # its seed: the warm and the compaction splice, above.)
    retry = Client(effort)
    retry.msgs = c.msgs[:2]
    t1b = retry.turn([reply(" it sweeps.")], features={"investigate": True})
    word_b = (t1b["d"]["x_yamadori"]["investigate"].get("seed") or {}).get(
        "word")
    sid_b = (t1b["d"]["x_yamadori"].get("session") or {}).get("id")
    check(sid_b and sid_b != sid and word_b and word_b != word
          and t1b["d"]["x_yamadori"]["session"]["source"] == "minted",
          f"{tag} the same opening again is a new conversation: a new id, a "
          f"fresh seed", f"{sid_b} vs {sid}; {word_b} vs {word}")


def _stream(body: dict) -> dict:
    """proxy.stream_body read as a streaming client reads it: the content,
    reasoning and tool calls it would store."""
    out = {"content": "", "reasoning": "", "calls": []}
    for b in proxy.stream_body(dict(body, stream=True)):
        for line in b.decode("utf-8").split("\n"):
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            ev = json.loads(line[5:].strip())
            for ch in ev.get("choices") or []:
                dl = ch.get("delta") or {}
                out["content"] += dl.get("content") or ""
                out["reasoning"] += dl.get("reasoning_content") or ""
                for c in dl.get("tool_calls") or []:
                    out["calls"].append({k: v for k, v in c.items()
                                         if k != "index"})
    return out


def test_a_streamed_hop_after_a_prefill_is_restored():
    """#10 (docs/SELF-IMPROVEMENT-LOG.md). Live gate 2026-09-24: a deep-
    thinking turn whose FIRST generation (the prefill) streamed its opening
    and a long preface as content, then called one of OUR tools, and the
    answer came on a later hop (7 generations). The client stored every
    content byte it was streamed; the ledger had keyed the turn by the last
    hop's content alone, so the hand-off reasoning and the hidden hop were
    never restored and the next request diverged at the start of the turn
    (reused 2513 of 3955: everything before it). The next request must
    extend what the slot holds, and the client's copy must be what keys it."""
    slots.reset(n=4)
    compaction.reset()
    msgs = [{"role": "system", "content": SYSTEM + " [hop session]"},
            {"role": "user", "content": "How does LatheGeometry build its "
                                        "points? Draw it too."}]
    preface = " it sweeps the profile around the Y axis." * 12   # > 400
    _script[:] = [reply(preface, calls=[call("yama_generate_image",
                                             {"prompt": "a lathe"}, "g9")]),
                  reply(" Here is the drawing.", reasoning="Show it.")]
    n0 = len(_gens)
    body = {"model": "yamadori", "reasoning_effort": "xhigh",
            "_account": ACCOUNT, "_client_ip": "127.0.0.1", "tools": [WRITE],
            "messages": json.loads(json.dumps(msgs)),
            "_features": json.dumps({"skills": True, "investigate": True,
                                     "fanout": 1})}
    got = _stream(body)
    t1 = {"gens": _gens[n0:], "warm": []}
    check(len(t1["gens"]) == 2 and preface.strip() in got["content"]
          and "Here is the drawing." in got["content"],
          "[hop] the client was streamed the prefill hop's content AND the "
          "answer (the shape of the live 7-generation turn)",
          f"gens={len(t1['gens'])} content={got['content'][:120]!r}")
    msgs.append({"role": "assistant", "content": got["content"]})
    msgs.append({"role": "user", "content": "Thanks."})
    _script[:] = [reply("You are welcome.")]
    n1 = len(_gens)
    _stream(dict(body, messages=json.loads(json.dumps(msgs)),
                 _features=json.dumps({"skills": True, "investigate": False,
                                       "fanout": 1})))
    t2 = {"gens": _gens[n1:], "warm": []}
    _extends(t1, t2, "[hop] the request after a streamed prefill + hidden "
                     "hop turn")
    x2 = t2["gens"][0]["request"]["messages"]
    check(any(m.get("role") == "tool" and m.get("tool_call_id") == "g9"
              for m in x2),
          "[hop] the hidden yama_generate_image hop is restored from the ledger")


def _think_blocks_balanced(text: str) -> bool:
    """Every assistant turn in a rendered prompt carries exactly one
    <think>...</think> pair, in order (the generation prompt's open <think>
    at the very end excepted)."""
    turns = text.split("<|im_start|>assistant\n")[1:]
    for i, t in enumerate(turns):
        body = t.split("<|im_end|>")[0]
        last = i == len(turns) - 1 and "<|im_end|>" not in t
        if last and body == "<think>\n":
            continue
        if body.count("<think>") != 1 or body.count("</think>") != 1 \
                or body.index("<think>") > body.index("</think>"):
            return False
    return True


def test_the_echo_path_renders_like_the_strip_path():
    """#12 (docs/SELF-IMPROVEMENT-LOG.md): the gate's ECHO run delivered
    'Done.\\n</think>\\n\\nDone.\\n</think>\\n\\nDone.' as its final answer,
    the strip run a clean 'Done.'. Suspects: the echoed reasoning carrying
    our markers or trace, the ledger restoring reasoning that contains
    '</think>', a prefill rendered twice. Rendered here with the served
    template: an echoing client (it sends back every reasoning byte it was
    shown, our trace lines included) and a stripping one, the same scripted
    agent loop -- every upstream request must be identical between the two,
    with one balanced think block per assistant turn."""
    renders: dict[str, list[str]] = {}
    draw = proxy._draw_seed
    # One seed for both runs: the fix-up's seed line is in the delivered
    # content, and only the echo is being compared.
    proxy._draw_seed = lambda prompt=None: {"word": "cedar", "token_id": 1,
                                            "u32": 1}
    try:
        _echo_runs(renders)
    finally:
        proxy._draw_seed = draw
    # SINCE 2026-09-27 PAST REASONING IS RESTORED (operator; the 2026-09-24
    # pass-through is switch restore_reasoning off): the stripping client's
    # second request renders the SLOT's reasoning, put back by the ledger;
    # the echoing client's renders ITS echo, as sent.
    s2 = renders["strip"][1].split("[agent loop]", 1)[1]
    e2 = renders["echo"][1].split("[agent loop]", 1)[1]
    check("<think>\nWrite it.\n</think>" in s2 and "fixing hi.py" not in s2,
          "[echo] a stripping client's past turn renders the slot's own "
          "reasoning (restored), not the trace it was shown")
    check("Write it." in e2 and "fixing hi.py" in e2,
          "[echo] an echoing client's past turn renders its echo, our trace "
          "line included, as sent")
    check(all(_think_blocks_balanced(t) for t in renders["echo"]
              + renders["strip"]),
          "[echo] one balanced think block per assistant turn in every "
          "request, either client")


def _echo_runs(renders: dict) -> None:
    for mode in ("strip", "echo"):
        slots.reset(n=4)
        msgs = [{"role": "system", "content": SYSTEM + " [agent loop]"},
                {"role": "user", "content": "Write hi.py with a function f "
                                            "that returns 1, then say done."}]
        script = [
            reply("", reasoning="Write it.", calls=[call(
                "write_file", {"path": "hi.py", "content": BROKEN}, "e1")]),
            reply("Done.", reasoning="It is written.")]
        out = []
        for step, r in enumerate(script):
            _script[:] = [r]
            n0, w0 = len(_gens), len(_warms)
            got = _stream({"model": "yamadori", "reasoning_effort": "xhigh",
                           "_account": ACCOUNT + "-" + mode,
                           "_client_ip": "127.0.0.1", "tools": [WRITE],
                           "messages": json.loads(json.dumps(msgs)),
                           "_features": json.dumps({"skills": False,
                                                    "investigate": False,
                                                    "fanout": 1})})
            out.append(render(_gens[n0]["request"]))
            m = {"role": "assistant", "content": got["content"]}
            if got["calls"]:
                m["tool_calls"] = got["calls"]
            if mode == "echo" and got["reasoning"]:
                m["reasoning_content"] = got["reasoning"]
            msgs.append(m)
            for c in got["calls"]:
                msgs.append({"role": "tool", "tool_call_id": c["id"],
                             "content": "ok"})
            end = time.time() + 5
            while time.time() < end and len(_warms) == w0 and step == 0:
                time.sleep(0.02)
        renders[mode] = out
        if mode == "echo":
            check(any("fixing" in (m.get("reasoning_content") or "")
                      for m in msgs if m.get("role") == "assistant"),
                  "[echo] the echoing client really echoed our trace line "
                  "(`fixing hi.py ...`) as reasoning")


def test_template_markers_are_scrubbed_on_our_path_and_recorded_from_the_model():
    """#12, the guard. OUR path never puts the template's markers into a
    prompt or into content; stray markers the MODEL wrote in its answer are
    removed from the client's copy (with the repeat after them) and
    recorded (x_yamadori.template_markers), not hidden (operator,
    2026-09-25, reversing "delivered as written"; mcp/test_stray_markers.py
    has the filter's own gate)."""
    global HANDOFF
    # 1. The model writes them: stripped from the client's copy, recorded as
    #    the model's.
    slots.reset(n=4)
    c = Client("xhigh")
    c.msgs[0]["content"] += " [markers 1]"
    t = c.turn([reply("Done.\n</think>\n\nDone.\n</think>\n\nDone.",
                      reasoning="Done.")], user="Say done.")
    x = t["d"]["x_yamadori"]
    tm = x.get("template_markers") or {}
    got = t["d"]["choices"][0]["message"]["content"]
    check(got.endswith("Done.") and "</think>" not in got
          and got.count("Done.") == 1
          and tm.get("source") == "model"
          and tm.get("stripped", {}).get("</think>") == 2
          and tm.get("model", {}).get("</think>") == 2 and not tm.get("ours"),
          "stray markers the model wrote are stripped from the client's copy "
          "(the repeats with them) and recorded as the model's",
          json.dumps({"content": got, "tm": tm}))
    # 2. The second brain's hand-off carries one: scrubbed before it is
    #    prefilled into main's think block, and counted.
    saved = HANDOFF
    HANDOFF = HANDOFF.replace("Y axis.", "Y axis.</think> <|im_end|>")
    try:
        slots.reset(n=4)
        c = Client("xhigh")
        c.msgs[0]["content"] += " [markers 2]"
        t = c.turn([reply(" it sweeps.")],
                   user="How does LatheGeometry build its points?",
                   features={"investigate": True})
        pre = t["gens"][0]["request"]["messages"][-1]
        tm = t["d"]["x_yamadori"].get("template_markers") or {}
        check("</think>" not in pre.get("reasoning_content", "")
              and "<|im_end|>" not in pre.get("reasoning_content", "")
              and "Y axis." in pre.get("reasoning_content", "")
              and (tm.get("scrubbed") or {}).get("hand-off") == 2,
              "a hand-off carrying markers is scrubbed before it is "
              "prefilled as main's reasoning, and counted",
              json.dumps({"tm": tm, "pre": pre.get("reasoning_content",
                                                   "")[-200:]}))
        check(_think_blocks_balanced(render(t["gens"][0]["request"])),
              "and the prefilled turn renders one balanced think block")
    finally:
        HANDOFF = saved
    # 3. A client echoes reasoning carrying a marker: since the pass-through
    #    design (2026-09-24) it goes upstream AS SENT -- what the client
    #    sends is what the model sees -- and is counted.
    slots.reset(n=4)
    c = Client("xhigh")
    c.msgs[0]["content"] += " [markers 3]"
    c.msgs += [{"role": "user", "content": "hi"},
               {"role": "assistant", "content": "hello there",
                "reasoning_content": "greet</think>\n\nhello"},
               ]
    t = c.turn([reply("ok")], user="again")
    sent = [m for m in t["gens"][0]["request"]["messages"]
            if m.get("role") == "assistant"]
    rs = (t["d"]["x_yamadori"].get("ledger") or {}).get("restored") or {}
    check(sent and sent[0].get("reasoning_content")
          == "greet</think>\n\nhello"
          and rs.get("echoed_reasoning") == 1
          and rs.get("markers_in_echo") == 1,
          "echoed reasoning carrying a marker passes through as sent, and "
          "x_yamadori.ledger.restored counts it", json.dumps(rs))
    # 4. OUR injected text (a skill here; library source and the work log
    #    take the same path) carries markers: scrubbed when it is decided,
    #    counted, and replayed byte for byte (pre-deploy review,
    #    2026-09-24: only the hand-off was scrubbed).
    global HINT
    saved_hint = HINT
    HINT = "\n\n---\nSkill: close the block with </think> then <|im_end|>."
    try:
        slots.reset(n=4)
        c = Client("xhigh")
        c.msgs[0]["content"] += " [markers 4]"
        t = c.turn([reply("ok")], user="Refactor it.")
        last_user = next(m for m in reversed(
            t["gens"][0]["request"]["messages"]) if m.get("role") == "user")
        tm = t["d"]["x_yamadori"].get("template_markers") or {}
        check("Skill: close the block with" in last_user["content"]
              and "</think>" not in last_user["content"]
              and "<|im_end|>" not in last_user["content"]
              and (tm.get("scrubbed") or {}).get("injection") == 2,
              "markers in OUR injected text are scrubbed before it is sent, "
              "and counted", json.dumps({"tm": tm, "user": last_user[
                  "content"][-120:]}))
        t2 = c.turn([reply("ok again")], user="And again.")
        prior = [m for m in t2["gens"][0]["request"]["messages"]
                 if m.get("role") == "user"
                 and "Skill: close the block" in (m.get("content") or "")]
        check(prior and prior[0]["content"] == last_user["content"],
              "the scrubbed injection is what the ledger replays, byte for "
              "byte", json.dumps([m["content"][-80:] for m in prior]))
    finally:
        HINT = saved_hint


def test_library_use_reaches_harness_traffic():
    """#19 (docs/SELF-IMPROVEMENT-LOG.md): every Hermes request routes
    agent_step, and library definitions were injected on library_question
    only, so the Octopus pilot's @pmndrs/glyph task got no source help. Now,
    whatever the class, a held package the conversation USES -- a manifest
    the harness read, an import in a file it read, one the model wrote -- is
    covered on the message the request ends on: the imported names'
    definitions, whole. Once per name; replayed byte-identically, so every
    request still extends the slot. (The package OVERVIEW for a package used
    with no names yet, and the caps USE_MAX_*, were removed 2026-09-27,
    docs/CONSTANTS-AUDIT.md: the one bound is what the window leaves.)"""
    import sqlite3
    import domains
    db = os.path.join(_TMP, "pmndrs__glyph@0.1.0.sqlite3")
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE defs(name TEXT, kind TEXT, path TEXT, "
                "start INT, end INT, line TEXT)")
    con.executemany("INSERT INTO defs VALUES(?,?,?,?,?,?)", [
        ("loadFont", "function", "src/index.ts", 3, 9,
         "export function loadFont(url: string): Promise<Font>"),
        ("Text", "class", "src/text/Text.ts", 10, 80,
         "export class Text extends Mesh {"),
        ("helper", "function", "src/internal/x.ts", 1, 2,
         "export function helper() {")])
    con.commit()
    con.close()
    asked: list[str] = []

    def run_on_package(d, name, args):
        asked.append(args.get("symbol"))
        return (f"src/text/{args['symbol']}.ts:10-80\nexport class "
                f"{args['symbol']} extends Mesh {{ constructor(text: string) }}")
    saved = (domains.held_sources, proxy._run_on_package)
    domains.held_sources = lambda store=None: {"@pmndrs/glyph": [("0.1.0",
                                                                   db)]}
    proxy._run_on_package = run_on_package
    try:
        slots.reset(n=4)
        compaction.reset()
        c = Client("xhigh")
        c.msgs[0]["content"] += " [library use]"
        feats = {"skills": False}
        t1 = c.turn([reply("", reasoning="Look first.", calls=[call(
            "read_file", {"path": "package.json"}, "r1")])],
            user="Build the title screen with the glyph text library.",
            features=feats)
        c.tool_result("r1", '{"dependencies": {"@pmndrs/glyph": "0.1.0", '
                            '"three": "0.185.1"}}')
        t2 = c.turn([reply("", reasoning="Read main.", calls=[call(
            "read_file", {"path": "src/main.ts"}, "r2")])], features=feats)
        last = t2["gens"][0]["request"]["messages"][-1]
        lu2 = t2["d"]["x_yamadori"].get("library_use") or {}
        check(t2["d"]["x_yamadori"]["route"]["class"] == "agent_step"
              and lu2.get("used_held") == ["@pmndrs/glyph"]
              and not lu2.get("names") and "overview" not in lu2
              and "==" not in last.get("content", ""),
              "an agent_step whose tool result is a manifest naming a held "
              "package: recorded as used, and nothing is injected until a "
              "name from it is (no overview)", json.dumps(lu2))
        c.tool_result("r2", "import { Text } from '@pmndrs/glyph'\n"
                            "const t = new Text('Title')\n")
        t3 = c.turn([reply("Writing it.", calls=[call(
            "write_file", {"path": "src/title.ts", "content":
                           "import { Text, loadFont } from '@pmndrs/glyph'\n"
                           "export const title = new Text('Play')\n"},
            "w1")])], features=feats)
        last3 = t3["gens"][0]["request"]["messages"][-1]
        check("== @pmndrs/glyph@0.1.0: Text ==" in last3.get("content", "")
              and asked == ["Text"],
              "the next tool result, which imports Text from it, gets Text's "
              "definition from the package's own index", json.dumps(asked))
        _extends(t2, t3, "[library use] the request after the manifest")
        c.tool_result("w1", "wrote src/title.ts")
        t4 = c.turn([reply("Done.", reasoning="ok")], features=feats)
        last4 = t4["gens"][0]["request"]["messages"][-1]
        check("== @pmndrs/glyph@0.1.0: loadFont ==" in last4.get("content",
                                                                  "")
              and ": Text ==" not in last4["content"]
              and asked == ["Text", "loadFont"],
              "the file the model wrote adds a new name (loadFont): its "
              "definition arrives on that write's tool result; Text, already "
              "covered, is not asked again", json.dumps(asked))
        _extends(t3, t4, "[library use] and the one after the second")
        r3 = next(m for m in t4["gens"][0]["request"]["messages"]
                  if m.get("tool_call_id") == c.ids.get("r2", "r2"))
        check("== @pmndrs/glyph@0.1.0: Text ==" in r3.get("content", ""),
              "the first injection is still on its tool result, restored "
              "from the ledger")
        # The window is the one bound: a definition that does not fit what
        # it leaves is not injected, and not covered (tried again later).
        nebari.ledger_reset()
        text, rec = proxy._library_use(
            [{"role": "tool", "tool_call_id": "x", "content":
              "import { Text } from '@pmndrs/glyph'\n"}],
            "acct-room", "lin-room", room=10)
        check(text == "" and rec.get("did_not_fit") == ["@pmndrs/glyph:Text"],
              "a definition larger than the room the window leaves is not "
              "injected, and is recorded as not fitting", json.dumps(rec))
        check(bool(t1["d"]["x_yamadori"].get("library_use") is not None),
              "the first turn records library use too (nothing held used "
              "yet)", json.dumps(t1["d"]["x_yamadori"].get("library_use")))
    finally:
        domains.held_sources, proxy._run_on_package = saved


def test_library_use_reads_the_conversations_version():
    """Pre-deploy review, 2026-09-24 (FIX SOON #4): library help used the
    NEWEST held version (held[pkg][0]) whatever the conversation used. Now
    the session's recorded version or a manifest in the conversation picks
    the index; only when neither says does it fall back to the newest, and
    the header says so."""
    import domains
    held = {"typegpu": [("0.12.0", "db-new"), ("0.11.2", "db-old")]}
    used: list[str] = []

    def run_on_package(db, name, args):
        used.append(db)
        return f"src/{args['symbol']}.ts:1-9\nexport function {args['symbol']}()"
    saved = (domains.held_sources, proxy._run_on_package)
    domains.held_sources = lambda store=None: held
    proxy._run_on_package = run_on_package
    imp = {"role": "tool", "tool_call_id": "r", "content":
           "import { tgpu } from 'typegpu'\n"}
    try:
        nebari.ledger_reset()
        manifest = {"role": "tool", "tool_call_id": "m", "content":
                    '{"dependencies": {"typegpu": "^0.11.2"}}'}
        text, rec = proxy._library_use([manifest, imp], "acct-v", "lin-1")
        check(used == ["db-old"] and "== typegpu@0.11.2: tgpu ==" in text
              and rec["versions"]["typegpu"]["label"] is None,
              "a manifest in the conversation stating ^0.11.2 reads the "
              "0.11.2 index, not the newest (0.12.0)",
              json.dumps({"used": used, "rec": rec["versions"]}))
        used.clear()
        text, rec = proxy._library_use([imp], "acct-v", "lin-2",
                                       {"versions": {"typegpu": "0.11.2"}})
        check(used == ["db-old"] and "typegpu@0.11.2: tgpu" in text,
              "the session's RECORDED version picks the index when no "
              "manifest is in these messages", json.dumps(used))
        used.clear()
        text, rec = proxy._library_use([imp], "acct-v", "lin-3")
        check(used == ["db-new"]
              and "typegpu@0.12.0 (the newest held; the conversation's "
                  "version is not known): tgpu" in text,
              "version unknown: the newest held, and the header labels it",
              text[:300])
        used.clear()
        text, rec = proxy._library_use([imp], "acct-v", "lin-4",
                                       {"versions": {"typegpu": "0.9.0"}})
        check(used == ["db-new"] and "0.9.0 is not held" in text,
              "a stated version that is not held: the newest, labelled",
              text[:300])
    finally:
        domains.held_sources, proxy._run_on_package = saved
        nebari.ledger_reset()


def test_a_duplicate_request_never_blanks_the_library_help():
    """Pre-deploy review, 2026-09-24 (MINOR): two copies of one request in
    flight (a client retry). The first records the library help for the tool
    result; the second, which found the package already covered, decided ""
    and overwrote it -- and every later replay lost the help. A recorded
    non-empty decision now stands, and the second request sends it too."""
    slots.reset(n=4)
    compaction.reset()
    c = Client("xhigh")
    c.msgs[0]["content"] += " [duplicate]"
    feats = {"skills": False}
    c.turn([reply("", calls=[call("read_file", {"path": "a.ts"}, "d1")])],
           user="Use the glyph library.", features=feats)
    c.tool_result("d1", "import { Text } from '@pmndrs/glyph'\n")
    saved = proxy._library_use
    first = "\n\n== @pmndrs/glyph@0.1.0: Text ==\nexport class Text"
    # The conversation's chain-key salt: its key (it has an id, #41).
    salt = proxy._LAST_SESSION[ACCOUNT][0]

    def duplicate_finished_first(raw, account, lineage, state=None,
                                 room=None):
        # The other copy of this request recorded its decision while this
        # one was deciding; this one found nothing left to add.
        nebari.ledger_put(account, lineage, proxy.chain_keys(raw, salt)[-1],
                          "inject", first)
        return "", {"packages": [], "names": [], "chars": 0}
    proxy._library_use = duplicate_finished_first
    try:
        t = c.turn([reply("ok")], features=feats)
    finally:
        proxy._library_use = saved
    last = t["gens"][0]["request"]["messages"][-1]
    key = proxy.chain_keys(c.msgs[:-1], salt)[-1]
    check(nebari.ledger_get(ACCOUNT, key, "inject") == first
          and last.get("role") == "tool"
          and last.get("content", "").endswith(first),
          "the recorded help stands (not overwritten with \"\"), and this "
          "request sends it too", json.dumps({
              "recorded": nebari.ledger_get(ACCOUNT, key, "inject"),
              "sent": last.get("content", "")[-80:]}))
    check(nebari.ledger_claim("acct-c", "s", "k-empty", "inject", "") == ""
          and nebari.ledger_claim("acct-c", "s", "k-empty", "inject", "X")
          == "X"
          and nebari.ledger_claim("acct-c", "s", "k-empty", "inject", "")
          == "X",
          "ledger_claim: an empty decision may be replaced, a non-empty one "
          "never")



def test_a_retryable_decision_is_decided_again():
    """Live gate 2026-09-24 (STALE DECISION REPLAY). A user turn's injection
    decided while the embedder could not answer came out "" and was replayed
    as final ever after -- for every conversation whose first messages hash
    the same, across restarts. Now a part that came out empty BECAUSE
    something was unavailable is recorded retryable and decided again by the
    next request that ends on that same turn; once the conversation has
    moved past it, what was sent replays unchanged (prefix stability)."""
    slots.reset(n=4)
    compaction.reset()
    saved = proxy._skills_tail
    state = {"up": False, "calls": 0}
    GOOD = "\n\n---\nRelevant engineering notes: prefix sums."

    def skills(messages, sel, route, client_tools=None, account=""):
        state["calls"] += 1
        if not state["up"]:
            return "", {"on": False, "ids": [], "versions": [],
                        "why": "skills raised NoRoom: embeddings is not "
                               "loaded (A4000_BUSY)"}
        return GOOD, {"on": True, "names": ["prefix-sums"],
                      "chars": len(GOOD), "ids": ["p1"], "versions": [1],
                      "why": "attached 1"}
    proxy._skills_tail = skills
    feats = {"skills": True}
    try:
        # 1. The embedder is down: nothing attached, recorded RETRYABLE. On
        #    a turn AFTER the opening: a retried opening is a new
        #    conversation (#41), so the retry is of a later request.
        c = Client("medium")
        c.msgs[0]["content"] += " [retryable]"
        c.turn([reply("hello")], user="hi", features=feats)
        t1 = c.turn([reply("first")], user="static array, many range-sum "
                                           "queries", features=feats)
        inj1 = t1["d"]["x_yamadori"]["ledger"]["inject"]
        sent1 = t1["gens"][0]["request"]["messages"][-1]["content"]
        check(GOOD not in sent1 and inj1.get("decided")
              and "skills" in (inj1.get("retryable") or {}),
              "[retry] an injection that came out empty because the embedder "
              "was down is recorded RETRYABLE", json.dumps(inj1))
        # 2. The SAME request again (a client retry: the same messages, the
        #    same conversation): decided again, and now attached.
        state["up"] = True
        again = Client("medium")
        again.msgs = c.msgs[:-1]
        t2 = again.turn([reply("second")], features=feats)
        inj2 = t2["d"]["x_yamadori"]["ledger"]["inject"]
        sent2 = t2["gens"][0]["request"]["messages"][-1]["content"]
        check(sent2.endswith(GOOD) and inj2.get("decided")
              and not inj2.get("replayed") and inj2.get("retried")
              == ["skills"] and "skills" in (inj2.get("parts") or []),
              "[retry] the same request, the embedder up: decided again and "
              "sent, never replayed as \"\"", json.dumps(inj2))
        sk2 = t2["d"]["x_yamadori"].get("skills") or {}
        check(sk2.get("names") == ["prefix-sums"],
              "[retry] x_yamadori.skills names what was attached",
              json.dumps(sk2))
        # 3. Now FINAL: a third copy replays it byte for byte, and reports
        #    the skills it carries (they are in the prompt).
        n0 = state["calls"]
        third = Client("medium")
        third.msgs = c.msgs[:-1]
        t3 = third.turn([reply("third")], features=feats)
        inj3 = t3["d"]["x_yamadori"]["ledger"]["inject"]
        sent3 = t3["gens"][0]["request"]["messages"][-1]["content"]
        sk3 = t3["d"]["x_yamadori"].get("skills") or {}
        check(sent3 == sent2 and inj3.get("replayed")
              and state["calls"] == n0 and inj3.get("parts") == ["skills"],
              "[retry] a final decision replays identically, nothing decided",
              json.dumps(inj3))
        check(sk3.get("names") == ["prefix-sums"] and sk3.get("replayed")
              and sk3.get("chars") == len(GOOD),
              "[retry] a replay reports the skills its turn carries (the live "
              "'nothing attached' on a replayed turn)", json.dumps(sk3))
        # 4. PREFIX STABILITY: a retryable turn the conversation has MOVED
        #    PAST replays what was sent (""), even with the embedder up.
        state["up"] = False
        d = Client("medium")
        d.msgs[0]["content"] += " [moved past]"
        d.turn([reply("one")], user="static array, many range-sum queries",
               features=feats)
        state["up"] = True
        t5 = d.turn([reply("two")], user="and with updates?", features=feats)
        req = t5["gens"][0]["request"]["messages"]
        first_user = next(m for m in req if m.get("role") == "user")
        check(first_user["content"] == "static array, many range-sum queries"
              and req[-1]["content"].endswith(GOOD),
              "[retry] once the conversation moved past a retryable turn, it "
              "replays as sent (the slot holds it); the new turn is decided",
              json.dumps([first_user["content"][-60:],
                          req[-1]["content"][-60:]]))
    finally:
        proxy._skills_tail = saved


def test_a_definitions_lookup_that_fails_is_retryable():
    """The definitions part of the injection: a find_definition_opt that
    could not run (a retryable failure envelope, or a raise) makes the
    decision retryable instead of final-and-empty."""
    fail = json.dumps({"tool": "find_definition_opt", "ok": False,
                       "error": "A4000_BUSY", "reason": "busy",
                       "retryable": True})
    ok = "class Foo  ->  src/foo.ts:1-3\n  export class Foo {}"
    why: list = []
    saved = (proxy.run_our_tool, proxy.selection.question_of,
             proxy.selection.defined_symbols)
    proxy.selection.question_of = lambda m: ("what is Foo?", "", True)
    proxy.selection.defined_symbols = lambda q, dbs: {"pkg": ["Foo"]}
    route = {"class": "library_question"}
    tier = {"retrieval": True}
    gate = {"offer": True}
    try:
        proxy.run_our_tool = lambda *a, **k: fail
        text = proxy._library_definitions(route, tier, gate, {}, [], {}, None,
                                          unavailable=why)
        check(text == "" and why and "A4000_BUSY" in why[0],
              "a retryable failure envelope is recorded as unavailable",
              json.dumps(why))
        why2: list = []
        proxy.run_our_tool = lambda *a, **k: ok
        text = proxy._library_definitions(route, tier, gate, {}, [], {}, None,
                                          unavailable=why2)
        check(text.startswith(proxy.DEFINITIONS_HEAD) and not why2,
              "a lookup that answers is a final decision", text[:120])
    finally:
        (proxy.run_our_tool, proxy.selection.question_of,
         proxy.selection.defined_symbols) = saved
    # The ledger's own rule.
    nebari.ledger_reset()
    got = nebari.ledger_decide("acct-r", "s", "k-r", "inject", "",
                               {"retry": {"skills": "down"}})
    got2 = nebari.ledger_decide("acct-r", "s", "k-r", "inject", "X",
                                {"parts": ["skills"]})
    got3 = nebari.ledger_decide("acct-r", "s", "k-r", "inject", "",
                                {"retry": {"skills": "down"}})
    check(got[0] == "" and got2 == ("X", {"parts": ["skills"]})
          and got3 == ("X", {"parts": ["skills"]}),
          "ledger_decide: a retryable decision is replaced; a final "
          "non-empty one never (a duplicate that decided less loses)",
          json.dumps([got, got2, got3]))


def test_text_turn_keys_are_per_conversation():
    """Pre-deploy review, 2026-09-24 (MINOR): a text turn's ledger key was
    the hash of its content alone, so two of an account's conversations in
    which the assistant said the same words shared it, and one's recorded
    content and hidden hops were restored into the other."""
    nebari.ledger_reset()
    a = [{"role": "user", "content": "Draw a fox, then say done."}]
    b = [{"role": "user", "content": "Rename the variable, then say done."}]
    hop = [{"role": "assistant", "content": "", "tool_calls": [call(
        "yama_generate_image", {"prompt": "a fox"}, "gA")]},
           {"role": "tool", "tool_call_id": "gA", "content": "drew it"}]
    proxy.ledger_record_turn("acct-k", "sess-a",
                             {"role": "assistant", "content": "Done."}, hop,
                             prev=proxy.chain_keys(a)[-1])
    nxt = [{"role": "assistant", "content": "Done."},
           {"role": "user", "content": "Thanks."}]
    got_a, n_a = proxy.ledger_restore(a + nxt, "acct-k")
    got_b, n_b = proxy.ledger_restore(b + nxt, "acct-k")
    check(n_a["hops"] == 1 and any(m.get("tool_call_id") == "gA"
                                   for m in got_a),
          "the conversation that ran the hidden hop gets it back",
          json.dumps(n_a))
    check(n_b["hops"] == 0 and not any(m.get("tool_call_id") == "gA"
                                       for m in got_b),
          "another conversation of the same account whose assistant said the "
          "same words does NOT get it", json.dumps(n_b))
    nebari.ledger_reset()


def test_a_warm_releases_only_its_own_waiters():
    """Pre-deploy review, 2026-09-24 (MINOR). _warm_now popped
    _WARM_PENDING[key] whatever it held: a NEWER warm's event, registered
    while the old one ran, was removed, and that warm's waiter went through
    before it finished. And a waiter gave up at WARM_WAIT (180 s) while the
    warm's own requests could run 840 s."""
    import threading as _th
    key = "warm-identity"
    old, new = _th.Event(), _th.Event()
    proxy._WARM_PENDING[key] = new          # a newer warm registered
    saved = (slots.acquire, slots.release)
    slots.acquire = lambda *a, **k: {"slot": None, "how": "test"}
    slots.release = lambda *a, **k: None
    try:
        proxy._warm_now(key, "bonsai", [], {}, {}, old)
    finally:
        slots.acquire, slots.release = saved
    check(old.is_set() and not new.is_set()
          and proxy._WARM_PENDING.get(key) is new,
          "an older warm ending sets ITS event and leaves the newer warm "
          "pending", str(proxy._WARM_PENDING.get(key) is new))
    # The waiter re-checks: its warm ends, a newer one is pending -> it
    # waits for that one too.
    e1, e2 = _th.Event(), _th.Event()
    proxy._WARM_PENDING[key] = e1

    def swap():
        time.sleep(0.2)
        proxy._WARM_PENDING[key] = e2
        e1.set()
        time.sleep(0.5)
        proxy._WARM_PENDING.pop(key, None)
        e2.set()
    t = _th.Thread(target=swap)
    t.start()
    waited = proxy.wait_for_warm(key, timeout=5)
    t.join(5)
    check(waited is not None and waited >= 0.6 and e2.is_set(),
          "a waiter whose warm is followed by a newer one waits for that one "
          "too", str(waited))
    # The warm's own requests are bounded by WARM_WAIT.
    seen: list = []
    saved_up, saved_wait = proxy._upstream_json, proxy.WARM_WAIT

    def up(path, body, timeout=120):
        seen.append(timeout)
        if path.endswith("apply-template"):
            return {"prompt": json.dumps(body.get("messages"))
                    + proxy.END_OF_TURN}
        return {"timings": {"cache_n": 1, "prompt_n": 1}}
    proxy._upstream_json, proxy.WARM_WAIT = up, 5.0
    slots.acquire = lambda *a, **k: {"slot": 0, "how": "test"}
    slots.release = lambda *a, **k: None
    try:
        proxy._warm_now(key, "bonsai", [{"role": "user", "content": "q"},
                                        {"role": "assistant",
                                         "content": "a"}], {}, {},
                        _th.Event())
    finally:
        proxy._upstream_json, proxy.WARM_WAIT = saved_up, saved_wait
        slots.acquire, slots.release = saved
    check(seen and max(seen) <= 5.0,
          "every request a warm makes is bounded by WARM_WAIT (here 5 s), so "
          "it cannot outlive its waiter", str(seen))
    proxy._WARM_PENDING.pop(key, None)


def test_the_fixup_never_touches_the_conversations_slot():
    """#11, the Octopus pilot's evidence (2026-09-24, the build before the
    slot reservation): after a step's fix-up the warm processed 48,910
    tokens from zero and the next request processed 48,988 from zero again
    -- the prompt paid twice -- while every warm with no fix-up before it
    reused ~49-60k. The fix-up is the second brain's job; it ran on, or
    evicted, the conversation's slot between the generation and the warm.
    The sequence, through the REAL doors (the main turn via proxy._post_events,
    the fix-up via mcp/model.py's own slot choice, the warm via /completion):
    generation -> fix-up -> warm -> next request, with every slot recorded.
    The conversation's generation, its warm and its next request share one
    slot; the fix-up is on the reserved helper slot; nobody is evicted."""
    saved = (shomen._post, _model.UPSTREAM)
    shomen._post = _REAL_SHOMEN_POST
    _model.UPSTREAM = proxy.UPSTREAM
    try:
        for others in (0, 1):
            slots.reset(n=4)
            compaction.reset()
            _helper_direct.clear()
            c = Client("xhigh")
            c.msgs[0]["content"] += f" [fixup slot {others}]"
            # Other live conversations first, so every conversation slot is
            # pinned when this one arrives (others=1: slots 0 and 1 held).
            for k in range(others + 1):
                slots.release(slots.acquire(f"other-{others}-{k}"))
            t = c.turn([reply("Writing it.", calls=[call(
                "write_file", {"path": "hi.py", "content": BROKEN}, "fx1")])],
                user="Write hi.py with a function f that returns 1.")
            gen_slot = t["gens"][0]["request"].get("id_slot")
            fix_slots = [b.get("id_slot") for b in _helper_direct]
            warm_slot = (t["warm"] or [{}])[-1].get("id_slot")
            x = t["d"]["x_yamadori"]
            c.tool_result("fx1", "wrote hi.py")
            t2 = c.turn([reply("Done.", reasoning="ok")])
            next_slot = t2["gens"][0]["request"].get("id_slot")
            ev = [((x.get("cache") or {}).get("evicted")),
                  ((t2["d"]["x_yamadori"].get("cache") or {}).get("evicted"))]
            tag = f"[fixup, {others + 1} other conversation(s)]"
            check((x.get("tool_code") or {}).get("stopped") == "fixed"
                  and fix_slots,
                  f"{tag} the write was repaired by the fix-up job, through "
                  f"model.post's own slot choice", json.dumps(fix_slots))
            check(fix_slots and all(s == slots.helper_slot(4)
                                    for s in fix_slots)
                  and gen_slot not in fix_slots,
                  f"{tag} the fix-up ran on the helper slot "
                  f"({slots.helper_slot(4)}), never the conversation's "
                  f"({gen_slot})", json.dumps({"gen": gen_slot,
                                               "fixup": fix_slots}))
            check(gen_slot is not None and warm_slot == gen_slot
                  and next_slot == gen_slot,
                  f"{tag} generation, warm and next request share the "
                  f"conversation's slot", json.dumps(
                      {"gen": gen_slot, "warm": warm_slot,
                       "next": next_slot}))
            check(ev[1] is None and fix_slots and not any(
                      s in (0, 1) for s in fix_slots),
                  f"{tag} neither the fix-up nor the next request evicts or "
                  f"uses a conversation's slot", json.dumps(ev))
            wb = t2["d"]["x_yamadori"].get("warm_before") or {}
            check(wb.get("slot") == gen_slot and not wb.get("moved_from"),
                  f"{tag} and the warm's record says it loaded that slot",
                  json.dumps(wb))
    finally:
        shomen._post, _model.UPSTREAM = saved


def test_the_fixup_releases_the_helper_slot_not_the_conversations():
    """RELEASE (mcp/slots.py): the fix-up job's slot is emptied when the
    job ends -- its cells would slow every decode of the conversation's
    next step -- and the conversation's slot, its warm and its next request
    are exactly what they were without it. The same sequence as the test
    above, through the real doors, with release on; model.release_slot is
    replaced, so no release reaches a server."""
    saved = (shomen._post, _model.UPSTREAM, _model.release_slot)
    shomen._post = _REAL_SHOMEN_POST
    _model.UPSTREAM = proxy.UPSTREAM
    released: list = []

    def fake(slot, model=None, timeout=None):
        released.append(slot)
        return {"ok": True, "method": "shrink", "cells_before": 1234,
                "ms": 4}
    _model.release_slot = fake
    old_env = os.environ.pop("YAMADORI_SLOT_RELEASE", None)
    slots.enable_release(True)
    try:
        slots.reset(n=4)
        compaction.reset()
        _helper_direct.clear()
        c = Client("xhigh")
        c.msgs[0]["content"] += " [fixup release]"
        t = c.turn([reply("Writing it.", calls=[call(
            "write_file", {"path": "hi.py", "content": BROKEN}, "fr1")])],
            user="Write hi.py with a function f that returns 1.")
        gen_slot = t["gens"][0]["request"].get("id_slot")
        warm_slot = (t["warm"] or [{}])[-1].get("id_slot")
        x = t["d"]["x_yamadori"]
        rel = (x.get("slots") or {}).get("released") or []
        check((x.get("tool_code") or {}).get("stopped") == "fixed"
              and _helper_direct,
              "[release] the write was repaired by the fix-up job",
              json.dumps((x.get("tool_code") or {}).get("stopped")))
        check(released == [slots.helper_slot(4)]
              and [r.get("slot") for r in rel] == [slots.helper_slot(4)]
              and rel[0].get("released") and "fixup" in rel[0].get("why", ""),
              "[release] the job ended: the helper slot released once, and "
              "x_yamadori.slots.released records it on that request",
              json.dumps({"calls": released, "record": rel}))
        c.tool_result("fr1", "wrote hi.py")
        t2 = c.turn([reply("Done.", reasoning="ok")])
        next_slot = t2["gens"][0]["request"].get("id_slot")
        check(gen_slot not in released and warm_slot == gen_slot
              and next_slot == gen_slot and released == [slots.helper_slot(4)],
              "[release] never the conversation's slot: its generation, warm "
              "and next request share it as before",
              json.dumps({"gen": gen_slot, "warm": warm_slot,
                          "next": next_slot, "released": released}))
    finally:
        slots.enable_release(False)
        shomen._post, _model.UPSTREAM, _model.release_slot = saved
        if old_env is not None:
            os.environ["YAMADORI_SLOT_RELEASE"] = old_env
        slots.reset(n=4)


def _fixup_step(mode: str, tag: str) -> dict:
    """A STREAMED agent step whose write is repaired and re-formatted by the
    fix-up, stored the way a `mode` ("strip" / "echo") client stores it, then
    the tool result sent AT ONCE (a fast harness). Returns the generations,
    the warm and the stored message."""
    msgs = [{"role": "system", "content": SYSTEM + f" [{tag} {mode}]"},
            {"role": "user", "content": "Write hi.py with a function f that "
                                        "returns 1."}]
    body = {"model": "yamadori", "reasoning_effort": "xhigh",
            "_account": f"{ACCOUNT}-{mode}", "_client_ip": "127.0.0.1",
            "tools": [WRITE],
            "_features": json.dumps({"skills": False, "investigate": False,
                                     "fanout": 1})}
    # The client's habit, seen once before on this account (echoing is a
    # property of the harness): an earlier two-turn conversation.
    past = [{"role": "system", "content": SYSTEM + f" [habit {mode}]"},
            {"role": "user", "content": "hi"},
            dict({"role": "assistant", "content": "hello"},
                 **({"reasoning_content": "greet"} if mode == "echo" else {})),
            {"role": "user", "content": "thanks"}]
    _script[:] = [reply("ok", reasoning="ack")]
    _stream(dict(body, messages=past))
    _script[:] = [reply("", reasoning="Write it.", calls=[call(
        "write_file", {"path": "hi.py", "content": BROKEN}, "fx9")])]
    n0, w0, d0 = len(_gens), len(_warms), len(_warm_done_at)
    got = _stream(dict(body, messages=json.loads(json.dumps(msgs))))
    stored = {"role": "assistant", "content": got["content"],
              "tool_calls": got["calls"]}
    if mode == "echo" and got["reasoning"]:
        stored["reasoning_content"] = got["reasoning"]
    msgs += [stored, {"role": "tool", "tool_call_id": "fx9",
                      "content": "wrote hi.py"}]
    _script[:] = [reply("Done.", reasoning="ok")]
    n1 = len(_gens)
    _stream(dict(body, messages=json.loads(json.dumps(msgs))))
    return {"first": _gens[n0:n1], "next": _gens[n1:],
            "warm": _warms[w0:], "warm_done": _warm_done_at[d0:],
            "stored": stored}


def test_a_fixup_step_and_its_warm():
    """#11 on the fix-up path (the V0 pilot, conversation on slot 1,
    2026-09-24): warms after a fix-up re-read everything (48,910 and 67,976
    from 0), warms after clean or formatted-only writes reused, and after
    the first repair the NEXT request missed completely too. Two candidates,
    one test each, both streamed, both clients:
      RACE    the next request must not reach the slot before the warm has
              finished: the fake /completion takes 1 s and the harness sends
              its tool result at once.
      RENDER  the warm renders the turn exactly as the client stores and
              resends it (the note, the repaired + formatted call, the call
              id, and -- for an echoing client -- its echoed reasoning), so
              the next request extends the warmed prompt."""
    global WARM_DELAY, FIXED
    saved = (WARM_DELAY, FIXED)
    WARM_DELAY = 1.0
    FIXED = "def f():\n  return  1\n"          # parses; ruff reformats it
    try:
        for mode in ("strip", "echo"):
            slots.reset(n=4)
            compaction.reset()
            t = _fixup_step(mode, "fixup-race")
            tag = f"[fixup {mode}]"
            content = t["stored"]["content"]
            args = json.loads(t["stored"]["tool_calls"][0]["function"]
                              ["arguments"])
            # #24: the repair is sent as the fix-up wrote it; the
            # formatter's view is only recorded, never in the note (#34).
            check("Repaired hi.py" in content
                  and "the repaired file was sent" in content
                  and "would change" not in content
                  and args.get("content") == FIXED and t["warm"],
                  f"{tag} the write is repaired (not reformatted), the note "
                  f"delivered, and the slot warmed",
                  json.dumps({"content": content[-120:], "args": args}))
            arrived = t["next"][0]["at"] if t["next"] else 0
            done = t["warm_done"][0] if t["warm_done"] else 1e18
            check(arrived >= done,
                  f"{tag} RACE: the next request reached the slot only after "
                  f"the warm finished (arrived {arrived - done:+.2f} s after)")
            if t["warm"] and t["next"]:
                h = t["warm"][-1]["prompt"]
                r = render(t["next"][0]["request"])
                n = next((i for i, (a, b) in enumerate(zip(h, r)) if a != b),
                         min(len(h), len(r)))
                check(r.startswith(h),
                      f"{tag} RENDER: the next request extends the warmed "
                      f"prompt ({len(h)} chars)",
                      f"diverges at {n}: warm {h[max(0, n - 80):n + 60]!r} "
                      f"vs sent {r[max(0, n - 80):n + 60]!r}")
    finally:
        WARM_DELAY, FIXED = saved


def test_a_client_that_acts_on_the_calls_waits_for_the_warm():
    """#5, the live gate of 2026-09-24 (the repair path: the next request
    extended less than the prompt the slot had generated on). A STREAMING
    harness has the calls the moment their chunk arrives, and one that runs
    the tool then and there sends its next request while this turn is still
    between that chunk and scheduling its warm -- when nothing was pending
    yet, so the next request went straight to the slot and the warm then
    overwrote it with the older prefix. Here the harness sends its next
    request on the calls chunk and holds the first stream until that
    request has either reached the model or visibly waited (3 s); the next
    request must reach the slot only after the warm finished."""
    global WARM_DELAY
    saved = WARM_DELAY
    WARM_DELAY = 1.0
    slots.reset(n=4)
    compaction.reset()
    msgs = [{"role": "system", "content": SYSTEM + " [pipelined]"},
            {"role": "user", "content": "Write hi.py with a function f that "
                                        "returns 1."}]
    body = {"model": "yamadori", "reasoning_effort": "xhigh",
            "_account": f"{ACCOUNT}-pipe", "_client_ip": "127.0.0.1",
            "tools": [WRITE],
            "_features": json.dumps({"skills": False, "investigate": False,
                                     "fanout": 1})}
    _script[:] = [reply("", reasoning="Write it.", calls=[call(
        "write_file", {"path": "hi.py", "content": BROKEN}, "px1")])]
    d0 = len(_warm_done_at)
    content, calls, box = "", [], {}
    th = None
    try:
        for b in proxy.stream_body(dict(body, stream=True,
                                        messages=json.loads(json.dumps(msgs)))):
            for line in b.decode("utf-8").split("\n"):
                if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                    continue
                for ch in json.loads(line[5:].strip()).get("choices") or []:
                    dl = ch.get("delta") or {}
                    content += dl.get("content") or ""
                    for c in dl.get("tool_calls") or []:
                        calls.append({k: v for k, v in c.items()
                                      if k != "index"})
            if calls and th is None:
                # The harness runs the tool NOW and sends the result.
                follow = msgs + [{"role": "assistant", "content": content,
                                  "tool_calls": calls},
                                 {"role": "tool", "tool_call_id": "px1",
                                  "content": "wrote hi.py"}]
                _script[:] = [reply("Done.", reasoning="ok")]
                box["n1"] = len(_gens)
                th = threading.Thread(target=lambda: box.setdefault(
                    "d", proxy.complete(dict(body, messages=follow))))
                th.start()
                end = time.time() + 3
                while time.time() < end and len(_gens) == box["n1"]:
                    time.sleep(0.02)
        if th is not None:
            th.join(20)
    finally:
        WARM_DELAY = saved
    n1 = box.get("n1", len(_gens))
    arrived = _gens[n1]["at"] if len(_gens) > n1 else None
    done = _warm_done_at[d0] if len(_warm_done_at) > d0 else None
    wb = ((box.get("d") or {}).get("x_yamadori") or {}).get("warm_before") \
        or {}
    check(arrived is not None and done is not None and arrived >= done
          and wb.get("state") == "done" and wb.get("waited_s") is not None,
          "[race] a harness that sends its next request on the calls chunk: "
          "that request reaches the slot only after the warm finished, and "
          "says it waited",
          json.dumps({"arrived_minus_done": (arrived - done) if arrived and
                      done else None, "warm_before": wb}))


def test_a_first_turn_warm_does_not_guess_echo_from_a_mixed_account():
    """Live gate 2026-09-24 (second run), the repair test: the next request
    reused 2374 of 2890 after a good warm. Rendered with the served
    template, the warm and that request differ at the repaired turn's think
    block: the warm carried the turn's reasoning, the stripping client sent
    none. The account's last observation said "echoes" -- the agent-loop
    test's echoing client ran just before on the same key. A first turn now
    guesses "echoes" only when the account was only ever seen echoing (the
    set of what it did; the 8-observation window, ECHO_WINDOW, was removed
    2026-09-27, docs/CONSTANTS-AUDIT.md)."""
    global WARM_DELAY
    saved = WARM_DELAY
    WARM_DELAY = 0.0
    slots.reset(n=4)
    compaction.reset()
    acct = f"{ACCOUNT}-mixed"
    body = {"model": "yamadori", "reasoning_effort": "xhigh",
            "_account": acct, "_client_ip": "127.0.0.1", "tools": [WRITE],
            "_features": json.dumps({"skills": False, "investigate": False,
                                     "fanout": 1})}
    try:
        # The account's recent past: a stripping client, then an echoing one.
        for i, echo in enumerate((False, True)):
            past = [{"role": "system", "content": SYSTEM + f" [past {i}]"},
                    {"role": "user", "content": "hi"},
                    dict({"role": "assistant", "content": "hello"},
                         **({"reasoning_content": "greet"} if echo else {})),
                    {"role": "user", "content": "thanks"}]
            _script[:] = [reply("ok", reasoning="ack")]
            _stream(dict(body, messages=past))
        check(not proxy._echo_guess(acct)
              and nebari.ledger_get(acct, "client", "echo_history") == "01"
              and not hasattr(proxy, "ECHO_WINDOW"),
              "[habit] a mixed account's first turn does not guess 'echoes' "
              "(the account keeps the set it was seen doing, no window)",
              nebari.ledger_get(acct, "client", "echo_history") or "")
        # Now a STRIPPING client's first turn: a repaired write, warmed.
        msgs = [{"role": "system", "content": SYSTEM + " [mixed strip]"},
                {"role": "user", "content": "Write hi.py with a function f "
                                            "that returns 1."}]
        _script[:] = [reply("", reasoning="Write it.", calls=[call(
            "write_file", {"path": "hi.py", "content": BROKEN}, "mx1")])]
        w0 = len(_warms)
        got = _stream(dict(body, messages=json.loads(json.dumps(msgs))))
        msgs += [{"role": "assistant", "content": got["content"],
                  "tool_calls": got["calls"]},
                 {"role": "tool", "tool_call_id": "mx1",
                  "content": "wrote hi.py"}]
        end = time.time() + 5
        while time.time() < end and len(_warms) == w0:
            time.sleep(0.02)
        _script[:] = [reply("Done.", reasoning="ok")]
        n1 = len(_gens)
        _stream(dict(body, messages=json.loads(json.dumps(msgs))))
    finally:
        WARM_DELAY = saved
    h = _warms[-1]["prompt"] if len(_warms) > w0 else ""
    r = render(_gens[n1]["request"]) if len(_gens) > n1 else ""
    n = next((i for i, (a, b) in enumerate(zip(h, r)) if a != b),
             min(len(h), len(r)))
    check(h and r.startswith(h),
          "[habit] the stripping client's next request extends the warmed "
          "prompt (no reasoning guessed into its first turn)",
          f"diverges at {n}: warm {h[max(0, n - 60):n + 60]!r} vs sent "
          f"{r[max(0, n - 60):n + 60]!r}")


def test_the_warm_reports_on_the_next_response():
    """#11 (docs/SELF-IMPROVEMENT-LOG.md): a warm's own numbers existed only
    in the proxy log (the response had gone out before the warm ended), so
    'every warm re-reads the prompt' could not be seen by the client or a
    test, nor told apart from a warm that works. The next response of the
    same conversation carries it as x_yamadori.warm_before, `short` when it
    reused less than the prompt the slot had just generated on."""
    global WARM_TIMINGS
    slots.reset(n=4)
    compaction.reset()
    saved = dict(WARM_TIMINGS)
    try:
        for label, timings, short in (
                ("a warm that extends", {"cache_n": 100, "prompt_n": 20},
                 False),
                ("a warm that re-reads everything (the live 0 of 4567)",
                 {"cache_n": 0, "prompt_n": 120}, True)):
            WARM_TIMINGS = timings
            c = Client("xhigh")
            c.msgs[0]["content"] += f" [{label}]"
            t = c.turn([reply("Writing it.", calls=[call(
                "write_file", {"path": "hi.py", "content": BROKEN}, "w7")])],
                user="Write hi.py with a function f that returns 1.")
            check(bool(t["warm"]), f"[warm] {label}: the repaired write "
                                   f"was warmed")
            c.tool_result("w7", "wrote hi.py")
            t2 = c.turn([reply("Done.", reasoning="ok")])
            wb = (t2["d"]["x_yamadori"] or {}).get("warm_before") or {}
            check(wb.get("state") == "done"
                  and wb.get("reused") == timings["cache_n"]
                  and wb.get("expect_reused_at_least") == 100
                  and wb.get("short") is short,
                  f"[warm] {label}: the next response reports it "
                  f"(reused {timings['cache_n']}, short={short})",
                  json.dumps(wb))
            t3 = c.turn([reply("Still done.")], user="ok?")
            check((t3["d"]["x_yamadori"] or {}).get("warm_before") is None,
                  f"[warm] {label}: and not again on the response after")
    finally:
        WARM_TIMINGS = saved


def _agent_step_opening(c: "Client", task: str) -> None:
    """Open a conversation on an agent step: the task, a read the model
    made, its result. The first request then answers a tool result -- not
    a new task, so no kickoff plan runs before main."""
    c.msgs += [{"role": "user", "content": task},
               {"role": "assistant", "content": "", "tool_calls": [call(
                   "read_file", {"path": "src/scene.js"}, "open1")]},
               {"role": "tool", "tool_call_id": "open1",
                "content": '{"content": "1|// scene"}'}]


class Unforced(Client):
    """A client whose requests leave deep thinking to the TIER and the
    triggers (Phase 0.6): no `investigate` in X-Yamadori-Features."""

    def __init__(self, effort: str, tag: str, tools: list | None = None):
        super().__init__(effort)
        self.tag = tag.replace(" ", "-")
        self.msgs[0]["content"] += f" [{tag}]"
        self.tools = tools or [WRITE]

    def body(self, features: dict | None = None) -> dict:
        b = super().body(features)
        feats = {"skills": True, "fanout": 1}
        feats.update(EXTRA_FEATURES)
        feats.update(features or {})
        b["_features"] = json.dumps(feats)
        b["tools"] = self.tools
        # An account of its own: session() compacts in place under ACCOUNT,
        # and a new conversation of that account soon after is linked to it
        # (proxy._continue_after_compaction) -- one lineage, one deep state.
        b["_account"] = f"{ACCOUNT}-{self.tag}"
        return b


TERMINAL = {"type": "function", "function": {
    "name": "terminal", "description": "Run a shell command.",
    "parameters": {"type": "object", "properties": {
        "command": {"type": "string"}}}}}
PLAN = ("FILES\n- index.html: the canvas\n- js/game.js: the loop\n"
        "ORDER\n- write index.html\n- write js/game.js\n"
        "KEY DECISIONS\n- canvas 2D, no libraries (reasoning, not checked "
        "against source)\nCONSTRAINTS\n- input is read on keydown\n")


def test_a_think_deeply_with_no_conclusion_says_so():
    """2026-09-26 (deploy check, handle d28941fb): after a yama_think_deeply run
    that wrote no conclusion (a machine-built hand-off), the next hop's
    prefilled reasoning is deep.THINK_REASONING_MACHINE -- not the line that
    asks for "the findings in full" -- and a concluded run keeps
    THINK_REASONING. The next request still extends the slot."""
    import deep
    slots.reset(n=4)
    compaction.reset()
    _helper.clear()
    real = shomen._post

    def empty_post(path, payload, timeout=3600):
        _helper.append(json.loads(json.dumps(payload)))
        return {"choices": [{"message": {"content": ""},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
    shomen._post = empty_post
    try:
        c = Unforced("xhigh", "think machine session",
                     tools=[WRITE, READ_TOOL])
        q = "Which release deprecated label() in three TSL?"
        # An agent step, not a new task: every new task is planned before
        # main (2026-09-27), and a pre-main run refuses yama_think_deeply.
        _agent_step_opening(c, "When was TSL label() deprecated?")
        t1 = c.turn([reply("", reasoning="Not sure.",
                           calls=[call("yama_think_deeply", {"question": q},
                                       "tdm")]),
                     reply(" r179, as far as I know.", reasoning="")])
    finally:
        shomen._post = real
    x = t1["d"]["x_yamadori"]
    calls = ((x.get("deep") or {}).get("think_tool") or {}).get("calls") or []
    pre = t1["gens"][1]["request"]["messages"][-1] if len(t1["gens"]) > 1 \
        else {}
    check(calls and calls[0].get("ran") is True
          and (calls[0].get("handoff") or {}).get("machine_built") is True,
          "[think-machine] the run wrote nothing: a machine-built hand-off",
          json.dumps(calls)[:300])
    check(pre.get("role") == "assistant"
          and pre.get("reasoning_content") == deep.THINK_REASONING_MACHINE
          and deep.think_reasoning(True) == deep.THINK_REASONING
          and "in full" not in deep.THINK_REASONING_MACHINE
          and "no conclusion written" in deep.THINK_REASONING_MACHINE
          and "above" not in deep.THINK_REASONING_MACHINE
          and not deep.THINK_REASONING_MACHINE.endswith(" "),
          "[think-machine] the next hop's reasoning says no conclusion was "
          "written and what the answer does instead",
          json.dumps(pre)[:300])
    t2 = c.turn([reply("You are welcome.")], user="Thanks.")
    _extends(t1, t2, "[think-machine] the request after a machine-built "
                     "yama_think_deeply hop")


def test_think_deeply_is_a_hidden_hop_the_next_request_extends():
    """Phase 0.6, trigger 1 (operator, 2026-09-24): main calls yama_think_deeply;
    the proxy runs it through the second brain as an INTERNAL hop -- the
    call and the hand-off as its tool result -- the client never sees it,
    the next hop is prefilled with the fold-back opening, and the ledger
    replays the hop so the next request extends the slot."""
    import deep
    slots.reset(n=4)
    compaction.reset()
    _helper.clear()
    c = Unforced("xhigh", "think session", tools=[WRITE, READ_TOOL])
    q = "Why does LatheGeometry need its points sorted by y?"
    # An agent step, not a new task: every new task is planned before main
    # (2026-09-27), and a pre-main run refuses yama_think_deeply.
    _agent_step_opening(c, "My LatheGeometry renders inside out. Why?")
    t1 = c.turn([reply("", reasoning="Not sure of the API.",
                       calls=[call("yama_think_deeply", {"question": q,
                                                    "tried": "reversing"},
                                   "td1")]),
                 reply(" sort the points by y.", reasoning="")])
    x = t1["d"]["x_yamadori"]
    req0 = t1["gens"][0]["request"]
    names = [t["function"]["name"] for t in req0.get("tools") or []]
    sys0 = req0["messages"][0]["content"]
    check("yama_think_deeply" in names and proxy.ADDENDUM_THINK_ROW.strip() in sys0,
          "[think] at xhigh main has yama_think_deeply, and the addendum its row",
          str(names))
    check(len(t1["gens"]) == 2 and any(
              q in m.get("content", "") and "Already tried" in m["content"]
              for h in _helper for m in h["messages"]
              if m.get("role") == "user"),
          "[think] the call ran through the second brain with the question "
          "and what was tried", str(len(_helper)))
    pre = t1["gens"][1]["request"]["messages"][-1]
    tool = t1["gens"][1]["request"]["messages"][-2]
    word = ((x.get("deep") or {}).get("think_tool") or {}).get("calls", [{}])
    word = ((word[0] if word else {}).get("seed") or {}).get("word")
    check(pre.get("role") == "assistant"
          and pre.get("reasoning_content") == deep.THINK_REASONING
          and pre.get("content") == f"Today I was inspired by {word}. After "
                                    f"thinking deeply,"
          and tool.get("role") == "tool" and tool.get("tool_call_id") == "td1"
          and "LatheGeometry sweeps" in tool.get("content", ""),
          "[think] the hand-off is the hop's tool result; the next hop is "
          "prefilled with the seed line and 'After thinking deeply,'",
          json.dumps(pre)[:200])
    ans = t1["d"]["choices"][0]["message"]
    check(not ans.get("tool_calls") and ans["content"].startswith(
              f"Today I was inspired by {word}. After thinking deeply, sort"),
          "[think] the client never sees the call; the answer opens with "
          "the fold-back", ans["content"][:120])
    dx = x.get("deep") or {}
    check(dx.get("think_tool", {}).get("offered") is True
          and dx["think_tool"]["calls"][0].get("ran") is True
          and dx.get("record"),
          "[think] x_yamadori.deep records the offer, the call and the row",
          json.dumps(dx)[:300])
    t2 = c.turn([reply("You are welcome.")], user="Thanks.")
    _extends(t1, t2, "[think] the request after a yama_think_deeply hop")
    msgs2 = t2["gens"][0]["request"]["messages"]
    check(any(m.get("role") == "tool" and m.get("tool_call_id") == "td1"
              and "LatheGeometry sweeps" in m.get("content", "")
              for m in msgs2)
          and t2["gens"][0]["request"]["tools"] == req0["tools"],
          "[think] the hidden hop is replayed from the ledger, and main's "
          "tool list is the same on every turn")
    # A header that forces deep thinking off later in the conversation: the
    # tool stays (the system block is cached), a call is refused, and the
    # refusal says so.
    t3 = c.turn([reply("", calls=[call("yama_think_deeply", {"question": q},
                                       "td2")]),
                 reply("Answering from context.")],
                user="And why is it dark?", features={"investigate": False})
    calls3 = (t3["d"]["x_yamadori"]["deep"]["think_tool"]["calls"])
    check(t3["gens"][0]["request"]["tools"] == req0["tools"]
          and calls3 and calls3[0].get("refused") == "DEEP_THINKING_OFF",
          "[think] forced off later: the tool stays, the call is refused "
          "with DEEP_THINKING_OFF", json.dumps(calls3))
    _extends(t2, t3, "[think] and the request after that")


def test_struggle_runs_deep_thinking_before_main_once_per_episode():
    """Phase 0.6, trigger 2: three struggle signals in what the harness sent
    back (a failing command re-run, the same tool erroring again) run deep
    thinking BEFORE main, its hand-off prefilled as main's reasoning; the
    next step counts from the run's boundary, and each request extends the
    slot."""
    slots.reset(n=4)
    compaction.reset()
    _helper.clear()
    c = Unforced("max", "struggle session", tools=[WRITE, TERMINAL])
    # Structured, as Hermes' terminal reports it: since 2026-09-27 only
    # structured fields make a failure (deep.is_error).
    fail = json.dumps({"output": "npm ERR! Test failed. See above for more "
                                 "details.", "exit_code": 1, "error": None})
    c.msgs += [{"role": "user", "content": "Make the tests pass."}]
    # Four failing runs of one command: three re-runs failing the same way,
    # three signals (#45: a re-run is one signal, not a re-run AND a
    # repeated error).
    for i, cid in enumerate(("t0", "t1", "t2", "t2b")):
        c.msgs.append({"role": "assistant", "content": "", "tool_calls": [
            call("terminal", {"command": "npm test"}, cid)]})
        c.tool_result(cid, fail)
    t1 = c.turn([reply("", reasoning="Try the fix.", calls=[call(
        "terminal", {"command": "npm test -- --runInBand"}, "t3")])])
    x = t1["d"]["x_yamadori"]
    dx = x.get("deep") or {}
    pre = t1["gens"][0]["request"]["messages"][-1]
    check(dx.get("kind") == "struggle" and dx.get("fire")
          and dx["signals"]["struggle"]["count"] >= 3
          and (x.get("investigate") or {}).get("trigger") == "struggle",
          "[struggle] four failing runs of the same command (three re-runs "
          "failing the same way): the signals reach the threshold and deep "
          "thinking runs",
          json.dumps({k: dx.get(k) for k in ("fire", "kind", "because", "last_run", "allowed", "forced")})[:600])
    check(pre.get("role") == "assistant"
          and "LatheGeometry sweeps" in pre.get("reasoning_content", "")
          and any("stuck" in m.get("content", "") and "npm ERR!" in
                  m.get("content", "") for h in _helper for m in h["messages"]
                  if m.get("role") == "user"),
          "[struggle] the second brain gets the task and the failing output; "
          "its hand-off is prefilled as main's reasoning",
          json.dumps(pre)[:200])
    c.tool_result("t3", fail)
    t2 = c.turn([reply("", reasoning="Again.", calls=[call(
        "terminal", {"command": "npm test -- --runInBand"}, "t4")])])
    d2 = t2["d"]["x_yamadori"]["deep"]
    check(not d2.get("fire")
          and d2["signals"]["struggle"]["count"] == 0
          and (d2.get("last_run") or {}).get("requests_since_run") == 1,
          "[struggle] the episode restarted at the run's boundary: one more "
          "failure is no signal, no second run (no cooldown since "
          "2026-09-27)", json.dumps(d2.get("last_run")))
    _extends(t1, t2, "[struggle] the request after a struggle prefill")


def _plan_post_for(real):
    """A second-brain fake: the plan job's request gets PLAN; anything else
    goes to `real`."""
    def plan_post(path, payload, timeout=3600):
        if str(payload["messages"][0].get("content", "")).startswith(
                "You are planning"):
            _helper.append(json.loads(json.dumps(payload)))
            return {"choices": [{"message": {"content": PLAN},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
        return real(path, payload, timeout)
    return plan_post


# What must never reach a client's stored turn after a plan delivered as a
# tool result (pagoda-h2, 2026-09-27): the visible fold-back and a cut
# marker main could echo.
FOLD_BACK_TEXT = ("Today I was inspired by", "After thinking deeply",
                  "cut at", "continues below")


def test_the_initial_prompt_plan_is_a_hidden_yama_plan_hop():
    """pagoda-h2 (2026-09-27): the kickoff plan was prefilled under "Today I
    was inspired by falsely. After thinking deeply,"; main continued it as
    prose about the second model, finish=stop, no tool call, and Hermes
    ended the run. Now the conversation's INITIAL prompt is planned through
    an INSERTED yama_plan call and its result -- a hidden hop before main's
    first generation, as if main had called it: main continues from a tool
    result (no visible prefill, no fold-back opening, no PLAN_HEAD
    reasoning), can make its first tool call at once, the client stores no
    fold-back phrase, the ledger replays the hop byte for byte and the next
    request extends the slot (the served template)."""
    import deep
    slots.reset(n=4)
    compaction.reset()
    _helper.clear()
    real = shomen._post
    shomen._post = _plan_post_for(real)
    try:
        c = Unforced("xhigh", "kickoff session")
        spec = ("Build a space shooter with vanilla JavaScript and canvas. "
                "PLAYER: moves with arrow keys, fires with space.")
        t1 = c.turn([reply("", reasoning="Start with the page.", calls=[call(
            "write_file", {"path": "index.html", "content": "<canvas>"},
            "w1")])], user=spec)
    finally:
        shomen._post = real
    x = t1["d"]["x_yamadori"]
    req0 = t1["gens"][0]["request"]
    msgs0 = req0["messages"]
    names = [t["function"]["name"] for t in req0.get("tools") or []]
    check((x.get("deep") or {}).get("kind") == "kickoff"
          and (x.get("investigate") or {}).get("job") == "plan"
          and (x.get("investigate") or {}).get("into") == "tool_result"
          and (x.get("investigate") or {}).get("tool") == "yama_plan"
          and _helper and _helper[-1]["messages"][0]["content"]
          == shomen.plan_system(tools=bool(_helper[-1].get("tools"))),
          "[kickoff] the initial prompt runs the plan job, delivered as a "
          "tool result (x_yamadori.investigate into: tool_result)",
          json.dumps(x.get("investigate"))[:300])
    call_msg, tool_msg = msgs0[-2], msgs0[-1]
    tc = (call_msg.get("tool_calls") or [{}])[0]
    check(msgs0[-3].get("role") == "user" and msgs0[-3]["content"].startswith(
              spec)
          and call_msg.get("role") == "assistant"
          and call_msg.get("content") == ""
          and not call_msg.get("reasoning_content")
          and tc.get("function", {}).get("name") == "yama_plan"
          and json.loads(tc["function"]["arguments"])
          == deep.KICKOFF_PLAN_ARGS
          and tool_msg.get("role") == "tool"
          and tool_msg.get("tool_call_id") == tc.get("id")
          and tool_msg["content"].startswith(proxy.PLAN_RESULT_HEAD)
          and "FILES\n- index.html" in tool_msg["content"]
          and all(h in tool_msg["content"] for h in (
              "ORDER", "KEY DECISIONS", "CONSTRAINTS"))
          and tool_msg["content"].endswith(proxy.PLAN_RESULT_TAIL),
          "[kickoff] main's first generation continues from a yama_plan "
          "call and its result, inserted after the user's message: the "
          "plan (FILES, ORDER, KEY DECISIONS, CONSTRAINTS) is the tool "
          "result",
          json.dumps(msgs0[-2:])[:600])
    check("yama_plan" in names and "yama_think_deeply" in names,
          "[kickoff] yama_plan is on main's tool list beside "
          "yama_think_deeply", str(names))
    ans = t1["d"]["choices"][0]["message"]
    stored = json.dumps(c.msgs[-1])
    check(ans.get("tool_calls")
          and ans["tool_calls"][0]["function"]["name"] == "write_file"
          and t1["d"]["choices"][0].get("finish_reason") == "tool_calls"
          and not any(p in stored for p in FOLD_BACK_TEXT)
          and "yama_plan" not in stored and "FILES" not in stored,
          "[kickoff] main's first generation is a client tool call; the "
          "client stores no fold-back phrase, no marker and nothing of the "
          "hidden hop", stored[:300])
    check(not (x.get("fold_back") or [{}])[0].get("phrase")
          and any(f.get("into") == "tool_result" and f.get("tool")
                  == "yama_plan" for f in x.get("fold_back") or []),
          "[kickoff] x_yamadori.fold_back records the delivery: into "
          "tool_result, no phrase", json.dumps(x.get("fold_back")))
    c.tool_result("w1", "wrote index.html")
    t2 = c.turn([reply("", reasoning="Now the loop.", calls=[call(
        "write_file", {"path": "js/game.js", "content": "loop()"}, "w2")])])
    check(not (t2["d"]["x_yamadori"].get("deep") or {}).get("fire"),
          "[kickoff] the next step (a tool result) is not a new task")
    msgs2 = t2["gens"][0]["request"]["messages"]
    check([m for m in msgs2 if m.get("role") == "tool"][0] == tool_msg
          and any(m.get("tool_calls") == call_msg.get("tool_calls")
                  for m in msgs2 if m.get("role") == "assistant"),
          "[kickoff] the ledger replays the hidden hop byte for byte")
    _extends(t1, t2, "[kickoff] the request after the inserted plan hop")
    c.tool_result("w2", "wrote js/game.js")
    t3 = c.turn([reply("Done: the page and the loop are written.")])
    _extends(t2, t3, "[kickoff] and the request after that")
    t4 = c.turn([reply("You are welcome.")], user="Thanks.")
    check(not (t4["d"]["x_yamadori"].get("deep") or {}).get("fire")
          and len(t4["gens"]) == 1,
          "[kickoff] only the initial prompt is planned: a follow-up after "
          "a finished answer runs nothing before main")
    _extends(t3, t4, "[kickoff] a new user turn after the plan hop")


def test_the_model_calls_yama_plan_as_a_hidden_hop():
    """yama_plan (operator, 2026-09-27): main calls it before a long
    implementation task; the proxy runs the plan job and returns the plan
    as the TOOL RESULT of a hidden hop. Nothing is prefilled after it --
    main goes on acting (here: a client write). One per request: a second
    call is refused ALREADY_PLANNED; the next request extends the slot."""
    slots.reset(n=4)
    compaction.reset()
    _helper.clear()
    real = shomen._post
    shomen._post = _plan_post_for(real)
    try:
        c = Unforced("xhigh", "plan call session", tools=[WRITE, READ_TOOL])
        # An agent step, not the initial prompt: nothing runs before main.
        _agent_step_opening(c, "Now add a level editor to the game.")
        t1 = c.turn([reply("", reasoning="A big feature: plan it.",
                           calls=[call("yama_plan", {
                               "task": "Add a level editor: a grid UI, "
                                       "save/load JSON, wire it into "
                                       "js/game.js."}, "yp1")]),
                     reply("", reasoning="Plan it again.", calls=[
                         call("yama_plan", {"task": "again, the level "
                                                    "editor"}, "yp2")]),
                     reply("", reasoning="Plan in hand.", calls=[
                         call("write_file", {"path": "js/editor.js",
                                             "content": "grid()"}, "w1")])])
    finally:
        shomen._post = real
    x = t1["d"]["x_yamadori"]
    g1 = t1["gens"][1]["request"]["messages"] if len(t1["gens"]) > 1 else []
    tool = next((m for m in g1 if m.get("role") == "tool"
                 and m.get("tool_call_id") == "yp1"), {})
    pt = (x.get("deep") or {}).get("plan_tool") or {}
    check(len(t1["gens"]) == 3 and tool.get("content", "").startswith(
              proxy.PLAN_RESULT_HEAD)
          and "FILES\n- index.html" in tool["content"]
          and g1[-1].get("role") == "tool"
          and pt.get("offered") is True
          and (pt.get("calls") or [{}])[0].get("ran") is True,
          "[yama_plan] the call ran the plan job; the plan is the tool "
          "result and the next hop continues from it (no prefill)",
          json.dumps(pt)[:300])
    ans = t1["d"]["choices"][0]["message"]
    names = [cl["function"]["name"] for cl in ans.get("tool_calls") or []]
    check(names == ["write_file"]
          and not any(p in json.dumps(c.msgs[-1]) for p in FOLD_BACK_TEXT),
          "[yama_plan] the client gets only its own call, and no fold-back "
          "phrase", json.dumps(ans)[:300])
    calls2 = [m for m in t1["gens"][2]["request"]["messages"]
              if m.get("role") == "tool" and m.get("tool_call_id") == "yp2"]
    check(calls2 and '"ALREADY_PLANNED"' in calls2[0]["content"],
          "[yama_plan] a second call in the same request is refused "
          "ALREADY_PLANNED (one per request)",
          calls2[0]["content"][:200] if calls2 else "")
    c.tool_result("w1", "wrote js/editor.js")
    t2 = c.turn([reply("Editor added.")])
    _extends(t1, t2, "[yama_plan] the request after the plan hop")


def test_the_deep_tool_hop_switch():
    """Behind the deep_tool_hop switch (OFF by default; operator deciding,
    2026-09-27), deep thinking run before main by a trigger or a header is
    delivered like the initial prompt's plan: an inserted yama_think_deeply
    call and its hand-off as the tool result, no visible "After thinking
    deeply," prefill. Off (the default), it is prefilled as before."""
    import tiers
    check(not tiers.behaviour({}, "deep_tool_hop")
          and tiers.behaviour_source({}, "deep_tool_hop") == (False,
                                                              "default"),
          "[tool hop] the switch is off by default")
    slots.reset(n=4)
    compaction.reset()
    c = Unforced("xhigh", "tool hop session", tools=[WRITE, TERMINAL])
    _agent_step_opening(c, "The scene renders black. Fix it.")
    t1 = c.turn([reply("", reasoning="Patch it.", calls=[call(
        "write_file", {"path": "src/scene.js", "content": "light()"},
        "w1")])], features={"investigate": True, "deep_tool_hop": True})
    x = t1["d"]["x_yamadori"]
    msgs0 = t1["gens"][0]["request"]["messages"]
    tc = (msgs0[-2].get("tool_calls") or [{}])[0]
    check((x.get("investigate") or {}).get("into") == "tool_result"
          and tc.get("function", {}).get("name") == "yama_think_deeply"
          and msgs0[-1].get("role") == "tool"
          and "LatheGeometry sweeps" in msgs0[-1]["content"]
          and msgs0[-1]["content"].rstrip().endswith(
              proxy.THINK_RESULT_TAIL.strip()),
          "[tool hop] on: the hand-off is an inserted yama_think_deeply "
          "result; main's generation continues from it",
          json.dumps(msgs0[-2:])[:400])
    check(not any(p in json.dumps(c.msgs[-1]) for p in FOLD_BACK_TEXT),
          "[tool hop] on: no fold-back phrase reaches the client")
    c.tool_result("w1", "ok")
    t2 = c.turn([reply("Fixed.")])
    _extends(t1, t2, "[tool hop] the request after the inserted hop")
    # Off: the same header-forced run is prefilled, as before.
    slots.reset(n=4)
    d = Unforced("xhigh", "tool hop off session", tools=[WRITE, TERMINAL])
    _agent_step_opening(d, "The scene renders black. Fix it.")
    u1 = d.turn([reply(" the light is missing.")],
                features={"investigate": True})
    pre = u1["gens"][0]["request"]["messages"][-1]
    check((u1["d"]["x_yamadori"].get("investigate") or {}).get("into")
          == "prefill" and pre.get("role") == "assistant"
          and pre.get("content", "").endswith("After thinking deeply,"),
          "[tool hop] off (the default): prefilled under the fold-back "
          "opening, unchanged", json.dumps(pre)[:200])


def test_old_tool_names_still_replay_and_run():
    """The yama_* rename (2026-09-27): a hidden hop stored under an OLD name
    (think_deeply, recall_craft, ...) replays byte for byte, and a call the
    model makes by an old name -- copied from such a hop -- runs as the new
    tool instead of going to the client."""
    slots.reset(n=4)
    compaction.reset()
    _helper.clear()
    c = Unforced("xhigh", "legacy names", tools=[WRITE, READ_TOOL])
    _agent_step_opening(c, "Why does my LatheGeometry render inside out?")
    t1 = c.turn([reply("", reasoning="Old habit.", calls=[call(
                     "think_deeply", {"question": "Why must LatheGeometry "
                                                  "points be sorted by y?"},
                     "old1")]),
                 reply(" sort the points by y.")])
    x = t1["d"]["x_yamadori"]
    calls = ((x.get("deep") or {}).get("think_tool") or {}).get("calls") or []
    ans = t1["d"]["choices"][0]["message"]
    check(calls and calls[0].get("ran") is True and not ans.get("tool_calls")
          and len(t1["gens"]) == 2,
          "[legacy] a call by the old name think_deeply runs as "
          "yama_think_deeply; the client never sees it",
          json.dumps(calls)[:300])
    t2 = c.turn([reply("You are welcome.")], user="Thanks.")
    msgs2 = t2["gens"][0]["request"]["messages"]
    check(any(m.get("role") == "assistant" and any(
                  cl["function"]["name"] == "think_deeply"
                  for cl in m.get("tool_calls") or []) for m in msgs2),
          "[legacy] the stored hop replays with the name it was stored "
          "under")
    _extends(t1, t2, "[legacy] the request after an old-name hop")
    check(proxy.canonical_tool_name("describe_image")
          == "yama_describe_image"
          and proxy.canonical_tool_name("generate_image")
          == "yama_generate_image"
          and proxy.canonical_tool_name("recall_craft") == "yama_recall_craft"
          and proxy.canonical_tool_name("write_file") == "write_file"
          and not set(proxy.LEGACY_TOOL_NAMES) & set(
              proxy.client_tool_names({"tools": [
                  {"type": "function", "function": {"name": n}}
                  for n in proxy.LEGACY_TOOL_NAMES]})),
          "[legacy] every old name maps to its yama_* name, and is never "
          "counted as the client's")


def _div(a: str, b: str) -> int:
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y),
                min(len(a), len(b)))


def test_a_prefilled_turn_warms_from_the_last_user_checkpoint():
    """Octopus v0b-V0-xhigh-1, step 1 (2026-09-25): a kickoff, the plan
    prefilled as main's reasoning, prompt 10660; the warm reused 7526,
    processed 3055 and was flagged SHORT against 10660, and the next request
    reused 10581 of 10611. Rendered here with the served template, one token
    per character: the warm renders the turn STRIPPED, as the client sends
    it (the plan is not in it), so it is right; the step-1 prompt diverges
    from it where the prefilled plan begins, after the last user message,
    and the server can restore only its checkpoint at that message's start.
    That is what a prefilled turn's warm is judged against; a real miss
    (reused 0) is still SHORT. The live shape -- a terminal call, nothing
    for the proxy to change -- is warmed too: the slot holds the plan as
    the turn's reasoning, which the client never sends back.

    Since 2026-09-27 the initial prompt's plan is NOT prefilled (it is an
    inserted yama_plan call and result, test_the_initial_prompt_plan_is_a_
    hidden_yama_plan_hop); a first turn whose deep thinking a HEADER forces
    is still prefilled (switch deep_tool_hop off), so that is the shape
    here -- the same prefill, the same checkpoint rule.

    THE PASS-THROUGH ARM (switch restore_reasoning off): since 2026-09-27
    the hand-off comes back with the turn and no warm is needed
    (test_a_prefilled_turn_is_the_slots_sequence_when_reasoning_is_restored).
    """
    global BY_CHARS
    saved = dict(EXTRA_FEATURES)
    EXTRA_FEATURES["restore_reasoning"] = False
    try:
        _prefilled_warm_pass_through()
    finally:
        EXTRA_FEATURES.clear()
        EXTRA_FEATURES.update(saved)


def _prefilled_warm_pass_through():
    global BY_CHARS
    real = shomen._post

    def plan_post(path, payload, timeout=3600):
        if str(payload["messages"][0].get("content", "")).startswith(
                "You are planning"):
            return {"choices": [{"message": {"content": PLAN},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
        return real(path, payload, timeout)
    spec = ("Build a space shooter with vanilla JavaScript and canvas. "
            "PLAYER: moves with arrow keys, fires with space. " * 120)
    shots = (("terminal (the live shape)", "last_user",
              call("terminal", {"command": "mkdir -p css js"}, "k1")),
             ("a verified write", "last_user",
              call("write_file", {"path": "hi.py", "content": FIXED}, "k1")),
             ("a real miss", "zero",
              call("terminal", {"command": "mkdir -p css js"}, "k1")))
    shomen._post = plan_post
    try:
        for label, mode, first in shots:
            slots.reset(n=4)
            compaction.reset()
            BY_CHARS = mode
            tag = f"[prefill warm: {label}]"
            c = Unforced("xhigh", f"prefill warm {label}",
                         tools=[WRITE, TERMINAL])
            t1 = c.turn([reply(" I will start.", reasoning="Go.",
                               calls=[first])], user=spec,
                        features={"investigate": True})
            c.tool_result("k1", "ok")
            t2 = c.turn([reply("Next.")])
            a = render(t1["gens"][0]["request"])     # step 1 as sent
            w = t1["warm"][-1]["prompt"] if t1["warm"] else ""
            r = render(t2["gens"][0]["request"])     # what the client sent
            x1 = t1["d"]["x_yamadori"]
            wb = t2["d"]["x_yamadori"].get("warm_before") or {}
            u = a.rfind("<|im_start|>user")
            n = _div(a, w)
            if mode == "last_user":
                check(x1["investigate"].get("job") == "investigate"
                      and x1["investigate"].get("into") == "prefill" and w
                      and "NEXT STEP" in a[u:] and "NEXT STEP"
                      not in w and a[:n].endswith(
                          "<|im_start|>assistant\n<think>\n") and n > u
                      and r.startswith(w),
                      f"{tag} RENDER: the warm is the turn as the client "
                      f"sends it (no hand-off); the step-1 prompt diverges "
                      f"from it where the prefilled hand-off begins (char {n}), "
                      f"after the last user message (char {u})",
                      json.dumps({"warm": x1.get("warm"), "div": n, "u": u,
                                  "at": a[max(0, n - 40):n + 40]}))
            check(wb.get("state") == "done" and wb.get("prefilled") is True
                  and wb.get("generated_prompt") == len(a)
                  and wb.get("expect_reused_at_least") == u
                  and wb.get("short") is (mode == "zero"),
                  f"{tag} the warm is judged against the checkpoint at the "
                  f"last user message ({u}), not the {len(a)}-token prompt "
                  f"with the prefill: short={mode == 'zero'}",
                  json.dumps(wb))
    finally:
        shomen._post = real
        BY_CHARS = None


HANDOFF_LONG = ("FACTS\n" + "".join(
    f"- step {i}: Player.unleash counts down in update() and the ring is "
    f"drawn from it (reasoning, not checked against source)\n"
    for i in range(30))
    + "SEARCHED, FOUND NOTHING\n- none\nOPEN QUESTIONS\n- none\n"
    "NEXT STEP\n- patch js/player.js\n")


def _mid_prefill_session(tag: str, mode: str,
                         prefill_reasoning: str = "Patch.") -> dict:
    """An agent session of terminal steps on one slot of the simulated
    server (BY_CHARS "server"), then a step whose hand-off is prefilled
    (forced), its warm served in `mode`, then the next step.
    `prefill_reasoning`: what the fake adds to the prefilled reasoning
    (a real server generates none after a closed think block)."""
    global BY_CHARS, SIM_MODE
    real = shomen._post

    def long_post(path, payload, timeout=3600):
        _helper.append(json.loads(json.dumps(payload)))
        return {"choices": [{"message": {"content": HANDOFF_LONG},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
    slots.reset(n=4)
    compaction.reset()
    _sim.clear()
    _sim_log.clear()
    BY_CHARS, SIM_MODE = "server", "thinned"
    shomen._post = long_post
    try:
        c = Unforced("xhigh", f"mid prefill {tag}", tools=[WRITE, TERMINAL])
        user = "Make the unleash ring fade out in js/player.js."
        for i in range(9):
            cid = f"s{i}"
            c.turn([reply(f" Step {i}.", reasoning=f"Look at part {i}.",
                          calls=[call("terminal", {"command": f"sed -n "
                                                   f"{i}p js/player.js"},
                                      cid)])],
                   user=user if i == 0 else None)
            c.tool_result(cid, "".join(f"line {i}.{k}: ring.alpha -= dt;\n"
                                       for k in range(130)))
        n_log = len(_sim_log)
        SIM_MODE = mode
        t = c.turn([reply(" I will patch the countdown.",
                          reasoning=prefill_reasoning,
                          calls=[call("terminal", {"command": "node --check "
                                                   "js/player.js"}, "sp")])],
                   features={"investigate": True})
        c.tool_result("sp", "ok")
        # The next request waits for the warm (wait_for_warm), which is
        # served in `mode` (it applies to warms only).
        t2 = c.turn([reply(" Checked.", reasoning="Fine.")])
        return {"t": t, "t2": t2, "log": _sim_log[n_log:],
                "all": list(_sim_log)}
    finally:
        shomen._post = real
        BY_CHARS, SIM_MODE = None, "thinned"


def test_a_prefill_mid_conversation_rereads_to_the_thinned_checkpoint():
    """Octopus v0f-V0-xhigh-1 step 25 (2026-09-26): a struggle hand-off
    prefilled as main's reasoning at prompt 63,397 (reused 59,343); the warm
    "reused 54,604 processed 8,280" for a 503-token step, and the watcher
    flagged a re-read. REPRODUCED here with the served template and a
    simulation of llama-server's checkpoints (_sim_serve, one token per
    character): the warm diverges where the hand-off begins (the client
    drops it), and the prefilled generation's own checkpoints -- the only
    ones near the end -- lie inside the hand-off, having THINNED the
    previous request's (within checkpoint_min_step of an older one); so
    the warm resumes at that older checkpoint and re-reads the conversation
    between it and the turn. That is inherent (proxy A PREFILL
    MID-CONVERSATION); what changes is the record: the expectation is the
    thinning bound below what the generation resumed at (not the last user
    message, which passed any miss), `reread_before_turn` measures the
    cost, and a warm that resumed further back than the bound is SHORT.

    THE PASS-THROUGH ARM (switch restore_reasoning off): with reasoning
    restored the prefilled turn is the slot's sequence and nothing is
    re-read (test_a_prefill_mid_conversation_extends_when_reasoning_is_
    restored)."""
    saved = dict(EXTRA_FEATURES)
    EXTRA_FEATURES["restore_reasoning"] = False
    try:
        _mid_prefill_pass_through()
    finally:
        EXTRA_FEATURES.clear()
        EXTRA_FEATURES.update(saved)


def _mid_prefill_pass_through():
    r = _mid_prefill_session("thinned", "thinned")
    t, t2, log = r["t"], r["t2"], r["log"]
    x = t["d"]["x_yamadori"]
    wb = t2["d"]["x_yamadori"].get("warm_before") or {}
    gen = next((g for g in log if g["kind"] == "gen"), {})
    warm = next((g for g in log if g["kind"] == "warm"), {})
    nxt = [g for g in log if g["kind"] == "gen"][1:2]
    w = t["warm"][-1]["prompt"] if t["warm"] else ""
    a = render(t["gens"][0]["request"])
    d = _div(a, w)
    turn_at = w.rfind("<|im_start|>assistant")
    check((x.get("investigate") or {}).get("ran") and w and gen and warm
          and d < len(a) - 500 and a[:d].endswith(
              "<|im_start|>assistant\n<think>\n"),
          "[mid prefill] the warm renders the turn as the client sends it "
          "and diverges from the prefilled prompt where the hand-off begins",
          json.dumps({"div": d, "prompt": len(a), "gen": gen, "warm": warm}))
    check(warm.get("reused", 0) < gen.get("reused", 0)
          and warm.get("reused", 0) < turn_at,
          "[mid prefill] REPRODUCED: the warm resumes BEFORE the checkpoint "
          "the generation resumed at (the generation's own checkpoints "
          "thinned it) and re-reads conversation before the turn",
          json.dumps({"gen": gen, "warm": warm, "turn_at": turn_at}))
    bound = gen.get("reused", 0) - proxy.CHECKPOINT_MIN_STEP - 4
    check(wb.get("prefilled") is True and wb.get("generated_reused")
          == gen.get("reused") and wb.get("expect_reused_at_least") == bound
          and warm.get("reused", 0) >= bound and wb.get("short") is False,
          f"[mid prefill] the warm is judged against the thinning bound "
          f"({bound}), which it meets: not SHORT",
          json.dumps(wb))
    check(wb.get("turn_starts_at") == turn_at
          and wb.get("reread_before_turn") == turn_at - warm.get("reused", 0)
          and "thinning" in (wb.get("expect_why") or ""),
          "[mid prefill] the record measures the re-read before the turn "
          "and says why", json.dumps(wb))
    check(nxt and nxt[0]["reused"] == len(w),
          "[mid prefill] the next request extends the warm (it processes "
          "only its own tail)", json.dumps({"next": nxt, "warm": len(w)}))
    # A REAL MISS: the server resumed at an OLD checkpoint -- above the
    # last user message, far below the bound. The old expectation (the last
    # user message) passed it.
    r = _mid_prefill_session("miss", "old")
    log = r["log"]
    gen = next((g for g in log if g["kind"] == "gen"), {})
    warm = next((g for g in log if g["kind"] == "warm"), {})
    wb = r["t2"]["d"]["x_yamadori"].get("warm_before") or {}
    u = _last_user_start_chars(r["t"]["warm"][-1]["prompt"]) \
        if r["t"]["warm"] else -1
    check(warm and u < warm.get("reused", 0)
          < gen.get("reused", 0) - proxy.CHECKPOINT_MIN_STEP - 4 - 8
          and wb.get("short") is True,
          "[mid prefill] a warm that resumed further back than the bound "
          "(but after the last user message) is SHORT",
          json.dumps({"u": u, "warm": warm, "wb": wb}))


def _last_user_start_chars(prompt: str) -> int:
    return proxy._last_user_start(prompt)


def test_the_served_template_renders_every_past_think_block():
    """THE TEMPLATE FINDING (2026-09-27): the served template (the GGUF's
    own, byte-identical to the fixture) renders a past assistant turn's
    think block when `preserve_thinking is undefined or preserve_thinking is
    true or loop.index0 > ns.last_query_index` -- the Qwen3.8 default. The
    proxy never sets it, so every past turn -- before the last user query
    too -- renders `<think>\\n{reasoning|trim}\\n</think>\\n\\n`, an EMPTY
    block when the client dropped the reasoning. Nothing to pass: restoring
    the reasoning is all it takes."""
    msgs = [{"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
            {"role": "user", "content": "q2"}]
    on = render({"messages": msgs})
    off = _TEMPLATE.render(messages=_as_template_input(msgs), tools=None,
                           add_generation_prompt=True, raise_exception=_raise,
                           preserve_thinking=False)
    bare = render({"messages": [dict(m) if m["role"] != "assistant" else
                                {"role": "assistant", "content": "a1"}
                                for m in msgs]})
    check("<think>\nr1\n</think>\n\na1" in on and "r1" not in off
          and "<think>\n\n</think>\n\na1" in bare,
          "the served template keeps a past turn's reasoning before the last "
          "user query by default (preserve_thinking undefined); false drops "
          "it; a stripped turn renders an empty think block",
          json.dumps({"on": on[-120:], "off": off[-80:]}))
    sent = [g["request"] for g in _gens]
    check(not any("preserve_thinking" in (b.get("chat_template_kwargs") or {})
                  for b in sent),
          "the proxy never sends preserve_thinking: the template's default "
          "(on) applies", f"{len(sent)} requests")


def test_a_prefilled_turn_is_the_slots_sequence_when_reasoning_is_restored():
    """PAST REASONING IS RESTORED (operator, 2026-09-27): a turn whose
    deep-thinking hand-off was prefilled as main's reasoning comes back
    WITH the hand-off (the ledger recorded what the slot holds), so the
    next request extends the slot -- no checkpoint re-read, and nothing to
    warm when the proxy changed nothing in the turn (the live shape: a
    terminal call). A turn the proxy did change (a verified write's note)
    is warmed with the hand-off in it."""
    global BY_CHARS
    real = shomen._post

    def plan_post(path, payload, timeout=3600):
        if str(payload["messages"][0].get("content", "")).startswith(
                "You are planning"):
            return {"choices": [{"message": {"content": PLAN},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
        return real(path, payload, timeout)
    spec = ("Build a space shooter with vanilla JavaScript and canvas. "
            "PLAYER: moves with arrow keys, fires with space. " * 40)
    shots = (("terminal (the live shape)",
              call("terminal", {"command": "mkdir -p css js"}, "k1")),
             ("a verified write",
              call("write_file", {"path": "hi.py", "content": FIXED}, "k1")))
    shomen._post = plan_post
    try:
        for label, first in shots:
            slots.reset(n=4)
            compaction.reset()
            BY_CHARS = None
            tag = f"[prefill restored: {label}]"
            c = Unforced("xhigh", f"prefill restored {label}",
                         tools=[WRITE, TERMINAL])
            # reasoning "": a real server generates none after the closed
            # think block of a prefill.
            t1 = c.turn([reply(" I will start.", reasoning="",
                               calls=[first])], user=spec,
                        features={"investigate": True})
            c.tool_result("k1", "ok")
            t2 = c.turn([reply("Next.")])
            x1, x2 = t1["d"]["x_yamadori"], t2["d"]["x_yamadori"]
            r = render(t2["gens"][0]["request"])
            held = reusable_after(t1)
            last = r[:len(held)].rfind("<|im_start|>assistant")
            check((x1.get("investigate") or {}).get("into") == "prefill"
                  and r.startswith(held) and "NEXT STEP" in r[last:],
                  f"{tag} the next request extends the slot: the prefilled "
                  f"hand-off is in the turn it renders",
                  json.dumps({"warm": x1.get("warm"),
                              "div": _div(held, r), "held": len(held)}))
            lg = x2.get("ledger") or {}
            check((lg.get("restore_reasoning") or {}).get("on") is True
                  and (lg.get("restored") or {}).get("reasoning", 0) >= 1
                  and (x1.get("ledger") or {}).get("reasoning_recorded", 0)
                  > len("NEXT STEP"),
                  f"{tag} x_yamadori.ledger: the switch, the turn recorded, "
                  f"the reasoning restored",
                  json.dumps({"t1": x1.get("ledger"), "t2": lg}))
            if label.startswith("terminal"):
                check(not (x1.get("warm") or {}).get("sent")
                      and "already holds" in (x1.get("warm") or {}).get(
                          "why", ""),
                      f"{tag} nothing to warm: the slot holds the turn as "
                      f"the next request renders it",
                      json.dumps(x1.get("warm")))
            else:
                w = t1["warm"][-1]["prompt"] if t1["warm"] else ""
                check(w and "NEXT STEP" in w[w.rfind(
                          "<|im_start|>assistant"):]
                      and "Verified hi.py" in w,
                      f"{tag} the warm loads the delivered turn WITH its "
                      f"hand-off", json.dumps(x1.get("warm")))
    finally:
        shomen._post = real
        BY_CHARS = None


def test_a_prefill_mid_conversation_extends_when_reasoning_is_restored():
    """The Octopus v0f-V0 step-25 shape (a struggle hand-off prefilled
    mid-conversation) on the simulated server, reasoning restored: every
    request of the session -- the nine agent steps, the prefilled step and
    the one after it -- EXTENDS what the slot holds (no checkpoint restore,
    reused == held), and the prefilled turn needs no warm. Before the
    switch the step after a prefill re-read back to a thinned checkpoint
    (8,793 tokens live; test_a_prefill_mid_conversation_rereads_to_the_
    thinned_checkpoint, the pass-through arm)."""
    r = _mid_prefill_session("restored", "thinned", prefill_reasoning="")
    x = r["t"]["d"]["x_yamadori"]
    gens = [g for g in r["all"] if g["kind"] == "gen" and g["held"]]
    bad = [g for g in gens if not g["extends"] or g["reused"] != g["held"]]
    check((x.get("investigate") or {}).get("ran") and len(gens) >= 10
          and not bad,
          f"[mid prefill restored] all {len(gens)} requests after the first "
          f"extend the slot: reused == what it held, nothing re-read",
          json.dumps(bad[:3]))
    check(not (x.get("warm") or {}).get("sent")
          and not [g for g in r["log"] if g["kind"] == "warm"],
          "[mid prefill restored] the prefilled turn needs no warm",
          json.dumps(x.get("warm")))
    nxt = [g for g in r["log"] if g["kind"] == "gen"][1:2]
    check(nxt and nxt[0]["processed"] < 400,
          "[mid prefill restored] the step after the hand-off processes only "
          "the tool result and the new part",
          json.dumps(nxt))


def test_the_window_check_counts_restored_reasoning():
    """Restored reasoning adds tokens to every request (V0 pilot: 6-10k a
    step). The client's window check (proxy.check_client_prompt) measures
    the request as it goes up -- the ledger's additions included -- so a
    request that fits as the client sent it but not with its reasoning
    restored is refused 400 context_length_exceeded (Hermes compacts on
    it); with the switch off the same request is served. The count is the
    served template's render and the fake tokenizer's (one token a
    character); the high estimate is forced high so the count is made."""
    slots.reset(n=4)
    compaction.reset()
    c = Unforced("xhigh", "window restore", tools=[WRITE])
    long_r = "I need to weigh how f should be written. " * 150   # ~6.3k
    c.turn([reply("Writing it.", reasoning=long_r, calls=[call(
        "write_file", {"path": "hi.py", "content": FIXED}, "wz")])],
        user="Write hi.py with a function f that returns 1.")
    c.tool_result("wz", "wrote hi.py")
    est = proxy.high_estimate
    shares = tiers._shares
    proxy.high_estimate = lambda payload, messages=None: 10 ** 9
    try:
        on = proxy.prepare(c.body())
        off = proxy.prepare(c.body({"restore_reasoning": False}))
        n_on = proxy.count_prompt_tokens(on)
        n_off = proxy.count_prompt_tokens(off)
        floor = proxy.generation_floor(on)
        limit = (n_on + n_off) // 2 + floor
        tiers._shares = lambda: {"main": limit, "helper": 1000,
                                 "pool": limit + 1000}
        refused = None
        _script[:] = [reply("Done.")]
        try:
            proxy.complete(c.body())
        except proxy.api_errors.ApiError as e:
            refused = e
        check(n_on - n_off >= len(long_r.strip()) and refused is not None
              and refused.code == "context_length_exceeded"
              and refused.extra.get("n_prompt_tokens") == n_on,
              "restored reasoning is counted: the request that no longer "
              "fits its window is refused context_length_exceeded",
              json.dumps({"n_on": n_on, "n_off": n_off, "limit": limit,
                          "floor": floor, "refused": refused.body()
                          if refused else None})[:500])
        _script[:] = [reply("Done.")]
        d = proxy.complete(c.body({"restore_reasoning": False}))
        ctx = (d.get("x_yamadori") or {}).get("context") or {}
        check(d["choices"][0]["message"].get("content") == "Done."
              and ctx.get("tokens") == n_off,
              "switch off (pass-through): the same request fits and is "
              "served", json.dumps(ctx))
    finally:
        proxy.high_estimate = est
        tiers._shares = shares
        _script.clear()


def test_prefix_stability_across_a_session():
    for effort in ("xhigh", "max"):
        _helper.clear()
        session(effort)
    check("Reasoning effort is set to xhigh" in render(
              {"messages": [{"role": "user", "content": "x"}],
               "reasoning_effort": "xhigh",
               "chat_template_kwargs": {"enable_thinking": True}}),
          "the renderer is the served template (it prints the xhigh effort "
          "line)")


def test_scope_persistence_and_pruning():
    nebari.ledger_put("a1", "s1", "k1", "inject", "hello")
    check(nebari.ledger_get("a1", "k1", "inject") == "hello",
          "a write is read back")
    check(nebari.ledger_get("a2", "k1", "inject") is None,
          "another account sees nothing")
    nebari.ledger_reset()
    check(nebari.ledger_get("a1", "k1", "inject") == "hello",
          "after a restart (memory emptied) it is read from sqlite")
    prev = nebari.LEDGER_PERSIST_REASONING
    nebari.LEDGER_PERSIST_REASONING = False
    try:
        nebari.ledger_put("a1", "s1", "k2", "reasoning", "thought")
        nebari.ledger_reset()
        check(nebari.ledger_get("a1", "k2", "reasoning") is None,
              "YAMADORI_LEDGER_PERSIST_REASONING=0: reasoning is memory only")
    finally:
        nebari.LEDGER_PERSIST_REASONING = prev
    saved = (nebari.LEDGER_ACCOUNT_BYTES, nebari.LEDGER_TOTAL_BYTES,
             nebari.LEDGER_MAX_AGE)
    try:
        nebari.ledger_forget_account("a1")
        for i, sess in enumerate(("old", "mid", "new")):
            nebari.ledger_put("p1", sess, f"k{i}", "inject", "x" * 100)
            time.sleep(0.01)
        nebari.ledger_put("p2", "other", "kz", "inject", "y" * 100)
        nebari.LEDGER_ACCOUNT_BYTES = 250
        r = nebari.ledger_prune()
        check(r["account_cap"] == 1
              and nebari.ledger_get("p1", "k0", "inject") is None
              and nebari.ledger_get("p1", "k2", "inject") == "x" * 100,
              "past an account's cap, its least recently seen session goes "
              "first", json.dumps(r))
        nebari.LEDGER_ACCOUNT_BYTES = 10 ** 9
        nebari.LEDGER_TOTAL_BYTES = 250
        r = nebari.ledger_prune()
        check(r["total_cap"] == 1
              and nebari.ledger_get("p1", "k1", "inject") is None
              and nebari.ledger_get("p2", "kz", "inject") == "y" * 100,
              "then the global cap, across accounts, oldest first",
              json.dumps(r))
        nebari.LEDGER_TOTAL_BYTES = 10 ** 9
        r = nebari.ledger_prune(now=time.time() + nebari.LEDGER_MAX_AGE + 60)
        check(r["aged"] >= 2 and nebari.ledger_get("p2", "kz", "inject")
              is None, "and past the max age everything goes (a privacy "
              "bound)", json.dumps(r))
    finally:
        (nebari.LEDGER_ACCOUNT_BYTES, nebari.LEDGER_TOTAL_BYTES,
         nebari.LEDGER_MAX_AGE) = saved
    check(nebari.LEDGER_ACCOUNT_BYTES == 256 * 1024 * 1024
          and nebari.LEDGER_TOTAL_BYTES == 2048 * 1024 * 1024
          and nebari.LEDGER_MAX_AGE == 30 * 86400,
          "the caps' defaults: 256 MB per account, 2048 MB in all, 30 days "
          "(choices, operator 2026-09-24)")


def test_keys():
    a = [{"role": "user", "content": "continue"},
         {"role": "assistant", "content": "ok"},
         {"role": "user", "content": "continue"}]
    k = proxy.chain_keys(a)
    check(k[0] != k[2], "the same words at two points of a conversation are "
          "two keys")
    b = [dict(m) for m in a]
    b[1]["reasoning_content"] = "echoed trace"
    check(proxy.chain_keys(b) == k,
          "a client echoing reasoning (or not) does not change the keys")
    c = [dict(m) for m in a]
    c[1]["content"] = "OK"
    check(proxy.chain_keys(c)[2] != k[2],
          "an edited earlier turn changes every key after it")


class SortingClient(Client):
    """Hermes' send path: every historical call's arguments re-serialised
    with sorted keys (agent/conversation_loop.py
    _canonicalize_tool_call_arguments: json.dumps(json.loads(a),
    separators=(",", ":"), sort_keys=True)); what it STORES keeps the order
    it was sent."""

    def body(self, features: dict | None = None) -> dict:
        b = super().body(features)
        for m in b["messages"]:
            for c in m.get("tool_calls") or []:
                fn = c["function"]
                fn["arguments"] = json.dumps(json.loads(fn["arguments"]),
                                             separators=(",", ":"),
                                             sort_keys=True)
        return b


def test_call_arguments_render_in_one_order():
    """Octopus v0e-V0-xhigh-1, 2026-09-25, step 6: reused 16187 of 25487
    although the warm after step 5 had loaded 25258. Step 5 was a patch the
    model wrote as path, old_string, new_string; the warm rendered that order
    and Hermes sent it back sorted (new_string first), so the request
    diverged ~80 tokens into the warmed turn. Every copy the proxy renders
    now sorts the keys (proxy ONE ORDER FOR TOOL-CALL ARGUMENTS), so a
    sorting client and an order-keeping one both extend the warm -- with a
    note (medium: the delivered turn differs anyway) and without one (low:
    the order alone makes the turn differ from what the slot generated)."""
    unsorted = [call("patch", {"path": "a.js", "old_string": "var a = 1;",
                               "new_string": "let a = 1;"}, "p1"),
                call("write_file", {"path": "b.js",
                                    "content": "let b = 2;\n"}, "w1")]
    for effort in ("medium", "low"):
        for cls, who in ((SortingClient, "sorting client (Hermes)"),
                         (Client, "order-keeping client")):
            slots.reset(n=4)
            compaction.reset()
            c = cls(effort)
            tag = f"[arg order {effort}, {who}]"
            t1 = c.turn([reply("Patching a.js, writing b.js.",
                               reasoning="Fix it.",
                               calls=json.loads(json.dumps(unsorted)))],
                        user=f"Fix a.js and write b.js. {tag}")
            check(bool(t1["warm"]),
                  f"{tag} a turn whose calls are not in sorted key order is "
                  f"warmed", json.dumps(t1["d"]["x_yamadori"].get("warm")))
            if t1["warm"]:
                p = t1["warm"][-1]["prompt"]
                check("<parameter=new_string>" in p and p.index(
                      "<parameter=new_string>") < p.index(
                      "<parameter=old_string>") < p.index("<parameter=path>"),
                      f"{tag} the warm renders the patch's keys sorted",
                      p[-400:])
            c.tool_result("p1", "patched a.js")
            c.tool_result("w1", "wrote b.js")
            t2 = c.turn([reply("Done.", reasoning="ok")])
            _extends(t1, t2, f"{tag} the next request extends the warm")


# ================================================ the overthinking traps ===
# docs/research/OVERTHINKING.md; docs/SELF-IMPROVEMENT-LOG.md #50-#56. Every
# line the proxy adds to a tool result goes through the ledger: decided once,
# replayed byte for byte, so each request still extends the slot (through
# the served template, _extends).
READ_TOOL = {"type": "function", "function": {
    "name": "read_file", "description": "Read a file.",
    "parameters": {"type": "object", "properties": {
        "path": {"type": "string"}, "offset": {"type": "integer"},
        "limit": {"type": "integer"}}}}}


# REMOVED 2026-09-27 (operator: "keep our system prompts clean; fixes go
# through skills" -- task-targeted steering in prompts; skills are the
# channel): the tool-result situations (#50's progress line, #54's
# unchanged-read line and the work log's files-read listing) and the kickoff
# plan's confirm-cwd (#32) and entry-first (#55) rows. None of their text
# may reach any request the proxy sends (test_no_removed_steering_reaches_
# any_request scans every generation, warm and second-brain request this
# suite made); a tool result that already carries one in the ledger replays
# it byte for byte (below).
REMOVED_STEERING = (
    "No project file has changed", "Unchanged since step",
    "gave the same text then", "Files read in this conversation",
    "Confirm the working directory", "RELATIVE to the working directory",
    "relative to the working directory", "the task is a browser app",
    "so the page opens from the first step on",
    "with the script tags for the modules",
    # NO CAP ON WHAT CROSSES (2026-09-27, pagoda-h2): a hand-off or plan
    # cut mid-text under "[hand-off cut at N characters of M]", which main
    # echoed as "[The rest of the second model's response continues
    # below]". No truncation text of ours reaches any request.
    "cut at", "continues below", "hand-off cut", "more; trace handle")
# Generations made by the replay test below, which replays an OLD row on
# purpose: (first, last) indexes into _gens, excluded from the scan.
_OLD_ROW_GENS: list[tuple[int, int]] = []
OLD_SITUATION = ("\n\n---\nNo project file has changed in the last 8 steps "
                 "(23 min); the last change was js/player.js at 14:02.")


def _salt_spy():
    """Record the salt the proxy keys this conversation's messages with."""
    seen: list[str] = []
    real = proxy.chain_keys

    def spy(messages, salt=""):
        seen.append(salt)
        return real(messages, salt)
    return seen, real, spy


def test_no_situation_line_and_old_rows_replay():
    """Removed 2026-09-27: after many steps with no project write, and on an
    identical re-read, the tool result goes up as the client sent it -- no
    line. A tool result whose situation line an older build recorded in the
    ledger replays it byte for byte, so that conversation's slot prefix
    holds."""
    import progress
    slots.reset(n=4)
    compaction.reset()
    c = Unforced("xhigh", "no situations", tools=[WRITE, READ_TOOL, TERMINAL])
    feats = {"investigate": False, "skills": False}
    turns = [c.turn([reply("", reasoning="Write it.", calls=[call(
        "write_file", {"path": "js/player.js",
                       "content": "const PLAYER = {};\n"}, "p0")])],
        user="Fix `PLAYER.hit is not a function` in js/player.js.",
        features=feats)]
    c.tool_result("p0", '{"bytes_written": 20}')
    body = '{"content": "1|const GAME = {};"}'
    for i in range(1, 12):
        turns.append(c.turn([reply("", reasoning="Look.", calls=[call(
            "read_file", {"path": "js/game.js"}, f"p{i}")])],
            features=feats))
        c.tool_result(f"p{i}", body)
    sent = [m.get("content") or "" for t in turns[1:]
            for m in t["gens"][0]["request"]["messages"]
            if m.get("role") == "tool"]
    check(sent and all(t == body or t == '{"bytes_written": 20}'
                       for t in sent),
          "[situations] 11 steps with no project write, every one an "
          "identical re-read: each tool result goes up exactly as sent",
          json.dumps(sorted(set(sent)))[:300])
    x = turns[-1]["d"]["x_yamadori"]
    prog = x.get("progress") or {}
    check("situations" not in prog
          and not {"progress_note", "unchanged_read"} & set(
              prog.get("switches") or {})
          and (prog.get("project") or {}).get("named") == 1,
          "[situations] x_yamadori.progress has no situations and no "
          "switch for them; the project is still learned",
          json.dumps(prog)[:400])
    for a, b in zip(turns[1:], turns[2:]):
        _extends(a, b, "[situations] each step extends the last")
    # AN OLD ROW. A conversation whose tool result an older build decorated:
    # the row is in the ledger before the request that ends on it.
    slots.reset(n=4)
    d = Unforced("xhigh", "old situation row", tools=[WRITE, READ_TOOL])
    seen, real, spy = _salt_spy()
    proxy.chain_keys = spy
    try:
        d.turn([reply("", reasoning="Read.", calls=[call(
            "read_file", {"path": "js/game.js"}, "o1")])],
            user="Fix the score in js/game.js.", features=feats)
    finally:
        proxy.chain_keys = real
    d.tool_result("o1", body)
    account = f"{ACCOUNT}-{d.tag}"
    lineage = proxy._LAST_SESSION[account][2]
    salt = next((x for x in reversed(seen) if x), "")
    key = proxy.chain_keys(d.msgs, salt)[-1]
    nebari.ledger_decide(account, lineage, key, "inject", OLD_SITUATION,
                         {"parts": ["situations"]})
    n0 = len(_gens)
    t2 = d.turn([reply("", reasoning="Again.", calls=[call(
        "read_file", {"path": "js/game.js"}, "o2")])], features=feats)
    d.tool_result("o2", body)
    t3 = d.turn([reply("Done.", reasoning="ok")], features=feats)
    _OLD_ROW_GENS.append((n0, len(_gens)))
    got2 = [m.get("content") for m in t2["gens"][0]["request"]["messages"]
            if m.get("role") == "tool"]
    got3 = [m.get("content") for m in t3["gens"][0]["request"]["messages"]
            if m.get("role") == "tool"]
    check(got2 == [body + OLD_SITUATION]
          and got3 == [body + OLD_SITUATION, body],
          "[old row] a stored situation line replays on its tool result "
          "byte for byte, on the request that ends on it and every later "
          "one; the new tool result gets none",
          json.dumps([got2, got3])[:500])
    _extends(t2, t3, "[old row] the request after the replayed line")
    check(not hasattr(progress, "observe"),
          "[old row] replay needs no generator: nothing regenerates it")


def test_no_removed_steering_reaches_any_request():
    """Every request this suite sent upstream -- each generation's body and
    its rendering through the served template, every warm prompt, every
    second-brain request -- is scanned for the removed text (the replay
    test's old row excepted)."""
    skip = {i for a, b in _OLD_ROW_GENS for i in range(a, b)}
    hits = []
    for i, g in enumerate(_gens):
        if i in skip:
            continue
        req = g["request"]
        for text in (json.dumps(req), render(req)):
            hits += [(i, s) for s in REMOVED_STEERING if s in text]
    for w in _warms:
        hits += [("warm", s) for s in REMOVED_STEERING
                 if s in (w.get("prompt") or "")]
    helpers = (list(_helper.archive) + list(_helper)
               + list(_helper_direct.archive) + list(_helper_direct))
    for h in helpers:
        hits += [("helper", s) for s in REMOVED_STEERING
                 if s in json.dumps(h)]
    check(len(_gens) > 50 and not hits,
          f"[removed] none of the removed steering text is in any of the "
          f"{len(_gens)} generations, {len(_warms)} warms or "
          f"{len(helpers)} second-brain requests "
          f"this suite sent", json.dumps(hits[:10]))


def test_step_cap_and_nudge():
    """Every agent step thinks at most AGENT_STEP_THINKING (the by-result
    caps of #53 were removed 2026-09-27, docs/CONSTANTS-AUDIT.md), and an
    agent step's nudge names the action; the nudge switches off by header.
    An identical re-read goes up as sent (the unchanged-read line was
    removed 2026-09-27)."""
    slots.reset(n=4)
    compaction.reset()
    c = Unforced("xhigh", "unchanged", tools=[WRITE, READ_TOOL, TERMINAL])
    feats = {"investigate": False, "skills": False}
    body = '{"content": "1|const GAME = {};", "total_lines": 1}'
    c.turn([reply("", reasoning="Read.", calls=[call(
        "read_file", {"path": "js/game.js"}, "u1")])],
        user="Fix the score in js/game.js.", features=feats)
    c.tool_result("u1", body)
    t1 = c.turn([reply("", reasoning="Again.", calls=[call(
        "read_file", {"path": "js/game.js"}, "u2")])], features=feats)
    req1 = t1["gens"][0]["request"]
    # The by-result caps were removed (docs/CONSTANTS-AUDIT.md,
    # 2026-09-27): a read step keeps the one agent-step cap; the nudge is
    # the operator's approved agent-step wording (2026-09-26).
    check(req1.get("reasoning_budget_tokens") == tiers.AGENT_STEP_THINKING
          and req1.get("reasoning_budget_nudge")
          == tiers.AGENT_STEP_NUDGE_MESSAGE
          and "previous" not in t1["d"]["x_yamadori"]["progress"]["step"],
          f"[step] the step answering a read thinks at most "
          f"{tiers.AGENT_STEP_THINKING} (one cap), with the agent-step nudge",
          json.dumps({k: req1.get(k) for k in (
              "reasoning_budget_tokens", "reasoning_budget_nudge")}))
    c.tool_result("u2", body)
    t2 = c.turn([reply("", reasoning="Run it.", calls=[call(
        "terminal", {"command": "node --check js/game.js"}, "u3")])],
        features=feats)
    last2 = t2["gens"][0]["request"]["messages"][-1]["content"]
    check(last2 == body,
          "[step] the identical re-read goes up as the client sent it",
          last2[-300:])
    _extends(t1, t2, "[step] the request after the re-read")
    c.tool_result("u3", '{"output": "SyntaxError: Unexpected token", '
                        '"exit_code": 1}')
    t3 = c.turn([reply("", reasoning="Fix.", calls=[call(
        "write_file", {"path": "js/game.js", "content": "const GAME = 1;\n"},
        "u4")])], features=feats)
    req3 = t3["gens"][0]["request"]
    restored = [m for m in req3["messages"] if m.get("role") == "tool"
                and m.get("content") == last2]
    check(req3.get("reasoning_budget_tokens")
          == tiers.AGENT_STEP_THINKING and len(restored) == 2,
          f"[step] after an error the cap is the same "
          f"{tiers.AGENT_STEP_THINKING}; the re-read's result is "
          f"unchanged", json.dumps(req3.get("reasoning_budget_tokens")))
    _extends(t2, t3, "[step] the request after it")
    # The nudge switched off by header: the general nudge, the same cap.
    slots.reset(n=4)
    d = Unforced("xhigh", "step off", tools=[WRITE, READ_TOOL])
    offf = dict(feats, step_nudge=False)
    d.turn([reply("", reasoning="Read.", calls=[call(
        "read_file", {"path": "js/game.js"}, "v1")])],
        user="Fix the score in js/game.js.", features=offf)
    d.tool_result("v1", body)
    e1 = d.turn([reply("", reasoning="Again.", calls=[call(
        "read_file", {"path": "js/game.js"}, "v2")])], features=offf)
    d.tool_result("v2", body)
    e2 = d.turn([reply("Done.", reasoning="ok")], features=offf)
    r1 = e1["gens"][0]["request"]
    check(r1.get("reasoning_budget_tokens") == tiers.AGENT_STEP_THINKING
          and r1.get("reasoning_budget_nudge") == tiers.NUDGE_MESSAGE
          and e2["gens"][0]["request"]["messages"][-1]["content"] == body,
          "[step] step_nudge false: the same cap, the general nudge",
          json.dumps({k: r1.get(k) for k in ("reasoning_budget_tokens",)}))


def test_the_work_log_returns_after_a_compaction_that_keeps_the_key():
    """#54 (and #38 under #41): a conversation with our session id keeps
    its key across a compaction, so the new-key link never fired and the
    work log was never re-injected. After _compaction_done the next request
    carries it on its user turn, recorded in the ledger and replayed. (The
    files-read listing #54 added to it was removed 2026-09-27.)"""
    slots.reset(n=4)
    compaction.reset()
    c = Unforced("xhigh", "worklog", tools=[WRITE, READ_TOOL])
    feats = {"investigate": False, "skills": False}
    c.turn([reply("", reasoning="Read.", calls=[call(
        "read_file", {"path": "js/game.js"}, "g1")])],
        user="Fix the score in js/game.js.", features=feats)
    c.tool_result("g1", '{"content": "1|const GAME = {};"}')
    t1 = c.turn([reply("Read it.", reasoning="ok")], features=feats)
    account = f"{ACCOUNT}-worklog"
    lineage = t1["d"]["x_yamadori"]["session"] and \
        proxy._LAST_SESSION[account][2]
    proxy._compaction_done(account, lineage, "## Summary\nFixing the score.")
    t2 = c.turn([reply("", reasoning="Go on.", calls=[call(
        "read_file", {"path": "js/game.js"}, "g2")])],
        user="Continue with the score.", features=feats)
    user2 = [m for m in t2["gens"][0]["request"]["messages"]
             if m.get("role") == "user"][-1]["content"]
    inj = t2["d"]["x_yamadori"]["ledger"]["inject"] or {}
    check("Work log of this conversation before it was summarised" in user2
          and "Files read in this conversation" not in user2
          and (inj.get("work_log") or {}).get("same_key") is True,
          "[work log] after a compaction that kept the session key, the "
          "next user turn carries the work log (and no files-read listing)",
          user2[-500:])
    c.tool_result("g2", '{"content": "1|const GAME = {};"}')
    t3 = c.turn([reply("Done.", reasoning="ok")], features=feats)
    user3 = [m for m in t3["gens"][0]["request"]["messages"]
             if m.get("role") == "user"][-1]["content"]
    check(user3 == user2 and not (t3["d"]["x_yamadori"]["ledger"]["inject"]
                                  or {}).get("work_log"),
          "[work log] replayed from the ledger byte for byte, not decided "
          "again")
    _extends(t2, t3, "[work log] the request after it")


def test_the_fixup_repairs_every_file():
    """The fix-up SCOPE (#56: project files only, a scratch-name table) was
    removed 2026-09-27 (docs/CONSTANTS-AUDIT.md: the table named one run's
    files). A broken write to /tmp is repaired like any other."""
    slots.reset(n=4)
    compaction.reset()
    _helper.clear()
    d = Unforced("xhigh", "fixup every file", tools=[WRITE])
    t3 = d.turn([reply("", reasoning="Harness.", calls=[call(
        "write_file", {"path": "/tmp/harness.py", "content": BROKEN},
        "h3")])], user="Write hi.py with a function f that returns 1.",
        features={"investigate": False, "skills": False})
    content3 = t3["d"]["choices"][0]["message"].get("content") or ""
    tc = t3["d"]["x_yamadori"].get("tool_code") or {}
    check("Repaired /tmp/harness.py" in content3
          and "scope_skipped" not in tc,
          "[fixup] a broken /tmp write is repaired (no scope rule)", content3)


# ------------------------------------------------ CONTINUE A STATED STEP
_judged: list[dict] = []


def _stub_judge(text, reasoning="", **kw):
    """A stub decider (decide_turn.judge_stop's contract): picks by the
    text, records every call."""
    _judged.append({"text": text, "reasoning": reasoning})
    t = text.lower()
    pick = ("asking" if t.rstrip().endswith("?") else
            "finished" if t.startswith(("done", "all done", "fixed"))
            else "next_step" if "let me" in t or "i'll" in t else "other")
    dist = {k: (0.7 if k == pick else 0.1)
            for k in ("finished", "next_step", "asking", "other")}
    return {"judged": True, "pick": pick, "raw_pick": pick, "tie": False,
            "distribution": dist, "ms": 12.5, "state": {"kind": "stop"}}


def _stated_client(tag: str, effort: str = "xhigh") -> "Unforced":
    slots.reset(n=4)
    compaction.reset()
    c = Unforced(effort, f"stated {tag}", tools=[WRITE, READ_TOOL])
    _agent_step_opening(c, "Fix the camera in src/scene.js.")
    return c


STATED = "Now let me read the camera setup to understand the task."
FEATS_STATED = {"investigate": False, "skills": False}


def test_a_stated_step_is_continued_once_and_yields_the_call():
    """CONTINUE A STATED STEP (operator-approved, 2026-09-27): an agent step
    that stops with no call and only says what it is about to do is judged
    by the decider and continued ONCE from its own text plus
    proxy.CONTINUE_LINE, the client's tools still offered; the client gets
    ONE turn (the text, the line, the call), the slot holds exactly that
    turn, and the next request extends it. A genuine final answer and a
    question to the user are untouched."""
    import decide_turn
    real = decide_turn.judge_stop
    decide_turn.judge_stop = _stub_judge
    _judged.clear()
    try:
        c = _stated_client("session")
        t1 = c.turn([reply(STATED, reasoning="I need the camera code."),
                     reply(".", calls=[call("read_file",
                                            {"path": "src/camera.js"},
                                            "rc1")])],
                    features=FEATS_STATED)
        x = t1["d"]["x_yamadori"]
        cont = x.get("continued") or {}
        g = t1["gens"]
        pre = g[1]["request"]["messages"][-1] if len(g) > 1 else {}
        line = proxy.CONTINUE_LINE
        check(len(g) == 2 and pre.get("role") == "assistant"
              and pre.get("content") == STATED + "\n\n" + line
              and pre.get("reasoning_content") == "I need the camera code."
              and g[1]["request"].get("tools") == g[0]["request"].get(
                  "tools"),
              "[stated] the stop is continued once: its own text + a blank "
              "line + CONTINUE_LINE prefilled, its reasoning kept, the "
              "client's tools still offered",
              json.dumps(pre)[:300])
        check(line[-1].isalpha(),
              "[stated] CONTINUE_LINE ends on a letter (AGENTS.md's prefill "
              "rule)", repr(line))
        m = t1["d"]["choices"][0]["message"]
        check(m.get("content") == STATED + "\n\n" + line + "."
              and [tc["function"]["name"] for tc in m.get("tool_calls")
                   or []] == ["read_file"]
              and t1["d"]["choices"][0]["finish_reason"] == "tool_calls",
              "[stated] the client receives ONE turn: the text, the line, "
              "the call", json.dumps(m)[:300])
        check(cont.get("judged") is True and cont.get("pick") == "next_step"
              and cont.get("raw_pick") == "next_step"
              and cont.get("continued") is True and cont.get("line") == line
              and cont.get("call_followed") is True
              and isinstance(cont.get("distribution"), dict)
              and cont.get("ms") == 12.5
              and len(_judged) == 1 and _judged[0]["text"] == STATED,
              "[stated] x_yamadori.continued records the judgment, the "
              "continuation and that a call followed",
              json.dumps(cont)[:300])
        check(not t1["warm"] and not (x.get("warm") or {}).get("sent"),
              "[stated] no warm: the slot holds the turn as delivered",
              json.dumps(x.get("warm")))
        # The next request: the tool result, then a genuine final answer.
        c.tool_result("rc1", '{"content": "1|const cam = null;"}')
        t2 = c.turn([reply("Done. The camera is created in main.js now.",
                           reasoning="ok")], features=FEATS_STATED)
        _extends(t1, t2, "[stated] the request after a continued step")
        c2 = t2["d"]["x_yamadori"].get("continued") or {}
        check(len(t2["gens"]) == 1 and c2.get("judged") is True
              and c2.get("pick") == "finished"
              and c2.get("continued") is False
              and t2["d"]["choices"][0]["message"]["content"]
              == "Done. The camera is created in main.js now.",
              "[stated] a genuine final answer is untouched (judged "
              "finished, one generation)", json.dumps(c2)[:300])
        c.msgs.append({"role": "user", "content": "Also add a zoom."})
        t3 = c.turn([reply("Should the zoom be on the wheel or on +/-?")],
                    features=FEATS_STATED)
        c3 = t3["d"]["x_yamadori"].get("continued") or {}
        check(len(t3["gens"]) == 1 and c3.get("pick") == "asking"
              and c3.get("continued") is False
              and t3["d"]["choices"][0]["message"]["content"].endswith("?"),
              "[stated] asking the user is untouched",
              json.dumps(c3)[:300])
        _extends(t2, t3, "[stated] and the request after the answer")

        # It stops again: delivered as is, never a second continuation.
        _judged.clear()
        d = _stated_client("again")
        s1 = d.turn([reply(STATED, reasoning="r"),
                     reply(" I'll check it.", reasoning="")],
                    features=FEATS_STATED)
        cs = s1["d"]["x_yamadori"].get("continued") or {}
        check(len(s1["gens"]) == 2 and len(_judged) == 1
              and cs.get("continued") is True
              and cs.get("call_followed") is False
              and s1["d"]["choices"][0]["message"]["content"]
              == STATED + "\n\n" + line + " I'll check it.",
              "[stated] a continuation that stops again is delivered as is "
              "(once per request, no loop)", json.dumps(cs)[:300])

        # The switch off: the old behaviour, and nothing is asked.
        _judged.clear()
        e = _stated_client("off")
        o1 = e.turn([reply(STATED, reasoning="r")],
                    features=dict(FEATS_STATED, continue_stated_step=False))
        co = o1["d"]["x_yamadori"].get("continued") or {}
        check(len(o1["gens"]) == 1 and not _judged
              and co.get("judged") is False and co.get("continued") is False
              and "switch continue_stated_step off (header)"
              == co.get("why")
              and o1["d"]["choices"][0]["message"]["content"] == STATED,
              "[stated] switch continue_stated_step off: one generation, "
              "nothing judged, delivered as generated",
              json.dumps(co)[:300])
        # Not an agent step (a first user turn, no calls yet): no trigger.
        _judged.clear()
        slots.reset(n=4)
        f = Unforced("xhigh", "stated chat", tools=[WRITE, READ_TOOL])
        p1 = f.turn([reply("Let me explain: the camera looks down -Z.")],
                    user="How does a perspective camera look?",
                    features=FEATS_STATED)
        check(p1["d"]["x_yamadori"].get("continued") is None and not _judged,
              "[stated] not an agent step: nothing judged, "
              "x_yamadori.continued is None",
              json.dumps(p1["d"]["x_yamadori"].get("continued")))
        # A tier that runs the model as it ships: not judged.
        _judged.clear()
        lo = _stated_client("low", effort="low")
        l1 = lo.turn([reply(STATED, reasoning="r")], features=FEATS_STATED)
        cl = l1["d"]["x_yamadori"].get("continued") or {}
        check(len(l1["gens"]) == 1 and not _judged
              and "tier low" in (cl.get("why") or ""),
              "[stated] at tier low nothing is judged",
              json.dumps(cl)[:300])
    finally:
        decide_turn.judge_stop = real


def test_a_stated_step_streams_like_it_blocks():
    """The streamed turn: the stated step's text goes out as it is
    generated, the continuation's re-sent text and reasoning are not sent
    again, and what a streaming client stores equals the blocking answer;
    the next request extends the slot."""
    import decide_turn
    real = decide_turn.judge_stop
    decide_turn.judge_stop = _stub_judge
    try:
        script = [reply(STATED, reasoning="I need the camera code."),
                  reply(".", calls=[call("read_file",
                                         {"path": "src/camera.js"}, "rs1")])]
        c = _stated_client("blocking")
        tb = c.turn(list(script), features=FEATS_STATED)
        mb = tb["d"]["choices"][0]["message"]
        s = _stated_client("streamed")
        _script[:] = list(script)
        n0 = len(_gens)
        got = _stream(s.body(FEATS_STATED))
        ts = {"gens": _gens[n0:], "warm": []}
        check(len(ts["gens"]) == 2 and got["content"] == mb["content"]
              and [x["function"]["name"] for x in got["calls"]]
              == [x["function"]["name"] for x in mb.get("tool_calls") or []]
              and [x["function"]["arguments"] for x in got["calls"]]
              == [x["function"]["arguments"] for x in mb.get("tool_calls")
                  or []],
              "[stated] streamed == blocking: the same content and call",
              f"stream={got['content']!r} block={mb['content']!r}")
        check(got["reasoning"] == "I need the camera code."
              and got["content"].count(STATED) == 1,
              "[stated] the re-sent text and reasoning are not streamed "
              "twice", repr(got["reasoning"]) + " " + repr(got["content"]))
        s.msgs.append({"role": "assistant", "content": got["content"],
                       "tool_calls": got["calls"]})
        s.msgs.append({"role": "tool",
                       "tool_call_id": got["calls"][0]["id"],
                       "content": '{"content": "1|const cam = null;"}'})
        _script[:] = [reply("Done. The camera is fixed.")]
        n1 = len(_gens)
        _stream(s.body(FEATS_STATED))
        _extends(ts, {"gens": _gens[n1:], "warm": []},
                 "[stated] the streamed request after a continued step")
    finally:
        decide_turn.judge_stop = real


# ------------------------------------------------ THE SERVER-TOOL TRIGGERS
def _no_recall_line(gens: list[dict]) -> bool:
    """No request of these generations carries a retired SERVER-TOOL RECALL
    line."""
    return not any("Remember (server tool" in json.dumps(g["request"])
                   for g in gens)


def test_a_package_probe_runs_deep_thinking_for_the_model():
    """The SERVER-TOOL TRIGGERS (operator, 2026-09-27, after pagoda-h5: "we
    should decide when to fire it, and be more heavy handed"): a step that
    reads inside node_modules/<pkg>/ makes the proxy run deep thinking on
    that package ITSELF, before main's generation: an inserted
    yama_think_deeply call and its hand-off (a hidden hop), then main's turn
    opens with one fixed line in its own voice (AUTO_DIRECTIVE_THINK). Once
    per package per conversation; the next request extends the slot; no
    recall line goes out; streamed == blocking."""
    import deep
    slots.reset(n=4)
    compaction.reset()
    _helper.clear()
    c = Unforced("xhigh", "auto probe", tools=[WRITE, READ_TOOL, TERMINAL])
    _agent_step_opening(c, "Wire koota into the scene.")
    t0 = c.turn([reply("", reasoning="Read its types.", calls=[call(
        "read_file", {"path": "node_modules/koota/dist/index.d.ts"},
        "pr1")])])
    check(not (t0["d"]["x_yamadori"].get("deep") or {}).get("fire"),
          "[auto] the request that MAKES the probe call runs nothing")
    c.tool_result("pr1", "1|export declare function createWorld(): World;")
    n_help = len(_helper)
    t1 = c.turn([reply(".", reasoning="", calls=[call(
        "write_file", {"path": "src/world.js",
                       "content": "import { createWorld } from 'koota'\n"},
        "pw1")])])
    x = t1["d"]["x_yamadori"]
    dx = x.get("deep") or {}
    auto = dx.get("auto") or {}
    msgs0 = t1["gens"][0]["request"]["messages"]
    call_msg, tool_msg, pre = msgs0[-3], msgs0[-2], msgs0[-1]
    tc = (call_msg.get("tool_calls") or [{}])[0]
    line = proxy.AUTO_DIRECTIVE_THINK.format(package="koota")
    check(dx.get("kind") == "auto" and auto.get("trigger") == "probe"
          and auto.get("key") == "think:koota"
          and auto.get("job") == "investigate"
          and auto.get("delivered_as") == "tool_result"
          and auto.get("directive") == line
          and isinstance(auto.get("seconds"), float)
          and len(_helper) > n_help
          and "koota" in json.dumps(_helper[n_help:]),
          "[auto] the probe of koota runs deep thinking on koota (x_yamadori"
          ".deep.auto: trigger, key, job, seconds, delivered_as, directive)",
          json.dumps(auto)[:400])
    check(tc.get("function", {}).get("name") == deep.TOOL_NAME
          and json.loads(tc["function"]["arguments"])
          == {"question": "How does koota work as this task uses it?"}
          and tool_msg.get("role") == "tool"
          and tool_msg.get("tool_call_id") == tc.get("id")
          and "LatheGeometry sweeps" in tool_msg.get("content", "")
          and pre.get("role") == "assistant"
          and pre.get("reasoning_content") == line and not pre.get("content"),
          "[auto] delivered as an inserted yama_think_deeply call and its "
          "result, then main's turn prefilled with the directive",
          json.dumps(msgs0[-3:])[:600])
    lines = (proxy.AUTO_DIRECTIVE_THINK, proxy.AUTO_DIRECTIVE_PLAN)
    check(all(t[-1].isalpha() and not any(
              w in t.lower() for w in ("source", "server", "stop ", "never",
                                       "don't", " not ", "{"))
              for t in lines)
          and "koota" not in line,
          "[auto] every directive ends on a letter, says to use what was "
          "just given and act, and names no source, server, package or "
          "prohibition (operator, 2026-09-28)", json.dumps(lines))
    m = t1["d"]["choices"][0]["message"]
    check(line not in (m.get("content") or "")
          and (m.get("reasoning_content") or "").startswith(line)
          and [k["function"]["name"] for k in m.get("tool_calls") or []]
          == ["write_file"] and "yama_think_deeply" not in json.dumps(
              c.msgs[-1]),
          "[auto] the client gets the directive and its own call, never "
          "the hidden hop", json.dumps(m)[:300])
    check(_no_recall_line(t0["gens"] + t1["gens"]),
          "[auto] no 'Remember (server tool ...)' line went out")
    c.tool_result("pw1", "wrote src/world.js")
    t2 = c.turn([reply("", reasoning="Again.", calls=[call(
        "read_file", {"path": "node_modules/koota/dist/react.d.ts"},
        "pr2")])])
    _extends(t1, t2, "[auto] the request after the inserted hop + "
                     "directive")
    msgs2 = t2["gens"][0]["request"]["messages"]
    check(any(m2.get("role") == "tool" and m2.get("tool_call_id")
              == tc.get("id") for m2 in msgs2),
          "[auto] the ledger replays the hidden hop")
    c.tool_result("pr2", "1|export function useQuery(): Entity[];")
    t3 = c.turn([reply("", reasoning="Write.", calls=[call(
        "write_file", {"path": "src/app.js", "content": "useQuery()\n"},
        "pw2")])])
    d3 = t3["d"]["x_yamadori"].get("deep") or {}
    check(not d3.get("fire") and "think:koota" in (
        (d3.get("signals") or {}).get("auto") or {}).get(
            "fired_before", []) and len(t3["gens"]) == 1,
          "[auto] a second koota probe: nothing (once per package)",
          json.dumps((d3.get("signals") or {}).get("auto"))[:300])
    _extends(t2, t3, "[auto] and the request after that")
    # Forced off by header: detected, never fired.
    slots.reset(n=4)
    e = Unforced("xhigh", "auto probe off", tools=[WRITE, READ_TOOL])
    _agent_step_opening(e, "Wire koota into the scene.")
    e.msgs += [{"role": "assistant", "content": "", "tool_calls": [call(
        "read_file", {"path": "node_modules/koota/dist/index.d.ts"}, "q1")]},
        {"role": "tool", "tool_call_id": "q1", "content": "1|export {}"}]
    o1 = e.turn([reply("Done.")], features={"investigate": False})
    check(not (o1["d"]["x_yamadori"].get("deep") or {}).get("fire")
          and len(o1["gens"]) == 1,
          "[auto] deep thinking forced off: no trigger fires")


class _Keyed(Unforced):
    """A client that names its conversation by prompt_cache_key (Hermes'
    Responses transport), which it may change mid-conversation."""

    def __init__(self, effort: str, tag: str, pck: str,
                 tools: list | None = None):
        super().__init__(effort, tag, tools=tools)
        self.pck = pck

    def body(self, features: dict | None = None) -> dict:
        b = super().body(features)
        b["prompt_cache_key"] = self.pck
        return b


def _probe(c: "Client", pkg: str, cid: str) -> dict:
    """The model reads inside node_modules/<pkg>/ (one turn), the harness
    returns the file; the NEXT request is the one a probe trigger fires
    on."""
    c.turn([reply("", reasoning="Read its types.", calls=[call(
        "read_file", {"path": f"node_modules/{pkg}/dist/index.d.ts"}, cid)])])
    c.tool_result(cid, f"1|export declare function {pkg}Api(): void;")
    return c.turn([reply(".", reasoning="", calls=[call(
        "write_file", {"path": f"src/{pkg}.js", "content": "x()\n"},
        cid + "w")])])


def _auto_of(t: dict) -> tuple[dict, dict]:
    dx = t["d"]["x_yamadori"].get("deep") or {}
    return dx, (dx.get("signals") or {}).get("auto") or {}


def _compacted(c: "Client", summary: str, keep: int) -> None:
    """A harness's own compaction: the system prompt, its summary as a user
    turn (NO line of ours -- our summariser never answered it), and the
    last `keep` messages of the history."""
    c.msgs = [c.msgs[0], {"role": "user", "content": summary}] \
        + c.msgs[-keep:]


def test_a_trigger_fires_once_across_a_compaction():
    """ONCE PER KEY SURVIVES A COMPACTION (operator, 2026-09-28, after
    pagoda-h6: the probe triggers for koota, math, three and
    @react-three/fiber fired, the client compacted -- its summariser call
    hung up, Hermes summarised on its own and its prompt_cache_key changed --
    and all four fired AGAIN, 14-18 minutes each). Replayed: a probe of koota
    fires; (a) a compaction that keeps the key: a second koota probe fires
    nothing; (b) a compaction with a NEW key and no summary line of ours,
    the kept tail carrying our id in its tool-call ids (the h6 shape, which
    session_identity names a fork): nothing again, the fork's deep state
    inherited from the conversation it came from; a package not probed
    before still fires, once."""
    slots.reset(n=4)
    compaction.reset()
    _helper.clear()
    c = _Keyed("xhigh", "refire", "hermes-key-1",
               tools=[WRITE, READ_TOOL, TERMINAL])
    _agent_step_opening(c, "Wire koota into the scene.")
    t1 = _probe(c, "koota", "k1")
    d1, _ = _auto_of(t1)
    check(d1.get("kind") == "auto"
          and (d1.get("auto") or {}).get("key") == "think:koota",
          "[refire] the first koota probe runs deep thinking (think:koota)",
          json.dumps(d1.get("auto"))[:300])
    c.tool_result("k1w", "wrote src/koota.js")
    # (a) The client compacts and keeps its key.
    _compacted(c, "Summary: koota is wired into src/koota.js; the scene "
                  "renders.", keep=2)
    n_help = len(_helper)
    t2 = _probe(c, "koota", "k2")
    d2, s2 = _auto_of(t2)
    check(not d2.get("fire") and "think:koota" in s2.get("fired_before", [])
          and len(_helper) == n_help and len(t2["gens"]) == 1,
          "[refire] after a compaction that keeps the key: a second koota "
          "probe fires nothing (think:koota fired before)",
          json.dumps({"deep": d2.get("kind"), "auto": s2})[:300])
    c.tool_result("k2w", "wrote src/koota.js")
    # (b) The client compacts again and its key CHANGES; no summary line of
    # ours; the kept tail carries our id in its tool-call ids.
    c.pck = "hermes-key-2"
    _compacted(c, "[CONTEXT COMPACTION] Earlier turns were compacted. "
                  "koota is wired in.", keep=2)
    n_help = len(_helper)
    t3 = _probe(c, "koota", "k3")
    d3, s3 = _auto_of(t3)
    ses3 = t3["d"]["x_yamadori"].get("session") or {}
    check(ses3.get("forked_from") and not ses3.get("aliased_to"),
          "[refire] the new key is named a fork of the conversation (the "
          "session rule is unchanged)", json.dumps(ses3)[:300])
    check(not d3.get("fire") and "think:koota" in s3.get("fired_before", [])
          and d3.get("inherited_from") and len(_helper) == n_help
          and len(t3["gens"]) == 1,
          "[refire] after a compaction under a NEW key with no summary line "
          "(pagoda-h6): a koota probe fires nothing -- the once-per-key "
          "state came with the fork (x_yamadori.deep.inherited_from)",
          json.dumps({"kind": d3.get("kind"), "auto": s3,
                      "inherited_from": d3.get("inherited_from")})[:400])
    c.tool_result("k3w", "wrote src/koota.js")
    t4 = _probe(c, "math", "m1")
    d4, _ = _auto_of(t4)
    check(d4.get("kind") == "auto"
          and (d4.get("auto") or {}).get("key") == "think:math",
          "[refire] a package not probed before still fires (think:math)",
          json.dumps(d4.get("auto"))[:300])
    c.tool_result("m1w", "wrote src/math.js")
    t5 = _probe(c, "math", "m2")
    d5, s5 = _auto_of(t5)
    check(not d5.get("fire") and {"think:koota", "think:math"} <= set(
        s5.get("fired_before", [])),
          "[refire] and once: a second math probe fires nothing",
          json.dumps(s5)[:300])


def test_a_probe_turn_streams_like_it_blocks():
    """The auto trigger's turn, streamed: the directive goes out as content
    (the server re-sends the prefill), the hidden hop never does, and the
    client stores what the blocking path returns."""
    got = {}
    line = proxy.AUTO_DIRECTIVE_THINK.format(package="koota")
    for streamed in (False, True):
        slots.reset(n=4)
        compaction.reset()
        c = Unforced("xhigh", f"auto stream {streamed}",
                     tools=[WRITE, READ_TOOL])
        _agent_step_opening(c, "Wire koota into the scene.")
        c.msgs += [{"role": "assistant", "content": "", "tool_calls": [call(
            "read_file", {"path": "node_modules/koota/dist/index.d.ts"},
            "s1")]}, {"role": "tool", "tool_call_id": "s1",
                      "content": "1|export {}"}]
        script = [reply(".", calls=[call("write_file", {
            "path": "src/world.js", "content": "x\n"}, "s2")])]
        if streamed:
            _script[:] = list(script)
            s = _stream(c.body())
            got[streamed] = (s["content"], [k["function"]["name"]
                                            for k in s["calls"]],
                             "yama_think_deeply" in json.dumps(
                                 [s["content"], s["calls"]]),
                             line in s["reasoning"])
        else:
            t = c.turn(script)
            m = t["d"]["choices"][0]["message"]
            got[streamed] = (m["content"], [k["function"]["name"] for k in
                                            m.get("tool_calls") or []],
                             "yama_think_deeply" in json.dumps(m),
                             line in (m.get("reasoning_content") or ""))
    check(got[True] == got[False] and line not in got[True][0]
          and got[True][1:] == (["write_file"], False, True),
          "[auto] streamed == blocking: the directive and the call, never "
          "the hidden hop", json.dumps(got))


def test_a_plan_trigger_runs_yama_plan_for_the_model():
    """The plan triggers: after the kickoff plan (FILES index.html,
    js/game.js), a write into js/editor/ -- a directory the plan never
    named -- makes the proxy run the plan job itself for the next piece
    (an inserted yama_plan hop + AUTO_DIRECTIVE_PLAN); writing every
    planned file runs it once more (plan_done); a user turn that asks for
    something to be built after an answer runs it for that (implement, the
    same directive). Each once; each next request extends."""
    import deep
    slots.reset(n=4)
    compaction.reset()
    _helper.clear()
    real = shomen._post
    shomen._post = _plan_post_for(real)
    # The plan triggers alone: the verify directive's moments (the entry,
    # the finished plan) are test_the_verify_directive's.
    EXTRA_FEATURES["verify_directive"] = False
    try:
        c = Unforced("xhigh", "auto plan", tools=[WRITE, READ_TOOL])
        t1 = c.turn([reply("", calls=[call("write_file", {
            "path": "index.html", "content": "<canvas>"}, "a1")])],
            user="Build a space shooter with canvas.")
        c.tool_result("a1", "wrote index.html")
        t2 = c.turn([reply("", calls=[call("write_file", {
            "path": "js/editor/grid.js", "content": "grid()\n"}, "a2")])])
        check(not (t2["d"]["x_yamadori"].get("deep") or {}).get("fire"),
              "[auto plan] a planned file: nothing fires")
        _extends(t1, t2, "[auto plan] the request after the kickoff")
        c.tool_result("a2", "wrote js/editor/grid.js")
        n_help = len(_helper)
        t3 = c.turn([reply(".", calls=[call("write_file", {
            "path": "js/editor/save.js", "content": "save()\n"}, "a3")])])
        x3 = t3["d"]["x_yamadori"]
        a3 = (x3.get("deep") or {}).get("auto") or {}
        m3 = t3["gens"][0]["request"]["messages"]
        tc = (m3[-3].get("tool_calls") or [{}])[0]
        check(a3.get("trigger") == "next_piece"
              and a3.get("key") == "piece:js/editor"
              and a3.get("job") == "plan"
              and a3.get("directive") == proxy.AUTO_DIRECTIVE_PLAN
              and tc.get("function", {}).get("name") == deep.PLAN_TOOL_NAME
              and json.loads(tc["function"]["arguments"])
              == {"task": "The next piece of the task above: js/editor/."}
              and m3[-2].get("content", "").startswith(
                  proxy.PLAN_RESULT_HEAD)
              and m3[-1].get("reasoning_content")
              == proxy.AUTO_DIRECTIVE_PLAN and not m3[-1].get("content")
              and len(_helper) > n_help
              and "js/editor/ is a new part of the work" in json.dumps(
                  _helper[n_help:]),
              "[auto plan] a write in a new directory runs yama_plan for "
              "the next piece: inserted hop, the directive after it",
              json.dumps(a3)[:400])
        _extends(t2, t3, "[auto plan] the request that ran it extends the "
                         "slot")
        c.tool_result("a3", "wrote js/editor/save.js")
        t4 = c.turn([reply("", calls=[call("write_file", {
            "path": "js/game.js", "content": "loop()\n"}, "a4")])])
        check(not (t4["d"]["x_yamadori"].get("deep") or {}).get("fire"),
              "[auto plan] the same new directory again: nothing")
        _extends(t3, t4, "[auto plan] the request after the plan hop")
        c.tool_result("a4", "wrote js/game.js")
        t5 = c.turn([reply(".", calls=[call("read_file", {
            "path": "index.html"}, "a5")])])
        a5 = ((t5["d"]["x_yamadori"].get("deep") or {}).get("auto") or {})
        check(a5.get("trigger") == "plan_done"
              and a5.get("key", "").startswith("plan_done:")
              and a5.get("directive") == proxy.AUTO_DIRECTIVE_PLAN,
              "[auto plan] every planned file written: yama_plan runs once "
              "more (plan_done)", json.dumps(a5)[:300])
        _extends(t4, t5, "[auto plan] the plan_done request")
        c.tool_result("a5", "<canvas>")
        t6 = c.turn([reply("The game is built.")])
        check(not (t6["d"]["x_yamadori"].get("deep") or {}).get("fire"),
              "[auto plan] after it: nothing more")
        t7 = c.turn([reply(".", calls=[call("write_file", {
            "path": "js/audio.js", "content": "beep()\n"}, "a7")])],
            user="Great, now build the sound effects next.")
        a7 = ((t7["d"]["x_yamadori"].get("deep") or {}).get("auto") or {})
        m7 = t7["gens"][0]["request"]["messages"]
        check(a7.get("trigger") == "implement"
              and a7.get("directive") == proxy.AUTO_DIRECTIVE_PLAN
              and (a7.get("evidence") or {}).get("intent") == "rule"
              and m7[-1].get("reasoning_content")
              == proxy.AUTO_DIRECTIVE_PLAN
              and (t7["d"]["choices"][0]["message"].get("reasoning_content")
                   or "").startswith(proxy.AUTO_DIRECTIVE_PLAN)
              and proxy.AUTO_DIRECTIVE_PLAN not in
              t7["d"]["choices"][0]["message"]["content"],
              "[auto plan] a build request after an answer runs yama_plan "
              "for it (intent by the rule: the decider is off offline)",
              json.dumps(a7)[:300])
        _extends(t6, t7, "[auto plan] the implement request")
        t8 = c.turn([reply("It uses the Web Audio API.")],
                    user="Why is the sound quiet?")
        check(not (t8["d"]["x_yamadori"].get("deep") or {}).get("fire"),
              "[auto plan] a question after an answer runs nothing")
        check(_no_recall_line(t1["gens"] + t2["gens"] + t3["gens"]
                              + t4["gens"] + t5["gens"] + t7["gens"]),
              "[auto plan] no 'Remember (server tool ...)' line went out")
    finally:
        shomen._post = real
        EXTRA_FEATURES.pop("verify_directive", None)


# ------------------------------------------------------ THE VERIFY DIRECTIVE
BROWSER_NAV = {"type": "function", "function": {
    "name": "browser_navigate", "description": "Navigate to a URL in the "
    "browser.", "parameters": {"type": "object", "properties": {
        "url": {"type": "string"}}}}}


def _stub_choose(messages, label, options, **kw):
    """A stub decider for verify_moment.choose_tool: a browser tool for a
    web page when there is one, else none."""
    names = [o["name"] for o in options]
    pick = next((n for n in names if n.startswith("browser")), None) \
        if label == "web page" else None
    return {"judged": True, "pick": pick, "none": pick is None,
            "tie": False, "rounds": [{"options": names}],
            "distribution": {n: (0.8 if n == pick else 0.05)
                             for n in names + ["none"]},
            "raw_pick": pick or "none", "ms": 7.0}


def test_the_verify_directive():
    """THE VERIFY DIRECTIVE (operator, 2026-09-27; mcp/verify_moment.py):
    the entry file's first write (index.html) opens main's next turn with a
    line naming the client tool the decider picked to run it; the model's
    turn after it called something else, so the next moment -- every
    planned file written, where the plan trigger also fires -- opens with
    the GENERIC line instead of the plan's directive (the plan result still
    inserted). Each moment once; x_yamadori.deep.auto.verify records the
    kind, the options, the pick, the line, its form and why; every next
    request extends the slot; streamed == blocking."""
    import verify_moment as V
    real_choose, real_post = V.choose_tool, shomen._post
    V.choose_tool = _stub_choose
    shomen._post = _plan_post_for(real_post)
    try:
        slots.reset(n=4)
        compaction.reset()
        c = Unforced("xhigh", "verify session",
                     tools=[WRITE, READ_TOOL, BROWSER_NAV])
        t1 = c.turn([reply("", calls=[call("write_file", {
            "path": "index.html", "content": "<canvas>"}, "v1")])],
            user="Build a space shooter with canvas.")
        c.tool_result("v1", "wrote index.html")
        t2 = c.turn([reply(".", calls=[call("write_file", {
            "path": "js/game.js", "content": "loop()\n"}, "v2")])])
        x2 = t2["d"]["x_yamadori"]
        vf = ((x2.get("deep") or {}).get("auto") or {}).get("verify") or {}
        named = V.NAMED.format(verb="open it", tool="browser_navigate")
        pre = t2["gens"][0]["request"]["messages"][-1]
        check(vf.get("moment") == "entry_written"
              and vf.get("kind") == "web_page"
              and vf.get("options") == ["write_file", "read_file",
                                        "browser_navigate"]
              and vf.get("pick") == "browser_navigate"
              and vf.get("form") == "named" and vf.get("line") == named
              and vf.get("delivered") is True
              and "gap" in (vf.get("skill") or {})
              and pre.get("role") == "assistant"
              and pre.get("reasoning_content") == named
              and not pre.get("content"),
              "[verify] index.html first written: main's turn opens with "
              "the NAMED line (the decider's pick among the client's "
              "tools); the record carries kind, options, pick, form, why "
              "and the craft gap", json.dumps(vf)[:500])
        m2 = t2["d"]["choices"][0]["message"]
        check((m2.get("reasoning_content") or "").startswith(named)
              and named not in (m2.get("content") or ""),
              "[verify] the line opens the turn's REASONING (the think "
              "block left open), never its content", json.dumps(m2)[:300])
        _extends(t1, t2, "[verify] the request with the named line")
        c.tool_result("v2", "wrote js/game.js")
        t3 = c.turn([reply(".", calls=[call("read_file", {
            "path": "index.html"}, "v3")])])
        d3 = (t3["d"]["x_yamadori"].get("deep") or {})
        a3 = d3.get("auto") or {}
        v3 = a3.get("verify") or {}
        m3 = t3["gens"][0]["request"]["messages"]
        check(a3.get("trigger") == "plan_done"
              and a3.get("directive") == V.GENERIC
              and a3.get("directive_replaced") == proxy.AUTO_DIRECTIVE_PLAN
              and v3.get("moment") == "piece_done"
              and v3.get("form") == "generic"
              and "did not land" in (v3.get("why") or "")
              and m3[-2].get("content", "").startswith(
                  proxy.PLAN_RESULT_HEAD)
              and m3[-1].get("reasoning_content") == V.GENERIC
              and not m3[-1].get("content"),
              "[verify] every planned file written, after a named line "
              "that did not land: the GENERIC line, in place of the plan's "
              "directive (the plan result still inserted)",
              json.dumps(a3)[:600])
        _extends(t2, t3, "[verify] the request with the generic line")
        c.tool_result("v3", "<canvas>")
        t4 = c.turn([reply("", calls=[call("write_file", {
            "path": "index.html", "content": "<canvas id=g>"}, "v4")])])
        c.tool_result("v4", "wrote index.html")
        t5 = c.turn([reply("Done.")])
        check(not (((t5["d"]["x_yamadori"].get("deep") or {}).get("auto")
                    or {}).get("verify")),
              "[verify] a later rewrite of the entry: no moment (once)")
        _extends(t4, t5, "[verify] and after it")
        # Streamed == blocking on the named-line request.
        got = {}
        for streamed in (False, True):
            slots.reset(n=4)
            compaction.reset()
            d = Unforced("xhigh", f"verify stream {streamed}",
                         tools=[WRITE, BROWSER_NAV])
            _agent_step_opening(d, "Make a canvas page.")
            d.msgs += [{"role": "assistant", "content": "", "tool_calls": [
                call("write_file", {"path": "index.html",
                                    "content": "<canvas>"}, "sv1")]},
                {"role": "tool", "tool_call_id": "sv1", "content": "ok"}]
            script = [reply(".", calls=[call("browser_navigate", {
                "url": "http://localhost:3000"}, "sv2")])]
            if streamed:
                _script[:] = list(script)
                s = _stream(d.body())
                got[streamed] = (s["content"], [k["function"]["name"]
                                                for k in s["calls"]],
                                 s["reasoning"].startswith(named))
            else:
                t = d.turn(script)
                m = t["d"]["choices"][0]["message"]
                got[streamed] = (m["content"], [k["function"]["name"] for k
                                                in m.get("tool_calls") or []],
                                 (m.get("reasoning_content") or "")
                                 .startswith(named))
        check(got[True] == got[False] == (".", ["browser_navigate"], True),
              "[verify] streamed == blocking: the named line and the call",
              json.dumps(got))
        # The switch off: no line.
        slots.reset(n=4)
        e = Unforced("xhigh", "verify off", tools=[WRITE, BROWSER_NAV])
        _agent_step_opening(e, "Make a canvas page.")
        e.msgs += [{"role": "assistant", "content": "", "tool_calls": [
            call("write_file", {"path": "index.html", "content": "<p>"},
                 "o1")]}, {"role": "tool", "tool_call_id": "o1",
                           "content": "ok"}]
        o1 = e.turn([reply("Done.")], features={"verify_directive": False})
        check(not (((o1["d"]["x_yamadori"].get("deep") or {}).get("auto")
                    or {}).get("verify"))
              and o1["gens"][0]["request"]["messages"][-1]["role"] == "tool",
              "[verify] switch verify_directive off: no moment, no line")
    finally:
        V.choose_tool, shomen._post = real_choose, real_post


def test_the_research_seed_line_is_replayed():
    """#52 / remedy 7: the investigate job's user message frames the seed as
    random and unrelated; the word is the ledger's, so the same request
    renders the same bytes; off by header, the old line."""
    payload = {"_ledger": {"turn_key": "u:seedframe", "account": ACCOUNT,
                           "session": "seedframe"}}
    s1 = proxy.ledger_seed(payload, "investigate", "q")
    s2 = proxy.ledger_seed(payload, "investigate", "q")
    a = shomen.seed_phrase(s1["word"], "test", job="investigate")
    b = shomen.seed_phrase(s2["word"], "test", job="investigate")
    check(s1 == s2 and a == b
          and a == shomen.SEED_LINE_RESEARCH.format(word=s1["word"])
          and shomen.seed_phrase(s1["word"], "t", job="plan") == a,
          "[seed] investigate and plan: the research line, the ledger's "
          "word, byte-identical on a replay", a)
    check(shomen.seed_phrase("oak", "t", job="fixup")
          == shomen.SEED_LINE.format(word="oak")
          and shomen.seed_phrase("oak", "t", job="investigate", frame=False)
          == shomen.SEED_LINE.format(word="oak")
          and shomen.PHRASES["seed"] == "Today I was inspired by {words}.",
          "[seed] the fix-up keeps the old line; switched off, investigate "
          "does too; the ANSWER's phrase is unchanged")


def main() -> int:
    for fn in (test_keys, test_scope_persistence_and_pruning,
               test_call_arguments_render_in_one_order,
               test_prefix_stability_across_a_session,
               test_a_streamed_hop_after_a_prefill_is_restored,
               test_think_deeply_is_a_hidden_hop_the_next_request_extends,
               test_a_think_deeply_with_no_conclusion_says_so,
               test_struggle_runs_deep_thinking_before_main_once_per_episode,
               test_the_initial_prompt_plan_is_a_hidden_yama_plan_hop,
               test_the_model_calls_yama_plan_as_a_hidden_hop,
               test_the_deep_tool_hop_switch,
               test_old_tool_names_still_replay_and_run,
               test_a_prefilled_turn_warms_from_the_last_user_checkpoint,
               test_a_prefill_mid_conversation_rereads_to_the_thinned_checkpoint,
               test_a_prefilled_turn_is_the_slots_sequence_when_reasoning_is_restored,
               test_a_prefill_mid_conversation_extends_when_reasoning_is_restored,
               test_the_window_check_counts_restored_reasoning,
               test_the_served_template_renders_every_past_think_block,
               test_the_warm_reports_on_the_next_response,
               test_a_fixup_step_and_its_warm,
               test_a_client_that_acts_on_the_calls_waits_for_the_warm,
               test_a_first_turn_warm_does_not_guess_echo_from_a_mixed_account,
               test_the_fixup_never_touches_the_conversations_slot,
               test_the_fixup_releases_the_helper_slot_not_the_conversations,
               test_library_use_reaches_harness_traffic,
               test_library_use_reads_the_conversations_version,
               test_a_warm_releases_only_its_own_waiters,
               test_text_turn_keys_are_per_conversation,
               test_a_duplicate_request_never_blanks_the_library_help,
               test_a_retryable_decision_is_decided_again,
               test_a_definitions_lookup_that_fails_is_retryable,
               test_the_echo_path_renders_like_the_strip_path,
               test_template_markers_are_scrubbed_on_our_path_and_recorded_from_the_model,
               test_no_situation_line_and_old_rows_replay,
               test_step_cap_and_nudge,
               test_the_work_log_returns_after_a_compaction_that_keeps_the_key,
               test_the_fixup_repairs_every_file,
               test_the_research_seed_line_is_replayed,
               test_a_stated_step_is_continued_once_and_yields_the_call,
               test_a_stated_step_streams_like_it_blocks,
               test_a_package_probe_runs_deep_thinking_for_the_model,
               test_a_probe_turn_streams_like_it_blocks,
               test_a_trigger_fires_once_across_a_compaction,
               test_a_plan_trigger_runs_yama_plan_for_the_model,
               test_the_verify_directive,
               # last: it scans every request the tests above sent
               test_no_removed_steering_reaches_any_request):
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
    print("\n  bytes each turn added to the ledger (to set the caps from "
          "data):")
    for label, n in SIZES:
        print(f"    {label:<48} {n:>7}")
    _srv.shutdown()
    passed = sum(1 for ok, _, _ in _results if ok)
    total = len(_results)
    print(f"\n{'=' * 70}\n  {passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
