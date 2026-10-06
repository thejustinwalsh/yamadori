#!/usr/bin/env python
"""The session ledger and the prefix-stability gate. No GPU.

    python mcp/test_ledger.py      -> "N/M checks passed"

WHAT THIS IS GATING (docs/SELF-IMPROVEMENT-PLAN.md Phase 0.5, operator
2026-09-24): one model, one cache.

  1. PREFIX STABILITY. A multi-turn session through the real complete(),
     with a client that strips everything the proxy added -- reasoning, the
     skills block and the concept seed on a user turn, image-tool hops --
     EXTENDS, on every request, everything the slot holds: the previous
     request's prompt plus what it generated, reasoning included (PAST
     REASONING IS RESTORED, operator 2026-09-27, switch restore_reasoning),
     or, when the proxy warmed the slot (proxy._warm), the whole warmed
     prompt. The processed tail is the tool result + the new part. (Switch
     off, the 2026-09-24 pass-through: the request reuses up to the previous
     turn's think block and processes that turn too; the prefilled-turn
     tests keep that arm.) The rendering is the SERVED chat template
     (mcp/fixtures/bonsai_chat_template.jinja, captured from llama-server
     /props on 2026-09-24, build b10738-285542d9), run by jinja2 -- STEP 0
     measured that llama-server reuses a slot fully only when a request
     extends its sequence.
     The session covers: an injected skill and the concept seed on the
     first user turn, a client write delivered as written (warmed: its
     arguments are rendered sorted), a yama_generate_image hop the client
     never sees (a hidden internal hop), an agent step, and an in-place
     compaction that THINKS at the conversation's effort. It runs at tier
     xhigh (medium thinking: no effort line) and max (xhigh: the effort line
     is rendered), so the compaction's cache check covers both.
  2. PERSISTENCE: a proxy restart (the memory layer and the compaction
     store emptied) costs nothing -- the next request still renders as the
     slot holds it, from sqlite.
  3. SCOPE AND SIZE: another account sees nothing; pruning drops whole
     sessions least-recently-seen first against the account cap, then the
     global cap, and anything past the max age.
  4. THE PREFILL CHANNEL (proxy.directive_prefill, kept 2026-09-29 for the
     skills renderer): a prefilled turn's warm, its checkpoint rule, and --
     with reasoning restored -- a prefilled turn that is the slot's
     sequence.
  5. THE BYTES: what each turn added to the ledger is printed, so the caps
     (choices) can be set from data.

REMOVED 2026-09-29 with the features they gated (docs/REMOVED.md): deep
thinking (yama_think_deeply, yama_plan, the kickoff plan, the struggle and
server-tool triggers, the verify directive), continue_stated_step, the
fix-up repair and its notes, library definitions and LIBRARY USE.
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


_helper_direct: list[dict] = _Archived()  # requests through model.post


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
    if r.get("drop"):
        # the model server died: llama-swap answers 200 and closes the body -- no finish chunk, no usage, no [DONE]
        # (test_stream.py's fake does the same; the Responses suite reads this path end to end)
        return b"".join(out)
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
            # An internal request through its real door (mcp/model.py posts
            # a non-streamed request): recorded with the slot it was sent
            # to.
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
        if r.get("delay"):
            time.sleep(r["delay"])     # a slow generation (a model load, a cold prefill): the pump's heartbeats
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
BROKEN = "def f(:\n    return 1\n"
FIXED = "def f():\n    return 1\n"
_real_run_our_tool = proxy.run_our_tool


def _run_our_tool(name, args, state=None):
    if name == "yama_generate_image":
        return "![a lathe](https://media.example/abc.png)"
    return _real_run_our_tool(name, args, state)


proxy.run_our_tool = _run_our_tool

# A DIRECTIVE PREFILL (proxy.directive_prefill; kept 2026-09-29 as a skill
# delivery channel): a line as the opening of main's reasoning, the think
# block left open. Nothing in prepare() sets one today, so a test that needs
# a prefilled turn wraps prepare() for the requests made inside the block.
DIRECTIVE = "Next I'll build the canvas and then the game loop"


class _Prefill:
    """proxy.prepare wrapped: its payload carries `_prefill`, a directive
    prefill of `line`, for the requests made inside the block."""

    def __init__(self, line: str = DIRECTIVE):
        self.line = line

    def __enter__(self):
        self.saved = proxy.prepare
        saved, line = self.saved, self.line

        def prep(body):
            out = saved(body)
            out["_prefill"] = proxy.directive_prefill(line)
            return out
        proxy.prepare = prep
        return self

    def __exit__(self, *exc):
        proxy.prepare = self.saved


def _prefilled(t: dict) -> bool:
    """The turn's first generation opened with a reasoning-only prefill."""
    req = t["gens"][0]["request"] if t.get("gens") else {}
    last = (req.get("messages") or [{}])[-1]
    return (last.get("role") == "assistant"
            and not (last.get("content") or "").strip()
            and bool((last.get("reasoning_content") or "").strip()))


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
        feats = {"skills": True}
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
    import concept_seed
    slots.reset(n=4)
    compaction.reset()
    c = Client(effort)
    tag = f"[{effort}]"

    # 1. The conversation's first user turn: the skill and the concept seed
    #    ride on it, decided once (proxy.prepare).
    b0 = _bytes()
    t1 = c.turn([reply("It sweeps the profile around the Y axis.",
                       reasoning="Answer it.")],
                user="How does LatheGeometry build its points?")
    SIZES.append((f"{tag} turn 1 (skill + seed)", _bytes() - b0))
    # A new conversation's answer carries NO session line (#41, operator
    # 2026-09-25): its id rides in tool-call ids, and this text-only
    # answer is recorded with it (proxy THE ANSWER RECORD).
    ans = t1["d"]["choices"][0]["message"]["content"]
    sess = t1["d"]["x_yamadori"].get("session") or {}
    sid = sess.get("id")
    check(sid and session_id.PREFIX not in ans
          and ans == "It sweeps the profile around the Y axis.",
          f"{tag} a new conversation's first answer has an id, no session "
          f"line and nothing of ours", ans[:60])
    word = (sess.get("seed") or {}).get("word")
    first = t1["gens"][0]["request"]["messages"][1]["content"]
    check(word and first.endswith(
              HINT + concept_seed.USER_TURN_LINE.format(word=word)),
          f"{tag} the skill and the concept seed ride on the first user turn",
          first[-240:])

    # 2. A client write: delivered as the model wrote it (the code check,
    #    the fix-up and their notes were removed 2026-09-29). The model
    #    wrote its arguments unsorted, so the slot is warmed with the turn
    #    as the next request renders it.
    b0 = _bytes()
    t2 = c.turn([reply("Writing it.", calls=[call(
        "write_file", {"path": "hi.py", "content": BROKEN}, "w1")])],
        user="Write hi.py with a function f that returns 1.")
    SIZES.append((f"{tag} turn 2 (a write + warm)", _bytes() - b0))
    _extends(t1, t2, f"{tag} turn 2 after turn 1 (stripped reasoning, "
                     f"stripped skill and seed)")
    m2 = t2["d"]["choices"][0]["message"]
    args2 = json.loads(m2["tool_calls"][0]["function"]["arguments"])
    check(m2.get("content") == "Writing it." and args2.get("content") == BROKEN,
          f"{tag} the write goes out exactly as written: no note, no repair",
          json.dumps(m2)[:200])
    users2 = [m for m in t2["gens"][0]["request"]["messages"]
              if m.get("role") == "user"]
    check(users2[0]["content"] == first
          and not users2[-1]["content"].endswith(
              concept_seed.USER_TURN_LINE.format(word=word))
          and ((t2["d"]["x_yamadori"].get("session") or {}).get("seed")
               or {}).get("word") == word,
          f"{tag} the seed is replayed on the first user turn only, and the "
          f"session keeps reporting it")
    c.tool_result("w1", "wrote hi.py")

    # 3. The agent step after the tool: a plain answer.
    b0 = _bytes()
    t3 = c.turn([reply("Done: hi.py is written.", reasoning="It worked.")])
    SIZES.append((f"{tag} turn 3 (agent step)", _bytes() - b0))
    _extends(t2, t3, f"{tag} turn 3 after the warmed write")

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
          f"({req.get('reasoning_effort')}), what its window leaves of it",
          json.dumps({k: req.get(k) for k in (
              "enable_thinking", "reasoning_effort",
              "reasoning_budget_tokens")}))
    check(((x.get("session") or {}).get("seed") or {}).get("word") == word,
          f"{tag} the compaction reports the conversation's seed")

    # The same OPENING again (a retry, or a new conversation that opens
    # alike): a NEW conversation, always (#41) -- its own id, its own seed,
    # nothing of turn 1's replayed.
    retry = Client(effort)
    retry.msgs = c.msgs[:2]
    t1b = retry.turn([reply(" it sweeps.")])
    sess_b = t1b["d"]["x_yamadori"].get("session") or {}
    word_b = (sess_b.get("seed") or {}).get("word")
    sid_b = sess_b.get("id")
    check(sid_b and sid_b != sid and word_b and word_b != word
          and sess_b["source"] == "minted",
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
    """#10 (docs/SELF-IMPROVEMENT-LOG.md). Live gate 2026-09-24: a turn
    whose FIRST generation (a prefill) streamed a long preface as content,
    then called one of OUR tools, and the answer came on a later hop. The
    client stored every content byte it was streamed; the ledger had keyed
    the turn by the last hop's content alone, so the prefilled reasoning and
    the hidden hop were never restored and the next request diverged at the
    start of the turn (reused 2513 of 3955: everything before it). The next
    request must extend what the slot holds, and the client's copy must be
    what keys it. (The prefill was deep thinking's hand-off; since
    2026-09-29 it is a directive, proxy.directive_prefill.)"""
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
            "_features": json.dumps({"skills": True})}
    with _Prefill():
        got = _stream(body)
    t1 = {"gens": _gens[n0:], "warm": []}
    check(len(t1["gens"]) == 2 and _prefilled(t1)
          and preface.strip() in got["content"]
          and "Here is the drawing." in got["content"],
          "[hop] the client was streamed the prefill hop's content AND the "
          "answer (the shape of the live 7-generation turn)",
          f"gens={len(t1['gens'])} content={got['content'][:120]!r}")
    msgs.append({"role": "assistant", "content": got["content"]})
    msgs.append({"role": "user", "content": "Thanks."})
    _script[:] = [reply("You are welcome.")]
    n1 = len(_gens)
    _stream(dict(body, messages=json.loads(json.dumps(msgs))))
    t2 = {"gens": _gens[n1:], "warm": []}
    _extends(t1, t2, "[hop] the request after a streamed prefill + hidden "
                     "hop turn")
    x2 = t2["gens"][0]["request"]["messages"]
    check(any(m.get("role") == "tool" and m.get("tool_call_id") == "g9"
              for m in x2),
          "[hop] the hidden yama_generate_image hop is restored from the ledger")
    check(any(m.get("role") == "assistant"
              and DIRECTIVE in (m.get("reasoning_content") or "")
              for m in x2),
          "[hop] and the prefilled line with the hop's reasoning")


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
    our trace, the ledger restoring reasoning that contains '</think>', a
    prefill rendered twice. Rendered here with the served template: an
    echoing client (it sends back every reasoning byte it was shown, our
    trace line of a hidden hop included) and a stripping one, the same
    scripted agent loop -- the next request must be IDENTICAL between the
    two (the echo is the hidden hops' reasoning run together: HIDDEN HOPS
    AND AN ECHOING CLIENT), with one balanced think block per assistant
    turn."""
    renders: dict[str, list[str]] = {}
    draw = proxy._draw_seed
    # One seed for both runs: each is a new conversation, and its first
    # user turn carries its own seed line; only the echo is being compared.
    proxy._draw_seed = lambda prompt=None: {"word": "cedar", "token_id": 1,
                                            "u32": 1}
    try:
        _echo_runs(renders)
    finally:
        proxy._draw_seed = draw
    s2 = renders["strip"][1].split("[agent loop]", 1)[1]
    e2 = renders["echo"][1].split("[agent loop]", 1)[1]
    check("<think>\nWrite it.\n</think>" in s2,
          "[echo] a stripping client's past turn renders the slot's own "
          "reasoning (restored), not the trace it was shown")
    check(s2 == e2,
          "[echo] an echoing client's next request renders exactly like the "
          "stripping client's (the echo is recognised as what it was shown)",
          f"diverge at {_div(s2, e2)}: strip "
          f"{s2[max(0, _div(s2, e2) - 120):_div(s2, e2) + 120]!r} vs echo "
          f"{e2[max(0, _div(s2, e2) - 120):_div(s2, e2) + 120]!r}")
    check(all(_think_blocks_balanced(t) for t in renders["echo"]
              + renders["strip"]),
          "[echo] one balanced think block per assistant turn in every "
          "request, either client")


def _echo_runs(renders: dict) -> None:
    for mode in ("strip", "echo"):
        slots.reset(n=4)
        msgs = [{"role": "system", "content": SYSTEM + " [agent loop]"},
                {"role": "user", "content": "Draw a lathe, write hi.py with a "
                                            "function f that returns 1, then "
                                            "say done."}]
        steps = [
            [reply("", reasoning="Draw it.", calls=[call(
                "yama_generate_image", {"prompt": "a lathe"}, "e0")]),
             reply("", reasoning="Write it.", calls=[call(
                 "write_file", {"content": FIXED, "path": "hi.py"}, "e1")])],
            [reply("Done.", reasoning="It is written.")]]
        out = []
        for step, script in enumerate(steps):
            _script[:] = list(script)
            n0, w0 = len(_gens), len(_warms)
            got = _stream({"model": "yamadori", "reasoning_effort": "xhigh",
                           "_account": ACCOUNT + "-" + mode,
                           "_client_ip": "127.0.0.1", "tools": [WRITE],
                           "messages": json.loads(json.dumps(msgs)),
                           "_features": json.dumps({"skills": False})})
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
            check(any("yama_generate_image" in (m.get("reasoning_content")
                                                or "")
                      for m in msgs if m.get("role") == "assistant"),
                  "[echo] the echoing client really echoed our trace line "
                  "(`yama_generate_image ...`) as reasoning")


def test_template_markers_are_scrubbed_on_our_path_and_recorded_from_the_model():
    """#12, the guard. OUR path never puts the template's markers into a
    prompt or into content; stray markers the MODEL wrote in its answer are
    removed from the client's copy (with the repeat after them) and
    recorded (x_yamadori.template_markers), not hidden (operator,
    2026-09-25, reversing "delivered as written"; mcp/test_stray_markers.py
    has the filter's own gate)."""
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
    # 4. OUR injected text (a skill here; the work log takes the same path)
    #    carries markers: scrubbed when it is decided, counted, and replayed
    #    byte for byte (pre-deploy review, 2026-09-24).
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


def test_the_ledger_never_blanks_a_final_decision():
    """nebari.ledger_decide / ledger_claim: the rule every injection decision
    goes through (pre-deploy review, 2026-09-24: a duplicate request that
    decided less overwrote a recorded decision with ""; the library help it
    blanked was removed 2026-09-29, the rule stays). A retryable decision is
    replaced; a final non-empty one never."""
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
    check(nebari.ledger_claim("acct-c", "s", "k-empty", "inject", "") == ""
          and nebari.ledger_claim("acct-c", "s", "k-empty", "inject", "X")
          == "X"
          and nebari.ledger_claim("acct-c", "s", "k-empty", "inject", "")
          == "X",
          "ledger_claim: an empty decision may be replaced, a non-empty one "
          "never")
    nebari.ledger_reset()


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


def test_a_client_that_acts_on_the_calls_waits_for_the_warm():
    """#5, the live gate of 2026-09-24 (then the repair path; here a write
    whose arguments the model wrote unsorted, which the next request renders
    sorted, so the turn is warmed: the next request extended less than the
    prompt the slot had generated on). A STREAMING
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
            "_features": json.dumps({"skills": False})}
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
    """Live gate 2026-09-24 (second run), then the repair test (here a write
    warmed for its argument order): the next request
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
            "_features": json.dumps({"skills": False})}
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
        # Now a STRIPPING client's first turn: a write, warmed (its
        # arguments' order).
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
            check(bool(t["warm"]), f"[warm] {label}: the write whose "
                                   f"arguments the model wrote unsorted was "
                                   f"warmed")
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
    a new task (and not the conversation's first user turn: no seed)."""
    c.msgs += [{"role": "user", "content": task},
               {"role": "assistant", "content": "", "tool_calls": [call(
                   "read_file", {"path": "src/scene.js"}, "open1")]},
               {"role": "tool", "tool_call_id": "open1",
                "content": '{"content": "1|// scene"}'}]


class Unforced(Client):
    """A client with an account of its own, its own tools, and a tag in its
    system prompt (so two of them never share a conversation)."""

    def __init__(self, effort: str, tag: str, tools: list | None = None):
        super().__init__(effort)
        self.tag = tag.replace(" ", "-")
        self.msgs[0]["content"] += f" [{tag}]"
        self.tools = tools or [WRITE]

    def body(self, features: dict | None = None) -> dict:
        b = super().body(features)
        feats = {"skills": True}
        feats.update(EXTRA_FEATURES)
        feats.update(features or {})
        b["_features"] = json.dumps(feats)
        b["tools"] = self.tools
        # An account of its own: session() compacts in place under ACCOUNT,
        # and a new conversation of that account soon after is linked to it
        # (proxy._continue_after_compaction) -- one lineage.
        b["_account"] = f"{ACCOUNT}-{self.tag}"
        return b


TERMINAL = {"type": "function", "function": {
    "name": "terminal", "description": "Run a shell command.",
    "parameters": {"type": "object", "properties": {
        "command": {"type": "string"}}}}}


def test_old_tool_names_still_replay_and_run():
    """The yama_* rename (2026-09-27): a hidden hop stored under an OLD name
    (generate_image, recall_craft, ...) replays byte for byte, and a call the
    model makes by an old name -- copied from such a hop -- runs as the new
    tool instead of going to the client."""
    slots.reset(n=4)
    compaction.reset()
    c = Unforced("xhigh", "legacy names", tools=[WRITE, READ_TOOL])
    _agent_step_opening(c, "Draw a lathe for the README.")
    t1 = c.turn([reply("", reasoning="Old habit.", calls=[call(
                     "generate_image", {"prompt": "a lathe"}, "old1")]),
                 reply("Here it is.")])
    x = t1["d"]["x_yamadori"]
    ran = [e for e in x.get("tools") or []
           if e.get("name") == "yama_generate_image"]
    ans = t1["d"]["choices"][0]["message"]
    check(ran and not ans.get("tool_calls") and len(t1["gens"]) == 2,
          "[legacy] a call by the old name generate_image runs as "
          "yama_generate_image; the client never sees it",
          json.dumps(x.get("tools"))[:300])
    t2 = c.turn([reply("You are welcome.")], user="Thanks.")
    msgs2 = t2["gens"][0]["request"]["messages"]
    check(any(m.get("role") == "assistant" and any(
                  cl["function"]["name"] == "generate_image"
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
    """Octopus v0b-V0-xhigh-1, step 1 (2026-09-25): a turn whose prefill
    (then the kickoff plan) opened main's reasoning, prompt 10660; the warm
    reused 7526, processed 3055 and was flagged SHORT against 10660, and the
    next request reused 10581 of 10611. Rendered here with the served
    template, one token per character: the warm renders the turn STRIPPED,
    as the client sends it (the prefill is not in it), so it is right; the
    prompt diverges from it where the prefill begins, after the last user
    message, and the server can restore only its checkpoint at that
    message's start. That is what a prefilled turn's warm is judged
    against; a real miss (reused 0) is still SHORT.

    Since 2026-09-29 the prefill is a DIRECTIVE (proxy.directive_prefill,
    the channel kept for skills), the think block left open -- the same
    prefill, the same checkpoint rule.

    THE PASS-THROUGH ARM (switch restore_reasoning off): with reasoning
    restored the prefill comes back with the turn and no warm is needed
    (test_a_prefilled_turn_is_the_slots_sequence_when_reasoning_is_restored).
    """
    saved = dict(EXTRA_FEATURES)
    EXTRA_FEATURES["restore_reasoning"] = False
    try:
        _prefilled_warm_pass_through()
    finally:
        EXTRA_FEATURES.clear()
        EXTRA_FEATURES.update(saved)


def _prefilled_warm_pass_through():
    global BY_CHARS
    spec = ("Build a space shooter with vanilla JavaScript and canvas. "
            "PLAYER: moves with arrow keys, fires with space. " * 120)
    shots = (("terminal (the live shape)", "last_user",
              call("terminal", {"command": "mkdir -p css js"}, "k1")),
             ("a write", "last_user",
              call("write_file", {"path": "hi.py", "content": FIXED}, "k1")),
             ("a real miss", "zero",
              call("terminal", {"command": "mkdir -p css js"}, "k1")))
    try:
        for label, mode, first in shots:
            slots.reset(n=4)
            compaction.reset()
            BY_CHARS = mode
            tag = f"[prefill warm: {label}]"
            c = Unforced("xhigh", f"prefill warm {label}",
                         tools=[WRITE, TERMINAL])
            with _Prefill():
                t1 = c.turn([reply(" I will start.", reasoning="Go.",
                                   calls=[first])], user=spec)
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
                check(_prefilled(t1) and w and DIRECTIVE in a[u:]
                      and DIRECTIVE not in w and a[:n].endswith(
                          "<|im_start|>assistant\n<think>\n") and n > u
                      and r.startswith(w),
                      f"{tag} RENDER: the warm is the turn as the client "
                      f"sends it (no prefill); the prompt diverges from it "
                      f"where the prefill begins (char {n}), after the last "
                      f"user message (char {u})",
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
        BY_CHARS = None


LONG_DIRECTIVE = ("".join(
    f"Step {i}: the unleash ring counts down in update() and is drawn from "
    f"it. " for i in range(30)) + "So next I'll patch js/player.js")


def _mid_prefill_session(tag: str, mode: str,
                         prefill_reasoning: str = "Patch.") -> dict:
    """An agent session of terminal steps on one slot of the simulated
    server (BY_CHARS "server"), then a step that opens with a long directive
    prefill (proxy.directive_prefill, injected: _Prefill), its warm served
    in `mode`, then the next step. `prefill_reasoning`: what the fake adds
    to the prefilled reasoning."""
    global BY_CHARS, SIM_MODE
    slots.reset(n=4)
    compaction.reset()
    _sim.clear()
    _sim_log.clear()
    BY_CHARS, SIM_MODE = "server", "thinned"
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
        with _Prefill(LONG_DIRECTIVE):
            t = c.turn([reply(" I will patch the countdown.",
                              reasoning=prefill_reasoning,
                              calls=[call("terminal", {"command": "node --check "
                                                       "js/player.js"}, "sp")])])
        c.tool_result("sp", "ok")
        # The next request waits for the warm (wait_for_warm), which is
        # served in `mode` (it applies to warms only).
        t2 = c.turn([reply(" Checked.", reasoning="Fine.")])
        return {"t": t, "t2": t2, "log": _sim_log[n_log:],
                "all": list(_sim_log)}
    finally:
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
    (Since 2026-09-29 the prefill here is a long DIRECTIVE, the channel the
    proxy keeps -- proxy.directive_prefill -- in place of the removed
    struggle hand-off: the same prefill mechanics.)

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
    wb = t2["d"]["x_yamadori"].get("warm_before") or {}
    gen = next((g for g in log if g["kind"] == "gen"), {})
    warm = next((g for g in log if g["kind"] == "warm"), {})
    nxt = [g for g in log if g["kind"] == "gen"][1:2]
    w = t["warm"][-1]["prompt"] if t["warm"] else ""
    a = render(t["gens"][0]["request"])
    d = _div(a, w)
    turn_at = w.rfind("<|im_start|>assistant")
    check(_prefilled(t) and w and gen and warm
          and d < len(a) - 500 and a[:d].endswith(
              "<|im_start|>assistant\n<think>\n"),
          "[mid prefill] the warm renders the turn as the client sends it "
          "and diverges from the prefilled prompt where the prefill begins",
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
    """PAST REASONING IS RESTORED (operator, 2026-09-27): a turn that opened
    with a prefill -- a directive (proxy.directive_prefill), the channel
    kept for skills -- comes back WITH it (the ledger recorded what the slot
    holds), so the next request extends the slot: no checkpoint re-read,
    and nothing to warm when the next request renders the turn as the slot
    holds it (the live shape: a terminal call). A turn it renders
    differently (a write whose arguments the model wrote unsorted) is warmed
    with the prefill in it."""
    global BY_CHARS
    spec = ("Build a space shooter with vanilla JavaScript and canvas. "
            "PLAYER: moves with arrow keys, fires with space. " * 40)
    shots = (("terminal (the live shape)",
              call("terminal", {"command": "mkdir -p css js"}, "k1")),
             ("a write",
              call("write_file", {"path": "hi.py", "content": FIXED}, "k1")))
    try:
        for label, first in shots:
            slots.reset(n=4)
            compaction.reset()
            BY_CHARS = None
            tag = f"[prefill restored: {label}]"
            c = Unforced("xhigh", f"prefill restored {label}",
                         tools=[WRITE, TERMINAL])
            with _Prefill():
                t1 = c.turn([reply(" I will start.", reasoning="",
                                   calls=[first])], user=spec)
            c.tool_result("k1", "ok")
            t2 = c.turn([reply("Next.")])
            x1, x2 = t1["d"]["x_yamadori"], t2["d"]["x_yamadori"]
            r = render(t2["gens"][0]["request"])
            held = reusable_after(t1)
            last = r[:len(held)].rfind("<|im_start|>assistant")
            check(_prefilled(t1) and r.startswith(held)
                  and DIRECTIVE in r[last:],
                  f"{tag} the next request extends the slot: the prefilled "
                  f"line is in the turn it renders",
                  json.dumps({"warm": x1.get("warm"),
                              "div": _div(held, r), "held": len(held)}))
            lg = x2.get("ledger") or {}
            check((lg.get("restore_reasoning") or {}).get("on") is True
                  and (lg.get("restored") or {}).get("reasoning", 0) >= 1
                  and (x1.get("ledger") or {}).get("reasoning_recorded", 0)
                  >= len(DIRECTIVE),
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
                check(w and DIRECTIVE in w[w.rfind("<|im_start|>assistant"):],
                      f"{tag} the warm loads the delivered turn WITH its "
                      f"prefilled line", json.dumps(x1.get("warm")))
    finally:
        BY_CHARS = None


def test_a_prefill_mid_conversation_extends_when_reasoning_is_restored():
    """The Octopus v0f-V0 step-25 shape (a prefill mid-conversation) on the
    simulated server, reasoning restored: every request of the session --
    the nine agent steps, the prefilled step and the one after it --
    EXTENDS what the slot holds (no checkpoint restore, reused == held), and
    the prefilled turn needs no warm. Before the switch the step after a
    prefill re-read back to a thinned checkpoint (8,793 tokens live;
    test_a_prefill_mid_conversation_rereads_to_the_thinned_checkpoint, the
    pass-through arm)."""
    r = _mid_prefill_session("restored", "thinned", prefill_reasoning="")
    x = r["t"]["d"]["x_yamadori"]
    gens = [g for g in r["all"] if g["kind"] == "gen" and g["held"]]
    bad = [g for g in gens if not g["extends"] or g["reused"] != g["held"]]
    check(_prefilled(r["t"]) and len(gens) >= 10 and not bad,
          f"[mid prefill restored] all {len(gens)} requests after the first "
          f"extend the slot: reused == what it held, nothing re-read",
          json.dumps(bad[:3]))
    check(not (x.get("warm") or {}).get("sent")
          and not [g for g in r["log"] if g["kind"] == "warm"],
          "[mid prefill restored] the prefilled turn needs no warm",
          json.dumps(x.get("warm")))
    nxt = [g for g in r["log"] if g["kind"] == "gen"][1:2]
    check(nxt and nxt[0]["processed"] < 400,
          "[mid prefill restored] the step after the prefill processes only "
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
    sorting client and an order-keeping one both extend the warm (the order
    alone makes the turn differ from what the slot generated; the code
    check's note, which also did at medium, was removed 2026-09-29)."""
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
    feats = {"skills": False}
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
    check("situations" not in prog and "project" not in prog
          and not {"progress_note", "unchanged_read"} & set(
              prog.get("switches") or {}),
          "[situations] x_yamadori.progress has no situations, no switch "
          "for them and no project (removed with deep thinking, 2026-09-29)",
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
    internal request -- is scanned for the removed text (the replay
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
    helpers = list(_helper_direct.archive) + list(_helper_direct)
    for h in helpers:
        hits += [("helper", s) for s in REMOVED_STEERING
                 if s in json.dumps(h)]
    check(len(_gens) > 50 and not hits,
          f"[removed] none of the removed steering text is in any of the "
          f"{len(_gens)} generations, {len(_warms)} warms or "
          f"{len(helpers)} internal requests "
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
    feats = {"skills": False}
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
    feats = {"skills": False}
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


def test_the_concept_seed_rides_the_first_user_turn():
    """THE CONCEPT SEED (operator, 2026-09-29): drawn once per conversation
    (concept_seed.seed_for, via proxy._draw_seed) and appended to its FIRST
    user turn with the rest of that turn's injection (proxy.prepare,
    concept_seed.user_turn_line), where the tier allows it (`seed`: high,
    xhigh, max). The ledger replays it byte for byte; it is recorded by the
    conversation's lineage and reported as x_yamadori.session.seed on every
    later request, a compaction included; a header {"seed": false} turns it
    off."""
    import concept_seed
    slots.reset(n=4)
    compaction.reset()
    c = Unforced("xhigh", "seed first turn", tools=[WRITE])
    feats = {"skills": False}
    t1 = c.turn([reply("Sure.")], user="Build a tiny canvas game.",
                features=feats)
    seed = (t1["d"]["x_yamadori"].get("session") or {}).get("seed") or {}
    word = seed.get("word")
    line = concept_seed.USER_TURN_LINE.format(word=word)
    u1 = [m for m in t1["gens"][0]["request"]["messages"]
          if m.get("role") == "user"]
    inj = t1["d"]["x_yamadori"]["ledger"]["inject"] or {}
    # (This suite's skills stub attaches HINT to every user turn it decides.)
    check(word and u1[0]["content"] == "Build a tiny canvas game." + HINT
          + line and inj.get("parts") == ["skills", "seed"]
          and set(seed) >= {"word", "u32"},
          "[seed] the first user turn carries the seed line, after the "
          "skills; x_yamadori.session.seed names it",
          json.dumps({"seed": seed, "inject": inj}))
    check((concept_seed.last() or {}).get("word") == word
          and (concept_seed.last() or {}).get("where") == "first_turn",
          "[seed] the dashboard's last-seed record names it")
    t2 = c.turn([reply("Done.")], user="Now add a score.", features=feats)
    u2 = [m for m in t2["gens"][0]["request"]["messages"]
          if m.get("role") == "user"]
    seed2 = (t2["d"]["x_yamadori"].get("session") or {}).get("seed")
    check(u2[0]["content"] == u1[0]["content"]
          and u2[-1]["content"] == "Now add a score." + HINT
          and seed2 == seed,
          "[seed] a later user turn carries none; the first turn's line "
          "replays byte for byte and the session still reports the seed",
          json.dumps({"first": u2[0]["content"][-120:],
                      "last": u2[-1]["content"][-80:], "seed": seed2}))
    _extends(t1, t2, "[seed] the request after the seeded turn")
    t3 = c.turn([reply("## Goal\nA canvas game with a score.")],
                user="Your task is to create a detailed summary of the "
                     "conversation so far.", features=feats)
    x3 = t3["d"]["x_yamadori"]
    check(x3.get("utility_kind") == "compaction"
          and (x3.get("session") or {}).get("seed") == seed,
          "[seed] an in-place compaction of the conversation reports the same "
          "seed", json.dumps(x3.get("session")))
    for effort, f, why in (("medium", feats, "tier medium"),
                           ("xhigh", dict(feats, seed=False),
                            'X-Yamadori-Features {"seed": false}')):
        slots.reset(n=4)
        d = Unforced(effort, f"seed off {effort} {len(f)}", tools=[WRITE])
        t = d.turn([reply("Ok.")], user="Build a tiny canvas game.",
                   features=f)
        u = [m for m in t["gens"][0]["request"]["messages"]
             if m.get("role") == "user"]
        check(u[0]["content"] == "Build a tiny canvas game." + HINT
              and (t["d"]["x_yamadori"].get("session") or {}).get("seed")
              is None,
              f"[seed] {why}: no seed line, no seed recorded",
              u[0]["content"][-120:])


def main() -> int:
    for fn in (test_keys, test_scope_persistence_and_pruning,
               test_call_arguments_render_in_one_order,
               test_prefix_stability_across_a_session,
               test_a_streamed_hop_after_a_prefill_is_restored,
               test_old_tool_names_still_replay_and_run,
               test_a_prefilled_turn_warms_from_the_last_user_checkpoint,
               test_a_prefill_mid_conversation_rereads_to_the_thinned_checkpoint,
               test_a_prefilled_turn_is_the_slots_sequence_when_reasoning_is_restored,
               test_a_prefill_mid_conversation_extends_when_reasoning_is_restored,
               test_the_window_check_counts_restored_reasoning,
               test_the_served_template_renders_every_past_think_block,
               test_the_warm_reports_on_the_next_response,
               test_a_client_that_acts_on_the_calls_waits_for_the_warm,
               test_a_first_turn_warm_does_not_guess_echo_from_a_mixed_account,
               test_a_warm_releases_only_its_own_waiters,
               test_text_turn_keys_are_per_conversation,
               test_the_ledger_never_blanks_a_final_decision,
               test_a_retryable_decision_is_decided_again,
               test_the_echo_path_renders_like_the_strip_path,
               test_template_markers_are_scrubbed_on_our_path_and_recorded_from_the_model,
               test_no_situation_line_and_old_rows_replay,
               test_step_cap_and_nudge,
               test_the_work_log_returns_after_a_compaction_that_keeps_the_key,
               test_the_concept_seed_rides_the_first_user_turn,
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
    # jjava is always in scope where skills serve (operator, 2026-09-29):
    # offline, the STUBBED decider answers (mcp/decider_stub.py).
    import decider_stub
    with decider_stub.installed():
        sys.exit(main())
