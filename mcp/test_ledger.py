#!/usr/bin/env python
"""The session ledger and the prefix-stability gate. No GPU.

    python mcp/test_ledger.py      -> "N/M checks passed"

WHAT THIS IS GATING (docs/SELF-IMPROVEMENT-PLAN.md Phase 0.5, operator
2026-09-24): one model, one cache.

  1. PREFIX STABILITY. A multi-turn session through the real complete(),
     with a client that strips everything the proxy added -- reasoning, the
     skills block on a user turn, image-tool hops -- reuses, on every
     request, everything the slot holds before the previous assistant turn:
     the previous request's prompt up to that turn's think block (past
     reasoning passes through since 2026-09-24, so the slot's generated
     reasoning is not re-sent), or, when the proxy warmed the slot
     (proxy._warm), the whole warmed prompt. The processed tail is that
     turn + the tool result + the new part. The rendering is the SERVED
     chat template (mcp/fixtures/bonsai_chat_template.jinja, captured from
     llama-server /props on 2026-09-24, build b10738-285542d9), run by
     jinja2 -- STEP 0 measured that llama-server reuses a slot fully only
     when a request extends its sequence.
     The session covers: an injected skill on a user turn, a deep-thinking
     hand-off prefilled as reasoning (with the seed line), a client write
     repaired by the second brain and warmed, a generate_image hop the
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

import compaction  # noqa: E402
import nebari  # noqa: E402
import proxy  # noqa: E402
import shomen  # noqa: E402
import slots  # noqa: E402
import tiers  # noqa: E402
import model as _model  # noqa: E402

proxy.PREAMBLE = False
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
    return text


# --------------------------------------------------------- the fake upstream
# What the fake /completion (a warm) reports: 100 prompt tokens reused of a
# generation whose prompt was 100 (every fake generation reports 90 + 10).
WARM_TIMINGS = {"cache_n": 100, "prompt_n": 20}
WARM_DELAY = 0.0                  # seconds the fake /completion takes
_warm_done_at: list[float] = []   # when each fake /completion finished
_script: list[dict] = []
_gens: list[dict] = []            # {request, reply} per generation
_warms: list[dict] = []           # /completion bodies
_helper: list[dict] = []          # the second brain's requests
_helper_direct: list[dict] = []   # ... when sent through model.post itself


def reply(content: str = "", reasoning: str = "", calls: list | None = None
          ) -> dict:
    return {"content": content, "reasoning": reasoning, "calls": calls or []}


def call(name: str, args: dict, cid: str) -> dict:
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def _sse(r: dict, finish: str) -> bytes:
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
        "total_tokens": 110}, "timings": {"cache_n": 90,
                                          "prompt_n": 10}}).encode() + b"\n\n")
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
        if self.path.endswith("/completion"):
            _warms.append(body)
            if WARM_DELAY:
                time.sleep(WARM_DELAY)
            _warm_done_at.append(time.time())
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
        data = _sse(r, "tool_calls" if r["calls"] else "stop")
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


def _skills_tail(messages, sel, route):
    return HINT, [], {"path": "test", "on": True, "ids": ["s1"],
                      "versions": [1], "why": "test"}


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
    if name == "generate_image":
        return "![a lathe](https://media.example/abc.png)"
    return _real_run_our_tool(name, args, db, root, turn, state)


proxy.run_our_tool = _run_our_tool

WRITE = {"type": "function", "function": {
    "name": "write_file", "description": "Write a file.",
    "parameters": {"type": "object", "properties": {
        "path": {"type": "string"}, "content": {"type": "string"}}}}}
SYSTEM = "You are a coding agent."
ACCOUNT = "ledger-acct"


class Client:
    """A harness that STRIPS: it keeps what it was sent as content and tool
    calls, drops reasoning, and never sees what the proxy added."""

    def __init__(self, effort: str):
        self.effort = effort
        self.msgs = [{"role": "system", "content": SYSTEM}]

    def body(self, features: dict | None = None) -> dict:
        b = {"model": "yamadori", "reasoning_effort": self.effort,
             "_account": ACCOUNT, "_client_ip": "127.0.0.1",
             "tools": [WRITE], "messages": json.loads(json.dumps(self.msgs))}
        feats = {"hints": True, "investigate": False, "fanout": 1}
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
        self.msgs.append(kept)
        warm = (d.get("x_yamadori") or {}).get("warm") or {}
        if warm.get("sent"):
            end = time.time() + 5
            while time.time() < end and len(_warms) == w0:
                time.sleep(0.02)
        return {"d": d, "gens": _gens[n0:], "warm": _warms[w0:]}

    def tool_result(self, cid: str, text: str) -> None:
        self.msgs.append({"role": "tool", "tool_call_id": cid,
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


def reusable_after(t: dict) -> str:
    """What the next request may reuse, now that past reasoning passes
    through (design change, 2026-09-24): the warmed prompt when the proxy
    warmed the slot (it renders the turn as the client will send it), else
    the FIRST generation's prompt of the turn up to its opening think block
    -- the slot generated the reasoning, the client sends none (or its echo),
    so the next request diverges there and rests on the checkpoint at that
    prompt's end. What follows it -- the last assistant turn (its hidden
    hops included), the tool result, the new part -- is the processed
    tail."""
    if t["warm"]:
        return t["warm"][-1]["prompt"]
    p = render(t["gens"][0]["request"])
    i = p.rfind(_OPEN)
    return p[:i + len(_OPEN)] if i >= 0 else p


def _extends(prev: dict, nxt: dict, label: str) -> bool:
    h, r = reusable_after(prev), first_render(nxt)
    n = next((i for i, (a, b) in enumerate(zip(h, r)) if a != b),
             min(len(h), len(r)))
    tail = r[len(h):] if r.startswith(h) else ""
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
    ans = t1["d"]["choices"][0]["message"]["content"]
    word = (t1["d"]["x_yamadori"]["investigate"].get("seed") or {}).get("word")
    check(pre.get("role") == "assistant" and "LatheGeometry sweeps"
          in pre.get("reasoning_content", "")
          and ans.startswith(f"Today I was inspired by {word}. After thinking "
                             f"deeply, it sweeps"),
          f"{tag} the hand-off is prefilled as reasoning; the answer opens "
          f"with the seed line and the phrase", ans[:120])
    check(any(f"Inspiration word: {word}" in m.get("content", "")
              for h in _helper for m in h["messages"]
              if m.get("role") == "user"),
          f"{tag} the investigate job's USER message names the same seed")
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

    # 4. An image: a hidden internal hop (generate_image) the client never
    #    sees, then the answer.
    b0 = _bytes()
    t4 = c.turn([reply("", reasoning="Draw it.",
                       calls=[call("generate_image",
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
          and req.get("reasoning_budget_tokens") == tiers.COMPACTION_THINKING,
          f"{tag} the compaction thinks at the conversation's effort "
          f"({req.get('reasoning_effort')}), {tiers.COMPACTION_THINKING} of "
          f"it", json.dumps({k: req.get(k) for k in (
              "enable_thinking", "reasoning_effort",
              "reasoning_budget_tokens")}))

    # A RETRY of turn 1's request (the same messages): the ledger replays
    # its seed; nothing new is drawn.
    retry = Client(effort)
    retry.msgs = c.msgs[:2]
    t1b = retry.turn([reply(" it sweeps.")], features={"investigate": True})
    word_b = (t1b["d"]["x_yamadori"]["investigate"].get("seed") or {}).get(
        "word")
    check(word_b == word,
          f"{tag} a replay of the request uses the recorded seed ({word})",
          f"{word_b} vs {word}")


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
    _script[:] = [reply(preface, calls=[call("generate_image",
                                             {"prompt": "a lathe"}, "g9")]),
                  reply(" Here is the drawing.", reasoning="Show it.")]
    n0 = len(_gens)
    body = {"model": "yamadori", "reasoning_effort": "xhigh",
            "_account": ACCOUNT, "_client_ip": "127.0.0.1", "tools": [WRITE],
            "messages": json.loads(json.dumps(msgs)),
            "_features": json.dumps({"hints": True, "investigate": True,
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
                 _features=json.dumps({"hints": True, "investigate": False,
                                       "fanout": 1})))
    t2 = {"gens": _gens[n1:], "warm": []}
    _extends(t1, t2, "[hop] the request after a streamed prefill + hidden "
                     "hop turn")
    x2 = t2["gens"][0]["request"]["messages"]
    check(any(m.get("role") == "tool" and m.get("tool_call_id") == "g9"
              for m in x2),
          "[hop] the hidden generate_image hop is restored from the ledger")


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
    # SINCE 2026-09-24 PAST REASONING PASSES THROUGH (design change): the
    # echoing client's second request renders ITS echo, the stripping
    # client's an empty think block. (At the live gate the ledger restored
    # reasoning, so the two rendered identically -- that is the evidence
    # #12's attribution rests on; this checks the new design.)
    s2 = renders["strip"][1].split("[agent loop]", 1)[1]
    e2 = renders["echo"][1].split("[agent loop]", 1)[1]
    check("<think>\n\n</think>" in s2 and "Write it." not in s2,
          "[echo] a stripping client's past turn renders with no reasoning "
          "(nothing restored)")
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
                           "_features": json.dumps({"hints": False,
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
    prompt or into content; markers the MODEL wrote reach the client as
    written and are recorded (x_yamadori.template_markers), not hidden."""
    global HANDOFF
    # 1. The model writes them: left as written, recorded as the model's.
    slots.reset(n=4)
    c = Client("xhigh")
    c.msgs[0]["content"] += " [markers 1]"
    t = c.turn([reply("Done.\n</think>\n\nDone.\n</think>\n\nDone.",
                      reasoning="Done.")], user="Say done.")
    x = t["d"]["x_yamadori"]
    tm = x.get("template_markers") or {}
    check(t["d"]["choices"][0]["message"]["content"].count("</think>") == 2
          and tm.get("source") == "model"
          and tm.get("model", {}).get("</think>") == 2 and not tm.get("ours"),
          "markers the model wrote are delivered as written and recorded as "
          "the model's", json.dumps(tm))
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
    covered on the message the request ends on: an overview when no name is
    known yet, else the imported names' definitions. Once per package and
    name; replayed byte-identically, so every request still extends the
    slot."""
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
        feats = {"hints": False}
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
              and "exported classes and functions" in last.get("content", "")
              and "loadFont" in last["content"]
              and "helper" not in last["content"]
              and lu2.get("overview") == ["@pmndrs/glyph"],
              "an agent_step whose tool result is a manifest naming a held "
              "package gets that package's exported API, appended to the "
              "tool result", json.dumps(lu2))
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
        _extends(t2, t3, "[library use] the request after the first "
                         "injection (replayed byte for byte)")
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
        r2 = next(m for m in t4["gens"][0]["request"]["messages"]
                  if m.get("tool_call_id") == "r1")
        check("exported classes and functions" in r2.get("content", ""),
              "the first injection is still on its tool result, restored "
              "from the ledger")
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
    feats = {"hints": False}
    c.turn([reply("", calls=[call("read_file", {"path": "a.ts"}, "d1")])],
           user="Use the glyph library.", features=feats)
    c.tool_result("d1", "import { Text } from '@pmndrs/glyph'\n")
    saved = proxy._library_use
    first = "\n\n== @pmndrs/glyph@0.1.0: Text ==\nexport class Text"

    def duplicate_finished_first(raw, account, lineage, state=None):
        # The other copy of this request recorded its decision while this
        # one was deciding; this one found nothing left to add.
        nebari.ledger_put(account, lineage, proxy.chain_keys(raw)[-1],
                          "inject", first)
        return "", {"packages": [], "names": [], "overview": [], "chars": 0}
    proxy._library_use = duplicate_finished_first
    try:
        t = c.turn([reply("ok")], features=feats)
    finally:
        proxy._library_use = saved
    last = t["gens"][0]["request"]["messages"][-1]
    key = proxy.chain_keys(c.msgs[:-1])[-1]
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


def test_text_turn_keys_are_per_conversation():
    """Pre-deploy review, 2026-09-24 (MINOR): a text turn's ledger key was
    the hash of its content alone, so two of an account's conversations in
    which the assistant said the same words shared it, and one's recorded
    content and hidden hops were restored into the other."""
    nebari.ledger_reset()
    a = [{"role": "user", "content": "Draw a fox, then say done."}]
    b = [{"role": "user", "content": "Rename the variable, then say done."}]
    hop = [{"role": "assistant", "content": "", "tool_calls": [call(
        "generate_image", {"prompt": "a fox"}, "gA")]},
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
            "_features": json.dumps({"hints": False, "investigate": False,
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
            # formatter's view is only reported.
            check("Repaired hi.py" in content and "would change" in content
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
        feats = {"hints": True, "fanout": 1}
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
        "against source)\nRISKS\n- input lag: test on keydown\n")


def test_think_deeply_is_a_hidden_hop_the_next_request_extends():
    """Phase 0.6, trigger 1 (operator, 2026-09-24): main calls think_deeply;
    the proxy runs it through the second brain as an INTERNAL hop -- the
    call and the hand-off as its tool result -- the client never sees it,
    the next hop is prefilled with the fold-back opening, and the ledger
    replays the hop so the next request extends the slot."""
    import deep
    slots.reset(n=4)
    compaction.reset()
    _helper.clear()
    c = Unforced("xhigh", "think session")
    q = "Why does LatheGeometry need its points sorted by y?"
    t1 = c.turn([reply("", reasoning="Not sure of the API.",
                       calls=[call("think_deeply", {"question": q,
                                                    "tried": "reversing"},
                                   "td1")]),
                 reply(" sort the points by y.", reasoning="")],
                user="My LatheGeometry renders inside out. Why?")
    x = t1["d"]["x_yamadori"]
    req0 = t1["gens"][0]["request"]
    names = [t["function"]["name"] for t in req0.get("tools") or []]
    sys0 = req0["messages"][0]["content"]
    check("think_deeply" in names and proxy.ADDENDUM_THINK_ROW.strip() in sys0,
          "[think] at xhigh main has think_deeply, and the addendum its row",
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
    _extends(t1, t2, "[think] the request after a think_deeply hop")
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
    t3 = c.turn([reply("", calls=[call("think_deeply", {"question": q},
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
    next steps are in the cooldown, and each request extends the slot."""
    slots.reset(n=4)
    compaction.reset()
    _helper.clear()
    c = Unforced("max", "struggle session", tools=[WRITE, TERMINAL])
    fail = "npm ERR! Test failed. See above for more details.\nexit code 1"
    c.msgs += [{"role": "user", "content": "Make the tests pass."}]
    for i, cid in enumerate(("t0", "t1", "t2")):
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
          "[struggle] three failing runs of the same command (two re-runs, "
          "two repeated errors): the signals reach the threshold and deep "
          "thinking runs",
          json.dumps({k: dx.get(k) for k in ("fire", "kind", "because", "cooldown", "allowed", "forced")})[:600])
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
    check(not d2.get("fire") and (d2.get("cooldown") or {}).get("active"),
          "[struggle] the next step is in the cooldown and the episode "
          "restarted: no second run", json.dumps(d2.get("cooldown")))
    _extends(t1, t2, "[struggle] the request after a struggle prefill")


def test_a_large_task_kickoff_prefills_the_plan():
    """Phase 0.6, trigger 4: a new task whose spec is at least the kickoff
    size sends the PLANNING to the second brain (shomen's plan job); the
    plan is prefilled as main's reasoning and main starts acting."""
    import deep
    slots.reset(n=4)
    compaction.reset()
    _helper.clear()
    real = shomen._post

    def plan_post(path, payload, timeout=3600):
        if str(payload["messages"][0].get("content", "")).startswith(
                "You are planning"):
            _helper.append(json.loads(json.dumps(payload)))
            return {"choices": [{"message": {"content": PLAN},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
        return real(path, payload, timeout)
    shomen._post = plan_post
    try:
        c = Unforced("xhigh", "kickoff session")
        spec = ("Build a space shooter with vanilla JavaScript and canvas. "
                "PLAYER: moves with arrow keys, fires with space. " * 120)
        t1 = c.turn([reply(" I will start with index.html.", calls=[call(
            "write_file", {"path": "index.html", "content": "<canvas>"},
            "w1")])], user=spec)
    finally:
        shomen._post = real
    x = t1["d"]["x_yamadori"]
    pre = t1["gens"][0]["request"]["messages"][-1]
    check((x.get("deep") or {}).get("kind") == "kickoff"
          and (x.get("investigate") or {}).get("job") == "plan"
          and _helper and _helper[-1]["messages"][0]["content"]
          == shomen.PLAN_SYSTEM,
          "[kickoff] a spec over the kickoff size runs the plan job",
          json.dumps((x.get("deep") or {}).get("signals", {}).get(
              "kickoff")))
    check(pre.get("role") == "assistant"
          and pre.get("reasoning_content", "").startswith(deep.PLAN_HEAD)
          and "FILES\n- index.html" in pre["reasoning_content"]
          and "RISKS" in pre["reasoning_content"]
          and pre.get("content", "").endswith("After thinking deeply,"),
          "[kickoff] the plan (FILES, ORDER, KEY DECISIONS, RISKS) is "
          "prefilled as main's reasoning, and main starts acting",
          json.dumps(pre)[:300])
    c.tool_result("w1", "wrote index.html")
    t2 = c.turn([reply("Done with the page.")])
    check(not (t2["d"]["x_yamadori"].get("deep") or {}).get("fire"),
          "[kickoff] the next step (a tool result) is not a new task")
    _extends(t1, t2, "[kickoff] the request after the plan prefill")


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


def main() -> int:
    for fn in (test_keys, test_scope_persistence_and_pruning,
               test_prefix_stability_across_a_session,
               test_a_streamed_hop_after_a_prefill_is_restored,
               test_think_deeply_is_a_hidden_hop_the_next_request_extends,
               test_struggle_runs_deep_thinking_before_main_once_per_episode,
               test_a_large_task_kickoff_prefills_the_plan,
               test_the_warm_reports_on_the_next_response,
               test_a_fixup_step_and_its_warm,
               test_the_fixup_never_touches_the_conversations_slot,
               test_library_use_reaches_harness_traffic,
               test_library_use_reads_the_conversations_version,
               test_a_warm_releases_only_its_own_waiters,
               test_text_turn_keys_are_per_conversation,
               test_a_duplicate_request_never_blanks_the_library_help,
               test_the_echo_path_renders_like_the_strip_path,
               test_template_markers_are_scrubbed_on_our_path_and_recorded_from_the_model):
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
