#!/usr/bin/env python
"""An OpenAI endpoint that brings its own tools.

THE POINT

A client adds one OpenAI-compatible model and gets the whole stack. It declares
no tools, configures no MCP server, and never learns any of this exists. The
proxy appends our tools to whatever the client sent, executes the ones that are
ours, and returns only the final answer.

    client --/v1/chat/completions--> [proxy] --> llama-swap --> model
                                        |
                                        +-- runs our tools itself, in a loop

This is the only way to reach a client that will not configure MCP. The chat
completions API gives a server no channel to initiate a tool call: the model
can only call tools the CLIENT declared, and the CLIENT executes them. So the
tools have to be injected into the request on the way past.

WHAT IT ADDS, AND WHAT THAT COSTS

  system prompt   a static capability block, appended after the client's own
  tools           ours, merged with the client's, ours dropped on name conflict
  repo awareness  worked out from the conversation, see detect.py
  preamble        one line on the first turn of a session, so the user knows
                  what they have without reading a README

KV CACHE. The prompt prefix must stay byte-identical between turns or every
turn pays a full prefill -- ~51s at 65k on this hardware. So the injected
system block is STATIC: no repo name, no index counts, nothing that changes.
Anything dynamic rides in tool results, which append at the end and leave the
prefix intact.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import code_search as cs  # noqa: E402
import detect  # noqa: E402
import repos  # noqa: E402
import corpus  # noqa: E402
import accounts  # noqa: E402
import streaming  # noqa: E402

UPSTREAM = os.environ.get("LLAMA_STACK_URL", "http://127.0.0.1:1234")
PORT = int(os.environ.get("YAMADORI_PROXY_PORT", "1233"))
MAX_TOOL_HOPS = int(os.environ.get("YAMADORI_MAX_HOPS", "12"))
PREAMBLE = os.environ.get("YAMADORI_PREAMBLE", "1") == "1"

# Static by construction -- see the KV CACHE note above. It describes what
# exists, never where we are.
CAPABILITY_BLOCK = """

---
You have a local code-intelligence stack available as tools. It runs on this
machine against an index of the repository in front of you: no network, no
quota. A symbol lookup costs about 19 tokens; reading a file blind to find the
same thing costs thousands. Calling them is close to free, and guessing is not.

  The request names a symbol            -> find_definition_opt
  You want what USES or CALLS it        -> find_references
  The text appears verbatim somewhere   -> find_by_pattern
  The code may use OTHER words          -> find_by_meaning
  You have a path and a line range      -> read_file_range
  You changed something                 -> run_check
  Worth not rediscovering later         -> record_step
  The conversation was just summarised  -> read_rings

These read; they do not write. Your harness supplies whatever edits files and
runs commands.

A result you have not checked is a guess, however good the reasoning behind it.
Never state what code does without having read it -- you can read it cheaply,
so there is no excuse to infer."""


def _post(path: str, payload: dict, timeout: int = 1800) -> dict:
    req = urllib.request.Request(f"{UPSTREAM}{path}",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def our_tools() -> list[dict]:
    return [{"type": "function",
             "function": {"name": t["name"], "description": t["description"],
                          "parameters": t["inputSchema"]}}
            for t in cs.TOOLS]


OUR_NAMES = {t["name"] for t in cs.TOOLS}


def merge_tools(client_tools: list | None) -> tuple[list, set]:
    """Client tools win every name collision.

    If both sides offer `read_file` the model cannot tell them apart and picks
    arbitrarily -- and the client's version is the one with side effects the
    client knows how to handle. Ours is dropped, silently and deliberately.
    """
    client_tools = client_tools or []
    taken = {t.get("function", {}).get("name") for t in client_tools}
    mine = [t for t in our_tools() if t["function"]["name"] not in taken]
    return client_tools + mine, {t["function"]["name"] for t in mine}


def augment_messages(messages: list[dict], status: str = "") -> list[dict]:
    """Append our block to the client's system message, or add one.

    The status goes here too, not at the end of the conversation: this model's
    chat template raises "System message must be at the beginning" outright, so
    a trailing system message is a hard 500 rather than a stylistic choice.

    Cache cost is avoided by saying nothing when there is nothing to act on.
    A healthy index produces an empty status, so the prefix stays
    byte-identical turn to turn; only the transient states -- missing, or
    building -- add a line and cost one prefill.
    """
    block = CAPABILITY_BLOCK + ("\n\n" + status if status else "")
    out = [dict(m) for m in messages]
    for m in out:
        if m.get("role") == "system" and isinstance(m.get("content"), str):
            m["content"] = m["content"] + block
            return out
    # No system message: add one rather than prepending to the user's turn,
    # which would put our text in their words.
    return [{"role": "system", "content": block.strip()}] + out


def is_first_turn(messages: list[dict]) -> bool:
    return not any(m.get("role") == "assistant" for m in messages)


def preamble_for(info: dict | None, checks: list[str]) -> str:
    if not info:
        return ("`yamadori` · no repository detected in this conversation · "
                "code tools available, retrieval limited\n\n")
    name = os.path.basename(info["root"])
    if info["building"]:
        state = "indexing now, search will be thin this turn"
    elif info["chunks"]:
        state = f"{info['chunks']:,} chunks indexed"
        if info.get("stale"):
            state += ", predates current commit"
    else:
        state = "not indexed"
    bits = [f"`yamadori` · **{name}** · {state}"]
    if checks:
        bits.append("checks: " + ", ".join(checks))
    return " · ".join(bits) + "\n\n"


def available_checks(root: str) -> list[str]:
    pkg = os.path.join(root, "package.json")
    found = []
    try:
        with open(pkg, encoding="utf-8") as f:
            scripts = (json.load(f).get("scripts") or {})
        for want in ("lint", "test", "typecheck", "build"):
            if want in scripts:
                found.append(want)
    except (OSError, json.JSONDecodeError):
        pass
    if os.path.exists(os.path.join(root, "Cargo.toml")):
        found.append("cargo test")
    return found


def run_our_tool(name: str, args: dict, db: str | None) -> str:
    prev = os.environ.get("CODE_INDEX_DB")
    if db:
        os.environ["CODE_INDEX_DB"] = db
        cs.INDEX_DB = db
    try:
        resp = cs.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                          "params": {"name": name, "arguments": args}})
        return resp["result"]["content"][0]["text"]
    except Exception as e:                                       # noqa: BLE001
        return f"{name} failed: {type(e).__name__}: {e}"
    finally:
        if db and prev:
            os.environ["CODE_INDEX_DB"] = prev
            cs.INDEX_DB = prev


def prepare(body: dict) -> dict:
    """Resolve repo, tools and system block without calling the model.

    Split out so an experiment can fan the SAME resolved request out several
    ways; otherwise a comparison measures prompt differences rather than the
    thing being tested.
    """
    messages = body.get("messages") or []
    root, _ = detect.detect_repo(messages)
    trusted_root, _ = detect.detect_repo(messages, trusted_only=True)
    info = (repos.ensure(root, from_trusted=(root == trusted_root))
            if root else None)
    status = ""
    if info and (info["building"] or not info["chunks"]):
        status = repos.status_line(info)
    tools, _injected = merge_tools(body.get("tools"))
    out = dict(body)
    out["messages"] = augment_messages(messages, status)
    out["tools"] = tools
    out.pop("stream", None)
    return out


def complete(body: dict) -> dict:
    messages = body.get("messages") or []
    root, _ev = detect.detect_repo(messages)
    # A NEW repository may only be established from harness- or user-supplied
    # text. Establishing one from a tool result would let any repository name
    # a directory and have its contents read and summarised back.
    trusted_root, _ = detect.detect_repo(messages, trusted_only=True)
    info = (repos.ensure(root, from_trusted=(root == trusted_root))
            if root else None)
    if info and info.get("blocked"):
        print(f"  refused new root {root}: {info['blocked']}", flush=True)
    db = repos.db_path(root) if root else None

    # Only actionable states are worth a line. A healthy index says nothing,
    # which keeps the cached prefix stable.
    status = ""
    if info and (info["building"] or not info["chunks"]):
        status = repos.status_line(info)

    tools, injected = merge_tools(body.get("tools"))
    payload = dict(body)
    payload["messages"] = augment_messages(messages, status)
    payload["tools"] = tools
    payload.pop("stream", None)

    # Every turn is logged as raw events, never as scores. This is the corpus
    # that later tunes the decision model; see corpus.py for why nothing is
    # labelled online.
    turn = corpus.new_turn()
    t_start = time.time()
    corpus.log_turn(turn, root, messages,
                    [t.get("function", {}).get("name") for t in tools],
                    is_first_turn(messages))

    convo = payload["messages"]
    for hop in range(MAX_TOOL_HOPS):
        d = _post("/v1/chat/completions", payload)
        msg = d["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        ours = [c for c in calls
                if c.get("function", {}).get("name") in injected]
        if not ours:
            # Either a final answer, or calls belonging to the client. Either
            # way it is the client's turn -- hand it back untouched.
            # A thinking model may return an empty `content` with the text in
            # `reasoning_content` when it runs out of budget mid-thought.
            # Returning that as a blank answer looks like the model failed.
            if not (msg.get("content") or "").strip() and msg.get("reasoning_content"):
                msg["content"] = ("[truncated while reasoning; raise max_tokens]\n\n"
                                  + msg["reasoning_content"])[:8000]
            corpus.log_answer(turn, root, msg.get("content") or "", hop,
                              (time.time() - t_start) * 1000)
            return d
        convo.append({"role": "assistant", "content": msg.get("content") or "",
                      "tool_calls": calls})
        for c in ours:
            fn = c["function"]["name"]
            try:
                args = json.loads(c["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            corpus.log_tool_call(turn, root, fn, args, hop)
            t0 = time.time()
            out = run_our_tool(fn, args, db)
            corpus.log_tool_result(turn, root, fn, out,
                                   (time.time() - t0) * 1000)
            convo.append({"role": "tool", "tool_call_id": c["id"],
                          "content": out[:6000]})
        payload["messages"] = convo
    return _post("/v1/chat/completions", payload)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):                                            # noqa: N802
        if self.path.rstrip("/") == "/v1/models":
            try:
                req = urllib.request.Request(f"{UPSTREAM}/v1/models")
                with urllib.request.urlopen(req, timeout=30) as r:
                    return self._send(200, json.load(r))
            except Exception as e:                               # noqa: BLE001
                return self._send(502, {"error": str(e)})
        if self.path.rstrip("/") == "/health":
            return self._send(200, {"ok": True, "repos": len(repos.known()),
                                    "corpus": corpus.stats()})
        self._send(404, {"error": "not found"})

    def do_POST(self):                                           # noqa: N802
        if self.path.rstrip("/") != "/v1/chat/completions":
            return self._send(404, {"error": "not found"})

        # The API key a client already sends is the account. Nothing extra to
        # configure: every OpenAI client has the field and already fills it.
        account, why = accounts.identify(self.headers.get("Authorization"))
        if account is None:
            return self._send(401, {"error": {
                "message": f"{why}. Set an API key in your client.",
                "type": "invalid_request_error", "code": "invalid_api_key"}})
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, json.JSONDecodeError) as e:
            return self._send(400, {"error": f"bad request: {e}"})

        if body.get("stream"):
            return self._stream(body)

        t0 = time.time()
        try:
            d = complete(body)
        except Exception as e:                                   # noqa: BLE001
            return self._send(502, {"error": f"{type(e).__name__}: {e}"})

        if PREAMBLE and is_first_turn(body.get("messages") or []):
            root, _ = detect.detect_repo(body.get("messages") or [])
            info = repos.ensure(root) if root else None
            note = preamble_for(info, available_checks(root) if root else [])
            try:
                m = d["choices"][0]["message"]
                m["content"] = note + (m.get("content") or "")
            except (KeyError, IndexError, TypeError):
                pass

        print(f"{self.path} {time.time() - t0:.1f}s", flush=True)
        self._send(200, d)

    def _stream(self, body: dict) -> None:
        """Run the tool loop while streaming, so a long turn is never silent."""
        cid = streaming.new_id()
        model = body.get("model", "bonsai-agent")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        def emit(b: bytes) -> bool:
            try:
                self.wfile.write(b)
                self.wfile.flush()
                return True
            except (BrokenPipeError, ConnectionResetError, OSError):
                return False   # client hung up; stop doing work for it

        messages = body.get("messages") or []
        root, _ = detect.detect_repo(messages)
        trusted, _ = detect.detect_repo(messages, trusted_only=True)
        info = repos.ensure(root, from_trusted=(root == trusted)) if root else None
        db = repos.db_path(root) if root else None

        if PREAMBLE and is_first_turn(messages):
            note = preamble_for(info, available_checks(root) if root else [])
            if not emit(streaming.text_chunk(cid, model, note)):
                return

        payload = prepare(body)
        injected = {t["function"]["name"] for t in (payload.get("tools") or [])
                    if t.get("function", {}).get("name") in OUR_NAMES}
        turn = corpus.new_turn()
        t_start = time.time()
        corpus.log_turn(turn, root, messages,
                        [t.get("function", {}).get("name")
                         for t in (payload.get("tools") or [])],
                        is_first_turn(messages))

        convo = payload["messages"]
        for hop in range(MAX_TOOL_HOPS):
            # Tool hops are resolved non-streamed: the answer is only known to
            # be final once the model stops asking for tools, and a token
            # streamed from a hop that turns out to be a tool call would have
            # to be retracted.
            try:
                d = _post("/v1/chat/completions", payload)
            except Exception as e:                               # noqa: BLE001
                emit(streaming.text_chunk(cid, model, f"\n[upstream error: {e}]"))
                emit(streaming.DONE)
                return
            msg = d["choices"][0]["message"]
            calls = [c for c in (msg.get("tool_calls") or [])
                     if c.get("function", {}).get("name") in injected]
            if not calls:
                break
            convo.append({"role": "assistant", "content": msg.get("content") or "",
                          "tool_calls": msg.get("tool_calls") or []})
            for c in calls:
                fn = c["function"]["name"]
                try:
                    args = json.loads(c["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                if not emit(streaming.text_chunk(
                        cid, model, f"`{streaming.describe_call(fn, args)}`\n")):
                    return
                corpus.log_tool_call(turn, root, fn, args, hop)
                t0 = time.time()
                out = run_our_tool(fn, args, db)
                corpus.log_tool_result(turn, root, fn, out,
                                       (time.time() - t0) * 1000)
                convo.append({"role": "tool", "tool_call_id": c["id"],
                              "content": out[:6000]})
            payload["messages"] = convo

        # No tools pending: this response is the answer, so stream it for real.
        final = dict(payload)
        answer = []
        try:
            for b in streaming.stream_upstream(UPSTREAM, final, cid, model):
                try:
                    j = json.loads(b[6:].decode())
                    answer.append((j["choices"][0]["delta"] or {}).get("content") or "")
                except Exception:                                # noqa: BLE001
                    pass
                if not emit(b):
                    return
        except Exception as e:                                   # noqa: BLE001
            emit(streaming.text_chunk(cid, model, f"\n[stream error: {e}]"))

        corpus.log_answer(turn, root, "".join(answer), MAX_TOOL_HOPS,
                          (time.time() - t_start) * 1000)
        emit(streaming.chunk(cid, model, {}, finish="stop"))
        emit(streaming.DONE)

    def log_message(self, *a):                                   # noqa: D102
        pass


def main() -> None:
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"yamadori proxy on :{PORT} -> {UPSTREAM}", flush=True)
    print("point any OpenAI client here; it needs no tool configuration.",
          flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
