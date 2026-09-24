#!/usr/bin/env python
"""Fan-out DELIVERY: the selected candidate reaches the client. No GPU.

    python mcp/test_fanout_delivery.py      -> "N/M checks passed"

Operator decision, 2026-09-23: the best fan-out output is delivered, not
only recorded. The rule under test (proxy._fan_out / _delivers_winner):

  1. The ORIGINAL answer is candidate A; the second brain writes B, in
     sequence (fanout.py, 2026-09-23), and the selection record
     (fanout.selection_record) rides on x_yamadori.fanout.
  2. Blocking path: when a code answer's winner (the grade after two,
     `code_grade`, or the medoid after a tie-breaker, `code_medoid`) is not
     the original, the winner's content REPLACES the answer;
     x_yamadori.fanout.replaced is true and d["_fanout"]["original"] keeps
     the original for audit.
  3. Streamed path: the original was already sent, so the winner's code
     follows it, after the fold-back line ("Compared two approaches");
     x_yamadori.fanout.appended is true.
  4. A prose answer (path_consensus) keeps the original on both paths.
     Two parsing candidates that agree keep the original too.
  5. Nothing a candidate contains is executed: a candidate whose code would
     create a sentinel file is selected, delivered, and the file never
     appears.

The main answer AND the second brain's candidates come from one fake
upstream (a real http.server, so the proxy's own SSE reader runs), answered
in call order: the original first, then B.
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

_TMP = tempfile.mkdtemp(prefix="yamadori_test_fanout_delivery_")
os.environ["YAMADORI_CORPUS_DB"] = os.path.join(_TMP, "corpus.sqlite3")
os.environ["YAMADORI_NEBARI_DB"] = os.path.join(_TMP, "nebari.sqlite3")
os.environ["RINGS_DB"] = os.path.join(_TMP, "rings.sqlite3")
os.environ["YAMADORI_PKG_DIR"] = os.path.join(_TMP, "pkgs")
os.makedirs(os.environ["YAMADORI_PKG_DIR"], exist_ok=True)
os.environ["CODE_INDEX_DB"] = os.path.join(_TMP, "code.sqlite3")
os.environ.pop("YAMADORI_IMAGEGEN_URL", None)
os.environ["CONCEPT_SEED_LAST"] = os.path.join(_TMP, "seed_last.json")

import proxy  # noqa: E402

proxy.PREAMBLE = False
_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


# No concept-seed matrix here (mcp/test_fanout.py tests seeds): a fixed
# pair, drawn in turn -- B's job first, then C's. The ledger records each
# per (request, job), so every later run of the same question replays them.
_WORDS = iter(["lantern", "harbour"] * 50)
proxy._draw_seed = lambda prompt=None: {"word": next(_WORDS), "token_id": 1,
                                        "u32": 1}

# --- the main answer: a fake upstream ----------------------------------------
_script: list[str] = []
_seen: list[dict] = []


def _sse(content: str) -> bytes:
    def ev(delta=None, finish=None):
        return b"data: " + json.dumps({
            "id": "u", "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": delta or {},
                         "finish_reason": finish}]}).encode() + b"\n\n"
    out = [ev({"role": "assistant"})]
    out += [ev({"content": content[i:i + 11]})
            for i in range(0, len(content), 11)]
    out.append(ev({}, "stop"))
    out.append(b"data: " + json.dumps({"id": "u", "choices": [], "usage": {
        "prompt_tokens": 50, "completion_tokens": 5,
        "total_tokens": 55}}).encode() + b"\n\n")
    return b"".join(out) + b"data: [DONE]\n\n"


class _Up(BaseHTTPRequestHandler):
    def do_POST(self):                                           # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path.startswith("/upstream/"):
            # proxy._warm: a render, then a zero-token /completion.
            data = json.dumps({"prompt": "".join(
                json.dumps(m) + "<|im_end|>" for m in body.get("messages")
                or [])} if self.path.endswith("apply-template")
                else {}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        _seen.append(body)
        if not _script:
            self.send_response(500)
            self.end_headers()
            return
        text = _script.pop(0)
        last = (body.get("messages") or [{}])[-1]
        if last.get("role") == "assistant":
            # A prefill: llama-server re-sends the prefilled content first.
            text = (last.get("content") or "") + text
        data = _sse(text)
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
# The second brain's door (mcp/model.py) goes to the same fake: no
# offline test may reach the live model server.
import model as _model  # noqa: E402
_model.UPSTREAM = proxy.UPSTREAM

HEADER = json.dumps({"retrieval": False, "hints": False, "investigate": False,
                     "fanout": 3, "effort": "medium"})
QUESTION = [{"role": "user", "content": "Write add_one(x) in Python."}]

SENTINEL = os.path.join(_TMP, "SENTINEL_fanout")
# The original does not parse; B does.
ORIGINAL_BROKEN = "Here:\n```python\ndef add_one(x:\n    return x + 1\n```"
V_DIRECT = ("```python\ndef add_one(x):\n    return x + 1\n```")
V_EVIDENCE = ("```python\ndef add_one(value):\n    return value + 1\n```")
V_SKEPTICAL = ("```python\n"
               f"open({SENTINEL!r}, 'w').write('ran')\n"
               "def add_one(n):\n    return n + 1\n```")


def setup(original: str, *helper: str) -> None:
    _script[:] = [original, *helper]
    _seen.clear()


def run_complete() -> dict:
    return proxy.complete({"model": "yamadori", "_client_ip": "127.0.0.1",
                           "_features": HEADER, "messages": list(QUESTION)})


def run_stream() -> tuple[str, dict]:
    events = []
    for b in proxy.stream_body({"model": "yamadori", "stream": True,
                                "_client_ip": "127.0.0.1",
                                "_features": HEADER,
                                "messages": list(QUESTION)}, "yamadori"):
        if b.strip() == b"data: [DONE]":
            continue
        events.append(json.loads(b[6:].decode()))
    content = "".join((e["choices"][0]["delta"].get("content") or "")
                      for e in events if e.get("choices"))
    xs = [e.get("x_yamadori") for e in events if e.get("x_yamadori")]
    return content, (xs[-1] if xs else {})


# ---------------------------------------------------------------------------
def test_code_winner_replaces_on_the_blocking_path():
    setup(ORIGINAL_BROKEN, V_SKEPTICAL)
    d = run_complete()
    check(len(_seen) == 2,
          "one main generation and ONE second-brain candidate", str(len(_seen)))
    fan = (d.get("x_yamadori") or {}).get("fanout") or {}
    check(fan.get("selection") == "code_grade" and fan.get("steps") == 2
          and fan.get("stop_reason") == "clear: only one parses",
          "only B parses: the grade after two picks it", json.dumps(fan)[:400])
    check((fan.get("candidates") or [{}])[0].get("variant") == "original"
          and (fan["candidates"][0].get("parses") is False),
          "the original is candidate 0, and its code does not parse",
          json.dumps(fan.get("candidates"))[:400])
    content = d["choices"][0]["message"]["content"]
    check(fan.get("replaced") is True and fan.get("winner") == "direct"
          and content.startswith(V_SKEPTICAL + "\n\nToday I was inspired by "
                                 "lantern. Compared two approaches: clear: "
                                 "only candidate 2's python code parses")
          and content.endswith("The code above is the one selected."),
          "the winner's content REPLACES the answer, then the fold-back line: "
          "the seed, 'Compared two approaches' and why",
          json.dumps({"winner": fan.get("winner"), "content": content})[:400])
    check((d.get("_fanout") or {}).get("original") is None
          and (d.get("_fanout") or {}).get("winner", {}).get("variant")
          == "direct", "the winner is kept in _fanout for audit")
    check(ORIGINAL_BROKEN not in json.dumps(d.get("x_yamadori")),
          "x_yamadori carries no answer text")
    check(not os.path.exists(SENTINEL),
          "a candidate's code is never executed, even when it is delivered")


def test_code_winner_is_appended_on_the_streamed_path():
    setup(ORIGINAL_BROKEN, V_DIRECT)
    content, x = run_stream()
    fan = x.get("fanout") or {}
    check(fan.get("selection") == "code_grade" and fan.get("appended") is True
          and fan.get("replaced") is False,
          "streamed: the winner is appended, not a replacement",
          json.dumps(fan)[:300])
    check(content.startswith(ORIGINAL_BROKEN + "\n\nToday I was inspired by")
          and "Compared two approaches: clear: only candidate 2's" in content
          and "The selected code:" in content
          and content.endswith(V_DIRECT),
          "the original, then the fold-back line, then the winner's code",
          content[-400:])
    check(not os.path.exists(SENTINEL), "and nothing was executed")


def test_prose_keeps_the_original():
    orig = "The pool is sized in src/a.ts."
    setup(orig, "See src/b.ts", " It is re-exported there.")
    d = run_complete()
    fan = (d.get("x_yamadori") or {}).get("fanout") or {}
    check(fan.get("selection") == "path_consensus" and fan.get("replaced") is False,
          "prose: path consensus, no replacement", json.dumps(fan)[:300])
    check(d["choices"][0]["message"]["content"].startswith(orig),
          "the original answer is delivered, first",
          d["choices"][0]["message"]["content"][:200])
    setup(orig, "See src/b.ts", " It is re-exported there.")
    content, x = run_stream()
    check(content.startswith(orig)
          and (x.get("fanout") or {}).get("appended") is False,
          "streamed prose: the original first, nothing swapped in", content[:200])


def test_an_original_that_wins_is_left_alone():
    good = "```python\ndef add_one(x):\n    return x + 1\n```"
    setup(good, V_EVIDENCE)
    d = run_complete()
    fan = (d.get("x_yamadori") or {}).get("fanout") or {}
    content = d["choices"][0]["message"]["content"]
    check(fan.get("winner") == "original"
          and fan.get("stop_reason") == "clear: agreement"
          and fan.get("replaced") is False
          and content.startswith(good + "\n\nToday I was inspired by lantern."
                                 " Compared two approaches: clear: both "
                                 "parse and agree"),
          "both parse and agree: the original is delivered, and one line "
          "says the second approach agreed", json.dumps(fan)[:300])
    check(len(_seen) == 2, "and no tie-breaker ran", str(len(_seen)))


# --- THE FOLD-BACK (operator, 2026-09-24) -----------------------------------
# The second brain's work is never thrown away. Prose: B's differing points
# are PREFILLED after main's own answer, in main's own turn, and main
# continues from them -- it weighs them itself. The weigh turn this replaces
# (a user turn in main's context the client never saw) is gone. Code: the
# winner is delivered as before, with one line of what the losers used.
PROSE_A = "The pool is sized in src/a.ts."
PROSE_B = ("The pool is sized in src/a.ts. Alternatively, a lock-free ring "
           "buffer avoids contention entirely under heavy write load.")
B_POINT = ("Alternatively, a lock-free ring buffer avoids contention "
           "entirely under heavy write load.")
CONT = ", and under heavy writes the ring buffer is the better fit."
FOLD = ("\n\nToday I was inspired by lantern. Compared two approaches: a "
        "second answer, written independently, makes these points that mine "
        "does not.\n\n- " + B_POINT + "\n\nWeighing them")
CODE_A = "```python\ndef add_one(x):\n    return x + 1\n```"
CODE_B = "```python\ndef add_one(x):\n    return operator.add(x, 1)\n```"
CODE_C = "```python\ndef add_one(value):\n    return value + 1\n```"


def test_prose_is_weighed_by_main_in_its_own_turn():
    setup(PROSE_A, PROSE_B, CONT)
    d = run_complete()
    fan = (d.get("x_yamadori") or {}).get("fanout") or {}
    hb = fan.get("handback") or {}
    check(len(_seen) == 3, "the original, B, and ONE continuation of main's "
          "own turn", str(len(_seen)))
    cont = _seen[2] if len(_seen) > 2 else {"messages": []}
    msgs = cont["messages"]
    check(msgs[-1].get("role") == "assistant"
          and msgs[-1].get("content") == PROSE_A + FOLD
          and msgs[-2].get("role") == "user"
          and msgs[-2].get("content") == QUESTION[-1]["content"],
          "the continuation's prompt is main's own: the question, then its "
          "answer plus the fold-back as a prefill -- no weigh turn",
          json.dumps(msgs[-2:])[:500])
    check(cont.get("tools") == _seen[0].get("tools")
          and cont.get("reasoning_effort") == _seen[0].get("reasoning_effort"),
          "with main's own tools and effort, so the slot's prefix is reused",
          str(cont.get("tools"))[:100])
    check(d["choices"][0]["message"]["content"] == PROSE_A + FOLD + CONT,
          "delivered: the answer, the fold-back, main's continuation",
          d["choices"][0]["message"]["content"][:300])
    check(hb.get("kind") == "prose" and hb.get("from") == ["B"]
          and hb.get("into") == "continuation" and hb.get("points") == 1,
          "x_yamadori.fanout.handback: {from, into: continuation}",
          json.dumps(hb))
    blob = json.dumps(d.get("x_yamadori"))
    check(B_POINT not in blob and "_handback" not in blob,
          "x_yamadori carries the counts, never the text", blob[:200])
    check(d["usage"].get("hops") == 2,
          "the continuation is banked in usage like any main generation",
          json.dumps(d.get("usage")))
    check(not hasattr(proxy, "_weigh_alternative")
          and not hasattr(proxy, "ALT_VIEW_HEAD"),
          "the weigh turn and its head are gone")


def test_a_continuation_that_fails_falls_back_to_the_note():
    setup(PROSE_A, PROSE_B)          # nothing scripted for the continuation
    d = run_complete()
    content = d["choices"][0]["message"]["content"]
    hb = ((d.get("x_yamadori") or {}).get("fanout") or {}).get("handback") or {}
    check(content == PROSE_A + FOLD[:-len("\n\nWeighing them")]
          and hb.get("into") == "note",
          "the original is delivered with B's points as a note", content[-300:])


def test_streamed_prose_continues_the_same_way():
    setup(PROSE_A, PROSE_B, CONT)
    content, x = run_stream()
    hb = (x.get("fanout") or {}).get("handback") or {}
    check(len(_seen) == 3 and content == PROSE_A + FOLD + CONT,
          "streamed: the original, the fold-back, the continuation -- the "
          "re-sent prefill is never streamed twice", content)
    check(hb.get("into") == "continuation" and hb.get("from") == ["B"],
          "x_yamadori.fanout.handback on the final chunk", json.dumps(hb))


def test_prose_that_does_not_differ_hands_back_nothing():
    setup(PROSE_A, PROSE_A)
    d = run_complete()
    fan = (d.get("x_yamadori") or {}).get("fanout") or {}
    check(len(_seen) == 2 and fan.get("handback") is None
          and d["choices"][0]["message"]["content"] == PROSE_A,
          "B says nothing new: no continuation, no note, handback null",
          json.dumps(fan)[:300])


def test_code_still_delivers_the_winner_and_notes_the_losers():
    setup(CODE_A, CODE_B, CODE_C)
    d = run_complete()
    fan = (d.get("x_yamadori") or {}).get("fanout") or {}
    hb = fan.get("handback") or {}
    content = d["choices"][0]["message"]["content"]
    check(fan.get("selection") == "code_medoid" and fan.get("winner")
          == "tiebreak" and fan.get("replaced") is True
          and content.startswith(CODE_C),
          "A and B disagree, C breaks the tie and is delivered as before",
          json.dumps({k: fan.get(k) for k in ("selection", "winner",
                                              "stop_reason")}))
    want = ("The other candidates differed: candidate B used `add`, "
            "`operator`. The delivered code uses none of these.")
    check(want in content and "Today I was inspired by lantern and harbour. "
          "Compared two approaches" in content
          and content.count("\n") <= CODE_C.count("\n") + 3,
          "one line: both seeds (B and C ran), the phrase, and what the "
          "loser used that the winner does not", content[len(CODE_C):])
    check(hb.get("kind") == "code" and hb.get("from") == ["B"]
          and hb.get("into") == "note" and hb.get("chars", 0) > 0,
          "x_yamadori.fanout.handback: code, from B, as a note",
          json.dumps(hb))
    setup(CODE_A, CODE_B, CODE_C)
    content, x = run_stream()
    check(content.startswith(CODE_A) and content.index("Compared two "
          "approaches") > len(CODE_A) and content.endswith(CODE_C),
          "streamed: the original, the fold-back line, then the winner's code",
          content)


TESTS = [test_code_winner_replaces_on_the_blocking_path,
         test_code_winner_is_appended_on_the_streamed_path,
         test_prose_keeps_the_original,
         test_an_original_that_wins_is_left_alone,
         test_prose_is_weighed_by_main_in_its_own_turn,
         test_a_continuation_that_fails_falls_back_to_the_note,
         test_streamed_prose_continues_the_same_way,
         test_prose_that_does_not_differ_hands_back_nothing,
         test_code_still_delivers_the_winner_and_notes_the_losers]


def main() -> int:
    for t in TESTS:
        try:
            t()
        except Exception as e:                                   # noqa: BLE001
            check(False, f"{t.__name__} raised",
                  f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
    passed = sum(1 for ok, _, _ in _results if ok)
    for ok, name, detail in _results:
        if not ok:
            print(f"FAIL  {name}\n      {detail[:1500]}")
    print(f"\n{passed}/{len(_results)} checks passed")
    _srv.shutdown()
    return 0 if passed == len(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
