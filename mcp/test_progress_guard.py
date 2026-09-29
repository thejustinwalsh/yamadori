#!/usr/bin/env python
"""The no-progress guard (mcp/progress_guard.py), asserted directly on
Hermes-, Pi- and Codex-shaped messages. No GPU, no network, no proxy.

    python mcp/test_progress_guard.py      -> "N/M checks passed"

WHAT THIS GATES (Atomic Agent's ToolLoopTracker, ported; the study and our
own counts are in docs/research/ATOMIC-AGENT.md):

  1. IDENTITY. A call is its tool plus its canonical arguments (key order
     does not matter, a value does). A result is its text, with Atomic's
     volatile JSON keys and Codex's "Chunk ID"/"Wall time" lines dropped.
  2. THE THRESHOLDS, as Atomic applies them: the 4th identical call warns
     (3 before it), the 6th vetoes when the 5 before it returned the same
     result, a changed result ends the streak, other calls in between do
     not (interleaving tolerated), a veto is not counted in the streak, and
     BREAKER_VETOES vetoes in a row give the breaker.
  3. SCOPE. The task opens after the last user turn; WINDOW_CALLS bounds it.
  4. THE WARNING rides on the result of the call it is about, once per
     bucket of WARN_BUCKET; the outcome repeat counts one result coming
     back whatever the arguments, and a successful write resets it.
  5. WORDING. What the model reads: the situation, retryable as a fact,
     the next step -- no prohibition, no capitals shouting, no doubt.
  6. THE RECORD carries no arguments and no result text.
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

_TMP = tempfile.mkdtemp(prefix="yamadori_test_progress_guard_")
os.environ["YAMADORI_NEBARI_DB"] = os.path.join(_TMP, "nebari.sqlite3")
os.environ["YAMADORI_CORPUS_DB"] = os.path.join(_TMP, "corpus.sqlite3")

import progress_guard as pg  # noqa: E402

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


class Convo:
    """Chat messages, one call and result at a time."""

    def __init__(self, prompt: str = "build it") -> None:
        self.msgs: list[dict] = [{"role": "system", "content": "sys"},
                                 {"role": "user", "content": prompt}]
        self.n = 0

    def call(self, name: str, args, result: str | None = None) -> str:
        self.n += 1
        cid = f"call_{self.n}"
        self.msgs.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": cid, "type": "function",
             "function": {"name": name,
                          "arguments": args if isinstance(args, str)
                          else json.dumps(args)}}]})
        if result is not None:
            self.msgs.append({"role": "tool", "tool_call_id": cid,
                              "content": result})
        return cid

    def batch(self, calls: list[tuple[str, dict, str]]) -> list[str]:
        ids = []
        tcs = []
        for name, args, _ in calls:
            self.n += 1
            ids.append(f"call_{self.n}")
            tcs.append({"id": ids[-1], "type": "function",
                        "function": {"name": name,
                                     "arguments": json.dumps(args)}})
        self.msgs.append({"role": "assistant", "content": "",
                          "tool_calls": tcs})
        for cid, (_, _, res) in zip(ids, calls):
            self.msgs.append({"role": "tool", "tool_call_id": cid,
                              "content": res})
        return ids

    def user(self, text: str = "next") -> None:
        self.msgs.append({"role": "user", "content": text})


def hermes(output: str, code: int = 0) -> str:
    return json.dumps({"output": output, "exit_code": code, "error": None})


TSC = {"command": "cd /work/pagoda && npx tsc --noEmit 2>&1 | head -40"}


# ------------------------------------------------------------------ tests --
def test_identity():
    a = pg.call_key("terminal", {"command": "ls", "timeout": 30})
    b = pg.call_key("terminal", '{"timeout": 30, "command": "ls"}')
    c = pg.call_key("terminal", {"command": "ls ", "timeout": 30})
    check(a == b, "key order does not change a call's identity")
    check(a != c, "a changed value is a different call (Atomic: exact)")
    check(pg.call_key("bash", {"command": "ls"}) !=
          pg.call_key("terminal", {"command": "ls"}),
          "the tool name is part of the identity")
    j1 = json.dumps({"ok": True, "timestamp": 1, "requestId": "a", "n": 3})
    j2 = json.dumps({"ok": True, "timestamp": 9, "requestId": "b", "n": 3})
    j3 = json.dumps({"ok": True, "timestamp": 9, "requestId": "b", "n": 4})
    check(pg.result_key(j1) == pg.result_key(j2),
          "Atomic's volatile keys (timestamp, requestId) are not the answer")
    check(pg.result_key(j2) != pg.result_key(j3), "a real change is a change")
    x1 = ("Chunk ID: 87eb5c\nWall time: 0.1 seconds\nProcess exited with "
          "code 1\nOutput:\nboom\n")
    x2 = ("Chunk ID: 11aa22\nWall time: 2.7 seconds\nProcess exited with "
          "code 1\nOutput:\nboom\n")
    check(pg.result_key(x1) == pg.result_key(x2),
          "Codex's Chunk ID / Wall time lines are volatile")
    check(pg.result_key("a\nb") != pg.result_key("a\nc"),
          "plain text is compared exactly")


def test_thresholds():
    check((pg.WARN_REPEATS, pg.VETO_STREAK, pg.BREAKER_VETOES,
           pg.WINDOW_CALLS) == (3, 5, 3, 30),
          "Atomic's defaults: warn 3, veto 5, breaker 3, window 30",
          str((pg.WARN_REPEATS, pg.VETO_STREAK, pg.BREAKER_VETOES,
               pg.WINDOW_CALLS)))
    cv = Convo()
    levels = []
    for _ in range(6):
        levels.append(pg.decide(cv.msgs, "terminal", TSC)["level"])
        cv.call("terminal", TSC, hermes("error TS2304", 2))
    check(levels == ["ok", "ok", "ok", "warn", "warn", "veto"],
          "4th identical call warns, 6th vetoes (5 identical results before)",
          str(levels))
    v = pg.decide(cv.msgs, "terminal", TSC)
    check(v["count"] == 6 and v["level"] == "veto",
          "the streak keeps counting the calls that ran", json.dumps(v))
    check(v["target"] == "cd", "the target is the leading command word only",
          str(v["target"]))

    # A changed result ends the streak (progress clears it).
    cv = Convo()
    for i in range(5):
        cv.call("terminal", TSC, hermes(f"error at line {i}", 2))
    v = pg.decide(cv.msgs, "terminal", TSC)
    check(v["level"] == "warn",
          "five identical calls with CHANGING results warn, never veto",
          v["level"])
    cv = Convo()
    for i in range(7):
        cv.call("terminal", TSC, hermes("same" if i != 3 else "other", 2))
    v = pg.decide(cv.msgs, "terminal", TSC)
    check(v["level"] == "warn" and pg.no_progress_streak(
        pg.history(cv.msgs), v["key"])[0] == 3,
          "the streak counts back only to the last change", json.dumps(v))

    # Interleaving tolerated: other calls between repeats do not reset it.
    cv = Convo()
    for i in range(5):
        cv.call("terminal", TSC, hermes("same", 2))
        cv.call("read_file", {"path": f"src/f{i}.ts"}, "content")
    check(pg.decide(cv.msgs, "terminal", TSC)["level"] == "veto",
          "interleaved other calls do not break the streak")


def test_veto_and_breaker():
    cv = Convo()
    for _ in range(5):
        cv.call("bash", TSC, "src/a.ts(1,1): error TS2304\n\n"
                "Command exited with code 2")
    v = pg.decide(cv.msgs, "bash", TSC)
    check(v["level"] == "veto", "Pi-shaped: the 6th identical call vetoes")
    body = pg.veto_result(v)
    # This request's hidden hops: the vetoed call and its NOT EXECUTED result.
    for i in range(pg.BREAKER_VETOES):
        cv.call("bash", TSC, body)
        v2 = pg.decide(cv.msgs, "bash", TSC)
        want = "breaker" if i + 1 >= pg.BREAKER_VETOES else "veto"
        check(v2["level"] == want,
              f"after {i + 1} veto(es) in a row the next is {want}",
              v2["level"])
    streak, _ = pg.no_progress_streak(pg.history(cv.msgs), v["key"])
    check(streak == 5, "vetoes are not counted in the streak (it plateaus)",
          str(streak))
    # THE BREAKER NEVER ENDS THE HARNESS'S TURN (operator, 2026-09-29): its
    # action is the veto's, and its result says the call was written again.
    check(v2["action"] == "veto" and v["action"] == "veto",
          "breaker and veto share one action: the NOT EXECUTED result",
          json.dumps(v2))
    stop, _ = pg.gate(cv.msgs, [{"id": "z", "type": "function", "function": {
        "name": "bash", "arguments": json.dumps(TSC)}}])
    back = pg.hand_back([{"id": "z"}], stop)
    check(stop["level"] == "breaker" and back[0]["content"].startswith(
        pg.VETO_HEAD) and "5 times" in back[0]["content"]
          and f"written {pg.BREAKER_VETOES} more times since then and not run"
          in back[0]["content"],
          "the breaker's hand-back is a NOT EXECUTED result naming the calls "
          "that ran and the ones that were not", back[0]["content"])
    check(pg.record(stop, "vetoed")["delivered"] == "vetoed",
          "the breaker is recorded as vetoed, never landed")
    # A different call's real outcome between vetoes resets the breaker.
    cv2 = Convo()
    for _ in range(5):
        cv2.call("bash", TSC, "same")
    cv2.call("bash", TSC, body)
    cv2.call("bash", TSC, body)
    cv2.call("read", {"path": "src/a.ts"}, "x")
    check(pg.vetoes_in_a_row(pg.history(cv2.msgs), v["key"]) == 0,
          "another call's outcome between vetoes resets the veto count")
    # The sibling result (a call written with the vetoed one) is neither.
    cv3 = Convo()
    for _ in range(5):
        cv3.call("bash", TSC, "same")
    cv3.batch([("bash", TSC, body),
               ("read", {"path": "a"}, pg.other_result(v))])
    h = pg.history(cv3.msgs)
    check(h[-1].get("other") and h[-1]["result"] is None and not h[-1]["veto"],
          "a not-run sibling is neither an outcome nor a veto")
    check(pg.vetoes_in_a_row(h, v["key"]) == 1,
          "a not-run sibling does not interrupt the veto count")


def test_scope():
    cv = Convo()
    for _ in range(5):
        cv.call("terminal", TSC, hermes("same", 2))
    cv.user("a new request")
    check(pg.decide(cv.msgs, "terminal", TSC)["level"] == "ok",
          "a user turn opens a new task (Atomic: one tracker per turn)")
    cv = Convo()
    for _ in range(5):
        cv.call("terminal", TSC, hermes("same", 2))
    for i in range(pg.WINDOW_CALLS):
        cv.call("read_file", {"path": f"f{i}"}, "x")
    check(pg.decide(cv.msgs, "terminal", TSC)["level"] == "ok",
          "calls more than WINDOW_CALLS back are forgotten")
    cv = Convo()
    cv.call("terminal", TSC)            # a call whose result has not come
    check(pg.history(cv.msgs)[-1]["result"] is None,
          "a call with no result yet is pending, not an outcome")


def test_notice_on_result():
    cv = Convo()
    got = []
    for i in range(14):
        # The file changes between reads (an edit elsewhere): no streak.
        cid = cv.call("read", {"path": "pagoda/src/world-data.ts"},
                      f"export const world = {i};\n")
        got.append(pg.notice_for_result(cv.msgs, cid))
    idx = [i + 1 for i, n in enumerate(got) if n]
    check(idx == [4, 14], "the warning goes on the 4th result, then once per "
          "bucket of 10 (the 14th)", str(idx))
    n4 = got[3]
    check(n4["detector"] == "generic_repeat" and n4["count"] == 4,
          "the 4th result's line names the 4th call", json.dumps(n4))
    check("4th time" in n4["text"] and "same result" not in n4["text"],
          "changing results are not called the same", n4["text"])
    # Identical calls with identical results: Atomic's outcome detector
    # speaks first (the 3rd identical result), the repeat warning on the
    # 4th; the 6th call would be vetoed before it runs.
    cv = Convo()
    got = []
    for i in range(5):
        cid = cv.call("read", {"path": "a.ts"}, "same text")
        got.append(pg.notice_for_result(cv.msgs, cid))
    kinds = [n and n["detector"] for n in got]
    check(kinds == [None, None, "outcome_repeat", "generic_repeat", None],
          "identical: outcome on the 3rd result, repeat on the 4th",
          str(kinds))
    check(got[3] and "same result each time" in got[3]["text"],
          "identical results are said to be identical",
          json.dumps(got[3]))
    # A parallel batch: the second identical call sees the first.
    cv = Convo()
    for i in range(2):
        cv.call("read", {"path": "a.ts"}, f"x{i}")
    ids = cv.batch([("read", {"path": "a.ts"}, "x2"),
                    ("read", {"path": "a.ts"}, "x3")])
    check(pg.notice_for_result(cv.msgs, ids[0]) is None and
          pg.notice_for_result(cv.msgs, ids[1]) is not None,
          "in a batch, earlier calls of the same turn count, in order")
    # No notice for a veto's own result.
    cv = Convo()
    for _ in range(5):
        cv.call("bash", TSC, "same")
    cid = cv.call("bash", TSC, pg.veto_result(
        pg.decide(cv.msgs, "bash", TSC)))
    check(pg.notice_for_result(cv.msgs, cid) is None,
          "a veto's own result gets no warning")


def test_outcome_repeat():
    cv = Convo()
    out = []
    for pat in ("hit\\(", "hit\\(.*\\)", "\\bhit\\("):
        cid = cv.call("search_files", {"pattern": pat, "path": "js/"},
                      json.dumps({"matches": []}))
        out.append(pg.notice_for_result(cv.msgs, cid))
    check(out[:2] == [None, None] and out[2]
          and out[2]["detector"] == "outcome_repeat" and out[2]["count"] == 3,
          "the same result a 3rd time, whatever the arguments, warns",
          json.dumps(out))
    cv = Convo()
    for i, pat in enumerate(("a", "b")):
        cv.call("search_files", {"pattern": pat}, json.dumps({"matches": []}))
    cv.call("write_file", {"path": "js/game.js", "content": "x"},
            json.dumps({"bytes_written": 1}))
    cid = cv.call("search_files", {"pattern": "c"},
                  json.dumps({"matches": []}))
    check(pg.notice_for_result(cv.msgs, cid) is None,
          "a successful write resets the outcome count")
    cv = Convo()
    for pat in ("a", "b"):
        cv.call("search_files", {"pattern": pat}, json.dumps({"matches": []}))
    cv.call("write_file", {"path": "js/game.js", "content": "x"},
            json.dumps({"success": False, "error": "denied"}))
    cid = cv.call("search_files", {"pattern": "c"},
                  json.dumps({"matches": []}))
    n = pg.notice_for_result(cv.msgs, cid)
    check(n and n["count"] == 3,
          "a FAILED write (structured) does not reset it", json.dumps(n))


def test_failed_is_structured():
    check(pg.failed(hermes("x", 1)) is True and pg.failed(hermes("x")) is
          False, "Hermes' exit_code decides")
    check(pg.failed(json.dumps({"success": False})) is True,
          "an ok/success flag decides")
    check(pg.failed("Command exited with code 1") is None,
          "plain text gives no failure signal (AGENTS.md)")


_FORBIDDEN = re.compile(r"\b(?:don'?t|do not|never|must not|critical|stop|"
                        r"blocked|may|might|perhaps|verify|double-check)\b",
                        re.I)


def test_wording():
    v = {"tool": "terminal", "count": 5, "target": "npx", "level": "veto"}
    texts = {"veto": pg.veto_result(v), "other": pg.other_result(v),
             "notice": pg.notice_text("read", 4, same=True, target=None),
             "outcome": pg.outcome_text("search_files", 3)}
    for k, t in texts.items():
        # THE ONE PROHIBITION (operator, 2026-09-29): the warning line says
        # "Do not repeat this identical call", then what to do instead.
        allowed = "Do not repeat this identical call" if k == "notice" else ""
        bad = _FORBIDDEN.findall(t.replace(allowed, "") if allowed else t)
        check(not bad, f"{k}: no other prohibition, no shouting, no doubt",
              str(bad))
        check(not re.search(r"\b[A-Z]{4,}\b", t.replace("NOT EXECUTED", "")),
              f"{k}: no capitals shouting")
        check("Next" in t or "Retryable" in t or "Instead" in t,
              f"{k}: carries the next step", t)
    n = texts["notice"]
    check(n.count("Do not repeat this identical call.") == 1 and
          n.index("Do not repeat") < n.index("Instead,"),
          "the warning says, once and plainly, not to repeat the identical "
          "call, followed by what to do instead", n)
    check(len(re.findall(r"\b(?:do not|don'?t|never)\b", n, re.I)) == 1,
          "it stays the one prohibition", n)
    check(texts["veto"].startswith(pg.VETO_HEAD) and
          "Retryable: no" in texts["veto"],
          "the veto opens with its marker and says retryable as a fact")
    check(pg.history([{"role": "user", "content": "x"},
                      {"role": "assistant", "tool_calls": [
                          {"id": "a", "function": {"name": "t",
                                                   "arguments": "{}"}}]},
                      {"role": "tool", "tool_call_id": "a",
                       "content": texts["veto"]}])[0]["veto"],
          "history reads the veto back by its marker")


def test_gate_and_hand_back():
    cv = Convo()
    for _ in range(5):
        cv.call("terminal", TSC, hermes("same", 2))
    calls = [{"id": "g1", "type": "function", "function": {
                "name": "read_file", "arguments": '{"path": "a.ts"}'}},
             {"id": "g2", "type": "function", "function": {
                 "name": "terminal", "arguments": json.dumps(TSC)}},
             {"id": "g3", "type": "function", "function": {
                 "name": "yama_generate_image", "arguments": "{}"}}]
    stop, vs = pg.gate(cv.msgs, calls,
                       is_client=lambda n: not n.startswith("yama_"))
    check(stop and stop["call_id"] == "g2" and stop["level"] == "veto",
          "the gate stops the looping call of a generation", json.dumps(stop))
    check([v["call_id"] for v in vs] == ["g1", "g2"],
          "our own tools are not gated")
    back = pg.hand_back(calls[:2], stop)
    check(back[1]["content"].startswith(pg.VETO_HEAD) and
          back[0]["content"].startswith(pg.OTHER_HEAD) and
          [b["tool_call_id"] for b in back] == ["g1", "g2"],
          "every call of the stopped generation gets a result, the vetoed "
          "one the veto")
    # A duplicate inside one generation is seen by the gate (Atomic's
    # synchronous gate): three prior + two in the batch -> the 2nd warns.
    cv = Convo()
    for i in range(2):
        cv.call("read", {"path": "a.ts"}, f"v{i}")
    dup = [{"id": f"d{i}", "type": "function", "function": {
        "name": "read", "arguments": '{"path": "a.ts"}'}} for i in range(2)]
    _, vs = pg.gate(cv.msgs, dup)
    check([v["level"] for v in vs] == ["ok", "warn"],
          "a duplicate inside the generation counts", str(vs))
    # Warnings for the results a request ends on.
    cv = Convo()
    for i in range(3):
        cv.call("read", {"path": "a.ts"}, f"v{i}")
    cv.batch([("read", {"path": "a.ts"}, "v3"),
              ("read", {"path": "b.ts"}, "w")])
    ns = pg.notices_for_trailing(cv.msgs)
    check(len(ns) == 1 and ns[0]["count"] == 4,
          "the trailing results' warnings, in order", json.dumps(ns))


def test_record():
    cv = Convo()
    for _ in range(5):
        cv.call("terminal", {"command": "cat secret.env"}, hermes("K=1"))
    v = pg.decide(cv.msgs, "terminal", {"command": "cat secret.env"})
    r = pg.record(v, "vetoed")
    blob = json.dumps(r)
    check("secret.env" not in blob and "K=1" not in blob,
          "the record carries no arguments and no result text", blob)
    check(r["delivered"] == "vetoed" and r["thresholds"]["veto"] == 5,
          "the record names the delivery and the thresholds in force")


def main() -> int:
    for fn in (test_identity, test_thresholds, test_veto_and_breaker,
               test_scope, test_notice_on_result, test_outcome_repeat,
               test_failed_is_structured, test_wording, test_gate_and_hand_back,
               test_record):
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
    passed = sum(1 for ok, _, _ in _results if ok)
    total = len(_results)
    print(f"\n{'=' * 70}\n  {passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
