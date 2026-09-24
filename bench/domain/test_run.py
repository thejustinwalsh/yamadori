#!/usr/bin/env python
"""Offline proof that bench/domain/run.py and analyse.py measure what they say.

    python bench/domain/test_run.py

A fake :1234 (a local http.server on a free port) plays the proxy: it returns
canned completions and builds `x_yamadori` from the X-Yamadori-Features header
the way proxy.prepare + selection.decide would -- or, in the lying modes,
contradicts it. Nothing here touches the GPU, the real proxy, or a toolchain;
grading is a stub.

What must hold:
  - every arm sends exactly its header, the same body fields, and the key
  - an x_yamadori that contradicts the arm is a stack_error, never a fail
  - HTTP != 200, finish != stop, an exception, a grader `error` are stack_error
  - a rerun skips done pairs and re-runs stack_error pairs (and re-grades,
    without regenerating, a grader-only error)
  - McNemar and the Bonferroni family on a hand-computed table
  - three_tsl never enters the headline block
  - the key never reaches any file the run writes
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import analyse  # noqa: E402
import run  # noqa: E402

_results: list[tuple[bool, str, str]] = []
KEY = "ym-test-SECRET-do-not-leak-4f1c"


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


# ------------------------------------------------------------ the fake proxy
class Fake:
    mode = "honest"            # honest | lie_hints | http500 | length | lie_fanout
    fail_arms: set = set()     # arms that get `mode`; others are honest
    requests: list = []
    busy_left = 0


def fake_x(feats: dict) -> dict:
    """x_yamadori as the real proxy would build it for this header at tier max."""
    forced = sorted(k for k in ("hints", "investigate", "fanout") if k in feats)
    retrieval = feats.get("retrieval", True)
    hints = feats.get("hints", True)
    inv = feats.get("investigate", False)
    n = feats.get("fanout", 1)
    return {
        "tier": "max", "effort_sent": feats.get("effort"),
        "tools_gate": ({"offer": True, "why": "held: typegpu"} if retrieval else None),
        "hints": [{"recipe": "use prefix sums"}] if hints else [],
        "suppressed_hints": [],
        "selection": {"hints": hints, "investigate": inv, "fanout_n": n,
                      "because": {}, "signals": {
                          "forced": forced,
                          "allowed": {"hints": True, "investigate": True, "fanout": 3}}},
        "fanout": ({"n": n, "asked": n, "seeds": []} if n > 1 else None),
        "investigate": ({"ran": True, "hops": 2, "injected": True} if inv else None),
        "hops": 1,
        "check_code": {"offered": bool(feats.get("check_code", retrieval)),
                       "calls": 0, "results": []},
        "repair": ({"enabled": True, "rounds": 0} if feats.get("repair") else None),
        "budget": {"max_tokens_sent": 4096 + feats.get("reasoning_cap", 32768),
                   "reasoning_budget_tokens": feats.get("reasoning_cap", 32768)},
    }


def _args(code: str) -> str:
    return json.dumps({"code": code})


# prompt keyword -> the model's move per number of tool results received
SCRIPTS = {
    "checky": [("check_solution", _args("export const x: number = 'a';")),
               ("check_solution", _args("export const x = 1; // PASS")),
               ("answer", "```ts\nexport const x = 1; // PASS\n```")],
    "repeaty": [("check_solution", _args("same code")),
                ("check_solution", _args("same code")),
                ("answer", "```ts\nexport const x = 1;\n```")],
    "strangey": [("read_file", _args("x")),
                 ("answer", "```ts\nexport const x = 1; // PASS\n```")],
    "badargs": [("check_solution", "{not json"),
                ("answer", "```ts\nexport const x = 1; // PASS\n```")],
    "direct": [("answer", "```ts\nexport const x = 1; // PASS\n```")],
}


def stub_check(task: dict, code: str) -> dict:
    """public_check's shape. 'PASS' compiles; 'BROKEN_CHECKER' cannot run."""
    if "BROKEN_CHECKER" in task.get("prompt", ""):
        return {"ok": False, "stage": "error", "tool": "tsc", "n_errors": 1,
                "errors": ["node not found"], "truncated": False,
                "checker_error": True, "seconds": 0.0}
    ok = "PASS" in code
    return {"ok": ok, "stage": "compile", "tool": "tsc --strict (stub)",
            "n_errors": 0 if ok else 1,
            "errors": [] if ok else ["solution.ts(1,14): error TS2322: Type "
                                     "'string' is not assignable to type 'number'."],
            "truncated": False, "checker_error": False, "seconds": 0.01}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):            # quiet
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        feats = json.loads(self.headers.get("X-Yamadori-Features") or "{}")
        Fake.requests.append({"path": self.path, "body": body, "feats": feats,
                              "auth": self.headers.get("Authorization"),
                              "session": self.headers.get("X-Yamadori-Session")})
        arm = next((a for a in run.ARMS if run.features(a, feats.get("effort"))
                    == feats), "?")
        mode = Fake.mode if (not Fake.fail_arms or arm in Fake.fail_arms) else "honest"
        if mode == "http500":
            return self._send(500, {"error": "upstream exploded"})
        if mode == "helper_busy" and Fake.busy_left > 0:
            Fake.busy_left -= 1
            x = fake_x(feats)
            x["investigate"] = {"ran": False, "why": "the helper lane is busy "
                                                     "with another request"}
            return self._send(200, {"choices": [{"message": {
                "role": "assistant", "content": "```ts\nexport const x = 1; // PASS\n```"},
                "finish_reason": "stop"}], "usage": {"prompt_tokens": 1,
                                                     "completion_tokens": 1},
                "x_yamadori": x})
        if mode == "busy" and Fake.busy_left > 0:
            Fake.busy_left -= 1
            return self._send(429, {"error": {"code": "server_busy",
                                              "message": "all 2 main lanes busy"}})
        x = fake_x(feats)
        finish = "stop"
        if mode == "lie_hints":
            x["hints"] = [{"recipe": "smuggled"}]
            x["selection"]["hints"] = True
        if mode == "lie_fanout":
            x["fanout"] = {"n": 2, "asked": 3}
        if mode == "length":
            finish = "length"
        prompt = body["messages"][0]["content"]
        content = "```ts\nexport const x = 1; // PASS\n```" if "good" in prompt \
            else "```ts\nexport const x = 1;\n```"
        # A self-check conversation: the model calls the client's tool, by
        # script, according to how many tool results it has been sent.
        script = next((v for k, v in SCRIPTS.items() if k in prompt), None)
        calls = None
        if script is not None and body.get("tools"):
            n_tool = sum(1 for m in body["messages"] if m.get("role") == "tool")
            step = script[min(n_tool, len(script) - 1)]
            if step[0] == "answer":
                content = step[1]
            else:
                name, args = step
                calls = [{"id": f"call_{n_tool}", "type": "function",
                          "function": {"name": name, "arguments": args}}]
                content, finish = "", "tool_calls"
        msg = {"role": "assistant", "content": content, "reasoning_content": "hmm"}
        if calls:
            msg["tool_calls"] = calls
        self._send(200, {"choices": [{"message": msg,
                                      "finish_reason": finish}],
                         "usage": {"prompt_tokens": 100, "completion_tokens": 50,
                                   "total_tokens": 150, "hops": 1},
                         "x_yamadori": x})

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)


def start_fake() -> tuple[ThreadingHTTPServer, str]:
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def stub_grade(task: dict, text: str) -> dict:
    if task.get("grader_error"):
        return {"passed": False, "stage": "error", "detail": "toolchain missing"}
    return {"passed": "PASS" in text, "stage": "test", "detail": "stub"}


def tasks_fixture() -> list[dict]:
    return [
        {"id": "ts01", "domain": "typescript", "prompt": "good task",
         "contaminated": False, "difficulty": "easy"},
        {"id": "ts02", "domain": "typescript", "prompt": "bad task",
         "contaminated": False, "difficulty": "easy"},
        {"id": "tsl01", "domain": "three_tsl", "prompt": "good tsl",
         "contaminated": True, "difficulty": "easy"},
    ]


def cfg_for(url: str, run_dir: str) -> dict:
    return {"run_id": os.path.basename(run_dir), "run_dir": run_dir, "url": url,
            "key": KEY, "effort": "medium", "max_tokens": 4096,
            "temperature": 0.0, "timeout": 30, "quiet": False}


def quiet(*_a, **_k):
    pass


def reset(mode="honest", fail_arms=()):
    Fake.mode, Fake.fail_arms, Fake.requests = mode, set(fail_arms), []


# ------------------------------------------------------------------- tests --
def test_every_arm_sends_its_header(url, tmp):
    reset()
    d = os.path.join(tmp, "headers")
    rows = run.run_pairs(tasks_fixture()[:1], list(run.ARMS), cfg_for(url, d),
                         stub_grade, log=quiet, check_fn=stub_check)
    check(len(Fake.requests) == len(run.ARMS), "one request per arm",
          str(len(Fake.requests)))
    want = {
        "A0": {"retrieval": False, "hints": False, "investigate": False, "fanout": 1},
        "A1": {"retrieval": True, "hints": False, "investigate": False, "fanout": 1},
        "A2": {"retrieval": False, "hints": True, "investigate": False, "fanout": 1},
        "A3": {"retrieval": False, "hints": False, "investigate": True, "fanout": 1},
        "A4": {"retrieval": False, "hints": False, "investigate": False, "fanout": 3},
        "A5": {},
        "A6": {"retrieval": True, "hints": True, "investigate": True, "fanout": 3,
               "check_code": True, "repair": True},
        "C8": {"retrieval": False, "hints": False, "investigate": False, "fanout": 1,
               "reasoning_cap": 8192},
        "C32": {"retrieval": False, "hints": False, "investigate": False, "fanout": 1,
                "reasoning_cap": 32768},
        "C128": {"retrieval": False, "hints": False, "investigate": False,
                 "fanout": 1, "reasoning_cap": 131072},
    }
    for s_arm, one_shot in run.SELF_CHECK.items():
        want[s_arm] = want[one_shot]          # the twin's header, unchanged
    for req, arm in zip(Fake.requests, run.ARMS):
        check(req["feats"] == dict(want[arm], effort="medium"),
              f"{arm}: X-Yamadori-Features is exactly the arm + effort",
              json.dumps(req["feats"]))
        tools = req["body"].get("tools")
        if arm in run.SELF_CHECK:
            check(tools == [run.CHECK_TOOL], f"{arm}: the body carries the "
                  "check_solution client tool and nothing else", json.dumps(tools)[:200])
        else:
            check(tools is None, f"{arm}: a one-shot body carries no tools")
    bodies = {json.dumps({k: r["body"][k] for k in
                          ("model", "reasoning_effort", "max_tokens", "temperature")},
                         sort_keys=True) for r in Fake.requests}
    check(len(bodies) == 1 and '"reasoning_effort": "max"' in bodies.pop(),
          "every arm: the same body fields, reasoning_effort=max, same max_tokens")
    check(all(r["auth"] == f"Bearer {KEY}" for r in Fake.requests),
          "the key is sent as a bearer header")
    sess = [r["session"] for r in Fake.requests]
    check(all(sess) and len(set(sess)) == len(sess),
          "every row gets its own proxy session (no two arms share one)")
    check(all(r["outcome"] == "pass" for r in rows),
          f"an honest x_yamadori verifies clean on all {len(run.ARMS)} arms",
          json.dumps([(r["arm"], r["mismatch"]) for r in rows if r["mismatch"]])[:300])


def test_verify_catches_contradictions():
    x = fake_x(run.features("A0", "medium"))
    ok = run.verify("A0", "medium", x, finish="stop", content="c")
    check(ok == [], "A0 honest record: no mismatch", str(ok))

    x = fake_x(run.features("A0", "medium"))
    x["hints"] = [{"recipe": "r"}]
    check(any("emitted" in m for m in run.verify("A0", "medium", x, finish="stop",
                                                 content="c")),
          "A0 with hints emitted -> mismatch")

    x = fake_x(run.features("A3", "medium"))
    x["investigate"] = {"ran": False, "why": "the helper lane is busy"}
    check(any("did not run" in m for m in run.verify("A3", "medium", x,
                                                     finish="stop", content="c")),
          "A3 whose investigation did not run -> mismatch")

    x = fake_x(run.features("A3", "medium"))
    x["tools_gate"] = {"offer": True}
    check(any("tools_gate ran" in m for m in run.verify("A3", "medium", x,
                                                        finish="stop", content="c")),
          "A3 (retrieval off) with a tool gate decision -> mismatch")

    x = fake_x(run.features("A4", "medium"))
    x["fanout"] = {"n": 2, "asked": 3}
    check(any("delivered 2" in m for m in run.verify("A4", "medium", x,
                                                     finish="stop", content="c")),
          "A4 fan-out 3 delivering 2 -> mismatch")
    # Sequential fan-out (2026-09-23): 3 is the MOST candidates, the original
    # included; stopping at 2 on a clear grade is the arm working.
    x["fanout"] = {"mode": "sequential", "n": 2, "asked": 3, "steps": 2}
    check(run.verify("A4", "medium", x, finish="stop", content="c") == [],
          "A4 sequential fan-out stopping at 2 of up to 3 -> no mismatch")
    x["fanout"] = {"mode": "sequential", "n": 1, "asked": 3,
                   "skipped": "helper busy"}
    check(any("helper busy" in m for m in run.verify(
              "A4", "medium", x, finish="stop", content="c")),
          "A4 sequential fan-out skipped (helper busy) -> mismatch")
    x["fanout"] = None
    check(run.verify("A4", "medium", x, finish="stop", content="") == [],
          "A4 with empty content: no fan-out expected (extract failure, not stack)")

    x = fake_x(run.features("A5", "medium"))
    x["selection"]["signals"]["forced"] = ["hints"]
    check(any("forced" in m for m in run.verify("A5", "medium", x, finish="stop",
                                                content="c")),
          "A5 where selection saw a forced flag -> mismatch")

    x = fake_x(run.features("A1", "medium"))
    x["tier"] = "high"
    check(any("tier" in m for m in run.verify("A1", "medium", x, finish="stop",
                                              content="c")),
          "a tier capped below max -> mismatch")
    x = fake_x(run.features("A1", "medium"))
    x["effort_sent"] = "xhigh"
    check(any("effort_sent" in m for m in run.verify("A1", "medium", x,
                                                     finish="stop", content="c")),
          "an effort other than the header's -> mismatch")
    x = fake_x(run.features("C8", "medium"))
    x["budget"]["reasoning_budget_tokens"] = 32768
    check(any("reasoning_budget_tokens" in m for m in
              run.verify("C8", "medium", x, finish="stop", content="c")),
          "C8 whose record shows the default 32768 cap -> mismatch")
    x = fake_x(run.features("C128", "medium"))
    check(run.verify("C128", "medium", x, finish="stop", content="c") == [],
          "C128 with reasoning_budget_tokens 131072: clean")
    check(run.timeout_for("A0", 30) == 3600 and run.timeout_for("A6", 7200) == 7200
          and run.timeout_for("C128", 3600) >= 131072 / 12,
          "timeouts: never under 3600 s; C128 sized to reach its cap at 12 tok/s",
          str([run.timeout_for(a, 30) for a in ("A0", "C8", "C32", "C128")]))
    check(run.verify("A1", "medium", None, finish="stop", content="c")
          == ["no x_yamadori on the response"], "no x_yamadori -> mismatch")


def test_stack_errors_are_not_failures(url, tmp):
    for mode, kind in (("lie_hints", "mismatch"), ("http500", "http"),
                       ("length", "finish"), ("lie_fanout", "mismatch")):
        reset(mode)
        d = os.path.join(tmp, f"se_{mode}")
        arms = ["A4"] if mode == "lie_fanout" else ["A0"]
        rows = run.run_pairs(tasks_fixture()[:1], arms, cfg_for(url, d),
                             stub_grade, log=quiet)
        r = rows[0]
        check(r["outcome"] == "stack_error" and r["stack_error_kind"] == kind
              and r["grade"] is None,
              f"{mode}: stack_error/{kind}, and never graded",
              json.dumps({k: r.get(k) for k in ("outcome", "stack_error_kind",
                                                "mismatch", "grade")})[:300])
    # a dead endpoint
    d = os.path.join(tmp, "se_dead")
    prev_down, run.DOWN_MAX_S = run.DOWN_MAX_S, 2 * run.BUSY_WAIT_S
    rows = run.run_pairs(tasks_fixture()[:1], ["A0"],
                         dict(cfg_for("http://127.0.0.1:9", d), timeout=3,
                              sleep=lambda s: None),
                         stub_grade, log=quiet)
    check(rows[0]["outcome"] == "stack_error"
          and rows[0]["stack_error_kind"] == "exception",
          "connection refused past DOWN_MAX_S: stack_error/exception",
          rows[0].get("error", ""))
    run.DOWN_MAX_S = prev_down
    calls = []
    prev = run.post

    def flaky(*a, **k):
        calls.append(1)
        if len(calls) < 3:
            import urllib.error
            raise urllib.error.URLError(ConnectionRefusedError(10061, "refused"))
        return prev(*a, **k)
    run.post = flaky
    try:
        reset("honest")
        rows = run.run_pairs(tasks_fixture()[:1], ["A0"],
                             dict(cfg_for(url, os.path.join(tmp, "se_restart")),
                                  sleep=lambda s: None), stub_grade, log=quiet)
    finally:
        run.post = prev
    check(rows[0]["outcome"] == "pass" and rows[0]["busy_429s"] == 2,
          "a proxy restart (connection refused, then back) is waited out",
          json.dumps({k: rows[0].get(k) for k in ("outcome", "busy_429s", "error")}))


def test_resume(url, tmp):
    d = os.path.join(tmp, "resume")
    ts = tasks_fixture()[:2]
    reset("http500", fail_arms=["A1"])
    rows1 = run.run_pairs(ts, ["A0", "A1"], cfg_for(url, d), stub_grade, log=quiet)
    check(len(Fake.requests) == 4, "first pass: 4 requests", str(len(Fake.requests)))
    check(sum(r["outcome"] == "stack_error" for r in rows1) == 2,
          "first pass: both A1 rows are stack_error")
    reset("honest")
    rows2 = run.run_pairs(ts, ["A0", "A1"], cfg_for(url, d), stub_grade, log=quiet)
    sent = [r["feats"].get("retrieval") for r in Fake.requests]
    check(len(Fake.requests) == 2 and all(s is True for s in sent),
          "rerun: only the 2 stack_error A1 pairs are requested again",
          json.dumps(sent))
    check(all(r["attempt"] == 2 for r in rows2),
          "rerun rows are attempt 2", str([r["attempt"] for r in rows2]))
    allrows, _ = run.load_rows(os.path.join(d, "rows.jsonl"))
    check(len(allrows) == 6, "rows.jsonl keeps the stack_error rows (append-only)",
          str(len(allrows)))
    check(len(run.done_pairs(allrows)) == 4, "all 4 pairs now done")
    reset("honest")
    run.run_pairs(ts, ["A0", "A1"], cfg_for(url, d), stub_grade, log=quiet)
    check(len(Fake.requests) == 0, "third pass: nothing left to request")
    check(os.path.exists(os.path.join(d, "answers", "ts01__A1__1.json"))
          and os.path.exists(os.path.join(d, "answers", "ts01__A1__2.json")),
          "each attempt's answer kept in its own answers/ file")

    # done_pairs: a stack_error LAST row is never done, even after a pass
    fake_rows = [{"task": "t", "arm": "A0", "outcome": "pass"},
                 {"task": "t", "arm": "A0", "outcome": "stack_error"}]
    check(run.done_pairs(fake_rows) == set(),
          "a pair whose last row is stack_error is not done")


def test_grader_error_is_regraded_not_regenerated(url, tmp):
    d = os.path.join(tmp, "grader")
    t = dict(tasks_fixture()[0], grader_error=True)
    reset("honest")
    rows = run.run_pairs([t], ["A0"], cfg_for(url, d), stub_grade, log=quiet)
    check(rows[0]["outcome"] == "stack_error"
          and rows[0]["stack_error_kind"] == "grader",
          "grade stage `error` -> stack_error/grader")
    reset("honest")
    t2 = dict(t, grader_error=False)
    rows = run.run_pairs([t2], ["A0"], cfg_for(url, d), stub_grade, log=quiet)
    check(len(Fake.requests) == 0 and rows and rows[0]["outcome"] == "pass"
          and rows[0].get("regraded_from") == 1,
          "rerun re-grades the saved answer without a new generation",
          json.dumps({"requests": len(Fake.requests),
                      "row": {k: rows[0].get(k) for k in ("outcome", "regraded_from")}
                      if rows else None}))


def test_stale_code_rows(url, tmp):
    d = os.path.join(tmp, "stale")
    ts = tasks_fixture()[:2]
    reset("honest")
    run.run_pairs(ts, ["A0", "A5"], cfg_for(url, d), stub_grade, log=quiet)
    n = run.mark_stale(d, "selection symbol lookup fixed")
    rows, _ = run.load_rows(os.path.join(d, "rows.jsonl"))
    check(n == 4 and len(rows) == 4 and all(r.get("stale_code") for r in rows),
          "mark_stale flags every row and deletes none", f"marked {n}, rows {len(rows)}")
    check(run.mark_stale(d, "again") == 0, "marking twice marks nothing new")
    check(run.done_pairs(rows) == set(), "stale rows are not done: the pairs run again")
    s = analyse.analyse(d)
    check(s["rows"] == 0 and s["stale_rows"] == 4
          and any("stale_code" in c for c in s["caveats"]),
          "analyse drops stale rows, counts them, and says why",
          json.dumps({"rows": s["rows"], "stale": s["stale_rows"]}))
    reset("honest")
    new = run.run_pairs(ts, ["A0", "A5"], cfg_for(url, d), stub_grade, log=quiet)
    check(len(Fake.requests) == 4 and all(r["attempt"] == 2 for r in new),
          "the rerun regenerates all 4 pairs as attempt 2 (no answer overwritten)",
          str([r["attempt"] for r in new]))
    s = analyse.analyse(d)
    check(s["rows"] == 4 and s["stale_rows"] == 4
          and s["per_arm"]["A0"]["scored"] == 2,
          "post-fix rows analysed alone, never mixed with stale ones")


def test_mcnemar_hand_computed():
    # b=8, c=2: n=10, tail = C(10,0)+C(10,1)+C(10,2) = 56, p = 2*56/1024
    check(abs(analyse.mcnemar(8, 2) - 0.109375) < 1e-12, "McNemar(8,2) = 0.109375",
          str(analyse.mcnemar(8, 2)))
    check(abs(analyse.mcnemar(0, 6) - 0.03125) < 1e-12, "McNemar(0,6) = 0.03125")
    check(analyse.mcnemar(0, 0) == 1.0, "McNemar(0,0) = 1")
    check(analyse.min_discordant(0.05) == 6, "6 discordant pairs are the floor at 0.05")
    check(analyse.min_discordant(0.05 / 4) == 8,
          "8 discordant pairs are the floor at 0.05/4 (2/2**8 = 0.0078)")

    # A hand-built table: 12 uncontaminated tasks.
    #   A1 only: 8   A0 only: 2   both pass: 1   both fail: 1
    # plus one stack_error pair (excluded) and 3 contaminated tasks all A1-only.
    rows = []

    def add(t, arm, outcome, dom="typescript", contaminated=False):
        rows.append({"task": t, "arm": arm, "outcome": outcome, "domain": dom,
                     "contaminated": contaminated, "seconds": 10.0,
                     "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                     "x_yamadori": {}})
    pattern = ([("pass", "fail")] * 2 + [("fail", "pass")] * 8
               + [("pass", "pass"), ("fail", "fail")])
    for i, (a0, a1) in enumerate(pattern):
        add(f"t{i:02d}", "A0", a0)
        add(f"t{i:02d}", "A1", a1)
        for arm in ("A5", "A6"):
            add(f"t{i:02d}", arm, "pass")
    add("t99", "A0", "pass")
    add("t99", "A1", "stack_error")
    for arm in ("A5", "A6"):
        add("t99", arm, "pass")
    for i in range(3):
        add(f"c{i}", "A0", "fail", "three_tsl", True)
        add(f"c{i}", "A1", "pass", "three_tsl", True)
        for arm in ("A5", "A6"):
            add(f"c{i}", arm, "pass", "three_tsl", True)
    last = analyse.last_by_pair(rows)
    clean = sorted({r["task"] for r in rows if not r["contaminated"]})
    c = analyse.compare(last, clean, "A0", "A1")
    check(c["n_paired"] == 12 and c["b_only"] == 8 and c["a_only"] == 2
          and c["both_solved"] == 1 and abs(c["p"] - 0.109375) < 1e-12,
          "compare(): 12 paired (stack_error excluded), b=8, c=2, p=0.109375",
          json.dumps(c))
    blk = analyse.block(last, clean, ["A0", "A1", "A5", "A6"], "x")
    names = [(x["a"], x["b"]) for x in blk["comparisons"]]
    check(names == [("A0", "A1"), ("A0", "A5"), ("A0", "A6"), ("A6", "A5")]
          and blk["families"]["augmentation"]["size"] == 4,
          "Bonferroni family: 3 arms vs A0 plus A5 vs A6 = 4", str(names))
    fam = analyse.families(["A0", "A1", "C8", "C32", "C128"])
    check(fam == {"augmentation": [("A0", "A1")],
                  "thinking_cap": [("C32", "C8"), ("C32", "C128")]},
          "thinking-cap arms are their own family, compared against C32",
          str(fam))
    a1 = blk["comparisons"][0]
    check(abs(a1["p_bonferroni"] - 0.4375) < 1e-12 and a1["verdict"] is None,
          "Bonferroni p = 4 x 0.109375 = 0.4375, no verdict", json.dumps(a1))

    d = tempfile.mkdtemp(prefix="domain_analyse_")
    try:
        with open(os.path.join(d, "rows.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        s = analyse.analyse(d)
        h = s["headline_uncontaminated"]["comparisons"][0]
        dirty = s["contaminated_three_tsl"]["comparisons"][0]
        check(h["n_paired"] == 12 and h["b_only"] == 8,
              "headline block excludes three_tsl", json.dumps(h))
        check(dirty["n_paired"] == 3 and dirty["b_only"] == 3,
              "three_tsl reported in its own block", json.dumps(dirty))
        check(s["per_arm"]["A1"]["stack_error_rows"] == 1
              and s["per_arm"]["A1"]["pairs_still_stack_error"] == 1,
              "stack errors counted per arm", json.dumps(s["per_arm"]["A1"]))
        check(s["dashboard"]["progress"]["complete_all_arms"] == 12
              and s["dashboard"]["state"] == "ready"
              and {"arm_table", "raw", "pairs", "power", "difficulty",
                   "failures", "suspect"} <= set(s["dashboard"]),
              "dashboard block is lcb-shaped and uncontaminated-only",
              json.dumps(s["dashboard"]["progress"]))
        with contextlib.redirect_stdout(io.StringIO()):
            analyse.main([d])
        check(os.path.exists(os.path.join(d, "summary.json"))
              and os.path.exists(os.path.join(d, "summary.md")),
              "analyse.main writes summary.json and summary.md")
        sec = analyse.dashboard_section(os.path.dirname(d))
        check(sec.get("arms") is not None or sec.get("state") == "empty",
              "dashboard_section() returns a section")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_key_never_written(tmp):
    leaked = []
    for dirpath, _dirs, files in os.walk(tmp):
        for fn in files:
            with open(os.path.join(dirpath, fn), encoding="utf-8",
                      errors="replace") as f:
                if KEY in f.read():
                    leaked.append(fn)
    check(not leaked, "the API key appears in no file the runs wrote", str(leaked))


def test_stale_proxy_detection():
    d = tempfile.mkdtemp(prefix="domain_stale_")
    try:
        with open(os.path.join(d, "server.py"), "w") as f:
            f.write("import proxy\n")
        with open(os.path.join(d, "proxy.py"), "w") as f:
            f.write("def f():\n    import hints\n")
        with open(os.path.join(d, "hints.py"), "w") as f:
            f.write("x = 1\n")
        with open(os.path.join(d, "unrelated.py"), "w") as f:
            f.write("x = 1\n")
        check(run.import_closure("server", d) == ["hints", "proxy", "server"],
              "import closure follows function-level imports, skips unrelated",
              str(run.import_closure("server", d)))
        started = time.time()
        os.utime(os.path.join(d, "server.py"), (started - 100, started - 100))
        os.utime(os.path.join(d, "proxy.py"), (started - 100, started - 100))
        os.utime(os.path.join(d, "hints.py"), (started + 50, started + 50))
        os.utime(os.path.join(d, "unrelated.py"), (started + 50, started + 50))
        st = [n for n, _ in run.stale_files(started, "server", d)]
        check(st == ["hints.py"], "a module edited after the process started is stale",
              str(st))
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_select_tasks():
    ts = [{"id": f"{d}{i}", "domain": d} for d in ("a", "b") for i in range(4)]
    got = [t["id"] for t in run.select_tasks(ts, 2, None, None)]
    check(got == ["a0", "a1", "b0", "b1"], "--per-domain 2 takes the first 2 of each",
          str(got))


def sc_task(tid: str, prompt: str) -> dict:
    return {"id": tid, "domain": "typescript", "prompt": prompt,
            "contaminated": False, "difficulty": "easy",
            "grader": {"kind": "ts"}}


def test_self_check_round_trip(url, tmp):
    reset("honest")
    d = os.path.join(tmp, "selfcheck")
    rows = run.run_pairs([sc_task("sc01", "checky task")], ["S0"], cfg_for(url, d),
                         stub_grade, log=quiet, check_fn=stub_check)
    r = rows[0]
    reqs = Fake.requests
    check(len(reqs) == 3, "check, fix, check, answer: 3 requests", str(len(reqs)))
    sess = {q["session"] for q in reqs}
    check(len(sess) == 1 and next(iter(sess)).startswith("bench-")
          and r["session"] in sess,
          "every round of a row carries the row's own X-Yamadori-Session")
    check(all(q["feats"] == run.features("A0", "medium") for q in reqs),
          "every round carries S0's header, which is exactly A0's",
          json.dumps([q["feats"] for q in reqs]))
    check(all(q["body"].get("tools") == [run.CHECK_TOOL] for q in reqs)
          and all(q["body"]["reasoning_effort"] == "max" for q in reqs),
          "every round carries the tool and the same body fields")
    m2 = reqs[1]["body"]["messages"]
    ok_shape = (len(m2) == 3 and m2[0]["role"] == "user"
                and m2[1]["role"] == "assistant" and m2[1]["tool_calls"]
                and m2[1]["tool_calls"][0]["function"]["name"] == "check_solution"
                and m2[2]["role"] == "tool"
                and m2[2]["tool_call_id"] == m2[1]["tool_calls"][0]["id"])
    check(ok_shape, "round 2 continues in OpenAI format: user, assistant with "
          "tool_calls, tool with the matching tool_call_id",
          json.dumps(m2)[:400])
    check(not any("reasoning_content" in m for q in reqs
                  for m in q["body"]["messages"]),
          "reasoning_content is never echoed back")
    t1 = m2[2]["content"]
    check("DOES NOT COMPILE" in t1 and "TS2322" in t1 and "Retryable: yes" in t1
          and "did not run the hidden tests" in t1,
          "a failed check says what failed, that it is retryable, and that the "
          "hidden tests were not run", t1)
    t2 = reqs[2]["body"]["messages"][4]["content"]
    check("COMPILES" in t2 and "did not run the hidden tests" in t2,
          "a clean check says it compiles and that it is not a test result", t2)
    check(r["outcome"] == "pass" and r["rounds"] == 3 and r["check_rounds"] == 2
          and [c["ok"] for c in r["check_results"]] == [False, True]
          and [c["n_errors"] for c in r["check_results"]] == [1, 0],
          "row: pass, 3 rounds, 2 checks (fail then ok) with error counts",
          json.dumps({k: r.get(k) for k in ("outcome", "rounds", "check_rounds",
                                            "check_results")})[:400])
    check(r["usage"]["prompt_tokens"] == 300 and r["usage"]["completion_tokens"] == 150
          and r["usage"]["rounds"] == 3 and len(r["usage_rounds"]) == 3,
          "usage is summed over every round", json.dumps(r["usage"]))
    check(len(r["x_rounds"]) == 3 and r["mismatch"] == [] and r["twin"] == "A0"
          and r["self_check"] is True,
          "x_yamadori kept and verified per round")
    check(r["final_public_check"]["ok"] is True,
          "the final answer's own public check is recorded")
    check(isinstance(r["seconds"], float) and "model_seconds" in r
          and "check_seconds" in r, "seconds: total, model and check")
    with open(os.path.join(d, r["answer_file"]), encoding="utf-8") as f:
        ans = json.load(f)
    check(len(ans["rounds"]) == 3 and len(ans["messages"]) == 5
          and "PASS" in ans["content"],
          "the answer file keeps every round and the whole conversation")


def test_self_check_edges(url, tmp):
    cfg = cfg_for(url, os.path.join(tmp, "sc_edges"))
    reset("honest")
    r = run.run_pairs([sc_task("sc02", "repeaty task")], ["S0"], cfg,
                      stub_grade, log=quiet, check_fn=stub_check)[0]
    tool2 = Fake.requests[2]["body"]["messages"][4]["content"]
    check(r["check_results"][1]["repeat_of"] == 1 and "same code as check #1" in tool2,
          "a repeated check says it is the same code as before", tool2)
    reset("honest")
    r = run.run_pairs([sc_task("sc03", "strangey task")], ["S0"], cfg,
                      stub_grade, log=quiet, check_fn=stub_check)[0]
    tool1 = Fake.requests[1]["body"]["messages"][2]["content"]
    check(r["other_tool_calls"] == ["read_file"] and r["check_rounds"] == 0
          and "UNKNOWN TOOL" in tool1 and r["outcome"] == "pass",
          "a call to a tool the client does not have is answered and recorded",
          tool1)
    reset("honest")
    r = run.run_pairs([sc_task("sc04", "badargs task")], ["S0"], cfg,
                      stub_grade, log=quiet, check_fn=stub_check)[0]
    tool1 = Fake.requests[1]["body"]["messages"][2]["content"]
    check("BAD ARGUMENTS" in tool1 and r["check_results"][0]["stage"] == "arguments",
          "unparseable arguments get a retryable BAD ARGUMENTS reply", tool1)
    reset("honest")
    r = run.run_pairs([sc_task("sc05", "direct task")], ["S0"], cfg,
                      stub_grade, log=quiet, check_fn=stub_check)[0]
    check(r["rounds"] == 1 and r["check_rounds"] == 0 and r["outcome"] == "pass",
          "a model that answers without checking: 1 round, 0 checks, graded")
    reset("honest")
    r = run.run_pairs([sc_task("sc06", "checky BROKEN_CHECKER")], ["S0"], cfg,
                      stub_grade, log=quiet, check_fn=stub_check)[0]
    check(r["outcome"] == "stack_error" and r["stack_error_kind"] == "checker"
          and r["grade"] is None,
          "a checker that could not run: stack_error/checker, never graded",
          json.dumps({k: r.get(k) for k in ("outcome", "stack_error_kind")}))
    reset("lie_hints")
    r = run.run_pairs([sc_task("sc07", "checky task")], ["S0"], cfg,
                      stub_grade, log=quiet, check_fn=stub_check)[0]
    check(len(Fake.requests) == 1 and r["outcome"] == "stack_error"
          and r["stack_error_kind"] == "mismatch"
          and r["mismatch"][0].startswith("round 1:"),
          "a mismatch on round 1 stops the loop and voids the row",
          json.dumps({k: r.get(k) for k in ("stack_error_kind", "mismatch")})[:300])
    reset("http500")
    r = run.run_pairs([sc_task("sc08", "checky task")], ["S0"], cfg,
                      stub_grade, log=quiet, check_fn=stub_check)[0]
    check(r["outcome"] == "stack_error" and r["stack_error_kind"] == "http",
          "an HTTP error mid-loop is stack_error/http")


def test_verify_self_check_arms():
    for s_arm, one_shot in run.SELF_CHECK.items():
        x = fake_x(run.features(s_arm, "medium"))
        check(run.verify(s_arm, "medium", x, finish="tool_calls", content="") == [],
              f"{s_arm}: an honest tool-call round verifies clean")
        check(run.expected(s_arm) == run.expected(one_shot),
              f"{s_arm} expects exactly what {one_shot} does")
    x = fake_x(run.features("S5", "medium"))
    x["selection"]["signals"]["allowed"] = {"hints": True, "investigate": False,
                                            "fanout": 3}
    check(any("A5 must allow" in m for m in run.verify("S5", "medium", x,
                                                       finish="stop", content="c")),
          "S5 is held to A5's allow-everything check")
    x = fake_x(run.features("S6", "medium"))
    x["fanout"] = None
    check(run.verify("S6", "medium", x, finish="tool_calls", content="") == [],
          "S6: no fan-out expected on a tool-call round (the proxy fans out "
          "only a finished answer)")
    check(any("fanout" in m for m in run.verify("S6", "medium", x, finish="stop",
                                                content="```ts\nx\n```")),
          "S6: a final answer with no fan-out record is a mismatch")
    d = run.CHECK_TOOL["function"]
    check(d["name"] == "check_solution"
          and d["parameters"]["required"] == ["code"]
          and "does NOT run the hidden tests" in d["description"]
          and d["description"].startswith("Does my code compile?"),
          "check_solution: Jane Street name, {code}, description leads with the "
          "question and says it does not run the hidden tests")


def test_suites():
    got = {s: run.load_suite(s) for s in ("core", "react", "tc", "all")}
    check(len(got["core"]) == 100, "core suite = tasks.jsonl, 100 tasks",
          str(len(got["core"])))
    check(len(got["react"]) == 40
          and all(t["grader"]["kind"] == "react" and t["contaminated"] is False
                  for t in got["react"]),
          "react suite: 40 tasks, kind react, not contaminated")
    check(got["tc"] and all(t["grader"]["kind"] == "tc_tests"
                            and t["contaminated"] is True for t in got["tc"]),
          f"tc suite: {len(got['tc'])} tasks, kind tc_tests, every row "
          f"contaminated")
    check(len(got["all"]) == sum(len(got[s]) for s in ("core", "react", "tc")),
          "all = core + react + tc")
    import grade
    kinds = {t["grader"]["kind"] for t in got["all"]}
    check(kinds <= set(grade.GRADERS) and kinds <= set(grade.PUBLIC),
          "every suite's grader kind has a grader and a public check",
          str(sorted(kinds)))
    d = tempfile.mkdtemp(prefix="domain_suite_")
    try:
        with open(os.path.join(d, "tasks.jsonl"), "w") as f:
            f.write(json.dumps({"id": "x", "prompt": "p"}) + "\n")
        try:
            run.load_suite("core", here=d)
            refused = False
        except ValueError:
            refused = True
        check(refused, "a row that does not say whether it is contaminated is refused")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_suite_combos_and_manifest_extras(url, tmp):
    check(run.normalise_suite("react,core") == "core,react"
          and run.normalise_suite("core") == "core"
          and run.normalise_suite("tc,all") == "all",
          "suite lists normalise to one spelling")
    try:
        run.normalise_suite("core,nope")
        refused = False
    except ValueError:
        refused = True
    check(refused, "an unknown suite name is refused")
    both = run.load_suite("core,react")
    check(len(both) == 140, "core,react loads 100 + 40 tasks", str(len(both)))
    check(run.manifest_clash({"suite": "core,react"}, {"suite": "core"})
          == {"suite": ("core,react", "core")},
          "a core,react run dir refuses a core resume")
    st = run.hints_corpus_state()
    check(st.get("rows", 0) > 0 and set(st.get("states") or {}) <= {
          "keep", "edited", "reject", "unreviewed", "unparseable"}
          and st.get("hint_buckets_sha256") and len(st["recipes_sha256"]) == 64,
          "hints corpus state: review counts and both hashes",
          json.dumps({k: st.get(k) for k in ("states", "rows")}))
    import argparse
    prev = run.RESULTS
    run.RESULTS = tmp
    try:
        args = argparse.Namespace(suite="core", per_domain=None, tasks="ts01",
                                  domains="", effort="medium", max_tokens=4096,
                                  temperature=0.0, proxy=url, timeout=3600,
                                  no_analyse=True,
                                  concurrent_with="swebench, livebench")
        reset("honest")
        prev_grade = run.run_pairs

        def fake_pairs(tasks, arms, cfg, grade_fn, **kw):
            return prev_grade(tasks, arms, dict(cfg, quiet=False), stub_grade,
                              log=quiet,
                              check_fn=stub_check,
                              style_fn=lambda t, x: {"lint_errors": 2,
                                                     "lint_warnings": 1,
                                                     "modern_flags": ["forwardRef"]})
        run.run_pairs = fake_pairs
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                rc = run._run(args, KEY, ["A0"], [], [], [], "conc")
        finally:
            run.run_pairs = prev_grade
        with open(os.path.join(tmp, "conc", "manifest.json"), encoding="utf-8") as f:
            man = json.load(f)
        check(rc == 0 and man["concurrent_with"] == ["livebench", "swebench"]
              and man["runs"][-1]["concurrent_with"] == ["livebench", "swebench"]
              and man["hints_corpus"]["hint_buckets_sha256"],
              "the manifest records concurrent_with and the hints corpus state",
              json.dumps({k: man.get(k) for k in ("concurrent_with",)}))
        rows, _ = run.load_rows(os.path.join(tmp, "conc", "rows.jsonl"))
        check(rows and rows[-1]["lint_errors"] == 2 and rows[-1]["lint_warnings"] == 1
              and rows[-1]["modern_flags"] == ["forwardRef"]
              and rows[-1]["outcome"] in ("pass", "fail"),
              "a graded row carries lint_errors, lint_warnings, modern_flags; the "
              "outcome is the grader's alone")
        s = analyse.analyse(os.path.join(tmp, "conc"))
        check(s["timing"]["clean"] is False
              and s["dashboard"]["timing"]["label"] == "CONCURRENT"
              and any("CONCURRENT" in c for c in s["caveats"])
              and "(CONCURRENT)" in analyse.markdown(s),
              "analyse labels timing CONCURRENT everywhere it is shown")
        a0 = next(x for x in s["dashboard"]["arm_table"] if x["arm"] == "A0")
        check(a0["lint_errors_mean"] == 2.0 and a0["modern_flags"] == {"forwardRef": 1}
              and a0["lint_n"] == 1,
              "the style score is reported per arm on the dashboard", json.dumps(a0)[:300])
    finally:
        run.RESULTS = prev
    r = run.attach_style({"grade": None, "answer_file": "x"}, {}, {"run_dir": tmp},
                         lambda t, x: 1 / 0)
    check("style" not in r, "an ungraded (stack_error) row gets no style score")
    d = os.path.join(tmp, "sty")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "a.json"), "w") as f:
        json.dump({"content": "x"}, f)
    r = run.attach_style({"grade": {"passed": True}, "answer_file": "a.json",
                          "outcome": "pass"}, {}, {"run_dir": d}, lambda t, x: 1 / 0)
    check(r["style"].get("style_error") and r["outcome"] == "pass",
          "a crashing style scorer is recorded as style_error; outcome untouched")


def test_busy_429_is_not_run(url, tmp):
    for arm, prompt in (("A0", "good task"), ("S0", "checky task")):
        reset("busy")
        Fake.busy_left = 2
        cfg = dict(cfg_for(url, os.path.join(tmp, f"busy_{arm}")), sleep=lambda s: None)
        r = run.run_pairs([sc_task(f"b_{arm}", prompt)], [arm], cfg, stub_grade,
                          log=quiet, check_fn=stub_check)[0]
        check(r["outcome"] == "pass" and r["busy_429s"] == 2
              and r["busy_wait_s"] == 2 * run.BUSY_WAIT_S
              and r["seconds"] < run.BUSY_WAIT_S,
              f"{arm}: two 429s are waited out, not recorded as a row; the wait "
              f"is kept out of seconds",
              json.dumps({k: r.get(k) for k in ("outcome", "busy_429s",
                                                "busy_wait_s", "seconds")}))


def test_parallel_plan(url, tmp):
    suite = run.load_suite("core,react")
    a = run.select_tasks(suite, None, None, ["typescript", "typegpu", "react"],
                         sample=8, seed=20260923)
    b = run.select_tasks(suite, None, None, ["rust_wasm", "three_tsl"],
                         sample=8, seed=20260923)
    whole = run.select_tasks(suite, None, None, None, sample=8, seed=20260923)
    check(len(a) == 24 and len(b) == 16 and len(whole) == 40
          and {t["id"] for t in a} | {t["id"] for t in b} == {t["id"] for t in whole},
          "stratified sample: 8 per domain, and the per-domain draw does not "
          "depend on which other domains are selected",
          f"{len(a)} + {len(b)} vs {len(whole)}")
    check([t["id"] for t in run.sample_tasks(suite, 8, 20260923)]
          == [t["id"] for t in run.sample_tasks(list(reversed(suite)), 8, 20260923)][::-1],
          "the draw does not depend on row order")
    # helper lane busy: waited out, not a row
    reset("helper_busy")
    Fake.busy_left = 2
    cfg = dict(cfg_for(url, os.path.join(tmp, "helper")), sleep=lambda s: None)
    rows = run.run_pairs([sc_task("hb1", "good")], ["A6"], cfg, stub_grade,
                         log=quiet, check_fn=stub_check)
    allrows, _ = run.load_rows(os.path.join(tmp, "helper", "rows.jsonl"))
    r = rows[0]
    check(len(allrows) == 1 and r["outcome"] == "pass" and r["helper_busy_retries"] == 2
          and r["attempt"] == 3 and os.path.exists(os.path.join(
              tmp, "helper", "answers", "hb1__A6__1.json")),
          "a helper-lane-busy row is retried after a wait, not recorded; the "
          "discarded attempts keep their answer files",
          json.dumps({k: r.get(k) for k in ("outcome", "helper_busy_retries",
                                            "attempt")}))
    check(not run.helper_busy({"stack_error_kind": "mismatch",
                               "mismatch": ["hints off but 1 emitted"]}),
          "any other mismatch is still a stack_error row")
    # import scored rows from another run dir
    src = os.path.join(tmp, "imp_src")
    reset("honest")
    run.run_pairs(tasks_fixture()[:2], ["A0"], cfg_for(url, src), stub_grade, log=quiet)
    dst = os.path.join(tmp, "imp_dst")
    os.makedirs(dst, exist_ok=True)
    n = run.import_rows(src, dst, tasks_fixture()[:1], ["A0", "A6"])
    got, _ = run.load_rows(os.path.join(dst, "rows.jsonl"))
    check(n == 1 and got[0]["imported_from"] == "imp_src"
          and os.path.exists(os.path.join(dst, got[0]["answer_file"]))
          and run.import_rows(src, dst, tasks_fixture()[:1], ["A0"]) == 0,
          "import copies only the selected scored pairs, with answers, once")
    reset("honest")
    run.run_pairs(tasks_fixture()[:1], ["A0"], cfg_for(url, dst), stub_grade, log=quiet)
    check(len(Fake.requests) == 0, "an imported pair is done: not requested again")
    # group merge in analyse
    for rid, t in (("ga", tasks_fixture()[0]), ("gb", tasks_fixture()[1])):
        d = os.path.join(tmp, "grp", rid)
        reset("honest")
        run.run_pairs([t], ["A0", "S0"], cfg_for(url, d), stub_grade, log=quiet,
                      check_fn=stub_check)
        with open(os.path.join(d, "manifest.json"), "w") as f:
            json.dump({"run_id": rid, "group": "g1", "effort": "medium",
                       "runs": [{"started": 1}]}, f)
    s1 = analyse.analyse(os.path.join(tmp, "grp", "ga"))
    check(s1["merged_run_dirs"] == ["ga", "gb"] and s1["tasks"]["all"] == 2
          and s1["run_id"] == "g1" and len(s1["runs"] if "runs" in s1 else [1, 1]) == 2,
          "analyse merges every run dir of the group", json.dumps(s1["merged_run_dirs"]))
    s2 = analyse.analyse(os.path.join(tmp, "grp", "ga"), merge=False)
    check(s2["tasks"]["all"] == 1, "merge=False analyses the dir alone")
    ck = run.preflight.__code__.co_varnames
    check("skip" in ck, "preflight takes checks to skip")


def test_server_reasoning_cap(url, tmp):
    base = {"task": "t", "arm": "A0", "outcome": "fail", "finish_reason": "stop",
            "empty_content": True, "hit_thinking_cap": False,
            "grade": {"passed": False, "stage": "extract"},
            "usage": {"completion_tokens": 32881}}
    check(analyse.server_cap_hit(base), "the probe's ts02/A0 row is the server's "
          "hard reasoning stop")
    for k, v, why in (("hit_thinking_cap", True, "the graceful close happened"),
                      ("finish_reason", "length", "a length finish"),
                      ("usage", {"completion_tokens": 20000}, "20k tokens"),
                      ("empty_content", False, "code was answered")):
        r = dict(base, **{k: v})
        if k == "empty_content":
            r["grade"] = {"passed": False, "stage": "test"}
        check(not analyse.server_cap_hit(r), f"not reclassified: {why}")
    s0 = dict(base, arm="S0", usage={"completion_tokens": 40000},
              usage_rounds=[{"completion_tokens": 7000}, {"completion_tokens": 32800}])
    check(analyse.server_cap_hit(s0), "an S row is judged on its LAST round")
    rows = analyse.annotate([base])
    check(rows[0]["outcome"] == "fail"
          and rows[0]["annotations"] == ["server_reasoning_cap"]
          and "annotations" not in base,
          "a server-cap row is ANNOTATED; its outcome stays fail (the model looped)")
    check(run.done_pairs([base]) == {run.pair_key("t", "A0")},
          "resume treats it as done: it is a scored model failure")


def test_pause_ownership():
    class J:
        def __init__(self, rec):
            self.rec, self.resumed, self.paused_calls = rec, False, 0

        def paused(self, lane):
            return self.rec

        def pause(self, lane, by, why, ttl_seconds):
            self.paused_calls += 1
            self.rec = {"by": by, "until": time.time() + ttl_seconds}

        def resume(self, lane):
            self.resumed = True
            return True
    prev = run._jobs
    try:
        op = J({"by": "operator", "until": time.time() + 3 * 3600})
        run._jobs = lambda: op
        run.keep_quiet("r1")
        check(op.paused_calls == 0 and op.rec["by"] == "operator",
              "a live operator pause that outlasts ours is left alone")
        check(run.release_worker() is False and not op.resumed,
              "release_worker never lifts a pause it did not set")
        short = J({"by": "operator", "until": time.time() + 60})
        run._jobs = lambda: short
        run.keep_quiet("r1")
        check(short.paused_calls == 1, "a foreign pause about to expire is extended by ours")
        run._FOREIGN_PAUSE.clear()
        mine = J({"by": "bench/domain/run.py r1", "until": time.time() + 600})
        run._jobs = lambda: mine
        check(run.release_worker() is True and mine.resumed,
              "release_worker lifts the runner's own pause")
    finally:
        run._jobs = prev


def test_server_sampling_condition():
    old = {"suite": "core", "effort": "medium"}
    check(run.manifest_clash(old, {"suite": "core",
                                   "server_sampling": {"min_p": "0.0"}}) == {},
          "a manifest from before server_sampling was recorded is not refused")
    check("server_sampling" in run.manifest_clash(
        {"suite": "core", "server_sampling": {"min_p": "0.05"}},
        {"suite": "core", "server_sampling": {"min_p": "0.0"}}),
        "a run dir made at min_p 0.05 refuses a resume at min_p 0.0")


def test_a6_forces_every_mechanism():
    x = fake_x(run.features("A6", "medium"))
    check(run.verify("A6", "medium", x, finish="tool_calls", content="") == [],
          "A6 with check_code offered and repair enabled verifies clean")
    x2 = json.loads(json.dumps(x))
    x2["repair"] = None
    check(any("repair forced on" in m for m in run.verify("A6", "medium", x2,
                                                         finish="tool_calls", content="")),
          "A6 whose repair pass is off: mismatch (stack_error, re-run)")
    x3 = json.loads(json.dumps(x))
    x3["check_code"] = {"offered": False}
    check(any("check_code forced on" in m for m in run.verify("A6", "medium", x3,
                                                             finish="tool_calls", content="")),
          "A6 whose check_code was not offered: mismatch")
    x4 = json.loads(json.dumps(x))
    x4["selection"]["signals"]["forced"] += ["check_code", "repair"]
    check(run.verify("A6", "medium", x4, finish="tool_calls", content="") == [],
          "forced check_code/repair in signals do not disturb the forced check")
    s = analyse.markdown({"run_id": "x", "conditions": {}, "rows": 0, "bad_lines": 0,
                          "tasks": {"uncontaminated": 0, "contaminated": 0},
                          "caveats": [], "headline_uncontaminated": {"label": "h", "tasks": 0, "families": {}, "comparisons": []},
                          "contaminated_three_tsl": {"label": "c", "tasks": 0, "families": {}, "comparisons": []},
                          "all_domains": {"label": "a", "tasks": 0, "families": {}, "comparisons": []},
                          "per_arm": {}, "by_domain": {}, "triggers": {}, "arms": []})
    check("Known limit" in s and "error code" in s,
          "summary.md states the tool-error-code limit")


def test_stale_by_arm_and_timing(url, tmp):
    d = os.path.join(tmp, "stale_arm")
    reset("honest")
    run.run_pairs(tasks_fixture()[:1], ["A0", "A6"], cfg_for(url, d), stub_grade,
                  log=quiet)
    n = run.mark_stale(d, "pre-sequential-fanout", ["A6"])
    rows, _ = run.load_rows(os.path.join(d, "rows.jsonl"))
    check(n == 1 and [bool(r.get("stale_code")) for r in rows] == [False, True],
          "--stale-arms marks only that arm's rows")
    man = {"runs": [{"started": 100, "concurrent_with": ["swebench"]},
                    {"started": 200, "concurrent_with": []}]}
    old = {"started_at": 150, "seconds": 10}
    new = {"started_at": 250, "seconds": 20}
    tagged = {"started_at": 50, "seconds": 30, "concurrent_with": []}
    check(analyse.row_concurrent(old, man) == ["swebench"]
          and analyse.row_concurrent(new, man) == []
          and analyse.row_concurrent(tagged, man) == [],
          "a row's GPU sharing: its own tag, else the invocation it ran in")
    t = analyse.timing_metrics([old, new, tagged], man)
    check(t["timing_rows"] == {"clean": 2, "concurrent": 1}
          and t["mean_s_clean"] == 25.0 and t["timing_label"] == "mixed",
          "timing is split clean / concurrent, never averaged together",
          json.dumps(t))
    reset("honest")
    r = run.run_pairs([sc_task("tc1", "good")], ["A0"],
                      dict(cfg_for(url, os.path.join(tmp, "tagrow")),
                           concurrent_with=["livebench"]), stub_grade, log=quiet)[0]
    check(r["concurrent_with"] == ["livebench"], "each row records its own GPU sharing")


def test_tool_turns(url, tmp):
    x = fake_x(run.features("A6", "medium"))
    x["tool_turns"] = {"limit": 10, "turns": 10, "hit": True}
    m = analyse.mechanisms({"arm": "A6", "x_yamadori": x})
    check(m["tool_turns"] == {"recorded": True, "limit": [10], "turns_max": 10,
                              "turns_sum": 10, "hit": True},
          "tool_turns recorded per row; hit is data", json.dumps(m["tool_turns"]))
    h = analyse.mechanism_health([{"arm": "A6", "x_yamadori": x},
                                  {"arm": "A6", "x_yamadori": fake_x(run.features("A6", "medium"))}])
    check(h["tool_turns"]["hit"] == 1 and h["tool_turns"]["recorded"] == 1
          and h["tool_turns"]["limit"] == [10], "tool-turn health per arm")
    d = os.path.join(tmp, "tt")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "manifest.json"), "w") as f:
        json.dump({"run_id": "tt"}, f)
    run.note_condition(d, "tool_turns_limit", [10])
    run.note_condition(d, "tool_turns_limit", [10])
    with open(os.path.join(d, "manifest.json")) as f:
        man = json.load(f)
    check(man["conditions_seen"] == {"tool_turns_limit": [10]},
          "the cap goes into the manifest's condition record, once")


def test_hints_epoch_pause():
    xh = fake_x(run.features("A6", "medium"))
    x0 = fake_x(run.features("A0", "medium"))
    xn = json.loads(json.dumps(xh))
    xn["hints"] = []
    rows = [{"task": "a", "arm": "A0", "outcome": "fail", "x_yamadori": x0},
            {"task": "a", "arm": "A6", "outcome": "pass", "x_yamadori": xh},
            {"task": "b", "arm": "A0", "outcome": "pass", "x_yamadori": x0},
            {"task": "b", "arm": "A6", "outcome": "fail", "x_yamadori": xn}]
    last = analyse.last_by_pair(rows)
    h = analyse.hint_metrics(last, ["a", "b"], "A6")
    check(h == {"hint_rows": 1, "hint_rate": 0.5, "hint_rows_paired": 1,
                "hint_rows_pass": 1, "hint_rows_a0_pass": 0},
          "hint injection rate, and pass on injected rows vs paired A0", json.dumps(h))

    class J:
        def __init__(self, rec):
            self.rec = rec

        def paused(self, lane):
            return self.rec if self.rec and self.rec["until"] > time.time() else None

        def pause(self, lane, by, why, ttl_seconds):
            self.rec = {"by": by, "why": why, "until": time.time() + ttl_seconds}

        def resume(self, lane):
            self.rec = None
            return True
    prev = run._jobs
    try:
        j = J({"by": "operator-benchmarks", "why": "w", "until": time.time() + 600})
        run._jobs = lambda: j
        run.keep_quiet("r")
        check(j.rec["by"].startswith("bench/domain/run.py"),
              "a foreign pause about to expire is extended by ours")
        run.release_worker()
        check(j.rec and j.rec["by"] == "operator-benchmarks"
              and 500 < j.rec["until"] - time.time() <= 600,
              "release puts the foreign pause back for its remaining time")
    finally:
        run._jobs = prev


def test_stop_file(url, tmp):
    d = os.path.join(tmp, "stopfile")
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "STOP"), "w").close()
    reset("honest")
    rows = run.run_pairs(tasks_fixture()[:2], ["A0"], cfg_for(url, d), stub_grade,
                         log=quiet)
    check(rows == [] and not Fake.requests and not os.path.exists(os.path.join(d, "STOP")),
          "a STOP file stops the runner before the next pair, and is consumed")


def test_mechanism_evidence(url, tmp):
    x6 = fake_x(run.features("A6", "medium"))
    x6["tools"] = [{"name": "find_definition_opt", "empty": False, "error": False,
                    "chars": 900},
                   {"name": "find_by_pattern", "empty": True, "error": False,
                    "chars": 40},
                   {"name": "find_by_meaning", "empty": False, "error": True,
                    "chars": 80}]
    x6["check_code"] = {"offered": True, "calls": 1, "results": [{"ok": True}]}
    x6["fanout"] = {"n": 3, "asked": 3, "selection": "code_medoid",
                    "winner": "seed-2", "replaced": True,
                    "candidates": [{"variant": "original", "parses": True},
                                   {"variant": "seed-1", "parses": False},
                                   {"variant": "seed-2", "parses": True}]}
    r6 = {"arm": "A6", "x_yamadori": x6}
    m = analyse.mechanisms(r6)
    check(m["retrieval"]["calls"] == 3 and m["retrieval"]["nonempty"] == 1
          and m["retrieval"]["empty"] == 1 and m["retrieval"]["errors"] == 1
          and m["retrieval"]["tools"]["find_by_pattern"] == 1,
          "retrieval evidence: calls, which tools, non-empty, empty, errors",
          json.dumps(m["retrieval"]))
    check(m["hints"]["injected"] == 1 and len(m["hints"]["ids"]) == 1,
          "hints evidence: count and ids")
    check(m["deep_thinking"]["ran"] and m["deep_thinking"]["searches"] == 2
          and m["deep_thinking"]["injected"], "deep-thinking evidence")
    f = m["fanout"]
    check(f["selection"] == "code_medoid" and f["replaced"] is True
          and f["delivered"] is True and f["candidates_parsed"] == 2
          and f["candidates"] == 3,
          "fan-out evidence: method, winner delivered, candidates parsed",
          json.dumps(f))
    check(m["check_code"]["allowed"] and m["check_code"]["calls"] == 1,
          "check_code evidence")
    x7 = json.loads(json.dumps(x6))
    x7["tools"].append({"name": "check_code", "empty": False, "error": False,
                        "chars": 143})
    m7 = analyse.mechanisms({"arm": "A6", "x_yamadori": x7})
    check(m7["retrieval"]["calls"] == 3 and "check_code" not in m7["retrieval"]["tools"]
          and m7["check_code"]["tool_calls_nonempty"] == 1,
          "a check_code call is counted as check_code, not as retrieval")
    x0 = fake_x(run.features("A0", "medium"))
    m0 = analyse.mechanisms({"arm": "A0", "x_yamadori": x0})
    check(not any(m0[k]["allowed"] for k in analyse.MECHANISMS),
          "A0: no mechanism allowed",
          json.dumps({k: m0[k]["allowed"] for k in analyse.MECHANISMS}))
    ms = analyse.mechanisms({"arm": "S0", "x_rounds": [x0, x0],
                             "check_results": [{"ok": False, "n_errors": 2},
                                               {"ok": True, "n_errors": 0}],
                             "rounds": 3})
    check(ms["self_check"]["allowed"] and ms["self_check"]["errors_per_check"] == [2, 0],
          "self-check evidence: error count per check")
    xh = fake_x(run.features("A2", "medium"))
    xh["hints"] = []
    mh = analyse.mechanisms({"arm": "A2", "x_yamadori": xh})
    check(mh["hints"]["allowed"] and mh["hints"]["ran"] and not mh["hints"]["data"],
          "hints that found nothing above the floor: ran, no data -- recorded, not excluded")
    h = analyse.mechanism_health([r6, {"arm": "A6", "x_yamadori": fake_x(
        run.features("A6", "medium"))}])
    check(h["fanout"]["allowed"] == 2 and h["fanout"]["data"] == 1
          and h["retrieval"]["data_pct"] == 50.0,
          "mechanism health: allowed / ran / data per mechanism", json.dumps(h["fanout"]))
    reset("honest")
    rows = run.run_pairs([sc_task("m1", "good")], ["A6"],
                         cfg_for(url, os.path.join(tmp, "mech")), stub_grade, log=quiet)
    check("mechanisms" in rows[0] and rows[0]["mechanisms"]["deep_thinking"]["ran"],
          "every written row carries its mechanism evidence")


def test_suite_clash_refused(tmp):
    fixed = {"effort": "medium", "max_tokens": 4096, "temperature": 0.0,
             "body_tier": "max", "proxy": "p", "suite": "core"}
    old_core = {k: v for k, v in fixed.items() if k != "suite"}
    check(run.manifest_clash(old_core, fixed) == {},
          "a manifest from before --suite resumes as core")
    check(run.manifest_clash(old_core, dict(fixed, suite="tc")) == {"suite": ("core", "tc")},
          "... and refuses tc")
    check(run.manifest_clash(dict(fixed, suite="react"), fixed)
          == {"suite": ("react", "core")}, "a react run dir refuses core")
    import argparse
    prev = run.RESULTS
    run.RESULTS = tmp
    try:
        rid = "clash"
        os.makedirs(os.path.join(tmp, rid), exist_ok=True)
        with open(os.path.join(tmp, rid, "manifest.json"), "w") as f:
            json.dump(dict(fixed, suite="react", run_id=rid, runs=[]), f)
        args = argparse.Namespace(suite="core", per_domain=None, tasks="ts01",
                                  domains="", effort="medium", max_tokens=4096,
                                  temperature=0.0, proxy="p", timeout=3600,
                                  no_analyse=True)
        reset("honest")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = run._run(args, KEY, ["A0"], [], [], [], rid)
        check(rc == 2 and "different conditions" in out.getvalue()
              and "suite" in out.getvalue() and not Fake.requests,
              "resuming a react run dir with --suite core aborts before any request",
              out.getvalue().strip()[:200])
    finally:
        run.RESULTS = prev


def test_analyse_self_check():
    check(run.SELF_CHECK == analyse.SELF_CHECK_TWIN,
          "analyse and run agree on each S arm's twin")
    fam = analyse.families(["A0", "A5", "S0", "S5"])
    check(fam == {"augmentation": [("A0", "A5")],
                  "self_check": [("A0", "S0"), ("A5", "S5"), ("A0", "S5")]},
          "S arms are their own family: vs twin, and vs A0", str(fam))
    rows = []

    def add(t, arm, outcome, st, dom="typescript", dirty=False, **kw):
        r = {"task": t, "arm": arm, "outcome": outcome, "domain": dom,
             "contaminated": dirty, "seconds": 10.0 if arm == "A0" else 30.0,
             "usage": {"prompt_tokens": 100, "completion_tokens": 50},
             "grade": {"passed": outcome == "pass", "stage": st},
             "x_yamadori": {"hops": 1}}
        r.update(kw)
        rows.append(r)
    # (A0 stage, S0 stage) per task
    plan = [("compile", "pass")] * 3 + [("compile", "test"), ("pass", "pass"),
                                        ("test", "test"), ("extract", "pass"),
                                        ("pass", "compile")]
    for i, (a, s) in enumerate(plan):
        add(f"t{i}", "A0", "pass" if a == "pass" else "fail",
            "test" if a == "pass" else a)
        add(f"t{i}", "S0", "pass" if s == "pass" else "fail",
            "test" if s == "pass" else s, check_rounds=i % 3,
            check_results=[{"ok": s != "compile"}] if i % 3 else [],
            final_public_check={"ok": s in ("pass", "test")},
            x_rounds=[{"hops": 2}, {"hops": 1}], rounds=2, self_check=True)
    for i in range(2):                         # contaminated type-challenges
        add(f"tc{i}", "A0", "fail", "compile", "typescript_tc", True)
        add(f"tc{i}", "S0", "pass", "test", "typescript_tc", True, check_rounds=1,
            x_rounds=[{"hops": 1}])
    last = analyse.last_by_pair(rows)
    clean = sorted({r["task"] for r in rows if not r["contaminated"]})
    tw = analyse.twins(last, clean, ["A0", "S0"])[0]
    check(tw["n_paired"] == 8 and tw["s_only"] == 4 and tw["one_shot_only"] == 1
          and tw["fixed_by_checking"] == 3 and tw["compile_to_test"] == 1
          and tw["extract_to_pass"] == 1 and abs(tw["p"] - analyse.mcnemar(4, 1)) < 1e-12,
          "twin: 8 paired, S only 4, twin only 1, fixed by checking 3 "
          "(compile -> pass), compile -> test 1, extract -> pass 1",
          json.dumps(tw))
    check(tw["fixed_with_check"] == 2,
          "fixed_with_check counts only S rows that checked (t0 made 0 checks)",
          str(tw["fixed_with_check"]))
    d = tempfile.mkdtemp(prefix="domain_sc_")
    try:
        with open(os.path.join(d, "rows.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        s = analyse.analyse(d)
        h = s["headline_uncontaminated"]
        check(h["tasks"] == 8 and s["self_check_twins_uncontaminated"][0]["n_paired"] == 8,
              "type-challenges (contaminated) stays out of the headline and the twins")
        check(s["by_domain"]["typescript_tc"]["contaminated"] is True
              and s["by_domain"]["typescript"]["contaminated"] is False
              and s["tasks"]["contaminated_domains"] == ["typescript_tc"],
              "a domain is contaminated because its rows say so, not by name")
        pa = s["per_arm_uncontaminated"]["S0"]
        check(pa["stages"] == {"pass": 5, "extract": 0, "compile": 1, "test": 2}
              and pa["tool_hops_mean"] == 1.0 and pa["final_public_ok"] == 7
              and pa["check_rounds_max"] == 2 and pa["checked_any"] == 5,
              "per arm: stages, proxy tool hops from x_rounds, check rounds, "
              "final public-check passes", json.dumps(pa)[:500])
        dash = s["dashboard"]
        row = next(x for x in dash["arm_table"] if x["arm"] == "S0")
        check({"stages", "median_s", "tok_total", "tool_hops", "final_compiles",
               "check_rounds_mean", "final_public_ok", "twin"} <= set(row)
              and row["tok_total"] == 150 and row["final_compiles"] == 7
              and row["twin"] == "A0",
              "dashboard arm_table carries stages, tokens, hops and S-arm fields",
              json.dumps(row)[:400])
        a0 = next(x for x in dash["arm_table"] if x["arm"] == "A0")
        check("check_rounds_mean" not in a0 and a0["self_check"] is False
              and {"arm", "k", "n", "rate", "lo", "hi", "mean_s", "tok_in",
                   "tok_out"} <= set(a0),
              "a one-shot arm keeps the lcb keys and gets no S-arm fields")
        check(dash["twins"] and dash["twins"][0]["fixed_by_checking"] == 3
              and dash["twins"][0]["p_bonferroni"] is not None
              and dash["excluded_contaminated"] == {"tasks": 2,
                                                    "domains": ["typescript_tc"]}
              and any(p["family"] == "self_check" for p in dash["pairs"]),
              "dashboard: twins (with the family-corrected p), the excluded "
              "contaminated count, and the self_check pair",
              json.dumps({k: dash[k] for k in ("twins", "excluded_contaminated")})[:400])
        cell = dash["difficulty"][0]["by_arm"]["S0"]
        check({"lo", "hi", "stages"} <= set(cell) and dash["domain_pairs"],
              "per domain: Wilson bounds, stages, and uncorrected pairs")
        md = analyse.markdown(s)
        check("Self-check vs its one-shot twin" in md and "fixed by checking" in md,
              "summary.md reports the twin comparison")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main() -> int:
    t0 = time.perf_counter()
    srv, url = start_fake()
    tmp = tempfile.mkdtemp(prefix="domain_run_")
    try:
        for fn, args in ((test_every_arm_sends_its_header, (url, tmp)),
                         (test_verify_catches_contradictions, ()),
                         (test_stack_errors_are_not_failures, (url, tmp)),
                         (test_resume, (url, tmp)),
                         (test_grader_error_is_regraded_not_regenerated, (url, tmp)),
                         (test_stale_code_rows, (url, tmp)),
                         (test_mcnemar_hand_computed, ()),
                         (test_key_never_written, (tmp,)),
                         (test_stale_proxy_detection, ()),
                         (test_select_tasks, ()),
                         (test_self_check_round_trip, (url, tmp)),
                         (test_self_check_edges, (url, tmp)),
                         (test_verify_self_check_arms, ()),
                         (test_suites, ()),
                         (test_suite_clash_refused, (tmp,)),
                         (test_busy_429_is_not_run, (url, tmp)),
                         (test_parallel_plan, (url, tmp)),
                         (test_server_reasoning_cap, (url, tmp)),
                         (test_pause_ownership, ()),
                         (test_server_sampling_condition, ()),
                         (test_mechanism_evidence, (url, tmp)),
                         (test_a6_forces_every_mechanism, ()),
                         (test_stale_by_arm_and_timing, (url, tmp)),
                         (test_stop_file, (url, tmp)),
                         (test_tool_turns, (url, tmp)),
                         (test_hints_epoch_pause, ()),
                         (test_suite_combos_and_manifest_extras, (url, tmp)),
                         (test_analyse_self_check, ()),
                         (test_key_never_written, (tmp,))):
            print(f"\n--- {fn.__name__} ---", flush=True)
            n0 = len(_results)
            try:
                fn(*args)
            except Exception:                                    # noqa: BLE001
                check(False, f"{fn.__name__} itself raised",
                      traceback.format_exc().strip().split("\n")[-1])
            for ok, nm, detail in _results[n0:]:
                print(("  pass  " if ok else "  FAIL  ") + nm
                      + (f"\n        <- {detail}" if not ok and detail else ""))
    finally:
        srv.shutdown()
        shutil.rmtree(tmp, ignore_errors=True)
    passed = sum(1 for ok, _, _ in _results if ok)
    print(f"\n{passed}/{len(_results)} checks passed "
          f"in {time.perf_counter() - t0:.1f}s")
    return 0 if passed == len(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
