#!/usr/bin/env python
"""The selection-engine benchmark on the target-domain tasks: one row per
(task, arm), through the proxy on :1234, one request at a time.

    python bench/domain/run.py --key-file K --effort medium --run-id med1
    python bench/domain/run.py --key-file K --preflight-only
    python bench/domain/run.py --key-file K --arms A0,A5 --per-domain 2 --run-id smoke
    python bench/domain/run.py --key-file K --suite react --arms A0,S0 --run-id r1

The spec is docs/SELECTION-BUILD.md section 5. Analysis is bench/domain/analyse.py.

THE ARMS

Every arm sends `reasoning_effort: "max"` in the body, so the tier ALLOWS
everything, and pins the thinking style with `effort` in X-Yamadori-Features,
so effort is held fixed across arms (tiers shift effort otherwise -- harm 4 in
SELECTION-BUILD). A flag named in the header is FORCED; a flag left out is the
selection engine's call (mcp/selection.py `_forced`).

  A0  everything forced off: the model as it ships
  A1  retrieval only        A2  hints only
  A3  deep thinking only (forced on with retrieval OFF; its tools come from
      proxy.deep_thinking_tools(), not the request -- analyse.py reports how
      often it actually searched, which is the evidence it had tools)
  A4  fan-out only, N=3
  A5  nothing forced: the selection engine decides (header carries only effort)
  A6  everything forced on

THE THINKING-CAP ARMS (let it cook)

  C8 / C32 / C128   A0 with `reasoning_cap` 8192 / 32768 / 131072 in the
                    header (tiers.budget clamps to [1024, 131072]). The
                    default cap is now derived from the main context's share
                    of the KV pool, so every C arm sets its cap explicitly and
                    C32 is the fixed reference; C128 is effectively unlimited. analyse.py compares C8 and C128
                    against C32. Each row checks x_yamadori.budget.
                    reasoning_budget_tokens equals the cap the arm sent.

THE SELF-CHECK ARMS (the model may compile its own answer)

  S0 / S5 / S6      A0 / A5 / A6 -- the same X-Yamadori-Features header on
                    every request -- plus ONE client-side tool in the body's
                    `tools`: `check_solution {"code": string}`. This runner is
                    the client harness (as Hermes or Claude Code would be):
                    the proxy merges it with its own tools (proxy.merge_tools;
                    client tools pass through even with retrieval off) and
                    returns a call to it to the client (proxy.complete, the
                    `not ours` return). The runner then runs grade.public_check:
                    the grader's extract + compile stages against the real
                    packages with NO hidden file in the sandbox -- no test.ts,
                    no host_test, no wasm_test, no TSL require/forbid list --
                    so the result can never carry a hidden test's name, value
                    or content (test_public_check.py proves every wrong_test
                    answer passes it). The result goes back as the tool message
                    and the conversation continues in OpenAI format (assistant
                    message with its tool_calls, then one `tool` message per
                    call; reasoning_content is not echoed -- proxy.strip_thinking
                    would drop it anyway). There is NO round cap: the loop ends
                    when the model answers without a tool call, and a
                    conversation whose context is full is ended by the proxy's
                    own landing (proxy.context_full), which withdraws the tools.
                    The final answer is graded exactly like every other arm's.
                    Each row records rounds, check_rounds, check_results (ok,
                    stage, error count per call), usage summed over every
                    round, seconds in total and split model/check, x_yamadori
                    per round (x_rounds; every one is verified against the arm)
                    and final_public_check (did the answer it gave compile?).
                    A check that could not RUN (toolchain missing) makes the
                    row stack_error/checker: the model was told nothing true.
                    analyse.py pairs each S arm with its one-shot twin.
                    S arms are not in the default --arms; name them
                    (--arms A0,S0).

THE SUITES (--suite, recorded in the manifest; a run dir holds one)

  core    tasks.jsonl -- typescript, typegpu, rust_wasm, three_tsl (the
          default, so every run dir made before --suite means the same thing)
  react   tasks_react.jsonl -- React 19.2 hooks and components (grade_react)
  tc      tasks_type_challenges.jsonl -- type-challenges, CONTAMINATED (public
          since 2020): rows carry contaminated=true and analyse.py keeps them
          out of every headline, as it does three_tsl
  all     the three together

A ROW THE STACK DID NOT RUN AS ASKED IS NOT A RESULT

Every response's `x_yamadori` is checked against the arm (`verify`). A row
whose record contradicts its arm -- tools gated when retrieval was off, hints
emitted when forced off, deep thinking chosen but never run, fan-out asked for
3 and delivered 2 -- is `stack_error`, as is HTTP != 200, finish != stop, an
exception, or a grader that could not run (stage `error`). None of those is
ever scored as a model failure (PROTOCOL rule 3).

RESUME

rows.jsonl is append-only. A rerun skips a (task, arm) pair whose LAST row is
pass or fail, and re-runs one whose last row is stack_error -- the resume bug
that once recorded a restart's 502s as done forever (livecodebench.load_done).
A stack_error that was only the grader (`stack_error_kind == "grader"`) is
re-graded from its saved answer rather than regenerated.

STALE CODE

Rows produced by a stack that has since been fixed are not deleted -- they are
marked `stale_code` (`--mark-stale RUN_ID --reason ...`). A stale row is
ignored by resume (its pair runs again) and by analyse.py (which counts it and
never mixes it with post-fix rows). The attempt counter keeps climbing across
them, so no answer file is overwritten.

The API key is read from --key-file and sent only as a header. It is never
printed and never written to any file this script produces.
"""
from __future__ import annotations

import argparse
import ast
import glob
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
MCP = os.path.join(ROOT, "mcp")
RESULTS = os.path.join(HERE, "results")
PROXY = os.environ.get("YAMADORI_PROXY", "http://127.0.0.1:1234")
LAYA = os.environ.get("LAYA_URL", "http://127.0.0.1:1237")
PY = sys.executable

sys.path.insert(0, HERE)

# ---------------------------------------------------------------- the arms --
ALL_OFF = {"retrieval": False, "hints": False, "investigate": False, "fanout": 1}
FANOUT_N = 3

ARMS: dict[str, dict] = {
    "A0": dict(ALL_OFF),
    "A1": dict(ALL_OFF, retrieval=True),
    "A2": dict(ALL_OFF, hints=True),
    "A3": dict(ALL_OFF, investigate=True),
    "A4": dict(ALL_OFF, fanout=FANOUT_N),
    "A5": {},                                   # nothing forced: selection decides
    "A6": {"retrieval": True, "hints": True, "investigate": True,
           "fanout": FANOUT_N},
    # The thinking-cap dimension: A0 with only the runaway breaker moved.
    "C8": dict(ALL_OFF, reasoning_cap=8192),
    "C32": dict(ALL_OFF, reasoning_cap=32768),
    "C128": dict(ALL_OFF, reasoning_cap=131072),
}
# The self-check arms: the twin's header, plus the client tool check_solution.
SELF_CHECK = {"S0": "A0", "S5": "A5", "S6": "A6"}
for _s, _a in SELF_CHECK.items():
    ARMS[_s] = dict(ARMS[_a])
ARM_NOTES = {
    "A0": "everything off (the model as it ships)",
    "A1": "retrieval only", "A2": "hints only",
    "A3": "deep thinking only (retrieval off)",
    "A4": f"fan-out only (N={FANOUT_N})",
    "A5": "all allowed, fired by the selection engine",
    "A6": "all forced on",
    "C8": "A0, thinking cap 8192", "C32": "A0, thinking cap 32768 (reference)",
    "C128": "A0, thinking cap 131072 (effectively unlimited)",
    "S0": "A0 + check_solution (compile only; hidden tests never run)",
    "S5": "A5 + check_solution (compile only; hidden tests never run)",
    "S6": "A6 + check_solution (compile only; hidden tests never run)",
}


def twin(arm: str) -> str:
    """The one-shot arm whose header this arm sends (itself for a one-shot)."""
    return SELF_CHECK.get(arm, arm)


# The client-side tool. Its description is a prompt (AGENTS.md: "Tool
# descriptions are prompts"): written as the trigger condition, it says what
# the result does and does NOT mean.
CHECK_TOOL_NAME = "check_solution"
CHECK_TOOL = {"type": "function", "function": {
    "name": CHECK_TOOL_NAME,
    "description": (
        "Does my code compile? Call this before answering to typecheck your "
        "complete solution: it runs the task's real compiler (tsc --strict "
        "against the pinned packages, or cargo for Rust) on exactly the code "
        "you pass and returns the compiler errors, or says it compiles. "
        "Answering without it means answering with code no compiler has seen. "
        "Use it for: 'does this compile', 'check my solution', 'fix the type "
        "errors', and again after each fix. Pass the whole solution, not a "
        "fragment. It does NOT run the hidden tests: a clean result means the "
        "code compiles, not that it is correct."),
    "parameters": {"type": "object", "properties": {"code": {
        "type": "string",
        "description": "The complete solution source, exactly as it would "
                       "appear in your final answer's code block."}},
        "required": ["code"]}}}
# A request may never be given less than the server's own generation timeout
# (llama-server -to 3600, and every generation timeout in mcp/): a 32k-token
# think alone can exceed 30 minutes on this card.
MIN_TIMEOUT = 3600.0
# Decode speed floor used to size a capped arm's timeout, tokens/s. Measured
# 16-40 tok/s on the shared lanes (fanout.run docstring); 12 leaves room for
# the dataset worker contending for the card.
SLOW_TOK_S = 12.0


def timeout_for(arm: str, base: float) -> float:
    """Per-request timeout: never under MIN_TIMEOUT, and long enough for the
    arm's thinking cap to be reached at SLOW_TOK_S plus the base allowance."""
    cap = ARMS[arm].get("reasoning_cap")
    t = max(base, MIN_TIMEOUT)
    if cap:
        t = max(t, cap / SLOW_TOK_S + 1800)
    return t


BODY_TIER = "max"          # every arm: the tier allows everything
EFFORTS = ("low", "medium", "xhigh")


def features(arm: str, effort: str) -> dict:
    """The X-Yamadori-Features object for an arm at a fixed effort."""
    return dict(ARMS[arm], effort=effort)


def arm_manifest(arm: str, effort: str) -> dict:
    m = {"features": features(arm, effort), "note": ARM_NOTES[arm]}
    if arm in SELF_CHECK:
        m.update(twin=SELF_CHECK[arm], client_tools=[CHECK_TOOL])
    return m


def expected(arm: str) -> dict:
    """What `x_yamadori` must show for this arm. Keys absent = not constrained."""
    forced = ARMS[arm]
    exp: dict = {"forced": sorted(k for k in ("hints", "investigate", "fanout")
                                  if k in forced)}
    # retrieval is not a selection flag: the tier's retrieval decides whether
    # the tool gate is consulted at all (proxy.prepare).
    exp["gate_consulted"] = bool(forced.get("retrieval", True))
    for k in ("hints", "investigate"):
        if k in forced:
            exp[k] = forced[k]
    if "fanout" in forced:
        exp["fanout_n"] = forced["fanout"]
    if "reasoning_cap" in forced:
        exp["reasoning_cap"] = min(max(int(forced["reasoning_cap"]), 1024), 131072)
    return exp


# ----------------------------------------------------------- verification --
def verify(arm: str, effort: str, x: dict | None, *, finish: str,
           content: str) -> list[str]:
    """Every way `x_yamadori` contradicts what the arm asked for. [] = clean."""
    if not isinstance(x, dict) or not x:
        return ["no x_yamadori on the response"]
    bad: list[str] = []
    exp = expected(arm)
    if x.get("tier") != BODY_TIER:
        bad.append(f"tier {x.get('tier')!r}, sent {BODY_TIER!r} (a ceiling capped it?)")
    if x.get("effort_sent") != effort:
        bad.append(f"effort_sent {x.get('effort_sent')!r}, header asked {effort!r}")
    if "reasoning_cap" in exp:
        got = (x.get("budget") or {}).get("reasoning_budget_tokens")
        if got != exp["reasoning_cap"]:
            bad.append(f"reasoning_budget_tokens {got}, arm sent cap "
                       f"{exp['reasoning_cap']} (header not honoured?)")
    gate = x.get("tools_gate")
    if exp["gate_consulted"] and not isinstance(gate, dict):
        bad.append("retrieval on but no tools_gate decision recorded")
    if not exp["gate_consulted"] and gate is not None:
        bad.append(f"retrieval forced off but tools_gate ran: {json.dumps(gate)[:160]}")

    sel = x.get("selection")
    if not isinstance(sel, dict):
        return bad + ["no selection record"]
    sig = sel.get("signals") or {}
    if sorted(sig.get("forced") or []) != exp["forced"]:
        bad.append(f"selection saw forced={sig.get('forced')}, arm forces "
                   f"{exp['forced']} (header not applied?)")
    if twin(arm) == "A5":
        allowed = sig.get("allowed") or {}
        if not (allowed.get("hints") and allowed.get("investigate")
                and int(allowed.get("fanout") or 1) > 1):
            bad.append(f"A5 must allow everything; selection saw allowed={allowed}")
    for k in ("hints", "investigate"):
        if k in exp and bool(sel.get(k)) != exp[k]:
            bad.append(f"selection.{k}={sel.get(k)}, arm forces {exp[k]}")
    if "fanout_n" in exp and int(sel.get("fanout_n") or 0) != exp["fanout_n"]:
        bad.append(f"selection.fanout_n={sel.get('fanout_n')}, arm forces "
                   f"{exp['fanout_n']}")

    # What the selection chose must then have HAPPENED.
    if not sel.get("hints") and (x.get("hints") or []):
        bad.append(f"hints off but {len(x['hints'])} emitted")
    inv = x.get("investigate")
    if sel.get("investigate"):
        if not isinstance(inv, dict):
            bad.append("deep thinking chosen, no investigate record")
        elif not inv.get("ran"):
            bad.append(f"deep thinking chosen but did not run: {inv.get('why')}")
    elif inv is not None:
        bad.append(f"deep thinking not chosen but an investigate record exists: "
                   f"{json.dumps(inv)[:160]}")
    n = int(sel.get("fanout_n") or 1)
    fan = x.get("fanout")
    if n > 1:
        # _fan_out runs only on a finished text answer; with no content the
        # answer is an extract failure, and the missing fan-out is not the arm's.
        if finish == "stop" and content.strip():
            if not isinstance(fan, dict):
                bad.append(f"fan-out {n} chosen, no fanout record")
            elif fan.get("error") or int(fan.get("n") or 0) != n:
                bad.append(f"fan-out asked {n}, delivered {fan.get('n')}"
                           + (f" ({fan.get('error')})" if fan.get("error") else ""))
    elif fan is not None:
        bad.append(f"fan-out 1 chosen but a fanout record exists: {json.dumps(fan)[:160]}")
    return bad


# ------------------------------------------------------------------ rows ---
def pair_key(task_id: str, arm: str) -> str:
    return f"{task_id}\x00{arm}"


def load_rows(path: str) -> tuple[list[dict], int]:
    rows, bad = [], 0
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    bad += 1
    return rows, bad


def last_rows(rows: list[dict]) -> dict[str, dict]:
    """The last row per (task, arm): a retry after a stack error is the outcome.
    Rows marked `stale_code` are ignored: they measured a stack since fixed."""
    out: dict[str, dict] = {}
    for r in rows:
        if "task" in r and "arm" in r and not r.get("stale_code"):
            out[pair_key(r["task"], r["arm"])] = r
    return out


def done_pairs(rows: list[dict]) -> set[str]:
    """Pairs whose LAST row is a scored outcome. stack_error is never done."""
    return {k for k, r in last_rows(rows).items()
            if r.get("outcome") in ("pass", "fail")}


def mark_stale(run_dir: str, reason: str) -> int:
    """Mark every not-yet-stale row of a run `stale_code`. Nothing is deleted:
    the file is rewritten whole (to a temp file, then replaced) with each row
    kept and flagged. Returns how many rows were newly marked."""
    path = os.path.join(run_dir, "rows.jsonl")
    rows, bad = load_rows(path)
    if bad:
        raise RuntimeError(f"{path}: {bad} unparseable lines; refusing to rewrite")
    n = 0
    stamp = {"reason": reason, "marked_at": time.time()}
    for r in rows:
        if not r.get("stale_code"):
            r["stale_code"] = stamp
            n += 1
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    return n


def append_row(path: str, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ------------------------------------------------------------ the request --
def post(url: str, key: str, body: dict, feats: dict,
         timeout: float) -> tuple[int, dict | str]:
    headers = {"Content-Type": "application/json", "Connection": "close",
               "Authorization": f"Bearer {key}",
               "X-Yamadori-Features": json.dumps(feats, sort_keys=True)}
    req = urllib.request.Request(f"{url}/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw, status = r.read().decode("utf-8", "replace"), r.status
    except urllib.error.HTTPError as e:
        raw, status = e.read().decode("utf-8", "replace"), e.code
    try:
        return status, json.loads(raw)
    except ValueError:
        return status, raw


def _grade_safe(grade_fn, task: dict, text: str) -> dict:
    try:
        g = grade_fn(task, text) or {}
    except Exception as e:                                       # noqa: BLE001
        g = {"passed": False, "stage": "error",
             "detail": f"grade() raised {type(e).__name__}: {e}"}
    return {"passed": bool(g.get("passed")), "stage": g.get("stage"),
            "detail": str(g.get("detail") or "")[:500]}


def classify(status, finish: str, mismatch: list[str], grade: dict | None,
             error: str | None, checker_error: bool = False
             ) -> tuple[str, str | None]:
    """(outcome, stack_error_kind)."""
    if error:
        return "stack_error", "exception"
    if status != 200:
        return "stack_error", "http"
    if finish != "stop":
        return "stack_error", "finish"
    if mismatch:
        return "stack_error", "mismatch"
    if checker_error:
        # check_solution could not RUN: the model was told something untrue
        # about its code, so the row measures the harness, not the arm.
        return "stack_error", "checker"
    if grade is None or grade.get("stage") == "error":
        return "stack_error", "grader"
    return ("pass" if grade["passed"] else "fail"), None


def run_one(task: dict, arm: str, cfg: dict, grade_fn, attempt: int,
            check_fn=None) -> dict:
    """One (task, arm) generation, verified and graded. Never raises."""
    if arm in SELF_CHECK:
        return run_self_check(task, arm, cfg, grade_fn, attempt,
                              check_fn or _public_check)
    feats = features(arm, cfg["effort"])
    body = {"model": "yamadori", "reasoning_effort": BODY_TIER,
            "messages": [{"role": "user", "content": task["prompt"]}],
            "max_tokens": cfg["max_tokens"], "temperature": cfg["temperature"]}
    row: dict = {"run_id": cfg["run_id"], "task": task["id"],
                 "domain": task.get("domain"),
                 "contaminated": bool(task.get("contaminated")),
                 "difficulty": task.get("difficulty"),
                 "needs_retrieval": task.get("needs_retrieval"),
                 "suite": cfg.get("suite"),
                 "arm": arm, "effort": cfg["effort"], "attempt": attempt,
                 "features_sent": feats, "started_at": time.time()}
    t0 = time.time()
    status, d, error = None, None, None
    try:
        status, d = post(cfg["url"], cfg["key"], body, feats,
                         timeout_for(arm, cfg["timeout"]))
    except Exception as e:                                       # noqa: BLE001
        error = f"{type(e).__name__}: {e}"[:500]
    row["seconds"] = round(time.time() - t0, 2)
    row["http_status"] = status
    content, reasoning, finish, x, usage = "", "", "", None, {}
    if isinstance(d, dict):
        ch = (d.get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        finish = ch.get("finish_reason") or ""
        x = d.get("x_yamadori")
        usage = d.get("usage") or {}
    elif d is not None and error is None and status != 200:
        error = None                     # a non-JSON error body: status says it
    row["finish_reason"] = finish
    row["usage"] = usage              # whole block: details vary by server build
    # The proxy's usage has no reasoning/answer split, so thinking length is
    # recorded from what came back: reasoning_content characters, any
    # reasoning_tokens a server reports, and whether the breaker fired.
    row["reasoning_chars"] = len(reasoning)
    row["reasoning_tokens"] = ((usage.get("completion_tokens_details") or {})
                               .get("reasoning_tokens"))
    row["hit_thinking_cap"] = "Thinking budget reached" in reasoning
    row["x_yamadori"] = x
    row["content_chars"] = len(content)
    row["empty_content"] = not content.strip()
    if status is not None and status != 200:
        row["error_body"] = (json.dumps(d) if isinstance(d, dict) else str(d))[:500]
    mismatch = ([] if error or status != 200 else
                verify(arm, cfg["effort"], x, finish=finish, content=content))
    row["mismatch"] = mismatch

    # The answer goes to its own file, whatever happened, so a grader bug or
    # a stack error can be inspected (and re-graded) without a regeneration.
    ans_rel = f"answers/{task['id']}__{arm}__{attempt}.json"
    with open(os.path.join(cfg["run_dir"], ans_rel), "w", encoding="utf-8") as f:
        json.dump({"task": task["id"], "arm": arm, "attempt": attempt,
                   "content": content, "reasoning_content": reasoning,
                   "fanout": (d.get("_fanout") if isinstance(d, dict) else None),
                   "error": error}, f, ensure_ascii=False, indent=1)
    row["answer_file"] = ans_rel

    grade = None
    if not error and status == 200 and finish == "stop" and not mismatch:
        row["grade_started_at"] = time.time()
        grade = _grade_safe(grade_fn, task, content)
        row["grade_seconds"] = round(time.time() - row["grade_started_at"], 2)
    row["grade"] = grade
    row["outcome"], row["stack_error_kind"] = classify(status, finish, mismatch,
                                                       grade, error)
    if error:
        row["error"] = error
    return row


# ------------------------------------------------------- the self-check --
def _public_check(task: dict, code: str) -> dict:
    import grade
    return grade.public_check(task, code)


def _parse(d) -> dict:
    """The fields of one chat completion the runner reads."""
    out = {"content": "", "reasoning": "", "finish": "", "x": None, "usage": {},
           "calls": []}
    if isinstance(d, dict):
        ch = (d.get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        out.update(content=msg.get("content") or "",
                   reasoning=msg.get("reasoning_content") or "",
                   finish=ch.get("finish_reason") or "",
                   x=d.get("x_yamadori"), usage=d.get("usage") or {},
                   calls=list(msg.get("tool_calls") or []))
    return out


def check_message(n: int, res: dict, lang: str, repeat_of: int | None) -> str:
    """The tool message for check #n. Says the situation, whether retrying
    helps, and what to do next (AGENTS.md: failure returns carry the next
    step) -- and, every time, that the hidden tests were not run."""
    head = f"check_solution #{n}"
    again = (f" This is the same code as check #{repeat_of}; the result is "
             f"unchanged." if repeat_of else "")
    tail = ("This check only compiles your code. It did not run the hidden "
            "tests, so it cannot tell you the code is correct.")
    if res.get("checker_error"):
        return (f"{head}: CHECKER UNAVAILABLE. The harness could not run the "
                f"compiler ({'; '.join(res.get('errors') or [])[:300]}). This "
                f"is not a problem with your code and retrying will not help. "
                f"Answer with your best complete solution in a single ```{lang} "
                f"code block.")
    if res.get("stage") == "extract":
        return (f"{head}: NO CODE.{again} {' '.join(res.get('errors') or [])} "
                f"Retryable: yes -- pass the complete solution source as "
                f"`code`, or answer.")
    if res.get("ok"):
        return (f"{head}: COMPILES.{again} {res.get('tool')} reports no errors "
                f"in your code. {tail} If you are done, answer with the "
                f"complete solution in a single ```{lang} code block.")
    errs = "\n".join(res.get("errors") or [])
    more = ("\n(first lines only; more errors follow)"
            if res.get("truncated") else "")
    return (f"{head}: DOES NOT COMPILE.{again} {res.get('n_errors')} error(s) "
            f"from {res.get('tool')}:\n{errs}{more}\nRetryable: yes -- fix "
            f"these and call check_solution again with the complete solution, "
            f"or answer. {tail}")


_LANG = {"ts": "ts", "tc_tests": "ts", "react": "tsx", "rust": "rust", "tsl": "js"}


def _run_check_call(c: dict, task: dict, check_fn, n: int, lang: str,
                    seen_code: dict) -> tuple[str, dict | None, float]:
    """One tool call from the model -> (tool message, check record, seconds)."""
    fn = (c.get("function") or {}).get("name")
    raw = (c.get("function") or {}).get("arguments")
    if fn != CHECK_TOOL_NAME:
        return (f"UNKNOWN TOOL `{fn}`: this client runs only {CHECK_TOOL_NAME}. "
                f"Retryable: call {CHECK_TOOL_NAME} with your complete "
                f"solution, or answer."), None, 0.0
    try:
        args = raw if isinstance(raw, dict) else json.loads(raw or "{}")
    except (TypeError, ValueError):
        args = None
    code = args.get("code") if isinstance(args, dict) else None
    if not isinstance(code, str):
        rec = {"ok": False, "stage": "arguments", "n_errors": 1,
               "checker_error": False, "repeat_of": None, "code_chars": 0,
               "seconds": 0.0}
        return (f"check_solution #{n}: BAD ARGUMENTS. It takes "
                f'{{"code": string}}, the complete solution; got '
                f"{str(raw)[:120]!r}. Retryable: yes."), rec, 0.0
    key = hashlib.sha1(code.strip().encode("utf-8")).hexdigest()
    rep = seen_code.get(key)
    t0 = time.time()
    res = check_fn(task, code)
    secs = time.time() - t0
    seen_code.setdefault(key, n)
    rec = {"ok": bool(res.get("ok")), "stage": res.get("stage"),
           "n_errors": int(res.get("n_errors") or 0),
           "checker_error": bool(res.get("checker_error")),
           "repeat_of": rep, "code_chars": len(code),
           "seconds": round(secs, 3)}
    return check_message(n, res, lang, rep), rec, secs


def run_self_check(task: dict, arm: str, cfg: dict, grade_fn, attempt: int,
                   check_fn) -> dict:
    """A self-check arm: request until the model answers without a tool call,
    running check_solution for it in between. NO round cap: the proxy's
    context_full landing ends a conversation that no longer fits. Every
    request carries the arm's header and is verified. Never raises."""
    feats = features(arm, cfg["effort"])
    lang = _LANG.get((task.get("grader") or {}).get("kind"), "")
    messages: list[dict] = [{"role": "user", "content": task["prompt"]}]
    row: dict = {"run_id": cfg["run_id"], "task": task["id"],
                 "domain": task.get("domain"),
                 "contaminated": bool(task.get("contaminated")),
                 "difficulty": task.get("difficulty"),
                 "needs_retrieval": task.get("needs_retrieval"),
                 "suite": cfg.get("suite"),
                 "arm": arm, "twin": twin(arm), "self_check": True,
                 "effort": cfg["effort"], "attempt": attempt,
                 "features_sent": feats, "started_at": time.time()}
    t0 = time.time()
    status, error, last_d = None, None, None
    p = _parse(None)
    usage_rounds: list[dict] = []
    x_rounds: list = []
    rounds: list[dict] = []
    mismatch: list[str] = []
    checks: list[dict] = []
    other_calls: list[str] = []
    seen_code: dict[str, int] = {}
    model_s = check_s = 0.0
    while True:
        body = {"model": "yamadori", "reasoning_effort": BODY_TIER,
                "messages": messages, "tools": [CHECK_TOOL],
                "max_tokens": cfg["max_tokens"], "temperature": cfg["temperature"]}
        t1 = time.time()
        try:
            status, last_d = post(cfg["url"], cfg["key"], body, feats,
                                  timeout_for(arm, cfg["timeout"]))
        except Exception as e:                                   # noqa: BLE001
            error = f"{type(e).__name__}: {e}"[:500]
        model_s += time.time() - t1
        if error or status != 200:
            break
        p = _parse(last_d)
        usage_rounds.append(p["usage"])
        x_rounds.append(p["x"])
        rounds.append({"content": p["content"], "reasoning_content": p["reasoning"],
                       "finish_reason": p["finish"], "tool_calls": p["calls"]})
        mismatch += [f"round {len(rounds)}: {m}" for m in
                     verify(arm, cfg["effort"], p["x"], finish=p["finish"],
                            content=p["content"])]
        if not p["calls"] or p["finish"] == "length" or mismatch:
            break              # an answer, a budget event, or a void row
        calls = []
        for i, c in enumerate(p["calls"]):
            c = dict(c)
            if not c.get("id"):
                c["id"] = f"call_{len(rounds)}_{i}"
            c.setdefault("type", "function")
            calls.append(c)
        # OpenAI continuation: the assistant turn with its calls, then one
        # tool message per call. reasoning_content is not echoed back.
        messages = messages + [{"role": "assistant", "content": p["content"],
                                "tool_calls": calls}]
        for c in calls:
            text, rec, secs = _run_check_call(c, task, check_fn, len(checks) + 1,
                                              lang, seen_code)
            check_s += secs
            if rec is None:
                other_calls.append(str((c.get("function") or {}).get("name")))
            else:
                checks.append(dict(rec, round=len(rounds)))
            messages.append({"role": "tool", "tool_call_id": c["id"],
                             "content": text})
    checker_error = any(c["checker_error"] for c in checks)
    row["seconds"] = round(time.time() - t0, 2)
    row["model_seconds"] = round(model_s, 2)
    row["check_seconds"] = round(check_s, 2)
    row["http_status"] = status
    row["rounds"] = len(rounds)
    row["check_rounds"] = len(checks)
    row["check_results"] = checks
    row["other_tool_calls"] = other_calls
    content, finish = p["content"], p["finish"]
    row["finish_reason"] = finish
    tot = {k: sum(int((u or {}).get(k) or 0) for u in usage_rounds)
           for k in ("prompt_tokens", "completion_tokens", "total_tokens", "hops")}
    row["usage"] = dict(tot, rounds=len(usage_rounds))
    row["usage_rounds"] = usage_rounds
    rt = [((u or {}).get("completion_tokens_details") or {}).get("reasoning_tokens")
          for u in usage_rounds]
    rt = [x for x in rt if isinstance(x, int)]
    row["reasoning_chars"] = sum(len(r["reasoning_content"]) for r in rounds)
    row["reasoning_tokens"] = sum(rt) if rt else None
    row["hit_thinking_cap"] = any("Thinking budget reached" in r["reasoning_content"]
                                  for r in rounds)
    row["x_yamadori"] = p["x"]
    row["x_rounds"] = x_rounds
    row["content_chars"] = len(content)
    row["empty_content"] = not content.strip()
    if status is not None and status != 200:
        row["error_body"] = (json.dumps(last_d) if isinstance(last_d, dict)
                             else str(last_d))[:500]
    row["mismatch"] = [] if error or status != 200 else mismatch

    ans_rel = f"answers/{task['id']}__{arm}__{attempt}.json"
    with open(os.path.join(cfg["run_dir"], ans_rel), "w", encoding="utf-8") as f:
        json.dump({"task": task["id"], "arm": arm, "attempt": attempt,
                   "content": content, "reasoning_content": p["reasoning"],
                   "fanout": (last_d.get("_fanout") if isinstance(last_d, dict)
                              else None),
                   "rounds": rounds, "messages": messages, "error": error},
                  f, ensure_ascii=False, indent=1)
    row["answer_file"] = ans_rel

    grade = None
    if (not error and status == 200 and finish == "stop" and not row["mismatch"]
            and not checker_error):
        row["grade_started_at"] = time.time()
        grade = _grade_safe(grade_fn, task, content)
        row["grade_seconds"] = round(time.time() - row["grade_started_at"], 2)
        # Did the answer it GAVE compile? It need not be the code it checked.
        fc = check_fn(task, content) or {}
        row["final_public_check"] = {"ok": bool(fc.get("ok")),
                                     "stage": fc.get("stage"),
                                     "n_errors": int(fc.get("n_errors") or 0),
                                     "checker_error": bool(fc.get("checker_error"))}
    row["grade"] = grade
    # A loop stopped mid-conversation by a mismatch ends on a tool-call
    # round; the mismatch, not that finish, is why the row is void.
    fin = "stop" if row["mismatch"] and finish == "tool_calls" else finish
    row["outcome"], row["stack_error_kind"] = classify(
        status, fin, row["mismatch"], grade, error, checker_error)
    if error:
        row["error"] = error
    return row


def regrade(prev: dict, task: dict, cfg: dict, grade_fn, attempt: int) -> dict:
    """A grader-only stack error: grade the saved answer again, no generation."""
    with open(os.path.join(cfg["run_dir"], prev["answer_file"]),
              encoding="utf-8") as f:
        content = json.load(f).get("content") or ""
    row = dict(prev, attempt=attempt, regraded_from=prev.get("attempt"),
               started_at=time.time())
    grade = _grade_safe(grade_fn, task, content)
    row["grade"] = grade
    row["outcome"], row["stack_error_kind"] = classify(
        prev.get("http_status"), prev.get("finish_reason") or "",
        prev.get("mismatch") or [], grade, None)
    return row


# ------------------------------------------------------------- the runner --
SUITES = {"core": ("tasks.jsonl",), "react": ("tasks_react.jsonl",),
          "tc": ("tasks_type_challenges.jsonl",)}
SUITES["all"] = SUITES["core"] + SUITES["react"] + SUITES["tc"]
DEFAULT_SUITE = "core"


def load_suite(name: str, here: str = HERE) -> list[dict]:
    """The task rows of a suite. Every row must say whether it is
    contaminated -- analyse.py keeps contaminated rows out of the headline, so
    a row that does not say is refused rather than guessed."""
    out: list[dict] = []
    for fn in SUITES[name]:
        with open(os.path.join(here, fn), encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        bad = [r.get("id") for r in rows if not isinstance(r.get("contaminated"), bool)]
        if bad:
            raise ValueError(f"{fn}: rows without a boolean `contaminated`: {bad[:5]}")
        out += rows
    ids = [t["id"] for t in out]
    if len(ids) != len(set(ids)):
        raise ValueError(f"suite {name}: duplicate task ids")
    return out


def manifest_clash(old: dict, fixed: dict) -> dict:
    """Conditions a run dir was made with that differ from this invocation.
    A manifest from before --suite existed was a core run."""
    return {k: (old.get(k, DEFAULT_SUITE if k == "suite" else None), v)
            for k, v in fixed.items()
            if old.get(k, DEFAULT_SUITE if k == "suite" else None) != v}


def select_tasks(tasks: list[dict], per_domain: int | None,
                 ids: list[str] | None, domains: list[str] | None) -> list[dict]:
    out = tasks
    if domains:
        out = [t for t in out if t["domain"] in domains]
    if ids:
        want = set(ids)
        out = [t for t in out if t["id"] in want]
    if per_domain:
        seen: dict[str, int] = {}
        keep = []
        for t in out:
            if seen.get(t["domain"], 0) < per_domain:
                keep.append(t)
                seen[t["domain"]] = seen.get(t["domain"], 0) + 1
        out = keep
    return out


def run_pairs(tasks: list[dict], arms: list[str], cfg: dict, grade_fn,
              log=print, check_fn=None) -> list[dict]:
    """Run every not-done (task, arm) pair, task-major, sequentially."""
    os.makedirs(os.path.join(cfg["run_dir"], "answers"), exist_ok=True)
    path = os.path.join(cfg["run_dir"], "rows.jsonl")
    rows, _bad = load_rows(path)
    done = done_pairs(rows)
    last = last_rows(rows)
    attempts: dict[str, int] = {}
    for r in rows:
        k = pair_key(r.get("task"), r.get("arm"))
        attempts[k] = max(attempts.get(k, 0), int(r.get("attempt") or 0))
    todo = [(t, a) for t in tasks for a in arms if pair_key(t["id"], a) not in done]
    log(f"  {len(tasks)} tasks x {len(arms)} arms = {len(tasks) * len(arms)} pairs; "
        f"{len(tasks) * len(arms) - len(todo)} already done, {len(todo)} to run")
    new: list[dict] = []
    for i, (t, a) in enumerate(todo, 1):
        keep_quiet(cfg["run_id"])
        k = pair_key(t["id"], a)
        attempt = attempts.get(k, 0) + 1
        prev = last.get(k)
        if (prev and prev.get("stack_error_kind") == "grader"
                and prev.get("answer_file")
                and os.path.exists(os.path.join(cfg["run_dir"], prev["answer_file"]))):
            row = regrade(prev, t, cfg, grade_fn, attempt)
        else:
            row = run_one(t, a, cfg, grade_fn, attempt, check_fn)
        append_row(path, row)
        new.append(row)
        g = row.get("grade") or {}
        log(f"  [{i}/{len(todo)}] {t['id']:<6} {a}  {row['outcome']:<11} "
            f"{row.get('seconds', 0):>7.1f}s  "
            + (f"{g.get('stage')}" if g else "")
            + (f"  rounds={row.get('rounds')} checks=[" + ",".join(
                "ok" if c["ok"] else f"{c['stage']}:{c['n_errors']}"
                for c in row.get("check_results") or []) + "]"
               if a in SELF_CHECK else "")
            + (f"  {row['stack_error_kind']}: "
               + ("; ".join(row.get("mismatch") or [])
                  or row.get("error") or row.get("finish_reason") or
                  str(row.get("http_status")))[:200]
               if row["outcome"] == "stack_error" else ""))
    return new


# ------------------------------------------------------------- preflight --
def _ps(cmd: str, timeout: float = 60) -> str:
    p = subprocess.run(["powershell", "-NoProfile", "-NonInteractive",
                        "-Command", cmd], capture_output=True, text=True,
                       timeout=timeout)
    return (p.stdout or "").strip()


def listeners(port: int) -> list[dict]:
    """[{pid, created (unix), cmd}] for every process listening on `port`."""
    out = _ps(
        f"$c = Get-NetTCPConnection -LocalPort {port} -State Listen "
        "-ErrorAction SilentlyContinue; "
        "$ids = @($c | ForEach-Object { $_.OwningProcess } | Sort-Object -Unique); "
        "$r = foreach ($i in $ids) { $p = Get-CimInstance Win32_Process "
        "-Filter \"ProcessId=$i\"; [pscustomobject]@{pid=[int]$i; "
        "created=([DateTimeOffset]$p.CreationDate).ToUnixTimeSeconds(); "
        "cmd=[string]$p.CommandLine} }; ConvertTo-Json -Compress -InputObject @($r)")
    if not out:
        return []
    d = json.loads(out)
    return d if isinstance(d, list) else [d]


def import_closure(entry: str, mcp_dir: str = MCP) -> list[str]:
    """Every mcp/*.py the entry script can import, transitively (function-level
    imports included). The proxy's code, as opposed to any file in mcp/."""
    mods = {os.path.basename(p)[:-3] for p in glob.glob(os.path.join(mcp_dir, "*.py"))}
    seen: set[str] = set()
    todo = [entry]
    while todo:
        m = todo.pop()
        if m in seen or m not in mods:
            continue
        seen.add(m)
        try:
            tree = ast.parse(open(os.path.join(mcp_dir, m + ".py"),
                                  encoding="utf-8").read())
        except (OSError, SyntaxError):
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                todo += [a.name.split(".")[0] for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.module and not n.level:
                todo.append(n.module.split(".")[0])
    return sorted(seen)


def stale_files(created: float, entry: str, mcp_dir: str = MCP) -> list[tuple[str, float]]:
    """Files in the entry's import closure modified after the process started."""
    out = []
    for m in import_closure(entry, mcp_dir):
        p = os.path.join(mcp_dir, m + ".py")
        mt = os.path.getmtime(p)
        if mt > created:
            out.append((m + ".py", mt))
    return sorted(out, key=lambda x: -x[1])


def _check_listener_and_freshness() -> list[tuple[bool, str, str]]:
    res = []
    try:
        ls = listeners(1234)
    except Exception as e:                                       # noqa: BLE001
        return [(False, "one_listener", f"could not ask Windows: {e}"),
                (False, "proxy_fresh", "no listener information")]
    res.append((len(ls) == 1, "one_listener",
                f"{len(ls)} process(es) on :1234: "
                + ", ".join(f"pid {x['pid']}" for x in ls)))
    if len(ls) != 1:
        return res + [(False, "proxy_fresh", "needs exactly one listener")]
    p = ls[0]
    cmd = p.get("cmd") or ""
    entry = next((os.path.basename(tok.strip('"'))[:-3] for tok in cmd.split()
                  if tok.strip('"').endswith(".py")), "server")
    stale = stale_files(float(p["created"]), entry)
    started = time.strftime("%H:%M:%S", time.localtime(float(p["created"])))
    res.append((not stale, "proxy_fresh",
                f"pid {p['pid']} ({entry}.py) started {started}; "
                + ("stale: " + ", ".join(
                    f"{n} modified {time.strftime('%H:%M:%S', time.localtime(t))}"
                    for n, t in stale[:8]) if stale else
                   f"all {len(import_closure(entry))} modules it imports are older")))
    return res


def _check_laya() -> tuple[bool, str, str]:
    try:
        with urllib.request.urlopen(f"{LAYA}/heads", timeout=15) as r:
            d = json.load(r)
        ri = (d.get("tasks") or {}).get("route_in") or {}
        return (bool(ri.get("loaded")), "laya_route_in",
                f"route_in loaded={ri.get('loaded')} kind={ri.get('kind')}")
    except Exception as e:                                       # noqa: BLE001
        return False, "laya_route_in", f"{LAYA}/heads: {type(e).__name__}: {e}"


def _check_hints_cache() -> tuple[bool, str, str]:
    path = os.environ.get("YAMADORI_HINT_CACHE",
                          os.path.join(ROOT, "index", "hints.npz"))
    try:
        import numpy as np
        mat = np.load(path, allow_pickle=False)["mat"]
        norms = np.linalg.norm(mat.astype("float32"), axis=1)
        zero = int((norms <= 1e-6).sum())
        return (mat.shape[0] > 0 and zero == 0, "hints_cache",
                f"{mat.shape[0]} vectors, {zero} zero-norm, "
                f"min norm {float(norms.min()) if len(norms) else 0:.4f}")
    except Exception as e:                                       # noqa: BLE001
        return False, "hints_cache", f"{path}: {type(e).__name__}: {e}"


# The packages the graded tasks are pinned to (grade.py). Above this share of
# zero vectors an index is an outage, not a fallback (PROTOCOL rule 2).
PACKAGES = {"typegpu": "0.12.5", "three": "0.185.1"}
MAX_ZERO_SHARE = 0.01


def _check_package_vectors() -> tuple[bool, str, str]:
    try:
        if MCP not in sys.path:
            sys.path.insert(0, MCP)
        import deps
        import domains
        held = domains.held_sources()
        bits, ok = [], True
        for name, ver in PACKAGES.items():
            db = next((d for v, d in held.get(name, []) if v == ver), None)
            if not db:
                ok = False
                bits.append(f"{name}@{ver} not held (have {[v for v, _ in held.get(name, [])]})")
                continue
            h = deps.index_health(db, measure_bytes=False)
            share = h["zero_vectors"] / h["chunks"] if h["chunks"] else 1.0
            good = (h["ok"] and h["vectors"] in ("all", "partial")
                    and h.get("norm_ok") and share <= MAX_ZERO_SHARE)
            ok = ok and bool(good)
            bits.append(f"{name}@{ver}: {h['chunks']} chunks, {h['zero_vectors']} "
                        f"zero, norm_ok={h.get('norm_ok')}")
        return ok, "package_vectors", "; ".join(bits)
    except Exception as e:                                       # noqa: BLE001
        return False, "package_vectors", f"{type(e).__name__}: {e}"


_SHINGLE = 8
# A shared 8-word run is a SUSPECT, not a verdict: a recipe can share a common
# idiom with a reference without carrying its answer. Each hit is judged once,
# keyed by task and the recipe's text hash (so an edited recipe is judged
# again), and the verdict is kept here with its reason.
LEAK_REVIEW = os.path.join(HERE, "leak_review.json")


def _shingles(text: str) -> set[tuple[str, ...]]:
    import re as _re
    w = _re.findall(r"[a-z0-9]+", (text or "").lower())
    return {tuple(w[i:i + _SHINGLE]) for i in range(len(w) - _SHINGLE + 1)}


def _check_no_task_leak() -> tuple[bool, str, str]:
    """No benchmark task text in the hints corpus.

    The hints arm would otherwise be shown the question it is graded on. Found
    as a live risk on 2026-09-22: TypeHero's site imports ~188 type-challenges
    READMEs, and type-challenges is becoming a benchmark subset. Any recipe or
    evidence quote sharing an 8-word run with a task prompt fails preflight,
    and the offending rows are named so they can be rejected on /dash.
    """
    try:
        tasks = []
        for f in glob.glob(os.path.join(HERE, "tasks*.jsonl")):
            with open(f, encoding="utf-8") as fh:
                tasks += [json.loads(line) for line in fh if line.strip()]
        task_sh: dict[tuple, str] = {}
        for t in tasks:
            for s in _shingles(t.get("prompt") or ""):
                task_sh.setdefault(s, t.get("id") or "?")
        # The ANSWERS too, not only the questions. A recipe can carry a task's
        # solution without repeating its prompt -- TypeHero's pages hold
        # solutions to the type-challenges tasks, which is the exact case.
        refs = 0
        live = {t.get("id") for t in tasks}
        for pat in ("tasks/*/reference.*", "tasks_type_challenges/*/reference.*"):
            for f in glob.glob(os.path.join(HERE, pat)):
                tid = os.path.basename(os.path.dirname(f))
                if tid not in live:
                    continue          # excluded or not indexed: never asked
                with open(f, encoding="utf-8", errors="replace") as fh:
                    for s in _shingles(fh.read()):
                        task_sh.setdefault(s, f"{tid} (reference)")
                refs += 1
        hits, idioms = [], 0
        review = {}
        if os.path.exists(LEAK_REVIEW):
            with open(LEAK_REVIEW, encoding="utf-8") as fh:
                review = json.load(fh)
        for f in glob.glob(os.path.join(ROOT, "bench", "recipes", "*.jsonl")):
            with open(f, encoding="utf-8") as fh:
                for i, line in enumerate(fh):
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    if r.get("_state") == "reject":
                        continue
                    text = f"{r.get('recipe') or ''} {r.get('evidence') or ''}"
                    rh = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
                    for tid in sorted({task_sh[s] for s in _shingles(text)
                                       if s in task_sh}):
                        bare = tid.replace(" (reference)", "")
                        v = (review.get(f"{bare}|{rh}") or {}).get("verdict")
                        if v == "idiom":
                            idioms += 1
                            continue
                        hits.append(f"{os.path.basename(f)}:{i} [{rh}] ~ {tid}"
                                    + (" (reviewed: leak -- exclude the task)"
                                       if v == "leak" else " (unreviewed)"))
        return (not hits, "task_leak",
                f"{len(tasks)} task prompts + {refs} reference solutions "
                f"vs the hints corpus: "
                + (f"{idioms} reviewed as shared idiom; " if idioms else "")
                + (f"{len(hits)} leaking row(s): {', '.join(hits[:8])}. Review "
                   f"each in {os.path.relpath(LEAK_REVIEW, ROOT)} as "
                   f"'task|hash': leak (and exclude the task) or idiom"
                   if hits else "no other shared 8-word runs"))
    except Exception as e:                                       # noqa: BLE001
        return False, "task_leak", f"{type(e).__name__}: {e}"


def _check_cmd(name: str, cmd: list[str], timeout: float) -> tuple[bool, str, str]:
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout,
                           env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    except subprocess.TimeoutExpired:
        return False, name, f"timed out after {timeout:.0f}s"
    out = ((p.stdout or "") + (p.stderr or "")).strip().splitlines()
    # The failing lines, not just the tail: "17/18 passed" says nothing about
    # WHICH check failed, and that is the part the operator acts on.
    fails = [ln.strip() for i, ln in enumerate(out)
             if ln.strip().startswith("FAIL") or "FAIL:" in ln
             or (i and out[i - 1].strip().startswith("FAIL")
                 and ln.strip().startswith("<-"))]
    tail = " | ".join(ln.strip() for ln in out[-2:] if ln.strip())
    return (p.returncode == 0, name,
            f"rc={p.returncode} in {time.time() - t0:.0f}s: {tail[:300]}"
            + (" || " + " | ".join(fails)[:900] if fails else ""))


def preflight(key_file: str, log=print) -> list[tuple[bool, str, str]]:
    """Every precondition, cheapest first. All are run; the caller decides."""
    checks: list[tuple[bool, str, str]] = []

    def add(c):
        checks.append(c)
        log(("  pass  " if c[0] else "  FAIL  ") + f"{c[1]:<16} {c[2]}")

    for c in _check_listener_and_freshness():
        add(c)
    add(_check_laya())
    add(_check_hints_cache())
    add(_check_package_vectors())
    add(_check_no_task_leak())
    add(_check_cmd("run_tests", [PY, os.path.join("scripts", "run_tests.py")], 3600))
    add(_check_cmd("live_stack", [PY, os.path.join("mcp", "test_live_stack.py"),
                                  "--live", "--key-file", key_file,
                                  "--only", "health,tiers"], 3600))
    return checks


# ----------------------------------------------------------- worker quiet --
# The worker's gpu lane runs extractions on the SAME model, in another slot.
# That does not change what the model answers, but it moves every seconds and
# tokens/s number in the row, differently for each row. So a run pauses the
# lane (mcp/jobs.py pause(): nothing is killed; a job already running is
# waited for; the pause expires by itself if this process dies) and refreshes
# the pause as it goes.
PAUSE_TTL = 1800


def _jobs():
    if MCP not in sys.path:
        sys.path.insert(0, MCP)
    import jobs
    return jobs


def quiet_worker(run_id: str, wait_s: float = 3600, log=print) -> tuple[bool, str, str]:
    try:
        j = _jobs()
        j.pause("gpu", by=f"bench/domain/run.py {run_id}",
                why="benchmark measuring seconds and tokens/s", ttl_seconds=PAUSE_TTL)
        t0 = time.time()
        n = j.running("gpu")
        while n and time.time() - t0 < wait_s:
            log(f"  waiting for {n} running gpu job(s) to finish "
                f"({time.time() - t0:.0f}s)")
            time.sleep(15)
            j.pause("gpu", by=f"bench/domain/run.py {run_id}",
                    why="benchmark measuring seconds and tokens/s",
                    ttl_seconds=PAUSE_TTL)
            n = j.running("gpu")
        return (n == 0, "worker_quiet",
                f"gpu lane paused; {n} gpu job(s) still running after "
                f"{time.time() - t0:.0f}s" if n else
                f"gpu lane paused and idle (waited {time.time() - t0:.0f}s)")
    except Exception as e:                                       # noqa: BLE001
        return False, "worker_quiet", f"{type(e).__name__}: {e}"


def keep_quiet(run_id: str) -> None:
    try:
        _jobs().pause("gpu", by=f"bench/domain/run.py {run_id}",
                      why="benchmark measuring seconds and tokens/s",
                      ttl_seconds=PAUSE_TTL)
    except Exception:                                            # noqa: BLE001
        pass


def release_worker() -> None:
    try:
        _jobs().resume("gpu")
    except Exception:                                            # noqa: BLE001
        pass


# ------------------------------------------------------------------- main --
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--key-file", default=None,
                    help="file holding the API key (required to run)")
    ap.add_argument("--mark-stale", default=None, metavar="RUN_ID",
                    help="flag every row of results/RUN_ID as stale_code and exit")
    ap.add_argument("--reason", default="",
                    help="why the rows are stale (with --mark-stale)")
    ap.add_argument("--effort", default="medium", choices=EFFORTS)
    ap.add_argument("--suite", default=DEFAULT_SUITE, choices=sorted(SUITES),
                    help="core = tasks.jsonl (default); react; tc "
                         "(type-challenges, contaminated); all")
    # The default is the one-shot set it always was; S arms are named.
    ap.add_argument("--arms", default=",".join(a for a in ARMS
                                               if a not in SELF_CHECK),
                    help=f"comma-separated; known: {','.join(ARMS)}")
    ap.add_argument("--run-id", default=None,
                    help="results/<run-id>/; rerun with the same id to resume")
    ap.add_argument("--per-domain", type=int, default=None,
                    help="first N tasks of each domain (smoke runs)")
    ap.add_argument("--tasks", default="", help="comma-separated task ids")
    ap.add_argument("--domains", default="", help="comma-separated domains")
    ap.add_argument("--max-tokens", type=int, default=4096,
                    help="the ANSWER allowance, same for every arm "
                         "(tiers.budget adds the thinking cap on top)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=7200.0,
                    help="seconds per request, never below 3600; A6 is "
                         "investigation + fan-out; C arms get cap/12 tok/s + 1800")
    ap.add_argument("--proxy", default=PROXY)
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--override-preflight", default="",
                    help="comma-separated preflight checks allowed to FAIL; "
                         "recorded in the manifest and the summary")
    ap.add_argument("--no-analyse", action="store_true")
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(line_buffering=True)     # live progress in a log
    except (AttributeError, ValueError):
        pass

    if args.mark_stale:
        if not args.reason:
            print("  --mark-stale needs --reason: say what was fixed")
            return 2
        n = mark_stale(os.path.join(RESULTS, args.mark_stale), args.reason)
        print(f"  {n} rows of {args.mark_stale} marked stale_code")
        return 0
    if not args.key_file:
        print("  --key-file is required to run")
        return 2
    with open(args.key_file, encoding="utf-8") as f:
        key = f.read().strip()
    if not key:
        print("  the key file is empty")
        return 2

    if args.timeout < MIN_TIMEOUT:
        print(f"  --timeout {args.timeout:.0f} raised to {MIN_TIMEOUT:.0f}: the "
              f"server's own generation timeout is 3600 s")
        args.timeout = MIN_TIMEOUT
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        print(f"  unknown arms {unknown}; known: {list(ARMS)}")
        return 2

    print("\n  PREFLIGHT")
    checks = preflight(args.key_file)
    failed = [c[1] for c in checks if not c[0]]
    overridden = [n for n in args.override_preflight.split(",") if n]
    blocking = [n for n in failed if n not in overridden]
    if args.preflight_only:
        print(f"\n  preflight: {len(checks) - len(failed)}/{len(checks)} passed")
        return 0 if not failed else 1
    if blocking:
        print(f"\n  ABORT: preflight failed: {', '.join(blocking)}")
        return 1

    run_id = args.run_id or f"{args.effort}-{time.strftime('%Y%m%d-%H%M%S')}"
    q = quiet_worker(run_id)
    print(("  pass  " if q[0] else "  FAIL  ") + f"{q[1]:<16} {q[2]}")
    checks.append(q)
    if not q[0] and "worker_quiet" not in overridden:
        release_worker()
        print("\n  ABORT: preflight failed: worker_quiet")
        return 1
    try:
        return _run(args, key, arms, checks, failed, overridden, run_id)
    finally:
        release_worker()


def _run(args, key, arms, checks, failed, overridden, run_id) -> int:
    import grade as grade_mod
    tasks = select_tasks(load_suite(args.suite), args.per_domain,
                         [t for t in args.tasks.split(",") if t] or None,
                         [d for d in args.domains.split(",") if d] or None)
    if not tasks:
        print("  no tasks selected")
        return 2
    run_dir = os.path.join(RESULTS, run_id)
    os.makedirs(os.path.join(run_dir, "answers"), exist_ok=True)

    # One run directory holds ONE condition set. Resuming with a different
    # effort, budget or temperature would mix conditions inside a pairing.
    man_path = os.path.join(run_dir, "manifest.json")
    fixed = {"effort": args.effort, "max_tokens": args.max_tokens,
             "temperature": args.temperature, "body_tier": BODY_TIER,
             "proxy": args.proxy, "suite": args.suite}
    if os.path.exists(man_path):
        with open(man_path, encoding="utf-8") as f:
            old = json.load(f)
        clash = manifest_clash(old, fixed)
        if clash:
            print(f"  ABORT: {run_id} was run with different conditions: {clash}")
            return 2
        man = old
        man.setdefault("suite", args.suite)
        for a in ARMS:                    # arms added since the dir was made
            man.setdefault("arms", {}).setdefault(a, arm_manifest(a, args.effort))
    else:
        man = dict(fixed, run_id=run_id, created=time.time(), runs=[],
                   arms={a: arm_manifest(a, args.effort) for a in ARMS})
    man["runs"].append({"started": time.time(), "arms": arms,
                        "tasks": [t["id"] for t in tasks],
                        "preflight": [{"ok": c[0], "name": c[1], "detail": c[2]}
                                      for c in checks],
                        "preflight_overridden": [n for n in failed if n in overridden]})
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(man, f, indent=1)

    cfg = {"run_id": run_id, "run_dir": run_dir, "url": args.proxy, "key": key,
           "effort": args.effort, "max_tokens": args.max_tokens,
           "temperature": args.temperature, "timeout": args.timeout,
           "suite": args.suite}
    print(f"\n  RUN {run_id}: suite={args.suite} effort={args.effort} "
          f"max_tokens={args.max_tokens} arms={','.join(arms)}")
    t0 = time.time()
    try:
        run_pairs(tasks, arms, cfg, grade_mod.grade)
    except KeyboardInterrupt:
        print("  interrupted; rerun with the same --run-id to resume")
        return 130
    except Exception:                                            # noqa: BLE001
        traceback.print_exc()
        return 1
    print(f"  finished in {time.time() - t0:.0f}s")
    if not args.no_analyse:
        import analyse
        analyse.main([run_dir])
    return 0


if __name__ == "__main__":
    sys.exit(main())
