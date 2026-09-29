#!/usr/bin/env python
"""The no-progress guard: the harness's own tool calls, read for repeats.

WHERE IT COMES FROM. Atomic Agent's `ToolLoopTracker`
(github.com/AtomicBot-ai/atomic-agent, MIT, commit 52f90e55,
src/agent/loop-detector.ts; "Ported from OpenClaw 2026.6.5" per its
AGENTS.md "No-progress loop detection"). docs/research/ATOMIC-AGENT.md has
the study, the evidence behind it (thin: see there) and the counts of how
often it would have fired on our own runs.

WHAT IT DECIDES, for one CLIENT tool call the model just wrote, against the
calls and results earlier in the same task (the messages after the last user
turn, at most WINDOW_CALLS calls back):

  warn     the same tool with the same arguments ran WARN_REPEATS times
           before (Atomic's `getRepeatCount`: args only, interleaving
           tolerated). The call RUNS. The line `notice_text` is DELIVERED
           on that call's result when it comes back (the next request):
           the situation as a fact, "Do not repeat this identical call",
           and what to do instead (operator, 2026-09-29).
  veto     the same call's last VETO_STREAK results were identical
           (Atomic's `getNoProgressStreak`: args AND result, interleaving
           tolerated, a changed result breaks the streak, a veto is not
           counted). The call does NOT run: the proxy hands the model a
           NOT EXECUTED result in a hidden hop (the image guard's mechanism,
           tool_code IMAGE GUARD) and the model writes its turn again.
  breaker  the call was vetoed BREAKER_VETOES times in a row already (in
           this request's hidden hops). Atomic ends the turn with a
           graceful reply. HERE IT NEVER ENDS THE HARNESS'S TURN (operator,
           2026-09-29): the call gets the same NOT EXECUTED result as a
           veto (saying how many times it was written and not run) and the
           run continues. `level` says "breaker" for the record only;
           `action` is "veto" for both.

  outcome_repeat (warn only) the same RESULT -- tool, status and its first
           OUTCOME_HEAD_CHARS characters, whitespace collapsed -- came back
           OUTCOME_REPEATS times since the last successful write, whatever
           the arguments (Atomic's `fingerprintToolOutcome`).

"IDENTICAL" (Atomic's `hashToolCall` / `hashToolOutcome`):
  a call  -- the tool name plus its arguments as canonical JSON (keys
             sorted, values exact). `ls src` and `ls src/` are different
             calls, as in Atomic.
  a result -- its text; a JSON result with Atomic's volatile keys dropped
             (VOLATILE_KEYS: timings, ids, dates), and Codex's two volatile
             header lines ("Chunk ID:", "Wall time:") dropped -- the same
             rule (per-call ids and timings are not the answer) applied to
             the one harness whose result text carries them.

EVERY NUMBER IS ATOMIC'S (src/config/config-schema.ts ENV_DEFAULTS and
src/agent/loop-detector.ts, commit 52f90e55), for the operator to confirm:
not measured on this model. The env variables pin them.

WHAT THE MODEL READS follows AGENTS.md "Failure returns carry the next
step": the situation as a fact, retryable as a fact, the next step and
whose it is. ONE prohibition, in the warning line only: "Do not repeat
this identical call", followed by what to do instead (operator,
2026-09-29: "do not repeat is a good exemption to add a rule for that helps
steer a session"; AGENTS.md "Prompting this model"). No capitals, no
doubt: Atomic's "BLOCKED", "STOP", "CRITICAL" are not used.

PURE: no I/O, no proxy import, no state. Everything is read from the
messages the caller passes: the client's messages as sent (never the
ledger-restored ones: a tool result there may carry our injections, which
would make two identical results differ) plus this request's hidden hops.
"""
from __future__ import annotations

import hashlib
import json
import os
import re

# ---------------------------------------------------------------- numbers --
# Atomic Agent ENV_DEFAULTS (src/config/config-schema.ts, 52f90e55).
WARN_REPEATS = int(os.environ.get("YAMADORI_GUARD_WARN_REPEATS", "3"))
#   LOOP_WARNING_THRESHOLD: 3 -- args-only repeats before a warning.
VETO_STREAK = max(WARN_REPEATS,
                  int(os.environ.get("YAMADORI_GUARD_VETO_STREAK", "5")))
#   LOOP_CRITICAL_THRESHOLD: 5 -- identical args+result streak that vetoes
#   (clamped >= the warning threshold, as Atomic clamps it).
BREAKER_VETOES = max(1, int(os.environ.get("YAMADORI_GUARD_BREAKER", "3")))
#   LOOP_BREAKER_VETO_STREAK: 3 -- consecutive vetoes before the turn ends.
WINDOW_CALLS = max(VETO_STREAK,
                   int(os.environ.get("YAMADORI_GUARD_WINDOW", "30")))
#   LOOP_HISTORY_SIZE: 30 -- calls the tracker remembers.
WARN_BUCKET = 10
#   LOOP_WARNING_BUCKET_SIZE: 10 -- a warning for one key is given once per
#   bucket of 10 repeats, so the line is not re-added on every step.
OUTCOME_REPEATS = 3
#   OUTCOME_REPEAT_WARNING_THRESHOLD: 3 (src/agent/loop-detector.ts).
OUTCOME_HEAD_CHARS = 200
#   OUTCOME_FINGERPRINT_CHARS: 200 (src/agent/loop-detector.ts).

# Atomic's VOLATILE_RESULT_KEYS (src/agent/loop-detector.ts), verbatim.
VOLATILE_KEYS = frozenset({
    "timestamp", "ts", "date", "time", "timeTotal", "timeTotalSeconds",
    "durationMs", "sizeDownload", "requestId", "request_id", "id",
    "traceId", "trace_id", "sentAt", "createdAt", "deliveredAt"})

# Codex exec_command's volatile header lines (bench/harness_shapes/codex/
# responses__struggle-same-error.json: "Chunk ID: 87eb5c", "Wall time: 0.1
# seconds").
_CODEX_VOLATILE = re.compile(r"^(?:Chunk ID|Wall time):.*$", re.M)

# The marker a veto's result opens with. `history` reads a result that opens
# with it as a veto, never as an outcome (Atomic's `isLoopVetoResult`).
VETO_HEAD = "NOT EXECUTED (no progress):"
OTHER_HEAD = "NOT EXECUTED (not run):"


# ------------------------------------------------------------ identities --
def _sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()[:12]


def _canonical(value) -> str:
    """Atomic's `canonicalJson`: keys sorted, arrays in order."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))


def _args_of(arguments) -> object:
    if isinstance(arguments, (dict, list)):
        return arguments
    if isinstance(arguments, str):
        try:
            return json.loads(arguments) if arguments.strip() else {}
        except ValueError:
            return arguments          # unparseable: the raw text is the call
    return {}


def call_key(name: str, arguments) -> str:
    """Atomic's `hashToolCall`: the tool and its canonical arguments."""
    return f"{name}:{_sha(_canonical(_args_of(arguments)))}"


def _strip_volatile(value):
    if isinstance(value, list):
        return [_strip_volatile(v) for v in value]
    if isinstance(value, dict):
        return {k: _strip_volatile(v) for k, v in value.items()
                if k not in VOLATILE_KEYS}
    return value


def result_text(content) -> str:
    """A tool message's content as text (a list of parts joined)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for p in content:
            if isinstance(p, dict) and isinstance(p.get("text"), str):
                out.append(p["text"])
            elif isinstance(p, dict):
                # an image part: its kind, never its data
                out.append(f"<{p.get('type') or 'part'}>")
        return "\n".join(out)
    return "" if content is None else str(content)


def failed(text: str) -> bool | None:
    """Did the call fail, read from STRUCTURED fields only (AGENTS.md: a
    failure is never read from plain text)? True/False from a JSON result's
    exit_code, error, ok/success; None when the result carries none (Pi,
    OpenCode and Codex results are plain text)."""
    s = (text or "").lstrip()
    if not s.startswith("{"):
        return None
    try:
        d = json.loads(s)
    except ValueError:
        return None
    if not isinstance(d, dict):
        return None
    if isinstance(d.get("exit_code"), int):
        return d["exit_code"] != 0
    for k in ("ok", "success"):
        if isinstance(d.get(k), bool):
            return not d[k]
    if "error" in d:
        return bool(d.get("error"))
    return None


def result_key(text: str) -> str:
    """Atomic's `hashToolOutcome`, on the text the harness sent."""
    s = text or ""
    st = s.lstrip()
    if st.startswith("{") or st.startswith("["):
        try:
            return "json:" + _sha(_canonical(_strip_volatile(json.loads(st))))
        except ValueError:
            pass
    return "text:" + _sha(_CODEX_VOLATILE.sub("", s))


def outcome_fingerprint(name: str, text: str) -> str:
    """Atomic's `fingerprintToolOutcome`: tool, status, the first
    OUTCOME_HEAD_CHARS characters with whitespace runs collapsed."""
    status = "error" if failed(text) else "ok"
    head = re.sub(r"\s+", " ", _CODEX_VOLATILE.sub("", text or "")).strip()
    return f"{name}|{status}|{head[:OUTCOME_HEAD_CHARS]}"


def _is_write(name: str) -> bool:
    """A write/edit tool of some harness: tool_code's known-names table
    (the one list), when it is importable."""
    try:
        import tool_code
    except Exception:                                   # noqa: BLE001
        return False
    return name in tool_code.KNOWN


# --------------------------------------------------------------- history --
def task_start(messages: list[dict]) -> int:
    """Index of the first message of the current task: the one after the
    last user turn (Atomic: one tracker per turn -- a user message, its tool
    steps, its reply)."""
    for i in range(len(messages) - 1, -1, -1):
        if (messages[i] or {}).get("role") == "user":
            return i + 1
    return 0


def history(messages: list[dict]) -> list[dict]:
    """The current task's calls, oldest first, each with its outcome:
    {name, key, result: key|None, veto, write_ok, fp}. A call with no result
    yet has result None; a veto's result is marked and carries no key. At
    most WINDOW_CALLS entries (Atomic's ring)."""
    msgs = messages[task_start(messages):]
    results: dict[str, str] = {}
    for m in msgs:
        if (m or {}).get("role") == "tool":
            results[m.get("tool_call_id") or ""] = result_text(m.get("content"))
    out: list[dict] = []
    for m in msgs:
        if (m or {}).get("role") != "assistant":
            continue
        for c in m.get("tool_calls") or []:
            fn = (c or {}).get("function") or {}
            name = fn.get("name") or ""
            e = {"name": name, "key": call_key(name, fn.get("arguments")),
                 "result": None, "veto": False, "write_ok": False, "fp": None}
            res = results.get(c.get("id") or "")
            if res is not None:
                if res.startswith(VETO_HEAD):
                    e["veto"] = True
                elif res.startswith(OTHER_HEAD):
                    e["other"] = True      # never ran: neither veto nor outcome
                else:
                    e["result"] = result_key(res)
                    e["fp"] = outcome_fingerprint(name, res)
                    e["write_ok"] = _is_write(name) and failed(res) is not True
            out.append(e)
    return out[-WINDOW_CALLS:]


def repeat_count(hist: list[dict], key: str) -> int:
    """Atomic's `getRepeatCount`: matching calls in the window."""
    return sum(1 for e in hist if e["key"] == key and not e.get("other"))


def no_progress_streak(hist: list[dict], key: str) -> tuple[int, str | None]:
    """Atomic's `getNoProgressStreak`: walking back over this call's
    completed outcomes, how many match the newest; a changed result ends
    the streak; pending, vetoed and not-run entries are skipped."""
    streak, latest = 0, None
    for e in reversed(hist):
        if e["key"] != key or e["result"] is None:
            continue
        if latest is None:
            latest, streak = e["result"], 1
        elif e["result"] == latest:
            streak += 1
        else:
            break
    return streak, latest


def vetoes_in_a_row(hist: list[dict], key: str) -> int:
    """Atomic's consecutive-veto counter: vetoes of `key` uninterrupted by
    any other call's outcome."""
    n = 0
    for e in reversed(hist):
        if e.get("other"):
            continue
        if e["veto"] and e["key"] == key:
            n += 1
        elif e["veto"] or e["result"] is not None:
            break
    return n


def _emit(count: int, floor: int) -> bool:
    """Atomic's `shouldEmitWarning` without its state: a key's count rises
    by one per call, so bucket b is first reached at floor + b*WARN_BUCKET."""
    return count >= floor and (count - floor) % WARN_BUCKET == 0


# -------------------------------------------------------------- decision --
def decide(messages: list[dict], name: str, arguments) -> dict:
    """The verdict on one call the model just wrote, before it is forwarded:
    {level: ok|warn|veto|breaker, detector, count, key, tool, target}.
    `messages`: the client's messages as sent plus this request's hidden
    hops (which carry the vetoes already given), in order."""
    hist = history(messages)
    key = call_key(name, arguments)
    base = {"tool": name, "key": key, "target": target_of(name, arguments)}
    streak, _ = no_progress_streak(hist, key)
    vetoed = vetoes_in_a_row(hist, key)
    if vetoed >= BREAKER_VETOES:
        # Atomic ends the turn here; the proxy never does (operator,
        # 2026-09-29): the same NOT EXECUTED result, the run continues.
        return dict(base, level="breaker", action="veto",
                    detector="no_progress", count=streak, vetoed=vetoed)
    if streak >= VETO_STREAK:
        return dict(base, level="veto", action="veto",
                    detector="no_progress", count=streak, vetoed=vetoed)
    n = repeat_count(hist, key)
    if n >= WARN_REPEATS:
        return dict(base, level="warn", detector="generic_repeat", count=n,
                    emit=_emit(n, WARN_REPEATS),
                    same_results=streak >= n)
    return dict(base, level="ok", detector="generic_repeat", count=n)


def notice_for_result(messages: list[dict], call_id: str) -> dict | None:
    """The warning for the result of call `call_id` (a message in
    `messages`), decided the way Atomic decides it -- against the calls
    BEFORE that one -- and given on its result, which is where the next
    prompt shows it. None when there is nothing to say. Decide ONCE per
    result and record it (the ledger), so a replay adds the same bytes.
    {detector, count, tool, text}."""
    idx = None
    for i, m in enumerate(messages):
        if (m or {}).get("role") == "assistant":
            for c in m.get("tool_calls") or []:
                if (c or {}).get("id") == call_id:
                    idx = (i, c)
    if idx is None:
        return None
    i, c = idx
    fn = c.get("function") or {}
    name = fn.get("name") or ""
    # The calls before this one: the messages up to this assistant turn,
    # plus this turn's earlier calls (a parallel batch counts in order).
    before = list(messages[:i])
    calls = messages[i].get("tool_calls") or []
    pos = calls.index(c)
    if pos:
        earlier = {x.get("id") for x in calls[:pos]}
        before.append(dict(messages[i], tool_calls=calls[:pos]))
        before.extend(r for r in messages[i + 1:]
                      if (r or {}).get("role") == "tool"
                      and r.get("tool_call_id") in earlier)
    res = next((result_text(r.get("content")) for r in messages[i + 1:]
                if (r or {}).get("role") == "tool"
                and r.get("tool_call_id") == call_id), None)
    if res is None or res.startswith((VETO_HEAD, OTHER_HEAD)):
        return None
    v = decide(before, name, fn.get("arguments"))
    if v["level"] == "warn" and v.get("emit"):
        _, latest = no_progress_streak(history(before), v["key"])
        same = bool(v.get("same_results")) and latest == result_key(res)
        return {"detector": "generic_repeat", "count": v["count"] + 1,
                "tool": name,
                "text": notice_text(name, v["count"] + 1, same=same,
                                    target=v.get("target"))}
    # Outcome repeat: this result's fingerprint, counted since the last
    # successful write (Atomic resets the counts at a write).
    if _is_write(name) and failed(res) is not True:
        return None
    fp = outcome_fingerprint(name, res)
    n = 1
    for e in reversed(history(before)):
        if e["write_ok"]:
            break
        if e["fp"] == fp:
            n += 1
    if _emit(n, OUTCOME_REPEATS):
        return {"detector": "outcome_repeat", "count": n, "tool": name,
                "text": outcome_text(name, n)}
    return None


def gate(messages: list[dict], tool_calls: list[dict],
         is_client=None) -> tuple[dict | None, list[dict]]:
    """Atomic's synchronous gate over ONE generation's calls, in order: an
    earlier call of the same generation counts for a later one (a duplicate
    inside a parallel batch is seen). `is_client(name)`: True for a call
    the harness runs (ours are not gated). Returns (stop, verdicts): stop
    is the first veto or breaker verdict, with its `call_id`, or None."""
    seen = list(messages)
    stop, verdicts = None, []
    for c in tool_calls or []:
        fn = (c or {}).get("function") or {}
        name = fn.get("name") or ""
        if is_client is not None and not is_client(name):
            continue
        v = dict(decide(seen, name, fn.get("arguments")),
                 call_id=c.get("id"))
        verdicts.append(v)
        if stop is None and v["level"] in ("veto", "breaker"):
            stop = v
        seen.append({"role": "assistant", "content": "", "tool_calls": [c]})
    return stop, verdicts


def hand_back(tool_calls: list[dict], stop: dict) -> list[dict]:
    """The tool messages that answer a stopped generation: the vetoed call
    its NOT EXECUTED result, every other call of the generation the
    not-run result (the image guard's shape: the model writes its turn
    again). Append after the generation's assistant message."""
    out = []
    for c in tool_calls or []:
        cid = (c or {}).get("id") or ""
        out.append({"role": "tool", "tool_call_id": cid,
                    "content": veto_result(stop)
                    if cid == stop.get("call_id") else other_result(stop)})
    return out


def notices_for_trailing(messages: list[dict]) -> list[dict]:
    """The warnings for the tool results the request ENDS on (the trailing
    run of tool messages), in order: what to append, once, to the last of
    them (the ledger's tool-result injection)."""
    out = []
    i = len(messages)
    while i and (messages[i - 1] or {}).get("role") == "tool":
        i -= 1
    for m in messages[i:]:
        n = notice_for_result(messages, m.get("tool_call_id") or "")
        if n:
            out.append(n)
    return out


# ---------------------------------------------------------------- wording --
def target_of(name: str, arguments) -> str | None:
    """Atomic's `extractLoopTarget`, coarse on purpose: a shell call's
    leading command word, a URL's host. Never the full arguments."""
    a = _args_of(arguments)
    if not isinstance(a, dict):
        return None
    cmd = a.get("command") or a.get("cmd")
    if isinstance(cmd, list):
        cmd = " ".join(str(x) for x in cmd)
    if isinstance(cmd, str) and cmd.strip():
        word = cmd.strip().split()[0]
        return re.sub(r"[`\r\n]+", " ", word)[:60]
    url = a.get("url") or a.get("uri")
    if isinstance(url, str):
        m = re.match(r"^[a-z][a-z0-9+.-]*://([^/:?#]+)", url.strip(), re.I)
        if m:
            return m.group(1)[:60]
    return None


def _nth(n: int) -> str:
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(
        n % 10, "th")
    return f"{n}{suf}"


def notice_text(tool: str, n: int, same: bool = False,
                target: str | None = None) -> str:
    """The warning line, appended to the result of the n-th identical call.
    A fact and the next action."""
    what = f"`{tool}`" + (f" (`{target}`)" if target else "")
    tail = (" and returned the same result each time" if same else "")
    return (f"This is the {_nth(n)} time this task has run {what} with "
            f"exactly these arguments{tail}. Do not repeat this identical "
            f"call. Instead, act on the result above: change the arguments "
            f"or the command, edit the file it points at, or answer with "
            f"what you have.")


def outcome_text(tool: str, n: int) -> str:
    """Atomic's `formatOutcomeRepeatNotice`, as a fact and the next action."""
    return (f"`{tool}` has returned this same result {n} times since the "
            f"last file write. Next: act on it -- edit or write the file it "
            f"points at, run a different command, or answer with what you "
            f"found.")


def veto_result(verdict: dict) -> str:
    """The NOT EXECUTED result for a vetoed call, in the shape AGENTS.md
    "Failure returns carry the next step" asks for. Opens with VETO_HEAD,
    which `history` reads back."""
    tool = verdict.get("tool") or "the tool"
    n = int(verdict.get("count") or VETO_STREAK)
    t = verdict.get("target")
    what = f"`{tool}`" + (f" (`{t}`)" if t else "")
    k = int(verdict.get("vetoed") or 0)
    again = (f" It has also been written {k} more times since then and not "
             f"run." if k else "")
    return (f"{VETO_HEAD} {what} with exactly these arguments already ran "
            f"{n} times in this task and returned the same result each time; "
            f"that result is in the conversation above, and running the call "
            f"again returns it again.{again} Retryable: no, with these "
            f"arguments. Next step (yours): act on that result -- change the "
            f"arguments or the command, edit the file the result points at, "
            f"or answer with what you have.")


def other_result(verdict: dict) -> str:
    """The result for a call written in the same generation as a vetoed one:
    not run either, since the model writes the turn again (the image guard's
    `image_other_result`)."""
    tool = verdict.get("tool") or "a call"
    return (f"{OTHER_HEAD} this call was written together with a `{tool}` "
            f"call that was not run (its result says why), so neither ran. "
            f"Retryable: yes -- make this call again if it is still needed.")


def record(verdict: dict, delivered: str) -> dict:
    """The x_yamadori.progress_guard row for one decision: never the
    arguments or the result text. `delivered`: forwarded | vetoed |
    noted (a warning line delivered on a result). Never "landed": the guard
    does not end a harness's turn."""
    return {k: verdict.get(k) for k in ("level", "detector", "count",
                                        "vetoed", "tool", "target",
                                        "key")} | {
        "delivered": delivered,
        "thresholds": {"warn": WARN_REPEATS, "veto": VETO_STREAK,
                       "breaker": BREAKER_VETOES, "window": WINDOW_CALLS,
                       "source": "atomic-agent 52f90e55 ENV_DEFAULTS"}}
