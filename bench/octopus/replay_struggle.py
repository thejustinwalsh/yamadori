#!/usr/bin/env python
"""Replay Octopus runs through deep thinking's trigger decision.

    python bench/octopus/replay_struggle.py                 # the gate set
    python bench/octopus/replay_struggle.py v0e-V0-xhigh-1:2 [--all]
    python bench/octopus/replay_struggle.py ... --old-module PATH/deep.py

WHAT IT DOES. The relay (logs/<run>/relay[.pN].jsonl) records each request's
x_yamadori.deep decision and the message count, not the messages; Hermes'
transcript (hermes[.pN].jsonl) has every tool call, its arguments and its
result. This rebuilds the message list the client sent at each request and
drives the REAL decision -- deep.decide, then deep.mark_ran when it fires --
over it, with its own temporary ledger and corpus (nothing under index/ is
touched; no model, no port). So the episode and a compaction's new epoch
are the production code's, not a re-implementation. (The cooldown, the
same-pattern suppression and the ENVIRONMENT table were removed from deep.py
2026-09-27, docs/CONSTANTS-AUDIT.md; an --old-module copy still shows them.)

THE REBUILD. Prompt 1 starts from [system, the task (prompt.md)]; prompt N
starts from the session Hermes saved at the end of prompt N-1
(session.jsonl: the final context, 100 messages for v0e) plus
prompt.pN.md. Each answered request adds its turn: the calls its response
made (the relay's `response.tool_calls` -- Hermes logs a turn's parallel
calls interleaved with their results, so the transcript alone cannot say
where a turn ends), paired in order with the transcript's calls and
results. After a compaction (the count drops) the client's list is [system,
a summary placeholder] + the last n-2 messages of the history -- Hermes
keeps the recent tail -- and the kept tail's older long outputs are pruned
to one line, as Hermes does (approximated: PRUNE_CHARS, PROTECT_TAIL); a
result the relay says went out as a one-line summary is replaced by it. The
summary's own text is not in the logs, and the transcript cuts outputs at
5,003 characters. Rows the proxy served as a utility call (the compaction
itself) or never decided (a 429, a stream that died before x_yamadori) are
skipped, as decide never saw them. `aligned` counts tool-result requests
whose rebuilt last message starts like the relay's `last_head`.

COLUMNS. `live` is what the proxy recorded (the rule in force that night);
`new` is the current deep.py; `old` (with --old-module, e.g. a saved copy of
the previous deep.py) replays another version the same way, which checks the
rebuild: where `old` matches `live`, the replay reproduces the run.

Offline: reads the log files, imports mcp/deep.py against temp databases.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "mcp"))

# Temp databases BEFORE any mcp import: the replay writes conversation state
# (the ledger) and nothing else, and never into index/.
_TMP = tempfile.mkdtemp(prefix="yamadori_replay_struggle_")
os.environ["YAMADORI_CORPUS_DB"] = os.path.join(_TMP, "corpus.sqlite3")
os.environ["YAMADORI_NEBARI_DB"] = os.path.join(_TMP, "nebari.sqlite3")
os.environ["YAMADORI_JOBS_DB"] = os.path.join(_TMP, "jobs.sqlite3")
os.environ["RINGS_DB"] = os.path.join(_TMP, "rings.sqlite3")
os.environ["YAMADORI_E1"] = "0"          # the live runs' escalate head was
for _k in ("YAMADORI_STRUGGLE_THRESHOLD",):
    os.environ.pop(_k, None)             # untrained: the rule decided

import deep  # noqa: E402

LOGS = os.environ.get("OCTO_LOGS", r"C:\Users\jwals\octo\logs")
# The gate (docs/SELF-IMPROVEMENT-LOG.md #45): v0e prompts 1 and 2, v0b and
# the pilot (the older V0 run).
GATE = ("pilot-V0-xhigh-1:1", "v0b-V0-xhigh-1:1", "v0e-V0-xhigh-1:1",
        "v0e-V0-xhigh-1:2")
TIER = {"name": "xhigh", "investigate": True}
# Hermes' compaction pruning, approximated (its rule is a token budget the
# logs do not show): tool outputs longer than PRUNE_CHARS in the kept tail,
# except its last PROTECT_TAIL messages, become one line. From the saved
# sessions: short outputs (<= ~200 chars) survive; Hermes' own default
# protects the last 20 messages. An approximation, stated in the report.
PRUNE_CHARS = 200
PROTECT_TAIL = 20


def jl(path: str) -> list[dict]:
    """JSON lines; Hermes' progress lines ("compacting context...") are
    kept as {"type": "_line"}."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                out.append({"type": "_line", "text": line.strip()})
    return out


def _args_json(v) -> str:
    if isinstance(v, str):
        return v
    return json.dumps(v if v is not None else {})


def _from_session(m: dict) -> dict:
    """A saved Hermes message as the client sends it."""
    out = {"role": m.get("role"), "content": m.get("content") or ""}
    calls = m.get("tool_calls")
    if isinstance(calls, str):
        try:
            calls = json.loads(calls)
        except ValueError:
            calls = None
    if isinstance(calls, list) and calls:
        out["tool_calls"] = [{
            "id": c.get("id"), "type": "function",
            "function": {"name": (c.get("function") or {}).get("name"),
                         "arguments": _args_json((c.get("function") or {})
                                                 .get("arguments"))}}
            for c in calls if isinstance(c, dict)]
    if m.get("tool_call_id"):
        out["tool_call_id"] = m["tool_call_id"]
    if m.get("tool_name"):
        out["name"] = m["tool_name"]
    return out


def pairs_of(hermes: list[dict]) -> tuple[list[dict], str]:
    """Every tool call with its result, in order, and the text after the
    last one. Hermes logs a turn's parallel calls interleaved with their
    results (use, result, use, result), so the transcript alone cannot say
    where a turn ends: the relay's response does (its tool_calls)."""
    out: list[dict] = []
    text: list[str] = []
    pending: list[dict] = []
    for e in hermes:
        t = e.get("type")
        if t == "text":
            text.append(e.get("text") or "")
        elif t == "tool_use":
            p = {"name": e.get("name") or "",
                 "arguments": _args_json(e.get("input")),
                 "text": "".join(text).strip(), "result": None}
            text = []
            out.append(p)
            pending.append(p)
        elif t == "tool_result":
            res = e.get("output")
            if pending:
                pending.pop(0)["result"] = (res if isinstance(res, str)
                                            else json.dumps(res))
    return out, "".join(text).strip()


def load(spec: str, logs: str) -> dict:
    run, _, p = spec.partition(":")
    p = int(p or 1)
    d = os.path.join(logs, run)
    suf = "" if p == 1 else f".p{p}"
    system = {"role": "system", "content": "(hermes system prompt)"}
    if p == 1:
        with open(os.path.join(d, "prompt.md"), encoding="utf-8") as f:
            prefix = [system, {"role": "user", "content": f.read()}]
    else:
        with open(os.path.join(d, "session.jsonl"), encoding="utf-8") as f:
            sess = json.load(f)
        with open(os.path.join(d, f"prompt{suf}.md"), encoding="utf-8") as f:
            ask = f.read()
        prefix = ([system] + [_from_session(m) for m in sess["messages"]]
                  + [{"role": "user", "content": ask}])
    relay = [r for r in jl(os.path.join(d, f"relay{suf}.jsonl"))
             if r.get("path") in ("/v1/chat/completions", "/v1/responses")]
    pairs, tail = pairs_of(jl(os.path.join(d, f"hermes{suf}.jsonl")))
    return {"spec": f"{run}:{p}", "relay": relay, "pairs": pairs,
            "tail": tail, "prefix": prefix}


def _head(s: str) -> str:
    return " ".join((s or "").split())[:40]


def requests(run: dict):
    """(step, n, the client's messages, the relay row, aligned or None) for
    every request decide saw. The history grows by each answered request's
    turn: its tool calls (the relay's response names them) paired, in
    order, with the transcript's calls and results."""
    hist = list(run["prefix"])
    run["full"] = list(run["prefix"])       # never pruned: the whole run
    pairs, cur = run["pairs"], 0
    compacted, last_n = False, None
    for s, r in enumerate(run["relay"], 1):
        req = r.get("request") or {}
        resp = r.get("response") or {}
        x = resp.get("x_yamadori") or {}
        util = bool(x.get("utility"))
        n, role = req.get("n_messages") or 0, req.get("last_role")
        decided = bool(x) and not util
        if decided:
            if last_n is not None and n < last_n:
                compacted = True
                # Hermes' compaction also PRUNES the kept tail's older tool
                # outputs to one line ("[terminal] ran `...` -> exit -1",
                # "[Duplicate tool output ...]"; the saved sessions show it),
                # so an old error is no longer readable as one.
                keep = max(0, n - 2)
                lo, hi = len(hist) - keep, len(hist) - PROTECT_TAIL
                for i in range(max(0, lo), max(0, hi)):
                    m = hist[i]
                    if m.get("role") == "tool" and \
                            len(m.get("content") or "") > PRUNE_CHARS:
                        hist[i] = dict(m, content=f"[{m.get('name')}] "
                                       "(output pruned by the compaction)")
            last_n = n
            if not compacted and n == len(hist):
                msgs = list(hist)
            else:
                compacted = True
                head = [hist[0], {"role": "user", "content":
                                  "[CONTEXT COMPACTION] (summary not logged)"}]
                if role == "user":
                    k = max(0, min(n - 3, len(hist)))
                    msgs = head + hist[len(hist) - k:] + [
                        {"role": "user", "content": req.get("last_head") or ""}]
                else:
                    k = max(0, min(n - 2, len(hist)))
                    msgs = head + hist[len(hist) - k:]
            ok = None
            if role == "tool" and msgs:
                ok = _head(msgs[-1].get("content")).startswith(
                    _head(req.get("last_head"))[:30])
                sent = req.get("last_chars")
                last = msgs[-1].get("content") or ""
                if isinstance(sent, int) and sent < PRUNE_CHARS < len(last) \
                        and not ok:
                    # Hermes sent this result as a one-line summary (the
                    # relay's size says so); the transcript has the output.
                    msgs[-1] = dict(msgs[-1], content=req.get("last_head"))
                    hist[-1] = msgs[-1]
                    ok = True
            yield s, n, msgs, r, ok
        if util:
            continue
        names = resp.get("tool_calls") or []
        if names:
            turn = pairs[cur:cur + len(names)]
            cur += len(turn)
            msg = {"role": "assistant",
                   "content": "\n".join(p["text"] for p in turn
                                        if p["text"]),
                   "tool_calls": []}
            res = []
            for j, p in enumerate(turn):
                cid = f"call_{cur - len(turn) + j + 1}"
                msg["tool_calls"].append({
                    "id": cid, "type": "function",
                    "function": {"name": p["name"],
                                 "arguments": p["arguments"]}})
                res.append({"role": "tool", "tool_call_id": cid,
                            "name": p["name"], "content": p["result"] or ""})
            hist += [msg] + res
            run["full"] += [msg] + res
        elif resp.get("content_chars") and resp.get("finish_reason") == "stop":
            hist.append({"role": "assistant", "content": run["tail"]
                         if cur >= len(pairs) else "(an answer)"})
            run["full"].append(hist[-1])


def _sig(e: dict) -> str:
    what = e.get("tool") or e.get("path") or e.get("phrase") or ""
    s = e.get("signature") or e.get("why") or ""
    return f"{e['kind']}({what}{': ' + s[:60] if s else ''})"


def drive(mod, run: dict, tag: str) -> dict:
    """Every request through mod.decide; mark_ran when it fires (and for a
    think_deeply call that ran live, which the replay cannot re-decide)."""
    acct, lin = "replay", f"{tag}:{run['spec']}"
    out = {}
    for s, n, msgs, r, _ok in requests(run):
        rec = mod.decide(raw=msgs, tier=dict(TIER),
                         route={"class": "agent_step"}, util={},
                         account=acct, lineage=lin, turn_key=f"{s}")
        live = ((r.get("response") or {}).get("x_yamadori") or {}).get(
            "deep") or {}
        calls = [c for c in (live.get("think_tool") or {}).get("calls") or []
                 if c.get("ran")]
        if rec.get("fire"):
            mod.mark_ran(acct, lin, n, rec.get("kind") or "forced",
                         turn_key=f"{s}")
        elif calls:
            mod.mark_ran(acct, lin, n, "model")
        out[s] = rec
    return out


def _cell(rec: dict | None) -> str:
    if not rec:
        return "-"
    st = (rec.get("signals") or {}).get("struggle") or {}
    env = (st.get("environment") or {}).get("count") or 0
    c = f"{st.get('count', 0)}" + (f"+{env}env" if env else "")
    if rec.get("fire"):
        return f"FIRE {rec.get('kind')} {c}"
    if rec.get("suppressed"):
        return f"suppressed:{rec['suppressed']} {c}"
    return c


def report(run: dict, old_mod, show_all: bool) -> dict:
    steps = list(requests(run))
    tcount = sum(1 for *_x, ok in steps if ok is not None)
    aligned = sum(1 for *_x, ok in steps if ok)
    new = drive(deep, run, "new")
    old = drive(old_mod, run, "old") if old_mod else {}
    live = {s: ((r.get("response") or {}).get("x_yamadori") or {}).get("deep")
            for s, _n, _m, r, _ok in steps}
    print(f"\n=== {run['spec']}: {len(steps)} decided requests, "
          f"{len(run['pairs'])} tool calls, tool-result requests aligned "
          f"with the relay {aligned}/{tcount}")
    hdr = f"{'step':>4} {'n':>4}  {'live':<24}"
    if old_mod:
        hdr += f" {'old (replay)':<24}"
    print(hdr + " new (replay)")
    for s, n, _m, _r, _ok in steps:
        cells = [live.get(s), old.get(s), new.get(s)]
        busy = any(c and (c.get("fire") or c.get("suppressed")) for c in cells)
        if not (show_all or busy):
            continue
        row = f"{s:>4} {n:>4}  {_cell(live.get(s)):<24}"
        if old_mod:
            row += f" {_cell(old.get(s)):<24}"
        print(row + " " + _cell(new.get(s)))

    def fires(d):
        return [s for s, rec in d.items() if rec and rec.get("fire")
                and rec.get("kind") == "struggle"]
    res = {"spec": run["spec"], "requests": len(steps),
           "aligned": f"{aligned}/{tcount}", "live": fires(live),
           "new": fires(new)}
    if old_mod:
        res["old"] = fires(old)
    print(f"struggle fired -- live: {res['live'] or 'none'}"
          + (f"; old replay: {res['old'] or 'none'}" if old_mod else "")
          + f"; new replay: {res['new'] or 'none'}")
    for s in res["new"]:
        ev = new[s]["signals"]["struggle"]["events"]
        print(f"  new fires at {s}: " + "; ".join(_sig(e) for e in ev))
    for s, rec in new.items():
        if rec.get("suppressed"):
            print(f"  new suppressed at {s}: {rec['because'][:160]}")
    # Every failure of the prompt, once, over the unpruned history (this
    # prompt's own messages only): how the new rule reads each.
    env: dict[str, int] = {}
    scan = deep.struggle_scan(run["full"], len(run["prefix"]))
    envs = scan.get("environment") or []     # none since 2026-09-27
    for e in envs:
        k = f"{e.get('class')}:{e.get('tool')}"
        env[k] = env.get(k, 0) + 1
    res["failures"] = {"all": len(scan["errors"]),
                       "environment": len(envs),
                       "task": len(scan["errors"]) - len(envs)}
    print(f"  failing tool results in this prompt: {res['failures']['all']} "
          f"({res['failures']['environment']} environment, "
          f"{res['failures']['task']} the task's)")
    if env:
        print("  environment (recorded, not counted): " + ", ".join(
            f"{k} x{v}" for k, v in sorted(env.items())))
    res["environment"] = env
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", default=list(GATE),
                    help="RUN[:PROMPT], default the #45 gate set")
    ap.add_argument("--logs", default=LOGS)
    ap.add_argument("--old-module", help="another deep.py to replay beside "
                                         "the current one")
    ap.add_argument("--all", action="store_true",
                    help="print every request, not only where one fires")
    ap.add_argument("--json", help="write the summary here")
    # --count-class (a what-if over deep.ENVIRONMENT) went with the table,
    # 2026-09-27: every failure counts now.
    a = ap.parse_args()
    old_mod = None
    if a.old_module:
        spec = importlib.util.spec_from_file_location("deep_old",
                                                      a.old_module)
        old_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old_mod)
    out = []
    for r in a.runs:
        try:
            run = load(r, a.logs)
        except FileNotFoundError as e:
            print(f"\n=== {r}: not replayed ({e})")
            continue
        out.append(report(run, old_mod, a.all))
    print("\nSUMMARY (struggle runs by request step)")
    for r in out:
        print(f"  {r['spec']:<22} live {r['live'] or '-'}"
              + (f"  old {r['old'] or '-'}" if old_mod else "")
              + f"  new {r['new'] or '-'}  (aligned {r['aligned']})")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
