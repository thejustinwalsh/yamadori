#!/usr/bin/env python
"""Replay our own harness run logs through the no-progress guard
(mcp/progress_guard.py) and the shell-result compression
(mcp/result_compress.py): how often each WOULD have fired. Read-only: it
reads C:\\Users\\jwals\\octo\\logs (or --logs) and writes nothing unless
--out is given. No stack port, no GPU.

    python bench/atomic/replay_logs.py [--logs DIR] [--out FILE.json]

WHAT IS REPLAYED. Each run's tool calls and results, in order, rebuilt as
the chat messages the harness would send:
  Pi      pi.jsonl: tool_execution_start (name, args) / _end (result text);
          a user message_start opens a task; a compaction_end is a new task
          (the history is replaced by a summary).
  Hermes  hermes.jsonl: tool_use (name, input) / tool_result (output); a
          task opens at the run's start and after each compaction or
          harness notice, read from relay.jsonl's request heads by time.
The guard is replayed on the calls AS THEY HAPPENED: a veto would have
changed what the model did next, so "would veto" counts the calls that met
the veto rule, and the first such call per (task, call) is an EPISODE.
Compression is replayed on every result as if it were the result the request
ended on (compress_last's rule).

Approximations, stated: parallel calls are replayed one per assistant
message (their order kept); a compaction's kept tail is not modelled (the
task restarts empty); Hermes' in-run user turns other than compaction and
notices are not modelled (the pagoda/v0 runs are single-prompt).
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "mcp"))

import progress_guard as pg  # noqa: E402
import result_compress as rc  # noqa: E402

CHARS_PER_TOKEN = 3.8
#   AGENTS.md "Past reasoning is restored": ~3.8 characters a token.

NOTICE_HEADS = ("You are a summarization agent", "[CONTEXT COMPACTION",
                "[STILL IN PROGRESS", "You've reached the maximum")


def _jsonl(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except ValueError:
                continue


def _call(cid, name, args):
    return {"role": "assistant", "content": "",
            "tool_calls": [{"id": cid, "type": "function",
                            "function": {"name": name,
                                         "arguments": json.dumps(args)}}]}


def pi_events(path):
    """(kind, payload) in order: ('user',), ('call', id, name, args),
    ('result', id, text)."""
    for d in _jsonl(path):
        t = d.get("type")
        if t == "message_start" and (d.get("message") or {}).get("role") == \
                "user":
            yield ("user",)
        elif t == "compaction_end":
            yield ("user",)
        elif t == "tool_execution_start":
            yield ("call", d.get("toolCallId"), d.get("toolName"),
                   d.get("args") or {})
        elif t == "tool_execution_end":
            parts = ((d.get("result") or {}).get("content") or [])
            text = "\n".join(p.get("text", "") for p in parts
                             if isinstance(p, dict))
            yield ("result", d.get("toolCallId"), text)


def hermes_events(run_dir):
    bounds = []
    for rel in glob.glob(os.path.join(run_dir, "relay*.jsonl")):
        for d in _jsonl(rel):
            rq = d.get("request") or {}
            if rq.get("last_role") == "user" and \
                    (rq.get("last_head") or "").startswith(NOTICE_HEADS):
                bounds.append(float(d.get("t0") or 0) * 1000)
    bounds.sort()
    files = sorted(glob.glob(os.path.join(run_dir, "hermes*.jsonl")))
    yield ("user",)
    n = 0
    for fp in files:
        if fp != files[0]:
            yield ("user",)                 # a follow-up prompt's session
        pending: list[str] = []
        for d in _jsonl(fp):
            ts = float(d.get("timestamp") or 0)
            while bounds and bounds[0] <= ts:
                bounds.pop(0)
                yield ("user",)
            if d.get("type") == "tool_use":
                n += 1
                cid = f"h{n}"
                pending.append(cid)
                yield ("call", cid, d.get("name"), d.get("input") or {})
            elif d.get("type") == "tool_result" and pending:
                yield ("result", pending.pop(0), d.get("output") or "")


READ_TOOLS = {"read", "read_file", "Read"}
#   tool_code.READ_KNOWN's names (Pi/OpenCode `read`, Hermes `read_file`).


def _probe(name: str, args: dict) -> bool:
    """A step that reads inside an installed dependency (node_modules/ or
    site-packages/): the pattern the retired auto-trigger `probe` named."""
    blob = json.dumps(args)
    return "node_modules/" in blob or "site-packages/" in blob


def replay(events):
    msgs: list[dict] = []
    names: dict[str, tuple] = {}
    task = 0
    out = collections.Counter()
    vetoed_keys: set = set()
    streak_max = collections.Counter()
    same_calls = collections.Counter()
    examples = collections.defaultdict(list)
    comp = []
    last_read: dict[tuple, str] = {}      # (task, path) -> last result text
    recent: list = []
    read_norep: collections.Counter = collections.Counter()
    for ev in events:
        if ev[0] == "user":
            task += 1
            recent = []
            msgs.append({"role": "user", "content": "task"})
            continue
        if ev[0] == "call":
            _, cid, name, args = ev
            recent.append((pg.call_key(name, args),
                           _probe(name, args if isinstance(args, dict)
                                  else {})))
            recent_now = recent[:-1]
            same_calls[(task, pg.call_key(name, args))] += 1
            if _probe(name, args if isinstance(args, dict) else {}):
                out["probe_calls"] += 1
                # Atomic's WANDERING rule (distinct arguments on one
                # wandering-prone tool in the 30-call window: a redirect
                # at 6, the turn ended at 12), applied to dependency probes
                # as ONE tool class -- an EXTENSION: Atomic excludes file
                # reads from it ("scanning many files is legitimate work").
                keys = {k for k, is_probe in recent_now[-pg.WINDOW_CALLS:]
                        if is_probe}
                spread = len(keys | {pg.call_key(name, args)})
                if spread >= 12:
                    out["probe_wander_ge12_calls"] += 1
                    if (task, "w12") not in vetoed_keys:
                        vetoed_keys.add((task, "w12"))
                        out["probe_wander_ge12_episodes"] += 1
                elif spread >= 6 and (task, "w6") not in vetoed_keys:
                    vetoed_keys.add((task, "w6"))
                    out["probe_wander_ge6_episodes"] += 1
            if name in rc.SHELL_TOOLS | READ_TOOLS:
                out["shell_or_read_calls"] += 1
            v = pg.decide(msgs, name, args)
            out["calls"] += 1
            out[f"level_{v['level']}"] += 1
            if v["level"] in ("veto", "breaker"):
                k = (task, v["key"])
                if k not in vetoed_keys:
                    vetoed_keys.add(k)
                    out["veto_episodes"] += 1
                    cmd = rc._command(json.dumps(args)) or json.dumps(args)
                    examples["veto"].append(f"{name}: {cmd[:120]}")
            hist = pg.history(msgs)
            s, _ = pg.no_progress_streak(hist, v["key"])
            streak_max[(task, v["key"])] = max(streak_max[(task, v["key"])],
                                               s + 1)
            msgs.append(_call(cid, name, args))
            names[cid] = (name, args)
            continue
        _, cid, text = ev
        msgs.append({"role": "tool", "tool_call_id": cid, "content": text})
        name, args = names.get(cid, ("", {}))
        out["result_chars"] += len(text)
        if _probe(name, args if isinstance(args, dict) else {}):
            out["probe_chars"] += len(text)
        # Atomic's read-coverage detector (#114), approximated: a read of
        # one path whose result is byte-identical to the previous read of
        # that path in the task (so no line is new and the file did not
        # change); a write to the path in between resets it.
        if isinstance(args, dict):
            for p in rc.written_paths([_call("x", name, args)]):
                for k in [k for k in last_read if k[0] == task and
                          k[1].endswith(p)]:
                    last_read.pop(k, None)
                    read_norep.pop(k, None)
        if name in READ_TOOLS and isinstance(args, dict):
            path = args.get("path") or args.get("filePath") or ""
            k = (task, rc._norm(str(path)))
            if last_read.get(k) == text:
                read_norep[k] += 1
                out["reads_unchanged_repeat"] += 1
                if read_norep[k] == 2:        # READ_REPEAT_WARNING_THRESHOLD
                    out["read_repeat_warn"] += 1
            else:
                read_norep[k] = 0
            last_read[k] = text
        nt = pg.notice_for_result(msgs, cid)
        if nt:
            out[f"notice_{nt['detector']}"] += 1
            if len(examples[nt["detector"]]) < 6:
                name, args = names.get(cid, ("?", {}))
                cmd = rc._command(json.dumps(args)) or json.dumps(args)
                examples[nt["detector"]].append(
                    f"{name} x{nt['count']}: {cmd[:120]}")
        name, args = names.get(cid, ("", {}))
        _, rec = rc.compress(name, json.dumps(args), text,
                             rc.written_paths(msgs))
        comp.append(rec)
    # identical-call streak lengths as they happened (args+result)
    out["keys_streak_ge_6"] = sum(1 for v in streak_max.values() if v >= 6)
    out["keys_streak_ge_9"] = sum(
        1 for v in streak_max.values()
        if v >= pg.VETO_STREAK + pg.BREAKER_VETOES + 1)
    out["max_identical_calls"] = max(same_calls.values(), default=0)
    out["tasks"] = task
    return out, examples, comp


def hermes_session_comp(run_dir):
    """Compression records from Hermes' own session export
    (session*.jsonl: the chat messages as Hermes sends them, whole).
    hermes.jsonl's results are cut at 5,000 characters by its stream
    logger (138 of 1,012 results are exactly 5,003 long), so sizes come
    from here. The export holds the conversation's LAST segment (after
    Hermes' compactions start a child session), so it covers fewer calls
    than hermes.jsonl."""
    comp = []
    for fp in sorted(glob.glob(os.path.join(run_dir, "session*.jsonl"))):
        for d in _jsonl(fp):
            msgs = d.get("messages") or []
            seen: list[dict] = []
            names = {}
            for m in msgs:
                m = {k: m.get(k) for k in ("role", "content", "tool_calls",
                                          "tool_call_id")}
                if m["role"] == "assistant":
                    for c in m.get("tool_calls") or []:
                        fn = (c or {}).get("function") or {}
                        names[c.get("id")] = (fn.get("name") or "",
                                              fn.get("arguments"))
                seen.append(m)
                if m["role"] == "tool" and isinstance(m["content"], str):
                    name, args = names.get(m["tool_call_id"], ("", None))
                    _, rec = rc.compress(name, args, m["content"],
                                         rc.written_paths(seen))
                    comp.append(rec)
    return comp


def summarise_comp(comp):
    by = collections.Counter(r["why"] for r in comp)
    shell = [r for r in comp if r["why"] != "not_shell"]
    done = [r for r in comp if r["why"] == "compressed"]
    before = sum(r["before_chars"] for r in done)
    after = sum(r["after_chars"] for r in done)
    all_before = sum(r["before_chars"] for r in comp)
    return {"results": len(comp), "shell_results": len(shell),
            "by_why": dict(by),
            "compressed": len(done),
            "chars_before": before, "chars_after": after,
            "chars_saved": before - after,
            "tokens_saved_est": round((before - after) / CHARS_PER_TOKEN),
            "all_result_chars": all_before,
            "largest_before": max((r["before_chars"] for r in done),
                                  default=0),
            "protected_chars": sum(r["before_chars"] for r in comp
                                   if r["why"] == "protected")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=r"C:\Users\jwals\octo\logs")
    ap.add_argument("--out")
    a = ap.parse_args()
    rows = []
    for run in sorted(os.listdir(a.logs)):
        d = os.path.join(a.logs, run)
        if not os.path.isdir(d):
            continue
        if os.path.exists(os.path.join(d, "pi.jsonl")):
            ev, harness = pi_events(os.path.join(d, "pi.jsonl")), "pi"
        elif glob.glob(os.path.join(d, "hermes*.jsonl")):
            ev, harness = hermes_events(d), "hermes"
        else:
            continue
        out, ex, comp = replay(ev)
        if harness == "hermes":
            comp = hermes_session_comp(d)
        row = {"run": run, "harness": harness, **dict(out),
               "compress": summarise_comp(comp),
               "examples": {k: v[:6] for k, v in ex.items()}}
        rows.append(row)
        c = row["compress"]
        print(f"{run:<30} {harness:<6} calls={out['calls']:>4} "
              f"tasks={out['tasks']:>2} maxsame={out['max_identical_calls']}"
              f" warn={out.get('notice_generic_repeat', 0)} "
              f"outcome={out.get('notice_outcome_repeat', 0)} "
              f"veto={out.get('level_veto', 0) + out.get('level_breaker', 0)} "
              f"reread={out.get('reads_unchanged_repeat', 0)}/"
              f"{out.get('read_repeat_warn', 0)} "
              f"probe={out.get('probe_calls', 0)}:"
              f"{out.get('probe_chars', 0)}/{out.get('result_chars', 0)}ch "
              f"wander6/12={out.get('probe_wander_ge6_episodes', 0)}/"
              f"{out.get('probe_wander_ge12_episodes', 0)} "
              f"| shell={c['shell_results']} comp={c['compressed']} "
              f"prot={c['by_why'].get('protected', 0)} "
              f"saved={c['chars_saved']}ch")
    tot = collections.Counter()
    for r in rows:
        for k in ("calls", "notice_generic_repeat", "notice_outcome_repeat",
                  "level_veto", "level_breaker", "veto_episodes",
                  "keys_streak_ge_6", "keys_streak_ge_9",
                  "reads_unchanged_repeat", "read_repeat_warn",
                  "probe_calls", "probe_chars", "result_chars",
                  "probe_wander_ge6_episodes", "probe_wander_ge12_episodes",
                  "probe_wander_ge12_calls"):
            tot[k] += r.get(k, 0)
        tot["max_identical_calls"] = max(tot["max_identical_calls"],
                                         r.get("max_identical_calls", 0))
        for k in ("results", "shell_results", "compressed", "chars_saved",
                  "tokens_saved_est", "all_result_chars", "protected_chars"):
            tot["c_" + k] += r["compress"][k]
        tot["c_protected"] += r["compress"]["by_why"].get("protected", 0)
    print("\nTOTAL", json.dumps(dict(tot)))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"runs": rows, "total": dict(tot)}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
