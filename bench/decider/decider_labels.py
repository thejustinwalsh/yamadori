#!/usr/bin/env python
"""The labels, states and scoring bench/decider/bonsai_decider.py measures
the Bonsai typed decider against.

Moved here from bench/clm/yesno.py on 2026-09-29, when CLM and its bench
were removed (the way back is commit e360d37). Only the parts the Bonsai
bench reads came across, unchanged except `_h4_steps`, whose failure
signature was deep.error_signature (mcp/deep.py, removed the same day) and
is now `error_signature` below, the same rule copied.

Labels are the files' own (bench/skills/work_intent.jsonl `intent`), or,
for pagoda-h4, MECHANICAL facts of the transcript (hermes.jsonl, the full
tool stream; session.jsonl holds only the post-compaction tail with most
results summarised), stated in `h4_labels` below. The daily eval carries
NO phase labels. Everything is in-sample to our own labels. Test material
(the Octopus / pagoda specs) is read, never printed or stored: rows are
named by id.
"""
from __future__ import annotations

import json
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
H4 = "C:/Users/jwals/octo/logs/pagoda-h4-pagoda-xhigh-1/hermes.jsonl"

# ------------------------------------------------------------ questions --
# Fixed before the first run; no wording is chosen after seeing results.
INTENT_Q = [
    "Is the user asking for something to be built, made or changed?",  # operator's
    "Does the user want something built, made or changed?",
    "Is this message a request to build, make or change something?",
]
PHASE_Q = {
    "plan": "Is the agent planning the work?",
    "implement": "Is the agent implementing the work?",
    "debug": "Is the agent debugging a failure?",          # operator's
    "verify": "Is the agent verifying or testing?",        # operator's
}
SITU_Q = {
    "repeat_failed": "Is the agent repeating a step that already failed?",
    "reads_package": ("Is the agent reading a package's installed files to "
                      "learn its API?"),
}


# ------------------------------------------------------------- scoring --
def confusion(pairs) -> dict:
    """pairs of (label bool, predicted bool)."""
    tp = sum(1 for y, p in pairs if y and p)
    fp = sum(1 for y, p in pairs if not y and p)
    fn = sum(1 for y, p in pairs if y and not p)
    tn = sum(1 for y, p in pairs if not y and not p)
    n = tp + fp + fn + tn
    return {"n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "acc": round((tp + tn) / n, 4) if n else None,
            "precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "recall": round(tp / (tp + fn), 4) if tp + fn else None}


def pdist(recs, labels) -> dict:
    """How confident the decider was, split by right and wrong answers:
    p(the answer it gave) for each."""
    right, wrong, pt_pos, pt_neg = [], [], [], []
    for r, y in zip(recs, labels):
        conf = r["p_true"] if r["yes"] else 1 - r["p_true"]
        (right if r["yes"] == y else wrong).append(conf)
        (pt_pos if y else pt_neg).append(r["p_true"])

    def q(xs):
        if not xs:
            return None
        xs = sorted(xs)
        return {"n": len(xs), "min": round(xs[0], 4),
                "p25": round(xs[len(xs) // 4], 4),
                "median": round(statistics.median(xs), 4),
                "p75": round(xs[(3 * len(xs)) // 4], 4),
                "max": round(xs[-1], 4)}
    return {"confidence_right": q(right), "confidence_wrong": q(wrong),
            "p_true_on_label_yes": q(pt_pos), "p_true_on_label_no": q(pt_neg)}


# ------------------------------------------------------------ the daily eval
def _daily_points():
    """(label, cat, messages, tools, is_source) for every point of the daily
    eval that carries an `expect`, built exactly as replay_selection.daily
    builds them."""
    gpu = os.environ.get("YAMADORI_GPU_ROOM")
    sys.path.insert(0, os.path.join(ROOT, "bench", "skills"))
    import replay_selection as rs         # sets YAMADORI_GPU_ROOM=0
    if gpu is None:
        os.environ.pop("YAMADORI_GPU_ROOM", None)
    else:
        os.environ["YAMADORI_GPU_ROOM"] = gpu
    sys.path.insert(0, os.path.join(ROOT, "mcp"))
    import route
    pts = []
    for row in rs._daily_rows():
        tools = row.get("tools") or rs.HERMES_TOOLS
        msgs = [{"role": "system", "content": row.get("system")
                 or rs.DAILY_SYSTEM}] + list(row.get("prior") or [])
        turns = row.get("turns") or [{"user": rs._user_text(row),
                                      "expect": row}]
        for ti, t in enumerate(turns):
            if "user" in t:
                msgs = msgs + [{"role": "user", "content": t["user"]}]
            if "reply" in t:
                msgs = msgs + [{"role": "assistant", "content": t["reply"]}]
            if "step" in t:
                st_ = dict(t["step"])
                res_ = str(st_.get("result") or "")
                if res_.startswith("FILL:"):
                    n = int(res_[5:])
                    line = ("npm warn deprecated inflight@1.0.6: this module "
                            "is not supported, and leaks memory\n")
                    st_["result"] = (line * (n // len(line) + 1))[:n]
                msgs = msgs + [{"role": "assistant", "content": "",
                                "tool_calls": [rs._call(ti, st_["name"],
                                                        st_.get("args") or {})]},
                               {"role": "tool", "tool_call_id": f"call_{ti}",
                                "content": st_.get("result") or ""}]
            if "expect" not in t:
                continue
            clean = [m for m in msgs if m.get("role") in (
                "system", "user", "assistant", "tool")]
            try:
                rc = route.classify(clean, client_tools=tools,
                                    util={"utility": False}, gate=None,
                                    dbs={})["class"]
            except Exception:                                    # noqa: BLE001
                rc = None
            label = row["id"] + (f"#{ti}" if len(turns) > 1 else "")
            pts.append({"id": label, "cat": row.get("cat"), "msgs": clean,
                        "tools": tools, "route": rc,
                        "source": bool(row.get("source"))})
    return pts


def transcript_pieces(msgs: list[dict]) -> list[str]:
    """The conversation as plain text pieces, oldest first; the system
    prompt left out. One piece per user turn, reply, or step (the call and
    its result together). No piece is capped."""
    out, pending = [], {}
    for m in msgs:
        role = m.get("role")
        c = m.get("content") or ""
        if role == "user":
            out.append(f"User: {c}")
        elif role == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                f = tc.get("function") or {}
                pending[tc.get("id")] = (c, f.get("name"),
                                         f.get("arguments") or "")
        elif role == "assistant":
            out.append(f"Assistant: {c}")
        elif role == "tool":
            said, name, args = pending.pop(m.get("tool_call_id"),
                                           ("", "?", ""))
            out.append(step_text(said, name, args, c))
    return out


def step_text(said: str, name: str, args: str, result: str) -> str:
    head = f"Assistant: {said.strip()}\n" if said and said.strip() else ""
    return f"{head}Assistant called {name}: {args}\nResult: {result}"


# ------------------------------------------------------------ pagoda-h4 --
_READ_VERB = re.compile(
    r"\b(?:cat|head|tail|grep|sed|awk|less|ls|find|wc)\b[^|;&\n]*?"
    r"(?<![\^\w])(?:\./)?node_modules/")


def h4_labels(name: str, args: dict, prior: list[dict], failed: bool,
              sig: str) -> dict:
    """MECHANICAL labels from the transcript's facts.

    reads_package: a read tool (read_file, search_files) whose path is under
      node_modules/, or a terminal command in which a read verb (cat, head,
      tail, grep, sed, awk, less, ls, find, wc) is followed, within the same
      shell segment, by a node_modules/ path (not a `^node_modules/` filter
      pattern). Intent ("to learn its API") is not visible: a version read
      counts.
    repeat_failed_strict: the SAME call (tool + arguments) was made before,
      that earlier call failed, and this one fails again with the same first
      error line.
    repeat_failed_same_error: this call fails with the same first error line
      that the same tool already failed with (any arguments). Failure is
      Hermes' own `is_error` flag on the result."""
    s = json.dumps(args, sort_keys=True)
    if name in ("read_file", "search_files"):
        path = str(args.get("path") or "")
        reads = "node_modules/" in path.replace("\\", "/")
    elif name == "terminal":
        reads = bool(_READ_VERB.search(str(args.get("command") or "")))
    else:
        reads = False
    strict = failed and any(p["name"] == name and p["args"] == s
                            and p["failed"] and p["sig"] == sig
                            for p in prior)
    same = failed and any(p["name"] == name and p["failed"]
                          and p["sig"] == sig for p in prior)
    return {"reads_package": reads, "repeat_failed_strict": strict,
            "repeat_failed_same_error": same}


_EXIT_KEYS = ("exit_code", "exitCode", "returncode", "return_code",
              "exit_status", "exitStatus")


def error_signature(text: str) -> str:
    """The first non-blank line of a failing tool result's error message,
    exactly as written (stripped); "exit N" when there is no text at all. A
    JSON result is read by its `error` field when it has one, else its
    output streams. (deep.error_signature's rule, copied when mcp/deep.py
    was removed.)"""
    t = (text or "").strip()
    msg, code = t, None
    if t.startswith("{"):
        try:
            d = json.loads(t)
        except ValueError:
            d = None
        if isinstance(d, dict):
            code = next((d[k] for k in _EXIT_KEYS if isinstance(d.get(k), int)
                         and not isinstance(d.get(k), bool)), None)
            err = d.get("error")
            if isinstance(err, dict):
                err = err.get("message") or json.dumps(err, sort_keys=True)
            if isinstance(err, str) and err.strip():
                msg = err
            else:
                msg = "\n".join(str(d.get(k) or "") for k in
                                ("stderr", "output", "stdout", "result"))
    first = next((ln.strip() for ln in msg.splitlines() if ln.strip()), "")
    if not first:
        return f"exit {code}" if code is not None else ""
    return first


def _h4_steps():
    ev = []
    with open(H4, encoding="utf-8") as f:
        for ln in f:
            try:
                ev.append(json.loads(ln))
            except ValueError:
                pass             # Hermes' "compacting context" status lines
    steps, pend, said = [], [], []
    for e in ev:
        t = e.get("type")
        if t == "text":
            said.append(e.get("text") or "")
        elif t == "tool_use":
            pend.append((e, "".join(said).strip()))
            said = []
        elif t == "tool_result":
            u, s = pend.pop(0)
            assert u["name"] == e["name"]
            out = e["output"] if isinstance(e["output"], str) \
                else json.dumps(e["output"])
            failed = bool(e.get("is_error"))
            steps.append({"name": u["name"], "args": u["input"],
                          "said": s, "result": out, "failed": failed,
                          "sig": error_signature(out) if failed else ""})
    return steps
