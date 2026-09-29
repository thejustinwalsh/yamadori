#!/usr/bin/env python
"""CLM as a FEATURE EXTRACTOR: typed yes/no (Noul) questions about a turn,
decided as argmax(yes, no), measured against labels that already exist.

    python bench/clm/yesno.py [--only intent,phase,h4,split]

Operator's question (2026-09-27): "if we just do yes/no runs can it just be a
list of yes/no then we decide what skills?"

THE RENDERING (Contrastive-LM/CLM @ bb42c6c5bf914fd449bed2f6ca65be80602cb1f7,
src/clm/schema.py, read only):
  * state text  = to_text(state).strip() + "\\n\\n" + instructions.strip()
                  (schema.state_text; mcp/clm.state_text is the same for a
                  string state)
  * candidates  = noul keys in NOUL_KEYS order ("false", "true"); with no
                  criteria each is  f"{k}: {d}"  where d is
                  "No. This is false: {instructions}" /
                  "Yes. This is true: {instructions}"  (schema.candidates)
  * answer      = softmax over the two scaled cosines; p_true = noul.
We decide argmax(yes, no) -- p_true > 0.5 -- with no threshold chosen.

The encoder is llama-swap's `clm-encoder` through mcp/clm.py (gpu_room
lease). Option vectors go to a scratch cache, never index/clm_actions.npz.
One GPU consumer: the main model's slots must be idle and no hermes.exe
running, checked before each part and every CHECK_EVERY calls; if either
changes, the run stops and says so (exit 2).

Labels are the files' own (bench/skills/work_intent.jsonl `intent`), or,
for pagoda-h4, MECHANICAL facts of the transcript (hermes.jsonl, the full
tool stream; session.jsonl holds only the post-compaction tail with most
results summarised), stated in `h4_labels` below. The daily eval carries
NO phase labels; see `phase` for what is compared instead. Everything is
in-sample to our own labels. Test material (the Octopus / pagoda specs) is
read, never printed or stored: rows are named by id.

Written: bench/clm/results/yesno.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "results", "yesno.json")
H4 = "C:/Users/jwals/octo/logs/pagoda-h4-pagoda-xhigh-1/hermes.jsonl"
SLOTS = "http://127.0.0.1:11434/upstream/bonsai/slots"
CHECK_EVERY = 25

# ------------------------------------------------------------ questions --
# Fixed before the run; no wording is chosen after seeing results.
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


def noul_candidates(instructions: str) -> list[str]:
    """schema.candidates for a noul question with no criteria, in
    NOUL_KEYS order (false, true)."""
    ins = instructions.strip()
    return [f"false: No. This is false: {ins}",
            f"true: Yes. This is true: {ins}"]


# ------------------------------------------------------- one GPU consumer --
class Busy(RuntimeError):
    pass


def preflight() -> dict:
    with urllib.request.urlopen(SLOTS, timeout=10) as r:
        slots = json.loads(r.read())
    busy = [s.get("id") for s in slots if s.get("is_processing")]
    tl = subprocess.run(["tasklist", "/FI", "IMAGENAME eq hermes.exe"],
                        capture_output=True, text=True).stdout
    hermes = "hermes.exe" in tl.lower()
    if busy or hermes:
        raise Busy(f"main model slots processing {busy}; hermes.exe "
                   f"running: {hermes}")
    return {"slots_busy": busy, "hermes": hermes, "at": time.time()}


# ------------------------------------------------------------------ ask --
CALLS: list[dict] = []
RETRIES: list[dict] = []


_NTOK: dict[str, int] = {}


def pretrim(pieces: list[str]) -> list[str]:
    """The newest pieces whose token counts alone already exceed the window,
    plus nothing older. clm.fit_state (keep=tail) would drop every older
    piece anyway -- the kept ones already overflow -- so the result is the
    same; it only spares fit_state re-tokenising a long history once per
    dropped piece (quadratic on h4)."""
    import clm
    tok = clm.tokenizer()
    total, out = 0, []
    for p in reversed(pieces):
        s = p.strip()
        if s not in _NTOK:
            _NTOK[s] = tok.count(s)
        out.append(p)
        total += _NTOK[s]
        if total > clm.CAP + 64:
            break
    return list(reversed(out))


def ask(state, instructions: str, *, keep: str = "tail", part: str = "",
        item: str = "") -> dict:
    import clm
    before = None
    if isinstance(state, list) and keep == "tail" and len(state) > 1:
        full = len(state)
        state = pretrim(state)
        before = full - len(state)
    if len(CALLS) % CHECK_EVERY == 0:
        preflight()
    for attempt in range(3):
        try:
            d = clm.decide_detail(state, noul_candidates(instructions),
                                  instructions=instructions, keep=keep)
            break
        except clm.ClmUnavailable as e:
            # A retryable fact from the client (a 502 from a stale upstream
            # connection was seen once: 1 ms, the encoder stayed loaded).
            RETRIES.append({"call": len(CALLS), "code": e.code,
                            "situation": e.situation[:160],
                            "retryable": e.retryable})
            if not e.retryable or attempt == 2:
                raise
            preflight()
            time.sleep(1)
    p = d["probabilities"]
    rec = {"part": part, "item": item, "q": instructions,
           "p_true": round(p[1], 6), "yes": p[1] > p[0],
           "tokens": d["state"]["tokens"],
           "tokens_before": d["state"].get("tokens_before",
                                           d["state"]["tokens"]),
           "truncated": d["state"]["truncated"] or bool(before),
           "pretrimmed_pieces": before or 0,
           "state_ms": d["timing"]["state_encode_ms"],
           "total_ms": d["timing"]["total_ms"],
           "encoded_options": d["options"]["encoded"]}
    CALLS.append(rec)
    return rec


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
    """How confident CLM was, split by right and wrong answers:
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


# ============================================================ 1. intent ==
def part_intent() -> dict:
    sys.path.insert(0, os.path.join(ROOT, "mcp"))
    import route
    rows = []
    with open(os.path.join(ROOT, "bench", "skills", "work_intent.jsonl"),
              encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln and not ln.startswith("//"):
                rows.append(json.loads(ln))
    preflight()
    out = {"n": len(rows), "positives": sum(r["intent"] for r in rows),
           "wordings": {}, "rows": []}
    rule = [bool(route.work_intent(r["text"])) for r in rows]
    out["route.work_intent"] = confusion(
        [(r["intent"], g) for r, g in zip(rows, rule)])
    per = {q: [] for q in INTENT_Q}
    for i, r in enumerate(rows):
        row = {"text": r["text"], "intent": r["intent"], "rule": rule[i],
               "p_true": {}}
        for q in INTENT_Q:
            a = ask(r["text"], q, part="intent", item=str(i))
            per[q].append(a)
            row["p_true"][q] = a["p_true"]
        out["rows"].append(row)
    labels = [r["intent"] for r in rows]
    for q in INTENT_Q:
        recs = per[q]
        out["wordings"][q] = {
            "confusion": confusion([(y, a["yes"]) for y, a in
                                    zip(labels, recs)]),
            **pdist(recs, labels),
            "vs_rule": {
                "both_right": sum(1 for y, a, g in zip(labels, recs, rule)
                                  if a["yes"] == y and g == y),
                "clm_only_right": [rows[i]["text"] for i, (y, a, g) in
                                   enumerate(zip(labels, recs, rule))
                                   if a["yes"] == y and g != y],
                "rule_only_right": [rows[i]["text"] for i, (y, a, g) in
                                    enumerate(zip(labels, recs, rule))
                                    if a["yes"] != y and g == y],
                "both_wrong": [rows[i]["text"] for i, (y, a, g) in
                               enumerate(zip(labels, recs, rule))
                               if a["yes"] != y and g != y]}}
    # Cross-check (NOT a phase label): the implement question on the same
    # turns, against the build-intent label.
    imp = [ask(r["text"], PHASE_Q["implement"], part="intent_x_implement",
               item=str(i)) for i, r in enumerate(rows)]
    out["implement_question_vs_intent_label"] = {
        "confusion": confusion([(y, a["yes"]) for y, a in zip(labels, imp)]),
        **pdist(imp, labels)}
    return out


# ============================================================= 2. phase ==
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
    its result together). No piece is capped: clm.fit_state drops whole
    pieces from the front, then cuts the boundary piece (CLM's own left
    truncation)."""
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


def part_phase() -> dict:
    sys.path.insert(0, os.path.join(ROOT, "mcp"))
    import skill_classify
    pts = _daily_points()
    preflight()
    res = {"n_points": len(pts), "labels": (
        "bench/skills/daily_eval.jsonl carries NO phase labels (its labels "
        "are skills and areas). The only phase evidence in it: cat "
        "'bug-report' (5 rows) and the sequence seq-phase-debug's last turn "
        "(its id), both debugging. So: debug recall on those 6 points, and "
        "for every phase an AGREEMENT table CLM vs skill_classify's "
        "request_signals phases -- agreement, not accuracy."),
        "state": "transcript_pieces (user turns, replies, steps as text; "
                 "system prompt left out), keep=tail",
        "points": []}
    for p in pts:
        sig = skill_classify.request_signals(p["msgs"], p["route"],
                                             p["tools"])
        rule = set((sig.get("phases") or {}).keys())
        pieces = transcript_pieces(p["msgs"])
        rec = {"id": p["id"], "cat": p["cat"], "route": p["route"],
               "source": p["source"], "rule": sorted(rule), "clm": {}}
        for ph, q in PHASE_Q.items():
            a = ask(pieces, q, part="phase", item=p["id"])
            rec["clm"][ph] = {"p_true": a["p_true"], "yes": a["yes"],
                              "tokens": a["tokens_before"],
                              "truncated": a["truncated"]}
        res["points"].append(rec)
    debug_ev = [r for r in res["points"] if r["cat"] == "bug-report"
                or r["id"] == "seq-phase-debug#3"]
    res["debug_evidence"] = {
        "n": len(debug_ev),
        "clm_yes": sum(r["clm"]["debug"]["yes"] for r in debug_ev),
        "rule_yes": sum("debug" in r["rule"] for r in debug_ev),
        "rows": [(r["id"], r["clm"]["debug"]["p_true"], "debug" in r["rule"])
                 for r in debug_ev]}
    res["agreement"] = {}
    for ph in PHASE_Q:
        pairs = [(ph in r["rule"], r["clm"][ph]["yes"]) for r in res["points"]]
        c = confusion(pairs)
        res["agreement"][ph] = {
            "rule_yes_clm_yes": c["tp"], "rule_yes_clm_no": c["fn"],
            "rule_no_clm_yes": c["fp"], "rule_no_clm_no": c["tn"],
            "agree": c["tp"] + c["tn"], "n": c["n"],
            "clm_yes_ids": [r["id"] for r in res["points"]
                            if r["clm"][ph]["yes"]],
            "p_true_median": round(statistics.median(
                r["clm"][ph]["p_true"] for r in res["points"]), 4)}
    # How many phases each point gets.
    res["phases_per_point"] = {
        "clm": _hist(sum(v["yes"] for v in r["clm"].values())
                     for r in res["points"]),
        "rule": _hist(len(set(r["rule"]) & set(PHASE_Q))
                      for r in res["points"])}
    return res


def _hist(xs) -> dict:
    h: dict = {}
    for x in xs:
        h[str(x)] = h.get(str(x), 0) + 1
    return dict(sorted(h.items()))


# ================================================================ 3. h4 ==
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


def _h4_steps():
    sys.path.insert(0, os.path.join(ROOT, "mcp"))
    import deep
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
                          "sig": deep.error_signature(out) if failed else ""})
    return steps


def part_h4() -> dict:
    sys.path.insert(0, os.path.join(ROOT, "mcp"))
    import deep
    import skill_select
    steps = _h4_steps()
    task = open(os.path.join(os.path.dirname(H4), "prompt.md"),
                encoding="utf-8").read().strip()
    # The existing rules, on the same steps.
    msgs = [{"role": "user", "content": task}]
    for i, s in enumerate(steps):
        msgs.append({"role": "assistant", "content": s["said"],
                     "tool_calls": [{"id": f"call_{i}", "type": "function",
                                     "function": {"name": s["name"],
                                                  "arguments": json.dumps(
                                                      s["args"])}}]})
        msgs.append({"role": "tool", "tool_call_id": f"call_{i}",
                     "content": s["result"]})
    scan = deep.struggle_scan(msgs)
    at_step = {}
    for e in scan["events"]:
        if e["kind"] in ("tool_error_repeat", "failing_command_rerun"):
            at_step[(e["at"] - 2) // 2] = e["kind"]    # tool msg index -> step
    preflight()
    rows, prior, pieces = [], [], [f"User: {task}"]
    for i, s in enumerate(steps):
        lab = h4_labels(s["name"], s["args"], prior, s["failed"], s["sig"])
        call = {"type": "function", "function": {
            "name": s["name"], "arguments": json.dumps(s["args"])}}
        rule_reads = bool(skill_select.probed_packages(call))
        cur = step_text(s["said"], s["name"], json.dumps(s["args"]),
                        s["result"])
        bare = step_text("", s["name"], json.dumps(s["args"]), s["result"])
        pieces.append(cur)
        r = {"step": i, "tool": s["name"], "failed": s["failed"],
             "labels": lab, "rule": {"probed_packages": rule_reads,
                                     "deep_repeat": at_step.get(i)},
             "clm": {}}
        for key, q in SITU_Q.items():
            r["clm"][key] = {}
            for var, st in (("step", [cur]), ("history", list(pieces))):
                a = ask(st, q, part=f"h4:{key}:{var}", item=str(i))
                r["clm"][key][var] = {"p_true": a["p_true"], "yes": a["yes"],
                                      "tokens": a["tokens_before"],
                                      "truncated": a["truncated"]}
            if key == "reads_package":
                a = ask([bare], q, part="h4:reads_package:bare", item=str(i))
                r["clm"][key]["bare"] = {"p_true": a["p_true"],
                                         "yes": a["yes"]}
        # Was the earlier failure the strict/same-error label needs still
        # inside the history window? (the fit keeps the tail)
        if lab["repeat_failed_same_error"]:
            import clm
            kept_list = pretrim(list(pieces))
            pre = len(pieces) - len(kept_list)
            ids, info = clm.fit_state(kept_list, SITU_Q["repeat_failed"])
            earlier = [j for j, p in enumerate(prior)
                       if p["name"] == s["name"] and p["failed"]
                       and p["sig"] == s["sig"]]
            dropped = pre + info.get("dropped_pieces", 0)  # pieces[0]: task
            r["earlier_failure_in_window"] = any(j + 1 >= dropped
                                                 for j in earlier)
            r["earlier_failure_steps_back"] = [i - j for j in earlier]
        rows.append(r)
        prior.append({"name": s["name"], "args": json.dumps(
            s["args"], sort_keys=True), "failed": s["failed"],
            "sig": s["sig"]})
    out = {"n_steps": len(steps), "source": H4, "rows": rows,
           "label_counts": {k: sum(r["labels"][k] for r in rows)
                            for k in rows[0]["labels"]}, "tables": {}}
    for key, lab_keys in (("reads_package", ["reads_package"]),
                          ("repeat_failed", ["repeat_failed_strict",
                                             "repeat_failed_same_error"])):
        for lk in lab_keys:
            ys = [r["labels"][lk] for r in rows]
            for var in (("step", "history", "bare") if key == "reads_package"
                        else ("step", "history")):
                recs = [{"yes": r["clm"][key][var]["yes"],
                         "p_true": r["clm"][key][var]["p_true"]}
                        for r in rows]
                out["tables"][f"{lk}|clm:{var}"] = {
                    "confusion": confusion([(y, a["yes"]) for y, a in
                                            zip(ys, recs)]),
                    **pdist(recs, ys)}
            if key == "reads_package":
                out["tables"][f"{lk}|rule:probed_packages"] = {
                    "confusion": confusion([(y, r["rule"]["probed_packages"])
                                            for y, r in zip(ys, rows)])}
            else:
                out["tables"][f"{lk}|rule:deep.struggle_scan"] = {
                    "confusion": confusion([(y, bool(r["rule"]["deep_repeat"]))
                                            for y, r in zip(ys, rows)])}
    return out


# ============================================================= 4. split ==
def part_split() -> dict:
    """Long user turns (the daily eval's source rows: Octopus and pagoda
    specs; nothing else in these sets passes 2,047 tokens). Each question is
    asked three ways: the whole turn with CLM's own left truncation
    (keep=tail), the head kept (keep=head), and in consecutive paragraph
    chunks that each fit, taking any chunk's yes."""
    import clm
    sys.path.insert(0, os.path.join(ROOT, "mcp"))
    import route
    pts = [p for p in _daily_points() if p["source"]]
    tok = clm.tokenizer()
    preflight()
    qs = [INTENT_Q[0]] + list(PHASE_Q.values())
    out = {"rows": []}
    for p in pts:
        text = next(m["content"] for m in reversed(p["msgs"])
                    if m["role"] == "user")
        n = tok.count(text)
        rec = {"id": p["id"], "tokens": n,
               "route.work_intent": bool(route.work_intent(text)), "q": {}}
        paras = [x for x in text.split("\n\n") if x.strip()]
        for q in qs:
            budget = clm.CAP - tok.count("\n\n" + q) - 8
            chunks, cur = [], []
            for para in paras:
                if cur and tok.count("\n\n".join(cur + [para])) > budget:
                    chunks.append("\n\n".join(cur))
                    cur = []
                cur.append(para)
            if cur:
                chunks.append("\n\n".join(cur))
            tail = ask(text, q, keep="tail", part="split:tail", item=p["id"])
            head = ask(text, q, keep="head", part="split:head", item=p["id"])
            ch = [ask(c, q, part="split:chunk", item=p["id"]) for c in chunks]
            rec["q"][q] = {"tail": tail["p_true"], "head": head["p_true"],
                           "chunks": [c["p_true"] for c in ch],
                           "chunk_tokens": [c["tokens_before"] for c in ch],
                           "any_chunk_yes": any(c["yes"] for c in ch),
                           "truncated": tail["truncated"]}
        out["rows"].append(rec)
    return out


# ================================================================= main ==
def latency_summary() -> dict:
    def q(xs):
        xs = sorted(xs)
        return {"n": len(xs), "p50": round(statistics.median(xs), 1),
                "p90": round(xs[int(0.9 * (len(xs) - 1))], 1),
                "max": round(xs[-1], 1)} if xs else None
    bands = [(0, 64), (64, 256), (256, 1024), (1024, 2048)]
    out = {"all_total_ms": q([c["total_ms"] for c in CALLS]),
           "all_state_ms": q([c["state_ms"] for c in CALLS]),
           "by_state_tokens": {}}
    for lo, hi in bands:
        xs = [c for c in CALLS if lo <= c["tokens"] < hi]
        out["by_state_tokens"][f"{lo}-{hi - 1}"] = {
            "state_ms": q([c["state_ms"] for c in xs]),
            "total_ms": q([c["total_ms"] for c in xs])}
    out["truncated_calls"] = sum(c["truncated"] for c in CALLS)
    out["option_texts_encoded"] = sum(c["encoded_options"] for c in CALLS)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="intent,phase,h4,split")
    a = ap.parse_args(argv)
    os.environ["YAMADORI_CLM_ACTIONS"] = os.path.join(
        tempfile.mkdtemp(prefix="clm_yesno_"), "actions.npz")
    sys.path.insert(0, os.path.join(ROOT, "mcp"))
    import clm
    res = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "preflight": preflight(),
           "rendering": {"schema": "Contrastive-LM/CLM@bb42c6c5 "
                         "src/clm/schema.py state_text + candidates (noul)",
                         "candidates_example": noul_candidates(INTENT_Q[0]),
                         "decision": "argmax(yes, no): p_true > 0.5"},
           "questions": {"intent": INTENT_Q, "phase": PHASE_Q,
                         "situations": SITU_Q},
           "encoder": clm.ENCODER_ID, "status": "complete"}
    parts = {"intent": part_intent, "phase": part_phase, "h4": part_h4,
             "split": part_split}
    code = 0
    try:
        for name in a.only.split(","):
            t0 = time.time()
            res[name] = parts[name]()
            res[name]["seconds"] = round(time.time() - t0, 1)
            print(f"[{name}] done in {res[name]['seconds']} s, "
                  f"{len(CALLS)} calls so far", flush=True)
            _write(res)
    except Busy as e:
        res["status"] = f"STOPPED: {e}"
        print(res["status"], flush=True)
        code = 2
    except Exception as e:                                       # noqa: BLE001
        res["status"] = f"FAILED: {type(e).__name__}: {e}"[:400]
        print(res["status"], flush=True)
        code = 1
    _write(res)
    print("wrote", OUT)
    return code


def _write(res: dict) -> None:
    res["retries"] = RETRIES
    res["latency"] = latency_summary()
    res["calls"] = CALLS
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    sys.exit(main())
