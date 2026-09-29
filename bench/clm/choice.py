#!/usr/bin/env python
"""CLM CHOICE questions (not Noul) as features: one closed set of genuinely
contrasting options per question, argmax, no threshold.

    python bench/clm/choice.py [--only intent,phase,h4]

Operator (2026-09-27): "none of these answer yes no questions right, they
always classify multiple choices right?" -- the follow-up to
bench/clm/yesno.py (whose Noul results this reads back for the tables).

THE RENDERING (Contrastive-LM/CLM @ bb42c6c5, src/clm/schema.py, read only):
  * state text = state.strip() + "\\n\\n" + instructions.strip()
  * a choice question's candidate for key k is to_text(criteria[k]) -- the
    DESCRIPTION ONLY, nothing prefixed (the key is never embedded unless the
    description is empty)
  * answer = softmax over scale * cos; the choice is the argmax.

The option sets below were fixed BEFORE the run, verbatim from the
coordinator's message, and are not tuned after. The state is cut HERE (never
by clm.fit_state): a conversation keeps its newest pieces and the oldest kept
piece is cut from its front; a single h4 step keeps its head (the assistant's
words and the call come first; the result's tail is cut). The final text is
checked to fit 2,047 tokens with the question, so fit_state never cuts.
Same one-GPU-consumer checks as yesno.py. In-sample to our own labels.

Written: bench/clm/results/choice.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "results", "choice.json")
NOUL = os.path.join(HERE, "results", "yesno.json")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "mcp"))

import yesno  # noqa: E402

# --------------------------------------------- FIXED BEFORE THE RUN -------
INTENT = ("What is this message?", {
    "build": "a request to build, make or change something",
    "question": "a question or a request for an explanation",
    "feedback": "feedback, thanks or an acknowledgement",
    "stop": "an instruction to stop, wait or hold off",
    "status": "a status check or other conversation",
})
PHASE = ("What is the agent doing right now?", {
    "plan": "planning (deciding files, order or design before writing)",
    "implement": "implementing (writing or changing code)",
    "debug": "debugging (a failure or error is being investigated)",
    "verify": "verifying (running, testing or checking the result)",
})
STEP = ("What is this step doing?", {
    "read_project": "reading the project's own files",
    "read_package": "reading an installed package's files to learn its API",
    "write_project": "writing or editing project code",
    "scratch": "writing a throwaway file to test behaviour",
    "run": "running a build, test or install",
    "repeat_failed": "repeating a command that already failed the same way",
})

CALLS: list[dict] = []


# ------------------------------------------------------------ the cut ----
def _fits(text: str, q: str) -> bool:
    import clm
    return len(clm.tokenizer().ids(clm.state_text(text, q))) <= clm.CAP


def cut_tail(pieces: list[str], q: str) -> tuple[str, dict]:
    """Newest pieces first; the oldest kept piece cut from its front."""
    import clm
    tok = clm.tokenizer()
    budget = clm.CAP - len(tok.ids("\n\n" + q.strip()))
    kept, total = [], 0
    for p in reversed([p.strip() for p in pieces if p.strip()]):
        n = len(tok.ids(p)) + (2 if kept else 0)
        if total + n <= budget:
            kept.insert(0, p)
            total += n
            continue
        room = budget - total - (2 if kept else 0)
        if room > 16:
            ids = tok.ids(p)[-room:]
            kept.insert(0, tok._t.decode(ids))
        break
    text = "\n\n".join(kept)
    info = {"pieces": len(pieces), "kept": len(kept),
            "tokens_before": len(tok.ids("\n\n".join(pieces)))}
    while not _fits(text, q):          # a decode that re-tokenised longer
        text = text[max(1, len(text) // 200):]
    info["cut"] = info["tokens_before"] > budget
    return text, info


def cut_head(text: str, q: str) -> tuple[str, dict]:
    import clm
    tok = clm.tokenizer()
    budget = clm.CAP - len(tok.ids("\n\n" + q.strip()))
    ids = tok.ids(text.strip())
    info = {"tokens_before": len(ids), "cut": len(ids) > budget}
    if len(ids) > budget:
        text = tok._t.decode(ids[:budget])
    while not _fits(text, q):
        text = text[:-max(1, len(text) // 200)]
    return text, info


# ------------------------------------------------------------------ ask --
def ask(state: str, spec, *, part: str, item: str, cutinfo=None) -> dict:
    import clm
    q, crit = spec
    if len(CALLS) % yesno.CHECK_EVERY == 0:
        yesno.preflight()
    keys = list(crit)
    for attempt in range(3):
        try:
            d = clm.decide_detail(state, [crit[k] for k in keys],
                                  instructions=q, keep="tail")
            break
        except clm.ClmUnavailable as e:
            if not e.retryable or attempt == 2:
                raise
            yesno.preflight()
            time.sleep(1)
    assert not d["state"]["truncated"], "the cut must leave nothing to cut"
    p = dict(zip(keys, (round(x, 6) for x in d["probabilities"])))
    pick = max(keys, key=p.__getitem__)
    rec = {"part": part, "item": item, "pick": pick, "p": p,
           "tokens": d["state"]["tokens"],
           "cut": bool((cutinfo or {}).get("cut")),
           "tokens_before": (cutinfo or {}).get("tokens_before",
                                                d["state"]["tokens"]),
           "state_ms": d["timing"]["state_encode_ms"],
           "total_ms": d["timing"]["total_ms"],
           "encoded_options": d["options"]["encoded"]}
    CALLS.append(rec)
    return rec


def _q(xs):
    if not xs:
        return None
    xs = sorted(xs)
    return {"n": len(xs), "min": round(xs[0], 4),
            "median": round(statistics.median(xs), 4),
            "max": round(xs[-1], 4)}


def auc(pos, neg):
    if not pos or not neg:
        return None
    s = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    return round(s / (len(pos) * len(neg)), 3)


def binary(recs, labels, key) -> dict:
    """Score argmax == key against a boolean label."""
    pairs = [(y, r["pick"] == key) for r, y in zip(recs, labels)]
    right = [r["p"][r["pick"]] for r, (y, g) in zip(recs, pairs) if y == g]
    wrong = [r["p"][r["pick"]] for r, (y, g) in zip(recs, pairs) if y != g]
    return {"confusion": yesno.confusion(pairs),
            "p_pick_when_right": _q(right), "p_pick_when_wrong": _q(wrong),
            f"p_{key}_on_label_yes": _q([r["p"][key] for r, y in
                                          zip(recs, labels) if y]),
            f"p_{key}_on_label_no": _q([r["p"][key] for r, y in
                                         zip(recs, labels) if not y]),
            "auc_p_key": auc([r["p"][key] for r, y in zip(recs, labels) if y],
                             [r["p"][key] for r, y in zip(recs, labels)
                              if not y])}


def _noul():
    try:
        with open(NOUL, encoding="utf-8") as f:
            return json.load(f)
    except OSError:
        return {}


# ============================================================ 1. intent ==
def part_intent() -> dict:
    import route
    rows = []
    with open(os.path.join(ROOT, "bench", "skills", "work_intent.jsonl"),
              encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln and not ln.startswith("//"):
                rows.append(json.loads(ln))
    yesno.preflight()
    labels = [r["intent"] for r in rows]
    recs = [ask(r["text"], INTENT, part="intent", item=str(i))
            for i, r in enumerate(rows)]
    rule = [bool(route.work_intent(r["text"])) for r in rows]
    out = {"n": len(rows), "question": INTENT[0], "options": INTENT[1],
           "clm_choice": binary(recs, labels, "build"),
           "route.work_intent": yesno.confusion(list(zip(labels, rule))),
           "pick_by_label": {
               "intent": _hist(r["pick"] for r, y in zip(recs, labels) if y),
               "no_intent": _hist(r["pick"] for r, y in zip(recs, labels)
                                  if not y)},
           "errors": [{"text": rows[i]["text"], "intent": labels[i],
                       "pick": recs[i]["pick"], "p": recs[i]["p"]}
                      for i in range(len(rows))
                      if (recs[i]["pick"] == "build") != labels[i]],
           "rows": [{"text": r["text"], "intent": r["intent"],
                     "rule": g, "pick": a["pick"], "p": a["p"]}
                    for r, g, a in zip(rows, rule, recs)]}
    nz = _noul().get("intent") or {}
    w1 = (nz.get("wordings") or {}).get(yesno.INTENT_Q[0])
    if w1:
        out["clm_noul_w1"] = w1["confusion"]
    return out


def _hist(xs) -> dict:
    h: dict = {}
    for x in xs:
        h[x] = h.get(x, 0) + 1
    return dict(sorted(h.items(), key=lambda kv: -kv[1]))


# ============================================================= 2. phase ==
def part_phase() -> dict:
    import skill_classify
    pts = yesno._daily_points()
    yesno.preflight()
    points = []
    for p in pts:
        sig = skill_classify.request_signals(p["msgs"], p["route"],
                                             p["tools"])
        rule = sorted(set((sig.get("phases") or {})) & set(PHASE[1]))
        text, info = cut_tail(yesno.transcript_pieces(p["msgs"]), PHASE[0])
        a = ask(text, PHASE, part="phase", item=p["id"], cutinfo=info)
        points.append({"id": p["id"], "cat": p["cat"], "rule": rule,
                       "pick": a["pick"], "p": a["p"], "cut": info["cut"],
                       "tokens_before": info["tokens_before"]})
    ev = [x for x in points if x["cat"] == "bug-report"
          or x["id"] == "seq-phase-debug#3"]
    out = {"n_points": len(points), "question": PHASE[0],
           "options": PHASE[1],
           "labels": "no phase labels exist in daily_eval.jsonl; debug "
                     "recall on the 6 evidence points, and AGREEMENT with "
                     "request_signals (not accuracy)",
           "debug_evidence": {"n": len(ev),
                              "clm_choice_debug": sum(x["pick"] == "debug"
                                                      for x in ev),
                              "rule_debug": sum("debug" in x["rule"]
                                                for x in ev),
                              "rows": [(x["id"], x["pick"], x["p"]["debug"],
                                        x["rule"]) for x in ev]},
           "pick_hist": _hist(x["pick"] for x in points),
           "rule_phase_count_hist": _hist(str(len(x["rule"]))
                                          for x in points),
           "pick_in_rule_phases": {
               "points_with_a_rule_phase": sum(1 for x in points
                                               if x["rule"]),
               "pick_is_one_of_them": sum(1 for x in points
                                          if x["rule"] and x["pick"]
                                          in x["rule"])},
           "agreement": {}, "points": points}
    for ph in PHASE[1]:
        c = yesno.confusion([(ph in x["rule"], x["pick"] == ph)
                             for x in points])
        out["agreement"][ph] = {"rule_yes_clm_yes": c["tp"],
                                "rule_yes_clm_no": c["fn"],
                                "rule_no_clm_yes": c["fp"],
                                "rule_no_clm_no": c["tn"],
                                "agree": c["tp"] + c["tn"], "n": c["n"],
                                "auc_p_vs_rule": auc(
                                    [x["p"][ph] for x in points
                                     if ph in x["rule"]],
                                    [x["p"][ph] for x in points
                                     if ph not in x["rule"]])}
    nz = _noul().get("phase") or {}
    if nz:
        out["clm_noul_debug_evidence"] = {
            k: nz["debug_evidence"][k] for k in ("n", "clm_yes", "rule_yes")}
        out["clm_noul_agreement"] = {
            ph: {k: v for k, v in a.items() if k != "clm_yes_ids"}
            for ph, a in (nz.get("agreement") or {}).items()}
    return out


# ================================================================ 3. h4 ==
def part_h4() -> dict:
    import deep
    import skill_select
    steps = yesno._h4_steps()
    task = open(os.path.join(os.path.dirname(yesno.H4), "prompt.md"),
                encoding="utf-8").read().strip()
    msgs = [{"role": "user", "content": task}]
    for i, s in enumerate(steps):
        msgs.append({"role": "assistant", "content": s["said"],
                     "tool_calls": [{"id": f"call_{i}", "type": "function",
                                     "function": {"name": s["name"],
                                                  "arguments": json.dumps(
                                                      s["args"])}}]})
        msgs.append({"role": "tool", "tool_call_id": f"call_{i}",
                     "content": s["result"]})
    rep = {(e["at"] - 2) // 2 for e in deep.struggle_scan(msgs)["events"]
           if e["kind"] in ("tool_error_repeat", "failing_command_rerun")}
    yesno.preflight()
    rows, prior = [], []
    for i, s in enumerate(steps):
        lab = yesno.h4_labels(s["name"], s["args"], prior, s["failed"],
                              s["sig"])
        call = {"type": "function", "function": {
            "name": s["name"], "arguments": json.dumps(s["args"])}}
        text, info = cut_head(yesno.step_text(s["said"], s["name"],
                                              json.dumps(s["args"]),
                                              s["result"]), STEP[0])
        a = ask(text, STEP, part="h4", item=str(i), cutinfo=info)
        rows.append({"step": i, "tool": s["name"], "labels": lab,
                     "rule": {"probed_packages": bool(
                         skill_select.probed_packages(call)),
                              "deep_repeat": i in rep},
                     "pick": a["pick"], "p": a["p"], "cut": info["cut"],
                     "tokens_before": info["tokens_before"]})
        prior.append({"name": s["name"], "args": json.dumps(
            s["args"], sort_keys=True), "failed": s["failed"],
            "sig": s["sig"]})
    out = {"n_steps": len(rows), "question": STEP[0], "options": STEP[1],
           "state": "current step only: the assistant's words, the call, "
                    "the result; cut at the head when over the window",
           "label_counts": {k: sum(r["labels"][k] for r in rows)
                            for k in rows[0]["labels"]},
           "pick_hist": _hist(r["pick"] for r in rows),
           "pick_on_reads_package": _hist(r["pick"] for r in rows
                                          if r["labels"]["reads_package"]),
           "pick_on_same_error": [(r["step"], r["pick"], r["p"])
                                  for r in rows
                                  if r["labels"]["repeat_failed_same_error"]],
           "tables": {}, "rows": rows}
    for lk, key, rk in (("reads_package", "read_package", "probed_packages"),
                        ("repeat_failed_same_error", "repeat_failed",
                         "deep_repeat")):
        ys = [r["labels"][lk] for r in rows]
        out["tables"][lk] = {
            "clm_choice": binary(rows, ys, key),
            f"rule:{rk}": yesno.confusion([(y, r["rule"][rk])
                                           for y, r in zip(ys, rows)])}
    nz = _noul().get("h4") or {}
    for k in ("reads_package|clm:step", "repeat_failed_same_error|clm:step"):
        if k in (nz.get("tables") or {}):
            out["tables"][k.split("|")[0]]["clm_noul_step"] = \
                nz["tables"][k]["confusion"]
    return out


# ================================================================= main ==
def latency() -> dict:
    def q(xs):
        xs = sorted(xs)
        return {"n": len(xs), "p50": round(statistics.median(xs), 1),
                "p90": round(xs[int(0.9 * (len(xs) - 1))], 1),
                "max": round(xs[-1], 1)} if xs else None
    out = {"all_total_ms": q([c["total_ms"] for c in CALLS]),
           "by_state_tokens": {}, "cut_states": sum(c["cut"] for c in CALLS),
           "option_texts_encoded": sum(c["encoded_options"] for c in CALLS)}
    for lo, hi in [(0, 64), (64, 256), (256, 1024), (1024, 2048)]:
        xs = [c for c in CALLS if lo <= c["tokens"] < hi]
        out["by_state_tokens"][f"{lo}-{hi - 1}"] = {
            "state_ms": q([c["state_ms"] for c in xs]),
            "total_ms": q([c["total_ms"] for c in xs])}
    return out


def _write(res):
    res["latency"] = latency()
    res["calls"] = CALLS
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="intent,phase,h4")
    a = ap.parse_args(argv)
    os.environ["YAMADORI_CLM_ACTIONS"] = os.path.join(
        tempfile.mkdtemp(prefix="clm_choice_"), "actions.npz")
    import clm
    res = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "preflight": yesno.preflight(), "encoder": clm.ENCODER_ID,
           "rendering": "schema.candidates (choice): each option embedded "
                        "as its description only; argmax",
           "status": "complete"}
    parts = {"intent": part_intent, "phase": part_phase, "h4": part_h4}
    code = 0
    try:
        for name in a.only.split(","):
            t0 = time.time()
            res[name] = parts[name]()
            res[name]["seconds"] = round(time.time() - t0, 1)
            print(f"[{name}] done in {res[name]['seconds']} s, "
                  f"{len(CALLS)} calls", flush=True)
            _write(res)
    except yesno.Busy as e:
        res["status"] = f"STOPPED: {e}"
        code = 2
    except Exception as e:                                       # noqa: BLE001
        res["status"] = f"FAILED: {type(e).__name__}: {e}"[:400]
        code = 1
    print(res["status"], flush=True)
    _write(res)
    return code


if __name__ == "__main__":
    sys.exit(main())
