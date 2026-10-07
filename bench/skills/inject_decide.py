#!/usr/bin/env python
"""READ THE INJECTOR'S QUESTIONS ON ONE MODEL (GPU): every labelled case's
stage-2 and stage-3 questions through the real decider (decide_turn.Turn ->
decider_bonsai.read, the typed readout), in each question VARIANT, the
decisions logged for bench/skills/inject_tune.py.

    python bench/skills/inject_decide.py --cases DIR/cases.jsonl \\
        --model bonsai [--variants a,b,c] [--limit N] [--dry-run]

ONE GPU CONSUMER AT A TIME (AGENTS.md): run only inside a window the
coordinator granted, with `--model` loaded. Before anything is sent it reads
llama-swap's GET /running (which never loads a model) and REFUSES unless
`--model` is loaded and nothing else is using the card's main slot.

THE VARIANTS (docs/JJAVA.md 4: "if a question set won't separate, rewrite
the question and state and measure again" -- the operator: "no easy
bails"). All are read in one window so a rewrite needs no second one:

  a  skill_inject.ITEM_Q / ITEM_LEVELS as served (a Score, 4 levels; the
     belief is P(top level))
  b  a Score of 3 levels: the "already done" level folded out (is the fact
     off the material / in its area but not the thing at hand / the thing
     at hand)
  c  a Noul per item: the top level as a statement to judge
  d  variant a's question over a REWRITTEN STATE (goal_state): the
     session's goal named first ("GOAL ... NOW: ..."), the turn after it
     -- the state rewrite of docs/JJAVA.md 4.2, read on a second Turn per
     case (one more state placement)
  e  variant a with the levels labelled by their own digits 0-3 instead of
     positional letters (decider_bonsai THE SCORE LABELS, SGLang's level
     labels; behind the switch, default letters, until this measures it)
  stage 3 (all variants): skill_inject.INJECT_Q over the shortlist the
     variant's own untuned gate passes; with none passed, over the variant's
     two items of highest belief (labelled `forced`, so stage 3 has
     decisions on every case to tune on).

THE REWORK OF 2026-10-06 (mcp/skill_inject.py QUESTION-SET VARIANTS: the
diagnosis of the a-e run, D1-D8, and why each is built as it is) adds three,
all read on the FRAMED STATE (skill_inject.framed_state: the session goal,
then the user's message or the assistant's latest step, the parts named) and
with each item shown under its craft's name:

  f  SPLIT: a NEXT noul and a DONE noul per item, need = P(next) x
     (1 - P(done))
  g  FIT: one noul per item, "the next code or command would be more correct
     with this fact in front of it"
  h  PICK THEN FIT: one choice per case over its facts with a "none" option,
     then g's noul on the top 3 the choice ranked; need = P(fit) x
     (1 - P(none))

They are NOT in the default `--variants` (the served five stay the default,
and --skip-done keeps resuming that run):

    python bench/skills/inject_decide.py --cases CASES --model bonsai-a4000 \
        --variants f,g,h --skip-done
    python bench/skills/inject_decide.py --cases CASES --model bonsai-a4000 \
        --variants f,g,h --estimate        # offline: GPU minutes, sends nothing

WHAT IS WRITTEN (never conversation text): the decisions log (decide_turn
DECISIONS, pointed at bench/skills/inject/results/decisions_<model>.jsonl
before import) and one row per case and variant in
bench/skills/inject/results/run_<model>.jsonl: the item keys, each item's
belief, expected level, confidence, argmax and decision id, the stage-3
noul and what it listed, and the milliseconds.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RESULTS = os.path.join(HERE, "inject", "results")
sys.path.insert(0, os.path.join(ROOT, "mcp"))

RUN_VERSION = "inject-decide/1"
SERVED_VARIANTS = ("a", "b", "c", "d", "e")   # the default run
NEW_VARIANTS = ("f", "g", "h")                # skill_inject.NEW_VARIANTS
VARIANTS = SERVED_VARIANTS + NEW_VARIANTS
LEVELS_B = [
    "It is about a library, API or task that the material does not "
    "involve.",
    "It is about a library the material involves, but not about the code, "
    "command or error the assistant is working on now.",
    "It is about the code, command or error the assistant is working on "
    "now.",
]
NOUL_C = ("This fact is about the code, command or error the assistant is "
          "working on now, and the material does not yet show it done the "
          "way the fact says.\n\nFACT: {fact}")


def running(upstream: str) -> list[str]:
    with urllib.request.urlopen(f"{upstream}/running", timeout=10) as r:
        d = json.loads(r.read().decode("utf-8"))
    rows = d.get("running") if isinstance(d, dict) else d
    return [str((x or {}).get("model")) for x in rows or []
            if isinstance(x, dict)]


def as_items(case_items: list[dict]) -> list[dict]:
    """A case's items as skill_inject's question builders read them: the
    craft's NAME (what the labellers saw beside each fact) and the fact."""
    return [{"key": it["key"], "sha": it.get("sha"), "name": it.get("skill"),
             "fact": it["fact"]} for it in case_items]


def questions(variant: str, items: list[dict]) -> list[dict]:
    import decider_bonsai as D
    import skill_inject as I
    if variant in NEW_VARIANTS:
        return I.planned_questions(variant, as_items(items))
    out = []
    for it in items:
        name = f"{I.QSET_ITEM}{'' if variant == 'a' else '_' + variant}:" \
               f"{it['key']}"
        if variant in ("a", "d"):
            out.append(D.q_score(name, I.ITEM_Q.format(fact=it["fact"]),
                                 I.ITEM_LEVELS))
        elif variant == "e":
            out.append(D.q_score(name, I.ITEM_Q.format(fact=it["fact"]),
                                 I.ITEM_LEVELS, label_kind="digits"))
        elif variant == "b":
            out.append(D.q_score(name, I.ITEM_Q.format(fact=it["fact"]),
                                 LEVELS_B))
        else:
            out.append(D.q_noul(name, NOUL_C.format(fact=it["fact"])))
    return out


def belief(variant: str, a: dict) -> float:
    if variant == "c":
        return float(a["noul"])
    p = a.get("probabilities") or {}
    top = str(len(p) - 1)
    return float(p.get(top, 0.0))


def passes(variant: str, a: dict) -> bool:
    if (a.get("diagnostics") or {}).get("tie"):
        return False
    if variant == "c":
        return float(a["noul"]) >= 0.5
    p = a.get("probabilities") or {}
    return bool(p) and max(p, key=p.get) == str(len(p) - 1)


def run_case(c: dict, variants, model: str, post=None, upstream=None
             ) -> list[dict]:
    """One case's rows: the served variants on the plain state (d on the goal
    state, as before), the new ones on the framed state -- one Turn (one
    state placement) per state. `post` / `upstream` are the decider's doors
    (None: the real ones; a test passes a fake)."""
    rows = []
    groups = [("base", [v for v in variants if v in ("a", "b", "c", "e")]),
              ("goal", [v for v in variants if v == "d"]),
              ("frame", [v for v in variants if v in NEW_VARIANTS])]
    for which, vs in groups:
        if vs:
            state = (goal_state(c) if which == "goal" else
                     frame_state(c) if which == "frame" else c["state"])
            rows += _run_state(c, vs, model, state, post, upstream)
    return rows


def frame_state(c: dict) -> str:
    """The new variants' STATE: skill_inject.framed_state over the case's
    own goal (`task`) and state, in the size the case's state was cut to."""
    import skill_inject as I
    return I.framed_state(c.get("task"), c["state"], dict(
        c.get("state_info") or {}, kind=c["kind"]))


def goal_state(c: dict) -> str:
    """Variant d's STATE: the session's goal named first, then the turn
    (docs/JJAVA.md 4.2 "Name the parts"); when the turn was cut to the
    lane's STATE_TOKENS, the goal takes its room from the turn's head so
    the state stays that size."""
    goal = "GOAL (the session's opening request): " + " ".join(
        str(c.get("task") or "").split())
    st = c["state"]
    if (c.get("state_info") or {}).get("cut"):
        st = st[len(goal) + 2:]
    return goal + "\n\nNOW:\n" + st


def _run_state(c: dict, variants, model: str, state: str, post=None,
               upstream=None) -> list[dict]:
    import decide_turn as T
    import decider_bonsai as D
    import skill_inject as I
    rows = []
    with T.Turn([], state=state, state_info=dict(
            c.get("state_info") or {}, kind=c["kind"]), on=True,
            key=f"inject:{c['case']}", request=c["case"], post=post,
            upstream=upstream) as t:
        for v in variants:
            t0 = time.time()
            extra = {}
            if v in NEW_VARIANTS:
                got, meta = I.item_beliefs(v, as_items(c["items"]), t)
                items = [{"key": it["key"], "sha": it["sha"],
                          "belief": r["belief"], "score": None,
                          "confidence": None, "argmax": None,
                          "pass": r["pass"], "tie": r["tie"],
                          "disagreement": r["disagreement"],
                          "decision_id": r["decision_id"],
                          "probs": r["probs"], "parts": r["parts"]}
                         for it, r in zip(c["items"], got)]
                if meta.get("pick"):
                    extra["pick"] = meta["pick"]
            else:
                ans = t.decide(questions(v, c["items"]))
                items = []
                for it, a in zip(c["items"], ans):
                    items.append({
                        "key": it["key"], "sha": it["sha"],
                        "belief": round(belief(v, a), 6),
                        "score": a.get("score"),
                        "confidence": a.get("confidence"),
                        "argmax": (a.get("diagnostics") or {}).get("argmax"),
                        "pass": passes(v, a),
                        "tie": bool((a.get("diagnostics") or {}).get("tie")),
                        "disagreement": (a.get("diagnostics") or {})
                        .get("disagreement"),
                        "decision_id": a.get("decision_id"),
                        "probs": (a.get("probabilities") if v != "c" else
                                  {"true": a["noul"]})})
            ranked = sorted(items, key=lambda r: (-r["belief"], r["key"]))
            short = [r for r in ranked if r["pass"]]
            how = "passed"
            if not short:
                short, how = ranked[:2], "forced"
            by = {it["key"]: it for it in c["items"]}
            listed_items = [{"key": r["key"], "form": "DO", "situation": "",
                             "text": by[r["key"]]["fact"]} for r in short]
            q, listed = I.inject_question(listed_items)
            q = D.q_noul(f"{I.QSET_INJECT}{'' if v == 'a' else '_' + v}",
                         q["text"])
            a3 = t.decide([q])[0]
            rows.append({"v": RUN_VERSION, "model": model, "variant": v,
                         "case": c["case"], "source": c["source"],
                         "group": c["group"], "kind": c["kind"],
                         "state_sha": c["state_sha"], "items": items,
                         "stage3": {"noul": a3["noul"], "how": how,
                                    "listed": [x["key"] for x in listed],
                                    "tie": bool((a3.get("diagnostics") or {})
                                                .get("tie")),
                                    "decision_id": a3.get("decision_id")},
                         "ms": round((time.time() - t0) * 1000, 1), **extra})
    return rows


# ------------------------------------------------------------- estimate ---
# THE GPU TIME OF A RUN, FROM THE LAST ONE (offline; docs/JJAVA.md 9). Two
# straight lines fitted to the decisions the 2026-10-06 run logged (variants
# a-e, a decision = one question's two reads): the tokens the server
# PROCESSED for a question against the characters of its two printed orders
# (the state is cached, so a question costs its own suffix), and its
# milliseconds against those tokens. A state's placement (the first question
# of a Turn) is fitted apart, from the first question of each served Turn.
# Stage 3 costs the run's own mean. The fit is checked against the run it was
# fitted on (`check`: predicted against the recorded minutes of a-e).
def _linfit(xs: list, ys: list) -> dict:
    n = len(xs)
    if n < 3:
        return {"intercept": 0.0, "slope": 0.0, "r2": None, "n": n}
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx else 0.0
    icpt = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (icpt + slope * x)) ** 2 for x, y in zip(xs, ys))
    return {"intercept": icpt, "slope": slope,
            "r2": round(1 - ss_res / ss_tot, 4) if ss_tot else None, "n": n}


def _chars(q: dict) -> int:
    import decider_bonsai as D
    return sum(len(D.question_text(c)) for c in D.rendered_orders(q))


def _decisions(path: str) -> dict:
    out = {}
    try:
        with open(path, encoding="utf-8") as f:
            for ln in f:
                try:
                    r = json.loads(ln)
                except ValueError:
                    continue
                if r.get("row") == "decision" and r.get("processed_tokens") \
                        is not None:
                    out[r["id"]] = (float(r["processed_tokens"]),
                                    float(r.get("ms") or 0.0))
    except OSError:
        pass
    return out


def calibrate(all_cases: list[dict], model: str, results: str | None = None,
              only: set | None = None) -> dict | None:
    """The fitted rate. `only`: fit on the rows of these cases alone (the
    hold-out check fits on one half and predicts the other)."""
    results = results or RESULTS
    run = os.path.join(results, f"run_{model}.jsonl")
    dec = _decisions(os.path.join(results, f"decisions_{model}.jsonl"))
    if not dec or not os.path.exists(run):
        return None
    by = {c["case"]: c for c in all_cases}
    tok_pts, ms_pts, firsts, s3, measured = [], [], [], [], 0.0
    seen: set = set()
    case_ms: dict = {}
    with open(run, encoding="utf-8") as f:
        for ln in f:
            r = json.loads(ln)
            c = by.get(r["case"])
            if c is None or r["variant"] not in SERVED_VARIANTS or (
                    only is not None and r["case"] not in only):
                continue
            measured += float(r.get("ms") or 0.0)
            case_ms[r["case"]] = case_ms.get(r["case"], 0.0) + float(
                r.get("ms") or 0.0)
            seen.add(r["case"])
            qs = questions(r["variant"], c["items"])
            for i, (q, it) in enumerate(zip(qs, r["items"])):
                d = dec.get(it.get("decision_id"))
                if d is None:
                    continue
                if i == 0 and r["variant"] in ("a", "d"):
                    firsts.append((c["state_info"].get("tokens") or 0,
                                   d[1], _chars(q), d[0]))
                else:
                    tok_pts.append((_chars(q), d[0]))
                    ms_pts.append((d[0], d[1]))
            d3 = dec.get((r.get("stage3") or {}).get("decision_id"))
            if d3 is not None:
                s3.append(d3[1])
    if not tok_pts:
        return None
    tk = _linfit([x for x, _ in tok_pts], [y for _, y in tok_pts])
    mf = _linfit([x for x, _ in ms_pts], [y for _, y in ms_pts])
    # a state's placement: what the first question cost beyond a question of
    # its own size, against the state's tokens
    extra = [(st, ms - (mf["intercept"] + mf["slope"] * (
        tk["intercept"] + tk["slope"] * ch))) for st, ms, ch, _t in firsts]
    pl = _linfit([x for x, _ in extra], [y for _, y in extra])
    return {"tokens_vs_chars": tk, "ms_vs_tokens": mf, "placement": pl,
            "stage3_ms": round(sum(s3) / len(s3), 1) if s3 else None,
            "recorded_minutes_a_to_e": round(measured / 60000, 1),
            "cases_run": sorted(seen), "case_ms": case_ms}


def _question_ms(cal: dict, q: dict) -> float:
    tok = cal["tokens_vs_chars"]["intercept"] + \
        cal["tokens_vs_chars"]["slope"] * _chars(q)
    return cal["ms_vs_tokens"]["intercept"] + cal["ms_vs_tokens"]["slope"] \
        * max(tok, 0.0)


def estimate(all_cases: list[dict], pending: list[dict], variants: list[str],
             model: str, results: str | None = None) -> dict:
    cal = calibrate(all_cases, model, results)
    if cal is None:
        return {"error": f"no run_{model}.jsonl / decisions_{model}.jsonl to "
                "fit the rate from"}
    place = lambda c: max(0.0, cal["placement"]["intercept"] + cal[          # noqa: E731
        "placement"]["slope"] * (c["state_info"].get("tokens") or 0))
    s3 = cal["stage3_ms"] or 0.0
    ran = set(cal.pop("cases_run"))
    out = {"model": model, "cases": len(pending), "calibration": cal,
           "variants": {}}
    groups = {"base": [v for v in variants if v in ("a", "b", "c", "e")],
              "goal": [v for v in variants if v == "d"],
              "frame": [v for v in variants if v in NEW_VARIANTS]}
    total = 0.0
    for g, vs in groups.items():
        if not vs:
            continue
        gp = sum(place(c) for c in pending) / 60000
        out.setdefault("state_placement_minutes", {})[g] = round(gp, 1)
        total += gp
        for v in vs:
            nq, qm = 0, 0.0
            for c in pending:
                for q in questions(v, c["items"]):
                    nq += 1
                    qm += _question_ms(cal, q)
            m = (qm + s3 * len(pending)) / 60000
            out["variants"][v] = {
                "questions": nq + len(pending), "stage2_questions": nq,
                "minutes": round(m, 1),
                "rows": len(pending),
                "rows_per_minute": round(len(pending) / m, 2) if m else None}
            total += m
    out["total_minutes"] = round(total, 1)
    # THE CHECK: fitted on one half of the run's cases, it predicts the
    # RECORDED minutes of the other half (halves by the case id's parity),
    # both ways round -- the minutes the run actually took, never the fit's own
    def predicted(c_, cal_) -> float:
        place_ = max(0.0, cal_["placement"]["intercept"] + cal_["placement"][
            "slope"] * (c_["state_info"].get("tokens") or 0))
        return (sum(_question_ms(cal_, q) for v in SERVED_VARIANTS
                    for q in questions(v, c_["items"]))
                + (cal_["stage3_ms"] or 0.0) * len(SERVED_VARIANTS)
                + 2 * place_)
    chk = []
    for half in (0, 1):
        fit_on = {c for c in ran if int(c, 16) % 2 != half}
        test_on = {c for c in ran if int(c, 16) % 2 == half}
        ch = calibrate(all_cases, model, results, only=fit_on)
        if ch is None:
            continue
        by = {c["case"]: c for c in all_cases}
        pred = sum(predicted(by[c], ch) for c in test_on) / 60000
        rec = sum(cal["case_ms"][c] for c in test_on) / 60000
        chk.append({"fit_cases": len(fit_on), "test_cases": len(test_on),
                    "predicted_minutes": round(pred, 1),
                    "recorded_minutes": round(rec, 1),
                    "error": round(pred / rec - 1, 3) if rec else None})
    out["check_held_out_halves"] = chk
    cal.pop("case_ms", None)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cases", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--variants", default=",".join(SERVED_VARIANTS),
                    help="comma list of " + ",".join(VARIANTS) + "; the new "
                    "variants f,g,h are not in the default")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--only-labelled", action="store_true",
                    help="only cases bench/skills/inject/labels.jsonl labels")
    ap.add_argument("--skip-done", action="store_true",
                    help="skip cases that already have a row in "
                    "run_<model>.jsonl for every requested variant (resume "
                    "a run that was stopped)")
    ap.add_argument("--dry-run", action="store_true",
                    help="build every question, send nothing")
    ap.add_argument("--estimate", action="store_true",
                    help="OFFLINE: the GPU minutes the requested variants "
                    "would take, from the rate of the run already in "
                    "run_<model>.jsonl and its decisions log; sends nothing")
    a = ap.parse_args(argv)
    os.makedirs(RESULTS, exist_ok=True)
    os.environ["YAMADORI_DECIDER_DECISIONS"] = os.path.join(
        RESULTS, f"decisions_{a.model}.jsonl")
    os.environ["YAMADORI_DECIDER_LOG"] = os.path.join(
        RESULTS, f"disagreements_{a.model}.jsonl")
    variants = [v for v in a.variants.split(",") if v in VARIANTS]
    with open(a.cases, encoding="utf-8") as f:
        cases = [json.loads(ln) for ln in f if ln.strip()]
    if a.only_labelled:
        sys.path.insert(0, HERE)
        import inject_labels
        # the cases inject_tune scores: the rubric truth is pass A plus the
        # blind pass B (labels("A") alone left out the 8 cases only B read,
        # 34 items, in the 2026-10-06 run)
        lab = {k[0] for k in inject_labels.truth("rubric")}
        cases = [c for c in cases if c["case"] in lab]
    if a.skip_done:
        have: dict = {}
        try:
            with open(os.path.join(RESULTS, f"run_{a.model}.jsonl"),
                      encoding="utf-8") as f:
                for ln in f:
                    r = json.loads(ln)
                    have.setdefault(r["case"], set()).add(r["variant"])
        except OSError:
            pass
        cases = [c for c in cases if not set(variants) <= have.get(
            c["case"], set())]
    cases = cases[:a.limit] if a.limit else cases
    if a.estimate:
        with open(a.cases, encoding="utf-8") as f:
            everything = [json.loads(ln) for ln in f if ln.strip()]
        print(json.dumps(estimate(everything, cases, variants, a.model),
                         indent=1))
        return 0
    if a.dry_run:
        n = sum(len(questions(v, c["items"])) + 1 for c in cases
                for v in variants)
        print(json.dumps({"cases": len(cases), "variants": variants,
                          "questions": n, "reads": 2 * n}))
        return 0
    import model as M
    up = M.UPSTREAM
    have = running(up)
    if a.model not in have:
        print(f"REFUSED: {a.model} is not loaded (GET /running: {have}); "
              "this bench never loads a model -- ask the coordinator for "
              "the window with it loaded", flush=True)
        return 2
    import cancel
    import max_mode
    out = os.path.join(RESULTS, f"run_{a.model}.jsonl")
    t0 = time.time()
    with cancel.bound(cancel.Token()):
        max_mode.set_current(a.model)
        with open(out, "a", encoding="utf-8") as f:
            for k, c in enumerate(cases):
                try:
                    for r in run_case(c, variants, a.model):
                        f.write(json.dumps(r) + "\n")
                    f.flush()
                except Exception as e:                           # noqa: BLE001
                    print(f"  case {c['case']}: {type(e).__name__}: {e}"
                          [:300], flush=True)
                if k % 10 == 9:
                    print(f"  {k + 1}/{len(cases)} cases, "
                          f"{round(time.time() - t0)} s", flush=True)
    print(json.dumps({"cases": len(cases), "variants": variants, "out": out,
                      "seconds": round(time.time() - t0, 1)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
