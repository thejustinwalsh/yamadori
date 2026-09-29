#!/usr/bin/env python
"""The tier tuner (bench/decider/tune.py) and the question sets' tiers it
proposes for (mcp/decide_turn.py THRESHOLDS, tier()). No GPU, no network:
the decisions and labels are temp files written here.

    python mcp/test_decider_tune.py      -> "N/M checks passed"

GATES: accuracy is binned by Jev's confidence (Choice / Score) or by the
noul (10 equal-width bins); the proposal needs the OWNER's targets (none
without them); each proposed threshold is the lowest (the highest, for a
noul's no side) whose kept decisions meet the target at the Wilson lower
bound (package_eval.ALPHA); a target nothing reaches stays untuned; the
coverage is computed through decide_turn.tier itself; unsure labels are
left out and the latest label wins; thresholds are per model; nothing is
written (THRESHOLDS is untouched, no file changes).
"""
from __future__ import annotations

import json
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "bench", "decider"))
import offline_stores  # noqa: E402

_TMP = offline_stores.isolate("yamadori_test_decider_tune_")

import decide_turn as T  # noqa: E402
import label as L  # noqa: E402
import tune  # noqa: E402

CHECKS: list[tuple[str, bool]] = []


def check(name: str, ok: bool, detail=None) -> None:
    CHECKS.append((name, bool(ok)))
    if not ok:
        print("FAIL", name, "" if detail is None else str(detail)[:400])


def row(i, name, qtype, p, model="bonsai", conf=None, noul=None):
    r = {"row": "decision", "v": 2, "id": f"d{i}", "model": model,
         "question": {"type": qtype, "name": name, "keys": sorted(p)},
         "p": p, "argmax": max(p, key=p.get), "pick": max(p, key=p.get),
         "tie": False, "tier": "untuned", "state": {"sha": f"s{i}"}}
    if conf is not None:
        r["confidence"] = conf
    if noul is not None:
        r["noul"] = noul
    return r


def main() -> int:
    check("every store is a temp file", all(
        os.path.abspath(p).startswith(os.path.abspath(_TMP))
        for p in (L.DECISIONS, L.LEGACY, L.LABELS)))
    decisions, labels = [], []
    # PHASE (choice): 40 decisions. Confidence 0.9+ -> all right; 0.5-0.9
    # -> 3 of 4 right; below 0.5 -> 1 of 2 right.
    i = 0
    for k in range(40):
        i += 1
        if k < 20:
            conf, right = 0.95, True
        elif k < 32:
            conf, right = 0.7, (k % 4 != 0)
        else:
            conf, right = 0.3, (k % 2 == 0)
        pmax = (conf * 3 + 1) / 4          # 4 options: invert Jev's form
        rest = (1 - pmax) / 3
        p = {"implement": pmax, "plan": rest, "debug": rest, "verify": rest}
        decisions.append(row(i, "phase", "choice", p, conf=round(conf, 6)))
        labels.append({"row": "label", "decision_id": f"d{i}",
                       "question": "phase", "state_sha": f"s{i}",
                       "truth": "implement" if right else "plan"})
    # BUILD INTENT (noul): high p true, low p false, the middle mixed.
    for k in range(30):
        i += 1
        pt = [0.97, 0.9, 0.55, 0.45, 0.1, 0.03][k % 6]
        truth = {0.97: "true", 0.9: "true", 0.55: "false", 0.45: "true",
                 0.1: "false", 0.03: "false"}[pt]
        decisions.append(row(i, "build_intent", "noul",
                             {"true": pt, "false": 1 - pt}, noul=pt))
        labels.append({"row": "label", "decision_id": f"d{i}",
                       "question": "build_intent", "state_sha": f"s{i}",
                       "truth": truth})
    # an unsure label, a relabel (latest wins), another model's decision
    labels.append({"row": "label", "decision_id": "d1", "truth": None,
                   "unsure": True})
    labels.append({"row": "label", "decision_id": "d33",
                   "question": "phase", "truth": "implement"})
    decisions.append(row(999, "phase", "choice",
                         {"implement": 0.9, "plan": 0.1}, model="other",
                         conf=0.8))
    labels.append({"row": "label", "decision_id": "d999", "question":
                   "phase", "truth": "implement"})
    with open(L.DECISIONS, "w", encoding="utf-8") as f:
        for r in decisions:
            f.write(json.dumps(r) + "\n")
    with open(L.LABELS, "w", encoding="utf-8") as f:
        for r in labels:
            f.write(json.dumps(r) + "\n")
    before = {p: os.path.getmtime(p) for p in (L.DECISIONS, L.LABELS)}

    ds, _ = L.load_decisions()
    by_id, by_state = tune.labels_by()
    rows = tune.labelled(ds, by_id, by_state)
    ph = tune.select(rows, "phase", "bonsai")
    check("labels join by decision id; unsure left out; the latest label "
          "wins", len(ph) == 40 and next(r for r in ph if r["id"] == "d33")[
              "truth"] == "implement", len(ph))
    check("thresholds are per model: another model's rows are apart",
          len(tune.select(rows, "phase", "other")) == 1)

    rep = tune.report(ph, "phase", "bonsai", None, None)
    b = {x["range"]: x for x in rep["bins"]}
    check("accuracy by Jev's confidence in 10 equal-width bins",
          rep["certainty"] == "confidence" and b["[0.9, 1.0]"]["n"] == 20
          and b["[0.9, 1.0]"]["accuracy"] == 1.0
          and b["[0.7, 0.8)"]["n"] == 12 and b["[0.3, 0.4)"]["n"] == 8,
          rep["bins"])
    check("no targets, no proposal (the owner's numbers, never defaults)",
          isinstance(rep["proposal"], str) and "--high-accuracy"
          in rep["proposal"])
    rep = tune.report(ph, "phase", "bonsai", 0.8, 0.7)
    p = rep["proposal"]
    z = rep["significance"]["z"]
    exp_high = tune.wilson_lower(20, 20, z)
    check("the significance is the repo's (package_eval.ALPHA, two-sided "
          "Wilson)", rep["significance"]["alpha"] == 0.05
          and abs(z - 1.96) < 0.01)
    check("high: the lowest confidence whose kept decisions meet the target "
          "at the Wilson lower bound (20/20 at 0.95 -> lower "
          f"{exp_high:.3f})", p["row"].get("high") == 0.95
          and p["evidence"]["high"]["n"] == 20
          and abs(p["evidence"]["high"]["lower"] - round(exp_high, 4))
          < 1e-9, p)
    lb07 = tune.wilson_lower(29, 32, z)
    check("medium: the same at the medium target (29/32 at >= 0.7, lower "
          f"{lb07:.3f})", p["row"].get("medium") == 0.7
          and p["evidence"]["medium"]["correct"] == 29, p["evidence"])
    check("the coverage is counted through decide_turn.tier (the code that "
          "reads the row)", p["coverage"]["high"]["n"] == 20
          and p["coverage"]["medium"]["n"] == 12
          and p["coverage"]["low"]["n"] == 8, p["coverage"])
    check("the paste names the model and question set, and waits for the "
          "owner", p["paste"].startswith("# decide_turn.THRESHOLDS['bonsai']"
                                         "['phase']")
          and "ACCEPTED BY" in p["paste"] and T.THRESHOLDS == {})
    rep = tune.report(ph, "phase", "bonsai", 0.999, 0.99)
    check("a target nothing reaches stays untuned (reported, no row)",
          rep["proposal"]["row"] == {} and rep["proposal"]["paste"] is None
          and set(rep["proposal"]["unreached"]) == {"high", "medium"},
          rep["proposal"])

    bi = tune.select(rows, "build_intent", "bonsai")
    rep = tune.report(bi, "build_intent", "bonsai", 0.7, 0.5)
    p = rep["proposal"]
    check("a noul bins its noul (no confidence), with each bin's truth rate",
          rep["certainty"] == "noul"
          and next(x for x in rep["bins"] if x["range"] == "[0.9, 1.0]")[
              "true_rate"] == 1.0, rep["bins"])
    check("a noul's sides are tuned apart: act_yes the lowest p whose "
          "yeses are true at the target, act_no the highest p whose noes "
          "are false", p["row"].get("act_yes") == 0.9
          and p["row"].get("act_no") == 0.1, p)
    check("each side stays on its own side of 0.5: the mixed middle (0.55 "
          "said yes and was false half the time) reaches no tier",
          all(p["row"][k] >= 0.5 for k in ("act_yes", "caution_yes"))
          and all(p["row"][k] < 0.5 for k in ("act_no", "caution_no"))
          and p["coverage"]["low"]["n"] == 10, p)
    T.THRESHOLDS = {"bonsai": {"build_intent": p["row"]}}
    try:
        t_hi = T.tier({"type": "noul", "name": "build_intent", "noul": 0.95},
                      "bonsai")
        t_lo = T.tier({"type": "noul", "name": "build_intent", "noul": 0.5},
                      "bonsai")
        t_other = T.tier({"type": "noul", "name": "build_intent",
                          "noul": 0.95}, "other")
    finally:
        T.THRESHOLDS = {}
    check("decide_turn.tier reads an accepted row: 0.95 high, 0.5 not "
          "high; another model untuned", t_hi["tier"] == "high"
          and t_lo["tier"] != "high" and t_other["tier"] == "untuned",
          (t_hi, t_lo, t_other))
    rc = tune.main(["--stats"])
    rc2 = tune.main(["--question", "phase", "--high-accuracy", "0.8",
                     "--medium-accuracy", "0.7", "--model", "bonsai"])
    rc3 = tune.main(["--question", "phase"])
    check("the CLI runs; two models labelled for one question set needs "
          "--model (exit 2)", rc == 0 and rc2 == 0 and rc3 == 2)
    check("nothing is written: the decisions and labels are untouched",
          all(os.path.getmtime(q) == t for q, t in before.items()))
    ok = sum(1 for _, o in CHECKS if o)
    print(f"{ok}/{len(CHECKS)} checks passed")
    return 0 if ok == len(CHECKS) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:                                            # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
