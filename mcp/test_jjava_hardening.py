#!/usr/bin/env python
"""jjava's SGLang hardening, the bench half (docs/JJAVA.md 8). No GPU, no
network: synthetic decisions and fake reads.

    python mcp/test_jjava_hardening.py      -> "N/M checks passed"

GATES:
  [fit]          bench/decider/fit_temperature.py: an over-confident reader
                 (raw = softmax(3 z), truth drawn from softmax(z)) is fitted
                 to T ~ 3 and HOLDS OUT BY RUN; a calibrated reader is fitted
                 near 1 and does NOT (nothing written); --write puts only the
                 held-out sets into the model's record, which the runtime
                 then reads (decider_bonsai.temperature_of).
  [retemper]     bench/skills/inject_tune.py re-scores a run's beliefs from
                 the decisions log's raw orders at the fitted T, and leaves
                 them alone at T = 1.
  [determinism]  bench/decider/determinism.py's spread, cross-regime and
                 plan arithmetic on fake reads; every question it asks passes
                 the think-block guard.
  [variant e]    bench/skills/inject_decide.py's digit variant: the same
                 question and levels as `a`, printed with digits.
(The runtime half -- the guard through the served template, the case-
variant diagnostics, the digit labels, the temperature in read() -- is in
mcp/test_decider_bonsai.py.)
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import offline_stores  # noqa: E402

_TMP = offline_stores.isolate("yamadori_test_jjava_hardening_")
PROFILES = os.path.join(_TMP, "decider_models")
os.environ["YAMADORI_DECIDER_MODELS_DIR"] = PROFILES
sys.path.insert(0, os.path.join(ROOT, "bench", "decider"))
sys.path.insert(0, os.path.join(ROOT, "bench", "skills"))

import decider_bonsai as D  # noqa: E402

CHECKS: list[tuple[str, bool]] = []


def check(name: str, ok: bool, detail=None) -> None:
    CHECKS.append((name, bool(ok)))
    if not ok:
        print("FAIL", name, "" if detail is None else str(detail)[:400])


def _softmax(z):
    m = max(z)
    e = [math.exp(x - m) for x in z]
    s = sum(e)
    return [x / s for x in e]


def synthetic(rng, sharp: float, n_runs=4, per=120, qset="skill_item"):
    rows = []
    keys = ["0", "1", "2", "3"]
    for g in range(n_runs):
        for i in range(per):
            z = [rng.gauss(0, 1.2) for _ in keys]
            true_p = _softmax(z)
            truth = rng.choices(keys, weights=true_p)[0]
            raw = dict(zip(keys, _softmax([sharp * x for x in z])))
            rows.append({"id": f"{qset}-{g}-{i}", "set": qset,
                         "run": f"run{g}", "orders_raw": [raw, dict(raw)],
                         "excluded": [], "truth": truth, "type": "score"})
    return rows


def fit_checks() -> None:
    import fit_temperature as F
    rng = random.Random(7)
    over = synthetic(rng, 3.0)
    cal = synthetic(rng, 1.0, qset="phase")
    rep = F.report(over + cal)
    o, c = rep["skill_item"], rep["phase"]
    check("[fit] an over-confident reader (3x the logits) is fitted near "
          "T = 3", 2.3 < o["fit_all"]["t"] < 3.8 and not o["fit_all"]["edge"],
          o["fit_all"])
    check("[fit] ... and it holds out by run (every fold fitted on the "
          "others, the NLL difference's 95% interval below zero)",
          o["heldout"]["heldout_ok"] and o["heldout"]["ci95"][1] < 0
          and len(o["heldout"]["runs"]) == 4, o["heldout"])
    check("[fit] tempering lowers the NLL and the ECE there",
          o["at_fit"]["nll"] < o["at_t1"]["nll"]
          and o["at_fit"]["ece"] < o["at_t1"]["ece"], [o["at_t1"],
                                                       o["at_fit"]])
    check("[fit] a calibrated reader is fitted near 1 and does NOT hold out "
          "(no gain to ship)", 0.7 < c["fit_all"]["t"] < 1.4
          and not c["heldout"]["heldout_ok"], [c["fit_all"], c["heldout"]])
    one = F.report(synthetic(random.Random(3), 3.0, n_runs=1))
    check("[fit] one run cannot be held out: never ok",
          one["skill_item"]["heldout"]["heldout_ok"] is False
          and "two runs" in one["skill_item"]["heldout"]["why"])
    w = F.write("bonsai", rep, "synthetic")
    tf = D.temperature_of("skill_item:x#0", "bonsai")
    check("[fit] --write puts ONLY the held-out set into the record, and the "
          "runtime reads it back (phase stays T = 1)",
          w["written"] and w["sets"] == ["skill_item"] and tf["fitted"]
          and abs(tf["value"] - o["fit_all"]["t"]) < 1e-6
          and D.temperature_of("phase", "bonsai")["value"] == 1.0
          and "temperature" not in D.profile("bonsai")["rejected"],
          [w, tf, D.profile("bonsai")["rejected"]])
    check("[fit] readout_of at the fitted T is what read() would answer "
          "(the averaged, tempered orders)",
          abs(F.dist(over[0], 2.0)["0"]
              - D.readout_of(over[0]["orders_raw"], 2.0)["0"]) < 1e-12)
    nothing = F.write("bonsai", {"phase": c}, "synthetic")
    check("[fit] with no set held out nothing is written, and it says so",
          nothing["written"] is False)
    # the join: an inject run file + decisions log + truth
    import inject_labels as IL
    res = os.path.join(_TMP, "inject_results")
    os.makedirs(res, exist_ok=True)
    raw = {"0": 0.1, "1": 0.2, "2": 0.3, "3": 0.4}
    dec = [{"row": "decision", "id": "d1", "model": "bonsai",
            "question": {"name": "skill_item:k#0", "type": "score",
                         "keys": list(raw)},
            "orders": [{"raw": raw, "p": raw}, {"raw": raw, "p": raw}]},
           {"row": "decision", "id": "d2", "model": "bonsai",
            "question": {"name": "skill_inject", "type": "noul",
                         "keys": ["true", "false"]},
            "orders": [{"raw": {"true": .7, "false": .3}},
                       {"raw": {"true": .6, "false": .4}}]},
           {"row": "decision", "id": "d3", "model": "bonsai",
            "question": {"name": "skill_item_c:k#0", "type": "noul",
                         "keys": ["true", "false"]},
            "orders": [{"p": {"true": .5, "false": .5}}]}]
    run = [{"variant": "a", "source": "pagoda-h4", "group": "hermes/step",
            "case": "c1", "items": [{"key": "k#0", "decision_id": "d1"}],
            "stage3": {"decision_id": "d2", "listed": ["k#0"]}},
           {"variant": "c", "source": "pagoda-h4", "group": "hermes/step",
            "case": "c1", "items": [{"key": "k#0", "decision_id": "d3"}],
            "stage3": {}}]
    with open(os.path.join(res, "run_bonsai.jsonl"), "w") as f:
        f.write("\n".join(json.dumps(r) for r in run))
    with open(os.path.join(res, "decisions_bonsai.jsonl"), "w") as f:
        f.write("\n".join(json.dumps(r) for r in dec))
    saved = (F.ROOT, IL.truth)
    root = os.path.join(_TMP, "root")
    os.makedirs(os.path.join(root, "bench", "skills", "inject"),
                exist_ok=True)
    os.replace(res, os.path.join(root, "bench", "skills", "inject",
                                 "results"))
    F.ROOT = root
    IL.truth = lambda kind="rubric": {("c1", "k#0"): 3}
    try:
        rows = F.inject_rows("bonsai")
    finally:
        F.ROOT, IL.truth = saved
    check("[fit] the inject join: a score item's truth is its level, stage "
          "3's is 'any listed item NEEDED', a decision without raw orders is "
          "skipped; the run is the case's source session",
          sorted((r["set"], r["truth"]) for r in rows)
          == [("skill_inject", "true"), ("skill_item", "3")]
          and all(r["run"] == "pagoda-h4" for r in rows), rows)


def retemper_checks() -> None:
    import inject_tune as IT
    res = os.path.join(_TMP, "retemper")
    os.makedirs(res, exist_ok=True)
    raw = {"0": 0.1, "1": 0.2, "2": 0.3, "3": 0.4}
    with open(os.path.join(res, "decisions_bonsai.jsonl"), "w") as f:
        f.write(json.dumps({"row": "decision", "id": "d1", "model": "bonsai",
                            "question": {"name": "skill_item:k#0",
                                         "type": "score",
                                         "keys": list(raw)},
                            "orders": [{"raw": raw}, {"raw": raw}]}) + "\n")
    runs = [{"items": [{"key": "k#0", "decision_id": "d1", "belief": 0.4,
                        "pass": True, "tie": False}], "stage3": {}}]
    saved = IT.RESULTS
    IT.RESULTS = res
    try:
        t = D.temperature_of("skill_item:k#0", "bonsai")["value"]
        out = IT.retemper(json.loads(json.dumps(runs)), "bonsai")
        D.TEMPERATURE_ON = False
        same = IT.retemper(json.loads(json.dumps(runs)), "bonsai")
    finally:
        D.TEMPERATURE_ON = True
        IT.RESULTS = saved
    want = D.readout_of([raw, raw], t)["3"]
    check("[retemper] a fitted T re-scores the belief from the logged raw "
          "orders; at T = 1 the row is untouched",
          t != 1.0 and abs(out[0]["items"][0]["belief"] - round(want, 6))
          < 1e-9 and out[0]["items"][0]["retempered"]
          and same[0]["items"][0] == runs[0]["items"][0],
          [t, out[0]["items"][0], same[0]["items"][0]])


def determinism_checks() -> None:
    import determinism as DM

    def rd(p, lm, ms=100.0):
        return {"p": p, "argmax": max(p, key=p.get),
                "orders": [{"raw": p, "label_mass": lm,
                            "variant_mass": 0.01, "word_mass": None,
                            "processed": 10}] * 2,
                "ms": ms, "processed": 20}
    same = {("s1", "q"): [rd({"true": .8, "false": .2}, .9)] * 3}
    moved = {("s1", "q"): [rd({"true": .8, "false": .2}, .9),
                           rd({"true": .74, "false": .26}, .76),
                           rd({"true": .45, "false": .55}, .9, 300.0)]}
    a, b = DM.summarise(same), DM.summarise(moved)
    check("[determinism] identical repeats: spread 0, every group identical",
          a["prob_per_order"]["max"] == 0.0
          and a["prob_per_order"]["identical_groups"]
          == a["prob_per_order"]["groups"] and a["argmax_flip_groups"] == 0,
          a)
    check("[determinism] moved repeats: max |p_i - p_1| (0.35), label mass "
          "spread (0.14), an argmax flip, seconds per question",
          abs(b["prob_answer"]["max"] - 0.35) < 1e-9
          and abs(b["label_mass"]["max"] - 0.14) < 1e-9
          and b["argmax_flip_groups"] == 1
          and b["seconds_per_question"]["max"] == 0.3, b)
    x = DM.across({"cold": same, "cached": moved})
    check("[determinism] across regimes: each regime's repeat mean against "
          "cold's", abs(x["cached_vs_cold"]["max"]
                        - abs(0.8 - (0.8 + 0.74 + 0.45) / 3)) < 1e-6
          and x["cached_vs_cold"]["argmax_agree"] == 1, x)
    # [choose] the cheapest mode that repeats exactly, and its band
    fixed = {("s1", "q"): [rd({"true": .8, "false": .2}, .9, 900.0)] * 3}
    shifted = {("s1", "q"): [rd({"true": .81, "false": .19}, .9, 950.0)] * 3}
    wobble = {("s1", "q"): [rd({"true": .8, "false": .2}, .9, 150.0),
                            rd({"true": .79, "false": .21}, .88, 150.0),
                            rd({"true": .8, "false": .2}, .9, 150.0)]}
    reads = {"cold": fixed, "cold_beside": shifted, "cached": wobble,
             "cached_beside": wobble}
    outs = {k: DM.summarise(v) for k, v in reads.items()}
    ch = DM.choose(outs, reads, 3)
    check("[choose] cached is cheaper but does not repeat exactly; cold "
          "repeats exactly in both environments -> cold, its band the "
          "between-environment difference (0.01)",
          ch["chosen"] == "cold" and ch["modes"]["cold"]["exact"]
          and not ch["modes"]["cached"]["exact"]
          and abs(ch["modes"]["cold"]["tie_band"] - 0.01) < 1e-9
          and ch["modes"]["cold"]["n"] == 6, ch)
    reads2 = dict(reads, cached=fixed, cached_beside=fixed)
    reads2["cached"] = {("s1", "q"): [rd({"true": .8, "false": .2}, .9,
                                         100.0)] * 3}
    outs2 = {k: DM.summarise(v) for k, v in reads2.items()}
    check("[choose] both exact -> the cheaper (cached)",
          DM.choose(outs2, reads2, 3)["chosen"] == "cached")
    rec = DM.record("bonsai", ch, {"states": [{"id": "s1"}]})
    rr, tb = D.read_regime_of("bonsai"), D.tie_band_of("bonsai")
    check("[choose] record(): read_regime cold and the tie band in that "
          "mode go into the model's record, and the runtime reads both",
          rec["written"] == ["tie_band", "read_regime"] and rr["mode"] == "cold"
          and rr["measured"] and D.read_cache("bonsai") is False
          and tb["measured"] and abs(tb["value"] - 0.01) < 1e-9, [rec, rr, tb])
    none = {"cold": wobble, "cold_beside": wobble, "cached": wobble,
            "cached_beside": wobble}
    ch3 = DM.choose({k: DM.summarise(v) for k, v in none.items()}, none, 3)
    check("[choose] no mode exact: nothing chosen, the band derived for "
          "cached (the path that stays)",
          ch3["chosen"] is None and ch3["band_mode"] == "cached"
          and abs(ch3["modes"]["cached"]["tie_band"] - 0.01) < 1e-9, ch3)
    pl = DM.plan(10, 5, DM.REGIMES)
    check("[determinism] the plan: 10 states x 3 questions x 5 repeats x 2 "
          "orders a regime, plus the cached regimes' placement reads",
          pl["reads_per_regime"] == 300 and pl["reads_total"] == 1240, pl)
    qs = DM.questions(DM.FIXED_FACT)
    ok = True
    for q in qs:
        for c in D.rendered_orders(q):
            ok = ok and D.guard_body(D.body("s", c, 2, 20, "bonsai"))["ok"]
    check("[determinism] one question of each type, every order passes the "
          "think-block guard",
          sorted(q["type"] for q in qs) == ["choice", "noul", "score"]
          and ok)


def variant_e_checks() -> None:
    import inject_decide as ID
    items = [{"key": "k#0", "fact": "DO: use x"}]
    qa, qe = ID.questions("a", items)[0], ID.questions("e", items)[0]
    check("[variant e] the digit variant is variant a's question and levels "
          "printed with digits, in its own question set",
          "e" in ID.VARIANTS and qe["label_kind"] == "digits"
          and qe["text"] == qa["text"] and qe["options"] == qa["options"]
          and qe["name"].startswith("skill_item_e:")
          and D.rendered_orders(qe)[0]["labels"] == ["0", "1", "2", "3"])


def main() -> int:
    fit_checks()
    retemper_checks()
    determinism_checks()
    variant_e_checks()
    ok = sum(1 for _, o in CHECKS if o)
    print(f"{ok}/{len(CHECKS)} checks passed")
    return 0 if ok == len(CHECKS) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:                                            # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
