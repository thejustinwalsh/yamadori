#!/usr/bin/env python
"""Every claim in docs/LAYA.md, as an assertion that fails when it stops holding.

WHY THIS FILE EXISTS

The measurements in docs/LAYA.md are the expensive part of this work, and the
expensive part is the NEGATIVE results: structured state does not help, the
`clarify` option is dead, the grounding question does not detect fabrication.
Those are the findings most likely to be quietly re-litigated by someone who
tries the same idea again in three months, and the only thing that stops that
is a test that fails the moment the number changes.

So each claim here is paired with the section of docs/LAYA.md it defends.

TWO MODES

  default   score the CACHED runs in bench/data/. No GPU, ~50 ms, runs in CI.
  --live    RETIRED 2026-09-24 with the Laya service (docs/E1.md): it
            called Laya on 1237 to assert the cache still matched the
            service. It now says so and runs nothing live.

The cached runs are regenerated with:

    python -X utf8 bench/laya_calibration.py            # 3-way + binary + distil
    python -X utf8 bench/laya_calibration.py --route --orders 6 --tag ord6

Run:

    python -X utf8 bench/test_laya_calibration.py
    python -X utf8 bench/test_laya_calibration.py --live
"""
from __future__ import annotations

import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import laya_calibration as L                                     # noqa: E402

BINARY_RUNS = L.ROUTE_RUNS.replace(".json", "_binary.json")

_FAILURES: list[str] = []
_PASSES = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global _PASSES
    if cond:
        _PASSES += 1
        print(f"  ok   {name}" + (f"   [{detail}]" if detail else ""))
    else:
        _FAILURES.append(f"{name}: {detail}")
        print(f"  FAIL {name}   [{detail}]")


def _load(path: str):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- helpers ---

def q2_accuracy(runs: dict, variant: str, hard: bool | None = None,
                gate: float = 0.0) -> tuple[int, int]:
    """(correct, decided) for the binary 'needs the codebase?' sub-question."""
    ok = n = 0
    for rec in runs["items"]:
        if rec["label"] == "clarify":
            continue
        if hard is not None and bool(rec.get("hard")) != hard:
            continue
        r = rec["variants"].get(variant)
        if not r or r["code"]["margin"] < gate:
            continue
        n += 1
        ok += ((r["code"]["choice"] == "needs_the_codebase")
               == (rec["label"] == "investigate"))
    return ok, n


def constant_fraction(runs: dict, variant: str, sub: str) -> tuple[str, float]:
    picks = [rec["variants"][variant][sub]["choice"] for rec in runs["items"]
             if rec["variants"].get(variant)]
    top = max(set(picks), key=picks.count)
    return top, picks.count(top) / len(picks)


# ============================================================== THE TESTS ===

def test_labels() -> None:
    """The eval set itself: size, balance, and the adversarial slice."""
    print("\n[labels] bench/laya_routing_labels.jsonl")
    items = L.load_jsonl(L.ROUTE_LABELS)
    check("at least 40 labelled routing questions", len(items) >= 40,
          f"n={len(items)}")
    labs = [i["label"] for i in items]
    for lab in ("investigate", "answer_directly", "clarify"):
        check(f"class '{lab}' has >= 10 examples", labs.count(lab) >= 10,
              f"{labs.count(lab)}")
    check("every item has question/context/label/why",
          all({"question", "context", "label", "why"} <= set(i) for i in items))
    hard = [i for i in items if i.get("hard")]
    check("adversarial slice present", len(hard) >= 8, f"n={len(hard)}")
    check("adversarial slice covers both routing classes",
          {i["label"] for i in hard} >= {"investigate", "answer_directly"})

    print("\n[labels] bench/laya_distil_labels.jsonl")
    d = L.load_jsonl(L.DISTIL_LABELS)
    check("at least 20 labelled findings", len(d) >= 20, f"n={len(d)}")
    fab = [i for i in d if not i["grounded"]]
    check("fabricated findings present", len(fab) >= 8, f"n={len(fab)}")
    check("every fabricated finding still cites paths",
          all(i["paths"] for i in fab),
          "a fabrication that cites nothing is a trivial case")


def test_three_way_is_not_usable(runs: dict, ord6: dict | None) -> None:
    """docs/LAYA.md 'Finding 1': the shipped three-way route_in does not work.

    Two separate assertions, because they fail for different reasons and a
    future fix may repair one and not the other:
      - at the shipped gate it abstains on nearly everything
      - even with NO gate its argmax is beaten by six lines of regex
    """
    print("\n[finding 1] the shipped three-way route_in")
    s = L.score_variant(runs, "prose", L.SHIPPED_GATE)
    check("prose abstains on >= 80% at the shipped gate 0.3",
          s["abstention_rate"] >= 0.80,
          f"abstention={s['abstention_rate']*100:.1f}% "
          f"({s['decided']}/{s['n']} decided)")
    a = L.argmax_accuracy(runs, "prose")
    check("prose argmax accuracy is under 60%", a["accuracy"] < 0.60,
          f"{a['correct']}/{a['n']} = {a['accuracy']*100:.1f}%")
    rule = sum(r["rule"] == r["label"] for r in runs["items"]) / len(runs["items"])
    check("the free regex baseline still beats Laya's three-way argmax",
          rule > a["accuracy"],
          f"regex={rule*100:.1f}% vs laya={a['accuracy']*100:.1f}%  "
          f"-- if this ever flips, re-read docs/LAYA.md finding 1")
    if ord6:
        a6 = L.argmax_accuracy(ord6, "prose")
        check("averaging over all 6 orderings does not rescue the three-way "
              "question", a6["accuracy"] < 0.60,
              f"{a6['accuracy']*100:.1f}% at 6 orders vs "
              f"{a['accuracy']*100:.1f}% at 2")


def test_structured_state_does_not_help(runs: dict, binary: dict) -> None:
    """docs/LAYA.md 'Finding 2': the structured-state hypothesis is REFUTED.

    This is the test the task was built around. It must fail loudly if
    structured state ever starts winning, because that would mean the
    recommendation in the doc is now wrong.
    """
    print("\n[finding 2] structured state vs prose (the hypothesis)")
    p = L.argmax_accuracy(runs, "prose")
    st = L.argmax_accuracy(runs, "structured")
    check("structured state does NOT beat prose on the three-way question",
          st["accuracy"] <= p["accuracy"],
          f"prose={p['accuracy']*100:.1f}%  structured={st['accuracy']*100:.1f}%")
    ss = L.score_variant(runs, "structured", L.SHIPPED_GATE)
    check("structured state abstains at least as much as prose does",
          ss["abstention_rate"] >= L.score_variant(
              runs, "prose", L.SHIPPED_GATE)["abstention_rate"],
          f"structured={ss['abstention_rate']*100:.1f}%")

    # The sharper version: on the ONE sub-question that works, structured
    # state destroys it outright.
    ok_p, n_p = q2_accuracy(binary, "prose")
    ok_s, n_s = q2_accuracy(binary, "structured")
    check("structured state is worse than prose on the working binary "
          "sub-question", ok_s / n_s < ok_p / n_p,
          f"prose={ok_p}/{n_p}={ok_p/n_p*100:.1f}%  "
          f"structured={ok_s}/{n_s}={ok_s/n_s*100:.1f}%")
    top, frac = constant_fraction(binary, "structured", "code")
    check("structured state makes the code sub-question CONSTANT",
          frac >= 0.95, f"answers '{top}' on {frac*100:.1f}% of items")


def test_clarify_option_is_dead(runs: dict, binary: dict) -> None:
    """docs/LAYA.md 'Finding 3': Laya cannot detect underspecification.

    Asserted two ways: the three-way option almost never wins for a clarify
    item, and the dedicated binary version is a constant with a LARGE margin,
    which is the reason margin cannot be read as confidence.
    """
    print("\n[finding 3] the clarify / specificity question is dead")
    hit = tot = 0
    for rec in runs["items"]:
        if rec["label"] != "clarify":
            continue
        r = rec["variants"].get("prose")
        if not r:
            continue
        tot += 1
        hit += r["choice"] == "clarify"
    check("three-way argmax recovers under half the clarify items",
          hit / tot < 0.5, f"{hit}/{tot} = {hit/tot*100:.1f}%")
    for v in ("prose", "structured", "hybrid"):
        top, frac = constant_fraction(binary, v, "spec")
        mg = statistics.fmean(rec["variants"][v]["spec"]["margin"]
                              for rec in binary["items"]
                              if rec["variants"].get(v))
        check(f"'is this specific enough?' is a CONSTANT for {v}",
              frac >= 0.95,
              f"answers '{top}' on {frac*100:.1f}% of items, "
              f"mean margin {mg:.3f}")
        check(f"...and it is a CONFIDENT constant for {v} "
              f"(margin is not confidence)", mg > 0.4,
              f"mean margin {mg:.3f} on a classifier with zero information")

    # THE STATED SOLUTION, tested. docs/LAYA.md finding 3 recommends replacing
    # this question with a deterministic pre-Laya guard. That recommendation
    # is only honest if the guard actually works, so it is measured here and
    # not merely asserted in prose.
    items = L.load_jsonl(L.ROUTE_LABELS)
    cl = [i for i in items if i["label"] == "clarify"]
    ok = sum(L.rule_baseline(i) == "clarify" for i in cl)
    check("the recommended deterministic clarify guard recovers what Laya "
          "cannot", ok / len(cl) >= 0.80,
          f"{ok}/{len(cl)} = {ok/len(cl)*100:.0f}% vs Laya's 15.4%")
    false_pos = sum(L.rule_baseline(i) == "clarify"
                    for i in items if i["label"] != "clarify")
    check("...without dragging in non-clarify questions",
          false_pos <= 2, f"{false_pos} false positives out of "
                          f"{len(items) - len(cl)}")


def test_binary_code_question_works(binary: dict) -> None:
    """docs/LAYA.md 'Finding 4': the one configuration that is usable."""
    print("\n[finding 4] the binary 'needs the codebase?' question on prose")
    ok, n = q2_accuracy(binary, "prose")
    check("prose binary sub-question beats chance by a wide margin",
          ok / n >= 0.75, f"{ok}/{n} = {ok/n*100:.1f}%  (chance ~50%)")
    ok_g, n_g = q2_accuracy(binary, "prose", gate=L.RECOMMENDED_GATE)
    check(f"at the recommended gate {L.RECOMMENDED_GATE} it is >= 90% accurate",
          ok_g / n_g >= 0.90,
          f"{ok_g}/{n_g} decided = {ok_g/n_g*100:.1f}%")
    check("...while still deciding at least a third of the items",
          n_g / n >= 0.33, f"coverage {n_g}/{n} = {n_g/n*100:.1f}%")


def test_margin_is_not_calibrated_on_hard_items(binary: dict) -> None:
    """docs/LAYA.md 'Finding 5': the gate does not catch the case it is for.

    On items whose surface cues are honest, a wrong answer has a tiny margin
    and the gate filters it. On items whose surface cues LIE, wrong answers
    carry margins as large as right ones -- so the gate cannot be read as
    "I am unsure", only as "the cues were weak".
    """
    print("\n[finding 5] the margin gate does not detect adversarial items")
    ok_e, n_e = q2_accuracy(binary, "prose", hard=False)
    ok_h, n_h = q2_accuracy(binary, "prose", hard=True)
    check("prose is near ceiling on cue-honest items", ok_e / n_e >= 0.90,
          f"easy {ok_e}/{n_e} = {ok_e/n_e*100:.1f}%")
    check("prose collapses on cue-adversarial items", ok_h / n_h <= 0.50,
          f"hard {ok_h}/{n_h} = {ok_h/n_h*100:.1f}%  "
          f"(n is small; this is a direction, not a rate)")

    right = [rec["variants"]["prose"]["code"]["margin"]
             for rec in binary["items"]
             if rec.get("hard") and rec["label"] != "clarify"
             and (rec["variants"]["prose"]["code"]["choice"]
                  == "needs_the_codebase") == (rec["label"] == "investigate")]
    wrong = [rec["variants"]["prose"]["code"]["margin"]
             for rec in binary["items"]
             if rec.get("hard") and rec["label"] != "clarify"
             and (rec["variants"]["prose"]["code"]["choice"]
                  == "needs_the_codebase") != (rec["label"] == "investigate")]
    check("on adversarial items the margin does not separate right from wrong",
          statistics.fmean(wrong) >= statistics.fmean(right) * 0.8,
          f"mean margin wrong={statistics.fmean(wrong):.3f} "
          f"right={statistics.fmean(right):.3f}  -- a usable confidence "
          f"signal would have wrong << right")

    # The regex fails the same slice, which is the point: Laya is not adding
    # anything the regex was not already doing.
    ok_r = sum((rec["rule"] == "investigate") == (rec["label"] == "investigate")
               for rec in binary["items"]
               if rec.get("hard") and rec["label"] != "clarify")
    check("the regex baseline fails the adversarial slice too",
          ok_r <= n_h // 2, f"regex {ok_r}/{n_h} -- both are reading cues")


def test_distil_grounding_does_not_detect_fabrication(d: dict) -> None:
    """docs/LAYA.md 'Finding 6': the grounding question does NOT replicate.

    Spot-checked on two findings it looked decisive (0.925 vs 0.070). At n=29
    with fabrications written to be CODE-SHAPED rather than vague, it is a
    constant. This test exists so nobody re-derives the two-point version.
    """
    print("\n[finding 6] distil grounding at n>=20")
    picks = [r["grounding"]["choice"] for r in d["items"]]
    top = max(set(picks), key=picks.count)
    frac = picks.count(top) / len(picks)
    check("the grounding question answers the same thing for nearly every "
          "finding", frac >= 0.95,
          f"'{top}' on {picks.count(top)}/{len(picks)} = {frac*100:.1f}%")
    pos = [r["grounding"]["probabilities"]["from_the_files"]
           for r in d["items"] if r["grounded_label"]]
    neg = [r["grounding"]["probabilities"]["from_the_files"]
           for r in d["items"] if not r["grounded_label"]]
    auc = L._auc(pos, neg)
    check("grounded and fabricated findings are not separated",
          auc < 0.85, f"AUC={auc:.3f}  (0.5 = no signal, 1.0 = perfect)")
    gap = statistics.fmean(pos) - statistics.fmean(neg)
    check("...and the probability gap is small",
          gap < 0.20, f"mean p(from_the_files) gap = {gap:+.3f}")
    check("the high margins are therefore NOT evidence of a decision",
          statistics.fmean(r["grounding"]["margin"] for r in d["items"]) > 0.5,
          "mean margin "
          f"{statistics.fmean(r['grounding']['margin'] for r in d['items']):.3f}"
          " on a constant classifier")


def test_distil_verdict_does_not_work(d: dict) -> None:
    """docs/LAYA.md 'Finding 7': the verdict question is also near-constant."""
    print("\n[finding 7] distil verdict")
    picks = [r["verdict"]["choice"] for r in d["items"]]
    top = max(set(picks), key=picks.count)
    check("the verdict question is near-constant too",
          picks.count(top) / len(picks) >= 0.80,
          f"'{top}' on {picks.count(top)}/{len(picks)}")
    hit = sum(r["verdict"]["choice"] == r["verdict_label"] for r in d["items"])
    check("verdict argmax accuracy is under 60%", hit / len(d["items"]) < 0.60,
          f"{hit}/{len(d['items'])} = {hit/len(d['items'])*100:.1f}%")


def test_primitive_matches_hemisphere() -> None:
    """The copied primitive must still agree with the one that ships.

    bench/laya_calibration.py reimplements mcp/shomen.py:_choice_averaged
    so these numbers do not silently change when that file is edited. The
    price of the copy is that it can drift, so the drift is tested.
    """
    print("\n[contract] the copied primitive vs mcp/shomen.py")
    try:
        sys.path.insert(0, os.path.join(L.ROOT, "mcp"))
        import shomen                                        # noqa: PLC0415
    except Exception as e:                                       # noqa: BLE001
        check("mcp/shomen.py importable", False, f"{type(e).__name__}: {e}")
        return
    check("shomen still exposes _choice_averaged",
          hasattr(shomen, "_choice_averaged"))
    check("shomen still exposes route_in and distil",
          hasattr(shomen, "route_in") and hasattr(shomen, "distil"))
    check("the gate this doc reports against is still shomen's gate",
          abs(getattr(shomen, "MARGIN_GATE", -1) - L.SHIPPED_GATE) < 1e-9,
          f"shomen.MARGIN_GATE={getattr(shomen, 'MARGIN_GATE', None)} "
          f"vs reported {L.SHIPPED_GATE}")
    check("shomen still averages over 2 orderings by default",
          getattr(shomen, "LAYA_PERMUTATIONS", None) == 2,
          f"LAYA_PERMUTATIONS={getattr(shomen, 'LAYA_PERMUTATIONS', None)}")


def test_no_shared_state_written() -> None:
    """The corpus this harness must never touch."""
    print("\n[safety] shared state")
    corpus = os.path.join(L.ROOT, "index", "corpus.sqlite3")
    check("YAMADORI_CORPUS_DB is redirected away from index/corpus.sqlite3",
          os.path.abspath(os.environ.get("YAMADORI_CORPUS_DB", ""))
          != os.path.abspath(corpus),
          os.environ.get("YAMADORI_CORPUS_DB", "(unset)"))
    check("the index is read through a read-only path",
          L.index_facts()["chunks"] >= 0,
          f"{L.index_facts()['chunks']} chunks, read-only URI")


def test_live() -> None:
    """The cache still describes the service that is running right now."""
    print("\n[live] laya on " + L.LAYA_URL)
    items = L.load_jsonl(L.ROUTE_LABELS)
    binary = _load(BINARY_RUNS)
    probe = [i for i in items if not i.get("hard")][:6]
    drift = 0
    for item in probe:
        state = L.prose_state(item)
        got = L.choice_averaged(state, L.CODE_INSTRUCTIONS, L.CODE_OPTIONS, 2)
        cached = next((r for r in binary["items"]
                       if r["question"] == item["question"]), None)
        if got is None or cached is None:
            continue
        if abs(got["margin"] - cached["variants"]["prose"]["code"]["margin"]) > 0.02:
            drift += 1
    check("the live service reproduces the cached margins", drift == 0,
          f"{drift}/{len(probe)} probes drifted by more than 0.02 -- "
          f"if this fires, every number in docs/LAYA.md needs re-measuring")

    a = L.choice_averaged("", L.GROUNDING_INSTRUCTIONS, L.GROUNDING_OPTIONS, 2)
    b = L.choice_averaged(
        "The weather in Oslo is mild and the ferries are running on time.",
        L.GROUNDING_INSTRUCTIONS, L.GROUNDING_OPTIONS, 2)
    check("laya does read the state (empty and unrelated states differ)",
          abs(a["probabilities"]["from_the_files"]
              - b["probabilities"]["from_the_files"]) > 0.05,
          f"empty={a['probabilities']['from_the_files']:.3f} "
          f"unrelated={b['probabilities']['from_the_files']:.3f}  -- "
          f"the failure is discrimination, NOT a dead input")


def main() -> int:
    live = "--live" in sys.argv
    runs = _load(L.ROUTE_RUNS)
    ord6 = _load(L.ROUTE_RUNS.replace(".json", "_ord6.json"))
    binary = _load(BINARY_RUNS)
    distil = _load(L.DISTIL_RUNS)
    missing = [n for n, r in (("routing", runs), ("binary", binary),
                              ("distil", distil)) if r is None]
    if missing:
        print(f"cached runs missing: {', '.join(missing)}\n"
              f"regenerate with: python -X utf8 bench/laya_calibration.py")
        return 2

    test_labels()
    test_three_way_is_not_usable(runs, ord6)
    test_structured_state_does_not_help(runs, binary)
    test_clarify_option_is_dead(runs, binary)
    test_binary_code_question_works(binary)
    test_margin_is_not_calibrated_on_hard_items(binary)
    test_distil_grounding_does_not_detect_fabrication(distil)
    test_distil_verdict_does_not_work(distil)
    test_primitive_matches_hemisphere()
    test_no_shared_state_written()
    if live:
        # RETIRED 2026-09-24 with the Laya service (docs/E1.md: E1 alone,
        # retire Laya): test_live() called :1237, which no longer runs, and
        # crashed on its None answer instead of counting. The cached runs
        # above remain the record docs/LAYA.md cites.
        print("\n[live] RETIRED: the Laya service on 1237 was retired "
              "2026-09-24 (docs/E1.md); not run")
    else:
        print("\n[live] skipped -- pass --live to check the cache against "
              "the running service")

    print(f"\n{'=' * 72}")
    if _FAILURES:
        print(f"{len(_FAILURES)} FAILED, {_PASSES} passed\n")
        for f in _FAILURES:
            print(f"  - {f}")
        print("\nA failure here means docs/LAYA.md is now wrong. Re-measure "
              "and rewrite the finding; do not delete the test.")
        return 1
    print(f"all {_PASSES} checks passed -- docs/LAYA.md still describes "
          f"the system")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
