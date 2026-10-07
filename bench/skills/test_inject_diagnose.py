#!/usr/bin/env python
"""The injector's diagnosis (bench/skills/inject_diagnose.py): offline,
synthetic runs with known answers.

    python bench/skills/test_inject_diagnose.py   -> "N/M checks passed"

GATES: the AUROC is the Mann-Whitney form with ties half; the per-level
separations, the per-bucket and per-skill tables, the within-case top-1 test,
the readout's per-order argmax shares and order agreement, the stage-3 table
and the paired d-minus-a bootstrap come out as they are constructed.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "mcp"))
import offline_stores  # noqa: E402

_TMP = offline_stores.isolate("yamadori_test_inject_diagnose_")
sys.path.insert(0, HERE)

import inject_diagnose as DG  # noqa: E402

CHECKS: list = []


def check(name, ok, detail=None):
    CHECKS.append((name, bool(ok)))
    if not ok:
        print("FAIL", name, "" if detail is None else str(detail)[:400])


def main() -> int:
    check("auroc: perfect, reversed, tied",
          DG.auroc([0.9, 0.8], [0.1, 0.2]) == 1.0
          and DG.auroc([0.1], [0.9]) == 0.0
          and DG.auroc([0.5], [0.5]) == 0.5
          and DG.auroc([], [0.5]) is None)
    levels = [3, 1, 0, 2]
    cases, runs, dec, truth = {}, [], {}, {}
    for k in range(6):
        kind = "step" if k < 3 else "user"
        cid = f"c{k}"
        cases[cid] = {"case": cid, "state_sha": f"s{k % 4}", "kind": kind,
                      "state_info": {"tokens": 30 if kind == "user" else 400},
                      "items": [{"key": f"{cid}#{i}", "skill": f"sk{i % 2}",
                                 "fact": "f"} for i in range(4)]}
        for v, bel in (("a", [0.9, 0.5, 0.1, 0.6]),
                       ("d", [0.4, 0.5, 0.45, 0.6])):
            its = []
            for i in range(4):
                did = f"{cid}{v}{i}"
                p3 = bel[i]
                dec[did] = {"orders": [
                    {"printed": ["0", "1", "2", "3"],
                     "p": {"0": 0.1, "1": 0.1, "2": 0.1, "3": 0.7}},
                    {"printed": ["3", "2", "1", "0"],
                     "p": {"0": 0.6, "1": 0.1, "2": 0.1, "3": 0.2}}],
                    "disagreement": 0.5, "label_mass_min": 0.9}
                its.append({"key": f"{cid}#{i}", "sha": f"sha{i}",
                            "belief": p3, "decision_id": did, "probs": {},
                            "parts": {"next": {"noul": p3}} if v == "a"
                            else {}})
            runs.append({"case": cid, "variant": v, "kind": kind,
                         "group": "hermes", "items": its,
                         "stage3": {"noul": 0.9 if (v == "a" or k % 2) else
                                    0.2, "listed": [f"{cid}#0", f"{cid}#1"] if k % 2 == 0
                                    else [f"{cid}#2", f"{cid}#3"],
                                    "how": "passed"}})
        for i, lv in enumerate(levels):
            truth[(cid, f"{cid}#{i}")] = lv
    rep = DG.report(cases, runs, dec, truth)
    a = rep["variants"]["a"]
    check("a variant that ranks NEEDED first separates fully, overall and "
          "by kind", a["auroc"]["all"] == 1.0 and a["auroc"]["step"] == 1.0
          and a["auroc"]["user"] == 1.0, a["auroc"])
    check("per-level separation and mean belief by true level",
          a["vs_level"] == {"3_vs_0": 1.0, "3_vs_1": 1.0, "3_vs_2": 1.0}
          and a["mean_belief_by_level"] == {"0": 0.1, "1": 0.5, "2": 0.6,
                                            "3": 0.9}, a["vs_level"])
    d = rep["variants"]["d"]
    check("a variant that scores NEEDED below its AREA / DONE items does not",
          d["vs_level"]["3_vs_1"] == 0.0 and d["auroc"]["all"] < 0.5
          and d["vs_level"]["3_vs_0"] == 0.0, d["vs_level"])
    t = a["at_threshold"]["all"]
    check("precision and recall at a threshold: >= 0.5 injects the 3 items "
          "of 4 per case, one needed",
          t["0.5"]["injected"] == 18 and t["0.5"]["recall"] == 1.0
          and abs(t["0.5"]["precision"] - 1 / 3) < 1e-3
          and t["base_rate"] == 0.25, t["0.5"])
    ro = a["readout"]
    check("readout: the argmax share of each level in each printed order, "
          "the orders' variation and agreement",
          ro["argmax_share_order1"]["3"] == 1.0
          and ro["argmax_share_order2"]["0"] == 1.0
          and ro["argmax_agree"] == 0.0 and ro["mean_total_variation"] == 0.5
          and ro["middle_level_argmax_share"] == 0.0
          and ro["label_mass_min_mean"] == 0.9, ro)
    wc = rep["within_case"]
    check("within a case: the top item is the needed one every time; "
          "(case, skill) groups counted",
          wc["top1_is_needed"] == 1.0 and wc["cases_with_a_needed_item"] == 6
          and wc["random_item_is_needed"] == 0.25
          and wc["case_skill_groups"] == 12, wc)
    check("kinds and state-length buckets are tabled with their needed rate",
          rep["by_kind"]["step"]["n"] == 12
          and rep["by_kind"]["user"]["rate"] == 0.25
          and "<=60" in rep["by_state_tokens"]
          and ">800" not in rep["by_state_tokens"]
          and "<=800" in rep["by_state_tokens"], rep["by_state_tokens"])
    ls = rep["label_set"]
    check("the label set: levels, distinct states per kind (a repeated "
          "state is ONE state), items per kind",
          ls["levels"] == {"0": 6, "1": 6, "2": 6, "3": 6}
          and ls["user_items"] == 12 and ls["step_items"] == 12
          and ls["distinct_states"]["user"] + ls["distinct_states"]["step"]
          >= 3, ls)
    s3 = rep["stage3"]
    check("stage 3: AUROC per variant and the paired d-minus-a bootstrap "
          "with its interval", s3["variants"]["a"]["cases"] == 6
          and "d_minus_a" in s3 and len(s3["d_minus_a"]["ci95"]) == 2,
          s3)
    pv = a["parts_vs_level"]["next"]
    check("a part is scored alone against each level (does NEXT alone read "
          "AREA as NEEDED?)", pv["3_vs_1"] == 1.0 and pv["3_vs_0"] == 1.0
          and "parts_vs_level" not in rep["variants"]["d"], pv)
    check("stage 3 has a by-case interval per variant",
          len(s3["variants"]["a"]["auroc_ci95_by_case"] or [0, 0]) == 2)
    ip = rep["item_prior"]
    check("the item prior table has overall, step and user scopes",
          set(ip) >= {"rows", "all", "step", "user"}, ip)
    ok = sum(1 for _n, v in CHECKS if v)
    print(f"{ok}/{len(CHECKS)} checks passed")
    return 0 if ok == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
