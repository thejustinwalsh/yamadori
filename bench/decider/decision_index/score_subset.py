"""Score a PARTIAL Decision Index run (a sampled subset) with the harness's own scorers, coverage-free.

The harness's `score` counts every unrun request as wrong (coverage), so a subset scores ~0. Here the suite is
restricted to the rows the run covers (groups are complete in `suite sample`), so coverage is 1 and the per-benchmark
raw / chance-corrected skill are the board's own formulas (index02) on those rows. The chance level is the BOARD's
full-benchmark chance, not recomputed for the subset. No index is produced or reported: areas are the harness's
weighted means of the benchmarks present (indicative only).

  python score_subset.py RUN_DIR [--board index-v0.2.1.board.json] --out summary.json
Run with the harness venv, PYTHONUTF8=1, cwd = harness checkout.
"""
import argparse, collections, json, math
from pathlib import Path

from decision_index.pipeline import score_run
from decision_index.scoring.report import load_results
from decision_index.suite.io import Suite

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--board", default="index-v0.2.1.board.json")
ap.add_argument("--out", required=True)
ap.add_argument("--entrants", nargs="*", default=["jev", "kev-9b-raised", "rune-26b-a4b-v3", "simple-jev-qwen3.8-27b"])
a = ap.parse_args()
run = Path(a.run)
results = load_results(run / "results.jsonl")


class Subset(Suite):
    def rows(self, apply_exclusions=False):
        for r in super().rows(apply_exclusions):
            if r["_evaluation"]["run_id"] in results:
                yield r


S = Subset("suite-0.2", "0.2.1")
tmp = run / "subset-scored"
score_run(S, run / "results.jsonl", "jjava-subset", tmp)
idx = json.load(open(tmp / "index.json"))
summ = {b["catalog_id"]: b for b in json.load(open(tmp / "benchmark-summary.json"))["benchmarks"]}
board = json.load(open(a.board))
ref = {"jev": board["jev"]}
for m in board["models"]:
    ref[m["engine"]] = m
meta = {int(k): v for k, v in board["benchmarks"].items()}
n_rows = collections.Counter(); n_groups = collections.defaultdict(set); names = {}
for r in S.rows(True):
    e = r["_evaluation"]; names[e["catalog_id"]] = e["dataset"]; n_rows[e["catalog_id"]] += 1; n_groups[e["catalog_id"]].add(e["group_id"])
rows = []
for k, v in idx["benchmarks"].items():
    k = int(k)
    if not n_rows[k]:
        continue
    n = len(n_groups[k]) if k in (2, 36, 38) else n_rows[k]   # retrieval and ACOS are scored per query / review
    raw, skill, ch = v["raw"], v["skill"], v["random"]
    p = min(max(raw, 0.0), 1.0)
    # rough 95% interval on raw (normal approximation, accuracy-like; crude for F1/nDCG/Brier), mapped through the chance correction
    if v["rule"] == "vs baseline" or not n:
        lo = hi = None          # ForecastBench (Brier vs the 0.5 baseline): no interval computed
    else:
        z = 1.96   # Wilson interval on the raw score treated as a proportion (crude for F1 / nDCG / macro-F1), mapped through the chance correction
        c = (p + z * z / (2 * n)) / (1 + z * z / n)
        h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        den = (1 - ch) if ch is not None and ch < 1 else 1.0
        lo, hi = max(0.0, (c - h - (ch or 0)) / den), min(1.0, max(0.0, (c + h - (ch or 0)) / den))
    row = dict(id=k, name=names[k], in_index=v["in_index"], rule=v["rule"], n=n, rows=n_rows[k], coverage=v["coverage"],
               raw=raw, chance=ch, skill=skill, lo=None if lo is None else round(lo, 4), hi=None if hi is None else round(hi, 4), ms_med=(summ.get(k) or {}).get("median_ms"))
    for e in a.entrants:
        b = (ref[e].get("benchmarks") or {}).get(str(k))
        row[e] = b["skill"] if b else None
    rows.append(row)
rows.sort(key=lambda r: r["id"])
json.dump({"engine": "jjava (jjava-bonsai-a4000)", "rows": rows, "areas_indicative": idx["areas"], "entrants": {e: {"name": ref[e]["name"], "balanced_skill": ref[e]["scores"]["balanced_skill"]} for e in a.entrants}}, open(a.out, "w"), indent=1)
print(len(rows), "benchmarks")
