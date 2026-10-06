"""Deterministic request lists for running jjava through the Decision Index harness (no outcome-based selection).

  python make_samples.py compat   OUT.jsonl.gz   # 2 requests per benchmark (86 when 43 benchmarks are present)
  python make_samples.py latency  OUT.jsonl.gz   # the harness's latency-v1 design (750 rows), reconstructed
  (the harness's own `suite sample --n N` gives the round-robin per-benchmark subset; not duplicated here)

Run with the harness venv, PYTHONUTF8=1, cwd = the harness checkout (suite-0.2/ staged by `suite import`).
Neither list is the lab's own file: the compat pass's exact rows and the 750-row latency sample's exact random start
are not in the kit (the lab's latency sample is sha256 d6666eb5..., seed 20260926); these follow the published
design and are labelled as reconstructions wherever they are reported.
"""
import collections
import gzip
import hashlib
import json
import random
import sys

from decision_index.suite.io import Suite, dumps

# methodology v0.2.1 `latency.reproductions.protocol.sample.per_catalog` (750 timed rows), HLE (45) = 3 is not buildable here
PER_CATALOG = {1: 9, 2: 5, 3: 2, 4: 16, 5: 27, 6: 50, 9: 0, 10: 13, 11: 0, 12: 16, 20: 25, 21: 13, 22: 10, 23: 25, 24: 70, 25: 1, 26: 11, 27: 6, 28: 7, 29: 49, 30: 13, 31: 25, 32: 4, 33: 8, 36: 3, 37: 25, 38: 8, 39: 4, 40: 23, 41: 15, 42: 28, 43: 2, 44: 25, 45: 3, 48: 50, 50: 8, 56: 10, 57: 60, 58: 27, 59: 14, 61: 20, 62: 18, 64: 2}


def write(rows, out):
    with gzip.open(out, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(dumps(r) + "\n")
    return len(rows)


def compat(suite, seed=20260919):
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in suite.rows(apply_exclusions=True):
        e = r["_evaluation"]
        by[e["catalog_id"]][e["track"]].append(r)
    out = []
    for cid in sorted(by):
        tracks = sorted(by[cid])
        first = []
        for t in tracks:  # best hash-ordered row of each track, then the benchmark's first two overall by hash
            rows = sorted(by[cid][t], key=lambda r: hashlib.sha256(f"{seed}:compat:{r['_evaluation']['run_id']}".encode()).hexdigest())
            first.append(rows[0])
        first.sort(key=lambda r: hashlib.sha256(f"{seed}:compat:{r['_evaluation']['run_id']}".encode()).hexdigest())
        chosen = first[:2]
        if len(chosen) < 2:  # a one-track benchmark: its second hash-ordered row
            rows = sorted(by[cid][tracks[0]], key=lambda r: hashlib.sha256(f"{seed}:compat:{r['_evaluation']['run_id']}".encode()).hexdigest())
            chosen = rows[:2]
        out += chosen
    return out


def latency(seed=20260926):
    suite = Suite("suite-0.2", "0.2")  # the 0.2 scope: ACOS subset only, 442 exclusions
    scope = collections.defaultdict(list)
    for r in suite.rows(apply_exclusions=True):
        e = r["_evaluation"]
        if e["catalog_id"] != 34:  # SimpleBench left out of the live scope
            scope[e["catalog_id"]].append((len(dumps({"state": r["state"], "questions": r["questions"]})), e["run_id"], r))
    rng = random.Random(seed)
    out = []
    for cid in sorted(scope):
        k = PER_CATALOG.get(cid, 0)
        rows = sorted(scope[cid], key=lambda x: (x[0], x[1]))
        if not k:
            continue
        step = len(rows) / k
        start = rng.random() * step
        out += [rows[min(len(rows) - 1, int(start + i * step))][2] for i in range(k)]
    return out


if __name__ == "__main__":
    mode, path = sys.argv[1], sys.argv[2]
    rows = compat(Suite("suite-0.2", "0.2.1")) if mode == "compat" else latency()
    print(json.dumps({"mode": mode, "rows": write(rows, path), "benchmarks": len({r["_evaluation"]["catalog_id"] for r in rows}), "fields": sum(len(r["questions"]) for r in rows)}))
