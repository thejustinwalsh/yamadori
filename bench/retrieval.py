#!/usr/bin/env python
"""Measure retrieval alone, at full N, with no model in the loop.

WHY SEPARATE FROM THE END-TO-END RUN

A question like "does the reranker help?" is a question about ORDERING, and
ordering is decided before the model ever sees anything. Answering it through
the full agent costs a generation per task, which is 5 to 30 seconds, which is
why the first attempt used 11 queries and produced p ~= 0.6 -- a result that
says nothing, from a sample too small to say anything.

Without the model, a task costs milliseconds. The same question at N=628 can
actually be answered, and the answer is paired: the same query is run under
both conditions, so the comparison is within-task and the variance from "some
symbols are just harder" cancels instead of drowning the effect.

WHAT IS MEASURED

  hit@1     the ground-truth file is the first result
  hit@5     it is somewhere in the top five
  MRR       1/rank of the first correct result, 0 if absent

hit@1 and hit@5 answer different questions and an intervention can move one
without the other. That is exactly what the first reranker run saw (1/11 to
3/11 at rank 1, 7/11 either way at rank 5) and it is why both are reported.

STATISTICS

McNemar's exact test on the discordant pairs, because the data is paired and
binary: what matters is how many tasks one condition got right and the other
got wrong, not the totals. Tasks both conditions agree on carry no information
about the difference between them and are correctly ignored.

A Wilson interval is given for each rate. It is asymmetric and behaves at the
extremes, where the usual normal approximation produces intervals that extend
past 100%.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "mcp"))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Confidence interval for a proportion that stays inside [0, 1]."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided p for b wins against c, ignoring the ties.

    Under the null the discordant pairs split 50/50, so this is a two-sided
    binomial test on b successes out of b+c.
    """
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1))
    return min(1.0, 2.0 * tail / (2 ** n))


def norm(path: str) -> str:
    return (path or "").replace("\\", "/").strip().lower().lstrip("./")


def rank_of(truth: str, results: list[str]) -> int:
    """1-based rank of the ground-truth path, or 0 if it is not there.

    Suffix matching, because a result may be repository-relative while the
    truth is package-relative or the other way around. Requiring a segment
    boundary stops `core/trait/types.ts` from being satisfied by
    `other/types.ts`.
    """
    t = norm(truth)
    for i, r in enumerate(results, 1):
        r = norm(r)
        if r == t or r.endswith("/" + t) or t.endswith("/" + r):
            return i
    return 0


def search_paths(query: str, top_k: int, rerank_cutoff: int) -> list[str]:
    """Ranked paths, or a raised exception. Never a quietly empty list.

    An earlier version caught everything here and returned []. The embedding
    endpoint was 404ing on every call, each one became "no results", and the
    benchmark reported 0.0% for every condition on every source as though that
    were a finding about retrieval. An error and a miss are different outcomes
    and the harness has to be able to tell them apart -- rule 3.
    """
    import code_search as cs
    old = cs.RERANK_MAX_K
    cs.RERANK_MAX_K = rerank_cutoff          # top_k > cutoff means "skip rerank"
    try:
        return [c["path"] if isinstance(c, dict) else c.path
                for c in cs.search(query, top_k=top_k)]
    finally:
        cs.RERANK_MAX_K = old


def symbol_paths(db: str, symbol: str, top_k: int) -> list[str]:
    """Exact symbol lookup against the tree-sitter index in sqlite.

    This is the tool that OWNS "where is X defined?". `defs` is populated by
    walking the parse tree, so the answer is not retrieved or ranked, it is
    looked up -- and the first benchmark did not measure it at all, asking
    embedding search to do symbol lookup instead. An identifier is exactly the
    query a vector model handles worst and a B-tree handles best.

    Ordered by kind so a declaration outranks a re-export of the same name.
    """
    import sqlite3

    con = sqlite3.connect(db)
    try:
        rows = con.execute(
            "SELECT path, kind FROM defs WHERE name = ? "
            "ORDER BY CASE kind WHEN 'class' THEN 0 WHEN 'interface' THEN 0 "
            "WHEN 'struct' THEN 0 WHEN 'type' THEN 1 WHEN 'function' THEN 1 "
            "ELSE 2 END LIMIT ?", (symbol, top_k)).fetchall()
    except sqlite3.Error:
        return []
    finally:
        con.close()
    return [p for p, _k in rows]


def con_complete(db: str) -> bool:
    import sqlite3
    try:
        con = sqlite3.connect(db)
        try:
            row = con.execute("SELECT v FROM meta WHERE k='complete'").fetchone()
        finally:
            con.close()
        return bool(row) and row[0] == "1"
    except sqlite3.Error:
        return False


def coverage(db: str) -> tuple[int, int] | None:
    """(files indexed, files on disk) for an index whose source is still here.

    The completion marker only exists on indexes built after it was added, so
    every index predating it would be refused forever -- which is the wrong
    kind of strict. Completeness can be checked directly instead: walk the
    source tree the index was built from and compare the file count.

    This is the stronger check of the two, because it verifies the index
    against the thing it claims to describe rather than against a flag the
    same run wrote about itself. None when the source tree is not available,
    in which case the marker is all there is.
    """
    import sqlite3

    sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
    try:
        from index_code import iter_files
    except Exception:                                            # noqa: BLE001
        return None

    con = sqlite3.connect(db)
    try:
        row = con.execute("SELECT path FROM roots LIMIT 1").fetchone()
        n_indexed = con.execute(
            "SELECT COUNT(DISTINCT path) FROM chunks").fetchone()[0]
    except sqlite3.Error:
        return None
    finally:
        con.close()
    if not row or not row[0] or not os.path.isdir(row[0]):
        return None
    return n_indexed, sum(1 for _ in iter_files(row[0]))


def check_alive(db: str, source: str) -> str | None:
    """Why this index must not be benchmarked, or None if it is fit -- rule 1.

    Checks both halves separately, because they fail independently: the symbol
    table is built by tree-sitter with no network, while the vectors need the
    embedding server. Both package indexes had a complete symbol table and not
    one usable vector, which is why "the index exists" is not a check.

    Returns a reason rather than exiting, so one unfit index does not block
    measurement of the others -- but the reason is reported at the top of the
    results, not swallowed. An index silently excluded is how you end up
    reporting a number from a corpus nobody meant to measure.
    """
    import sqlite3

    import numpy as np

    con = sqlite3.connect(db)
    try:
        n_defs = con.execute("SELECT COUNT(*) FROM defs").fetchone()[0]
        rows = con.execute(
            "SELECT vec FROM chunks ORDER BY RANDOM() LIMIT 200").fetchall()
    finally:
        con.close()
    if not rows:
        return "index has no chunks"
    cov = coverage(db)
    if cov is not None:
        indexed, on_disk = cov
        # Not every file yields a chunk -- empty files, unsupported grammars --
        # so exact equality is the wrong bar. Losing a tenth of the tree is not.
        if on_disk and indexed < on_disk * 0.9:
            return (f"only {indexed} of {on_disk} source files are indexed "
                    f"({indexed / on_disk:.0%}), so the build did not finish")
        print(f"    coverage: {indexed}/{on_disk} source files indexed")
    elif not con_complete(db):
        return ("no completion marker and no source tree to check against, so "
                "completeness cannot be established. A truncated index looks "
                "healthy -- every row it holds is correct -- and three.js "
                "silently kept 1,760 of 15,021 chunks this way")
    dead = sum(1 for (b,) in rows
               if not b or float(np.linalg.norm(
                   np.frombuffer(b, dtype=np.float32))) < 1e-6)
    if dead > len(rows) * 0.02:
        return (f"{dead} of {len(rows)} sampled chunks have zero vectors, so "
                f"it cannot answer a semantic query at all")
    print(f"    alive: {n_defs} symbols, {len(rows) - dead}/{len(rows)} "
          f"sampled vectors non-zero")
    return None


def db_for(source: str) -> str | None:
    import repos
    for e in repos.known():
        if os.path.basename(e["root"]).lower() == source:
            return repos.db_path(e["root"])
    pkg = os.path.join(HERE, "..", "index", "packages")
    if os.path.isdir(pkg):
        for fn in sorted(os.listdir(pkg)):
            if fn.endswith(".sqlite3") and fn.rpartition("@")[0].lower() == source:
                return os.path.join(pkg, fn)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=os.path.join(HERE, "tasks.jsonl"))
    ap.add_argument("--limit", type=int, default=0, help="0 means all")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(HERE, "retrieval_results.jsonl"))
    args = ap.parse_args()

    import code_search as cs

    tasks = [json.loads(l) for l in open(args.tasks, encoding="utf-8")]
    tasks = [t for t in tasks if t["kind"] == "locate"]
    if args.limit:
        tasks = tasks[:args.limit]

    # Grouped by source so each index is loaded once rather than per task.
    by_source: dict[str, list[dict]] = {}
    for t in tasks:
        by_source.setdefault(t["source"], []).append(t)

    rows: list[dict] = []
    errors: list[dict] = []
    skipped: list[tuple] = []
    for source, group in sorted(by_source.items()):
        db = db_for(source)
        if not db or not os.path.exists(db):
            print(f"  {source}: no index, skipped")
            continue
        print(f"  {source}: {len(group)} tasks", flush=True)
        unfit = check_alive(db, source)
        if unfit:
            print(f"    SKIPPED -- {unfit}")
            skipped.append((source, len(group), unfit))
            continue
        os.environ["CODE_INDEX_DB"] = db
        cs.INDEX_DB = db

        for i, t in enumerate(group):
            query = t["symbol"]
            try:
                t0 = time.time()
                sym = symbol_paths(db, query, args.top_k)
                t_sym = time.time() - t0
                t0 = time.time()
                # top_k > cutoff disables reranking inside search(), so a
                # cutoff of 0 is "never rerank" and a large one is "always".
                base = search_paths(query, args.top_k, rerank_cutoff=0)
                t_base = time.time() - t0
                t0 = time.time()
                rr = search_paths(query, args.top_k, rerank_cutoff=999)
                t_rr = time.time() - t0
            except Exception as e:                               # noqa: BLE001
                errors.append({"id": t["id"], "error": f"{type(e).__name__}: {e}"})
                continue

            rows.append({
                "id": t["id"], "source": source, "symbol": query,
                "truth": t["truth"]["path"],
                "symbol_rank": rank_of(t["truth"]["path"], sym),
                "embed_rank": rank_of(t["truth"]["path"], base),
                "rerank_rank": rank_of(t["truth"]["path"], rr),
                "symbol_ms": round(t_sym * 1000, 1),
                "embed_ms": round(t_base * 1000, 1),
                "rerank_ms": round(t_rr * 1000, 1),
            })
            if (i + 1) % 25 == 0:
                print(f"    {i + 1}/{len(group)}", end="\r", flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    if skipped:
        # Stated before the numbers, never after. A reader has to know which
        # corpora are NOT in the table they are about to read.
        print("\n  " + "!" * 60)
        for src, n, why in skipped:
            print(f"  EXCLUDED  {src} ({n} tasks): {why}")
        print("  " + "!" * 60)

    report(rows)
    if errors:
        # Reported separately and never folded into the miss rate. A run where
        # most tasks errored is a broken harness, not a poor result.
        print(f"\n  ERRORS: {len(errors)} of {len(rows) + len(errors)} tasks "
              f"raised and were excluded")
        for e in errors[:3]:
            print(f"    {e['id']}: {e['error']}")
        if len(errors) > len(rows):
            print("  more tasks errored than succeeded -- treat the table above "
                  "as a harness failure, not a measurement")
    print(f"\n  written {os.path.abspath(args.out)}")


CONDITIONS = (
    ("symbol table", "symbol_rank", "symbol_ms"),
    ("embedding", "embed_rank", "embed_ms"),
    ("embed+rerank", "rerank_rank", "rerank_ms"),
)


def report(rows: list[dict]) -> None:
    """Three conditions, because two of them were answering the wrong question.

    "Where is X defined?" is a lookup, not a retrieval problem: the tree-sitter
    index in `defs` holds the answer exactly and a B-tree finds it. The first
    version of this benchmark did not measure that path at all -- it compared
    embedding search against embedding search plus a reranker, on a query type
    neither is for, and the winner was still the wrong tool.
    """
    n = len(rows)
    if not n:
        print("  no rows")
        return

    print(f"\n{'=' * 78}\n  SYMBOL LOOKUP, N = {n}, mechanical ground truth, "
          f"paired\n{'=' * 78}")
    head = "  {:<10}".format("metric") + "".join(
        f"{c:<23}" for c, _, _ in CONDITIONS)
    print(head)
    print("  " + "-" * 74)

    for label, pred in (("hit@1", lambda r: r == 1),
                        ("hit@5", lambda r: 1 <= r <= 5)):
        cells = []
        for _c, rk, _ms in CONDITIONS:
            k = sum(pred(r[rk]) for r in rows)
            lo, hi = wilson(k, n)
            cells.append(f"{k / n:6.1%} [{lo:.0%}-{hi:.0%}]".ljust(23))
        print(f"  {label:<10}" + "".join(cells))

    cells = []
    for _c, rk, _ms in CONDITIONS:
        mrr = sum(1 / r[rk] for r in rows if r[rk]) / n
        cells.append(f"{mrr:6.3f}".ljust(23))
    print(f"  {'MRR':<10}" + "".join(cells))

    base_ms = sum(r[CONDITIONS[0][2]] for r in rows) / n
    cells = []
    for _c, _rk, ms in CONDITIONS:
        v = sum(r[ms] for r in rows) / n
        cells.append(f"{v:8.2f} ms ({v / max(base_ms, 1e-9):>5.0f}x)".ljust(23))
    print(f"  {'latency':<10}" + "".join(cells))

    print("\n  McNemar, paired and exact -- only discordant tasks count")
    pairs = (("symbol table", "symbol_rank", "embedding", "embed_rank"),
             ("embedding", "embed_rank", "embed+rerank", "rerank_rank"),
             ("symbol table", "symbol_rank", "embed+rerank", "rerank_rank"))
    for a_name, a_rk, b_name, b_rk in pairs:
        b = sum(1 for r in rows if r[b_rk] == 1 and r[a_rk] != 1)
        c = sum(1 for r in rows if r[a_rk] == 1 and r[b_rk] != 1)
        p = mcnemar(b, c)
        verdict = (f"{b_name} better" if b > c and p < 0.05 else
                   f"{a_name} better" if c > b and p < 0.05 else
                   "no detectable difference")
        print(f"    hit@1  {a_name} vs {b_name}: "
              f"{b_name} wins {b}, {a_name} wins {c}, p = {p:.4f}  -> {verdict}")

    print("\n  by source, hit@1")
    srcs: dict[str, list[dict]] = {}
    for r in rows:
        srcs.setdefault(r["source"], []).append(r)
    for s, g in sorted(srcs.items()):
        cells = "".join(
            f"{sum(1 for r in g if r[rk] == 1) / len(g):>9.1%}"
            for _c, rk, _ms in CONDITIONS)
        print(f"    {s:<10} n={len(g):<5}{cells}")


if __name__ == "__main__":
    main()
