#!/usr/bin/env python
"""The arithmetic of docs/DECIDE-BATCH.md section 7: the per-read cost fit on the real bonsai-a4000 JevBench run, the
upper bound on what one request instead of two can save there, the recurrent-memory table of section 4.1, and the speed
table by K. Offline, no network, no GPU, nothing written.

    python bench/decider/batch_arith.py

Every parameter says where it comes from. The SUPPLIED fit (0.26 s per read, 2.3 ms per token) is the request's (Decision
Index run, 2026-10-06, 86 rows); the other is COMPUTED here from the committed JevBench items. Block token counts are
ESTIMATES from the template's structure until the engine's `processed` field says otherwise.
"""
from __future__ import annotations

import json
import math
import os
import statistics as st

ITEMS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "jevbench",
                     "bonsai-a4000-20261006-193326", "items.jsonl")

A = 0.26              # s per read, fixed: supplied fit
B = 0.0023            # s per processed token: supplied fit
W_BYTES = 5946648928  # Ternary-Bonsai-2-27B-PTQ1_0.gguf, the file's size on disk
BW = 448e9            # RTX A4000 memory bandwidth, NVIDIA datasheet
F = W_BYTES / BW      # the weights are read once per decode call: a lower bound on one call's time
MIB_PER_CELL = 2244.38 / 15   # one recurrent cell: bench/results/kv_placement/*/kvplace.log, 3 cells x (1 + 4 rs_seq) rows
KV_B = 35840          # attention KV bytes per cell, q8_0: bench/a4000_fit.py


def ols(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return a, b, 1 - ss_res / ss_tot


def block(q):
    """(total, shared head, own tail) tokens of one order's block for a question text of q tokens: header 3 + "QUESTION: "
    3 + q + "\\n\\nOPTIONS:\\n" 4 + the two options 8 + the instruction line 9 + the assistant header and "Answer:" 13; the
    head runs through the first option's letter and dot."""
    total = 40 + q
    head = 12 + q
    return total, head, total - head


def seq(K, S, q, state_cached=False):
    t, _, _ = block(q)
    reads = 2 * K
    return reads * A + B * ((0 if state_cached else S) + reads * t)


def batched(K, S, q, W, share=True, state_cached=False):
    t, h, o = block(q)
    toks = (0 if state_cached else S) + K * ((h + 2 * o) if share else 2 * t)
    waves = math.ceil(K / max(1, W // 2))
    calls = waves * (2 if share else 1) + (0 if state_cached else math.ceil(S / 512))
    return A + B * toks + F * calls, waves, calls


def main() -> None:
    print(f"weights-read floor per decode call: {F * 1000:.1f} ms")
    print()
    print("recurrent memory (np=2 slots today = 2 cells):")
    for w in (1, 2, 4, 8, 26):
        extra = (w + 1) * MIB_PER_CELL
        cells = extra * 2 ** 20 / KV_B
        print(f"  W={w:2d}: +{w + 1:2d} cells = +{extra:7.1f} MiB = {cells / 1000:5.1f}k KV cells of -c "
              f"(141,312 -> {141312 - cells:,.0f})")
    print()

    # ---- the fit computed from the committed run
    rows = []
    for ln in open(ITEMS, encoding="utf-8"):
        r = json.loads(ln)
        d = r.get("diagnostics") or {}
        if r.get("ok") and d.get("ms") is not None and d.get("processed_tokens") is not None:
            rows.append((d["ms"], d["reads"], d["processed_tokens"]))
    ms = [r[0] for r in rows]
    pt = [r[2] for r in rows]
    print(f"bonsai-a4000 JevBench run: {len(rows)} items, reads per item {sorted({r[1] for r in rows})}, processed tokens "
          f"min {min(pt)} median {int(st.median(pt))} max {max(pt)}, ms median {st.median(ms):.0f}")
    a, b, r2 = ols(pt, ms)
    print(f"  ms = {a:.1f} + {b:.3f} * processed_tokens   (R2 {r2:.3f}); per read intercept {a / 2:.1f} ms")
    big = [(p, m) for p, m in zip(pt, ms) if p > 600]
    a2, b2, r22 = ols([p for p, _ in big], [m for _, m in big])
    print(f"  items over 600 tokens (n={len(big)}): ms = {a2:.1f} + {b2:.3f} * tokens (R2 {r22:.3f})")
    tot = sum(ms)
    gain = len(rows) * (a / 2)
    print(f"  sum of reads {tot / 1000:.1f} s; one request instead of two saves at most one read's intercept per item: "
          f"{gain / 1000:.1f} s = {100 * gain / tot:.1f}%  (x{tot / (tot - gain):.2f})")
    for lo, hi in ((0, 200), (200, 400), (400, 1000), (1000, 10000)):
        s = [(m, p) for m, p in zip(ms, pt) if lo <= p < hi]
        if s:
            t = sum(m for m, _ in s)
            g = len(s) * (a / 2)
            print(f"    processed {lo:5d}-{hi:5d}: n={len(s):3d} median ms {st.median([m for m, _ in s]):6.0f}  bound x{t / (t - g):.2f}")
    print()

    # ---- the speed table (supplied fit, estimated block tokens)
    for q in (30, 60):
        t, h, o = block(q)
        print(f"question text q = {q} tokens: block {t} tokens, shared head {h}, own tail {o} per order (W=4)")
        print(f"  {'K':>3} {'S':>5} | {'seq s':>7} | {'batch no-share':>14} {'x':>5} | {'batch shared':>12} {'x':>5}  waves calls")
        for S in (500, 2000, 8000):
            for K in (1, 2, 5, 10, 26):
                s = seq(K, S, q)
                b1, _, _ = batched(K, S, q, 4, share=False)
                b2, w, c = batched(K, S, q, 4, share=True)
                print(f"  {K:3d} {S:5d} | {s:7.2f} | {b1:14.2f} {s / b1:5.2f} | {b2:12.2f} {s / b2:5.2f}  {w:5d} {c:5d}")
        print("  state already resident:")
        for K in (1, 2, 5, 10, 26):
            s = seq(K, 0, q, True)
            b2, w, c = batched(K, 0, q, 4, share=True, state_cached=True)
            print(f"  {K:3d}   -  | {s:7.2f} | {'':14} {'':5} | {b2:12.2f} {s / b2:5.2f}  {w:5d} {c:5d}")
        print()


if __name__ == "__main__":
    main()
