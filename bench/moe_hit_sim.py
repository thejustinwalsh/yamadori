"""How many routed expert reads a VRAM expert cache of N experts would serve, from a real routing trace.

The question it answers (docs/FLASH-NEXT.md section 8-9, the coordinator's 2026-09-29 cache track): before porting
a cache or freeing VRAM for it, what share of the routed (layer, expert) reads of a decode would hit VRAM at the
room we have (~2,200 experts beside the trunk, KV and MTP) and at the room KV streaming would give (~4,000-5,000)?

The trace is llama.cpp's own GGML_MOE_LOG (llama-upstream-flash 0003's diagnostic): one line per token and layer,
"blk.<l>.ffn_gate_exps.weight <id> ...", written by every CPU MUL_MAT_ID of the gate experts -- so run the engine
with every layer's experts on the CPU (--n-cpu-moe 48) and no cache, and decode; prompt batches of 32+ tokens run
on the GPU and are not logged. Policies, each over the same trace in order:

  profile   the N highest-ranked pairs of an STRP profile (Strata's data/expert-profile.bin), fixed
  lru       N / layers slots per layer, least recently used evicted on a miss (#27861's default policy)
  adaptive  the profile's N pairs, then Strata's adaptive tier as 0004 implements it: every 4 tokens the missing
            experts with usage >= 2 replace the least-used resident ones of their layer while the gain is >= 1.5,
            at most 96 swaps a step, then usage *= 0.7 (Strata generate.cpp; every constant is Strata's)
  oracle    the N pairs the trace itself uses most (an upper bound for any static placement; not achievable)

    python bench/moe_hit_sim.py TRACE --profile PROFILE.bin [--sizes 2200,3000,4000,5000,7000] [--skip-tokens 0]
    python bench/moe_hit_sim.py --selftest

Stdlib only; reads local files.
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

LINE = re.compile(r"^blk\.(\d+)\.ffn_gate_exps\.weight((?:\s+-?\d+)+)\s*$")


def read_trace(path: str) -> list[tuple[int, list[int]]]:
    out = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        m = LINE.match(line)
        if m:
            out.append((int(m.group(1)), [int(x) for x in m.group(2).split()]))
    return out


def read_profile(path: str) -> list[tuple[int, int]]:
    blob = Path(path).read_bytes()
    if blob[:4] != b"STRP":
        raise SystemExit(f"{path}: not an STRP profile")
    _ver, _nl, _ne, _slots, n = struct.unpack_from("<5I", blob, 4)
    return [struct.unpack_from("<HH", blob, 24 + 4 * i) for i in range(n)]


def sim_static(trace, resident: set) -> float:
    hits = total = 0
    for layer, ids in trace:
        for e in ids:
            total += 1
            hits += (layer, e) in resident
    return hits / total if total else 0.0


def sim_lru(trace, n: int, layers: int) -> float:
    per = max(1, n // layers)
    caches: dict[int, OrderedDict] = defaultdict(OrderedDict)
    hits = total = 0
    for layer, ids in trace:
        c = caches[layer]
        for e in ids:
            total += 1
            if e in c:
                hits += 1
                c.move_to_end(e)
            else:
                c[e] = True
                if len(c) > per:
                    c.popitem(last=False)
    return hits / total if total else 0.0


def sim_adaptive(trace, seed: list[tuple[int, int]], every: int = 4, swaps: int = 96, min_use: float = 2.0,
                 gain: float = 1.5, decay: float = 0.7) -> float:
    resident = set(seed)
    usage: dict[tuple[int, int], float] = defaultdict(float)
    hits = total = 0
    last_layer = None
    step = 0
    for layer, ids in trace:
        if last_layer is not None and layer < last_layer:     # a new token starts at a lower layer
            step += 1
            if step % every == 0:
                by_layer_res: dict[int, list] = defaultdict(list)
                for p in resident:
                    by_layer_res[p[0]].append(p)
                cands = sorted(((u, p) for p, u in usage.items() if p not in resident and u >= min_use), reverse=True)
                done = 0
                for u, p in cands:
                    if done >= swaps:
                        break
                    pool = by_layer_res.get(p[0]) or []
                    if not pool:
                        continue
                    victim = min(pool, key=lambda q: usage.get(q, 0.0))
                    if u >= gain * max(usage.get(victim, 0.0), 1e-9):
                        resident.discard(victim)
                        resident.add(p)
                        pool.remove(victim)
                        pool.append(p)
                        done += 1
                for p in list(usage):
                    usage[p] *= decay
        last_layer = layer
        for e in ids:
            total += 1
            hits += (layer, e) in resident
            usage[(layer, e)] += 1.0
    return hits / total if total else 0.0


def simulate(trace, profile, sizes, layers: int) -> dict:
    freq = Counter((layer, e) for layer, ids in trace for e in ids)
    oracle = [p for p, _ in freq.most_common()]
    out = {"tokens": sum(1 for layer, _ in trace if layer == trace[0][0]) if trace else 0,
           "reads": sum(len(ids) for _, ids in trace), "distinct_pairs": len(freq), "sizes": {}}
    for n in sizes:
        row = {"lru": round(sim_lru(trace, n, layers), 4), "oracle": round(sim_static(trace, set(oracle[:n])), 4)}
        if profile:
            row["profile"] = round(sim_static(trace, set(profile[:n])), 4)
            row["adaptive"] = round(sim_adaptive(trace, profile[:n]), 4)
        out["sizes"][n] = row
    return out


def selftest() -> int:
    bad = 0

    def check(ok, what):
        nonlocal bad
        print(("ok    " if ok else "FAIL  ") + what)
        bad += 0 if ok else 1
    # two layers, 4 experts, top-2: layer 0 always {0, 1}, layer 1 alternates {2,3} / {0,1}
    trace = []
    for t in range(40):
        trace.append((0, [0, 1]))
        trace.append((1, [2, 3] if t % 2 else [0, 1]))
    prof = [(0, 0), (0, 1), (1, 2), (1, 3), (1, 0), (1, 1)]
    r = simulate(trace, prof, [4, 6], 2)
    check(r["sizes"][6]["profile"] == 1.0 and r["sizes"][6]["oracle"] == 1.0, "every used pair resident: all hits")
    check(abs(r["sizes"][4]["profile"] - 0.75) < 1e-9, "the profile's top 4: layer 0 always, layer 1 half the time")
    check(r["sizes"][4]["oracle"] >= r["sizes"][4]["profile"], "the oracle bounds a static profile")
    check(0.0 < r["sizes"][4]["lru"] <= 1.0 and r["tokens"] == 40, "LRU runs; tokens counted")
    check(r["sizes"][4]["adaptive"] >= 0.75, "the adaptive tier never loses to its seed here")
    print("\nall passed" if not bad else f"\n{bad} FAILED")
    return 1 if bad else 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("trace", nargs="?")
    ap.add_argument("--profile")
    ap.add_argument("--sizes", default="2200,3000,4000,5000,7000")
    ap.add_argument("--layers", type=int, default=48)
    ap.add_argument("--skip-tokens", type=int, default=0, help="drop the first N tokens (warm-up)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.trace:
        ap.error("a trace is required")
    trace = read_trace(a.trace)
    if a.skip_tokens:
        trace = trace[a.skip_tokens * a.layers:]
    prof = read_profile(a.profile) if a.profile else []
    print(json.dumps(simulate(trace, prof, [int(x) for x in a.sizes.split(",")], a.layers), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
