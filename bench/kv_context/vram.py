#!/usr/bin/env python
"""VRAM arithmetic for the main model on the RTX 5060 Ti: q8_0 K/V today vs
q4_0 K/V at a larger -c. Offline: standard library only, sends nothing.

    python bench/kv_context/vram.py            # the table in docs/CONTEXT-EXPANSION.md
    python bench/kv_context/vram.py --json     # the same rows, machine-readable

EVERY INPUT NAMES ITS SOURCE AND ITS n. Nothing here is a measurement of a
q4_0 configuration on the 5060 Ti -- there is none yet. The projections are
anchored on ONE live reading of today's q8_0 server and move from there by
per-token costs measured elsewhere. bench/kv_context/run.py (phase FIT) is what
replaces them with measurements.

THE MODEL (all per token of -c, KiB)

    main KV     16 full-attention layers (qwen35.full_attention_interval 4 of
                64 blocks) x 4 KV heads x 256 dims, for K and for V.
                  q8_0  34 bytes / 32 values  -> 17.0 KiB each, 34.0 both
                  q4_0  18 bytes / 32 values  ->  9.0 KiB each, 18.0 both
                34.0 is MEASURED (docs/MTP-STAGING.md s.9: 544.00 MiB at 16,384
                cells, A4000, this build). 18.0 is the same geometry at q4_0's
                block size; it agrees with sudoingX's RTX 3060 serve sweep,
                whose no-MTP q4_0/q4_0 slope is 23.0 KiB/token total
                (8,754 MiB @131072 -> 10,226 @196608 -> 11,698 @262144,
                sweeps/rtx3060.md @ eb52d9d7) = 18.0 KV + 5.0 compute.
    compute     main compute buffer growth, ~5 KiB (MTP-STAGING s.9: 170.28 MiB
                @16K -> 210.28 @24K, np 1, A4000; extrapolated 8K -> 262K).
    MTP         draft KV (f16, 1 layer: 4 KiB) + draft compute (~1 KiB),
                MTP-STAGING s.9, extrapolated likewise.
    margin      +3.6 KiB: the one MTP-on q4_0 slope measured anywhere is
                31.6 KiB/token (sudoingX RTX 3060, lean file + MTP n-max 1,
                np 1: 9,956 MiB @131072 -> 11,976 @196608), against the
                component sum of 28.0. The "conservative" column adds the
                unexplained 3.6 to every token past the anchor.

THE ANCHOR: nvidia-smi on 2026-09-23 ~22:30 EDT, 5060 Ti (UUID
GPU-de660e90-...), the live `bonsai` process (-c 163840, q8_0/q8_0, MTP n-max
1, -np auto = 4 slots, unified KV) and nothing else on the card:
14,340 MiB used, 1,711 free, utilisation 78%. n = 1 reading. mcp/vitals.py
independently records "the MTP build at -c 163840 idles at ~1.7 GB free".
used + free = 16,051 MiB (nvidia-smi reports 16,311 total; the other 260 are
not allocatable and are left out of every free figure here).

CROSS-CHECK (not an input): MTP-STAGING s.9 predicted the MTP file at
-c 163840 from the official build's warm 147,456 measurement (12,122 MiB):
12,122 + 16,384 x 39 KiB + 1,855 = 14,601 MiB. The live reading is 14,340,
261 MiB under the prediction -- the display has since moved to the iGPU.

PEAK. config.yaml records a worst-case stress (main + helper shares filled)
adding ~194 MiB over idle at 172,032; docs/CONSTRAINTS.md records +37 MiB
(13,868 -> 13,905) at 163,840. The table subtracts the larger, 194.
"""
from __future__ import annotations

import argparse
import json

CARD_MIB = 16_051            # used + free on the 5060 Ti (anchor reading)
ANCHOR = {"c": 163_840, "k": "q8_0", "v": "q8_0", "mtp": True, "slots": 4,
          "used_mib": 14_340, "free_mib": 1_711,
          "source": "nvidia-smi 2026-09-23 ~22:30 EDT, live bonsai, n=1, util 78%"}

# KiB per token per tensor (K or V) across the 16 attention layers.
ELEMS_PER_TOKEN = 16 * 4 * 256          # layers x kv heads x head dim
BYTES_PER_32 = {"f16": 64, "q8_0": 34, "q4_0": 18}
COMPUTE_KIB = 5.0
MTP_KIB = 5.0                            # draft KV f16 4 + draft compute 1
EMPIRICAL_MARGIN_KIB = 31.6 - (18.0 + COMPUTE_KIB + MTP_KIB)   # = 3.6
RECURRENT_MTP_MIB_PER_SLOT = 299.25      # MTP-STAGING s.9, with MTP
PEAK_ADDER_MIB = 194                     # config.yaml, 172032 stress

FLOORS = {"documented 3 GB (config.yaml / KNOWN-ISSUES)": 3_072,
          "2.0 GB rule (config.yaml, 2026-09-22)": 2_048,
          "dashboard red, TIGHT_MIB (mcp/vitals.py, 2026-09-23)": 1_280,
          "HANDOFF run-queue item 4 (>= 1 GB at peak)": 1_024}


def kv_kib(k: str, v: str) -> float:
    """Main KV cost per token, KiB, for a K and a V cache type."""
    return ELEMS_PER_TOKEN * (BYTES_PER_32[k] + BYTES_PER_32[v]) / 32 / 1024


def slope_kib(k: str, v: str, conservative: bool) -> float:
    """Total VRAM per token of -c with the MTP head on."""
    s = kv_kib(k, v) + COMPUTE_KIB + MTP_KIB
    return s + (EMPIRICAL_MARGIN_KIB if conservative else 0.0)


def project(c: int, k: str = "q4_0", v: str = "q4_0", slots: int = 4,
            conservative: bool = False) -> dict:
    """Projected used/free MiB at -c `c` from the live anchor.

    Step 1 swaps the anchor's KV type at the anchor's own -c (KV only; the
    compute and MTP terms are type-independent in this model). Step 2 moves
    -c by the full per-token slope. Step 3 changes the slot count (each slot
    with MTP carries 299.25 MiB of recurrent state)."""
    a = ANCHOR
    used = a["used_mib"]
    used -= a["c"] * (kv_kib(a["k"], a["v"]) - kv_kib(k, v)) / 1024
    used += (c - a["c"]) * slope_kib(k, v, conservative) / 1024
    used += (slots - a["slots"]) * RECURRENT_MTP_MIB_PER_SLOT
    free = CARD_MIB - used
    return {"c": c, "k": k, "v": v, "slots": slots,
            "model": "conservative" if conservative else "component",
            "used_mib": round(used), "free_idle_mib": round(free),
            "free_peak_mib": round(free - PEAK_ADDER_MIB)}


def largest_c(floor_mib: int, k: str = "q4_0", v: str = "q4_0", slots: int = 4,
              conservative: bool = True, step: int = 8192,
              native: int = 262_144) -> int | None:
    """Largest multiple of `step` (<= native) whose projected PEAK free is at
    or above `floor_mib`, or None."""
    best = None
    c = step
    while c <= native:
        if project(c, k, v, slots, conservative)["free_peak_mib"] >= floor_mib:
            best = c
        c += step
    return best


ROWS = [  # (label, c, k, v, slots)
    ("today (anchor, measured)", 163_840, "q8_0", "q8_0", 4),
    ("q4_0/q4_0 at today's -c", 163_840, "q4_0", "q4_0", 4),
    ("candidate", 196_608, "q4_0", "q4_0", 4),
    ("between", 229_376, "q4_0", "q4_0", 4),
    ("candidate, native window", 262_144, "q4_0", "q4_0", 4),
    ("native window, -np 3", 262_144, "q4_0", "q4_0", 3),
    ("native window, -np 2", 262_144, "q4_0", "q4_0", 2),
    ("q4_0 K / q8_0 V (not buildable here, see doc)", 196_608, "q4_0", "q8_0", 4),
    ("q4_0 K / q8_0 V (not buildable here, see doc)", 262_144, "q4_0", "q8_0", 4),
]


def table() -> list[dict]:
    out = []
    for label, c, k, v, slots in ROWS:
        lo = project(c, k, v, slots, conservative=False)
        hi = project(c, k, v, slots, conservative=True)
        out.append({"label": label, "c": c, "k": k, "v": v, "slots": slots,
                     "used_mib": [lo["used_mib"], hi["used_mib"]],
                     "free_idle_mib": [lo["free_idle_mib"], hi["free_idle_mib"]],
                     "free_peak_mib": [lo["free_peak_mib"], hi["free_peak_mib"]]})
    return out


def render() -> str:
    lines = [f"  anchor: {ANCHOR['used_mib']:,} MiB used / {ANCHOR['free_mib']:,} free "
             f"at -c {ANCHOR['c']:,} q8_0/q8_0, MTP, 4 slots ({ANCHOR['source']})",
             f"  per-token slope with MTP: q8_0/q8_0 {slope_kib('q8_0', 'q8_0', False):.1f}, "
             f"q4_0/q4_0 {slope_kib('q4_0', 'q4_0', False):.1f} "
             f"(conservative {slope_kib('q4_0', 'q4_0', True):.1f}) KiB",
             "",
             f"  {'row':<46}{'-c':>9} {'K/V':>10} {'np':>3} "
             f"{'used MiB':>14} {'free idle':>13} {'free peak':>13}"]
    for r in table():
        def pair(x):
            return f"{x[0]:,}" if x[0] == x[1] else f"{x[0]:,}..{x[1]:,}"
        lines.append(f"  {r['label']:<46}{r['c']:>9,} {r['k'][:2] + '/' + r['v'][:2]:>10} "
                     f"{r['slots']:>3} {pair(sorted(r['used_mib'])):>14} "
                     f"{pair(sorted(r['free_idle_mib'])):>13} "
                     f"{pair(sorted(r['free_peak_mib'])):>13}")
    lines.append("")
    lines.append("  largest -c (multiple of 8192) whose projected PEAK free clears each floor,")
    lines.append("  q4_0/q4_0, 4 slots, component .. conservative slope:")
    for name, f in FLOORS.items():
        a = largest_c(f, conservative=False)
        b = largest_c(f, conservative=True)
        lines.append(f"    {name:<52} {f:>5} MiB   {b or 0:>7,} .. {a or 0:,}")
    return "\n".join(lines)


def _selftest() -> int:
    ok = 0
    checks = [
        (abs(kv_kib("q8_0", "q8_0") - 34.0) < 1e-9, "q8_0 K+V is the measured 34.0 KiB"),
        (abs(kv_kib("q4_0", "q4_0") - 18.0) < 1e-9, "q4_0 K+V is 18.0 KiB"),
        (abs(kv_kib("f16", "f16") / 16 * 1 - 4.0) < 1e-9,
         "one f16 layer is the measured 4 KiB draft KV"),
        (project(163_840, "q8_0", "q8_0")["used_mib"] == ANCHOR["used_mib"],
         "the anchor row reproduces the anchor"),
        (abs(slope_kib("q4_0", "q4_0", True) - 31.6) < 1e-9,
         "conservative q4 slope is the 3060 MTP measurement"),
        # 196,608 sits ON the 3 GB floor: 3,066 MiB at peak on the conservative
        # slope (6 under), 3,181 on the component slope.
        (largest_c(3_072) == 188_416, "3 GB floor, conservative slope -> 188,416"),
        (largest_c(3_072, conservative=False) == 196_608,
         "3 GB floor, component slope -> 196,608"),
        (largest_c(1_024) == 262_144, "1 GB floor -> the native 262,144 on both slopes"),
    ]
    for cond, what in checks:
        print(("  ok    " if cond else "  FAIL  ") + what)
        ok += bool(cond)
    print(f"  {ok}/{len(checks)} checks passed")
    return 0 if ok == len(checks) else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(_selftest())
    print(json.dumps(table(), indent=1) if a.json else render())
