"""Build an expert profile (Strata's STRP format) for llama-upstream-moe's expert cache.

The profile ranks every (layer, expert) pair of a MoE model by how often the router picks it; with
LLAMA_MOE_CACHE_PROFILE=<file> the cache (engines/patches/llama-upstream-moe/0004) fills its VRAM slots with the
pairs in rank order before the first decode, so the ranking decides which experts start resident, and each layer
gets as many slots as the ranking gives it.

A port of Strata's tools/make_profile.py (github.com/Niko1221/Strata @ d551edf42c157f1c8c20f07f68c5dfe506b1edda,
credit Niko1221; no licence file, used with the operator's authorisation 2026-09-28), generalised from Flash-Next's
48 x 512 to any MoE model, and reading llama.cpp's own routing traces as well as Strata's:

  - GGML_MOE_LOG=<file> (0003's diagnostic in llama-upstream-moe): one text line per token and layer,
    "blk.<layer>.ffn_gate_exps.weight <id> <id> ...", appended by every CPU MUL_MAT_ID of the gate experts;
  - Strata's --dump-routing binary: records of int32 layer, int32 k, k int32 ids, k float weights.

The order is Strata's: the base profile's ranking first (default none), then the pairs the traces used, most
frequent first (ties by (layer, expert)), then every pair still missing, interleaved across the layers.

    python scripts/moe_profile.py TRACE... --layers 48 --experts 512 --out profile.bin [--base BASE.bin]
    python scripts/moe_profile.py --show profile.bin

Stdlib only; reads and writes local files, nothing else.
"""
from __future__ import annotations

import argparse
import re
import struct
import sys
from collections import defaultdict
from pathlib import Path

MAGIC, VERSION = b"STRP", 1
LINE = re.compile(r"^blk\.(\d+)\.ffn_gate_exps\.weight((?:\s+-?\d+)+)\s*$")


def read_profile(path: str) -> tuple[int, int, list[tuple[int, int]]]:
    blob = Path(path).read_bytes()
    if blob[:4] != MAGIC:
        raise SystemExit(f"{path}: not an STRP profile")
    _ver, nl, ne, _slots, n = struct.unpack_from("<5I", blob, 4)
    return nl, ne, [struct.unpack_from("<HH", blob, 24 + 4 * i) for i in range(n)]


def read_trace(path: str, n_layers: int, n_expert: int) -> dict[tuple[int, int], int]:
    """GGML_MOE_LOG text, or Strata's binary --dump-routing records."""
    blob = Path(path).read_bytes()
    freq: dict[tuple[int, int], int] = defaultdict(int)
    if blob[:4] == b"blk.":
        for line in blob.decode("utf-8", "replace").splitlines():
            m = LINE.match(line)
            if not m:
                continue
            layer = int(m.group(1))
            for e in m.group(2).split():
                e = int(e)
                if 0 <= layer < n_layers and 0 <= e < n_expert:
                    freq[(layer, e)] += 1
        return freq
    off = 0
    while off + 8 <= len(blob):
        layer, k = struct.unpack_from("<ii", blob, off)
        off += 8
        for e in struct.unpack_from("<%di" % k, blob, off):
            if 0 <= layer < n_layers and 0 <= e < n_expert:
                freq[(layer, e)] += 1
        off += 8 * k   # the ids and the weights
    return freq


def write_profile(path: str, n_layers: int, n_expert: int, ranked: list[tuple[int, int]]) -> None:
    # Strata's layout (make_profile.py write_profile): header, the ranked pairs, then the slot of every pair
    table = [[-1] * n_expert for _ in range(n_layers)]
    for slot, (layer, e) in enumerate(ranked):
        table[layer][e] = slot
    with open(path, "wb") as f:
        f.write(MAGIC + struct.pack("<5I", VERSION, n_layers, n_expert, len(ranked), len(ranked)))
        for layer, e in ranked:
            f.write(struct.pack("<HH", layer, e))
        for layer in range(n_layers):
            f.write(struct.pack("<%di" % n_expert, *table[layer]))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("traces", nargs="*", help="GGML_MOE_LOG files or Strata --dump-routing files")
    ap.add_argument("--layers", type=int, help="the model's layer count (hparams n_layer)")
    ap.add_argument("--experts", type=int, help="experts per layer")
    ap.add_argument("--base", help="a profile whose ranking is kept first")
    ap.add_argument("--out")
    ap.add_argument("--show", metavar="PROFILE", help="print a profile's shape and per-layer share of its head")
    a = ap.parse_args(argv)

    if a.show:
        nl, ne, ranked = read_profile(a.show)
        print(f"{a.show}: {nl} layers x {ne} experts, {len(ranked)} ranked pairs")
        for head in (1000, 4000, 8000):
            if head <= len(ranked):
                per = defaultdict(int)
                for layer, _ in ranked[:head]:
                    per[layer] += 1
                print(f"  first {head}: {min(per.values(), default=0)}-{max(per.values(), default=0)} per layer "
                      f"over {len(per)} layers")
        return 0

    nl, ne = a.layers, a.experts
    base: list[tuple[int, int]] = []
    if a.base:
        bl, be, base = read_profile(a.base)
        nl, ne = nl or bl, ne or be
        if (bl, be) != (nl, ne):
            raise SystemExit(f"--base is {bl}x{be}, not {nl}x{ne}")
    if not nl or not ne or not a.out:
        raise SystemExit("--layers, --experts (or --base) and --out are required")

    ranked, seen = [], set()

    def take(pairs) -> None:
        for p in pairs:
            p = (int(p[0]), int(p[1]))
            if p not in seen:
                seen.add(p)
                ranked.append(p)

    take(base)
    n_base = len(ranked)
    freq: dict[tuple[int, int], int] = defaultdict(int)
    for t in a.traces:
        for p, c in read_trace(t, nl, ne).items():
            freq[p] += c
    take(p for p, _ in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0])))
    n_trace = len(ranked) - n_base
    take((layer, e) for e in range(ne) for layer in range(nl))   # the rest, across the layers
    write_profile(a.out, nl, ne, ranked)
    if read_profile(a.out)[2] != ranked:
        raise SystemExit("the profile did not survive the round trip")
    print(f"wrote {a.out}: {len(ranked)} ranked pairs ({n_base} from the base, {n_trace} from the traces, "
          f"{len(ranked) - n_base - n_trace} filled in)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
