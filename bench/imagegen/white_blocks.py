"""Count white-block artifacts in generated images (SELF-IMPROVEMENT-LOG #25).

The artifact: solid (255, 255, 255) blocks aligned to the 8-px latent grid,
at recurring positions (x ~496-520, y ~256-280 and ~512-536 at 1024x1024),
seen in both image models.

A tile is one 8x8 latent cell. It is CLIPPED when >= 95% of its pixels are
exactly (255, 255, 255). A clipped tile counts as a white block when:

  1. its 8-connected cluster of clipped tiles is small (<= MAX_CLUSTER tiles):
     a large clipped area is a real white background, not a block; and
  2. the 1-px ring just outside the cluster is mostly NOT clipped
     (exact-255 fraction < RING_MAX): the clipping stops at the tile grid.
     A clipped background continues past it (e.g. the white-background
     image c0b057cd38 has lone clipped tiles at the right edge with ring
     0.96-1.00; the artifacts measured 0.18-0.81); and
  3. the ring is not a mostly-clipped white background: skipped when the
     ring is >= RING_NEAR_MAX near-white (>= 248 in every channel) AND
     >= RING_NEAR_CLIPPED exactly 255. ADDED AFTER a false positive: the
     white-background control c0b057cd38, decoded untiled, has a clipped 8x8
     tile at its right edge (1016, 448) in 253-255 background (ring 0.88
     clipped, 1.00 near-white) -- the brightest part of a real background.
     Every artifact seen had ring near-white == ring clipped (0.18-0.81):
     the only near-white pixels around a block are its own 255 spill.
     A block on UNclipped near-white paper (ring 0.00 clipped) still counts.

Why exact 255 and not the ">= 248" near-white test: generated white areas
(snow, paper, flowchart backgrounds, clouds) sit at 248-254 and are almost
never exactly 255 (exact fraction ~0.00-0.17 per near-white tile on the 33
images of 2026-09-24), while every artifact tile was 255 on every pixel.
A ">= 248" rule counted snow in dc3f702405 and flowchart paper in
9fe2d7a853 as blocks.

Known limit: a clipped (exactly 255) highlight that covers whole 8x8 tiles
and is not part of a large clipped area DOES count: the synthetic unaligned
disc of radius 20 in --selftest scores 11. None of the 37 production images
or the bench samples of 2026-09-24 has one (every count there sits on a
tile seam), but compare variants on the same seeds, not absolute counts
across different content.

Usage:
  python bench/imagegen/white_blocks.py PNG_OR_DIR ...   one line per image
  python bench/imagegen/white_blocks.py --json PNG ...   JSON lines
  python bench/imagegen/white_blocks.py --selftest       synthetic checks
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

TILE = 8
TILE_CLIPPED = 0.95   # fraction of a tile's pixels that are exactly 255
MAX_CLUSTER = 16      # tiles; a bigger clipped cluster is a white region
RING_MAX = 0.9        # exact-255 fraction of the ring; above = region continues
RING_NEAR_MAX = 0.9   # near-white (>= 248) fraction of the ring, and
RING_NEAR_CLIPPED = 0.5  # its exact-255 fraction: both above = the tile is the
                         # brightest part of an already-clipped white background
NEAR = 248


def _rgb(img) -> np.ndarray:
    if isinstance(img, np.ndarray):
        return img[..., :3]
    return np.asarray(Image.open(img).convert("RGB"))


def white_blocks(img) -> dict:
    """{blocks, clusters:[{x, y, w, h, tiles, ring}], clipped_px}."""
    a = _rgb(img)
    h, w = a.shape[:2]
    exact = (a == 255).all(axis=2)
    hh, ww = h // TILE * TILE, w // TILE * TILE
    frac = exact[:hh, :ww].reshape(hh // TILE, TILE, ww // TILE, TILE).mean(axis=(1, 3))
    clipped = frac >= TILE_CLIPPED
    lab, n = ndimage.label(clipped, structure=np.ones((3, 3)))
    clusters, blocks = [], 0
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        if len(ys) > MAX_CLUSTER:
            continue
        m = np.zeros((h, w), bool)
        for ty, tx in zip(ys, xs):
            m[ty * TILE:(ty + 1) * TILE, tx * TILE:(tx + 1) * TILE] = True
        ring = ndimage.binary_dilation(m, np.ones((3, 3))) & ~m
        ring_frac = float(exact[ring].mean()) if ring.any() else 1.0
        ring_near = float((a[ring].min(axis=1) >= NEAR).mean()) if ring.any() else 1.0
        if ring_frac >= RING_MAX or (ring_near >= RING_NEAR_MAX
                                         and ring_frac >= RING_NEAR_CLIPPED):
            continue
        blocks += len(ys)
        clusters.append({"x": int(xs.min() * TILE), "y": int(ys.min() * TILE),
                         "w": int((xs.max() - xs.min() + 1) * TILE),
                         "h": int((ys.max() - ys.min() + 1) * TILE),
                         "tiles": int(len(ys)), "ring": round(ring_frac, 2),
                         "ring_near": round(ring_near, 2)})
    return {"blocks": blocks, "clusters": clusters, "clipped_px": int(exact.sum())}


def _paths(args):
    for a in args:
        if os.path.isdir(a):
            yield from sorted(glob.glob(os.path.join(a, "*.png")))
        else:
            yield a


def selftest() -> int:
    rng = np.random.default_rng(0)
    fails = 0

    def check(name, img, want):
        nonlocal fails
        got = white_blocks(img)["blocks"]
        ok = got == want if isinstance(want, int) else want(got)
        fails += not ok
        print(f"{'ok ' if ok else 'FAIL'} {name}: blocks={got}")

    base = rng.integers(20, 200, (1024, 1024, 3), dtype=np.uint8)
    check("noise, no white", base, 0)
    art = base.copy()
    art[256:280, 496:512] = 255            # 2x3 tiles, the observed shape
    check("aligned 16x24 block (6 tiles)", art, 6)
    wbg = np.full((1024, 1024, 3), 255, np.uint8)
    wbg[300:700, 300:700] = (40, 90, 200)   # pure-255 background + a shape
    check("exact-255 white background + shape", wbg, 0)
    nearw = rng.integers(248, 255, (1024, 1024, 3), dtype=np.uint8)
    check("near-white (248-254) background", nearw, 0)
    blk = nearw.copy()
    blk[512:520, 512:520] = 255             # block on near-white paper
    check("one aligned block on near-white paper", blk, 1)
    disc = base.copy()
    yy, xx = np.mgrid[:1024, :1024]
    disc[(yy - 403) ** 2 + (xx - 611) ** 2 <= 20 ** 2] = 255  # unaligned clipped disc
    print(f"LIMIT unaligned clipped disc r=20 (a highlight): "
          f"blocks={white_blocks(disc)['blocks']} (counted; see the docstring)")
    return fails


def main(argv) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if argv[0] == "--selftest":
        return 1 if selftest() else 0
    as_json = argv[0] == "--json"
    for p in _paths(argv[1:] if as_json else argv):
        r = white_blocks(p)
        if as_json:
            print(json.dumps({"path": p, **r}))
        else:
            where = " ".join(f"({c['x']},{c['y']} {c['w']}x{c['h']})" for c in r["clusters"])
            print(f"{r['blocks']:3d}  {os.path.basename(p)}  {where}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
