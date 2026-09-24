#!/usr/bin/env python
"""Per-image pixel diff of a VAE variant against the baseline, OUTSIDE the
baseline's white blocks (SELF-IMPROVEMENT-LOG #25).

    python bench/imagegen/diff_variants.py BASELINE_DIR VARIANT_DIR

Pairs PNGs by file name (the tag). Masks every white-block cluster that
bench/imagegen/white_blocks.py finds in EITHER image, grown by MARGIN px,
and reports over the rest: mean |diff| (0-255, all channels), the share of
pixels whose largest channel diff exceeds 8 and 32, the max diff, and PSNR.
Identical outside the blocks -> mean 0.000, PSNR inf.
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from white_blocks import white_blocks  # noqa: E402

MARGIN = 8


def diff(a_path: str, b_path: str) -> dict:
    a = np.asarray(Image.open(a_path).convert("RGB")).astype(np.int16)
    b = np.asarray(Image.open(b_path).convert("RGB")).astype(np.int16)
    keep = np.ones(a.shape[:2], bool)
    for p in (a_path, b_path):
        for c in white_blocks(p)["clusters"]:
            keep[max(0, c["y"] - MARGIN):c["y"] + c["h"] + MARGIN,
                 max(0, c["x"] - MARGIN):c["x"] + c["w"] + MARGIN] = False
    d = np.abs(a - b)[keep]
    dmax = d.max(axis=1)
    mse = float((d.astype(np.float64) ** 2).mean())
    return {"mean_abs": round(float(d.mean()), 3),
            "px_gt8": round(float((dmax > 8).mean()), 5),
            "px_gt32": round(float((dmax > 32).mean()), 5),
            "max": int(d.max()),
            "psnr": round(10 * math.log10(255 ** 2 / mse), 2) if mse else math.inf,
            "masked_px": int((~keep).sum())}


def main(argv) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    base, var = argv
    for bp in sorted(glob.glob(os.path.join(base, "*.png"))):
        vp = os.path.join(var, os.path.basename(bp))
        if os.path.exists(vp):
            print(json.dumps({"tag": os.path.basename(bp)[:-4], **diff(bp, vp)}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
