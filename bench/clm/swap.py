#!/usr/bin/env python
"""On-demand swap of the CLM encoder on the A4000: how long a swap-in takes
from a COLD file (not in the OS page cache) and from a WARM one, and what a
call costs once loaded. (docs/CLM.md "Serving: on-demand swap".)

    python bench/clm/swap.py [--gguf <models>/Qwen3-8B-Q8_0.gguf]

COLD is made, not assumed: the GGUF is copied with `robocopy /J` (unbuffered
I/O, so the copy's pages never enter the file cache) to a scratch directory
on the same drive, and that copy is loaded first. The second load of the same
copy is WARM (its pages were just read). Each load is config.yaml's
clm-encoder flags (bench/clm/standalone.serving, under gpu_room's room lock);
swap-in = process start to /health 200 plus the first request (the server's
own warm-up), per-call = distinct inputs at fixed lengths.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "mcp"))
sys.path.insert(0, HERE)

MODELS = "C:/Users/jwals/textgen/user_data/models"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default=os.path.join(MODELS, "Qwen3-8B-Q8_0.gguf"))
    ap.add_argument("--lengths", default="40,300,800,1500")
    a = ap.parse_args(argv)
    import clm
    import standalone
    scratch = tempfile.mkdtemp(prefix="clm_cold_", dir=os.path.dirname(a.gguf))
    src_dir, name = os.path.split(a.gguf)
    t = time.time()
    r = subprocess.run(["robocopy", src_dir, scratch, name, "/J", "/NP",
                        "/NFL", "/NDL", "/NJH", "/NJS"], capture_output=True)
    copy_s = round(time.time() - t, 1)
    if r.returncode >= 8:
        print("robocopy failed", r.returncode, r.stdout[-400:])
        return 1
    path = os.path.join(scratch, name)
    tok = clm.tokenizer()
    lengths = [int(x) for x in a.lengths.split(",") if x]
    out = {"gguf": name, "copy_s_unbuffered": copy_s, "loads": []}
    try:
        for label in ("cold", "warm"):
            with standalone.serving(path, 10993, tag=f"swap_{label}") as (base, rec):
                enc = clm.HttpEncoder(url=base + "/v1/embeddings", model=None,
                                      gpu_room_model=None)
                warm = standalone.latency_inputs(tok, 16, 6)[5]
                t = time.perf_counter()
                enc.embed_ids([warm])
                first_ms = (time.perf_counter() - t) * 1000
                lat = {}
                for n in lengths:
                    ts = []
                    for x in standalone.latency_inputs(tok, n, 3):
                        t = time.perf_counter()
                        enc.embed_ids([x])
                        ts.append(round((time.perf_counter() - t) * 1000, 1))
                    lat[str(n)] = {"median": sorted(ts)[1], "all": ts}
                out["loads"].append({
                    "label": label, "load_s": rec["load_s"],
                    "first_call_ms": round(first_ms, 1),
                    "swap_in_s": round(rec["load_s"] + first_ms / 1000, 2),
                    "vram_idle_mib": rec["vram_idle_mib"],
                    "per_call_ms": lat})
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "swap.json"), "w",
              encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
