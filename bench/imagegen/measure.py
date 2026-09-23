#!/usr/bin/env python
"""Measure Qwen-Image-2.1 in stable-diffusion.cpp on CUDA1, one config at a time.

    python bench/imagegen/measure.py --config a_te_cpu
    python bench/imagegen/measure.py --config b_offload --size 1024
    python bench/imagegen/measure.py --list

WHAT IT RECORDS, PER IMAGE (bench/imagegen/results.jsonl, one row each)

  wall_s            process start to exit (includes model load)
  load_s            sd-cli's own "loading ... completed" figure where printed
  encode_s          text-encoder time ("get_learned_condition completed")
  sample_s          denoising time ("sampling completed"), and sec/step
  decode_s          VAE decode time ("decode_first_stage completed")
  base_used_mib     CUDA1 used before start
  peak_used_mib     highest CUDA1 used seen while it ran (nvidia-smi, 0.25 s)
  min_free_mib      lowest CUDA1 free seen while it ran

SAFETY. The live stack's rootstock and Laya sit on CUDA1. The run is started
only if CUDA1 has at least --need MiB free, is killed the moment free memory
drops below --floor MiB (default 1024: AGENTS.md's ">= 1 GB free" rule), and a
kill is recorded as a row with killed=true, never as a result. CUDA0 -- the
live 5060 Ti -- is hidden from the process with CUDA_VISIBLE_DEVICES, so sd.cpp
cannot even enumerate it.

nvidia-smi numbers are PCI order; index 1 is the A4000. Inside the process the
A4000 is `cuda0`, because it is the only visible device.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SD = r"C:/Users/jwals/stable-diffusion.cpp/build/bin/sd-cli.exe"
M = r"C:/Users/jwals/textgen/user_data/models/qwen-image-2.1"
DIFF = f"{M}/qwen-image-2.1-Q5_K_M.gguf"
VAE = f"{M}/vae/qwen_image_2.1_vae_bf16.safetensors"
TE = os.environ.get("IMAGEGEN_TE", f"{M}/qwen3vl_8b_heretic-Q4_K_M.gguf")
GPU = "1"                         # nvidia-smi index (PCI order) of the A4000

PROMPTS = {
    "fox": ("A red fox sitting in a snowy birch forest at dawn, soft golden "
            "light, photorealistic, shallow depth of field"),
    "sign": ('A vintage enamel shop sign on a red brick wall that reads '
             '"YAMADORI BONSAI" in bold cream serif letters, with a small '
             'painted bonsai tree beside the words, late afternoon sun'),
    "crane": ("Flat vector icon of a folded paper crane, teal and orange, "
              "white background, clean minimal logo style"),
}

COMMON = ["--diffusion-model", DIFF, "--vae", VAE, "--llm", TE,
          "--cfg-scale", "6.0", "--sampling-method", "euler",
          "--steps", "20", "--diffusion-fa", "-v",
          # 8 of 24 threads: the box shut down thermally mid-benchmark.
          "-t", "8"]

CONFIGS = {
    # (a) text encoder on the CPU, denoiser + VAE resident on the GPU.
    "a_te_cpu": ["--auto-fit", "off", "--backend", "all=cuda0,te=cpu"],
    # (a) plus VAE tiling, if (a)'s decode is what peaks.
    "a_te_cpu_tiled": ["--auto-fit", "off", "--backend", "all=cuda0,te=cpu",
                       "--vae-tiling"],
    # (a) with small VAE tiles: the untiled wan_vae decode asked for 8.7 GB at
    # 1024x1024 and the default tile for 2.9 GB, with the denoiser's 5.1 GB
    # still resident (first a_te_cpu run, 2026-09-22 23:21).
    "a_te_cpu_tile16": ["--auto-fit", "off", "--backend", "all=cuda0,te=cpu",
                        "--vae-tiling", "--vae-tile-size", "16x16"],
    # (b) everything runs on the GPU; weights live in RAM and are staged in.
    # UNBOUNDED -- DO NOT RUN AGAIN. Its first run (2026-09-22 23:27) took
    # CUDA1 from 8,615 MiB free to 124 MiB free before the floor watcher
    # killed it: with no --max-vram, offload keeps compute replicas resident
    # up to the card's live free memory. Kept only so its row reads.
    "b_offload": ["--backend", "cuda0", "--offload-to-cpu"],
    # (b) with a managed budget. 6 GiB = 6,144 MiB for weights + runner
    # buffers; driver context is outside it (docs/performance.md).
    "b_offload_budget6": ["--backend", "cuda0", "--offload-to-cpu",
                          "--max-vram", "6"],
    # (a)+(b): encoder on the CPU, denoiser + VAE weights staged from RAM,
    # same 6 GiB budget.
    "ab_te_cpu_offload": ["--backend", "all=cuda0,te=cpu", "--offload-to-cpu",
                          "--max-vram", "6"],
}


TEMP_LIMIT = 88   # C: kill the run above this (the box shut down thermally
                  # at 23:42 on 2026-09-22 during this benchmark's session)


def smi() -> tuple[int, int]:
    used, free, _t = smi3()
    return used, free


def smi3() -> tuple[int, int, int]:
    out = subprocess.run(
        ["nvidia-smi", f"--id={GPU}",
         "--query-gpu=memory.used,memory.free,temperature.gpu",
         "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    used, free, temp = [int(x) for x in out.strip().split(",")]
    return used, free, temp


def timing(log: str, what: str, last: bool = True) -> float | None:
    """Seconds from sd.cpp's '<what> completed, taking 2.83s' (or '... ms')."""
    m = re.findall(re.escape(what) + r" completed, taking ([\d.]+)\s*(ms|s)\b", log)
    if not m:
        return None
    v, unit = m[-1] if last else m[0]
    return round(float(v) / (1000 if unit == "ms" else 1), 2)


def run_one(config: str, prompt_key: str, size: int, seed: int, need: int,
            floor: int, out_dir: str) -> dict:
    used0, free0 = smi()
    row = {"config": config, "prompt": prompt_key, "size": f"{size}x{size}",
           "seed": seed, "steps": 20, "te": os.path.basename(TE),
           "base_used_mib": used0, "base_free_mib": free0,
           "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if free0 < need:
        row.update(skipped=True, why=f"CUDA1 free {free0} MiB < need {need}")
        return row
    out = os.path.join(out_dir, f"{config}_{prompt_key}_{size}_s{seed}.png")
    cmd = [SD, *COMMON, *CONFIGS[config], "-p", PROMPTS[prompt_key],
           "-W", str(size), "-H", str(size), "-s", str(seed), "-o", out]
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID",
               CUDA_VISIBLE_DEVICES=GPU)
    peak = {"used": used0, "free": free0, "killed": False, "temp": 0,
            "why": ""}
    t0 = time.time()
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace")
    stop = threading.Event()

    def poll():
        while not stop.is_set():
            try:
                u, f, t = smi3()
            except Exception:                                    # noqa: BLE001
                continue
            peak["used"] = max(peak["used"], u)
            peak["free"] = min(peak["free"], f)
            peak["temp"] = max(peak["temp"], t)
            if (f < floor or t > TEMP_LIMIT) and p.poll() is None:
                peak["killed"] = True
                peak["why"] = f"free {f} MiB" if f < floor else f"{t} C"
                p.kill()
            stop.wait(0.25)

    th = threading.Thread(target=poll, daemon=True)
    th.start()
    log = p.stdout.read()
    p.wait()
    wall = time.time() - t0
    stop.set()
    th.join()
    with open(out + ".log", "w", encoding="utf-8") as f:
        f.write(log)
    sample = timing(log, "sampling")
    loads = re.findall(r"loading tensors completed, taking ([\d.]+)s", log)
    row.update({
        "wall_s": round(wall, 2), "rc": p.returncode,
        "killed": peak["killed"], "kill_why": peak["why"],
        "peak_temp_c": peak["temp"],
        "peak_used_mib": peak["used"], "min_free_mib": peak["free"],
        "delta_peak_mib": peak["used"] - used0,
        "encode_s": timing(log, "get_learned_condition"),
        "sample_s": sample,
        "decode_s": timing(log, "decode_first_stage"),
        "load_s": round(sum(float(x) for x in loads), 2) if loads else None,
        "vae_retiled": "retrying with spatial tiling" in log,
        "errors": [ln.strip()[:160] for ln in log.splitlines()
                   if "[ERROR" in ln][:4],
        "image": os.path.relpath(out, os.path.join(HERE, "..", "..")).replace(os.sep, "/")
        if os.path.exists(out) else None,
    })
    if row["sample_s"]:
        row["sec_per_step"] = round(row["sample_s"] / 20, 2)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=sorted(CONFIGS))
    ap.add_argument("--prompts", default="fox,sign,crane")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--need", type=int, default=6000,
                    help="MiB CUDA1 must have free before a run starts")
    ap.add_argument("--floor", type=int, default=1024,
                    help="kill the run if CUDA1 free drops below this")
    ap.add_argument("--out", default=os.path.join(HERE, "samples"))
    ap.add_argument("--results", default=os.path.join(HERE, "results.jsonl"))
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list or not a.config:
        for k, v in CONFIGS.items():
            print(f"  {k:<20} {' '.join(v)}")
        return 0
    os.makedirs(a.out, exist_ok=True)
    for pk in a.prompts.split(","):
        row = run_one(a.config, pk, a.size, a.seed, a.need, a.floor, a.out)
        print(json.dumps(row), flush=True)
        with open(a.results, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        if row.get("killed") or row.get("skipped"):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
