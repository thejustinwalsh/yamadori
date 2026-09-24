#!/usr/bin/env python
"""Turbo vs base on the 8 PartiPrompts of bench/imagegen/parti-20260923.

    python bench/imagegen/compare_turbo.py                  # both arms, 8 prompts
    python bench/imagegen/compare_turbo.py --arms turbo     # turbo only
    python bench/imagegen/compare_turbo.py --prompts 0,5    # a subset
    python bench/imagegen/compare_turbo.py --list           # print the plan, run nothing

READY TO RUN, NOT RUN. It waits for the benchmarks on the stack to end: it is
a CUDA1 consumer for ~20 minutes (base ~8 x 110 s, turbo ~8 x 25 s by the
n=1 smoke test), and AGENTS.md rule 4 is one GPU consumer at a time. Queue it
through bench/queue_runner.py or run it on an idle card.

WHAT IS COMPARED

The same 8 prompts, seeds and size as the base run of 2026-09-23
(parti-20260923/manifest.json). That run went through the live Images API
with no seed, so each seed was chosen by mcp/images.py; it is recovered from
index/media/<sha>.json, whose sha is the sha256 of the saved PNG. SEEDS below
is that lookup, done once on 2026-09-23 and checked again at run time.

  base   qwen-image-2.1-Q5_K_M.gguf, 20 steps, cfg 6.0, euler -- the
         config.yaml `imagegen` flags.
  turbo  qwen_image_2.1_turbo_Q5_K_M.gguf, 4 steps, cfg 1.0, euler,
         --extra-sample-args base_shift=0.5,max_shift=0.69355 -- the
         config.yaml `imagegen-turbo` flags (docs/IMAGEGEN-TURBO.md §2).

Both: Heretic Q4_K_M encoder, bf16 VAE, --backend cuda0 --offload-to-cpu
--max-vram 6 --diffusion-fa -t 8, 1024x1024.

The base arm is RE-RUN here, through sd-cli, rather than read from the old
manifest: the old times are sd-server through the proxy, and a comparison
needs one harness on one day. Whether sd-cli reproduces the saved base PNG
byte for byte is recorded per row (`same_as_saved`); it is a finding either
way, not an assumption.

PER ROW (results.jsonl in the output folder)

  wall_s, encode_s, sample_s, sec_per_step, decode_s, load_s, base_used_mib,
  peak_used_mib, delta_peak_mib, min_free_mib, peak_temp_c, killed, rc.

Then one side-by-side PNG per prompt (base | turbo) and a contact sheet of all
eight. Quality is judged by eye from those (docs/IMAGEGEN-TURBO.md §6: no
automatic metric exists, and Laya is not a quality judge).

SAFETY (as measure.py)

  - The A4000 is pinned by UUID; sd-cli cannot see the 5060 Ti.
  - A run starts only if CUDA1 has --need MiB free (default 7000).
  - It is killed the moment CUDA1 free drops below --floor (1024 MiB, the
    >= 1 GB rule) or the card passes 88 C. A kill is a row with killed=true,
    never a result, and it stops the script.
  - It refuses to start while llama-swap has an image model loaded: that
    would be two image consumers on one card.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PARTI = os.path.join(HERE, "parti-20260923")

SD = r"C:/Users/jwals/stable-diffusion.cpp/build/bin/sd-cli.exe"
M = r"C:/Users/jwals/textgen/user_data/models/qwen-image-2.1"
VAE = f"{M}/vae/qwen_image_2.1_vae_bf16.safetensors"
TE = f"{M}/qwen3vl_8b_heretic-Q4_K_M.gguf"
A4000 = "GPU-43e37d0c-4104-9056-2552-6109d4d3382c"
SWAP = "http://127.0.0.1:11434"

SHARED = ["--vae", VAE, "--llm", TE,
          "--backend", "cuda0", "--offload-to-cpu", "--max-vram", "6",
          "--diffusion-fa", "--sampling-method", "euler", "-t", "8", "-v"]

ARMS = {
    "base": {"diffusion": f"{M}/qwen-image-2.1-Q5_K_M.gguf", "steps": 20,
             "extra": ["--cfg-scale", "6.0"]},
    "turbo": {"diffusion": f"{M}/qwen_image_2.1_turbo_Q5_K_M.gguf", "steps": 4,
              "extra": ["--cfg-scale", "1.0", "--extra-sample-args",
                        "base_shift=0.5,max_shift=0.69355"]},
}

# Seeds of the 2026-09-23 base images, from index/media/<sha256(png)>.json.
SEEDS = {0: 2070389620, 1: 236760194, 2: 572182185, 3: 1497176387,
         4: 1947947581, 5: 1801139049, 6: 1476881344, 7: 48442099}

TEMP_LIMIT = 88


def smi() -> tuple[int, int, int]:
    out = subprocess.run(
        ["nvidia-smi", f"--id={A4000}",
         "--query-gpu=memory.used,memory.free,temperature.gpu",
         "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    used, free, temp = [int(x) for x in out.strip().split(",")]
    return used, free, temp


def timing(log: str, what: str) -> float | None:
    m = re.findall(re.escape(what) + r" completed, taking ([\d.]+)\s*(ms|s)\b", log)
    if not m:
        return None
    v, unit = m[-1]
    return round(float(v) / (1000 if unit == "ms" else 1), 2)


def image_models_loaded() -> list[str]:
    """Image models llama-swap has running now; [] if llama-swap is down."""
    try:
        with urllib.request.urlopen(SWAP + "/running", timeout=3) as r:
            d = json.load(r)
    except Exception:                                            # noqa: BLE001
        return []
    return [x.get("model") for x in d.get("running") or []
            if str(x.get("model", "")).startswith("imagegen")]


def prompts() -> list[dict]:
    with open(os.path.join(PARTI, "manifest.json"), encoding="utf-8") as f:
        rows = json.load(f)
    out = []
    for r in rows:
        png = os.path.join(PARTI, f"{r['i']:02d}.png")
        sha = None
        seed = SEEDS.get(r["i"])
        if os.path.exists(png):
            with open(png, "rb") as f:
                sha = hashlib.sha256(f.read()).hexdigest()
            meta = os.path.join(ROOT, "index", "media", sha + ".json")
            if os.path.exists(meta):
                with open(meta, encoding="utf-8") as f:
                    m = json.load(f)
                if m.get("seed") != seed:
                    raise SystemExit(f"prompt {r['i']}: SEEDS says {seed}, "
                                     f"{meta} says {m.get('seed')}")
        out.append({"i": r["i"], "prompt": r["prompt"], "category": r["category"],
                    "seed": seed, "saved_png": png if sha else None,
                    "saved_sha": sha})
    return out


def run_one(arm: str, p: dict, seed: int, size: int, out_dir: str,
            need: int, floor: int) -> dict:
    a = ARMS[arm]
    used0, free0, temp0 = smi()
    row = {"arm": arm, "i": p["i"], "category": p.get("category"),
           "prompt": p["prompt"], "seed": seed, "size": f"{size}x{size}",
           "steps": a["steps"], "flags": " ".join(a["extra"]),
           "diffusion": os.path.basename(a["diffusion"]),
           "base_used_mib": used0, "base_free_mib": free0,
           "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if free0 < need:
        row.update(skipped=True, why=f"CUDA1 free {free0} MiB < need {need}")
        return row
    out = os.path.join(out_dir, f"{arm}_{p['i']:02d}_s{seed}.png")
    cmd = [SD, "--diffusion-model", a["diffusion"], *SHARED, *a["extra"],
           "--steps", str(a["steps"]), "-p", p["prompt"],
           "-W", str(size), "-H", str(size), "-s", str(seed), "-o", out]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=A4000)
    peak = {"used": used0, "free": free0, "temp": temp0, "killed": False,
            "why": ""}
    t0 = time.time()
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace")
    stop = threading.Event()

    def watch():
        while not stop.is_set():
            try:
                u, f, t = smi()
            except Exception:                                    # noqa: BLE001
                stop.wait(0.25)
                continue
            peak["used"] = max(peak["used"], u)
            peak["free"] = min(peak["free"], f)
            peak["temp"] = max(peak["temp"], t)
            if (f < floor or t > TEMP_LIMIT) and proc.poll() is None:
                peak["killed"] = True
                peak["why"] = f"free {f} MiB" if f < floor else f"{t} C"
                proc.kill()
            stop.wait(0.25)

    th = threading.Thread(target=watch, daemon=True)
    th.start()
    log = proc.stdout.read()
    proc.wait()
    wall = time.time() - t0
    stop.set()
    th.join()
    with open(out + ".log", "w", encoding="utf-8") as f:
        f.write(log)
    loads = re.findall(r"loading tensors completed, taking ([\d.]+)s", log)
    sample = timing(log, "sampling")
    row.update({
        "wall_s": round(wall, 2), "rc": proc.returncode,
        "killed": peak["killed"], "kill_why": peak["why"],
        "peak_temp_c": peak["temp"], "peak_used_mib": peak["used"],
        "min_free_mib": peak["free"], "delta_peak_mib": peak["used"] - used0,
        "encode_s": timing(log, "get_learned_condition"),
        "sample_s": sample,
        "sec_per_step": round(sample / a["steps"], 2) if sample else None,
        "decode_s": timing(log, "decode_first_stage"),
        "load_s": round(sum(float(x) for x in loads), 2) if loads else None,
        "vae_retiled": "retrying with spatial tiling" in log,
        "errors": [ln.strip()[:160] for ln in log.splitlines()
                   if "[ERROR" in ln][:4],
        "image": out if os.path.exists(out) else None,
    })
    if row["image"]:
        with open(out, "rb") as f:
            row["sha"] = hashlib.sha256(f.read()).hexdigest()
        if arm == "base" and p.get("saved_sha"):
            row["same_as_saved"] = row["sha"] == p["saved_sha"]
    return row


def side_by_side(pairs: list[tuple[dict, str | None, str | None]], out_dir: str) -> None:
    """One base|turbo PNG per prompt, plus a contact sheet of every pair."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("  PIL missing: no side-by-side written")
        return
    tiles = []
    for p, base, turbo in pairs:
        ims = [Image.open(x).convert("RGB") if x and os.path.exists(x) else None
               for x in (base, turbo)]
        w, h = 512, 512
        canvas = Image.new("RGB", (2 * w + 8, h + 28), (16, 16, 16))
        d = ImageDraw.Draw(canvas)
        for k, (im, name) in enumerate(zip(ims, ("base 20 steps", "turbo 4 steps"))):
            x = k * (w + 8)
            if im is not None:
                canvas.paste(im.resize((w, h)), (x, 28))
            else:
                d.text((x + 8, 28 + h // 2), "no image", fill=(220, 80, 80))
            d.text((x + 6, 8), f"{name}  #{p['i']}", fill=(220, 220, 220))
        path = os.path.join(out_dir, f"pair_{p['i']:02d}.png")
        canvas.save(path)
        tiles.append(canvas)
    if tiles:
        tw, th = tiles[0].size
        sheet = Image.new("RGB", (tw, th * len(tiles)), (0, 0, 0))
        for k, t in enumerate(tiles):
            sheet.paste(t, (0, k * th))
        sheet.save(os.path.join(out_dir, "contact_sheet.png"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arms", default="base,turbo")
    ap.add_argument("--prompts", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--seed", type=int, default=None,
                    help="override every prompt's seed (default: the base run's)")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--need", type=int, default=7000)
    ap.add_argument("--floor", type=int, default=1024)
    ap.add_argument("--out", default=os.path.join(
        HERE, "turbo-compare-" + time.strftime("%Y%m%d")))
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    arms = [x for x in a.arms.split(",") if x]
    for x in arms:
        if x not in ARMS:
            raise SystemExit(f"unknown arm {x!r}; arms are {sorted(ARMS)}")
    want = {int(x) for x in a.prompts.split(",") if x != ""}
    plan = [p for p in prompts() if p["i"] in want]
    if a.list:
        for p in plan:
            print(f"  #{p['i']} seed {a.seed if a.seed is not None else p['seed']}: {p['prompt']}")
        for x in arms:
            print(f"  {x}: {os.path.basename(ARMS[x]['diffusion'])} "
                  f"--steps {ARMS[x]['steps']} {' '.join(ARMS[x]['extra'])}")
        return 0
    loaded = image_models_loaded()
    if loaded:
        print(f"refusing: llama-swap has {loaded} loaded on CUDA1 -- a second "
              f"image consumer. Unload it (or wait for its ttl) and rerun.")
        return 2
    for x in arms:
        if not os.path.exists(ARMS[x]["diffusion"]):
            raise SystemExit(f"missing {ARMS[x]['diffusion']}")
    os.makedirs(a.out, exist_ok=True)
    results = os.path.join(a.out, "results.jsonl")
    got: dict[tuple[str, int], str | None] = {}
    for p in plan:
        seed = a.seed if a.seed is not None else p["seed"]
        for arm in arms:
            row = run_one(arm, p, seed, a.size, a.out, a.need, a.floor)
            print(json.dumps({k: row.get(k) for k in (
                "arm", "i", "seed", "wall_s", "sample_s", "decode_s",
                "delta_peak_mib", "min_free_mib", "peak_temp_c", "killed",
                "skipped", "same_as_saved")}), flush=True)
            with open(results, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            if row.get("killed") or row.get("skipped"):
                print(f"stopped: {row.get('kill_why') or row.get('why')}")
                return 1
            got[(arm, p["i"])] = row.get("image")
    # Without a base re-run, the saved base PNG stands in -- but only at the
    # base run's own seed; with --seed it would pair two different seeds.
    side_by_side([(p, got.get(("base", p["i"]))
                   or (p.get("saved_png") if a.seed is None else None),
                   got.get(("turbo", p["i"]))) for p in plan], a.out)
    print(f"done: {results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
