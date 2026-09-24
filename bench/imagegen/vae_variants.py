#!/usr/bin/env python
"""Regenerate fixed seeds under one VAE-decode variant (SELF-IMPROVEMENT-LOG #25).

    python bench/imagegen/vae_variants.py --model imagegen-turbo --variant base \
        --jobs jobs.json --out DIR [--set=--max-vram=8] [--add=--vae-tiling] ...

Starts ITS OWN sd-server on --port (never llama-swap's), with EXACTLY the
command config.yaml gives --model (macros expanded) except the changes named:
  --set FLAG=VALUE   replace FLAG's value (e.g. --set=--max-vram=10)
  --add FLAG[=VALUE] append a flag     (e.g. --add --vae-tile-size=64x64)
  --bin PATH         a different sd-server executable (a patched build)
Pass them as --set=FLAG=VALUE / --add=FLAG=VALUE (argparse would read a bare
"--max-vram=8" as an option of its own).
pinned to the A4000 by UUID (CUDA_VISIBLE_DEVICES), as config.yaml pins it.
The request body is the one mcp/images.generate sends (prompt, width, height,
steps, seed, batch_size 1).

A4000 memory is sampled by UUID every 0.5 s. Before the server starts, the
card must have --need MiB free above the 1,331 MiB headroom
(mcp/gpu_room.HEADROOM_MIB) or the variant is SKIPPED and recorded as such;
while it runs, the server is killed the moment free memory drops below the
headroom (recorded killed=true, never a result). One image at a time. The
server is always stopped on exit.

jobs.json: [{"tag", "prompt", "seed", "steps", "size"}], one image each.
Each row appended to --results: variant, flags, tag, seed, wall_s, peak_used,
min_free, used_before, decode (tiled / untiled / retried, decode seconds from
sd.cpp's log), white_blocks (bench/imagegen/white_blocks.py), png path.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from white_blocks import white_blocks  # noqa: E402

UUID = "GPU-43e37d0c-4104-9056-2552-6109d4d3382c"
HEADROOM = 1331


def smi() -> tuple[int, int]:
    out = subprocess.run(
        ["nvidia-smi", f"--id={UUID}", "--query-gpu=memory.used,memory.free",
         "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    used, free = [int(x) for x in out.strip().split(",")]
    return used, free


def command(model: str, port: int, sets: list[str], adds: list[str],
            binary: str | None = None) -> list[str]:
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config.yaml"), encoding="utf-8"))
    cmd = cfg["models"][model]["cmd"]
    for k, v in cfg.get("macros", {}).items():
        cmd = cmd.replace("${" + k + "}", str(v))
    cmd = cmd.replace("${PORT}", str(port))
    toks = cmd.split()
    if binary:
        toks[0] = binary
    for s in sets:
        flag, val = s.split("=", 1)
        i = toks.index(flag)
        toks[i + 1] = val
    for a in adds:
        flag, _, val = a.partition("=")
        toks += [flag] + ([val] if val else [])
    return toks


def decode_info(log: str) -> dict:
    retried = "retrying with" in log
    tile = re.findall(r"VAE Tile size: (\d+x\d+)", log)
    dec = re.findall(r"decode_first_stage completed, taking ([\d.]+)s", log)
    oom = "ran out of memory" in log
    return {"retried_tiled": retried, "oom_msgs": log.count("ran out of memory") if oom else 0,
            "tile": tile[-1] if tile else None,
            "decode_s": float(dec[-1]) if dec else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--results", default=os.path.join(HERE, "vae_variants.jsonl"))
    ap.add_argument("--port", type=int, default=18191)
    ap.add_argument("--need", type=int, default=6400,
                    help="MiB this variant may add at peak; skip unless free - need >= headroom")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--add", action="append", default=[])
    ap.add_argument("--warmup", action="store_true", help="one 512x512 draw first, not recorded")
    ap.add_argument("--bin", default=None, help="another sd-server executable, same flags")
    a = ap.parse_args()

    jobs = json.load(open(a.jobs, encoding="utf-8"))
    if os.path.exists(a.results):   # resume: skip tags this variant already drew
        done = set()
        for line in open(a.results, encoding="utf-8"):
            r = json.loads(line)
            if r.get("variant") == a.variant and r.get("png") and not r.get("killed"):
                done.add(r.get("tag"))
        jobs = [j for j in jobs if j["tag"] not in done]
        if not jobs:
            print("nothing to do")
            return 0
    os.makedirs(a.out, exist_ok=True)
    toks = command(a.model, a.port, a.set, a.add, a.bin)
    base_row = {"variant": a.variant, "model": a.model, "set": a.set, "add": a.add, "bin": a.bin,
                "when": time.strftime("%Y-%m-%dT%H:%M:%S")}

    def record(row):
        with open(a.results, "a", encoding="utf-8") as f:
            f.write(json.dumps({**base_row, **row}) + "\n")
        print(json.dumps(row), flush=True)

    used0, free0 = smi()
    if free0 - a.need < HEADROOM:
        record({"skipped": True, "why": f"A4000 free {free0} MiB - need {a.need} < headroom {HEADROOM}"})
        return 2

    log_path = os.path.join(a.out, f"server_{a.variant}.log")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=UUID)
    logf = open(log_path, "w", encoding="utf-8", errors="replace")
    p = subprocess.Popen(toks, stdout=logf, stderr=subprocess.STDOUT, env=env)
    state = {"peak": used0, "minfree": free0, "killed": None, "stop": False}

    def watch():
        while not state["stop"] and p.poll() is None:
            try:
                u, f = smi()
            except Exception:                                         # noqa: BLE001
                time.sleep(0.5)
                continue
            state["peak"] = max(state["peak"], u)
            state["minfree"] = min(state["minfree"], f)
            if f < HEADROOM and p.poll() is None:
                state["killed"] = f"A4000 free {f} MiB < {HEADROOM}"
                p.kill()
            time.sleep(0.5)

    th = threading.Thread(target=watch, daemon=True)
    th.start()
    url = f"http://127.0.0.1:{a.port}"
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(url + "/v1/models", timeout=2).read()
                break
            except Exception:                                         # noqa: BLE001
                if p.poll() is not None:
                    raise RuntimeError("sd-server exited at start; see " + log_path)
                time.sleep(0.5)

        def draw(prompt, seed, steps, size):
            w, h = [int(x) for x in size.split("x")]
            body = {"model": a.model, "prompt": prompt, "width": w, "height": h,
                    "steps": steps, "seed": seed, "batch_size": 1}
            req = urllib.request.Request(url + "/sdapi/v1/txt2img", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=900) as r:
                d = json.loads(r.read())
            return base64.b64decode(d["images"][0]), round(time.time() - t0, 2)

        if a.warmup:
            draw(jobs[0]["prompt"], 1, jobs[0]["steps"], "512x512")
        for j in jobs:
            if state["killed"]:
                break
            u_before, f_before = smi()
            state["peak"], state["minfree"] = u_before, f_before
            logf.flush()
            mark = os.path.getsize(log_path)
            try:
                png, wall = draw(j["prompt"], j["seed"], j["steps"], j["size"])
            except Exception as e:                                    # noqa: BLE001
                record({"tag": j["tag"], "seed": j["seed"], "error": f"{type(e).__name__}: {e}",
                        "killed": state["killed"], "peak_used": state["peak"],
                        "min_free": state["minfree"], "used_before": u_before})
                continue
            time.sleep(0.6)  # one more sample after the answer
            logf.flush()
            with open(log_path, encoding="utf-8", errors="replace") as lf:
                lf.seek(mark)
                seg = lf.read()
            path = os.path.join(a.out, f"{j['tag']}.png")
            open(path, "wb").write(png)
            wb = white_blocks(path)
            record({"tag": j["tag"], "seed": j["seed"], "steps": j["steps"], "size": j["size"],
                    "wall_s": wall, "used_before": u_before, "peak_used": state["peak"],
                    "peak_delta": state["peak"] - u_before, "min_free": state["minfree"],
                    "killed": state["killed"], **decode_info(seg),
                    "white_blocks": wb["blocks"], "clusters": wb["clusters"], "png": path})
    finally:
        state["stop"] = True
        if p.poll() is None:
            p.kill()
        p.wait(timeout=30)
        logf.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
