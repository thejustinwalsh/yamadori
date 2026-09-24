#!/usr/bin/env python
"""Prove sd-server + mcp/images.py work together, end to end, then stop.

    python bench/imagegen/smoke_server.py

Starts sd-server on 127.0.0.1:1240 (never 1234: that default is the proxy's
port) with EXACTLY the flags docs/IMAGEGEN.md proposes for llama-swap, pinned
to the A4000 by UUID, waits for /v1/models, then calls `images.generate`
twice -- the same client code the proxy runs -- with the body llama-swap would
route on (`model: imagegen`). Records wall time and CUDA1 peak, kills the
server on exit (or if CUDA1 free drops under 1 GiB), and leaves nothing
listening. The media store is a temp dir; the PNGs are copied to samples/.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "mcp"))
sys.path.insert(0, HERE)
import measure  # noqa: E402

PORT = 1240
A4000 = "GPU-43e37d0c-4104-9056-2552-6109d4d3382c"
SERVER = r"C:/Users/jwals/stable-diffusion.cpp/build/bin/sd-server.exe"
FLAGS = ["--listen-ip", "127.0.0.1", "--listen-port", str(PORT),
         "--diffusion-model", measure.DIFF, "--vae", measure.VAE,
         "--llm", measure.TE,
         "--cfg-scale", "6.0", "--sampling-method", "euler", "--steps", "20",
         "--diffusion-fa", "--backend", "cuda0", "--offload-to-cpu",
         "--max-vram", "6", "-t", "8"]


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="imagegen_smoke_")
    os.environ.update(YAMADORI_IMAGEGEN_URL=f"http://127.0.0.1:{PORT}",
                      YAMADORI_MEDIA_DIR=os.path.join(tmp, "media"),
                      YAMADORI_MEDIA_SECRET_FILE=os.path.join(tmp, "key"))
    import images
    used0, free0 = measure.smi()
    if free0 < 7500:
        print(json.dumps({"skipped": True, "free_mib": free0}))
        return 1
    log = open(os.path.join(tmp, "server.log"), "w", encoding="utf-8")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=A4000)
    t0 = time.time()
    p = subprocess.Popen([SERVER, *FLAGS], env=env, stdout=log,
                         stderr=subprocess.STDOUT)
    peak = {"used": used0, "free": free0, "killed": False}
    stop = threading.Event()

    def poll():
        while not stop.is_set():
            try:
                u, f = measure.smi()
            except Exception:                                    # noqa: BLE001
                continue
            peak["used"], peak["free"] = max(peak["used"], u), min(peak["free"], f)
            if f < 1024 and p.poll() is None:
                peak["killed"] = True
                p.kill()
            stop.wait(0.25)

    threading.Thread(target=poll, daemon=True).start()
    rows = []
    try:
        ready = None
        for _ in range(240):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models",
                                            timeout=2) as r:
                    if r.status == 200:
                        ready = round(time.time() - t0, 2)
                        break
            except Exception:                                    # noqa: BLE001
                time.sleep(0.5)
        print(json.dumps({"ready_s": ready}), flush=True)
        for prompt, seed in ((measure.PROMPTS["sign"], 7),
                             ("A watercolor painting of a small juniper bonsai "
                              "on a wooden stand, soft morning light", 8)):
            t1 = time.time()
            try:
                r = images.generate(prompt, size="1024x1024", seed=seed)[0]
                dst = os.path.join(HERE, "samples", f"server_{seed}_{r['id'][:12]}.png")
                shutil.copy(images.media_path(r["id"]), dst)
                row = {"ok": True, "seconds_client": round(time.time() - t1, 2),
                       "seconds_meta": r["seconds"], "seed": seed,
                       "sample": os.path.relpath(dst, os.path.join(HERE, "..", "..")).replace(os.sep, "/"),
                       "meta": images.metadata(r["id"])}
            except images.ImageError as e:
                row = {"ok": False, "code": e.code, "reason": e.reason}
            rows.append(row)
            print(json.dumps(row), flush=True)
    finally:
        stop.set()
        if p.poll() is None:
            p.kill()
        p.wait()
        log.close()
    out = {"config": "sd-server b_offload_budget6", "base_used_mib": used0,
           "peak_used_mib": peak["used"], "min_free_mib": peak["free"],
           "delta_peak_mib": peak["used"] - used0, "killed": peak["killed"],
           "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "images": rows}
    print(json.dumps(out), flush=True)
    with open(os.path.join(HERE, "results.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(out) + "\n")
    shutil.copy(os.path.join(tmp, "server.log"),
                os.path.join(HERE, "samples", "server_smoke.log"))
    return 0 if all(r["ok"] for r in rows) and not peak["killed"] else 1


if __name__ == "__main__":
    sys.exit(main())
