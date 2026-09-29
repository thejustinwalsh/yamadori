"""Clean swaps both ways, with no reload leaks (operator, 2026-09-29: "If we get clean swaps ..."), measured through
:1234. Each step is one tiny request at a tier; the proxy (mcp/max_mode.py) swaps the card for it. After every
landing: the 5060 Ti's used MiB (nvidia-smi, by UUID), llama-swap's /running (which never loads), the swap record the
proxy returned (x_yamadori.capacity.swap: from, to, load_s, left_loaded), the wall seconds.

    python bench/tier_swap_probe.py --key-file PATH --out DIR [--cycles 2]

The cycle is medium -> xhigh -> max -> medium -> max -> xhigh -> medium (every direction between the three). CLEAN
means: every landing has exactly its model among the main models on /running, left_loaded is empty, and the used MiB
of each model's landings agree within --tolerance MiB across cycles (a leak grows).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

PROXY = os.environ.get("YAMADORI_PROXY", "http://127.0.0.1:1234")
SWAP = "http://127.0.0.1:11434"
CARD = "GPU-de660e90-0e9c-d465-b389-6df63021b920"
MAIN = {"bonsai", "mirai-s", "flash-next"}
CYCLE = ["medium", "xhigh", "max", "medium", "max", "xhigh", "medium"]


def used_mib() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=uuid,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
    except Exception:                                                # noqa: BLE001
        return None
    for ln in out.splitlines():
        u, _, m = ln.partition(",")
        if u.strip() == CARD:
            return int(m.strip())
    return None


def running() -> list[str]:
    with urllib.request.urlopen(f"{SWAP}/running", timeout=10) as r:
        return sorted(m.get("model") for m in json.load(r).get("running") or []
                      if m.get("state") != "stopped" and m.get("model") in MAIN)


def ask(key: str, tier: str) -> dict:
    body = {"model": "yamadori", "reasoning_effort": tier, "max_tokens": 16,
            "messages": [{"role": "user", "content": "Reply with exactly: ok"}]}
    feats = {"investigate": False, "fanout": 1, "repair": False, "retrieval": False}
    req = urllib.request.Request(f"{PROXY}/v1/chat/completions", data=json.dumps(body).encode(), headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json",
        "X-Yamadori-Features": json.dumps(feats)})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    cap = (d.get("x_yamadori") or {}).get("capacity") or {}
    return {"s": round(time.time() - t0, 1), "model": cap.get("model"), "swap": cap.get("swap"),
            "content": ((d.get("choices") or [{}])[0].get("message") or {}).get("content", "")[:40]}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--key-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cycles", type=int, default=2)
    ap.add_argument("--tolerance", type=int, default=256, help="MiB a model's landings may differ by")
    a = ap.parse_args(argv)
    key = open(a.key_file, encoding="utf-8").read().strip()
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for c in range(a.cycles):
        for tier in CYCLE:
            r = ask(key, tier)
            time.sleep(3)
            r.update(cycle=c, tier=tier, used_mib=used_mib(), running=running())
            rows.append(r)
            print(json.dumps(r), flush=True)
    by: dict = {}
    for r in rows:
        by.setdefault(r["model"], []).append(r["used_mib"])
    clean = {
        "one_main_model_each_landing": all(r["running"] == [r["model"]] for r in rows),
        "no_left_loaded": all(not (r.get("swap") or {}).get("left_loaded") for r in rows),
        "vram_stable": {m: (max(v) - min(v)) for m, v in by.items() if None not in v},
    }
    clean["CLEAN"] = (clean["one_main_model_each_landing"] and clean["no_left_loaded"]
                      and all(d <= a.tolerance for d in clean["vram_stable"].values()))
    loads = [(r["swap"] or {}).get("load_s") for r in rows if r.get("swap")]
    out = {"rows": rows, "clean": clean, "load_s": loads, "n_swaps": len(loads)}
    json.dump(out, open(os.path.join(a.out, "swaps.json"), "w"), indent=1)
    print(json.dumps(clean, indent=1))
    return 0 if clean["CLEAN"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
