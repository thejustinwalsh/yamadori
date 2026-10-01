"""THE FIT FOR `bonsai-a4000` (tier models, 2026-09-30): the window of the second Bonsai on the A4000, sized with its
real load -- jjava (the decider's burst on slot 1: a 3,071-token state, one-token reads) and side calls (titles and
summaries on slot 0) -- beside what stays resident there (embeddings; the 2026-09-30 fit, -c 141,312, was measured
with the reranker resident too -- it was removed 2026-10-01, docs/REMOVED.md, and that fit is kept as measured). Nothing here is a claim:
the window is DERIVED FROM TWO MEASUREMENTS (bench/mirai_s_gate.py derive_fit's arithmetic), never from the estimate
in mcp/gpu_room.py's SIZES row, which the result replaces.

    python bench/a4000_fit.py --dry-run                     # the command, touch nothing
    python bench/a4000_fit.py --window --out DIR            # GPU (the A4000): the fit

THE WINDOW'S DISCIPLINE (the coordinator's, 2026-09-29/30): the worker's gpu lane held for the whole run
(engine_corruption.hold_lane / release_lane, the error path included); bonsai-vision, imagegen and imagegen-turbo
unloaded first through gpu_room.unload (llama-swap's /api/models/unload), embeddings loaded and
kept; production `bonsai` on the 5060 Ti is NOT touched (this card is the A4000). One arm server at a time on a
test port, CUDA_VISIBLE_DEVICES = the A4000's UUID.

THE MEASUREMENT: the fragment's argv (config.bonsai-a4000.fragment.yaml) at -c C1 and -c C2; each warmed with a
16,384-token side call on slot 0 and a 10-read jjava burst on slot 1 at the same time; the A4000's used MiB polled
every 0.5 s (vitals.gpus, by UUID) -> the peak. bytes per cell = the slope; the largest -c whose peak keeps
gpu_room.HEADROOM_MIB (1,331) free = THE WINDOW (whole 1,024s). Written to DIR/fit.json; bench/deploy_tier_models.py
--a4000-fit DIR reads it.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "mcp"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

MODELS = "C:/Users/jwals/textgen/user_data/models"
TRUNK = f"{MODELS}/Ternary-Bonsai-2-27B-PTQ1_0.gguf"
FIT_POINTS = (16384, 49152)
SIDE_TOKENS = 16384
LANE_TOKENS = 3072
EVICT = ("bonsai-vision", "imagegen", "imagegen-turbo")
KEEP = ("embeddings",)          # ("embeddings", "reranker") for the 2026-09-30 fit; the reranker went 2026-10-01


def argv(ctx: int, port: int, exe: str) -> list[str]:
    """config.bonsai-a4000.fragment.yaml's command (sampling as config.yaml's macro)."""
    return [exe, "--port", str(port), "-m", TRUNK, "-dev", "CUDA0", "-ngl", "999", "-c", str(ctx),
            "--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "-fa", "on", "-np", "2", "--kv-unified",
            "-b", "1024", "-ub", "512", "--jinja", "--reasoning-format", "deepseek", "--no-context-shift",
            "--no-cache-idle-slots", "--reasoning-budget", "32768",
            "--temp", "1.0", "--top-p", "0.95", "--top-k", "20", "--min-p", "0.0", "--presence-penalty", "0",
            "--repeat-penalty", "1.0"]


def engine_exe() -> str:
    import build_engine as be
    return (be.load_yaml(be.MANIFEST)["engines"]["llama-bonsai2-ada"].get("shipped") or {}).get("path", "")


class Peak:
    """The A4000's used MiB, polled until stop()."""

    def __init__(self):
        import gpu_room
        self.gr, self.max_used, self._stop = gpu_room, 0, threading.Event()
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def _run(self):
        while not self._stop.is_set():
            c = self.gr.card()
            if c:
                self.max_used = max(self.max_used, int(c["used_mib"]))
            time.sleep(0.5)

    def stop(self) -> int:
        self._stop.set()
        self.th.join(5)
        return self.max_used


def post(base: str, body: dict, timeout: float = 3600) -> dict:
    import urllib.request
    req = urllib.request.Request(base + "/completion", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def warm(base: str, ctx: int) -> dict:
    """The load it is sized for, at once: a side call's long prompt on slot 0 and a jjava burst on slot 1. The side
    prompt is at most half the pool (the two slots share it: -np 2 --kv-unified; a prompt past the slot's window is a
    400, the first run's error)."""
    import engine_phase2 as ep
    toks = ep.corpus_tokens(base)
    toks = (toks * (SIDE_TOKENS // max(len(toks), 1) + 2))
    n_side = min(SIDE_TOKENS, ctx // 2)
    side = toks[:n_side]
    state = toks[SIDE_TOKENS:SIDE_TOKENS + LANE_TOKENS - 1]
    out: dict = {"side_tokens": n_side}
    th = threading.Thread(target=lambda: out.update(side=(post(base, {
        "prompt": side, "n_predict": 64, "id_slot": 0, "cache_prompt": True, "temperature": 0}).get("timings"))))
    th.start()
    t0 = time.time()
    for _ in range(10):
        post(base, {"prompt": state, "n_predict": 1, "id_slot": 1, "cache_prompt": True, "temperature": 0})
    out["burst_s"] = round(time.time() - t0, 1)
    th.join()
    return out


def one_point(ctx: int, port: int, out: str, exe: str) -> dict:
    import gpu_room
    import urllib.request
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_room.CARD_UUID, CUDA_DEVICE_ORDER="PCI_BUS_ID",
               GGML_CUDA_BATCH_INVARIANT="1")
    log = open(os.path.join(out, f"fit-{ctx}.server.log"), "w", encoding="utf-8")
    peak = Peak()
    proc = subprocess.Popen(argv(ctx, port, exe), env=env, stdout=log, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    rec: dict = {"ctx": ctx}
    try:
        t0 = time.time()
        while time.time() - t0 < 900:
            if proc.poll() is not None:
                raise RuntimeError(f"server exited {proc.returncode}")
            try:
                with urllib.request.urlopen(base + "/health", timeout=5) as r:
                    if r.status == 200:
                        break
            except Exception:                                        # noqa: BLE001
                time.sleep(2)
        rec["load_s"] = round(time.time() - t0, 1)
        rec["warm"] = warm(base, ctx)
    except Exception as e:                                           # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"[:300]
    finally:
        proc.terminate()
        try:
            proc.wait(60)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
        rec["peak_used_mib"] = peak.stop()
    return rec


def main(argv_: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--window", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--port", type=int, default=18097)
    a = ap.parse_args(argv_)
    exe = engine_exe()
    if a.dry_run:
        print(" ".join(argv(FIT_POINTS[0], a.port, exe or "llama-server.exe")))
        return 0
    if not a.window or not a.out:
        print("refusing: this unloads the A4000's on-demand models. Pass --window (a granted window) and --out.")
        return 2
    import engine_corruption as ec
    import gpu_room
    import mirai_s_gate as mg
    os.makedirs(a.out, exist_ok=True)
    # the A4000's own gpu scope (mcp/jobs.py GPU_SCOPES): the main card's jobs are not this window's
    hold = ec.hold_lane("bonsai-a4000 fit (bench/a4000_fit.py)", 2 * 3600, lane="gpu_a4000")
    res: dict = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "points": []}
    try:
        up = gpu_room.default_upstream()
        res["evicted"] = {m: gpu_room.unload(up, m) for m in EVICT if gpu_room.model_loaded(up, m)[0]}
        for m in KEEP:
            if not gpu_room.model_loaded(up, m)[0]:
                import urllib.request
                urllib.request.urlopen(f"{up}/upstream/{m}/health", timeout=300).read()
        time.sleep(3)
        for c in FIT_POINTS:
            res["points"].append(one_point(c, a.port, a.out, exe))
        card = gpu_room.card() or {}
        pts = [(p["ctx"], p["peak_used_mib"]) for p in res["points"] if not p.get("error")]
        if card and len(pts) == 2:
            res["fit"] = mg.derive_fit(pts, int(card["total_mib"]), gpu_room.HEADROOM_MIB)
            res["ctx"] = res["fit"]["max_ctx"]
    finally:
        res["lane"] = ec.release_lane(hold)
        res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        json.dump(res, open(os.path.join(a.out, "fit.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: res.get(k) for k in ("ctx", "fit")}, indent=1))
    return 0 if res.get("ctx") else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
