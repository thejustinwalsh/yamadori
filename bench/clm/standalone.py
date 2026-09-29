#!/usr/bin/env python
"""Run config.yaml's `clm-encoder` flags with a given GGUF on the A4000, on a
scratch port, for a measurement -- then stop it. (docs/CLM.md "How it was
measured".)

    python bench/clm/standalone.py --gguf <models>/Qwen3-8B-Q8_0.gguf --tag q8_0
    python bench/clm/standalone.py --gguf ... --tag q8_0_cpu --ngl 0     # op offload
    python bench/clm/standalone.py --gguf ... --tag q8_0_ngl18 --ngl 18  # a split
    python bench/clm/standalone.py --gguf <bonsai>.gguf --tag bonsai --probe-only

WHY NOT THROUGH LLAMA-SWAP. The running llama-swap was started without
-watch-config, so a new config.yaml entry is served only after llama-swap
restarts -- which restarts the main model too. That is a deploy (AGENTS.md
"Before you claim anything works"), not a measurement. This runs the SAME
binary (config.yaml macros.server, unless --server) with the SAME arguments
(the entry's cmd, parsed from config.yaml; -m and --port replaced, -ngl when
--ngl is given, --add appended) and the same card pin (the entry's
CUDA_VISIBLE_DEVICES). The live check through llama-swap is
mcp/test_live_stack.py --only clm, after the restart.

ONE GPU CONSUMER AT A TIME. It holds mcp/gpu_room.py's room lock for its
whole run, so every coordinated A4000 load (vision, a draw, a reload of
search) waits for it or fails fast; it refuses to start when the card has
less free than the model needs plus the headroom (full offload only).

What it measures: load time (to /health), VRAM held idle (free before -
free after load) and at peak (after a full-window input), per-call latency
at fixed state lengths (distinct inputs: llama-server reuses a slot's cached
prefix even for embeddings), per-call latency over the fidelity states,
determinism, the prefix-cache effect, and the fidelity vectors
(bench/clm/fidelity.py served).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "mcp"))
sys.path.insert(0, HERE)

import gpu_room  # noqa: E402

RESULTS = os.path.join(HERE, "results")
LENGTHS = (40, 300, 800, 1500, 2047)
# Latency text: this repo's own prose, deterministic.
LATENCY_SOURCE = os.path.join(ROOT, "docs", "CLM-EVAL.md")
FIRST_WORDS = ("Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot")


def entry_argv(gguf: str, port: int, *, server: str | None = None,
               ngl: int | None = None, add: str = "",
               config: str = os.path.join(ROOT, "config.yaml")
               ) -> tuple[list[str], dict]:
    import yaml
    with open(config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    e = cfg["models"]["clm-encoder"]
    macros = cfg.get("macros") or {}
    lines = []
    for ln in e["cmd"].splitlines():
        ln = ln.split("#", 1)[0].strip()
        if ln:
            lines.append(ln)
    cmd = " ".join(lines)
    for k, v in macros.items():
        cmd = cmd.replace("${" + k + "}", str(v))
    cmd = cmd.replace("${PORT}", str(port))
    argv = shlex.split(cmd, posix=True)
    argv[argv.index("-m") + 1] = gguf
    if server:
        argv[0] = server
    if ngl is not None:
        argv[argv.index("-ngl") + 1] = str(ngl)
    argv += shlex.split(add, posix=True) if add else []
    env = dict(os.environ)
    for kv in e.get("env") or []:
        k, v = kv.split("=", 1)
        env[k] = v
    return argv, env


@contextlib.contextmanager
def serving(gguf: str, port: int = 10991, *, server: str | None = None,
            ngl: int | None = None, add: str = "", tag: str = "serve"):
    """A llama-server with the clm-encoder flags for the duration of the
    block, holding the A4000's room lock. Yields (base url, record)."""
    argv_srv, env = entry_argv(gguf, port, server=server, ngl=ngl, add=add)
    size_mib = os.path.getsize(gguf) / 2**20
    full = ngl is None or ngl >= 99
    os.makedirs(RESULTS, exist_ok=True)
    lock = gpu_room.room_lock()
    if not lock.acquire(60):
        raise RuntimeError("the A4000's room lock is held (a load or a draw)")
    proc = None
    rec: dict = {"gguf": os.path.basename(gguf), "argv": argv_srv}
    try:
        c0 = gpu_room.card()
        rec["free_before_mib"] = c0 and c0["free_mib"]
        need = (size_mib if full else 0) + 1200 + gpu_room.HEADROOM_MIB
        if not c0 or c0["free_mib"] < need:
            raise RuntimeError(f"not enough room: free {c0} < ~{need:.0f} MiB")
        log = open(os.path.join(RESULTS, f"standalone_{tag}.log"), "w",
                   encoding="utf-8")
        t0 = time.time()
        proc = subprocess.Popen(argv_srv, env=env, stdout=log,
                                stderr=subprocess.STDOUT)
        base = f"http://127.0.0.1:{port}"
        while True:
            if proc.poll() is not None or time.time() - t0 > 900:
                raise RuntimeError(f"llama-server did not come up; see "
                                   f"standalone_{tag}.log")
            try:
                if _get(base + "/health")[0] == 200:
                    break
            except Exception:                                    # noqa: BLE001
                pass
            time.sleep(0.2)
        rec["load_s"] = round(time.time() - t0, 2)
        c1 = gpu_room.card()
        rec["vram_idle_mib"] = c1 and c0["free_mib"] - c1["free_mib"]
        yield base, rec
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(30)
            except subprocess.TimeoutExpired:
                proc.kill()
        lock.release()


def _get(url: str, timeout: float = 5.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status, r.read()


def _ms(fn) -> float:
    t = time.perf_counter()
    fn()
    return (time.perf_counter() - t) * 1000


def latency_inputs(tok, n: int, k: int) -> list[list[int]]:
    """k distinct inputs of n tokens each (distinct FIRST token, so no two
    share a cached prefix)."""
    with open(LATENCY_SOURCE, encoding="utf-8") as fh:
        base = tok.ids(fh.read())
    while len(base) < n:
        base = base + base
    return [(tok.ids(FIRST_WORDS[i])[:1] + base[1:n]) for i in range(k)]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--port", type=int, default=10991)
    ap.add_argument("--server", default=None)
    ap.add_argument("--ngl", type=int, default=None)
    ap.add_argument("--add", default="", help="extra llama-server arguments")
    ap.add_argument("--no-fidelity", action="store_true")
    ap.add_argument("--probe-only", action="store_true",
                    help="load, report VRAM and the vector width, one call "
                         "per length; no fidelity (another model)")
    ap.add_argument("--lengths", default=",".join(map(str, LENGTHS)))
    ap.add_argument("--finetune-embed", action="store_true",
                    help="only fill the fine-tune embedding cache "
                         "(bench/clm/finetune_data.py embed)")
    a = ap.parse_args(argv)

    import clm
    import fidelity

    argv_srv, env = entry_argv(a.gguf, a.port, server=a.server, ngl=a.ngl,
                               add=a.add)
    size_mib = os.path.getsize(a.gguf) / 2**20
    full = a.ngl is None or a.ngl >= 99
    os.makedirs(RESULTS, exist_ok=True)
    lock = gpu_room.room_lock()
    if not lock.acquire(60):
        print("the A4000's room lock is held (a load or a draw): not run")
        return 3
    proc = None
    rec: dict = {"tag": a.tag, "gguf": os.path.basename(a.gguf),
                 "argv": argv_srv, "file_mib": round(size_mib)}
    try:
        c0 = gpu_room.card()
        rec["free_before_mib"] = c0 and c0["free_mib"]
        need = (size_mib if full else 0) + 1200 + gpu_room.HEADROOM_MIB
        if not c0 or c0["free_mib"] < need:
            print(f"not enough room: free {c0} < ~{need:.0f} MiB; not run")
            return 3
        log = open(os.path.join(RESULTS, f"standalone_{a.tag}.log"), "w",
                   encoding="utf-8")
        t0 = time.time()
        proc = subprocess.Popen(argv_srv, env=env, stdout=log,
                                stderr=subprocess.STDOUT)
        base = f"http://127.0.0.1:{a.port}"
        while time.time() - t0 < 900:
            if proc.poll() is not None:
                raise RuntimeError(f"llama-server exited {proc.returncode}; "
                                   f"see standalone_{a.tag}.log")
            try:
                if _get(base + "/health")[0] == 200:
                    break
            except Exception:                                    # noqa: BLE001
                pass
            time.sleep(0.2)
        rec["load_s"] = round(time.time() - t0, 2)
        time.sleep(2)
        c1 = gpu_room.card()
        rec["free_loaded_mib"] = c1 and c1["free_mib"]
        rec["vram_idle_mib"] = c0["free_mib"] - c1["free_mib"]

        if a.finetune_embed:
            import finetune_data
            rec["finetune_embed"] = finetune_data.embed(base)
            return 0
        enc = clm.HttpEncoder(url=base + "/v1/embeddings", model=None,
                              gpu_room_model=None)
        clm.set_encoder(enc)
        # Another model's vocabulary: token ids come from its own server.
        tok = _ServerTok(base) if a.probe_only else clm.tokenizer()
        if a.probe_only:
            import urllib.request as ur
            body = json.dumps({"input": "hello world"}).encode()
            req = ur.Request(base + "/v1/embeddings", data=body,
                             headers={"Content-Type": "application/json"})
            with ur.urlopen(req, timeout=300) as r:
                d = json.loads(r.read())
            rec["vector_width"] = len(d["data"][0]["embedding"])
        # Per-call latency at fixed lengths: 3 distinct inputs each, the
        # first call of the run (warm-up) timed separately.
        lengths = [int(x) for x in a.lengths.split(",") if x]
        warm = latency_inputs(tok, 16, 6)[5]
        if a.probe_only:
            rec["first_call_ms"] = round(_ms(lambda: _raw(base, warm)), 1)
        else:
            rec["first_call_ms"] = round(_ms(lambda: enc.embed_ids([warm])), 1)
        lat = {}
        for n in lengths:
            xs = latency_inputs(tok, n, 3)
            if a.probe_only:
                ts = [_ms(lambda x=x: _raw(base, x)) for x in xs]
            else:
                ts = [_ms(lambda x=x: enc.embed_ids([x])) for x in xs]
            lat[str(n)] = {"median": round(float(np.median(ts)), 1),
                           "all": [round(t, 1) for t in ts]}
        rec["per_call_ms"] = lat
        c2 = gpu_room.card()
        rec["free_after_full_window_mib"] = c2 and c2["free_mib"]
        rec["vram_peak_mib"] = c0["free_mib"] - c2["free_mib"]
        if a.probe_only:
            return 0

        # Latency over the fidelity states, 3 passes; determinism.
        qs = fidelity.load_questions()
        states = list(dict.fromkeys(clm.state_text(q["state"], q["instructions"])
                                    for q in qs))
        ids = [tok.ids(s)[-clm.CAP:] for s in states]
        ms, first = [], None
        for p in range(3):
            vs = []
            for x in ids:
                t = time.perf_counter()
                vs.append(enc.embed_ids([x])[0])
                ms.append((time.perf_counter() - t) * 1000)
            vs = np.stack(vs)
            if first is None:
                first = vs
            else:
                rec["determinism_max_abs"] = max(
                    rec.get("determinism_max_abs", 0.0),
                    float(np.abs(vs - first).max()))
        lens = [len(x) for x in ids]
        rec["fidelity_states_ms"] = {
            "n": len(ms), "p50": round(float(np.percentile(ms, 50)), 1),
            "p90": round(float(np.percentile(ms, 90)), 1),
            "max": round(float(np.max(ms)), 1),
            "tokens_p50": int(np.percentile(lens, 50)),
            "tokens_max": int(np.max(lens))}
        # THE PREFIX CACHE: the same input twice (the second reuses the
        # slot's KV and recomputes only the last token), vs the same input
        # after another (computed whole).
        sa, sb = ids[-1], ids[0]
        enc.embed_ids([sb])
        e1 = enc.embed_ids([sa])[0]
        e2 = enc.embed_ids([sa])[0]
        enc.embed_ids([sb])
        e3 = enc.embed_ids([sa])[0]
        rec["prefix_cache"] = {
            "tokens": len(sa),
            "repeat_cached_cos": float(e1 @ e2),
            "repeat_cached_max_abs": float(np.abs(e1 - e2).max()),
            "recomputed_max_abs": float(np.abs(e1 - e3).max())}
        if not a.no_fidelity:
            rec["fidelity"] = fidelity.served(a.tag, base)
        c3 = gpu_room.card()
        rec["free_end_mib"] = c3 and c3["free_mib"]
        rec["vram_peak_mib"] = max(rec["vram_peak_mib"],
                                   c0["free_mib"] - c3["free_mib"])
    finally:
        if proc is not None:
            t = time.time()
            proc.terminate()
            try:
                proc.wait(30)
            except subprocess.TimeoutExpired:
                proc.kill()
            rec["stop_s"] = round(time.time() - t, 2)
        lock.release()
        with open(os.path.join(RESULTS, f"standalone_{a.tag}.json"), "w",
                  encoding="utf-8", newline="\n") as fh:
            json.dump(rec, fh, indent=1, sort_keys=True)
        print(json.dumps({k: v for k, v in rec.items() if k != "argv"},
                         indent=1))
    return 0


class _ServerTok:
    def __init__(self, base: str):
        self.base = base

    def ids(self, text: str) -> list[int]:
        body = json.dumps({"content": text, "add_special": False}).encode()
        req = urllib.request.Request(self.base + "/tokenize", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())["tokens"]


def _raw(base: str, ids: list[int]) -> list[float]:
    """One embedding request for a model whose width is not CLM's."""
    body = json.dumps({"input": [ids]}).encode()
    req = urllib.request.Request(base + "/v1/embeddings", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["data"][0]["embedding"]


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
