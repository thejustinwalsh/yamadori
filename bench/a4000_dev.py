#!/usr/bin/env python
"""Flash-Next engine development on the A4000 ONLY (the operator, 2026-09-30: "Yes to everything pending"; the
coordinator: engine dev on CUDA1, sm_86, never the 5060 Ti, where the operator's inference runs).

Every process this script starts sees ONE device: CUDA_VISIBLE_DEVICES is the A4000's UUID, and the server log is
checked for it (a run that finds the 5060 Ti is stopped at once). VRAM is kept modest: a 40K window, every routed
expert in RAM, no MTP, no projector. llama-swap is never asked for anything but GET /running.

    python bench/a4000_dev.py profile --bin DIR [--ctx 4 32] [--out DIR]
        a 256-token decode (ignore_eos) at each context, with 0010's op timing (-lv 4) -- where the decode's
        GPU time goes as the context grows, and the server's own ms per token
    python bench/a4000_dev.py logits --bin DIR --ref-bin DIR [--ctx 4 32]
        the same prompts on two binaries, greedy, n_probs 8 per token: the tokens and the top-8 probabilities
        compared -- a changed graph (M2) against the reference path on the same card
    python bench/a4000_dev.py --selftest

Numbers from here are CORRECTNESS evidence and relative op costs; speed claims wait for the 5060 Ti.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bench"))

A4000_UUID = "GPU-43e37d0c-4104-9056-2552-6109d4d3382c"
MODELS = "C:/Users/jwals/textgen/user_data/models"
SHARD1 = f"{MODELS}/flash-next/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf"
PORT = 18195
OPT_RE = re.compile(r"op timing over (\d+) graphs: ([0-9.]+) ms per graph")
OPT_ROW = re.compile(r"^\s*([0-9.]+) ms/graph\s+([0-9.]+)%\s+x(\d+)\s+(.*)$")


def env_a4000(extra: dict | None = None) -> dict:
    e = dict(os.environ)
    e["CUDA_VISIBLE_DEVICES"] = A4000_UUID
    e.update(extra or {})
    return e


def args_for(ctx_k: int, threads: int) -> list[str]:
    # modest VRAM: the non-expert weights (~3.3 GB), a window of max(ctx)+8K cells, a 512 batch
    return ["-m", SHARD1, "-dev", "CUDA0", "-ngl", "999", "--n-cpu-moe", "48", "-c", str((ctx_k + 8) * 1024),
            "--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "-fa", "on", "-np", "1", "-b", "512", "-ub", "512",
            "-t", str(threads), "-lm", "mmap", "--lazy-mode", "on", "--jinja", "--no-context-shift",
            "--host", "127.0.0.1", "--port", str(PORT)]


class Server:
    def __init__(self, bin_dir: str, ctx_k: int, logp: str, env_extra: dict | None, threads: int, extra: list[str]):
        exe = os.path.join(bin_dir, "llama-server.exe")
        self.logp = logp
        self.log = open(logp, "w", encoding="utf-8")
        self.proc = subprocess.Popen([exe] + args_for(ctx_k, threads) + extra, stdout=self.log,
                                     stderr=subprocess.STDOUT, env=env_a4000(env_extra))
        self.base = f"http://127.0.0.1:{PORT}"
        import engine_corruption as ec
        self.ec = ec
        t0 = time.time()
        while time.time() - t0 < 900:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited {self.proc.returncode}: {logp}")
            st, _ = ec.http("GET", self.base + "/health", timeout=5)
            if st == 200:
                break
            time.sleep(2)
        else:
            self.stop()
            raise RuntimeError("server did not come up")
        self.load_s = round(time.time() - t0, 1)
        text = open(logp, encoding="utf-8", errors="replace").read()
        if "5060" in text:
            # the device list llama-server logs must never name the 5060 Ti
            self.stop()
            raise RuntimeError("the 5060 Ti is visible to this run")

    def stop(self):
        try:
            self.proc.terminate()
            self.proc.wait(timeout=60)
        except Exception:                                                # noqa: BLE001
            self.proc.kill()
        self.log.close()


def op_table(lines: list[str]) -> dict | None:
    idx = [i for i, ln in enumerate(lines) if OPT_RE.search(ln)]
    if not idx:
        return None
    i = idx[-1]
    m = OPT_RE.search(lines[i])
    rows = []
    for ln in lines[i + 1: i + 31]:
        mm = OPT_ROW.match(ln.split(" I ", 1)[-1] if " I " in ln else ln)
        if mm:
            rows.append({"ms_per_graph": float(mm.group(1)), "pct": float(mm.group(2)), "n": int(mm.group(3)),
                         "op": mm.group(4)})
    return {"graphs": int(m.group(1)), "ms_per_graph": float(m.group(2)), "top": rows}


def completion(base: str, prompt: str, n: int, n_probs: int = 0) -> dict:
    import engine_corruption as ec
    body = {"prompt": prompt, "n_predict": n, "cache_prompt": False, "temperature": 0.0, "top_k": 1, "seed": 0,
            "ignore_eos": True}
    if n_probs:
        body["n_probs"] = n_probs
    st, txt = ec.http("POST", base + "/completion", body, timeout=7200)
    if st != 200:
        raise RuntimeError(f"/completion HTTP {st}: {txt[:200]}")
    return json.loads(txt)


def corpus() -> str:
    import glob
    files = sorted(glob.glob(os.path.join(ROOT, "mcp", "*.py")))
    return "".join(open(f, encoding="utf-8", errors="replace").read() for f in files)


def cmd_profile(a) -> int:
    os.makedirs(a.out, exist_ok=True)
    text = corpus()
    rec = {"bin": a.bin, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "device": "A4000 " + A4000_UUID}
    logp = os.path.join(a.out, f"profile-{a.tag}.server.log")
    env = {} if a.no_op_timing else {"GGML_CUDA_OP_TIMING": "1", "GGML_CUDA_DISABLE_GRAPHS": "1"}
    env.update(dict(kv.split("=", 1) for kv in (a.env or [])))
    srv = Server(a.bin, max(a.ctx), logp, env, a.threads, ["-lv", "4"] + (a.extra.split() if a.extra else []))
    rec["load_s"] = srv.load_s
    try:
        for k in a.ctx:
            mark = os.path.getsize(logp)
            r = completion(srv.base, text[: k * 1024 * 4], a.n_predict)
            t = r.get("timings") or {}
            with open(logp, encoding="utf-8", errors="replace") as f:
                f.seek(mark)
                tail = f.read().splitlines()
            sched = [ln.split(" I ", 1)[-1] for ln in tail if "sched timing" in ln]
            rec[f"{k}k"] = {"sched_timing": sched[-3:], "prompt_n": t.get("prompt_n"), "prompt_tps": t.get("prompt_per_second"),
                            "predicted_n": t.get("predicted_n"), "tps": t.get("predicted_per_second"),
                            "ms_per_token": t.get("predicted_per_token_ms"), "op_timing": op_table(tail)}
            print(json.dumps({k: {kk: rec[f'{k}k'][kk] for kk in ('prompt_n', 'tps', 'ms_per_token')}}), flush=True)
    finally:
        srv.stop()
    json.dump(rec, open(os.path.join(a.out, f"profile-{a.tag}.json"), "w"), indent=1)
    return 0


def cmd_logits(a) -> int:
    os.makedirs(a.out, exist_ok=True)
    text = corpus()
    got = {}
    new_env = dict(kv.split("=", 1) for kv in (a.env or []))
    for tag, b, extra_env in (("ref", a.ref_bin or a.bin, {}), ("new", a.bin, new_env)):
        srv = Server(b, max(a.ctx), os.path.join(a.out, f"logits-{a.tag}-{tag}.server.log"), extra_env,
                     a.threads, a.extra.split() if a.extra else [])
        try:
            got[tag] = {}
            for k in a.ctx:
                r = completion(srv.base, text[: k * 1024 * 4], a.n, n_probs=8)
                got[tag][k] = [(p.get("content") if "content" in p else p.get("token"),
                                [(q.get("tok_str") or q.get("token"), round(q.get("prob", 0.0), 6))
                                 for q in (p.get("probs") or p.get("top_probs") or [])])
                               for p in (r.get("completion_probabilities") or [])]
        finally:
            srv.stop()
    rep = {}
    for k in a.ctx:
        ref, new = got["ref"][k], got["new"][k]
        same = 0
        for x, y in zip(ref, new):
            if x[0] != y[0]:
                break
            same += 1
        diffs = []
        for x, y in zip(ref[:same], new[:same]):
            px, py = dict(x[1]), dict(y[1])
            common = set(px) & set(py)
            diffs.append(max((abs(px[t] - py[t]) for t in common), default=1.0))
        rep[f"{k}k"] = {"tokens": len(ref), "identical_prefix": same,
                        "max_top8_prob_diff": max(diffs) if diffs else None}
    print(json.dumps(rep, indent=1))
    json.dump({"report": rep, "bins": {"ref": a.ref_bin, "new": a.bin}},
              open(os.path.join(a.out, f"logits-{a.tag}.json"), "w"), indent=1)
    return 0


def selftest() -> int:
    ok = True
    e = env_a4000()
    ok &= e["CUDA_VISIBLE_DEVICES"] == A4000_UUID
    ok &= "--n-cpu-moe" in args_for(4, 8) and "48" in args_for(4, 8)
    t = op_table(["x op timing over 100 graphs: 1.254 ms per graph", "  0.085 ms/graph   6.8%  x2    MUL_MAT bf16"])
    ok &= t is not None and t["top"][0]["op"].startswith("MUL_MAT")
    print("all passed" if ok else "FAIL")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", nargs="?", choices=["profile", "logits"])
    ap.add_argument("--bin")
    ap.add_argument("--ref-bin")
    ap.add_argument("--ctx", type=int, nargs="*", default=[4, 32])
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--n-predict", type=int, default=256, help="profile: tokens decoded at each context")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--no-op-timing", action="store_true", help="without 0010's per-op events (they slow every op)")
    ap.add_argument("--env", nargs="*", help="KEY=VALUE for the server")
    ap.add_argument("--extra", default="", help="extra server arguments, one string")
    ap.add_argument("--out", default="C:/Users/jwals/octo/a4000-dev")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    return {"profile": cmd_profile, "logits": cmd_logits}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
