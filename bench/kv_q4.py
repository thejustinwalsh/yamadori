"""q4_0 K/V on Bonsai at layout v3 (-np 2), against q8_0 -- the coordinator's ask of 2026-09-30, in the Bonsai window.
Bonsai's engine already applies upstream's Hadamard rotation to quantized KV (engines/src/llama-bonsai2-ada/src/
llama-kv-cache.cpp:349-373, on by default), so q4_0 is a flag change. q4_0 was rolled back on 2026-09-25 at ~9 tok/s
at depth (a different engine and layout then). Nothing here is a claim: each number is one run of the step named.

  fit      the FULL trained window at q4_0: -c 262,144, -np 2 --kv-unified, --kv-vram-cells 262,144, warmed with an
           8K prefill + decode on slot 0 and a lane state on slot 1; min free VRAM from the guard. FITS when it keeps
           the 1,000 MiB margin (engine_phase2.margin_mib, the same as kv_rank's fit).
  speed    decode tok/s on slot 0 at 8K / 64K / 128K, n=3, from llama-server's timings -- both arms.
  drift    TOP-20 APPROXIMATE KL of the next-token distribution, q4_0 vs q8_0, teacher-forced by the TEXT: 8 chunks x
           4,096 tokens of repository text, the distribution read at 16 cut points per chunk (n_probs 20, n_predict 1,
           greedy; each chunk's prefixes extend the cache). KL over the union of both top-20s, renormalised, missing
           mass floored (an approximation: llama-perplexity is not built for this engine). Plus the top-1 agreement.
  needles  5 needles at 32K and 128K, thinking OFF, greedy -- both arms (PASS 5/5 per depth).
The q8_0 arm is layout v3's own argv at its line (--q8-line N, from bench/kv_rank.py --layout v3).

    python bench/kv_q4.py --window --out DIR --q8-line N

The gpu lane is held for the whole run (engine_corruption.hold_lane / release_lane, the error path included); each
arm is one engine_phase2.in_window (quiet stack, production unloaded and restored after); --attempts 1: a client
request that trips the guard stops the arm, and the run ends after restoring production.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import engine_corruption as ec                                       # noqa: E402
import engine_phase2 as ep                                           # noqa: E402
import kv_rank as kr                                                 # noqa: E402

FULL = 262144
DEPTHS = (8192, 65536, 131072)
KL_CHUNKS, KL_LEN, KL_CUTS, TOPK = 8, 4096, 16, 20
NEEDLE_DEPTHS = (32768, 131072)


def args_for(kv: str, c: int) -> dict:
    a = kr.target_args(c, layout="v3")
    a["--cache-type-k"] = kv
    a["--cache-type-v"] = kv
    return a


def _probs(base: str, tokens: list[int]) -> dict[int, float]:
    d = ep.post(base, "/completion", {"prompt": tokens, "n_predict": 1, "id_slot": 0, "cache_prompt": True,
                                      "temperature": 0, "top_k": 1, "n_probs": TOPK, "seed": 0,
                                      "kv_rank": 2, "kv_ranks": [2, kr.RANK_LANE]})
    cp = (d.get("completion_probabilities") or [{}])[0]
    # the server's n_probs shapes: newer builds {id, token, logprob, top_logprobs: [{id, token, logprob}]}, older
    # {content, probs: [{tok_str, prob}]} / top_probs
    rows = cp.get("top_logprobs") or cp.get("top_probs") or cp.get("probs") or []
    if not rows and d.get("_raw_debug"):
        pass
    out = {}
    for r in rows:
        tid = r.get("id", r.get("tok"))
        p = r.get("prob")
        if p is None and r.get("logprob") is not None:
            p = math.exp(r["logprob"])
        if tid is not None and p is not None:
            out[int(tid)] = float(p)
    return out


DRIFT_ONLY = {"on": False}


def arm_fn(kv: str, fit: bool):
    def fn(base: str, rec: dict, guard) -> None:
        toks = ep.corpus_tokens(base)
        need = max(DEPTHS) + 8192
        toks = (toks * (need // max(len(toks), 1) + 1))[:need]
        if DRIFT_ONLY["on"]:
            kr.clear(base, 2)
            probe = ep.post(base, "/completion", {"prompt": toks[:64], "n_predict": 1, "id_slot": 0, "n_probs": 3,
                                                  "temperature": 0, "top_k": 1})
            rec["probe_shape"] = json.dumps(probe.get("completion_probabilities"))[:400]
            rec["dists"] = [{str(t): p for t, p in _probs(base, toks[c * KL_LEN:(c + 1) * KL_LEN][:k * KL_LEN // KL_CUTS]).items()}
                            for c in range(KL_CHUNKS) for k in range(1, KL_CUTS + 1)]
            return
        if fit:
            kr.line_warm_v3(base, rec, guard)
        speed = {}
        for d in DEPTHS:
            kr.clear(base, 2)
            kr.gen(base, 0, toks[:d], 1, 2, [2, kr.RANK_LANE])
            speed[str(d)] = [kr.strip(kr.gen(base, 0, toks[:d], kr.GEN, 2, [2, kr.RANK_LANE])) for _ in range(3)]
            ec.log(f"[{kv}] speed {d}: {[x.get('predicted_per_second') for x in speed[str(d)]]}")
        rec["speed"] = speed
        kr.clear(base, 2)
        dists = []
        for c in range(KL_CHUNKS):
            chunk = toks[c * KL_LEN:(c + 1) * KL_LEN]
            for k in range(1, KL_CUTS + 1):
                dists.append({str(t): p for t, p in _probs(base, chunk[:k * KL_LEN // KL_CUTS]).items()})
        rec["dists"] = dists
        import mirai_s_gate as mg
        text = mg.corpus()
        port = int(base.rsplit(":", 1)[1])
        needles = {}
        for depth in NEEDLE_DEPTHS:
            hits = 0
            for i in range(5):
                key = hashlib.sha256(f"{depth}-{i}".encode()).hexdigest()[:8].upper()
                kr.clear(base, 2)
                r = ec.Server(port).chat([{"role": "user", "content": mg.needle_prompt(depth, key, text[i * 997:])}],
                                         max_tokens=64, thinking=False, sampling={"temperature": 0.0, "top_k": 1},
                                         seed=0)
                hits += key in (r["choices"][0]["message"].get("content") or "")
            needles[str(depth)] = hits
            ec.log(f"[{kv}] needles {depth}: {hits}/5")
        rec["needles"] = needles
    return fn


def kl(p: dict, q: dict, floor: float = 1e-6) -> float:
    keys = set(p) | set(q)
    ps = sum(p.values()) or 1.0
    qs = sum(q.values()) or 1.0
    return sum((p.get(k, floor) / ps) * math.log((p.get(k, floor) / ps) / (q.get(k, floor) / qs)) for k in keys)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--window", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--q8-line", type=int, required=True)
    ap.add_argument("--port", type=int, default=18094)
    ap.add_argument("--drift-only", action="store_true", help="only the drift reads, both arms (a rerun)")
    a = ap.parse_args(argv)
    if not a.window:
        print("refusing: --window (a granted window)")
        return 2
    os.makedirs(a.out, exist_ok=True)
    DRIFT_ONLY["on"] = a.drift_only
    ep.LANE_BY = kr.LANE_BY
    hold = ec.hold_lane("kv_q4 (bench/kv_q4.py)", 4 * 3600)
    ep._LANE_HOLD = hold
    res: dict = {}
    try:
        s = ep.setup()
        s["port"], s["arms"] = a.port, {}
        s["arms"]["q4"] = (kr.CANDIDATE, {}, args_for("q4_0", FULL))
        s["arms"]["q8"] = (kr.CANDIDATE, {}, args_for("q8_0", a.q8_line))
        margin, _ = ep.margin_mib()
        for name, fit in (("q4", True), ("q8", False)):
            rec, restored = ep.in_window(name, s, arm_fn(s["arms"][name][2]["--cache-type-k"], fit), a.out, 1, 1)
            res[name] = {k: v for k, v in rec.items() if not str(k).startswith("_")}
            res[name]["restored"] = restored
            json.dump(res, open(os.path.join(a.out, "kv_q4_drift.json" if a.drift_only else "kv_q4.json"), "w",
                                encoding="utf-8"), indent=1, default=str)
            if rec.get("error") or rec.get("guard"):
                break
        q4, q8 = res.get("q4") or {}, res.get("q8") or {}
        summ = {"fit_full_262k_q4": {"min_free_mib": q4.get("gpu_min_free"), "margin_mib": margin,
                                     "fits": (q4.get("gpu_min_free") or -1) >= margin}}
        for d in DEPTHS:
            summ[f"tps_{d}"] = {arm: kr.med((x.get("speed") or {}).get(str(d)) or []) for arm, x in (("q4", q4),
                                                                                                  ("q8", q8))}
        if q4.get("dists") and q8.get("dists"):
            ks = [kl(p, q) for p, q in zip(q8["dists"], q4["dists"])]
            top1 = [max(p, key=p.get) == max(q, key=q.get) for p, q in zip(q8["dists"], q4["dists"]) if p and q]
            summ["drift"] = {"kl_mean": round(sum(ks) / len(ks), 5), "kl_max": round(max(ks), 5), "n": len(ks),
                             "top1_agree": f"{sum(top1)}/{len(top1)}",
                             "how": "top-20 approximate KL(q8 || q4), 8 chunks x 16 cuts"}
        summ["needles"] = {"q4": q4.get("needles"), "q8": q8.get("needles")}
        res["summary"] = summ
    finally:
        res["lane"] = ec.release_lane(hold)
        json.dump(res, open(os.path.join(a.out, "kv_q4_drift.json" if a.drift_only else "kv_q4.json"), "w",
                            encoding="utf-8"), indent=1, default=str)
    print(json.dumps(res.get("summary"), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
