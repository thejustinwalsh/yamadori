#!/usr/bin/env python
"""THE REAL MODEL ON THE CPU: sequential reads vs one /decide-batch call, same server, real Bonsai weights, real chat
template, real tokenizer. No GPU, no stack port: it talks to a private CPU llama-server you started with patch 0042, e.g.

    CUDA_VISIBLE_DEVICES= llama-server.exe -m Ternary-Bonsai-2-27B-PTQ1_0.gguf -ngl 0 -np 2 --kv-unified -c 8192 -b 1024 -ub 512 \
        --jinja --reasoning-format deepseek --no-context-shift --decide-seqs 2 -t 8 --port 19900 --cache-ram 0
    python bench/decider/batch_cpu_real.py --base http://127.0.0.1:19900

It runs decider_bonsai.read (the per-read path) and decider_batch.read_many on the same state and questions and prints,
per question, the largest probability difference, the answers, and what the engine reported.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "mcp"))
os.environ.setdefault("YAMADORI_DECIDER_BATCH", "1")

STATE = ("Customer: my payout failed three times this week and I need it fixed before Friday.\n"
         "Support: have you tried a different card?\n"
         "Customer: yes, same error. Can you just refund the fees instead?")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:19900")
    ap.add_argument("--slot", type=int, default=1)
    a = ap.parse_args()

    import decider_bonsai as D
    import decider_batch as B

    def call(path, payload=None, timeout=3600):
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(a.base + path, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode() or "null")

    def upstream(path, payload=None, timeout=3600):
        return call(path, payload, timeout)

    def post(b, timeout=3600):
        return call("/v1/chat/completions", b, timeout)

    D.model_name = lambda: "bonsai-cpu"                      # the server's own model; no max_mode / llama-swap involved
    qs = [D.q_noul("angry", "Is the customer angry?"),
          D.q_choice("ask", "What does the customer want?", ["a refund of the fees", "a different card", "an apology"],
                     keys=["refund", "card", "apology"]),
          D.q_noul("fixed", "Has the problem been fixed already?")]
    t0 = time.time()
    seq = [D.read(STATE, q, slot=a.slot, post=post, upstream=upstream, cache=True) for q in qs]
    t_seq = time.time() - t0
    print(f"sequential: {t_seq:.1f} s for {len(qs)} questions ({sum(x['diagnostics']['reads'] for x in seq)} reads)")
    t0 = time.time()
    bat = B.read_many(STATE, qs, slot=a.slot, post=post, upstream=upstream, cache=True)
    t_bat = time.time() - t0
    print(f"batched:    {t_bat:.1f} s, one call; stats {json.dumps(B.stats(), default=str)[:600]}")
    worst = 0.0
    flips = 0
    for q, s, b in zip(qs, seq, bat):
        ps, pb = D.answer_probs(s), D.answer_probs(b)
        d = max(abs(ps[k] - pb[k]) for k in ps)
        worst = max(worst, d)
        ans_s = s.get("choice") or ("yes" if s.get("noul", 0) >= 0.5 else "no")
        ans_b = b.get("choice") or ("yes" if b.get("noul", 0) >= 0.5 else "no")
        flips += ans_s != ans_b
        print(f"  {q['name']:6s} seq {ps}  batch {pb}  max|dp| {d:.2e}  read_path {b['diagnostics'].get('read_path')}")
    print(f"worst |dp| {worst:.3e} (TIE_BAND {D.TIE_BAND}), answer flips {flips}/{len(qs)}")
    print("last call:", json.dumps(B.last_call(), default=str)[:700])
    return 0


if __name__ == "__main__":
    sys.exit(main())
