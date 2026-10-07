#!/usr/bin/env python
"""POST /decide-batch against a REAL llama-server on the CPU (no GPU, no stack port): the HTTP wire, the loop
thread, tokenization and the error paths, with the tiny random qwen35 model the engine test writes.

    # 1. the model (the engine's own test writes it; sensitive weights, tokenizer "no_vocab": token ids only)
    test-decide-batch.exe --save tiny-qwen35.gguf
    # 2. this script starts its own private servers on a loopback port and stops them
    python bench/decide_batch_server_smoke.py --server PATH/llama-server.exe --model tiny-qwen35.gguf

Needs a build that has patch 0042 (engines/patches/llama-bonsai2-ada/0042-server-decide-batch.patch). Never
touches :1234 / :11434 / :1235, config.yaml or the live stack; the servers are CPU-only (-ngl 0, no CUDA device
is visible). The reference is the SAME server's own /completion on the same tokens (cache_prompt false, one
token, n_probs): what jjava's per-read path asks of it (mcp/decider_bonsai.py ask_one), minus the chat template.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

N_VOCAB = 128
TOL = 5e-4          # engine test: float32 order-of-sum noise on the CPU, 1.4e-4 .. 2.3e-4 measured (docs/DECIDE-BATCH.md)

checks = {"n": 0, "failed": 0}


def check(cond: bool, what: str, detail: str = "") -> None:
    checks["n"] += 1
    if not cond:
        checks["failed"] += 1
        print(f"FAIL {what} {detail}")


def http(base: str, path: str, body=None, timeout: float = 120):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode() or "null")
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(raw)
        except ValueError:
            return e.code, {"raw": raw}


class Server:
    def __init__(self, exe: str, model: str, port: int, extra: list[str], b: int = 64, ub: int = 16):
        self.base = f"http://127.0.0.1:{port}"
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        cmd = [exe, "-m", model, "--host", "127.0.0.1", "--port", str(port), "-ngl", "0", "-c", "2048",
               "-b", str(b), "-ub", str(ub), "--no-webui", "-t", "4", "--no-warmup", "--no-cache-idle-slots",
               "--cache-ram", "0"] + extra
        self.log = open(f"decide_smoke_{port}.log", "w")
        self.p = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT, env=env)
        t0 = time.time()
        while time.time() - t0 < 120:
            if self.p.poll() is not None:
                raise RuntimeError(f"server exited {self.p.returncode}; see decide_smoke_{port}.log")
            try:
                s, _ = http(self.base, "/health", timeout=2)
                if s == 200:
                    return
            except Exception:                                    # noqa: BLE001
                pass
            time.sleep(0.5)
        raise RuntimeError("server did not become healthy")

    def stop(self):
        self.p.terminate()
        try:
            self.p.wait(10)
        except Exception:                                        # noqa: BLE001
            self.p.kill()
        self.log.close()


def reference(base: str, tokens: list[int], slot: int = 0) -> dict[int, float]:
    """The per-read path's answer for these tokens: the top-20 log-probs of the next token."""
    s, d = http(base, "/completion", {"prompt": tokens, "n_predict": 1, "n_probs": 20, "temperature": 0,
                                     "cache_prompt": False, "id_slot": slot, "samplers": ["temperature"]})
    assert s == 200, (s, d)
    cp = d.get("completion_probabilities") or d.get("probs") or []
    assert cp, d
    top = cp[0].get("top_logprobs") or cp[0].get("top_probs") or []
    out = {}
    for e in top:
        lp = e.get("logprob")
        if lp is None and "prob" in e:
            lp = math.log(max(e["prob"], 1e-30))
        out[int(e["id"])] = float(lp)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=19876)
    ap.add_argument("-b", type=int, default=64, help="server -b (the stack runs 1024)")
    ap.add_argument("-ub", type=int, default=16, help="server -ub (the stack runs 512)")
    a = ap.parse_args()
    rnd = random.Random(20261007)

    def toks(n):
        return [rnd.randrange(N_VOCAB) for _ in range(n)]

    # ---------------------------------------------------------------- a server without --decide-seqs
    s0 = Server(a.server, a.model, a.port, ["-np", "2", "--kv-unified"], a.b, a.ub)
    try:
        st, d = http(s0.base, "/decide-batch", {"prefix": toks(5), "groups": [[toks(3)]]})
        check(st == 501, "no --decide-seqs: 501", f"got {st} {d}")
        st, props = http(s0.base, "/props")
        check(st == 200 and props.get("decide_batch", {}).get("seqs") == 0, "props says decide_batch.seqs 0", str(props.get("decide_batch")))
    finally:
        s0.stop()

    # ---------------------------------------------------------------- the real thing
    W = 4
    s1 = Server(a.server, a.model, a.port + 1, ["-np", "2", "--kv-unified", "--decide-seqs", str(W)], a.b, a.ub)
    base = s1.base
    try:
        st, props = http(base, "/props")
        check(props.get("decide_batch", {}).get("seqs") == W, "props decide_batch.seqs", str(props.get("decide_batch")))

        prefix = toks(48)
        groups = [[toks(10)], [toks(7), toks(9)], [toks(14)], [toks(5), toks(5)]]
        # a group's blocks that share a head (the two orders of a question)
        head = toks(12)
        groups.append([head + toks(4), head + toks(6)])
        ids = list(range(N_VOCAB))
        st, d = http(base, "/decide-batch", {"prefix": prefix, "groups": groups, "token_ids": ids,
                                              "top_logprobs": 20, "keep_prefix": True})
        check(st == 200, "decide-batch 200", f"{st} {str(d)[:300]}")
        if st == 200:
            check(d["prefix"] == {"tokens": 48, "reused": 0, "processed": 48, "resident": True}, "prefix record", str(d["prefix"]))
            worst, n = 0.0, 0
            for g, blocks in enumerate(groups):
                for b, blk in enumerate(blocks):
                    ref = reference(base, prefix + blk)
                    row = d["groups"][g][b]
                    got = {e["id"]: e["logprob"] for e in row["top_logprobs"]}
                    common = set(ref) & set(got)
                    check(len(common) >= 15, f"block {g}/{b}: top-20 overlap", f"{len(common)}")
                    for i in common:
                        worst = max(worst, abs(ref[i] - got[i]))
                    check(max(ref, key=ref.get) == max(got, key=got.get), f"block {g}/{b}: argmax equals the per-read path")
                    check(set(map(int, row["logprobs"])) == set(ids), f"block {g}/{b}: requested ids returned")
                    check(abs(row["logprobs"][str(max(got, key=got.get))] - max(got.values())) < 1e-6, "requested logprob equals the top entry")
                    n += 1
            check(worst <= TOL, "logprobs within tolerance of the per-read path", f"worst {worst:.3e} over {n} blocks")
            print(f"  wire: {n} blocks, worst |dlogprob| vs the server's own /completion = {worst:.3e} (tolerance {TOL})")
            print(f"  timings: {json.dumps(d['timings'])}")
            check(d["groups"][4][0]["shared"] >= 12, "shared head detected", str(d["groups"][4][0]))

        # prefix kept: the next call reuses it whole; an extension continues it
        st, d2 = http(base, "/decide-batch", {"prefix": prefix, "groups": [[toks(6)]], "keep_prefix": True})
        check(st == 200 and d2["prefix"]["reused"] == 48 and d2["prefix"]["processed"] == 0, "resident prefix reused", str(d2.get("prefix")))
        st, d3 = http(base, "/decide-batch", {"prefix": prefix + toks(5), "groups": [[toks(6)]], "keep_prefix": False})
        check(st == 200 and d3["prefix"]["reused"] == 48 and d3["prefix"]["processed"] == 5 and not d3["prefix"]["resident"], "extension continues, then freed", str(d3.get("prefix")))
        st, d4 = http(base, "/decide-batch", {"prefix": prefix, "groups": [[toks(6)]]})
        check(st == 200 and d4["prefix"]["reused"] == 0, "freed prefix is processed again", str(d4.get("prefix")))
        st, d5 = http(base, "/decide-batch", {"release": True})
        check(st == 200 and d5.get("released") is True, "release")

        # refusals
        # a group with more blocks than forks is cut into chunks: still correct
        big = [toks(6) for _ in range(W + 1)]
        st, d = http(base, "/decide-batch", {"prefix": prefix, "groups": [big]})
        check(st == 200 and len(d["groups"][0]) == W + 1, "a group larger than the forks is split, not refused", f"{st} {str(d)[:200]}")
        if st == 200:
            w2 = 0.0
            for b, blk in enumerate(big):
                ref = reference(base, prefix + blk)
                got = {e["id"]: e["logprob"] for e in d["groups"][0][b]["top_logprobs"]}
                w2 = max([w2] + [abs(ref[i] - got[i]) for i in set(ref) & set(got)])
            check(w2 <= TOL, "the split group's rows equal the per-read path", f"{w2:.3e}")
        st, d = http(base, "/decide-batch", {"prefix": prefix + [N_VOCAB + 5], "groups": [[toks(3)]]})
        check(st == 400, "token outside the vocabulary: 400", f"{st} {d}")
        st, d = http(base, "/decide-batch", {"groups": [[toks(3)]]})
        check(st == 400, "no prefix: 400", f"{st} {d}")
        st, d = http(base, "/decide-batch", {"prefix": prefix, "groups": [[[]]]})
        check(st in (400, 500) and st != 200, "an empty block is refused", f"{st} {d}")
        st, d = http(base, "/decide-batch", {"prefix": prefix, "groups": [[toks(3)]], "token_ids": [N_VOCAB + 1]})
        check(st == 400, "label id outside the vocabulary: 400", f"{st} {d}")
        try:
            req = urllib.request.Request(base + "/decide-batch", data=b"{not json", headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=10)
            check(False, "malformed JSON refused")
        except urllib.error.HTTPError as e:
            check(e.code == 400, "malformed JSON: 400", str(e.code))
        st, d = http(base, "/decide-batch", {"prefix": prefix, "groups": [[toks(3)]]})
        check(st == 200, "the server still answers after the refusals", f"{st}")

        # a slot decoding while batches run: its tokens equal an uninterrupted run (the recurrent state of slot 0
        # is not disturbed by the scratch sequences)
        gen_prompt = toks(30)
        body = {"prompt": gen_prompt, "n_predict": 48, "temperature": 0, "cache_prompt": False, "id_slot": 0,
                "samplers": ["temperature"], "n_probs": 0}
        st, quiet = http(base, "/completion", body)
        check(st == 200, "quiet generation")
        out: dict = {}

        def gen():
            out["r"] = http(base, "/completion", body)

        th = threading.Thread(target=gen)
        th.start()
        n_batches = 0
        while th.is_alive():
            http(base, "/decide-batch", {"prefix": prefix, "groups": [[toks(8), toks(8)], [toks(11)]], "keep_prefix": n_batches % 2 == 0})
            n_batches += 1
        th.join()
        st2, busy = out["r"]
        t_quiet = quiet.get("content") if "content" in quiet else quiet.get("tokens")
        t_busy = busy.get("content") if "content" in busy else busy.get("tokens")
        check(st2 == 200 and t_quiet == t_busy, "generation beside batches equals the quiet run", f"{n_batches} batches in between")
        print(f"  interleaved: {n_batches} decide-batch calls while one slot generated 48 tokens; identical output = {t_quiet == t_busy}")
        http(base, "/decide-batch", {"release": True})
    finally:
        s1.stop()

    print(f"decide_batch_server_smoke: {checks['n']} checks, {checks['failed']} failed")
    return 0 if checks["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
