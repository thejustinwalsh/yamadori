"""Where a conversation's KV cells land in the unified pool, and what that costs.

The pagoda-h6 failure (docs/ENGINES.md "KV placement"): after the client
compacted, the new conversation's prompt was prefilled while the old,
abandoned one still held the bottom of the pool, so every cell of the new
one landed ABOVE the tiered-KV line (--kv-vram-cells, 0028), in pinned host
RAM; when the old one was cleared, the new one's cells stayed where they
were, and every decode step of the rest of the run staged the host tail over
PCIe (6-12 tok/s at 67k-113k, where a conversation from cell 0 runs ~50).

This replays that sequence on ONE llama-server with EXACTLY config.yaml's
`bonsai` argv and env (the arm's binary, a spare loopback port), with the
window discipline of bench/engine_corruption.py (quiet stack, the worker's
gpu lane paused, production `bonsai` unloaded, a guard that kills the arm
the moment llama-swap starts anything on the card, production restored in
`finally`):

  A  slot 0: a --deep-token prompt (default 160k, past the line), decode --n
  B  slot 1: a DIFFERENT --new-token prompt (default 60k), decode --n
     (the post-compaction conversation, while the old one holds the pool)
  C  slot 0 cleared the way the proxy clears a slot (slots.release_idle: a
     one-token /completion on it, n_predict 0)
  D  slot 1 continued (its prompt + what it generated + 256 new tokens):
     decode --n (the h6 case: the old one is gone, the new one runs on)
  E  slot 1 cleared; slot 2: the same --new-token prompt from an empty pool,
     decode --n (the same conversation from cell 0: the reference)
  F  --idle placement rows (optional): an idle --idle-tokens slot kept, an
     active slot at each --depth, decode --n; the multi-slot case by depth
  G  --concurrent N (optional): slots 1 and 2 decode N-token prompts AT ONCE
     (one batch), each text scanned for the corruption symptoms, then each
     alone for comparison (identity informative only)

Decode tok/s is llama-server's own timings.predicted_per_second (greedy,
temperature 0, the same token prompts in every arm). Where the cells are:
the server runs with LLAMA_KV_CACHE_DEBUG=1 and -lv 5, whose find_slot line
prints the pool's highest used cell + 1 (`n`), the cells in use (`used`)
and the search head; a build with 0038+ also prints its own INFO lines
(`kv placement:` / `kv promote:`). The log is parsed per stage.

    python bench/kv_placement.py --window --out DIR \\
        --arm "current=C:/Users/jwals/engines/llama-bonsai2-ada-92ffd4db/src/build/bin/llama-server.exe"

Numbers only: no verdict.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_corruption as ec                                       # noqa: E402

ROOT = ec.ROOT
RE_FIND = re.compile(r"find_slot: stream\[(\d+)\], n = +(\d+), used = +(\d+), head = +(\d+), size = +(\d+)")
RE_OURS = re.compile(r"kv (placement|promote): .*")


def corpus_text(which: str) -> str:
    """Two disjoint texts from this repository: A = mcp/*.py, B = docs/**/*.md + bench/**/*.py."""
    if which == "A":
        files = sorted(glob.glob(os.path.join(ROOT, "mcp", "*.py")))
    else:
        files = (sorted(glob.glob(os.path.join(ROOT, "docs", "**", "*.md"), recursive=True))
                 + sorted(glob.glob(os.path.join(ROOT, "bench", "**", "*.py"), recursive=True)))
    parts = []
    for f in files:
        try:
            parts.append(f"\n\n# ==== {os.path.relpath(f, ROOT)} ====\n" + open(f, encoding="utf-8").read())
        except (OSError, UnicodeDecodeError):
            continue
    return "".join(parts)


class Stage:
    def __init__(self, logpath: str):
        self.logpath = logpath
        self.pos = 0

    def mark(self) -> None:
        self.pos = os.path.getsize(self.logpath) if os.path.exists(self.logpath) else 0

    def read(self) -> dict:
        """find_slot lines and our INFO lines since the last mark."""
        with open(self.logpath, encoding="utf-8", errors="replace") as f:
            f.seek(self.pos)
            text = f.read()
        self.pos += len(text.encode("utf-8", "replace"))
        finds = [tuple(int(x) for x in m.groups()) for m in RE_FIND.finditer(text)]
        ours = [m.group(0)[:400] for m in RE_OURS.finditer(text)]
        out: dict = {"find_slot_calls": len(finds), "ours": ours[-40:], "ours_n": len(ours)}
        if finds:
            out["pool_max_p1_last"] = finds[-1][1]
            out["pool_used_last"] = finds[-1][2]
            out["head_last"] = finds[-1][3]
            out["pool_max_p1_max"] = max(f[1] for f in finds)
        return out


def completion(base: str, prompt, slot: int, n: int, timeout: float = 3600) -> dict:
    body = {"prompt": prompt, "id_slot": slot, "n_predict": n, "cache_prompt": True, "temperature": 0.0,
            "top_k": 1, "seed": 0, "return_tokens": True}
    t0 = time.time()
    st, txt = ec.http("POST", base + "/completion", body, timeout=timeout)
    if st != 200:
        raise RuntimeError(f"/completion slot {slot}: HTTP {st}: {txt[:300]}")
    r = json.loads(txt)
    t = r.get("timings") or {}
    return {"slot": slot, "prompt_n": t.get("prompt_n"), "cache_n": t.get("cache_n"),
            "prompt_ms": t.get("prompt_ms"), "prompt_tps": t.get("prompt_per_second"),
            "predicted_n": t.get("predicted_n"), "tps": t.get("predicted_per_second"),
            "draft_n": t.get("draft_n"), "draft_n_accepted": t.get("draft_n_accepted"),
            "wall": round(time.time() - t0, 2), "tokens": r.get("tokens") or [],
            "content_head": (r.get("content") or "")[:120], "content": r.get("content") or ""}


def run_arm(name: str, binary: str, extra_env: dict, prod_argv: list[str], prod_env: dict, port: int,
            out: str, a) -> dict:
    cmd = ec.arm_command(prod_argv, binary, port, dict(a.arg_overrides.get(name, {}), **{"-lv": "5"}))
    env = dict(os.environ)
    for k in ("GGML_CUDA_PDL", "GGML_CUDA_BATCH_INVARIANT", "GGML_CUDA_DISABLE_GRAPHS"):
        env.pop(k, None)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env.update(prod_env)
    env["LLAMA_KV_CACHE_DEBUG"] = "1"
    env.update(extra_env)
    rec: dict = {"arm": name, "binary": binary, "cmd": cmd,
                 "env": {k: env[k] for k in list(prod_env) + list(extra_env) + ["LLAMA_KV_CACHE_DEBUG"]},
                 "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "stages": {}}
    logpath = os.path.join(out, f"{name}.log")
    logf = open(logpath, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)
    guard = ec.Guard(proc, ec.card_models())
    guard.start()
    base = f"http://127.0.0.1:{port}"
    stage = Stage(logpath)
    try:
        t0 = time.time()
        while http_ok(base) is False:
            if proc.poll() is not None:
                raise RuntimeError(f"server exited {proc.returncode} during load ({guard.tripped})")
            if time.time() - t0 > 600:
                raise RuntimeError("load timeout")
            time.sleep(1)
        rec["load_s"] = round(time.time() - t0, 1)
        rec["gpu_loaded"] = ec.gpu()
        ec.log(f"[{name}] loaded in {rec['load_s']} s; card {rec['gpu_loaded']}")

        def tok(text: str, n: int) -> list[int]:
            st, txt = ec.http("POST", base + "/tokenize", {"content": text[: n * 6], "add_special": False},
                              timeout=600)
            ids = json.loads(txt)["tokens"]
            if len(ids) < n:
                raise RuntimeError(f"corpus too short: {len(ids)} < {n} tokens")
            return ids[:n]

        pa = tok(corpus_text("A"), a.deep_tokens)
        pb_all = tok(corpus_text("B"), a.new_tokens + 256)
        pb, pb_more = pb_all[: a.new_tokens], pb_all[a.new_tokens:]

        def run(label: str, fn):
            stage.mark()
            ec.log(f"[{name}] {label} ...")
            r = fn()
            time.sleep(0.5)
            r.update(stage.read())
            r["gpu"] = ec.gpu()
            toks = r.pop("tokens", None)
            rec["stages"][label] = r
            ec.log(f"[{name}] {label}: prompt_n={r.get('prompt_n')} ({r.get('prompt_tps') or 0:.0f} tok/s) "
                   f"decode {r.get('tps') or 0:.2f} tok/s acc {r.get('draft_n_accepted')}/{r.get('draft_n')} "
                   f"pool_max_p1={r.get('pool_max_p1_last')} used={r.get('pool_used_last')} ours={r['ours'][-3:]}")
            if guard.tripped:
                raise RuntimeError(guard.tripped)
            return toks

        if a.only_g:
            a.idle_tokens = 0
        else:
            run("A_deep_slot0", lambda: completion(base, pa, 0, a.n))
            gen_b = run("B_new_slot1_while_slot0_holds", lambda: completion(base, pb, 1, a.n))
            run("C_clear_slot0", lambda: completion(base, pa[:1], 0, 0))
            run("D_slot1_continues_after_clear", lambda: completion(base, pb + (gen_b or []) + pb_more, 1, a.n))
            run("E_clear_slot1", lambda: completion(base, pb[:1], 1, 0))
            run("E_same_prompt_from_empty_pool_slot2", lambda: completion(base, pb, 2, a.n))
        if a.idle_tokens:
            # F: the multi-slot case by depth: an idle slot kept, an active one at each depth
            run("F_clear_slot2", lambda: completion(base, pb[:1], 2, 0))
            run(f"F_idle_{a.idle_tokens}_slot0", lambda: completion(base, pa[: a.idle_tokens], 0, 8))
            for d in a.depth:
                run(f"F_clear_slot1_before_{d}", lambda: completion(base, pb[:1], 1, 0))
                run(f"F_active_{d}_slot1", lambda d=d: completion(base, pb_all[:d] if d <= len(pb_all) else pb, 1, a.n))
        if a.concurrent:
            # G: two slots decoding at once (one batch, both sequences protected from each other's moves)
            # beside an idle one; each text is scanned for the corruption symptoms and compared with the
            # same request run alone afterwards (identity is informative: batch composition changes n_kv)
            import threading
            prompts = {1: pb_all[: a.concurrent], 2: pa[a.deep_tokens - a.concurrent: a.deep_tokens]}
            outs: dict = {}

            def one(slot: int) -> None:
                outs[slot] = completion(base, prompts[slot], slot, a.n)

            def both() -> dict:
                th = [threading.Thread(target=one, args=(s,)) for s in prompts]
                for t in th:
                    t.start()
                for t in th:
                    t.join()
                return {"tps": min((outs[s].get("tps") or 0) for s in prompts),
                        "per_slot": {s: {k: outs[s][k] for k in ("tps", "prompt_n", "draft_n", "draft_n_accepted")}
                                     for s in prompts},
                        "symptoms": {s: ec.scan(outs[s]["content"]) for s in prompts}}
            run("G_concurrent_slots_1_2", both)
            conc = {s: dict(outs[s]) for s in prompts}
            for s in prompts:
                run(f"G_clear_slot{s}", lambda s=s: completion(base, prompts[s][:1], s, 0))
                run(f"G_alone_slot{s}", lambda s=s: completion(base, prompts[s], s, a.n))
                alone = rec["stages"][f"G_alone_slot{s}"]
                txt_c = ec.scan(conc[s].get("content", ""))
                rec["stages"][f"G_alone_slot{s}"]["same_head_as_concurrent"] = (
                    alone.get("content") == conc[s].get("content"))
                rec["stages"][f"G_alone_slot{s}"]["concurrent_symptoms"] = txt_c
    except Exception as e:                                           # noqa: BLE001
        rec["error"] = repr(e)
        ec.log(f"[{name}] ERROR {e!r}")
    finally:
        guard.stop()
        rec["gpu_min_free"] = guard.min_free
        rec["guard"] = guard.tripped
        proc.terminate()
        try:
            proc.wait(30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(30)
        logf.close()
        rec["ended"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        with open(os.path.join(out, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=1)
        ec.log(f"[{name}] stopped; min free {guard.min_free} MiB")
    return rec


def http_ok(base: str) -> bool:
    return ec.http("GET", base + "/health", timeout=5)[0] == 200


def summary(recs: list[dict]) -> str:
    rows = ["| arm | stage | prompt_n | prompt tok/s | decode tok/s | draft acc | pool max+1 | used |",
            "|---|---|---|---|---|---|---|---|"]
    for r in recs:
        for k, s in (r.get("stages") or {}).items():
            acc = (f"{s['draft_n_accepted']}/{s['draft_n']}" if s.get("draft_n") else "-")
            rows.append(f"| {r['arm']} | {k} | {s.get('prompt_n')} | {s.get('prompt_tps') or 0:.0f} | "
                        f"{s.get('tps') or 0:.2f} | {acc} | {s.get('pool_max_p1_last')} | {s.get('pool_used_last')} |")
        if r.get("error"):
            rows.append(f"| {r['arm']} | ERROR | {r['error'][:120]} | | | | | |")
    return "\n".join(rows)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--window", action="store_true", help="the operator grants the window (production unloaded)")
    ap.add_argument("--arm", action="append", default=[], help="NAME=BINARY[,ENV=VAL...][,ARG:FLAG=VALUE...]")
    ap.add_argument("--out", required=False)
    ap.add_argument("--port", type=int, default=18092)
    ap.add_argument("--deep-tokens", type=int, default=160_000)
    ap.add_argument("--new-tokens", type=int, default=60_000)
    ap.add_argument("--n", type=int, default=256, help="tokens decoded per measured stage")
    ap.add_argument("--idle-tokens", type=int, default=0, help="F rows: an idle slot of this many tokens")
    ap.add_argument("--depth", type=int, action="append", default=[], help="F rows: active depths")
    ap.add_argument("--concurrent", type=int, default=0, help="G rows: two slots of this many tokens at once")
    ap.add_argument("--only-g", action="store_true", help="skip A-F (the G rows only)")
    ap.add_argument("--wait-quiet", type=float, default=30.0)
    ap.add_argument("--summarise", nargs="*")
    a = ap.parse_args(argv)
    if a.summarise:
        print(summary([json.load(open(p, encoding="utf-8")) for p in a.summarise]))
        return 0
    if not a.window or not a.out or not a.arm:
        print("refusing: needs --window (production is unloaded), --out and at least one --arm")
        return 2
    os.makedirs(a.out, exist_ok=True)
    prod_argv, prod_env = ec.production_command()
    arms = [ec.parse_arm(s) for s in a.arm]
    a.arg_overrides = {n: args for n, _, _, args in arms}
    recs, res = [], {}
    lane_hold = ec.hold_lane("KV placement repro (bench/kv_placement.py)", 12 * 3600)
    try:
        return _arms(a, arms, prod_argv, prod_env, recs, res)
    finally:
        ec.release_lane(lane_hold)


def _arms(a, arms, prod_argv, prod_env, recs, res) -> int:
    """(the lane is held by the caller for every arm and released at its one exit)"""
    for name, binary, env, _args in arms:
        why = ec.wait_quiet(prod_argv, a.wait_quiet * 60)
        if why:
            recs.append({"arm": name, "error": f"not run: {why}"})
            continue
        try:
            sys.path.insert(0, os.path.join(ROOT, "mcp"))
            st, txt = ec.http("POST", f"{ec.SWAP}/api/models/unload/{ec.PROD_ID}", timeout=120)
            ec.log(f"[{name}] unload {ec.PROD_ID}: HTTP {st}")
            t0 = time.time()
            while ec.PROD_ID in ec.running() or any("llama-server" in p for p in ec.card_pids()):
                if time.time() - t0 > 180:
                    raise RuntimeError("bonsai did not leave the card in 180 s")
                time.sleep(2)
            time.sleep(3)
            recs.append(run_arm(name, binary, env, prod_argv, prod_env, a.port, a.out, a))
        finally:
            res = ec.restore(prod_argv)
        if res.get("proxy_health") != 200:
            ec.log("stopping: :1234 is not healthy after restore")
            break
    print(summary(recs))
    with open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(summary(recs) + "\n")
    return 0 if all("error" not in r for r in recs) and res.get("proxy_health") == 200 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
