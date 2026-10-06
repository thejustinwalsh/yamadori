"""Why is Flash-Next's first prompt after a load slow? Measures a ~24.6K-token fresh prompt on a freshly loaded
flash-next (arm a) and a second, different fresh prompt on the same loaded server (arm b), n reps, with the Windows
counters sampled at ~1 s beside it (bench/fn_first_prompt_sampler.ps1) and the llama-server's own log lines (per-batch
timings) read from llama-swap's /logs/stream/flash-next.

    python bench/fn_first_prompt.py calibrate                      # needs flash-next loaded: tokens per character
    python bench/fn_first_prompt.py run --out DIR --reps 3 [--arms a,b] [--target 24600]
    python bench/fn_first_prompt.py summarize DIR                  # tables from the raw records

Requests go to llama-swap (:11434) with the model name `flash-next`, streamed, `max_tokens` 24 (the prefill is what is
measured; the prompt shape is the soak's: a system message of numbered guidelines built from bench/harness_soak.py's
own tables, then the soak's first user task). Not through the proxy at :1234: that needs an account key and this run was
given none; the proxy adds heartbeats and a tool list, not prefill work. The prompts differ from the first token (a
random header and a seeded permutation of the guideline table), so nothing is reused (usage reports cached tokens).

Records: DIR/requests.jsonl (one row a request), DIR/samples.jsonl (the sampler), DIR/serverlog.jsonl (the server's
lines with the time they arrived), DIR/summary.txt (summarize).
"""
from __future__ import annotations

import argparse
import http.client
import itertools
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
import uuid
from urllib.parse import urlsplit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import harness_soak as hs                                                                # noqa: E402

SWAP = os.environ.get("YAMADORI_SWAP", "http://127.0.0.1:11434")
MODEL = "flash-next"
PS = r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"


# ------------------------------------------------------------------------------------------------ http helpers
def _conn(base: str, timeout: float):
    u = urlsplit(base)
    return http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)


def http_json(method: str, path: str, body=None, timeout: float = 60.0):
    c = _conn(SWAP, timeout)
    raw = json.dumps(body).encode() if body is not None else None
    c.request(method, path, body=raw, headers={"Content-Type": "application/json"} if raw else {})
    r = c.getresponse()
    data = r.read()
    c.close()
    try:
        return r.status, json.loads(data) if data else None
    except ValueError:
        return r.status, data.decode("utf-8", "replace")


def running() -> list[dict]:
    st, j = http_json("GET", "/running", timeout=10)
    return (j or {}).get("running", []) if isinstance(j, dict) else []


def loaded(model: str = MODEL) -> bool:
    return any(m.get("model") == model and m.get("state") == "ready" for m in running())


# ------------------------------------------------------------------------------------------------ the prompts
def build_prompt(seed: int, target_tokens: int, cpt: float) -> tuple[str, str]:
    """A system message of ~target_tokens (at cpt characters a token) and the soak's first user task. The guideline
    table's order is permuted by `seed` and a random header opens the text: two seeds share no prefix past the chat
    template's own opening tokens."""
    rng = random.Random(seed)
    nonce = uuid.UUID(int=rng.getrandbits(128)).hex
    head = (f"Session {nonce}. You are an expert coding agent working inside the user's editor.\n"
            "You can read, search and edit files in the open workspace and run commands in its terminal. Work in small "
            "verified steps, and finish by telling the user what you did and what you saw.\n\n# Guidelines\n")
    combos = list(itertools.product(hs._SITUATIONS, hs._LANGS, hs._ACTIONS))
    rng.shuffle(combos)
    target_chars = int(target_tokens * cpt)
    lines, size, i = [], len(head), 0
    while size < target_chars:
        sit, lang, (act, then) = combos[i % len(combos)]
        line = (f"Guideline {rng.randrange(10**6)}: when {sit} in a {lang} project, {act} before you {then}. Keep the "
                "step small, report what you saw in one sentence, and prefer the result of a tool over a recollection "
                "of how the project usually works.")
        lines.append(line)
        size += len(line) + 1
        i += 1
    user = hs.FIRST_TASKS[0] if hasattr(hs, "FIRST_TASKS") else \
        "Create a small TypeScript module in this workspace and run its tests."
    return head + "\n".join(lines), user


def tokenize(text: str) -> int:
    st, j = http_json("POST", f"/upstream/{MODEL}/tokenize", {"content": text, "add_special": False}, timeout=120)
    if st != 200 or not isinstance(j, dict):
        raise RuntimeError(f"tokenize failed: HTTP {st} {str(j)[:200]}")
    return len(j.get("tokens", []))


def cmd_calibrate(a) -> int:
    if not loaded():
        print("flash-next is not loaded; load it first (this never loads it)")
        return 1
    sys_text, _ = build_prompt(1, a.target, 3.6)
    n = tokenize(sys_text)
    cpt = len(sys_text) / n
    print(f"seed 1: {len(sys_text)} chars -> {n} tokens: {cpt:.4f} chars a token")
    for cand in (cpt, cpt * 0.995):
        s2, _ = build_prompt(2, a.target, cand)
        print(f"  cpt {cand:.4f}: seed 2 -> {tokenize(s2)} tokens")
    return 0


# ------------------------------------------------------------------------------------------------ the server log
class LogTap(threading.Thread):
    """Reads llama-swap's /logs/stream/flash-next, stamping each line with the time it arrived. A line already seen
    (the stream replays the buffer on connect) is dropped."""

    def __init__(self, path: str):
        super().__init__(daemon=True)
        self.path = path
        self.stop = False
        self.seen: set[str] = set()
        self.fh = open(path, "w", encoding="utf-8")
        self.lock = threading.Lock()
        self.lines: list[tuple[float, str]] = []

    def run(self) -> None:
        while not self.stop:
            try:
                c = _conn(SWAP, 30)
                c.request("GET", f"/logs/stream/{MODEL}")
                r = c.getresponse()
                if r.status != 200:
                    c.close()
                    time.sleep(1.0)
                    continue
                buf = b""
                while not self.stop:
                    chunk = r.read1(65536)
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n" in buf:
                        ln, buf = buf.split(b"\n", 1)
                        s = ln.decode("utf-8", "replace").rstrip("\r")
                        if not s:
                            continue
                        now = time.time()
                        with self.lock:
                            if s in self.seen:
                                continue
                            self.seen.add(s)
                            self.lines.append((now, s))
                            self.fh.write(json.dumps({"t": round(now, 3), "line": s}) + "\n")
                            self.fh.flush()
                c.close()
            except Exception:                                                              # noqa: BLE001
                time.sleep(1.0)

    def since(self, t0: float, t1: float) -> list[tuple[float, str]]:
        with self.lock:
            return [(t, s) for t, s in self.lines if t0 <= t <= t1]


# ------------------------------------------------------------------------------------------------ one request
def one_request(label: str, system: str, user: str, tap: LogTap, rec_sink, timeout: float = 900.0,
                prep: dict | None = None) -> dict:
    body = {"model": MODEL, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": True, "stream_options": {"include_usage": True}, "max_tokens": 24, "reasoning_effort": "xhigh"}
    raw = json.dumps(body).encode()
    rec: dict = {"label": label, "body_bytes": len(raw), "mem_before": hs.mem_sample(), "prep": prep or {},
                 "running_before": [m["model"] for m in running()]}
    t_send = time.time()
    rec["t_send"] = round(t_send, 3)
    c = _conn(SWAP, timeout)
    t_head = t_first = None
    timings = usage = None
    n_delta = 0
    finish = None
    try:
        c.request("POST", "/v1/chat/completions", body=raw, headers={"Content-Type": "application/json"})
        r = c.getresponse()
        t_head = time.time()
        rec["status"] = r.status
        if r.status != 200:
            rec["error"] = r.read().decode("utf-8", "replace")[:400]
        else:
            buf = b""
            while True:
                chunk = r.read1(65536)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    ln, buf = buf.split(b"\n", 1)
                    s = ln.decode("utf-8", "replace").strip()
                    if not s.startswith("data:"):
                        continue
                    s = s[5:].strip()
                    if s == "[DONE]":
                        continue
                    try:
                        j = json.loads(s)
                    except ValueError:
                        continue
                    if j.get("timings"):
                        timings = j["timings"]
                    if j.get("usage"):
                        usage = j["usage"]
                    for ch in j.get("choices") or []:
                        d = ch.get("delta") or {}
                        if (d.get("content") or d.get("reasoning_content")) and t_first is None:
                            t_first = time.time()
                        if d.get("content") or d.get("reasoning_content"):
                            n_delta += 1
                        if ch.get("finish_reason"):
                            finish = ch["finish_reason"]
    except Exception as e:                                                                 # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
    finally:
        c.close()
    t_end = time.time()
    rec.update({"t_headers": round(t_head, 3) if t_head else None, "t_first_token": round(t_first, 3) if t_first else None,
                "t_end": round(t_end, 3), "ttft_s": round(t_first - t_send, 2) if t_first else None,
                "total_s": round(t_end - t_send, 2), "finish": finish, "n_deltas": n_delta, "usage": usage,
                "timings": timings, "mem_after": hs.mem_sample()})
    time.sleep(0.6)                                                                       # the log lines trail the stream
    rec["log"] = [{"t": round(t, 3), "line": s} for t, s in tap.since(t_send - 2.0, time.time())
                  if re.search(r"print_timing|checkpoint|layer-major|prompt processing|launch_slot|listening|"
                               r"model loaded|CUDA|error|warn|failed", s, re.I)]
    rec_sink(rec)
    return rec


# ------------------------------------------------------------------------------------------------ the cache arms
MODELS = "C:/Users/jwals/textgen/user_data/models/flash-next"
SHARD1 = MODELS + "/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf"        # attention, router, experts
EVICT = [MODELS + "/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf",          # a model file the stack never
         MODELS + "/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf"]           # loads: 61.9 GB through the cache


def read_files(paths: list[str], chunk: int = 16 << 20) -> dict:
    """Read each file once, front to back, with ordinary cached reads (what a pre-read of a model would do); the bytes
    are discarded. Returns seconds, bytes, and the machine's memory when it ended."""
    t0 = time.time()
    n = 0
    for p in paths:
        with open(p, "rb", buffering=0) as f:
            while True:
                b = f.read(chunk)
                if not b:
                    break
                n += len(b)
    dt = time.time() - t0
    return {"t0": round(t0, 3), "t1": round(t0 + dt, 3), "s": round(dt, 2), "GB": round(n / 1e9, 2),
            "GBps": round(n / 1e9 / dt, 2), "mem_after": hs.mem_sample()}


# ------------------------------------------------------------------------------------------------ the run
def unload_and_wait(model: str = MODEL, wait: float = 120.0) -> float:
    t0 = time.time()
    st, j = http_json("POST", f"/api/models/unload/{model}", timeout=wait)
    if st >= 300:
        raise RuntimeError(f"unload {model}: HTTP {st} {str(j)[:200]}")
    while loaded(model) or any(m.get("model") == model for m in running()):
        if time.time() - t0 > wait:
            raise RuntimeError("flash-next did not leave /running")
        time.sleep(0.5)
    return time.time() - t0


def cmd_run(a) -> int:
    os.makedirs(a.out, exist_ok=True)
    tag = a.tag
    stop = os.path.join(a.out, f"sampler{tag}.stop")
    if os.path.exists(stop):
        os.remove(stop)
    sampler = subprocess.Popen([PS, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                                os.path.join(HERE, "fn_first_prompt_sampler.ps1"), "-Out",
                                os.path.join(a.out, f"samples{tag}.jsonl"), "-StopFile", stop],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    tap = LogTap(os.path.join(a.out, f"serverlog{tag}.jsonl"))
    tap.start()
    fh = open(os.path.join(a.out, f"requests{tag}.jsonl"), "a", encoding="utf-8")

    def sink(rec):
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        t = rec.get("timings") or {}
        print(f"  {rec['label']}: HTTP {rec.get('status')} ttft {rec.get('ttft_s')} s, prompt {rec.get('usage', {}) and rec['usage'].get('prompt_tokens')}, "
              f"server prefill {t.get('prompt_per_second') and round(t['prompt_per_second'], 1)} tok/s "
              f"({t.get('prompt_n')} tok, {t.get('prompt_ms') and round(t['prompt_ms'] / 1000, 1)} s, cached {t.get('cache_n')}) {rec.get('error', '')}",
              flush=True)

    arms = a.arms.split(",")
    seed = a.seed
    try:
        time.sleep(3.0)
        for rep in range(1, a.reps + 1):
            print(f"== rep {rep}", flush=True)
            if "a" in arms:
                if loaded() or any(m.get("model") == MODEL for m in running()):
                    s = unload_and_wait()
                    print(f"  unloaded {MODEL} in {s:.1f} s; /running: {[m['model'] for m in running()]}", flush=True)
                time.sleep(a.settle)
                prep: dict = {"mem_after_unload": hs.mem_sample()}
                if a.evict:
                    prep["evict"] = read_files(EVICT)
                    e = prep["evict"]
                    print(f"  evicted: read {e['GB']} GB of other model files in {e['s']} s ({e['GBps']} GB/s); "
                          f"available {e['mem_after']['phys_avail_mb']} MiB", flush=True)
                if a.preread:
                    prep["preread"] = read_files([SHARD1])
                    e = prep["preread"]
                    print(f"  pre-read shard 1: {e['GB']} GB in {e['s']} s ({e['GBps']} GB/s); "
                          f"available {e['mem_after']['phys_avail_mb']} MiB", flush=True)
                sysm, user = build_prompt(seed, a.target, a.cpt)
                seed += 1
                one_request(f"a{rep} first prompt after load{a.name}", sysm, user, tap, sink, prep=prep)
            if "b" in arms:
                if not loaded():
                    print("  flash-next is not loaded for arm b; skipping", flush=True)
                else:
                    sysm, user = build_prompt(seed, a.target, a.cpt)
                    seed += 1
                    one_request(f"b{rep} second different prompt{a.name}", sysm, user, tap, sink)
        if "c" in arms:
            print("== arm c: fresh prompt on the loaded server (no unload)", flush=True)
            sysm, user = build_prompt(seed, a.target, a.cpt)
            one_request("c fresh prompt, server already loaded", sysm, user, tap, sink)
    finally:
        time.sleep(2.0)
        open(stop, "w").close()
        tap.stop = True
        try:
            sampler.wait(timeout=15)
        except subprocess.TimeoutExpired:
            sampler.terminate()
        fh.close()
    return 0


# ------------------------------------------------------------------------------------------------ the summary
def load_jsonl(p: str) -> list[dict]:
    if not os.path.exists(p):
        return []
    return [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]


def window_stats(samples: list[dict], t0: float, t1: float) -> dict:
    """Counters over [t0, t1]: bytes are rate x the interval since the previous sample."""
    ss = [s for s in samples if t0 - 0.5 <= s["t"] <= t1 + 0.5]
    if not ss:
        return {}
    prev = {s["t"]: p for p, s in zip([None] + samples[:-1], samples)}
    disk_c = disk_t = pin = io = trans = 0.0
    avail_min = min(s.get("available mbytes", 0) for s in ss)
    for s in ss:
        p = prev.get(s["t"])
        dt = (s["t"] - p["t"]) if p else 1.0
        dt = min(max(dt, 0.0), 3.0)
        disk_c += s.get("disk_read_bps 1 c:", 0) * dt
        disk_t += s.get("disk_read_bps _total", 0) * dt
        pin += s.get("pages input/sec", 0) * 4096 * dt
        trans += s.get("transition faults/sec", 0) * dt
        for name, pr in (s.get("procs") or {}).items():
            if pr.get("working set", 0) > 5e9 or pr.get("private bytes", 0) > 15e9:
                io += pr.get("io read bytes/sec", 0) * dt
    first, last = ss[0], ss[-1]

    def fn_ws(s):
        return max(((pr.get("working set", 0) for pr in (s.get("procs") or {}).values()
                     if pr.get("private bytes", 0) > 15e9)), default=0)
    return {"n": len(ss), "disk_c_GB": disk_c / 1e9, "disk_total_GB": disk_t / 1e9, "pages_input_GB": pin / 1e9,
            "process_io_read_GB": io / 1e9, "transition_faults_M": trans / 1e6,
            "avail_start_GB": first.get("available mbytes", 0) / 1024, "avail_min_GB": avail_min / 1024,
            "standby_normal_GB_start": first.get("standby cache normal priority bytes", 0) / 1e9,
            "standby_reserve_GB_start": first.get("standby cache reserve bytes", 0) / 1e9,
            "standby_normal_GB_end": last.get("standby cache normal priority bytes", 0) / 1e9,
            "standby_reserve_GB_end": last.get("standby cache reserve bytes", 0) / 1e9,
            "fn_ws_GB_start": fn_ws(first) / 1e9, "fn_ws_GB_end": fn_ws(last) / 1e9,
            "disk_peak_MBps": max(s.get("disk_read_bps 1 c:", 0) for s in ss) / 1e6}


def batches(rec: dict) -> list[tuple[int, float, float]]:
    """(n_tokens, seconds the batch took, tok/s) from the server's `prompt processing, n_tokens = N ... t = T s` lines."""
    out, last_n, last_t = [], 0, 0.0
    for e in rec.get("log", []):
        m = re.search(r"prompt processing, n_tokens =\s*(\d+).*?t =\s*([\d.]+) s", e["line"])
        if not m:
            continue
        n, t = int(m.group(1)), float(m.group(2))
        if n <= last_n:
            continue
        out.append((n - last_n, t - last_t, (n - last_n) / max(t - last_t, 1e-9)))
        last_n, last_t = n, t
    return out


def log_time(rec: dict, pat: str, last: bool = False) -> float | None:
    hits = [e["t"] for e in rec.get("log", []) if re.search(pat, e["line"])]
    return (hits[-1] if last else hits[0]) if hits else None


def cmd_summarize(a) -> int:
    import glob
    recs = sorted((r for f in glob.glob(os.path.join(a.dir, "requests*.jsonl")) for r in load_jsonl(f)),
                  key=lambda r: r["t_send"])
    samples = sorted((r for f in glob.glob(os.path.join(a.dir, "samples*.jsonl")) for r in load_jsonl(f)),
                     key=lambda r: r["t"])
    slog = sorted((x for f in glob.glob(os.path.join(a.dir, "serverlog*.jsonl")) for x in load_jsonl(f)), key=lambda x: x["t"])
    for r in recs:                                       # the server's lines of each request's span, from the raw tap
        r["log"] = [e for e in slog if r["t_send"] - 2.0 <= e["t"] <= r["t_end"] + 1.5
                    and re.search(r"print_timing|checkpoint|layer-major|prompt processing|launch_slot|listening|model loaded|"
                                  r"CUDA|error|warn|failed", e["line"], re.I)]
    lines = []
    w = lines.append
    w("fn_first_prompt: raw records in requests*.jsonl, samples*.jsonl, serverlog*.jsonl")
    w("disk = PhysicalDisk(1 c:) Disk Read Bytes/sec x interval; pgin = Memory Pages Input/sec x 4 KiB; both over the "
      "prefill window (the server's launch_slot line to the first streamed token)\n")
    w(f"{'request':46s} {'ttft':>6s} {'load':>5s} {'prompt':>6s} {'srv t/s':>7s} {'pf s':>6s} {'disk GB':>7s} {'pgin GB':>7s} "
      f"{'avail GB start/min':>18s} {'fnWS GB end':>11s} {'standby GB start':>16s}")
    for r in recs:
        t = r.get("timings") or {}
        t_launch = log_time(r, r"launch_slot")
        t_listen = log_time(r, r"listening on")
        t0 = t_launch or r["t_send"]
        ws = window_stats(samples, t0, r.get("t_first_token") or r["t_end"])
        pr = (r.get("usage") or {}).get("prompt_tokens")
        load = f"{t_listen - r['t_send']:.1f}" if t_listen and t_listen > r["t_send"] else "-"
        sb = ws.get("standby_normal_GB_start", 0) + ws.get("standby_reserve_GB_start", 0)
        w(f"{r['label']:46s} {r.get('ttft_s') or 0:6.1f} {load:>5s} {pr or 0:6d} {t.get('prompt_per_second', 0):7.1f} "
          f"{t.get('prompt_ms', 0) / 1000:6.1f} {ws.get('disk_c_GB', 0):7.2f} {ws.get('pages_input_GB', 0):7.2f} "
          f"{ws.get('avail_start_GB', 0):9.1f}/{ws.get('avail_min_GB', 0):<8.1f} {ws.get('fn_ws_GB_end', 0):11.1f} {sb:16.1f}")
    w("\nper batch from the server's own `prompt processing, n_tokens = N ... t = T s` lines (tokens, seconds, tok/s) and the disk "
      "read GB over that batch's span (print-line arrival times):")
    for r in recs:
        bs = batches(r)
        ends = [e["t"] for e in r.get("log", []) if re.search(r"prompt processing, n_tokens", e["line"])]
        t_prev = log_time(r, r"launch_slot")
        parts = []
        for (n, s, v), te in zip(bs, ends):
            gb = window_stats(samples, t_prev, te).get("disk_c_GB", 0) if t_prev else 0
            parts.append(f"{n} tok {s:.1f} s {v:.0f} t/s {gb:.1f} GB")
            t_prev = te
        w(f"  {r['label']}: " + "; ".join(parts))
    preps = [(r["label"], r.get("prep") or {}) for r in recs if (r.get("prep") or {}).get("evict") or (r.get("prep") or {}).get("preread")]
    if preps:
        w("\nprep reads before the request (cached sequential reads, 16 MiB):")
        for lab, p in preps:
            for k in ("evict", "preread"):
                if p.get(k):
                    e = p[k]
                    w(f"  {lab}: {k}: {e['GB']} GB in {e['s']} s = {e['GBps']} GB/s; physical available after {e['mem_after']['phys_avail_mb'] / 1024:.1f} GB")
    open(os.path.join(a.dir, "summary.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("calibrate")
    c.add_argument("--target", type=int, default=24600)
    r = sp.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--reps", type=int, default=3)
    r.add_argument("--arms", default="a,b")
    r.add_argument("--target", type=int, default=24600)
    r.add_argument("--cpt", type=float, required=True, help="characters a token (from `calibrate`)")
    r.add_argument("--seed", type=int, default=1000)
    r.add_argument("--tag", default="", help="suffix for this invocation's record files (requests<tag>.jsonl, ...)")
    r.add_argument("--evict", action="store_true", help="after the unload, read 61.9 GB of other model files (cached reads) "
                   "so the IQ2_XS pages leave the file cache")
    r.add_argument("--preread", action="store_true", help="after the unload (and the eviction), read IQ2_XS shard 1 front to "
                   "back into the file cache: the candidate fix")
    r.add_argument("--name", default="", help="text added to each label")
    r.add_argument("--settle", type=float, default=2.0, help="seconds after the unload before the request")
    s = sp.add_parser("summarize")
    s.add_argument("dir")
    a = ap.parse_args()
    return {"calibrate": cmd_calibrate, "run": cmd_run, "summarize": cmd_summarize}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
