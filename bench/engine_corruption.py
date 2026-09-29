"""Corruption checks for a main-model llama-server BINARY on the 5060 Ti.

T0 of docs/research/SUDOINGX-VISION-CU.md: the PTQ1_0 mat-vec PDL race
(engines/patches/llama-bonsai2/0002-cuda-ptq1_0-pdl-sync.patch). The checks
are docs/MTP-STAGING.md section 5's, which were only ever run on the A4000
(sm_86, which never takes the PDL path), now on the card the model serves on:

  greedy   3 prompts (ts, bash, prose) x --reps, thinking off, temperature 0,
           top_k 1, seed 0, max_tokens 512: rep-to-rep and arm-to-arm text
           identity, and decode tok/s (llama-server's own
           timings.predicted_per_second; same prompt, same n). Rep 0 is a
           cold prefill and later reps reuse the slot's cached prompt, so
           rep 0 can differ from them on a correct binary (seen 2026-09-27
           on `ts` in every arm); rep-to-rep identity is read on reps 1..n.
  plain    --plain generations, thinking on, the production sampling
           (config.yaml `sampling`: temp 1.0, top_p 0.95, top_k 20, min_p 0),
           seeds 1000+i, max_tokens 3000, two alternating prose prompts.
           Reasoning + answer scanned for runs of `!` or `/` (4+ -- the
           reported symptom -- and 10+, the corpus scan's threshold), any
           punctuation character repeated 6+ times (MTP-STAGING 5.2), and a
           word repeated 5+ times.
  tools    --tools requests with two tools, thinking on, production sampling,
           seeds 2000+i: a call is valid when it names one of the tools and
           its arguments parse as JSON.

Each ARM is a separate llama-server process with EXACTLY the flags and env of
config.yaml's `bonsai` (read from llama-swap's /running, cross-checked against
config.yaml), on a spare loopback port, with the binary and extra env the arm
names. The 5060 Ti holds one main model, so the run UNLOADS production
`bonsai` through llama-swap (:11434/api/models/unload) and refuses to start
without --window (the operator's statement that the window is theirs). Each
arm gets its own short window: wait for a quiet stack (no test suite running
-- they ask llama-swap for /upstream/bonsai/props, which reloads it -- the
card idle, no slot processing), pause the worker's gpu lane, unload, run the
arm, and ALWAYS put production back in `finally`: lane resumed, `bonsai`
reloaded through llama-swap, its cmd checked, :1234/health checked. It never
fights for VRAM: if llama-swap starts anything on the card while an arm runs
(a client asked for the model), the arm is killed at once, production is
restored, the partial record is kept as <arm>.abortedN.json, and the arm is
retried from the start in a later quiet spell (--attempts). `--restore` only
puts production back.

    python bench/engine_corruption.py --window --out DIR \\
        --arm "current=C:/.../llama-server.exe" \\
        --arm "current-pdl0=C:/.../llama-server.exe,GGML_CUDA_PDL=0" \\
        --arm "fixed=C:/.../new/llama-server.exe"

Writes DIR/<arm>.json (every text, every timing) and DIR/<arm>.log (the
server's log), and prints a summary table. Numbers only: no verdict.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SWAP = os.environ.get("LLAMA_STACK_URL", "http://127.0.0.1:11434")
PROXY = "http://127.0.0.1:1234"
PROD_ID = "bonsai"
CARD_UUID = "GPU-de660e90-0e9c-d465-b389-6df63021b920"   # RTX 5060 Ti (config.yaml)
CARD_NAME = "RTX 5060 Ti"
LANE_BY = "engine_corruption"

GREEDY = {
    "ts": "Write a TypeScript function `debounce<T extends (...args: any[]) => void>(fn: T, ms: number)` "
          "that returns a debounced function with a `cancel()` method. Include JSDoc comments and a short "
          "usage example.",
    "bash": "Write a bash script that takes a directory as its first argument, finds every *.log file under it "
            "older than 7 days, gzips each one, and prints how many files it compressed. Use set -euo pipefail "
            "and handle file names that contain spaces.",
    "prose": "In about 300 words of plain prose, explain to a new engineer why speculative decoding can make "
             "text generation faster without changing what the model would have written.",
}
PLAIN = [GREEDY["prose"],
         "Describe, in plain prose, how a hash map handles collisions and when it resizes."]
TOOLS = [
    {"type": "function", "function": {
        "name": "find_definition_opt", "description": "Find where a symbol is defined in the repository.",
        "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}},
                       "required": ["symbol"]}}},
    {"type": "function", "function": {
        "name": "read_file_range", "description": "Read lines from a file.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "start": {"type": "integer"}, "end": {"type": "integer"}},
            "required": ["path"]}}},
]
TOOL_ASKS = ["Where is the function `apply_budget` defined?", "Show me lines 10 to 40 of mcp/proxy.py.",
             "Find the definition of class SessionStore.", "Read the first 20 lines of README.md.",
             "Where is `run_check` defined?"]
PROD_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0}

RE_SYMPTOM = re.compile(r"!{4,}|/{4,}")
RE_LONG = re.compile(r"[!/]{10,}")
RE_PUNCT = re.compile(r"([^\w\s])\1{5,}")
RE_WORDS = re.compile(r"(\b\w+\b)(?:\s+\1\b){4,}")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def http(method: str, url: str, body: dict | None = None, timeout: float = 30) -> tuple[int, str]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        try:
            return e.code, e.read().decode("utf-8", "replace")
        except Exception:                                            # noqa: BLE001
            return e.code, ""
    except Exception as e:                                           # noqa: BLE001
        return 0, f"{type(e).__name__}: {e}"


def gpu() -> dict | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu",
                              "--format=csv,noheader,nounits", "-i", CARD_UUID],
                             capture_output=True, text=True, timeout=15).stdout
        used, free, util = [int(x) for x in out.strip().split(",")]
        return {"used": used, "free": free, "util": util}
    except Exception:                                                # noqa: BLE001
        return None


def card_pids() -> list[str]:
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name",
                          "--format=csv,noheader"], capture_output=True, text=True, timeout=15).stdout
    return [ln.strip() for ln in out.splitlines() if ln.startswith(CARD_UUID)]


def running() -> dict[str, dict]:
    st, txt = http("GET", f"{SWAP}/running", timeout=10)
    if st != 200:
        return {}
    try:
        return {m.get("model"): m for m in json.loads(txt).get("running") or [] if m.get("model")}
    except ValueError:
        return {}


# ----------------------------------------------------------------- config --
def _config() -> tuple[dict, dict]:
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import build_engine as be                                        # noqa: E402
    cfg = be.load_yaml(os.path.join(ROOT, "config.yaml"))
    macros = {k: v for k, v in (cfg.get("macros") or {}).items() if isinstance(v, (str, int, float))}
    return cfg, macros


def production_command() -> tuple[list[str], dict[str, str]]:
    """`bonsai`'s argv and env as config.yaml defines them (macros expanded,
    comment lines dropped, as llama-swap does). argv keeps `${PORT}`."""
    cfg, macros = _config()
    import build_engine as be                                        # noqa: E402  (path set by _config)
    m = cfg["models"][PROD_ID]
    lines = [ln.strip() for ln in str(m["cmd"]).splitlines()]
    text = " ".join(ln for ln in lines if ln and not ln.startswith("#"))
    argv = shlex.split(be._expand(text, macros))
    env = {}
    for kv in m.get("env") or []:
        k, _, v = str(kv).partition("=")
        env[k] = v
    return argv, env


def card_models() -> set[str]:
    """config.yaml model ids (and aliases) that run on this card."""
    cfg, _ = _config()
    out = set()
    for mid, m in (cfg.get("models") or {}).items():
        if any(CARD_UUID in str(e) for e in (m or {}).get("env") or []):
            out.add(str(mid))
            out.update(str(a) for a in (m or {}).get("aliases") or [])
    return out


def same_as_live(argv: list[str], live_cmd: str) -> bool:
    """llama-swap's /running cmd (comment lines stripped, ${PORT} filled) is
    config.yaml's argv with the port filled in."""
    live = shlex.split(live_cmd)
    if len(live) != len(argv):
        return False
    return all(a == b or a == "${PORT}" for a, b in zip(argv, live))


def arm_command(argv: list[str], binary: str, port: int,
                args: dict[str, str] | None = None) -> list[str]:
    """`bonsai`'s argv with this arm's binary and port. `args` overrides
    flags ({"-m": path, "--spec-draft-n-max": "2"}): the value after an
    existing flag is replaced; a flag not in argv is appended with it (a
    value of "" appends the bare flag); a value of None REMOVES the flag and
    its value, if it has one (layout v2: `--mmproj` off the main card)."""
    out = [binary] + argv[1:]
    out = [str(port) if a == "${PORT}" else a for a in out]
    for flag, value in (args or {}).items():
        if value is None:
            while flag in out:
                i = out.index(flag)
                has_value = i + 1 < len(out) and not out[i + 1].startswith("-")
                del out[i:i + (2 if has_value else 1)]
            continue
        if flag in out:
            i = out.index(flag)
            if value != "" and i + 1 < len(out):
                out[i + 1] = value
        else:
            out += [flag] + ([value] if value != "" else [])
    return out


def parse_arm(spec: str) -> tuple[str, str, dict[str, str], dict[str, str]]:
    """NAME=BINARY[,ENV=VAL...][,ARG:FLAG=VALUE...][,DROP:FLAG...] -> (name,
    binary, env, argv overrides). `ARG:-m=C:/x.gguf` swaps the model file,
    `ARG:--spec-draft-n-max=2` the draft size, `DROP:--mmproj` removes the
    projector (arm_command: a None value removes the flag)."""
    name, _, rest = spec.partition("=")
    parts = rest.split(",")
    env: dict[str, str] = {}
    args: dict[str, str | None] = {}
    for p in parts[1:]:
        if p.startswith("DROP:"):
            args[p[5:]] = None
        elif p.startswith("ARG:"):
            k, _, v = p[4:].partition("=")
            args[k] = v
        else:
            k, _, v = p.partition("=")
            env[k] = v
    return name, parts[0], env, args


# ------------------------------------------------------------------ guard --
class Guard(threading.Thread):
    """Every second: the card's free memory (min kept), and whether llama-swap
    has started anything that lives on this card. If it has, kill the arm:
    never fight a real client for VRAM."""

    def __init__(self, proc: subprocess.Popen, ids: set[str]):
        super().__init__(daemon=True)
        self.proc = proc
        self.ids = ids
        self.min_free: int | None = None
        self.max_used: int | None = None
        self.tripped: str | None = None
        self._stop = threading.Event()

    def run(self):
        n = 0
        while not self._stop.is_set():
            g = gpu()
            if g:
                self.min_free = g["free"] if self.min_free is None else min(self.min_free, g["free"])
                self.max_used = g["used"] if self.max_used is None else max(self.max_used, g["used"])
            if n % 3 == 0:
                r = running()
                on_card = [k for k in r if k in self.ids]
                if on_card and not self.tripped:
                    self.tripped = f"llama-swap started {on_card} on the card"
                    log(f"GUARD: {self.tripped}; killing the arm")
                    try:
                        self.proc.kill()
                    except Exception:                                # noqa: BLE001
                        pass
            n += 1
            self._stop.wait(1.0)

    def stop(self):
        self._stop.set()
        self.join(timeout=10)


# ------------------------------------------------------------------ tests --
class Server:
    def __init__(self, port: int):
        self.base = f"http://127.0.0.1:{port}"

    def chat(self, messages, max_tokens, thinking, sampling, seed, extra=None, timeout=1800):
        body = {"model": "x", "messages": messages, "max_tokens": max_tokens, "seed": seed,
                "chat_template_kwargs": {"enable_thinking": thinking}, "reasoning_effort": "medium"}
        body.update(sampling)
        body.update(extra or {})
        t0 = time.time()
        st, txt = http("POST", self.base + "/v1/chat/completions", body, timeout=timeout)
        if st != 200:
            raise RuntimeError(f"HTTP {st}: {txt[:300]}")
        r = json.loads(txt)
        r["_wall"] = round(time.time() - t0, 2)
        return r


VISION = {"on": False}      # --vision: the arm's server has --mmproj


def two_colour_png(w: int = 256, h: int = 256) -> bytes:
    """A PNG whose left half is red and right half blue."""
    import struct
    import zlib
    rows = b"".join(b"\x00" + b"".join((b"\xff\x00\x00" if x < w // 2 else b"\x00\x00\xff")
                                       for x in range(w)) for y in range(h))

    def chunk(t: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + t + data
                + struct.pack(">I", zlib.crc32(t + data) & 0xFFFFFFFF))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def scan(text: str) -> dict:
    return {"symptom_runs": RE_SYMPTOM.findall(text)[:10],
            "long_runs": RE_LONG.findall(text)[:10],
            "punct_runs": [m.group(0)[:20] for m in RE_PUNCT.finditer(text)][:10],
            "word_repeats": RE_WORDS.findall(text)[:10]}


def run_arm(name: str, binary: str, extra_env: dict, prod_argv: list[str], prod_env: dict,
            port: int, out: str, reps: int, n_plain: int, n_tools: int,
            extra_args: dict | None = None) -> dict:
    cmd = arm_command(prod_argv, binary, port, extra_args)
    env = dict(os.environ)
    for k in ("GGML_CUDA_PDL", "GGML_CUDA_BATCH_INVARIANT", "GGML_CUDA_DISABLE_GRAPHS"):
        env.pop(k, None)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env.update(prod_env)
    env.update(extra_env)
    rec = {"arm": name, "binary": binary, "cmd": cmd, "arg_overrides": extra_args or {},
           "env": {k: env[k] for k in list(prod_env) + list(extra_env)},
           "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "gpu_before": gpu()}
    log(f"[{name}] start: {binary} env {rec['env']}")
    logf = open(os.path.join(out, f"{name}.log"), "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)
    guard = Guard(proc, card_models())
    guard.start()
    s = Server(port)
    try:
        t0 = time.time()
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"server exited {proc.returncode} during load"
                                   + (f" ({guard.tripped})" if guard.tripped else ""))
            if http("GET", s.base + "/health", timeout=5)[0] == 200:
                break
            if time.time() - t0 > 600:
                raise RuntimeError("load timeout")
            time.sleep(1)
        rec["load_s"] = round(time.time() - t0, 1)
        time.sleep(2)
        rec["gpu_loaded"] = gpu()
        # The server's default verbosity prints no device line: ask the driver
        # which card the process holds a context on.
        dev = [p for p in card_pids() if p.split(",")[1].strip() == str(proc.pid)]
        rec["device"] = dev
        if not dev:
            raise RuntimeError(f"pid {proc.pid} holds no context on the {CARD_NAME} ({CARD_UUID})")
        st, props = http("GET", s.base + "/props", timeout=10)
        try:
            rec["build_info"] = json.loads(props).get("build_info")
        except ValueError:
            rec["build_info"] = None
        log(f"[{name}] loaded in {rec['load_s']} s; card {rec['gpu_loaded']}; {dev[0]}; build {rec['build_info']}")
        s.chat([{"role": "user", "content": "Say hi."}], 16, False, {"temperature": 0, "top_k": 1}, 0)

        rec["greedy"] = []
        for rep in range(reps):
            for pk, p in GREEDY.items():
                r = s.chat([{"role": "user", "content": p}], 512, False, {"temperature": 0, "top_k": 1}, 0)
                t = r.get("timings", {})
                c = r["choices"][0]["message"].get("content") or ""
                row = {"prompt": pk, "rep": rep, "finish": r["choices"][0]["finish_reason"], "content": c,
                       "predicted_n": t.get("predicted_n"), "tps": t.get("predicted_per_second"),
                       "prompt_tps": t.get("prompt_per_second"), "draft_n": t.get("draft_n"),
                       "draft_n_accepted": t.get("draft_n_accepted"), "wall": r["_wall"], **scan(c)}
                rec["greedy"].append(row)
                log(f"[{name}] greedy {pk} rep{rep}: n={row['predicted_n']} {row['tps']:.2f} tok/s "
                    f"acc {row['draft_n_accepted']}/{row['draft_n']} finish={row['finish']} "
                    f"symptom={row['symptom_runs']}")
                if guard.tripped:
                    raise RuntimeError(guard.tripped)

        rec["plain"] = []
        for i in range(n_plain):
            r = s.chat([{"role": "user", "content": PLAIN[i % 2]}], 3000, True, PROD_SAMPLING, 1000 + i)
            m = r["choices"][0]["message"]
            full = (m.get("reasoning_content") or "") + "\n" + (m.get("content") or "")
            t = r.get("timings", {})
            row = {"i": i, "finish": r["choices"][0]["finish_reason"],
                   "reasoning_chars": len(m.get("reasoning_content") or ""),
                   "content_chars": len(m.get("content") or ""), "predicted_n": t.get("predicted_n"),
                   "tps": t.get("predicted_per_second"), "reasoning": m.get("reasoning_content"),
                   "content": m.get("content"), **scan(full)}
            rec["plain"].append(row)
            log(f"[{name}] plain {i}: finish={row['finish']} n={row['predicted_n']} "
                f"think={row['reasoning_chars']}c answer={row['content_chars']}c symptom={row['symptom_runs']} "
                f"punct={row['punct_runs']} words={row['word_repeats']}")
            if guard.tripped:
                raise RuntimeError(guard.tripped)

        rec["tools"] = []
        for i in range(n_tools):
            r = s.chat([{"role": "user", "content": TOOL_ASKS[i % len(TOOL_ASKS)]}], 3000, True,
                       PROD_SAMPLING, 2000 + i, extra={"tools": TOOLS})
            m = r["choices"][0]["message"]
            calls = m.get("tool_calls") or []
            ok = bool(calls)
            for c in calls:
                try:
                    json.loads(c["function"]["arguments"])
                    ok = ok and c["function"]["name"] in ("find_definition_opt", "read_file_range")
                except Exception:                                    # noqa: BLE001
                    ok = False
            full = (m.get("reasoning_content") or "") + "\n" + (m.get("content") or "")
            row = {"i": i, "finish": r["choices"][0]["finish_reason"], "ok": ok,
                   "calls": [(c["function"]["name"], c["function"]["arguments"]) for c in calls],
                   "reasoning": m.get("reasoning_content"), "content": m.get("content"), **scan(full)}
            rec["tools"].append(row)
            log(f"[{name}] tool {i}: ok={ok} {row['calls']} finish={row['finish']} symptom={row['symptom_runs']}")
            if guard.tripped:
                raise RuntimeError(guard.tripped)

        if VISION["on"]:
            # a server started with --mmproj: an attached image (left half red, right half blue), described
            import base64
            uri = "data:image/png;base64," + base64.b64encode(two_colour_png()).decode()
            rec["vision"] = []
            for i in range(2):
                r = s.chat([{"role": "user", "content": [
                    {"type": "text", "text": "What colour is the left half of this image and what colour is "
                                             "its right half? Answer exactly: left=<colour>, right=<colour>"},
                    {"type": "image_url", "image_url": {"url": uri}}]}],
                    2000, True, PROD_SAMPLING, 3000 + i)
                m = r["choices"][0]["message"]
                c = (m.get("content") or "").lower()
                ok = bool(re.search(r"left\W{0,3}red", c) and re.search(r"right\W{0,3}blue", c))
                t = r.get("timings") or {}
                row = {"i": i, "ok": ok, "finish": r["choices"][0]["finish_reason"], "content": m.get("content"),
                       "prompt_n": t.get("prompt_n"), "usage": r.get("usage"), **scan(c)}
                rec["vision"].append(row)
                log(f"[{name}] vision {i}: ok={ok} finish={row['finish']} prompt_n={row['prompt_n']} "
                    f"content={(m.get('content') or '')[:80]!r}")
                if guard.tripped:
                    raise RuntimeError(guard.tripped)
    except Exception as e:                                           # noqa: BLE001
        rec["error"] = repr(e)
        log(f"[{name}] ERROR {e!r}")
    finally:
        guard.stop()
        rec["gpu_min_free"] = guard.min_free
        rec["gpu_max_used"] = guard.max_used
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
        log(f"[{name}] stopped; min free {guard.min_free} MiB")
    return rec


# -------------------------------------------------------------- summary ----
def summary(recs: list[dict]) -> str:
    rows = ["| arm | greedy decode tok/s median (min-max), n | greedy reps 1..n identical | "
            "plain: with `!!!!`/`////` | plain: 10+ `!`/`/` | plain: 6+ punct run | plain: 5x word | "
            "plain finish | tools valid | error |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for r in recs:
        g = r.get("greedy") or []
        tps = [x["tps"] for x in g if x.get("tps")]
        same = []
        for pk in GREEDY:
            # rep 0 is a cold prefill; reps 1+ reuse the slot's cached prompt
            # (one token re-evaluated), so rep 0 may differ on a correct binary.
            texts = [x["content"] for x in g if x["prompt"] == pk and x["rep"] >= 1]
            same.append(f"{pk} {'-' if len(texts) < 2 else 'yes' if len(set(texts)) == 1 else 'NO'}")
        p = r.get("plain") or []
        t = r.get("tools") or []
        fin: dict[str, int] = {}
        for x in p:
            fin[x["finish"]] = fin.get(x["finish"], 0) + 1
        rows.append(
            f"| {r['arm']} | "
            + (f"{statistics.median(tps):.2f} ({min(tps):.2f}-{max(tps):.2f}), n={len(tps)}" if tps else "-")
            + f" | {', '.join(same)} | {sum(1 for x in p if x['symptom_runs'])}/{len(p)} "
            f"| {sum(1 for x in p if x['long_runs'])}/{len(p)} | {sum(1 for x in p if x['punct_runs'])}/{len(p)} "
            f"| {sum(1 for x in p if x['word_repeats'])}/{len(p)} | {fin} "
            f"| {sum(1 for x in t if x['ok'])}/{len(t)} | {r.get('error') or ''} |")
    if len(recs) > 1:
        rows.append("")
        rows.append("Greedy text identity between arms (rep 0 of each prompt; first differing character):")
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                parts = []
                for pk in GREEDY:
                    a = next((x["content"] for x in recs[i].get("greedy") or [] if x["prompt"] == pk), None)
                    b = next((x["content"] for x in recs[j].get("greedy") or [] if x["prompt"] == pk), None)
                    if a is None or b is None:
                        parts.append(f"{pk} -")
                    elif a == b:
                        parts.append(f"{pk} identical")
                    else:
                        k = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
                        parts.append(f"{pk} differs @{k}")
                rows.append(f"- {recs[i]['arm']} vs {recs[j]['arm']}: {'; '.join(parts)}")
    return "\n".join(rows)


# -------------------------------------------------------------- restore ----
def restore(prod_argv: list[str]) -> dict:
    rec: dict = {}
    try:
        sys.path.insert(0, os.path.join(ROOT, "mcp"))
        import jobs                                                  # noqa: E402
        p = jobs.paused("gpu")
        if p and str(p.get("by", "")).startswith(LANE_BY):
            jobs.resume("gpu")
        rec["lane_resumed"] = True
    except Exception as e:                                           # noqa: BLE001
        rec["lane_resumed"] = f"failed: {e!r}"
    if PROD_ID not in running():
        t0 = time.time()
        while time.time() - t0 < 900:
            st, _ = http("GET", f"{SWAP}/upstream/{PROD_ID}/health", timeout=900)
            if st == 200:
                break
            time.sleep(3)
        rec["reload_s"] = round(time.time() - t0, 1)
    m = running().get(PROD_ID) or {}
    rec["bonsai_state"] = m.get("state")
    rec["bonsai_cmd_is_config"] = same_as_live(prod_argv, m.get("cmd") or "")
    rec["proxy_health"] = http("GET", f"{PROXY}/health", timeout=30)[0]
    rec["gpu"] = gpu()
    log(f"restore: {rec}")
    return rec


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--window", action="store_true", help="the operator grants the window (production unloaded)")
    ap.add_argument("--restore", action="store_true", help="only put production back")
    ap.add_argument("--arm", action="append", default=[],
                    help="NAME=BINARY[,ENV=VAL...][,ARG:FLAG=VALUE...]")
    ap.add_argument("--out", required=False)
    ap.add_argument("--port", type=int, default=18091)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--plain", type=int, default=16)
    ap.add_argument("--tools", type=int, default=10)
    ap.add_argument("--attempts", type=int, default=4, help="tries per arm when the guard aborts one")
    ap.add_argument("--wait-quiet", type=float, default=60.0, help="minutes to wait for a quiet stack per attempt")
    ap.add_argument("--summarise", nargs="*", help="print the summary of existing <arm>.json files")
    ap.add_argument("--vision", action="store_true", help="also describe an attached image (--mmproj arms)")
    a = ap.parse_args(argv)
    VISION["on"] = a.vision

    if a.summarise:
        recs = [json.load(open(p, encoding="utf-8")) for p in a.summarise]
        print(summary(recs))
        return 0
    prod_argv, prod_env = production_command()
    if a.restore:
        r = restore(prod_argv)
        return 0 if r.get("proxy_health") == 200 else 1
    if not a.window:
        print("refusing: this run unloads the production model. Pass --window only inside a "
              "maintenance window the operator has given.")
        return 2
    if not a.out or not a.arm:
        print("--out and at least one --arm are required")
        return 2
    os.makedirs(a.out, exist_ok=True)
    arms = []
    for spec in a.arm:
        name, binary, env, args = parse_arm(spec)
        missing = [p for p in [binary] + ([args["-m"]] if "-m" in args else []) if not os.path.exists(p)]
        if missing:
            print(f"arm {name}: {missing} does not exist")
            return 2
        arms.append((name, binary, env, args))

    recs = []
    res: dict = {}
    for name, binary, env, args in arms:
        rec: dict = {}
        for attempt in range(1, a.attempts + 1):
            why = wait_quiet(prod_argv, a.wait_quiet * 60)
            if why:
                log(f"[{name}] not run: {why}")
                rec = {"arm": name, "error": f"not run: {why}"}
                break
            with open(os.path.join(a.out, "production.json"), "w", encoding="utf-8") as f:
                json.dump({"argv": prod_argv, "env": prod_env,
                           "llama_swap_cmd": (running().get(PROD_ID) or {}).get("cmd")}, f, indent=1)
            try:
                sys.path.insert(0, os.path.join(ROOT, "mcp"))
                import jobs                                          # noqa: E402
                jobs.pause("gpu", by=LANE_BY, why="T0 PDL corruption checks (bench/engine_corruption.py)",
                           ttl_seconds=3600)
                st, txt = http("POST", f"{SWAP}/api/models/unload/{PROD_ID}", timeout=120)
                log(f"[{name}] attempt {attempt}: unload {PROD_ID}: HTTP {st} {txt[:100]}")
                t0 = time.time()
                while PROD_ID in running() or any("llama-server" in p for p in card_pids()):
                    if time.time() - t0 > 180:
                        raise RuntimeError("bonsai did not leave the card in 180 s")
                    time.sleep(2)
                time.sleep(3)
                log(f"[{name}] card empty of llama-server: {gpu()}")
                rec = run_arm(name, binary, env, prod_argv, prod_env, a.port, a.out, a.reps, a.plain,
                              a.tools, args)
            finally:
                res = restore(prod_argv)
                with open(os.path.join(a.out, f"restore-{name}-{attempt}.json"), "w", encoding="utf-8") as f:
                    json.dump(res, f, indent=1)
            if not rec.get("guard"):
                break
            # A client (or a test suite) asked for the model: production is back;
            # keep the partial record and try this arm again in a later quiet spell.
            for ext in ("json", "log"):
                src = os.path.join(a.out, f"{name}.{ext}")
                if os.path.exists(src):
                    os.replace(src, os.path.join(a.out, f"{name}.aborted{attempt}.{ext}"))
            log(f"[{name}] attempt {attempt} aborted by the guard ({rec['guard']}); production restored")
        recs.append(rec)
        if res and res.get("proxy_health") != 200:
            log("stopping: :1234 is not healthy after restore")
            break
    print(summary([r for r in recs if r.get("greedy") is not None] or recs))
    return 0 if all("error" not in r for r in recs) and res.get("proxy_health") == 200 else 1


def test_processes() -> list[str]:
    """Offline suites import modules that ask llama-swap for
    /upstream/bonsai/props (budget.pool_size, tiers.accepted_efforts); with
    `bonsai` unloaded, that request starts it again (seen 2026-09-26: the
    guard tripped 4 s and 3 min after an unload)."""
    ps = ("Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | "
          "ForEach-Object { $_.ProcessId.ToString() + ' ' + $_.CommandLine }")
    procs = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                           capture_output=True, text=True, timeout=60).stdout.splitlines()
    return [p.strip()[:160] for p in procs if "run_tests.py" in p or re.search(r"test_\w+\.py", p)]


def quiet_reasons(prod_argv: list[str]) -> list[str]:
    why = []
    live = (running().get(PROD_ID) or {}).get("cmd") or ""
    if not same_as_live(prod_argv, live):
        why.append(f"running `{PROD_ID}` is not config.yaml's command ({live[:200]!r})")
    others = [p for p in card_pids() if prod_argv[0].replace("/", "\\") not in p]
    if others:
        why.append(f"other processes on the card: {others}")
    tests = test_processes()
    if tests:
        why.append("test suites running: " + "; ".join(tests))
    utils = []
    for _ in range(10):
        g = gpu()
        utils.append(g["util"] if g else 100)
        time.sleep(1)
    if statistics.mean(utils) > 20:
        why.append(f"card busy ({statistics.mean(utils):.0f}% mean utilisation over 10 s)")
    st, txt = http("GET", f"{SWAP}/upstream/{PROD_ID}/slots", timeout=20)
    try:
        busy = [s.get("id") for s in json.loads(txt) if s.get("is_processing")]
    except Exception:                                                # noqa: BLE001
        busy = []
    if busy:
        why.append(f"{PROD_ID} slots processing: {busy}")
    return why


def wait_quiet(prod_argv: list[str], seconds: float) -> str | None:
    """None once the stack is quiet; the last reasons if it never was."""
    t0 = time.time()
    last = None
    while True:
        why = quiet_reasons(prod_argv)
        if not why:
            return None
        if why != last:
            log("waiting: " + " | ".join(w[:200] for w in why))
            last = why
        if time.time() - t0 > seconds:
            return " | ".join(why)
        time.sleep(20)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
