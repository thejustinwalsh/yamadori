"""The GPU gate for Mirai S (Qwen3.8-27B, 2.4-bit trellis) on the llama-mirai-s engine (engines/manifest.yaml
llama-mirai-s, models/manifest.yaml mirai-s-*, config.mirai-s.fragment.yaml). Waits for the operator's "GPU free":
it unloads production `bonsai` from the 5060 Ti, one GPU consumer at a time, through bench/engine_corruption.py's
window discipline (quiet stack, the worker's gpu lane paused, a guard that kills the arm if llama-swap starts
anything on the card, production restored in `finally`) -- the same discipline as bench/flashnext_gate.py.

Every arm is its own llama-server with MIRAI_ARGV (the fragment's command) plus the arm's env and flag overrides.
Nothing here is a claim: each step MEASURES, and PASS/FAIL is per the rule written beside it.

  kernels  test-backend-ops on CUDA0 (the 5060 Ti, sm_120: the fork's author tested sm_86 only) and CUDA1 (the
           A4000, sm_86): the fork's own cases -- MIRAI.* (the trellis GEMV / MMA / cuBLASLt int8 / I3 head paths,
           33 cases on the author's 3090), FLASH_ATTN_EXT (its GQA-packed vec kernel, q8_0/q4_0/f16 K/V, GQA 2-12),
           GATED_DELTA_NET (its chunked prefill, rollback snapshots) -- plus MUL_MAT for the dense types the file
           carries (q8_0 MTP block, f16, f32). PASS: every case passes on both cards.
  fit      the window is DERIVED FROM TWO MEASUREMENTS, never from the card's claim: the arm launched at -c C1 and at
           -c C2 (-np 2 --kv-unified, q8_0 K/V; the mmproj
           on the GPU unless --no-mmproj), each warmed with an 8K prompt, a 2K-token ubatch
           prefill and an image through the mmproj; peak used VRAM from the guard (nvidia-smi, 1 s). bytes per cell =
           (peak2 - peak1) / (C2 - C1); the largest -c whose peak keeps MARGIN_MIB free = the recommendation,
           rounded down to 1,024. Reported (the operator sets -c), with and without MTP.
  api      what the proxy needs from this llama-server (docs/FLASH-NEXT.md section 5 list): /props chat_template,
           /apply-template (a tool-call turn, a `developer` message, reasoning_effort low/medium/high/xhigh/max --
           which the template accepts), /tokenize with_pieces, /completion id_slot + n_predict 0, /slots,
           /slots/N?action=erase (501 without --slot-save-path -> our one-token fallback), chat logprobs +
           top_logprobs, reasoning_budget_tokens + reasoning_budget_message + our nudge fields, timings.cache_n.
           PASS: every endpoint answers the way mcp/ reads it.
  switches the fork's three GENERAL changes, each against its off switch on the same binary: GGML_CUDA_FA_VEC_GQA=0,
           GGML_CUDA_GDN_CHUNKED=0, LLAMA_SERVER_TEXT_ALIGN=0. 256-token greedy continuations of an 8K and a 32K
           prompt and decode/prefill tok/s. The author reports 8/10 and 9/10 greedy-identical against vLLM with GQA
           packing / chunked prefill on (ties): REPORTED, not gated.
  align    the prompt-cache-by-text change (server-context.cpp) on a multi-turn chat whose sampled answers carry
           emoji (the re-tokenization case it targets), on vs LLAMA_SERVER_TEXT_ALIGN=0: per turn, cache_n /
           prompt_n of the next request. Reported.
  needles  5 needles at 1K, 32K and 128K, and at the fitted window when it is larger. PASS: 5/5 at every depth.
  corrupt  engine_corruption.py's run_arm (greedy rep identity, 16 sampled runs scanned for the corruption
           symptoms, 10 tool calls; the image check -- the mmproj is loaded). PASS: its own rules.
  lane     the jjava lane's three arms (layout v3): main decode tok/s at 8K/32K/64K, n=3, with the lane KEPT,
           ACTIVE (a burst during the decode) and CLEARED after its burst, and the next burst's re-prefill ms.
  decider  jjava on Mirai S (operator 2026-09-29), after `api` confirmed logprobs / top_logprobs:
           bench/decider/measure_model.py and jevbench_run.py against this arm's server (--base-url), their
           outputs where the Bonsai and Flash-Next runs put theirs (DECIDER_BENCHES). PASS: both exit 0.
  speed    prefill and decode tok/s at 4K / 32K / 64K / 128K, n=3, from llama-server's own timings, the idle-slot
           shape (an idle 64K slot beside an 8K decode), and the card's lowest free VRAM. Arms: base, mtp (3 draft
           tokens, the card's setting), mtp-np1 (the card's exact launch, to set our numbers beside theirs).

    python bench/mirai_s_gate.py --dry-run               # print every arm's command, touch nothing
    python bench/mirai_s_gate.py --selftest              # the pure parts, offline
    python bench/mirai_s_gate.py --window --out DIR [--steps ...] [--arms ...] [--ctx N]

Numbers and PASS/FAIL per rule; no deploy. Nothing here changes config.yaml.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_corruption as ec                                        # noqa: E402

ROOT = ec.ROOT
MODELS = "C:/Users/jwals/textgen/user_data/models"
MODEL = f"{MODELS}/mirai-s/Qwen3.8-27B-S-mirai.gguf"
MMPROJ = f"{MODELS}/mirai-s/mmproj-Qwen3.8-27B-base-f16.gguf"
ENGINE = "llama-mirai-s"
MARGIN_MIB = 1000                                      # docs/ENGINES.md "The VRAM line" (bonsai-ada-surgery's soak)
# The first launch's window, before the fit step measures. Derived (config.mirai-s.fragment.yaml, "FIT"): the
# card's 14.6 GB peak at 128K q8_0 + MTP 3, -np 1, mmproj NOT loaded, plus our additions (the mmproj on the GPU,
# the card's +1.3 GB at peak; two more slots' DeltaNet state, 3 x ~150 MiB) gives 94,208 - 125,952 cells inside
# 16,311 - 1,000 MiB (the spread is whether "GB" there is GiB or 10^9). Both fit points sit below the low end so
# the launches load; the fit step measures the slope and sets the window. Starting points, not claims.
CTX_START = 90112
FIT_POINTS = (49152, 90112)
KERNEL_OPS = ["MIRAI.*", "FLASH_ATTN_EXT", "GATED_DELTA_NET"]
MUL_MAT_TYPES = ["q8_0", "f16", "f32"]
EFFORTS = ["low", "medium", "high", "xhigh", "max"]
SWITCHES = {"off-gqa": {"GGML_CUDA_FA_VEC_GQA": "0"},
            "off-gdn": {"GGML_CUDA_GDN_CHUNKED": "0"},
            "off-align": {"LLAMA_SERVER_TEXT_ALIGN": "0"}}

# config.mirai-s.fragment.yaml's command, sampling as config.yaml's macro. -c is the fit step's to set.
MIRAI_ARGV = [
    "llama-server.exe", "--port", "${PORT}",
    "-m", MODEL, "--mmproj", MMPROJ,
    "-dev", "CUDA0", "-ngl", "999",
    "-c", str(CTX_START), "--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "-fa", "on",
    "-np", "2", "--kv-unified", "-b", "2048", "-ub", "1024",   # layout v3: one conversation + the lane
    "--jinja", "--reasoning-format", "deepseek", "--no-context-shift", "--no-cache-idle-slots",
    "--reasoning-budget", "32768",
    "--reasoning-budget-message", "Thinking budget reached. I will stop deliberating and write the final answer now.",
    "--temp", "1.0", "--top-p", "0.95", "--top-k", "20", "--min-p", "0.0",
    "--presence-penalty", "0", "--repeat-penalty", "1.0",
]
MTP = {"--spec-type": "draft-mtp", "--spec-draft-n-max": "3"}
# --no-mmproj (2026-09-29, tier models): every arm WITHOUT the projector, as layout v2 serves bonsai ("Vision can go
# to second card and swap in and out"; images go to bonsai-vision on the A4000). The fit then measures the window
# the card holds without it, and the warm and the corruption check skip the image.
NO_MMPROJ = {"on": False}
# THE LAYOUT (operator, 2026-09-30, for Bonsai and Flash-Next: a model whose jjava lane MATERIALLY slows its decode
# runs -np 1, its jjava and side calls on bonsai-a4000; coordinator 2026-10-01: Mirai S "measured at the layout the
# operator's rules imply"). LAYOUT["np"] is set by main() from the lane step (run FIRST, at -np 2): ACTIVE material
# at any depth -> 1, else 2. Every later arm (fit, api, decider, needles, corrupt, speed) runs at it.
LAYOUT = {"np": 2, "why": "default (-np 2: the conversation + the lane) until the lane step decides"}
# THE WINDOW IS THE 5060 Ti ONLY (coordinator 2026-10-01: "The Flash-Next agent is on the A4000"): --card-only
# runs the kernel cases on the 5060 Ti alone (CUDA_VISIBLE_DEVICES = its UUID), never the A4000.
CARD_ONLY = {"on": False}


def argv() -> list[str]:
    if not NO_MMPROJ["on"]:
        return list(MIRAI_ARGV)
    out, skip = [], False
    for x in MIRAI_ARGV:
        if skip:
            skip = False
            continue
        if x == "--mmproj":
            skip = True
            continue
        out.append(x)
    return out


# main()'s own parameter is named `argv`: the arms' command is reached there by this name (2026-10-01: the corrupt
# step called the list)
_argv = argv


def engine_bin() -> str:
    env = os.environ.get("MIRAI_BIN")
    if env:
        return env
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import build_engine as be                                         # noqa: E402
    m = be.load_yaml(be.MANIFEST)
    p = ((m["engines"].get(ENGINE) or {}).get("shipped") or {}).get("path", "")
    return os.path.dirname(p)


def arms(ctx: int = CTX_START, np_: int | None = None) -> dict[str, dict]:
    """name -> {env, args, checks}. `np_` (LAYOUT["np"] by default): -np for every arm but mtp-np1."""
    c = {"-c": str(ctx), "-np": str(np_ or LAYOUT["np"])}
    table = {
        "base":    {"env": {}, "args": c, "checks": ["api", "needles", "corrupt", "speed"]},
        "mtp":     {"env": {}, "args": {**c, **MTP}, "checks": ["corrupt", "speed"]},
        "mtp-np1": {"env": {}, "args": {**c, **MTP, "-np": "1"}, "checks": ["speed"]},
    }
    for name, env in SWITCHES.items():
        table[name] = {"env": env, "args": c,
                       "checks": ["align"] if name == "off-align" else ["switches", "speed"]}
    return table


def command(arm: dict, port: int, binary: str | None = None) -> list[str]:
    exe = os.path.join(binary if binary is not None else engine_bin(), "llama-server.exe")
    return ec.arm_command(argv(), exe, port, arm["args"])


def derive_fit(points: list[tuple[int, int]], total_mib: int, margin_mib: int = MARGIN_MIB) -> dict:
    """points = [(ctx, peak_used_mib), ...] from two warmed launches of one arm. Linear in -c: the K/V cells are
    the only allocation that scales with it (the recurrent state scales with -np, the compute buffers with -ub).
    Arithmetic on measurements only."""
    (c1, u1), (c2, u2) = sorted(points)[:2]
    if c2 == c1:
        raise ValueError("two different -c values are needed")
    per_cell_mib = (u2 - u1) / (c2 - c1)
    fixed_mib = u1 - per_cell_mib * c1
    room = total_mib - margin_mib - fixed_mib
    max_ctx = int(room / per_cell_mib) // 1024 * 1024 if per_cell_mib > 0 else 0
    return {"bytes_per_cell": round(per_cell_mib * 2**20), "fixed_mib": round(fixed_mib),
            "total_mib": total_mib, "margin_mib": margin_mib, "max_ctx": max_ctx}


# The repository text runs denser than the 4 characters a token the prompts are cut by: the 2026-10-01 gate's
# "131,072-token" needle was 140,551 tokens (bench/results/mirai_s/gate-20261001/base.server.log) -- 1.0723x. A
# depth is kept only when it fits the window at that measured ratio plus 8% (TOKENS_PER_EST), with the answer's room.
TOKENS_PER_EST = 1.08


def fits(depth_tokens: int, ctx: int | None, answer: int) -> bool:
    return not ctx or depth_tokens * TOKENS_PER_EST + answer + 512 <= ctx


def cap_depth(ctx: int, answer: int) -> int:
    """The deepest estimated depth (whole 1,024s) that fits `ctx` (fits())."""
    return int((ctx - answer - 512) / TOKENS_PER_EST) // 1024 * 1024


def needle_prompt(depth_tokens: int, key: str, filler: str) -> str:
    """~depth_tokens of repository text with one sentence holding `key` in the middle (4 chars ~ 1 token)."""
    body = filler[: depth_tokens * 4]
    mid = len(body) // 2
    return (body[:mid] + f"\n\nThe secret code for the vault is {key}.\n\n" + body[mid:]
            + "\n\nWhat is the secret code for the vault? Answer with the code only.")


def corpus() -> str:
    files = sorted(glob.glob(os.path.join(ROOT, "mcp", "*.py")))
    return "".join(open(f, encoding="utf-8", errors="replace").read() for f in files)


# ------------------------------------------------------------------ steps --
def step_kernels(out: str) -> dict:
    exe = os.path.join(engine_bin(), "test-backend-ops.exe")
    rec: dict = {}
    runs = [(op, None) for op in KERNEL_OPS] + [("MUL_MAT", f"type_a={t}") for t in MUL_MAT_TYPES]
    env = None
    devs = ("CUDA0", "CUDA1")
    if CARD_ONLY["on"]:
        # the 5060 Ti alone: the only CUDA device the process can see is CUDA0
        env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=ec.CARD_UUID)
        devs = ("CUDA0",)
        rec["devices"] = {"CUDA0": f"{ec.CARD_NAME} ({ec.CARD_UUID}) only: --card-only (the A4000 is another "
                                   "agent's window)"}
    for dev in devs:
        for op, params in runs:
            key = f"{dev}/{op}" + (f"/{params}" if params else "")
            cmd = [exe, "test", "-o", op, "-b", dev] + (["-p", params] if params else [])
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, env=env)
            m = re.search(r"(\d+)/(\d+) tests passed", r.stdout)
            rec[key] = {"passed": int(m.group(1)), "total": int(m.group(2))} if m else {"error": r.stdout[-400:]}
            ec.log(f"kernels {key}: {rec[key]}")
    rec["PASS"] = all(v.get("total") and v.get("passed") == v.get("total")
                      for k, v in rec.items() if isinstance(v, dict) and k != "devices")
    json.dump(rec, open(os.path.join(out, "kernels.json"), "w"), indent=1)
    return rec


class Launched:
    """One arm's llama-server, with the guard; waits for /health."""

    def __init__(self, name: str, arm: dict, port: int, out: str, extra: list[str] | None = None):
        self.cmd = command(arm, port) + (extra or [])
        env = dict(os.environ)
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = ec.CARD_UUID
        env.update(arm["env"])
        self.base = f"http://127.0.0.1:{port}"
        self.logf = open(os.path.join(out, f"{name}.server.log"), "a", encoding="utf-8")
        t0 = time.time()
        self.proc = subprocess.Popen(self.cmd, env=env, stdout=self.logf, stderr=subprocess.STDOUT)
        self.guard = ec.Guard(self.proc, ec.card_models())
        self.guard.start()
        while time.time() - t0 < 1800:
            if self.proc.poll() is not None:
                raise RuntimeError(f"{name}: server exited {self.proc.returncode}")
            if self.guard.tripped:
                raise RuntimeError(self.guard.tripped)
            if ec.http("GET", self.base + "/health", timeout=5)[0] == 200:
                break
            time.sleep(2)
        self.load_s = round(time.time() - t0, 1)

    def stop(self) -> dict:
        self.guard.stop()
        self.proc.terminate()
        try:
            self.proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.logf.close()
        return {"gpu_min_free": self.guard.min_free, "gpu_max_used": self.guard.max_used, "guard": self.guard.tripped}


def completion(base: str, prompt: str, n: int, slot: int = 0, sampled: bool = False, seed: int = 0) -> dict:
    body = {"prompt": prompt, "id_slot": slot, "n_predict": n, "cache_prompt": True, "seed": seed}
    body.update(ec.PROD_SAMPLING if sampled else {"temperature": 0.0, "top_k": 1})
    st, txt = ec.http("POST", base + "/completion", body, timeout=7200)
    if st != 200:
        raise RuntimeError(f"/completion: HTTP {st}: {txt[:300]}")
    r = json.loads(txt)
    t = r.get("timings") or {}
    return {"content": r.get("content") or "", "prompt_n": t.get("prompt_n"), "cache_n": t.get("cache_n"),
            "prompt_tps": t.get("prompt_per_second"), "predicted_n": t.get("predicted_n"),
            "tps": t.get("predicted_per_second")}


def warm(base: str, port: int) -> None:
    """The fit's worst case short of a full window: an 8K prompt, a 2K prompt read in ubatches, an image."""
    text = corpus()
    completion(base, text[: 8192 * 4], 64, slot=0)
    completion(base, text[50_000: 50_000 + 2048 * 4], 16, slot=min(1, LAYOUT["np"] - 1))
    if NO_MMPROJ["on"]:
        return
    import base64
    uri = "data:image/png;base64," + base64.b64encode(ec.two_colour_png()).decode()
    ec.Server(port).chat([{"role": "user", "content": [
        {"type": "text", "text": "What colours are in this image?"},
        {"type": "image_url", "image_url": {"url": uri}}]}], max_tokens=256, thinking=False,
        sampling={"temperature": 0.0, "top_k": 1}, seed=0)


def step_fit(arm_name: str, arm: dict, port: int, out: str) -> dict:
    rec: dict = {"points": []}
    for c in FIT_POINTS:
        a = dict(arm, args={**arm["args"], "-c": str(c)})
        srv = Launched(f"fit-{arm_name}-{c}", a, port, out)
        try:
            warm(srv.base, port)
            g = ec.gpu() or {}
        finally:
            s = srv.stop()
        rec["points"].append({"ctx": c, "load_s": srv.load_s, "after_warm": g, **s})
    total = None
    g = ec.gpu()
    if g:
        total = g["used"] + g["free"]
    pts = [(p["ctx"], p["gpu_max_used"]) for p in rec["points"] if p.get("gpu_max_used") is not None]
    if total and len(pts) == 2:
        rec["derived"] = derive_fit(pts, total)
    return rec


def step_api(port: int, arm: dict, out: str) -> dict:
    srv = Launched("api", arm, port, out)
    b = srv.base
    rec: dict = {}
    try:
        st, txt = ec.http("GET", b + "/props")
        props = json.loads(txt) if st == 200 else {}
        tmpl = props.get("chat_template") or ""
        rec["props"] = {"status": st, "chat_template_sha256": hashlib.sha256(tmpl.encode()).hexdigest(),
                        "n_ctx": (props.get("default_generation_settings") or {}).get("n_ctx")}
        tool_turn = [{"role": "developer", "content": "You are terse."},
                     {"role": "user", "content": "List the files."},
                     {"role": "assistant", "content": "", "reasoning_content": "I will list them.",
                      "tool_calls": [{"id": "call_1", "type": "function",
                                      "function": {"name": "ls", "arguments": "{\"path\": \".\"}"}}]},
                     {"role": "tool", "tool_call_id": "call_1", "content": "a.py\nb.py"}]
        tools = [{"type": "function", "function": {"name": "ls", "description": "list a directory",
                                                    "parameters": {"type": "object",
                                                                   "properties": {"path": {"type": "string"}}}}}]
        rec["apply_template"] = {}
        for eff in EFFORTS:
            st, txt = ec.http("POST", b + "/apply-template",
                              {"messages": tool_turn, "tools": tools, "reasoning_effort": eff})
            rec["apply_template"][eff] = {"status": st, "prompt_tail": (json.loads(txt).get("prompt", "")[-400:]
                                                                        if st == 200 else txt[:300])}
        st, txt = ec.http("POST", b + "/tokenize", {"content": "Hello world", "with_pieces": True})
        rec["tokenize"] = {"status": st, "body": txt[:300]}
        lane = min(1, LAYOUT["np"] - 1)          # the lane at -np 2; the one slot at -np 1
        r = completion(b, "The quick brown fox", 0, slot=lane)
        rec["completion_n0"] = r
        st, txt = ec.http("GET", b + "/slots")
        rec["slots"] = {"status": st, "n": len(json.loads(txt)) if st == 200 else None}
        st, txt = ec.http("POST", b + f"/slots/{lane}?action=erase", {})
        rec["erase"] = {"status": st, "body": txt[:200], "fallback_needed": st == 501}
        body = {"model": "x", "messages": [{"role": "user", "content": "Say one word."}], "max_tokens": 1024,
                "logprobs": True, "top_logprobs": 5, "reasoning_budget_tokens": 64,
                "reasoning_budget_message": "Budget reached.", "reasoning_effort": "medium",
                "temperature": 0.0, "top_k": 1, "seed": 0}
        st, txt = ec.http("POST", b + "/v1/chat/completions", body, timeout=600)
        ch = (json.loads(txt).get("choices") or [{}])[0] if st == 200 else {}
        lp = (ch.get("logprobs") or {}).get("content") or []
        rec["chat"] = {"status": st, "has_logprobs": bool(lp),
                       # the decider (jjava) reads the top-k alternatives of the first answer token
                       "top_logprobs_n": len((lp[0] or {}).get("top_logprobs") or []) if lp else 0,
                       "reasoning_chars": len((ch.get("message") or {}).get("reasoning_content") or ""),
                       "content": ((ch.get("message") or {}).get("content") or "")[:120], "body": txt[:200]}
        again = completion(b, "The quick brown fox jumps", 0, slot=lane)
        rec["cache_n_after_extend"] = again.get("cache_n")
        rec["PASS"] = all([rec["props"]["status"] == 200, rec["tokenize"]["status"] == 200,
                           rec["slots"]["status"] == 200, rec["chat"]["status"] == 200,
                           rec["chat"]["has_logprobs"], rec["chat"]["top_logprobs_n"] >= 5,
                           rec["completion_n0"].get("prompt_n") is not None,
                           all(v["status"] == 200 for k, v in rec["apply_template"].items()
                               if k in ("low", "medium", "xhigh")),
                           rec["erase"]["status"] in (200, 501)])
    finally:
        rec.update(srv.stop())
    return rec


# JJAVA ON MIRAI S (operator, 2026-09-29: "Jjava should run on all models, every model should have the jjava
# unlock"). The same three benches, the same output layout as the Bonsai and Flash-Next runs, so the three compare:
#   bench/decider/measure_model.py --run --model mirai-s --base-url ...   -> results/models/mirai-s.json (labels,
#       letter_prior, label_bias, tie_band, and legacy_vs_typed as its last part -> results/legacy_vs_typed.mirai-s.json)
#   bench/decider/jevbench_run.py --run --model mirai-s --base-url ...    -> results/jevbench/<run>/ (the 231 public
#       items of the clone at jevbench@bb05a335)
# legacy_vs_typed.py is NOT run a second time: measure_model.py runs it as its last part and files its summary.
# The runtime then reads results/models/mirai-s.json for every request mirai-s serves (decider_bonsai.profile keys by
# max_mode.current(), the tier table's model). Only after the api step confirmed logprobs / top_logprobs.
DECIDER_BENCHES = [
    ["bench/decider/measure_model.py", "--run", "--model", "mirai-s"],
    ["bench/decider/jevbench_run.py", "--run", "--model", "mirai-s"],
]


def step_decider(port: int, arm: dict, out: str) -> dict:
    srv = Launched("decider", arm, port, out)
    rec: dict = {"load_s": srv.load_s, "runs": []}
    try:
        for cmd in DECIDER_BENCHES:
            full = [sys.executable, "-X", "utf8"] + cmd + ["--base-url", srv.base]
            t0 = time.time()
            logp = os.path.join(out, os.path.basename(cmd[0]).replace(".py", ".mirai-s.log"))
            with open(logp, "w", encoding="utf-8") as lf:
                r = subprocess.run(full, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT, timeout=6 * 3600)
            rec["runs"].append({"cmd": " ".join(cmd + ["--base-url", srv.base]), "exit": r.returncode,
                                "seconds": round(time.time() - t0, 1), "log": logp})
            ec.log(f"decider {cmd[0]}: exit {r.returncode} in {rec['runs'][-1]['seconds']} s")
            if r.returncode in (2, 3):          # the benches' NOT RUN: busy / unreachable -- stop, never a result
                break
    finally:
        rec.update(srv.stop())
    rec["PASS"] = all(x["exit"] == 0 for x in rec["runs"]) and len(rec["runs"]) == len(DECIDER_BENCHES)
    return rec


def step_lane(port: int, arm: dict, out: str) -> dict:
    """THE LANE'S THREE ARMS on Mirai S (layout v3; the same step as bench/kv_rank.py --layout v3 lane3 for Bonsai):
    main's decode tok/s on slot 0 at 8K / 32K / 64K (n=3 each) with the lane (slot 1) KEPT idle with a decider
    state above the conversation's cells, ACTIVE (a 10-read burst during the decode) and CLEARED after its burst,
    plus the next burst's re-prefill ms."""
    import threading
    srv = Launched("lane", arm, port, out)
    rec: dict = {"load_s": srv.load_s}
    lane_tokens = 3072                         # budget.LANE_TOKENS
    try:
        text = corpus()
        state = text[-(lane_tokens - 64) * 4:]

        def one(slot, prompt, n):
            return completion(srv.base, prompt, n, slot=slot)

        def burst():
            t0 = time.time()
            for _ in range(10):
                one(1, state, 0)
            return round((time.time() - t0) * 1000, 1)

        def clear_lane():
            one(1, "x", 0)
        for k in (8, 32, 64):
            convo = text[: k * 1024 * 4]
            row = {"kept": [], "active": [], "cleared": [], "reprefill_ms": [], "burst_ms": []}
            one(0, convo, 0)
            one(1, state, 0)
            for _ in range(3):
                row["kept"].append(one(0, convo, 256)["tps"])
            for _ in range(3):
                got: list = []
                th = threading.Thread(target=lambda: got.append(burst()))
                th.start()
                row["active"].append(one(0, convo, 256)["tps"])
                th.join()
                row["burst_ms"].append(got[0] if got else None)
            for _ in range(3):
                burst()
                clear_lane()
                row["cleared"].append(one(0, convo, 256)["tps"])
                t0 = time.time()
                one(1, state, 0)
                row["reprefill_ms"].append(round((time.time() - t0) * 1000, 1))
                clear_lane()
            med = {a_: (sorted(v)[1] if len(v) == 3 and None not in v else None)
                   for a_, v in (("kept", row["kept"]), ("active", row["active"]), ("cleared", row["cleared"]))}
            cmin = min((x for x in row["cleared"] if x), default=None)
            row["median_tps"] = med
            row["material"] = {a_: (med[a_] is not None and cmin is not None and med[a_] < cmin)
                               for a_ in ("kept", "active")}
            rec[f"{k}k"] = row
            ec.log(f"lane {k}k: kept {med['kept']} active {med['active']} cleared {med['cleared']} tok/s")
    finally:
        rec.update(srv.stop())
    return rec


def step_switches(name: str, arm: dict, port: int, out: str, base_greedy: dict | None) -> dict:
    srv = Launched(name, arm, port, out)
    rec: dict = {"load_s": srv.load_s}
    try:
        text = corpus()
        for k in (8, 32):
            r = completion(srv.base, text[: k * 1024 * 4], 256)
            rec[f"{k}k"] = {"greedy_sha256": hashlib.sha256(r["content"].encode()).hexdigest(),
                            "prompt_tps": r["prompt_tps"], "tps": r["tps"]}
            if base_greedy:
                rec[f"{k}k"]["same_as_base"] = rec[f"{k}k"]["greedy_sha256"] == base_greedy.get(k)
    finally:
        rec.update(srv.stop())
    return rec


def step_align(name: str, arm: dict, port: int, out: str) -> dict:
    """A 6-turn chat, sampled, asking for emoji-heavy answers; the next request's cache reuse per turn."""
    srv = Launched(name, arm, port, out)
    rec: dict = {"turns": []}
    try:
        msgs = [{"role": "user", "content": "Describe a garden in three sentences, with several emoji."}]
        s = ec.Server(port)
        for i in range(6):
            r = s.chat(msgs, max_tokens=2048, thinking=True, sampling=ec.PROD_SAMPLING, seed=3000 + i,
                       extra={"id_slot": 0, "cache_prompt": True})
            m = r["choices"][0]["message"]
            t = r.get("timings") or {}
            rec["turns"].append({"prompt_n": t.get("prompt_n"), "cache_n": t.get("cache_n")})
            msgs += [{"role": "assistant", "content": m.get("content") or "",
                      "reasoning_content": m.get("reasoning_content") or ""},
                     {"role": "user", "content": "Another one, different emoji."}]
    finally:
        rec.update(srv.stop())
    return rec


def step_needles(name: str, arm: dict, port: int, out: str, fitted: int | None) -> dict:
    srv = Launched(name, arm, port, out)
    rec: dict = {}
    try:
        text = corpus()
        want = [1024, 32768, 131072] + ([fitted - 8192] if fitted and fitted - 8192 > 131072 else [])
        depths = [d for d in want if fits(d, fitted, 4096)]
        if fitted and cap_depth(fitted, 4096) not in depths and cap_depth(fitted, 4096) > max(depths or [0]):
            depths.append(cap_depth(fitted, 4096))           # the deepest that fits the window: the window's own end
        rec["depths"] = depths
        rec["skipped_depths"] = [d for d in want if d not in depths]
        for depth in depths:
            ok = 0
            for i in range(5):
                key = hashlib.sha256(f"{depth}-{i}".encode()).hexdigest()[:8].upper()
                try:
                    r = ec.Server(port).chat([{"role": "user", "content": needle_prompt(depth, key, text[i * 997:])}],
                                             max_tokens=4096, thinking=True,
                                             sampling={"temperature": 0.0, "top_k": 1}, seed=0)
                except RuntimeError as e:
                    rec[f"{depth}-{i}"] = {"hit": False, "error": str(e)[:200]}
                    continue
                ans = r["choices"][0]["message"].get("content") or ""
                ok += key in ans
                rec[f"{depth}-{i}"] = {"hit": key in ans, "answer": ans[:80]}
            rec[f"hits_{depth}"] = ok
        rec["PASS"] = all(rec[f"hits_{d}"] == 5 for d in depths)
    finally:
        rec.update(srv.stop())
    return rec


def step_speed(name: str, arm: dict, port: int, out: str) -> dict:
    srv = Launched(name, arm, port, out)
    rec: dict = {"load_s": srv.load_s}
    np_ = int(arm["args"].get("-np", "2"))
    ctx = int(arm["args"].get("-c") or 0) or None
    ks = [k for k in (4, 32, 64, 128) if fits(k * 1024, ctx, 256)]
    if ctx and 128 not in ks:
        ks.append(cap_depth(ctx, 256) // 1024)                # the deepest that fits, in place of 128K
    rec["depths_k"] = ks
    try:
        text = corpus()
        for k in ks:
            runs = []
            for rep in range(3):
                prompt = text[rep * 4096: rep * 4096 + k * 1024 * 4]
                r = completion(srv.base, prompt, 256, slot=rep % np_)
                runs.append({kk: r[kk] for kk in ("prompt_n", "prompt_tps", "predicted_n", "tps")})
            rec[f"{k}k"] = runs
        if np_ >= 2:
            completion(srv.base, text[: 64 * 1024 * 4], 0, slot=1)
            rec["8k_with_idle_64k"] = [
                {kk: r[kk] for kk in ("prompt_n", "prompt_tps", "predicted_n", "tps")}
                for r in (completion(srv.base, text[rep * 997: rep * 997 + 8 * 1024 * 4], 256, slot=0)
                          for rep in range(3))]
    finally:
        rec.update(srv.stop())
    return rec


def selftest() -> int:
    fails = 0

    def check(ok, what):
        nonlocal fails
        print(("ok    " if ok else "FAIL  ") + what)
        fails += 0 if ok else 1

    a = arms(98304)
    c = command(a["mtp"], 18100, binary="X")
    check(c[c.index("-c") + 1] == "98304" and c.count("-c") == 1, "-c is overridden in place, not duplicated")
    check(c[c.index("--spec-type") + 1] == "draft-mtp" and c[c.index("--spec-draft-n-max") + 1] == "3",
          "the mtp arm adds the card's draft settings")
    check(c[c.index("--port") + 1] == "18100" and c[0].endswith("llama-server.exe"), "port and binary land")
    c1 = command(a["mtp-np1"], 18100, binary="X")
    check(c1[c1.index("-np") + 1] == "1" and c1.count("-np") == 1, "mtp-np1 is the card's single slot")
    check(a["off-gqa"]["env"] == {"GGML_CUDA_FA_VEC_GQA": "0"} and a["off-align"]["checks"] == ["align"],
          "each switch arm turns exactly one change off")
    check("--mmproj" in MIRAI_ARGV and "--no-mmproj-offload" not in MIRAI_ARGV, "the mmproj is on the GPU")
    # fit: 10,000 MiB at 64K and 12,000 at 128K -> 2,000 MiB per 65,536 cells; fixed 8,000; 16,311 total.
    f = derive_fit([(65536, 10000), (131072, 12000)], 16311)
    per = 2000 / 65536
    check(f["fixed_mib"] == 8000 and f["bytes_per_cell"] == round(per * 2**20), "fit: slope and intercept")
    check(f["max_ctx"] == int((16311 - 1000 - 8000) / per) // 1024 * 1024 and f["max_ctx"] % 1024 == 0,
          "fit: the largest -c that keeps the margin, in whole 1,024s")
    NO_MMPROJ["on"] = True
    cn = command(a["base"], 18100, binary="X")
    NO_MMPROJ["on"] = False
    check("--mmproj" not in cn and "--jinja" in cn and cn[cn.index("-m") + 1] == MODEL,
          "--no-mmproj drops the projector and nothing else")
    p = needle_prompt(100, "ABCD1234", "x" * 1000)
    check("ABCD1234" in p and p.endswith("code only."), "the needle is in the haystack")
    print(f"\n{'all passed' if not fails else f'{fails} FAILED'}")
    return 1 if fails else 0


class StopWindow(Exception):
    """A STOP-WINDOW check between stages said stop: production is restored in main()'s finally."""


STACK_LOG = os.path.join(ROOT, "logs", "stack.log")
# What llama-swap logs for a GENERATION the proxy sends (a client turn, or anything it runs for one): a POST for a
# completion, or a POST to the main card's model. GETs are not counted -- the watchdog's /upstream/bonsai/health
# polls every 5 minutes. (The gate's own arms run their own llama-server on --port, never through llama-swap.)
CLIENT_RE = re.compile(r'"POST (?:/v1/(?:chat/)?completions|/v1/responses|/v1/messages|'
                       r'/upstream/(?:bonsai|bonsai-agent)/)')
_LOG_AT = {"offset": None}


def stop_reason(out: str, t_start: float) -> str | None:
    """Between stages (coordinator 2026-10-01): a STOP-WINDOW file (in --out or the repo root: the coordinator's or
    main's signal), or a client request since the window started -- llama-swap's own log (logs/stack.log) shows a
    request for the main card's model after the window began (read from the offset recorded at its start), or the
    corpus's newest client event (mcp/idle.py last_request_at, read-only) is newer than the start -> why to stop;
    None to go on."""
    for f in (os.path.join(out, "STOP-WINDOW"), os.path.join(ROOT, "STOP-WINDOW")):
        if os.path.exists(f):
            return f"STOP-WINDOW file: {f}"
    try:
        if _LOG_AT["offset"] is not None and os.path.exists(STACK_LOG):
            with open(STACK_LOG, "rb") as f:
                f.seek(_LOG_AT["offset"])
                new = f.read().decode("utf-8", "replace")
            hit = next((ln for ln in new.splitlines() if CLIENT_RE.search(ln)), None)
            if hit:
                return f"a request for the main card's model reached llama-swap: {hit.strip()[:200]}"
    except OSError as e:
        ec.log(f"stop check: {STACK_LOG} could not be read ({e}); going on")
    try:
        sys.path.insert(0, os.path.join(ROOT, "mcp"))
        import idle                                                  # noqa: E402
        at, _how = idle.last_request_at()
        if at and at > t_start:
            return f"a client request arrived at {time.strftime('%H:%M:%S', time.localtime(at))}"
    except Exception as e:                                           # noqa: BLE001
        ec.log(f"stop check: the corpus could not be read ({type(e).__name__}: {e}); going on")
    return None


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--window", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--port", type=int, default=18096)
    ap.add_argument("--steps", nargs="*",
                    default=["kernels", "fit", "api", "lane", "decider", "switches", "align", "needles", "corrupt",
                             "speed"])
    ap.add_argument("--arms", nargs="*")
    ap.add_argument("--no-mmproj", action="store_true",
                    help="every arm without the projector (images on bonsai-vision, the A4000)")
    ap.add_argument("--ctx", type=int, help="the -c for every arm (default: the fit step's max_ctx, else "
                                            f"{CTX_START})")
    ap.add_argument("--card-only", action="store_true",
                    help="the window is the 5060 Ti alone: kernel cases on it only, never the A4000")
    ap.add_argument("--np", type=int, choices=(1, 2),
                    help="force the layout (default: the lane step decides; -np 2 without it)")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    NO_MMPROJ["on"] = bool(a.no_mmproj)
    CARD_ONLY["on"] = bool(a.card_only)
    if a.np:
        LAYOUT.update(np=a.np, why=f"--np {a.np}")
    table = arms(a.ctx or CTX_START)
    names = a.arms or list(table)
    if a.dry_run:
        for n in names:
            print(f"{n:9s} {table[n]['checks']}\n  env {table[n]['env']}\n  {' '.join(command(table[n], a.port))}")
        return 0
    if not a.window or not a.out:
        print("refusing: this unloads production bonsai. Pass --window (the operator's GPU window) and --out.")
        return 2
    if not engine_bin():
        print(f"refusing: {ENGINE} has no shipped build in engines/manifest.yaml (scripts/build_engine.py {ENGINE})")
        return 2
    os.makedirs(a.out, exist_ok=True)
    prod_argv, _prod_env = ec.production_command()
    results: dict = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "arms": {}, "steps": a.steps,
                     "mmproj": not NO_MMPROJ["on"]}

    def save():
        json.dump(results, open(os.path.join(a.out, "gate.json"), "w"), indent=1)

    # the main card's gpu lane is held FIRST (a job already running finishes; no new one starts while the stack
    # quiets), then the quiet wait; released at every exit
    sys.path.insert(0, os.path.join(ROOT, "mcp"))
    lane_hold = ec.hold_lane("Mirai S gate (bench/mirai_s_gate.py)", 12 * 3600)
    results["lane_hold"] = lane_hold
    save()
    # a gpu job already running on the main card (a skill prove's several generations) finishes first: its next
    # generation would otherwise reach llama-swap mid-window and load bonsai beside the arm
    import jobs                                                      # noqa: E402
    t_jobs = time.time()
    while jobs.running("gpu") and time.time() - t_jobs < 3600:
        ec.log(f"waiting: {jobs.running('gpu')} gpu job(s) running on the main card")
        time.sleep(20)
    why = ec.wait_quiet(prod_argv, 3600)
    if not why and jobs.running("gpu"):
        why = f"{jobs.running('gpu')} gpu job(s) still running on the main card after 3,600 s"
    if why:
        results["lane"] = ec.release_lane(lane_hold)
        results["not_run"] = why
        save()
        print(f"not run: {why}")
        return 3
    t_start = time.time()
    _LOG_AT["offset"] = os.path.getsize(STACK_LOG) if os.path.exists(STACK_LOG) else None

    def gate(stage: str) -> None:
        why_stop = stop_reason(a.out, t_start)
        if why_stop:
            results["stopped"] = {"before": stage, "why": why_stop, "at": time.strftime("%H:%M:%S")}
            ec.log(f"STOP before {stage}: {why_stop}")
            raise StopWindow(why_stop)
    try:
        ec.http("POST", f"{ec.SWAP}/api/models/unload/{ec.PROD_ID}", timeout=120)
        t0 = time.time()
        while ec.PROD_ID in ec.running() or any("llama-server" in p for p in ec.card_pids()):
            if time.time() - t0 > 180:
                raise RuntimeError("bonsai did not leave the card in 180 s")
            time.sleep(2)
        if "lane" in a.steps:
            # FIRST, at -np 2 (the lane needs its slot): it decides the layout every later arm runs at
            # ("lane3": results["lane"] is the gpu-lane hold's record, ec.release_lane)
            gate("lane")
            results["lane3"] = step_lane(a.port, dict(table["base"], args={**table["base"]["args"], "-np": "2"}),
                                         a.out)
            if not a.np:
                mat = {k: (results["lane3"].get(k) or {}).get("material") or {} for k in ("8k", "32k", "64k")}
                active = [k for k, m in mat.items() if m.get("active")]
                if active:
                    LAYOUT.update(np=1, why=f"the lane is MATERIAL (ACTIVE below CLEARED's minimum at {active}): "
                                            "-np 1, jjava and side calls on bonsai-a4000 (the operator's rule for "
                                            "Bonsai and Flash-Next, 2026-09-30)")
                else:
                    LAYOUT.update(np=2, why="the lane is not material at any depth: -np 2 (the conversation + "
                                            "the lane)")
                table = arms(a.ctx or CTX_START)
            results["layout"] = dict(LAYOUT)
            ec.log(f"layout: -np {LAYOUT['np']} ({LAYOUT['why']})")
            save()
        if "kernels" in a.steps:
            gate("kernels")
            results["kernels"] = step_kernels(a.out)
            save()
        if "fit" in a.steps:
            gate("fit")
            results["fit"] = {"base": step_fit("base", table["base"], a.port, a.out),
                              "mtp": step_fit("mtp", table["mtp"], a.port, a.out)}
            save()
            if not a.ctx:
                d = [(results["fit"][k].get("derived") or {}).get("max_ctx") for k in ("base", "mtp")]
                d = [x for x in d if x]
                if d:
                    table = arms(min(d))      # one window for every arm: the smaller (MTP holds more)
                    results["ctx_used"] = min(d)
            results["fit"]["np"] = LAYOUT["np"]
        fitted = results.get("ctx_used") or a.ctx
        if "api" in a.steps:
            gate("api")
            results["api"] = step_api(a.port, table["base"], a.out)
            save()
        if "decider" in a.steps:
            gate("decider")
            api = results.get("api") or {}
            if (api.get("chat") or {}).get("has_logprobs") and (api.get("chat") or {}).get("top_logprobs_n", 0) >= 5:
                results["decider"] = step_decider(a.port, table["base"], a.out)
            else:
                results["decider"] = {"PASS": False, "not_run": "the api step did not confirm logprobs / "
                                                                "top_logprobs (run it first: --steps api decider)"}
            save()
        base_greedy = None
        if "switches" in a.steps:
            gate("switches")
            r = step_switches("switch-base", table["base"], a.port, a.out, None)
            results["switches"] = {"base": r}
            base_greedy = {k: r[f"{k}k"]["greedy_sha256"] for k in (8, 32)}
            save()
        for n in names:
            arm, r = table[n], results["arms"].setdefault(n, {})
            if "switches" in a.steps and "switches" in arm["checks"]:
                gate(f"{n} switches")
                r["switches"] = step_switches(n, arm, a.port, a.out, base_greedy)
                save()
            if "align" in a.steps and "align" in arm["checks"]:
                gate(f"{n} align")
                r["align"] = step_align(n, arm, a.port, a.out)
                results["arms"].setdefault("base", {})["align"] = step_align("align-base", table["base"],
                                                                             a.port, a.out)
                save()
            if "needles" in a.steps and "needles" in arm["checks"]:
                gate(f"{n} needles")
                r["needles"] = step_needles(n, arm, a.port, a.out, fitted)
                save()
            if "corrupt" in a.steps and "corrupt" in arm["checks"]:
                gate(f"{n} corrupt")
                ec.VISION["on"] = not NO_MMPROJ["on"]
                r["corrupt"] = ec.run_arm(n, os.path.join(engine_bin(), "llama-server.exe"), arm["env"],
                                          _argv(), {"CUDA_VISIBLE_DEVICES": ec.CARD_UUID}, a.port, a.out,
                                          3, 16, 10, arm["args"])
                save()
            if "speed" in a.steps and "speed" in arm["checks"]:
                gate(f"{n} speed")
                r["speed"] = step_speed(n, arm, a.port, a.out)
                save()
    except StopWindow:
        pass
    finally:
        results["restore"] = ec.restore(prod_argv)
        results["lane"] = ec.release_lane(lane_hold)
        results["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        save()
    print(json.dumps({k: (v.get("PASS") if isinstance(v, dict) else v)
                      for k, v in results.items() if k in ("kernels", "api", "decider")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
