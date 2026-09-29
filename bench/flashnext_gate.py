"""The GPU gate for Flash-Next's engines and the Strata ports (docs/FLASH-NEXT.md, docs/ENGINES.md "Strata's MoE
work, ported"). Waits for the operator's "GPU free": it unloads production `bonsai` from the 5060 Ti, one GPU
consumer at a time, through bench/engine_corruption.py's window discipline (quiet stack, the worker's gpu lane
paused, a guard that kills the arm if llama-swap starts anything on the card, production restored in `finally`).

Every arm is its own llama-server (or tool) process with FLASHNEXT_ARGV plus the arm's binary, env and flag
overrides. The steps, and what each one gates:

  kernels  test-backend-ops MUL_MAT / MUL_MAT_ID for every quant type in the IQ2_XS file, on CUDA0 (the 5060 Ti,
           sm_120) and CUDA1 (the A4000, sm_86). PASS: every case passes. (ggml-org#27902: IQ2_S/IQ3_S fail on
           Blackwell with CUDA 13.2; we build with 12.8 -- this is where that is checked.)
  fit      one base launch: VRAM and RAM after load and after a warm-up, load time; derives --n-cpu-moe and the
           expert-cache slots from the card's measured free memory and a 1,000 MiB margin (bonsai-ada-surgery's
           soak, docs/ENGINES.md "The VRAM line"). Reported.
  exact    Strata's "bit-identical" steps (bench/results/2026-09-28-prefill-speed, the GDN state hash): a change
           that only moves WHERE bytes live or WHEN they are copied must not change a bit. Per arm: an 8K and a
           32K prompt read with n_predict 0, then the slot saved (--slot-save-path; the file holds the KV and the
           recurrent state) and hashed, plus a 256-token greedy continuation. PASS: hashes and text equal the
           base arm's. Arms: moe-off (the patched build with every switch off), prefetch, pinned, largepages,
           eager.
  kl       Strata's teacher-forced check for steps that change rounding (the expert cache splits each MoE sum
           between GPU and CPU): llama-perplexity --kl-divergence at -b/-ub 16 (the CPU path the cache serves)
           against the base engine's logits. The yardstick is Strata's: the SAME base engine with only its
           batch changed (-b/-ub 8), measured in the same run. PASS: KL <= the yardstick's and same-top-1 >=
           the yardstick's; otherwise reported for the operator, never auto-passed.
  needles  5 needles at 1K, 32K and 128K (Strata: 5/5 at every depth). PASS: 15/15.
  corrupt  engine_corruption.py's checks (greedy rep identity, 16 sampled runs scanned for the corruption
           symptoms, 10 tool calls; vision with --vision). PASS: its own rules.
  speed    prefill and decode tok/s at 4K / 32K / 64K / 128K, n=3, from llama-server's own timings, and the
           card's lowest free VRAM. Reported, never gated here.

    python bench/flashnext_gate.py --dry-run                 # print every arm's command, touch nothing
    python bench/flashnext_gate.py --selftest                # the pure parts, offline
    python bench/flashnext_gate.py --window --out DIR [--steps kernels fit exact kl needles corrupt speed]
                                   [--arms base moe-off ...]

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
FN = f"{MODELS}/flash-next"
SHARD1 = f"{FN}/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf"
MMPROJ = f"{FN}/mmproj-Qwen3.8-Flash-Next-BF16.gguf"
MTP_DRAFT = f"{FN}/mtp-Qwen3.8-Flash-Next-Q8_0-shared-embd.gguf"
PROFILE = os.environ.get("FLASHNEXT_PROFILE", f"{FN}/expert-profile-strata-d551edf4.bin")
ENG = "C:/Users/jwals/engines"
BIN = {
    "upstream": f"{ENG}/llama-upstream-1b446647/src/build/bin",
    "mtp": f"{ENG}/llama-upstream-mtp-3534183c/src/build/bin",
    "moe": os.environ.get("FLASHNEXT_MOE_BIN", ""),   # filled from engines/manifest.yaml below
}
MARGIN_MIB = 1000                                      # docs/ENGINES.md "The VRAM line" (their soak)
N_LAYER = 48
EXPERT_BYTES_PER_LAYER = 35_454_976_000 // N_LAYER     # read from shard 1 (docs/FLASH-NEXT.md section 2)
QUANT_TYPES = ["iq2_s", "iq3_s", "iq1_m", "iq2_xxs", "iq4_xs", "iq4_nl", "q2_0", "q6_K", "q8_0", "bf16"]

# The proposed llama-swap entry's argv (docs/FLASH-NEXT.md section 4), sampling as config.yaml's macro.
FLASHNEXT_ARGV = [
    "llama-server.exe", "--port", "${PORT}",
    "-m", SHARD1, "--mmproj", MMPROJ,
    "-dev", "CUDA0", "-ngl", "999", "--n-cpu-moe", "42",
    "-lm", "mmap", "--lazy-mode", "on",
    "-c", "262144", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "-fa", "on",
    # --kv-unified: one pool the slots share, so one conversation can use the whole -c (without it each
    # slot gets -c / -np = 65,536 and the 128K needle was refused: 2026-09-28 run, HTTP 400
    # exceed_context_size_error, n_ctx 65536). Bonsai's production pool is unified the same way.
    "-np", "4", "--kv-unified", "-b", "2048", "-ub", "512",
    "--jinja", "--reasoning-format", "deepseek", "--no-context-shift", "--no-cache-idle-slots",
    "--reasoning-budget", "32768",
    "--temp", "1.0", "--top-p", "0.95", "--top-k", "20", "--min-p", "0.0",
]


def shipped_bin(engine: str, env: str = "") -> str:
    """The directory of an engine's shipped llama-server (engines/manifest.yaml), or the env override."""
    if env and os.environ.get(env):
        return os.environ[env]
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import build_engine as be                                         # noqa: E402
    m = be.load_yaml(be.MANIFEST)
    p = ((m["engines"].get(engine) or {}).get("shipped") or {}).get("path", "")
    return os.path.dirname(p)


def moe_bin() -> str:
    return BIN["moe"] or shipped_bin("llama-upstream-moe")


def flash_bin() -> str:
    """llama-upstream-flash (2026-09-29): the moe engine + MTP (#28243) + the AVX2 Q2_0 kernel + the cache fix."""
    return shipped_bin("llama-upstream-flash", "FLASHNEXT_FLASH_BIN")


SHARD1_Q2 = f"{FN}/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf"
# one thread per logical CPU of the 8 P-cores (i7-13700K: logical 0-15 are the P-cores' two threads each, 16-23 the
# E-cores; read with GetLogicalProcessorInformationEx semantics in docs/FLASH-NEXT.md section 8)
P_CORES = {"-t": "16", "-C": "0xFFFF", "--cpu-strict": "1"}
MTP_ARGS = {"--spec-type": "draft-mtp", "--spec-draft-model": MTP_DRAFT, "-ngld": "999", "--spec-draft-n-cpu-moe": "1",
            "--spec-draft-n-max": "3", "--spec-draft-p-min": "0.5"}


A4000_UUID = "GPU-43e37d0c-4104-9056-2552-6109d4d3382c"      # config.yaml's A4000 (CUDA1 by PCI order)
EXPERT_MIB_PER_LAYER = EXPERT_BYTES_PER_LAYER / 2**20          # 676 MiB (IQ2_XS routed experts of one layer)
A4000_RESERVE_MIB = 1500     # the margin (1,000) + CUDA1's own compute buffers at -ub 4096 (~500, an estimate the
                             # arm's measured A4000 peak replaces in the report)
# the A4000's on-demand models the gate may unload for the a4000 arm (llama-swap reloads them on their next use);
# embeddings and the reranker stay
A4000_ONDEMAND = ["imagegen-turbo", "imagegen", "bonsai-vision"]


def a4000_free() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", A4000_UUID],
                         capture_output=True, text=True, timeout=15).stdout
    return int(out.strip().splitlines()[0])


class A4000Watch:
    """The A4000's used memory every 2 s while an arm runs: the peak is what the placement needs there."""

    def __init__(self):
        import threading
        self.peak, self.samples, self._stop = 0, 0, threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                                      "-i", A4000_UUID], capture_output=True, text=True, timeout=15).stdout
                self.peak = max(self.peak, int(out.strip().splitlines()[0]))
                self.samples += 1
            except Exception:                                            # noqa: BLE001
                pass
            self._stop.wait(2)

    def stop(self) -> dict:
        self._stop.set()
        self._t.join(timeout=20)
        return {"a4000_peak_used_mib": self.peak, "samples": self.samples}


def a4000_layers(free_mib: int) -> int:
    """How many layers' routed experts fit on the A4000 with its measured free memory. Arithmetic only."""
    return max(0, min(N_LAYER, int((free_mib - A4000_RESERVE_MIB) // EXPERT_MIB_PER_LAYER)))


def a4000_args(m: int, slots_all: int) -> dict:
    """The LAST m layers' routed experts on CUDA1 (-ot), the first 48-m in RAM (--n-cpu-moe) with the profile cache
    over them (the same total VRAM slots as the all-RAM arm: slots x 48 / (48 - m) per RAM layer), every dense
    weight, the KV and the draft on CUDA0 (-ts 1,0, -devd CUDA0)."""
    k = N_LAYER - m
    layers = "|".join(str(i) for i in range(k, N_LAYER))
    return {"-dev": "CUDA0,CUDA1", "-ts": "1,0", "--n-cpu-moe": str(k),
            "-ot": rf"blk\.({layers})\.ffn_(up|gate|down)_exps\.weight=CUDA1",
            "--moe-expert-cache": str(max(1, (slots_all * N_LAYER) // max(1, k)))}


def arms(n_cpu_moe: int = 42, cache_slots: int = 0, all_slots: dict | None = None, a4000_m: int = 0) -> dict[str, dict]:
    """name -> {bin, env, args, checks}. `checks` says which steps the arm takes part in. `all_slots`: the
    expert-cache slots per layer when EVERY layer's experts are in RAM ({iq2, q2, mmcpu}; derive_cache_all from the
    fit2 step, else 1 -- a placeholder that only a --dry-run shows)."""
    up, mtp, moe, fl = BIN["upstream"], BIN["mtp"], moe_bin(), flash_bin()
    base_args = {"--n-cpu-moe": str(n_cpu_moe)}
    slots = str(cache_slots or 1)
    s = all_slots or {}
    all_slots, all_slots_q2, all_slots_mmcpu = (str(s.get(k) or 1) for k in ("iq2", "q2", "mmcpu"))
    strata = {"LLAMA_MOE_CACHE_PROFILE": PROFILE, "LLAMA_MOE_CACHE_POLICY": "strata"}
    return {
        "base":       {"bin": up,  "env": {}, "args": base_args, "checks": ["exact", "kl", "needles", "corrupt", "speed"]},
        "moe-off":    {"bin": moe, "env": {}, "args": base_args, "checks": ["exact"]},
        "prefetch":   {"bin": moe, "env": {}, "args": {**base_args, "--prefetch-experts-slots": "3"},
                       "checks": ["exact", "speed"]},
        "pinned":     {"bin": moe, "env": {"LLAMA_PIN_EXPERTS": "1"}, "args": base_args, "checks": ["exact", "speed"]},
        "largepages": {"bin": moe, "env": {"LLAMA_PIN_EXPERTS": "1", "GGML_CUDA_HOST_LARGE_PAGES": "1"},
                       "args": base_args, "checks": ["exact", "speed"]},
        "eager":      {"bin": up,  "env": {"CUDA_MODULE_LOADING": "EAGER"}, "args": base_args,
                       "checks": ["exact", "speed"]},
        "cache-lru":  {"bin": moe, "env": {}, "args": {**base_args, "--moe-expert-cache": slots},
                       "checks": ["kl", "needles", "corrupt", "speed"]},
        "cache-strata": {"bin": moe, "env": {"LLAMA_MOE_CACHE_PROFILE": PROFILE, "LLAMA_MOE_CACHE_POLICY": "strata"},
                         "args": {**base_args, "--moe-expert-cache": slots},
                         "checks": ["kl", "needles", "corrupt", "speed"]},
        "all":        {"bin": moe, "env": {"LLAMA_PIN_EXPERTS": "1", "LLAMA_MOE_CACHE_PROFILE": PROFILE,
                                           "LLAMA_MOE_CACHE_POLICY": "strata"},
                       "args": {**base_args, "--moe-expert-cache": slots, "--prefetch-experts-slots": "3",
                                "-ub": "2048"},
                       "checks": ["kl", "needles", "corrupt", "speed"]},
        # the KV placement question (docs/ENGINES.md "KV placement" 0038-0040, not yet ported to this base):
        # one stream per slot spans only its own cells by construction, at a fixed per-slot window
        "nounified":  {"bin": up,  "env": {}, "args": {**base_args, "--no-kv-unified": ""},
                       "checks": ["needles", "speed"]},
        "ub2048":     {"bin": up,  "env": {}, "args": {**base_args, "-ub": "2048"}, "checks": ["speed"]},
        "ub4096":     {"bin": up,  "env": {}, "args": {**base_args, "-b": "4096", "-ub": "4096"}, "checks": ["speed"]},
        "mtp":        {"bin": mtp, "env": {}, "args": {**base_args, "--spec-type": "draft-mtp",
                                                       "--spec-draft-model": MTP_DRAFT, "-ngld": "999",
                                                       "--spec-draft-n-cpu-moe": "1",
                                                       "--spec-draft-n-max": "3", "--spec-draft-p-min": "0.5"},
                       "checks": ["corrupt", "speed"]},
        # ---- 2026-09-29 (docs/FLASH-NEXT.md section 8): the causes of 18 tok/s, one arm each, then together.
        # threads: the base engine, 16 threads held to the P-cores
        "t16p":       {"bin": up,  "env": {}, "args": {**base_args, **P_CORES}, "checks": ["speed"]},
        # the AVX2 Q2_0 kernel alone (every down expert is Q2_0): changes rounding, so KL, not exact
        "flash-cpu":  {"bin": fl,  "env": {}, "args": base_args, "checks": ["kl", "speed"]},
        # + pinned experts and a 4K prompt batch (PCIe-bound prefill: fewer expert copies per token)
        "flash-ub4k": {"bin": fl,  "env": {"LLAMA_PIN_EXPERTS": "1"},
                       "args": {**base_args, "-b": "4096", "-ub": "4096"}, "checks": ["speed"]},
        # + the profile-ranked VRAM expert cache over EVERY layer (the ggml_get_rows crash fixed), pinned
        "flash-cache": {"bin": fl, "env": {"LLAMA_PIN_EXPERTS": "1", **strata},
                        "args": {**base_args, "--n-cpu-moe": "48", "--moe-expert-cache": all_slots},
                        "checks": ["kl", "needles", "speed"]},
        # + MTP (the draft layer's experts on the CPU)
        "flash-mtp":  {"bin": fl,  "env": {}, "args": {**base_args, **MTP_ARGS}, "checks": ["corrupt", "speed"]},
        # everything
        "flash-all":  {"bin": fl, "env": {"LLAMA_PIN_EXPERTS": "1", **strata},
                       "args": {**base_args, "--n-cpu-moe": "48", "--moe-expert-cache": all_slots, "-b": "4096",
                                "-ub": "4096", **P_CORES, **MTP_ARGS},
                       "checks": ["kl", "needles", "corrupt", "contract", "speed"]},
        # everything, the projector on the CPU (its ~1.9 GB of VRAM holds experts instead)
        "flash-all-mmcpu": {"bin": fl, "env": {"LLAMA_PIN_EXPERTS": "1", **strata},
                            "args": {**base_args, "--n-cpu-moe": "48", "--moe-expert-cache": all_slots_mmcpu,
                                     "-b": "4096", "-ub": "4096", **P_CORES, **MTP_ARGS, "--no-mmproj-offload": ""},
                            "checks": ["needles", "speed"]},
        # the operator's second-GPU question (2026-09-29): the experts the 5060 Ti cannot hold on the A4000 instead
        # of the CPU -- the last m layers' experts on CUDA1, the rest in RAM with the cache, as flash-all otherwise
        "flash-a4000": {"bin": fl, "env": {"LLAMA_PIN_EXPERTS": "1", **strata,
                                           "CUDA_VISIBLE_DEVICES": f"{ec.CARD_UUID},{A4000_UUID}"},
                        "args": {**base_args, "-b": "4096", "-ub": "4096", **P_CORES, **MTP_ARGS, "-devd": "CUDA0",
                                 **a4000_args(a4000_m or 1, int(all_slots))},
                        "checks": ["kl", "needles", "corrupt", "speed"], "a4000": True},
        # everything on the Q2_0 file (all experts Q2_0: the AVX2 kernel serves every CPU expert). A different
        # quant: its quality is ISTA's table, not our KL yardstick (no kl step)
        "flash-all-q2": {"bin": fl, "env": {"LLAMA_PIN_EXPERTS": "1", **strata},
                         "args": {**base_args, "-m": SHARD1_Q2, "--n-cpu-moe": "48", "--moe-expert-cache": all_slots_q2,
                                  "-b": "4096", "-ub": "4096", **P_CORES, **MTP_ARGS},
                         "checks": ["needles", "corrupt", "contract", "speed"]},
    }


def command(arm: dict, port: int) -> list[str]:
    return ec.arm_command(FLASHNEXT_ARGV, os.path.join(arm["bin"], "llama-server.exe"), port, arm["args"])


def derive_fit(free_mib_after_warm: int, n_cpu_moe_now: int) -> dict:
    """From the card's measured free memory: how many more expert layers fit on the GPU, or how many
    expert-cache slots per CPU layer, keeping the margin. Arithmetic only."""
    spare = max(0, free_mib_after_warm - MARGIN_MIB) * 1024 * 1024
    layer = EXPERT_BYTES_PER_LAYER
    more_layers = spare // layer
    per_expert = layer // 512
    cpu_layers = n_cpu_moe_now
    slots = (spare // per_expert // cpu_layers) if cpu_layers else 0
    return {"spare_bytes": spare, "n_cpu_moe_if_static": max(0, n_cpu_moe_now - int(more_layers)),
            "cache_slots_per_cpu_layer": int(slots)}


EXPERT_BYTES_Q2 = (37_623_740_192 - 3_759_953_920) // (N_LAYER * 512)   # Q2_0 shard 1 less the dense part
MMPROJ_MIB = 1865            # the projector (865 MiB file) + its ~1 GB compute buffer (docs/FLASH-NEXT.md section 4)


def derive_cache_all(free_mib: int, extra_free_mib: int = 0, per_expert: int = EXPERT_BYTES_PER_LAYER // 512) -> int:
    """Slots per layer for a cache over all 48 layers, from the free VRAM measured with that very configuration
    at one slot per layer (the fit2 step), keeping the margin; `extra_free_mib` what an arm frees beside it (the
    projector off the card). Arithmetic only."""
    spare = max(0, free_mib + extra_free_mib - MARGIN_MIB) * 1024 * 1024
    return int(1 + spare // per_expert // N_LAYER)


KLD_RE = re.compile(r"Mean\s+KLD:\s+([0-9.eE+-]+)")
TOP1_RE = re.compile(r"Same top p:\s+([0-9.]+)\s*\u00b1?")


def parse_kld(text: str) -> dict:
    k = KLD_RE.search(text)
    t = TOP1_RE.search(text)
    return {"mean_kld": float(k.group(1)) if k else None, "same_top1_pct": float(t.group(1)) if t else None}


def kl_verdict(arm: dict, yard: dict) -> str:
    if None in (arm.get("mean_kld"), arm.get("same_top1_pct"), yard.get("mean_kld"), yard.get("same_top1_pct")):
        return "NOT RUN"
    ok = arm["mean_kld"] <= yard["mean_kld"] and arm["same_top1_pct"] >= yard["same_top1_pct"]
    return "PASS" if ok else "REPORT (outside the yardstick: the operator judges)"


def needle_prompt(depth_tokens: int, key: str, filler: str) -> str:
    """~depth_tokens of repository text with one sentence holding `key` in the middle (4 chars ~ 1 token)."""
    body = filler[: depth_tokens * 4]
    mid = len(body) // 2
    return (body[:mid] + f"\n\nThe secret code for the vault is {key}.\n\n" + body[mid:]
            + "\n\nWhat is the secret code for the vault? Answer with the code only.")


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ steps --
def step_kernels(out: str) -> dict:
    # the flash engine's own test-backend-ops when it is built: it compares each GPU backend against ITS CPU backend,
    # which carries the AVX2 Q2_0 kernel (so q2_0 checks the new CPU dot against CUDA's, and the reverse)
    fl = flash_bin()
    exe = os.path.join(fl if fl and os.path.exists(os.path.join(fl, "test-backend-ops.exe")) else BIN["upstream"],
                       "test-backend-ops.exe")
    rec: dict = {"exe": exe}
    for dev in ("CUDA0", "CUDA1"):
        for op in ("MUL_MAT", "MUL_MAT_ID"):
            for t in QUANT_TYPES:
                key = f"{dev}/{op}/{t}"
                r = subprocess.run([exe, "test", "-o", op, "-b", dev, "-p", f"type_a={t}"],
                                   capture_output=True, text=True, timeout=3600)
                m = re.search(r"(\d+)/(\d+) tests passed", r.stdout)
                rec[key] = {"passed": int(m.group(1)), "total": int(m.group(2))} if m else {"error": r.stdout[-400:]}
                ec.log(f"kernels {key}: {rec[key]}")
    rec["PASS"] = all(v.get("passed") == v.get("total") and v.get("total") for v in rec.values() if isinstance(v, dict))
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


def completion(base: str, prompt: str, n: int, slot: int = 0) -> dict:
    body = {"prompt": prompt, "id_slot": slot, "n_predict": n, "cache_prompt": True, "temperature": 0.0,
            "top_k": 1, "seed": 0}
    st, txt = ec.http("POST", base + "/completion", body, timeout=7200)
    if st != 200:
        raise RuntimeError(f"/completion: HTTP {st}: {txt[:300]}")
    r = json.loads(txt)
    t = r.get("timings") or {}
    # draft_n / draft_n_accepted: llama-server's speculative counts (MTP acceptance; absent without a draft)
    return {"content": r.get("content") or "", "prompt_n": t.get("prompt_n"), "prompt_tps": t.get("prompt_per_second"),
            "predicted_n": t.get("predicted_n"), "tps": t.get("predicted_per_second"),
            "draft_n": t.get("draft_n"), "draft_n_accepted": t.get("draft_n_accepted")}


def corpus() -> str:
    files = sorted(glob.glob(os.path.join(ROOT, "mcp", "*.py")))
    return "".join(open(f, encoding="utf-8", errors="replace").read() for f in files)


def step_exact(name: str, arm: dict, port: int, out: str) -> dict:
    save = os.path.join(out, f"slots-{name}")
    os.makedirs(save, exist_ok=True)
    srv = Launched(name, arm, port, out, ["--slot-save-path", save])
    rec: dict = {"load_s": srv.load_s}
    try:
        text = corpus()
        for k in (8, 32):
            prompt = text[: k * 1024 * 4]
            completion(srv.base, prompt, 0)
            fn = f"{k}k.bin"
            st, txt = ec.http("POST", srv.base + "/slots/0?action=save", {"filename": fn}, timeout=600)
            rec[f"{k}k_state_sha256"] = sha256_file(os.path.join(save, fn)) if st == 200 else f"HTTP {st} {txt[:100]}"
            rec[f"{k}k_greedy"] = completion(srv.base, prompt, 256)["content"]
    finally:
        rec.update(srv.stop())
    return rec


def step_kl(name: str, arm: dict, out: str, base_logits: str, batch: int = 16, write_base: bool = False) -> dict:
    exe = os.path.join(arm["bin"], "llama-perplexity.exe")
    if not os.path.exists(exe):
        exe = os.path.join(moe_bin(), "llama-perplexity.exe")   # the only tree that builds it
    textf = os.path.join(out, "kl-text.txt")
    if not os.path.exists(textf):
        open(textf, "w", encoding="utf-8").write(corpus()[: 4096 * 4 * 8])
    cmd = [exe, "-m", SHARD1, "-f", textf, "-c", "4096", "--chunks", "8", "-b", str(batch), "-ub", str(batch),
           "-dev", "CUDA0", "-ngl", "999", "--n-cpu-moe", arm["args"].get("--n-cpu-moe", "42"),
           "-lm", "mmap", "--lazy-mode", "on", "--kl-divergence-base", base_logits]
    if not write_base:
        cmd.append("--kl-divergence")
    for k, v in arm["args"].items():
        # what changes the arithmetic or where experts live; the batch stays the step's, the server-only flags
        # (slots, MTP, the projector) do not apply to llama-perplexity
        if k in ("--moe-expert-cache", "--prefetch-experts-slots", "-t", "-C", "--cpu-strict", "-ts", "-ot"):
            cmd += [k] + ([v] if v != "" else [])
        elif k in ("--n-cpu-moe", "-dev"):
            cmd[cmd.index(k) + 1] = v
    env = dict(os.environ)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = ec.CARD_UUID
    env.update(arm["env"])
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=14400)
    open(os.path.join(out, f"kl-{name}-b{batch}.log"), "w", encoding="utf-8").write(r.stdout + r.stderr)
    return parse_kld(r.stdout + r.stderr) if not write_base else {"base_written": r.returncode == 0}


def step_needles(name: str, arm: dict, port: int, out: str) -> dict:
    srv = Launched(name, arm, port, out)
    rec: dict = {}
    try:
        text = corpus()
        ok = 0
        for depth in (1024, 32768, 131072):
            for i in range(5):
                key = hashlib.sha256(f"{depth}-{i}".encode()).hexdigest()[:8].upper()
                r = ec.Server(port).chat([{"role": "user", "content": needle_prompt(depth, key, text[i * 997:])}],
                                         max_tokens=4096, thinking=True,
                                         sampling={"temperature": 0.0, "top_k": 1}, seed=0)
                ans = (r["choices"][0]["message"].get("content") or "")
                hit = key in ans
                ok += hit
                rec[f"{depth}-{i}"] = {"hit": hit, "answer": ans[:80]}
        rec["hits"] = ok
        rec["PASS"] = ok == 15
    finally:
        rec.update(srv.stop())
    return rec


def step_contract(name: str, arm: dict, port: int, out: str) -> dict:
    """What the proxy needs from llama-server (docs/FLASH-NEXT.md section 5, "What our features need from the
    engine"), on this arm's binary and flags: /slots, /props with the template, /apply-template, /tokenize with
    pieces, /completion on a named slot (timings.cache_n on a repeat), chat logprobs + top_logprobs (the decider),
    the per-request reasoning budget honoured, a tool call parsed from the served template, and /slots erase
    (501 without --slot-save-path is the answer the proxy expects). PASS: every item."""
    srv = Launched(name, arm, port, out)
    rec: dict = {}
    b = srv.base
    try:
        st, txt = ec.http("GET", b + "/slots")
        rec["slots"] = st == 200 and isinstance(json.loads(txt), list) and len(json.loads(txt)) >= 3
        st, txt = ec.http("GET", b + "/props")
        rec["props_template"] = st == 200 and "<|im_start|>" in (json.loads(txt).get("chat_template") or "")
        msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]
        st, txt = ec.http("POST", b + "/apply-template", {"messages": msgs})
        rec["apply_template"] = st == 200 and "<|im_start|>user" in (json.loads(txt).get("prompt") or "")
        st, txt = ec.http("POST", b + "/tokenize", {"content": " A", "with_pieces": True})
        toks = json.loads(txt).get("tokens") if st == 200 else None
        rec["tokenize_pieces"] = bool(toks) and isinstance(toks[0], dict) and "piece" in toks[0]
        p = corpus()[:6000]
        first = completion(b, p, 1, slot=1)
        again = ec.http("POST", b + "/completion", {"prompt": p, "id_slot": 1, "n_predict": 1, "cache_prompt": True,
                                                  "temperature": 0.0}, timeout=600)
        cache_n = (json.loads(again[1]).get("timings") or {}).get("cache_n") if again[0] == 200 else None
        rec["slot_cache_reuse"] = {"first_prompt_n": first.get("prompt_n"), "repeat_cache_n": cache_n,
                                   "ok": bool(cache_n) and cache_n >= (first.get("prompt_n") or 1) - 2}
        r = ec.Server(port).chat([{"role": "user", "content": "Answer with one letter: A, B or C. Which is first?"}],
                                 max_tokens=8, thinking=False, sampling={"temperature": 0.0}, seed=0,
                                 extra={"logprobs": True, "top_logprobs": 5})
        lp = ((r["choices"][0].get("logprobs") or {}).get("content") or [])
        rec["logprobs"] = bool(lp) and len(lp[0].get("top_logprobs") or []) == 5
        r = ec.Server(port).chat([{"role": "user", "content": "Think step by step: what is 17 * 23?"}],
                                 max_tokens=4096, thinking=True, sampling={"temperature": 0.0}, seed=0,
                                 extra={"reasoning_budget_tokens": 64})
        reasoning = r["choices"][0]["message"].get("reasoning_content") or ""
        n_reason = len(json.loads(ec.http("POST", b + "/tokenize", {"content": reasoning})[1]).get("tokens") or [])
        rec["reasoning_budget"] = {"reasoning_tokens": n_reason, "ok": n_reason <= 64 + 64}
        tools = [{"type": "function", "function": {"name": "read_file", "description": "Read a file",
                  "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
        r = ec.Server(port).chat([{"role": "user", "content": "Read the file README.md with the tool."}],
                                 max_tokens=2048, thinking=False, sampling={"temperature": 0.0}, seed=0,
                                 extra={"tools": tools})
        calls = r["choices"][0]["message"].get("tool_calls") or []
        rec["tool_call"] = bool(calls) and calls[0]["function"]["name"] == "read_file"
        st, _ = ec.http("POST", b + "/slots/1?action=erase", {})
        rec["slot_erase"] = st in (200, 501)
        rec["PASS"] = all((v.get("ok") if isinstance(v, dict) else v) for v in list(rec.values()))
    except Exception as e:                                               # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["PASS"] = False
    finally:
        rec.update(srv.stop())
    return rec


def step_speed(name: str, arm: dict, port: int, out: str) -> dict:
    srv = Launched(name, arm, port, out)
    rec: dict = {"load_s": srv.load_s}
    try:
        text = corpus()
        for k in (4, 32, 64, 128):
            runs = []
            for rep in range(3):
                prompt = text[rep * 4096: rep * 4096 + k * 1024 * 4]
                r = completion(srv.base, prompt, 256, slot=rep % 2)
                runs.append({kk: r[kk] for kk in ("prompt_n", "prompt_tps", "predicted_n", "tps", "draft_n",
                                                  "draft_n_accepted")})
            rec[f"{k}k"] = runs
        # placement: an idle 64K conversation on slot 2 while slot 3 decodes at 8K (the #59 shape)
        completion(srv.base, text[: 64 * 1024 * 4], 0, slot=2)
        rec["8k_with_idle_64k"] = [
            {kk: r[kk] for kk in ("prompt_n", "prompt_tps", "predicted_n", "tps")}
            for r in (completion(srv.base, text[rep * 997: rep * 997 + 8 * 1024 * 4], 256, slot=3) for rep in range(3))]
    finally:
        rec.update(srv.stop())
    return rec


def selftest() -> int:
    fails = 0

    def check(ok, what):
        nonlocal fails
        print(("ok    " if ok else "FAIL  ") + what)
        fails += 0 if ok else 1

    a = arms(42, 7)
    check(set(a["cache-strata"]["env"]) == {"LLAMA_MOE_CACHE_PROFILE", "LLAMA_MOE_CACHE_POLICY"},
          "cache-strata sets the profile and the policy")
    c = command(a["prefetch"], 18100)
    check(c[c.index("--prefetch-experts-slots") + 1] == "3" and c[c.index("--port") + 1] == "18100",
          "arm flags and port land in the argv")
    check(c[c.index("--n-cpu-moe") + 1] == "42", "--n-cpu-moe is overridden in place, not duplicated")
    f = derive_fit(1000 + 4096, 42)
    check(f["n_cpu_moe_if_static"] == 42 - (4096 * 1024 * 1024) // EXPERT_BYTES_PER_LAYER,
          "fit: the spare over the margin, in whole expert layers")
    check(parse_kld("Mean    KLD:   0.012345 ± 0.0001\nSame top p: 97.123 ± 0.1 %") ==
          {"mean_kld": 0.012345, "same_top1_pct": 97.123}, "parses llama-perplexity's KL summary")
    check(kl_verdict({"mean_kld": 0.1, "same_top1_pct": 95}, {"mean_kld": 0.2, "same_top1_pct": 90}) == "PASS",
          "within the yardstick passes")
    check(kl_verdict({"mean_kld": 0.3, "same_top1_pct": 95}, {"mean_kld": 0.2, "same_top1_pct": 90}).startswith("REPORT"),
          "outside the yardstick is reported, not passed")
    b = arms(42, 7, {"iq2": 60, "q2": 66, "mmcpu": 80})
    ca = command(b["flash-all-q2"], 18100)
    check(ca[ca.index("-m") + 1].endswith("Q2_0-00001-of-00002.gguf") and ca[ca.index("--moe-expert-cache") + 1] == "66"
          and ca[ca.index("--n-cpu-moe") + 1] == "48" and "--spec-type" in ca and ca[ca.index("-C") + 1] == "0xFFFF",
          "flash-all-q2: the Q2_0 file, its own slot count, every layer's experts in RAM, MTP, the P-cores")
    cm = command(b["flash-all-mmcpu"], 18100)
    check("--no-mmproj-offload" in cm and cm[cm.index("--moe-expert-cache") + 1] == "80", "flash-all-mmcpu")
    check(derive_cache_all(1000) == 1 and derive_cache_all(1000 + 48 * 3) >= 2 and
          derive_cache_all(1000, MMPROJ_MIB) > derive_cache_all(1000), "fit2: slots from the measured free VRAM")
    c4 = command(arms(42, 7, {"iq2": 60, "q2": 66, "mmcpu": 80}, 12)["flash-a4000"], 18100)
    check(c4[c4.index("--n-cpu-moe") + 1] == "36" and c4[c4.index("-dev") + 1] == "CUDA0,CUDA1"
          and c4[c4.index("-ts") + 1] == "1,0" and "blk\\.(36|37|" in c4[c4.index("-ot") + 1]
          and c4[c4.index("-ot") + 1].endswith("47)\\.ffn_(up|gate|down)_exps\\.weight=CUDA1")
          and c4[c4.index("--moe-expert-cache") + 1] == str(60 * 48 // 36),
          "flash-a4000: the last 12 layers' experts on CUDA1, 36 in RAM with the same total cache slots")
    check(a4000_layers(11076) == int((11076 - A4000_RESERVE_MIB) // EXPERT_MIB_PER_LAYER) and a4000_layers(100) == 0,
          "a4000: layers from the free VRAM less the reserve")
    p = needle_prompt(100, "ABCD1234", "x" * 1000)
    check("ABCD1234" in p and p.endswith("code only."), "the needle is in the haystack")
    print(f"\n{'all passed' if not fails else f'{fails} FAILED'}")
    return 1 if fails else 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--window", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--port", type=int, default=18095)
    ap.add_argument("--steps", nargs="*",
                    default=["kernels", "fit", "fit2", "exact", "kl", "needles", "corrupt", "contract", "speed"])
    ap.add_argument("--arms", nargs="*")
    ap.add_argument("--n-cpu-moe", type=int, default=42)
    ap.add_argument("--cache-slots", type=int, default=0)
    ap.add_argument("--all-slots", help="iq2,q2,mmcpu slots per layer for the all-layer cache arms (else fit2's)")
    ap.add_argument("--vision", action="store_true")
    ap.add_argument("--a4000-layers", type=int, default=0, help="flash-a4000: layers on CUDA1 (else from its free VRAM)")
    ap.add_argument("--free-a4000", action="store_true",
                    help=f"unload {A4000_ONDEMAND} from the A4000 before the flash-a4000 arm (llama-swap reloads on use)")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    all_slots = dict(zip(("iq2", "q2", "mmcpu"), map(int, a.all_slots.split(",")))) if a.all_slots else None
    if all_slots is None and a.out and os.path.exists(os.path.join(a.out, "gate.json")):
        try:
            all_slots = (json.load(open(os.path.join(a.out, "gate.json"), encoding="utf-8")).get("fit2") or {}).get("slots")
        except Exception:                                                # noqa: BLE001
            all_slots = None
    table = arms(a.n_cpu_moe, a.cache_slots, all_slots)
    names = a.arms or list(table)
    if a.dry_run:
        for n in names:
            print(f"{n:13s} {table[n]['checks']}\n  env {table[n]['env']}\n  {' '.join(command(table[n], a.port))}")
        return 0
    if not a.window or not a.out:
        print("refusing: this unloads production bonsai. Pass --window (the operator's GPU window) and --out.")
        return 2
    os.makedirs(a.out, exist_ok=True)
    prod_argv, _prod_env = ec.production_command()
    results: dict = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "arms": {}, "steps": a.steps}
    why = ec.wait_quiet(prod_argv, 3600)
    if why:
        print(f"not run: {why}")
        return 3
    sys.path.insert(0, os.path.join(ROOT, "mcp"))
    import jobs                                                        # noqa: E402
    try:
        jobs.pause("gpu", by=ec.LANE_BY, why="Flash-Next gate (bench/flashnext_gate.py)", ttl_seconds=6 * 3600)
        ec.http("POST", f"{ec.SWAP}/api/models/unload/{ec.PROD_ID}", timeout=120)
        t0 = time.time()
        while ec.PROD_ID in ec.running() or any("llama-server" in p for p in ec.card_pids()):
            if time.time() - t0 > 180:
                raise RuntimeError("bonsai did not leave the card in 180 s")
            time.sleep(2)
        if "kernels" in a.steps:
            results["kernels"] = step_kernels(a.out)
        if "fit" in a.steps:
            srv = Launched("fit", table["base"], a.port, a.out)
            completion(srv.base, corpus()[: 8192 * 4], 64)
            g = ec.gpu() or {}
            results["fit"] = {"load_s": srv.load_s, "gpu_after_warm": g, **srv.stop()}
            if g.get("free") is not None:
                results["fit"]["derived"] = derive_fit(min(g["free"], results["fit"]["gpu_min_free"] or g["free"]),
                                                       a.n_cpu_moe)
            json.dump(results, open(os.path.join(a.out, "gate.json"), "w"), indent=1)
        if "fit2" in a.steps and flash_bin() and not all_slots:
            # the all-layer cache configuration at ONE slot per layer (flash-all: -ub 4096, MTP, the profile), warmed
            # with a prompt longer than one 4K batch; its lowest free VRAM sizes the cache (derive_cache_all)
            probe = arms(a.n_cpu_moe, a.cache_slots, {"iq2": 1, "q2": 1, "mmcpu": 1})["flash-all"]
            srv = Launched("fit2", probe, a.port, a.out)
            completion(srv.base, corpus()[: 12288 * 4], 64)
            g = ec.gpu() or {}
            rec = {"load_s": srv.load_s, "gpu_after_warm": g, **srv.stop()}
            free = min(g.get("free") or 0, rec.get("gpu_min_free") or g.get("free") or 0)
            all_slots = {"iq2": derive_cache_all(free), "q2": derive_cache_all(free, 0, EXPERT_BYTES_Q2),
                         "mmcpu": derive_cache_all(free, MMPROJ_MIB)}
            rec["slots"] = all_slots
            results["fit2"] = rec
            table = arms(a.n_cpu_moe, a.cache_slots, all_slots)
            json.dump(results, open(os.path.join(a.out, "gate.json"), "w"), indent=1)
        base_logits = os.path.join(a.out, "kl-base.logits")
        # A rerun into the same --out reuses the base logits and the yardstick a finished kl step left there
        # (the base engine and the text are fixed; the logits file is written only by that step).
        prev = {}
        try:
            prev = json.load(open(os.path.join(a.out, "gate.json"), encoding="utf-8"))
        except Exception:                                                # noqa: BLE001
            prev = {}
        if "kl" in a.steps and os.path.exists(base_logits) and prev.get("kl_yardstick"):
            results["kl_yardstick"] = dict(prev["kl_yardstick"], reused_from=prev.get("started"))
            ec.log(f"kl: reusing the base logits and the yardstick from the run started {prev.get('started')}")
        elif "kl" in a.steps:
            step_kl("base", table["base"], a.out, base_logits, 16, write_base=True)
            results["kl_yardstick"] = step_kl("base", table["base"], a.out, base_logits, 8)
        def save():
            json.dump(results, open(os.path.join(a.out, "gate.json"), "w"), indent=1)

        # A rerun into the same --out carries forward what it does not redo: the arms not named this time, and
        # the kernels / fit records (so gate.json stays the one record deploy_flash_next.py reads).
        for k in ("kernels", "fit", "fit2"):
            if k not in results and prev.get(k):
                results[k] = prev[k]
        for k, v in (prev.get("arms") or {}).items():
            if k not in names:
                results["arms"][k] = v
        for n in names:
            a4w = None
            if table[n].get("a4000"):
                # the second GPU: free its on-demand models if asked, size the layers from its measured free memory,
                # and watch its peak for the report (what gpu_room would have to leave free)
                unloaded = []
                if a.free_a4000:
                    for mid in A4000_ONDEMAND:
                        if mid in ec.running():
                            ec.http("POST", f"{ec.SWAP}/api/models/unload/{mid}", timeout=120)
                            unloaded.append(mid)
                    time.sleep(5)
                free = a4000_free()
                m = a.a4000_layers or a4000_layers(free)
                table[n] = arms(a.n_cpu_moe, a.cache_slots, all_slots, m)[n]
                a4w = A4000Watch()
                results.setdefault("a4000", {})[n] = {"free_before_mib": free, "layers_on_a4000": m,
                                                      "unloaded": unloaded, "reserve_mib": A4000_RESERVE_MIB}
            arm, r = table[n], {}
            # Recorded as each step finishes, so a later step's failure keeps the earlier steps' results.
            results["arms"][n] = r
            done_before = (prev.get("arms") or {}).get(n) or {}

            def reuse(step: str, n=n, r=r, done_before=done_before) -> bool:
                """A rerun into the same --out keeps a step this arm FINISHED there (no error, no guard)."""
                got = done_before.get(step)
                if not isinstance(got, dict) or got.get("error") or got.get("guard"):
                    return False
                r[step] = dict(got, reused_from=prev.get("started"))
                ec.log(f"{step} {n}: reusing the run started {prev.get('started')}")
                save()
                return True
            if "exact" in a.steps and "exact" in arm["checks"] and done_before.get("exact"):
                # a rerun into the same --out keeps an arm's finished exact step (its hashes are the record)
                r["exact"] = dict(done_before["exact"], reused_from=prev.get("started"))
                ec.log(f"exact {n}: reusing the run started {prev.get('started')}")
            elif "exact" in a.steps and "exact" in arm["checks"]:
                r["exact"] = step_exact(n, arm, a.port, a.out)
                save()
            if "kl" in a.steps and "kl" in arm["checks"] and n != "base" and reuse("kl"):
                pass
            elif "kl" in a.steps and "kl" in arm["checks"] and n != "base":
                r["kl"] = step_kl(n, arm, a.out, base_logits, 16)
                r["kl"]["verdict"] = kl_verdict(r["kl"], results.get("kl_yardstick") or {})
                save()
            if "needles" in a.steps and "needles" in arm["checks"] and reuse("needles"):
                pass
            elif "needles" in a.steps and "needles" in arm["checks"]:
                r["needles"] = step_needles(n, arm, a.port, a.out)
                save()
            if "corrupt" in a.steps and "corrupt" in arm["checks"] and reuse("corrupt"):
                pass
            elif "corrupt" in a.steps and "corrupt" in arm["checks"]:
                ec.VISION["on"] = a.vision
                r["corrupt"] = ec.run_arm(n, os.path.join(arm["bin"], "llama-server.exe"), arm["env"],
                                          FLASHNEXT_ARGV, {"CUDA_VISIBLE_DEVICES": ec.CARD_UUID}, a.port, a.out,
                                          3, 16, 10, arm["args"])
                save()
            if "contract" in a.steps and "contract" in arm["checks"] and reuse("contract"):
                pass
            elif "contract" in a.steps and "contract" in arm["checks"]:
                r["contract"] = step_contract(n, arm, a.port, a.out)
                save()
            if "speed" in a.steps and "speed" in arm["checks"] and reuse("speed"):
                pass
            elif "speed" in a.steps and "speed" in arm["checks"]:
                r["speed"] = step_speed(n, arm, a.port, a.out)
                save()
            results["arms"][n] = r
            if a4w is not None:
                results["a4000"][n].update(a4w.stop())
            json.dump(results, open(os.path.join(a.out, "gate.json"), "w"), indent=1)
        base_exact = (results["arms"].get("base") or {}).get("exact") or {}
        for n, r in results["arms"].items():
            if "exact" in r and n != "base":
                keys = [k for k in base_exact if k.endswith(("_state_sha256", "_greedy"))]
                r["exact"]["PASS"] = bool(keys) and all(r["exact"].get(k) == base_exact.get(k) for k in keys)
    finally:
        results["restore"] = ec.restore(prod_argv)
        results["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        json.dump(results, open(os.path.join(a.out, "gate.json"), "w"), indent=1)
    print(json.dumps({n: {s: (v.get("PASS", v.get("verdict")) if isinstance(v, dict) else v) for s, v in r.items()}
                      for n, r in results["arms"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
