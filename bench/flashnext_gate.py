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
MTP_ARGS = {"--spec-type": "draft-mtp", "--spec-draft-model": MTP_DRAFT, "-ngld": "999", "--spec-draft-n-cpu-moe": "49",
            "--spec-draft-n-max": "3", "--spec-draft-p-min": "0.5",
            # an explicit split: with it the draft never divides the free memory (0 MiB on a full card: NaN splits,
            # devices.at() out of range -- "invalid vector subscript", 2026-09-29, bisect-*.log)
            "-ts": "1"}


A4000_UUID = "GPU-43e37d0c-4104-9056-2552-6109d4d3382c"      # config.yaml's A4000 (CUDA1 by PCI order)
EXPERT_MIB_PER_LAYER = EXPERT_BYTES_PER_LAYER / 2**20          # 676 MiB (IQ2_XS routed experts of one layer)
A4000_RESERVE_MIB = 1500     # the margin (1,000) + CUDA1's own compute buffers at -ub 4096 (~500, an estimate the
                             # arm's measured A4000 peak replaces in the report)
# the A4000's on-demand models the gate may unload for the a4000 arm (llama-swap reloads them on their next use);
# embeddings stay (the reranker beside it was removed 2026-10-01)
A4000_ONDEMAND = ["imagegen-turbo", "imagegen", "bonsai-vision", "clm-encoder"]


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


def static_layers(slots_per_layer: int) -> int:
    """The VRAM fit2 measured for a cache, as whole expert LAYERS on the card instead (slots x 48 of 512 experts
    per layer). 2026-09-29: the cache's multi-token GPU chain faults (CUDA illegal memory access in the KL step's
    batch-16 path and at a 142K prompt's tail batch, flash-cache.server.log / kl-flash-cache-b16.log) and bought
    ~3% on decode at 46 slots a layer, so the combined arms place whole layers, as llama.cpp does, instead."""
    return max(0, int(slots_per_layer * N_LAYER // 512))


def a4000_args(m: int, g: int) -> dict:
    """The first k = 48 - m - g layers' routed experts in RAM (--n-cpu-moe), the next m on CUDA1 (-ot), the last g
    on CUDA0; every dense weight, the KV and the draft on CUDA0 (-ts 1,0, -devd CUDA0)."""
    k = max(0, N_LAYER - m - g)
    layers = "|".join(str(i) for i in range(k, min(N_LAYER, k + m)))
    return {"-dev": "CUDA0,CUDA1", "-ts": "1,0", "--n-cpu-moe": str(k),
            "-ot": rf"blk\.({layers})\.ffn_(up|gate|down)_exps\.weight=CUDA1"}


def arms(n_cpu_moe: int = 42, cache_slots: int = 0, all_slots: dict | None = None, a4000_m: int = 0) -> dict[str, dict]:
    """name -> {bin, env, args, checks}. `checks` says which steps the arm takes part in. `all_slots`: the
    expert-cache slots per layer when EVERY layer's experts are in RAM ({iq2, q2, mmcpu}; derive_cache_all from the
    fit2 step, else 1 -- a placeholder that only a --dry-run shows)."""
    up, mtp, moe, fl = BIN["upstream"], BIN["mtp"], moe_bin(), flash_bin()
    base_args = {"--n-cpu-moe": str(n_cpu_moe)}
    slots = str(cache_slots or 1)
    s = all_slots or {}
    all_slots, all_slots_q2, all_slots_mmcpu = (str(s.get(k) or 1) for k in ("iq2", "q2", "mmcpu"))
    g_iq2, g_q2 = static_layers(int(all_slots)), static_layers(int(all_slots_q2))
    strata = {"LLAMA_MOE_CACHE_PROFILE": PROFILE, "LLAMA_MOE_CACHE_POLICY": "strata"}
    table = {
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
                                                       "--spec-draft-n-cpu-moe": "49",
                                                       "--spec-draft-n-max": "3", "--spec-draft-p-min": "0.5",
            # an explicit split: with it the draft never divides the free memory (0 MiB on a full card: NaN splits,
            # devices.at() out of range -- "invalid vector subscript", 2026-09-29, bisect-*.log)
            "-ts": "1"},
                       "checks": ["corrupt", "speed"]},
        # ---- 2026-09-29 (docs/FLASH-NEXT.md section 8): the causes of 18 tok/s, one arm each, then together.
        # threads: the base engine, 16 threads held to the P-cores
        "t16p":       {"bin": up,  "env": {}, "args": {**base_args, **P_CORES}, "checks": ["speed"]},
        # the AVX2 Q2_0 kernel alone (every down expert is Q2_0): changes rounding, so KL, not exact
        "flash-cpu":  {"bin": fl,  "env": {}, "args": {**base_args, **NO_MMPROJ}, "checks": ["kl", "speed"]},
        # + pinned experts, every layer's experts in RAM and a 1K prompt batch (PCIe-bound prefill: half the expert
        # copies per token of -ub 512; -ub 4096 measured a 14,502 MiB compute buffer, 1024 3,770: vram-*.log)
        "flash-ub1k": {"bin": fl,  "env": {"LLAMA_PIN_EXPERTS": "1"},
                       "args": {**base_args, **NO_MMPROJ, "--n-cpu-moe": "48", "-b": "2048", "-ub": "1024"}, "checks": ["speed"]},
        # + the profile-ranked VRAM expert cache over EVERY layer (the ggml_get_rows crash fixed), pinned. Needles run
        # on flash-all (the same cache with MTP): a 128K needle prompt is ~15 min at this prompt rate
        "flash-cache": {"bin": fl, "env": {"LLAMA_PIN_EXPERTS": "1", **strata},
                        "args": {**base_args, **NO_MMPROJ, "--n-cpu-moe": "48", "--moe-expert-cache": all_slots},
                        "checks": ["kl", "speed"]},
        # + MTP (the draft layer's experts on the CPU)
        "flash-mtp":  {"bin": fl,  "env": {}, "args": {**base_args, **NO_MMPROJ, **MTP_ARGS}, "checks": ["corrupt", "speed"]},
        # everything
        "flash-all":  {"bin": fl, "env": {"LLAMA_PIN_EXPERTS": "1"},
                       "args": {**base_args, **NO_MMPROJ, "--n-cpu-moe": str(N_LAYER - g_iq2), "-b": "2048",
                                "-ub": "512", **P_CORES, **MTP_ARGS},
                       "checks": ["kl", "needles", "corrupt", "contract", "speed"]},
        # the operator's second-GPU question (2026-09-29): the experts the 5060 Ti cannot hold on the A4000 instead
        # of the CPU -- the last m layers' experts on CUDA1, the rest in RAM with the cache, as flash-all otherwise
        "flash-a4000": {"bin": fl, "env": {"LLAMA_PIN_EXPERTS": "1",
                                           "CUDA_VISIBLE_DEVICES": f"{ec.CARD_UUID},{A4000_UUID}"},
                        "args": {**base_args, **NO_MMPROJ, "-b": "2048", "-ub": "512", **P_CORES, **MTP_ARGS, "-devd": "CUDA0",
                                 **a4000_args(a4000_m or 1, g_iq2)},
                        "checks": ["kl", "corrupt", "speed"], "a4000": True},
        # everything on the Q2_0 file (all experts Q2_0: the AVX2 kernel serves every CPU expert). A different
        # quant: its quality is ISTA's table, not our KL yardstick (no kl step)
        "flash-all-q2": {"bin": fl, "env": {"LLAMA_PIN_EXPERTS": "1"},
                         "args": {**base_args, **NO_MMPROJ, "-m": SHARD1_Q2, "--n-cpu-moe": str(N_LAYER - g_q2),
                                  "-b": "2048", "-ub": "512", **P_CORES, **MTP_ARGS},
                         "checks": ["needles", "corrupt", "contract", "speed"]},
    }
    # WINDOW B STEP 1 (the coordinator, 2026-09-29): routing traces for the hit-rate simulation and the op profile
    opt = {"GGML_CUDA_OP_TIMING": "1", "GGML_CUDA_DISABLE_GRAPHS": "1"}
    table["probe-trace"] = dict(table["flash-cpu"], env={**table["flash-cpu"]["env"],
                                                        "GGML_MOE_LOG": "C:/Users/jwals/octo/moe-trace-flash-cpu.txt"},
                                args={**table["flash-cpu"]["args"], "--n-cpu-moe": "48"}, checks=["probe"])
    # -lv 4: 0010 prints with GGML_LOG_INFO, which common_log maps to TRACE (4) and drops at the default verbosity 3
    # (common/log.cpp common_log_get_verbosity; the first probe run, 2026-09-30, printed no table)
    lv = {"-lv": "4"}
    table["probe-opt-cpu"] = dict(table["flash-cpu"], env={**table["flash-cpu"]["env"], **opt},
                                  args={**table["flash-cpu"]["args"], **lv}, checks=["probe"])
    table["probe-opt-cache"] = dict(table["flash-cache"], env={**table["flash-cache"]["env"], **opt},
                                    args={**table["flash-cache"]["args"], **lv}, checks=["probe"])
    table["probe-opt-all"] = dict(table["flash-all"], env={**table["flash-all"]["env"], **opt},
                                  args={**table["flash-all"]["args"], **lv}, checks=["probe"])
    # M3 step 1 (0012): the attention KV in mapped host memory, the operator's layout (-np 2 + the lane). fit-kvmap
    # measures the VRAM that frees (every layer's experts in RAM; its probe records the card's lowest free MiB);
    # the kvmap arms then put G whole expert layers on the card: FLASHNEXT_KVMAP_GPU (with MTP) and
    # FLASHNEXT_KVMAP_GPU_NOMTP (without: the draft's ~1 GB and the MTP recurrent snapshots freed too)
    kv = {"LLAMA_KV_HOST_MAPPED": "1"}
    lay = {"-np": "2", "-c": str(262144 + LANE_TOKENS)}
    g_mtp = int(os.environ.get("FLASHNEXT_KVMAP_GPU", "0") or 0)
    g_nomtp = int(os.environ.get("FLASHNEXT_KVMAP_GPU_NOMTP", "0") or 0)
    base_all = {k: v for k, v in table["flash-all"]["args"].items()}
    no_mtp = {k: None for k in MTP_ARGS if k != "-ts"}
    table["fit-kvmap"] = {"bin": fl, "env": {**table["flash-all"]["env"], **kv},
                          "args": {**base_all, **lay, "--n-cpu-moe": "48"}, "checks": ["probe"]}
    table["fit-kvmap-nomtp"] = {"bin": fl, "env": {**table["flash-all"]["env"], **kv},
                                "args": {**base_all, **lay, **no_mtp, "--n-cpu-moe": "48"}, "checks": ["probe"]}
    # f16 K/V: ggml-cuda's sparse flash attention (the MMA kernel, the only one with the sparse gather) needs f16 K and
    # V, and converts a q8_0 cache WHOLE on every call (fattn-common.cuh: to_fp16 over ggml_nelements(K)) -- over PCIe
    # when the cache is host-mapped (fit-kvmap, 2026-09-30: 4K decode 9.3 tok/s). An f16 cache is read only at the
    # cells the selection names.
    f16kv = {"--cache-type-k": "f16", "--cache-type-v": "f16"}
    table["fit-kvmap-f16"] = {"bin": fl, "env": {**table["flash-all"]["env"], **kv},
                              "args": {**base_all, **lay, **f16kv, "--n-cpu-moe": "48"}, "checks": ["probe"]}
    # the same with the caches on the card (f16 K/V is ~2x q8_0's VRAM, so every expert layer in RAM) and with q8_0 on
    # the card: the conversion's cost with and without PCIe in the way
    table["probe-f16-cpu"] = {"bin": fl, "env": {**table["flash-all"]["env"]},
                              "args": {**base_all, **lay, **f16kv, "--n-cpu-moe": "48"}, "checks": ["probe"]}
    table["probe-q8-cpu"] = {"bin": fl, "env": {**table["flash-all"]["env"]},
                             "args": {**base_all, **lay, "--n-cpu-moe": "48"}, "checks": ["probe"]}
    # M3 + M4 (0012-0014): the KV in host memory and the freed VRAM as the profile-seeded adaptive expert cache over
    # EVERY layer (Strata's recipe: --kv-resident + the expert cache). FLASHNEXT_KVCACHE_SLOTS slots a layer, derived
    # from fit-kvmap's lowest free MiB less the 1,000 MiB reserve the kvmap arms use, over 48 layers of 1.379 MiB
    # experts, less the layer's zero slot (the cache's own log at 63 slots: "4236.8 MiB device memory" = 64 rows x 48
    # layers x 1.379 MiB; cache-fit, 2026-09-30). 123 (from 1.32 MiB) left 30 MiB free: kvcache-fit, the same day
    kvslots = os.environ.get("FLASHNEXT_KVCACHE_SLOTS", "0") or "0"
    table["kvcache-fit"] = {"bin": fl, "env": {**table["flash-all"]["env"], **kv, **strata},
                            "args": {**base_all, **lay, "--n-cpu-moe": "48", "--moe-expert-cache": kvslots, **lv},
                            "checks": ["probe"]}
    table["kvcache-all"] = {"bin": fl, "env": {**table["flash-all"]["env"], **kv, **strata},
                            "args": {**base_all, **lay, "--n-cpu-moe": "48", "--moe-expert-cache": kvslots, **lv},
                            "checks": ["needles", "corrupt", "contract", "lane", "speed"]}
    # the same cache with the KV on the card: FLASHNEXT_CACHE_SLOTS from probe-q8-cpu's lowest free MiB, the same rule
    cslots = os.environ.get("FLASHNEXT_CACHE_SLOTS", "0") or "0"
    table["cache-fit"] = {"bin": fl, "env": {**table["flash-all"]["env"], **strata},
                          "args": {**base_all, **lay, "--n-cpu-moe": "48", "--moe-expert-cache": cslots, **lv},
                          "checks": ["probe"]}
    table["cache-all"] = {"bin": fl, "env": {**table["flash-all"]["env"], **strata},
                          "args": {**base_all, **lay, "--n-cpu-moe": "48", "--moe-expert-cache": cslots, **lv},
                          "checks": ["kl", "needles", "corrupt", "contract", "lane", "speed"],
                          # 0013 builds the cache chain only on batches <= MMVQ's (7 here): KL at 4 (an MTP verify
                          # window) so the cache serves the scored tokens
                          "kl_batch": 4}
    table["kvmap-all"] = {"bin": fl, "env": {**table["flash-all"]["env"], **kv},
                          "args": {**base_all, **lay, "--n-cpu-moe": str(N_LAYER - g_mtp)},
                          "checks": ["kl", "needles", "corrupt", "contract", "lane", "speed"]}
    table["kvmap-nomtp"] = {"bin": fl, "env": {**table["flash-all"]["env"], **kv},
                            "args": {**base_all, **lay, **no_mtp, "--n-cpu-moe": str(N_LAYER - g_nomtp)},
                            "checks": ["kl", "speed"]}
    table["kvmap-a4000"] = {"bin": fl, "env": {**table["flash-a4000"]["env"], **kv},
                            "args": {**table["flash-a4000"]["args"], **lay,
                                     **a4000_args(a4000_m or 1, g_mtp)},
                            "checks": ["kl", "corrupt", "lane", "speed"], "a4000": True}
    # THE OPERATOR'S LAYOUT (2026-09-29): one conversation slot + the jjava lane: -np 2, the window of one conversation
    # plus budget.LANE_TOKENS (3,072) in one unified pool. final-<x> is arm <x> in that layout, with the lane step.
    for src in ("flash-all", "flash-a4000", "flash-all-q2"):
        f = dict(table[src], args={**table[src]["args"], "-np": "2", "-c": str(262144 + LANE_TOKENS)})
        f["checks"] = ["needles", "corrupt", "contract", "lane", "speed"]
        table["final-" + src.split("-", 1)[1]] = f
    table["fault-cache"] = dict(table["flash-cache"], checks=["fault"])
    # 2026-09-30 diagnosis: cache-all answered 7/15 needles (nothing at 128K, 2/5 at 32K; flash-all 15/15). One switch
    # each: 0015 off (the vector kernel for a q8_0 decode), 0014 and 0015 off, and no expert cache
    table["diag-nosq"] = dict(table["cache-all"], env={**table["cache-all"]["env"], "GGML_CUDA_FA_SPARSE_QUANT": "0"},
                              checks=["needles"])
    table["diag-norows"] = dict(table["cache-all"], env={**table["cache-all"]["env"], "GGML_CUDA_FA_SPARSE_QUANT": "0",
                                                        "GGML_CUDA_FA_SPARSE_ROWS": "0"}, checks=["needles"])
    table["diag-nocache"] = dict(table["probe-q8-cpu"], checks=["needles"])
    # NEXT WINDOW (prepared offline, 2026-09-30): where cache-all's decode goes as the context grows (31-36 tok/s at
    # ~35K, 18-25 at ~142K in its needle runs), and the prompt-side switches already built
    table["probe-opt-cache-all"] = dict(table["cache-all"], env={**table["cache-all"]["env"], **opt}, checks=["probe"])
    table["cache-all-nomtp"] = dict(table["cache-all"], args={**table["cache-all"]["args"], **no_mtp}, checks=["probe"])
    table["cache-all-prefetch"] = dict(table["cache-all"], args={**table["cache-all"]["args"],
                                                                 "--prefetch-experts-slots": "3"}, checks=["probe"])
    # a 1K prompt batch halves the expert copies per prompt token (op offload copies every expert of a layer per
    # ubatch) and costs 1,898 MiB more compute buffer (3,770 vs 1,872 MiB: vram-cache48-ub1024.log / vram-cache48.log)
    # = 29 slot rows of 66.2 MiB (4,236.8 MiB / 64 rows): 63 - 29 = 34 slots a layer
    table["cache-all-ub1k"] = dict(table["cache-all"], args={**table["cache-all"]["args"], "-ub": "1024",
                                                             "--moe-expert-cache": os.environ.get(
                                                                 "FLASHNEXT_CACHE_SLOTS_UB1K", "34")},
                                   checks=["probe"])
    # THE OPERATOR, 2026-09-30: "no jjava slowing down this highly tuned masterpiece, it stays locked in once it is
    # swapped" -- Flash-Next at -np 1 (one conversation, no lane; jjava goes to a Bonsai on the A4000). The same arm
    # with one slot and the native window: its probe records the VRAM that frees for experts against -np 2
    np1 = {"-np": "1", "-c": "262144"}
    table["cache-fit-np1"] = dict(table["cache-all"], args={**table["cache-all"]["args"], **np1}, checks=["probe"])
    # -np 1's freed VRAM as slots: 1,289 - 767 MiB (cache-fit-np1 vs cache-fit) = 522 = 7 rows of 66.2 -> 70
    s_np1 = os.environ.get("FLASHNEXT_CACHE_SLOTS_NP1", "70")
    table["cache-all-np1"] = dict(table["cache-all"], args={**table["cache-all"]["args"], **np1,
                                                           "--moe-expert-cache": s_np1},
                                  # speed only: the cache is exact (a GPU chain added to the CPU sum) and needles/corrupt
                                  # passed on the same path at 63 slots (cache-all); only the slot count changes
                                  checks=["speed"])
    # 0017 on the 5060 Ti: where a token's time goes (waits on the GPU, CPU experts, copies), cache-all at -np 1
    table["probe-sched-np1"] = dict(table["cache-all-np1"], env={**table["cache-all-np1"]["env"],
                                                                 "GGML_SCHED_TIMING": "1"}, checks=["probe"])
    # q4_0 K/V (upstream rotates a quantized cache with a Walsh-Hadamard matrix: llama-kv-cache.cpp attn_rot_k/v):
    # 3,264 MiB of q8_0 at 262,144 cells (vram-cache48.log) x 0.5625/1.0625 bytes a value = 1,728 -> 1,536 MiB = 23
    # rows more -> 93. KL at batch 4 (the cache serves those batches), needles, then speed
    table["cache-all-np1-q4kv"] = dict(table["cache-all"], args={**table["cache-all"]["args"], **np1,
                                                                "--cache-type-k": "q4_0", "--cache-type-v": "q4_0",
                                                                "--moe-expert-cache": os.environ.get(
                                                                    "FLASHNEXT_CACHE_SLOTS_Q4KV", "93")},
                                       checks=["kl", "needles", "corrupt", "speed"], kl_batch=4)
    # M2b (0019, LLAMA_QSA_BLOCK_TOPK=1): the QSA budget chosen as whole blocks, no n_kv-sized op per QSA layer. It
    # changes which cells are attended (KLD ~0.009 vs the cell path on the A4000, the size of any one-block change),
    # so needles and corrupt on the model itself, then speed
    table["cache-all-np1-blk"] = dict(table["cache-all"], env={**table["cache-all"]["env"], "LLAMA_QSA_BLOCK_TOPK": "1"},
                                      args={**table["cache-all"]["args"], **np1,
                                            "--moe-expert-cache": s_np1},
                                      checks=["kl", "needles", "corrupt", "speed"], kl_batch=4)
    # a verify batch of one fixed size: MTP's drafts stop early under --spec-draft-p-min 0.5, so the target graph's
    # batch varies step to step, every GPU split's properties change and ggml-cuda re-warms its CUDA graphs instead
    # of replaying them (ggml_cuda_graph_update_required); D1 on the 5060 Ti put 15.6 ms of a ~128K verify step in
    # launches. p-min 0 always drafts the full n: the batch is constant and the graphs can replay
    table["cache-all-np1-pmin0"] = dict(table["cache-all"], env={**table["cache-all"]["env"], "GGML_SCHED_TIMING": "1"},
                                        args={**table["cache-all"]["args"], **np1, "--moe-expert-cache": s_np1,
                                              "--spec-draft-p-min": "0"},
                                        checks=["probe"])
    # 0020, LLAMA_GRAPH_CACHE=8: one graph per verify-batch size, so ggml-cuda's CUDA graphs replay instead of warming
    # up on every MTP step (A4000: launches 15.5-18 -> 2.5-3.9 ms a verify step, greedy output identical). Exact:
    # speed, and the probe's sched timing beside it
    table["cache-all-np1-gcache"] = dict(table["cache-all"], env={**table["cache-all"]["env"], "LLAMA_GRAPH_CACHE": "8"},
                                         args={**table["cache-all"]["args"], **np1, "--moe-expert-cache": s_np1},
                                         checks=["speed"])
    # E2 (0021): the QSA layers' K/V in host-mapped memory (LLAMA_KV_HOST_MAPPED=1 maps only the sparse layers; the
    # MTP draft's dense layer stays on the card) and the VRAM it frees as slots: ~3.2 GB = ~48 rows of 66.2 MiB on top
    # of 70 -> 118 (FLASHNEXT_CACHE_SLOTS_KV; the fit probe's lowest free MiB decides). No graph cache: window E1 was
    # inconclusive and 0020 stays pinned and off (coordinator, 2026-10-01: "no graph cache")
    kvenv = {"LLAMA_KV_HOST_MAPPED": "1"}
    s_kv = os.environ.get("FLASHNEXT_CACHE_SLOTS_KV", "118")
    table["kvcache-fit-np1"] = dict(table["cache-all"], env={**table["cache-all"]["env"], **kvenv},
                                    args={**table["cache-all"]["args"], **np1, "--moe-expert-cache": s_kv},
                                    checks=["probe"])
    table["cache-all-np1-kv"] = dict(table["kvcache-fit-np1"], checks=["kl", "needles", "corrupt", "speed"], kl_batch=4)
    # E2, both: 0021's host-mapped sparse K/V at the larger cache AND 0019's block top-k
    table["cache-all-np1-kv-blk"] = dict(table["cache-all-np1-kv"],
                                         env={**table["cache-all-np1-kv"]["env"], "LLAMA_QSA_BLOCK_TOPK": "1"})
    # THE PROFILES (the coordinator, 2026-10-01). Decode: 0010's per-op GPU time on the deployed arm (CUDA graphs off
    # for it; they do not replay under MTP anyway, window D). Prefill (step `prefill`): the plain rate, 0017's
    # scheduler split, 0010's per-op table, and the two prompt-side switches already built -- no op offload (the
    # CPU computes the prompt's experts) and 0002's streaming ring (two staging slots of one expert tensor each,
    # 2 x 256.25 MiB = 512.5 MiB = 8 cache rows of 66.2 MiB fewer: 70 - 8 = 62; FLASHNEXT_CACHE_SLOTS_RING)
    table["probe-opt-np1"] = dict(table["cache-all-np1"], env={**table["cache-all-np1"]["env"], **opt},
                                  checks=["probe"])
    # 0023 (--checkpoint-every; step `ckpt`): Strata's 16,384 (generate.cpp:283 prompt_cache_every), and the same
    # arm without the flag (today's behaviour)
    table["ckpt-np1"] = dict(table["cache-all-np1"],
                             args={**table["cache-all-np1"]["args"], "--checkpoint-every": "16384"}, checks=["ckpt"])
    table["ckpt-off-np1"] = dict(table["cache-all-np1"], checks=["ckpt"])
    # the same per-op table with 0021's host-mapped sparse K/V (118 slots): the attention ops' time against
    # probe-opt-np1's is the per-step PCIe read of the selected cells -- what 0026's page window has to remove
    table["probe-opt-np1-kv"] = dict(table["kvcache-fit-np1"], env={**table["kvcache-fit-np1"]["env"], **opt},
                                     checks=["probe"])
    # THE PORTS (item 5, 2026-10-02; the integration build int0028 = 0001-0028, every port off unless its switch is
    # set; FLASHNEXT_FLASH_BIN points the arms at it). int-off: the build with nothing switched on, against
    # cache-all-np1's numbers on the shipped build.
    full = ["kl", "needles", "corrupt", "speed"]
    table["int-off-np1"] = dict(table["cache-all-np1"], checks=full, kl_batch=4)
    # 0025: refills on a second stream, published when their event completes
    table["refill-np1"] = dict(table["cache-all-np1"],
                               env={**table["cache-all-np1"]["env"], "LLAMA_MOE_CACHE_QUEUED_REFILL": "1"},
                               checks=full, kl_batch=4)
    # 0026 on 0021: the page window (1 = Strata's 32,768 cells per K or V tensor; the patch states 427 MB of VRAM
    # at q8_0 = 7 rows of 66.2 MiB: 118 - 7 = 111; FLASHNEXT_CACHE_SLOTS_KVPAGE, the fit probe decides)
    kvp = {**table["kvcache-fit-np1"]["env"], "LLAMA_KV_PAGE_WINDOW": "1", "GGML_CUDA_KV_PAGE_STATS": "1"}
    s_kvp = os.environ.get("FLASHNEXT_CACHE_SLOTS_KVPAGE", "111")
    table["kvpage-fit-np1"] = dict(table["kvcache-fit-np1"], env=kvp,
                                   args={**table["kvcache-fit-np1"]["args"], "--moe-expert-cache": s_kvp},
                                   checks=["probe"])
    table["cache-all-np1-kvpage"] = dict(table["kvpage-fit-np1"], checks=full, kl_batch=4)
    # 0027 / 0028, the prompt side: prefill rate, and needles + corrupt (a 128K prompt read by the new kernel)
    for tag, env in (("qsa1", {"GGML_CUDA_QSA_PROMPT_ATTN": "1"}), ("qsa2", {"GGML_CUDA_QSA_PROMPT_ATTN": "2"}),
                     ("ple", {"LLAMA_PLE_PREFETCH": "1"})):
        table[f"prefill-{tag}-np1"] = dict(table["cache-all-np1"], env={**table["cache-all-np1"]["env"], **env},
                                          checks=["prefill"])
        table[f"{tag}-np1"] = dict(table["cache-all-np1"], env={**table["cache-all-np1"]["env"], **env},
                                   checks=["needles", "corrupt"])
    table["prefill-int-off-np1"] = dict(table["cache-all-np1"], checks=["prefill"])
    # the experts NOT pinned (LLAMA_PIN_EXPERTS=0: 36.4 GB less commit and locked RAM, docs/FLASH-NEXT.md 11.1), with
    # 0023's periodic checkpoints as deployed: needles and corrupt prove the output does not depend on the pin
    table["cache-all-np1-nopin"] = dict(table["cache-all-np1"],
                                        env={**table["cache-all-np1"]["env"], "LLAMA_PIN_EXPERTS": "0"},
                                        args={**table["cache-all-np1"]["args"], "--checkpoint-every": "16384"},
                                        checks=["needles", "corrupt", "speed"])
    # PREFILL FIRST (the coordinator, 2026-10-02: the first prompt of a conversation, 24.6K tokens in 155-178 s, is the
    # operator's main wait). `dep` is the deployed setting: unpinned, --checkpoint-every 16384, 70 slots, -np 1.
    dep = dict(table["cache-all-np1-nopin"], checks=["prefill"])
    table["pf-dep"] = dep
    table["pf-int-off"] = dep          # the same arm under another name: the integration build with every port off
    table["pf-dep-sched"] = dict(dep, env={**dep["env"], "GGML_SCHED_TIMING": "1"})
    table["pf-dep-opt"] = dict(dep, env={**dep["env"], **opt})
    table["pf-dep-nooffload"] = dict(dep, args={**dep["args"], "--no-op-offload": ""})
    # -ub 1024: half the expert uploads per token; 1,898 MiB more compute buffer = 29 rows of 66.2 MiB: 70 - 29 = 41
    # (FLASHNEXT_CACHE_SLOTS_UB1K_NP1; the arm's lowest free VRAM says whether it fits). `speed` = what that does to
    # the decode
    table["pf-ub1k"] = dict(dep, args={**dep["args"], "-ub": "1024", "--moe-expert-cache":
                                       os.environ.get("FLASHNEXT_CACHE_SLOTS_UB1K_NP1", "41")},
                            checks=["prefill", "speed", "needles", "corrupt"])
    # 0002's ring: 2 staging slots of one expert tensor each (512.5 MiB = 8 rows: 62)
    table["pf-ring"] = dict(dep, args={**dep["args"], "--prefetch-experts-slots": "2", "--moe-expert-cache":
                                       os.environ.get("FLASHNEXT_CACHE_SLOTS_RING", "62")})
    table["pf-qsa1"] = dict(dep, env={**dep["env"], "GGML_CUDA_QSA_PROMPT_ATTN": "1"},
                            checks=["prefill", "needles", "corrupt"])
    table["pf-qsa2"] = dict(dep, env={**dep["env"], "GGML_CUDA_QSA_PROMPT_ATTN": "2"},
                            checks=["prefill", "needles", "corrupt"])
    table["pf-ple"] = dict(dep, env={**dep["env"], "LLAMA_PLE_PREFETCH": "1"}, checks=["prefill", "needles", "corrupt"])
    # the best combination: FLASHNEXT_PF_BEST_ENV (k=v,k=v) and FLASHNEXT_PF_BEST_ARGS (flag=value,flag=value)
    best_env = dict(kv.split("=", 1) for kv in (os.environ.get("FLASHNEXT_PF_BEST_ENV") or "").split(",") if kv)
    best_args = dict(kv.split("=", 1) for kv in (os.environ.get("FLASHNEXT_PF_BEST_ARGS") or "").split(",") if kv)
    table["pf-best"] = dict(dep, env={**dep["env"], **best_env}, args={**dep["args"], **best_args},
                            checks=["prefill", "speed", "needles", "corrupt", "kl"], kl_batch=4)
    # 0029, LAYER-MAJOR PREFILL (LLAMA_LAYER_MAJOR=1; build int0029 = 0001-0029). The expert slots are device memory:
    # LLAMA_LAYER_MAJOR_SLOTS x the largest expert tensor (256.25 MiB, docs/FLASH-NEXT.md 12.1): 6 = 1,537.5 MiB = 24
    # rows of 66.2 MiB (70 - 24 = 46), 3 = 768.75 MiB = 12 rows (58); -b 8192 adds the residual buffers (2 x 40 KiB a
    # token: 640 MiB at 8,192 against 160 at 2,048: 480 MiB = 8 rows more). FLASHNEXT_LM_ROWS_<tag> overrides each;
    # the arm's lowest free VRAM says whether it fits. DEBUG=1: one line per batch (ms, bytes uploaded, hits)
    lm = {**dep["env"], "LLAMA_LAYER_MAJOR": "1", "LLAMA_LAYER_MAJOR_DEBUG": "1"}
    for tag, slots, b, rows in (("lm6", "6", "2048", "46"), ("lm3", "3", "2048", "58"),
                                ("lm6-b8k", "6", "8192", "38"), ("lm3-b8k", "3", "8192", "50")):
        rows = os.environ.get("FLASHNEXT_LM_ROWS_" + tag.replace("-", "_").upper(), rows)
        table[f"pf-{tag}"] = dict(dep, env={**lm, "LLAMA_LAYER_MAJOR_SLOTS": slots},
                                  args={**dep["args"], "-b": b, "--moe-expert-cache": rows},
                                  checks=["prefill", "speed", "needles", "corrupt", "kl", "ckpt", "cancel"],
                                  kl_batch=int(b), kl_ub=512)
    # the ship candidate cand0029 ITSELF at the gated setting (the same arm under its own name)
    table["pf-lm3-b8k-c29"] = table["pf-lm3-b8k"]
    # 0030, THE STAGER (LLAMA_STAGER=1; build cand0030 = 0001-0018, 0022, 0023, 0029, 0030; FLASHNEXT_FLASH_BIN points at it).
    # The sweep's start (bench/results/fn_probe/20261006-024716): 3 memcpy threads, 4 MiB chunks, a ring of 16 (64 MiB pinned).
    # st0 = the switch OFF (the control: must read like pf-lm3-b8k), st1 = on; lm3 / lm6 = 3 / 6 slots (rows 50 / 38 of
    # 66.2 MiB: 6 slots is +769 MiB = 12 rows); st1p = on with the experts PINNED (LLAMA_PIN_EXPERTS=1: the pinned ceiling,
    # the stager takes no pinned weight). `-b8k` = prefill only (n=3, 8K and 32K prompts); `-full` = the rest of the gate.
    st_on = {"LLAMA_STAGER": "1", "LLAMA_STAGER_THREADS": "3", "LLAMA_STAGER_CHUNK_MIB": "4", "LLAMA_STAGER_RING": "16"}
    for tag, stg, slots, rows, pin in (("st0-lm3", False, "3", "50", False), ("st1-lm3", True, "3", "50", False),
                                        ("st0-lm6", False, "6", "38", False), ("st1-lm6", True, "6", "38", False),
                                        ("st1p-lm6", True, "6", "38", True)):
        env = {**lm, "LLAMA_LAYER_MAJOR_SLOTS": slots, "LLAMA_STAGER": "0", **(st_on if stg else {})}
        if pin:
            env["LLAMA_PIN_EXPERTS"] = "1"
        a8 = {**dep["args"], "-b": "8192", "--moe-expert-cache": rows}
        table[f"pf-{tag}-b8k"] = dict(dep, env=env, args=a8, checks=["prefill"], kl_batch=8192, kl_ub=512)
        table[f"pf-{tag}-b8k-full"] = dict(dep, env=env, args=a8, kl_batch=8192, kl_ub=512,
                                           checks=["speed", "needles", "corrupt", "kl", "ckpt", "cancel"])
        table[f"pf-{tag}-b8k-cold"] = dict(dep, env=env, args=a8, checks=["cold"])
    # the switch unset on the same build (the equivalence against cand0023's pf-dep)
    table["pf-lm-off"] = dict(dep, checks=["prefill", "cancel"])
    # the prompt with 0021's mapped K/V, and with a 1,024-token ubatch (half the expert uploads per token; 1,898 MiB
    # more compute buffer = 29 cache rows of 66.2 MiB: 70 - 29 = 41 with the K/V on the card, 118 - 29 = 89 mapped)
    table["prefill-kv-np1"] = dict(table["kvcache-fit-np1"], checks=["prefill"])
    table["prefill-ub1k-np1"] = dict(table["cache-all-np1"],
                                     args={**table["cache-all-np1"]["args"], "-ub": "1024", "--moe-expert-cache":
                                           os.environ.get("FLASHNEXT_CACHE_SLOTS_UB1K_NP1", "41")}, checks=["prefill"])
    table["prefill-kv-ub1k-np1"] = dict(table["kvcache-fit-np1"],
                                        args={**table["kvcache-fit-np1"]["args"], "-ub": "1024", "--moe-expert-cache":
                                              os.environ.get("FLASHNEXT_CACHE_SLOTS_KV_UB1K", "89")}, checks=["prefill"])
    table["prefill-np1"] = dict(table["cache-all-np1"], checks=["prefill"])
    table["prefill-sched-np1"] = dict(table["cache-all-np1"],
                                      env={**table["cache-all-np1"]["env"], "GGML_SCHED_TIMING": "1"},
                                      checks=["prefill"])
    table["prefill-opt-np1"] = dict(table["cache-all-np1"], env={**table["cache-all-np1"]["env"], **opt},
                                    checks=["prefill"])
    table["prefill-nooffload-np1"] = dict(table["cache-all-np1"],
                                          args={**table["cache-all-np1"]["args"], "--no-op-offload": ""},
                                          checks=["prefill"])
    table["prefill-ring-np1"] = dict(table["cache-all-np1"],
                                     args={**table["cache-all-np1"]["args"], "--prefetch-experts-slots": "2",
                                           "--moe-expert-cache": os.environ.get("FLASHNEXT_CACHE_SLOTS_RING", "62")},
                                     checks=["prefill"])
    # M2b on the E2 base (the graph cache on, K/V on the card)
    table["cache-all-np1-blk-gc"] = dict(table["cache-all-np1-blk"],
                                         env={**table["cache-all-np1-blk"]["env"], "LLAMA_GRAPH_CACHE": "8"})
    # the greedy-identity check: the exact step (slot-state hashes and the greedy text at ~8K and ~32K) on the two arms
    table["exact-np1"] = dict(table["cache-all-np1"], checks=["exact"])
    table["exact-np1-gcache"] = dict(table["cache-all-np1-gcache"], checks=["exact"])
    # the same arm again, for the card's own run-to-run variation
    table["exact-np1-rerun"] = dict(table["cache-all-np1"], checks=["exact"])
    table["probe-sched-np1-gcache"] = dict(table["cache-all-np1-gcache"],
                                           env={**table["cache-all-np1-gcache"]["env"], "GGML_SCHED_TIMING": "1"},
                                           checks=["probe"])
    # a longer MTP draft: the verify window amortises the per-layer GPU latency (GGML_SCHED_TIMING, 2026-09-30)
    table["cache-all-np1-draft5"] = dict(table["cache-all"], args={**table["cache-all"]["args"], **np1,
                                                                  "--moe-expert-cache": s_np1,
                                                                  "--spec-draft-n-max": "5"},
                                         checks=["speed"])
    # the base arm's ~35K row again (its first was n=1, EOS-cut): run with FLASHNEXT_SPEED_CONTEXTS="32"
    table["base-35k"] = dict(table["base"], checks=["speed"])
    # jjava on a winner: --arms jjava-<arm> runs the arm as configured with only the jjava step
    for src in ("kvmap-all", "kvmap-nomtp", "kvmap-a4000", "flash-all", "final-all", "cache-all", "kvcache-all"):
        table["jjava-" + src] = dict(table[src], checks=["jjava"])
    return table


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
# 2026-09-29, the coordinator (layout v2, the operator's "Vision can go to second card and swap in and out"):
# Flash-Next carries no projector on the 5060 Ti; images at max go to bonsai-vision on the A4000
NO_MMPROJ = {"--mmproj": None}
MMPROJ_MIB = 1865
LANE_TOKENS = 3072         # mcp/budget.py LANE_TOKENS (layout v2): the decider lane's cells            # the projector (865 MiB file) + its ~1 GB compute buffer (docs/FLASH-NEXT.md section 4)


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
        # 0014: the sparse (top-k) flash attention over q8_0 and f16 caches, the QSA shape (its eval cases: top-k 2,048
        # and the model's own 2,051), with the f16 buffers poisoned so a read of an unconverted row is a NaN, not a pass
        # (the first 0014 passed these unpoisoned and lost needles at 32K/128K, 2026-09-30)
        key = f"{dev}/FLASH_ATTN_EXT/n_kv_max=20xx+poison"
        r = subprocess.run([exe, "test", "-o", "FLASH_ATTN_EXT", "-b", dev, "-p", "n_kv_max=20"],
                           capture_output=True, text=True, timeout=3600,
                           env={**os.environ, "GGML_CUDA_FA_SPARSE_ROWS_POISON": "1"})
        m = re.search(r"(\d+)/(\d+) tests passed", r.stdout)
        rec[key] = {"passed": int(m.group(1)), "total": int(m.group(2))} if m else {"error": r.stdout[-400:]}
        ec.log(f"kernels {key}: {rec[key]}")
    rec["PASS"] = all(v.get("passed") == v.get("total") and v.get("total") for v in rec.values() if isinstance(v, dict))
    json.dump(rec, open(os.path.join(out, "kernels.json"), "w"), indent=1)
    return rec


def commit_free_mib() -> int | None:
    """The host's commit headroom (Windows: the commit limit less what is committed), MiB. With flash-next loaded the
    box had ~1.4 GB of it (2026-10-01); a server's host buffers, checkpoints and prompt cache all come out of it."""
    try:
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MS()
        m.dwLength = ctypes.sizeof(MS)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            return None
        return int(m.ullAvailPageFile // (1024 * 1024))
    except Exception:                                                    # noqa: BLE001
        return None


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
        self.commit_loaded = commit_free_mib()

    def stop(self) -> dict:
        commit_end = commit_free_mib()
        self.guard.stop()
        self.proc.terminate()
        try:
            self.proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.logf.close()
        return {"gpu_min_free": self.guard.min_free, "gpu_max_used": self.guard.max_used, "guard": self.guard.tripped,
                "commit_free_mib": {"loaded": self.commit_loaded, "end": commit_end}}


def completion(base: str, prompt: str, n: int, slot: int = 0, ignore_eos: bool = False, fresh: bool = False) -> dict:
    # fresh: nothing of the slot's cache is reused (the prefill step: every prompt is processed whole)
    body = {"prompt": prompt, "id_slot": slot, "n_predict": n, "cache_prompt": not fresh, "temperature": 0.0,
            "top_k": 1, "seed": 0}
    if ignore_eos:
        # the speed step measures decoding, not where the model would stop: a raw slice of source code sometimes
        # greedily ends at once (2026-09-29, base arm: 3 of 15 runs decoded 1 token, 2 decoded 11), which left
        # rows with 0.0 tok/s or a rate over 11 tokens. Every speed run now decodes exactly n tokens.
        body["ignore_eos"] = True
    st, txt = ec.http("POST", base + "/completion", body, timeout=7200)
    if st != 200:
        raise RuntimeError(f"/completion: HTTP {st}: {txt[:300]}")
    r = json.loads(txt)
    t = r.get("timings") or {}
    # draft_n / draft_n_accepted: llama-server's speculative counts (MTP acceptance; absent without a draft)
    return {"content": r.get("content") or "", "prompt_n": t.get("prompt_n"), "prompt_tps": t.get("prompt_per_second"),
            "predicted_n": t.get("predicted_n"), "tps": t.get("predicted_per_second"),
            "draft_n": t.get("draft_n"), "draft_n_accepted": t.get("draft_n_accepted"),
            "stop_type": r.get("stop_type"), "n_ctx_used": (t.get("prompt_n") or 0) + (t.get("cache_n") or 0)}


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
    # an arm's kl_ub: the ubatch, when the batch must hold several (0029's layer-major path needs >= 2 ubatches)
    cmd = [exe, "-m", SHARD1, "-f", textf, "-c", "4096", "--chunks", "8", "-b", str(batch),
           "-ub", str(arm.get("kl_ub") or batch),
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
        elif k in ("--cache-type-k", "--cache-type-v") and v == "q4_0":
            # the KV type is what a q4kv arm tests; every earlier KL (and the base logits) ran llama-perplexity's
            # default f16 cache, so only a q4_0 arm passes its type through
            cmd += [k, v]
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
        # FLASHNEXT_NEEDLE_DEPTHS (e.g. "32768"): a diagnosis at one depth; PASS then means every needle asked
        depths = [int(x) for x in (os.environ.get("FLASHNEXT_NEEDLE_DEPTHS") or "1024 32768 131072").split()]
        for depth in depths:
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
        rec["PASS"] = ok == 5 * len(depths)
        rec["depths"] = depths
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
        # every slot the arm asks for (-np; the operator's layout is 2: the conversation and the jjava lane -- this
        # read ">= 3" until 2026-09-30, the four-slot layout's check, and failed every -np 2 arm)
        n_slots = int(arm["args"].get("-np") or 4)
        rec["slots"] = st == 200 and isinstance(json.loads(txt), list) and len(json.loads(txt)) >= n_slots
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
        # a hybrid (recurrent) model resumes at its rollback snapshot: the last 1 + n_rs_seq tokens are read again,
        # n_rs_seq = the MTP draft's --spec-draft-n-max (the snapshots a verify window may roll back; llama-memory-
        # hybrid-idx TAG_RECURRENT_ROLLBACK_SPLITS), 0 without MTP; one more for the repeat's own last token
        n_rs = int(arm["args"].get("--spec-draft-n-max") or 0) if "--spec-type" in arm["args"] else 0
        rec["slot_cache_reuse"] = {"first_prompt_n": first.get("prompt_n"), "repeat_cache_n": cache_n,
                                   "allowed_reread": n_rs + 2,
                                   "ok": bool(cache_n) and cache_n >= (first.get("prompt_n") or 1) - (n_rs + 2)}
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


def step_lane(name: str, arm: dict, port: int, out: str) -> dict:
    """The operator's layout (2026-09-29: "we should not have a second conversation at all ... the jjava engine should
    be the only other thing we need ready to go"): -np 2, slot 0 the conversation, slot 1 the decider lane. Slot 0's
    decode tok/s at ~4K / ~35K / ~68K, n=3 each, with the lane IDLE (nothing on slot 1) and ACTIVE (a thread sending
    decider-shaped reads to slot 1 the whole time slot 0 decodes: a ~1.5K-token state, n_predict 1, top_logprobs 5,
    the prompt cached -- jjava's shape). Decides whether jjava stays on the main card."""
    import threading
    srv = Launched(name, arm, port, out)
    rec: dict = {"load_s": srv.load_s}
    text = corpus()
    lane_state = text[-6000:]
    try:
        for mode in ("idle", "active"):
            stop = threading.Event()
            reads = {"n": 0, "errors": 0, "ms": []}

            def lane():
                q = 0
                while not stop.is_set():
                    body = {"prompt": lane_state + f"\n\nQuestion {q % 7}: which option fits? Answer with one letter:",
                            "id_slot": 1, "n_predict": 1, "cache_prompt": True, "temperature": 0.0, "n_probs": 5}
                    t0 = time.time()
                    st, _ = ec.http("POST", srv.base + "/completion", body, timeout=600)
                    reads["ms"].append(round((time.time() - t0) * 1000))
                    reads["n"] += 1
                    reads["errors"] += st != 200
                    q += 1
            for k in (4, 32, 64):
                runs = []
                for rep in range(3):
                    prompt = text[rep * 4096: rep * 4096 + k * 1024 * 4]
                    completion(srv.base, prompt, 0, slot=0)           # read the prompt first, lane quiet
                    t = threading.Thread(target=lane, daemon=True) if mode == "active" else None
                    if t:
                        stop.clear()
                        t.start()
                    r = completion(srv.base, prompt, 256, slot=0, ignore_eos=True)
                    if t:
                        stop.set()
                        t.join(timeout=600)
                    runs.append({kk: r[kk] for kk in ("predicted_n", "tps", "stop_type", "n_ctx_used", "draft_n",
                                                      "draft_n_accepted")})
                rec[f"{mode}_{k}k"] = runs
            if mode == "active":
                ms = sorted(reads["ms"]) or [0]
                rec["lane_reads"] = {"n": reads["n"], "errors": reads["errors"], "ms_median": ms[len(ms) // 2],
                                     "ms_p90": ms[int(len(ms) * 0.9)]}
    except Exception as e:                                               # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
    finally:
        rec.update(srv.stop())
    return rec


OPT_RE = re.compile(r"op timing over (\d+) graphs: ([0-9.]+) ms per graph")
OPT_ROW = re.compile(r"^\s*([0-9.]+) ms/graph\s+([0-9.]+)%\s+x(\d+)\s+(.*)$")


def step_probe(name: str, arm: dict, port: int, out: str) -> dict:
    """Window B's step 1 (the coordinator, 2026-09-29): a decode at ~4K and ~35K (256 tokens each, ignore_eos) under
    the arm's diagnostic env -- GGML_MOE_LOG (routing traces for bench/moe_hit_sim.py) or GGML_CUDA_OP_TIMING
    (0010's per-op GPU time; needs GGML_CUDA_DISABLE_GRAPHS=1). Records the decode timings and, for op timing, the
    last printed top-30 table per context from the server log."""
    srv = Launched(name, arm, port, out)
    rec: dict = {"load_s": srv.load_s, "env": {k: v for k, v in arm["env"].items() if k.startswith("GGML_")}}
    logp = os.path.join(out, f"{name}.server.log")
    try:
        text = corpus()
        # FLASHNEXT_PROBE_CONTEXTS (e.g. "4 32 128"): the contexts probed, in K tokens (default 4 and 32)
        for k in [int(x) for x in (os.environ.get("FLASHNEXT_PROBE_CONTEXTS") or "4 32").split()]:
            mark = os.path.getsize(logp)
            c0 = cache_counts(logp)
            r = completion(srv.base, text[: k * 1024 * 4], int(os.environ.get("FLASHNEXT_PROBE_N") or 256), slot=0,
                           ignore_eos=True)  # FLASHNEXT_PROBE_N: tokens decoded (768 gives 0017 a decode-only window)
            row = {kk: r[kk] for kk in ("prompt_n", "prompt_tps", "predicted_n", "tps", "stop_type", "n_ctx_used",
                                        "draft_n", "draft_n_accepted")}
            with open(logp, encoding="utf-8", errors="replace") as f:
                f.seek(mark)
                tail = f.read().splitlines()
            tables = [i for i, ln in enumerate(tail) if OPT_RE.search(ln)]
            if tables:
                i = tables[-1]
                m = OPT_RE.search(tail[i])
                rows = []
                for ln in tail[i + 1: i + 31]:
                    mm = OPT_ROW.match(ln.split(" I ", 1)[-1] if " I " in ln else ln)
                    if mm:
                        rows.append({"ms_per_graph": float(mm.group(1)), "pct": float(mm.group(2)),
                                     "n": int(mm.group(3)), "op": mm.group(4)})
                row["op_timing"] = {"graphs": int(m.group(1)), "ms_per_graph": float(m.group(2)), "top": rows}
            hr = cache_rate(c0, cache_counts(logp))
            if hr:
                row["cache"] = hr
            sched = [ln.split(" I ", 1)[-1].strip() for ln in tail if "sched timing" in ln]
            if sched:
                row["sched_timing"] = sched[-3:]
            rec[f"{k}k"] = row
    except Exception as e:                                               # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
    finally:
        rec.update(srv.stop())
    return rec


SANITIZER = ("C:/Users/jwals/engines/tools/cuda-sanitizer-api-12.8.93/pkg/Library/compute-sanitizer/"
             "compute-sanitizer.exe")      # engines/manifest.yaml toolchains msvc-cuda128 sanitizer


def step_fault(name: str, arm: dict, out: str) -> dict:
    """M0, ONE bounded attempt (the coordinator, 2026-09-29) at the expert cache's multi-token fault: the KL step's
    command (llama-perplexity at -b/-ub 16, the batch that faulted) for ONE chunk, first with CUDA_LAUNCH_BLOCKING=1
    (the faulting kernel named at its launch), then under compute-sanitizer memcheck (the first bad access)."""
    exe = os.path.join(arm["bin"], "llama-perplexity.exe")
    textf = os.path.join(out, "kl-text.txt")
    cmd = [exe, "-m", SHARD1, "-f", textf, "-c", "4096", "--chunks", "1", "-b", "16", "-ub", "16",
           "-dev", "CUDA0", "-ngl", "999", "-lm", "mmap", "--lazy-mode", "on"]
    for k, v in arm["args"].items():
        if k in ("--moe-expert-cache", "--n-cpu-moe", "-t", "-C", "--cpu-strict"):
            cmd += [k] + ([v] if v != "" else [])
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=ec.CARD_UUID, **arm["env"])
    rec: dict = {}
    for tag, pre, extra, tmo in (("blocking", [], {"CUDA_LAUNCH_BLOCKING": "1"}, 1800),
                                 ("memcheck", [SANITIZER, "--tool", "memcheck", "--print-limit", "20",
                                               "--show-backtrace", "device"], {}, 3600)):
        logf = os.path.join(out, f"fault-{tag}.log")
        t0 = time.time()
        try:
            with open(logf, "w", encoding="utf-8") as f:
                p = subprocess.run(pre + cmd, env={**env, **extra}, stdout=f, stderr=subprocess.STDOUT, timeout=tmo)
            code = p.returncode
        except subprocess.TimeoutExpired:
            code = "timeout"
        tail = open(logf, encoding="utf-8", errors="replace").read().splitlines()
        rec[tag] = {"exit": code, "seconds": round(time.time() - t0), "log": logf,
                    "errors": [ln for ln in tail if re.search(r"error|Invalid|illegal|out of bounds|at 0x", ln,
                                                              re.I)][:30]}
        ec.log(f"fault {tag}: exit {code}")
    return rec


def step_jjava(name: str, arm: dict, port: int, out: str) -> dict:
    """jjava on this engine (the operator, 2026-09-29: "every model should have the jjava unlock"): the decider's
    per-model measurement (bench/decider/measure_model.py: labels, letter priors, label bias, tie band,
    legacy_vs_typed) and JevBench's 231 public items (bench/decider/jevbench_run.py), against this arm's own server
    by base URL (never llama-swap's /upstream). Records each script's exit code and its log; the records land where
    the runtime reads them (bench/decider/results/models/flash-next.json and the jevbench run dir)."""
    srv = Launched(name, arm, port, out)
    rec: dict = {"load_s": srv.load_s}
    py = sys.executable
    try:
        for tag, cmd, tmo in (
                ("measure_model", [py, os.path.join(ROOT, "bench", "decider", "measure_model.py"), "--run",
                                   "--model", "flash-next", "--base-url", srv.base], 4 * 3600),
                ("jevbench", [py, os.path.join(ROOT, "bench", "decider", "jevbench_run.py"), "--run",
                              "--model", "flash-next", "--base-url", srv.base], 4 * 3600)):
            logf = os.path.join(out, f"jjava-{tag}.log")
            t0 = time.time()
            with open(logf, "w", encoding="utf-8") as f:
                p = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, timeout=tmo)
            rec[tag] = {"exit": p.returncode, "seconds": round(time.time() - t0), "log": logf}
            ec.log(f"jjava {tag}: exit {p.returncode} in {rec[tag]['seconds']} s")
    except Exception as e:                                               # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
    finally:
        rec.update(srv.stop())
    return rec


CACHE_RE = re.compile(r"moe-cache: steps=(\d+) hits=(\d+) misses=(\d+)")


SCHED_RE = re.compile(r"sched timing (\S+) over (\d+) graphs: wall ([0-9.]+) ms = wait ([0-9.]+) \(([0-9.]+) waits\) "
                      r"\+ cpu ([0-9.]+) \+ launch ([0-9.]+) \+ copy ([0-9.]+) \+ other ([0-9.]+) ms per graph; "
                      r"([0-9.]+) splits per graph")


def timing_tables(lines: list[str]) -> dict:
    """Every 0010 op-timing table and 0017 sched-timing line in a stretch of a server log, summed: the op tables as
    total ms per op (a table is ms per graph over its graphs; only its top 30 rows are printed), the sched lines per
    scheduler (the target model and the MTP draft have their own) as total ms by part. Both print every 100 graphs,
    so the last <100 graphs of a stretch are not in it: `graphs` says how many are."""
    ops: dict[str, float] = {}
    op_graphs, op_ms = 0, 0.0
    for i, ln in enumerate(lines):
        m = OPT_RE.search(ln)
        if not m:
            continue
        g = int(m.group(1))
        op_graphs += g
        op_ms += float(m.group(2)) * g
        for row in lines[i + 1: i + 31]:
            mm = OPT_ROW.match(row.split(" I ", 1)[-1] if " I " in row else row)
            if not mm:
                break
            ops[mm.group(4).strip()] = ops.get(mm.group(4).strip(), 0.0) + float(mm.group(1)) * g
    sched: dict[str, dict] = {}
    for ln in lines:
        m = SCHED_RE.search(ln)
        if not m:
            continue
        g = int(m.group(2))
        s = sched.setdefault(m.group(1), {"graphs": 0, "wall": 0.0, "wait": 0.0, "waits": 0.0, "cpu": 0.0,
                                          "launch": 0.0, "copy": 0.0, "other": 0.0, "splits": 0.0})
        s["graphs"] += g
        for k, j in (("wall", 3), ("wait", 4), ("waits", 5), ("cpu", 6), ("launch", 7), ("copy", 8), ("other", 9),
                     ("splits", 10)):
            s[k] += float(m.group(j)) * g
    out: dict = {}
    if op_graphs:
        top = sorted(ops.items(), key=lambda kv: -kv[1])[:20]
        out["op_timing"] = {"graphs": op_graphs, "ms": round(op_ms, 1),
                            "top": [{"op": k, "ms": round(v, 1), "pct": round(100 * v / op_ms, 1)} for k, v in top]}
    if sched:
        out["sched_timing"] = {k: {kk: (round(vv, 1) if kk != "graphs" else vv) for kk, vv in v.items()}
                               for k, v in sched.items()}
    return out


def step_prefill(name: str, arm: dict, port: int, out: str) -> dict:
    """Where a prompt token's time goes (the coordinator, 2026-10-01: the operator's 26-30K prompts ran at 94-120
    tok/s cold). Per context (FLASHNEXT_PREFILL_CONTEXTS, K tokens, default "8 32"): a fresh server, one ~1K warm-up
    request, then FLASHNEXT_PREFILL_N (3) prompts of that size, each with its own first line and cache_prompt off (no
    reuse), one token decoded; the server's own prompt tok/s per run, and the arm's diagnostics over that stretch of
    the log (timing_tables). A server per context: the diagnostics print every 100 graphs, so one server's stretch
    holds one context only; a sched-timing arm runs enough prompts for at least 110 target graphs (512 a ubatch)."""
    rec: dict = {"env": {k: v for k, v in arm["env"].items() if k.startswith("GGML_")}}
    logp = os.path.join(out, f"{name}.server.log")
    text = corpus()
    n = int(os.environ.get("FLASHNEXT_PREFILL_N") or 3)
    ub = int(arm["args"].get("-ub") or 512)
    for k in [int(x) for x in (os.environ.get("FLASHNEXT_PREFILL_CONTEXTS") or "8 32").split()]:
        runs_n = n
        if arm["env"].get("GGML_SCHED_TIMING"):
            runs_n = max(n, -(-110 // max(1, k * 1024 // ub)))
        row: dict = {"runs": []}
        srv = None
        try:
            srv = Launched(name, arm, port, out)
            row["load_s"] = srv.load_s
            w = completion(srv.base, "Warm-up.\n" + text[: 1024 * 4], 1, slot=0, fresh=True)
            row["warmup"] = {"prompt_n": w["prompt_n"], "prompt_tps": w["prompt_tps"]}
            mark = os.path.getsize(logp)
            for i in range(runs_n):
                # the same prompts in every arm (cache_prompt off re-reads them whole), so the first token's top-5
                # probabilities compare across arms: the prompt path's logits
                head = f"Run {k}k-{i}: read the following source.\n"
                r = _raw_completion(srv.base, head + text[: k * 1024 * 4], 1, fresh=True)
                row["runs"].append({"prompt_n": r["prompt_n"], "prompt_tps": r["prompt_tps"],
                                    "first_top": [(t["token"], round(t["logprob"], 4)) for t in r["first_top"]
                                                  if t.get("logprob") is not None]})
            with open(logp, encoding="utf-8", errors="replace") as f:
                f.seek(mark)
                tail = f.read().splitlines()
                row.update(timing_tables(tail))
                # 0029's per-batch lines (LLAMA_LAYER_MAJOR_DEBUG=1, or its slot setup at load): kept as written
                lm_lines = [ln.split(" I ", 1)[-1].strip()[:300] for ln in tail
                            if "ubatches," in ln or "ubatch by ubatch" in ln or "layer graphs" in ln]
                if lm_lines:
                    row["layer_major"] = lm_lines[-12:]
        except Exception as e:                                               # noqa: BLE001
            row["error"] = f"{type(e).__name__}: {e}"
        finally:
            if srv is not None:
                row.update(srv.stop())
        rec[f"{k}k"] = row
    return rec


def step_cold(name: str, arm: dict, port: int, out: str) -> dict:
    """The first prompt after a load with a COLD file cache (docs/FLASH-NEXT.md 13.3, bench/fn_first_prompt.py's
    `cold cache` arm without the proxy): n (FLASHNEXT_COLD_N, 3) times: read 61.9 GB of a model file the stack never loads through
    the cache (pushes the experts out), a fresh server, one ~24.6K-token prompt, cache_prompt off; the server's own prompt tok/s and
    each layer-major batch's ms (LLAMA_LAYER_MAJOR_DEBUG). Never touches the proxy."""
    evict = ["C:/Users/jwals/textgen/user_data/models/flash-next/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-0000%d-of-00002.gguf" % i
             for i in (1, 2)]
    logp = os.path.join(out, f"{name}.server.log")
    text = corpus()[: 24600 * 37 // 10]
    rec: dict = {"runs": []}
    for i in range(int(os.environ.get("FLASHNEXT_COLD_N") or 3)):
        row: dict = {}
        srv = None
        try:
            t0 = time.time()
            nbytes = 0
            for p in evict:
                with open(p, "rb", buffering=0) as f:
                    while True:
                        b = f.read(16 << 20)
                        if not b:
                            break
                        nbytes += len(b)
            row["evict_s"] = round(time.time() - t0, 1)
            row["evict_GB"] = round(nbytes / 1e9, 1)
            srv = Launched(name, arm, port, out)
            row["load_s"] = srv.load_s
            mark = os.path.getsize(logp)
            r = _raw_completion(srv.base, f"Cold {i}: read the following source." + chr(10) + text, 1, fresh=True)
            row.update({"prompt_n": r["prompt_n"], "prompt_ms": r["prompt_ms"], "prompt_tps": r["prompt_tps"]})
            with open(logp, encoding="utf-8", errors="replace") as f:
                f.seek(mark)
                tail = f.read().splitlines()
            row["batch_ms"] = [float(m.group(1)) for ln in tail for m in [re.search(r"ubatches, \d+ tokens through 48 layers in ([0-9.]+) ms", ln)] if m]
            row["stager_lines"] = [ln.split(" I ", 1)[-1].strip()[:260] for ln in tail if "staging ring" in ln][-4:]
        except Exception as e:                                               # noqa: BLE001
            row["error"] = f"{type(e).__name__}: {e}"
        finally:
            if srv is not None:
                row.update(srv.stop())
        rec["runs"].append(row)
    return rec


CKPT_RE = re.compile(r"created context checkpoint (\d+) of (\d+) \(pos_min = (-?\d+), pos_max = (-?\d+), "
                     r"n_tokens = (\d+), size = ([0-9.]+) MiB(?:, ([0-9.]+) ms)?\)")


def _raw_completion(base: str, prompt: str, n: int, fresh: bool = False) -> dict:
    """One greedy /completion with the first tokens' probabilities: what the checkpoint check compares."""
    body = {"prompt": prompt, "id_slot": 0, "n_predict": n, "cache_prompt": not fresh, "temperature": 0.0,
            "top_k": 1, "seed": 0, "n_probs": 5}
    st, txt = ec.http("POST", base + "/completion", body, timeout=7200)
    if st != 200:
        raise RuntimeError(f"/completion: HTTP {st}: {txt[:300]}")
    r = json.loads(txt)
    t = r.get("timings") or {}
    probs = r.get("completion_probabilities") or []
    first = probs[0] if probs else {}
    top = first.get("top_logprobs") or first.get("top_probs") or []
    return {"content": r.get("content") or "", "prompt_n": t.get("prompt_n"), "cache_n": t.get("cache_n"),
            "prompt_ms": round(t.get("prompt_ms") or 0), "prompt_tps": t.get("prompt_per_second"),
            "predicted_n": t.get("predicted_n"),
            "first_top": [{"token": x.get("token"), "logprob": x.get("logprob"), "prob": x.get("prob")} for x in top]}


def _tokens(base: str, text: str) -> list[int]:
    st, txt = ec.http("POST", base + "/tokenize", {"content": text}, timeout=600)
    if st != 200:
        raise RuntimeError(f"/tokenize: HTTP {st}")
    return json.loads(txt)["tokens"]


def _lcp(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def step_ckpt(name: str, arm: dict, port: int, out: str) -> dict:
    """0023's check (the coordinator, 2026-10-02): prompt A (~30K tokens), then B that shares A's first ~20K and
    differs after, then C that differs inside the first 16K -- all greedy, 48 tokens. With --checkpoint-every N in
    the arm, B must reuse exactly the last periodic checkpoint at or before the first differing token
    (floor(lcp / N) * N tokens) and C must reuse 0 when its lcp is below N (no worse than today); without the flag B
    reuses 0 (today). Then a fresh server reads B cold, twice (the second with cache_prompt off): the resumed B's
    text must equal the cold one's -- judged only if the two cold runs equal each other (this card's own run-to-run
    variation, window E1). The first token's top-5 probabilities of each run are kept beside the texts, and the
    log's checkpoint lines (size, ms) for the budget."""
    every = int(arm["args"].get("--checkpoint-every") or 0)
    rec: dict = {"checkpoint_every": every}
    logp = os.path.join(out, f"{name}.server.log")
    text = corpus()
    a = text[: 30 * 1024 * 4]
    b = text[: 20 * 1024 * 4] + text[1_000_000: 1_000_000 + 10 * 1024 * 4]
    c = text[: 8 * 1024 * 4] + text[2_000_000: 2_000_000 + 22 * 1024 * 4]
    srv = None
    try:
        srv = Launched(name, arm, port, out)
        mark = os.path.getsize(logp)
        ta, tb, tc = _tokens(srv.base, a), _tokens(srv.base, b), _tokens(srv.base, c)
        rec["tokens"] = {"A": len(ta), "B": len(tb), "C": len(tc), "lcp_AB": _lcp(ta, tb), "lcp_BC": _lcp(tb, tc)}
        rec["A"] = _raw_completion(srv.base, a, 48)
        rec["B"] = _raw_completion(srv.base, b, 48)
        rec["C"] = _raw_completion(srv.base, c, 48)
        with open(logp, encoding="utf-8", errors="replace") as f:
            f.seek(mark)
            tail = f.read().splitlines()
        cps = [CKPT_RE.search(ln) for ln in tail]
        rec["checkpoints"] = [{"n_tokens": int(m.group(5)), "mib": float(m.group(6)),
                               "ms": float(m.group(7)) if m.group(7) else None} for m in cps if m]
        rec["restored"] = [ln.split(" | ", 2)[-1].strip() for ln in tail if "restored context checkpoint" in ln]
        rec["full_reprocess"] = sum("forcing full prompt re-processing" in ln for ln in tail)
        rec.update(srv.stop())
        srv = None
        srv = Launched(name, arm, port, out)
        rec["B_cold"] = _raw_completion(srv.base, b, 48)
        rec["B_cold2"] = _raw_completion(srv.base, b, 48, fresh=True)
    except Exception as e:                                               # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
    finally:
        if srv is not None:
            srv.stop()
    if "error" not in rec:
        lcp_ab, lcp_bc = rec["tokens"]["lcp_AB"], rec["tokens"]["lcp_BC"]
        want_b = (lcp_ab // every) * every if every else 0
        # after B the slot holds B and its answer; C shares lcp_BC tokens with it
        want_c = (lcp_bc // every) * every if every else 0
        cold_same = rec["B_cold"]["content"] == rec["B_cold2"]["content"]
        rec["expect"] = {"B_cache_n": want_b, "C_cache_n": want_c}
        rec["reuse_PASS"] = (rec["A"]["cache_n"] in (0, None) and (rec["B"]["cache_n"] or 0) == want_b
                             and (rec["C"]["cache_n"] or 0) == want_c)
        rec["cold_runs_equal"] = cold_same
        rec["resumed_equals_cold"] = rec["B"]["content"] == rec["B_cold"]["content"]
        rec["PASS"] = bool(rec["reuse_PASS"] and (rec["resumed_equals_cold"] or not cold_same))
    return rec


LM_RE = re.compile(r"layer[- ]major.*", re.I)


def step_cancel(name: str, arm: dict, port: int, out: str) -> dict:
    """A client that hangs up in the middle of a long prompt, then asks again (the coordinator, 2026-10-02, for 0029:
    a larger batch means a cancel mid-chunk). A ~25K prompt is streamed and the connection closed after
    FLASHNEXT_CANCEL_AFTER seconds (default 20: inside the prompt at ~150-260 tok/s); then the SAME prompt (it must
    finish, reusing what the slot kept), then a different one, then the first again cold-checked against a fresh
    server is not repeated here: the greedy text of the retry is compared with a run that was never cancelled
    (the first request of a second server)."""
    import socket
    rec: dict = {}
    text = corpus()
    p1 = "Cancel check.\n" + text[: 23 * 1024 * 4]
    p2 = "Another prompt.\n" + text[3_000_000: 3_000_000 + 8 * 1024 * 4]
    after = float(os.environ.get("FLASHNEXT_CANCEL_AFTER") or 20)
    srv = None
    try:
        srv = Launched(name, arm, port, out)
        body = json.dumps({"prompt": p1, "n_predict": 48, "cache_prompt": True, "temperature": 0.0, "top_k": 1,
                           "seed": 0, "id_slot": 0, "stream": True}).encode()
        s = socket.create_connection(("127.0.0.1", port), timeout=600)
        s.sendall(b"POST /completion HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
                  + f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
        time.sleep(after)
        s.close()
        rec["cancelled_after_s"] = after
        time.sleep(2)
        r1 = _raw_completion(srv.base, p1, 48)
        r2 = _raw_completion(srv.base, p2, 48)
        rec["retry_same"] = {k: r1[k] for k in ("cache_n", "prompt_n", "prompt_ms", "content")}
        rec["then_other"] = {k: r2[k] for k in ("cache_n", "prompt_n", "prompt_ms", "content")}
        rec["alive"] = ec.http("GET", srv.base + "/health", timeout=10)[0] == 200
        rec.update(srv.stop())
        srv = None
        srv = Launched(name, arm, port, out)
        r0 = _raw_completion(srv.base, p1, 48)
        rec["never_cancelled"] = {k: r0[k] for k in ("cache_n", "prompt_n", "prompt_ms", "content")}
        rec["retry_equals_uncancelled"] = r1["content"] == r0["content"]
    except Exception as e:                                               # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
    finally:
        if srv is not None:
            rec.update(srv.stop())
    return rec


def cache_counts(logp: str) -> list[tuple[int, int, int]]:
    """0016's moe-cache lines in a server log (-lv 4): (steps, hits, misses), running totals."""
    try:
        with open(logp, encoding="utf-8", errors="replace") as f:
            return [tuple(int(x) for x in m.groups()) for m in CACHE_RE.finditer(f.read())]
    except OSError:
        return []


def cache_rate(before: list, after: list) -> dict | None:
    """The VRAM cache's hit rate between two readings of cache_counts (the lines printed in between)."""
    if not after or after == before:
        return None
    s0, h0, m0 = before[-1] if before else (0, 0, 0)
    s1, h1, m1 = after[-1]
    return {"steps": s1 - s0, "hits": h1 - h0, "misses": m1 - m0,
            "hit_rate": round((h1 - h0) / max(1, (h1 - h0) + (m1 - m0)), 4)}


def step_speed(name: str, arm: dict, port: int, out: str) -> dict:
    srv = Launched(name, arm, port, out)
    rec: dict = {"load_s": srv.load_s}
    logp = os.path.join(out, f"{name}.server.log")
    try:
        text = corpus()
        # FLASHNEXT_SPEED_CONTEXTS (e.g. "32"): only those rows -- a like-for-like rerun of one row of another arm
        ks = [int(x) for x in (os.environ.get("FLASHNEXT_SPEED_CONTEXTS") or "4 32 64 128").split()]
        for k in ks:
            runs = []
            c0 = cache_counts(logp)
            for rep in range(3):
                prompt = text[rep * 4096: rep * 4096 + k * 1024 * 4]
                r = completion(srv.base, prompt, 256, slot=rep % 2, ignore_eos=True)
                runs.append({kk: r[kk] for kk in ("prompt_n", "prompt_tps", "predicted_n", "tps", "draft_n",
                                                  "draft_n_accepted", "stop_type", "n_ctx_used")})
            rec[f"{k}k"] = runs
            hr = cache_rate(c0, cache_counts(logp))
            if hr:
                rec[f"{k}k_cache"] = hr
        if int(arm["args"].get("-np") or 4) < 4 or os.environ.get("FLASHNEXT_SPEED_CONTEXTS"):
            # the operator's layout (-np 2: one conversation + the jjava lane) has no second conversation to idle
            return rec
        # placement: an idle 64K conversation on slot 2 while slot 3 decodes at 8K (the #59 shape)
        completion(srv.base, text[: 64 * 1024 * 4], 0, slot=2)
        rec["8k_with_idle_64k"] = [
            {kk: r[kk] for kk in ("prompt_n", "prompt_tps", "predicted_n", "tps", "draft_n", "draft_n_accepted",
                                  "stop_type", "n_ctx_used")}
            for r in (completion(srv.base, text[rep * 997: rep * 997 + 8 * 1024 * 4], 256, slot=3, ignore_eos=True)
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
    check(ca[ca.index("-m") + 1].endswith("Q2_0-00001-of-00002.gguf") and "--moe-expert-cache" not in ca
          and ca[ca.index("--n-cpu-moe") + 1] == str(48 - 66 * 48 // 512) and "--spec-type" in ca
          and ca[ca.index("-C") + 1] == "0xFFFF",
          "flash-all-q2: the Q2_0 file, its measured room as whole GPU expert layers, MTP, the P-cores")
    check("--mmproj" not in ca and "--mmproj" not in command(b["flash-cpu"], 18100)
          and "--mmproj" in command(b["base"], 18100), "flash arms carry no projector; base keeps the deployed one")
    check(derive_cache_all(1000) == 1 and derive_cache_all(1000 + 48 * 3) >= 2 and
          derive_cache_all(1000, MMPROJ_MIB) > derive_cache_all(1000), "fit2: slots from the measured free VRAM")
    c4 = command(arms(42, 7, {"iq2": 60, "q2": 66, "mmcpu": 80}, 12)["flash-a4000"], 18100)
    check(c4[c4.index("--n-cpu-moe") + 1] == "31" and c4[c4.index("-dev") + 1] == "CUDA0,CUDA1"
          and c4[c4.index("-ts") + 1] == "1,0" and r"blk\.(31|32|" in c4[c4.index("-ot") + 1]
          and c4[c4.index("-ot") + 1].endswith(r"42)\.ffn_(up|gate|down)_exps\.weight=CUDA1")
          and "--moe-expert-cache" not in c4,
          "flash-a4000: 31 layers in RAM, the next 12 on CUDA1, the last 5 on CUDA0 (60 slots -> 5 layers)")
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
    ap.add_argument("--redo", nargs="*", default=[],
                    help="arms whose finished steps in --out are run again, not reused (a changed engine or arm)")
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
    # what an earlier run into this --out recorded is carried from the start, so a run that fails early (a launch
    # that exits) does not overwrite gate.json without it (2026-09-29: fit2's failure dropped the seeded yardstick)
    try:
        _prev0 = json.load(open(os.path.join(a.out, "gate.json"), encoding="utf-8"))
    except Exception:                                                    # noqa: BLE001
        _prev0 = {}
    for _k in ("kernels", "fit", "fit2", "kl_yardstick"):
        if _prev0.get(_k):
            results[_k] = _prev0[_k]
    results["arms"].update(_prev0.get("arms") or {})
    why = ec.wait_quiet(prod_argv, 3600)
    if why:
        print(f"not run: {why}")
        return 3
    sys.path.insert(0, os.path.join(ROOT, "mcp"))
    lane_hold = None
    try:
        lane_hold = ec.hold_lane("Flash-Next gate (bench/flashnext_gate.py)", 12 * 3600)
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
            # -lv 4: the engine's own allocation log (every buffer it reserves), to name what grows after load
            probe = dict(probe, args={**probe["args"], "-lv": "4"})
            srv = Launched("fit2", probe, a.port, a.out)
            g_load = ec.gpu() or {}
            completion(srv.base, corpus()[: 12288 * 4], 64)
            g = ec.gpu() or {}
            rec = {"load_s": srv.load_s, "gpu_after_load": g_load, "gpu_after_warm": g, **srv.stop()}
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
            # the steps this run does not ask for stay as recorded (2026-09-30: a run with --steps contract lane speed
            # rewrote cache-all's record without its kl, needles and corrupt)
            r.update({k: v for k, v in done_before.items() if k not in a.steps})

            def reuse(step: str, n=n, r=r, done_before=done_before) -> bool:
                """A rerun into the same --out keeps a step this arm FINISHED there (no error, no guard)."""
                got = done_before.get(step)
                if n in (a.redo or []) or not isinstance(got, dict) or got.get("error") or got.get("guard"):
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
                r["kl"] = step_kl(n, arm, a.out, base_logits, arm.get("kl_batch", 16))
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
            if "fault" in a.steps and "fault" in arm["checks"] and reuse("fault"):
                pass
            elif "fault" in a.steps and "fault" in arm["checks"]:
                r["fault"] = step_fault(n, arm, a.out)
                save()
            if "jjava" in a.steps and "jjava" in arm["checks"] and reuse("jjava"):
                pass
            elif "jjava" in a.steps and "jjava" in arm["checks"]:
                r["jjava"] = step_jjava(n, arm, a.port, a.out)
                save()
            if "probe" in a.steps and "probe" in arm["checks"] and reuse("probe"):
                pass
            elif "probe" in a.steps and "probe" in arm["checks"]:
                r["probe"] = step_probe(n, arm, a.port, a.out)
                save()
            if "ckpt" in a.steps and "ckpt" in arm["checks"] and reuse("ckpt"):
                pass
            elif "ckpt" in a.steps and "ckpt" in arm["checks"]:
                r["ckpt"] = step_ckpt(n, arm, a.port, a.out)
                save()
            if "cancel" in a.steps and "cancel" in arm["checks"] and reuse("cancel"):
                pass
            elif "cancel" in a.steps and "cancel" in arm["checks"]:
                r["cancel"] = step_cancel(n, arm, a.port, a.out)
                save()
            if "prefill" in a.steps and "prefill" in arm["checks"] and reuse("prefill"):
                pass
            elif "prefill" in a.steps and "prefill" in arm["checks"]:
                r["prefill"] = step_prefill(n, arm, a.port, a.out)
                save()
            if "cold" in a.steps and "cold" in arm["checks"] and reuse("cold"):
                pass
            elif "cold" in a.steps and "cold" in arm["checks"]:
                r["cold"] = step_cold(n, arm, a.port, a.out)
                save()
            if "lane" in a.steps and "lane" in arm["checks"] and reuse("lane"):
                pass
            elif "lane" in a.steps and "lane" in arm["checks"]:
                r["lane"] = step_lane(n, arm, a.port, a.out)
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
        results["lane"] = ec.release_lane(lane_hold)
        results["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        json.dump(results, open(os.path.join(a.out, "gate.json"), "w"), indent=1)
    print(json.dumps({n: {s: (v.get("PASS", v.get("verdict")) if isinstance(v, dict) else v) for s, v in r.items()}
                      for n, r in results["arms"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
