"""PHASE 2 of the bonsai-ada-surgery merge (docs/ENGINES.md "bonsai-ada-surgery
merge"): every GPU gate in one go, then -- only when every gate passes and
--deploy is given -- the deploy of the TARGET configuration.

THE TARGET (operator, 2026-09-27: "Use their q8, it's the whole point, they fit
q8 with the vram thing"): llama-bonsai2-ada; q8_0 K/V AND q8_0 draft K/V
(-ctkd/-ctvd); -c 262,144 (the trained window) with the tiered KV cache
(--kv-vram-cells N: cells [0, N) in VRAM, the rest in pinned host RAM); draft
n-max 2, tail 4 (their recipe; our 0037 keys the tail on pool cells); the
ProCreations head when its graft is recorded and wins, else the lean head.

    python bench/engine_phase2.py --window --out bench/results/engine_corruption/ada-phase2-<date>
    python bench/engine_phase2.py --window --out DIR --deploy --key-file PATH
    python bench/engine_phase2.py --gates DIR           # re-read a finished run's records
    python bench/engine_phase2.py --record-graft STRIP_CHECK_GGUF

--window is the operator's statement that the 5060 Ti is theirs ("GPU free"):
each arm unloads production `bonsai`, runs on a spare loopback port with
config.yaml's exact `bonsai` argv and env (the binary, the model file or flags
swapped per arm), and ALWAYS puts production back (bench/engine_corruption.py's
window discipline, reused: quiet stack, the worker's gpu lane paused, a guard
that kills the arm the moment llama-swap starts anything on the card).

STEPS, in order (each writes its records under --out):

  0 fit         for each head: size N the way start-server.ps1 does, but on
                MEASURED free VRAM instead of its 4070 constants: launch the
                target at a first N, warm it (8k prefill + decode on one slot),
                read the card's lowest free VRAM, and move cells into VRAM
                until free = MARGIN (their demotion margin: 1,000 MiB when
                nvidia-smi says the card drives no display, else 1,300;
                start-server.ps1 and docs/Q8_FULL_CONTEXT.md "The VRAM line",
                measured on their 4070: 600 paged at once, 1,000 held a
                10-minute soak headless). A cell moved into VRAM costs its
                target K/V (34,816 B) and its draft K/V (2,176 B) less the
                row of the shared staging buffer it frees (net 34,816 B).
                Relaunch at N and check; up to 3
                rounds. fit-<head>.json holds every round.
  1 corruption  bench/engine_corruption.py (greedy x3 reps, 16 plain, 10 tool
                requests) on `current`, `ada` (our flags), `ada-vec` (decode on
                the FA vec kernel: GGML_CUDA_FA_MMA_DECODE_MIN_KV=0) and the
                target arms `target-lean` / `target-pc`.
  2 speed       llama-server's own timings (greedy code continuation,
                /completion with id_slot, 256 tokens):
                  depth      one conversation alone at 512 / 8k / 32k / 64k /
                             128k; the target arms also at 196,608 and 245,760
                             (past today's pool: reported, marked by whether
                             the pool's cells reach N);
                  placement  (#59) A at 8k / 32k / 64k: alone, an idle 41k B
                             below / above A, 3 x 40k above A;
                  spill      target arms: A at 8k decoding while 3 idle slots
                             hold 3 x 60k cells, so the pool's highest used cell
                             is past N (what every conversation pays then);
                  concurrent a shallow (8k) and a deep (64k) slot decoding at
                             once: the shallow slot's draft acceptance;
                  accept     draft acceptance through /v1/chat/completions.
                Arms: current, ada, ada-vec, target-lean, target-pc (graft
                recorded), target-window (the target + --spec-draft-window
                16384 with 0036), and REPORT-ONLY arms for attribution:
                ada-draft2, ada-dkv-q8, ada-np1, ada-np1-window,
                ada-window-0031 (the first build, 0031 without 0036).
  3 gates       gates.json + gates.md, comparisons with the current engine;
                the one tolerance (0.95) is the live suite's own slots rule
                ("cleared >= 95% of kept"), --tolerance changes it.
                  G1 corruption  ada, ada-vec and the chosen target: no error,
                                 greedy reps 1..n identical on 3/3 prompts,
                                 0/16 plain with a symptom run, tools 10/10.
                                 Cross-arm text identity is NOT a gate.
                  G2 depth       target decode median >= tol x current's at
                                 every depth current can hold (<= 131,072).
                  G3 placement   target decode median >= tol x current's in
                                 every placement cell.
                  G4 VRAM        the lowest free VRAM while each gated arm ran
                                 >= 600 MiB (config.yaml's floor).
                  head           ProCreations only when its graft is recorded
                                 in models/manifest.yaml, target-pc passes G1,
                                 and its decode (sum of the gated depth
                                 medians) beats target-lean's; else lean.
                Past-the-line depths, spill, the window arm and the report-only
                arms are REPORTED, never gated.
  4 deploy      (--deploy, only when G1-G4 pass) config.yaml backed up; in
                `bonsai`'s block: server_nudge -> the new engine, -c 262144,
                --kv-vram-cells N (the fit), -ctkd/-ctvd q8_0, draft n-max 2,
                tail 4, -m -> the graft if it won; the engine and model
                checks clean; the stack restarted the watchdog's way; then
                scripts/deploy_check.py --key-file must exit 0. Anything else
                rolls config.yaml back and restarts again.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import engine_corruption as ec                                       # noqa: E402

PY = sys.executable
LANE_BY = "engine_phase2"
ENGINE = "llama-bonsai2-ada"
PC_FILE = "Ternary-Bonsai-2-27B-Abliterated-PTQ1_0-mtp-procreations.gguf"
DEPTHS = (512, 8192, 32768, 65536, 131072)
PLACEMENT = (8192, 32768, 65536)
B_TOKENS = 41000        # the idle slot of the #59 probe (41,326 cells there)
OTHER_TOKENS = 40000    # each of 3 idle slots in the "3 x 40k above" case
GEN = 256
REPS = 3
VRAM_FLOOR_MIB = 600    # config.yaml, bonsai's -c note (operator, 2026-09-25)
TOLERANCE = 0.95        # mcp/test_live_stack.py slots check: cleared >= 95% of kept
REPORT_ARMS = ("ada-draft2", "ada-dkv-q8", "ada-np1", "ada-np1-window", "ada-window-0031", "target-window")
TARGET_CTX = 262144      # the trained window (operator: q8 fitted with the tiered cache)
TARGET_DEPTHS = DEPTHS + (196608, 245760)
CELL_BYTES = 34816       # q8_0 K+V per cell: 16 attention layers x 2 x 4 heads x 256 x 34/32 B
DRAFT_CELL_BYTES = 2176  # the MTP layer's q8_0 K+V per cell (1/16 of the above)
MARGIN_HEADLESS_MIB = 1000   # start-server.ps1: "1000 held a 10-minute soak" with no display on the card
MARGIN_DISPLAY_MIB = 1300    # start-server.ps1: "1300 MiB held, 800 did not" with the display on it
FIT_START_CELLS = 131072     # first round: well inside today's all-VRAM pool (181,248)
FIT_ROUNDS = 3
# The first llama-bonsai2-ada build (51d64e6f): the series' 0031 without our 0036.
ADA_0031 = "C:/Users/jwals/engines/llama-bonsai2-ada-51d64e6f/src/build/bin/llama-server.exe"


def log(msg: str) -> None:
    ec.log(msg)


# ---------------------------------------------------------------- setup ----
def setup() -> dict:
    import build_engine as be
    prod_argv, prod_env = ec.production_command()
    man = be.load_yaml(be.MANIFEST)
    ada = ((man["engines"].get(ENGINE) or {}).get("shipped") or {}).get("path")
    lean = prod_argv[prod_argv.index("-m") + 1]
    pc = os.path.dirname(lean) + "/" + PC_FILE     # config.yaml's form: ${models}/<file>, forward slashes
    cfg, macros = ec._config()
    return {"prod_argv": prod_argv, "prod_env": prod_env, "current": prod_argv[0],
            "ada": ada, "lean": lean, "pc": pc if os.path.exists(pc) else None,
            "pool": int(prod_argv[prod_argv.index("-c") + 1])}


def pc_recorded(pc: str | None) -> bool:
    """Is the graft a models/manifest.yaml artifact with these bytes' size?"""
    if not pc:
        return False
    import verify_artifacts as va
    m = va.load_manifest()
    for a in m["artifacts"]:
        if str(a.get("path", "")).replace("\\", "/").endswith("/" + PC_FILE):
            return a.get("size") == os.path.getsize(pc)
    return False


# --------------------------------------------------------------- window ----
_LANE_HOLD: dict | None = None


def _hold_lane_once(why: str) -> None:
    """The gpu lane paused for the whole run (every in_window call), released at the process's exit: resuming it
    after each window let the worker start an idle-gated gpu job between them (2026-09-29, the coordinator)."""
    global _LANE_HOLD
    if _LANE_HOLD is None:
        import atexit
        _LANE_HOLD = ec.hold_lane(why, 12 * 3600)
        atexit.register(ec.release_lane, _LANE_HOLD)


def in_window(name: str, s: dict, fn, out: str, attempts: int, wait_min: float):
    """Run fn(base_url, rec) against a fresh arm server inside a window:
    engine_corruption's discipline (quiet stack, lane paused, guard, restore)."""
    sys.path.insert(0, os.path.join(ROOT, "mcp"))
    rec: dict = {}
    res: dict = {}
    for attempt in range(1, attempts + 1):
        why = ec.wait_quiet(s["prod_argv"], wait_min * 60)
        if why:
            return {"arm": name, "error": f"not run: {why}"}, res
        try:
            _hold_lane_once(f"engine PHASE 2 ({name})")
            st, txt = ec.http("POST", f"{ec.SWAP}/api/models/unload/{ec.PROD_ID}", timeout=120)
            log(f"[{name}] attempt {attempt}: unload bonsai: HTTP {st}")
            t0 = time.time()
            while ec.PROD_ID in ec.running() or any("llama-server" in p for p in ec.card_pids()):
                if time.time() - t0 > 180:
                    raise RuntimeError("bonsai did not leave the card in 180 s")
                time.sleep(2)
            time.sleep(3)
            rec = run_server(name, s, fn, out)
        finally:
            # the lane stays paused between windows of this run; released once, at exit (_hold_lane_once)
            res = ec.restore(s["prod_argv"])
        if not rec.get("guard"):
            break
        os.replace(os.path.join(out, f"{name}.json"), os.path.join(out, f"{name}.aborted{attempt}.json"))
        log(f"[{name}] aborted by the guard ({rec['guard']}); retrying in a later quiet spell")
    return rec, res


def run_server(name: str, s: dict, fn, out: str) -> dict:
    binary, env_extra, args = s["arms"][name]
    cmd = ec.arm_command(s["prod_argv"], binary, s["port"], args)
    env = dict(os.environ)
    for k in ("GGML_CUDA_PDL", "GGML_CUDA_BATCH_INVARIANT", "GGML_CUDA_DISABLE_GRAPHS",
              "GGML_CUDA_FA_MMA_DECODE_MIN_KV"):
        env.pop(k, None)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env.update(s["prod_env"])
    env.update(env_extra)
    rec = {"arm": name, "binary": binary, "cmd": cmd, "env_extra": env_extra, "arg_overrides": args,
           "vram_cells": int(args["--kv-vram-cells"]) if "--kv-vram-cells" in args else None,
           "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    logf = open(os.path.join(out, f"{name}.log"), "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)
    guard = ec.Guard(proc, ec.card_models())
    guard.start()
    base = f"http://127.0.0.1:{s['port']}"
    try:
        t0 = time.time()
        while ec.http("GET", base + "/health", timeout=5)[0] != 200:
            if proc.poll() is not None:
                raise RuntimeError(f"server exited {proc.returncode} during load ({guard.tripped})")
            if time.time() - t0 > 600:
                raise RuntimeError("load timeout")
            time.sleep(1)
        rec["load_s"] = round(time.time() - t0, 1)
        if not [p for p in ec.card_pids() if p.split(",")[1].strip() == str(proc.pid)]:
            raise RuntimeError(f"pid {proc.pid} holds no context on the {ec.CARD_NAME}")
        rec["build_info"] = json.loads(ec.http("GET", base + "/props")[1] or "{}").get("build_info")
        rec["gpu_loaded"] = ec.gpu()
        log(f"[{name}] loaded in {rec['load_s']} s, build {rec['build_info']}, card {rec['gpu_loaded']}")
        fn(base, rec, guard)
    except Exception as e:                                           # noqa: BLE001
        rec["error"] = repr(e)
        log(f"[{name}] ERROR {e!r}")
    finally:
        guard.stop()
        rec.update(gpu_min_free=guard.min_free, gpu_max_used=guard.max_used, guard=guard.tripped,
                   ended=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        proc.terminate()
        try:
            proc.wait(30)
        except subprocess.TimeoutExpired:
            proc.kill()
        logf.close()
        with open(os.path.join(out, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=1)
    return rec


# ----------------------------------------------------------- the target ----
def target_args(n_cells: int, model: str | None = None, window: bool = False) -> dict:
    a = {"-c": str(TARGET_CTX), "--cache-type-k-draft": "q8_0", "--cache-type-v-draft": "q8_0",
         "--spec-draft-n-max": "2", "--spec-draft-n-max-tail": "4"}
    if n_cells < TARGET_CTX:           # the whole window fits: a plain cache (start-server.ps1 does the same)
        a["--kv-vram-cells"] = str(n_cells)
    if model:
        a["-m"] = model
    if window:
        a["--spec-draft-window"] = "16384"
    return a


def margin_mib() -> tuple[int, str]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=display_active", "--format=csv,noheader",
                              "-i", ec.CARD_UUID], capture_output=True, text=True, timeout=15).stdout.strip()
    except Exception:                                                # noqa: BLE001
        out = "unknown"
    return (MARGIN_HEADLESS_MIB, out) if out == "Disabled" else (MARGIN_DISPLAY_MIB, out)


def host_free_gb() -> float | None:
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"],
                             capture_output=True, text=True, timeout=60).stdout.strip()
        return round(int(out) * 1024 / 1e9, 2)
    except Exception:                                                # noqa: BLE001
        return None


def warm_fn(base: str, rec: dict, guard) -> None:
    """What a fit round measures on: a real prefill and decode (compute buffers allocated)."""
    toks = corpus_tokens(base)
    rec["warm"] = [gen(base, 0, toks[:8192], GEN), gen(base, 1, toks[8192:8192 + 2048], GEN)]


def fit(head: str, model: str | None, s: dict, out: str, attempts: int, wait_min: float) -> dict:
    """N for this head: measured free VRAM, their margin. fit-<head>.json."""
    margin, display = margin_mib()
    # A cell moved into VRAM adds its target K/V (34,816 B) and its draft K/V (2,176 B: without
    # --spec-draft-window the draft context is tiered at the same N) and frees its row of the ONE shared
    # staging buffer (one layer's K+V: 2,176 B; start-server.ps1's "(Ctx-N)*CellBytes/16").
    per_cell = CELL_BYTES
    n = FIT_START_CELLS
    rounds = []
    for r in range(FIT_ROUNDS):
        name = f"fit-{head}-{r}"
        s["arms"][name] = (s["ada"], {}, target_args(n, model))
        rec, res = in_window(name, s, warm_fn, os.path.join(out, "fit"), attempts, wait_min)
        free = rec.get("gpu_min_free")
        rounds.append({"n": n, "min_free_mib": free, "error": rec.get("error")})
        log(f"[fit {head}] round {r}: N={n} min free {free} MiB (margin {margin}, display_active {display})")
        if rec.get("error") or free is None:
            break
        step = int((free - margin) * 1048576 / per_cell) // 256 * 256
        if step == 0 or (0 < step and n >= TARGET_CTX):
            break
        n = max(16384, min(TARGET_CTX, n + step))
        if res and res.get("proxy_health") != 200:
            break
    ok = [x for x in rounds if not x["error"] and x["min_free_mib"] is not None and x["min_free_mib"] >= margin]
    best = max(ok, key=lambda x: x["n"]) if ok else None
    result = {"head": head, "model": model, "margin_mib": margin, "display_active": display,
              "per_cell_bytes": per_cell, "rounds": rounds, "n": best["n"] if best else None,
              "per_slot_note": "the line is on POOL cells: all 4 slots share [0, N); a slot's cells go wherever "
                               "the allocator puts them, and every slot reads the host tail while the pool's "
                               "highest used cell is past N (0037 keys the tail draft size on that)"}
    if result["n"]:
        result["host_ram_gb"] = round((TARGET_CTX - result["n"]) * (CELL_BYTES + DRAFT_CELL_BYTES) / 1e9, 2)
        result["free_ram_gb_at_fit"] = host_free_gb()
    json.dump(result, open(os.path.join(out, f"fit-{head}.json"), "w", encoding="utf-8"), indent=1)
    return result


# ---------------------------------------------------------------- speed ----
def post(base: str, path: str, body: dict, timeout: float = 3600) -> dict:
    st, txt = ec.http("POST", base + path, body, timeout=timeout)
    if st != 200:
        raise RuntimeError(f"{path}: HTTP {st}: {txt[:300]}")
    return json.loads(txt)


def corpus_tokens(base: str) -> list[int]:
    """Real code as filler (not a repeated paragraph, which the MTP head would
    predict too well): the repo's non-test mcp/*.py, sorted, about 2 MB."""
    text, n = [], 0
    for f in sorted(os.listdir(os.path.join(ROOT, "mcp"))):
        if f.endswith(".py") and not f.startswith("test_"):
            t = open(os.path.join(ROOT, "mcp", f), encoding="utf-8", errors="replace").read()
            text.append(t)
            n += len(t)
            if n > 2_000_000:
                break
    return post(base, "/tokenize", {"content": "\n".join(text), "add_special": False})["tokens"]


def shrink_all(base: str) -> None:
    """Empty every slot the way slots.release_idle does without --slot-save-path."""
    for i in range(len(json.loads(ec.http("GET", base + "/slots")[1]))):
        post(base, "/completion", {"prompt": f"x{i}", "n_predict": 1, "id_slot": i, "cache_prompt": True})


def gen(base: str, slot: int, tokens: list[int], n: int) -> dict:
    d = post(base, "/completion", {"prompt": tokens, "n_predict": n, "id_slot": slot, "cache_prompt": True,
                                   "temperature": 0, "top_k": 1, "seed": 0, "ignore_eos": True})
    t = d.get("timings") or {}
    return {k: t.get(k) for k in ("prompt_n", "prompt_per_second", "predicted_n", "predicted_per_second",
                                  "draft_n", "draft_n_accepted")}


def med(rows: list[dict]) -> float | None:
    v = [r["predicted_per_second"] for r in rows if r.get("predicted_per_second")]
    return round(statistics.median(v), 2) if v else None


def speed_fn(depth, placement: bool, accept: bool, pool: int, concurrent: bool = False,
             spill: bool = False):
    """depth: True (DEPTHS), a tuple of depths, or False."""
    depths = DEPTHS if depth is True else tuple(depth or ())

    def fn(base: str, rec: dict, guard) -> None:
        toks = corpus_tokens(base)
        rec["corpus_tokens"] = len(toks)

        def seg(n: int, off: int) -> list[int]:
            if off + n > len(toks):
                raise RuntimeError(f"filler corpus has {len(toks)} tokens; needs {off + n}")
            return toks[off:off + n]
        if depths:
            rec["depth"] = []
            for d in depths:
                shrink_all(base)
                cold = gen(base, 0, seg(d, 0), GEN)
                reps = [gen(base, 0, seg(d, 0), GEN) for _ in range(REPS)]
                row = {"depth": d, "cold": cold, "reps": reps, "decode_median": med(reps),
                       "past_line": bool(rec.get("vram_cells")) and d + GEN > rec["vram_cells"]}
                rec["depth"].append(row)
                log(f"  depth {d}: prefill {cold['prompt_per_second']:.0f} tok/s (n={cold['prompt_n']}), "
                    f"decode median {row['decode_median']}")
                if guard.tripped:
                    raise RuntimeError(guard.tripped)
        if placement:
            rec["placement"] = []
            for a in PLACEMENT:
                cases = {"alone": [], "B below A": [(1, B_TOKENS, 1)], "B above A": [(1, B_TOKENS, 0)]}
                if a + 3 * OTHER_TOKENS + 4 * GEN < pool:
                    cases["3 x 40k above A"] = [(i, OTHER_TOKENS, 0) for i in (1, 2, 3)]
                for case, fills in cases.items():
                    shrink_all(base)
                    for slot, n, before in fills:           # idle slots filled BEFORE A (below it)
                        if before:
                            gen(base, slot, seg(n, 140_000 + slot * 45_000), 1)
                    gen(base, 0, seg(a, 0), 1)
                    for slot, n, before in fills:           # ... or AFTER A (above it)
                        if not before:
                            gen(base, slot, seg(n, 140_000 + slot * 45_000), 1)
                    reps = [gen(base, 0, seg(a, 0), GEN) for _ in range(REPS)]
                    row = {"A": a, "case": case, "reps": reps, "decode_median": med(reps)}
                    rec["placement"].append(row)
                    log(f"  placement A={a} {case}: decode median {row['decode_median']}")
                    if guard.tripped:
                        raise RuntimeError(guard.tripped)
        if concurrent:
            # A shallow conversation (slot 0, 8k) and a deep one (slot 1, 64k) decoding AT THE SAME TIME, so
            # their tokens share batches: what --spec-draft-window's trim does to the shallow one's drafts
            # (0031 trims every slot by the batch's lowest position; 0036 each by its own).
            import threading
            rec["concurrent"] = []
            for rep in range(REPS):
                shrink_all(base)
                gen(base, 0, seg(8192, 0), 1)
                gen(base, 1, seg(65536, 140_000), 1)
                out: dict = {}

                def run(slot, toks_, key):
                    out[key] = gen(base, slot, toks_, 512)
                ts = [threading.Thread(target=run, args=(0, seg(8192, 0), "shallow")),
                      threading.Thread(target=run, args=(1, seg(65536, 140_000), "deep"))]
                for t in ts:
                    t.start()
                for t in ts:
                    t.join()
                rec["concurrent"].append(out)
                sh = out.get("shallow") or {}
                log(f"  concurrent rep {rep}: shallow {sh.get('predicted_per_second')} tok/s, "
                    f"acc {sh.get('draft_n_accepted')}/{sh.get('draft_n')}")
                if guard.tripped:
                    raise RuntimeError(guard.tripped)
        if spill:
            # tiered KV: push the pool's highest used cell past --kv-vram-cells with idle slots, then decode a
            # shallow conversation below the line (what every conversation pays while the tail is in use)
            rec["spill"] = []
            for case, fills in {"alone": [], "3 x 60k after A (past the line)": [(i, 60_000) for i in (1, 2, 3)]}.items():
                shrink_all(base)
                gen(base, 0, seg(8192, 0), 1)
                for slot, n in fills:
                    gen(base, slot, seg(n, 140_000 + (slot - 1) * 61_000), 1)
                reps = [gen(base, 0, seg(8192, 0), GEN) for _ in range(REPS)]
                rec["spill"].append({"A": 8192, "case": case, "reps": reps, "decode_median": med(reps)})
                log(f"  spill {case}: decode median {med(reps)}")
        if accept:
            shrink_all(base)
            rows = []
            s = ec.Server(int(base.rsplit(":", 1)[1]))
            for pk, p in ec.GREEDY.items():
                r = s.chat([{"role": "user", "content": p}], 512, False, {"temperature": 0, "top_k": 1}, 0)
                rows.append({"kind": "greedy", "prompt": pk, **{k: (r.get("timings") or {}).get(k) for k in
                             ("predicted_n", "predicted_per_second", "draft_n", "draft_n_accepted")}})
            for i in range(8):
                r = s.chat([{"role": "user", "content": ec.PLAIN[i % 2]}], 1500, True, ec.PROD_SAMPLING, 1000 + i)
                rows.append({"kind": "plain", "i": i, **{k: (r.get("timings") or {}).get(k) for k in
                             ("predicted_n", "predicted_per_second", "draft_n", "draft_n_accepted")}})
            dn = sum(r.get("draft_n") or 0 for r in rows)
            da = sum(r.get("draft_n_accepted") or 0 for r in rows)
            rec["accept"] = {"rows": rows, "draft_n": dn, "draft_n_accepted": da,
                             "rate": round(da / dn, 4) if dn else None}
            log(f"  acceptance {da}/{dn} = {rec['accept']['rate']}")
    return fn


# ---------------------------------------------------------------- gates ----
def corruption_ok(r: dict | None) -> tuple[bool, str]:
    if not r:
        return False, "no record"
    if r.get("error"):
        return False, r["error"]
    g = r.get("greedy") or []
    for pk in ec.GREEDY:
        texts = [x["content"] for x in g if x["prompt"] == pk and x["rep"] >= 1]
        if len(texts) < 2 or len(set(texts)) != 1:
            return False, f"greedy reps 1..n differ on {pk}"
    p = r.get("plain") or []
    bad = [x["i"] for x in p if x["symptom_runs"] or x["long_runs"] or x["punct_runs"] or x["word_repeats"]]
    if bad or len(p) < 16:
        return False, f"plain symptoms in {bad} (n={len(p)})"
    t = r.get("tools") or []
    if len(t) < 10 or not all(x["ok"] for x in t):
        return False, f"tools {sum(1 for x in t if x['ok'])}/{len(t)}"
    fin: dict = {}
    for x in p:
        fin[x["finish"]] = fin.get(x["finish"], 0) + 1
    return True, f"reps identical, 0/{len(p)} symptoms, tools {len(t)}/{len(t)}, finishes {fin}"


def load(path: str) -> dict | None:
    try:
        return json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError):
        return None


def gates(out: str, tol: float, s: dict | None = None) -> dict:
    cor = {n: load(os.path.join(out, "corruption", f"{n}.json"))
           for n in ("current", "ada", "ada-vec", "target-lean", "target-pc")}
    spd = {n: load(os.path.join(out, "speed", f"{n}.json"))
           for n in ("current", "ada", "ada-vec", "target-lean", "target-pc") + REPORT_ARMS}
    fits = {h: load(os.path.join(out, f"fit-{h}.json")) for h in ("lean", "pc")}
    g: dict = {"tolerance": tol, "fit": fits}

    def depth_map(r):
        return {x["depth"]: x["decode_median"] for x in (r or {}).get("depth") or []}

    def place_map(r):
        return {(x["A"], x["case"]): x["decode_median"] for x in (r or {}).get("placement") or []}

    def dsum(r):
        return sum(v for d, v in depth_map(r).items() if v and d in DEPTHS)
    # the head
    recorded = pc_recorded((s or {}).get("pc")) if s else None
    pc_ok, pc_why = corruption_ok(cor["target-pc"]) if cor["target-pc"] else (False, "no target-pc run")
    lean_sum, pc_sum = dsum(spd["target-lean"]), dsum(spd["target-pc"])
    choice = "pc" if (pc_ok and recorded and pc_sum > lean_sum) else "lean"
    tgt = f"target-{choice}"
    g["head"] = {"choice": choice, "procreations_recorded": recorded, "procreations_G1": pc_why,
                 "lean_decode_sum": round(lean_sum, 2), "procreations_decode_sum": round(pc_sum, 2),
                 "lean_accept": ((spd["target-lean"] or {}).get("accept") or {}).get("rate"),
                 "procreations_accept": ((spd["target-pc"] or {}).get("accept") or {}).get("rate")}
    g["target"] = tgt
    g["G1"] = {n: dict(zip(("ok", "why"), corruption_ok(cor[n]))) for n in ("ada", "ada-vec", tgt)}
    g["G1_pass"] = all(v["ok"] for v in g["G1"].values())
    cur_d, tgt_d = depth_map(spd["current"]), depth_map(spd[tgt])
    g["G2"] = [{"depth": d, "current": cur_d.get(d), "target": tgt_d.get(d),
                "ok": bool(cur_d.get(d) and tgt_d.get(d) and tgt_d[d] >= tol * cur_d[d])} for d in DEPTHS]
    g["G2_pass"] = all(x["ok"] for x in g["G2"])
    cur_p, tgt_p = place_map(spd["current"]), place_map(spd[tgt])
    ada_p, vec_p = place_map(spd["ada"]), place_map(spd["ada-vec"])
    g["G3"] = [{"A": k[0], "case": k[1], "current": v, "target": tgt_p.get(k), "ada": ada_p.get(k),
                "ada_vec": vec_p.get(k), "ok": bool(v and tgt_p.get(k) and tgt_p[k] >= tol * v)}
               for k, v in sorted(cur_p.items())]
    g["G3_pass"] = bool(g["G3"]) and all(x["ok"] for x in g["G3"])
    gated = [("corruption/" + n, cor[n]) for n in ("ada", "ada-vec", tgt)] +             [("speed/" + n, spd[n]) for n in ("ada", "ada-vec", tgt)]
    g["G4"] = {n: {"min_free_mib": (r or {}).get("gpu_min_free"),
                   "ok": (r or {}).get("gpu_min_free") is not None and r["gpu_min_free"] >= VRAM_FLOOR_MIB}
               for n, r in gated if r}
    g["G4_pass"] = bool(g["G4"]) and all(x["ok"] for x in g["G4"].values())
    fit = fits.get(choice) or {}
    g["G0_fit"] = {"n": fit.get("n"), "ok": bool(fit.get("n"))}
    # reported, never gated
    past = {d: v for d, v in tgt_d.items() if d not in DEPTHS}

    def acc(r):
        rows = [x for d in (r or {}).get("depth") or [] for x in d["reps"]]
        dn = sum(x.get("draft_n") or 0 for x in rows)
        da = sum(x.get("draft_n_accepted") or 0 for x in rows)
        return round(da / dn, 4) if dn else None

    def conc(r):
        sh = [c.get("shallow") or {} for c in (r or {}).get("concurrent") or []]
        dn = sum(x.get("draft_n") or 0 for x in sh)
        da = sum(x.get("draft_n_accepted") or 0 for x in sh)
        v = [x["predicted_per_second"] for x in sh if x.get("predicted_per_second")]
        return {"shallow_accept": round(da / dn, 4) if dn else None,
                "shallow_decode_median": round(statistics.median(v), 2) if v else None}
    g["report"] = {"target_past_line": past}
    for n in ("current", "ada", "target-lean", "target-pc") + REPORT_ARMS:
        r = spd.get(n)
        if not r:
            continue
        g["report"][n] = {"decode_by_depth": depth_map(r), "accept_at_depth": acc(r),
                          "accept_chat": ((r.get("accept") or {}).get("rate")),
                          "concurrent": conc(r) if r.get("concurrent") else None,
                          "spill": [(x["case"], x["decode_median"]) for x in r.get("spill") or []] or None,
                          "min_free_mib": r.get("gpu_min_free"), "error": r.get("error")}
    g["all_pass"] = g["G0_fit"]["ok"] and g["G1_pass"] and g["G2_pass"] and g["G3_pass"] and g["G4_pass"]
    with open(os.path.join(out, "gates.json"), "w", encoding="utf-8") as f:
        json.dump(g, f, indent=1, default=str)
    lines = [f"# PHASE 2 gates ({time.strftime('%Y-%m-%d %H:%M')})", "",
             f"ALL PASS: **{g['all_pass']}** (tolerance {tol}); target = {tgt}, N = {fit.get('n')}", "",
             "## G0 fit (VRAM line)", "```", json.dumps(fits, indent=1, default=str), "```", "", "## G1 corruption"]
    lines += [f"- {n}: {'PASS' if v['ok'] else 'FAIL'} -- {v['why']}" for n, v in g["G1"].items()]
    lines += ["", "## G2 depth (decode tok/s, median of 3)", "| depth | current | target | ok |", "|---|---|---|---|"]
    lines += [f"| {x['depth']} | {x['current']} | {x['target']} | {x['ok']} |" for x in g["G2"]]
    lines += [f"| {d} (past today's pool) | - | {v} | reported |" for d, v in sorted(past.items())]
    lines += ["", "## G3 placement (decode tok/s of A)", "| A | case | current | target | ada | ada-vec | ok |",
              "|---|---|---|---|---|---|---|"]
    lines += [f"| {x['A']} | {x['case']} | {x['current']} | {x['target']} | {x['ada']} | {x['ada_vec']} | {x['ok']} |"
              for x in g["G3"]]
    lines += ["", "## G4 VRAM (min free MiB)"] + [f"- {n}: {v['min_free_mib']} ({'ok' if v['ok'] else 'UNDER 600'})"
                                                  for n, v in g["G4"].items()]
    lines += ["", "## MTP head", "```", json.dumps(g["head"], indent=1), "```",
              "", "## reported (window, spill, attribution arms)", "```",
              json.dumps(g["report"], indent=1, default=str), "```"]
    open(os.path.join(out, "gates.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return g


# --------------------------------------------------------------- deploy ----
def restart_stack() -> None:
    """scripts/watchdog.ps1 Restart-Service, Kind 'task': the whole stack."""
    ps = ("Get-Process llama-swap, llama-server -ErrorAction SilentlyContinue | Stop-Process -Force "
          "-ErrorAction SilentlyContinue; "
          "Get-CimInstance Win32_Process -Filter \"Name='wscript.exe'\" -ErrorAction SilentlyContinue | "
          "Where-Object { $_.CommandLine -like '*start-stack-hidden*' } | ForEach-Object { Stop-Process -Id "
          "$_.ProcessId -Force -ErrorAction SilentlyContinue }; "
          "try { Stop-ScheduledTask -TaskName 'llama-stack' -ErrorAction SilentlyContinue } catch {}; "
          "Start-Sleep -Seconds 6; Start-ScheduledTask -TaskName 'llama-stack'")
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=False, timeout=120)
    t0 = time.time()
    while time.time() - t0 < 900:
        if ec.http("GET", f"{ec.SWAP}/health", timeout=5)[0] == 200 and \
                ec.http("GET", f"{ec.PROXY}/health", timeout=5)[0] == 200:
            return
        time.sleep(5)
    raise RuntimeError("the stack did not come back within 900 s")


def edit_config(text: str, ada: str, args: dict, lean: str) -> str:
    """config.yaml with `server_nudge` -> the new engine and, inside `bonsai`'s block only, each flag of
    `args` set: the value after an existing flag replaced (a model path as ${models}/<file>), a missing flag
    added on its own line after `--spec-draft-n-max`. Comment lines are never touched."""
    lines = text.split("\n")
    n = [i for i, ln in enumerate(lines) if ln.lstrip().startswith("server_nudge:")]
    if len(n) != 1:
        raise RuntimeError("config.yaml: expected exactly one `server_nudge:` macro")
    indent = lines[n[0]][:len(lines[n[0]]) - len(lines[n[0]].lstrip())]
    lines[n[0]] = f'{indent}server_nudge: "{ada}"'
    start = [i for i, ln in enumerate(lines) if ln.rstrip() in (f"  {ec.PROD_ID}:", f'  "{ec.PROD_ID}":')]
    if len(start) != 1:
        raise RuntimeError(f"config.yaml: expected one `  {ec.PROD_ID}:` model block")
    end = next((j for j in range(start[0] + 1, len(lines))
                if lines[j].startswith("  ") and not lines[j].startswith("   ")
                and lines[j].strip() and not lines[j].lstrip().startswith("#")), len(lines))
    added = []
    for flag, value in args.items():
        if flag == "-m":
            value = "${models}/" + os.path.basename(value)
        hit = None
        for j in range(start[0], end):
            toks = lines[j].split()
            if toks and not toks[0].startswith("#") and flag in toks:
                hit = j
                break
        if hit is None:
            added.append(f"{flag} {value}")
            continue
        toks = lines[hit].split()
        k = toks.index(flag)
        toks[k + 1] = value
        pad = lines[hit][:len(lines[hit]) - len(lines[hit].lstrip())]
        lines[hit] = pad + " ".join(toks)
    if added:
        anchor = next((j for j in range(start[0], end) if "--spec-draft-n-max" in lines[j].split()), None)
        if anchor is None:
            raise RuntimeError("config.yaml: no `--spec-draft-n-max` line in bonsai's cmd to add flags after")
        pad = lines[anchor][:len(lines[anchor]) - len(lines[anchor].lstrip())]
        lines[anchor + 1:anchor + 1] = (
            [pad + "# 2026-09-27 (operator): the bonsai-ada-surgery target -- q8_0 K/V and draft K/V, "
                   "the trained window with the tiered KV cache (docs/ENGINES.md, bench/engine_phase2.py)"]
            + [pad + x for x in added])
    return "\n".join(lines)


def deploy(s: dict, g: dict, out: str, key_file: str) -> int:
    import build_engine as be
    import verify_artifacts as va
    cfg = os.path.join(ROOT, "config.yaml")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = f"{cfg}.bak-{stamp}"
    shutil.copyfile(cfg, backup)
    before = {str(p) for p in va.verify(cfg) if p.severity == "error"}
    text = open(cfg, encoding="utf-8", newline="").read()
    choice = g["head"]["choice"]
    n = (g["fit"].get(choice) or {}).get("n")
    args = target_args(n, s["pc"] if choice == "pc" else None)
    new = edit_config(text, s["ada"], args, s["lean"])
    open(cfg, "w", encoding="utf-8", newline="").write(new)
    rec = {"backup": backup, "server_nudge": s["ada"], "args": args, "head": choice, "started": stamp}

    def rollback(why: str) -> int:
        log(f"DEPLOY NOT GOOD: {why}; rolling config.yaml back to {backup}")
        shutil.copyfile(backup, cfg)
        rec.update(verdict="rolled back", why=why)
        try:
            restart_stack()
            rec["rollback_restart"] = "ok"
        except Exception as e:                                       # noqa: BLE001
            rec["rollback_restart"] = repr(e)
        json.dump(rec, open(os.path.join(out, "deploy.json"), "w", encoding="utf-8"), indent=1)
        return 1
    argv, _ = ec.production_command()
    want = ec.arm_command(s["prod_argv"], s["ada"], "${PORT}", args)
    if sorted(argv) != sorted(want):
        return rollback(f"config.yaml's new argv is not the gated arm's: {argv} vs {want}")
    problems, _ = be.verify_deploy()
    if problems:
        return rollback("engines: " + "; ".join(problems))
    errs = [str(p) for p in va.verify(cfg) if p.severity == "error" and str(p) not in before]
    if errs:
        return rollback("models: " + "; ".join(errs))
    try:
        restart_stack()
    except Exception as e:                                           # noqa: BLE001
        return rollback(repr(e))
    r = subprocess.run([PY, os.path.join(ROOT, "scripts", "deploy_check.py"), "--key-file", key_file],
                       cwd=ROOT)
    rec["deploy_check_exit"] = r.returncode
    if r.returncode != 0:
        return rollback(f"deploy_check.py exited {r.returncode} (3 = a 429: run it again on an idle stack)")
    rec["verdict"] = "DEPLOYED"
    json.dump(rec, open(os.path.join(out, "deploy.json"), "w", encoding="utf-8"), indent=1)
    log(f"DEPLOYED: {args}. Record it: engines/manifest.yaml (llama-bonsai2-ada config_refs [bonsai], the "
        "shipped comment; llama-bonsai2 -> previous), models/manifest.yaml (the head's status in_service if "
        "it won), docs/ENGINES.md.")
    return 0


# --------------------------------------------------------- record graft ----
TRUNK_SHA = "94dd53cbad55db5a515f245887c9f0317484502a451ccc0424c7d7788b52900a"   # bonsai-2-27b-abliterated
HEAD_SHA = "c047329010e6e614e2a997ecc57f1ce8e289937145d2ff4fcfd2df6faea2e5ba"    # procreations-mtp-head


def record_graft(strip_check: str) -> int:
    """After the operator ran merge.py (docs/ENGINES.md "The ProCreations MTP
    head"): check the strip proof and the head, then add the graft to
    models/manifest.yaml as a candidate artifact with its recipe."""
    import hashlib
    import verify_artifacts as va

    def sha(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(1 << 24), b""):
                h.update(b)
        return h.hexdigest()
    s = setup()
    models = os.path.dirname(s["lean"])
    graft = os.path.join(models, PC_FILE)
    head = os.path.join(models, "procreations-mtp", "head-procreations.gguf")
    if not os.path.exists(graft):
        print(f"no graft at {graft}")
        return 1
    got = sha(strip_check)
    if got != TRUNK_SHA:
        print(f"STRIP CHECK FAILED: {strip_check} is {got}, the trunk is {TRUNK_SHA}; not recorded")
        return 1
    if sha(head) != HEAD_SHA:
        print(f"{head} is not the recorded head; not recorded")
        return 1
    if pc_recorded(graft):
        print("already recorded")
        return 0
    size, digest = os.path.getsize(graft), sha(graft)
    entry = f"""
  - id: bonsai-2-27b-abliterated-mtp-procreations
    role: "CANDIDATE main model: the abliterated trunk with ProCreations' on-policy MTP head (r3-mtp) grafted on as block 64; acceptance on this trunk is measured by bench/engine_phase2.py, not assumed"
    status: candidate
    path: "${{models}}/{PC_FILE}"
    size: {size}
    sha256: {digest}
    provenance:
      status: recipe
      recipe:
        tool: {{repo: "https://github.com/sudoingX/bonsai2-small-gpu", commit: eb52d9d7363cda2d910146f4e37f4b8c64c30c46, path: graft/tools, version: "bonsai2-mtp merge 1"}}
        interpreter: stack-python
        inputs:
          - {{id: bonsai-2-27b-abliterated, sha256: {TRUNK_SHA}, as: trunk}}
          - {{id: procreations-mtp-head, sha256: {HEAD_SHA}, as: head}}
        command: |
          python graft/tools/merge.py Ternary-Bonsai-2-27B-Abliterated-PTQ1_0.gguf procreations-mtp/head-procreations.gguf {PC_FILE}
        check: "merge.py --strip on the output reproduces the trunk byte for byte (sha256 {TRUNK_SHA}, checked {time.strftime('%Y-%m-%d')} by bench/engine_phase2.py --record-graft)"
        doc: docs/ENGINES.md
      licence: apache-2.0
"""
    path = va.MANIFEST
    text = open(path, encoding="utf-8", newline="").read()
    anchor = "\n  - id: bonsai-2-27b-mmproj-q8"
    if text.count(anchor) != 1:
        print("models/manifest.yaml: anchor not found; add the entry by hand:" + entry)
        return 1
    text = text.replace(anchor, entry.rstrip("\n") + "\n" + anchor)
    open(path, "w", encoding="utf-8", newline="").write(text)
    probs = [str(p) for p in va.schema_problems(va.load_manifest())]
    print("recorded" if not probs else "recorded, but the schema check says: " + "; ".join(probs))
    return 0 if not probs else 1


# ----------------------------------------------------------------- main ----
def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--window", action="store_true", help="the operator has said the GPU is free")
    ap.add_argument("--out")
    ap.add_argument("--gates", metavar="DIR", help="only (re)compute the gates of a finished run")
    ap.add_argument("--deploy", action="store_true", help="deploy the target when every gate passes")
    ap.add_argument("--key-file", help="the account key for scripts/deploy_check.py (needed with --deploy)")
    ap.add_argument("--port", type=int, default=18091)
    ap.add_argument("--tolerance", type=float, default=TOLERANCE)
    ap.add_argument("--skip", nargs="*", default=[], choices=["fit", "corruption", "speed"])
    ap.add_argument("--arms", nargs="*", help="run only these speed arms (default: all)")
    ap.add_argument("--attempts", type=int, default=4)
    ap.add_argument("--wait-quiet", type=float, default=60.0, help="minutes to wait for a quiet stack per arm")
    ap.add_argument("--record-graft", metavar="STRIP_CHECK_GGUF",
                    help="check merge.py --strip's output against the trunk and record the graft")
    a = ap.parse_args(argv)
    if a.record_graft:
        return record_graft(a.record_graft)
    if a.gates:
        g = gates(a.gates, a.tolerance, setup())
        return 0 if g["all_pass"] else 1
    if not a.window or not a.out:
        print("refusing: needs --window (the operator's 'GPU free') and --out")
        return 2
    if a.deploy and not a.key_file:
        print("--deploy needs --key-file")
        return 2
    s = setup()
    if not s["ada"] or not os.path.exists(s["ada"]):
        print(f"{ENGINE} has no shipped binary on disk: {s['ada']}")
        return 2
    s["port"] = a.port
    s["arms"] = {}
    for d in ("corruption", "speed", "fit"):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    json.dump({k: v for k, v in s.items() if k not in ("prod_env", "arms")},
              open(os.path.join(a.out, "setup.json"), "w", encoding="utf-8"), indent=1)
    heads = {"lean": None}
    if s["pc"] and pc_recorded(s["pc"]):
        heads["pc"] = s["pc"]
    # step 0: the VRAM line for each head
    fits = {}
    for h, model in heads.items():
        f = load(os.path.join(a.out, f"fit-{h}.json")) if "fit" in a.skip else None
        fits[h] = f or fit(h, model, s, a.out, a.attempts, a.wait_quiet)
        if not fits[h].get("n"):
            log(f"fit {h}: no N fits with the margin ({fits[h]}); its target arms are skipped")
    targets = {f"target-{h}": target_args(fits[h]["n"], heads[h]) for h in heads if fits[h].get("n")}
    # step 1: corruption
    vec = "GGML_CUDA_FA_MMA_DECODE_MIN_KV=0"
    if "corruption" not in a.skip:
        arms = [f"current={s['current']}", f"ada={s['ada']}", f"ada-vec={s['ada']},{vec}"]
        for name, targs in targets.items():
            arms.append(f"{name}={s['ada']}," + ",".join(f"ARG:{k}={v}" for k, v in targs.items()))
        cmd = [PY, os.path.join(HERE, "engine_corruption.py"), "--window", "--out",
               os.path.join(a.out, "corruption"), "--port", str(a.port), "--attempts", str(a.attempts),
               "--wait-quiet", str(a.wait_quiet)]
        for arm in arms:
            cmd += ["--arm", arm]
        log("step 1: " + " ".join(cmd))
        subprocess.run(cmd, cwd=ROOT)
    # step 2: speed
    if "speed" not in a.skip:
        W = {"--spec-draft-window": "16384"}
        s["arms"].update({
            "current": (s["current"], {}, {}),
            "ada": (s["ada"], {}, {}),
            "ada-vec": (s["ada"], {"GGML_CUDA_FA_MMA_DECODE_MIN_KV": "0"}, {}),
            "ada-draft2": (s["ada"], {}, {"--spec-draft-n-max": "2"}),
            "ada-dkv-q8": (s["ada"], {}, {"--cache-type-k-draft": "q8_0", "--cache-type-v-draft": "q8_0"}),
            "ada-np1": (s["ada"], {}, {"-np": "1"}),
            "ada-np1-window": (s["ada"], {}, {"-np": "1", **W}),
        })
        plan = [("current", speed_fn(True, True, True, s["pool"], concurrent=True)),
                ("ada", speed_fn(True, True, True, s["pool"], concurrent=True)),
                ("ada-vec", speed_fn(False, True, False, s["pool"]))]
        for name, targs in targets.items():
            s["arms"][name] = (s["ada"], {}, targs)
            plan.append((name, speed_fn(TARGET_DEPTHS, True, True, TARGET_CTX, concurrent=True, spill=True)))
        if targets:
            h = "pc" if "target-pc" in targets else "lean"
            s["arms"]["target-window"] = (s["ada"], {}, {**targets[f"target-{h}"], **W})
            plan.append(("target-window", speed_fn(TARGET_DEPTHS, True, True, TARGET_CTX, concurrent=True,
                                                   spill=True)))
        plan += [("ada-draft2", speed_fn(True, False, True, s["pool"])),
                 ("ada-dkv-q8", speed_fn(True, False, True, s["pool"])),
                 ("ada-np1", speed_fn(True, False, True, s["pool"])),
                 ("ada-np1-window", speed_fn(True, False, True, s["pool"]))]
        if os.path.exists(ADA_0031):
            s["arms"]["ada-window-0031"] = (ADA_0031, {}, W)
            plan.append(("ada-window-0031", speed_fn(False, False, False, s["pool"], concurrent=True)))
        if a.arms:
            plan = [(n, f) for n, f in plan if n in a.arms]
        for name, fn in plan:
            log(f"step 2: speed arm {name}")
            rec, res = in_window(name, s, fn, os.path.join(a.out, "speed"), a.attempts, a.wait_quiet)
            if res and res.get("proxy_health") != 200:
                log("stopping: :1234 is not healthy after restore")
                return 1
    g = gates(a.out, a.tolerance, s)
    if not a.deploy:
        log("gates written; no --deploy, nothing changed")
        return 0 if g["all_pass"] else 1
    if not g["all_pass"]:
        log("NOT DEPLOYING: a gate failed (gates.md)")
        return 1
    return deploy(s, g, a.out, a.key_file)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
