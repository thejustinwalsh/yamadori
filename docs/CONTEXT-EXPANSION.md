# Context expansion: q4_0 K/V + K mean-centering on the 5060 Ti

Status 2026-09-23: **staged, nothing run.** The live stack was not restarted,
nothing was loaded, and no request went to any port. What exists:

| piece | where |
|---|---|
| trial entries `bonsai-q4kv` (-c 262144) and `bonsai-q4kv-196k` (-c 196608), group `context-trial`, profiles `context-trial-262k` / `-196k` | `config.yaml` (`bonsai` byte-identical; `llama-swap -validate` passes) |
| VRAM arithmetic | `bench/kv_context/vram.py` |
| gated experiment | `bench/kv_context/run.py` |
| gates, applied from the manifest | `bench/kv_context/analyse.py` |
| synthetic codebase for the needle test | `bench/kv_context/synth_code.py` |
| bias calibration corpus | `bench/kv_context/make_calib_corpus.py` |
| offline tests (57 checks) | `bench/kv_context/test_kv_context.py` |

This is HANDOFF run-queue item 4 ("KV-cache quantisation, max out the KV"),
gated as that item asks.

## 1. Where the pool stands, measured

The shipped pool came down 262,144 (advertised) -> 208k -> 196,608 (backed
out) -> 147,456 -> 163,840 (config.yaml `bonsai`). Two of the reasons behind
those cuts no longer hold. `docs/CONSTRAINTS.md` 1c found no OOM in any
`arc193_a` row. config.yaml itself says the 3 GB floor "was never measured".

**Today's server is already under every floor except the dashboard's.** Read
with `nvidia-smi` on 2026-09-23 at about 22:30. The only process on the 5060 Ti
was the live `bonsai` (-c 163840, q8_0/q8_0, MTP n-max 1, 4 slots). It used
**14,340 MiB, with 1,711 free**, at 78% utilisation. This is **n = 1**.
`mcp/vitals.py` independently says the build "idles at ~1.7 GB free". That is
why `TIGHT_MIB` was lowered to 1,280.

The per-conversation window today is the main share, 5/8 of 163,840 =
**102,400** (`mcp/budget.py`).

## 2. External evidence, checked at the source

| claim | verified | source |
|---|---|---|
| 262,144 in 11.7 GB with q4_0 K/V | yes. `12gb.sh` runs `-c 262144 -np 1 -ctk q4_0 -ctv q4_0`. The sweep measured 11,698 MiB (prebuilt) and 11,682 (patched) with no MTP | sudoingX/bonsai2-small-gpu @ `eb52d9d7`, `serve/12gb.sh`, `sweeps/rtx3060.md` |
| 131,072 with MTP in 10.6 GB | yes. `12gb-mtp.sh` runs `-c 131072 --spec-type draft-mtp --spec-draft-n-max 1 -np 1 -ctk q4_0 -ctv q4_0`. It measured 10,638 MiB with the full MTP file and 9,956 with the lean file. The lean file also loaded at 196,608 (11,976 MiB) | same, `serve/12gb-mtp.sh`, `sweeps/rtx3060.md` |
| `--kv-mean-center` is a "hard requirement" for q4_0 K | the recipe says so ("recovers q4_0 K-cache accuracy; hard requirement"), but the same repo **did not measure quality**: "we did not run perplexity or KLD comparisons" | snailium/bonsai2-8gb @ `51cef900`, `RECIPE.md`, `evidence/results/kv-cache-options.md` |
| bias made by a script | yes. `scripts/make-kv-bias.sh` calls `llama-kv-mean-center -fa on -ctk q4_0`. Its default corpus is 8 sentences repeated at random | same, `scripts/make-kv-bias.sh` |
| our build supports `--kv-mean-center FNAME` and requires q4_0 K | yes. `common/arg.cpp:2452`, and `llama-context.cpp:3905` errors unless `type_k == Q4_0`. Commits `7fb3d5bf` and `36d1deef` (hybrid-model support) are ancestors of HEAD `285542d9`. The string is in the built `llama-common.dll` | `llamacpp-sudoingx-bonsai2`, `docs/kv-mean-center.md` |
| the tool that makes the bias is built | **no.** `build/bin` has only `llama-server` and `llama-bench` (MTP-STAGING s.2 built only those targets). The `llama-kv-mean-center` target is configured in `build.ninja` | same |

Quality evidence, stated plainly. The fork's measured gain is on "a
hybrid-attention model", not this one. Logit KLD against f16 over 12 x 512
tokens was 0.00144 with rotation alone and 0.00111 with rotation plus a
matched-basis bias (`tools/kv-mean-center/README.md`). The feature doc says the
repo "does not ship a measured number for a specific trained model". **Nobody
has measured q4_0 K/V quality on Bonsai 2.** That is what this trial is for.

The speed risk comes from `sweeps/rtx3060ti-8gb.md`. An RTX 3060 Ti with q4_0
K/V and short prompts decoded at 45.7 tok/s in a 65,536 window and 17.5 at
114,688. The fall is binary-independent and the cause is unidentified. The
12 GB 3060 did not show it: 25.0 at 131,072 against 26.3 flat. Whether the
5060 Ti does is unknown. The SPEED phase measures a short prompt in each arm's
full allocated window for exactly this reason.

## 3. The arithmetic

Geometry comes from the GGUF header: `head_count_kv 4`, `key_length 256`,
`full_attention_interval 4`, so 16 attention layers. Per token of `-c`:

| term | KiB/token | source |
|---|---|---|
| main KV, q8_0/q8_0 | 34.0 | measured, MTP-STAGING s.9 (544.00 MiB @ 16,384) |
| main KV, q4_0/q4_0 | 18.0 | same geometry at 18 B/32 values. Agrees with the 3060's no-MTP q4 slope of 23.0 = 18 + 5 |
| main compute buffer growth | ~5 | MTP-STAGING s.9, 16K -> 24K, extrapolated |
| MTP draft KV (f16) + draft compute | ~5 | MTP-STAGING s.9 |
| **slope, q8_0 + MTP** | **44** | the same figure `mcp/budget.py` uses |
| **slope, q4_0 + MTP** | **28** (component) .. **31.6** (conservative) | 31.6 is the only measured MTP-on q4 slope (3060, lean, np 1) |

The table is anchored on the live reading: switch the KV type at 163,840,
then move `-c` by the slope. 194 MiB is subtracted for peak, the config.yaml
stress figure. `python bench/kv_context/vram.py` prints it.

| -c | K/V | slots | used MiB | free idle | free peak |
|---:|---|---:|---:|---:|---:|
| 163,840 (today, measured) | q8/q8 | 4 | 14,340 | 1,711 | 1,517 |
| 163,840 | q4/q4 | 4 | 11,780 | 4,271 | 4,077 |
| **196,608** | q4/q4 | 4 | 12,676..12,791 | **3,260..3,375** | **3,066..3,181** |
| 229,376 | q4/q4 | 4 | 13,572..13,802 | 2,249..2,479 | 2,055..2,285 |
| **262,144** | q4/q4 | 4 | 14,468..14,814 | **1,237..1,583** | **1,043..1,389** |
| 262,144 | q4/q4 | 3 | 14,169..14,514 | 1,537..1,882 | 1,343..1,688 |
| 262,144 | q4/q4 | 2 | 13,870..14,215 | 1,836..2,182 | 1,642..1,988 |
| 262,144 | q4/**q8** | 4 | 16,516..16,862 | **does not fit** | |

Largest `-c` whose projected **peak** clears each floor (q4/q4, 4 slots,
conservative .. component slope):

| floor | MiB | largest -c |
|---|---:|---|
| documented "3 GB" (config.yaml, KNOWN-ISSUES) | 3,072 | 188,416 .. 196,608. 196,608 sits on the line, 6 MiB under on the conservative slope |
| 2.0 GB rule (config.yaml, 2026-09-22) | 2,048 | 229,376 .. 237,568 |
| dashboard red, `TIGHT_MIB` | 1,280 | 253,952 .. 262,144 |
| HANDOFF item 4, ">= 1 GB free at peak" | 1,024 | 262,144 |

Cross-check (not an input): MTP-STAGING s.9's projection for today's config
was 14,601 MiB. The live reading is 14,340.

**Why V is q4_0.** This build has `GGML_CUDA_FA_ALL_QUANTS=OFF` (CMakeCache).
With that setting, `ggml-cuda/fattn.cu:442` returns `BEST_FATTN_KERNEL_NONE`
whenever K and V types differ, so mixed K/V has no CUDA flash-attention kernel
here. It also costs 26 KiB/token against 18: at 262,144 that is over the card.
On an RTX 5060 (sm_120, n = 1 run each), snailium measured q4_0/q8_0 at 58.71
tok/s against 65.95 for q4_0/q4_0. The fork rotates quantized V (Hadamard,
head dim 256 % 64 == 0, `llama-kv-cache.cpp:335`), which is its V mitigation.
Mean-centering is K-only. The draft cache stays f16: MTP-STAGING measured a
q8_0 draft cache as +33 MiB **more** in total.

## 4. The config entry

`bonsai-q4kv` is `bonsai` with exactly three changes. `test_kv_context.py`
asserts that no other flag differs.

```yaml
      -c 262144                                   # bonsai: 163840
      --cache-type-k q4_0 --cache-type-v q4_0     # bonsai: q8_0 / q8_0
      --kv-mean-center ${models}/Ternary-Bonsai-2-27B-Abliterated-PTQ1_0-mtp-lean.kv-mean-center-q4_0.gguf
```

Everything else is identical to `bonsai`: binary `${server_mtp}`, weights,
`--spec-type draft-mtp --spec-draft-n-max 1`, `-dev CUDA0 -ngl 999`, `-fa on
--cont-batching -b 1024 -ub 512 --jinja`, the reasoning flags,
`--no-context-shift`, `${sampling}`, and the env
`CUDA_VISIBLE_DEVICES=GPU-de660e90-...` plus `GGML_CUDA_BATCH_INVARIANT=1`.
It adds `unlisted: true` and `ttl: 1800`. The ttl frees the card if a trial is
forgotten, and with it the watchdog treats the entry as on-demand: it unloads
it alone and never restarts the stack. `bonsai-q4kv-196k` is the same entry at
`-c 196608`.

How it is reached:

- **Group `context-trial`** (`swap: true, exclusive: false`). An ungrouped model
  would fall into llama-swap's `(default)` group, which is swap *and*
  exclusive (`internal/config/config.go`). Loading a trial would then evict
  retrieval and image generation on the other card.
- **Profiles `context-trial-262k` / `-196k`** pin `bonsai` and `bonsai-agent`
  to a trial entry. The rewrite applies to `/v1/...` and `/upstream/<id>/...`
  (llama-swap v256 `internal/server/profiles.go`). With one active, the proxy,
  deep thinking and the worker all keep asking for `bonsai` and get the
  trial. No profile is active at startup or after a reload.

Consequences to know:

- **The running llama-swap does not know these entries yet.** It was started
  at 17:24 without `-watch-config`. They exist only after the next stack
  restart, which `run.py`'s preflight checks.
- llama-swap assigns `${PORT}` in sorted model-id order. `bonsai` keeps 10001.
  The other models shift by two (embeddings 10004 -> 10006, reranker
  10007 -> 10009). Nothing in the repo hardcodes those ports.
- `config.template.yaml` (the tracked copy) is not updated. config.yaml is
  gitignored.

## 5. Making the bias file (not run: it needs a GPU)

**What it is.** One F32 tensor per attention layer (16 x 1,024 values, about
64 KiB) holding the mean K vector per (KV head, channel). It is measured
*after* the fork's Hadamard K rotation, and the loader refuses a file whose
recorded basis (`kv_mean_center.k_rot`) does not match the server's. The
rotation is on for any quantized K when head dim % 64 == 0. So it **must be
calibrated with `-ctk q4_0`**.

**1. Build the tool** (CPU only, about a minute; the build dir is already
configured):

```bat
call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
set "PATH=C:\Users\jwals\textgen\installer_files\cudabuild\Library\bin;%PATH%"
"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" ^
  --build C:\Users\jwals\llamacpp-sudoingx-bonsai2\build --target llama-kv-mean-center -j 12
```

**2. Build the corpus** (offline, about 460 KB and 130k tokens; record the
printed sha256):

```bat
C:\Users\jwals\textgen\installer_files\env\python.exe bench\kv_context\make_calib_corpus.py ^
  --out C:\Users\jwals\textgen\user_data\models\kv-calib-corpus.txt
```

**3. Calibrate** against the served file itself:

```bat
set CUDA_DEVICE_ORDER=PCI_BUS_ID
rem EITHER the A4000, beside retrieval + Laya (7.6 GB resident, 8.6 free; needs ~6.2 GB):
set CUDA_VISIBLE_DEVICES=GPU-43e37d0c-4104-9056-2552-6109d4d3382c
rem OR the 5060 Ti, only inside the window with `bonsai` unloaded:
rem set CUDA_VISIBLE_DEVICES=GPU-de660e90-0e9c-d465-b389-6df63021b920
C:\Users\jwals\llamacpp-sudoingx-bonsai2\build\bin\llama-kv-mean-center.exe ^
  -m C:/Users/jwals/textgen/user_data/models/Ternary-Bonsai-2-27B-Abliterated-PTQ1_0-mtp-lean.gguf ^
  -f C:/Users/jwals/textgen/user_data/models/kv-calib-corpus.txt ^
  -o C:/Users/jwals/textgen/user_data/models/Ternary-Bonsai-2-27B-Abliterated-PTQ1_0-mtp-lean.kv-mean-center-q4_0.gguf ^
  -dev CUDA0 -ngl 999 -fa on -ctk q4_0 -ctv q4_0 -c 512 --chunks 250
```

- **Output.** The `-o` path above, next to the weights, which is where
  config.yaml points. Without `-o` the tool writes `kv-mean-center.gguf` in the
  current directory.
- **GPU.** Either card: the bias is a mean, and the rotation is software, so
  it is the same basis on sm_86 and sm_120. It needs the weights (5,730 MiB)
  plus a tiny KV and compute buffer at `-c 512`, about 6.2 GB. The **A4000**
  holds it now without touching the main model. Run it while image generation
  and `bonsai-vision` are unloaded (imagegen peaks +6.4 GB), and treat it as the
  one GPU consumer of that card while it runs. The 5060 Ti has 1.7 GB free, so
  it can be used only in the window.
- **Time.** Estimated, not measured. 250 chunks x 512 is 128k tokens of
  prefill with a per-node eval callback, like `llama-imatrix`. At the 5060 Ti's
  ~490 tok/s short-prompt prefill (KNOWN-ISSUES) that is about 5 minutes. Allow
  up to 15 on the A4000 with callback overhead, plus about 20 s to load.
- **Check the log** for `wrote K-cache mean-centering bias for 16 layer(s) ...
  (measured with K rotation active)`. 16 layers and "active" are both
  required. Anything else means the basis or the architecture hook is wrong.

## 6. The experiment

Everything goes through **:1234**. The proxy cannot route to another upstream
model per request: `catalog.resolve` maps every public name to `bonsai`, and
`budget.pool_size` reads only :10001. So the trial is selected one layer down,
by the llama-swap profile pin. **No proxy change is needed.** A per-request
alternative would need `catalog.INTERNAL["yamadori-q4kv"] =
("bonsai-q4kv", None)` plus a per-model pool in `budget.py`. It would still
swap the card on every switch, because the two cannot co-reside, so it buys
nothing.

The control plane is llama-swap's API on :11434, which the proxy does not
expose: unload, load via `/upstream/bonsai/health`, set the profile,
`/running`, and erasing idle slot caches before each long haystack so one
prompt can use the unified pool.

`python bench/kv_context/run.py --plan` prints all of this and sends nothing.

| phase | what | through |
|---|---|---|
| FIT | For each candidate (262,144, then 196,608): unload `bonsai`, wait for an empty card, activate the profile, load and time it. Read VRAM at idle (median of 5). Then fill the pool with a ~55% prompt followed by a ~30% prompt, polling VRAM every second. **Below the floor at idle or at peak: unloaded at once, FAIL, and no later phase uses it.** A guard unloads any trial below 512 MiB free at any moment. | profile + :1234 |
| SPEED | Per arm (q8 = production, q4 = the largest passing candidate): a ~2k and a ~98k prompt, 3 reps, a cold `code` request and a cached `prose` request. Thinking off (tier `minimal`). Wall clock only: the proxy drops llama-server's timings, so MTP acceptance is not visible. | :1234 |
| ACCURACY | A 4-needle + 4-decoy probe (`bench/longctx/haystack.py`: single, multi, reason) in a **synthetic TypeScript codebase** (`synth_code.py`). No model has read it, and it carries more than 1,000 needle-shaped distractors (`export function x() { return NNNN; }`). 12 items, same seed in both arms. Shared rungs 8k, 32k, 64k, 96k, 128k and 148k (151,552) fit q8's pool. q4-only rungs go up to 244k (249,856) at 262,144, or 176k (180,224) at 196,608. Thinking off, every feature forced off. | :1234 |
| QUALITY | The 21 LiveBench coding questions of `lb-20260923-minp0`, bare @ medium with `check_code`/`repair` forced off (the cached answers predate check_code). A **fresh q8 control arm** runs first, then the q4 arm, via `bench/livebench/run_arm.sh` in WSL and `score.py`. | :1234 |

**Before the window, three things must exist:**

1. **The bias file** (s.5).
2. **A llama-swap that knows the trial.** Restart the stack at the start of
   the window.
3. **Two LiveBench arms.** Answers are stored per display name, so a second
   `bonsai` run would find the cached answers and skip every question. Add
   these to `bench/livebench/drive.py` `ARMS` (not done here, because
   drive.py is outside this change). `run.py` prints the same snippet and
   skips QUALITY if they are missing.

```python
    "kv-q8": {"display": "yamadori-kv-q8-arm",
              "features": {"retrieval": False, "hints": False, "investigate": False, "fanout": 1,
                           "effort": "medium", "check_code": False, "repair": False}},
    "kv-q4": {"display": "yamadori-kv-q4-arm",
              "features": {"retrieval": False, "hints": False, "investigate": False, "fanout": 1,
                           "effort": "medium", "check_code": False, "repair": False}},
```

**Run** (in the window, with the stack interpreter):

```bat
C:\Users\jwals\textgen\installer_files\env\python.exe bench\kv_context\run.py --plan
C:\Users\jwals\textgen\installer_files\env\python.exe bench\kv_context\run.py ^
  --window --key-file C:\path\to\key --run-id kvq4-20260924 --floor-mib 3072 --wait-idle 60
C:\Users\jwals\textgen\installer_files\env\python.exe bench\kv_context\analyse.py bench\kv_context\results\kvq4-20260924
```

**Or queue it.** It waits for an idle card: no other gpu-lane pause, no
benchmark process, utilisation under 30%, no `bonsai` slot processing.

```bat
C:\Users\jwals\textgen\installer_files\env\python.exe bench\queue_runner.py add kv_context/run.py --needs model,proxy ^
  --args "--window --key-file C:/path/to/key --run-id kvq4-20260924 --floor-mib 3072 --wait-idle 720"
```

It was **not** added to `bench/queue.jsonl`. Queued, it would take the
production model down as soon as the card is idle, and that is the operator's
call. If it never finds an idle card within `--wait-idle`, or preflight
refuses, it exits 3. `queue_runner.py` records that as `failed` and does not
retry it, so add it again.

**How it behaves while running:**

- **Timing.** At an *assumed* 300 tok/s prefill it takes about 23 h at 262,144
  and 17 h at 196,608. Most of that is q4 accuracy prefill.
- **Resuming.** `--max-minutes` stops it cleanly. Re-running the same
  `--run-id` resumes, and a stack error is never counted as done. The floor is
  fixed in `manifest.json` at the first start, and a resume with a different
  floor is refused.
- **Restoring production.** The `finally` block always turns the profile off,
  unloads the trial and reloads `bonsai`, then checks it is `-c 163840` q8_0.
  `run.py --restore` does only that. By hand:
  `PUT :11434/api/profiles/active {"name": null}`, then
  `POST :11434/api/models/unload/bonsai-q4kv`, then
  `GET :11434/upstream/bonsai/health`.
- **Budget while the trial runs.** The proxy caches its pool (163,840) at
  first use, so both arms are budgeted identically, which is what the pairing
  wants. `mcp/vitals.py` refreshes it from :10001 when the dashboard polls.
  While a trial holds the card :10001 is down, and the fallback is 131,072.
  Keep the dashboard closed during the q4 arms. Every answer records its
  `x_yamadori.budget`, so a difference is visible.

## 7. Gates, pre-registered

These are `run.py` `GATES`, copied into `manifest.json` before the first
request. `analyse.py` reads them from there.

| gate | passes when | n and power |
|---|---|---|
| G1 fit | Loads; free VRAM at idle **and** at the stress peak >= `floor_mib` (default 3,072; the operator fixes it before the run); no stress request fails; the guard never trips | one load per candidate. A reading, not a sample |
| G2 speed | q4/q8 median ratio >= **0.85** for decode at ~2k, decode at ~98k, and cold prefill at ~98k | 3 reps per cell. 0.85 is a chosen tolerance, not a measurement. It catches a 3060 Ti-style window collapse (-60%) |
| G3a accuracy vs q8 | On the shared rungs, pooled over (L, item, task): **no significant paired drop** (exact McNemar, two-sided, alpha 0.05) **and** q4 accuracy >= q8 - **5 points** | 216 pairs (6 rungs x 12 items x 3 tasks). A single cell (n = 12) can only show a 6-0 collapse |
| G3b accuracy beyond q8 | No q4-only rung significantly below q4's own 8k (exact McNemar, Bonferroni over those rungs). The largest clean rung is the **usable context** | 36 pairs per rung. At 262,144 (m = 3, alpha 0.0167) about 7 net flips (~19 points) are needed. This is HANDOFF item 4's rule |
| G4 quality | LiveBench coding, q4 vs the fresh q8 control on the same questions: **no significant paired drop** (exact McNemar, alpha 0.05) **and** q4 total >= q8 total - **2 questions**. Cached `bonsai` answers (15/21) are reported alongside, not gated | **n = 21. Detects a collapse only** (6-0, 7-0, 8-1). It cannot show "not worse" at any useful margin |

The verdict is **ADOPT** only if all five pass. Any fail means stay at q8_0
163,840, or re-run at the smaller candidate. Any gate with no data gives
INCOMPLETE, never a pass.

## 8. If it is adopted

- **config.yaml.** Change `bonsai`'s own `-c`, `--cache-type-k/-v` and add
  `--kv-mean-center`. Adopt by editing `bonsai`, not by leaving a profile
  active: a profile does not survive a restart, and `budget.pool_size` reads
  :10001, which is `bonsai`'s port. Update its display `name` (the existing
  NOTE). Keep or remove the trial entries. Regenerate `config.template.yaml`.
- **`mcp/budget.py`.** The pool is discovered, so the shares follow on the
  next `pool_size()`:

  | pool | main 5/8 | helper 3/8 |
  |---:|---:|---:|
  | 163,840 (today) | 102,400 | 61,440 |
  | 196,608 | 122,880 | 73,728 |
  | 262,144 | 163,840 | 98,304 |

  - Stale on adoption: `KV_KIB_PER_TOKEN` 44 (q8 + MTP) becomes 28, measured
    in FIT.
  - Stale on adoption: `what_if`'s `5.95 + 0.5` overhead (pre-MTP).
  - Stale on adoption: the 131,072 fallback (CONSTRAINTS #19).
  - Hardcoded fallbacks elsewhere: `tiers._shares` (102400/61440/163840) and
    `shomen` (61440).
  - Test pins: `mcp/test_budget.py SHIPPED_POOL`, `test_utility`, `test_catalog`.
  - The split itself is the operator's (2026-09-22) and does not change here.
- **Advertised `context_length`.** Another agent is making `/v1/models` report
  the main share. It becomes 163,840 (262,144 pool) or 122,880 (196,608), from
  102,400 today. For Hermes that is +60% or +20%.
- **A cheaper lever, stated for completeness.** Raising the main share at
  today's q8_0 163,840 costs no VRAM and no quality risk. Main 3/4 gives
  122,880, the same window as q4 at 196,608, but shrinks deep thinking to
  40,960.
- **Watchdog.** No code change: it reads model ids from `/running` and checks
  `/health` and slot progress, and none of that is model-specific.
  - An adopted `bonsai` keeps ttl 0, so it stays on the two-strike restart
    path. A trial entry (ttl 1800) is unloaded alone.
  - A ~250k prefill takes ~15-20 min. The watchdog's progress check reads
    `n_prompt_tokens_processed`, so it is not mistaken for a hang.
  - At 262,144 the card idles near the dashboard's 1,280 MiB red line
    (projected 1,237-1,583 free).
- **Docs with the old numbers:** AGENTS.md (budget paragraph), README (effort
  tiers), KNOWN-ISSUES ("VRAM headroom"), MTP-STAGING s.9 ("never lower"),
  `mcp/admission.py` docstring.

## 9. Recommendation and its uncertainty

- **Generate the bias and run the trial. Do not change `bonsai` before the
  gates pass.**
- **Pre-register the floor before the window.** It decides the outcome more
  than anything measured will:
  - **At the documented 3,072 floor, 262,144 is projected to fail FIT** (1.0-1.4
    GB at peak). The realistic candidate is **196,608**, projected 3.1-3.4 GB
    free, which is *more* headroom than today's 1.7 GB for a 20% larger
    window. It sits on the line: 6 MiB under on the conservative slope.
  - **At the floor the stack actually runs at today** (1,280, `TIGHT_MIB`) or
    at HANDOFF item 4's 1,024, **262,144** is projected to pass narrowly.
    `-np 3` would add ~300 MiB, but that is an operator decision: fan-out and
    helpers depend on slots, and KNOWN-ISSUES has an open MTP abort under
    concurrency.

**What could make this wrong, largest first:**

1. **Quality of q4_0 K/V on this model is unmeasured everywhere.** snailium
   says so, and the fork's KLD numbers are for a different model. G3 and G4
   are the first evidence. G4 at n = 21 detects only a collapse.
2. **The VRAM projection.** It rests on one live reading (n = 1). Compute
   buffer slopes measured over 16-24K on the A4000 are extrapolated to 262K,
   and the MTP q4 slope comes from a Linux 3060 at `-np 1`. The two slope
   models differ by ~350 MiB at 262,144. FIT exists to replace all of this.
3. **Speed.** The 3060 Ti's decode step above ~114K allocated window is
   unexplained and may or may not appear on the 5060 Ti. Measured on an sm_120
   card at 32K, q4_0/q4_0 was the fastest KV type. G2 decides.
4. **MTP under concurrency** (KNOWN-ISSUES, open). The trial keeps 4 slots
   like production, so it can hit the same abort. That is recorded as
   `stack_error`, not as a wrong answer.
