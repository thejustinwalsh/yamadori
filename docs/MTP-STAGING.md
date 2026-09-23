# MTP staging: sudoingX `bonsai2` build + Bonsai 2 MTP head

Staged 2026-09-22 on this box **without touching production**. Every test server
ran on the RTX A4000 (CUDA1) only, one at a time, on port 18090, and was stopped.
llama-swap, the proxy, tools API, Laya, the worker, the watchdog and `config.yaml`
were not touched. Nothing was committed.

**Summary.** On the A4000, the fork's PTQ1_0 kernel alone lifts single-stream decode
from 32.4 to 43.0 tok/s on the production file (+33%). Adding the MTP head (n-max 1)
lifts it to 54.2–54.9 tok/s (+68% over the production binary). Output stayed clean
in every check that was run. The cost is VRAM: the head, a doubled per-slot
recurrent state and the draft cache add about 1.86 GB at today's `-c 163840`. Keeping
2.0–2.4 GB free on CUDA0 would mean dropping to roughly **118,784** tokens of
context (projected, not measured on CUDA0). No measurement was taken on the 5060 Ti.

---

## 1. Source and what the branch actually is

| item | value |
|---|---|
| repo / branch | `https://github.com/sudoingX/llama.cpp`, branch `bonsai2` |
| commit | **`285542d98d37d0f07f491cd206aefa31f1848f33`** (`build 10738`) |
| clone | `C:/Users/jwals/llamacpp-sudoingx-bonsai2` (blobless clone, full history) |
| production pin | `9a9394a8` (prism-b10709) **is an ancestor**; the branch is production + 29 commits |

**The branch is the same line as the one that corrupted this stack.** The head of
PrismML-Eng/llama.cpp PR #218 is `sudoingX:pr-ptq1-mmv` at `285542d9`, the same commit
as `bonsai2` HEAD (GitHub API, 2026-09-22). `docs/KNOWN-ISSUES.md` records
`pr-ptq1-mmv` as numerically broken: runs of `/`, 0/10 tool calls. That was an
older state of the branch, based on prism-b9601 when the model needs prism-b10685 or
newer. The current branch sits on b10709. Because of that history, the degeneration
and tool-call checks in section 5 were run as primary tests, not as afterthoughts.

What the 29 commits contain:

- **PR #218, the kernel (10 commits, `e148e857`..`285542d9`).** A planar-transposed
  activation layout, a dedicated PTQ1_0 mat-vec kernel, and
  `GGML_CUDA_BATCH_INVARIANT`. Mat-vec is capped at 4 columns, and 5 or more go to
  the MMQ tile path.
- **PR #214** (`bdc23b56`). A branch-free PTQ1_0 MMQ tile loader (prefill).
- **The qwen35 MTP Hadamard fix.** This is **present, verified in source.** Commit
  `518ad108` "qwen35: apply the Hadamard inverse to the MTP token-embedding lookup",
  merged as PR #205 (`422590f5`). At HEAD, `src/models/qwen35.cpp:638-650` applies
  `hadamard_inverses` (rot + signs) to the MTP `get_rows` on `tok_embd`. PR #217 (the
  alternative fix) is closed and unmerged; it is not needed. The lean file loads,
  creates the MTP draft context, and drafts with high acceptance (section 6). Without
  this fix the lean file cannot start the draft graph.
- Unrelated upstream PrismML merges: Vulkan, Metal, HIP, and CPU AVX2 PQ2_0/Q1_0.

`GGML_CUDA_BATCH_INVARIANT` is read as `getenv(...) != nullptr`
(`ggml/src/ggml-cuda/common.cuh:185`). **Any value turns it on, including `0`.** To
turn it off, leave the variable unset. Per the source comment, its guarantee covers
only the F16/BF16 mat-vec, the PTQ1_0 mat-vec at 1–4 columns, and flash attention
up to 8 queries. It is not a whole-model guarantee (see section 5.3).

## 2. Build

The toolchain and flags are the same as the production build. I read them from
`llamacpp-prism-official/build/CMakeCache.txt` and its `build-cuda.bat`. **Nothing
was installed.**

| | |
|---|---|
| generator | Ninja (bundled with VS 2022 Enterprise) |
| compiler | MSVC 19.36.32543 (`14.36.32532`, Hostx64/x64) via `vcvars64.bat` |
| cmake | VS-bundled `cmake.exe` |
| CUDA | 12.8 (V12.8.93) at `C:\Users\jwals\textgen\installer_files\cudabuild\Library` |
| flags | `-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="86;120" -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON` (GGML_NATIVE defaults ON, same as production) |
| targets | `llama-server`, `llama-bench` only (`-j 12`, 479 steps) |
| output | `C:/Users/jwals/llamacpp-sudoingx-bonsai2/build/bin/` |

**Post-build step.** `cudart64_12.dll`, `cublas64_12.dll` and `cublasLt64_12.dll`
are not placed next to the exe by the build, and without them the server exits with
127. I copied them from the same toolkit's `bin`. The md5 of `cudart64_12.dll` is
identical to the copy in production's `build/bin`. The production build directory
was not touched.

The build script is in the session scratchpad, not the repo. To reproduce, point
`build-cuda.bat` at the new clone.

## 3. Files and hashes

All files are in `C:/Users/jwals/textgen/user_data/models/`. Every file was re-hashed
after testing, and the originals are unchanged.

| file | bytes | sha256 | note |
|---|---:|---|---|
| `Ternary-Bonsai-2-27B-PTQ1_0-mtp-lean.gguf` | 6,297,658,848 | `1e33c571a5ce7a9a3e42474d66192923d5a6d77da7fb3a22986dc809522b5685` | **matches** the HF LFS oid and the repo's `SHA256SUMS` |
| `Ternary-Bonsai-2-27B-Abliterated-PTQ1_0.gguf` | 5,946,648,928 | `94dd53cbad55db5a515f245887c9f0317484502a451ccc0424c7d7788b52900a` | production file, unchanged |
| `Ternary-Bonsai-2-27B-Abliterated-PTQ1_0-mtp-lean.gguf` | 6,297,658,880 | `4ca238c1ac2bf3299d1d012fb1d4357c4840e0dbbc95c14701db7ab9cb137a3b` | **new**, local graft (section 4) |
| `Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf` | 629,246,976 | `6807ede61d570bb86ba34b756a0fa109edc33668604de867c6ea6d8f1d631903` | unchanged |

The graft tooling is `sudoingX/bonsai2-small-gpu` @ `eb52d9d7`, `graft/tools`
(scratchpad clone). The extracted head (`blk.64.*`, 15 tensors, 351,010,432 bytes)
has sha256 `ec9ecad8b3cef6c5fdadbdfd4c3c72842dde4e6afe6b50085040088a42404acb`.

## 4. Abliterated + MTP graft

### Compatibility check

I compared the two files with the graft repo's own raw GGUF reader (read-only):

- **Header.** All 49 KV pairs of the Abliterated file are byte-for-byte equal to the
  lean file's, including every `prism.hadamard.*` key: `transform
  normalized-sylvester-walsh-hadamard`, `block_size 1024`, the full `sign_values`,
  `weight_names`, `inverse_weight_names = [token_embd.weight]` and
  `gdn_v_grouped`. The only differences are the graft's own additions:
  `block_count` 64→65, `nextn_predict_layers = 1` and four `graft.*` keys.
- **Tensor table.** All 851 tensors have identical names, dims, types (PTQ1_0 / F32)
  and offsets.
- **Tensor bytes.** 753 of the 851 are byte-identical, including `token_embd`,
  `output`, `output_norm` and every norm. Only 98 differ: `ffn_down` ×49,
  `ssm_out` ×36 and `attn_output` ×13. These are the projections that write into the
  residual stream, which is the signature of abliteration. The head's inputs are the
  trunk's hidden state, `token_embd` (with the inverse) and `output.weight` as the LM
  head, and the last two are identical in both files.

**No mismatch was found, so the graft went ahead.**

### Procedure

The procedure writes new files only and opens the inputs read-only.

```
python graft/tools/extract_head.py Ternary-Bonsai-2-27B-PTQ1_0-mtp-lean.gguf head-lean.gguf --no-embed-tokens
python graft/tools/merge.py Ternary-Bonsai-2-27B-Abliterated-PTQ1_0.gguf head-lean.gguf Ternary-Bonsai-2-27B-Abliterated-PTQ1_0-mtp-lean.gguf
```

The head comes from the downloaded lean file rather than from the unsloth Q4_K_M
donor, which is not on this machine. The tensors are the same: the lean file's 15
`blk.64.*` tensors are the donor's, copied verbatim.

### Verification

- `merge.py --strip` on the new file reproduces the Abliterated original **byte for
  byte** (sha256 `94dd53cb…`).
- All 15 `blk.64.*` tensors in the new file are byte-identical to the lean file's.
- One cosmetic difference: `graft.donor.name` records
  `Ternary-Bonsai-2-27B-PTQ1_0-mtp-lean.gguf`, not the Qwen file. That is why the
  file is 32 bytes larger than the lean one (header padding).

## 5. Test conditions and correctness

**Isolation.** Every test server ran with `CUDA_DEVICE_ORDER=PCI_BUS_ID` and
`CUDA_VISIBLE_DEVICES=1`, so the 5060 Ti was invisible to the process and got no
CUDA context. The A4000 is then named `CUDA0` inside that process, so the flag was
`-dev CUDA0`. Every log confirms `CUDA0 : NVIDIA RTX A4000 … 0000:03:00.0`.

**Common flags** (mirroring production): `-ngl 999 -fa on -b 1024 -ub 512 --jinja
--reasoning-format deepseek --reasoning-budget 32768 --no-context-shift -ctk q8_0
-ctv q8_0 -np 1`. Where noted, add `--spec-type draft-mtp --spec-draft-n-max 1`.
The draft KV was left at its default (f16) unless stated.

**Speed runs.** Thinking off, `temperature 0, top_k 1, seed 0`, `max_tokens 512`.
The decode tok/s is the server's own `timings.predicted_per_second`, and acceptance
is the server's `timings.draft_n_accepted / draft_n` (the same counters as its
`draft acceptance =` log line).

### 5.1 Smoke

The lean file, MTP on, `BATCH_INVARIANT=1`, `-c 16384`: it loads in 4.5 s and
`/health` returns 200. The log shows `creating MTP draft context against the target
model`. Asked "17 × 23", it answered "17 multiplied by 23 equals 391." with 151
characters of reasoning, so thinking is present.

### 5.2 Degeneration and tool calls

This is the failure mode the old `pr-ptq1-mmv` branch had. Plain generation used
thinking on, temp 0.3 (production), `max_tokens 3000`, and 2 alternating prose
prompts. Output was scanned for 6+ repeated punctuation characters (the `/` runs)
and for words repeated 5+ times.

| config | plain gen, degenerate | finish | tool calls (5 prompts, temp 0.3, thinking on) | decode tok/s (median, range) | acceptance |
|---|---|---|---|---|---|
| production binary + Abliterated (A4000) | 0/6 | 5 stop, 1 `length` (thinking used the full 3000) | 5/5 valid | 32.32 (32.01–32.45) | – |
| new build + Abliterated, head off | 0/4 | 4 stop | – | 42.68 (42.45–42.81) | – |
| new build + lean, MTP, BI=1 | **0/6** | 6 stop | **5/5 valid** | 52.09 (50.35–56.24) | 0.729 (2806/3848) |
| new build + Abliterated graft, MTP, BI=1 | **0/6** | 6 stop | **5/5 valid** | 51.77 (50.72–54.74) | 0.675 (2217/3285) |

The tool calls were `find_definition_opt` and `read_file_range`, and all had valid
JSON arguments naming the right symbol or file. Answers were read by eye (code,
prose, 16K-token summaries, image descriptions) and were coherent. This is **n=6**
per config at production sampling, not a quality benchmark. The old corruption
showed up in 3/5 plain generations, which this sample would have caught.

### 5.3 Greedy identity: head on vs head off

Greedy identity was checked with `temperature 0, top_k 1`, on 3 prompts × 3 reps.
Every config was deterministic rep to rep.

| comparison | TS | bash | prose |
|---|---|---|---|
| lean, BI=1: head off vs head on | **identical** | **identical** | **identical** |
| Abliterated graft, BI=1: head off vs head on | identical | identical | **differs at char 1780 of ~2200** ("key insight" vs "key constraint") |
| lean, no BI: head off vs head on | differs @148 | differs @53 | differs @869 |
| Abliterated graft, no BI: head off vs head on | differs @347 | differs @632 | differs @60 |
| lean head off: BI unset vs BI=1 | differs @1097 | differs @1552 | differs @888 |
| production binary vs new build, Abliterated, head off, no BI | differs @347 | differs @632 | **identical** |

**What this shows.** With BI=1, the head reproduced head-off greedy output on 5 of 6
prompts. The sixth diverged at a near-tie late in the text, and both versions are
coherent. This matches the source comment: BI covers the PTQ1_0 mat-vec and FA
paths, while other ops, for example the gated-delta-net recurrence, can still
depend on batch size. **Head-on greedy output is therefore not guaranteed
byte-identical.** Under sampling (temperature > 0) speculative verification keeps
the target distribution either way.

BI=1 costs about 2% of decode with the head off (43.88 → 43.00 tok/s, lean). With
the head on, the difference is within noise (54.46 vs 54.19). Acceptance is similar
with or without it.

## 6. Speed on the A4000

All runs used `-c 16384` with q8_0 KV and `-np 1`. Each cell is the median tok/s of
3 reps. Rep-to-rep spread was 1.5% or less. The prompts were a TypeScript debounce,
a bash log-gzip script, and ~300 words of prose.

| run | binary / file | head | BI | TS | bash | prose | mean | acceptance TS / bash / prose |
|---|---|---|---|---:|---:|---:|---:|---|
| E | **production** / Abliterated | – | – | 32.36 | 32.37 | 32.40 | **32.38** | – |
| F | new / Abliterated | off | – | 43.12 | 42.91 | 43.00 | 43.01 | – |
| A | new / lean | off | – | 44.03 | 43.83 | 43.77 | 43.88 | – |
| B | new / lean | off | 1 | 43.17 | 42.88 | 42.94 | 43.00 | – |
| C | new / lean | on | – | 57.32 | 55.15 | 50.91 | 54.46 | 0.838 / 0.771 / 0.629 |
| D | new / lean | on | 1 | 57.09 | 55.63 | 49.86 | 54.19 | 0.841 / 0.793 / 0.606 |
| G | new / Abliterated graft | off | 1 | 42.92 | 42.87 | 42.81 | 42.86 | – |
| H | new / Abliterated graft | on | 1 | 57.03 | 55.88 | 51.78 | **54.89** | 0.837 / 0.793 / 0.669 |
| I | new / Abliterated graft | on | – | 55.39 | 54.58 | 49.28 | 53.08 | 0.831 / 0.793 / 0.624 |

On the same card, the kernel alone is **1.33x** the production binary. The kernel
plus head is **1.68–1.70x**, and the head over the kernel is 1.26–1.28x.

For context, the card's RTX 3060 figures were 25.0 → 39.8 → 50.1 tok/s. The ratios
here are similar.

**Not measured: the 5060 Ti.** The sm_120 code is compiled but never ran. Production
decode on the 5060 Ti is 46.06 tok/s (`docs/KNOWN-ISSUES.md`). Any CUDA0 speed
figure is an extrapolation until it is measured.

## 7. Draft acceptance by content type (card format)

All rows use MTP n-max 1 with `GGML_CUDA_BATCH_INVARIANT=1`, thinking off, greedy,
`max_tokens 512`, q8_0 KV, `-np 1`, on the A4000.

- **Acceptance** is llama-server's own `draft_n_accepted / draft_n`, pooled over the
  prompts.
- **tok/s** is the server's decode rate, as a median.
- **Head off** is the lean file on the same build with BI=1 and no `--spec-type`,
  run on the same prompts. The two files run at the same speed with the head off
  (runs B vs G: 43.00 vs 42.86).
- **Content rows** use 3 prompts each. For the lean file, the content rows were run
  twice (P and P2). Every text and every acceptance count repeated exactly.
  - The lean tok/s shown are from the repeat (P2).
  - Run P had rust #1, rust #2 and image #2 at about 33 tok/s. On repeat the two
    rust prompts ran at 59.7 and 53.1, so those slow readings were transient
    contention on the shared A4000, not the build. Image #2 was not repeated.
- **Long-document rows**: 3 different sets of this repo's `docs/*.md`, trimmed to an
  exact token count with the server's `/tokenize`, then "summarise section by
  section, ~350 words".
- **Image rows**: 3 dashboard screenshots from `web/screenshots`, with the
  projector loaded via `--mmproj … --no-mmproj-offload`, i.e. the vision encoder on
  CPU to keep A4000 headroom. Prompts were 1,431–3,996 tokens.

### Lean file (`Ternary-Bonsai-2-27B-PTQ1_0-mtp-lean.gguf`)

| content | n prompts | acceptance | per prompt | tok/s head on | tok/s head off | speedup |
|---|---:|---:|---|---:|---:|---:|
| TypeScript | 3 | **0.839** (655/781) | 0.84, 0.87, 0.81 | 60.0 | 43.5 | 1.38x |
| Rust | 3 | **0.799** (680/851) | 0.84, 0.89, 0.68 | 58.8 | 43.4 | 1.35x |
| Python | 3 | **0.827** (694/839) | 0.84, 0.77, 0.86 | 58.0 | 42.9 | 1.35x |
| bash | 3 | **0.803** (682/849) | 0.79, 0.83, 0.79 | 56.5 | 43.2 | 1.31x |
| prose | 3 | **0.605** (508/840) | 0.61, 0.61, 0.59 | 50.6 | 43.0 | 1.18x |
| long document, 16,035 tokens | 3 | **0.721** (642/890) | 0.76, 0.73, 0.68 | 44.4 | 36.1 | 1.23x |
| long document, 24,035 tokens | 3 | **0.709** (635/896) | 0.69, 0.72, 0.71 | 40.8 | 33.4 | 1.22x |
| image (projector loaded) | 3 | **0.726** (644/887) | 0.72, 0.72, 0.74 | 54.1 | not run | – |

### Abliterated graft (`Ternary-Bonsai-2-27B-Abliterated-PTQ1_0-mtp-lean.gguf`)

| content | n prompts | acceptance | per prompt | tok/s head on | tok/s head off | speedup |
|---|---:|---:|---|---:|---:|---:|
| TypeScript | 3 | **0.832** (669/804) | 0.84, 0.81, 0.85 | 58.2 | 43.5 | 1.34x |
| Rust | 3 | **0.798** (679/851) | 0.82, 0.90, 0.69 | 57.1 | 43.4 | 1.31x |
| Python | 3 | **0.841** (700/832) | 0.88, 0.82, 0.82 | 57.2 | 42.9 | 1.33x |
| bash | 3 | **0.811** (686/846) | 0.79, 0.81, 0.82 | 56.8 | 43.2 | 1.32x |
| prose | 3 | **0.614** (499/813) | 0.67, 0.60, 0.58 | 50.1 | 43.0 | 1.17x |
| long document, 16,035 tokens | 3 | **0.732** (624/853) | 0.74, 0.73, 0.73 | 44.8 | 36.1 | 1.24x |
| long document, 24,035 tokens | 3 | **0.751** (657/875) | 0.78, 0.76, 0.71 | 42.3 | 33.4 | 1.27x |
| image (projector loaded) | 3 | **0.716** (639/892) | 0.71, 0.73, 0.71 | 54.1 | not run | – |

**Card, RTX 3060, for comparison:** code 0.85–0.95 (Python), bash 0.73–0.81,
prose 0.45–0.68, long document 0.65–0.73 at 18K–120K, image 0.88.

Our code acceptance (0.80–0.84 pooled) is slightly below the card's Python range.
Bash, prose and long-document acceptance fall inside the card's ranges. Image acceptance is lower
(0.72 vs 0.88, n=3 screenshots, different images).

**Abliteration does not hurt the head.** The per-type acceptance of the two files is
within ±0.04 on every row.

**Depth.** Rows at **32K and 64K were not run.** With the resident embeddings,
reranker and Laya taking ~7.5 GB of the A4000, MTP at q8_0 leaves ~1.10 GB free at
`-c 25600`. `-c 33792` projects to ~0.76 GB free, below the ~1 GB floor for the
resident services. 24K is as far as CUDA1 allows, and the 64K row needs the 5060 Ti
or an idle A4000.

**Thinking on.** With production sampling (temp 0.3, 6 prompts, section 5.2),
acceptance is 0.729 (lean) and 0.675 (graft), and decode is 52.1 and 51.8 tok/s.
The thinking text drafts about like prose.

## 8. Concurrency: MTP with production's slot shape

Production launches without `-np`. This fork's server treats that as auto, which
means **4 slots and `kv_unified = true`** (`tools/server/server.cpp:152`).

- **`-np 2`, 2 concurrent requests** (`-c 16384`). Both requests completed with
  normal acceptance (0.83 and 0.79). Per-stream decode was 36.3 and 38.0 tok/s, and
  the aggregate was **66.4 tok/s**.
- **`-np` auto (4 slots, unified), 4 concurrent requests** (`-c 8192`). All 4
  completed, with acceptance of 0.80, 0.78, 0.59 and 0.79. Per-stream decode was
  18.6–24.3 tok/s, and the aggregate was **74.7 tok/s**.

**This is a smoke test at n=1 per shape, not a throughput benchmark.** It shows that
MTP runs in the multi-slot, unified-KV mode production uses. Greedy text under
concurrency differs from single-stream text, because the batch changes kernels.

## 9. Memory at q8_0 KV (the operator's standing decision: never lower)

The components come from the server's own buffer log lines (`-lv 4`) on the A4000,
using this build and `-ub 512`.

| component | measured | per-token | scales with |
|---|---:|---:|---|
| weights, Abliterated (GPU) | 5,395.33 MiB | – | – |
| weights, lean/graft (GPU) | 5,730.08 MiB (**+334.75**, the `blk.64` head) | – | – |
| main KV, q8_0, 16 attention layers | 544.00 MiB @ 16,384; 816.00 @ 24,576 | **34.0 KiB** | `-c` |
| recurrent state, no MTP | 149.62 MiB per slot | – | slots |
| recurrent state, **with MTP** | **299.25 MiB per slot** (1 → 2 cells for rollback) | – | slots |
| main compute buffer | 170.28 @ 16K; 210.28 @ 24K (np 1) | ~5 KiB | `-c` |
| draft KV, **f16** (default), 1 layer | 64 MiB @ 16K; 96 @ 24K | 4 KiB | `-c` |
| draft compute, f16 draft KV | 114.02 @ 16K; 122.02 @ 24K; 130.02 @ 8K np 4 | ~1 KiB | `-c` |
| draft KV **q8_0** (`-ctkd/-ctvd q8_0`) | 51 MiB @ 24K | 2.125 KiB | `-c` |
| draft compute with q8_0 draft KV | **200.28** @ 24K | – | `-c` |

- **q8_0 main KV is 34.0 KiB/token, exactly.** 16,384 cells come to 544.00 MiB.
  This confirms the `config.yaml` figure. The 44 KiB/token that `mcp/budget.py`
  uses is not the KV alone. These logs show 34 KiB of KV plus ~5 KiB of compute
  buffer growth per token (39 KiB); where budget.py's other 5 KiB comes from was
  not traced.
- **Do not quantise the draft cache.** A q8_0 draft KV saves 45 MiB of cache at 24K
  but adds 78 MiB of compute buffer, for a net cost of **+33 MiB** (251 vs 218 MiB).
  The f16 draft cache is also higher precision than q8_0, so leaving it at the
  default meets the q8_0 floor.
- **Long-prompt residency.** Actual use after a 15.5K or 23.5K-token prompt grew
  only 28–36 MiB over the at-load figure. The buffers are reserved at load.
- **A4000 totals** (whole card, including ~7.5 GB resident):
  - lean head off, `-c 16384`: 13,992 MiB used
  - MTP, `-c 16384`: 14,664 MiB
  - MTP, `-c 24576`: 15,016 MiB
  - MTP, `-c 16384`, `-np 2`: 14,972 MiB
  - MTP, `-c 8192`, `-np` auto (4 slots): 15,256 MiB

### Projection for the 5060 Ti (CUDA0): **not measured, arithmetic only**

With production's shape (`-np` auto = 4 slots, q8_0, f16 draft KV), the MTP file
adds the following over today's file at the same `-c`:

```
head 334.75 + extra recurrent 4 x 149.63 = 598.5 + draft compute ~122 + (4 + 1) KiB/token x c
= 1,055 MiB + 5 KiB x c   ->  1,855 MiB at c = 163,840
```

Each token removed from `-c` frees 39 KiB with the head off (34 KV + 5 compute) and
44 KiB with the head on.

Today's file at `-c 163840` has three possible anchors for free VRAM:

- **Derived idle, ~2,445 MiB free.** `config.yaml` records 172032 idling at 14,178
  MiB; subtract 8,192 × 39 KiB.
- **Derived peak, ~2,251 MiB.** The same figure less the recorded 194 MiB
  worst-case stress.
- **Observed this session, 1,953–2,222 MiB.** Read with `nvidia-smi` while the
  5060 Ti was at 93–100% utilisation, desktop apps included.

Projected free on CUDA0 with the MTP file:

| `-c` | idle anchor | peak anchor | observed-low anchor |
|---:|---:|---:|---:|
| 163,840 (today) | 590 | 396 | 98 |
| 131,072 | 1,998 | 1,804 | 1,506 |
| 122,880 | 2,350 | 2,156 | 1,858 |
| **118,784** | **2,526** | **2,332** | **2,034** |
| 114,688 | 2,702 | 2,508 | 2,210 |

**The head costs context: about 45K tokens (−27%)** to hold the same 2.0–2.4 GB of
headroom. The largest `-c` that keeps ≥ 2.0 GB free projects to 119.6K–131.0K,
depending on the anchor. **118,784 keeps ≥ 2.0 GB free under all three anchors.**

**Assumptions to verify on CUDA0 before relying on this:**

- The compute-buffer slopes are extrapolated from 8K–24K to 118K–164K.
- sm_120 kernels may size buffers differently.
- The recurrent-state doubling was measured at np 1, 2 and 4, so that part is solid.

Two levers the operator could consider instead:

- **Fewer slots.** Each slot costs 299 MiB with MTP, which is about 7K tokens of
  context. This is an operator decision; the fan-out and helpers depend on slots.
- **Keep today's file and 163,840 context**, and take only the kernel speedup: the
  new build with the plain Abliterated file, no `--spec-type`. That is +33% on the
  A4000 with no memory change (13,992 vs 14,008 MiB at 16K).

## 10. Proven vs not

**Proven here** (A4000, n as stated):

- The `bonsai2` branch contains the qwen35 MTP Hadamard fix, confirmed in source,
  and the lean file runs MTP on it.
- The lean file matches its published sha256.
- The Abliterated graft is structurally sound. Its header and Hadamard conventions
  are identical, and the strip round-trip is byte-exact. Its acceptance matches the
  stock file's (±0.04 per row).
- The new kernel is +33% decode over the production binary on the A4000, and the
  head adds +26–28% on top (n = 3 prompts × 3 reps, deterministic).
- No `/`-run degeneration: 0/16 sampled generations across 3 new-build configs.
  Tool calls were 10/10 valid with MTP on (5 per file).
- MTP works with 1, 2 and 4 slots, including unified KV.
- Per-component VRAM, including 34.0 KiB/token for q8_0 KV.
- A q8_0 draft KV costs more total VRAM than f16.

**Not proven:**

- **Anything on the 5060 Ti:** speed, memory, the sm_120 kernels, and the context
  projection in section 9. Measure before adopting.
- **Quality beyond smoke level.** No benchmark was run, and the sample sizes are
  n=4–6 generations and n=5 tool calls per config.
- **Greedy byte-identity.** It is not guaranteed even with BI=1 (5/6 prompts).
- **Acceptance at 32K–120K context.** It was not reachable on CUDA1.
- **Long-context behaviour of `--reasoning-budget` forcing with MTP.** No run hit
  the budget.
- **Behaviour through `:1234`.** The proxy, tiers, budgets and fan-out were not
  tested. Per `AGENTS.md`, the live suite through the proxy is required before any
  claim that this works in the stack.
- **llama-swap `env:` support** for `GGML_CUDA_BATCH_INVARIANT`. It is used below
  but was not exercised on this v256 binary.

## 11. Proposed `config.yaml` block (q8_0 KV; **NOT APPLIED**)

This block would replace the `bonsai` model's command. It depends on:

- a warm CUDA0 stress test of the kind used for 163840 (main share filled plus two
  deep-thinking contexts), showing ≥ 2.0 GB free at peak with throughput intact;
  step down to 114688 if it fails;
- the live suite through `:1234`;
- a decision on the fact that this binary is not the pinned official PrismML
  release.

`mcp/budget.py` follows `-c` automatically. The display `name` would need its
context label updated deliberately, as the existing NOTE says.

```yaml
macros:
  # sudoingX/llama.cpp bonsai2 @ 285542d9 (PR #218 kernel + qwen35 MTP Hadamard fix,
  # on top of prism b10709). Built with the production toolchain, CUDA 86;120.
  # Staged and measured in docs/MTP-STAGING.md.
  server_mtp: "C:/Users/jwals/llamacpp-sudoingx-bonsai2/build/bin/llama-server.exe"

models:
  "bonsai":
    # ... name / aliases / filters unchanged ...
    env:
      # Batch-invariant PTQ1_0 mat-vec and FA for 1-4 columns. Read as "set or not":
      # ANY value enables it, including "0". Remove the line to disable.
      - "GGML_CUDA_BATCH_INVARIANT=1"
    cmd: |
      ${server_mtp}
      --port ${PORT}
      -m ${models}/Ternary-Bonsai-2-27B-Abliterated-PTQ1_0-mtp-lean.gguf
      -dev CUDA0
      -ngl 999
      # MTP costs ~1,055 MiB + 5 KiB/token over the plain file (docs/MTP-STAGING.md
      # section 9). 118784 is the PROJECTED largest context keeping >= 2.0 GB free
      # on CUDA0 at 4 slots; it must be warm-measured before it ships.
      -c 118784
      --cache-type-k q8_0 --cache-type-v q8_0
      # Draft cache deliberately left at its f16 default: q8_0 draft KV measured
      # +33 MiB MORE in total at 24K (larger draft compute buffer).
      -fa on
      --cont-batching
      -b 1024 -ub 512
      --jinja
      --spec-type draft-mtp
      --spec-draft-n-max 1
      --reasoning-budget 32768
      --reasoning-budget-message "Thinking budget reached. I will stop deliberating and write the final answer now."
      --reasoning-format deepseek
      --no-context-shift
      ${sampling}
    ttl: 0
```

## 12. Test hygiene

- Every test server was started by the harness and stopped by it. After the last
  run, no test server remained: port 18090 had only `TIME_WAIT` connections. The A4000 was back
  at its 7,555 MiB baseline after every run.
- **One breach.** During run N (MTP, 4 slots, `-c 8192`), A4000 free memory dipped
  to **890 MiB for about 25 s**, under the ~1 GB floor. The harness then only
  enforced the floor during load. `:1234`, `:1235` and `:1237` all returned 200 on
  `/health` immediately afterwards. The harness was then changed to kill the test
  server whenever free memory drops below 1,000 MiB, and no later run came within
  60 MiB of that.
- **A process outside the stack was seen.** At 23:21:10, `sd-cli.exe`
  (stable-diffusion.cpp, qwen-image) started on the A4000, 27 s after the last test
  run ended at 23:20:43. It did not overlap any measurement here. It is not part of
  the stack and was left alone.
- The bench harness (`mtpbench.py`), the drivers, the per-run JSON with full output
  texts, and the server logs are in the session scratchpad under `runs/`.
