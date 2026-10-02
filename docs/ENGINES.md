# Engines: what we run, and how to rebuild it exactly

Every inference engine the stack runs is pinned in `engines/manifest.yaml`:
the upstream source at a full SHA, our patches as files in
`engines/patches/<engine>/`, the inputs a build fetches, the toolchain and
flags read from the original build trees, and the SHA-256 of every file the
binary we ship loads. `scripts/build_engine.py` rebuilds an engine from that
entry into a new directory; `scripts/deploy_check.py` refuses a deploy whose
`config.yaml` runs anything the manifest does not pin.

| engine | what | base | patches | runs |
|---|---|---|---|---|
| `llama-bonsai2` | sudoingX/llama.cpp `bonsai2`, no web UI | `285542d98d37d0f07f491cd206aefa31f1848f33` | `0001-reasoning-budget-nudge.patch`, `0002-cuda-ptq1_0-pdl-sync.patch` (sudoingX/llama.cpp#1's `967d312`) | `shipped` is the 0001+0002 build (`llama-bonsai2-f1ef2064`), NOT YET DEPLOYED: `bonsai` (`server_nudge`) runs the 0001-only build (`llama-bonsai2-eebbaae2`), pinned as `previous` until the switch (below, "The PTQ1_0 PDL race") |
| `llama-bonsai2-base` | the same base, no patches | `285542d98d37…` | none | `bonsai-q4kv`, `bonsai-q4kv-196k` (`build/`) |
| `llama-bonsai2-ada` | PrismML-Eng/llama.cpp `prism` + the bonsai-ada-surgery series, CANDIDATE | `adfffbe41b2cabcd51fff326ab045662265062bb` | their 33 (`0001`-`0033`), our nudge (`0034` = llama-bonsai2's 0001), FA-vec skip (`0035` = its 0003), per-sequence draft window (`0036`), tiered-KV tail by pool cells (`0037`); their `0026` replaces our 0002 | nothing yet: `shipped` = `92ffd4db`, the PHASE 2 candidate (below, "bonsai-ada-surgery merge") |
| `llama-upstream` | ggml-org/llama.cpp `master`, CANDIDATE for the max tier's Flash-Next (qwen4exp), which no fork of ours can load | `4da6337767f973e2b4d0797e5b323d77d8565e4a` | our nudge (`0001` = llama-bonsai2's 0001) | nothing yet: `shipped` = `llama-upstream-1b446647`; docs/FLASH-NEXT.md |
| `llama-upstream-mtp` | the same + the Flash-Next MTP draft head | `4da6337767f9…` | `0001` + `0002` (PR #28243's net diff, head `6fcaa16f`) | nothing yet: `shipped` = `llama-upstream-mtp-3534183c`, a PHASE 2 arm |
| `llama-upstream-moe` | the same + Strata's MoE-offload work, ported (expert streaming ring, GPU expert cache with Strata's profile and adaptive tier, pinned / large-page experts) | `4da6337767f9…` | `0001` nudge, `0002` PR #28414, `0003` PR #27861, `0004`-`0005` Strata ports | nothing yet: `shipped` = `llama-upstream-moe-2c8dbf5f`; "Strata's MoE work, ported" below |
| `llama-prism` | PrismML-Eng/llama.cpp `prism` | `9a9394a895b96003ca842a6041cb28ac49a108f7` | none | vision, embeddings, critic (the reranker removed 2026-10-01, docs/REMOVED.md) |
| `sd-cpp` | leejet/stable-diffusion.cpp | `c92d73c408515c94beef32161bb5960764fde7a0` | `0001-vae-conv-weight-dtype-and-vae-compute-precision.patch` (PR #2043's `f047986`) | `imagegen`, `imagegen-turbo` |
| `llama-swap` | mostlygeek/llama-swap release v256 | tag commit `6701d0d9…` | none (release zip, not built here) | the router |

`python scripts/build_engine.py --list` prints the same, with each entry's
build hash.

## No web UI (the default for llama.cpp engines)

llama-server can embed a web UI. Nothing here uses it -- the dashboard is the
proxy's `/dash` (operator, 2026-09-25) -- so llama.cpp engines build without
it: `-DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF`. **Both** flags:
`LLAMA_BUILD_UI=OFF` only skips the npm build; `scripts/ui-assets.cmake`
still downloads the ggml-org HF prebuilt and embeds it while
`LLAMA_USE_PREBUILT_UI` is ON. With both OFF the UI step fetches nothing and
writes an empty asset table (`tools/ui/ui.h` has no `LLAMA_UI_HAS_ASSETS`,
which the builder checks), and the server registers no `/` route: `GET /`
is a 404, `/health` and `/v1` are unchanged.

`build_engine.py` refuses a llama.cpp entry without both flags unless it
carries `embeds_ui: <reason>`. Two do, because they reproduce live binaries
that were built with a UI: `llama-bonsai2-base` (to be retired, not
rebuilt) and `llama-prism`, whose entry records under `next_rebuild` that its
next rebuild drops the UI. It is serving vision and embeddings (the reranker
was removed 2026-10-01, docs/REMOVED.md) and is not rebuilt now.

## Verify what runs

    python scripts/build_engine.py --verify-only            # every binary config.yaml points at + llama-swap
    python scripts/build_engine.py llama-prism --verify-only

For each binary `config.yaml` runs (the first word of every model's `cmd`,
macros expanded) the path must be some engine's `shipped.path`, and the
binary and every DLL it loads from its own directory must match
`shipped.files` in size and SHA-256. A `previous` block that still records
its files is accepted the same way (reported as `<engine> (previous)`): it is
how the binary `config.yaml` still runs stays pinned between a rebuild and
the switch to it. Delete the block once nothing runs it. "Loads" is read from the PE import
tables, delay-loads included, transitively: a DLL dropped beside the exe that
the server would load but the manifest does not pin fails the check. About 4 s
(it hashes ~2.7 GB, most of it cuBLAS).

`scripts/deploy_check.py` runs this first, as step 0, and exits 1 (NOT GOOD,
recorded in `logs/deploy_check.jsonl`) before it waits for anything. An
unpinned binary is not deployed. (A `previous` block marked
`rebuildable: false` is pinned but has no recipe any more; it is accepted
only until `config.yaml` stops using it.)

Not pinned, because the system supplies them at run time: the MSVC runtime
(`vcruntime140`, `msvcp140`, `vcomp140`) from the VC++ redistributable, and
`nvcuda.dll` from the NVIDIA driver. A driver update changes what runs
without changing a hash here.

## Rebuild

    python scripts/build_engine.py sd-cpp --jobs 8
    python scripts/build_engine.py llama-prism --jobs 8 --out C:/Users/jwals/eng-prism01

The default output is `C:/Users/jwals/engines/<engine>-<hash>`, where the hash
covers the entry's source-defining keys, its toolchain and its patch files --
not the `shipped` block, so recording a new shipped hash does not rename the
build. The layout mirrors the originals: `<out>/src` is the checkout,
`<out>/src/build` the build tree, logs and fetched inputs in `<out>`.

It refuses (exit 2, nothing created) a pending patch, an `--out` that exists
and is not empty, one inside or around a checkout a shipped engine lives in,
and one a running process uses. It never touches the original trees.

Then, in order, each step failing loudly:

1. **Source.** `git init` + `git fetch` of the pinned SHA -- shallow for
   `llama-prism` (the original is a `--depth 1` clone and embeds build number
   1), blob-less full history for the bonsai2 engines (they embed 10738) --
   with `core.autocrlf=true` (the originals are CRLF checkouts: it changes
   `build-info.cpp`, and would change any multi-line raw string) and
   `core.abbrev` set to the short-hash length the original embedded. Then the
   submodules, each checked against its pin.
2. **Patches**, in order, through the index (`git apply --cached --check`,
   apply, check out), so their line endings convert the way a checkout's do.
   HEAD stays at the base, as it was for the originals, which were built from
   uncommitted changes.
3. **Inputs.** Only the two entries that reproduce a UI-embedding binary
   have one: the web UI archive (URL + SHA-256), placed in `tools/ui/dist`,
   which the UI step uses before anything else. `sd-cpp`'s original was built from a copy with no `.git`, so its `.git`
   is removed and git is fenced out (`GIT_CEILING_DIRECTORIES`): it embeds
   version and commit "unknown", as the original does.
4. **Environment.** vcvars64 run over a minimal PATH (System32 and git),
   then CUDA's `bin` first, then the tools the manifest says were on PATH
   (Git's `gzip`, for `llama-prism`, whose UI was gzipped); npm and pnpm must
   be absent. Checks that vcvars chose the recorded MSVC toolset (14.36, not
   the newest installed 14.37) and Windows SDK.
5. **Configure** with the recorded flags; then the CMakeCache is checked
   against every flag and `expect_cache`, `build.ninja` against the native
   CPU flags GGML_NATIVE resolved to, the configure log against
   `expect_configure_log`, and `build-info.cpp` against its recorded hash.
6. **Build** the recorded targets; check the generated UI source (its hash
   for an entry with a UI, the absence of `LLAMA_UI_HAS_ASSETS` without);
   copy the CUDA DLLs (source and copy hash-checked).
7. **Compare** every file the new binary loads with the shipped one, print
   the new binary's path and SHA-256, and write `<out>/build-report.json`.
8. **Tests**, if the entry lists them, in a separate tree
   (`<out>/src/build-tests`) so the shipped flags are untouched:
   `llama-bonsai2` builds `test-reasoning-budget` and `test-chat` in a static
   CPU tree (Windows skips them in a shared build) and runs them with ctest.

Do not run a rebuilt binary against the GPU while the stack is serving.

## Add a patch

1. Make the change in a scratch checkout of the engine's base (or export it
   read-only from a working tree: `git diff <base> > NNNN-name.patch`,
   wrapped with a `From:`/`Subject:` header like `git format-patch` writes;
   `0001-reasoning-budget-nudge.patch` is an example).
2. Put it in `engines/patches/<engine>/NNNN-<name>.patch`, next number.
3. Add it to the entry's `patches` with its `sha256`
   (`sha256sum`/`Get-FileHash`). A patch whose file is not final yet is
   listed with `status: pending`; the builder refuses to build it.
4. `python scripts/build_engine.py <engine> --jobs 8`. A conflict stops the
   build at step 2.
5. Deploy the new binary, then record it: `python scripts/build_engine.py
   --describe <path to exe>` prints the `shipped.files` block (hashes of the
   exe and every DLL it loads). Replace `shipped` (keep the old block as
   `previous` while anything still runs it), set `built_with_patches`, and
   point `config.yaml` at it. `deploy_check.py` fails until the manifest and
   `config.yaml` agree.

## Bump a base commit

1. Change `base_commit` (full SHA), and `submodules` if they moved.
2. Rebuild; every patch must still apply. Refresh the ones that do not
   against the new base and update their `sha256`.
3. The generated-file pins (`expect_generated`) are for the OLD base: the
   build number and short hash change `build-info.cpp`. Re-pin it from the
   new build and say so in the entry.
4. Keep the no-UI flags. A llama.cpp engine being bumped that still has
   `embeds_ui` (llama-prism) follows its `next_rebuild` note and drops the
   UI.
5. Record `shipped` as in step 5 above.

## The PTQ1_0 PDL race: `0002-cuda-ptq1_0-pdl-sync.patch` (2026-09-26/27)

**The defect.** At `285542d98` the PTQ1_0 mat-vec kernel
`mul_mat_vec_ptq1_0_pt` (`ggml/src/ggml-cuda/mmvq-ptq1_0.cuh`) is launched
through `ggml_cuda_kernel_launch`, which uses Programmatic Dependent Launch
on sm >= 90 when the build defines `GGML_CUDA_USE_PDL` (CUDA >= 12.3 under
MSVC; ours is 12.8). The kernel never calls `ggml_cuda_pdl_sync()`, and the
kernel that writes its input, `quantize_q8_1`, triggers launch completion
as its first statement (`quantize.cu:63`): the mat-vec may read its q8_1
activations before they are written. Every decode step takes this path
(`mmvq.cu:1065-1071`, 1-4 columns, with or without
`GGML_CUDA_BATCH_INVARIANT`). The A4000 (sm_86) never does, which is why
`docs/MTP-STAGING.md` section 5 -- run only on the A4000 -- could not see it.
Reported symptom on an RTX 5080: `!!!!` / `////` a few tokens in
(sudoingX/bonsai2-small-gpu#3, PrismML-Eng/llama.cpp#218).

**The patch.** sudoingX/llama.cpp#1, head `967d31271cc3` (LamplighterPaul,
unmerged at export), taken as its single commit in `git format-patch` form:
one `ggml_cuda_pdl_sync()` at the top of the kernel, before any read, as
`mul_mat_vec_q` does (`mmvq.cu:607`). Parent = our base; blob `baa90977` ->
`d70ae441`; +4/-0. It compiles to nothing below Hopper. Reviewed: no other
kernel in this tree launched through `ggml_cuda_kernel_launch` lacks the
sync (`gdn_precompute_exp` has none but is launched with `<<<>>>`, no PDL).
The sm_120 SASS was not inspected (no `cuobjdump` in the build's CUDA env).

**The build.** `python scripts/build_engine.py llama-bonsai2 --jobs 8` into
`C:/Users/jwals/engines/llama-bonsai2-f1ef2064`: every check passed (all 207
CMakeCache options equal eebbaae2's, `build-info.cpp` byte-identical, no UI,
CUDA DLLs identical, engine tests 2/2). Nothing new was downloaded (the
pinned source fetch only). `llama-server.exe` sha256
`8d6b9a2764b545ccc9fb79149c2e71eaa0ff5170aa4aa8ee433648da5717e47e`; it is
the manifest's `shipped` block. **Not deployed**: `config.yaml`'s
`server_nudge` still runs eebbaae2 (`previous`) until the deploy flow
switches it.

**T0 on the 5060 Ti** (`bench/engine_corruption.py`; records in
`bench/results/engine_corruption/t0-pdl-20260927/`). Three arms, each a
separate llama-server with exactly `bonsai`'s argv and env from
config.yaml (checked against llama-swap's `/running`), on `127.0.0.1:18091`,
production `bonsai` unloaded for each arm's ~7-minute window and reloaded
after (reload 4.3 s each, `:1234/health` 200). The CPU was shared with
other agents' Docker grading; the card had nothing else on it. n per arm:
greedy 3 prompts x 3 reps (thinking off, temp 0, top_k 1, 512 tokens);
16 plain generations (thinking on, production sampling, seeds 1000-1015,
3,000 tokens); 10 tool requests (seeds 2000-2009).

| arm | greedy decode tok/s, median (min-max), n=9 | greedy reps 1-2 identical (3 prompts) | plain with `!!!!`/`////` | 10+ `!`/`/` | 6+ punct run | 5x word | plain finish | tools valid |
|---|---|---|---|---|---|---|---|---|
| (a) current, eebbaae2 | 62.09 (55.76-63.86) | 0/3 | 0/16 | 0/16 | 0/16 | 0/16 | 16 stop | 10/10 |
| (b) current, `GGML_CUDA_PDL=0` | 60.40 (54.79-62.23) | 3/3 | 0/16 | 0/16 | 0/16 | 0/16 | 15 stop, 1 length | 10/10 |
| (c) fixed, f1ef2064 | 61.14 (54.76-62.73) | 3/3 | 0/16 | 0/16 | 0/16 | 0/16 | 15 stop, 1 length | 10/10 |

Text identity between arms, every request:

| comparison | greedy (9) | plain (16) | tools (10) |
|---|---|---|---|
| (b) vs (c) | 9/9 identical | 16/16 identical (reasoning and answer) | 10/10 identical |
| (a) vs (c) | 0/9 (first difference at char 60-588) | 0/16 | 8/10 |

- No `!!!!`/`////` run appeared in any arm at this n (0/16 plain, 0/9
  greedy, 0/10 tools each). The reported symptom was not reproduced here.
- The unpatched binary is not deterministic at temperature 0 on this card:
  warm reps 1 and 2 differed on 3 of 3 prompts in the full run and 2 of 3 in
  an earlier run aborted after 6 plain generations (its record is kept as
  `current.aborted-partial-...json`). With PDL off, or with the patch, they
  were identical on 3 of 3, and the two arms produced the same bytes on all
  35 requests. So on the 5060 Ti the race is live in the shipped binary: it
  changes what the model writes, without (at n=16) producing the garbage
  runs.
- Rep 0 (a cold prefill) differed from the warm reps on `ts` in every arm;
  that is the cache path, not the race.
- Speed. (b) and (c) decoded identical tokens, so they compare directly:
  (c) was 0.5% (ts 62.06 -> 62.38), 0.9% (bash 60.81 -> 61.38) and 0.6%
  (prose 54.83 -> 55.17) faster than (b), means of 3 reps. (a) wrote
  different text with different draft acceptance (bash 0.79 vs 0.77), so
  its 62.09 is not a like-for-like number; the per-prompt spread in (a)
  (ts 62.3-63.9) overlaps (c)'s. Plain generations (thinking on), median
  tok/s over 16: (a) 52.94, (b) 52.67, (c) 52.79. n=1 run per arm.

**To deploy** (with the deploy flow, not before): point `server_nudge` at
the shipped build, restart, then `python scripts/deploy_check.py --key-file
PATH` must exit 0; then remove the eebbaae2 `previous` block once nothing
runs it. The stopgap without a rebuild, `GGML_CUDA_PDL=0` in `bonsai`'s
`env`, gave the same bytes as the patch here at ~0.5-0.9% lower decode.

## Masked KV cells: `0003-cuda-fattn-vec-skip-masked-kv.patch` (2026-09-27, CANDIDATE)

**The defect.** With the unified pool (auto `-np`: 4 slots) a decode step
attends over `n_kv` = the pool up to its highest used cell
(`llama_kv_cache::get_n_kv` pads `used_max_p1()`), and every cell of
another sequence, or empty, is masked -inf. The FA vec kernel -- decode and
MTP verification, <= 8 queries under `GGML_CUDA_BATCH_INVARIANT` -- reads
and multiplies all of them; `flash_attn_mask_to_KV_max` trims a masked tail
only for >= 1,024 queries. So an idle conversation's cells, or the holes it
leaves, tax every other decode (SELF-IMPROVEMENT-LOG #58/#59).

**The patch.** In `flash_attn_ext_vec`'s KV loop, a warp whose 32 cells of
the chunk are -inf for every valid query column skips its body (each warp
owns its cells for both the KQ and the VKQ part). Masked cells add
exp(-inf) = 0 and leave the running max unchanged, so the result is the
same bits for finite K/V. +19 lines, one file. **Not in
`engines/manifest.yaml`** (the shipped entry is base + 0001 + 0002): built
from a copy of the manifest with this entry appended to `llama-bonsai2`'s
patches,

    - file: 0003-cuda-fattn-vec-skip-masked-kv.patch
      sha256: df392653cef4bdc554ada2d6bd8c07d57261480eb5e4320aa568a9a9aa06f041

into `C:/Users/jwals/engines/llama-bonsai2-790071e8` (every check passed,
207 CMakeCache options equal, build-info byte-identical, engine tests 2/2;
`llama-server.exe` sha256 `dc411c74...f9`, `ggml-cuda.dll` `b3c21691...12`).

**Speed on the 5060 Ti** (bonsai's exact argv/env on :18091, production
unloaded per arm; active A, idle B of 41,326 cells; decode tok/s median of
n=3; two runs for patched and current):

| A at | case | current (f1ef2064) | 0003 |
|---|---|---|---|
| 8k | alone | 62.0 | 59.6 / 60.5 |
| 8k | B below A, kept | 36.9 | 50.1 / 57.7 |
| 8k | B above A, kept | 37.1 | 59.1 / 58.2 |
| 8k | 3 slots x 40k above A | 20.2 | 55.1 / 54.0 |
| 32k | alone | 44.4 | 42.7 / 42.7 |
| 32k | B below A, kept | 29.7 | 41.7 / 41.6 |
| 32k | 3 slots x 40k above A | 17.8 | 39.9 / 40.0 |
| 64k | alone | 30.1 | 29.8 / 29.1 |
| 64k | B below A, kept | 23.4 | 29.6 / 29.3 |

The cost: a LONE conversation is ~3.5% slower at 32k (42.7 vs 44.4, and
44.0 / 43.9 in two in-place runs of the current engine). A variant that
loads each lane's mask once and shuffles it into the KQ loop
(`llama-bonsai2-dfacc1f0`) was slower still (32k alone 38.3, one partial
run, stopped for an operator run); it is not the patch.

**Not done:** `bench/engine_corruption.py` on the patched binary (greedy
identity against `t0-pdl-20260927/fixed.json` is the expectation, since
the skip is exact); removing the lone-conversation overhead. **Not
deployed**: switching means adding the entry above to the manifest,
recording `shipped`, pointing `server_nudge` at the build, and
`deploy_check.py` exiting 0.

## bonsai-ada-surgery merge (2026-09-27, PHASE 1: CPU only)

Operator, 2026-09-27: "Fresh bonsai tune incoming!!! ... We are going to
need a lot of this merged into our existing patch." Cary Palmer's
[bonsai-ada-surgery](https://github.com/professorpalmer/bonsai-ada-surgery)
(MIT) is 33 `git am` patches on PrismML's llama.cpp. PHASE 1 was done with
no GPU and nothing sent to :11434 or :1234: the sources read, the overlap
analysed, the engine built. Nothing here is measured on our card yet; every
speed number quoted from the bundle is THEIRS (RTX 4070 12 GB, GDDR6X
+1500 MHz overclock, `-np 1`, their flags), not a prediction for ours.

### Sources (read-only, pinned)

| what | where | commit |
|---|---|---|
| the bundle (docs, bench, patches/) | github.com/professorpalmer/bonsai-ada-surgery `main` | `5158a8df3bc4c8b9375ed521021eb6118feda78d` (tag `bundle-20260927` is `f9dda2ea`, four docs/cards/pagoda commits older; `patches/` is identical) |
| the series as a branch | github.com/professorpalmer/llama.cpp-ada-ternary `bonsai-q8-product` | `c8b8993a6211db9edc4df9de9e1f8b1915d183df` |
| the base | github.com/PrismML-Eng/llama.cpp `prism` (tag `prism-b10743-adfffbe`) | `adfffbe41b2cabcd51fff326ab045662265062bb` |
| our base, for comparison | github.com/sudoingX/llama.cpp `bonsai2` | `285542d98d37d0f07f491cd206aefa31f1848f33` |

Checked: the 33 patch files, applied with `git am` to `adfffbe`, give
EXACTLY the tree of `c8b8993a` (tree hashes equal), and each file's
`From <sha>` line is that branch's commit (33/33). The files are vendored
byte for byte in `engines/patches/llama-bonsai2-ada/` (the bundle marks
`*.patch` `-text`, so its blobs are these bytes) with the bundle's
`LICENSE` as `LICENSE.bonsai-ada-surgery`. Read: README, `docs/Q8_FULL_CONTEXT.md`,
`docs/QUALITY.md`, `docs/RECEIPTS.md`, `surgery/ADA4070_PTQ1.md`,
`start-server.ps1`, `build/make_mtp_procreations.ps1`.

### Where the two trees stand

`git merge-base 285542d98 adfffbe` is `bdc23b56` (PrismML #214). So:

- **Our base** = PrismML up to #214 (the MTP Hadamard-embedding fix #205 is
  in it) + sudoingX's 10 #218 commits.
- **Their base** = PrismML up to `adfffbe`: #214 + 16 more commits. Only
  one touches the CUDA path we run: #216 (4-column GDN warp layout on every
  Ampere+ card, their "+6% prefill"). The rest are CPU/SYCL/ROCm/Vulkan/
  Metal kernels, DFlash2 (#261), tied Hadamard output weights (#257),
  Hadamard tensors out of CPU_REPACK (#245), a speculative-visibility log
  (#203/#272) and x86 SSE (#248).
- **Their series**, by `git range-diff bdc23b56..285542d98 adfffbe..c8b8993a`:

| their patches | what | vs our engine |
|---|---|---|
| 0001-0005 | sudoingX #218: planar-transposed q8 activations, the dedicated PTQ1_0 mat-vec, `GGML_CUDA_BATCH_INVARIANT`, bf16 small-row mat-vec | **identical** (`=`) to our base's first five #218 commits |
| 0018, 0020, 0008 | restrict off the kernel params (+ Ampere takes the PT 1-column path), one smem budget for guard and launch, MMQ from 5 columns | the same fixes as our base's remaining #218 commits `ccffa9c25`, `0b159700d`, `285542d98` (two docs-only commits have no counterpart) |
| 0026 | `ggml_cuda_pdl_sync()` at the top of `mul_mat_vec_ptq1_0_pt` | **the same fix as our 0002** (same line, same comment; theirs is `09b6cce` rebased). Our 0002 does not apply on their stack and is not needed |
| 0006 | recurrent-state gather folded into GATED_DELTA_NET (#220) | new to us |
| 0007, 0010 | SoA q8 activations with exact integer sums, warp-per-row small-K GEMV (#215: "+14% TG" on the 4070); multi-column mat-vec with per-pair epilogue ("+13% 3-col, +24% 4-col") | new; 0010 is the MTP-verify path (2 columns at our `--spec-draft-n-max 1`) |
| 0009 | flash attention MMA kernel reads q4_0/q8_0 K/V in place (no F16 scratch copy) | new |
| 0029 | quantized-KV GQA decode on that MMA kernel: <= 8 queries, K == V type quantized, GQA ratio > 4, one stream, KV >= `GGML_CUDA_FA_MMA_DECODE_MIN_KV` (256) | new; **takes our decode off the vec kernel** (below) |
| 0011-0016, 0021, 0023 | MTP graph/catch-up fixes, Hadamard self-quantization (0013), out-of-vocab guard (0014), FWHT pool teardown (0016) | new; our MTP path has none of them |
| 0017, 0031 | `--spec-draft-depth-max`; `--spec-draft-window`, `--spec-draft-n-max-tail` | new flags, all off by default |
| 0027, 0030 | BATCH_INVARIANT: warp-reduce epilogue restored; FA KV split sized from a fixed 4 blocks/SM; PTQ1_0 mat-vec up to 8 columns | changes what our `GGML_CUDA_BATCH_INVARIANT=1` does: 5-8 column batches (4 slots x draft 1) now stay on the mat-vec, as 1-4 do |
| 0028 | tiered KV (`--kv-vram-cells`): VRAM head, pinned-host tail | new flag, off by default |
| 0032 | `--reasoning-effort-allow/-fallback`, `--reasoning-max-tokens-floor` | new flags, off by default ("Harness-proofing" below) |
| 0019, 0022, 0024, 0025 | review fixes (MUSA scalar vec-dot, header order, HIP/MUSA FA instances skipped, ALiBi/softcap tests) | new |
| 0033 | `GGML_CUDA_OP_TIMING=1` per-node GPU time | diagnostics, off by default |

**Our patches on their stack:**

- `0001-reasoning-budget-nudge.patch` applies cleanly (offsets only, 8
  files, +513/-22). Nothing in the series touches `common/reasoning-budget.*`
  or `server-schema.cpp`; its `common/sampling.cpp` change (0014) is an
  out-of-vocab guard beside the backend-sampler branch, not the budget
  sampler. Vendored as `0034-reasoning-budget-nudge.patch`, same bytes.
- `0002` is superseded by their 0026 (above).
- `0003-cuda-fattn-vec-skip-masked-kv.patch` applies cleanly, but on this
  stack it is **mostly dormant for our flags**: with q8_0 K/V, 24 query
  heads over 4 KV heads, one unified stream and <= 8 queries, 0029 routes
  every decode and MTP verify with >= 256 KV cells to the MMA kernel, and
  it comes BEFORE the batch-invariant "vec for 1-8 queries" rule, so
  `GGML_CUDA_BATCH_INVARIANT=1` does not keep us on vec either. **The #59
  defect is NOT fixed there**: the MMA kernel also iterates every KV tile
  up to `n_kv` (the pool's highest used cell); `flash_attn_mask_to_KV_max`
  still runs only for >= 1,024 queries or several streams, and nothing in
  the MMA loop skips a fully masked tile. What changes is the per-cell
  cost: the MMA kernel reads each K/V row once per KV head, the vec kernel
  once per query head (6x). Taken anyway, as
  `0035-cuda-fattn-vec-skip-masked-kv.patch` (same bytes): it is exact, it
  keeps the vec route (below 256 cells, and `GGML_CUDA_FA_MMA_DECODE_MIN_KV=0`)
  from regressing to the #59 tax, and it lets PHASE 2's placement probe
  compare MMA-with-masked-reads against vec-with-the-skip in ONE binary by
  an environment variable. If MMA wins even with idle slots, 0035 can be
  dropped; if the tax persists on MMA, the fix is a tile-level skip in
  `fattn-mma-f16.cuh` (a new patch), not 0035.

**PDL audit (source, coarse).** Every kernel launched through
`ggml_cuda_kernel_launch` in the patched tree calls `ggml_cuda_pdl_sync()`
somewhere in its body, the series' new ones included (`fwht_quantize_q8_1`
from 0013, the PTQ1_0 planar mat-vec from 0026). Presence, not ordering
before the first read, was checked; the bundle's `tests/check_pdl_codegen.py`
checks ordering in the sm_120a SASS of the planar kernel but needs CUDA 13's
`cuobjdump`, which our 12.8 env does not have. The 5060 Ti runs PDL, so
PHASE 2's rep-to-rep determinism is the runtime check (it is what exposed
0002's race).

**Two multi-slot hazards found reading the code, and our fixes (0036, 0037):**

- **`--spec-draft-window` (0031) with 4 slots -> our 0036.** 0031 trims the
  DRAFT context of EVERY slot by the view's lowest position minus the window
  (`for (auto & slot : slots) llama_memory_seq_rm(ctx_dft, slot.id, 0,
  pos_min - n_window)`). With `-np 1` (all their measurements) that is the
  one slot. With our 4 slots, a view carrying a deep conversation's tokens
  erases a shallow conversation's recent draft rows, whose drafts then come
  from missing context: output stays correct (the target verifies every
  token), acceptance drops. `0036-spec-draft-window-per-sequence.patch`
  (ours, +19/-6, one file) computes, for each sequence that has a token IN
  THE VIEW, its own lowest position there, and trims that sequence by that
  bound. **Why it cannot touch another slot's rows, from the source:** the
  only call is `llama_memory_seq_rm(llama_get_memory(ctx_dft), s, 0, hi)`,
  and `s` ranges over `seq_pos_min`, which is filled only from
  `batch_view.seq_id[i][k]` for `i < batch_view.n_tokens`; the server uses
  the slot id as the sequence id, so a slot with no token in the view is
  never named. For a slot that is named, `hi = (its own lowest position in
  the view) - n_window`: the rows it drops are older than the window before
  the rows it is about to add, the same bound 0031 applies with one slot.
  With one slot the two are identical. The draft context is sized for
  `n_seq x (window + 2 x n_batch + 256)` cells (`common_speculative_init`),
  which every sequence's trimmed window fits.
- **`--kv-vram-cells` (0028) with the unified 4-slot pool -> our 0037.**
  STORAGE IS SAFE AS IS: the tier is applied only when `n_stream == 1`
  (a unified pool, ours), per layer, as the first `N / kv_size` of the K and
  V buffers, whose rows are cells in pool order: the line is on POOL CELLS,
  for every slot alike; kernels are unchanged; the host tail is mapped into
  the same device range, and an attention op whose K/V view reaches it copies
  those rows into a VRAM staging buffer first (same bytes); slot save/restore
  and `--cache-ram` go through `ggml_backend_tensor_get/set` on the same
  virtual range. What mixes cells and positions is one HEURISTIC: 0031
  drafts `--spec-draft-n-max-tail` once a slot's POSITION reaches N
  (`spec_tail_depth`). With a unified pool every sequence's attention spans
  `n_kv = pad(used_max_p1, 256)` cells of the one stream
  (`llama_kv_cache::get_n_kv`), so the host tail is read by EVERY slot as
  soon as ANY slot's cells pass N, and by none before, whatever a slot's own
  depth. `0037-tiered-kv-tail-by-pool-cells.patch` (ours, +46/-2) adds
  `llama_memory_cells_used_max_p1()` (the kv cache: the max of
  `v_cells[s].used_max_p1()` over streams, the value `get_n_kv` pads; the
  hybrid memory ours uses: its attention cache; any other memory: 0) and
  drafts the tail size while that exceeds N. Draft size only: nothing stored
  or computed changes. One shared-staging note: without the draft window the
  MTP context is tiered at the same N and shares the staging buffer; its
  decode runs after the target's outputs are read (synchronised), so the
  two never use it at once; with the window the draft context (4 x 18,688
  cells) is below N and not tiered at all.
- Their recipe drafts 2 tokens with a q8_0 draft cache; ours drafts 1 with
  q4_0. Their GQA-decode gain (0029) is quoted with draft 2. (The target
  below takes theirs.)

### Every proxy dependency, on their stack (source-level)

`git diff 285542d98 c8b8993a` touches none of `tools/server/server-http.cpp`,
`server.cpp`, `server-task.*`, `server-queue.cpp`, `common/chat*.{cpp,h}`,
`common/reasoning-budget.*`; `server-context.cpp` changes only the
speculative paths (depth max, window, tail, a log of the active spec types)
and passes 0032's three fields; `common/arg.cpp` only ADDS seven flags. So:

| dependency (who uses it) | on their stack |
|---|---|
| `/slots` (`slots.py`: `is_processing`, `n_ctx`, cells), `POST /slots/<id>?action=erase` answering 501 without `--slot-save-path` (`slots.release_idle` falls back to a one-token `/completion` with `id_slot`, `n_predict` 0) | unchanged code |
| `/apply-template`, `/tokenize` (`with_pieces`, `add_special`), `/props` `chat_template` (`tiers.accepted_efforts` parses the guard clause) | unchanged |
| chat completions `logprobs` / `top_logprobs` (the Bonsai decider), `id_slot`, `cache_prompt`, `timings.cache_n` / `prompt_n` / `predicted_per_second` (`x_yamadori.cache`) | unchanged |
| `reasoning_budget_tokens` (alias `thinking_budget_tokens`), `reasoning_budget_message`; `reasoning_budget_nudge` / `_at` | the first two upstream, unchanged; the nudge is 0034 (= our 0001) |
| `--reasoning-budget`, `--reasoning-budget-message`, `--reasoning-format deepseek`, `--jinja`, `--no-context-shift`, `--cont-batching`, `-b 1024 -ub 512`, `-fa on`, `-c 181248`, `--cache-type-k/v q8_0`, `--cache-type-k/v-draft q4_0`, `-dev CUDA0 -ngl 999`, sampling flags, `--no-cache-idle-slots`, `--cache-ram` (default 8,192 MiB) | all present (config.yaml's `bonsai` argv, read from config.yaml) |
| `--spec-type draft-mtp --spec-draft-n-max 1` | present; the draft path has 0011-0016, 0021, 0023 on top |
| Qwen3-Coder XML tool calls with the lazy grammar | `common/chat*` unchanged |
| `GGML_CUDA_BATCH_INVARIANT=1` (bonsai's env) | present, with 0027 and 0030's wider meaning (above) |
| the idle-slot save/restore to host RAM (`resumed_cold` `restored`) | `get_available_slot` / `[TAG_IDLE_SLOT_CLEAR]` unchanged |

Source-level only: every row is a PHASE 2 live check.

### Harness-proofing: theirs vs ours

Operator, 2026-09-27: "We may want to merge their harness proofing with
ours or take theirs where it is better than ours." Their mechanism is two
server flags (0032, OFF by default) plus `start-server.ps1` defaults; ours
is the proxy, which sits in front of every client. Doing a thing twice
(server + proxy) must be avoided or made consistent.

| behaviour | theirs | ours | verdict |
|---|---|---|---|
| an effort word the template rejects (`high` -> HTTP 500) | `--reasoning-effort-allow LIST` + `-fallback medium`: any word not listed becomes medium. Their recipe allows ONLY `medium`, so `low` and `xhigh` become medium too. Evidence: Killy's plate `effort: "high"` 0 (HTTP 500) -> 160/164 | the client's word picks a TIER (`tiers.resolve`); the tier sends `medium` at `low`..`xhigh` and `xhigh` only at `max` (`tiers.TIERS`); `safe_effort` rounds any other word UP to an accepted one, and the accepted set is PARSED FROM THE SERVED TEMPLATE (`tiers.accepted_efforts`), not a list; a template refusal that still happens is a 400 `invalid_prompt`, never a 502 (`api_errors.template_refusal`) | **Ours is at least as good and already covers it**: no request through the proxy can reach the template with `high`. Theirs is a static list that can drift from the template (ours drifted once: AGENTS.md "WHAT THE TEMPLATE WILL ACTUALLY ACCEPT"). Enabling 0032's allow list with THEIR value (`medium`) would silently turn our `max` tier's `xhigh` into medium: a double rewrite. If we want a backstop for direct `:11434` callers, allow exactly the template's set (`low,medium,xhigh`), which is a no-op for the proxy. Recommendation: leave it off (the proxy is the one door, AGENTS.md "One door to the model"). |
| a small client output cap with thinking on | `--reasoning-max-tokens-floor N`: `max_tokens` below N is raised to N (their N = budget + 4,096 = 24,576), thinking and answer still share it. Evidence: a 4096 cap 137 -> 157/164 (148 with the floor off on the same server) | the client's `max_tokens` is an ANSWER allowance, `max(client, A_MIN=2048)`, and thinking is a separate budget on top: `max_tokens = thinking room + answer`, `reasoning_budget_tokens = thinking` (`tiers.budget`); a 256-token cap gets 2,048 of answer after its thinking | **Ours is better-founded**: it keeps the answer's room separate from thinking, which theirs does not (a 24,576 floor with a 20,480 budget leaves 4,096 for the answer whatever the client asked). Enabling the floor server-side would CONFLICT: it would rewrite the two deliberate totals the proxy sends -- a benchmark's `reasoning_cap` header (thinking + answer is the whole allowance, "measured meaning") and a compaction's fixed budget. Keep it off. |
| server default effort | `--chat-template-kwargs {"reasoning_effort":"medium"}` | llama-swap filter `reasoning_effort?: "medium"`, and the proxy writes it on every request | same choice, same evidence (medium). |
| thinking budget | one server default, 20,480, force-close message "Now produce the complete answer."; Evidence: Killy's MBPP/HumanEval grid where medium overtakes thinking-off from a ~20k TOTAL cap; in their replay 2 of 3 misses at medium hit 20,480 and were force-closed | a runaway breaker at the server (32,768) and per-request caps: user turn 12,288 (`USER_TURN_THINKING`), agent step 6,144, second-brain jobs 3,072-12,288, compaction 2,048; the nudge (our 0001, at 0.6 of the budget) plus `BUDGET_MESSAGE` | **A design difference, and a question their data raises**: their evidence is a TOTAL-cap grid (a cut-off answer scored as a fail), where our force-close still yields an answer, so it does not measure our caps. But it is the only HumanEval-shaped evidence for a user-turn cap above 12,288, and Killy's grid moves with the cap up to 80k (MBPP medium 42.2 -> 45.3). The caps are operator choices from agent-loop overthinking (Octopus v0b), not code-quality measurements. PHASE 2 can run their `bench/humaneval_run.py` plates through `:1234` at the current caps to see where we stand; any change of a cap is the operator's. |
| tool calls | Qwen3-Coder XML parsed by the server with the lazy grammar; 9/9 parsed vs 1/9 when the model writes Hermes JSON in content (n=9, thinking off) | the same server path: the proxy passes the client's tools through untouched, so every harness that sends `tools` gets the grammar; plus `tool_code` checks and repairs the code INSIDE a write call | ours already has theirs, plus the code check. Their 9/9 vs 1/9 is evidence FOR the path we use. |
| thinking on tool-heavy agents | advice: `enable_thinking: false` per request (8/9 parsed thinking-off vs 6/9 at medium, n=9; the medium misses spent 10-17k tokens thinking and hit their 9,000-token cap) | thinking stays ON at every tier but `minimal`; agent steps capped at 6,144 thinking with the agent-step nudge, answer room separate | **Design choice with our own evidence**: config.yaml's injection test (thinking off exfiltrated 5/5, on 0/5, n=5) is why thinking stays on. Their 6/9 failures are their shared-cap mechanism (thinking ate the 9,000 tokens); our split budget does not have that failure. Not taken. |
| "run what it wrote" | `pagoda_plate.py --repair`: render headless, send facts back, 3/6 -> 6/6 | the harness runs tools; `tool_code` checks syntax only | their loop is a client/harness behaviour, not a server one. As a BENCHMARK practice it feeds grader facts back to the model, which our rules forbid (memory "No graded follow-ups"). Not taken. |
| backend (GPU) sampling | `--backend-sampling` on by default; off whenever a grammar or the reasoning budget is active | not used; every thinking request carries a budget | nothing to take: the budget sampler is host-side on both stacks. |

**Does their HumanEval data speak to our `xhigh` default?** Our default is
not xhigh: the default tier is `medium` (`YAMADORI_TIER_DEFAULT`), and every
tier from `low` to `xhigh` sends the template `medium`; only tier `max` sends
`xhigh`. Their data (Killy's grid: medium beats xhigh "at every cap", low
behaves close to xhigh; one seed per row, on PQ2_0 weights for Killy's own
rows and PTQ1_0 for the replays, not our abliterated trunk) agrees with
`tiers.TIERS`' own note that "no result in this repo says xhigh helps", and
is the first external evidence AGAINST `max`'s xhigh effort. Whether `max`
should send medium is the operator's call; it is not changed here.

### The build

`python scripts/build_engine.py llama-bonsai2-ada --jobs 8`, CPU only, the
same toolchain as `llama-bonsai2` (`msvc-cuda128`: MSVC 14.36, SDK
10.0.22621.0, CUDA 12.8.93) and the same configure flags (native sm_86 for
the A4000 and sm_120 for the 5060 Ti, no web UI). Three builds on
2026-09-27, each with every check passed (source `adfffbe`, every patch
applied through the index, every flag in the CMakeCache, `/arch:AVX2`,
`ui.h` without assets, CUDA DLLs hash-checked, `test-reasoning-budget` +
`test-chat` 2/2 in the static CPU tree -- 0034's own tests on the new base):

| build | patches | what it is for | `llama-server.exe` sha256 |
|---|---|---|---|
| `llama-bonsai2-ada-51d64e6f` | 0001-0035 | the entry's `previous`: the series' 0031 draft-window trim as shipped, kept ONLY for PHASE 2's `ada-window-0031` arm | `b97e805c...f1a` |
| `llama-bonsai2-ada-3c366404` | 0001-0036 | intermediate, superseded, not recorded (delete after PHASE 2) | `67369df9...0a3` |
| **`llama-bonsai2-ada-92ffd4db`** | **0001-0037** | the entry's **`shipped`**: the PHASE 2 candidate | `beac021e2818708abc9187c6554db465372074b3eaa16f5db6db2275daa5540f` |

`build-info.cpp` (build 10743, commit `adfffbe41`, sha256 `0b14858b...61d0`)
is pinned in `expect_generated` since the 0036 rebuild, and the 0036 and
0037 rebuilds' 208 CMakeCache options equal 51d64e6f's. `build_engine.py
llama-bonsai2-ada --verify-only`: 1/1; the deploy check's engine step: 5/5;
`mcp/test_engines.py`: 157/157. `ggml-cuda.dll` is 111,764,480 bytes
(93,095,936 in `790071e8`): 0009's quantized-K/V MMA instances. **Not
deployed, not run on any GPU, `config.yaml` untouched.**

### The ProCreations MTP head (fetched on the operator's yes, 2026-09-27)

`ProCreations/Ternary-Bonsai-2-27B-MTP` @ `efffdea64c1f9e93cc7fa6bb24f72ae9d66ecf51`
(Apache-2.0, not gated): the r3-mtp checkpoint, trained on-policy against
the ORIGINAL Bonsai 2 PQ2_0 trunk. Fetched the way the bundle's
`build/make_mtp_procreations.ps1` does: `surgery/hf_sparse_fetch.py` pulled
only the header (96 MiB) and the 15 `blk.64.*` tensors (451,319,808 bytes)
of the 7,657,489,728-byte combined GGUF into a sparse file, and sudoingX's
`extract_head.py --no-embed-tokens` (`bonsai2-small-gpu` @ `eb52d9d7`, the
pin our lean graft used; `graft/tools` is unchanged at today's head
`faa71475`) wrote `head-procreations.gguf` (451,321,504 bytes, sha256
`c0473290...e5ba`). Everything is under the models directory
(`procreations-mtp/`), none in the repo; recorded in `models/manifest.yaml`
(`procreations-mtp-head`, `procreations-mtp-master-bf16`) with the byte
ranges and each tensor's sha256.

**Verification.** SHA256SUMS covers the donor only as a whole file, which a
sparse fetch cannot check (the response did carry `X-Repo-Commit` =
`efffdea6` and `X-Linked-ETag` = the SHA256SUMS hash). So the head was
checked against a file SHA256SUMS DOES cover: `model_mtp.safetensors`
(849,400,392 bytes, `7a4a18b2...`, equal to SHA256SUMS and manifest.json),
the BF16 master weights. Re-exported with ProCreations' own
`training/export_mtp.py` rules (bf16 -> f32; 1-D norms + 1 as f32; 2-D
through gguf-py's Q8_0 quantize), it reproduces **all 15 fetched tensor
payloads byte for byte**.

**The graft is NOT done.** `merge.py` onto our abliterated trunk (writing a
NEW file, `Ternary-Bonsai-2-27B-Abliterated-PTQ1_0-mtp-procreations.gguf`,
the served file untouched) was refused by this session's permission
classifier. The command, and the proof that must follow it (as the bundle
and our lean graft did):

    python graft/tools/merge.py Ternary-Bonsai-2-27B-Abliterated-PTQ1_0.gguf procreations-mtp/head-procreations.gguf Ternary-Bonsai-2-27B-Abliterated-PTQ1_0-mtp-procreations.gguf
    python graft/tools/merge.py --strip Ternary-Bonsai-2-27B-Abliterated-PTQ1_0-mtp-procreations.gguf strip-check.gguf
    # sha256(strip-check.gguf) must equal the trunk's 94dd53cbad55db5a515f245887c9f0317484502a451ccc0424c7d7788b52900a

The head was trained for the original trunk's features; the abliterated
trunk's differ. Its acceptance on our trunk is unknown and is measured in
PHASE 2, not assumed from their +4.3 pp.

### The target: q8_0 everywhere, fitted with the tiered KV cache

Operator, 2026-09-27: "Use their q8, it's the whole point, they fit q8 with
the vram thing." The configuration PHASE 2 gates and, if it passes, deploys:

| | today (`bonsai`) | the target |
|---|---|---|
| engine | `llama-bonsai2-790071e8` | `llama-bonsai2-ada-92ffd4db` |
| K/V | q8_0, `-c 181248`, all in VRAM | q8_0, `-c 262144` (the trained window), cells `[0, N)` in VRAM, the rest in pinned host RAM (`--kv-vram-cells N`, 0028) |
| draft (MTP) K/V | q4_0 (`-ctkd/-ctvd`, chosen for VRAM, config.yaml's `bonsai` note) | q8_0 |
| draft size | `--spec-draft-n-max 1` | 2, and 4 while the pool's cells are past N (`--spec-draft-n-max-tail 4`, keyed on pool cells by our 0037) |
| MTP head | lean (the Qwen 3.8 teacher graft) | ProCreations if its graft is recorded and it wins on this engine, else lean |
| slots | 4, unified pool | unchanged |
| everything else | config.yaml's `bonsai` argv and env | unchanged |

**Is tiering a speed feature on a card whose pool fits?** No, from the
source. 0028 changes the buffer type of a layer's K/V only when
`N < kv_size` (and `n_stream == 1`, `type_k == type_v`); with no
`--kv-vram-cells` every buffer is the ordinary device buffer. The one change
on the all-VRAM path is in `ggml_cuda_flash_attn_ext`: a mutex-guarded scan
of the registered tier buffers per attention op, which finds none and
returns (negligible; G2 measures it anyway). The staging copy and the
attention split it mentions are for the host tail. 0030's fixed 4 blocks/SM
KV split is a BATCH_INVARIANT change that applies to every non-stream-k
launch, tiered or not; the `ada` arms measure it. So tiering is capacity,
and the question is what the capacity costs.

**The VRAM line, sized like `start-server.ps1` does -- but measured.**
Their script computes `N` from free VRAM minus a demotion MARGIN minus 4070
constants (weights, recurrent state, compute buffers, CUDA context).
PHASE 2's step 0 keeps their margin logic and replaces the constants with a
measurement: launch the target at `N = 131,072`, warm it, read the card's
lowest free VRAM, move `(free - margin) / 34,816 B` cells into VRAM (a cell
costs its target K/V, 34,816 B, and -- without the draft window, when the
MTP context is tiered at the same N -- its draft K/V, 2,176 B, and frees
its row of the ONE staging buffer, one layer's K+V, 2,176 B:
`start-server.ps1`'s `(Ctx-N)*CellBytes/16`), relaunch, check; up to three
rounds. **The margin** is theirs: 1,000 MiB when `nvidia-smi` reports no
display on the card, 1,300 MiB when it does (`start-server.ps1`;
`docs/Q8_FULL_CONTEXT.md` "The VRAM line": on their 4070, Windows demotes a
background process's allocations to shared memory SILENTLY near full --
"600 paged at once", 800 "held for minutes, then 79.9 -> 56.8 tok/s",
1,000 held a 10-minute soak headless, 1,300 with the display on). Our
5060 Ti: `nvidia-smi` says `display_active Disabled` (read 2026-09-27; PCIe
link gen 5 x8), so 1,000 MiB -- 400 MiB more than config.yaml's 600 MiB
floor, which was an operator target, never a demotion measurement.

**Estimate before the fit** (arithmetic, not a measurement: today's pool
sits at the 600 MiB floor by config.yaml's own sizing note; 0009's in-place
K/V read, 0013's FWHT pool and 0028's buffers change the other allocations,
which is why step 0 measures):

| | K/V + draft + staging budget | N (cells in VRAM, x256) | host RAM pinned |
|---|---|---|---|
| today, `-c 181248`, q4_0 draft, 600 MiB floor | 6,519 MB (181,248 x 34,816 + 181,248 x 1,152) | 181,248 (all) | 0 |
| target, lean head, 1,000 MiB margin | 6,100 MB = N x 34,816 + 262,144 x 2,176 | **~158,720** | (262,144 - N) x 36,992 = **~3.83 GB** |
| target, ProCreations head (+100 MB of head) | 6,000 MB | **~155,904** | ~3.93 GB |
| target + draft window (MTP context 4 x 18,688 cells, not tiered) | 6,100 MB - 163 MB fixed | ~164,352 | ~3.40 GB |

**What N means per slot.** The line is on POOL cells, shared by the four
slots: every conversation runs at full speed while the pool's highest used
cell is below N -- one conversation of ~158k, or four of ~39k -- and every
conversation reads the host tail on every step once any slot's cells pass
it (0037 then drafts 4). Where a conversation's cells land is the
allocator's (`find_slot` from the head, #59), and our slot release and idle
clear (AGENTS.md) are what keep the high cells free. Today the same cells
are simply refused: the proxy's window check answers
`context_length_exceeded` and the harness compacts.

**Can `-c` go to 262,144?** Yes, with these costs: (1) the fast region
shrinks from 181,248 cells to ~158k (the larger margin is ~12k cells, the
q8_0 draft cache ~4k, the ProCreations head ~3k); (2) ~3.8 GB of pinned
host RAM (this machine: 63.7 GB, 11.1 GB free when read, and
`--cache-ram`'s host prompt cache, default 8,192 MiB, draws on the same
RAM); (3) past the line, every step of every conversation copies the used
tail over PCIe (5.0 x8 here): e.g. 20k cells past N is ~700 MB per step.
Their 4070 (PCIe 4.0 x16, line at 113k) decoded 41 tok/s at 131k, 27 at
180k, 14.7 at 258k -- their numbers, not ours; PHASE 2's `spill` and
past-the-line depths measure ours; (4) the proxy advertises the new main
share: `mcp/budget.py` reads the pool from `/props`, so the main share
becomes 262,144 - 49,152 = 212,992 tokens and Hermes compacts later.

### Their recipe, flag by flag, against `bonsai`

Their values from `start-server.ps1` (and the README's Linux line); evidence
from `docs/QUALITY.md`, `docs/Q8_FULL_CONTEXT.md`, `docs/RECEIPTS.md`, all
on their RTX 4070 (GDDR6X +1500 MHz), `-np 1`, the ORIGINAL (not
abliterated) trunk.

| flag | theirs | ours | their evidence | ours | take |
|---|---|---|---|---|---|
| slots | `-np 1` | auto (4), one unified pool | every number is one slot | the proxy pins conversations to slots and reserves the helper slot (AGENTS.md) | keep 4; `ada-np1` / `ada-np1-window` report what one slot buys |
| window / K/V | 262,144, q8_0, tiered at ~113k | 181,248, q8_0, all VRAM | q8_0 KLD 0.00017, top-1 99.38% vs q4_0 0.00218, 97.93% (16k chunks, n=4) | q8_0 since the 09-25 rollback | **target: 262,144 tiered** (operator) |
| draft K/V `-ctkd/-ctvd` | q8_0 (= the main type) | q4_0, chosen for VRAM (config.yaml `bonsai`, 2026-09-24) | none for the draft cache alone | none: speed vs f16 was to be measured, acceptance never compared | **target: q8_0** (operator); `ada-dkv-q8` isolates it |
| `--spec-draft-n-max` | 2 | 1 | "2 stays best (prose at 16k: 87 vs 82.5 for 3; code +3% for 3)"; n-max 1 is in their `mtp_sweep.py` but no result is recorded | MTP-STAGING ran 1 only; never compared with 2 | **target: 2**; `ada-draft2` isolates it |
| `--spec-draft-n-max-tail` | 4 past the line | - | "+26% code / +15% prose at 180k for 4 vs 2" | - | **target: 4**, keyed on pool cells (0037) |
| `--spec-draft-window` | 16,384 | - | 32k 54.5 -> 103.6, 64k 48.0 -> 90.1 tok/s with the MMA decode route, acceptance unchanged; "window alone +17-18% at 131k" | - | arm `target-window` (with our 0036); `ada-window-0031` shows the multi-slot hazard |
| `--spec-draft-depth-max` | 24,576 (older recipe) | - | draft a net loss past ~28k before 0029 | - | not taken: superseded by the window |
| MTP head | ProCreations (on-policy, r3-mtp) | lean (teacher graft) | +3.9% tok/s, +4.3 pp acceptance (3 prompts x 3 x 400 tokens, original trunk) | lean on the abliterated trunk: acceptance 0.80-0.84 code, 0.61 prose (A4000, n=3) | fetched and verified; graft pending; `target-pc` vs `target-lean` decides |
| `GGML_CUDA_BATCH_INVARIANT` | 1 | 1 | "at no measurable cost" | T0 | same |
| `-b` / `-ub` | 2048 / 512 | 1024 / 512 | none cited | none | same ubatch; `-b` only chunks the logical batch (and sizes the windowed draft context, `2 x max(n_batch, n_ubatch)`): not taken without a measurement |
| `-n` | 24,576 | server default | - | the proxy sets `max_tokens` on every request | nothing to take |
| thinking budget | `--reasoning-budget 20480` + "Now produce the complete answer." | server breaker 32,768 + the proxy's per-request caps, nudge (0001) at 0.6, `BUDGET_MESSAGE` | Killy's grid (below) | the caps' notes in `mcp/tiers.py` | proposal below |
| effort | `--chat-template-kwargs {"reasoning_effort":"medium"}`, allow `medium`, fallback `medium` | tiers send medium (`max`: xhigh, deliberate) | Killy's plates | `tiers.accepted_efforts` | ours (comparison above) |
| output floor | `--reasoning-max-tokens-floor` = budget + 4,096 | the proxy's answer allowance | 4,096 cap: 148 -> 157 | `tiers.budget` | proposal below |
| `--backend-sampling` | on | off | GPU-side sampling, ~4% decode, off whenever a grammar or a budget is active | every thinking request carries a budget | nothing to take |
| `--prio 2 --poll 100` | set | defaults | not measured on its own in their docs | - | not taken without a measurement |
| `--metrics` | on | off | - | the proxy records `x_yamadori` | no |
| min-p | not set (llama.cpp's 0.05) | 0.0 (Qwen's published value) | - | config.yaml `sampling` | keep ours |
| `--cache-ram` | default (8,192 MiB) | default | - | the idle-slot restore (`resumed_cold` `restored`) | same; note the pinned tail competes for the same RAM |
| VRAM margin | 1,000 headless / 1,300 with display (soak-tested) | 600 MiB floor (operator target) | "600 paged at once" | none on demotion | **target: theirs** (operator), in step 0 |
| display on the iGPU | recommended: desktop VRAM 930 -> 285 MiB, line 95k -> 113k | `nvidia-smi` shows no display on the 5060 Ti | measured | - | already so |
| K mean-centering | measured, no gain on top of the Hadamard rotation | retired 2026-09-25 with q4_0 | KLD table | - | agreement |

### Token floor and thinking budget: a merge proposal (for the operator; `tiers.py` not changed)

**Their mechanism.** `--reasoning-max-tokens-floor N` raises a client
`max_tokens` below N to N when thinking is on; their N = budget + 4,096 =
24,576, and thinking and answer share that total. `--reasoning-budget 20480`
with "Now produce the complete answer." at the force-close.

**Their evidence.** Killy's HumanEval plates, replayed on their server
(164 problems, tests executed, one seed per row): a 256-token cap 17 ->
**160**; `effort: "high"` HTTP 500 -> **160**; a 4,096-token cap 137 (Killy,
PQ2_0) -> 148 (their server, floor off) -> **157** (floor on); medium at 8k+
**161**. Killy's own grid (MBPP / HumanEval pass counts by TOTAL output cap,
thinking off vs medium): medium loses to thinking-off at 10,224 and wins
from 20,488 up (MBPP medium 35.44 -> 42.22 -> 44.11 -> 45.33 at 10k / 20k /
41k / 82k). Their 20,480 budget is "the first cap in his table where medium
wins"; in their replay 2 of the 3 medium misses hit 20,480 and were
force-closed.

**Ours.** `tiers.budget`: the client's `max_tokens` is an ANSWER allowance,
`answer = max(client, A_MIN = 2,048)`, and thinking is a separate budget on
top, `max_tokens = thinking room + answer`, the room derived from the
request's share of the KV pool; then capped: a user turn thinks at most
`USER_TURN_THINKING` 12,288, an agent step `AGENT_STEP_THINKING` 6,144, the
second brain's jobs 3,072-12,288; the nudge (our 0001) at 0.6 of the cap,
the hard stop with `BUDGET_MESSAGE`. h4's measured agent-step reasoning:
median 128, p90 3,737 tokens.

**What their floor proves, and where it already lives.** Their floor is
founded: a small client cap must never end a thinking request inside its
thinking. **Our proxy already does exactly that, for every client, in ONE
place**: a 256-token cap gets the thinking budget PLUS 2,048 of answer, and
a 4,096 cap gets thinking plus 4,096 of answer -- at least as much room as
their 24,576 total, and the answer's room is never eaten by thinking (their
shared total leaves 4,096 for the answer after a full 20,480 think). So:
**adopt the principle as already implemented; never enable
`--reasoning-max-tokens-floor` on the server** -- it would rewrite the two
totals the proxy sends on purpose (a benchmark's `reasoning_cap`, whose
meaning is "thinking + answer is the whole allowance", and a compaction's
fixed budget). One place, the proxy.

**What their budget suggests.** Their data is about SINGLE-SHOT coding
answers, and it is the only HumanEval-shaped evidence either repo has for a
thinking budget. It is a total-cap grid (a truncated answer scores 0),
while our cap force-closes into an answer, so it does not measure our cap
directly -- but it says medium keeps gaining past our 12,288.

**Decided 2026-09-27 (operator: "Yes bump user turn, and I trust their numbers
more than ours, our benchmarks are pretty meaningless until we complete a piece
of work"):** the user-turn row below is APPLIED (`mcp/tiers.py`
`USER_TURN_THINKING` = 20,480); the other rows stay as they are.

| where | before | proposal | why |
|---|---|---|---|
| user turn (`USER_TURN_THINKING`) | 12,288 | **20,480 (applied)** | their break-even budget for medium on code, and 2 of 3 of their medium misses still hit it; our answer room stays separate, so 20,480 of thinking is at least their 24,576 total. The nudge at 0.6 then fires at 12,288 -- today's hard stop -- so today's point becomes the "wrap up" signal rather than the cut |
| agent step (`AGENT_STEP_THINKING`) | 6,144 | **keep** | h4: median 128, p90 3,737 -- the cap binds on under ~10% of steps; their evidence is not about agent steps (their tool-call runs at medium spent 10-17k thinking before one `write_file` and they turn thinking OFF for agents, which our injection evidence rules out) |
| second-brain jobs (`JOB_THINKING`), compaction | 3,072-12,288; 2,048 | keep | nothing in their data speaks to them |
| answer floor (`A_MIN`) | 2,048 | keep | their 4,096 is the answer room left after a full think in a shared total; ours is separate and grows with the client's cap |
| server `--reasoning-budget` | 32,768 (breaker) | keep | only direct callers of llama-server see it; the proxy sets every request's budget |
| force-close text | "Thinking budget reached. I will stop deliberating and write the final answer now." | **keep ours** | ours: 2 of 2 clean closes, and the operator kept it over a variant (tiers.py); theirs is unmeasured as a text, and our nudge (0001) already opens the converging summary theirs lacks |

Measuring it (the operator's call): their `bench/humaneval_run.py` plates
through `:1234` at 12,288 vs 20,480, same seeds -- a cap value, not a
feature on/off.

### PHASE 2 (waits for the operator's "GPU free")

Scripted in one go: `bench/engine_phase2.py` (its docstring holds every
step, arm and gate rule):

    # after the operator's graft (above), once -- CPU only:
    python bench/engine_phase2.py --record-graft <models>/procreations-mtp/strip-check.gguf
    # on "GPU free": fit, corruption, speed, gates; nothing deployed
    python bench/engine_phase2.py --window --out bench/results/engine_corruption/ada-phase2-<date>
    # ... or all of it, and the deploy of the target only if every gate passes
    python bench/engine_phase2.py --window --out DIR --deploy --key-file PATH
    # re-read a finished run; run a subset: --skip fit corruption, --arms target-lean ...
    python bench/engine_phase2.py --gates DIR

Every arm runs on the 5060 Ti with production `bonsai` unloaded through
`bench/engine_corruption.py`'s window discipline (a quiet stack, the
worker's gpu lane paused, a guard that kills the arm the moment llama-swap
starts anything on the card, production restored in `finally`), one GPU
consumer at a time, with config.yaml's exact `bonsai` argv and env except
what the arm overrides. Arm overrides are `ARG:FLAG=VALUE` in
`engine_corruption.py --arm` (added for this).

0. **Fit** (`fit-lean.json`, `fit-pc.json`): the VRAM line N per head,
   measured with their margin (above, "The target").
1. **Corruption gate**: `current` (`790071e8`), `ada` (the new engine, our
   flags), `ada-vec` (decode back on the vec kernel, where 0035 acts),
   `target-lean`, `target-pc`. Cross-arm greedy text identity will NOT hold
   (the MMA and vec kernels accumulate attention in a different order, and
   0007/0010 change the mat-vec epilogue; the bundle's own identity matrix
   shows drafting on vs off differing at the rounding level on the MMA
   route) and is not a gate. What must hold: rep-to-rep identity on reps
   1..n (3/3 prompts: the PDL runtime check), 0/16 symptom runs, tools 10/10.
2. **Speed** (llama-server's own timings, n=3 per cell):
   depth (512 / 8k / 32k / 64k / 128k; the target arms also 196,608 and
   245,760, past today's pool), placement (#59: A at 8k / 32k / 64k with an
   idle 41k slot below / above, 3 x 40k above), spill (target: A at 8k while
   3 x 60k idle cells push the pool past N), concurrent (a shallow 8k and a
   deep 64k slot decoding together: the shallow slot's draft acceptance --
   the 0031 vs 0036 question), accept (draft acceptance through chat
   completions). Arms: `current`, `ada`, `ada-vec`, `target-lean`,
   `target-pc`, `target-window` (the target + `--spec-draft-window 16384`
   with 0036), and for attribution only: `ada-draft2`, `ada-dkv-q8`,
   `ada-np1`, `ada-np1-window` (their one-slot recipe: what the window buys
   at depth), `ada-window-0031` (51d64e6f: 0031's all-slot trim, 4 slots).
3. **Gates** (`gates.json`, `gates.md`): G0 the fit found an N; G1 the
   target (and `ada`, `ada-vec`) pass the corruption gate; G2 the target's
   decode >= 0.95 x current's at every depth current can hold; G3 the same
   in every placement cell; G4 >= 600 MiB free VRAM while each gated arm
   ran; the head: ProCreations only if recorded, G1-clean and faster (sum of
   the gated depth medians) than lean on the target. 0.95 is the live
   suite's own slots rule (`--tolerance`). Past-the-line depths, spill, the
   window arm and the attribution arms are reported, never gated.
4. **Deploy** (`--deploy`, only when G0-G4 pass): `config.yaml` backed up;
   in `bonsai`'s block `server_nudge` -> `92ffd4db`, `-c 262144`,
   `--kv-vram-cells N`, `-ctkd/-ctvd q8_0`, `--spec-draft-n-max 2`,
   `--spec-draft-n-max-tail 4`, `-m` -> the graft if it won (tested on the
   real config.yaml in memory: exactly those lines change); the new argv must
   equal the gated arm's; the engine and model checks must be clean; the
   stack restarts the watchdog's way (Scheduled Task `llama-stack`); then
   `scripts/deploy_check.py --key-file` must exit 0 -- that run IS every
   proxy dependency live (the slot release and idle clear with
   `resumed_cold`, the nudge, the decider's logprobs, tool calls,
   `/apply-template`, `/tokenize`, the served template's efforts). Anything
   else rolls `config.yaml` back and restarts again. After a deploy, by
   hand: the manifests' `config_refs` / statuses and this doc.

Rough duration: each speed arm is a load plus up to ~245k tokens of
prefill per depth row; the whole plan is several hours of GPU window.
`--arms` and `--skip` split it.

### Deployed (2026-09-27, the operator's quick path)

The long PHASE 2 speed arms were skipped by the operator's choice ("measure
speed on the pagoda run"); the gate was a fit, a smoke and `deploy_check`.

**What `bonsai` runs now** (config.yaml; backup `config.yaml.bak-20260927-233721`):
`llama-bonsai2-ada-92ffd4db`, the ORIGINAL trunk with the ProCreations head
(`Ternary-Bonsai-2-27B-PTQ1_0-mtp-procreations.gguf`, models/manifest.yaml
`bonsai-2-27b-mtp-procreations`; operator: "I don't know that I have seen any
performance gains with abliterated"), `--mmproj
Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf` (vision folded into the main model:
operator, "fold vision into the main models on the main card"), `-c 262144`,
`--kv-vram-cells 114432`, q8_0 K/V and draft K/V, `--spec-draft-n-max 2`,
`--spec-draft-n-max-tail 4`; everything else as before. `bonsai-vision` (the
A4000 copy) and its `ondemand` group are retired; the A4000's image
generation is untouched.

**The fit** (`bench/results/engine_corruption/ada-orig-20260927/fit-orig.json`):
their 1,000 MiB margin (no display on the card): N = 120,832 left 843 MiB,
115,968 left 971, 114,944 left 1,035 without an image; the smoke's image
requests took the minimum to 993 MiB (the projector's compute buffer), so
512 more cells came off: 114,432. The projector costs ~24.8k cells against
the build without it (139,264). Host RAM pinned for the tail: ~5.5 GB.

**The smoke** (`.../ada-orig-20260927/smoke/`): greedy reps 1..3 identical on
ts, bash and prose; decode median 74.42 tok/s (63.81-79.69, n=9); 0/4 plain
generations with a symptom (one ran into the 3,000-token test cap while
thinking); tools 5/5; an attached 256x256 PNG (left red, right blue) described
exactly "left=red, right=blue" 2/2; minimum free VRAM 993 MiB. Draft acceptance
on the same greedy prompts vs the abliterated trunk with the same head:
ts 0.80-0.81 vs 0.75, bash 0.748 vs 0.64-0.70, prose 0.56-0.57 vs 0.56-0.57.

**The gate**: `scripts/deploy_check.py` DEPLOY GOOD (exit 0):
`test_live_stack --live` 176/176, `test_tools_live --live` 35/35,
`bench/test_hint_collapse --live` 50/51 (the expected reranker failure,
FINDINGS #20). Live decode through :1234 (`x_yamadori.cache.decode_tps`, two
fresh sessions, effort low): 61.9 and 64.34 tok/s.

**The proxy side** (mcp/vision.py, image_input.py, proxy.py, tiers.py,
catalog.py, api_errors.py, gpu_room.py): `vision.main_sees()` reads the main
model's `/props` `modalities.vision`; a readable image in a user turn passes
to the main model as a data: URI image part with a label naming its id; every
other image is a placeholder; yama_describe_image asks the main model, and
without a projector answers VISION_NO_PROJECTOR (no second card). Prompt
estimates count an image as the projector does (`image_input.image_tokens`:
the served mmproj's patch 16 x merge 2, clip.cpp's QWEN3VL limits 8-4,096
tokens, mtmd-image.cpp's resize), never its base64. `main_sees` keeps its
answer for the process's life (the projector is a launch flag, and a deploy
restarts every service) and drops it the moment the server refuses an image.

### KV placement: `0038`-`0040` (2026-09-28, pagoda-h6)

**The failure.** pagoda-h6 (`octo/logs/pagoda-h6-pagoda-xhigh-1/relay.jsonl`,
`x_yamadori.cache`): decode 52-68 tok/s to ~110k, falling to 13.5 at 158k
(past the 114,432-cell line), then -- after Hermes compacted -- a NEW session
(a fork, `x_yamadori.session.forked_from`) on **slot 1** prefilled 67,211
tokens at 210 tok/s and decoded 6-12 tok/s from 67k to 119k, below the line,
for the rest of the run. Slot 0 still held the abandoned conversation
(176,223 cells, cleared by the idle clear 621 s later).

**Root cause (source).** Not the bundle's tier: how our 4-slot unified
pool places and spans cells, which their `-np 1` recipe never exercises.
(1) `llama_kv_cache::get_n_kv` padded the POOL's highest used cell, so every
attention op of every slot spanned `[0, pool max)`; 0028's staging
(`ggml_cuda_tier_stage`) copies every host-backed byte of that K/V view --
cells `[114,432, n_kv)`, every layer, every step -- whoever owns them.
(2) `find_slot` placed new cells from the search head (after the last cell
placed), so a conversation prefilled while another held the bottom of the
pool landed above it, and stayed there after the other was cleared.

**Reproduced live** (`bench/kv_placement.py`, bonsai's exact argv on the
5060 Ti; `LLAMA_KV_CACHE_DEBUG=1` prints the pool's highest used cell):

| stage | 92ffd4db (deployed) | 746c8af6 (0038-0040) |
|---|---|---|
| A: 160k on slot 0 (one conversation past the line) | 19.86 tok/s | see below |
| B: a new 60k on slot 1 while slot 0 holds the pool | **9.20** (pool max 220,426; prefill 233 tok/s) | see below |
| D: slot 1 continues after slot 0 is cleared (used 60,628) | **7.75** (pool max still 220,427) | see below |
| E: the same 60k from an empty pool (reference) | 64.11 | see below |
| F: idle 40k slot + active 8k / 32k / 60k | 58.55 / 54.04 / 55.83 | see below |

**The fix** (ours, `engines/patches/llama-bonsai2-ada/`):

- `0038` an attention op spans its ubatch's own sequences: per-sequence
  counts of each 256-cell block in `llama_kv_cells`, `get_n_kv(sinfo,
  ubatch)` pads the highest cell of the ubatch's sequences. Exact (only
  masked cells leave the view). `llama_memory_seq_span_p1()`; the server's
  tail draft size (0031/0037) keys on the slot's own span.
- `0039` a sequence grows in place (the cell after its own highest cell),
  else takes the lowest free cell (a hint, amortised O(1)).
- `0040` tiered caches only: before a batch whose span would pass the VRAM
  line, its cells move to the lowest cells (get_rows/set_rows over a raw
  32-bit alias: bytes, not values) and idle sequences' cells move to the
  lowest empty cells just above the new span. Sequences in any of the last
  `n_seq_max` batches never move (draft decodes run per slot).

**Why moves only past the line** (the first build, 979cc902, moved at any
span gain): the gate's greedy reps stopped repeating (0v1 / 1v2 differ at
148-1097 characters on all three prompts) while two fresh launches of the
same binary were identical 9/9, and `LLAMA_KV_PROMOTE=0` restored identity
3/3 -- cell moves reorder the cells an attention op reads, which changes the
rounding of its sums, and small workloads churned idle cells to the top of
the pool and back. Restricted to the one moment a move pays (the span would
cross the line) and with evictions landing just above the span, the gate's
workloads never move a cell.

Switches: `LLAMA_KV_SPAN_BY_SEQ=0`, `LLAMA_KV_PLACE_LOW=0`,
`LLAMA_KV_PROMOTE=0`; `LLAMA_KV_PROMOTE_VERIFY=1` reads moved rows back
(979cc902: 211/211 byte-identical).

**Flash-Next (phase 2, not started).** The upstream base (4da63377) has the
same `find_slot` / `get_n_kv` / `init_update` structure, but 0038-0040 do not
apply as they are (their context includes 0028/0037, and qwen4exp's memory
classes -- hybrid-idx, msa -- need `set_next_batch` / `seq_span_p1`
forwarding): a rebased pair per base. The expert side is the same principle
for weights: Flash-Next's static `--n-cpu-moe` split keeps routed experts in
RAM; the analogue is a VRAM expert cache whose promotions happen at
deliberate moments (Strata's `expert_cache`, upstream draft PR #27861).
Strata's code may be used (operator, 2026-09-28) as separate, attributed
patches. Now: section "Strata's MoE work, ported" below.

### KV rank: `0041` and the three-slot layout (2026-09-28, BUILT, NOT RUN ON A GPU)

**The operator's layout** (after pagoda-h6): always run from VRAM; the main
cap near the measured VRAM line, not 262k; 3 slots = two conversation slots +
one child slot (the second brain, the decider, side calls, as-sent
compactions); the child 64k+, swapped into VRAM while its conversation's main
pauses; every conversation advertised the full cap; "the first conversation
never spills; if that is not possible, either spilling under pressure is
fine". Concurrency is rare (relay data: at most 2 at once).

**Why 0040 is not enough.** Every sequence is equal there: one that ran in
the last `n_seq_max` batches never moves, anything else may. So (1) a second
conversation that arrives while the first is between two requests takes the
first one's VRAM; (2) the child cannot take its paused main's cells for 3
batches (it spills meanwhile), nor the main its finished child's; (3) two
running conversations both claim the lowest cells.

**How rank reaches the engine: a per-request field, not the slot id.** Slot
order cannot express it: the child's rank must follow the conversation it
works for (the primary's child keeps VRAM, the secondary's spills with it),
and primacy follows activity, which cannot move a conversation between slots
without a re-prefill. So `0041` adds, in llama-server's completion schema:

- `kv_rank` (int >= 0, default 0; higher keeps VRAM): the slot's rank from
  its task's launch, KEPT after the task ends (an idle primary keeps it);
- `kv_ranks` (array by slot id, < 0 = unchanged): the OTHER slots' ranks, set
  at the same moment, so a slot nobody uses is demoted by anyone's request;
- an ACTIVE flag per sequence: set at task launch, cleared at release
  (`server_slot::set_kv_rank`); `llama_memory_seq_set_rank(mem, seq, rank,
  active)`, forwarded by the hybrid memory to its attention cache;
- `/props` reports `kv_vram_cells` (the line), which `mcp/budget.py` reads
  as the main cap.

**The plan (`plan_moves`).** The batch's sequences are grouped by rank,
highest first; the first group whose span would pass the line and that can
gain makes the plan (the next batch plans the next group). Against a group of
rank r a sequence KEEPS its cells if its rank is higher (running or idle), or
equal and active; everything else may move out: a lower rank even while it
runs, an idle sequence of the same rank (the main while its child runs, the
child after). Exchanges (the evicted cell lands where the claimant's was,
deepest in the host tier) take the lowest rank first, idle before active; the
cells left over move just above the new span, the highest rank first. A
sequence never told keeps 0040's recency rule. `LLAMA_KV_RANK=0` = 0040
exactly. Moves still happen only when a span would pass the line.

**What placement cannot fix.** Two conversations decoding at once share ONE
`llama_decode` per step (and, with `split_equal`, one ubatch), so a spilled
lower-rank sequence's host staging lengthens every step the primary waits
for; splitting them into separate ubatches would only add a weight pass. 0041
keeps the primary's own cells in VRAM (it runs at full speed alone and after
the other finishes); protecting its steps while the other decodes spilled
would take a scheduling rule (the lower rank decoding only every k-th step,
or waiting while the primary generates) -- an operator decision, not built.
Concurrency costs without any spill too: two 8k slots decoding at once ran
18.02-18.15 tok/s each vs 64-83 alone (`bench/results/kv_placement/`
after-746c8af6 and concurrent-20260928b, G rows, n=1 each).

**The proxy side** (`mcp/slots.py` THE RULE and RANKS, `mcp/budget.py` THE
CAP LAYOUT): the child slot is n-1 (`helper_slot()` and the transient slot
are it; requests queue there at the server); conversations live on 0..n-2;
levels 2 (the primary: the conversation live longest), 1 (another live
conversation), 0 (left, empty, or a side call for no conversation); a fork
or an adopted key takes over primacy; `PRIMARY_HOLD_S` = 60 s (derived from the relay
bounds -- above every in-task gap, max 38.2 s of 482; below every
compaction-spanning gap, min 81 s of 11 -- operator-accepted 2026-09-28;
one conversation at a time is the typical use, and the RAM spill tier
stays);
`YAMADORI_KV_RANK=0` sends no fields. Budget: main = the cap (the served
line, or `YAMADORI_MAIN_CAP`), the child `YAMADORI_CHILD_TOKENS` (65,536)
not taken from main, one request's window (a compaction's included) = the
cap, advertised `context_length` = the cap; -c = 2 x cap + child.

**Built**: `scripts/build_engine.py llama-bonsai2-ada` ->
`C:/Users/jwals/engines/llama-bonsai2-ada-80d2c60d` (all patches applied,
all 208 CMakeCache options equal, build-info byte-identical, engine tests
2/2; llama-server.exe sha256 0aea2925...). **Not run on a GPU.** PHASE B:
`bench/kv_rank.py` (the line with the draft window and the projector, the
cap alone, the second conversation, the child swap at 49k/64k/98k, the
corruption gate), then `bench/deploy_kv_rank.py --line N` (`--preview` shows
its edits without writing).

**PHASE B, measured 2026-09-29** (`bench/results/kv_rank/20260929b/`, the
0041 build with config.yaml's `bonsai` argv plus `-np 3 --kv-unified
--spec-draft-window 16384 --kv-vram-cells N -c 2N+65536`, q8_0 draft K/V; each
decode figure a median of 3 greedy 256-token reps, n=1 run):

- **`--kv-unified` is required.** The first attempt (`20260929/`) passed
  `-np 3` alone: llama-server then runs 3 separate streams (`kv_unified =
  'false'`, `n_ctx_slot` = -c/3), 0028 does not tier a multi-stream cache,
  the whole pool was device memory oversubscribed into system memory (21-29
  MiB free at N 147,456 and 119,552; prefill ~95 tok/s, decode 3.8 tok/s).
  `bench/kv_rank.py` and `bench/deploy_kv_rank.py` now add `--kv-unified`.
- **The line: N = 141,824** (their 1,000 MiB margin, display inactive; round
  0 N=147,456: 805 MiB min free; round 1 N=141,824: 1,001 MiB), with the
  projector's image warm. -c 349,184; host RAM pinned for the tail ~7.2 GB.
  Below the operator's ~160-186k: the projector costs ~24.8k cells (the
  2026-09-27 fit, above).
- **Alone at the cap** (slot 0, span <= N, no moves): decode 69.1 tok/s at
  8k, 59.8 at 64k, 55.7 at 140,544 (the operator's live Hermes conversation
  at 156k with the old line decoded ~13.5).
- **Two conversations**: both below the line, decoding at once, 34.0 / 34.0;
  the second spilled (A 133,632 at rank 2 kept its span <= N, B's reached
  167,168): A alone with B idle 57.1, at once A 17.9 / B 18.7 -- B's host
  staging lengthens the shared step, as predicted. The second arriving
  FIRST: A's prefill to 140,544 moved B out (26 moves, 622 ms), A ended at
  or below the line, decode 56.8.
- **The child swap** (A at N-4k idle at rank 2, the child at rank 2 on slot
  2, then A continued): the child's cells claim VRAM a ubatch at a time
  (1,024 of A's cells out per batch, 12-25 ms each), and A takes it back in
  one move on its next request (49k: 3.3 GB in 256 ms; 64k: 4.4 GB in 345
  ms; 98k: 6.5 GB in 564 ms). A's decode after: 57.5 / 59.3 / 60.0 vs 59.4
  with no child; its 257-token continuation's prompt 1,119 / 1,203 / 1,420 ms
  vs 854. The child decoded 58.2-62.1. One check failed on a technicality:
  in the 64k stage A's span reached 142,080 (one 256-cell block past the
  line) during its own re-prefill, before the move that brought it to
  137,728.
- **Corruption gate** (`engine_corruption.py --vision` on the arm): greedy
  reps 1..3 identical on ts, bash and prose; 0/16 plain with a symptom;
  tools 10/10; decode 75.27 tok/s median (62.55-80.29, n=9).

### Layout v2: vision off the main card, the lane, one model (2026-09-29, PREPARED OFFLINE, NOT DEPLOYED)

**The operator's decisions** (2026-09-29): (a) "The point is to get more
context at speed in vram, so decider was the only thing that needed room."
(b) "Vision can go to second card and swap in and out." -- reversing
2026-09-27's fold of vision into the main model. (The single-model decision --
the second brain removed -- is a separate removal, not part of this deploy.)

**What changes** (the deploy is `bench/deploy_layout_v2.py`; nothing below
runs until it does):

- `bonsai` loses `--mmproj`. The 2026-09-27 fit put the projector's cost at
  ~24.8k cells of the line (114,432 with it vs 139,264 without, the fit
  above); those cells go back to the KV line. The new line is measured, not
  added: `bench/kv_rank.py --layout v2` (its default) fits without the
  projector and with the lane.
- **Vision on the A4000 again**: `bonsai-vision`, the retired entry
  (config.yaml.bak-20260927-233721) on the ORIGINAL trunk the main model
  serves (`Ternary-Bonsai-2-27B-PTQ1_0.gguf` + the Q8_0 mmproj, `${server}`,
  `-c 16384`, ttl 300) and its `ondemand` group (swap, exclusive), loaded on
  demand; `mcp/gpu_room.py` makes room (SIZES `bonsai-vision` 9,449 MiB, the
  estimate it had: never measured alone). `mcp/vision.py` asks it whenever the
  main model's `/props` has no `modalities.vision` -- the pre-fold route --
  and the main model when it has one (max mode). Attached images are
  placeholders naming their id again (`vision.normalise`, `see=False`).
- **The lane.** The child slot (n-1 of `-np 3`) serves the decider and small
  side calls (titles) only. It is NOT released after a decider turn or a side
  call (the releases cost median 395 ms, p90 824 ms each: logs/proxy.log, the
  last 200 decider turns), and every request ranks it `slots.RANK_LANE` = 3,
  above the primary's 2, so 0041's `plan_moves` never moves it out of VRAM
  (a sequence of a higher rank keeps its cells). Its size,
  `budget.LANE_TOKENS` = 3,072: the slot holds one request at a time, so the
  largest one -- the decider's state is capped at `STATE_TOKENS` 2,048 by
  construction, and the rest of its prompt measured at most 999 cells (the
  200 turns: median 837, p90 1,609, max 3,047 cells) -- 2,048 + 999 rounded
  up to whole 256-cell blocks (0038's unit) = 3,072; side calls measured
  median 439, p90 614, max 1,682 cells (n=69). **What it costs the main
  line**: 3,072 cells, taken off the cap (`budget.main_cap`: YAMADORI_MAIN_CAP
  = N - 3,072 written by the deploy, else `kv_vram_cells` less the lane). The
  rank is not measured yet (PHASE B ran ranks 0-2): `kv_rank.py`'s `lane` step
  checks both spans stay at or below the line with nothing moved, in both
  orders, and times a lane request after a kept lane vs after a release.
- An as-sent compaction (unmapped; 150,659 tokens in the proxy log) no longer
  fits the lane: it goes to the least recently used CONVERSATION slot
  (coordinator: a cache miss beats a failed compaction), recorded in its
  grant (`how`, `displaced`), and that slot is emptied after it; the
  displaced conversation's next request reports `resumed_cold`
  (`by: a compaction sent as is`).
- `-c` = 2N + 3,072. `/props` `n_ctx` is the SLOT's window, which
  llama-server caps at the trained 262,144 (server-context.cpp:1203-1206 in
  the 0041 build), so the fixture's intended `n_ctx` is min(-c, 262,144):
  deploy_kv_rank's `as_intended` compared it with -c (349,184 vs 262,144).
- A second-brain job, while that machinery exists, runs on the lane's slot
  at its conversation's rank with its old 65,536 window, which -c no longer
  reserves (beside two conversations at the cap it has no room).
- `budget.budgets()["child"]` names the slot for the dashboard: role
  "decider lane", 3,072 tokens, what it serves, kept, ranked, rank 3.

**Commands** (in order; each waits for the operator's "GPU free"):

    python bench/kv_rank.py --window --out bench/results/kv_rank/<date>-v2
        # prints THE LINE TO USE: N
    python bench/deploy_layout_v2.py --line N --preview   # the full diff
    python bench/deploy_layout_v2.py --line N
    python scripts/deploy_check.py --key-file PATH

### Layout v3: one conversation + the jjava lane (2026-09-29, BUILT OFFLINE, NOT DEPLOYED)

**The operator** (2026-09-29, verbatim): "we should be doing that for all models, we should not have a second
conversation at all, it is too slow, we have a second gpu if we want a second conversation, that is how it has to
play out, the jjava engine should be the only other thing we need ready to go, and if that also makes it dog shit
slow, then we just use jjava bansai and put it on the other gpu, done and done." And: "We clear jjava lane too after
it is done right, not slow down slop." **Evidence**: Flash-Next decoded an 8K conversation at 8.6 tok/s beside an
idle 64K slot (the unified pool's top-cell cost, #59; the same shape measured on Bonsai).

**The numbers (derived; each is re-measured before it ships):**

| model | layout v3 | derivation |
|---|---|---|
| `bonsai` | `-np 2 --kv-unified`, `-c` = N, `--kv-vram-cells` N, main cap N - 3,072 | v2's line N = 170,496 was the VRAM-resident cells (`bench/results/kv_rank/20260929-v2/line.json`: 36,992 B/cell, min free 1,017 MiB at the 1,000 MiB margin); the cells above N (`-c` 344,064) were host RAM for a second conversation (~6.04 GB pinned) and go. One slot fewer frees one slot's DeltaNet state -- 48 GDN layers x 48 heads x 128 x 128 x f32 = 144 MiB + ~6 MiB conv -- = ~4,250 cells, so N ~ 174,592 and the cap ~ 171,520 (today 167,424). MEASURED by `bench/kv_rank.py --window --layout v3` (the fit at `-np 2`, `-c` = N) before `bench/deploy_layout_v3.py --line N` |
| `mirai-s` | `-np 2`, `-c` = the gate's fit, cap = fit - 3,072 | `bench/mirai_s_gate.py` fits at `-np 2` (36,992 B/cell: 17 attention layers x 4 KV heads x 256 x K+V at q8_0) |
| `flash-next` | `-np 2`, `-c` 262,144 (cap 259,072); 265,216 (cap 262,144, the full trained window) only if its fit confirms | 3,072 cells x 13,056 B = +38 MiB, against two dropped slots' recurrent state; the Flash-Next agent's fragment |

**The proxy** (`mcp/slots.py` ONE CONVERSATION, on when `/props` says 1 or 2 slots): the owner keeps slot 0; another
conversation is 503 `conversation_at_capacity` + Retry-After; the owner's compactions and as-sent compactions run on
slot 0 (never refused for being compactions, never the lane); the lane is released after every burst (`LANE_KEEP`
False; why `lane burst ended`). AGENTS.md "Layout v3".

**The lane's three arms** (per model, n=3 at 8K / 32K / 64K; `bench/kv_rank.py --layout v3` step `lane3` for Bonsai,
`bench/mirai_s_gate.py` step `lane` for Mirai S): main decode tok/s with the lane KEPT (a 3,071-token state placed
after the conversation, so above its cells), ACTIVE (a 10-read burst during main's decode) and CLEARED after its
burst, plus the next burst's re-prefill ms. MATERIAL: an arm's median below the CLEARED arm's minimum, with the %.
If the lane materially slows main, jjava moves to `bonsai-a4000` and the main card runs `-np 1`.

**MEASURED 2026-09-30 (the Bonsai window; one run each):**
- `bench/kv_rank.py --window --layout v3` (bench/results/kv_rank/20260930-v3/): **N = 209,920** at `-np 2` (rounds
  147,456 -> 3,143 MiB free, 208,128 -> 1,069, 209,920 -> 1,009 at the 1,000 margin): `-c` 209,920, main cap 206,848.
  The lane's arms, main decode tok/s medians (n=3): 8K kept 73.8 / active 51.3 / cleared 72.5; 32K 68.5 / 52.4 /
  72.7; 64K 71.3 / 51.0 / 73.0; the next burst's re-prefill 984-1,025 ms. MATERIAL: ACTIVE at every depth (~-30%),
  KEPT at 32K/64K (-6% / -2%).
- `bench/kv_q4.py` (bench/results/kv_rank/20260930-q4/): q4_0 K/V at the full 262,144 (`-np 2`) leaves 3,319 MiB
  free; decode q4_0 vs q8_0 73.9 / 73.3 (8K), 73.5 / 72.4 (64K), 57.9 / 55.9 (128K); top-20 approximate KL(q8||q4)
  mean 0.0156, max 0.174 over 128 reads, top-1 agreement 125/128 (no q8-vs-q8 baseline in this run); needles 5/5 at
  32K and 128K on both.
- `bench/a4000_fit.py` (bench/results/a4000/20260930/): `bonsai-a4000` fits **-c 141,312** (cap 138,240) beside
  embeddings + the reranker, 35,840 B/cell, fixed 10,193 MiB; alone it reads prompts at 739 tok/s and decodes at
  44.9 on the ada engine (prism: 400 / 33.6). gpu_room's row is now measured (10,893 MiB).

**DECIDED 2026-09-30 (operator, verbatim: "1. Kv8 2. Yes 3. Yeah, it compacts on the same card it came from
right? To get cache gains."):**
1. Bonsai's KV stays **q8_0**; q4_0 at 262K is a measured option (above), not deployed.
2. Bonsai is **LOCKED at `-np 1`** like Flash-Next (the ACTIVE arm's ~-30% above): its jjava and side calls run on
   `bonsai-a4000` (`mcp/tier_models.yaml` bonsai `locked: true`, `helpers` bonsai-a4000); no lane on the main card.
   `-c` = N = 209,920 (measured at `-np 2`; at `-np 1` the unified pool is still `-c` cells and one slot's recurrent
   state is freed, so it is SAFE, a few thousand cells under the `-np 1` maximum -- a short `-np 1` fit may raise
   it), main cap = N, `YAMADORI_LANE_TOKENS` 0. `bench/deploy_layout_v3.py --line 209920` (default `--np 1`) after
   `bench/deploy_tier_models.py` (it refuses until bonsai-a4000 is in config.yaml and YAMADORI_TIER_MODELS set). The
   decider lane lives only on `bonsai-a4000` (its own slots; the proxy sends no slot id there,
   `slots._helper_server_grant`). Mirai S: its own lane step decides, by the same rule.
3. **Compactions stay on the card of the conversation they summarise**, on its own slot -- never `bonsai-a4000`,
   even at max or while the card is locked (`max_mode.decide(kind="compaction")`, `touch_allowed("compaction")`).

**`bonsai-a4000`** (the second conversation's and jjava's home; fitted, not deployed): the original trunk on the
A4000 by UUID, `-np 2` (conversation + lane), in `ondemand` swap with `bonsai-vision` and imagegen, embeddings and
the reranker kept. Its room is an ESTIMATE until a fit: 16,376 MiB - embeddings 2,100 - reranker 3,000 (both
gpu_room ESTIMATES) - headroom 1,331 = ~9,945 MiB; less the weights 6,093 MiB and a 1-2 GiB compute buffer (the
bonsai-vision row's assumption) leaves ~1,850-2,850 MiB of KV = ~52K-80K cells at 36,992 B. Until it is deployed a
second conversation is refused as above.

## Strata's MoE work, ported: `llama-upstream-moe` (2026-09-28, CPU only)

Operator, 2026-09-28: "the whole [point] is that we get some of the
improvements with other models we run too. So it is worth faithfully porting in
to keep our own version." Source: Niko1221/Strata `d551edf42c157f1c8c20f07f68c5dfe506b1edda`
(21 commits after `b742ff99`; no licence file -- used with the operator's
authorisation, every port a separate patch naming its source commit and
file:line). Nothing here has run on a GPU.

**What Strata's "2x" is.** Prompt reading, not decode, on Strata's own engine
against its own 0.1.12 (`928b0e07`, engine 0.1.13;
`bench/results/2026-09-28-prefill-speed/README.md`): 32K-token prompt, Q2_0
572 -> 1,290 tok/s, IQ3_S 383 -> 1,208 (RTX 5070 12 GB, Ryzen 5 7600, one
prompt, one run per step); "Output speed is unchanged (the decode path is the
same)". The 0.1.14 matrix (`bench/results/2026-09-28-speed-0114`, one prompt per
length, 256 tokens, MTP on) gives IQ2_XS prompt 461-1,238 tok/s and output
74.4 / 71.5 / 64.3 / 59.8 / 52.8 tok/s at 1K / 32K / 64K / 128K / 262K, with
3,565-4,386 experts resident; prompts "1.2x (1K) to 2.4x (32K-128K)" faster than
0.1.12. No llama.cpp baseline anywhere in their results.

**Strata's steps (their table), and what each is in llama.cpp:**

| Strata step (928b0e07 unless noted) | their measured effect | llama.cpp at `4da63377` | port |
|---|---|---|---|
| MMQ kernels for experts in prefill instead of dequantize + cuBLAS | Q2_0 1052 -> 1130 | **already so**: `MUL_MAT_ID` runs ggml-cuda's MMQ (Strata compiles llama.cpp's own `mmq.cuh` into `strata_mmq`) | none needed; `test-backend-ops` checks the kernels |
| expert streaming ring: at chunks >= 2048 every non-resident expert streams in a fixed order so the next layer's arrive during attention | 1130 -> 1290 | op offload copies a layer's USED experts right before its `MUL_MAT_ID`, on the compute stream, not overlapped (`ggml-backend.cpp` `ggml_backend_sched_compute_splits`) | **0002** = upstream PR #28414 (rebased): full expert tensors one split ahead on a second stream into rotating staging buffers at prefill scale -- the same fixed-order, full-tensor lookahead, 1 split deep (Strata: a 384-slot ring) |
| prefill chunks up to 8,192, sized to the cache slots it may borrow (`--prefill auto`) | 791 / 878 / 973 at 4096 / 6144 / 8192 | `-b` / `-ub`; the compute buffer is reserved at start, it cannot borrow expert-cache slots | **configuration**: `-ub` swept in the gate (512 / 2048 / 4096); borrowing is not portable (llama.cpp's allocator reserves the worst case once) |
| attention and MoE halves share one scratch region | bit-identical, 1.78 vs 2.84 GiB at chunk 4096 | the graph allocator already reuses every intermediate by liveness | none needed |
| PLE block batched over the chunk; next chunk's PLE rows read on a thread | 981 -> 1053; 762 -> 790 | the graph is batched per ubatch already; the lazy PLE rows are gathered on the CPU in `set_inputs`, synchronously (qwen4exp / gemma4 only) | **not yet**: a prefetch thread in the lazy-rows reader (upstream #29030 rewrites that reader with direct reads, +65-121% prefill on Strix Halo; open). Qwen4exp/gemma4-specific |
| unpinned experts copied to pinned buffers by helper threads; the pinned arena | bit-identical; IQ3_S 551 -> 652 | under mmap (which `--lazy-mode` needs) CPU weights are pageable: the host buffer is swapped for the CPU one (`llama-model-loader.cpp`; upstream even warns) | **0005**: `LLAMA_PIN_EXPERTS=1` keeps experts in the pinned host buffer under mmap |
| Windows large-page arena (0bf3216e, pipeob0: SeLockMemoryPrivilege, GetLargePageMinimum rounding) | not measured alone | `cudaMallocHost` | **0005**: `GGML_CUDA_HOST_LARGE_PAGES=1` (needs the account's "Lock pages in memory" right, never changed by us) |
| `CUDA_MODULE_LOADING=EAGER` (66099605: a kernel first used mid-prompt found no VRAM for its code) | fixes an OOM; ~30 MB on their kernel set | lazy loading (CUDA 12.x default) | **configuration**: an env line; our `ggml-cuda.dll` is 92 MB of fatbin, so its eager VRAM cost is measured (gate arm `eager`), not assumed to be theirs |
| GPU expert cache: profile-selected residents, adaptive promotions (`include/strata/core/expert_cache.hpp`; `generate.cpp:2418-2457`) | decode (not part of the 2x) | none upstream; draft PR #27861 has the exact CPU/GPU split mechanism | **0003** = #27861 (rebased), **0004** = Strata's selection and promotion policy on it |
| AVX2 multi-token i-quant CPU kernels (ed14227c, pipeob0) | 48.5 vs 50.9 ms/round (-4.7%) on a 5700X3D + 5060 Ti | ggml-cpu's single-token `vec_dot` per row and token | **not yet**: ggml-cpu's `nrc` multi-column path would carry it; small gain on their numbers |
| the GPU's PCIe share (e6265c7: the GPU reads part of the missed experts straight from pinned RAM while the CPU computes the rest) | IQ3_S decode 45.3 -> 44.8 (it fixed a stall, not a speed-up) | none | **not ported**: needs a mapped-host MUL_MAT_ID path; large |

**The patches** (`engines/patches/llama-upstream-moe/`, applied in order to
`4da63377`; each checked with `git apply --check` in sequence):

| patch | source | what | default |
|---|---|---|---|
| `0001` | ours | the reasoning-budget nudge (llama-upstream's 0001) | - |
| `0002` | PR #28414 `b43a3c63` (leshchukandrej), rebased: one hunk by hand | `--prefetch-experts-slots N` | off |
| `0003` | PR #27861 `bccbacdb` (csantiago78), rebased: two hunks by hand; `flockfile` made MSVC-portable | `--moe-expert-cache N [--moe-expert-cache-inserts K]`: exact split, LRU | off |
| `0004` | Strata port | `LLAMA_MOE_CACHE_PROFILE` (STRP profile -> residents and per-layer slot counts), `LLAMA_MOE_CACHE_POLICY=strata` (the adaptive tier, Strata's constants), every CPU-computed batch (< `GGML_OP_OFFLOAD_MIN_BATCH`) instead of one token | LRU, no profile |
| `0005` | Strata port | `LLAMA_PIN_EXPERTS=1`, `GGML_CUDA_HOST_LARGE_PAGES=1` | off |

Generalised: 0002 and 0005 serve any model with CPU-placed experts, 0003/0004
any MoE with separate gate/up/down and SiLU (#27861's scope), with a profile
per model (`scripts/moe_profile.py` builds one from `GGML_MOE_LOG` traces or
Strata's `--dump-routing`, in Strata's STRP format; Strata's own
`data/expert-profile.bin` round-trips through it byte for byte). With every
switch off the build must be bit-identical to `llama-upstream` (gate arm
`moe-off`).

**Two claims in conflict, measured before either is believed:** Strata's
profile hit rate h = 0.6447 at 4,105 slots, ten-fold leave-one-out
(`expert_cache.hpp`), against #27861's "no exploitable static skew" (a top-32
per layer covering ~10% on its workload) with LRU-64 ~67%. The gate runs
`cache-lru` and `cache-strata` side by side.

**Applies to `llama-bonsai2-ada` (Bonsai, dense, all in VRAM)?** The expert
parts (0002-0004, the pinned experts) have nothing to act on. What carries
over: `CUDA_MODULE_LOADING=EAGER` (a VRAM-tight card; measure its cost first),
the `-ub` sweep for prefill, and the gate's quality checks (slot-state hash
for "exact" changes, teacher-forced KL for rounding changes). The large-page
buffer does not reach 0028's KV tail, which CUDA VMM allocates itself.

**KV placement (0038-0040) on this base: not ported yet, deliberately.**
Flash-Next's K/V fit in VRAM (no tier: 0040/0041 have nothing to act on), so
what remains is 0038's span (an attention op reading only its batch's
sequences' cells). On qwen4exp the QSA indexer's block tables are built from
`n_kv` (`llama-memory-hybrid-idx.cpp` `set_input_qsa`, where #27994 and
#29166 found multi-sequence bugs), so a changed span must be threaded through
them -- a real rewrite, not a rebase. The gate measures first: the `nounified`
arm (one KV stream per slot spans only its own cells by construction) against
the unified pool, with an idle 64K slot beside an active 8K one. If the
unified pool shows the #59 tax, the choice is `--no-kv-unified` (no code) or
a QSA-aware 0038.

**Gate:** `bench/flashnext_gate.py` (kernels, fit, exact, kl, needles,
corrupt, speed; `--dry-run`, `--selftest` 8/8).

### `llama-upstream-flash` (2026-09-29): the moe series + MTP + the CPU fixes

Why: docs/FLASH-NEXT.md section 8 (the 18 tok/s measured apart). The same base
(`4da63377`), twenty-three patches in `engines/patches/llama-upstream-flash/`, every
header `From: Justin Walsh`, the original authors credited in each body:

| patch | what | from |
|---|---|---|
| 0001 | reasoning-budget nudge | ours (llama-upstream's 0001) |
| 0002 | `--prefetch-experts-slots` | PR #28414, rebased |
| 0003 | `--moe-expert-cache` (the exact GPU/CPU split) | PR #27861, rebased |
| 0004 | expert-profile selection + adaptive tier | Strata, MIT (re-attributed at `3ce2523c`) |
| 0005 | pinned experts under mmap, large-page host buffer | Strata, MIT |
| 0006 | Qwen3.8-Flash-Next MTP | PR #28243's net diff |
| 0007 | the cache's multi-token slot lookup: fixes `GGML_ASSERT(a->ne[2] == b->ne[1])` | ours |
| 0008 | AVX2 `ggml_vec_dot_q2_0_q8_0` (x86 had the scalar generic only) | ours |
| 0009 | multi-token expert rows for the CPU `mul_mat_id` (`GGML_CPU_MMID_MT`) | Strata `iq_avx2.cpp` (ed14227), MIT; Q2_0 rows ours |
| 0010 | `GGML_CUDA_OP_TIMING=1`: per-op GPU time (diagnostic, off by default; its table prints at `-lv 4`) | llama-bonsai2-ada's 0033 (Cary Palmer, MIT) |
| 0011 | CUDA top-k by radix select when CUB has no DeviceTopK (CCCL < 3.2) | ours |
| 0012 | `LLAMA_KV_HOST_MAPPED=1`: the attention K/V in pinned, device-mapped host memory | ours (Strata's KV-streaming idea, no Strata code) |
| 0013 | the expert cache only on batches `mul_mat_id` serves with MMVQ: fixes the cache's CUDA illegal memory access | ours |
| 0014 | the sparse (QSA) flash attention converts only the selected q8_0 cells to f16, not the whole cache | ours |
| 0015 | a quantized cache's 1-2 query decode takes the sparse MMA kernel, not the vector kernel that reads every cell | ours |
| 0016 | the expert cache's hit-rate line at INFO every 64 steps (it never printed under the strata policy) | ours |
| 0017 | `GGML_SCHED_TIMING=1`: a scheduler graph's wall time split into waits, CPU compute, launches, copies (diagnostic) | ours |
| 0018 | 0014's selected-cell conversion for a q4_0 cache too | ours |
| 0019 | `LLAMA_QSA_BLOCK_TOPK=1`: QSA's budget selected as whole blocks + the tail, no n_kv-sized op (M2b; not in the series yet) | ours |
| 0020 | `LLAMA_GRAPH_CACHE=N`: one graph per verify-batch size, so CUDA graphs replay under MTP (not in the series yet) | ours |
| 0021 | `LLAMA_KV_HOST_MAPPED=1` maps only the sparse (QSA) layers; the MTP draft's dense layer stays on the card (not in the series yet) | ours |
| 0022 | `mul_mat_id`'s MMQ pads src1 for the tile width it picks, not by ne11: fixes a fresh server's illegal memory access on a 508-token ubatch (upstream bug at the base; applies on 0018, independent of 0019-0021) | ours |
| 0023 | `--checkpoint-every N`: llama-server also makes a context checkpoint every N prompt tokens, so a prompt that differs inside a long system + tools block resumes from the last one before the difference instead of from 0 (off by default; Strata's N is 16,384; not in the series until its live check) | ours |

Operator, 2026-09-29: "I approve strata engine source patches"; "Port kernels we
are not re-writing everything from scratch". Strata's MIT notice:
`engines/patches/llama-upstream-flash/LICENSE.strata` (blob `6687a423`), also
inside 0009's source file. The development tree is
`C:/Users/jwals/engines/dev-flash/src` (branch `series`, one commit per patch);
the nine files applied to a clean `4da63377` give exactly that tree (checked).

Offline, CPU only (`C:/Users/jwals/engines/dev-flash/tools/moe_cpu_bench.cpp`, the
CPU half of one Flash-Next MoE layer, 16 threads, median of 300 steps, the live
stack running): 0008 takes the down projection 0.575 -> 0.118 ms and the layer
0.87 -> 0.42-0.48 ms; 0009 is within the noise to ~10% on 2-4 token batches
(Strata measured -4.7% per round on a 5700X3D); its single-token rows were
SLOWER than ggml's (0.62 vs 0.42 ms), so single tokens stay on ggml's dot by
default. Both match the stock kernels to relative L2 <= 1.7e-7. (0010-0014 were
exported from the same tree, one commit each.)

0013 and 0014 (2026-09-30, window B of the Flash-Next gate; docs/FLASH-NEXT.md
section 8). 0013: the #27861 cache maps every uncached expert of a token to one
zero slot, so a token's slot ids repeat; MMVQ is exact with repeats, MMQ's ids
helper (`mmid.cu mm_ids_helper`) counts one row per (token, expert) and leaves
inverse-map rows unwritten -- the fault `CUDA_LAUNCH_BLOCKING=1` placed at
`mmq.cu:291`. The cache chain is now built only up to MMVQ's `mul_mat_id` batch
for the cached types (a copy of `get_mmvq_mmid_max_batch`'s tables).
0014: the sparse MMA kernel needs f16 K/V and reads only the listed cells, but a
q8_0 cache was converted whole every call (`fattn-common.cuh`, `to_fp16` over
`ggml_nelements(K)`): a decode cost linear in the context, and every cell over
PCIe with 0012 (fit-kvmap: 4K decode 9.3 tok/s). Now the listed cells only, after
the lists exist, into the same dense f16 layout; test-backend-ops gains eval cases
(the QSA shape, q8_0 and f16, 1 and 4 queries, top-k 2048).

## Vendored source (2026-09-29)

Operator, 2026-09-29: "why is our engine source not in this repo, only our
patches are, this needs to be reproducible by others on other machines" and
"We should maintain a copy of the source vendored that we use, then we should
be able to regenerate, update, and continue patching upstream as well".

The source every engine is built from lives in this repo, under
`engines/src/<engine>`, so a rebuild depends on nobody's fork staying up.
The repos we PORT FROM (Strata at its commits, upstream PRs, professorpalmer's
bonsai-ada-surgery, sudoingX's PR) are not vendored: they stay pinned in each
entry's `patch_source`, and what we took from them is in our patch files.

| vendored | base | patches | what runs it |
|---|---|---|---|
| `llama-bonsai2-ada` | PrismML-Eng/llama.cpp `adfffbe41b2c` | 41 | `bonsai` |
| `llama-upstream` | ggml-org/llama.cpp `4da6337767f9` | 1 | `flash-next` (config.yaml `server_upstream_moe`) |
| `llama-upstream-moe` | ggml-org/llama.cpp `4da6337767f9` | 5 | the Flash-Next fragment's arms |
| `llama-mirai-s` | alesha-pro/llama.cpp-mirai-s `b59ae80f419a` | 1 | config.mirai-s.fragment.yaml |
| `llama-prism` | PrismML-Eng/llama.cpp `9a9394a895b9` | 0 | vision, embeddings, critic (the reranker removed 2026-10-01, docs/REMOVED.md) |
| `llama-bonsai2-base` | sudoingX/llama.cpp `285542d98d37` | 0 | `bonsai-q4kv`, `bonsai-q4kv-196k` |
| `sd-cpp` | leejet/stable-diffusion.cpp `c92d73c40851` + 4 submodules | 1 | `imagegen`, `imagegen-turbo` |

Not vendored: `llama-bonsai2`, `llama-upstream-mtp` (nothing runs them;
`vendor` takes them any time), `llama-upstream-flash` (in development in
`engines/dev-flash`; vendor it when its series lands:
`build_engine.py vendor llama-upstream-flash`), and `llama-swap` (below).

**What a tree is.** The base commit's tree, submodules read in at their
pinned commits, WITH THE PATCH SERIES APPLIED (through a git index, exactly
as the fetch build applies it), minus what the exclusion rule drops. Applied,
not applied-at-build, because the tree is then the exact source that is
compiled: it can be read, grepped and diffed as it runs, and `build` needs
nothing but it. The series stays the record of our changes: it is what
`vendor` applies, what `update` rebases, and what `check --derive` takes
back OUT of the tree (offline, `git apply -R`) to land on the recorded base
-- so the tree cannot drift from base + series without a check failing.
Each file is the upstream blob's bytes (LF where upstream is LF), and
`.gitattributes` marks `engines/src/** -text` so git never converts them.

**The record.** Each vendored entry has a `vendor:` block written by
`vendor` (never by hand): `repo`, `commit`, `upstream_tree` (the commit's
tree id on GitHub), `submodules` (sd-cpp), `patches_digest` (the series'
names and sha256s), `patched_tree` (the git tree of base + series before the
exclusion), `base_tree` and `tree` (git tree ids over the vendored files,
every file as mode 100644: `tree` is exactly what `git rev-parse
HEAD:engines/src/<engine>` gives once committed from Windows -- checked for
all seven), sizes, `licences` (every licence/notice file kept), and
`build_info` (below). `executable_files` counts the upstream files that are
100755; Windows commits them 100644, and no step of the CMake build runs one.
`eol_converted` lists the files a nested upstream `.gitattributes` still
converts on checkout (sd-cpp's libwebp `*.bat text eol=crlf` outranks our
root file); `check` hashes those as git stores them (LF).

**The exclusion rule** (`vendoring.rules` in the manifest; last match wins,
anchored at the tree root; a licence or notice file is always kept):

- llama.cpp trees: `.github/ .devops/ docs/ media/ benches/ models/
  !models/templates/` -- CI and packaging, the op-support CSVs (39 MB) and
  images, benchmark logs, and the tokenizer test vocabularies
  (`models/ggml-vocab-*.gguf` + `.inp`/`.out`, ~80 MB, read only by
  `test-tokenizer-*`, which no entry builds). `models/templates` stays:
  `test-chat` reads it. 229-237 files, ~117-121 MB per tree.
- sd.cpp: `.github/ docker/ docs/ assets/` -- CI, packaging, docs, and the
  README's example images and videos (191 files, 106 MB). The tokenizer
  vocabularies in `src/tokenizers/vocab/*.hpp` (146 MB) are compiled in and
  stay.

**The cost, measured** (2026-09-29: the seven trees added in order to a
scratch repository with `git add -f`, `git repack -a -d -f` after each, git's
default compression):

| added | pack after | added by it |
|---|---|---|
| llama-bonsai2-ada | 11.2 MiB | 11.2 MiB |
| llama-upstream | 12.3 MiB | 1.1 MiB |
| llama-upstream-moe | 12.3 MiB | 16 KiB |
| llama-mirai-s | 12.4 MiB | 72 KiB |
| llama-prism | 12.4 MiB | 20 KiB |
| llama-bonsai2-base | 12.4 MiB | 4 KiB |
| sd-cpp | 37.7 MiB | 25.3 MiB |

Six llama.cpp trees cost 12.4 MiB together (identical blobs are stored once,
the rest delta against each other); sd.cpp's vocabulary headers are most of
its 25 MiB. `--window=250 --depth=50` gives 37.5 MiB. The largest file is
`sd-cpp/src/tokenizers/vocab/umt5.hpp`, 45.7 MB (under GitHub's 50 MB
warning). The working tree grows by ~530 MB. THIS MACHINE's global git
config sets `core.compression 0`: a push from it sends the pack uncompressed
(~195 MiB measured); push with `git -c pack.compression=9 push` (or unset
it) to send ~38 MiB. GitHub stores its own repack either way.

**llama-swap is not vendored.** It is the upstream release binary (not a
fork), pinned by URL and SHA-256. Building it from source at v256 needs Go
1.27.1, ~70 Go modules (tailscale.com and modernc.org/sqlite among them) and
an npm build of its React UI, which `//go:embed all:ui_dist` compiles in
and which is not committed upstream: vendoring that means `go mod vendor`
plus a built UI, both downloads beyond pinned git commits. If the release
asset ever matters, attach the pinned zip to one of this repo's GitHub
Releases instead.

### The commands

    python scripts/build_engine.py vendor <engine>             # regenerate from the manifest
    python scripts/build_engine.py vendor <engine> --check-upstream   # regenerate in a temp dir, compare, write nothing
    python scripts/build_engine.py check [<engine> ...]        # offline: hash, series, licences, rule
    python scripts/build_engine.py check <engine> --derive     # + take the series back out, require the base
    python scripts/build_engine.py build <engine> --jobs 8     # build from engines/src, NO network
    python scripts/build_engine.py build <engine> --configure-only
    python scripts/build_engine.py update <engine> --to <sha>          # dry run
    python scripts/build_engine.py update <engine> --to <sha> --write

(`scripts/engine_vendor.py`; the fetching build, `build_engine.py <engine>`,
is unchanged.)

- **`vendor`** fetches the base (and submodules) at depth 1 into
  `<engines_root>/.vendor-cache` (outside the repo; one object store for all
  forks), applies the series in a temp index, writes the files, checks the
  written tree's hash against the blobs, replaces `engines/src/<engine>` and
  rewrites the entry's `vendor:` block. Deterministic: the same manifest and
  series give the same tree id. A changed patch series makes `check` (and
  `mcp/test_engines.py`) fail until `vendor` is run again.
- **`build`** first runs `check`, then checks the vendored tree out into
  `<out>/src` through a throwaway index with `core.autocrlf` as the entry's
  `clone` says (the originals were CRLF checkouts; `build-info.cpp.in`'s
  line endings reach `build-info.cpp`), leaving no `.git`. llama.cpp embeds
  a build number and short hash it reads from git: the fetch build had
  them from the clone, the offline build passes them as
  `-DLLAMA_BUILD_NUMBER` / `-DLLAMA_BUILD_COMMIT` from `vendor.build_info`
  (read from the shipped build's `build-info.cpp`, or `git rev-list
  --count` over a commits-only fetch: 11223 for `4da63377` both ways), with
  `GIT_CEILING_DIRECTORIES` fencing out any repository above. The network is
  fenced for the whole build (proxy variables point at a closed port, git
  may use `file://` only). An entry with a download input (the web-UI
  archive of `llama-prism` and `llama-bonsai2-base`) is refused unless
  `--inputs DIR` holds it as `<key>.tar.gz`; it is checked against its
  SHA-256 as before. Everything after the source step is the fetch build's.
- **`update`** rebases the series with git's own 3-way rebase: the old and
  new bases as commits in a scratch repo (submodules flattened), one commit
  per patch, `git rebase --onto`. A conflict stops it, names the patch and
  the files, aborts, and writes nothing -- with or without `--write`. A
  patch that becomes empty (upstream took it) stops it too: remove it from
  the manifest yourself. Moved submodule pins are reported and must be set
  by hand. Otherwise each patch is `unchanged` (the original file still
  applies and gives the same tree: its bytes are kept) or `rebased` (the
  file is rewritten: its header and credits kept, the diffstat and the diff
  regenerated). `--write` then writes those files, their sha256s and
  `base_commit`, and re-vendors. The entry's `expect_generated` pins are for
  the old base: re-pin them from the next build (as "Bump a base commit").

### Proof (2026-09-29)

- `check --derive` on every tree: ok (the series reverse-applies and lands
  on each recorded base).
- `build llama-bonsai2-ada --configure-only`: all 208 CMakeCache options
  equal the deployed `80d2c60d` tree's (bar the two build-info overrides),
  `/arch:AVX2`, and `common/build-info.cpp` byte-identical to its pin
  (`0b14858b...`).
- `build --configure-only` of `sd-cpp` (all 204 options equal, the four
  "unknown" configure lines), `llama-mirai-s` and `llama-upstream-moe` (210
  equal each): their `build-info.cpp` is byte-identical to the shipped
  builds' (`bd02898f...`, `eaa07321...`).
- `build llama-upstream --jobs 16` into a scratch directory, network
  fenced: compiled, every check passed, `build-info.cpp` byte-identical to
  the shipped build's (`eaa07321...`), engine tests 2/2
  (test-reasoning-budget, test-chat). As with every rebuild here, the DLLs
  are not bit-identical to the shipped ones ("How reproducible this is").
- `vendor --all --check-upstream`: each tree regenerated from GitHub + the
  series in a temp directory equals the recorded and the on-disk tree.
- `build llama-prism` without `--inputs`: refused before anything is created.
- `update` dry runs: `llama-upstream` onto ggml-org master `d280808f5d82`
  (36 commits, 87 files later): 1/1 patch unchanged. `llama-upstream-moe`
  onto the same: 5/5 unchanged. `llama-bonsai2-ada` onto prism `87268f775d74`:
  41/41 unchanged. Nothing written. The conflict, empty-patch and rewrite
  paths are gated on a throwaway upstream in `mcp/test_engines.py`.

### Reproducing on another machine

1. Clone this repository. Nothing else is fetched.
2. Install the toolchain the manifest's `toolchains.msvc-cuda128` names
   (Visual Studio 2022 with MSVC 14.36 and Windows SDK 10.0.22621.0; CUDA
   12.8.93 -- nvcc, cudart, cuBLAS -- in a directory you set as
   `toolchains.msvc-cuda128.cuda.root`; Git for Windows) and PyYAML for the
   script. `defaults.engines_root` is where builds go by default.
3. `python scripts/build_engine.py check` -- every tree is what the
   manifest records.
4. `python scripts/build_engine.py build <engine> --jobs 8 [--out DIR]`.

The CPU code follows the build machine (`GGML_NATIVE`, "native_cpu" in the
manifest): on a CPU without AVX2 the `/arch:AVX2` check fails -- that is the
recorded build, not a broken one. The result is source-reproducible, as
below; it is not the shipped binary's hash, which only deploy_check
verifies.

## How reproducible this is

**Source-reproducible, not bit-for-bit.** A rebuild from the manifest
compiles the same source, patches and generated inputs with the same
compiler, flags and CUDA, and the checks above prove each of those. It does
not produce the same bytes, and nothing here claims it does:

- **MSVC's linker stamps the build time** into every PE header (and its
  checksum), since nothing passes `/Brepro`.
- **`__FILE__` embeds the source path**: ggml's asserts carry
  `C:\Users\jwals\<checkout>\...`. A rebuild elsewhere differs in those
  strings, and in length if the path length differs.
- **nvcc's fatbins carry temporary names** (`tmpxft_<pid>_...`).
- **GGML_NATIVE is ON** (llama.cpp's and sd.cpp's default, never set
  explicitly). Under MSVC it probes the build machine's CPU at configure
  time: here it resolved to `/arch:AVX2` with `GGML_AVX2`, `GGML_F16C`,
  `GGML_FMA`. The CPU code is tied to this machine; the builder checks the
  resolution matches and fails on another CPU that resolves differently.
- The web UI llama.cpp embeds was an input, not source, which is one
  reason the default is now no UI. `llama-prism` and `llama-bonsai2-base`
  downloaded a prebuilt archive from the ggml-org HF bucket (pinned by version
  and SHA-256; prism's original asked for "latest", which was `b11065` at the
  time, identified by its hash). The first nudge build (`build-nudge/`, now
  `llama-bonsai2`'s `previous`) embedded an npm build carrying a build
  timestamp; the archive that reproduced it was deleted with the no-UI
  decision, so that binary is pinned but no longer rebuildable.

That is why the manifest records the SHA-256 of the binary we **ship**, and
why `deploy_check` verifies that one: the claim is "this exact binary, and
here is how to build its equivalent from our repo", not "a rebuild reproduces
these bytes".

### Proof: sd-cpp rebuilt from the manifest (2026-09-25)

`python scripts/build_engine.py sd-cpp --jobs 8 --out C:/Users/jwals/eng-sd01`
(`C:\Users\jwals\eng-sd01\src` is 27 characters, like the original's
`C:\Users\jwals\sdcpp-pr2043`, so path strings line up). Every check passed;
the build had no errors.

| what | result |
|---|---|
| source tree | byte-identical to `C:/Users/jwals/sdcpp-pr2043` (`diff -r`, build and logs excluded) except the copy's four dangling submodule gitlinks (`ggml/.git` = `gitdir: ../.git/modules/ggml`, pointing into a `.git` the copy does not have) |
| patch | `f047986`'s diff applies to `c92d73c`; the ten files it touches are byte-identical to the snapshot |
| CMakeCache | all 204 BOOL/STRING options identical to the original's |
| build.ninja | identical, all 5,568 lines, once the source root is mapped: every compile and link flag, define and include of every target |
| configure | `version unknown`, `commit unknown`, `ggml commit: unknown`, `frontend build disabled`, `/arch:AVX2` -- as the original |
| CUDA DLLs | identical (copied from the toolkit, hashes checked) |
| `sd-server.exe` | **same size** (88,427,008 B), **different SHA-256** (`8410237f…` vs shipped `75349c74…`) |

Why the exe differs, from the objects up (`<out>/src/build` against
`sdcpp-pr2043/build`, 381 objects):

- **169 of 237 C/C++ objects are equivalent** once the COFF timestamp is
  dropped and the source root mapped. The other 68 differ in names MSVC
  derives from the file path: lambda classes are named
  `<lambda_<32-hex hash>>` and path string literals `??_C@_0..@<CRC>@<escaped
  prefix>@`, both hashes of the full path (in `model_manager_files.cpp.obj`
  6 of 7 lambda names differ), plus the COMDAT checksums over them.
- **All 144 CUDA objects differ** in their embedded fatbin and `.rdata`:
  nvcc compresses the device code (`GGML_CUDA_COMPRESSION_MODE=size`) and its
  inputs carry temp names (`tmpxft_<pid>_...`) and paths.
- In the exe, every section has the same RVA and size except `.nv_fatb`
  (224 bytes smaller); `.reloc` and `.rsrc` are byte-identical. `.text`
  and `.rdata` differ in place: the linker orders string-literal COMDATs
  by their (path-hashed) names, so RIP-relative displacements change -- the
  first differing bytes are 2-byte `lea` displacements.

So the rebuild compiles the same source with the same flags into the same
layout; the bytes differ only where the toolchain writes the build path, the
build time or temp names into the output. That is the expected result, and it
is why deployment trusts the recorded hash of the shipped binary.

### llama-bonsai2 without the UI: the shipped build (2026-09-25)

`python scripts/build_engine.py llama-bonsai2 --jobs 8` into
`C:/Users/jwals/engines/llama-bonsai2-eebbaae2`: every check passed. The
CMakeCache equals `build-nudge`'s in all 207 options except `LLAMA_BUILD_UI`
(ON -> OFF; `LLAMA_USE_PREBUILT_UI` was already OFF there); `build-info.cpp`
is byte-identical; `ui.cpp` is 286 bytes, an empty asset table, and nothing
was downloaded; `test-reasoning-budget` and `test-chat` passed. Without the
UI, `llama-server-impl.dll` is 3,187,712 bytes (6,310,400 with it).

Smoke test on CPU (CUDA hidden with `CUDA_VISIBLE_DEVICES=-1`, Qwen3.5-0.8B
Q8_0, `-ngl 0`, 127.0.0.1:18777; never the GPU, never :1234/:11434):
`GET /health` 200 `{"status":"ok"}`, `GET /v1/models` 200,
`POST /v1/chat/completions` 200 (fingerprint `b10738-285542d98`), `GET /` and
`GET /index.html` 404 with the server's JSON `File Not Found`, and `/health`
still 200 afterwards.

### Proof: llama-bonsai2 with the UI (the first nudge build) rebuilt from the manifest (2026-09-25)

This was the recipe before the no-UI decision; the archive it used is gone,
so it cannot be repeated. Kept as the evidence for the method.

`python scripts/build_engine.py llama-bonsai2 --jobs 8 --out
C:/Users/jwals/eng-llama-bonsai2-n01`: every check passed.

| what | result |
|---|---|
| patch | `0001-reasoning-budget-nudge.patch` (7 files) applies to `285542d98` |
| CMakeCache | all 207 BOOL/STRING options identical to `build-nudge`'s |
| `build-info.cpp` | byte-identical (build 10738, commit `285542d98`) |
| `tools/ui/ui.cpp` | byte-identical (the archived npm-built UI, gzipped with Git's gzip 1.13; archive since deleted) |
| engine tests | `test-reasoning-budget`, `test-chat`: 2/2 passed (static CPU tree) |
| CUDA DLLs | identical |
| the other 9 files | every size identical except `ggml-cuda.dll` (93,090,816 vs 93,102,080 B); every SHA-256 differs |

`ggml-cuda.dll`'s whole difference in size is its `.nv_fatb` section (11,176
bytes smaller; every other section the same size): nvcc's compressed device
code, whose inputs carry the build path (the rebuild's tree is
`src\build`, the original's `build-nudge`) and temp names. The rest differ
in place, as for sd-cpp.

The comparison scripts are not in the repo (one-off). To repeat: map the
source root, zero the COFF `TimeDateStamp`, and compare section by section.
