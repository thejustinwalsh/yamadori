# Installing Yamadori

This page is the path from a clone to a running stack, and an honest list of what that path does not do for you.
Everything here was written against the machine the stack was built on; where a step has only been read, not run,
it says so. If something here disagrees with the code, the code wins: `scripts/install.ps1`, `scripts/make_config.py`
and `mcp/test_install_docs.py` (which keeps them true) are the sources.

## What you need

| | required | notes |
|---|---|---|
| OS | Windows 11 | the launchers are `.bat` / `.ps1`; the Python is portable, the start scripts are not |
| GPUs | **two** NVIDIA cards, 16 GB each | built on an RTX 5060 Ti (main model, alone) + an RTX A4000 (jjava, side calls, vision, images, embedder). One card is not a supported layout: the second conversation, the decider and the side calls all assume the other card |
| NVIDIA driver | current | `nvidia-smi` must list both cards |
| Python | 3.13 (3.12 expected to work) | the lock was frozen on 3.13.15; the installer makes `.venv` from `requirements.txt` |
| RAM | built with 64 GB; **64 GB** for Flash-Next | the Bonsai tiers' RAM use was not measured for this page; the host-RAM prompt caches are ceilings of 16 GiB + 5 GiB (`--cache-ram`) beside the OS, the proxy and the worker; Flash-Next keeps ~38 GB of expert weights in RAM |
| Disk | ~12 GB for the Bonsai tiers; +10 GB Mirai S; +15 GB images; +66 GB Flash-Next | sizes are from `models/manifest.yaml` |
| Git | yes | engine builds, the main-model graft |

| | optional | what it buys |
|---|---|---|
| Visual Studio 2022 ("Desktop development with C++") + CUDA toolkit 12.x | to **build the engines** | there are no prebuilt binaries: the engine source is vendored in `engines/src` and built on your machine |
| Docker Engine (Docker Desktop, or the WSL setup in [DOCKER-WSL.md](DOCKER-WSL.md)) | package lookups | `yama_find_package` and friends run PackageLens in a container; without Docker they are switched off |
| WSL | only for the Docker setup above | |
| Caddy + a DNSimple token | HTTPS on one hostname | `caddy/Caddyfile` is the author's (it names his host): copy and edit it |

## The short path

```powershell
git clone https://github.com/thejustinwalsh/yamadori
cd yamadori
powershell -ExecutionPolicy Bypass -File scripts\install.ps1 -DryRun     # read-only: checks, then says what each step would do
powershell -ExecutionPolicy Bypass -File scripts\install.ps1             # does it, asking before every download and build
```

Options: `-ModelsDir`, `-EnginesRoot`, `-PublicBase http://host:1234`, `-With vision,images,mirai,flash`,
`-MainModel lean|graft`, `-FetchModels` / `-BuildEngines` (pre-approve those two), `-Yes`, `-SkipAccount`,
`-Python path\to\python.exe`. It is idempotent (a finished step is recognised and skipped), it never overwrites
`stack.env`, `config.yaml` or an existing model file, and it deletes nothing.

Then:

```powershell
scripts\start-stack.bat                      # in a console; logs go to logs\*.log
curl.exe http://127.0.0.1:1234/health
```

and point a client at `http://127.0.0.1:1234/v1` with the key the installer saved in `.keys\` (README "Quick start").
`scripts\install-autostart.ps1 -DryRun` shows what registering the start-at-logon task and the five-minute watchdog
would do.

## What the installer does, step by step

1. **Prerequisites**, read-only: GPUs (`nvidia-smi`), RAM, Python, Git, Visual Studio, CUDA, Docker, free disk.
2. **`.venv`** from [requirements.txt](../requirements.txt): the ~15 packages the code imports, pinned to the versions in
   `requirements.lock.txt` (which is the author's whole conda environment). `mcp/test_install_docs.py` fails if an import
   appears that is not named there. A clean-machine `pip install` of this file has **not been run**.
3. **`stack.env`** (gitignored): this machine's `YAMADORI_PYTHON`, `YAMADORI_CUDA_BIN`, `YAMADORI_PUBLIC_BASE`,
   `YAMADORI_MODELS_DIR`, and `YAMADORI_MCP_TOOLS=0` + `YAMADORI_WSL_ENGINE=0` when no Docker engine answers.
   `start-stack.bat` and `watchdog.ps1` read it; a variable already in the environment wins over the file, and a
   machine with no `stack.env` behaves exactly as before it existed. `YAMADORI_SEARXNG_DIR` is also read (SearXNG
   is retired: nothing needs it).
4. **llama-swap** `v256`: downloaded from its GitHub release and checked against the sha256 pinned in
   `engines/manifest.yaml`.
5. **Engines**: see below.
6. **Models**: `scripts/fetch_models.py fetch <id> --models-dir <dir>` for each pinned artifact, sizes shown first.
   The download streams to `<name>.part`, is hashed as it goes, and is renamed only when size and sha256 match the
   manifest. It never overwrites a different file.
7. **`config.yaml`**: rendered from `config.example.yaml` by `scripts/make_config.py` (below).
8. **`index/token_embd.npz`**: the concept seed's vocabulary, from the 27B's own token embeddings
   (`scripts/extract_token_embd.py`, CPU, ~1 minute). It is extracted from the *original* trunk; the author's file came
   from an abliterated trunk and is a different matrix (same vocabulary), so the two files differ byte for byte.
9. **An API key**: `mcp/accounts.py create`. Until one account exists the server is single-user and open (right for a
   private network); with one, every request needs `Authorization: Bearer <key>`. The key goes to `.keys\<label>.key`.
10. **PackageLens** (optional): prints the two Docker steps; see "Package lookups".

## Engines: building llama-server

`engines/src/<engine>` is each engine's upstream base commit with our patch series applied, vendored, so a clone is
enough. Five matter here:

| engine | for | needed |
|---|---|---|
| `llama-bonsai2-ada` | `bonsai`, `bonsai-a4000` (and, in a fresh install, the embedder and `bonsai-vision` too) | always |
| `llama-mirai-s` | `mirai-s`, the xhigh tier | with `-With mirai` |
| `llama-upstream-flash` | `flash-next`, the max tier | with `-With flash` |
| `sd-cpp` | `imagegen`, `imagegen-turbo` | with `-With images` |
| `llama-prism` | the author's embedder / vision server | not needed: the installer points `server` at the ada build |

The installer runs, per engine, `python scripts\build_engine.py build <engine> --portable --no-tests --jobs N --out
<EnginesRoot>\<engine>`, and `config.example.yaml`'s paths default to `<EnginesRoot>\<engine>\src\build\bin\`.

**What `--portable` means, and what it costs.** `build_engine.py` was written to prove a rebuild is the *recorded*
toolchain's: it checks the MSVC toolset (14.36.32532), the Windows SDK (10.0.22621), the CUDA 12.8.93 DLL hashes and
the CPU's native flags, and without `--portable` it fails on any other machine. With `--portable` those identity checks
become `WARN` lines (and are recorded as waived in `build-report.json`); the source and configuration checks (the
vendored tree is the recorded one, the patches are in it, the CMake flags and cache are as recorded) still fail the
build. The result is **the recorded source built with your toolchain, not the recorded binary**. Builds are
source-reproducible, not bit-for-bit, even on the author's machine (docs/ENGINES.md).

**Where your Visual Studio and CUDA are.** `engines/manifest.local.yaml` (gitignored, merged over `defaults` and
`toolchains` only; the installer offers to write it from what it detected) names them:

```yaml
defaults:
  engines_root: D:/yamadori/engines
toolchains:
  msvc-cuda128:
    vcvars: C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Auxiliary/Build/vcvars64.bat
    cmake: {path: C:/Program Files/Microsoft Visual Studio/2022/Community/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe}
    ninja: {path: C:/Program Files/Microsoft Visual Studio/2022/Community/Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe}
    cuda:  {root: C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8}
    git:   {path: C:/Program Files/Git/cmd/git.exe}
    gzip:  {path: C:/Program Files/Git/usr/bin/gzip.exe}
```

`scripts\build_engine.py check` (offline) proves the vendored trees are what the manifest records. **No portable build
has been run to completion by anyone but the code's own tests** (`mcp/test_install_docs.py` covers the waiver rule and
the local-manifest merge); a first engine build on a new machine is the least-tested step in this document. It
compiles CUDA: expect tens of minutes and several GB of disk per engine.

## Models

`python scripts\fetch_models.py list` shows every artifact with its size and provenance; `--verify` hashes what
`config.yaml` points at against the manifest. The installer's sets:

| set | files | GiB | notes |
|---|---|---|---|
| core | Bonsai 2 27B PTQ1_0 trunk, Qwen3-Embedding-0.6B Q8_0 | 6.1 | the trunk is what the second card serves |
| lean (default) | `Ternary-Bonsai-2-27B-PTQ1_0-mtp-lean.gguf` | 5.9 | the main model, quick path |
| vision | the mmproj | 0.6 | `bonsai-vision`, on demand |
| images | Qwen-Image-2.1 base + turbo, text encoder, VAE | 15.0 | **Qwen Research License, non-commercial** |
| mirai | Mirai S Qwen3.8-27B GGUF | 10.4 | a third-party conversion of Mirai Labs' checkpoint |
| flash | Flash-Next IQ2_XS, two shards | 63.4 | plus two files that are recipes, below |

Licences are in `models/manifest.yaml` per artifact; the base models are Apache-2.0 except the image set above and
Flash-Next (Qwen Community License 1.0).

### The main model: quick path and exact path

The author's main model is a **local graft**: the original Bonsai 2 27B trunk with ProCreations' on-policy MTP head
added as block 64 (`models/manifest.yaml` `bonsai-2-27b-mtp-procreations`, sha256 `ed55a7aa...`). A graft cannot be
fetched, so the installer's default (`-MainModel lean`) uses sudoingX's published `...-PTQ1_0-mtp-lean.gguf` (a pinned,
verified download that carries its own MTP head). It is expected to run and **has not been measured by us as the main
model**; the numbers in the README are for the graft.

For the exact file (`-MainModel graft`, the file must already be in the models directory), the manifest's recipe is:

```powershell
git clone https://github.com/professorpalmer/bonsai-ada-surgery; git -C bonsai-ada-surgery checkout 5158a8df3bc4c8b9375ed521021eb6118feda78d
git clone https://github.com/sudoingX/bonsai2-small-gpu;         git -C bonsai2-small-gpu checkout eb52d9d7363cda2d910146f4e37f4b8c64c30c46
$env:PYTHONPATH = "<repo>\engines\src\llama-bonsai2-ada\gguf-py"
python bonsai-ada-surgery\surgery\hf_sparse_fetch.py https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-MTP/resolve/efffdea64c1f9e93cc7fa6bb24f72ae9d66ecf51/Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf procreations-pq2-mtp.sparse.gguf blk.64.
python bonsai2-small-gpu\graft\tools\extract_head.py procreations-pq2-mtp.sparse.gguf head-procreations.gguf --no-embed-tokens
python bonsai2-small-gpu\graft\tools\merge.py Ternary-Bonsai-2-27B-PTQ1_0.gguf head-procreations.gguf Ternary-Bonsai-2-27B-PTQ1_0-mtp-procreations.gguf
```

(the head should be 451,321,504 bytes, sha256 `c0473290...`; the graft 6,397,969,888 bytes, sha256 `ed55a7aa...`.) This is the
manifest's recorded recipe; the installer does not run it and it was not re-run for this document.

### Flash-Next (the max tier) is not a one-command install

Besides its two shards it needs `mtp-Qwen3.8-Flash-Next-Q8_0-shared-embd.gguf` (a draft built from Qwen's BF16 MTP
tensors with a patched converter, then quantised) and `expert-profile-strata-d551edf4.bin` (Strata's expert profile).
Both are recipes in `models/manifest.yaml` (`flash-next-mtp-draft-q8_0`, `flash-next-expert-profile-strata`), not
downloads. Its launch line also carries `-t 16 -C 0xFFFF`: threads and an affinity mask for a 16-thread CPU; set
yours. Treat `-With flash` as expert mode.

## config.yaml

`config.yaml` is llama-swap's file and it is gitignored (it holds paths and GPU UUIDs). `config.example.yaml` is the
committed copy of what the stack runs, with ten `@@PLACEHOLDERS@@` for the machine-specific parts;
`python scripts\make_config.py` fills them (`--list-placeholders`, `--list-gpus`, `--stdout`, `--set NAME=VALUE`,
`--force` to replace an existing file, keeping a `.bak`). The first listed GPU is taken as the main card; the cards are
pinned by UUID (`CUDA_VISIBLE_DEVICES`), and `start-stack.bat` also pins `CUDA_DEVICE_ORDER=PCI_BUS_ID`.

**Numbers in it that are measurements of the author's cards, not constants:** `-c 209920` and `--kv-vram-cells` for the
main model (every KV cell in the 5060 Ti's VRAM beside ~6 GB of weights, with the headroom `docs/ENGINES.md` "The VRAM
line" derives), `-c 128000` for `bonsai-a4000`, `--cache-ram 16384` / `5120`, and the Flash-Next block. On other cards
re-measure with `bench/kv_rank.py` and `bench/a4000_fit.py` before trusting them; too large a window loads and then dies
when a long prefill allocates its compute buffer.

`mcp/tier_models.yaml` sends `xhigh` to `mirai-s` and `max` to `flash-next`. On a machine without those two models a
request at those efforts fails (llama-swap cannot start a model whose files are absent); use `high` and below, or edit the table. (Emptying the table's
`tiers:` map is not the answer: the table then switches off the helper wiring the Bonsai layout depends on.)

## Starting, checking, using

`scripts\start-stack.bat` starts, in order: the WSL Docker keep-alive (skipped with `YAMADORI_WSL_ENGINE=0`), the tools
API (:1235), Caddy (only if `caddy\.env` holds a token), the job worker, SearXNG (only if its venv exists), llama-swap
(loopback :11434) and, in the foreground, the proxy (:1234). `scripts\watchdog.ps1` restarts a service only on its second
consecutive failed `/health`, five minutes apart.

Checks: `curl.exe http://127.0.0.1:1234/health`; the dashboard at `/`; `python scripts\run_tests.py` (every offline suite
plus `ruff --select=E9,F`); `python scripts\run_tests.py --live --key-file .keys\me.key` against the running stack (needs
the models loaded; read AGENTS.md "Before you claim anything works" first).

## Package lookups (optional)

`yama_find_package`, `yama_list_package_versions`, `yama_read_package_readme`, `yama_resolve_packages` run PackageLens
(and an npm resolver) in containers on an `--internal` network behind an egress gate. Needs a Docker engine that
answers `docker version`. Steps: `docker pull` the base image named in `mcp_servers/packagelens/Dockerfile` (`ARG BASE`),
then `python mcp\mcp_host.py build packagelens`. **A rebuild gets a new image id** (npm writes timestamps) and the proxy
starts the server only from the id `models/manifest.yaml` records (`runtimes: mcp-packagelens`, `image_id`): put the id
the build prints there, or the lookups answer `MCP_SERVER_DOWN`. This is the sharpest edge a new user will meet; it is
not automated. Without Docker the installer sets `YAMADORI_MCP_TOOLS=0`.

## The skill library is not in the repository

The pipeline that makes skills (`mcp/skill_pipeline.py`, docs/SKILL-FACTORY.md) and the channels that deliver them are
in the repo; the library they produced on the author's machine (`index/skills/`, hundreds of skills) is derived data and
is not. On a fresh install nothing is armed, so `yama_recall_craft` and the package-tool results carry no skills until
you build some (a GPU job per skill; docs/PACKAGE-ONBOARDING.md).

## Refreshing the dashboard screenshots

`docs/img/dashboard.png` and `docs/img/dashboard-performance.png` come from `scripts/dashboard_screenshots.mjs`
(Node 24 and Microsoft Edge; headless; reads a key from `.keys\live-test.key`; writes to the directory given as its
argument): `node scripts\dashboard_screenshots.mjs docs\img` with the stack running. Look at the result before
committing it: the dashboard shows concept words, slot activity and hardware.

## What this path does not do

- It has not been run end to end on a machine other than the author's. The dry run was checked here; `pip install -r
  requirements.txt` on a clean venv, a portable engine build, a first start with no `index/`, and the PackageLens
  image-id step are the parts most likely to need you.
- One GPU, AMD, Linux and macOS are not supported layouts.
- The exact main model is a graft you build; the quick path substitutes a published file.
- Flash-Next needs two recipe-built files and a CPU-specific launch line.
- PackageLens needs its image id recorded by hand after a build.
- The skill library is yours to build.
- It does not set up TLS or DNS. `caddy/Caddyfile` is the author's.

## Troubleshooting

- **The main model loaded on the wrong card.** CUDA orders devices fastest-first unless `CUDA_DEVICE_ORDER=PCI_BUS_ID`;
  `start-stack.bat` sets it, and each model also sets `CUDA_VISIBLE_DEVICES` to its UUID. If you start llama-swap by
  hand, set both.
- **A server exits at once with status 0.** `ggml-cuda.dll` could not load: the CUDA runtime DLLs are not on `PATH`.
  Set `YAMADORI_CUDA_BIN` in `stack.env`.
- **`PTQ1_0` unsupported.** Stock llama.cpp rejects it. Build the vendored engines, not an upstream release. Do not
  build sudoingX's `pr-ptq1-mmv` branch: faster, and it corrupts generation (README "Things that will bite you").
- **llama-swap answers but a model never loads.** `logs\stack.log` has llama-swap's view; run the model's `cmd` from
  `config.yaml` by hand with `--port 10099` to see the server's own error.
