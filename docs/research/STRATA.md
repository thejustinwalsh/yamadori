# Strata, and the model it runs: should Yamadori build on either?

Research note, 2026-09-26. It is read-only: nothing was cloned into this repo, installed or run. Two questions are
answered separately. First, **the framework**: should Strata be Yamadori's long-term base? Second, **the model**: is
Qwen3.8-Flash-Next, which Strata runs, worth moving to from Bonsai 2 27B? The operator believes it is the more capable
model.

**Verdicts, in brief**

- **Framework: do not adopt Strata, and do not copy its code.** It is a two-day-old, single-author engine written for
  one architecture. It cannot load our model. It has no licence file (2026-09-29: superseded -- MIT since
  `c0107c3`, see "Update 2026-09-29"). It lacks the three things our proxy is built on:
  a prompt cache, sampling, and concurrent slots. Borrow a few ideas (section 5).
- **Model: worth one decisive, paired test, served through OUR stack on upstream llama.cpp, not on Strata.** Qwen's
  own table shows Flash-Next ahead of Qwen3.8-27B, the base of our Bonsai, most of all on agentic coding. That table
  is self-reported, and it is for BF16. Several things are unmeasured on our box: what a 2-bit quant keeps, how fast
  it runs with experts offloaded to our AVX2 CPU, and how much RAM is left for the rest of the stack. The licence is
  also more restrictive. So this is a prior, not a result (section M).

## Update 2026-09-27: Strata at `b742ff9` (what changed since `6da1f66`)

Re-read at HEAD `b742ff998704638903461a2d7d6c1e03b8032509` (pushed 2026-09-27 19:51 UTC), for the operator's
decision to serve Flash-Next as the max tier (docs/FLASH-NEXT.md). `gh api .../compare/6da1f66...main`: 50 commits,
72 files. Still **no licence** (`license: null`), so the rule in section 2 stands: read for ideas, copy nothing.
*(2026-09-29: superseded, MIT since `c0107c3`; see "Update 2026-09-29".)*

| area | at `6da1f66` (above) | at `b742ff9` | commits |
|---|---|---|---|
| Activity | 4 commits, 1 author, 133 stars | 50 more commits; 5 contributors (Niko1221 35, Mirtraxxx 3, coolio986 2, code-martin 2, Vistawizard 1); 512 stars, 55 forks; releases v0.1.3-v0.1.12 in two days | - |
| Conversation cache | none ("every request processes its whole prompt again") | **merged** (PR #8/#10): the live session plus up to 6 RAM checkpoints (~118 MB each) at each assistant turn and every 16K tokens; still **one conversation cached at a time** and one request at a time | `23f6b15`, `54cf7b6` |
| Sampling | greedy only | temperature/top_p/top_k/min_p/seed and the penalties forwarded per request; with no `sampling` block in the run config (setup writes none) a request that sends no values is still **greedy** | `012992f`, `968512c`, `e1d9e99` |
| Quants | Q2_0, IQ2_XS, IQ3_XXS | + **IQ3_S** (ISTA's new "recommended" build, 83.6 GB) | `36215b9` |
| KV | int8 above 8K | + optional **q4_0 KV with a Hadamard rotation** (PR #21; their bench: document perplexity +8% at 1K, +12% at 8K; int8 stays default); **KV streaming** from 64K (`--kv-resident 32768`: the KV lives in RAM, 32K positions per layer in VRAM, so more experts fit: Q2_0 at 262K 50.9 -> 62.6 tok/s, their run) | `a278979`, `e02643e`, `f02dc49` |
| MTP / drafting | MTP, up to 3 drafts | + a prompt-lookup (suffix) drafter "that pays" beside MTP; setup runs `--spec 4 --spec-min-p 0.5` | `a9047fd` |
| CPU expert pool | spun at 50% idle (#4); default worker count stalled two machines (#6, #11) | sleeps between requests (PR #9); a race in the pool fixed and a watchdog so it "never hangs silently" (#29, engine 0.1.12); Linux `physical_cores()` fixed (PR #14) | `0cf01f5`, `06438fd`, `ca1758d` |
| Expert cache | adaptive | sized around the output head so VRAM stays free (0.1.9) | `0680ab6` |
| Serving | 127.0.0.1 only; 400 when `max_tokens` exceeds the room | `--host 0.0.0.0 --api-key` (#26); optional `--fit-max-tokens` clamps instead of 400 (PR #24); unset/-1 `max_tokens` = the rest of the context (PR #18); an engine that dies mid-answer is an error and restarts (#27); `/metrics` | `968cc0d`, `556fa27`, `a40fe52`, `c22ca55` |
| Web app | one page | Chat / Monitor / About tabs; text attachments; `serve/telemetry.py` (NVML + psutil hardware readings, local only: no network calls in it) | `7cb0ca3`, `481e0cd` |
| **Experimental speed projection** | - | a 480 KB control vector applied after layers 4-44 (`h -= (h.v)v`), **off by default**. Its own package calls it a **refusal-direction projection** (declines 1/50 vs 50/50 on its set). Their measurement: top-1 changes at 10% of positions, code perplexity +15%. Not a speed optimisation (0.2-0.4% slower on the same text). **Not taken.** | `9e599c0` |
| Setup | - | recompiles a local engine when its source changes (#31); RAM floor enforced (48 GB for the 2-bit models) | `b742ff9`, `11bb7e2` |

**Speed claims now** (`docs/DETAILS.md`, RTX 5070 12 GB + Ryzen 5 7600 AVX-512 + DDR5-5200, one prompt per length,
256 tokens, MTP on): IQ2_XS prompt 332-495 tok/s, output 82.0 at 1K -> 48.0 at 262K; Q2_0 output 88.7 -> 56.3.
Their estimate for our 5060 Ti (Q2_0) is ~80-87 tok/s at 1-4K, "±20%", estimated, not measured. Still no llama.cpp
baseline in the repo.

**What did NOT change:** the llama.cpp pin (`3cf03257f`, used for gguf-py, ggml's CPU backend and `mtmd`), the ISTA
download URLs (`resolve/main`, unpinned), `tools/mtp_fetch.py` (byte-identical), the unverified prebuilt engine zip.

**Corrections and additions to this note, found on the re-read:**
- Section M2 says ISTA publishes "no SHA256 sums". There is no SHA256SUMS file, but Hugging Face LFS publishes a
  sha256 per file (the API's `lfs.sha256`, and `X-Linked-ETag` on each resolve URL). Our downloads were checked against
  them (models/manifest.yaml `flash-next-*`).
- ISTA's repo moved from `c67535c` to `2a55d759` (IQ3_S added, README). The IQ2_XS files are unchanged since `89123389`.
- The chat template embedded in ISTA's GGUFs is byte-identical to Strata's `serve/chat_template.jinja` (sha256
  `12827f24...`), an Unsloth-fixed variant of Qwen's (merged leading system/developer messages, `high` accepted as
  `xhigh`, tool-call arguments must be a mapping). How Strata drives the model, against our max tier:
  docs/FLASH-NEXT.md "How Strata steers the model".

**2026-09-28: `d551edf4`** (21 more commits). Engine 0.1.13-0.1.15: prompt reading ~2x faster than 0.1.12 (MMQ
experts, an expert streaming ring, larger chunks, pinned-copy helper threads, a batched PLE block; decode unchanged),
PR #43's AVX2 multi-token i-quant kernels, PR #42's Windows large-page arena, `CUDA_MODULE_LOADING=EAGER`, a full
24,576-pair expert profile. Still no licence; the operator authorised using its code directly as attributed
patches, and chose to PORT it into our llama.cpp engine rather than serve Strata: docs/FLASH-NEXT.md section 0,
docs/ENGINES.md "Strata's MoE work, ported".

## Update 2026-09-29: Strata at `3ce2523c` (130 commits after `d551edf4`), and the licence

**Licence correction (2026-09-29).** Strata IS licensed now: **MIT**, `LICENSE` added in `c0107c3`
(2026-09-28 14:29 +0200, "License: MIT (the ggml code, the font and the projection vector keep their own
licenses)"), blob `6687a423dde6b7aff1b748c07f5ee93ff788089c`, "Copyright (c) 2026 Niko1221 and the Strata
contributors"; `gh api repos/Niko1221/Strata` reports `license: mit`. Every "no licence / copy nothing" line
above (section 2 "Licence", section 5, the 2026-09-27 update) was true when written and is superseded:
code may be ported under MIT, taken from the tree at or after `c0107c3` (read at HEAD), with the MIT notice kept
in each patch that carries it and the Strata commit named. Exceptions it states: the vendored ggml code (MIT,
ggml's notice), the Outfit font (OFL), and the experimental speed projection vector (Qwen Community License).
Our moe engine's 0004/0005 were ported from `d551edf4`, BEFORE the licence commit, on the operator's
authorisation of 2026-09-28; their Strata-derived parts must be re-derived from the MIT tree and carry the notice
(engines/manifest.yaml `llama-upstream-moe` records the status).

Read with `git clone` + `git log d551edf4..3ce2523c` (read only; 130 commits, many merges of community PRs).
Engine 0.1.16 -> **0.1.24**. Activity: 1,481 stars; PRs from ~15 outside contributors.

**What Strata claims now, per mechanism** (their numbers, their machine unless stated: RTX 5070 12 GB PCIe 5.0
x16, Ryzen 5 7600 AVX-512, 64 GB DDR5-5200, Windows; one prompt per cell, n=1, MTP on, greedy):

| mechanism | claim | commit / source |
|---|---|---|
| Output, IQ2_XS (0.1.22) | 77.2 / 74.6 / 67.5 / 63.5 / 59.1 / 54.7 tok/s at 1K / 4K / 32K / 64K / 128K / 262K; Q2_0 84.7 / 88.2 / 77.4 / 71.0 / 63.1 / 61.1 | `bench/results/2026-09-29-speed-0122` |
| Prompt, IQ2_XS (0.1.22) | 524 / 1,196 / 1,799 / 1,611 / 1,495 / 1,181 tok/s (same lengths); Q2_0 519 ... 1,844 | same |
| MTP (`--spec 4 --spec-min-p 0.5`) | draft acceptance 0.63-0.86, 2.7-3.2 tokens per verify round | `2026-09-28-speed-0114/matrix.json` |
| VRAM expert cache (profile-ranked + adaptive swaps) | 3,565-4,386 IQ2_XS experts resident on 12 GB; 70-92% of routed reads hit VRAM on a 16 GB 5080 (Coder), "decode speed now comes from VRAM hits" | `2026-09-29-layer-split`; `df6980d` |
| CPU expert pool is memory-bound | pool reads ~30 GB/s (DDR5 through 4 KB pages); SMT threads do not help; a 2 KB row prefetch: 37.1 -> 36.5 ms/round | `df6980d` (E-2) |
| AVX2 multi-token i-quant kernels (for AVX2-only CPUs, ours) | 5700X3D + RTX 5060 Ti: gate/up 15.16 -> 13.84 ms/round, round 50.9 -> 48.5 ms (-4.7%); single thread, 3 tokens, IQ2_XS 536 -> 416 us | `ed14227` (pipeob0, PR #43) |
| Q2_0 CPU kernel | "ggml-cpu has only a scalar one on x86" (its own AVX2/AVX-512 Q2_0 rows) | `src/kernels/cpu/pool.cpp:383`, `q2_avx2.cpp` |
| PCIe share of the misses (GPU computes some missed experts by copying them) | a 5060 Ti at x8 probes 14.1 GB/s pinned H2D -> `pcie_frac` 0.29 (0.55 at x16) | `7a4b627` (PR #44) |
| KV streaming `--kv-resident 32768` | Q2_0 262K 50.9 -> 62.6 tok/s (1,589 -> 3,872 experts in VRAM); ~+6% at 128K | DETAILS.md |
| q4_0 KV + Hadamard (optional) | ~4% faster at 128K; perplexity +8-12% | DETAILS.md |
| Prompt-lookup drafter beside MTP | code edits 6-11% faster, other text unchanged | DETAILS.md |
| Batched verify-window kernels (bit-identical) | not quantified in the commit | `1e4515c` (PR #109) |
| E-6 device plan: a verify layer whose experts are all resident skips the host | not quantified | `efddd74`, `0abcd5b` |
| Prompt path 0.1.13 (8K chunks, MMQ experts, streaming ring, batched PLE, pinned-copy threads) | Q2_0 32K 572 -> 1,290; IQ3_S 383 -> 1,208 | `2026-09-28-prefill-speed` |
| D-1 QSA prompt attention on tensor cores | Q2_0 32K prompt 1,386 -> 1,646 (+18.8%); not bitwise | `dce4598` |
| C-1/C-2 QSA select grid + chunked indexer appends | Coder 4K +18%, 20K +13% | `758eb1a` |
| D-2 GDN recurrence split over value columns | Q2_0 32K 1,258 -> 1,308 | `66f4341` |
| C-4 one embedding gather; PLE rows read by 4 threads | Q2_0 4K PLE 591 -> 291 ms, 941 -> 1,010 tok/s | `5b7e316` |
| D-4/D-5 queued slot refills; expert stream from its own host thread | 162 -> 96 ms per prompt; IQ3_S 32K 1,143 -> 1,213 | `581765a`, `cf68b00` |
| QSA select on tensor cores + register top-k | 128K prompt 1,608 -> 1,843 (+14.6%), TTFT 82.7 -> 72.3 s; KL 0.005-0.03 | `731899f` (0.1.24) |
| Layer split over several GPUs (experimental) | Coder, 5080 + 3090: prompt +18-20%, decode +0-7% | `d733199` ff., `2026-09-29-layer-split` |
| Conversation cache keeps its shared prefix; a checkpoint at the end of the system prompt | a new chat re-reads only what follows the system prompt | `6fb2085`, `c1e9033` |
| Bulk expert-arena reads on MSVC | load time (std::ifstream did 4,095-byte freads) | `5edb9d6` |

Still: no llama.cpp baseline in the repo; every row n=1 per cell on one machine; the prompts are not published.

**Against our box** (i7-13700K AVX2, 8P+8E cores, 64 GB DDR5-4800 in 4 DIMMs, RTX 5060 Ti 16 GB at PCIe x8): the
AVX2 rows, the x8 PCIe probe and the Q2_0 row are the ones that describe us. Our own measurements of where the
18 tok/s goes are in docs/FLASH-NEXT.md section 8.

## How to read the citations

Every Strata claim cites the tree at commit `6da1f66` (HEAD on 2026-09-26). The prefixes are:

- `S:` = `https://github.com/Niko1221/Strata/blob/6da1f66/`
- `Q:` = `https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4/`
- `I:` = `https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/blob/c67535c/`
- `U:` = `https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/blob/b22d729/`
- `B:` = `https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/blob/main/` (the card as fetched 2026-09-26, sha
  `b072e1d`)

GitHub API facts come from `gh api repos/Niko1221/Strata[/...]`, fetched 2026-09-26. Hugging Face facts come from
`https://huggingface.co/api/models/<repo>?blobs=true`.

**Not read:** `docs/paper/Strata-Paper.pdf`. No PDF text extractor was available without installing one, and
WebFetch returned nothing usable. Any claim that exists only in the paper is unverified here.

---

## 1. What Strata is

**Summary.** A custom C++/CUDA inference engine for exactly one model family: Qwen3.8-Flash-Next and its fine-tune
Swift 1.5. A Python HTTP front end sits on top. It also ships a one-click installer for a PC with one NVIDIA card
(12-24 GB) and 64 GB of RAM (`S:README.md#L3-L8`, `S:README.md#L23-L28`).

**Language and size.** GitHub's language breakdown (`gh api .../languages`) is:

| language | bytes |
|---|---:|
| C++ | 1,597,840 |
| CUDA | 603,518 |
| Python | 265,239 |
| other | small |

### The engine (`src/`, `include/strata/`)

The engine is not llama.cpp. Its kernels are named after the architecture's own parts:

| kernel | model part |
|---|---|
| `qsa*` | Qwen Sparse Attention |
| `gdn*` | Gated DeltaNet |
| `gr*` | Gated Residual |
| `ple*` | the n-gram embedding |
| `router_top10` | the 10-of-512 expert router |
| `s2_expert_grouped`, `shared_expert` | the experts |
| `mtp` | the MTP draft layer |

Source: the tree listing, `gh api repos/Niko1221/Strata/git/trees/main?recursive=1`.

**What it borrows from ggml.** It uses ggml's i-quant block layouts and CUDA dot products (transcribed into
`src/kernels/cuda/iq_kernels.cu`). It links ggml's CPU backend for the i-quant experts, and uses llama.cpp's `mtmd`
for vision (`S:docs/DETAILS.md#L327-L330`). llama.cpp is fetched at pinned commit `3cf03257f`
(`S:setup.py#L46-L47`).

**Memory tiering** (`S:docs/DETAILS.md#L305-L315`):

- VRAM holds attention, DeltaNet, the routers, the shared experts, the head, the MTP layer, the KV cache, and an
  adaptive expert cache.
- RAM holds all 24,576 experts, pinned. The CPU computes the missed experts in place, concurrently with the GPU.
- The SSD holds the 28.8 GB n-gram table, read through the OS cache.
- MTP speculation drafts up to 3 tokens.
- Prompts go in 2,048-token chunks, with experts streamed over PCIe.

### The server (`serve/server.py`)

- **Process model.** Python `ThreadingHTTPServer` over a resident `strata --serve` process, spoken to over
  stdin/stdout (`S:serve/server.py#L1-L16`).
- **Endpoints** (`S:serve/server.py#L6-L7`, `S:docs/DETAILS.md#L159-L164`):
  - `POST /v1/chat/completions` (OpenAI, streaming, tools)
  - `POST /v1/messages` (Anthropic)
  - `GET /v1/models`, `/health`, `/status`
- **Not present:** the OpenAI Responses endpoint, `/v1/completions`, embeddings, and MCP.
- **Efforts.** It maps `none/low/medium/high` onto the template's `low/medium/xhigh` or `enable_thinking=false`
  (`S:serve/frontend.py#L61-L77`).

**Stated limits.** `S:docs/DETAILS.md#L203-L205` says it verbatim: "one request at a time; greedy decoding
(temperature is ignored); every request processes its whole prompt again (no conversation cache yet ...)". The README
repeats the consequence: about 1 minute per 30,000 tokens of conversation before every answer
(`S:README.md#L101-L102`).

### What it does NOT have

- **Agent, harness, memory, skills or tool features.** None found. It passes the client's tools through the chat
  template and parses tool calls back out. There is no memory, no skills, no routing, no second model.
- **Multi-model routing.** None. It advertises one model, and any name is accepted (`S:README.md#L94-L95`).
- **Multi-GPU.** Not found. It is documented for "one NVIDIA card" (`S:README.md#L4`). The CMake default
  architecture is `120` (`S:CMakeLists.txt#L42-L43`).

## 2. Maturity and trust

### Age and activity

- **Created** 2026-09-24T16:40Z; last push 2026-09-25T04:24Z. That is two days old (`gh api repos/Niko1221/Strata`).
- **History: 4 commits, all by `Niko1221`.** The first one, `f2a08d4`, adds the whole engine at once. The history is
  squashed or imported: source comments cite a "plan v0.3 P8" that is not in the repo (`S:serve/server.py#L1`).

| commit | date | change |
|---|---|---|
| `f2a08d4` | 09-24 | initial |
| `28c47b1` | 09-24 | Swift 1.5 |
| `1ee8b66` | 09-24 | README |
| `6da1f66` | 09-25 | engine v0.1.2 |

- **Contributors:** one (`gh api .../contributors`).
- **Releases:** v0.1.0, v0.1.1 and v0.1.2, each with one asset, `strata-windows-x64.zip` (~80 MB). Downloads were
  25, 46 and 222 (`gh api .../releases`).
- **Stars and forks:** 133 stars and 14 forks. The fork owners' accounts vary. The stargazer list returned HTTP 404
  from the API, so whether the stars are organic was not checked.
- **Issues and PRs:** 13 in two days, from 11 distinct users. They include:
  - bug reports: #2, #4, #6 and #11
  - third-party PRs: a conversation cache (#8), a CPU-spin fix (#9), a request race fix (#7) and V100 support (#3)

  The activity looks like real early adopters (`gh api .../issues?state=all`).

### Author

- `github.com/Niko1221`: account created 2021-07-03, 1 public repo, 2 followers, bio "AI & fun".
- There is no track record to judge by.

### Licence

- **There is no LICENSE file.** GitHub reports `license: null`.
- Yet the README says "**Free and open source.**" (`S:README.md#L13`).
- Without a licence, the code is all rights reserved by default. **We cannot legally copy it.**
- *2026-09-29: superseded. MIT since `c0107c3` (2026-09-28); see "Update 2026-09-29".*
- The one licence in the tree is ggml's MIT, at `S:third_party/ggml/LICENSE`, for the vendored `ggml-common.h`.

### Tests and CI

- **CI: none.** `.github/` returns 404.
- **Tests:** CMake registers about 30 `add_test` parity and self-tests (`S:CMakeLists.txt#L137-L428`). Some need an
  external pinned llama.cpp build (`STRATA_ORACLE_SOURCE_DIR`, `S:CMakeLists.txt#L209-L210`).
- There is no published run of these tests.

### Documentation

- The README and `docs/DETAILS.md` are clear and specific. They include a troubleshooting table and exact API
  mappings.

### Supply chain and red flags

**1. The prebuilt engine is fetched unverified.**
- setup.py downloads `releases/latest/download/strata-windows-x64.zip` (`S:setup.py#L52`, `S:setup.py#L399-L423`)
  and checks no hash, only a version and architecture read from its own `BUILD.json`.
- The script that builds it, `tools/make_release.py`, is named in `S:setup.py#L49-L51` but **is not in the repo**.
  With no CI either, the shipped binary cannot be tied to the source.

**2. Nothing else downloaded is hash-checked or pinned.**
- Model shards, the mmproj and the llama.cpp zip have no hash check (`S:setup.py#L281-L343`). The packer runs with
  `--skip-hash` (`S:setup.py#L785`).
- The Python dependencies are unpinned (`S:setup.py#L58`). Only the CUDA wheels are pinned (`S:setup.py#L55`).

**3. The installers run system-level installs.**
- `START-HERE.bat` runs `winget install ... --accept-package-agreements --silent` (`S:START-HERE.bat#L15`).
- Otherwise it downloads the python.org installer through `powershell -ExecutionPolicy Bypass` and runs it
  unverified (`S:START-HERE.bat#L20-L21`).
- `setup.sh` runs `sudo apt-get install` (`S:setup.sh#L13-L20`).
- These are not `curl | bash` from an untrusted host. They are unverified binaries from official hosts.

**4. The image loader will fetch any URL and read any local file named in a request.** `Vision.load` accepts
`http(s)://` URLs, which is SSRF, and any local path (`S:serve/server.py#L185-L195`). The server binds `127.0.0.1` by
default. But the docs recommend a cloudflared tunnel for remote use (`S:docs/DETAILS.md#L197-L201`), and then any
image file on the host becomes readable by an API client.

**5. Minor issues.**
- The API key comparison is not constant-time (`S:serve/server.py#L614`).
- The CPU pool spins at about 50% while idle: issue #4, open, with PR #9 open.
- The pool's default worker count stalled or crawled two users' machines until they set `--pool-workers` by hand
  (issues #6 and #11). One of them was an i7 13th-gen with 64 GB and a 12 GB Ada card: 9-10 tok/s by default, 35 tok/s
  after tuning (issue #6, comments).

**6. Unexplained binary data.** `data/expert-profile.bin` (130 KB) and `data/draft_vocab.bin` (162 KB) are data, not
executables. No script that generates them is in the repo.

**7. Telemetry: none found on the Python side.** The only hosts it contacts are huggingface.co, github.com,
python.org and developer.download.nvidia.com (setup.py; `S:tools/mtp_fetch.py#L22`). The web page's scripts are
inline and call only the local server (`S:serve/index.html`). The C++ was not audited line by line. A GitHub code
search for `socket|WinHttp|curl_easy|InternetOpen|getaddrinfo` returned nothing, but the index of a two-day-old repo
may not be built.

## 3. Claims vs evidence

| claim | where | evidence in the repo |
|---|---|---|
| Decode 88.7/94.6/.../56.3 tok/s and prefill 389-571 tok/s (Q2_0, 1K-262K, RTX 5070 12 GB) | `S:docs/DETAILS.md#L14-L31` | **Backed by raw rows**: `S:bench/results/2026-09-24-final/matrix.json` (per tier: prompt tokens, TTFT, spec acceptance, VRAM slots). n=1 per cell, 256 generated tokens. **The prompts are not published**: `.gitattributes` names `bench/prompts/**` (`S:.gitattributes#L2`), which is not in the tree, and no bench runner script is in the tree. Not reproducible from the repo. |
| Estimates for other GPUs (5060 Ti 16 GB, 3090) | `S:docs/DETAILS.md#L39-L54` | **Labelled estimated**, ±20%, scaled from the 5070 run (`S:bench/results/2026-09-24-final/estimates.md`). No data. |
| MTP gives 1.6-1.8x, with the answer "exactly the same" | `S:README.md#L127-L128` | Per-tier `spec_accept` and `tokens_per_round` are in matrix.json. The exactness claim is plausible for greedy decoding (verify-then-accept). A no-MTP baseline row was not found; it may be in the paper, which was not read. |
| Vision matches llama.cpp "token for token on our test images" | `S:docs/DETAILS.md#L276-L279` | `S:bench/results/2026-09-24-vision/vision.md` holds a speed table and a "same tokens" column (n=3 models). The test images are not in the repo. |
| Swift 1.5: 8/8 vs 8/8, 1,234 vs 2,682 tokens | `S:docs/DETAILS.md#L78-L80` | The author calls it "Not a benchmark". n=8, no data file. |
| Faster than llama.cpp | not claimed in README/DETAILS | No llama.cpp baseline appears in the repo's text files. |
| "Free and open source" | `S:README.md#L13` | **Contradicted** at `6da1f66`: there was no licence. MIT since `c0107c3` (2026-09-28). |

## 4. Fit against Yamadori, point by point

### What both have, and how Strata's compares

| capability | Yamadori | Strata |
|---|---|---|
| OpenAI Chat Completions | yes, streaming, tools, errors in OpenAI form (`mcp/api_errors.py`) | yes (`S:serve/server.py#L668-L687`) |
| Responses API | yes (`mcp/server.py:348`) | **no** |
| Vision | `describe_image` via mmproj on the A4000 | mmproj through a `strata-vision` helper; images decoded inline into the prompt |
| Reasoning effort | read from the served template (`tiers.accepted_efforts`) | a hard-coded map (`S:serve/frontend.py#L63-L64`) |
| Context overflow | 400 `context_length_exceeded` (`proxy.check_client_prompt`) | 400, "never truncated" (`S:serve/server.py#L351-L353`) |
| Cancel on disconnect | yes | yes (`S:serve/server.py#L685-L687`) |
| Keep-alive during prefill | heartbeats | SSE comments (`S:serve/server.py#L680`) |

### What Strata has that we lack

- **An engine that runs a 125B MoE on one 12-16 GB card**, with a GPU expert cache plus concurrent CPU expert
  compute.
- **An Anthropic Messages endpoint.** Ours has none: `mcp/server.py` routes `/v1/chat/completions`, `/v1/responses`
  and others, but not `/v1/messages`.
- **`GET /status`**: a live phase and token count.
- **A one-click installer.** It is not relevant to us.

### What we have that Strata lacks

Almost everything this repo is about:

- per-conversation slot pinning and a prompt cache (Strata re-reads the whole prompt every request)
- the ledger
- session ids
- sampling (Strata is greedy only; Qwen recommends temp 1.0, top_p 0.95, top_k 20 for thinking mode,
  `Q:README.md#L341-L343`)
- concurrent requests (Strata serves one at a time)
- the second brain (deep thinking, fan-out, fix-up)
- skills
- search
- image generation
- the reasoning budget
- compaction handling
- reproducible engine and model manifests with hashes
- the live test suite and deploy gate
- the dashboard

### Our constraints

| constraint | Strata |
|---|---|
| One box with two cards (5060 Ti 16 GB main, A4000 16 GB) | Single-GPU design. It would need the 5060 Ti alone, which is fine, but it leaves the A4000 idle for the main model. |
| Our model: ternary PTQ1_0 GGUF on a patched llama.cpp fork | **No.** Its kernels are qwen4exp-specific. It cannot load Qwen3.8-27B or PTQ1_0. |
| Any unmodified harness via standard OpenAI APIs | Chat Completions yes, Responses no. |
| Prompt cache and slot discipline | **No.** It has no cache (a third-party PR, #8, proposes one) and one sequence. |
| Reproducible builds | **No.** The release binary is unpinned, the build script is absent, and there is no CI. |
| Windows host | Yes. Windows is its primary target. |

## 5. Options (the framework)

### (a) Adopt Strata as the base and port our features onto it

**Cost.**
- Replace the model: Strata runs only Flash-Next.
- Rebuild in its engine everything llama-server gives our proxy today: slots, `/apply-template`, `/tokenize`,
  `/completion` warming, `id_slot`, the prompt cache and checkpoints, sampling, the reasoning budget, and parallel
  sequences.
- All of that on a two-day-old, single-author C++/CUDA codebase with no licence.

**Risk.** Legal: we cannot copy unlicensed code. Maintenance: the bus factor is 1, and the code is one architecture's
kernels. Security: see section 2. **Not viable.**

### (b) Borrow specific ideas (not code)

Licence compatibility: none for Strata's own code. The ggml-derived parts are MIT, but we would take those from
llama.cpp itself. We could reimplement these ideas or ask the author to add a licence:

1. **An Anthropic Messages endpoint** as a thin translation in front of our Chat Completions path (`S:serve/frontend.py#L167`
   `anthropic_to_messages`; `S:serve/server.py#L504-L592` `anthropic_events`/`anthropic_collect`). It would let Claude
   Code-style clients connect without a shim. Our own work, our own tests.
2. **`GET /status`**: current phase (reading the prompt / thinking / answering / writing a tool call), tokens so far
   and tok/s (`S:serve/server.py#L356-L387`, `#L631-L641`). Useful on the dashboard for multi-minute prefills.
3. **Its image-format normalisation list** (WebP/TIFF/AVIF → PNG before the encoder; `S:serve/server.py#L198-L224`).
   Only if our vision path ever needs it.
4. **The expert-cache design**, for the model question, not the framework: only if we serve a large MoE and upstream
   llama.cpp offload proves too slow. It is described in `S:docs/DETAILS.md#L305-L315` and
   `S:include/strata/core/expert_cache.hpp`. Read it for design; do not copy it.

### (c) Ignore it

Cost: nothing. Risk: we miss the engine's speed if Flash-Next becomes our model. That speed does not help agentic use
without a prompt cache, and upstream llama.cpp already runs the model (section M).

### Recommendation for the framework

**(c) plus the two small ideas in (b):** an Anthropic Messages endpoint and a status endpoint, both our own code.

**What would change it:**
- Strata gains a licence, a published build and benchmark pipeline, and PR #8's conversation cache, plus sampling and
  parallel sequences.
- And a paired measurement shows its engine is much faster than upstream llama.cpp for Flash-Next on this box at
  agentic context depths.

**A sandbox experiment, described and not run.**
- Set-up: on an idle stack, in a maintenance window, one GPU consumer at a time (PROTOCOL rule 4). Run Strata's
  release binary in a throwaway VM or a separate folder, not in this repo, against IQ2_XS.
- Compare it with upstream llama.cpp at the same commit Strata pins (`3cf03257f`) with experts on the CPU.
- Measure decode and prefill at 4K, 32K and 128K, n=3 each, same prompts.
- Then measure a 10-turn agent replay's total time. That is where Strata's missing cache shows.

---

## M. The model: Qwen3.8-Flash-Next

### M1. What Strata runs

**Base model: `Qwen/Qwen3.8-Flash-Next`**, revision `de4b8e4`, created 2026-08-24.
- Parameters: "125B with 6B activated, plus 51B n-gram embedding and 4B MTP" (`Q:README.md#L46`). The safetensors
  total is 179,999,981,424 BF16 parameters over 131 shards, 360 GB (HF API).
- Architecture: `qwen4_exp`, 48 layers laid out as 12 × (3 Gated DeltaNet + 1 Qwen Sparse Attention).
- Experts: 512 per layer, 10 routed + 1 shared.
- Other parts: a Gated Residual with 4 branches, and an n-gram embedding (20M bigrams/trigrams at layer 2).
- Context: 262,144 native, extensible to 1M. It has a vision encoder (`Q:README.md#L41-L71`, `Q:config.json`).
- Card: "This experimental preview of the architecture that will underpin Qwen4" (`Q:README.md#L26`).

**The files Strata downloads: `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`**, revision `c67535c`
(`S:setup.py#L45`, `S:setup.py#L61-L82`). They are GSQ + RCO non-uniform quants, two shards each. Shard 2 is the
n-gram table: 28.8 GB, IQ4_NL, identical across sizes (`I:README.md#L69-L77`).

| quant | bpw | total | resident (shard 1) |
|---|---:|---:|---:|
| Q2_0 | 2.40 | 66.4 GB | 37.6 GB |
| IQ2_XS | 2.50 | 68.0 GB | 39.2 GB |
| IQ3_XXS | 3.00 | 75.8 GB | 47.0 GB |
| IQ3_S (not offered by Strata) | 3.50 | 83.6 GB | 54.8 GB |

Plus the mmproj, BF16, 0.91 GB.

**Strata's defaults.** It recommends IQ2_XS (`S:README.md#L75-L88`). It grafts the 4B MTP layer out of the BF16
checkpoint by HTTP range requests, because the GGUF ships none (`S:tools/mtp_fetch.py#L1-L10`).

**Alternative weights: Swift 1.5** (`ukisai/...-GSQ-RCO-GGUF`, `b22d729`), a fine-tune claimed to use "63.4% fewer
thinking tokens ... accuracy loss <1% vs base on xhigh" (`U:README.md#L33`). Its evaluation is KLD against its own
BF16 reference, "not direct capability rankings" (`U:README.md#L68-L73`).

### M2. Licence and provenance

**Base licence: "Qwen Community License 1.0"** (`Q:LICENSE`). It is permissive: use, modify, host and sell are all
allowed. It adds two conditions:

- Attribution above 100M MAU or US$20M monthly revenue.
- The one that matters to us, quoted from `Q:LICENSE#L9`: "If the licensee or any of its affiliates conducts a Model
  as a Service or AI Work Assistant business, the licensee shall obtain a separate license from Qwen before Using the
  Software ... for any commercial purpose. The foregoing requirement shall not apply to the licensee's internal Use
  ... provided that such Use does not make the Software, its outputs, or its underlying model capabilities available
  to any third party."
- "AI Work Assistant" explicitly means "AI-assisted coding" products (`Q:LICENSE#L12`).

**What that means for us.** Personal or internal use is fine. Offering Yamadori to third parties commercially would
need a Qwen licence. Our current base, Qwen3.8-27B, is Apache-2.0 (docs/SWE-BENCH.md:106). This is a real change.

**Quant licence metadata contradicts itself.** The ISTA card's metadata says `license: apache-2.0`
(`I:README.md#L6`). Its text says the weights "inherit the license of the base model" (`I:README.md#L224-L226`). The
base licence governs.

**Swift 1.5** adds the Swift Open License 1.0 on top of the Qwen licence (`U:LICENSE`).

**Provenance summary.**

| layer | who | notes |
|---|---|---|
| Base weights | Qwen | the official repo, 1.07M downloads |
| Quants | IST Austria DASLab | a research lab. The methods are papers: arXiv 2604.18556 and 2605.00649. The per-tensor allocation is published (`I:README.md#L187-L193`). **There are no SHA256 sums.** |
| Swift 1.5 | UkisAI | ships `SHA256SUMS` and a release manifest (HF file list) |

### M3. Published evaluations

**All of them are self-reported. None are independent.**

**Qwen's table (`Q:README.md`, "Benchmark Results").** BF16. Qwen's own runs. Claude Code or mini-SWE-agent
harness, temp 1.0, top_p 0.95, 256K context. The 27B column is **our model's base** at full precision.

| benchmark | Flash-Next | Qwen3.8-27B | Δ |
|---|---:|---:|---:|
| DeepSWE 1.1 | 58.7 | 42.2 | +16.5 |
| SWE-bench Pro | 62.5 | 61.7 | +0.8 |
| SWE-bench Multilingual | 81.0 | 73.8 | +7.2 |
| NL2Repo-Bench | 48.1 | 42.3 | +5.8 |
| CoWorkBench (in-house) | 73.9 | 70.7 | +3.2 |
| JobBench | 55.7 | 33.4 | +22.3 |
| Toolathlon Verified | 73.5 | 67.1 | +6.4 |
| IFBench | 81.3 | 79.5 | +1.8 |
| GPQA Diamond | 91.7 | 89.2 | +2.5 |
| HLE | 35.9 | 30.8 | +5.1 |
| LiveCodeBench v6 | 91.9 | 90.3 | +1.6 |

The vision rows are similar: RecreationBench 49.9 vs 47.1, Vision2Web 64.0 vs 62.9.

**ISTA's quant table (`I:README.md#L115-L131`).** Their own runs, relative to BF16. The harness is not stated, and
throughput was measured "with llama.cpp".

| variant | AIME25 | GPQA-D | LCB v6 |
|---|---:|---:|---:|
| BF16 | 100.00 | 91.92 | 87.43 |
| Q2_0 | 96.67 | 89.39 | 81.14 |
| IQ2_XS | 96.67 | 87.37 | 83.43 |
| IQ3_XXS | 100.00 | 91.41 | 86.29 |

ISTA's BF16 LCB v6, 87.43, differs from Qwen's 91.9. Their harnesses differ, so numbers must not be mixed across the
two tables.

### M4. How it compares with ours

**Our model.** `BoldingBuilds/Ternary-Bonsai-2-27B-Abliterated-PTQ1_0-GGUF`: PrismML's ternary requant of
Qwen3.8-27B, abliterated, 1.75 bpw, with an MTP head grafted in (docs/SWE-BENCH.md:102-108; config.yaml:168-172).

**Like-for-like evidence that exists:**

1. **Base against base, from one source (Qwen).** Flash-Next BF16 beats Qwen3.8-27B BF16 on every listed row. The
   margins are large on agentic coding and tool use (DeepSWE +16.5, SWE-bench Multilingual +7.2, Toolathlon +6.4)
   and small on SWE-bench Pro (+0.8) and LCB v6 (+1.6). This is the best available evidence for the operator's
   belief. It is self-reported, n is not stated, and it is for BF16.
2. **Quantised against quantised, only on AIME25 and LiveCodeBench, and across sources.**
   - PrismML (`B:README.md`, "Full Per-Benchmark Results": EvalScope + vLLM on H100, thinking mode) reports Bonsai 2
     27B at AIME25 95.00 and LiveCodeBench 90.07, against 27B FP16's 96.67 and 90.05. That is essentially no loss on
     these two.
   - ISTA reports 2-bit Flash-Next losing 3.3 on AIME25 and 4.0-6.3 on LCB v6 against its BF16. IQ3_XXS loses 0 and
     1.1.
   - The LCB versions and harnesses differ (PrismML does not name the version), so no subtraction across the two
     tables is valid.
   - Qualitatively: **at 2 bits, Flash-Next's small LCB lead over the 27B (+1.6) is smaller than its reported quant
     loss (-4 to -6).** Its large agentic leads were **never measured after quantisation** by anyone.

**Our own measurements of Bonsai.** None are on a benchmark Qwen or ISTA report, so none compare directly:

- **Octopus V0:** the `v0e-V0-xhigh-1` final graded 19/27 after prompts 1 and 2 (bench/octopus/results/grades.jsonl
  rows 20-21; wall time about 4.0 h and 5.4 h, runs.jsonl rows 6-7). **Caveat:** the reference solution
  `fx-v0-ref` graded 19, 21 and 23/27 on three consecutive grader runs (grades.jsonl rows 4-6). The grader may have
  changed between them, but the spread means a difference of a few points at n=1 is not a result.
- **LiveBench coding sample, n=21:** bare 70.9 vs xhigh 72.3 (bench/livebench/README.md:159-164).
- **SWE-bench Verified:** 2/2, which is not an estimate (docs/SWE-BENCH.md:16, :453-456).
- **Contamination:** three.js is contaminated for every model.

**No Flash-Next run exists on any of these.** The comparison the operator wants does not exist yet (section M7).

### M5. Hardware fit

**Our box.**
- Main card: RTX 5060 Ti, 16 GB (config.yaml:59, :184). Latest reading: 14,472 MiB used at 163,840 context q8_0
  (config.yaml:250-252).
- Second card: RTX A4000, 16,376 MiB (mcp/gpu_room.py:18), full of other services.
- Host, read 2026-09-26 with `Win32_ComputerSystem` / `Win32_PhysicalMemory`: **63.7 GB RAM** (4 × 16 GB
  DDR5-4800) and an i7-13700K, **AVX2, no AVX-512**.
- Free disk: C: 952 GB, D: 3,449 GB.

**The weights do not fit in VRAM on either card or both.** Even Q2_0's resident shard is 37.6 GB. So the experts
live in system RAM and the CPU computes most of them. This is the design both Strata and llama.cpp's `-ot`/`--n-cpu-moe`
use.

**RAM.** Q2_0 or IQ2_XS need about 34-36 GB of experts plus about 6 GB (`S:docs/DETAILS.md#L61-L67`), roughly
40-42 GB of 63.7 GB. That leaves about 22 GB for:

- Windows
- the n-gram table's page cache (28.8 GB wanted; it would page from the NVMe per token, `I:README.md#L83-L94`)
- the proxy, tools API, worker and SearXNG
- **Docker/WSL2 for Octopus runs**

**Tight.** Strata's own docs say to close the browser for IQ3_XXS, which does not fit alongside our stack. **Our
stack's own RAM footprint was not measured for this note.**

**VRAM on the 5060 Ti.** Estimated from `Q:config.json`, not measured:

- **Full-attention KV.** 12 layers × 2 KV heads × 256 dims × (K + V) = 12,288 values per token. At q8_0 (8.5 bits)
  that is about 13 KB per token. **At our `-c 181248`, about 2.4 GB**; 3.4 GB at 262K. Compare our budget's 44
  KiB/token for Bonsai (mcp/budget.py:126). The model's KV is about 3.4× cheaper per token.
- **QSA indexer keys** (1 head, 128 dims, compress ratio 4): about 0.1 GB at 181K.
- **DeltaNet state:** 36 layers × 48 heads × 128 × 128 × fp32, about 113 MB per sequence. That agrees with Strata
  PR #8's "about 118 MB" checkpoint.
- **Dense weights:** shard 1 minus experts is about 3.6 GB at Q2_0 (37.6 − 34.0). The MTP layer is about 1-1.5 GB
  quantised (a guess), plus compute buffers.

**Total:** about 8-9 GB before any expert cache, leaving about 6-7 GB of the 16 GB for GPU-resident experts. Strata
fitted about 4,500 of 24,576 expert slots on a 12 GB card at 4K (`S:bench/results/2026-09-24-final/matrix.json`,
`vram_slots`).

**Decode speed.**
- Strata **estimates** our exact main card at about 87 tok/s at 4K and about 62 at 128K (Q2_0), about 77 and 51
  (IQ2_XS). The estimate assumes an AVX-512 Ryzen 5 7600 and DDR5-5200 (`S:docs/DETAILS.md#L44-L48`, ±20%).
- **The closest real datapoint to our CPU** is issue #6: i7 13th-gen, 64 GB DDR5 and a 12 GB card on IQ2_XS got 9-10
  tok/s, then 35 tok/s after tuning.
- ISTA's llama.cpp figure, Q2_0 at 367 prefill / 93.8 decode (`I:README.md#L102-L105`), states **no hardware**.
- **For comparison, Bonsai on the 5060 Ti:** 55.33 tok/s decode and about 490 tok/s prefill (docs/KNOWN-ISSUES.md:42-43).
- Honest range for Flash-Next here: somewhere between about 30 and about 80 tok/s decode. **Unmeasured.**

**Prefill.** Upstream llama.cpp has an open report that decode slows linearly with context for this model on CUDA
(ggml-org/llama.cpp#28734, open, 7 comments).

**Using both cards.** The main model across both cards (`-ts`) would hold far more experts in 32 GB of VRAM. But it
would evict embeddings, reranker, vision and image generation from the A4000. That is an operator decision, and a
separate test.

### M6. What serving it through OUR stack would take

**The engine.**
- Neither of our llama.cpp forks can load it. `src/models/qwen4exp.cpp` is absent in PrismML `prism@9a9394a`
  (2026-09-18; 404) and in `sudoingX/llama.cpp@285542d` (404).
- Upstream llama.cpp added `qwen4exp` on 2026-08-27 (ggml-org/llama.cpp PR #27742, commit `6c84c7d5d`), with fixes
  through `3cf03257f` (2026-09-20, "CUDA: enable sparse fa for qwen4"). That is Strata's pin.
- So: a **new `engines/manifest.yaml` entry**, `llama-upstream`, for upstream at a pinned commit, built with our
  toolchain `msvc-cuda128` and `CMAKE_CUDA_ARCHITECTURES=120`. It becomes a second shipped `llama-server.exe` that
  `deploy_check.py` will accept.
- MTP for Flash-Next upstream is in flux: PR #28243 is open, and issue #29148 reports an MTP load regression. The
  ISTA GGUF has no MTP head. A first test runs **without MTP**.
- Our patch `engines/patches/llama-bonsai2/0001-reasoning-budget-nudge.patch` (+494/-22) is Bonsai-fork-specific. It
  would need porting, or the nudge would be off for this arm. Whether upstream at the pin serves per-request
  `reasoning_budget_tokens` natively must be checked in its `server-schema.cpp` before the test.

**The model files.** New `models/manifest.yaml` entries: both shards and the mmproj, with sha256. ISTA publishes no
sums, so we record our own at download and pin the revision `c67535c`. The licence is recorded as Qwen Community 1.0.

**config.yaml.** A new model entry, say `flashnext`, bound to the 5060 Ti's UUID, with:
- `-ot 'exps=CPU'` or `--n-cpu-moe N`
- `-ot per_layer_token_embd\.weight=CPU` (the form used in llama.cpp#28734), with the n-gram shard memory-mapped
- `-c 181248`, `--cache-type-k/v q8_0`
- the same `--jinja --reasoning-format deepseek`
- the sampler macro: Qwen recommends the same 1.0/0.95/20/0.0 for thinking mode (`Q:README.md#L341-L343`)

**What in `mcp/` is Bonsai-specific, and what a swap touches.** Surveyed for this note:

| area | where | a swap to Flash-Next |
|---|---|---|
| Template markers `</think>`, `<think>`, `<|im_start|>`, `<|im_end|>` | proxy.py:4769, :4993 (`STRAY_MARKERS`), :6479 (`END_OF_TURN`), :6684-6685 (`USER_TURN`, `TOOL_TURN`); `<think>` stripping in worker.py:359/761, skill_prompts.py:138/218/265, skill_pipeline.py:344, skill_select.py:425, skill_builder.py:119 | **Probably unchanged.** Flash-Next's template uses the same ChatML markers, `<think>\n...\n</think>\n\n` and the same `<tool_call><function=...>` format (`S:serve/chat_template.jinja#L73-L83`, `#L112-L122`). Verify byte-for-byte against the served template. |
| Efforts | tiers.py:313-364 parses `reasoning_effort not in (...)` from `/props` | Flash-Next's template says `resolved_reasoning_effort not in ('xhigh', 'medium', 'low')`, the same set (`S:serve/chat_template.jinja#L59-L64`). The regex should still match the substring. **Its default is `xhigh`** (`#L59`), so a request that sends no effort thinks at xhigh unless tiers sends one. |
| Preserved thinking | the ledger's pass-through design | Flash-Next's template renders every past `reasoning_content` by default (`preserve_thinking`, `S:serve/chat_template.jinja#L119-L120`; `Q:README.md#L553-L556`). Hermes strips it, so the prompt is unchanged. Our cache analysis should be re-checked. |
| Template fixture | mcp/fixtures/bonsai_chat_template.jinja (proxy.py:4740, system_roles.py:4) | A second fixture; the offline ledger and stream gates re-run against it. |
| llama-server endpoints | `/apply-template`, `/tokenize`, `/completion` + `id_slot`, `/props` (proxy.py:3588/3594/6577/6704/6803, budget.py:152-161, slots.py) | Present in upstream llama-server. **Hybrid-model checkpoint and slot reuse must be re-measured** (STEP 0's probes), because the recurrent state differs. |
| Reasoning budget and nudge | tiers.py:515-548, :620-621; compaction.py:610-611 | Needs the patch ported, or the nudge disabled for this arm. |
| MTP | config.yaml:168-174 (`--spec-type draft-mtp`, draft KV q4_0) | Off for the first test (above). |
| Concept seeds | concept_seed.py (5120-d, `index/token_embd.npz`); scripts/extract_token_embd.py:158-163 refuses anything but PTQ1_0 | Re-extract: 2560-d, vocab 248,320 (`Q:config.json`). The extractor must accept the quant's `token_embd` type. |
| KV and pool budget | budget.py:119-126 (`HELPER_TOKENS` 49,152, `KV_KIB_PER_TOKEN=44`) | KV is about 13 KB/token, so the same VRAM allows a much larger pool. Re-derive; the pool size is read from `/props` anyway. |
| Model names | model.py:54/59, tiers.py:329, catalog.py:62/87-90/102/317 (`== "bonsai"`) | Point `YAMADORI_MODEL` at the new entry. catalog.py:317 hard-codes `"bonsai"`. |
| Vision | vision.py:99, :400-444 (mmproj name, `-c 16384`); gpu_room.py:175-180 | Flash-Next has its own mmproj, BF16 0.91 GB. Either keep Bonsai-vision on the A4000 (it is a separate model already) or add a `SIZES` row. Unchanged for a first test. |
| Sampling | config.yaml:69-71, tiers.py:711/716 | Same vendor values. Unchanged. |

### M7. The cheapest decisive test

Described here, not run. Per the fix-before-run rule it is queued behind in-flight fixes, and it needs an operator
decision to download about 68 GB and build one engine.

**Arms.**
- **Bonsai**, exactly as deployed.
- **Flash-Next IQ2_XS** on upstream llama.cpp at a pinned commit, with no MTP.

Both go through `:1234`, with the same Hermes profile, the same effort and the same sampler. They run one at a time,
since there is one GPU consumer, and are switched by `YAMADORI_MODEL` or the catalog. IQ2_XS is the arm because Strata
recommends it and it is the best 2-bit on LCB. Q2_0 is a speed arm if IQ2_XS decodes too slowly.

**Step 0: a gate before anything expensive.** About an hour.
1. The offline suites against the new template fixture (`run_tests.py`).
2. A throughput probe: decode and prefill at 4K, 32K and 128K, n=3.
3. A cache probe: STEP 0's prefill, warm and diverge requests, reading `x_yamadori.cache`.
4. RAM headroom with Docker running.

**Stop if any of these is true:**
- decode below about 20 tok/s at 64K
- a request that should extend the slot reprocesses the whole prompt
- the host pages under an Octopus-sized Docker load

**Step 1: Octopus V0, paired.**
- Same prompt (`prompt_sha256 d30c0264...`), `--arm xhigh`, `--iterative 6`.
- **n ≥ 2 per model**, interleaved, so rule 10 is met.
- Record: `grade.py` spec_passed/27, wall time, tool calls, completion and reasoning tokens (relay `usage`), and
  deep-thinking runs.
- Grade the reference fixture in the same session to calibrate grader spread.
- **Decision rule, set before the runs:** switch only if Flash-Next's mean exceeds Bonsai's by more than the spread
  seen on the reference fixture, at a wall time no worse than about 1.5×.

**Step 2: cheap corroboration on existing suites.**
- `bench/livebench` bare arms, paired n=21, both models.
- `bench/domain` (TypeScript/Rust/TSL domains, the stated target).
- A 10-20 instance SWE-bench Verified slice through the mini-swe-agent runner (docs/SWE-BENCH.md: "the runner only
  needs an OpenAI endpoint"), the same instances for both.

**What each outcome means.**
- **Flash-Next wins step 1 and does not lose step 2:** plan the migration (M6) and the licence check.
- **It ties or loses:** its BF16 lead does not survive 2-bit quantisation and CPU offload on this box. Bonsai stays.
- **Step 0 fails on speed:** the Strata-vs-llama.cpp sandbox experiment in section 5 becomes the next question.
  Without a cache it still cannot serve agents.

### Recommendation on the model

Run M7. The prior is real: in Qwen's own same-harness table, the model beats our base by 16.5 points on DeepSWE and
7.2 on SWE-bench Multilingual. It thinks at 6B active parameters, and its KV cost per token is about a third of
Bonsai's.

The counter-weights are just as concrete:
- the 2-bit quant losses ISTA reports
- CPU-offload decode on an AVX2 box, where the only nearby datapoint is 35 tok/s after tuning
- RAM contention with the rest of the stack on 64 GB
- a one-month-old upstream implementation with open bugs
- no MTP at first
- a licence that restricts commercial coding-assistant use, where the base of Bonsai is Apache-2.0

None of these is measured here. Do not switch until step 1 says so.
