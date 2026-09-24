# Image generation

Qwen-Image-2.1 on the A4000 (CUDA1) through stable-diffusion.cpp's
`sd-server`, beside the resident rootstock and Laya. It is reached two ways:

- **`generate_image`**, a model tool that the proxy runs. This is the primary
  path: it needs no client configuration.
- **`POST /v1/images/generations`**, the OpenAI Images API.

Results are served from **`GET /media/<sha>.png`** through signed capability
URLs.

**Two image models** (2026-09-23), one loaded at a time:

| id | public name | llama-swap model | steps | 1024x1024 | licence |
|---|---|---|---:|---|---|
| `base` | `yamadori-image` | `imagegen` | 20, cfg 6 | 105-111 s, n=10 | Qwen Research License |
| `turbo` | `yamadori-image-turbo` | `imagegen-turbo` | 4, cfg 1 | 23.6 s, **n=1** smoke test (sd-cli) | Qwen Research License |

Each caller picks one; see "Two models: choosing, and the default".

Status on 2026-09-23 01:15:
- **Built, measured and tested offline.** Not switched on: `YAMADORI_IMAGEGEN_URL`
  is unset, so the tool stays hidden.
- **The proxy restarted after the reboot and is running this code.** The
  descriptor lists `/v1/images/generations`. Without a key the route answers
  401. `/media` answers 403 for a bad signature and 404 for a non-id.
- **Switching it on** is the "Go live" section at the end.

---

## What was downloaded

Everything is in `C:/Users/jwals/textgen/user_data/models/qwen-image-2.1/`. Each
file was checked against `https://huggingface.co/api/models/<repo>/tree/main`,
which gives the byte size and the LFS sha256. All three matched after the
2026-09-22 thermal shutdown.

| file | repo @ revision | bytes | sha256 | licence |
|---|---|---:|---|---|
| `qwen-image-2.1-Q5_K_M.gguf` (denoiser, 7B single-stream DiT) | `unsloth/Qwen-Image-2.1-GGUF` @ `2c31ccd` | 5,390,223,072 | `4b53321654dd3bf0aa8dd6bb821fdcb1c9053ca735070e359a23cbb9f97268ce` | **Qwen Research License** (`license: other`, `license_name: qwen-research`). It is **not** Apache. |
| `vae/qwen_image_2.1_vae_bf16.safetensors` | `unsloth/Qwen-Image-2.1-FP8` @ `9e52064` | 675,508,656 | `71879ffd5321e6d10c3c87513e2b474b1252efa7f3dec2969214a9bf06a6dd5c` | The card declares no licence. It is a derivative of `Qwen/Qwen-Image-2.1`, so treat it as Qwen Research License. |
| `qwen3vl_8b_heretic-Q4_K_M.gguf` (text encoder) | `pottokao/Qwen-Image-2.1-Text-Encoder-Heretic-GGUF` @ `68ceaa7` | 5,027,785,376 | `1338274ac7a6344f262a16c7a52d1bd7fe789307d252733b23ea421126e5d343` | Apache-2.0. It is Qwen3-VL-8B-Instruct with refusal directions ablated (Heretic: 5/100 refusals, KL 0.022, per its card). |

**The turbo denoiser** (downloaded 2026-09-23, operator-approved), same folder:

| file | repo @ revision | bytes | sha256 | licence |
|---|---|---:|---|---|
| `qwen_image_2.1_turbo_Q5_K_M.gguf` (Viggle's DMD-distilled 4-step student, full transformer) | `Abiray/Qwen-Image-2.1-viggle-4-steps-turbo-GGUF` @ `0016110` | 5,007,399,712 | `92521fe8d6e25cd8e23dae57f077cac29d07d23ee128c50e79d9b1e60343888a` | **Qwen Research License** (`license: other`, `license_name: qwen-research`, `base_model: Viggle/Qwen-Image-2.1-viggle-turbo`). Viggle's `LICENSE` is the base's, unchanged. Non-commercial: research or evaluation. |

Size and sha256 matched the tree API. sd.cpp `c92d73c` loads it as
`Version: Qwen Image 2.1` (297 tensors; the unsloth base has 265, because this
one keeps the MLP split). The research, the schedule fix and the LoRA
incompatibility are in `docs/IMAGEGEN-TURBO.md`.

**Text encoder.** The Heretic GGUF loads and works in sd.cpp:
- `llm: num_layers = 36, hidden_size = 4096`, and `no vision weights detected, vision disabled`, which is expected with no `--llm_vision`.
- Every sample below is coherent, and text renders correctly.
- The card's warning that GGUF cannot be the text encoder applies to ComfyUI only.
- The stock `unsloth/Qwen3-VL-8B-Instruct-GGUF` fallback was **not** downloaded because it was not needed.

## Runtime: stable-diffusion.cpp

- **Source:** `C:/Users/jwals/stable-diffusion.cpp`, commit
  `c92d73c408515c94beef32161bb5960764fde7a0` (2026-09-23 02:23 +0800), with the
  `ggml` submodule at `4bf5f60`.
- **Qwen-Image-2.1 support:** `docs/qwen_image_2.1.md` in that tree covers
  text-to-image and editing, with the Qwen3-VL LLM encoder passed as `--llm`.
  Both image edges must be multiples of 32.
- **Build:** `build-cuda.bat` in that folder uses the same toolchain as
  `llamacpp-prism-official`:
  - MSVC 2022 Enterprise, with the cmake and ninja bundled in VS
  - nvcc 12.8 from `textgen/installer_files/cudabuild`
  - `-DSD_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="86;120" -DCMAKE_BUILD_TYPE=Release`, built with `-j 12`
- **Outputs:** `build/bin/sd-cli.exe` and `build/bin/sd-server.exe`.
  `cudart64_12.dll`, `cublas64_12.dll` and `cublasLt64_12.dll` were copied in
  beside them. Without them the binaries exit with `0xC0000135`.

### sd-server API (what `mcp/images.py` uses)

`sd-server` exposes three API families (`examples/server/api.md`):
- OpenAI: `POST /v1/images/generations`, `POST /v1/images/edits`, `GET /v1/models`
- SD-WebUI: `POST /sdapi/v1/txt2img`, `img2img`, and others
- native async: `/sdcpp/v1/*`

It has no `/health` endpoint, so the llama-swap entry checks `/v1/models`.

**Its default port is 1234, which is the proxy's.** Always pass
`--listen-port`.

The proxy calls **`POST /sdapi/v1/txt2img`** with this body:

```json
{"model": "imagegen", "prompt": "...", "width": 1024, "height": 1024,
 "steps": 20, "seed": 42, "batch_size": 1}
```

For turbo, `model` is `imagegen-turbo` and `steps` is 4. Steps are per model
(`images.MODELS`): the proxy sends `steps` on every request, so the server's
own `--steps` never applies to a proxy request.

The answer is `{"images": [<base64 PNG>...], "parameters": {...}, "info": "<json string>"}`.

- **Why sdapi and not OpenAI:** sdapi takes `seed` and `steps` as fields. The
  OpenAI route only takes them embedded in the prompt as `<sd_cpp_extra_args>`.
- **The `model` field:** llama-swap routes on it, and sd-server ignores it.
- **Defaults:** `cfg_scale`, the sampler and turbo's
  `--extra-sample-args` come from each server's command line, because the
  body sends none of them (sd-server copies its command-line defaults and
  then applies the body's fields, `routes_sdapi.cpp:110-117`).
- **Verified end to end:** `bench/imagegen/smoke_server.py` ran the real
  sd-server with the proposed flags and called `images.generate` twice. Both
  returned PNGs and metadata, at 107.6 s and 105.8 s.

---

## Measurements

The harness is `bench/imagegen/measure.py` (sd-cli). Every row is in
`bench/imagegen/results.jsonl`, and the PNGs and per-run logs are in
`bench/imagegen/samples/`.

**Fixed settings:** Q5_K_M denoiser, Heretic Q4_K_M encoder, bf16 VAE,
1024x1024, 20 steps, cfg 6.0, euler, `--diffusion-fa`, seed 42. The three
prompts were:
- a fox (photographic)
- an enamel sign that must read **YAMADORI BONSAI** (legible text)
- a paper-crane icon

**Sampling:** CUDA1 was sampled every 0.25 s with nvidia-smi. "Peak Δ" is the
highest used memory minus the used memory before start. CUDA0 was hidden from
every run (`CUDA_VISIBLE_DEVICES`).

**Baseline drift:** CUDA1's resident baseline moved between runs: 7,555, then
6,699, then 4,931 MiB. The comparison is therefore made on **Δ**, not on
absolute use.

### The three options the operator asked about

**Option (a): text encoder on the CPU; denoiser and VAE resident on the GPU.**

`--auto-fit off --backend all=cuda0,te=cpu`

| run | n | wall per image | encode (CPU) | sampling | s/step | VAE decode | peak Δ | result |
|---|---|---:|---:|---:|---:|---:|---:|---|
| untiled VAE | 3 | 107–115 s | not reached | 103.6–111.7 s | 5.2–5.6 | **failed** | 6,281–6,287 MiB | Decode failed on all 3. |
| `--vae-tiling --vae-tile-size 16x16` | 5 | 113.3–115.3 s | 2.3–2.9 s | 95.7–97.5 s | 4.8–4.9 | 13.7–14.3 s | 6,347 MiB | All 5 OK. |

- **Why the untiled decode failed:** it asked for 8.7 GB, and the default
  tiles for 2.9 GB, while the denoiser's 5.1 GB stayed resident. Only 2.7 GB
  was free.
- **Encoder share of wall time:** 2.3–2.9 s of ~114 s, about **2%**. The CPU
  encoder is not slow at prompt lengths like these. The cost that was feared
  does not exist here.
- **Tile artifact:** the 16x16 tiles left a small white-square artifact on
  one of the two sign images. It was the first sighting of the F16 VAE
  overflow described in "White blocks" below, at (384, 640) on 128-px tile
  seams (`bench/imagegen/white_blocks.py` counts 2 tiles).
- **Killed run:** a sixth (a) run is not counted. It was killed by the
  harness because CUDA1 fell to 190 MiB free. Something else loaded about
  2.9 GB on CUDA1 during it (peak 15,977 MiB against this config's usual
  13,046). That was a second consumer, not this config. See "Incidents".

**Option (b): everything on the GPU, with weights kept in RAM and staged in.**

`--backend cuda0 --offload-to-cpu --max-vram 6`

| run | n | wall | encode (GPU) | sampling | s/step | decode | peak Δ | result |
|---|---|---:|---:|---:|---:|---:|---:|---|
| **budget 6 GiB** | 3 | **109.4–110.4 s** | 2.2–2.3 s | 99.4–100.1 s | 5.0 | 6.8–7.1 s | **6,281–6,287 MiB** | All OK. |
| no `--max-vram` | 1 | killed | 2.3 s | 103.7 s | 5.2 | not reached | **8,491 MiB** | **Killed at 124 MiB free.** |

- **Transfer overhead:** sampling takes about 3 s longer than (a), roughly 3%.
  It is repaid in the decode. Once the denoiser's replicas are released, the
  VAE's own untiled attempt still fails, and sd.cpp retries with its default
  tiles in about 7 s, against 14 s for (a)'s small tiles.
- **Net per image:** (b) is about 4 s faster than (a). At n=3 against n=5 that
  is suggestive, not a result.
- **Without a budget,** offload keeps replicas up to whatever the card has
  free. Never run it without `--max-vram`.

**(a)+(b) combined: encoder on the CPU plus a staged denoiser.**

`--backend all=cuda0,te=cpu --offload-to-cpu --max-vram 6`

| n | wall | encode (CPU) | sampling | decode | peak Δ |
|---|---:|---:|---:|---:|---:|
| 3 | 109.2–110.9 s | 2.3–2.9 s | 98.5–100.0 s | 6.8–7.1 s | 6,283–6,287 MiB |

It is indistinguishable from (b) on wall time and peak.

**Option (c): evict the retrieval group.** This was a paper figure only; the
operator ruled out eviction. It is not needed: the image stack's peak Δ fits
beside everything that is resident.

### Also measured

| what | n | numbers |
|---|---|---|
| sd-server, (b) flags, through `images.generate` | 2 | 107.6 s and 105.8 s end to end; peak Δ 6,371 MiB; `/v1/models` ready 0.6 s after start (weights load lazily on the first request, about 2 s) |
| sd-server idle after a generation | 1 | **+319 MiB** held; the weights and compute are released at the end of the run, and the context stays. A 512x512 image took 28.4 s. |
| **1344x1344**, (b) | 1 | 220.5 s wall, 10.3 s/step (2.1x for 1.72x the pixels); peak Δ 6,389 MiB (the budget holds); VAE retiled automatically |
| GPU temperature | 13 | peak 69–81 C on the A4000 (the harness kills a run above 88 C) |

### Decision: (b), with a 6 GiB budget

- **Speed does not decide it.** (b), (a)+(b) and (a) with tiling land within a
  few seconds of each other at n=3–5. Sampling is about 95% of wall time,
  identical in every option, and bound by the A4000.
- **(b) is chosen on the other axes:**
  - Its VRAM is **bounded by a number sd.cpp enforces**. The budget caps
    managed weights and runner buffers. Live free memory can lower it
    further but never raise it.
  - The VAE falls back to tiling on its own. *"No artifact seen" was wrong
    (corrected 2026-09-24):* production later showed solid white 8-px
    blocks on those tile seams, in 10 of 55 stored images. Cause and fix:
    "White blocks" below.
  - The 2 s encode uses the GPU instead of 8 CPU threads. That matters on a
    box that shut down thermally during this benchmark.

**Coexistence (no eviction):**
- **Worst-case resident baseline:** 7,565 MiB, from config.yaml's measured
  retrieval + Laya figure.
- **Largest peak Δ measured under the budget:** 6,389 MiB (at 1344x1344).
- **Capacity:** the card showed 16,167 MiB of used + free. The remaining
  headroom is 16,167 − 7,565 − 6,389 = **2,213 MiB free at peak**, which is
  over the 1 GiB floor.
- **At today's 6,699 MiB baseline,** the same sum gives 3,079 MiB.
- **Idle:** the server holds 319 MiB.

### Incidents, reported as they happened (PROTOCOL rule 11)

1. **Unbudgeted offload hit 124 MiB free.** The first (b) run had no
   `--max-vram` (2026-09-22 23:27). It took CUDA1 from 8,615 MiB free to
   124 MiB free before the harness's floor watcher killed it. That breached
   the 1 GB rule for well under a second.
   - The config is kept in `measure.py` only so its row reads, and is marked
     *do not run*.
2. **A second consumer on CUDA1.** An (a) run at 2026-09-23 00:38 was killed
   at 190 MiB free. It was not the image stack's own usage: its Δ is about
   6.3 GB every time, and the card reached 15,977 MiB. Another process loaded
   roughly 2.9 GB on CUDA1 mid-run, then the baseline dropped to 4,931.
   - Nothing in production has a watcher like the harness. If another
     consumer arrives on CUDA1 during a generation, the budget bounds only
     sd-server's share, and the newcomer can still find the card full.
   - AGENTS.md "one GPU consumer at a time" applies to CUDA1 too.
3. **Thermal shutdown.** The box shut down thermally at 23:42 on 2026-09-22.
   The benchmark had been running configs on the A4000 and the CPU (the
   encoder then used all 16 physical cores) since 23:21.
   - This does not show the benchmark caused it, but it was running.
   - Every later run used `-t 8` and a temperature kill-switch.
   - Rows from 23:29 until the crash were never written.

### The samples, looked at

All of these are in `bench/imagegen/samples/`.

- **Sign** (`b_offload_budget6_sign_1024_s42.png`, `server_7_*.png`,
  `b_offload_budget6_sign_1344_s42.png`):
  - Coherent brick wall, enamel sign and painted bonsai in a pot.
  - **The text renders correctly and legibly** in all three: "YAMADORI
    BONSAI", in cream serif capitals.
  - `a_te_cpu_tile16_sign_1024_s42.png` also renders it correctly, but has
    a small white square on the "B", a tiling artifact.
- **Fox** (`b_offload_budget6_fox_1024_s42.png`): a photographic red fox
  sitting in snow among birches. Anatomy is correct, it has a shallow depth
  of field and golden light. It is coherent and at card quality.
- **Crane** (`b_offload_budget6_crane_1024_s42.png`): a clean teal and
  orange flat-vector shape on white. It reads as a stylised folded paper
  bird rather than a clearly recognisable crane. It is coherent, but the
  weakest of the three.
- **Watercolor bonsai** (`server_8_*.png`, via sd-server): a juniper bonsai
  on a carved wooden stand in watercolor style. It is coherent and good.

---

## API and tool

### The model tool: `generate_image`

- **Offered** only when `YAMADORI_IMAGEGEN_URL` is set (`proxy.image_tools()`),
  and then on every tier, `minimal` included: a capability, not a gate
  (operator, 2026-09-23).
  - It is offered **even when the code-tool gate withholds the code tools**:
    "draw me a fox" carries no code domain, and that gate is about indexes.
  - Deep thinking gets it too (`proxy.deep_thinking_tools`, operator
    2026-09-23), with `describe_image` beside it so it can look at what it
    drew (see "Seeing: `describe_image`" below).
  - A client's own tool with the same name wins.
- **Description:** written as a trigger condition. It fires on draw, render,
  paint, sketch, generate, make an image, picture, photo, illustration,
  drawing, icon, logo, poster or wallpaper.
- **Arguments:** `{prompt, size?, seed?}`.
- **A successful result:**

```json
{"tool": "generate_image", "ok": true,
 "markdown": "![a red fox in snow](https://ai.thejustinwalsh.me/media/<sha>.png?exp=...&sig=...)",
 "url": "https://ai.thejustinwalsh.me/media/<sha>.png?exp=...&sig=...",
 "seed": 42, "size": "1024x1024", "steps": 20, "seconds": 107.6,
 "instruction": "Put the markdown line above in your answer exactly as given, on its own line. ..."}
```

- **Failures** use the tool envelope (`ok:false`, `error`, `reason`,
  `retryable` as a fact, `remedies` with an owner). The codes are:
  - `IMAGEGEN_NOT_CONFIGURED`
  - `IMAGEGEN_DOWN`, whose remedy says how to start it
  - `IMAGEGEN_OUT_OF_MEMORY`, which offers a smaller size
  - `IMAGEGEN_UNAVAILABLE` (llama-swap 502/503: the process died or never
    loaded)
  - `IMAGEGEN_TIMEOUT` (900 s by default; it never hangs the proxy)
  - `IMAGEGEN_BUSY`
  - `IMAGEGEN_EMPTY`
  - `BAD_ARGUMENTS`
  - `NO_PUBLIC_BASE`
- **Lane:** it takes `admission.image_lane`, one lane of its own that the
  HTTP route shares. An image never takes or waits for a chat lane. A second
  image waits up to `YAMADORI_IMAGE_WAIT` (30 s) and is then refused as
  busy.
- **Streaming:** on the streamed path the tool runs in a thread, and an empty
  delta goes out every 5 s so a two-minute generation is never silence.
- **`x_yamadori.images`** gets one entry per call: `ok`, a 16-character id
  prefix, size, seed, steps, `model` (the public name, e.g.
  `yamadori-image-turbo`), `model_source` (`account` or `default`), seconds,
  or the error code. It never holds the prompt or the URL.
- **Which model:** the caller's account preference, else the server default.
  The tool has no model argument: the model being called does not choose. This added one key to `x_yamadori`, and
  `mcp/test_tools.py` asserts it.

### `POST /v1/images/generations`

It is authenticated like every `/v1` route (`accounts.identify`).

**Body:** `{prompt, n? (1-4), size? ("WxH"), response_format? ("url" | "b64_json"), seed?, model?}`.

- **`model`:** `yamadori-image` or `yamadori-image-turbo` picks the image
  model for this one request (`catalog.IMAGE_MODELS`). Any other value, such
  as an OpenAI SDK's own default (`dall-e-3`, `gpt-image-1`), is not a choice
  and is ignored: the caller's preference applies, else the default.
- Unknown fields, such as `quality`, are ignored.

**Answer:**

```json
{"created": 1790140005,
 "data": [{"url": "https://.../media/<sha>.png?exp=...&sig=...", "revised_prompt": "..."}],
 "x_yamadori": {"images": [{"id": "5761f4056071988f", "size": "1024x1024", "seed": 7, "steps": 20,
                           "model": "yamadori-image", "model_source": "default", "seconds": 107.58}]}}
```

- **`b64_json`** is sent only when asked for. It is never inlined in a chat
  answer, because that answer is resent to the model every turn.
- **Errors** use OpenAI's envelope plus `code`, `retryable` and `remedies`:

| status | when |
|---|---|
| 400 | bad arguments |
| 401 | no key |
| 429 + Retry-After | busy |
| 503 | not configured, down or unavailable |
| 504 | timeout |
| 507 | out of memory |

**Sizes:**
- Edges must be multiples of 32, from 256 to 1536.
- The limit is at most 1344x1344 pixels, which is the measured ceiling.
- So `1536x1024` and `1024x1536`, which Hermes's own tool sends, are
  accepted, and `1536x1536` is refused.

### `GET /media/<sha>.png?exp=<unix>&sig=<hex>`

It uses a capability URL, not a Bearer header, because a browser that
renders `![](url)` sends no Authorization header.

- **Signature:** `sig = HMAC-SHA256(key, "<sha>|<exp>")`. The key is created
  once at `index/media_url.key` (index/ is gitignored). It is never logged,
  printed or put in a URL.
- **Expiry:** the default is 30 days (`YAMADORI_MEDIA_URL_TTL` seconds).
- **Id check:** the name must be exactly 64 lowercase hex characters plus
  `.png`. Anything else is a 404 before the filesystem is touched. That
  covers `../`, `%2F` and `%5C`, uppercase, `.json`, and the key file.
- **Signature check:** a tampered signature, a moved expiry, a signature for
  another sha, a missing signature or an expired link is a **403**. That
  holds whether or not the image exists, and an API key does not substitute
  for the signature.
- **A validly signed but unknown sha** is a 404.
- **Storage:** images live in `index/media/<sha>.png`, with
  `index/media/<sha>.json` beside each holding prompt, seed, size, steps,
  model, seconds, created and bytes. The same bytes are stored once.
- **Public base:** from `YAMADORI_PUBLIC_BASE`. Otherwise it is the request's
  own base, with `X-Forwarded-Proto` honoured.

### Tests

- **`mcp/test_images.py` (210 checks)** runs against a fake sd-server
  (`http.server`) and covers:
  - the request shape and stored metadata
  - every failure code, including a timeout that returns in under 2.5 s
  - the busy lane as a 429
  - route auth and response shapes
  - the signed URL working with no header
  - tampered, expired and wrong-sha signatures refused
  - `../` and non-hex ids refused
  - the tool gate on and off, and by tier
  - a full `proxy.complete` turn that calls the tool and records it
  - Hermes's own gateway regex and desktop media check, run against the
    tool's markdown
  - the two models: each sends its own llama-swap name and steps, and the
    metadata and `x_yamadori.images` record the model, steps and source
  - resolution order: request `model` > account preference > default, and
    an SDK default name ignored
  - `GET`/`PUT /dash/api/settings/image`: auth, validation, and one key unable
    to write another account (a PUT naming another account id writes the
    caller's own)
- **`web/src/screens/Settings.test.ts`** renders the picker: every option's
  steps, seconds and licence, the checked model, the default marker, and the
  way back to the default.
- **`mcp/test_tools.py`** is extended: the tool count is now 13,
  `generate_image` is in the never-empty table and has a gated-and-says-why
  test, and `x_yamadori` carries `images`.
- **`python scripts/run_tests.py`:** all suites green, ruff clean.

---

## How Hermes shows it

**The Hermes install is not on this machine now.**
- `C:\Users\jwals\AppData\Local\hermes` and `~/.hermes` do not exist.
- The install of 2026-09-21 was made from inside the Claude app sandbox, and
  Windows redirected it into `...\Packages\Claude_pzs8sxrjxfjjc\LocalCache\`,
  which is gone too.
- Only the PATH entries remain.

The lines below are therefore from upstream `NousResearch/hermes-agent` at
`1a90fad`. That is the `main` commit the 2026-09-21 installer cloned
(`hermes-install.ps1` lines 386–387 clone `main`). **Reinstall Hermes before
the acceptance test.**

| front end | what `![alt](https://…/media/<sha>.png?exp=…&sig=…)` becomes | source |
|---|---|---|
| **Desktop app (Electron)** | **An inline image.** `MarkdownImage` renders any `src` that passes `isInlineMediaSrc`, which is `/^(?:https?\|data):/i`. The media kind is read from the extension after cutting `?#` (`.png` means image). No CSP. | `apps/desktop/src/components/assistant-ui/markdown-text.tsx:365-387`, `apps/desktop/src/lib/media.ts:33-34, 75-77`; the system prompt tells the model "Remote image URLs render via ![alt](url)" (`agent/prompt_builder.py:702-709`) |
| **Messaging gateway (Telegram, Discord, Slack, Signal)** | `extract_images()` matches `!\[([^\]]*)\]\((https?://[^\s\)]+)\)` and keeps URLs containing `.png`. A substring test, so the query string does not break it. The image is sent as a photo attachment. **But** `send_image` first runs `is_safe_url`, which blocks 10.0.0.0/8, and `ai.thejustinwalsh.me` resolves to 10.242.120.152, so the gateway posts the URL as text unless `security.allow_private_urls: true` or `HERMES_ALLOW_PRIVATE_URLS=1` is set. | `gateway/platforms/base.py:2914-2924`, `plugins/platforms/telegram/adapter.py:5326-5333`, `tools/url_safety.py:134, 144-157` |
| **TUI (`hermes --tui`) and web dashboard chat (xterm running the TUI)** | **No inline image.** It renders as the text `[image: alt] url`. The link opens in a browser, and the signed URL works there without a key. | `ui-tui/src/components/markdown.tsx:183, 555` |
| **Classic CLI** | Plain text; no image. | `agent/prompt_builder.py:687-694` ("Markdown does NOT render") |

**So "the image shows inline in Hermes" is achievable in the desktop app** (and
in the messaging gateway once private URLs are allowed). It is not
achievable in the TUI or CLI, which cannot display images at all.

- **Tool output:** its markdown line is exactly what the desktop app and the
  gateway parse, and `mcp/test_images.py` runs both of Hermes's checks
  against it.
- **Signed URL:** it needs no Authorization header, and none of these
  front ends would send one.

### Hermes's own image tool (the second path, needs Hermes config)

- **The tool:** Hermes has `image_generate`, and its `openai` plugin targets
  any OpenAI-compatible `/v1/images/generations`
  (`plugins/image_gen/openai/__init__.py:1-6, 65-76`).
- **Endpoint:** `image_gen.openai.base_url`, or
  `image_gen.openai.provider: <name of the providers: entry that already
  points at Yamadori>`, which reuses that entry's URL and key. The last
  fallback is `OPENAI_BASE_URL`.
- **Model:** `image_gen.provider: openai`. The model id is passed through
  unchanged, and our route ignores it.
- **Sizes:** it sends `1024x1024`, `1536x1024` or `1024x1536`
  (`plugins/image_gen/_common.py:18`), all of which are accepted here.
- **No `response_format`:** it sends none (`openai/__init__.py:179-181`), so
  it gets our default `url`. It tries to download that URL into
  `$HERMES_HOME/cache/images/`. The same private-IP check refuses the
  download, and it falls back to handing the model the bare URL.
- **Recommendation:** the model-tool path needs none of this and is the
  primary one.

---

## Two models: choosing, and the default

**Resolution, per request** (`images.resolve_model`):

1. On `POST /v1/images/generations`, a `model` of `yamadori-image` or
   `yamadori-image-turbo` wins for that request.
2. Else the caller's saved choice. It is stored per account in
   `index/accounts/<account id>/prefs.json` as `{"image_model": "turbo"}`
   (`accounts.prefs` / `accounts.set_pref`). The account is the one the API
   key resolves to (`accounts.identify`); in single-user mode it is
   `anonymous`.
3. Else `YAMADORI_IMAGEGEN_DEFAULT` (`base` or `turbo`). Unset or unknown
   means `base`.

**How a person chooses:**

- **Dashboard:** the SETTINGS screen (`/settings`) lists both models with
  steps, seconds with their n, and licence, and saves on selection. "USE
  SERVER DEFAULT" clears the choice.
- **API:** `GET /dash/api/settings/image` answers
  `{ok, choice, default, effective, options:[{id, name, label, steps, est_seconds, est_basis, licence}]}`.
  `choice` is `null` when the account never chose. `PUT` with
  `{"choice": "turbo"}` (or `"base"`, or `null` to clear) saves it and
  answers the same shape; anything else is a 400 with the reason.
- **Whose account:** only the caller's. The routes take no account id; an
  `account` field in the body is not read. `accounts.py` has no admin role,
  so there is no cross-account write.

**Changing the default** (the operator, once `compare_turbo.py` has run):

- Set `YAMADORI_IMAGEGEN_DEFAULT=turbo` in the proxy's environment, in both
  places the proxy is started: `scripts/start-stack.bat` and the proxy's
  `Env` table in `scripts/watchdog.ps1` (see "Go live", step 2).
- Restart the proxy. Accounts that saved a choice keep it; everyone else
  moves.

**The comparison that decides it:** `bench/imagegen/compare_turbo.py`,
ready and **not run** (it waits for the benchmarks). It re-runs the 8
PartiPrompts of `parti-20260923` on both arms with sd-cli, at the base run's
own seeds (recovered from `index/media/<sha>.json`; the Images API had chosen
them), records wall, encode, sampling, decode, peak VRAM and temperature per
image, and writes base|turbo side-by-sides and a contact sheet. It refuses to
start while llama-swap has an image model loaded, and kills a run below
1 GiB free or above 88 C. `--list` prints the plan.

**Smoke test** (2026-09-23 10:01, n=1, not a result):

| what | value |
|---|---|
| command | sd-cli, the `imagegen-turbo` flags, A4000 pinned by UUID |
| prompt, seed, size | PartiPrompts #0 "a panda bear with aviator glasses on its head", 42, 1024x1024 |
| wall | 23.6 s (load 3.7, encode 3.1, sampling 12.1 = 3.0 s/step, decode 7.1) |
| VRAM | peak +5,527 MiB over a 7,567 MiB baseline; min free 3,073 MiB |
| temperature | 51 C |
| schedule | the log shows `Flux scheduler: image_seq_len=4096, steps=4, mu=0.694`: the fix applied |
| decode | the untiled VAE attempt failed and sd.cpp retried tiled, as it does for base |
| image | `bench/imagegen/samples/turbo-smoke-20260923/turbo_00_s42.png` |

Looked at: a coherent, photographic giant panda with metal-rimmed aviator
goggles, sharp fur and a forest background; no artifacts. The goggles are
worn over the eyes, not on the head as the prompt says. The base image for
this prompt (`parti-20260923/00.png`) has them on the head, but at a
different seed, so this is one anecdote and not a comparison. The white fur
reads beige.

## Go live

The operator applies these steps. Nothing below has been applied, and
nothing was restarted.

### 1. config.yaml: the `imagegen` model

Add it under `models:` beside `bonsai-vision`, pinned by UUID the way every
model in the file now is. Inside the process the A4000 is the only visible
device, so it is `cuda0`.

```yaml
  # =====================================================================
  # IMAGE GENERATION -- Qwen-Image-2.1 (Q5_K_M) via stable-diffusion.cpp,
  # on the A4000 BESIDE retrieval and Laya (no eviction). docs/IMAGEGEN.md.
  #
  # --offload-to-cpu --max-vram 6: weights live in RAM and are staged in;
  # sd.cpp keeps its managed VRAM under 6 GiB. Measured peak +6,389 MiB at
  # 1344x1344 (+6,281-6,371 at 1024x1024, n=13), idle +319 MiB. Worst-case
  # resident 7,565 + 6,389 leaves 2,213 MiB free. NEVER drop --max-vram:
  # unbudgeted offload took CUDA1 to 124 MiB free.
  # ~107 s per 1024x1024 image, 20 steps; ~220 s at 1344x1344.
  # No /health on sd-server, hence checkEndpoint. Its default port is 1234,
  # the proxy's -- --listen-port is not optional.
  # =====================================================================
  "imagegen":
    name: "Qwen-Image-2.1 Q5_K_M (sd.cpp, A4000)"
    env:
      - "CUDA_VISIBLE_DEVICES=GPU-43e37d0c-4104-9056-2552-6109d4d3382c"
    cmd: |
      C:/Users/jwals/stable-diffusion.cpp/build/bin/sd-server.exe
      --listen-ip 127.0.0.1
      --listen-port ${PORT}
      --diffusion-model ${models}/qwen-image-2.1/qwen-image-2.1-Q5_K_M.gguf
      --vae ${models}/qwen-image-2.1/vae/qwen_image_2.1_vae_bf16.safetensors
      --llm ${models}/qwen-image-2.1/qwen3vl_8b_heretic-Q4_K_M.gguf
      --backend cuda0
      --offload-to-cpu
      --max-vram 6
      --diffusion-fa
      --cfg-scale 6.0
      --sampling-method euler
      --steps 20
      -t 8
    checkEndpoint: /v1/models
    ttl: 600
```

Add a group under `routing.router.settings.groups`. Do **not** add it to
`ondemand`, which is `exclusive: true` and would evict retrieval:

```yaml
        # IMAGE GENERATION COEXISTS with retrieval and Laya: its measured
        # peak (+6,389 MiB under --max-vram 6) fits beside the worst-case
        # resident 7,565 MiB with 2,213 MiB free. Not exclusive, so loading
        # it evicts nothing. Not persistent, so bonsai-vision (exclusive)
        # still reclaims the card from it.
        "imagegen":
          swap: true
          exclusive: false
          members:
            - "imagegen"
```

**The turbo model** (`imagegen-turbo`) is in `config.yaml` beside
`imagegen`, with the same pinning, encoder, VAE and budget, plus
`--steps 4 --cfg-scale 1.0 --sampling-method euler` and
`--extra-sample-args base_shift=0.5,max_shift=0.69355`. Both are members of
the one `imagegen` group:

```yaml
        "imagegen":
          swap: true
          exclusive: false
          members:
            - "imagegen"
            - "imagegen-turbo"
```

`swap: true` runs one member at a time, so asking for one image model stops
the other and nothing else. `exclusive: false` keeps retrieval resident.
**llama-swap has not been restarted with it.**

Do not add it to the `preload` hook. It loads on the first request, and
readiness takes 0.6 s because the weights load lazily, about 2 s on the
first image.

### 2. The proxy's environment

The proxy needs three variables:

| variable | value | why |
|---|---|---|
| `YAMADORI_IMAGEGEN_URL` | `http://127.0.0.1:11434` | llama-swap; it starts `imagegen` on demand |
| `YAMADORI_PUBLIC_BASE` | `https://ai.thejustinwalsh.me` | the base for signed `/media` links. Behind Caddy the request says `127.0.0.1:1234`. |
| `YAMADORI_IMAGEGEN_MODEL` | (optional) `imagegen` | the default; set it only if the llama-swap name differs |
| `YAMADORI_IMAGEGEN_TURBO_MODEL` | (optional) `imagegen-turbo` | the same, for turbo |
| `YAMADORI_IMAGEGEN_DEFAULT` | `turbo` in the launch scripts (code default `base`) | the image model for callers who never chose. Set to `turbo` by the operator on 2026-09-23 for speed (23.6 s vs ~108 s); the quality comparison has not run yet |

`YAMADORI_IMAGEGEN_TIMEOUT` (900), `YAMADORI_MEDIA_URL_TTL` (2592000) and
`YAMADORI_IMAGE_WAIT` (30) have defaults.

**`scripts/start-stack.bat`.** Next to the two existing `set` lines before
the proxy launch, which are currently:

```bat
set "LLAMA_STACK_URL=http://127.0.0.1:11434"
set "YAMADORI_PROXY_PORT=1234"
```

add:

```bat
set "YAMADORI_IMAGEGEN_URL=http://127.0.0.1:11434"
set "YAMADORI_PUBLIC_BASE=https://ai.thejustinwalsh.me"
```

**`scripts/watchdog.ps1`.** It restarts the proxy with `Start-Process`, which
inherits the watchdog's own environment. The `set` lines above never reach
it. In the `$Services` proxy entry, add an `Env` table:

```powershell
    @{ Name = 'proxy';      Url = 'http://127.0.0.1:1234/health';  Kind = 'process'
       Exe = $py; Args = @("$root\mcp\server.py"); Log = "$root\logs\proxy.log"
       Match = 'mcp[\\/]server\.py'
       Env = @{ YAMADORI_IMAGEGEN_URL = 'http://127.0.0.1:11434'
                YAMADORI_PUBLIC_BASE  = 'https://ai.thejustinwalsh.me' } }
```

In `Restart-Service`, add this immediately before the `Start-Process` line
(at about line 271):

```powershell
    if ($svc.Env) {
        foreach ($k in $svc.Env.Keys) {
            [Environment]::SetEnvironmentVariable($k, $svc.Env[$k], 'Process')
        }
    }
```

### 3. Restart and prove it (PROTOCOL rule 1)

1. Restart llama-swap so it reads the new config, then the proxy.
2. With CUDA1 near its baseline, check the imagegen model directly through
   llama-swap (loopback, no key):

   ```bash
   curl -s http://127.0.0.1:11434/sdapi/v1/txt2img -H "Content-Type: application/json" \
     -d '{"model":"imagegen","prompt":"a red fox in snow","width":1024,"height":1024,"steps":20,"seed":1}' \
     | python -c "import sys,json;print(len(json.load(sys.stdin)['images']))"
   ```

   This should print `1` after about 2 minutes. Watch `nvidia-smi` during the
   run: CUDA1 free should stay above 2 GB.
3. Through the proxy, with a key:

   ```bash
   curl -s https://ai.thejustinwalsh.me/v1/images/generations -H "Authorization: Bearer $KEY" \
     -H "Content-Type: application/json" -d '{"prompt":"a red fox in snow"}'
   ```

   Open the returned `url` in a browser with no key. It should show the PNG.
4. **Hermes acceptance.**
   - Reinstall Hermes; it is gone from this machine.
   - Use the **desktop app**, pointed at the Yamadori provider.
   - Send: "generate an image of a lighthouse at dusk".
   - Expected:
     - The model calls `generate_image`, and the stream sends keep-alive
       deltas for about 2 minutes.
     - The answer contains the `![...](https://ai.thejustinwalsh.me/media/...)`
       line, and the desktop app renders it inline.
     - `x_yamadori.images` records the call.
   - In the TUI the same answer shows as `[image: …] url`, which is Hermes's
     renderer and not a failure. Gateway platforms also need
     `security.allow_private_urls: true`.

### Running sd-server by hand

This is what the `IMAGEGEN_DOWN` remedy points at, for use without
llama-swap:

```powershell
$env:CUDA_VISIBLE_DEVICES = "GPU-43e37d0c-4104-9056-2552-6109d4d3382c"
C:\Users\jwals\sdcpp-pr2043\build\bin\sd-server.exe --listen-ip 127.0.0.1 --listen-port 1240 `
  --diffusion-model C:/Users/jwals/textgen/user_data/models/qwen-image-2.1/qwen-image-2.1-Q5_K_M.gguf `
  --vae C:/Users/jwals/textgen/user_data/models/qwen-image-2.1/vae/qwen_image_2.1_vae_bf16.safetensors --vae-dtype bf16 `
  --llm C:/Users/jwals/textgen/user_data/models/qwen-image-2.1/qwen3vl_8b_heretic-Q4_K_M.gguf `
  --backend cuda0 --offload-to-cpu --max-vram 6 --diffusion-fa --cfg-scale 6.0 --sampling-method euler --steps 20 -t 8
```

With it running, set `YAMADORI_IMAGEGEN_URL=http://127.0.0.1:1240` for the
proxy.

## Seeing: `describe_image`

**UNTESTED LIVE.** Every claim in this section is from code and fake-server
tests (`mcp/test_vision.py`). `bonsai-vision` has never been loaded: the
llama-swap logs contain no entry for it. It stays untested until the live
check at the end of this section has run.

The chat model is text-only, so it could draw but not see what it drew. A
user's attached image was worse off. The image part went upstream to a
llama-server started without a projector, which refuses the whole request
("image input is not supported", `tools/server/server-common.cpp:1212` in
the prism build), so the turn failed. `mcp/vision.py` fixes both.

### The model tool

- **Name:** `describe_image`. It is snake_case, verb first and spelled out,
  per AGENTS.md. It is not `_opt`, because a failure is an envelope, not an
  absent answer.
- **Arguments:** `{image, question}`. `image` is either:
  - an attached image's id, `image-<10 hex>`, which the placeholder names;
  - or the `url` that `generate_image` returned. The markdown line and the
    bare sha also work.
- **Description:** written as a trigger condition. It leads with the question
  it answers ("what is IN an image"). It contrasts itself with
  `generate_image`: that tool turns words into a picture, and this one turns
  a picture into words. It lists the phrasings that should fire it: "what is
  in this image", "describe this screenshot", "read the text in this
  picture", "check the image you just drew".
- **Offered** wherever `generate_image` is: on every tier where
  `YAMADORI_IMAGEGEN_URL` is set, and to deep thinking.
  - It is also offered on every tier to a request that carries a readable
    attached image, even with no image server.
  - `YAMADORI_VISION=0` withholds it everywhere.
  - A client's own tool with the same name wins.
- **Deep thinking** (`shomen.SYSTEM`) is told to draw a mockup, look at it
  with `describe_image`, and draw it again if it is off.
- **The call:** one generation through `mcp/model.py` (`model.chat`,
  `model=bonsai-vision`), shaped by `tiers.apply` with vendor sampling at
  effort `low`. The image goes as an OpenAI `image_url` part holding a
  `data:` URI built in `vision.py`. Its MIME type is sniffed from the bytes,
  not taken from the client's claim.
- **Token budget:** the one change to the door is `cap`, which passes
  `tiers.budget`'s existing one-request thinking override. The vision server
  runs `-c 16384`, so thinking is capped at 16,384 − 4,096 (image) − 2,048
  (answer, `A_MIN`) − the text prompt estimate, about 10,000 tokens. That is
  arithmetic, not a measurement.
  - The 4,096 is llama.cpp's cap for the Qwen-VL projector family
    (`clip.cpp:1632`). It is **assumed**, not verified, that this mmproj is
    of that family.
- **Lane:** it takes `admission.image_lane`, the one A4000 lane, so drawing
  and looking never overlap. On the streamed path it runs in a thread with
  keep-alive deltas, as `generate_image` does.
- **Timeout:** 900 s (`YAMADORI_VISION_TIMEOUT`), which covers a cold load.
  It is not measured.

### Where an image may come from (the security contract)

`vision.resolve` takes text a model wrote and matches it against ids. It
never opens it. Only three sources are read:

1. **This request's attachments**, looked up by id in the register that
   `vision.extract` built from the request's own messages.
2. **Our media store**, through `images.png_bytes(sha)`. That path admits
   only 64 hex characters and checks containment. A sha is read only when
   the conversation holds a capability for it:
   - a signed `/media` link that verifies (`images.verify`, the same check
     as the `/media` route);
   - a verified link that appeared anywhere in the conversation's messages;
   - or an image that `generate_image` made in this request.
3. **Nothing else.** The rest is refused before anything touches a file or
   the network:
   - a bare sha from another conversation;
   - a `/media` link without its signature;
   - a URL (`IMAGE_URL_NOT_ALLOWED`);
   - a path of any shape (`UNKNOWN_IMAGE`).

A client's `http(s)` image part is **not forwarded**. llama-server would
download it itself (`handle_media`, `server-common.cpp:1065`), which would
be a fetch of an arbitrary URL. The placeholder tells the model to ask the
user for the file instead.

`test_security_contract` checks all of this with `urllib.request.urlopen`
and `open` wrapped: no URL is fetched and no named path is opened. A
mutation that fetches or opens makes three of its checks fail.

### Attached images: the design

This is the simplest correct design: the conversation is the store.

1. `proxy.prepare` calls `vision.extract(messages)` right after
   `session_context`. The session key still hashes the messages as the
   client sent them, as `complete()` does.
2. Each image part (the OpenAI `image_url`, the Responses `input_image` and
   the Anthropic `image` shapes) is replaced by a text part. For a readable
   image:

   > [image-3f9a1c2b7d: an attached image (PNG, 245 KB). You cannot see it
   > directly. To look at it, call describe_image with image "image-3f9a1c2b7d"
   > and a question about it.]

3. **The id is the first 10 hex characters of the image's sha256.** Every
   turn resends the history, so the same image gets the same id each turn,
   even if the client drops earlier messages. Nothing is written to disk:
   the image lives only for the request that carries it.
4. **Bad attachments get a placeholder that says why.** Each one is also
   registered, so a call on its id returns its error:
   - not an image, or not base64: `NOT_AN_IMAGE`;
   - over 10 MB, which is llama-server's own download limit
     (`server-common.cpp:1068`; set with `YAMADORI_VISION_MAX_BYTES`):
     `IMAGE_TOO_LARGE`;
   - a link: `IMAGE_URL_NOT_ALLOWED`.
5. **Audio, video and file parts** get a placeholder saying they were not
   passed on. The text server would refuse those parts too.
6. **A text-only request is returned as the same list object,** so its
   prefix stays byte-identical.
7. **`tiers.apply` estimates the prompt from the extracted messages.** Base64
   counted as text would have floored the main model's thinking at
   `MIN_THINKING` for any turn with a picture. Measured in the test: a 600 KB
   attachment costs the budget only the length of its placeholder.
8. **The register is JSON-safe** and moves from the payload into the
   session state before anything deep-copies the payload (fan-out does,
   through JSON).

### Failures

Every failure uses the tool envelope: `retryable` as a fact, remedies with
an owner, and "nothing was looked at". `UNKNOWN_IMAGE` also names what IS
available (`available.attached_ids`, `available.generated_ids`).

| code | when | retryable |
|---|---|---|
| `BAD_ARGUMENTS` | no image, no question, question over 4,000 chars | yes |
| `VISION_OFF` | `YAMADORI_VISION=0` | no |
| `UNKNOWN_IMAGE` | not an id this conversation holds; a path; a sha from elsewhere | yes (with an id from `available`) |
| `IMAGE_URL_NOT_ALLOWED` | a link to another site | no |
| `IMAGE_LINK_INVALID` | a `/media` link whose signature is wrong or expired | no |
| `IMAGE_NOT_FOUND` | a valid link to a file no longer in `index/media` | no |
| `NOT_AN_IMAGE` | attached bytes are not PNG/JPEG/GIF/BMP/WebP, or not base64 | no |
| `IMAGE_TOO_LARGE` | over 10 MB | no |
| `VISION_BUSY` | the A4000 image lane is held (drawing or looking) | yes |
| `A4000_BUSY` | the coordinator must unload a model another request is using, or another load holds the card (mcp/gpu_room.py; `generate_image` too) | yes |
| `A4000_NO_ROOM` | vision cannot fit with 1,331 MiB free even with every unloadable A4000 model gone; nothing was unloaded (`generate_image` too) | no |
| `VISION_LOADING` | HTTP 503 "Loading model" | yes |
| `VISION_UNAVAILABLE` | llama-swap 502/503/504: failed load, crash or eviction | no |
| `VISION_NO_PROJECTOR` | "image input is not supported": launched without `--mmproj` | no |
| `VISION_OUT_OF_MEMORY` | CUDA OOM on the A4000 | no |
| `VISION_CONTEXT_FULL` | image + question exceed `-c 16384` | no |
| `VISION_REJECTED_IMAGE` | the server could not decode the image | no |
| `VISION_NOT_CONFIGURED` | llama-swap has no `bonsai-vision` (404) | no |
| `VISION_TIMEOUT` | no answer within 900 s | no |
| `VISION_DOWN` | llama-swap not answering | no |
| `VISION_BUDGET` | `finish_reason: length`, a budget event and never an answer | yes (narrower question) |
| `VISION_EMPTY` / `VISION_FAILED` | nothing written / a non-JSON or other HTTP error | no |

### `x_yamadori`

This adds two keys (`mcp/test_tools.py` asserts the key set):

- **`vision`:** one entry per call, with `ok`, `image` (the attached id or a
  16-character sha prefix), `source` (`attached` or `generated`), `format`,
  `bytes`, `seconds`, `finish`, `prompt_tokens`, `completion_tokens`,
  `answer_chars` and `ms`, or the error code.
- **`attachments`:** what the request carried: `id`, `source`, `format`,
  `bytes` and `error`.

Neither key ever holds the image, the question or the answer text.

### VRAM on the A4000: a concern, from the config's own numbers

These figures come from config.yaml and this document. None of the sums is a
measurement of vision, because vision has never run.

| figure | value | source |
|---|---|---|
| A4000 used + free | 16,167 MiB | "Decision" above |
| retrieval + Laya resident | 7,565 MiB | config.yaml, `retrieval` group comment |
| `bonsai-vision` demand | 8,265–9,449 MiB | config.yaml, estimate (weights 5.95 GiB + mmproj 0.59 + KV + compute) |
| image generation peak Δ | +6,389 MiB (1344²); +6,281–6,371 at 1024² | "Also measured" above |
| image server idle | +319 MiB | "Also measured" above |
| Laya alone | **not measured separately** | none |

What the groups do, as config.yaml describes them:

- `ondemand`, which holds vision, is `exclusive: true`. Loading vision
  evicts every non-persistent group, retrieval and imagegen alike. Laya is
  not managed by llama-swap and stays.
- `imagegen` and `retrieval` are `exclusive: false`. Loading either one
  evicts nothing.
- Vision's `ttl` was 900 s when this was written (300 s since 2026-09-23).

Two sequences follow from those settings (without the coordinator described
below).

1. **Draw → look → draw again within 15 minutes.** This is the refine loop
   deep thinking is now told to run. The second draw loads the image server
   beside the resident vision model, because imagegen evicts nothing:
   - vision plus the image peak is 8,265–9,449 + 6,389 = 14,654–15,838 MiB
     before Laya;
   - that leaves 329–1,513 MiB of the card for Laya;
   - Laya's share of the 7,565 is at least 7,565 − 5,120 = 2,445 MiB, taking
     config.yaml's "together under 5 GiB" for retrieval.

   The likely outcome is `IMAGEGEN_OUT_OF_MEMORY` or worse. Incident 1 above
   is what an overfull card looked like.
2. **A code search while vision is resident.** Embeddings and the reranker
   reload beside vision: 7,565 + 8,265–9,449 = 15,830–17,014 MiB against
   16,167. That is 337 MiB spare at best and 847 MiB short at worst.

The image lane serialises drawing and looking. It does nothing about what
stays **resident** afterwards. (Vision's `ttl` has since been cut to 300 s;
that shortens the stay, it does not stop the overlap.)

The chat model on the 5060 Ti is not at risk: the `primary` group is
`persistent: true`.

### The A4000 coordinator (2026-09-24): who decides now

The live gate proved the concern (below, "Measured 2026-09-24": 308 MiB
free). The operator's decision, verbatim: "Everything on the A4000 needs to
co-operate, swap, and be coordinated. If it fits with headroom fine, if it
doesn't drop them and load in what you need on use."

`mcp/gpu_room.py` owns it. `images.generate` (both image models) and the
vision call (through `model.post`) wrap their request in
`gpu_room.use(model)`, as do the embedding and rerank calls. Before the
request that would load the model:

1. `GET /running` on llama-swap (it never loads anything) and nvidia-smi for
   the A4000 by UUID give what is loaded and what is free.
2. The model's need comes from `gpu_room.SIZES`, whose rows carry their
   sources: `imagegen` 6,389 peak / 319 idle (measured, above); `imagegen-turbo`
   the same budget's ceiling (its own peak read once, +5,527); `bonsai-vision`
   9,449, the top of the estimate in the table above (never measured alone).
   A loaded, idle image server still needs its growth, 6,389 − 319 = 6,070,
   before each draw.
3. If free − need ≥ 1,331 MiB (1.3 GiB, `YAMADORI_A4000_HEADROOM_MIB`),
   nothing happens. Otherwise other A4000 models are unloaded through
   `POST /api/models/unload/<id>`, least recently used first, one at a time,
   re-reading nvidia-smi after each, until it fits. A model another request
   is using is never unloaded; the caller waits (up to 300 s).
4. If it cannot fit even with everything unloadable gone, nothing is
   unloaded and the call fails with `A4000_NO_ROOM` (not retryable: an
   outsider or Laya holds the card; the operator remedy says so) or
   `A4000_BUSY` (retryable). Both appear in `generate_image` and
   `describe_image` envelopes with `need_mib`, `free_mib` and `headroom_mib`.

What that does to the two sequences above, by the table's numbers:

- **Draw with search resident:** fits (≈2.2 GB free at the image peak);
  nothing leaves.
- **Look with search resident:** does not fit; the least recently used search
  model is unloaded first. llama-swap's `ondemand` group (still exclusive,
  kept as the backstop) then takes the rest of retrieval with it on load.
- **Draw after a look:** vision is unloaded before the draw, never beside it.
- **Search after a draw:** embeddings and the reranker load beside the idle
  image server (319 MiB); nothing leaves.

`x_yamadori.gpu_room` records each decision: model, action (`loaded`, `fit`,
`evicted`, `busy`, `no_room`, `uncoordinated`), need, free before and after,
what was unloaded and how much each freed. Offline, `mcp/test_gpu_room.py`
drives the real callers against a fake llama-swap and card (36 checks):
look → draw → search never dropped below 4,462 MiB free where the same
sequence with the coordinator off went 1,927 MiB past the card. **Not yet run
live**; `mcp/test_live_stack.py --only images` now ends with look → draw →
search and asserts the A4000 stays ≥ 1,331 MiB free in nvidia-smi's
one-second samples. What the live run must still measure: vision's real
footprint and Laya's share, which decide whether the 9,449 estimate
over-evicts.

### The live check (run after the benchmark, through :1234)

This is one request. It makes the model draw, then look at what it drew.

```bash
curl -s --max-time 1800 http://127.0.0.1:1234/v1/chat/completions \
  -H "Authorization: Bearer $YAMADORI_TEST_KEY" -H "Content-Type: application/json" \
  -H "X-Yamadori-Session: vision-live-check-1" \
  -d '{"model":"yamadori","reasoning_effort":"low","messages":[{"role":"user","content":"Draw a red fox sitting in snow with generate_image. Then look at the picture you made with describe_image and tell me in two sentences what it shows and whether it matches what you asked for. Include the image line."}]}' \
  > vision-live.json
python -c "import json;d=json.load(open('vision-live.json'));x=d['x_yamadori'];print(json.dumps({k:x[k] for k in ('images','vision','attachments','tools','tool_turns')},indent=1));print(d['choices'][0]['message']['content'][:600])"
```

**Verify in `x_yamadori`:**

- `images`: one entry, `ok: true`.
- `vision`: one entry, with:
  - `ok: true`, `source: "generated"`, `format: "png"`;
  - `image` equal to `images[0].id`, since both are the same 16-character
    sha prefix;
  - `finish: "stop"`.
- `vision[0].prompt_tokens` is the first measured image token cost. It
  replaces the assumed 4,096. If it is over 4,096 plus about 200, the cap
  arithmetic is wrong.
- `vision[0].completion_tokens` and `seconds` are the first measured cost of
  a look. `seconds` includes the cold load.
- `tools` lists `generate_image` then `describe_image`, both with
  `error: false`. `attachments` is `[]`.
- The answer contains the `![...](.../media/<sha>.png?...)` line and a
  description consistent with a fox.

**Watch while it runs:**

- `nvidia-smi` on the A4000: the peak when vision loads, and free memory.
- llama-swap's log: vision loading and retrieval being evicted.

Then send one code-search request within 15 minutes. That is sequence 2
above, and its A4000 peak is the number the VRAM concern needs.

**Measured 2026-09-24 (n=1, `mcp/test_live_stack.py --only images`, part of
the live gate).** This is not the draw-then-look request above. That test
draws once (`generate_image`, turbo, 4 steps, 25.2 s, `ok`), then sends a
separate request that attaches a 256x256 two-colour PNG and asks
`describe_image` about it: `ok`, `source: attached`, `finish: stop`, 5.93 s,
168 prompt tokens, 125 completion tokens, and the answer `left=red,
right=blue` was correct. nvidia-smi sampled the A4000 every second through
both: baseline 12,457 MiB at the start (what was resident was not recorded),
**peak 16,068 of 16,376 MiB, 308 MiB free**, and 10,514 MiB at the end. The
VRAM concern above is real: drawing and looking in one session left almost
nothing on the card. The refine loop (draw, look, draw again) and sequence 2
are still unmeasured.

A failure envelope is a result too. Record its code:

- `VISION_NO_PROJECTOR` or `VISION_UNAVAILABLE` points at the config;
- `VISION_OUT_OF_MEMORY` points at the VRAM section above.

## White blocks (SELF-IMPROVEMENT-LOG #25): cause, measurement, fix

**Symptom.** Solid (255, 255, 255) blocks, 8-px aligned, 8-24 px across, at
recurring positions (x ≈ 488-520, y ≈ 256-280 and 512-536 at 1024²). They
appeared in 10 of the 55 images in `index/media` on 2026-09-24, on both models.
The earlier "no artifact seen" in the Decision above was wrong.

**Cause.** sd.cpp `c92d73c` creates every Wan-family VAE conv weight as F16,
whatever the file holds (`src/model/vae/wan_vae.hpp:36` for `CausalConv3d`,
`src/model/common/ggml_block.hpp` `Conv2d` for the 2-D path this VAE takes;
its `ggml_conv_2d` also forces an F16 im2col). The Qwen-Image-2.1 decoder's
activations exceed the F16 range. The values become inf, then NaN, then 255.
The upstream diagnosis is sd.cpp PR #2043 (see `IMAGEGEN-COMMUNITY.md` §2).
Tiling decides only **where** the blocks land. Under `--max-vram 6` the
untiled decode fails and sd.cpp retries with half-image tiles (32 latent =
512 px, stride 256 px), whose seams are at 256/512/768 px. The earlier
`--vae-tile-size 16x16` run put its block on a 128-px seam.

**Metric.** `bench/imagegen/white_blocks.py`. It counts 8×8 tiles that are
≥ 95% exactly (255, 255, 255), in clusters of ≤ 16 tiles, whose 1-px ring is
not itself clipped white background. Exact 255 is what separates the
artifact: painted whites (snow, paper, clouds) sit at 248-254. Validation:
- `--selftest` passes on synthetic cases: a planted block counts 6; a clipped
  white background, near-white paper, and a block on near-white paper each
  come out right.
- Production images: 0 on every white-background and snow image; every count
  is on a seam.
- Known limit: a synthetic unaligned clipped disc counts 11. None occurs in
  real images. Compare variants on the same seeds.
- One rule was added after a false positive: an untiled decode of a
  white-background control clipped one edge tile inside clipped background.

**Harness.** `bench/imagegen/vae_variants.py` runs its own sd-server with
config.yaml's exact command plus one change. It samples the A4000 by UUID
every 0.5 s, kills the run below 1,331 MiB free, and appends to
`bench/imagegen/vae_variants.jsonl`. `bench/imagegen/diff_variants.py`
diffs pixels outside the blocks.

**Repro set.** Our own prompts only: a 28-prompt benign screen generated for
this test (the operator's own images and prompts are private and are not
test material). The failing cases in that screen (turbo: a fisherman, an
astronaut and a flowchart, 9 blocks; base: 1 block) plus controls (white-
background flowcharts, an apple on white, a fox in snow, a white teacup, a
marble statue).

| variant (only this changed) | blocks | wall, median | A4000 peak Δ | pixels vs today, outside blocks | verdict |
|---|---|---:|---:|---|---|
| today: `--max-vram 6`, fallback tiles | turbo 9 in 3/8; base 1 in 1/3; 1344² 2 in 1/3 | turbo 18.0 s, base 105.5 s, 1344² 30.7 s | +5,300-5,694 | — | defect |
| `--max-vram 8` | turbo 9 in 3/8 (identical) | 18.0 s | +5,300 | bit-identical | still tiles (needs 8,772 MB vs 8,260 budget) |
| `--max-vram 9` | 1024²: 0 (turbo 8, base 3); 1344²: **2** (still tiled there) | turbo 14.2 s, base 101.8 s | **+8,262-8,694** (+3 GB) | PSNR ≥ 46 dB, max 46 | untiled only at 1024²; forces search off the card |
| `--max-vram 10`, `12` | not run | | | | skipped: free − need < 1,331 MiB headroom |
| 48×48 tiles, 6 GiB | 0 in 30 turbo + 3 base + 6 sizes | turbo 17.6 s, base 104.6 s | +5,300-6,054 | PSNR ≥ 46 dB, max 50, ≤ 0.24% of pixels off by > 8 | moves the seams; F16 still overflows |
| 56×56 / 64×64 tiles, 6 GiB | — | — | — | — | HTTP 500: 56×56 needs 6,949 MB of a 6,437 budget |
| `--backend all=cuda0,vae=cpu` | 0 in 8 | **83.2 s** (decode 72 s) | +5,300 | — | too slow; F16 weights still |
| **PR #2043 build + `--vae-dtype bf16`, 6 GiB, default tiles** | **0 in 39** (30 turbo, 3 base, 6 at 1344²/1536×1024) | turbo **19.1 s** (+6%), base **106.3 s** (+1%), 1344² 32.4 s (+6%), 1536×1024 28.2 s (+6%) | +5,300-6,078 (unchanged) | **PSNR ≥ 56 dB, max 12, no pixel off by > 8** (n=39 pairs) | **shipped** |

**Visual check.** Two baseline/PR pairs, the astronaut and the flowchart, full
frame and crop. The blocks are gone and nothing else visibly changes.

**What shipped** (config.yaml, both image models; template too):
- the binary is `C:/Users/jwals/sdcpp-pr2043/build/bin/sd-server.exe`:
  sd.cpp `c92d73c` plus PR #2043's fix commit `f047986` ("vae conv uses weight
  dtype and add support for specifying vae compute precision"), applied
  cleanly;
- built with `build-pr2043.bat` (the old `build-cuda.bat`, CUDA arch 86 only,
  target sd-server);
- cuBLAS/cudart DLLs copied beside the exe as in the old `build/bin`;
- plus `--vae-dtype bf16`. ComfyUI and diffusers run this VAE in bf16.

Nothing else changed: the same flags, `--max-vram 6`, the same encoder (the
log shows `loading llm ... qwen3vl_8b_heretic-Q4_K_M.gguf`), the same
diffusion GGUFs and the same VAE file. The old binary in
`C:/Users/jwals/stable-diffusion.cpp` is untouched. `mcp/gpu_room.py`'s
SIZES need no change: the peak stayed within the 6,389 MiB row. Timings are
at n=3-30 per cell on one day, and not yet repeated on another.

**Not tried: f32.** bf16 passed first.

**Unused local patch.** `C:/Users/jwals/sdcpp-vae-f32` is a local F32-conv
patch that was built but never run. It can be deleted.

**Upgrade hazard.** sd.cpp `b167b94`+ changes the Qwen-2.1 flow-schedule
defaults: an 8,192 anchor, `max_shift` 0.9 and `shift_terminal` 0.02.
- On that build our turbo `--extra-sample-args` gives μ 0.594 and a last sigma
  of 0.02, not Viggle's 0.40.
- Moving past `c92d73c` therefore needs absolute per-size `--sigmas` for turbo
  (1024²: `1.0,0.8572,0.6668,0.4001,0`; the per-size table is in
  `IMAGEGEN-COMMUNITY.md` §4.2-4.3).
- It also changes the base model's schedule, which is a separate paired test.
- The PR build here is `c92d73c` plus one commit, so the schedule is
  unchanged.

**Real alpha kept, noise alpha flattened.** sd-server returns RGBA (the
Qwen-Image-2.1 VAE is RGBA, and the model does real transparency when asked:
"This is an RGBA image with transparency ... the background is transparent").
Images that did not ask carry near-opaque alpha noise (248-254) that a dark
chat background shows through; an image that asked has a large share of
pixels near 0. `images.drop_alpha` keeps the PNG as it is when at least 0.5%
of its pixels are under alpha 128, and otherwise drops the alpha channel,
keeping the painted RGB (the id is then the sha of the RGB bytes). The cut
sits in the gap between the two cases; it is a choice, not tuned.
Test: `mcp/test_images.py` `test_alpha_is_dropped_on_store` (noise flattened,
a transparent background kept byte for byte).

## What is not measured

- **Quality against other quants.** Only Q5_K_M was tried.
- **Speed differences between (a), (b) and (a)+(b).** They are within a few
  seconds at n=3–5 per config, and nothing claims one is faster.
- **Image editing.** It needs `--llm_vision` and the mmproj, which were not
  downloaded.
- **Concurrent chat load on CUDA1** (embeddings and Laya active during a
  generation). Only idle retrieval was present.
- **Whether a second CUDA1 consumer arriving mid-generation fails cleanly.**
  Incident 2 says the card can fill.
- **Anything about `describe_image` on real hardware.** That covers load
  time, seconds per look, image token cost, answer quality, and the A4000
  footprint of `bonsai-vision`. See "Seeing: `describe_image`", the live
  check and the VRAM concern.
