# Image control: Qwen-Image-2.1 Fun ControlNet-Union, and the road to pixel-art LoRAs

Research done on 2026-09-25, during an Octopus benchmark run. **Nothing
ran on either GPU.** No model weights were downloaded, nothing was built
or restarted, and no file other than this one was changed. What was read:

- model cards, `config.json` files and repo trees on Hugging Face;
- the ControlNet's **safetensors header** (20,280 bytes, fetched with an
  HTTP range request; no weights);
- VideoX-Fun, ComfyUI PR #16519 and the demo Space's source;
- sd.cpp's issues and PRs;
- our sd.cpp tree at `C:/Users/jwals/sdcpp-pr2043` (c92d73c plus PR #2043's
  fix commit).

The only reads on this machine were read-only: one `nvidia-smi` query and
the RAM size.

The labels are the ones `docs/IMAGEGEN-COMMUNITY.md` uses:

| label | meaning |
|---|---|
| **[official]** | the model vendor (Alibaba PAI / Qwen): the card, the README, and VideoX-Fun code |
| **[maintainer]** | an sd.cpp or ComfyUI maintainer or contributor, in their own PR or issue |
| **[community]** | anyone else |
| **[ours]** | our arithmetic, or our reading of code. **Not a measurement.** |

Every number labelled [ours] is unmeasured, and `docs/PROTOCOL.md` applies
before any of them is cited as a result.

---

## 0. Verdict

**GO WITH A PATCH.** The ControlNet can run in stable-diffusion.cpp, but
only after a local C++ patch that no upstream build has.

- **The engine cannot load it today.**
  - sd.cpp's ControlNet support is the UNet one: SD1, SD2 and SDXL
    (`src/model/diffusion/control.hpp`). Its outputs are consumed only by
    `unet.hpp:684-698`.
  - `qwen_image_2_1.hpp` has no control input.
  - No sd.cpp issue or PR adds DiT ControlNets. The one for the same
    vendor's Z-Image Fun ControlNet, which has the same VACE-style design,
    has been open since 2025-12-02 (#1033). A contributor's comment on it:
    "it looks like it's for u-net models only".
- **The patch is modest (~300-450 lines of C++, an estimate).** The branch
  is 16 copies of the base block the engine already implements, plus
  three linear layers. sd.cpp already has the same pattern in
  `wan.hpp`'s VACE blocks (`before_proj` / `after_proj` skips) and already
  VAE-encodes a control image (`image.cpp:250-256`).
- **Memory fits the A4000 alone, at any quantisation**, if it runs staged
  under today's `--max-vram 6` budget. The estimated peak is unchanged
  from today's image server (about 6.4-6.7 GB) and about 4.8 GB stays
  free beside the resident retrieval models.
  - Keeping everything resident instead fits only at Q5_K or Q4_K, with
    about 2.0 GB free.
  - Nothing has to span the 5060 Ti, and nothing can: it has 753 MiB free.
- **Cost per image (an estimate):** about 1.5x the base model's time, so
  about 160 s at 1024x1024 with the vendor's 40 steps.
- **Licence:** Qwen Research License, which means non-commercial use only.
  That is the same constraint as the two image models in service today.

**Recommended next step.**

1. **Judge the quality before writing any C++.** Use the vendor's demo
   Space (remote ZeroGPU, which does not touch our card) with our own
   synthetic test inputs, never the operator's media.
2. **After the benchmark, with operator approval:**
   - download the 7.55 GB checkpoint;
   - convert it to Q8_0 and Q5_K with `sd-cli -M convert`, which runs on
     the CPU;
   - rebase sd.cpp to master at or after `510bccf` (the LoRA fix, §5.2);
   - write the patch against reference images from VideoX-Fun.

---

## 1. Corrections to the brief

| brief said | what the repo and upstream show |
|---|---|
| "we already load the turbo LoRA" | **We do not.** `imagegen-turbo` serves Viggle's v0.1 **full transformer** GGUF (`config.yaml:514-536`; IMAGEGEN.md:45-54). The LoRA was *blocked*: our unsloth base GGUF fuses the MLP into `img_mlp.gate_up`, and sd.cpp had no split-to-fused mapping (IMAGEGEN-TURBO.md:121-147). **Upstream fixed that on 2026-09-25**: PR #2057, `510bccf`, "map Qwen Image 2.1 LoRAs to fused MLP weights" [maintainer]. Our pin, `c92d73c`, predates it. |
| "an abliterated Qwen2.5-VL text encoder" | It is **Qwen3-VL-8B-Instruct Heretic**, `qwen3vl_8b_heretic-Q4_K_M.gguf` (IMAGEGEN.md:43; `models/manifest.yaml:195-210`). Qwen2.5-VL-7B is the encoder of the *original* Qwen-Image. |
| the A4000 as a normal x16 card | `nvidia-smi` reads the A4000's link as **width 4 of a possible 16** (PCIe 4.0 x4, about 7.9 GB/s in theory) [ours, one reading on 2026-09-25]. Every "stage weights from RAM" option streams over that link (§3.4). |
| "peak ~10.4 GB of 16 GB in today's live test with vision loaded" | I found no file in the repo that records this figure. The recorded live number is 2026-09-24: **peak 16,068 of 16,376 MiB, 308 MiB free** (`docs/LIVE-COVERAGE.md:241-245`). The A4000 read **4,844 MiB used, 11,323 free** at about 21:30 today, with retrieval only and Laya retired. I use that as the baseline. |

---

## 2. The model

**`alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union`** @ `8a47020`
(created 2026-09-23, modified 2026-09-24):

| fact | value | source |
|---|---|---|
| file | `Qwen-Image-2.1-Fun-Controlnet-Union.safetensors`, 7,550,979,904 B, LFS sha256 `65d6b66d…fcd` | HF tree API [official] |
| tensors | 180, all **BF16**, `__metadata__ {"format":"pt"}` | safetensors header [ours] |
| **parameters** | **3,775,479,808** (3.78 B) | header, summed [ours] |
| layout | `control_img_in` Linear **129 → 4096**, plus **16 `control_blocks`**. Each block has `attn.to_q/k/v/to_out.0` (4096²), `norm_q/k` (128), a **split** MLP (`img_mlp.gate_layer` and `img_mlp.proj` 4096→12288, `img_mlp.out` 12288→4096), and `after_proj` (4096², with bias). `before_proj` exists in block 0 only. No per-block modulation weights: the blocks reuse the base's shared `modulation`. | header [ours]; VideoX-Fun `qwenimage21_transformer2d_control.py` [official] |
| control input | 129 channels = **control latents (64) \| keep mask (1) \| masked-image latents (64)**. For pure control, the mask and masked-image channels are zero-padded. | card [official]; `pipeline_qwenimage21_control.py:677-724` |
| injection | skips after base blocks **0, 2, 4, …, 30**, scaled by `control_context_scale` (default 1.0) | card; model code [official] |
| conditions | Canny, Depth, Grayscale, HED, Lineart, MLSD, Pose (DWPose), Scribble, plus inpainting, alone or combined with a control | card [official] |
| sampling | **40 steps, `guidance_scale` 1.0** ("CFG-distilled fast sampling") | card; Space `app.py:6-7` [official] |
| the control image and the text encoder | the control image is **never shown to the text encoder**. It is VAE-encoded into `control_context`. | pipeline docstring, `pipeline_qwenimage21_control.py:503-515` [official] |
| prefix KV cache | **disabled when control is on** (`cache_enabled = … and control_context is None`) | pipeline line 763 [official] |
| licence | `other` / `qwen-research`. The LICENSE is the Qwen Research License of 2026-09-20: "FOR NON-COMMERCIAL PURPOSES ONLY" | card and LICENSE [official] |

### 2.1 How the branch runs

This summary is [ours], from VideoX-Fun and kijai's ComfyUI port; the two
agree.

1. The control stream starts as a joint-sequence tensor. Text and prefix
   rows are zero; target-image rows are `control_img_in(control_context)`.
2. At block 0: `c = before_proj(c) + x`, where `x` is the base joint
   stream *before* base block 0.
3. Control block *k* is a full Qwen-Image-2.1 transformer block. It uses
   the same `modulation`, RoPE and block-causal segments as the base. Its
   output feeds block *k*+1, and `after_proj(c)` is the skip.
4. The skip is added, times the strength, to the output of base block
   2*k*.
5. The control chain depends only on `x` at the input and on itself, not
   on the base stream after block 0. ComfyUI therefore *interleaves* it,
   running one control block after each even base block. That holds one
   extra stream instead of all 16 skips.

### 2.2 The variants the brief mentions

| repo | what it is | works with Qwen-Image-2.1? |
|---|---|---|
| `alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union` | the original (§2) | **yes** |
| `t8star/Qwen-Image-2.1-Fun-Controlnet-Union-Comfy` | the same tensors. "Only safetensors header metadata was changed for ComfyUI; tensor names, shapes, offsets, and payload bytes are unchanged". 7,550,980,272 B, sha256 `b3f0f3a5…`, with source sha recorded in `MODEL_SHA256.json`. Qwen Research License. | **yes, identical weights** |
| `Kijai/QwenImage_experimental` `model_patches/qwen_image_2.1_fun_controlnet_union_bf16.safetensors` (7,550,977,992 B) and `…_int8_convrot.safetensors` (3,779,298,944 B) | kijai's test files for ComfyUI PR #16519. The int8 file is a **ComfyUI-specific** "convrot" quant. sd.cpp has no loader for it; its open PR #2010 is packed *int4* convrot. No licence field on the repo. | bf16: yes. int8: ComfyUI only. |
| `InstantX/Qwen-Image-ControlNet-Union` (Apache-2.0) | a ControlNet for the **original Qwen-Image** (2025): `QwenImageControlNetModel`, 5 layers, 24 heads (3072 hidden), `joint_attention_dim` 3584 (Qwen2.5-VL-7B), `patch_size` 2, `out_channels` 16 | **no.** Qwen-Image-2.1 is a single-stream 32-layer DiT: hidden 4096, context 4096 (Qwen3-VL-8B), `patch_size` 1, a 64-channel RGBA latent (`Qwen/Qwen-Image-2.1` `transformer/config.json`). Nothing lines up. |
| `Runware/Qwen-Image-ControlNet-Union` (Apache-2.0) | **byte-identical to InstantX**: same size (3,536,027,816 B), same LFS sha256 `d51dca00…` | **no**, as above |
| `alibaba-pai/Qwen-Image-2512-Fun-Controlnet-Union` (Apache-2.0) | the same vendor's union ControlNet for Qwen-Image-**2512** | **no**: a different base architecture. It is noted because it is Apache while the 2.1 one is not. |

---

## 3. Question 1: the engine

### 3.1 What sd.cpp has today (our pin and master)

| piece | where | useful for this? |
|---|---|---|
| ControlNet runner | `src/model/diffusion/control.hpp` (the UNet `ControlNetBlock`, "Reference: … comfy/cldm/cldm.py"); `model_builders.cpp:650-670`; `diffusion_engine.cpp:592-611` (`load_control_net_from_file`), `2160-2183` (`compute_sample_controls`) | **No.** Its outputs are UNet residuals, read only by `unet.hpp:684-698` (`controls`, `control_strength`). No DiT consumes `DiffusionParams::controls` (`model.hpp:50-51`). |
| Qwen-Image-2.1 DiT | `qwen_image_2_1.hpp`: the block at `:193-244` supports **both** fused and split MLP layouts (`:199-205`); the model's forward is at `:264-298`; the runner at `:301-383` | **Yes.** A control block *is* this block (split MLP) plus two Linears. |
| VACE (the same design, in Wan) | `wan.hpp:402-450` (`VaceWanAttentionBlock`: `before_proj` on block 0, `after_proj` skip); `:599-620` (detects `vace_blocks.N` from the weights, maps them to layers); `:770-810` (runs the chain, scales the skip by `vace_strength`); `WanDiffusionExtra{vace_context, vace_strength}` (`diffusion_engine.cpp:2436-2438`) | **Yes: the template.** VideoX-Fun's code calls the design "VACE-style". Upstream PR #2062 (open, 2026-09-25) shows Wan2.2 VACE-*Fun* checkpoints, from the same vendor family, running on this path. |
| control image to latent | `image.cpp:197-199` (`control_image` to a tensor), `:250-256` (`encode_first_stage`) | **Yes.** It exists for Flux-Controls and Flex.2. |
| inpaint mask and masked-image latents | `image.cpp:208-215` (latent mask, `NearestMax`), `:337-383` (masked image at mid-gray `((1-mask)*(init-0.5))+0.5`, VAE-encoded) | **Yes.** Mid-gray is the fill both VideoX-Fun and ComfyUI use. The channel order to build is `control \| keep \| masked`, and **keep = 1 - mask**. |
| Canny preprocessor | `runtime/preprocessing.hpp:296` `preprocess_canny`; the rule `target=control,canny=true` (`common.cpp:1139`, `image_preprocess.cpp:385-394`) | **Yes**: Canny with no Python dependency |
| server fields | the native async `POST /sdcpp/v1/img_gen` takes `control_image`, `control_strength`, `mask_image` and `init_image` (`examples/server/api.md:583-584, 671-678, 793-804`). **`/sdapi/v1/*` parses `init_images` and `mask` but no control field** (`routes_sdapi.cpp:199-227`). | Partly. Use the native API with job polling, or add about 15 lines to sdapi. |
| quantising a new file | `sd-cli -M convert -m in.safetensors -o out.gguf --type q8_0` (`convert.cpp:354-409`). Names are left alone unless `--convert-name`. `tensor_should_be_converted` (`model_loader.cpp:1511-1541`) keeps biases, anything containing `img_in.` (so `control_img_in` stays bf16) and any tensor whose row length is not a block multiple. | **Yes, on the CPU.** The loader also quantises at load with `--type`. |

**GitHub search, 2026-09-25**, for "controlnet qwen", "qwen image
controlnet", "qwen inpaint" and "VideoX-Fun": nothing. The only related
items:

- #1033, "Support Z-Image Control nets" (open);
- #1365, "Enhanced ControlNet Support" (open, 2026-03);
- #1406, "sdapi server ignores mask" (open; a commenter found that the
  native `/sdcpp/v1/img_gen` honours the mask).

leejet merges several Qwen-2.1 PRs a day (#2045-#2061 in two days), but
none is about control.

### 3.2 The patch (an estimate, [ours])

The branch mirrors the base, so most of the work is plumbing.

| part | where | size (estimated lines) |
|---|---|---:|
| `QwenImage21ControlBlock` (a subclass of `QwenImage21TransformerBlock`, plus `before_proj` / `after_proj`); `control_img_in`; counting `control_blocks.N` from the weights, as Wan counts `vace_blocks.N`; the interleaved forward (§2.1) with a strength scale | `qwen_image_2_1.hpp` | 120-180 |
| `QwenImage21DiffusionExtra` gains `control_context` and `control_strength`; the runner passes them as inputs | `qwen_image_2_1.hpp`, `diffusion_engine.cpp:2430` | 20-40 |
| build the 129-channel `control_context`: VAE-encode the control image (RGB, **padded to RGBA with alpha 1**, because this VAE is 4-channel), keep = 1 - mask at latent size, masked-image latent (zeros when absent), concatenated on the channel axis | `image.cpp` beside the Flux-Controls branch (`:385`) | 50-90 |
| loading: **either** (a) `--control-net` routed into the DiT's tensor map under the diffusion prefix when the version is Qwen-Image-2.1, **or** (b) no loader change, with an offline script (gguf-py) that merges the control tensors into a copy of the base GGUF | `diffusion_engine.cpp:592-611`, `model_builders.cpp` / a Python script | (a) 30-60, (b) 0 C++ plus about 80 lines of Python |
| sdapi `control_image` / `control_strength` (or use `/sdcpp/v1/img_gen`) | `routes_sdapi.cpp` | 0-20 |
| the CFG path passes the control to the uncond pass too, as VideoX-Fun does (`pipeline_qwenimage21_control.py:794, 811`) | `diffusion_engine.cpp:2513-2545` | 5-15 |

**Total: about 250-400 lines of C++, plus tests.** For scale, kijai's
ComfyUI port is +242/-17 lines of Python (PR #16519).

The risky part is **validation, not code**:

- The output must match a reference: VideoX-Fun at bf16 on a small size,
  with a fixed seed and the same sigmas.
- The traps are:
  - the order of the three input groups (`control | keep | masked`);
  - the keep mask's polarity (keep = 1 - mask, where white = regenerate);
  - `before_proj` taking the stream *before* base block 0;
  - the skip landing *after* even blocks;
  - RGBA padding of the control image.
- A silently wrong port produces plausible images that simply ignore the
  control. The card warns that "a mismatched config silently drops or
  misplaces control weights and produces wrong outputs".

**The patch needs a rebase first.** Our build is `c92d73c` plus one commit
of PR #2043. Upstream has since merged four things that matter here:

- `510bccf` / #2057, LoRA split-to-fused mapping (§5.2);
- `0a9340c` / #2054, leejet's own VAE F16-overflow fix: it scales conv
  inputs by 1/128. It supersedes #2043, **but needs our white-block
  re-test**.
- `b167b94` / #2048, the official flow schedule. That is the turbo
  **upgrade hazard**: turbo must move to absolute `--sigmas` first
  (IMAGEGEN-COMMUNITY.md §4.3).
- #2059, VAE tile sizes in pixels, which changes a flag's meaning.

The rebase is worth doing for the LoRA fix alone. It is also a change to
both models in service, so it is **its own paired A/B**
(`compare_turbo.py`, `white_blocks.py`), before the ControlNet work
starts.

### 3.3 Other engines, ranked

| rank | engine | what it costs us |
|---|---|---|
| **1** | **sd.cpp plus our patch** | The C++ patch and the validation above. It **keeps everything we have**: one exe pinned in `engines/manifest.yaml`, llama-swap managing it, `gpu_room`'s unload-by-API, `--max-vram` as a hard bound, GGUF quants, the Heretic **GGUF** encoder, the bf16-VAE fix, and the same sdapi/native API `mcp/images.py` speaks. Windows and CUDA are already solved (`build-pr2043.bat`). Upstream may never take it; carrying it is a second local patch in `engines/patches/`. |
| 2 | **ComfyUI headless**, PR #16519 (kijai, **open**, +242/-17) | The quickest path to a *correct* first image once the PR merges. It is a second runtime: Python, torch, custom nodes (`ComfyUI-GGUF` for our GGUF DiT; that it loads unsloth's fused Qwen-2.1 GGUF is **not verified**). Pinning means ComfyUI, node commits and a pip freeze, which is heavier than one exe. A new client is needed: workflow JSON through `/prompt` plus polling, about 200 lines in `images.py`. **VRAM:** ComfyUI keeps models resident and manages its own memory (`--reserve-vram`, `--lowvram`, `POST /free`). It has no hard budget like `--max-vram`, so `gpu_room` must treat it as a coarse unit that llama-swap kills to free the card. **Text encoder:** the Heretic card warns GGUF encoders do not work in ComfyUI (IMAGEGEN.md:59), **but** the same repo ships `qwen3vl_8b_fp8_heretic.safetensors` (9.3 GB) and bf16 (17.5 GB), so the abliterated encoder survives. Licence GPL-3.0; we run it and do not distribute it. |
| 3 | **diffusers / VideoX-Fun** (Apache-2.0) | The **reference** implementation, and the right tool to make golden images. Poor for serving: bf16 DiT 14 GB + ControlNet 7.55 GB + encoder 16-17 GB. It needs `model_group_offload` or qfloat8 (VideoX-Fun `predict_t2i_control.py:43, 177-192`), and group offload streams most of 21 GB per step over our **x4** link. The stack interpreter already has torch 2.9.0+cu128. No GGUF path is verified for this model in diffusers. |

**Does ControlNet change what the text encoder must do?** No. The prompt
is encoded exactly as for text-to-image, and the control image goes
through the VAE only [official, §2]. The abliterated Heretic encoder is
used the same way it is today.

The one caveat is the same one we already accepted for text-to-image: the
branch was trained on the **stock** Qwen3-VL-8B's hidden states. The
Heretic card reports KL 0.022 on the language-model outputs. How far its
hidden states drift is unmeasured.

---

## 4. Question 2: memory

### 4.1 The ControlNet branch at each quantisation

Arithmetic from the header [ours]:

- 3,774,873,600 of the 3,775,479,808 parameters are 2-D weights whose row
  length (4096 or 12288) divides 256, so they quantise.
- The remaining 606,208 (biases, norms, `control_img_in`) stay bf16.
- Bits per weight are ggml's block sizes: Q8_0 8.5, Q6_K 6.5625, Q5_K 5.5,
  Q4_K 4.5.

| type | bytes | MiB | available? |
|---|---:|---:|---|
| bf16 (as shipped) | 7,550,959,616 | **7,201** | yes (the vendor, t8star, kijai) |
| Q8_0 | 4,012,015,616 | **3,826** | **no; convert** (`sd-cli -M convert --type q8_0`, CPU) |
| Q6_K | 3,097,788,416 | 2,954 | convert |
| Q5_K | 2,596,438,016 | **2,476** | convert |
| Q4_K | 2,124,578,816 | **2,026** | convert |

- **No GGUF of this ControlNet exists** on Hugging Face as of 2026-09-25.
  The only other quant is kijai's ComfyUI-only int8 "convrot".
- **Compute:** the branch's matmul parameters are 54% of the base DiT's
  (3.76 B against 32 × 218.1 M = 6.98 B), and it adds 16 attention layers
  to the base's 32. Each forward pass therefore costs about **1.5x**
  [ours, arithmetic].

### 4.2 Does it fit the A4000?

The card holds 16,376 MiB. The headroom `gpu_room` keeps is
`HEADROOM_MIB` = 1,331 (`gpu_room.py:101`).

**Inputs:**

| input | value | source |
|---|---|---|
| DiT Q5_K_M | 5,140 MiB | the file, 5,390,223,072 B |
| image server peak under `--max-vram 6` | +6,389 MiB | measured, IMAGEGEN.md "Also measured" |
| retrieval resident now | 4,844 MiB used | nvidia-smi, 2026-09-25 |

| arm | A4000 need (peak Δ) | free at peak, beside retrieval (4,844) | fits with 1,331 headroom? |
|---|---:|---:|---|
| **staged, `--max-vram 6`, any quant** (weights in RAM, streamed in) | ≈ 6,400-6,700 (today's 6,389 plus ~150-300 for the extra control stream and skip at 1024², which is ~4,400 tokens x 4096 x 4 B = 72 MiB per tensor) | ≈ **4,830-5,130** | **yes**, with retrieval resident. Vision (9,449 estimated) must be unloaded first, which `gpu_room` already does for every draw. |
| resident, ControlNet **Q4_K**: 5,140 + 2,026 weights + ~1,100 compute/context → `--max-vram 9` | ≈ 8,300-9,300 | ≈ 2,200-3,200 | **yes** |
| resident, ControlNet **Q5_K**: 5,140 + 2,476 + ~1,100 | ≈ 8,700-9,500 | ≈ **2,000-2,800** | **yes, narrowly** |
| resident, ControlNet **Q8_0**: 5,140 + 3,826 + ~1,100 | ≈ 10,100-10,900 | ≈ 600-1,400 | **no** (at or under the headroom) |
| resident, ControlNet **bf16**: 5,140 + 7,201 + ~1,100 | ≈ 13,400 | negative | **no**: fits only with retrieval evicted (≈ 2,600 free on an empty card) |
| span onto the 5060 Ti | — | the 5060 Ti has **753 MiB free** (15,298 used; nvidia-smi 2026-09-25) | **no.** sd.cpp's multi-device split (`--backend "diffusion=cuda0&cuda1"` with `--split-mode layer`, `common.cpp:545-560`) places *whole blocks* per device. A control block at Q5_K is ~155 MiB, so even one per device breaks the main model's ≥ 600 MB peak target, and only with `bonsai` (persistent) unloaded. |

The "~1,100 compute/context" is today's 6,389 peak less the 5,140 MiB of
weights, rounded up [ours]. The two `--max-vram 9` runs in
IMAGEGEN.md's white-block table (+8,262-8,694) are the only measured
large-budget peaks, and they are **without** the ControlNet.

**Host RAM under staging:**

| part | RAM |
|---|---:|
| DiT | 5.0 GiB |
| encoder | 4.7 GiB |
| VAE | 0.6 GiB |
| ControlNet | 1.9-7.0 GiB |
| **total** | **≈ 12-17 GiB** |

The machine has 63.7 GB of RAM, with 26.2 GB free at the time of reading
(during the benchmark).

### 4.3 sd.cpp's ways to split, and what each costs

| flag | what it does | cost |
|---|---|---|
| `--offload-to-cpu` (`common.cpp:599`) = `--params-backend *=cpu` | weights live in RAM and are staged into VRAM per graph segment; `--max-vram` caps it | **measured:** about 3% of sampling at the 6 GiB budget (IMAGEGEN.md option (b)). With the ControlNet the whole branch streams each pass: Q8_0 ≈ 3.7 GiB, Q5_K ≈ 2.4 GiB. |
| `--max-vram N` / `-N` (`:567`) | per-device budget; a negative value is a reserve | this is the bound `gpu_room` relies on. Never omit it (IMAGEGEN.md incident 1). |
| `--backend te=cpu` (was `--clip-on-cpu`, `:611`) | encoder computes on the CPU | **measured:** +2.3-2.9 s per image, about 2% |
| `--backend vae=cpu` (was `--vae-on-cpu`, `:615`) | VAE computes on the CPU | **measured:** 83.2 s wall, decode 72 s: too slow (IMAGEGEN.md white-block table). A control image and an inpaint source each add an **encode**. |
| `--backend controlnet=cpu` (was `--control-net-cpu`, `:607`) | the **UNet** ControlNet runner on the CPU | **does not apply.** Our branch lives inside the DiT runner. Running 3.78 B parameters on the CPU at ~4,400 tokens is about 33 TFLOP per pass [ours], roughly a minute or more per pass on an i7-13700K, times 40 passes. |
| `--params-backend disk` (`:550`) | weights stay on disk | no reason to use it with 64 GB of RAM |
| `--split-mode layer\|row` (`:555`) | multiple GPUs per module | the 5060 Ti has no room (§4.2) |

**Streaming over the x4 link** [ours, arithmetic]:

- One Q8_0 control block is about 250 MB, which takes about 38 ms at
  ~6.5 GB/s effective.
- A base block computes in about 52 ms (≈ 2.5 s per pass ÷ 48 blocks).
- sd.cpp prefetches the next segment during compute (#1905, merged
  2026-08-22, so in our build).
- The transfer should therefore mostly hide behind compute. The PCIe
  generation read 1 at idle (power state); the width of 4 is the fixed
  part. **Unmeasured.**

### 4.4 Answer

- **One card: yes, the A4000 alone.**
- **Recommended arm: staged `--max-vram 6` with a Q8_0 ControlNet.**
  - It is the highest-precision quant that costs no VRAM.
  - Its peak is today's measured budget plus the control stream: about
    +6.4-6.7 GB, so about 4.8 GB stays free beside retrieval.
- **A Q5_K arm is second.** It streams less and is the only arm that also
  fits resident (`--max-vram 9`, about 2 GB free).
- bf16 fits only staged. Nothing spans to the 5060 Ti.

---

## 5. Question 3: the product, a second tool

### 5.1 Name

The rules are AGENTS.md "Naming": snake_case, verb first, spelled out, no
abbreviations.

| candidate | verdict |
|---|---|
| **`generate_image_from_control`** | **recommended.** Verb first. It reads beside `yama_generate_image` as a sibling, so the model sees the two as one family. "Control" is the concept the argument names. |
| `generate_image_from_structure` | clearer to a layperson for canny, depth and pose, but inpainting is not "structure" |
| `generate_image_from_image` | rejected: it would steal "edit this photo" requests, which are Qwen-Image-2.1's own edit mode and not this tool |
| `redraw_image` / `repaint_image_region` | a verb for the inpaint half only. Consider it later if inpaint needs its own trigger list. |

No `_opt`: a failure is an envelope, not an absent answer. No `_exn`.

### 5.2 Description (a trigger condition, per AGENTS.md "Tool descriptions are prompts")

It leads with the question the tool answers, contrasts itself with the
tools it could be confused with, and lists the phrasings that should fire
it. It says nothing about what the model must not do.

> Makes a new picture that keeps the SHAPE of an existing picture -- its
> outlines, depth, a person's pose, or its layout -- or repaints one region
> of a picture and keeps the rest. Use it when the user gives or points at
> an image and wants a new one built on it: "turn this sketch into a
> painting", "same pose, but a knight in armour", "keep the composition
> and make it night", "colour this line art", "trace this photo as a
> pixel-art scene", "redraw just the sky", "replace the dog with a cat",
> "fill in the masked area". yama_generate_image makes a picture from words
> alone; yama_describe_image turns a picture into words; this one takes a
> picture AND words. The image is one attached to this conversation
> (image-xxxxxxxxxx) or one generated here (its url); it cannot fetch a
> link. Describe the WHOLE picture you want in the prompt, not only the
> changed part. [+ the shown-to-user sentence, as `images.MAIN_TOOL`
> does for main]

### 5.3 Arguments

```json
{"prompt":   "string, required: the whole target picture",
 "image":    "string, required: image-<10 hex> | our signed /media url | a sha made in this request",
 "control":  "enum, required: canny | lineart | soft_edges | scribble | straight_lines | depth | pose | grayscale | as_is | none",
 "mask":     "string, optional: an image id or url, like image; WHITE = repaint",
 "region":   "object, optional: {x0,y0,x1,y1} as fractions 0-1, a box to repaint (instead of a mask)",
 "strength": "number, optional: 0-1, default 1.0 (control_context_scale)",
 "size":     "string, optional: WIDTHxHEIGHT; default = the source's aspect, snapped to 32, ~1 MP",
 "seed":     "integer, optional"}
```

- **`control` values** are spelled out, not the preprocessor names:
  - `soft_edges` is HED;
  - `straight_lines` is MLSD;
  - `as_is` means "the image already is a control map";
  - `none` means inpaint only, with zero control channels. The vendor
    trained that recipe (§2).
  - Each enum value's description lists when to use it: canny for photos,
    lineart for drawings, pose for "same pose", depth for "same space or
    layout", grayscale for "recolour" and "colourise".
- **`region` exists because a text-only model cannot draw a mask.** Without
  it, inpainting would work only when the *user* supplies a mask. The proxy
  rasterises the box into a white-on-black mask (a CPU and PIL job). A
  later version could add an ellipse, or polygon points.
- **`image` and `mask` resolve through `vision.resolve`** (`vision.py:720`),
  with the *same* security contract as `yama_describe_image` (IMAGEGEN.md:854-882):
  - this request's attachments by id, or our media store by a verified
    signed link or a sha generated in this request;
  - never a URL (`IMAGE_URL_NOT_ALLOWED`) or a path (`UNKNOWN_IMAGE`);
  - the `test_security_contract` wrap (`urlopen` and `open` patched)
    extends to this tool.

### 5.4 Preprocessing: where and how

It runs on the **CPU, in the proxy's image lane** (`admission.image_lane`,
which it shares with drawing and looking), before `gpu_room.use`. The
A4000 is never touched for it.

The code would be a new module, `mcp/control_maps.py`: bytes in, PNG bytes
out, and a record of `{type, ms, w, h}`.

| control | phase | how | new dependency or download |
|---|---|---|---|
| `canny` | 1 | sd.cpp's own `preprocess_canny`, through the request's image-preprocess rule `target=control,canny=true` (`common.cpp:1139`). That makes it a server option with no Python. The alternative is numpy/scipy (both in the stack interpreter) with the Space's thresholds 100/200 (`app.py:125-127`). | none |
| `grayscale` | 1 | PIL `convert("L")` then back to RGB (`app.py:129`) | none |
| `scribble` | 1 | Canny (or HED later), dilated and thresholded to thick binary strokes (scipy `binary_dilation`) | none |
| `as_is` | 1 | resized to the canvas only | none |
| `none` (inpaint only) | 1 | nothing | none |
| `depth` | 2 | **Depth-Anything-V2-Small**, ONNX (`onnx-community/depth-anything-v2-small`, 99 MB fp32 / 27 MB int8, **Apache-2.0**). Not Base or Large: those are **CC-BY-NC-4.0**. | `onnxruntime` (CPU, MIT), plus the download |
| `pose` | 2 | **DWPose** ONNX: `yolox_l.onnx` 216.7 MB + `dw-ll_ucoco_384.onnx` 134.4 MB (`yzd-v/DWPose`, **Apache-2.0**). The card names DWPose skeletons. | `onnxruntime`, plus the download |
| `soft_edges` (HED), `lineart`, `straight_lines` (MLSD) | 3 | the `controlnet_aux` models (Apache-2.0 code). The `lllyasviel/Annotators` repo is licence "other", so check each file. | torch (present) or ONNX exports, plus the downloads |

- **The prepared map is stored** in the media store as its own sha, with
  metadata `{"kind": "control_map", "control": ...}`. The result carries
  its signed link as `control_map_url`, so the model or user can see what
  steered the drawing. It is **not** emitted to the chat.
- **Phase 1 needs no download and no new pip package.** Phase 2 needs
  operator approval for about 450 MB of ONNX files plus `onnxruntime`.

### 5.5 Serving, the A4000 and the record

- **llama-swap.** A new entry, `imagegen-control`, in the same `imagegen`
  swap group (`config.yaml:726-731`), so one image server runs at a time.
  It is the base DiT, the ControlNet GGUF, the same encoder and VAE, and
  `--max-vram 6`.
- **Sampling.** The vendor's contract is 40 steps and cfg 1. That is 40
  passes, the same count as today's cfg 6 × 20.
  - Base schedule: see IMAGEGEN-COMMUNITY.md §1.
  - Turbo: whether the ControlNet works on Viggle's student is **unknown**.
    The branch was trained against the base's features, and the student
    is a full fine-tune.
  - PAI's own 4-step `Qwen-Image-2.1-Fun-Acc-4Step` LoRA is the natural
    fast pairing, from the same vendor. **Its key layout probably does
    not load in sd.cpp as-is:** its keys end in `.lora_down` / `.lora_up`
    with no `.weight`, and it carries full `norm_q/k` and `proj_out`
    replacements (header, `__metadata__.format =
    qwenimage21_extracted_prefused_v1`). sd.cpp's suffix map
    (`name_conversion.cpp:1470-1506`) has neither [ours, reading].
- **Why a separate entry rather than one server for both tools.** A
  single server could carry the branch and skip it when no control is
  sent. But then every `yama_generate_image` would carry the branch's RAM and
  load time, and `SIZES` could not tell the two apart.
- **API.** Either `POST /sdcpp/v1/img_gen` (it has `control_image`,
  `control_strength`, `mask_image` and `init_image`) plus
  `GET /sdcpp/v1/jobs/{id}` polling, or sdapi `img2img` once the patch
  adds control there.
- **`gpu_room.SIZES` row** (a proposal; `gpu_room.py:160-196` format):

  ```python
  "imagegen-control": Size(
      6700, 319, False,
      "ESTIMATE, not measured: imagegen's measured +6,389 MiB under "
      "--max-vram 6 plus ~150-300 MiB for the control stream and skip at "
      "1024^2 (docs/IMAGE-CONTROL.md 4.2). The ControlNet is staged from "
      "RAM inside the same budget. Idle hold assumed equal to imagegen's"),
  ```

  and `SWAP_GROUPS = (frozenset({"imagegen", "imagegen-turbo",
  "imagegen-control"}),)`. `mcp/test_gpu_room.py` checks rows against
  `config.yaml`'s pins, so the row and the config entry land together.
- **Emitting the picture.** It uses the path `yama_generate_image` uses
  (IMAGEGEN.md:325-370; `proxy._run_turn`, "IMAGES REACH THE CHAT"):
  - `images.shown_on_main` today accepts only
    `d.get("tool") == TOOL_NAME` (`images.py:803`). It becomes a set of
    image-making tool names.
  - `_ImageDedup`, the ledger's replay of hidden hops, and the streamed
    keep-alive all key on the result shape, which stays the same:
    `{tool, ok, markdown, url, seed, size, steps, seconds, instruction}`,
    plus `control_map_url`.
  - `mcp/test_image_emit.py` gains the new tool's cases.
- **`x_yamadori.images[]`.** The same entry, plus:

  ```
  tool: "generate_image_from_control"
  control: {type, strength, source: attached|generated,
            mask: none|image|region, map_id: <16 hex>, preprocess_ms}
  ```

  It holds no prompt and no URL, the same rule as today. `mcp/test_tools.py`
  asserts the `x_yamadori` key set, and nothing new is added at the top
  level.
- **Where it is offered.** It follows `yama_generate_image` for cache
  stability: a static tool list on every tier where `YAMADORI_IMAGEGEN_URL`
  is set **and** the new `YAMADORI_IMAGE_CONTROL=1`. It is offered to deep
  thinking too (`proxy.deep_thinking_tools`), so it can refine its own
  mockups. It counts toward `proxy.OUR_NAMES` (18 → 19).
- **Failures** use the envelope and carry the next step:
  - the `vision.resolve` codes (`UNKNOWN_IMAGE` with `available`,
    `IMAGE_URL_NOT_ALLOWED`, `IMAGE_LINK_INVALID`, `NOT_AN_IMAGE`,
    `IMAGE_TOO_LARGE`);
  - `CONTROL_TYPE_UNAVAILABLE` (depth or pose not installed): it names
    the controls that are available, and the operator remedy is to
    install;
  - `CONTROL_PREPROCESS_FAILED`;
  - `MASK_EMPTY` (a mask or region that repaints nothing, or everything
    when `control` is `none`);
  - `BAD_ARGUMENTS`;
  - every `IMAGEGEN_*` and `A4000_*` code from today's tool.

---

## 6. Question 4: the pixel-art LoRA journey

### 6.1 What exists today (searched 2026-09-25)

**For Qwen-Image-2.1: none.**

- Hugging Face has 39 adapters with `base_model:adapter:Qwen/Qwen-Image-2.1`.
  None is pixel art.
- A search for "qwen-image-2.1 pixel" returns nothing.
- Civitai has a `Qwen 2.1` base-model tag. Searching it for pixel, sprite
  and 8-bit returns nothing.
- The model is five days old.

**For older Qwen-Image releases** (**not compatible**: those are the 60-layer
MMDiT with a 16-channel VAE; §2.2):

| LoRA | base | size | trigger | licence |
|---|---|---:|---|---|
| `prithivMLmods/Qwen-Image-2512-Pixel-Art-LoRA` @ `e14eea5` | Qwen-Image-2512 | 1,179,885,016 B | `Pixel Art` | Apache-2.0 |
| `laoK888/Qwen-Image-2512-Pixel-Art-LoRA` | — | **byte-identical re-upload** of the above (sha256 `2a2775ac…`), plus `Qwen4Play-2512.1_e10.safetensors` 590 MB | `Pixel Art` | Apache-2.0 (as declared) |
| `artificialguybr/PIXELART-REDMOND-QWENIMAGE` @ `d887696` | Qwen-Image-2512 | 590,058,920 B | `Pixel Art, PixArFK` | Apache-2.0 |
| Civitai 144684 "PixelArtRedmond", Qwen v1.0 | Qwen | 563 MB | `Pixel Art`, `PixArFK` | Civitai terms (commercial: image, rent) |
| Civitai 1770073 "Pixel Art Style Lora", Qwen | Qwen | 563 MB | `Pixel art style.` | Civitai terms |
| Civitai 10706 "Z-Image and Qwen Pixel Art Refiner" | Qwen | 281 MB | — | Civitai terms |

**What they are good for:** as evidence that the style is learnable at
rank 32-64 (590 MB ≈ rank 64 on the old 20 B MMDiT), and for their
captioning conventions ("Pixel Art, a pixelated image of …, rendered in
16-bit pixel art"). They cannot be loaded on 2.1, and they are not a
dataset.

### 6.2 Can sd.cpp load Qwen-Image-2.1 LoRAs?

**Yes, on master; partly on our pin.**

- **The mechanism.** A server-side LoRA directory (`--lora-model-dir`).
  `sd-server` refuses prompt tags ("Intentionally disable prompt-embedded
  LoRA tag parsing", `routes_sdapi.cpp:253`) and takes
  `"lora": [{"path": ..., "multiplier": ...}]` per request
  (`routes_sdapi.cpp:153-183`). `GET /sdapi/v1/loras` lists the directory.
- **Key formats.** Diffusers/PEFT (`lora_A`/`lora_B`, `transformer.`
  prefix) and kohya-style `lora_down`/`lora_up` are translated
  (`name_conversion.cpp:1470-1530`).
- **The fused-MLP problem, and the fix.** On `c92d73c`, every LoRA that
  targets the split MLP names (every diffusers or ai-toolkit Qwen-2.1
  LoRA) applies only partly on our fused unsloth base. Issue #2051 counted
  326 of 454 tensors applied. **Fixed by #2057 (`510bccf`, 2026-09-25):**
  `gate_layer` maps to `gate_up.weight` and `proj` to `gate_up.weight.1`.
  A rebase (§3.2) brings it in.
- **Apply mode.** `at_runtime` is chosen automatically on a quantised base
  (`common.cpp:764-770`; IMAGEGEN-TURBO.md:149-160). That is the unmerged
  form, which is what LoRA authors intend.

**LoRA plus ControlNet together** [ours, reading]:

- A LoRA is matched by tensor name, `model.diffusion_model.transformer_blocks.*`.
  The control blocks would be named `control_blocks.*`, so a base-model
  LoRA leaves the branch alone. That is what VideoX-Fun and ComfyUI do.
- The vendor's own control script **composes the two**: it merges a LoRA
  (`lora_weight` 0.55) into the pipeline and then runs the control
  (`predict_t2i_control.py:65, 85, 197-198`).
- So they compose in design. In sd.cpp this is **unmeasured** until the
  patch exists.

### 6.3 Training our own

**Tools that support Qwen-Image-2.1 LoRA training** (checked 2026-09-25):

| tool | licence | 2.1 support | notes |
|---|---|---|---|
| **ostris/ai-toolkit** | MIT | since `c2622ed` (2026-09-20). **Alpha-channel training** since `086b663` (2026-09-20), `match_target_res` since `07abdbe` | Windows UI; supports quantising the base (qfloat8, uint4), low_vram and text-embedding caching. **Recommended first**, because it can train transparent sprites. |
| modelscope/DiffSynth-Studio | Apache-2.0 | `examples/qwen_image_21/model_training/lora/Qwen-Image-2.1.sh`; RGBA datasets by default | `--fp8_models` for frozen parts; gradient-checkpoint offload; "minimum of 7GB VRAM" for *inference* with disk offload |
| AcademiaSD Qwen-Image 2.1 LoRAlab | licence unknown | `AcademiaSD/Qwen-Image-2.1-NF4-for-LoRA-Training` (Qwen Research License) | its card claims consumer GPUs from 8 GB: the NF4 DiT takes **~3.9 GB** of VRAM, with velocity cosine 0.998-0.9996 against bf16, and text embeddings pre-cached with an exact bf16 encoder |
| bghira/SimpleTuner | **AGPL-3.0** | `simpletuner/helpers/models/qwen_image/transformer_21.py` | the licence matters only if we redistribute |
| aigc-apps/VideoX-Fun | Apache-2.0 | `scripts/qwenimage21/train.py` (full fine-tune, DeepSpeed/FSDP) | multi-GPU oriented |
| kohya-ss/musubi-tuner, OneTrainer | — | **no 2.1 support found** (code search, 2026-09-25) | — |

**VRAM on the 16 GB A4000** [ours, arithmetic, unmeasured]:

| part | size |
|---|---|
| DiT, bf16 | 13.3 GiB (6.98 B × 2): too much |
| DiT, qfloat8 | 6.7 GiB |
| DiT, NF4 | 3.9 GiB (the card's measurement) |
| rank-32 LoRA on every block linear | ≈ 84 M parameters. Weights, grads and AdamW states ≈ 1.3 GiB. |
| activations with gradient checkpointing | 2-4 GiB at 1024² (4,096 image tokens); 4x fewer tokens at 512² |
| text encoder and VAE | unloaded after caching (encode once) |

That gives:

- qfloat8 ≈ **10-12 GiB**, which needs **retrieval evicted**: a maintenance
  window, and `gpu_room` or llama-swap unloading embeddings and the
  reranker;
- NF4 at 512-768² ≈ **6-8 GiB**, which fits beside retrieval (11.3 GB
  free now).

Never train while the image server or vision is loaded: that is "one GPU
consumer at a time".

**Time** [ours, arithmetic]:

- Forward is about 2.5 s per pass at 1024² (measured sampling).
- A training step with checkpointing is about 4x that: ~10 s at 1024²,
  or ~3 s at 512².
- A style LoRA usually needs 1,500-3,000 steps, so **~1.5-3 h at 512²** or
  **~5-8 h at 1024²**.
- NF4's dequantisation overhead is not included.

**Dataset.**

- Style LoRAs in the community use 20-100 curated images. The rights
  matter as much as the count.
- **CC0 pixel-art sources exist:** kenney.nl asset packs, and the CC0
  subset of OpenGameArt.
- LPC (Liberated Pixel Cup) art is CC-BY-SA / GPL, which carries
  attribution and share-alike duties.
- The operator's own generated images are **not** training material
  (memory: private media).

### 6.4 Pixel-art specifics: what the community does

**Training data:**

- **One fixed pixel scale per LoRA.**
  - Store sprites at their native size, then upscale them to the training
    resolution by an **integer nearest-neighbour** factor. For example, a
    64 px sprite ×16 gives 1024; ×8 gives 512.
  - The model then learns crisp, square "art pixels" on a known grid, and
    post-processing knows the cell size without having to guess it.
- **Captions.** Lead with the trigger. Name the palette size and the era:
  "16-colour", "8-bit NES palette", "GameBoy 4-shade". Name the view:
  "side view sprite", "isometric tile".
- **Transparency.** Qwen-Image-2.1 is **RGBA end to end**: its VAE decodes
  alpha, and our store keeps real alpha (`images.drop_alpha`,
  IMAGEGEN.md:1217-1227). ai-toolkit and DiffSynth train on alpha.
  - So a sprite LoRA can learn transparent backgrounds directly.
  - The official prompt template for transparency applies:
    IMAGEGEN-COMMUNITY.md §1, "transparency".

**Post-processing** (all CPU; PIL, numpy and scipy are already in the stack
interpreter):

1. **Find the grid.**
   - With a fixed training scale, the cell size is known (for example
     8 px at 1024).
   - Otherwise, detect it:
     - `Astropulse/pixeldetector` (MIT);
     - `KennethJAllen/proper-pixel-art` (MIT): Canny, morphological
       close, probabilistic Hough for near-axis lines, cluster the lines,
       median spacing;
     - `jenissimo/unfake.js` / `painebenjamin/unfake.py` (MIT):
       edge-aware scale detection.
2. **Downscale per cell.** Take the **most common colour in each cell**
   (proper-pixel-art), or the cell centre. Never bilinear.
3. **Quantise the palette.**
   - Either k-means, median-cut or Wu (unfake.py) to N colours;
   - or map to a **fixed palette** (PICO-8 16, NES, GameBoy 4, DB32,
     ENDESGA-32), nearest in a perceptual space such as OKLab.
   - No dithering unless asked.
4. **Binarise alpha.** Threshold at 128, which makes sprites
   hard-edged. Optionally add a 1-px outline cleanup (a morphological
   open or close; unfake.py does this).
5. **Show it larger.** Nearest-neighbour upscale by an integer factor for
   display. PIL's `Image.NEAREST` does it in the proxy. sd-server also
   lists a built-in `Nearest` upscaler (`api.md:311, 480`).

**Ready-made references:**

- `Retro-Diffusion/pixel-art-fixer` (MIT), from the makers of a
  pixel-artist-built generator;
- `HappyOnigiri/PixelRefiner` (MIT): anti-alias removal, grid, palette,
  transparency;
- `dimtoneff/ComfyUI-PixelArt-Detector` (MIT): pixeldetector plus
  palettes.

**Proposal:** `mcp/pixel_art.py`, a pure function using PIL, numpy and
scipy, and a `style: "pixel_art"` option on both image tools. It resolves
to an allowlisted server-side LoRA plus this post-processor, and never to
a path the model names. Until a LoRA exists, it can run on base-model
output prompted "pixel art". The post-processor makes the grid and palette
true whatever the model drew.

**Where ControlNet helps pixel art:**

- `scribble` or `lineart` from a low-resolution sketch;
- `grayscale` to recolour a sprite under a new palette prompt;
- `pose` for character-sheet consistency;
- `depth` for isometric tiles;
- `none` plus a region to repaint one part of a sprite sheet.

### 6.5 The pixel-art plan, in order

1. **Now, CPU only:**
   - build `pixel_art.py` and its tests on synthetic images;
   - prompt the **base** model for pixel art (after the benchmark) and
     measure how far post-processing alone gets. A LoRA may not be needed
     for a first version.
2. **Rebase sd.cpp** (§3.2), so that 2.1 LoRAs load fully.
3. **Curate a CC0 set** of 40-80 images at one pixel scale, and caption
   it.
4. **Train with ai-toolkit on the A4000** in a maintenance window:
   - NF4 or qfloat8, rank 16-32, 512²;
   - compare against the base model on the same prompts and seeds, with
     n and repeats (PROTOCOL);
   - measure grid adherence: the fraction of cells that come out
     single-colour before post-processing.
5. Serve it through `style`. The Heretic encoder's conditioning at
   inference versus the stock encoder's during training is an open
   question: cache the training embeddings with the Heretic **safetensors**
   (`qwen3vl_8b_bf16_heretic.safetensors`, 17.5 GB, in the same repo) for
   consistency.

---

## 7. Question 5: licences

| item | licence | what it means for us |
|---|---|---|
| Qwen-Image-2.1-Fun-Controlnet-Union (and t8star's re-header, kijai's copies) | **Qwen Research License** (2026-09-20) | **non-commercial only** (research or evaluation). Redistribution must carry the licence and mark modified files. Our GGUF conversion is a derivative, so keep it private or ship the LICENSE with it and a notice. Same class as our base and turbo models (`models/manifest.yaml:175, 192`). |
| Qwen-Image-2.1 base, VAE, turbo | Qwen Research License | unchanged |
| Qwen-Image-2.1-Fun-Acc LoRA | Qwen Research License | the same |
| Heretic Qwen3-VL-8B encoder | Apache-2.0 | unchanged |
| InstantX / Runware / 2512-Fun ControlNets | Apache-2.0 | irrelevant: they do not fit 2.1 |
| 2512 pixel-art LoRAs (prithivMLmods, artificialguybr) | Apache-2.0 | irrelevant: they do not fit 2.1 |
| a LoRA **we** train on 2.1 | a derivative of a Qwen Research License model, so non-commercial, whatever licence the dataset has | the dataset's own terms stack on top: CC0 is clean; CC-BY-SA carries attribution and share-alike |
| Depth-Anything-V2-**Small** / DWPose | Apache-2.0 | fine. Depth-Anything-V2 **Base/Large are CC-BY-NC-4.0**, so avoid them. |
| sd.cpp / ai-toolkit / VideoX-Fun / DiffSynth / pixel-art tools | MIT / MIT / Apache-2.0 / Apache-2.0 / MIT | fine |
| ComfyUI / SimpleTuner | GPL-3.0 / AGPL-3.0 | running them locally is fine; redistributing is copyleft |

---

## 8. Manifest placeholders (proposal only; neither manifest was edited)

**`models/manifest.yaml`**, after `qwen-image-2.1-vae-bf16`:

```yaml
  - id: qwen-image-2.1-fun-controlnet-union-bf16
    role: "ControlNet-Union branch (source, not served): converted to GGUF for `imagegen-control`"
    status: proposed
    config_entries: []
    path: "${models}/qwen-image-2.1/controlnet/Qwen-Image-2.1-Fun-Controlnet-Union.safetensors"
    size: 7550979904
    sha256: 65d6b66d734da9e7ff5ef04e7db3a133553a52a3f29a7fcb3e9cce8fa21dcfcd
    provenance:
      status: proposed
      repo: alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union
      revision: 8a4702014d4dabb5f896fcba917e2ee0a961465f
      revision_basis: tree-api
      filename: Qwen-Image-2.1-Fun-Controlnet-Union.safetensors
      checked: "2026-09-25"
      licence: "Qwen Research License (other / qwen-research); non-commercial"
      doc: docs/IMAGE-CONTROL.md

  - id: qwen-image-2.1-fun-controlnet-union-q8_0
    role: "ControlNet-Union for `imagegen-control` (staged under --max-vram 6)"
    status: proposed
    config_entries: [imagegen-control]
    path: "${models}/qwen-image-2.1/controlnet/qwen-image-2.1-fun-cn-union-Q8_0.gguf"
    size: null          # ~4,012,015,616 expected (docs/IMAGE-CONTROL.md 4.1)
    sha256: null        # recorded after `sd-cli -M convert --type q8_0`
    provenance:
      status: built_here
      built_from: qwen-image-2.1-fun-controlnet-union-bf16
      tool: "sd-cli -M convert (engine sd-cpp, rebased build)"
      licence: "Qwen Research License (derivative; ship LICENSE + modification notice if ever redistributed)"
      doc: docs/IMAGE-CONTROL.md
```

**`engines/manifest.yaml`**, under `sd-cpp`: bump `base_commit` to a
master commit at or after `510bccf`, and replace the patch list:

```yaml
    patches:
      # 0001 (PR #2043) dropped IF #2054's 1/128 conv scaling passes the
      # white-block re-test (bench/imagegen/white_blocks.py, same seeds).
      - file: 0002-qwen-image-2.1-fun-controlnet-union.patch   # local, docs/IMAGE-CONTROL.md 3.2
        sha256: null
        source: local
    config_refs: [imagegen, imagegen-turbo, imagegen-control]
```

**`config.yaml`** (not a manifest, and not edited): `imagegen-control`
would be `imagegen`'s command plus `--control-net
${models}/qwen-image-2.1/controlnet/qwen-image-2.1-fun-cn-union-Q8_0.gguf`
(or the merged-GGUF variant), with `--cfg-scale 1.0 --steps 40`. It
would join the `imagegen` swap group.

---

## 9. Open questions (each needs a measurement or an operator decision)

1. **Quality.** Does the branch deliver on our kinds of request? Judge it
   first with our own synthetic inputs on the vendor's demo Space. It is
   remote and costs no A4000 time. Never send the operator's private
   media.
2. **Does the ControlNet work on the Viggle v0.1 turbo student**, or on
   the base with PAI's Fun-Acc 4-step LoRA? That decides whether a
   control image takes ~30 s or ~160 s. It is unknown; there is no report
   either way.
3. **Quantisation.** Q8_0 against Q5_K against bf16 for the branch, on the
   same seeds. No one has published this for any DiT ControlNet in GGUF.
4. **Peak VRAM** of `imagegen-control` under `--max-vram 6` at 1024² and
   at 1344². This replaces the estimated `SIZES` row.
5. **The rebase.** Does #2054 (leejet's conv scaling) remove the white
   blocks as #2043 did (0 in 39)? Turbo's absolute sigmas must land first.
6. **The Heretic encoder** against stock Qwen3-VL-8B hidden states: the
   drift for text-to-image, control and LoRA training alike.
7. **Where to run the phase-2 preprocessors** (depth, pose): in the proxy
   process with onnxruntime, or in the worker's cpu lane. The proxy
   stays lean if it is the worker.
8. **Should `region` grow shapes** (an ellipse, a polygon)? Should inpaint
   get its own trigger-listed tool if the model mis-selects? AGENTS.md:
   fix the description first.
9. **Commercial use.** Everything here is under the non-commercial Qwen
   Research License. Is the stack's use research or evaluation? The
   question already stands for the models in service.

## 10. Next steps, in order

1. *(No GPU, now.)* Operator review of this document and the tool design
   (§5). Quality probe on the demo Space with synthetic inputs.
2. *(After the benchmark, operator approval.)*
   - Download the ControlNet (7.55 GB).
   - Convert it to Q8_0 and Q5_K on the CPU.
   - Make golden reference images with VideoX-Fun: canny, depth, pose and
     inpaint; seeds fixed; 512² and 1024².
   - That runs in a maintenance window, as the one GPU consumer.
3. **Rebase sd.cpp** to master (at or after `510bccf`) with turbo's
   absolute sigmas. Run the white-block and turbo A/B re-tests.
   `deploy_check.py`.
4. **The sd.cpp patch** (§3.2). Match it against the goldens, then
   measure the peak and seconds (open questions 3 and 4).
5. **The tool**, phase 1: canny, grayscale, scribble, as_is and none, plus
   `region`. Offline tests, then a live test in `mcp/test_live_stack.py`
   (the images group).
6. **Pixel art** (§6.5), in parallel with steps 3-5 where it is CPU-only.

---

## Sources

Retrieved 2026-09-25. "HF" means huggingface.co.

**The ControlNet and its ports**
- ControlNet card, LICENSE, tree API and safetensors header (range-read):
  https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union
  (revision `8a47020`) [official]
- VideoX-Fun, `main`: `videox_fun/models/qwenimage21_transformer2d_control.py`,
  `videox_fun/pipeline/pipeline_qwenimage21_control.py`,
  `examples/qwenimage21_fun/predict_t2i_control.py`,
  `scripts/qwenimage21/README_TRAIN.md`:
  https://github.com/aigc-apps/VideoX-Fun [official]
- Demo Space `app.py` (40 steps, cfg 1, Canny 100/200, "1024 → 37-38 s" on
  ZeroGPU): https://huggingface.co/spaces/hugging-apps/qwen-image-2-1-controlnet-union-demo
  [official]
- ComfyUI PR #16519, "Support Qwen-Image 2.1 union fun controlnet"
  (kijai, open, +242/-17; diff read): https://github.com/Comfy-Org/ComfyUI/pull/16519
  [maintainer]
- https://huggingface.co/t8star/Qwen-Image-2.1-Fun-Controlnet-Union-Comfy ;
  https://huggingface.co/Kijai/QwenImage_experimental [community]

**Other models and adapters**
- https://huggingface.co/InstantX/Qwen-Image-ControlNet-Union (`config.json`) ;
  https://huggingface.co/Runware/Qwen-Image-ControlNet-Union ;
  https://huggingface.co/alibaba-pai/Qwen-Image-2512-Fun-Controlnet-Union
- `Qwen/Qwen-Image-2.1` `transformer/config.json`:
  https://huggingface.co/Qwen/Qwen-Image-2.1 [official]
- https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Acc-LoRAs (card and
  header) [official]
- https://huggingface.co/pottokao/Qwen-Image-2.1-Text-Encoder-Heretic-GGUF
  (tree: bf16 and fp8 safetensors present) [community]

**stable-diffusion.cpp**
- Issues #1033, #1365, #1406, #2051; PRs #2043, #2054, #2057, #2059, #2062;
  the PR list for 2026-09-13 → 25: https://github.com/leejet/stable-diffusion.cpp
  [maintainer / community]
- Local source at `C:/Users/jwals/sdcpp-pr2043`:
  - `src/model/diffusion/{control,unet,model,qwen_image_2_1,wan}.hpp`
  - `src/pipeline/{image,diffusion_engine,model_builders}.cpp`
  - `src/convert.cpp`, `src/model_loader.cpp:1511-1541`
  - `src/name_conversion.cpp:1470-1530`
  - `src/runtime/preprocessing.hpp:296`
  - `examples/common/common.cpp`, `examples/server/{routes_sdapi.cpp,api.md}`
  - [ours]

**Pixel-art LoRAs and trainers**
- https://huggingface.co/prithivMLmods/Qwen-Image-2512-Pixel-Art-LoRA ;
  https://huggingface.co/laoK888/Qwen-Image-2512-Pixel-Art-LoRA ;
  https://huggingface.co/artificialguybr/PIXELART-REDMOND-QWENIMAGE [community]
- Civitai API (`/api/v1/models`, queries pixel / sprite / 8-bit / retro; base
  models Qwen, Qwen 2, Qwen 2.1): https://civitai.com/api/v1/models [community]
- ai-toolkit `extensions_built_in/diffusion_models/qwen_image_2/`, commits
  `c2622ed`, `086b663`, `07abdbe`: https://github.com/ostris/ai-toolkit
- DiffSynth-Studio `docs/en/Model_Details/Qwen-Image-2.1.md`:
  https://github.com/modelscope/DiffSynth-Studio
- https://huggingface.co/AcademiaSD/Qwen-Image-2.1-NF4-for-LoRA-Training ;
  https://github.com/bghira/SimpleTuner

**Pixel-art post-processing**
- https://github.com/KennethJAllen/proper-pixel-art ;
  https://github.com/jenissimo/unfake.js ;
  https://github.com/painebenjamin/unfake.py ;
  https://github.com/Astropulse/pixeldetector ;
  https://github.com/Retro-Diffusion/pixel-art-fixer ;
  https://github.com/HappyOnigiri/PixelRefiner ;
  https://github.com/dimtoneff/ComfyUI-PixelArt-Detector (all MIT, per the
  GitHub API)

**Preprocessors**
- https://huggingface.co/onnx-community/depth-anything-v2-small ;
  https://huggingface.co/depth-anything/Depth-Anything-V2-Large
  (cc-by-nc-4.0) ; https://huggingface.co/yzd-v/DWPose ;
  https://github.com/huggingface/controlnet_aux

**Local**
- `nvidia-smi --query-gpu` (2026-09-25 ~21:30):
  - 5060 Ti 15,298 used / 753 free, link x8 gen5;
  - A4000 4,844 used / 11,323 free, link **x4** (max x16).
- RAM: 63.7 GB total, 26.2 GB free.
- Repo files: `docs/IMAGEGEN.md`, `docs/IMAGEGEN-COMMUNITY.md`,
  `docs/IMAGEGEN-TURBO.md`, `docs/LIVE-COVERAGE.md`, `mcp/images.py`,
  `mcp/gpu_room.py`, `mcp/vision.py`, `mcp/proxy.py`, `config.yaml`,
  `models/manifest.yaml`, `engines/manifest.yaml`.
