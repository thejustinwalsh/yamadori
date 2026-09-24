# Image generation: what upstream and the community recommend

Research done on 2026-09-24. It is about the two image models in `config.yaml`,
`imagegen` (Qwen-Image-2.1 Q5_K_M, 20 steps, cfg 6) and `imagegen-turbo`
(Viggle v0.1 4-step, cfg 1), both run by stable-diffusion.cpp `c92d73c` under
`--offload-to-cpu --max-vram 6 --diffusion-fa` with the bf16 VAE and the
Heretic Q4_K_M Qwen3-VL-8B encoder.

**How it was done.**
- Nothing ran on either GPU, and no file other than this one was changed.
- Read: model cards, config files and repo trees on Hugging Face; the
  diffusers, ComfyUI and sd.cpp source on GitHub; sd.cpp issues and PRs; and
  the Hugging Face discussions under every repo involved.
- Read locally: our sd.cpp tree at `c92d73c`.
- One read-only CPU check: the alpha channel inside three white blocks
  already in `index/media` (§2).

**Labels.** Every claim carries one of these:

| label | meaning |
|---|---|
| **[official]** | Qwen's card, README or config; the diffusers pipeline Qwen points to; Viggle's own card, Space code or config for the turbo |
| **[maintainer]** | a repo owner or contributor, in their own issue, PR or discussion: leejet (sd.cpp owner), CarlGao4 and stduhpf (sd.cpp contributors), yycc (answers on Viggle's repo as its author) |
| **[community]** | anyone else. Anecdotal unless the source gives n. |
| **[ours]** | arithmetic, or a reading of code, done for this document. Not a measurement. |

Dates are the date of the post or commit. Links are in "Sources" at the end,
numbered [S1]...

---

## 0. The short version

### Where we differ from the recommendations

| setting | ours | recommended | who says so |
|---|---|---|---|
| **base cfg** | `--cfg-scale 6.0` (so 2 passes per step) | **1.0: no guidance** | [official] diffusers default `true_cfg_scale=1.0`, docstring "meant to be sampled without guidance" [S2]. ComfyUI's official template uses cfg 1 [S5]. Our 6.0 comes from sd.cpp's own doc example [S10], which unsloth's card copies [S16]. An sd.cpp contributor points out the mismatch [S12]. |
| **base steps** | 20 | **40** | [official] README "Default Parameters" and the diffusers default [S1][S2]. ComfyUI's template starts at 25 and says the official pipeline uses 40–50 [S5]. |
| **base schedule** | μ = 1.15 at 1024², no terminal stretch, so the last sigma is 0.1425 | μ = 0.694, `shift_terminal` 0.02, so the last sigma is 0.02 | [official] scheduler config [S3]. [maintainer] Fixed in sd.cpp `b167b94`, which is after our build [S13]. |
| **VAE precision** | the file is bf16, but sd.cpp `c92d73c` **turns it into F16** and computes in F16 | bf16 or fp32. Qwen ships the VAE in fp32; diffusers and ComfyUI run it in bf16. | [official] repo tree [S4], ComfyUI `sd.py` [S6]. [maintainer] F16 is what makes the white blocks (§2) [S11]. |
| **turbo version** | v0.1 full fine-tune (Abiray Q5_K_M), 4 steps | **v0.2.1 LoRA r256 (or r128), 6 steps**, raw sigmas `[1, .9375, .875, .75, .5, .25]` | [official] Viggle card, 2026-09-24: v0.1 is kept "for reproducibility; there is no reason to prefer" it [S20]. [community] Some users still prefer v0.1 for text-to-image (§4.4). |
| **turbo sampling** | 4 steps, cfg 1, euler, `base_shift=0.5,max_shift=0.69355`, no terminal | **the same, for v0.1** | [official] v0.1 is `num_inference_steps=4`, the shipped scheduler (`shift_terminal: null`), `true_cfg_scale=1.0` [S20][S21]. [ours] Our sigmas equal the reference to 4 decimals (§4.2). |
| **turbo sizes** | anything up to 1344², including 1536x1024 | the 1024²-area buckets (1024², 1184x896, 1248x832, 1376x768, ...) or the 2048² area | [official] Viggle's resolution table [S20]. 1536x1024 falls between the two trained areas. |
| **prompt** | the 27B writes it | the official PE-T2I rewriter (Qwen3.5-9B) | [official] Qwen README [S1]. Viggle: its student was "distilled against prompt-enhanced teacher targets" [S20]. |

### The white blocks: cause and fix

- **The cause is known, and it is the VAE precision, not the budget.**
  - [maintainer] sd.cpp forces the Wan-family VAE's conv weights to F16.
    The Qwen-Image-2.1 decoder's residual stream "legitimately reaches
    magnitudes beyond the f16 range (±65504) in some regions". It becomes
    `+inf`, then NaN, then 255 in the uint8 output (CarlGao4, PR #2043,
    2026-09-24) [S11].
  - [ours] Our build has this. `src/model/vae/wan_vae.hpp:36` creates every
    VAE conv weight as `GGML_TYPE_F16`, whatever the file holds.
  - [ours] In all three of our production blocks, every pixel is 255 in
    **all four channels**, alpha included. That fits NaN saturation, not a
    white region the model painted.
- **Why VRAM pressure matters.** It decides the tiling, and the tiling decides
  where the overflow lands. See §2.3.
- **The fix is upstream and not merged.**
  - PR #2043 keeps the VAE's stored dtype, adds `--vae-dtype`, and forces
    F32 im2col for F32 weights [S11].
  - leejet (2026-09-24) would rather force only the offending op to F32, "to
    keep the performance impact and VRAM increase minimal" [S11].
- **Until then**, larger tiles or an untiled decode only move the seams.
  `--backend vae=cpu` does **not** help on `c92d73c`: the weights are still
  F16 (§2.4).

### Worth trying (details in §6)

1. **Build sd.cpp with PR #2043, or its successor.** Re-decode the known
   failing seeds under the unchanged 6 GiB budget and count blocks with
   `bench/imagegen/white_blocks.py`. Measure decode VRAM too; F32 im2col
   costs more.
2. **Base: cfg 1 at 40 steps.** That is the same 40 transformer passes as
   cfg 6 at 20 steps, so about the same time. Also use the official schedule
   (sd.cpp ≥ `b167b94`, or `--sigmas`). Run it as a paired comparison,
   not as a switch.
3. **Before any sd.cpp upgrade, fix the turbo flags.** On `b167b94`, our
   `--extra-sample-args` gives μ 0.594 and a final sigma of 0.02, which is
   what Viggle says "would wreck the last step". Use absolute `--sigmas` per
   size instead (§4.3).
4. **Optional: v0.2.1.** It needs a LoRA path that sd.cpp does not have for
   our fused-MLP base GGUF (§4.5).

---

## 1. Qwen-Image-2.1: official settings

Qwen-Image-2.1 was released 2026-09-20 [S1]. It is not Qwen-Image
(2025) or Qwen-Image-2512: those use a different VAE and pipeline, and their
settings (cfg 4.0, 50 steps [S7]) do not carry over.

| setting | value | label and source |
|---|---|---|
| steps | **40** | [official] README "Default Parameters"; diffusers `num_inference_steps: int = 40` [S1][S2] |
| guidance | **`true_cfg_scale=1.0`: no CFG.** CFG turns on only with `true_cfg_scale > 1` **and** a `negative_prompt`, and then doubles the work per step. | [official] diffusers `pipeline_qwenimage21.py` (added 2026-09-18, PR #14804) and `docs/.../qwenimage21.md`: "The defaults are the values Qwen recommends: 40 steps and no guidance." [S2] The SGLang example in the README uses `--guidance-scale 1` [S1]. |
| `guidance_scale` | unused (it is for a future guidance-distilled model) | [official] diffusers docstring [S2] |
| negative prompt | none by default. It is ignored at cfg 1. | [official] [S2]; ComfyUI template note: "negative_prompt: unused while cfg is 1" [S5] |
| sampler | Flow-matching Euler (`FlowMatchEulerDiscreteScheduler`) | [official] `model_index.json`, README "Architecture" [S1][S3] |
| schedule | nodes `linspace(1, 1/N, N)`; dynamic **exponential** shift with `base_shift` 0.5, `max_shift` **0.9**, `base_image_seq_len` 256, `max_image_seq_len` **8192**; **`shift_terminal` 0.02** | [official] `scheduler/scheduler_config.json` [S3]; pipeline line 723 [S2] |
| image tokens | (H/16)(W/16); 4,096 at 1024², giving μ = 0.694 | [official] pipeline: `vae_scale_factor = 16`, latents unpatched [S2] |
| resolutions | **native 2K**: 2048², 2400x1792, 1792x2400, 2528x1696, 1696x2528, 2752x1536, 1536x2752. Sides must be multiples of 32. diffusers' `output_resolution` defaults to 1024. | [official] README [S1]; pipeline [S2]. ComfyUI's template defaults to 1024² and says "Prefer multiples of 32" [S5]. |
| dtype | bf16 for everything in the examples | [official] README, diffusers [S1][S2] |
| **VAE** | `AutoencoderKLQwenImage21`: **64-channel RGBA**, 16× spatial compression, `z_dim` 64, Wan 2.2 layout with a temporal kernel of 1. Shipped as **fp32** (`vae/diffusion_pytorch_model.safetensors`, 1,350,989,512 B). | [official] `vae/config.json`, repo tree [S4]. The diffusers VAE is largely "Copied from ... autoencoder_kl_wan" [S2]. ComfyUI's loader comment: "Qwen Image 2.1 VAE: Wan 2.2 layout, temporal kernel 1, no patchify, RGBA" [S6]. |
| VAE dtype in practice | the pipeline casts latents to `vae.dtype`, which is bf16 when loaded as in the examples. **ComfyUI's working dtypes for this VAE are `[bfloat16, float16, float32]`, bf16 first.** Nobody upstream runs it in fp16 by choice. | [official] [S2]; ComfyUI `sd.py:843` [S6] |
| VAE tiling (diffusers) | off by default. With `enable_tiling()`: 256 px tiles, stride 192, a 64 px blended overlap. Output clamped to [-1, 1]. | [official] `autoencoder_kl_qwenimage21.py:1186-1194, 1328` [S2] |
| prompt | the official rewriters `Qwen/Qwen-Image-2.1-PE-T2I` / `-PE-I2I` (Qwen3.5 9B) are recommended "for best results" | [official] README [S1] |
| transparency | prompt template "This is an RGBA image with transparency. ... The image has alpha channel and the background is transparent." | [official] README; sd.cpp doc [S1][S10] |

**What people actually run.**

- **cfg.**
  - [official] ComfyUI's template (2026-09-20): cfg 1, 25 steps, euler,
    `simple`, 1024² [S5].
  - [community] The sd.cpp 6 GB report found cfg 1 about twice as fast as
    cfg > 1, and said it "followed the prompt's style better" [S15].
  - [community] Some raise it: "set cfg to 3-5 & use negative prompt"
    (Qwen #10, 2026-09-20). One suggests cfg 1.5–3.5 for the first 8–12
    steps, then cfg 1 (Comfy #7, 2026-09-21) [S8].
  - [community] Another (Qwen #14, 2026-09-22) says a higher cfg plus a
    negative prompt is fine for generation, but not for editing [S8].
  - All of these are anecdotes with no n.
- **Steps.** [community] 25 steps showed "faint but consistent banding" on
  skin and fur, which "disappeared completely at 40" (Comfy #11, 2026-09-22,
  one user's images) [S8].
- **Size.** [community] On sd.cpp `c92d73c`, Q4_0, 54 images (6 prompts × 3
  seeds × 3 settings): 25 steps at about 1.4 MP (1184²) got text right in
  9/9 images with text, against 4/9 at 1 MP with 15 steps and 6/9 at
  2.4 MP [S15].

---

## 2. White blocks: cause, evidence and fixes

### 2.1 What upstream found

- **The report.** [maintainer] CarlGao4 saw white squares in RGBA outputs
  (PR #2021, 2026-09-21). stduhpf reproduced them at Q8_0.
  - The diffusers Space showed none.
  - sd.cpp with *unquantized* DiT, VAE and encoder still showed them
    (2026-09-23).
  - So it is not the quant [S9].
- **The diagnosis.** PR #2043 (2026-09-24, **open**) [S11]:
  > VAE weights are forced to be loaded and computed using FP16, while the
  > Qwen Image 2.1 VAE decoder residual stream legitimately reaches
  > magnitudes beyond the f16 range (±65504) in some regions. Then it becomes
  > `+inf`, which produces `NaN` in later computation, and be clamped to 255
  > when converting to uint8.
- **What the PR changes.** VAE weights keep their stored dtype, and a new
  `--vae-dtype f16|bf16|f32` sets the compute dtype. Two further changes:
  - `ggml_conv_2d` hard-codes F16 im2col for non-bf16 weights, so F32
    weights get an explicit F32 im2col + mul_mat;
  - the 3-D conv takes the F32 path for F32 weights and skips
    `--vae-conv-direct` there.
- **Its cost.** On Metal, F32 conv falls back to the CPU.
- **Owner's reply.** leejet, 2026-09-24: "Perhaps we can identify the
  specific operator involved and force it to use F32 precision" [S11]. Expect
  a narrower fix, possibly different from #2043.
- **Also fixes #2024.** [community] That issue (Metal, 2026-09-22) has most
  of a transparent background decode as **opaque white**, in islands that
  follow the 16 px latent grid [S9b]. PR #2043 lists it as fixed.

### 2.2 The same thing seen elsewhere

- **The 6 GB report** (leejet/Qwen-Image-2.1-GGUF #3 and its write-up,
  2026-09-23, sd.cpp `c92d73c`, CUDA) [S15]:
  - "a hard-edged rectangle of pure white, 12–16 px wide", aligned to the
    16 px grid, and the same seed reproduces it exactly;
  - 12 of 54 images: 1/18 at 1184², 4/18 at 1024², 7/18 at 2.4 MP;
  - the same with Q4_0, Q4_K and Q5_0;
  - its author did not rule out VAE tiling, and says "GPU precision in the
    VAE" was ruled out. §2.4 explains why that test could not have ruled it
    out on this build.
- **ComfyUI** runs this VAE in bf16 [S6], and we found no white-square
  reports there. The ComfyUI complaints are a grid or moiré (§5).

### 2.3 How it fits our observations ([ours], a reading of our code, not a measurement)

- **Where the blocks sit.** They are at x ≈ 496–520 and y ≈ 256–280 /
  512–536 at 1024² (`bench/imagegen/white_blocks.py`,
  SELF-IMPROVEMENT-LOG #25).
- **Why the decode is tiled at all.** Under `--max-vram 6` the untiled
  decode fails and sd.cpp retries tiled (IMAGEGEN.md measured this every
  time).
- **What the retry tiles are.**
  - `backend_fit.cpp:481-508` sets `rel_size_x = rel_size_y = 0.5`: tiles
    of half the image, 512 px at 1024².
  - The default `--vae-tile-overlap` is 0.5, so tile edges fall at 256, 512
    and 768 px.
  - The recurring positions sit on those edges.
- **Why the budget changes the result.**
  - The overflow depends on the activations. A tile edge sees padding
    instead of its neighbours, so each tiling is a different set of
    activations.
  - `--vae-tile-size 48` (768 px tiles) moves the edges. A 9 GiB budget
    removes tiling altogether.
  - Both removing the blocks on turbo is consistent with this. Neither
    change touches the F16 cause, so neither should be trusted to hold on
    other prompts or seeds.
- **A second route to tiling.** A second consumer on the A4000 lowers free
  memory, and that alone can force the tiled retry. The retry is decided by
  live free memory, not only by the budget.
- **Base versus turbo.** The same VAE and decode path serve both models, so
  there is no reason to expect the base model to differ in kind. Only the
  latents differ.

### 2.4 What does and does not address it

| option | addresses the cause? | notes |
|---|---|---|
| PR #2043 (or leejet's narrower fix) with bf16 or f32 VAE compute | **yes** | [maintainer] [S11]. Decode memory rises, by an unmeasured amount. It may need explicit `--vae-tiling` under a 6 GiB budget, which is then safe to use. |
| the fp32 VAE file (`Qwen/Qwen-Image-2.1` `vae/`, 1.35 GB) on `c92d73c` | **no** | `c92d73c` converts every VAE conv weight to F16 on load (`wan_vae.hpp:36`), whatever the file holds. It only helps together with #2043. |
| `--backend vae=cpu` on `c92d73c` | **no** | Same F16 weights and F16 im2col on the CPU. SELF-IMPROVEMENT-LOG #25 lists it as "(a) full precision"; on this build that premise is wrong. It is also slow, and the 6 GB report saw an untiled 1024² CPU decode use 16 GB of RAM [S15]. |
| `--vae-tiling --vae-tile-size 48x48`, or a larger budget | moves or removes seams only | matches our turbo result; not a fix |
| `--vae-conv-direct` | no, and risky | [community] #2013: `GGML_ASSERT` at 1536x2048 (ROCm; "same code on CUDA") [S14]. #2043 bypasses it for F32 weights. |
| lower-quant or other DiT GGUFs | no | [maintainer] and [community]: reproduced with unquantized weights, Q8_0, Q5_0, Q4_K and Q4_0 [S9][S15] |

---

## 3. stable-diffusion.cpp specifics

### 3.1 Flags (from `c92d73c`'s `common.cpp` unless noted)

| flag | default and meaning | community and maintainer notes |
|---|---|---|
| `--vae-tiling` | off; sd.cpp retries tiled only after an out-of-memory failure | [community] 6 GB report: "Keep `--vae-tiling`", because the untiled 1 MP decode needs more than 6 GB and falls back anyway [S15]. [official] Viggle's Space code runs "No VAE tiling anywhere ... tiled decodes are not the validated pipeline either" [S21]. |
| `--vae-tile-size` | `32x32` **latent** units (512 px for this 16× VAE) | [community] 6 GB report used `24x24` (384 px) [S15]. No official recommendation for Qwen-Image-2.1. |
| `--vae-relative-tile-size` | overrides `--vae-tile-size`; the OOM retry uses 0.5 (half the image) | [ours] `backend_fit.cpp:492` |
| `--vae-tile-overlap` | 0.5 of the tile | No Qwen-specific advice found. diffusers blends 64 px of a 256 px tile (25%) [S2]. |
| `--vae-conv-direct` | off | see §2.4: an assert at large sizes, #2013 [S14] |
| `--diffusion-fa` | off | [community] "the biggest win": −35% sampling time with a visually identical image; it is also what makes 2.4 MP fit in 6 GB [S15]. [maintainer] #2041's grid shows with and without FA [S12b]. #2040: store the Qwen prefix K/V as F16 when FA is on [S14]. No quality caveat found for CUDA. |
| `--offload-to-cpu` | weights live in RAM and are staged in | [community] #2042 (CUDA, 16 GB): the budget check passed and the device check missed by 0.64 MB, with no fallback. [maintainer] Fixed by #2046 (`1a2330d`, 128 MiB headroom), which is after our build [S14]. |
| `--max-vram` | GiB budget per device; 0 = live free memory; **negative = reserve that much free** | [community] #2042: VMM-pool memory from earlier phases (the encoder, or the VAE on the same device) "is invisible to the model manager"; "a negative `--max-vram` reserve is the only flag that accounts for it" [S14]. [ours] Relevant to the A4000 coordinator: the 6 GiB cap bounds sd.cpp's managed buffers, not the pool. |
| `--extra-sample-args` | Flux scheduler keys: `base_shift`, `max_shift` only. There is no key for the anchor or for `shift_terminal`, on `c92d73c` or on master. | [ours] `denoiser.hpp`, local and master |
| `--sigmas` | absolute sigmas; no shift is applied | [ours] `common.cpp:1702`. Per request, the parser reads `sample_params.custom_sigmas` from `<sd_cpp_extra_args>` (`common.cpp:2151`). Not tried. |

### 3.2 Upstream changes after our build (`c92d73c` → `b167b94`, 2026-09-23 → 24) [S13][S14]

- **`b167b94`, "align Qwen Image 2.1 flow schedule with official defaults"
  (#2048, leejet).**
  - For `VERSION_QWEN_IMAGE_2_1` it sets `max_image_seq_len` 8192,
    `max_shift` 0.9 and `shift_terminal` 0.02, and stretches the last
    sigma.
  - It closes #2041: a periodic 8 px / 4 px grid at native 2K sizes. In
    #2041 CarlGao4 showed the grid is already in sd.cpp's latent (diffusers'
    VAE decoding sd.cpp's latent reproduces it), while diffusers' own
    pipeline does not show it.
  - [ours] The fix is aimed at 2K. Whether it also changes 1024² output is
    not reported.
  - **It changes what our turbo flags mean** (§4.3).
- **`70c1dbc`, #2038.** It runs one-frame Wan VAE convolutions as 2D
  convolutions. That makes decode faster on Metal (74.5 s → 7.3 s at 512²).
  "the activations go through im2col in f16 either way", so it does not
  touch the white-block cause.
- **`1a2330d`, #2046.** It adds 128 MiB of headroom before choosing
  monolithic execution (the #2042 fix).
- **Also landed:** #2032 (latent2rgba preview), #2034 and #2045
  (conditioning, prefix-cache and cache types), #2033 (cfg special cases),
  #2026 (upscale endpoint). None bears on quality at our settings, as far as
  the titles and descriptions say.

### 3.3 Quantization

- **DiT.**
  - [official] Qwen ships bf16 only. leejet's sd.cpp GGUFs run Q2_K–Q8_0;
    unsloth's "Dynamic 2.0" GGUFs (ours) upcast sensitive tensors "from a
    measured sensitivity scan" [S16][S17].
  - [community] On a 6 GB card, Q4_K and Q5_0 "did not look better than
    Q4_0" [S15].
  - [maintainer] #2041's grid is the same at Q8_0 and Q6_K.
  - No published comparison of Q5_K_M against Q8_0 or bf16 exists for 2.1.
    Q5_K_M is not contradicted by anything found.
- **An older warning, a different model.** [community] #1385 (Qwen-Image-2512,
  ROCm, 2026-04): K-quants produced black images from activation overflow,
  while Q5_0 and Q8_0 worked [S14]. Not reported for 2.1.
- **Text encoder.**
  - [official] The sd.cpp doc uses `Qwen3VL-8B-Instruct-Q4_K_M.gguf`
    [S10].
  - [community] unsloth measured its UD-Q4_K_XL encoder against Q4_K_M at a
    shared seed: LPIPS 0.029, SSIM 0.959 [S16].
  - Nothing found on the Heretic encoder beyond its own card.

### 3.4 VAE file provenance ([ours])

- **Ours** is `unsloth/Qwen-Image-2.1-FP8` `vae/qwen_image_2.1_vae_bf16.safetensors`:
  675,508,656 B, sha256 `71879ffd…`.
- **The one sd.cpp's doc, ComfyUI's template and Viggle all name** is
  `Comfy-Org/Qwen-Image-2.1` `vae/qwen_image_2.1_vae_bf16.safetensors`:
  675,509,688 B, sha256 `bb21f747…`.
- **They are different files,** 1,032 bytes apart. That is plausibly only
  header metadata, but it is not verified.
- [community] One user found unsloth's *FP8* VAE rejected by ComfyUI and the
  official one working (unsloth #5, 2026-09-23). Ours is the bf16 file, not
  the FP8 one [S16b].

---

## 4. The turbo: Viggle's student, and the build we serve

### 4.1 What Viggle recommends now (model card as of commit `b77064be`, 2026-09-24 05:39 UTC) [S20]

| version | form | steps and schedule | Viggle's verdict |
|---|---|---|---|
| **v0.2.1** (2026-09-24) | LoRA r256, alpha 256, bf16, 1.36 GB; **r128** SVD cut, 680 MB, LPIPS vs r256 "at [the] noise floor" | **6 steps, raw sigma nodes `[1.0, 0.9375, 0.875, 0.75, 0.5, 0.25]`**, shifted by the pipeline's resolution-dependent μ, **no `shift_terminal`** | "use this one"; what the Space runs |
| v0.2 (2026-09-23) | LoRA r256 / r128 | the same 6-step nodes (launched at 5) | kept for reference |
| **v0.1** (2026-09-22) | **full fine-tune `transformer/` (what we serve)**, and the r64 LoRA | `num_inference_steps=4`, no `sigmas=`, the shipped scheduler | "kept in the repository unchanged for reproducibility; there is no reason to prefer them" |

**Viggle's own numbers** (96 held-out requests, against the 40-step base with
prompt enhancement; "our own" metrics, no standard benchmark):

| | v0.1 full | v0.2.1, 6 steps |
|---|---|---|
| diversity × base | 0.72 | 0.98 |
| composition drift | −0.033 | +0.000 |

**Rules on the card** ("Rules that matter") [S20]:
- `true_cfg_scale=1.0` and no negative prompt. "CFG ... do[es] not help."
- Change the step count only at the high-noise end: keep
  `0.875, 0.75, 0.5, 0.25` and split `1 → 0.875`.
  - 7 steps is `[1, 0.9583, 0.9167, 0.875, 0.75, 0.5, 0.25]`.
  - Uniform `linspace` schedules at other step counts do not help.
- Use the shipped scheduler, or `shift_terminal=None`. "The base config's
  `shift_terminal: 0.02` would wreck the last step."
- LoRA scale 1.0.
- **Never merge the LoRA** into bf16: "merging into bf16 is lossy, loading it
  at runtime is exact". The Space's comment says a merge keeps about 47% of
  the delta; the ComfyUI node's docstring says about 70% [S21].
- Prompt rewriting with PE-T2I is recommended.
- **Sizes:** trained at the 1024² and 2048² areas for text-to-image
  (1024², 1184x896, 896x1184, 1376x768, 768x1376, 1248x832, 832x1248; 2048²,
  2368x1760, 2720x1536, ...). "2K output is not validated."
- **Known limits:** small or long text garbles more than with the 40-step
  base; complicated edits degrade; RGBA output is untested.

**The Space's code** (`app.py`) [S21]:
- `STEPS = 6`, `sigmas=raw_nodes(steps)`, `true_cfg_scale=1.0`, bf16, and
  the scheduler rebuilt with `shift_terminal=None`.
- **No VAE tiling anywhere.** On the encode side "tiling ... wrecks
  reference-conditioned edits"; on the decode side "tiled decodes are not
  the validated pipeline either".

**The ComfyUI sigma node** (`comfyui/viggle_turbo.py`) [S21]:
- `mu = 0.5 + 0.4 * (tokens - 256) / (8192 - 256)`, with
  tokens = (H/16)(W/16).
- `sigma = e^mu / (e^mu + 1/t - 1)`, then a final 0.
- That is exactly the diffusers formula our v0.1 config reproduces.

### 4.2 Our turbo config against the v0.1 contract ([ours], arithmetic)

Our flags give sd.cpp `c92d73c`, per size:
μ = 0.5 + (0.69355 − 0.5)(seq − 256)/(4096 − 256).
That equals Viggle's 0.5 + 0.4(seq − 256)/7936 at every size.

| size | μ (ref = ours) | v0.1 4-step sigmas (ref = ours) |
|---|---:|---|
| 1024² | 0.694 | 1.0, 0.8572, 0.6668, 0.4001, 0 |
| 1344² | 0.843 | 1.0, 0.8745, 0.6990, 0.4364, 0 |
| 1536x1024 | 0.797 | 1.0, 0.8694, 0.6893, 0.4251, 0 |
| 1344x768 | 0.690 | 1.0, 0.8568, 0.6660, 0.3993, 0 |

**For v0.1, the config matches Viggle's contract exactly:** 4 steps, cfg 1,
euler, this schedule, and no terminal stretch.

- The smoke test's log line `mu=0.694` (IMAGEGEN.md) agrees.
- [community] An unrelated sd-server launcher arrived at the same trick
  independently (`--extra-sample-args "max_shift=0.6935"`, with Abiray
  **Q8_0**, rupertgermann/ai-image-aura PR #107, 2026-09-24) [S23].

### 4.3 The upgrade hazard ([ours], arithmetic from `b167b94`'s code)

On sd.cpp ≥ `b167b94` the Qwen-2.1 defaults change to an 8,192 anchor,
`max_shift` 0.9 and `shift_terminal` 0.02 [S13].

| 1024², 4 steps | μ | sigmas |
|---|---:|---|
| Viggle v0.1 reference | 0.694 | 1.0, 0.8572, 0.6668, **0.4001**, 0 |
| **our current flags on `b167b94`** | **0.594** | 1.0, 0.7557, 0.4409, **0.02**, 0 |
| no `--extra-sample-args` on `b167b94` | 0.694 | 1.0, 0.7667, 0.4556, **0.02**, 0 |

- **Both results are wrong.** The terminal stretch cannot be switched off
  with `--extra-sample-args`: only `base_shift` and `max_shift` are parsed.
- **After an upgrade, turbo must pass absolute sigmas:**
  - 1024²: `--sigmas 1.0,0.8572,0.6668,0.4001,0` (v0.1);
  - or per request, as `custom_sigmas` computed from each size (table in
    §4.2).
- **The base model is the opposite case:** it *should* get the new defaults.
  So an upgrade has to change the two configs in different directions.

### 4.4 What we serve: `Abiray/Qwen-Image-2.1-viggle-4-steps-turbo-GGUF` `qwen_image_2.1_turbo_Q5_K_M.gguf`

**The card** [S18]:
- Built from Viggle's **v0.1** full transformer, with "city96/ComfyUI-GGUF
  and llama.cpp".
- `norm_q`, `norm_k` and `text_norm` stay F32 "to eliminate NaN errors and
  color banding"; `img_in` and `txt_in` are BF16; `to_v` is Q6_K in the `_M`
  tiers.
- **Recommended:** 4 **or 8** steps, cfg 1.0, euler, scheduler `simple` ("or
  FlowMatch Euler"), negative blank, "Do not increase to 25 or 40".
- **Quant advice:** Q4_K_M "Recommended for most GPUs"; Q5_K_M "Best
  quality-to-size balance"; Q8_0 "Near-lossless", for 16 GB and up.
- ComfyUI only. **No mention of sd.cpp.**
- [ours] "`simple`" is a ComfyUI KSampler scheduler. Viggle's own ComfyUI
  node exists to replace it, so it is not Viggle's schedule. "8 steps" is
  not in Viggle's v0.1 contract either.

**Discussions.** One, #1 (2026-09-23), a ComfyUI LoRA-versus-transformer
workflow. No bug reports [S18b].

**Outside ComfyUI.**
- **No report was found** of the Abiray GGUF misbehaving in sd.cpp, and no
  report of black or white blocks tied to it.
- One public sd-server script uses Abiray **Q8_0** in sd.cpp with the same
  shift fix [S23].
- Our own load (sd.cpp reports `Qwen Image 2.1`, 297 tensors, split MLP) and
  the smoke test are the only other sd.cpp evidence (IMAGEGEN.md).

**`realrebelai/Viggle_Qwen-Image-2.1-Turbo_GGUFs`** [S19]:
- Also from the v0.1 full transformer (sha256 verified against upstream).
- Its mixed "HQv3" ladder keeps attention at Q8_0 even in its Q5_K_M.
- It warns the GGUF runtime "did **not** reproduce the upstream ... output
  exactly even at BF16-equivalent GGUF precision", with differences seen
  "**before** lower-bit quantization". [ours] That is a statement about its
  ComfyUI-GGUF runtime, not about sd.cpp.
- It recommends Q4_K_M for most users, and Q5_K_M, Q6_K or Q8_0 with more
  memory.
- Its one discussion is praise [S19b].

**Higher quant?**
- Nobody publishes a quality comparison between turbo quants.
- The sources give only generic "more bits if you have the memory" advice;
  Abiray calls its Q8_0 near-lossless.
- [ours] Under a 6 GiB staged budget the file size costs transfer time, not
  resident VRAM. Q8_0 (7.59 GB) is plausible, but unmeasured.

**Known turbo artifacts** (v0.1 unless noted):

| artifact | label and source |
|---|---|
| seed collapse (diversity 0.72× base), composition drift | [official] card [S20] |
| small or long text garbles more than the 40-step base | [official] card [S20] |
| edits shifted >100 px left (v0.1 LoRA, edit mode). Maintainer: "v0.2 coming soon with that fixed" | [community] #5, 2026-09-23; [maintainer] yycc [S22] |
| "Oversharpness is hard to avoid though due to DMD"; leg and anatomy issues "should be fixable if you try it on 8 steps (more steps on high-noise)" | [maintainer] yycc, #8, 2026-09-24 [S22] |
| v0.2 prompt leak (one attribute spreads across a crowd), body horror; the user prefers v0.1 | [community] #7, 2026-09-24. Maintainer: possibly "over distilled", and prompt enhancement helps [S22]. |
| v0.2 T2I "completely broken" in ComfyUI; the retraction says those images were v0.2, not v0.2.1; v0.2.1 "still has minor artifacts and blurry composition", and v0.1 has "fewer artifacts" but less detail | [community] #8, 2026-09-24 [S22] |
| v0.2 LoRA keys not loaded by ComfyUI's stock loader (`img_mlp.gate_up...`), which breaks T2I | [community] #4, 2026-09-23. Viggle then shipped its own unmerged-LoRA node [S22]. |
| better anime results; one user runs **cfg 3 with 10–12 steps** on v0.1 | [community] #2 and #3 (off-contract) [S22] |
| white or black blocks | **none reported** for the turbo or its GGUFs. Ours come from the shared VAE (§2). |

### 4.5 v0.2.1 on sd.cpp: what it would take ([ours])

- **Only v0.1 has a full transformer.** v0.2 and v0.2.1 are LoRA-only.
- **The LoRA targets the split MLP names.** Viggle's ComfyUI node documents
  "fused SwiGLU: gate_up = [gate_layer; proj]" [S21]. That is the mapping
  IMAGEGEN-TURBO.md §2 found missing in sd.cpp for our fused-MLP unsloth base
  (`img_mlp.gate_up`).
- **So it needs either:**
  - (a) the gate_layer → `gate_up.weight`, proj → `gate_up.weight.1` mapping
    patch; or
  - (b) a base GGUF that keeps the MLP split. Whether leejet's GGUFs do was
    not checked, because that means reading their headers.
- **Viggle requires the LoRA unmerged.** sd.cpp chooses `at_runtime` for
  quantized weights by itself (IMAGEGEN-TURBO.md §2), which is the unmerged
  form.
- **The schedule is 6 absolute sigmas per size:**

| size | v0.2.1 6-step sigmas (from Viggle's formula) |
|---|---|
| 1024² | 1.0, 0.9678, 0.9334, 0.8572, 0.6668, 0.4001, 0 |
| 1344² | 1.0, 0.9721, 0.9421, 0.8745, 0.6990, 0.4364, 0 |
| 1536x1024 | 1.0, 0.9708, 0.9395, 0.8694, 0.6893, 0.4251, 0 |
| 1344x768 / 768x1344 | 1.0, 0.9677, 0.9332, 0.8568, 0.6660, 0.3993, 0 |

- **Cost:** 6 passes instead of 4, plus the rank-128 or rank-256 side
  matmuls on every targeted linear. Unmeasured.
- **Community verdict on v0.2.1 against v0.1 for text-to-image:** mixed
  (above).

---

## 5. Other VAE artifacts people report (not the white blocks)

- **Diamond grid / checkerboard / moiré on fine texture.**
  - [community] Reported as a property of the Qwen VAE (Qwen #12,
    2026-09-20), reproducible with an encode→decode round trip.
  - Mitigations mentioned: a GLSL node, a node pack, re-encoding through
    another VAE [S8].
  - [community] madebyollin's "Texture-Fix-VAE-for-Qwen-Image-2.1"
    (2026-09-24) fine-tunes the decoder's last two stages. It reports rFID
    3.37 → 2.08, and PSNR 33.30 → 32.86 (COCO 256²). It is a bf16 drop-in
    (675,509,688 B), under the Qwen Research License [S24].
  - [ours] It would be one more file in the F16 path. It does nothing for the
    white blocks until §2's fix lands.
- **A 4 / 8 px grid at 2K in sd.cpp only.** DiT-side, not VAE (§3.2, #2041).
- **Yellow tint.** [community] Fixed with a post-process node (Comfy #10,
  2026-09-22) [S8].
- **Alpha.**
  - [community] Outputs are RGBA even for opaque prompts; 13–21% of pixels
    come back with alpha 224–254 (#2024) [S9b][S15].
  - [ours] The three production images checked have 1.8%, 5.1% and 20.9% of
    pixels with alpha < 255 (minimum 230).
  - A client that composites onto a dark background shows those as slightly
    see-through. Unrelated to the blocks, but it is served today.

---

## 6. What to try, in order (each is a proposal; none has run)

PROTOCOL rules apply to each item: pair on the same seeds, repeat, one GPU
consumer at a time, and label every number with its n.

1. **Fix the VAE precision first.**
   - Build sd.cpp from PR #2043, or from whatever leejet merges in its
     place.
   - Re-decode the known failing seeds with the production flags and the
     6 GiB budget, both with the automatic tiling retry and with explicit
     `--vae-tiling`. The seeds: a1223ed2e6 (turbo, seed 1737429857),
     0f9b34e19f (278719470), f639fdfb21 (base, 48442099), and the rest
     from SELF-IMPROVEMENT-LOG #25.
   - Count with `white_blocks.py`; diff the rest with `diff_variants.py`.
   - Expected: zero blocks at any tile size.
   - Measure decode peak VRAM and seconds, because F32 im2col is larger.
   - Then **drop the tile-size and budget workarounds**; they treat a
     symptom.
2. **Base model: move towards the official path, with a paired A/B.**
   - Arms: cfg 6 at 20 steps (today) against cfg 1 at 40 steps. Both are 40
     transformer passes.
   - Use the official schedule: sd.cpp ≥ `b167b94`, or absolute `--sigmas`
     on `c92d73c`.
   - Keep the text-correctness k/n from IMAGEGEN-TURBO.md §6.
   - One-sided community reports favour cfg 1 at 40 steps (banding gone,
     style fidelity). Others prefer cfg 2.5–5 with a negative prompt. The
     official default is cfg 1.
3. **Turbo: nothing to change today;** it matches v0.1.
   - **Before any sd.cpp upgrade,** replace `--extra-sample-args` with
     absolute sigmas: the server default for 1024², plus
     `sample_params.custom_sigmas` per request from `mcp/images.py` for
     other sizes.
   - Otherwise the upgrade silently moves turbo to μ 0.594 and a 0.02
     final sigma.
4. **Turbo sizes.** Consider snapping requests to Viggle's 1024²-area
   buckets (1184x896 and so on) rather than 1536x1024. The student was not
   trained between the 1024² and 2048² areas.
5. **Optional, with operator approval (downloads and code):**
   - v0.2.1 r128 (680 MB) through a LoRA key-mapping patch or a split-MLP
     base, against v0.1 on the PartiPrompts plan in
     `bench/imagegen/compare_turbo.py`;
   - a stock-encoder or UD-Q4_K_XL encoder arm;
   - the texture-fix VAE, for the grid or moiré complaint.

---

## Sources

Retrieved 2026-09-24. "HF" means huggingface.co.

- [S1] Qwen-Image-2.1 README (QwenLM GitHub; HF card is the same examples): https://github.com/QwenLM/Qwen-Image-2.1 ; https://huggingface.co/Qwen/Qwen-Image-2.1 — release 2026-09-20; Default Parameters; resolutions; prompt rewriting; SGLang `--guidance-scale 1`. [official]
- [S2] diffusers `src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py` (added in `6256aa76`, PR #14804, 2026-09-18), `docs/source/en/api/pipelines/qwenimage21.md`, `src/diffusers/models/autoencoders/autoencoder_kl_qwenimage21.py`: https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/qwenimage21 [official]
- [S3] `Qwen/Qwen-Image-2.1` `scheduler/scheduler_config.json`, `model_index.json`: https://huggingface.co/Qwen/Qwen-Image-2.1/tree/main [official]
- [S4] `Qwen/Qwen-Image-2.1` `vae/config.json` and the tree API (VAE 1,350,989,512 B): https://huggingface.co/api/models/Qwen/Qwen-Image-2.1/tree/main?recursive=true [official]
- [S5] Comfy-Org workflow template `templates/image_qwen_image_2_1_t2i.json` (2026-09-20): https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_qwen_image_2_1_t2i.json [official, ComfyUI]
- [S6] ComfyUI `comfy/sd.py` (Qwen-Image-2.1 VAE branch, `working_dtypes`): https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/sd.py [official, ComfyUI]
- [S7] diffusers QwenImage (original Qwen-Image) docs: https://huggingface.co/docs/diffusers/main/en/api/pipelines/qwenimage [official]
- [S8] HF discussions: Qwen/Qwen-Image-2.1 #10 (2026-09-20), #12 "Diamond grid pattern caused by VAE" (2026-09-20 → 24), #14 (2026-09-21); Comfy-Org/Qwen-Image-2.1 #7 (2026-09-21), #10 yellow tint (2026-09-22), #11 banding at 25 steps (2026-09-22), #14 recommended settings (2026-09-22): https://huggingface.co/Qwen/Qwen-Image-2.1/discussions , https://huggingface.co/Comfy-Org/Qwen-Image-2.1/discussions [community]
- [S9] sd.cpp PR #2021 "Add alpha channel input for Qwen Image 2.1" (2026-09-21, merged 2026-09-22) and its comments by stduhpf and CarlGao4 (2026-09-22/23): https://github.com/leejet/stable-diffusion.cpp/pull/2021 [maintainer]
- [S9b] sd.cpp issue #2024 "transparent (RGBA) output: most of the background decodes opaque white (Metal)" (2026-09-22, open): https://github.com/leejet/stable-diffusion.cpp/issues/2024 [community; maintainer reply]
- [S10] sd.cpp `docs/qwen_image_2.1.md` (at `c92d73c` and master): https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/qwen_image_2.1.md [official, sd.cpp]
- [S11] sd.cpp PR #2043 "fix: Qwen Image 2.1 alpha output white squares (VAE compute path changed)" (CarlGao4, 2026-09-24, open) and leejet's reply (2026-09-24): https://github.com/leejet/stable-diffusion.cpp/pull/2043 [maintainer]
- [S12] The same PR's note on cfg: diffusers' default for this model is 1.0 while sd.cpp defaults to 7.0 globally. [S12b] sd.cpp issue #2041 "periodic 8px/4px grid artifacts at native 2K resolutions" (2026-09-24, closed by #2048), comments by stduhpf, CarlGao4, Green-Sky: https://github.com/leejet/stable-diffusion.cpp/issues/2041 [community report; maintainer diagnosis]
- [S13] sd.cpp PR #2048 "align Qwen Image 2.1 flow schedule with official defaults" (leejet, merged 2026-09-24 as `b167b94`): https://github.com/leejet/stable-diffusion.cpp/pull/2048 ; compare `c92d73c...master`: https://github.com/leejet/stable-diffusion.cpp/compare/c92d73c...master [maintainer]
- [S14] sd.cpp issues and PRs: #2013 `--vae-conv-direct` assert (2026-09-21); #2038 one-frame Wan VAE conv as 2D (2026-09-23); #2040 prefix K/V F16 with FA (2026-09-24); #2042 monolithic check and `--max-vram -N` (2026-09-24); #2046 headroom fix (2026-09-24); #1385 Qwen-Image-2512 k-quant black images (2026-04-01): https://github.com/leejet/stable-diffusion.cpp/issues [community; maintainer fixes]
- [S15] leejet/Qwen-Image-2.1-GGUF discussion #3 "Settings that work well on a 6 GB GPU" (ruben-salas20, 2026-09-23) and its write-up: https://huggingface.co/leejet/Qwen-Image-2.1-GGUF/discussions/3 , https://github.com/ruben-salas20/recipes/tree/main/Qwen-Image-2.1 [community, n=54]
- [S16] unsloth/Qwen-Image-2.1-GGUF card (sd.cpp example at cfg 6.0 and 20 steps; encoder LPIPS): https://huggingface.co/unsloth/Qwen-Image-2.1-GGUF ; [S16b] its discussion #5 (2026-09-23): https://huggingface.co/unsloth/Qwen-Image-2.1-GGUF/discussions/5 [community]
- [S17] leejet/Qwen-Image-2.1-GGUF card and tree: https://huggingface.co/leejet/Qwen-Image-2.1-GGUF [maintainer]
- [S18] Abiray/Qwen-Image-2.1-viggle-4-steps-turbo-GGUF card and tree: https://huggingface.co/Abiray/Qwen-Image-2.1-viggle-4-steps-turbo-GGUF ; [S18b] discussion #1 (2026-09-23) [community]
- [S19] realrebelai/Viggle_Qwen-Image-2.1-Turbo_GGUFs card: https://huggingface.co/realrebelai/Viggle_Qwen-Image-2.1-Turbo_GGUFs ; [S19b] discussion #1 (2026-09-23) [community]
- [S20] Viggle/Qwen-Image-2.1-viggle-turbo model card (v0.2.1, commit `b77064be`, 2026-09-24), NOTICE, `scheduler/scheduler_config.json`, commit history: https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo [official]
- [S21] Viggle Space `app.py` and `comfyui/viggle_turbo.py` in the model repo: https://huggingface.co/spaces/Viggle/Qwen-Image-2.1-viggle-turbo , https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo/tree/main/comfyui [official]
- [S22] Viggle model discussions #2, #3, #4, #5, #7, #8 (2026-09-23/24; yycc answers as the author): https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo/discussions [community; maintainer]
- [S23] rupertgermann/ai-image-aura PR #107 "Qwen Image 2.1 as a local image model via sd-server" (merged 2026-09-24), `scripts/qwen-sd-server.sh`: https://github.com/rupertgermann/ai-image-aura/pull/107 [community]
- [S24] madebyollin/texture-fix-vae-for-qwen-image-2.1 (2026-09-24): https://huggingface.co/madebyollin/texture-fix-vae-for-qwen-image-2.1 [community]
- Local: stable-diffusion.cpp `c92d73c`, `src/model/vae/wan_vae.hpp:28-44`, `src/core/backend_fit.cpp:481-508`, `examples/common/common.cpp` (flag help, `parse_sample_params_json`), `src/runtime/denoiser.hpp` (`FluxScheduler`); `bench/imagegen/white_blocks.py`; `index/media/{a1223ed2e6,0f9b34e19f,f639fdfb21}*.png` (alpha check). [ours]
