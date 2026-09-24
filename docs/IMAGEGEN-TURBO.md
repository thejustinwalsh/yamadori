# Image generation: the Viggle 4-step turbo (research, not trialled)

The operator asked for a look at the Space
`https://huggingface.co/spaces/Viggle/Qwen-Image-2.1-viggle-turbo`.

**Update, 2026-09-23 10:05.** The operator approved the download. Arm T
(`Abiray/…/qwen_image_2.1_turbo_Q5_K_M.gguf`) is downloaded and verified
(size and sha256 below), configured as the llama-swap model
`imagegen-turbo`, selectable per user, and smoke-tested once (n=1: 23.6 s
wall, peak +5,527 MiB, coherent image, `mu=0.694` in the log). The
measurement plan in §6 is now `bench/imagegen/compare_turbo.py`, ready and not
run. docs/IMAGEGEN.md has the details. The text below is the research as
written before the download.

Status on 2026-09-23, before the download:
- **Research only.** Nothing was downloaded, installed or run on either GPU.
- **What was read:**
  - the Space's `app.py` and metadata
  - the model repo's card, `LICENSE`, `NOTICE` and the three config JSONs
  - the HF tree API for every repo named below
  - the local stable-diffusion.cpp tree at `c92d73c`
  - the tensor directory of our own GGUF. Its header was read on the CPU;
    nothing ran on a GPU.
- **Every number in this file is a size, a config value or arithmetic.** None is
  a measurement on this box. The speed figure in §3 is an estimate with n=0.

---

## 1. What it is

**A DMD-distilled 4-step student of `Qwen/Qwen-Image-2.1`.** It was trained by
Viggle and published 2026-09-22 as a "v0.1 preview".

- **Method:** Distribution Matching Distillation. The card says DMD2 or
  SenseFlow style.
- **Training areas:** text-to-image at 1024² and 2048²; editing at 1024² and
  1536².
- **Base:** exactly Qwen-Image-2.1. `transformer/config.json` is
  `QwenImage21Transformer2DModel`: 32 layers, 32 heads × 128, `in_channels` 64,
  `patch_size` 1 and `context_in_dim` 4096. That is our architecture.
- **Frozen parts:** only the transformer changed. `NOTICE` says the processor,
  text encoder and VAE are "loaded from Qwen/Qwen-Image-2.1 at runtime".
- **Two forms, same training:**

| form | what | card's verdict |
|---|---|---|
| full transformer (`transformer/`) | full-parameter fine-tune, bf16. NOTICE: run `v4_anchor` step 400, EMA | **recommended**, and the Space's default (`STUDENT = "full"`). It "edits more faithfully than the LoRA". |
| LoRA, rank 64, alpha 64 | run `v2_16gpu` step 400, EMA. Targets `to_q`, `to_k`, `to_v`, `to_out.0`, `img_mlp.proj`, `img_mlp.gate_layer`, `img_mlp.out`, `modulation.1` and `timestep_embedder.linear_1/2`. Scale 1.0. | "Smaller download, slightly weaker" |

**Inference contract** (the card and `app.py`):

| setting | value | source |
|---|---|---|
| steps | **4** (the Space's slider allows 3–8) | card; `app.py` `STEPS = 4` |
| CFG | **`true_cfg_scale=1.0`**, which means no CFG, a single pass per step and no negative prompt | card; `app.py` |
| sampler | `FlowMatchEulerDiscreteScheduler` (Euler, flow matching) | `scheduler/scheduler_config.json` |
| schedule | dynamic exponential shift: `base_shift` 0.5, `max_shift` 0.9, `base_image_seq_len` 256, `max_image_seq_len` 8192. **`shift_terminal: null`**. The base ships 0.02, the only difference between the two configs. | both `scheduler_config.json` files |
| dtype | bf16 | card |

- **Why `shift_terminal` matters:** the card says the base config's
  `shift_terminal: 0.02` "would wreck the last of the four steps."
- **Prompt enhancement in the Space:** it is **on by default**. The Qwen3-VL-8B
  encoder rewrites the prompt with the official Qwen system prompts before
  generation, at +4–15 s. The Space's impressions therefore include rewritten
  prompts. Our model tool already has the 27B write the prompt.

**Quality claims.** The card claims none quantitatively:
- "Evaluation so far is qualitative, on held-out user requests; no
  quantitative metric is claimed."
- "Text-to-image at 4 steps is usable."
- "Small or long rendered text can garble more often than with the 40-step
  base model."
- Complicated editing is "clearly worse" than the base: multi-reference,
  face swaps and identity edits.
- 2K output is not validated.

Community feedback is from its first day:
- Discussions #1–3 are positive and anecdotal.
- #5 reports an image shifted >100 px to the left **in edit mode** with the
  LoRA.
- No published side-by-side or metric exists.

**Speed claims.** The card claims "4 transformer passes instead of 40" and no
timings. The Space runs on ZeroGPU A10G with `spaces.GPU(duration=90)`.

## 2. The files

**Official repo: `Viggle/Qwen-Image-2.1-viggle-turbo`.** Sizes are from the tree API.

| path | bytes | sha256 (LFS) |
|---|---:|---|
| `transformer/diffusion_pytorch_model.safetensors` (bf16) | 14,230,284,584 | `a011a284b2dcb4b5f73aa7041b770d6bbff601bbb0850f3b6d49740b567b24e5` |
| `Qwen-Image-2.1-viggle-turbo-4step-lora-r64.safetensors` (bf16, diffusers keys) | 339,832,808 | `5c494dce662a95898d0988831af733b203ceced9116a82aa099c997f96ea1a3a` |
| `peft/adapter_model.safetensors` (the same LoRA, F32, peft keys) | 679,597,776 | `c7d944042f506ce46963b6f93795681e6c55b460a677ee4a11c510f48cf797aa` |
| `transformer/config.json`, `scheduler/scheduler_config.json`, `peft/adapter_config.json` | 369 / 484 / 1,126 | |
| `LICENSE` | 7,831 | git oid `13ae08d5…`, **identical** to `Qwen/Qwen-Image-2.1`'s LICENSE |

Viggle publishes **no GGUF**. Three community conversions appeared within 24
hours. Each is **unaffiliated and unverified**, and none mentions sd.cpp.

| repo | built from | files (bytes) |
|---|---|---|
| `Abiray/Qwen-Image-2.1-viggle-4-steps-turbo-GGUF` | Viggle's full transformer, via city96 ComfyUI-GGUF + llama.cpp. `norm_q`, `norm_k` and `text_norm` are kept F32, `img_in` BF16, and `to_v` Q6_K in the higher tiers. Tested only in ComfyUI. | `qwen_image_2.1_turbo_Q5_K_M.gguf` **5,007,399,712** (sha256 `92521fe8d6e25cd8e23dae57f077cac29d07d23ee128c50e79d9b1e60343888a`); Q6_K 5,876,581,152; Q8_0 7,591,582,496; Q4_K_M 4,189,346,592; Q4_K_S 4,059,323,168; Q3_K_M 3,185,947,424 |
| `realrebelai/Viggle_Qwen-Image-2.1-Turbo_GGUFs` | Viggle's full transformer. The tool is not named; it uses a mixed "HQv3" ladder with `img_in`, `txt_in`, `time_text_embed`, `modulation`, `norm_out` and `proj_out` at higher precision. Tested only in ComfyUI. Its card says its GGUF output "did not reproduce the upstream … output exactly even at BF16-equivalent GGUF precision". | `Qwen-Image-2.1-viggle-turbo-Q5_K_M.gguf` **5,959,105,632** (sha256 `9c0f50a82288758a26b63626681dd7e9307ec25c112ecd82cef1d112d8db1b37`); Q6_K 6,907,018,336; Q8_0 7,687,158,880; Q4_K_M 5,556,452,448 |
| `t8star/Qwen-Image-2.1-viggle-turbo-4step-r64-comfy` | Viggle's LoRA with keys renamed to ComfyUI's Qwen-Image-2.1 layout, including "32 fused MLP segment mappings". The payload is unchanged. Its author says full output testing is not done. | `Qwen-Image-2.1-viggle-turbo-4step-r64-comfyui-T8.safetensors` 339,826,976 |

For comparison, our base denoiser `qwen-image-2.1-Q5_K_M.gguf` (unsloth) is
5,390,223,072 bytes.

### Can sd.cpp (`c92d73c`) use it?

**A full-student GGUF: yes, very likely, as a drop-in `--diffusion-model`.**
- sd.cpp detects Qwen-Image-2.1 from tensor names
  (`qwen_image_2_1.hpp: detect_from_weights`).
- It supports both MLP layouts: fused `img_mlp.gate_up`, or split
  `img_mlp.proj` + `img_mlp.gate_layer`.
- Its docs point to city96-style (QuantStack) GGUFs for Qwen-Image.
- **Not verified:** that these particular files load. Read the header before
  the first run (§5, step 0).

**The LoRA on our existing Q5_K_M: not as-is. This is the blocking finding.**

- **What works:**
  - sd.cpp LoRA support is general. `--lora-model-dir` plus `<lora:name:1>`
    works in `sd-cli`. `sd-server` refuses prompt tags and takes a structured
    `"lora": [{"path": ..., "multiplier": 1.0}]` field on
    `/sdapi/v1/txt2img` (`examples/server/api.md`, `routes_sdapi.cpp:153`).
  - The diffusers key suffixes (`lora_A`/`lora_B`, `transformer.` prefix) are
    translated (`name_conversion.cpp:1470-1539`).
- **What fails: our GGUF's MLP is fused.** Its tensor directory has
  `transformer_blocks.N.img_mlp.gate_up.weight` (4096 → 24576, Q5_K) and no
  `img_mlp.proj` or `img_mlp.gate_layer`.
  - The Viggle LoRA targets the split names.
  - sd.cpp's Qwen-Image-2.1 name conversion has no split→fused mapping. It has
    one for Flux and SD3 qkv (`name_conversion.cpp:487-617`, the `.weight.1`
    concat convention in `lora.hpp:get_lora_weight_diff`), not for this model.
- **Effect:** all 128 MLP LoRA tensors (32 blocks × proj/gate × A/B) would be
  logged as `unused lora tensor`. sd.cpp prints `Only (x / y) LoRA tensors
  have been applied` and carries on. The student would run with its attention
  and modulation deltas but not its MLP deltas: a half-applied distillation,
  and likely broken output at 4 steps.
- **The fix:**
  - In sd.cpp, map `gate_layer` → `gate_up.weight` and `proj` →
    `gate_up.weight.1`. The order is gate then proj, because sd.cpp's fused
    forward takes `parts[0]` as the gate (`qwen_image_2_1.hpp:231-234`).
  - That is a code change, either upstream or a local patch. Otherwise the LoRA
    keys need rewriting offline. Neither is done here.

**A LoRA on a quantised base (Q5_K_M), once the keys match.**
- sd.cpp picks the **`at_runtime`** apply mode by itself: the weights are
  quantised and offloaded (`diffusion_engine.cpp:1019-1034`).
- Each targeted linear computes `W_q·x + B·A·x`, with the LoRA held at
  higher precision.
- The LoRA was trained against bf16 weights. Q5_K's weight error is small
  next to a rank-64 delta. That is the usual ComfyUI GGUF+LoRA situation and
  it generally works, but nothing here measures it for a 4-step DMD student.
- **Cost:** extra rank-64 matmuls on every targeted linear per pass, and the
  adapter's tensors also have to live inside the 6 GiB budget.
- **Neither is measured.** The full-student GGUF avoids both, and it is the
  form Viggle recommends.

### The schedule: sd.cpp's default is wrong for this model

sd.cpp does not read a diffusers scheduler config.
- **Its default for Qwen-Image-2.1 is the Flux scheduler**
  (`request.cpp:68`, added in `137f740`) with its Flux defaults: `base_shift`
  0.5, `max_shift` **1.15**, anchors 256 and **4096**.
- **It has no `shift_terminal`.** That happens to be what the turbo wants.
- **The sequence length is not the problem.** sd.cpp's `image_seq_len` is
  (H/16)(W/16), which is 4096 at 1024². diffusers feeds `calculate_shift` the
  same 4096 target tokens at 1024² (diffusers issue #14824).
- **The shift is.** The resulting 4-step sigmas are computed from both
  formulas (plain arithmetic from the constants above; no script needed):

| size | seq | μ diffusers (Viggle) | sigmas, Viggle config | μ sd.cpp default | sigmas, sd.cpp default |
|---|---:|---:|---|---:|---|
| 1024² | 4096 | 0.694 | 1.0, 0.857, 0.667, **0.400**, 0 | 1.150 | 1.0, 0.905, 0.760, **0.513**, 0 |
| 1344² | 7056 | 0.843 | 1.0, 0.875, 0.699, **0.436**, 0 | 1.651 | 1.0, 0.940, 0.839, **0.635**, 0 |
| 1536×1024 | 6144 | 0.797 | 1.0, 0.869, 0.689, **0.425**, 0 | 1.497 | 1.0, 0.931, 0.817, **0.598**, 0 |

- **What the defaults would do:** the last step would have to jump from 0.51
  (or 0.64 at 1344²) to 0, where the student was trained on 0.40. That is the
  kind of mismatch the card warns about.
- **The exact correction:** the two formulas are both linear in the sequence
  length and share the 256 anchor. So
  **`--extra-sample-args base_shift=0.5,max_shift=0.69355`** makes sd.cpp's μ
  equal diffusers' μ at *every* resolution: 0.5 + 0.4 × 3840/7936 = 0.69355.
- **Mechanics:** it is a server command-line flag. It can also go per request
  inside `<sd_cpp_extra_args>` as `sample_params.extra_sample_args`. For one
  fixed size, `--sigmas 1.0,0.8572,0.6668,0.4001,0` does the same.
- **An aside, not measured:** the same arithmetic says today's 20-step *base*
  run also uses μ 1.15, where the reference is 0.694, with no terminal stretch.
  It looked fine at cfg 6, but it is not the reference schedule.

**The rest of the settings:**
- `--sampling-method euler` matches.
- `--cfg-scale 1.0`: sd.cpp then skips the unconditioned pass, because
  `use_uncond` is set only when `img_cfg != txt_cfg` (`request.cpp:303`).
- Negative prompts are ignored.
- The two community GGUF cards recommend ComfyUI's "simple" scheduler. That
  is a ComfyUI setting and does not carry over.

## 3. Expected effect on our setup (estimate, n=0)

**The speed-up is ~10× in passes, not 5×.** Today's run is 20 steps × **2
passes** (cfg 6.0 runs the conditioned and unconditioned passes), which is 40
transformer passes. Sampling measured 99.4–100.1 s at n=3, about 2.5 s per
pass. The turbo is 4 steps × 1 pass = **4 passes**.

| phase | today (b, n=3) | turbo, estimated |
|---|---:|---:|
| encode (GPU) | 2.2–2.3 s | same, ~2.2 s |
| sampling | 99.4–100.1 s | **~10 s** (4 × ~2.5 s), if the per-pass staging cost holds under `--offload-to-cpu` |
| VAE decode | 6.8–7.1 s. This includes a failed untiled attempt and then the tiled retry. | same, ~7 s. It becomes about a third of the wall. |
| sd-server end to end | 105.8–107.6 s (n=2) | **~20–25 s** |

**Other effects:**
- **Peak VRAM:** the weights are the same size class and are staged under the
  same `--max-vram 6` budget. Peak Δ should stay near +6.3 GB. Unmeasured.
- **VAE decode:** with sampling this short, `--vae-tiling` from the start
  (skipping the failed untiled attempt) could be worth a separate arm.
- **Quality:** expect some loss against the base. By the card's own words the
  main risk is garbled small or long text. Our `sign` prompt ("YAMADORI
  BONSAI") is the direct probe for that.
- **Nothing published compares them,** and quantisation adds a second unknown
  on top.

## 4. The text encoder and the quant

**The Heretic encoder.**
- The student never saw it. Viggle distilled against the stock
  Qwen3-VL-8B-Instruct in bf16, loaded from the base repo.
- The base never saw Heretic or a Q4_K_M encoder either, and it renders
  coherent images and correct text with them (IMAGEGEN.md). So the turbo is
  in the same position the base is in today.
- **An open question:** a few-step DMD student has less room to correct its
  course than a 20-step CFG sampler. Whether conditioning drift from the
  ablated directions (KL 0.022, per the Heretic card) plus Q4_K_M costs more
  at 4 steps is **not known**. The measurement below would show it only as an
  unexplained gap.
- **A clean attribution would need a stock-encoder arm.** That means another
  download of about 5 GB. It is left as optional.

**Q5_K_M.**
- **Full-student GGUF:** it is quantised from the student's own bf16 weights,
  the same relationship our base GGUF has to the base. realrebelai's warning
  that GGUF does not reproduce the bf16 output exactly applies to every GGUF,
  ours included.
- **LoRA:** see "A LoRA on a quantised base" in §2. It is fine in principle
  under `at_runtime`, blocked in practice by the fused-MLP key mismatch, and
  unmeasured.

## 5. Licence

**The same agreement as the base, with nothing added by Viggle.**
- The Qwen Research License Agreement, effective 2026-09-20. Viggle's
  `LICENSE` has the same git blob as `Qwen/Qwen-Image-2.1`'s (`13ae08d5`).
- Viggle's `NOTICE` records its modifications (the transformer, the added
  LoRA, and `shift_terminal` set to null) and adds no terms of its own.
- The Space and the model card both declare `license: other`, `qwen-research`.

**The terms that bind:**
- **Grant:** use, reproduce, modify and distribute "FOR NON-COMMERCIAL
  PURPOSES ONLY". It defines non-commercial as "for research or evaluation
  purposes only".
- **Commercial use:** needs a separate licence from Qwen.
- **Redistribution:** ship the agreement, mark modifications, and keep the
  Qwen copyright notice.
- **Attribution:** a model trained or improved with the Materials *or their
  outputs* that is then distributed must say "Built with Qwen" or "Improved
  using Qwen".
- **Termination on breach:** the Materials must then be deleted.

**What it means here:**
- **This is not new exposure.** IMAGEGEN.md already records the base as
  Qwen Research License, not Apache.
- **Whether serving images to Yamadori's users counts as "research or
  evaluation" is the operator's call.** The agreement does not define
  personal or internal use.
- **The community GGUFs** say they inherit the upstream terms.
- **The Heretic encoder** stays Apache-2.0.

## 6. Proposed measurement plan (awaiting operator approval)

**The "PartiPrompts seeded sample" does not exist in this repo.**
- `bench/imagegen/results.jsonl` (20 rows) and `measure.py` hold only three
  prompts, `fox`, `sign` and `crane`, at seed 42, plus two sd-server smoke
  images at seeds 7 and 8.
- `grep -i parti` finds no image benchmark.
- So this plan has two parts:
  - **Part A:** a repeat of the existing prompts and seeds, directly
    comparable to the saved PNGs.
  - **Part B:** a new seeded PartiPrompts sample. It must be run on **both**
    the base and the turbo, because no base numbers exist for it.

**Preconditions** (AGENTS.md rules 4 and 5, and IMAGEGEN.md's incidents):
- CUDA1 has no other consumer. That means the benchmarks now running must be
  finished or queued behind this through `bench/queue_runner.py`.
- `measure.py`'s free-memory floor (1024 MiB) and 88 C kill-switch stay on.
- Use `-t 8`.

**Step 0: download and verify.** This is the only download.

| arm | file | bytes |
|---|---|---:|
| **T** (primary) | `Abiray/…/qwen_image_2.1_turbo_Q5_K_M.gguf` | 5,007,399,712 |
| T′ (optional, second quantiser) | `realrebelai/…/Qwen-Image-2.1-viggle-turbo-Q5_K_M.gguf` | 5,959,105,632 |
| (alternative to T and T′: own provenance) | Viggle `transformer/diffusion_pytorch_model.safetensors`, then local `sd-cli -M convert --type q5_K` (CPU) | 14,230,284,584 |

- Check each file's byte size and sha256 against the tree API, as was done for
  the base.
- Read the GGUF header on the CPU to confirm the Qwen-Image-2.1 tensor names.
- **The LoRA is not in the plan.** It needs the sd.cpp key-mapping patch
  first (§2).

**Step 1: harness change** (a code change, for approval with the plan).
- `measure.py` hard-codes the diffusion model, `--steps 20` and
  `--cfg-scale 6.0` in `COMMON`. Parametrise those three so that a turbo config
  adds:
  `--diffusion-model <T> --steps 4 --cfg-scale 1.0 --sampling-method euler --extra-sample-args base_shift=0.5,max_shift=0.69355`
  on top of the unchanged `b_offload_budget6` flags
  (`--backend cuda0 --offload-to-cpu --max-vram 6 --diffusion-fa`).
- Record `steps`, `cfg` and `extra_sample_args` in each row. Today `steps` is
  written as a literal 20.

**Part A: the direct side-by-side.** Use the existing prompts, seeds and size.

| arm | config | prompts × seed | n |
|---|---|---|---:|
| B (base, re-run for same-day drift) | `b_offload_budget6`, 20 steps, cfg 6 | fox, sign, crane × 42 | 3 |
| T | turbo, 4 steps, cfg 1, corrected shift | fox, sign, crane × 42 | 3 |
| T-def (ablation: does the shift fix matter?) | as T, but without `--extra-sample-args` | fox, sign, crane × 42 | 3 |
| T-1344 | T at 1344×1344 (the size ceiling) | sign × 42 | 1 |

- **Repeat T once in full,** to check that the time holds on a second run
  (PROTOCOL: a single run is not a result).
- **Record per row:** `wall_s`, `encode_s`, `sample_s`, `sec_per_step`,
  `decode_s`, `delta_peak_mib`, `min_free_mib` and `peak_temp_c`. These are
  the columns that exist today.

**Part B: the seeded PartiPrompts sample** (new; both arms).
- **Source:** `PartiPrompts.tsv` from `google-research/parti` (1,632 prompts,
  with categories). It is a small text file, and its download is part of this
  approval.
- **Sample:** 32 prompts drawn with `random.Random(20260923)`, stratified by
  category. Force-include every prompt in the text-rendering categories. Store
  the chosen ids in `bench/imagegen/parti_sample.json`, so the sample is fixed
  before either arm runs.
- **Seeds:** 42 and 7 per prompt, so 64 images per arm.
- **Arms:** B and T only.
- **Cost:** B is about 64 × 110 s, roughly 2 hours of CUDA1. T is roughly
  25 minutes by the estimate.

**How quality is judged.** There is no automatic metric; the card claims none,
and Laya is not a quality judge.

- **Blind pairwise:** show the B and T images for each prompt and seed in
  random left/right order. The operator picks better, same or worse, and marks
  prompt adherence.
- **Text:** for every sign or text prompt, record whether the rendered string
  is exactly correct. It is pass or fail per image, reported as k/n for each
  arm.
- **Artifacts:** record burn, over-saturation, ghosting, duplicated subjects
  and blur/softness.
- **Reporting:** T is "comparable" only if its text-correctness k/n is within
  the other arm's range across both seeds, and the blind judgement is not
  mostly "worse". Label every number with its n.

**The decision this feeds.** Should `imagegen` switch its denoiser to T?
- **In config.yaml:** swap `--diffusion-model`, and set `--steps 4 --cfg-scale
  1.0 --extra-sample-args base_shift=0.5,max_shift=0.69355`.
- **In the proxy:** `mcp/images.py` `DEFAULT_STEPS = 20` must become 4. The
  proxy sends `steps` explicitly, so the server's `--steps` alone would not
  change it.
- **Alternatively, keep both.** Run T as the default and B for a "quality"
  path.
- **None of this is proposed until the plan has run.**

## Sources

- Space: https://huggingface.co/spaces/Viggle/Qwen-Image-2.1-viggle-turbo (`app.py`; `/api/spaces/…`: Gradio 5.50, zero-a10g, license `qwen-research`)
- Model: https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo (README, LICENSE, NOTICE, `transformer/config.json`, `scheduler/scheduler_config.json`, `peft/adapter_config.json`, tree API), discussions https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo/discussions
- Base: https://huggingface.co/Qwen/Qwen-Image-2.1 (README, LICENSE, `scheduler/scheduler_config.json`)
- Community: https://huggingface.co/Abiray/Qwen-Image-2.1-viggle-4-steps-turbo-GGUF, https://huggingface.co/realrebelai/Viggle_Qwen-Image-2.1-Turbo_GGUFs, https://huggingface.co/t8star/Qwen-Image-2.1-viggle-turbo-4step-r64-comfy
- diffusers token count at 1024²: https://github.com/huggingface/diffusers/issues/14824
- stable-diffusion.cpp `c92d73c` (local): `docs/lora.md`, `docs/qwen_image_2.1.md`, `docs/quantization_and_gguf.md`, `examples/server/api.md`, `examples/server/routes_sdapi.cpp`, `examples/common/common.cpp` (`--extra-sample-args`, `--sigmas`, `--lora-apply-mode`), `src/pipeline/request.cpp`, `src/runtime/denoiser.hpp` (`FluxScheduler`), `src/model/diffusion/qwen_image_2_1.hpp`, `src/name_conversion.cpp`, `src/model/adapter/lora.hpp`, `src/pipeline/diffusion_engine.cpp`
- Our GGUF's tensor directory: `C:/Users/jwals/textgen/user_data/models/qwen-image-2.1/qwen-image-2.1-Q5_K_M.gguf`. It has 265 tensors, fused `img_mlp.gate_up`, and prefix `model.diffusion_model.`.
