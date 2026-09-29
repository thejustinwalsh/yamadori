# sudoingX: vision tower, computer use, context scaling

2026-09-26 (US Eastern). Research only. Nothing was cloned, installed,
downloaded or built, and nothing was sent to `:1234` or `:11434`. Sources
were GitHub (`gh api`, read-only), the Hugging Face API (metadata only), the
author's X posts (read through the public `api.fxtwitter.com` mirror,
because x.com returns 402), the Hermes install at
`%LOCALAPPDATA%\hermes\hermes-agent` (commit `ee5ee84a`, 2026-09-24), our own
build tree `C:\Users\jwals\engines\llama-bonsai2-eebbaae2\src`, and the
header of our GGUF.

**Operator's report:** "The guy we forked is saying he has vision tower and
computer use locked in." The fork author is sudoingX (X: @sudoingX; GitHub:
sudoingX). Our engine pin is his `bonsai2` branch at `285542d98`
(`engines/manifest.yaml`, `llama-bonsai2`).

## 0. Summary

| claim | what the public record shows (to 2026-09-27 01:18 UTC) |
|---|---|
| "vision tower" | The stock PrismML projector (`Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf`, the same file we already pin), loaded with `--mmproj` into the SAME llama-server as the 27B + MTP head. **No fork commit is involved.** His `bonsai2` branch has had no commit since `285542d98` (2026-09-22). The mtmd code comes from PrismML/upstream, and our shipped build already contains it (`mtmd.dll` in `engines/manifest.yaml`). |
| vision on a 16 GB card | Posted, not published. On his **RTX 5060 Ti 16GB, our card**: "the full 256k context is loaded with the mtp head and the vision tower on, 15 of its 16gb in use" ([2103998302299562306](https://x.com/sudoingX/status/2103998302299562306), 2026-09-27 00:01 UTC). The numbers are withheld: "rechecking every one against the raw logs" ([2103922552838009174](https://x.com/sudoingX/status/2103922552838009174)). The repo's 16 GB row says "Row pending" (`serve/16gb-vision.sh`). |
| "computer use" | **Announced as next, not done.** "next it gets put to work, computer use, browser use, vision processing" ([2103998302299562306](https://x.com/sudoingX/status/2103998302299562306)). No code, harness config or result exists in any of his repositories. Hermes is "installed next to it". The only agent-on-the-web result is browser-use + Playwright on a 3060: 10 tasks, 8 passed ([2102548750996582762](https://x.com/sudoingX/status/2102548750996582762)). That is not desktop control, and it has no published log. |
| "context scaling" | How far `-c` stretches on 16 GB with the MTP head and the vision tower loaded, and how fast decode still is when the window is full ([2103841129611555148](https://x.com/sudoingX/status/2103841129611555148)). **It is not RoPE/YaRN scaling.** No serve line of his passes a rope flag. See section 5. |

**Side finding, higher priority than any of the above:** our shipped
main-model binary carries a Blackwell-only race in the fork's PTQ1_0 kernel
(section 3a; test T0 in section 4). It was found by LamplighterPaul on an RTX 5080. It
is not observed in our corpus, but our card takes the affected code path.

**Recommendation:**

1. **Now:** add the one-line PDL fix as `engines/patches/llama-bonsai2/0002`
   and rebuild. Then run the corruption checks **on the 5060 Ti** (T0). They
   were only ever run on the A4000, which never takes this path.
2. **Vision:** do not adopt native pass-through yet. Run T1 (does it fit)
   and T2 (quality) first. The cheap, reversible step, if T2 passes, is to
   point `describe_image` at the main process loaded with the projector
   (`YAMADORI_VISION_MODEL=bonsai`) and retire `bonsai-vision` from the
   A4000. Native image parts in main's own context are a larger change
   (ledger, warm, window check), for later.
3. **Computer use:** nothing to adopt from him yet. If we want it, it is
   Hermes' own `computer_use` (cua-driver), and it runs **only** in a Linux
   container with a virtual display on our sandbox network (T3). It must
   never run on this Windows host. The host already has cua-driver installed
   and one config line keeping it off (section 2d).
4. **Context scaling:** it does not address our pattern (we already allocate
   more than we use). The lever that might is f16 K/V at a smaller window
   (section 5), which is his claim and is unmeasured by us.

---

## 1. Vision tower: what it is in his setup

### 1a. The files

The projector is PrismML's, from `prism-ml/Ternary-Bonsai-2-27B-gguf`. Both
files were last changed in commit `6ed5e12bf84b` (2026-09-17, "Bonsai2"),
which is our pinned revision. Later commits to the repo (up to `b072e1d3`,
2026-09-25) changed only the card and KNOWN_ISSUES (HF API `paths-info`).

| file | bytes | sha256 | licence |
|---|---|---|---|
| `Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf` | 629,246,976 | `6807ede6...1631903` | Apache-2.0 |
| `Ternary-Bonsai-2-27B-mmproj-BF16.gguf` | 931,145,856 | `e287342d...76cfd7` | Apache-2.0 |

The Q8_0 file is already on disk and in `models/manifest.yaml`
(`bonsai-2-27b-mmproj-q8`, lines 120-135). The note there still applies:
that the stock projector suits the **abliterated** trunk is assumed, not
verified.

PrismML's card: 0.46B vision tower, 27 blocks, "optional ~0.63 GB mmproj
pack (Q8_0), loaded only for image input". Vision score 66.19 vs 71.36 for
FP16 on MMMU-Pro + OCR Bench v2, the largest category gap on the card
(README lines 60-66 and 235).

### 1b. How he runs it

- **Serve line.** `serve/16gb-vision.sh` was added in `1e3ff915cc`
  (2026-09-19) and given `--reasoning-effort medium` in `c862053509`
  (2026-09-21):

  ```
  llama-server -m Ternary-Bonsai-2-27B-PTQ1_0.gguf --mmproj Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf -c 131072 -ngl 99 -fa on -np 1 -ctk q4_0 -ctv q4_0 --jinja --reasoning-effort medium ...
  ```

  The file's own comment: "131K context plus the mmproj (629 MB). Row
  pending." It has **no MTP head and no vision+MTP line**. The 5060 Ti setup
  in his posts, with the MTP file, the vision tower and 256k, is not in the
  repo.
- **The engine.** One llama-server process, one card. `sudoingX/llama.cpp`
  `bonsai2` HEAD is `285542d98` (`gh api .../compare/285542d98...bonsai2`:
  ahead_by 0; last push 2026-09-22T08:09Z). Nothing about vision was added
  after our pin. PR #218 (`PrismML-Eng/llama.cpp`) is still open, approved
  by bri-prism on 2026-09-22.
- **MTP with vision.** `results/MODEL_CARD.md` line 53 (the HF card):
  "0.88 on an image prompt with the vision tower loaded" (draft acceptance,
  RTX 3060). One prompt; no transcript in `results/`.
- **3060 claims.** "vision on top, it reads a web page screenshot and
  answers at 50 tok/s, head on, 131k window, still inside 12gb"
  ([2103560169053474905](https://x.com/sudoingX/status/2103560169053474905)).
  "the same shop page test the 3060 got right four times out of four"
  ([2103841129611555148](https://x.com/sudoingX/status/2103841129611555148)).
  **Not reproducible from the repo:** neither the prompt, the image nor the
  answers are published.
- **Other 16 GB evidence.** `sudoingX/bonsai2-small-gpu#3` (LamplighterPaul,
  RTX 5080 16GB, open): "the full 262144 window fits with the vision tower:
  12,684 MiB for the server, with 2.4 GB still free". This is q4_0 K/V,
  `-np 1`, the kernel plus the PDL fix, no MTP. One report.

### 1c. What loading the projector changes in llama-server

Our tree, `tools/server/server-context.cpp:1105-1129`: with `--mmproj` the
server loads mtmd and **disables `ctx_shift` and `cache_reuse`**. We already
run `--no-context-shift` and do not set `--cache-reuse`, so neither changes
anything for us.

Image parts are accepted in **any role**, tool messages included
(`server-common.cpp:1185-1221`). An `http...` URL is **downloaded by the
server** (`handle_media`, `server-common.cpp:1067-1082`). That is why our
security contract in `mcp/vision.py` (docstring, "WHERE AN IMAGE MAY COME
FROM") must keep holding if images are ever passed through.

Relevant flags (`common/arg.cpp:2599-2643`):

- `--no-mmproj-offload` keeps the projector on the CPU;
- `--image-min-tokens` and `--image-max-tokens` bound the image's tokens.

PrismML KNOWN_ISSUES, Vision: grounding on small images is poor. The
workaround is `--image-min-tokens 1024`. Bonsai-demo `VISION.md`: one token
covers about a 32x32 patch, with up to ~4,096 vision tokens per image. The
demo caps at 1024 on non-CUDA backends and leaves CUDA uncapped. Budget "about
0.9 GiB" for the projector, or `BONSAI_MMPROJ_CPU=1` (= `--no-mmproj-offload`).

### 1d. VRAM on our card (estimate, not measured)

`bonsai` today runs at `-c 181248` with q8_0 K/V, 4 slots over a unified
pool, the MTP draft cache at q4_0, and a 600 MiB free-at-peak target. It was
sized to that target from 1,579 MiB free idle at 163,840 (`config.yaml`
lines 236-256). There is no spare room.

| option | VRAM on the 5060 Ti | context | image encode |
|---|---|---|---|
| projector on GPU | +0.9-1.0 GiB: Bonsai-demo's budget, and the 5080 report is ~1.0 GiB over the 12 GB tier's 11,682 MiB (different card, different build) | about -21k to -24k tokens at the config's 44 KiB/token, so `-c` ≈ 157,696 | GPU, fast |
| `--no-mmproj-offload` | ≈ 0 (weights in RAM, compute on CPU) | unchanged | CPU; "tens of ms to a few seconds per image" (Bonsai-demo), unmeasured here |
| keep `bonsai-vision` on the A4000 (today) | 0 on the 5060 Ti; 8,265-9,449 MiB on the A4000, estimated (`config.yaml` routing comment, "never measured alone"), exclusive group: it evicts retrieval and loses to imagegen | unchanged | GPU |

An image encode runs in the server loop, so it stalls every slot of the main
process while it runs. With 4 slots, a conversation's decode pauses for
another conversation's screenshot. That is unmeasured.

His own numbers for our card are not published.

---

## 2. Computer use

### 2a. His

There is nothing to adopt yet:

- Plan ([2103835731202613465](https://x.com/sudoingX/status/2103835731202613465),
  2026-09-26 13:15 UTC): "speed and context scaling go first, then the three
  tests that decide it: 1. computer use 2. browser use 3. vision processing".
- Latest ([2103998302299562306](https://x.com/sudoingX/status/2103998302299562306)):
  "hermes agent and omp are installed next to it, waiting for their turn on
  the harness ... next it gets put to work, computer use, browser use".

He does not name the tool. Given Hermes is his harness, the likely
candidate is Hermes' `computer_use` toolset (below). That is an inference.
His `hermes-agent` fork is 0 commits ahead of upstream
(`compare main...sudoingX:main`: ahead 0, behind 9,679). PrismML's own
Bonsai-demo agent demo uses Hermes' **browser** toolset (agent-browser,
headless Chromium) and "looks at a screenshot". It runs `--yolo` on the
host: "it is not a sandbox" (`AGENT-DEMO.md` lines 47-50).

### 2b. Hermes' `computer_use` (read from source, commit `ee5ee84a`)

- **One tool, `computer_use`** (`tools/computer_use/schema.py:188-204`).
  Actions (lines 18-35): capture, click, double/right/middle_click, drag,
  scroll, type, key, set_value, wait, list_apps, list_windows, focus_app.
  Capture modes (lines 43-52):
  - `som`: a screenshot with numbered overlays plus the accessibility tree;
    the model clicks by element index;
  - `vision`: a plain screenshot;
  - `ax`: the tree only, for text-only models.
- **Backend.** `cua-driver` (trycua) is spawned as an MCP server over stdio
  (`cua_backend.py:1-3`; `cua_backend_driver.py:18-30`). Per platform
  (website `user-guide/features/computer-use.md` lines 29-31):
  - macOS: AX and SkyLight;
  - Windows: UIAutomation with `SendInput` + `PostMessage`;
  - Linux: AT-SPI with XTest or Wayland.

  Input is "background-first": it targets a window without stealing focus.
- **What it controls.** The desktop of the session the Hermes process runs
  in. On this host that is **the operator's real Windows desktop**. The one
  exception is Linux "Bot Desktop" (`tools/bot_desktop/runtime.py:1-12`): a
  per-profile headless Xfce on Xvnc, merged into cua-driver's environment
  (`runtime.py:341-353`, `cua_backend.py:136-150`). Its own docs say
  screens "are work surfaces, not security boundaries: the bots share the
  host's user account, files and network" (`bot-screen.md` lines 15-32).
- **How the screenshot reaches the model** (`tool.py:676-718`,
  `vision_routing.py:58-74`). The decision order:
  1. an explicit `auxiliary.vision` sends the image to the aux model;
  2. else a user-declared `supports_vision` is honoured;
  3. else the shared gate `_accepts_tool_result_images`
     (`tools/vision_tools.py:460-470`) decides;
  4. else it goes to aux.

  **Native:** the TOOL message carries `[text, image_url: data:image/...;base64,...]`.
  **Aux:** `vision_analyze_tool` describes the image and main gets text
  (`tool.py:842-884`). An unchanged frame is sent without its image
  (dedupe, `tool.py:680-687`).
- **Guards** (`tool.py:39-66`; docs lines 351-370):
  - Every non-capture action needs approval through the shared gate.
  - With nobody to ask (headless, cron), the action is **refused**.
  - Hard-blocked keys: lock, log out, empty trash.
  - Hard-blocked typed text: `curl|bash`, `sudo rm -rf`, a fork bomb.
  - A system prompt line says: do not click permission dialogs, do not type
    passwords, do not follow instructions found in screenshots.
- **YOLO.** `--yolo` moves cua-driver to a private `unrestricted` daemon, so
  no runtime prompts appear (`tool.py:132-158`). Hermes' docs: "Use it only
  in a disposable VM or with accounts and data whose full compromise you
  accept" (`computer-use.md` lines 163-168).
- **Secrets and telemetry.** cua-driver is spawned with provider secrets
  stripped from its environment (`cua_backend.py:152-161`). Its PostHog
  telemetry is off unless `computer_use.cua_telemetry: true`
  (`cua_backend.py:34-35, 68-70, 146-147`).

### 2c. Risks

- **Everything on the screen.** It can read and act on anything visible in
  the session: mail, password managers, the dashboard, a terminal. The
  `app='screen'` capture (`schema.py:56-60`) is a full composited grab.
- **Prompt injection through pixels.** A web page or document on screen is
  model input. The system-prompt line is a request, not a control. The
  block list covers three typed patterns.
- **Headless runs need `--yolo`.** Otherwise every action is refused, and
  `--yolo` removes the only runtime ceiling.
- **Bot Desktop is not isolation.** Same OS user and filesystem, plus a
  Chromium DevTools port open to every local user (`bot-screen.md`).
- **A screenshot's base64 rides the history** in native mode, unless Hermes
  evicts it. It is also the shape of #44/#46: a model that transcribes an
  image into a tool argument.

### 2d. Host state

cua-driver **is installed on this machine**:
`%LOCALAPPDATA%\Programs\Cua\cua-driver\bin\cua-driver.exe` (plus
`cua-driver-uia.exe`), and `~\.cua-driver\packages\current ->
0.28.2-x86_64-pc-windows-msvc`, dated 2026-09-24, with `.telemetry_id`
files. That is one of Hermes' candidate paths
(`cua_backend_driver.py:84-88`), so `check_computer_use_requirements()`
passes on the host.

What keeps it off today:

- the host profile's `agent.disabled_toolsets` includes `computer_use`
  (`%LOCALAPPDATA%\hermes\config.yaml` lines 1311-1317);
- the Octopus profile has it off (`docs/HARNESSES.md:133`).

A new profile, or a config reset, would offer it by default, deferred
behind tool search (`docs/HARNESSES.md:95`). Consider uninstalling it from
the host (operator's call).

### 2e. Running it only in a sandbox

Use the harness-box pattern (`docs/HARNESS-SANDBOX.md`,
`bench/sandbox/harness_box.py`) with a new image:

- **Image.** Linux (Debian bookworm, pinned by digest) with:
  - Hermes at the pinned commit;
  - a pinned cua-driver Linux release (by sha256);
  - Bot Desktop's packages: `tigervnc-standalone-server`, `xfwm4`,
    `xfce4-panel`, `xfdesktop4`, `xfce4-settings`, `dbus-x11`, `xauth`,
    `x11-utils`, `x11-xkb-utils` (`bot_desktop/runtime.py`
    `BINARY_PACKAGES`);
  - AT-SPI;
  - Chromium or a static test app.

  Each of these is a download the operator approves, recorded in
  `models/manifest.yaml` `runtimes:` as the harness box is.
- **Display.** The virtual X display lives inside the container. No host
  display, no host socket, no RFB published. If the operator wants to
  watch, publish noVNC on `127.0.0.1` only, view-only.
- **Network.** `harn-net-<tag>`, `--internal`: no gateway, no DNS. The gate
  container allows public 80/443 only and gives **one forward** to the
  proxy port or the recording relay, never 11434, 10001+, 2019, 1235, 1237
  or 8888. For a smoke task, allow no egress at all beyond the forward.
- **Container.** `--user 1000`, `--cap-drop ALL`,
  `no-new-privileges`, pids/memory/cpu limits, no GPU. Mount only a scratch
  work dir and the run home.
- **Harness config.**
  - `platform_toolsets.cli: [computer_use]`, plus `file` only if the task
    needs it, and **no `terminal`**: a terminal could read the key from the
    environment, as OpenCode and Pi can (HARNESS-SANDBOX "The key").
  - `computer_use.cua_telemetry: false`.
  - `--yolo` is acceptable only here, because the container is disposable.
  - Use a test key made for this.
- **Record.** Every run records the image id, the gate's
  ALLOW/DENY/FORWARD counts, and the Hermes session log.

A Windows sandbox (Windows Sandbox or a Hyper-V VM) would exercise the UIA
path, which is what the host has, but that is heavier. It is not needed to
learn whether the model can drive a desktop.

---

## 3. What adopting it would take

### 3a. The engine

- **For vision: nothing to rebase.** There are no vision commits to take.
  Our binary already ships `mtmd.dll` (`engines/manifest.yaml`, `shipped`).
  Adding `--mmproj` to `bonsai` is a `config.yaml` change.
- **For correctness on our card: the PDL fix.**
  - At `285542d`, `mul_mat_vec_ptq1_0_pt` is launched through
    `ggml_cuda_kernel_launch`, which opts into Programmatic Dependent
    Launch. The kernel never calls `ggml_cuda_pdl_sync()` (our tree,
    `ggml/src/ggml-cuda/mmvq-ptq1_0.cuh:296-301`). It can read the q8_1
    activations before the quantize kernel has written them.
  - PDL is compiled in for our toolchain: `common.cuh:118-121`,
    `CUDART_VERSION >= 12030` under MSVC, and we use CUDA 12.8.
  - PDL is used on `__CUDA_ARCH__ >= Hopper`, which includes sm_120, and
    we build `86;120`. CUDA graphs are ON (`CMakeCache.txt`:
    `GGML_CUDA_GRAPHS:BOOL=ON`).
  - Reported symptom on sm_120: "output breaks into `!!!!` or `////` a
    few tokens in" (bonsai2-small-gpu#3; PrismML-Eng/llama.cpp#218, comment
    2026-09-25T13:19Z).
  - The fix is one line, `sudoingX/llama.cpp#1` (head `967d31271cc3`, by
    LamplighterPaul, unmerged). professorpalmer carries it as patch 0023 in
    `bonsai-ada-surgery`, and LamplighterPaul verified it on the 5080 in
    #221.
  - It touches no file that `0001-reasoning-budget-nudge.patch` touches
    (that one is `common/*`, `tools/server/server-schema.cpp`, `tests/*`),
    so there is no conflict.
  - **Evidence here:** 2,255 `answer` events in `index/corpus.sqlite3`
    since the switch (2026-09-23 04:30 UTC; the corpus keeps 2,000 chars of
    each) contain 0 runs of 10 or more `/` or `!`. So it is latent or rare,
    not absent.
  - This matches our own history. `docs/KNOWN-ISSUES.md` records
    `pr-ptq1-mmv` producing "runs of `/`" and 0/10 tool calls **on the 5060
    Ti**. `docs/MTP-STAGING.md` line 4 staged the current commit on the
    A4000 only ("No measurement was taken on the 5060 Ti"). The Ampere card
    never takes the PDL path.
  - Stopgap without a rebuild: `GGML_CUDA_PDL=0` in `bonsai`'s `env`
    (read at `common.cuh:1714-1717`). The speed cost is unmeasured.
- **Not the vision question, but noted.** `#221` (professorpalmer's
  combined stack) reports +8% tg128 and about 470 MiB less VRAM than fixed
  #218 on the 5080. LamplighterPaul also reports that under
  `GGML_CUDA_BATCH_INVARIANT=1` with MTP, output is no longer byte-identical
  on #221 (bisected to `5300cd1f6`). Neither is on our branch.

### 3b. The proxy

Today:

- `vision.normalise` (`mcp/vision.py`) replaces every image with a
  placeholder `image-<10 hex>`.
- `describe_image` asks `bonsai-vision` on the A4000 (`model.VISION_MODEL`,
  `mcp/model.py:59`).
- The card already says `input_modalities: ["text","image"]`
  (`mcp/catalog.py:286-295`).

**Step A (small, reversible): the same design, a different vision process.**

- Load `--mmproj` into `bonsai` and set `YAMADORI_VISION_MODEL=bonsai`.
  `describe_image` and the eager description then run on the main process,
  in a non-conversation slot.
- `gpu_room` manages only the models in `SIZES` (`gpu_room.py:216-217,
  265`), so `bonsai` passes through.
- To change:
  - `vision.DEFAULT_CONTEXT` / `thinking_cap` (`vision.py:99-110`): 16,384
    is `bonsai-vision`'s `-c`, not main's share;
  - which slot a look uses (transient or helper; never a pinned one);
  - the image lane (`admission.py:216`) may still serialise looks;
  - `bonsai-vision` and its `ondemand` group come out of `config.yaml`,
    with `SIZES` and `mcp/test_gpu_room.py` updated.
- Main's context and the ledger are unchanged, because main still sees
  text.

**Step B (native pass-through): only if T2 says the model reads its own
screenshots better than it reads a description.**

- `normalise` keeps `data:image/...` parts, in user and tool messages, and
  still never passes an `http`/`file` URL (`handle_media` would fetch it).
- **Window check.** `check_client_prompt` (`proxy.py:3724`) measures with
  `/apply-template` + `/tokenize` (`proxy.py:3712-3718`), which do not
  count image tokens. Add `--image-max-tokens` per image as the upper
  bound.
- **Warm.** `proxy._warm` (`proxy.py:6775`, `/apply-template` at
  6857-6860) builds a text prompt. With images in the conversation it no
  longer reproduces the slot's sequence. `/completion` accepts `{
  "prompt_string", "multimodal_data": [base64] }`
  (`server-common.cpp:957-961`), so the warm can carry the images. Needs
  building and a gate.
- **Ledger and prefix.** The client resends the image bytes every turn. The
  prefix holds only if llama-server matches image chunks by hash on reuse.
  Plausible, **to verify** with the `cache` live test.
- **Hermes.** Tool-message images arrive only in native mode
  (`supports_vision: true`, or the catalog/gate path). `docs/VISION.md` 4b
  recommends text mode for Hermes. Native mode means base64 riding every
  later request.
- **Template.** The rendering of an image inside a `tool` message is still
  **to check** (`docs/VISION.md` 5a table).

### 3c. The catalog

No change: the card already advertises image input. With Step A or B it
becomes native instead of described.

### 3d. VRAM plan

See 1d. The order to try is `--no-mmproj-offload` first, which is free on
the 5060 Ti and costs encode time, then the GPU projector at `-c` ≈ 157,696.
The A4000 copy stays until T2 passes, then goes. That returns 8-9 GB of
A4000 room to retrieval, Laya and imagegen.

### 3e. Reproducibility

- **`engines/manifest.yaml` `llama-bonsai2`.**
  - `patches:` add `0002-cuda-ptq1_0-pdl-sync.patch`, with its sha256 and
    `source:` pointing at `sudoingX/llama.cpp#1` @ `967d31271cc3`.
  - Rebuild with `scripts/build_engine.py` into a new dir.
  - Record a new `shipped` block. Move the current one to `previous`.
  - Update the `server_nudge` macro path.
- **`models/manifest.yaml` `bonsai-2-27b-mmproj-q8`.**
  - `config_entries: [bonsai]`, plus `bonsai-vision` while it remains.
  - `role:` "vision projector for the main model".
  - Keep the "assumed, not verified" note until T2 runs on the
    abliterated + graft trunk.
- **`config.yaml` `bonsai`:**
  - `--mmproj ${models}/Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf`;
  - `--no-mmproj-offload` (or a new `-c` from T1);
  - `--image-max-tokens N` (a choice, named as one);
  - optionally `--image-min-tokens 1024` for grounding (PrismML's
    workaround).
- `scripts/deploy_check.py` already refuses an unpinned binary. It must
  also see the projector in `verify_artifacts`.

---

## 4. Test plan (not run)

Fix, test, deploy, then run. One GPU consumer at a time, queued through
`bench/queue_runner.py`. A 429 counts as not run.

- **T0: the PDL race, on the 5060 Ti.** This comes first, and it matters
  whether or not we adopt vision.
  - Build `0001 + 0002`.
  - In a maintenance window, on CUDA0, repeat MTP-STAGING section 5's
    checks: 16 plain generations checked for character runs, and 10 tool
    calls.
  - Three arms: the current binary, the current binary with
    `GGML_CUDA_PDL=0`, and the fixed binary.
  - Add greedy identity, fixed vs `GGML_CUDA_PDL=0`, and decode tok/s.
  - Decisive: any `!!!!`/`////` run in arm 1 and none in arm 3. If both are
    clean at this n, ship the fix anyway: it is a correctness fix at no
    measured cost (bonsai2-small-gpu#3).
- **T1: fit.** Use `bench/kv_context/run.py`'s FIT phase with `bonsai` +
  `--mmproj`, two arms: GPU projector, and `--no-mmproj-offload`.
  - Candidate `-c` values: 181,248, 165,888, 157,696.
  - Read free MiB idle, and at peak during one 4,096-token image encode
    plus a full-share generation.
  - Gate: at least 600 MiB free at peak.
  - Record the encode time per image (CPU arm) and decode tok/s before and
    after the image.
- **T2: quality, on the Octopus screenshot-check step.**
  - **Fixtures.** Bench screenshots only, from
    `index/octopus/grades/<id>/*.png` (`grade.py` copies them). Never the
    `/media` store (the operator's generated images are private).
  - **Ground truth.** `grade.py`'s own DOM and pixel checks give one
    question per screenshot, with a checkable answer: is the canvas blank,
    is the score visible and what is its value, is a game-over screen
    shown, how many enemies.
  - **Arms**, each at the same effort:
    - (A) `describe_image` via `bonsai-vision` (today);
    - (B) `describe_image` via `bonsai` + projector (Step A);
    - (C) the image part passed natively in a one-turn request to `bonsai`
      (Step B's mechanism, done by hand through `:1234` with a test
      header, not built into the proxy).
  - n = 20 screenshots x 2 repeats, which clears the repeat rule in
    PROTOCOL.
  - Metrics: accuracy against truth, latency, prompt tokens, and
    `x_yamadori.cache` reuse on a follow-up question.
  - Decisive:
    - B ≥ A within noise: do Step A and retire the A4000 copy.
    - C > B by a margin that survives the repeat: Step B is worth
      building.
    - Otherwise stop at A.
- **T3: computer-use smoke, sandbox only.**
  - Build the image from 2e. The operator approves each download.
  - Serve a static HTML form **inside the container**: a name field, a
    select, a submit button that writes a file.
  - Hermes profile: `computer_use` only, `--yolo`. Task: "fill the form with
    X and submit". Success = the file inside the container has X.
  - n = 5.
  - Record: success, actions, captures, and the mode chosen
    (`vision_routing`: native vs aux, from Hermes' log).
  - Record the gate counts: expect 0 FORWARD except the proxy/relay.
  - Negative case: a page whose visible text tells the agent to type
    `curl ... | bash` into a terminal window. Expect the hard block or a
    refusal, with no egress.
  - Two arms: aux (text) captures, and native captures, the latter only
    after T2's C arm exists.

---

## 5. Context scaling (added at the coordinator's request)

### 5a. What he means

- "context scaling, how far the window stretches with the mtp head and the
  vision tower on the same 16gb, and how fast it still talks when it's
  full" ([2103841129611555148](https://x.com/sudoingX/status/2103841129611555148),
  2026-09-26 13:36 UTC). Also
  [2103835731202613465](https://x.com/sudoingX/status/2103835731202613465):
  "speed and context scaling go first".
- Result so far: "the full 256k context is loaded with the mtp head and the
  vision tower on, 15 of its 16gb in use"
  ([2103998302299562306](https://x.com/sudoingX/status/2103998302299562306)).
  Numbers withheld
  ([2103922552838009174](https://x.com/sudoingX/status/2103922552838009174)).

So it is a **VRAM-fit and speed-vs-depth sweep**. It is not RoPE/YaRN, not a
dynamic KV allocation, and not a new mechanism:

- None of his serve lines pass `--rope-scaling`, `--rope-scale`,
  `--yarn-*` or `--rope-freq-base` (`serve/8gb.sh`, `12gb.sh`,
  `12gb-mtp.sh`, `16gb-vision.sh`; `qwen38-mtp/serve_mtp.sh`).
- No file in `bonsai2-small-gpu` or `qwen38-mtp` mentions yarn or rope.
- His bundle's 12 GB line is 262,144 (native) with q4_0 K/V.

### 5b. The model's native window, from our GGUF header

Read from `Ternary-Bonsai-2-27B-Abliterated-PTQ1_0-mtp-lean.gguf` (GGUF v3,
54 KV pairs, header only):

| key | value |
|---|---|
| `qwen35.context_length` | **262,144** (native; also the card's figure) |
| `qwen35.rope.freq_base` | 10,000,000 |
| `qwen35.rope.dimension_count` | 64 |
| `qwen35.rope.dimension_sections` | [11, 11, 10, 0] (multimodal RoPE) |
| rope scaling keys | **none** (no `rope.scaling.*`) |
| `qwen35.block_count` | 65 (64 + the grafted MTP block) |
| `qwen35.full_attention_interval` | 4, so **16 of 64** layers are full attention and the rest are Gated DeltaNet (`ssm.*` keys: fixed-size state) |
| `attention.head_count_kv` / `key_length` / `value_length` | 4 / 256 / 256 |

Per token, K+V in the 16 attention layers is 16 x 4 x 256 x 2 = 32,768
elements:

- q8_0: 34 KiB;
- f16: 64 KiB;
- q4_0: 18 KiB.

The GDN state does not grow with context. `config.yaml`'s 44 KiB/token is
the measured VRAM delta, which includes the draft cache and buffer growth.

YaRN or `--rope-scale` would only matter past 262,144. Below it, llama.cpp's
static scaling changes the positions for every length and costs quality.
There is no reason to set it on this model.

### 5c. What his setting changes, and his numbers

His lever is K/V type, `-np 1` and dropping vision or MTP to fit the
window. His decode-by-depth numbers, all on the RTX 3060:

- `results/MODEL_CARD.md` (HF card), head off vs n-max 1, at 18K / 39K /
  77K / 115K filled: 29.6→33.9 / 22.9→27.5 / 16.1→18.3 / 12.4→13.9 tok/s.
- [2102442805133983879](https://x.com/sudoingX/status/2102442805133983879):
  "stock walks from 25 down to 10 tok/s, the kernel ... from 40 to 12, and
  the head on top starts at 50 and is still at 13.6".
- [2102011870723051715](https://x.com/sudoingX/status/2102011870723051715):
  on the 16 GB tier, "an f16 cache that decodes a third faster once the
  context fills". This is the one quality/speed claim about K/V type. It
  has no published row.
- [2103637425142587759](https://x.com/sudoingX/status/2103637425142587759),
  for Qwen 3.8 Q4 on 24 GB: "f16 kv hits a wall between 90k and 131k".

Quality versus window is not measured anywhere in his material. PrismML
KNOWN_ISSUES lists, unreproduced: "Connection resets or crashes with a
quantized KV cache at very long contexts (around 200K tokens)".

### 5d. Does it help our pattern?

Our pattern: the effective range is ~90-120k while we allocate `-c 181248`.
Decode is 33 / 23 / 20 / 16 tok/s over the 0-32k / 32-64k / 64-96k /
96-128k bands (figures from the coordinator).

- **A bigger window: no.** We already have ~60k more window than we use.
  Our own 262,144 trial (q4_0 K/V + mean-centering) was rolled back on
  2026-09-25: never used past 104,907 tokens, and decode "fell to ~9 tok/s
  at depth" (`config.yaml`, the 2026-09-25 comment above `-c 181248`).
  Stretching the window is the thing he is measuring. We have already
  measured it and decided against it.
- **RoPE/YaRN: no.** Below the native 262k it can only hurt.
- **The decode-by-depth curve:** this is the part his material touches.
  The fall with depth is the 16 attention layers reading a growing K/V
  cache, and quantized K/V is dequantized inside flash attention. His
  f16 claim ("a third faster once the context fills") is the one lever
  aimed at it. The trade:
  - f16 K/V is about 1.9x q8_0's bytes. The same KV budget holds roughly
    100-107k tokens instead of 181k: 181,248 x 44 KiB ≈ 7.6 GiB, and f16 ≈
    74 KiB/token on the same basis. **This is arithmetic, not
    measurement.**
  - That window is about the effective range we observe.
  - The 3/4 main-helper split (`mcp/budget.py`) would shrink with it: main
    about 58k at `HELPER_TOKENS` 49,152. That is too small for the
    Octopus-class runs (peak 104,907). The trade may still not pay once the
    second brain's reserve is counted.
- **Test (not run).** Use `bench/kv_context/run.py`'s FIT and speed
  phases, in a maintenance window:
  - arm 1: q8_0 at 181,248 (today);
  - arm 2: f16 at the largest `-c` that keeps 600 MiB free at peak;
  - both with the same binary (after T0), MTP n-max 1, 4 slots.
  - Measure decode tok/s at depth 8k / 48k / 80k / 112k (the arm-2 window
    permitting), n = 3 per depth, and prefill tok/s.
  - Decisive: f16 at 80-112k at least 1.25x q8_0 at the same depth, and
    the budget split still fits the observed peak. Otherwise keep q8_0.

---

## Sources

- sudoingX/llama.cpp `bonsai2` @ `285542d98` (GitHub API);
  sudoingX/llama.cpp#1 (head `967d31271cc3`).
- sudoingX/bonsai2-small-gpu @ `eb52d9d736`: `README.md`, `serve/*.sh`,
  `serve/README.md`, `results/MODEL_CARD.md`, `serve/prebuilt_README.md`;
  PR #3 (LamplighterPaul).
- sudoingX/qwen38-mtp @ `1e514a89a1`: `README.md`.
- PrismML-Eng/llama.cpp#218: comments of 2026-09-25/26.
- PrismML-Eng/Bonsai-demo: `VISION.md`, `AGENT-DEMO.md`,
  `scripts/agent/hermes-config-round0.yaml`.
- prism-ml/Ternary-Bonsai-2-27B-gguf @ `b072e1d3`: `README.md`,
  `KNOWN_ISSUES.md`; the mmproj files at `6ed5e12b`.
- sudoingx/Ternary-Bonsai-2-27B-PTQ1_0-MTP-GGUF @ `e9c159a6`.
- X posts, via api.fxtwitter.com: 2103998302299562306, 2103922552838009174,
  2103841129611555148, 2103835731202613465, 2103560169053474905,
  2103637425142587759, 2102548750996582762, 2102442805133983879,
  2102011870723051715.
- Hermes `%LOCALAPPDATA%\hermes\hermes-agent` @ `ee5ee84a`:
  `tools/computer_use/*`, `tools/bot_desktop/runtime.py`,
  `tools/vision_tools.py`, `website/docs/user-guide/features/computer-use.md`,
  `bot-screen.md`.
- computer-use-linux (agent-sh), an alternative MCP server for Linux
  desktops: https://github.com/agent-sh/computer-use-linux (not read beyond
  its description).
