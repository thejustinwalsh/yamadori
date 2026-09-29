# Models and derived artifacts: what is pinned, and how to get it back

Every model file that `config.yaml` passes to a server, every derived artifact
our code loads as a model input, and the runtimes around them are recorded in
`models/manifest.yaml`: role, path, size, sha256 and provenance. The binaries
are recorded in `engines/manifest.yaml` (docs/ENGINES.md). The two manifests
together should let you re-obtain or rebuild exactly what runs, and prove at
deploy time that what is loaded is what is recorded.

Recorded 2026-09-25. Each HF entry was checked against the HF API at its
pinned revision on that date: 11 of 11 match.

## What runs, and where it came from

| artifact | role | status | provenance |
|---|---|---|---|
| `Ternary-Bonsai-2-27B-Abliterated-PTQ1_0-mtp-lean.gguf` | main model (`bonsai`) | in service | **rebuilt byte-identical** 2026-09-25 (MTP graft, below) |
| `Ternary-Bonsai-2-27B-Abliterated-PTQ1_0.gguf` | graft trunk; `bonsai-vision` | in service | pinned `BoldingBuilds/Ternary-Bonsai-2-27B-Abliterated-PTQ1_0-GGUF` @ `30a3dcbf` |
| `Ternary-Bonsai-2-27B-PTQ1_0-mtp-lean.gguf` | MTP head donor (input only) | input | pinned `sudoingx/Ternary-Bonsai-2-27B-PTQ1_0-MTP-GGUF` @ `e9c159a6` |
| `Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf` | vision projector | in service | pinned `prism-ml/Ternary-Bonsai-2-27B-gguf` @ `6ed5e12b` |
| `...mtp-lean.kv-mean-center-q4_0.gguf` | K-cache bias for q4_0 K | **retired** | recipe only; **not reproducible** (below) |
| `qwen-image-2.1/qwen-image-2.1-Q5_K_M.gguf` | image denoiser, base | in service | pinned `unsloth/Qwen-Image-2.1-GGUF` @ `2c31ccd3` |
| `qwen-image-2.1/qwen_image_2.1_turbo_Q5_K_M.gguf` | image denoiser, turbo | in service | pinned `Abiray/Qwen-Image-2.1-viggle-4-steps-turbo-GGUF` @ `0016110b` |
| `qwen-image-2.1/qwen3vl_8b_heretic-Q4_K_M.gguf` | image text encoder (both) | in service | pinned `pottokao/Qwen-Image-2.1-Text-Encoder-Heretic-GGUF` @ `68ceaa76` |
| `qwen-image-2.1/vae/qwen_image_2.1_vae_bf16.safetensors` | image VAE (both) | in service | pinned `unsloth/Qwen-Image-2.1-FP8` @ `9e520642` |
| `Qwen3-Embedding-0.6B-Q8_0.gguf` | embedder: code search, E1, skills | in service | pinned `Qwen/Qwen3-Embedding-0.6B-GGUF` @ `370f27d7` |
| `Qwen3-Reranker-0.6B-Q8_0.gguf` | reranker | in service | pinned `mradermacher/Qwen3-Reranker-0.6B-GGUF` @ `6727da81` (named `Qwen3-Reranker-0.6B.Q8_0.gguf` there) |
| `Huihui-Qwen3.8-27B-abliterated-UD-DW-Q4_K_M.gguf` | critic (`critic-disabled`) | disabled | pinned `huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF` @ `3d5cf9ef` |
| `index/token_embd.npz` | concept seeds, bonsai viz | in service | **rebuilt byte-identical** 2026-09-25 |
| `index/e1/heads/route_in/v0001.json` | E1 route_in head | in service (behind `YAMADORI_E1=1`) | **weights rebuilt bit-identical** 2026-09-25 from the stored vectors |
| Laya `model.safetensors` | Laya | **retired** 2026-09-24 | pinned `convaiinnovations/laya` @ `55cf4c4e` |
| `index/laya/route_in.json`, `grounded_excerpt.json`, `index/calibration.json` | Laya heads | **retired** | recipe only, not re-run |

The full revisions, sizes, hashes and licences are in the manifest. "Pinned" means
the repo at that full commit serves a file whose LFS sha256 equals ours. The
revision is the commit that was current when the file was downloaded (from a doc,
or the newest commit before the file's mtime), so it records what was fetched,
not just a later commit that happens to hold the same bytes.

Licences to know: the three Qwen-Image files are the **Qwen Research License**
(non-commercial). The rest are Apache-2.0.

## Re-obtaining a pinned file

```
python scripts/fetch_models.py list
python scripts/fetch_models.py fetch qwen3-embedding-0.6b-q8 --dest D:/restore [--dry-run]
python scripts/fetch_models.py check-remote          # every pin still served? (no download)
```

`fetch` downloads `https://huggingface.co/<repo>/resolve/<full commit>/<file>`
to `<name>.part` and hashes it while streaming. It renames the file only when
the size and sha256 match the record. It never overwrites a file that differs.
Ad hoc: `fetch --repo R --revision SHA --filename F --dest DIR` checks the file
against what the HF API says it is at that revision (the LFS sha256, or the git
blob id for a small file). Tested 2026-09-25 on a README (git blob id) and a
260 KB LFS asset (sha256). No large file was downloaded.

## Rebuilding the derived ones

**Main model: the MTP graft** (docs/MTP-STAGING.md §4). The inputs are the two
pinned files above. The tool is `github.com/sudoingX/bonsai2-small-gpu` @
`eb52d9d7363cda2d910146f4e37f4b8c64c30c46`, `graft/tools`, run with the stack
interpreter:

```
python graft/tools/extract_head.py Ternary-Bonsai-2-27B-PTQ1_0-mtp-lean.gguf head-lean.gguf --no-embed-tokens
python graft/tools/merge.py Ternary-Bonsai-2-27B-Abliterated-PTQ1_0.gguf head-lean.gguf Ternary-Bonsai-2-27B-Abliterated-PTQ1_0-mtp-lean.gguf
```

Re-run on 2026-09-25 in a scratch directory: `head-lean.gguf` came out as
`ec9ecad8…` and the graft as `4ca238c1…`, both byte-identical to the originals.
It is CPU-only and takes about a minute.

**`index/token_embd.npz`**:
`python scripts/extract_token_embd.py --gguf <models>/Ternary-Bonsai-2-27B-Abliterated-PTQ1_0.gguf`.
It is CPU-only (about 2 minutes) and refuses to write when its word probes fail.
Re-run on 2026-09-25 with numpy 2.2.6: the file and every array are
byte-identical.

**E1 `route_in` head**: `python bench/e1/eval_e1.py embed` (needs the live
`embeddings` model; fills `index/e1/vectors.sqlite3`), then `... train --force`.
The training labels are three tracked `bench/laya_routing_labels*.jsonl` files,
recorded by hash. Refitted offline on 2026-09-25 from the stored vectors: `W`
sha256 `83f990eb…` and `b` came out identical. Compare `W_sha256`, not the
file hash: the JSON carries a creation timestamp. The head self-tunes (`e1.learn`
promotes new versions), so the deploy check warns when `state.json` serves a
version the manifest does not record.

## What is NOT reproducible, and why

- **The K-cache mean-centering bias** (retired, 66,400 B, `2267a33a…`). Its
  calibration corpus was not kept and its hash was not recorded. The corpus
  also cannot be regenerated: `bench/kv_context/make_calib_corpus.py` reads the
  repo *working tree* (docs, mcp, web/src, the typegpu sources), which had
  uncommitted edits on 2026-09-23. A GPU mean is not guaranteed bit-stable
  either. The recipe is recorded: the tool built in engines' `llama-bonsai2-base`
  tree (sudoingX/llama.cpp `285542d9`) on 2026-09-23, the command, and the A4000.
  Whether the reasoning-budget patch was in that tree when the tool compiled is
  not recorded, though that patch does not touch the KV cache. So a rebuild
  gives a comparable bias but not these bytes. To make the next one reproducible, keep the corpus
  next to the bias and record its sha256.
- **The Laya heads and calibration** (retired). They were not re-run. The
  features came from a bf16 CUDA encoder. `grounded_excerpt`'s training inputs
  were never recorded by hash. The previous `route_in` head is kept in
  `index/laya/_backup_20260922_191425` with its own `SHA256SUMS.txt`.
- **E1's vectors.** The head reproduces exactly from the stored vectors. The
  vectors themselves come from the embedder served by a specific llama-server
  build (engines/manifest.yaml). Re-embedding on another build or card may
  differ in the last bits. That has not been measured.
- **Anything with a floating-point reduction on a GPU.** No recipe here claims
  more than it was shown to do.

## Runtimes and locks

| runtime | recorded | lock |
|---|---|---|
| stack Python (`textgen/installer_files/env`, 3.13.15 conda-forge) | pip freeze + conda-meta package list | `requirements.lock.txt` (**error** on drift) |
| `.venv-laya` | retired with Laya | `locks/laya.lock.txt` |
| SearXNG | commit `3cd69d30e2a7…` (docs/SEARCH.md), settings template, renderer, limiter by sha256 | `locks/searxng.lock.txt` |
| Hermes (measurement harness) | v0.21.5 @ `ee5ee84a345204a3b1d6ef6ba1ab747e602867b9`; the tree differs only in line endings | `locks/hermes.lock.txt` (venv has no pip: importlib list) |
| web dashboard | `web/package-lock.json` is committed; `web/dist/.buildinfo` hashes it and every source file (mcp/dash_static.py) | the committed lock |
| llama-swap | the version it reports, v256 (6701d0d) | the binary itself is pinned by engines/manifest.yaml `llama-swap` |

`python scripts/verify_artifacts.py --write-locks` rewrites every lock from what
is installed. Lines pip cannot install (the conda-built `pip`, SearXNG's editable
`src`) are kept as `#local:` comments, so `pip install -r` skips them but the
check still compares them. SearXNG's rendered `etc/settings.yml` holds its secret
key and is not hashed.

## At deploy: `verify(config_path)`

`scripts/verify_artifacts.py` exposes `verify(config_path) -> list[Problem]`.
It reads `config.yaml` the way llama-swap does (macros expanded, comment lines
dropped) and takes every model-file argument (`-m`, `--mmproj`,
`--kv-mean-center`, `--diffusion-model`, `--vae`, `--llm`, any `.gguf` or
`.safetensors`). Each one must be recorded, present, the recorded size and the
recorded sha256. Then it checks the code-loaded artifacts, then the runtimes.
Each `Problem` has a severity, the artifact, the fact and a remedy. `error`
blocks a deploy. `warn` is drift around the stack (the E1 head, the dashboard,
llama-swap, SearXNG, Hermes). An empty list is the only pass. The CLI is
`python scripts/verify_artifacts.py` (or `fetch_models.py --verify`). It exits 1
on any error.

The first run hashes about 45 GB, which took 47 s here (warm page cache). Later
runs use `index/artifact_hashes.json` (keyed on size and mtime_ns) and take
about 2 s, most of it pip freeze. The cache catches accidents, not deliberate
mtime forgery. Use `--rehash` (`rehash=True`) when that matters. Tests:
`mcp/test_artifacts.py`.

## Changing a model

1. Put the new file in the models directory. Do not overwrite the old one until
   the switch has been through the live gate.
2. Add or replace its manifest entry: size, sha256
   (`python -c "import sys;sys.path.insert(0,'scripts');import verify_artifacts as v;print(v.sha256_file(r'PATH'))"`),
   provenance (`repo`, the full `revision`, `filename`, `checked`,
   `revision_basis`, `licence`), or the recipe with its inputs by id and sha256.
3. `python scripts/fetch_models.py check-remote <id>` for a pinned file.
4. Point `config.yaml` (and `config.template.yaml`) at it. Then `verify` must
   return nothing before `deploy_check.py` runs.
5. Mark the old entry `retired` rather than deleting it while results measured
   on it are still cited.

## Privacy

The repo is public. A model whose name does not already appear in a tracked file
goes in `models/manifest.local.yaml` (gitignored, same shape). `verify` merges
it. Everything recorded on 2026-09-25 was already named in tracked files
(config.template.yaml, docs/IMAGEGEN.md, docs/MTP-STAGING.md,
bench/livebench/README.md), so there is no local file yet. Files in the models
directory that `config.yaml` does not load are not recorded. Neither are the
data indexes the pipeline rebuilds (`index/hints.npz`,
`index/skill_triggers.npz`, the code and package indexes).
