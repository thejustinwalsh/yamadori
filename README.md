<div align="center">

# 山採り · YAMADORI

**Collected, not bought.**

</div>

---

A *yamadori* is a tree taken off the mountain. Nobody grew it. Nobody potted
it. The wind bent it, the rock starved it, lightning took a limb. Collectors
prize it over nursery stock for exactly that reason: the damage is the
provenance.

Then someone spends thirty years deciding which branches live.

This is that, for models. Open weights. Abliterated — the nursery-safe
behaviours stripped out. Running on iron you own, shaped for your work, by
you. Nobody can revoke it, meter it, or deprecate it out from under you.

## The bet

Big models are grown. Yours is **shaped**.

A frontier model answers once, because every answer is billed. Yours answers
as many times as you want, because it costs nothing. Give it a compiler and
that stops being a party trick:

> **Sixteen tries and a compiler beats one try and a genius.**

That is the whole thesis. Everything here serves it.

## The craft

Bonsai has a vocabulary for cutting things away, and it fits.

| | |
|---|---|
| **剪定 · sentei** | Pruning. Eight tools on the MCP surface; the model sees your client's tools plus a few of ours (images, package lookups). `judge` was cut at 6/10 against a coin flip; deep thinking, fan-out and the fix-up were cut on 2026-09-29 ([docs/REMOVED.md](docs/REMOVED.md)). |
| **芽摘み · metsumi** | Bud-pinching. Sample many answers. Keep the one that compiles. |
| **舎利 · shari** | Deadwood, bleached and kept. Every failed idea stays in the repo with its evidence. |
| **根張り · nebari** | Root flare. Retrieval you can see: `taproot`, `branch`, `shoot`. |
| **年輪 · rings** | Growth rings. A work log that outlives the context window. |
| **床の間 · tokonoma** | The alcove. Where the tree gets displayed. |

## What it does

You point any OpenAI client at one URL. That is the whole setup.

The stack gives the model what it gets wrong on its own: the exact names,
versions and READMEs of the packages you use, looked up in the registries when
the model asks (the MCP host's package lookups). Your own code stays with your
client: the model reads it through your harness's file tools, and the server
never looks at your disk. It remembers what it already did, so a long session
does not redo its own work, and it never breaks the model's prompt cache.

The client never learns any of this. It thinks it added a model.

## What it refuses to claim

The tree is not finished. It will not be.

Nothing here is asserted without a measurement, and the measurements have
been unkind. BM25 beat the embedding index 92 to 77 and the embeddings got
switched off. Every reranker number turned out to be void, because its
scores change with what else is in the batch (`docs/FINDINGS.md` #20), and
the reranker was removed (2026-10-01, `docs/REMOVED.md`). A fan-out
experiment returned a clean null. And two separate things happened to the
decision model, which used to be written here as one sentence and are not
related:

- **Calibration went nowhere.** Temperature-scaling it on 151 held-out
  contrastive pairs (`index/calibration.json`, `scripts/calibrate_laya.py`,
  deleted with Laya on 2026-09-29 and in git at `e360d37`)
  moved ECE from 0.279 to 0.032 and left accuracy at **0.5066 against a 0.500
  majority baseline the pairs have by construction** — which is the only
  outcome possible, because a single scalar divisor is monotonic and cannot
  reorder an argmax. Honest confidence, no new capability. `docs/LAYA.md`
  Finding 10.
- **A different tool was cut.** `judge` — the decision model asked to say
  whether a proposition holds — scored **6/10 against a 5/10 coin flip** on
  unambiguous yes/no engineering questions (`scripts/eval_judge.py`, also
  deleted with Laya), with
  three of four misses being false positives on the *negative* cases. It is
  now advertised in no tool list and no longer dispatches by name.

Neither caused the other. They are different label sets, different scripts and
different questions — one is a calibration number on generated code pairs, the
other is a tool-surface decision on hand-written yes/no questions.

**This repository is mostly a record of things that did not survive testing.**
That is the point. Every number below names the script that produced it and
the sample size it ran at. Where the sample is too small to carry a claim, it
says so.

The verification loop is still being built. Until it lands, the line at the
top of this file is a promise, not a receipt.

---

## Endpoints

| | |
|---|---|
| OpenAI API | `https://ai.thejustinwalsh.me/v1` (via Caddy) |
| Dashboard | `https://ai.thejustinwalsh.me/` (old `/dash` links redirect; Python pages at `/dash/classic`) |
| Code-intelligence API | `https://ai.thejustinwalsh.me/tools` |
| OpenAPI spec | `https://ai.thejustinwalsh.me/tools/openapi.json` (needs the account key, like every tools route except `/health`) |

Direct ports still work if Caddy is not running: `:1234` for the API and
dashboard, `:1235` for the tools API.

`ai.thejustinwalsh.me` is public DNS pointing at the ZeroTier address, so the
endpoints keep working if ZeroTier reassigns the IP.

## Effort tiers

The standard `reasoning_effort` field is the only switch a client needs. Each
tier says what the service may add (`mcp/tiers.py`, `TIERS`); nothing else
has to be configured, and every decision comes back on the response as
`x_yamadori`.

Our code-search tools are not offered to the model you talk to: it sees your
client's tools, plus image tools where an image server is configured and, from
`medium` up, the MCP host's package lookups (`yama_find_package`,
`yama_list_package_versions`, `yama_read_package_readme`,
`yama_resolve_packages`). Everything the service adds is replayed byte for
byte on every request, so the model's prompt cache is never broken by it.

**Removed 2026-09-29** ([docs/REMOVED.md](docs/REMOVED.md); the way back is
commit `e360d37`): the second brain -- deep thinking, fan-out, the fix-up
repair and its "Verified / Repaired" notes -- the addendum, library
definitions, Laya and CLM. They were tried and did not help.

**Skills are off at every tier** (operator, 2026-09-29: "Stop skills until we
have a good skill injector." -- "Skills are still valuable we just haven't
found the unlock yet. TBD."). Nothing of the skills system reaches the
model's context; the skills code stays.

| `reasoning_effort` | thinking | skills | MCP tools | images | concept seed | adds |
|---|---|---|---|---|---|---|
| `minimal` | off | – | – | yes | – | the fastest answer: no thinking, nothing of ours (least injection-resistant) |
| `low` | on | – | – | yes | yes | the concept seed only: the model's own thinking plus the seed (2026-10-06) |
| `medium` | on | – | yes | yes | yes | the MCP host's package lookups and the concept seed |
| `high` | on | – | yes | yes | yes | a concept seed on the conversation's first user turn |
| `xhigh` | on | – | yes | yes | yes | the same as `high`: the effort-matched pair to `max` |
| `max` | on | – | yes | yes | yes | everything, with the longest thinking (slowest) |

- **Concept seed:** one word drawn from the model's own vocabulary, once per
  conversation, appended to its first user turn so the conversation starts
  somewhere the model would not otherwise go; `x_yamadori.session.seed`
  names it.
- **Default:** `medium`, when a client sends nothing.
- **Accepted values:** any of the six names above picks its tier; anything
  else gets the default, never an error.
- **Thinking budget:** it isn't set by the tier. It comes from the request's
  share of the KV cache, minus the prompt and the answer allowance, and is
  capped per kind of turn (`mcp/tiers.py`).
- **Images:** `yama_generate_image` is offered on every tier when image generation
  is configured: it is a capability, not a gate (`docs/IMAGEGEN.md`).

## TLS

Caddy terminates TLS on 443 so nothing needs a port number. The certificate is
a real Let's Encrypt cert obtained via **DNS-01**, which is the only challenge
that can work here: `ai.thejustinwalsh.me` resolves to a ZeroTier address, so
Let's Encrypt cannot reach it over HTTP for an HTTP-01 or TLS-ALPN challenge.
DNS-01 proves control by writing a TXT record instead.

Setup is one secret:

```powershell
copy caddy\env.example caddy\.env
# paste a DNSimple ACCOUNT token with zone write access, then restart
Start-ScheduledTask -TaskName llama-stack
```

`caddy/.env` is gitignored. The launcher starts Caddy only when the token is
present, so the stack still runs on raw ports without it.

## Two transports for the same tools

MCP is for agents; the HTTP API is for your own software. Identical logic
behind both — `mcp/code_search.py` and `mcp/tools_api.py` import the same
module, so they cannot drift.

```bash
# your code, no JSON-RPC needed -- the same account key as :1234
curl "http://ai.thejustinwalsh.me:1235/definition?symbol=parse_tool_call" \
  -H "Authorization: Bearer $YAMADORI_API_KEY"

curl http://ai.thejustinwalsh.me:1235/search \
  -H "Authorization: Bearer $YAMADORI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query":"how are tool calls parsed","top_k":5}'
```

Every route except `/health` needs the key. A browser request whose Origin is
not listed in `YAMADORI_TOOLS_ALLOWED_ORIGINS` is refused (2026-09-26). Client
configs for Claude Code, OpenCode and Hermes are in `docs/TOOLS-API.md`.

| route | use when |
|---|---|
| `POST /definition` | you know the identifier — sqlite lookup, no GPU |
| `POST /references` | before changing a signature or deleting code |
| `POST /search` | you cannot name it — BM25 + symbols (see note) |
| `GET /status` | index coverage |
| `GET /openapi.json` | self-discovery for tooling |

> **Retrieval defaults.** `CODE_SEARCH_SEMANTIC=0`, so at shipped defaults
> `/search` runs BM25 plus the symbol table; embeddings are off. The reranker
> that reordered results at k<=2 was removed 2026-10-01 (`docs/REMOVED.md`;
> its scores were unreliable, `docs/FINDINGS.md` #20). See `docs/SELECTION.md`
> for when each mechanism applies.


## Models

Clients on `:1234` see one model, **`yamadori`**, and any unknown name
resolves to it (`mcp/catalog.py`). The ids below are llama-swap's, on loopback
`:11434` behind the proxy.

| id | what | where | notes |
|---|---|---|---|
| `bonsai` | Ternary Bonsai 2 27B, **147,456 ctx** | 5060 Ti | thinking ON |
| `bonsai-agent` | same process, same settings, alias only | 5060 Ti | thinking ON (see config.yaml) |
| `bonsai-vision` | + multimodal projector | A4000 | on demand, ttl 900 |
| `embeddings` | Qwen3-Embedding-0.6B | A4000 | resident |

(`reranker`, Qwen3-Reranker-0.6B, was removed 2026-10-01: `docs/REMOVED.md`.)

### `bonsai-agent` is now an alias and nothing more

**SUPERSEDED — the reversal is left visible on purpose.** This section used to
say `bonsai-agent` forced `enable_thinking: false` server-side because thinking
ate the whole budget before a tool call. That was measured on the **corrupt
`pr-ptq1-mmv` build** and did not survive fixing the fork:

| config | tool calls, 5 tasks x 3 reps, official build, temp 0.3 |
|---|---|
| thinking OFF | 13/15 correct, **0/15 failures to call** |
| thinking ON | 12/15 correct, **0/15 failures to call** |

Equivalent within noise, and zero failures to call either way. `config.yaml`
now sets `enable_thinking: true` on **both** ids, so `bonsai` and
`bonsai-agent` are the same process with the same settings and the alias exists
only so older clients keep working. Point agents at either.

The reason thinking is ON rather than OFF is prompt injection, not tool use:
against one naive exfiltration payload, thinking OFF leaked 5/5 and thinking ON
at the shipped temp 0.3 leaked 1/5. **1/5 is not a defence**, it is a reduction
on one payload against an abliterated model — the real controls are
architectural and are listed in `config.yaml` above the alias.

## Performance

Measured on this hardware, ternary PTQ1_0 on the **official `prism` build** —
not the `pr-ptq1-mmv` kernel, which is faster (55.33 t/s) and numerically
broken, and which nothing here runs. See "Things that will bite you" below and
[docs/KNOWN-ISSUES.md](docs/KNOWN-ISSUES.md):

| config | prefill | decode |
|---|---|---|
| 5060 Ti alone | 490.5 t/s | **46.06 t/s** |
| A4000 alone | ~413 t/s | ~38 t/s (est.) |
| both GPUs, split | ~447 t/s | slower than one card |

**One card beats two cards split.** A pipeline split runs the halves
sequentially with an activation handoff, so it costs ~8%. Nothing in this stack
is split across GPUs.

For reference, the journey to get here (same model family, same machine):

| | decode @ 65k ctx |
|---|---|
| exl3 4.0bpw, split | 11.1 t/s |
| GGUF Q4_K_M + MTP, split | 24.9 t/s |
| ternary PTQ1_0, single card | **~46 t/s** |

## Layout

```
bin/                 llama-swap (downloaded, not committed)
config.yaml          the whole stack definition
engines/manifest.yaml  every engine pinned: base commit, patches, flags, shipped hashes
engines/patches/     our patch series, one directory per engine
engines/src/         the engine SOURCE we build (upstream base + our patches),
                     vendored; `scripts/build_engine.py build <engine>` builds
                     it with no network (docs/ENGINES.md)
mcp/code_search.py   MCP server: the 8 tools in TOOLS (find_by_meaning,
                     find_definition_opt, find_references, find_by_pattern,
                     read_file_range, describe_index, run_check,
                     summarize_text). The names search_code / find_definition
                     are the pre-rename ones and no longer exist.
scripts/
  index_code.py      build the vector + symbol index
  ast_chunker.py     tree-sitter chunking (17 languages)
  symbols.py         definitions / call sites / references
  start-stack.bat    launcher (sets CUDA_DEVICE_ORDER + CUDA dll path)
  start-stack-hidden.vbs   window-less shim for the Scheduled Task
  install-autostart.ps1    registers autostart + watchdog
  watchdog.ps1       health check and self-heal
docs/HERMES.md       wiring an agent to this stack
docs/KNOWN-ISSUES.md open problems, with the measurements behind them
mcp/server.py        the front door on :1234 (auth, /v1, dashboard at /); logic in proxy.py
mcp/model.py         the one door for internal generation (same tiers.apply)
mcp/selection.py     per-request: may skills run; the side-call (utility) rule
mcp/mcp_host.py      the MCP client the proxy runs (PackageLens), yama_* lookups
mcp/pinned_fetch.py  the pinned GET the MCP host reads READMEs through
mcp/skills.py        the skill store (Agent Skills folders under index/skills/library)
mcp/skill_pipeline.py  the one skill pipeline; prompts in skill_prompts.py
mcp/skill_select.py  which skills a request gets; docs/SKILL-FACTORY.md is the API
skills/authored/     skills written from our own evidence (SKILL.md + tests.json)
mcp/worker.py        claims dataset jobs from index/jobs.sqlite3
mcp/tool_shim.py     UNUSED. Kept as a record; see known issues
mcp/tools_api.py     HTTP transport for the same tools
web/                 React dashboard; committed web/dist is served at the site root
design/              design system source for the dashboard
attic/               gitignored: superseded backups and stale logs
```

Run the tests with `python scripts/run_tests.py` (offline + ruff), then
`--live` through `:1234` before claiming anything works. See `AGENTS.md`.

## Setup

```powershell
# 1. autostart at logon + 5-minute watchdog
.\scripts\install-autostart.ps1

# 2. index a repo (needs the stack running)
& C:\Users\jwals\textgen\installer_files\env\python.exe `
  .\scripts\index_code.py C:\path\to\repo
```

`install-autostart.ps1` also **disables** the old `text-generation-webui`
autostart so it cannot contend for port 1234. The task is kept, not deleted —
re-enable it any time.

**The engines are built from this repo.** Every llama.cpp and sd.cpp tree the
stack runs is vendored under `engines/src/<engine>` (the upstream base with
our patch series applied), so a clone is enough to rebuild them and nothing
is fetched (the two entries that reproduce a binary with llama-server's web
UI, `llama-prism` and `llama-bonsai2-base`, also need that UI archive,
pinned by hash):

```powershell
python scripts\build_engine.py check                  # the trees are what the manifest records
python scripts\build_engine.py build llama-bonsai2-ada --jobs 8
```

The toolchain (VS 2022 + CUDA 12.8) is the manifest's `toolchains` entry.
Moving an engine to a newer upstream is `build_engine.py update <engine> --to
<sha>`, which rebases our patches and reports conflicts. llama-swap is the
upstream release binary, pinned by hash. See
[docs/ENGINES.md](docs/ENGINES.md) "Vendored source".

## Things that will bite you

**Device numbering.** `start-stack.bat` pins `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
Without it CUDA orders devices *fastest-first*, which reverses these two cards
and silently puts the primary model on the slower GPU. `config.yaml` assumes
`CUDA0 = 5060 Ti`.

**CUDA DLLs.** `llama-server` is built against the isolated CUDA toolkit in
`textgen/installer_files/cudabuild`. Without that directory on `PATH`,
`ggml-cuda.dll` fails to load and the server exits silently with status 0.

**Build from the OFFICIAL fork only.** `PTQ1_0` is a PrismML quant type and
stock llama.cpp rejects it, but picking the wrong fork is worse than picking
none: `sudoingX/llama.cpp` branch `pr-ptq1-mmv` carries a faster PTQ1_0 decode
kernel that is numerically broken. It is ~20% quicker (55.33 vs 46.06 tok/s)
and collapses generation into runs of `/`, taking tool calling from 9/9 to
0/10. Use `PrismML-Eng/llama.cpp` branch `prism`, release
`prism-b10709-9a9394a`. Full comparison in
[docs/KNOWN-ISSUES.md](docs/KNOWN-ISSUES.md). The trees we build are the ones
vendored in `engines/src`, each recorded with its origin in
`engines/manifest.yaml`.

**Embeddings are asymmetric.** Qwen3-Embedding needs an instruction prefix on
*queries* and none on documents. Getting it wrong does not error — it silently
returns near-random results. Handled in `mcp/code_search.py`.

**VRAM headroom matters.** 240k context *loads* on the 5060 Ti but leaves ~2%
free, and then dies when a long prefill allocates its compute buffer. 208k was
called "the largest size measured stable" here for a while; it is not what
ships and the number came down twice since. **147,456 is what `config.yaml`
launches with** (`-c 147456`), and it stands on its VRAM measurement alone:
warm at 147,456, 12,122 MiB used and 3,931 free, roughly the 3 GB of headroom
that absorbs compute-buffer spikes. It was once attributed to `arc193_a`
failing on all three benchmark arms as a compute-buffer spike; that attribution
is unsupported — the one completed `arc193_a` row died of `max_tokens`, not OOM
(`docs/CONSTRAINTS.md` §1c). `mcp/budget.py` splits whatever the server reports
5/8 to the conversation and 3/8 to one deep-thinking helper, reserve 0 (at
147,456: 92,160 / 55,296), so `-c` is the only place the number lives. Every
request's thinking budget is derived from its role's share
(`docs/CONSTRAINTS.md` #19).

## Not included, and why

**Adversarial critic.** A second 27B with *different* weights reviewing the
primary's output is the highest-value addition, but Q4_K_M needs ~19.7 GiB
(15.41 weights + KV + compute + MTP) and will not fit on a 16 GiB card beside
anything. It needs a ~12 GiB quant. See the disabled entry in `config.yaml`.

**A second chat worker.** Doubles *concurrent* throughput, but a single agent
is sequential and never touches it. That VRAM buys context and retrieval
quality instead.
