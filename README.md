# Yamadori

## It won't hand you code that doesn't build.

Runs on your hardware. Costs nothing per run. Knows your repo, not a public one.

> **Status: not true yet.** The verification loop is being built now. Everything
> below is either measured and cited, or labelled as unbuilt. Nothing in this
> README claims to work that hasn't been tested — that is the point of the
> project, not a disclaimer.

Collected, not bought.

---

## What it actually is

A **verifier-driven sampling engine wrapped around a local model.** The model is
a component; the engine is the product.

A 27B does not beat a frontier model one-shot and this does not pretend it does.
It beats one where the task has a **mechanical verifier** — `tsc` with
`expectTypeOf`, `cargo test` across a WASM boundary, `naga` on a shader — because
then "sample sixteen times and keep what passes" is available to a free local
model and is not available to a metered one.

That is not a hunch. With a sound external verifier, published results move
5% → 38%, 16% → 37%, 40% → 87%, and a 30B goes 32% → 87% on tool use. A model
grading its *own* work made GPT-4 **worse** on two of three domains. The
verifier has to be a compiler, never an opinion.

Every run also labels its own training data: candidates that pass and fail are
ground truth, free, produced as a by-product. That trains the selector that
picks winners when several compile or none do.

## What got cut, and why that matters

This repo is mostly a record of things that did not survive measurement:

| cut | evidence |
|---|---|
| `judge` (a small classifier as a truth judge) | 6/10 against a 5/10 coin flip; systematic false positives on negated claims |
| semantic search as the default retriever | BM25 beat it 92/120 vs 77/120 on private repos, McNemar p=0.0041 |
| the cross-encoder reranker above top-2 | identical recall at the shipped cutoff, ~1s/query |
| `apply_edit` | the harness already writes files; a second write path is a hazard |

Every threshold and default here names the script that justifies it and the
sample size it was measured at. Where a number is too small to carry a claim,
it says so. Retrieval is a subsystem, and a shrinking one.

## Honest status: this is NOT zero-config

It runs unattended once it is up, and it is genuinely one line to point a
client at. Getting to that line currently takes: building a llama.cpp fork from
source, five model downloads, filling two paths in `config.yaml`, a separate
venv for the decision model, and an index build. A bootstrap script is the next
thing being built; until it lands, read `## Setup` below and expect an evening,
not a minute.

## Endpoints

| | |
|---|---|
| OpenAI API | `https://ai.thejustinwalsh.me/v1` (via Caddy) |
| Control / stats UI | `https://ai.thejustinwalsh.me/ui/` |
| Code-intelligence API | `https://ai.thejustinwalsh.me/tools` |
| OpenAPI spec | `https://ai.thejustinwalsh.me/tools/openapi.json` |

Direct ports still work if Caddy is not running: `:1234` for the API and
dashboard, `:1235` for the tools API.

`ai.thejustinwalsh.me` is public DNS pointing at the ZeroTier address, so the
endpoints keep working if ZeroTier reassigns the IP.

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
# your code, no JSON-RPC needed
curl "http://ai.thejustinwalsh.me:1235/definition?symbol=parse_tool_call"

curl http://ai.thejustinwalsh.me:1235/search \
  -H "Content-Type: application/json" \
  -d '{"query":"how are tool calls parsed","top_k":5}'
```

| route | use when |
|---|---|
| `POST /definition` | you know the identifier — sqlite lookup, no GPU |
| `POST /references` | before changing a signature or deleting code |
| `POST /search` | you cannot name it — embeddings + rerank |
| `GET /status` | index coverage |
| `GET /openapi.json` | self-discovery for tooling |

## Models

| id | what | where | notes |
|---|---|---|---|
| `bonsai` | Ternary Bonsai 2 27B, **208k ctx** | 5060 Ti | thinking ON |
| `bonsai-agent` | same process, alias | 5060 Ti | **thinking OFF — use for tools** |
| `bonsai-vision` | + multimodal projector | A4000 | on demand, ttl 900 |
| `embeddings` | Qwen3-Embedding-0.6B | A4000 | resident |
| `reranker` | Qwen3-Reranker-4B | A4000 | resident |

### Use `bonsai-agent` for anything with tools

This is the single most important operational detail. Measured, with a `tools`
array attached:

| config | result |
|---|---|
| thinking on, `max_tokens` 2000 | 2000 tokens of `<think>`, **no tool call** |
| `reasoning_effort: low` | 2000 tokens of `<think>`, **no tool call** |
| `enable_thinking: false` | **correct tool call in 29 tokens** |

The model reasons past its token budget before emitting a call, so an agent
pointed at plain `bonsai` looks like it is ignoring its tools. `bonsai-agent`
is the same process with `enable_thinking: false` forced server-side, so
clients need no special handling.

## Performance

Measured on this hardware, ternary PTQ1_0 with the custom `pr-ptq1-mmv` kernel:

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
mcp/code_search.py   MCP server: search_code, find_definition, find_references
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
mcp/tool_shim.py     UNUSED. Kept as a record; see known issues
mcp/tools_api.py     HTTP transport for the same tools
```

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
[docs/KNOWN-ISSUES.md](docs/KNOWN-ISSUES.md).

**Embeddings are asymmetric.** Qwen3-Embedding needs an instruction prefix on
*queries* and none on documents. Getting it wrong does not error — it silently
returns near-random results. Handled in `mcp/code_search.py`.

**Rerankers fail silently too.** A cross-encoder reads query+document together;
overflow its context and it returns `0.0000` for everything, replacing a good
embedding ranking with noise. Documents are truncated before reranking and
there is a fallback to embedding order when scores look degenerate.

**VRAM headroom matters.** 240k context *loads* on the 5060 Ti but leaves ~2%
free, and then dies when a long prefill allocates its compute buffer. 208k is
the largest size measured stable.

## Not included, and why

**Adversarial critic.** A second 27B with *different* weights reviewing the
primary's output is the highest-value addition, but Q4_K_M needs ~19.7 GiB
(15.41 weights + KV + compute + MTP) and will not fit on a 16 GiB card beside
anything. It needs a ~12 GiB quant. See the disabled entry in `config.yaml`.

**A second chat worker.** Doubles *concurrent* throughput, but a single agent
is sequential and never touches it. That VRAM buys context and retrieval
quality instead.
