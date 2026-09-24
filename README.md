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
| **剪定 · sentei** | Pruning. Eight tools on the MCP surface, eleven the model sees (twelve with the `delegate_investigation` benchmark flag). `judge` was cut at 6/10 against a coin flip and now dispatches nowhere. |
| **芽摘み · metsumi** | Bud-pinching. Sample many answers. Keep the one that compiles. |
| **舎利 · shari** | Deadwood, bleached and kept. Every failed idea stays in the repo with its evidence. |
| **根張り · nebari** | Root flare. Retrieval you can see: `taproot`, `branch`, `shoot`. |
| **年輪 · rings** | Growth rings. A work log that outlives the context window. |
| **床の間 · tokonoma** | The alcove. Where the tree gets displayed. |

## What it does

You point any OpenAI client at one URL. That is the whole setup.

The stack reads the source of every library you import, at the version you
use, and searches it on the model's behalf. Your own code stays with your
client: the model reads it through your harness's file tools, and the server
never looks at your disk. It checks the code it writes before it answers, and
it remembers what it already did, so a long session does not redo its own
work.

The client never learns any of this. It thinks it added a model.

## What it refuses to claim

The tree is not finished. It will not be.

Nothing here is asserted without a measurement, and the measurements have
been unkind. BM25 beat the embedding index 92 to 77 and the embeddings got
switched off. Every reranker number turned out to be void, because its
scores change with what else is in the batch (`docs/FINDINGS.md` #20). A fan-out
experiment returned a clean null. And two separate things happened to the
decision model, which used to be written here as one sentence and are not
related:

- **Calibration went nowhere.** Temperature-scaling it on 151 held-out
  contrastive pairs (`index/calibration.json`, `scripts/calibrate_laya.py`)
  moved ECE from 0.279 to 0.032 and left accuracy at **0.5066 against a 0.500
  majority baseline the pairs have by construction** — which is the only
  outcome possible, because a single scalar divisor is monotonic and cannot
  reorder an argmax. Honest confidence, no new capability. `docs/LAYA.md`
  Finding 10.
- **A different tool was cut.** `judge` — the decision model asked to say
  whether a proposition holds — scored **6/10 against a 5/10 coin flip** on
  unambiguous yes/no engineering questions (`scripts/eval_judge.py`), with
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
| OpenAPI spec | `https://ai.thejustinwalsh.me/tools/openapi.json` |

Direct ports still work if Caddy is not running: `:1234` for the API and
dashboard, `:1235` for the tools API.

`ai.thejustinwalsh.me` is public DNS pointing at the ZeroTier address, so the
endpoints keep working if ZeroTier reassigns the IP.

## Effort tiers

The standard `reasoning_effort` field is the only switch a client needs. Each
tier turns on one more augmentation (`mcp/tiers.py`, `TIERS`); nothing else has
to be configured. A tier says what is *allowed*: the selection engine still
decides per request whether fan-out and deep thinking are worth running, and
every decision comes back on the response as `x_yamadori`.

Fan-out means **up to 3** candidates, the original answer included, generated
**in sequence by the second brain** (the helper context deep thinking also
uses), never in parallel: the second brain writes one more answer, the code
check grades the two, and only when neither clearly wins does it write a
tie-breaker from both candidates and their check results. At most two
contexts are live at once. A code answer's winner is delivered; for a prose
answer, the second answer's differing points follow the model's own answer
and the model weighs them in its own turn.

Our code-search tools are not offered to the model you talk to: it sees your
client's tools (plus image tools where an image server is configured, and at
`xhigh` and `max` one more, `think_deeply`, to call when it is stuck). The
service works beside it -- library definitions after a library question, the
second brain's investigation, repair and comparison -- and folds the result
back in fixed phrases ("After thinking deeply,", "Verified", "Repaired",
"Compared two approaches"), each second-brain result opening with "Today I
was inspired by <word>." for the concept seed it drew. Everything it adds is
replayed byte for byte on every request, so the model's prompt cache is
never broken by the service.

| `reasoning_effort` | thinking | library help | skills | code check | fan-out | deep thinking | addendum | images | concept seed | adds |
|---|---|---|---|---|---|---|---|---|---|---|
| `minimal` | off | – | – | – | 1 | – | – | yes | – | the fastest answer: no thinking, nothing of ours (least injection-resistant) |
| `low` | on | – | – | – | 1 | – | – | yes | – | nothing: the model as it ships, the benchmark baseline |
| `medium` | on | definitions | yes | note | 1 | – | – | yes | – | library definitions for a library question, skills (formerly hints; armed without review, screened), a note when a client write does not parse |
| `high` | on | definitions | yes | repair | up to 3 | – | yes | yes | yes | the second brain: code that does not parse is repaired, a second approach is compared, and the addendum says so |
| `xhigh` | on | definitions | yes | repair | up to 3 | allowed | yes | yes | yes | everything: deep thinking on top of `high` |
| `max` | on | definitions | yes | repair | up to 3 | allowed | yes | yes | yes | everything, with the longest thinking (slowest) |

- **Code check:** syntax and lint of what the model writes; its code is
  changed only where a repair of real errors needs it (formatting is
  reported, never applied).
- **Deep thinking** (`xhigh`, `max`) runs on a trigger, on any kind of
  request, agent steps included: the model calls `think_deeply`; the
  service sees the work stall (the same error again, a failing command
  re-run, "still broken"); the conversation uses a library newer than the
  model; or a large new task arrives, whose plan the second brain writes
  first. It researches library source, skills, the service's notes and the
  web (a local SearXNG). Every trigger and non-trigger is recorded with its
  outcome, and the thresholds are tuned from them within bounds
  (`mcp/deep.py`, `mcp/deep_learn.py`, `GET /dash/api/deep`).
- **Default:** `medium`, when a client sends nothing.
- **Accepted values:** any of the six names above picks its tier; anything
  else gets the default, never an error.
- **Thinking budget:** it isn't set by the tier. It comes from the request's
  share of the KV cache: 5/8 of the pool for the conversation, minus the
  prompt and the answer allowance.
- **Images:** `generate_image` is offered on every tier when image generation
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
| `POST /search` | you cannot name it — BM25 + symbols (see note) |
| `GET /status` | index coverage |
| `GET /openapi.json` | self-discovery for tooling |

> **Retrieval defaults.** `CODE_SEARCH_SEMANTIC=0` and `RERANK_MAX_K=2`, so at
> shipped defaults `/search` runs BM25 plus the symbol table; embeddings are off
> and the reranker only participates at k<=2, where its scores are unreliable
> (`docs/FINDINGS.md` #20). The 4B reranker is NOT used: its
> GGUF returns inverted near-zero scores through llama.cpp's rank path and fails
> silently. See `docs/SELECTION.md` for when each mechanism applies.


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
| `reranker` | Qwen3-Reranker-0.6B | A4000 | resident |

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
mcp/selection.py     per-request: hints, deep thinking, fan-out
mcp/deep.py          deep thinking's triggers, their records and outcome labels
mcp/deep_learn.py    idle-time learner: thresholds within bounds, reversible
mcp/research_tools.py  the second brain's skills, notes and web sources
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
