# Wiring Hermes (or any agent) to this stack

Two connections: the **chat endpoint** and the **MCP server**. They are
independent — the agent talks OpenAI to one and MCP to the other.

## 1. Chat endpoint

```
base URL : http://10.242.120.152:1234/v1     (ZeroTier address)
API key  : required -- see .keys/ on the server
model    : bonsai-agent
```

Port 1234 is now the PROXY, not llama-swap. llama-swap moved to
127.0.0.1:11434 and is no longer reachable from the network, because it has no
authentication of its own and exposing it bypassed everything: no key, no
tools, no account isolation, no logging. Nothing downstream had to change --
1234 is the port every client was already pointed at.

The key is required. A request without one returns 401. Keys are minted with
`python mcp/accounts.py create <label>` and only their hash is stored.

**`bonsai` and `bonsai-agent` are now the same thing.** This file used to say
to use `bonsai-agent` for anything with tools, because with thinking enabled
the model burned its budget reasoning and never emitted a call. That was
measured on the corrupt `pr-ptq1-mmv` build and did not survive the fork fix —
re-measured on the official build, 5 tasks x 3 reps, thinking OFF 13/15 and
thinking ON 12/15, with **0/15 failures to call in both arms**. `config.yaml`
sets `enable_thinking: true` on both ids and `bonsai-agent` is an alias kept so
older clients keep working. See `docs/KNOWN-ISSUES.md`.

`reasoning_effort` accepts `minimal`, `low`, `medium`, `high`, `xhigh` and the
empty string. **`max` is the one that raises** — this chat template returns a
500 from the Jinja on it, as do any other unknown values. The set is empirical
(probed against the running server) and lives in `mcp/tiers.py:ACCEPTED_EFFORTS`.
Through the proxy you cannot hit that crash anyway: `safe_effort()` maps `max`
and `ultra` onto `xhigh` and anything unrecognised onto `medium`. Sending
`max` straight to llama-swap on 11434 does 500.

Some clients want the URL without `/v1`. If you get 404s, drop it.

## 2. MCP server — pick the transport that matches where Hermes runs

This is the thing that silently breaks remote setups. MCP's default transport
is **stdio**: the `"command"` field is a CLI the client *spawns as a local
subprocess*. It cannot cross machines. If Hermes runs on your laptop and the
stack is on the GPU box, a stdio config produces no error — the tools simply
never appear, which looks exactly like the model ignoring them.

### A. Hermes runs ON the GPU box -> stdio

```json
{
  "mcpServers": {
    "code-search": {
      "command": "C:\Users\jwals\textgen\installer_files\env\python.exe",
      "args": ["C:\Users\jwals\llama-stack\mcp\code_search.py"],
      "env": {
        "LLAMA_STACK_URL": "http://127.0.0.1:1234",
        "LAYA_URL": "http://127.0.0.1:1237",
        "CODE_INDEX_DB": "C:\Users\jwals\llama-stack\index\code.sqlite3"
      }
    }
  }
}
```

### B. Hermes runs ANYWHERE ELSE -> MCP over HTTP

Same server, same tools, same code path — reachable over ZeroTier:

```
POST http://ai.thejustinwalsh.me:1235/mcp
```

It speaks plain MCP JSON-RPC: one request in, one response out.

```bash
curl http://ai.thejustinwalsh.me:1235/mcp -H "Content-Type: application/json"   -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'
```

For a client that wants a config block:

```json
{
  "mcpServers": {
    "code-search": {
      "type": "http",
      "url": "http://ai.thejustinwalsh.me:1235/mcp"
    }
  }
}
```

Exact key names vary between clients (`type`/`transport`, `url`/`endpoint`) —
check yours. If it only supports stdio, run a thin local relay that forwards
stdin to that URL; the protocol is identical on both sides.

**Which do you want?** stdio is lower latency and needs no open port. HTTP is
the only option when the agent is not on the GPU box. Both serve the same
eight tools from one implementation, so there is nothing to keep in sync.

### Tools it exposes

**EIGHT, not four, and three of the names below are pre-rename.** The list the
server actually advertises is `code_search.TOOLS`; `tools/list` is the
authority and this table is not. The old names in it do not resolve:

| old name (does NOT resolve) | current name |
|---|---|
| `find_definition` | `find_definition_opt` |
| `search_code` | `find_by_meaning` |
| `index_status` | `describe_index` |
| `grep` | `find_by_pattern` |
| `read_file` | `read_file_range` |
| `verify` | `run_check` |
| `judge`, `compact` | **gone** — `judge` dispatches nowhere (6/10 vs a 5/10 coin flip), `compact` was never built |

The eight on the MCP surface are `find_by_meaning`, `find_definition_opt`,
`find_references`, `find_by_pattern`, `read_file_range`, `describe_index`,
`run_check` and `summarize_text`. `record_step` and `read_rings` exist but are
**not** on this surface — the work log is keyed to a conversation only the
proxy knows, so an outside MCP client is not offered a tool that cannot work
correctly for it. Through the chat endpoint the model sees twelve, those two
plus `bind_project_context` and `delegate_investigation`.

| tool | use when |
|---|---|
| `find_definition_opt(symbol)` | you know the name — instant, exact, no GPU |
| `find_references(symbol, calls_only)` | before changing a signature or deleting code |
| `find_by_pattern(pattern)` | the text appears verbatim somewhere |
| `find_by_meaning(query, top_k)` | you *cannot* name it |
| `read_file_range(path, start, end)` | you already have a location |
| `describe_index()` | check coverage |
| `run_check(check)` | you changed something |
| `summarize_text(text, focus)` | a long log or transcript |

The split matters. `find_definition_opt` is a sqlite lookup and is always the
right first move when an identifier is known. `find_by_meaning` is for
questions like *"how does this project handle retries"*. Agents that reach for
it when they already have a symbol name waste time and get worse answers.

**Note on what `find_by_meaning` runs today:** at shipped defaults it is BM25
plus the symbol table, *not* embeddings + reranking. `CODE_SEARCH_SEMANTIC=0`
(keyword beat semantic 92/120 to 77/120, McNemar p=0.0041) and
`RERANK_MAX_K=2` against `DEFAULT_TOP_K=5`, so the cross-encoder does not run
at the default k. Either way the model never sees those models; they are
implementation details behind the tool.

### Indexing

The index is not built automatically. Run it per repo:

```powershell
& C:\Users\jwals\textgen\installer_files\env\python.exe `
  C:\Users\jwals\llama-stack\scripts\index_code.py C:\path\to\repo
```

Full rebuild each time; incremental updates are not implemented. Chunking is
tree-sitter based across 17 languages including Rust, C++, Zig, TypeScript/TSX
and WGSL/GLSL/HLSL, falling back to a regex heuristic for anything without a
grammar.

## 2b. HTTP API (for your own code, not the agent)

Same tools, no JSON-RPC:

```
http://ai.thejustinwalsh.me:1235
  POST /search      {"query": "...", "top_k": 5}
  POST /definition  {"symbol": "..."}
  POST /references  {"symbol": "...", "calls_only": true}
  GET  /status
  GET  /openapi.json
```

GET forms work too: `/definition?symbol=foo`. Use this from build scripts,
editors and CI; use MCP from agents.

## 3. System prompt

The stack does not inject one. Put this in your agent, adapted:

```
You are a principal-level engineer working on realtime systems: Rust, C++,
Zig, WASM, TypeScript, and GPU work (WebGPU/WGSL, Skia, React Native).
Data-oriented design: the data layout is the problem, not the code.

TOOLS — reach for them before answering from memory:
- Know the identifier? call find_definition_opt. Never guess at a definition.
- About to change a signature or delete code? call find_references first.
- Cannot name what you need? call find_by_meaning with a plain-language question.
- Retrieved snippets insufficient? refine the query and call again.

Never assert what code does without having read it. If you have not seen a
definition in this conversation, look it up.

Performance claims require evidence. Do not state that something is faster
without a measurement. Propose hypotheses; let the benchmark decide.

Platform constraints are data: state cache line size, SIMD width and frame
budget when they bear on a decision.
```

The tool-use lines matter more than they look. Tool descriptions alone are a
weak signal; an explicit "you know the name → find_definition_opt" rule
measurably changes which tool gets picked.

## 4. Checking it works

```powershell
# models present
curl http://ai.thejustinwalsh.me:1234/v1/models

# tool calling actually fires
curl http://ai.thejustinwalsh.me:1234/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model":"bonsai-agent",
  "messages":[{"role":"user","content":"Where is parse_tool_call defined?"}],
  "tools":[{"type":"function","function":{
    "name":"find_definition",
    "description":"Find where a symbol is defined.",
    "parameters":{"type":"object","properties":{"symbol":{"type":"string"}},"required":["symbol"]}}}],
  "max_tokens":500}'
```

Expect `finish_reason: tool_calls`. Note that with thinking ON — which is now
both ids — the call arrives after the reasoning block rather than within ~30
tokens, so give it room: a `finish_reason: length` with empty content means
`max_tokens` was too small, **not** that you picked the wrong id. Switching to
`bonsai-agent` will not change it; the two are the same process.

## 5. Operations

| | |
|---|---|
| Dashboard | `http://ai.thejustinwalsh.me:1234/ui/` — load/unload, live token stats, request inspector |
| Stack log | `logs/stack.log` |
| Watchdog log | `logs/watchdog.log` |
| Manual restart | `Start-ScheduledTask -TaskName llama-stack` |
| Health check now | `.\scripts\watchdog.ps1 -WhatIfOnly` |

The watchdog runs every 5 minutes and asks exactly one question per service:
does a `/health` route answer? It checks llama-swap (`:11434`), the proxy
(`:1234`), the tools API (`:1235`) and Laya (`:1237`), plus one `/health` per
llama-swap upstream with the names taken from llama-swap's own `/running`. On
failure it restarts **that service**, not the stack, with a 10-minute cooldown
so a service that crashes on load fails loudly instead of thrashing the GPUs.

**It deliberately knows no model names, reads no model status and runs no
generation probe.** The earlier version did all three, decided a healthy stack
was dead because the proxy advertises `yamadori` rather than `bonsai`, and
killed llama-swap every ten minutes for hours — see `docs/FINDINGS.md` §1 and
`docs/PROTOCOL.md` rule 12. A description of the old behaviour stood here until
this audit.
