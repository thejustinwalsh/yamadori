# Wiring Hermes (or any agent) to this stack

Two connections: the **chat endpoint** and the **MCP server**. They are
independent — the agent talks OpenAI to one and MCP to the other.

## 1. Chat endpoint

```
base URL : http://ai.thejustinwalsh.me:1234/v1
API key  : none (any dummy string if the client insists)
model    : bonsai-agent
```

Use **`bonsai-agent`**, not `bonsai`, for anything with tools. See the table in
the README: with thinking enabled the model burns its entire token budget
reasoning and never emits a tool call. `bonsai-agent` is the same process with
thinking forced off server-side.

Use `bonsai` when you want deep reasoning and are *not* passing tools — long
analysis, architecture discussion, hard debugging. `reasoning_effort` accepts
`low`, `medium`, `xhigh` **only**; `high` makes the chat template raise and
aborts generation.

Some clients want the URL without `/v1`. If you get 404s, drop it.

## 2. MCP server

stdio transport. Config shape used by most MCP clients:

```json
{
  "mcpServers": {
    "code-search": {
      "command": "C:\\Users\\jwals\\textgen\\installer_files\\env\\python.exe",
      "args": ["C:\\Users\\jwals\\llama-stack\\mcp\\code_search.py"],
      "env": {
        "LLAMA_STACK_URL": "http://127.0.0.1:1234",
        "CODE_INDEX_DB": "C:\\Users\\jwals\\llama-stack\\index\\code.sqlite3"
      }
    }
  }
}
```

### Tools it exposes

| tool | use when |
|---|---|
| `find_definition(symbol)` | you know the name — instant, exact, no GPU |
| `find_references(symbol, calls_only)` | before changing a signature or deleting code |
| `search_code(query, top_k)` | you *cannot* name it — fuzzy semantic search |
| `index_status()` | check coverage |

The split matters. `find_definition` is a sqlite lookup and is always the right
first move when an identifier is known. `search_code` runs embeddings +
reranking on the GPU and is for questions like *"how does this project handle
retries"*. Agents that reach for semantic search when they already have a
symbol name waste time and get worse answers.

The model never sees the embedding model or the reranker. They are
implementation details behind `search_code`.

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
- Know the identifier? call find_definition. Never guess at a definition.
- About to change a signature or delete code? call find_references first.
- Cannot name what you need? call search_code with a plain-language question.
- Retrieved snippets insufficient? refine the query and call again.

Never assert what code does without having read it. If you have not seen a
definition in this conversation, look it up.

Performance claims require evidence. Do not state that something is faster
without a measurement. Propose hypotheses; let the benchmark decide.

Platform constraints are data: state cache line size, SIMD width and frame
budget when they bear on a decision.
```

The tool-use lines matter more than they look. Tool descriptions alone are a
weak signal; an explicit "you know the name → find_definition" rule measurably
changes which tool gets picked.

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

Expect `finish_reason: tool_calls` within ~30 completion tokens. If you get
`finish_reason: length` with empty content, you are on `bonsai` instead of
`bonsai-agent`.

## 5. Operations

| | |
|---|---|
| Dashboard | `http://ai.thejustinwalsh.me:1234/ui/` — load/unload, live token stats, request inspector |
| Stack log | `logs/stack.log` |
| Watchdog log | `logs/watchdog.log` |
| Manual restart | `Start-ScheduledTask -TaskName llama-stack` |
| Health check now | `.\scripts\watchdog.ps1 -WhatIfOnly` |

The watchdog runs every 5 minutes: it checks the process, the API, that every
resident model is loaded, and that the primary can actually produce a token. On
failure it restarts the stack, with a 10-minute cooldown so a model that
crashes on load fails loudly instead of thrashing the GPUs. It deliberately
does not restart while an on-demand model (vision) has evicted the resident set
— that is normal operation.
