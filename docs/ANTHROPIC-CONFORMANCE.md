# Anthropic Messages API conformance (Claude Code)

Operator, 2026-09-29: "Yes add anthropic api endpoints for claude".

`POST /v1/messages` and `POST /v1/messages/count_tokens` let Claude Code
(and any Anthropic SDK client) use Yamadori as its model. Like
`/v1/responses` (docs/OPENAI-CONFORMANCE.md, R1) it is a **translation over
the one turn**, never a second pipeline: `mcp/messages_api.py` `to_chat` ->
`server._serve_turn` -> `proxy.complete` / `proxy.stream_body` -> `of_chat`
/ `Stream`. The module doc is the authoritative mapping; this page is the
evidence behind it.

**Status: offline only.** `mcp/test_messages_api.py` replays a
Claude-Code-shaped client through the served chat template. No real Claude
Code has run against the stack; every statement below about what Claude Code
sends or does is read from its public docs (code.claude.com, read
2026-09-29), not observed.

## Sources

| what | where |
|---|---|
| the endpoints, headers, streaming rules, pings, stall watchdog, retries, error forwarding | code.claude.com/docs/en/llm-gateway-protocol |
| effort levels, `/effort`, `effortLevel`, `modelSettings`, model env vars, `CLAUDE_CODE_MAX_CONTEXT_TOKENS` | code.claude.com/docs/en/model-config |
| retries, "Prompt is too long", `API_TIMEOUT_MS`, `CLAUDE_STREAM_FIRST_BYTE_TIMEOUT_MS` | code.claude.com/docs/en/errors |
| a base URL alone keeps the claude.ai login as the credential | code.claude.com/docs/en/llm-gateway |
| status -> error type, the envelope, `request_id` | platform.claude.com/docs/en/api/errors |
| what llama-server serves for `tool_choice` | engines/src/llama-bonsai2-ada common/chat.cpp `common_chat_tool_choice_parse_oaicompat` (auto, none, required; anything else throws) and tools/server/server-common.h `json_value` (an object falls back to "auto" with a warning) |

## Request -> chat

| Anthropic | chat body | note |
|---|---|---|
| `system` string / text blocks | the one leading `system` message, blocks joined by a blank line | cache_control dropped, counted |
| `messages[].role: "system"` | through `system_roles.one_system`: a later one becomes a user turn | Claude Code appends them mid-conversation |
| user `text` blocks | one string (`"".join`) | as the template renders a text-part list |
| user `image` (base64 / url / file) | `image_url` data: URL / URL (never fetched) / `input_image` file id | the chat image path (vision placeholders) |
| user `document`, anything else | 400 `invalid_request_error`, `param` = `messages.i.content.j.type` | |
| `tool_result` (string / blocks, `is_error`) | a `tool` message; images as `image_url` parts; document / search_result / tool_reference become a text note | is_error counted (no chat flag) |
| assistant `text` + `tool_use` | one assistant turn: content + `tool_calls` (id kept, input serialized in its key order) | consecutive assistant messages merge |
| assistant `thinking` | `reasoning_content` **only if its signature verifies as ours**; otherwise dropped (the ledger restores the slot's own) | HMAC-SHA256(account, text), key derived from the media-link secret |
| `redacted_thinking`, `server_tool_use`, `*_tool_result` | ignored, counted | |
| `tools[]` custom (name, description, input_schema, strict) | function tools | `defer_loading` ignored (all offered), counted |
| server tools (`web_search_*`, `web_fetch_*`, `code_execution_*`, `tool_search_tool_*`, `advisor_*`, `mcp_toolset`) | not offered, recorded | as Responses ignores hosted tools |
| Anthropic-schema client tools (`bash_*`, `text_editor_*`, `computer_*`, `memory_*`), unknown types | 400 | their schema is not in the request |
| `tool_choice` auto / none / any / tool{name} | "auto" / "none" / "required" / the tools narrowed to that one + "required" | a name not among the tools: 400 |
| `disable_parallel_tool_use` | `parallel_tool_calls` = not it | |
| `max_tokens` (required) | `max_tokens` | the tier's model profile sizes the turn |
| `stop_sequences` | `stop` | |
| `temperature`, `top_p`, `top_k` | the same | vendor sampling enforced, recorded |
| `output_config.format` json_schema | `response_format` json_schema | other formats 400 |
| `output_config.effort`, `thinking`, `model` | `reasoning_effort` (the tier) | below |
| `metadata.user_id` | a session only when it carries one (JSON `session_id`, or legacy `..._session_<uuid>`) | a bare user id is ignored |
| `context_management`, `container`, `mcp_servers`, `service_tier`, `inference_geo`, `speed` | ignored, recorded | the kit sets `CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1` |
| `stream` | `stream` + `stream_options.include_usage` | |

Headers: `anthropic-version`, `anthropic-beta` recorded
(`x_yamadori.messages`). `x-claude-code-session-id` (+ `x-claude-code-agent-id`)
-> `prompt_cache_key` = `claude-code:<session>[:<agent>]`; our
`X-Yamadori-Session` wins. The hint headers (`x-claude-code-request-class`,
`-agent-type`, `-compaction`, `-context-compacted`, `-prompt-id`) are
recorded; `request-class: auxiliary` on a request with no tools is a utility
call (X-Yamadori-Features `utility`; the caller's own header wins).

## Effort -> tier -> model

Claude Code sends `output_config.effort` (and adaptive thinking) for a model
id it does not recognise, such as `yamadori` (gateway protocol, "Requests and
defaults by connection method"). First match wins:

| input | tier | model (mcp/tier_models.yaml) |
|---|---|---|
| `output_config.effort` low | low | bonsai |
| `output_config.effort` medium | medium | bonsai |
| `output_config.effort` high | high | bonsai |
| `output_config.effort` xhigh | xhigh | mirai-s |
| `output_config.effort` max | max | flash-next |
| (no effort) `thinking.type` disabled or between_tools | minimal | bonsai |
| no effort, model `yamadori-max` / `-xhigh` / `-fast` | that name's tier (catalog) | per table |
| none of these | the server default (`tiers.DEFAULT`, medium) | bonsai |

`thinking: {enabled, budget_tokens}` and `{adaptive}` pick nothing: the model's
profile owns the budget (operator, 2026-09-29). An effort value outside the
five is 400. X-Yamadori-Features applies on top, as on every wire.

## Result -> Message

| chat | Message |
|---|---|
| `reasoning_content` | a `thinking` block (signed); `""` + signature with `thinking.display: "omitted"`; none when the client turned thinking off |
| `content` | a `text` block (the proxy's image lines and check notes included, as the client stores them) |
| `tool_calls` (client tools only; ours are hidden hops) | `tool_use` blocks, `input` an object (non-JSON arguments carried as `{"_unparsed_arguments": ...}`, counted) |
| finish `tool_calls` / any call | `stop_reason` `tool_use` |
| finish `length` | `max_tokens` |
| finish `stop` | `end_turn` (`stop_sequence` never claimed: the chat route does not say which stop fired) |
| finish `incomplete` (upstream dropped) | 502 `api_error` (blocking) / `error` event (stream) |
| usage | `input_tokens` = prompt - cached, `cache_read_input_tokens` = cached, `cache_creation_input_tokens` 0, `output_tokens` = completion |
| `x_yamadori` | the chat record plus `messages` (on the Message, and on `message_delta`) |

## The stream

`message_start` (usage zeros) on the turn's first chunk -- nothing before it
(E1). Then per block `content_block_start` / `content_block_delta` /
`content_block_stop`: `thinking` (thinking_delta per chunk, then
`signature_delta` when the answer starts), `text` (text_delta), `tool_use`
(one `input_json_delta` with the whole input: calls are checked whole). After
text has started, reasoning is a `ping` (CHANNEL ORDER). A chat heartbeat is a
`ping` (at most one per second). A pump thread sends a `ping` whenever the
turn has sent nothing for `proxy.HEARTBEAT` (5 s): Claude Code "counts every
byte ... and aborts a stream that goes silent for 300 seconds by default".
The end: `message_delta` {stop_reason, stop_sequence null, usage, x_yamadori}
and `message_stop`. A failure after the first byte: one `error` event, nothing
after it.

## Errors

`{"type": "error", "error": {"type", "message", "x_yamadori": {code, ...}},
"request_id": "req_..."}`, plus a `request-id` header.

| our error | status | Anthropic type |
|---|---|---|
| bad body, a field we refuse, an upstream 4xx, the template's refusal | 400 | invalid_request_error |
| prompt past the window (`context_length_exceeded`) | 400 | invalid_request_error, message "prompt is too long: N tokens > M maximum" (M = window - generation floor) |
| no / unknown key | 401 | authentication_error |
| 403 | 403 | permission_error |
| unknown route | 404 | not_found_error |
| superseded before it started | 409 | conflict_error |
| body too large | 413 | request_too_large |
| lanes full (`server_busy`) | 429 + Retry-After 30 | overloaded_error |
| another tier's model on the card (`model_at_capacity`), model loading (`model_unavailable`) | 503 + Retry-After | overloaded_error |
| upstream failure, ours | 502 / 500 | api_error |
| upstream timeout | 504 | timeout_error |

Known gaps: the body-size middleware (413) and `v1_unknown` (404/405 on other
`/v1` paths) still answer in OpenAI's envelope; Claude Code only calls the
two routes above.

## count_tokens

`{"input_tokens": N, "x_yamadori": {tier, model, counted, ...}}`: the served
template's render of the client's messages and tools, tokenized by the model
server (`proxy.count_prompt_tokens`), on the tier's model when it is the one on
the card (counting must never make llama-swap load a model). Otherwise, or
when the server cannot count, the window check's high estimate (chars / 3),
labelled `counted: false`. What the proxy adds on the turn (restored
reasoning, injections, our tools) is not counted: counting decides and records
nothing.

## The Claude Code setup (harness kit)

The dashboard's HARNESS TOOLS export for Claude Code (mcp/harness_kit_seed.py)
carries `settings.json` (ANTHROPIC_BASE_URL = the public base, every model
role pinned to `yamadori`, effort medium, the proxy's window as
`CLAUDE_CODE_MAX_CONTEXT_TOKENS`, hint headers on, experimental betas and the
attribution block off, first-byte 30 min / request 60 min), `claude-code.env`
(ANTHROPIC_AUTH_TOKEN and YAMADORI_API_KEY as placeholders only) and
`mcp.json` (the code-intelligence MCP server). All three are marked
unmeasured until a real Claude Code run.

## Not yet run

- A real Claude Code session against the stack (the kit, then
  `scripts/deploy_check.py` after the next deploy).
- Whether Claude Code offers xhigh/max in `/effort` for an unrecognised model
  id (the docs list levels per known model).
- Whether a compaction Claude Code sends is recognised by
  `compaction.harness_of` (its summarise turn is read like any in-place
  compaction; not replayed from a capture).
