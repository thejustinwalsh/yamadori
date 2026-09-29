# Harnesses on the Responses API

Research, 2026-09-26. The operator wants harnesses to use Responses
(`POST /v1/responses`). The best case is that a harness notices it from what
we advertise. The fallback is that we configure it. **We never patch a
harness**: this file lists configuration only.

For each harness this file gives:

- whether it detects Responses on its own, with evidence;
- the exact config that selects Responses;
- what it sends and what it needs back;
- the gaps against the adapter being built (`mcp/responses_api.py`, read on
  2026-09-26, not edited).

## Versions, sources, and how the claims were checked

| harness | version | where the source came from | official? |
|---|---|---|---|
| Hermes | 0.21.5, commit `ee5ee84` (2026-09-24) | `%LOCALAPPDATA%\hermes\hermes-agent` (Python). This is the Windows install that `bench/octopus/run.py:82` runs. | `origin` is `git@github.com:NousResearch/hermes-agent.git` |
| OpenCode | 1.18.32 | `nvm\v20.15.0\node_modules\opencode-ai\bin\opencode.exe`. It is a bun-compiled binary, so claims were read from its embedded JS after extracting the strings. It bundles `@ai-sdk/openai` **3.0.84**. | npm maintainer `thdxr` (Dax Raad, the OpenCode author) |
| Pi | 0.87.1 (`pi-coding-agent` and `pi-ai`) | `nvm\v24.21.0\node_modules\@earendil-works\pi-coding-agent` (JS, readable) | npm `repository` is `github.com/earendil-works/pi`; maintainers `badlogic` and `mitsuhiko` |
| Codex CLI | 0.133.0 | Installed in WSL (`~/.local/share/fnm/.../@openai/codex`). The Rust source is `github.com/openai/codex` at tag `rust-v0.133.0`, fetched with `gh`. | npm `@openai/codex`, repository `openai/codex` |

**Captures.** Every harness was also run once against a local recording
server on `127.0.0.1:18998` (WSL `:18999`). The server replied with a fixed,
valid Responses stream that says "ok". Nothing went to `:1234` and no GPU was
used. Each run used an isolated home or config dir and a dummy key.

- OpenCode: `OPENCODE_CONFIG` plus XDG dirs.
- Pi: `PI_CODING_AGENT_DIR`.
- Codex: a temporary `CODEX_HOME`.
- Hermes: a temporary `HERMES_HOME` holding a copy of the octo profile's
  `config.yaml`, not its `.env`.

Each capture is n=1: one request shape per harness, one prompt ("say ok"),
no tool calls. The captures are marked **(captured)** below. Everything else
comes from source.

## Summary

| harness | detects Responses by itself? | the switch | what fixes the wire today |
|---|---|---|---|
| Hermes | **No** for any custom endpoint | `providers.<name>.api_mode: codex_responses` on a **named** provider | Hostname allowlist only (api.openai.com, api.x.ai, ...). A plain `provider: custom` **ignores** `api_mode: codex_responses` for a non-OpenAI URL. |
| OpenCode | **No** | `"npm": "@ai-sdk/openai"` on the provider | `npm`, which defaults to `@ai-sdk/openai-compatible` (Chat Completions). No probe. |
| Pi | **No** | `"api": "openai-responses"` in `models.json` | `api`, which is required for a custom provider. There is no default and no probe. |
| Codex | Not needed: Responses is its only wire | `wire_api = "responses"` (the default) | `wire_api = "chat"` is a parse error |

**No harness switches wire protocol based on anything a server publishes.**
That covers `/v1/models` fields, headers, an OpenAPI document, and a
well-known URL. Section 6 lists the few advertising hooks that exist. None of
them selects the wire for a custom endpoint.

**Adapter blocker found (P0 for Codex).** Codex 0.133.0 sends a `namespace`
tool by default (`multi_agent_v1`, captured). `mcp/responses_api.py:367-372`
answers any `namespace` tool with 400, so **every Codex request would fail**.
See section 7.

---

## 1. What the OpenAI spec lets a server advertise

- **`GET /v1/models`.** The spec's `Model` object is `{id, object: "model",
  created, owned_by}` plus `shutdown_date` (openapi.json v2.3.0,
  `components.schemas.Model`, from `github.com/openai/openai-openapi` `main`).
  The Python SDK that Hermes uses has the same four fields
  (`openai/types/model.py:10-23`, openai 2.24.0). There is no capability,
  endpoint or wire-protocol field.
- **Capability discovery: none.** The spec has no endpoint, header or
  well-known document that announces which APIs a server supports. The only
  model paths in openapi.json are `/models`, `/models/{model}` and
  fine-tuning permissions. The OpenAPI document is published on GitHub, not
  served by the API.
- **What our `/v1/models` carries.** `mcp/catalog.py:261-301` (`_chat_card`)
  returns `id`/`object`/`created`/`owned_by`, three context fields
  (`CONTEXT_FIELDS`, `catalog.py:149`), `modalities`, `top_provider`,
  `supported_parameters`, and `x_yamadori`. The JSON service descriptor
  already lists `/v1/responses` among its endpoints (`mcp/server.py:125-126`).
  **No harness reads either for the wire choice.**
- **Hermes reads `/v1/models` for metadata only (captured).** Before its
  first turn it sent `GET /api/v1/models`, `GET /v1/models/yamadori`,
  `GET /v1/models` and `POST /api/show` (an Ollama probe). These are for
  context length and capabilities, not for the api_mode.

## 2. Hermes 0.21.5

### Auto-detect: no

- **URL detection is a fixed hostname table.** `api.x.ai`, `api.meta.ai` and
  `api.router.com` map to `codex_responses`, as do official OpenAI hosts
  (`hermes_cli/runtime_provider.py:89-92`, `99-127`;
  `hermes_cli/providers.py:285-296` `is_official_openai_host`). Nothing about
  a server's responses is consulted.
- **The model-name upgrade to Responses never applies to custom providers.**
  - `agent/agent_init.py:470-496` upgrades to Responses only when `api_mode`
    was not explicit and `_is_direct_openai_url()` or
    `_provider_model_requires_responses_api` holds.
  - `run_agent.py:683-684` returns False for provider `custom`. Named
    `providers:` entries resolve to runtime provider `custom`
    (`hermes_cli/runtime_provider_custom.py:455-458`, `576-578`).
- **Plain `provider: custom` ignores the setting.**
  `_resolve_plain_custom_api_mode` drops a persisted
  `model.api_mode: codex_responses` for any non-OpenAI URL
  (`runtime_provider.py:183-194`: "Ignoring persisted custom
  api_mode=codex_responses for non-OpenAI endpoint"). **Setting
  `model.api_mode` next to `provider: custom` does nothing.**
- **Named providers opt in.**
  - A `providers:` entry's `api_mode` or `transport` is honoured
    (`runtime_provider_custom.py:160-167`) and becomes the runtime api_mode
    (`576-578`).
  - Aliases are canonicalised: `responses`, `openai_responses` and
    `openai-responses` all become `codex_responses`
    (`hermes_cli/config_providers.py:40-54`).
  - Hermes' own example uses exactly this (`cli-config.yaml.example:245-253`,
    entry `dbx-gpt`).

### Config (Responses against `:1234`)

```yaml
# HERMES_HOME/config.yaml
model:
  default: "yamadori"
  provider: "yamadori"            # a providers: key (runtime_provider.py:484-486)
providers:
  yamadori:
    base_url: "http://127.0.0.1:1234/v1"
    api_mode: codex_responses     # alias "responses" also works
    key_env: YAMADORI_API_KEY     # runtime_provider_custom.py:154; or api_key: ${VAR}
```

With `--provider <name>` on the command line, as `bench/octopus/run.py:356`
does, `model.provider` is not needed.

**For the Octopus runner** this is one added line under `octo-relay` in
`bench/octopus/make_profile.py`: `api_mode: codex_responses`. The relay
forwards any path (`relay.py:214`). Its collector reads chat-shaped events
(`relay.py:75-115`), though. On Responses, `x_yamadori`, `usage` and the
finish reason sit inside the terminal event's `response`, so the relay
would record none of them until it learns that shape.

**Verified (captured):** with such an entry (`cap-responses`), Hermes sent
`POST /v1/responses`.

### What Hermes sends

**Main turn** (`agent/transports/codex.py:651-806`; captured):

- `model`, and `instructions`: the system prompt, 10,891 characters in the
  capture. System messages never go into `input`
  (`codex_responses_adapter.py:605`).
- `input` items:
  - user messages as `{"role":"user","content":"say ok"}`: no `type`, with
    string content;
  - past assistant turns as `message` items with `id`, `status` and `phase`;
  - `function_call` and `function_call_output` items.
- `store: false` (`codex.py:719`).
- `tools` as flat functions with `strict: false`
  (`codex_responses_adapter.py:344-357`), `tool_choice: "auto"` and
  `parallel_tool_calls: true` (`codex.py:722-725`).
- `reasoning: {effort, summary: "auto"}` and
  `include: ["reasoning.encrypted_content"]` (`codex.py:570-599`).
  - The effort vocabulary for provider `custom` is the widest one: `none`,
    `minimal`, `low`, `medium`, `high`, `xhigh` and `max`
    (`plugins/model-providers/custom/__init__.py:33-44`). So `max` reaches
    us.
  - Reasoning switched off becomes `{"effort":"none"}` and `include: []`
    (`codex.py:596-599`).
- `prompt_cache_key` (next subsection), and `max_output_tokens` only when
  `agent.max_tokens` is set (`codex.py:786-787`; absent in the capture).
- `text.verbosity` only if configured (`codex.py:748-751`). There is never
  `previous_response_id`.

**Replay rules** (`agent/codex_responses_adapter.py`):

- **Reasoning items** are replayed only when they carry `encrypted_content`
  (`406`). Their `id` is stripped (`429`). With our `encrypted_content: null`,
  Hermes sends no past reasoning back. That matches the chat path, where
  Hermes strips `reasoning_content`. Since 2026-09-27 the proxy's ledger
  puts the slot's own reasoning back into those turns on both paths
  (AGENTS.md "Past reasoning is restored", switch `restore_reasoning`).
- **Tool-call ids.** A call's `call_id` is kept verbatim up to 64 characters
  (`286-291`). Our 26-character `call_<session>_<8hex>` carrier survives. A
  repeated id gets a `_dup<n>` suffix on the wire (`470-495`).
- **Tool outputs** may be arrays of `input_text`/`input_image` (`519-536`).

**Side calls use Responses too.** The auxiliary client follows the main
runtime's mode (`auxiliary_client.py:2867-2869`, `2997-2998`). That covers
compaction, titles and flush_memories. The captured title call had:

- its own 942-character `instructions`;
- no tools and no `reasoning`;
- **a different `prompt_cache_key`**;
- `x-stainless-read-timeout: 30.0`.

Aux calls never send `max_output_tokens` (`auxiliary_client.py:1484-1499`).

**`prompt_cache_key` is a content hash, not a session id.** It is
`"pck_" + sha256(session scope, instructions, name-sorted tools)[:24]`
(`codex.py:414-435`, `726-736`). This matters because AGENTS.md makes
`prompt_cache_key` session source 1. On the chat path Hermes sends no key to
a custom endpoint and our `call_...` carrier names the session. On Responses:

- the key changes whenever `instructions` or the tool list change. At a
  compaction Hermes rebuilds the system prompt and refreshes the tool
  schemas (`agent/conversation_compression.py:3138-3149`), so **a compaction
  may start a new session on our side** if the rebuilt prompt is not
  byte-identical;
- every aux call gets its own key (`auxiliary_client.py:1527-1545`), so a
  flattened compaction arrives under a key no conversation has.

**What Hermes needs back** (`agent/codex_runtime.py:680-943`):

- **Events.** It consumes raw SSE:
  - output items come from `response.output_item.done`, not from the
    terminal `response.output`;
  - text comes from `*output_text.delta`;
  - reasoning comes from any event whose type contains `reasoning` and
    `delta`;
  - terminal events are `response.completed`, `response.incomplete` and
    `response.failed`;
  - a `type: "error"` frame raises (`codex_runtime.py:892-903`).
- **Terminal.** A stream with no terminal frame and no output raises
  (`942`).
- **Usage.** It reads `input_tokens`, `input_tokens_details.cached_tokens`
  and `output_tokens`.
- **Finish reasons.** For a custom endpoint, `completed` is trusted as final.
  A reasoning-only answer is not forced "incomplete" (`1212-1227`).
- **`_leaked_tool_call_text`.** An answer that looks like
  `to=functions.x {...}`, or shell JSON after a lead-in, is treated as
  incomplete and retried (`76-84`, `1177-1186`).
- **Watchdogs** (`chat_completion_helpers.py:1154-1224`):
  - an idle gap of **12 s** between events on a context under 10k tokens,
    60 s / 120 s / 180 s above 10k / 50k / 100k. Any parsed event counts, so
    the adapter's 1 s `response.in_progress` keepalive satisfies it;
  - time to first event: 120 s, raised for a local base URL.
  - **Aux calls** count only substantive deltas as progress. Keepalives do
    not re-arm them (`auxiliary_client.py:1370-1378`). The first substantive
    payload must arrive within **60 s** (`auxiliary_client.py:392`;
    per task: `auxiliary.<task>.no_progress_timeout`).
- **Off by default, keep it off:** `compression.codex_responses_native`
  (`hermes_cli/config_defaults.py:654`). It would send `context_management`
  and expect `compaction` items.

## 3. OpenCode 1.18.32

### Auto-detect: no

- The package comes from config. A config model's
  `npm = C.provider?.npm ?? s.npm ?? existing ?? ... ?? "@ai-sdk/openai-compatible"`.
- A models.dev entry's is
  `npm: ... ?? Z.provider?.npm ?? $.npm ?? "@ai-sdk/openai-compatible"`.

  Both are in opencode.exe's bundled provider module (upstream
  `packages/opencode/src/provider/provider.ts`). OpenCode never calls
  `/v1/models` on a custom provider (also `docs/OPENAI-CONFORMANCE.md`
  section 9).
- **`@ai-sdk/openai` means Responses.**
  - The SDK factory is `createOpenAI(options)`.
  - The model is `h.language ?? m.languageModel(id)`. The `openai` plugin's
    `sdk.responses(...)` hook fires only for providerID `openai`.
  - In the bundled `@ai-sdk/openai` 3.0.84, `languageModel` is the Responses
    model: `V=(j)=>P(j)`, with `P` constructing the `${name}.responses` model
    that POSTs `/responses`.
  - `.chat` is the only Chat Completions path, and nothing selects it for a
    custom provider.

### Config

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "yamadori/yamadori",
  "small_model": "yamadori/yamadori",
  "provider": {
    "yamadori": {
      "npm": "@ai-sdk/openai",
      "name": "Yamadori",
      "options": {
        "baseURL": "http://127.0.0.1:1234/v1",
        "apiKey": "{env:YAMADORI_API_KEY}"
      },
      "models": {
        "yamadori": {
          "name": "Yamadori",
          "attachment": true,
          "reasoning": true,
          "tool_call": true,
          "modalities": { "input": ["text", "image"], "output": ["text"] },
          "limit": { "context": 132096, "output": 26419 },
          "variants": {
            "minimal": { "reasoningEffort": "minimal" },
            "xhigh":   { "reasoningEffort": "xhigh" },
            "max":     { "reasoningEffort": "max" }
          }
        }
      }
    }
  }
}
```

This is `docs/HARNESS-OPENCODE.md`'s recommended file with two changes:

- `npm` is `@ai-sdk/openai`;
- `setCacheKey` is gone. For `@ai-sdk/openai`, OpenCode sets
  `promptCacheKey = sessionID` unless `setCacheKey` is `false`.

`@ai-sdk/openai` is bundled in the binary, so no npm install happens at
runtime. **Verified (captured):** `POST /v1/responses`, and the run finished
with `ok` and `tokens {input 10, output 1}` read from our usage.

### What OpenCode sends (captured, and bundled transform)

- **Provider options go under the `openai` key.** The providerOptions key
  for `@ai-sdk/openai` is `openai` whatever the provider id is (`Ey` /
  `_J`). With `reasoning: true` it adds `forceReasoning: true`. Every
  request then gets:
  - `store: false`;
  - `include: ["reasoning.encrypted_content"]`;
  - the system prompt as a leading `{"role":"developer","content":"..."}`
    item with no `type` and no `instructions`;
  - no `temperature` or `top_p`, which are stripped for "reasoning models".
- **Main turn:**
  - `max_output_tokens` (min(limit.output, 32000); 26,419 captured);
  - `tools` as flat functions with `strict: false`, and
    `tool_choice: "auto"`;
  - `prompt_cache_key = "ses_..."` (the OpenCode session id);
  - **no `reasoning` field unless a variant is chosen**, so the tier is our
    default;
  - headers `x-session-id` and `x-session-affinity` carry the same `ses_...`
    id.
- **Title call** (small_model): no tools, no `prompt_cache_key`,
  `reasoning: {effort: "low", summary: "auto"}`, and the same session
  headers.
- **Replay** (AI SDK `w8`, store false):
  - reasoning parts without `encrypted_content` are **dropped** with a
    warning ("Reasoning parts without encrypted content are not supported
    when store is false"). So nothing of ours is echoed;
  - OpenCode strips `itemId` from providerOptions when `store !== true`, so
    assistant text goes back as plain messages;
  - `function_call` items carry `call_id` = our id;
  - tool results are `function_call_output`, whose `output` may be an array
    with `input_image` / `input_file`.

### What OpenCode needs back

The AI SDK's zod chunk schema `A8` validates each known event. **A
known-type event with the wrong shape falls through to the catch-all
`{type: string}` branch and is silently dropped as `unknown_chunk`.** It is
not an error. Required shapes:

- **`response.created`:** `response.id` (string), `created_at` (number),
  `model` (string).
- **`response.output_item.added`:** `output_index` (number) and `item`:
  - a message with `id`, where `phase`, if present, is `commentary` or
    `final_answer`;
  - a reasoning item with `id` and optional `encrypted_content`;
  - a function_call with `id`, `call_id`, `name` and `arguments` (a string).
- **`response.output_text.delta`:** `item_id`, which must equal the message
  item's `id`, and `delta`.
- **`response.completed` / `response.incomplete`:** `response.usage` is
  REQUIRED, non-null, with numeric `input_tokens` and `output_tokens`.
  `incomplete_details.reason` maps the finish reason.
- **`response.failed`:** `sequence_number` (number) is REQUIRED.
- **Reasoning is shown ONLY through summaries.**
  - `response.reasoning_summary_part.added/done` and
    `response.reasoning_summary_text.delta` produce it, as does the item's
    `summary` on the blocking path.
  - This SDK version has **no handler for `response.reasoning_text.delta`**.
  - `output_item.done` for a reasoning item dereferences state created by
    its `output_item.added`, so `added` must come first.

## 4. Pi 0.87.1

### Auto-detect: no

- `api` is a free string on a provider or a model
  (`pi-coding-agent/dist/core/model-config.js:161`, `199`).
- A custom provider's model **must** name it. There is no default, only an
  error: `Provider X, model Y: no "api" specified`
  (`dist/core/provider-composer.js:146-150`).
- The only remote catalog is `https://pi.dev/api/models/providers/<id>`. It
  overlays models onto **built-in** providers
  (`dist/core/remote-catalog-provider.js:40-68`), not custom ones. There is
  no `/v1/models` probe.

### Config

```json
// PI_CODING_AGENT_DIR/models.json (default ~/.pi/agent/models.json)
{
  "providers": {
    "yamadori": {
      "name": "Yamadori",
      "baseUrl": "http://127.0.0.1:1234/v1",
      "api": "openai-responses",
      "apiKey": "$YAMADORI_API_KEY",
      "models": [
        { "id": "yamadori", "name": "Yamadori", "reasoning": true,
          "input": ["text", "image"], "contextWindow": 132096, "maxTokens": 26419 }
      ]
    }
  }
}
```

`$VAR` in `apiKey` is an environment reference, and `!cmd` runs a command
(`dist/core/resolve-config-value.js:65-72`, `123-129`). Run it with
`pi --provider yamadori --model yamadori`. **Verified (captured):**
`POST /v1/responses`, answer `ok`.

### What Pi sends

Source: `pi-ai/dist/api/openai-responses.js`, `openai-responses-shared.js`;
captured.

- `model`, `input`, `stream: true`, `store: false`, and `max_output_tokens`
  (26,419 captured).
- `prompt_cache_key = sessionId` (a UUIDv7, per session; `openai-responses.js:225`).
  The headers `session_id` and `x-client-request-id` carry the same id
  (`186-196`).
- `reasoning: {effort, summary: "auto"}` and
  `include: ["reasoning.encrypted_content"]` when a thinking level is set
  (`248-258`). Thinking off sends `{"effort": "none"}` (`259-263`). Our
  tiers map `none` to `minimal` (`mcp/tiers.py:84`).
- **No `instructions`.** The system prompt is a leading
  `{"role":"developer","content":"..."}` item with no `type`
  (`openai-responses-shared.js:125`).
- Tools are flat functions **without** a `strict` key. `tool_choice` is sent
  only if set.
- **Replay** (`openai-responses-shared.js`):
  - **Pi replays the WHOLE reasoning item it received**, as JSON:
    `thinkingSignature = JSON.stringify(item)` (`585`), pushed back verbatim
    (`176-181`). That includes `id`, `summary`, `encrypted_content: null`,
    `status`, and `content` if we sent one. In `content` mode Pi echoes the
    reasoning text, like Codex. In `summary` mode it echoes display text,
    which the adapter ignores.
  - Assistant text becomes `message` items with `id` (ours, or
    `msg_pi_<n>`) and `phase` (`184-206`).
  - Function calls are re-serialised with
    `arguments: JSON.stringify(parsedArguments)` (`235`). This is not
    byte-identical to what the model wrote.
  - `call_id` stays verbatim: Pi keeps `${call_id}|${item.id}` internally
    (`368`).

### What Pi needs back

- **Terminal events:** `response.completed` or `response.incomplete`.
  Without one it throws "OpenAI Responses stream ended before a terminal
  response event" (`657`).
- **Incomplete reasons.** `incomplete` with reason `max_output_tokens`
  becomes `length`. **Any other reason becomes an error** (`666-675`).
- **Output items.** Slots are created on `response.output_item.added`
  (`481-483`). Deltas without a slot are ignored.
- **Reasoning.** Pi reads both `reasoning_summary_text.delta` and
  `reasoning_text.delta`.
- **Errors.** `type: "error"` and `response.failed` throw with
  `error.code: error.message`.

## 5. Codex CLI 0.133.0

### Auto-detect: not needed

`WireApi` has one variant, `Responses`, and it is the default.
`wire_api = "chat"` is a deserialisation error: "`wire_api = "chat"` is no
longer supported" (`codex-rs/model-provider-info/src/lib.rs:45`, `50-79`).

### Config

```toml
# ~/.codex/config.toml (or CODEX_HOME)
model = "yamadori"
model_provider = "yamadori"
model_reasoning_effort = "medium"          # none|minimal|low|medium|high|xhigh
model_supports_reasoning_summaries = true  # REQUIRED, or no `reasoning` is sent
model_context_window = 132096              # else the fallback 272,000 is assumed

[model_providers.yamadori]
name = "Yamadori"
base_url = "http://127.0.0.1:1234/v1"
env_key = "YAMADORI_API_KEY"
wire_api = "responses"
# stream_idle_timeout_ms = 300000          # the default
```

The keys are defined at `codex-rs/config/src/config_toml.rs:147-158`,
`345-356` and `model-provider-info/src/lib.rs:84-136`.

**Why the three `model_*` lines matter:**

- `yamadori` is an unknown slug, so Codex uses fallback metadata
  (`models-manager/src/model_info.rs:66-103`):
  `supports_reasoning_summaries: false`, `context_window: 272_000`,
  `supports_parallel_tool_calls: false`, and no reasoning levels.
- With summaries unsupported, `build_reasoning` returns None
  (`core/src/client.rs:697-713`). **No `reasoning` field is sent**, so the
  effort setting is lost.
- The effort enum stops at `xhigh` (`protocol/src/openai_models.rs:45-53`).
  **Codex cannot request tier `max`.**

**Verified (captured,** with the config above pointing at the recorder**):**
`POST /v1/responses`, answer `ok`.

### What Codex sends (captured; `core/src/client.rs:716-768`)

- `instructions`: 20,771 characters of Codex base instructions.
- `input`:
  1. a `developer` message: permissions and sandbox text;
  2. a `user` message with `<environment_context>`;
  3. the user's message.

  All three are `{"type":"message", "content":[{"type":"input_text",...}]}`.
- `tools` (captured, default features):
  - functions `exec_command`, `write_stdin`, `update_plan`, `get_goal`,
    `create_goal`, `update_goal`, `request_user_input`, `view_image`;
  - **`{"type":"namespace","name":"multi_agent_v1","tools":[5 functions]}`**;
  - `{"type":"web_search","external_web_access":false}`.
- **Namespaces are sent by default.** They go out whenever
  `provider.capabilities().namespace_tools`, which defaults to true for
  every provider (`model-provider/src/provider.rs:27-41`;
  `core/src/tools/spec_plan.rs:194-197`, `254-256`). Tool types in this
  version: function, namespace, tool_search, image_generation, web_search
  and custom/freeform (`tools/src/tool_spec.rs:18-50`).
- Other fields:
  - `tool_choice: "auto"`, and `parallel_tool_calls: false` from the
    fallback metadata;
  - `reasoning: {effort, summary: "auto"}` and
    `include: ["reasoning.encrypted_content"]`;
  - `store: false`, `stream: true`;
  - `prompt_cache_key` = the thread id, a UUID stable for the thread
    (`client.rs:750`);
  - `client_metadata: {"x-codex-installation-id": ...}`.
- It never sends `previous_response_id` over HTTP (`codex-api/src/common.rs:170-212`).
- Headers: `session-id`, `thread-id`, `x-client-request-id`,
  `x-codex-window-id`, `x-codex-turn-metadata`, `x-codex-beta-features`,
  `originator: codex_exec`.
- **Replay** (`protocol/src/models.rs:751-800`, `978-985`):
  - reasoning items are sent back with `summary` and `encrypted_content`.
    `content` is included only when it holds `reasoning_text`. So in the
    adapter's `content` mode Codex echoes the reasoning text;
  - `id` fields are never serialised back;
  - function calls carry `namespace` when the call was namespaced.

### What Codex needs back (`codex-api/src/sse/responses.rs`)

- **Terminal event.** `response.completed` is required, with
  `response.id`. A stream that closes without it is an error: "stream closed
  before response.completed" (`420-425`).
- **Usage** is optional. If present, `input_tokens`, `output_tokens` and
  **`total_tokens`** are required integers. If `input_tokens_details` or
  `output_tokens_details` is present, its `cached_tokens` or
  `reasoning_tokens` must be an integer. Otherwise: "failed to parse
  ResponseCompleted" (`100-143`, `358-374`).
- **`response.incomplete` is always an error**, whatever the reason
  (`347-357`).
- **Parse failures drop items silently.** Items are parsed from
  `output_item.added` and `.done` as `ResponseItem`
  (`protocol/src/models.rs:751`). A reasoning item **without `summary`**
  does not parse and is dropped (debug log).
- **`response.failed`** is mapped by `error.code`
  (`context_length_exceeded`, quota, `server_is_overloaded`, ...), and
  everything else is retryable (`312-345`).
- **Idle timeout** is 300 s between parsed events. `stream_max_retries` is
  5 and `request_max_retries` is 4 (`model-provider-info/src/lib.rs:26-28`).

## 6. "Detect from what we advertise": the hooks that exist

None of these selects the wire for a custom endpoint. They are listed so no
one searches for them again.

| harness | hook | what it could carry | limit |
|---|---|---|---|
| OpenCode | `OPENCODE_MODELS_URL`: fetches `${url}/api.json`, a models.dev-shaped catalog (opencode.exe `ModelsDev.fetchApi`) | A `yamadori` provider with `npm: "@ai-sdk/openai"`, `api: "<our /v1>"`, `env: ["YAMADORI_API_KEY"]` and model limits. OpenCode then offers it with no provider block in `opencode.json`. | Still one environment variable. It **replaces** the whole catalog (other providers vanish). We would have to serve `/api.json` (not built). |
| Codex | Remote `GET {base_url}/models?client_version=...` in Codex's own `ModelsResponse {models: [ModelInfo]}` shape (`codex-api/src/endpoint/models.rs:31-73`) | Context window, reasoning levels and summary support. It would remove the three `model_*` lines. | Fetched only for ChatGPT auth or a provider with **command auth** (`auth.command`) (`models-manager/src/manager.rs:314-316`). Our `/v1/models` is OpenAI's list shape, so such a fetch would fail to decode. `model_catalog_json` (a local file path, `config_toml.rs:354-356`) is the static equivalent. |
| Hermes | none. `/v1/models` feeds context length only (probes captured above). | -- | -- |
| Pi | none. The pi.dev catalog covers built-in providers only. | -- | -- |

## 7. What the adapter must handle: checklist against `mcp/responses_api.py`

"Adapter" below means `mcp/responses_api.py` as read on 2026-09-26. Line
numbers point there unless a harness path is given.

### Gaps

**Status 2026-09-26 (after this checklist, `mcp/test_responses_api.py`):**
1 closed (namespaces flattened, calls carry `namespace`, replayed
namespaced calls accepted); 2 closed (usage always an object of integers);
3 closed (a dropped upstream is `failed`/`server_error`; `incomplete` only
for `max_output_tokens`); 4 closed by the coordinator's rule (a NEW key
whose history carries our SUMMARY LINE continues that conversation as an
alias; our id in call ids alone is a fork, a new conversation with
`forked_from`; a key-named conversation's call ids and compaction summaries
carry its carrier; AGENTS.md, sessions); 5 unchanged (summary default, `summary` on every item, `added`
first); 6 checked (none..xhigh and max all pick a tier); 7 open (Hermes aux
calls' 60 s no-progress limit; operator config).

1. **P0 (Codex): `namespace` tools.** The adapter raises 400 for any tool
   type it does not know, and `namespace` is one of them (`367-372`;
   docstring line 86). Codex 0.133.0 sends `multi_agent_v1` as a namespace
   on every request (captured). The fix, sketched:
   - flatten `namespace.tools[]` into function tools;
   - when the model calls one, emit the `function_call` item with
     `"namespace": "<name>"`, since Codex routes by (namespace, name)
     (`protocol/src/models.rs:787-800`);
   - accept `namespace` on input `function_call` items.
2. **P1 (OpenCode): always send a `usage` object.** Send one on
   `response.completed` and `response.incomplete`, with zeros when unknown.
   `usage_of` returns None when the chat usage is missing (`728-729`). The
   AI SDK then drops the terminal event, so OpenCode records finish `other`
   and zero tokens.
3. **P1: the meaning of `response.incomplete` differs per client.**
   - Codex: always an error.
   - Pi: an error unless the reason is `max_output_tokens`.
   - Hermes: triggers its continuation path.
   - OpenCode: finish reason.

   So the adapter's `upstream_dropped` reason (`_status_of`, `744-750`)
   becomes an error in Pi and Codex. That is probably correct for a dropped
   upstream. Keep `max_output_tokens` for `finish_reason: length`.
4. **P1 (Hermes): `prompt_cache_key` is a content hash** of (session scope,
   instructions, tools), not a stable session id. The adapter keeps it as
   session source 1 (docstring line 90). Consequences:
   - a compaction that rebuilds Hermes' system prompt can open a new session;
   - aux calls (title, compaction summary) arrive under their own keys.

   This is for the operator to decide. It is new with Responses: over chat,
   Hermes sends no key and our tool-call-id carrier names the session.
   Codex (thread UUID), Pi (session UUID) and OpenCode (`ses_...`, main
   turns only) send stable per-session keys.
5. **P2: reasoning display.**
   - The default `summary` mode is the only one OpenCode displays: its AI
     SDK has no `reasoning_text` handler.
   - `content` mode changes behaviour only for Codex and Pi, which echo it.
     Hermes and OpenCode drop reasoning without `encrypted_content`.
   - Keep `summary: []` on every reasoning item: Codex fails to parse one
     without it.
   - Keep `output_item.added` before any summary event: OpenCode and Pi
     need the slot.
6. **P2: effort values that arrive.**
   - Codex: `none` through `xhigh` (no `max`), and only with
     `model_supports_reasoning_summaries`.
   - Pi: `none` when thinking is off.
   - Hermes: the widest set, including `max`.
   - OpenCode: whatever a variant says, and `low` on its title call.

   `tiers.py:84` already maps `none` and `off`.
7. **P2 (Hermes): side-call timing.** Hermes aux calls ignore keepalives as
   progress and allow 60 s to the first substantive delta. A compaction
   summary that thinks longer than that before streaming text is cut off.
   Reasoning deltas count as progress; `response.in_progress` does not. If
   `YAMADORI_RESPONSES_REASONING=off`, compaction has only keepalives until
   its text starts. The operator can raise it with
   `auxiliary.compression.no_progress_timeout` (config).

### Input shapes to accept (all observed or in source)

- Messages with no `type`: `{"role":..., "content": "<string>"}` from Hermes,
  or `content` arrays from Pi and OpenCode. The adapter handles this
  (`469`).
- A leading `developer` item instead of `instructions` (Pi, OpenCode). Codex
  sends **both**: `instructions` and a developer message, then a
  user-environment message.
- Assistant `message` items carrying `id`, `status` and `phase`.
- `function_call` items with or without `id`, with `call_id`, and with
  `namespace` (Codex).
- `function_call_output` whose `output` is a string or an array of
  `input_text` / `input_image` / `input_file`.
- Reasoning items echoed in full by Pi (`id`, `summary`,
  `encrypted_content: null`, `status`, maybe `content`) and by Codex
  (`summary`, `encrypted_content`, maybe `content`).
- Tool `strict: false` (Hermes, OpenCode, Codex) or absent (Pi).
- `parallel_tool_calls` true (Hermes) or false (Codex).
- `web_search` with `external_web_access` (Codex): ignore it.
- Fields to accept and ignore: `client_metadata` (Codex), `max_output_tokens`
  (OpenCode, Pi; absent from Hermes and Codex), and
  `include: ["reasoning.encrypted_content"]` (everyone).
- `store: false` (everyone). **No harness sends `previous_response_id`**, so
  the stateless design costs nothing for these four.

### Output invariants these clients check

- **`response.created`** with `response.id`, `created_at` and `model`
  (OpenCode schema). Hermes and Codex only need it to exist.
- **Every event carries `sequence_number`.** OpenCode requires it on
  `response.failed`.
- **Message items:** `output_item.added` with the message `id`, then
  `output_text.delta` with `item_id` equal to that `id`, then
  `output_item.done` with the full content. Hermes builds output from
  `.done` items, and OpenCode ends a text part on `.done`.
- **Function calls:** `output_item.added` with `id`, `call_id`, `name` and
  `arguments: ""`, then argument delta/done, then `output_item.done` with
  the full `arguments`. Codex reads `.done`. Pi and OpenCode use `added` to
  open the call.
- **Terminal:** `response.completed` with `response.id` and `usage`
  (`input_tokens`, `output_tokens`, `total_tokens`, and
  `input_tokens_details.cached_tokens` as integers), or `response.failed`
  with `error.code` (for example `context_length_exceeded`).
- **Keepalives** must be parsed events, not SSE comments. The adapter's
  `response.in_progress` every 1 s works for Hermes' main-turn watchdog and
  Codex's 300 s idle timer. It does not count as progress for Hermes' aux
  calls (gap 7).

### Hosted tools

- No harness above sends `image_generation` to a custom provider. Codex
  offers it only with ChatGPT (codex-backend) auth, the provider capability
  and `Feature::ImageGeneration` all together (`spec_plan.rs:298-306`).
- Codex sends `web_search`. The adapter ignores it, which is correct.
