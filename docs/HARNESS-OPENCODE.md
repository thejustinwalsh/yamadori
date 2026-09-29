# OpenCode against Yamadori

Hands-on test, 2026-09-26. OpenCode **1.18.32** (npm `opencode-ai`, AI SDK
`@ai-sdk/openai-compatible`, runtime bun 1.3.14) drove the live stack
through `:1234`. A recording relay sat in between (`bench/octopus/relay.py`,
wrapped by a throwaway `capture_relay.py`, which also logs content-part
types and sizes, tool-call ids, and a few named headers; it never logs the
key or full base64). All numbers are **n=1** unless a row says otherwise.

Evidence: `C:\Users\jwals\octo\opencode-test\logs\` (`relay.1.jsonl` has
experiments 1, 3a and 5; `relay.2.jsonl` has 3b, 4, 2, 5a and the fork;
`e3a/e3b/e4.events.jsonl` are OpenCode's own `--format json` events), the
corpus (`index/corpus.sqlite3` `events`, 2026-09-26 14:05-14:33 UTC), and
OpenCode's isolated DB (`octo/opencode-test/home/data/opencode/opencode.db`).

**The running proxy is yesterday's code.** PID 25928 started 2026-09-25
18:03. `mcp/tool_code.py` and `mcp/proxy.py` were last written 2026-09-26
09:10 and 09:15. The tool-call heartbeat (#44) and the image guard (#46) are
"BUILT, NOT deployed" (`docs/SELF-IMPROVEMENT-LOG.md`), so this test does
not exercise either. The session carrier (`session_id.py`, 2026-09-25
15:29) is live.

## The config used

The config lives in its own place, apart from the user's global config.
Environment for every run: `XDG_CONFIG_HOME`, `XDG_DATA_HOME`,
`XDG_CACHE_HOME` and `XDG_STATE_HOME` point under
`C:\Users\jwals\octo\opencode-test\home\`. `OPENCODE_CONFIG` points at the
file below. `OPENCODE_DISABLE_AUTOUPDATE=1`, `OPENCODE_DISABLE_SHARE=1` and
`OPENCODE_DISABLE_CLAUDE_CODE=1` are set, and `YAMADORI_OPENCODE_KEY` holds
the key, read from a file. `opencode debug paths` confirmed data, config,
cache and state are all under that directory.

This is the recommended file (`cfg/full.json`). Every key in it is justified
by a finding below.

```json
{
  "$schema": "https://opencode.ai/config.json",
  "autoupdate": false,
  "share": "disabled",
  "model": "yamadori/yamadori",
  "small_model": "yamadori/yamadori",
  "permission": { "edit": "allow", "bash": "allow", "webfetch": "deny" },
  "provider": {
    "yamadori": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Yamadori",
      "options": {
        "baseURL": "http://127.0.0.1:1234/v1",
        "apiKey": "{env:YAMADORI_OPENCODE_KEY}",
        "setCacheKey": true
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

In the tests, `baseURL` was the relay (`http://127.0.0.1:18235/v1`). The
first runs used the minimal file: the same provider with only
`"models": {"yamadori": {"name": "Yamadori"}}`.

Headless runs:

- `opencode run --format json [-f FILE] [--session ID | --continue] [--fork] [--variant EFFORT] -- "prompt"`
- `-f` attaches a file.
- `--` is needed after `-f`, because `-f` takes an array.
- Under Windows PowerShell 5.1, keep double quotes out of the prompt. The
  native-argument passing split `"hello from opencode"` into fragments, and
  experiment 1's prompt reached the model mangled. That is our fault, not
  OpenCode's.

## 1. What OpenCode sends, and sessions

### Every request

| field | value |
|---|---|
| `model` | `yamadori` |
| `stream` | `true` |
| `stream_options` | `{"include_usage": true}` |
| `max_tokens` | 32000 with no `limit.output`; 26419 with it |
| `tools` | 9: `bash edit glob grep read skill task todowrite write` (20,473 bytes). `tool_choice: "auto"` |
| system prompt | one system message, 9,572 chars; about 7,940 prompt tokens with the tools |
| `reasoning_effort` | none by default (tier `medium`). Sent only with `--variant` |
| `temperature`, `top_p`, `user`, `metadata`, `prompt_cache_key` | none |
| headers | `User-Agent: opencode/1.18.32 ai-sdk/provider-utils/4.0.23 runtime/bun/1.3.14`, plus `x-session-affinity` and `X-Session-Id`, both set to OpenCode's session id (`ses_` + 26 chars) |
| past turns | echoes `reasoning_content` on every assistant turn (for example 125, 33, 61 chars) |

The session headers are on **every** request, the title call included. They
come from the binary, for any provider not named `opencode*`:
`{"x-session-affinity": sessionID, "X-Session-Id": sessionID}`.

### Title call

A new session also sends a **title** side call first:

- system `You are a title generator...` (2,096 chars);
- two user messages;
- no tools.

The proxy served it as a utility call (tier `minimal`, sessionless,
transient slot 3, 0.8-3 s). It is recorded as
`utility_kind: "compaction"` (form `summarise_conversation`), which is wrong:
it is a title request. The only effect is on records. FIXED 2026-09-26 (not
yet live): the utility rule checks a `title` form first
(`selection.names_a_title`), so it is `utility_kind: "title"`;
`mcp/test_utility.py` replays the corpus rows (6580, 6584, ...).

### Session identity (`x_yamadori.session`)

| case | what reached the proxy | `session.source` | id |
|---|---|---|---|
| text-only question | nothing explicit | `minted` | f73d3a4dd673 |
| same OpenCode session, 2nd `run --session`, text answer | history with our text answer | `answer_record` | f73d3a4dd673 (kept) |
| same, 3rd `run --continue`, tool call then answer | our ids echoed | `answer_record`, then `tool_call_id` | f73d3a4dd673 (kept) |
| tool task (hello.py) | our ids `call_4b1a6b4fe4f0_*` echoed verbatim on both sides | `minted`, then `tool_call_id` on every step | 4b1a6b4fe4f0 |
| `setCacheKey: true` | body key **`promptCacheKey`** (camelCase), not `prompt_cache_key` | `minted` | new each time |
| `--fork` of the hello.py session | a NEW OpenCode session id in `X-Session-Id`, forked history | `tool_call_id` | **4b1a6b4fe4f0: the original's** |

- **Our carrier works with OpenCode.** OpenCode echoes `tool_calls[].id` and
  `tool_call_id` verbatim. The session held across separate `opencode run`
  processes, and the cache followed it. The hello.py steps reused 7950/8100,
  8167/8215 and 8289/8306 prompt tokens. The continuation reused 8008/8027.
- **`prompt_cache_key` never arrives.** With `setCacheKey: true`, OpenCode
  puts `promptCacheKey = sessionID` in the provider options. For a custom
  provider it sets the option only with that flag (plain `@ai-sdk/openai`,
  azure, xai, mistral and venice get it by default; deepinfra and cerebras
  get `prompt_cache_key`). `@ai-sdk/openai-compatible` does not map the
  option. It passes unknown provider options through as-is, so the body
  carries `"promptCacheKey": "ses_..."`. The proxy ignores that key and
  forwards it upstream, where llama-server accepted it (200).
- **The fork merges two conversations.** An OpenCode fork is a different
  session with a different `X-Session-Id`. Its history carries the
  original's tool-call ids, so it resolves to the original conversation:
  its ledger, deep-thinking state, work log and slot pin. OpenCode's own id
  would have kept them apart. The fork also gained a 10th tool,
  `question`, so its system block changed, and the pin had been evicted:
  reused 0/8765.

**Proposed fix (proxy, not made).** Take OpenCode's own session id. Every
request already carries it, with no config.

1. In `server.chat` and `proxy.Handler`, when `X-Yamadori-Session` is
   absent, read `X-Session-Id`, else `x-session-affinity`. Validate it with
   `nebari.session_token`: `ses_` + 26 alphanumerics is 30 characters and
   passes. Record it as its own source (for example `client_session`).
2. In `session_id.cache_key_of`, accept `promptCacheKey` as an alias of
   `prompt_cache_key`, and strip it before the body goes upstream.
3. **Guard:** a side call carries the same header. It must stay sessionless
   and off the conversation's slot, or the title call would evict the
   conversation's own cache. Today that holds because utility calls have no
   session. Keep it that way: a utility call ignores the header.

This fixes the fork merge. It also removes the `answer_record` deviation
for OpenCode, whose text-only turns currently depend on it.

## 2. Capability detection

- **OpenCode never calls `/v1/models`.** There was no GET in either relay
  log, across 34 requests from every OpenCode process of the test (and
  `opencode models` sent nothing at all).
  None of our card is read: not `context_length`, not
  `max_completion_tokens`, not `architecture.input_modalities`.
- **It does fetch models.dev.** `home/cache/opencode/models.json`, 4.9 MB,
  appeared at first start. A custom provider id `yamadori` matches nothing
  there, so the model gets these defaults (`opencode models yamadori --verbose`):
  - `reasoning: false`, `attachment: false`;
  - `input.image: false`;
  - `limit: {context: 0, output: 0}`;
  - no variants.
- **`limit.context: 0` disables auto-compaction.** The binary's check
  returns false when `model.limit.context === 0`. An unconfigured OpenCode
  never compacts. It runs into our context landing, or a 400 overflow,
  instead. `limit.output: 0` gives `max_tokens: 32000`, which is above our
  advertised 26,419. The proxy accepted it; the budget sent upstream was its
  own.
- **The config keys** (`provider.<id>.models.<id>`), read from the binary's
  schema and confirmed with `opencode models --verbose`:

  | key | sets | observed |
  |---|---|---|
  | `modalities: {input: [...], output: [...]}` | `capabilities.input.image` | experiment 3 |
  | `attachment: true` | attachment capability | |
  | `reasoning: true` | reasoning capability | turns on built-in variants `low`/`medium`/`high`, and the title call then sends `reasoning_effort: "low"` |
  | `limit: {context, output}` | window and answer cap | `max_tokens` became 26419 |
  | `variants: {name: {reasoningEffort: ...}}` | extra efforts | `--variant max` sent `reasoning_effort: "max"`; tier `max`, effort sent `xhigh` |

  Without `modalities`, attaching an image fails inside OpenCode
  (experiment 3a).

## 3. Vision, user-attached (`-f circle.png`)

The test image was a synthetic 128x128 PNG, a red circle on white, 643
bytes, drawn with PIL for this test.

### 3a. Without `modalities`: OpenCode drops the image

OpenCode drops the image itself. The user message has four text parts:

1. `Called the Read tool with the following input: {"filePath":...circle.png}`
2. `Image read successfully`
3. `ERROR: Cannot read "circle.png" (this model does not support image input). Inform the user.`
4. the question

No image part reached the proxy (`x_yamadori.attachments: []`).
`yama_describe_image` is on main anyway (the image server is configured), so the
model tried it:

1. With the file path. This got `UNKNOWN_IMAGE`, correct by design: it never
   opens a path.
2. It base64'd the file with `bash` (862 chars).
3. The next request ran **427.5 s, then OpenCode dropped it and retried it
   silently** (below, "a silent retry").
4. The retry called `yama_describe_image` with a 1,282-character
   `data:image/png;base64,...` argument. `vision.resolve` refused it with
   `UNKNOWN_IMAGE`, "pass the image's id, not its data". The image guard is
   not deployed, so the call executed.
5. The model then analysed pixels with PowerShell `System.Drawing`, over 3
   failed attempts.

Final answer: "The image contains a red circle." That took 9 main requests (one of them the
dropped attempt) plus the title call, about 680 s.

### 3b. With `modalities.input: ["text","image"]`: works

Same user message, but the error part is replaced by an **`image_url` part
carrying a data URL**: `data:image/png;base64`, 882 chars, `detail`
absent.

- The proxy registered the attachment:
  `attachments: [{"id": "image-39187bf0ad", "source": "attached", "format": "png", "bytes": 643}]`.
- Main called `yama_describe_image("image-39187bf0ad", ...)` in a hidden hop.
- `vision: ok`, 12.0 s, 115 prompt / 83 completion tokens. The vision
  model said "a single red circle centered on a white background".
- Answer: "The image contains a red circle on a white background." It
  took 20.8 s for main and 30 s end to end.

**Side effect, the title call.** The title call carried the same image. A
request carrying an image is never a utility call (AGENTS.md). So the title
call:

- went through the full `medium` pipeline (route `library_question`, 10.1 s);
- **minted its own session** (f5b6e150f6f0);
- **took a pinned slot, evicting another conversation's pin**
  (`cache.evicted: "11a8bfc1"`, slot 1).

That costs one pin per image-bearing session start. FIXED 2026-09-26 (not
yet live; operator decision): a tool-less, single-exchange title or summary
call keeps its utility status when it carries an image -- the image becomes
its placeholder, nothing is described, and the call is sessionless on the
transient slot (`selection.utility_call`; `mcp/test_harness_decisions.py`
opencode/chat/title-with-image).

## 4. Vision, agent self-view (draw `square.png`, then look)

Config: `vision.json`. Prompt: write `make_square.py` using only zlib and
struct, run it with `py`, open `square.png` with the read tool, and say its
colour.

1. `write`: `Verified make_square.py (python): parses, lint clean; sent as written.`
2. `bash` `py make_square.py`: exit 0.
3. Before calling `read`, the model tried `yama_describe_image` with the file
   path in a hidden hop, and got `UNKNOWN_IMAGE`. The tool's description
   tells it to use the client's file tools, and it did.
4. **OpenCode's `read` of a PNG returns the image to the model.** The
   tool message's content is only the text `Image read successfully`.
   OpenCode then appends a **synthetic user message**: a text part,
   `Attached media from tool result:`, and an **`image_url` part**
   (`data:image/png;base64`, 198 chars, the 132-byte PNG). It is neither
   bytes nor base64 as text.
5. The proxy treated it as an attachment (`image-e8ef205b78`,
   `source: attached`). Main called `yama_describe_image` (`ok`, 8.6 s). The
   answer was correct: "solid, fully saturated pure blue".

The image guard did not trip, and had no reason to: no image data went into
a tool argument. The route of the final request was `prose`, because it
ends on a user message.

Cache after the image turn: the final request reused only 7,438 of 9,674
tokens on its first hop, so 2,236 were processed. (Cause found offline
2026-09-26 and fixed, not yet live: the ledger replayed the hidden
`yama_describe_image` hop with its reasoning emptied while OpenCode echoes the
rest exactly; docs/HARNESS-PI.md gap 5, `mcp/test_harness_forms.py`.) That is the length of the
system block plus tools, even though the history before it was unchanged.
The previous request had a hidden `yama_describe_image` hop. This cost was
observed, and its cause is not established. The same shape appeared once in
3a (reused 7,426 of 12,709 after the dropped request).

## 5. What breaks, and conformance as OpenCode sees it

### A silent retry (the #44 failure, reproduced with OpenCode)

OpenCode's `chunkTimeout` is 300,000 ms by default. It is a provider option:
"Timeout in milliseconds between streamed SSE chunks". When no bytes arrive
for 300 s, OpenCode aborts the stream with "SSE read timed out" and **retries
the request silently**:

- nothing at INFO level in its log;
- nothing in its DB (the first attempt's reasoning part is stored empty).

In 3a, request 14 streamed 9,001 reasoning chars, then ended at 427.5 s
with `client_gone: BrokenPipeError`. It carries no `x_yamadori`, and the
corpus has a `turn` event for it and nothing else. OpenCode sent the same
request again 2 s later.

The retry's first action was a `yama_describe_image` call with a
1,282-character base64 argument. The model writes arguments to our tools
without anything streaming to the client, and the running proxy has no
tool-call heartbeat (#44 is not deployed). So the most likely cause is a
long hidden-hop argument, image data, being written in silence past 300 s.
The upstream work was stopped: the relay closes upstream on hang-up, and
the retry got a 200, not a 429. About 7 minutes of GPU time were lost.

- **Fix:** already built (#44 heartbeat, #46 image guard). Once deployed,
  rerun experiment 3a to check them live.
- **Client-side mitigation:** a larger
  `provider.yamadori.options.chunkTimeout`.

### Everything else

| check | result |
|---|---|
| Invalid key | 401 on both the title call and the main call. OpenCode shows `APIError` "unrecognised API key. Set an API key in your client." with `statusCode: 401`, `isRetryable: false`, and the body `{"error": {"message", "type": "invalid_request_error", "param": null, "code": "invalid_api_key"}}`. Exit 1. The duplicated `date` header is the relay's, not ours |
| usage | `prompt_tokens_details.cached_tokens` and `completion_tokens_details.reasoning_tokens` are present. OpenCode shows them (`cache.read`, `reasoning` in its step tokens) |
| finish reasons | `stop` and `tool_calls` map to OpenCode's `stop` / `tool-calls` |
| `x_yamadori` on the final chunk | tolerated |
| any other 4xx/5xx | none in the other 32 requests |
| our notes in the transcript | `Verified <path> (python): parses, lint clean; sent as written.` appears as visible assistant text on every file write, and OpenCode stores and resends it. Expected by design (the ledger keys it). A user reads it as the assistant speaking |
| unknown body key | `promptCacheKey` forwarded upstream, accepted |

**Not tested:** OpenCode's own compaction (it needs `limit.context`, and a
long session), subagents (`task`: a child session with
`x-parent-session-id`), MCP, and the `question` tool.

## OpenCode vs Hermes

Hermes facts are from AGENTS.md, `docs/OPENAI-CONFORMANCE.md` §9 and the
v0e entries (#44-#46) in `docs/SELF-IMPROVEMENT-LOG.md`. OpenCode facts are
from this test.

| | Hermes | OpenCode 1.18.32 |
|---|---|---|
| session id it sends | none to a custom endpoint (`prompt_cache_key` only for a provider marked capable) | `X-Session-Id` / `x-session-affinity` on every request (ignored by us today). `promptCacheKey` (camelCase, ignored) only with `setCacheKey` |
| our tool-call-id carrier | echoed, from source; not observed live | **observed live**: echoed verbatim, the session held across processes |
| fork / branch | n/a | a fork merges into the original conversation (our id is in its history) |
| past reasoning | stripped (no echo) | echoed as `reasoning_content` |
| `/v1/models` | read (`context_length`, `max_completion_tokens`) | never called. Window and modalities must be in config; with none, no compaction and `max_tokens` 32000 |
| `reasoning_effort` | sent | none by default. `--variant`: `low`/`medium`/`high` with `reasoning: true`, and custom variants for `minimal`/`xhigh`/`max` |
| user-attached image | n/a in the Octopus runs | `image_url` data-URL part (needs `modalities`). Placeholder plus `yama_describe_image` works |
| agent looks at its own image | `vision_analyze(image_url=path \| data URL)`: failed 29/29 in v0e (#46, a path translation bug in the harness and pasted base64) | `read` of a PNG, then a synthetic user message with an `image_url` part. Placeholder plus `yama_describe_image` works (n=1) |
| side calls | utility (41/41 in the corpus) | title generator is utility (labelled `compaction`). With an image attached it is not utility: it mints a session and evicts a pin |
| silence tolerance | stale-stream detector killed the stream and retried (#44) | `chunkTimeout` 300 s, then a silent retry (seen once, 427 s) |
| compaction | flattened `TURNS TO SUMMARIZE` (mapped) | not observed; disabled unless `limit.context` is set. Its form ("Here is the conversation so far: <conversation>", read from the binary) is recognised and mapped since 2026-09-26 (offline only) |
| error surfacing | E1 patterns | error object shown with `statusCode` and `isRetryable` (401 checked) |
