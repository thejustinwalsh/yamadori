# OpenAI API conformance audit

(The Anthropic Messages API -- `/v1/messages`, for Claude Code, 2026-09-29 --
has its own page: docs/ANTHROPIC-CONFORMANCE.md.)

2026-09-25. READ-ONLY audit (operator: "audit the OpenAI API and bring us up
to conformance"); no code was changed. The premise it is measured against:
any OpenAI-compatible client adds ONE model and gets the stack with no
configuration. So a gap matters in proportion to the clients that hit it.

**Method.** Our code paths read at the lines cited (`mcp/server.py`,
`mcp/proxy.py`, `mcp/streaming.py`, `mcp/catalog.py`, `mcp/tiers.py`).
The spec: `openai/openai-openapi` `main` (`openapi.json`, `info.version`
2.3.0, pushed 2026-09-25) and the API reference guides. Clients:
Hermes from its local checkout (`hermes-agent` at `ee5ee84`); OpenCode (Vercel
AI SDK `@ai-sdk/openai-compatible`), Codex CLI, Continue, Cline and Pi from
their public source (section 9 names what was verified and what was not).
**Live probes:** 24 requests through `:1234` on 2026-09-25 with the dev key
-- 9 route/error probes that never reach the model, 15 chat requests
(10 generated, 5 refused before generating) at `reasoning_effort: minimal`,
`max_tokens` <= 64 -- with no benchmark running (checked first). Each
"observed" below is one of them, n=1: a shape, not a measurement.

Priority: **P0** breaks a target client (Hermes, OpenCode, Pi; Codex once the
Responses adapter exists). **P1** a spec violation clients may hit. **P2**
nice to have. **Won't do** says why. Fix size: S (< 1 h, one place), M (a
day, several places plus tests), L (a new surface).

---

## Status: fixes 1-3 landed (2026-09-25, same day; operator: "Before V0 fixes should land")

E1, E2, C1, U1 and U2 are **done**, plus what came with them: N1 (`n > 1`
refused), `logprobs`/`top_logprobs`, audio output, `input_audio`/`file`
parts and non-function tool forms refused with 400, unknown `/v1` routes
404, and the `include_usage` chunk (section 3). Code: `mcp/api_errors.py`
(new: the one error path, request validation, the overflow wording),
`mcp/server.py` (`chat`, `_first_chunk`, `v1_unknown`), `mcp/proxy.py`
(`check_client_prompt`, `fit_window`, `_usage_of`, `stream_body`, `_run_turn`),
`mcp/streaming.py` (`UpstreamError` carries status + error object,
`usage_chunk`). The rules are in `AGENTS.md` ("The client's own window",
"One error path", "Usage").

**Decisions made in the fix** (each a choice, not a measurement):

- *Commit point.* `server.chat` pulls the stream's first chunk before it
  returns the StreamingResponse; `proxy.stream_body` yields nothing until
  `_run_turn`'s first event -- the first upstream generation answered
  (llama-server itself sends no header before its first result, and returns
  an HTTP error when that result is one), or deep thinking's first heartbeat.
  So a failure in `prepare`, the context check, `ensure_room` or the first
  upstream call is a real HTTP status; long pre-main work still commits early
  and keeps its heartbeats; a failure after that is ONE `data: {"error":
  {...}}` event then `[DONE]`. A client that hangs up before the first byte
  cancels the token (`_first_chunk` watches `http.disconnect`). Waiting for
  the headers instead of the first body bytes costs a client nothing: Hermes'
  stale detector and httpx's read timeout both run from the request.
- *An upstream in-stream error event* (`data: {"error"}` after deltas) now
  raises with the server's error object instead of being turned into a
  "dropped" partial answer with `finish_reason: incomplete`. A plain
  connection DROP with text in hand still returns the partial with
  `incomplete` -- that is F1 (step 4), not done here.
- *Refused vs ignored fields:* the list is in `api_errors.py` (REFUSED /
  ACCEPTED AND IGNORED). Ignored: `user`, `safety_identifier`, `metadata`,
  `store`, `service_tier`, `prediction`, `verbosity`, `web_search_options`,
  `seed`, `functions`/`function_call`; sampling fields stay overridden.
- *The limit.* `context_length` (the main share, 132,096 today) covers prompt
  AND generation; a prompt is refused when it leaves less than A_MIN +
  MIN_THINKING (3,072; A_MIN alone without thinking). Counted with the model
  server's `/apply-template` + `/tokenize` only near the edge (a high
  estimate first: chars / 3 plus half the extra UTF-8 bytes), so an ordinary
  request pays no round trip. A compaction's limit is its budget window. The
  landing survives only for OUR hops (hop 1+).
- *Usage.* The final main generation's prompt and completion; hidden hops
  have no spec field (their output is inside that prompt) and their numbers,
  with the old sums, are in `x_yamadori.usage`. `reasoning_tokens` is
  counted with `/tokenize` (llama-server does not report it), absent if the
  count fails. Usage is sent on a stream only with `include_usage`.

**Evidence.** Offline: `mcp/test_openai_conformance.py` 58/58 (every error
case blocking and streamed, the SSE event against openai-python's own rule,
the threshold at count + floor exactly, hop 0 never landed, usage final vs
summed); the full suite 68/68 (11,103 checks). Tests changed because they
asserted the old behaviour: `test_stream.py` (usage summed -> final; three
landing tests' `context_full` counters, since hop 0 no longer asks it; the
fake answers `/tokenize`), `test_gpu_room.py` (a pre-byte refusal is raised
and answered 507, not an SSE event on a 200), `test_fanout_delivery.py`
(`usage.hops` -> `x_yamadori.usage`), `test_tools.py` (two new
`x_yamadori` keys), `test_live_stack.py` `tokens` (compares the ledger with
`x_yamadori.usage.summed`). Live, through `:1234` (n=1 each):

| probe | before (old proxy, same day) | after (18:03 restart) |
|---|---|---|
| bad JSON | 400 `{"error": "bad request: ..."}` | 400, four-field object, `invalid_json` |
| missing `messages` | 502 `{"error": "HTTPError: HTTP Error 500 ..."}` | 400 `param: messages` |
| `n: 2` | 200, `"pongpong"` (audit) | 400 `param: n` |
| POST `/v1/embeddings` | 405 Method Not Allowed | 404 `unknown_url` |
| a 163,007-token prompt (window 132,096) | LANDED: tools withdrawn, LANDING_PROMPT appended, 162,168 prompt tokens processed, 419 s | 400 `context_length_exceeded`, "maximum context length is 132096 tokens ... resulted in 163007 tokens", 2.7 s blocking / 3.1 s streamed, HTTP error before any byte |
| streamed, `include_usage` | usage on the finish chunk | finish chunk, then `choices: []` + usage `{prompt 925, completion 65, cached 414, reasoning 59}` |

`test_live_stack.py --only conformance`: 7/7. The multi-generation usage
check rides on the `images` group: a `yama_generate_image` turn of 3 generations
reported `prompt_tokens` 2,292 (the final one) where the old sum was 4,780,
with `cached_tokens` 1,844 and `reasoning_tokens` 36. `scripts/deploy_check.py`
after the restart: **DEPLOY GOOD** (exit 0; `test_live_stack.py` 150/150,
3/3 live suites; the known `test_hint_collapse` expected failure only).
`docs/SELF-IMPROVEMENT-LOG.md` #43.

**Hermes, from its source (`hermes-agent` at `ee5ee84`).** A 400 is
classified by message first (`error_classifier._classify_400`): our wording
hits `_CONTEXT_OVERFLOW_PATTERNS` ("maximum context", "context length",
"reduce the length") -> context_overflow -> compress, and avoids every
earlier rule (no "not supported"/"unsupported", no reasoning-field token, no
output-cap phrase such as "requested"/"in the completion", which would make
it shrink max_tokens instead). The window is read back from "maximum
context length is (\d+)" and only lowers a larger configured window. The
streamed error event reaches Hermes as openai-python's status-less
`APIError(body=error)`; it is not one of `_SSE_CONN_PHRASES` (no
"connection"/"terminated" wording in ours), so `_handle_stream_error` does
not retry it as a drop: it propagates to `classify_api_error`, where
`_by_error_code` maps `context_length_exceeded` to overflow (compress) and
any other code falls to the message rules. **Not observed in a live Hermes
run yet** -- V0 will be the first.

---

## Status: R1 landed -- `POST /v1/responses`, stateless (2026-09-26)

Operator, 2026-09-26: "If everyone supports responses api... both need to
exist"; Responses is the primary path harnesses should use. Built as
section 10 planned -- a TRANSLATION over the one turn, no second pipeline:
`mcp/responses_api.py` (`to_chat`: the request -> the chat body
`_run_turn` takes; `of_chat`: the blocking result -> a Response; `Stream`:
`proxy.stream_body`'s own chunks -> the Responses events), and in
`mcp/server.py` one serving path for both routes (`_serve_turn`:
admission, the retry supersede, the E1 commit point, cancel on
disconnect). So everything chat has, Responses has: sessions
(`prompt_cache_key` > headers > our id in the call ids), the ledger and
warms, tiers and caps, skills, deep thinking and its fold-back, the code
check / repair and its notes, the image guard, image tools, usage (U1/U2)
and `x_yamadori` (plus `x_yamadori.responses`: items and tools ignored,
reasoning restored or not, store requested). Two hooks outside the new
module: `proxy._run_turn` puts the request's `_image_options` on the
session state and `_run_our_tool` hands them to `images.run_tool`
(`options`: the hosted image tool's size).

**Decisions made in the build** (each a choice, not a measurement):

- *Reasoning display* (`YAMADORI_RESPONSES_REASONING`, default `summary`).
  The turn's reasoning goes out as a reasoning item's `summary` and the
  `reasoning_summary_*` events -- what Codex, Hermes, Pi and the AI SDK
  display by default. On input only `content[].reasoning_text` is given to
  the model (pass-through: what an echoing client sends); `summary[]` is
  display text and `encrypted_content` is opaque -- both counted in
  `x_yamadori.responses`, neither restored. (Since 2026-09-27 the LEDGER
  restores the slot's own reasoning for a turn that arrives without any,
  on both wires: AGENTS.md "Past reasoning is restored".) Effect: Codex (which echoes
  what it was shown; codex-rs models.rs serializes a summary always) and
  any client that replays the item behave as NON-echoing clients; Hermes'
  codex transport replays a reasoning item only when it carries
  `encrypted_content` (ours is null), so it sends none -- like Hermes over
  chat (the reference run's form;
  AGENTS.md "The ledger"). `content` emits `reasoning_text` instead, and
  Codex then echoes it and the model reads it back (6-10k tokens of context
  per step in the V0 pilot); `off` emits no reasoning item. Section 10
  proposed `content`; changed because echoing costs context and the
  measured reference run did not echo -- the operator's to revisit.
- *Hosted tools.* `web_search`, `file_search`, `code_interpreter`,
  `computer_use`, `mcp`, `tool_search`: ACCEPTED AND IGNORED (the model
  never sees them; recorded in `x_yamadori.responses.tools_ignored`), not
  400 as section 10 proposed: Codex offers `web_search` to an unknown model
  by default (`models-manager` `model_info_from_slug`: `web_search_tool_type:
  Text`), so a 400 would end every Codex session. Client-run built-ins we
  cannot render (`local_shell`, `shell`, `apply_patch`, any unknown type)
  are 400 naming the type. `namespace` tools are FLATTENED (Codex 0.133
  sends `multi_agent_v1` on every request; docs/HARNESS-RESPONSES.md gap
  1): each function under its own name, or `<namespace>__<name>` on a
  clash; a call comes back with `namespace` set and its own name (Codex
  routes by both), and a replayed namespaced call renders as the flat
  name. `custom` (freeform) tools are
  served: a function with one string parameter `input` (a grammar goes into
  the description), a call comes back as `custom_tool_call` with
  `custom_tool_call_input.delta/.done`.
- *The hosted `image_generation` tool is SERVED* (operator scope addition,
  2026-09-26): our `yama_generate_image` (the same engine, gpu_room, media
  store, signed links, account model preference -- turbo by default) with
  the tool's `size` winning over the model's; the image comes back as an
  `image_generation_call` item (`result`: the PNG as base64,
  `revised_prompt`, `size`, `output_format: png`, and `x_yamadori.url`:
  our signed link). `quality` accepted and does not pick the model (same
  rule as the Images API); `output_format` other than png, `background:
  transparent`, `action: edit`, `input_image_mask` and unsupported sizes
  are 400 naming the field. With no image server configured the tool is
  ignored and recorded. Streamed, the item goes out the moment the image
  exists (IMAGES REACH THE CHAT): `output_item.added`,
  `response.image_generation_call.completed`, `output_item.done`. NOT
  emitted: `.in_progress`, `.generating`, `.partial_image` -- the image is
  made inside a hidden hop, the stream learns of the call when it has
  completed, and a partial it never had is not invented. The image's
  markdown line also stays in the message text (it is what the client
  stores, so what the ledger keys the turn by); an image shown after
  content closes the message item and opens a new one, and the replay's
  consecutive assistant items merge back into ONE assistant turn.
- *Sessions under a changing key* (coordinator's decision, 2026-09-26):
  Hermes' Responses `prompt_cache_key` changes at a compaction, so a
  conversation named by a key also carries an id of ours in its call ids
  and on its compaction summaries' line; a NEW key (no session header)
  continues a conversation as an alias only on our SUMMARY LINE
  (compaction evidence); our id in tool-call ids alone is a fork -- a new
  conversation, `forked_from` (AGENTS.md, sessions).
- *Keep-alive.* A chat heartbeat becomes `response.in_progress` with a
  compact snapshot (id, status, model, output `[]`), at most once a second.
  NOT an SSE comment: Hermes' codex transport gives up after 12 s without a
  PARSED event on a small context (`codex_runtime.py`
  `HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS`) and Codex after 300 s
  (`stream_idle_timeout_ms`); the decoders drop comments before either
  timer sees them.
- *One system message.* The served template refuses a system message
  anywhere but first ("System message must be at the beginning"), so
  `instructions` and every system/developer item BEFORE the first other
  item join into one system message; a later one (Codex's mid-session
  developer messages) becomes a user message, counted in
  `x_yamadori.responses.developer_as_user`.
- *Finish mapping.* `length` -> `status: incomplete`,
  `incomplete_details.reason: max_output_tokens` (`response.incomplete`);
  a dropped upstream with the partial kept (F1's `incomplete`) ->
  `status: failed`, `error.code: server_error` (`x_yamadori_code:
  upstream_dropped`), the partial in the output -- not `incomplete`: Codex
  reads every `response.incomplete` as an error, Pi every one but
  `max_output_tokens`, and Hermes would CONTINUE it as a length cut
  (docs/HARNESS-RESPONSES.md gap 3); a failure after the first byte ->
  `response.failed` with
  `error.code` in the codes Codex branches on (`context_length_exceeded`,
  `rate_limit_exceeded`, `server_is_overloaded`, `invalid_prompt`, else
  `server_error`; ours in `x_yamadori_code`). No flat `error` event, no
  `[DONE]`: clients end on the terminal event.
- *Usage*, ALWAYS an object of integers on the terminal response (zeros
  when the turn reported none: the AI SDK drops a terminal event whose
  usage does not parse, and OpenCode then records finish `other`).
  `input_tokens` = the final generation's prompt (U1),
  `input_tokens_details {cached_tokens, cache_write_tokens: 0}`,
  `output_tokens`, `output_tokens_details {reasoning_tokens}` -- LEFT OUT
  when the reasoning count is unknown (never guessed; the spec marks it
  required, Codex reads it as optional).
- *Refused (400 `unsupported_parameter`):* `previous_response_id`,
  `conversation`, `background: true`, `prompt`, `item_reference` items --
  each needs server state. `store: true` is accepted and answered `store:
  false`. `input_file` in a message is 400 (as chat's `file` parts); in a
  `function_call_output` it becomes a note (refusing would wedge the
  conversation: it is in every later request). Hosted-call records in the
  input (`web_search_call`, `image_generation_call` -- Codex replays the
  image's base64 -- `compaction`, ...) are ignored and do not split the
  turn they sit in.

**How state would map onto the ledger, when it is built.** A stored
response is the turn `ledger_record_turn` already keeps (the delivered turn
under the conversation's chain key; its hidden hops expanded on replay). A
`previous_response_id` would name that row, but resolving it means
rebuilding the chat messages up to and including that turn, and the
caller's own input items of earlier requests are NOT in the ledger -- only
their hashes are (nebari's rule: "Not kept: the caller's code"). So state
needs a new per-account store of callers' input items by response id,
under the ledger's size-based eviction. That is a decision about keeping
callers' content -- the operator's. Until then: 400 with "send the whole
conversation in `input`".

**Evidence.** Offline: `mcp/test_responses_api.py` (every item type
translated, the 400s, a Codex-shaped request with a `view_image` output
and an `--image` attachment, blocking and streamed multi-turn clients that
replay every output item and keep the slot's prefix through the SERVED
template, the carrier ids without `prompt_cache_key`, the stream read by
openai-python's decoder rule and checked against the spec's required
fields per event, the SDK helper's accumulation and Codex's item parser,
the route through FastAPI, the hosted image tool with a real PNG from a
fake generator, the Images API options). `mcp/test_openai_conformance.py`
no longer lists `/v1/responses` among the 404s; `bench/octopus/
test_relay_responses.py` gates the relay's Responses collector. Live,
2026-09-26 11:35 proxy restart, idle card, n=1 each, through `:1234`
(`mcp/test_live_stack.py --only responses`: 7/7, 288 s): a plain answer
(completed, usage), the same streamed (created, in_progress, ...,
completed; deltas = the message), a `get_weather` function-call round trip
replayed Codex-style (the second request: session `prompt_cache_key`,
cache reused 976 of 1,044 prompt tokens, the answer "sunny and 18°C"), an
`input_image` (an attachment of the chat image path; `yama_describe_image` ran,
4.8 s; answer "Red"), the hosted `image_generation` tool (an
`image_generation_call` item with a PNG, streamed), `previous_response_id`
400. **Not yet run with a real Codex, Hermes `codex_responses`, OpenCode or
Pi.**

---

## 0. The short list

| # | gap | priority |
|---|---|---|
| E1 **DONE** | Every failure on the STREAMED path is sent as assistant *content* (`"\n[upstream error: ...]"`) on a 200, then `[DONE]` -- no `finish_reason`, no error event. An exception outside the turn aborts the socket. Hermes retries it as a drop; Pi, Cline and the current AI SDK raise "ended without finish_reason"; nobody sees the real error or compacts on an overflow. | **P0** |
| C1 **DONE** | A conversation past the main share is "landed": the CLIENT's tools are withdrawn and a user message the client never sent is appended; the client gets a prose answer, never `context_length_exceeded`. The threshold is a chars/3 over-estimate plus reserved answer and thinking room, so it can fire before the advertised `context_length`. An agent loop sees its tools vanish and the task end. | **P0** (agent loops) |
| U1 **DONE** | `usage.prompt_tokens` is SUMMED over the turn's hops. Harnesses read it as the context size (Hermes `context_compressor.py:2766`) and compact early on every multi-hop turn. | P1 (high) |
| E2 **DONE** | Non-streamed errors: bad JSON 400 and every upstream error (502) carry `{"error": "<string>"}`, not the error object; upstream 4xx (a client error) become 502; a missing `messages` is a 502 "HTTP Error 500". | P1 |
| F1 | `finish_reason: "incomplete"` (a dropped upstream) is not in the enum; Pi treats any unknown finish as an error; in Hermes' vocabulary it marks a Codex interim turn. | P1 |
| N1 **DONE (refused, 400)** | `n > 1` is passed upstream, llama-server returns n choices, and the reader concatenates them into ONE message (observed: `"pongpong"`). | P1 |
| J1 | JSON mode is protected only while the request is a utility call (no tools, one exchange). Otherwise the proxy may add text to `content` (session line, notes, fold-back, budget notices) and break `json_object`/`json_schema`. A `length` notice is appended even in utility mode. | P1 |
| R1 **DONE (2026-09-26, stateless; see Status: R1)** | `/v1/responses` does not exist (and returns 405). Codex CLI has REMOVED `wire_api = "chat"`, so without it Codex cannot use the stack at all. | P1 (P0 once Codex is a target) |
| U2 **DONE** | `prompt_tokens_details.cached_tokens` and `completion_tokens_details.reasoning_tokens` are dropped (Hermes, OpenCode, Cline and Pi all read `cached_tokens`); a non-standard `usage.hops` is added. | P1 (cheap) |

Everything else is P2 or won't do (sections 1-8).

---

## 1. Chat Completions: request fields

What happens to a field: **used** (the proxy reads it), **passed** (reaches
llama-server as sent; `tiers.apply` starts from `dict(body)`,
`tiers.py:741`, and only `_`-keys are stripped, `proxy.py:343`), **dropped**,
or **overridden**.

| item | spec says | we do | gap | client impact | fix | priority |
|---|---|---|---|---|---|---|
| `messages` roles `system`/`developer`/`user`/`assistant`/`tool` | all five; `function` deprecated | passed; `developer` renders (observed: a developer instruction obeyed); an assistant turn with `content: null` + `tool_calls` and a `tool` result as a parts array both work (observed) | none | -- | -- | -- |
| `messages` missing / not a list / body not an object | 400 `invalid_request_error`, `param: "messages"` | no validation. Missing `messages`: **502** `{"error":"HTTPError: HTTP Error 500: ..."}` (observed). A JSON array body: **500 `text/plain` "Internal Server Error"** (observed; `server.py:313` indexes a list) | wrong status, wrong shape | a client bug reads as our outage; SDKs retry 5xx | S: validate in `server.chat` before admission | P1 |
| content parts `text`, `image_url` | both; `image_url.url` may be http(s) or a data URI | `text` passed; images become placeholders read by `yama_describe_image` (`vision.extract`, `proxy.py:2007`); http(s) image URLs are deliberately NOT fetched (`vision.py:41`), the model asks for the file | URL images unsupported | a client that sends a hosted image URL gets "please send the file" | -- | won't do (SSRF: the server would fetch arbitrary URLs); say so in the model card, which already does (`catalog.py:231`) |
| content parts `input_audio`, `file` | allowed | passed to llama-server, which rejects -> 502 | wrong status | none of the targets send them to a text model | S: 400 `unsupported_content_type` naming the part | P2 |
| `model` | required | any name resolves (`catalog.resolve`, unknown -> `yamadori`); the reply echoes the name sent (`server.py:326`, `catalog.rewrite_response`) | none (lenient by design) | -- | -- | -- |
| `max_tokens` / `max_completion_tokens` | upper bound on generated tokens; `max_completion_tokens` INCLUDES reasoning; `max_tokens` deprecated | both read as an ANSWER allowance, floored at `A_MIN` = 2,048 (`tiers.py:592`); thinking added on top; `max_completion_tokens` read when `max_tokens` is absent and never sent upstream (`tiers.py:746-749`). Observed: `max_completion_tokens: 3` -> 111 completion tokens, `finish_reason: stop` | the limit is not a limit | a client capping a title or a probe gets more text and waits longer; nothing breaks in the targets | S: honour the client's number exactly when thinking is off (`minimal`, utility calls), where the floor protects nothing | P2 (the floor is a deliberate budget rule, `AGENTS.md` "One budget rule"; keep it for thinking tiers and say so -- section 8) |
| `n` | 1-128 choices | passed; llama-server returns n choices; `_post_events_raw` appends every choice's delta to ONE buffer (`proxy.py:427-436`). Observed `n: 2` -> `"content": "...pongpong"`, one choice | silent corruption | anyone sending `n > 1` (none of the targets) | S: 400 `n must be 1` (fan-out is ours, and it is not `n`) | P1 |
| `stop` | string or up to 4 strings | passed (observed: `stop: ["3"]` -> `"1 2 "`) | proxy-added text (J1) can follow a stop | -- | see J1 | -- |
| `temperature`, `top_p` (+ llama.cpp `top_k`, `min_p`, `repeat_penalty`) | client-chosen | **overridden** with the vendor's values (`tiers.enforce_sampling`, `tiers.py:711-729`) | deliberate; see section 8 | -- | -- | won't do (operator, 2026-09-23) |
| `presence_penalty` | client-chosen | overridden (0.0 thinking, 1.5 instruct) | as above | -- | -- | won't do |
| `frequency_penalty`, `logit_bias` | client-chosen | **passed** -- not in `VENDOR_SAMPLING`, so NOT enforced, although the catalogue says sampling is | inconsistent with the policy | a client's penalty changes our sampling | S: add to the enforced set, or document as honoured | P2 |
| `seed` | best-effort determinism (now marked deprecated) | passed; fan-out can deliver another candidate (`catalog.py:195`) | not determinism | -- | -- | won't do (documented in the catalogue) |
| `logprobs`, `top_logprobs` | returns `choices[].logprobs` | passed; llama-server refuses because the proxy always streams upstream AND always adds its own tools: **502** `"logprobs is not supported with tools + stream"` (observed), on the streamed path an error-as-content (E1) | unsupported, reported as our outage | no target client asks for logprobs | S: 400 `unsupported_parameter` with the reason; M to support on the no-tools path | P2 |
| `response_format` `text` / `json_object` / `json_schema` | enforced shape | passed; llama-server compiles a grammar. A request with no client tools and one exchange is a utility call: bare model, nothing added (observed: `{"word": "pong"}`, 19 prompt tokens). See J1 for the rest | J1 | -- | -- | -- |
| `tools` (`function`) | list | the client's, untouched and first; ours appended (`yama_generate_image`, `yama_describe_image`, `yama_think_deeply` at xhigh/max; `proxy.main_tools`) | our tools cost ~900 prompt tokens on every non-utility request (observed: a 6-word prompt = 899 prompt tokens at `minimal`) and appear in `usage` | cost only | -- | by design |
| `tools` (`custom`), `tool_choice: {type: "allowed_tools"}` | newer tool forms | passed; llama-server does not know them | unsupported | no target sends them | S: 400 naming the form | P2 |
| `tool_choice` `none`/`auto`/`required`/`{type:function,function:{name}}` | forces | passed (observed: `required` and the object form both returned the named call, `finish_reason: tool_calls`) | `required` can be satisfied by one of OUR tools, which the client never sees | theoretical | S: when the client forces, send only the client's tools on that hop | P2 |
| `parallel_tool_calls` | bool | passed (llama-server honours it) | none | -- | -- | -- |
| `stream`, `stream_options.include_usage` | usage only when asked, in a final chunk with `choices: []` | usage always sent, in the finish chunk (section 3) | placement | none observed | S | P2 |
| `reasoning_effort` | `none`/`minimal`/`low`/`medium`/`high`/`xhigh`/`max`, default `medium` | picks the TIER (`tiers.resolve`); `none` -> `minimal`; `max`, `ultra` accepted; unknown -> `medium` silently; OpenRouter's `reasoning.effort` too (`tiers.py:410-414`) | an unknown value is not a 400 | none (lenient on purpose: clients send vendor extensions) | -- | -- |
| `prompt_cache_key` | cache routing hint | **used** as the conversation id (`session_id.cache_key_of`); suppresses the minted session line | none -- the best use of the field there is | -- | -- | -- |
| `user`, `safety_identifier`, `metadata` | identifiers / tags | passed, ignored | none | -- | -- | -- |
| `store` | store for distillation/evals | passed, ignored; there is no `GET /v1/chat/completions/{id}` | no stored completions | none of the targets reads them back | -- | won't do |
| `service_tier` | `auto`/`default`/`flex`/`scale`/`priority`/`fast` | passed, ignored; response has no `service_tier` | none that matters | -- | -- | won't do |
| `modalities: ["audio"]`, `audio` | audio output | passed, ignored: text only | silent | no target asks for audio | S: 400 when `audio` is in `modalities` | P2 |
| `prediction`, `verbosity`, `web_search_options` | hints / hosted search | passed, ignored (deep thinking has its own search, `docs/SEARCH.md`) | silent no-op | -- | -- | won't do |
| `functions`, `function_call` (deprecated) | legacy tools | passed; llama-server ignores them | silent | none of the targets | -- | won't do |
| llama.cpp-native fields (`grammar`, `id_slot`, `cache_prompt`, `n_probs`, `samplers`, `chat_template_kwargs`, `reasoning_format`, ...) | not in the spec | **passed** (only `id_slot` is overwritten when a slot is granted, `proxy.py:265`; `chat_template_kwargs` merged, `tiers.py:779`) | a client can steer the server below the proxy -- turn thinking off, pick a slot, change the reasoning parser | not a target-client issue; a correctness and isolation one | M: an allowlist of spec fields + the extensions we mean to honour | P2 |

## 2. Chat Completions: the response object

Observed non-streamed at `minimal` (`"pong"`):
`{id, object: "chat.completion", created, model: "yamadori", choices: [{index, message: {role, content, reasoning_content}, finish_reason}], usage: {prompt_tokens, completion_tokens, total_tokens, hops}, x_yamadori}`.

| item | spec says | we do | gap | client impact | fix | priority |
|---|---|---|---|---|---|---|
| `id`, `object`, `created`, `model` | required | present (upstream's id, `proxy.py:370`; model = the name sent) | none | -- | -- | -- |
| `system_fingerprint`, `service_tier` | optional | absent | none | -- | -- | -- |
| `message.refusal` | required, nullable | absent | typed clients expect the key | a strict decoder that requires it; none of the targets (Hermes reads it with `getattr`, `chat_completions.py:631`) | S: `"refusal": null` | P2 |
| `message.annotations`, `message.audio` | optional | absent | none | -- | -- | -- |
| `message.reasoning_content` | not in the spec (de-facto: DeepSeek, llama.cpp, vLLM) | always present, `""` when empty | extension | read by Hermes, the AI SDK and Pi as reasoning (section 9) | -- | keep |
| `choices[].logprobs` | required, nullable | absent | as `refusal` | as `refusal` | S: `"logprobs": null` | P2 |
| `choices` length | `n` choices | always 1; `n > 1` corrupts (N1) | N1 | -- | -- | -- |
| `finish_reason` | `stop`, `length`, `tool_calls`, `content_filter`, `function_call` | `stop`, `length`, `tool_calls`; plus **`incomplete`** when the upstream connection dropped with text in hand (`proxy.py:484`, `4881`) | F1: outside the enum | Pi treats any finish other than stop/end/length/tool_calls/function_call as an ERROR. Hermes lower-cases and keeps it (`message_sanitization.py:313`), and in its own vocabulary a message with `finish_reason == "incomplete"` is a Codex interim turn, replayed verbatim (`agent_runtime_helpers.py:372-378`) -- the wrong branch for a dropped chat answer | S: report `length` (or `stop`) and keep the drop in `x_yamadori` and the notice | P1 |
| a `length` finish | content truncated, `finish_reason: length` | a bracketed notice appended to `content` (`proxy._budget_notice`, `proxy.py:3059`) | the notice becomes part of the client's history; breaks JSON mode (J1) | a harness stores `[answer cut off ...]` as the model's words | S: no notice in JSON mode; consider moving it to `x_yamadori` only | P1 (JSON) / P2 |
| `usage.prompt_tokens` | the request's input tokens | the SUM over every upstream generation of the turn (`proxy.py:4715-4718`, `5064-5074`): hidden hops (image tools, `yama_think_deeply`), a fan-out continuation | U1: double- or triple-counts the context on a multi-hop turn | **Hermes** sets `last_prompt_tokens = usage.prompt_tokens` (`context_compressor.py:2766`) and compacts on it: a multi-hop turn reads as a context 2-3x its size. OpenCode (AI SDK) and Cline also budget from it | S: report the LAST generation's prompt tokens (the context the conversation now occupies); keep the sum in `x_yamadori` | P1 |
| `usage.completion_tokens` | generated tokens incl. reasoning | sum over hops (includes reasoning) | fine as a sum | -- | -- | -- |
| `usage.prompt_tokens_details.cached_tokens` | optional | dropped (only three keys are kept, `proxy.py:4716`), though `x_yamadori.cache.reused` has the number | U2 | read by Hermes (`extract_cache_stats`, `chat_completions.py:670`), OpenCode and Cline (AI SDK) and Pi: all show zero cache hits on a stack whose design is the cache | S: fill from the cache record | P1 (cheap) |
| `usage.completion_tokens_details.reasoning_tokens` | optional | absent | U2 | OpenCode shows reasoning tokens separately | S-M: llama-server's own count if present, else tokenize the reasoning | P2 |
| `usage.hops` | not in the spec | added | an unknown key inside a typed object | none observed (the AI SDK's zod objects strip unknown keys) | S: move to `x_yamadori` | P2 |
| `x_yamadori` (top level, ~3.8 KB) | not in the spec | added on every response | extension (section 7) | none: unknown top-level keys are ignored by the OpenAI SDKs and the AI SDK | -- | keep |
| content the proxy adds | content is the model's | the session line `yamadori session <12 hex>` on a new conversation's first answer (observed on every probe without `prompt_cache_key`), the repo preamble, `Verified`/`Repaired`/`Checked` notes, fold-back openings, image lines, bracketed notices | extension (section 7), J1 | a harness stores and displays them; JSON mode breaks | see J1 | P1 (JSON) |

## 3. Streaming

Observed (`minimal`, `include_usage`): chunk 1 `delta: {content: "yamadori session ...\n\n"}`, chunk 2 `delta: {content: "pong"}`, chunk 3 `delta: {}`, `finish_reason: "stop"`, `usage` and `x_yamadori` on the same chunk, then `data: [DONE]`. `Content-Type: text/event-stream; charset=utf-8`.

| item | spec says | we do | gap | client impact | fix | priority |
|---|---|---|---|---|---|---|
| first chunk | every delta field is optional; OpenAI's own sequence opens `delta: {role: "assistant", content: ""}` | never sends `role` (`streaming.chunk`, `streaming.py:49-62`; `proxy.stream_body`) | not a violation; a deviation from what OpenAI sends | none of the targets needs it (Hermes accumulates; the AI SDK has `role` nullish; Pi needs none). Accumulator helpers that copy `role` from the first delta store none (openai-node's `ChatCompletionStream` helper is believed to throw on a missing role -- not re-read for this audit) | S: emit `{role: "assistant", content: ""}` first, before the preamble and the session line. Never send any other role: the AI SDK 2.0.41 zod enum accepts only `assistant` | P2 |
| `id`, `object`, `model` per chunk | the same on every chunk | yes (`streaming.new_id`) | none | -- | -- | -- |
| `created` | the same on every chunk | re-read per chunk (`streaming.py:55`) | cosmetic | none | S | P2 |
| reasoning | not in the spec | `delta.reasoning_content`, and never after the first `content` delta (CHANNEL ORDER, `proxy._Out`) | extension | Hermes, the AI SDK and Pi read it (section 9) | -- | keep |
| heartbeats | not in the spec | `delta: {}` chunks while the second brain works | legal (an empty delta is a valid chunk) | none | -- | keep |
| tool calls | incremental `delta.tool_calls[{index, id, type, function{name, arguments}}]` | the client's calls sent WHOLE, in one chunk at the end, after the check (`proxy.py:6034-6038`) | legal: a client assembles by `index` | none | -- | keep (the check needs the whole call) |
| `finish_reason` | on the last choice chunk | on the chunk with `usage` | none | -- | -- | -- |
| usage | only with `include_usage`; a separate LAST chunk with `choices: []`; `usage: null` on the others | always sent, on the finish chunk (`proxy.py:6050`) | placement; sent unasked | none observed: Hermes and the AI SDK read usage from any chunk; a client that asserts `choices == []` on the usage chunk would miss it | S: split into a `choices: []` chunk when asked | P2 |
| terminator | `data: [DONE]` | yes (`streaming.DONE`) | none | -- | -- | -- |
| **errors** | before the first byte: an HTTP error status and the error object. After it: `data: {"error": {...}}` (the OpenAI SDKs raise on an `error` key in an SSE event) | the 200 is committed BEFORE `prepare` and the first upstream call (`server.py:328-344`). An upstream refusal or failure becomes `content: "\n[upstream error: UpstreamError: ...]"` then `[DONE]`, with no `finish_reason` (`proxy.py:4703-4713`, `6046`). Observed: a `logprobs` request -> exactly that. Only `TurnRefused` gets a proper `data: {"error": ...}` (`proxy.py:6014-6021`). Any other exception (in `prepare`, in the preamble) escapes the generator and the socket is cut | E1 | **Hermes** (streams by default): a stream with text and no `finish_reason` is "a mid-stream drop" (`chat_completion_helpers.py:3212`) -> a partial stub and a retry of a DETERMINISTIC error; a context overflow is never recognised, so it never compacts. **Pi**: throws "Stream ended without finish_reason". **Cline** and the current AI SDK (3.x): raise "ended without a finish reason". **OpenCode** (AI SDK 2.0.41): stores the bracketed text as the assistant's message with finish `other`. Every client: the real error (type, code) is lost | M: (1) run `prepare` and open the first upstream generation BEFORE returning the StreamingResponse (peek the first event), so a refusal is a real status with the error object; (2) after that, send `data: {"error": {message, type, code}}` then `[DONE]`; (3) map upstream 4xx to 400 and overflow to `context_length_exceeded` (E2) | **P0** |

## 4. Errors and status codes

| item | spec says | we do | gap | client impact | fix | priority |
|---|---|---|---|---|---|---|
| error object | `{"error": {"message", "type", "param", "code"}}`, all four required (`param`/`code` nullable). The spec documents 429 (`rate_limit_error`) and 503 (`service_unavailable_error`, `server_is_overloaded`) on this route, each with `Retry-After`; the SDKs map 400/401/403/404/409/422/429/5xx to typed errors and retry 408/409/429/5xx twice, honouring `Retry-After` up to 120 s | right on 401, 404/405 (`server.openai_shaped_errors`), 429 (`param` missing), `TurnRefused`, images. WRONG (`{"error": "<string>"}`) on: bad JSON 400 (`server.py:307`, observed), every non-streamed turn failure 502 (`server.py:358-360`) | E2 | the OpenAI SDKs fall back to the raw body; the AI SDK's error schema fails and it shows the raw text; `error.code`/`type` are lost, so no client can branch on them | S | P1 |
| upstream 4xx | the client's error: 400 with the reason | 502 on the blocking path (`UpstreamError`, `proxy.py:463-465` -> `server.py:358`); content on the streamed path (E1) | wrong class | SDKs retry 5xx (openai-python: 2 retries by default), so a deterministic error is sent three times | S: `UpstreamError` carries the status; 400 + llama-server's own `message`/`type` | P1 |
| context overflow | 400, `code: "context_length_exceeded"` | never returned. Below the pool, the proxy LANDS the request instead (C1, next row); above it, llama-server's 400 becomes 502 / content | C1 + E2 | harnesses compact on this code or message: Hermes matches "context length"/"maximum context" in any 400/5xx message (`error_classifier.py:253`, `591`) -- which works today only on the blocking path, by accident of the message text | S (once E1/E2 exist) | P1 |
| **C1 the landing on a client's own conversation** | the request is served as sent, or refused | `context_full` (`proxy.py:3087-3114`) estimates the prompt at chars/3 (`tiers.estimate_prompt_tokens`, a deliberate OVER-estimate) and, when prompt + answer + 1,024 >= the main share, `_land` (`proxy.py:3140-3176`) withdraws ALL tools -- the client's included -- drops `tool_choice` and appends the user message `LANDING_PROMPT` ("Stop searching and answer now ...") | the client never sent that message; its tools vanish; nothing says so in the response except `x_yamadori` | an agent loop gets a prose answer where it expected a tool call, and ends the task. It can fire BEFORE the advertised `context_length` (132,096 observed): chars/3 over JSON-serialised messages and tools is an over-estimate by design (the size of the error is not measured), and the answer allowance and 1,024 thinking tokens are reserved on top. A client that compacts near its configured window reaches the landing first; Hermes compacts at a threshold below the window and is safer | S: when the prompt is the CLIENT's (hop 0, no hop of ours in it), return 400 `context_length_exceeded` instead of landing; keep landing for our own hops. M: count with `/tokenize` near the edge | **P0** (agent loops) |
| 401 | `invalid_request_error`, `code: invalid_api_key` | yes (observed); `param` missing | none that matters | -- | S | P2 |
| 403, 404 | object | yes | none | -- | -- | -- |
| unknown POST route | 404 | **405 "Method Not Allowed"** for every unknown POST path (`/v1/responses`, `/v1/embeddings`, `/v1/completions`, observed): the SPA's GET catch-all `/{rest:path}` (`dash_static.py:261`) owns the path | wrong status | a client probing `/v1/responses` or `/v1/embeddings` reads "wrong method" where the truth is "no such endpoint" | S: a POST catch-all under `/v1/` returning 404 `unknown_url` | P2 |
| 429 | `rate_limit_error`, `Retry-After` | yes, `Retry-After: 30`, `code: server_busy` (`server.py:126-139`); no `x-ratelimit-*` headers | none that matters | -- | -- | -- |
| 5xx | `server_error` / `api_error` | 502 for every turn failure; 507 for "no room on the A4000" (`TurnRefused.of_no_room`) | 507 is unusual but a 5xx | clients retry | -- | -- |
| `/v1/models` authentication | Bearer required | none (`server.py:164`), unlike the rest of `/v1` | lenient, not a conformance break; contradicts the proxy's own "every route authenticates" comment | -- | S | P2 |

## 5. Models

| item | spec says | we do | gap | client impact | fix | priority |
|---|---|---|---|---|---|---|
| `GET /v1/models` | `{object: "list", data: [{id, object: "model", created, owned_by}]}` | exactly that, one row `yamadori`, plus OpenRouter/vLLM-style metadata: `context_length`/`max_model_len`/`context_window` (132,096 observed), `max_completion_tokens`/`max_output_tokens`, `architecture` (OpenRouter's `input_modalities`/`output_modalities`/`modality`), `modalities: {input, output}` (models.dev's shape, the same lists; since 2026-09-26, docs/VISION.md 4a), `pricing`, `supported_parameters`, `x_yamadori` (`catalog._chat_card`). Both modality fields say `["text","image"]` only while vision is available -- `YAMADORI_VISION` not `0` AND `bonsai-vision` in llama-swap's config with `--mmproj` (`catalog.vision_configured`) -- else `["text"]`; same on `/v1/models/{id}` | extension only. No target harness reads modalities from a custom endpoint (VISION.md 2a): configuration is the switch (VISION.md 4c) | this is what lets Hermes size its window with no configuration; Continue and Cline list models from it (`data[].id`); OpenCode does not call it (models come from its config / models.dev) | -- | keep |
| `GET /v1/models/{id}` | the model, or 404 | never 404: an unknown id gets the `yamadori` row, with `id: "yamadori"` (observed for `gpt-4o`, `catalog.model_card`) | the returned id differs from the one asked | a client that verifies `id == requested` would reject it; none of the targets does | -- | won't do (deliberate: any name is served) |
| `supported_parameters` | OpenRouter's vocabulary | `max_tokens, max_completion_tokens, tools, tool_choice, reasoning, reasoning_effort, response_format, stop, stream` (`catalog.py:201`) | `parallel_tool_calls` is honoured and not listed; `seed` is listed nowhere and passed | cosmetic | S | P2 |
| `yamadori-embed`, `yamadori-rerank` | -- | resolve in the CHAT catalogue (`catalog.INTERNAL`), so a chat request naming them goes to the embedding server | a footgun the catalogue's own docstring warns about | none unless named | S: resolve them only on an embeddings route | P2 |

## 6. Other endpoints

| endpoint | spec | we do | client impact | fix | priority |
|---|---|---|---|---|---|
| `POST /v1/embeddings` | `{input, model, encoding_format, dimensions}` -> `{object: "list", data: [{object: "embedding", index, embedding}], model, usage}` | absent (405, section 4). The embedder exists (`embeddings` in llama-swap, used through `code_search._post` inside `gpu_room.use`) | Continue, when configured with an OpenAI-compatible embeddings provider (`POST {apiBase}/embeddings {input: string[], model}`, reads `data[].embedding`; it has a local default otherwise); Hermes, OpenCode, Codex, Cline, Pi do not call it | M: authenticated pass-through inside `gpu_room.use("embeddings")`, `encoding_format` both ways, `dimensions` refused (not a Matryoshka model unless it is) | P2 |
| `POST /v1/completions` (legacy) | still in the spec: `prompt`, `suffix`, `echo`, `logprobs` (int), `best_of` -> `text_completion` | absent Continue's autocomplete with provider `openai` goes to `/completions` by default (its `/fim/completions` path is not OpenAI's and is off for that class; inferred from code, not traced); nothing else in the target set | M-L: FIM needs the model's FIM tokens, a separate lane and no tier logic | P2 / won't do until someone asks for autocomplete |
| `POST /v1/responses` | the Responses API (section 10) | **served since 2026-09-26** (`mcp/responses_api.py`, stateless; Status: R1) | **Codex CLI** (section 9) | -- | done |
| `POST /v1/images/generations` | `{prompt, n, size, response_format, model, quality, output_format, background, style, moderation, output_compression, stream, partial_images, user}` -> `{created, data: [{url|b64_json, revised_prompt}], output_format, size, background, quality}` | implemented (`server.images_generations`). Checked against the spec 2026-09-26: `quality` accepted (it does NOT pick the image model: the caller's saved preference does, turbo by default -- a client that always sends `high` would otherwise override the operator's choice; echoed), `output_format` png only (jpeg/webp 400: nothing converts), `background` opaque/auto (transparent 400: ask for it in the prompt, real alpha is kept), `stream` 400, `model` picks ours only by our names (`yamadori-image[-turbo]`; `dall-e-3`, `gpt-image-1` ignored), `n` 1-4 (spec 1-10: 400 above 4), sizes per `images.parse_size` (1792x1024 400 with the remedy); `style`, `moderation`, `output_compression`, `user`, `partial_images` ignored; bad JSON is the one error object (`invalid_json`); the response now carries `output_format`, `size`, `background` and the requested `quality`; `usage` is not sent (token counts do not apply) | none | -- | -- |
| `GET /v1/chat/completions/{id}`, `/v1/files`, `/v1/batches`, `/v1/audio/*`, `/v1/moderations` | -- | absent | no target calls them | -- | won't do |

## 7. Our extensions, and whether they are spec-safe

| extension | where | spec-safe? |
|---|---|---|
| `x_yamadori` top-level object | every chat response and the final stream chunk; `/v1/models` rows; image responses | **Yes.** Unknown top-level keys are ignored by the OpenAI SDKs (pydantic `extra` allowed) and stripped by the AI SDK's zod schemas. It is ~3.8 KB per response (observed); fine for the wire. |
| `message.reasoning_content` / `delta.reasoning_content` | every response | **De-facto standard** (DeepSeek, llama.cpp, vLLM). OpenRouter spells it `reasoning`; clients that read only `reasoning` see none (section 9). |
| `usage.hops` | inside `usage` | **Mostly.** An unknown key inside a typed object; tolerated by every client checked. Belongs in `x_yamadori` (U2). |
| extra model-card fields | `/v1/models` | **Yes** (OpenRouter/vLLM conventions); chosen so Hermes cannot mistake one for another (`catalog.py:143-154`). |
| `error.retryable`, `error.remedies`, facts | `TurnRefused`, image errors | **Yes.** Extra keys inside `error` are ignored. |
| request header `X-Yamadori-Features`, `X-Yamadori-Session` | benchmark arms, explicit sessions | **Yes.** Headers, never body fields. |
| request `reasoning` object (OpenRouter) | `tiers.resolve` | **Yes** (accepted, not required). |
| model names `yamadori-fast/-xhigh/-max/-vision` | `catalog.INTERNAL` | **Yes**; unadvertised. |
| **text the proxy adds to `content`** | session line, repo preamble, `Verified`/`Repaired`/`Checked` notes, fold-back openings (`Today I was inspired by ...`, `After thinking deeply,`), `Compared two approaches`, image lines, bracketed budget/drop/empty/upstream-error notices | **Syntactically yes, semantically no.** The client stores these as the model's words and replays them. That is the product's design (`AGENTS.md`, "Fold-back phrases") for the notes and fold-backs; it is NOT acceptable for (a) JSON mode (J1), (b) errors (E1), (c) the `length`/drop notices, which duplicate `finish_reason`. |

**J1, JSON mode.** `selection.utility_call` makes a `response_format`
request a utility call only with no client tools and one exchange
(`selection.py:747-765`). A json request with history or tools is an
ordinary turn, and the session line (first turn only), notes, fold-back
prefills and a fan-out prose continuation can all land in `content`; a
`length` notice lands even on a utility call. Fix (S): when `response_format`
is `json_object`/`json_schema`, add nothing to `content` -- no lead, no
notes, no fold-back prefill, no fan-out continuation, no notices -- and
report through `finish_reason` and `x_yamadori` only. **P1.**

## 8. Where the proxy overrides the client, and how that is surfaced

| override | what the client sent | what runs | surfaced today | recommendation |
|---|---|---|---|---|
| sampling | `temperature`, `top_p`, `presence_penalty` (+ `top_k`, `min_p`, `repeat_penalty`) | the vendor's values, thinking or instruct (`tiers.py:711-729`) | `x_yamadori.sampling.{enforced, client_overridden}`; the catalogue omits them from `supported_parameters` | Keep. It is spec-permitted (the parameters are hints to a provider that serves one model). Add a response header `X-Yamadori-Overridden: temperature,top_p` so a client log shows it without parsing the body; do NOT write it into content. `frequency_penalty` and `logit_bias` are NOT enforced today -- decide, then make the code match |
| token limit | `max_tokens` / `max_completion_tokens` | an answer allowance >= 2,048 plus a separate thinking budget (`tiers.budget`) | `x_yamadori.budget`; the model card's `answer_allowance_floor` | Keep for thinking tiers (the empty-reply bug it fixed is real). Honour the number exactly where thinking is off. State the rule in the model card's `description` |
| tier | `reasoning_effort` | a bundle: thinking effort AND augmentations; a utility call is forced to `minimal` | `x_yamadori.tier`, `x_yamadori.utility.tier_overridden` | Keep |
| tools | the client's list | client's + ours | `x_yamadori.tools`; never in the response's `tool_calls` (ours are run server-side and withheld, `proxy.py:4866-4874`) | Keep; see the `tool_choice: required` row |
| the landing (C1) | the client's conversation and tools | tools withdrawn, a user message appended | `x_yamadori.tool_turns` / the log only | Replace with `context_length_exceeded` for the client's own prompt (section 4) |
| model | any name | `bonsai` | the response echoes the name sent | Keep |
| images by URL | `image_url` http(s) | not fetched | the model says so; the model card says so | Keep |

## 9. What the target clients send and expect

Hermes from the local checkout; the rest from their public repositories on
2026-09-25 (OpenCode `anomalyco/opencode` `dev` with
`@ai-sdk/openai-compatible@2.0.41`; Codex `openai/codex` `main`; Continue
`continuedev/continue`; Cline `cline/cline` with
`@ai-sdk/openai-compatible` 3.x; Pi `earendil-works/pi`, formerly
`badlogic/pi-mono`). "Unverified" marks what was inferred and not traced.

| client | sends | relies on | breaks today on |
|---|---|---|---|
| **Hermes** (chat_completions transport) | `max_tokens`; `stream: true` + `include_usage`; `reasoning_effort` or OpenRouter `reasoning` via `extra_body`; `prompt_cache_key` only where the endpoint is marked capable (`chat_completions.py:71-110`); strips its own sidecar keys and `tool_calls: []` | `choices` non-empty; `usage.prompt_tokens` as the context size for compaction (`context_compressor.py:2766`); `prompt_tokens_details.cached_tokens`; `finish_reason` present (else "drop", `chat_completion_helpers.py:3188-3220`); `reasoning_content`; error text patterns for overflow in 400 AND 5xx (`error_classifier.py:253`, `591`); `context_length`/`max_completion_tokens` from `/v1/models` | E1 (retries deterministic errors, never compacts on a streamed overflow), U1 (early compaction), U2 (no cache stats), C1, F1 |
| **OpenCode** (AI SDK openai-compatible 2.0.41, patched only to pass the error object through) | `max_tokens` (<= min(model limit, 32,000)); `stream_options.include_usage` (forced on); `reasoning_effort` from variants (sent by default: unverified); `response_format` `json_schema` or `json_object`; `tools`, `tool_choice`; echoes past reasoning as `reasoning_content` | loose top-level schema (unknown keys fine; `id`/`created`/`model` nullish); `delta.role` must be `assistant` or absent; the first delta of a tool index must carry `id` and `function.name` (ours do: calls go whole); `reasoning_content ?? reasoning`; `cached_tokens`, `reasoning_tokens`; a top-level `error` key in a chunk becomes a stream error; no `/v1/models` call | E1 (error text stored as the answer; finish `other`), C1 (compacts near its configured window, so it meets the landing first), U2 |
| **Codex CLI** | Responses only: `wire_api = "chat"` now fails to parse ("no longer supported"; deprecated 2025-12-11, PR #7897; removal "slated for early February 2026", release not verified). Sends `instructions`, `input`, flat function `tools`, `tool_choice: "auto"`, `parallel_tool_calls`, `reasoning {effort, summary}`, `store: false`, `stream: true`, `include: ["reasoning.encrypted_content"]`, `prompt_cache_key` (always), `text {verbosity, format}`; replays reasoning items (with `encrypted_content`, possibly null). Never sends `max_output_tokens` or `temperature` | `response.created` (id), `response.output_item.added`/`.done` (final items; a `function_call` needs `call_id`, `name`, `arguments`), `response.output_text.delta`, `response.reasoning_text.delta` / `reasoning_summary_*`, `response.custom_tool_call_input.delta`; `response.completed` REQUIRED (with `response.id`; `usage` ints if present); `response.failed` with `error.code` (`context_length_exceeded`, `server_is_overloaded`, `rate_limit_exceeded`, ...); ignores `content_part.*`, `function_call_arguments.*`, `in_progress`, `output_text.done` | nothing known offline since R1 (2026-09-26); not yet run with a real Codex |
| **Continue** | openai SDK; `max_tokens`, `temperature`, `top_p`, penalties, `stream` (default on) + `include_usage`, `stop` (not truncated to 4 for a non-OpenAI host), `prediction`, `tools` (with `strict`), `tool_choice`; no `reasoning_effort` on the chat path; echoes `reasoning_content` only when its provider flag is set | `/v1/models` (`data[].id`); `delta.reasoning_content || delta.reasoning`; the openai SDK's error handling | `/v1/completions` for autocomplete and `/v1/embeddings` for indexing when configured (both absent: 405); E1/E2 through the SDK |
| **Cline** (SDK, AI SDK openai-compatible 3.x) | `include_usage`; `max_tokens`; the AI SDK request mapping | a `finish_reason` (3.x raises without one); `delta.role` `assistant`/`""`/null; `cached_tokens`; `/v1/models` on settings refresh | E1, U2 |
| **Pi** (openai SDK, `openai-completions.ts`) | for an unknown host: `max_completion_tokens`, `include_usage`, `store: false`, `reasoning_effort`, `developer` role for reasoning models, `tool_choice` only if set, `tools: []` when history has calls but none are offered; `prompt_cache_key` only for api.openai.com | a `finish_reason` in {stop, end, length, tool_calls, function_call} -- anything else, or none, is an error; `reasoning_content`/`reasoning`/`reasoning_text`, echoed back in the field it came in; `cached_tokens ?? prompt_cache_hit_tokens`, `reasoning_tokens`; a top-level `error` key or `event: error` throws (openai SDK) | E1 (throws), F1 (error), U2. `tools: []` with a history of calls: llama-server's handling of an empty list is not verified here |

What all six agree on, and what we already do right: the OpenAI chunk shape,
`data: [DONE]`, whole tool calls with ids, `reasoning_content`, unknown
top-level keys (`x_yamadori`) tolerated, `developer` rendered, `/v1/models`
(for the ones that read it), `prompt_cache_key` used as the session.

## 10. A minimal stateless `/v1/responses` adapter

For Codex, and for any client that has moved to Responses. Stateless:
`store: false` is the only mode served; the conversation arrives whole in
`input` every time, exactly like chat, so the ledger, slots, tiers and every
feature keep working. It is a TRANSLATION LAYER over the one turn
implementation (`proxy._run_turn`'s events: reasoning, content, heartbeat,
calls), in `server.py` beside `chat`, not a second pipeline.

**Request -> the chat body `_run_turn` already takes**

| Responses field | becomes | note |
|---|---|---|
| `instructions` | a leading `system` message | must render byte-identically every turn (the cache) |
| `input` string | one `user` message | |
| `input` `message` items (`user`/`system`/`developer`/`assistant`; `input_text`, `output_text`, `input_image` parts) | chat messages, parts mapped (`input_image` -> `image_url`) | `phase` ignored |
| `function_call` items `{call_id, name, arguments}` | `tool_calls` on the preceding assistant message (consecutive calls merged into ONE assistant message) | the merge must be deterministic, or the ledger's chain hashes change between turns |
| `function_call_output` `{call_id, output}` | `tool` messages | |
| `reasoning` items | the next assistant message's `reasoning_content` (from `content[].reasoning_text`, else `summary[]`), or dropped | operator decision: pass-through says "what the client sends is what the model sees"; Codex replays what we emitted, so emitting the reasoning as `content: [{type: "reasoning_text"}]` makes Codex an ECHOING client |
| flat function `tools` `{type, name, description, parameters, strict}` | nested `{type: "function", function: {...}}` | built-in tools (`web_search`, `file_search`, `computer_use`, `mcp`, ...) -> 400 naming the tool type |
| `custom` tools (Codex's freeform `apply_patch` on some model families -- which families: unverified) | a function with one string parameter `input`; a call comes back as a `custom_tool_call` item | needed only if Codex sends them to a non-OpenAI provider; check with a live Codex run |
| `tool_choice`, `parallel_tool_calls` | same fields (`{type: "function", name}` -> the nested form) | |
| `reasoning.effort` | `reasoning_effort` (the tier dial) | `reasoning.summary` accepted and ignored |
| `max_output_tokens` | `max_tokens` | Codex never sends it |
| `text.format` `{type: "json_schema", name, schema, strict}` | `response_format` in the chat (nested) form | `text.verbosity` ignored |
| `prompt_cache_key` | kept (the conversation id; Codex always sends one, so no session line) | |
| `store: true` | accepted, not stored (`"store": false` in the response) -- or 400; operator's choice | |
| `previous_response_id`, `conversation`, `background` | 400 `unsupported_parameter`: "this server is stateless; send the whole input" | Codex does not use them with `store: false` |
| `include` | ignored except `reasoning.encrypted_content` (see output) | |
| `temperature`, `top_p` | enforced as in chat (section 8) | echoed back as the values used |
| `metadata`, `user`, `safety_identifier`, `service_tier`, `truncation` | accepted, ignored; `metadata` echoed | |

**Response object** (non-streamed, and the payload of the terminal event):
`id: "resp_<hex>"`, `object: "response"`, `created_at`, `status`
(`completed`, or `incomplete` with `incomplete_details: {reason:
"max_output_tokens"}` on a `length` finish), `error: null`, `model` (the name
sent), `output` in order -- a `reasoning` item (`id`, `summary: []`,
`content: [{type: "reasoning_text", text}]`, `encrypted_content: null`), a
`message` item (`id`, `role: "assistant"`, `status: "completed"`, `content:
[{type: "output_text", text, annotations: [], logprobs: []}]`), then one
`function_call` item per client call (`id: "fc_<hex>"`, `call_id` = the chat
tool-call id, `name`, `arguments`, `status: "completed"`) -- plus the
required echoes (`instructions`, `tools`, `tool_choice`,
`parallel_tool_calls`, `metadata`, `temperature`, `top_p`, `reasoning`,
`text`, `store`), `usage: {input_tokens, input_tokens_details:
{cached_tokens, cache_write_tokens: 0}, output_tokens,
output_tokens_details: {reasoning_tokens}, total_tokens}` (with U1/U2
fixed first), and `x_yamadori`.

**Streaming** (`stream: true`): SSE as `event: <type>\ndata: <json>\n\n`,
every event with an increasing `sequence_number`, no `[DONE]` (clients end
on the terminal event). The minimum a spec-correct stream emits, in order:

1. `response.created`, then `response.in_progress` (the response with `status: "in_progress"`, `output: []`).
2. Reasoning, while `_run_turn` yields `reasoning`: `response.output_item.added` (reasoning item), `response.reasoning_text.delta` per chunk, `response.reasoning_text.done`, `response.output_item.done`.
3. Text, on the first `content`: `response.output_item.added` (message), `response.content_part.added` (`output_text`, empty), `response.output_text.delta` per chunk (with `logprobs: []` -- the schema requires it), then `response.output_text.done`, `response.content_part.done`, `response.output_item.done`.
4. Calls, on `calls` (whole, as today): per call `response.output_item.added`, one `response.function_call_arguments.delta` with the whole string, `response.function_call_arguments.done`, `response.output_item.done`.
5. Terminal: `response.completed` (or `response.incomplete`), carrying the full response object and usage.
6. Errors: before the first byte, an HTTP status and the chat error object (E1's fix gives this for free); after it, `response.failed` with `response.error: {code, message}` using OpenAI's codes (`context_length_exceeded`, `server_is_overloaded`, `rate_limit_exceeded`) -- Codex maps exactly these -- then close. A flat `error` event is optional (Codex ignores it; openai-node throws on it).
7. Heartbeats (`_run_turn`'s `heartbeat`): an SSE comment line (`: keepalive`), which SSE parsers discard. That Codex's parser discards it: unverified -- check in the live test.

CHANNEL ORDER already guarantees reasoning before content, which is the
item order Responses wants. Size: **L** (a module of ~400 lines, offline
tests replaying Codex's own SSE fixtures from `codex-rs`, and a live test:
`codex exec` against `:1234` with a one-file task).

## 11. Proposed implementation order

Ordered so each step makes the next one's failures visible. Per
`AGENTS.md`, each ships with an offline test AND a live check in
`mcp/test_live_stack.py` (cheap deterministic triggers exist: `n: 2` and
`logprobs: true` are refused before generating), and `deploy_check.py` after
the restart. Nothing here changes sampling, tiers or the cache design.

1. **DONE 2026-09-25 (see Status).** **One error path (E1, E2, request validation).** M. An `ApiError(status,
   type, code, message, param)` used everywhere; `UpstreamError` carries
   llama-server's status and body (4xx -> 400 with its message and type;
   llama-server's context-size refusal -> 400 `context_length_exceeded`; its
exact type string, believed to be `exceed_context_size_error`, was not
reproduced in this audit).
   `server.chat` validates the body (an object, `messages` a non-empty list,
   `n` in {1, null}) before admission. Streamed: run `prepare` and wait for
   the first upstream event BEFORE returning the StreamingResponse, so a
   refusal is a real status; after the first byte, `data: {"error": {...}}`
   then `[DONE]`, never content. An exception anywhere in the stream ends in
   that event, not a cut socket. `logprobs`, `input_audio`/`file` parts,
   `modalities: ["audio"]`, custom/allowed-tools forms: 400 naming the field.
   A POST catch-all under `/v1/` -> 404. Fixes the P0 for Hermes, Pi, Cline,
   OpenCode.
2. **DONE 2026-09-25 (see Status; the model server's count near the edge).** **The client's own overflow (C1).** S, after 1. On hop 0, when the prompt
   is the client's (no hop of ours in it), `context_full` returns 400
   `context_length_exceeded` instead of landing; landing stays for our own
   hops. Decide with the operator whether a `/tokenize` count replaces the
   chars/3 estimate near the edge (it is the difference between "refused at
   the advertised window" and "refused early").
3. **DONE 2026-09-25 (see Status).** **Usage (U1, U2).** S. `prompt_tokens` = the last generation's prompt;
   `completion_tokens` = the sum; `prompt_tokens_details.cached_tokens` from
   the cache record; `completion_tokens_details.reasoning_tokens` (llama-
   server's count, or tokenize the reasoning); `hops` and the per-hop sums
   into `x_yamadori`. Check against Hermes' compaction on a multi-hop turn.
4. **Finish reasons and stream framing (F1, first-chunk role, usage chunk).**
   S. No `incomplete`: a drop with text in hand becomes the error of step 1
   (streamed: the partial text, then the error event; blocking: 502
   `server_error` with the partial text in `x_yamadori`) -- or, if the
   operator prefers keeping the partial as an answer, `length`. Emit `{role:
   "assistant", content: ""}` first; one `created` per stream; with
   `include_usage`, usage on a final `choices: []` chunk and `usage: null`
   elsewhere; `refusal: null` and `logprobs: null` in the object.
5. **JSON mode (J1).** S. With `response_format` json, the proxy adds nothing
   to `content` and reports only through `finish_reason` and `x_yamadori`.
6. **`/v1/responses` (R1).** L. Section 10, on top of steps 1-4 (its error,
   usage and finish mapping reuse them). Then a live Codex run.
7. **P2 cleanups.** An allowlist of request fields (spec + the extensions we
   honour: `reasoning`, `chat_template_kwargs`? -- decide); decide
   `frequency_penalty`/`logit_bias` (enforce or document); an
   `X-Yamadori-Overridden` response header for enforced sampling; honour
   `max_tokens` exactly where thinking is off; `tool_choice: required` sends
   only the client's tools; `/v1/models` behind the key (operator's call:
   it is public today); `supported_parameters` made exact; `yamadori-embed`
   / `-rerank` out of the chat catalogue; then `/v1/embeddings` as a
   pass-through if Continue indexing is wanted.
8. **Won't do** (and why): stored completions / `store` retrieval, audio,
   `prediction`, `web_search_options`, `service_tier`, legacy `functions`,
   client sampling values (enforced by design), `seed` determinism (fan-out),
   fetching image URLs (SSRF), a 404 for unknown model ids (any name is
   served), `/v1/completions` FIM until autocomplete is a goal.
