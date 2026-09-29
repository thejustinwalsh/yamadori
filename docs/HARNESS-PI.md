# Pi against Yamadori

Hands-on test, 2026-09-26, 14:56-15:17 UTC. Pi **0.87.1** (npm
`@earendil-works/pi-coding-agent`, `repository`
`git+https://github.com/earendil-works/pi.git` `packages/coding-agent`,
author Mario Zechner, MIT) drove the live stack through `:1234`. It ran on
Node 24.21.0 by path, without switching the global node:
`...\nvm\v24.21.0\node.exe ...\@earendil-works\pi-coding-agent\dist\bundle\cli.js`
(the package's `bin.pi`). The same recording relay as the OpenCode test sat
in between: `capture_relay.py`, extended by `pi_relay.py` to log Pi's
affinity headers by value and a few more body keys. It never logs the key or
full base64. All numbers are **n=1** unless a row says otherwise.

Evidence: `C:\Users\jwals\octo\pi-test\`. `logs\relay.jsonl` has 44 rows,
one per request, each with the response's `x_yamadori` verbatim. Rows are
numbered `#0`-`#43` below. `logs\<tag>.out` holds Pi's own `--mode json`
events. `cfg\` has the models.json variants, `sessions\` Pi's session files,
and `run_pi.sh`, `rpc_compact.py` and `show.py` are the drivers. A scan found
the key in none of these files.

**The running proxy is yesterday's code.** It is PID 25928, started
2026-09-25 18:03, the same process the OpenCode test ran against. Two things
in the working tree are built but not deployed, so this test does not
exercise them:

- `mcp/server.py` (written 10:51 today) reads `X-Session-Id` /
  `x-session-affinity` as a session source.
- `mcp/image_input.py` (10:54) recognises Pi's synthetic tool-media turn.

## The config used

Pi's home is isolated. `PI_CODING_AGENT_DIR` and
`PI_CODING_AGENT_SESSION_DIR` point into the test directory. `PI_OFFLINE=1`,
`PI_SKIP_VERSION_CHECK=1` and `PI_TELEMETRY=0` are set. The key is in
`YAMADORI_PI_KEY`, read from a file in the launcher, and `models.json` refers
to it as `$YAMADORI_PI_KEY`. `~/.pi` does not exist and was never created.
The project directory was a throwaway one with no `AGENTS.md` above it.

This is the recommended `<agent-dir>/models.json`. Every key in it is
justified by a finding below.

```json
{
  "providers": {
    "yamadori": {
      "baseUrl": "http://127.0.0.1:1234/v1",
      "api": "openai-completions",
      "apiKey": "$YAMADORI_PI_KEY",
      "models": [
        {
          "id": "yamadori",
          "name": "Yamadori",
          "reasoning": true,
          "thinkingLevelMap": { "off": "minimal", "xhigh": "xhigh", "max": "max" },
          "input": ["text", "image"],
          "contextWindow": 132096,
          "maxTokens": 26419,
          "compat": {
            "supportsDeveloperRole": false,
            "sendSessionAffinityHeaders": true
          }
        }
      ]
    }
  }
}
```

Also set this in Pi's environment:

```
PI_CACHE_RETENTION=long
```

It is what makes Pi send `prompt_cache_key` (section 2). It is a process-wide
setting: it also asks every other provider that Pi talks to for long
retention. For Anthropic that means 1 h cache writes; for OpenAI, 24 h.

In the tests, `baseUrl` was the relay (`http://127.0.0.1:18236/v1`).

Headless runs:

- `pi --mode json [--thinking LEVEL] [--session-id ID | --fork ID --session-id NEW] [@file] "prompt" < /dev/null`
- **Close stdin.** With a non-TTY stdin that never reaches EOF, Pi waits to
  read it as part of the prompt. The first run of experiment `e2` hung for 10
  minutes with no request sent, until it was killed.
- **Under Git Bash, set `MSYS_NO_PATHCONV=1`.** Without it, MSYS rewrote the
  argument `/compact` to `C:/Program Files/Git/compact` before Pi saw it
  (`#27`). That was our launcher's fault. The model then spent 2 turns
  running read-only `ls`/`find` on `C:/Program Files/Git` looking for the
  "file".
- **Slash commands are not commands in print or JSON mode.** `/compact` is
  sent to the model as a prompt (`c3`). The model then claimed that "Pi's
  built-in context compaction command ... ran on your end", which is false.
  Use RPC mode (`{"type": "compact"}`) to compact headlessly.
- `-p` exits 1 on a 401. `--mode json` exited 0 on every failure seen: the
  401 in `k1`, and four 502s in `e3`.

## 1. What Pi sends

### Every request

| field | value |
|---|---|
| `model` | `yamadori` |
| `stream` | `true`, with `stream_options: {"include_usage": true}` |
| `max_completion_tokens` | `maxTokens`: 16384 when unset (Pi's default), 26419 with it. Compaction calls send 13107 and 8192. Never `max_tokens` |
| `store` | `false` (accepted and ignored, `api_errors`) |
| `tools` | 4 by default: `read(path, offset, limit)`, `bash(command, timeout)`, `edit(path, edits[{oldText, newText}])`, `write(path, content)`. 2,956 bytes, no `strict`, no `tool_choice` |
| system prompt | one message, 2,785 chars. With the tools it is about 2,345 prompt tokens, a quarter of OpenCode's 7,940 |
| instruction role | **`developer`** when the model has `reasoning: true`; `system` otherwise, or with `compat.supportsDeveloperRole: false` (section 5, gap 1) |
| `reasoning_effort` | only with `reasoning: true`. It is the Pi thinking level, mapped through `thinkingLevelMap`. The default level is `medium` |
| `prompt_cache_key`, `prompt_cache_retention` | only with `PI_CACHE_RETENTION=long`: the Pi session id, and `"24h"` |
| headers | `User-Agent: pi (win32 10.0.26200; x64)` plus the OpenAI SDK's `x-stainless-*`. With `compat.sendSessionAffinityHeaders: true`, also `session_id`, `x-client-request-id` and `x-session-affinity`, all three set to the Pi session id |
| past turns | echoes `reasoning_content` on every assistant turn it replays (72, 241 and 1,433 chars, for example). `x_yamadori.ledger.restored.echoed_reasoning` counts them |
| side calls | **none per turn**. Unlike OpenCode, Pi sends no title call. The only side calls seen were compaction's (section 6) |
| `/v1/models` | **never called**: all 44 rows are `POST /v1/chat/completions`. `--list-models` and model selection use Pi's bundled catalog and `models.json` |

The source of the `openai-completions` compat defaults is `detectCompat` in
`dist/bundle/chunks/openai-completions-*.js`. For a base URL Pi does not
recognise, the defaults are:

- `supportsDeveloperRole: true`
- `supportsStore: true`
- `maxTokensField: "max_completion_tokens"`
- `sendSessionAffinityHeaders: false`
- `supportsLongCacheRetention: true`

### Capability detection

Pi never reads our card. With the minimal config (`{"id": "yamadori"}`
only), `--list-models` shows these defaults:

- context 128K
- max-out 16.4K
- thinking no
- images no

Consequences:

- **No `reasoning_effort` is ever sent.** Every request runs at tier
  `medium`, the proxy's default (`#0`).
- **Images are dropped by Pi itself.** Its `read` tool returns
  "[Current model does not support images...]", from `dist/core/tools/read.js`.

The recommended config fixes both: `--list-models` then shows 132.1K /
26.4K / thinking yes / images yes.

**Thinking levels.** Pi has `off minimal low medium high xhigh max`. `xhigh`
and `max` are offered only when `thinkingLevelMap` names them
(`getSupportedThinkingLevels`). `off` sends nothing unless the map gives it
a value. With the map above:

| `--thinking` | sent | tier | observed |
|---|---|---|---|
| (default) | `medium` | medium | `#24` |
| `off` | `minimal` | minimal, `effort_sent: null`, no reasoning | `#25` |
| `low` | `low` | low | `#1`, `#6` |
| `high` | `high` | high | 502 with `developer`, see gap 1 |
| `xhigh` | `xhigh` | xhigh, `effort_sent: medium`, `deep.think_tool.offered: true` | `#26` |

## 2. Sessions

| case | what reached the proxy | `session.source` | id / slot |
|---|---|---|---|
| default config, text turn 1 (`--session-id pia1`) | nothing explicit | `minted` | 7c9beb310d4b, slot 0 |
| same session, turn 2 | history with our text answer | `answer_record` | 7c9beb310d4b (kept), slot 0, reused 2368/2391 |
| affinity headers on (`pib1`, 2 turns) | `session_id` / `x-client-request-id` / `x-session-affinity: pib1` | `minted`, then `answer_record` | 9a9803c18878 (kept). **Headers ignored** by the running proxy |
| a second session with the same opening (`pib2`) | headers `pib2` | `minted` | 047864fe5f83: separate |
| `PI_CACHE_RETENTION=long`, `pic1`, 2 turns | `prompt_cache_key: "pic1"` | `prompt_cache_key` both turns | 46db78df, slot 1, reused 2372/2397 |
| `--fork pic1 --session-id picf` | `prompt_cache_key: "picf"`, forked history | `prompt_cache_key` | **8e4567d6: its own session.** It adopted `pic1`'s slot through the shared prefix (`cache.adopted`, reused 2416/2441) |
| another session, `pic2` | `prompt_cache_key: "pic2"` | `prompt_cache_key` | 6096ab47: separate |
| default config, tool task (`pie1`) | our ids `call_722041522cd7_fc1176ab` echoed verbatim on both sides | `minted`, then `tool_call_id` | 722041522cd7 |
| default config, `--fork pie1 --session-id pief` | forked history carrying `pie1`'s tool-call ids | `tool_call_id` | **722041522cd7: merged into the original**, as with OpenCode |

- **Pi sends nothing stable by default.** Two settings change that:
  - `PI_CACHE_RETENTION=long` sends `prompt_cache_key` = the Pi session id.
    The running proxy honours it. It keeps sessions apart, forks included,
    and is kept across a Pi compaction (section 6).
  - `sendSessionAffinityHeaders: true` sends the session id in three
    headers. Once `server.py` is deployed, its `SESSION_HEADERS` reads
    `x-session-affinity`. It does not read `session_id` or
    `x-client-request-id`.

  Pi's session ids are UUIDv7 (36 characters), or whatever
  `--session-id` sets. Both pass `nebari.session_token` (`[A-Za-z0-9_-]{1,64}`).
- **Our carrier works with Pi** when nothing explicit is sent. Pi echoes
  `tool_calls[].id` and `tool_call_id` verbatim, across separate
  processes. So does the `answer_record` fallback for text-only turns.
- **When the client sends its own id, the ids are llama-server's**, for
  example `7JQBYc5GsCgTOdHrWXadCy85OBfPNUuq`. That is by design: a client
  with its own id gets its ids untouched.
- **The cache follows the session.** Agent steps reused 2510/2529,
  2632/2657, 2752/2770 and 2827/2873 (`#16`-`#19`).

## 3. Vision, user-attached (`@pitest.png`)

The test image is synthetic: `pitest.png`, 256x256, a red square on white
with the white text `PI-TEST`, 2,162 bytes, drawn with PIL for this test.

- **Pi's message.** A text part `<file name="C:\...\project\pitest.png"></file>`
  followed by the question, 228 chars, and an **`image_url` part** holding
  `data:image/png;base64,...` (2,906 chars). `detail` is absent. The file
  tag carries the absolute path.
- **The proxy.** It registered
  `attachments: [{"id": "image-eda8a35a00", "source": "attached", "format": "png", "bytes": 2162}]`.
  Main called `yama_describe_image` in a hidden hop:
  `vision: ok`, 8.3 s, 176 prompt / 242 completion tokens.
- **The answer.** It was correct: "a solid **red** ... square centred on a
  **white** background ... white, uppercase ... text ... reading exactly:
  **`PI-TEST`**". That took 14.6 s end to end (`#20`).

## 4. Vision, agent self-view (`read pitest.png`), then an HTML canvas

The prompt was: read `pitest.png`, say what it shows, then write
`canvas.html` that draws it (`#21`-`#23`).

1. `read {"path": "pitest.png"}`. The tool message is the text
   **`Read image file [image/png]`**. The `(see attached image)` fallback is
   used only when a tool returns no text.
2. Pi then appends a **synthetic user message**: a text part
   **`Attached image(s) from tool result:`** and an `image_url` part (the
   same 2,906-char data URL). The bytes are unchanged, not re-encoded.
   `inputLimits.images.resize` defaults to 2000x2000 / 4.5 MiB, so a larger
   image would be resized.
3. The proxy treated it as an attachment. It was the same id,
   `image-eda8a35a00`, because the id is derived from the bytes. Main called
   `yama_describe_image` (`ok`, 9.6 s), then wrote a correct description and a
   `write` of `canvas.html`, all in one request (`#22`, 26.6 s).
4. The route of `#22` was **`prose`**, because the request ends on the
   synthetic user message. (`selection.question_of` now looks past it too,
   2026-09-26, like `route` and `deep`.) The deployed proxy reads it as a new user turn.
   `image_input.TOOL_MEDIA_TURNS` (not deployed) has Pi's row, with exactly
   the string seen here. This run confirms that string live.
5. **Cache after the image turn.** The next request (`#23`) restored the
   hidden hop (`ledger.restored.hops: 1`), yet reused only **1,829 of 3,663**
   tokens: 1,834 were processed. OpenCode showed the same shape (7,438 of
   9,674). The cause is not established. 1,829 is the reuse figure that
   recurs across new sessions, probably a context checkpoint's position.
6. `canvas.html` is a valid page. The model wrote `ctx.font = 'light 24px
   sans-serif'`, which is not a valid CSS font value, so the browser keeps
   its 10 px default. That is model quality (n=1), not the proxy.
   **tool_code did not check it** (section 5, gap 4).

The image guard (#46) is not deployed and had no reason to trip: no image
data went into a tool argument.

## 5. Agentic task, tool calls and `tool_code` (`a1`, `#15`-`#19`)

The prompt was: write `hello.py` with the `write` tool, change "hello" to
"howdy" with the `edit` tool, then run `py hello.py`. Tier `medium`,
`PI_CACHE_RETENTION=long`. It took 5 requests and 12 s, and the answer
was correct (`howdy pi`).

| step | call | `tool_code` | note delivered as content |
|---|---|---|---|
| 1 | `write {"path": "hello.py", "content": ...}` | `detected: "table"` (`KNOWN["write"]` has `_w("path", "content")`), `errors 0`, `format_would_change: 1` | `Verified hello.py (python): parses, lint clean; sent as written.` |
| 2 | `edit {"path": "hello.py", "edits": [{"oldText": "return 'hello ' + name", "newText": "return 'howdy ' + name"}]}` | **`detected: "shape"`**, kind `edits`, `flagged: "fragment"` | `Checked hello.py (python, edit): an edit fragment, not checkable on its own.` |
| 3 | `bash {"command": "py hello.py"}` | `no_code` | (the model's own) `Ran hello.py (python, edit): the edit succeeded, but ...` |
| 4 | `read {"path": "hello.py"}` | `no_code` | |
| 5 | answer | | |

Pi's `bash` tool ran under Git Bash on Windows. Its default tool set also
has `powershell`, `grep`, `find` and `ls`, which were not enabled.

## Gaps for the proxy, with evidence

Status 2026-09-26: gaps 1, 3, 4 and 5 are FIXED in the working tree, with
offline tests that failed before the fix (`mcp/test_harness_forms.py`,
`mcp/test_tool_code.py`, `mcp/test_utility.py`). **Not deployed and not yet
run live**: the proxy that served this test is still yesterday's.

1. **`developer` role plus the addendum is a hard 502 at `high`, `xhigh`
   and `max`.** FIXED (not yet live): `system_roles.one_system` (shared with the Responses adapter) maps
   `developer` to `system` on the way in (leading instruction messages join,
   a later one becomes a user turn), so every reader and the addendum see
   one system message; a template refusal is now a 400 `invalid_prompt`,
   not a retried 502 (`api_errors.template_refusal`). The
   `supportsDeveloperRole: false` workaround is no longer needed once
   deployed.
   - What Pi sends: with `reasoning: true`, and without
     `compat.supportsDeveloperRole: false`, the instruction message is
     `{"role": "developer"}`.
   - What the proxy does: `proxy.add_addendum` looks only for
     `role == "system"`. It finds none, so it prepends a new system message.
     llama-server renders `developer` as system (the same 2,345 prompt
     tokens as `system` at `low`, `#0` vs `#1`). The template then sees two
     system messages and raises "System message must be at the beginning".
   - Observed (`e3`, `#2`-`#5`): 502 `upstream_error`, deterministic. Pi
     retried 3 times (2/4/8 s backoff), gave up, and exited 0 in JSON mode.
     At `low` (no addendum) the `developer` role worked (`#1`).
   - Workaround: `supportsDeveloperRole: false`, in the config above.
     `high` was not re-run with it. `xhigh` was, and got 200 (`#26`); it
     carries the same addendum.
   - Fix (made 2026-09-26): `developer` is normalised to `system` at the
     top of `_run_turn` and `prepare`; `add_addendum` also accepts it (and a
     list-content system message). `route.py`, `selection.py`,
     `deep.py`, `slots.py` and `responses_api.py` already accept both
     roles. `proxy.conversation_versions` (the `("user", "tool", "system")`
     scan) does not.
   - Any OpenAI-SDK client that marks a model as a reasoning model can hit
     this, not only Pi.
2. **By default, a fork merges into the original conversation**, as with
   OpenCode (`#40`-`#43`). With `PI_CACHE_RETENTION=long` it does not
   (`#13`). Once `server.py` is deployed, `sendSessionAffinityHeaders`
   also fixes it through `x-session-affinity`. Nothing else to build for
   Pi.
3. **Pi's compaction is not mapped, and it runs without thinking.** See
   section 6. FIXED (not yet live): `compaction.parse_transcript` reads the
   `<conversation>` and `# Conversation` forms (records from
   `utils.js:94`), maps them onto the stored conversation like Hermes' form
   (Pi's summariser system prompt opens the instruction), and a summary that
   maps onto nothing thinks at the effort Pi sent (`compaction.client_fields`;
   the recorded reason names the client's tier). `x_yamadori.compaction
   .harness` is `pi`.
4. **`tool_code` and Pi's tools.**
   - `write(path, content)` matches the table.
   - `edit(path, edits[{oldText, newText}])` is caught only by the shape
     fallback. `KNOWN["edit"]` lists Roo's and OpenCode's replace shapes.
     Pi's shape is VERIFIED from `dist/core/tools/edit.js` and could be a
     table row. Pi also accepts `edits` as a JSON string, and a legacy
     top-level `oldText`/`newText` (`prepareEditArguments`). The shape
     fallback would catch the legacy form as `replace`, but not the string
     form. Report only; `tool_code.py` was not edited.
   - **An `.html` write is not checked.** `tool_code.language_of("canvas.html")`
     is `None`, so the unit has no language and the record says
     `stopped: "no_code"` (`#22`). Inline `<script>` in HTML is where
     three.js / TSL pages live.
   - FIXED (not yet live): Pi's `edit` and `write` are VERIFIED rows of
     `tool_code.KNOWN` (edit.js:10-21, write.js:8-11), the string and
     legacy forms included; an `.html` write's inline scripts are checked as
     JavaScript (`tool_code.script_units`); a clean fragment gets no note.
   - **The model imitated our note** (the #36 pattern, n=1). After
     `Checked hello.py (python, edit): an edit fragment, not checkable on
     its own.`, the model's next content began
     `Ran hello.py (python, edit): the edit succeeded, but I'm not showing
     the full contents of the file`. That sentence is also false: it had
     shown nothing. The "Checked" phrase marks problems remaining
     (AGENTS.md), yet here it was used for a clean fragment.
5. **The cache after a hidden `yama_describe_image` hop** fell back to 1,829
   tokens (section 4, item 5). This is the second harness to show it.
   CAUSE, found offline (`mcp/test_harness_forms.py`, through the served
   template): ours. The ledger replayed the hidden hop with its reasoning
   EMPTIED, while the slot had generated it with its reasoning, and Pi
   echoes everything else exactly -- so #23 diverged at the hop's think
   block, and the warm had said "the slot already holds the turn". FIXED
   (not yet live): when the turn's reasoning is exactly the echo of what
   the client was shown, the hops keep their reasoning and the turn gets
   the slot's (kind `hop_echo`); a stripping client is unchanged. The
   placeholder and the synthetic tool-media turn render identically on
   every request (checked in the same test).
6. **A title call with an image does not arise for Pi**, because Pi makes
   no title call. OpenCode's pin-eviction side effect does not apply.

## 6. Compaction

Compaction was triggered through RPC (`rpc_compact.py`), with
`compaction.keepRecentTokens: 500` set temporarily in the isolated
`settings.json`. With the default 20,000, Pi answered "Nothing to compact
(session too small)". The session was `pid1`; `tokensBefore` was 8,091.

| row | what | proxy |
|---|---|---|
| `#34` | history summary. System `You are a context summarization assistant...` (310 chars); user `<conversation>\n[User]: ...` (7,840 chars); no tools; no `prompt_cache_key`, no affinity headers; `reasoning_effort: "low"`; `max_completion_tokens: 13107` | `utility: true`, `utility_kind: "compaction"`, form `summarise_conversation`. Tier overridden to `minimal`, `session: null`, transient slot 3, reused 0/2579, 13.1 s. `compaction.shape: "flattened"`, `mode: "as_sent"`, why "no TURNS TO SUMMARIZE block of [ROLE]: records"; `thinking: "off: the conversation itself runs without it"` |
| `#35` | split-turn prefix summary. User `# Conversation\n[User]: /compact ...` (4,280 chars), `max_completion_tokens: 8192` | the same: utility, minimal, 7.5 s |
| `#36` | next turn. The summary is a user message, `The conversation history before this point was compacted into the following summary:\n\n<summary>...` | `prompt_cache_key: "pid1"`, the same session 9ec77d21, slot 1; reused **1,829/3,856** (the history was replaced) |

The details:

- **Pi strips its session from compaction calls.** They use
  `cacheRetention: "none"`, so no key and no headers are sent, even with
  both settings on. The proxy cannot tie them to the conversation, and they
  run sessionless on the transient slot, as a utility call should.
- **Thinking was off, contrary to AGENTS.md.** AGENTS.md says a compaction
  thinks at the conversation's own effort. Here the utility path overrode
  to `minimal`, although Pi sent `reasoning_effort: "low"` and the
  conversation ran with thinking. The recorded reason ("the conversation
  itself runs without it") is wrong for this conversation. The summaries
  were usable: a structured Goal / Constraints / Progress summary of 3,070
  chars.
- **The mapping handles only Hermes' form.** Pi's form is `<conversation>`
  or `# Conversation` with `[User]:` / `[Assistant]:` /
  `[Assistant thinking]:` records, and `_serve_compaction` maps only
  Hermes' `TURNS TO SUMMARIZE`. Mapping it onto the stored prompt would let
  the summary reuse the slot. As sent, it processed all 2,579 tokens.
- **The key carries the conversation across a compaction.** Without
  `PI_CACHE_RETENTION`, the post-compaction history keeps none of our
  tool-call ids, so it would most likely mint a new session. **Not
  tested.**

## 7. `/v1/responses`

`:1234` answers `POST /v1/responses` with 404 `unknown_url` (the running
proxy, probed directly). No Pi run used it. The config that would switch
Pi is the same file with `"api": "openai-responses"`. Its compat keys
differ: `supportsDeveloperRole`, `sessionAffinityFormat`,
`supportsLongCacheRetention`, `supportsStrictMode`,
`supportsOpenAIGrammarTools` and `supportsMaxOutputTokens`.

What Pi would send, read from `dist/bundle/chunks/openai-responses-*.js`
and **not observed**:

- `prompt_cache_key` = the session id **by default**, with no env var.
  Only retention `none` (compaction) omits it.
- `store: false`.
- `reasoning: {effort, summary: "auto"}` and
  `include: ["reasoning.encrypted_content"]`.
- `max_output_tokens`, only with `compat.supportsMaxOutputTokens: true`.
- Headers `session_id` and `x-client-request-id`, but no
  `x-session-affinity`.
- No `previous_response_id`: the full input is sent every time.

`mcp/responses_api.py` (not deployed) already names `prompt_cache_key`,
`max_output_tokens`, `developer` and `encrypted_content`. On the Responses
path, Pi's `developer` role would be merged by the adapter.

## 8. Conformance as Pi sees it

| check | result |
|---|---|
| invalid key | 401 `invalid_api_key`, one request each, **not retried**. `-p` prints `401: {"message": "unrecognised API key...", ...}` and exits 1; `--mode json` exits 0 (`#38`, `#39`) |
| 5xx | retried 3 times with backoff (gap 1) |
| usage | `cached_tokens` and `reasoning_tokens` are read into Pi's `cacheRead` / `reasoning` |
| finish reasons | `stop` / `tool_calls` map to Pi's `stop` / `toolUse` |
| `x_yamadori` on the final chunk | tolerated |
| unknown body keys | `store`, `prompt_cache_retention: "24h"`: accepted; the latter is forwarded upstream, where it was accepted |
| `max_completion_tokens` | honoured as the answer allowance (`tiers.budget`) |
| our notes in the transcript | stored and resent as assistant content, as designed. They are imitated (gap 4) |

**Not tested:** automatic compaction (it needs a long session), branch
summarization (`/tree`), Pi extensions and skills, the `powershell` tool,
`high` with `supportsDeveloperRole: false`, deep thinking actually firing,
images above Pi's resize limits, and a long silent stream.

**Silence tolerance, from Pi's docs (`docs/settings.md`), not observed.**
The settings involved:

- `httpIdleTimeoutMs`, default 300,000: the "HTTP header and body idle
  timeout". It is OpenCode's `chunkTimeout` under another name.
- `retry.provider.timeoutMs`, which defaults to that value.
- Agent-level retry: `retry.maxRetries` 3, `retry.baseDelayMs` 2,000.

So a stream that is silent for 300 s would be aborted and retried, as
OpenCode's was (docs/HARNESS-OPENCODE.md). The #44 tool-call heartbeat is
the proxy-side fix. On the client side, raise `httpIdleTimeoutMs` in the
agent directory's `settings.json` for `xhigh` / `max` work.

## Pi vs OpenCode vs Hermes

| | Hermes | OpenCode 1.18.32 | Pi 0.87.1 |
|---|---|---|---|
| session id sent | none to a custom endpoint | `X-Session-Id` / `x-session-affinity` always; camelCase `promptCacheKey` with `setCacheKey` | **nothing by default**. `prompt_cache_key` with `PI_CACHE_RETENTION=long` (honoured today); `session_id` / `x-client-request-id` / `x-session-affinity` with `sendSessionAffinityHeaders` |
| our tool-call-id carrier | echoed (from source) | echoed, observed | echoed, observed |
| fork | n/a | merges into the original | merges by default; **separate with `prompt_cache_key`** |
| instruction role | system | system | **`developer`** with `reasoning: true`: 502 at high+ on the tested proxy; mapped to system since 2026-09-26 (not yet live) |
| past reasoning | stripped | echoed | echoed |
| `/v1/models` | read | never called | never called; bundled catalog plus `models.json` |
| `reasoning_effort` | sent | only with `--variant` | only with `reasoning: true`; default `medium`; `xhigh`/`max` need `thinkingLevelMap` |
| title / side calls | utility | a title call per session | **none**; compaction's summaries only |
| user-attached image | n/a | `image_url` part (needs `modalities`) | `image_url` part plus a `<file name=...>` text part (needs `input`) |
| agent looks at its own image | `vision_analyze(path \| data URL)` | `read`, then a synthetic user turn "Attached media from tool result:" | `read`, then a synthetic user turn "Attached image(s) from tool result:"; the tool message reads "Read image file [image/png]" |
| compaction | flattened `TURNS TO SUMMARIZE`, mapped | not observed | `<conversation>` / `# Conversation` records, sessionless; on the tested proxy not mapped and served without thinking; mapped and thinking since 2026-09-26 (not yet live) |
| retries | stale-stream detector | `chunkTimeout` 300 s, then a silent retry | 3 retries on 5xx (2/4/8 s); none on 401; `httpIdleTimeoutMs` 300 s (from docs, not observed) |
