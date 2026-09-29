# Codex CLI against Yamadori

Prepared offline on 2026-09-26. Codex CLI **0.157.1** is installed and
verified, and an isolated `CODEX_HOME` is written. **Nothing has been sent
to `:1234`.** The `/v1/responses` adapter (`mcp/responses_api.py`) is not
deployed yet, and another harness test held the GPU. Every "captured" fact
below comes from Codex talking to a local **capture stub**:
`capture_stub.py` on `127.0.0.1:18299`, which is not a model and does no
generation. It records each request and returns a one-line Responses
stream, or a scripted tool call so Codex runs the tool and sends its
output back. For the same runs `HTTP(S)_PROXY` pointed at the stub, so any
outbound connection Codex tried was logged and refused.

`docs/HARNESS-RESPONSES.md` §5 covers Codex **0.133.0** from source
(`codex-rs/...` paths). This document covers what **0.157.1**, the current
npm `latest`, actually sends. Where the two disagree, the capture wins for
0.157.1.

Evidence lives in the session scratchpad
(`C:\Users\jwals\AppData\Local\Temp\claude\C--Users-jwals-llama-stack\d16e1f09-4699-483b-a257-1ba77b329ce7\scratchpad\`,
called SP below):

- `capture_stub.py`, `cap.ps1`, `summ.py`
- one request body per variant: `exec1`, `ww`, `cat_ff`, `lean`, `rui`, `vis`, `patch_unel`, `execc`, `final` `.body.json`
- the extracted strings of `codex.exe`: `codex-strings.txt`, offsets in hex
- `dbg-models.json`: the bundled model catalog
- `codex-verify\att.json`: the provenance attestations

All counts are n=1 captures of a deterministic request builder, not model
behaviour.

## 1. Install and its verification

| check | result |
|---|---|
| package | `@openai/codex`, `latest` = **0.157.1** (registry `time.modified` 2026-09-26T07:25Z); license Apache-2.0; `repository: git+https://github.com/openai/codex.git`, directory `codex-cli`; maintainers include `openai-publisher <oai-package-publish-npm@openai.com>` and 17 `*@openai.com` accounts |
| publisher | `_npmUser: GitHub Actions <npm-oidc-no-reply@github.com>` (trusted publishing), for both `@openai/codex@0.157.1` and the platform package `@openai/codex@0.157.1-win32-x64` (aliased as the optional dependency `@openai/codex-win32-x64`) |
| signatures + provenance | `npm audit signatures`, run in a scratch project (it refuses `-g`): "2 packages have verified registry signatures", "2 packages have verified attestations" |
| provenance content | SLSA v1 for both packages: repository `https://github.com/openai/codex`, ref `refs/tags/rust-v0.157.1`, commit `36650394c5b38c2990ccf2a3457165ca3e9d9726`, workflow `.github/workflows/rust-release.yml` (`SP\codex-verify\att.json`) |
| integrity | `@openai/codex@0.157.1`: `sha512-qJ/UZ0bmYP+/Umav1L9WpmtMYeA6q1+4r4qILSYOKQZhP7WRdjyTQWz3O0dTImZ9RT7AazZa85D87xRDHogcHw==`. `-win32-x64`: `sha512-vgqs/VRXNwhLYMsZDgYfnRSXpRh5Nm782L8lacGskw86kOxbMaquvQKxkuZHUBrJA2XGcksB7rMUHy1XaCJgrA==` (446 MB unpacked) |
| install scripts | none in either package (`npm view ... scripts` is empty). Installed with `--ignore-scripts` anyway |
| binary | `codex.exe`: Authenticode **Valid**, signer `CN="OpenAI OpCo, LLC"`. SHA-256 `8CB0E69E99FF2A158C54815DB82D0F2E524D8F301BC30184722CFD1AE5973574`, identical to the copy in the attested scratch install |
| runs | `codex --version` prints `codex-cli 0.157.1` |

Install command. It uses Node 24's own npm, so the global Node is not
switched: `node` on PATH stays `C:\Program Files\nodejs` (v20.15.0). Pi was
installed the same way.

```powershell
$env:PATH = "C:\Users\jwals\AppData\Roaming\nvm\v24.21.0;$env:PATH"   # this shell only
C:\Users\jwals\AppData\Roaming\nvm\v24.21.0\npm.cmd install -g --ignore-scripts @openai/codex@0.157.1
```

Where it landed:

- **Shim:** `C:\Users\jwals\AppData\Roaming\nvm\v24.21.0\codex.cmd`. It is
  not on PATH.
- **Native binary:** `...\v24.21.0\node_modules\@openai\codex\node_modules\@openai\codex-win32-x64\vendor\x86_64-pc-windows-msvc\bin\codex.exe`
  (`codex.js` only locates and spawns it).
- **Also shipped beside it:**
  - `codex-path\rg.exe`
  - `codex-resources\codex-command-runner.exe`
  - `codex-windows-sandbox-setup.exe`
  - `bin\codex-code-mode-host.exe`
  - a voice host with GStreamer DLLs

To pin it: record the `sha512` values and the `codex.exe` SHA-256 above in
`engines/manifest.yaml` or `models/manifest.yaml` if Codex becomes a
benchmark harness. Not done here.

**Known warning, harmless for our use.** Every run with `CODEX_HOME` under
`%TEMP%` prints `WARNING: proceeding, even though we could not create PATH
aliases: Refusing to create helper binaries under temporary dir`.
`apply_patch` still worked (captured, §3). Move `CODEX_HOME` out of `%TEMP%`
to silence it.

## 2. The config (`SP\codex-home\`)

Launching:

```powershell
$env:CODEX_HOME = "C:\Users\jwals\AppData\Local\Temp\claude\C--Users-jwals-llama-stack\d16e1f09-4699-483b-a257-1ba77b329ce7\scratchpad\codex-home"
$env:YAMADORI_CODEX_KEY = (Get-Content "$env:CODEX_HOME\..\codex-dogfood.key" -Raw).Trim()   # never echoed
C:\Users\jwals\AppData\Roaming\nvm\v24.21.0\codex.cmd exec --json -C <workspace> "<prompt>"
```

The key is read from the environment variable (`env_key`) and never stored
in the config. The key file `SP\codex-dogfood.key` will be minted by the
operator. It does not exist yet.

`config.toml`:

```toml
model = "yamadori"
model_provider = "yamadori"
model_catalog_json = '<SP>\codex-home\yamadori-catalog.json'
model_reasoning_effort = "medium"
sandbox_mode = "workspace-write"
approval_policy = "never"
web_search = "disabled"
check_for_update_on_startup = false

[windows]
sandbox = "unelevated"

[model_providers.yamadori]
name = "Yamadori"
base_url = "http://127.0.0.1:1234/v1"
env_key = "YAMADORI_CODEX_KEY"
wire_api = "responses"
stream_idle_timeout_ms = 900000
request_max_retries = 1
stream_max_retries = 0

[tools.update_plan]
enabled = true

[tools.experimental_request_user_input]
enabled = false

[features]
multi_agent = false
goals = false
plugins = false
remote_plugin = false
apps = false
memories = false

[analytics]
enabled = false

[feedback]
enabled = false
```

**Validated offline.** A byte copy with only `base_url` swapped to the stub
(`SP\codex-home-validate`) ran `codex exec --strict-config` to completion.
It sent 5 tools and made no outbound CONNECT (`final.body.json`).

Why each key is there:

- **`model_catalog_json`** points at `yamadori-catalog.json`.
  - The problem: `yamadori` is not in the bundled catalog (`codex debug models`: 11 OpenAI slugs). With no entry, `codex exec` emits `"Model metadata for yamadori not found. Defaulting to fallback metadata"`. The fallback offers **no `apply_patch` tool** (captured, `exec1`/`ww`) and sends `reasoning: {"summary": "auto"}` with **no effort**.
  - The entry: `gpt-5.5`'s entry with our values: slug `yamadori`, `context_window` / `max_context_window` 132,096 (the main share), `input_modalities: ["text","image"]`, `apply_patch_tool_type: "freeform"`, efforts low/medium/high/xhigh, `supports_search_tool: false`, plugins and apps instructions off, `base_instructions` = the 17,174-character fallback instructions.
  - `apply_patch_tool_type` accepts only `freeform`. `"function"` fails with `unknown variant function, expected freeform`.
  - With the entry, the request carries `reasoning: {"effort": "medium"}`.
  - This is the static equivalent of the remote catalog in HARNESS-RESPONSES §6. Every catalog value is a CHOICE, unmeasured.
- **`sandbox_mode` + `[windows] sandbox = "unelevated"`.** On Windows,
  `-s workspace-write` **alone ran read-only.**
  - Evidence: the turn metadata said `sandbox: "none"`, `sandbox_mode: "read-only"`, and `apply_patch` returned `patch rejected: writing is blocked by read-only sandbox; rejected by user approval settings`.
  - With `unelevated` the metadata says `sandbox: "windows_sandbox"`, `sandbox_mode: "workspace-write"`, and the patch applied.
  - That first run took **51.8 s** wall time for one Add File, which includes the sandbox setup.
  - What the setup changed:
    - it wrote `.sandbox`, `.sandbox-bin` and `cap_sid` into `CODEX_HOME`;
    - it added a capability-SID ACE (`S-1-15-3-...`, FullControl) to the workspace directory;
    - it wrote `[projects.'<workspace>'] trust_level = "trusted"` into `config.toml` itself.
  - It needs no admin rights. The elevated backend (`codex sandbox setup --elevated`) was not used.
  - The prompt says "Network access is restricted". Whether a sandboxed `curl http://localhost:<port>` reaches a dev server is **not verified** (live plan, item 6).
- **`approval_policy = "never"`** is `exec`'s behaviour anyway. A command
  asking for `sandbox_permissions: "require_escalated"` is rejected, never
  run unsandboxed (the permissions developer message says so).
- **`web_search = "disabled"`** removes the hosted `web_search` tool.
  `llama-server` cannot run it. The adapter accepts and ignores it
  (`mcp/responses_api.py:153`, `:407`), so leaving it on is harmless but
  sends a tool the model can call to no effect.
- **`stream_idle_timeout_ms = 900000`, `request_max_retries = 1`,
  `stream_max_retries = 0`.** These are CHOICES. The model can go minutes
  between parsed events (#44), and a retry re-runs a whole generation on
  the one GPU. The 0.133.0 defaults are 300 s, 4 and 5 (HARNESS-RESPONSES
  §5). The adapter's 1 s `response.in_progress` keepalive should make the
  idle timer moot. Not verified live.
- **`[features] plugins/remote_plugin/apps = false`.** With the defaults
  every `codex exec` tried `CONNECT chatgpt.com:443` (Codex) and
  `CONNECT github.com:443` (User-Agent `git/2.45.2.windows.1`). With these
  three off, none was observed (`noplug`). Only connections that honour the
  proxy variables can be seen this way.
- **`multi_agent`, `goals`, `request_user_input` off.** See §3.
- **`analytics`, `feedback`, `check_for_update_on_startup`** are off so
  nothing phones home. `x-codex-turn-metadata` then says
  `"analytics_enabled": false`.

## 3. The tools Codex offers the model (0.157.1, captured)

Names are exactly as sent. Chars are `len(json.dumps(tool))`.

| tool | type | what it does | default (no catalog) | with our catalog | our profile | chars |
|---|---|---|---|---|---|---|
| `exec_command` | function | run a command in a PTY; returns output or a session id. Params `cmd, justification, login, max_output_tokens, prefix_rule, sandbox_permissions, shell, tty, workdir, yield_time_ms`. The description carries Windows safety rules | on | on | **on** | 2,706 |
| `write_stdin` | function | write to a running `exec_command` session and read recent output (`chars, max_output_tokens, session_id, yield_time_ms`) | on | on | **on** | 819 |
| `apply_patch` | **custom** (freeform, Lark grammar `*** Begin Patch ... *** End Patch`) | add, delete, update or move files | **absent** | on | **on** | 896 |
| `view_image` | function | "View a local image file" (`path`). The image comes back IN the tool output (§4) | on | on | **on** | 391 |
| `update_plan` | function | task plan (`explanation`, `plan[{step, status}]`) | absent | absent | **on**, via `[tools.update_plan] enabled = true` | 781 |
| `request_user_input` | function | ask 1-3 questions. Its description says "only available in Plan mode", yet it is sent in exec | on | on | **off**: `[tools.experimental_request_user_input] enabled = false` | 1,425 |
| `multi_agent_v1` = `close_agent, resume_agent, send_input, spawn_agent, wait_agent` | **namespace** | subagents. `spawn_agent` lists OpenAI model overrides (`gpt-6-astra`, ...) | on | on | **off**: `features.multi_agent = false` | 10,157 (9,491 with catalog) |
| `get_goal, create_goal, update_goal` | function | thread goals with token budgets ("only when explicitly requested") | on | on | **off**: `features.goals = false` | 3,278 |
| `web_search` | hosted (`external_web_access: false`) | OpenAI-side search | on | on | **off**: `web_search = "disabled"` | 52-95 |
| `image_generation` / `image_gen` | hosted | image generation | **not sent** | not sent (also with `experimental_supported_tools: ["image_generation","image_gen"]`) | n/a | – |

Tools that do not exist in the CLI:

- **Code search:** there is no grep or find tool. Search goes through
  `exec_command` (`rg.exe` ships in `codex-path\`; `rg` is also on PATH
  here through scoop).
- **Browser, console or screenshot:** none. The features `browser_use`,
  `in_app_browser` and `computer_use` are listed as stable/true by
  `codex features list`, but no such tool was sent in any exec capture.
  They belong to the desktop app. A browser would have to come from an MCP
  server (`mcp_servers` in config; not tried).

Tool sets:

| set | tools | chars |
|---|---|---|
| fallback default | 9 | 18,828 |
| default + catalog | 10 | 19,101 |
| **our profile** | **5** (`exec_command write_stdin update_plan apply_patch view_image`) | **5,593** |

The system text is on top of the tools. It is the same for every set:

- `instructions`: 17,174 chars;
- a developer message: 5,152 + 596 chars (permissions and skills listing);
- the environment context: 1,446 chars.

The whole first request body was 32,832 bytes.

**Image generation.** Not offered to this provider in 0.157.1, and no
config key found here turns it on. HARNESS-RESPONSES reads the 0.133.0
source (`spec_plan.rs:298-306`): it needs ChatGPT auth, the provider
capability and `Feature::ImageGeneration` together. `codex.exe` also
bundles an `imagegen` system skill (copied into `CODEX_HOME\skills\.system`
on first run) that points at the built-in `image_gen` tool or a fallback
`scripts/image_gen.py` needing `OPENAI_API_KEY`. The skill is listed in
every request's developer message. **With this provider the skill names a
tool the model does not have.** Removing it means Codex's own skills
config, which was not explored.

**Risky:**

- `exec_command` with `sandbox_permissions: "require_escalated"` is
  unsandboxed execution. `approval_policy = "never"` rejects it.
- `--dangerously-bypass-approvals-and-sandbox` means host execution with no
  sandbox. Never use it here.
- `multi_agent` spawns more contexts on the same endpoint.
- `web_search`: hosted.
- plugins, apps and remote plugins reach chatgpt.com and github.com.
- `mcp_servers` runs host processes.
- hooks (`features.hooks` stable/true) run host commands on events. None
  are configured.

## 4. What Codex will send us

Captured from the stub. It agrees with HARNESS-RESPONSES §5 except where
marked.

**The body.** `model`, `instructions`, `input`, `tools`,
`tool_choice: "auto"`, **`parallel_tool_calls: true`** (0.133.0 source
read: false), `reasoning` (see §2), `store: false`, `stream: true`,
`include: ["reasoning.encrypted_content"]`, `prompt_cache_key` = the thread
id (a UUID) and `client_metadata`. There is no `max_output_tokens` and no
`previous_response_id`.

**Headers.**

- `session-id`, `thread-id`, `x-client-request-id` and `x-codex-window-id`
  (thread id plus `:0`);
- `x-codex-turn-metadata`: JSON with installation id, turn id, sandbox,
  sandbox mode, `turn_trigger: "exec"` and `model`;
- `x-codex-beta-features: remote_compaction_v2`;
- `originator: codex_exec`;
- `user-agent: codex_exec/0.157.1 (Windows 10.0.26200; x86_64) ...`.

The proxy takes `prompt_cache_key` as session source 1
(`mcp/session_id.py`), so a Codex thread is one Yamadori conversation.

**Input items.**

1. `developer`: skills and permissions.
2. `user`: `<environment_context>` (cwd, shell `powershell`, date, timezone, filesystem profile).
3. The user's message.

**A user-attached image** (`-i square.png`) is one user message:

- `input_text` `<image name=[Image #1] path="square.png">`;
- `input_image` with `image_url: "data:image/png;base64,..."` and `detail: "high"`;
- `input_text` `</image>`;
- the prompt text.

The adapter maps `input_image` to an `image_url` part (`responses_api.py`
docstring, lines 48 and 62). The proxy then treats it as an attachment for
`yama_describe_image`.

**Tool round trips** (the stub scripted each call, and Codex ran it):

- `view_image` produces a `function_call_output` whose `output` is an
  **array** `[{"type": "input_image", "image_url": "data:image/png;base64,...", "detail": "high"}]`.
  There is no text part. This is Codex's self-view path. It never puts
  base64 into a tool ARGUMENT, so the image guard (#46) has nothing to
  catch.
- `apply_patch` arrives as a `custom_tool_call` with `input` = the patch
  text. The result is a `custom_tool_call_output`, a string
  `"Exit code: 0\nWall time: 51.8 seconds\nOutput:\nSuccess. Updated the following files:\nA hello.txt\n"`.
  It is a reject string when sandboxed read-only.
- `exec_command` produces a `function_call_output` string
  `"Chunk ID: 87eb5c\nWall time: 1.8484 seconds\nProcess exited with code 0\nOriginal token count: 5\nOutput:\nhello from stub\r\n"`.

**What the adapter already covers** (read in `mcp/responses_api.py`, not
run):

- custom tools become a one-string function and come back as
  `custom_tool_call` (`:281`, `:391-396`, `:844-850`);
- `namespace` tools are flattened, and calls carry `namespace` back
  (`:345-418`, `:858-859`);
- `custom_tool_call_output` / `function_call_output` image arrays become
  `image_url` parts (docstring line 62);
- hosted `web_search` is ignored (`:153`, `:407`).

**What 0.157.1 adds that nobody has checked:**

- `tool_code` has to recognise the freeform `apply_patch`, whose single
  string is a V4A patch. The names line up on paper, but it has not been
  run:
  - the adapter turns the custom tool into a function whose one parameter
    is `input` (`responses_api.py:293`);
  - `tool_code`'s known-names table has `apply_patch(input)` for Codex,
    marked UNVERIFIED (`mcp/tool_code.py:184-186`);
- the `x-codex-beta-features: remote_compaction_v2` header may mean Codex
  tries a remote compaction call. What it sends when it compacts is
  unknown.

## 5. Live test plan (after the adapter deploys and the key is minted)

Preconditions:

- `python scripts/deploy_check.py --key-file PATH` exits 0 on the build
  that has `/v1/responses`;
- the card is idle: one GPU consumer, and no other harness test running;
- runs go through the recording relay (`bench/octopus/relay.py` in front
  of `:1234`, with `base_url` pointed at the relay), so every body is kept;
- the workspace is a scratch git repo outside `llama-stack`.

A 429 is **not run**, never a failure.

| # | test | command sketch | pass when |
|---|---|---|---|
| 1 | smoke / exact output | `codex exec --json "Reply with exactly: ok"` | the answer is `ok` and nothing else; `x_yamadori.session.source = prompt_cache_key`; one `response.completed` with integer `usage` |
| 2 | sessions | (a) exec, then `codex exec resume --last "..."`; (b) `codex exec fork <id>` | (a) the same `prompt_cache_key` and the same Yamadori session, with cache reuse on the resume (`x_yamadori.cache`); (b) a new thread id means a NEW session (the opposite of OpenCode's fork merge, HARNESS-OPENCODE §1) |
| 3 | tools: write, patch, run | a task: create `app.py` with `apply_patch`, run it with `exec_command`, fix it, add a plan with `update_plan` | the custom tool round-trips through the adapter, and the patch applies inside the sandbox; `tool_code` records a Verified/Checked note for the freeform patch, or `tool_code.unknown`: record which; `write_stdin` works on an interactive `python -i` |
| 4 | vision, attached | `codex exec -i square.png "what colour is this?"` | `x_yamadori.attachments` has the image, `yama_describe_image` runs, and the answer is "blue" |
| 5 | vision, self-view (`view_image`) | "draw a 16x16 red PNG with Python (zlib/struct only), then look at it with view_image and say its colour" | the `function_call_output` image array becomes an attachment, `yama_describe_image` runs, and the answer is "red"; no image data in any argument (`x_yamadori.image_guard` silent) |
| 6 | web app check inside the sandbox | "serve index.html with `python -m http.server 8765` in the background and fetch it with curl" | record whether the unelevated sandbox lets `curl localhost` through (§2 unknown). If not, that is a harness-config finding, not a model failure |
| 7 | image generation | (a) the tools array as sent (relay); (b) "draw a small pixel-art tree" | (a) no `image_generation` tool from Codex (expected, §3); (b) record whether the proxy offers `yama_generate_image` on the Responses path and whether the markdown image line reaches Codex's output. The `imagegen` system skill naming a missing tool is a known confounder |
| 8 | hosted and namespace tolerance | one run with `web_search` and `multi_agent` back on (`-c web_search=cached --enable multi_agent`), a plain question | no 400; `x_yamadori` records the flattened namespace; the model does not call `spawn_agent` for a one-line question (if it does, record it) |
| 9 | compaction | a long session with `-c model_auto_compact_token_limit=20000` | capture what Codex sends when it compacts (a summarise turn? a remote-compaction call because of `remote_compaction_v2`?), and whether the session and ledger survive it |
| 10 | long silence | an `xhigh` request that triggers deep thinking (> 300 s) | no silent retry, since `stream_idle_timeout_ms` is 900 s and the adapter keepalive runs; one generation upstream |
| 11 | errors | a bad key; an oversized prompt | 401 is shown and not retried; overflow comes back as `response.failed` `context_length_exceeded`, which Codex maps to its compaction path |

Report each result as n=1 with its relay line and `x_yamadori` evidence,
and repeat whatever a decision rests on (PROTOCOL).
