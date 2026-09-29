# Harness tools: what each harness can give the model

Operator direction, 2026-09-26: proxy-level work is near its limit. The next
gains are:

- **harness tools that help the model succeed**, turned on by configuration
  (never a harness patch);
- **server-injected skills**, verified on every harness.

The evidence is Octopus v0e (#47). Hermes had no browser tool, because its
Blank Slate setup disabled the toolset. The model made 37
Playwright-through-terminal calls, and the final page error ("PLAYER.hit is
not a function") never reached it.

This matrix is what skills key off. A skill is selected on the request's
tool NAMES, so every name below is exact, as the model sees it.

**Where they run.** The OpenCode, Pi and Codex tests below ran on the
Windows host, so the model's shell was the host (SELF-IMPROVEMENT-LOG #49).
`bench/sandbox/harness_box.py` runs the same three harnesses, at the same
versions, in a container. Its only route to the host is one forward to the
proxy or the relay. See docs/HARNESS-SANDBOX.md. The image was built on
2026-09-26 and is recorded in models/manifest.yaml `harness-box`. It was
rebuilt on 2026-09-27 (02:53Z), with the tag suffix `-tools1`, to carry the
default loadout (§0). So far it has run only against stand-ins (no model).

**The default loadout is §0.** The tables in §1-§5 describe each harness's
own defaults and mechanisms. Their "ours" columns predate §0 wherever the
two disagree.

**Legend**

- **V**: verified from source (file:line in the harness install) or from a
  captured request.
- **NV**: not verified.
- **chars**: `len(json.dumps(tool))` as offered, measured offline. Each
  section says how. Divide by 3-4 for a rough token count; none was
  tokenized.
- **default**: offered with the harness's shipped defaults for a custom
  OpenAI-compatible provider.
- **ours**: what our current test profile offers.

**Sources and versions**

| harness | where | version | per-harness doc |
|---|---|---|---|
| Hermes | `%LOCALAPPDATA%\hermes\hermes-agent` | commit `ee5ee84` | `bench/octopus/toolset_arms.py`; #47 |
| OpenCode | `C:\Users\jwals\AppData\Roaming\nvm\v20.15.0` (npm `opencode-ai`, a compiled bun binary) | 1.18.32 | `docs/HARNESS-OPENCODE.md` |
| Pi | `...\nvm\v24.21.0\node_modules\@earendil-works\pi-coding-agent` | 0.87.1 | `docs/HARNESS-PI.md` |
| Codex CLI | `...\nvm\v24.21.0\node_modules\@openai\codex` (native `codex.exe`) | 0.157.1 | `docs/HARNESS-CODEX.md` |

The Responses-API view of all four is `docs/HARNESS-RESPONSES.md`.

## 0. Default loadout (2026-09-26)

Operator, 2026-09-26: "All of our harnesses should have usable browser
tools attached to them, and a generally useful normal tool loadout", and
"LSP should probably be in the default loadout for everyone". So §8's
recommended profile is now the **default**, the browser included; it is
not an A/B arm any more. Each harness keeps an explicit opt-out
(Hermes `run.py --tools default`, the box `--loadout as-tested`).

Everything below is configuration, with no harness patched, and uses
official packages pinned by integrity. It was verified in Docker with **no
model** (a scripted stand-in plays the model and the harness executes the
calls for real), and **no run has used it yet**. The chars are
`len(json.dumps(tool))` as the harness sent it; none was tokenized (divide
by 3-4).

**Hermes changed on 2026-09-28** (operator: "hermes loads too many tools,
think it out to the tools we need for success"): its default is now the
**lean** arm, loadout-2 (below). The 2026-09-26 loadout is kept as the
opt-out `run.py --tools browser` (loadout-1).

| | Hermes (Octopus profile) | OpenCode 1.18.32 | Pi 0.87.1 | Codex 0.157.1 |
|---|---|---|---|---|
| tools offered | **10** (lean; loadout-1: 25) | 20 | 6 | 21 (8 + 13 in the `mcp__playwright` namespace) |
| chars | **14,855** (loadout-1: 28,210; before any loadout: 12 / 18,668) | **24,638** (was 9 / 20,473) | **4,689** (was 4 / 2,841) | **15,476** (was 5 / 5,593) |
| browser | Playwright MCP 0.0.82 (`mcp_servers.playwright`, three tools), run by `docker` from the harness box image in the sidecar's namespace (loadout-1: its own `browser` toolset through agent-browser 0.26.0 over CDP) | Playwright MCP 0.0.82 (`mcp.playwright`) over CDP | agent-browser 0.26.0 CLI through `bash`, over CDP (Pi has no MCP) | Playwright MCP 0.0.82 (`[mcp_servers.playwright]`) over CDP |
| where the browser runs | the Octopus sidecar: `octo-playwright:1.63.0`, Chromium 153, sharing the terminal container's namespace | the harness-box sidecar: the same image and script, sharing the harness container's namespace | the same | the same |
| looking at the page | `browser_take_screenshot`, then `vision_analyze(<the MEDIA: path>)` (below) | `playwright_browser_take_screenshot` | `agent-browser screenshot`, then `read` of the PNG (an image part) | `browser_take_screenshot` |
| diagnostics | pinned `tsc` + `pyright` mounted in the terminal container (`/opt/yamadori-tools`); the project's own `tsc` through its build. (loadout-1 adds the skill `type-check`) | its own LSP: diagnostics in every `write` / `edit` result, and the `lsp` tool | `tsc` + `pyright` through `bash`, skill `type-check` | `tsc` + `pyright` through `exec_command`, skill `type-check` |
| an installed package's API (2026-09-29) | none as a tool: Hermes' LSP is local-backend only and diagnostics-only; an LSP MCP server is a candidate awaiting approval ("The language-server MCP bridge", below) | the `lsp` tool: `hover` and `goToDefinition` on koota's `createWorld` answered in the box | skill `package-api`: a script that prints a package's exports and signatures from its own declarations with the image's TypeScript | – (not added) |
| verified (Docker, no model) | lean: `toolset_arms.py verify` **10/10, 2026-09-29** (first run); loadout-1: `toolset_arms.py verify --arm browser` 8/8 | `loadout_check.py opencode`: **12/12, 2026-09-29** (10 + the package hover and definition) | `loadout_check.py pi`: **19/19, 2026-09-29** (11 + screenshot, `read` as image, four package-api checks, skills listed, `prompt_cache_key`) | `loadout_check.py codex`: 9/9 |

The two scripts check the same things:

- the tool list;
- the model's shell serves a page on localhost;
- the browser opens it and reads both its `console.error` and its
  uncaught exception;
- the browser gets nothing from the host (`host.docker.internal:11434` and
  `192.168.65.254:1234` come back as the gate's 403 or
  ERR_PROXY_CONNECTION_FAILED);
- the shell gets nothing from the host either;
- a TypeScript and a Python type error come back.

The rows are in `bench/sandbox/results/loadout_check.jsonl`.

### Hermes, lean (the default since 2026-09-28, loadout-2)

The task is a web project: files, a shell with a dev server in the
background, and a browser to check progress -- open the page, read its
console errors, look at it. Tool by tool, measured with Hermes' own builder
offline (`toolset_arms.py lean --schemas`; chars = `len(json.dumps(tool))`):

| tool | chars | why it is there |
|---|---|---|
| `read_file` | 1,218 | read the project's files |
| `write_file` | 1,106 | create them; its result carries new lint errors |
| `patch` | 1,014 | edit them without rewriting |
| `search_files` | 2,329 | find a name or a file without `grep -r` into node_modules |
| `terminal` | 3,891 | `npm install`, the build, `tsc` |
| `process_manage` | 1,574 | the dev server in the background: start, read its log, stop |
| `vision_analyze` | 841 | the only Hermes tool that turns a screenshot into something the model sees; with it offered the proxy withholds `yama_describe_image` (`proxy.TOOL_OVERLAPS`) |
| `mcp__playwright__browser_navigate` | 344 | open the page (and reload it after a change) |
| `mcp__playwright__browser_console_messages` | 896 | console errors **and** uncaught page errors, with their text (`ReferenceError: undefinedFn is not defined`, loadout check) |
| `mcp__playwright__browser_take_screenshot` | 1,622 | a picture of the page, for `vision_analyze` |
| **total** | **14,855** | 10 tools (the tools sum to 14,835; the list's brackets and separators are the rest) |

Against the 2026-09-26 default (loadout-1, 25 tools / 28,210 chars) that is
13,355 chars fewer; against the profile before any loadout (12 / 18,668),
3,813 fewer. `file`, `terminal` and `vision` are whole toolsets; each tool
in them is used.

**Cut, and why:**

- the `browser` toolset: 10 tools plus the 5 `browser_vault_*` that ride
  with it and no key removes (15 tools, 11,255 chars; the vault five 4,339). `click`, `type`, `scroll`,
  `back`, `press` and `get_images` are interaction the check does not need;
  the vault tools are credential access; and its `browser_console` drops the
  text of uncaught errors (below, FOUND).
- `skills` (`skills_list` 321, `skill_view` 929, `skill_manage` 3,682: 4,932 chars, and the
  skills index in the system prompt, which Hermes adds only with these tools,
  `agent/system_prompt.py:303`). `skill_manage` writes persistent state. The
  type check is the project's own `tsc` (the build runs it); the checkers stay
  mounted at `/opt/yamadori-tools`. The screenshot skill taught the
  vision_analyze path route, which the MCP screenshot now hands over.
- the Tool Search bridges (off in every arm since loadout-1).

**How the MCP server runs, with no host install.** The profile's
`mcp_servers.playwright` (one tagged line, `toolset_arms.mcp_line`, rewritten
per run with the run's sidecar) makes Hermes spawn it over stdio as

```
docker run -i --rm --init --pull never --network container:<the run's sidecar>
  --label octo-browser-mcp=1 --label octo-mcp-of=<sidecar> --user 1000:1000
  --cap-drop ALL --security-opt no-new-privileges --pids-limit 1024
  yamadori-harness-box:oc1.18.32-pi0.87.1-cx0.157.1-tools1
  playwright-mcp --cdp-endpoint http://127.0.0.1:9322 --no-webmcp --output-dir /home/node/.cache/playwright-mcp
```

It is the image the box already built (never pulled), in the sidecar's
network namespace, attached to the sidecar's Chrome: it sees only what that
browser sees, and the gate publishes nothing on the host for this arm.
`tools.include` keeps the three tools (`tools/mcp_tool_registration.py`
`_make_tool_filter`); `tools.resources` / `tools.prompts` false keep Hermes'
generated `list_resources` / `read_resource` / `list_prompts` / `get_prompt`
out (`_select_utility_schemas`). Hermes passes the stdio child a filtered
environment that keeps the Windows variables the docker CLI needs
(`tools/mcp_tool_config.py:74-83`). When Hermes exits the server's stdin
closes and `--rm` removes it; `run.py` removes any left by label. The
sidecar's error mirror is off for this arm: Playwright reports page errors
itself.

**What Hermes does with an MCP screenshot** (`tools/mcp_tool_content.py`
`_cache_mcp_image_block`): the image block is written to the profile's image
cache on the host and the tool result carries `MEDIA:<host path>` as TEXT.
No pixels reach the model, native mode or not. `vision_analyze` reads that
path: under the docker backend a host path inside a media cache root is a
permitted host read (`tools/image_source.py` `_permitted_host_read_target`).
So looking is two calls: `browser_take_screenshot`, then
`vision_analyze(<the MEDIA path>)`, which goes to Hermes' auxiliary vision
client and our proxy (text mode, docs/VISION.md 4b). Loadout-1's
`browser_vision` did it in one call. The Playwright text of the same result
also names its own container path (`/home/node/.cache/playwright-mcp/...`),
which `vision_analyze` cannot read; which path the model picks is not
observed.

**Found while measuring:** building the tool list connects to the profile's
endpoint (`check_vision_requirements` -> `detect_local_server_type`,
127.0.0.1:1234). The offline builder and `verify` now refuse every connect
to a stack port and print what was tried (`toolset_arms._GUARD`); before
2026-09-28 `--schemas` reached the proxy.

**The MCP tools' schemas in the measurement** are Playwright MCP 0.0.82's
`tools/list` as captured from OpenCode's request in the box loadout check
(`bench/octopus/playwright_mcp_0.0.82_tools.json`), registered through
Hermes' own cached-manifest path (`_register_from_cache_sync`), so the
include filter and name prefixing are Hermes'. A live connect may differ by
whatever OpenCode changed in the schemas (NV).

**Verified 2026-09-29** (first run): `python bench/octopus/toolset_arms.py
verify` (Docker, no model), 10/10:

| check | evidence |
|---|---|
| `tools_exactly_lean` | the 10 tools of the table, nothing else |
| `terminal_serves_page`, `terminal_host_blocked`, `terminal_type_check` | `SERVER_200`, `PROXIED_HOST_403`, `/opt/yamadori-tools/bin/tsc` TS2322 |
| `browser_opened_page` | `browser_navigate` -> `Page Title: OctoPage`, `Console: 2 errors` |
| `browser_console_error`, `browser_page_exception` | `browser_console_messages(level=error)` -> `[ERROR] OCTO_CONSOLE_ERROR` and `ReferenceError: undefinedFn is not defined at http://localhost:3001/:1:149` |
| `screenshot_media_path_readable` | `browser_take_screenshot` -> `MEDIA:<temp home>\cache\images\img_<hex>.png`, read the way `vision_analyze` reads it: a 7,804-byte PNG, origin `file` |
| `browser_host_blocked` | `host.docker.internal:11434` and `192.168.65.254:1234` -> `net::ERR_PROXY_CONNECTION_FAILED` |
| `mcp_containers_removed` | 0 left by label |

The Hermes probe's four connects to `127.0.0.1:1234` (the vision
requirement check, above) were refused by the guard, as designed. The
verify prints its record; it writes no results file.

**No skills reach the lean arm.** The `skills` toolset is cut, and Hermes
adds the skills index to the system prompt only with it
(`agent/system_prompt.py:303`, `_skills_prompt`); `skills.auto_load`
(pinned skills in the prompt) has the same gate (`_auto_load_parts`, :316).
So `bench/octopus/hermes_skills/` is read by loadout-1 only. The slow-WebGL fact (2026-09-29, `look-at-a-screenshot`,
Pi's `page-check`) therefore reaches Hermes only under `--tools browser`.
Config-only ways to give it to the lean arm, none taken: put `skills` back
(4,932 chars and `skill_manage`), or a project context file (Hermes reads
`AGENTS.md` / `CLAUDE.md` / `.cursorrules` from the working directory,
`agent/system_prompt.py:709`, `agent/coding_context.py:38`; that is the
task's workspace, so it would change the task, not the harness).

### Hermes, loadout-1 (`--tools browser`, the opt-out; the default 2026-09-26 to 09-28)

The tools, per Hermes' own builder (`toolset_arms.py browser --schemas`,
offline):

- `read_file`, `write_file`, `patch`, `search_files`;
- `terminal`, `process_manage`;
- `vision_analyze`;
- `skills_list`, `skill_view`, `skill_manage`;
- `browser_navigate`, `browser_snapshot`, `browser_click`, `browser_type`,
  `browser_scroll`, `browser_back`, `browser_press`, `browser_get_images`,
  `browser_vision`, `browser_console`;
- `browser_vault_list`, `_unlock`, `_fill`, `_save_login`, `_enter_code`.

`make_profile.py` writes the profile with the loadout, and `run.py`
re-applies it on every run (`toolset_arms.apply`, tagged lines):

- **Browser** (`--browser-host sandbox`, the default): `browser` in
  `platform_toolsets.cli`, with these keys:
  - `browser.backend: "off"`, so no `browser_exec` (host Python);
  - `allow_private_urls: true`, required for localhost;
  - `restrict_evaluate: true` (now on both hosts);
  - `inactivity_timeout: 21600`;
  - `cdp_url` pointing at the sidecar through the gate;
  - `browser-cdp` disabled.

  agent-browser 0.26.0 is installed in `hermes-home\node`.
- **Tool Search off** (`tools: {tool_search: {enabled: "off"}}`):
  `process_manage` is offered directly. The three bridge tools go, which
  removes 3,315 chars and adds 1,574.
- **Vault tools: they cannot be excluded.** They are in the `browser`
  toolset with a browser check_fn, and no key removes one tool (§2). They
  cost 4,339 chars. The profile turns off 1Password and Bitwarden as vault
  sources (`vault.*.enabled: false`), which leaves only the profile's own
  local vault. It is empty.
- **`skill_manage`: it cannot be excluded either.** It is in the `skills`
  toolset with `skills_list` and `skill_view`. The loadout sets
  `skills.create_dir: "run-skills"`, and `run.py` empties that folder before
  each run's first prompt, so a skill the model saves never reaches the
  next run. Its edits to existing skills are not contained; `run.py`
  re-installs ours on every prompt.
- **Off, for benchmark hygiene** (the Blank Slate `disabled_toolsets` is
  otherwise unchanged): memory, session_search, delegation, cronjob, web,
  todo, code_execution, clarify, computer_use, image_gen, tts and the
  messaging toolsets.
- **Types.** Hermes runs LSP only on the local backend
  (`tools/file_operations_lint.py` `_lsp_local_only`). Its post-write lint
  skips a `.ts` file under a `tsconfig.json`, and otherwise needs the
  project's own `tsc`. So the terminal container mounts the volume
  `yamadori-typecheck-tools1` read-only at `/opt/yamadori-tools`:
  - it holds typescript 5.9.3 and pyright 1.1.414, copied from the harness
    box image with `--network none`, and the volume's label names that
    image;
  - the skill `type-check` (in `bench/octopus/hermes_skills/`) says to run
    `npx --no-install tsc --noEmit -p .` or `/opt/yamadori-tools/bin/tsc` /
    `pyright`.
- **FOUND: Hermes drops the text of uncaught page errors.** This is a
  version mismatch between Hermes and agent-browser, and neither is ours
  to patch.
  - Hermes ee5ee84's `browser_console` reads `errors[].message`
    (`tools/browser_tool.py`).
  - agent-browser 0.26.0, the only release in Hermes' pin `^0.26.0`, names
    that field `text`.
  - So the model got `js_errors: [{"message": ""}]`. In #47's terms, "PLAYER.hit is not a function" would
    have arrived as an empty string.

  The sidecar now carries a fix. With `SIDECAR_ERROR_MIRROR=1`, its own
  CDP client adds a listener to every page before the page's scripts run.
  The listener repeats each uncaught error and unhandled rejection as a
  `console.error` line, which Hermes passes through
  (`browser_sidecar.py`, "error mirror"). Verified: `browser_console` now
  returns `Uncaught ReferenceError: undefinedFn is not defined at ...`.
  The empty `js_errors` entry is still there.

### OpenCode

- **Tools:** `bash`, `read`, `edit`, `write`, `glob`, `grep`, `lsp`, and
  13 `playwright_browser_*` tools: `navigate`, `navigate_back`, `snapshot`,
  `click`, `type`, `press_key`, `select_option`, `wait_for`, `evaluate`,
  `console_messages`, `network_requests`, `take_screenshot`,
  `handle_dialog`.
- **Permissions:** the six tools and `lsp` are `allow`. `task`,
  `todowrite`, `skill`, `question`, `webfetch` and `websearch` are `deny`.
  `playwright_*` is `deny`, followed by `allow` for the 13. The last
  matching rule wins, and the check confirmed that exactly those 13 are
  offered. `formatter: false`.
- **Browser:** `mcp.playwright = {type: "local", command: ["playwright-mcp",
  "--cdp-endpoint", "http://127.0.0.1:9322", "--no-webmcp", "--output-dir",
  ...]}`.
  - 0.0.82 offers 25 tools by default, 20.6k chars. The 12 left out are
    `run_code_unsafe` (arbitrary Node code), drag/drop, emulate_media,
    file_upload, find, fill_form, hover, network_request, tabs, resize and
    close.
  - `--no-webmcp`: a page cannot register tools into the model's list.
  - Snapshots from `browser_navigate` are written to files; `browser_snapshot`
    returns them inline.
- **Language servers:**
  - an `lsp` object turns on all 37 built-ins (`LSP.state` @101919300), so
    each is set to `{disabled: true}`;
  - two of our own are added. Their own ids give them the project
    directory as root; the built-in `typescript` needs a lockfile.
    - `typescript-box`: `typescript-language-server --stdio` 6.0.1. It
      uses the project's typescript when the project has one, else the
      image's 5.9.3.
    - `pyright-box`: `pyright-langserver --stdio` 1.1.414.
  - `OPENCODE_DISABLE_LSP_DOWNLOAD=1` stops the downloads.
  - `OPENCODE_EXPERIMENTAL_LSP_TOOL=1` turns on the `lsp` tool, and it is
    the only experimental flag set. The tool has no diagnostics operation;
    diagnostics arrive in the results.
- **Verified:**
  - a `write` of a `.ts` file with an error returned `LSP errors detected
    in this file, please fix: ... ERROR [1:14] Type 'string' is not
    assignable to type 'number'.`;
  - a `.py` file returned pyright's `reportArgumentType`;
  - `lsp` `hover` answered.
- **An installed package's API through `lsp`** (2026-09-29,
  `loadout_check.py opencode`, 12/12). The check copies koota 0.6.6 from a
  pagoda run's `node_modules` into the project (no download) with
  `src/api_use.ts` (`import { createWorld } from 'koota'`) and a bundler
  `tsconfig.json`:
  - `lsp` `hover` at `createWorld` returned `(alias) function
    createWorld(options: WorldOptions): World (+1 overload)`;
  - `lsp` `goToDefinition` returned
    `file:///work/node_modules/koota/dist/index.d.ts`, line 3 (0-based),
    both overloads.

  So the model can ask the language server instead of reading the package.
  What it returns is a location and one hover; the definition's text is a
  `read` of that file at that line. No skill teaches the `lsp` tool yet
  (the tool's own description is the only prompt).

### Pi

- **Tools:** `read`, `bash`, `edit`, `write`, `grep`, `find`. They come from
  `settings.json` `defaultTools`. The box now ships fd 10.5.0 (the
  sharkdp/fd release, pinned by sha256), and Pi runs with `PI_OFFLINE=1`,
  so neither `find` nor `grep` downloads anything.
- **Pi has no MCP client.** 0.87.1 has none in `dist/` or `docs/`.
  Third-party extensions add one, for example `pi-mcp-adapter` (nicobailon,
  MIT) and `pi-mcp-extension`. They are not official, run in-process with
  full permissions, and were not installed.
- **The nearest config-only option, built:** the agent-browser 0.26.0 CLI
  (Vercel, the same package Hermes drives) is on `PATH`. The skill
  `page-check` (`~/.agents/skills/`, which Pi lists in its prompt) teaches
  `agent-browser --cdp 9322 open|errors --json|console|snapshot|click|screenshot`.
  - Use `errors --json`. Plain `errors` prints an empty marker per error in
    0.26.0, the same field mismatch as in Hermes.
- **Looking at the page** (2026-09-28): `agent-browser --cdp 9322 screenshot
  /tmp/page.png`, then `read /tmp/page.png`. Pi's `read` returns an image
  part and, to an openai-completions provider, a synthetic user turn
  "Attached image(s) from tool result:" with an `image_url` part, when the
  model entry's `input` has `"image"` (docs/HARNESS-PI.md s3: observed with a
  user-attached and a read image; the proxy describes it with
  `yama_describe_image`). The skill `page-check` now says so. Not in the
  loadout check (NV in the box).
- **The model entry** is the host test's `models.json`
  (`octo\pi-test\cfg\full.json`) with `contextWindow`, `maxTokens` and
  `input` replaced, per run, by the proxy's advertised `/v1/models` row
  (`context_length` = the main share, `max_completion_tokens`,
  `modalities.input`), read through the run's relay (`run.py --task pagoda
  --harness pi`; `pagoda.pi_models`). The file's 132096 / 26419 were an
  earlier deploy's window.
- **Thinking level:** Pi's `--thinking <level>` (`dist/cli/args.js`; off
  minimal low medium high xhigh max). The entry's `thinkingLevelMap` sends
  `xhigh` as `reasoning_effort: "xhigh"` and `max` as `"max"`
  (`pi-ai/dist/api/openai-completions.js:714`; xhigh and max are offered only
  when the map names them, `models.js` `getSupportedThinkingLevels`).
- **`PI_CACHE_RETENTION=long`** is now in the box's Pi environment
  (docs/HARNESS-PI.md "The config used"): Pi sends `prompt_cache_key` = its
  session id.
- **Types:** `tsc` and `pyright` are on `PATH`. The skill `type-check` says
  how to use them.
- **An installed package's API: skill `package-api`** (2026-09-29). Pi has
  no language server and no MCP client, so the skill ships a script
  (`bench/sandbox/harness_skills/package-api/api.cjs`, copied with the
  SKILL.md into `~/.agents/skills/package-api/`; `harness_box.write_home`
  now copies a skill's whole folder). It runs the TypeScript compiler
  already in the image (`/opt/harness/node_modules/typescript` 5.9.3, or the
  project's own) over a one-line probe file that imports the module, with
  the project's `tsconfig.json`, and prints what a hover would:
  - `api.cjs koota`: every export on one line with its signature, then the
    package's other entry points (from its `exports` map);
  - `api.cjs koota createWorld World World.query`: every overload, a type's
    own members (members inherited from another package are counted by
    package, e.g. `(+276 members from @types/react)` for r3f's
    `CanvasProps`), the doc comment, and `file:line`;
  - a name the root does not export is looked up in the entry points and
    one level into namespaces (`api.cjs math fromEuler` -> `quat.fromEuler`;
    `mulberry32` -> `math/random`); a miss says so and exits 1.

  Read-only, no network, nothing downloaded. Output bytes on the pagoda
  stack (host Node 22, the p3 workspace, ~1 s a call, 3 s for r3f's whole
  list): the koota list 67 exports / 7.2k, `createWorld` 0.16k, `World`
  2.9k, r3f's `useFrame` + `Canvas` 0.9k, `CanvasProps` 3.6k, the math list
  2.5k; whole lists run large for big packages (r3f 20.4k, three 46.4k), so
  the skill's table leads with names and patterns and says three's list
  has about 680 exports (681 in @types/three 0.186). The offline
  evaluation against the runs' own probes is below ("Would it have cut the
  wandering reads?").
- **The slow-WebGL fact** (2026-09-29, skill `page-check`): the sidecar's
  Chromium renders WebGL in software (`browser_sidecar.py`:
  `--use-angle=swiftshader`), so a heavy scene can take seconds per frame
  and `open` / `reload` / `screenshot` / `eval` answer `CDP command timed
  out` while it renders. pagoda-p3 (Pi, low) got 13 such results between
  its calls 200 and 283 and spent calls 221-288 debugging the browser
  (Chrome processes, raw CDP scripts, a minimal WebGL page) with
  `errors --json` empty throughout. The skill now says, as a fact, that the
  page is slow, not broken, and to read `errors --json` and take one
  screenshot; with no errors the page is working. The same paragraph is in
  Hermes' `look-at-a-screenshot` (1.1.0), which only loadout-1 reads (§0,
  "No skills reach the lean arm").
- **Verified:**
  - the page's title;
  - `BOX_CONSOLE_ERROR`;
  - `undefinedFn` in `errors --json`;
  - the gate's refusals of the host;
  - example.com through the gate;
  - TS2322 and pyright's error;
  - `find` and `grep` working;
  - (2026-09-29) `package-api` in the box: both `createWorld` overloads,
    `World.query`, `other entry points: koota/react`, and `koota exports no
    nope` with exit 1; all three loadout skills listed in Pi's system prompt
    (the `package-api` entry is ~0.4k chars of it).

#### Would it have cut the wandering reads? (offline, 2026-09-29)

What the runs searched for inside their dependencies, against what the
script answers. Script: the session scratchpad `eval_api.py` (not kept in
the repo) over `bench/atomic/replay_logs.py`'s events. For every call that
read inside `node_modules/` it takes the package and the identifiers of the
call's grep / search pattern, then asks `api.cjs` in that run's own
workspace (`C:\Users\jwals\octo\runs\<run>\pagoda`, the versions the run
installed). Five runs have a workspace: pagoda-h4, h6, p1 (max), p3, p4.
n = 292 distinct (package, identifier) pairs; the dependency reads of these
five runs returned 636,903 chars.

| answer | identifiers | examples |
|---|---|---|
| `api.cjs <pkg> <name>` finds it (an export, an entry point's export, or a namespace member) | 94 (32%) | `createWorld`, `useFrame`, `vec3`/`quat` functions, `mulberry32` |
| in the package's API output as a member or prop (`api.cjs koota World`, `CanvasProps`) | 81 (28%) | `query`, `spawn`, `updateEach`, `readEach`, `shadows`, `camera` |
| not in the API | 117 (40%) | implementation (`createQueryInstance`, `R3F_CATALOGUE`, `chooseRenderer`), three's shader defines (`USE_INSTANCING_COLOR`), React hooks grepped inside r3f, names that do not exist (`OrbitalControls`, `SkyDome`, `easeInOutCubic`) |

Reading: about 60% of what the model went into node_modules for is the
package's API, which the script answers in a few hundred characters to a
few thousand; the other 40% is implementation or a wrong name (the script
answers a wrong name with "exports no X; near: ..."). Much of p3's
implementation reading was renderer internals during its frozen-browser
episode, which the slow-WebGL fact addresses instead. **Limits:** the
identifiers come only from grep/search patterns (a plain `read` of a
`.d.ts` carries none); "in the API output" is a word match over the whole
API dump, an upper bound; and **no model has used the skill** -- whether
Bonsai reaches for it instead of `sed -n` on a bundle is unmeasured until a
Pi run.

### Codex

- **Tools:** `exec_command`, `write_stdin`, `apply_patch` (the catalog
  entry), `view_image`, `update_plan`, the 13 Playwright tools in the
  namespace `mcp__playwright`, and three tools that come with ANY MCP
  server.
  - The three are `list_mcp_resources`, `list_mcp_resource_templates` and
    `read_mcp_resource`, 1,943 chars. No key was found that removes them.
- **Browser:** `[mcp_servers.playwright]`:
  - `command = "playwright-mcp"`, the same args as OpenCode;
  - `enabled_tools` is the 13;
  - `startup_timeout_sec = 60`, `tool_timeout_sec = 120`;
  - `default_tools_approval_mode = "approve"`. Without it, every tool not
    marked read-only was refused under `approval_policy = "never"` ("MCP
    tool call requires approval, but approval policy is never"; seen in the
    first check).
- **`[sandbox_workspace_write] network_access = true`.** Without it, a dev
  server started in Codex's bwrap sandbox lives in an empty network
  namespace, and the browser cannot reach it.
  - What this opens is the box's own network: the sidecar's localhost, the
    gate (public 80/443) and the one forward. OpenCode's and Pi's shells
    have always had that.
  - The key stays out of the shell (`KEYVARS=0`).
- **Types:** `tsc` and `pyright` through `exec_command`. The skill
  `type-check` is in `$CODEX_HOME/skills`.
- Update_plan was already on, and apply_patch already came from the catalog
  entry.

### The language-server MCP bridge: vetted, none adopted

The operator asked for LSP diagnostics everywhere. For Hermes, Pi and Codex
that would need an MCP bridge to a language server. Pi could not take one
anyway. The candidates, 2026-09-26:

| candidate | who / licence | state | verdict |
|---|---|---|---|
| `cclsp` 0.7.0 (npm) | ktnyt, one maintainer, MIT; no provenance attestation; a 1.2 MB bun bundle that includes its setup wizard, which can `npx` other packages | last release 2026-01 | **REJECTED, measured.** In the box, with typescript-language-server 6.0.1 and pyright 1.1.414, `get_diagnostics` answered "No diagnostics found ... The file has no errors, warnings, or hints." for a `.ts` file with TS2322 and a `.py` file with a pyright error. It did so on every call, after 5-8 s waits, with and without a `tsconfig.json` or a read-only mount. `tsc` and `pyright` on the same files found the errors. Its stderr: `Failed to parse LSP message: SyntaxError: Unexpected non-whitespace character after JSON`, and `textDocument/diagnostic` timed out after 30 s. A checker that reports clean code when the code is broken is worse than no checker. It was removed from the image. |
| `isaacphi/mcp-language-server` | a user, BSD-3; 1.6k stars | last commit 2025-06-03, no release binaries, an open issue about corrupted LSP frames; Go source only | not maintained; not installed |
| `oraios/serena` | Oraios AI (an organisation); the app is GPL-3.0-or-later, SolidLSP is MIT; 29.8k stars | v1.7.0, 2026-08-09; has `GetDiagnosticsForFile` | the maintained option. Not installed: Python with a large dependency tree to lock with hashes; npm-installs typescript-language-server 5.1.3 + typescript 5.9.3 at first use unless pre-seeded; 30+ tools including shell execution, file edits and memories; a web dashboard. **The candidate to evaluate next** if a bridge is wanted |
| Microsoft / TypeScript | – | TypeScript 7's native `tsc --lsp` is an LSP server, not MCP; no official MCP bridge found | – |

So Hermes, Pi and Codex get the coordinator's named fallback: pinned `tsc`
and `pyright` run through the shell, plus a skill that says when and how.
OpenCode is the only harness with type errors after every edit.

**2026-09-29: the question is now "learn a package's API without reading
node_modules"** (hover, definition), not only diagnostics. Per harness:

- **OpenCode**: its `lsp` tool answers hover and goToDefinition on an
  installed package (verified, OpenCode above). Nothing to add.
- **Pi**: no MCP client, so no bridge; the skill `package-api` (Pi above)
  gives the same answers through `bash`, config only.
- **Hermes**: its own LSP cannot serve the docker backend. Read from ee5ee84:
  `tools/file_operations_lint.py` `_lsp_local_only` returns true only for a
  `LocalEnvironment` ("LSP servers run on the host and can't see files
  inside Docker/Modal/SSH/Daytona sandboxes"), with no config key around
  it; and Hermes' LSP is diagnostics-only in any case (it feeds
  `write_file` / `patch` results; `website/docs/user-guide/features/lsp.md`),
  with no hover or definition tool for the model. The config-only route is
  an MCP server run like Playwright MCP (`docker run` of the harness box
  image in the run's namespace, `tools.include` to pick its tools). That
  needs a new package in the image: **a download, for the operator to
  approve.** Candidates, 2026-09-29 (read from their pages, nothing
  installed):

  | candidate | who / licence | what it offers | concerns | verdict |
  |---|---|---|---|---|
  | `@mizchi/lsmcp` 0.10.0 (npm) | mizchi (one maintainer), MIT; ~450 stars; signed registry releases (provenance not checked) | `lsp_get_hover`, `lsp_get_definitions`, `lsp_find_references`, `lsp_get_diagnostics`, `lsp_get_document_symbols`, `lsp_get_signature_help`, and `search_external_library_symbols` / `resolve_symbol` (library symbols, the node_modules question itself); typescript-language-server or tsgo | also offers editing tools (`replace_range`, `replace_regex`, rename, delete symbol) and a memory store (`write_memory`) that `tools.include` must leave out; Node >= 22 (the image has 24.21.0); its dependency tree is unlocked and unread; cclsp's failure mode (reports clean when broken) must be ruled out by the same box test | **the candidate to vet first**: TypeScript-first and the read tools map one-to-one to the question. Vetting = lock the tree by integrity, read `dist/` for network and exec, then the cclsp test (a TS2322 must come back) plus hover/definition on koota in the box |
  | `oraios/serena` 1.7.0 | Oraios AI; GPL-3.0-or-later app, MIT SolidLSP | symbol overview, find symbol, references, `GetDiagnosticsForFile` | as in the 2026-09-26 row above (Python tree, 30+ tools with shell and edits, a dashboard, first-use downloads) | second choice |
  | `isaacphi/mcp-language-server` | BSD-3 | `definition`, `references`, `hover`, `diagnostics` | unmaintained since 2025-06 (above) | no |
  | `cclsp` 0.7.0 | MIT | – | reported clean code for TS2322 (above) | rejected |

  The other bridges a search turned up (`Tritlo/lsp-mcp`, `ts-lsp-mcp`,
  `jgauffin/ts-language-mcp`, `lsp-mcp-rs`) are one-person projects not read
  further. Alternative with no download: the `package-api` script is plain
  Node over the TypeScript already in `/opt/yamadori-tools`, so Hermes'
  `terminal` could run it too (checked: the Octopus terminal image, Node
  22.23.3, with the tools volume, printed koota's `createWorld`) -- but the lean arm has no skills channel to
  teach it (§0), so it would reach loadout-1 only. Not done.
- **Codex**: not changed here; its `exec_command` could run the same
  script from `$CODEX_HOME/skills` if the skill were added to
  `loadout_skills("codex")`.

## 1. The matrix: which tool does what, per harness

"–" means no such built-in tool.

| what it helps with | Hermes | OpenCode | Pi | Codex |
|---|---|---|---|---|
| **read files** | `read_file` (on) | `read` (on) | `read` (on) | – (reads through `exec_command`) |
| **write / edit files** | `write_file`, `patch` (on) | `write`, `edit` (on); `apply_patch` replaces them for GPT-family model ids (below) | `write`, `edit` (on) | `apply_patch` (freeform; **needs a model catalog entry**) |
| **run commands** | `terminal` (on; local host shell or docker); `process_manage` (behind tool_search by default) | `bash` (on, host) | `bash` (on, host, Git Bash on Windows); `powershell` (off) | `exec_command`, `write_stdin` (on; Windows sandbox) |
| **search code** | `search_files` (on; ripgrep content + file glob) | `grep`, `glob` (on) | `grep`, `find` (off; `--tools`), `ls` (off) | – (`rg` through `exec_command`) |
| **run / verify a web app** (browser, console errors, screenshots) | `browser_*` (10 tools incl. `browser_console`, `browser_vision`; **off** in ours, and needs agent-browser + `browser.backend: "off"`) | – (MCP only) | – (extension or MCP only) | – (MCP only; the app's browser features are not CLI tools) |
| **look at images** | `vision_analyze` (on in ours) | `read` of an image file (returns it; needs `modalities`) | `read` of an image file (returns it; needs `input: ["text","image"]`) | `view_image` (on) |
| **web search / fetch** | `web_search`, `web_extract` (on by default, third-party keyless providers; off in ours) | `webfetch` (on; any URL incl. loopback; we deny it); `websearch` (off; Exa/Parallel flags) | – | `web_search` (hosted; `web_search = "disabled"`) |
| **planning / todo** | `todo_list` (behind tool_search) | `todowrite` (on) | – (example extension `todo`) | `update_plan` (off until `[tools.update_plan] enabled = true`) |
| **subagents** | `delegate_task` (on by default; off in ours) | `task` (on) | – (example extension `subagent`) | `multi_agent_v1` namespace: `spawn_agent` etc. (on) |
| **skills / knowledge** | `skills_list`, `skill_view`, `skill_manage` | `skill` (on) | skills listed in the prompt, read with `read` | skills listed in the developer message |
| **ask the user** | `clarify` | `question` (off in `run`, on in the TUI and `--fork`) | – | `request_user_input` (on; "Plan mode only") |
| **code intelligence / type errors after an edit** | lint and `tsc` deltas in `write_file`/`patch` results; LSP on the local backend only | LSP errors in `edit`/`write` results with the `lsp` key; `lsp` tool (experimental) | – | – |

The tools that would most help this model and are **not** on by default
(or not on in our profile) are ranked in §6.

## 2. Hermes (ee5ee84)

Measured with Hermes' own `model_tools.get_tool_definitions`, driven
through `hermes_cli.tools_config._get_platform_tools(cfg, "cli")` (the
`cli.py:1516-1524` path). Each scenario ran in a scratch HERMES_HOME
holding a copy of `config.yaml` and no keys, with a socket guard that
refused every connect. Scripts: session scratchpad `hermes-inv\`.

### Selection mechanics (V)

- **Toolsets.** Toolsets are in `toolsets.py:77-183`, the core list at
  `:12-41`. The CLI default toolset is `hermes-cli`
  (`hermes_cli/platforms.py:15`). Default-off toolsets are
  `hermes_cli/tools_config.py:97`.
- **Profile keys.** `platform_toolsets.cli` is a list of toolsets.
  `agent.disabled_toolsets` is applied LAST (`tools_config.py:622-625`,
  `model_tools.py:333-340`). It takes toolset names only, so **no key
  removes a single tool**.
- **Filters that follow.** A tool's check_fn filters it next
  (`tools/registry.py:837-865`). Then `browser_exec` is dropped unless
  `terminal` is present (`model_tools.py:391-395`).
- **Tool Search** is on by default: `tools.tool_search.enabled: "auto"`
  (`hermes_cli/config_defaults.py:1959,1967`), with a `defer` list at
  `:1986-1990`. It hides `todo_list`, `process_manage`, `session_search`,
  `computer_use`, `cronjob_manage` and `image_generate` behind three
  bridge tools:
  - `tool_search` (1,829-2,107 chars);
  - `tool_describe` (496);
  - `tool_call` (992).

  MCP tools are always deferrable unless tool_search is off
  (`tools/tool_search.py:150-165`).
- **The :1234 probe.** Building the tool list probes our endpoint. In the
  Octopus profile `check_vision_requirements` reaches
  `agent/model_metadata.py:761` `detect_local_server_type`, which walks
  `/api/v1/models`, `/api/tags`, `/v1/props`, `/props` and `/version`
  against `127.0.0.1:1234` (8 connects per build, refused by the guard).
  Every Hermes startup sends these probes to the proxy.

### Tools

| tool | category | default | ours | enable / needs | chars | risk | source | V |
|---|---|---|---|---|---|---|---|---|
| `read_file` | read | on | on | always | 1,218 | runs through the terminal backend (in the container under docker) | `tools/file_tools.py:1133`, reg `:1371` | V |
| `write_file` | write | on | on | always | 1,106 | same. The result carries new lint errors: `.py/.json/.yaml/.toml` in-process, and `node --check`, `py_compile`, single-file `tsc` (`tools/file_operations_lint.py:17-23,145-157`) | `:1155`, `:1372` | V |
| `patch` | edit (replace + V4A) | on | on | always | 1,014 | same | `:1173`, `:1395` | V |
| `search_files` | search code (ripgrep + glob) | on | on | always | 2,329 | same | `:1270`, `:1396` | V |
| `terminal` | run commands | on | on | `terminal.backend` (default `local` = **host shell**, `config_defaults.py:278`; ours `docker`). Docker: cap-drop ALL, no-new-privileges, pids limit, **network on** unless `docker_network: false` (`tools/environments/docker.py:680-685`) | 3,891 | local = host; docker can reach host services | `tools/terminal_tool.py:1377`, `:1490` | V |
| `process_manage` | poll / kill background processes (dev servers) | deferred | deferred | direct with `tools.tool_search.enabled: off` | 1,574 | as terminal | `tools/process_registry.py:2461`, `:2638` | V |
| `execute_code` | run Python that calls Hermes tools | on | off | `code_execution` toolset | 2,999 | local backend = a **host** kernel | `tools/code_execution_tool.py:886`, `:924` | V |
| `vision_analyze` | look at images | off (no aux vision client without keys) | on (routes to :1234, our `yama_describe_image`) | an aux vision client | 841 | takes an http URL, a path or a data URL (#46) | `tools/vision_tools.py:858`, `:930` | V |
| `browser_navigate`, `browser_snapshot`, `browser_click`, `browser_type`, `browser_scroll`, `browser_back`, `browser_press`, `browser_get_images`, `browser_vision`, `browser_console` | run / verify a web app | **not offered on this host**: `browser.backend` unset selects `browser_exec`, because `browser-use.exe` is installed | **on since 2026-09-26** (the default loadout, §0; sandbox sidecar) | toolset `browser`, `browser.backend: "off"`, agent-browser `^0.26.0` + Chromium (`tools/browser_tool.py:147-149`; `tools/browser_tool_install.py:295-327`), `browser.allow_private_urls: true` for localhost, `inactivity_timeout` (120 s) raised | 6,946 as offered (#47) | Chrome on the **host**. With private URLs it can GET host loopback (:1234, :11434). `browser_console(expression=)` runs JS in the page; `restrict_evaluate` limits it | `tools/browser_tool.py:467-607`, reg `:1316-1350` | V |
| `browser_exec` | web app, via model-written Python | **on on this host** (whenever the browser toolset is on and `backend` is unset) | off | browser-use CLI | 3,478 | **model-written Python run on the host**, outside docker | `tools/browser_use_cli.py:219-227,790,813` | V |
| `browser_vault_list`, `_unlock`, `_fill`, `_save_login`, `_enter_code` | credentials | come with any browser backend | off | no key removes them alone | 4,339 | **credential access** | `tools/browser_vault_tool.py:700-770` | V |
| `browser_cdp`, `browser_dialog` | raw CDP | off | off | `browser.cdp_url` | 3,549 / 1,688 | raw CDP | `tools/browser_cdp_tool.py:325` | V |
| `web_search`, `web_extract` | web search / fetch | on (keyless free-tier ring) | off | `web.search_backend` / `extract_backend` (`config_defaults.py:370-395`); local: `searxng` + `SEARXNG_URL=http://127.0.0.1:8888` | 794 / 1,112 | **queries go to third parties** by default | `tools/web_tools.py:475,497,519,525` | V |
| `todo_list` | planning | deferred | off | toolset `todo` | 1,345 | none (in memory) | `tools/todo_tool.py:222`, `:281` | V |
| `delegate_task` | subagents | on | off | `delegation.*` | 4,636 | more contexts on the same endpoint | `tools/delegate_tool.py:635`, `:738` | V |
| `memory` | cross-session memory | on | off (`memory.memory_enabled: false`) | | 3,524 | contaminates runs | `tools/memory_tool.py:306` | V |
| `session_search` | past sessions | deferred | off | | 4,009 | | `tools/session_search_tool.py:652` | V |
| `skills_list`, `skill_view`, `skill_manage` | skills | on | on | toolset `skills` (all or nothing) | 321 / 929 / ~3,785 | `skill_manage` **writes persistent skills** to HERMES_HOME | `tools/skills_tool.py:655,670`; `tools/skill_manager_tool.py:878` | V |
| `clarify` | ask the user | on | off | | 1,695 | blocks a batch run | `tools/clarify_tool.py:245` | V |
| `computer_use` | desktop control | deferred (cua-driver present) | off | | 5,860 | **host desktop** | `tools/computer_use/schema.py:189` | V |
| `image_generate`, `text_to_speech`, `cronjob_manage`, `manage_connections`, `video_analyze`, `x_search`, `ha_*`, `kanban_*`, `spotify_*`, `discord*`, `feishu_*`, `yb_*`, `a2a_*` | not SWE | mixed | off | | – | third parties / host | `toolsets.py` | V |
| MCP: `mcp__<server>__<tool>` | anything, e.g. a Playwright server | – | lean (§0): `playwright`, 3 tools; an image result becomes a `MEDIA:<host path>` text line (`tools/mcp_tool_content.py`) | `mcp_servers.<name>: {command, args, env}` or `{url, headers}`, with per-server `tools.include/exclude` (`tools/mcp_tool_registration.py:210-224`; names `tools/mcp_tool_schema.py:159-176`) | not measured | stdio servers run on the host | | V (mechanism) / NV (Playwright) |

**Hermes has no LSP or check tool.** Diagnostics arrive only inside
`write_file` / `patch` results. LSP diagnostics (`lsp.enabled`, which
auto-installs servers) run **only on the local backend**
(`file_operations_lint.py:195-205`). Under our docker backend only the
shell linters run, inside the container.

**Sizes**

| set | tools | chars |
|---|---|---|
| shipped defaults on this host | 24 | 42,864 |
| **ours (Octopus: `file, skills, terminal, vision`)** | 12 | **18,771** (toolset_arms: 18,673) |
| ours + `tool_search: off` | 10 | 17,030 |
| ours + browser arm | 27 | 29,958 |
| loadout-1 (§0, the default 2026-09-26 to 09-28; now `--tools browser`): browser + Tool Search off | 25 | 28,210 |
| `--tools default` (the loadout without a browser) | 10 | 16,925 |
| **loadout-2, `lean` (§0, the default since 2026-09-28): file, terminal, vision + 3 Playwright MCP tools** | **10** | **14,855** |

All four rows re-measured 2026-09-28 with `toolset_arms.py lean --schemas`
(every stack connect refused; the builder tried 127.0.0.1:1234 and nothing
else).

## 3. OpenCode 1.18.32

OpenCode ships no JS files. `opencode.exe` (in
`...\nvm\v20.15.0\node_modules\opencode-ai\node_modules\opencode-windows-x64\bin\`,
180,133,928 bytes; `opencode-ai\bin\opencode.exe` is a hardlink to it) is a Bun-compiled binary with the minified bundle stored as plain
text. **Citations are byte offsets in that exe**, written `@N`; the
extraction scripts are in the session scratchpad `opencode-inv\`.

**Sizes.** Only the totals are measured, from the relay logs of the live
test (`docs/HARNESS-OPENCODE.md`):

- the default 9 tools are 20,473 bytes;
- with `question` they are 22,053 bytes, so `question` is 1,580.

Per-tool sizes are **estimates**: the description text from the binary
plus a rebuilt parameter schema. Summed over the 9 defaults, the
estimates come out 4-7% below the measured total. A stub capture of
OpenCode's exact tools array was not run.

### Selection mechanics (V)

- **Registry order** (`ToolRegistry` @101083334): `invalid`, `question`,
  `bash`, `read`, `glob`, `grep`, `edit`, `write`, `task`, `webfetch`,
  `todowrite`, `websearch`, `skill`, `apply_patch`.
  - `question` is offered only when `OPENCODE_CLIENT` is app, cli or
    desktop, or with `OPENCODE_ENABLE_QUESTION_TOOL`.
  - Three more are experimental: `execute` (`OPENCODE_EXPERIMENTAL_CODE_MODE`),
    `lsp` (`OPENCODE_EXPERIMENTAL_LSP_TOOL`) and `plan_exit`
    (`OPENCODE_EXPERIMENTAL_PLAN_MODE`).
  - Custom tools from `{tool,tools}/*.{js,ts}` in the config directories,
    and plugin tools, are appended after these.
  - **`OPENCODE_EXPERIMENTAL=1` turns on several of these at once**
    (@108393299).
- **Per-model filter** (@101085052):
  - `websearch` is kept only for the `opencode` provider, or with Exa or
    Parallel enabled.
  - `apply_patch` **replaces `edit` and `write`** when the model id
    contains `gpt-` (and not `oss` or `gpt-4`). Ours is `yamadori`, so it
    gets `edit` and `write`.
- **Permissions decide the list.** A tool is removed only when its last
  matching rule is `"*": "deny"` (@100892579, @103308213).
  - `edit`, `write` and `apply_patch` all share the `edit` permission.
  - A pattern-level deny (`bash: {"rm *": "deny"}`) keeps the tool in
    the list.
  - The legacy `tools: {name: false}` is converted to a deny
    (@108050095).
- **Defaults** (@101190507): the base is `"*": "allow"`, so `bash`, `edit`
  and `webfetch` run **without asking**.
- **`question` flips the tool list.** `opencode run` denies `question`,
  `plan_enter` and `plan_exit` for its session (@100357399), but `--fork`
  and the TUI do not. That is why the fork got 10 tools, which changes the
  cached system block.

### Tools

| tool | category | default (`run`) | gate | chars | risk / notes | source | V |
|---|---|---|---|---|---|---|---|
| `bash` | run commands | on | `permission.bash` | ~5.2k (Git Bash text) / ~5.8k (PowerShell text), est. | **host, no sandbox**, full `process.env`; timeout 120 s | @100994284, @101015015 | V |
| `read` | read files; **look at images** (PNG/JPEG/GIF/WebP/PDF come back as an attachment, moved into a synthetic user message `Attached media from tool result:` for `@ai-sdk/openai-compatible`) | on | `permission.read`; images need `modalities.input` to include `"image"` | ~1.6k est. | | @101036495; @102561920, @102567313 | V (+ live n=1) |
| `edit` | edit (fuzzy match) | on | `permission.edit` | ~1.9k est. | host | @101022464 | V |
| `write` | write | on | `permission.edit` | ~1.0k est. | host | @101061607 | V |
| `glob` | search code | on | `permission.glob` | ~1.1k est. | ripgrep from PATH, **else downloads rg 15.1.0 from GitHub** (@108411735); rg is on PATH here | @101031372 | V |
| `grep` | search code | on | `permission.grep` | ~1.2k est. | as `glob` | @101033354 | V |
| `task` | subagents | on | `permission.task` | ~3.8k est. | a child session on the same model | @101048291 | V |
| `todowrite` | planning | on | `permission.todowrite` | ~2.7k est. | no `todoread` in this version | @101056330 | V |
| `skill` | loads a SKILL.md | on | `permission.skill` | ~0.65k est. + a list in the system prompt | the only built-in skill is `customize-opencode` | @101063557 | V |
| `webfetch` | web fetch | **on** | `permission.webfetch` | ~1.3k est. | GET to **any** http(s) URL: **loopback is reachable** (:1234, :1235, :8888, :11434). Chrome User-Agent | @101058021 | V |
| `websearch` | web search | off | provider `opencode`, or `OPENCODE_ENABLE_EXA` / `OPENCODE_ENABLE_PARALLEL` / `OPENCODE_EXPERIMENTAL` | ~1.9k est. | queries go to `mcp.exa.ai` / `search.parallel.ai` | @101101210 | V |
| `apply_patch` | edit | off (gpt- model ids only) | model id | ~1.4k est. | | @101076983 | V |
| `question` | ask the user | off in `run`; on in the TUI and `--fork` | `permission.question` | **1,580 measured** | changes the tool list mid-conversation | @100991940 | V |
| `lsp` | code intelligence (definition, references, hover, symbols, calls) | off | `OPENCODE_EXPERIMENTAL_LSP_TOOL` + an `lsp` config | ~2.2k est. | server downloads (below) | @101066554 | V |
| `plan_exit`, `execute` | plan mode, code mode | off | experimental flags | ~0.75k / – | | @100989792, @101083334 | V |
| MCP: `<server>_<tool>` | anything, e.g. a browser | none | `mcp: {name: {type: "local", command, environment, enabled} \| {type: "remote", url, headers}}`; permission by name (`"playwright_*": "deny"`) | not measured | local servers run on the host | @108032675, @103211493 | V (mechanism) / NV (Playwright) |

**Checked and absent in 1.18.32:** `list`, `multiedit`, `todoread`,
`codesearch`, a `batch` tool, and any browser, console or screenshot tool.

### LSP diagnostics after edits (V in code, NV live)

After `write` and `edit`, OpenCode appends `LSP errors detected in this
file, please fix:` to the result. This is **off unless the `lsp` key is
set** (@101919303). This is the closest thing any of the four harnesses
has to type errors after each edit.

Setting it has side effects:

- `lsp: true`, or any object, enables **all** of about 38 servers, and
  many of them **auto-download**.
- `OPENCODE_DISABLE_LSP_DOWNLOAD` stops most downloads, but not
  `typescript-language-server`, which is npm-installed into OpenCode's
  cache when missing (@108256188).

The `formatter` key is off by default (@101095395). When on, it rewrites
files after each edit and does **not** tell the model, which is the #24
failure. Keep it off.

### Sizes

| set | tools | chars |
|---|---|---|
| **default `run` (and ours)** | 9 | **20,473 measured** |
| + `question` (fork or TUI) | 10 | 22,053 measured |
| recommended 6 (`bash read edit write glob grep`) | 6 | ~12k est. |
| **default loadout (§0): the 6 + `lsp` + 13 Playwright tools** | **20** | **24,638 measured** (a stand-in's capture) |

## 4. Pi 0.87.1

The package is `@earendil-works/pi-coding-agent`; PKG =
`...\nvm\v24.21.0\node_modules\@earendil-works\pi-coding-agent`. Sizes
were measured by importing the real definitions with Node 24 and
stringifying `{type: "function", function: {name, description,
parameters}}`, the wire shape: no `strict` for a custom endpoint,
`pi-ai/dist/api/openai-completions.js:1147-1156`.
`docs/HARNESS-PI.md` observed 2,956 bytes on the wire for the default
four; the gap is NV.

### Tool registry and selection (V)

- The registry is `dist/core/tools/index.js:19-28`: read, bash,
  powershell, edit, write, grep, find, ls. Defaults are
  `["read","bash","edit","write"]` (`dist/core/sdk.js:140`).
- **CLI:** `--tools/-t a,b` (an allowlist over built-in, extension and
  custom tools), `--exclude-tools`, `--no-builtin-tools`, `--no-tools`
  (`dist/cli/args.js:104-121`).
- **Settings:** `defaultTools` in `~/.pi/agent/settings.json` or
  `.pi/settings.json` (`settings-manager.js:952-955`).
- **Extensions** (`-e`, `~/.pi/agent/extensions/`, `.pi/extensions/`)
  register tools, and **each is active by default**
  (`agent-session.js:175-178`). They run in-process with full OS
  permissions.
- **No sandbox and no approvals** (`docs/security.md:97`).

### Tools

| tool | category | default | enable / needs | chars | risk | source | V |
|---|---|---|---|---|---|---|---|
| `read` | read files; **look at images** (returns jpg/png/gif/webp/bmp as image content; to an openai-completions provider a synthetic user message with `image_url` parts follows, when `input` has `"image"`) | on | models.json `input: ["text","image"]` for images | 684 | paths are not confined to cwd (`tools/path-utils.js:42-44`) | `tools/read.js:30-41,63-90`; `pi-ai/.../openai-completions.js:1062-1119` | V |
| `bash` | run commands | on | Git Bash on Windows, or `shellPath` | 543 | host; **no default timeout** | `tools/bash.js:26-29,145-157` | V |
| `edit` | edit (`edits[{oldText,newText}]`) | on | | 1,179 | host | `tools/edit.js:10-21,80-92` | V |
| `write` | write | on | | 430 | host | `tools/write.js:8-29` | V |
| `grep` | search code (rg `--json`, .gitignore, 100 matches) | off | `--tools ...,grep`; needs `rg` (on PATH here) | 1,039 | **downloads rg from GitHub** if missing, unless `--offline` / `PI_OFFLINE=1` | `tools/grep.js:11-34,52`; `utils/tools-manager.js:300-330` | V |
| `find` | search files (fd glob) | off | `--tools ...,find`; needs `fd` (**not** on PATH here) | 616 | downloads fd from GitHub if missing | `tools/find.js:17-39,119-125` | V |
| `ls` | list a directory | off | `--tools ...,ls` | 472 | | `tools/ls.js:8-27` | V |
| `powershell` | run commands | off | Windows only | 555 | host, `-ExecutionPolicy Bypass` | `tools/powershell.js:15-28` | V |
| `todo`, `subagent`, `question`, `questionnaire`, `structured_output` (example extensions, not loaded) | planning, subagents, UI | off | `-e examples/extensions/<x>` | 383 / 1,812 / 616 / 1,258 / 550 | `subagent` spawns more `pi` processes | `examples/extensions/todo.ts:136`, `subagent/index.ts:472` | V |
| browser / console / screenshot / web search / MCP | – | – | none shipped; an extension would have to add them | – | | | V (absence, searched in `dist/` and `examples/`) |

**Sizes:** default 4 = **2,841**; with `grep`, `find`, `ls` = 4,971; all
8 = 5,527. System prompt: 2,666-2,768 chars. **Default loadout (§0):**
6 tools, **4,689 chars** measured on the wire (a stand-in's capture); the
system prompt, with the two loadout skills listed, is 3,599 chars.

**Skills and AGENTS.md** (V):

- Agent Skills are listed by name and description and read with `read`,
  so they are offered only when `read` or `bash` is active
  (`core/system-prompt.js:99-104`).
- AGENTS.md is read from the cwd **and every parent directory**
  (`core/resource-loader.js:32-33`). A Pi run anywhere under
  `C:\Users\jwals\llama-stack` injects this repo's 79 kB AGENTS.md; use
  `-nc` or run outside it.

## 5. Codex CLI 0.157.1

Every row comes from requests captured by a local stub (never `:1234`).
Details and the config are in `docs/HARNESS-CODEX.md` §3. Codex is a
native binary, so "source" here is the captured request plus the
binary's own strings and `codex features list`.

| tool | category | default | enable / disable | chars | risk | V |
|---|---|---|---|---|---|---|
| `exec_command` | run commands (PTY). Params `cmd, workdir, shell, tty, login, yield_time_ms, max_output_tokens, sandbox_permissions, justification, prefix_rule` | on | the catalog's `shell_type: unified_exec` | 2,706 | `sandbox_permissions: "require_escalated"` = unsandboxed (rejected under `approval_policy = "never"`). On Windows, `workspace-write` needs `[windows] sandbox = "unelevated"`, else read-only | V (captured) |
| `write_stdin` | interact with a running command | on | with `exec_command` | 819 | as above | V |
| `apply_patch` | write / edit (freeform, Lark grammar, V4A patch) | **absent without a catalog entry** | `model_catalog_json` entry with `apply_patch_tool_type: "freeform"` (the only accepted value) | 896 | inside the sandbox | V |
| `view_image` | look at images (the image returns in the tool output) | on | `features.view_image` | 391 | reads any local path | V |
| `update_plan` | planning | off | `[tools.update_plan] enabled = true` | 781 | none | V |
| `request_user_input` | ask the user | on | `[tools.experimental_request_user_input] enabled = false` | 1,425 | blocks in exec | V |
| namespace `multi_agent_v1`: `spawn_agent`, `send_input`, `wait_agent`, `resume_agent`, `close_agent` | subagents | on | `features.multi_agent = false` | 9,491-10,157 | more contexts on the one GPU | V |
| `get_goal`, `create_goal`, `update_goal` | thread goals | on | `features.goals = false` | 3,278 | | V |
| `web_search` (hosted) | web search | on (`external_web_access: false`) | `web_search = "disabled"` | 52-95 | OpenAI-side; llama-server cannot run it | V |
| `image_generation` (hosted) | images | not offered to an API-key provider | none found (HARNESS-RESPONSES: needs ChatGPT auth, 0.133.0 `spec_plan.rs:298-306`) | – | | V (absence captured) |
| search code, browser, web fetch | – | – | MCP only (`mcp_servers`) | – | | V (absence) |

**Sizes:** fallback default 9 tools, 18,828 chars. **Our profile: 5
tools, 5,593 chars.** **Default loadout (§0): 21 tools, 15,476 chars**
(a stand-in's capture). It adds the 13 Playwright tools (9,031 chars as
the namespace), and 1,943 chars of MCP resource tools that come with any
MCP server. On top of the tools, every request carries 17,174
chars of `instructions` and about 7.2k chars of developer and environment
text.

**Outbound calls.** With `plugins`, `remote_plugin` and `apps` left on,
every run CONNECTs to chatgpt.com and github.com (captured). The profile
turns all three off.

## 6. What would help this model most, and is not on by default

This ranking is by the failure evidence, **not measured**. It was written
as arms to run paired (PROTOCOL). **Since 2026-09-26, items 1-5 and a form
of 6 are the default loadout (§0), by operator decision.** A paired run of
the loadout against its opt-out is still owed. It is a measurement of the
default now, not a gate on it.

1. **A browser with console errors, for web work.**
   - The evidence: #47. The final page error was never seen, and 37
     terminal Playwright scripts were the substitute.
   - Hermes has the only built-in one (`browser_console`,
     `browser_snapshot`, `browser_vision`, `browser_navigate`,
     `browser_press`). The arm is built (`toolset_arms.py browser`); it
     still needs the agent-browser install and a paired run.
   - OpenCode, Pi and Codex would need an MCP browser server (OpenCode's
     `mcp` key, Hermes' `mcp_servers`, Codex's `mcp_servers`; Pi only
     through an extension). No MCP server has been vetted or installed.
     Any one runs on the host, with the same loopback question as
     Hermes' `allow_private_urls`.
   - What works everywhere today: a shell-run headless script that
     prints console errors, plus a screenshot the model looks at with
     `read` or `view_image`.
2. **Bounded code search.**
   - Pi's `grep` / `find` are off by default. Without them a small model
     runs `grep -r` through `bash` into node_modules.
   - Codex has no search tool at all.
   - Hermes and OpenCode have it on.
3. **A direct background-process tool** for dev servers: Hermes'
   `process_manage` is hidden behind Tool Search by default. Setting
   `tools.tool_search.enabled: off` makes it direct and removes 3 bridge
   tools, saving 1,741 chars in our profile.
4. **Planning:**
   - Codex's `update_plan` is off by default;
   - Hermes' `todo_list` is deferred, and off in our profile;
   - OpenCode's `todowrite` is on;
   - Pi has none.

   All are cheap (0.4-1.3k chars). None is measured on this model.
5. **Codex `apply_patch`.** It is not a toggle: without a catalog entry
   Codex offers no edit tool at all, and the model edits through shell
   heredocs.
6. **Type errors after each edit.** Two harnesses can return these
   inside the edit result, and neither does so in our profiles today:
   - OpenCode, with its `lsp` key set (§3);
   - Hermes with `lsp.enabled`, but on the local backend only.

   Hermes on docker gets only single-file `tsc` / `node --check`.

## 7. Risky tools, across harnesses

| risk | tools |
|---|---|
| code on the **host**, outside any sandbox | Pi `bash`, `powershell` (always); OpenCode `bash` (always; default allow); Hermes `terminal` on the `local` backend, `execute_code` (local), **`browser_exec`**; Codex `exec_command` with `require_escalated`, and `--dangerously-bypass-approvals-and-sandbox` |
| reaches **host services** (:1234, :11434, dashboards) | Hermes browser with `allow_private_urls`; Hermes docker terminal (network on); Hermes' own startup probes of :1234; any MCP server; OpenCode `webfetch` (on by default, any URL including loopback) |
| **downloads** at run time | OpenCode `glob`/`grep` (rg, if missing), `lsp` (servers), `formatter` (prettier); Pi `grep`/`find` (rg/fd); Hermes browser without agent-browser installed (`npx`) |
| **credentials** | Hermes `browser_vault_*` (they ride with any browser backend) |
| sends data to **third parties** | Hermes `web_search` / `web_extract` (keyless providers), `image_generate`, `text_to_speech`; OpenCode `websearch` (Exa or Parallel); Codex hosted `web_search`, plugins/apps (chatgpt.com, github.com) |
| extra contexts on the **one GPU** | Hermes `delegate_task`; OpenCode `task`; Codex `spawn_agent`; Pi example `subagent` |
| **persistent state** that contaminates runs | Hermes `memory`, `session_search`, `skill_manage` |

## 8. Recommended profile per harness, for this model (coding work)

Each profile is a CHOICE, unmeasured. **Since 2026-09-26 this is the
default** (§0 has what was actually configured and verified, including
where it differs: Hermes' vault tools and `skill_manage`, which cannot be
removed; Codex's `network_access`; Pi's browser through the shell).

| harness | enable | leave off | config |
|---|---|---|---|
| Hermes | `read_file`, `write_file`, `patch`, `search_files`, `terminal` (docker), `process_manage` (direct), `vision_analyze`, `skills_list`/`skill_view`. For web work the **browser arm**: the 10 `browser_*` tools, with `browser_console` the reason | `browser_exec` (host Python), vault tools (unavoidable with the browser: 4.3k chars), `browser_cdp`/`browser_dialog`, `execute_code`, `delegate_task`, `web_search`/`web_extract` (third-party; if wanted, SearXNG on loopback), `memory`, `session_search`, `clarify`, `computer_use`, image/TTS/cron/messaging | `platform_toolsets.cli: [file, skills, terminal, vision]` (+ `browser` for the arm); `tools.tool_search.enabled: off`; browser arm keys as in `toolset_arms.py` (`browser.backend: "off"`, `allow_private_urls: true`, `restrict_evaluate: true`, `inactivity_timeout: 21600`, `docker_extra_args -p 127.0.0.1:3001:3001`). Consider `terminal.docker_network: false` for tasks that need no network (NV) |
| OpenCode | `bash`, `read` (images, with `modalities`), `edit`, `write`, `glob`, `grep`. That is ~12k chars instead of 20.5k, and the tool list stays the same across `run`, `--fork` and the TUI | `task` (GPU; ~3.8k), `todowrite` (~2.7k; the proxy's kickoff `plan` job plans), `skill` (only `customize-opencode`; our skills are injected by the proxy), `question`, `webfetch` (loopback), `websearch` (third party), `lsp`/plan/code mode (experimental), `formatter` (#24). Never set `OPENCODE_EXPERIMENTAL`; never give the model an id containing `gpt-` | `"permission": {"bash": "allow", "edit": "allow", "read": "allow", "glob": "allow", "grep": "allow", "task": "deny", "todowrite": "deny", "skill": "deny", "question": "deny", "webfetch": "deny", "websearch": "deny", "lsp": "deny"}`, `"formatter": false`, and the provider block of HARNESS-OPENCODE. Arm (NV): `lsp` with only `typescript` (`typescript-language-server` preinstalled, `OPENCODE_DISABLE_LSP_DOWNLOAD=1`), for type errors after each edit |
| Pi | `read`, `bash`, `edit`, `write`, `grep`, `find` | `ls` (bash does it), `powershell` (a second shell confuses a small model), example `subagent` (GPU), `question`/`questionnaire` (print mode), `todo` until measured | `pi --tools read,bash,edit,write,grep,find --offline`, or `{"defaultTools": [...]}` in settings; install `fd` first (a download: operator); `input: ["text","image"]` in models.json so `read` returns images; run outside `llama-stack` or with `-nc` |
| Codex | `exec_command`, `write_stdin`, `apply_patch`, `view_image`, `update_plan` | `multi_agent_v1` (GPU), goals, `request_user_input` (exec), hosted `web_search`, plugins/apps (outbound) | `SP\codex-home\config.toml` (HARNESS-CODEX §2) |

## 9. Keying skills off tool names

A skill that says "open the page and read the console" must name the
tool the harness actually has. The canonical capability names below map
to exact tool names. A request's `tools[].name` (or
`function.name`) decides which capability a skill may assume.

| capability | tool names that provide it |
|---|---|
| `read_file` | `read_file` (Hermes); `read` (OpenCode, Pi) |
| `edit_file` | `patch` (Hermes); `edit` (OpenCode, Pi); `apply_patch` (Codex, and OpenCode for GPT ids) |
| `write_file` | `write_file` (Hermes); `write` (OpenCode, Pi); `apply_patch` Add File (Codex) |
| `shell` | `terminal` (Hermes); `bash` (OpenCode, Pi); `powershell` (Pi); `exec_command` (Codex) |
| `background_process` | `process_manage` (Hermes); `terminal(background=true)`; `exec_command` + `write_stdin` (Codex) |
| `search_code` | `search_files` (Hermes); `grep`, `glob` (OpenCode); `grep`, `find` (Pi) |
| `browser` | `mcp__playwright__browser_navigate` + `mcp__playwright__browser_console_messages` (Hermes lean, §0); `browser_navigate` + `browser_console` + `browser_snapshot` (Hermes loadout-1); `playwright_browser_navigate` + `_snapshot` + `_console_messages` (OpenCode, §0); `browser_navigate` etc. in the namespace `mcp__playwright` (Codex, §0); none as a tool for Pi (the `agent-browser` CLI through `bash`, skill `page-check`) |
| `console_errors` | `mcp__playwright__browser_console_messages` (Hermes lean); `browser_console` (Hermes loadout-1; uncaught errors arrive through the sidecar's mirror, §0); `playwright_browser_console_messages` (OpenCode); `browser_console_messages` (Codex) |
| `screenshot_look` | `mcp__playwright__browser_take_screenshot` then `vision_analyze(<MEDIA path>)` (Hermes lean); `browser_vision` (Hermes loadout-1); `agent-browser screenshot` then `read` (Pi); `playwright_browser_take_screenshot` (OpenCode); `browser_take_screenshot` (Codex) |
| `type_check` | the `write` / `edit` results (OpenCode's lsp); a shell command (`tsc`, `pyright`: Hermes, Pi, Codex; skill `type-check`) |
| `package_api` (2026-09-29) | `lsp` `hover` / `goToDefinition` (OpenCode); a shell command, `node ~/.agents/skills/package-api/api.cjs <module> [name]` (Pi, skill `package-api`); none (Hermes, Codex) |
| `view_image` | `vision_analyze` (Hermes); `read` on an image file (OpenCode, Pi); `view_image` (Codex); `yama_describe_image` (ours, on main) |
| `plan` | `todo_list` (Hermes); `todowrite` (OpenCode); `update_plan` (Codex) |
| `subagent` | `delegate_task` (Hermes); `task` (OpenCode); `spawn_agent` (Codex) |
| `web_search` | `web_search` (Hermes, and Codex hosted); `websearch` (OpenCode, gated) |
| `web_fetch` | `web_extract` (Hermes); `webfetch` (OpenCode) |

When `browser` is absent, a web-app skill must say the honest fallback:
run a headless check through the shell, and print the page's console
errors and uncaught exceptions to stdout so they reach the model.
