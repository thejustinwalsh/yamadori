#!/usr/bin/env python
"""The harness kit's seed: what our research says to put into each harness,
each entry naming its evidence (a doc path, run or link, and what it showed).
mcp/harness_kit.py `seed` writes these into the store once; after that the
store is the operator's (edits and deletes win over a later seed revision).

Sources, all read 2026-09-29: docs/HARNESSES.md (s0 default loadout table,
the lean arm, s8, s9, the language-server MCP candidates), docs/HARNESS-PI.md,
docs/HARNESS-OPENCODE.md, docs/HARNESS-CODEX.md, docs/HARNESS-SANDBOX.md,
docs/HARNESS-RESPONSES.md, docs/TOOLS-API.md, docs/HERMES.md,
bench/sandbox/harness_skills/, bench/octopus/hermes_skills/,
bench/octopus/make_profile.py, bench/octopus/toolset_arms.py,
bench/sandbox/harness_box.py and its lockfile bench/sandbox/harness/
package-lock.json (every npm integrity below is copied from it), and
mcp/server.py's route list.

PROOF, per AGENTS.md "Claims carry their evidence": `operator` (a decision,
quoted with its date), `measured` (a run with its n), `verified` (read from
source, or a Docker check with no model), `unmeasured` (no run behind the
recommendation -- said so, never dressed up). No model run has used any
loadout below (docs/HARNESSES.md s0: "no run has used it yet"), so a
loadout's effect on the model is unmeasured even where its mechanics are
verified; each entry says which.

Bodies may carry the export's tokens ({{API_BASE}}, {{CONTEXT_WINDOW}}, ...:
harness_kit.TOKENS); never a key.
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SEED_VERSION = 1

LOCK = "bench/sandbox/harness/package-lock.json"
S0 = "docs/HARNESSES.md s0 (Default loadout)"


def ev(ref: str, showed: str) -> dict:
    return {"ref": ref, "showed": showed}


def npm_download(name: str, version: str, integrity: str, licence: str, *,
                 scripts: bool = False, verify_extra: str = "") -> dict:
    flag = "" if scripts else "--ignore-scripts "
    cmd = f"npm install -g {flag}{name}@{version}"
    return {"source_url": f"https://www.npmjs.com/package/{name}/v/{version}",
            "version": version, "hash": integrity, "licence": licence,
            "install": {"windows": cmd, "unix": cmd},
            "verify": (f"npm view {name}@{version} dist.integrity  # must print {integrity}"
                       + (f"; {verify_extra}" if verify_extra else ""))}


PLAYWRIGHT_TOOLS = ("browser_navigate", "browser_navigate_back", "browser_snapshot", "browser_click",
                    "browser_type", "browser_press_key", "browser_select_option", "browser_wait_for",
                    "browser_evaluate", "browser_console_messages", "browser_network_requests",
                    "browser_take_screenshot", "browser_handle_dialog")
OPENCODE_LSP_BUILTINS = (
    "deno", "typescript", "vue", "eslint", "oxlint", "biome", "gopls", "ruby-lsp", "ty",
    "pyright", "elixir-ls", "zls", "csharp", "razor", "fsharp", "sourcekit-lsp", "rust",
    "clangd", "svelte", "astro", "jdtls", "kotlin-ls", "yaml-ls", "lua-ls", "prisma", "dart",
    "ocaml-lsp", "bash", "terraform", "texlab", "dockerfile", "gleam", "clojure-lsp", "nixd",
    "tinymist", "haskell-language-server", "julials")
TS_EXTENSIONS = [".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".mts", ".cts"]


def _opencode_json() -> str:
    """docs/HARNESS-OPENCODE.md "The config used" + harness_box.opencode_loadout,
    with the Playwright MCP server launching the machine's own browser (no
    --cdp-endpoint: the remote machine has no sidecar)."""
    perm = {"bash": "allow", "edit": "allow", "read": "allow", "glob": "allow", "grep": "allow",
            "lsp": "allow", "task": "deny", "todowrite": "deny", "skill": "deny", "question": "deny",
            "webfetch": "deny", "websearch": "deny", "playwright_*": "deny"}
    for t in PLAYWRIGHT_TOOLS:
        perm[f"playwright_{t}"] = "allow"
    lsp = {sid: {"disabled": True} for sid in OPENCODE_LSP_BUILTINS}
    lsp["typescript-box"] = {"command": ["typescript-language-server", "--stdio"], "extensions": TS_EXTENSIONS}
    lsp["pyright-box"] = {"command": ["pyright-langserver", "--stdio"], "extensions": [".py", ".pyi"]}
    cfg = {
        "$schema": "https://opencode.ai/config.json",
        "autoupdate": False, "share": "disabled",
        "model": "yamadori/yamadori", "small_model": "yamadori/yamadori",
        "permission": perm, "formatter": False, "lsp": lsp,
        "mcp": {"playwright": {"type": "local", "enabled": True,
                               "command": ["playwright-mcp", "--no-webmcp", "--output-dir", ".playwright-mcp"]}},
        "provider": {"yamadori": {
            "npm": "@ai-sdk/openai-compatible", "name": "Yamadori",
            "options": {"baseURL": "{{API_BASE}}", "apiKey": "{env:YAMADORI_OPENCODE_KEY}", "setCacheKey": True},
            "models": {"yamadori": {
                "name": "Yamadori", "attachment": True, "reasoning": True, "tool_call": True,
                "modalities": {"input": ["text", "image"], "output": ["text"]},
                "limit": {"context": "__CW__", "output": "__MO__"},
                "variants": {"minimal": {"reasoningEffort": "minimal"}, "xhigh": {"reasoningEffort": "xhigh"},
                             "max": {"reasoningEffort": "max"}}}}}},
    }
    return (json.dumps(cfg, indent=2).replace('"__CW__"', "{{CONTEXT_WINDOW}}")
            .replace('"__MO__"', "{{MAX_OUTPUT}}"))


PI_MODELS = """{
  "providers": {
    "yamadori": {
      "baseUrl": "{{API_BASE}}",
      "api": "openai-completions",
      "apiKey": "$YAMADORI_PI_KEY",
      "models": [
        {
          "id": "yamadori",
          "name": "Yamadori",
          "reasoning": true,
          "thinkingLevelMap": { "off": "minimal", "xhigh": "xhigh", "max": "max" },
          "input": ["text", "image"],
          "contextWindow": {{CONTEXT_WINDOW}},
          "maxTokens": {{MAX_OUTPUT}},
          "compat": {
            "supportsDeveloperRole": false,
            "sendSessionAffinityHeaders": true
          }
        }
      ]
    }
  }
}"""

CODEX_TOML = """# Codex CLI 0.157.1 -> Yamadori over the public base (docs/HARNESS-CODEX.md s2,
# plus the harness box loadout: bench/sandbox/harness_box.py codex_config / codex_loadout).
# The key is read from YAMADORI_CODEX_KEY (env_key); it is never written here.

model = "yamadori"
model_provider = "yamadori"
# Codex's bundled catalog has no "yamadori": without this entry it offers NO apply_patch.
# Replace the path with where you put yamadori-catalog.json (inside CODEX_HOME, not under %TEMP%).
model_catalog_json = 'REPLACE-WITH-YOUR-CODEX_HOME/yamadori-catalog.json'
model_reasoning_effort = "medium"
sandbox_mode = "workspace-write"
approval_policy = "never"
web_search = "disabled"
check_for_update_on_startup = false

# Windows: without this, -s workspace-write ran read-only and apply_patch was rejected.
[windows]
sandbox = "unelevated"

[model_providers.yamadori]
name = "Yamadori"
base_url = "{{API_BASE}}"
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
shell_snapshot = false   # the key stays off disk (harness_box, checked in Docker)

[shell_environment_policy]
exclude = ["YAMADORI_CODEX_KEY", "YAMADORI_*"]   # not in the model's shell

[analytics]
enabled = false

[feedback]
enabled = false

# The browser reaches the dev server the model starts inside Codex's sandbox.
[sandbox_workspace_write]
network_access = true

# The browser: the official Playwright MCP server, launching this machine's own browser
# (the harness box attaches it to a sidecar over --cdp-endpoint instead).
[mcp_servers.playwright]
command = "playwright-mcp"
args = ["--no-webmcp", "--output-dir", ".playwright-mcp"]
startup_timeout_sec = 60
tool_timeout_sec = 120
default_tools_approval_mode = "approve"
enabled_tools = [""" + ", ".join(f'"{t}"' for t in PLAYWRIGHT_TOOLS) + "]"


def _codex_catalog() -> str:
    p = os.path.join(HERE, "harness_kit_data", "codex", "yamadori-catalog.json")
    d = json.load(open(p, encoding="utf-8"))
    for m in d.get("models", []):
        if m.get("slug") == "yamadori":
            m["context_window"] = "__CW__"
            m["max_context_window"] = "__CW__"
    return json.dumps(d, indent=2).replace('"__CW__"', "{{CONTEXT_WINDOW}}")


HERMES_PROVIDER = """# Merge into HERMES_HOME/config.yaml (docs/HARNESS-RESPONSES.md "Config"; a NAMED
# provider, as bench/octopus/make_profile.py writes its octo-relay one).
# Run with: hermes chat --provider yamadori -m yamadori
providers:
  yamadori:
    base_url: "{{API_BASE}}"
    key_env: YAMADORI_API_KEY
    # api_mode: codex_responses   # to speak /v1/responses instead of chat completions"""

HERMES_LEAN = """# The lean loadout (loadout-2): 10 tools. Merge into HERMES_HOME/config.yaml.
# Needs the playwright MCP server entry (the next config file) for the three browser tools.
platform_toolsets:
  cli: [file, terminal, vision, playwright]
# Tool Search off: process_manage offered directly, the three bridge tools gone.
tools: {tool_search: {enabled: "off"}}
# The external password managers off as vault sources.
vault: {onepassword: {enabled: false}, bitwarden: {enabled: false}}
# The Octopus profile also keeps Hermes' Blank Slate agent.disabled_toolsets
# (memory, session_search, delegation, cronjob, web, todo, code_execution, clarify,
# computer_use, image_gen, tts, messaging): keep yours if you ran that setup."""

HERMES_PLAYWRIGHT_REMOTE = """# Merge into HERMES_HOME/config.yaml. The official Playwright MCP server (0.0.82),
# only the three tools the lean arm uses, no generated resource/prompt utilities.
# On this machine it launches the machine's own browser; the Octopus box runs the
# same server by `docker run` attached to a sidecar Chrome instead.
mcp_servers:
  playwright:
    command: "playwright-mcp"
    args: ["--no-webmcp", "--output-dir", ".playwright-mcp"]
    tools:
      include: [browser_navigate, browser_console_messages, browser_take_screenshot]
      resources: false
      prompts: false"""

HERMES_PLAYWRIGHT_BOX = """# What bench/octopus/toolset_arms.py mcp_line writes per run (the box form):
mcp_servers: {"playwright": {"command": "docker", "args": ["run", "-i", "--rm", "--init",
  "--pull", "never", "--network", "container:<the run's sidecar>", "--user", "1000:1000",
  "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--pids-limit", "1024",
  "yamadori-harness-box:oc1.18.32-pi0.87.1-cx0.157.1-tools1", "playwright-mcp",
  "--cdp-endpoint", "http://127.0.0.1:9322", "--no-webmcp", "--output-dir",
  "/home/node/.cache/playwright-mcp"],
  "tools": {"include": ["browser_navigate", "browser_console_messages",
  "browser_take_screenshot"], "resources": false, "prompts": false}}}"""

HERMES_REVIEW_OFF = """# Merge into HERMES_HOME/config.yaml: no post-turn skill/memory review.
auxiliary:
  background_review:
    enabled: false"""

CLAUDE_MCP = """{
  "mcpServers": {
    "yamadori-tools": {
      "type": "http",
      "url": "{{TOOLS_MCP_URL}}",
      "headers": { "Authorization": "Bearer ${YAMADORI_API_KEY}" }
    }
  }
}"""

# Claude Code -> Yamadori over the Anthropic Messages API (mcp/messages_api.py,
# 2026-09-29). Every key names its source in claude-code.settings' evidence.
CLAUDE_SETTINGS = """{
  "model": "yamadori",
  "effortLevel": "medium",
  "modelSettings": { "yamadori": { "effort": "medium" } },
  "env": {
    "ANTHROPIC_BASE_URL": "{{PUBLIC_BASE}}",
    "ANTHROPIC_MODEL": "yamadori",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "yamadori",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "yamadori",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "yamadori",
    "ANTHROPIC_DEFAULT_FABLE_MODEL": "yamadori",
    "CLAUDE_CODE_SUBAGENT_MODEL": "yamadori",
    "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "{{CONTEXT_WINDOW}}",
    "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
    "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
    "CLAUDE_CODE_GATEWAY_HINT_HEADERS": "1",
    "CLAUDE_STREAM_FIRST_BYTE_TIMEOUT_MS": "1800000",
    "API_TIMEOUT_MS": "3600000"
  }
}"""

CLAUDE_ENV = """# Set these in the environment Claude Code runs in -- never in a file you commit.
# ANTHROPIC_AUTH_TOKEN: the key Claude Code sends to the proxy (Authorization: Bearer).
# Without it a saved claude.ai login stays the active credential and every request is refused 401.
ANTHROPIC_AUTH_TOKEN={{KEY_PLACEHOLDER}}
# YAMADORI_API_KEY: the key .mcp.json sends to the code-intelligence MCP server (it may be the same key).
YAMADORI_API_KEY={{KEY_PLACEHOLDER}}"""

CLAUDE_EFFORT_TABLE = """Claude Code talks to the proxy's POST /v1/messages (the Anthropic Messages API, a translation over the same turn every other harness gets: mcp/messages_api.py). Its effort picks the tier, and the tier picks the model (mcp/tier_models.yaml):

| Claude Code sends | tier | model |
|---|---|---|
| /effort low (output_config.effort low) | low | bonsai |
| /effort medium (the kit's default) | medium | bonsai |
| /effort high | high | bonsai |
| /effort xhigh | xhigh | mirai-s |
| /effort max | max | flash-next |
| thinking off (thinking.type disabled) and no effort | minimal | bonsai |
| no effort at all | the server's default (medium) | bonsai |

Claude Code's thinking budget (MAX_THINKING_TOKENS, thinking.budget_tokens) picks nothing: the model's profile owns the budget. Thinking comes back as thinking blocks signed by the proxy; an echoed block the proxy did not sign is dropped (the proxy puts back its own). The window Claude Code is told (CLAUDE_CODE_MAX_CONTEXT_TOKENS) is the default tier's; a prompt past a tier's window is refused "prompt is too long: N tokens > M maximum", which Claude Code compacts on."""

LOADOUT_TABLE = """The default loadout per harness (docs/HARNESSES.md s0; chars = len(json.dumps(tool)) as offered, not tokenized):

| | Hermes (lean) | OpenCode 1.18.32 | Pi 0.87.1 | Codex 0.157.1 |
|---|---|---|---|---|
| tools offered | 10 | 20 | 6 | 21 (8 + 13 mcp__playwright) |
| chars | 14,855 | 24,638 | 4,689 | 15,476 |
| browser | Playwright MCP 0.0.82 (3 tools) | Playwright MCP 0.0.82 (13) | agent-browser 0.26.0 CLI through bash | Playwright MCP 0.0.82 (13) |
| types | tsc + pyright through terminal | its LSP in write/edit results, and the lsp tool | tsc + pyright through bash (skill type-check) | tsc + pyright through exec_command (skill type-check) |
| a package's API | none (lsmcp candidate awaiting the operator) | lsp hover / goToDefinition | skill package-api | none |
| checked (Docker, no model) | toolset_arms.py verify 10/10 | loadout_check.py opencode 12/12 | loadout_check.py pi 19/19 | loadout_check.py codex 9/9 |

No model run has used any of these loadouts yet: their effect on the model is unmeasured."""

CAPABILITY_MAP = """Keying skills off tool names (docs/HARNESSES.md s9): a skill that says "open the page and read the console" must name the tool the harness actually has.

| capability | tool names that provide it |
|---|---|
| read_file | read_file (Hermes); read (OpenCode, Pi) |
| edit_file | patch (Hermes); edit (OpenCode, Pi); apply_patch (Codex) |
| shell | terminal (Hermes); bash (OpenCode, Pi); exec_command (Codex) |
| background_process | process_manage (Hermes); exec_command + write_stdin (Codex) |
| search_code | search_files (Hermes); grep, glob (OpenCode); grep, find (Pi) |
| browser | mcp__playwright__browser_navigate + _console_messages (Hermes lean); playwright_browser_* (OpenCode); browser_* in namespace mcp__playwright (Codex); agent-browser CLI through bash (Pi) |
| console_errors | mcp__playwright__browser_console_messages (Hermes lean); playwright_browser_console_messages (OpenCode); browser_console_messages (Codex) |
| screenshot_look | browser_take_screenshot then vision_analyze(<MEDIA path>) (Hermes lean); agent-browser screenshot then read (Pi); playwright_browser_take_screenshot (OpenCode); browser_take_screenshot (Codex) |
| type_check | write/edit results (OpenCode's lsp); tsc / pyright through the shell (Hermes, Pi, Codex) |
| package_api | lsp hover / goToDefinition (OpenCode); node ~/.agents/skills/package-api/api.cjs (Pi); none (Hermes, Codex) |
| view_image | vision_analyze (Hermes); read on an image file (OpenCode, Pi); view_image (Codex) |

When `browser` is absent, a web-app skill must say the honest fallback: a headless check through the shell that prints the page's console errors and uncaught exceptions."""

HERMES_SYSTEM_PROMPT = """You are a principal-level engineer working on realtime systems: Rust, C++,
Zig, WASM, TypeScript, and GPU work (WebGPU/WGSL, Skia, React Native).
Data-oriented design: the data layout is the problem, not the code.

TOOLS -- reach for them before answering from memory:
- Know the identifier? call find_definition_opt. Never guess at a definition.
- About to change a signature or delete code? call find_references first.
- Cannot name what you need? call find_by_meaning with a plain-language question.
- Retrieved snippets insufficient? refine the query and call again.

Never assert what code does without having read it. If you have not seen a
definition in this conversation, look it up.

Performance claims require evidence. Do not state that something is faster
without a measurement. Propose hypotheses; let the benchmark decide.

Platform constraints are data: state cache line size, SIMD width and frame
budget when they bear on a decision."""


def items() -> list[dict]:
    I: list[dict] = []

    def add(seed_id, **kw):
        kw.setdefault("where", "both")
        kw.setdefault("needs_operator", False)
        kw.setdefault("seed_rev", 1)
        I.append({"seed_id": seed_id, **kw})

    # ---------------------------------------------------------------- any
    add("any.where-runs-happen", harness="any", kind="note", proof="operator", status="recommended",
        title="Where the next waves run: the operator's own machine; the box for no-GPU tests",
        body=("Operator, 2026-09-29: \"We are going to run the next waves on my remote machine with a monitor "
              "and gpu attached, the way I would really do the work. We can still use the local harnesses for "
              "testing things that don't need access to a browser with a GPU to test the product.\"\n\n"
              "So the kit export is for that machine (entries marked remote or both); entries marked box "
              "describe the local harness box / Octopus sandbox, whose browser has no GPU."),
        evidence=[ev("operator, 2026-09-29 (this page's brief)", "the quote above")])
    add("any.no-harness-patches", harness="any", kind="note", proof="operator", status="recommended",
        title="Harnesses are changed by configuration only, never patched",
        body=("Harness tools that help the model are turned on by configuration, never a harness patch "
              "(operator, 2026-09-26). A harness's own usage skills live in the harness's skills folder "
              "(operator, 2026-09-26: \"We shouldn't carry harness specific skills, that is the job of the harness\")."),
        evidence=[ev("docs/HARNESSES.md (opening)", "operator direction 2026-09-26: tools turned on by configuration (never a harness patch)"),
                  ev("bench/octopus/make_profile.py", "harness skills installed into the harness's own skills folder, quoting the operator")])
    add("any.loadout-table", harness="any", kind="note", proof="verified", status="recommended",
        title="The default loadout per harness (tools, size, browser, types)",
        body=LOADOUT_TABLE,
        evidence=[ev(S0, "the table; each loadout checked in Docker with a scripted stand-in, no model"),
                  ev("bench/sandbox/results/loadout_check.jsonl", "the loadout check rows (opencode 12/12, pi 19/19, codex 9/9, 2026-09-29)"),
                  ev("operator, 2026-09-26 (docs/HARNESSES.md s0)", "\"All of our harnesses should have usable browser tools attached to them, and a generally useful normal tool loadout\"")])
    add("any.capability-map", harness="any", kind="note", proof="verified", status="recommended",
        title="Capability -> exact tool name, per harness (for writing skills)",
        body=CAPABILITY_MAP,
        evidence=[ev("docs/HARNESSES.md s9", "the capability table, from each harness's tool list as offered")])
    add("any.slow-webgl", harness="any", kind="note", proof="unmeasured", status="recommended", where="box",
        title="The sandbox browser renders WebGL in software: slow, not broken",
        body=("The harness box / Octopus sidecar Chromium renders WebGL in software (--use-angle=swiftshader). "
              "A heavy 3D scene can take seconds per frame, and open / reload / screenshot / eval can answer "
              "\"CDP command timed out\" while it renders. Read the page's errors and take one screenshot; with no "
              "errors, the page is working. The fact is in Pi's page-check skill and Hermes' look-at-a-screenshot "
              "1.1.0 (read only by loadout-1). Whether telling the model changes what it does: unmeasured.\n\n"
              "The remote machine has a GPU: this fact does not apply there."),
        evidence=[ev("bench/octopus/browser_sidecar.py", "Chromium launched with --use-angle=swiftshader (software WebGL)"),
                  ev("pagoda-p3 (Pi, low; docs/HARNESSES.md Pi, 'The slow-WebGL fact')",
                     "13 'CDP command timed out' results between calls 200 and 283; calls 221-288 spent debugging the browser; errors --json empty throughout (n=1 run)")])
    add("any.download.typescript", harness="any", kind="download", proof="verified", status="recommended",
        title="typescript 5.9.3 (tsc)",
        body="The type checker the type-check skill runs; package-api's script also loads it when the project has none.",
        download=npm_download("typescript", "5.9.3", "sha512-jl1vZzPDinLr9eUt3J/t7V6FgNEw9QjvBPdysz9KfQDD41fQrC2Y4vKQdiaUpFT4bXlb1RHhLpp8wtm6M5TgSw==", "Apache-2.0"),
        evidence=[ev(LOCK, "typescript 5.9.3 pinned by that sha512"),
                  ev("bench/sandbox/results/loadout_check.jsonl", "a TS2322 came back through tsc in the box (Pi, Codex, Hermes lean: terminal_type_check)")])
    add("any.download.pyright", harness="any", kind="download", proof="verified", status="recommended",
        title="pyright 1.1.414",
        body="Python type checker (the type-check skill; OpenCode's pyright-langserver).",
        download=npm_download("pyright", "1.1.414", "sha512-FPZZb51jepDX4eP7TEYDeNFtmE3WgwkkEcJpvH3/QmUSsj0EAy3LXu+xB4T/FWDejtsXlCfFr/rbWFjwuwuXww==", "MIT"),
        evidence=[ev(LOCK, "pyright 1.1.414 pinned by that sha512"),
                  ev(S0, "pyright's error came back in the box's loadout checks")])
    add("any.download.playwright-mcp", harness="any", kind="download", proof="verified", status="recommended",
        title="@playwright/mcp 0.0.82 (the browser for Hermes, OpenCode and Codex)",
        body=("The official Playwright MCP server (Microsoft). The configs in this kit run it as `playwright-mcp "
              "--no-webmcp` (a page cannot register tools into the model's list). In the box it attaches to the "
              "sidecar's Chrome over --cdp-endpoint and downloads no browser; on the remote machine it must find a "
              "browser of its own -- not run there yet."),
        download=npm_download("@playwright/mcp", "0.0.82", "sha512-OCqftfb8H4dnqm/njbTBRk3seUvUPttOlJUxCtEzXGETYOlRH5Qt3bbXIjmZIuWAxD9RF+yg1ASrPeXvm0y5cA==", "Apache-2.0",
                              verify_extra="then playwright-mcp --help must list --no-webmcp"),
        evidence=[ev(LOCK, "@playwright/mcp 0.0.82 pinned by that sha512"),
                  ev(S0, "Hermes lean 10/10, OpenCode 12/12, Codex 9/9: page opened, console error and uncaught exception read, host refused (Docker, no model)")])
    add("any.bridge.serena", harness="any", kind="mcp_server", proof="unmeasured", status="trying",
        title="oraios/serena 1.7.0 (language-server MCP bridge, second choice)",
        status_why=("The maintained option, not installed: a Python tree to lock with hashes, first-use npm installs "
                    "unless pre-seeded, 30+ tools including shell execution, edits and memories, a web dashboard. Second "
                    "choice after @mizchi/lsmcp."),
        body="Offers symbol overview, find symbol, references and GetDiagnosticsForFile.",
        download={"source_url": "https://github.com/oraios/serena", "version": "1.7.0", "licence": "GPL-3.0-or-later (app); MIT (SolidLSP)",
                  "pin_missing_why": "not fetched: nothing installed or locked (no downloads without the operator)"},
        evidence=[ev("docs/HARNESSES.md (The language-server MCP bridge)", "v1.7.0, 2026-08-09; the concerns listed; verdict 'second choice'")])
    add("any.bridge.cclsp", harness="any", kind="mcp_server", proof="verified", status="rejected",
        title="cclsp 0.7.0 (language-server MCP bridge)",
        status_why="Reported clean code when it was broken: a checker that says 'no errors' for a type error is worse than none.",
        download={"source_url": "https://www.npmjs.com/package/cclsp/v/0.7.0", "version": "0.7.0", "licence": "MIT",
                  "pin_missing_why": "rejected; removed from the image"},
        evidence=[ev("docs/HARNESSES.md (The language-server MCP bridge)",
                     "in the box get_diagnostics answered 'No diagnostics found' for a .ts file with TS2322 and a .py file with a pyright error, on every call; tsc and pyright found both")])
    add("any.bridge.isaacphi", harness="any", kind="mcp_server", proof="verified", status="rejected",
        title="isaacphi/mcp-language-server",
        status_why="Unmaintained: last commit 2025-06-03, no release binaries, an open issue about corrupted LSP frames. Not installed.",
        evidence=[ev("docs/HARNESSES.md (The language-server MCP bridge)", "the state read from its repository page")])
    add("any.prompt.code-search", harness="any", kind="prompt", proof="unmeasured", status="trying",
        title="System prompt for an agent with the code-intelligence MCP tools",
        status_why=("Only for a harness wired to the tools API's MCP server (find_definition_opt, find_references, "
                    "find_by_meaning). The routing gain behind 'the tool-use lines matter' is at risk: that eval sent "
                    "max_tokens 400 to a thinking model and did not record finish_reason (AGENTS.md; docs/CONSTRAINTS.md #31)."),
        body=HERMES_SYSTEM_PROMPT, file_name="code-search-system-prompt.md",
        target="the harness's system prompt / AGENTS.md",
        evidence=[ev("docs/HERMES.md s3", "the prompt, 'Put this in your agent, adapted'"),
                  ev("AGENTS.md 'Tool descriptions are prompts'", "10.7/14 -> 11.7/14 first-call routing, flagged 'These numbers are at risk'")])

    # ---------------------------------------------------------------- hermes
    add("hermes.provider", harness="hermes", kind="config", proof="verified", status="recommended",
        title="Hermes: a named provider at the proxy", body=HERMES_PROVIDER, file_name="hermes-provider.yaml",
        target="HERMES_HOME/config.yaml (merge)",
        evidence=[ev("docs/HARNESS-RESPONSES.md (Config)", "providers.<name>.base_url + key_env read by runtime_provider_custom.py:154; --provider <name> selects it"),
                  ev("bench/octopus/make_profile.py", "the Octopus profile's named provider (octo-relay) and `hermes chat --provider ... -m yamadori` (run.py hermes_cmd)")])
    add("hermes.lean", harness="hermes", kind="config", proof="operator", status="recommended",
        title="Hermes lean loadout (loadout-2): 10 tools", body=HERMES_LEAN, file_name="hermes-lean.yaml",
        target="HERMES_HOME/config.yaml (merge)",
        evidence=[ev("operator, 2026-09-28 (docs/HARNESSES.md s0)", "\"hermes loads too many tools, think it out to the tools we need for success\""),
                  ev("docs/HARNESSES.md s0 'Hermes, lean'", "read_file, write_file, patch, search_files, terminal, process_manage, vision_analyze + 3 Playwright MCP tools: 14,855 chars (loadout-1: 25 tools / 28,210)"),
                  ev("bench/octopus/toolset_arms.py verify (2026-09-29)", "10/10 in Docker with no model; its effect on a model run is unmeasured")])
    add("hermes.playwright.remote", harness="hermes", kind="mcp_server", proof="unmeasured", status="recommended", where="remote",
        title="Hermes: Playwright MCP launching this machine's browser (remote form)",
        body=HERMES_PLAYWRIGHT_REMOTE, file_name="hermes-playwright-mcp.yaml", target="HERMES_HOME/config.yaml (merge)",
        evidence=[ev("bench/octopus/toolset_arms.py mcp_server_config", "the tool filter (tools.include) and resources/prompts off, as verified in the box"),
                  ev("docs/HARNESSES.md s0 'Hermes, lean'", "the box form runs the same server by docker against a sidecar; this remote form (own browser, no CDP) has not run")])
    add("hermes.playwright.box", harness="hermes", kind="mcp_server", proof="verified", status="recommended", where="box",
        title="Hermes: Playwright MCP in the Octopus sidecar (box form)",
        body=HERMES_PLAYWRIGHT_BOX, file_name="hermes-playwright-mcp-box.yaml", target="written per run by bench/octopus/run.py",
        evidence=[ev("bench/octopus/toolset_arms.py mcp_argv / mcp_line", "the command"),
                  ev("docs/HARNESSES.md s0 'Verified 2026-09-29'", "toolset_arms.py verify 10/10: console error and ReferenceError read, MEDIA path readable, host refused, containers removed")])
    add("hermes.review-off", harness="hermes", kind="config", proof="operator", status="recommended",
        title="Hermes: post-turn background review off", body=HERMES_REVIEW_OFF, file_name="hermes-review-off.yaml",
        target="HERMES_HOME/config.yaml (merge)",
        evidence=[ev("bench/octopus/make_profile.py", "background_review asked the model to 'update the skill library' after v0f's prompt 1: 601 s of GPU and a skill_manage that writes into the profile; off for benchmark runs (operator, 2026-09-26: fix every identified issue before the next run)")])
    add("hermes.lean-skills", harness="hermes", kind="note", proof="unmeasured", status="trying", needs_operator=True,
        title="DECISION: the lean arm has no skills channel -- give it one?",
        status_why=("Awaiting the operator. The lean arm cuts the skills toolset, and Hermes adds the skills index to the "
                    "system prompt only with it, so no harness skill (the slow-WebGL fact, type-check) reaches it."),
        body=("Config-only ways, none taken: (a) put `skills` back in platform_toolsets.cli: +4,932 chars and "
              "skill_manage, which writes persistent state; (b) a project context file (Hermes reads AGENTS.md / "
              "CLAUDE.md / .cursorrules from the working directory) -- that is the task's workspace, so it changes the "
              "task, not the harness."),
        evidence=[ev("docs/HARNESSES.md s0 'No skills reach the lean arm'", "agent/system_prompt.py:303 _skills_prompt and :316 _auto_load_parts gate on the skills toolset")])
    add("hermes.lsmcp", harness="hermes", kind="mcp_server", proof="unmeasured", status="trying", needs_operator=True,
        title="DECISION: @mizchi/lsmcp 0.10.0 for Hermes (a package's API without reading node_modules)",
        status_why=("A download for the operator to approve. Hermes' own LSP serves the local backend only and is "
                    "diagnostics-only; an MCP bridge run like Playwright MCP would give hover / definition / library symbols."),
        body=("Vetting before use: lock its dependency tree by integrity; read dist/ for network and exec; the cclsp test "
              "(a TS2322 must come back); hover and definition on koota's createWorld in the box. tools.include must leave "
              "out its editing tools (replace_range, replace_regex, rename, delete symbol) and write_memory. Needs Node >= 22."),
        download={"source_url": "https://www.npmjs.com/package/@mizchi/lsmcp/v/0.10.0", "version": "0.10.0", "licence": "MIT",
                  "pin_missing_why": "not fetched: its tree is unlocked and unread (no downloads without the operator)"},
        evidence=[ev("docs/HARNESSES.md (The language-server MCP bridge, 2026-09-29)", "'the candidate to vet first': lsp_get_hover, lsp_get_definitions, search_external_library_symbols map one-to-one to the question"),
                  ev("docs/HARNESSES.md Pi 'Would it have cut the wandering reads?'", "about 60% of what runs read node_modules for is the package's API (n=292 identifier pairs, 5 runs, offline)")])
    add("hermes.skill.look-at-a-screenshot", harness="hermes", kind="skill", proof="unmeasured", status="recommended", where="box",
        title="Hermes skill: look-at-a-screenshot 1.1.0 (loadout-1 only)",
        skill_path="bench/octopus/hermes_skills/look-at-a-screenshot",
        body="Read only by loadout-1 (--tools browser): the lean arm has no skills channel. Its paths are the Octopus profile's.",
        target="HERMES_HOME/skills/software-development/look-at-a-screenshot/",
        evidence=[ev("bench/octopus/make_profile.py install_skills", "installed into the Octopus profile; read only by arms with the skills toolset"),
                  ev("bench/octopus/hermes_skills/look-at-a-screenshot/SKILL.md", "the shared screenshot folder and the slow-WebGL paragraph")])
    add("hermes.skill.type-check", harness="hermes", kind="skill", proof="unmeasured", status="recommended", where="box",
        title="Hermes skill: type-check 1.0.0 (/opt/yamadori-tools, loadout-1 only)",
        skill_path="bench/octopus/hermes_skills/type-check",
        body="Points at /opt/yamadori-tools (the Octopus terminal container's mount); the remote machine has no such path.",
        target="HERMES_HOME/skills/software-development/type-check/",
        evidence=[ev("docs/HARNESSES.md s0 loadout-1 'Types'", "the tools volume yamadori-typecheck-tools1: typescript 5.9.3 + pyright 1.1.414, read-only")])
    add("hermes.console-text", harness="hermes", kind="note", proof="verified", status="recommended",
        title="Hermes: read page errors from Playwright MCP, not its own browser_console",
        body=("Hermes ee5ee84's browser_console reads errors[].message; agent-browser 0.26.0 (the only release in Hermes' "
              "pin) names it text, so uncaught errors arrive as empty strings. Playwright MCP's "
              "browser_console_messages carries them with their text (the lean arm uses it)."),
        evidence=[ev("docs/HARNESSES.md s0 loadout-1 'FOUND'", "js_errors: [{\"message\": \"\"}]; the sidecar's error mirror as a workaround"),
                  ev("docs/HARNESSES.md s0 'Verified 2026-09-29'", "browser_console_messages returned 'ReferenceError: undefinedFn is not defined'")])
    add("hermes.startup-probes", harness="hermes", kind="note", proof="verified", status="recommended",
        title="Hermes probes the endpoint while it builds its tool list",
        body=("check_vision_requirements -> detect_local_server_type walks /api/v1/models, /api/tags, /v1/props, /props "
               "and /version against the configured endpoint (8 connects per build). Expect them in the proxy's log."),
        evidence=[ev("docs/HARNESSES.md s2 'The :1234 probe'", "agent/model_metadata.py:761; refused by the offline guard in the measurement")])
    add("hermes.no-package-api", harness="hermes", kind="note", proof="verified", status="recommended",
        title="Hermes has no tool for a package's API",
        body=("Its LSP runs only on the local backend (_lsp_local_only) and only feeds diagnostics into write_file / patch "
              "results; there is no hover or definition tool for the model. See the lsmcp decision."),
        evidence=[ev("docs/HARNESSES.md (The language-server MCP bridge, Hermes)", "tools/file_operations_lint.py _lsp_local_only; website/docs/user-guide/features/lsp.md")])

    # ---------------------------------------------------------------- pi
    add("pi.models", harness="pi", kind="config", proof="verified", status="recommended",
        title="Pi: models.json (the provider, thinking map, window from /v1/models)", body=PI_MODELS,
        file_name="models.json", target="<PI_CODING_AGENT_DIR>/models.json",
        evidence=[ev("docs/HARNESS-PI.md 'The config used'", "every key justified by a finding there (n=1 each)"),
                  ev("docs/HARNESSES.md s0 Pi 'The model entry'", "contextWindow / maxTokens / input from the proxy's advertised /v1/models row per run (pagoda.pi_models)"),
                  ev("docs/HARNESS-RESPONSES.md", "`pi --provider yamadori --model yamadori`, captured")])
    add("pi.settings", harness="pi", kind="config", proof="verified", status="recommended",
        title="Pi: settings.json with the six default tools",
        body='{\n  "defaultTools": ["read", "bash", "edit", "write", "grep", "find"]\n}',
        file_name="settings.json", target="<PI_CODING_AGENT_DIR>/settings.json",
        evidence=[ev("bench/sandbox/harness_box.py pi_settings", "settings-manager defaultTools"),
                  ev(S0, "loadout_check.py pi 19/19, 2026-09-29 (Docker, no model)")])
    add("pi.env", harness="pi", kind="config", proof="verified", status="recommended",
        title="Pi: environment (PI_CACHE_RETENTION=long and friends)",
        body="PI_CACHE_RETENTION=long\nPI_SKIP_VERSION_CHECK=1\nPI_TELEMETRY=0",
        file_name="pi.env", target="the environment Pi runs in",
        evidence=[ev("docs/HARNESS-PI.md 'The config used' and s2", "PI_CACHE_RETENTION=long makes Pi send prompt_cache_key = its session id (captured); process-wide: it also asks other providers for long retention"),
                  ev("bench/sandbox/harness_box.py HARNESSES['pi']", "the same variables in the box")])
    add("pi.headless", harness="pi", kind="note", proof="verified", status="recommended",
        title="Pi headless: close stdin, MSYS_NO_PATHCONV=1, no slash commands in print mode",
        body=("`pi --mode json ... \"prompt\" < /dev/null`: with a non-TTY stdin that never reaches EOF Pi waits (a run hung "
              "10 minutes). Under Git Bash set MSYS_NO_PATHCONV=1 (MSYS rewrote /compact). Slash commands are prompts in "
              "print/JSON mode: compact through RPC mode ({\"type\": \"compact\"}). --mode json exits 0 on failures."),
        evidence=[ev("docs/HARNESS-PI.md 'The config used' (headless runs)", "experiments e2, #27, c3, k1, e3 (n=1 each)")])
    add("pi.skill.package-api", harness="pi", kind="skill", proof="unmeasured", status="recommended",
        title="Pi skill: package-api (an installed package's API from its own declarations)",
        skill_path="bench/sandbox/harness_skills/package-api",
        body=("On the remote machine the script finds TypeScript in the project (else it exits 2: its other lookups are "
              "the box's /opt paths). No model has used the skill."),
        target="~/.agents/skills/package-api/",
        evidence=[ev("docs/HARNESSES.md s0 Pi 'package-api'", "in the box: createWorld's overloads, World.query, koota/react; a miss exits 1 (Docker, no model)"),
                  ev("docs/HARNESSES.md Pi 'Would it have cut the wandering reads?'", "offline: 94 + 81 of 292 identifiers the runs grepped node_modules for are in the script's answers; no model run yet")])
    add("pi.skill.type-check", harness="pi", kind="skill", proof="unmeasured", status="recommended",
        title="Skill: type-check (tsc / pyright through the shell)",
        skill_path="bench/sandbox/harness_skills/type-check",
        body="It says tsc 5.9.3 and pyright 1.1.414 are installed: install both (the any-harness downloads) on the remote machine.",
        target="~/.agents/skills/type-check/",
        evidence=[ev("bench/sandbox/harness_box.py loadout_skills", "Pi and Codex get type-check"),
                  ev(S0, "TS2322 and pyright's error came back through the shell (Docker, no model); the skill's effect is unmeasured")])
    add("pi.skill.page-check", harness="pi", kind="skill", proof="unmeasured", status="recommended", where="box",
        title="Pi skill: page-check (agent-browser against the box's sidecar Chrome)",
        skill_path="bench/sandbox/harness_skills/page-check",
        body="Box only: it drives the sidecar over --cdp 9322 and carries the slow-WebGL paragraph.",
        target="~/.agents/skills/page-check/",
        evidence=[ev("docs/HARNESSES.md s0 Pi", "title, BOX_CONSOLE_ERROR, undefinedFn in errors --json, the gate's refusals (Docker, no model)"),
                  ev("pagoda-p3 (docs/HARNESSES.md)", "why the slow-WebGL paragraph exists")])
    add("pi.remote-browser", harness="pi", kind="download", proof="unmeasured", status="trying", where="remote",
        title="Pi on the remote machine: agent-browser 0.26.0 launching its own Chrome",
        status_why=("Pi has no MCP client; in the box its browser is agent-browser attached to the sidecar (page-check). "
                    "On a machine with its own browser the skill needs a variant without --cdp 9322 and without the "
                    "slow-WebGL paragraph; not written or run yet."),
        download=npm_download("agent-browser", "0.26.0", "sha512-pdqSfjwbFSp+qnwlb2g23e9wXveIOfMi19xpPA9xZUbzEAUp6W4YBZj6Ybj8z4M7WkcbGDDYc+oDIHDt9R3EDQ==", "Apache-2.0"),
        evidence=[ev("docs/HARNESSES.md s0 Pi", "Pi has no MCP client (0.87.1); agent-browser 0.26.0 through bash is the box's browser"),
                  ev(LOCK, "agent-browser 0.26.0 pinned by that sha512")])
    add("pi.fd-rg", harness="pi", kind="note", proof="verified", status="trying", where="remote",
        title="Pi on the remote machine: fd and rg for its find / grep tools",
        status_why=("With PI_OFFLINE=1 Pi never downloads rg or fd; without it, it fetches them unpinned. The box pins fd "
                    "10.5.0's linux-musl asset only; no Windows asset hash is recorded."),
        body="Install ripgrep and fd from a pinned source before running Pi offline, or accept Pi's own download.",
        evidence=[ev("docs/HARNESS-SANDBOX.md 'The image'", "ripgrep linked from Codex's vendored rg 15.2.0; Pi runs with PI_OFFLINE=1"),
                  ev("bench/sandbox/harness/Dockerfile", "fd v10.5.0 x86_64-unknown-linux-musl, sha256 761c72dc8e120d85b22292063be8a796e2eeb20eb3e4f38b8fa2343ccf3514a7")])
    add("pi.download", harness="pi", kind="download", proof="verified", status="recommended",
        title="Pi 0.87.1 (@earendil-works/pi-coding-agent)",
        body="The version every Pi finding here was made on. Ran on Node 24.21.0.",
        download=npm_download("@earendil-works/pi-coding-agent", "0.87.1", "sha512-m8ArJUtVcQMSe1lLE/Ei7vX/JV7O39sWmWBsXV2NOU70F0qCp8GubA24pT3LnwTmM6LL2xV80/h6sQg85n69ew==", "MIT"),
        evidence=[ev(LOCK, "0.87.1 pinned by that sha512"),
                  ev("docs/HARNESS-PI.md", "hands-on test against the live stack, 2026-09-26 (n=1 per experiment)")])

    # ---------------------------------------------------------------- opencode
    add("opencode.config", harness="opencode", kind="config", proof="verified", status="recommended",
        title="OpenCode: opencode.json (provider + the default loadout)", body=_opencode_json(),
        file_name="opencode.json", target="the file OPENCODE_CONFIG points at",
        evidence=[ev("docs/HARNESS-OPENCODE.md 'The config used'", "the provider block, each key justified by a finding"),
                  ev("bench/sandbox/harness_box.py opencode_loadout", "permissions (deny = not offered), formatter off (#24), every built-in LSP disabled, typescript-box + pyright-box"),
                  ev(S0, "loadout_check.py opencode 12/12 in the box (no model); the mcp command here launches the machine's browser instead of --cdp-endpoint: not run on the remote machine")])
    add("opencode.env", harness="opencode", kind="config", proof="verified", status="recommended",
        title="OpenCode: environment",
        body=("OPENCODE_DISABLE_AUTOUPDATE=1\nOPENCODE_DISABLE_SHARE=1\nOPENCODE_DISABLE_CLAUDE_CODE=1\n"
              "OPENCODE_DISABLE_LSP_DOWNLOAD=1\nOPENCODE_EXPERIMENTAL_LSP_TOOL=1"),
        file_name="opencode.env", target="the environment OpenCode runs in (plus OPENCODE_CONFIG = the path of opencode.json)",
        evidence=[ev("docs/HARNESS-OPENCODE.md 'The config used'", "the host test's variables"),
                  ev("bench/sandbox/harness_box.py HARNESSES['opencode'] loadout_env", "no LSP downloads; the lsp tool; never OPENCODE_EXPERIMENTAL")])
    add("opencode.lsp-package-api", harness="opencode", kind="note", proof="verified", status="recommended",
        title="OpenCode's lsp tool answers a package's API",
        body=("lsp hover at koota's createWorld returned '(alias) function createWorld(options: WorldOptions): World (+1 "
              "overload)'; goToDefinition returned node_modules/koota/dist/index.d.ts line 3. No skill teaches the lsp "
              "tool yet; its own description is the only prompt."),
        evidence=[ev("docs/HARNESSES.md s0 OpenCode (2026-09-29)", "loadout_check.py opencode 12/12, koota 0.6.6 copied from a pagoda run (no download)")])
    add("opencode.ids", harness="opencode", kind="note", proof="verified", status="recommended",
        title="OpenCode: never a model id containing gpt-, never OPENCODE_EXPERIMENTAL",
        body=("For GPT-family ids OpenCode replaces edit/write with apply_patch; OPENCODE_EXPERIMENTAL turns on every "
              "experimental feature (only OPENCODE_EXPERIMENTAL_LSP_TOOL is set). Under Windows PowerShell 5.1 keep double "
              "quotes out of an `opencode run` prompt (native argument passing split it)."),
        evidence=[ev("docs/HARNESSES.md s8 OpenCode", "'Never set OPENCODE_EXPERIMENTAL; never give the model an id containing gpt-'"),
                  ev("docs/HARNESS-OPENCODE.md 'The config used'", "experiment 1's prompt reached the model mangled")])
    add("opencode.download", harness="opencode", kind="download", proof="verified", status="recommended",
        title="OpenCode 1.18.32 (opencode-ai)",
        body="Its own install script places the platform binary (the box installs with --ignore-scripts, then `npm rebuild opencode-ai`).",
        download=npm_download("opencode-ai", "1.18.32", "sha512-SCrZWdq44y/EoH2+fE4HLcXS+DzpVqHPzmXk3p2RrufYy8LWvpfhRhKtijb5ktvx8r94ScqToqwHxR0BwS65OQ==", "MIT", scripts=True),
        evidence=[ev(LOCK, "1.18.32 pinned by that sha512"),
                  ev("docs/HARNESS-SANDBOX.md 'The image'", "npm ci --ignore-scripts, then npm rebuild opencode-ai: only its own postinstall")])
    add("opencode.download.tsls", harness="opencode", kind="download", proof="verified", status="recommended",
        title="typescript-language-server 6.0.1 (OpenCode's typescript-box)",
        body="The language server opencode.json's typescript-box runs; it uses the project's typescript when it has one.",
        download=npm_download("typescript-language-server", "6.0.1", "sha512-c5hEHM/7rdFRbQWoHXuddyJ53oF0M2dKeXQOYdwUek556oarltKeumKJGLp/ZMcGV7gW30mGn1MR3V5GDiLiKg==", "Apache-2.0"),
        evidence=[ev(LOCK, "6.0.1 pinned by that sha512"),
                  ev("docs/HARNESSES.md s0 OpenCode 'Verified'", "a write of a .ts file with an error returned 'LSP errors detected ... Type 'string' is not assignable to type 'number''")])

    # ---------------------------------------------------------------- codex
    add("codex.config", harness="codex", kind="config", proof="verified", status="recommended",
        title="Codex: config.toml (provider, sandbox, key kept out of the shell, the browser)", body=CODEX_TOML,
        file_name="config.toml", target="<CODEX_HOME>/config.toml",
        evidence=[ev("docs/HARNESS-CODEX.md s2", "every key and why; validated offline with --strict-config against a stub"),
                  ev("docs/HARNESS-SANDBOX.md 'The key'", "shell_snapshot = false + shell_environment_policy.exclude: env | grep -c = 0, no file held the key (scripted stand-in)"),
                  ev(S0, "loadout_check.py codex 9/9 (box form, --cdp-endpoint); the remote form launches the machine's browser: not run")])
    add("codex.catalog", harness="codex", kind="config", proof="verified", status="recommended",
        title="Codex: yamadori-catalog.json (without it: no apply_patch)", body=_codex_catalog(),
        file_name="yamadori-catalog.json", target="<CODEX_HOME>/yamadori-catalog.json (the path model_catalog_json names)",
        evidence=[ev("docs/HARNESS-CODEX.md s2 model_catalog_json", "without the entry Codex warns 'Model metadata for yamadori not found', offers no apply_patch and sends no effort (captured)"),
                  ev("mcp/harness_kit_data/codex/yamadori-catalog.json", "the entry the host test used (gpt-5.5's with our values); the export writes the proxy's current window into it")])
    add("codex.skill.type-check", harness="codex", kind="skill", proof="unmeasured", status="recommended",
        title="Skill: type-check (tsc / pyright through exec_command)",
        skill_path="bench/sandbox/harness_skills/type-check", target="<CODEX_HOME>/skills/type-check/",
        evidence=[ev("bench/sandbox/harness_box.py loadout_skills / SKILL_HOMES", "codex gets type-check in $CODEX_HOME/skills"),
                  ev(S0, "tsc + pyright through exec_command in the box (no model)")])
    add("codex.package-api", harness="codex", kind="skill", proof="unmeasured", status="trying", needs_operator=True,
        title="DECISION: package-api for Codex",
        status_why=("Awaiting the operator. Codex has no tool for a package's API; its exec_command could run the same "
                    "script if the skill were added to loadout_skills('codex'). Not done."),
        skill_path="bench/sandbox/harness_skills/package-api", target="<CODEX_HOME>/skills/package-api/",
        evidence=[ev("docs/HARNESSES.md (The language-server MCP bridge, Codex)", "'its exec_command could run the same script from $CODEX_HOME/skills if the skill were added'")])
    add("codex.windows-sandbox", harness="codex", kind="note", proof="verified", status="recommended",
        title="Codex on Windows: [windows] sandbox = \"unelevated\"; CODEX_HOME outside %TEMP%",
        body=("-s workspace-write alone ran READ-ONLY on Windows (apply_patch 'writing is blocked by read-only sandbox'); "
              "with unelevated the patch applied (first run 51.8 s including setup, no admin). Under %TEMP% every run warns "
              "it could not create PATH aliases. Under approval_policy never, MCP tools need default_tools_approval_mode = "
              "\"approve\" or they are refused."),
        evidence=[ev("docs/HARNESS-CODEX.md s1-s2", "the turn metadata before/after; the warning"),
                  ev("docs/HARNESSES.md s0 Codex", "'MCP tool call requires approval, but approval policy is never' seen in the first check")])
    add("codex.download", harness="codex", kind="download", proof="verified", status="recommended",
        title="Codex CLI 0.157.1 (@openai/codex)",
        body="No install scripts in either package. Windows binary codex.exe is Authenticode-signed by OpenAI OpCo, LLC.",
        download=npm_download("@openai/codex", "0.157.1", "sha512-qJ/UZ0bmYP+/Umav1L9WpmtMYeA6q1+4r4qILSYOKQZhP7WRdjyTQWz3O0dTImZ9RT7AazZa85D87xRDHogcHw==", "Apache-2.0",
                              verify_extra="Windows: Get-FileHash of ...\\@openai\\codex-win32-x64\\vendor\\x86_64-pc-windows-msvc\\bin\\codex.exe must be 8CB0E69E99FF2A158C54815DB82D0F2E524D8F301BC30184722CFD1AE5973574"),
        evidence=[ev("docs/HARNESS-CODEX.md s1", "signatures and SLSA provenance verified (npm audit signatures), integrity, codex.exe SHA-256"),
                  ev(LOCK, "0.157.1 pinned by that sha512")])

    # ---------------------------------------------------------------- claude code
    # Rev 2 (2026-09-29, operator: "Yes add anthropic api endpoints for claude"): the proxy serves the Anthropic
    # Messages API, so Claude Code uses Yamadori as its model. No real Claude Code run yet: every entry is unmeasured.
    add("claude-code.no-model-route", harness="claude-code", kind="note", proof="unmeasured", status="recommended",
        seed_rev=2,
        title="Claude Code uses Yamadori as its model (/v1/messages); effort picks the tier and the model",
        body=CLAUDE_EFFORT_TABLE,
        evidence=[ev("mcp/messages_api.py (module doc)", "the request/response/stream/error mapping and the effort -> tier table, with Claude Code's own docs cited for each input"),
                  ev("mcp/test_messages_api.py", "offline, through the served template: a Claude-Code-shaped tool loop extends the slot, signed thinking echoes, streamed == blocking, errors, count_tokens, auth by both headers, effort -> tier; no real Claude Code run"),
                  ev("code.claude.com/docs/en/llm-gateway-protocol (read 2026-09-29)", "an ANTHROPIC_BASE_URL gateway serves /v1/messages (+ optional count_tokens); an unrecognised model id gets adaptive thinking, output_config.effort and context management")])
    add("claude-code.settings", harness="claude-code", kind="config", proof="unmeasured", status="recommended",
        title="Claude Code: settings.json (Yamadori as the model, effort medium, the proxy as ANTHROPIC_BASE_URL)",
        body=CLAUDE_SETTINGS, file_name="settings.json",
        target="~/.claude/settings.json (your user) or <project>/.claude/settings.json (merge into an existing file)",
        evidence=[ev("code.claude.com/docs/en/llm-gateway-connect / llm-gateway-protocol (read 2026-09-29)", "ANTHROPIC_BASE_URL selects the Anthropic Messages format; ANTHROPIC_AUTH_TOKEN goes out as Authorization: Bearer; CLAUDE_CODE_GATEWAY_HINT_HEADERS=1 sends x-claude-code-request-class (the proxy serves an auxiliary side request as a utility call); CLAUDE_CODE_ATTRIBUTION_HEADER=0 keeps the attribution block out of the prompt when a gateway reshapes system content (the proxy joins system blocks into one message); CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1 stops context_management and beta tool fields, which the proxy does not serve"),
                  ev("code.claude.com/docs/en/model-config (read 2026-09-29)", "ANTHROPIC_MODEL / ANTHROPIC_DEFAULT_*_MODEL / CLAUDE_CODE_SUBAGENT_MODEL pin every role (background tasks, subagents) to one id; an id not starting with claude- is a custom spelling, so CLAUDE_CODE_MAX_CONTEXT_TOKENS sets the window Claude Code compacts at (the proxy's /v1/models window); effortLevel and modelSettings.<id>.effort set the default effort"),
                  ev("code.claude.com/docs/en/errors (read 2026-09-29)", "API_TIMEOUT_MS is the per-request timeout (default 10 min) and CLAUDE_STREAM_FIRST_BYTE_TIMEOUT_MS the first-byte deadline, clamped to 30 min: 1,800,000 is that clamp, and 3,600,000 covers the longest deep-thinking runs on record (mcp/tiers.py JOB_THINKING note: three struggle runs took 756-2,658 s); the proxy commits its first byte only when the turn has started (E1)")])
    add("claude-code.env", harness="claude-code", kind="config", proof="unmeasured", status="recommended",
        title="Claude Code: the key variables (placeholders only)", body=CLAUDE_ENV, file_name="claude-code.env",
        target="the environment Claude Code runs in (never a committed file)",
        evidence=[ev("code.claude.com/docs/en/llm-gateway (read 2026-09-29)", "with only ANTHROPIC_BASE_URL set a saved claude.ai login stays the active credential; a gateway credential variable replaces it"),
                  ev("mcp/server.py _identify_anthropic", "the proxy accepts an account key as Authorization: Bearer or x-api-key")])
    add("claude-code.tools-mcp", harness="claude-code", kind="mcp_server", proof="unmeasured", status="recommended",
        seed_rev=2,
        title="Claude Code: the code-intelligence MCP server over HTTPS",
        body=CLAUDE_MCP, file_name="mcp.json", target=".mcp.json in the project root",
        evidence=[ev("docs/TOOLS-API.md 'Configuring a client'", "the .mcp.json form, ${VAR} expansion from code.claude.com/docs/en/mcp; the key from YAMADORI_API_KEY"),
                  ev("caddy/Caddyfile", "https://<site>/tools/* reaches the tools API :1235 with the prefix stripped")])
    return I
