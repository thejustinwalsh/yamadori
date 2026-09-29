#!/usr/bin/env python
"""The harness box: OpenCode, Pi and Codex run INSIDE a container whose only
way to the Windows host is ONE fixed forward to the proxy or a recording
relay (docs/HARNESS-SANDBOX.md, SELF-IMPROVEMENT-LOG #49). Configuration
only: the official npm packages at the host tests' exact versions, no patch
to any harness.

    python bench/sandbox/harness_box.py plan opencode [--target-port 18235]
    python bench/sandbox/harness_box.py build            # refuses without the base image locally
    python bench/sandbox/harness_box.py verify [--target-port 1234]   # Docker only, no model
    python bench/sandbox/harness_box.py run opencode --key-file K --project DIR \\
        --run-dir DIR [--target-port 18235] [--config SRC] -- run --format json -- "prompt"
    python bench/sandbox/loadout_check.py all --key-file DUMMY    # the default loadout, no model

THE DEFAULT LOADOUT (operator, 2026-09-26; docs/HARNESSES.md s0): every run
gets a browser sidecar (the grader's Playwright image, browser_sidecar.py)
whose network namespace the harness joins, and each harness's recommended
tools -- the Playwright MCP server (OpenCode, Codex) or agent-browser
through the shell (Pi), language-server diagnostics (OpenCode's lsp; tsc
and pyright elsewhere) -- written into its config. `--loadout as-tested`
writes the host test's config alone.

WHY. The host-run harness tests (C:\\Users\\jwals\\octo\\opencode-test,
pi-test, the Codex home in the session scratchpad) gave the model a shell ON
THE WINDOWS HOST: every loopback service (llama-swap :11434 with no auth,
llama-server, Caddy's admin API :2019, the tools API :1235, SearXNG), the
filesystem and the key files beside the configs. Codex's own Windows sandbox
narrows writes, not reads or loopback.

WHAT A RUN IS (all per run, all removed after):
  - a sandbox network (sandbox_net.py, prefix `harn`): `--internal`, no
    route anywhere; the gate on it as `egress` -- HTTP(S)_PROXY to global
    addresses on 80/443, and ONE forward, egress:<port> ->
    host.docker.internal:<port>, the proxy (:1234) or the relay the test
    records through. Nothing else on the host is reachable: not by address
    (no route), not through the proxy (the gate refuses non-global
    addresses), not through the forward (its target is fixed here).
  - the harness container, on that network only: the pinned image
    (harness/Dockerfile), uid 1000, every capability dropped,
    no-new-privileges, a pids/memory/cpu limit. Mounted: the project at
    /work and a per-run home at /home/node (<run-dir>/home), nothing else.
    The harness's base URL is http://egress:<port>/v1, and `egress` is in
    NO_PROXY so that one request goes to the forward.
  - the harness config, WRITTEN by this runner into the run home from the
    host test's config: base URL rewritten, the key replaced by the
    harness's own env reference ({env:VAR} / $VAR / env_key). A literal key
    in the source is dropped, never copied; the home is scanned for the key
    before the container starts.
  - the key: read from --key-file into the environment of the `docker run`
    child only, passed as `-e VAR` (name only: the value never appears in an
    argv, a file this runner writes, or its record). It is in the
    container's environment, so the MODEL CAN READ IT (`env` in its shell):
    use a key made for these tests.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sandbox_net  # noqa: E402

PREFIX = "harn"
HARNESS_DIR = os.path.join(HERE, "harness")
VERSIONS = {"opencode": "1.18.32", "pi": "0.87.1", "codex": "0.157.1"}
LOADOUT_VERSION = "tools1"             # Dockerfile label yamadori.harness.loadout
IMAGE = "yamadori-harness-box:oc{opencode}-pi{pi}-cx{codex}-".format(**VERSIONS) + LOADOUT_VERSION
# Node 24 LTS, the multi-arch index digest (Docker Hub, 2026-09-26): the
# Dockerfile's ARG BASE default. Not in the local store on 2026-09-26.
BASE_IMAGE = ("node:24.21.0-bookworm@sha256:"
              "64af3819f9275802414d7cdc38c27e9d82bd564dec4d4da87d008255d36c63b4")
BASE_PULL_MB = 410                      # amd64 layers, compressed (manifest sizes)
TARGET_HOST = "host.docker.internal"    # the one non-global target: the host's proxy/relay port
HOME = "/home/node"
WORK = "/work"
# Host ports a forward must never point at: the services #48 found open.
FORBIDDEN_PORTS = {11434, 10001, 10002, 10003, 10004, 10005, 10006, 10007, 10008,
                   2019, 1235, 1237, 8888, 445, 139, 135, 2222, 80, 443}

# ------------------------------------------------------------------ the default loadout
# Operator, 2026-09-26: "All of our harnesses should have usable browser tools
# attached to them, and a generally useful normal tool loadout" and "LSP should
# probably be in the default loadout for everyone". docs/HARNESSES.md "Default
# loadout" is the spec; `--loadout as-tested` gives the host test's config
# unchanged (no browser, no sidecar), the explicit opt-out.
LOADOUTS = ("recommended", "as-tested")
# The loadout's packages in the image (harness/package.json, pinned by the
# lockfile, registry signatures checked at build by `npm audit signatures`).
LOADOUT_PACKAGES = {"@playwright/mcp": "0.0.82", "agent-browser": "0.26.0",
                    "pyright": "1.1.414", "typescript": "5.9.3",
                    "typescript-language-server": "6.0.1"}
DEFAULT_LOADOUT = "recommended"
# THE BROWSER: the grader's own Playwright image (Chromium headless shell;
# no new download) runs bench/octopus/browser_sidecar.py as a SIDECAR that
# owns the run's network namespace; the harness container joins it, so
# `localhost` is one place for the model's shell, its dev server and the
# browser. Chrome's DevTools port is on that namespace's loopback only.
SIDECAR_IMAGE = "octo-playwright:1.63.0"
SIDECAR_DIR = os.path.join(os.path.dirname(HERE), "octopus")      # browser_sidecar.py
CDP_URL = "http://127.0.0.1:9322"                                  # browser_sidecar.CHROME_PORT
# The official Playwright MCP server (Microsoft, npm @playwright/mcp, pinned in
# harness/package-lock.json), ATTACHED to the sidecar's Chrome over CDP: it
# downloads and launches no browser. --no-webmcp: a page must not be able to
# register tools into the model's list.
PLAYWRIGHT_MCP = ["playwright-mcp", "--cdp-endpoint", CDP_URL, "--no-webmcp",
                  "--output-dir", f"{HOME}/.cache/playwright-mcp"]
# The tools of that server a loadout offers (0.0.82 offers 25 by default,
# 20.6k chars as OpenCode sends them): what checking a page needs, about
# 10.8k chars. Left out: browser_run_code_unsafe (arbitrary Node code in the
# server; the shell is there for that), drag/drop, emulate_media,
# file_upload, find, fill_form, hover, network_request (one request's
# detail), tabs, resize and close (the sidecar's one browser is shared).
PLAYWRIGHT_TOOLS = (
    "browser_navigate", "browser_navigate_back", "browser_snapshot", "browser_click",
    "browser_type", "browser_press_key", "browser_select_option", "browser_wait_for",
    "browser_evaluate", "browser_console_messages", "browser_network_requests",
    "browser_take_screenshot", "browser_handle_dialog")
# OpenCode 1.18.32's built-in language servers (the `U1` table in the binary,
# @101880000-101918000): with an `lsp` object ALL of them are enabled unless
# disabled one by one (LSP.state, @101919300), and many download themselves.
OPENCODE_LSP_BUILTINS = (
    "deno", "typescript", "vue", "eslint", "oxlint", "biome", "gopls", "ruby-lsp", "ty",
    "pyright", "elixir-ls", "zls", "csharp", "razor", "fsharp", "sourcekit-lsp", "rust",
    "clangd", "svelte", "astro", "jdtls", "kotlin-ls", "yaml-ls", "lua-ls", "prisma", "dart",
    "ocaml-lsp", "bash", "terraform", "texlab", "dockerfile", "gleam", "clojure-lsp", "nixd",
    "tinymist", "haskell-language-server", "julials")
TS_EXTENSIONS = [".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".mts", ".cts"]
# The loadout skill Pi and Codex read from their own skills folders
# (harness-specific skills live in the harness, not in our server store).
SKILLS_DIR = os.path.join(HERE, "harness_skills")

SCRATCH = (r"C:\Users\jwals\AppData\Local\Temp\claude\C--Users-jwals-llama-stack"
           r"\d16e1f09-4699-483b-a257-1ba77b329ce7\scratchpad")
HARNESSES = {
    "opencode": {
        "key_env": "YAMADORI_OPENCODE_KEY", "bin": "opencode",
        "config_src": r"C:\Users\jwals\octo\opencode-test\cfg\full.json",
        # docs/HARNESS-OPENCODE.md "The config used": the same variables
        "env": {"XDG_CONFIG_HOME": f"{HOME}/.config", "XDG_DATA_HOME": f"{HOME}/.local/share",
                "XDG_CACHE_HOME": f"{HOME}/.cache", "XDG_STATE_HOME": f"{HOME}/.local/state",
                "OPENCODE_CONFIG": f"{HOME}/.config/opencode/opencode.json",
                "OPENCODE_DISABLE_AUTOUPDATE": "1", "OPENCODE_DISABLE_SHARE": "1",
                "OPENCODE_DISABLE_CLAUDE_CODE": "1"},
        # the recommended loadout adds: no language-server downloads (the two
        # it runs are in the image), and the `lsp` tool (definition,
        # references, hover, symbols, calls; diagnostics come in edit/write
        # results). Only this flag, never OPENCODE_EXPERIMENTAL (@108393299).
        "loadout_env": {"OPENCODE_DISABLE_LSP_DOWNLOAD": "1",
                        "OPENCODE_EXPERIMENTAL_LSP_TOOL": "1"},
    },
    "pi": {
        "key_env": "YAMADORI_PI_KEY", "bin": "pi",
        "config_src": r"C:\Users\jwals\octo\pi-test\cfg\full.json",
        # docs/HARNESS-PI.md "The config used". PI_CACHE_RETENTION=long is its
        # "Also set this in Pi's environment": Pi then sends prompt_cache_key =
        # its session id (s2), which the proxy honours as the conversation's
        # name, keeps across a Pi compaction and keeps apart from a fork;
        # without it a session is named by our tool-call-id carrier alone.
        "env": {"PI_CODING_AGENT_DIR": f"{HOME}/.pi/agent",
                "PI_CODING_AGENT_SESSION_DIR": f"{HOME}/.pi/sessions",
                "PI_OFFLINE": "1", "PI_SKIP_VERSION_CHECK": "1", "PI_TELEMETRY": "0",
                "PI_CACHE_RETENTION": "long"},
    },
    "codex": {
        "key_env": "YAMADORI_CODEX_KEY", "bin": "codex",
        "config_src": os.path.join(SCRATCH, "codex-home"),
        # docs/HARNESS-CODEX.md: CODEX_HOME holds config.toml + the catalog
        "env": {"CODEX_HOME": f"{HOME}/.codex"},
    },
}


# ------------------------------------------------------------------ configs

def base_url(port: int) -> str:
    return f"http://{sandbox_net.GATE_ALIAS}:{port}/v1"


SECRET_FIELD = re.compile(r"^(api[_-]?key|key|token|access[_-]?token|secret|authorization|bearer)$",
                          re.I)


def opencode_config(src: dict, url: str, key_env: str) -> dict:
    """The host test's opencode.json with every provider pointed at the
    forward and its key replaced by `{env:VAR}` (OpenCode's own
    substitution). Anything key-shaped in a provider's options goes."""
    cfg = copy.deepcopy(src)
    for prov in (cfg.get("provider") or {}).values():
        opts = prov.setdefault("options", {})
        for k in [k for k in opts if SECRET_FIELD.match(k)]:     # not setCacheKey
            del opts[k]
        opts["baseURL"] = url
        opts["apiKey"] = "{env:%s}" % key_env
        for k in [k for k in (opts.get("headers") or {})
                  if re.search(r"auth|key|token", k, re.I)]:
            del opts["headers"][k]
    cfg["autoupdate"] = False
    cfg["share"] = "disabled"
    return cfg


def opencode_loadout(cfg: dict) -> dict:
    """The recommended loadout (docs/HARNESSES.md "Default loadout"):
    bash read edit write glob grep + lsp, and the Playwright MCP server's
    browser tools (`playwright_*`). A permission whose value is a plain
    "deny" removes the tool from the list (@100892579); a pattern deny would
    keep it. The formatter stays off (#24: it rewrites files and does not say
    so). Language servers: every built-in disabled one by one, then two of
    our own, which run from the image (typescript-language-server uses the
    project's typescript when it has one, else the image's 5.9.3; pyright
    1.1.414). Own ids, so the root is the project directory: the built-in
    `typescript` starts only under a lockfile."""
    cfg = copy.deepcopy(cfg)
    perm = {k: v for k, v in (cfg.get("permission") or {}).items()}
    perm.update({"bash": "allow", "edit": "allow", "read": "allow", "glob": "allow",
                 "grep": "allow", "lsp": "allow",
                 "task": "deny", "todowrite": "deny", "skill": "deny", "question": "deny",
                 "webfetch": "deny", "websearch": "deny"})
    # every playwright_* tool denied, then the chosen ones allowed: the LAST
    # matching rule decides (@100892579), and JSON keeps this order
    perm["playwright_*"] = "deny"
    for t in PLAYWRIGHT_TOOLS:
        perm[f"playwright_{t}"] = "allow"
    cfg["permission"] = perm
    cfg["formatter"] = False
    lsp = {sid: {"disabled": True} for sid in OPENCODE_LSP_BUILTINS}
    lsp["typescript-box"] = {"command": ["typescript-language-server", "--stdio"],
                             "extensions": TS_EXTENSIONS}
    lsp["pyright-box"] = {"command": ["pyright-langserver", "--stdio"],
                          "extensions": [".py", ".pyi"]}
    cfg["lsp"] = lsp
    cfg["mcp"] = {"playwright": {"type": "local", "command": list(PLAYWRIGHT_MCP),
                                 "enabled": True}}
    return cfg


PI_TOOLS = ["read", "bash", "edit", "write", "grep", "find"]


def pi_settings() -> dict:
    """Pi's settings.json (PI_CODING_AGENT_DIR): the recommended six tools
    (settings-manager `defaultTools`). Pi has no MCP client (0.87.1: none in
    dist/ or docs/; only third-party extensions add one), so its browser is
    agent-browser through `bash` and its type check `tsc`/`pyright` through
    `bash`, both on PATH in the image, taught by the loadout skill."""
    return {"defaultTools": list(PI_TOOLS)}


def codex_loadout(text: str) -> str:
    """Codex's recommended loadout on top of the host config (which already
    has update_plan on, the catalog entry that gives apply_patch, and
    multi_agent/goals/plugins/apps/memories off): the Playwright MCP server,
    and network access inside Codex's own sandbox -- without it a dev server
    the model starts lives in bwrap's empty network namespace and the
    browser cannot reach it. What that opens is the box's network: the
    sidecar's localhost, the gate (public 80/443) and the one forward (whose
    key is excluded from the shell's environment)."""
    lines = text.rstrip("\n").split("\n")
    lines = [ln for ln in lines if not re.match(r"\s*network_access\s*=", ln)]
    out = "\n".join(lines) + "\n"
    if "[sandbox_workspace_write]" in out:
        out = out.replace("[sandbox_workspace_write]", "[sandbox_workspace_write]\n"
                          "network_access = true   # harness_box loadout: the browser reaches the dev server", 1)
    else:
        out += ("\n[sandbox_workspace_write]\n"
                "network_access = true   # harness_box loadout: the browser reaches the dev server\n")
    args = ", ".join(json.dumps(a) for a in PLAYWRIGHT_MCP[1:])
    out += ("\n# harness_box loadout: the browser (docs/HARNESSES.md \"Default loadout\")\n"
            "[mcp_servers.playwright]\n"
            f"command = {json.dumps(PLAYWRIGHT_MCP[0])}\n"
            f"args = [{args}]\n"
            "startup_timeout_sec = 60\n"
            "tool_timeout_sec = 120\n"
            # without it every tool not marked read-only is refused under
            # approval_policy = "never" ("MCP tool call requires approval")
            'default_tools_approval_mode = "approve"\n'
            f"enabled_tools = [{', '.join(json.dumps(t) for t in PLAYWRIGHT_TOOLS)}]\n")
    return out


def pi_config(src: dict, url: str, key_env: str) -> dict:
    """Pi's models.json: each provider at the forward, the key as `$VAR`
    (Pi's resolve-config-value; never a `!command`)."""
    cfg = copy.deepcopy(src)
    for prov in (cfg.get("providers") or {}).values():
        prov["baseUrl"] = url
        prov["apiKey"] = "$" + key_env
        for k in [k for k in (prov.get("headers") or {})
                  if re.search(r"auth|key|token", k, re.I)]:
            del prov["headers"][k]
    return cfg


def codex_config(text: str, url: str, key_env: str, catalog: str | None,
                 sandbox_off: bool = False) -> str:
    """Codex's config.toml, line-edited (comments kept): every provider's
    base_url at the forward, env_key = VAR, the catalog at its container
    path; any inline token line dropped. `sandbox_off` writes
    danger-full-access (the container is the boundary) for the case where
    Codex's Linux sandbox cannot start inside it (docs/HARNESS-SANDBOX.md).

    KEEPING THE KEY FROM THE MODEL'S SHELL (checked 2026-09-26 with a scripted
    stand-in for the model, no model): on Linux Codex writes a shell snapshot
    of its environment, key included, to CODEX_HOME/shell_snapshots/ (on the
    host's disk through the run-home mount) and sources it into every
    command, and its default env policy let YAMADORI_CODEX_KEY through
    (`env | grep -c` = 1). `features.shell_snapshot = false` alone stopped
    the file but not the variable; `shell_environment_policy.exclude` alone
    did not either (the snapshot re-exports it); both together: 0, and no
    file. Codex itself still reads the key from its own environment."""
    out = []
    section = ""
    seen_snapshot = False
    for line in text.splitlines():
        s = line.strip()
        if re.match(r"\[[^\]]+\]$", s):
            if section == "features" and not seen_snapshot:
                out.append("shell_snapshot = false   # harness_box: the key stays off disk")
                seen_snapshot = True
            section = s[1:-1].strip()
        if re.match(r"(experimental_bearer_token|bearer_token|api_key|http_headers)\s*=", s):
            continue
        if section == "features" and re.match(r"shell_snapshot\s*=", s):
            line = "shell_snapshot = false   # harness_box: the key stays off disk"
            seen_snapshot = True
        if section == "shell_environment_policy" and re.match(r"exclude\s*=", s):
            continue                             # replaced below
        if re.match(r"base_url\s*=", s):
            line = f'base_url = "{url}"'
        elif re.match(r"env_key\s*=", s):
            line = f'env_key = "{key_env}"'
        elif re.match(r"model_catalog_json\s*=", s):
            if not catalog:
                continue
            line = f"model_catalog_json = '{catalog}'"
        elif sandbox_off and re.match(r"sandbox_mode\s*=", s):
            line = 'sandbox_mode = "danger-full-access"   # harness_box --codex-sandbox off'
        out.append(line)
    if section == "features" and not seen_snapshot:
        out.append("shell_snapshot = false   # harness_box: the key stays off disk")
        seen_snapshot = True
    if not seen_snapshot:
        out += ["", "[features]", "shell_snapshot = false   # harness_box: the key stays off disk"]
    exclude = f'exclude = ["{key_env}", "YAMADORI_*"]   # harness_box: not in the model\'s shell'
    for i, ln in enumerate(out):
        if ln.strip() == "[shell_environment_policy]":
            out.insert(i + 1, exclude)
            break
    else:
        out += ["", "[shell_environment_policy]", exclude]
    return "\n".join(out) + "\n"


def loadout_skills(harness: str) -> list[str]:
    """The loadout skills a harness gets in its own skills folder: Pi and
    Codex check types through the shell; Pi also drives the browser through
    it (it has no MCP client) and, having no language server, looks up an
    installed package's API with the `package-api` skill's script (the
    TypeScript compiler in the image reading the package's declarations:
    docs/HARNESSES.md "Pi"). OpenCode has all three built in (lsp, MCP)."""
    return {"pi": ["type-check", "page-check", "package-api"],
            "codex": ["type-check"]}.get(harness, [])


SKILL_HOMES = {"pi": ".agents/skills",           # Pi reads the Agent Skills location (docs/skills.md)
               "codex": ".codex/skills"}          # $CODEX_HOME/skills


def write_home(harness: str, home: str, port: int, src: str | None = None,
               codex_sandbox_off: bool = False, loadout: str = DEFAULT_LOADOUT) -> list[str]:
    """Write the harness config into the run home; returns the files written.
    `loadout` "recommended" (the default) adds the default loadout
    (docs/HARNESSES.md); "as-tested" writes the host test's config only."""
    if loadout not in LOADOUTS:
        raise SystemExit(f"unknown loadout {loadout!r} (known: {', '.join(LOADOUTS)})")
    rec = loadout == "recommended"
    h = HARNESSES[harness]
    src = src or h["config_src"]
    url, key_env = base_url(port), h["key_env"]
    written = []

    def put(rel: str, data: str) -> None:
        p = os.path.join(home, *rel.split("/"))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            f.write(data)
        written.append(p)

    if harness == "opencode":
        with open(src, encoding="utf-8") as f:
            cfg = opencode_config(json.load(f), url, key_env)
        put(".config/opencode/opencode.json",
            json.dumps(opencode_loadout(cfg) if rec else cfg, indent=2))
    elif harness == "pi":
        with open(src, encoding="utf-8") as f:
            put(".pi/agent/models.json", json.dumps(pi_config(json.load(f), url, key_env), indent=2))
        if rec:
            put(".pi/agent/settings.json", json.dumps(pi_settings(), indent=2))
    else:
        d = src if os.path.isdir(src) else os.path.dirname(src)
        cat_src = os.path.join(d, "yamadori-catalog.json")
        catalog = None
        if os.path.exists(cat_src):
            with open(cat_src, encoding="utf-8") as f:
                put(".codex/yamadori-catalog.json", f.read())
            catalog = f"{HOME}/.codex/yamadori-catalog.json"
        with open(os.path.join(d, "config.toml"), encoding="utf-8") as f:
            text = codex_config(f.read(), url, key_env, catalog, codex_sandbox_off)
        put(".codex/config.toml", codex_loadout(text) if rec else text)
    if rec:
        for name in loadout_skills(harness):
            # the whole skill folder: SKILL.md and any script it runs
            for fn in sorted(os.listdir(os.path.join(SKILLS_DIR, name))):
                with open(os.path.join(SKILLS_DIR, name, fn), encoding="utf-8") as f:
                    put(f"{SKILL_HOMES[harness]}/{name}/{fn}", f.read())
    for sub in (".cache", ".local/share", ".local/state"):
        os.makedirs(os.path.join(home, *sub.split("/")), exist_ok=True)
    return written


def secret_in(root: str, secret: str) -> list[str]:
    """Files under `root` that contain `secret` (the key must be in none)."""
    if not secret:
        return []
    hits = []
    needle = secret.encode()
    for dp, _dn, fn in os.walk(root):
        for n in fn:
            p = os.path.join(dp, n)
            try:
                with open(p, "rb") as f:
                    if needle in f.read():
                        hits.append(p)
            except OSError:
                pass
    return hits


# ------------------------------------------------------------------ docker argv

def check_port(port: int) -> None:
    if not 1 <= port <= 65535 or port in FORBIDDEN_PORTS:
        raise SystemExit(f"--target-port {port} refused: a forward must point at the proxy "
                         f"(:1234) or a recording relay, never at {sorted(FORBIDDEN_PORTS)}")


def forwards(port: int) -> list[tuple[int, str, int, None]]:
    """The gate's one outward forward: egress:<port> -> the host's <port>."""
    return [(port, TARGET_HOST, port, None)]


def container_env(harness: str, loadout: str = DEFAULT_LOADOUT) -> dict:
    env = dict(sandbox_net.proxy_env((sandbox_net.GATE_ALIAS,)))
    env.update({"HOME": HOME, "NODE_USE_ENV_PROXY": "1"})    # Node 24 fetch honours HTTP(S)_PROXY
    env.update(HARNESSES[harness]["env"])
    if loadout == "recommended":
        env.update(HARNESSES[harness].get("loadout_env") or {})
    return env


def sidecar_name(tag: str) -> str:
    return f"{PREFIX}-browser-{sandbox_net._safe(tag)}"


def sidecar_argv(tag: str) -> list[str]:
    """`docker run` for the browser sidecar: on the run's internal network as
    `harness` (the harness container joins its namespace), Chrome's proxy the
    gate (SIDECAR_PROXY), so the page reaches the sandbox's loopback directly
    and public 80/443 through the gate -- never the host. Publishes nothing."""
    return ["run", "-d", "--rm", "--init", "--name", sidecar_name(tag),
            "--label", f"{PREFIX}-browser=1", "--label", f"{PREFIX}-tag={sandbox_net._safe(tag)}",
            *sandbox_net.network_args(tag, "harness", PREFIX),
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--pids-limit", "1024",
            "--memory", "4g", "--cpus", "2",
            "-e", f"SIDECAR_PROXY={sandbox_net.PROXY_URL}",
            "-v", f"{SIDECAR_DIR}:/octo:ro", "--entrypoint", "python3",
            SIDECAR_IMAGE, "/octo/browser_sidecar.py"]


def start_sidecar(tag: str, wait_s: int = 60) -> dict:
    """Start the sidecar and wait for Chrome's DevTools inside its namespace
    (a throwaway curl container joined to it; nothing is published)."""
    rec: dict = {"name": sidecar_name(tag), "image": SIDECAR_IMAGE, "argv": sidecar_argv(tag),
                 "image_id": image_id(SIDECAR_IMAGE), "t_start": time.time(), "ok": False}
    r = _docker(rec["argv"])
    if r.returncode != 0:
        rec["error"] = (r.stderr or r.stdout)[-400:]
        return rec
    rec["container_id"] = r.stdout.strip()[:12]
    t0 = time.time()
    while time.time() - t0 < wait_s:
        q = _docker(["exec", rec["name"], "python3", "-c",
                     "import json,urllib.request;print(json.load(urllib.request.urlopen("
                     f"'{CDP_URL}/json/version',timeout=2)).get('Browser',''))"], 30)
        if q.returncode == 0 and q.stdout.strip():
            rec.update(ok=True, browser=q.stdout.strip(), ready_s=round(time.time() - t0, 1))
            return rec
        time.sleep(1)
    rec["error"] = f"no DevTools answer at {CDP_URL} within {wait_s}s"
    rec["logs"] = _docker(["logs", "--tail", "20", rec["name"]]).stdout[-1500:]
    return rec


def stop_sidecar(tag: str) -> dict:
    name = sidecar_name(tag)
    logs = _docker(["logs", name])
    text = (logs.stdout or "") + (logs.stderr or "")
    r = _docker(["rm", "-f", name])
    return {"removed": r.returncode == 0,
            "chrome_starts": len(re.findall(r"^chrome start n=", text, re.M)),
            "log_tail": text[-600:]}


def run_argv(harness: str, tag: str, project: str, home: str, args: list[str],
             image: str = IMAGE, tty: bool = False, stdin: bool = False,
             loadout: str = DEFAULT_LOADOUT) -> list[str]:
    """`docker run` for the harness. The key is `-e NAME` only; its value
    comes from the environment of this docker child. The recommended loadout
    joins the browser sidecar's network namespace (which is on the run's
    network as `harness`); as-tested joins the network itself."""
    h = HARNESSES[harness]
    net = (["--network", f"container:{sidecar_name(tag)}"] if loadout == "recommended"
           else sandbox_net.network_args(tag, "harness", PREFIX))
    # No stdin unless asked: with `-i` and an open pipe, `opencode run` and
    # `pi -p` wait for EOF before starting (seen 2026-09-26: both hung).
    argv = ["run", "--rm", *(["-i"] if stdin or tty else []), *(["-t"] if tty else []), "--init",
            "--name", f"{PREFIX}-{harness}-{sandbox_net._safe(tag)}",
            "--label", f"{PREFIX}-harness={harness}",
            *net,
            "--user", "1000:1000", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges", "--pids-limit", "1024",
            "--memory", "8g", "--cpus", "4",
            "-v", f"{project}:{WORK}", "-v", f"{home}:{HOME}", "-w", WORK]
    for k, v in container_env(harness, loadout).items():
        argv += ["-e", f"{k}={v}"]
    argv += ["-e", h["key_env"], image, h["bin"], *args]
    return argv


def _docker(args: list[str], timeout: int = 120, env: dict | None = None,
            capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=capture, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout, env=env)


def image_id(ref: str) -> str | None:
    """The local image's content digest (`docker image inspect .Id`): what
    the run record names as the image actually used."""
    try:
        r = _docker(["image", "inspect", "--format", "{{.Id}}", ref], 60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout.strip() or None if r.returncode == 0 else None


MANIFEST = os.path.join(os.path.dirname(os.path.dirname(HERE)), "models", "manifest.yaml")


def recorded_image_id(path: str = MANIFEST) -> str | None:
    """models/manifest.yaml `runtimes: harness-box` `image_id`: the image
    built and checked on record (a rebuild without the cache gets a new id)."""
    try:
        import yaml
        with open(path, encoding="utf-8") as f:
            rts = yaml.safe_load(f).get("runtimes") or []
    except (OSError, ImportError, ValueError):
        return None
    return next((r.get("image_id") for r in rts if r.get("id") == "harness-box"), None)


def image_present(ref: str) -> bool:
    try:
        return _docker(["image", "inspect", ref], 60).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


# ------------------------------------------------------------------ commands

def cmd_build(a) -> int:
    base = a.base or BASE_IMAGE
    if not image_present(base):
        print(f"NOT BUILT: the base image is not in the local store:\n  {base}\n"
              f"Pulling it is a download (~{BASE_PULL_MB} MB compressed, amd64) for the operator "
              f"to decide:\n  docker pull {base}\nthen run this again. Nothing was pulled.")
        return 2
    argv = ["build", "--pull=false", "-t", IMAGE, "--build-arg", f"BASE={base}", HARNESS_DIR]
    print("docker " + " ".join(argv), flush=True)
    return subprocess.run(["docker", *argv]).returncode


def cmd_plan(a) -> int:
    check_port(a.target_port)
    tag = "<tag>"
    print(sandbox_net.plan(tag, PREFIX, "harness", forwards(a.target_port), (sandbox_net.GATE_ALIAS,)))
    if a.loadout == "recommended":
        print("  browser sidecar: docker " + " ".join(sidecar_argv(tag)))
    print("  harness: docker " + " ".join(run_argv(a.harness, tag, "<project>", "<run-dir>\\home",
                                                   ["<args>"], loadout=a.loadout)))
    print(f"  loadout: {a.loadout}" + (f"; skills {loadout_skills(a.harness)}"
                                       if a.loadout == "recommended" else ""))
    print(f"  base URL in the written config: {base_url(a.target_port)}; key: "
          f"{HARNESSES[a.harness]['key_env']} (from --key-file, env only)")
    return 0


def cmd_run(a) -> int:
    check_port(a.target_port)
    h = HARNESSES[a.harness]
    if not image_present(a.image):
        print(f"not_run: image {a.image} is not built (harness_box.py build)", file=sys.stderr)
        return 2
    with open(a.key_file, encoding="utf-8") as f:
        key = f.read().strip()
    if not key:
        print(f"not_run: key file is empty: {a.key_file}", file=sys.stderr)
        return 2
    project = os.path.abspath(a.project)
    run_dir = os.path.abspath(a.run_dir)
    home = os.path.join(run_dir, "home")
    os.makedirs(project, exist_ok=True)
    os.makedirs(home, exist_ok=True)
    written = write_home(a.harness, home, a.target_port, a.config, a.codex_sandbox == "off",
                         a.loadout)
    leaked = secret_in(home, key) + secret_in(project, key)
    if leaked:
        print(f"not_run: the key is in {leaked} (the home and project must not hold it)",
              file=sys.stderr)
        return 2
    tag = a.tag or f"{a.harness}-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
    rec: dict = {"harness": a.harness, "version": VERSIONS[a.harness], "image": a.image,
                 "image_id": image_id(a.image), "base_image": BASE_IMAGE,
                 "tag": tag, "target": f"{TARGET_HOST}:{a.target_port}",
                 "base_url": base_url(a.target_port), "written": written, "loadout": a.loadout,
                 "t0": time.time()}
    if a.harness == "codex":
        # keep = the host test's sandbox_mode (workspace-write), enforced by
        # Codex's bundled bwrap INSIDE the container (checked 2026-09-26);
        # off = danger-full-access, the container alone is the boundary
        rec["codex_sandbox"] = a.codex_sandbox
    net = sandbox_net.up(tag, forwards(a.target_port), prefix=PREFIX)
    rec["sandbox_net"] = net
    rc = 3
    try:
        if not net["ok"]:
            print(f"not_run: sandbox_net: {net.get('error')}", file=sys.stderr)
            rec["not_run"] = "sandbox_net"
            return rc
        if a.loadout == "recommended":
            rec["browser_sidecar"] = start_sidecar(tag)
            if not rec["browser_sidecar"]["ok"]:
                print(f"not_run: browser_sidecar: {rec['browser_sidecar'].get('error')}",
                      file=sys.stderr)
                rec["not_run"] = "browser_sidecar"
                return rc
        argv = run_argv(a.harness, tag, project, home, a.args, a.image, a.tty, a.stdin, a.loadout)
        rec["argv"] = argv                      # names only: the key is `-e NAME`
        env = dict(os.environ)
        env[h["key_env"]] = key
        try:
            rc = subprocess.run(["docker", *argv], env=env, timeout=a.timeout,
                                stdin=None if (a.stdin or a.tty) else subprocess.DEVNULL).returncode
        except subprocess.TimeoutExpired:
            # the docker CLI dies; the container would not: remove it by name
            _docker(["rm", "-f", argv[argv.index("--name") + 1]], 60)
            rc, rec["timed_out"] = 124, a.timeout
        rec["rc"] = rc
        return rc
    finally:
        if "browser_sidecar" in rec:        # after the harness: it lives in the sidecar's namespace
            rec["browser_sidecar"]["stop"] = stop_sidecar(tag)
        rec["stop"] = sandbox_net.down(tag, prefix=PREFIX)
        rec["t1"] = time.time()
        with open(os.path.join(run_dir, "harness_box.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")


CURL = r"""
set +e
p=$1
echo "allowed  http://egress:$p/health -> $(curl -sS -m 8 -o /dev/null -w '%{http_code}' http://egress:$p/health 2>&1)"
for t in host.docker.internal:11434 host.docker.internal:2019 192.168.65.254:11434 192.168.65.254:2019; do
  echo "direct   http://$t/ -> $(curl -sS -m 5 --noproxy '*' -o /dev/null -w '%{http_code}' http://$t/ 2>&1 | tr '\n' ' ')"
  echo "proxied  http://$t/ -> $(curl -sS -m 8 -o /dev/null -w '%{http_code}' http://$t/ 2>&1 | tr '\n' ' ')"
done
for t in egress:11434 egress:2019; do
  echo "gate     http://$t/ -> $(curl -sS -m 5 --noproxy '*' -o /dev/null -w '%{http_code}' http://$t/ 2>&1 | tr '\n' ' ')"
done
"""


def curl_verdict(lines: list[str], port: int) -> dict:
    """curl's %{http_code} is 000 when nothing answered (no route, no name,
    refused); the gate's refusal is a 403."""
    by = {ln.split(" -> ")[0].split()[-1] + "|" + ln.split()[0]: ln.split(" -> ", 1)[1]
          for ln in lines if " -> " in ln}
    allowed = by.get(f"http://egress:{port}/health|allowed", "")
    direct = [v for k, v in by.items() if k.endswith("|direct")]
    proxied = [v for k, v in by.items() if k.endswith("|proxied")]
    gate = [v for k, v in by.items() if k.endswith("|gate")]
    return {"curl_allowed_200": allowed.strip().startswith("200"),
            "curl_direct_no_answer": bool(direct) and all("000" in v for v in direct),
            "curl_proxied_403": bool(proxied) and all("403" in v for v in proxied),
            "curl_gate_ports_closed": bool(gate) and all("000" in v for v in gate)}


def cmd_verify(a) -> int:
    """Docker only, no model: the network as a harness run gets it, probed
    from inside (sandbox_net's probe + curl), and each harness's --version
    when the image is built."""
    check_port(a.target_port)
    port = a.target_port
    built = image_present(a.image)
    image = a.image if built else sandbox_net.GATE_IMAGE
    tag = f"verify-{os.getpid()}"
    extra_deny = [f"{TARGET_HOST}:{port}", f"192.168.65.254:{port}"]
    curl_out: list[str] = []

    def runner(spec: dict) -> tuple[int, str]:
        spec["deny"] = sorted(set(spec["deny"]) | set(extra_deny))
        env = container_env("opencode")
        env_args = [x for k, v in env.items() if k.upper().endswith("PROXY")
                    for x in ("-e", f"{k}={v}")]
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "probe.py"), "w", encoding="utf-8") as f:
                f.write(sandbox_net.PROBE)
            with open(os.path.join(d, "curl.sh"), "w", encoding="utf-8", newline="\n") as f:
                f.write(CURL)
            base = ["run", "--rm", *sandbox_net.network_args(tag, "harness", PREFIX), *env_args,
                    "--user", "1000:1000", "--cap-drop", "ALL", "--security-opt",
                    "no-new-privileges", "-e", "HOME=/tmp", "-v", f"{d}:/p:ro"]
            c = _docker([*base, "--entrypoint", "bash", image, "/p/curl.sh", str(port)], 300)
            curl_out.extend((c.stdout or "").splitlines())
            r = _docker([*base, "--entrypoint", "python3", image, "/p/probe.py",
                         json.dumps(spec)], 900)
        return r.returncode, r.stdout + ("\n" + r.stderr if r.returncode else "")

    res = sandbox_net.verify(tag, image=image, tools=("npm", "git"), prefix=PREFIX,
                             forwards=forwards(port), allow_http=[f"http://egress:{port}/health"],
                             refuse_direct=["egress:11434", "egress:2019", "egress:1235",
                                            "egress:80", "egress:443"],
                             no_proxy_extra=(sandbox_net.GATE_ALIAS,), extra_ports=(port,),
                             runner=runner)
    res["verdict"].pop("ok", None)
    res["verdict"].update(curl_verdict(curl_out, port))
    versions = {}
    if built:
        for hname, h in HARNESSES.items():
            r = _docker(["run", "--rm", "--network", "none", "--user", "1000:1000",
                         "--cap-drop", "ALL", "-e", "HOME=/tmp", a.image, h["bin"], "--version"], 120)
            versions[hname] = (r.stdout or r.stderr).strip()[-120:]
            res["verdict"][f"{hname}_version"] = VERSIONS[hname] in versions[hname]
        # the default loadout's tools, at their pinned versions
        r = _docker(["run", "--rm", "--network", "none", "--user", "1000:1000", "--cap-drop", "ALL",
                     "-e", "HOME=/tmp", "--entrypoint", "sh", a.image, "-c",
                     "playwright-mcp --version; typescript-language-server --version; tsc --version; "
                     "pyright --version; agent-browser --version; fd --version"], 120)
        versions["loadout"] = " | ".join((r.stdout or "").split("\n")).strip(" |")
        want_v = ["Version " + LOADOUT_PACKAGES["@playwright/mcp"], LOADOUT_PACKAGES["typescript-language-server"],
                  "Version " + LOADOUT_PACKAGES["typescript"], "pyright " + LOADOUT_PACKAGES["pyright"],
                  "agent-browser " + LOADOUT_PACKAGES["agent-browser"], "fd 10.5.0"]
        res["verdict"]["loadout_versions"] = all(w in versions["loadout"] for w in want_v)
        got, want = image_id(a.image), recorded_image_id()
        res["image_id"], res["recorded_image_id"] = got, want
        res["verdict"]["image_is_the_recorded_one"] = bool(got) and got == want
    res["verdict"]["ok"] = all(res["verdict"].values())
    print(f"image: {image}" + ("" if built else f"  (STAND-IN: {a.image} is not built; "
                                                 "the network checks do not depend on it)"))
    print(f"forward: egress:{port} -> {TARGET_HOST}:{port} (the one non-global target)")
    for ln in curl_out:
        print("curl " + ln)
    sandbox_net.report(res)
    print("harness --version: " + (json.dumps(versions) if built else
                                   "NOT RUN (image not built: harness_box.py build)"))
    if built:
        print(f"image id {res['image_id']}; models/manifest.yaml harness-box records "
              f"{res['recorded_image_id']}")
    if a.json:
        print(json.dumps(res, indent=1, default=str))
    return 0 if res["verdict"]["ok"] else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--base", help=f"default {BASE_IMAGE}")
    p = sub.add_parser("plan")
    p.add_argument("harness", choices=sorted(HARNESSES))
    p.add_argument("--target-port", type=int, default=1234)
    p.add_argument("--loadout", choices=LOADOUTS, default=DEFAULT_LOADOUT)
    v = sub.add_parser("verify")
    v.add_argument("--target-port", type=int, default=1234)
    v.add_argument("--image", default=IMAGE)
    v.add_argument("--json", action="store_true")
    r = sub.add_parser("run")
    r.add_argument("harness", choices=sorted(HARNESSES))
    r.add_argument("--key-file", required=True)
    r.add_argument("--project", required=True, help="mounted at /work")
    r.add_argument("--run-dir", required=True, help="the run home (<run-dir>/home) and record")
    r.add_argument("--target-port", type=int, default=1234,
                   help="the host port the one forward reaches: :1234 or a relay")
    r.add_argument("--config", help="source config (default: the host test's)")
    r.add_argument("--image", default=IMAGE)
    r.add_argument("--tag")
    r.add_argument("--tty", action="store_true")
    r.add_argument("--stdin", action="store_true", help="pass this process's stdin to the harness")
    r.add_argument("--loadout", choices=LOADOUTS, default=DEFAULT_LOADOUT,
                   help="recommended (default): the default loadout, browser sidecar included; "
                        "as-tested: the host test's config only")
    r.add_argument("--codex-sandbox", choices=("keep", "off"), default="keep",
                   help="keep (default): workspace-write, Codex's bwrap runs in the container")
    r.add_argument("--timeout", type=float, default=None,
                   help="seconds; the container is removed when it expires")
    argv = sys.argv[1:] if argv is None else argv
    # Everything after the first `--` goes to the harness untouched (argparse's
    # REMAINDER would also swallow our own options written after the name).
    cut = argv.index("--") if "--" in argv else len(argv)
    a = ap.parse_args(argv[:cut])
    a.args = argv[cut + 1:]
    return {"build": cmd_build, "plan": cmd_plan, "verify": cmd_verify, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
