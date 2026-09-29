#!/usr/bin/env python
"""Harness tool arms for the Octopus runs: which Hermes toolsets the model
gets. Configuration only -- Hermes is never patched (operator rule).

THE DEFAULT LOADOUT (operator, 2026-09-26: "All of our harnesses should have
usable browser tools attached to them, and a generally useful normal tool
loadout"; "LSP should probably be in the default loadout for everyone").
The browser is the DEFAULT arm (DEFAULT_ARM), sandbox host, not an A/B arm;
`--tools default` is the explicit opt-out (the loadout without the browser).
Every arm also gets the LOADOUT lines (_LOADOUT_EDITS, docs/HARNESSES.md
"Default loadout"): Tool Search off (process_manage direct, three bridge
tools gone), the external password managers off as vault sources, and the
pinned type checkers (TOOLS_VOLUME: tsc + pyright from the harness box
image, read-only at /opt/yamadori-tools in the terminal container; run.py
mounts it; the `type-check` skill says how). Hermes has no LSP under the
docker backend (tools/file_operations_lint.py _lsp_local_only) and no MCP
language-server bridge passed vetting (docs/HARNESSES.md), so types are
checked through the terminal.

    python bench/octopus/toolset_arms.py browser                      # dry run (sandbox browser)
    python bench/octopus/toolset_arms.py browser --browser-host host  # dry run, host browser (opt-in)
    python bench/octopus/toolset_arms.py browser --schemas            # + the tools it adds, their size
    python bench/octopus/toolset_arms.py lean --schemas               # the default (loadout-2), measured
    python bench/octopus/toolset_arms.py measure <run_id>             # what a run did with its page
    python bench/octopus/toolset_arms.py verify                       # the default (lean), in Docker, no model
    python bench/octopus/toolset_arms.py verify --arm browser         # loadout-1's browser, in Docker
    python bench/octopus/run.py --variant V0 --arm xhigh --tools browser --dry-run

run.py --tools <arm> [--browser-host sandbox|host] applies an arm on every run
(set_profile_run). Every run states its arm, so a control run after a browser
run is back to control: each edit is a tagged line (`# OCTO-ARM:<tag>`),
rewritten in whichever direction the run needs; a line it replaces or follows
must occur exactly once.

WHY `browser` WAS OFF (until 2026-09-26). Nobody chose it. The operator's Windows config
(make_profile.py copies it) got `agent.disabled_toolsets` and
`platform_toolsets.cli: [file, skills, terminal, vision]` in ONE write at
2026-09-24 10:37:07 (%LOCALAPPDATA%\\hermes\\backups\\config\\
config.yaml.good.20260924-103650 vs -103707): Hermes' "Blank Slate" setup,
hermes_cli/setup_quick.py:164-191. Hermes' shipped default is
`disabled_toolsets: []` (hermes_cli/config_defaults.py:259), `cli: [hermes-cli]`.

BOTH BROWSER HOSTS (Hermes ee5ee84, %LOCALAPPDATA%\\hermes\\hermes-agent):
  - platform_toolsets.cli gains `browser`, disabled_toolsets loses it (both
    layers: disabled_toolsets runs LAST, hermes_cli/tools_config.py:622-625).
  - browser.backend "off": unset means the Browser Use CLI whenever it can
    run (tools/browser_use_cli.py:219-227) and it is installed here
    (%LOCALAPPDATA%\\hermes\\bin\\browser-use.exe): the model would get
    `browser_exec`, Python run on the HOST (model_tools.py:391-395), and the
    built-in tools report unavailable in that mode
    (tools/browser_tool_install.py:303-304).
  - browser.allow_private_urls true: REQUIRED in both. Hermes judges a URL on
    the host before the browser sees it; with TERMINAL_ENV=docker, or with a
    CDP endpoint (never trusted as local), the browser is not "local"
    (tools/browser_tool_cloud.py:164-183) and http://localhost:3001 is refused
    as private (tools/browser_tool.py:634-661). What "private" then reaches is
    what differs between the hosts (below).
  - browser.inactivity_timeout 120 -> 21600: idle sessions are closed
    (tools/browser_tool_lifecycle.py:147-172); this model thinks for minutes.
  - agent-browser on the Windows host, in BOTH: Hermes drives every backend
    through that CLI (tools/browser_tool_session.py:713-745, `--cdp <ws>` for
    a CDP endpoint) and refuses a command without it
    (tools/browser_tool_session.py:577-584). PREREQUISITE (operator, once; a
    download): npm install -g --prefix C:\\Users\\jwals\\octo\\hermes-home\\node
    agent-browser@0.26.0 -- Hermes' managed location for this profile
    (tools/browser_tool_install.py:39-44), the only release in Hermes' pin
    ^0.26.0 (tools/browser_tool.py:147-149). Without it Hermes falls back to
    an unpinned `npx` download; prereqs() refuses the arm instead.

--browser-host sandbox (DEFAULT). The browser runs in a SIDECAR container
(browser_sidecar.py in the grader's octo-playwright:1.63.0 image: Chromium
headless shell 153, no new download), started by run.py before Hermes and
removed after it. The sidecar owns a network namespace and Hermes' terminal
container joins it (terminal.docker_extra_args `--network container:<sidecar>`;
Hermes passes extra args last, tools/environments/docker.py:600-603, and adds
no --network of its own while docker_network is true, :683-693), so
`localhost:3001` is the model's own server for its shell AND its browser. The
sidecar sits on the run's internal network (sandbox_net.py, every arm: no
route to the host, internet only through the allow-public-only gate); the
gate publishes only CDP, 127.0.0.1:9222 -> gate -> the sidecar, and Hermes attaches
with browser.cdp_url (tools/browser_tool_cdp.py:51-60; env BROWSER_CDP_URL
would win), resolved through /json/version (:13-48). With an endpoint set,
Hermes needs no Chromium on the host (tools/browser_tool_install.py:308-310;
_is_local_mode false, tools/browser_tool_cloud.py:158-161). The sidecar's
Chromium sends everything but loopback to a closed proxy, so the page reaches
the sandbox and nothing else. Docker check 2026-09-26 (no Hermes, no model):
host /json/version answered with ws://127.0.0.1:9222/...; a CDP client in the
namespace loaded the game (200, its title) and got ERR_PROXY_CONNECTION_FAILED
for example.com and for host.docker.internal -- which the TERMINAL container
did reach then, even a host listener bound to 127.0.0.1 (Docker Desktop; closed
since by sandbox_net.py, #47); Browser.close restarted Chrome and the joined
container stayed online.

--browser-host host (opt-in). agent-browser + the installed Chrome on the
Windows host (AGENT_BROWSER_EXECUTABLE_PATH; Hermes finds no Chromium
otherwise, tools/browser_tool_install.py:228-243), the game port published as
127.0.0.1:3001 by the gate (-> octo-term:3001; the terminal container itself is on
the internal network). RISK: the host browser can navigate to the host's loopback
services (llama-swap :11434, SearXNG :8888, ...) -- Hermes names it,
tools/browser_tool_cloud.py:180-181. restrict_evaluate (no fetch/XHR/storage in
evaluated JS, tools/browser_tool_eval_policy.py:139-151) limits it to GETs.

What reaches the model from browser_vision: agent-browser writes the
screenshot on the HOST (<HERMES_HOME>\\cache\\screenshots), Hermes reads the
file and sends text + one image_url to its auxiliary vision client
(tools/browser_tool_vision.py:85-144). No path or base64 is written by the
model -- not the #46 path.

THE LEAN ARM (`lean`, THE DEFAULT since loadout-2; operator, 2026-09-28:
"hermes loads too many tools, think it out to the tools we need for
success"). The task needs files, a shell with a background dev server, and a
browser to CHECK PROGRESS: open the page, read its console errors, look at
it. Tool by tool (docs/HARNESSES.md s0 "Hermes, lean"):
  file      read_file, write_file, patch, search_files -- the project's files
            (one toolset, all four; each is used).
  terminal  terminal (npm install, the build), process_manage (the dev
            server in the background, its log, stop it).
  vision    vision_analyze: the only Hermes tool that turns a screenshot
            into something the model sees (text mode, the auxiliary client
            -> our proxy, docs/VISION.md 4b). With it offered, the proxy
            withholds yama_describe_image (proxy.TOOL_OVERLAPS).
  playwright (MCP) browser_navigate, browser_console_messages,
            browser_take_screenshot, from the official @playwright/mcp
            0.0.82 in the harness box image, attached to the run's sidecar
            Chrome over CDP (`tools.include`; utilities off: `resources`,
            `prompts` false). Its console output carries uncaught page errors
            with their text (loadout_check: "ReferenceError: undefinedFn is
            not defined"), which Hermes' own browser_console drops (s0 FOUND).
CUT, and why: the `browser` toolset (10 tools + the 5 browser_vault_* that
ride with it and cannot be removed: click/type/scroll/back/press/get_images
are interaction the check does not need, the vault is credential access),
`skills` (skills_list/skill_view/skill_manage: skill_manage writes persistent
state; the type check is the project's own `tsc` through the build and the
checkers stay mounted at /opt/yamadori-tools; the screenshot skill served
the vision_analyze path route, which the MCP screenshot now feeds).
HOW HERMES TAKES AN MCP SCREENSHOT (tools/mcp_tool_content.py
_cache_mcp_image_block): an MCP image block is written to the profile's image
cache on the host and the tool result carries `MEDIA:<host path>` as TEXT --
no pixels reach the model. vision_analyze reads that path (a host path inside
a media cache root is a permitted host read under the docker backend,
tools/image_source.py _permitted_host_read_target), so looking is two calls:
take_screenshot, then vision_analyze(<the MEDIA path>).
The MCP server runs WITHOUT a host install: Hermes spawns it as a stdio
server `docker run -i --rm --pull never ... <harness box image>
playwright-mcp --cdp-endpoint http://127.0.0.1:9322` in the sidecar's
network namespace (mcp_argv), so it sees only what the sidecar's Chrome sees.
The gate publishes nothing for this arm (no CDP on the host). The sidecar's
error mirror is off for it (Playwright reports page errors itself; the mirror
would print each twice). `browser` (loadout-1, the previous default) and
`default` (no browser) stay as opt-out arms.
"""
from __future__ import annotations

import argparse
import atexit
import difflib
import glob
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sandbox_net  # noqa: E402  (the internal network + egress gate every arm runs on)

HERMES_HOME = r"C:\Users\jwals\octo\hermes-home"
HERMES_SRC = os.path.join(os.environ.get("LOCALAPPDATA", r"C:\Users\jwals\AppData\Local"),
                          "hermes", "hermes-agent")
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
AGENT_BROWSER_VERSION = "0.26.0"
PORT = 3001                          # the game (the spec's port)
CDP_HOST_PORT = 9222                 # sandbox: host loopback -> the sidecar's forwarder
SIDECAR_IMAGE = "octo-playwright:1.63.0"   # grade.py's PW_IMAGE (playwright.Dockerfile)
SIDECAR_FORWARD_PORT = 9323          # browser_sidecar.FORWARD_PORT
GATE_CDP_PORT = 9223                 # the gate's CDP forward (host 9222 -> gate 9223 -> sidecar 9323)
BROWSER_HOSTS = ("sandbox", "host")
DEFAULT_BROWSER_HOST = "sandbox"
# The default loadout had the browser toolset from 2026-09-26 (operator);
# since 2026-09-28 it is the lean arm (operator: "hermes loads too many tools,
# think it out to the tools we need for success"). `browser` is the opt-out
# that keeps loadout-1.
DEFAULT_ARM = "lean"
LOADOUT_VERSION = "loadout-2"        # run rows carry it: loadout-1 = the browser arm as default
# THE LEAN ARM's browser: @playwright/mcp (the version harness_box pins,
# LOADOUT_PACKAGES) as Hermes MCP server `playwright`, only these tools.
MCP_SERVER = "playwright"
MCP_TOOLS = ("browser_navigate", "browser_console_messages", "browser_take_screenshot")
# Its tools/list, captured (offline measurement only: Hermes' own registration
# of a cached manifest, the lazy path, needs no server running).
MCP_FIXTURE = os.path.join(HERE, "playwright_mcp_0.0.82_tools.json")
# The pinned type checkers for the terminal container (the default loadout):
# typescript 5.9.3 + pyright 1.1.414 copied out of the harness box image
# (bench/sandbox/harness, pinned by its lockfile) into a named volume, mounted
# read-only. The volume's label names the image it came from.
TOOLS_VOLUME = "yamadori-typecheck-tools1"
TOOLS_MOUNT = "/opt/yamadori-tools"
SIDECAR_PLACEHOLDER = "octo-browser-<run_id>-p<n>"

HOST_RISK = ("RISK (--browser-host host): the browser runs on the Windows host with "
             "allow_private_urls on, so it can navigate to the host's loopback services "
             "(llama-swap :11434, SearXNG :8888, the tools API ...); restrict_evaluate blocks "
             "fetch/XHR/storage in evaluated JS, not navigation (Hermes: "
             "tools/browser_tool_cloud.py:180-181). The sandbox host has no such reach.")


def _line(content: str, tag: str) -> str:
    return f"{content}   # OCTO-ARM:{tag} (toolset_arms.py)"


# One entry per tagged line. `control`: the line it replaces (restored on the
# way back); else `after` / `after_re`: the line it is inserted after (removed
# on the way back). `hosts`: only for these browser hosts (None = both).
_BROWSER_EDITS = [
    {"tag": "disabled", "control": "    - browser", "arm": lambda c: "    # - browser"},
    # With a CDP endpoint Hermes also offers browser_cdp (raw CDP, 3,565 chars)
    # and browser_dialog (1,688): the `browser-cdp` toolset (tools/
    # browser_cdp_tool.py:396-398, tools/browser_dialog_tool.py:101-103).
    # Kept off in both hosts, so the two arms offer the same 15 tools.
    {"tag": "no_cdp_tools", "after": "  disabled_toolsets:", "arm": lambda c: "    - browser-cdp"},
    {"tag": "cli", "control": "  cli: [file, skills, terminal, vision]",
     "arm": lambda c: "  cli: [browser, file, skills, terminal, vision]"},
    {"tag": "inactivity", "control": "  inactivity_timeout: 120",
     "arm": lambda c: "  inactivity_timeout: 21600"},
    {"tag": "backend", "after": "browser:", "arm": lambda c: '  backend: "off"'},
    {"tag": "private", "after": "browser:", "arm": lambda c: "  allow_private_urls: true"},
    {"tag": "cdp", "after": "browser:", "hosts": ("sandbox",),
     "arm": lambda c: f'  cdp_url: "http://127.0.0.1:{CDP_HOST_PORT}"'},
    # both hosts (the default loadout): no fetch/XHR/storage/cookies in
    # browser_console(expression=...), tools/browser_tool_eval_policy.py
    {"tag": "restrict", "after": "browser:", "arm": lambda c: "  restrict_evaluate: true"},
]

# THE LEAN ARM (the default, loadout-2): file, terminal, vision and the
# Playwright MCP server's three tools; no `browser`, no `skills` toolset.
# `append`: a top-level line added at the end of the profile (before the
# loadout lines), removed on the way back; a profile that already has an
# untagged top-level key of that name is refused.
_LEAN_EDITS = [
    {"tag": "cli_lean", "control": "  cli: [file, skills, terminal, vision]",
     "arm": lambda c: f"  cli: [file, terminal, vision, {MCP_SERVER}]"},
    {"tag": "mcp", "append": "mcp_servers", "arm": lambda c: mcp_line(c["sidecar"])},
]

# The loadout lines EVERY arm gets (the opt-out included), appended at the end
# of the profile as one-line top-level keys. `key`: the top-level key; a
# profile that already has it untagged is refused (never two).
_LOADOUT_EDITS = [
    # Tool Search off: `process_manage` (dev servers) is offered directly and
    # the tool_search/tool_describe/tool_call bridges go (1,741 chars in this
    # profile, docs/HARNESSES.md s2); MCP tools would stay direct too.
    # "off" quoted: YAML 1.1 reads a bare off as false.
    {"tag": "tool_search", "key": "tools",
     "line": 'tools: {tool_search: {enabled: "off"}}'},
    # The vault tools ride with the browser and no key removes them (4,339
    # chars, tools/browser_vault_tool.py:727-770). What they can reach: the
    # profile's own local vault (empty) and detected password managers,
    # which this turns off as sources (hermes_cli/config_defaults.py vault).
    {"tag": "vault", "key": "vault",
     "line": "vault: {onepassword: {enabled: false}, bitwarden: {enabled: false}}"},
]
# Loadout lines placed INSIDE an existing block: after an anchor line that
# must occur exactly once. skill_manage cannot be turned off alone (it is in
# the `skills` toolset with skills_list/skill_view, toolsets.py:109, and no
# key removes one tool), so what it creates goes to a run-scoped folder that
# run.py empties before each run's first prompt (RUN_SKILLS_DIR): a skill the
# model saves never reaches the next run. Its edits to existing skills are
# not contained; run.py re-installs ours every prompt.
RUN_SKILLS_DIR = "run-skills"          # relative to HERMES_HOME (skills.create_dir)
_LOADOUT_INSERTS = [
    {"tag": "create_dir", "after": "  creation_nudge_interval: 0",
     "line": f'  create_dir: "{RUN_SKILLS_DIR}"'},
]

# Tags an earlier version wrote and no arm writes now: removed on every apply.
# `docker_args` (the terminal's docker_extra_args) moved to run.py's
# OCTO-RUN-NET line, which every arm needs (docker_extra_args, below).
RETIRED_TAGS = ("docker_args",)

ARMS = {"default": {"edits": []}, "browser": {"edits": _BROWSER_EDITS},
        "lean": {"edits": _LEAN_EDITS}}


def uses_sidecar(arm: str, browser_host: str = DEFAULT_BROWSER_HOST) -> bool:
    """Does this arm run the browser sidecar (and the terminal join its
    namespace)? The lean arm always; the browser arm in the sandbox host."""
    return arm == "lean" or (arm == "browser" and browser_host == "sandbox")


def _hb():
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "sandbox"))
    import harness_box
    return harness_box


def mcp_argv(sidecar: str = SIDECAR_PLACEHOLDER) -> list[str]:
    """The stdio command Hermes spawns for the Playwright MCP server (no host
    install): the harness box image, already built and never pulled
    (`--pull never`), in the sidecar's network namespace -- the Chrome it
    attaches to is the sidecar's (CDP on that namespace's loopback), the page
    is the model's dev server there, and the sidecar's proxy rules are the
    browser's. The image's own user (node, uid 1000), every capability
    dropped, no-new-privileges, the harness box's pids limit. `-i` keeps
    stdin open (the MCP channel); when Hermes exits, stdin closes, the server
    exits and --rm removes the container. The labels let run.py remove one
    Hermes left behind. The server's args are harness_box.PLAYWRIGHT_MCP,
    the ones OpenCode and Codex run (--no-webmcp: a page cannot add tools)."""
    hb = _hb()
    return ["run", "-i", "--rm", "--init", "--pull", "never",
            "--network", f"container:{sidecar}",
            "--label", "octo-browser-mcp=1", "--label", f"octo-mcp-of={sidecar}",
            "--user", "1000:1000", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--pids-limit", "1024", hb.IMAGE, *hb.PLAYWRIGHT_MCP]


def mcp_server_config(sidecar: str = SIDECAR_PLACEHOLDER) -> dict:
    """Hermes' mcp_servers.<name> entry: the command, only MCP_TOOLS
    (`tools.include`, tools/mcp_tool_registration.py _make_tool_filter), and
    no generated resource/prompt utilities (`resources`/`prompts` false,
    _select_utility_schemas)."""
    return {"command": "docker", "args": mcp_argv(sidecar),
            "tools": {"include": list(MCP_TOOLS), "resources": False, "prompts": False}}


def mcp_line(sidecar: str = SIDECAR_PLACEHOLDER) -> str:
    """The profile's top-level mcp_servers, one line (JSON is YAML flow)."""
    return "mcp_servers: " + json.dumps({MCP_SERVER: mcp_server_config(sidecar)})


def mcp_cleanup(sidecar: str) -> dict:
    """Remove any Playwright MCP container Hermes left for this sidecar
    (before the sidecar: it lives in its namespace)."""
    r = _docker(["ps", "-aq", "--filter", f"label=octo-mcp-of={sidecar}"], 60)
    ids = r.stdout.split()
    if ids:
        _docker(["rm", "-f", *ids], 60)
    return {"left": len(ids)}


def _once(lines: list[str], pred, what: str) -> int:
    hits = [i for i, ln in enumerate(lines) if pred(ln)]
    if len(hits) != 1:
        raise SystemExit(f"toolset_arms: {what} matched {len(hits)} times")
    return hits[0]


def apply_loadout(text: str, on: bool = True) -> str:
    """The profile with the loadout lines present (`on`) or gone. Idempotent."""
    src = text.split("\n")
    lines = [ln for ln in src
             if not any(f"# OCTO-LOADOUT:{e['tag']} " in ln for e in _LOADOUT_EDITS)]
    inserts = [ln for ln in lines
               if any(f"# OCTO-LOADOUT:{e['tag']} " in ln for e in _LOADOUT_INSERTS)]
    lines = [ln for ln in lines if ln not in inserts]
    if on:
        for e in _LOADOUT_INSERTS:
            i = _once(lines, lambda ln: ln == e["after"], repr(e["after"]))
            lines.insert(i + 1, f"{e['line']}   # OCTO-LOADOUT:{e['tag']} (toolset_arms.py)")
    if not on:
        if len(lines) != len(src):          # drop the blank line the loadout added before its lines
            while lines and lines[-1] == "":
                lines.pop()
            lines.append("")
        return "\n".join(lines)
    for e in _LOADOUT_EDITS:
        if any(re.match(rf"{e['key']}\s*:", ln) for ln in lines):
            raise SystemExit(f"toolset_arms: the profile already has a top-level {e['key']}: "
                             f"(the loadout would duplicate it)")
    while lines and lines[-1] == "":
        lines.pop()
    lines += [""] + [f"{e['line']}   # OCTO-LOADOUT:{e['tag']} (toolset_arms.py)"
                     for e in _LOADOUT_EDITS] + [""]
    return "\n".join(lines)


def clear_run_skills(home: str = HERMES_HOME) -> int:
    """Empty the run-scoped skill folder (before a run's first prompt);
    returns how many skills a previous run left there."""
    import shutil
    d = os.path.join(home, RUN_SKILLS_DIR)
    n = len(os.listdir(d)) if os.path.isdir(d) else 0
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)
    return n


def apply(text: str, arm: str, browser_host: str = DEFAULT_BROWSER_HOST,
          sidecar: str = SIDECAR_PLACEHOLDER, loadout: bool = True) -> str:
    """The profile text with every arm's tagged lines set for `arm` (others to
    control), and the loadout lines present (`loadout`, every arm) or gone."""
    return apply_loadout(_apply_arm(text, arm, browser_host, sidecar), loadout)


def _apply_arm(text: str, arm: str, browser_host: str = DEFAULT_BROWSER_HOST,
               sidecar: str = SIDECAR_PLACEHOLDER) -> str:
    if arm not in ARMS:
        raise SystemExit(f"unknown tools arm {arm!r} (known: {', '.join(ARMS)})")
    if browser_host not in BROWSER_HOSTS:
        raise SystemExit(f"unknown browser host {browser_host!r} (known: {', '.join(BROWSER_HOSTS)})")
    if arm != "default":
        # always from control, so switching arms or hosts lands on the same bytes
        text = _apply_arm(text, "default")
    ctx = {"browser_host": browser_host, "sidecar": sidecar}
    lines = [ln for ln in text.split("\n")
             if not any(f"# OCTO-ARM:{t} " in ln for t in RETIRED_TAGS)]
    for name, spec in ARMS.items():
        # reverse order: several insertions after one anchor keep their order
        for e in reversed(spec["edits"]):
            want = name == arm and browser_host in (e.get("hosts") or BROWSER_HOSTS)
            mark = f"# OCTO-ARM:{e['tag']} "
            present = [i for i, ln in enumerate(lines) if mark in ln]
            if len(present) > 1:
                raise SystemExit(f"toolset_arms: tag {e['tag']} on {len(present)} lines")
            if want:
                new = _line(e["arm"](ctx), e["tag"])
                if present:
                    lines[present[0]] = new
                elif "append" in e:
                    if any(re.match(rf"{e['append']}\s*:", ln) for ln in lines):
                        raise SystemExit(f"toolset_arms: the profile already has a top-level "
                                         f"{e['append']}: (the arm would duplicate it)")
                    # before the loadout's trailing lines (apply_loadout
                    # re-adds them after it), so a re-apply lands on the same bytes
                    lines[:] = [ln for ln in lines if not any(
                        f"# OCTO-LOADOUT:{x['tag']} " in ln for x in _LOADOUT_EDITS)]
                    while lines and lines[-1] == "":
                        lines.pop()
                    lines += [new, ""]
                elif "control" in e:
                    lines[_once(lines, lambda ln: ln == e["control"], repr(e["control"]))] = new
                else:
                    if "after" in e:
                        i = _once(lines, lambda ln: ln == e["after"], repr(e["after"]))
                    else:
                        i = _once(lines, lambda ln: re.fullmatch(e["after_re"], ln) is not None,
                                  e["after_re"])
                    lines.insert(i + 1, new)
            elif present:
                if "control" in e:
                    lines[present[0]] = e["control"]
                else:
                    del lines[present[0]]
    return "\n".join(lines)


# ------------------------------------------------------------------ network
# Every arm's terminal container is on the run's sandbox network
# (sandbox_net.py: internal network + allow-public-only gate; #47). What
# differs per arm is only HOW it joins, and what the gate forwards in.

def docker_extra_args(arm: str, browser_host: str = DEFAULT_BROWSER_HOST,
                      sidecar: str = SIDECAR_PLACEHOLDER,
                      net_tag: str = "<run_id>-p<n>") -> list[str]:
    """terminal.docker_extra_args for the Hermes terminal container (run.py
    writes them into the profile's OCTO-RUN-NET line and TERMINAL_DOCKER_EXTRA_ARGS).
    Sandbox browser: the sidecar's namespace (the sidecar is on the internal
    network). Otherwise: the internal network itself. Both: the gate as proxy.
    Never a published port: an internal network publishes nothing, so the
    gate publishes what the host needs (gate_forwards)."""
    if uses_sidecar(arm, browser_host):
        net = ["--network", f"container:{sidecar}"]
    else:
        net = sandbox_net.network_args(net_tag, sandbox_net.TERMINAL_ALIAS)
    return net + sandbox_net.env_args()


def gate_forwards(arm: str, browser_host: str = DEFAULT_BROWSER_HOST) -> list[tuple]:
    """What the gate publishes on host loopback: (listen, target alias,
    target port, host port). Sandbox browser: CDP for agent-browser. Host
    browser: the game port, for Chrome on Windows. Lean: nothing (its MCP
    server runs in the sidecar's namespace, not on the host)."""
    if arm != "browser":
        return []
    if browser_host == "sandbox":
        return [(GATE_CDP_PORT, sandbox_net.BROWSER_ALIAS, SIDECAR_FORWARD_PORT, CDP_HOST_PORT)]
    return [(PORT, sandbox_net.TERMINAL_ALIAS, PORT, PORT)]


def env(arm: str, browser_host: str = DEFAULT_BROWSER_HOST) -> dict:
    """Env added to the Hermes process."""
    if arm == "browser" and browser_host == "host":
        return {"AGENT_BROWSER_EXECUTABLE_PATH": CHROME,
                "AGENT_BROWSER_ARGS": "--use-angle=swiftshader,--enable-unsafe-swiftshader"}
    return {}


def _chrome_version() -> str | None:
    if not os.path.isfile(CHROME):
        return None
    p = subprocess.run(["powershell", "-NoProfile", "-Command",
                        f"(Get-Item '{CHROME}').VersionInfo.ProductVersion"],
                       capture_output=True, text=True, timeout=30)
    return p.stdout.strip() or None


def _port_free(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) != 0


def _docker(args: list[str], timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)


def prereqs(arm: str, browser_host: str = DEFAULT_BROWSER_HOST,
            home: str = HERMES_HOME) -> dict:
    """What the arm needs on the host. `ok` False -> run.py refuses the run."""
    if arm == "default":
        return {"ok": True}
    if arm == "lean":
        # the sidecar's image and the harness box image (the MCP server);
        # nothing on the host, and neither is ever pulled
        out = {}
        for key, ref in (("sidecar_image", SIDECAR_IMAGE), ("mcp_image", _hb().IMAGE)):
            try:
                r = _docker(["image", "inspect", ref, "--format", "{{.Id}}"], 60)
                out[key] = r.stdout.strip() if r.returncode == 0 else None
            except (OSError, subprocess.TimeoutExpired):
                out[key] = None
        out["problems"] = ([] if out["sidecar_image"] else
                           [f"no {SIDECAR_IMAGE} image (grade.py builds it from playwright.Dockerfile)"]) + \
                          ([] if out["mcp_image"] else
                           [f"no {_hb().IMAGE} image (bench/sandbox/harness_box.py build)"])
        out["ok"] = not out["problems"]
        return out
    out: dict = {"browser_host": browser_host}
    pkg = os.path.join(home, "node", "node_modules", "agent-browser", "package.json")
    try:
        out["agent_browser"] = json.load(open(pkg, encoding="utf-8")).get("version")
    except OSError:
        out["agent_browser"] = None
    out["agent_browser_cmd"] = os.path.isfile(os.path.join(home, "node", "agent-browser.cmd"))
    problems = []
    if out["agent_browser"] != AGENT_BROWSER_VERSION or not out["agent_browser_cmd"]:
        problems.append(f"agent-browser {AGENT_BROWSER_VERSION} not in {home}\\node: run "
                        f"npm install -g --prefix {home}\\node agent-browser@{AGENT_BROWSER_VERSION}")
    if browser_host == "host":
        out["chrome"] = _chrome_version()
        out["port_free"] = _port_free(PORT)
        if not out["chrome"]:
            problems.append(f"no Chrome at {CHROME}")
        if not out["port_free"]:
            problems.append(f"host port {PORT} is taken (the container publishes it)")
    else:
        try:
            r = _docker(["image", "inspect", SIDECAR_IMAGE, "--format", "{{.Id}}"], 60)
            out["sidecar_image"] = r.stdout.strip() if r.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            out["sidecar_image"] = None
        out["cdp_port_free"] = _port_free(CDP_HOST_PORT)
        if not out["sidecar_image"]:
            problems.append(f"no {SIDECAR_IMAGE} image (grade.py builds it from "
                            "playwright.Dockerfile on first use)")
        if not out["cdp_port_free"]:
            problems.append(f"host port {CDP_HOST_PORT} is taken (the sidecar publishes CDP there)")
    out["problems"] = problems
    out["ok"] = not problems
    return out


# ------------------------------------------------------------------ type checkers
# The loadout's type checkers for the TERMINAL container (every arm). Hermes
# runs LSP only on its local backend (tools/file_operations_lint.py
# _lsp_local_only), and its own post-write lint skips .ts files under a
# tsconfig.json and otherwise runs `npx tsc`, which needs the project's own
# typescript. So the loadout mounts pinned checkers the model runs itself
# (skill `type-check`): copied out of the harness box image, whose lockfile
# pins them (typescript 5.9.3, pyright 1.1.414), into a named volume; no
# download. The volume's label records the image id it came from.

def _box_image() -> str:
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "sandbox"))
    import harness_box
    return harness_box.IMAGE


POPULATE = (
    "set -e; cp -a /opt/harness/node_modules/typescript /opt/harness/node_modules/pyright /dst/; "
    "mkdir -p /dst/bin; "
    "printf '#!/bin/sh\\nexec node %(m)s/typescript/bin/tsc \"$@\"\\n' > /dst/bin/tsc; "
    "printf '#!/bin/sh\\nexec node %(m)s/pyright/index.js \"$@\"\\n' > /dst/bin/pyright; "
    "chmod 0755 /dst/bin/tsc /dst/bin/pyright; chmod -R a+rX /dst; "
    "node /dst/typescript/bin/tsc --version; node /dst/pyright/index.js --version"
) % {"m": TOOLS_MOUNT}


def tools_volume(create: bool = False) -> dict:
    """The type-checker volume: present and from the current box image? With
    `create`, (re)make it from that image (docker only, --network none)."""
    image = _box_image()
    r = _docker(["image", "inspect", image, "--format", "{{.Id}}"], 60)
    want = r.stdout.strip() if r.returncode == 0 else None
    v = _docker(["volume", "inspect", TOOLS_VOLUME, "--format",
                 '{{index .Labels "yamadori.image"}}'], 60)
    have = v.stdout.strip() if v.returncode == 0 else None
    out = {"volume": TOOLS_VOLUME, "mount": f"{TOOLS_MOUNT}:ro", "image": image,
           "image_id": want, "volume_image_id": have}
    if create and want and have != want:
        _docker(["volume", "rm", "-f", TOOLS_VOLUME], 60)
        _docker(["volume", "create", "--label", f"yamadori.image={want}", TOOLS_VOLUME], 60)
        p = _docker(["run", "--rm", "--network", "none", "--user", "0:0", "-v",
                     f"{TOOLS_VOLUME}:/dst", "--entrypoint", "sh", image, "-c", POPULATE], 300)
        out["created"] = {"rc": p.returncode, "out": (p.stdout + p.stderr)[-400:]}
        if p.returncode != 0:
            _docker(["volume", "rm", "-f", TOOLS_VOLUME], 60)
            have = None
        else:
            have = want
        out["volume_image_id"] = have
    out["ok"] = bool(want) and have == want
    if not want:
        out["problem"] = f"the harness box image {image} is not built (bench/sandbox/harness_box.py build)"
    elif not out["ok"]:
        out["problem"] = f"volume {TOOLS_VOLUME} is missing or from another image (tools_volume(create=True))"
    return out


def volume_spec() -> str:
    """The docker_volumes entry for the terminal container."""
    return f"{TOOLS_VOLUME}:{TOOLS_MOUNT}:ro"


# ------------------------------------------------------------------ sidecar

def sidecar_name(run_id: str, n: int) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "-", f"octo-browser-{run_id}-p{n}")


def sidecar_argv(name: str, run_id: str, net_tag: str = "<run_id>-p<n>",
                 mirror: bool = True) -> list[str]:
    """`docker run` for the sandbox browser, on the run's INTERNAL network
    (sandbox_net.py) as `octo-browser`. It publishes nothing: the gate
    publishes CDP on host loopback and forwards it here (gate_forwards), or,
    for the lean arm, nothing at all. `mirror`: the browser arm's error
    mirror (Hermes' browser_console drops uncaught errors' text); off for the
    lean arm, whose Playwright console output has them already."""
    return ["run", "-d", "--rm", "--init", "--name", name,
            "--label", "octo-browser=1", "--label", f"octo-run={run_id}",
            *sandbox_net.network_args(net_tag, sandbox_net.BROWSER_ALIAS),
            # repeat uncaught page errors as console.error lines: Hermes drops
            # their text (browser_sidecar.py "error mirror")
            *(["-e", "SIDECAR_ERROR_MIRROR=1"] if mirror else []),
            "-v", f"{HERE}:/octo:ro", "--entrypoint", "python3",
            SIDECAR_IMAGE, "/octo/browser_sidecar.py"]


_LIVE: set[str] = set()     # sidecars this process started and has not removed


@atexit.register
def _stop_leftovers() -> None:
    """A runner that dies between start and stop must not leave a sidecar holding
    the CDP port (--rm removes it once stopped)."""
    for name in list(_LIVE):
        stop_sidecar(name)


def _cdp_in_namespace(name: str) -> dict:
    """/json/version of the sidecar's Chrome, read INSIDE its namespace (the
    lean arm publishes no CDP on the host)."""
    import browser_sidecar
    q = _docker(["exec", name, "python3", "-c",
                 "import json,urllib.request;print(json.dumps(json.load(urllib.request.urlopen("
                 f"'http://127.0.0.1:{browser_sidecar.CHROME_PORT}/json/version',timeout=2))))"], 30)
    if q.returncode != 0 or not q.stdout.strip():
        raise OSError((q.stderr or "no answer")[-200:])
    return json.loads(q.stdout.strip().splitlines()[-1])


def start_sidecar(name: str, run_id: str, wait_s: int = 60,
                  net_tag: str = "<run_id>-p<n>", arm: str = "browser") -> dict:
    """Start the sidecar (after the gate: sandbox_net.up) and wait for CDP:
    on the host through the gate (browser arm), or inside the sidecar's own
    namespace (lean arm: nothing is published). The record goes in the run row."""
    rec: dict = {"name": name, "image": SIDECAR_IMAGE,
                 "argv": sidecar_argv(name, run_id, net_tag, mirror=arm == "browser"),
                 "t_start": time.time(), "ok": False}
    r = _docker(rec["argv"])
    if r.returncode == 0:
        _LIVE.add(name)
    rec["container_id"] = r.stdout.strip()[:12]
    if r.returncode != 0:
        rec["error"] = (r.stderr or r.stdout)[-400:]
        return rec
    url = f"http://127.0.0.1:{CDP_HOST_PORT}/json/version"
    t0 = time.time()
    while time.time() - t0 < wait_s:
        try:
            v = (_cdp_in_namespace(name) if arm == "lean"
                 else json.loads(urllib.request.urlopen(url, timeout=2).read()))
            rec.update(ok=True, browser=v.get("Browser"),
                       cdp_ws=str(v.get("webSocketDebuggerUrl", "")).split("/devtools")[0],
                       ready_s=round(time.time() - t0, 1))
            return rec
        except (OSError, ValueError):
            time.sleep(1)
    rec["error"] = f"no CDP answer at {url} within {wait_s}s"
    rec["logs"] = _docker(["logs", "--tail", "20", name]).stdout[-1500:]
    return rec


def stop_sidecar(name: str) -> dict:
    """Remove the sidecar (after Hermes' containers: they live in its namespace)."""
    _LIVE.discard(name)
    logs = _docker(["logs", name])
    text = (logs.stdout or "") + (logs.stderr or "")
    r = _docker(["rm", "-f", name])
    return {"t_stop": time.time(), "removed": r.returncode == 0,
            "chrome_starts": len(re.findall(r"^chrome start n=", text, re.M)),
            "log_tail": text[-800:]}


# ------------------------------------------------------------------ dry run

# THE STACK GUARD for every Hermes probe this module runs (the schema build
# and verify): building the tool list probes the profile's endpoint
# (check_vision_requirements -> agent/model_metadata.py detect_local_server_type
# walks /api/v1/models, /v1/props, /props, /version on 127.0.0.1:1234;
# docs/HARNESSES.md s2), and a probe must never reach the stack. A connect is
# refused unless it is to loopback on a port that is not a stack port
# (Windows asyncio's self-pipe is a loopback socketpair on an ephemeral port).
# The refused addresses are printed, so the probe shows what Hermes tried.
_GUARD = r"""
import socket as _s, sys as _sys
_STACK = set(__STACK_PORTS__)
_REFUSED = []
_orig_connect = _s.socket.connect
def _guarded(self, addr):
    host, port = (addr[0], addr[1]) if isinstance(addr, tuple) else (str(addr), None)
    if host in ("127.0.0.1", "::1", "localhost") and port not in _STACK:
        return _orig_connect(self, addr)
    _REFUSED.append("%s:%s" % (host, port))
    raise ConnectionRefusedError("offline probe: connect to %s:%s refused" % (host, port))
_s.socket.connect = _guarded
_orig_cc = _s.create_connection
def _guarded_cc(addr, *a, **k):
    host, port = addr[0], addr[1]
    if host in ("127.0.0.1", "::1", "localhost") and port not in _STACK:
        return _orig_cc(addr, *a, **k)
    _REFUSED.append("%s:%s" % (host, port))
    raise ConnectionRefusedError("offline probe: connect to %s:%s refused" % (host, port))
_s.create_connection = _guarded_cc
"""


def stack_ports() -> list[int]:
    """Ports a Hermes probe must never connect to: the proxy, its relay, and
    every host service the harness box refuses as a forward target."""
    import run as runmod
    return sorted({1234, runmod.RELAY_PORT, *_hb().FORBIDDEN_PORTS})


def _guard_src() -> str:
    return _GUARD.replace("__STACK_PORTS__", json.dumps(stack_ports()))


# Hermes' own tool builder over the profile. When the profile has an MCP
# server, its tools are registered from a CACHED MANIFEST (the captured
# tools/list, MCP_FIXTURE) through Hermes' lazy-startup path
# (tools/mcp_tool_registration.py _register_from_cache_sync), which applies the
# profile's tools.include and utility switches exactly as a live connect does
# and spawns nothing.
_SCHEMA_PROBE = r"""
import json, os, sys
sys.path.insert(0, os.getcwd())
import yaml, model_tools
cfg = yaml.safe_load(open(os.path.join(os.environ["HERMES_HOME"], "config.yaml"), encoding="utf-8"))
dis = cfg["agent"].get("disabled_toolsets") or []
en = cfg["platform_toolsets"]["cli"]
mcp = {}
for name, scfg in (cfg.get("mcp_servers") or {}).items():
    fx = json.load(open(os.environ["OCTO_MCP_FIXTURE"], encoding="utf-8"))
    from tools.mcp_tool_registration import _register_from_cache_sync
    mcp[name] = _register_from_cache_sync(name, scfg, {"tools": fx["tools"], "utility_tools": []})
defs = model_tools.get_tool_definitions(enabled_toolsets=en, disabled_toolsets=dis, quiet_mode=True)
print(json.dumps({"defs": defs, "mcp": mcp, "refused": _REFUSED}, ensure_ascii=False))
"""


_PROBE_REFUSED: list[str] = []     # what the last probes tried to reach (and were refused)


def _tool_defs(text: str, browser_host: str, env_arm: str) -> list:
    py = os.path.join(HERMES_SRC, "venv", "Scripts", "python.exe")
    with tempfile.TemporaryDirectory() as home:
        with open(os.path.join(home, "config.yaml"), "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        e = {**os.environ, "HERMES_HOME": home, "TERMINAL_ENV": "docker",
             "PYTHONIOENCODING": "utf-8", "OCTO_MCP_FIXTURE": MCP_FIXTURE, **env(env_arm, browser_host)}
        p = subprocess.run([py, "-c", _guard_src() + _SCHEMA_PROBE], cwd=HERMES_SRC, env=e,
                           capture_output=True, text=True, encoding="utf-8", timeout=300)
        # the temp home's name appears in skill_manage's description
        last = (p.stdout.strip().splitlines() or [""])[-1].replace(os.path.basename(home), "<home>")
        try:
            out = json.loads(last)
        except json.JSONDecodeError:
            raise SystemExit(f"schema probe failed: {p.stderr[-2000:]}")
        _PROBE_REFUSED.extend(out.get("refused") or [])
        return out["defs"]


# The profiles schemas() builds: before the loadout, and every arm with it.
SCHEMA_PROFILES = ("before", "default", "browser", "lean")


def schemas(arm: str, base_text: str, browser_host: str = DEFAULT_BROWSER_HOST) -> dict:
    """Hermes' own tool builder (model_tools.get_tool_definitions), run OFFLINE
    in Hermes' venv on a temp copy of each profile (no .env; every connect to
    the stack refused, _GUARD; the schema build reads cdp_url without
    connecting, tools/browser_tool_cdp.py:51-58; an MCP server's tools come
    from its captured manifest). Profiles: `before` (control, no loadout
    lines), `default` (the loadout without a browser), `browser` (loadout-1,
    the default until 2026-09-28), `lean` (the default since), and `arm`."""
    res = {"before": _tool_defs(apply(base_text, "default", loadout=False), browser_host, "default")}
    for a in ("default", "browser", "lean"):
        res[a] = _tool_defs(apply(base_text, a, browser_host), browser_host, a)
    res["opt_out"] = res["default"]
    res["arm"] = res[arm]
    size = lambda x: len(json.dumps(x, ensure_ascii=False))  # noqa: E731
    names = {k: [t["function"]["name"] for t in v] for k, v in res.items()}
    base = {t["function"]["name"]: t for t in res["before"]}
    added = [t for t in res["arm"] if t["function"]["name"] not in base]
    removed = [n for n in names["before"] if n not in names["arm"]]
    changed = [t["function"]["name"] for t in res["arm"]
               if t["function"]["name"] in base and t != base[t["function"]["name"]]]
    return {**{f"{k}_tools": len(v) for k, v in res.items()},
            **{f"{k}_chars": size(v) for k, v in res.items()},
            "names": names, "per_tool": {t["function"]["name"]: size(t) for t in res["arm"]},
            "per_tool_all": {k: {t["function"]["name"]: size(t) for t in v} for k, v in res.items()},
            # the old keys (control = before the loadout)
            "control_tools": len(res["before"]), "control_chars": size(res["before"]),
            "added": [(t["function"]["name"], size(t)) for t in added], "removed": removed,
            "changed": changed, "added_defs": added, "refused": sorted(set(_PROBE_REFUSED))}


def plan(arm: str, browser_host: str = DEFAULT_BROWSER_HOST, home: str = HERMES_HOME,
         with_schemas: bool = False) -> str:
    path = os.path.join(home, "config.yaml")
    text = open(path, encoding="utf-8").read()
    before, target = apply(text, "default", loadout=False), apply(text, arm, browser_host)
    label = arm if arm in ("default", "lean") else f"{arm}, browser host {browser_host}"
    if arm == "default":
        label += " (the opt-out: the loadout WITHOUT a browser)"
    elif arm == "browser":
        label += " (the opt-out: loadout-1, Hermes' own browser toolset)"
    if arm == DEFAULT_ARM:
        label += " (THE DEFAULT LOADOUT)"
    out = [f"tools arm: {label}", f"profile: {path}", "",
           "the profile, against the operator's toolsets before the default loadout:"]
    diff = list(difflib.unified_diff(before.split("\n"), target.split("\n"),
                                     "config.yaml (before the loadout)", f"config.yaml ({label})",
                                     n=1, lineterm=""))
    out += diff or ["(no profile change)"]
    xa = docker_extra_args(arm, browser_host)
    out += ["", "every arm, per prompt (run.py; run row: `sandbox_net`):",
            "  profile: terminal.docker_extra_args: " + json.dumps(xa)
            + "   # OCTO-RUN-NET (and TERMINAL_DOCKER_EXTRA_ARGS, the same list)",
            f"  profile: terminal.docker_volumes gains {volume_spec()!r}   # OCTO-RUN-VOLUMES "
            "(the pinned type checkers, /opt/yamadori-tools/bin/tsc and pyright; the browser "
            "arm's skill `type-check` names them)"]
    out += ["  " + ln for ln in sandbox_net.plan().split("\n")]
    fw = gate_forwards(arm, browser_host)
    out += ["  gate publishes: " + (", ".join(f"127.0.0.1:{hp} -> {h}:{p}" for _l, h, p, hp in fw)
                                    if fw else "nothing")]
    out += ["", "env added to the Hermes process: " + json.dumps(env(arm, browser_host))]
    if uses_sidecar(arm, browser_host):
        out += ["sidecar, started before each prompt and removed after it (run row: "
                "`browser_sidecar`): docker " + " ".join(
                    sidecar_argv(SIDECAR_PLACEHOLDER, "<run_id>", mirror=arm == "browser")),
                "  the browser reaches only the sandbox's loopback (closed proxy for the rest); "
                + ("the host reaches only CDP on 127.0.0.1:%d, through the gate" % CDP_HOST_PORT
                   if arm == "browser" else "nothing of it is published on the host")]
    if arm == "lean":
        out += [f"MCP server `{MCP_SERVER}`, spawned by Hermes over stdio (tools: "
                f"{', '.join(MCP_TOOLS)}): docker " + " ".join(mcp_argv(SIDECAR_PLACEHOLDER)),
                "  an MCP screenshot reaches the model as a `MEDIA:<host path>` line; "
                "vision_analyze(<that path>) looks at it"]
    if arm == "browser" and browser_host == "host":
        out += ["", HOST_RISK]
    out += ["prerequisites: " + json.dumps(prereqs(arm, browser_host, home), indent=1)]
    try:
        tv = tools_volume()
    except (OSError, subprocess.TimeoutExpired) as e:
        tv = {"ok": False, "problem": str(e)}
    out += ["type-checker volume: " + json.dumps(tv)]
    if with_schemas:
        s = schemas(arm, text, browser_host)
        added = sum(n for _, n in s["added"])
        out += ["", "tool list (Hermes' own builder, offline, every stack connect refused; "
                    "chars = len(json.dumps(tool)), not tokenized; ~4-3 chars/token):",
                f"  before the loadout: {s['before_tools']} tools / {s['before_chars']} chars",
                f"  --tools default (loadout, no browser): {s['default_tools']} tools / "
                f"{s['default_chars']} chars",
                f"  --tools browser (loadout-1): {s['browser_tools']} tools / {s['browser_chars']} chars",
                f"  --tools lean (loadout-2): {s['lean_tools']} tools / {s['lean_chars']} chars",
                f"  {label}: {s['arm_tools']} tools / {s['arm_chars']} chars "
                f"({s['arm_chars'] - s['before_chars']:+d} chars against before; added "
                f"{added} chars, ~{added // 4}-{added // 3} tokens)"]
        out += [f"  + {n}: {c} chars" for n, c in s["added"]]
        out += [f"  - {n}" for n in s["removed"]]
        out += [f"  ~ {n}: description changed" for n in s["changed"]]
        out += ["  offered, with chars: " + ", ".join(f"{n} {c}" for n, c in s["per_tool"].items())]
        out += ["  connects the builder tried (refused): " + (", ".join(s["refused"]) or "none")]
    return "\n".join(out)


# ------------------------------------------------------------------ verify (Docker + Hermes' own code, no model)

_HERMES_BROWSER_PROBE = r"""
import json, os, sys
sys.path.insert(0, os.getcwd())
import yaml, model_tools
cfg = yaml.safe_load(open(os.path.join(os.environ["HERMES_HOME"], "config.yaml"), encoding="utf-8"))
en, dis = cfg["platform_toolsets"]["cli"], cfg["agent"].get("disabled_toolsets") or []
mcp = None
if cfg.get("mcp_servers"):
    # the CLI's own path (hermes_cli/mcp_startup.py -> discover_mcp_tools):
    # Hermes spawns the configured stdio server (docker run ... playwright-mcp)
    from tools.mcp_tool_discovery import discover_mcp_tools
    mcp = discover_mcp_tools()
names = [t["function"]["name"] for t in model_tools.get_tool_definitions(
    enabled_toolsets=en, disabled_toolsets=dis, quiet_mode=True)]
out = {"tools": names, "mcp": mcp, "calls": [], "media": None}
for name, args in json.loads(sys.argv[1]):
    try:
        r = model_tools.handle_function_call(name, args, task_id="verify")
    except Exception as e:  # noqa: BLE001
        r = "RAISED " + repr(e)
    out["calls"].append({"tool": name, "args": args, "result": str(r)[:3000]})
    # the screenshot's MEDIA:<host path>: resolved the way vision_analyze
    # resolves an image source (no vision call: that would reach the proxy)
    import re
    def _strings(x):
        if isinstance(x, str):
            yield x
        elif isinstance(x, dict):
            for v in x.values():
                yield from _strings(v)
        elif isinstance(x, list):
            for v in x:
                yield from _strings(v)
    try:
        blob = " ".join(_strings(json.loads(r)))
    except Exception:  # noqa: BLE001
        blob = str(r)
    m = re.search(r"MEDIA:(\S+)", blob)
    if m and out["media"] is None:
        import asyncio
        from tools.image_source import ResolveContext, resolve_image_source
        path = m.group(1)
        try:
            img = asyncio.run(resolve_image_source(path, ResolveContext(task_id="verify")))
            out["media"] = {"path": path, "bytes": len(img.data), "mime": img.mime, "origin": img.origin,
                            "png": img.data[:8] == b"\x89PNG\r\n\x1a\n"}
        except Exception as e:  # noqa: BLE001
            out["media"] = {"path": path, "error": repr(e)[:400]}
out["refused"] = _REFUSED
print("PROBE_JSON " + json.dumps(out))
"""

# The calls each arm's verify makes, in order: open the page, read it, read
# its console, (lean: take a screenshot), then try the host twice.
_VERIFY_CALLS = {
    "browser": [["browser_navigate", {"url": f"http://localhost:{PORT}/"}],
                ["browser_snapshot", {}],
                ["browser_console", {}],
                ["browser_navigate", {"url": "http://host.docker.internal:11434/"}],
                ["browser_navigate", {"url": "http://192.168.65.254:1234/health"}]],
    "lean": [[f"mcp__{MCP_SERVER}__browser_navigate", {"url": f"http://localhost:{PORT}/"}],
             [f"mcp__{MCP_SERVER}__browser_take_screenshot", {"scale": "css"}],
             [f"mcp__{MCP_SERVER}__browser_console_messages", {"level": "error"}],
             [f"mcp__{MCP_SERVER}__browser_navigate", {"url": "http://host.docker.internal:11434/"}],
             [f"mcp__{MCP_SERVER}__browser_navigate", {"url": "http://192.168.65.254:1234/health"}]],
}


def verify_verdict(arm: str, res: dict) -> dict:
    """What verify() must see, per arm (a pure function of its record)."""
    calls = (res.get("hermes") or {}).get("calls") or []
    txt = [c["result"] for c in calls] + [""] * 5
    tools = (res.get("hermes") or {}).get("tools") or []
    term = (res.get("terminal") or {}).get("out", "")
    v = {"terminal_serves_page": "SERVER_200" in term,
         "terminal_host_blocked": "PROXIED_HOST_403" in term,
         "terminal_type_check": "TS2322" in term}
    host_blocked = (all("ERR_" in x or "error" in x.lower() or "403" in x for x in (txt[3], txt[4]))
                    and not any('"ok"' in x or "Ollama" in x for x in (txt[3], txt[4])))
    if arm == "lean":
        want = {f"mcp__{MCP_SERVER}__{t}" for t in MCP_TOOLS} | {
            "read_file", "write_file", "patch", "search_files", "terminal", "process_manage",
            "vision_analyze"}
        media = (res.get("hermes") or {}).get("media") or {}
        v.update({
            "tools_exactly_lean": set(tools) == want,
            "browser_opened_page": "OctoPage" in txt[0],
            "screenshot_media_path_readable": bool(media.get("png")) and media.get("bytes", 0) > 0,
            "browser_console_error": "OCTO_CONSOLE_ERROR" in txt[2],
            "browser_page_exception": "ReferenceError: undefinedFn" in txt[2],
            "browser_host_blocked": host_blocked,
            "mcp_containers_removed": (res.get("mcp_cleanup") or {}).get("left", 1) == 0,
        })
    else:
        v.update({
            "tools_listed": all(t in tools for t in ("browser_navigate", "browser_console",
                                                     "browser_snapshot", "browser_vision",
                                                     "process_manage"))
            and "browser_exec" not in tools and "tool_search" not in tools,
            "browser_opened_page": "OctoPage" in txt[0] or "octo page" in txt[1],
            "browser_console_error": "OCTO_CONSOLE_ERROR" in txt[2],
            "browser_page_exception": "Uncaught ReferenceError: undefinedFn" in txt[2],
            "browser_host_blocked": host_blocked,
        })
    v["ok"] = all(v.values())
    return v


def verify(home: str = HERMES_HOME, arm: str = DEFAULT_ARM) -> dict:
    """An arm's browser, end to end in Docker with Hermes' own tool code (its
    venv, a temp HERMES_HOME holding this profile with the arm applied) and NO
    model: the run's sandbox network and gate, the sidecar, and a
    terminal-like container in the sidecar's namespace (run.py's image and
    args) serving a page on localhost:3001 and running the mounted type
    checkers. Then Hermes opens that page, reads its console, and tries the
    host. `browser`: Hermes' own browser tools over CDP (agent-browser 0.26.0
    from `home`\\node). `lean`: Hermes spawns the Playwright MCP server
    (mcp_argv: docker run of the harness box image in the sidecar's
    namespace), and the screenshot's MEDIA path is resolved as vision_analyze
    would (no vision call: that reaches the proxy). Every connect of the
    Hermes probe to a stack port is refused and recorded (_GUARD). Everything
    is removed afterwards."""
    import shutil
    if arm not in _VERIFY_CALLS:
        raise SystemExit(f"verify: arm {arm!r} has no browser to verify (known: {', '.join(_VERIFY_CALLS)})")
    tag = f"verify-{os.getpid()}"
    name = sidecar_name("verify", os.getpid())
    page = ("<!doctype html><html><head><title>OctoPage</title></head><body><h1>octo page</h1>"
            "<script>console.error('OCTO_CONSOLE_ERROR');setTimeout(function(){ undefinedFn(); }, 10);"
            "</script></body></html>")
    res: dict = {"tag": tag, "arm": arm}
    site = tempfile.mkdtemp(prefix="octo-verify-")
    hh = tempfile.mkdtemp(prefix="octo-verify-home-")
    term = f"octo-verify-term-{os.getpid()}"
    try:
        with open(os.path.join(site, "index.html"), "w", encoding="utf-8") as f:
            f.write(page)
        with open(os.path.join(site, "bad.ts"), "w", encoding="utf-8", newline="\n") as f:
            f.write('export const n: number = "str";\n')
        tv = tools_volume(create=True)
        res["tools_volume"] = {k: tv.get(k) for k in ("ok", "image_id", "problem")}
        res["net"] = sandbox_net.up(tag, gate_forwards(arm, "sandbox"))
        if not res["net"]["ok"]:
            return res
        res["sidecar"] = start_sidecar(name, "verify", net_tag=tag, arm=arm)
        if not res["sidecar"]["ok"]:
            return res
        sys.path.insert(0, HERE)
        import run as runmod
        r = _docker(["run", "-d", "--rm", "--name", term, *docker_extra_args(arm, "sandbox", name, tag),
                     "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                     "-v", f"{site}:/workspace:ro", "-v", volume_spec(), "-w", "/workspace",
                     runmod.IMAGE, "python3", "-m", "http.server", str(PORT), "--bind", "127.0.0.1"])
        res["terminal"] = {"rc": r.returncode, "err": r.stderr[-300:]}
        time.sleep(2)
        t = _docker(["exec", term, "sh", "-c",
                     f"curl -s -o /dev/null -w 'SERVER_%{{http_code}}' --noproxy '*' http://127.0.0.1:{PORT}/; echo; "
                     "curl -s -m 5 -o /dev/null -w 'PROXIED_HOST_%{http_code}' http://host.docker.internal:11434/; echo; "
                     f"{TOOLS_MOUNT}/bin/tsc --noEmit --strict bad.ts; echo TSC_RC=$?"], 120)
        res["terminal"]["out"] = t.stdout[-800:]
        with open(os.path.join(hh, "config.yaml"), "w", encoding="utf-8", newline="\n") as f:
            f.write(apply(open(os.path.join(home, "config.yaml"), encoding="utf-8").read(),
                          arm, "sandbox", name))
        py = os.path.join(HERMES_SRC, "venv", "Scripts", "python.exe")
        e = {**os.environ, "HERMES_HOME": hh, "TERMINAL_ENV": "docker", "PYTHONIOENCODING": "utf-8",
             "PATH": os.path.join(home, "node") + os.pathsep + os.environ.get("PATH", "")}
        p = subprocess.run([py, "-c", _guard_src() + _HERMES_BROWSER_PROBE, json.dumps(_VERIFY_CALLS[arm])],
                           cwd=HERMES_SRC, env=e, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=600)
        line = next((ln for ln in p.stdout.splitlines() if ln.startswith("PROBE_JSON ")), None)
        res["hermes"] = json.loads(line[11:]) if line else {"error": (p.stderr or p.stdout)[-2000:]}
        stack = set(stack_ports())
        res["hermes"]["refused_stack"] = [x for x in res["hermes"].get("refused") or []
                                          if x.rsplit(":", 1)[-1].isdigit()
                                          and int(x.rsplit(":", 1)[-1]) in stack]
    finally:
        _docker(["rm", "-f", term], 60)
        if arm == "lean":
            res["mcp_cleanup"] = mcp_cleanup(name)
        if "sidecar" in res:
            res["sidecar"]["stop"] = stop_sidecar(name)
        if "net" in res:
            res["net"]["stop"] = sandbox_net.down(tag)
        shutil.rmtree(site, ignore_errors=True)
        shutil.rmtree(hh, ignore_errors=True)
    res["verdict"] = verify_verdict(arm, res)
    return res


# ------------------------------------------------------------------ measure

def _events(log_dir: str) -> list[dict]:
    ev = []
    for p in sorted(glob.glob(os.path.join(log_dir, "hermes*.jsonl"))):
        for ln in open(p, encoding="utf-8", errors="replace"):
            try:
                ev.append({**json.loads(ln), "_file": os.path.basename(p)})
            except json.JSONDecodeError:
                pass
    return ev


_TERMINAL_PAGE = re.compile(r"playwright|puppeteer|chromium|headless", re.I)


def measure(run_id: str, logs: str = r"C:\Users\jwals\octo\logs") -> dict:
    """Did the model open its page, and did it see the errors the grader sees?
    Same numbers for both arms (control opens pages through `terminal`)."""
    ev = _events(os.path.join(logs, run_id))
    uses = [e for e in ev if e.get("type") == "tool_use"]
    results = [e for e in ev if e.get("type") == "tool_result"]
    # Hermes' own browser tools, or the lean arm's MCP ones (mcp__playwright__browser_*),
    # counted under the bare name
    mcp_prefix = f"mcp__{MCP_SERVER}__"
    for e in uses + results:
        if str(e.get("name", "")).startswith(mcp_prefix):
            e["name"] = e["name"][len(mcp_prefix):]
    browser = [e for e in uses if str(e.get("name", "")).startswith("browser_")]
    navs = [e for e in browser if e["name"] == "browser_navigate"]
    local_navs = [e for e in navs if f":{PORT}" in json.dumps(e.get("input"))]
    terminal_pages = [e for e in uses if e.get("name") == "terminal"
                      and _TERMINAL_PAGE.search(json.dumps(e.get("input")))]
    seen = set()
    for r in results:
        if r.get("name") == "browser_console_messages":
            # Playwright MCP's text: "### Result", a count line, then one entry
            # per message ("[ERROR] ..." or an uncaught error's own line)
            text = r.get("output") or ""
            try:
                text = json.loads(text).get("result", text) if text.startswith("{") else text
            except (json.JSONDecodeError, AttributeError):
                pass
            seen.update(ln.removeprefix("[ERROR] ").strip() for ln in str(text).splitlines()
                        if ln.strip() and not ln.startswith(("###", "Total messages", " ", "\t")))
            continue
        if r.get("name") != "browser_console":
            continue
        try:
            out = json.loads(r.get("output") or "{}")
        except json.JSONDecodeError:
            continue
        seen.update(x.get("message", "") for x in out.get("js_errors") or [])
        seen.update(m.get("text", "") for m in out.get("console_messages") or []
                    if m.get("type") == "error")
    grade = None
    gpath = os.path.join(HERE, "results", "grades.jsonl")
    if os.path.exists(gpath):
        for ln in open(gpath, encoding="utf-8"):
            r = json.loads(ln)
            if r.get("grade_id", "").split("@")[0] == run_id:
                grade = r                                   # the latest grade of the run
    final = []
    if grade and grade.get("runtime"):
        rt = grade["runtime"]
        final = [x.get("text", "") for x in (rt.get("page_errors") or []) + (rt.get("console_errors") or [])]
    return {
        "run_id": run_id,
        "tool_calls": len(uses),
        "browser_calls": {n: sum(1 for e in browser if e["name"] == n)
                          for n in sorted({e["name"] for e in browser})},
        "opened_page_browser": bool(local_navs),
        "first_open_s": (round((local_navs[0]["timestamp"] - uses[0]["timestamp"]) / 1000)
                         if local_navs and uses else None),
        "opened_page_terminal": len(terminal_pages),
        "errors_seen_by_model": sorted(e for e in seen if e),
        "grade_id": grade and grade.get("grade_id"),
        "spec": grade and f"{grade.get('spec_passed')}/{grade.get('spec_total')}",
        "final_errors": final,
        "final_errors_seen_before_finishing": [x for x in final if any(x in s or s in x for s in seen if s)],
        "final_clean": grade is not None and not final,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", help=f"an arm ({', '.join(ARMS)}), 'measure' or 'verify' (Docker, no model)")
    ap.add_argument("run_id", nargs="?")
    ap.add_argument("--browser-host", choices=BROWSER_HOSTS, default=DEFAULT_BROWSER_HOST)
    ap.add_argument("--home", default=HERMES_HOME)
    ap.add_argument("--schemas", action="store_true",
                    help="also run Hermes' tool builder offline on every arm's profile")
    ap.add_argument("--arm", choices=("lean", "browser"), default=DEFAULT_ARM,
                    help=f"verify: which arm's browser (default {DEFAULT_ARM})")
    a = ap.parse_args()
    if a.what == "measure":
        if not a.run_id:
            ap.error("measure needs a run id")
        print(json.dumps(measure(a.run_id), indent=1))
        return 0
    if a.what == "verify":
        r = verify(a.home, a.arm)
        print(json.dumps(r, indent=1, default=str)[-12000:])
        return 0 if r.get("verdict", {}).get("ok") else 1
    print(plan(a.what, a.browser_host, a.home, a.schemas))
    return 0


if __name__ == "__main__":
    sys.exit(main())
