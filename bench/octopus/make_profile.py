#!/usr/bin/env python
"""Create the dogfood Hermes profile for the Octopus runs (Windows Hermes).

    python bench/octopus/make_profile.py [--home C:\\Users\\jwals\\octo\\hermes-home]
                                         [--wire chat|responses]

Operator, 2026-09-24: a SEPARATE profile, so the operator's own config,
sessions, memories and key are never touched.

  config.yaml  the operator's Windows config (%LOCALAPPDATA%\\hermes\\config.yaml)
               copied with exactly these changes, each required to match once:
                 compression.enabled        false -> true (operator)
                 terminal.backend           local -> docker (the sandbox)
                 terminal.docker_mount_cwd_to_workspace  false -> true (only
                                            the run folder is mounted)
                 terminal.container_persistent  true -> false (fresh per run)
                 terminal.lifetime_seconds  300 -> 21600 (the idle reaper must
                                            not recycle the container while the
                                            model generates between calls)
               plus, under terminal:, docker_image (pinned digest),
               docker_persist_across_processes false, and three marked lines
               run.py rewrites per run: docker_shared_container_key,
               docker_volumes ([<run folder>:/workspace]) and
               docker_extra_args (the run's sandbox network, sandbox_net.py;
               written here as --network none, so a profile used outside
               run.py has no network at all, never the host). They are in the FILE,
               not only TERMINAL_* env: Hermes' config bridge lets explicit
               terminal.* keys override env (tools/terminal_tool.py
               _ensure_terminal_env_bridged), and the smoke run of 2026-09-24
               created its tool container with `volumes: []` although
               TERMINAL_DOCKER_VOLUMES was set.
               and one block appended: a named provider `octo-relay` whose
               base_url is bench/octopus/relay.py on 127.0.0.1:18234. The
               default provider stays custom @ http://127.0.0.1:1234/v1
               (operator's spec); runs select the relay with --provider
               octo-relay so every request's x_yamadori is recorded. The relay
               forwards to :1234 unchanged.
               --wire responses adds `api_mode: codex_responses` to that
               named provider: Hermes then speaks POST /v1/responses (its
               codex_responses transport). It must be a NAMED provider -- a
               plain `provider: custom` with `model.api_mode` is silently
               ignored for a non-OpenAI URL (hermes_cli/runtime_provider.py
               _resolve_plain_custom_api_mode; docs/HARNESS-RESPONSES.md).
               The relay records the Responses stream as it does chat.
               THE DEFAULT LOADOUT (docs/HARNESSES.md "Default loadout"): the
               profile is written with toolset_arms.apply(text, DEFAULT_ARM).
               Since 2026-09-28 (operator: "hermes loads too many tools, think
               it out to the tools we need for success") that is the LEAN arm,
               loadout-2: platform_toolsets.cli [file, terminal, vision,
               playwright] -- read_file, write_file, patch, search_files,
               terminal, process_manage, vision_analyze and the Playwright MCP
               server's browser_navigate, browser_console_messages and
               browser_take_screenshot (mcp_servers.playwright: the harness
               box image run by docker in the run's browser sidecar's
               namespace; run.py rewrites that line per run with the real
               sidecar name) -- plus Tool Search off, the external password
               managers off as vault sources. No `skills`, no `browser`
               toolset. The Blank Slate's disabled toolsets stay disabled
               (benchmark hygiene). run.py re-applies the arm per run
               (`--tools browser` = loadout-1, Hermes' own browser toolset;
               `--tools default` = no browser) and mounts the pinned type
               checkers (toolset_arms.TOOLS_VOLUME).
  skills/      bench/octopus/hermes_skills/* into skills/software-development/
               (look-at-a-screenshot, type-check): read only by the arms that
               offer the `skills` toolset (browser, default), not by lean.
  .env         HERMES_CUSTOM_127_0_0_1_1234_API_KEY = the hermes-dogfood dev
               key, read from WSL ~/.hermes/.env (OPENAI_API_KEY) and written
               directly. Never printed; this script prints only its length.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
HERMES_SKILLS = os.path.join(HERE, "hermes_skills")
sys.path.insert(0, HERE)
import toolset_arms  # noqa: E402  (the default loadout: the browser arm + the loadout lines)
SRC = os.path.join(os.environ.get("LOCALAPPDATA", ""), "hermes", "config.yaml")
IMAGE = ("nikolaik/python-nodejs@sha256:"
         "140156d7165a3d18b919bc8e9e21584c0b6099d7c2161efa05ee98b50f9f5d73")
RELAY = "http://127.0.0.1:18234/v1"
KEY_VAR = "HERMES_CUSTOM_127_0_0_1_1234_API_KEY"


def install_skills(home: str) -> list[str]:
    """Copy bench/octopus/hermes_skills/* into the profile's skills folder
    (run.py also calls this per run, so a profile made before a skill was
    added gets it)."""
    names = sorted(os.listdir(HERMES_SKILLS))
    for name in names:
        dst = os.path.join(home, "skills", "software-development", name)
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(os.path.join(HERMES_SKILLS, name), dst)
    return names


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", default=r"C:\Users\jwals\octo\hermes-home")
    ap.add_argument("--wire", choices=("chat", "responses"), default="chat",
                    help="the API the octo-relay provider speaks")
    a = ap.parse_args()
    os.makedirs(a.home, exist_ok=True)
    text = open(SRC, encoding="utf-8").read()
    subs = [
        (r"(?m)^(compression:\n(?:  #[^\n]*\n)*  enabled: )false", r"\1true"),
        (r'(?m)^(  backend: )"local"', r'\1"docker"'),
        (r"(?m)^(  docker_mount_cwd_to_workspace: )false", r"\1true"),
        # The container's cwd, named for the FILE tools too (pagoda-h3,
        # 2026-09-27). With cwd "." the terminal runs in /workspace, but the
        # file tools anchor a RELATIVE path on the process cwd, a Windows path,
        # posix-joined onto itself (tools/file_tools_paths.py _resolve_base_dir
        # -> _anchor): write_file("pagoda/package.json") resolved to
        # "C:\...\run/C:\...\run/pagoda/package.json" and nothing reached the
        # run folder. "/workspace" makes $TERMINAL_CWD the anchor
        # (_configured_terminal_cwd). The mount is unaffected: docker_volumes
        # names the run folder explicitly (run.py).
        (r'(?m)^(  cwd: )"\."', r'\1"/workspace"'),
        (r"(?m)^(  container_persistent: )true[^\n]*",
         r"\1false\n"
         f'  docker_image: "{IMAGE}"\n'
         "  docker_persist_across_processes: false\n"
         '  docker_shared_container_key: ""   # OCTO-RUN-KEY (run.py rewrites per run)\n'
         "  docker_volumes: []   # OCTO-RUN-VOLUMES (run.py rewrites per run)\n"
         '  docker_extra_args: ["--network", "none"]'
         "   # OCTO-RUN-NET (run.py rewrites per run; sandbox_net.py)"),
        (r"(?m)^(  lifetime_seconds: )300", r"\g<1>21600"),
        # No skill-creation reminders inside a benchmark run (Hermes nudges
        # the model to save a skill every N tool iterations; agent_init.py
        # _skill_nudge_interval). 0 disables it.
        (r"(?m)^(  creation_nudge_interval: )\d+", r"\g<1>0"),
    ]
    for pat, rep in subs:
        text, n = re.subn(pat, rep, text)
        if n != 1:
            sys.exit(f"override matched {n} times: {pat}")
    if re.search(r"(?m)^providers:", text):
        sys.exit("the source config already has a top-level providers: block")
    if not re.search(r'(?m)^  base_url: "http://127\.0\.0\.1:1234/v1"', text):
        sys.exit("the source config's base_url is not http://127.0.0.1:1234/v1")
    text = text.rstrip("\n") + (
        "\n\n# Octopus runs (bench/octopus): the recording relay in front of :1234.\n"
        "providers:\n"
        "  octo-relay:\n"
        f'    base_url: "{RELAY}"\n'
        f"    api_key: ${{{KEY_VAR}}}\n"
        + ("    api_mode: codex_responses   # --wire responses\n"
           if a.wire == "responses" else "")
        # On Responses, Hermes' side calls must show a substantive event
        # within auxiliary.<task>.no_progress_timeout (default 60 s;
        # keep-alives do not count -- agent/auxiliary_client.py
        # _CodexStreamGuard); a compaction thinks at the conversation's own
        # effort first. 300 s is a CHOICE (docs/HARNESS-RESPONSES.md).
        # Hermes' post-turn background review (agent/background_review.py,
        # default on) asked the model to "update the skill library" after
        # v0f's prompt 1: 601 s of GPU, and a skill_manage tool that can
        # write into this profile -- state that would carry into later runs.
        # Off for benchmark runs (operator, 2026-09-26: fix every identified
        # issue before the next run).
        + "\n# Octopus runs: no post-turn skill/memory review.\n"
        "auxiliary:\n"
        "  background_review:\n"
        "    enabled: false\n"
        + ("  compression:   # Responses: room for a compaction to think\n"
           "    no_progress_timeout: 300\n"
           if a.wire == "responses" else ""))
    # the default loadout (toolset_arms: the browser arm, sandbox host, and
    # the loadout lines every arm gets)
    text = toolset_arms.apply(text, toolset_arms.DEFAULT_ARM, toolset_arms.DEFAULT_BROWSER_HOST)
    with open(os.path.join(a.home, "config.yaml"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    # Harness-specific skills live in the HARNESS's own skills folder, not in
    # our server store (operator, 2026-09-26: "We shouldn't carry harness
    # specific skills, that is the job of the harness").
    install_skills(a.home)

    p = subprocess.run(["wsl.exe", "-d", "Ubuntu", "--", "bash", "-lc",
                        "grep -m1 '^OPENAI_API_KEY=' ~/.hermes/.env"],
                       capture_output=True, text=True)
    line = p.stdout.strip()
    key = line.split("=", 1)[1].strip().strip('"').strip("'") if "=" in line else ""
    if not key:
        sys.exit("no OPENAI_API_KEY in WSL ~/.hermes/.env")
    env_path = os.path.join(a.home, ".env")
    with open(env_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{KEY_VAR}={key}\n")
    print(f"profile at {a.home}: config.yaml written (wire: {a.wire}); "
          f".env holds a {len(key)}-char key")
    return 0


if __name__ == "__main__":
    sys.exit(main())
