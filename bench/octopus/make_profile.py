#!/usr/bin/env python
"""Create the dogfood Hermes profile for the Octopus runs (Windows Hermes).

    python bench/octopus/make_profile.py [--home C:\\Users\\jwals\\octo\\hermes-home]

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
               docker_persist_across_processes false, and two marked lines
               run.py rewrites per run: docker_shared_container_key and
               docker_volumes ([<run folder>:/workspace]). They are in the FILE,
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
  .env         HERMES_CUSTOM_127_0_0_1_1234_API_KEY = the hermes-dogfood dev
               key, read from WSL ~/.hermes/.env (OPENAI_API_KEY) and written
               directly. Never printed; this script prints only its length.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

SRC = os.path.join(os.environ.get("LOCALAPPDATA", ""), "hermes", "config.yaml")
IMAGE = ("nikolaik/python-nodejs@sha256:"
         "140156d7165a3d18b919bc8e9e21584c0b6099d7c2161efa05ee98b50f9f5d73")
RELAY = "http://127.0.0.1:18234/v1"
KEY_VAR = "HERMES_CUSTOM_127_0_0_1_1234_API_KEY"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", default=r"C:\Users\jwals\octo\hermes-home")
    a = ap.parse_args()
    os.makedirs(a.home, exist_ok=True)
    text = open(SRC, encoding="utf-8").read()
    subs = [
        (r"(?m)^(compression:\n(?:  #[^\n]*\n)*  enabled: )false", r"\1true"),
        (r'(?m)^(  backend: )"local"', r'\1"docker"'),
        (r"(?m)^(  docker_mount_cwd_to_workspace: )false", r"\1true"),
        (r"(?m)^(  container_persistent: )true[^\n]*",
         r"\1false\n"
         f'  docker_image: "{IMAGE}"\n'
         "  docker_persist_across_processes: false\n"
         '  docker_shared_container_key: ""   # OCTO-RUN-KEY (run.py rewrites per run)\n'
         "  docker_volumes: []   # OCTO-RUN-VOLUMES (run.py rewrites per run)"),
        (r"(?m)^(  lifetime_seconds: )300", r"\g<1>21600"),
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
        f"    api_key: ${{{KEY_VAR}}}\n")
    with open(os.path.join(a.home, "config.yaml"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)

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
    print(f"profile at {a.home}: config.yaml written; .env holds a {len(key)}-char key")
    return 0


if __name__ == "__main__":
    sys.exit(main())
