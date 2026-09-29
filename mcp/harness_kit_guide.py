#!/usr/bin/env python
"""The README of a harness kit (mcp/harness_kit.py export): where each file
goes on the operator's machine, per OS, and how to run the harness. Every
location is the one the harness docs of this repo name, with the doc; where
a doc does not say, the README says so instead of guessing.
"""
from __future__ import annotations

# Per harness: where its files live, per OS, the key's environment
# variable, how it is run, and the doc each fact comes from.
HARNESS = {
    "hermes": {
        "label": "Hermes",
        "tested": "hermes-agent commit ee5ee84 (docs/HARNESSES.md 'Sources and versions')",
        "install": ("Hermes is not installed from npm here: install it by its own instructions. Every finding in "
                    "this kit was made on commit ee5ee84."),
        "home": {"windows": r"%LOCALAPPDATA%\hermes  (or wherever HERMES_HOME points)",
                 "unix": "~/.hermes  (or wherever HERMES_HOME points)"},
        "home_doc": "docs/HARNESSES.md 'Sources and versions' (Windows); docs/TOOLS-API.md (~/.hermes/config.yaml)",
        "config_note": ("Hermes' config.yaml is your own file: MERGE each config/*.yaml block into "
                        "HERMES_HOME/config.yaml by hand (a top-level key that already exists is edited, not "
                        "duplicated)."),
        "skills_dir": {"windows": r"$env:LOCALAPPDATA\hermes\skills\software-development",
                       "unix": "~/.hermes/skills/software-development"},
        "skills_doc": "bench/octopus/make_profile.py install_skills",
        "key_env": "YAMADORI_API_KEY",
        "key_note": ("The provider reads it through key_env; it may also live in HERMES_HOME/.env as "
                     "`YAMADORI_API_KEY=<your key>` (Hermes loads .env first: docs/TOOLS-API.md)."),
        "run": ["hermes chat --provider yamadori -m yamadori"],
        "run_doc": "bench/octopus/run.py hermes_cmd; docs/HARNESS-RESPONSES.md",
    },
    "pi": {
        "label": "Pi",
        "tested": "0.87.1 on Node 24.21.0 (docs/HARNESS-PI.md)",
        "install": "npm (below). Node 24.21.0 is what the tests ran on.",
        "home": {"windows": r"%USERPROFILE%\.pi\agent  (set PI_CODING_AGENT_DIR to it)",
                 "unix": "~/.pi/agent  (set PI_CODING_AGENT_DIR to it)"},
        "home_doc": "bench/sandbox/harness_box.py HARNESSES['pi'] (PI_CODING_AGENT_DIR=$HOME/.pi/agent); docs/HARNESS-PI.md",
        "config_note": "models.json and settings.json go in the agent dir; pi.env lists variables to set in Pi's environment.",
        "skills_dir": {"windows": r"$env:USERPROFILE\.agents\skills", "unix": "~/.agents/skills"},
        "skills_doc": "bench/sandbox/harness_box.py SKILL_HOMES (Pi lists them in its system prompt: verified in the box)",
        "key_env": "YAMADORI_PI_KEY",
        "key_note": "models.json reads it as \"$YAMADORI_PI_KEY\" (an environment reference; never a !command).",
        "run": ["pi --provider yamadori --model yamadori",
                "pi --mode json --thinking xhigh \"prompt\" < /dev/null    # headless: close stdin"],
        "run_doc": "docs/HARNESS-RESPONSES.md (captured); docs/HARNESS-PI.md (headless)",
    },
    "opencode": {
        "label": "OpenCode",
        "tested": "1.18.32 (docs/HARNESS-OPENCODE.md)",
        "install": "npm (below).",
        "home": {"windows": r"any folder of yours, e.g. %USERPROFILE%\.config\opencode",
                 "unix": "any folder of yours, e.g. ~/.config/opencode"},
        "home_doc": "docs/HARNESS-OPENCODE.md 'The config used' (OPENCODE_CONFIG points at the file)",
        "config_note": ("Put opencode.json somewhere and set OPENCODE_CONFIG to its full path; opencode.env lists the "
                        "other variables."),
        "skills_dir": None,
        "skills_doc": "the loadout denies OpenCode's own skill tool (docs/HARNESSES.md s0)",
        "key_env": "YAMADORI_OPENCODE_KEY",
        "key_note": "opencode.json reads it as {env:YAMADORI_OPENCODE_KEY}.",
        "run": ["opencode", "opencode run --format json --variant xhigh -- \"prompt\""],
        "run_doc": "docs/HARNESS-OPENCODE.md 'The config used'",
    },
    "codex": {
        "label": "Codex CLI",
        "tested": "0.157.1 (docs/HARNESS-CODEX.md)",
        "install": "npm (below).",
        "home": {"windows": r"a folder of yours, NOT under %TEMP%, e.g. %USERPROFILE%\.codex  (set CODEX_HOME to it)",
                 "unix": "a folder of yours, e.g. ~/.codex  (set CODEX_HOME to it)"},
        "home_doc": "docs/HARNESS-CODEX.md s1-s2 (CODEX_HOME; the %TEMP% warning)",
        "config_note": ("config.toml and yamadori-catalog.json go in CODEX_HOME; then edit model_catalog_json in "
                        "config.toml to the catalog's full path."),
        "skills_dir": {"windows": r"$env:CODEX_HOME\skills", "unix": "$CODEX_HOME/skills"},
        "skills_doc": "bench/sandbox/harness_box.py SKILL_HOMES ($CODEX_HOME/skills)",
        "key_env": "YAMADORI_CODEX_KEY",
        "key_note": "config.toml names it (env_key) and keeps it out of the model's shell (shell_environment_policy).",
        "run": ["codex", "codex exec --json -C <workspace> \"<prompt>\""],
        "run_doc": "docs/HARNESS-CODEX.md s2",
    },
    "claude-code": {
        "label": "Claude Code",
        "tested": ("not run against this stack yet: the Messages API it speaks (POST /v1/messages) is gated offline "
                   "by mcp/test_messages_api.py; UNMEASURED until a real Claude Code run"),
        "install": "by Anthropic's own instructions (code.claude.com/docs).",
        "home": {"windows": r"%USERPROFILE%\.claude\settings.json (your user), or <project>\.claude\settings.json; "
                            ".mcp.json in the project root",
                 "unix": "~/.claude/settings.json (your user), or <project>/.claude/settings.json; .mcp.json in the "
                         "project root"},
        "home_doc": "code.claude.com/docs/en/settings and /mcp; docs/TOOLS-API.md 'Configuring a client'",
        "config_note": ("settings.json points Claude Code at the proxy (ANTHROPIC_BASE_URL = the public base; Claude "
                        "Code adds /v1/messages), pins every model role to `yamadori` and sets effort medium: MERGE its "
                        "keys into an existing settings.json. claude-code.env lists the key variables. mcp.json is the "
                        "code-intelligence MCP server: save it as .mcp.json in the project root."),
        "skills_dir": None,
        "skills_doc": "no Claude Code skill of ours",
        "key_env": "ANTHROPIC_AUTH_TOKEN",
        "key_note": ("Claude Code sends it as Authorization: Bearer (the proxy also accepts x-api-key, i.e. "
                     "ANTHROPIC_API_KEY). Set it: with only ANTHROPIC_BASE_URL a saved claude.ai login stays the "
                     "credential and every request is refused 401. .mcp.json reads YAMADORI_API_KEY (may be the same "
                     "key)."),
        "run": ["claude", "claude --effort xhigh    # Mirai S (the tier table); /effort in a session",
                "claude -p \"prompt\"         # headless"],
        "run_doc": "code.claude.com/docs/en/model-config ('How to Set Effort Level'); mcp/messages_api.py (effort -> tier)",
    },
    "any": {
        "label": "any harness",
        "tested": "",
        "install": "the downloads below are the general ones (type checkers, the browser server).",
        "home": {}, "home_doc": "", "config_note": "", "skills_dir": None, "skills_doc": "",
        "key_env": "YAMADORI_API_KEY",
        "key_note": "each harness's kit names its own variable.",
        "run": [], "run_doc": "",
    },
}


def _dl_block(d: dict) -> list[str]:
    out = [f"### {d['title']}", "",
           f"- source: {d['source_url']}", f"- version: `{d['version']}` (exact)",
           f"- hash: `{d['hash']}`" if d.get("hash") else f"- commit: `{d['commit']}`",
           f"- licence: {d['licence']}"]
    inst = d.get("install") or {}
    for os_, label in (("windows", "Windows (PowerShell)"), ("unix", "Linux / macOS"), ("any", "any OS")):
        if inst.get(os_):
            out += [f"- {label}:", "", "  ```", f"  {inst[os_]}", "  ```"]
    if d.get("verify"):
        out += ["- verify before you install:", "", "  ```", f"  {d['verify']}", "  ```"]
    return out + [""]


def readme(harness: str, m: dict, rec: list[dict], trying: list[dict], ctx: dict) -> str:
    g = HARNESS[harness]
    base = m["public_base"]
    L = [f"# Yamadori kit: {g['label']}", "",
         "Generated by the Yamadori dashboard (HARNESS TOOLS page) for the machine the next waves run on. "
         "It is text only: it downloaded nothing, and running it installs nothing until YOU run a command below.", "",
         f"- proxy (public base): `{base['url']}` -- from {base['source']}",
         f"- chat API: `{ctx['API_BASE']}`",
         f"- tested version: {g['tested'] or 'n/a'}"]
    w = m["window"]
    if w["context"]:
        L.append(f"- context window {w['context']}, output ceiling {w['max_output']}: from the proxy's /v1/models when this kit was made")
    else:
        L += ["- context window: NOT filled in. Replace `{{CONTEXT_WINDOW}}` and `{{MAX_OUTPUT}}` in the config files with "
              "`context_length` and `max_completion_tokens` from:", "",
              "  ```", f"  curl -H \"Authorization: Bearer $YAMADORI_API_KEY\" {ctx['API_BASE']}/models", "  ```"]
    L += ["", "## The key", "",
          f"**No key is in this kit.** Every place a key goes reads the environment variable `{g['key_env']}` "
          f"(or holds `{ctx['KEY_PLACEHOLDER']}`). {g['key_note']}", "",
          "Mint a key for this harness on the server (`python mcp/accounts.py create <label>`, docs/HERMES.md), then "
          "set it on this machine -- never in a file you commit:", "",
          "- Windows (PowerShell, for your user; open a new terminal after):", "",
          "  ```", f"  [Environment]::SetEnvironmentVariable(\"{g['key_env']}\", \"{ctx['KEY_PLACEHOLDER']}\", \"User\")", "  ```",
          "- Linux / macOS (in ~/.bashrc or ~/.zshrc):", "",
          "  ```", f"  export {g['key_env']}='{ctx['KEY_PLACEHOLDER']}'", "  ```",
          "", f"Replace `{ctx['KEY_PLACEHOLDER']}` with the key.", ""]
    L += ["## 1. Install", "", g["install"], ""]
    if m["downloads"]:
        L += ["Recommended downloads, each pinned. Check the hash first (the `verify` line), then install:", ""]
        for d in m["downloads"]:
            L += _dl_block(d)
    else:
        L += ["No recommended download for this harness.", ""]
    L += ["## 2. Put the files in place", ""]
    if g["home"]:
        L += [f"Where {g['label']}'s files live ({g['home_doc']}):", "",
              f"- Windows: `{g['home'].get('windows', '')}`", f"- Linux / macOS: `{g['home'].get('unix', '')}`", ""]
    if g["config_note"]:
        L += [g["config_note"], ""]
    for c in m["configs"]:
        L.append(f"- `{c['file']}` -> {c['target'] or 'see its entry'}  ({c['title']})")
    if m["configs"]:
        L.append("")
    if any(c["file"].endswith(".env") for c in m["configs"]):
        L += ["A `.env` file here is a list of variables to set in the harness's environment, the same way as the key "
              "(PowerShell SetEnvironmentVariable, or export in your shell profile).", ""]
    if m["skills"]:
        L += ["### Skills", ""]
        if g["skills_dir"]:
            w, u = g["skills_dir"]["windows"], g["skills_dir"]["unix"]
            L += [f"Copy each folder under `skills/` into the harness's skills folder ({g['skills_doc']}), "
                  "from the kit's folder:", "",
                  "- Windows (PowerShell):", "", "  ```",
                  f"  New-Item -ItemType Directory -Force \"{w}\" | Out-Null",
                  f"  Copy-Item -Recurse -Force skills\\* \"{w}\\\"", "  ```",
                  "- Linux / macOS:", "", "  ```", f"  mkdir -p {u} && cp -r skills/* {u}/", "  ```", ""]
        else:
            L += [f"This harness has no skills folder of ours ({g['skills_doc']}); the skills are here for reference.", ""]
        L += [f"- `{s}`" for s in m["skills"]] + [""]
    if m["prompts"]:
        L += ["### Prompts", ""]
        L += [f"- `{p['file']}` -> {p['target'] or 'the harness system prompt'}  ({p['title']})" for p in m["prompts"]]
        L.append("")
    L += ["## 3. Run it", ""]
    if g["run"]:
        L += ["```"] + g["run"] + ["```", "", f"({g['run_doc']})" if g["run_doc"] else "", ""]
    notes = [r for r in rec if r["kind"] == "note"]
    if notes:
        L += ["## Notes (recommended facts)", ""]
        for r in notes:
            L += [f"### {r['title']}", "", r.get("body") or "", "",
                  "Evidence: " + "; ".join(f"{e['ref']} -- {e['showed']}" for e in r["evidence"]), ""]
    if trying:
        L += ["## Trying: NOT recommended yet", "",
              "These are in `trying/`. They are not installed by the steps above; each says why it is still open."
              " Entries marked NEEDS OPERATOR wait for the operator's decision.", ""]
        for r in trying:
            flag = " -- NEEDS OPERATOR" if r["needs_operator"] else ""
            L += [f"### {r['title']}{flag}", "", f"Why open: {r.get('status_why') or ''}", ""]
            if r.get("body") and not r.get("file_name") and r["kind"] not in ("skill", "prompt"):
                L += [r["body"], ""]
            if r.get("download"):
                d = r["download"]
                pin = d.get("hash") or d.get("commit") or f"no pin yet ({d.get('pin_missing_why')})"
                L += [f"Download (not recommended yet): {d.get('source_url')} `{d.get('version')}`, {pin}", ""]
    if m["left_out"]:
        L += ["## Left out of this kit", ""]
        for r in m["left_out"]:
            L.append(f"- {r['title']} ({r['status']}, {r['where']}){': ' + r['why'] if r['why'] else ''}")
        L.append("")
    L += ["## Evidence", "",
          "`kit.json` lists every entry used, its version on the dashboard, its proof class (operator / measured / "
          "verified / unmeasured) and its evidence. UNMEASURED means no run has shown the effect on the model.", ""]
    return "\n".join(L).replace("\n\n\n", "\n\n")
