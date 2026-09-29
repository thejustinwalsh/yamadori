#!/usr/bin/env python
"""THE MCP SERVERS THE PROXY HOSTS FOR THE MODEL: their storage.

Operator, 2026-09-28: "the proxy provides MCP servers to the model
zero-config for every harness, starting with PackageLens", and a dashboard
page for them "like skills" later. This module is the storage that page will
write; mcp/mcp_host.py runs what it lists; GET /dash/api/mcp (mcp/dash_mcp.py)
reads it now. The React page is NOT built.

WHERE. `YAMADORI_MCP_SERVERS` (default index/mcp/servers.json). When the file
does not exist, DEFAULT is what is in force; nothing is written until a
writer (save, set_enabled) is called, so a read never creates live state and
an offline suite that only reads touches nothing.

ONE SERVER (a row of `servers`):
  id            short name: the container is yama-mcp-<id>, its network
                yama-mcp-net-<id> (bench/sandbox/sandbox_net.py)
  enabled       offered to new conversations and started with the proxy
  runtime       "docker": `image`, built from `recipe` (a directory with a
                Dockerfile and a lockfile), run with every capability
                dropped, a read-only root, no credentials, its network only
                through the egress gate (mcp_host.run_argv).
                "local": `command` (an argv), no container -- the offline
                suites' fake server; refused unless YAMADORI_MCP_ALLOW_LOCAL=1
                (a config file must not be a way to run a host command)
  package       what it is (name@version), for the record
  pinned        where its reproducibility record lives (models/manifest.yaml)
  call_timeout_s / call_timeout_why
                one tools/call's bound and its derivation (a number exists
                only with its evidence, AGENTS.md "Claims carry their
                evidence")
  tools         the tools OF OURS it backs, each:
                  name        yama_* (AGENTS.md: every tool of ours on main),
                              verb first (AGENTS.md "Naming")
                  legacy      names it had before (read as this tool, so a
                              stored ledger hop and a conversation's kept
                              offer still find it; like proxy.LEGACY_TOOL_NAMES)
                  upstream    the server's own tool it calls -- or
                  runner      a runner of ours in the server's image and on
                              its gated network instead (RUNNERS:
                              npm_resolve, mcp/npm_resolve.py), with its own
                              call_timeout_s / call_timeout_why
                  render      how its JSON result is shown (mcp_host.RENDER)
                  kind        screen kind: markdown | text
                  description a trigger condition (AGENTS.md "Tool
                              descriptions are prompts"); says it is a server
                              tool that does not touch the workspace
                  parameters  OUR schema (JSON Schema)
                  args        {our argument: the upstream's argument}
                  overlaps    client tool names that answer the same
                              question: ours is withheld beside one
                              (proxy.tool_conflicts, like TOOL_OVERLAPS)
  line          the one system-prompt line added where its tools are
                offered (proxy prepare); {tools} is replaced by the offered
                names
"""
from __future__ import annotations

import copy
import json
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import npm_resolve  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.environ.get("YAMADORI_MCP_SERVERS",
                      os.path.join(ROOT, "index", "mcp", "servers.json"))
VERSION = 1
# The runners of ours a tool row may name instead of an upstream tool.
RUNNERS = {npm_resolve.RUNNER: npm_resolve}

# ---------------------------------------------------------------- PackageLens
# npm packagelens-mcp 0.1.11 (MIT; operator-approved download 2026-09-28,
# "Yes get package lens"). Pinned: mcp_servers/packagelens/ (Dockerfile,
# package.json, package-lock.json; 16 packages, every sha512 in the lockfile,
# registry signatures checked at build) and models/manifest.yaml
# `runtimes: mcp-packagelens` (the image id built from it, the source audit).
#
# THE SUBSET (3 of its 8 tools). The operator's goal: the model DISCOVERS a
# package -- a vague phrase to the registry's exact name, the versions that
# exist now (dist-tags, prereleases), the README -- instead of guessing.
#   smart_search      -> yama_find_package      words -> exact names
#   smart_get_versions-> yama_list_package_versions  every version, date,
#     dist-tags (+ each listed npm version's peerDependencies, which
#     PackageLens does not return: mcp_host.npm_peers, 2026-09-29)
#   smart_get_readme  -> yama_read_package_readme    the README (usage
#     lives there)
# Renamed verb first 2026-09-29 (docs/TOOL-FACTORY.md N3; they were
# yama_package_versions and yama_package_readme: each row's `legacy`).
# And one tool of OURS on the same image and network (2026-09-29, operator:
# "packagelens should be able to resolve peer dep ranges ..."):
#   npm's own resolver -> yama_resolve_packages  every package together: one
#     install line with exact versions, the peers and type packages they
#     need, or the conflict and what fits (mcp/npm_resolve.py)
#     KNOWN, measured 2026-09-28: it reads the packument's `readme` field,
#     which the npm registry serves EMPTY for many packages -- 7 of 20
#     popular ones (express, axios, typescript, zod, date-fns, react-router,
#     tailwindcss) and math, koota, @react-three/fiber, react, zustand; the
#     result then says the README is in the installed package
#     (mcp_host.render_readme). Since 2026-09-29 the proxy reads it from
#     the package's GitHub repository instead (mcp_host.github_readme). The 13 non-empty READMEs of the 20 all
#     passed the screen once their images read as alt text (n=1 read each).
# Left out, from reading its source (dist/, 0.1.11):
#   smart_package_info    for npm it reads the packument's top-level
#                         `version`, which a packument does not have: no
#                         version comes back; the rest is GitHub stars and
#                         download counts
#   smart_get_dependencies  without a version it reads `dependencies` off the
#                         packument (none there): always empty; with one it
#                         returns `dependencies` only -- never
#                         peerDependencies, which is what the operator asked
#                         for (a GAP: nothing here answers peer deps)
#   smart_get_usage_snippet the README's first code block under a
#                         usage/example heading; yama_read_package_readme returns
#                         the whole README, so a second tool would only cost
#                         prompt tokens on every request
#   smart_get_downloads, compare_packages  popularity, not discovery
#
# `ecosystem` is REQUIRED on all three: without it smart_search reads the
# query for language words and else answers "Could not determine which
# package ecosystem" (a vague phrase names none), and the other two search
# all six registries -- PyPI's by downloading its whole simple index -- to
# guess one.
#
# THE TIMEOUT, derived from the server's own code (dist/http.js): each GET
# aborts after DEFAULT_TIMEOUT_MS = 12,000 ms and is retried once
# (retries: 1) after a 300 ms sleep: 2 x 12 s + 0.3 s = 24.3 s per GET. The
# most sequential GETs any of the three makes is 2 (crates.io's README:
# the crate's info, then its readme; npm and PyPI make 1, PyPI's search one
# index GET then parallel 5 s metadata GETs, 24.3 + 10.3 s). 2 x 24.3 =
# 48.6 s. The abort covers only the wait for headers (http.js clears it once
# fetch resolves), so a body still arriving then is bounded by this same
# number on our side.
_PL_GET_S = 2 * 12.0 + 0.3
PL_CALL_TIMEOUT_S = round(2 * _PL_GET_S, 1)

ECOSYSTEM = {
    "type": "string",
    "enum": ["npm", "pypi", "crates", "rubygems", "packagist", "hex"],
    "description": ("The registry: npm for JavaScript and TypeScript, pypi "
                    "for Python, crates for Rust, rubygems for Ruby, "
                    "packagist for PHP, hex for Elixir."),
}
_SERVER_TOOL = ("A server tool: it runs on the Yamadori server and does not "
                "touch your workspace.")

PACKAGELENS = {
    "id": "packagelens",
    "title": "PackageLens",
    "enabled": True,
    "runtime": "docker",
    "package": "packagelens-mcp@0.1.11",
    "licence": "MIT",
    "image": "yamadori-mcp-packagelens:0.1.11",
    "recipe": "mcp_servers/packagelens",
    "pinned": "models/manifest.yaml runtimes: mcp-packagelens",
    # The proxy starts it only when the local image id is the one recorded
    # there (mcp_host.check_image).
    "manifest_id": "mcp-packagelens",
    "network": "egress",
    "call_timeout_s": PL_CALL_TIMEOUT_S,
    "call_timeout_why": (
        "PackageLens dist/http.js: 12 s abort per GET, retried once after "
        "300 ms (24.3 s); at most 2 sequential GETs per tool (crates.io's "
        "README): 48.6 s"),
    "tools": [
        {"name": "yama_find_package", "upstream": "smart_search",
         "render": "search", "kind": "text",
         "description": (
             "Answers 'which package is this, and what is its exact name "
             "in the registry?': searches a package registry by words and "
             "returns the best matches with their exact names, latest "
             "versions, descriptions and repositories. Call it before you "
             "install, import or add to a manifest a package the task names "
             "loosely -- by a nickname, by an organisation and a topic, by "
             "what it does, or by a name you are not sure is the "
             "registry's -- then install the exact name it returns. "
             + _SERVER_TOOL),
         "parameters": {
             "type": "object",
             "properties": {
                 "query": {"type": "string", "description": (
                     "Words for the package: its name as the task gives it, "
                     "a nickname, its organisation and topic, or what it "
                     "does.")},
                 "ecosystem": ECOSYSTEM},
             "required": ["query", "ecosystem"]},
         "args": {"query": "query", "ecosystem": "ecosystem"},
         "overlaps": ["smart_search", "search_npm", "npm_search",
                      "search_packages", "package_search"]},
        {"name": "yama_list_package_versions",
         "legacy": ["yama_package_versions"],
         "upstream": "smart_get_versions",
         "render": "versions", "kind": "text",
         "description": (
             "Answers 'which versions of this package exist right now?': "
             "lists its published versions, newest first, each with its "
             "date and its dist-tags (latest, next, beta, alpha, canary, "
             "...), prereleases included, and each version's peer "
             "dependencies. Call it before you pin, install "
             "or upgrade a package, when the task names a major version or "
             "a prerelease, and when an install reports no matching "
             "version. " + _SERVER_TOOL),
         "parameters": {
             "type": "object",
             "properties": {
                 "package": {"type": "string", "description": (
                     "The package's exact registry name, scope included.")},
                 "ecosystem": ECOSYSTEM,
                 "limit": {"type": "integer", "description": (
                     "Only the newest N versions; leave it out for every "
                     "version.")}},
             "required": ["package", "ecosystem"]},
         "args": {"package": "packageName", "ecosystem": "ecosystem",
                  "limit": "limit"},
         "overlaps": ["smart_get_versions", "get_versions",
                      "package_versions"]},
        {"name": "yama_read_package_readme",
         "legacy": ["yama_package_readme"],
         "upstream": "smart_get_readme",
         "render": "readme", "kind": "markdown",
         "description": (
             "Answers 'how is this package used?': returns its README from "
             "the registry, or the package's GitHub repository when the "
             "registry has none -- the install line, the imports, the API and "
             "the usage examples -- for the latest version or the one you "
             "name. Call it once you know a package's exact name and before "
             "you write the first import or call of a library you have not "
             "used. " + _SERVER_TOOL),
         "parameters": {
             "type": "object",
             "properties": {
                 "package": {"type": "string", "description": (
                     "The package's exact registry name, scope included.")},
                 "ecosystem": ECOSYSTEM,
                 "version": {"type": "string", "description": (
                     "A version or dist-tag; leave it out for the "
                     "registry's README.")}},
             "required": ["package", "ecosystem"]},
         "args": {"package": "packageName", "ecosystem": "ecosystem",
                  "version": "version"},
         "overlaps": ["smart_get_readme", "get_readme", "package_readme"]},
        {"name": npm_resolve.TOOL_NAME, "runner": npm_resolve.RUNNER,
         "render": "resolve", "kind": "text",
         "call_timeout_s": npm_resolve.FETCH_WORST_S,
         "call_timeout_why": npm_resolve.CALL_TIMEOUT_WHY,
         "description": (
             "Answers 'which exact versions of these packages install "
             "together, and what else do they need?': runs npm's own "
             "resolver on every package you name, together, and returns one "
             "install line with an exact version of each -- the peer "
             "dependencies they need (react, react-dom, three, ...) and "
             "their type packages included -- or, when they cannot install "
             "together, which package needs which range and the newest "
             "versions that fit. Call it once with every package you plan to "
             "use before you write package.json or run an install, and again "
             "when an install reports ERESOLVE or a peer dependency "
             "conflict. " + _SERVER_TOOL),
         "parameters": {
             "type": "object",
             "properties": {
                 "packages": {
                     "type": "array", "items": {"type": "string"},
                     "description": (
                         "Every package you plan to use, each its exact "
                         "registry name, optionally with a version, range or "
                         "dist-tag: react, three@^0.186, "
                         "@react-three/fiber@alpha.")},
                 "ecosystem": {"type": "string", "enum": ["npm"],
                               "description": "The registry: npm."},
                 "allow_prerelease": {"type": "boolean", "description": (
                     "true when a prerelease may be chosen for a package "
                     "whose asked range no stable version satisfies (a major "
                     "published only as alpha, beta, rc or canary builds); "
                     "leave it out for stable versions.")}},
             "required": ["packages", "ecosystem"]},
         "overlaps": ["resolve_packages", "resolve_dependencies"]},
    ],
    # Positive and procedural, no prohibition (AGENTS.md "Prompting this
    # model"); the operator's example: "To find a package, its current
    # version and its README, call yama_find_package before installing it."
    # 2026-09-29: it names the resolver's question and its moment (before
    # package.json is written). The wording is UNMEASURED.
    "line": ("\n\nTo find a package, its versions and its README, and the "
             "exact versions that install together, call {tools} before you "
             "write package.json or install it."),
}

DEFAULT = {"version": VERSION, "servers": [PACKAGELENS]}

_LOCK = threading.Lock()
_CACHE: dict = {"mtime": None, "path": None, "cfg": None}


def path() -> str:
    return os.environ.get("YAMADORI_MCP_SERVERS") or PATH


def load() -> dict:
    """The configuration in force: the file when it exists, else DEFAULT.
    Re-read when the file changes (a dashboard write needs no restart)."""
    p = path()
    try:
        mt = os.path.getmtime(p)
    except OSError:
        return copy.deepcopy(DEFAULT)
    with _LOCK:
        if _CACHE["path"] == p and _CACHE["mtime"] == mt and _CACHE["cfg"]:
            return copy.deepcopy(_CACHE["cfg"])
    try:
        with open(p, encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, ValueError) as e:
        print(f"  mcp config {p} unreadable ({type(e).__name__}: {e}); "
              f"the built-in default is in force", flush=True)
        return copy.deepcopy(DEFAULT)
    problems = validate(cfg)
    if problems:
        print(f"  mcp config {p} refused ({'; '.join(problems[:3])}); the "
              f"built-in default is in force", flush=True)
        return copy.deepcopy(DEFAULT)
    with _LOCK:
        _CACHE.update(mtime=mt, path=p, cfg=cfg)
    return copy.deepcopy(cfg)


def validate(cfg: dict) -> list[str]:
    """What is wrong with a configuration ([] when nothing is)."""
    out: list[str] = []
    if not isinstance(cfg, dict) or cfg.get("version") != VERSION:
        return [f"`version` must be {VERSION}"]
    servers = cfg.get("servers")
    if not isinstance(servers, list):
        return ["`servers` must be a list"]
    ids, names = set(), set()
    for s in servers:
        if not isinstance(s, dict) or not s.get("id"):
            out.append("a server without an `id`")
            continue
        sid = str(s["id"])
        if not sid.replace("-", "").replace("_", "").isalnum():
            out.append(f"{sid}: `id` must be letters, digits, - and _")
        if sid in ids:
            out.append(f"{sid}: duplicate id")
        ids.add(sid)
        rt = s.get("runtime")
        if rt == "docker":
            if not s.get("image"):
                out.append(f"{sid}: a docker server needs `image`")
        elif rt == "local":
            if not (isinstance(s.get("command"), list) and s["command"]):
                out.append(f"{sid}: a local server needs `command` (an argv)")
        else:
            out.append(f"{sid}: `runtime` is docker or local")
        if not isinstance(s.get("call_timeout_s"), (int, float)) \
                or not s.get("call_timeout_why"):
            out.append(f"{sid}: `call_timeout_s` needs `call_timeout_why`")
        for t in s.get("tools") or []:
            n = str((t or {}).get("name") or "")
            for m in [n] + [str(x) for x in (t or {}).get("legacy") or []]:
                if not m.startswith("yama_"):
                    out.append(f"{sid}: tool {m!r} must be named yama_*")
                if m in names:
                    out.append(f"{sid}: tool {m!r} is offered twice")
                names.add(m)
            runner = t.get("runner")
            if runner is not None:
                if runner not in RUNNERS:
                    out.append(f"{sid}: tool {n!r} names runner {runner!r}; "
                               f"the runners are {sorted(RUNNERS)}")
                if not isinstance(t.get("call_timeout_s"), (int, float)) \
                        or not t.get("call_timeout_why"):
                    out.append(f"{sid}: tool {n!r}: a runner needs "
                               f"`call_timeout_s` with `call_timeout_why`")
                need = ("description", "parameters")
            else:
                need = ("upstream", "description", "parameters", "args")
            for k in need:
                if not t.get(k):
                    out.append(f"{sid}: tool {n!r} has no `{k}`")
    return out


def save(cfg: dict) -> dict:
    """Write the configuration (the dashboard's writer). Refused -- nothing
    written -- when it does not validate. {ok, path, problems}."""
    problems = validate(cfg)
    if problems:
        return {"ok": False, "path": path(), "problems": problems}
    p = path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)
    os.replace(tmp, p)
    with _LOCK:
        _CACHE.update(mtime=None, path=None, cfg=None)
    return {"ok": True, "path": p, "problems": []}


def set_enabled(server_id: str, on: bool) -> dict:
    """Enable or disable one server (the dashboard's toggle). A disabled
    server is not offered to NEW conversations; a conversation that has its
    tools keeps them in its list (the tool list never changes mid-
    conversation) and a call gets MCP_SERVER_OFF."""
    cfg = load()
    hit = [s for s in cfg["servers"] if s.get("id") == server_id]
    if not hit:
        return {"ok": False, "path": path(),
                "problems": [f"no server {server_id!r}"]}
    hit[0]["enabled"] = bool(on)
    return save(cfg)


def servers(enabled_only: bool = False) -> list[dict]:
    return [s for s in load()["servers"]
            if s.get("enabled") or not enabled_only]


def server(server_id: str) -> dict | None:
    return next((s for s in load()["servers"] if s.get("id") == server_id),
                None)


def tool(name: str) -> tuple[dict, dict] | None:
    """(server, tool) for one of our MCP-backed tools, any server -- by its
    name or a name it had before (`legacy`)."""
    for s in load()["servers"]:
        for t in s.get("tools") or []:
            if t.get("name") == name or name in (t.get("legacy") or []):
                return s, t
    return None


def legacy_names() -> dict[str, str]:
    """{old name: current name} for every configured tool (the proxy reads
    an old name as its current one: LEGACY_TOOL_NAMES)."""
    return {old: t["name"] for s in load()["servers"]
            for t in s.get("tools") or [] for old in t.get("legacy") or []
            if t.get("name")}


def canonical(name: str) -> str:
    """A tool name as it is served now: an old name read as its current
    one, anything else as is."""
    return legacy_names().get(name, name)


def all_tool_names() -> set[str]:
    """Every yama_* name any configured server backs (enabled or not): what
    the proxy recognises as its own (proxy.OUR_NAMES). Current names only;
    the old ones are legacy_names()."""
    return {t["name"] for s in load()["servers"] for t in s.get("tools") or []
            if t.get("name")}


def definition(t: dict) -> dict:
    """The OpenAI function tool main is sent for one of ours."""
    return {"type": "function",
            "function": {"name": t["name"], "description": t["description"],
                         "parameters": t["parameters"]}}
