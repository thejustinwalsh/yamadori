#!/usr/bin/env python
"""yama_resolve_packages: npm's OWN resolver on every package the model plans
to use, together -- one install line with exact versions, the peers they need
and their type packages, or the conflict and the versions that fit.

Operator, 2026-09-29: "packagelens should be able to resolve peer dep ranges,
and give advice for other dependency ranges that a core package needs, if the
tool was called with all names the agent was interested in, it could de-dupe
version ranges, and recommend not only the core package names but the
dependency exact names and version too ... All mechanically from package json
info." Evidence: bench/mcp/results/lookup_probe.jsonl probe2-* -- 5/5 trials
called a lookup before installing and named the right packages, but chose
versions one package at a time (three@0.170 beside @react-three/fiber 10's
`three >=0.185`; @types/react@^18 with React 19; a math nightly canary instead
of 0.1.0); trial 4 got react 19.2.2 / three 0.186.1 right from the peers line.

NOT A RE-IMPLEMENTATION. The resolving is npm's: `npm install
--package-lock-only --ignore-scripts --no-audit --no-fund` (npm >= 7 installs
peers and answers ERESOLVE on a conflict), run by
mcp_servers/packagelens/resolve_driver.js -- read that file's header for the
steps -- with npm's own npm-package-arg and semver. WHERE: a throwaway
container of the PackageLens image (mcp_config.PACKAGELENS `image`, the id
checked against models/manifest.yaml when its server starts): node 24.21.0
and its bundled npm 11.19.0, nothing downloaded to build it. On the
PackageLens server's gated network (its one way out is the egress gate:
global addresses only), uid 1000, every capability dropped,
no-new-privileges, a read-only root with a tmpfs /tmp (npm's cache and the
work directory, gone with the container), no volume, no credential. No
package code runs (--ignore-scripts; lock-only: no tarball is unpacked).

THE BOUND. npm bounds each of its own fetches: npm 11.19.0's defaults, read
in the image with `npm config get` on 2026-09-29 -- fetch-timeout 300000 ms,
fetch-retries 2, fetch-retry-mintimeout 10000 ms, fetch-retry-maxtimeout
60000 ms, fetch-retry-factor 10 -- so one fetch that keeps failing ends after
3 x 300 s + 10 s + min(100 s, 60 s) = 970 s (FETCH_WORST_S). The driver's own
registry GETs use the same fetch-timeout. The container is killed after that
same 970 s: a backstop for a hung Docker, not a bound on a working resolve.
Measured on the live smoke through mcp_host.run_tool (2026-09-29, n=1 per
set): koota / math / @react-three/fiber@^10 / three / react with
allow_prerelease 5.4 s (3 npm runs, 3.7 s of them); the same with
@react-three/fiber@alpha and react-dom 6.3 s (6 runs: two moves); a conflict
1.7 s; refused specs 0.7 s (no npm run).

x_yamadori.mcp.calls[] for this tool (mcp_host.run_tool): {tool, server,
runner, ms, ok, bytes, args, packages (the specs asked, 120 chars each),
allow_prerelease, resolved (packages in the install lines), pins {name:
version}, fixes, conflicts, errors [codes], npm {version, runs}, screen,
names}.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DRIVER = os.path.join(ROOT, "mcp_servers", "packagelens", "resolve_driver.js")
TOOL_NAME = "yama_resolve_packages"
RUNNER = "npm_resolve"
RESULT_MARK = "YAMA_RESOLVE_RESULT "

# npm 11.19.0's fetch defaults (the module docstring, THE BOUND).
NPM_FETCH_TIMEOUT_MS = 300_000
NPM_FETCH_RETRIES = 2
NPM_RETRY_MIN_S = 10.0
NPM_RETRY_MAX_S = 60.0
NPM_RETRY_FACTOR = 10


def fetch_worst_s() -> float:
    """One npm fetch that keeps failing: every attempt times out, with the
    backoff between attempts (retry's min(min x factor^k, max))."""
    waits = sum(min(NPM_RETRY_MIN_S * NPM_RETRY_FACTOR ** k, NPM_RETRY_MAX_S)
                for k in range(NPM_FETCH_RETRIES))
    return (NPM_FETCH_RETRIES + 1) * NPM_FETCH_TIMEOUT_MS / 1000 + waits


FETCH_WORST_S = fetch_worst_s()
CALL_TIMEOUT_WHY = (
    "npm 11.19.0's own bound on one fetch (fetch-timeout 300 s, 2 retries, "
    "backoff 10 s then 60 s: 970 s); the container is killed after it -- a "
    "backstop for a hung Docker, a working resolve takes seconds")

# The container's name and label share the MCP host's prefix.
PREFIX = "yama-mcp"


class ResolveError(Exception):
    def __init__(self, code: str, message: str, retryable: bool):
        super().__init__(message)
        self.code, self.retryable = code, retryable


def check_args(args: dict) -> tuple[list[str], bool] | str:
    """(the package specs, allow_prerelease), or why the arguments are
    refused. The specs are parsed by npm's own npm-package-arg in the driver;
    here only the shape."""
    a = args or {}
    eco = a.get("ecosystem")
    if eco != "npm":
        return (f"ecosystem is {json.dumps(eco)}; this tool resolves npm "
                f"packages only (ecosystem \"npm\")")
    pk = a.get("packages")
    if isinstance(pk, str):
        # A model that sends one string for the list: split on whitespace
        # and commas, as it would be typed on an npm command line.
        pk = [x for x in re.split(r"[\s,]+", pk) if x]
    if not isinstance(pk, list) or not pk:
        return "packages must be a list of package names, each optionally with @version, @range or @dist-tag"
    specs = []
    for x in pk:
        if not isinstance(x, str) or not x.strip():
            return f"packages holds {json.dumps(x)[:80]}, not a package name"
        specs.append(x.strip())
    ap = a.get("allow_prerelease")
    if ap not in (None, True, False):
        if isinstance(ap, str) and ap.strip().lower() in ("true", "false"):
            ap = ap.strip().lower() == "true"
        else:
            return "allow_prerelease is true or false"
    return specs, bool(ap)


def driver_input(specs: list[str], allow_prerelease: bool) -> str:
    import package_net
    return json.dumps({"packages": specs, "allow_prerelease": allow_prerelease,
                       "registry": package_net.REGISTRY,
                       "fetch_timeout_ms": NPM_FETCH_TIMEOUT_MS},
                      separators=(",", ":"))


def container_name(tag: str) -> str:
    return f"{PREFIX}-resolve-{tag}-{uuid.uuid4().hex[:8]}"


def run_argv(image: str, network: str | None, name: str, input_json: str,
             env_args: list[str] | None = None) -> list[str]:
    """The throwaway container: the image's node reads the driver from
    stdin (`node -`); the request is one environment variable."""
    argv = ["docker", "run", "--rm", "-i", "--init", "--name", name,
            "--label", f"{PREFIX}=resolve"]
    if network:
        argv += ["--network", network] + list(env_args or []) + [
            "-e", "NODE_USE_ENV_PROXY=1"]
    else:
        argv += ["--network", "none"]
    argv += ["-e", "npm_config_cache=/tmp/.npm",
             "-e", "npm_config_update_notifier=false",
             "-e", f"YAMA_RESOLVE_INPUT={input_json}",
             "--user", "1000:1000", "--cap-drop", "ALL",
             "--security-opt", "no-new-privileges", "--read-only",
             "--tmpfs", "/tmp", "--workdir", "/tmp",
             "--entrypoint", "node", image, "-"]
    return argv


def docker_runner(argv: list[str], stdin_text: str, timeout_s: float,
                  name: str) -> tuple[int, str, str]:
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, creationflags=flags)
    try:
        out, err = p.communicate(stdin_text.encode("utf-8"), timeout=timeout_s)
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True,
                       timeout=60)
        p.kill()
        p.communicate()
        raise ResolveError("MCP_TIMEOUT", f"the resolve did not finish within "
                                          f"{timeout_s:g} s", True)
    return (p.returncode, out.decode("utf-8", "replace"),
            err.decode("utf-8", "replace"))


# The offline suites replace this (a fake container runner).
RUNNER_FN = docker_runner


def resolve(specs: list[str], allow_prerelease: bool, image: str,
            network: str | None, tag: str,
            env_args: list[str] | None = None,
            timeout_s: float = FETCH_WORST_S) -> dict:
    """The driver's result (its JSON), or ResolveError."""
    with open(DRIVER, encoding="utf-8") as f:
        src = f.read()
    name = container_name(tag)
    argv = run_argv(image, network, name,
                    driver_input(specs, allow_prerelease), env_args)
    try:
        rc, out, err = RUNNER_FN(argv, src, timeout_s, name)
    except OSError as e:
        raise ResolveError("MCP_SERVER_DOWN", f"docker could not be run "
                                              f"({type(e).__name__}: {e})",
                           True) from e
    line = next((ln for ln in reversed(out.splitlines())
                 if ln.startswith(RESULT_MARK)), None)
    if line is None:
        tail = (err or out or "").strip().splitlines()[-3:]
        raise ResolveError("RESOLVE_FAILED", f"the resolver container exited "
                                             f"{rc} with no result: "
                                             f"{' | '.join(tail)[:300]}", True)
    try:
        d = json.loads(line[len(RESULT_MARK):])
    except ValueError as e:
        raise ResolveError("RESOLVE_FAILED", f"the resolver's result is not "
                                             f"JSON ({e})", True) from e
    if not isinstance(d, dict):
        raise ResolveError("RESOLVE_FAILED", "the resolver's result is not "
                                             "an object", True)
    return d


# ------------------------------------------------------------------ the text

_REMEDY = {
    "NOT_A_PACKAGE_NAME": "call yama_find_package with those words, then pass the exact name it returns",
    "NOT_FOUND": "call yama_find_package with words for it, then pass the exact name it returns",
    "NOT_A_REGISTRY_SPEC": "pass a registry name, optionally with @version, @range or @dist-tag",
    "ONLY_PRERELEASES": "call it again with allow_prerelease true, or name one dist-tag (name@alpha)",
    "NO_SUCH_TAG": "name one of its dist-tags, or leave the tag out",
    "NO_SUCH_VERSION": "name a version it has, a range, or a dist-tag",
    "NO_MATCHING_VERSION": "name a range its versions satisfy, or a dist-tag",
    "DUPLICATE": "name each package once",
    "REGISTRY_UNREACHABLE": "call it again",
    "ETARGET": "name a version, range or dist-tag the package has (yama_list_package_versions lists them)",
    "E404": "call yama_find_package with words for it, then pass the exact name it returns",
}


def _set_names(d: dict) -> set[str]:
    return ({r.get("name") for r in d.get("requested") or []}
            | {p.get("name") for p in d.get("packages") or []})


def _needs(by: list[dict], inset: set[str]) -> str:
    """`@react-three/fiber >=19.0 <19.4, koota >=18.0.0 (optional)`, the set's
    own packages first, the rest of the tree counted."""
    mine = [b for b in by if b.get("name") in inset]
    rest = [b for b in by if b.get("name") not in inset]
    parts = [f"{b['name']} {b['range']}" + (" (optional)" if b.get("optional") else "")
             for b in mine]
    if rest:
        parts.append(f"{len(rest)} more package{'s' if len(rest) != 1 else ''} "
                     f"in the tree")
    return ", ".join(parts)


def render(d: dict, args: dict) -> tuple[str, list[str]]:
    """The model's text, the best answer first: the install lines, then each
    package and why it is at that version, then what could not be resolved."""
    lines: list[str] = []
    names: list[str] = []
    req = [r for r in d.get("requested") or [] if r.get("version")]
    peers = d.get("packages") or []
    types = d.get("types") or []
    inset = _set_names(d)
    npm = f"npm {d.get('npm_version')}" if d.get("npm_version") else "npm"
    if d.get("ok"):
        lines.append(
            f"Resolved by {npm}'s own resolver (npm install --package-lock-only, "
            f"no install scripts run): {len(req)} package"
            f"{'s' if len(req) != 1 else ''} asked for"
            + (f", {len(peers)} peer{'s' if len(peers) != 1 else ''} they need"
               if peers else "")
            + f"; {d.get('lock_packages')} packages in the whole tree. These "
              f"versions install together:")
        lines.append("")
        lines.append(f"  {d['install']}")
        if d.get("install_dev"):
            lines.append(f"  {d['install_dev']}")
        lines.append("")
        dep = {r["name"]: r["version"] for r in req}
        dep.update({p["name"]: p["version"] for p in peers})
        block = {"dependencies": dep}
        if types:
            block["devDependencies"] = {t["name"]: t["version"] for t in types}
        lines.append("package.json:")
        lines.append(json.dumps(block, indent=2))
        lines.append("")
        lines.append("Each package:")
        for r in req:
            names.append(r["name"])
            bits = [f"asked {r['spec']}"]
            if r.get("prerelease") and r.get("how", "").startswith("a prerelease"):
                bits.append(r["how"])
            if r.get("channels"):
                bits.append("other channels in range: " + ", ".join(
                    f"{c['tag']} {c['version']}" for c in r["channels"]))
            need = _needs(r.get("needed_by") or [], inset)
            if need:
                bits.append(f"required by {need}")
            lines.append(f"- {r['name']} {r['version']} ({'; '.join(bits)})")
        for p in peers:
            names.append(p["name"])
            need = _needs(p.get("needed_by") or [], inset)
            lines.append(f"- {p['name']} {p['version']} (a peer dependency"
                         + (f": required by {need}" if need else "") + ")")
        for t in types:
            names.append(t["name"])
            if t.get("for"):
                lines.append(f"- {t['name']} {t['version']} (types for "
                             f"{t['for']} {t['for_version']}, which ships none; "
                             f"DefinitelyTyped's version matches the library's "
                             + ("major.minor" if t.get("match") == "major.minor"
                                else "major") + ")")
            else:
                need = _needs(t.get("needed_by") or [], inset | {x["name"] for x in types})
                lines.append(f"- {t['name']} {t['version']} (a peer of the type "
                             f"packages" + (f": {need}" if need else "") + ")")
    fixes = d.get("fixes") or []
    if fixes:
        lines.append("")
        lines.append("Moved to fit together:")
        for f in fixes:
            lines.append(f"- {f['name']} {f.get('from') or '?'} -> {f['to']}: "
                         f"{f['because']}")
    opt = d.get("optional_peers") or []
    if opt:
        lines.append("")
        lines.append("Optional peers npm did not install (name one in the "
                     "packages to get its exact version):")
        for o in opt:
            need = _needs(o.get("needed_by") or [], inset)
            lines.append(f"- {o['name']}: {need}")
    miss = d.get("types_missing") or []
    if miss:
        lines.append("")
        lines.append("Type definitions not found:")
        for m in miss:
            what = " ".join(x for x in (m.get("for"), m.get("version")) if x)
            lines.append(f"- {what}: {m['why']}")
    for c in d.get("conflicts") or []:
        lines.append("")
        e, f, fo = c.get("edge") or {}, c.get("from") or {}, c.get("found") or {}
        lines.append(f"These packages do not install together; npm's resolver "
                     f"reports: {c.get('summary') or 'a conflict'}.")
        if e and f and fo:
            who = "the project" if f.get("root") else f"{f.get('name')}@{f.get('version')}"
            lines.append(f"- {who} needs {e.get('name')} {e.get('range')}; "
                         f"the set has {fo.get('name')} {fo.get('version')}"
                         + (f" (asked: {', '.join(c.get('asked') or [])})"
                            if c.get("asked") else "") + ".")
            if c.get("peer_fits"):
                lines.append(f"- {e.get('name')} {c['peer_fits']} is the newest "
                             f"{e.get('name')} that fits {e.get('range')}.")
            if c.get("dependent_fits") and not f.get("root"):
                lines.append(f"- {f.get('name')} {c['dependent_fits']} is the newest "
                             f"{f.get('name')} whose peer range admits "
                             f"{fo.get('name')} {fo.get('version')}.")
            lines.append("Call it again with one of those versions, or with a "
                         "range in place of an exact version.")
    errs = d.get("errors") or []
    if errs:
        lines.append("")
        lines.append("Not resolved:" if d.get("ok") else "Could not resolve:")
        for x in errs:
            spec = x.get("spec") or "the set"
            rem = _REMEDY.get(x.get("code"), "")
            why = str(x.get("why") or x.get("code")).rstrip(". ")
            lines.append(f"- {spec}: {why}"
                         + (f"; {rem}" if rem else ""))
    if not lines:
        lines.append("Nothing was resolved: no package was given.")
    return "\n".join(lines).strip("\n"), names


def record(d: dict, specs: list[str], allow_prerelease: bool,
           arg_chars: int) -> dict:
    """The x_yamadori fields of one call (no free text)."""
    pins = {r["name"]: r["version"] for r in d.get("requested") or []
            if r.get("version")}
    pins.update({p["name"]: p["version"] for p in d.get("packages") or []})
    pins.update({t["name"]: t["version"] for t in d.get("types") or []})
    return {"runner": RUNNER,
            "packages": [s[:arg_chars] for s in specs],
            "allow_prerelease": allow_prerelease,
            "resolved": len(pins) if d.get("ok") else 0,
            "pins": pins if d.get("ok") else {},
            "fixes": len(d.get("fixes") or []),
            "conflicts": len(d.get("conflicts") or []),
            "errors": sorted({str(x.get("code")) for x in d.get("errors") or []}),
            "verified": bool(d.get("verified")),
            "npm": {"version": d.get("npm_version"),
                    "runs": len(d.get("runs") or []),
                    "ms": sum(int(r.get("ms") or 0) for r in d.get("runs") or [])}}
