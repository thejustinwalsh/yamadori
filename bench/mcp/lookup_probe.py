#!/usr/bin/env python
"""THE LOOKUP PROBE: does the model call the package lookup (yama_find_package,
yama_list_package_versions, yama_read_package_readme, yama_resolve_packages;
mcp/mcp_host.py) before it installs a package, when the prompt names the
packages vaguely -- and does it install the versions the resolver returned?

    python bench/mcp/lookup_probe.py                  # print the plan; runs nothing
    python bench/mcp/lookup_probe.py --run --trials N --max-requests R --max-minutes M
                                     [--tag T] [--key-file F]

Operator, 2026-09-28: the proxy offers MCP tools zero-config so Bonsai can
DISCOVER packages ("pmndrs math" -> npm `math`) instead of guessing names;
"then we measure whether the model calls it". The coordinator runs the
trials after the proxy restart, with the operator's go and an idle GPU
(AGENTS.md "One GPU consumer at a time"). THIS SCRIPT RUNS NOTHING UNLESS
--run IS GIVEN.

ONE TRIAL (the pagoda's Pi path, bench/octopus/run.py run_pagoda_pi, reused):
  - the prompt: index/octopus/prompts/pagoda.md (bench/voxel's r3f-stack
    prompt, byte for byte): it names "r3f (react-three-fiber) v10 and Koota
    and pmndrs math" and no npm name;
  - Pi 0.87.1 in the harness box (bench/sandbox/harness_box.py `run pi`, its
    default loadout), a fresh project folder, `--thinking medium`;
  - every request through the recording relay (bench/octopus/relay.py) to
    :1234, so x_yamadori.mcp of each request is on record;
  - Pi's models.json = the host test's, with the proxy's advertised window
    (pagoda.pi_models) and ONE provider header, X-Yamadori-Features
    {"mcp_tools": true, "skills": false}: tier medium, skills off -- the
    model plus our MCP tools only (tiers.BEHAVIOURS `mcp_tools`);
  - STOPPED at the first package-install action Pi starts (a bash command
    matching INSTALL_RX, read from Pi's own JSON events,
    tool_execution_start), or after --max-requests requests through the
    relay, or --max-minutes: the harness container is removed by name, and
    harness_box takes its network down. The numbers are the coordinator's
    (no default: a bound nobody chose is not written here).

RECORDED per trial (bench/mcp/results/lookup_probe.jsonl): the stop and
why; the install command and the package names it names; every yama_* call
before it (x_yamadori.mcp.calls: tool, args, names returned, ok) and whether
there was any; the names the lookups returned; which install names a lookup
had returned; the offer as the proxy recorded it (switch source, tools on
main) -- a trial where the header did not reach the proxy, or the tools were
not on main, is marked `invalid` with why.
Since 2026-09-29 (yama_resolve_packages, mcp/npm_resolve.py): whether the
resolver ran before the install (`resolved_before_install`), the exact
versions it returned (`resolve_pins`, the last resolve's), the versions the
install command and the package.json files name (`install_specs`,
`manifest_specs`), and `versions_vs_resolved` {matched, differ, unpinned}:
each named package the resolver pinned, compared by its version with any
leading ^, ~ or = set aside. Rows before 2026-09-29 name the tools by their
old names (yama_package_versions, yama_package_readme).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OCTO_DIR = os.path.join(ROOT, "bench", "octopus")
sys.path.insert(0, OCTO_DIR)

RESULTS = os.path.join(HERE, "results", "lookup_probe.jsonl")
EFFORT = "medium"
# retrieval off too (operator, 2026-09-29: "time to execute"): strictly the model plus the MCP tools -- no held-package
# definitions injected once the conversation imports koota / math / r3f.
FEATURES = {"mcp_tools": True, "skills": False, "retrieval": False, "check_code": False}
HEADER = "X-Yamadori-Features"
TOOLS = ("yama_find_package", "yama_list_package_versions",
         "yama_read_package_readme", "yama_resolve_packages")
RESOLVE = "yama_resolve_packages"
# The first package-install action: a package manager fetching a package
# into the project (install/add/ci, or a create-* scaffold, which fetches its
# template package). Read from the command Pi starts, not from its result.
# A SCAFFOLD (npm create vite, npx create-*) is not the moment the probe measures: it fetches a template, not the
# task's packages (2026-09-29, bench/mcp/results/lookup_probe.jsonl trials 1-2 stopped on `npm create vite` and
# recorded `create-vite` as the "install"; docs/TOOL-FACTORY.md defect 1). SCAFFOLD_RX marks it so the probe keeps
# watching for the real install.
SCAFFOLD_RX = re.compile(r"\b(?:npm|pnpm|yarn|bun)\s+create\b|\bnpx\s+(?:-y\s+|--yes\s+)?create-")
INSTALL_RX = re.compile(
    r"\b(?:npm|pnpm|yarn|bun)\s+(?:install|i|add|ci)\b"
    r"|\b(?:npm|pnpm|yarn|bun)\s+create\b|\bnpx\s+(?:-y\s+|--yes\s+)?create-"
    r"|\bpip3?\s+install\b|\bcargo\s+add\b")
# How often the watcher reads Pi's event file and the relay's rows: a file
# tail's wake-up, not a measured quantity; it bounds how late a stop is
# seen (the command has started by then either way).
TAIL_S = 0.5


def _mods():
    import pagoda
    import run as R
    return R, pagoda


def trial_ids(tag: str, n: int) -> dict:
    R, _p = _mods()
    rid = f"{tag}-lookup-pi-{EFFORT}-{n}"
    log_dir = os.path.join(R.LOGS_DIR, rid)
    return {"run_id": rid, "run_dir": os.path.join(R.RUNS_DIR, rid),
            "log_dir": log_dir, "box_dir": os.path.join(log_dir, "box"),
            "models_out": os.path.join(log_dir, "pi-models.json"),
            "stdout": os.path.join(log_dir, "pi.jsonl"),
            "stderr": os.path.join(log_dir, "pi.err"),
            "relay_out": os.path.join(log_dir, "relay.jsonl")}


def with_header(cfg: dict) -> dict:
    """Pi's models.json with the one provider header (Pi 0.87.1: a
    provider's `headers`, dist/core/model-config.js ProviderConfigSchema)."""
    cfg = json.loads(json.dumps(cfg))
    for prov in (cfg.get("providers") or {}).values():
        prov.setdefault("headers", {})[HEADER] = json.dumps(FEATURES)
    return cfg


def box_cmd(ids: dict, key_file: str, max_minutes: float, prompt: str) -> list[str]:
    R, pagoda = _mods()
    return [sys.executable, os.path.join(ROOT, "bench", "sandbox", "harness_box.py"),
            "run", "pi", "--key-file", key_file, "--project", ids["run_dir"],
            "--run-dir", ids["box_dir"], "--target-port", str(R.RELAY_PORT),
            "--config", ids["models_out"], "--timeout", str(int(max_minutes * 60)),
            "--tag", ids["run_id"], "--", *pagoda.pi_args(EFFORT, prompt)]


def container_name(ids: dict) -> str:
    R, _p = _mods()
    hb = R._hb()
    return f"{hb.PREFIX}-pi-{hb.sandbox_net._safe(ids['run_id'])}"


def install_packages(cmd: str) -> list[str]:
    """The package names an install command names (flags and paths left
    out), from the first install segment."""
    m = INSTALL_RX.search(cmd or "")
    if not m:
        return []
    # the first command only: a newline ends it as `;` does
    seg = re.split(r"&&|\|\||;|\||\n", cmd[m.end():])[0]
    try:
        toks = shlex.split(seg)
    except ValueError:
        toks = seg.split()
    # shell redirections and descriptors are not packages (`2>&1` was read as one: trials 3-4, TOOL-FACTORY defect 1)
    toks = [t for t in toks if t and not t.startswith(("-", ".", "/", "~"))
            and not re.search(r"[<>&|]", t) and not t.isdigit()]
    # a flag's value is not a package (`--prefix app`, `--registry URL`)
    toks = _drop_flag_values(seg, toks)
    if "create" in m.group(0):
        # `npm create vite@latest app` fetches create-vite; `npx create-vite`
        # matched up to "create-". The rest are the scaffold's arguments.
        if not toks:
            return []
        first = toks[0]
        if m.group(0).endswith("create-"):
            return ["create-" + first]
        if first.startswith("@"):
            scope, _, rest = first.partition("/")
            return [f"{scope}/create" + (f"-{rest}" if rest else "")]
        return ["create-" + first]
    return toks


# npm flags that take a value (`npm help config`): the value is not a package.
_VALUE_FLAGS = ("--prefix", "-C", "--registry", "--cache", "--tag",
                "--workspace", "-w", "--userconfig", "--omit", "--include")


def _drop_flag_values(seg: str, toks: list[str]) -> list[str]:
    try:
        raw = shlex.split(seg)
    except ValueError:
        raw = seg.split()
    drop = {raw[i + 1] for i, t in enumerate(raw[:-1]) if t in _VALUE_FLAGS}
    return [t for t in toks if t not in drop]


def _spec_version(spec: str) -> str | None:
    """`@scope/name@^1.2` -> `^1.2`; `name` -> None."""
    at = spec.find("@", 1) if spec.startswith("@") else spec.find("@")
    return (spec[at + 1:] or None) if at > 0 else None


def versions_vs_resolved(specs: dict, pins: dict) -> dict:
    """{matched, differ [{name, named, resolved}], unpinned [names]} for the
    named packages the resolver pinned."""
    out: dict = {"matched": [], "differ": [], "unpinned": []}
    for name, v in sorted(specs.items()):
        if name not in pins:
            continue
        if not v:
            out["unpinned"].append(name)
        elif v.lstrip("^~=") == pins[name]:
            out["matched"].append(name)
        else:
            out["differ"].append({"name": name, "named": v,
                                  "resolved": pins[name]})
    return out


def manifest_specs(project: str) -> dict:
    """{package.json path relative to the project: {name: the version spec
    it names}} for every package.json the model wrote (node_modules left
    out)."""
    out = {}
    for dp, dn, fn in os.walk(project):
        dn[:] = [d for d in dn if d != "node_modules"]
        if "package.json" in fn:
            p = os.path.join(dp, "package.json")
            try:
                d = json.load(open(p, encoding="utf-8"))
            except (OSError, ValueError):
                continue
            specs = {}
            for k in ("dependencies", "devDependencies", "peerDependencies"):
                blk = (d.get(k) or {}) if isinstance(d, dict) else {}
                if isinstance(blk, dict):
                    specs.update({str(n): str(v) for n, v in blk.items()})
            out[os.path.relpath(p, project).replace("\\", "/")] = specs
    return out


def manifest_names(project: str) -> dict:
    """{package.json path relative to the project: the dependency names it
    lists} for every package.json the model wrote (node_modules left out)."""
    out = {}
    for dp, dn, fn in os.walk(project):
        dn[:] = [d for d in dn if d != "node_modules"]
        if "package.json" in fn:
            p = os.path.join(dp, "package.json")
            try:
                d = json.load(open(p, encoding="utf-8"))
            except (OSError, ValueError):
                continue
            names = []
            for k in ("dependencies", "devDependencies", "peerDependencies"):
                names += list((d.get(k) or {}) if isinstance(d, dict) else [])
            out[os.path.relpath(p, project).replace("\\", "/")] = sorted(set(names))
    return out


def _bare(spec: str) -> str:
    """`@scope/name@1.2` / `name@alpha` -> the package name."""
    if spec.startswith("@"):
        at = spec.find("@", 1)
        return spec if at < 0 else spec[:at]
    return spec.split("@", 1)[0]


def _rows(path: str) -> list[dict]:
    out = []
    try:
        for ln in open(path, encoding="utf-8", errors="replace"):
            try:
                out.append(json.loads(ln))
            except ValueError:
                continue
    except OSError:
        pass
    return out


def _gens(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("method") == "POST"
            and str(r.get("path", "")).endswith(("/chat/completions", "/responses"))]


class Watch:
    """Tails Pi's events and the relay's rows; stops the trial at the first
    install action, the request bound or the time bound."""

    def __init__(self, ids: dict, max_requests: int, max_minutes: float):
        self.ids, self.max_requests, self.max_s = ids, max_requests, max_minutes * 60
        self.stop_reason: str | None = None
        self.install: dict | None = None
        self.t0 = time.time()
        self._done = threading.Event()

    def _pi_events(self):
        for ev in _rows(self.ids["stdout"]):
            if ev.get("type") == "tool_execution_start":
                yield ev

    def check(self) -> bool:
        for ev in self._pi_events():
            cmd = ((ev.get("args") or {}).get("command")
                   if ev.get("toolName") == "bash" else None)
            if cmd and INSTALL_RX.search(cmd) and not (SCAFFOLD_RX.search(cmd) and not re.search(
                    r"\b(?:npm|pnpm|yarn|bun)\s+(?:install|i|add|ci)\b|\bpip3?\s+install\b|\bcargo\s+add\b", cmd)):
                self.install = {"command": cmd[:2000], "t_seen": time.time(),
                                "tool_call_id": ev.get("toolCallId"),
                                "packages": install_packages(cmd)}
                self.stop_reason = "install"
                return True
        if len(_gens(_rows(self.ids["relay_out"]))) >= self.max_requests:
            self.stop_reason = "max_requests"
            return True
        if time.time() - self.t0 >= self.max_s:
            self.stop_reason = "max_minutes"
            return True
        return False

    def loop(self, proc: subprocess.Popen) -> None:
        while proc.poll() is None and not self._done.is_set():
            if self.check():
                subprocess.run(["docker", "rm", "-f", container_name(self.ids)],
                               capture_output=True, timeout=120)
                return
            time.sleep(TAIL_S)
        if self.stop_reason is None:
            self.stop_reason = "exited"


def summarise(ids: dict, watch: Watch) -> dict:
    rows = _gens(_rows(ids["relay_out"]))
    t_install = (watch.install or {}).get("t_seen")
    first = ((rows[0].get("response") or {}).get("x_yamadori") or {}) if rows else {}
    mcp0 = first.get("mcp") or {}
    calls, found, pins = [], [], {}
    for r in rows:
        if t_install and (r.get("t_end") or r.get("t0") or 0) > t_install:
            break
        x = (r.get("response") or {}).get("x_yamadori") or {}
        for c in (x.get("mcp") or {}).get("calls") or []:
            calls.append({k: c.get(k) for k in ("tool", "called_as", "args",
                                                "names", "ok", "error", "ms",
                                                "pins", "fixes", "conflicts")
                          if k in c})
            if c.get("ok"):
                found += list(c.get("names") or [])
            if c.get("tool") == RESOLVE and c.get("ok") and c.get("pins"):
                pins = dict(c["pins"])
    inst_specs = {_bare(p): _spec_version(p)
                  for p in (watch.install or {}).get("packages") or []}
    inst = list(inst_specs)
    manifests = manifest_names(ids["run_dir"])
    mspecs = manifest_specs(ids["run_dir"])
    if watch.install and not inst:
        # A bare `npm install`: the names are the ones the model wrote into
        # its package.json before it.
        inst = sorted({n for names in manifests.values() for n in names})
    invalid = []
    if not rows:
        invalid.append("no request reached the relay")
    else:
        if (mcp0.get("switch") or {}).get("source") != "header":
            invalid.append(f"the header did not reach the proxy (switch "
                           f"{mcp0.get('switch')})")
        if not set(TOOLS) <= set(mcp0.get("on_main") or []):
            invalid.append(f"the tools were not all on main "
                           f"({mcp0.get('on_main')}; why {mcp0.get('why')})")
        if first.get("tier") != EFFORT:
            invalid.append(f"tier {first.get('tier')}, not {EFFORT}")
    return {"requests": len(rows),
            "requests_before_install": sum(
                1 for r in rows if not t_install
                or (r.get("t_end") or r.get("t0") or 0) <= t_install),
            "offer": {k: mcp0.get(k) for k in ("switch", "offered", "on_main",
                                                "why")},
            "tier": first.get("tier"),
            "skills_on": bool((first.get("skills") or {}).get("on")),
            "stop": watch.stop_reason,
            "install": watch.install,
            "install_names": inst,
            "manifests": manifests,
            "called_before_install": bool(calls),
            "calls_before_install": calls,
            "found_names": sorted(set(found)),
            "install_names_found": sorted(set(inst) & set(found)),
            "install_names_not_found": sorted(set(inst) - set(found)),
            "resolved_before_install": any(c.get("tool") == RESOLVE
                                           and c.get("ok") for c in calls),
            "resolve_pins": pins,
            "install_specs": inst_specs,
            "manifest_specs": mspecs,
            "versions_vs_resolved": {
                "install": versions_vs_resolved(inst_specs, pins),
                "manifests": {k: versions_vs_resolved(v, pins)
                              for k, v in mspecs.items()}},
            "invalid": invalid or None}


def run_trial(n: int, a) -> dict:
    R, pagoda = _mods()
    ids = trial_ids(a.tag, n)
    key_file = a.key_file or R.PI_KEY_FILE
    pf = R.pi_preflight({}, key_file)
    base = {"run_id": ids["run_id"], "trial": n, "tag": a.tag, "effort": EFFORT,
            "features": FEATURES, "harness": "pi", "t": time.time(),
            "max_requests": a.max_requests, "max_minutes": a.max_minutes}
    if not pf["ok"]:
        return dict(base, outcome="not_run", why="preflight", preflight=pf)
    for d in (ids["run_dir"], ids["log_dir"]):
        if os.path.exists(d):
            raise SystemExit(f"refusing: {d} exists (a fresh folder per trial)")
    os.makedirs(ids["run_dir"])
    os.makedirs(ids["log_dir"])
    prompt, sha = pagoda.prompt()
    if R._port_open(R.RELAY_PORT):
        raise SystemExit(f"port {R.RELAY_PORT} is taken: another relay is running")
    relay = subprocess.Popen(R.relay_cmd(ids["relay_out"]),
                             stdout=open(os.path.join(ids["log_dir"], "relay.out"), "w"),
                             stderr=subprocess.STDOUT)
    try:
        for _ in range(40):
            if R._port_open(R.RELAY_PORT):
                break
            time.sleep(0.25)
        else:
            return dict(base, outcome="not_run", why="relay did not start")
        key = open(key_file, encoding="utf-8").read().strip()
        card = pagoda.advertised_card(R.RELAY_PORT, key)
        src = json.load(open(R._hb().HARNESSES["pi"]["config_src"], encoding="utf-8"))
        cfg, card_rec = pagoda.pi_models(src, card)
        with open(ids["models_out"], "w", encoding="utf-8", newline="\n") as f:
            json.dump(with_header(cfg), f, indent=2)
        cmd = box_cmd(ids, key_file, a.max_minutes, prompt)
        watch = Watch(ids, a.max_requests, a.max_minutes)
        t0 = time.time()
        with open(ids["stdout"], "wb") as out, open(ids["stderr"], "wb") as err:
            proc = subprocess.Popen(cmd, stdout=out, stderr=err,
                                    stdin=subprocess.DEVNULL)
            watch.loop(proc)
            try:
                rc = proc.wait(timeout=a.max_minutes * 60 + 900)
            except subprocess.TimeoutExpired:
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                               capture_output=True)
                rc = proc.wait()
        t1 = time.time()
    finally:
        relay.terminate()
        try:
            relay.wait(timeout=10)
        except subprocess.TimeoutExpired:
            relay.kill()
    s = summarise(ids, watch)
    return dict(base, outcome="invalid" if s["invalid"] else "ok",
                prompt_sha256=sha, model_card=card_rec, exit=rc,
                wall_s=round(t1 - t0, 1), preflight=pf, **s)


def plan(a) -> str:
    R, pagoda = _mods()
    ids = trial_ids(a.tag, 1)
    key_file = a.key_file or R.PI_KEY_FILE
    prompt, sha = pagoda.prompt()
    mm = a.max_minutes if a.max_minutes is not None else "<--max-minutes>"
    cmd = box_cmd(ids, key_file, a.max_minutes or 0, prompt)
    cmd = cmd[:cmd.index("--") + 1] + pagoda.pi_args(EFFORT, "<the prompt>")
    if a.max_minutes is None:
        cmd[cmd.index("--timeout") + 1] = "<--max-minutes x 60>"
    return "\n".join([
        "THE LOOKUP PROBE -- the plan (nothing was run, written or started)",
        "",
        f"prompt: {pagoda.PROMPT_PATH} ({len(prompt.encode('utf-8'))} bytes, sha256 {sha})",
        f"  {prompt}",
        "",
        f"per trial n (1..--trials), trial 1 shown: run id {ids['run_id']}",
        f"  project (/work, Pi's cwd): {ids['run_dir']}",
        f"  logs: {ids['log_dir']} (pi.jsonl = Pi's events, relay.jsonl = one row per request "
        "with x_yamadori)",
        "",
        "1. preflight (bench/octopus/run.py pi_preflight): no other GPU consumer, every slot idle, "
        "the proxy answers, Docker answers, the harness box image, the key file",
        "2. the recording relay: " + " ".join(R.relay_cmd(ids["relay_out"])),
        f"3. Pi's models.json -> {ids['models_out']}: the host test's "
        f"({R._hb().HARNESSES['pi']['config_src']}), the window from GET /v1/models through the "
        f"relay, and the provider header\n     {HEADER}: {json.dumps(FEATURES)}",
        f"4. the harness box (Pi 0.87.1, default loadout, --thinking {EFFORT}):",
        "     " + " ".join(cmd),
        "5. the stop, whichever comes first:",
        "     - Pi starts a bash command matching INSTALL_RX (its JSON event "
        "tool_execution_start):",
        f"       {INSTALL_RX.pattern}",
        f"     - {a.max_requests if a.max_requests is not None else '<--max-requests>'} requests "
        f"through the relay; {mm} minutes (also harness_box's --timeout)",
        f"     then: docker rm -f {container_name(ids)} (harness_box takes its network down)",
        f"6. one row appended to {RESULTS}:",
        "     stop, install {command, packages}, install_names, called_before_install, "
        "calls_before_install [{tool, args, names, ok, pins}], found_names, "
        "install_names_found / _not_found, resolved_before_install, resolve_pins, "
        "install_specs, manifest_specs, versions_vs_resolved, offer {switch, offered, "
        "on_main}, tier, skills_on, requests, invalid",
        "",
        "BEFORE RUNNING (the coordinator):",
        "  - the proxy restarted with this code; GET /dash/api/mcp shows packagelens `ready`",
        "  - an idle GPU and the operator's go",
        f"  - then: python bench/mcp/lookup_probe.py --run --trials N --max-requests R "
        f"--max-minutes M --tag {a.tag}",
    ])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", action="store_true", help="run the trials (default: print the plan)")
    ap.add_argument("--trials", type=int)
    ap.add_argument("--max-requests", type=int)
    ap.add_argument("--max-minutes", type=float)
    ap.add_argument("--tag", default="probe")
    ap.add_argument("--key-file", help="the key Pi sends (default: the pi-dogfood test key)")
    a = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    if not a.run:
        print(plan(a))
        return 0
    missing = [f"--{k.replace('_', '-')}" for k in ("trials", "max_requests", "max_minutes")
               if getattr(a, k) is None]
    if missing:
        print(f"--run needs {', '.join(missing)} (no defaults: the bounds are the "
              f"coordinator's)", file=sys.stderr)
        return 2
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    for n in range(1, a.trials + 1):
        row = run_trial(n, a)
        with open(RESULTS, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps({k: row.get(k) for k in (
            "run_id", "outcome", "why", "stop", "called_before_install", "install_names",
            "install_names_found", "found_names", "resolved_before_install",
            "versions_vs_resolved", "invalid")}, indent=1), flush=True)
        if row.get("outcome") == "not_run":
            return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
