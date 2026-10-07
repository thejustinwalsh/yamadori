#!/usr/bin/env python
"""The harness box (harness_box.py, harness/, and the shared sandbox network
it uses; SELF-IMPROVEMENT-LOG #49), offline: the configs it writes (base URL
at the one forward, the key only as an env reference, a literal key never
copied), the docker argv (no key value, no publish, one network, no
privileges), the gate's single outward forward, the fail-closed paths, the
build's refusal to pull, and the image's pins (Dockerfile + lockfile).
No docker, no network, no model. The live check is
`python bench/sandbox/harness_box.py verify` (Docker, no model).

    python bench/sandbox/test_harness_box.py      -> "N/M checks passed"
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import tomllib
import traceback
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# argv checks never depend on this machine's docker context (run_tests.py sets the same via its
# offline guard); test_wsl_engine forces the other engine itself
os.environ.setdefault("YAMADORI_DOCKER_PATHS", "native")
import harness_box as hb  # noqa: E402
import sandbox_net as sn  # noqa: E402

_results: list[tuple[bool, str, str]] = []
SECRET = "sk-yamadori-TEST-0123456789abcdef"


def check(ok: bool, name: str, detail: str = "") -> None:
    _results.append((bool(ok), name, detail))


OPENCODE_SRC = {
    "$schema": "https://opencode.ai/config.json", "autoupdate": True,
    "model": "yamadori/yamadori",
    "permission": {"edit": "allow", "bash": "allow", "webfetch": "deny"},
    "provider": {"yamadori": {"npm": "@ai-sdk/openai-compatible", "name": "Yamadori",
                              "options": {"baseURL": "http://127.0.0.1:18235/v1", "apiKey": SECRET,
                                          "setCacheKey": True,
                                          "headers": {"Authorization": "Bearer " + SECRET,
                                                      "X-Title": "t"}},
                              "models": {"yamadori": {"name": "Yamadori"}}}}}
PI_SRC = {"providers": {"yamadori": {"baseUrl": "http://127.0.0.1:18236/v1",
                                     "api": "openai-completions", "apiKey": SECRET,
                                     "models": [{"id": "yamadori"}]}}}
CODEX_SRC = "\n".join([
    "# comment kept", 'model = "yamadori"', 'model_provider = "yamadori"',
    r"model_catalog_json = 'C:\Users\x\codex-home\yamadori-catalog.json'",
    'sandbox_mode = "workspace-write"', "", "[windows]", 'sandbox = "unelevated"', "",
    "[model_providers.yamadori]", 'name = "Yamadori"', 'base_url = "http://127.0.0.1:1234/v1"',
    'env_key = "SOMETHING_ELSE"', f'experimental_bearer_token = "{SECRET}"',
    'wire_api = "responses"', ""])


def test_configs() -> None:
    url = hb.base_url(18235)
    check(url == "http://egress:18235/v1", "the base URL is the gate's forward", url)
    oc = hb.opencode_config(OPENCODE_SRC, url, "YAMADORI_OPENCODE_KEY")
    o = oc["provider"]["yamadori"]["options"]
    check(o["baseURL"] == url and o["apiKey"] == "{env:YAMADORI_OPENCODE_KEY}"
          and o.get("setCacheKey") is True and "Authorization" not in o["headers"]
          and o["headers"].get("X-Title") == "t" and SECRET not in json.dumps(oc),
          "opencode: forward URL, {env:VAR} key, auth header dropped, the rest kept", json.dumps(oc))
    check(oc["autoupdate"] is False and oc["share"] == "disabled"
          and oc["permission"]["webfetch"] == "deny"
          and OPENCODE_SRC["provider"]["yamadori"]["options"]["apiKey"] == SECRET,
          "opencode: autoupdate off, share off, permissions kept; the source is not mutated")
    pc = hb.pi_config(PI_SRC, url, "YAMADORI_PI_KEY")
    p = pc["providers"]["yamadori"]
    check(p["baseUrl"] == url and p["apiKey"] == "$YAMADORI_PI_KEY" and SECRET not in json.dumps(pc),
          "pi: forward URL, $VAR key (never a literal or a !command)", json.dumps(pc))
    cx = hb.codex_config(CODEX_SRC, url, "YAMADORI_CODEX_KEY", "/home/node/.codex/cat.json")
    t = tomllib.loads(cx)
    prov = t["model_providers"]["yamadori"]
    check(prov["base_url"] == url and prov["env_key"] == "YAMADORI_CODEX_KEY"
          and "experimental_bearer_token" not in prov and SECRET not in cx
          and t["model_catalog_json"] == "/home/node/.codex/cat.json"
          and t["sandbox_mode"] == "workspace-write" and "# comment kept" in cx,
          "codex: forward URL, env_key, token line dropped, catalog at its container path", cx)
    check(t["features"]["shell_snapshot"] is False
          and "YAMADORI_CODEX_KEY" in t["shell_environment_policy"]["exclude"],
          "codex: no shell snapshot (it held the key) and the key excluded from the model's shell",
          cx)
    src2 = CODEX_SRC + "\n[features]\nmulti_agent = false\nshell_snapshot = true\n\n" \
        "[shell_environment_policy]\ninherit = \"all\"\nexclude = [\"X\"]\n"
    t2 = tomllib.loads(hb.codex_config(src2, url, "YAMADORI_CODEX_KEY", None))
    check(t2["features"] == {"multi_agent": False, "shell_snapshot": False}
          and t2["shell_environment_policy"]["inherit"] == "all"
          and "YAMADORI_CODEX_KEY" in t2["shell_environment_policy"]["exclude"],
          "codex: an existing [features] / [shell_environment_policy] is edited, not duplicated",
          json.dumps(t2.get("features")))
    cx2 = tomllib.loads(hb.codex_config(CODEX_SRC, url, "K", None, sandbox_off=True))
    check(cx2["sandbox_mode"] == "danger-full-access" and "model_catalog_json" not in cx2,
          "codex --codex-sandbox off: danger-full-access (the container is the boundary)")


def test_write_home() -> None:
    with tempfile.TemporaryDirectory() as d:
        srcs = {}
        for name, data in (("oc.json", json.dumps(OPENCODE_SRC)), ("pi.json", json.dumps(PI_SRC))):
            srcs[name] = os.path.join(d, name)
            open(srcs[name], "w", encoding="utf-8").write(data)
        cdir = os.path.join(d, "codex-home")
        os.makedirs(cdir)
        open(os.path.join(cdir, "config.toml"), "w", encoding="utf-8").write(CODEX_SRC)
        open(os.path.join(cdir, "yamadori-catalog.json"), "w").write('{"models": []}')
        open(os.path.join(cdir, "auth.json"), "w").write(json.dumps({"token": SECRET}))
        homes, bare = {}, {}
        for h, src in (("opencode", srcs["oc.json"]), ("pi", srcs["pi.json"]), ("codex", cdir)):
            home = os.path.join(d, "run-" + h, "home")
            homes[h] = (home, hb.write_home(h, home, 18235, src))
            bhome = os.path.join(d, "bare-" + h, "home")
            bare[h] = (bhome, hb.write_home(h, bhome, 18235, src, loadout="as-tested"))
        rel = {h: sorted(os.path.relpath(p, home).replace("\\", "/") for p in w)
               for h, (home, w) in bare.items()}
        check(rel == {"opencode": [".config/opencode/opencode.json"],
                      "pi": [".pi/agent/models.json"],
                      "codex": [".codex/config.toml", ".codex/yamadori-catalog.json"]},
              "as-tested: each harness gets only its config (Codex: config + catalog, never auth.json)",
              json.dumps(rel))
        rel = {h: sorted(os.path.relpath(p, home).replace("\\", "/") for p in w)
               for h, (home, w) in homes.items()}
        check(rel == {"opencode": [".config/opencode/opencode.json"],
                      "pi": [".agents/skills/package-api/SKILL.md", ".agents/skills/package-api/api.cjs",
                             ".agents/skills/page-check/SKILL.md", ".agents/skills/type-check/SKILL.md",
                             ".pi/agent/models.json", ".pi/agent/settings.json"],
                      "codex": [".codex/config.toml", ".codex/skills/package-api/SKILL.md",
                                ".codex/skills/package-api/api.cjs", ".codex/skills/type-check/SKILL.md",
                                ".codex/yamadori-catalog.json"]},
              "the default loadout adds Pi's settings and the loadout skills (Pi, Codex), "
              "never auth.json", json.dumps(rel))
        oc = json.load(open(os.path.join(homes["opencode"][0], ".config", "opencode", "opencode.json")))
        perm = oc["permission"]
        keys = list(perm)
        check(all(perm[t] == "allow" for t in ("bash", "read", "edit", "glob", "grep", "lsp"))
              and all(perm[t] == "deny" for t in ("task", "todowrite", "skill", "question",
                                                  "webfetch", "websearch"))
              and perm["playwright_*"] == "deny"
              and all(perm[f"playwright_{t}"] == "allow" and keys.index(f"playwright_{t}")
                      > keys.index("playwright_*") for t in hb.PLAYWRIGHT_TOOLS)
              and oc["formatter"] is False,
              "opencode: the six tools + lsp, the rest denied, playwright_* denied then the chosen "
              "tools allowed AFTER it (last rule wins), formatter off", json.dumps(perm))
        lsp = oc["lsp"]
        check(all(lsp[i] == {"disabled": True} for i in hb.OPENCODE_LSP_BUILTINS)
              and len(hb.OPENCODE_LSP_BUILTINS) == 37
              and lsp["typescript-box"]["command"] == ["typescript-language-server", "--stdio"]
              and lsp["pyright-box"]["command"] == ["pyright-langserver", "--stdio"],
              "opencode: every built-in language server disabled, ours from the image", json.dumps(lsp)[:300])
        check(oc["mcp"] == {"playwright": {"type": "local", "command": hb.PLAYWRIGHT_MCP, "enabled": True}}
              and "--cdp-endpoint" in hb.PLAYWRIGHT_MCP and "--no-webmcp" in hb.PLAYWRIGHT_MCP,
              "opencode: the Playwright MCP server, attached over CDP, no page-registered tools")
        pis = json.load(open(os.path.join(homes["pi"][0], ".pi", "agent", "settings.json")))
        check(pis == {"defaultTools": ["read", "bash", "edit", "write", "grep", "find"]},
              "pi: the six tools (grep and find on)", json.dumps(pis))
        cx = tomllib.loads(open(os.path.join(homes["codex"][0], ".codex", "config.toml"),
                                encoding="utf-8").read())
        pw = cx["mcp_servers"]["playwright"]
        check(pw["command"] == "playwright-mcp" and pw["args"] == hb.PLAYWRIGHT_MCP[1:]
              and pw["enabled_tools"] == list(hb.PLAYWRIGHT_TOOLS)
              and pw["default_tools_approval_mode"] == "approve"
              and cx["sandbox_workspace_write"]["network_access"] is True
              and cx["features"]["shell_snapshot"] is False,
              "codex: the Playwright MCP server (chosen tools, approved), network in its sandbox, "
              "the key still off disk", json.dumps({k: cx.get(k) for k in ("mcp_servers",
                                                                             "sandbox_workspace_write")}))
        leaks = [p for home, _ in homes.values() for p in hb.secret_in(home, SECRET)]
        check(not leaks, "a literal key in the source reaches no file in the run home", str(leaks))
        home = homes["codex"][0]
        open(os.path.join(home, "planted.txt"), "w").write("x" + SECRET)
        check(hb.secret_in(home, SECRET) == [os.path.join(home, "planted.txt")],
              "the key scan finds a planted key")


def test_argv() -> None:
    lo = hb.run_argv("opencode", "t1", r"C:\p", r"C:\r\home", ["run"])
    check([lo[i + 1] for i, a in enumerate(lo) if a == "--network"] == ["container:harn-browser-t1"]
          and "--network-alias" not in lo,
          "the default loadout: the harness joins the browser sidecar's namespace")
    lenv = dict(a.split("=", 1) for a in lo[1::] if "=" in a and not a.startswith("-"))
    check(lenv.get("OPENCODE_DISABLE_LSP_DOWNLOAD") == "1" and lenv.get("OPENCODE_EXPERIMENTAL_LSP_TOOL") == "1"
          and "OPENCODE_EXPERIMENTAL" not in lenv,
          "opencode loadout env: no language-server downloads, the lsp tool, never OPENCODE_EXPERIMENTAL")
    sc = hb.sidecar_argv("t1")
    check(sc[sc.index("--network") + 1] == "harn-net-t1" and sc[sc.index("--network-alias") + 1] == "harness"
          and "-p" not in sc and "--publish" not in sc and hb.SIDECAR_IMAGE in sc
          and f"SIDECAR_PROXY={sn.PROXY_URL}" in sc and "--cap-drop" in sc,
          "the sidecar: on the run's network as `harness`, nothing published, Chrome's proxy the gate",
          json.dumps(sc))
    argv = hb.run_argv("opencode", "t1", r"C:\p", r"C:\r\home", ["run", "--format", "json"],
                       loadout="as-tested")
    joined = " ".join(argv)
    check(SECRET not in joined and "-e" in argv
          and argv[argv.index("YAMADORI_OPENCODE_KEY") - 1] == "-e",
          "the key is `-e NAME` only: no value in the argv", joined)
    nets = [argv[i + 1] for i, a in enumerate(argv) if a == "--network"]
    check(nets == ["harn-net-t1"] and "-p" not in argv and "--publish" not in argv
          and "--privileged" not in argv,
          "one network (the run's internal one), nothing published, not privileged", json.dumps(nets))
    check("-i" not in argv and "-t" not in argv,
          "no stdin by default (an open pipe makes opencode run / pi -p wait for EOF)")
    check(argv[argv.index("--user") + 1] == "1000:1000"
          and argv[argv.index("--cap-drop") + 1] == "ALL"
          and argv[argv.index("--security-opt") + 1] == "no-new-privileges"
          and "--pids-limit" in argv,
          "uid 1000, every capability dropped, no-new-privileges, a pids limit")
    mounts = [argv[i + 1] for i, a in enumerate(argv) if a == "-v"]
    check(mounts == [r"C:\p:/work", r"C:\r\home:/home/node"],
          "mounted: the project and the run home, nothing else", json.dumps(mounts))
    env = dict(a.split("=", 1) for a in argv[1::] if "=" in a and not a.startswith("-"))
    check(env.get("NO_PROXY", "").split(",")[-1] == "egress" and env.get("HTTPS_PROXY") == sn.PROXY_URL
          and env.get("NODE_USE_ENV_PROXY") == "1" and env.get("OPENCODE_DISABLE_AUTOUPDATE") == "1",
          "proxy env with egress bypassed (the forward), Node honours the proxy, autoupdate off",
          json.dumps(env))
    i = argv.index(hb.IMAGE)
    check(argv[i + 1:] == ["opencode", "run", "--format", "json"],
          "the pinned image, then the harness binary and its args")
    for h in ("pi", "codex"):
        a = hb.run_argv(h, "t", "p", "h", ["--version"])
        check(hb.HARNESSES[h]["key_env"] in a and a[a.index(hb.IMAGE) + 1] == hb.HARNESSES[h]["bin"],
              f"{h}: its own key variable and binary")


def test_forward() -> None:
    f = hb.forwards(18235)
    check(f == [(18235, "host.docker.internal", 18235, None)], "one forward, to the host's port")
    g = sn.gate_argv("t", f, prefix="harn")
    fw = [g[i + 1] for i, a in enumerate(g) if a == "--forward"]
    check(fw == ["18235:host.docker.internal:18235"] and "-p" not in g
          and g[g.index("--name") + 1] == "harn-gate-t",
          "the gate: exactly that forward, published nowhere on the host", json.dumps(g))
    for bad in (11434, 2019, 1235, 10001, 8888, 445, 80, 0, 70000):
        try:
            hb.check_port(bad)
            check(False, f"--target-port {bad} is refused")
        except SystemExit:
            pass
    ok = True
    for good in (1234, 18235, 18236):
        try:
            hb.check_port(good)
        except SystemExit:
            ok = False
    check(ok, "the proxy port and the relay ports are accepted; the #48 services are refused")
    pl = sn.plan("t", "harn", "harness", f, ("egress",))
    check("except the fixed forward(s) egress:18235 -> host.docker.internal:18235" in pl
          and "harn-net-t" in pl, "plan names the one exception", pl)


class Fake:
    def __init__(self, up_ok=True):
        self.events: list[str] = []
        self.env_seen: dict = {}
        self.argv_seen: list = []
        self.up_ok = up_ok

    def up(self, tag, forwards=(), wait_s=30, prefix="octo"):
        self.events.append(f"up:{prefix}:{forwards}")
        return {**sn.names(tag, prefix), "ok": self.up_ok, **({} if self.up_ok else {"error": "x"})}

    def down(self, tag, prefix="octo"):
        self.events.append("down")
        return {"allowed": 0, "denied": 0, "forwarded": 1}

    def run(self, cmd, **kw):
        self.events.append("docker-run")
        self.argv_seen = cmd
        self.env_seen = kw.get("env") or {}
        return subprocess.CompletedProcess(cmd, 0)


def _cmd_run(fake: Fake, plant: bool = False, loadout: str = "as-tested",
             sidecar_ok: bool = True) -> tuple[int, str]:
    real_sc = (hb.start_sidecar, hb.stop_sidecar)
    hb.start_sidecar = lambda tag: fake.events.append("sidecar-up") or {"ok": sidecar_ok, "error": "x"}
    hb.stop_sidecar = lambda tag: fake.events.append("sidecar-down") or {"removed": True}
    try:
        return _cmd_run_inner(fake, plant, loadout)
    finally:
        hb.start_sidecar, hb.stop_sidecar = real_sc


def _cmd_run_inner(fake: Fake, plant: bool, loadout: str) -> tuple[int, str]:
    real = (sn.up, sn.down, hb.image_present, hb.subprocess.run, hb.image_id)
    with tempfile.TemporaryDirectory() as d:
        key = os.path.join(d, "k.key")
        open(key, "w").write(SECRET + "\n")
        src = os.path.join(d, "oc.json")
        open(src, "w").write(json.dumps(OPENCODE_SRC))
        proj = os.path.join(d, "proj")
        os.makedirs(proj)
        if plant:
            open(os.path.join(proj, "notes.txt"), "w").write(SECRET)
        sn.up, sn.down, hb.image_present, hb.subprocess.run, hb.image_id = (
            fake.up, fake.down, lambda ref: True, fake.run, lambda ref: "sha256:" + "ab" * 32)
        try:
            a = types.SimpleNamespace(harness="opencode", target_port=18235, image=hb.IMAGE,
                                      key_file=key, project=proj, run_dir=os.path.join(d, "run"),
                                      config=src, codex_sandbox="keep", tag="t", tty=False, timeout=None, stdin=False,
                                      args=["run", "hi"], loadout=loadout)
            rc = hb.cmd_run(a)
        finally:
            sn.up, sn.down, hb.image_present, hb.subprocess.run, hb.image_id = real
        rp = os.path.join(d, "run", "harness_box.jsonl")
        rec = open(rp).read() if os.path.exists(rp) else ""
    return rc, rec


def test_run() -> None:
    f = Fake()
    rc, rec = _cmd_run(f)
    check(rc == 0 and f.events == ["up:harn:[(18235, 'host.docker.internal', 18235, None)]",
                                   "docker-run", "down"],
          "run: the network (with its one forward) up, the harness, the network down",
          json.dumps(f.events))
    check(f.env_seen.get("YAMADORI_OPENCODE_KEY") == SECRET and SECRET not in " ".join(f.argv_seen),
          "the key reaches the docker child's environment, never its argv")
    check(rec and SECRET not in rec and json.loads(rec)["rc"] == 0,
          "the run record holds no key", rec[:200])
    check(json.loads(rec).get("image_id") == "sha256:" + "ab" * 32
          and json.loads(rec).get("base_image") == hb.BASE_IMAGE,
          "the run record names the image digest used and its base")
    f = Fake(up_ok=False)
    rc, rec = _cmd_run(f)
    check(rc == 3 and "docker-run" not in f.events and json.loads(rec).get("not_run") == "sandbox_net",
          "no gate: the harness never starts (fail closed)", json.dumps(f.events))
    f = Fake()
    rc, _ = _cmd_run(f, plant=True)
    check(rc == 2 and f.events == [], "a key found in the project: not run, nothing started")
    f = Fake()
    rc, rec = _cmd_run(f, loadout="recommended")
    check(rc == 0 and f.events == ["up:harn:[(18235, 'host.docker.internal', 18235, None)]",
                                   "sidecar-up", "docker-run", "sidecar-down", "down"]
          and json.loads(rec).get("loadout") == "recommended",
          "default loadout: network, sidecar, harness, sidecar removed, network down", json.dumps(f.events))
    f = Fake()
    rc, rec = _cmd_run(f, loadout="recommended", sidecar_ok=False)
    check(rc == 3 and "docker-run" not in f.events and json.loads(rec).get("not_run") == "browser_sidecar"
          and f.events[-1] == "down",
          "no browser: the harness never starts (fail closed), everything removed", json.dumps(f.events))


def test_build_refuses() -> None:
    calls: list = []
    real = (hb.image_present, hb.subprocess.run)
    hb.image_present = lambda ref: False
    hb.subprocess.run = lambda *a, **k: calls.append(a) or subprocess.CompletedProcess(a, 0)
    try:
        rc = hb.cmd_build(types.SimpleNamespace(base=None))
    finally:
        hb.image_present, hb.subprocess.run = real
    check(rc == 2 and not calls, "build: base not local -> refused, no pull, no build")


def test_image_pins() -> None:
    df = open(os.path.join(hb.HARNESS_DIR, "Dockerfile"), encoding="utf-8").read()
    m = re.search(r"^ARG BASE=(\S+)$", df, re.M)
    check(m and m.group(1) == hb.BASE_IMAGE and re.search(r"@sha256:[0-9a-f]{64}$", m.group(1))
          and "FROM ${BASE}" in df and "node:24." in m.group(1),
          "Dockerfile: Node 24 base pinned by digest, the same as harness_box.BASE_IMAGE")
    check("npm audit signatures" in df and re.search(r"ADD --checksum=sha256:[0-9a-f]{64} \\\n\s+https://github.com/sharkdp/fd/releases/download/v10\.5\.0/", df),
          "Dockerfile: registry signatures verified at build; fd pinned by sha256")
    check("npm ci" in df and "--ignore-scripts" in df and "npm install" not in df
          and not re.search(r"^(ENV|ARG)\s+\S*(KEY|TOKEN|SECRET)", df, re.M | re.I)
          and "USER node" in df,
          "Dockerfile: npm ci from the lockfile, scripts off, no key variable, runs as node")
    pj = json.load(open(os.path.join(hb.HARNESS_DIR, "package.json"), encoding="utf-8"))
    want = {"opencode-ai": hb.VERSIONS["opencode"], "@earendil-works/pi-coding-agent": hb.VERSIONS["pi"],
            "@openai/codex": hb.VERSIONS["codex"], **hb.LOADOUT_PACKAGES}
    check(pj["dependencies"] == want and "cclsp" not in json.dumps(pj),
          "package.json: exactly the three harnesses and the loadout's packages, exact versions "
          "(no cclsp: it reported a file with type errors clean)",
          json.dumps(pj["dependencies"]))
    lock = json.load(open(os.path.join(hb.HARNESS_DIR, "package-lock.json"), encoding="utf-8"))
    pk = lock["packages"]
    top = {n: pk.get(f"node_modules/{n}", {}).get("version") for n in want}
    check(top == want and pk[""]["dependencies"] == want and lock["lockfileVersion"] >= 3,
          "lockfile: the same packages at the same versions", json.dumps(top))
    bad = [k for k, v in pk.items() if k and not v.get("link")
           and not (str(v.get("integrity", "")).startswith("sha512-")
                    and str(v.get("resolved", "")).startswith("https://registry.npmjs.org/"))]
    check(not bad, f"lockfile: all {len(pk) - 1} packages from registry.npmjs.org with a sha512",
          json.dumps(bad[:5]))
    check(all(f"node_modules/{p}" in pk for p in ("opencode-linux-x64", "@openai/codex-linux-x64")),
          "lockfile: the linux-x64 binaries of OpenCode and Codex are pinned too")
    import hashlib
    import yaml
    rts = yaml.safe_load(open(hb.MANIFEST, encoding="utf-8"))["runtimes"]
    rt = next((r for r in rts if r["id"] == "harness-box"), {})
    got = {os.path.basename(f["path"]): hashlib.sha256(open(os.path.join(
        hb.HARNESS_DIR, os.path.basename(f["path"])), "rb").read()).hexdigest() == f["sha256"]
        for f in rt.get("files") or []}
    check(got == {"Dockerfile": True, "package.json": True, "package-lock.json": True}
          and rt.get("base") == hb.BASE_IMAGE and rt.get("image") == hb.IMAGE
          and re.fullmatch(r"sha256:[0-9a-f]{64}", hb.recorded_image_id() or ""),
          "models/manifest.yaml harness-box: the recipe's sha256s match these files; "
          "base, image and image_id recorded", json.dumps(got))


def test_down_leftovers() -> None:
    """sandbox_net.down removes a container still attached to the network (a
    harness whose cleanup died with it) before the network, never the gate
    twice, and counts the gate's forwards."""
    calls: list[list[str]] = []

    def fake(args, timeout=120):
        calls.append(list(args))
        out = ""
        if args[:2] == ["ps", "-aq"] and args[3].startswith("network="):
            out = "gate111\nstray222\n"
        elif args[:2] == ["ps", "-aq"]:
            out = "gate111\n"
        elif args[0] == "logs":
            out = "10:00 gate ready\n10:01 FORWARD host.docker.internal:1234 from 172.18.0.3\n"
        return subprocess.CompletedProcess(["docker", *args], 0, out, "")
    real = sn._docker
    sn._docker = fake
    try:
        st = sn.down("t", prefix="harn")
    finally:
        sn._docker = real
    rms = [c for c in calls if c[:2] == ["rm", "-f"]]
    order = [" ".join(c[:2]) for c in calls]
    check(rms[0] == ["rm", "-f", "stray222"] and rms[1] == ["rm", "-f", "harn-gate-t"]
          and order.index("network rm") > order.index("rm -f")
          and st["leftovers_removed"] == 1 and st["forwarded"] == 1,
          "down: a leftover container goes first, then the gate, then the network",
          json.dumps([calls, st]))


def test_loadout_check() -> None:
    """The no-model loadout check (loadout_check.py + scripted_model.py): the
    stand-in plays each step once, to a tool the harness offers, on both
    wires; each harness's script calls only tools its loadout offers; the
    recorded results (bench/sandbox/results/loadout_check.jsonl) passed."""
    import loadout_check as lc
    import scripted_model as smod
    smod.STEPS[:] = [{"tool": ["bash"], "args": {"command": "x"}},
                     {"tool": ["mcp__playwright::browser_navigate", "playwright_browser_navigate"],
                      "args": {"url": "u"}}]
    chat = {"tools": [{"type": "function", "function": {"name": "bash"}},
                      {"type": "function", "function": {"name": "playwright_browser_navigate"}}],
            "messages": [{"role": "user", "content": "hi"}]}
    check(smod.pick(chat)[:3] == (0, "bash", {"command": "x"}), "stand-in: step 0 first")
    chat["messages"] += [{"role": "assistant"}, {"role": "tool", "content": "r"}]
    check(smod.pick(chat)[1] == "playwright_browser_navigate", "stand-in: step 1 to the name offered")
    chat["messages"] += [{"role": "tool", "content": "r"}]
    check(smod.pick(chat)[1] is None, "stand-in: steps done -> ok")
    resp = {"tools": [{"type": "function", "name": "exec_command"},
                      {"type": "namespace", "name": "mcp__playwright",
                       "tools": [{"type": "function", "name": "browser_navigate"}]}],
            "input": [{"type": "function_call_output", "output": "r"}]}
    check(smod.pick(resp)[1] == "mcp__playwright::browser_navigate"
          and b'"namespace": "mcp__playwright"' in smod.responses_body("mcp__playwright::browser_navigate", {}, 1),
          "stand-in: a Responses namespace tool is called with its namespace")
    check(smod.pick({"messages": [{"role": "user"}]})[1] is None, "stand-in: a side call without tools -> ok")
    offered = {"opencode": {"bash", "write", "lsp", *(f"playwright_{t}" for t in hb.PLAYWRIGHT_TOOLS)},
               "pi": set(hb.PI_TOOLS),
               "codex": {"exec_command", *(f"mcp__playwright::{t}" for t in hb.PLAYWRIGHT_TOOLS)}}
    for h in ("opencode", "pi", "codex"):
        bad = [st["tool"] for st in lc.script(h) if not set(st["tool"]) & offered[h]]
        check(not bad, f"{h}: the check calls only tools its loadout offers", json.dumps(bad))
    v = lc.judge("pi", ["FIXTURES_OK SERVER_200 PROXIED_HOST_403 DIRECT_HOST_000", "BoxPage", "", "", "", ""])
    check(v["fixtures_and_server"] and v["shell_host_blocked"] and not v["browser_console_error"]
          and not v["types_ts"], "judge: a missing result fails its check")
    rp = os.path.join(HERE, "results", "loadout_check.jsonl")
    rows = {}
    for ln in open(rp, encoding="utf-8"):
        r = json.loads(ln)
        rows[r["harness"]] = r
    check(set(rows) == {"opencode", "pi", "codex"}
          and all(r["ok"] and all(r["verdict"].values()) and r["results"] == r["steps"]
                  for r in rows.values())
          and {r["image_id"] for r in rows.values()} == {hb.recorded_image_id()},
          "recorded: all three harnesses passed every check, on the recorded image",
          json.dumps({h: r["verdict"] for h, r in rows.items()})[:400])


def test_wsl_engine():
    """THE WSL ENGINE (docs/DOCKER-WSL.md, 2026-10-07): every bind mount is the
    engine's own path (/mnt/c/...), and the gate's one forward points at the
    Windows side of the WSL NAT, not at host.docker.internal; Docker Desktop's
    argv is unchanged."""
    saved = {k: os.environ.get(k) for k in ("YAMADORI_DOCKER_PATHS", "YAMADORI_DOCKER_HOST_IP")}
    proj, home = "C:\\work\\proj", "C:\\work\\run\\home"
    try:
        os.environ["YAMADORI_DOCKER_PATHS"] = "native"
        os.environ.pop("YAMADORI_DOCKER_HOST_IP", None)
        native = hb.run_argv("pi", "t", proj, home, ["x"])
        check(proj + ":" + hb.WORK in native and hb.target_host() == "host.docker.internal"
              and hb.forwards(18234) == [(18234, "host.docker.internal", 18234, None)],
              "Docker Desktop: the Windows paths and host.docker.internal, as before")
        os.environ["YAMADORI_DOCKER_PATHS"] = "wsl"
        os.environ["YAMADORI_DOCKER_HOST_IP"] = "172.29.32.1"
        argv = hb.run_argv("pi", "t", proj, home, ["x"])
        mounts = [argv[i + 1] for i, x in enumerate(argv) if x == "-v"]
        check(mounts == ["/mnt/c/work/proj:" + hb.WORK, "/mnt/c/work/run/home:" + hb.HOME]
              and not any(re.match(r"^[A-Za-z]:", m) for m in mounts),
              "WSL engine: the project and the run home are mounted by their /mnt/c path", str(mounts))
        side = hb.sidecar_argv("t")
        sm = [side[i + 1] for i, x in enumerate(side) if x == "-v"]
        check(len(sm) == 1 and sm[0].startswith("/mnt/c/") and sm[0].endswith(":/octo:ro"),
              "WSL engine: the browser sidecar's /octo mount is a /mnt/c path", str(sm))
        gate = sn.gate_argv("t", hb.forwards(18234), "yh")
        check(hb.target_host() == "172.29.32.1"
              and hb.forwards(18234) == [(18234, "172.29.32.1", 18234, None)]
              and "18234:172.29.32.1:18234" in gate,
              "WSL engine: the gate's ONE forward targets the Windows side of the WSL NAT")
        check(all(f[3] is None for f in hb.forwards(18234)) and len(hb.forwards(18234)) == 1
              and "-p" not in gate,
              "...still one forward, kept inside the internal network (nothing published)")
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def main() -> int:
    for fn in (test_configs, test_write_home, test_argv, test_forward, test_run, test_wsl_engine,
               test_build_refuses, test_image_pins, test_down_leftovers, test_loadout_check):
        print(f"\n--- {fn.__name__} ---")
        n0 = len(_results)
        try:
            fn()
        except Exception:                                        # noqa: BLE001
            check(False, f"{fn.__name__} itself raised", traceback.format_exc().strip().split("\n")[-1])
            traceback.print_exc()
        for ok, name, detail in _results[n0:]:
            print(("  pass  " if ok else "  FAIL  ") + name
                  + (f"   <- {detail}" if not ok and detail else ""))
    passed = sum(1 for ok, _, _ in _results if ok)
    print(f"\n{'=' * 70}\n  {passed}/{len(_results)} checks passed")
    return 0 if passed == len(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
