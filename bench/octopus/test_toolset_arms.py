#!/usr/bin/env python
"""The Octopus tools arms (toolset_arms.py, browser_sidecar.py): the profile
edits go both ways for both browser hosts, exactly once each; the sandbox
sidecar publishes only CDP on host loopback; its lifecycle is recorded; and
`measure` reads what a run did with its page. No GPU, no network, no docker,
no Hermes, no live run (docker is faked).

    python bench/octopus/test_toolset_arms.py      -> "N/M checks passed"
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import browser_sidecar as bs  # noqa: E402
import sandbox_net as sn  # noqa: E402
import toolset_arms as ta  # noqa: E402

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    _results.append((bool(ok), name, detail))


# The lines of the real profile the arm touches, in its order (octo/hermes-home
# config.yaml, make_profile.py + the Blank Slate setup), with neighbours.
PROFILE = "\n".join([
    "terminal:",
    '  backend: "docker"',
    "  docker_volumes: ['C:\\x:/workspace']   # OCTO-RUN-VOLUMES (run.py rewrites per run)",
    "",
    "browser:",
    "  # Inactivity timeout in seconds",
    "  inactivity_timeout: 120",
    "  extension_control:",
    "    enabled: false",
    "agent:",
    "  disabled_toolsets:",
    "    - bot_room",
    "    - browser",
    "    - clarify",
    "skills:",
    "  creation_nudge_interval: 0",
    "platform_toolsets:",
    "  cli: [file, skills, terminal, vision]",
    "  telegram: [hermes-telegram]",
    "",
])
SC = "octo-browser-t-V0-xhigh-1-p1"


class FakeDocker:
    """Stands in for toolset_arms._docker: records argv, answers by verb."""

    def __init__(self, rc: int = 0, logs: str = ""):
        self.calls: list[list[str]] = []
        self.rc, self.logs = rc, logs

    def __call__(self, args, timeout=120):
        self.calls.append(list(args))
        out = {"logs": self.logs, "image": "sha256:abc", "run": "0123456789abcdef"}.get(args[0], "")
        return subprocess.CompletedProcess(["docker", *args], self.rc if args[0] == "run" else 0,
                                           out, "boom" if self.rc else "")


def test_round_trip() -> None:
    sb = ta.apply(PROFILE, "browser", "sandbox", SC)
    ho = ta.apply(PROFILE, "browser", "host")
    check(ta.apply(PROFILE, "default", loadout=False) == PROFILE,
          "control without the loadout leaves a control profile unchanged")
    for name, arm in (("sandbox", sb), ("host", ho)):
        check(ta.apply(arm, "default", loadout=False) == PROFILE,
              f"{name} -> default, loadout off, restores the profile byte for byte")
        check(ta.apply(arm, "browser", name, SC) == arm, f"{name}: applying the arm twice changes nothing")
    check(ta.apply(ho, "browser", "sandbox", SC) == sb, "host -> sandbox equals sandbox from control")
    check(ta.apply(sb, "browser", "host") == ho, "sandbox -> host equals host from control")
    check("docker_extra_args" not in sb and "docker_extra_args" not in ho,
          "the arm writes no docker_extra_args (run.py's OCTO-RUN-NET line does, every arm)")
    retired = PROFILE.replace(
        "   # OCTO-RUN-VOLUMES (run.py rewrites per run)\n",
        "   # OCTO-RUN-VOLUMES (run.py rewrites per run)\n"
        '  docker_extra_args: ["--network", "container:old"]   # OCTO-ARM:docker_args (toolset_arms.py)\n')
    check(retired != PROFILE and ta.apply(retired, "default", loadout=False) == PROFILE
          and ta.apply(retired, "browser", "sandbox", SC) == sb,
          "an old run's retired docker_args line is removed on the next apply")
    import yaml
    d = yaml.safe_load(sb)
    check(d["browser"] == {"backend": "off", "allow_private_urls": True,
                           "cdp_url": f"http://127.0.0.1:{ta.CDP_HOST_PORT}",
                           "restrict_evaluate": True,
                           "inactivity_timeout": 21600, "extension_control": {"enabled": False}},
          "sandbox: browser section = backend off, private URLs, cdp_url, restrict_evaluate, 21600 s",
          json.dumps(d["browser"]))
    check("browser" not in d["agent"]["disabled_toolsets"]
          and "browser-cdp" in d["agent"]["disabled_toolsets"]
          and "browser" in d["platform_toolsets"]["cli"],
          "browser on, browser-cdp (raw CDP + dialog tools) off", json.dumps(d["agent"]))
    h = yaml.safe_load(ho)
    check(h["browser"].get("restrict_evaluate") is True and "cdp_url" not in h["browser"],
          "host: restrict_evaluate, no cdp_url", json.dumps(h["browser"]))


def test_default_loadout() -> None:
    """The browser is the DEFAULT (operator, 2026-09-26), and every arm --
    the opt-out included -- gets the loadout lines: Tool Search off, the
    external password managers off as vault sources."""
    import yaml
    check(ta.DEFAULT_ARM == "lean" and ta.DEFAULT_BROWSER_HOST == "sandbox"
          and ta.LOADOUT_VERSION == "loadout-2",
          "the default arm is lean (loadout-2, operator 2026-09-28); browser stays as the opt-out")
    for arm, host in (("default", "sandbox"), ("browser", "sandbox"), ("browser", "host"),
                      ("lean", "sandbox")):
        t = ta.apply(PROFILE, arm, host, SC)
        d = yaml.safe_load(t)
        check(d.get("tools") == {"tool_search": {"enabled": "off"}}
              and d.get("vault") == {"onepassword": {"enabled": False},
                                     "bitwarden": {"enabled": False}},
              f"{arm}/{host}: tool_search off (a string, not YAML false), vault managers off",
              json.dumps({k: d.get(k) for k in ("tools", "vault")}))
        check(ta.apply(t, arm, host, SC) == t, f"{arm}/{host}: applying again changes nothing")
        check(t.count("# OCTO-LOADOUT:") == 3, f"{arm}/{host}: the loadout lines appear once each")
        check(d["skills"] == {"creation_nudge_interval": 0, "create_dir": ta.RUN_SKILLS_DIR},
              f"{arm}/{host}: skills skill_manage creates go to the run-scoped folder",
              json.dumps(d["skills"]))
    opt = yaml.safe_load(ta.apply(PROFILE, "default"))
    check("browser" in opt["agent"]["disabled_toolsets"] and "browser" not in opt["platform_toolsets"]["cli"],
          "--tools default (the opt-out) has no browser")
    try:
        ta.apply(PROFILE + "tools:\n  x: 1\n", "browser", "sandbox", SC)
        check(False, "refuses a profile that already has a top-level tools: key")
    except SystemExit as e:
        check("already has" in str(e), "refuses a profile that already has a top-level tools: key", str(e))
    check(ta.volume_spec() == f"{ta.TOOLS_VOLUME}:{ta.TOOLS_MOUNT}:ro" and ta.TOOLS_MOUNT == "/opt/yamadori-tools",
          "the type checkers mount read-only at /opt/yamadori-tools")
    check("/opt/yamadori-tools/typescript/bin/tsc" in ta.POPULATE
          and "/opt/yamadori-tools/pyright/index.js" in ta.POPULATE
          and "--network" not in ta.POPULATE,
          "the volume is filled from the box image's pinned typescript and pyright")
    with tempfile.TemporaryDirectory() as home:
        os.makedirs(os.path.join(home, ta.RUN_SKILLS_DIR, "a-saved-skill"))
        n = ta.clear_run_skills(home)
        check(n == 1 and os.listdir(os.path.join(home, ta.RUN_SKILLS_DIR)) == [],
              "a skill a previous run saved is removed before the next run")
    real = ta._docker
    fd = FakeDocker()
    ta._docker = fd
    try:
        tv = ta.tools_volume(create=True)
        check(tv["ok"] is True, "the volume is made when missing", json.dumps(tv))
        runs = [c for c in fd.calls if c[0] == "run"]
        check(runs and runs[0][:4] == ["run", "--rm", "--network", "none"]
              and f"{ta.TOOLS_VOLUME}:/dst" in runs[0],
              "the volume is made with --network none, from the box image", json.dumps(runs))
        check(["volume", "create", "--label", "yamadori.image=sha256:abc", ta.TOOLS_VOLUME] in fd.calls,
              "the volume's label names the image it came from", json.dumps(fd.calls))
    finally:
        ta._docker = real
    sc = ta.sidecar_argv(SC, "t", "t-p1")
    check("SIDECAR_ERROR_MIRROR=1" in sc and "Uncaught " in bs.MIRROR_JS
          and "unhandledrejection" in bs.MIRROR_JS and bs.PROXY == bs.DEAD_PROXY,
          "the Octopus sidecar mirrors uncaught page errors to console.error (Hermes drops their "
          "text); its proxy stays the dead one")
    skill = open(os.path.join(HERE, "hermes_skills", "type-check", "SKILL.md"), encoding="utf-8").read()
    check("/opt/yamadori-tools/bin/tsc" in skill and "/opt/yamadori-tools/bin/pyright" in skill
          and skill.startswith("---\nname: type-check\n"),
          "the Hermes type-check skill names the mounted checkers")


def test_lean_arm() -> None:
    """The lean arm (loadout-2): file, terminal, vision and the Playwright MCP
    server's three tools; no browser toolset, no skills; the MCP server runs
    from the harness box image in the sidecar's namespace, never pulled."""
    import yaml
    t = ta.apply(PROFILE, "lean", "sandbox", SC)
    d = yaml.safe_load(t)
    check(d["platform_toolsets"]["cli"] == ["file", "terminal", "vision", "playwright"],
          "cli = file, terminal, vision, playwright (no skills, no browser)",
          json.dumps(d["platform_toolsets"]))
    check("browser" in d["agent"]["disabled_toolsets"] and "browser-cdp" not in d["agent"]["disabled_toolsets"]
          and "backend" not in d["browser"] and "cdp_url" not in d["browser"],
          "the browser toolset stays disabled and its keys untouched", json.dumps(d["browser"]))
    srv = d["mcp_servers"]["playwright"]
    args = srv["args"]
    check(srv["command"] == "docker" and args[:2] == ["run", "-i"]
          and args[args.index("--network") + 1] == f"container:{SC}"
          and args[args.index("--pull") + 1] == "never" and "--rm" in args
          and args[args.index("--user") + 1] == "1000:1000" and "ALL" in args,
          "the MCP server: docker run -i --rm --pull never, the sidecar's namespace, uid 1000, caps dropped",
          json.dumps(args))
    hb = ta._hb()
    i = args.index(hb.IMAGE)
    check(args[i + 1:] == hb.PLAYWRIGHT_MCP and "--no-webmcp" in args,
          "the harness box image runs the same playwright-mcp args OpenCode and Codex run")
    check(srv["tools"] == {"include": list(ta.MCP_TOOLS), "resources": False, "prompts": False}
          and ta.MCP_TOOLS == ("browser_navigate", "browser_console_messages", "browser_take_screenshot"),
          "only navigate, console_messages, take_screenshot; no resource/prompt utilities",
          json.dumps(srv["tools"]))
    check(ta.apply(t, "lean", "sandbox", SC) == t, "lean: applying twice changes nothing")
    check(ta.apply(t, "default", loadout=False) == PROFILE, "lean -> control restores the profile byte for byte")
    b = ta.apply(PROFILE, "browser", "sandbox", SC)
    check(ta.apply(b, "lean", "sandbox", SC) == t and ta.apply(t, "browser", "sandbox", SC) == b,
          "browser <-> lean land on the same bytes as from control")
    other = ta.apply(t, "lean", "sandbox", "octo-browser-other-p2")
    check("container:octo-browser-other-p2" in other and SC not in other
          and other.count("mcp_servers:") == 1,
          "a new run rewrites the MCP line with its own sidecar, once")
    try:
        ta.apply(PROFILE + "mcp_servers:\n  x: {command: y}\n", "lean", "sandbox", SC)
        check(False, "refuses a profile that already has a top-level mcp_servers:")
    except SystemExit as e:
        check("already has" in str(e), "refuses a profile that already has a top-level mcp_servers:", str(e))
    check(ta.uses_sidecar("lean") and ta.uses_sidecar("browser", "sandbox")
          and not ta.uses_sidecar("browser", "host") and not ta.uses_sidecar("default"),
          "the sidecar: lean always, browser in the sandbox host")
    check(ta.gate_forwards("lean") == [] and ta.env("lean") == {},
          "lean: the gate publishes nothing on the host, no env added")
    xa = ta.docker_extra_args("lean", "sandbox", SC, "t-p1")
    check(xa[:2] == ["--network", f"container:{SC}"], "lean: the terminal joins the sidecar's namespace")
    check("SIDECAR_ERROR_MIRROR=1" not in ta.sidecar_argv(SC, "t", "t-p1", mirror=False)
          and "SIDECAR_ERROR_MIRROR=1" in ta.sidecar_argv(SC, "t", "t-p1"),
          "the error mirror is the browser arm's only (Playwright reports page errors itself)")
    fx = json.load(open(ta.MCP_FIXTURE, encoding="utf-8"))
    names = [x["name"] for x in fx["tools"]]
    check(fx["version"] == hb.LOADOUT_PACKAGES["@playwright/mcp"] and len(names) == 25
          and all(n in names for n in ta.MCP_TOOLS)
          and all(isinstance(x["inputSchema"], dict) for x in fx["tools"]),
          "the captured tools/list is the pinned 0.0.82's, all 25, with the three chosen", str(len(names)))
    check("_register_from_cache_sync" in ta._SCHEMA_PROBE and "OCTO_MCP_FIXTURE" in ta._SCHEMA_PROBE,
          "the offline builder registers the MCP tools through Hermes' own cached-manifest path")
    g = ta._guard_src()
    check("1234" in g and "18234" in g and "11434" in g and "ConnectionRefusedError" in g,
          "every Hermes probe refuses the stack's ports (the builder probes :1234)")
    real = ta._docker
    fd = FakeDocker()
    ta._docker = fd
    try:
        p = ta.prereqs("lean")
        check(p["ok"] is True and p["sidecar_image"] == "sha256:abc" and p["mcp_image"] == "sha256:abc"
              and not any("agent-browser" in x for x in p.get("problems", [])),
              "lean needs the sidecar and harness box images, nothing on the host", json.dumps(p))
        fd2 = FakeDocker()
        ta._docker = fd2
        rec = ta.start_sidecar(SC, "t", wait_s=0, net_tag="t-p1", arm="lean")
        check("SIDECAR_ERROR_MIRROR=1" not in rec["argv"], "lean's sidecar starts without the mirror")
        ta._LIVE.discard(SC)
        fd3 = FakeDocker()
        ta._docker = fd3
        ta.mcp_cleanup(SC)
        check(["ps", "-aq", "--filter", f"label=octo-mcp-of={SC}"] in fd3.calls,
              "a Playwright MCP container Hermes left is found by its sidecar label", json.dumps(fd3.calls))
    finally:
        ta._docker = real


def test_verify_verdict() -> None:
    """verify()'s judgement, on records shaped like its own."""
    tools = ([f"mcp__playwright__{t}" for t in ta.MCP_TOOLS]
             + ["read_file", "write_file", "patch", "search_files", "terminal", "process_manage",
                "vision_analyze"])
    good = {"terminal": {"out": "SERVER_200\nPROXIED_HOST_403\nbad.ts(1,14): error TS2322\n"},
            "mcp_cleanup": {"left": 0},
            "hermes": {"tools": tools,
                       "media": {"png": True, "bytes": 1234},
                       "calls": [{"result": "### Page\n- Page Title: OctoPage"},
                                 {"result": "MEDIA:C:\\x\\cache\\images\\img.png"},
                                 {"result": "[ERROR] OCTO_CONSOLE_ERROR @ x\n"
                                            "ReferenceError: undefinedFn is not defined"},
                                 {"result": "Error: net::ERR_PROXY_CONNECTION_FAILED"},
                                 {"result": "Error: net::ERR_PROXY_CONNECTION_FAILED"}]}}
    v = ta.verify_verdict("lean", good)
    check(v["ok"] is True, "lean: a record with every check met passes", json.dumps(v))
    bad = json.loads(json.dumps(good))
    bad["hermes"]["tools"].append("browser_vault_list")
    check(ta.verify_verdict("lean", bad)["tools_exactly_lean"] is False,
          "lean: one extra tool fails the exact list")
    bad = json.loads(json.dumps(good))
    bad["hermes"]["media"] = {"error": "not found"}
    check(ta.verify_verdict("lean", bad)["screenshot_media_path_readable"] is False,
          "lean: a MEDIA path vision_analyze could not read fails")


def test_measure_lean() -> None:
    with tempfile.TemporaryDirectory() as logs:
        d = os.path.join(logs, "t-lean-1")
        os.makedirs(d)
        ev = [{"type": "tool_use", "name": "mcp__playwright__browser_navigate",
               "input": {"url": "http://localhost:3001/"}, "timestamp": 1_000_000},
              {"type": "tool_use", "name": "mcp__playwright__browser_console_messages",
               "input": {"level": "error"}, "timestamp": 1_010_000},
              {"type": "tool_result", "name": "mcp__playwright__browser_console_messages",
               "output": "### Result\nTotal messages: 2 (Errors: 2, Warnings: 0)\n\n"
                         "[ERROR] BOX_CONSOLE_ERROR @ http://localhost:3001/:0\n"
                         "ReferenceError: undefinedFn is not defined\n    at http://localhost:3001/:1:146"}]
        with open(os.path.join(d, "hermes.jsonl"), "w", encoding="utf-8") as f:
            f.write("\n".join(json.dumps(x) for x in ev) + "\n")
        m = ta.measure("t-lean-1", logs)
    check(m["opened_page_browser"] is True
          and m["browser_calls"] == {"browser_console_messages": 1, "browser_navigate": 1},
          "lean: the MCP tools count under their bare names", json.dumps(m))
    check(m["errors_seen_by_model"] == ["BOX_CONSOLE_ERROR @ http://localhost:3001/:0",
                                        "ReferenceError: undefinedFn is not defined"],
          "lean: errors read through Playwright's console output are collected",
          json.dumps(m["errors_seen_by_model"]))


def test_network_args() -> None:
    """Every arm's terminal container is on the sandbox network (#47); none
    publishes a port itself; the gate publishes what the host needs."""
    tag = "t-V0-xhigh-1-p1"
    proxy = [f"HTTPS_PROXY={sn.PROXY_URL}", f"HTTP_PROXY={sn.PROXY_URL}",
             f"NO_PROXY={sn.NO_PROXY}"]
    for arm, host in (("default", "sandbox"), ("browser", "sandbox"), ("browser", "host")):
        xa = ta.docker_extra_args(arm, host, SC, tag)
        envs = [xa[i + 1] for i, a in enumerate(xa) if a == "-e"]
        check(all(p in envs for p in proxy) and "-p" not in xa and "--publish" not in xa
              and xa.count("--network") == 1,
              f"{arm}/{host}: one --network, the gate as proxy, no published port", json.dumps(xa))
    check(ta.docker_extra_args("default", net_tag=tag)[:4]
          == ["--network", f"octo-net-{tag}", "--network-alias", sn.TERMINAL_ALIAS],
          "control: the internal network, as octo-term")
    check(ta.docker_extra_args("browser", "host", SC, tag)[:2] == ["--network", f"octo-net-{tag}"],
          "host browser: the internal network too (the gate publishes the game port)")
    check(ta.docker_extra_args("browser", "sandbox", SC, tag)[:2] == ["--network", f"container:{SC}"],
          "sandbox browser: the sidecar's namespace (the sidecar is on the internal network)")
    check(ta.gate_forwards("default") == [], "control: the gate publishes nothing")
    check(ta.gate_forwards("browser", "sandbox")
          == [(ta.GATE_CDP_PORT, sn.BROWSER_ALIAS, ta.SIDECAR_FORWARD_PORT, ta.CDP_HOST_PORT)],
          "sandbox: host 127.0.0.1:9222 -> gate -> octo-browser:9323 (CDP)")
    check(ta.gate_forwards("browser", "host")
          == [(ta.PORT, sn.TERMINAL_ALIAS, ta.PORT, ta.PORT)],
          "host: host 127.0.0.1:3001 -> gate -> octo-term:3001 (the game)")


def test_refuses_what_it_cannot_place() -> None:
    for broken, why in ((PROFILE.replace("    - browser\n", ""), "no browser entry"),
                        (PROFILE.replace("    - clarify", "    - browser"), "two browser entries"),
                        (PROFILE.replace("  cli: [file, skills, terminal, vision]",
                                         "  cli: [hermes-cli]"), "a different cli list"),
                        (PROFILE.replace("browser:\n", "browser: {}\n"), "no browser: section line")):
        try:
            ta.apply(broken, "browser", "sandbox", SC)
            check(False, f"refuses a profile with {why}")
        except SystemExit as e:
            check("matched" in str(e), f"refuses a profile with {why}", str(e))
    for args in (("nope",), ("browser", "cloud")):
        try:
            ta.apply(PROFILE, *args)
            check(False, f"refuses {args}")
        except SystemExit:
            check(True, f"refuses {args}")


def test_env_and_prereqs() -> None:
    check(ta.env("default") == {} and ta.env("browser", "sandbox") == {},
          "control and sandbox add no env (no Chromium on the host)")
    check(set(ta.env("browser", "host")) == {"AGENT_BROWSER_EXECUTABLE_PATH", "AGENT_BROWSER_ARGS"},
          "host adds the Chrome path and software-GL args")
    check(ta.prereqs("default") == {"ok": True}, "the control needs nothing")
    real = ta._docker, ta._chrome_version
    ta._docker, ta._chrome_version = FakeDocker(), lambda: "155.0"
    try:
        with tempfile.TemporaryDirectory() as home:
            p = ta.prereqs("browser", "sandbox", home)
            check(p["ok"] is False and p["sidecar_image"] == "sha256:abc"
                  and any("agent-browser" in x for x in p["problems"]) and "chrome" not in p,
                  "sandbox: needs agent-browser on the host and the sidecar image, not Chrome",
                  json.dumps(p))
            os.makedirs(os.path.join(home, "node", "node_modules", "agent-browser"))
            with open(os.path.join(home, "node", "node_modules", "agent-browser",
                                   "package.json"), "w") as f:
                json.dump({"version": ta.AGENT_BROWSER_VERSION}, f)
            open(os.path.join(home, "node", "agent-browser.cmd"), "w").close()
            p = ta.prereqs("browser", "sandbox", home)
            check(p["ok"] is (p["cdp_port_free"] is True),
                  "sandbox: with agent-browser installed only the CDP port can block", json.dumps(p))
            p = ta.prereqs("browser", "host", home)
            check(p.get("chrome") == "155.0" and "sidecar_image" not in p,
                  "host: needs Chrome, not the sidecar image", json.dumps(p))
    finally:
        ta._docker, ta._chrome_version = real


def test_sidecar() -> None:
    argv = ta.sidecar_argv(SC, "t-V0-xhigh-1", "t-V0-xhigh-1-p1")
    pubs = [argv[i + 1] for i, a in enumerate(argv) if a == "-p"]
    check(pubs == [] and argv[argv.index("--network") + 1] == "octo-net-t-V0-xhigh-1-p1"
          and argv[argv.index("--network-alias") + 1] == sn.BROWSER_ALIAS,
          "the sidecar is on the internal network as octo-browser and publishes nothing",
          json.dumps(argv))
    gate = sn.gate_argv("t-V0-xhigh-1-p1", ta.gate_forwards("browser", "sandbox"))
    check([gate[i + 1] for i, a in enumerate(gate) if a == "-p"]
          == [f"127.0.0.1:{ta.CDP_HOST_PORT}:{ta.GATE_CDP_PORT}"]
          and f"{ta.GATE_CDP_PORT}:{sn.BROWSER_ALIAS}:{ta.SIDECAR_FORWARD_PORT}" in gate,
          "CDP reaches the host through the gate, on host loopback only", json.dumps(gate))
    check(ta.SIDECAR_IMAGE in argv and f"{HERE}:/octo:ro" in argv and "--rm" in argv
          and "--init" in argv, "grader image, read-only mount of bench/octopus, --rm, --init")
    check(bs.FORWARD_PORT == ta.SIDECAR_FORWARD_PORT and bs.CHROME_PORT != bs.FORWARD_PORT,
          "the host maps to the forwarder, not to Chrome's loopback-only port")
    check(f"--proxy-server={bs.DEAD_PROXY}" in bs.FLAGS and bs.DEAD_PROXY.startswith("http://127.0.0.1:")
          and f"--remote-debugging-port={bs.CHROME_PORT}" in bs.FLAGS,
          "Chromium: everything but loopback goes to a closed proxy; DevTools on CHROME_PORT")
    check(ta.sidecar_name("a b/c", 2) == "octo-browser-a-b-c-p2", "sidecar names are docker-safe")

    real = ta._docker
    try:
        ta._docker = FakeDocker(rc=1)
        rec = ta.start_sidecar(SC, "t", wait_s=0)
        check(rec["ok"] is False and "boom" in rec.get("error", "") and SC not in ta._LIVE,
              "a failed docker run is recorded, not raised, and not tracked", json.dumps(rec))
        fd = FakeDocker(logs="chrome start n=1 x\nchrome exit n=1 rc=0\nchrome start n=2 x\n")
        ta._docker = fd
        ta._LIVE.add(SC)
        st = ta.stop_sidecar(SC)
        check(st["chrome_starts"] == 2 and st["removed"] is True and SC not in ta._LIVE
              and ["rm", "-f", SC] in fd.calls,
              "stop records Chrome restarts, removes the container, forgets it", json.dumps(st))
    finally:
        ta._docker = real


def test_plan_prints_the_risk() -> None:
    real = ta._docker, ta._chrome_version
    ta._docker, ta._chrome_version = FakeDocker(), lambda: "155.0"
    try:
        with tempfile.TemporaryDirectory() as home:
            with open(os.path.join(home, "config.yaml"), "w", encoding="utf-8") as f:
                f.write(PROFILE)
            host = ta.plan("browser", "host", home)
            sand = ta.plan("browser", "sandbox", home)
    finally:
        ta._docker, ta._chrome_version = real
    check("RISK" in host and "11434" in host, "--browser-host host prints the loopback risk")
    check("RISK" not in sand and "browser_sidecar.py" in sand and "container:" in sand,
          "sandbox prints the sidecar command and the profile diff, no risk line")


def test_measure() -> None:
    with tempfile.TemporaryDirectory() as logs:
        d = os.path.join(logs, "t-V0-xhigh-1")
        os.makedirs(d)
        ev = [
            {"type": "tool_use", "name": "write_file", "input": {"path": "/workspace/a.js"},
             "timestamp": 1_000_000},
            {"type": "tool_use", "name": "browser_navigate",
             "input": {"url": "http://localhost:3001/"}, "timestamp": 1_120_000},
            {"type": "tool_result", "name": "browser_navigate", "output": "{}"},
            {"type": "tool_use", "name": "browser_console", "input": {}, "timestamp": 1_130_000},
            {"type": "tool_result", "name": "browser_console", "output": json.dumps(
                {"success": True, "console_messages": [{"type": "error", "text": "404 x.png"}],
                 "js_errors": [{"message": "PLAYER.hit is not a function"}]})},
            {"type": "tool_use", "name": "terminal",
             "input": {"command": "npx playwright screenshot http://localhost:3001"},
             "timestamp": 1_140_000},
        ]
        with open(os.path.join(d, "hermes.jsonl"), "w", encoding="utf-8") as f:
            f.write("\n".join(json.dumps(x) for x in ev) + "\nnot json\n")
        m = ta.measure("t-V0-xhigh-1", logs)
    check(m["opened_page_browser"] is True and m["first_open_s"] == 120,
          "a browser_navigate to :3001 counts as opening the page, 120 s in", json.dumps(m))
    check(m["browser_calls"] == {"browser_console": 1, "browser_navigate": 1},
          "browser calls are counted by name")
    check(m["opened_page_terminal"] == 1, "a Playwright command in terminal is counted too")
    check(m["errors_seen_by_model"] == ["404 x.png", "PLAYER.hit is not a function"],
          "errors the model read through browser_console are collected")


def main() -> int:
    for fn in (test_round_trip, test_default_loadout, test_lean_arm, test_verify_verdict,
               test_measure_lean, test_network_args, test_refuses_what_it_cannot_place,
               test_env_and_prereqs, test_sidecar, test_plan_prints_the_risk, test_measure):
        print(f"\n--- {fn.__name__} ---")
        n0 = len(_results)
        try:
            fn()
        except Exception:                                        # noqa: BLE001
            check(False, f"{fn.__name__} itself raised",
                  traceback.format_exc().strip().split("\n")[-1])
            traceback.print_exc()
        for ok, name, detail in _results[n0:]:
            print(("  pass  " if ok else "  FAIL  ") + name
                  + (f"   <- {detail}" if not ok and detail else ""))
    passed = sum(1 for ok, _, _ in _results if ok)
    print(f"\n{'=' * 70}\n  {passed}/{len(_results)} checks passed")
    return 0 if passed == len(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
