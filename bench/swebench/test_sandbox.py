#!/usr/bin/env python
"""The SWE-bench model-command container on the sandbox network
(SELF-IMPROVEMENT-LOG #49), offline: the overlay's `environment` keys, which
network dockerfix's LowLevelDockerEnvironment joins (never the default
bridge), and wsl_side.cmd_agent's order -- the network up before mini, down
after it even when mini raises, no agent run when the gate does not come up.
Docker, mini and the proxy are faked. The live check is
`<swebench venv>\\Scripts\\python.exe bench/swebench/dockerfix.py netcheck`
(Docker, no model).

    python bench/swebench/test_sandbox.py      -> "N/M checks passed"
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dockerfix  # noqa: E402
import wsl_side  # noqa: E402
import sandbox_net  # noqa: E402  (wsl_side put bench/sandbox on the path)

CHECKS: list[tuple[str, bool]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok)))
    print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"   <- {detail}" if not ok and detail else ""))


def test_network_of() -> None:
    cases = {(): "none", ("--rm",): "none", ("--rm", "--network", "swe-net-x"): "swe-net-x",
             ("--network=swe-net-y",): "swe-net-y", ("--net", "swe-net-z"): "swe-net-z",
             ("--network", "bridge"): "none", ("--network", "host"): "none",
             ("--network", "a", "--network", "b"): "b"}
    bad = {k: dockerfix.network_of(list(k)) for k, v in cases.items()
           if dockerfix.network_of(list(k)) != v}
    check("network_of: the run_args network, else none -- bridge and host are refused", not bad,
          json.dumps({" ".join(k): v for k, v in bad.items()}))
    src = open(os.path.join(HERE, "dockerfix.py"), encoding="utf-8").read()
    check("LowLevelDockerEnvironment creates the container with that network",
          "network_mode=network_of(self.config.run_args)" in src)


def test_overlay() -> None:
    with tempfile.TemporaryDirectory() as d:
        p = wsl_side.write_overlay(d, "bonsai", "http://127.0.0.1:1234/v1", "swe-net-i-1")
        ov = json.load(open(p, encoding="utf-8"))
        p0 = wsl_side.write_overlay(d, "bonsai", "http://127.0.0.1:1234/v1")
        ov0 = json.load(open(p0, encoding="utf-8"))
    env = ov.get("environment") or {}
    check("overlay: the container joins the instance network (--rm kept)",
          env.get("run_args") == ["--rm", "--network", "swe-net-i-1"], json.dumps(env))
    pe = sandbox_net.proxy_env()
    check("overlay: the gate as proxy, both cases, loopback bypassed",
          env.get("env") == pe and pe["https_proxy"] == "http://egress:3128"
          and "PAGER" not in env.get("env", {}), json.dumps(env.get("env")))
    check("overlay: no network given, no environment keys", "environment" not in ov0)
    check("every overlay key is a recorded deviation",
          {"environment.run_args", "environment.env"} <= set(wsl_side.OVERLAY_DEVIATIONS))
    check("the merged env keeps the leaderboard's own variables (mini's recursive_merge)",
          _merge({"env": {"PAGER": "cat"}, "timeout": 60}, env)["env"].get("PAGER") == "cat"
          and _merge({"env": {"PAGER": "cat"}}, env)["env"]["HTTP_PROXY"] == pe["HTTP_PROXY"])


def _merge(*ds):
    out: dict = {}
    for d in ds:
        for k, v in d.items():
            out[k] = _merge(out[k], v) if isinstance(out.get(k), dict) and isinstance(v, dict) else v
    return out


def _agent(up_ok: bool, mini_raises: bool = False) -> tuple[int | None, list[str], dict]:
    """Run cmd_agent with docker, mini and the proxy faked. Returns (rc,
    event order, the overlay mini was given)."""
    events: list[str] = []
    seen: dict = {}
    real = (wsl_side._read_key, wsl_side.resolve_api_base, wsl_side._get,
            wsl_side.default_config, wsl_side.versions, wsl_side.subprocess.run,
            sandbox_net.up, sandbox_net.down, wsl_side.merge)

    def fake_up(tag, forwards=(), wait_s=30, prefix="octo"):
        events.append(f"up:{prefix}")
        n = sandbox_net.names(tag, prefix)
        return {**n, "ok": up_ok, "ready_s": 0.1, **({} if up_ok else {"error": "gate run: boom"})}

    def fake_down(tag, prefix="octo"):
        events.append(f"down:{prefix}")
        return {"allowed": 3, "denied": 1, "denied_sample": ["DENY x"], "leftovers_removed": 1,
                "network_removed": True}

    def fake_run(cmd, **kw):
        if cmd[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(cmd, 0, b"", b"")
        events.append("mini")
        ov = cmd[cmd.index("-c", cmd.index("-c") + 1) + 1]
        seen.update(json.load(open(ov, encoding="utf-8")))
        seen["key_in_argv"] = any("sk-test" in str(c) for c in cmd)
        if mini_raises:
            raise KeyboardInterrupt
        return subprocess.CompletedProcess(cmd, 0)

    with tempfile.TemporaryDirectory() as d:
        cfg = os.path.join(d, "swebench.yaml")
        open(cfg, "w").write("x: 1\n")
        wsl_side._read_key = lambda p: "sk-test"
        wsl_side.resolve_api_base = lambda e: "http://127.0.0.1:1234/v1"
        wsl_side._get = lambda url, key=None, timeout=5: (200, b'{"data":[{"id":"yamadori"}]}')
        wsl_side.default_config = lambda: cfg
        wsl_side.versions = lambda: {}
        wsl_side.subprocess.run = fake_run
        wsl_side.merge = lambda out, work, iid: None
        sandbox_net.up, sandbox_net.down = fake_up, fake_down
        a = types.SimpleNamespace(arm="bonsai", instance="django__django-11999",
                                  out=os.path.join(d, "bonsai"), key_file="k", subset="verified",
                                  api_base=None, redo=False)
        rc = None
        try:
            rc = wsl_side.cmd_agent(a)
        except KeyboardInterrupt:
            events.append("raised")
        finally:
            (wsl_side._read_key, wsl_side.resolve_api_base, wsl_side._get,
             wsl_side.default_config, wsl_side.versions, wsl_side.subprocess.run,
             sandbox_net.up, sandbox_net.down, wsl_side.merge) = real
        tp = os.path.join(d, "bonsai", "timings.jsonl")
        seen["timing"] = json.loads(open(tp).read().splitlines()[-1]) if os.path.exists(tp) else {}
    return rc, events, seen


def test_agent_order() -> None:
    rc, ev, seen = _agent(True)
    check("cmd_agent: network up, then mini, then down",
          ev == ["up:swe", "mini", "down:swe"], json.dumps(ev))
    env = seen.get("environment") or {}
    check("mini gets the instance's network in its overlay",
          env.get("run_args", [None, None, ""])[2].startswith("swe-net-django__django-11999-"),
          json.dumps(env))
    check("the key is never an argument", seen.get("key_in_argv") is False)
    t = seen.get("timing") or {}
    check("the timing row records the gate's counts",
          (t.get("sandbox_net") or {}).get("denied") == 1 and t.get("rc") == 0, json.dumps(t))
    rc, ev, _ = _agent(True, mini_raises=True)
    check("the network comes down even when mini raises", ev == ["up:swe", "mini", "down:swe", "raised"],
          json.dumps(ev))
    rc, ev, seen = _agent(False)
    t = seen.get("timing") or {}
    check("no gate: mini never runs, rc 97, `not_run: sandbox_net` recorded (fail closed)",
          rc == 97 and "mini" not in ev and t.get("not_run") == "sandbox_net", json.dumps([ev, t]))


def main() -> int:
    for fn in (test_network_of, test_overlay, test_agent_order):
        print(f"\n--- {fn.__name__} ---")
        try:
            fn()
        except Exception as e:                                   # noqa: BLE001
            import traceback
            traceback.print_exc()
            check(f"{fn.__name__} itself raised: {e}", False)
    passed = sum(ok for _, ok in CHECKS)
    print(f"\n{passed}/{len(CHECKS)} checks passed")
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
