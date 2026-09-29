#!/usr/bin/env python
"""The pagoda runner for both harnesses (run.py --task pagoda [--harness pi]
[--print], pagoda.py's Pi helpers): Pi's model entry from the proxy's
advertised card, Pi's argv and thinking level, the harness box command, the
orchestration end to end with every process, network and docker call faked,
and --print touching nothing. No GPU, no network, no docker, no Hermes, no
Pi, no live state.

    python bench/octopus/test_pagoda_runner.py      -> "N/M checks passed"
"""
from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import tempfile
import traceback
import urllib.request
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pagoda  # noqa: E402
import run  # noqa: E402
import toolset_arms as ta  # noqa: E402

hb = ta._hb()
_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    _results.append((bool(ok), name, detail))


# The host test's models.json, shape for shape (C:\Users\jwals\octo\pi-test\cfg\full.json).
SRC = {"providers": {"yamadori": {
    "baseUrl": "http://127.0.0.1:18236/v1", "api": "openai-completions", "apiKey": "$YAMADORI_PI_KEY",
    "models": [{"id": "yamadori", "name": "Yamadori", "reasoning": True,
                "thinkingLevelMap": {"off": "minimal", "xhigh": "xhigh", "max": "max"},
                "input": ["text", "image"], "contextWindow": 132096, "maxTokens": 26419,
                "compat": {"supportsDeveloperRole": False, "sendSessionAffinityHeaders": True}}]}}}
CARD = {"id": "yamadori", "object": "model", "context_length": 163840, "max_model_len": 163840,
        "max_completion_tokens": 32768, "modalities": {"input": ["text", "image"], "output": ["text"]}}


def ns(**kw) -> argparse.Namespace:
    d = {"task": "pagoda", "harness": "pi", "arm": "xhigh", "tag": "t", "rep": 1, "key_file": None,
         "run_budget": 25200, "print": False, "iterative": 0, "tools": ta.DEFAULT_ARM,
         "browser_host": ta.DEFAULT_BROWSER_HOST, "max_turns": None}
    d.update(kw)
    return argparse.Namespace(**d)


def test_ids_and_args() -> None:
    check(pagoda.run_id("h7", "xhigh", 1) == "h7-pagoda-xhigh-1",
          "Hermes keeps the run ids it always had")
    check(pagoda.run_id("p1", "max", 2, "pi") == "p1-pagoda-pi-max-2", "a Pi run names its harness")
    text, _sha = pagoda.prompt()
    for level in ("xhigh", "max"):
        args = pagoda.pi_args(level, text)
        check(args == ["--mode", "json", "--thinking", level, "-p", text],
              f"Pi: JSON events, --thinking {level}, the prompt itself as the message")
    thinking = SRC["providers"]["yamadori"]["models"][0]["thinkingLevelMap"]
    check(thinking.get("xhigh") == "xhigh" and thinking.get("max") == "max",
          "the model entry maps xhigh and max to themselves (Pi offers them only when mapped)")


def test_pi_models() -> None:
    cfg, rec = pagoda.pi_models(SRC, CARD)
    m = cfg["providers"]["yamadori"]["models"][0]
    check(m["contextWindow"] == 163840 and m["maxTokens"] == 32768 and m["input"] == ["text", "image"],
          "window, output ceiling and inputs come from the advertised card", json.dumps(m))
    check(SRC["providers"]["yamadori"]["models"][0]["contextWindow"] == 132096,
          "the source config is not modified")
    check(rec["changed"][0]["before"]["contextWindow"] == 132096
          and rec["changed"][0]["after"]["contextWindow"] == 163840,
          "the record says what changed", json.dumps(rec))
    cfg2, _ = pagoda.pi_models(SRC, {**CARD, "modalities": {"input": ["text"]}})
    check(cfg2["providers"]["yamadori"]["models"][0]["input"] == ["text"],
          "a card without image input is followed (Pi then does not send images)")
    for bad, why in (({"providers": {}}, "no entry"),
                     ({"providers": {"a": SRC["providers"]["yamadori"],
                                     "b": SRC["providers"]["yamadori"]}}, "two entries")):
        try:
            pagoda.pi_models(bad, CARD)
            check(False, f"refuses a models.json with {why}")
        except SystemExit:
            check(True, f"refuses a models.json with {why}")


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_advertised_card() -> None:
    seen = {}
    real = urllib.request.urlopen

    def fake(req, timeout=None):
        seen["url"], seen["auth"] = req.full_url, req.get_header("Authorization")
        return _Resp(json.dumps({"object": "list", "data": [
            {"id": "yamadori-vision"}, CARD]}).encode())
    urllib.request.urlopen = fake
    try:
        card = pagoda.advertised_card(18234, "k-123")
        check(card["context_length"] == 163840 and seen["url"] == "http://127.0.0.1:18234/v1/models"
              and seen["auth"] == "Bearer k-123",
              "reads the yamadori row through the relay port, with the key", json.dumps(seen))
        urllib.request.urlopen = lambda req, timeout=None: _Resp(json.dumps(
            {"data": [{"id": "yamadori"}]}).encode())
        try:
            pagoda.advertised_card(18234, "k")
            check(False, "a row with no window is refused")
        except SystemExit as e:
            check("context_length" in str(e), "a row with no window is refused", str(e))
    finally:
        urllib.request.urlopen = real


def test_pi_plan() -> None:
    p = run.pi_plan(ns(arm="max", key_file=r"C:\k\pi.key"))
    rid = "t-pagoda-pi-max-1"
    cmd = p["box_cmd"]
    cut = cmd.index("--")
    check(p["run_id"] == rid and p["run_dir"] == os.path.join(run.RUNS_DIR, rid)
          and p["box_dir"] == os.path.join(run.LOGS_DIR, rid, "box"),
          "a fresh run folder is the project; Pi's home and record sit under the logs, not in it")
    check(cmd[2:4] == ["run", "pi"] and cmd[cmd.index("--project") + 1] == p["run_dir"]
          and cmd[cmd.index("--target-port") + 1] == str(run.RELAY_PORT)
          and cmd[cmd.index("--config") + 1] == p["models_out"]
          and cmd[cmd.index("--key-file") + 1] == r"C:\k\pi.key"
          and cmd[cmd.index("--timeout") + 1] == "25200",
          "harness_box run pi: the project, the relay's port as the one forward, the card-built "
          "models.json, the key file, the run budget", json.dumps(cmd[:cut]))
    check(cmd[cut + 1:] == pagoda.pi_args("max", p["prompt"]) and "--loadout" not in cmd,
          "Pi's args follow `--`; the default loadout (no --loadout flag)")
    h = p["harness"]
    envs = [h[i + 1] for i, x in enumerate(h) if x == "-e"]
    check(f"{hb.WORK}" in " ".join(h) and h[h.index("-w") + 1] == hb.WORK
          and "PI_CACHE_RETENTION=long" in envs and "PI_OFFLINE=1" in envs
          and h[h.index("--network") + 1] == f"container:{hb.sidecar_name(rid)}",
          "Pi runs in /work (the run folder), in the sidecar's namespace, with a stable "
          "prompt_cache_key (PI_CACHE_RETENTION=long)", json.dumps(envs))
    check("--forward 18234:host.docker.internal:18234" in p["net"],
          "the gate's one forward is the relay's port", p["net"][-300:])
    check(run.pi_plan(ns())["box_cmd"][run.pi_plan(ns())["box_cmd"].index("--key-file") + 1]
          == run.PI_KEY_FILE and run.PI_KEY_FILE.endswith("pi-dogfood.key"),
          "the default key is the pi-dogfood test key")
    check(hb.loadout_skills("pi") == ["type-check", "page-check", "package-api"]
          and hb.pi_settings() == {"defaultTools": ["read", "bash", "edit", "write", "grep", "find"]},
          "Pi's loadout: read bash edit write grep find, skills type-check, page-check and package-api")
    skill = open(os.path.join(hb.SKILLS_DIR, "page-check", "SKILL.md"), encoding="utf-8").read()
    check("errors --json" in skill and "screenshot /tmp/page.png" in skill and "read /tmp/page.png" in skill,
          "page-check: errors --json, a screenshot, and `read` to look at it")
    check("renders WebGL in software" in skill and "CDP command timed out" in skill,
          "page-check: the sandbox browser renders WebGL in software; a timeout is a slow page")
    api = os.path.join(hb.SKILLS_DIR, "package-api")
    check(os.path.exists(os.path.join(api, "api.cjs"))
          and "~/.agents/skills/package-api/api.cjs" in open(os.path.join(api, "SKILL.md"), encoding="utf-8").read(),
          "package-api: the SKILL.md runs its own script from Pi's skills folder")


def _nothing_allowed(*a, **k):
    raise AssertionError(f"--print started something: {a[:1]}")


def test_print_touches_nothing() -> None:
    real = (run.RUNS_DIR, run.LOGS_DIR, subprocess.Popen, subprocess.run, ta._docker,
            urllib.request.urlopen)
    before = os.path.getmtime(pagoda.PROMPT_PATH) if os.path.exists(pagoda.PROMPT_PATH) else None
    with tempfile.TemporaryDirectory() as d:
        run.RUNS_DIR, run.LOGS_DIR = os.path.join(d, "runs"), os.path.join(d, "logs")
        subprocess.Popen = subprocess.run = ta._docker = urllib.request.urlopen = _nothing_allowed
        try:
            outs = {}
            for harness in ("pi", "hermes"):
                for arm in ("xhigh", "max"):
                    buf = io.StringIO()
                    with redirect_stdout(buf):
                        rc = run.run_pagoda(ns(harness=harness, arm=arm, print=True))
                    outs[(harness, arm)] = (rc, buf.getvalue())
        finally:
            (run.RUNS_DIR, run.LOGS_DIR, subprocess.Popen, subprocess.run, ta._docker,
             urllib.request.urlopen) = real
        check(os.listdir(d) == [], "--print creates no run or log folder", str(os.listdir(d)))
    after = os.path.getmtime(pagoda.PROMPT_PATH) if os.path.exists(pagoda.PROMPT_PATH) else None
    check(before == after, "--print does not write the prompt file")
    _t, sha = pagoda.prompt()
    for (harness, arm), (rc, text) in outs.items():
        check(rc == 0 and sha in text and pagoda.PROMPT_PATH in text,
              f"{harness}/{arm}: prints the prompt file and its sha256")
    pi = outs[("pi", "max")][1]
    check("harness_box.py run pi" in pi and "--thinking max" in pi and "/v1/models" in pi
          and "--harness pi --arm max" in pi,
          "pi/max: the box command, the thinking level, the card read, the command to run it")
    he = outs[("hermes", "max")][1]
    check("--reasoning max" in he and "mcp_servers:" in he and "playwright-mcp" in he
          and "--provider octo-relay" in he,
          "hermes/max: the Hermes command, the lean arm's MCP line and container")


class _FakeProc:
    def __init__(self, argv, stdout=None, **k):
        self.argv, self.pid = argv, 4242
        if isinstance(argv, list) and "harness_box.py" in " ".join(argv):
            stdout.write(b'{"type":"session","version":3,"id":"sess-1","cwd":"/work"}\n'
                         b'{"type":"agent_start"}\n')
            box = argv[argv.index("--run-dir") + 1]
            os.makedirs(box, exist_ok=True)
            with open(os.path.join(box, "harness_box.jsonl"), "w", encoding="utf-8") as f:
                f.write(json.dumps({"image": hb.IMAGE, "image_id": "sha256:box", "tag": "t", "rc": 0,
                                    "browser_sidecar": {"ok": True, "browser": "Chrome/153"}}) + "\n")
            _FakeProc.box_argv = argv

    def wait(self, timeout=None):
        return 0

    def terminate(self):
        pass

    def kill(self):
        pass


def test_run_pi_end_to_end_faked() -> None:
    """run_pagoda_pi with every process, network and docker call faked: the
    order of things and what the row and meta record."""
    saved = {n: getattr(run, n) for n in ("RUNS_DIR", "LOGS_DIR", "pi_preflight", "_port_open",
                                          "snapshot", "append")}
    saved_p = {n: getattr(pagoda, n) for n in ("write_prompt", "advertised_card", "measure")}
    saved_src = hb.HARNESSES["pi"]["config_src"]
    real_popen, real_sleep = subprocess.Popen, run.time.sleep
    rows, measured = [], {}
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "full.json")
        with open(src, "w", encoding="utf-8") as f:
            json.dump(SRC, f)
        key = os.path.join(d, "pi.key")
        with open(key, "w", encoding="utf-8") as f:
            f.write("k-secret\n")
        prompt = os.path.join(d, "pagoda.md")
        text, sha = pagoda.prompt()
        with open(prompt, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        ports = iter([False, True])
        run.RUNS_DIR, run.LOGS_DIR = os.path.join(d, "runs"), os.path.join(d, "logs")
        run.pi_preflight = lambda p, k: {"ok": True}
        run._port_open = lambda port: next(ports, True)
        run.snapshot = lambda: {}
        run.append = rows.append
        run.time.sleep = lambda s: None
        pagoda.write_prompt = lambda: (prompt, sha)
        pagoda.advertised_card = lambda port, k: (measured.setdefault("card_key", k), CARD)[1]
        pagoda.measure = lambda rid, rd, ld, row: measured.update(rid=rid, rd=rd, row=row) or {"run_id": rid}
        hb.HARNESSES["pi"]["config_src"] = src
        subprocess.Popen = _FakeProc
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = run.run_pagoda(ns(key_file=key, arm="xhigh", tag="e2e"))
            rid = "e2e-pagoda-pi-xhigh-1"
            log_dir = os.path.join(run.LOGS_DIR, rid)
            written = json.load(open(os.path.join(log_dir, "pi-models.json"), encoding="utf-8"))
            meta = json.load(open(os.path.join(log_dir, "meta.json"), encoding="utf-8"))
            logs_text = "".join(open(os.path.join(dp, f), encoding="utf-8", errors="replace").read()
                                for dp, _dn, fn in os.walk(d) for f in fn if f != "pi.key")
        finally:
            for n, v in saved.items():
                setattr(run, n, v)
            for n, v in saved_p.items():
                setattr(pagoda, n, v)
            hb.HARNESSES["pi"]["config_src"] = saved_src
            subprocess.Popen, run.time.sleep = real_popen, real_sleep
    check(rc == 0 and measured.get("rid") == rid and measured.get("card_key") == "k-secret",
          "the card is read with the key, the run is measured", json.dumps({k: measured.get(k)
                                                                           for k in ("rid",)}))
    m = written["providers"]["yamadori"]["models"][0]
    check(m["contextWindow"] == 163840 and m["maxTokens"] == 32768,
          "Pi's models.json source carries the card's window and output ceiling")
    check(rows and rows[-1]["harness"] == "pi" and rows[-1]["session_id"] == "sess-1"
          and rows[-1]["effort"] == "xhigh" and rows[-1]["exit"] == 0
          and rows[-1]["prompt_sha256"] == sha
          and rows[-1]["model_card"]["card"]["context_length"] == 163840,
          "the run row: harness, Pi's session id, effort, exit, prompt sha, the card",
          json.dumps({k: rows[-1].get(k) for k in ("harness", "session_id", "effort", "exit")}) if rows else "")
    check("k-secret" not in logs_text and "k-secret" not in json.dumps(rows),
          "the key is in no file the runner wrote and in no row")
    check(text not in json.dumps(meta["box_cmd"]) and any("sha256" in x for x in meta["box_cmd"]),
          "the recorded command names the prompt by its sha, not its text")
    check(_FakeProc.box_argv[-1] == text and "--thinking" in _FakeProc.box_argv,
          "the box got the prompt byte for byte as the last argument")


def test_relay_summary_efforts() -> None:
    with tempfile.TemporaryDirectory() as d:
        rows = [{"method": "GET", "path": "/v1/models", "status": 200},
                {"method": "POST", "path": "/v1/chat/completions", "status": 200,
                 "request": {"reasoning_effort": "max"}, "response": {"tool_calls": ["bash"]}},
                {"method": "POST", "path": "/v1/chat/completions", "status": 200,
                 "request": {"reasoning_effort": "max"}, "response": {}}]
        with open(os.path.join(d, "relay.jsonl"), "w", encoding="utf-8") as f:
            f.write("\n".join(json.dumps(r) for r in rows) + "\n")
        s = pagoda.relay_summary(d)
    check(s["requests"] == 2 and s["efforts_sent"] == {"max": 2} and s["client_calls"] == {"bash": 1},
          "the relay summary counts the efforts the harness sent (a clamp would show), chat wire too",
          json.dumps(s["efforts_sent"]))


def test_refusals() -> None:
    buf = io.StringIO()
    real = sys.stderr
    sys.stderr = buf
    try:
        rc = run.run_pagoda(ns(iterative=3))
    finally:
        sys.stderr = real
    check(rc == 2 and "single-shot" in buf.getvalue(), "--iterative is refused for Pi too")
    old = sys.argv
    sys.argv = ["run.py", "--variant", "V0", "--arm", "xhigh", "--harness", "pi"]
    try:
        with redirect_stdout(io.StringIO()):
            try:
                import contextlib
                with contextlib.redirect_stderr(io.StringIO()):
                    run.main()
                check(False, "--harness pi without --task pagoda is refused")
            except SystemExit as e:
                check(e.code == 2, "--harness pi without --task pagoda is refused")
    finally:
        sys.argv = old


def main() -> int:
    for fn in (test_ids_and_args, test_pi_models, test_advertised_card, test_pi_plan,
               test_print_touches_nothing, test_run_pi_end_to_end_faked, test_relay_summary_efforts,
               test_refusals):
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
