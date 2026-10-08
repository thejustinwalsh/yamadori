#!/usr/bin/env python
"""Is this deploy good? Run the live suite through :1234 right after a restart.

    python scripts/deploy_check.py --key-file PATH
    python scripts/deploy_check.py --key-file PATH --maintenance   # + intrusive
    python scripts/deploy_check.py --key-file PATH --wait 900      # longer boot

A deploy is NOT good until this exits 0 (AGENTS.md, "Before you claim
anything works"). The operator's rule: if we don't have tests that exercise
the real models, we don't have tests.

WHAT IT DOES, IN ORDER

  0. Checks every engine binary config.yaml points at (and llama-swap
     itself) against engines/manifest.yaml: the path must be an engine's
     shipped binary, and it and every DLL it loads from its own directory
     must match the recorded size and SHA-256 (scripts/build_engine.py
     --verify-only; docs/ENGINES.md). A binary we cannot rebuild exactly is
     not deployed: this fails NOT GOOD (exit 1) before anything else runs.
  1. Waits for every service to answer /health: llama-swap :11434, the proxy
     :1234, the tools API :1235 -- and for llama-swap to report the chat
     model `ready`. A stack that is still loading is not failing. (Laya on
     :1237 is retired, 2026-09-24, docs/E1.md: E1's heads replace it, and
     the suites that tested its live service run offline only --
     scripts/run_tests.py RETIRED_LIVE.)
  2. Refuses to start while another live run is on the card (a python
     process running a live suite). Two consumers on one GPU degrade each
     other into 429s and 502s, and the loser looks like the one with the bug.
  3. Waits until the chat model's /slots are idle on two reads 5 s apart.
  4. Runs `scripts/run_tests.py --live --live-only`, which streams every
     check's pass/FAIL with its evidence, treats a 429 as NOT RUN, and exits
     1 on any failure, 3 when nothing failed but something did not run.
  5. Appends one line to logs/deploy_check.jsonl: when, the git HEAD, the
     exit code and the verdict. The key is never printed or recorded.

Exit code: run_tests.py's (0 GOOD, 1 NOT GOOD, 3 INCOMPLETE), or 2 when the
stack never came up or the card was never free (the suite did not start).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    ".."))
# The stack interpreter: YAMADORI_PYTHON (stack.env / the environment), else the repo's .venv, else the
# textgen env under the home directory (the original install), else the one running this script.
PY = (os.environ.get("YAMADORI_PYTHON")
      or next((c for c in (os.path.join(ROOT, ".venv", "Scripts", "python.exe"),
                           os.path.expanduser("~/textgen/installer_files/env/python.exe"))
               if os.path.exists(c)), ""))
if not os.path.exists(PY):
    PY = sys.executable

HEALTH = {"llama-swap": "http://127.0.0.1:11434/health",
          "proxy": "http://127.0.0.1:1234/health",
          "tools-api": "http://127.0.0.1:1235/health"}
SWAP = "http://127.0.0.1:11434"
CHAT_MODEL = os.environ.get("YAMADORI_CHAT_MODEL", "bonsai")
LIVE_MARKERS = ("test_live_stack.py", "test_tools_live.py", "run_tests.py --live",
                "deploy_check.py")
LOG = os.path.join(ROOT, "logs", "deploy_check.jsonl")
CONFIG = os.path.join(ROOT, "config.yaml")
MANIFEST = os.path.join(ROOT, "engines", "manifest.yaml")


def engines_check(config: str = CONFIG, manifest: str = MANIFEST,
                  extra: list[tuple[str, str]] | None = None
                  ) -> tuple[bool, list[str]]:
    """(every engine binary matches the manifest, lines to print). `extra`
    replaces the default llama-swap entry (tests pass their own)."""
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    try:
        import build_engine
        problems, ok = build_engine.verify_deploy(config, manifest, extra)
    except Exception as e:                                       # noqa: BLE001
        problems, ok = [f"could not verify the engine binaries: "
                        f"{type(e).__name__}: {e}"], []
    lines = [f"ok    {x}" for x in ok] + [f"FAIL  {x}" for x in problems]
    return not problems, lines


def _get(url: str, timeout: int = 10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status, r.read().decode("utf-8", "replace")


def services_up() -> dict[str, str]:
    """{service: problem} for every service not yet answering."""
    bad = {}
    for name, url in HEALTH.items():
        try:
            code, _ = _get(url)
            if not 200 <= code < 300:
                bad[name] = f"HTTP {code}"
        except Exception as e:                                   # noqa: BLE001
            bad[name] = f"{type(e).__name__}: {e}"[:120]
    if "llama-swap" not in bad:
        try:
            _, body = _get(f"{SWAP}/running")
            run = {m.get("model"): m.get("state")
                   for m in json.loads(body).get("running") or []}
            if run.get(CHAT_MODEL) != "ready":
                bad["chat model"] = f"{CHAT_MODEL} is {run.get(CHAT_MODEL)!r}"
        except Exception as e:                                   # noqa: BLE001
            bad["chat model"] = f"/running: {type(e).__name__}: {e}"[:120]
    return bad


def other_tier_models() -> set[str]:
    """Every main model of the tier -> model table other than the chat model (mcp/tier_models.py: mirai-s at xhigh,
    flash-next at max), read the way the proxy reads it (YAMADORI_TIER_MODELS, else YAMADORI_MAX_MODEL). With no
    table in this environment, the config's flash-next (the 2026-09-28 check) and mirai-s."""
    try:
        sys.path.insert(0, os.path.join(ROOT, "mcp"))
        import tier_models
        t = tier_models.load()
        if t.enabled:
            return {m for m in t.main_models() if m != CHAT_MODEL}
    except Exception:                                            # noqa: BLE001
        pass
    return {os.environ.get("YAMADORI_MAX_MODEL", "flash-next"), "mirai-s"}


def max_model_holds_card() -> bool:
    """llama-swap reports another tier's model (other_tier_models: mirai-s, flash-next) loaded and the chat model
    not. /running never loads anything."""
    try:
        _, body = _get(f"{SWAP}/running")
        run = {m.get("model") for m in json.loads(body).get("running") or []}
    except Exception:                                            # noqa: BLE001
        return False
    return bool(other_tier_models() & run) and CHAT_MODEL not in run


def ask_chat_model_back(key_file: str) -> str:
    """One medium request through the proxy (the door users use): max mode swaps the chat model back for it."""
    import urllib.request
    try:
        key = open(key_file, encoding="utf-8").read().strip()
        req = urllib.request.Request(
            "http://127.0.0.1:1234/v1/chat/completions",
            data=json.dumps({"model": "yamadori", "reasoning_effort": "medium", "max_tokens": 64,
                             "messages": [{"role": "user", "content": "Reply with exactly: ok"}]}).encode(),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return f"HTTP {r.status}"
    except Exception as e:                                       # noqa: BLE001
        return f"{type(e).__name__}: {e}"[:200]


def other_live_runs() -> list[str]:
    """Command lines of python processes running a live suite, other than
    this process and its parents."""
    ps = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
          "ForEach-Object { \"$($_.ProcessId)`t$($_.CommandLine)\" }")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                             capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    me = {os.getpid(), os.getppid()}
    hits = []
    for line in out.splitlines():
        pid, _, cmd = line.partition("\t")
        if not pid.strip().isdigit() or int(pid) in me:
            continue
        if any(m in cmd for m in LIVE_MARKERS):
            hits.append(f"pid {pid.strip()}: {cmd.strip()[:160]}")
    return hits


def slots_busy() -> list | None:
    """The ids of processing slots, or None when /slots cannot be read.
    Never loads the chat model: asked only while llama-swap reports it
    running (under max mode the max model may hold the card, and an
    /upstream/<model>/ request would swap it out; mcp/max_mode.py)."""
    try:
        _, run = _get(f"{SWAP}/running")
        if CHAT_MODEL not in {m.get("model") for m in json.loads(run).get("running") or []}:
            return None
        _, body = _get(f"{SWAP}/upstream/{CHAT_MODEL}/slots")
        return [s.get("id") for s in json.loads(body) if s.get("is_processing")]
    except Exception:                                            # noqa: BLE001
        return None


def head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True,
                              timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def record(rc: int, verdict: str, secs: float, note: str = "") -> None:
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"at": time.time(), "head": head(), "rc": rc,
                            "verdict": verdict, "seconds": round(secs),
                            "note": note}) + "\n")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--key-file", required=True,
                    help="file holding the proxy API key (never printed)")
    ap.add_argument("--wait", type=int, default=600,
                    help="seconds to wait for the stack and for an idle card")
    ap.add_argument("--maintenance", action="store_true",
                    help="also run the intrusive live tests (restart the proxy)")
    args = ap.parse_args(argv)
    t0 = time.time()
    deadline = t0 + args.wait

    print("  0. engine binaries match engines/manifest.yaml", flush=True)
    good, lines = engines_check()
    for line in lines:
        print("     " + line, flush=True)
    if not good:
        print("  NOT GOOD: config.yaml runs a binary engines/manifest.yaml does "
              "not pin (above). Rebuild it with scripts/build_engine.py and "
              "record it, or restore the recorded binary; docs/ENGINES.md.")
        record(1, "NOT GOOD", time.time() - t0,
               "engine binaries do not match engines/manifest.yaml")
        return 1

    # The same rigor for what the engines load (operator, 2026-09-25):
    # every model file, derived artifact and runtime lock against
    # models/manifest.yaml (scripts/verify_artifacts.py; docs/MODELS.md).
    # An error fails the deploy; a warning is printed and does not.
    print("  0b. model files and runtimes match models/manifest.yaml",
          flush=True)
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import verify_artifacts
    probs = verify_artifacts.verify(CONFIG)
    for p in probs:
        print("     " + str(p), flush=True)
    errs = [p for p in probs if p.severity == "error"]
    if not probs:
        print("     every artifact matches", flush=True)
    if errs:
        print(f"  NOT GOOD: {len(errs)} artifact(s) do not match "
              "models/manifest.yaml (above); docs/MODELS.md says how to "
              "re-obtain or rebuild each.")
        record(1, "NOT GOOD", time.time() - t0,
               f"{len(errs)} artifacts do not match models/manifest.yaml")
        return 1

    print("  1. waiting for every service to answer /health", flush=True)
    asked_back = False
    while True:
        bad = services_up()
        if not bad:
            break
        if not asked_back and set(bad) == {"chat model"} and max_model_holds_card():
            # ANOTHER TIER'S MODEL holds the card (mcp/max_mode.py: mirai-s at xhigh, flash-next at max): the chat
            # model returns on the next request of its own tiers and nothing else brings it back (2026-09-28: this
            # waited 600 s for bonsai while flash-next served). One tiny medium request through the proxy, the
            # ordinary path back.
            asked_back = True
            print(f"     another tier's model holds the card; one medium request through :1234 brings {CHAT_MODEL} "
                  "back: " + ask_chat_model_back(args.key_file), flush=True)
            continue
        if time.time() > deadline:
            print(f"  NOT STARTED: services not up after {args.wait}s: "
                  + json.dumps(bad))
            record(2, "NOT STARTED", time.time() - t0, "services down")
            return 2
        time.sleep(10)
    print("     all up", flush=True)

    print("  2. checking no other live run is on the card", flush=True)
    others = other_live_runs()
    if others:
        print("  NOT STARTED: another live run is using the card:\n    "
              + "\n    ".join(others))
        record(2, "NOT STARTED", time.time() - t0, "another live run")
        return 2

    print("  3. waiting for idle /slots on two reads 5 s apart", flush=True)
    while True:
        a = slots_busy()
        time.sleep(5)
        b = slots_busy()
        if a == [] and b == []:
            break
        if time.time() > deadline:
            print(f"  NOT STARTED: /slots never idle (busy {a} then {b})")
            record(2, "NOT STARTED", time.time() - t0, "slots busy")
            return 2
        time.sleep(10)
    print("     idle", flush=True)

    # The skill document index (mcp/skill_match.py UPKEEP): rebuilt here, on
    # the idle card after the restart, when it does not match the armed
    # set; a deploy whose index is still stale is NOT GOOD.
    print("  3b. the skill document index matches the armed set", flush=True)
    sys.path.insert(0, os.path.join(ROOT, "mcp"))
    import skill_match
    st = skill_match.index_state()
    if not st["fresh"]:
        print(f"     {st['why']}: rebuilding", flush=True)
        try:
            print("     " + json.dumps(skill_match.handle_build({})),
                  flush=True)
        except Exception as e:                                   # noqa: BLE001
            print(f"     rebuild failed: {type(e).__name__}: {e}", flush=True)
        st = skill_match.index_state()
    if not st["fresh"]:
        print(f"  NOT GOOD: the skill document index is stale ({st['why']});"
              " python mcp/skill_match.py --build on an idle A4000.")
        record(1, "NOT GOOD", time.time() - t0, "skill index stale")
        return 1
    print(f"     fresh: {st['documents']} documents for {st['skills']} "
          "skills", flush=True)

    print("  4. the live suite, through :1234", flush=True)
    cmd = [PY, "-X", "utf8", os.path.join(ROOT, "scripts", "run_tests.py"),
           "--live", "--live-only", "--key-file", args.key_file]
    if args.maintenance:
        cmd.append("--maintenance")
    rc = subprocess.run(cmd, cwd=ROOT).returncode
    verdict = {0: "GOOD", 3: "INCOMPLETE"}.get(rc, "NOT GOOD")
    record(rc, verdict, time.time() - t0)
    print(f"\n  DEPLOY {verdict} (exit {rc}); recorded in "
          f"{os.path.relpath(LOG, ROOT)}")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
