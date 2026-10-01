#!/usr/bin/env python
"""bench/deploy_layout_v2.py's edits, offline: pure functions of the files' text, applied in memory to the
config the deploy was written for (bench/fixtures/config.layout-v1.yaml, below) and to the CURRENT
start-stack.bat, watchdog.ps1 and served fixture. Nothing is written, nothing is restarted, no port is
reached. Also bench/kv_rank.py's v2 arm (target_args) and engine_corruption's flag removal, which the fit
runs through.

THE CONFIG IS A FIXTURE (2026-10-01): layout v2's deploy ran on 2026-09-29, and the live config.yaml has been
layout v3 since 2026-10-01 (bench/deploy_layout_v3.py: bonsai -np 1, operator 2026-09-30), which edit_config
rightly refuses ("-np 3 missing"). bench/deploy_layout_v3.py still imports this module's edits, so they stay
tested -- against the last layout-v1 config, the input they were written for (config.yaml.bak-20260929-131409,
machine paths replaced).

    python bench/test_deploy_layout_v2.py      -> "N/M checks passed"
"""
from __future__ import annotations

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "mcp"))

import deploy_layout_v2 as d  # noqa: E402

_results: list[tuple[bool, str, str]] = []
N = 166400          # an example line (a multiple of 256); the real one is kv_rank.py's


def check(ok: bool, name: str, detail: str = "") -> None:
    _results.append((bool(ok), name, detail))


def _read(rel: str) -> str:
    return open(os.path.join(ROOT, rel), encoding="utf-8", newline="").read()


def _bonsai(c: str) -> str:
    i0, i1 = d._bonsai_block(c.replace("\r\n", "\n"))
    return c.replace("\r\n", "\n")[i0:i1]


def _argv(block: str) -> list[str]:
    lines = [ln.strip() for ln in block.split("cmd: |", 1)[1].splitlines()]
    return " ".join(ln for ln in lines if ln and not ln.startswith("#")).split()


CONFIG_FIXTURE = "bench/fixtures/config.layout-v1.yaml"


def test_config() -> None:
    import yaml
    old = _read(CONFIG_FIXTURE)
    new = d.edit_config(old, N)
    b_old, b_new = _bonsai(old), _bonsai(new)
    a_old, a_new = _argv(b_old), _argv(b_new)
    check("--mmproj" not in a_new, "bonsai: --mmproj removed", str([x for x in a_new if "mmproj" in x]))
    check(a_new[a_new.index("-c") + 1] == str(2 * N + d.LANE)
          and a_new[a_new.index("--kv-vram-cells") + 1] == str(N),
          "bonsai: -c 2N + the lane and --kv-vram-cells N",
          f"{a_new[a_new.index('-c') + 1]} {a_new[a_new.index('--kv-vram-cells') + 1]}")

    def rest(av):
        out, skip = [], False
        for x in av:
            if skip:
                skip = False
                continue
            if x in ("-c", "--kv-vram-cells", "--mmproj"):
                skip = True
                continue
            out.append(x)
        return out
    check(rest(a_old) == rest(a_new), "bonsai: nothing else in its argv changes")
    cfg_old, cfg_new = yaml.safe_load(old), yaml.safe_load(new)
    check(cfg_old["models"]["bonsai"].get("env") == cfg_new["models"]["bonsai"].get("env"),
          "bonsai: its env is unchanged")
    check(d.edit_config(new, N) == new, "edit_config is idempotent")
    v = cfg_new["models"].get("bonsai-vision") or {}
    check(v and any(d.A4000_UUID in e for e in v.get("env") or []) and v.get("ttl") == 300,
          "bonsai-vision: on the A4000 by UUID, ttl 300", json.dumps(v.get("env")))
    cmd = str(v.get("cmd") or "")
    check(f"--mmproj ${{models}}/{d.MMPROJ}" in cmd and f"-m ${{models}}/{d.VISION_TRUNK}" in cmd
          and "-c 16384" in cmd and "${server}" in cmd,
          "bonsai-vision: the retired entry's argv on the original trunk, with the projector", cmd[:300])
    groups = cfg_new["routing"]["router"]["settings"]["groups"]
    og = groups.get("ondemand") or {}
    check(og.get("members") == ["bonsai-vision"] and og.get("exclusive") is True and og.get("swap") is True,
          "the ondemand group is back (swap, exclusive)", json.dumps(og))
    check(groups.get("primary", {}).get("persistent") is True, "the primary group is still persistent")
    others = {k: v for k, v in cfg_new["models"].items() if k != "bonsai"}
    same = {k: v for k, v in cfg_old["models"].items() if k != "bonsai"}
    same["bonsai-vision"] = others.get("bonsai-vision")
    check(others == same, "no other model entry changes")
    rec = d.check_vision_entry(new)
    check(all(v for k, v in rec.items() if k != "main_has_mmproj") and not rec["main_has_mmproj"],
          "check_vision_entry passes on the edited config (gpu_room has the row)", json.dumps(rec))
    check("VISION variant, ON DEMAND (ttl 900)" not in new, "the stale vision banner is replaced")
    notes = re.findall(r"# " + re.escape(d.MARK), b_new)
    check(len(notes) == 2 and "kv layout (bench/deploy_kv_rank.py)): 3 slots" not in b_new,
          "bonsai: the v1 layout note is replaced by the v2 note (and the --mmproj note)", str(len(notes)))


def test_env_files() -> None:
    envs = d.envs_for(N)
    check(envs == {"YAMADORI_MAIN_CAP": str(N - d.LANE), "YAMADORI_LANE_TOKENS": str(d.LANE)},
          "the proxy's values: the main cap is the line less the lane", json.dumps(envs))
    bat = d.edit_bat(_read("scripts/start-stack.bat"), envs)
    b = bat.replace("\r\n", "\n")
    check("YAMADORI_CHILD_TOKENS" not in b and "REM kv layout (bench/deploy_kv_rank.py)" not in b,
          "start-stack.bat: the v1 kv-layout block is gone")
    check(b.count(f"REM {d.MARK}\n") == 1 and f'set "YAMADORI_MAIN_CAP={N - d.LANE}"' in b
          and f'set "YAMADORI_LANE_TOKENS={d.LANE}"' in b,
          "start-stack.bat: one v2 block with the cap and the lane")
    check(d.edit_bat(bat, envs) == bat, "edit_bat is idempotent")
    check(b.index(f"REM {d.MARK}") < b.index('start "" /B "%PY%" "%CD%\\mcp\\tools_api.py"'),
          "start-stack.bat: set before the first Python service starts")
    ps = d.edit_watchdog(_read("scripts/watchdog.ps1"), envs)
    p = ps.replace("\r\n", "\n")
    check(p.count(f"# {d.MARK}") == 3 and "YAMADORI_CHILD_TOKENS" not in p
          and "# kv layout (bench/deploy_kv_rank.py)" not in p,
          "watchdog.ps1: Env2 replaced for the proxy, the tools API and the worker", str(p.count(d.MARK)))
    check(d.edit_watchdog(ps, envs) == ps, "edit_watchdog is idempotent")


def test_fixture_and_props() -> None:
    want = d.intended_props(N)
    check(want == {"n_ctx": min(2 * N + d.LANE, 262144), "total_slots": 3, "kv_vram_cells": N,
                   "modalities": {"vision": False}},
          "intended /props: the slot's n_ctx is capped at the trained 262,144; no vision on bonsai",
          json.dumps(want))
    fx = json.loads(d.edit_fixture(_read("mcp/fixtures/bonsai_props.json"), want))
    check(fx["modalities"] == {"vision": False} and fx["kv_vram_cells"] == N
          and "modalities.vision: false since layout v2" in fx["_what"],
          "the re-recorded fixture says what is served", json.dumps({k: fx[k] for k in want}))


def test_refusals() -> None:
    check(d.main(["--line", "1000"]) == 2, "a line that is not a multiple of 256 is refused")
    check(d.main([]) == 2, "no --line: refused")


def test_the_fit_arm() -> None:
    import engine_corruption as ec
    import kv_rank
    t = kv_rank.target_args(N, layout="v2")
    check(t["--mmproj"] is None and t["-c"] == str(2 * N + kv_rank.LANE) and t["--kv-vram-cells"] == str(N),
          "kv_rank v2 arm: --mmproj dropped, -c 2N + the lane", json.dumps(t))
    t1 = kv_rank.target_args(N, layout="v1")
    check("--mmproj" not in t1 and t1["-c"] == str(2 * N + kv_rank.CHILD), "kv_rank v1 arm unchanged")
    check(kv_rank.main_cap(N, "v2") == N - kv_rank.LANE and kv_rank.main_cap(N, "v1") == N,
          "the cap on the arm: the line less the lane (v2), the line (v1)")
    argv = ["exe", "--port", "${PORT}", "-m", "x.gguf", "--mmproj", "C:/m/mmproj.gguf", "--kv-unified", "-c", "1"]
    out = ec.arm_command(argv, "new.exe", 9, {"--mmproj": None, "--kv-unified": None, "-c": "5"})
    check(out == ["new.exe", "--port", "9", "-m", "x.gguf", "-c", "5"],
          "engine_corruption.arm_command: None removes a flag and its value (and a bare flag)", str(out))
    name, binary, env, args = ec.parse_arm("a=bin.exe,DROP:--mmproj,ARG:-c=7,X=1")
    check(args == {"--mmproj": None, "-c": "7"} and env == {"X": "1"}, "parse_arm reads DROP:", str(args))


def main() -> int:
    for fn in (test_config, test_env_files, test_fixture_and_props, test_refusals, test_the_fit_arm):
        try:
            fn()
        except Exception as e:                                   # noqa: BLE001
            import traceback
            traceback.print_exc()
            check(False, f"{fn.__name__} raised", f"{type(e).__name__}: {e}")
    for ok, name, detail in _results:
        print(("  pass  " if ok else "  FAIL  ") + name + (f"   <- {detail}" if not ok and detail else ""))
    passed = sum(1 for ok, _, _ in _results if ok)
    print(f"\n{'=' * 70}\n  {passed}/{len(_results)} checks passed")
    return 0 if passed == len(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
